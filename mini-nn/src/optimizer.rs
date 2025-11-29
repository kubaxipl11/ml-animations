//! Optimizers for training neural networks
//!
//! Optimizers update network parameters based on computed gradients.
//! Each optimizer has different strategies for learning rate adaptation.

use crate::tensor::Tensor;
use crate::layer::Dense;
use ndarray::Array1;

/// Available optimizers
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD {
        learning_rate: f64,
    },
    /// SGD with Momentum
    SGDMomentum {
        learning_rate: f64,
        momentum: f64,
    },
    /// Adam optimizer (Adaptive Moment Estimation)
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// RMSprop optimizer
    RMSprop {
        learning_rate: f64,
        decay: f64,
        epsilon: f64,
    },
}

impl Optimizer {
    /// Create SGD optimizer
    pub fn sgd(learning_rate: f64) -> Self {
        Optimizer::SGD { learning_rate }
    }

    /// Create SGD with momentum
    pub fn sgd_momentum(learning_rate: f64, momentum: f64) -> Self {
        Optimizer::SGDMomentum { learning_rate, momentum }
    }

    /// Create Adam optimizer with default hyperparameters
    pub fn adam(learning_rate: f64) -> Self {
        Optimizer::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Create RMSprop optimizer
    pub fn rmsprop(learning_rate: f64) -> Self {
        Optimizer::RMSprop {
            learning_rate,
            decay: 0.9,
            epsilon: 1e-8,
        }
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f64 {
        match self {
            Optimizer::SGD { learning_rate } => *learning_rate,
            Optimizer::SGDMomentum { learning_rate, .. } => *learning_rate,
            Optimizer::Adam { learning_rate, .. } => *learning_rate,
            Optimizer::RMSprop { learning_rate, .. } => *learning_rate,
        }
    }
}

/// State for momentum-based optimizers
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// First moment (mean of gradients)
    pub m_weights: Vec<Tensor>,
    pub m_biases: Vec<Array1<f64>>,
    /// Second moment (variance of gradients) for Adam
    pub v_weights: Vec<Tensor>,
    pub v_biases: Vec<Array1<f64>>,
    /// Training step counter
    pub step: usize,
}

impl OptimizerState {
    /// Create new optimizer state for given layer shapes
    pub fn new(layers: &[Dense]) -> Self {
        let mut m_weights = Vec::new();
        let mut m_biases = Vec::new();
        let mut v_weights = Vec::new();
        let mut v_biases = Vec::new();

        for layer in layers {
            let (rows, cols) = layer.weights.shape();
            m_weights.push(Tensor::zeros(rows, cols));
            v_weights.push(Tensor::zeros(rows, cols));
            m_biases.push(Array1::zeros(layer.out_features()));
            v_biases.push(Array1::zeros(layer.out_features()));
        }

        Self {
            m_weights,
            m_biases,
            v_weights,
            v_biases,
            step: 0,
        }
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        for m in &mut self.m_weights {
            m.data.fill(0.0);
        }
        for v in &mut self.v_weights {
            v.data.fill(0.0);
        }
        for m in &mut self.m_biases {
            m.fill(0.0);
        }
        for v in &mut self.v_biases {
            v.fill(0.0);
        }
        self.step = 0;
    }
}

/// Update a single dense layer's parameters
pub fn update_dense_layer(
    optimizer: &Optimizer,
    layer: &mut Dense,
    state: &mut OptimizerState,
    layer_idx: usize,
) {
    match optimizer {
        Optimizer::SGD { learning_rate } => {
            sgd_update(layer, *learning_rate);
        }
        Optimizer::SGDMomentum { learning_rate, momentum } => {
            sgd_momentum_update(layer, *learning_rate, *momentum, state, layer_idx);
        }
        Optimizer::Adam { learning_rate, beta1, beta2, epsilon } => {
            adam_update(layer, *learning_rate, *beta1, *beta2, *epsilon, state, layer_idx);
        }
        Optimizer::RMSprop { learning_rate, decay, epsilon } => {
            rmsprop_update(layer, *learning_rate, *decay, *epsilon, state, layer_idx);
        }
    }
}

/// SGD update: w = w - lr * grad
fn sgd_update(layer: &mut Dense, lr: f64) {
    // Update weights
    layer.weights.data = &layer.weights.data - &(&layer.grad_weights.data * lr);
    // Update biases
    layer.bias = &layer.bias - &(&layer.grad_bias * lr);
}

/// SGD with momentum: v = momentum * v + lr * grad, w = w - v
fn sgd_momentum_update(
    layer: &mut Dense,
    lr: f64,
    momentum: f64,
    state: &mut OptimizerState,
    idx: usize,
) {
    // Update velocity for weights
    state.m_weights[idx].data = &(&state.m_weights[idx].data * momentum) 
        + &(&layer.grad_weights.data * lr);
    layer.weights.data = &layer.weights.data - &state.m_weights[idx].data;

    // Update velocity for biases
    state.m_biases[idx] = &(&state.m_biases[idx] * momentum) + &(&layer.grad_bias * lr);
    layer.bias = &layer.bias - &state.m_biases[idx];
}

/// Adam update with bias correction
fn adam_update(
    layer: &mut Dense,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    state: &mut OptimizerState,
    idx: usize,
) {
    state.step += 1;
    let t = state.step as f64;

    // Update biased first moment estimate (weights)
    state.m_weights[idx].data = &(&state.m_weights[idx].data * beta1) 
        + &(&layer.grad_weights.data * (1.0 - beta1));
    
    // Update biased second moment estimate (weights)
    let grad_sq = layer.grad_weights.square();
    state.v_weights[idx].data = &(&state.v_weights[idx].data * beta2) 
        + &(&grad_sq.data * (1.0 - beta2));

    // Bias correction
    let m_hat = state.m_weights[idx].scale(1.0 / (1.0 - beta1.powf(t)));
    let v_hat = state.v_weights[idx].scale(1.0 / (1.0 - beta2.powf(t)));

    // Update weights
    let v_sqrt = v_hat.sqrt().map(|x| x + epsilon);
    let update = Tensor::new(&m_hat.data / &v_sqrt.data).scale(lr);
    layer.weights.data = &layer.weights.data - &update.data;

    // Same for biases
    state.m_biases[idx] = &(&state.m_biases[idx] * beta1) 
        + &(&layer.grad_bias * (1.0 - beta1));
    let grad_bias_sq: Array1<f64> = layer.grad_bias.mapv(|x| x * x);
    state.v_biases[idx] = &(&state.v_biases[idx] * beta2) 
        + &(&grad_bias_sq * (1.0 - beta2));

    let m_hat_b = &state.m_biases[idx] / (1.0 - beta1.powf(t));
    let v_hat_b = &state.v_biases[idx] / (1.0 - beta2.powf(t));
    let v_sqrt_b: Array1<f64> = v_hat_b.mapv(|x| x.sqrt() + epsilon);
    let update_b: Array1<f64> = &(&m_hat_b / &v_sqrt_b) * lr;
    layer.bias = &layer.bias - &update_b;
}

/// RMSprop update
fn rmsprop_update(
    layer: &mut Dense,
    lr: f64,
    decay: f64,
    epsilon: f64,
    state: &mut OptimizerState,
    idx: usize,
) {
    // Update cache for weights
    let grad_sq = layer.grad_weights.square();
    state.v_weights[idx].data = &(&state.v_weights[idx].data * decay) 
        + &(&grad_sq.data * (1.0 - decay));

    // Update weights
    let v_sqrt = state.v_weights[idx].sqrt().map(|x| x + epsilon);
    let update = Tensor::new(&layer.grad_weights.data / &v_sqrt.data).scale(lr);
    layer.weights.data = &layer.weights.data - &update.data;

    // Same for biases
    let grad_bias_sq: Array1<f64> = layer.grad_bias.mapv(|x| x * x);
    state.v_biases[idx] = &(&state.v_biases[idx] * decay) 
        + &(&grad_bias_sq * (1.0 - decay));
    let v_sqrt_b: Array1<f64> = state.v_biases[idx].mapv(|x| x.sqrt() + epsilon);
    let update_b: Array1<f64> = &(&layer.grad_bias / &v_sqrt_b) * lr;
    layer.bias = &layer.bias - &update_b;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sgd_update() {
        let mut layer = Dense::new(2, 2);
        layer.weights = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        layer.grad_weights = Tensor::new(arr2(&[[0.1, 0.2], [0.3, 0.4]]));
        
        sgd_update(&mut layer, 1.0);
        
        // weights should be reduced by gradients
        assert!((layer.weights.data[[0, 0]] - 0.9).abs() < 1e-10);
        assert!((layer.weights.data[[1, 1]] - 3.6).abs() < 1e-10);
    }

    #[test]
    fn test_optimizer_state_creation() {
        let layers = vec![
            Dense::new(10, 20),
            Dense::new(20, 5),
        ];
        
        let state = OptimizerState::new(&layers);
        
        assert_eq!(state.m_weights.len(), 2);
        assert_eq!(state.m_weights[0].shape(), (10, 20));
        assert_eq!(state.m_weights[1].shape(), (20, 5));
    }

    #[test]
    fn test_adam_step_counter() {
        let layers = vec![Dense::new(2, 2)];
        let mut state = OptimizerState::new(&layers);
        
        assert_eq!(state.step, 0);
        
        let mut layer = Dense::new(2, 2);
        layer.grad_weights = Tensor::new(arr2(&[[0.1, 0.1], [0.1, 0.1]]));
        layer.grad_bias = Array1::from_vec(vec![0.1, 0.1]);
        
        let optimizer = Optimizer::adam(0.001);
        update_dense_layer(&optimizer, &mut layer, &mut state, 0);
        
        assert_eq!(state.step, 1);
    }
}
