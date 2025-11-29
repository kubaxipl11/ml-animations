//! Activation functions for neural networks
//!
//! Activations introduce non-linearity, allowing networks to learn complex patterns.
//! Each activation has a forward pass and a derivative for backpropagation.

use crate::tensor::Tensor;

/// Available activation functions
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// Linear (identity) - no transformation
    Linear,
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: max(alpha*x, x) where alpha is small (0.01)
    LeakyReLU(f64),
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// Softmax: exp(x_i) / sum(exp(x_j)) - for multi-class classification
    Softmax,
}

impl Default for Activation {
    fn default() -> Self {
        Activation::Linear
    }
}

/// Activation layer for use in sequential networks
#[derive(Debug, Clone)]
pub struct ActivationLayer {
    pub activation: Activation,
    /// Cached input for backpropagation
    input: Option<Tensor>,
    /// Cached output for backpropagation
    output: Option<Tensor>,
}

impl ActivationLayer {
    pub fn new(activation: Activation) -> Self {
        Self {
            activation,
            input: None,
            output: None,
        }
    }

    /// Forward pass - apply activation
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input = Some(input.clone());
        let output = apply_activation(&self.activation, input);
        self.output = Some(output.clone());
        output
    }

    /// Backward pass - compute gradient w.r.t. input
    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        let input = self.input.as_ref().expect("Must call forward before backward");
        let grad = activation_derivative(&self.activation, input, self.output.as_ref());
        grad_output.mul(&grad)
    }
}

/// Apply activation function to tensor
pub fn apply_activation(activation: &Activation, x: &Tensor) -> Tensor {
    match activation {
        Activation::Linear => x.clone(),
        
        Activation::ReLU => x.map(|val| val.max(0.0)),
        
        Activation::LeakyReLU(alpha) => {
            let alpha = *alpha;
            x.map(|val| if val > 0.0 { val } else { alpha * val })
        }
        
        Activation::Sigmoid => x.map(|val| sigmoid(val)),
        
        Activation::Tanh => x.map(|val| val.tanh()),
        
        Activation::Softmax => softmax(x),
    }
}

/// Compute derivative of activation function
/// For backpropagation: gradient of loss w.r.t. activation input
pub fn activation_derivative(
    activation: &Activation,
    input: &Tensor,
    output: Option<&Tensor>,
) -> Tensor {
    match activation {
        Activation::Linear => Tensor::ones(input.shape().0, input.shape().1),
        
        Activation::ReLU => input.map(|val| if val > 0.0 { 1.0 } else { 0.0 }),
        
        Activation::LeakyReLU(alpha) => {
            let alpha = *alpha;
            input.map(|val| if val > 0.0 { 1.0 } else { alpha })
        }
        
        Activation::Sigmoid => {
            // σ'(x) = σ(x) * (1 - σ(x))
            // Use cached output if available for efficiency
            let s = output
                .map(|o| o.clone())
                .unwrap_or_else(|| input.map(sigmoid));
            let one_minus_s = s.map(|x| 1.0 - x);
            s.mul(&one_minus_s)
        }
        
        Activation::Tanh => {
            // tanh'(x) = 1 - tanh²(x)
            let t = output
                .map(|o| o.clone())
                .unwrap_or_else(|| input.map(|x| x.tanh()));
            t.map(|x| 1.0 - x * x)
        }
        
        Activation::Softmax => {
            // For softmax + cross-entropy, the gradient simplifies
            // This returns the Jacobian diagonal approximation
            // Full Jacobian is: S_i * (δ_ij - S_j)
            // But typically handled in loss function
            let s = output
                .map(|o| o.clone())
                .unwrap_or_else(|| softmax(input));
            s.mul(&s.map(|x| 1.0 - x))
        }
    }
}

/// Sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Softmax function - numerically stable version
/// exp(x_i - max(x)) / sum(exp(x_j - max(x)))
fn softmax(x: &Tensor) -> Tensor {
    let mut result = x.data.clone();
    
    for mut row in result.rows_mut() {
        // Subtract max for numerical stability
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - max_val).exp());
        
        // Normalize
        let sum: f64 = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    
    Tensor::new(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_relu() {
        let x = Tensor::new(arr2(&[[-1.0, 0.0, 1.0, 2.0]]));
        let y = apply_activation(&Activation::ReLU, &x);
        assert_eq!(y.data[[0, 0]], 0.0);
        assert_eq!(y.data[[0, 1]], 0.0);
        assert_eq!(y.data[[0, 2]], 1.0);
        assert_eq!(y.data[[0, 3]], 2.0);
    }

    #[test]
    fn test_relu_derivative() {
        let x = Tensor::new(arr2(&[[-1.0, 0.0, 1.0, 2.0]]));
        let d = activation_derivative(&Activation::ReLU, &x, None);
        assert_eq!(d.data[[0, 0]], 0.0);
        assert_eq!(d.data[[0, 2]], 1.0);
    }

    #[test]
    fn test_sigmoid() {
        let x = Tensor::new(arr2(&[[0.0]]));
        let y = apply_activation(&Activation::Sigmoid, &x);
        assert!((y.data[[0, 0]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_bounds() {
        let x = Tensor::new(arr2(&[[-100.0, 0.0, 100.0]]));
        let y = apply_activation(&Activation::Sigmoid, &x);
        assert!(y.data[[0, 0]] > 0.0 && y.data[[0, 0]] < 0.01);
        assert!((y.data[[0, 1]] - 0.5).abs() < 1e-10);
        assert!(y.data[[0, 2]] > 0.99 && y.data[[0, 2]] < 1.0);
    }

    #[test]
    fn test_tanh() {
        let x = Tensor::new(arr2(&[[0.0]]));
        let y = apply_activation(&Activation::Tanh, &x);
        assert!(y.data[[0, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let x = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let y = apply_activation(&Activation::Softmax, &x);
        
        // Each row should sum to 1
        let row0_sum: f64 = y.data.row(0).sum();
        let row1_sum: f64 = y.data.row(1).sum();
        
        assert!((row0_sum - 1.0).abs() < 1e-10);
        assert!((row1_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not cause overflow
        let x = Tensor::new(arr2(&[[1000.0, 1001.0, 1002.0]]));
        let y = apply_activation(&Activation::Softmax, &x);
        
        // Should still sum to 1
        let sum: f64 = y.data.row(0).sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        // Max should have highest probability
        assert!(y.data[[0, 2]] > y.data[[0, 1]]);
        assert!(y.data[[0, 1]] > y.data[[0, 0]]);
    }

    #[test]
    fn test_leaky_relu() {
        let x = Tensor::new(arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]));
        let y = apply_activation(&Activation::LeakyReLU(0.01), &x);
        
        assert!((y.data[[0, 0]] - (-0.02)).abs() < 1e-10);
        assert_eq!(y.data[[0, 2]], 0.0);
        assert_eq!(y.data[[0, 4]], 2.0);
    }

    #[test]
    fn test_activation_layer() {
        let mut layer = ActivationLayer::new(Activation::ReLU);
        let x = Tensor::new(arr2(&[[-1.0, 2.0], [3.0, -4.0]]));
        
        let y = layer.forward(&x);
        assert_eq!(y.data[[0, 0]], 0.0);
        assert_eq!(y.data[[0, 1]], 2.0);
        
        let grad_out = Tensor::ones(2, 2);
        let grad_in = layer.backward(&grad_out);
        assert_eq!(grad_in.data[[0, 0]], 0.0);
        assert_eq!(grad_in.data[[0, 1]], 1.0);
    }
}
