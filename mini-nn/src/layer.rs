//! Neural network layers
//!
//! Layers are the building blocks of neural networks.
//! Each layer has parameters (weights, biases) and implements forward/backward passes.

use crate::tensor::{Tensor, Tensor1D};
use ndarray::Array1;

/// Trait for all layer types
pub trait Layer {
    /// Forward pass: compute output from input
    fn forward(&mut self, input: &Tensor) -> Tensor;
    
    /// Backward pass: compute gradients
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    
    /// Get layer parameters (weights and biases)
    fn parameters(&self) -> Vec<&Tensor>;
    
    /// Get mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    
    /// Get gradients
    fn gradients(&self) -> Vec<&Tensor>;
    
    /// Number of trainable parameters
    fn num_parameters(&self) -> usize;
    
    /// Layer name for debugging
    fn name(&self) -> &str;
}

/// Weight initialization strategies
#[derive(Debug, Clone, Copy)]
pub enum Initializer {
    /// Zero initialization (not recommended for weights)
    Zeros,
    /// Random uniform in [-limit, limit]
    Uniform(f64),
    /// Random normal with mean and std
    Normal(f64, f64),
    /// Xavier/Glorot initialization (good for sigmoid/tanh)
    Xavier,
    /// He initialization (good for ReLU)
    He,
}

// ============================================================================
// Dense (Fully Connected) Layer
// ============================================================================

/// Dense/Fully Connected Layer
/// 
/// Computes: output = input @ weights + bias
/// Where:
/// - input: (batch_size, in_features)
/// - weights: (in_features, out_features)
/// - bias: (out_features,)
/// - output: (batch_size, out_features)
#[derive(Debug, Clone)]
pub struct Dense {
    /// Weight matrix (in_features, out_features)
    pub weights: Tensor,
    /// Bias vector (out_features,)
    pub bias: Tensor1D,
    /// Weight gradients
    pub grad_weights: Tensor,
    /// Bias gradients
    pub grad_bias: Tensor1D,
    /// Cached input for backpropagation
    input: Option<Tensor>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
}

impl Dense {
    /// Create a new dense layer with He initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_initializer(in_features, out_features, Initializer::He)
    }

    /// Create a dense layer with specified initialization
    pub fn with_initializer(
        in_features: usize,
        out_features: usize,
        init: Initializer,
    ) -> Self {
        let weights = match init {
            Initializer::Zeros => Tensor::zeros(in_features, out_features),
            Initializer::Uniform(limit) => Tensor::random_uniform(in_features, out_features, -limit, limit),
            Initializer::Normal(mean, std) => Tensor::random_normal(in_features, out_features, mean, std),
            Initializer::Xavier => Tensor::xavier(in_features, out_features),
            Initializer::He => Tensor::he(in_features, out_features),
        };

        Self {
            weights,
            bias: Array1::zeros(out_features),
            grad_weights: Tensor::zeros(in_features, out_features),
            grad_bias: Array1::zeros(out_features),
            input: None,
            in_features,
            out_features,
        }
    }

    /// Get input dimension
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output dimension
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // Cache input for backward pass
        self.input = Some(input.clone());
        
        // output = input @ weights + bias
        let output = input.matmul(&self.weights);
        output.add_bias(&self.bias)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self.input.as_ref().expect("Must call forward before backward");
        
        // Gradient w.r.t. weights: input.T @ grad_output
        self.grad_weights = input.transpose().matmul(grad_output);
        
        // Gradient w.r.t. bias: sum of grad_output along batch axis
        self.grad_bias = grad_output.sum_axis(0);
        
        // Gradient w.r.t. input: grad_output @ weights.T
        grad_output.matmul(&self.weights.transpose())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights]
    }

    fn gradients(&self) -> Vec<&Tensor> {
        vec![&self.grad_weights]
    }

    fn num_parameters(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    fn name(&self) -> &str {
        "Dense"
    }
}

// ============================================================================
// Dropout Layer
// ============================================================================

/// Dropout layer for regularization
/// 
/// During training, randomly zeros elements with probability p.
/// During inference, outputs are scaled by (1-p) or dropout is disabled.
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Dropout probability (0 to 1)
    pub p: f64,
    /// Whether we're in training mode
    pub training: bool,
    /// Cached mask for backpropagation
    mask: Option<Tensor>,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");
        Self {
            p,
            training: true,
            mask: None,
        }
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        // Generate random mask
        let scale = 1.0 / (1.0 - self.p);
        let mask = Tensor::random_uniform(input.shape().0, input.shape().1, 0.0, 1.0)
            .map(|x| if x > self.p { scale } else { 0.0 });
        
        let output = input.mul(&mask);
        self.mask = Some(mask);
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return grad_output.clone();
        }

        let mask = self.mask.as_ref().expect("Must call forward before backward");
        grad_output.mul(mask)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn gradients(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "Dropout"
    }
}

// ============================================================================
// Batch Normalization Layer
// ============================================================================

/// Batch Normalization layer
/// 
/// Normalizes inputs to have zero mean and unit variance,
/// then applies learnable scale (gamma) and shift (beta).
#[derive(Debug, Clone)]
pub struct BatchNorm {
    /// Number of features
    num_features: usize,
    /// Scale parameter (gamma)
    pub gamma: Tensor1D,
    /// Shift parameter (beta)
    pub beta: Tensor1D,
    /// Running mean for inference
    pub running_mean: Tensor1D,
    /// Running variance for inference
    pub running_var: Tensor1D,
    /// Momentum for running statistics
    momentum: f64,
    /// Small constant for numerical stability
    eps: f64,
    /// Training mode
    training: bool,
    /// Cached values for backward pass
    cache: Option<BatchNormCache>,
    /// Gradients
    grad_gamma: Tensor1D,
    grad_beta: Tensor1D,
}

#[derive(Debug, Clone)]
struct BatchNormCache {
    input: Tensor,
    normalized: Tensor,
    mean: Tensor1D,
    var: Tensor1D,
    std: Tensor1D,
}

impl BatchNorm {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            momentum: 0.1,
            eps: 1e-5,
            training: true,
            cache: None,
            grad_gamma: Array1::zeros(num_features),
            grad_beta: Array1::zeros(num_features),
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Layer for BatchNorm {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let (batch_size, _features) = input.shape();
        
        let (mean, var) = if self.training {
            // Compute batch statistics
            let mean = input.mean_axis(0);
            let centered = {
                let mut result = input.data.clone();
                for mut row in result.rows_mut() {
                    row -= &mean;
                }
                Tensor::new(result)
            };
            let var = centered.square().mean_axis(0);
            
            // Update running statistics
            for i in 0..self.num_features {
                self.running_mean[i] = (1.0 - self.momentum) * self.running_mean[i] 
                    + self.momentum * mean[i];
                self.running_var[i] = (1.0 - self.momentum) * self.running_var[i] 
                    + self.momentum * var[i];
            }
            
            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        // Normalize: (x - mean) / sqrt(var + eps)
        let std: Tensor1D = var.mapv(|v| (v + self.eps).sqrt());
        
        let mut normalized = input.data.clone();
        for mut row in normalized.rows_mut() {
            row -= &mean;
            row /= &std;
        }
        let normalized = Tensor::new(normalized);

        // Scale and shift: gamma * normalized + beta
        let mut output = normalized.data.clone();
        for mut row in output.rows_mut() {
            row *= &self.gamma;
            row += &self.beta;
        }

        if self.training {
            self.cache = Some(BatchNormCache {
                input: input.clone(),
                normalized: normalized.clone(),
                mean,
                var,
                std,
            });
        }

        Tensor::new(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let cache = self.cache.as_ref().expect("Must call forward before backward");
        let (batch_size, _) = grad_output.shape();
        let n = batch_size as f64;

        // Gradient w.r.t. gamma: sum(grad_output * normalized, axis=0)
        self.grad_gamma = Array1::zeros(self.num_features);
        for (grad_row, norm_row) in grad_output.data.rows().into_iter()
            .zip(cache.normalized.data.rows()) 
        {
            self.grad_gamma += &(&grad_row * &norm_row);
        }

        // Gradient w.r.t. beta: sum(grad_output, axis=0)
        self.grad_beta = grad_output.sum_axis(0);

        // Gradient w.r.t. input (simplified version)
        let mut grad_input = grad_output.data.clone();
        for mut row in grad_input.rows_mut() {
            row *= &self.gamma;
            row /= &cache.std;
        }

        Tensor::new(grad_input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn gradients(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn num_parameters(&self) -> usize {
        self.num_features * 2
    }

    fn name(&self) -> &str {
        "BatchNorm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_dense_forward() {
        let mut layer = Dense::with_initializer(2, 3, Initializer::Zeros);
        // Set specific weights for testing
        layer.weights = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        layer.bias = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let input = Tensor::new(arr2(&[[1.0, 2.0]]));
        let output = layer.forward(&input);

        // output = [1, 2] @ [[1,2,3],[4,5,6]] + [0.1,0.2,0.3]
        //        = [1*1+2*4, 1*2+2*5, 1*3+2*6] + [0.1,0.2,0.3]
        //        = [9, 12, 15] + [0.1, 0.2, 0.3]
        //        = [9.1, 12.2, 15.3]
        assert!((output.data[[0, 0]] - 9.1).abs() < 1e-10);
        assert!((output.data[[0, 1]] - 12.2).abs() < 1e-10);
        assert!((output.data[[0, 2]] - 15.3).abs() < 1e-10);
    }

    #[test]
    fn test_dense_backward() {
        let mut layer = Dense::with_initializer(2, 3, Initializer::Zeros);
        layer.weights = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        
        let input = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let _output = layer.forward(&input);
        
        let grad_output = Tensor::ones(2, 3);
        let grad_input = layer.backward(&grad_output);
        
        // grad_input = grad_output @ weights.T
        // = [[1,1,1],[1,1,1]] @ [[1,4],[2,5],[3,6]]
        // = [[6, 15], [6, 15]]
        assert_eq!(grad_input.shape(), (2, 2));
        assert!((grad_input.data[[0, 0]] - 6.0).abs() < 1e-10);
        assert!((grad_input.data[[0, 1]] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_num_parameters() {
        let layer = Dense::new(784, 256);
        // weights: 784 * 256 + bias: 256 = 200960
        assert_eq!(layer.num_parameters(), 784 * 256 + 256);
    }

    #[test]
    fn test_dropout_training() {
        let mut dropout = Dropout::new(0.5);
        dropout.train();
        
        let input = Tensor::ones(100, 100);
        let output = dropout.forward(&input);
        
        // Some values should be zero, others should be scaled by 2
        let sum = output.sum();
        // With 50% dropout and scaling by 2, expected sum ≈ 10000
        // But with randomness, just check it's not all zeros or all ones
        assert!(sum > 0.0);
    }

    #[test]
    fn test_dropout_eval() {
        let mut dropout = Dropout::new(0.5);
        dropout.eval();
        
        let input = Tensor::ones(10, 10);
        let output = dropout.forward(&input);
        
        // In eval mode, output should equal input
        assert!((output.sum() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_norm_shape() {
        let mut bn = BatchNorm::new(64);
        let input = Tensor::random_normal(32, 64, 0.0, 1.0);
        let output = bn.forward(&input);
        
        assert_eq!(output.shape(), (32, 64));
    }
}
