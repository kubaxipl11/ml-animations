//! Neural Network container
//!
//! The Network struct provides a high-level interface for building,
//! training, and using neural networks.

use crate::tensor::Tensor;
use crate::layer::{Dense, Layer};
use crate::activation::{Activation, ActivationLayer};
use crate::loss::{Loss, LossFunction, binary_accuracy};
use crate::optimizer::{Optimizer, OptimizerState, update_dense_layer};

/// A layer in the network (either Dense or Activation)
#[derive(Debug, Clone)]
pub enum NetworkLayer {
    Dense(Dense),
    Activation(ActivationLayer),
}

/// Sequential neural network
#[derive(Debug)]
pub struct Network {
    /// Network layers
    layers: Vec<NetworkLayer>,
    /// Loss function
    loss_fn: Option<LossFunction>,
    /// Optimizer
    optimizer: Option<Optimizer>,
    /// Optimizer state (for momentum, Adam, etc.)
    optimizer_state: Option<OptimizerState>,
    /// Training mode
    training: bool,
}

impl Network {
    /// Create a new empty network
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            loss_fn: None,
            optimizer: None,
            optimizer_state: None,
            training: true,
        }
    }

    /// Add a dense layer
    pub fn add_dense(mut self, in_features: usize, out_features: usize) -> Self {
        self.layers.push(NetworkLayer::Dense(Dense::new(in_features, out_features)));
        self
    }

    /// Add an activation layer
    pub fn add_activation(mut self, activation: Activation) -> Self {
        self.layers.push(NetworkLayer::Activation(ActivationLayer::new(activation)));
        self
    }

    /// Add a layer (generic)
    pub fn add(mut self, layer: NetworkLayer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Configure the network for training
    pub fn compile(&mut self, loss: Loss, optimizer: Optimizer) {
        self.loss_fn = Some(LossFunction::new(loss));
        self.optimizer = Some(optimizer.clone());
        
        // Initialize optimizer state
        let dense_layers: Vec<Dense> = self.layers.iter()
            .filter_map(|l| match l {
                NetworkLayer::Dense(d) => Some(d.clone()),
                _ => None,
            })
            .collect();
        
        self.optimizer_state = Some(OptimizerState::new(&dense_layers));
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = match layer {
                NetworkLayer::Dense(d) => d.forward(&x),
                NetworkLayer::Activation(a) => a.forward(&x),
            };
        }
        x
    }

    /// Backward pass through the network
    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = match layer {
                NetworkLayer::Dense(d) => d.backward(&grad),
                NetworkLayer::Activation(a) => a.backward(&grad),
            };
        }
        grad
    }

    /// Update parameters using optimizer
    pub fn update_parameters(&mut self) {
        let optimizer = self.optimizer.as_ref().expect("Must compile network first");
        let state = self.optimizer_state.as_mut().expect("Must compile network first");
        
        let mut dense_idx = 0;
        for layer in &mut self.layers {
            if let NetworkLayer::Dense(d) = layer {
                update_dense_layer(optimizer, d, state, dense_idx);
                dense_idx += 1;
            }
        }
    }

    /// Train for one batch
    pub fn train_step(&mut self, x: &Tensor, y: &Tensor) -> (f64, f64) {
        // Forward pass
        let predictions = self.forward(x);
        
        let loss_fn = self.loss_fn.as_ref().expect("Must compile network first");
        
        // Compute loss
        let loss = loss_fn.compute(&predictions, y);
        
        // Compute accuracy
        let accuracy = binary_accuracy(&predictions, y);
        
        // Backward pass
        let grad = loss_fn.gradient(&predictions, y);
        self.backward(&grad);
        
        // Update parameters
        self.update_parameters();
        
        (loss, accuracy)
    }

    /// Predict (forward pass without training)
    pub fn predict(&mut self, x: &Tensor) -> Tensor {
        self.eval();
        let output = self.forward(x);
        self.train();
        output
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter()
            .map(|l| match l {
                NetworkLayer::Dense(d) => d.num_parameters(),
                NetworkLayer::Activation(_) => 0,
            })
            .sum()
    }

    /// Print network summary
    pub fn summary(&self) {
        println!("┌─────────────────────────────────────────────────────┐");
        println!("│                   Network Summary                    │");
        println!("├─────────────────────────────────────────────────────┤");
        println!("│ Layer                  │ Output Shape │ Parameters  │");
        println!("├─────────────────────────────────────────────────────┤");
        
        let mut total_params = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                NetworkLayer::Dense(d) => {
                    let params = d.num_parameters();
                    total_params += params;
                    println!("│ Dense({:4}, {:4})      │   ({:4},)    │  {:9}  │", 
                        d.in_features(), d.out_features(), d.out_features(), params);
                }
                NetworkLayer::Activation(a) => {
                    let name = match a.activation {
                        Activation::ReLU => "ReLU",
                        Activation::Sigmoid => "Sigmoid",
                        Activation::Tanh => "Tanh",
                        Activation::Softmax => "Softmax",
                        Activation::LeakyReLU(_) => "LeakyReLU",
                        Activation::Linear => "Linear",
                    };
                    println!("│ {:22} │   (same)     │          0  │", name);
                }
            }
        }
        
        println!("├─────────────────────────────────────────────────────┤");
        println!("│ Total Parameters: {:34} │", total_params);
        println!("└─────────────────────────────────────────────────────┘");
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating networks easily
pub struct NetworkBuilder {
    layers: Vec<NetworkLayer>,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn dense(mut self, in_features: usize, out_features: usize) -> Self {
        self.layers.push(NetworkLayer::Dense(Dense::new(in_features, out_features)));
        self
    }

    pub fn relu(mut self) -> Self {
        self.layers.push(NetworkLayer::Activation(ActivationLayer::new(Activation::ReLU)));
        self
    }

    pub fn sigmoid(mut self) -> Self {
        self.layers.push(NetworkLayer::Activation(ActivationLayer::new(Activation::Sigmoid)));
        self
    }

    pub fn tanh(mut self) -> Self {
        self.layers.push(NetworkLayer::Activation(ActivationLayer::new(Activation::Tanh)));
        self
    }

    pub fn softmax(mut self) -> Self {
        self.layers.push(NetworkLayer::Activation(ActivationLayer::new(Activation::Softmax)));
        self
    }

    pub fn build(self) -> Network {
        Network {
            layers: self.layers,
            loss_fn: None,
            optimizer: None,
            optimizer_state: None,
            training: true,
        }
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_network_creation() {
        let network = Network::new()
            .add_dense(2, 4)
            .add_activation(Activation::ReLU)
            .add_dense(4, 1)
            .add_activation(Activation::Sigmoid);
        
        assert_eq!(network.layers.len(), 4);
    }

    #[test]
    fn test_network_forward() {
        let mut network = Network::new()
            .add_dense(2, 4)
            .add_activation(Activation::ReLU)
            .add_dense(4, 1)
            .add_activation(Activation::Sigmoid);
        
        let input = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let output = network.forward(&input);
        
        assert_eq!(output.shape(), (2, 1));
        // Output should be between 0 and 1 due to sigmoid
        assert!(output.data[[0, 0]] >= 0.0 && output.data[[0, 0]] <= 1.0);
    }

    #[test]
    fn test_network_builder() {
        let network = NetworkBuilder::new()
            .dense(10, 5)
            .relu()
            .dense(5, 2)
            .softmax()
            .build();
        
        assert_eq!(network.layers.len(), 4);
    }

    #[test]
    fn test_network_num_parameters() {
        let network = Network::new()
            .add_dense(10, 5)  // 10*5 + 5 = 55
            .add_activation(Activation::ReLU)
            .add_dense(5, 2);  // 5*2 + 2 = 12
        
        assert_eq!(network.num_parameters(), 55 + 12);
    }
}
