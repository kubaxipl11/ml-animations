//! # Mini Neural Network
//!
//! A minimal neural network implementation in Rust, built from scratch for educational purposes.
//!
//! This crate provides:
//! - **Tensor operations**: Matrix math with ndarray
//! - **Layers**: Dense/Linear layers with various initializations
//! - **Activations**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
//! - **Loss functions**: MSE, CrossEntropy, BinaryCrossEntropy
//! - **Optimizers**: SGD, SGD with Momentum, Adam
//! - **Training**: Backpropagation, batching, early stopping
//!
//! ## Example
//!
//! ```rust,ignore
//! use mini_nn::{Network, Dense, Activation, Loss, Optimizer};
//!
//! // Create a simple network for XOR
//! let mut network = Network::new()
//!     .add(Dense::new(2, 4))
//!     .add(Activation::ReLU)
//!     .add(Dense::new(4, 1))
//!     .add(Activation::Sigmoid);
//!
//! // Train the network
//! network.compile(Loss::BinaryCrossEntropy, Optimizer::Adam(0.01));
//! network.fit(&x_train, &y_train, epochs, batch_size);
//! ```

pub mod tensor;
pub mod layer;
pub mod activation;
pub mod loss;
pub mod optimizer;
pub mod network;
pub mod training;
pub mod data;

// Re-exports for convenience
pub use tensor::Tensor;
pub use layer::{Layer, Dense};
pub use activation::{Activation, ActivationLayer};
pub use loss::{Loss, LossFunction};
pub use optimizer::{Optimizer, OptimizerState};
pub use network::Network;
pub use training::{Trainer, TrainingConfig};
pub use data::{Dataset, DataLoader};
