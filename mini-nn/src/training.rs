//! Training utilities
//!
//! This module provides high-level training functionality including
//! batch training, validation, early stopping, and progress tracking.

use crate::tensor::{Tensor, shuffle_together};
use crate::network::Network;
use crate::loss::{Loss, binary_accuracy};
use crate::optimizer::Optimizer;
use indicatif::{ProgressBar, ProgressStyle};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Validation split (0.0 to 1.0)
    pub validation_split: f64,
    /// Shuffle data each epoch
    pub shuffle: bool,
    /// Early stopping patience (0 = disabled)
    pub early_stopping_patience: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            validation_split: 0.2,
            shuffle: true,
            early_stopping_patience: 10,
            verbose: true,
        }
    }
}

/// Training history
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    pub train_loss: Vec<f64>,
    pub train_accuracy: Vec<f64>,
    pub val_loss: Vec<f64>,
    pub val_accuracy: Vec<f64>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get best validation loss
    pub fn best_val_loss(&self) -> Option<f64> {
        self.val_loss.iter().cloned().reduce(f64::min)
    }

    /// Get best validation accuracy
    pub fn best_val_accuracy(&self) -> Option<f64> {
        self.val_accuracy.iter().cloned().reduce(f64::max)
    }
}

/// Trainer for neural networks
pub struct Trainer {
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train the network
    pub fn fit(
        &self,
        network: &mut Network,
        x: &Tensor,
        y: &Tensor,
        loss: Loss,
        optimizer: Optimizer,
    ) -> TrainingHistory {
        // Compile the network
        network.compile(loss, optimizer);
        
        // Split data into train/val
        let n_samples = x.shape().0;
        let n_val = (n_samples as f64 * self.config.validation_split) as usize;
        let n_train = n_samples - n_val;
        
        let mut x_train = x.slice_rows(0, n_train);
        let mut y_train = y.slice_rows(0, n_train);
        let x_val = x.slice_rows(n_train, n_samples);
        let y_val = y.slice_rows(n_train, n_samples);
        
        let mut history = TrainingHistory::new();
        let mut best_val_loss = f64::MAX;
        let mut patience_counter = 0;

        // Training loop
        for epoch in 0..self.config.epochs {
            // Shuffle training data
            if self.config.shuffle {
                shuffle_together(&mut x_train, &mut y_train);
            }

            // Train for one epoch
            let (train_loss, train_acc) = self.train_epoch(network, &x_train, &y_train);
            
            // Validate
            let (val_loss, val_acc) = self.evaluate(network, &x_val, &y_val);
            
            // Record history
            history.train_loss.push(train_loss);
            history.train_accuracy.push(train_acc);
            history.val_loss.push(val_loss);
            history.val_accuracy.push(val_acc);
            
            // Print progress
            if self.config.verbose {
                println!(
                    "Epoch {}/{}: train_loss={:.4}, train_acc={:.4}, val_loss={:.4}, val_acc={:.4}",
                    epoch + 1, self.config.epochs, train_loss, train_acc, val_loss, val_acc
                );
            }
            
            // Early stopping check
            if self.config.early_stopping_patience > 0 {
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.early_stopping_patience {
                        if self.config.verbose {
                            println!("Early stopping at epoch {}", epoch + 1);
                        }
                        break;
                    }
                }
            }
        }
        
        history
    }

    /// Train for one epoch
    fn train_epoch(&self, network: &mut Network, x: &Tensor, y: &Tensor) -> (f64, f64) {
        network.train();
        
        let n_samples = x.shape().0;
        let n_batches = (n_samples + self.config.batch_size - 1) / self.config.batch_size;
        
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(n_samples);
            
            let x_batch = x.slice_rows(start, end);
            let y_batch = y.slice_rows(start, end);
            
            let (loss, acc) = network.train_step(&x_batch, &y_batch);
            total_loss += loss;
            total_acc += acc;
        }
        
        (total_loss / n_batches as f64, total_acc / n_batches as f64)
    }

    /// Evaluate the network
    pub fn evaluate(&self, network: &mut Network, x: &Tensor, y: &Tensor) -> (f64, f64) {
        network.eval();
        
        let predictions = network.forward(x);
        let loss_fn = crate::loss::LossFunction::new(Loss::BinaryCrossEntropy);
        let loss = loss_fn.compute(&predictions, y);
        let accuracy = binary_accuracy(&predictions, y);
        
        network.train();
        
        (loss, accuracy)
    }
}

/// Simple training function for quick experiments
pub fn train_simple(
    network: &mut Network,
    x: &Tensor,
    y: &Tensor,
    epochs: usize,
    learning_rate: f64,
) -> TrainingHistory {
    let config = TrainingConfig {
        epochs,
        batch_size: 32,
        validation_split: 0.2,
        shuffle: true,
        early_stopping_patience: 0,
        verbose: true,
    };
    
    let trainer = Trainer::new(config);
    trainer.fit(network, x, y, Loss::BinaryCrossEntropy, Optimizer::adam(learning_rate))
}

/// Train with progress bar (for longer training)
pub fn train_with_progress(
    network: &mut Network,
    x: &Tensor,
    y: &Tensor,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) -> TrainingHistory {
    network.compile(Loss::BinaryCrossEntropy, Optimizer::adam(learning_rate));
    
    let n_samples = x.shape().0;
    let n_batches = (n_samples + batch_size - 1) / batch_size;
    
    let mut x_train = x.clone();
    let mut y_train = y.clone();
    let mut history = TrainingHistory::new();
    
    let pb = ProgressBar::new((epochs * n_batches) as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));
    
    for epoch in 0..epochs {
        shuffle_together(&mut x_train, &mut y_train);
        
        let mut epoch_loss = 0.0;
        let mut epoch_acc = 0.0;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n_samples);
            
            let x_batch = x_train.slice_rows(start, end);
            let y_batch = y_train.slice_rows(start, end);
            
            let (loss, acc) = network.train_step(&x_batch, &y_batch);
            epoch_loss += loss;
            epoch_acc += acc;
            
            pb.inc(1);
        }
        
        history.train_loss.push(epoch_loss / n_batches as f64);
        history.train_accuracy.push(epoch_acc / n_batches as f64);
    }
    
    pb.finish_with_message("Training complete!");
    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use crate::activation::Activation;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        history.val_loss = vec![0.5, 0.4, 0.3, 0.35];
        history.val_accuracy = vec![0.7, 0.75, 0.8, 0.78];
        
        assert!((history.best_val_loss().unwrap() - 0.3).abs() < 1e-10);
        assert!((history.best_val_accuracy().unwrap() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_simple_training() {
        // XOR dataset
        let x = Tensor::new(arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]));
        let y = Tensor::new(arr2(&[
            [0.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0],
            [1.0],
        ]));
        
        let mut network = Network::new()
            .add_dense(2, 4)
            .add_activation(Activation::ReLU)
            .add_dense(4, 1)
            .add_activation(Activation::Sigmoid);
        
        let config = TrainingConfig {
            epochs: 10,
            batch_size: 4,
            validation_split: 0.2,
            shuffle: false,
            early_stopping_patience: 0,
            verbose: false,
        };
        
        let trainer = Trainer::new(config);
        let history = trainer.fit(&mut network, &x, &y, Loss::BinaryCrossEntropy, Optimizer::adam(0.1));
        
        assert_eq!(history.train_loss.len(), 10);
    }
}
