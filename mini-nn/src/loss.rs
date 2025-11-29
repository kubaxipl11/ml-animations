//! Loss functions for neural network training
//!
//! Loss functions measure the difference between predictions and targets.
//! Each loss function provides both the loss value and the gradient for backpropagation.

use crate::tensor::Tensor;

/// Available loss functions
#[derive(Debug, Clone, Copy)]
pub enum Loss {
    /// Mean Squared Error - for regression
    MSE,
    /// Binary Cross-Entropy - for binary classification
    BinaryCrossEntropy,
    /// Categorical Cross-Entropy - for multi-class classification
    CrossEntropy,
    /// Mean Absolute Error - for regression (robust to outliers)
    MAE,
}

/// Loss function implementation
#[derive(Debug, Clone)]
pub struct LossFunction {
    pub loss_type: Loss,
}

impl LossFunction {
    pub fn new(loss_type: Loss) -> Self {
        Self { loss_type }
    }

    /// Compute loss value
    pub fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        match self.loss_type {
            Loss::MSE => mse_loss(predictions, targets),
            Loss::BinaryCrossEntropy => binary_cross_entropy_loss(predictions, targets),
            Loss::CrossEntropy => cross_entropy_loss(predictions, targets),
            Loss::MAE => mae_loss(predictions, targets),
        }
    }

    /// Compute gradient of loss w.r.t. predictions
    pub fn gradient(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        match self.loss_type {
            Loss::MSE => mse_gradient(predictions, targets),
            Loss::BinaryCrossEntropy => binary_cross_entropy_gradient(predictions, targets),
            Loss::CrossEntropy => cross_entropy_gradient(predictions, targets),
            Loss::MAE => mae_gradient(predictions, targets),
        }
    }
}

// ============================================================================
// Mean Squared Error
// ============================================================================

/// MSE = (1/n) * sum((pred - target)²)
fn mse_loss(predictions: &Tensor, targets: &Tensor) -> f64 {
    let diff = predictions.sub(targets);
    let squared = diff.square();
    squared.mean()
}

/// d(MSE)/d(pred) = (2/n) * (pred - target)
fn mse_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let diff = predictions.sub(targets);
    let n = (predictions.shape().0 * predictions.shape().1) as f64;
    diff.scale(2.0 / n)
}

// ============================================================================
// Binary Cross-Entropy
// ============================================================================

/// BCE = -(1/n) * sum(target * log(pred) + (1-target) * log(1-pred))
fn binary_cross_entropy_loss(predictions: &Tensor, targets: &Tensor) -> f64 {
    let eps = 1e-15; // Small value to prevent log(0)
    let pred_clipped = predictions.clip(eps, 1.0 - eps);
    
    let term1 = targets.mul(&pred_clipped.ln());
    let term2 = targets.map(|x| 1.0 - x).mul(&pred_clipped.map(|x| (1.0 - x).ln()));
    
    let loss = term1.add(&term2);
    -loss.mean()
}

/// d(BCE)/d(pred) = -(target/pred - (1-target)/(1-pred)) / n
fn binary_cross_entropy_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let eps = 1e-15;
    let pred_clipped = predictions.clip(eps, 1.0 - eps);
    let n = predictions.shape().0 as f64;
    
    // gradient = -target/pred + (1-target)/(1-pred)
    // = (pred - target) / (pred * (1 - pred))
    let numerator = pred_clipped.sub(targets);
    let denominator = pred_clipped.mul(&pred_clipped.map(|x| 1.0 - x)).clip(eps, 1.0);
    
    let grad = Tensor::new(&numerator.data / &denominator.data);
    grad.scale(1.0 / n)
}

// ============================================================================
// Categorical Cross-Entropy (with Softmax)
// ============================================================================

/// CE = -(1/n) * sum(target * log(pred))
/// Assumes predictions are softmax outputs (sum to 1)
fn cross_entropy_loss(predictions: &Tensor, targets: &Tensor) -> f64 {
    let eps = 1e-15;
    let pred_clipped = predictions.clip(eps, 1.0 - eps);
    let log_pred = pred_clipped.ln();
    let loss = targets.mul(&log_pred);
    -loss.sum() / predictions.shape().0 as f64
}

/// For softmax + cross-entropy, gradient simplifies to: pred - target
fn cross_entropy_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let n = predictions.shape().0 as f64;
    predictions.sub(targets).scale(1.0 / n)
}

// ============================================================================
// Mean Absolute Error
// ============================================================================

/// MAE = (1/n) * sum(|pred - target|)
fn mae_loss(predictions: &Tensor, targets: &Tensor) -> f64 {
    let diff = predictions.sub(targets);
    let abs_diff = diff.map(|x| x.abs());
    abs_diff.mean()
}

/// d(MAE)/d(pred) = sign(pred - target) / n
fn mae_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let diff = predictions.sub(targets);
    let n = (predictions.shape().0 * predictions.shape().1) as f64;
    diff.map(|x| x.signum() / n)
}

// ============================================================================
// Accuracy metrics
// ============================================================================

/// Binary accuracy (for sigmoid output)
pub fn binary_accuracy(predictions: &Tensor, targets: &Tensor) -> f64 {
    let pred_classes: Vec<f64> = predictions.data.iter()
        .map(|&x| if x >= 0.5 { 1.0 } else { 0.0 })
        .collect();
    
    let correct: usize = pred_classes.iter()
        .zip(targets.data.iter())
        .filter(|(p, t)| (*p - *t).abs() < 0.5)
        .count();
    
    correct as f64 / predictions.data.len() as f64
}

/// Categorical accuracy (for softmax output)
pub fn categorical_accuracy(predictions: &Tensor, targets: &Tensor) -> f64 {
    let pred_classes = predictions.argmax_axis1();
    let target_classes = targets.argmax_axis1();
    
    let correct: usize = pred_classes.iter()
        .zip(target_classes.iter())
        .filter(|(p, t)| p == t)
        .count();
    
    correct as f64 / pred_classes.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let target = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        
        let loss = mse_loss(&pred, &target);
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Tensor::new(arr2(&[[1.0], [2.0]]));
        let target = Tensor::new(arr2(&[[2.0], [4.0]]));
        
        // MSE = ((1-2)² + (2-4)²) / 2 = (1 + 4) / 2 = 2.5
        let loss = mse_loss(&pred, &target);
        assert!((loss - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_mse_gradient() {
        let pred = Tensor::new(arr2(&[[1.0], [2.0]]));
        let target = Tensor::new(arr2(&[[2.0], [4.0]]));
        
        let grad = mse_gradient(&pred, &target);
        // grad = 2/2 * [-1, -2] = [-1, -2]
        assert!((grad.data[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((grad.data[[1, 0]] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_binary_cross_entropy() {
        // Perfect predictions should have near-zero loss
        let pred = Tensor::new(arr2(&[[0.99], [0.01]]));
        let target = Tensor::new(arr2(&[[1.0], [0.0]]));
        
        let loss = binary_cross_entropy_loss(&pred, &target);
        assert!(loss < 0.1);
    }

    #[test]
    fn test_cross_entropy() {
        // One-hot encoded targets
        let pred = Tensor::new(arr2(&[[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]));
        let target = Tensor::new(arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]));
        
        let loss = cross_entropy_loss(&pred, &target);
        assert!(loss > 0.0);
        assert!(loss < 1.0); // Should be reasonably low for good predictions
    }

    #[test]
    fn test_mae_loss() {
        let pred = Tensor::new(arr2(&[[1.0], [3.0]]));
        let target = Tensor::new(arr2(&[[2.0], [5.0]]));
        
        // MAE = (|1-2| + |3-5|) / 2 = (1 + 2) / 2 = 1.5
        let loss = mae_loss(&pred, &target);
        assert!((loss - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_binary_accuracy() {
        let pred = Tensor::new(arr2(&[[0.7], [0.3], [0.8], [0.2]]));
        let target = Tensor::new(arr2(&[[1.0], [0.0], [1.0], [0.0]]));
        
        let acc = binary_accuracy(&pred, &target);
        assert!((acc - 1.0).abs() < 1e-10); // All correct
    }

    #[test]
    fn test_categorical_accuracy() {
        let pred = Tensor::new(arr2(&[[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]));
        let target = Tensor::new(arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]));
        
        let acc = categorical_accuracy(&pred, &target);
        assert!((acc - 1.0).abs() < 1e-10); // Both correct
    }
}
