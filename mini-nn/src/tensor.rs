//! Tensor operations using ndarray
//!
//! This module provides a thin wrapper around ndarray with common operations
//! needed for neural networks: matrix multiplication, element-wise operations,
//! broadcasting, and more.

use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform};
use rand::Rng;

/// Type alias for 1D tensor (vector)
pub type Tensor1D = Array1<f64>;

/// Type alias for 2D tensor (matrix)
pub type Tensor2D = Array2<f64>;

/// Tensor wrapper for neural network operations
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Tensor2D,
}

impl Tensor {
    /// Create a new tensor from a 2D array
    pub fn new(data: Tensor2D) -> Self {
        Self { data }
    }

    /// Create a tensor of zeros with given shape
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: Array2::zeros((rows, cols)),
        }
    }

    /// Create a tensor of ones with given shape
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: Array2::ones((rows, cols)),
        }
    }

    /// Create a tensor with random values from uniform distribution
    pub fn random_uniform(rows: usize, cols: usize, low: f64, high: f64) -> Self {
        Self {
            data: Array2::random((rows, cols), Uniform::new(low, high)),
        }
    }

    /// Create a tensor with random values from normal distribution
    pub fn random_normal(rows: usize, cols: usize, mean: f64, std: f64) -> Self {
        Self {
            data: Array2::random((rows, cols), Normal::new(mean, std).unwrap()),
        }
    }

    /// Xavier/Glorot initialization for weights
    /// Good for sigmoid and tanh activations
    pub fn xavier(fan_in: usize, fan_out: usize) -> Self {
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        Self::random_uniform(fan_in, fan_out, -limit, limit)
    }

    /// He initialization for weights
    /// Good for ReLU activations
    pub fn he(fan_in: usize, fan_out: usize) -> Self {
        let std = (2.0 / fan_in as f64).sqrt();
        Self::random_normal(fan_in, fan_out, 0.0, std)
    }

    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.data.nrows(), self.data.ncols())
    }

    /// Matrix multiplication: self @ other
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        Tensor::new(self.data.dot(&other.data))
    }

    /// Matrix-vector multiplication
    pub fn dot_vec(&self, vec: &Tensor1D) -> Tensor1D {
        self.data.dot(vec)
    }

    /// Transpose the tensor
    pub fn transpose(&self) -> Tensor {
        Tensor::new(self.data.t().to_owned())
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor::new(&self.data + &other.data)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        Tensor::new(&self.data - &other.data)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor) -> Tensor {
        Tensor::new(&self.data * &other.data)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Tensor {
        Tensor::new(&self.data * scalar)
    }

    /// Sum all elements
    pub fn sum(&self) -> f64 {
        self.data.sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    /// Sum along axis (0 = columns, 1 = rows)
    pub fn sum_axis(&self, axis: usize) -> Tensor1D {
        self.data.sum_axis(Axis(axis))
    }

    /// Mean along axis
    pub fn mean_axis(&self, axis: usize) -> Tensor1D {
        self.data.mean_axis(Axis(axis)).unwrap()
    }

    /// Apply a function element-wise
    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f64) -> f64,
    {
        Tensor::new(self.data.mapv(|x| f(x)))
    }

    /// Add bias (broadcast across rows)
    /// Input shape: (batch_size, features)
    /// Bias shape: (features,)
    pub fn add_bias(&self, bias: &Tensor1D) -> Tensor {
        let mut result = self.data.clone();
        for mut row in result.rows_mut() {
            row += bias;
        }
        Tensor::new(result)
    }

    /// Get a batch of rows
    pub fn slice_rows(&self, start: usize, end: usize) -> Tensor {
        Tensor::new(self.data.slice(s![start..end, ..]).to_owned())
    }

    /// Clip values to a range
    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        self.map(|x| x.max(min).min(max))
    }

    /// Element-wise square
    pub fn square(&self) -> Tensor {
        self.map(|x| x * x)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Tensor {
        self.map(|x| x.sqrt())
    }

    /// Element-wise power
    pub fn pow(&self, exp: f64) -> Tensor {
        self.map(|x| x.powf(exp))
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Tensor {
        self.map(|x| x.exp())
    }

    /// Element-wise natural logarithm
    pub fn ln(&self) -> Tensor {
        self.map(|x| x.ln())
    }

    /// Element-wise max with scalar
    pub fn maximum(&self, val: f64) -> Tensor {
        self.map(|x| x.max(val))
    }

    /// Argmax along axis 1 (returns index of max in each row)
    pub fn argmax_axis1(&self) -> Vec<usize> {
        self.data
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }
}

/// Shuffle two tensors together (maintaining correspondence)
pub fn shuffle_together(x: &mut Tensor, y: &mut Tensor) {
    let n = x.shape().0;
    let mut rng = rand::thread_rng();
    
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        // Swap rows in x
        for col in 0..x.shape().1 {
            let tmp = x.data[[i, col]];
            x.data[[i, col]] = x.data[[j, col]];
            x.data[[j, col]] = tmp;
        }
        // Swap rows in y
        for col in 0..y.shape().1 {
            let tmp = y.data[[i, col]];
            y.data[[i, col]] = y.data[[j, col]];
            y.data[[j, col]] = tmp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(3, 4);
        assert_eq!(t.shape(), (3, 4));
        assert_eq!(t.sum(), 0.0);
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(2, 3);
        assert_eq!(t.sum(), 6.0);
    }

    #[test]
    fn test_matmul() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        let c = a.matmul(&b);
        
        assert_eq!(c.data[[0, 0]], 19.0);
        assert_eq!(c.data[[0, 1]], 22.0);
        assert_eq!(c.data[[1, 0]], 43.0);
        assert_eq!(c.data[[1, 1]], 50.0);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let t = a.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.data[[0, 0]], 1.0);
        assert_eq!(t.data[[2, 1]], 6.0);
    }

    #[test]
    fn test_xavier_shape() {
        let w = Tensor::xavier(784, 256);
        assert_eq!(w.shape(), (784, 256));
    }

    #[test]
    fn test_he_shape() {
        let w = Tensor::he(256, 128);
        assert_eq!(w.shape(), (256, 128));
    }

    #[test]
    fn test_element_wise_ops() {
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[2.0, 3.0], [4.0, 5.0]]));
        
        let sum = a.add(&b);
        assert_eq!(sum.data[[0, 0]], 3.0);
        
        let diff = b.sub(&a);
        assert_eq!(diff.data[[0, 0]], 1.0);
        
        let prod = a.mul(&b);
        assert_eq!(prod.data[[0, 0]], 2.0);
    }

    #[test]
    fn test_scale() {
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let scaled = a.scale(2.0);
        assert_eq!(scaled.data[[1, 1]], 8.0);
    }

    #[test]
    fn test_add_bias() {
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
        let bias = Array1::from_vec(vec![10.0, 20.0]);
        let result = a.add_bias(&bias);
        
        assert_eq!(result.data[[0, 0]], 11.0);
        assert_eq!(result.data[[0, 1]], 22.0);
        assert_eq!(result.data[[2, 0]], 15.0);
    }

    #[test]
    fn test_argmax() {
        let a = Tensor::new(arr2(&[[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]]));
        let argmax = a.argmax_axis1();
        assert_eq!(argmax, vec![1, 0]);
    }
}
