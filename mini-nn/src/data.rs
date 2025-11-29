//! Data loading and preprocessing
//!
//! This module provides utilities for loading datasets (especially CSV),
//! preprocessing, normalization, and batching.

use crate::tensor::Tensor;
use ndarray::{Array2, Array1};
use csv::ReaderBuilder;
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;

/// Dataset container
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature matrix (n_samples, n_features)
    pub x: Tensor,
    /// Target matrix (n_samples, n_outputs)
    pub y: Tensor,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Number of samples
    pub n_samples: usize,
}

impl Dataset {
    pub fn new(x: Tensor, y: Tensor) -> Self {
        let n_samples = x.shape().0;
        Self {
            x,
            y,
            feature_names: Vec::new(),
            n_samples,
        }
    }

    /// Split into train and test sets
    pub fn train_test_split(&self, test_ratio: f64) -> (Dataset, Dataset) {
        let n_test = (self.n_samples as f64 * test_ratio) as usize;
        let n_train = self.n_samples - n_test;
        
        let x_train = self.x.slice_rows(0, n_train);
        let y_train = self.y.slice_rows(0, n_train);
        let x_test = self.x.slice_rows(n_train, self.n_samples);
        let y_test = self.y.slice_rows(n_train, self.n_samples);
        
        let train = Dataset {
            x: x_train,
            y: y_train,
            feature_names: self.feature_names.clone(),
            n_samples: n_train,
        };
        
        let test = Dataset {
            x: x_test,
            y: y_test,
            feature_names: self.feature_names.clone(),
            n_samples: n_test,
        };
        
        (train, test)
    }
}

/// Data loader for batching
pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    current_idx: usize,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            current_idx: 0,
        }
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.dataset.n_samples + self.batch_size - 1) / self.batch_size
    }

    /// Reset and optionally shuffle
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            crate::tensor::shuffle_together(&mut self.dataset.x, &mut self.dataset.y);
        }
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.n_samples {
            return None;
        }
        
        let end = (self.current_idx + self.batch_size).min(self.dataset.n_samples);
        let x_batch = self.dataset.x.slice_rows(self.current_idx, end);
        let y_batch = self.dataset.y.slice_rows(self.current_idx, end);
        
        self.current_idx = end;
        Some((x_batch, y_batch))
    }
}

// ============================================================================
// CSV Loading
// ============================================================================

/// Load a CSV file into a 2D array
pub fn load_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<(Array2<String>, Vec<String>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(has_header)
        .from_reader(file);
    
    let headers: Vec<String> = if has_header {
        reader.headers()?.iter().map(|s| s.to_string()).collect()
    } else {
        Vec::new()
    };
    
    let records: Vec<Vec<String>> = reader
        .records()
        .filter_map(|r| r.ok())
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect();
    
    if records.is_empty() {
        return Err("Empty CSV file".into());
    }
    
    let n_rows = records.len();
    let n_cols = records[0].len();
    
    let mut data = Array2::from_elem((n_rows, n_cols), String::new());
    for (i, row) in records.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            data[[i, j]] = val.clone();
        }
    }
    
    Ok((data, headers))
}

// ============================================================================
// Titanic Dataset
// ============================================================================

/// Load and preprocess the Titanic dataset
/// Returns (x, y, feature_names)
pub fn load_titanic<P: AsRef<Path>>(path: P) -> Result<Dataset, Box<dyn std::error::Error>> {
    let (data, headers) = load_csv(path, true)?;
    let n_samples = data.nrows();
    
    // Find column indices
    let col_indices: HashMap<String, usize> = headers.iter()
        .enumerate()
        .map(|(i, h)| (h.to_lowercase(), i))
        .collect();
    
    // Parse features
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();
    let feature_names = vec![
        "pclass".to_string(),
        "sex".to_string(),
        "age".to_string(),
        "sibsp".to_string(),
        "parch".to_string(),
        "fare".to_string(),
        "embarked_c".to_string(),
        "embarked_q".to_string(),
        "embarked_s".to_string(),
    ];
    
    for row_idx in 0..n_samples {
        // Get target (Survived)
        let survived_idx = *col_indices.get("survived").unwrap_or(&1);
        let survived = data[[row_idx, survived_idx]].parse::<f64>().unwrap_or(0.0);
        targets.push(survived);
        
        // Get features
        let mut row_features = Vec::new();
        
        // Pclass (1, 2, 3) - normalize to 0-1
        let pclass_idx = *col_indices.get("pclass").unwrap_or(&2);
        let pclass = data[[row_idx, pclass_idx]].parse::<f64>().unwrap_or(3.0);
        row_features.push((pclass - 1.0) / 2.0);
        
        // Sex (male=0, female=1)
        let sex_idx = *col_indices.get("sex").unwrap_or(&4);
        let sex = if data[[row_idx, sex_idx]].to_lowercase() == "female" { 1.0 } else { 0.0 };
        row_features.push(sex);
        
        // Age (normalize, handle missing)
        let age_idx = *col_indices.get("age").unwrap_or(&5);
        let age = data[[row_idx, age_idx]].parse::<f64>().unwrap_or(30.0);
        row_features.push(age / 100.0);  // Normalize to ~0-1
        
        // SibSp (siblings/spouses)
        let sibsp_idx = *col_indices.get("sibsp").unwrap_or(&6);
        let sibsp = data[[row_idx, sibsp_idx]].parse::<f64>().unwrap_or(0.0);
        row_features.push(sibsp / 8.0);  // Max is typically ~8
        
        // Parch (parents/children)
        let parch_idx = *col_indices.get("parch").unwrap_or(&7);
        let parch = data[[row_idx, parch_idx]].parse::<f64>().unwrap_or(0.0);
        row_features.push(parch / 6.0);  // Max is typically ~6
        
        // Fare (normalize)
        let fare_idx = *col_indices.get("fare").unwrap_or(&9);
        let fare = data[[row_idx, fare_idx]].parse::<f64>().unwrap_or(32.0);
        row_features.push(fare / 512.0);  // Max is ~512
        
        // Embarked (one-hot: C, Q, S)
        let embarked_idx = *col_indices.get("embarked").unwrap_or(&11);
        let embarked = data[[row_idx, embarked_idx]].to_uppercase();
        row_features.push(if embarked == "C" { 1.0 } else { 0.0 });
        row_features.push(if embarked == "Q" { 1.0 } else { 0.0 });
        row_features.push(if embarked == "S" { 1.0 } else { 0.0 });
        
        features.push(row_features);
    }
    
    // Convert to tensors
    let n_features = features[0].len();
    let mut x_data = Array2::zeros((n_samples, n_features));
    let mut y_data = Array2::zeros((n_samples, 1));
    
    for (i, (row, target)) in features.iter().zip(targets.iter()).enumerate() {
        for (j, val) in row.iter().enumerate() {
            x_data[[i, j]] = *val;
        }
        y_data[[i, 0]] = *target;
    }
    
    Ok(Dataset {
        x: Tensor::new(x_data),
        y: Tensor::new(y_data),
        feature_names,
        n_samples,
    })
}

// ============================================================================
// Normalization
// ============================================================================

/// Normalize features to have zero mean and unit variance
pub fn standardize(x: &Tensor) -> (Tensor, Array1<f64>, Array1<f64>) {
    let mean = x.mean_axis(0);
    let n_samples = x.shape().0;
    
    // Compute std
    let mut variance = Array1::zeros(x.shape().1);
    for row in x.data.rows() {
        let diff = &row - &mean;
        variance += &(&diff * &diff);
    }
    variance /= n_samples as f64;
    let std: Array1<f64> = variance.mapv(|v| (v + 1e-8).sqrt());
    
    // Normalize
    let mut normalized = x.data.clone();
    for mut row in normalized.rows_mut() {
        row -= &mean;
        row /= &std;
    }
    
    (Tensor::new(normalized), mean, std)
}

/// Min-max normalization to [0, 1]
pub fn min_max_normalize(x: &Tensor) -> (Tensor, Array1<f64>, Array1<f64>) {
    let n_cols = x.shape().1;
    
    let mut min_vals = Array1::from_elem(n_cols, f64::MAX);
    let mut max_vals = Array1::from_elem(n_cols, f64::MIN);
    
    for row in x.data.rows() {
        for (j, &val) in row.iter().enumerate() {
            min_vals[j] = min_vals[j].min(val);
            max_vals[j] = max_vals[j].max(val);
        }
    }
    
    let range: Array1<f64> = &max_vals - &min_vals;
    let range: Array1<f64> = range.mapv(|v| if v.abs() < 1e-8 { 1.0 } else { v });
    
    let mut normalized = x.data.clone();
    for mut row in normalized.rows_mut() {
        row -= &min_vals;
        row /= &range;
    }
    
    (Tensor::new(normalized), min_vals, max_vals)
}

// ============================================================================
// Data Generation
// ============================================================================

/// Generate XOR dataset
pub fn generate_xor(n_samples: usize) -> Dataset {
    let samples_per_class = n_samples / 4;
    let total = samples_per_class * 4;
    
    let mut x_data = Array2::zeros((total, 2));
    let mut y_data = Array2::zeros((total, 1));
    
    let xor_patterns = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)];
    
    for (i, &(x1, x2, y)) in xor_patterns.iter().enumerate() {
        for j in 0..samples_per_class {
            let idx = i * samples_per_class + j;
            x_data[[idx, 0]] = x1 + (rand::random::<f64>() - 0.5) * 0.1;
            x_data[[idx, 1]] = x2 + (rand::random::<f64>() - 0.5) * 0.1;
            y_data[[idx, 0]] = y;
        }
    }
    
    Dataset::new(Tensor::new(x_data), Tensor::new(y_data))
}

/// Generate spiral dataset for classification
pub fn generate_spiral(n_samples: usize, n_classes: usize) -> Dataset {
    let samples_per_class = n_samples / n_classes;
    let total = samples_per_class * n_classes;
    
    let mut x_data = Array2::zeros((total, 2));
    let mut y_data = Array2::zeros((total, n_classes));
    
    for c in 0..n_classes {
        for i in 0..samples_per_class {
            let idx = c * samples_per_class + i;
            let r = i as f64 / samples_per_class as f64;
            let theta = 4.0 * r * std::f64::consts::PI + (c as f64 * 2.0 * std::f64::consts::PI / n_classes as f64);
            
            x_data[[idx, 0]] = r * theta.sin() + (rand::random::<f64>() - 0.5) * 0.2;
            x_data[[idx, 1]] = r * theta.cos() + (rand::random::<f64>() - 0.5) * 0.2;
            y_data[[idx, c]] = 1.0;  // One-hot encoding
        }
    }
    
    Dataset::new(Tensor::new(x_data), Tensor::new(y_data))
}

/// Generate sine wave for regression
pub fn generate_sine(n_samples: usize) -> Dataset {
    let mut x_data = Array2::zeros((n_samples, 1));
    let mut y_data = Array2::zeros((n_samples, 1));
    
    for i in 0..n_samples {
        let x = (i as f64 / n_samples as f64) * 4.0 * std::f64::consts::PI;
        x_data[[i, 0]] = x / (4.0 * std::f64::consts::PI);  // Normalize to [0, 1]
        y_data[[i, 0]] = (x.sin() + 1.0) / 2.0;  // Normalize to [0, 1]
    }
    
    Dataset::new(Tensor::new(x_data), Tensor::new(y_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_split() {
        let x = Tensor::random_normal(100, 5, 0.0, 1.0);
        let y = Tensor::random_normal(100, 1, 0.0, 1.0);
        let dataset = Dataset::new(x, y);
        
        let (train, test) = dataset.train_test_split(0.2);
        
        assert_eq!(train.n_samples, 80);
        assert_eq!(test.n_samples, 20);
    }

    #[test]
    fn test_dataloader() {
        let x = Tensor::random_normal(100, 5, 0.0, 1.0);
        let y = Tensor::random_normal(100, 1, 0.0, 1.0);
        let dataset = Dataset::new(x, y);
        
        let mut loader = DataLoader::new(dataset, 32, false);
        
        assert_eq!(loader.num_batches(), 4);  // ceil(100/32) = 4
        
        let mut count = 0;
        for (x_batch, y_batch) in &mut loader {
            count += x_batch.shape().0;
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn test_generate_xor() {
        let dataset = generate_xor(100);
        assert_eq!(dataset.n_samples, 100);
        assert_eq!(dataset.x.shape().1, 2);
        assert_eq!(dataset.y.shape().1, 1);
    }

    #[test]
    fn test_generate_spiral() {
        let dataset = generate_spiral(300, 3);
        assert_eq!(dataset.n_samples, 300);
        assert_eq!(dataset.x.shape().1, 2);
        assert_eq!(dataset.y.shape().1, 3);  // One-hot
    }

    #[test]
    fn test_standardize() {
        let x = Tensor::random_normal(100, 5, 10.0, 5.0);
        let (normalized, _mean, _std) = standardize(&x);
        
        // Mean should be close to 0
        let new_mean = normalized.mean_axis(0);
        for &m in new_mean.iter() {
            assert!(m.abs() < 0.5);  // Should be approximately 0
        }
    }

    #[test]
    fn test_min_max_normalize() {
        let x = Tensor::random_normal(100, 5, 10.0, 5.0);
        let (normalized, _min, _max) = min_max_normalize(&x);
        
        // All values should be in [0, 1] (approximately, due to floating point)
        for &val in normalized.data.iter() {
            assert!(val >= -0.01 && val <= 1.01);
        }
    }
}
