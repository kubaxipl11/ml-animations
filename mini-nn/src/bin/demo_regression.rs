//! Demo: Regression with Sine Wave
//!
//! This demo shows how neural networks can approximate continuous functions.
//! We train a network to learn the sine function.

use mini_nn::{Network, Activation, Loss, Optimizer};
use mini_nn::data::generate_sine;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("          Mini-NN: Function Approximation Demo");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    
    // Generate sine wave data
    println!("📊 Generating sine wave dataset...");
    let dataset = generate_sine(500);
    println!("   Samples: {}", dataset.n_samples);
    println!("   Learning to approximate: y = sin(x)");
    println!();
    
    // Create network for regression
    // Using Tanh in hidden layers (works well for bounded outputs)
    println!("🧠 Creating regression network...");
    let mut network = Network::new()
        .add_dense(1, 32)
        .add_activation(Activation::Tanh)
        .add_dense(32, 32)
        .add_activation(Activation::Tanh)
        .add_dense(32, 1)
        .add_activation(Activation::Sigmoid);  // Output in [0, 1]
    
    network.summary();
    println!();
    
    // Custom training loop to show progress
    println!("🎯 Training network to approximate sin(x)...");
    println!("───────────────────────────────────────────────────────────");
    
    network.compile(Loss::MSE, Optimizer::adam(0.01));
    
    let mut best_loss = f64::MAX;
    let n_samples = dataset.x.shape().0;
    let batch_size = 32;
    let n_batches = (n_samples + batch_size - 1) / batch_size;
    
    for epoch in 0..200 {
        let mut epoch_loss = 0.0;
        
        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n_samples);
            
            let x_batch = dataset.x.slice_rows(start, end);
            let y_batch = dataset.y.slice_rows(start, end);
            
            let (loss, _) = network.train_step(&x_batch, &y_batch);
            epoch_loss += loss;
        }
        
        let avg_loss = epoch_loss / n_batches as f64;
        if avg_loss < best_loss {
            best_loss = avg_loss;
        }
        
        // Print every 20 epochs
        if (epoch + 1) % 20 == 0 {
            println!("   Epoch {:3}: MSE Loss = {:.6}", epoch + 1, avg_loss);
        }
    }
    
    println!("───────────────────────────────────────────────────────────");
    println!("   Best MSE Loss: {:.6}", best_loss);
    println!();
    
    // Test predictions
    println!("🔍 Testing predictions:");
    println!("───────────────────────────────────────────────────────────");
    println!("   │    x    │  sin(x)  │ Predicted │  Error  │");
    println!("   ├─────────┼──────────┼───────────┼─────────┤");
    
    use ndarray::arr2;
    use mini_nn::Tensor;
    
    let test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    let mut total_error = 0.0;
    
    for &x in &test_points {
        let actual_sin = ((x * 4.0 * std::f64::consts::PI).sin() + 1.0) / 2.0;
        let input = Tensor::new(arr2(&[[x]]));
        let pred = network.predict(&input);
        let pred_val = pred.data[[0, 0]];
        let error = (pred_val - actual_sin).abs();
        total_error += error;
        
        println!("   │  {:.2}   │  {:.4}   │   {:.4}   │ {:.4}  │", 
            x, actual_sin, pred_val, error);
    }
    
    println!("   └─────────┴──────────┴───────────┴─────────┘");
    println!("   Average Error: {:.4}", total_error / test_points.len() as f64);
    println!();
    
    // ASCII plot
    println!("📈 ASCII Plot (actual vs predicted):");
    println!("───────────────────────────────────────────────────────────");
    
    let width = 50;
    let height = 15;
    let mut plot: Vec<Vec<char>> = vec![vec![' '; width]; height];
    
    // Plot actual sine wave (dots)
    for i in 0..width {
        let x = i as f64 / width as f64;
        let y = ((x * 4.0 * std::f64::consts::PI).sin() + 1.0) / 2.0;
        let row = ((1.0 - y) * (height - 1) as f64) as usize;
        let row = row.min(height - 1);
        plot[row][i] = '·';
    }
    
    // Plot predictions (stars)
    for i in 0..width {
        let x = i as f64 / width as f64;
        let input = Tensor::new(arr2(&[[x]]));
        let pred = network.predict(&input);
        let y = pred.data[[0, 0]].clamp(0.0, 1.0);
        let row = ((1.0 - y) * (height - 1) as f64) as usize;
        let row = row.min(height - 1);
        if plot[row][i] == '·' {
            plot[row][i] = '*';  // Overlap
        } else {
            plot[row][i] = '+';
        }
    }
    
    println!("   1.0 ┤");
    for (i, row) in plot.iter().enumerate() {
        let label = if i == 0 { "    │" } 
            else if i == height / 2 { "0.5 ┤" }
            else if i == height - 1 { "0.0 ┤" }
            else { "    │" };
        print!("   {}", label);
        for &c in row {
            print!("{}", c);
        }
        println!();
    }
    let line = "─".repeat(width);
    println!("       └{}", line);
    println!("        0                    x →                    1");
    println!();
    println!("   Legend: · = actual sin(x), + = predicted, * = overlap");
    println!();
    
    // Explanation
    println!("💡 Key Insights:");
    println!("   • Neural networks are universal function approximators");
    println!("   • With enough hidden units, they can learn any continuous function");
    println!("   • MSE loss is appropriate for regression tasks");
    println!("   • Tanh activation works well for bounded outputs");
    println!();
    println!("═══════════════════════════════════════════════════════════");
}

// Helper trait for string repeat (Rust doesn't have this built-in for char)
trait RepeatStr {
    fn repeat(&self, n: usize) -> String;
}

impl RepeatStr for char {
    fn repeat(&self, n: usize) -> String {
        std::iter::repeat(*self).take(n).collect()
    }
}
