//! Demo: XOR Problem
//!
//! The XOR (exclusive or) problem is a classic test for neural networks.
//! It's non-linearly separable, so a single-layer perceptron cannot solve it.
//! A network with at least one hidden layer is required.

use mini_nn::{Network, Activation, Loss, Optimizer, Trainer, TrainingConfig};
use mini_nn::data::generate_xor;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("            Mini-NN: XOR Problem Demo");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    
    // Generate XOR dataset
    println!("📊 Generating XOR dataset...");
    let dataset = generate_xor(1000);
    println!("   Samples: {}", dataset.n_samples);
    println!("   Features: {}", dataset.x.shape().1);
    println!();
    
    // Show the XOR truth table
    println!("📋 XOR Truth Table:");
    println!("   ┌─────┬─────┬─────────┐");
    println!("   │  A  │  B  │ A XOR B │");
    println!("   ├─────┼─────┼─────────┤");
    println!("   │  0  │  0  │    0    │");
    println!("   │  0  │  1  │    1    │");
    println!("   │  1  │  0  │    1    │");
    println!("   │  1  │  1  │    0    │");
    println!("   └─────┴─────┴─────────┘");
    println!();
    
    // Create the network
    // Architecture: 2 → 8 → 8 → 1
    println!("🧠 Creating network...");
    let mut network = Network::new()
        .add_dense(2, 8)
        .add_activation(Activation::ReLU)
        .add_dense(8, 8)
        .add_activation(Activation::ReLU)
        .add_dense(8, 1)
        .add_activation(Activation::Sigmoid);
    
    network.summary();
    println!();
    
    // Training configuration
    let config = TrainingConfig {
        epochs: 100,
        batch_size: 32,
        validation_split: 0.2,
        shuffle: true,
        early_stopping_patience: 20,
        verbose: true,
    };
    
    // Train the network
    println!("🎯 Training network...");
    println!("───────────────────────────────────────────────────────────");
    
    let trainer = Trainer::new(config);
    let history = trainer.fit(
        &mut network,
        &dataset.x,
        &dataset.y,
        Loss::BinaryCrossEntropy,
        Optimizer::adam(0.01),
    );
    
    println!("───────────────────────────────────────────────────────────");
    println!();
    
    // Results
    println!("📈 Training Results:");
    println!("   Final train loss: {:.4}", history.train_loss.last().unwrap_or(&0.0));
    println!("   Final train acc:  {:.1}%", history.train_accuracy.last().unwrap_or(&0.0) * 100.0);
    println!("   Best val loss:    {:.4}", history.best_val_loss().unwrap_or(0.0));
    println!("   Best val acc:     {:.1}%", history.best_val_accuracy().unwrap_or(0.0) * 100.0);
    println!();
    
    // Test on the 4 XOR cases
    println!("🔍 Testing on XOR truth table:");
    println!("───────────────────────────────────────────────────────────");
    
    use ndarray::arr2;
    use mini_nn::Tensor;
    
    let test_cases: [([f64; 2], f64); 4] = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];
    
    let mut correct = 0;
    for (inputs, expected) in &test_cases {
        let x = Tensor::new(arr2(&[[inputs[0], inputs[1]]]));
        let pred = network.predict(&x);
        let pred_val = pred.data[[0, 0]];
        let pred_class: f64 = if pred_val >= 0.5 { 1.0 } else { 0.0 };
        let is_correct = (pred_class - expected).abs() < 0.5;
        
        println!(
            "   Input: [{:.0}, {:.0}] → Pred: {:.4} ({:.0}) | Expected: {:.0} | {}",
            inputs[0], inputs[1],
            pred_val, pred_class,
            expected,
            if is_correct { "✓" } else { "✗" }
        );
        
        if is_correct {
            correct += 1;
        }
    }
    
    println!("───────────────────────────────────────────────────────────");
    println!("   Accuracy: {}/4 ({:.0}%)", correct, correct as f64 * 25.0);
    println!();
    
    // Explanation
    println!("💡 Why XOR Matters:");
    println!("   XOR is the simplest problem that requires a hidden layer.");
    println!("   A single perceptron can only create linear decision boundaries,");
    println!("   but XOR needs a non-linear boundary. Our network with ReLU");
    println!("   activations learns this non-linear mapping successfully!");
    println!();
    println!("═══════════════════════════════════════════════════════════");
}
