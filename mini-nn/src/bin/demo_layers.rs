//! Demo: Layers and Forward Propagation
//!
//! This demo shows how data flows through neural network layers,
//! illustrating the matrix multiplication and activation concepts.

use mini_nn::tensor::Tensor;
use mini_nn::layer::{Dense, Layer, Initializer};
use mini_nn::activation::{Activation, apply_activation};
use ndarray::arr2;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("        Mini-NN: Layers and Forward Propagation Demo");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    
    // =========================================================================
    // Part 1: Matrix-Vector Multiplication (Like the Clojure article)
    // =========================================================================
    
    println!("📐 Part 1: Matrix-Vector Multiplication");
    println!("───────────────────────────────────────────────────────────");
    println!();
    
    println!("The core operation in neural networks is matrix multiplication.");
    println!("Each layer transforms input x using weights W:");
    println!();
    println!("   h = W × x");
    println!();
    
    // Example from the article
    let x = Tensor::new(arr2(&[[0.3, 0.9]]));  // Input as row vector
    let w1 = Tensor::new(arr2(&[
        [0.3, 0.6, 0.1, 2.0],    // Transposed from article for row-major
        [0.9, 3.7, 0.0, 1.0],
    ]));
    
    println!("Input x = [0.3, 0.9]");
    println!();
    println!("Weight matrix W1 (2×4):");
    println!("   ┌                      ┐");
    println!("   │  0.3   0.6   0.1   2.0 │");
    println!("   │  0.9   3.7   0.0   1.0 │");
    println!("   └                      ┘");
    println!();
    
    let h1 = x.matmul(&w1);
    println!("h1 = x × W1 = [");
    for val in h1.data.row(0) {
        print!("  {:.2}", val);
    }
    println!(" ]");
    println!();
    
    // Second layer
    let w2 = Tensor::new(arr2(&[
        [0.75],
        [0.15],
        [0.22],
        [0.33],
    ]));
    
    println!("Weight matrix W2 (4×1):");
    println!("   ┌      ┐");
    println!("   │ 0.75 │");
    println!("   │ 0.15 │");
    println!("   │ 0.22 │");
    println!("   │ 0.33 │");
    println!("   └      ┘");
    println!();
    
    let y = h1.matmul(&w2);
    println!("y = h1 × W2 = [{:.2}]", y.data[[0, 0]]);
    println!();
    
    // =========================================================================
    // Part 2: Dense Layer Abstraction
    // =========================================================================
    
    println!("───────────────────────────────────────────────────────────");
    println!("📦 Part 2: Dense Layer Abstraction");
    println!("───────────────────────────────────────────────────────────");
    println!();
    
    println!("The Dense layer encapsulates: output = input @ weights + bias");
    println!();
    
    let mut dense = Dense::with_initializer(3, 4, Initializer::Xavier);
    println!("Created Dense layer: 3 → 4");
    println!("Parameters: {} ({} weights + {} biases)", 
        dense.num_parameters(),
        3 * 4,
        4
    );
    println!();
    
    let input = Tensor::new(arr2(&[[1.0, 2.0, 3.0]]));
    println!("Input: [1.0, 2.0, 3.0]");
    
    let output = dense.forward(&input);
    println!("Output shape: {:?}", output.shape());
    print!("Output: [");
    for (i, val) in output.data.row(0).iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.4}", val);
    }
    println!("]");
    println!();
    
    // =========================================================================
    // Part 3: Activation Functions
    // =========================================================================
    
    println!("───────────────────────────────────────────────────────────");
    println!("⚡ Part 3: Activation Functions");
    println!("───────────────────────────────────────────────────────────");
    println!();
    
    println!("Without non-linear activations, stacking layers would just");
    println!("be equivalent to a single linear transformation!");
    println!();
    
    let z = Tensor::new(arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]));
    println!("Input z: [-2.0, -1.0, 0.0, 1.0, 2.0]");
    println!();
    
    // ReLU
    let relu_out = apply_activation(&Activation::ReLU, &z);
    print!("ReLU(z):    [");
    for (i, val) in relu_out.data.row(0).iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.2}", val);
    }
    println!("]");
    println!("            max(0, x) - zeros out negatives");
    println!();
    
    // Sigmoid
    let sigmoid_out = apply_activation(&Activation::Sigmoid, &z);
    print!("Sigmoid(z): [");
    for (i, val) in sigmoid_out.data.row(0).iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.2}", val);
    }
    println!("]");
    println!("            1/(1+exp(-x)) - squashes to (0,1)");
    println!();
    
    // Tanh
    let tanh_out = apply_activation(&Activation::Tanh, &z);
    print!("Tanh(z):    [");
    for (i, val) in tanh_out.data.row(0).iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.2}", val);
    }
    println!("]");
    println!("            (exp(x)-exp(-x))/(exp(x)+exp(-x)) - squashes to (-1,1)");
    println!();
    
    // Softmax
    let softmax_out = apply_activation(&Activation::Softmax, &z);
    print!("Softmax(z): [");
    for (i, val) in softmax_out.data.row(0).iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{:.2}", val);
    }
    println!("]");
    println!("            exp(x)/sum(exp(x)) - probability distribution");
    println!("            Sum = {:.4}", softmax_out.sum());
    println!();
    
    // =========================================================================
    // Part 4: Building a Full Network
    // =========================================================================
    
    println!("───────────────────────────────────────────────────────────");
    println!("🧠 Part 4: Full Network Forward Pass");
    println!("───────────────────────────────────────────────────────────");
    println!();
    
    println!("Architecture: 2 → Dense(4) → ReLU → Dense(2) → Softmax");
    println!();
    
    // Build manually to show each step
    let mut layer1 = Dense::with_initializer(2, 4, Initializer::Xavier);
    let mut layer2 = Dense::with_initializer(4, 2, Initializer::Xavier);
    
    let input = Tensor::new(arr2(&[[0.5, 0.8], [0.1, 0.9]]));
    println!("Input (2 samples, 2 features):");
    for (i, row) in input.data.rows().into_iter().enumerate() {
        println!("   Sample {}: [{:.1}, {:.1}]", i + 1, row[0], row[1]);
    }
    println!();
    
    // Forward through layer 1
    let z1 = layer1.forward(&input);
    println!("After Dense(4):");
    println!("   Shape: {:?}", z1.shape());
    
    // Apply ReLU
    let a1 = apply_activation(&Activation::ReLU, &z1);
    println!("After ReLU:");
    for (i, row) in a1.data.rows().into_iter().enumerate() {
        print!("   Sample {}: [", i + 1);
        for (j, val) in row.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:.3}", val);
        }
        println!("]");
    }
    println!();
    
    // Forward through layer 2
    let z2 = layer2.forward(&a1);
    println!("After Dense(2):");
    println!("   Shape: {:?}", z2.shape());
    
    // Apply Softmax
    let output = apply_activation(&Activation::Softmax, &z2);
    println!("After Softmax (final output):");
    for (i, row) in output.data.rows().into_iter().enumerate() {
        println!("   Sample {}: [{:.4}, {:.4}] (sum={:.4})", 
            i + 1, row[0], row[1], row[0] + row[1]);
    }
    println!();
    
    println!("These outputs are probability distributions over 2 classes!");
    println!();
    
    // =========================================================================
    // Summary
    // =========================================================================
    
    println!("───────────────────────────────────────────────────────────");
    println!("📚 Key Takeaways");
    println!("───────────────────────────────────────────────────────────");
    println!();
    println!("1. Neural networks are just matrix multiplications + non-linearities");
    println!("2. Each Dense layer: output = input @ weights + bias");
    println!("3. Activations introduce non-linearity (essential for learning!)");
    println!("4. ReLU is popular for hidden layers (fast, avoids vanishing gradients)");
    println!("5. Sigmoid/Softmax are used for outputs (probabilities)");
    println!();
    println!("═══════════════════════════════════════════════════════════");
}
