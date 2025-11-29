//! Titanic Survival Prediction
//!
//! Train a neural network to predict passenger survival on the Titanic.
//! This is a classic ML benchmark - we'll compare our results to official benchmarks.
//!
//! Official benchmark accuracy: ~77-82% (depending on feature engineering)

use mini_nn::{Network, Activation, Loss, Optimizer, Trainer, TrainingConfig, Tensor};
use mini_nn::data::{load_titanic, Dataset};
use mini_nn::loss::binary_accuracy;
use std::path::Path;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("         Mini-NN: Titanic Survival Prediction");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    
    // Try to load the Titanic dataset
    let data_path = "data/titanic.csv";
    
    let dataset = if Path::new(data_path).exists() {
        println!("📊 Loading Titanic dataset from {}...", data_path);
        match load_titanic(data_path) {
            Ok(d) => d,
            Err(e) => {
                println!("Error loading data: {}", e);
                println!("Using synthetic data instead...");
                create_synthetic_titanic()
            }
        }
    } else {
        println!("📊 Titanic CSV not found at {}", data_path);
        println!("   Creating synthetic Titanic-like dataset...");
        println!();
        println!("   To use real data, download from:");
        println!("   https://www.kaggle.com/c/titanic/data");
        println!("   Place train.csv as data/titanic.csv");
        println!();
        create_synthetic_titanic()
    };
    
    println!("   Samples: {}", dataset.n_samples);
    println!("   Features: {}", dataset.x.shape().1);
    println!();
    
    // Feature descriptions
    println!("📋 Features:");
    println!("   ┌─────────────┬────────────────────────────────────┐");
    println!("   │ Feature     │ Description                        │");
    println!("   ├─────────────┼────────────────────────────────────┤");
    println!("   │ Pclass      │ Ticket class (1st, 2nd, 3rd)       │");
    println!("   │ Sex         │ Gender (0=male, 1=female)          │");
    println!("   │ Age         │ Age in years                       │");
    println!("   │ SibSp       │ # siblings/spouses aboard          │");
    println!("   │ Parch       │ # parents/children aboard          │");
    println!("   │ Fare        │ Passenger fare                     │");
    println!("   │ Embarked_C  │ Embarked at Cherbourg              │");
    println!("   │ Embarked_Q  │ Embarked at Queenstown             │");
    println!("   │ Embarked_S  │ Embarked at Southampton            │");
    println!("   └─────────────┴────────────────────────────────────┘");
    println!();
    
    // Split data
    let (train_data, test_data) = dataset.train_test_split(0.2);
    println!("   Train samples: {}", train_data.n_samples);
    println!("   Test samples:  {}", test_data.n_samples);
    println!();
    
    // Create the network
    println!("🧠 Creating neural network...");
    let n_features = train_data.x.shape().1;
    
    let mut network = Network::new()
        .add_dense(n_features, 32)
        .add_activation(Activation::ReLU)
        .add_dense(32, 16)
        .add_activation(Activation::ReLU)
        .add_dense(16, 8)
        .add_activation(Activation::ReLU)
        .add_dense(8, 1)
        .add_activation(Activation::Sigmoid);
    
    network.summary();
    println!();
    
    // Training configuration
    let config = TrainingConfig {
        epochs: 100,
        batch_size: 32,
        validation_split: 0.15,
        shuffle: true,
        early_stopping_patience: 15,
        verbose: true,
    };
    
    // Train the network
    println!("🎯 Training network...");
    println!("───────────────────────────────────────────────────────────────");
    
    let trainer = Trainer::new(config);
    let history = trainer.fit(
        &mut network,
        &train_data.x,
        &train_data.y,
        Loss::BinaryCrossEntropy,
        Optimizer::adam(0.001),
    );
    
    println!("───────────────────────────────────────────────────────────────");
    println!();
    
    // Training results
    println!("📈 Training Results:");
    println!("   Final train loss: {:.4}", history.train_loss.last().unwrap_or(&0.0));
    println!("   Final train acc:  {:.1}%", history.train_accuracy.last().unwrap_or(&0.0) * 100.0);
    println!("   Best val loss:    {:.4}", history.best_val_loss().unwrap_or(0.0));
    println!("   Best val acc:     {:.1}%", history.best_val_accuracy().unwrap_or(0.0) * 100.0);
    println!();
    
    // Evaluate on test set
    println!("🔍 Evaluating on test set...");
    let predictions = network.predict(&test_data.x);
    let test_accuracy = binary_accuracy(&predictions, &test_data.y);
    
    println!("───────────────────────────────────────────────────────────────");
    println!("   Test Accuracy: {:.1}%", test_accuracy * 100.0);
    println!("───────────────────────────────────────────────────────────────");
    println!();
    
    // Confusion matrix
    let mut tp = 0;  // True Positive
    let mut tn = 0;  // True Negative
    let mut fp = 0;  // False Positive
    let mut fn_ = 0; // False Negative
    
    for i in 0..test_data.n_samples {
        let pred = if predictions.data[[i, 0]] >= 0.5 { 1.0 } else { 0.0 };
        let actual = test_data.y.data[[i, 0]];
        
        if pred == 1.0 && actual == 1.0 { tp += 1; }
        else if pred == 0.0 && actual == 0.0 { tn += 1; }
        else if pred == 1.0 && actual == 0.0 { fp += 1; }
        else { fn_ += 1; }
    }
    
    println!("📊 Confusion Matrix:");
    println!("                   Predicted");
    println!("                 │  Died  │ Survived │");
    println!("   ┌─────────────┼────────┼──────────┤");
    println!("   │ Actual Died │  {:4}  │   {:4}   │", tn, fp);
    println!("   │ Act. Surviv │  {:4}  │   {:4}   │", fn_, tp);
    println!("   └─────────────┴────────┴──────────┘");
    println!();
    
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    
    println!("📊 Metrics:");
    println!("   Precision: {:.1}%", precision * 100.0);
    println!("   Recall:    {:.1}%", recall * 100.0);
    println!("   F1 Score:  {:.1}%", f1 * 100.0);
    println!();
    
    // Compare to benchmarks
    println!("───────────────────────────────────────────────────────────────");
    println!("📊 Comparison to Official Benchmarks:");
    println!("───────────────────────────────────────────────────────────────");
    println!("   │ Method                    │ Accuracy │");
    println!("   ├───────────────────────────┼──────────┤");
    println!("   │ Logistic Regression       │  ~77%    │");
    println!("   │ Random Forest             │  ~78%    │");
    println!("   │ Gradient Boosting         │  ~80%    │");
    println!("   │ Neural Network (sklearn)  │  ~79%    │");
    println!("   │ Top Kaggle Submissions    │  ~83%    │");
    println!("   ├───────────────────────────┼──────────┤");
    println!("   │ Our Mini-NN               │  {:.1}%   │", test_accuracy * 100.0);
    println!("   └───────────────────────────┴──────────┘");
    println!();
    
    // Feature importance insights
    println!("💡 Key Insights from Titanic Data:");
    println!("   • Women had much higher survival rates (Sex is most important)");
    println!("   • 1st class passengers survived more often than 3rd class");
    println!("   • Children had higher survival rates");
    println!("   • \"Women and children first\" policy is evident in the data");
    println!();
    
    // Sample predictions
    println!("🔍 Sample Predictions:");
    println!("───────────────────────────────────────────────────────────────");
    
    let n_samples_to_show = 5.min(test_data.n_samples);
    for i in 0..n_samples_to_show {
        let pred_prob = predictions.data[[i, 0]];
        let pred_class = if pred_prob >= 0.5 { "Survived" } else { "Died" };
        let actual_class = if test_data.y.data[[i, 0]] >= 0.5 { "Survived" } else { "Died" };
        let correct = if pred_class == actual_class { "✓" } else { "✗" };
        
        println!("   Sample {}: Pred={:.2} ({:8}) | Actual: {:8} | {}", 
            i + 1, pred_prob, pred_class, actual_class, correct);
    }
    println!();
    
    println!("═══════════════════════════════════════════════════════════════");
}

/// Create a synthetic Titanic-like dataset for demo purposes
fn create_synthetic_titanic() -> Dataset {
    use ndarray::Array2;
    use rand::Rng;
    
    let n_samples = 891;  // Same as real Titanic dataset
    let n_features = 9;
    
    let mut rng = rand::thread_rng();
    let mut x_data = Array2::zeros((n_samples, n_features));
    let mut y_data = Array2::zeros((n_samples, 1));
    
    for i in 0..n_samples {
        // Generate features that roughly match Titanic distributions
        let pclass: f64 = rng.gen_range(0..3) as f64 / 2.0;  // 0, 0.5, or 1
        let sex: f64 = if rng.gen_bool(0.35) { 1.0 } else { 0.0 };  // 35% female
        let age: f64 = rng.gen_range(1..80) as f64 / 100.0;
        let sibsp: f64 = rng.gen_range(0..5) as f64 / 8.0;
        let parch: f64 = rng.gen_range(0..4) as f64 / 6.0;
        let fare: f64 = rng.gen_range(5..200) as f64 / 512.0;
        
        // Embarked (one-hot)
        let embarked = rng.gen_range(0..3);
        let embarked_c = if embarked == 0 { 1.0 } else { 0.0 };
        let embarked_q = if embarked == 1 { 1.0 } else { 0.0 };
        let embarked_s = if embarked == 2 { 1.0 } else { 0.0 };
        
        x_data[[i, 0]] = pclass;
        x_data[[i, 1]] = sex;
        x_data[[i, 2]] = age;
        x_data[[i, 3]] = sibsp;
        x_data[[i, 4]] = parch;
        x_data[[i, 5]] = fare;
        x_data[[i, 6]] = embarked_c;
        x_data[[i, 7]] = embarked_q;
        x_data[[i, 8]] = embarked_s;
        
        // Generate realistic survival based on features
        // Women, children, 1st class had higher survival
        let base_survival = 0.38;  // Overall survival rate was ~38%
        let sex_bonus = if sex == 1.0 { 0.4 } else { 0.0 };  // Women much more likely
        let class_bonus = (1.0 - pclass) * 0.2;  // 1st class bonus
        let age_bonus = if age < 0.15 { 0.15 } else { 0.0 };  // Children bonus
        
        let survival_prob = (base_survival + sex_bonus + class_bonus + age_bonus)
            .min(0.95).max(0.05);
        
        y_data[[i, 0]] = if rng.gen_bool(survival_prob) { 1.0 } else { 0.0 };
    }
    
    Dataset::new(Tensor::new(x_data), Tensor::new(y_data))
}
