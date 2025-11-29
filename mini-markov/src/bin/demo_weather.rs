//! Weather Simulation Demo
//! 
//! Demonstrates Markov chains for modeling state transitions with weather patterns.

use mini_markov::StateChain;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       Mini Markov - Weather State Machine Demo                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Use fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // ══════════════════════════════════════════════════════════════════════════
    // 1. Simple Weather Model
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("1. SIMPLE WEATHER MODEL (3 States)");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    println!("States: Sunny ☀️, Cloudy ☁️, Rainy 🌧️");
    println!("\nTransition probabilities (from historical data):");
    println!("  Sunny  → Sunny: 70%  Cloudy: 20%  Rainy: 10%");
    println!("  Cloudy → Sunny: 30%  Cloudy: 40%  Rainy: 30%");
    println!("  Rainy  → Sunny: 20%  Cloudy: 40%  Rainy: 40%");
    println!();

    let mut weather = StateChain::first_order()
        .with_states(&["sunny", "cloudy", "rainy"]);
    
    // Build from probabilities (multiply by 10 to get counts)
    weather.add_transition_count("sunny", "sunny", 70);
    weather.add_transition_count("sunny", "cloudy", 20);
    weather.add_transition_count("sunny", "rainy", 10);
    
    weather.add_transition_count("cloudy", "sunny", 30);
    weather.add_transition_count("cloudy", "cloudy", 40);
    weather.add_transition_count("cloudy", "rainy", 30);
    
    weather.add_transition_count("rainy", "sunny", 20);
    weather.add_transition_count("rainy", "cloudy", 40);
    weather.add_transition_count("rainy", "rainy", 40);
    
    // Verify probabilities
    println!("Verification - Computed probabilities:");
    for state in &["sunny", "cloudy", "rainy"] {
        let probs = weather.probabilities_from(state);
        print!("  {} →", state);
        for (next, prob) in &probs {
            print!("  {}: {:.0}%", next, prob * 100.0);
        }
        println!();
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 2. Simulate Weather for a Week
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("2. WEEKLY WEATHER SIMULATION");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    let days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    
    for starting_weather in &["sunny", "cloudy", "rainy"] {
        println!("Starting: {} on Monday", starting_weather);
        let forecast = weather.simulate(starting_weather, 6, &mut rng);
        
        print!("  ");
        for (i, state) in forecast.iter().enumerate() {
            let emoji = match state.as_str() {
                "sunny" => "☀️",
                "cloudy" => "☁️",
                "rainy" => "🌧️",
                _ => "❓",
            };
            print!("{}: {} ", days[i], emoji);
        }
        println!("\n");
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3. Long-term Distribution (Stationary Distribution)
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("3. STATIONARY DISTRIBUTION (Long-term Probabilities)");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    let stationary = weather.stationary_distribution(1000);
    
    println!("After infinite time, regardless of starting state:");
    for (state, prob) in &stationary {
        let emoji = match state.as_str() {
            "sunny" => "☀️",
            "cloudy" => "☁️",
            "rainy" => "🌧️",
            _ => "❓",
        };
        let bar_len = (prob * 50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {} {} {:.1}% {}", emoji, state.chars().next().unwrap().to_uppercase().collect::<String>() + &state[1..], prob * 100.0, bar);
    }
    
    // Verify by simulation
    println!("\nVerification by simulation (10,000 steps from 'sunny'):");
    let long_sim = weather.simulate("sunny", 10000, &mut rng);
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for state in &long_sim {
        *counts.entry(state.as_str()).or_insert(0) += 1;
    }
    let total = long_sim.len() as f64;
    for state in &["sunny", "cloudy", "rainy"] {
        let count = counts.get(state).unwrap_or(&0);
        println!("  {}: {:.1}%", state, (*count as f64 / total) * 100.0);
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 4. Extended Weather Model with Seasons
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("4. EXTENDED MODEL: Weather + Temperature");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    // Now let's model temperature as a separate chain
    let mut temp = StateChain::first_order()
        .with_states(&["cold", "mild", "warm", "hot"]);
    
    // Summer-like transitions (tendency towards warm/hot)
    temp.add_transition_count("cold", "cold", 20);
    temp.add_transition_count("cold", "mild", 50);
    temp.add_transition_count("cold", "warm", 25);
    temp.add_transition_count("cold", "hot", 5);
    
    temp.add_transition_count("mild", "cold", 15);
    temp.add_transition_count("mild", "mild", 40);
    temp.add_transition_count("mild", "warm", 35);
    temp.add_transition_count("mild", "hot", 10);
    
    temp.add_transition_count("warm", "cold", 5);
    temp.add_transition_count("warm", "mild", 25);
    temp.add_transition_count("warm", "warm", 45);
    temp.add_transition_count("warm", "hot", 25);
    
    temp.add_transition_count("hot", "cold", 2);
    temp.add_transition_count("hot", "mild", 10);
    temp.add_transition_count("hot", "warm", 38);
    temp.add_transition_count("hot", "hot", 50);
    
    println!("Combined Weather + Temperature Forecast:\n");
    
    let weather_forecast = weather.simulate("sunny", 6, &mut rng);
    let temp_forecast = temp.simulate("mild", 6, &mut rng);
    
    println!("  Day   | Weather | Temp  ");
    println!("  ------+---------+-------");
    for (i, (w, t)) in weather_forecast.iter().zip(temp_forecast.iter()).enumerate() {
        let w_emoji = match w.as_str() {
            "sunny" => "☀️ ",
            "cloudy" => "☁️ ",
            "rainy" => "🌧️ ",
            _ => "❓ ",
        };
        let t_emoji = match t.as_str() {
            "cold" => "🥶",
            "mild" => "😊",
            "warm" => "😓",
            "hot" => "🥵",
            _ => "❓",
        };
        println!("  {} |  {} |  {}", days[i], w_emoji, t_emoji);
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 5. Expected Time to Reach State
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("5. EXPECTED HITTING TIMES");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    println!("Average days until sunny (from 1000 simulations):");
    
    for start in &["cloudy", "rainy"] {
        let expected = weather.expected_steps_to(start, "sunny", 365, 1000, &mut rng);
        match expected {
            Some(days) => println!("  From {}: {:.1} days", start, days),
            None => println!("  From {}: unreachable", start),
        }
    }
    
    println!("\nAverage days until rainy (from 1000 simulations):");
    for start in &["sunny", "cloudy"] {
        let expected = weather.expected_steps_to(start, "rainy", 365, 1000, &mut rng);
        match expected {
            Some(days) => println!("  From {}: {:.1} days", start, days),
            None => println!("  From {}: unreachable", start),
        }
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 6. Learning from Observed Data
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("6. LEARNING FROM OBSERVED DATA");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    // Simulated historical weather observations
    let historical_data = vec![
        vec!["sunny", "sunny", "cloudy", "rainy", "rainy", "cloudy", "sunny"],
        vec!["rainy", "rainy", "cloudy", "cloudy", "sunny", "sunny", "sunny"],
        vec!["cloudy", "sunny", "sunny", "sunny", "cloudy", "rainy", "rainy"],
        vec!["sunny", "cloudy", "cloudy", "rainy", "cloudy", "sunny", "sunny"],
    ];
    
    let mut learned_weather = StateChain::first_order();
    learned_weather.train_many(&historical_data);
    
    println!("Learned transition probabilities from 4 weeks of data:");
    for state in &["sunny", "cloudy", "rainy"] {
        let probs = learned_weather.probabilities_from(state);
        print!("  {} →", state);
        for next in &["sunny", "cloudy", "rainy"] {
            let p = probs.get(&next.to_string()).unwrap_or(&0.0);
            print!("  {}: {:.0}%", next, p * 100.0);
        }
        println!();
    }
    
    println!("\nLearned stationary distribution:");
    let learned_stationary = learned_weather.stationary_distribution(1000);
    for (state, prob) in &learned_stationary {
        println!("  {}: {:.1}%", state, prob * 100.0);
    }
    
    println!("\n════════════════════════════════════════════════════════════════");
    println!("Key Insights:");
    println!("  • Markov chains can model any system with discrete states");
    println!("  • The stationary distribution shows long-term behavior");
    println!("  • Can be learned from observed data or specified directly");
    println!("════════════════════════════════════════════════════════════════\n");
}
