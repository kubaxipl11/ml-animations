//! Music Chord Progression Demo
//! 
//! Demonstrates Markov chains for modeling musical chord progressions.

use mini_markov::StateChain;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       Mini Markov - Musical Chord Progression Demo               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Use fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // ══════════════════════════════════════════════════════════════════════════
    // 1. Pop/Rock Chord Progressions
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("1. POP/ROCK CHORD PROGRESSION MODEL");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    // Common pop chord progressions in Roman numerals
    // I = Tonic, IV = Subdominant, V = Dominant, vi = Relative minor
    
    let mut pop = StateChain::first_order()
        .with_states(&["I", "ii", "iii", "IV", "V", "vi", "vii°"]);
    
    // Common pop transitions based on music theory
    // I (Tonic) - can go anywhere
    pop.add_transition_count("I", "I", 5);
    pop.add_transition_count("I", "IV", 35);
    pop.add_transition_count("I", "V", 30);
    pop.add_transition_count("I", "vi", 25);
    pop.add_transition_count("I", "ii", 5);
    
    // IV (Subdominant) - tends to V or I
    pop.add_transition_count("IV", "I", 30);
    pop.add_transition_count("IV", "V", 40);
    pop.add_transition_count("IV", "vi", 15);
    pop.add_transition_count("IV", "IV", 10);
    pop.add_transition_count("IV", "ii", 5);
    
    // V (Dominant) - strongly resolves to I
    pop.add_transition_count("V", "I", 60);
    pop.add_transition_count("V", "vi", 20);
    pop.add_transition_count("V", "IV", 15);
    pop.add_transition_count("V", "V", 5);
    
    // vi (Relative minor) - often to IV or V
    pop.add_transition_count("vi", "IV", 40);
    pop.add_transition_count("vi", "V", 25);
    pop.add_transition_count("vi", "I", 15);
    pop.add_transition_count("vi", "ii", 15);
    pop.add_transition_count("vi", "vi", 5);
    
    // ii (Supertonic) - tends to V
    pop.add_transition_count("ii", "V", 50);
    pop.add_transition_count("ii", "IV", 25);
    pop.add_transition_count("ii", "I", 15);
    pop.add_transition_count("ii", "vi", 10);
    
    println!("Common chord transitions (in key of C major):");
    println!("  I=C, ii=Dm, iii=Em, IV=F, V=G, vi=Am, vii°=Bdim\n");
    
    // Generate some chord progressions
    println!("Generated 8-bar progressions:\n");
    
    for i in 1..=5 {
        let progression = pop.simulate("I", 7, &mut rng);
        let chords_c: Vec<&str> = progression.iter()
            .map(|s| match s.as_str() {
                "I" => "C",
                "ii" => "Dm",
                "iii" => "Em",
                "IV" => "F",
                "V" => "G",
                "vi" => "Am",
                "vii°" => "Bdim",
                _ => s.as_str(),
            })
            .collect();
        
        println!("  #{}: │ {} │ {} │ {} │ {} │ {} │ {} │ {} │ {} │",
            i,
            chords_c.get(0).unwrap_or(&"?"),
            chords_c.get(1).unwrap_or(&"?"),
            chords_c.get(2).unwrap_or(&"?"),
            chords_c.get(3).unwrap_or(&"?"),
            chords_c.get(4).unwrap_or(&"?"),
            chords_c.get(5).unwrap_or(&"?"),
            chords_c.get(6).unwrap_or(&"?"),
            chords_c.get(7).unwrap_or(&"?"),
        );
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 2. Learn from Famous Songs
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("2. LEARNING FROM FAMOUS SONGS");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    let mut learned = StateChain::first_order();
    
    // Canon in D (Pachelbel) - I V vi iii IV I IV V
    learned.train(&["I", "V", "vi", "iii", "IV", "I", "IV", "V"]);
    
    // Let It Be (Beatles) - I V vi IV
    learned.train(&["I", "V", "vi", "IV", "I", "V", "vi", "IV"]);
    
    // With or Without You (U2) - I V vi IV
    learned.train(&["I", "V", "vi", "IV", "I", "V", "vi", "IV"]);
    
    // No Woman No Cry - I V vi IV
    learned.train(&["I", "V", "vi", "IV", "I", "V", "vi", "IV"]);
    
    // Someone Like You (Adele) - I V vi IV
    learned.train(&["I", "V", "vi", "IV", "I", "V", "vi", "IV"]);
    
    // Stand By Me - I vi IV V
    learned.train(&["I", "vi", "IV", "V", "I", "vi", "IV", "V"]);
    
    // Twist and Shout - I IV V
    learned.train(&["I", "IV", "V", "I", "IV", "V", "I", "IV", "V"]);
    
    // Blue Moon - I vi IV V
    learned.train(&["I", "vi", "IV", "V", "I", "vi", "IV", "V"]);
    
    println!("Trained on famous progressions:");
    println!("  • Canon in D (Pachelbel)");
    println!("  • Let It Be (Beatles)");
    println!("  • With or Without You (U2)");
    println!("  • Someone Like You (Adele)");
    println!("  • Stand By Me (Ben E. King)");
    println!("  • Twist and Shout");
    println!("  • Blue Moon");
    println!();
    
    println!("Learned transition probabilities:");
    for state in &["I", "IV", "V", "vi"] {
        let probs = learned.probabilities_from(state);
        print!("  {} →", state);
        for (next, prob) in &probs {
            if *prob > 0.05 {
                print!("  {}: {:.0}%", next, prob * 100.0);
            }
        }
        println!();
    }
    
    println!("\nGenerated progressions from learned model:");
    for i in 1..=3 {
        let prog = learned.simulate("I", 7, &mut rng);
        println!("  #{}: {}", i, prog.join(" → "));
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 3. Jazz Chord Progressions (ii-V-I)
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("3. JAZZ CHORD PROGRESSION MODEL");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    let mut jazz = StateChain::first_order();
    
    // Jazz strongly favors ii-V-I movement
    // Also includes tritone substitutions and chromatic movement
    
    // ii (minor 7) - almost always goes to V
    jazz.add_transition_count("iim7", "V7", 70);
    jazz.add_transition_count("iim7", "bII7", 15);  // Tritone sub of V
    jazz.add_transition_count("iim7", "Imaj7", 10);
    jazz.add_transition_count("iim7", "vim7", 5);
    
    // V7 (Dominant 7) - strongly resolves to I
    jazz.add_transition_count("V7", "Imaj7", 70);
    jazz.add_transition_count("V7", "vim7", 15);  // Deceptive resolution
    jazz.add_transition_count("V7", "iim7", 10);  // Back to ii
    jazz.add_transition_count("V7", "IVmaj7", 5);
    
    // Imaj7 (Major 7) - can go many places
    jazz.add_transition_count("Imaj7", "iim7", 35);
    jazz.add_transition_count("Imaj7", "IVmaj7", 25);
    jazz.add_transition_count("Imaj7", "vim7", 20);
    jazz.add_transition_count("Imaj7", "iiim7", 10);
    jazz.add_transition_count("Imaj7", "Imaj7", 10);
    
    // vim7 - often to ii
    jazz.add_transition_count("vim7", "iim7", 50);
    jazz.add_transition_count("vim7", "V7", 25);
    jazz.add_transition_count("vim7", "IVmaj7", 15);
    jazz.add_transition_count("vim7", "vim7", 10);
    
    // IVmaj7 - various destinations
    jazz.add_transition_count("IVmaj7", "iiim7", 30);
    jazz.add_transition_count("IVmaj7", "iim7", 25);
    jazz.add_transition_count("IVmaj7", "V7", 25);
    jazz.add_transition_count("IVmaj7", "Imaj7", 20);
    
    // iiim7 - to vi or ii
    jazz.add_transition_count("iiim7", "vim7", 50);
    jazz.add_transition_count("iiim7", "iim7", 30);
    jazz.add_transition_count("iiim7", "IVmaj7", 20);
    
    // Tritone sub of V
    jazz.add_transition_count("bII7", "Imaj7", 80);
    jazz.add_transition_count("bII7", "iim7", 20);
    
    println!("Jazz chord vocabulary:");
    println!("  Imaj7=Cmaj7, iim7=Dm7, iiim7=Em7, IVmaj7=Fmaj7");
    println!("  V7=G7, vim7=Am7, bII7=Db7 (tritone sub)");
    println!();
    
    println!("Generated jazz progressions (8 bars):\n");
    for i in 1..=4 {
        let prog = jazz.simulate("Imaj7", 7, &mut rng);
        let chords: Vec<&str> = prog.iter()
            .map(|s| match s.as_str() {
                "Imaj7" => "Cmaj7",
                "iim7" => "Dm7",
                "iiim7" => "Em7",
                "IVmaj7" => "Fmaj7",
                "V7" => "G7",
                "vim7" => "Am7",
                "bII7" => "Db7",
                _ => s.as_str(),
            })
            .collect();
        println!("  #{}: {}", i, chords.join(" → "));
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 4. Blues Progression
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("4. 12-BAR BLUES MODEL");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    // Standard 12-bar blues has a very predictable structure
    // But we can model it with some variations
    
    let mut blues = StateChain::first_order();
    
    // Classic 12-bar: I I I I | IV IV I I | V IV I V
    blues.train(&["I7", "I7", "I7", "I7", "IV7", "IV7", "I7", "I7", "V7", "IV7", "I7", "V7"]);
    
    // Quick change variation: I I7 IV IV7 | ...
    blues.train(&["I7", "IV7", "I7", "I7", "IV7", "IV7", "I7", "I7", "V7", "IV7", "I7", "V7"]);
    
    // Minor blues variation
    blues.train(&["im7", "im7", "im7", "im7", "ivm7", "ivm7", "im7", "im7", "V7", "ivm7", "im7", "V7"]);
    
    println!("Classic 12-bar blues structure:");
    println!("  │ I7  │ I7  │ I7  │ I7  │ (bars 1-4)");
    println!("  │ IV7 │ IV7 │ I7  │ I7  │ (bars 5-8)");
    println!("  │ V7  │ IV7 │ I7  │ V7  │ (bars 9-12)");
    println!();
    
    println!("Generated blues variations (12 bars each):\n");
    for i in 1..=3 {
        let prog = blues.simulate("I7", 11, &mut rng);
        println!("  #{}: │ {} │", i, prog.join(" │ "));
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 5. Analyze Chord Distribution
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("5. LONG-TERM CHORD DISTRIBUTION");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    println!("Pop model stationary distribution:");
    let pop_stationary = pop.stationary_distribution(1000);
    let mut pop_sorted: Vec<_> = pop_stationary.iter().collect();
    pop_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    
    for (chord, prob) in pop_sorted {
        let bar_len = (prob * 50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:>5} │{} {:.1}%", chord, bar, prob * 100.0);
    }
    
    println!("\nJazz model stationary distribution:");
    let jazz_stationary = jazz.stationary_distribution(1000);
    let mut jazz_sorted: Vec<_> = jazz_stationary.iter().collect();
    jazz_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    
    for (chord, prob) in jazz_sorted {
        let bar_len = (prob * 50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:>7} │{} {:.1}%", chord, bar, prob * 100.0);
    }
    println!();

    println!("════════════════════════════════════════════════════════════════");
    println!("Key Insights:");
    println!("  • Pop music heavily favors I, IV, V, vi (the 'four chord' song)");
    println!("  • Jazz emphasizes ii-V-I motion with extended harmonies");
    println!("  • Blues follows a strict 12-bar form with dominant 7ths");
    println!("  • Markov chains capture harmonic 'expectations' in music");
    println!("════════════════════════════════════════════════════════════════\n");
}
