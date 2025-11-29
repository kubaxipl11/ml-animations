//! Text Generation Demo
//! 
//! Demonstrates Markov chain text generation using famous literature.

use mini_markov::TextGenerator;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       Mini Markov - Text Generation Demo                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Sample texts for training
    let sherlock = r#"
        Sherlock Holmes took his bottle from the corner of the mantelpiece and his hypodermic syringe 
        from its neat morocco case. With his long, white, nervous fingers he adjusted the delicate 
        needle, and rolled back his left shirt-cuff. For some little time his eyes rested thoughtfully 
        upon the sinewy forearm and wrist all dotted and scarred with innumerable puncture-marks. 
        Finally he thrust the sharp point home, pressed down the tiny piston, and sank back into the 
        velvet-lined armchair with a long sigh of satisfaction.
        
        I had been watching these proceedings with curiosity and concern. I had noticed such signs 
        of his increasing agitation, without being able to divine its cause. Now, however, the 
        mystery was explained. My friend was a habitual cocaine user, using it as a stimulant to 
        his alert mind.
    "#;
    
    let dickens = r#"
        It was the best of times, it was the worst of times, it was the age of wisdom, it was the 
        age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was 
        the season of Light, it was the season of Darkness, it was the spring of hope, it was the 
        winter of despair. We had everything before us, we had nothing before us, we were all 
        going direct to Heaven, we were all going direct the other way.
        
        There were a king with a large jaw and a queen with a plain face, on the throne of England. 
        There were a king with a large jaw and a queen with a fair face, on the throne of France. 
        In both countries it was clearer than crystal to the lords of the State preserves of loaves 
        and fishes, that things in general were settled forever.
    "#;
    
    let shakespeare = r#"
        To be, or not to be, that is the question. Whether tis nobler in the mind to suffer the 
        slings and arrows of outrageous fortune, or to take arms against a sea of troubles, and 
        by opposing end them. To die, to sleep, no more, and by a sleep to say we end the heartache, 
        and the thousand natural shocks that flesh is heir to. Tis a consummation devoutly to be 
        wished. To die, to sleep, to sleep, perchance to dream. Ay, there is the rub, for in that 
        sleep of death what dreams may come, when we have shuffled off this mortal coil, must give 
        us pause.
    "#;

    // Use fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // ══════════════════════════════════════════════════════════════════════════
    // 1. Unigram (Order 1) - Word-level
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("1. UNIGRAM MODEL (Order 1)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Training on Sherlock Holmes excerpt...\n");
    
    let mut gen1 = TextGenerator::new(1);
    gen1.train(sherlock);
    
    let stats = gen1.stats();
    println!("Statistics:");
    println!("  Unique word contexts: {}", stats.num_states);
    println!("  Total transitions: {}", stats.num_transitions);
    println!("  Entropy: {:.3} bits", stats.entropy);
    
    println!("\nGenerated text (30 words):");
    println!("────────────────────────────────────────────────────────────────");
    let text = gen1.generate(30, &mut rng);
    println!("{}", text);
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 2. Bigram (Order 2) - Word-level
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("2. BIGRAM MODEL (Order 2)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Training on Dickens excerpt...\n");
    
    let mut gen2 = TextGenerator::new(2);
    gen2.train(dickens);
    
    let stats = gen2.stats();
    println!("Statistics:");
    println!("  Unique word pair contexts: {}", stats.num_states);
    println!("  Total transitions: {}", stats.num_transitions);
    println!("  Entropy: {:.3} bits", stats.entropy);
    
    println!("\nGenerated text (30 words):");
    println!("────────────────────────────────────────────────────────────────");
    let text = gen2.generate(30, &mut rng);
    println!("{}", text);
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 3. Trigram (Order 3) - Word-level
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("3. TRIGRAM MODEL (Order 3)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Training on Shakespeare excerpt...\n");
    
    let mut gen3 = TextGenerator::new(3);
    gen3.train(shakespeare);
    
    let stats = gen3.stats();
    println!("Statistics:");
    println!("  Unique word triplet contexts: {}", stats.num_states);
    println!("  Total transitions: {}", stats.num_transitions);
    println!("  Entropy: {:.3} bits", stats.entropy);
    
    println!("\nGenerated text (30 words):");
    println!("────────────────────────────────────────────────────────────────");
    let text = gen3.generate(30, &mut rng);
    println!("{}", text);
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 4. Mixed Training - Multiple Sources
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("4. MIXED MODEL (Order 2, Multiple Sources)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Training on all three texts...\n");
    
    let mut gen_mixed = TextGenerator::new(2);
    gen_mixed.train_many(&[sherlock, dickens, shakespeare]);
    
    let stats = gen_mixed.stats();
    println!("Statistics:");
    println!("  Unique word pair contexts: {}", stats.num_states);
    println!("  Total transitions: {}", stats.num_transitions);
    println!("  Entropy: {:.3} bits", stats.entropy);
    println!("  Training sequences: {}", stats.num_sequences);
    
    println!("\nGenerated text (40 words):");
    println!("────────────────────────────────────────────────────────────────");
    let text = gen_mixed.generate(40, &mut rng);
    println!("{}", text);
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 5. Prompt-based Generation
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("5. PROMPT-BASED GENERATION");
    println!("═══════════════════════════════════════════════════════════════");
    
    let prompts = ["it was the", "to be or", "sherlock holmes"];
    
    for prompt in prompts {
        println!("\nPrompt: \"{}\"", prompt);
        println!("Generated: {}", gen_mixed.generate_from(prompt, 20, &mut rng));
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // 6. Comparing Different Orders
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("6. COMPARING DIFFERENT N-GRAM ORDERS");
    println!("═══════════════════════════════════════════════════════════════");
    
    let combined_text = format!("{} {} {}", sherlock, dickens, shakespeare);
    
    for order in 1..=4 {
        let mut gen = TextGenerator::new(order);
        gen.train(&combined_text);
        
        let stats = gen.stats();
        let sample = gen.generate(15, &mut rng);
        
        println!("\nOrder {}: {} states, entropy={:.2} bits", order, stats.num_states, stats.entropy);
        println!("  Sample: {}", sample);
    }
    
    println!("\n════════════════════════════════════════════════════════════════");
    println!("Observation: Higher order = more coherent but less creative");
    println!("             Lower order = more random but potentially novel");
    println!("════════════════════════════════════════════════════════════════\n");
}
