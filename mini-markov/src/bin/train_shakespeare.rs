//! Shakespeare Text Generation Demo
//! 
//! Trains on a larger Shakespeare corpus to demonstrate text generation quality.

use mini_markov::TextGenerator;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║       Mini Markov - Shakespeare Text Generation                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Extended Shakespeare corpus
    let hamlet = r#"
        To be, or not to be, that is the question.
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles,
        And by opposing end them. To die, to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to. 'Tis a consummation
        Devoutly to be wished. To die, to sleep;
        To sleep, perchance to dream. Ay, there's the rub;
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause. There's the respect
        That makes calamity of so long life.
        For who would bear the whips and scorns of time,
        The oppressor's wrong, the proud man's contumely,
        The pangs of despised love, the law's delay,
        The insolence of office, and the spurns
        That patient merit of the unworthy takes,
        When he himself might his quietus make
        With a bare bodkin? Who would fardels bear,
        To grunt and sweat under a weary life,
        But that the dread of something after death,
        The undiscovered country from whose bourn
        No traveler returns, puzzles the will,
        And makes us rather bear those ills we have
        Than fly to others that we know not of?
        Thus conscience does make cowards of us all,
        And thus the native hue of resolution
        Is sicklied o'er with the pale cast of thought,
        And enterprises of great pitch and moment
        With this regard their currents turn awry
        And lose the name of action.
    "#;

    let macbeth = r#"
        Tomorrow, and tomorrow, and tomorrow,
        Creeps in this petty pace from day to day,
        To the last syllable of recorded time;
        And all our yesterdays have lighted fools
        The way to dusty death. Out, out, brief candle!
        Life's but a walking shadow, a poor player,
        That struts and frets his hour upon the stage,
        And then is heard no more. It is a tale
        Told by an idiot, full of sound and fury,
        Signifying nothing.
        
        Double, double toil and trouble;
        Fire burn and caldron bubble.
        Fillet of a fenny snake,
        In the caldron boil and bake;
        Eye of newt and toe of frog,
        Wool of bat and tongue of dog.
        
        Is this a dagger which I see before me,
        The handle toward my hand? Come, let me clutch thee.
        I have thee not, and yet I see thee still.
        Art thou not, fatal vision, sensible
        To feeling as to sight? Or art thou but
        A dagger of the mind, a false creation,
        Proceeding from the heat-oppressed brain?
    "#;

    let romeo_juliet = r#"
        But, soft! What light through yonder window breaks?
        It is the east, and Juliet is the sun.
        Arise, fair sun, and kill the envious moon,
        Who is already sick and pale with grief,
        That thou her maid art far more fair than she.
        Be not her maid, since she is envious;
        Her vestal livery is but sick and green
        And none but fools do wear it; cast it off.
        
        O Romeo, Romeo, wherefore art thou Romeo?
        Deny thy father and refuse thy name.
        Or if thou wilt not, be but sworn my love
        And I'll no longer be a Capulet.
        
        What's in a name? That which we call a rose
        By any other name would smell as sweet.
        So Romeo would, were he not Romeo called,
        Retain that dear perfection which he owes
        Without that title.
        
        Good night, good night! Parting is such sweet sorrow,
        That I shall say good night till it be morrow.
    "#;

    let merchant_venice = r#"
        The quality of mercy is not strained;
        It droppeth as the gentle rain from heaven
        Upon the place beneath. It is twice blest;
        It blesseth him that gives and him that takes.
        'Tis mightiest in the mightiest; it becomes
        The throned monarch better than his crown.
        His scepter shows the force of temporal power,
        The attribute to awe and majesty,
        Wherein doth sit the dread and fear of kings;
        But mercy is above this sceptered sway;
        It is enthroned in the hearts of kings;
        It is an attribute to God himself;
        And earthly power doth then show likest God's
        When mercy seasons justice.
    "#;

    let midsummer = r#"
        The course of true love never did run smooth.
        Love looks not with the eyes, but with the mind,
        And therefore is winged Cupid painted blind.
        If we shadows have offended,
        Think but this, and all is mended,
        That you have but slumbered here
        While these visions did appear.
        And this weak and idle theme,
        No more yielding but a dream.
    "#;

    // Use fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // ══════════════════════════════════════════════════════════════════════════
    // 1. Train on Combined Shakespeare
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("1. TRAINING ON SHAKESPEARE CORPUS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let texts = [hamlet, macbeth, romeo_juliet, merchant_venice, midsummer];
    
    println!("Corpus includes:");
    println!("  • Hamlet (To be or not to be)");
    println!("  • Macbeth (Tomorrow and tomorrow)");
    println!("  • Romeo and Juliet (But soft what light)");
    println!("  • Merchant of Venice (Quality of mercy)");
    println!("  • A Midsummer Night's Dream");
    println!();

    // Compare different orders
    for order in 1..=3 {
        println!("────────────────────────────────────────────────────────────────");
        println!("Order {} (N-gram size: {} word{})", order, order, if order > 1 { "s" } else { "" });
        println!("────────────────────────────────────────────────────────────────");
        
        let mut gen = TextGenerator::new(order);
        for text in &texts {
            gen.train(text);
        }
        
        let stats = gen.stats();
        println!("\nStatistics:");
        println!("  Unique contexts: {}", stats.num_states);
        println!("  Total transitions: {}", stats.num_transitions);
        println!("  Entropy: {:.3} bits", stats.entropy);
        
        println!("\nGenerated samples:");
        for i in 1..=3 {
            let text = gen.generate_sentence(25, &mut rng);
            println!("  {}. {}", i, text);
        }
        println!();
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 2. Style Continuation
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("2. STYLE CONTINUATION");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut gen2 = TextGenerator::new(2);
    for text in &texts {
        gen2.train(text);
    }

    let prompts = [
        "to be or",
        "the quality of",
        "what light through",
        "tomorrow and",
        "love looks not",
    ];

    for prompt in prompts {
        println!("Prompt: \"{}\"", prompt);
        let continued = gen2.generate_from(prompt, 15, &mut rng);
        println!("  → {}\n", continued);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 3. Individual Play Styles
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("3. INDIVIDUAL PLAY STYLES");
    println!("═══════════════════════════════════════════════════════════════\n");

    let plays = [
        ("Hamlet", hamlet),
        ("Macbeth", macbeth),
        ("Romeo & Juliet", romeo_juliet),
    ];

    for (name, text) in plays {
        println!("{}:", name);
        
        let mut gen = TextGenerator::new(2);
        gen.train(text);
        
        let stats = gen.stats();
        let sample = gen.generate_sentence(20, &mut rng);
        
        println!("  Entropy: {:.3} bits | Contexts: {}", stats.entropy, stats.num_states);
        println!("  Sample: {}\n", sample);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 4. Entropy Analysis
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("4. ENTROPY VS ORDER ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Order │ States │ Entropy │ Sample Quality");
    println!("──────┼────────┼─────────┼──────────────────────────────────────");

    for order in 1..=4 {
        let mut gen = TextGenerator::new(order);
        for text in &texts {
            gen.train(text);
        }
        
        let stats = gen.stats();
        let quality = if stats.entropy < 0.5 {
            "Very deterministic (memorized)"
        } else if stats.entropy < 1.5 {
            "Low creativity (mostly verbatim)"
        } else if stats.entropy < 2.5 {
            "Balanced creativity"
        } else {
            "High creativity (possibly incoherent)"
        };
        
        println!("  {}   │ {:>6} │  {:.3}  │ {}", 
            order, stats.num_states, stats.entropy, quality);
    }
    
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("Key Observations:");
    println!("  • Order 1: Random word soup, high entropy, little coherence");
    println!("  • Order 2: Some local coherence, good balance");
    println!("  • Order 3: More coherent but starts memorizing phrases");
    println!("  • Order 4+: Mostly verbatim reproduction of training text");
    println!("════════════════════════════════════════════════════════════════\n");
}
