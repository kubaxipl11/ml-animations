# Mini Markov

A from-scratch implementation of Markov Chains in Rust for educational purposes.

## Overview

A Markov Chain is a stochastic model describing a sequence of possible events where the probability of each event depends only on the state attained in the previous event(s). This is known as the **Markov property** (memorylessness).

### The Math

For a first-order Markov chain:

$$P(X_{n+1} = x | X_1, X_2, ..., X_n) = P(X_{n+1} = x | X_n)$$

The transition probability from state $i$ to state $j$ is:

$$P_{ij} = P(X_{n+1} = j | X_n = i)$$

These probabilities form a **transition matrix** $P$ where:
- Each row sums to 1
- Entry $P_{ij}$ gives probability of transitioning from state $i$ to state $j$

## Features

- ✅ Generic Markov chain implementation
- ✅ N-gram support (configurable order: 1st, 2nd, 3rd order, etc.)
- ✅ Text generation (word-level)
- ✅ State machine modeling
- ✅ Stationary distribution calculation
- ✅ Entropy measurement
- ✅ Probability calculations and transition matrices

## Project Structure

```
mini-markov/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs          # Library exports
    ├── chain.rs        # Core MarkovChain implementation
    ├── text.rs         # TextGenerator for text generation
    ├── state.rs        # StateChain for state modeling
    ├── utils.rs        # Utility functions (KL divergence, etc.)
    └── bin/
        ├── demo_text.rs         # Text generation demo
        ├── demo_weather.rs      # Weather simulation demo
        ├── demo_music.rs        # Music chord progression demo
        └── train_shakespeare.rs # Shakespeare text demo
```

## Usage

### Building

```bash
cargo build
```

### Running Tests

```bash
cargo test
```

### Running Demos

```bash
# Text generation demo
cargo run --bin demo_text

# Weather simulation demo  
cargo run --bin demo_weather

# Music chord progression demo
cargo run --bin demo_music

# Shakespeare text generation
cargo run --bin train_shakespeare
```

## Examples

### Basic Markov Chain

```rust
use mini_markov::MarkovChain;

// Create a first-order chain
let mut chain: MarkovChain<char> = MarkovChain::new(1);

// Train on a sequence
chain.train(&['a', 'b', 'a', 'b', 'a', 'c']);

// Get transition probabilities
let probs = chain.get_probabilities(&['a']).unwrap();
// After 'a': P(b) = 2/3, P(c) = 1/3

// Generate a sequence
let mut rng = rand::thread_rng();
let generated = chain.generate(10, &mut rng);
```

### Text Generation

```rust
use mini_markov::TextGenerator;

// Create a bigram (order 2) text generator
let mut gen = TextGenerator::new(2);

// Train on text
gen.train("The quick brown fox jumps over the lazy dog.");
gen.train("The quick rabbit hops over the fence.");

// Generate text
let text = gen.generate(20, &mut rng);

// Generate from a prompt
let continued = gen.generate_from("The quick", 15, &mut rng);
```

### State Machine Modeling

```rust
use mini_markov::StateChain;

// Create a weather model
let mut weather = StateChain::first_order()
    .with_states(&["sunny", "cloudy", "rainy"]);

// Add transition probabilities
weather.add_transition_count("sunny", "sunny", 70);
weather.add_transition_count("sunny", "cloudy", 20);
weather.add_transition_count("sunny", "rainy", 10);
// ... more transitions

// Simulate weather for a week
let forecast = weather.simulate("sunny", 7, &mut rng);

// Get long-term probabilities
let stationary = weather.stationary_distribution(1000);
```

## Demo Results

### Text Generation (Order Comparison)

| Order | States | Entropy | Quality |
|-------|--------|---------|---------|
| 1 | 209 | 1.61 bits | Random but creative |
| 2 | 358 | 0.20 bits | Locally coherent |
| 3 | 388 | 0.12 bits | Memorizing phrases |
| 4 | 405 | 0.09 bits | Verbatim reproduction |

### Weather Model

Starting from "sunny", stationary distribution after many steps:
- ☀️ Sunny: 46.2%
- ☁️ Cloudy: 30.8%
- 🌧️ Rainy: 23.1%

Expected steps to reach "sunny":
- From cloudy: ~3.7 days
- From rainy: ~4.2 days

### Music Chord Progressions

Pop/Rock long-term chord distribution:
- I (Tonic): 27.6%
- V (Dominant): 26.1%
- IV (Subdominant): 24.1%
- vi (Relative minor): 17.1%

This matches the famous "four chord song" pattern (I-V-vi-IV) that powers countless pop hits!

## Key Concepts

### Order (N-gram Size)

- **Order 1 (Unigram)**: Only considers current state
- **Order 2 (Bigram)**: Considers previous 2 states  
- **Order 3 (Trigram)**: Considers previous 3 states
- Higher order = more context = more coherent but less creative

### Entropy

Entropy measures the "randomness" of the chain:
- **Low entropy (< 1 bit)**: Deterministic, predictable transitions
- **High entropy (> 2 bits)**: Random, many equally likely transitions

$$H = -\sum_{i,j} P(i) \cdot P_{ij} \cdot \log_2(P_{ij})$$

### Stationary Distribution

The long-term probability distribution that the chain converges to, regardless of starting state. Found by solving:

$$\pi P = \pi$$

where $\pi$ is the stationary distribution vector and $P$ is the transition matrix.

## Applications

1. **Text Generation**: Predictive text, auto-complete, chatbots
2. **Speech Recognition**: Language modeling
3. **Bioinformatics**: DNA sequence analysis
4. **Finance**: Stock price modeling
5. **Weather Forecasting**: Short-term predictions
6. **Music Generation**: Chord progressions, melodies
7. **Game AI**: NPC behavior modeling
8. **Network Analysis**: PageRank algorithm

## References

- [Markov Chains (Wikipedia)](https://en.wikipedia.org/wiki/Markov_chain)
- [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) - Claude Shannon
- [The PageRank Citation Ranking](http://ilpubs.stanford.edu:8090/422/) - Larry Page & Sergey Brin

## License

MIT License - Feel free to use for learning and teaching!
