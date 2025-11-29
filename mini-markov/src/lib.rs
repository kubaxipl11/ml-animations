//! Mini Markov Chain Library
//! 
//! A from-scratch implementation of Markov Chains in Rust.
//! Supports text generation, state transition modeling, and higher-order chains.

pub mod chain;
pub mod text;
pub mod state;
pub mod utils;

pub use chain::{MarkovChain, Transition};
pub use text::TextGenerator;
pub use state::StateChain;
pub use utils::{kl_divergence, js_divergence, perplexity, normalize_distribution, ascii_histogram};
