//! Text Generation using Markov Chains
//! 
//! This module provides word-level and character-level text generation.

use crate::chain::MarkovChain;
use rand::Rng;
use std::collections::HashSet;

/// A text generator that uses Markov chains for word-level generation.
#[derive(Debug, Clone)]
pub struct TextGenerator {
    /// Underlying Markov chain operating on words
    chain: MarkovChain<String>,
    
    /// Set of sentence-ending tokens
    end_tokens: HashSet<String>,
    
    /// Whether to preserve capitalization
    preserve_case: bool,
}

impl TextGenerator {
    /// Creates a new text generator with the specified n-gram size.
    /// 
    /// # Arguments
    /// * `n` - The n-gram size (1 = unigram, 2 = bigram, etc.)
    pub fn new(n: usize) -> Self {
        let mut end_tokens = HashSet::new();
        end_tokens.insert(".".to_string());
        end_tokens.insert("!".to_string());
        end_tokens.insert("?".to_string());
        
        TextGenerator {
            chain: MarkovChain::new(n),
            end_tokens,
            preserve_case: false,
        }
    }
    
    /// Sets whether to preserve case in training and generation.
    pub fn preserve_case(mut self, preserve: bool) -> Self {
        self.preserve_case = preserve;
        self
    }
    
    /// Adds a custom end token.
    pub fn add_end_token(mut self, token: &str) -> Self {
        self.end_tokens.insert(token.to_string());
        self
    }
    
    /// Tokenizes text into words.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.preserve_case {
            text.to_string()
        } else {
            text.to_lowercase()
        };
        
        // Simple tokenization: split on whitespace, keep punctuation attached
        text.split_whitespace()
            .flat_map(|word| Self::split_punctuation(word))
            .filter(|w| !w.is_empty())
            .collect()
    }
    
    /// Splits punctuation from words, returning owned strings.
    fn split_punctuation(word: &str) -> Vec<String> {
        let mut result = Vec::new();
        
        // Handle leading punctuation
        let mut start = 0;
        for (i, c) in word.char_indices() {
            if c.is_alphanumeric() {
                break;
            }
            result.push(word[i..i + c.len_utf8()].to_string());
            start = i + c.len_utf8();
        }
        
        // Handle main word and trailing punctuation
        let remaining = &word[start..];
        if !remaining.is_empty() {
            let mut end = remaining.len();
            for (i, c) in remaining.char_indices().rev() {
                if c.is_alphanumeric() {
                    end = i + c.len_utf8();
                    break;
                }
            }
            
            let main_word = &remaining[..end];
            let trailing = &remaining[end..];
            
            if !main_word.is_empty() {
                result.push(main_word.to_string());
            }
            
            // Split trailing punctuation into individual characters
            for c in trailing.chars() {
                result.push(c.to_string());
            }
        }
        
        result
    }
    
    /// Trains the generator on a text corpus.
    /// 
    /// # Arguments
    /// * `text` - The training text
    pub fn train(&mut self, text: &str) {
        let tokens = self.tokenize(text);
        if !tokens.is_empty() {
            self.chain.train(&tokens);
        }
    }
    
    /// Trains on multiple texts.
    pub fn train_many(&mut self, texts: &[&str]) {
        for text in texts {
            self.train(text);
        }
    }
    
    /// Generates text with a maximum number of words.
    /// 
    /// # Arguments
    /// * `max_words` - Maximum number of words to generate
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// Generated text as a string
    pub fn generate<R: Rng>(&self, max_words: usize, rng: &mut R) -> String {
        let tokens = self.chain.generate(max_words, rng);
        self.join_tokens(&tokens)
    }
    
    /// Generates text starting with a specific prompt.
    /// 
    /// # Arguments
    /// * `prompt` - The starting text
    /// * `max_words` - Maximum number of additional words
    /// * `rng` - Random number generator
    pub fn generate_from<R: Rng>(&self, prompt: &str, max_words: usize, rng: &mut R) -> String {
        let prompt_tokens = self.tokenize(prompt);
        
        if prompt_tokens.len() < self.chain.order() {
            // Prompt too short, just generate normally
            return self.generate(max_words, rng);
        }
        
        let mut result = prompt_tokens.clone();
        
        while result.len() < prompt_tokens.len() + max_words {
            let current = &result[result.len() - self.chain.order()..];
            match self.chain.sample_next(current, rng) {
                Some(next) => {
                    let is_end = self.end_tokens.contains(&next);
                    result.push(next);
                    if is_end {
                        break;
                    }
                }
                None => break,
            }
        }
        
        self.join_tokens(&result)
    }
    
    /// Generates a complete sentence.
    pub fn generate_sentence<R: Rng>(&self, max_words: usize, rng: &mut R) -> String {
        let mut result = match self.chain.sample_start(rng) {
            Some(start) => start,
            None => return String::new(),
        };
        
        while result.len() < max_words {
            let current = &result[result.len() - self.chain.order()..];
            match self.chain.sample_next(current, rng) {
                Some(next) => {
                    let is_end = self.end_tokens.contains(&next);
                    result.push(next);
                    if is_end {
                        break;
                    }
                }
                None => break,
            }
        }
        
        self.join_tokens(&result)
    }
    
    /// Joins tokens back into text with proper spacing.
    fn join_tokens(&self, tokens: &[String]) -> String {
        let mut result = String::new();
        let punctuation: HashSet<char> = ".!?,;:'\"".chars().collect();
        
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                // Don't add space before punctuation
                let first_char = token.chars().next().unwrap_or(' ');
                if !punctuation.contains(&first_char) {
                    result.push(' ');
                }
            }
            result.push_str(token);
        }
        
        result
    }
    
    /// Returns statistics about the generator.
    pub fn stats(&self) -> TextGeneratorStats {
        TextGeneratorStats {
            order: self.chain.order(),
            num_states: self.chain.num_states(),
            num_transitions: self.chain.num_transitions(),
            num_sequences: self.chain.num_sequences(),
            entropy: self.chain.entropy(),
        }
    }
}

/// Statistics about a text generator.
#[derive(Debug, Clone)]
pub struct TextGeneratorStats {
    pub order: usize,
    pub num_states: usize,
    pub num_transitions: usize,
    pub num_sequences: usize,
    pub entropy: f64,
}

impl std::fmt::Display for TextGeneratorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Text Generator Statistics:")?;
        writeln!(f, "  Order (n-gram size): {}", self.order)?;
        writeln!(f, "  Unique states: {}", self.num_states)?;
        writeln!(f, "  Total transitions: {}", self.num_transitions)?;
        writeln!(f, "  Training sequences: {}", self.num_sequences)?;
        writeln!(f, "  Entropy: {:.4} bits", self.entropy)
    }
}

/// Character-level text generator.
#[derive(Debug, Clone)]
pub struct CharGenerator {
    chain: MarkovChain<char>,
}

impl CharGenerator {
    /// Creates a new character-level generator.
    /// 
    /// # Arguments
    /// * `n` - Number of previous characters to consider
    pub fn new(n: usize) -> Self {
        CharGenerator {
            chain: MarkovChain::new(n),
        }
    }
    
    /// Trains on a text string.
    pub fn train(&mut self, text: &str) {
        let chars: Vec<char> = text.chars().collect();
        if !chars.is_empty() {
            self.chain.train(&chars);
        }
    }
    
    /// Generates text of a given length.
    pub fn generate<R: Rng>(&self, length: usize, rng: &mut R) -> String {
        self.chain.generate(length, rng).into_iter().collect()
    }
    
    /// Generates text starting with a specific string.
    pub fn generate_from<R: Rng>(&self, start: &str, length: usize, rng: &mut R) -> String {
        let start_chars: Vec<char> = start.chars().collect();
        
        if start_chars.len() < self.chain.order() {
            return self.generate(length, rng);
        }
        
        let mut result = start_chars.clone();
        
        while result.len() < start_chars.len() + length {
            let current = &result[result.len() - self.chain.order()..];
            match self.chain.sample_next(current, rng) {
                Some(next) => result.push(next),
                None => break,
            }
        }
        
        result.into_iter().collect()
    }
    
    /// Returns the underlying chain.
    pub fn chain(&self) -> &MarkovChain<char> {
        &self.chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_generator() {
        let mut gen = TextGenerator::new(1);
        gen.train("the cat sat on the mat");
        gen.train("the dog ran on the grass");
        
        let stats = gen.stats();
        assert!(stats.num_states > 0);
        assert!(stats.num_transitions > 0);
    }
    
    #[test]
    fn test_char_generator() {
        let mut gen = CharGenerator::new(3);
        gen.train("hello world hello world hello world");
        
        let mut rng = rand::thread_rng();
        let generated = gen.generate(20, &mut rng);
        
        assert!(!generated.is_empty());
    }
    
    #[test]
    fn test_generation_from_prompt() {
        let mut gen = TextGenerator::new(1);
        gen.train("the quick brown fox jumps over the lazy dog");
        
        let mut rng = rand::thread_rng();
        let text = gen.generate_from("the", 10, &mut rng);
        
        assert!(text.starts_with("the"));
    }
}
