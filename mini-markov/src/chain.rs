//! Core Markov Chain Implementation
//! 
//! A Markov Chain is a stochastic model where the probability of each event
//! depends only on the state attained in the previous event (Markov property).
//! 
//! P(Xn+1 = x | X1, X2, ..., Xn) = P(Xn+1 = x | Xn)

use std::collections::HashMap;
use std::hash::Hash;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// A generic Markov Chain that can work with any hashable state type.
/// 
/// # Type Parameters
/// * `T` - The type of states in the chain (must be Clone, Eq, Hash)
/// 
/// # Example
/// ```
/// use mini_markov::MarkovChain;
/// 
/// let mut chain: MarkovChain<char> = MarkovChain::new(1);
/// chain.train(&['a', 'b', 'a', 'b', 'a', 'c']);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovChain<T> 
where 
    T: Clone + Eq + Hash,
{
    /// Order of the Markov chain (how many previous states to consider)
    order: usize,
    
    /// Transition counts: maps state sequences to counts of next states
    /// Key: Previous state(s) as a vector
    /// Value: HashMap of next_state -> count
    transitions: HashMap<Vec<T>, HashMap<T, usize>>,
    
    /// Starting states (beginning of sequences)
    start_states: HashMap<Vec<T>, usize>,
    
    /// Total number of training sequences
    num_sequences: usize,
}

impl<T> MarkovChain<T> 
where 
    T: Clone + Eq + Hash,
{
    /// Creates a new Markov chain with the specified order.
    /// 
    /// # Arguments
    /// * `order` - Number of previous states to consider (1 = first-order, 2 = second-order, etc.)
    /// 
    /// # Panics
    /// Panics if order is 0.
    pub fn new(order: usize) -> Self {
        assert!(order > 0, "Order must be at least 1");
        
        MarkovChain {
            order,
            transitions: HashMap::new(),
            start_states: HashMap::new(),
            num_sequences: 0,
        }
    }
    
    /// Returns the order of the chain.
    pub fn order(&self) -> usize {
        self.order
    }
    
    /// Trains the chain on a single sequence of states.
    /// 
    /// # Arguments
    /// * `sequence` - A slice of states to learn from
    pub fn train(&mut self, sequence: &[T]) {
        if sequence.len() <= self.order {
            return; // Sequence too short
        }
        
        self.num_sequences += 1;
        
        // Record start state
        let start: Vec<T> = sequence[..self.order].to_vec();
        *self.start_states.entry(start).or_insert(0) += 1;
        
        // Record transitions
        for i in 0..sequence.len() - self.order {
            let current: Vec<T> = sequence[i..i + self.order].to_vec();
            let next = sequence[i + self.order].clone();
            
            self.transitions
                .entry(current)
                .or_insert_with(HashMap::new)
                .entry(next)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
    }
    
    /// Trains the chain on multiple sequences.
    /// 
    /// # Arguments
    /// * `sequences` - Iterator of sequences to learn from
    pub fn train_many<'a, I>(&mut self, sequences: I)
    where
        I: IntoIterator<Item = &'a [T]>,
        T: 'a,
    {
        for seq in sequences {
            self.train(seq);
        }
    }
    
    /// Gets the probability distribution for the next state given the current state(s).
    /// 
    /// # Arguments
    /// * `current` - The current state sequence
    /// 
    /// # Returns
    /// A HashMap mapping each possible next state to its probability
    pub fn get_probabilities(&self, current: &[T]) -> Option<HashMap<T, f64>> {
        if current.len() != self.order {
            return None;
        }
        
        let key: Vec<T> = current.to_vec();
        self.transitions.get(&key).map(|counts| {
            let total: usize = counts.values().sum();
            counts
                .iter()
                .map(|(state, count)| (state.clone(), *count as f64 / total as f64))
                .collect()
        })
    }
    
    /// Samples the next state given the current state(s).
    /// 
    /// # Arguments
    /// * `current` - The current state sequence
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// The sampled next state, or None if the current state is unknown
    pub fn sample_next<R: Rng>(&self, current: &[T], rng: &mut R) -> Option<T> {
        if current.len() != self.order {
            return None;
        }
        
        let key: Vec<T> = current.to_vec();
        let counts = self.transitions.get(&key)?;
        
        let total: usize = counts.values().sum();
        let mut threshold = rng.gen_range(0..total);
        
        for (state, count) in counts {
            if threshold < *count {
                return Some(state.clone());
            }
            threshold -= count;
        }
        
        // Should never reach here
        counts.keys().next().cloned()
    }
    
    /// Samples a random starting state.
    /// 
    /// # Arguments
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// A random starting state sequence
    pub fn sample_start<R: Rng>(&self, rng: &mut R) -> Option<Vec<T>> {
        if self.start_states.is_empty() {
            return None;
        }
        
        let total: usize = self.start_states.values().sum();
        let mut threshold = rng.gen_range(0..total);
        
        for (state, count) in &self.start_states {
            if threshold < *count {
                return Some(state.clone());
            }
            threshold -= count;
        }
        
        self.start_states.keys().next().cloned()
    }
    
    /// Generates a sequence of states.
    /// 
    /// # Arguments
    /// * `length` - Maximum length of the generated sequence
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// A vector of generated states
    pub fn generate<R: Rng>(&self, length: usize, rng: &mut R) -> Vec<T> {
        let mut result = match self.sample_start(rng) {
            Some(start) => start,
            None => return Vec::new(),
        };
        
        while result.len() < length {
            let current = &result[result.len() - self.order..];
            match self.sample_next(current, rng) {
                Some(next) => result.push(next),
                None => break, // No known transitions from this state
            }
        }
        
        result
    }
    
    /// Returns the number of unique state sequences in the chain.
    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }
    
    /// Returns the total number of transitions recorded.
    pub fn num_transitions(&self) -> usize {
        self.transitions
            .values()
            .map(|counts| counts.values().sum::<usize>())
            .sum()
    }
    
    /// Returns the number of training sequences used.
    pub fn num_sequences(&self) -> usize {
        self.num_sequences
    }
    
    /// Gets the transition matrix as a nested HashMap (for visualization).
    pub fn get_transition_matrix(&self) -> &HashMap<Vec<T>, HashMap<T, usize>> {
        &self.transitions
    }
    
    /// Calculates the entropy of the chain (measure of randomness).
    /// Higher entropy means more unpredictable transitions.
    pub fn entropy(&self) -> f64 {
        let mut total_entropy = 0.0;
        let mut total_weight = 0.0;
        
        for counts in self.transitions.values() {
            let total: usize = counts.values().sum();
            if total == 0 {
                continue;
            }
            
            let mut state_entropy = 0.0;
            for &count in counts.values() {
                if count > 0 {
                    let p = count as f64 / total as f64;
                    state_entropy -= p * p.log2();
                }
            }
            
            total_entropy += state_entropy * total as f64;
            total_weight += total as f64;
        }
        
        if total_weight > 0.0 {
            total_entropy / total_weight
        } else {
            0.0
        }
    }
    
    /// Clears all learned transitions.
    pub fn clear(&mut self) {
        self.transitions.clear();
        self.start_states.clear();
        self.num_sequences = 0;
    }
    
    /// Merges another chain into this one.
    pub fn merge(&mut self, other: &MarkovChain<T>) {
        assert_eq!(self.order, other.order, "Cannot merge chains with different orders");
        
        for (state, counts) in &other.transitions {
            let entry = self.transitions.entry(state.clone()).or_insert_with(HashMap::new);
            for (next, count) in counts {
                *entry.entry(next.clone()).or_insert(0) += count;
            }
        }
        
        for (state, count) in &other.start_states {
            *self.start_states.entry(state.clone()).or_insert(0) += count;
        }
        
        self.num_sequences += other.num_sequences;
    }
}

/// Transition probability for a specific state pair
#[derive(Debug, Clone)]
pub struct Transition<T> {
    pub from: Vec<T>,
    pub to: T,
    pub probability: f64,
    pub count: usize,
}

impl<T> MarkovChain<T>
where
    T: Clone + Eq + Hash + Ord,
{
    /// Gets all transitions sorted by probability (descending).
    pub fn get_top_transitions(&self, limit: usize) -> Vec<Transition<T>> {
        let mut transitions: Vec<Transition<T>> = Vec::new();
        
        for (from, counts) in &self.transitions {
            let total: usize = counts.values().sum();
            for (to, count) in counts {
                transitions.push(Transition {
                    from: from.clone(),
                    to: to.clone(),
                    probability: *count as f64 / total as f64,
                    count: *count,
                });
            }
        }
        
        transitions.sort_by(|a, b| {
            b.probability.partial_cmp(&a.probability).unwrap()
        });
        
        transitions.truncate(limit);
        transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_chain() {
        let mut chain: MarkovChain<char> = MarkovChain::new(1);
        chain.train(&['a', 'b', 'a', 'b', 'a', 'b']);
        
        assert_eq!(chain.order(), 1);
        assert!(chain.num_states() > 0);
        
        // After 'a', should always go to 'b' (probability 1.0)
        let probs = chain.get_probabilities(&['a']).unwrap();
        assert_eq!(*probs.get(&'b').unwrap(), 1.0);
    }
    
    #[test]
    fn test_second_order() {
        let mut chain: MarkovChain<char> = MarkovChain::new(2);
        chain.train(&['a', 'b', 'c', 'a', 'b', 'd']);
        
        // After 'a','b', we've seen 'c' once and 'd' once
        let probs = chain.get_probabilities(&['a', 'b']).unwrap();
        assert_eq!(*probs.get(&'c').unwrap(), 0.5);
        assert_eq!(*probs.get(&'d').unwrap(), 0.5);
    }
    
    #[test]
    fn test_generation() {
        let mut chain: MarkovChain<i32> = MarkovChain::new(1);
        chain.train(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        
        let mut rng = rand::thread_rng();
        let generated = chain.generate(10, &mut rng);
        
        assert!(!generated.is_empty());
        assert!(generated.len() <= 10);
    }
    
    #[test]
    fn test_entropy() {
        // Deterministic chain (entropy = 0)
        let mut chain1: MarkovChain<char> = MarkovChain::new(1);
        chain1.train(&['a', 'b', 'a', 'b', 'a', 'b']);
        assert!(chain1.entropy() < 0.01);
        
        // Random chain (higher entropy)
        let mut chain2: MarkovChain<char> = MarkovChain::new(1);
        chain2.train(&['a', 'b', 'a', 'c', 'a', 'd', 'a', 'e']);
        assert!(chain2.entropy() > 1.0);
    }
}
