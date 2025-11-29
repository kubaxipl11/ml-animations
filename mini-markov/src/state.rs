//! State Machine Modeling with Markov Chains
//! 
//! This module provides tools for modeling discrete state systems,
//! such as weather patterns, game states, or any finite state machine.

use crate::chain::MarkovChain;
use rand::Rng;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A state chain for modeling discrete state systems.
/// 
/// Provides a more ergonomic API for named states and includes
/// analysis tools for understanding state behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChain {
    chain: MarkovChain<String>,
    states: Vec<String>,
}

impl StateChain {
    /// Creates a new state chain with the specified order.
    pub fn new(order: usize) -> Self {
        StateChain {
            chain: MarkovChain::new(order),
            states: Vec::new(),
        }
    }
    
    /// Creates a first-order state chain (most common).
    pub fn first_order() -> Self {
        Self::new(1)
    }
    
    /// Defines the possible states (optional, for validation).
    pub fn with_states(mut self, states: &[&str]) -> Self {
        self.states = states.iter().map(|s| s.to_string()).collect();
        self
    }
    
    /// Adds a single transition observation.
    /// 
    /// # Arguments
    /// * `from` - Current state
    /// * `to` - Next state
    pub fn add_transition(&mut self, from: &str, to: &str) {
        let seq = vec![from.to_string(), to.to_string()];
        self.chain.train(&seq);
    }
    
    /// Adds a transition with a specific count (for building from known probabilities).
    pub fn add_transition_count(&mut self, from: &str, to: &str, count: usize) {
        for _ in 0..count {
            self.add_transition(from, to);
        }
    }
    
    /// Trains on a sequence of states.
    pub fn train(&mut self, sequence: &[&str]) {
        let seq: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();
        self.chain.train(&seq);
    }
    
    /// Trains on multiple sequences.
    pub fn train_many(&mut self, sequences: &[Vec<&str>]) {
        for seq in sequences {
            self.train(seq);
        }
    }
    
    /// Gets the probability of transitioning to a specific state.
    pub fn probability(&self, from: &str, to: &str) -> f64 {
        let from_vec = vec![from.to_string()];
        match self.chain.get_probabilities(&from_vec) {
            Some(probs) => *probs.get(&to.to_string()).unwrap_or(&0.0),
            None => 0.0,
        }
    }
    
    /// Gets all transition probabilities from a state.
    pub fn probabilities_from(&self, state: &str) -> HashMap<String, f64> {
        let state_vec = vec![state.to_string()];
        self.chain
            .get_probabilities(&state_vec)
            .unwrap_or_default()
    }
    
    /// Samples the next state.
    pub fn next_state<R: Rng>(&self, current: &str, rng: &mut R) -> Option<String> {
        let current_vec = vec![current.to_string()];
        self.chain.sample_next(&current_vec, rng)
    }
    
    /// Simulates the chain for a number of steps.
    /// 
    /// # Arguments
    /// * `start` - Starting state
    /// * `steps` - Number of steps to simulate
    /// * `rng` - Random number generator
    pub fn simulate<R: Rng>(&self, start: &str, steps: usize, rng: &mut R) -> Vec<String> {
        let mut result = vec![start.to_string()];
        
        for _ in 0..steps {
            let current = result.last().unwrap();
            match self.next_state(current, rng) {
                Some(next) => result.push(next),
                None => break,
            }
        }
        
        result
    }
    
    /// Calculates the stationary distribution (long-term probabilities).
    /// Uses power iteration method.
    /// 
    /// # Arguments
    /// * `iterations` - Number of iterations for convergence
    /// 
    /// # Returns
    /// HashMap mapping each state to its long-term probability
    pub fn stationary_distribution(&self, iterations: usize) -> HashMap<String, f64> {
        // Get all unique states
        let states: Vec<String> = self.chain
            .get_transition_matrix()
            .keys()
            .flat_map(|k| k.iter().cloned())
            .chain(
                self.chain
                    .get_transition_matrix()
                    .values()
                    .flat_map(|v| v.keys().cloned())
            )
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        
        if states.is_empty() {
            return HashMap::new();
        }
        
        let n = states.len();
        let state_to_idx: HashMap<&str, usize> = states
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        
        // Build transition matrix
        let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        
        for (from, tos) in self.chain.get_transition_matrix() {
            if from.len() != 1 {
                continue; // Only works for first-order chains
            }
            
            let from_state = &from[0];
            if let Some(&from_idx) = state_to_idx.get(from_state.as_str()) {
                let total: usize = tos.values().sum();
                for (to, count) in tos {
                    if let Some(&to_idx) = state_to_idx.get(to.as_str()) {
                        matrix[from_idx][to_idx] = *count as f64 / total as f64;
                    }
                }
            }
        }
        
        // Power iteration
        let mut dist: Vec<f64> = vec![1.0 / n as f64; n];
        
        for _ in 0..iterations {
            let mut new_dist = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_dist[j] += dist[i] * matrix[i][j];
                }
            }
            dist = new_dist;
        }
        
        // Normalize
        let sum: f64 = dist.iter().sum();
        if sum > 0.0 {
            for d in &mut dist {
                *d /= sum;
            }
        }
        
        states.into_iter().zip(dist).collect()
    }
    
    /// Calculates the expected number of steps to reach a target state.
    /// Uses simulation to estimate.
    pub fn expected_steps_to<R: Rng>(
        &self,
        from: &str,
        to: &str,
        max_steps: usize,
        simulations: usize,
        rng: &mut R,
    ) -> Option<f64> {
        let mut total_steps = 0;
        let mut reached = 0;
        
        for _ in 0..simulations {
            let path = self.simulate(from, max_steps, rng);
            if let Some(pos) = path.iter().position(|s| s == to) {
                total_steps += pos;
                reached += 1;
            }
        }
        
        if reached > 0 {
            Some(total_steps as f64 / reached as f64)
        } else {
            None
        }
    }
    
    /// Returns the underlying Markov chain.
    pub fn chain(&self) -> &MarkovChain<String> {
        &self.chain
    }
    
    /// Returns all observed states.
    pub fn observed_states(&self) -> Vec<String> {
        self.chain
            .get_transition_matrix()
            .keys()
            .flat_map(|k| k.iter().cloned())
            .chain(
                self.chain
                    .get_transition_matrix()
                    .values()
                    .flat_map(|v| v.keys().cloned())
            )
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    
    /// Prints the transition matrix.
    pub fn print_matrix(&self) {
        let states = self.observed_states();
        
        // Header
        print!("{:>12}", "");
        for s in &states {
            print!("{:>12}", s);
        }
        println!();
        
        // Rows
        for from in &states {
            print!("{:>12}", from);
            let probs = self.probabilities_from(from);
            for to in &states {
                let p = probs.get(to).unwrap_or(&0.0);
                print!("{:>12.3}", p);
            }
            println!();
        }
    }
}

/// Builder for creating state chains from transition probabilities.
pub struct StateChainBuilder {
    transitions: Vec<(String, String, f64)>,
    order: usize,
}

impl StateChainBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        StateChainBuilder {
            transitions: Vec::new(),
            order: 1,
        }
    }
    
    /// Sets the order of the chain.
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }
    
    /// Adds a transition with a probability.
    /// 
    /// Note: Probabilities from each state should sum to 1.0
    pub fn transition(mut self, from: &str, to: &str, probability: f64) -> Self {
        self.transitions.push((from.to_string(), to.to_string(), probability));
        self
    }
    
    /// Builds the state chain.
    /// 
    /// Converts probabilities to counts by multiplying by 1000.
    pub fn build(self) -> StateChain {
        let mut chain = StateChain::new(self.order);
        
        for (from, to, prob) in self.transitions {
            let count = (prob * 1000.0).round() as usize;
            chain.add_transition_count(&from, &to, count);
        }
        
        chain
    }
}

impl Default for StateChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_chain() {
        let mut chain = StateChain::first_order();
        
        // Weather transitions
        chain.add_transition("sunny", "sunny");
        chain.add_transition("sunny", "sunny");
        chain.add_transition("sunny", "rainy");
        chain.add_transition("rainy", "sunny");
        chain.add_transition("rainy", "rainy");
        
        // P(sunny -> sunny) should be ~0.67
        let p = chain.probability("sunny", "sunny");
        assert!((p - 0.67).abs() < 0.1);
    }
    
    #[test]
    fn test_simulation() {
        let chain = StateChainBuilder::new()
            .transition("A", "B", 1.0)
            .transition("B", "A", 1.0)
            .build();
        
        let mut rng = rand::thread_rng();
        let path = chain.simulate("A", 10, &mut rng);
        
        assert_eq!(path[0], "A");
        assert_eq!(path[1], "B");
        assert_eq!(path[2], "A");
    }
    
    #[test]
    fn test_stationary_distribution() {
        // Simple two-state chain with known stationary distribution
        let chain = StateChainBuilder::new()
            .transition("A", "A", 0.7)
            .transition("A", "B", 0.3)
            .transition("B", "A", 0.4)
            .transition("B", "B", 0.6)
            .build();
        
        let dist = chain.stationary_distribution(100);
        
        // Expected: P(A) ≈ 0.571, P(B) ≈ 0.429
        let p_a = *dist.get("A").unwrap_or(&0.0);
        let p_b = *dist.get("B").unwrap_or(&0.0);
        
        assert!((p_a - 0.571).abs() < 0.05);
        assert!((p_b - 0.429).abs() < 0.05);
    }
}
