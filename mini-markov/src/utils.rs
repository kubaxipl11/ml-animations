//! Utility functions for Markov chains

use std::collections::HashMap;

/// Calculates the Kullback-Leibler divergence between two probability distributions.
/// 
/// KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
/// 
/// # Arguments
/// * `p` - The first distribution (reference)
/// * `q` - The second distribution (approximation)
/// 
/// # Returns
/// The KL divergence (always >= 0, lower is more similar)
pub fn kl_divergence(p: &HashMap<String, f64>, q: &HashMap<String, f64>) -> f64 {
    let epsilon = 1e-10;
    let mut divergence = 0.0;
    
    for (key, &p_val) in p {
        let q_val = q.get(key).unwrap_or(&epsilon);
        if p_val > epsilon {
            divergence += p_val * (p_val / q_val.max(epsilon)).ln();
        }
    }
    
    divergence
}

/// Calculates the Jensen-Shannon divergence (symmetric version of KL).
/// 
/// JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
/// where M = 0.5 * (P + Q)
pub fn js_divergence(p: &HashMap<String, f64>, q: &HashMap<String, f64>) -> f64 {
    // Create M = (P + Q) / 2
    let mut m: HashMap<String, f64> = HashMap::new();
    
    for (key, val) in p {
        *m.entry(key.clone()).or_insert(0.0) += val / 2.0;
    }
    for (key, val) in q {
        *m.entry(key.clone()).or_insert(0.0) += val / 2.0;
    }
    
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Calculates perplexity of a sequence given a model.
/// 
/// Perplexity = 2^(-1/N * Σ log2(P(xi | xi-1)))
/// 
/// Lower perplexity means the model predicts the sequence better.
pub fn perplexity<F>(sequence: &[String], prob_fn: F) -> f64
where
    F: Fn(&str, &str) -> f64,
{
    if sequence.len() < 2 {
        return f64::INFINITY;
    }
    
    let epsilon = 1e-10;
    let mut log_prob_sum = 0.0;
    
    for i in 1..sequence.len() {
        let prev = &sequence[i - 1];
        let curr = &sequence[i];
        let prob = prob_fn(prev, curr).max(epsilon);
        log_prob_sum += prob.log2();
    }
    
    let n = (sequence.len() - 1) as f64;
    2.0_f64.powf(-log_prob_sum / n)
}

/// Normalizes a frequency distribution to probabilities.
pub fn normalize_distribution(counts: &HashMap<String, usize>) -> HashMap<String, f64> {
    let total: usize = counts.values().sum();
    if total == 0 {
        return HashMap::new();
    }
    
    counts
        .iter()
        .map(|(k, v)| (k.clone(), *v as f64 / total as f64))
        .collect()
}

/// Samples from a probability distribution.
pub fn sample_from_distribution<R: rand::Rng>(
    dist: &HashMap<String, f64>,
    rng: &mut R,
) -> Option<String> {
    if dist.is_empty() {
        return None;
    }
    
    let total: f64 = dist.values().sum();
    let mut threshold = rng.gen::<f64>() * total;
    
    for (key, &prob) in dist {
        threshold -= prob;
        if threshold <= 0.0 {
            return Some(key.clone());
        }
    }
    
    dist.keys().next().cloned()
}

/// Formats a probability as a percentage string.
pub fn format_probability(p: f64) -> String {
    format!("{:.1}%", p * 100.0)
}

/// Creates a simple ASCII histogram.
pub fn ascii_histogram(dist: &HashMap<String, f64>, width: usize) -> String {
    let mut result = String::new();
    
    // Sort by probability descending
    let mut items: Vec<_> = dist.iter().collect();
    items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    
    let max_key_len = items.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    
    for (key, &prob) in items {
        let bar_len = (prob * width as f64) as usize;
        let bar: String = "█".repeat(bar_len);
        result.push_str(&format!(
            "{:>width$} │{} {:.1}%\n",
            key,
            bar,
            prob * 100.0,
            width = max_key_len
        ));
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kl_divergence() {
        let p: HashMap<String, f64> = [("A".to_string(), 0.5), ("B".to_string(), 0.5)]
            .into_iter()
            .collect();
        
        // KL(P || P) should be 0
        let kl_same = kl_divergence(&p, &p);
        assert!(kl_same.abs() < 0.001);
        
        // KL with different distribution should be > 0
        let q: HashMap<String, f64> = [("A".to_string(), 0.9), ("B".to_string(), 0.1)]
            .into_iter()
            .collect();
        let kl_diff = kl_divergence(&p, &q);
        assert!(kl_diff > 0.0);
    }
    
    #[test]
    fn test_js_divergence() {
        let p: HashMap<String, f64> = [("A".to_string(), 0.5), ("B".to_string(), 0.5)]
            .into_iter()
            .collect();
        
        // JS is symmetric
        let q: HashMap<String, f64> = [("A".to_string(), 0.9), ("B".to_string(), 0.1)]
            .into_iter()
            .collect();
        
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);
        
        assert!((js_pq - js_qp).abs() < 0.001);
    }
    
    #[test]
    fn test_normalize() {
        let counts: HashMap<String, usize> = [
            ("A".to_string(), 3),
            ("B".to_string(), 1),
        ]
        .into_iter()
        .collect();
        
        let probs = normalize_distribution(&counts);
        assert!((probs.get("A").unwrap() - 0.75).abs() < 0.001);
        assert!((probs.get("B").unwrap() - 0.25).abs() < 0.001);
    }
}
