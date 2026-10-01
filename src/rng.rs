//! Every random source is a seeded ChaCha stream derived from a root seed and a fixed label, so
//! the same root seed fixes all of them.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

/// An independent generator from a root seed and a label.
pub fn derive(root: u64, label: &str) -> ChaCha20Rng {
    let mut h = Sha256::new();
    h.update(root.to_le_bytes());
    h.update(label.as_bytes());
    let digest = h.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest[..32]);
    ChaCha20Rng::from_seed(seed)
}
