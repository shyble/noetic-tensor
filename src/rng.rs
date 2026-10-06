//! Seeded random streams: each is a ChaCha stream derived from a root seed and a fixed label,
//! so the same root and label always give the same stream, bit for bit.

use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use sha2::{Digest, Sha256};

/// Derive an independent generator from a root seed and a label.
pub fn derive(root: u64, label: &str) -> ChaCha20Rng {
    let mut h = Sha256::new();
    h.update(root.to_le_bytes());
    h.update(label.as_bytes());
    let digest = h.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest[..32]);
    ChaCha20Rng::from_seed(seed)
}
