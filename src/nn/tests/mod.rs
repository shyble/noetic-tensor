//! Tests of `nn`: naive f64 references per component, f64 finite differences, property tests,
//! the decoder's options, MoE and the trainer.

/// The test path of the module it is written in, followed by `$s` (for fresh-process filters).
#[allow(unused_macros)]
macro_rules! here {
    ($s:literal) => {
        format!("{}::{}", module_path!().split_once("::").map_or(module_path!(), |x| x.1), $s)
    };
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
mod gpu;
#[cfg(feature = "cuda")]
mod cuda_perf;
mod fd;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_perf;
mod components;
mod options;
mod naive;
mod props;

use crate::tensor::{IntTensor, Tensor};
use rand::{Rng, SeedableRng};

/// A seeded stream for test data.
pub(crate) fn rng(seed: u64) -> rand_chacha::ChaCha8Rng {
    rand_chacha::ChaCha8Rng::seed_from_u64(seed)
}

pub(crate) fn rnd(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
    let mut r = rng(seed);
    (0..n).map(|_| r.gen_range(lo..hi)).collect()
}

pub(crate) fn rnd_ints(n: usize, seed: u64, hi: i64) -> Vec<i64> {
    let mut r = rng(seed);
    (0..n).map(|_| r.gen_range(0..hi)).collect()
}




/// Bit equality of two f32 sequences, with the first mismatch in the message.
#[track_caller]
pub(crate) fn assert_bits(what: &str, ours: &[f32], theirs: &[f32]) {
    assert_eq!(ours.len(), theirs.len(), "{what}: length {} vs {}", ours.len(), theirs.len());
    if let Some(i) = (0..ours.len()).find(|&i| ours[i].to_bits() != theirs[i].to_bits()) {
        let diff = ours.iter().zip(theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        panic!("{what}: {diff} of {} values differ; first at {i}: nn {:e} vs reference {:e}", ours.len(), ours[i], theirs[i]);
    }
}

/// |a − b| ≤ tol · (1 + |b|) elementwise.
#[track_caller]
pub(crate) fn assert_close(what: &str, ours: &[f64], theirs: &[f64], tol: f64) {
    assert_eq!(ours.len(), theirs.len(), "{what}: length {} vs {}", ours.len(), theirs.len());
    for (i, (a, b)) in ours.iter().zip(theirs).enumerate() {
        assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{what}[{i}]: {a:e} vs {b:e} (tol {tol:e})");
    }
}

/// A small decoder used across the tests: d 32, 4 heads, context 16, 2 blocks, MLP width 96.
pub(crate) fn small(vocab: usize) -> crate::nn::DecoderConfig {
    crate::nn::DecoderConfig::new(vocab, 32, 4, 16, 2, 96)
}

#[allow(dead_code)] // used by the GPU tests
/// A tiny decoder: d 16, 2 heads, 2 blocks, MLP width 96.
pub(crate) fn tiny(vocab: usize, context: usize) -> crate::nn::DecoderConfig {
    crate::nn::DecoderConfig::new(vocab, 16, 2, context, 2, 96)
}

/// The per-seed mean next-token cross-entropy of `logits` `[S, B, T, V]`: position t ≥ 1
/// predicts token t + 1 and the last position predicts `next` (`[S, B]`); position 0 is not
/// scored.
pub(crate) fn next_token_loss(logits: Tensor, tokens: &IntTensor, next: &IntTensor) -> Tensor {
    let [s, b, t, v] = logits.dims();
    let shifted = IntTensor::cat(vec![tokens.clone().slice([0..s, 0..b, 1..t]), next.clone().reshape([s, b, 1])], 2);
    let targets = shifted.slice([0..s, 0..b, 1..t]);
    crate::nn::cross_entropy(logits.slice([0..s, 0..b, 1..t, 0..v]), &targets)
}
