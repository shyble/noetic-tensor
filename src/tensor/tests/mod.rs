//! Parity tests. The per-op and whole-model tests compare with burn 0.21's outputs
//! (`NdArray<f32>`, `Autodiff<NdArray<f32>>` and `NdArray<f64>`), recorded from burn before it
//! left the dev-dependencies (`fixture`, fixtures/burn021-<os>-<arch>.txt). Every comparison is bit equality
//! (`f32::to_bits`) unless a test says otherwise.

/// The test path of the module it is written in, followed by `$s` (for fresh-process filters).
#[allow(unused_macros)]
macro_rules! here {
    ($s:literal) => {
        format!("{}::{}", module_path!().split_once("::").map_or(module_path!(), |x| x.1), $s)
    };
}

pub(crate) mod fixture;
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
mod gpu;
mod checks;
mod fast;
mod f64ops;
mod fd;
mod ops;
mod standard;

use super::{IntTensor, Tensor};
use rand::{Rng, SeedableRng};

/// Uniform values in [lo, hi) from a seeded stream.
pub(crate) fn rnd(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| r.gen_range(lo..hi)).collect()
}

pub(crate) fn numel(s: &[usize]) -> usize {
    s.iter().product()
}

/// Bit equality of two f32 sequences, with the first mismatch in the message.
pub(crate) fn assert_bits(what: &str, ours: impl AsRef<[f32]>, reference: impl AsRef<[f32]>) {
    let (ours, reference) = (ours.as_ref(), reference.as_ref());
    assert_eq!(ours.len(), reference.len(), "{what}: length {} vs {}", ours.len(), reference.len());
    if let Some(i) = (0..ours.len()).find(|&i| ours[i].to_bits() != reference[i].to_bits()) {
        let diff = ours.iter().zip(reference).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        panic!("{what}: {diff} of {} values differ; first at {i}: {:e} vs reference {:e}", ours.len(), ours[i], reference[i]);
    }
}

/// One-input check against the recorded reference: forward values and the input's gradient of
/// `sum(op(x) · w)`, bit for bit.
pub(crate) fn check1<const DI: usize>(name: &str, shape: [usize; DI], x: &[f32], ours: impl Fn(Tensor) -> Tensor) {
    let xo = Tensor::from_data(x.to_vec(), shape).require_grad();
    let yo = ours(xo.clone());
    fixture::f32s(&format!("{name} forward"), yo.as_slice());
    let osh = yo.shape().to_vec();
    let w = rnd(numel(&osh), 99, -1.0, 1.0);
    let go = (yo * Tensor::from_data(w, osh)).sum().backward();
    fixture::f32s(&format!("{name} gradient"), xo.grad(&go).expect("our gradient").to_vec());
}

/// Two-input check (both tracked): forward and both gradients against the recorded reference.
pub(crate) fn check2<const D1: usize, const D2: usize>(name: &str, s1: [usize; D1], x1: &[f32], s2: [usize; D2], x2: &[f32], ours: impl Fn(Tensor, Tensor) -> Tensor) {
    let (ao, bo) = (Tensor::from_data(x1.to_vec(), s1).require_grad(), Tensor::from_data(x2.to_vec(), s2).require_grad());
    let yo = ours(ao.clone(), bo.clone());
    fixture::f32s(&format!("{name} forward"), yo.as_slice());
    let osh = yo.shape().to_vec();
    let w = rnd(numel(&osh), 98, -1.0, 1.0);
    let go = (yo * Tensor::from_data(w, osh)).sum().backward();
    fixture::f32s(&format!("{name} gradient (lhs)"), ao.grad(&go).unwrap().to_vec());
    fixture::f32s(&format!("{name} gradient (rhs)"), bo.grad(&go).unwrap().to_vec());
}

/// The forward values only, against the recorded reference: for operations whose backward
/// differs from burn's (their gradients are checked in `standard` against f64 and finite
/// differences instead).
pub(crate) fn check1_fwd<const DI: usize>(name: &str, shape: [usize; DI], x: &[f32], ours: impl Fn(Tensor) -> Tensor) {
    fixture::f32s(&format!("{name} forward"), ours(Tensor::from_data(x.to_vec(), shape)).as_slice());
}

/// `check1_fwd` for two inputs.
pub(crate) fn check2_fwd<const D1: usize, const D2: usize>(name: &str, s1: [usize; D1], x1: &[f32], s2: [usize; D2], x2: &[f32], ours: impl Fn(Tensor, Tensor) -> Tensor) {
    fixture::f32s(&format!("{name} forward"), ours(Tensor::from_data(x1.to_vec(), s1), Tensor::from_data(x2.to_vec(), s2)).as_slice());
}

pub(crate) fn our_ints<const D: usize>(v: &[i64], shape: [usize; D]) -> IntTensor {
    IntTensor::from_data(v.to_vec(), shape)
}

/// Run every ignored test whose name contains `filter` in one fresh process, sequentially: the
/// reference pin is process-wide, and a test that pins the shared test process makes Metal and
/// CUDA refuse.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub(crate) fn isolated_bodies(filter: &str, expect: usize) {
    let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args([filter, "--ignored", "--test-threads=1", "--nocapture"])
        .output()
        .expect("run the test binary");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains(&format!("{expect} passed; 0 failed")), "{filter} in a fresh process:\n{text}");
    for l in text.lines().filter(|l| l.starts_with("metal") || l.starts_with("cuda") || l.starts_with("greedy") || l.starts_with("forward")) {
        eprintln!("{l}");
    }
}

