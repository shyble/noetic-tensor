// The reduced-precision product (`MatmulPrecision::Half`): one set of bodies for every GPU
// backend (bf16 operands on CUDA's tensor cores, f16 on Metal's simdgroup matrices; f32
// accumulation). Its results are compared with a stated tolerance, never byte for byte with f32.
// - The error on fixed shapes (square, ragged edges, long k, transposed views, broadcast batches),
//   against two f64 references: the product of the 16-bit-rounded operands (what the kernel
//   computes, up to f32 accumulation) and the product of the f32 operands.
// - Run to run: the same products, forward and backward, give the same bits twice in a process
//   (across processes and a restart: doc/gates/c1/).
// - The option: off by default; the key is unchanged while it is off and gains the product's
//   part while it is on; it is per thread; a CPU tensor refuses it.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::tensor::tests::*;
use crate::tensor::half::{BF16, F16};
use crate::tensor::{self as nt, MatmulPrecision, Tensor};

fn ht(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_data(rnd(numel(shape), seed, -2.0, 2.0), shape.to_vec())
}

/// The backend's 16-bit operand type: (rounding, unit roundoff, name).
fn operand() -> (fn(f32) -> f32, f64, &'static str) {
    if TAG == "cuda" {
        (|x| BF16::from_f32(x).to_f32(), 2f64.powi(-8), "bf16")
    } else {
        (|x| F16::from_f32(x).to_f32(), 2f64.powi(-11), "f16")
    }
}

/// Row-major [batch, m, k] and [batch, k, n] values (contiguous copies of the views), and the
/// f64 products of `round`-ed operands with Σ|a||b| per output.
fn reference(a: &[f32], b: &[f32], nb: usize, m: usize, k: usize, n: usize, round: impl Fn(f32) -> f32) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; nb * m * n];
    let mut s = vec![0f64; nb * m * n];
    for z in 0..nb {
        for i in 0..m {
            for j in 0..n {
                let (mut acc, mut abs) = (0f64, 0f64);
                for l in 0..k {
                    let x = round(a[z * m * k + i * k + l]) as f64 * round(b[z * k * n + l * n + j]) as f64;
                    acc += x;
                    abs += x.abs();
                }
                v[z * m * n + i * n + j] = acc;
                s[z * m * n + i * n + j] = abs;
            }
        }
    }
    (v, s)
}

/// The largest |got − want| / Σ|a||b| over the outputs.
fn worst(got: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    got.iter().zip(want).zip(scale).map(|((g, w), s)| (*g as f64 - w).abs() / s.max(f64::MIN_POSITIVE)).fold(0.0, f64::max)
}

type Case = (&'static str, Vec<usize>, Vec<usize>, fn(&Tensor, &Tensor, MatmulPrecision) -> Tensor);

fn cases() -> Vec<Case> {
    vec![
        ("[1,64,64]x[1,64,64]", vec![1, 64, 64], vec![1, 64, 64], |a, b, p| a.clone().matmul_with(b.clone(), p)),
        ("[1,256,256]x[1,256,256]", vec![1, 256, 256], vec![1, 256, 256], |a, b, p| a.clone().matmul_with(b.clone(), p)),
        ("[2,130,300]x[2,300,257] ragged", vec![2, 130, 300], vec![2, 300, 257], |a, b, p| a.clone().matmul_with(b.clone(), p)),
        ("[1,7,5]x[1,5,3] tiny", vec![1, 7, 5], vec![1, 5, 3], |a, b, p| a.clone().matmul_with(b.clone(), p)),
        ("[1,64,2048]x[1,2048,64] long k", vec![1, 64, 2048], vec![1, 2048, 64], |a, b, p| a.clone().matmul_with(b.clone(), p)),
        ("[3,100,96]x[1,96,70] broadcast", vec![3, 100, 96], vec![1, 96, 70], |a, b, p| a.clone().matmul_with(b.clone(), p)),
        ("Aᵀ·Bᵀ views [2,150,300],[2,260,150]", vec![2, 150, 300], vec![2, 260, 150], |a, b, p| a.clone().swap_dims(1, 2).matmul_with(b.clone().swap_dims(1, 2), p)),
        ("[1,512,512]x[1,512,512]", vec![1, 512, 512], vec![1, 512, 512], |a, b, p| a.clone().matmul_with(b.clone(), p)),
    ]
}

/// The error on fixed shapes, as a tolerance: every output within 2u + 2⁻²¹·k of Σ|a||b| of the
/// f32 operands' exact product (u the 16-bit type's unit roundoff: each operand's rounding is
/// at most u, and f32 accumulation over k adds at most about k·2⁻²⁴), and within 2⁻²¹·k of Σ|a||b|
/// of the rounded operands' exact product (the accumulation alone).
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_half_matmul_error_against_f32_on_fixed_shapes() {
    let (round, u, name) = operand();
    for (i, (label, sa, sb, f)) in cases().into_iter().enumerate() {
        let (a, b) = (ht(&sa, 80 + i as u64), ht(&sb, 90 + i as u64));
        let half = f(&a.clone().to(M), &b.clone().to(M), MatmulPrecision::Half);
        let shape = half.shape().to_vec();
        let half = half.to_vec();
        let f32gpu = f(&a.clone().to(M), &b.clone().to(M), MatmulPrecision::F32).to_vec();
        // The operands as the product reads them (views made contiguous), batch broadcast.
        let (nb, m, n) = (shape[0], shape[1], shape[2]);
        let (av, bv) = {
            let (aa, bb) = if label.starts_with("Aᵀ") { (a.clone().swap_dims(1, 2), b.clone().swap_dims(1, 2)) } else { (a.clone(), b.clone()) };
            let k = aa.shape()[2];
            let aa = if aa.shape()[0] == nb { aa } else { aa.expand([nb, m, k]) };
            let bb = if bb.shape()[0] == nb { bb } else { bb.expand([nb, k, n]) };
            (aa.to_vec(), bb.to_vec())
        };
        let k = av.len() / (nb * m);
        let (want16, s16) = reference(&av, &bv, nb, m, k, n, &round);
        let (want32, s32) = reference(&av, &bv, nb, m, k, n, |x| x);
        let e16 = worst(&half, &want16, &s16);
        let e32 = worst(&half, &want32, &s32);
        let ef = worst(&f32gpu, &want32, &s32);
        let acc_bound = 2f64.powi(-21) * k as f64;
        eprintln!("{TAG} half matmul {label} ({name}, k {k}): largest error / Σ|a||b|: {e16:.3e} against the rounded operands' product (bound {acc_bound:.3e}), {e32:.3e} against the f32 operands' (bound {:.3e}); the f32 path's {ef:.3e}", 2.0 * u + acc_bound);
        assert!(e16 <= acc_bound, "{label}: {e16} against the rounded operands");
        assert!(e32 <= 2.0 * u + acc_bound, "{label}: {e32} against the f32 operands");
        assert!(half.iter().all(|x| x.is_finite()));
    }
}

/// The same products, forward and backward, twice: the same bits (run to run in one process).
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_half_matmul_is_run_to_run_identical() {
    let run = || -> Vec<Vec<u32>> {
        let mut out = vec![];
        for (i, (_, sa, sb, f)) in cases().into_iter().enumerate() {
            let (a, b) = (ht(&sa, 80 + i as u64).to(M), ht(&sb, 90 + i as u64).to(M));
            out.push(f(&a, &b, MatmulPrecision::Half).to_vec().iter().map(|x| x.to_bits()).collect());
        }
        // Backward: a two-product chain's gradients.
        let w1 = ht(&[1, 96, 80], 7).to(M).require_grad();
        let w2 = ht(&[1, 80, 40], 8).to(M).require_grad();
        let x = ht(&[1, 33, 96], 9).to(M);
        let y = x.matmul_with(w1.clone(), MatmulPrecision::Half).matmul_with(w2.clone(), MatmulPrecision::Half);
        let g = (y.clone() * y).sum().backward();
        out.push(w1.grad(&g).unwrap().to_vec().iter().map(|x| x.to_bits()).collect());
        out.push(w2.grad(&g).unwrap().to_vec().iter().map(|x| x.to_bits()).collect());
        out
    };
    let (first, second) = (run(), run());
    assert_eq!(first, second);
    eprintln!("{TAG} half matmul: {} products and 2 gradients the same bits twice", first.len() - 2);
}

/// Off by default and keyed when on; per thread; refused on the CPU; the thread setting is what
/// `matmul` takes on a GPU tensor.
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_half_precision_is_keyed_and_off_by_default() {
    assert_eq!(nt::matmul_precision(), MatmulPrecision::F32);
    let (a, b) = (ht(&[1, 70, 90], 1).to(M), ht(&[1, 90, 50], 2).to(M));
    let f32_bits: Vec<u32> = a.clone().matmul(b.clone()).to_vec().iter().map(|x| x.to_bits()).collect();
    let key = nt::gpu_platform_key(M).unwrap();
    assert!(!key.contains("gemm-"), "{key}");
    assert_eq!(nt::set_matmul_precision(MatmulPrecision::Half), MatmulPrecision::F32);
    let half_key = nt::gpu_platform_key(M).unwrap();
    let part = nt::half_gemm_key(M).unwrap();
    assert_eq!(half_key, format!("{key}-{part}"));
    assert!(part.starts_with("gemm-") && part.contains("f32acc"), "{part}");
    let by_thread: Vec<u32> = a.clone().matmul(b.clone()).to_vec().iter().map(|x| x.to_bits()).collect();
    let named: Vec<u32> = a.clone().matmul_with(b.clone(), MatmulPrecision::Half).to_vec().iter().map(|x| x.to_bits()).collect();
    assert_eq!(by_thread, named, "the thread's setting is the named precision's product");
    assert_ne!(by_thread, f32_bits, "the reduced-precision product differs from f32 on these values");
    // Another thread keeps f32.
    let other = std::thread::spawn(|| (nt::matmul_precision(), nt::gpu_platform_key(M).unwrap())).join().unwrap();
    assert_eq!(other, (MatmulPrecision::F32, key.clone()));
    // A CPU tensor multiplies in f32 under the thread's setting, and refuses a named Half.
    let (ca, cb) = (ht(&[1, 70, 90], 1), ht(&[1, 90, 50], 2));
    let cpu_bits: Vec<u32> = ca.clone().matmul(cb.clone()).to_vec().iter().map(|x| x.to_bits()).collect();
    assert_eq!(nt::set_matmul_precision(MatmulPrecision::F32), MatmulPrecision::Half);
    assert_eq!(cpu_bits, ca.clone().matmul(cb.clone()).to_vec().iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ca.matmul_with(cb, MatmulPrecision::Half)));
    assert!(refused.is_err(), "a CPU tensor refuses the reduced-precision product");
    // Off again: the key and the f32 product are as before.
    assert_eq!(nt::gpu_platform_key(M).unwrap(), key);
    assert_eq!(a.matmul(b).to_vec().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), f32_bits);
    eprintln!("{TAG} half matmul key part: {}", part.split("-kernels-").next().unwrap());
}

/// The bodies above, in a fresh process.
#[test]
fn half_matmul_tolerance_repeat_and_key() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 3);
}

/// Throughput of f32 against the reduced-precision product on resident data (ignored; median of
/// 21 after a warm-up, synchronised).
#[test]
#[ignore]
fn half_matmul_throughput() {
    let shapes: [(&str, [usize; 4]); 8] = [
        ("1024³", [1, 1024, 1024, 1024]),
        ("2048³", [1, 2048, 2048, 2048]),
        ("4096³", [1, 4096, 4096, 4096]),
        ("512×512×2048", [1, 512, 512, 2048]),
        ("batched 32 × 256×64×256", [32, 256, 64, 256]),
        ("768×3072 × 4096 rows", [1, 4096, 768, 3072]),
        ("Aᵀ·B 512×512×2048", [1, 512, 512, 2048]),
        ("A·Bᵀ 512×2048×512", [1, 512, 2048, 512]),
    ];
    for (name, [nb, m, k, n]) in shapes {
        // Transposed operands are views of a contiguous transpose, as in the backward pass.
        let a = if name.starts_with("Aᵀ") { ht(&[nb, k, m], 1).to(M).swap_dims(1, 2) } else { ht(&[nb, m, k], 1).to(M) };
        let b = if name.contains("Bᵀ") { ht(&[nb, n, k], 2).to(M).swap_dims(1, 2) } else { ht(&[nb, k, n], 2).to(M) };
        let mut line = format!("{TAG} matmul throughput {name}:");
        for p in [MatmulPrecision::F32, MatmulPrecision::Half] {
            for _ in 0..3 {
                drop(a.clone().matmul_with(b.clone(), p));
            }
            nt::synchronize(M).unwrap();
            let reps = if m * n * k * nb >= 1 << 33 { 7 } else { 21 };
            let mut v: Vec<f64> = (0..reps)
                .map(|_| {
                    let t0 = std::time::Instant::now();
                    let y = a.clone().matmul_with(b.clone(), p);
                    nt::synchronize(M).unwrap();
                    drop(y);
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let ms = v[reps / 2];
            line.push_str(&format!(" {} {ms:.3} ms ({:.0} GFLOP/s);", p.name(), 2.0 * (nb * m * k * n) as f64 / (ms * 1e-3) / 1e9));
        }
        eprintln!("{line}");
    }
}
