// The raw device, one set of bodies for every GPU backend (included by tests/gpu/mod.rs per
// backend, with `M` the backend's device): its vector add bit-exact against CpuRef, its tiled
// sgemm within |Δ| ≤ 2k·ε·(|A|·|B|) per element and repeatable. The backend's raw entry
// points are `raw_add` and `raw_matmul` of the backend module.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::tensor::tests::*;
use crate::tensor::kernels as k;

/// The raw-device parity bodies, in a fresh process.
#[test]
fn raw_parity() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 2);
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_vector_add_is_bit_exact() {
    // Metal's sizes and value ranges, then CUDA's.
    for n in [1usize, 7, 256, 1000, 1 << 16, (1 << 20) + 3] {
        let (a, b) = (rnd(n, 1, -3.0, 3.0), rnd(n, 2, -3.0, 3.0));
        let g = raw_add(&a, &b);
        let c: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert!(g.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()), "add n {n}");
    }
    for (s, n) in [1usize, 7, 1000, 65_537, 1 << 20, (1 << 22) + 3].into_iter().enumerate() {
        let (a, b) = (rnd(n, 10 + s as u64, -4.0, 4.0), rnd(n, 20 + s as u64, -1e-3, 1e3));
        let g = raw_add(&a, &b);
        let c: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert!(g.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()), "vector add of {n} is not bit-exact");
    }
}

/// Max of |Δ| / bound over the outputs, with bound = 2k·ε·(|A||B|)ᵢⱼ (ε = f32::EPSILON), and
/// the number of outputs equal bit for bit.
fn against_cpu(want: &[f32], got: &[f32], a: &[f32], b: &[f32], batch: usize, m: usize, kk: usize, n: usize) -> (f64, usize) {
    let eps = f32::EPSILON as f64;
    let (mut worst, mut equal) = (0f64, 0usize);
    for t in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut abs = 0f64;
                for l in 0..kk {
                    abs += (a[t * m * kk + i * kk + l] as f64).abs() * (b[t * kk * n + l * n + j] as f64).abs();
                }
                let o = t * m * n + i * n + j;
                let bound = 2.0 * kk as f64 * eps * abs;
                let delta = (want[o] as f64 - got[o] as f64).abs();
                if want[o].to_bits() == got[o].to_bits() {
                    equal += 1;
                } else {
                    worst = worst.max(if bound > 0.0 { delta / bound } else { f64::INFINITY });
                }
            }
        }
    }
    (worst, equal)
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_tiled_sgemm_within_bound() {
    // Metal's shapes (batch 1), then CUDA's (batched).
    let mut shapes: Vec<(usize, usize, usize, usize)> = [(1, 1, 1), (3, 5, 7), (17, 33, 9), (65, 129, 63), (64, 64, 64), (256, 256, 256), (100, 300, 70), (512, 512, 512), (33, 1000, 17), (8, 257, 8)].iter().map(|&(m, kk, n)| (1, m, kk, n)).collect();
    shapes.extend([(1, 1, 1, 1), (1, 7, 13, 5), (1, 65, 33, 129), (3, 37, 50, 29), (1, 64, 64, 64), (32, 64, 64, 64), (1, 256, 256, 256), (1, 300, 1000, 17), (1, 512, 512, 512), (1, 1024, 1024, 1024)]);
    for (s, &(batch, m, kk, n)) in shapes.iter().enumerate() {
        let a = rnd(batch * m * kk, 100 + s as u64, -1.0, 1.0);
        let b = rnd(batch * kk * n, 200 + s as u64, -1.0, 1.0);
        let (want, sh) = k::matmul(&a, &[batch, m, kk], &b, &[batch, kk, n]);
        assert_eq!(sh, vec![batch, m, n]);
        let got = raw_matmul(&a, &b, batch, m, kk, n);
        let again = raw_matmul(&a, &b, batch, m, kk, n);
        assert!(got.iter().zip(&again).all(|(x, y)| x.to_bits() == y.to_bits()), "[{batch},{m},{kk},{n}] is not repeatable");
        let (worst, equal) = against_cpu(&want, &got, &a, &b, batch, m, kk, n);
        eprintln!("{TAG} sgemm [{batch},{m},{kk},{n}]: max |Δ|/bound {worst:.3e}, bit-equal {equal}/{}", want.len());
        assert!(worst <= 1.0, "[{batch},{m},{kk},{n}] exceeds 2k·ε·|A||B| (ratio {worst})");
    }
}
