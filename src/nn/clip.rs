//! Gradient clipping per seed: each seed's gradients are clipped by that
//! seed's own statistics, because a global norm would couple the seeds (seed i's step would
//! depend on the others). Norms are accumulated on the host in f64, var by var in the map's
//! order and element by element in row-major order.
//!
//! The gradients reach the host in **one** download: every gradient is
//! viewed as `[S, n_v]` and concatenated along its second axis on its device, so row i holds seed
//! i's elements var by var, each var's in row-major order; the per-var partial sums and their
//! accumulation are the same f64 operations in the same order as one download per var, and the
//! factors go up once as `[S]` and are viewed per var. Bit for bit the per-var path's
//! (`grad_norms_per_seed_by_var`, kept as the reference; test `one_download_equals_one_per_var`,
//! and the parameter trace `examples/perf_trace.rs` on CpuRef, Metal and CUDA).

use crate::tensor::Tensor;

/// The L2 norm of each seed's gradients over every var (None slots are skipped).
pub fn grad_norms_per_seed(grads: &[Option<Tensor>]) -> Vec<f64> {
    let present: Vec<&Tensor> = grads.iter().flatten().collect();
    let Some(first) = present.first() else { return vec![] };
    let s = first.shape()[0];
    for g in &present { assert_eq!(g.shape()[0], s, "every gradient has the same seed count"); }
    // Mixed dtypes cannot share one buffer: one download per var, as before.
    if present.iter().any(|g| g.dtype() != first.dtype()) || s == 0 {
        return grad_norms_per_seed_by_var(grads);
    }
    let pers: Vec<usize> = present.iter().map(|g| g.shape().iter().product::<usize>() / s).collect();
    let flat: Vec<Tensor> = present.iter().zip(&pers).map(|(g, p)| (*g).clone().reshape([s, *p])).collect();
    let v = if flat.len() == 1 { flat.into_iter().next().unwrap() } else { Tensor::cat(flat, 1) }.to_vec_f64();
    let row: usize = pers.iter().sum();
    let mut sq = vec![0.0f64; s];
    let mut at = 0;
    for per in &pers {
        for (i, acc) in sq.iter_mut().enumerate() {
            let base = i * row + at;
            *acc += v[base..base + per].iter().map(|x| x * x).sum::<f64>();
        }
        at += per;
    }
    sq.into_iter().map(f64::sqrt).collect()
}

/// The reference: one download per var (the slower path, kept as the reference).
pub fn grad_norms_per_seed_by_var(grads: &[Option<Tensor>]) -> Vec<f64> {
    let s = grads.iter().flatten().next().map_or(0, |g| g.shape()[0]);
    let mut sq = vec![0.0f64; s];
    for g in grads.iter().flatten() {
        assert_eq!(g.shape()[0], s, "every gradient has the same seed count");
        let v = g.to_vec_f64();
        let per = v.len() / s;
        for (i, acc) in sq.iter_mut().enumerate() {
            *acc += v[i * per..(i + 1) * per].iter().map(|x| x * x).sum::<f64>();
        }
    }
    sq.into_iter().map(f64::sqrt).collect()
}

/// Scale each seed's gradients by `min(1, max_norm / (norm + 1e-6))` (torch's
/// `clip_grad_norm_`, per seed); returns the norms before clipping. A seed within the bound is
/// multiplied by exactly 1.
pub fn clip_grad_norm_per_seed(grads: &mut [Option<Tensor>], max_norm: f64) -> Vec<f64> {
    let norms = grad_norms_per_seed(grads);
    let factors: Vec<f64> = norms.iter().map(|n| if *n > max_norm { max_norm / (n + 1e-6) } else { 1.0 }).collect();
    if factors.iter().all(|f| *f == 1.0) {
        return norms;
    }
    // The factors go up once per (dtype, device) as `[S]`, then are viewed per var's rank.
    let mut up: Vec<((crate::tensor::DType, crate::tensor::Device), Tensor)> = vec![];
    for g in grads.iter_mut().flatten() {
        let key = (g.dtype(), g.device());
        let f = match up.iter().find(|(k, _)| *k == key) {
            Some((_, t)) => t.clone(),
            None => {
                let t = Tensor::from_f64s(factors.clone(), vec![factors.len()], g.dtype()).to(g.device());
                up.push((key, t.clone()));
                t
            }
        };
        *g = g.clone() * seed_view(f, g.rank());
    }
    norms
}

/// `[S]` viewed as `[S, 1, …, 1]` of `rank` dimensions.
fn seed_view(f: Tensor, rank: usize) -> Tensor {
    let s = f.shape()[0];
    match rank {
        1 => f,
        2 => f.reshape([s, 1]),
        3 => f.reshape([s, 1, 1]),
        4 => f.reshape([s, 1, 1, 1]),
        5 => f.reshape([s, 1, 1, 1, 1]),
        6 => f.reshape([s, 1, 1, 1, 1, 1]),
        r => panic!("a gradient of rank {r}"),
    }
}

/// Clamp every gradient element to [−v, v] (elementwise, so per seed by construction).
pub fn clip_grad_value(grads: &mut [Option<Tensor>], v: f64) {
    for g in grads.iter_mut().flatten() {
        *g = g.clone().clamp(-v, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// The slower clip: one download per var, one factor upload per var.
    fn clip_by_var(grads: &mut [Option<Tensor>], max_norm: f64) -> Vec<f64> {
        let norms = grad_norms_per_seed_by_var(grads);
        let factors: Vec<f64> = norms.iter().map(|n| if *n > max_norm { max_norm / (n + 1e-6) } else { 1.0 }).collect();
        if factors.iter().all(|f| *f == 1.0) { return norms; }
        for g in grads.iter_mut().flatten() {
            let mut shape = vec![1; g.rank()];
            shape[0] = factors.len();
            let f = Tensor::from_f64s(factors.clone(), shape, g.dtype()).to(g.device());
            *g = g.clone() * f;
        }
        norms
    }

    /// The one-download norms and the clipped gradients equal the per-var path's
    /// bit for bit (ranks 2–4, a None slot, binding and non-binding bounds).
    #[test]
    fn one_download_equals_one_per_var() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(36);
        let shapes: [&[usize]; 5] = [&[3, 7, 5], &[3, 11], &[3, 2, 3, 4], &[3, 1, 9], &[3, 64, 16]];
        let grads: Vec<Option<Tensor>> = shapes.iter().enumerate().map(|(i, s)| {
            if i == 3 { return None; }
            let n: usize = s.iter().product();
            Some(Tensor::from_data((0..n).map(|_| rng.gen_range(-2.0f32..2.0) * if i == 1 { 1e-3 } else { 1.0 }).collect(), s.to_vec()))
        }).collect();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&grad_norms_per_seed(&grads)), bits(&grad_norms_per_seed_by_var(&grads)));
        for max in [0.5, 3.0, 1e9] {
            let (mut a, mut b) = (grads.clone(), grads.clone());
            assert_eq!(bits(&clip_grad_norm_per_seed(&mut a, max)), bits(&clip_by_var(&mut b, max)));
            for (x, y) in a.iter().zip(&b) {
                match (x, y) {
                    (Some(x), Some(y)) => {
                        assert_eq!(x.shape(), y.shape());
                        assert_eq!(x.to_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), y.to_vec().iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    }
                    (None, None) => {}
                    _ => panic!("a slot changed"),
                }
            }
        }
        assert!(grad_norms_per_seed(&[None, None]).is_empty());
    }
}
