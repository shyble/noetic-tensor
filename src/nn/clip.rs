//! Gradient clipping per seed: each seed's gradients are clipped by that
//! seed's own statistics, because a global norm would couple the seeds (seed i's step would
//! depend on the others). Norms are accumulated on the host in f64, var by var in the map's
//! order and element by element in row-major order.

use crate::tensor::Tensor;

/// The L2 norm of each seed's gradients over every var (None slots are skipped).
pub fn grad_norms_per_seed(grads: &[Option<Tensor>]) -> Vec<f64> {
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
    for g in grads.iter_mut().flatten() {
        let mut shape = vec![1; g.rank()];
        shape[0] = factors.len();
        let f = Tensor::from_f64s(factors.clone(), shape, g.dtype()).to(g.device());
        *g = g.clone() * f;
    }
    norms
}

/// Clamp every gradient element to [−v, v] (elementwise, so per seed by construction).
pub fn clip_grad_value(grads: &mut [Option<Tensor>], v: f64) {
    for g in grads.iter_mut().flatten() {
        *g = g.clone().clamp(-v, v);
    }
}
