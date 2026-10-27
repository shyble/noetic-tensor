//! Losses. Every loss is per seed, `[S]`: summing the result gives each seed its
//! own gradient, since the seeds' losses do not interact. The operation order is fixed
//! (`log_softmax`, one-hot product, negated sum).

use crate::tensor::{log_softmax, IntTensor, Tensor};

/// Per-token negative log-likelihood of `targets` under `logits`: `[S, B, T, V]` and `[S, B, T]`
/// → `[S, B·T]`.
pub fn nll(logits: Tensor, targets: &IntTensor) -> Tensor {
    let [s, b, t, v] = logits.dims();
    let n = b * t;
    let logp = log_softmax(logits, 3).reshape([s, n, v]);
    let onehot = targets.clone().reshape([s, n]).one_hot_float(v, logp.dtype(), logp.device());
    -(logp * onehot).sum_dim(2).reshape([s, n])
}

/// Per-seed mean cross-entropy over every position, `[S]`.
pub fn cross_entropy(logits: Tensor, targets: &IntTensor) -> Tensor {
    let s = logits.shape()[0];
    nll(logits, targets).mean_dim(1).reshape([s])
}

/// Per-seed mean cross-entropy over the masked-in positions, `[S]`: `mask` (`[S, B, T]`, 0 or 1)
/// selects the scored positions, and the mean is `sum(nll · mask) /
/// max(count, 1)`, so a seed with no scored position has loss 0.
pub fn masked_cross_entropy(logits: Tensor, targets: &IntTensor, mask: &Tensor) -> Tensor {
    let [s, b, t, _] = logits.dims();
    let n = b * t;
    let nll = nll(logits, targets);
    let mask = mask.clone().reshape([s, n]);
    let count = mask.clone().sum_dim(1).clamp_min(1.0).reshape([s]);
    (nll * mask).sum_dim(1).reshape([s]) / count
}

/// Options of `cross_entropy_with`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct CeOptions {
    /// Targets equal to this id are not scored (as if masked out).
    pub ignore_index: Option<i64>,
    /// Label smoothing ε: the loss is `(1 − ε)·nll + ε·mean_v(−log p_v)`.
    pub smoothing: f64,
}

/// Per-seed mean cross-entropy with an optional mask `[S, B, T]`, `ignore_index` and label
/// smoothing, `[S]`: `sum(loss · mask) / max(count, 1)` over the scored positions. With default
/// options it is `masked_cross_entropy` (with a mask) or `cross_entropy` (without), to the bit.
pub fn cross_entropy_with(logits: Tensor, targets: &IntTensor, mask: Option<&Tensor>, opts: CeOptions) -> Tensor {
    if opts == CeOptions::default() {
        return match mask {
            Some(m) => masked_cross_entropy(logits, targets, m),
            None => cross_entropy(logits, targets),
        };
    }
    assert!((0.0..=1.0).contains(&opts.smoothing), "label smoothing {} outside [0, 1]", opts.smoothing);
    let [s, b, t, v] = logits.dims();
    let n = b * t;
    let dtype = logits.dtype();
    // The scored positions: the mask times "not ignored"; ignored targets read class 0.
    let tv = targets.to_vec();
    let mut keep: Vec<f64> = match mask {
        Some(m) => m.to_vec_f64(),
        None => vec![1.0; s * n],
    };
    let tv: Vec<i64> = tv.iter().zip(keep.iter_mut()).map(|(&x, k)| if Some(x) == opts.ignore_index { *k = 0.0; 0 } else { x }).collect();
    let logp = log_softmax(logits, 3).reshape([s, n, v]);
    let onehot = IntTensor::from_data(tv, [s, n]).one_hot_float(v, dtype, logp.device());
    let nll = -(logp.clone() * onehot).sum_dim(2).reshape([s, n]);
    let loss = if opts.smoothing > 0.0 {
        let uniform = -logp.mean_dim(2).reshape([s, n]);
        nll.mul_scalar(1.0 - opts.smoothing) + uniform.mul_scalar(opts.smoothing)
    } else {
        nll
    };
    let keep = Tensor::from_f64s(keep, [s, n], dtype).to(loss.device());
    let count = keep.clone().sum_dim(1).clamp_min(1.0).reshape([s]);
    (loss * keep).sum_dim(1).reshape([s]) / count
}

/// Per-seed mean squared error over every element but the seed axis, `[S]`.
pub fn mse(pred: Tensor, target: &Tensor) -> Tensor {
    let s = pred.shape()[0];
    let n = pred.numel() / s;
    let d = pred - target.clone();
    (d.clone() * d).reshape([s, n]).mean_dim(1).reshape([s])
}

/// Per-seed mean KL(P ‖ Q) over positions, `[S]`, with P = softmax(target_logits) and
/// Q = softmax(pred_logits) along the last axis: `Σ_v P_v (log P_v − log Q_v)`. Gradients flow to
/// both; detach the target for a fixed teacher.
pub fn kl_div(pred_logits: Tensor, target_logits: Tensor) -> Tensor {
    let r = pred_logits.rank();
    let s = pred_logits.shape()[0];
    let n = pred_logits.numel() / s / pred_logits.shape()[r - 1];
    let logq = log_softmax(pred_logits, r - 1);
    let logp = log_softmax(target_logits, r - 1);
    let per = (logp.clone().exp() * (logp - logq)).sum_dim(r - 1);
    per.reshape([s, n]).mean_dim(1).reshape([s])
}

/// Weights and constants of `vicreg` (Bardes, Ponce and LeCun 2022). The default is the paper's:
/// invariance λ = 25, variance μ = 25, covariance ν = 1, the hinge target γ = 1 and ε = 1e-4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VicregWeights {
    pub invariance: f64,
    pub variance: f64,
    pub covariance: f64,
    /// The standard deviation each dimension is pushed up to.
    pub gamma: f64,
    /// Added to the variance under the square root.
    pub eps: f64,
}

impl Default for VicregWeights {
    fn default() -> Self {
        VicregWeights { invariance: 25.0, variance: 25.0, covariance: 1.0, gamma: 1.0, eps: 1e-4 }
    }
}

/// The VICReg loss and its terms, each per seed `[S]`.
#[derive(Clone, Debug)]
pub struct Vicreg {
    /// λ·invariance + μ·variance + ν·covariance.
    pub loss: Tensor,
    /// The mean squared difference of the two branches.
    pub invariance: Tensor,
    /// The mean of the two branches' variance hinges.
    pub variance: Tensor,
    /// The sum of the two branches' covariance terms.
    pub covariance: Tensor,
}

/// The variance hinge of `z` `[S, N, D]` per seed, `[S]`: `mean_j max(0, γ − sqrt(Var_j + ε))`
/// with the unbiased variance over the N rows (N ≥ 2).
pub fn vicreg_variance(z: &Tensor, gamma: f64, eps: f64) -> Tensor {
    let [s, n, _] = z.dims();
    assert!(n >= 2, "VICReg: {n} rows (at least 2)");
    let zc = z.clone() - z.clone().mean_dim(1);
    let var = (zc.clone() * zc).sum_dim(1).div_scalar((n - 1) as f64); // [S, 1, D]
    let std = var.add_scalar(eps).sqrt();
    std.neg().add_scalar(gamma).clamp_min(0.0).mean_dim(2).reshape([s])
}

/// The covariance term of `z` `[S, N, D]` per seed, `[S]`: the sum of the squared off-diagonal
/// entries of the unbiased covariance `Zcᵀ·Zc / (N − 1)`, divided by D.
pub fn vicreg_covariance(z: &Tensor) -> Tensor {
    let [s, n, d] = z.dims();
    assert!(n >= 2, "VICReg: {n} rows (at least 2)");
    let zc = z.clone() - z.clone().mean_dim(1);
    let cov = zc.clone().swap_dims(1, 2).matmul(zc).div_scalar((n - 1) as f64); // [S, D, D]
    let off: Vec<f64> = (0..d * d).map(|i| if i / d == i % d { 0.0 } else { 1.0 }).collect();
    let off = Tensor::from_f64s(off, [1, d, d], cov.dtype()).to(cov.device());
    (cov.clone() * cov * off).reshape([s, d * d]).sum_dim(1).reshape([s]).div_scalar(d as f64)
}

/// VICReg on two branches `za`, `zb` `[S, N, D]` (N rows each, row i of one the pair of row i of
/// the other), per seed: `λ·mse(za, zb) + μ·(v(za) + v(zb))/2 + ν·(c(za) + c(zb))`, the
/// reference implementation's grouping (the variance hinges averaged, the covariance terms
/// summed). Gradients flow to both branches; detach one for a target branch.
pub fn vicreg(za: Tensor, zb: Tensor, w: VicregWeights) -> Vicreg {
    assert_eq!(za.shape(), zb.shape(), "VICReg: branch shapes");
    assert_eq!(za.rank(), 3, "VICReg: [S, N, D] inputs");
    let variance = (vicreg_variance(&za, w.gamma, w.eps) + vicreg_variance(&zb, w.gamma, w.eps)).div_scalar(2.0);
    let covariance = vicreg_covariance(&za) + vicreg_covariance(&zb);
    let invariance = mse(za, &zb);
    let loss = invariance.clone().mul_scalar(w.invariance) + variance.clone().mul_scalar(w.variance) + covariance.clone().mul_scalar(w.covariance);
    Vicreg { loss, invariance, variance, covariance }
}
