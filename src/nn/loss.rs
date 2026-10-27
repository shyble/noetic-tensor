//! Losses. Every loss is per seed, `[S]`: summing the result gives each seed its
//! own gradient, since the seeds' losses do not interact. The operation order is fixed
//! (`log_softmax`, one-hot product, negated sum).

use super::attention::PAD_FILL;
use crate::tensor::{log_softmax, BoolTensor, IntTensor, Tensor};

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

/// How a loss reduces its elementwise values: per seed, over every element but the seed axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reduction {
    /// The elementwise losses, in the input's shape.
    None,
    /// The per-seed mean, `[S]` (weights scale the elements; the mean divides by the element
    /// count, not by the weights' sum).
    #[default]
    Mean,
    /// The per-seed sum, `[S]`.
    Sum,
}

/// `loss` `[S, …]` reduced per seed.
fn reduce(loss: Tensor, reduction: Reduction) -> Tensor {
    let s = loss.shape()[0];
    let n = loss.numel() / s;
    match reduction {
        Reduction::None => loss,
        Reduction::Mean => loss.reshape([s, n]).mean_dim(1).reshape([s]),
        Reduction::Sum => loss.reshape([s, n]).sum_dim(1).reshape([s]),
    }
}

/// The weights of `binary_cross_entropy_with_logits`.
#[derive(Clone, Debug, Default)]
pub struct BceWeights {
    /// A per-element weight, the logits' shape or broadcastable to it at the same rank.
    pub weight: Option<Tensor>,
    /// The positive class's weight p: `[C]` along the last axis (C its width), or a tensor of
    /// the logits' rank broadcastable to them. It scales the positive term, `−p·y·log σ(x)`.
    pub pos_weight: Option<Tensor>,
}

/// Binary cross-entropy of `targets` (in [0, 1], the logits' shape) under the logits `x`, in the
/// stable form `max(x, 0) − x·y + log(1 + exp(−|x|))`: no term overflows, so the loss stays
/// finite for any finite x. `max(x, 0)` is computed as `(x + |x|)/2` (equal to it in floating
/// point) so the gradient at x = 0 is σ(0) − y = 1/2 − y, not a one-sided subgradient; elsewhere
/// the gradient is σ(x) − y.
///
/// With a `pos_weight` p, the form is `(1 − y)·x + (1 + (p − 1)·y)·(max(−x, 0) + log(1 +
/// exp(−|x|)))`, i.e. `−p·y·log σ(x) − (1 − y)·log(1 − σ(x))`; p = 1 is the plain loss. A
/// per-element `weight` multiplies the elementwise loss before `reduction`. Gradients flow to
/// the logits and to the targets.
pub fn binary_cross_entropy_with_logits(logits: Tensor, targets: &Tensor, weights: Option<&BceWeights>, reduction: Reduction) -> Tensor {
    assert_eq!(logits.shape(), targets.shape(), "BCE with logits: targets shape");
    let r = logits.rank();
    let y = targets.clone();
    let ax = logits.clone().abs();
    let softplus_neg_abs = ax.clone().neg().exp().add_scalar(1.0).log(); // log(1 + exp(−|x|))
    let pos_weight = weights.and_then(|w| w.pos_weight.clone());
    let loss = match pos_weight {
        None => {
            let relu = (logits.clone() + ax).div_scalar(2.0); // max(x, 0)
            relu - logits * y + softplus_neg_abs
        }
        Some(p) => {
            let p = if p.rank() == 1 && r > 1 {
                let c = logits.shape()[r - 1];
                assert_eq!(p.shape(), &[c], "BCE with logits: pos_weight [C] with C the last axis ({c})");
                let mut sh = vec![1; r];
                sh[r - 1] = c;
                p.reshape_dyn(sh)
            } else {
                assert_eq!(p.rank(), r, "BCE with logits: pos_weight rank");
                p
            };
            let lw = (p.sub_scalar(1.0) * y.clone()).add_scalar(1.0); // 1 + (p − 1)·y
            let relu_neg = (ax - logits.clone()).div_scalar(2.0); // max(−x, 0)
            y.neg().add_scalar(1.0) * logits + lw * (relu_neg + softplus_neg_abs)
        }
    };
    let loss = match weights.and_then(|w| w.weight.clone()) {
        Some(w) => {
            assert_eq!(w.rank(), r, "BCE with logits: weight rank");
            loss * w
        }
        None => loss,
    };
    reduce(loss, reduction)
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

/// Weights and constants of `vicreg` (Bardes, Ponce and LeCun 2022). The default is the reference
/// code's: invariance λ = 25, variance μ = 25, covariance ν = 1, the hinge target γ = 1 and
/// ε = 1e-4. `vicreg` averages the two branches' hinges, as that code does, so its μ = 25 equals
/// μ = 12.5 in the paper's equation 6, `λ·s + μ·[v(Z) + v(Z′)] + ν·[c(Z) + c(Z′)]` (the paper's
/// sum of the two hinges).
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

/// InfoNCE over variable-size groups, per seed `[S]`: group g of seed s scores its candidates
/// `logits[s, g, ..]` (`[S, G, K]`, similarities already divided by the temperature), of which
/// `valid[s, g, k]` (`[S, G, K]`, true = a candidate) are real and the rest padding, and
/// `targets[s, g]` (`[S, G]`) is the positive's index (it must be a valid one). The loss is the
/// masked log-softmax's negative log-likelihood of the positive: padded entries are filled with
/// `PAD_FILL` (−1e9, finite) before `log_softmax`, so they take no probability and get exactly
/// zero gradient. `groups` (`[S, G]`, 0 or 1) optionally drops whole groups: the mean is then
/// over the kept groups (`masked_cross_entropy`'s). With every entry valid and no group mask this
/// is `cross_entropy` on `[S, G, 1, K]`, to the bit.
///
/// Not in F16: −1e9 is beyond its range (it would become −∞), so F16 logits are refused; run
/// padded losses in F32 (or BF16, F64). Debug builds check that every kept group's target is a
/// valid entry (`check_group_targets`, a host read); release builds do not.
pub fn masked_group_info_nce(logits: Tensor, targets: &IntTensor, valid: &BoolTensor, groups: Option<&Tensor>) -> Tensor {
    let [s, g, k] = logits.dims();
    assert_eq!(valid.shape(), &[s, g, k], "group InfoNCE: valid mask shape");
    assert_eq!(targets.shape(), &[s, g], "group InfoNCE: targets shape");
    assert!(logits.dtype() != crate::tensor::DType::F16, "group InfoNCE: padded losses do not run in F16 (the pad fill −1e9 overflows it)");
    #[cfg(debug_assertions)]
    if let Err(e) = check_group_targets(targets, valid, groups) {
        panic!("{e}");
    }
    let l4 = logits.mask_fill(valid.clone().bool_not(), PAD_FILL).reshape([s, g, 1, k]);
    let t3 = targets.clone().reshape([s, g, 1]);
    match groups {
        None => cross_entropy(l4, &t3),
        Some(m) => masked_cross_entropy(l4, &t3, &m.clone().reshape([s, g, 1])),
    }
}

/// Every kept group's target (`groups` 1, or every group without a mask) is in range and a valid
/// entry: the check `masked_group_info_nce` runs in debug builds (it reads the masks to the
/// host). The error names the first bad group.
pub fn check_group_targets(targets: &IntTensor, valid: &BoolTensor, groups: Option<&Tensor>) -> std::result::Result<(), String> {
    let sh = valid.shape().to_vec();
    let (sg, k) = (sh[0] * sh[1], sh[2]);
    let t = targets.to_vec();
    let v = valid.to_vec();
    let keep = groups.map(|m| m.to_vec_f64());
    for i in 0..sg {
        if keep.as_ref().is_some_and(|m| m[i] == 0.0) {
            continue;
        }
        let ti = t[i];
        if ti < 0 || ti as usize >= k || !v[i * k + ti as usize] {
            return Err(format!("group InfoNCE: group {i} (seed {}, group {}) has target {ti}, not a valid entry of {k}", i / sh[1], i % sh[1]));
        }
    }
    Ok(())
}
