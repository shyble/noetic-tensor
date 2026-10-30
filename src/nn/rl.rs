//! Building blocks for policy-gradient post-training of sequence decoders whose samples are
//! scored outside the engine: per-token log-probabilities, a policy-gradient loss with signed
//! weights, the clipped-ratio (PPO) objective, the categorical KL to a frozen reference (exact
//! and per sampled token), the categorical entropy, group-relative advantages and masked
//! categorical sampling. The rollout loop, the verifier and the reward design stay with the
//! caller.
//!
//! Shapes follow the rest of `nn`: the leading axis is the seed axis, so logits are
//! `[S, B, T, V]` and per-token values `[S, B, T]` (a single model is S = 1). A mask is a float
//! tensor `[S, B, T]` of 0 (not scored: padding, the prompt) and 1 (scored); `None` scores every
//! position. Each loss returns its per-token values or a per-seed reduction (`TokenReduction`);
//! a masked position's per-token value is exactly 0 and gets exactly zero gradient, provided its
//! inputs are finite. Weights, advantages, old log-probabilities and reference logits are
//! constants: each loss detaches them.
//!
//! Logits may hold −∞ (a token excluded by an additive mask, see `tempered_logits`): the
//! log-probabilities, the KL and the entropy give a token of probability exactly 0 a zero term
//! and a zero gradient instead of `0 · (−∞)`. An additive −1e9 (the decoder's grammar-mask
//! fill) gives the same values: its probability underflows to exactly 0.
//!
//! One masking convention for the caller: build the logits of every distribution that enters a
//! loss with `tempered_logits` — the sampling temperature and the same vocabulary mask (−∞ or
//! −1e9 for disallowed tokens) — for sampling, for the policy's log-probabilities in training,
//! for the old log-probabilities and for the reference. A mask applied to the policy but not to
//! the reference makes the KL +∞ wherever π > 0 and π_ref = 0 (or about 1e9 with −1e9); a
//! sampling mask missing from the training logits leaves the ratio and the log-probabilities
//! inconsistent. Debug builds check that every scored log-probability is finite
//! (`check_scored_finite`).

use crate::tensor::{log_softmax, BoolTensor, IntTensor, Tensor};
use rand::RngCore;
use rand_chacha::ChaCha20Rng;

/// How a per-token loss `[S, B, T]` reduces, per seed. Every count is of scored positions (the
/// mask's sum); no reduction divides by the weights' or the advantages' sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TokenReduction {
    /// The per-token values `[S, B, T]`, 0 at masked positions.
    None,
    /// The per-seed sum over the scored positions, `[S]`.
    Sum,
    /// The per-seed mean over the scored positions, `[S]`: `Σ l·m / max(Σ m, 1)` (every scored
    /// token of the batch counts once; a seed with no scored position gives 0).
    #[default]
    TokenMean,
    /// Each sequence's mean over its scored positions, `[S, B]`: `Σ_t l·m / max(Σ_t m, 1)` (a
    /// sequence with no scored position gives 0).
    PerSequence,
    /// The per-seed mean of `PerSequence` over the sequences with at least one scored position,
    /// `[S]` (each sequence counts once, whatever its length; 0 if there is none). This carries a
    /// length bias: a token of a short sequence weighs more than one of a long sequence, so with
    /// a negative advantage a long response is penalised less per token than a short one.
    /// `TokenMean` weighs every scored token equally.
    SequenceMean,
}

/// The mask as a float tensor of `like`'s shape, dtype and device (ones without a mask), detached.
fn mask_or_ones(mask: Option<&Tensor>, like: &Tensor) -> Tensor {
    match mask {
        Some(m) => {
            assert_eq!(m.shape(), like.shape(), "rl: mask shape");
            m.clone().detach()
        }
        None => Tensor::full_dtype(like.shape().to_vec(), 1.0, like.dtype()).to(like.device()),
    }
}

/// `[S, B, T]` per-token values, already zero at masked positions, reduced per seed.
fn reduce_tokens(per: Tensor, mask: &Tensor, reduction: TokenReduction) -> Tensor {
    let [s, b, t] = per.dims();
    match reduction {
        TokenReduction::None => per,
        TokenReduction::Sum => per.reshape([s, b * t]).sum_dim(1).reshape([s]),
        TokenReduction::TokenMean => {
            let count = mask.clone().reshape([s, b * t]).sum_dim(1).clamp_min(1.0).reshape([s]);
            per.reshape([s, b * t]).sum_dim(1).reshape([s]) / count
        }
        TokenReduction::PerSequence => per_sequence(per, mask),
        TokenReduction::SequenceMean => {
            let seq = per_sequence(per, mask);
            // Sequences with a scored position: min(count, 1) on integer counts.
            let nonempty = mask.clone().sum_dim(2).clamp_max(1.0).reshape([s, b]).sum_dim(1).clamp_min(1.0).reshape([s]);
            seq.sum_dim(1).reshape([s]) / nonempty
        }
    }
}

fn per_sequence(per: Tensor, mask: &Tensor) -> Tensor {
    let [s, b, _] = per.dims();
    let count = mask.clone().sum_dim(2).clamp_min(1.0).reshape([s, b]);
    per.sum_dim(2).reshape([s, b]) / count
}

/// The masked positions of a mask (`m == 0`), for `mask_fill`.
fn masked_out(mask: &Tensor) -> BoolTensor {
    mask.clone().equal_elem(0.0)
}

/// Zero at masked positions (a select, not a product: a non-finite value there does not leak).
fn zero_masked(x: Tensor, mask: &Tensor) -> Tensor {
    x.mask_fill(masked_out(mask), 0.0)
}

/// A per-sequence `[S, B]` or per-token `[S, B, T]` weight as `[S, B, T]`-broadcastable,
/// detached.
fn token_weights(w: &Tensor, like: &Tensor, what: &str) -> Tensor {
    let [s, b, t] = like.dims();
    let w = w.clone().detach();
    match w.rank() {
        2 => {
            assert_eq!(w.shape(), &[s, b], "{what}: per-sequence weights [S, B]");
            w.reshape([s, b, 1])
        }
        3 => {
            assert_eq!(w.shape(), &[s, b, t], "{what}: per-token weights [S, B, T]");
            w
        }
        r => panic!("{what}: weights of rank {r} (per-sequence [S, B] or per-token [S, B, T])"),
    }
}

/// The elementwise larger of `a` and `b` as a select: `a` where `a > b`, else `b` (a tie or a NaN
/// difference takes `b`). The value is exactly one of the two, and the gradient goes to the selected one only.
fn select_max(a: Tensor, b: Tensor) -> Tensor {
    let d = (a.clone() - b.clone()).untracked();
    let a_wins = d.greater_elem(0.0);
    let b_wins = a_wins.clone().bool_not();
    a.mask_fill(b_wins, 0.0) + b.mask_fill(a_wins, 0.0)
}

/// The log-probability of each token under the logits: `log_softmax(logits)[…, tokens[…]]`,
/// logits `[…, V]` and tokens `[…]` → `[…]`. Differentiable in the logits (the max-shifted
/// log-softmax; its gradient is `onehot − softmax`). A token whose logit is −∞ gives −∞.
pub fn token_log_probs(logits: Tensor, tokens: &IntTensor) -> Tensor {
    let r = logits.rank();
    assert!(r >= 1, "token_log_probs: logits of rank 0");
    assert_eq!(&logits.shape()[..r - 1], tokens.shape(), "token_log_probs: tokens shape (the logits' without the vocabulary axis)");
    let out = tokens.shape().to_vec();
    let logp = log_softmax(logits, r - 1);
    logp.gather(r - 1, tokens.clone().unsqueeze_dim(r - 1)).reshape_dyn(out)
}

/// The default bound on the log-ratio `logp − old_logp` in `clipped_policy_loss`
/// (`Clip::new`'s): `exp(20)` ≈ 4.9e8 is finite in f32, and a ratio that far out is clipped (A > 0)
/// or, for A < 0 without a dual clip, would otherwise reach +∞.
pub const LOG_RATIO_CLAMP: f64 = 20.0;

/// The recommended bound on δ in `kl_estimate` (`Some(KL_DELTA_CLAMP)`): `exp(20)` is finite
/// in f32.
pub const KL_DELTA_CLAMP: f64 = 20.0;

/// The clip settings of `clipped_policy_loss`: the ratio is clipped to `[1 − low, 1 + high]`;
/// `dual` (c > 1) caps the loss of a negative-advantage token at `−c·A` (dual-clip PPO; `None`
/// by default); `log_ratio_clamp` bounds `logp − old_logp` to `[−b, b]` before the exponential
/// (`Some(LOG_RATIO_CLAMP)` by default; `None` for the unbounded ratio). Where a bound is active
/// the token gets no gradient. The dual clip bounds the loss's value, not the ratio: with the
/// log-ratio clamp off, a ratio that overflows to +∞ leaves a dual-capped token's value finite
/// but its gradient NaN (zero upstream times the exponential's infinite derivative), so keep the
/// clamp on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clip {
    pub low: f64,
    pub high: f64,
    pub dual: Option<f64>,
    pub log_ratio_clamp: Option<f64>,
}

impl Clip {
    /// `[1 − low, 1 + high]`, no dual clip, the log-ratio bounded by `LOG_RATIO_CLAMP`.
    pub fn new(low: f64, high: f64) -> Clip {
        Clip { low, high, dual: None, log_ratio_clamp: Some(LOG_RATIO_CLAMP) }
    }

    /// `[1 − ε, 1 + ε]`, PPO's, with `new`'s defaults.
    pub fn symmetric(eps: f64) -> Clip {
        Clip::new(eps, eps)
    }

    /// With a dual clip c > 1: a token with A < 0 loses at most `−c·A`.
    pub fn with_dual(self, c: f64) -> Clip {
        Clip { dual: Some(c), ..self }
    }

    /// With another log-ratio bound (`None`: unbounded).
    pub fn with_log_ratio_clamp(self, b: Option<f64>) -> Clip {
        Clip { log_ratio_clamp: b, ..self }
    }

    fn check(self) {
        assert!(self.low >= 0.0 && self.low < 1.0 && self.high >= 0.0 && self.low.is_finite() && self.high.is_finite(), "clip range [1 − {}, 1 + {}] (need 0 ≤ low < 1, 0 ≤ high)", self.low, self.high);
        if let Some(c) = self.dual {
            assert!(c > 1.0 && c.is_finite(), "dual clip {c} (need c > 1)");
        }
        if let Some(b) = self.log_ratio_clamp {
            assert!(b > 0.0, "log-ratio clamp {b} (need b > 0)");
        }
    }
}

/// Every scored log-probability (mask ≠ 0) is finite: the check the losses run in debug builds
/// (a host read); release builds do not. A −∞ there means a scored token the distribution
/// excludes (a sampling mask not applied to the training logits, or a token from elsewhere); the
/// error names the first such position.
pub fn check_scored_finite(what: &str, logp: &Tensor, mask: Option<&Tensor>) -> std::result::Result<(), String> {
    let v = logp.to_vec_f64();
    let m = mask.map(|m| m.to_vec_f64());
    match (0..v.len()).find(|&i| m.as_ref().is_none_or(|m| m[i] != 0.0) && !v[i].is_finite()) {
        Some(i) => Err(format!("{what}: the scored log-probability at flat position {i} is {} (not finite)", v[i])),
        None => Ok(()),
    }
}

#[cfg(debug_assertions)]
fn debug_check(what: &str, logp: &Tensor, mask: Option<&Tensor>) {
    if let Err(e) = check_scored_finite(what, logp, mask) {
        panic!("{e}");
    }
}

#[cfg(not(debug_assertions))]
fn debug_check(_: &str, _: &Tensor, _: Option<&Tensor>) {}

/// The policy-gradient (REINFORCE) loss with signed weights, `l = −w · log π(token)` per scored
/// token: `logp` `[S, B, T]` (from `token_log_probs`), `weights` the advantages or rewards per
/// token `[S, B, T]` or per sequence `[S, B]` (detached; any sign). Minimising it raises the
/// log-probability of positively weighted tokens and lowers that of negatively weighted ones.
/// Debug builds check that the scored log-probabilities are finite (`check_scored_finite`).
pub fn policy_gradient_loss(logp: Tensor, weights: &Tensor, mask: Option<&Tensor>, reduction: TokenReduction) -> Tensor {
    assert_eq!(logp.rank(), 3, "policy_gradient_loss: logp [S, B, T]");
    debug_check("policy_gradient_loss", &logp, mask);
    let m = mask_or_ones(mask, &logp);
    let w = token_weights(weights, &logp, "policy_gradient_loss");
    let per = zero_masked(-(logp * w), &m);
    reduce_tokens(per, &m, reduction)
}

/// The log-ratio `logp − old`, zeroed at masked positions (so arbitrary old values at padding
/// cannot overflow) and bounded by `clamp`.
fn log_ratio(logp: Tensor, old_logp: &Tensor, m: &Tensor, clamp: Option<f64>) -> Tensor {
    assert_eq!(old_logp.shape(), logp.shape(), "rl: old log-probabilities shape");
    let d = zero_masked(logp - old_logp.clone().detach(), m);
    match clamp {
        Some(b) => d.clamp(-b, b),
        None => d,
    }
}

/// Per token: `t` where `pick` is true, else `f` (a select; the gradient goes to the chosen one).
fn select(pick: BoolTensor, t: Tensor, f: Tensor) -> Tensor {
    let not = pick.clone().bool_not();
    t.mask_fill(not, 0.0) + f.mask_fill(pick, 0.0)
}

/// PPO's clipped surrogate as a loss, per scored token: with `r = exp(logp − old_logp)` (the
/// log-ratio bounded by `clip.log_ratio_clamp`), `l = −min(r·A, clip(r, 1 − low, 1 + high)·A)`,
/// computed as `max(−r·A, −clip(r)·A)` by a select (the gradient goes to the active branch; the
/// clipped branch's is zero outside the range); with a dual clip c, a token with A < 0 has
/// `l = min(that, −c·A)`. `old_logp` `[S, B, T]` are the sampling policy's log-probabilities of
/// the same tokens, under the same `tempered_logits` (temperature and mask) as `logp`, and
/// `advantages` `[S, B, T]` or `[S, B]`; both are detached.
///
/// At `logp == old_logp` (on policy: the first update on a fresh batch) the gradient equals
/// `policy_gradient_loss`'s with the advantages as weights; later epochs on the same batch are
/// off policy, and the ratio corrects for that only within the clip range.
pub fn clipped_policy_loss(logp: Tensor, old_logp: &Tensor, advantages: &Tensor, mask: Option<&Tensor>, clip: Clip, reduction: TokenReduction) -> Tensor {
    clip.check();
    assert_eq!(logp.rank(), 3, "clipped_policy_loss: logp [S, B, T]");
    debug_check("clipped_policy_loss", &logp, mask);
    debug_check("clipped_policy_loss (old log-probabilities)", old_logp, mask);
    let m = mask_or_ones(mask, &logp);
    let a = token_weights(advantages, &logp, "clipped_policy_loss");
    let r = log_ratio(logp, old_logp, &m, clip.log_ratio_clamp).exp();
    let unclipped = -(r.clone() * a.clone());
    let clipped = -(r.clamp(1.0 - clip.low, 1.0 + clip.high) * a.clone());
    let ppo = select_max(unclipped, clipped);
    let per = match clip.dual {
        None => ppo,
        Some(c) => {
            let [s, b, t] = ppo.dims();
            let a = a.expand([s, b, t]);
            let cap = a.clone().mul_scalar(-c); // −c·A, positive where A < 0
            let capped = -select_max(-ppo.clone(), -cap); // min(ppo, −c·A)
            select(a.untracked().lower_elem(0.0), capped, ppo)
        }
    };
    reduce_tokens(zero_masked(per, &m), &m, reduction)
}

/// The fraction of scored tokens that `clipped_policy_loss` gives no gradient through a bound,
/// per seed `[S]`: A > 0 with r > 1 + high, A < 0 with r < 1 − low, A < 0 with r > c under a
/// dual clip, or the log-ratio beyond its clamp. Untracked, for logging; a seed with no scored
/// token gives 0.
pub fn clip_fraction(logp: &Tensor, old_logp: &Tensor, advantages: &Tensor, mask: Option<&Tensor>, clip: Clip) -> Tensor {
    clip.check();
    let logp = logp.clone().detach();
    let m = mask_or_ones(mask, &logp);
    let a = token_weights(advantages, &logp, "clip_fraction");
    let d = log_ratio(logp, old_logp, &m, None);
    let r = d.clone().exp();
    let [s, b, t] = r.dims();
    let (dt, dev) = (m.dtype(), m.device());
    let f = |x: BoolTensor| x.float_dtype(dt).to(dev);
    let pos = f(a.clone().greater_elem(0.0));
    let neg = f(a.lower_elem(0.0));
    let up = f(r.clone().greater_elem(1.0 + clip.high));
    let down = f(r.clone().lower_elem(1.0 - clip.low));
    let mut hit = up * pos + down * neg.clone();
    if let Some(c) = clip.dual {
        hit = hit + f(r.greater_elem(c)) * neg;
    }
    if let Some(bound) = clip.log_ratio_clamp {
        hit = hit + f(d.abs().greater_elem(bound));
    }
    let hit = hit.clamp_max(1.0) * m.clone();
    let count = m.reshape([s, b * t]).sum_dim(1).clamp_min(1.0).reshape([s]);
    hit.reshape([s, b * t]).sum_dim(1).reshape([s]) / count
}

/// The exact KL divergence `KL(π ‖ π_ref) = Σ_v π_v (log π_v − log π_ref,v)` per position, with
/// π = softmax(logits) and π_ref = softmax(ref_logits) along the last axis (`[S, B, T, V]`; the
/// reference detached): the reverse KL used as a penalty toward a frozen reference, and the
/// recommended default penalty (exact, no sampling noise, valid on and off policy; its cost is
/// one pass over V per position). A token with π_v = 0 (logit −∞, or an additive −1e9 that
/// underflows) contributes 0; one with π_v > 0 and π_ref,v = 0 makes the KL +∞, so apply the same
/// vocabulary mask to both sides (`reference_kl` does). Compare `nn::kl_div`, the forward KL
/// `KL(target ‖ pred)` with both inputs differentiable.
pub fn categorical_kl(logits: Tensor, ref_logits: &Tensor, mask: Option<&Tensor>, reduction: TokenReduction) -> Tensor {
    assert_eq!(logits.rank(), 4, "categorical_kl: logits [S, B, T, V]");
    assert_eq!(logits.shape(), ref_logits.shape(), "categorical_kl: reference logits shape");
    let [s, b, t, _] = logits.dims();
    let logp = log_softmax(logits, 3);
    let logr = log_softmax(ref_logits.clone().detach(), 3);
    let zero_p = logp.clone().untracked().equal_elem(f64::NEG_INFINITY);
    let diff = (logp.clone() - logr).mask_fill(zero_p, 0.0);
    let per = (logp.exp() * diff).sum_dim(3).reshape([s, b, t]);
    let m = mask_or_ones(mask, &per);
    reduce_tokens(zero_masked(per, &m), &m, reduction)
}

/// The reference penalty under the engine's convention: `categorical_kl` between the policy and
/// the reference **at the sampling temperature, with the same vocabulary mask, on both sides**:
/// `KL(softmax(l/τ + mask) ‖ softmax(l_ref/τ + mask))`, i.e. between the distributions the
/// sampler draws from (`tempered_logits`). Mixing temperatures (τ on one side, 1 on the other)
/// biases the penalty, and a mask on one side only makes it +∞; the per-token estimators take
/// their `ref_logp` the same way (`token_log_probs(tempered_logits(ref_logits, τ, mask), tokens)`).
/// At τ = 1 without a mask this is `categorical_kl`, to the bit.
pub fn reference_kl(logits: Tensor, ref_logits: &Tensor, temperature: f64, vocab_mask: Option<&Tensor>, mask: Option<&Tensor>, reduction: TokenReduction) -> Tensor {
    let p = tempered_logits(logits, temperature, vocab_mask);
    let r = tempered_logits(ref_logits.clone().detach(), temperature, vocab_mask);
    categorical_kl(p, &r, mask, reduction)
}

/// The per-sampled-token estimators of `KL(π ‖ π_ref)` (Schulman's k1, k2, k3), from the
/// sampled tokens' log-probabilities under π and under the reference, with `δ = log π_ref − log
/// π`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KlEstimator {
    /// `−δ`: unbiased for the KL under samples from π; can be negative; high variance.
    K1,
    /// `δ²/2`: biased, non-negative, low variance.
    K2,
    /// `exp(δ) − δ − 1`: unbiased under samples from π, non-negative, low variance.
    K3,
}

/// A per-sampled-token estimate of `KL(π ‖ π_ref)`: `logp` `[S, B, T]` the sampled tokens'
/// log-probabilities under the current policy (tracked) and `ref_logp` under the reference
/// (detached), both under the same `tempered_logits` (see `reference_kl`). δ is zeroed at masked
/// positions before the exponential and, with `delta_clamp = Some(b)`, bounded to `[−b, b]` (no
/// gradient where the bound is active); `Some(KL_DELTA_CLAMP)` is the recommended setting, `None`
/// leaves δ unbounded (k3's `exp(δ)` then overflows to +∞ for δ above about 88 in f32).
///
/// These statements hold **on policy**, for tokens sampled from the current π (the first update
/// on a fresh batch): the value's expectation is the KL for k1 and k3 (k2 is biased); as a loss
/// term the expected per-sample gradient is zero for k1, the gradient of `KL(π ‖ π_ref)` for k2,
/// and the gradient of the forward `KL(π_ref ‖ π)` for k3 (the form GRPO-style trainers put in
/// the loss), not of the reverse KL. After the first PPO epoch the tokens come from an older
/// policy and none of them holds without importance weights. `categorical_kl` (`reference_kl`)
/// is exact on and off policy and is the recommended default penalty.
pub fn kl_estimate(logp: Tensor, ref_logp: &Tensor, estimator: KlEstimator, delta_clamp: Option<f64>, mask: Option<&Tensor>, reduction: TokenReduction) -> Tensor {
    assert_eq!(logp.rank(), 3, "kl_estimate: logp [S, B, T]");
    assert_eq!(logp.shape(), ref_logp.shape(), "kl_estimate: reference log-probabilities shape");
    if let Some(b) = delta_clamp {
        assert!(b > 0.0, "kl_estimate: δ clamp {b} (need b > 0)");
    }
    debug_check("kl_estimate", &logp, mask);
    debug_check("kl_estimate (reference log-probabilities)", ref_logp, mask);
    let m = mask_or_ones(mask, &logp);
    let delta = zero_masked(ref_logp.clone().detach() - logp, &m);
    let delta = match delta_clamp {
        Some(b) => delta.clamp(-b, b),
        None => delta,
    };
    let per = match estimator {
        KlEstimator::K1 => -delta,
        KlEstimator::K2 => (delta.clone() * delta).div_scalar(2.0),
        KlEstimator::K3 => delta.clone().exp() - delta.add_scalar(1.0),
    };
    reduce_tokens(zero_masked(per, &m), &m, reduction)
}

/// The entropy of the categorical softmax(logits) per position, `H = −Σ_v π_v log π_v`, logits
/// `[S, B, T, V]`; a token of probability 0 (logit −∞, or an additive −1e9 that underflows)
/// contributes 0.
pub fn categorical_entropy(logits: Tensor, mask: Option<&Tensor>, reduction: TokenReduction) -> Tensor {
    assert_eq!(logits.rank(), 4, "categorical_entropy: logits [S, B, T, V]");
    let [s, b, t, _] = logits.dims();
    let logp = log_softmax(logits, 3);
    let zero_p = logp.clone().untracked().equal_elem(f64::NEG_INFINITY);
    let safe = logp.clone().mask_fill(zero_p, 0.0);
    let per = -(logp.exp() * safe).sum_dim(3).reshape([s, b, t]);
    let m = mask_or_ones(mask, &per);
    reduce_tokens(zero_masked(per, &m), &m, reduction)
}

/// How `group_advantages` normalises a group's centred rewards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GroupNorm {
    /// `r − mean_group`.
    MeanOnly,
    /// `(r − mean_group) / (std_group + eps)`, with the sample (N − 1, needs N ≥ 2) or the
    /// population (N) standard deviation.
    Std { eps: f64, unbiased: bool },
}

/// Group-relative advantages of rewards `[G, N]` (row-major; G groups of N samples of the same
/// prompt), on the host in f64: each reward minus its group's mean, divided by the group's
/// standard deviation plus ε with `GroupNorm::Std`. A group whose rewards are all equal (max ==
/// min) gives exact zeros, whatever ε (the mean's rounding would otherwise leave ±1-size noise
/// at ε = 0).
///
/// Dividing by the group's standard deviation weights groups unevenly: a group with one success
/// in N gets larger advantages than a balanced one, so nearly solved and nearly failed prompts
/// count more. For binary verifier rewards, `MeanOnly` with `TokenReduction::TokenMean` is the
/// safer default.
pub fn group_advantages(rewards: &[f64], n: usize, norm: GroupNorm) -> Vec<f64> {
    assert!(n >= 1 && rewards.len().is_multiple_of(n), "group_advantages: {} rewards in groups of {n}", rewards.len());
    if let GroupNorm::Std { eps, unbiased } = norm {
        assert!(eps >= 0.0, "group_advantages: ε {eps} < 0");
        assert!(!unbiased || n >= 2, "group_advantages: the sample standard deviation needs groups of at least 2");
    }
    let mut out = Vec::with_capacity(rewards.len());
    for g in rewards.chunks(n) {
        let (lo, hi) = g.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &r| (lo.min(r), hi.max(r)));
        if lo == hi {
            out.extend(std::iter::repeat_n(0.0, n));
            continue;
        }
        let mean = g.iter().sum::<f64>() / n as f64;
        let scale = match norm {
            GroupNorm::MeanOnly => 1.0,
            GroupNorm::Std { eps, unbiased } => {
                let ss: f64 = g.iter().map(|r| (r - mean) * (r - mean)).sum();
                let var = ss / if unbiased { (n - 1) as f64 } else { n as f64 };
                var.sqrt() + eps
            }
        };
        out.extend(g.iter().map(|r| (r - mean) / scale));
    }
    out
}

/// `group_advantages` over the last axis of a rewards tensor `[…, N]`, read to the host; the
/// result has the rewards' shape, dtype and device (untracked).
pub fn group_advantages_tensor(rewards: &Tensor, norm: GroupNorm) -> Tensor {
    let n = *rewards.shape().last().expect("group_advantages_tensor: rewards of rank 0");
    let adv = group_advantages(&rewards.to_vec_f64(), n, norm);
    Tensor::from_f64s(adv, rewards.shape().to_vec(), rewards.dtype()).to(rewards.device())
}

/// The logits of the sampling distribution: `logits / temperature + mask`, the mask additive
/// (−∞ excludes a token; it broadcasts at the logits' rank). The log-probabilities of sampled
/// tokens for the ratio or the KL must be taken under this same distribution
/// (`token_log_probs(tempered_logits(..), tokens)`). Temperature must be positive.
pub fn tempered_logits(logits: Tensor, temperature: f64, mask: Option<&Tensor>) -> Tensor {
    assert!(temperature > 0.0 && temperature.is_finite(), "tempered_logits: temperature {temperature} (must be positive)");
    let z = if temperature == 1.0 { logits } else { logits.div_scalar(temperature) };
    match mask {
        Some(m) => {
            assert_eq!(m.rank(), z.rank(), "tempered_logits: mask rank");
            assert!(m.shape().iter().zip(z.shape()).all(|(a, b)| a == b || *a == 1), "tempered_logits: mask {:?} does not broadcast to the logits {:?}", m.shape(), z.shape());
            z + m.clone().detach()
        }
        None => z,
    }
}

/// One Gumbel(0, 1) draw per element, `−ln(−ln u)` with u uniform on the open interval (0, 1)
/// from 53 random bits: `u = (k + 1/2) / 2^53`. One `u64` of the stream per element, row-major.
fn gumbel(n: usize, rng: &mut ChaCha20Rng) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let k = rng.next_u64() >> 11;
            let u = (k as f64 + 0.5) / (1u64 << 53) as f64;
            -(-u.ln()).ln()
        })
        .collect()
}

/// One token per position from softmax(`logits / temperature + mask`) along the last axis,
/// logits `[…, V]` → tokens `[…]`, by the Gumbel-max trick: `argmax(z + g)` with one Gumbel draw
/// per logit from `rng` (an engine stream, `crate::rng::derive`), drawn on the host in row-major
/// order and added on the logits' device, where the argmax runs. The same stream gives the same
/// tokens on every run of a backend; CPU and GPU backends draw the same noise, so they differ
/// only where two perturbed logits round differently. Temperature 0 is greedy (`argmax` of the
/// masked logits) and draws nothing from the stream.
///
/// The sampler itself is untracked. A position at which every token is excluded (or a logit is
/// NaN) is refused; this reads the per-position maxima to the host.
pub fn sample_categorical(logits: &Tensor, temperature: f64, mask: Option<&Tensor>, rng: &mut ChaCha20Rng) -> IntTensor {
    assert!(temperature >= 0.0 && temperature.is_finite(), "sample_categorical: temperature {temperature} (must be ≥ 0)");
    let r = logits.rank();
    assert!(r >= 1, "sample_categorical: logits of rank 0");
    let out = logits.shape()[..r - 1].to_vec();
    let z = tempered_logits(logits.clone().detach(), if temperature == 0.0 { 1.0 } else { temperature }, mask);
    let top = z.clone().max_dim(r - 1).to_vec_f64();
    if let Some(i) = top.iter().position(|m| m.is_nan() || *m == f64::NEG_INFINITY) {
        panic!("sample_categorical: position {i} has no allowed token (every logit −∞ or a NaN)");
    }
    let z = if temperature == 0.0 {
        z
    } else {
        let g = Tensor::from_f64s(gumbel(z.numel(), rng), z.shape().to_vec(), z.dtype()).to(z.device());
        z + g
    };
    let idx = z.argmax(r - 1);
    assert_eq!(idx.shape().iter().product::<usize>(), out.iter().product::<usize>(), "sample_categorical: argmax shape");
    idx.reshaped(out)
}
