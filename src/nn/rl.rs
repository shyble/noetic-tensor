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
//! and a zero gradient instead of `0 · (−∞)`.

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
    /// `[S]` (each sequence counts once, whatever its length; 0 if there is none).
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

/// The policy-gradient (REINFORCE) loss with signed weights, `l = −w · log π(token)` per scored
/// token: `logp` `[S, B, T]` (from `token_log_probs`), `weights` the advantages or rewards per
/// token `[S, B, T]` or per sequence `[S, B]` (detached; any sign). Minimising it raises the
/// log-probability of positively weighted tokens and lowers that of negatively weighted ones.
pub fn policy_gradient_loss(logp: Tensor, weights: &Tensor, mask: Option<&Tensor>, reduction: TokenReduction) -> Tensor {
    assert_eq!(logp.rank(), 3, "policy_gradient_loss: logp [S, B, T]");
    let m = mask_or_ones(mask, &logp);
    let w = token_weights(weights, &logp, "policy_gradient_loss");
    let per = zero_masked(-(logp * w), &m);
    reduce_tokens(per, &m, reduction)
}

/// The clip range of `clipped_policy_loss`: the ratio is clipped to `[1 − low, 1 + high]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clip {
    pub low: f64,
    pub high: f64,
}

impl Clip {
    /// `[1 − ε, 1 + ε]`, PPO's.
    pub fn symmetric(eps: f64) -> Clip {
        Clip { low: eps, high: eps }
    }

    fn check(self) {
        assert!(self.low >= 0.0 && self.low < 1.0 && self.high >= 0.0 && self.low.is_finite() && self.high.is_finite(), "clip range [1 − {}, 1 + {}] (need 0 ≤ low < 1, 0 ≤ high)", self.low, self.high);
    }
}

/// The ratio `exp(logp − old)` with the difference zeroed at masked positions before the
/// exponential (so arbitrary old values at padding cannot overflow).
fn ratio(logp: Tensor, old_logp: &Tensor, m: &Tensor) -> Tensor {
    assert_eq!(old_logp.shape(), logp.shape(), "rl: old log-probabilities shape");
    zero_masked(logp - old_logp.clone().detach(), m).exp()
}

/// PPO's clipped surrogate as a loss, per scored token: with `r = exp(logp − old_logp)`,
/// `l = −min(r·A, clip(r, 1 − low, 1 + high)·A)`, computed as `max(−r·A, −clip(r)·A)` by a
/// select (the gradient goes to the active branch; the clipped branch's is zero outside the
/// range). `old_logp` `[S, B, T]` are the sampling policy's log-probabilities of the same tokens
/// and `advantages` `[S, B, T]` or `[S, B]`; both are detached. At `logp == old_logp` the
/// gradient equals `policy_gradient_loss`'s with the advantages as weights.
pub fn clipped_policy_loss(logp: Tensor, old_logp: &Tensor, advantages: &Tensor, mask: Option<&Tensor>, clip: Clip, reduction: TokenReduction) -> Tensor {
    clip.check();
    assert_eq!(logp.rank(), 3, "clipped_policy_loss: logp [S, B, T]");
    let m = mask_or_ones(mask, &logp);
    let a = token_weights(advantages, &logp, "clipped_policy_loss");
    let r = ratio(logp, old_logp, &m);
    let unclipped = -(r.clone() * a.clone());
    let clipped = -(r.clamp(1.0 - clip.low, 1.0 + clip.high) * a);
    let per = zero_masked(select_max(unclipped, clipped), &m);
    reduce_tokens(per, &m, reduction)
}

/// The fraction of scored tokens at which `clipped_policy_loss` takes its clipped branch with
/// the ratio outside the range (the tokens that get no gradient), per seed `[S]`; untracked, for
/// logging. A seed with no scored token gives 0.
pub fn clip_fraction(logp: &Tensor, old_logp: &Tensor, advantages: &Tensor, mask: Option<&Tensor>, clip: Clip) -> Tensor {
    clip.check();
    let logp = logp.clone().detach();
    let m = mask_or_ones(mask, &logp);
    let a = token_weights(advantages, &logp, "clip_fraction");
    let r = ratio(logp, old_logp, &m);
    let [s, b, t] = r.dims();
    // Clipped and active: A > 0 with r > 1 + high, or A < 0 with r < 1 − low.
    let up = r.clone().greater_elem(1.0 + clip.high).float_dtype(r.dtype()).to(r.device());
    let down = r.lower_elem(1.0 - clip.low).float_dtype(m.dtype()).to(m.device());
    let pos = a.clone().greater_elem(0.0).float_dtype(m.dtype()).to(m.device());
    let neg = a.lower_elem(0.0).float_dtype(m.dtype()).to(m.device());
    let hit = (up * pos + down * neg) * m.clone();
    let count = m.reshape([s, b * t]).sum_dim(1).clamp_min(1.0).reshape([s]);
    hit.reshape([s, b * t]).sum_dim(1).reshape([s]) / count
}

/// The exact KL divergence `KL(π ‖ π_ref) = Σ_v π_v (log π_v − log π_ref,v)` per position, with
/// π = softmax(logits) and π_ref = softmax(ref_logits) along the last axis (`[S, B, T, V]`; the
/// reference detached): the reverse KL used as a penalty toward a frozen reference. A token with
/// π_v = 0 (logit −∞) contributes 0; one with π_v > 0 and π_ref,v = 0 makes the KL +∞. The
/// gradient is that of the exact KL (no sampling). Compare `nn::kl_div`, the forward KL
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
/// (detached), on tokens sampled from π. δ is zeroed at masked positions before the exponential.
///
/// The value's expectation under samples from π is the KL for k1 and k3 (k2 is biased). As a
/// loss term its per-sample gradient, on policy, is in expectation: zero for k1; the gradient of
/// `KL(π ‖ π_ref)` for k2; and the gradient of the forward `KL(π_ref ‖ π)` for k3 (the form used
/// in the loss by GRPO-style trainers), not of the reverse KL. For the exact reverse-KL gradient
/// use `categorical_kl` or k2.
///
/// No clamp is applied: k3's `exp(δ)` overflows to +∞ where the reference gives a sampled token
/// far more probability than the policy (δ above about 88 in f32); clamp δ upstream, or use k2,
/// if that can happen.
pub fn kl_estimate(logp: Tensor, ref_logp: &Tensor, estimator: KlEstimator, mask: Option<&Tensor>, reduction: TokenReduction) -> Tensor {
    assert_eq!(logp.rank(), 3, "kl_estimate: logp [S, B, T]");
    assert_eq!(logp.shape(), ref_logp.shape(), "kl_estimate: reference log-probabilities shape");
    let m = mask_or_ones(mask, &logp);
    let delta = zero_masked(ref_logp.clone().detach() - logp, &m);
    let per = match estimator {
        KlEstimator::K1 => -delta,
        KlEstimator::K2 => (delta.clone() * delta).div_scalar(2.0),
        KlEstimator::K3 => delta.clone().exp() - delta.add_scalar(1.0),
    };
    reduce_tokens(zero_masked(per, &m), &m, reduction)
}

/// The entropy of the categorical softmax(logits) per position, `H = −Σ_v π_v log π_v`, logits
/// `[S, B, T, V]`; a token of probability 0 (logit −∞) contributes 0.
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
/// standard deviation plus ε with `GroupNorm::Std`. A group of equal rewards gives zeros.
pub fn group_advantages(rewards: &[f64], n: usize, norm: GroupNorm) -> Vec<f64> {
    assert!(n >= 1 && rewards.len().is_multiple_of(n), "group_advantages: {} rewards in groups of {n}", rewards.len());
    if let GroupNorm::Std { eps, unbiased } = norm {
        assert!(eps >= 0.0, "group_advantages: ε {eps} < 0");
        assert!(!unbiased || n >= 2, "group_advantages: the sample standard deviation needs groups of at least 2");
    }
    let mut out = Vec::with_capacity(rewards.len());
    for g in rewards.chunks(n) {
        let mean = g.iter().sum::<f64>() / n as f64;
        let scale = match norm {
            GroupNorm::MeanOnly => 1.0,
            GroupNorm::Std { eps, unbiased } => {
                let ss: f64 = g.iter().map(|r| (r - mean) * (r - mean)).sum();
                let var = ss / if unbiased { (n - 1) as f64 } else { n as f64 };
                var.sqrt() + eps
            }
        };
        out.extend(g.iter().map(|r| {
            let c = r - mean;
            if c == 0.0 { 0.0 } else { c / scale }
        }));
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
