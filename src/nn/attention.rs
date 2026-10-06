//! Multi-head and grouped-query self-attention, in a fixed layout and operation order. The residual stream is the seed-batched token layout `[S, B·T, d]`; heads
//! split as `[S·B, T, H, dh] → swap → [S·B·H, T, dh]`; scores are `q · kᵀ` then scaled by 1/√dh
//! (the scale after the product); the future is filled with −∞ and the
//! softmax is the tensor core's; heads merge back and `wo` projects.
//!
//! Options, each adding operations only when used (so the default sequence is unchanged):
//! - grouped-query attention: `kv_heads` < `heads` key/value heads, each shared by
//!   `heads / kv_heads` query heads (query head h reads key head h / group);
//! - RoPE on queries and keys (at absolute positions, also under the KV-cache);
//! - a key padding mask `[S, B, T]` (true = pad): padded keys get −1e9 (finite, so a query whose
//!   every visible key is a pad reads a finite average instead of NaN), and padded query rows
//!   output exactly 0 (before `wo`, which has no bias).

use super::linear::{linear, Linear};
use super::module::Module;
use super::rotary::{Rope, RopeConfig};
use super::var_builder::VarBuilder;
use crate::error::{NnError, Result};
use super::kv_cache::KvCache;
use crate::tensor::{softmax, BoolTensor, Tensor};

/// The padding fill: finite, far below any score.
pub const PAD_FILL: f64 = -1e9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionConfig {
    pub d: usize,
    pub heads: usize,
    /// Key/value heads (= heads for multi-head attention).
    pub kv_heads: usize,
    pub causal: bool,
    pub rope: Option<RopeConfig>,
}

impl AttentionConfig {
    /// Causal multi-head attention without RoPE (the default).
    pub fn causal(d: usize, heads: usize) -> Self {
        AttentionConfig { d, heads, kv_heads: heads, causal: true, rope: None }
    }
}

#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    heads: usize,
    kv_heads: usize,
    causal: bool,
    rope: Option<Rope>,
}

impl MultiHeadAttention {
    /// Multi-head attention (`kv_heads` = `heads`, no RoPE).
    pub fn new(wq: Linear, wk: Linear, wv: Linear, wo: Linear, heads: usize, causal: bool) -> Self {
        MultiHeadAttention { wq, wk, wv, wo, heads, kv_heads: heads, causal, rope: None }
    }

    /// Grouped-query attention: `wk` and `wv` project to `kv_heads · dh`; optional RoPE.
    /// The projections are `[wq, wk, wv, wo]`.
    pub fn with_options(proj: [Linear; 4], heads: usize, kv_heads: usize, causal: bool, rope: Option<Rope>) -> Self {
        assert!(kv_heads > 0 && heads.is_multiple_of(kv_heads), "{heads} heads do not group into {kv_heads} key/value heads");
        let [wq, wk, wv, wo] = proj;
        MultiHeadAttention { wq, wk, wv, wo, heads, kv_heads, causal, rope }
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// `[S·B, t, KVH·dh]`-shaped projections split to `[S·B·KVH, t, dh]`.
    fn split(z: Tensor, sb: usize, t: usize, heads: usize, dh: usize) -> Tensor {
        z.reshape([sb, t, heads, dh]).swap_dims(1, 2).reshape([sb * heads, t, dh])
    }

    /// Key/value heads repeated for their query heads: `[S·B·KVH, L, dh]` → `[S·B·H, L, dh]`
    /// (no operation for multi-head attention).
    fn repeat_kv(&self, z: Tensor, sb: usize) -> Tensor {
        let (h, kvh) = (self.heads, self.kv_heads);
        if kvh == h {
            return z;
        }
        let [_, l, dh] = z.dims();
        z.reshape([sb, kvh, 1, l, dh]).expand([sb, kvh, h / kvh, l, dh]).reshape([sb * h, l, dh])
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn projections(&self) -> [&Linear; 4] {
        [&self.wq, &self.wk, &self.wv, &self.wo]
    }

    /// `x` is `[S, B·T, d]` holding B sequences of length `t`; returns `[S, B·T, d]`.
    pub fn forward(&self, x: &Tensor, t: usize) -> Tensor {
        self.forward_with(x, t, None)
    }

    /// As `forward`, with a key padding mask `[S, B, t]` (true = pad).
    pub fn forward_with(&self, x: &Tensor, t: usize, pad: Option<&BoolTensor>) -> Tensor {
        let [s, n, d] = x.dims();
        assert!(t > 0 && n % t == 0, "attention: {n} tokens are not whole sequences of {t}");
        let (b, h, kvh) = (n / t, self.heads, self.kv_heads);
        let dh = d / h;
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let vv = self.wv.forward(x);
        let (q, k, vv) = (Self::split(q, s * b, t, h, dh), Self::split(k, s * b, t, kvh, dh), Self::split(vv, s * b, t, kvh, dh));
        let (q, k) = match &self.rope {
            Some(r) => (r.apply(&q, 0), r.apply(&k, 0)),
            None => (q, k),
        };
        let (k, vv) = (self.repeat_kv(k, s * b), self.repeat_kv(vv, s * b));
        let scale = 1.0 / (dh as f64).sqrt();
        let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale); // [S·B·H, T, T]
        let scores = match pad {
            Some(p) => {
                assert_eq!(p.shape(), &[s, b, t], "padding mask shape");
                scores.mask_fill(p.clone().reshape([s * b, 1, 1, t]).expand([s * b, h, t, t]).reshape([s * b * h, t, t]), PAD_FILL)
            }
            None => scores,
        };
        let scores = if self.causal {
            // tril_mask marks the strict upper triangle: the future is masked.
            let mask = BoolTensor::tril_mask([t, t], 0).unsqueeze_dim(0).expand([s * b * h, t, t]);
            scores.mask_fill(mask, f64::NEG_INFINITY)
        } else {
            scores
        };
        let att = softmax(scores, 2).matmul(vv);
        let att = att.reshape([s * b, h, t, dh]).swap_dims(1, 2).reshape([s, n, d]);
        let att = match pad {
            Some(p) => att * p.clone().bool_not().float_dtype(x.dtype()).reshape([s, n, 1]),
            None => att,
        };
        self.wo.forward(&att)
    }

    /// Incremental attention: `x` (`[S, B·t, d]`) holds the next `t` positions of B sequences
    /// whose earlier positions are in `cache`; their keys and values are appended to it. With an
    /// empty cache this is `forward`'s operation sequence; a query at absolute position p sees the
    /// keys at 0..=p. Panics on a tracked input (the cache refuses it).
    pub fn forward_cached(&self, x: &Tensor, t: usize, cache: &mut KvCache) -> Tensor {
        let [s, n, d] = x.dims();
        assert!(t > 0 && n % t == 0, "attention: {n} tokens are not whole sequences of {t}");
        let (b, h, kvh) = (n / t, self.heads, self.kv_heads);
        let dh = d / h;
        let past = cache.len();
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let vv = self.wv.forward(x);
        let (q, k, vv) = (Self::split(q, s * b, t, h, dh), Self::split(k, s * b, t, kvh, dh), Self::split(vv, s * b, t, kvh, dh));
        let (q, k) = match &self.rope {
            Some(r) => (r.apply(&q, past), r.apply(&k, past)),
            None => (q, k),
        };
        // The cache holds the key/value heads (after RoPE), repeated for the query heads on read.
        let (k, vv) = cache.append(k, vv).unwrap_or_else(|e| panic!("{e}"));
        let (k, vv) = (self.repeat_kv(k, s * b), self.repeat_kv(vv, s * b));
        let l = k.shape()[1]; // past + t, or the capacity
        let scale = 1.0 / (dh as f64).sqrt();
        let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale); // [S·B·H, t, L]
        // Query i sits at position past + i: keys after it (future, or empty capacity) are masked.
        let scores = if self.causal && (t > 1 || past + t < l) {
            let mask = BoolTensor::tril_mask([t, l], past as i64).unsqueeze_dim(0).expand([s * b * h, t, l]);
            scores.mask_fill(mask, f64::NEG_INFINITY)
        } else {
            scores
        };
        let att = softmax(scores, 2).matmul(vv);
        let att = att.reshape([s * b, h, t, dh]).swap_dims(1, 2).reshape([s, n, d]);
        self.wo.forward(&att)
    }
}

/// Causal attention of width `d` with `heads` heads; projections "wq", "wk", "wv", "wo" (in that
/// order), each `[S, d, d]` at N(0, 1/d).
pub fn causal_attention(d: usize, heads: usize, vb: &VarBuilder) -> Result<MultiHeadAttention> {
    attention(AttentionConfig::causal(d, heads), vb)
}

/// Attention by `cfg`: "wq" `[d, d]`, "wk" and "wv" `[d, kv_heads · dh]`, "wo" `[d, d]`, each at
/// N(0, 1/d).
pub fn attention(cfg: AttentionConfig, vb: &VarBuilder) -> Result<MultiHeadAttention> {
    let (d, heads, kvh) = (cfg.d, cfg.heads, cfg.kv_heads);
    if heads == 0 || !d.is_multiple_of(heads) {
        return Err(NnError::Config(format!("d = {d} must divide into {heads} heads")));
    }
    if kvh == 0 || !heads.is_multiple_of(kvh) {
        return Err(NnError::Config(format!("{heads} heads do not group into {kvh} key/value heads")));
    }
    let dh = d / heads;
    let rope = cfg.rope.map(|r| Rope::new(dh, r)).transpose()?;
    let wq = linear(d, d, "wq", vb)?;
    let wk = linear(d, kvh * dh, "wk", vb)?;
    let wv = linear(d, kvh * dh, "wv", vb)?;
    let wo = linear(d, d, "wo", vb)?;
    if kvh == heads && rope.is_none() {
        return Ok(MultiHeadAttention { wq, wk, wv, wo, heads, kv_heads: heads, causal: cfg.causal, rope: None });
    }
    Ok(MultiHeadAttention::with_options([wq, wk, wv, wo], heads, kvh, cfg.causal, rope))
}
