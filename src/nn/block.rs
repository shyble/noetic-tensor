//! The pre-norm block: `x + attn(norm_att(x))`, then `x + ff(norm_mlp(x))`. By
//! default: RMS norms, causal multi-head attention and the gated MLP.
//! Vars, in order: "norm_att" (and "norm_att_bias" for LayerNorm), the attention's "wq", "wk",
//! "wv", "wo", "norm_mlp" (and its bias), then the feed-forward's vars.
//!
//! Options:
//! - LayerNorm in place of RMSNorm;
//! - GQA and RoPE in the attention;
//! - a plain MLP or an MoE as the feed-forward;
//! - residual dropout on the attention and feed-forward outputs, in training only, labelled
//!   "{prefix}attn" and "{prefix}ff";
//! - a key padding mask.
//!
//! Every option adds operations only when used.

use super::attention::{attention, AttentionConfig, MultiHeadAttention};
use super::dropout::Dropout;
use super::kv_cache::KvCache;
use super::mlp::{mlp, swiglu, GatedMlp, GatedMlpConfig, Mlp};
use super::moe::{moe, Moe, MoeConfig, MoeOutput};
use super::module::Module;
use super::norm::{layer_norm, rms_norm, LayerNorm, RmsNorm};
use super::activation::Activation;
use super::var_builder::VarBuilder;
use crate::error::Result;
use crate::tensor::{BoolTensor, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NormKind {
    #[default]
    Rms,
    Layer,
}

#[derive(Clone, Debug)]
pub enum Norm {
    Rms(RmsNorm),
    Layer(LayerNorm),
}

impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            Norm::Rms(n) => n.forward(xs),
            Norm::Layer(n) => n.forward(xs),
        }
    }
}

/// A norm of `kind` over width `d` named `name`.
pub fn norm(kind: NormKind, d: usize, name: &str, vb: &VarBuilder) -> Result<Norm> {
    Ok(match kind {
        NormKind::Rms => Norm::Rms(rms_norm(d, name, vb)?),
        NormKind::Layer => Norm::Layer(layer_norm(d, name, vb)?),
    })
}

/// The feed-forward's form.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum MlpKind {
    /// The SiLU-gated MLP (SwiGLU).
    #[default]
    Gated,
    /// `out(act(in(x)))`.
    Plain(Activation),
    /// A mixture of experts; `d` and `hidden` are taken from the block.
    Moe(MoeSpec),
}

/// An MoE feed-forward's settings (width and expert width come from the block).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeSpec {
    pub experts: usize,
    pub top_k: usize,
    pub normalize_topk: bool,
    pub capacity_factor: Option<f64>,
    pub expert: super::moe::ExpertKind,
}

#[derive(Clone, Debug)]
pub enum FeedForward {
    Gated(Box<GatedMlp>),
    Plain(Box<Mlp>),
    Moe(Box<Moe>),
}

/// A block's settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockConfig {
    pub attention: AttentionConfig,
    pub norm: NormKind,
    pub mlp: MlpKind,
    pub hidden: usize,
    /// Residual dropout probability and the root of its streams.
    pub dropout: f64,
    pub dropout_root: u64,
}

/// What one pass of a block reports besides its output.
#[derive(Clone, Debug)]
pub struct BlockAux {
    pub moe: Option<MoeOutput>,
}

#[derive(Clone, Debug)]
pub struct Block {
    norm_att: Norm,
    attn: MultiHeadAttention,
    norm_mlp: Norm,
    ff: FeedForward,
    drop: Option<(Dropout, Dropout)>,
}

impl Block {
    pub fn new(norm_att: Norm, attn: MultiHeadAttention, norm_mlp: Norm, ff: FeedForward, drop: Option<(Dropout, Dropout)>) -> Self {
        Block { norm_att, attn, norm_mlp, ff, drop }
    }

    pub fn attention(&self) -> &MultiHeadAttention {
        &self.attn
    }

    pub fn feed_forward(&self) -> &FeedForward {
        &self.ff
    }

    /// The gated MLP, if the feed-forward is one.
    pub fn mlp(&self) -> Option<&GatedMlp> {
        match &self.ff {
            FeedForward::Gated(m) => Some(m),
            _ => None,
        }
    }

    /// `[S, B·T, d]` → `[S, B·T, d]` over sequences of length `t` (inference).
    pub fn forward(&self, x: &Tensor, t: usize) -> Tensor {
        self.forward_observed(x, t, &mut |_, _| {})
    }

    /// As `forward`, calling `observe(xn, hidden)` with the MLP's normalised input `[S, N, d]` and
    /// hidden activations `[S, N, m]` (read-only; not called for an MoE).
    pub fn forward_observed(&self, x: &Tensor, t: usize, observe: &mut dyn FnMut(&Tensor, &Tensor)) -> Tensor {
        self.forward_full(x, t, None, None, observe).0
    }

    /// The full pass: an optional key padding mask `[S, B, t]`, training at `train_step` (dropout
    /// masks of that step, MoE capacity) or inference (None).
    pub fn forward_full(&self, x: &Tensor, t: usize, pad: Option<&BoolTensor>, train_step: Option<u64>, observe: &mut dyn FnMut(&Tensor, &Tensor)) -> (Tensor, BlockAux) {
        let xn = self.norm_att.forward(x);
        let a = match pad {
            Some(p) => self.attn.forward_with(&xn, t, Some(p)),
            None => self.attn.forward(&xn, t),
        };
        let a = self.dropped(a, train_step, 0);
        let x = x.clone() + a;
        let xn = self.norm_mlp.forward(&x);
        let (f, aux) = match &self.ff {
            FeedForward::Gated(m) => {
                let h = m.hidden(&xn);
                observe(&xn, &h);
                (m.project(&h), None)
            }
            FeedForward::Plain(m) => {
                let h = m.hidden(&xn);
                observe(&xn, &h);
                (m.project(&h), None)
            }
            FeedForward::Moe(m) => {
                let o = m.forward_moe(&xn, train_step.is_some());
                (o.y.clone(), Some(o))
            }
        };
        let f = self.dropped(f, train_step, 1);
        (x + f, BlockAux { moe: aux })
    }

    fn dropped(&self, y: Tensor, train_step: Option<u64>, which: usize) -> Tensor {
        match (&self.drop, train_step) {
            (Some(d), Some(step)) => (if which == 0 { &d.0 } else { &d.1 }).forward_at(&y, true, step),
            _ => y,
        }
    }

    /// As `forward` for the next `t` positions, attention reading and extending `cache`.
    pub fn forward_cached(&self, x: &Tensor, t: usize, cache: &mut KvCache) -> Tensor {
        let xn = self.norm_att.forward(x);
        let x = x.clone() + self.attn.forward_cached(&xn, t, cache);
        let xn = self.norm_mlp.forward(&x);
        x + match &self.ff {
            FeedForward::Gated(m) => m.forward(&xn),
            FeedForward::Plain(m) => m.forward(&xn),
            FeedForward::Moe(m) => m.forward(&xn),
        }
    }
}

/// The default block: width `d`, `heads` heads and the given gated MLP.
pub fn block(d: usize, heads: usize, mlp: GatedMlpConfig, vb: &VarBuilder) -> Result<Block> {
    block_with(BlockConfig { attention: AttentionConfig::causal(d, heads), norm: NormKind::Rms, mlp: MlpKind::Gated, hidden: mlp.hidden, dropout: 0.0, dropout_root: 0 }, vb)
}

/// A block by `cfg`; `vb.path("")` labels its dropout streams.
pub fn block_with(cfg: BlockConfig, vb: &VarBuilder) -> Result<Block> {
    let d = cfg.attention.d;
    let norm_att = norm(cfg.norm, d, "norm_att", vb)?;
    let attn = attention(cfg.attention, vb)?;
    let norm_mlp = norm(cfg.norm, d, "norm_mlp", vb)?;
    let ff = match cfg.mlp {
        MlpKind::Gated => FeedForward::Gated(Box::new(swiglu(d, cfg.hidden, vb)?)),
        MlpKind::Plain(a) => FeedForward::Plain(Box::new(mlp(d, cfg.hidden, a, vb)?)),
        MlpKind::Moe(m) => FeedForward::Moe(Box::new(moe(MoeConfig { d, hidden: cfg.hidden, experts: m.experts, top_k: m.top_k, normalize_topk: m.normalize_topk, capacity_factor: m.capacity_factor, expert: m.expert }, &vb.pp("moe"))?)),
    };
    let drop = (cfg.dropout > 0.0).then(|| {
        let p = vb.path("");
        let label = |w: &str| if p.is_empty() { w.to_string() } else { format!("{p}.{w}") };
        (Dropout::new(cfg.dropout, cfg.dropout_root, label("attn")), Dropout::new(cfg.dropout, cfg.dropout_root, label("ff")))
    });
    Ok(Block::new(norm_att, attn, norm_mlp, ff, drop))
}
