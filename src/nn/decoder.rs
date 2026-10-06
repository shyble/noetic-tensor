//! The decoder: token embedding plus positions, `blocks` pre-norm blocks, a final norm and a
//! linear readout, every position read out. The defaults (`DecoderConfig::new`): RMS norms, the
//! gated MLP, learned positions, kv_heads = heads, no dropout, `BlockNaming::Compact` and the
//! one-hot embedding.
//!
//! Vars, in order: "embed" `[V, d]`, "pos" `[context, d]` (learned positions only; both N(0, 1)),
//! each block's (under its prefix), "norm_out" (and "norm_out_bias" for LayerNorm), "readout"
//! `[d, V]`.
//!
//! Options:
//! - `norm`: RMSNorm or LayerNorm;
//! - `mlp`: the gated MLP, SwiGLU, a plain MLP, or an MoE (`MlpKind::Moe`);
//! - `positions`: learned or RoPE;
//! - `kv_heads`: grouped-query attention;
//! - `dropout`: residual dropout in training, labelled per block, plus "embed" after the
//!   position sum;
//! - `forward_with`: a key padding mask, a training step, and the MoE's auxiliary losses and
//!   loads.
//!
//! `moe_small` is a small MoE decoder configuration.

use super::attention::AttentionConfig;
use super::block::{block_with, norm, Block, BlockConfig, MlpKind, MoeSpec, Norm, NormKind};
use super::dropout::Dropout;
use super::embedding::{embedding, Embedding, EmbeddingMode};
use super::init::Init;
use super::kv_cache::DecoderCache;
use super::linear::{linear, Linear};
use super::moe::ExpertKind;
use super::module::Module;
use super::rotary::RopeConfig;
use super::var::VarMap;
use super::var_builder::VarBuilder;
use crate::error::{NnError, Result};
use crate::tensor::{BoolTensor, IntTensor, Tensor};
use std::cell::RefCell;

/// How block j's vars are prefixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BlockNaming {
    /// Block 0 unprefixed ("wq"), block j ≥ 1 "b{j}." ("b1.wq").
    #[default]
    Compact,
    /// Every block "b{j}." ("b0.wq", "b1.wq", …).
    Indexed,
}

impl BlockNaming {
    pub fn prefix(self, j: usize) -> String {
        match (self, j) {
            (BlockNaming::Compact, 0) => String::new(),
            _ => format!("b{j}"),
        }
    }
}

/// How positions enter.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum Positions {
    /// A learned table "pos" `[context, d]` added to the embeddings.
    #[default]
    Learned,
    /// Rotary embedding of queries and keys; no position var.
    Rope(RopeConfig),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecoderConfig {
    pub vocab: usize,
    pub d: usize,
    pub heads: usize,
    /// Positions (the longest sequence).
    pub context: usize,
    pub blocks: usize,
    /// The MLP's hidden width (each expert's, for an MoE).
    pub mlp_hidden: usize,
    pub embedding: EmbeddingMode,
    pub naming: BlockNaming,
    pub norm: NormKind,
    pub mlp: MlpKind,
    pub positions: Positions,
    /// Key/value heads (= heads: multi-head attention).
    pub kv_heads: usize,
    /// Residual and embedding dropout in training (0: none), and the root of its streams.
    pub dropout: f64,
    pub dropout_root: u64,
    /// An extension of every block's gated MLP (`MlpExtension`; gated MLP only). Not serialised: a
    /// caller that uses one records it in its own terms.
    #[serde(skip)]
    pub mlp_ext: Option<super::mlp::MlpExt>,
}

impl DecoderConfig {
    /// The default decoder at these sizes: one-hot embedding, unprefixed naming, RMS norms, the
    /// gated MLP, learned positions, multi-head attention, no dropout, no MLP extension.
    pub fn new(vocab: usize, d: usize, heads: usize, context: usize, blocks: usize, mlp_hidden: usize) -> Self {
        DecoderConfig {
            vocab,
            d,
            heads,
            context,
            blocks,
            mlp_hidden,
            embedding: EmbeddingMode::OneHot,
            naming: BlockNaming::Compact,
            norm: NormKind::Rms,
            mlp: MlpKind::Gated,
            positions: Positions::Learned,
            kv_heads: heads,
            dropout: 0.0,
            dropout_root: 0,
            mlp_ext: None,
        }
    }

    /// A small MoE decoder: d 32, 4 heads over 2 key/value heads, RoPE, 2 blocks, 4 SwiGLU experts
    /// of width 64 with top-2 routing (renormalised) and a training capacity factor of 1.25,
    /// gather embedding.
    pub fn moe_small(vocab: usize, context: usize) -> Self {
        DecoderConfig {
            embedding: EmbeddingMode::Gather,
            naming: BlockNaming::Indexed,
            positions: Positions::Rope(RopeConfig::default()),
            kv_heads: 2,
            mlp: MlpKind::Moe(MoeSpec { experts: 4, top_k: 2, normalize_topk: true, capacity_factor: Some(1.25), expert: ExpertKind::SwiGlu }),
            ..DecoderConfig::new(vocab, 32, 4, context, 2, 64)
        }
    }

    fn block_config(&self) -> BlockConfig {
        let rope = match self.positions {
            Positions::Rope(r) => Some(r),
            Positions::Learned => None,
        };
        BlockConfig {
            attention: AttentionConfig { d: self.d, heads: self.heads, kv_heads: self.kv_heads, causal: true, rope },
            norm: self.norm,
            mlp: self.mlp,
            hidden: self.mlp_hidden,
            mlp_ext: self.mlp_ext.clone(),
            dropout: self.dropout,
            dropout_root: self.dropout_root,
        }
    }
}

/// Options of one decoder pass.
#[derive(Clone, Debug, Default)]
pub struct ForwardOptions {
    /// Key padding mask `[S, B, T]` (true = pad).
    pub pad: Option<BoolTensor>,
    /// Training at this step (dropout masks of the step, MoE capacity); None: inference.
    pub train_step: Option<u64>,
}

/// One decoder pass: logits and, with MoE blocks, the auxiliary losses and loads.
#[derive(Clone, Debug)]
pub struct DecoderOutput {
    /// `[S, B, T, V]`.
    pub logits: Tensor,
    /// Sum over MoE blocks of the Switch balance loss, `[S]` (None without MoE blocks).
    pub balance: Option<Tensor>,
    /// Sum over MoE blocks of the router z-loss, `[S]`.
    pub z_loss: Option<Tensor>,
    /// Per MoE block, per seed, kept assignments per expert.
    pub load: Vec<Vec<Vec<usize>>>,
    /// Per MoE block, per seed, assignments dropped by capacity.
    pub dropped: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub struct Decoder {
    cfg: DecoderConfig,
    embed: Embedding,
    pos: Option<Tensor>,
    blocks: Vec<Block>,
    norm_out: Norm,
    readout: Linear,
    drop: Option<Dropout>,
}

impl Decoder {
    /// Build from `vb`: in init mode this creates the vars in the order listed above.
    pub fn new(cfg: DecoderConfig, vb: &VarBuilder) -> Result<Self> {
        if cfg.blocks == 0 || cfg.vocab == 0 || cfg.context == 0 || cfg.mlp_hidden == 0 {
            return Err(NnError::Config(format!("decoder: vocab, context, blocks and MLP width must be positive ({cfg:?})")));
        }
        if !(0.0..1.0).contains(&cfg.dropout) {
            return Err(NnError::Config(format!("decoder: dropout {} outside [0, 1)", cfg.dropout)));
        }
        let embed = embedding(cfg.vocab, cfg.d, "embed", cfg.embedding, vb)?;
        let pos = match cfg.positions {
            Positions::Learned => Some(vb.get(&[cfg.context, cfg.d], "pos", Init::Normal { std: 1.0 })?),
            Positions::Rope(_) => None,
        };
        let bc = cfg.block_config();
        let blocks = (0..cfg.blocks).map(|j| block_with(bc.clone(), &vb.pp(&cfg.naming.prefix(j)))).collect::<Result<Vec<_>>>()?;
        let norm_out = norm(cfg.norm, cfg.d, "norm_out", vb)?;
        let readout = linear(cfg.d, cfg.vocab, "readout", vb)?;
        let drop = (cfg.dropout > 0.0).then(|| Dropout::new(cfg.dropout, cfg.dropout_root, vb.path("embed")));
        Ok(Decoder { cfg, embed, pos, blocks, norm_out, readout, drop })
    }
    /// A fresh decoder for seeds 0..`seeds` from `root`, with its vars.
    pub fn init(cfg: DecoderConfig, seeds: usize, root: u64) -> Result<(Decoder, VarMap)> {
        Self::init_indexed(cfg, &(0..seeds).collect::<Vec<_>>(), root)
    }

    /// As `init`, slot j initialised as seed `indices[j]`.
    pub fn init_indexed(cfg: DecoderConfig, indices: &[usize], root: u64) -> Result<(Decoder, VarMap)> {
        let map = RefCell::new(VarMap::new());
        let dec = Decoder::new(cfg, &VarBuilder::init_indexed(&map, indices, root))?;
        Ok((dec, map.into_inner()))
    }

    /// The decoder reading `vars` (a loaded or lifted map).
    pub fn load(cfg: DecoderConfig, vars: &VarMap) -> Result<Decoder> {
        Decoder::new(cfg, &VarBuilder::from_varmap(vars))
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Embedding, positions, the blocks and the final norm: tokens `[S, B, T]` → `[S, B·T, d]`.
    pub fn trunk(&self, tokens: &IntTensor) -> Tensor {
        self.trunk_observed(tokens, &mut |_, _, _| {})
    }

    /// As `trunk`, calling `observe(j, xn, hidden)` for block j's MLP input and hidden units.
    pub fn trunk_observed(&self, tokens: &IntTensor, observe: &mut dyn FnMut(usize, &Tensor, &Tensor)) -> Tensor {
        self.trunk_full(tokens, &ForwardOptions::default(), observe).0
    }

    /// Positions `past..past + t` of the learned table, broadcast to `[S, B·t, d]`.
    fn positions(&self, s: usize, b: usize, t: usize, past: usize) -> Option<Tensor> {
        let d = self.cfg.d;
        self.pos.as_ref().map(|p| {
            let p = if past == 0 && t == self.cfg.context { p.clone() } else { p.clone().slice([0..s, past..past + t, 0..d]) };
            p.unsqueeze_dim(1).expand([s, b, t, d]).reshape([s, b * t, d])
        })
    }

    /// The trunk with options; also the blocks' MoE outputs.
    fn trunk_full(&self, tokens: &IntTensor, opts: &ForwardOptions, observe: &mut dyn FnMut(usize, &Tensor, &Tensor)) -> (Tensor, Vec<super::moe::MoeOutput>) {
        let [s, b, t] = tokens.dims();
        let n = b * t;
        assert!(t <= self.cfg.context, "decoder: sequence of {t} beyond the context {}", self.cfg.context);
        let flat = tokens.clone().reshape([s, n]);
        let x = self.embed.forward(&flat);
        let mut x = match self.positions(s, b, t, 0) {
            Some(p) => x + p,
            None => x,
        };
        if let (Some(d), Some(step)) = (&self.drop, opts.train_step) {
            x = d.forward_at(&x, true, step);
        }
        let mut moes = Vec::new();
        for (j, blk) in self.blocks.iter().enumerate() {
            let (y, aux) = blk.forward_full(&x, t, opts.pad.as_ref(), opts.train_step, &mut |xn, h| observe(j, xn, h));
            x = y;
            moes.extend(aux.moe);
        }
        (self.norm_out.forward(&x), moes)
    }

    /// Readout of trunk rows: `[S, M, d]` → `[S, M, V]`.
    pub fn head(&self, x: &Tensor) -> Tensor {
        self.readout.forward(x)
    }

    /// Logits at every position, `[S, B, T, V]`.
    pub fn forward(&self, tokens: &IntTensor) -> Tensor {
        let [s, b, t] = tokens.dims();
        self.head(&self.trunk(tokens)).reshape([s, b, t, self.cfg.vocab])
    }

    /// A pass with options: logits `[S, B, T, V]` and the MoE blocks' auxiliary losses (summed
    /// over blocks in order) and loads.
    pub fn forward_with(&self, tokens: &IntTensor, opts: &ForwardOptions) -> DecoderOutput {
        let [s, b, t] = tokens.dims();
        let (x, moes) = self.trunk_full(tokens, opts, &mut |_, _, _| {});
        let logits = self.head(&x).reshape([s, b, t, self.cfg.vocab]);
        let sum = |f: &dyn Fn(&super::moe::MoeOutput) -> Tensor| moes.iter().map(f).reduce(|a, c| a + c);
        DecoderOutput {
            logits,
            balance: sum(&|m| m.balance.clone()),
            z_loss: sum(&|m| m.z_loss.clone()),
            load: moes.iter().map(|m| m.load.clone()).collect(),
            dropped: moes.iter().map(|m| m.dropped.clone()).collect(),
        }
    }

    /// Logits at the last position, `[S, B, V]` (the readout runs on that position only).
    pub fn forward_last(&self, tokens: &IntTensor) -> Tensor {
        let [s, b, t] = tokens.dims();
        let d = self.cfg.d;
        let x = self.trunk(tokens).reshape([s, b, t, d]).slice([0..s, 0..b, t - 1..t, 0..d]).reshape([s, b, d]);
        self.head(&x)
    }

    /// Empty, extendable caches for this decoder's blocks.
    pub fn cache(&self) -> DecoderCache {
        DecoderCache::new(self.blocks.len())
    }

    /// Caches of the context's capacity: cached logits equal full recompute's to the bit.
    pub fn cache_padded(&self) -> DecoderCache {
        DecoderCache::with_capacity(self.blocks.len(), self.cfg.context)
    }

    /// Logits `[S, B, t, V]` of the next `t` positions of B sequences whose earlier positions
    /// are in `cache` (prefill with an empty cache; then one position at a time). Positions use
    /// their absolute index. The decoder must be untracked.
    pub fn forward_cached(&self, tokens: &IntTensor, cache: &mut DecoderCache) -> Tensor {
        let [s, b, t] = tokens.dims();
        let n = b * t;
        let past = cache.len();
        assert!(past + t <= self.cfg.context, "decoder: {past} cached + {t} new positions beyond the context {}", self.cfg.context);
        let x = self.embed.forward(&tokens.clone().reshape([s, n]));
        let mut x = match self.positions(s, b, t, past) {
            Some(p) => x + p,
            None => x,
        };
        for (j, blk) in self.blocks.iter().enumerate() {
            x = blk.forward_cached(&x, t, cache.block_mut(j));
        }
        let x = self.norm_out.forward(&x);
        self.head(&x).reshape([s, b, t, self.cfg.vocab])
    }
}
