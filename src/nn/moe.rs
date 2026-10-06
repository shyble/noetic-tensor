//! Mixture of experts, dense dispatch: exact and deterministic.
//!
//! - **Router:** a linear layer "router" `[d, E]`, then softmax over the experts.
//! - **Selection:** each token's top k experts, chosen on the host from the probabilities, ties
//!   broken by the lower expert index. The tensor sort is unstable, so it is not used here.
//! - **Weights:** the router probability of each chosen expert. With `normalize_topk` they are
//!   renormalised to sum to 1 over the chosen experts.
//! - **Capacity (training only):** with a capacity factor c, expert e accepts at most
//!   ⌈c·N·k/E⌉ assignments per seed. Assignments are taken in token order, then rank order, and
//!   an overflowing one is dropped (weight 0; its token keeps its other experts). At inference
//!   there is no capacity, so a token's output never depends on the other tokens, which keeps
//!   cached decoding equal to full recompute.
//! - **Dispatch:** every expert runs on every token, and the output is Σₑ wₑ·expertₑ(x),
//!   summed in expert order starting from expert 0's term (no zero accumulator, so E = 1, k = 1
//!   is its expert to the bit).
//! - **Auxiliary losses, per seed:**
//!   - Switch balance `E·Σₑ fₑ·Pₑ`, with fₑ the fraction of top-k assignments to e (before
//!     capacity; a constant) and Pₑ the mean router probability of e. It is exactly 1 whenever
//!     the probabilities are uniform.
//!   - z-loss: the mean over tokens of logsumexp(router logits)².
//! - `MoeOutput::load` is each seed's kept assignments per expert.

use super::activation::Activation;
use super::linear::{linear, Linear};
use super::mlp::{mlp, swiglu, GatedMlp, Mlp};
use super::module::Module;
use super::var_builder::VarBuilder;
use crate::error::{NnError, Result};
use crate::tensor::{softmax, Tensor};

/// The experts' form.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum ExpertKind {
    /// `(in(x) · silu(gate(x))) · out`.
    #[default]
    SwiGlu,
    /// `out(act(in(x)))`.
    Plain(Activation),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeConfig {
    pub d: usize,
    /// Each expert's hidden width.
    pub hidden: usize,
    pub experts: usize,
    pub top_k: usize,
    pub normalize_topk: bool,
    /// Training-time capacity factor (None: no capacity).
    pub capacity_factor: Option<f64>,
    pub expert: ExpertKind,
}

#[derive(Clone, Debug)]
pub enum Expert {
    Gated(Box<GatedMlp>),
    Plain(Box<Mlp>),
}

impl Module for Expert {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            Expert::Gated(m) => m.forward(xs),
            Expert::Plain(m) => m.forward(xs),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Moe {
    cfg: MoeConfig,
    router: Linear,
    experts: Vec<Expert>,
}

/// One MoE pass.
#[derive(Clone, Debug)]
pub struct MoeOutput {
    /// `[S, N, d]`.
    pub y: Tensor,
    /// Switch balance loss per seed, `[S]`.
    pub balance: Tensor,
    /// Router z-loss per seed, `[S]`.
    pub z_loss: Tensor,
    /// Kept assignments per seed and expert.
    pub load: Vec<Vec<usize>>,
    /// Assignments dropped by capacity, per seed.
    pub dropped: Vec<usize>,
    /// The per-expert capacity applied, if any.
    pub capacity: Option<usize>,
}

impl Moe {
    pub fn new(cfg: MoeConfig, router: Linear, experts: Vec<Expert>) -> Self {
        assert_eq!(experts.len(), cfg.experts, "one module per expert");
        assert!(cfg.top_k >= 1 && cfg.top_k <= cfg.experts, "top_k {} of {} experts", cfg.top_k, cfg.experts);
        Moe { cfg, router, experts }
    }

    pub fn config(&self) -> &MoeConfig {
        &self.cfg
    }

    pub fn router(&self) -> &Linear {
        &self.router
    }

    pub fn experts(&self) -> &[Expert] {
        &self.experts
    }

    /// The chosen experts of one token: the k largest probabilities, ties to the lower index.
    pub fn top_k(probs: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
        idx.truncate(k);
        idx
    }

    /// `x` `[S, N, d]`; `train` applies the capacity factor.
    pub fn forward_moe(&self, x: &Tensor, train: bool) -> MoeOutput {
        let [s, n, _] = x.dims();
        let (e, k) = (self.cfg.experts, self.cfg.top_k);
        let dtype = x.dtype();
        let logits = self.router.forward(x); // [S, N, E]
        let probs = softmax(logits.clone(), 2);
        let pv = probs.to_vec_f64();
        let capacity = if train { self.cfg.capacity_factor.map(|c| (c * (n * k) as f64 / e as f64).ceil() as usize) } else { None };
        let mut chosen = vec![0.0f64; s * n * e];
        let mut kept = vec![0.0f64; s * n * e];
        let mut counts = vec![vec![0usize; e]; s];
        let mut load = vec![vec![0usize; e]; s];
        let mut dropped = vec![0usize; s];
        for seed in 0..s {
            for tok in 0..n {
                let row = (seed * n + tok) * e;
                for ex in Self::top_k(&pv[row..row + e], k) {
                    chosen[row + ex] = 1.0;
                    counts[seed][ex] += 1;
                    if capacity.is_none_or(|c| load[seed][ex] < c) {
                        kept[row + ex] = 1.0;
                        load[seed][ex] += 1;
                    } else {
                        dropped[seed] += 1;
                    }
                }
            }
        }
        let mk = |v: Vec<f64>, shape: Vec<usize>| Tensor::from_f64s(v, shape, dtype).to(x.device());
        let w = probs.clone() * mk(chosen, vec![s, n, e]);
        let w = if self.cfg.normalize_topk { w.clone() / w.sum_dim(2) } else { w };
        let w = if capacity.is_some() { w * mk(kept, vec![s, n, e]) } else { w };
        let mut y: Option<Tensor> = None;
        for (i, ex) in self.experts.iter().enumerate() {
            let term = ex.forward(x) * w.clone().slice([0..s, 0..n, i..i + 1]);
            y = Some(match y {
                None => term,
                Some(acc) => acc + term,
            });
        }
        // Switch balance: E · Σ f_e · P_e.
        let f: Vec<f64> = counts.iter().flat_map(|c| c.iter().map(|x| *x as f64 / (n * k) as f64).collect::<Vec<_>>()).collect();
        let p_mean = probs.mean_dim(1); // [S, 1, E]
        let balance = (p_mean * mk(f, vec![s, 1, e])).sum_dim(2).reshape([s]).mul_scalar(e as f64);
        // z-loss: mean over tokens of logsumexp(logits)².
        let m = logits.clone().detach().max_dim(2);
        let lse = (logits - m.clone()).exp().sum_dim(2).log() + m;
        let z_loss = (lse.clone() * lse).reshape([s, n]).mean_dim(1).reshape([s]);
        MoeOutput { y: y.expect("at least one expert"), balance, z_loss, load, dropped, capacity }
    }
}

impl Module for Moe {
    /// The inference pass (no capacity).
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_moe(xs, false).y
    }
}

/// An MoE: "router" `[d, E]` (N(0, 1/d)), then expert e's vars under "e{e}" ("e0.mlp_in", …).
pub fn moe(cfg: MoeConfig, vb: &VarBuilder) -> Result<Moe> {
    if cfg.experts == 0 || cfg.top_k == 0 || cfg.top_k > cfg.experts {
        return Err(NnError::Config(format!("MoE: top_k {} of {} experts", cfg.top_k, cfg.experts)));
    }
    if cfg.capacity_factor.is_some_and(|c| c <= 0.0 || c.is_nan()) {
        return Err(NnError::Config(format!("MoE: capacity factor {:?} must be positive", cfg.capacity_factor)));
    }
    let router = linear(cfg.d, cfg.experts, "router", vb)?;
    let experts = (0..cfg.experts)
        .map(|i| {
            let vb = vb.pp(&format!("e{i}"));
            Ok(match cfg.expert {
                ExpertKind::SwiGlu => Expert::Gated(Box::new(swiglu(cfg.d, cfg.hidden, &vb)?)),
                ExpertKind::Plain(a) => Expert::Plain(Box::new(mlp(cfg.d, cfg.hidden, a, &vb)?)),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Moe::new(cfg, router, experts))
}
