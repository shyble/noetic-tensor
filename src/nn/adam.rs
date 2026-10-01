//! Adam, operation for operation: coupled weight decay added to the gradient; moments `β·m + (1−β)·g`
//! and `β·v + (1−β)·g²`; bias correction by `div_scalar(1 − βᵗ)`; ε outside the square root;
//! then lazy-row, per-seed and multiplier factors; the step `p − upd · lr·scale·mult`; and
//! decoupled decay `p · (1 − lr·scale·decay)`.
//!
//! AdamW's decay order is explicit (`DecayOrder`): `AfterStep` shrinks after the step (the
//! default), `Torch` shrinks before it (torch.optim.AdamW:
//! `p ← p·(1 − lr·wd)`, then the Adam step).

use super::optim::{Optimizer, ParamGroups};
use super::var::VarMap;
use crate::tensor::Tensor;

#[derive(Clone, Debug, PartialEq)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Coupled L2 decay: `g + decay · p` before the moments.
    pub weight_decay: f64,
    /// Decoupled decay (AdamW): `p · (1 − lr·scale·decay)`, after or before the step by `decay_order`.
    pub decoupled_decay: f64,
    pub decay_order: DecayOrder,
}

/// Where AdamW's decoupled decay sits relative to the Adam step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DecayOrder {
    /// After the step: `(p − lr·upd)·(1 − lr·wd)`.
    #[default]
    AfterStep,
    /// Before the step: `p·(1 − lr·wd) − lr·upd` (torch.optim.AdamW).
    Torch,
}

impl AdamConfig {
    /// AdamW at `lr` with decoupled decay `wd` in the given order.
    pub fn adamw(lr: f64, wd: f64, order: DecayOrder) -> Self {
        AdamConfig { lr, decoupled_decay: wd, decay_order: order, ..Default::default() }
    }
}

impl Default for AdamConfig {
    fn default() -> Self {
        AdamConfig { lr: 3e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, decoupled_decay: 0.0, decay_order: DecayOrder::AfterStep }
    }
}

pub struct Adam {
    pub cfg: AdamConfig,
    groups: ParamGroups,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    t: u64,
    seed_lr: Option<Tensor>,
}

impl Adam {
    pub fn new(cfg: AdamConfig, groups: ParamGroups, vars: &VarMap) -> Self {
        assert_eq!(groups.names(), vars.names(), "the parameter groups are for another var map");
        let lazy = (0..vars.len()).any(|i| groups.options(i).lazy_rows);
        assert!(!(lazy && cfg.weight_decay > 0.0), "weight decay would give every row a gradient and silently disable lazy rows");
        let m = vars.tensors().iter().map(|p| p.zeros_like()).collect();
        let v = vars.tensors().iter().map(|p| p.zeros_like()).collect();
        let seed_lr = seed_lr_tensor(&groups, vars);
        Adam { cfg, groups, m, v, t: 0, seed_lr }
    }

    pub fn groups(&self) -> &ParamGroups {
        &self.groups
    }

    pub fn groups_mut(&mut self) -> &mut ParamGroups {
        &mut self.groups
    }

    /// Set the learning-rate multiplier tensor of var `name`.
    pub fn set_multiplier(&mut self, name: &str, mult: Tensor) {
        self.groups.set_multiplier(name, mult);
    }

    /// The state (first moments, second moments, step count), for saving.
    pub fn state(&self) -> (&[Tensor], &[Tensor], u64) {
        (&self.m, &self.v, self.t)
    }

    /// Restore a saved state; the moments must match the vars' shapes.
    pub fn set_state(&mut self, m: Vec<Tensor>, v: Vec<Tensor>, t: u64) {
        assert_eq!(m.len(), self.m.len(), "one first moment per var");
        assert_eq!(v.len(), self.v.len(), "one second moment per var");
        for (a, b) in m.iter().zip(&self.m) {
            assert_eq!(a.shape(), b.shape(), "moment shape");
        }
        for (a, b) in v.iter().zip(&self.v) {
            assert_eq!(a.shape(), b.shape(), "moment shape");
        }
        self.m = m;
        self.v = v;
        self.t = t;
    }

    /// Zero the moments of var `name` where `mask` is 1 (broadcastable), for example after
    /// re-initialising part of a var.
    pub fn reset_moments(&mut self, name: &str, mask: Tensor) {
        let i = self.groups.names().iter().position(|n| n == name).unwrap_or_else(|| panic!("no var named {name:?}"));
        let keep = mask.neg() + 1.0;
        self.m[i] = self.m[i].clone() * keep.clone();
        self.v[i] = self.v[i].clone() * keep;
    }
}

impl Optimizer for Adam {
    fn step(&mut self, vars: &mut VarMap, grads: Vec<Option<Tensor>>, lr_scale: f64) {
        assert_eq!(grads.len(), self.m.len(), "one gradient slot per var");
        self.t += 1;
        let c = &self.cfg;
        let bc1 = 1.0 - c.beta1.powi(self.t as i32);
        let bc2 = 1.0 - c.beta2.powi(self.t as i32);
        for (i, g) in grads.into_iter().enumerate() {
            let Some(g) = g else { continue };
            let opts = self.groups.options(i);
            if opts.frozen {
                continue;
            }
            let p = vars.tensors()[i].clone();
            let torch_decay = c.decoupled_decay > 0.0 && c.decay_order == DecayOrder::Torch;
            // On a GPU, the plain update (no lazy rows, per-seed rates or multipliers) runs as
            // one fused kernel with the same f32 operations as the sequence below.
            if !opts.lazy_rows && self.seed_lr.is_none() && self.groups.multiplier(i).is_none() {
                let lr = c.lr * lr_scale * opts.lr_mult;
                let decay = 1.0 - c.lr * lr_scale * c.decoupled_decay;
                let s = crate::tensor::gpu::AdamScalars {
                    wd: c.weight_decay as f32,
                    b1: c.beta1 as f32,
                    omb1: (1.0 - c.beta1) as f32,
                    b2: c.beta2 as f32,
                    omb2: (1.0 - c.beta2) as f32,
                    bc1: bc1 as f32,
                    bc2: bc2 as f32,
                    eps: c.eps as f32,
                    lr: lr as f32,
                    pre: decay as f32,
                    post: decay as f32,
                    flags: (c.weight_decay > 0.0) as u32 | ((torch_decay as u32) << 1) | (((c.decoupled_decay > 0.0 && !torch_decay) as u32) << 2),
                };
                if let Some((p, m, v)) = crate::tensor::gpu::gpu_adam(&p, &g, &self.m[i], &self.v[i], &s) {
                    self.m[i] = m;
                    self.v[i] = v;
                    vars.set_index(i, p).expect("the update keeps the var's shape");
                    continue;
                }
            }
            let g = if c.weight_decay > 0.0 { g + p.clone().mul_scalar(c.weight_decay) } else { g };
            let m_new = self.m[i].clone().mul_scalar(c.beta1) + g.clone().mul_scalar(1.0 - c.beta1);
            let v_new = self.v[i].clone().mul_scalar(c.beta2) + g.clone().powf_scalar(2.0).mul_scalar(1.0 - c.beta2);
            // Lazy rows: a row with no gradient keeps its moments and does not move.
            let hit = if opts.lazy_rows {
                assert_eq!(g.rank(), 3, "lazy rows need a [S, R, C] var");
                Some(g.clone().abs().sum_dim(2).greater_elem(0.0).float_dtype(g.dtype())) // [S, R, 1]
            } else {
                None
            };
            match &hit {
                Some(h) => {
                    self.m[i] = m_new * h.clone() + self.m[i].clone() * (h.clone().neg() + 1.0);
                    self.v[i] = v_new * h.clone() + self.v[i].clone() * (h.clone().neg() + 1.0);
                }
                None => {
                    self.m[i] = m_new;
                    self.v[i] = v_new;
                }
            }
            let mhat = self.m[i].clone().div_scalar(bc1);
            let vhat = self.v[i].clone().div_scalar(bc2);
            let mut upd = mhat / (vhat.sqrt() + c.eps);
            if let Some(h) = hit {
                upd = upd * h;
            }
            if let Some(l) = &self.seed_lr {
                upd = upd * l.clone();
            }
            if let Some(mu) = self.groups.multiplier(i) {
                upd = upd * mu.clone();
            }
            let lr = c.lr * lr_scale * opts.lr_mult;
            let p = if torch_decay { p.mul_scalar(1.0 - c.lr * lr_scale * c.decoupled_decay) } else { p };
            let mut p = p - upd.mul_scalar(lr);
            if c.decoupled_decay > 0.0 && !torch_decay {
                p = p.mul_scalar(1.0 - c.lr * lr_scale * c.decoupled_decay);
            }
            vars.set_index(i, p).expect("the update keeps the var's shape");
        }
    }
}

/// The per-seed rates of `groups` as `[S, 1, 1]` (rounded to f32 first).
pub(crate) fn seed_lr_tensor(groups: &ParamGroups, vars: &VarMap) -> Option<Tensor> {
    groups.seed_lr().map(|l| {
        let s = vars.seeds().expect("a var map with vars");
        assert_eq!(l.len(), s, "one learning-rate multiplier per seed");
        let f: Vec<f64> = l.iter().map(|x| (*x as f32) as f64).collect();
        Tensor::from_f64s(f, [s, 1, 1], vars.tensors()[0].dtype()).to(vars.tensors()[0].device())
    })
}
