//! SGD with momentum, dampening and Nesterov, in torch.optim.SGD's form:
//! `g ← g + wd·p`; with momentum `buf ← g` on a var's first step, else `buf ← μ·buf + (1 − τ)·g`;
//! then `g ← g + μ·buf` (Nesterov) or `g ← buf`; and `p ← p − lr·scale·mult · g`, with the
//! per-seed rates and multiplier tensors of the `ParamGroups` applied to the update. Frozen vars
//! are skipped; lazy rows are Adam's only (an error here).

use super::optim::{Optimizer, ParamGroups};
use super::var::VarMap;
use crate::tensor::Tensor;

#[derive(Clone, Debug, PartialEq)]
pub struct SgdConfig {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub weight_decay: f64,
}

impl Default for SgdConfig {
    fn default() -> Self {
        SgdConfig { lr: 1e-2, momentum: 0.0, dampening: 0.0, nesterov: false, weight_decay: 0.0 }
    }
}

pub struct Sgd {
    pub cfg: SgdConfig,
    groups: ParamGroups,
    buf: Vec<Option<Tensor>>,
    seed_lr: Option<Tensor>,
}

impl Sgd {
    pub fn new(cfg: SgdConfig, groups: ParamGroups, vars: &VarMap) -> Self {
        assert_eq!(groups.names(), vars.names(), "the parameter groups are for another var map");
        assert!(!(cfg.nesterov && (cfg.momentum <= 0.0 || cfg.dampening != 0.0)), "Nesterov needs momentum and zero dampening");
        assert!((0..vars.len()).all(|i| !groups.options(i).lazy_rows), "lazy rows are an Adam option");
        let seed_lr = super::adam::seed_lr_tensor(&groups, vars);
        Sgd { cfg, groups, buf: vec![None; vars.len()], seed_lr }
    }

    pub fn groups_mut(&mut self) -> &mut ParamGroups {
        &mut self.groups
    }

    /// The momentum buffers (None before a var's first step or without momentum).
    pub fn buffers(&self) -> &[Option<Tensor>] {
        &self.buf
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, vars: &mut VarMap, grads: Vec<Option<Tensor>>, lr_scale: f64) {
        assert_eq!(grads.len(), self.buf.len(), "one gradient slot per var");
        let c = &self.cfg;
        for (i, g) in grads.into_iter().enumerate() {
            let Some(g) = g else { continue };
            let opts = self.groups.options(i);
            if opts.frozen {
                continue;
            }
            let p = vars.tensors()[i].clone();
            let g = if c.weight_decay > 0.0 { g + p.clone().mul_scalar(c.weight_decay) } else { g };
            let g = if c.momentum > 0.0 {
                let buf = match self.buf[i].take() {
                    None => g.clone(),
                    Some(b) => b.mul_scalar(c.momentum) + g.clone().mul_scalar(1.0 - c.dampening),
                };
                self.buf[i] = Some(buf.clone());
                if c.nesterov { g + buf.mul_scalar(c.momentum) } else { buf }
            } else {
                g
            };
            let mut upd = g;
            if let Some(l) = &self.seed_lr {
                upd = upd * l.clone();
            }
            if let Some(mu) = self.groups.multiplier(i) {
                upd = upd * mu.clone();
            }
            let p = p - upd.mul_scalar(c.lr * lr_scale * opts.lr_mult);
            vars.set_index(i, p).expect("the update keeps the var's shape");
        }
    }
}
