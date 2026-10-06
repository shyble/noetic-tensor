//! The optimizer trait and parameter groups. An optimizer updates the untracked
//! vars of a `VarMap` from gradients by var index (`VarMap::grads` of the lifted map); None
//! means the var took no part and is left alone.
//!
//! `ParamGroups` holds the per-var options: a learning-rate multiplier, frozen (never updated), lazy
//! rows (a row of a `[S, R, C]` var with no gradient
//! this step keeps its moments and does not move) and an optional multiplier tensor broadcast
//! against the var (per-element learning-rate factors); plus per-seed learning rates, so a learning-rate grid
//! runs on the seed axis in one pass.

use super::var::VarMap;
use crate::tensor::Tensor;

pub trait Optimizer {
    /// One update at the optimizer's learning rate × `lr_scale` (a schedule's value).
    fn step(&mut self, vars: &mut VarMap, grads: Vec<Option<Tensor>>, lr_scale: f64);
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamOptions {
    pub lr_mult: f64,
    pub frozen: bool,
    pub lazy_rows: bool,
}

impl Default for ParamOptions {
    fn default() -> Self {
        ParamOptions { lr_mult: 1.0, frozen: false, lazy_rows: false }
    }
}

#[derive(Clone, Debug)]
pub struct ParamGroups {
    names: Vec<String>,
    opts: Vec<ParamOptions>,
    mult: Vec<Option<Tensor>>,
    seed_lr: Option<Vec<f64>>,
}

impl ParamGroups {
    /// Default options for every var of `vars`, in its order.
    pub fn new(vars: &VarMap) -> Self {
        let n = vars.len();
        ParamGroups { names: vars.names().to_vec(), opts: vec![ParamOptions::default(); n], mult: vec![None; n], seed_lr: None }
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    #[track_caller]
    fn index(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no var named {name:?}"))
    }

    pub fn options(&self, i: usize) -> &ParamOptions {
        &self.opts[i]
    }

    pub fn options_mut(&mut self, name: &str) -> &mut ParamOptions {
        let i = self.index(name);
        &mut self.opts[i]
    }

    /// Apply `f` to the options of every var whose name satisfies `pred`.
    pub fn update_where(&mut self, pred: impl Fn(&str) -> bool, f: impl Fn(&mut ParamOptions)) -> &mut Self {
        for (n, o) in self.names.iter().zip(self.opts.iter_mut()) {
            if pred(n) {
                f(o);
            }
        }
        self
    }

    pub fn freeze(&mut self, name: &str) -> &mut Self {
        self.options_mut(name).frozen = true;
        self
    }

    pub fn set_lr_mult(&mut self, name: &str, mult: f64) -> &mut Self {
        self.options_mut(name).lr_mult = mult;
        self
    }

    pub fn set_lazy_rows(&mut self, name: &str) -> &mut Self {
        self.options_mut(name).lazy_rows = true;
        self
    }

    /// The learning-rate multiplier tensor of var `name` (broadcastable to it: `[S, 1, m]` on
    /// unit columns, `[S, m, 1]` on unit rows).
    pub fn set_multiplier(&mut self, name: &str, mult: Tensor) -> &mut Self {
        let i = self.index(name);
        self.mult[i] = Some(mult);
        self
    }

    pub fn multiplier(&self, i: usize) -> Option<&Tensor> {
        self.mult[i].as_ref()
    }

    /// One learning-rate multiplier per seed.
    pub fn set_seed_lr(&mut self, lr: Vec<f64>) -> &mut Self {
        self.seed_lr = Some(lr);
        self
    }

    pub fn seed_lr(&self) -> Option<&[f64]> {
        self.seed_lr.as_deref()
    }
}
