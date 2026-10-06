//! An exponential moving average of the vars: the
//! average starts at zero, `acc · decay + x · (1 − decay)` per update, and `averaged()` divides by
//! 1 − decayᵗ (bias correction), so early evaluations are not dominated by the initial weights.

use super::var::VarMap;
use crate::tensor::Tensor;

pub struct Ema {
    pub decay: f64,
    acc: Vec<Tensor>,
    names: Vec<String>,
    t: u64,
}

impl Ema {
    pub fn new(decay: f64, vars: &VarMap) -> Self {
        Ema { decay, acc: vars.tensors().iter().map(|x| x.zeros_like()).collect(), names: vars.names().to_vec(), t: 0 }
    }

    pub fn update(&mut self, vars: &VarMap) {
        assert_eq!(vars.names(), self.names.as_slice(), "the EMA is for another var map");
        self.t += 1;
        for (e, x) in self.acc.iter_mut().zip(vars.tensors()) {
            *e = e.clone().mul_scalar(self.decay) + x.clone().mul_scalar(1.0 - self.decay);
        }
    }

    /// The bias-corrected average (one update at least is assumed, as the trainer's).
    pub fn averaged(&self) -> VarMap {
        let c = 1.0 - self.decay.powi(self.t.max(1) as i32);
        let mut out = VarMap::new();
        for (n, x) in self.names.iter().zip(&self.acc) {
            out.insert(n.clone(), x.clone().div_scalar(c)).expect("distinct names");
        }
        out
    }
}
