//! Dropout: inverted dropout (kept elements scaled by 1/(1 − p)) with
//! labelled random streams. Call c of the dropout labelled L draws seed slot i's mask from
//! `rng::derive(root, "dropout:{L}:{c}:{i}")`, so a seed's mask never depends on S and a run is
//! reproducible from the root. In eval mode, or at p = 0, the input is
//! returned unchanged and no operation or draw happens.

use super::module::ModuleT;
use crate::rng::derive;
use crate::tensor::Tensor;
use rand::Rng;
use std::cell::Cell;

#[derive(Clone, Debug)]
pub struct Dropout {
    p: f64,
    root: u64,
    label: String,
    calls: Cell<u64>,
}

impl Dropout {
    pub fn new(p: f64, root: u64, label: impl Into<String>) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability {p} outside [0, 1)");
        Dropout { p, root, label: label.into(), calls: Cell::new(0) }
    }

    pub fn p(&self) -> f64 {
        self.p
    }

    /// Training calls made so far (the next call's index).
    pub fn calls(&self) -> u64 {
        self.calls.get()
    }

    /// The mask of call `call` for a `[S, …]` input: 0 or 1/(1 − p) per element.
    pub fn mask(&self, shape: &[usize], call: u64, like: &Tensor) -> Tensor {
        let s = shape[0];
        let per: usize = shape[1..].iter().product();
        let keep = 1.0 / (1.0 - self.p);
        let mut v = Vec::with_capacity(s * per);
        for i in 0..s {
            let mut rng = derive(self.root, &format!("dropout:{}:{call}:{i}", self.label));
            v.extend((0..per).map(|_| if rng.gen::<f64>() < self.p { 0.0 } else { keep }));
        }
        Tensor::from_f64s(v, shape.to_vec(), like.dtype()).to(like.device())
    }

    /// Dropout with an explicit call index (does not advance the counter).
    pub fn forward_at(&self, xs: &Tensor, train: bool, call: u64) -> Tensor {
        if !train || self.p == 0.0 {
            return xs.clone();
        }
        xs.clone() * self.mask(xs.shape(), call, xs)
    }
}

impl ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        if !train || self.p == 0.0 {
            return xs.clone();
        }
        let c = self.calls.get();
        self.calls.set(c + 1);
        self.forward_at(xs, train, c)
    }
}
