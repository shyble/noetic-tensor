//! `Linear`: `x · W (+ b)` with the weight in the seed-batched `[S, in, out]` layout (not
//! candle's `[out, in]`), so the product is `x.matmul(w)` with no transpose. On the
//! seed-batched token layout `[S, N, in]` it is exactly one matmul (and one add with a bias).

use super::init::Init;
use super::module::Module;
use super::var_builder::VarBuilder;
use crate::error::Result;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// From a weight `[S, in, out]` and an optional bias `[S, 1, out]`.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Linear { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn in_dim(&self) -> usize {
        self.weight.shape()[1]
    }

    pub fn out_dim(&self) -> usize {
        self.weight.shape()[2]
    }
}

/// Lift a `[S, a, b]` parameter to the rank of an input with extra middle dimensions
/// (`[S, 1, …, a, b]`); the rank-3 path adds no operation.
pub(crate) fn to_rank(p: &Tensor, rank: usize) -> Tensor {
    let mut p = p.clone();
    while p.rank() < rank {
        p = p.unsqueeze_dim(1);
    }
    p
}

impl Module for Linear {
    /// `[S, …, in]` → `[S, …, out]`.
    fn forward(&self, xs: &Tensor) -> Tensor {
        let y = xs.clone().matmul(to_rank(&self.weight, xs.rank()));
        match &self.bias {
            Some(b) => y + to_rank(b, xs.rank()),
            None => y,
        }
    }
}

/// A linear layer named `name` (the weight var) with the default init, N(0, 1/in), and no bias.
pub fn linear(in_dim: usize, out_dim: usize, name: &str, vb: &VarBuilder) -> Result<Linear> {
    linear_init(in_dim, out_dim, name, Init::fan_in(in_dim), vb)
}

/// A linear layer without bias, weight initialised by `init`.
pub fn linear_init(in_dim: usize, out_dim: usize, name: &str, init: Init, vb: &VarBuilder) -> Result<Linear> {
    Ok(Linear::new(vb.get(&[in_dim, out_dim], name, init)?, None))
}

/// A linear layer with a bias var `bias_name` (`[S, 1, out]`, zeros), created after the weight.
pub fn linear_b(in_dim: usize, out_dim: usize, name: &str, bias_name: &str, vb: &VarBuilder) -> Result<Linear> {
    let w = vb.get(&[in_dim, out_dim], name, Init::fan_in(in_dim))?;
    let b = vb.get(&[1, out_dim], bias_name, Init::Const(0.0))?;
    Ok(Linear::new(w, Some(b)))
}
