//! `GatedMlp`: the SiLU-gated MLP (SwiGLU), `(in(x) · silu(gate(x))) · out`, in that order; and
//! the plain MLP `out(act(in(x)))`.

use super::activation::Activation;
use super::linear::{linear, Linear};
use super::module::Module;
use super::var_builder::VarBuilder;
use crate::error::Result;
use crate::tensor::{silu, Tensor};

#[derive(Clone, Debug)]
pub struct GatedMlp {
    w_in: Linear,
    w_gate: Linear,
    w_out: Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedMlpConfig {
    pub d: usize,
    pub hidden: usize,
}

impl GatedMlp {
    pub fn new(w_in: Linear, w_gate: Linear, w_out: Linear) -> Self {
        GatedMlp { w_in, w_gate, w_out }
    }

    /// The hidden activations `in(x) · silu(gate(x))`, `[S, N, m]`.
    pub fn hidden(&self, xs: &Tensor) -> Tensor {
        self.w_in.forward(xs) * silu(self.w_gate.forward(xs))
    }

    /// The output projection of hidden activations.
    pub fn project(&self, hidden: &Tensor) -> Tensor {
        self.w_out.forward(hidden)
    }
}

impl Module for GatedMlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.project(&self.hidden(xs))
    }
}

/// The gated MLP's vars, in this order: "mlp_in" `[d, m]`, "mlp_gate" `[d, m]` (both N(0, 1/d))
/// and "mlp_out" `[m, d]` (N(0, 1/m)).
pub fn gated_mlp(cfg: GatedMlpConfig, vb: &VarBuilder) -> Result<GatedMlp> {
    let (d, m) = (cfg.d, cfg.hidden);
    let w_in = linear(d, m, "mlp_in", vb)?;
    let w_gate = linear(d, m, "mlp_gate", vb)?;
    let w_out = linear(m, d, "mlp_out", vb)?;
    Ok(GatedMlp::new(w_in, w_gate, w_out))
}

/// SwiGLU, `(in(x) · silu(gate(x))) · out`: the gated MLP at width `d` and hidden width `hidden`.
pub fn swiglu(d: usize, hidden: usize, vb: &VarBuilder) -> Result<GatedMlp> {
    gated_mlp(GatedMlpConfig { d, hidden }, vb)
}

/// The plain MLP `out(act(in(x)))`.
#[derive(Clone, Debug)]
pub struct Mlp {
    w_in: Linear,
    w_out: Linear,
    act: Activation,
}

impl Mlp {
    pub fn new(w_in: Linear, w_out: Linear, act: Activation) -> Self {
        Mlp { w_in, w_out, act }
    }

    pub fn activation(&self) -> Activation {
        self.act
    }

    /// The hidden activations `act(in(x))`.
    pub fn hidden(&self, xs: &Tensor) -> Tensor {
        self.act.apply(&self.w_in.forward(xs))
    }

    pub fn project(&self, hidden: &Tensor) -> Tensor {
        self.w_out.forward(hidden)
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.project(&self.hidden(xs))
    }
}

/// A plain MLP: "mlp_in" `[d, m]` (N(0, 1/d)) and "mlp_out" `[m, d]` (N(0, 1/m)).
pub fn mlp(d: usize, hidden: usize, act: Activation, vb: &VarBuilder) -> Result<Mlp> {
    Ok(Mlp::new(linear(d, hidden, "mlp_in", vb)?, linear(hidden, d, "mlp_out", vb)?, act))
}
