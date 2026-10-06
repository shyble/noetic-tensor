//! `GatedMlp`: the SiLU-gated MLP, `(in(x) · silu(gate(x))) · out`, in that order; and the plain
//! MLP `out(act(in(x)))`. The gated MLP takes an optional caller-supplied extension
//! (`MlpExtension`): built once per MLP after its weights, it may add vars, and its transform is
//! applied to the gate's pre-activation and to the hidden activations.

use super::activation::Activation;
use super::linear::{linear, Linear};
use super::module::Module;
use super::var_builder::VarBuilder;
use crate::error::Result;
use crate::tensor::{silu, Tensor};
use std::fmt;
use std::sync::Arc;

/// A caller-supplied extension of the gated MLP. `build` runs once per gated MLP, right after its
/// three weights are created, and may create further vars through `vb` (in init mode they follow
/// "mlp_out", in the order `build` asks for them). The returned transform is applied on every pass.
pub trait MlpExtension: Send + Sync + fmt::Debug {
    /// A stable name: two extensions are equal when their names are.
    fn name(&self) -> &str;
    /// The transform of one gated MLP of hidden width `hidden`.
    fn build(&self, hidden: usize, vb: &VarBuilder) -> Result<Arc<dyn MlpTransform>>;
}

/// The per-pass part of an `MlpExtension`. Both methods default to the identity.
pub trait MlpTransform: Send + Sync + fmt::Debug {
    /// The gate's pre-activation `gate(x)`, `[S, N, m]` (or with more middle dimensions), before the SiLU.
    fn gate(&self, pre: Tensor) -> Tensor {
        pre
    }
    /// The hidden activations `in(x) · silu(·)`, before the output projection.
    fn hidden(&self, hidden: Tensor) -> Tensor {
        hidden
    }
}

/// A shared `MlpExtension`, compared by name.
#[derive(Clone)]
pub struct MlpExt(pub Arc<dyn MlpExtension>);

impl MlpExt {
    pub fn new(e: impl MlpExtension + 'static) -> Self {
        MlpExt(Arc::new(e))
    }
}

impl PartialEq for MlpExt {
    fn eq(&self, other: &Self) -> bool {
        self.0.name() == other.0.name()
    }
}

impl fmt::Debug for MlpExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MlpExt({:?})", self.0.name())
    }
}

#[derive(Clone, Debug)]
pub struct GatedMlp {
    w_in: Linear,
    w_gate: Linear,
    w_out: Linear,
    ext: Option<Arc<dyn MlpTransform>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GatedMlpConfig {
    pub d: usize,
    pub hidden: usize,
    /// An optional extension (None: the plain gated MLP).
    pub ext: Option<MlpExt>,
}

impl GatedMlp {
    pub fn new(w_in: Linear, w_gate: Linear, w_out: Linear) -> Self {
        GatedMlp { w_in, w_gate, w_out, ext: None }
    }

    /// As `new`, with an extension's transform.
    pub fn with_transform(w_in: Linear, w_gate: Linear, w_out: Linear, ext: Option<Arc<dyn MlpTransform>>) -> Self {
        GatedMlp { w_in, w_gate, w_out, ext }
    }

    /// The hidden activations `in(x) · silu(gate(x))`, `[S, N, m]`, with the extension's transform.
    pub fn hidden(&self, xs: &Tensor) -> Tensor {
        let gate_pre = self.w_gate.forward(xs);
        let gate_pre = match &self.ext {
            Some(t) => t.gate(gate_pre),
            None => gate_pre,
        };
        let hdn = self.w_in.forward(xs) * silu(gate_pre);
        match &self.ext {
            Some(t) => t.hidden(hdn),
            None => hdn,
        }
    }

    /// The output projection of hidden activations.
    pub fn project(&self, hidden: &Tensor) -> Tensor {
        self.w_out.forward(hidden)
    }

    /// The extension's transform, if any.
    pub fn transform(&self) -> Option<&Arc<dyn MlpTransform>> {
        self.ext.as_ref()
    }
}

impl Module for GatedMlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.project(&self.hidden(xs))
    }
}

/// The gated MLP's vars, in order: "mlp_in" `[d, m]`, "mlp_gate" `[d, m]` (both N(0, 1/d)),
/// "mlp_out" `[m, d]` (N(0, 1/m)), then the extension's, if any.
pub fn gated_mlp(cfg: GatedMlpConfig, vb: &VarBuilder) -> Result<GatedMlp> {
    let (d, m) = (cfg.d, cfg.hidden);
    let w_in = linear(d, m, "mlp_in", vb)?;
    let w_gate = linear(d, m, "mlp_gate", vb)?;
    let w_out = linear(m, d, "mlp_out", vb)?;
    let ext = match &cfg.ext {
        Some(e) => Some(e.0.build(m, vb)?),
        None => None,
    };
    Ok(GatedMlp::with_transform(w_in, w_gate, w_out, ext))
}

/// SwiGLU (`(in(x) · silu(gate(x))) · out`): the gated MLP without an extension.
pub fn swiglu(d: usize, hidden: usize, vb: &VarBuilder) -> Result<GatedMlp> {
    gated_mlp(GatedMlpConfig { d, hidden, ext: None }, vb)
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
