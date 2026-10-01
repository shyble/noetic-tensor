//! `LayerNorm`: `(x − mean) · (var + ε)^(-1/2) · weight + bias` over the last dimension
//! (biased variance, ε = 1e-5 by default); weight `[S, 1, d]` starts at 1, bias at 0.
//!
//! `RmsNorm`: `x · (mean(x²) + ε)^(-1/2) · scale` over the last dimension, with
//! a fixed operation order (square as `x · x`, mean, add ε, sqrt, recip, then two
//! products) and ε = 1e-6.

use super::init::Init;
use super::linear::to_rank;
use super::module::Module;
use super::var_builder::VarBuilder;
use crate::error::Result;
use crate::tensor::Tensor;

/// The default ε.
pub const RMS_EPS: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// From a scale `[S, 1, d]`.
    pub fn new(scale: Tensor, eps: f64) -> Self {
        RmsNorm { scale, eps }
    }

    pub fn scale(&self) -> &Tensor {
        &self.scale
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        rms_normalize(xs, self.eps) * to_rank(&self.scale, xs.rank())
    }
}

/// `x · (mean(x²) + ε)^(-1/2)` over the last dimension, without a scale.
pub fn rms_normalize(xs: &Tensor, eps: f64) -> Tensor {
    let last = xs.rank() - 1;
    let ms = xs.clone().powf_scalar(2.0).mean_dim(last);
    let inv = (ms + eps).sqrt().recip();
    xs.clone() * inv
}

/// An RMS norm over width `d` whose scale var `name` (`[S, 1, d]`) starts at 1.
pub fn rms_norm(d: usize, name: &str, vb: &VarBuilder) -> Result<RmsNorm> {
    Ok(RmsNorm::new(vb.get(&[1, d], name, Init::Const(1.0))?, RMS_EPS))
}

/// LayerNorm's default ε.
pub const LN_EPS: f64 = 1e-5;

#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    /// From a weight `[S, 1, d]` and an optional bias `[S, 1, d]`.
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f64) -> Self {
        LayerNorm { weight, bias, eps }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let last = xs.rank() - 1;
        let mean = xs.clone().mean_dim(last);
        let c = xs.clone() - mean;
        let var = c.clone().powf_scalar(2.0).mean_dim(last);
        let y = c * (var + self.eps).sqrt().recip() * to_rank(&self.weight, xs.rank());
        match &self.bias {
            Some(b) => y + to_rank(b, xs.rank()),
            None => y,
        }
    }
}

/// A LayerNorm over width `d`: weight var `name` (ones) and bias var `{name}_bias` (zeros).
pub fn layer_norm(d: usize, name: &str, vb: &VarBuilder) -> Result<LayerNorm> {
    let w = vb.get(&[1, d], name, Init::Const(1.0))?;
    let b = vb.get(&[1, d], &format!("{name}_bias"), Init::Const(0.0))?;
    Ok(LayerNorm::new(w, Some(b), LN_EPS))
}
