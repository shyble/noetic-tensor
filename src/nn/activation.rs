//! Activations, composed of tensor operations so they carry gradient.
//! - `silu` and `sigmoid` are the tensor core's (the stable sigmoid, in the tensor's
//!   dtype).
//! - `gelu_tanh`: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, with `tanh(z) = 2·sigmoid(2z) − 1`
//!   (finite value and gradient for every finite x).
//! - `gelu_erf`: `0.5·x·(1 + erf(x/√2))` on the in-house `erf`.
//! - `erf`: Abramowitz & Stegun 7.1.26, `1 − (a₁t + … + a₅t⁵)·e^(−x²)`, t = 1/(1 + p|x|), odd
//!   extension. **Error bound: |erf_approx(x) − erf(x)| ≤ 1.5e-7 for every real x** (A&S), plus
//!   rounding of the dtype (measured over [−6, 6] in steps of 0.01: 1.39e-7 in f64, 2.7e-7 in
//!   f32; tested at ≤ 1.76e-7 and ≤ 4e-7); so |gelu_erf(x) − GELU(x)| ≤ 0.75e-7·|x| plus
//!   rounding. Its gradient is the approximation's derivative, which differs from
//!   2/√π·e^(−x²) by up to 5.02e-6 on that grid (tested at ≤ 6e-6), and is 0 at exactly x = 0
//!   (where sign(x) has no gradient); `gelu_erf`'s gradient is unaffected there (the erf term is
//!   multiplied by x).
//! - `relu`: `clamp_min(0)` (gradient 1 where x ≥ 0).

use crate::tensor::{sigmoid, Tensor};

pub use crate::tensor::silu;

/// The coefficients of A&S 7.1.26.
const ERF_P: f64 = 0.327_591_1;
const ERF_A: [f64; 5] = [0.254_829_592, -0.284_496_736, 1.421_413_741, -1.453_152_027, 1.061_405_429];

/// erf, A&S 7.1.26 (error ≤ 1.5e-7).
pub fn erf(x: &Tensor) -> Tensor {
    let ax = x.clone().abs();
    let t = (ax.clone().mul_scalar(ERF_P) + 1.0).recip();
    // Horner: ((((a5 t + a4) t + a3) t + a2) t + a1) t
    let mut poly = t.clone().mul_scalar(ERF_A[4]) + ERF_A[3];
    for a in [ERF_A[2], ERF_A[1], ERF_A[0]] {
        poly = poly * t.clone() + a;
    }
    let poly = poly * t;
    let y = (poly * (ax.clone() * ax).neg().exp()).neg() + 1.0;
    x.clone().sign() * y
}

/// tanh as `2·sigmoid(2z) − 1`.
pub fn tanh(z: &Tensor) -> Tensor {
    sigmoid(z.clone().mul_scalar(2.0)).mul_scalar(2.0) - 1.0
}

/// GELU, tanh form.
pub fn gelu_tanh(x: &Tensor) -> Tensor {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let inner = (x.clone() + x.clone() * x.clone() * x.clone() * 0.044715).mul_scalar(c);
    x.clone().mul_scalar(0.5) * (tanh(&inner) + 1.0)
}

/// GELU, erf form (on the in-house erf).
pub fn gelu_erf(x: &Tensor) -> Tensor {
    x.clone().mul_scalar(0.5) * (erf(&x.clone().mul_scalar(std::f64::consts::FRAC_1_SQRT_2)) + 1.0)
}

pub fn relu(x: &Tensor) -> Tensor {
    x.clone().clamp_min(0.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    #[default]
    Silu,
    GeluTanh,
    GeluErf,
    Relu,
}

impl Activation {
    pub fn apply(self, x: &Tensor) -> Tensor {
        match self {
            Activation::Silu => silu(x.clone()),
            Activation::GeluTanh => gelu_tanh(x),
            Activation::GeluErf => gelu_erf(x),
            Activation::Relu => relu(x),
        }
    }
}
