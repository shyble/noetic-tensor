//! Static loss scaling for mixed precision: the loss is multiplied by 2^k before the backward pass
//! and the gradients by 2^−k after it, so that small gradients survive the 16-bit rounding of a
//! reduced-precision matrix product's operands (`tensor::MatmulPrecision::Half`; f16 on Metal has
//! a narrow exponent range, bf16 on CUDA has f32's).
//!
//! The factor is a power of two, so both multiplications are exact in f32 wherever no value
//! overflows or falls below the normal range: the f32 parts of the backward pass give the same
//! values with or without the scale, and only the 16-bit operands see the larger magnitudes. A
//! step whose unscaled gradients are not all finite (the scaled backward pass overflowed) is to be
//! skipped (`all_finite`); the decision depends only on the gradients' values, so it is the same
//! on every run and, for summed gradients, on every rank. The scale is static (no schedule raises or
//! lowers it), so it carries no state that a checkpoint would have to hold.

use crate::error::{NnError, Result};
use crate::tensor::Tensor;

/// A static loss scale of 2^k (k in −60..=60; k = 0 is no scaling).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LossScale {
    log2: i32,
}

impl LossScale {
    /// No scaling (2^0).
    pub fn none() -> LossScale {
        LossScale { log2: 0 }
    }

    /// A scale of 2^`log2`.
    pub fn power_of_two(log2: i32) -> Result<LossScale> {
        if !(-60..=60).contains(&log2) {
            return Err(NnError::Dist(format!("a loss scale of 2^{log2} (−60 to 60)")));
        }
        Ok(LossScale { log2 })
    }

    pub fn log2(&self) -> i32 {
        self.log2
    }

    /// The factor 2^k.
    pub fn factor(&self) -> f64 {
        2f64.powi(self.log2)
    }

    /// The loss times 2^k (the loss itself when k = 0).
    pub fn scale(&self, loss: Tensor) -> Tensor {
        if self.log2 == 0 {
            loss
        } else {
            loss.mul_scalar(self.factor())
        }
    }

    /// Every gradient times 2^−k, in place (untouched when k = 0).
    pub fn unscale(&self, grads: &mut [Option<Tensor>]) {
        if self.log2 == 0 {
            return;
        }
        let inv = 2f64.powi(-self.log2);
        for g in grads.iter_mut() {
            if let Some(t) = g.take() {
                *g = Some(t.mul_scalar(inv));
            }
        }
    }

    /// For records: `loss-scale=2^k`.
    pub fn name(&self) -> String {
        format!("loss-scale=2^{}", self.log2)
    }
}

/// Whether every element of every gradient is finite. One reduction and one host read per
/// gradient: the sum of `g · 0`, which is ±0 for finite values and NaN when any element is
/// infinite or NaN.
pub fn all_finite(grads: &[Option<Tensor>]) -> bool {
    grads.iter().flatten().all(|g| g.clone().mul_scalar(0.0).sum().to_vec_f64()[0] == 0.0)
}
