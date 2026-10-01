//! Activations in their standard formulations: each
//! is one node on the tape with the analytic backward PyTorch uses, computed in the tensor's own
//! dtype. The forward values of softmax, log_softmax and logsumexp equal burn 0.21's.

use super::autodiff::{attach, record_with_output};
use super::dtype::FloatElem;
use super::storage::Storage;
use super::DType;
use super::kernels as k;
use super::Tensor;

/// softmax along `dim`: `exp(x − max) / Σ exp(x − max)`, the max taken per lane (a lane with NaN
/// or with every entry −∞ gives NaN). Backward: `y · (g − Σ g·y)` from the saved output.
pub fn softmax(x: Tensor, dim: usize) -> Tensor {
    let xu = x.clone().untracked();
    let max = xu.clone().max_dim(dim);
    let e = xu.sub(max).exp();
    let y = e.clone().div(e.sum_dim(dim));
    attach(y, &[&x], move |y| {
        Box::new(move |g, _| {
            let s = g.clone().mul(y.clone()).sum_dim(dim);
            vec![Some(y.clone().mul(g.sub(s)))]
        })
    })
}

/// log_softmax along `dim`: `(x − max) − log Σ exp(x − max)`. Backward: `g − exp(y) · Σ g` from
/// the saved output.
pub fn log_softmax(x: Tensor, dim: usize) -> Tensor {
    let xu = x.clone().untracked();
    let max = xu.clone().max_dim(dim);
    let shifted = xu.sub(max);
    let lse = shifted.clone().exp().sum_dim(dim).log();
    let y = shifted.sub(lse);
    attach(y, &[&x], move |y| {
        Box::new(move |g, _| {
            let s = g.clone().sum_dim(dim);
            vec![Some(g.sub(y.clone().exp().mul(s)))]
        })
    })
}

/// The logistic sigmoid, in the tensor's dtype, in the stable two-branch form: `1 / (1 + e^−x)`
/// for x ≥ 0 and `e^x / (1 + e^x)` below, so no intermediate overflows and small outputs keep
/// their relative precision (NaN stays NaN). Backward: `g · (1 − y) · y` from the saved output.
pub fn sigmoid(x: Tensor) -> Tensor {
    fn f<T: FloatElem>(a: T) -> T {
        if a >= T::ZERO {
            T::ONE / (T::ONE + (-a).exp())
        } else {
            let e = a.exp();
            e / (T::ONE + e)
        }
    }
    let st = match super::gpu::gpu_unary(&x, super::gpu::UnaryOp::Sigmoid) {
        Some(st) => st,
        None => match x.dtype() {
            DType::F64 => Storage::from_vec(k::map(&x.vals::<f64>(), f::<f64>)),
            _ => Storage::from_vec(k::map(&x.vals::<f32>(), f::<f32>)),
        },
    };
    record_with_output(st, x.layout.shape.clone(), &[&x], move |y| {
        Box::new(move |g, _| vec![Some(g.mul(y.clone().neg().add_scalar(1.0)).mul(y.clone()))])
    })
}

/// log Σ exp along `dim`, keeping it: `m + log Σ exp(x − m)` with m the lane's maximum, shifted
/// by 0 instead where m is ±∞ (PyTorch's rule: a lane of −∞ gives −∞, one with +∞ gives +∞,
/// not NaN); NaN propagates. Backward: `g · exp(x − out)` (the softmax of the lane).
pub fn logsumexp(x: Tensor, dim: usize) -> Tensor {
    let xu = x.clone().untracked();
    let m = xu.clone().max_dim(dim);
    let m = m.clone().mask_fill(m.abs().equal_elem(f64::INFINITY), 0.0);
    let out = xu.clone().sub(m.clone()).exp().sum_dim(dim).log().add(m);
    attach(out, &[&x], move |y| Box::new(move |g, _| vec![Some(g.mul(xu.clone().sub(y.clone()).exp()))]))
}

/// silu: `x · sigmoid(x)`.
pub fn silu(x: Tensor) -> Tensor {
    x.clone().mul(sigmoid(x))
}
