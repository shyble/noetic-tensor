//! Finite-difference checks of every differentiable operation, independent of any reference: the
//! gradient of `sum(op(x) · w)` against central differences. f32 throughout, so the step is
//! 1e-2 and the tolerance 5e-3 · (1 + |g|) (rounding of the loss over 2·step dominates).
//! Inputs keep away from kinks (abs, clamp, max, topk ties).

use super::*;
use crate::tensor::{self as nt, BoolTensor};

fn loss_of(op: &dyn Fn(Tensor) -> Tensor, x: &[f32], shape: &[usize], w: &[f32]) -> f64 {
    let y = op(Tensor::from_data(x.to_vec(), shape.to_vec()));
    y.as_slice().iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum()
}

fn fd_check(name: &str, shape: &[usize], x: &[f32], op: impl Fn(Tensor) -> Tensor) {
    let xt = Tensor::from_data(x.to_vec(), shape.to_vec()).require_grad();
    let y = op(xt.clone());
    let w = rnd(y.numel(), 7, -1.0, 1.0);
    let g = (y.clone() * Tensor::from_data(w.clone(), y.shape().to_vec())).sum().backward();
    let g = xt.grad(&g).expect("gradient").to_vec();
    let eps = 1e-2f32;
    for j in (0..x.len()).step_by((x.len() / 12).max(1)) {
        let (mut p, mut m) = (x.to_vec(), x.to_vec());
        p[j] += eps;
        m[j] -= eps;
        let fd = (loss_of(&op, &p, shape, &w) - loss_of(&op, &m, shape, &w)) / (2.0 * eps as f64);
        let tol = 5e-3 * (1.0 + g[j].abs() as f64);
        assert!((fd - g[j] as f64).abs() < tol, "{name}[{j}]: finite difference {fd} vs autodiff {}", g[j]);
    }
}

#[test]
fn every_operation_matches_finite_differences() {
    let s = [2usize, 4, 6];
    let x = rnd(48, 31, -1.5, 1.5);
    let pos = rnd(48, 32, 0.5, 2.0);
    let c = Tensor::from_data(rnd(48, 33, 0.5, 1.5), s);
    let cb = Tensor::from_data(rnd(12, 34, 0.5, 1.5), [2, 1, 6]);
    fd_check("add", &s, &x, |t| t + c.clone());
    fd_check("sub", &s, &x, |t| c.clone() - t);
    fd_check("mul broadcast", &s, &x, |t| t * cb.clone());
    fd_check("mul self", &s, &x, |t| t.clone() * t);
    fd_check("div lhs", &s, &x, |t| t / c.clone());
    fd_check("div rhs", &s, &pos, |t| c.clone() / t);
    fd_check("scalars", &s, &x, |t| ((t + 0.5).mul_scalar(3.0) - 1.0).div_scalar(2.0));
    fd_check("neg exp", &s, &x, |t| (-t).exp());
    fd_check("log", &s, &pos, |t| t.log());
    fd_check("sqrt", &s, &pos, |t| t.sqrt());
    fd_check("recip", &s, &pos, |t| t.recip());
    fd_check("abs", &s, &pos, |t| (t - 1.25).abs());
    fd_check("powf 2", &s, &x, |t| t.powf_scalar(2.0));
    fd_check("powf 1.5", &s, &pos, |t| t.powf_scalar(1.5));
    fd_check("clamp_min", &s, &x, |t| t.clamp_min(0.05));
    fd_check("sum", &s, &x, |t| t.sum());
    fd_check("mean", &s, &x, |t| t.mean());
    for d in 0..3 {
        fd_check("sum_dim", &s, &x, |t| t.sum_dim(d));
        fd_check("mean_dim", &s, &x, |t| t.mean_dim(d));
        fd_check("max_dim", &s, &x, |t| t.max_dim(d));
        fd_check("softmax", &s, &x, |t| nt::softmax(t, d));
        fd_check("log_softmax", &s, &x, |t| nt::log_softmax(t, d));
        fd_check("logsumexp", &s, &x, |t| nt::logsumexp(t, d));
    }
    fd_check("sigmoid", &s, &x, nt::sigmoid);
    fd_check("silu", &s, &x, nt::silu);
    fd_check("reshape swap", &s, &x, |t| t.reshape([8, 6]).swap_dims(0, 1).exp());
    fd_check("unsqueeze expand", &s, &x, |t| t.unsqueeze_dim(1).expand([2, 3, 4, 6]).exp());
    fd_check("slice", &s, &x, |t| t.slice([0..2, 1..3, 2..6]).exp());
    fd_check("slice_assign", &s, &x, |t| Tensor::zeros([2, 5, 6]).slice_assign([0..2, 1..5, 0..6], t.exp()));
    fd_check("cat", &s, &x, |t| Tensor::cat(vec![t.clone(), t.exp()], 1));
    let m = Tensor::from_data(rnd(6 * 5, 35, -1.0, 1.0), [1, 6, 5]);
    fd_check("matmul lhs", &s, &x, |t| t.matmul(m.clone()));
    let l = Tensor::from_data(rnd(2 * 3 * 4, 36, -1.0, 1.0), [2, 3, 4]);
    fd_check("matmul rhs", &s, &x, |t| l.clone().matmul(t));
    fd_check("mask_fill", &s, &x, |t| t.mask_fill(BoolTensor::tril_mask([4, 6], 0).unsqueeze_dim(0).expand([2, 4, 6]), -3.0));
    let gi: Vec<i64> = (0..2 * 4 * 9).map(|i| (i * 5 % 6) as i64).collect();
    fd_check("gather", &s, &x, |t| t.gather(2, our_ints(&gi, [2, 4, 9])));
    fd_check("select", &s, &x, |t| t.select(2, our_ints(&[5, 0, 5], [3])));
    fd_check("topk", &s, &x, |t| t.topk_with_indices(3, 2).0);
}
