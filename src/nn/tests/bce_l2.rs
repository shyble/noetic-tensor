//! `binary_cross_entropy_with_logits` and `l2_normalize` on the CPU reference backend, against f64
//! host references and central finite differences.

use super::joint_embed::fd_input;
use super::*;
use crate::nn::*;
use crate::tensor::DType;

fn t64(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F64)
}

fn t32(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F32)
}

// ------------------------------------------------------------------------------------ BCE

/// The stable elementwise loss on the host, in f64: `(1 − y)·x + l·(max(−x, 0) + ln(1 +
/// exp(−|x|)))` with `l = 1 + (p − 1)·y`, times w.
fn bce_host(x: f64, y: f64, w: f64, p: f64) -> f64 {
    let l = 1.0 + (p - 1.0) * y;
    w * ((1.0 - y) * x + l * ((-x).max(0.0) + (-x.abs()).exp().ln_1p()))
}

/// The plain formula, `−w·(p·y·ln σ(x) + (1 − y)·ln(1 − σ(x)))`, safe for small |x|.
fn bce_plain(x: f64, y: f64, w: f64, p: f64) -> f64 {
    let s = 1.0 / (1.0 + (-x).exp());
    -w * (p * y * s.ln() + (1.0 - y) * (1.0 - s).ln())
}

/// Per-seed reduction of host values `[S, n]`.
fn reduce_host(v: &[f64], s: usize, r: Reduction) -> Vec<f64> {
    let n = v.len() / s;
    match r {
        Reduction::None => v.to_vec(),
        Reduction::Mean => v.chunks(n).map(|c| c.iter().sum::<f64>() / n as f64).collect(),
        Reduction::Sum => v.chunks(n).map(|c| c.iter().sum::<f64>()).collect(),
    }
}

struct BceCase {
    shape: [usize; 3],
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    p: Vec<f64>,
}

fn bce_case(lo: f64, hi: f64, seed: u64) -> BceCase {
    let shape = [2, 5, 4];
    let n = shape.iter().product();
    // Soft and hard targets.
    let y = rnd(n, seed + 1, 0.0, 1.0).into_iter().enumerate().map(|(i, u)| if i % 3 == 0 { u.round() } else { u }).collect();
    BceCase { shape, x: rnd(n, seed, lo, hi), y, w: rnd(n, seed + 2, 0.2, 2.0), p: rnd(shape[2], seed + 3, 0.5, 4.0) }
}

impl BceCase {
    fn weights(&self, weight: bool, pos: bool, dtype: DType) -> BceWeights {
        BceWeights {
            weight: weight.then(|| Tensor::from_f64s(self.w.clone(), self.shape.to_vec(), dtype)),
            pos_weight: pos.then(|| Tensor::from_f64s(self.p.clone(), vec![self.shape[2]], dtype)),
        }
    }

    fn host(&self, weight: bool, pos: bool, f: fn(f64, f64, f64, f64) -> f64) -> Vec<f64> {
        let c = self.shape[2];
        (0..self.x.len()).map(|i| f(self.x[i], self.y[i], if weight { self.w[i] } else { 1.0 }, if pos { self.p[i % c] } else { 1.0 })).collect()
    }

    fn run(&self, weight: bool, pos: bool, r: Reduction, dtype: DType) -> Vec<f64> {
        let w = self.weights(weight, pos, dtype);
        let l = binary_cross_entropy_with_logits(Tensor::from_f64s(self.x.clone(), self.shape.to_vec(), dtype), &Tensor::from_f64s(self.y.clone(), self.shape.to_vec(), dtype), Some(&w), r);
        l.to_vec_f64()
    }
}

const OPTIONS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];
const REDUCTIONS: [Reduction; 3] = [Reduction::None, Reduction::Mean, Reduction::Sum];

/// Every weight option and reduction against the f64 host loss: within 1e-12 in f64 and 1e-6 in
/// f32, with logits up to ±30.
#[test]
fn bce_with_logits_matches_f64_reference() {
    let c = bce_case(-30.0, 30.0, 600);
    for (weight, pos) in OPTIONS {
        let host = c.host(weight, pos, bce_host);
        for r in REDUCTIONS {
            let want = reduce_host(&host, 2, r);
            assert_close(&format!("BCE f64 weight {weight} pos {pos} {r:?}"), &c.run(weight, pos, r, DType::F64), &want, 1e-12);
            assert_close(&format!("BCE f32 weight {weight} pos {pos} {r:?}"), &c.run(weight, pos, r, DType::F32), &want, 1e-6);
        }
    }
    // No weights at all is the same as empty weights, to the bit; the default reduction is the mean.
    let a = binary_cross_entropy_with_logits(t32(c.x.clone(), &c.shape), &t32(c.y.clone(), &c.shape), None, Reduction::default()).to_vec_f64();
    let b = c.run(false, false, Reduction::Mean, DType::F32);
    assert_eq!(a, b);
    // pos_weight 1 equals no pos_weight within f64 rounding.
    let ones = BceWeights { weight: None, pos_weight: Some(t64(vec![1.0; 4], &[4])) };
    let p1 = binary_cross_entropy_with_logits(t64(c.x.clone(), &c.shape), &t64(c.y.clone(), &c.shape), Some(&ones), Reduction::None).to_vec_f64();
    assert_close("BCE pos_weight 1", &p1, &c.run(false, false, Reduction::None, DType::F64), 1e-14);
    // A pos_weight of the logits' rank broadcasts like the [C] form, to the bit.
    let full = BceWeights { weight: None, pos_weight: Some(t64(c.p.clone(), &[1, 1, 4])) };
    let pf = binary_cross_entropy_with_logits(t64(c.x.clone(), &c.shape), &t64(c.y.clone(), &c.shape), Some(&full), Reduction::None).to_vec_f64();
    assert_eq!(pf, c.run(false, true, Reduction::None, DType::F64));
}

/// f64 central differences of the logits' and the targets' gradients, every option, at every
/// element (x = 0 included, where the gradient is 1/2 − y).
#[test]
fn bce_with_logits_matches_finite_differences() {
    let c = bce_case(-6.0, 6.0, 610);
    let mut x = c.x.clone();
    x[3] = 0.0;
    x[17] = 0.0;
    for (weight, pos) in OPTIONS {
        let w = c.weights(weight, pos, DType::F64);
        for r in REDUCTIONS {
            let y = t64(c.y.clone(), &c.shape);
            fd_input(&format!("BCE d/dx weight {weight} pos {pos} {r:?}"), &x, &c.shape, &|xt| binary_cross_entropy_with_logits(xt, &y, Some(&w), r), 1e-6, 1e-7);
            let xt = t64(x.clone(), &c.shape);
            fd_input(&format!("BCE d/dy weight {weight} pos {pos} {r:?}"), &c.y, &c.shape, &|yt| binary_cross_entropy_with_logits(xt.clone(), &yt, Some(&w), r), 1e-6, 1e-7);
        }
    }
    // The analytic gradient, σ(x) − y (times w, without pos_weight), at x = 0 as well.
    let xt = t64(x.clone(), &c.shape).require_grad();
    let w = c.weights(true, false, DType::F64);
    let g = binary_cross_entropy_with_logits(xt.clone(), &t64(c.y.clone(), &c.shape), Some(&w), Reduction::Sum).sum().backward();
    let want: Vec<f64> = (0..x.len()).map(|i| c.w[i] * (1.0 / (1.0 + (-x[i]).exp()) - c.y[i])).collect();
    assert_close("BCE gradient σ(x) − y", &xt.grad(&g).unwrap().to_vec_f64(), &want, 1e-14);
}

/// Logits up to |x| = 1e4, in f32 and f64: finite losses equal to the host's, and finite
/// gradients equal to the host's `w·((1 + (p − 1)·y)·σ(x) − p·y)`.
#[test]
fn bce_with_logits_is_finite_for_large_logits() {
    let shape = [2, 3, 4];
    let xs: Vec<f64> = (0..24).map(|i| [1e4, -1e4, 1e3, -1e3, 100.0, -100.0, 88.0, -88.0, 50.0, -50.0, 1e4, -1e4][i % 12] * if i >= 12 { 0.5 } else { 1.0 }).collect();
    let ys: Vec<f64> = (0..24).map(|i| [0.0, 1.0, 0.3][i % 3]).collect();
    let c = BceCase { shape, x: xs.clone(), y: ys.clone(), w: rnd(24, 620, 0.5, 2.0), p: vec![0.5, 1.0, 2.0, 3.0] };
    for dtype in [DType::F32, DType::F64] {
        for (weight, pos) in OPTIONS {
            let w = c.weights(weight, pos, dtype);
            let xt = Tensor::from_f64s(xs.clone(), shape.to_vec(), dtype).require_grad();
            let l = binary_cross_entropy_with_logits(xt.clone(), &Tensor::from_f64s(ys.clone(), shape.to_vec(), dtype), Some(&w), Reduction::None);
            let lv = l.to_vec_f64();
            assert!(lv.iter().all(|v| v.is_finite()), "{dtype:?} weight {weight} pos {pos}: loss {lv:?}");
            let tol = if dtype == DType::F64 { 1e-12 } else { 1e-6 };
            assert_close(&format!("{dtype:?} large-logit loss weight {weight} pos {pos}"), &lv, &c.host(weight, pos, bce_host), tol);
            let gv = xt.grad(&l.sum().backward()).unwrap().to_vec_f64();
            assert!(gv.iter().all(|v| v.is_finite()), "{dtype:?} weight {weight} pos {pos}: gradient {gv:?}");
            let want: Vec<f64> = (0..24)
                .map(|i| {
                    let (s, y) = (if xs[i] >= 0.0 { 1.0 / (1.0 + (-xs[i]).exp()) } else { xs[i].exp() / (1.0 + xs[i].exp()) }, ys[i]);
                    let p = if pos { c.p[i % 4] } else { 1.0 };
                    let w = if weight { c.w[i] } else { 1.0 };
                    w * ((1.0 + (p - 1.0) * y) * s - p * y) // d/dx of −p·y·ln σ − (1 − y)·ln(1 − σ)
                })
                .collect();
            assert_close(&format!("{dtype:?} large-logit gradient weight {weight} pos {pos}"), &gv, &want, tol);
        }
    }
}

/// Where the plain formula is safe (|x| ≤ 8, targets in [0, 1]), the stable form equals it:
/// within 1e-12 in f64 and 1e-6 in f32.
#[test]
fn bce_with_logits_equals_plain_formula_where_safe() {
    let c = bce_case(-8.0, 8.0, 630);
    for (weight, pos) in OPTIONS {
        let plain = c.host(weight, pos, bce_plain);
        assert_close(&format!("BCE vs plain f64 weight {weight} pos {pos}"), &c.run(weight, pos, Reduction::None, DType::F64), &plain, 1e-12);
        assert_close(&format!("BCE vs plain f32 weight {weight} pos {pos}"), &c.run(weight, pos, Reduction::None, DType::F32), &plain, 1e-6);
    }
}

// ----------------------------------------------------------------------------- l2_normalize

/// `x / max(‖x‖, ε)` along `dim` on the host, for a row-major `shape`.
fn l2_host(x: &[f64], shape: &[usize], dim: usize, eps: f64) -> Vec<f64> {
    let inner: usize = shape[dim + 1..].iter().product();
    let len = shape[dim];
    let outer = x.len() / (inner * len);
    let mut y = vec![0.0; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| (o * len + k) * inner + i;
            let norm = (0..len).map(|k| x[at(k)] * x[at(k)]).sum::<f64>().sqrt().max(eps);
            for k in 0..len {
                y[at(k)] = x[at(k)] / norm;
            }
        }
    }
    y
}

/// Every axis of a rank-3 tensor against the f64 host: within 1e-14 in f64 and 1e-6 in f32.
#[test]
fn l2_normalize_matches_f64_reference() {
    let shape = [2, 5, 7];
    let x = rnd(70, 640, -3.0, 3.0);
    for dim in 0..3 {
        let want = l2_host(&x, &shape, dim, L2_EPS);
        assert_close(&format!("l2_normalize f64 dim {dim}"), &l2_normalize(&t64(x.clone(), &shape), dim, L2_EPS).to_vec_f64(), &want, 1e-14);
        assert_close(&format!("l2_normalize f32 dim {dim}"), &l2_normalize(&t32(x.clone(), &shape), dim, L2_EPS).to_vec_f64(), &want, 1e-6);
    }
    // Rows below ε are divided by ε.
    let small = vec![1e-4, -2e-4, 0.0, 3e-4, 1.0, 2.0, -2.0, 1.0];
    assert_close("l2_normalize below ε", &l2_normalize(&t64(small.clone(), &[2, 4]), 1, 1e-2).to_vec_f64(), &l2_host(&small, &[2, 4], 1, 1e-2), 1e-14);
}

/// f64 central differences along every axis, and with rows below ε.
#[test]
fn l2_normalize_matches_finite_differences() {
    let shape = [2, 4, 5];
    let x = rnd(40, 650, -2.0, 2.0);
    for dim in 0..3 {
        fd_input(&format!("l2_normalize dim {dim}"), &x, &shape, &|t| l2_normalize(&t, dim, L2_EPS), 1e-6, 1e-7);
    }
    // Row 0 has norm ~5e-4 < ε = 1e-2 (divided by ε, linear), row 1 is well above it.
    let x = vec![1e-4, -2e-4, 3e-4, 2e-4, 0.5, -1.0, 2.0, 0.25];
    fd_input("l2_normalize below ε", &x, &[2, 4], &|t| l2_normalize(&t, 1, 1e-2), 1e-7, 1e-7);
}

/// A zero row: the output is exactly zero, finite, and its gradient is the upstream gradient / ε
/// (the clamped norm passes none); the other rows are untouched by it. f32 and f64, ε = 1e-12
/// and 1e-6.
#[test]
fn l2_normalize_zero_row_is_finite() {
    let (r, d) = (3, 6);
    let mut x = rnd(r * d, 660, -1.0, 1.0);
    x[d..2 * d].iter_mut().for_each(|v| *v = 0.0);
    let up = rnd(r * d, 661, -1.0, 1.0);
    for dtype in [DType::F32, DType::F64] {
        for eps in [L2_EPS, 1e-6] {
            let xt = Tensor::from_f64s(x.clone(), vec![r, d], dtype).require_grad();
            let y = l2_normalize(&xt, 1, eps);
            let yv = y.to_vec_f64();
            assert!(yv[d..2 * d].iter().all(|v| *v == 0.0), "{dtype:?} ε {eps:e}: zero row gives {:?}", &yv[d..2 * d]);
            assert!(yv.iter().all(|v| v.is_finite()));
            let g = (y * Tensor::from_f64s(up.clone(), vec![r, d], dtype)).sum().backward();
            let gv = xt.grad(&g).unwrap().to_vec_f64();
            assert!(gv.iter().all(|v| v.is_finite()), "{dtype:?} ε {eps:e}: gradient {gv:?}");
            let want: Vec<f64> = up[d..2 * d].iter().map(|u| u / eps).collect();
            let tol = if dtype == DType::F64 { 1e-14 } else { 1e-6 };
            for (a, b) in gv[d..2 * d].iter().zip(&want) {
                assert!((a - b).abs() <= tol * b.abs(), "{dtype:?} ε {eps:e}: zero-row gradient {a:e} vs upstream/ε {b:e}");
            }
            // The other rows: as if normalised alone.
            for row in [0, 2] {
                let alone = Tensor::from_f64s(x[row * d..(row + 1) * d].to_vec(), vec![1, d], dtype).require_grad();
                let ya = l2_normalize(&alone, 1, eps);
                let ga = alone.grad(&(ya.clone() * Tensor::from_f64s(up[row * d..(row + 1) * d].to_vec(), vec![1, d], dtype)).sum().backward()).unwrap().to_vec_f64();
                assert_eq!(&yv[row * d..(row + 1) * d], &ya.to_vec_f64()[..], "{dtype:?}: row {row} output");
                assert_eq!(&gv[row * d..(row + 1) * d], &ga[..], "{dtype:?}: row {row} gradient");
            }
        }
    }
}

/// Non-zero rows have unit norm within 1e-6 in f32 (and 1e-14 in f64), along every axis,
/// for rows of widths 1 to 512 and norms from 1e-3 to 1e3.
#[test]
fn l2_normalize_rows_have_unit_norm() {
    for (i, &d) in [1usize, 2, 3, 16, 100, 512].iter().enumerate() {
        let r = 8;
        let x: Vec<f64> = rnd(r * d, 670 + i as u64, -1.0, 1.0).iter().enumerate().map(|(j, v)| v * 10f64.powi((j / d) as i32 % 7 - 3)).collect();
        for (dtype, tol) in [(DType::F32, 1e-6), (DType::F64, 1e-14)] {
            for (shape, dim) in [(vec![r, d], 1), (vec![d, r], 0)] {
                let xs = if dim == 1 { x.clone() } else { (0..r * d).map(|k| x[(k % r) * d + k / r]).collect() };
                let y = l2_normalize(&Tensor::from_f64s(xs, shape.clone(), dtype), dim, L2_EPS).to_vec_f64();
                for row in 0..r {
                    let n2: f64 = (0..d).map(|k| if dim == 1 { y[row * d + k] } else { y[k * r + row] }).map(|v| v * v).sum();
                    assert!((n2.sqrt() - 1.0).abs() <= tol, "{dtype:?} d {d} dim {dim} row {row}: norm {:e}", n2.sqrt());
                }
            }
        }
    }
}
