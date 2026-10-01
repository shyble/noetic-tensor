//! The standard semantics that replace burn 0.21's where burn's differed: NaN-propagating max and argmax, a stable
//! descending sort (ties by source position on every lane length), `detach` that always
//! untracks, and a stable `logsumexp`.

use super::*;
use crate::tensor::{self as nt, DType};

#[test]
fn max_and_argmax_propagate_nan() {
    let nan = f32::NAN;
    // Lanes: NaN in the middle, NaN first, two NaNs, none.
    let v = vec![1.0, nan, 5.0, 2.0, nan, 3.0, 9.0, 0.0, 4.0, nan, nan, 1.0, 1.0, 7.0, 7.0, 2.0];
    let t = Tensor::from_data(v.clone(), [4, 4]);
    assert_eq!(t.clone().argmax(1).to_vec(), vec![1, 0, 1, 1], "the first NaN is the maximum; ties take the first");
    let m = t.clone().max_dim(1).to_vec();
    assert!(m[0].is_nan() && m[1].is_nan() && m[2].is_nan() && m[3] == 7.0);
    assert!(t.clone().max().to_vec()[0].is_nan(), "max of all elements is NaN if any is");
    assert!(Tensor::from_data(vec![1.0, 2.0, nan], [3]).max().to_vec()[0].is_nan(), "a trailing NaN propagates too");
    assert_eq!(Tensor::from_data(vec![1.0, 3.0, 2.0], [3]).max().to_vec(), vec![3.0]);
    // Along dim 0 as well (the strided lane).
    let c = Tensor::from_data(vec![1.0, 2.0, nan, 0.0, 3.0, nan], [3, 2]);
    assert_eq!(c.argmax(0).to_vec(), vec![1, 2]);
    // The gradient of max_dim goes to the argmax position, NaN lanes included.
    let x = Tensor::from_data(vec![1.0, nan, 5.0, 2.0], [1, 4]).require_grad();
    let g = x.clone().max_dim(1).sum().backward();
    assert_eq!(x.grad(&g).unwrap().to_vec(), vec![0.0, 1.0, 0.0, 0.0]);
}

/// A stable reference, written independently of the kernel: positions sorted descending with
/// every NaN (either sign) first and equal to the others, then `total_cmp` (+0 before −0), ties
/// by ascending position.
fn reference_sort(lane: &[f32]) -> Vec<i64> {
    let key = |x: f32| if x.is_nan() { (1u8, 0.0f32) } else { (0u8, x) };
    let mut idx: Vec<usize> = (0..lane.len()).collect();
    idx.sort_by(|&a, &b| {
        let (ka, kb) = (key(lane[a]), key(lane[b]));
        kb.0.cmp(&ka.0).then(kb.1.total_cmp(&ka.1)).then(a.cmp(&b))
    });
    idx.into_iter().map(|j| j as i64).collect()
}

#[test]
fn sort_is_stable_on_long_lanes() {
    // 300 values from 7 distinct levels (many ties, −1e9 among them), NaN
    // and signed zeros: lanes far past std's small-slice insertion sort.
    // NaN of both signs (x86's default NaN from 0/0 has the sign bit set) and a payload NaN.
    let levels = [-1e9f32, -0.5, 0.0, -0.0, 0.25, 1.0, f32::NAN, f32::from_bits(0xFFC0_0000), f32::NEG_INFINITY, f32::from_bits(0x7F80_0001), f32::INFINITY];
    for n in [33usize, 300, 5000] {
        let v: Vec<f32> = (0..2 * n).map(|i| levels[(i * 7 + i / 3) % levels.len()]).collect();
        let (vals, idx) = Tensor::from_data(v.clone(), [2, n]).sort_descending_with_indices(1);
        let (vals, idx) = (vals.to_vec(), idx.to_vec());
        for r in 0..2 {
            let lane = &v[r * n..(r + 1) * n];
            let want = reference_sort(lane);
            assert_eq!(&idx[r * n..(r + 1) * n], want.as_slice(), "lane of {n}: ties keep ascending positions");
            for (k, &j) in want.iter().enumerate() {
                assert_eq!(vals[r * n + k].to_bits(), lane[j as usize].to_bits());
            }
            assert!(vals[r * n].is_nan(), "NaN sorts first");
            // Every NaN of the lane, either sign, precedes every number.
            let nans = lane.iter().filter(|x| x.is_nan()).count();
            assert!(lane.iter().any(|x| x.is_sign_negative() && x.is_nan()), "the lane has a sign-bit NaN");
            assert!(vals[r * n..r * n + nans].iter().all(|x| x.is_nan()) && !vals[r * n + nans..(r + 1) * n].iter().any(|x| x.is_nan()));
        }
    }
    // The rule in small: [−NaN, 1, +0, −0, +NaN, −inf] sorts to −NaN, +NaN, 1, +0, −0, −inf.
    let v = vec![f32::from_bits(0xFFC0_0000), 1.0, 0.0, -0.0, f32::NAN, f32::NEG_INFINITY];
    let (s, i) = Tensor::from_data(v, [1, 6]).sort_descending_with_indices(1);
    assert_eq!(i.to_vec(), vec![0, 4, 1, 2, 3, 5]);
    let s = s.to_vec();
    assert!(s[2] == 1.0 && s[3].to_bits() == 0 && s[4].to_bits() == 0x8000_0000, "+0 sorts before −0");
    // topk takes the first k of the stable order, along a middle dimension too.
    let v: Vec<f32> = (0..3 * 64 * 2).map(|i| levels[(i / 2 * 5) % 6]).collect();
    let (_, i) = Tensor::from_data(v.clone(), [3, 64, 2]).topk_with_indices(10, 1);
    let i = i.to_vec();
    for o in 0..3 {
        for c in 0..2 {
            let lane: Vec<f32> = (0..64).map(|k| v[(o * 64 + k) * 2 + c]).collect();
            let want = &reference_sort(&lane)[..10];
            let got: Vec<i64> = (0..10).map(|k| i[(o * 10 + k) * 2 + c]).collect();
            assert_eq!(got.as_slice(), want);
        }
    }
}

#[test]
fn detach_always_untracks() {
    let x = Tensor::from_data(vec![1.0, 2.0], [2]).require_grad();
    let d = x.clone().detach();
    assert!(!d.is_tracked() && !d.is_require_grad(), "a detached leaf is not a leaf");
    let y = (x.clone() * d.clone()).sum();
    let g = y.backward();
    assert_eq!(x.grad(&g).unwrap().to_vec(), vec![1.0, 2.0], "no gradient through the detached copy");
    assert!(d.grad(&g).is_none());
    // detach then require_grad makes a fresh leaf (how parameters are lifted each step).
    let p = x.detach().require_grad();
    assert!(p.is_require_grad());
}

#[test]
fn logsumexp_is_stable_and_exact_at_infinities() {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let v = vec![1000.0, 1000.0, -inf, 0.0, -inf, -inf, -inf, -inf, 1.0, inf, 0.0, 2.0, 0.5, nan, 1.0, 3.0];
    let out = nt::logsumexp(Tensor::from_data(v, [4, 4]), 1).to_vec();
    assert_eq!(out[0], 1000.0 + std::f32::consts::LN_2, "no overflow");
    assert_eq!(out[1], -inf, "a lane of −∞ gives −∞");
    assert_eq!(out[2], inf, "+∞ gives +∞");
    assert!(out[3].is_nan());
    // Against f64 on ordinary values, and its gradient is the softmax.
    let x = rnd(3 * 50, 41, -30.0, 30.0);
    let t = Tensor::from_data(x.clone(), [3, 50]).require_grad();
    let y = nt::logsumexp(t.clone(), 1);
    for (r, got) in y.to_vec().iter().enumerate() {
        let lane = &x[r * 50..(r + 1) * 50];
        let m = lane.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b as f64));
        let want = m + lane.iter().map(|z| (*z as f64 - m).exp()).sum::<f64>().ln();
        assert!((*got as f64 - want).abs() <= 4.0 * f32::EPSILON as f64 * want.abs().max(1.0), "lane {r}: {got} vs {want}");
    }
    let g = y.sum().backward();
    let sm = nt::softmax(Tensor::from_data(x, [3, 50]), 1).to_vec();
    for (a, b) in t.grad(&g).unwrap().to_vec().iter().zip(&sm) {
        assert!((a - b).abs() <= 1e-6, "gradient {a} vs softmax {b}");
    }
    // f64 stays f64.
    let d = nt::logsumexp(Tensor::from_f64s(vec![0.1, 0.2, 0.3], [1, 3], DType::F64), 1);
    let want = (0.1f64.exp() + 0.2f64.exp() + 0.3f64.exp()).ln();
    assert!((d.to_vec_f64()[0] - want).abs() < 1e-15);
}

// ------------------------------------------------------------------ the standard formulations

fn f64t(x: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_f64s(x.to_vec(), shape.to_vec(), DType::F64)
}

/// The gradient of `sum(op(x) · w)` in f64 against central finite differences (step 1e-6), at
/// 1e-6·(1 + |g|): every entry for small inputs.
fn fd64(name: &str, shape: &[usize], x: &[f64], op: impl Fn(Tensor) -> Tensor) {
    let w: Vec<f64> = rnd(4096, 77, -1.0, 1.0).iter().map(|v| *v as f64).collect();
    let loss = |x: &[f64]| -> f64 { op(f64t(x, shape)).to_vec_f64().iter().zip(&w).map(|(a, b)| a * b).sum() };
    let xt = f64t(x, shape).require_grad();
    let y = op(xt.clone());
    let wt = f64t(&w[..y.numel()], y.shape());
    let g = xt.grad(&(y * wt).sum().backward()).expect("gradient").to_vec_f64();
    let h = 1e-6;
    for j in 0..x.len() {
        let (mut p, mut m) = (x.to_vec(), x.to_vec());
        p[j] += h;
        m[j] -= h;
        let fd = (loss(&p) - loss(&m)) / (2.0 * h);
        assert!((fd - g[j]).abs() <= 1e-6 * (1.0 + g[j].abs()), "{name}[{j}]: finite difference {fd} vs autodiff {}", g[j]);
    }
}

/// The f32 gradient of `sum(op(x) · w)` against the same computation in f64 on the same inputs,
/// at `tol`·(1 + |g64|): the standard formulations' accuracy in f32.
fn against_f64(name: &str, shape: &[usize], x: &[f32], tol: f64, op: impl Fn(Tensor) -> Tensor) {
    let w = rnd(4096, 78, -1.0, 1.0);
    let x32 = Tensor::from_data(x.to_vec(), shape.to_vec()).require_grad();
    let y32 = op(x32.clone());
    let n = y32.numel();
    let g32 = x32.grad(&(y32.clone() * Tensor::from_data(w[..n].to_vec(), y32.shape().to_vec())).sum().backward()).unwrap().to_vec();
    let x64 = f64t(&x.iter().map(|v| *v as f64).collect::<Vec<_>>(), shape).require_grad();
    let y64 = op(x64.clone());
    let w64: Vec<f64> = w[..n].iter().map(|v| *v as f64).collect();
    let g64 = x64.grad(&(y64.clone() * f64t(&w64, y64.shape())).sum().backward()).unwrap().to_vec_f64();
    for (j, (a, b)) in g32.iter().zip(&g64).enumerate() {
        assert!((*a as f64 - b).abs() <= tol * (1.0 + b.abs()), "{name}[{j}]: f32 gradient {a} vs f64 {b}");
    }
    for (j, (a, b)) in y32.to_vec().iter().zip(y64.to_vec_f64()).enumerate() {
        assert!((*a as f64 - b).abs() <= tol * (1.0 + b.abs()), "{name}[{j}]: f32 value {a} vs f64 {b}");
    }
}

/// Every operation whose numerics differ from burn's, in f64 against finite differences and in f32
/// against f64.
#[test]
fn changed_operations_match_finite_differences_and_f64() {
    let s = [2usize, 3, 5];
    let x64: Vec<f64> = rnd(30, 51, -2.0, 2.0).iter().map(|v| *v as f64).collect();
    let pos64: Vec<f64> = rnd(30, 52, 0.5, 2.5).iter().map(|v| *v as f64).collect();
    let c = f64t(&rnd(30, 53, 0.5, 1.5).iter().map(|v| *v as f64).collect::<Vec<_>>(), &s);
    let cb = f64t(&rnd(10, 54, 0.5, 1.5).iter().map(|v| *v as f64).collect::<Vec<_>>(), &[2, 1, 5]);
    fd64("div lhs", &s, &x64, |t| t / c.clone());
    fd64("div rhs", &s, &pos64, |t| c.clone() / t);
    fd64("div rhs broadcast", &[2, 1, 5], &pos64[..10], |t| c.clone() / t);
    fd64("div lhs broadcast", &[2, 1, 5], &x64[..10], |t| t / c.clone());
    fd64("div_scalar", &s, &x64, |t| t.div_scalar(7.0));
    fd64("log", &s, &pos64, |t| t.log());
    fd64("sqrt", &s, &pos64, |t| t.sqrt());
    fd64("recip", &s, &pos64, |t| t.recip());
    fd64("mean", &s, &x64, |t| t.mean());
    fd64("sigmoid", &s, &x64, nt::sigmoid);
    fd64("silu", &s, &x64, nt::silu);
    fd64("rms chain", &s, &x64, |t| (t.clone().powf_scalar(2.0).mean_dim(2) + 1e-6).sqrt().recip() * t);
    for d in 0..3 {
        fd64("mean_dim", &s, &x64, |t| t.mean_dim(d));
        fd64("softmax", &s, &x64, |t| nt::softmax(t, d));
        fd64("log_softmax", &s, &x64, |t| nt::log_softmax(t, d));
    }
    let e: Vec<f64> = rnd(2 * 4 * 6, 55, -1.0, 1.0).iter().map(|v| *v as f64).collect();
    let f = f64t(&rnd(2 * 6 * 5, 56, -1.0, 1.0).iter().map(|v| *v as f64).collect::<Vec<_>>(), &[2, 1, 6, 5]);
    fd64("matmul [b,1,k]×[1,k,n] lhs", &[2, 4, 1, 6], &e, |t| t.matmul(f.clone()));
    let l = f64t(&e, &[2, 4, 1, 6]);
    fd64("matmul [b,1,k]×[1,k,n] rhs", &[2, 1, 6, 5], &rnd(60, 56, -1.0, 1.0).iter().map(|v| *v as f64).collect::<Vec<_>>(), |t| l.clone().matmul(t));
    let _ = cb;

    let x = rnd(30, 51, -2.0, 2.0);
    let pos = rnd(30, 52, 0.5, 2.5);
    let c32 = |t: &Tensor| if t.dtype() == DType::F64 { f64t(&rnd(30, 53, 0.5, 1.5).iter().map(|v| *v as f64).collect::<Vec<_>>(), &s) } else { Tensor::from_data(rnd(30, 53, 0.5, 1.5), s) };
    against_f64("div", &s, &x, 1e-6, |t| { let c = c32(&t); t / c });
    against_f64("div rhs", &s, &pos, 1e-6, |t| { let c = c32(&t); c / t });
    against_f64("div_scalar", &s, &x, 1e-6, |t| t.div_scalar(7.0));
    against_f64("log", &s, &pos, 1e-6, |t| t.log());
    against_f64("sqrt", &s, &pos, 1e-6, |t| t.sqrt());
    against_f64("recip", &s, &pos, 1e-6, |t| t.recip());
    against_f64("mean", &s, &x, 1e-6, |t| t.mean());
    against_f64("sigmoid", &s, &x, 1e-6, nt::sigmoid);
    against_f64("silu", &s, &x, 1e-6, nt::silu);
    for d in 0..3 {
        against_f64("mean_dim", &s, &x, 1e-6, |t| t.mean_dim(d));
        against_f64("softmax", &s, &x, 1e-6, |t| nt::softmax(t, d));
        against_f64("log_softmax", &s, &x, 1e-6, |t| nt::log_softmax(t, d));
    }
}

/// The sigmoid: within 2 ulp of the correctly rounded value on a dense grid over the whole f32
/// range where it is not saturated (burn's exp(−ln(1 + e^−x)) lost relative precision for large
/// negative x, the ulp error growing with |x|), exactly 0 and 1 in the saturated tails, NaN for
/// NaN; in f64, computed in f64 (burn's cast to f32).
#[test]
fn sigmoid_is_accurate_over_the_range() {
    let xs: Vec<f32> = (-10400..=10400).map(|i| i as f32 * 0.01).chain([-103.97, -87.5, 88.0, 1e-30, -1e-30, 0.0, -0.0]).collect();
    let n = xs.len();
    let y = nt::sigmoid(Tensor::from_data(xs.clone(), [n])).to_vec();
    let ulps = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
    let mut worst = 0;
    for (x, y) in xs.iter().zip(&y) {
        let exact = 1.0 / (1.0 + (-(*x as f64)).exp());
        worst = worst.max(ulps(*y, exact as f32));
    }
    assert!(worst <= 2, "sigmoid: {worst} ulp from the correctly rounded value");
    let tails = nt::sigmoid(Tensor::from_data(vec![200.0, -200.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN], [5])).to_vec();
    assert_eq!(&tails[..4], &[1.0, 0.0, 1.0, 0.0]);
    assert!(tails[4].is_nan());
    let x64 = vec![-700.0, -30.0, -1e-9, 0.3, 30.0];
    let y64 = nt::sigmoid(f64t(&x64, &[5])).to_vec_f64();
    for (x, y) in x64.iter().zip(&y64) {
        let want = if *x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) };
        assert!((y - want).abs() <= 4.0 * f64::EPSILON * want, "f64 sigmoid({x}) = {y}, want {want}");
    }
    assert!(y64[0] > 0.0 && y64[0] < 1e-300, "f64 keeps the tail f32 would flush to 0");
}

// ------------------------------------------------------------------ ulp guards

/// Distance in f32 units in the last place between `a` and the f64 value `r` rounded to f32.
fn ulps_to(a: f32, r: f64) -> u64 {
    let b = r as f32;
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN.wrapping_sub(i) as i64 } else { i as i64 }
    };
    (key(a) - key(b)).unsigned_abs()
}

/// The f32 gradient of `sum(op(inputs) · w)` with respect to input `wrt`, elementwise, and its
/// worst ulp distance from `want(inputs, w)` computed in f64.
fn grad_ulps(n: usize, inputs: &[Vec<f32>], wrt: usize, op: impl Fn(&[Tensor]) -> Tensor, want: impl Fn(&[f64], f64) -> f64) -> u64 {
    let ts: Vec<Tensor> = inputs.iter().map(|v| Tensor::from_data(v.clone(), [n]).require_grad()).collect();
    let y = op(&ts);
    let w = rnd(y.numel(), 90, -1.0, 1.0);
    let g = ts[wrt].grad(&(y * Tensor::from_data(w.clone(), [n])).sum().backward()).unwrap().to_vec();
    (0..n)
        .map(|j| {
            let xs: Vec<f64> = inputs.iter().map(|v| v[j] as f64).collect();
            ulps_to(g[j], want(&xs, w[j] as f64))
        })
        .max()
        .unwrap()
}

/// The standard backward forms at their measured worst case: a future edit that loses precision fails here, where
/// `changed_operations_match_finite_differences_and_f64` (1e-6 relative) would still pass.
#[test]
fn changed_backward_forms_stay_within_their_ulp_bounds() {
    let n = 100_000;
    let pos = rnd(n, 91, 0.05, 8.0);
    let pos2 = rnd(n, 92, 0.05, 8.0);
    let l = rnd(n, 93, -4.0, 4.0);
    let x = rnd(n, 94, -6.0, 6.0);
    let cases: Vec<(&str, u64, u64)> = vec![
        ("div lhs", 1, grad_ulps(n, &[l.clone(), pos.clone()], 0, |t| t[0].clone() / t[1].clone(), |v, w| w / v[1])),
        ("div rhs", 2, grad_ulps(n, &[l.clone(), pos.clone()], 1, |t| t[0].clone() / t[1].clone(), |v, w| -w * v[0] / (v[1] * v[1]))),
        ("log", 1, grad_ulps(n, &[pos.clone()], 0, |t| t[0].clone().log(), |v, w| w / v[0])),
        ("sqrt", 1, grad_ulps(n, &[pos.clone()], 0, |t| t[0].clone().sqrt(), |v, w| w / (2.0 * v[0].sqrt()))),
        ("recip", 3, grad_ulps(n, &[pos2.clone()], 0, |t| t[0].clone().recip(), |v, w| -w / (v[0] * v[0]))),
        ("div_scalar(7)", 1, grad_ulps(n, &[x.clone()], 0, |t| t[0].clone().div_scalar(7.0), |_, w| w / 7.0)),
        ("exp", 1, grad_ulps(n, &[x.clone()], 0, |t| t[0].clone().exp(), |v, w| w * v[0].exp())),
    ];
    for (name, bound, got) in &cases {
        eprintln!("ulp guard {name:<14}: {got} ulp (bound {bound})");
    }
    for (name, bound, got) in cases {
        assert!(got <= bound, "{name} backward: {got} ulp from the f64 value, bound {bound}");
    }
    // mean over lanes of 13 (the gradient is w_lane / 13 at every element).
    let m = rnd(13 * 7_000, 95, -1.0, 1.0);
    let t = Tensor::from_data(m, [7_000, 13]).require_grad();
    let w = rnd(7_000, 96, -1.0, 1.0);
    let g = t.grad(&(t.clone().mean_dim(1) * Tensor::from_data(w.clone(), [7_000, 1])).sum().backward()).unwrap().to_vec();
    let worst = (0..7_000 * 13).map(|j| ulps_to(g[j], w[j / 13] as f64 / 13.0)).max().unwrap();
    assert!(worst <= 1, "mean_dim backward: {worst} ulp");
}

/// softmax and log_softmax backward, rows of 17 and 1000: the worst |error| over the row's largest
/// gradient against f64, at twice the measured value.
#[test]
fn softmax_backward_errors_stay_within_their_bounds() {
    for (len, rows, sm_bound, lsm_bound) in [(17usize, 6_000usize, 7.9e-5, 1.3e-6), (1000, 100, 3.0e-6, 9.1e-7)] {
        let x = rnd(len * rows, 97, -6.0, 6.0);
        let w = rnd(len * rows, 98, -1.0, 1.0);
        for log in [false, true] {
            let t = Tensor::from_data(x.clone(), [rows, len]).require_grad();
            let y = if log { nt::log_softmax(t.clone(), 1) } else { nt::softmax(t.clone(), 1) };
            let g = t.grad(&(y * Tensor::from_data(w.clone(), [rows, len])).sum().backward()).unwrap().to_vec();
            let mut worst = 0f64;
            for r in 0..rows {
                let xs: Vec<f64> = x[r * len..(r + 1) * len].iter().map(|v| *v as f64).collect();
                let ws: Vec<f64> = w[r * len..(r + 1) * len].iter().map(|v| *v as f64).collect();
                let m = xs.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = xs.iter().map(|v| (v - m).exp()).collect();
                let s: f64 = e.iter().sum();
                let p: Vec<f64> = e.iter().map(|v| v / s).collect();
                let want: Vec<f64> = if log {
                    let sw: f64 = ws.iter().sum();
                    (0..len).map(|j| ws[j] - p[j] * sw).collect()
                } else {
                    let gy: f64 = ws.iter().zip(&p).map(|(a, b)| a * b).sum();
                    (0..len).map(|j| p[j] * (ws[j] - gy)).collect()
                };
                let rowmax = want.iter().fold(0f64, |a, b| a.max(b.abs()));
                for j in 0..len {
                    worst = worst.max((g[r * len + j] as f64 - want[j]).abs() / rowmax);
                }
            }
            let bound = if log { lsm_bound } else { sm_bound };
            eprintln!("ulp guard {} rows of {len}: max err/rowmax {worst:.2e} (bound {bound:.1e})", if log { "log_softmax" } else { "softmax" });
            assert!(worst <= bound, "{} backward, rows of {len}: max err/rowmax {worst:.2e} > {bound:.1e}", if log { "log_softmax" } else { "softmax" });
        }
    }
}
