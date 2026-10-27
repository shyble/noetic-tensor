//! The joint-embedding additions on the CPU reference backend: the key padding mask of non-causal (set)
//! attention with `masked_mean_pool`, `TargetEma`, the VICReg loss, masked group InfoNCE and the
//! effective-rank monitor. References are f64 host computations and central finite differences.

use super::*;
use crate::nn::*;
use crate::tensor::{BoolTensor, DType};
use std::cell::RefCell;

fn t64(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F64)
}

fn t32(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F32)
}

/// Central differences of `f(x)·w` (w fixed random) against autodiff, at every element (f64).
fn fd_input(what: &str, x: &[f64], shape: &[usize], f: &dyn Fn(Tensor) -> Tensor, eps: f64, tol: f64) {
    let y0 = f(t64(x.to_vec(), shape));
    let wt = t64(rnd(y0.numel(), 78, -1.0, 1.0), y0.shape());
    let xt = t64(x.to_vec(), shape).require_grad();
    let g = (f(xt.clone()) * wt.clone()).sum().backward();
    let gv = xt.grad(&g).expect("a gradient").to_vec_f64();
    let loss = |v: Vec<f64>| -> f64 { (f(t64(v, shape)) * wt.clone()).sum().to_vec_f64()[0] };
    for j in 0..x.len() {
        let (mut p, mut m) = (x.to_vec(), x.to_vec());
        p[j] += eps;
        m[j] -= eps;
        let fd = (loss(p) - loss(m)) / (2.0 * eps);
        assert!((fd - gv[j]).abs() <= tol * (1.0 + fd.abs()), "{what}[{j}]: fd {fd:e} vs autodiff {:e}", gv[j]);
    }
}

// ------------------------------------------------------------------------- set attention

/// Non-causal attention vars of width `d`, `h` heads, S seeds.
fn set_attn(d: usize, h: usize, s: usize, dtype: DType, root: u64) -> (MultiHeadAttention, VarMap) {
    let map = RefCell::new(VarMap::new());
    let a = attention(AttentionConfig::non_causal(d, h), &VarBuilder::init(&map, s, root).with_dtype(dtype)).unwrap();
    (a, map.into_inner())
}

/// A batch of sets: B sets per seed, set r holding `sizes[r]` examples (the same sizes in every
/// seed) at the positions `slots(r)` of a common length t; pads hold garbage.
struct Sets {
    s: usize,
    t: usize,
    d: usize,
    sizes: Vec<usize>,
    /// Per (seed, set): the positions of its examples in order.
    pos: Vec<Vec<usize>>,
    /// The examples, per (seed, set), `sizes[r] · d` values.
    ex: Vec<Vec<f64>>,
}

impl Sets {
    fn new(s: usize, t: usize, d: usize, sizes: &[usize], interleave: bool, seed: u64) -> Sets {
        let mut pos = Vec::new();
        let mut ex = Vec::new();
        for k in 0..s {
            for (r, &n) in sizes.iter().enumerate() {
                assert!(n <= t);
                // Pads at the end, or spread through the sequence.
                let p: Vec<usize> = if interleave { (0..n).map(|i| (i * t) / n + ((r + k) % ((t / n).max(1)))).collect() } else { (0..n).collect() };
                assert!(p.windows(2).all(|w| w[0] < w[1]) && p.iter().all(|&i| i < t));
                pos.push(p);
                ex.push(rnd(n * d, seed + (k * sizes.len() + r) as u64, -1.0, 1.0));
            }
        }
        Sets { s, t, d, sizes: sizes.to_vec(), pos, ex }
    }

    fn b(&self) -> usize {
        self.sizes.len()
    }

    /// The padded input `[S, B·t, d]` (pads hold `fill(i)`) and its mask `[S, B, t]`.
    fn padded(&self, fill: f64) -> (Vec<f64>, Vec<bool>) {
        let (b, t, d) = (self.b(), self.t, self.d);
        let mut x = rnd(self.s * b * t * d, 991, -1.0, 1.0).into_iter().map(|v| v * fill).collect::<Vec<_>>();
        let mut pad = vec![true; self.s * b * t];
        for k in 0..self.s {
            for r in 0..b {
                let i = k * b + r;
                for (j, &p) in self.pos[i].iter().enumerate() {
                    let row = i * t + p;
                    pad[row] = false;
                    x[row * d..(row + 1) * d].copy_from_slice(&self.ex[i][j * d..(j + 1) * d]);
                }
            }
        }
        (x, pad)
    }

    /// Set r alone, unpadded: `[S, n_r, d]`.
    fn alone(&self, r: usize) -> Vec<f64> {
        (0..self.s).flat_map(|k| self.ex[k * self.b() + r].clone()).collect()
    }
}

/// Largest |a − b| / (1 + |b|).
fn rel32(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs() / (1.0 + (*y as f64).abs())).fold(0.0, f64::max)
}

/// Padded sets equal their unpadded selves at the real positions, and pool to the same vector,
/// whether the pads sit at the end or between the examples, within fp32 1e-6 (not to the bit: the
/// reductions over t and over n positions group their sums differently).
#[test]
fn set_attention_padded_equals_unpadded() {
    let (s, t, d, h) = (2, 10, 16, 4);
    let (att, _) = set_attn(d, h, s, DType::F32, 31);
    for interleave in [false, true] {
        let sets = Sets::new(s, t, d, &[2, 5, 10, 3, 7], interleave, 40);
        let b = sets.b();
        let (x, pad) = sets.padded(3.0);
        let pad_t = BoolTensor::from_data(pad.clone(), [s, b, t]);
        let y = att.forward_with(&t32(x, &[s, b * t, d]), t, Some(&pad_t));
        let pooled = masked_mean_pool(&y, t, Some(&pad_t)).to_vec();
        let y = y.to_vec();
        for r in 0..b {
            let n = sets.sizes[r];
            let ya = att.forward(&t32(sets.alone(r), &[s, n, d]), n);
            let pa = masked_mean_pool(&ya, n, None).to_vec();
            let ya = ya.to_vec();
            for k in 0..s {
                let i = k * b + r;
                let ours: Vec<f32> = sets.pos[i].iter().flat_map(|&p| y[(i * t + p) * d..(i * t + p + 1) * d].to_vec()).collect();
                let e = rel32(&ours, &ya[k * n * d..(k + 1) * n * d]);
                assert!(e <= 1e-6, "interleave {interleave} set {r} seed {k}: outputs differ by {e:e}");
                let e = rel32(&pooled[i * d..(i + 1) * d], &pa[k * d..(k + 1) * d]);
                assert!(e <= 1e-6, "interleave {interleave} set {r} seed {k}: pooled differs by {e:e}");
            }
        }
    }
}

/// Every padded position gets exactly zero gradient (input rows), through the attention and the
/// pool; the vars' gradients do not depend on what sits at the pads.
#[test]
fn set_attention_masked_positions_get_zero_gradient() {
    let (s, t, d, h) = (2, 10, 8, 2);
    let (att, vars) = set_attn(d, h, s, DType::F32, 32);
    let sets = Sets::new(s, t, d, &[2, 6, 10, 4], true, 50);
    let b = sets.b();
    let pad_t = |p: &[bool]| BoolTensor::from_data(p.to_vec(), [s, b, t]);
    let run = |fill: f64| -> (Vec<f32>, Vec<Vec<f32>>, Vec<bool>) {
        let (x, pad) = sets.padded(fill);
        let lifted = vars.lifted();
        let a = attention(AttentionConfig::non_causal(d, h), &VarBuilder::from_varmap(&lifted)).unwrap();
        let xt = t32(x, &[s, b * t, d]).require_grad();
        let pooled = masked_mean_pool(&a.forward_with(&xt, t, Some(&pad_t(&pad))), t, Some(&pad_t(&pad)));
        let w = t32(rnd(s * b * d, 51, -1.0, 1.0), &[s, b, d]);
        let g = (pooled * w).sum().backward();
        let gx = xt.grad(&g).unwrap().to_vec();
        let gv = lifted.grads(&g).into_iter().map(|g| g.unwrap().to_vec()).collect();
        (gx, gv, pad)
    };
    let (gx, gv, pad) = run(1.0);
    let mut zeros = 0;
    for (row, p) in pad.iter().enumerate() {
        let gr = &gx[row * d..(row + 1) * d];
        if *p {
            assert!(gr.iter().all(|v| v.to_bits() == 0), "pad row {row} has gradient {gr:?}");
            zeros += 1;
        } else {
            assert!(gr.iter().any(|v| *v != 0.0), "real row {row} has no gradient");
        }
    }
    assert_eq!(zeros, s * (10 - 2 + 10 - 6 + 0 + 10 - 4));
    let (gx2, gv2, _) = run(-7.0);
    assert_bits("input gradients, other pad values", &gx, &gx2);
    for (i, (a, b)) in gv.iter().zip(&gv2).enumerate() {
        assert_bits(&format!("var {i} gradient, other pad values"), a, b);
    }
    let _ = att;
}

/// Permuting the examples of every set (and where the pads sit) leaves the pooled output
/// unchanged within fp32 1e-6.
#[test]
fn set_attention_pool_is_permutation_invariant() {
    let (s, t, d, h) = (3, 10, 16, 4);
    let (att, _) = set_attn(d, h, s, DType::F32, 33);
    let sets = Sets::new(s, t, d, &[2, 4, 7, 10], false, 60);
    let b = sets.b();
    let pooled_of = |sets: &Sets| -> Vec<f32> {
        let (x, pad) = sets.padded(1.0);
        let pad_t = BoolTensor::from_data(pad, [s, b, t]);
        masked_mean_pool(&att.forward_with(&t32(x, &[s, b * t, d]), t, Some(&pad_t)), t, Some(&pad_t)).to_vec()
    };
    let base = pooled_of(&sets);
    for trial in 0..4u64 {
        let mut p = Sets::new(s, t, d, &sets.sizes, trial % 2 == 1, 60);
        let mut r = rng(70 + trial);
        for i in 0..s * b {
            let n = sets.sizes[i % b];
            // A random permutation of the examples (Fisher–Yates).
            let mut order: Vec<usize> = (0..n).collect();
            for j in (1..n).rev() {
                use rand::Rng;
                order.swap(j, r.gen_range(0..=j));
            }
            p.ex[i] = order.iter().flat_map(|&j| sets.ex[i][j * d..(j + 1) * d].to_vec()).collect();
        }
        let got = pooled_of(&p);
        let worst = rel32(&got, &base);
        assert!(worst <= 1e-6, "trial {trial}: pooled output moved by {worst:e}");
    }
}

/// Two runs give the same bits (outputs, pooled, input and var gradients).
#[test]
fn set_attention_is_bit_identical_across_runs() {
    let (s, t, d, h) = (2, 10, 16, 4);
    let sets = Sets::new(s, t, d, &[2, 10, 5], true, 80);
    let b = sets.b();
    let run = || -> Vec<Vec<f32>> {
        let (_, vars) = set_attn(d, h, s, DType::F32, 34);
        let lifted = vars.lifted();
        let a = attention(AttentionConfig::non_causal(d, h), &VarBuilder::from_varmap(&lifted)).unwrap();
        let (x, pad) = sets.padded(1.0);
        let pad_t = BoolTensor::from_data(pad, [s, b, t]);
        let xt = t32(x, &[s, b * t, d]).require_grad();
        let y = a.forward_with(&xt, t, Some(&pad_t));
        let pooled = masked_mean_pool(&y, t, Some(&pad_t));
        let g = (pooled.clone() * pooled.clone()).sum().backward();
        let mut out = vec![y.to_vec(), pooled.to_vec(), xt.grad(&g).unwrap().to_vec()];
        out.extend(lifted.grads(&g).into_iter().map(|g| g.unwrap().to_vec()));
        out
    };
    let (a, b2) = (run(), run());
    for (i, (x, y)) in a.iter().zip(&b2).enumerate() {
        assert_bits(&format!("run-to-run {i}"), x, y);
    }
}

/// Finite differences (f64) of padded non-causal attention with the pool, at every input element
/// and at four entries of every var.
#[test]
fn set_attention_matches_finite_differences() {
    let (s, t, d, h) = (2, 5, 6, 2);
    let sets = Sets::new(s, t, d, &[2, 5, 3], true, 90);
    let b = sets.b();
    let (x, pad) = sets.padded(1.0);
    let pad_t = BoolTensor::from_data(pad, [s, b, t]);
    let (att, vars) = set_attn(d, h, s, DType::F64, 35);
    fd_input("set attention + pool: input", &x, &[s, b * t, d], &|xt| masked_mean_pool(&att.forward_with(&xt, t, Some(&pad_t)), t, Some(&pad_t)), 1e-6, 1e-7);
    let xt = t64(x, &[s, b * t, d]);
    let w = t64(rnd(s * b * d, 91, -1.0, 1.0), &[s, b, d]);
    let loss = |m: &VarMap| -> Tensor {
        let a = attention(AttentionConfig::non_causal(d, h), &VarBuilder::from_varmap(m)).unwrap();
        (masked_mean_pool(&a.forward_with(&xt, t, Some(&pad_t)), t, Some(&pad_t)) * w.clone()).sum()
    };
    let lifted = vars.lifted();
    let grads = lifted.grads(&loss(&lifted).backward());
    let mut checked = 0;
    for (i, name) in vars.names().iter().enumerate() {
        let gv = grads[i].as_ref().unwrap().to_vec_f64();
        let base = vars.var(name).to_vec_f64();
        let sh = vars.var(name).shape().to_vec();
        for j in [0, base.len() / 3, base.len() / 2, base.len() - 1] {
            let at = |delta: f64| -> f64 {
                let mut v = base.clone();
                v[j] += delta;
                let mut m = vars.clone();
                m.set(name, t64(v, &sh)).unwrap();
                loss(&m).to_vec_f64()[0]
            };
            let fd = (at(1e-6) - at(-1e-6)) / 2e-6;
            assert!((fd - gv[j]).abs() <= 1e-7 * (1.0 + fd.abs()), "{name}[{j}]: fd {fd:e} vs autodiff {:e}", gv[j]);
            checked += 1;
        }
    }
    assert_eq!(checked, 16);
}
