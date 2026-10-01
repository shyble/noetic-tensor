//! Naive f64 references, finite differences and properties for LayerNorm, the activations
//! (erf bound), Dropout, RoPE, grouped-query attention, padding masks, SwiGLU and the plain MLP,
//! SGD and AdamW against scalar references, schedules, per-seed clipping, MSE, KL, the CE
//! options and 16-bit var save.

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

/// Central-difference gradient of `f(x).sum()·w` against autodiff at every element (f64).
fn fd_check(what: &str, x: &[f64], shape: &[usize], f: &dyn Fn(Tensor) -> Tensor, eps: f64, tol: f64) {
    let y0 = f(t64(x.to_vec(), shape));
    let w = rnd(y0.numel(), 77, -1.0, 1.0);
    let wt = t64(w.clone(), y0.shape());
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

// ------------------------------------------------------------------------------------ norms

#[test]
fn layer_norm_matches_naive_and_finite_differences() {
    let (s, n, d) = (2, 3, 5);
    let x = rnd(s * n * d, 1, -2.0, 2.0);
    let (w, b) = (rnd(s * d, 2, 0.5, 1.5), rnd(s * d, 3, -0.5, 0.5));
    let ln = LayerNorm::new(t64(w.clone(), &[s, 1, d]), Some(t64(b.clone(), &[s, 1, d])), 1e-5);
    let got = ln.forward(&t64(x.clone(), &[s, n, d])).to_vec_f64();
    let mut want = Vec::new();
    for k in 0..s {
        for r in 0..n {
            let row = &x[(k * n + r) * d..(k * n + r + 1) * d];
            let mean = row.iter().sum::<f64>() / d as f64;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
            want.extend((0..d).map(|i| (row[i] - mean) / (var + 1e-5).sqrt() * w[k * d + i] + b[k * d + i]));
        }
    }
    assert_close("layer norm", &got, &want, 1e-12);
    fd_check("layer norm input", &x, &[s, n, d], &|xt| ln.forward(&xt), 1e-6, 1e-7);
    // Built by the builder: weight ones, bias zeros, in that order.
    let map = RefCell::new(VarMap::new());
    let _ = layer_norm(d, "ln", &VarBuilder::init(&map, 2, 1)).unwrap();
    assert_eq!(map.borrow().names(), &["ln".to_string(), "ln_bias".to_string()]);
}

/// RMSNorm without ε is invariant to the input's scale; with the default ε nearly so.
#[test]
fn rms_norm_is_scale_invariant() {
    let (s, n, d) = (2, 4, 8);
    let x = rnd(s * n * d, 4, -1.0, 1.0);
    let g = t64(rnd(s * d, 5, 0.5, 1.5), &[s, 1, d]);
    for c in [1e-3, 0.5, 7.0, 1e4] {
        let xc: Vec<f64> = x.iter().map(|v| v * c).collect();
        let exact = RmsNorm::new(g.clone(), 0.0);
        assert_close(&format!("rms ε=0 c={c}"), &exact.forward(&t64(xc.clone(), &[s, n, d])).to_vec_f64(), &exact.forward(&t64(x.clone(), &[s, n, d])).to_vec_f64(), 1e-12);
        if c >= 0.5 {
            let circ = RmsNorm::new(g.clone(), 1e-6);
            assert_close(&format!("rms ε=1e-6 c={c}"), &circ.forward(&t64(xc, &[s, n, d])).to_vec_f64(), &circ.forward(&t64(x.clone(), &[s, n, d])).to_vec_f64(), 1e-5);
        }
    }
}

// ------------------------------------------------------------------------------ activations

/// erf by its Maclaurin series (|x| ≤ 4, where it converges to ~1e-9 in f64), else ±1
/// (|erf(x) − ±1| = erfc(4) ≈ 1.5e-8 there).
fn erf_ref(x: f64) -> f64 {
    if x.abs() > 4.0 {
        return x.signum();
    }
    let (mut term, mut sum) = (x, x);
    for n in 1..200 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

#[test]
fn erf_is_within_its_bound_and_gelus_match_naive() {
    let xs: Vec<f64> = (0..=1200).map(|i| -6.0 + i as f64 * 0.01).collect();
    let n = xs.len();
    let e64 = erf(&t64(xs.clone(), &[n])).to_vec_f64();
    let e32 = erf(&t32(xs.clone(), &[n])).to_vec_f64();
    let (mut w64, mut w32) = (0.0f64, 0.0f64);
    for (i, x) in xs.iter().enumerate() {
        let r = erf_ref((*x as f32) as f64);
        w64 = w64.max((erf_ref(*x) - e64[i]).abs());
        w32 = w32.max((r - e32[i]).abs());
    }
    eprintln!("erf max error: f64 {w64:e}, f32 {w32:e}");
    assert!(w64 <= 1.6e-7 + 1.6e-8, "f64 erf error {w64:e}");
    assert!(w32 <= 4e-7, "f32 erf error {w32:e}");
    // The gradient against 2/√π·e^(−x²), away from 0.
    let xt = t64(xs.clone(), &[n]).require_grad();
    let g = erf(&xt).sum().backward();
    let gv = xt.grad(&g).unwrap().to_vec_f64();
    let mut wg = 0.0f64;
    for (i, x) in xs.iter().enumerate() {
        if x.abs() > 1e-9 {
            wg = wg.max((gv[i] - 2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp()).abs());
        }
    }
    eprintln!("erf gradient max error: {wg:e}");
    assert!(wg <= 6e-6, "erf gradient error {wg:e}");
    // GELU, both forms, against their definitions (erf form against the exact erf).
    let gelu_exact = |x: f64| 0.5 * x * (1.0 + erf_ref(x / 2f64.sqrt()));
    let gelu_t = |x: f64| 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
    let ge = gelu_erf(&t64(xs.clone(), &[n])).to_vec_f64();
    let gt = gelu_tanh(&t64(xs.clone(), &[n])).to_vec_f64();
    for (i, x) in xs.iter().enumerate() {
        assert!((ge[i] - gelu_exact(*x)).abs() <= 0.8e-7 * x.abs() + 1e-12, "gelu_erf({x})");
        // tanh is 2·sigmoid(2z) − 1, in f64.
        assert!((gt[i] - gelu_t(*x)).abs() <= 1e-12 * (1.0 + x.abs()), "gelu_tanh({x}): {} vs {}", gt[i], gelu_t(*x));
    }
    let r = relu(&t64(vec![-1.0, 0.0, 2.0], &[3])).to_vec_f64();
    assert_eq!(r, vec![0.0, 0.0, 2.0]);
    // Finite differences in f64, both forms tight.
    let x = rnd(24, 6, -3.0, 3.0);
    fd_check("gelu_erf", &x, &[24], &|t| gelu_erf(&t), 1e-6, 1e-6);
    fd_check("gelu_tanh", &x, &[24], &|t| gelu_tanh(&t), 1e-6, 1e-6);
    fd_check("relu", &x, &[24], &|t| relu(&t), 1e-6, 1e-7);
    // Large inputs keep finite values and gradients.
    let big = t64(vec![-80.0, -30.0, 30.0, 80.0], &[4]).require_grad();
    let y = gelu_tanh(&big);
    let gb = big.grad(&y.clone().sum().backward()).unwrap().to_vec_f64();
    assert!(y.to_vec_f64().iter().chain(&gb).all(|v| v.is_finite()));
}

#[test]
fn mlps_match_naive() {
    let (s, n, d, m) = (2, 3, 4, 6);
    let map = RefCell::new(VarMap::new());
    let vb = VarBuilder::init(&map, s, 2).with_dtype(DType::F64);
    let sw = swiglu(d, m, &vb.pp("sw")).unwrap();
    let gm = gated_mlp(GatedMlpConfig { d, hidden: m }, &VarBuilder::from_varmap(&map.borrow()).pp("sw")).unwrap();
    let plain = mlp(d, m, Activation::GeluErf, &vb.pp("p")).unwrap();
    let vars = map.into_inner();
    let x = rnd(s * n * d, 7, -1.0, 1.0);
    let xt = t64(x.clone(), &[s, n, d]);
    assert_eq!(sw.forward(&xt).to_vec_f64(), gm.forward(&xt).to_vec_f64(), "SwiGLU is the gated MLP");
    let g = |name: &str| vars.var(name).to_vec_f64();
    let (wi, wo) = (g("p.mlp_in"), g("p.mlp_out"));
    let mut want = Vec::new();
    for k in 0..s {
        for r in 0..n {
            let row = &x[(k * n + r) * d..(k * n + r + 1) * d];
            let h: Vec<f64> = (0..m).map(|j| { let z: f64 = (0..d).map(|i| row[i] * wi[k * d * m + i * m + j]).sum(); 0.5 * z * (1.0 + erf_ref(z / 2f64.sqrt())) }).collect();
            want.extend((0..d).map(|o| (0..m).map(|j| h[j] * wo[k * m * d + j * d + o]).sum::<f64>()));
        }
    }
    assert_close("plain mlp (GELU)", &plain.forward(&xt).to_vec_f64(), &want, 1e-6);
}

// ---------------------------------------------------------------------------------- dropout

#[test]
fn dropout_streams_and_identities() {
    let x = t32(rnd(3 * 400, 8, 0.5, 1.5), &[3, 20, 20]);
    let off = Dropout::new(0.0, 1, "a");
    assert_bits("p = 0", &off.forward_t(&x, true).to_vec(), &x.to_vec());
    assert_eq!(off.calls(), 0, "p = 0 draws nothing");
    let d = Dropout::new(0.25, 1, "a");
    assert_bits("eval", &d.forward_t(&x, false).to_vec(), &x.to_vec());
    assert_eq!(d.calls(), 0);
    let y1 = d.forward_t(&x, true).to_vec();
    let y2 = d.forward_t(&x, true).to_vec();
    assert_eq!(d.calls(), 2);
    assert_ne!(y1, y2, "each call draws a fresh mask");
    assert_bits("call 0 reproduces", &d.forward_at(&x, true, 0).to_vec(), &y1);
    let xv = x.to_vec();
    let dropped = y1.iter().filter(|v| **v == 0.0).count() as f64 / y1.len() as f64;
    assert!((dropped - 0.25).abs() < 0.04, "dropped {dropped}");
    for (a, b) in y1.iter().zip(&xv) {
        assert!(*a == 0.0 || *a == b * (1.0f32 / 0.75), "kept elements are scaled by 1/(1 − p)");
    }
    // Seed slot i's mask does not depend on S; another label is another stream.
    let m3 = d.mask(&[3, 20, 20], 5, &x).to_vec();
    let m5 = d.mask(&[5, 20, 20], 5, &x).to_vec();
    assert_eq!(&m3[400..800], &m5[400..800]);
    assert_ne!(Dropout::new(0.25, 1, "b").mask(&[3, 20, 20], 5, &x).to_vec(), m3);
}

// ------------------------------------------------------------------------------------- RoPE

fn rope_ref(x: &[f64], t: usize, dh: usize, offset: usize, style: RopeStyle, base: f64) -> Vec<f64> {
    let h = dh / 2;
    let mut out = x.to_vec();
    for row in 0..x.len() / dh {
        let m = (row % t + offset) as f64;
        for i in 0..h {
            let a = m * base.powf(-2.0 * i as f64 / dh as f64);
            let (p, q) = match style { RopeStyle::RotateHalf => (i, i + h), RopeStyle::Interleaved => (2 * i, 2 * i + 1) };
            let (u, v) = (x[row * dh + p], x[row * dh + q]);
            out[row * dh + p] = u * a.cos() - v * a.sin();
            out[row * dh + q] = u * a.sin() + v * a.cos();
        }
    }
    out
}

#[test]
fn rope_matches_naive_preserves_norms_and_is_relative() {
    let (nh, t, dh) = (3, 7, 8);
    let x = rnd(nh * t * dh, 9, -1.0, 1.0);
    for style in [RopeStyle::RotateHalf, RopeStyle::Interleaved] {
        let r = Rope::new(dh, RopeConfig { base: 10_000.0, style }).unwrap();
        for off in [0, 5] {
            let y = r.apply(&t64(x.clone(), &[nh, t, dh]), off).to_vec_f64();
            assert_close(&format!("{style:?} offset {off}"), &y, &rope_ref(&x, t, dh, off, style, 10_000.0), 1e-12);
            // Norm of every head vector is preserved.
            for row in 0..nh * t {
                let (a, b): (f64, f64) = (x[row * dh..(row + 1) * dh].iter().map(|v| v * v).sum(), y[row * dh..(row + 1) * dh].iter().map(|v| v * v).sum());
                assert!((a - b).abs() < 1e-12, "norm");
            }
        }
        // q_m · k_n depends only on m − n: shift both by δ.
        let (q, k) = (rnd(dh, 10, -1.0, 1.0), rnd(dh, 11, -1.0, 1.0));
        let at = |v: &[f64], m: usize| -> Vec<f64> { r.apply(&t64(v.to_vec(), &[1, 1, dh]), m).to_vec_f64() };
        let dot = |a: Vec<f64>, b: Vec<f64>| -> f64 { a.iter().zip(&b).map(|(x, y)| x * y).sum() };
        for (m, n) in [(3usize, 1usize), (9, 2), (4, 4)] {
            let base = dot(at(&q, m), at(&k, n));
            for delta in [1usize, 17, 250] {
                assert!((dot(at(&q, m + delta), at(&k, n + delta)) - base).abs() < 1e-10, "{style:?} ({m}, {n}) + {delta}");
            }
        }
        fd_check(&format!("rope {style:?}"), &x, &[nh, t, dh], &|xt| r.apply(&xt, 2), 1e-6, 1e-7);
    }
    assert!(Rope::new(5, RopeConfig::default()).is_err());
}

// -------------------------------------------------------------------------------- attention

fn attn_vars(cfg: AttentionConfig, s: usize, dtype: DType, root: u64) -> (MultiHeadAttention, VarMap) {
    let map = RefCell::new(VarMap::new());
    let a = attention(cfg, &VarBuilder::init(&map, s, root).with_dtype(dtype)).unwrap();
    (a, map.into_inner())
}

/// Naive attention of one seed with GQA, RoPE and a key padding mask.
#[allow(clippy::too_many_arguments)]
fn attn_ref(x: &[f64], t: usize, d: usize, h: usize, kvh: usize, rope: Option<RopeConfig>, pad: Option<&[bool]>, w: [&[f64]; 4]) -> Vec<f64> {
    let n = x.len() / d;
    let dh = d / h;
    let mm = |a: &[f64], k: usize, b: &[f64], c: usize| -> Vec<f64> { (0..a.len() / k).flat_map(|i| (0..c).map(move |j| (0..k).map(|q| a[i * k + q] * b[q * c + j]).sum::<f64>())).collect() };
    let (q, k, v) = (mm(x, d, w[0], d), mm(x, d, w[1], kvh * dh), mm(x, d, w[2], kvh * dh));
    let rot = |z: &[f64], m: usize| -> Vec<f64> { match rope { Some(c) => rope_ref(z, 1, dh, m, c.style, c.base), None => z.to_vec() } };
    let mut out = vec![0.0; n * d];
    for seq in 0..n / t {
        for head in 0..h {
            let kh = head / (h / kvh);
            for i in 0..t {
                let row = seq * t + i;
                if pad.is_some_and(|p| p[row]) {
                    continue;
                }
                let qi = rot(&q[row * d + head * dh..row * d + (head + 1) * dh], i);
                let sc: Vec<f64> = (0..=i).map(|j| {
                    let r = seq * t + j;
                    if pad.is_some_and(|p| p[r]) { return -1e9; }
                    let kj = rot(&k[r * kvh * dh + kh * dh..r * kvh * dh + (kh + 1) * dh], j);
                    qi.iter().zip(&kj).map(|(a, b)| a * b).sum::<f64>() / (dh as f64).sqrt()
                }).collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = sc.iter().map(|s| (s - mx).exp()).sum();
                for (j, s) in sc.iter().enumerate() {
                    let r = seq * t + j;
                    for e in 0..dh {
                        out[row * d + head * dh + e] += (s - mx).exp() / z * v[r * kvh * dh + kh * dh + e];
                    }
                }
            }
        }
    }
    mm(&out, d, w[3], d)
}

#[test]
fn gqa_rope_and_padding_match_naive_and_finite_differences() {
    let (s, b, t, d, h) = (2, 2, 5, 8, 4);
    let x = rnd(s * b * t * d, 12, -1.0, 1.0);
    let pad: Vec<bool> = (0..s * b * t).map(|i| { let (r, p) = (i / t, i % t); (r == 1 && p < 2) || (r == 3 && p == 0) }).collect();
    let pad_t = BoolTensor::from_data(pad.clone(), [s, b, t]);
    for (kvh, rope, with_pad) in [(4, None, false), (2, None, false), (1, Some(RopeConfig::default()), false), (2, Some(RopeConfig { base: 100.0, style: RopeStyle::Interleaved }), true), (4, None, true)] {
        let cfg = AttentionConfig { d, heads: h, kv_heads: kvh, causal: true, rope };
        let (att, vars) = attn_vars(cfg, s, DType::F64, 3);
        let p = with_pad.then_some(&pad_t);
        let got = att.forward_with(&t64(x.clone(), &[s, b * t, d]), t, p).to_vec_f64();
        let per = |name: &str, k: usize| -> Vec<f64> { let v = vars.var(name).to_vec_f64(); let n = v.len() / s; v[k * n..(k + 1) * n].to_vec() };
        let want: Vec<f64> = (0..s).flat_map(|k| {
            let xs = &x[k * b * t * d..(k + 1) * b * t * d];
            let pk = with_pad.then(|| pad[k * b * t..(k + 1) * b * t].to_vec());
            attn_ref(xs, t, d, h, kvh, rope, pk.as_deref(), [&per("wq", k), &per("wk", k), &per("wv", k), &per("wo", k)])
        }).collect();
        let what = format!("kv {kvh} rope {rope:?} pad {with_pad}");
        assert_close(&what, &got, &want, 1e-12);
        let pp = p.cloned();
        fd_check(&format!("{what} input"), &x, &[s, b * t, d], &|xt| att.forward_with(&xt, t, pp.as_ref()), 1e-6, 1e-6);
    }
}

/// GQA at kv = heads is multi-head attention, operation for operation (bit-equal); GQA at kv < heads equals MHA whose key/value columns are repeated.
#[test]
fn gqa_equals_mha() {
    let (s, b, t, d, h) = (3, 2, 6, 16, 4);
    let x = t32(rnd(s * b * t * d, 13, -1.0, 1.0), &[s, b * t, d]);
    let (gqa, vars) = attn_vars(AttentionConfig { d, heads: h, kv_heads: h, causal: true, rope: None }, s, DType::F32, 4);
    let mha = causal_attention(d, h, &VarBuilder::from_varmap(&vars)).unwrap();
    assert_bits("kv = heads", &gqa.forward(&x, t).to_vec(), &mha.forward(&x, t).to_vec());
    let dh = d / h;
    for kvh in [1, 2] {
        let (g, gv) = attn_vars(AttentionConfig { d, heads: h, kv_heads: kvh, causal: true, rope: None }, s, DType::F32, 5);
        // Repeat each key/value head's columns for its query heads.
        let widen = |name: &str| -> Tensor {
            let w = gv.var(name).to_vec();
            let mut out = Vec::with_capacity(s * d * d);
            for k in 0..s {
                for r in 0..d {
                    for c in 0..d {
                        let (head, e) = (c / dh, c % dh);
                        out.push(w[k * d * kvh * dh + r * kvh * dh + (head / (h / kvh)) * dh + e] as f64);
                    }
                }
            }
            t32(out, &[s, d, d])
        };
        let lin = |t: Tensor| Linear::new(t, None);
        let m = MultiHeadAttention::new(lin(gv.var("wq").clone()), lin(widen("wk")), lin(widen("wv")), lin(gv.var("wo").clone()), h, true);
        assert_bits(&format!("kv {kvh} vs widened MHA"), &g.forward(&x, t).to_vec(), &m.forward(&x, t).to_vec());
    }
}

/// Padded rows are never NaN (values and gradients, even for an all-pad sequence), output
/// exactly 0, and what sits at padded positions does not reach the other rows.
#[test]
fn padding_never_nans_and_isolates() {
    let (s, b, t, d, h) = (2, 3, 4, 8, 2);
    let (att, _) = attn_vars(AttentionConfig { d, heads: h, kv_heads: 1, causal: true, rope: Some(RopeConfig::default()) }, s, DType::F32, 6);
    let pad: Vec<bool> = (0..s * b * t).map(|i| { let (r, p) = ((i / t) % b, i % t); r == 0 || (r == 1 && p < 2) }).collect();
    let pad_t = BoolTensor::from_data(pad.clone(), [s, b, t]);
    let x = rnd(s * b * t * d, 14, -1.0, 1.0);
    let xt = t32(x.clone(), &[s, b * t, d]).require_grad();
    let y = att.forward_with(&xt, t, Some(&pad_t));
    let g = xt.grad(&y.clone().sum().backward()).unwrap().to_vec();
    let yv = y.to_vec();
    assert!(yv.iter().chain(&g).all(|v| v.is_finite()), "NaN or ∞");
    for (i, p) in pad.iter().enumerate() {
        if *p {
            assert!(yv[i * d..(i + 1) * d].iter().all(|v| *v == 0.0), "padded row {i} is not 0");
        }
    }
    // Other values at the padded positions leave the unpadded rows unchanged, to the bit.
    let mut x2 = x.clone();
    for (i, p) in pad.iter().enumerate() {
        if *p {
            x2[i * d..(i + 1) * d].iter_mut().for_each(|v| *v = *v * -3.0 + 0.7);
        }
    }
    let y2 = att.forward_with(&t32(x2, &[s, b * t, d]), t, Some(&pad_t)).to_vec();
    assert_bits("unpadded rows", &yv, &y2);
}

/// GQA and RoPE under the fixed-capacity KV-cache equal the full forward to the bit.
#[test]
fn gqa_rope_cached_equals_full() {
    let (s, b, t, d, h) = (2, 2, 9, 16, 4);
    for (kvh, style) in [(2, RopeStyle::RotateHalf), (1, RopeStyle::Interleaved), (4, RopeStyle::RotateHalf)] {
        let (att, _) = attn_vars(AttentionConfig { d, heads: h, kv_heads: kvh, causal: true, rope: Some(RopeConfig { base: 10_000.0, style }) }, s, DType::F32, 7);
        let x = rnd(s * b * t * d, 15, -1.0, 1.0);
        let full = att.forward(&t32(x.clone(), &[s, b * t, d]), t).to_vec();
        let mut cache = KvCache::with_capacity(t);
        let pre = 3;
        let rows = |from: usize, to: usize| -> Vec<f64> { (0..s * b).flat_map(|r| x[(r * t + from) * d..(r * t + to) * d].to_vec()).collect() };
        let mut out = vec![0.0f32; full.len()];
        let z = att.forward_cached(&t32(rows(0, pre), &[s, b * pre, d]), pre, &mut cache).to_vec();
        for r in 0..s * b {
            out[r * t * d..(r * t + pre) * d].copy_from_slice(&z[r * pre * d..(r + 1) * pre * d]);
        }
        for i in pre..t {
            let z = att.forward_cached(&t32(rows(i, i + 1), &[s, b, d]), 1, &mut cache).to_vec();
            for r in 0..s * b {
                out[(r * t + i) * d..(r * t + i + 1) * d].copy_from_slice(&z[r * d..(r + 1) * d]);
            }
        }
        assert_bits(&format!("cached kv {kvh} {style:?}"), &out, &full);
        assert_eq!(cache.get().unwrap().0.shape()[0], s * b * kvh, "the cache holds the key/value heads");
    }
}

// ------------------------------------------------------------------------------- optimizers

fn one_var(v: Vec<f64>, shape: &[usize], dtype: DType) -> VarMap {
    let mut m = VarMap::new();
    m.insert("w", Tensor::from_f64s(v, shape.to_vec(), dtype)).unwrap();
    m
}

#[test]
fn sgd_matches_scalar_reference() {
    let n = 12;
    let p0 = rnd(n, 20, -1.0, 1.0);
    let grads: Vec<Vec<f64>> = (0..6).map(|i| rnd(n, 30 + i, -1.0, 1.0)).collect();
    for cfg in [
        SgdConfig { lr: 0.1, ..Default::default() },
        SgdConfig { lr: 0.05, momentum: 0.9, weight_decay: 0.01, ..Default::default() },
        SgdConfig { lr: 0.05, momentum: 0.8, dampening: 0.3, ..Default::default() },
        SgdConfig { lr: 0.05, momentum: 0.9, nesterov: true, weight_decay: 0.1, ..Default::default() },
    ] {
        let mut vars = one_var(p0.clone(), &[2, 2, 3], DType::F64);
        let mut opt = Sgd::new(cfg.clone(), ParamGroups::new(&vars), &vars);
        let (mut p, mut buf): (Vec<f64>, Option<Vec<f64>>) = (p0.clone(), None);
        for (step, g) in grads.iter().enumerate() {
            let scale = 1.0 - 0.1 * step as f64;
            opt.step(&mut vars, vec![Some(t64(g.clone(), &[2, 2, 3]))], scale);
            let g: Vec<f64> = g.iter().zip(&p).map(|(g, p)| g + cfg.weight_decay * p).collect();
            let g = if cfg.momentum > 0.0 {
                let b: Vec<f64> = match &buf { None => g.clone(), Some(b) => b.iter().zip(&g).map(|(b, g)| cfg.momentum * b + (1.0 - cfg.dampening) * g).collect() };
                buf = Some(b.clone());
                if cfg.nesterov { g.iter().zip(&b).map(|(g, b)| g + cfg.momentum * b).collect() } else { b }
            } else {
                g
            };
            p = p.iter().zip(&g).map(|(p, g)| p - cfg.lr * scale * g).collect();
            assert_close(&format!("{cfg:?} step {step}"), &vars.var("w").to_vec_f64(), &p, 1e-12);
        }
    }
}

#[test]
fn adamw_matches_scalar_reference_in_both_orders() {
    let n = 10;
    let p0 = rnd(n, 40, -1.0, 1.0);
    let grads: Vec<Vec<f64>> = (0..8).map(|i| rnd(n, 50 + i, -1.0, 1.0)).collect();
    for order in [DecayOrder::AfterStep, DecayOrder::Torch] {
        let cfg = AdamConfig { weight_decay: 0.0, ..AdamConfig::adamw(0.01, 0.1, order) };
        let mut vars = one_var(p0.clone(), &[1, 2, 5], DType::F64);
        let mut opt = Adam::new(cfg.clone(), ParamGroups::new(&vars), &vars);
        let (mut p, mut m, mut v) = (p0.clone(), vec![0.0; n], vec![0.0; n]);
        for (step, g) in grads.iter().enumerate() {
            let t = step as i32 + 1;
            let scale = 0.5 + 0.1 * step as f64;
            opt.step(&mut vars, vec![Some(t64(g.clone(), &[1, 2, 5]))], scale);
            let lr = cfg.lr * scale;
            let shrink = 1.0 - lr * cfg.decoupled_decay;
            for i in 0..n {
                m[i] = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * g[i];
                v[i] = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * g[i] * g[i];
                let upd = (m[i] / (1.0 - cfg.beta1.powi(t))) / ((v[i] / (1.0 - cfg.beta2.powi(t))).sqrt() + cfg.eps);
                p[i] = match order {
                    DecayOrder::AfterStep => (p[i] - lr * upd) * shrink,
                    DecayOrder::Torch => p[i] * shrink - lr * upd,
                };
            }
            assert_close(&format!("{order:?} step {step}"), &vars.var("w").to_vec_f64(), &p, 1e-12);
        }
    }
    // The two orders differ (by lr²·wd·upd per step), so the choice is real.
    let run = |order| {
        let mut vars = one_var(p0.clone(), &[1, 2, 5], DType::F64);
        let mut opt = Adam::new(AdamConfig::adamw(0.1, 0.5, order), ParamGroups::new(&vars), &vars);
        opt.step(&mut vars, vec![Some(t64(grads[0].clone(), &[1, 2, 5]))], 1.0);
        vars.var("w").to_vec_f64()
    };
    assert_ne!(run(DecayOrder::AfterStep), run(DecayOrder::Torch));
}

#[test]
fn schedules() {
    let w = LinearWarmup { warmup: 4 };
    assert_eq!((0..6).map(|s| w.scale(s)).collect::<Vec<_>>(), vec![0.25, 0.5, 0.75, 1.0, 1.0, 1.0]);
    let st = StepDecay { step_size: 3, gamma: 0.5 };
    assert_eq!((0..7).map(|s| st.scale(s)).collect::<Vec<_>>(), vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25]);
    assert_eq!(Constant.scale(123), 1.0);
    assert_eq!(LinearWarmup { warmup: 0 }.scale(0), 1.0);
}

#[test]
fn clipping_is_per_seed() {
    let g1 = rnd(3 * 4, 60, -1.0, 1.0);
    let g2 = rnd(3 * 6, 61, -1.0, 1.0);
    let mut grads = vec![Some(t32(g1.clone(), &[3, 4])), None, Some(t32(g2.clone(), &[3, 2, 3]))];
    let naive: Vec<f64> = (0..3).map(|k| (g1[k * 4..(k + 1) * 4].iter().chain(&g2[k * 6..(k + 1) * 6]).map(|x| { let x = (*x as f32) as f64; x * x }).sum::<f64>()).sqrt()).collect();
    let norms = grad_norms_per_seed(&grads);
    assert_close("norms", &norms, &naive, 1e-12);
    // A bound between the seeds' norms: the small seeds stay exactly, the large ones clip.
    let mut sorted = norms.clone();
    sorted.sort_by(f64::total_cmp);
    let bound = (sorted[0] + sorted[1]) / 2.0;
    let before: Vec<Vec<f32>> = grads.iter().flatten().map(|g| g.to_vec()).collect();
    let _ = clip_grad_norm_per_seed(&mut grads, bound);
    let after = grad_norms_per_seed(&grads);
    for k in 0..3 {
        if norms[k] <= bound {
            for (b, a) in before.iter().zip(grads.iter().flatten()) {
                let per = b.len() / 3;
                assert_bits("unclipped seed", &a.to_vec()[k * per..(k + 1) * per], &b[k * per..(k + 1) * per]);
            }
        } else {
            assert!((after[k] - bound).abs() < 1e-5 * bound, "seed {k}: {} vs {bound}", after[k]);
        }
    }
    // A seed's clipping does not depend on the other seeds' gradients.
    let mut other = vec![Some(t32(g1.iter().enumerate().map(|(i, x)| if i < 4 { *x } else { x * 100.0 }).collect(), &[3, 4])), None, Some(t32(g2.iter().enumerate().map(|(i, x)| if i < 6 { *x } else { -x * 50.0 }).collect(), &[3, 2, 3]))];
    let mut mine = vec![Some(t32(g1.clone(), &[3, 4])), None, Some(t32(g2.clone(), &[3, 2, 3]))];
    let _ = clip_grad_norm_per_seed(&mut other, 0.5);
    let _ = clip_grad_norm_per_seed(&mut mine, 0.5);
    assert_bits("seed 0 alone", &other[0].as_ref().unwrap().to_vec()[..4], &mine[0].as_ref().unwrap().to_vec()[..4]);
    let mut v = vec![Some(t32(vec![-3.0, 0.5, 2.0], &[1, 3]))];
    clip_grad_value(&mut v, 1.0);
    assert_eq!(v[0].as_ref().unwrap().to_vec(), vec![-1.0, 0.5, 1.0]);
}

// ----------------------------------------------------------------------------------- losses

#[test]
fn mse_kl_and_ce_options_match_naive() {
    let (s, b, t, v) = (2, 2, 3, 5);
    let n = b * t;
    let a = rnd(s * n * v, 70, -2.0, 2.0);
    let c = rnd(s * n * v, 71, -2.0, 2.0);
    let mse_want: Vec<f64> = (0..s).map(|k| (0..n * v).map(|i| (a[k * n * v + i] - c[k * n * v + i]).powi(2)).sum::<f64>() / (n * v) as f64).collect();
    assert_close("mse", &mse(t64(a.clone(), &[s, b, t, v]), &t64(c.clone(), &[s, b, t, v])).to_vec_f64(), &mse_want, 1e-12);
    let ls = |r: &[f64]| -> Vec<f64> { let mx = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max); let z: f64 = r.iter().map(|x| (x - mx).exp()).sum(); r.iter().map(|x| x - mx - z.ln()).collect() };
    let kl_want: Vec<f64> = (0..s).map(|k| (0..n).map(|i| { let o = (k * n + i) * v; let (lq, lp) = (ls(&a[o..o + v]), ls(&c[o..o + v])); (0..v).map(|j| lp[j].exp() * (lp[j] - lq[j])).sum::<f64>() }).sum::<f64>() / n as f64).collect();
    let kl = kl_div(t64(a.clone(), &[s, b, t, v]), t64(c.clone(), &[s, b, t, v])).to_vec_f64();
    assert_close("kl", &kl, &kl_want, 1e-12);
    assert!(kl.iter().all(|x| *x >= 0.0));
    assert!(kl_div(t64(a.clone(), &[s, b, t, v]), t64(a.clone(), &[s, b, t, v])).to_vec_f64().iter().all(|x| x.abs() < 1e-15));
    fd_check("kl pred", &a, &[s, b, t, v], &|x| kl_div(x, t64(c.clone(), &[s, b, t, v])), 1e-6, 1e-7);
    fd_check("mse pred", &a, &[s, b, t, v], &|x| mse(x, &t64(c.clone(), &[s, b, t, v])), 1e-6, 1e-7);

    // CE options.
    let logits = t32(a.clone(), &[s, b, t, v]);
    let mut tg = rnd_ints(s * n, 72, v as i64);
    tg[1] = -100;
    tg[7] = -100;
    let mask = rnd(s * n, 73, 0.0, 1.0).into_iter().map(|u| if u < 0.8 { 1.0 } else { 0.0 }).collect::<Vec<f64>>();
    let targets = IntTensor::from_data(tg.clone(), [s, b, t]);
    let opts = CeOptions { ignore_index: Some(-100), smoothing: 0.0 };
    // ignore_index = masked CE with those positions masked out (and any class as target), to the bit.
    let mut m2 = mask.clone();
    let mut tg2 = tg.clone();
    for i in 0..s * n {
        if tg[i] == -100 { m2[i] = 0.0; tg2[i] = 0; }
    }
    assert_bits("ignore_index", &cross_entropy_with(logits.clone(), &targets, Some(&t32(mask.clone(), &[s, b, t])), opts).to_vec(), &masked_cross_entropy(logits.clone(), &IntTensor::from_data(tg2.clone(), [s, b, t]), &t32(m2.clone(), &[s, b, t])).to_vec());
    // Default options are the plain losses, to the bit.
    let tg_ok = IntTensor::from_data(tg2.clone(), [s, b, t]);
    assert_bits("default, masked", &cross_entropy_with(logits.clone(), &tg_ok, Some(&t32(mask.clone(), &[s, b, t])), CeOptions::default()).to_vec(), &masked_cross_entropy(logits.clone(), &tg_ok, &t32(mask.clone(), &[s, b, t])).to_vec());
    assert_bits("default, unmasked", &cross_entropy_with(logits.clone(), &tg_ok, None, CeOptions::default()).to_vec(), &cross_entropy(logits.clone(), &tg_ok).to_vec());
    // Label smoothing against the naive formula (f64).
    let eps = 0.1;
    let got = cross_entropy_with(t64(a.clone(), &[s, b, t, v]), &targets, Some(&t64(mask.clone(), &[s, b, t])), CeOptions { ignore_index: Some(-100), smoothing: eps }).to_vec_f64();
    let want: Vec<f64> = (0..s).map(|k| {
        let (mut sum, mut cnt) = (0.0, 0.0);
        for i in 0..n {
            let j = k * n + i;
            if tg[j] == -100 || mask[j] == 0.0 { continue; }
            let lp = ls(&a[j * v..(j + 1) * v]);
            sum += (1.0 - eps) * -lp[tg[j] as usize] + eps * -(lp.iter().sum::<f64>() / v as f64);
            cnt += 1.0;
        }
        sum / f64::max(cnt, 1.0)
    }).collect();
    assert_close("label smoothing", &got, &want, 1e-12);
}

// ------------------------------------------------------------------------------ 16-bit save

#[test]
fn half_vars_save_and_load_exactly() {
    for dt in [DType::F16, DType::BF16] {
        let map = RefCell::new(VarMap::new());
        let w = VarBuilder::init(&map, 2, 3).with_dtype(dt).get(&[3, 4], "w", Init::fan_in(3)).unwrap();
        assert_eq!(w.dtype(), dt);
        let f32w = VarBuilder::init(&RefCell::new(VarMap::new()), 2, 3).get(&[3, 4], "w", Init::fan_in(3)).unwrap();
        assert_eq!(w.to_vec(), f32w.cast(dt).to_vec(), "drawn in f32, stored rounded");
        let mut vars = map.into_inner();
        // Bit patterns a conversion would not keep: a signalling NaN, −0, a subnormal, ∞.
        let special: Vec<u16> = vec![0x7c01, 0x8000, 0x0001, 0x7c00, 0xfc00, 0x3c00];
        let t = if dt == DType::F16 { Tensor::raw_t(special.iter().map(|b| crate::tensor::half::F16(*b)).collect::<Vec<_>>(), vec![1, 2, 3]) } else { Tensor::raw_t(special.iter().map(|b| crate::tensor::half::BF16(*b)).collect::<Vec<_>>(), vec![1, 2, 3]) };
        vars.insert("special", t).unwrap();
        let json = vars.to_json().unwrap();
        let back = VarMap::from_json(&json).unwrap();
        assert_eq!(back.to_json().unwrap(), json, "{dt:?}: save → load → save");
        assert_eq!(back.var("special").dtype(), dt);
        let f = back.to_file().unwrap();
        let sp = f.vars.iter().find(|r| r.name == "special").unwrap();
        let bits: Vec<u16> = (0..6).map(|i| u16::from_str_radix(&sp.data[i * 4..i * 4 + 4], 16).unwrap().swap_bytes()).collect();
        assert_eq!(bits, special, "{dt:?}: stored bits");
    }
}
