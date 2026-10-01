//! f64 finite differences against autodiff: a stack without the sigmoid (norm, attention,
//! readout, masked cross-entropy) and the whole decoder (its SiLU gate included), both tightly.

use super::*;
use crate::nn::*;
use crate::tensor::DType;
use std::cell::RefCell;

fn t64(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F64)
}

/// Central differences of `loss` at four entries of every var, against `grads`.
fn check(what: &str, vars: &VarMap, grads: &[Option<Tensor>], loss: &dyn Fn(&VarMap) -> f64, eps: f64, tol: f64) -> usize {
    let mut checked = 0;
    for (i, name) in vars.names().iter().enumerate() {
        let Some(g) = &grads[i] else { continue };
        let gv = g.to_vec_f64();
        let t = vars.var(name);
        let n = t.numel();
        for j in [0usize, n / 3, n / 2, n - 1] {
            let base = t.to_vec_f64();
            let at = |delta: f64| -> f64 {
                let mut v = base.clone();
                v[j] += delta;
                let mut m = vars.clone();
                m.set(name, t64(v, t.shape())).unwrap();
                loss(&m)
            };
            let fd = (at(eps) - at(-eps)) / (2.0 * eps);
            assert!((fd - gv[j]).abs() <= tol * (1.0 + fd.abs()), "{what}: {name}[{j}]: fd {fd:e} vs autodiff {:e}", gv[j]);
            checked += 1;
        }
    }
    checked
}

#[test]
fn norm_attention_readout_match_finite_differences() {
    let (s, b, t, d, h, v) = (2, 2, 4, 6, 2, 5);
    let map = RefCell::new(VarMap::new());
    let vb = VarBuilder::init(&map, s, 9).with_dtype(DType::F64);
    let _ = rms_norm(d, "norm", &vb).unwrap();
    let _ = causal_attention(d, h, &vb.pp("att")).unwrap();
    let _ = linear_b(d, v, "readout", "readout_b", &vb).unwrap();
    let mut vars = map.into_inner();
    for (i, name) in vars.names().to_vec().iter().enumerate() {
        let sh = vars.var(name).shape().to_vec();
        let lo = if name == "norm" { 0.5 } else { -0.8 };
        vars.set(name, t64(rnd(sh.iter().product(), 100 + i as u64, lo, lo + 1.0), &sh)).unwrap();
    }
    let x = t64(rnd(s * b * t * d, 7, -1.0, 1.0), &[s, b * t, d]);
    let targets = IntTensor::from_data(rnd_ints(s * b * t, 8, v as i64), [s, b, t]);
    let mask = t64(rnd(s * b * t, 9, 0.0, 1.0).into_iter().map(|u| if u < 0.6 { 1.0 } else { 0.0 }).collect(), &[s, b, t]);
    let build = |m: &VarMap| -> Tensor {
        let vb = VarBuilder::from_varmap(m);
        let norm = rms_norm(d, "norm", &vb).unwrap();
        let att = causal_attention(d, h, &vb.pp("att")).unwrap();
        let ro = linear_b(d, v, "readout", "readout_b", &vb).unwrap();
        let y = x.clone() + att.forward(&norm.forward(&x), t);
        masked_cross_entropy(ro.forward(&y).reshape([s, b, t, v]), &targets, &mask).sum()
    };
    let lifted = vars.lifted();
    let grads = lifted.grads(&build(&lifted).backward());
    let n = check("norm/attention/readout", &vars, &grads, &|m| build(m).to_vec_f64()[0], 1e-6, 1e-7);
    assert_eq!(n, 4 * vars.len());
}

#[test]
fn decoder_matches_finite_differences() {
    let cfg = DecoderConfig::new(8, 8, 2, 4, 2, 16);
    let s = 2;
    let map = RefCell::new(VarMap::new());
    let _ = Decoder::new(cfg.clone(), &VarBuilder::init(&map, s, 7).with_dtype(DType::F64)).unwrap();
    let mut vars = map.into_inner();
    for (i, name) in vars.names().to_vec().iter().enumerate() {
        let sh = vars.var(name).shape().to_vec();
        let (base, scale) = if name.contains("norm") { (1.0, 0.0) } else { (0.0, 0.5) };
        vars.set(name, t64(rnd(sh.iter().product(), 200 + i as u64, -1.0, 1.0).into_iter().map(|u| base + scale * u).collect(), &sh)).unwrap();
    }
    let tokens = IntTensor::from_data(vec![1, 2, 3, 4, 4, 3, 2, 1, 0, 5, 6, 7, 7, 6, 5, 0, 2, 2, 2, 2, 1, 3, 5, 7], [2, 3, 4]);
    let next = IntTensor::from_data(vec![5, 6, 7, 1, 0, 2], [2, 3]);
    let loss = |m: &VarMap| -> Tensor { let dec = Decoder::load(cfg.clone(), m).unwrap(); next_token_loss(dec.forward(&tokens), &tokens, &next).sum() };
    let lifted = vars.lifted();
    let grads = lifted.grads(&loss(&lifted).backward());
    let n = check("decoder", &vars, &grads, &|m| loss(m).to_vec_f64()[0], 1e-6, 1e-7);
    assert!(n >= 60, "{n}");
}
