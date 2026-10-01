//! Naive f64 references per component: plain loops over the definitions, compared with the nn
//! components built in F64. Every component computes in f64 (the sigmoid too), so all are
//! checked at 1e-12.

use super::*;
use crate::nn::*;
use crate::tensor::DType;
use std::cell::RefCell;

fn t64(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F64)
}

/// Row-major [m, k] · [k, n].
fn mm(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|q| a[i * k + q] * b[q * n + j]).sum();
        }
    }
    c
}

fn rms(x: &[f64], d: usize, scale: &[f64], eps: f64) -> Vec<f64> {
    x.chunks(d).flat_map(|r| {
        let ms = r.iter().map(|v| v * v).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        r.iter().zip(scale).map(move |(v, g)| v * inv * g).collect::<Vec<_>>()
    }).collect()
}

/// Causal attention of one seed: x [n = b·t, d] with weights [d, d] each.
fn attention(x: &[f64], t: usize, d: usize, h: usize, w: [&[f64]; 4]) -> Vec<f64> {
    let n = x.len() / d;
    let (q, k, v) = (mm(x, n, d, w[0], d), mm(x, n, d, w[1], d), mm(x, n, d, w[2], d));
    let dh = d / h;
    let mut out = vec![0.0; n * d];
    for seq in 0..n / t {
        for head in 0..h {
            for i in 0..t {
                let row = seq * t + i;
                let sc: Vec<f64> = (0..=i).map(|j| (0..dh).map(|e| q[row * d + head * dh + e] * k[(seq * t + j) * d + head * dh + e]).sum::<f64>() / (dh as f64).sqrt()).collect();
                let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = sc.iter().map(|s| (s - mx).exp()).sum();
                for (j, s) in sc.iter().enumerate() {
                    let wgt = (s - mx).exp() / z;
                    for e in 0..dh {
                        out[row * d + head * dh + e] += wgt * v[(seq * t + j) * d + head * dh + e];
                    }
                }
            }
        }
    }
    mm(&out, n, d, w[3], d)
}

fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

#[allow(clippy::too_many_arguments)]
fn gated(x: &[f64], d: usize, m: usize, w_in: &[f64], w_gate: &[f64], w_out: &[f64]) -> Vec<f64> {
    let n = x.len() / d;
    let (a, g) = (mm(x, n, d, w_in, m), mm(x, n, d, w_gate, m));
    let hdn: Vec<f64> = (0..n * m).map(|i| a[i] * silu(g[i])).collect();
    mm(&hdn, n, m, w_out, d)
}

fn seed(v: &[f64], s: usize, per: usize) -> &[f64] {
    &v[s * per..(s + 1) * per]
}

#[test]
fn linear_matches_naive() {
    let (s, n, i, o) = (2, 5, 4, 3);
    let x = rnd(s * n * i, 1, -1.0, 1.0);
    let w = rnd(s * i * o, 2, -1.0, 1.0);
    let b = rnd(s * o, 3, -1.0, 1.0);
    let lin = Linear::new(t64(w.clone(), &[s, i, o]), Some(t64(b.clone(), &[s, 1, o])));
    let y = lin.forward(&t64(x.clone(), &[s, n, i])).to_vec_f64();
    let mut want = Vec::new();
    for k in 0..s {
        let yk = mm(seed(&x, k, n * i), n, i, seed(&w, k, i * o), o);
        want.extend(yk.chunks(o).flat_map(|r| r.iter().zip(seed(&b, k, o)).map(|(a, c)| a + c).collect::<Vec<_>>()));
    }
    assert_close("linear", &y, &want, 1e-12);
    // Rank 4 input [S, B, T, in]: the weight broadcasts over B.
    let y4 = lin.forward(&t64(x.clone(), &[s, 1, n, i])).to_vec_f64();
    assert_close("linear rank 4", &y4, &want, 1e-12);
}

#[test]
fn rms_norm_matches_naive() {
    let (s, n, d) = (3, 4, 6);
    let x = rnd(s * n * d, 4, -2.0, 2.0);
    let g = rnd(s * d, 5, 0.5, 1.5);
    let norm = RmsNorm::new(t64(g.clone(), &[s, 1, d]), 1e-6);
    let y = norm.forward(&t64(x.clone(), &[s, n, d])).to_vec_f64();
    let want: Vec<f64> = (0..s).flat_map(|k| rms(seed(&x, k, n * d), d, seed(&g, k, d), 1e-6)).collect();
    assert_close("rms norm", &y, &want, 1e-12);
}

#[test]
fn attention_matches_naive() {
    let (s, b, t, d, h) = (2, 3, 5, 8, 2);
    let x = rnd(s * b * t * d, 6, -1.0, 1.0);
    let ws: Vec<Vec<f64>> = (0..4).map(|i| rnd(s * d * d, 10 + i, -0.5, 0.5)).collect();
    let lin = |w: &Vec<f64>| Linear::new(t64(w.clone(), &[s, d, d]), None);
    let att = MultiHeadAttention::new(lin(&ws[0]), lin(&ws[1]), lin(&ws[2]), lin(&ws[3]), h, true);
    let y = att.forward(&t64(x.clone(), &[s, b * t, d]), t).to_vec_f64();
    let want: Vec<f64> = (0..s).flat_map(|k| attention(seed(&x, k, b * t * d), t, d, h, [seed(&ws[0], k, d * d), seed(&ws[1], k, d * d), seed(&ws[2], k, d * d), seed(&ws[3], k, d * d)])).collect();
    assert_close("attention", &y, &want, 1e-12);
}

#[test]
fn gated_mlp_matches_naive() {
    let (s, n, d, m) = (2, 6, 4, 7);
    let x = rnd(s * n * d, 20, -1.0, 1.0);
    let (wi, wg, wo) = (rnd(s * d * m, 21, -1.0, 1.0), rnd(s * d * m, 22, -1.0, 1.0), rnd(s * m * d, 23, -1.0, 1.0));
    let lin = |w: &Vec<f64>, a, b| Linear::new(t64(w.clone(), &[s, a, b]), None);
    let mlp = GatedMlp::new(lin(&wi, d, m), lin(&wg, d, m), lin(&wo, m, d));
    let y = mlp.forward(&t64(x.clone(), &[s, n, d])).to_vec_f64();
    let want: Vec<f64> = (0..s).flat_map(|k| gated(seed(&x, k, n * d), d, m, seed(&wi, k, d * m), seed(&wg, k, d * m), seed(&wo, k, m * d))).collect();
    assert_close("gated mlp", &y, &want, 1e-12);
}

#[test]
fn embedding_modes_read_the_rows() {
    let (s, v, d, n) = (2, 7, 3, 9);
    let table = rnd(s * v * d, 30, -1.0, 1.0);
    let ids = rnd_ints(s * n, 31, v as i64);
    let want: Vec<f64> = (0..s).flat_map(|k| (0..n).flat_map(|i| { let r = ids[k * n + i] as usize; table[(k * v + r) * d..(k * v + r + 1) * d].to_vec() }).collect::<Vec<_>>()).collect();
    for mode in [EmbeddingMode::OneHot, EmbeddingMode::Gather] {
        let e = Embedding::new(t64(table.clone(), &[s, v, d]), mode);
        let y = e.forward(&IntTensor::from_data(ids.clone(), [s, n])).to_vec_f64();
        assert_eq!(y, want, "{mode:?}");
    }
}

fn lse(r: &[f64]) -> f64 {
    let mx = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    mx + r.iter().map(|x| (x - mx).exp()).sum::<f64>().ln()
}

#[test]
fn losses_match_naive() {
    let (s, b, t, v) = (3, 2, 4, 5);
    let logits = rnd(s * b * t * v, 40, -3.0, 3.0);
    let targets = rnd_ints(s * b * t, 41, v as i64);
    let mut mask: Vec<f64> = rnd(s * b * t, 42, 0.0, 1.0).into_iter().map(|u| if u < 0.5 { 1.0 } else { 0.0 }).collect();
    mask[2 * b * t..].iter_mut().for_each(|m| *m = 0.0); // seed 2 scores nothing
    let lt = t64(logits.clone(), &[s, b, t, v]);
    let tg = IntTensor::from_data(targets.clone(), [s, b, t]);
    let n = b * t;
    let nll = |k: usize, i: usize| { let r = &logits[(k * n + i) * v..(k * n + i + 1) * v]; lse(r) - r[targets[k * n + i] as usize] };
    let ce: Vec<f64> = (0..s).map(|k| (0..n).map(|i| nll(k, i)).sum::<f64>() / n as f64).collect();
    assert_close("cross entropy", &cross_entropy(lt.clone(), &tg).to_vec_f64(), &ce, 1e-12);
    let mce: Vec<f64> = (0..s).map(|k| (0..n).map(|i| nll(k, i) * mask[k * n + i]).sum::<f64>() / (0..n).map(|i| mask[k * n + i]).sum::<f64>().max(1.0)).collect();
    let got = masked_cross_entropy(lt, &tg, &t64(mask, &[s, b, t])).to_vec_f64();
    assert_close("masked cross entropy", &got, &mce, 1e-12);
    assert_eq!(got[2], 0.0);
}

/// The whole decoder in F64 against the naive composition of the components above.
#[test]
fn decoder_matches_naive() {
    let cfg = DecoderConfig::new(9, 8, 2, 5, 2, 12);
    let (s, b, t) = (2, 3, 5);
    let map = RefCell::new(VarMap::new());
    let _ = Decoder::new(cfg.clone(), &VarBuilder::init(&map, s, 3).with_dtype(DType::F64)).unwrap();
    let mut vars = map.into_inner();
    // Non-trivial norms.
    for (i, name) in vars.names().to_vec().iter().enumerate() {
        let sh = vars.var(name).shape().to_vec();
        let n: usize = sh.iter().product();
        let v = if name.contains("norm") { Some(rnd(n, 60 + i as u64, 0.5, 1.5)) } else { None };
        if let Some(v) = v {
            vars.set(name, t64(v, &sh)).unwrap();
        }
    }
    assert!(vars.tensors().iter().all(|t| t.dtype() == DType::F64));
    let tokens = rnd_ints(s * b * t, 61, cfg.vocab as i64);
    let dec = Decoder::load(cfg.clone(), &vars).unwrap();
    let got = dec.forward(&IntTensor::from_data(tokens.clone(), [s, b, t])).to_vec_f64();

    let (d, m, v, n) = (cfg.d, cfg.mlp_hidden, cfg.vocab, b * t);
    let g = |name: &str, k: usize| -> Vec<f64> { let x = vars.var(name); let per = x.numel() / s; seed(&x.to_vec_f64(), k, per).to_vec() };
    let mut want = Vec::new();
    for k in 0..s {
        let (embed, pos) = (g("embed", k), g("pos", k));
        let mut x: Vec<f64> = (0..n).flat_map(|i| { let tok = tokens[k * n + i] as usize; let p = i % t; (0..d).map(|e| embed[tok * d + e] + pos[p * d + e]).collect::<Vec<_>>() }).collect();
        for j in 0..cfg.blocks {
            let nm = |base: &str| cfg.naming.prefix(j).is_empty().then(|| base.to_string()).unwrap_or(format!("{}.{base}", cfg.naming.prefix(j)));
            let xn = rms(&x, d, &g(&nm("norm_att"), k), 1e-6);
            let a = attention(&xn, t, d, cfg.heads, [&g(&nm("wq"), k), &g(&nm("wk"), k), &g(&nm("wv"), k), &g(&nm("wo"), k)]);
            x.iter_mut().zip(&a).for_each(|(p, q)| *p += q);
            let xn = rms(&x, d, &g(&nm("norm_mlp"), k), 1e-6);
            let o = gated(&xn, d, m, &g(&nm("mlp_in"), k), &g(&nm("mlp_gate"), k), &g(&nm("mlp_out"), k));
            x.iter_mut().zip(&o).for_each(|(p, q)| *p += q);
        }
        let xn = rms(&x, d, &g("norm_out", k), 1e-6);
        want.extend(mm(&xn, n, d, &g("readout", k), v));
    }
    assert_close("decoder logits", &got, &want, 1e-12);
}
