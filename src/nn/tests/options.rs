//! The decoder's options, the nn trainer, MoE (E = 1 equals its expert, aux = 1 under uniform
//! routing, tie-breaking, capacity, finite differences, seed independence, determinism) and the
//! small MoE decoder.

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

fn moe_vars(cfg: MoeConfig, indices: &[usize], dtype: DType, root: u64) -> (Moe, VarMap) {
    let map = RefCell::new(VarMap::new());
    let m = moe(cfg, &VarBuilder::init_indexed(&map, indices, root).with_dtype(dtype)).unwrap();
    (m, map.into_inner())
}

fn mcfg(experts: usize, top_k: usize, normalize: bool, cap: Option<f64>) -> MoeConfig {
    MoeConfig { d: 8, hidden: 12, experts, top_k, normalize_topk: normalize, capacity_factor: cap, expert: ExpertKind::SwiGlu }
}

// ---------------------------------------------------------------------------------------- MoE

#[test]
fn moe_with_one_expert_is_its_expert() {
    for s in [1, 3] {
        for (normalize, kind) in [(false, ExpertKind::SwiGlu), (true, ExpertKind::SwiGlu), (true, ExpertKind::Plain(Activation::GeluTanh))] {
            let cfg = MoeConfig { expert: kind, ..mcfg(1, 1, normalize, None) };
            let (m, vars) = moe_vars(cfg, &(0..s).collect::<Vec<_>>(), DType::F32, 3);
            let x = t32(rnd(s * 10 * 8, 4, -2.0, 2.0), &[s, 10, 8]);
            let vb = VarBuilder::from_varmap(&vars).pp("e0");
            let expert = match kind {
                ExpertKind::SwiGlu => Expert::Gated(Box::new(swiglu(8, 12, &vb).unwrap())),
                ExpertKind::Plain(a) => Expert::Plain(Box::new(mlp(8, 12, a, &vb).unwrap())),
            };
            let want = expert.forward(&x).to_vec();
            for train in [false, true] {
                let o = m.forward_moe(&x, train);
                assert_bits(&format!("S={s} normalize={normalize} {kind:?} train={train}"), &o.y.to_vec(), &want);
                assert!(o.balance.to_vec().iter().all(|b| *b == 1.0), "one expert: balance 1");
            }
            // At capacity factor 0.5 the expert takes the first ⌈0.5·10⌉ = 5 tokens in training;
            // the others are dropped (output 0), and inference has no capacity.
            let (mc, _) = moe_vars(MoeConfig { capacity_factor: Some(0.5), ..cfg }, &(0..s).collect::<Vec<_>>(), DType::F32, 3);
            let o = mc.forward_moe(&x, true);
            assert_eq!((o.load[0].clone(), o.dropped[0]), (vec![5], 5));
            let y = o.y.to_vec();
            for q in 0..s {
                assert_bits("kept tokens", &y[q * 80..q * 80 + 40], &want[q * 80..q * 80 + 40]);
                assert!(y[q * 80 + 40..(q + 1) * 80].iter().all(|v| *v == 0.0), "dropped tokens");
            }
            assert_bits("inference ignores capacity", &mc.forward_moe(&x, false).y.to_vec(), &want);
        }
    }
}

/// With uniform router probabilities the Switch balance loss is 1 (for any assignment), and
/// ties go to the lower expert indices.
#[test]
fn balance_is_one_under_uniform_routing_and_ties_go_low() {
    for (e, k) in [(2, 1), (4, 1), (4, 2), (8, 3)] {
        let (m, mut vars) = moe_vars(mcfg(e, k, false, None), &[0, 1], DType::F64, 5);
        vars.set("router", t64(vec![0.0; 2 * 8 * e], &[2, 8, e])).unwrap();
        let m2 = moe(*m.config(), &VarBuilder::from_varmap(&vars)).unwrap();
        let o = m2.forward_moe(&t64(rnd(2 * 6 * 8, 6, -1.0, 1.0), &[2, 6, 8]), false);
        assert_close(&format!("E={e} k={k} balance"), &o.balance.to_vec_f64(), &[1.0, 1.0], 1e-12);
        let want: Vec<usize> = (0..e).map(|i| if i < k { 6 } else { 0 }).collect();
        assert_eq!(o.load, vec![want.clone(), want], "ties to the lower index");
        // z-loss at zero logits: logsumexp = ln E.
        assert_close("z-loss", &o.z_loss.to_vec_f64(), &[(e as f64).ln().powi(2); 2], 1e-12);
    }
    assert_eq!(Moe::top_k(&[0.2, 0.3, 0.3, 0.2], 3), vec![1, 2, 0]);
}

#[test]
fn moe_matches_naive_and_finite_differences() {
    let (s, n, d, e) = (2, 5, 8, 4);
    for (k, normalize) in [(4, false), (4, true), (2, true)] {
        let (m, vars) = moe_vars(mcfg(e, k, normalize, None), &[0, 1], DType::F64, 7);
        let x = rnd(s * n * d, 8, -1.0, 1.0);
        let o = m.forward_moe(&t64(x.clone(), &[s, n, d]), false);
        // Naive: probabilities, top-k, weights, the experts' sum.
        let xt = t64(x.clone(), &[s, n, d]);
        let ys: Vec<Vec<f64>> = m.experts().iter().map(|ex| ex.forward(&xt).to_vec_f64()).collect();
        let r = vars.var("router").to_vec_f64();
        let mut want = vec![0.0; s * n * d];
        let mut pbar = vec![vec![0.0; e]; s];
        let mut f = vec![vec![0.0; e]; s];
        for seed in 0..s {
            for tok in 0..n {
                let row = &x[(seed * n + tok) * d..(seed * n + tok + 1) * d];
                let lg: Vec<f64> = (0..e).map(|j| (0..d).map(|i| row[i] * r[seed * d * e + i * e + j]).sum()).collect();
                let mx = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = lg.iter().map(|l| (l - mx).exp()).sum();
                let p: Vec<f64> = lg.iter().map(|l| (l - mx).exp() / z).collect();
                let top = Moe::top_k(&p, k);
                let norm: f64 = if normalize { top.iter().map(|j| p[*j]).sum() } else { 1.0 };
                for &j in &top {
                    f[seed][j] += 1.0 / (n * k) as f64;
                    for c in 0..d {
                        want[(seed * n + tok) * d + c] += p[j] / norm * ys[j][(seed * n + tok) * d + c];
                    }
                }
                for j in 0..e {
                    pbar[seed][j] += p[j] / n as f64;
                }
            }
        }
        assert_close(&format!("k={k} normalize={normalize} output"), &o.y.to_vec_f64(), &want, 1e-12);
        let bal: Vec<f64> = (0..s).map(|q| e as f64 * (0..e).map(|j| f[q][j] * pbar[q][j]).sum::<f64>()).collect();
        assert_close("balance", &o.balance.to_vec_f64(), &bal, 1e-12);

        // Finite differences on the router and the experts: loss = Σ y·w + balance + z, strict on
        // both expert kinds; SwiGLU with every expert selected (no routing flips).
        let w = t64(rnd(s * n * d, 9, -1.0, 1.0), &[s, n, d]);
        let cases = [(ExpertKind::Plain(Activation::GeluErf), k, 1e-6, 1e-6), (ExpertKind::SwiGlu, e, 1e-6, 1e-6)];
        for (kind, kk, eps, tol) in cases {
            let cfg = MoeConfig { expert: kind, top_k: kk, ..*m.config() };
            let (_, vars) = moe_vars(cfg, &[0, 1], DType::F64, 17);
            let loss = |vm: &VarMap| -> Tensor {
                let mm = moe(cfg, &VarBuilder::from_varmap(vm)).unwrap();
                let o = mm.forward_moe(&xt, false);
                (o.y * w.clone()).sum() + o.balance.sum() + o.z_loss.sum()
            };
            let lifted = vars.lifted();
            let g = lifted.grads(&loss(&lifted).backward());
            let mut checked = 0;
            for (i, name) in vars.names().iter().enumerate() {
                let gv = g[i].as_ref().expect("every var has a gradient (dense dispatch)").to_vec_f64();
                let base = vars.var(name).to_vec_f64();
                for j in [0, base.len() / 3, base.len() - 1] {
                    let at = |dl: f64| { let mut v = base.clone(); v[j] += dl; let mut vm = vars.clone(); vm.set(name, t64(v, vars.var(name).shape())).unwrap(); loss(&vm).to_vec_f64()[0] };
                    let fd = (at(eps) - at(-eps)) / (2.0 * eps);
                    assert!((fd - gv[j]).abs() <= tol * (1.0 + fd.abs()), "k={kk} normalize={normalize} {kind:?} {name}[{j}]: fd {fd:e} vs {:e}", gv[j]);
                    checked += 1;
                }
            }
            assert_eq!(checked, 3 * vars.len());
        }
    }
}

#[test]
fn moe_capacity_seed_independence_and_determinism() {
    let cfg = mcfg(4, 2, true, Some(0.5));
    let (m3, _) = moe_vars(cfg, &[0, 1, 2], DType::F32, 11);
    let (one_vars, _) = moe_vars(cfg, &[1], DType::F32, 11);
    let (s, n, d) = (3, 16, 8);
    let x = rnd(s * n * d, 12, -1.0, 1.0);
    for train in [false, true] {
        let a = m3.forward_moe(&t32(x.clone(), &[s, n, d]), train);
        let b = m3.forward_moe(&t32(x.clone(), &[s, n, d]), train);
        assert_bits("determinism", &a.y.to_vec(), &b.y.to_vec());
        assert_eq!((&a.load, &a.dropped), (&b.load, &b.dropped));
        let one = one_vars.forward_moe(&t32(x[n * d..2 * n * d].to_vec(), &[1, n, d]), train);
        assert_bits(&format!("seed 1 alone (train {train})"), &a.y.to_vec()[n * d..2 * n * d], &one.y.to_vec());
        assert_bits("balance of seed 1", &a.balance.to_vec()[1..2], &one.balance.to_vec());
        assert_eq!(a.load[1], one.load[0]);
        if train {
            // ⌈0.5 · 16 · 2 / 4⌉ = 4 per expert; every assignment is kept or dropped.
            assert_eq!(a.capacity, Some(4));
            for q in 0..s {
                assert!(a.load[q].iter().all(|l| *l <= 4));
                assert_eq!(a.load[q].iter().sum::<usize>() + a.dropped[q], n * 2);
            }
            assert!(a.dropped.iter().sum::<usize>() > 0);
        } else {
            assert_eq!(a.capacity, None);
            assert!(a.dropped.iter().all(|x| *x == 0));
        }
    }
}

// ---------------------------------------------------------------------------- decoder options

fn option_configs() -> Vec<(&'static str, DecoderConfig)> {
    let base = DecoderConfig::new(20, 16, 4, 8, 2, 24);
    vec![
        ("layer norm", DecoderConfig { norm: NormKind::Layer, ..base.clone() }),
        ("gated", base.clone()),
        ("plain gelu", DecoderConfig { mlp: MlpKind::Plain(Activation::GeluErf), ..base.clone() }),
        ("rope", DecoderConfig { positions: Positions::Rope(RopeConfig::default()), ..base.clone() }),
        ("gqa", DecoderConfig { kv_heads: 1, ..base.clone() }),
        ("moe", DecoderConfig { mlp: MlpKind::Moe(MoeSpec { experts: 3, top_k: 2, normalize_topk: false, capacity_factor: Some(1.0), expert: ExpertKind::Plain(Activation::Relu) }), ..base.clone() }),
        ("everything", DecoderConfig { norm: NormKind::Layer, positions: Positions::Rope(RopeConfig { base: 500.0, style: RopeStyle::Interleaved }), kv_heads: 2, dropout: 0.1, dropout_root: 4, ..DecoderConfig::moe_small(20, 8) }),
    ]
}

#[test]
fn decoder_options_build_their_vars() {
    let names = |cfg: &DecoderConfig| -> Vec<String> { Decoder::init(cfg.clone(), 1, 1).unwrap().1.names().to_vec() };
    let o = option_configs();
    let has = |i: usize, n: &str| names(&o[i].1).iter().any(|x| x == n);
    assert!(has(0, "norm_att_bias") && has(0, "b1.norm_mlp_bias") && has(0, "norm_out_bias"));
    assert!(has(1, "mlp_gate"));
    assert!(has(2, "mlp_in") && !has(2, "mlp_gate"));
    assert!(!has(3, "pos"));
    assert_eq!(Decoder::init(o[4].1.clone(), 1, 1).unwrap().1.var("wk").shape(), &[1, 16, 4]);
    assert!(has(5, "moe.router") && has(5, "b1.moe.e2.mlp_out"));
    assert!(has(6, "b0.moe.e3.mlp_gate") && !has(6, "pos"));
    // Invalid combinations are refused.
    assert!(Decoder::init(DecoderConfig { kv_heads: 3, ..DecoderConfig::new(8, 8, 2, 4, 1, 8) }, 1, 1).is_err());
}

/// Every option: the fixed-capacity cache equals the full forward to the bit; padding gives no
/// NaN; dropout acts only in training, per step, reproducibly.
#[test]
fn decoder_options_cache_padding_and_dropout() {
    for (what, cfg) in option_configs() {
        let (dec, _) = Decoder::init(cfg.clone(), 2, 3).unwrap();
        let (s, b, t) = (2, 3, cfg.context);
        let ids = rnd_ints(s * b * t, 13, cfg.vocab as i64);
        let toks = IntTensor::from_data(ids.clone(), [s, b, t]);
        let full = dec.forward(&toks).to_vec();
        let mut cache = dec.cache_padded();
        let pre = 3;
        let v = cfg.vocab;
        let mut rows = vec![0.0f32; full.len()];
        let z = dec.forward_cached(&IntTensor::from_data((0..s * b).flat_map(|r| ids[r * t..r * t + pre].to_vec()).collect(), [s, b, pre]), &mut cache).to_vec();
        for r in 0..s * b {
            rows[r * t * v..(r * t + pre) * v].copy_from_slice(&z[r * pre * v..(r + 1) * pre * v]);
        }
        for i in pre..t {
            let z = dec.forward_cached(&IntTensor::from_data((0..s * b).map(|r| ids[r * t + i]).collect(), [s, b, 1]), &mut cache).to_vec();
            for r in 0..s * b {
                rows[(r * t + i) * v..(r * t + i + 1) * v].copy_from_slice(&z[r * v..(r + 1) * v]);
            }
        }
        assert_bits(&format!("{what}: cached"), &rows, &full);
        // Inference through forward_with is forward.
        let o = dec.forward_with(&toks, &ForwardOptions::default());
        assert_bits(&format!("{what}: forward_with"), &o.logits.to_vec(), &full);
        assert_eq!(o.balance.is_some(), matches!(cfg.mlp, MlpKind::Moe(_)), "{what}");
        // Padding: the first row of each seed is all pads, the second half-padded.
        let pad = BoolTensor::from_data((0..s * b * t).map(|i| { let (r, p) = ((i / t) % b, i % t); r == 0 || (r == 1 && p < t / 2) }).collect(), [s, b, t]);
        let lp = dec.forward_with(&toks, &ForwardOptions { pad: Some(pad), train_step: None }).logits.to_vec();
        assert!(lp.iter().all(|x| x.is_finite()), "{what}: padding NaN");
        // Dropout: only with p > 0 and a training step, by the step.
        let tr = |step| dec.forward_with(&toks, &ForwardOptions { pad: None, train_step: Some(step) }).logits.to_vec();
        let (a, a2, c) = (tr(5), tr(5), tr(6));
        assert_bits(&format!("{what}: a training step reproduces"), &a, &a2);
        if cfg.dropout > 0.0 {
            assert_ne!(a, c, "{what}: steps draw different masks");
            assert_ne!(a, full, "{what}: dropout acts in training");
        } else if !matches!(cfg.mlp, MlpKind::Moe(MoeSpec { capacity_factor: Some(_), .. })) {
            assert_bits(&format!("{what}: no dropout, no capacity: training = inference"), &a, &full);
        }
    }
}

/// Without dropout or MoE, `train_step` is the plain step written out: the masked cross-entropy
/// of the forward pass, its gradient, one optimizer step; to the bit, with per-seed rates.
#[test]
fn trainer_equals_the_plain_step() {
    let cfg = small(32);
    let (_, mut v1) = Decoder::init(cfg.clone(), 2, 9).unwrap();
    let mut v2 = v1.clone();
    let adam = |v: &VarMap| { let mut g = ParamGroups::new(v); g.set_seed_lr(vec![1.0, 0.5]); Adam::new(AdamConfig::default(), g, v) };
    let (mut o1, mut o2) = (adam(&v1), adam(&v2));
    for step in 0..6u64 {
        let (s, b, t) = (2, 2, cfg.context);
        let toks = IntTensor::from_data(rnd_ints(s * b * t, 20 + step, 32), [s, b, t]);
        let tg = IntTensor::from_data(rnd_ints(s * b * t, 40 + step, 32), [s, b, t]);
        let mask = t32(rnd(s * b * t, 60 + step, 0.0, 1.0).into_iter().map(|u| if u < 0.5 { 1.0 } else { 0.0 }).collect(), &[s, b, t]);
        let lifted = v1.lifted();
        let per_seed = masked_cross_entropy(Decoder::load(cfg.clone(), &lifted).unwrap().forward(&toks), &tg, &mask);
        let a = per_seed.to_vec();
        let grads = per_seed.sum().backward();
        o1.step(&mut v1, lifted.grads(&grads), 0.7);
        let r = train_step(&cfg, &mut v2, &mut o2, &toks, &tg, &mask, step, AuxWeights::default(), 0.7);
        assert_bits(&format!("step {step} loss"), &r.ce, &a);
        assert!(r.balance.is_none() && r.load.is_empty());
    }
    for ((n, a), b) in v1.iter().zip(v2.tensors()) {
        assert_bits(n, &a.to_vec(), &b.to_vec());
    }
}

// --------------------------------------------------------------------------- the MoE decoder

/// The small MoE decoder learns a copy task (`a b c SEP a b c`, the answer scored) and uses
/// every expert.
#[test]
fn moe_small_decoder_trains() {
    let cfg = DecoderConfig::moe_small(16, 10);
    let (_, mut vars) = Decoder::init(cfg.clone(), 2, 21).unwrap();
    let n_params = vars.count_per_seed();
    let mut opt = Adam::new(AdamConfig::adamw(3e-3, 0.01, DecayOrder::Torch), ParamGroups::new(&vars), &vars);
    let (s, b, t) = (2, 16, 7);
    let batch = |seed: u64| {
        let a = rnd_ints(s * b * 3, seed, 14);
        let seq: Vec<Vec<i64>> = (0..s * b).map(|r| { let x: Vec<i64> = a[r * 3..r * 3 + 3].iter().map(|v| v + 2).collect(); [x.clone(), vec![1], x, vec![0]].concat() }).collect();
        let toks = IntTensor::from_data(seq.iter().flat_map(|q| q[..t].to_vec()).collect(), [s, b, t]);
        let tg = IntTensor::from_data(seq.iter().flat_map(|q| q[1..].to_vec()).collect(), [s, b, t]);
        let mask = Tensor::from_data((0..s * b * t).map(|i| if (3..6).contains(&(i % t)) { 1.0 } else { 0.0 }).collect(), [s, b, t]);
        (toks, tg, mask)
    };
    let mean = |r: &StepReport| r.ce.iter().sum::<f32>() / r.ce.len() as f32;
    let (mut first, mut last) = (0.0, 0.0);
    for step in 0..200u64 {
        let (toks, tg, mask) = batch(100 + step);
        let r = train_step(&cfg, &mut vars, &mut opt, &toks, &tg, &mask, step, AuxWeights::default(), 1.0);
        if step == 0 {
            first = mean(&r);
        }
        last = mean(&r);
    }
    eprintln!("moe_small: {n_params} params per seed; answer loss {first:.3} → {last:.3}");
    assert!(last < 0.5 * first, "loss did not fall: {first} → {last}");
    // Loads of a batch: every expert is used.
    let toks = IntTensor::from_data(rnd_ints(2 * 8 * 10, 3, 16), [2, 8, 10]);
    let out = Decoder::load(cfg.clone(), &vars).unwrap().forward_with(&toks, &ForwardOptions::default());
    eprintln!("moe_small loads (block, seed, expert): {:?}", out.load);
    assert_eq!(out.load.len(), 2);
    assert!(out.load.iter().flatten().flatten().all(|l| *l > 0), "an unused expert: {:?}", out.load);
    let bal = out.balance.unwrap().to_vec();
    assert!(bal.iter().all(|b| *b > 1.5 && *b < 3.0), "balance per seed (sum of 2 blocks, 2 at perfect balance): {bal:?}");
}
