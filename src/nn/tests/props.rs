//! Property tests: causal invariance, seed independence, save → load → save, the embedding
//! modes, the schedule and the builder's checks.

use super::*;
use crate::nn::*;
use crate::tensor::DType;
use std::cell::RefCell;

/// Logits at positions before k do not depend on the tokens at k and after, to the bit.
#[test]
fn causal_invariance() {
    let cfg = small(32);
    let (s, b, t) = (2, 3, cfg.context);
    let (dec, _) = Decoder::init(cfg.clone(), s, 5).unwrap();
    let a = rnd_ints(s * b * t, 1, 32);
    for k in [1, 7, t - 1] {
        let mut c = a.clone();
        let changed = rnd_ints(s * b * t, 2 + k as u64, 32);
        for i in 0..s * b * t {
            if i % t >= k {
                c[i] = changed[i];
            }
        }
        let la = dec.forward(&IntTensor::from_data(a.clone(), [s, b, t])).to_vec();
        let lc = dec.forward(&IntTensor::from_data(c, [s, b, t])).to_vec();
        let v = cfg.vocab;
        for row in 0..s * b * t {
            if row % t < k {
                assert_bits(&format!("k={k} row {row}"), &la[row * v..(row + 1) * v], &lc[row * v..(row + 1) * v]);
            }
        }
    }
}

/// Seed i's init, logits and gradients do not depend on the seeds beside it.
#[test]
fn seed_independence() {
    let cfg = small(32);
    let (b, t, v) = (2, cfg.context, cfg.vocab);
    let (_, three) = Decoder::init_indexed(cfg.clone(), &[0, 1, 2], 13).unwrap();
    let (_, one) = Decoder::init_indexed(cfg.clone(), &[1], 13).unwrap();
    let (_, five) = Decoder::init(cfg.clone(), 5, 13).unwrap();
    let slot = |x: &Tensor, i: usize, s: usize| -> Vec<f32> { let per = x.numel() / s; x.to_vec()[i * per..(i + 1) * per].to_vec() };
    for ((name, a), (c, e)) in three.iter().zip(one.tensors().iter().zip(five.tensors())) {
        assert_bits(&format!("init {name} vs S=1"), &slot(a, 1, 3), &c.to_vec());
        assert_bits(&format!("init {name} vs S=5"), &slot(a, 1, 3), &slot(e, 1, 5));
    }
    let toks = rnd_ints(3 * b * t, 3, v as i64);
    let next = rnd_ints(3 * b, 4, v as i64);
    let run = |vars: &VarMap, toks: Vec<i64>, next: Vec<i64>, s: usize| -> (Vec<f32>, Vec<Option<Tensor>>) {
        let lifted = vars.lifted();
        let dec = Decoder::load(cfg.clone(), &lifted).unwrap();
        let tk = IntTensor::from_data(toks, [s, b, t]);
        let lg = dec.forward(&tk);
        let loss = next_token_loss(lg.clone(), &tk, &IntTensor::from_data(next, [s, b]));
        (lg.to_vec(), lifted.grads(&loss.sum().backward()))
    };
    let (l3, g3) = run(&three, toks.clone(), next.clone(), 3);
    let (l1, g1) = run(&one, toks[b * t..2 * b * t].to_vec(), next[b..2 * b].to_vec(), 1);
    let per = b * t * v;
    assert_bits("logits of seed 1", &l3[per..2 * per], &l1);
    for (i, name) in three.names().iter().enumerate() {
        let (a, c) = (g3[i].as_ref().unwrap(), g1[i].as_ref().unwrap());
        assert_bits(&format!("gradient {name} of seed 1"), &slot(a, 1, 3), &c.to_vec());
    }
}

#[test]
fn save_load_save_is_exact() {
    let (dec, mut vars) = Decoder::init(small(32), 3, 17).unwrap();
    // Values JSON numbers could not carry: NaN with a payload, ±∞, −0 and a subnormal.
    let e = vars.var("embed").clone();
    let mut ev = e.to_vec();
    ev[0] = f32::from_bits(0x7fc0_1234);
    ev[1] = f32::INFINITY;
    ev[2] = f32::NEG_INFINITY;
    ev[3] = -0.0;
    ev[4] = f32::from_bits(1);
    vars.set("embed", Tensor::from_data(ev, e.shape().to_vec())).unwrap();
    vars.insert("extra_f64", Tensor::from_f64s(vec![std::f64::consts::PI, -0.0, f64::from_bits(0x7ff8_0000_0000_0abc), 1e-310], [1, 2, 2], DType::F64)).unwrap();
    let json = vars.to_json().unwrap();
    let back = VarMap::from_json(&json).unwrap();
    assert_eq!(back.names(), vars.names());
    for ((name, a), b) in vars.iter().zip(back.tensors()) {
        assert_eq!(a.shape(), b.shape(), "{name}");
        assert_eq!(a.dtype(), b.dtype(), "{name}");
        match a.dtype() {
            DType::F64 => assert_eq!(a.to_vec_f64().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.to_vec_f64().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{name}"),
            _ => assert_bits(name, &a.to_vec(), &b.to_vec()),
        }
    }
    assert_eq!(back.to_json().unwrap(), json, "save → load → save");
    // Through a file, and the loaded model computes the same logits.
    let path = std::env::temp_dir().join(format!("noetic-nn-varmap-{}.json", std::process::id()));
    let (_, clean) = Decoder::init(small(32), 3, 17).unwrap();
    clean.save(&path).unwrap();
    let loaded = VarMap::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let toks = IntTensor::from_data(rnd_ints(3 * 2 * 16, 5, 32), [3, 2, 16]);
    assert_bits("logits after load", &Decoder::load(small(32), &loaded).unwrap().forward(&toks).to_vec(), &dec.forward(&toks).to_vec());
    let mut bad = clean.to_file().unwrap();
    bad.format = "noetic.nn.v0".into();
    assert!(VarMap::from_file(&bad).is_err());
}

#[test]
fn gather_embedding_equals_one_hot() {
    let (s, v, d, n) = (2, 11, 4, 30);
    let table = Tensor::from_f64s(rnd(s * v * d, 1, -1.0, 1.0), [s, v, d], DType::F32);
    let ids = IntTensor::from_data(rnd_ints(s * n, 2, v as i64), [s, n]);
    let w = Tensor::from_f64s(rnd(s * n * d, 3, -1.0, 1.0), [s, n, d], DType::F32);
    let run = |mode| {
        let t = table.clone().require_grad();
        let y = Embedding::new(t.clone(), mode).forward(&ids);
        let g = (y.clone() * w.clone()).sum().backward();
        (y.to_vec(), t.grad(&g).unwrap().to_vec())
    };
    let (y1, g1) = run(EmbeddingMode::OneHot);
    let (y2, g2) = run(EmbeddingMode::Gather);
    assert_bits("gather forward", &y2, &y1);
    let (g1, g2): (Vec<f64>, Vec<f64>) = (g1.iter().map(|x| *x as f64).collect(), g2.iter().map(|x| *x as f64).collect());
    assert_close("gather gradient", &g2, &g1, 1e-6);
}

/// Linear warmup over the first 1% of steps, then cosine decay to 10% at the end, written out.
fn cosine_scale(step: u64, total: u64) -> f64 {
    let warm = (total / 100).max(1);
    if step < warm {
        return (step + 1) as f64 / warm as f64;
    }
    let frac = (step - warm) as f64 / (total - warm).max(1) as f64;
    0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * frac.min(1.0)).cos())
}

#[test]
fn warmup_cosine_is_bit_equal_to_its_definition() {
    for total in [1u64, 2, 3, 50, 99, 100, 101, 250, 1000, 4096, 12345, 100_000] {
        let s = WarmupCosine::new(total);
        let steps: Vec<u64> = if total <= 5000 { (0..total + 3).collect() } else { (0..total + 3).step_by(7).chain([total - 1, total]).collect() };
        for step in steps {
            assert_eq!(s.scale(step).to_bits(), cosine_scale(step, total).to_bits(), "total {total} step {step}");
        }
    }
}

#[test]
fn builder_checks_names_and_shapes() {
    let map = RefCell::new(VarMap::new());
    let vb = VarBuilder::init(&map, 2, 1);
    assert_eq!(vb.pp("").path("wq"), "wq");
    assert_eq!(vb.pp("b1").path("wq"), "b1.wq");
    assert_eq!(vb.pp("b1").pp("att").path("wq"), "b1.att.wq");
    let a = vb.get(&[3, 4], "w", Init::fan_in(3)).unwrap();
    assert_eq!(a.shape(), &[2, 3, 4]);
    // Asking again returns the same var; another shape is an error.
    assert_bits("same var", &vb.get(&[3, 4], "w", Init::Const(0.0)).unwrap().to_vec(), &a.to_vec());
    assert!(vb.get(&[4, 3], "w", Init::Const(0.0)).is_err());
    let vars = map.into_inner();
    let load = VarBuilder::from_varmap(&vars);
    assert!(load.get(&[3, 4], "missing", Init::Const(0.0)).is_err());
    assert!(load.get(&[3, 5], "w", Init::Const(0.0)).is_err());
    let mut v2 = vars.clone();
    assert!(v2.insert("w", a.clone()).is_err());
    assert!(v2.set("w", Tensor::zeros([2, 3, 3])).is_err());
    // A module is anything with a forward; ModuleT forwards to it.
    let double = |x: &Tensor| x.clone().mul_scalar(2.0);
    assert_eq!(double.forward_t(&Tensor::ones([1, 2]), true).to_vec(), vec![2.0, 2.0]);
}
