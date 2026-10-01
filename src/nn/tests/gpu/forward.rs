// Forward, one set of bodies for every GPU backend: the nn decoder's forward and loss on the device
// against CpuRef (logits within 1e-5, |Δ|/(1+|ref|)), at S ∈ {1, 3}; cached greedy tokens
// identical wherever the CPU's top two logits are not tied; a forward is byte-identical run to
// run; a non-MoE forward reads the GPU back once (its logits) and makes no host round trip.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::nn::tests::*;
use crate::nn::*;
use crate::tensor::{self as nt, CpuMode, Device, IntTensor, Tensor};

const R: Device = Device::Cpu(CpuMode::Reference);

fn configs() -> Vec<(&'static str, DecoderConfig)> {
    let mut rope_ln = tiny(26, 12);
    rope_ln.positions = Positions::Rope(RopeConfig::default());
    rope_ln.norm = NormKind::Layer;
    rope_ln.kv_heads = 1;
    rope_ln.embedding = EmbeddingMode::Gather;
    vec![
        ("small", small(32)),
        ("tiny", tiny(26, 12)),
        ("rope+layernorm+gqa+gather", rope_ln),
        ("moe_small", DecoderConfig::moe_small(20, 10)),
    ]
}

/// The same vars on `device`.
fn vars_on(vars: &VarMap, device: Device) -> VarMap {
    let mut m = VarMap::new();
    for (name, t) in vars.iter() {
        m.insert(name, t.clone().to(device)).unwrap();
    }
    m
}

/// Largest |a − b| / (1 + |b|).
fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| ((*x as f64 - *y as f64).abs()) / (1.0 + (*y as f64).abs())).fold(0.0, f64::max)
}

/// Logits and per-seed loss of `cfg` on `device` (vars built on the CPU, then moved).
fn forward_on(cfg: &DecoderConfig, vars: &VarMap, toks: &[i64], tgts: &[i64], dims: [usize; 3], device: Device) -> (Vec<f32>, Vec<f32>) {
    nt::set_default_device(device).unwrap();
    let dec = Decoder::load(cfg.clone(), &vars_on(vars, device)).unwrap();
    let logits = dec.forward(&IntTensor::from_data(toks.to_vec(), dims));
    assert_eq!(logits.device(), device);
    let loss = cross_entropy(logits.clone(), &IntTensor::from_data(tgts.to_vec(), dims));
    let out = (logits.to_vec(), loss.to_vec());
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_decoder_forward_and_loss_within_1e5() {
    for (what, cfg) in configs() {
        for s in [1, 3] {
            let (_, vars) = Decoder::init(cfg.clone(), s, 23).unwrap();
            let (b, t) = (2, cfg.context);
            let dims = [s, b, t];
            let toks = rnd_ints(s * b * t, 5, cfg.vocab as i64);
            let tgts = rnd_ints(s * b * t, 6, cfg.vocab as i64);
            let (lc, cc) = forward_on(&cfg, &vars, &toks, &tgts, dims, R);
            let (lm, cm) = forward_on(&cfg, &vars, &toks, &tgts, dims, M);
            let (dl, dc) = (rel(&lm, &lc), rel(&cm, &cc));
            let same = lm.iter().zip(&lc).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            eprintln!("{TAG} vs cpu_ref decoder {what:<26} S={s}: logits max rel {dl:.2e} ({same}/{} bit-equal), loss max rel {dc:.2e}", lc.len());
            assert!(dl <= 1e-5 && dc <= 1e-5, "{what} S={s}: logits {dl:e}, loss {dc:e}");
            // Run to run on the device: byte-identical.
            let (lm2, cm2) = forward_on(&cfg, &vars, &toks, &tgts, dims, M);
            assert_bits(&format!("{what} S={s} logits twice"), &lm2, &lm);
            assert_bits(&format!("{what} S={s} loss twice"), &cm2, &cm);
        }
    }
}

/// The vars of `cfg` with `s` seeds, trained a few steps on the CPU on a copy task so the
/// logits are not those of a fresh init.
fn trained_vars(cfg: &DecoderConfig, s: usize, steps: usize) -> VarMap {
    let (_, mut vars) = Decoder::init(cfg.clone(), s, 19).unwrap();
    let mut opt = Adam::new(AdamConfig { lr: 1e-2, ..Default::default() }, ParamGroups::new(&vars), &vars);
    let (b, t) = (4, 7);
    for step in 0..steps as u64 {
        let a = rnd_ints(s * b * 3, 50 + step, cfg.vocab as i64 - 1);
        let seq: Vec<Vec<i64>> = (0..s * b).map(|r| { let x: Vec<i64> = a[r * 3..r * 3 + 3].iter().map(|v| v + 1).collect(); [x.clone(), vec![1], x, vec![0]].concat() }).collect();
        let toks = IntTensor::from_data(seq.iter().flat_map(|q| q[..t].to_vec()).collect(), [s, b, t]);
        let tg = IntTensor::from_data(seq.iter().flat_map(|q| q[1..].to_vec()).collect(), [s, b, t]);
        let mask = Tensor::from_data((0..s * b * t).map(|i| if (3..6).contains(&(i % t)) { 1.0 } else { 0.0 }).collect(), [s, b, t]);
        train_step(cfg, &mut vars, &mut opt, &toks, &tg, &mask, step, AuxWeights::default(), 1.0);
    }
    vars
}

/// Greedy decoding of `len` tokens after the prompts (`[s, q, n]`) through a fixed-capacity cache
/// on the default device: the tokens per row and the logits each token was chosen from.
fn greedy(dec: &Decoder, prompts: &[i64], [s, q, n]: [usize; 3], len: usize, v: usize) -> (Vec<Vec<i64>>, Vec<Vec<Vec<f32>>>) {
    let mut cache = dec.cache_padded();
    let mut z = dec.forward_cached(&IntTensor::from_data(prompts.to_vec(), [s, q, n]), &mut cache).to_vec();
    let mut last: Vec<Vec<f32>> = (0..s * q).map(|r| z[(r * n + n - 1) * v..(r * n + n) * v].to_vec()).collect();
    let (mut toks, mut rows) = (vec![Vec::new(); s * q], vec![Vec::new(); s * q]);
    for step in 0..len {
        let next: Vec<i64> = last.iter().map(|l| (0..v).fold(0, |m, i| if l[i] > l[m] { i } else { m }) as i64).collect();
        for r in 0..s * q {
            toks[r].push(next[r]);
            rows[r].push(last[r].clone());
        }
        if step + 1 < len {
            z = dec.forward_cached(&IntTensor::from_data(next, [s, q, 1]), &mut cache).to_vec();
            last = (0..s * q).map(|r| z[r * v..(r + 1) * v].to_vec()).collect();
        }
    }
    (toks, rows)
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_greedy_tokens_identical_where_untied() {
    let (mut compared, mut tied) = (0, 0);
    for (what, cfg) in configs().into_iter().take(2) {
        for s in [1, 3] {
            let vars = trained_vars(&cfg, s, 5);
            let (q, n, len, v) = (3, 4, 5, cfg.vocab);
            let r = rnd_ints(s * q * n, 31, v as i64 - 2);
            let prompts: Vec<i64> = (0..s * q).flat_map(|k| { let mut p: Vec<i64> = (0..n - 1).map(|j| 2 + r[k * n + j]).collect(); p.push(1); p }).collect();
            let (ca, crows) = greedy(&Decoder::load(cfg.clone(), &vars).unwrap(), &prompts, [s, q, n], len, v);
            nt::set_default_device(M).unwrap();
            let (ga, grows) = greedy(&Decoder::load(cfg.clone(), &vars_on(&vars, M)).unwrap(), &prompts, [s, q, n], len, v);
            nt::set_default_device(R).unwrap();
            for k in 0..s * q {
                compared += 1;
                if ga[k] == ca[k] {
                    continue;
                }
                // A differing answer is allowed only after a step whose top two CPU logits tie
                // within the logit tolerance.
                let step = (0..len).find(|&i| ga[k][i] != ca[k][i]).unwrap();
                let mut row = crows[k][step].clone();
                row.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let gap = (row[0] - row[1]) as f64;
                assert!(gap <= 1e-5 * (1.0 + row[0].abs() as f64), "{what} S={s} row {k}: tokens differ at step {step} with a CPU top-2 gap of {gap:e}");
                tied += 1;
            }
            let d = grows.iter().flatten().zip(crows.iter().flatten()).map(|(a, b)| rel(a, b)).fold(0.0, f64::max);
            eprintln!("{TAG} vs cpu_ref greedy {what:<10} S={s}: answers equal {}, decoded-row logits max rel {d:.2e}", ga == ca);
        }
    }
    eprintln!("greedy: {compared} answers compared, {tied} differed after a tie");
}

/// The forward bodies (decoder forward and loss, cached greedy, forward traffic), in a fresh process.
#[test]
fn decoder_on_the_device() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 3);
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_forward_traffic() {
    let cfg = small(32);
    let (_, vars) = Decoder::init(cfg.clone(), 3, 23).unwrap();
    nt::set_default_device(M).unwrap();
    let dec = Decoder::load(cfg.clone(), &vars_on(&vars, M)).unwrap();
    let toks = IntTensor::from_data(rnd_ints(3 * 2 * cfg.context, 5, cfg.vocab as i64), [3, 2, cfg.context]);
    let (_, d0, r0) = nt::transfer_counts();
    let k0 = launches();
    let logits = dec.forward(&toks);
    let (_, d1, r1) = nt::transfer_counts();
    eprintln!("forward: {} kernel launches", launches() - k0);
    let _ = logits.to_vec();
    let (_, d2, r2) = nt::transfer_counts();
    eprintln!("forward: downloads {} then {}, round trips {}", d1 - d0, d2 - d1, r2 - r0);
    assert_eq!((d1 - d0, r1 - r0), (0, 0), "the forward stays on the device");
    assert_eq!((d2 - d1, r2 - r1), (1, 0), "one read of the logits");
}

/// Decoder forward latency on CpuRef, CpuFast and the GPU (ignored; `cargo test --release
/// --features metal --lib decoder_latency -- --ignored --nocapture`).
#[test]
#[ignore]
fn decoder_latency() {
    let time = |f: &mut dyn FnMut()| {
        f();
        let mut v: Vec<f64> = (0..5)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[2]
    };
    for d in [64usize, 256, 512] {
        let cfg = DecoderConfig::new(256, d, 8, 64, 4, 4 * d);
        let (_, vars) = Decoder::init(cfg.clone(), 1, 3).unwrap();
        let toks = rnd_ints(8 * 64, 4, 256);
        let mut row = Vec::new();
        for dev in [R, Device::Cpu(CpuMode::Fast), M] {
            nt::set_default_device(dev).unwrap();
            let dec = Decoder::load(cfg.clone(), &vars_on(&vars, dev)).unwrap();
            let ids = IntTensor::from_data(toks.clone(), [1, 8, 64]);
            row.push(time(&mut || drop(dec.forward(&ids).to_vec())));
            nt::set_default_device(R).unwrap();
        }
        eprintln!("{TAG} decoder forward d {d:>4} (4 blocks, B 8, T 64, V 256): cpu_ref {:8.2} ms  cpu_fast {:8.2} ms  {TAG} {:8.2} ms  ({:.2}x over ref)", row[0], row[1], row[2], row[0] / row[2]);
    }
}
