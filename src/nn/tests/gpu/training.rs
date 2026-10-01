// Training, one set of bodies for every GPU backend: training on the device. Decoder gradients against
// CpuRef within 1e-4 (|Δ|/(1+|ref|)); Adam (with lazy rows, per-seed rates, coupled and
// decoupled decay) on the device; a training step's host traffic; a 200-step trajectory test (a
// 16-member ±1/±2-ulp CPU ensemble sets the envelope) on the small and the tiny decoder; and
// bitwise repeatability (two runs byte-identical).

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::nn::tests::*;
use crate::nn::*;
use crate::tensor::{self as nt, CpuMode, Device, IntTensor, Tensor};

const R: Device = Device::Cpu(CpuMode::Reference);

fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| ((*x as f64 - *y as f64).abs()) / (1.0 + (*y as f64).abs())).fold(0.0, f64::max)
}

fn configs() -> Vec<(&'static str, DecoderConfig)> {
    let mut rope_ln = tiny(26, 12);
    rope_ln.positions = Positions::Rope(RopeConfig::default());
    rope_ln.norm = NormKind::Layer;
    rope_ln.kv_heads = 1;
    rope_ln.embedding = EmbeddingMode::Gather;
    let mut dropout = tiny(26, 12);
    dropout.dropout = 0.1;
    dropout.dropout_root = 5;
    vec![
        ("small", small(32)),
        ("tiny", tiny(26, 12)),
        ("rope+layernorm+gqa+gather", rope_ln),
        ("tiny+dropout", dropout),
        ("moe_small", DecoderConfig::moe_small(20, 10)),
    ]
}

fn vars_on(vars: &VarMap, device: Device) -> VarMap {
    let mut m = VarMap::new();
    for (name, t) in vars.iter() {
        m.insert(name, t.clone().to(device)).unwrap();
    }
    m
}

/// A batch `[S, B, T]`: copy items `a · 1 · a` (a: 3 random symbols ≥ 2) at random offsets,
/// PAD 0 elsewhere; targets are the next token, the mask scores non-PAD targets.
struct Batch {
    dims: [usize; 3],
    toks: Vec<i64>,
    tgts: Vec<i64>,
    mask: Vec<f32>,
}

fn batch(s: usize, b: usize, t: usize, v: usize, seed: u64) -> Batch {
    use rand::Rng;
    let mut r = rng(seed);
    let (mut toks, mut tgts, mut mask) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..s * b {
        let a: Vec<i64> = (0..3).map(|_| r.gen_range(2..v as i64)).collect();
        let item = [a.clone(), vec![1], a].concat();
        let off = r.gen_range(0..=t + 1 - item.len());
        let mut seq = vec![0i64; t + 1];
        seq[off..off + item.len()].copy_from_slice(&item);
        toks.extend_from_slice(&seq[..t]);
        tgts.extend_from_slice(&seq[1..]);
        mask.extend(seq[1..].iter().map(|x| if *x == 0 { 0.0f32 } else { 1.0 }));
    }
    Batch { dims: [s, b, t], toks, tgts, mask }
}

fn tensors(b: &Batch) -> (IntTensor, IntTensor, Tensor) {
    (IntTensor::from_data(b.toks.clone(), b.dims), IntTensor::from_data(b.tgts.clone(), b.dims), Tensor::from_data(b.mask.clone(), b.dims))
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_decoder_gradients_within_1e4() {
    for (what, cfg) in configs() {
        for s in [1, 3] {
            let (_, vars) = Decoder::init(cfg.clone(), s, 29).unwrap();
            let bt = batch(s, 3, cfg.context, cfg.vocab, 41);
            let run = |device: Device| {
                nt::set_default_device(device).unwrap();
                let lifted = vars_on(&vars, device).lifted();
                let model = Decoder::load(cfg.clone(), &lifted).unwrap();
                let (tk, tg, mk) = tensors(&bt);
                let out = model.forward_with(&tk, &ForwardOptions { pad: None, train_step: Some(3) });
                let mut loss = masked_cross_entropy(out.logits, &tg, &mk);
                if let Some(bl) = out.balance {
                    loss = loss + bl.mul_scalar(0.01);
                }
                let g = loss.clone().sum().backward();
                let grads: Vec<Vec<f32>> = lifted.grads(&g).into_iter().map(|x| x.map(|t| t.to_vec()).unwrap_or_default()).collect();
                nt::set_default_device(R).unwrap();
                (loss.to_vec(), grads)
            };
            let (lc, gc) = run(R);
            let (lm, gm) = run(M);
            let dl = rel(&lm, &lc);
            let dg = gm.iter().zip(&gc).map(|(a, b)| rel(a, b)).fold(0.0, f64::max);
            let (same, all) = gm.iter().zip(&gc).fold((0, 0), |(s0, n0), (a, b)| (s0 + a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count(), n0 + b.len()));
            eprintln!("{TAG} decoder {what:<26} S={s}: loss rel {dl:.2e}, gradients max rel {dg:.2e} ({same}/{all} bit-equal)");
            assert!(dl <= 1e-5 && dg <= 1e-4, "{what} S={s}: loss {dl:e}, gradients {dg:e}");
        }
    }
}

/// Train `steps` Adam steps from `vars` on `device` over `batches`; the per-step per-seed CE,
/// the final CE on `eval` (forward only) and the final vars (as host values).
fn train(cfg: &DecoderConfig, vars: &VarMap, adam: &AdamConfig, groups: Option<&dyn Fn(&mut ParamGroups)>, batches: &[Batch], eval: &Batch, device: Device) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
    nt::set_default_device(device).unwrap();
    let mut vars = vars_on(vars, device);
    let mut g = ParamGroups::new(&vars);
    if let Some(f) = groups {
        f(&mut g);
    }
    let mut opt = Adam::new(adam.clone(), g, &vars);
    let mut losses = Vec::with_capacity(batches.len());
    for (step, b) in batches.iter().enumerate() {
        let (tk, tg, mk) = tensors(b);
        losses.push(train_step(cfg, &mut vars, &mut opt, &tk, &tg, &mk, step as u64, AuxWeights::default(), 1.0).ce);
    }
    let (tk, tg, mk) = tensors(eval);
    let model = Decoder::load(cfg.clone(), &vars).unwrap();
    let fin = masked_cross_entropy(model.forward(&tk), &tg, &mk).to_vec();
    let out = vars.tensors().iter().map(|t| t.to_vec()).collect();
    nt::set_default_device(R).unwrap();
    (losses, fin, out)
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_adam_options_on_the_device() {
    // Lazy rows on the gather embedding (rows of unseen symbols get no gradient), per-seed
    // rates, coupled decay off (lazy rows forbid it) and decoupled decay on; 20 steps.
    let mut cfg = tiny(26, 12);
    cfg.embedding = EmbeddingMode::Gather;
    let s = 3;
    let (_, vars) = Decoder::init(cfg.clone(), s, 31).unwrap();
    let spec = AdamConfig { lr: 1e-2, decoupled_decay: 0.05, ..Default::default() };
    let lazy = |g: &mut ParamGroups| {
        g.options_mut("embed").lazy_rows = true;
        g.set_seed_lr(vec![1.0, 0.5, 2.0]);
    };
    let batches: Vec<Batch> = (0..20).map(|i| batch(s, 4, cfg.context, 12, 500 + i)).collect();
    let eval = batch(s, 8, cfg.context, 12, 999);
    let (lc, fc, vc) = train(&cfg, &vars, &spec, Some(&lazy), &batches, &eval, R);
    let (lm, fm, vm) = train(&cfg, &vars, &spec, Some(&lazy), &batches, &eval, M);
    let dl = lc.iter().zip(&lm).map(|(a, b)| rel(b, a)).fold(0.0, f64::max);
    let dv = vc.iter().zip(&vm).map(|(a, b)| rel(b, a)).fold(0.0, f64::max);
    let df = rel(&fm, &fc);
    // Lazy rows: symbols 12..26 never occur, so their embedding rows only take the decoupled
    // decay, an exact scalar product: equal on both devices, and not moved by Adam.
    let d = cfg.d;
    let rows = |e: &[f32]| (0..s).flat_map(|si| (12..26).flat_map(move |row| (0..d).map(move |j| (si * 26 + row) * d + j))).map(|k| e[k]).collect::<Vec<f32>>();
    assert_bits("unseen embedding rows", &rows(&vm[0]), &rows(&vc[0]));
    assert_eq!(vars.names()[0], "embed");
    eprintln!("{TAG} adam options (lazy rows, seed lr, AdamW) 20 steps: losses rel {dl:.2e}, final loss rel {df:.2e}, vars rel {dv:.2e}");
    assert!(dl <= 1e-4 && df <= 1e-4 && dv <= 1e-4, "losses {dl:e}, final {df:e}, vars {dv:e}");
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_training_step_traffic() {
    let cfg = small(32);
    let (_, vars) = Decoder::init(cfg.clone(), 4, 3).unwrap();
    nt::set_default_device(M).unwrap();
    let mut vars = vars_on(&vars, M);
    let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&vars), &vars);
    let b = batch(4, 16, cfg.context, cfg.vocab, 7);
    let (tk, tg, mk) = tensors(&b);
    train_step(&cfg, &mut vars, &mut opt, &tk, &tg, &mk, 0, AuxWeights::default(), 1.0);
    let (u0, d0, r0) = nt::transfer_counts();
    train_step(&cfg, &mut vars, &mut opt, &tk, &tg, &mk, 1, AuxWeights::default(), 1.0);
    let (u1, d1, r1) = nt::transfer_counts();
    eprintln!("{TAG} training step (small, S 4, B 16): uploads {}, downloads {}, host round trips {}", u1 - u0, d1 - d0, r1 - r0);
    assert_eq!(r1 - r0, 0, "no host round trip in a step");
    assert_eq!(d1 - d0, 1, "one read per step: the per-seed losses");
    assert!(vars.tensors().iter().all(|t| t.device() == M), "the vars stay on the device");
    assert!(opt.state().0.iter().chain(opt.state().1).all(|t| t.device() == M), "Adam's moments stay on the device");
    nt::set_default_device(R).unwrap();
}

/// f32 values moved by `ulps` units in the last place where the mask stream says so (each
/// entry with probability 1/2).
fn perturb(vars: &VarMap, ulps: i32, mask_seed: u64) -> VarMap {
    use rand::Rng;
    let mut r = rng(mask_seed);
    let mut m = VarMap::new();
    for (name, t) in vars.iter() {
        let v: Vec<f32> = t.to_vec().into_iter().map(|x| if r.gen_bool(0.5) && x != 0.0 { f32::from_bits((x.to_bits() as i64 + ulps as i64) as u32) } else { x }).collect();
        m.insert(name, Tensor::from_data(v, t.shape().to_vec())).unwrap();
    }
    m
}

/// ±1 and ±2 ulps, each on four fixed masks (the ensemble).
const MEMBERS: [(i32, u64); 16] = [(1, 1), (-1, 2), (2, 3), (-2, 4), (1, 5), (-1, 6), (2, 7), (-2, 8), (1, 9), (-1, 10), (2, 11), (-2, 12), (1, 13), (-1, 14), (2, 15), (-2, 16)];
/// t_{0.995, 15}.
const T_995_15: f64 = 2.946_712_883;

/// The trajectory test: 200 steps, 4 seeds, batch 16, lr 3e-3 constant, Metal against CpuRef
/// from the same weights and batches; the envelope from the 16-member CPU ensemble.
fn trajectory(what: &str, cfg: &DecoderConfig) {
    const S: usize = 4;
    const STEPS: usize = 200;
    let (_, vars) = Decoder::init(cfg.clone(), S, 101).unwrap();
    let spec = AdamConfig::default();
    let batches: Vec<Batch> = (0..STEPS).map(|i| batch(S, 16, cfg.context, cfg.vocab, 10_000 + i as u64)).collect();
    let eval = batch(S, 64, cfg.context, cfg.vocab, 99_999);
    let (cpu, cpu_final, _) = train(cfg, &vars, &spec, None, &batches, &eval, R);
    // The ensemble, in parallel (each thread's default device is Reference).
    let members: Vec<(Vec<Vec<f32>>, Vec<f32>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = MEMBERS
            .iter()
            .map(|&(u, m)| {
                let (vars, spec, batches, eval) = (&vars, &spec, &batches, &eval);
                sc.spawn(move || {
                    let (l, f, _) = train(cfg, &perturb(vars, u, m), spec, None, batches, eval, R);
                    (l, f)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (gpu, gpu_final, gpu_vars) = train(cfg, &vars, &spec, None, &batches, &eval, M);
    let (gpu2, gpu2_final, gpu2_vars) = train(cfg, &vars, &spec, None, &batches, &eval, M);
    // Bitwise repeatability on the device.
    assert_bits(&format!("{what}: losses of two {TAG} runs"), &gpu.concat(), &gpu2.concat());
    assert_bits(&format!("{what}: final loss of two {TAG} runs"), &gpu_final, &gpu2_final);
    assert_bits(&format!("{what}: final vars of two {TAG} runs"), &gpu_vars.concat(), &gpu2_vars.concat());
    let dmax = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let e: Vec<f32> = (0..STEPS).map(|t| members.iter().map(|(l, _)| dmax(&l[t], &cpu[t])).fold(0f32, f32::max)).collect();
    let e_final = members.iter().map(|(_, f)| dmax(f, &cpu_final)).fold(0f32, f32::max);
    let delta: Vec<f32> = (0..STEPS).map(|t| dmax(&gpu[t], &cpu[t])).collect();
    let d_final = dmax(&gpu_final, &cpu_final);
    let mean_signed = |l: &[Vec<f32>]| (0..STEPS).flat_map(|t| (0..S).map(move |s| (t, s))).map(|(t, s)| l[t][s] as f64 - cpu[t][s] as f64).sum::<f64>() / (STEPS * S) as f64;
    let means: Vec<f64> = members.iter().map(|(l, _)| mean_signed(l)).collect();
    let m = means.iter().sum::<f64>() / means.len() as f64;
    let sd = (means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (means.len() - 1) as f64).sqrt();
    let b = mean_signed(&gpu);
    let pi = T_995_15 * sd * (1.0 + 1.0 / means.len() as f64).sqrt();
    let worst = (0..STEPS).map(|t| delta[t] / 1e-5f32.max(2.0 * e[t])).fold(0f32, f32::max);
    let first20 = delta[..20].iter().cloned().fold(0f32, f32::max);
    eprintln!(
        "{TAG} trajectory {what}: loss {:.4} → {:.4}; max Δ_t {:.2e} (E_t max {:.2e}), worst Δ_t/max(1e-5, 2E_t) {worst:.3}; first 20 steps max Δ {first20:.2e}; final Δ {d_final:.2e} (E_final {e_final:.2e}); bias b {b:.2e}, members m {m:.2e} s {sd:.2e}, |b−m| {:.2e} vs PI {pi:.2e}; two {TAG} runs byte-identical",
        cpu[0].iter().sum::<f32>() / S as f32,
        cpu[STEPS - 1].iter().sum::<f32>() / S as f32,
        delta.iter().cloned().fold(0f32, f32::max),
        e.iter().cloned().fold(0f32, f32::max),
        (b - m).abs()
    );
    assert!(worst <= 1.0, "{what}: a step outside max(1e-5, 2 E_t)");
    assert!(first20 <= 1e-5, "{what}: the first 20 steps differ by {first20:e}");
    assert!(d_final <= 2.5e-4f32.max(2.0 * e_final), "{what}: final Δ {d_final:e}");
    if (b - m).abs() > pi {
        assert!((b - m).abs() < 10.0 * sd, "{what}: bias {b:e} outside 10 s of the members' mean");
        eprintln!("{TAG} trajectory {what}: CHECK — bias outside the 99% prediction interval but within 10 s");
    }
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_trajectory_small() {
    trajectory("small", &small(32));
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_trajectory_tiny() {
    trajectory("tiny", &tiny(26, 12));
}

/// The training bodies, in a fresh process.
#[test]
fn training_on_the_device() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 5);
}

/// Training-step latency on CpuRef, CpuFast and the GPU (ignored; `cargo test --release
/// --features metal --lib train_latency -- --ignored --nocapture`).
#[test]
#[ignore]
fn train_latency() {
    for d in [64usize, 256, 512] {
        let cfg = DecoderConfig::new(256, d, 8, 64, 4, 4 * d);
        let (_, vars) = Decoder::init(cfg.clone(), 1, 3).unwrap();
        let b = batch(1, 8, 64, 256, 4);
        let mut row = Vec::new();
        for dev in [R, Device::Cpu(CpuMode::Fast), M] {
            nt::set_default_device(dev).unwrap();
            let mut v = vars_on(&vars, dev);
            let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&v), &v);
            let (tk, tg, mk) = tensors(&b);
            let mut times: Vec<f64> = (0..6)
                .map(|i| {
                    let t = std::time::Instant::now();
                    train_step(&cfg, &mut v, &mut opt, &tk, &tg, &mk, i, AuxWeights::default(), 1.0);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .skip(1)
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            row.push(times[2]);
            nt::set_default_device(R).unwrap();
        }
        eprintln!("{TAG} train step d {d:>4} (4 blocks, B 8, T 64, V 256): cpu_ref {:8.2} ms  cpu_fast {:8.2} ms  {TAG} {:8.2} ms  ({:.2}x over ref)", row[0], row[1], row[2], row[0] / row[2]);
    }
}
