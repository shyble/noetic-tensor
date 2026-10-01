//! CUDA performance. Train-step latency before and after each change, and a
//! kernel-level profile of a small step and a d512 step (ignored; run with
//! `cargo test --release --features cuda --lib cuda_perf -- --ignored --nocapture --test-threads=1`).

use super::*;
use crate::nn::*;
use crate::tensor::{self as nt, CpuMode, Device};

const M: Device = Device::Cuda(0);
const R: Device = Device::Cpu(CpuMode::Reference);

fn vars_on(vars: &VarMap, device: Device) -> VarMap {
    let mut m = VarMap::new();
    for (name, t) in vars.iter() {
        m.insert(name, t.clone().to(device)).unwrap();
    }
    m
}

/// Copy items `a · 1 · a` at random offsets (as the training tests), as tensors `[S, B, T]`.
fn batch(s: usize, b: usize, t: usize, v: usize, seed: u64) -> (IntTensor, IntTensor, Tensor) {
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
    (IntTensor::from_data(toks, [s, b, t]), IntTensor::from_data(tgts, [s, b, t]), Tensor::from_data(mask, [s, b, t]))
}

/// Median ms of `runs` training steps on `device` after a warm-up step.
fn step_ms(cfg: &DecoderConfig, s: usize, b: usize, device: Device, runs: usize) -> f64 {
    let (_, vars) = Decoder::init(cfg.clone(), s, 3).unwrap();
    nt::set_default_device(device).unwrap();
    let mut v = vars_on(&vars, device);
    let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&v), &v);
    let (tk, tg, mk) = batch(s, b, cfg.context, cfg.vocab, 4);
    train_step(cfg, &mut v, &mut opt, &tk, &tg, &mk, 0, AuxWeights::default(), 1.0);
    let mut times: Vec<f64> = (0..runs)
        .map(|i| {
            let t = std::time::Instant::now();
            train_step(cfg, &mut v, &mut opt, &tk, &tg, &mk, 1 + i as u64, AuxWeights::default(), 1.0);
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    nt::set_default_device(R).unwrap();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[runs / 2]
}

/// Train-step latency at d 64, 256 and 512 (the benchmark configuration: 4 blocks, B 8, T 64, V 256, S 1),
/// CpuRef and CUDA, median of 9.
#[test]
#[ignore]
fn cuda_perf_train_latency() {
    for d in [64usize, 256, 512] {
        let cfg = DecoderConfig::new(256, d, 8, 64, 4, 4 * d);
        let cpu = step_ms(&cfg, 1, 8, R, 5);
        let gpu = step_ms(&cfg, 1, 8, M, 9);
        eprintln!("cuda train step d {d:>4} (4 blocks, B 8, T 64, V 256): cpu_ref {cpu:8.2} ms  cuda {gpu:8.2} ms  ({:.2}x over ref)", cpu / gpu);
    }
}

fn report(what: &str, p: &nt::cuda::ProfileReport) {
    eprintln!(
        "cuda profile {what}: {} launches (host {:.2} ms in launch calls); GPU busy {:.2} ms over a span of {:.2} ms; {} allocations ({:.1} MB, host {:.2} ms), {} frees (host {:.2} ms)",
        p.launches, p.launch_host_ms, p.gpu_busy_ms, p.gpu_span_ms, p.allocs, p.alloc_bytes as f64 / 1e6, p.alloc_host_ms, p.frees, p.free_host_ms
    );
    for (k, n, ms) in p.kernels.iter().take(16) {
        eprintln!("cuda profile {what}:   {k:<18} {n:>5} launches {ms:8.3} ms ({:4.1}% of busy)", 100.0 * ms / p.gpu_busy_ms.max(1e-9));
    }
}

/// One training step (trainer::train_step's body), profiled in two parts: forward + backward,
/// and the optimizer step.
fn profile(what: &str, cfg: &DecoderConfig, s: usize, b: usize) {
    let (_, vars) = Decoder::init(cfg.clone(), s, 3).unwrap();
    nt::set_default_device(M).unwrap();
    let mut v = vars_on(&vars, M);
    let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&v), &v);
    let (tk, tg, mk) = batch(s, b, cfg.context, cfg.vocab, 4);
    for i in 0..3 {
        train_step(cfg, &mut v, &mut opt, &tk, &tg, &mk, i, AuxWeights::default(), 1.0);
    }
    let mut plain: Vec<f64> = (0..5)
        .map(|i| {
            let t = std::time::Instant::now();
            train_step(cfg, &mut v, &mut opt, &tk, &tg, &mk, 3 + i, AuxWeights::default(), 1.0);
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    plain.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // The same step as trainer::train_step, in two profiled parts.
    nt::cuda::profile_start().unwrap();
    let t = std::time::Instant::now();
    let lifted = v.lifted();
    let model = Decoder::load(cfg.clone(), &lifted).unwrap();
    let out = model.forward_with(&tk, &ForwardOptions { pad: None, train_step: Some(9) });
    let ce = masked_cross_entropy(out.logits, &tg, &mk);
    let _ = ce.to_vec();
    let g = ce.sum().backward();
    let grads = lifted.grads(&g);
    let fb = nt::cuda::profile_take().unwrap();
    let t_fb = t.elapsed().as_secs_f64() * 1e3;
    nt::cuda::profile_start().unwrap();
    let t = std::time::Instant::now();
    opt.step(&mut v, grads, 1.0);
    let op = nt::cuda::profile_take().unwrap();
    let t_op = t.elapsed().as_secs_f64() * 1e3;
    nt::set_default_device(R).unwrap();
    eprintln!("cuda profile {what}: step median {:.2} ms unprofiled; profiled forward+backward {t_fb:.2} ms, optimizer {t_op:.2} ms ({} vars)", plain[2], v.tensors().len());
    report(&format!("{what} forward+backward"), &fb);
    report(&format!("{what} optimizer"), &op);
}

/// A kernel-level profile of one training step: the small decoder (S 4, B 16, as the training traffic
/// test) and d512 (the latency benchmark's largest).
#[test]
#[ignore]
fn cuda_perf_profile() {
    profile("small S4 B16", &small(32), 4, 16);
    profile("d512 S1 B8", &DecoderConfig::new(256, 512, 8, 64, 4, 2048), 1, 8);
}
