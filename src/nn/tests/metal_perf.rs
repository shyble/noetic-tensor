//! Metal performance. A profile of a small training step and of
//! a d512 nn training step (where host and GPU time goes), and the train-step speed table
//! (both ignored: `cargo test --release --features metal --lib metal_perf -- --ignored
//! --nocapture`).

use super::*;
use crate::nn::*;
use crate::tensor::{self as nt, CpuMode, Device};

const M: Device = Device::Metal(0);
const R: Device = Device::Cpu(CpuMode::Reference);

fn vars_on(vars: &VarMap, device: Device) -> VarMap {
    let mut m = VarMap::new();
    for (name, t) in vars.iter() {
        m.insert(name, t.clone().to(device)).unwrap();
    }
    m
}

/// A copy-task batch `[s, b, t]` (items `a·1·a` at random offsets; PAD 0 elsewhere).
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

fn ms(ns: u64, steps: usize) -> f64 {
    ns as f64 / 1e6 / steps as f64
}

fn profile(what: &str, cfg: &DecoderConfig, s: usize, b: usize) {
    nt::set_default_device(M).unwrap();
    let (_, vars) = Decoder::init(cfg.clone(), s, 3).unwrap();
    let mut vars = vars_on(&vars, M);
    let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&vars), &vars);
    let (tk, tg, mk) = batch(s, b, cfg.context, cfg.vocab, 7);
    for i in 0..3 {
        train_step(cfg, &mut vars, &mut opt, &tk, &tg, &mk, i, AuxWeights::default(), 1.0);
    }
    const N: usize = 10;
    let _ = nt::metal_profile();
    let t0 = std::time::Instant::now();
    for i in 0..N {
        train_step(cfg, &mut vars, &mut opt, &tk, &tg, &mk, 3 + i as u64, AuxWeights::default(), 1.0);
    }
    let wall = t0.elapsed().as_nanos() as u64;
    let p = nt::metal_profile();
    let host = wall.saturating_sub(p.wait_ns);
    eprintln!("metal profile {what}: per step wall {:.2} ms = host {:.2} ms (encode {:.2}, buffer creation {:.2}, uploads {:.2}, other host work {:.2}) + blocked in sync {:.2} ms; GPU busy {:.2} ms",
        ms(wall, N), ms(host, N), ms(p.encode_ns, N), ms(p.alloc_ns, N), ms(p.upload_ns, N), ms(host.saturating_sub(p.encode_ns + p.alloc_ns + p.upload_ns), N), ms(p.wait_ns, N), ms(p.gpu_ns, N));
    eprintln!("metal profile {what}: per step {} kernels, {} command buffers, {} syncs, {} buffer allocations ({} served by the pool), {} uploads ({} bytes)",
        p.dispatches / N as u64, p.commits / N as u64, p.syncs / N as u64, p.allocs / N as u64, p.pool_hits / N as u64, p.uploads / N as u64, p.upload_bytes / N as u64);
    // Per-kernel GPU time (each kernel its own command buffer: inflated totals, true split).
    // Steady-state GPU time per kernel: record one step's dispatches, replay each 10 times
    // back to back.
    nt::synchronize(M).unwrap();
    nt::metal_trace(true);
    train_step(cfg, &mut vars, &mut opt, &tk, &tg, &mk, 30, AuxWeights::default(), 1.0);
    nt::synchronize(M).unwrap();
    let k = nt::metal_replay_trace(10);
    let total: u64 = k.values().map(|v| v.1).sum();
    let mut ks: Vec<_> = k.iter().collect();
    ks.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let top: Vec<String> = ks.iter().take(10).map(|(k, (n, t))| format!("{k} {:.0}% ({n}, {:.2} ms)", 100.0 * *t as f64 / total.max(1) as f64, *t as f64 / 1e6)).collect();
    eprintln!("metal profile {what}: GPU time by kernel (replayed, {:.2} ms per step): {}", total as f64 / 1e6, top.join("; "));
    nt::set_default_device(R).unwrap();
}

#[test]
#[ignore]
fn metal_perf_profile() {
    profile("small (S 4, B 16)", &small(32), 4, 16);
    profile("d512 (4 blocks, S 1, B 8, T 64, V 256)", &DecoderConfig::new(256, 512, 8, 64, 4, 2048), 1, 8);
}

/// The train-step table (the benchmark configuration: 4 blocks, B 8, T 64, V 256).
#[test]
#[ignore]
fn metal_perf_train_speed() {
    for d in [64usize, 256, 512] {
        let cfg = DecoderConfig::new(256, d, 8, 64, 4, 4 * d);
        let (_, vars) = Decoder::init(cfg.clone(), 1, 3).unwrap();
        let (tk, tg, mk) = batch(1, 8, 64, 256, 4);
        let mut row = Vec::new();
        for dev in [R, M] {
            nt::set_default_device(dev).unwrap();
            let mut v = vars_on(&vars, dev);
            let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&v), &v);
            let (tk, tg, mk) = (tk.clone(), tg.clone(), mk.clone().to(dev));
            let mut times: Vec<f64> = (0..8)
                .map(|i| {
                    let t = std::time::Instant::now();
                    train_step(&cfg, &mut v, &mut opt, &tk, &tg, &mk, i, AuxWeights::default(), 1.0);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .skip(2)
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            row.push(times[2]);
            nt::set_default_device(R).unwrap();
        }
        eprintln!("metal train step d {d:>4}: cpu_ref {:8.2} ms  metal {:8.2} ms  ({:.2}x)", row[0], row[1], row[0] / row[1]);
    }
}

/// How long one isolated kernel takes on the GPU (per-kernel mode), for a trivial and a
/// one-hot kernel, back to back and after an idle gap: the per-command-buffer floor that the
/// isolated split includes.
#[test]
#[ignore]
fn metal_perf_isolated_floor() {
    nt::set_default_device(M).unwrap();
    let ids = IntTensor::from_data(rnd_ints(512, 3, 256), [1, 512]);
    let x = Tensor::from_data(vec![1.0; 262144], [512, 512]).to(M);
    nt::metal_per_kernel_timing(true);
    for gap in [0u64, 5] {
        let _ = nt::metal_profile();
        for _ in 0..20 {
            std::thread::sleep(std::time::Duration::from_millis(gap));
            let _ = ids.clone().one_hot_float(256, crate::tensor::DType::F32, M);
            let _ = x.clone().add_scalar(1.0);
        }
        let p = nt::metal_profile();
        for (k, (n, t)) in &p.kernels {
            eprintln!("metal isolated gap {gap} ms: {k} {:.3} ms each ({n})", *t as f64 / 1e6 / *n as f64);
        }
    }
    nt::metal_per_kernel_timing(false);
    nt::set_default_device(R).unwrap();
}

/// A/B of the matmul rule against 64×64 tiles everywhere, interleaved in one process (the
/// machine's load moves GPU clocks, so only interleaved runs compare).
#[test]
#[ignore]
fn metal_perf_matmul_rule_ab() {
    use std::sync::atomic::Ordering;
    let over = &crate::tensor::metal::MATMUL_OVERRIDE;
    for (what, cfg, s, b) in [("small", small(32), 4, 16), ("d512", DecoderConfig::new(256, 512, 8, 64, 4, 2048), 1, 8)] {
        nt::set_default_device(M).unwrap();
        let (_, vars) = Decoder::init(cfg.clone(), s, 3).unwrap();
        let mut vars = vars_on(&vars, M);
        let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&vars), &vars);
        let (tk, tg, mk) = batch(s, b, cfg.context, cfg.vocab, 7);
        let mut t = [Vec::new(), Vec::new()];
        for round in 0..16 {
            let arm = (round % 2) as usize;
            over.store(if arm == 0 { usize::MAX } else { 0 }, Ordering::SeqCst);
            train_step(&cfg, &mut vars, &mut opt, &tk, &tg, &mk, round, AuxWeights::default(), 1.0);
            nt::synchronize(M).unwrap();
            let _ = nt::metal_profile();
            let t0 = std::time::Instant::now();
            train_step(&cfg, &mut vars, &mut opt, &tk, &tg, &mk, round, AuxWeights::default(), 1.0);
            nt::synchronize(M).unwrap();
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            let gpu = nt::metal_profile().gpu_ns as f64 / 1e6;
            if round >= 2 {
                t[arm].push((wall, gpu));
            }
        }
        over.store(usize::MAX, Ordering::SeqCst);
        let med = |v: &mut Vec<(f64, f64)>| {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            v[v.len() / 2]
        };
        let (r, f) = (med(&mut t[0]), med(&mut t[1]));
        eprintln!("metal matmul rule {what}: rule wall {:.2} ms (GPU {:.2}) vs 64x64 everywhere {:.2} ms (GPU {:.2})", r.0, r.1, f.0, f.1);
        nt::set_default_device(R).unwrap();
    }
}

/// Kernels per command buffer (committed without waiting): an interleaved A/B over sizes.
#[test]
#[ignore]
fn metal_perf_flush_ab() {
    use std::sync::atomic::Ordering;
    let sizes = [16usize, 32, 64, 128, 256, 1024];
    for (what, cfg, s, b) in [("small", small(32), 4, 16), ("d512", DecoderConfig::new(256, 512, 8, 64, 4, 2048), 1, 8)] {
        nt::set_default_device(M).unwrap();
        let (_, vars) = Decoder::init(cfg.clone(), s, 3).unwrap();
        let mut vars = vars_on(&vars, M);
        let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&vars), &vars);
        let (tk, tg, mk) = batch(s, b, cfg.context, cfg.vocab, 7);
        let mut t: Vec<Vec<f64>> = vec![Vec::new(); sizes.len()];
        for round in 0..(8 * sizes.len()) {
            let arm = round % sizes.len();
            crate::tensor::metal::FLUSH.store(sizes[arm], Ordering::SeqCst);
            nt::synchronize(M).unwrap();
            let t0 = std::time::Instant::now();
            train_step(&cfg, &mut vars, &mut opt, &tk, &tg, &mk, round as u64, AuxWeights::default(), 1.0);
            nt::synchronize(M).unwrap();
            if round >= sizes.len() {
                t[arm].push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        crate::tensor::metal::FLUSH.store(256, Ordering::SeqCst);
        let row: Vec<String> = sizes.iter().zip(t.iter_mut()).map(|(n, v)| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); format!("{n}: {:.2} ms", v[v.len() / 2]) }).collect();
        eprintln!("metal flush {what}: {}", row.join(" | "));
        nt::set_default_device(R).unwrap();
    }
}
