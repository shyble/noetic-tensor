//! The reduced-precision product's training gate: a decoder trained on one device with the matmul
//! precision `f32` or `half` (`MatmulPrecision::Half`: bf16 operands on CUDA's tensor cores, f16 on
//! Metal's simdgroup matrices, f32 accumulation), for the run-to-run and restart determinism test
//! and the loss and wall-time comparison against f32.
//!
//! The model and data: a decoder (vocab 16, 4 heads, context 8; width, blocks and MLP width as
//! given), 2 models on the seed axis, AdamW at the given rate, a copy task (`a b c SEP a b c 0 0`,
//! loss on the answer) with `--batch B` fresh items per step (item i drawn from its own stream).
//!
//! It prints, per step, the sha256 of the vars (`step <s> <sha256>`) and the summed loss
//! (`loss <s> <value>`); then the final weights' sha256, the device's key (with the product's part
//! when the precision is `half`) and the training loop's wall time. With `--out DIR` the per-step
//! lines go to `DIR/steps` and `DIR/losses`. `--ckpt-every E` takes a checkpoint (vars and Adam's
//! state, `nn::dist::Checkpoint`) into `--ckpt-dir D` every E steps; `--resume` continues from the
//! latest one; the checkpoint holds the run's key, so a resume under another precision is refused.
//!
//! ```sh
//! cargo build --release --features cuda --examples
//! E=target/release/examples
//! $E/half_gemm_gate --device cuda --precision half --steps 200
//! ```
//!
//! Options (defaults in brackets): `--device cuda|metal|cpu` [cuda]; `--precision f32|half`
//! [half]; `--steps N` [200]; `--width D` [32]; `--blocks L` [2]; `--mlp F` [64]; `--batch B` [2];
//! `--lr X` [0.01]; `--out DIR`; `--ckpt-every E`; `--ckpt-dir D` [DIR/ckpt]; `--resume`;
//! `--no-hash` (timing: no per-step hash, the final weights' hash only).

use noetic::nn::dist::{state_hash, Checkpoint, GroupOptions, ProcessGroup};
use noetic::nn::{masked_cross_entropy, Adam, AdamConfig, Decoder, DecoderConfig, ForwardOptions, Optimizer, ParamGroups, VarMap};
use noetic::rng::derive;
use noetic::tensor::{gpu_platform_key, pin_reference, set_default_device, set_matmul_precision, CpuMode, Device, IntTensor, MatmulPrecision, Tensor};
use rand::Rng;
use std::io::Write;
use std::path::PathBuf;

const SEEDS: usize = 2;

/// Item i: `a b c SEP a b c 0 0` as (tokens, targets, answer mask), from the item's own stream.
fn item(i: usize) -> (Vec<i64>, Vec<i64>, Vec<f32>) {
    let mut rng = derive(5, &format!("item {i}"));
    let a: Vec<i64> = (0..3).map(|_| rng.gen_range(2..16i64)).collect();
    let seq = [a.clone(), vec![1], a, vec![0, 0]].concat();
    let mask = (0..8).map(|t| if (3..6).contains(&t) { 1.0 } else { 0.0 }).collect();
    (seq[..8].to_vec(), seq[1..].to_vec(), mask)
}

/// Items `first..first + n` on the seed axis, on the default device (set before training).
fn batch(first: usize, n: usize) -> (IntTensor, IntTensor, Tensor) {
    let (mut t, mut y, mut m) = (vec![], vec![], vec![]);
    for _ in 0..SEEDS {
        for i in first..first + n {
            let (a, b, c) = item(i);
            t.extend(a);
            y.extend(b);
            m.extend(c);
        }
    }
    let shape = [SEEDS, n, 8];
    (IntTensor::from_data(t, shape), IntTensor::from_data(y, shape), Tensor::from_data(m, shape))
}

struct Args {
    device: String,
    precision: String,
    steps: u64,
    width: usize,
    blocks: usize,
    mlp: usize,
    batch: usize,
    lr: f64,
    out: Option<PathBuf>,
    ckpt_every: u64,
    ckpt_dir: Option<PathBuf>,
    resume: bool,
    no_hash: bool,
}

fn args() -> Result<Args, String> {
    let mut a = Args { device: "cuda".into(), precision: "half".into(), steps: 200, width: 32, blocks: 2, mlp: 64, batch: 2, lr: 1e-2, out: None, ckpt_every: 0, ckpt_dir: None, resume: false, no_hash: false };
    let mut it = std::env::args().skip(1);
    let num = |v: Option<String>, k: &str| -> Result<u64, String> { v.ok_or(format!("{k} needs a value"))?.parse().map_err(|_| format!("{k}: not a number")) };
    while let Some(k) = it.next() {
        match k.as_str() {
            "--device" => a.device = it.next().ok_or("--device needs a value")?,
            "--precision" => a.precision = it.next().ok_or("--precision needs f32 or half")?,
            "--steps" => a.steps = num(it.next(), &k)?,
            "--width" => a.width = num(it.next(), &k)? as usize,
            "--blocks" => a.blocks = num(it.next(), &k)? as usize,
            "--mlp" => a.mlp = num(it.next(), &k)? as usize,
            "--batch" => a.batch = num(it.next(), &k)? as usize,
            "--lr" => a.lr = it.next().ok_or("--lr needs a value")?.parse().map_err(|_| "--lr: not a number")?,
            "--out" => a.out = Some(it.next().ok_or("--out needs a value")?.into()),
            "--ckpt-every" => a.ckpt_every = num(it.next(), &k)?,
            "--ckpt-dir" => a.ckpt_dir = Some(it.next().ok_or("--ckpt-dir needs a value")?.into()),
            "--resume" => a.resume = true,
            "--no-hash" => a.no_hash = true,
            other => return Err(format!("unknown option {other:?} (see the file's header)")),
        }
    }
    if !["f32", "half"].contains(&a.precision.as_str()) || a.batch == 0 {
        return Err("--precision f32|half, --batch at least 1".into());
    }
    Ok(a)
}

fn main() -> noetic::Result<()> {
    let a = match args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("half_gemm_gate: {e}");
            std::process::exit(2);
        }
    };
    let device = match a.device.as_str() {
        "cuda" => Device::Cuda(0),
        "metal" => Device::Metal(0),
        "cpu" => Device::Cpu(CpuMode::Reference),
        d => {
            eprintln!("half_gemm_gate: --device {d}: cuda, metal or cpu");
            std::process::exit(2);
        }
    };
    if device == Device::Cpu(CpuMode::Reference) {
        pin_reference();
    }
    // The precision is set before the key is read.
    let precision = if a.precision == "half" { MatmulPrecision::Half } else { MatmulPrecision::F32 };
    set_matmul_precision(precision);
    let err = |e: std::io::Error| noetic::NnError::Dist(e.to_string());
    let key = match device {
        Device::Cpu(_) => "cpu reference backend".to_string(),
        d => gpu_platform_key(d)?,
    };
    let mut cfg = DecoderConfig::new(16, a.width, 4, 8, a.blocks, a.mlp);
    cfg.dropout = 0.0;
    let (_, init) = Decoder::init(cfg.clone(), SEEDS, 5)?;
    let mut vars = VarMap::new();
    for (n, t) in init.iter() {
        vars.insert(n, t.clone().to(device))?;
    }
    let mut opt = Adam::new(AdamConfig { lr: a.lr, ..AdamConfig::default() }, ParamGroups::new(&vars), &vars);
    set_default_device(device)?;
    // A single-process group on the device, for the checkpoint's key: the device's key carries the
    // product's part, so a checkpoint taken at one precision is refused at the other (the job key
    // leaves the precision out, so that the key is what refuses).
    let mut group = ProcessGroup::from_env(GroupOptions::new(format!("half gemm gate {} width={} blocks={} mlp={} batch={} lr={}", a.device, a.width, a.blocks, a.mlp, a.batch, a.lr)).with_device(device))?;
    let ckpt = a.ckpt_dir.clone().or_else(|| a.out.as_ref().map(|o| o.join("ckpt")));
    let mut start = 0;
    if a.resume {
        let dir = ckpt.as_ref().ok_or_else(|| noetic::NnError::Dist("--resume needs --ckpt-dir or --out".into()))?;
        let c = Checkpoint::load_latest(&mut group, dir)?.ok_or_else(|| noetic::NnError::Dist(format!("no checkpoint under {}", dir.display())))?;
        start = c.cursor(0).unwrap_or(0);
        c.restore(&mut vars, &mut opt)?;
        println!("resumed at step {start}");
    }
    if let Some(o) = &a.out {
        std::fs::create_dir_all(o).map_err(err)?;
    }
    let open = |name: &str| -> noetic::Result<Option<std::fs::File>> {
        Ok(match &a.out {
            Some(o) => Some(std::fs::OpenOptions::new().create(true).append(true).open(o.join(name)).map_err(err)?),
            None => None,
        })
    };
    let (mut steps_log, mut loss_log) = (open("steps")?, open("losses")?);
    println!("half_gemm_gate: {} parameters per model, {SEEDS} models; device {device:?}; precision {}; batch {}; key {key}", vars.count_per_seed(), precision.name(), a.batch);
    let started = std::time::Instant::now();
    for step in start..a.steps {
        let lifted = vars.lifted();
        let (tokens, targets, mask) = batch(step as usize * a.batch, a.batch);
        let model = Decoder::load(cfg.clone(), &lifted)?;
        let out = model.forward_with(&tokens, &ForwardOptions { pad: None, train_step: Some(step) });
        let loss = masked_cross_entropy(out.logits, &targets, &mask).sum();
        let lv = loss.to_vec_f64()[0];
        let grads = loss.backward();
        opt.step(&mut vars, lifted.grads(&grads), 1.0);
        // The per-step hash serialises every var to the host (with --no-hash, for timing, only the
        // final weights are hashed, after the loop's wall time is taken).
        let h = if a.no_hash { String::from("-") } else { state_hash(&vars)? };
        println!("step {} {h}", step + 1);
        println!("loss {} {lv:.9e}", step + 1);
        if let Some(f) = &mut steps_log {
            writeln!(f, "{} {h}", step + 1).map_err(err)?;
        }
        if let Some(f) = &mut loss_log {
            writeln!(f, "loss {} {lv:.9e}", step + 1).map_err(err)?;
        }
        if a.ckpt_every > 0 && (step + 1) % a.ckpt_every == 0 {
            let dir = ckpt.as_ref().ok_or_else(|| noetic::NnError::Dist("--ckpt-every needs --ckpt-dir or --out".into()))?;
            Checkpoint::save(&mut group, dir, step + 1, step + 1, &vars, &opt)?;
        }
    }
    noetic::tensor::synchronize(device).or_else(|e| if matches!(device, Device::Cpu(_)) { Ok(()) } else { Err(e) })?;
    let wall = started.elapsed().as_secs_f64();
    let fin = state_hash(&vars)?;
    println!("final weights sha256 {fin}");
    println!("wall {wall:.3} seconds for {} steps", a.steps - start);
    println!("group key {}", group.key());
    if let Some(o) = &a.out {
        std::fs::write(o.join("final.txt"), format!("{fin}\n{key}\n")).map_err(err)?;
    }
    Ok(())
}
