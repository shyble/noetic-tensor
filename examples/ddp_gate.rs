//! The data-parallel gate as an example binary: the gate's model (a decoder, vocab 16, width 32, 4 heads,
//! context 8, 2 blocks, MLP 64, dropout 0.1 keyed by the global micro-step; 2 models on the seed
//! axis; micro-batches of 2 items from a 24-item copy task), trained data-parallel in
//! deterministic mode on the CPU reference backend or on GPUs, one device per rank from
//! `LOCAL_RANK` (torchrun's `cuda:LOCAL_RANK`), or several ranks per device.
//!
//! Run alone (no rendezvous variables) it is the reference: one process on one device that
//! accumulates the global micro-steps itself. Under the launcher, W ranks with M/W micro-steps
//! each must give the reference's weights at M micro-steps, bit for bit, on every step.
//!
//! Every rank prints the sha256 of the vars after each step (`step <s> <sha256>`, checked equal
//! on every rank), and rank 0 prints the final weights' sha256 and the run's record (key, world
//! size, authentication, weight checks). With `--out DIR`, the per-step hashes go to
//! `DIR/rank-<r>.steps` and rank 0 writes `DIR/final.json` (the saved vars) and `DIR/record.txt`.
//!
//! ```sh
//! cargo build --release --features cuda --examples
//! E=target/release/examples
//! $E/ddp_gate --device cuda --micro 2                                        # the reference: 1 GPU × 2 micro-steps
//! $E/launch --nproc-per-node 2 -- $E/ddp_gate --device cuda --micro 2        # 2 ranks, one GPU each
//! $E/launch --nproc-per-node 2 -- $E/ddp_gate --device cuda --micro 2 --ranks-per-device 2   # 2 ranks sharing GPU 0
//! ```
//!
//! Options (defaults in brackets):
//! - `--device cpu|cuda|metal` [cpu]; `--ranks-per-device K` [1]: rank r of a node trains on GPU
//!   `LOCAL_RANK / K`;
//! - `--micro M` [2]: global micro-steps per optimizer step (M must divide among the ranks);
//!   `--steps N` [12];
//! - `--opt adamw|sgd` [adamw]: Adam at lr 0.01, or SGD at lr 0.05 with momentum 0.9 and weight
//!   decay 0.001; `--clip X`: clip the summed gradients per seed to norm X before each step;
//!   `--absent`: a var only some micro-steps use (its gradient holding −0 and +0); `--idle after|torch`:
//!   a var no micro-step uses, under AdamW with decoupled decay 0.1 in that order;
//! - `--ckpt-every E` (coordinated checkpoints into `--ckpt-dir D`, default `DIR/ckpt`), `--resume`,
//!   `--node-copies` (the first rank of every other node keeps a copy in its own `--ckpt-dir`);
//! - `--pause R:S`: rank R stops before step S and waits to be killed (for kill-and-resume gates);
//! - `--cpu-ranks R,…`: those ranks train on the CPU (a mixed job, which the rendezvous refuses);
//! - `--out DIR`, `--job-key K` [ddp gate];
//! - `--fast`: the fast mode (`DistMode::Fast`: per-rank sums, buckets, a ring all-reduce,
//!   overlap with the backward pass; its own key; not reproducible bit for bit), with
//!   `--bucket-kb N` [25600] and `--no-overlap`;
//! - `--losses`: rank 0 prints each step's mean loss over the global micro-steps
//!   (`loss <step> <value>`; an all-gather of the ranks' losses, after the step's reduction);
//! - `--width D` [32], `--blocks L` [2], `--mlp F` [64]: a larger decoder (for timing); the
//!   defaults are the gate's model. Rank 0 prints the training loop's wall time (`wall <seconds>`);
//! - `--mixed`: mixed precision on a GPU (`MatmulPrecision::Half` for the products, set before the
//!   group is made so that its key names it; f32 vars and optimizer state), with `--loss-scale K`
//!   [0]: a static loss scale of 2^K; a step whose gradients are not finite is skipped on every rank
//!   (`skipped <step>`); rank 0 prints the mixed record;
//! - `--zero S:P`: sharded data parallelism (`ShardedDataParallel`, AdamW only, no clipping) at
//!   stage S (1, 2, 3) with partition P (flat, by-var); its checkpoints are consolidated ones.

use noetic::nn::dist::{check_in_sync, data_parallel_grads, data_parallel_step, mixed_data_parallel_grads, mixed_data_parallel_step, mixed_record, mixed_sharded_step, state_hash, Checkpoint, CheckpointOptimizer, DeviceKind, DistEnv, DistMode, FastOptions, GroupOptions, Partition, ProcessGroup, Reduce, Shard, ShardOptions, ShardSpec, ShardStage, ShardedDataParallel};
use noetic::nn::{clip_grad_norm_per_seed, masked_cross_entropy, Adam, AdamConfig, DecayOrder, Decoder, DecoderConfig, ForwardOptions, Optimizer, ParamGroups, Sgd, SgdConfig, VarMap};
use noetic::nn::LossScale;
use noetic::tensor::{pin_reference, set_default_device, set_matmul_precision, CpuMode, Device, IntTensor, MatmulPrecision, Tensor};
use noetic::rng::derive;
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

fn batch(items: &[usize]) -> (IntTensor, IntTensor, Tensor) {
    let (mut t, mut y, mut m) = (vec![], vec![], vec![]);
    for _ in 0..SEEDS {
        for &i in items {
            let (a, b, c) = item(i);
            t.extend(a);
            y.extend(b);
            m.extend(c);
        }
    }
    let shape = [SEEDS, items.len(), 8];
    (IntTensor::from_data(t, shape), IntTensor::from_data(y, shape), Tensor::from_data(m, shape))
}

fn config(width: usize, blocks: usize, mlp: usize) -> DecoderConfig {
    let mut c = DecoderConfig::new(16, width, 4, 8, blocks, mlp);
    c.dropout = 0.1;
    c.dropout_root = 3;
    c
}

/// Global micro-step j of step `step` (of `micro`): the masked cross-entropy, and with `absent`
/// a branch taken only when `(step + j) % 3 == 1` whose gradient holds −0 and +0.
fn loss(cfg: &DecoderConfig, lifted: &VarMap, items: &[usize], step: u64, j: u64, micro: usize, absent: bool) -> Tensor {
    let (tokens, targets, mask) = batch(items);
    let model = Decoder::load(cfg.clone(), lifted).expect("the vars fit the config");
    let out = model.forward_with(&tokens, &ForwardOptions { pad: None, train_step: Some(step * micro as u64 + j) });
    let l = masked_cross_entropy(out.logits, &targets, &mask).sum();
    if absent && (step + j) % 3 == 1 {
        let c: Vec<f32> = (0..SEEDS).flat_map(|_| [-0.0, 0.5 * (j + 1) as f32, -0.25, 0.0]).collect();
        l + (lifted.var("branch.w").clone() * Tensor::from_data(c, [SEEDS, 4])).sum()
    } else {
        l
    }
}

enum Opt {
    Adam(Adam),
    Sgd(Sgd),
}

impl Opt {
    fn opt(&mut self) -> &mut dyn Optimizer {
        match self {
            Opt::Adam(a) => a,
            Opt::Sgd(s) => s,
        }
    }
    fn ckpt(&self) -> &dyn CheckpointOptimizer {
        match self {
            Opt::Adam(a) => a,
            Opt::Sgd(s) => s,
        }
    }
    fn ckpt_mut(&mut self) -> &mut dyn CheckpointOptimizer {
        match self {
            Opt::Adam(a) => a,
            Opt::Sgd(s) => s,
        }
    }
}

struct Args {
    device: String,
    ranks_per_device: usize,
    micro: usize,
    steps: u64,
    opt: String,
    clip: Option<f64>,
    absent: bool,
    idle: Option<String>,
    ckpt_every: u64,
    ckpt_dir: Option<PathBuf>,
    resume: bool,
    node_copies: bool,
    pause: Option<(usize, u64)>,
    cpu_ranks: Vec<usize>,
    out: Option<PathBuf>,
    job_key: String,
    fast: bool,
    bucket_kb: usize,
    overlap: bool,
    losses: bool,
    width: usize,
    blocks: usize,
    mlp: usize,
    mixed: bool,
    loss_scale: i32,
    zero: Option<(String, String)>,
}

fn args() -> Result<Args, String> {
    let mut a = Args { device: "cpu".into(), ranks_per_device: 1, micro: 2, steps: 12, opt: "adamw".into(), clip: None, absent: false, idle: None, ckpt_every: 0, ckpt_dir: None, resume: false, node_copies: false, pause: None, cpu_ranks: vec![], out: None, job_key: "ddp gate".into(), fast: false, bucket_kb: 25 << 10, overlap: true, losses: false, width: 32, blocks: 2, mlp: 64, mixed: false, loss_scale: 0, zero: None };
    let mut it = std::env::args().skip(1);
    let num = |v: Option<String>, k: &str| -> Result<u64, String> { v.ok_or(format!("{k} needs a value"))?.parse().map_err(|_| format!("{k}: not a number")) };
    while let Some(k) = it.next() {
        match k.as_str() {
            "--device" => a.device = it.next().ok_or("--device needs a value")?,
            "--ranks-per-device" => a.ranks_per_device = num(it.next(), &k)? as usize,
            "--micro" => a.micro = num(it.next(), &k)? as usize,
            "--steps" => a.steps = num(it.next(), &k)?,
            "--opt" => a.opt = it.next().ok_or("--opt needs a value")?,
            "--clip" => a.clip = Some(it.next().ok_or("--clip needs a value")?.parse().map_err(|_| "--clip: not a number")?),
            "--absent" => a.absent = true,
            "--idle" => a.idle = Some(it.next().ok_or("--idle needs after or torch")?),
            "--ckpt-every" => a.ckpt_every = num(it.next(), &k)?,
            "--ckpt-dir" => a.ckpt_dir = Some(it.next().ok_or("--ckpt-dir needs a value")?.into()),
            "--resume" => a.resume = true,
            "--node-copies" => a.node_copies = true,
            "--pause" => {
                let v = it.next().ok_or("--pause needs R:S")?;
                let (r, s) = v.split_once(':').ok_or("--pause needs R:S")?;
                a.pause = Some((r.parse().map_err(|_| "--pause: rank")?, s.parse().map_err(|_| "--pause: step")?));
            }
            "--cpu-ranks" => a.cpu_ranks = it.next().ok_or("--cpu-ranks needs a list")?.split(',').map(|r| r.parse().map_err(|_| "--cpu-ranks: a rank")).collect::<Result<_, _>>()?,
            "--out" => a.out = Some(it.next().ok_or("--out needs a value")?.into()),
            "--job-key" => a.job_key = it.next().ok_or("--job-key needs a value")?,
            "--fast" => a.fast = true,
            "--bucket-kb" => a.bucket_kb = num(it.next(), &k)? as usize,
            "--no-overlap" => a.overlap = false,
            "--losses" => a.losses = true,
            "--width" => a.width = num(it.next(), &k)? as usize,
            "--blocks" => a.blocks = num(it.next(), &k)? as usize,
            "--mlp" => a.mlp = num(it.next(), &k)? as usize,
            "--mixed" => a.mixed = true,
            "--loss-scale" => a.loss_scale = it.next().ok_or("--loss-scale needs a value")?.parse().map_err(|_| "--loss-scale: not a number")?,
            "--zero" => {
                let v = it.next().ok_or("--zero needs S:P")?;
                let (st, p) = v.split_once(':').ok_or("--zero needs S:P (1|2|3:flat|by-var)")?;
                a.zero = Some((st.to_string(), p.to_string()));
            }
            other => return Err(format!("unknown option {other:?} (see the file's header)")),
        }
    }
    if !["adamw", "sgd"].contains(&a.opt.as_str()) || a.idle.as_deref().is_some_and(|i| !["after", "torch"].contains(&i)) || a.micro == 0 || a.ranks_per_device == 0 {
        return Err("--opt adamw|sgd, --idle after|torch, --micro and --ranks-per-device at least 1".into());
    }
    if a.idle.is_some() && a.opt == "sgd" {
        return Err("--idle trains under AdamW (--opt adamw)".into());
    }
    if !a.mixed && a.loss_scale != 0 {
        return Err("--loss-scale belongs to mixed precision (--mixed)".into());
    }
    if a.zero.is_some() && (a.opt != "adamw" || a.clip.is_some() || a.idle.is_some()) {
        return Err("--zero trains under AdamW without --clip or --idle".into());
    }
    if !a.fast && (a.bucket_kb != 25 << 10 || !a.overlap) {
        return Err("--bucket-kb and --no-overlap belong to the fast mode (--fast)".into());
    }
    Ok(a)
}

fn main() -> noetic::Result<()> {
    let a = match args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ddp_gate: {e}");
            std::process::exit(2);
        }
    };
    let env = DistEnv::from_env()?.unwrap_or_else(DistEnv::single);
    let kind = match a.device.as_str() {
        _ if a.cpu_ranks.contains(&env.rank) => DeviceKind::Cpu,
        "cuda" => DeviceKind::Cuda,
        "metal" => DeviceKind::Metal,
        "cpu" => DeviceKind::Cpu,
        d => {
            eprintln!("ddp_gate: --device {d}: cpu, cuda or metal");
            std::process::exit(2);
        }
    };
    let device = env.device(kind, a.ranks_per_device)?;
    if device == Device::Cpu(CpuMode::Reference) {
        pin_reference();
    }
    // The gate's job key, unchanged for the gate's model; a larger model names its size.
    let size = if (a.width, a.blocks, a.mlp) == (32, 2, 64) { String::new() } else { format!(" width={} blocks={} mlp={}", a.width, a.blocks, a.mlp) };
    // The job key, unchanged for the gate's runs; a loss scale and the sharding name themselves.
    // Mixed precision itself is in the group's key (the device's part names the product), so a
    // mixed and an f32 run with the same options share a job key and the group key keeps them
    // apart (a mixed job, or a resume across precisions, is refused by it).
    let mixed_part = if a.loss_scale != 0 { format!(" loss-scale=2^{}", a.loss_scale) } else { String::new() };
    let zero_part = a.zero.as_ref().map_or(String::new(), |(st, p)| format!(" zero={st}:{p}"));
    let mut opts = GroupOptions::new(format!("{} micro={} absent={} idle={:?} opt={} clip={:?}{size}{mixed_part}{zero_part}", a.job_key, a.micro, a.absent, a.idle, a.opt, a.clip)).with_device(device);
    // Mixed precision: the products at Half, set before the group is made (its key names it).
    let scale = LossScale::power_of_two(a.loss_scale)?;
    if a.mixed {
        set_matmul_precision(MatmulPrecision::Half);
    }
    if a.fast {
        opts = opts.with_mode(DistMode::Fast(FastOptions::new().with_bucket_bytes(a.bucket_kb << 10).with_overlap(a.overlap)));
    }
    let mut group = ProcessGroup::from_env(opts)?;
    let (rank, world) = (group.rank(), group.world_size());
    // The vars are made on the CPU reference backend (the same initial bits on every device), then
    // moved to the rank's device, where the optimizer's state is made.
    let cfg = config(a.width, a.blocks, a.mlp);
    let (_, mut init) = Decoder::init(cfg.clone(), SEEDS, 5)?;
    if a.absent {
        init.insert("branch.w", Tensor::from_data((0..SEEDS * 4).map(|i| 0.1 * i as f32 - 0.3).collect(), [SEEDS, 4]))?;
    }
    let adam_cfg = match a.idle.as_deref() {
        Some(order) => {
            init.insert("idle.w", Tensor::from_data((0..SEEDS * 3).map(|i| 0.25 * i as f32 - 0.5).collect(), [SEEDS, 3]))?;
            AdamConfig::adamw(1e-2, 0.1, if order == "torch" { DecayOrder::Torch } else { DecayOrder::AfterStep })
        }
        None => AdamConfig { lr: 1e-2, ..AdamConfig::default() },
    };
    let mut vars = VarMap::new();
    for (n, t) in init.iter() {
        vars.insert(n, t.clone().to(device))?;
    }
    let mut opt = match a.opt.as_str() {
        "sgd" => Opt::Sgd(Sgd::new(SgdConfig { lr: 0.05, momentum: 0.9, weight_decay: 1e-3, ..Default::default() }, ParamGroups::new(&vars), &vars)),
        _ => Opt::Adam(Adam::new(adam_cfg.clone(), ParamGroups::new(&vars), &vars)),
    };
    set_default_device(device)?;
    check_in_sync(&mut group, &vars)?;
    // Sharded data parallelism (--zero): the vars and AdamW's state split among the ranks.
    let mut sdp = match &a.zero {
        None => None,
        Some((st, p)) => {
            let stage = match st.as_str() {
                "1" => ShardStage::OptimizerState,
                "2" => ShardStage::Gradients,
                "3" => ShardStage::Parameters,
                s => return Err(noetic::NnError::Dist(format!("--zero {s}: stage 1, 2 or 3"))),
            };
            let partition = if p == "by-var" { Partition::ByVar } else { Partition::Flat };
            let cfg = adam_cfg.clone();
            Some(ShardedDataParallel::new(&mut group, vars.clone(), ParamGroups::new(&vars), ShardOptions::new(stage).with_partition(partition), move |v, g| Adam::new(cfg, g, v))?)
        }
    };
    // The data: every rank checks every rank's shard hash for this run's steps.
    let spec = ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: a.micro };
    let shard = Shard::new(spec.clone(), rank, world)?;
    let shard_hash = shard.hash(0..a.steps);
    for (q, h) in group.all_gather(shard_hash.as_bytes())?.iter().enumerate() {
        if h.as_slice() != Shard::new(spec.clone(), q, world)?.hash(0..a.steps).as_bytes() {
            return Err(noetic::NnError::Dist(format!("rank {q}'s shard hash differs from this rank's computation of it")));
        }
    }
    let out = a.out.clone();
    if let Some(o) = &out {
        std::fs::create_dir_all(o).map_err(|e| noetic::NnError::Dist(format!("{}: {e}", o.display())))?;
    }
    let ckpt = a.ckpt_dir.clone().or_else(|| out.as_ref().map(|o| o.join("ckpt")));
    if (a.ckpt_every > 0 || a.resume) && ckpt.is_none() {
        return Err(noetic::NnError::Dist("checkpoints need --ckpt-dir or --out".into()));
    }
    let mut start = 0;
    if a.resume {
        let dir = ckpt.as_ref().expect("checked above");
        let c = Checkpoint::load_latest(&mut group, dir)?.ok_or_else(|| noetic::NnError::Dist(format!("no checkpoint under {}", dir.display())))?;
        start = c.cursor(rank).ok_or_else(|| noetic::NnError::Dist("no cursor for this rank".into()))?;
        match sdp.as_mut() {
            None => {
                c.restore(&mut vars, opt.ckpt_mut())?;
            }
            Some(z) => {
                z.restore_consolidated(c)?;
                vars = z.full_params(&mut group)?;
            }
        }
        check_in_sync(&mut group, &vars)?;
        println!("rank {rank}: resumed at step {start}");
    }
    if rank == 0 {
        println!("ddp_gate: {} parameters per model, {SEEDS} models; device {device:?}; {} micro-steps, {} per rank; shard {shard_hash}", vars.count_per_seed(), a.micro, shard.micro_steps_per_rank());
    }
    let mut log = match &out {
        Some(o) => Some(std::fs::OpenOptions::new().create(true).append(true).open(o.join(format!("rank-{rank}.steps"))).map_err(|e| noetic::NnError::Dist(e.to_string()))?),
        None => None,
    };
    let mut clipped = 0;
    let started = std::time::Instant::now();
    for step in start..a.steps {
        if a.pause == Some((rank, step)) {
            println!("rank {rank}: paused before step {step}");
            if let Some(o) = &out {
                let _ = std::fs::write(o.join(format!("paused-{rank}")), step.to_string());
            }
            loop {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        let k = shard.micro_steps_per_rank();
        // Dropout masks from the global micro-step, never the rank's.
        // With --losses, each micro-step's loss is read (a host copy of the scalar; the
        // gradients do not depend on it).
        let mut seen: Vec<f64> = vec![];
        let f = |lifted: &VarMap, i: usize| {
            let l = loss(&cfg, lifted, &shard.batch(step, i), step, shard.global_micro(i) as u64, a.micro, a.absent);
            if a.losses {
                seen.push(l.to_vec_f64()[0]);
            }
            l
        };
        let mut stepped = true;
        match (&mut sdp, a.mixed, a.clip) {
            (Some(z), false, _) => {
                z.step(&mut group, k, Reduce::Mean, 1.0, f)?;
            }
            (Some(z), true, _) => {
                stepped = mixed_sharded_step(z, &mut group, k, Reduce::Mean, 1.0, scale, f)?.stepped;
            }
            (None, false, None) => {
                data_parallel_step(&mut group, &mut vars, opt.opt(), k, Reduce::Mean, 1.0, f)?;
            }
            (None, true, None) => {
                stepped = mixed_data_parallel_step(&mut group, &mut vars, opt.opt(), k, Reduce::Mean, 1.0, scale, f)?.stepped;
            }
            (None, false, Some(max)) => {
                let (mut grads, _) = data_parallel_grads(&mut group, &vars, k, Reduce::Mean, f)?;
                let norms = clip_grad_norm_per_seed(&mut grads, max);
                clipped += norms.iter().any(|n| *n > max) as usize;
                opt.opt().step(&mut vars, grads, 1.0);
            }
            (None, true, Some(max)) => {
                let (mut grads, _, finite) = mixed_data_parallel_grads(&mut group, &vars, k, Reduce::Mean, scale, f)?;
                stepped = finite;
                if finite {
                    let norms = clip_grad_norm_per_seed(&mut grads, max);
                    clipped += norms.iter().any(|n| *n > max) as usize;
                    opt.opt().step(&mut vars, grads, 1.0);
                }
            }
        }
        if !stepped && rank == 0 {
            println!("skipped {} (gradients not finite)", step + 1);
        }
        if let Some(z) = &sdp {
            vars = z.full_params(&mut group)?;
        }
        if a.losses {
            // The global mean over the step's micro-steps, added in rank order on every rank.
            let mine: Vec<u8> = seen.iter().flat_map(|x| x.to_le_bytes()).collect();
            let all = group.all_gather(&mine)?;
            let sum: f64 = all.iter().flat_map(|b| b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))).sum();
            if rank == 0 {
                println!("loss {} {:.9e}", step + 1, sum / a.micro as f64);
            }
        }
        let h = check_in_sync(&mut group, &vars)?;
        println!("step {} {h}", step + 1);
        if let Some(l) = &mut log {
            writeln!(l, "{} {h}", step + 1).map_err(|e| noetic::NnError::Dist(e.to_string()))?;
        }
        if a.ckpt_every > 0 && (step + 1) % a.ckpt_every == 0 {
            match &sdp {
                None => {
                    Checkpoint::save_with(&mut group, ckpt.as_ref().expect("checked above"), step + 1, step + 1, &vars, opt.ckpt(), a.node_copies)?;
                }
                Some(z) => {
                    z.save_consolidated(&mut group, ckpt.as_ref().expect("checked above"), step + 1, step + 1)?;
                }
            }
        }
    }
    let wall = started.elapsed().as_secs_f64();
    if rank == 0 {
        println!("wall {wall:.3} seconds for {} steps", a.steps - start);
        let fin = state_hash(&vars)?;
        println!("final weights sha256 {fin}");
        if a.clip.is_some() {
            println!("clipped on {clipped} of {} steps", a.steps - start);
        }
        // The run's record: the group's (unchanged for the gate's runs), with the sharding and the
        // mixed precision when used.
        let mut record = match &sdp {
            None => group.record(),
            Some(z) => z.record(&group),
        };
        if a.mixed {
            record = format!("{record} {}", mixed_record(&group, scale).trim_start_matches(&group.record()).trim());
        }
        println!("record {record}");
        if let Some(o) = &out {
            vars.save(o.join("final.json"))?;
            std::fs::write(o.join("record.txt"), format!("{record}\nfinal {fin}\nshard {shard_hash}\n")).map_err(|e| noetic::NnError::Dist(e.to_string()))?;
        }
    }
    group.barrier()?;
    Ok(())
}
