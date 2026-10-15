//! Tests of sharded data parallelism (`ShardedDataParallel`), a child of the distributed tests.
//! - The layout: every element of every var once, the ranges increasing with the rank.
//! - The reduce-scatter: each rank's pieces equal the all-reduce's total cut into pieces, bit
//!   for bit, at world sizes 1, 2 and 4 (ranks as threads over localhost TCP).
//! - Clipping: the sharded norms equal `grad_norms_per_seed` of the whole sum, and the clipped
//!   pieces the whole clipped sum cut into pieces, bit for bit.
//! - Checkpoints: a sharded checkpoint round-trips byte for byte; consolidated and data-parallel
//!   checkpoints resume into each other; refusals.
//! - The memory each rank holds at every stage.
//! - The gates, in separate processes through the launcher: every stage and partition gives
//!   data parallelism's weights and per-step hashes at world sizes 1, 2 and 4, with absent
//!   gradients, an idle weight and clipping; the consolidated checkpoints are data parallelism's
//!   byte for byte; a rank killed mid-run and a resume from a sharded checkpoint give the
//!   uninterrupted run.

use super::*;
use crate::nn::{Optimizer, Sgd, SgdConfig};

/// The options of `DIST_TEST_ZERO` and `DIST_TEST_PARTITION`.
pub(super) fn shard_options(stage: &str, partition: &str) -> ShardOptions {
    let stage = match stage {
        "1" => ShardStage::OptimizerState,
        "2" => ShardStage::Gradients,
        "3" => ShardStage::Parameters,
        s => panic!("DIST_TEST_ZERO={s}"),
    };
    let partition = match partition {
        "by-var" => Partition::ByVar,
        "flat" | "" => Partition::Flat,
        p => panic!("DIST_TEST_PARTITION={p}"),
    };
    ShardOptions::new(stage).with_partition(partition)
}

/// A maker of the gate's optimizer (its kind and configuration) for a rank's pieces.
pub(super) fn gate_opt_maker(o: &GateOpt) -> Box<dyn FnOnce(&VarMap, ParamGroups) -> GateOpt> {
    match o {
        GateOpt::Adam(a) => {
            let cfg = a.cfg.clone();
            Box::new(move |v, g| GateOpt::Adam(Adam::new(cfg, g, v)))
        }
        GateOpt::Sgd(s) => {
            let cfg = s.cfg.clone();
            Box::new(move |v, g| GateOpt::Sgd(Sgd::new(cfg, g, v)))
        }
    }
}

impl Optimizer for GateOpt {
    fn step(&mut self, vars: &mut VarMap, grads: Vec<Option<Tensor>>, lr_scale: f64) {
        self.opt().step(vars, grads, lr_scale)
    }
}

impl CheckpointOptimizer for GateOpt {
    fn checkpoint_kind(&self) -> &'static str {
        self.ckpt().checkpoint_kind()
    }

    fn checkpoint_state(&self) -> (Vec<(&'static str, Vec<Option<Tensor>>)>, u64) {
        self.ckpt().checkpoint_state()
    }

    fn restore_checkpoint_state(&mut self, sets: Vec<Vec<Option<Tensor>>>, t: u64) -> Result<()> {
        self.ckpt_mut().restore_checkpoint_state(sets, t)
    }
}

const ALL: [(&str, &str); 6] = [("1", "flat"), ("2", "flat"), ("3", "flat"), ("1", "by-var"), ("2", "by-var"), ("3", "by-var")];
/// The configurations the slower GPU gates run: every stage, both partitions.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
const SOME: [(&str, &str); 3] = [("1", "flat"), ("2", "by-var"), ("3", "flat")];

fn zero(g: Gate, stage: &str, partition: &str) -> Gate {
    g.with("DIST_TEST_ZERO", stage).with("DIST_TEST_PARTITION", partition)
}

fn adam_for(vars: &VarMap) -> Adam {
    Adam::new(AdamConfig { lr: 1e-2, ..AdamConfig::default() }, ParamGroups::new(vars), vars)
}

fn adam_maker() -> impl FnOnce(&VarMap, ParamGroups) -> Adam {
    |v, g| Adam::new(AdamConfig { lr: 1e-2, ..AdamConfig::default() }, g, v)
}

fn hash_of(text: &str) -> String {
    crate::hash::sha256_hex(text.as_bytes())
}

// ------------------------------------------------------------------------------ the layout

#[test]
fn the_layout_covers_every_element_once_in_rank_order() {
    let mut vars = VarMap::new();
    for (i, n) in [1usize, 3, 7, 16, 5, 64].into_iter().enumerate() {
        vars.insert(format!("v{i}"), Tensor::zeros([1, n])).unwrap();
    }
    for p in [Partition::Flat, Partition::ByVar] {
        for w in 1..=5 {
            let l = ShardLayout::new(&vars, w, p).unwrap();
            for v in 0..vars.len() {
                let n = vars.tensors()[v].numel();
                let mut at = 0;
                for r in 0..w {
                    let (lo, hi) = l.range(v, r);
                    assert!(lo == at || lo == hi, "{p:?} W {w} var {v} rank {r}: {lo}..{hi} after {at}");
                    if hi > lo {
                        at = hi;
                    }
                }
                assert_eq!(at, n, "{p:?} W {w} var {v}: every element");
                if p == Partition::ByVar {
                    assert!(!l.is_split(v));
                }
            }
            let total: usize = (0..w).map(|r| l.bytes(r)).sum();
            assert_eq!(total, 4 * (1 + 3 + 7 + 16 + 5 + 64));
            if p == Partition::Flat {
                // Flat pieces: each rank within one element per var of an even share.
                let per: Vec<usize> = (0..w).map(|r| l.bytes(r) / 4).collect();
                assert!(per.iter().max().unwrap() - per.iter().min().unwrap() <= vars.len(), "{per:?}");
            }
        }
    }
    // By var: from the largest var, each to the least loaded rank; ties to the lowest rank.
    let l = ShardLayout::new(&vars, 2, Partition::ByVar).unwrap();
    let holders: Vec<usize> = (0..vars.len()).map(|v| (0..2).find(|r| l.range(v, *r).1 > l.range(v, *r).0).unwrap()).collect();
    assert_eq!(holders, vec![1, 1, 1, 1, 1, 0]);
    let mut empty = VarMap::new();
    empty.insert("e", Tensor::zeros([1, 0])).unwrap();
    assert!(ShardLayout::new(&empty, 2, Partition::Flat).unwrap_err().message().contains("no elements"));
}

// -------------------------------------------------------------------------- the reduce-scatter

#[test]
fn the_reduce_scatter_is_the_all_reduce_cut_into_pieces() {
    // 4 global micro-steps as 1 × 4, 2 × 2 and 4 × 1 (with absent gradients): each rank's pieces
    // hold the bits of the all-reduce's total, for flat and by-var pieces, with chunks of 5
    // elements and of 2²⁰.
    let total = {
        let mut g = ProcessGroup::single();
        let mut s = g.grad_sum(&small_vars());
        for j in 0..4 {
            s.add(micro_grads(j)).unwrap();
        }
        g.all_reduce_grads(&mut s).unwrap();
        grads_bits(&s.finish(Reduce::Sum))
    };
    for p in [Partition::Flat, Partition::ByVar] {
        for chunk in [5usize, 1 << 20] {
            for w in [1usize, 2, 4] {
                let k = 4 / w;
                let res = on_ranks(w, opts("scatter").with_chunk_elems(chunk), move |g| {
                    let vars = small_vars();
                    let l = ShardLayout::new(&vars, g.world_size(), p).unwrap();
                    let segs: Vec<Vec<(usize, usize, usize)>> = (0..g.world_size()).map(|r| l.pieces(r).iter().map(|x| (x.var, x.lo, x.hi)).collect()).collect();
                    let mut s = g.grad_sum(&vars);
                    for i in 0..k {
                        s.add(micro_grads(g.rank() * k + i)).unwrap();
                    }
                    let (vals, count) = g.reduce_scatter_grads(&mut s, &segs).unwrap();
                    (l.pieces(g.rank()), vals.into_iter().map(|v| v.map(|v| bits(&v))).collect::<Vec<_>>(), count)
                });
                for (r, x) in res.into_iter().enumerate() {
                    let (pieces, vals, count) = x.unwrap();
                    assert_eq!(count, 4);
                    for (pc, v) in pieces.iter().zip(vals) {
                        let want = total[pc.var].as_ref().map(|t| t[pc.lo..pc.hi].to_vec());
                        assert_eq!(v, want, "{p:?} chunk {chunk} W {w} rank {r} var {}", pc.var);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------- clipping

/// The gate's model with the absent var, 2 steps' worth of state: (DDP's whole summed gradients
/// clipped and their norms, this rank's sharded pieces clipped and their norms), on `w` ranks.
#[test]
fn sharded_clipping_equals_clipping_the_whole_sum() {
    for p in [Partition::Flat, Partition::ByVar] {
        for (w, micro) in [(1usize, 4usize), (2, 4), (3, 3), (4, 4)].into_iter().filter(|(w, _)| may_run(*w, &format!("sharded clipping {p:?} W {w}"))) {
            for max in [0.5f64, 1e9] {
                let res = on_ranks(w, opts("clip"), move |g| {
                    let cfg = gate_cfg();
                    let (vars, _) = gate_state(true);
                    let shard = Shard::new(ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: micro }, g.rank(), g.world_size()).unwrap();
                    let k = shard.micro_steps_per_rank();
                    let mut out = vec![];
                    // Step 1 leaves "branch.w" without a gradient at some micro-steps; step 0 at others.
                    for step in 0..2u64 {
                        let loss = |l: &VarMap, i: usize| gate_micro(&cfg, l, &shard.batch(step, i), step, shard.global_micro(i) as u64, micro, true);
                        let (mut whole, _) = data_parallel_grads(g, &vars, k, Reduce::Mean, loss).unwrap();
                        let wn = crate::nn::clip_grad_norm_per_seed(&mut whole, max);
                        let mut model = ShardedDataParallel::new(g, vars.clone(), ParamGroups::new(&vars), ShardOptions::new(ShardStage::Gradients).with_partition(p), |v, gr| Adam::new(AdamConfig::default(), gr, v)).unwrap();
                        let mut pieces = model.grads(g, k, Reduce::Mean, loss).unwrap();
                        let pn = pieces.clip_grad_norm_per_seed(g, max).unwrap();
                        let wn: Vec<u64> = wn.iter().map(|x| x.to_bits()).collect();
                        let pn: Vec<u64> = pn.iter().map(|x| x.to_bits()).collect();
                        assert_eq!(pn, wn, "norms");
                        for (pc, t) in pieces.pieces().iter().zip(pieces.grads()) {
                            let want = whole[pc.var].as_ref().map(|t| bits(&t.to_vec()[pc.lo..pc.hi]));
                            assert_eq!(t.as_ref().map(|t| bits(&t.to_vec())), want, "var {} piece {}..{}", pc.var, pc.lo, pc.hi);
                            if pc.whole {
                                if let (Some(a), Some(b)) = (t, &whole[pc.var]) {
                                    assert_eq!(a.shape(), b.shape());
                                }
                            }
                        }
                        out.push(wn);
                    }
                    out
                });
                let all: Vec<_> = res.into_iter().map(|x| x.unwrap()).collect();
                assert!(all.iter().all(|x| *x == all[0]), "every rank has the same norms");
                if max < 1.0 {
                    assert!(all[0].iter().flatten().any(|b| f64::from_bits(*b) > max), "{p:?} W {w}: the clip engaged");
                }
            }
        }
    }
    // The fold's start is the neutral element of f64's sum, which the one-process norm uses.
    let x = [3.0f64, -1.5e-300, 7.25];
    let mut a: f64 = std::iter::empty::<f64>().sum();
    for v in x {
        a += v * v;
    }
    assert_eq!(a.to_bits(), x.iter().map(|v| v * v).sum::<f64>().to_bits());
}

// ------------------------------------------------------------- steps, refusals and memory

#[test]
fn sharded_steps_equal_data_parallel_steps_in_one_process() {
    // Distribution off (the single-process group): every stage and partition, 3 steps of 2
    // micro-steps with the absent var, AdamW with decay and the idle var: the same vars and the
    // same consolidated optimizer state as data_parallel_step, bit for bit.
    let cfg = gate_cfg();
    let run_ddp = || {
        let (mut vars, mut adam) = gate_state_with(true, Some(crate::nn::DecayOrder::Torch));
        let mut g = ProcessGroup::single();
        for step in 0..3u64 {
            data_parallel_step(&mut g, &mut vars, &mut adam, 2, Reduce::Mean, 1.0, |l, i| gate_micro(&cfg, l, &[(4 * step as usize + 2 * i) % 24, (4 * step as usize + 2 * i + 1) % 24], step, i as u64, 2, true)).unwrap();
        }
        (state_hash(&vars).unwrap(), adam)
    };
    let (want, adam) = run_ddp();
    let (m, v, t) = adam.state();
    for (stage, part) in ALL {
        let (vars, adam0) = gate_state_with(true, Some(crate::nn::DecayOrder::Torch));
        let acfg = adam0.cfg.clone();
        let mut g = ProcessGroup::single();
        let mut model = ShardedDataParallel::new(&mut g, vars.clone(), ParamGroups::new(&vars), shard_options(stage, part), move |v, gr| Adam::new(acfg, gr, v)).unwrap();
        for step in 0..3u64 {
            model.step(&mut g, 2, Reduce::Mean, 1.0, |l, i| gate_micro(&cfg, l, &[(4 * step as usize + 2 * i) % 24, (4 * step as usize + 2 * i + 1) % 24], step, i as u64, 2, true)).unwrap();
        }
        assert_eq!(state_hash(&model.full_params(&mut g).unwrap()).unwrap(), want, "stage {stage} {part}");
        let (sm, sv, st) = model.optimizer().state();
        assert_eq!(st, t);
        for (a, b) in sm.iter().zip(m).chain(sv.iter().zip(v)) {
            assert_eq!(bits(&a.to_vec()), bits(&b.to_vec()), "stage {stage} {part}: a moment");
        }
    }
}

#[test]
fn sharding_refuses_what_its_pieces_cannot_hold() {
    let res = on_ranks(2, opts("refusals"), |g| {
        let (vars, _) = gate_state(false);
        let name = vars.names()[0].clone();
        let mut lazy = ParamGroups::new(&vars);
        lazy.set_lr_mult(&name, 1.0);
        let mut rates = ParamGroups::new(&vars);
        rates.set_seed_lr(vec![1.0, 0.5]);
        let flat = ShardOptions::new(ShardStage::Parameters);
        let e1 = ShardedDataParallel::new(g, vars.clone(), rates.clone(), flat.clone(), adam_maker()).unwrap_err().message().to_string();
        // By var, every var stays whole: per-seed rates are kept.
        let ok = ShardedDataParallel::new(g, vars.clone(), rates, flat.clone().with_partition(Partition::ByVar), adam_maker()).is_ok();
        // The ranks start from different vars: refused on every rank.
        let (_, other) = Decoder::init(gate_cfg(), GATE_SEEDS, 6).unwrap();
        let mine = if g.rank() == 0 { vars.clone() } else { other };
        let e2 = ShardedDataParallel::new(g, mine, ParamGroups::new(&vars), flat.clone(), adam_maker()).unwrap_err().message().to_string();
        // A var that is not f32.
        let mut f64s = VarMap::new();
        f64s.insert("x", Tensor::from_f64s(vec![1.0, 2.0], vec![2, 1], crate::tensor::DType::F64)).unwrap();
        let e3 = ShardedDataParallel::new(g, f64s.clone(), ParamGroups::new(&f64s), flat, adam_maker()).unwrap_err().message().to_string();
        (e1, ok, e2, e3)
    });
    for r in res {
        let (e1, ok, e2, e3) = r.unwrap();
        assert!(e1.contains("per-seed learning rates need its seed axis"), "{e1}");
        assert!(ok);
        assert!(e2.contains("differ"), "{e2}");
        assert!(e3.contains("f32"), "{e3}");
    }
}

/// Lazy rows, multiplier tensors and per-seed learning rates need a var's shape: each is refused,
/// with its own error, on vars cut into flat pieces at world size 2, on every rank, and accepted
/// by var (every var whole).
#[test]
fn flat_pieces_refuse_lazy_rows_multipliers_and_per_seed_rates() {
    let res = on_ranks(2, opts("flat refusals"), |g| {
        let (vars, _) = gate_state(false);
        // A [S, R, C] var cut into pieces at world size 2.
        let name = vars.iter().find(|(_, t)| t.rank() == 3).map(|(n, _)| n.to_string()).expect("a [S, R, C] var");
        let shape = vars.var(&name).shape().to_vec();
        let mut lazy = ParamGroups::new(&vars);
        lazy.set_lazy_rows(&name);
        let mut mult = ParamGroups::new(&vars);
        mult.set_multiplier(&name, Tensor::ones([shape[0], 1, shape[2]]));
        let mut rates = ParamGroups::new(&vars);
        rates.set_seed_lr(vec![1.0, 0.5]);
        let mut out = vec![];
        for (what, groups) in [("lazy rows", lazy), ("multiplier", mult), ("per-seed rates", rates)] {
            let flat = ShardedDataParallel::new(g, vars.clone(), groups.clone(), ShardOptions::new(ShardStage::Gradients), adam_maker());
            let by_var = ShardedDataParallel::new(g, vars.clone(), groups, ShardOptions::new(ShardStage::Gradients).with_partition(Partition::ByVar), adam_maker());
            out.push((what, flat.map(|_| ()).unwrap_err().message().to_string(), by_var.is_ok()));
        }
        out
    });
    for x in res {
        for (what, e, by_var) in x.unwrap() {
            let want = match what {
                "lazy rows" => "lazy rows need its [S, R, C] shape",
                "multiplier" => "its multiplier tensor broadcasts against its shape",
                _ => "per-seed learning rates need its seed axis",
            };
            assert!(e.contains("is cut into pieces") && e.contains(want) && e.contains("refused"), "{what}: {e}");
            assert!(by_var, "{what}: accepted by var");
        }
    }
}

/// The bytes each rank holds after one step of the gate's model under Adam, at every stage and
/// partition and at world sizes 1, 2 and 4, against data parallelism (every rank: the whole vars,
/// the whole summed gradient and Adam's two whole moments). Printed for the record.
#[test]
fn sharding_divides_the_optimizer_state_gradients_and_vars() {
    let (vars, adam) = gate_state(false);
    let whole: usize = vars.tensors().iter().map(|t| 4 * t.numel()).sum();
    let (sets, _) = adam.checkpoint_state();
    let ddp_state: usize = sets.iter().flat_map(|(_, s)| s.iter().flatten()).map(|t| 4 * t.numel()).sum();
    assert_eq!(ddp_state, 2 * whole);
    eprintln!("memory DDP (every rank): params={whole} grads={whole} optimizer_state={ddp_state}");
    for (stage, part) in ALL {
        for w in [1usize, 2, 4] {
            if !may_run(w, &format!("memory stage {stage} {part} W {w}")) {
                continue;
            }
            let k = 4 / w;
            let res = on_ranks(w, opts("memory"), move |g| {
                let cfg = gate_cfg();
                let (vars, _) = gate_state(false);
                let mut model = ShardedDataParallel::new(g, vars.clone(), ParamGroups::new(&vars), shard_options(stage, part), adam_maker()).unwrap();
                let shard = Shard::new(ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: 4 }, g.rank(), g.world_size()).unwrap();
                model.step(g, k, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &shard.batch(0, i), shard.global_micro(i) as u64)).unwrap();
                model.memory()
            });
            let m: Vec<ShardMemory> = res.into_iter().map(|x| x.unwrap()).collect();
            let sum = |f: fn(&ShardMemory) -> usize| m.iter().map(f).sum::<usize>();
            let max = |f: fn(&ShardMemory) -> usize| m.iter().map(f).max().unwrap();
            eprintln!(
                "memory stage {stage} {part} W {w}: per rank params={:?} grads={:?} optimizer_state={:?} (largest rank {:.3} of DDP's optimizer state)",
                m.iter().map(|x| x.params).collect::<Vec<_>>(),
                m.iter().map(|x| x.grads).collect::<Vec<_>>(),
                m.iter().map(|x| x.optimizer_state).collect::<Vec<_>>(),
                max(|x| x.optimizer_state) as f64 / ddp_state as f64
            );
            assert_eq!(sum(|x| x.optimizer_state), ddp_state, "the optimizer state is split, not copied");
            match stage {
                "1" => assert!(m.iter().all(|x| x.params == whole && x.grads == whole)),
                "2" => assert!(m.iter().all(|x| x.params == whole) && sum(|x| x.grads) == whole),
                _ => assert!(sum(|x| x.params) == whole && sum(|x| x.grads) == whole),
            }
            if part == "flat" {
                assert!(max(|x| x.optimizer_state) <= ddp_state / w + 8 * vars.len(), "flat pieces are an even share");
            }
        }
    }
}

// ----------------------------------------------------------------------------- checkpoints

/// Every file under `dir` (relative path → bytes).
fn files_under(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    fn walk(base: &Path, d: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(base, &p, out);
            } else {
                out.insert(p.strip_prefix(base).unwrap().display().to_string().replace('\\', "/"), std::fs::read(&p).unwrap());
            }
        }
    }
    walk(dir, dir, &mut out);
    out
}

#[test]
fn a_sharded_checkpoint_round_trips_byte_for_byte_and_resumes_exactly() {
    // 2 ranks as threads; for stage 3 flat under Adam and stage 1 by-var under SGD with momentum:
    // 2 steps, a sharded checkpoint, a fresh model restored from it and saved again (every file
    // the same bytes), 2 more steps (the vars of 4 straight steps). Then the refusals: a changed
    // byte, another partition, another world size.
    let _guard = procs();
    for (stage, part, sgd) in [("3", "flat", false), ("1", "by-var", true)] {
        let root = scratch(&format!("shard-roundtrip-{stage}-{part}"));
        let dir_a = root.clone();
        let res = on_ranks(2, opts("roundtrip"), move |g| {
            let cfg = gate_cfg();
            let make = move |v: &VarMap, gr: ParamGroups| if sgd { GateOpt::Sgd(Sgd::new(SgdConfig { lr: 0.05, momentum: 0.9, weight_decay: 1e-3, ..Default::default() }, gr, v)) } else { GateOpt::Adam(Adam::new(AdamConfig { lr: 1e-2, ..AdamConfig::default() }, gr, v)) };
            let shard = Shard::new(ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: 2 }, g.rank(), 2).unwrap();
            let steps = |g: &mut ProcessGroup, m: &mut ShardedDataParallel<GateOpt>, from: u64, to: u64| {
                for step in from..to {
                    m.step(g, 1, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &shard.batch(step, i), step * 2 + shard.global_micro(i) as u64)).unwrap();
                }
            };
            let fresh = |g: &mut ProcessGroup, part: &str| {
                let (vars, _) = gate_state(false);
                ShardedDataParallel::new(g, vars.clone(), ParamGroups::new(&vars), shard_options(stage, part), make)
            };
            let mut straight = fresh(g, part).unwrap();
            steps(g, &mut straight, 0, 4);
            let want = state_hash(&straight.full_params(g).unwrap()).unwrap();
            let mut m = fresh(g, part).unwrap();
            steps(g, &mut m, 0, 2);
            m.save_sharded(g, &dir_a.join("a"), 2, 2).unwrap();
            let mut back = fresh(g, part).unwrap();
            let c = back.load_sharded_latest(g, &dir_a.join("a")).unwrap().expect("a checkpoint");
            assert_eq!((c.meta.step, c.cursor(g.rank())), (2, Some(2)));
            assert_eq!(back.restore_sharded(g, c).unwrap(), 2);
            back.save_sharded(g, &dir_a.join("b"), 2, 2).unwrap();
            steps(g, &mut back, 2, 4);
            let got = state_hash(&back.full_params(g).unwrap()).unwrap();
            // Another partition: refused on every rank.
            let other = if part == "flat" { "by-var" } else { "flat" };
            let e1 = fresh(g, other).unwrap().load_sharded_latest(g, &dir_a.join("a")).unwrap_err().message().to_string();
            (want, got, e1)
        });
        for r in res {
            let (want, got, e1) = r.unwrap();
            assert_eq!(got, want, "stage {stage} {part}: the resumed run");
            assert!(e1.contains("another layout") && e1.contains("resume refused"), "{e1}");
        }
        let (a, b) = (files_under(&root.join("a")), files_under(&root.join("b")));
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        assert!(a.keys().any(|k| k.ends_with("rank-00001-params.json")) && a.keys().any(|k| k.ends_with("meta.json")));
        for (k, x) in &a {
            assert!(*x == b[k], "stage {stage} {part}: {k} differs after the round trip");
        }
        eprintln!("sharded round trip stage {stage} {part}: {} files byte-identical, meta sha256 {}", a.len(), crate::hash::sha256_hex(&a["step-0000000002/meta.json"]));
        // One process: another world size, refused.
        let mut single = ProcessGroup::single();
        let (vars, _) = gate_state(false);
        let one = ShardedDataParallel::new(&mut single, vars.clone(), ParamGroups::new(&vars), shard_options(stage, part), adam_maker()).unwrap();
        let e = one.load_sharded_latest(&mut single, &root.join("a")).unwrap_err();
        assert!(e.message().contains("world size") && e.message().contains("resume refused"), "{e}");
        // A changed byte in rank 1's file: both ranks refuse.
        let f = Checkpoint::folder(&root.join("a"), 2).join("rank-00001-params.json");
        let mut text = std::fs::read_to_string(&f).unwrap();
        let at = text.rfind("\"data\":\"").unwrap() + 8;
        let c = if &text[at..at + 1] == "0" { "1" } else { "0" };
        text.replace_range(at..at + 1, c);
        std::fs::write(&f, text).unwrap();
        let dir_b = root.clone();
        for x in on_ranks(2, opts("roundtrip"), move |g| {
            let (vars, _) = gate_state(false);
            let m = ShardedDataParallel::new(g, vars.clone(), ParamGroups::new(&vars), shard_options(stage, part), adam_maker()).unwrap();
            m.load_sharded_latest(g, &dir_b.join("a")).map(|_| ()).unwrap_err().message().to_string()
        }) {
            let e = x.unwrap();
            assert!(e.contains("sha256"), "{e}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}

#[test]
fn consolidated_and_data_parallel_checkpoints_resume_into_each_other() {
    // 2 ranks: a data-parallel run checkpoints at step 2 and a sharded run (stage 3) resumes it;
    // a sharded run's consolidated checkpoint at step 2 resumes into data parallelism. Both reach
    // the vars of 4 straight data-parallel steps.
    let _guard = procs();
    let root = scratch("consolidated-cross");
    let dir_a = root.clone();
    let res = on_ranks(2, opts("cross"), move |g| {
        let cfg = gate_cfg();
        let shard = Shard::new(ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: 2 }, g.rank(), 2).unwrap();
        let ddp_steps = |g: &mut ProcessGroup, vars: &mut VarMap, adam: &mut Adam, from: u64, to: u64| {
            for step in from..to {
                data_parallel_step(g, vars, adam, 1, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &shard.batch(step, i), step * 2 + shard.global_micro(i) as u64)).unwrap();
            }
        };
        let zero_steps = |g: &mut ProcessGroup, m: &mut ShardedDataParallel<Adam>, from: u64, to: u64| {
            for step in from..to {
                m.step(g, 1, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &shard.batch(step, i), step * 2 + shard.global_micro(i) as u64)).unwrap();
            }
        };
        let (mut vars, _) = gate_state(false);
        let mut adam = adam_for(&vars);
        ddp_steps(g, &mut vars, &mut adam, 0, 2);
        Checkpoint::save(g, &dir_a.join("ddp"), 2, 2, &vars, &adam).unwrap();
        ddp_steps(g, &mut vars, &mut adam, 2, 4);
        let want = state_hash(&vars).unwrap();
        // DDP's checkpoint into a sharded run.
        let (v0, _) = gate_state(false);
        let mut m = ShardedDataParallel::new(g, v0.clone(), ParamGroups::new(&v0), ShardOptions::new(ShardStage::Parameters), adam_maker()).unwrap();
        let c = Checkpoint::load_latest(g, &dir_a.join("ddp")).unwrap().unwrap();
        assert_eq!(m.restore_consolidated(c).unwrap(), 2);
        zero_steps(g, &mut m, 2, 4);
        let a = state_hash(&m.full_params(g).unwrap()).unwrap();
        // A sharded run's consolidated checkpoint into data parallelism.
        let (v0, _) = gate_state(false);
        let mut m = ShardedDataParallel::new(g, v0.clone(), ParamGroups::new(&v0), ShardOptions::new(ShardStage::Gradients).with_partition(Partition::ByVar), adam_maker()).unwrap();
        zero_steps(g, &mut m, 0, 2);
        m.save_consolidated(g, &dir_a.join("zero"), 2, 2).unwrap();
        let (mut vars, _) = gate_state(false);
        let mut adam = adam_for(&vars);
        Checkpoint::load_latest(g, &dir_a.join("zero")).unwrap().unwrap().restore(&mut vars, &mut adam).unwrap();
        ddp_steps(g, &mut vars, &mut adam, 2, 4);
        let b = state_hash(&vars).unwrap();
        (want, a, b)
    });
    for r in res {
        let (want, a, b) = r.unwrap();
        assert_eq!(a, want, "a data-parallel checkpoint resumed by a sharded run");
        assert_eq!(b, want, "a consolidated checkpoint resumed by data parallelism");
    }
    // The two step-2 checkpoints are the same bytes.
    let (d, z) = (files_under(&root.join("ddp")), files_under(&root.join("zero")));
    assert_eq!(d, z, "the consolidated checkpoint is data parallelism's, byte for byte");
    std::fs::remove_dir_all(&root).unwrap();
}

// ------------------------------------------------------------- the gates, in separate processes

/// The memory records of a gate run's ranks.
fn memory_lines(out: &Path, world: usize) -> Vec<String> {
    (0..world).map(|r| std::fs::read_to_string(out.join(format!("rank-{r}.memory"))).unwrap().trim().to_string()).collect()
}

/// The gate: at every (stage, partition) of `configs`, W ranks × k micro-steps give data
/// parallelism's final weights and every per-step hash at the same W and k (2 global
/// micro-steps at W 1 and 2; 4 at W 1, 2 and 4), 12 steps.
fn zero_gate(tag: &str, on: &dyn Fn(Gate) -> Gate, configs: &[(&str, &str)]) {
    let root = scratch(&format!("zero-{tag}"));
    let steps = 12;
    for (micro, worlds) in [(2usize, [1usize, 2].as_slice()), (4, [1, 2, 4].as_slice())] {
        for &w in worlds.iter().filter(|w| may_run(**w, &format!("{tag} zero gate M {micro} W {w}"))) {
            let out = root.join(format!("ddp-m{micro}-w{w}"));
            std::fs::create_dir_all(&out).unwrap();
            on(Gate::new(w, 1, micro, steps)).run(&out);
            let want = results(&out, w);
            assert_eq!(want.1.len(), steps);
            eprintln!("{tag} zero gate M {micro} W {w}: DDP final weights sha256 {}", hash_of(&want.0));
            for &(stage, part) in configs {
                let out = root.join(format!("zero{stage}-{part}-m{micro}-w{w}"));
                std::fs::create_dir_all(&out).unwrap();
                let t = Instant::now();
                on(zero(Gate::new(w, 1, micro, steps), stage, part)).run(&out);
                let got = results(&out, w);
                let record = std::fs::read_to_string(out.join("key.txt")).unwrap();
                assert!(record.contains(&format!("world={w} auth=hmac-sha256 weight_checks={} ok last_weights=", steps + 1)) && record.contains(&format!("sharding=stage-{stage}-")) && record.contains(&format!("partition={part}")), "{record}");
                eprintln!("{tag} zero gate stage {stage} {part} M {micro} W {w}: final weights sha256 {} ({:.1} s); memory {:?}", hash_of(&got.0), t.elapsed().as_secs_f64(), memory_lines(&out, w));
                assert!(got.0 == want.0, "{tag} stage {stage} {part} M {micro} W {w}: the final weights differ from DDP's");
                assert_eq!(got.1, want.1, "{tag} stage {stage} {part} M {micro} W {w}: the per-step hashes");
            }
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn gate_zero_every_stage_and_partition_equals_data_parallelism() {
    let _guard = procs();
    zero_gate("cpu", &|g| g, &ALL);
}

/// B1's absent-gradient and idle-weight gates through sharding: the weights of one process that
/// puts zeros in place of the absent gradients (`want_absent`), and of one process that leaves
/// the idle weight alone, decay included (`want_idle`), at every world size.
fn zero_absent_idle_gate(tag: &str, on: &dyn Fn(Gate) -> Gate, configs: &[(&str, &str)], want_absent: &dyn Fn(&Path, usize, usize) -> String, want_idle: &dyn Fn(&Path, &str, usize, usize) -> String) {
    let root = scratch(&format!("zero-absent-idle-{tag}"));
    let steps = 9;
    for (micro, worlds) in [(2usize, [1usize, 2].as_slice()), (4, [1, 2, 4].as_slice())] {
        let want = want_absent(&root, micro, steps);
        for &(stage, part) in configs {
            for &w in worlds.iter().filter(|w| may_run(**w, &format!("{tag} zero absent gate M {micro} W {w}"))) {
                let out = root.join(format!("absent-{stage}-{part}-m{micro}-w{w}"));
                std::fs::create_dir_all(&out).unwrap();
                on(zero(Gate::new(w, 1, micro, steps), stage, part).with("DIST_TEST_ABSENT", 1)).run(&out);
                let got = hash_of(&results(&out, w).0);
                eprintln!("{tag} zero absent gate stage {stage} {part} M {micro} W {w}: final weights sha256 {got} (one process with zeros {want})");
                assert_eq!(got, want, "{tag} absent stage {stage} {part} M {micro} W {w}");
            }
        }
    }
    let (steps, micro) = (6, 4);
    let init = bits(&gate_state_with(false, Some(decay_order("after"))).0.var("idle.w").to_vec());
    for order in ["after", "torch"] {
        let want = want_idle(&root, order, micro, steps);
        for &(stage, part) in configs {
            for w in [1usize, 2, 4].into_iter().filter(|w| may_run(*w, &format!("{tag} zero idle gate {order} W {w}"))) {
                let out = root.join(format!("idle-{order}-{stage}-{part}-w{w}"));
                std::fs::create_dir_all(&out).unwrap();
                on(zero(Gate::new(w, 1, micro, steps), stage, part).with("DIST_TEST_IDLE", order)).run(&out);
                let (fin, _) = results(&out, w);
                let got = hash_of(&fin);
                eprintln!("{tag} zero idle gate {order} stage {stage} {part} W {w}: final weights sha256 {got} (one process {want})");
                assert_eq!(got, want, "{tag} idle {order} stage {stage} {part} W {w}");
                assert_eq!(bits(&VarMap::from_json(&fin).unwrap().var("idle.w").to_vec()), init, "{tag} {order} stage {stage} {part} W {w}: the idle weight");
            }
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn gate_zero_absent_gradients_and_an_idle_weight_equal_one_process() {
    let _guard = procs();
    zero_absent_idle_gate("cpu", &|g| g, &ALL, &|_, micro, steps| absent_reference(micro, steps as u64), &|_, order, micro, steps| idle_reference(micro, steps as u64, order).0);
}

/// Clipping through sharding: every rank clips its pieces of the summed gradients per seed to
/// norm 1.0 (the global norm, `ShardGrads::clip_grad_norm_per_seed`); the weights and per-step
/// hashes equal data parallelism's with `clip_grad_norm_per_seed` at the same W, and the clip
/// engaged.
fn zero_clip_gate(tag: &str, on: &dyn Fn(Gate) -> Gate, configs: &[(&str, &str)]) {
    let root = scratch(&format!("zero-clip-{tag}"));
    let steps = 12;
    for (micro, worlds) in [(2usize, [1usize, 2].as_slice()), (4, [1, 2, 4].as_slice())] {
        for &w in worlds.iter().filter(|w| may_run(**w, &format!("{tag} zero clip gate M {micro} W {w}"))) {
            let out = root.join(format!("ddp-m{micro}-w{w}"));
            std::fs::create_dir_all(&out).unwrap();
            on(Gate::new(w, 1, micro, steps).with("DIST_TEST_CLIP", "1.0")).run(&out);
            let want = results(&out, w);
            eprintln!("{tag} zero clip gate M {micro} W {w}: DDP final weights sha256 {}", hash_of(&want.0));
            for &(stage, part) in configs {
                let out = root.join(format!("zero{stage}-{part}-m{micro}-w{w}"));
                std::fs::create_dir_all(&out).unwrap();
                on(zero(Gate::new(w, 1, micro, steps), stage, part).with("DIST_TEST_CLIP", "1.0")).run(&out);
                let got = results(&out, w);
                let clipped: usize = std::fs::read_to_string(out.join("clipped.txt")).unwrap().trim().parse().unwrap();
                eprintln!("{tag} zero clip gate stage {stage} {part} M {micro} W {w}: final weights sha256 {} (clipped on {clipped} of {steps} steps)", hash_of(&got.0));
                assert!(clipped > 0, "{tag}: the clip never engaged");
                assert!(got.0 == want.0, "{tag} clip stage {stage} {part} M {micro} W {w}: the final weights differ from DDP's");
                assert_eq!(got.1, want.1, "{tag} clip stage {stage} {part} M {micro} W {w}: the per-step hashes");
            }
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn gate_zero_clipping_equals_data_parallelism() {
    let _guard = procs();
    zero_clip_gate("cpu", &|g| g, &ALL);
}

/// The consolidated checkpoints of a sharded run (every 3 or 4 steps of 12) are data
/// parallelism's checkpoints, every file byte for byte (vars, optimizer state, meta, `latest`),
/// under Adam and under SGD with momentum.
fn zero_consolidated_gate(tag: &str, on: &dyn Fn(Gate) -> Gate, configs: &[(&str, &str)]) {
    let root = scratch(&format!("zero-consolidated-{tag}"));
    let steps = 12;
    for (w, micro, every, opt) in [(2usize, 2usize, 3usize, "adam"), (4, 4, 4, "adam"), (2, 2, 3, "sgd")] {
        if !may_run(w, &format!("{tag} consolidated gate W {w} {opt}")) {
            continue;
        }
        let out = root.join(format!("ddp-w{w}-{opt}"));
        std::fs::create_dir_all(&out).unwrap();
        on(Gate::new(w, 1, micro, steps).with("DIST_TEST_CKPT", every).with("DIST_TEST_OPT", opt)).run(&out);
        let want = files_under(&out.join("ckpt"));
        assert_eq!(want.len(), 1 + (steps / every) * if opt == "sgd" { 3 } else { 4 });
        for &(stage, part) in configs {
            let z = root.join(format!("zero{stage}-{part}-w{w}-{opt}"));
            std::fs::create_dir_all(&z).unwrap();
            on(zero(Gate::new(w, 1, micro, steps), stage, part).with("DIST_TEST_CKPT", every).with("DIST_TEST_OPT", opt)).run(&z);
            let got = files_under(&z.join("ckpt"));
            assert_eq!(got.keys().collect::<Vec<_>>(), want.keys().collect::<Vec<_>>(), "{tag} stage {stage} {part} W {w} {opt}: the files");
            for (k, x) in &want {
                assert!(got[k] == *x, "{tag} stage {stage} {part} W {w} {opt}: {k} differs from DDP's");
            }
            let last = format!("step-{steps:010}/vars.json");
            eprintln!("{tag} consolidated gate stage {stage} {part} W {w} {opt}: {} files byte-identical to DDP's; last vars.json sha256 {}", want.len(), crate::hash::sha256_hex(&got[&last]));
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn gate_zero_consolidated_checkpoints_are_data_parallelisms() {
    let _guard = procs();
    zero_consolidated_gate("cpu", &|g| g, &ALL);
}

/// A rank killed mid-run, then a resume from the last sharded checkpoint, gives the uninterrupted
/// sharded run's weights and per-step hashes (which are data parallelism's); resuming at another
/// world size or with another partition is refused.
fn zero_resume_gate(tag: &str, on: &dyn Fn(Gate) -> Gate, cases: &[(usize, usize, usize, usize, u64, &str, &str, &str)]) {
    let root = scratch(&format!("zero-resume-{tag}"));
    let steps = 12;
    for &(nproc, micro, every, kill_rank, at, stage, part, opt) in cases {
        if !may_run(nproc, &format!("{tag} sharded resume gate W {nproc}")) {
            continue;
        }
        let name = format!("zero{stage}-{part}-w{nproc}-{opt}");
        let ddp = root.join(format!("{name}-ddp"));
        std::fs::create_dir_all(&ddp).unwrap();
        on(Gate::new(nproc, 1, micro, steps).with("DIST_TEST_OPT", opt)).run(&ddp);
        let want = results(&ddp, nproc);
        let whole = root.join(format!("{name}-uninterrupted"));
        std::fs::create_dir_all(&whole).unwrap();
        on(zero(Gate::new(nproc, 1, micro, steps), stage, part).with("DIST_TEST_OPT", opt)).run(&whole);
        let straight = results(&whole, nproc);
        assert!(straight.0 == want.0 && straight.1 == want.1, "{tag} {name}: the uninterrupted sharded run is not DDP's");
        let with = |g: Gate| zero(on(g), stage, part).with("DIST_TEST_OPT", opt).with("DIST_TEST_CKPT_KIND", "sharded");
        let out = killed_and_resumed(&format!("{name}-killed"), &root, nproc, micro, steps, every, kill_rank, at, &with);
        let got = results(&out, nproc);
        let meta = Checkpoint::folder(&out.join("ckpt"), at / every as u64 * every as u64).join("meta.json");
        assert!(std::fs::read_to_string(&meta).unwrap().contains(SHARD_CHECKPOINT_FORMAT));
        eprintln!("{tag} sharded resume {name}: rank {kill_rank} killed at step {at}; final weights sha256 {} (uninterrupted {})", hash_of(&got.0), hash_of(&straight.0));
        assert!(got.0 == straight.0, "{tag} {name}: the resumed run's final weights differ");
        assert_eq!(got.1, straight.1, "{tag} {name}: the per-step hashes");
        // One process: another world size, refused.
        let cfg = with(Gate::new(1, 1, micro, steps)).with("DIST_TEST_RESUME", 1).configs(&out).remove(0);
        assert!(run(&cfg).is_err());
        assert!(logs(&out).contains("resume refused") && logs(&out).contains("world size"), "{}", logs(&out));
        // The same world size, the other partition: refused.
        let other = if part == "flat" { "by-var" } else { "flat" };
        let cfg = zero(on(Gate::new(nproc, 1, micro, steps)), stage, other).with("DIST_TEST_OPT", opt).with("DIST_TEST_CKPT_KIND", "sharded").with("DIST_TEST_RESUME", 1).configs(&out).remove(0);
        assert!(run(&cfg).is_err());
        assert!(logs(&out).contains("another layout"), "{}", logs(&out));
    }
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn gate_zero_kill_and_sharded_resume_equals_the_uninterrupted_run() {
    let _guard = procs();
    zero_resume_gate(
        "cpu",
        &|g| g,
        &[(2, 2, 3, 1, 7, "3", "flat", "adam"), (2, 2, 3, 1, 7, "2", "by-var", "sgd"), (4, 4, 2, 2, 5, "1", "flat", "adam"), (4, 4, 2, 2, 5, "3", "by-var", "adam")],
    );
}

// ------------------------------------------------------------------------------ GPU gates

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn gpu_gate_zero_every_stage_and_partition_equals_data_parallelism() {
    let _guard = procs();
    let kind = gpu_kind();
    zero_gate(kind, &|g| g.on_gpu(kind), &ALL);
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn gpu_gate_zero_absent_gradients_and_an_idle_weight_equal_one_gpu() {
    let _guard = procs();
    let kind = gpu_kind();
    zero_absent_idle_gate(
        kind,
        &|g| g.on_gpu(kind),
        &SOME,
        &|root, micro, steps| gpu_reference(kind, root, &format!("absent:{micro}:{steps}")),
        &|root, order, micro, steps| gpu_reference(kind, root, &format!("idle:{order}:{micro}:{steps}")).split_once('\n').unwrap().0.to_string(),
    );
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn gpu_gate_zero_clipping_equals_data_parallelism() {
    let _guard = procs();
    let kind = gpu_kind();
    zero_clip_gate(kind, &|g| g.on_gpu(kind), &SOME);
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn gpu_gate_zero_consolidated_checkpoints_are_data_parallelisms() {
    let _guard = procs();
    let kind = gpu_kind();
    zero_consolidated_gate(kind, &|g| g.on_gpu(kind), &SOME);
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
#[test]
fn gpu_gate_zero_kill_and_sharded_resume_equals_the_uninterrupted_run() {
    let _guard = procs();
    let kind = gpu_kind();
    zero_resume_gate(kind, &|g| g.on_gpu(kind), &[(2, 2, 3, 1, 7, "3", "flat", "adam"), (2, 2, 3, 1, 7, "2", "by-var", "sgd"), (4, 4, 2, 2, 5, "1", "flat", "adam")]);
}
