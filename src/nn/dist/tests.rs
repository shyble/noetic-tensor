//! Tests of distributed training.
//! - The ordered sum: its first contribution, absent gradients, the mean, and its agreement
//!   with the tensor engine's own additions.
//! - The sampler: the ranks' shares partition the single process's stream; recorded hashes.
//! - The collectives at world sizes 1, 2 and 4 (ranks as threads over localhost TCP): rank-order
//!   exact sums on values whose sum depends on the order; broadcast, all-gather, barrier;
//!   refusals and failures.
//! - The gates, in separate processes through the launcher: data-parallel training at world
//!   sizes 1, 2 and 4 (and two nodes of one process) gives the same weights, bit for bit, as one
//!   process with the same number of accumulation micro-steps; a rank killed mid-run, then a
//!   resume from the last coordinated checkpoint, gives the uninterrupted run's weights.

use super::reduce::{fold, Part};
use crate::error::Result;
use super::*;
use crate::nn::{masked_cross_entropy, train_step, Adam, AdamConfig, AuxWeights, Decoder, DecoderConfig, ForwardOptions, ParamGroups, VarMap};
use crate::tensor::{IntTensor, Tensor};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ------------------------------------------------------------------------------- helpers

fn opts(key: &str) -> GroupOptions {
    GroupOptions { job_key: key.into(), timeout: Duration::from_secs(60), connect_timeout: Duration::from_secs(30), ..GroupOptions::default() }
}

/// Run `f` on `w` ranks, each a thread of this process with its own group over localhost TCP;
/// each rank's result, in rank order.
fn on_ranks<T: Send + 'static>(w: usize, o: GroupOptions, f: impl Fn(&mut ProcessGroup) -> T + Send + Sync + 'static) -> Vec<Result<T>> {
    on_ranks_with(w, move |_| o.clone(), f)
}

fn on_ranks_with<T: Send + 'static>(w: usize, o: impl Fn(usize) -> GroupOptions, f: impl Fn(&mut ProcessGroup) -> T + Send + Sync + 'static) -> Vec<Result<T>> {
    let port = free_port().unwrap();
    let f = Arc::new(f);
    let hs: Vec<_> = (0..w)
        .map(|r| {
            let (f, o) = (f.clone(), o(r));
            std::thread::spawn(move || {
                let env = DistEnv { rank: r, world_size: w, rank_in_node: r, node_size: w, node_rank: 0, master_addr: "127.0.0.1".into(), master_port: port };
                let mut g = ProcessGroup::init(&env, o)?;
                Ok(f(&mut g))
            })
        })
        .collect();
    hs.into_iter().map(|h| h.join().expect("a rank thread")).collect()
}

/// Values whose f32 sum depends on the order of the additions.
fn order_sensitive(rank: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| [0.5f32, 1.0e8, -1.0e8, 3.0e-8, 0.25, -0.0][(rank + i) % 6] * (1.0 + i as f32 / 7.0)).collect()
}

/// The left fold of `xs` in the given order.
fn fold_in(xs: &[Vec<f32>]) -> Vec<f32> {
    let mut acc = xs[0].clone();
    for x in &xs[1..] {
        for (a, b) in acc.iter_mut().zip(x) {
            *a += *b;
        }
    }
    acc
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn small_vars() -> VarMap {
    let mut v = VarMap::new();
    v.insert("a", Tensor::zeros([2, 3])).unwrap();
    v.insert("b", Tensor::zeros([1, 5])).unwrap();
    v.insert("c", Tensor::zeros([3, 2, 2])).unwrap();
    v
}

/// Micro-step j's gradients for `small_vars`: order-sensitive values; var "b" is absent in odd
/// micro-steps and var "c" in all but micro-step 2.
fn micro_grads(j: usize) -> Vec<Option<Tensor>> {
    vec![
        Some(Tensor::from_data(order_sensitive(j, 6), [2, 3])),
        j.is_multiple_of(2).then(|| Tensor::from_data(order_sensitive(j + 1, 5), [1, 5])),
        (j == 2).then(|| Tensor::from_data(order_sensitive(j + 2, 12), [3, 2, 2])),
    ]
}

fn grads_bits(g: &[Option<Tensor>]) -> Vec<Option<Vec<u32>>> {
    g.iter().map(|t| t.as_ref().map(|t| bits(&t.to_vec()))).collect()
}

// ---------------------------------------------------------------------- the ordered sum

#[test]
fn the_ordered_sum_starts_from_the_first_contribution() {
    // From the first contribution, never from zeros: −0 stays −0.
    let mut acc: Part = vec![None, None];
    fold(&mut acc, &[Some(vec![-0.0, 1.0]), None]);
    fold(&mut acc, &[None, Some(vec![-0.0])]);
    assert_eq!(acc[0].as_ref().map(|v| bits(v)), Some(bits(&[-0.0, 1.0])));
    assert_eq!(acc[1].as_ref().map(|v| bits(v)), Some(bits(&[-0.0])));
    // One micro-step passes through bit for bit, NaN payload included, also under the mean.
    let vars = {
        let mut v = VarMap::new();
        v.insert("x", Tensor::zeros([1, 3])).unwrap();
        v
    };
    let odd = f32::from_bits(0x7fc0_1234);
    let mut s = GradSum::new(&vars);
    s.add(vec![Some(Tensor::from_data(vec![-0.0, odd, 3.5], [1, 3]))]).unwrap();
    let out = s.finish(Reduce::Mean);
    assert_eq!(bits(&out[0].as_ref().unwrap().to_vec()), bits(&[-0.0, odd, 3.5]));
    // A var no micro-step touched stays None.
    let mut s = GradSum::new(&vars);
    s.add(vec![None]).unwrap();
    assert!(s.finish(Reduce::Sum)[0].is_none());
}

#[test]
fn accumulation_equals_the_engines_own_additions() {
    // ((x₀ + x₁) + x₂) / 3 through GradSum, against Tensor additions and div_scalar.
    let vars = small_vars();
    let xs: Vec<Tensor> = (0..3).map(|j| Tensor::from_data(order_sensitive(j, 6), [2, 3])).collect();
    let mut s = GradSum::new(&vars);
    for x in &xs {
        s.add(vec![Some(x.clone()), None, None]).unwrap();
    }
    let ours = s.finish(Reduce::Mean);
    let theirs = ((xs[0].clone() + xs[1].clone()) + xs[2].clone()).div_scalar(3.0);
    assert_eq!(bits(&ours[0].as_ref().unwrap().to_vec()), bits(&theirs.to_vec()));
    assert!(ours[1].is_none() && ours[2].is_none());
    // The order matters for these values (so the tests above can see a wrong order).
    let rev = fold_in(&[order_sensitive(2, 6), order_sensitive(1, 6), order_sensitive(0, 6)]);
    assert_ne!(bits(&rev), bits(&fold_in(&[order_sensitive(0, 6), order_sensitive(1, 6), order_sensitive(2, 6)])));
}

#[test]
fn a_deferred_sum_folds_like_an_eager_one() {
    let vars = small_vars();
    let (mut e, mut d) = (GradSum::new(&vars), GradSum::deferred(&vars));
    for j in 0..4 {
        e.add(micro_grads(j)).unwrap();
        d.add(micro_grads(j)).unwrap();
    }
    assert_eq!(grads_bits(&e.finish(Reduce::Mean)), grads_bits(&d.finish(Reduce::Mean)));
}

#[test]
fn the_sum_checks_shapes_and_dtypes() {
    let vars = small_vars();
    let mut s = GradSum::new(&vars);
    assert!(s.add(vec![None]).is_err(), "one slot per var");
    assert!(s.add(vec![Some(Tensor::zeros([3, 2])), None, None]).is_err(), "the var's shape");
    let f64s = Tensor::from_f64s(vec![0.0; 6], vec![2, 3], crate::tensor::DType::F64);
    assert!(s.add(vec![Some(f64s), None, None]).is_err(), "f32 only");
}

// --------------------------------------------------------------------------- the sampler

fn spec(m: usize) -> ShardSpec {
    ShardSpec { seed: 11, items: 10, micro_batch: 3, micro_steps: m }
}

#[test]
fn shards_partition_the_single_process_stream() {
    for m in [2usize, 4] {
        let one = Shard::new(spec(m), 0, 1).unwrap();
        for w in [1usize, 2, 4].into_iter().filter(|w| m % w == 0) {
            let shards: Vec<Shard> = (0..w).map(|r| Shard::new(spec(m), r, w).unwrap()).collect();
            for step in 0..9u64 {
                let want: Vec<Vec<usize>> = (0..m).map(|i| one.batch(step, i)).collect();
                let got: Vec<Vec<usize>> = shards.iter().flat_map(|s| (0..s.micro_steps_per_rank()).map(move |i| s.batch(step, i))).collect();
                assert_eq!(got, want, "M {m}, W {w}, step {step}");
            }
        }
    }
    // Each epoch is a permutation of the items, and epochs differ.
    let s = Shard::new(ShardSpec { seed: 11, items: 10, micro_batch: 5, micro_steps: 2 }, 0, 1).unwrap();
    let (e0, e1): (Vec<usize>, Vec<usize>) = ((0..2).flat_map(|i| s.batch(0, i)).collect(), (0..2).flat_map(|i| s.batch(1, i)).collect());
    let sorted = |v: &[usize]| {
        let mut v = v.to_vec();
        v.sort();
        v
    };
    assert_eq!(sorted(&e0), (0..10).collect::<Vec<_>>());
    assert_eq!(sorted(&e1), (0..10).collect::<Vec<_>>());
    assert_ne!(e0, e1);
    assert!(Shard::new(spec(4), 0, 3).is_err(), "4 micro-steps do not divide among 3 ranks");
    assert!(Shard::new(spec(2), 2, 2).is_err(), "rank 2 of 2");
}

#[test]
fn shard_hashes_are_recorded_per_seed_rank_and_world_size() {
    // Recorded values: a change to the sampler's stream changes them.
    let h = |seed: u64, r: usize, w: usize| Shard::new(ShardSpec { seed, ..spec(4) }, r, w).unwrap().hash(0..6);
    let got = [h(11, 0, 1), h(11, 0, 2), h(11, 1, 2), h(11, 3, 4), h(12, 0, 1)];
    for (i, a) in got.iter().enumerate() {
        for b in &got[i + 1..] {
            assert_ne!(a, b, "the key and the data enter the hash");
        }
    }
    let want = [
        "663f417b002a4d1b5295e0797627e52e328bdd55535e084f1228144877d31f1d",
        "949385655470c11ccbd1468e249c9a05e22dae92793cf145b5d17253e4684794",
        "1986ac87c52854d3d072127861337c88239c4fa1eb19abad51282cee64099818",
        "daf5142389dcd6f3a1da1c5696b148848c8079cfe53634f432f972b601e0e596",
        "169c5506029f4b733c09ff071376d0b52027b373ea1c8ae2260dcb895af3bcf8",
    ];
    assert_eq!(got, want);
}

// ------------------------------------------------------------------------- the environment

#[test]
fn the_environment_is_torchruns() {
    let map = |kv: &[(&str, &str)]| {
        let m: BTreeMap<String, String> = kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k: &str| m.get(k).cloned()
    };
    assert_eq!(DistEnv::from_lookup(map(&[])).unwrap(), None, "distribution off");
    let e = DistEnv::from_lookup(map(&[("RANK", "3"), ("WORLD_SIZE", "4"), ("MASTER_ADDR", "10.0.0.2"), ("MASTER_PORT", "29500"), ("LOCAL_RANK", "1"), ("LOCAL_WORLD_SIZE", "2"), ("GROUP_RANK", "1")]))
        .unwrap()
        .unwrap();
    assert_eq!(e, DistEnv { rank: 3, world_size: 4, rank_in_node: 1, node_size: 2, node_rank: 1, master_addr: "10.0.0.2".into(), master_port: 29500 });
    let back: BTreeMap<String, String> = e.to_vars().into_iter().collect();
    assert_eq!(DistEnv::from_lookup(|k| back.get(k).cloned()).unwrap(), Some(e));
    for bad in [&[("RANK", "0")][..], &[("RANK", "4"), ("WORLD_SIZE", "4"), ("MASTER_ADDR", "h"), ("MASTER_PORT", "1")], &[("RANK", "0"), ("WORLD_SIZE", "2")], &[("RANK", "x"), ("WORLD_SIZE", "2")]] {
        assert!(DistEnv::from_lookup(map(bad)).is_err(), "{bad:?}");
    }
    let lone = DistEnv::from_lookup(map(&[("RANK", "0"), ("WORLD_SIZE", "1")])).unwrap().unwrap();
    assert_eq!(lone.world_size, 1);
}

// ------------------------------------------------------------------------- the collectives

/// What a rank saw: the all-reduce's sum, the broadcasts, the all-gather and the group's key.
type Seen = (Vec<f32>, Vec<Vec<u8>>, Vec<Vec<u8>>, String);

#[test]
fn collectives_are_rank_order_exact_at_world_sizes_1_2_4() {
    for w in [1usize, 2, 4] {
        for chunk in [1usize << 20, 5] {
            let o = GroupOptions { chunk_elems: chunk, ..opts("collectives") };
            let res = on_ranks(w, o, move |g| -> Result<Seen> {
                let r = g.rank();
                g.barrier()?;
                let mut got = vec![];
                for root in 0..g.world_size() {
                    let mut b = if r == root { format!("from {root}").into_bytes() } else { vec![] };
                    g.broadcast(root, &mut b)?;
                    got.push(b);
                }
                let all = g.all_gather(format!("rank {r}").as_bytes())?;
                let sum = g.all_reduce_f32(&order_sensitive(r, 23))?;
                g.barrier()?;
                Ok((sum, got, all, g.key()))
            });
            let want = fold_in(&(0..w).map(|r| order_sensitive(r, 23)).collect::<Vec<_>>());
            for (r, x) in res.into_iter().enumerate() {
                let (sum, got, all, key) = x.and_then(|x| x).unwrap_or_else(|e| panic!("W {w} rank {r}: {e}"));
                assert_eq!(bits(&sum), bits(&want), "W {w} rank {r} chunk {chunk}: the rank-order sum");
                assert_eq!(got, (0..w).map(|q| format!("from {q}").into_bytes()).collect::<Vec<_>>());
                assert_eq!(all, (0..w).map(|q| format!("rank {q}").into_bytes()).collect::<Vec<_>>());
                assert_eq!(key, format!("backend=tcp mode=deterministic order=rank world={w}"));
            }
            if w == 4 {
                // The reverse order gives other bits: the check above can see an order change.
                let rev = fold_in(&(0..w).rev().map(|r| order_sensitive(r, 23)).collect::<Vec<_>>());
                assert_ne!(bits(&rev), bits(&want));
            }
        }
    }
}

#[test]
fn the_gradient_all_reduce_folds_the_global_micro_steps_in_order() {
    // 4 global micro-steps as W 1 × k 4, W 2 × k 2 and W 4 × k 1: the same bits.
    let mut one = GradSum::new(&small_vars());
    for j in 0..4 {
        one.add(micro_grads(j)).unwrap();
    }
    let want = grads_bits(&one.finish(Reduce::Mean));
    for w in [1usize, 2, 4] {
        for chunk in [1usize << 20, 4] {
            let k = 4 / w;
            let o = GroupOptions { chunk_elems: chunk, ..opts("grads") };
            let res = on_ranks(w, o, move |g| -> Result<(usize, Vec<Option<Vec<u32>>>)> {
                let mut s = g.grad_sum(&small_vars());
                for i in 0..k {
                    s.add(micro_grads(g.rank() * k + i))?;
                }
                g.all_reduce_grads(&mut s)?;
                Ok((s.count(), grads_bits(&s.finish(Reduce::Mean))))
            });
            for (r, x) in res.into_iter().enumerate() {
                let (count, got) = x.and_then(|x| x).unwrap_or_else(|e| panic!("W {w} rank {r}: {e}"));
                assert_eq!(count, 4);
                assert_eq!(got, want, "W {w} × k {k}, rank {r}, chunk {chunk}");
            }
        }
    }
}

#[test]
fn the_chunk_plan_covers_every_element_once_in_order() {
    let sizes = [7usize, 0, 3, 12, 1];
    for max in [1usize, 4, 5, 23, 100] {
        let plan = super::group::plan_for_tests(&sizes, max);
        let segs: Vec<(usize, usize, usize)> = plan.iter().flatten().copied().collect();
        let mut want = vec![];
        for (v, &n) in sizes.iter().enumerate() {
            want.extend((0..n).map(|e| (v, e)));
        }
        let got: Vec<(usize, usize)> = segs.iter().flat_map(|&(v, lo, hi)| (lo..hi).map(move |e| (v, e))).collect();
        assert_eq!(got, want, "max {max}");
        assert!(segs.iter().any(|&(v, _, _)| v == 1), "the empty var travels");
        assert!(plan.iter().all(|c| c.iter().map(|&(_, lo, hi)| hi - lo).sum::<usize>() <= max));
    }
}

#[test]
fn a_later_rank_may_not_pre_add_its_micro_steps() {
    let res = on_ranks(2, opts("pre-added"), |g| -> Result<()> {
        let mut s = GradSum::new(&small_vars());
        for j in 0..2 {
            s.add(micro_grads(j))?;
        }
        g.all_reduce_grads(&mut s)
    });
    let e = res[1].as_ref().unwrap().as_ref().unwrap_err();
    assert!(e.message().contains("pre-added"), "{e}");
}

#[test]
fn the_rendezvous_refuses_another_job_key_or_world_size() {
    let res = on_ranks_with(2, |r| opts(if r == 0 { "job a" } else { "job b" }), |_| ());
    for (r, x) in res.iter().enumerate() {
        let e = x.as_ref().unwrap_err();
        assert!(e.message().contains("refused") && e.message().contains("job key"), "rank {r}: {e}");
    }
    // Rank 1 believes the world has 3 ranks.
    let port = free_port().unwrap();
    let env = move |r: usize, w: usize| DistEnv { rank: r, world_size: w, rank_in_node: r, node_size: w, node_rank: 0, master_addr: "127.0.0.1".into(), master_port: port };
    let a = std::thread::spawn(move || ProcessGroup::init(&env(0, 2), opts("k")).map(|_| ()));
    let b = std::thread::spawn(move || ProcessGroup::init(&env(1, 3), opts("k")).map(|_| ()));
    for (r, x) in [a.join().unwrap(), b.join().unwrap()].iter().enumerate() {
        let e = x.as_ref().unwrap_err();
        assert!(e.message().contains("world size"), "rank {r}: {e}");
    }
}

#[test]
fn a_stopped_rank_or_another_collective_fails_the_others() {
    // Rank 2 leaves after the barrier: the others' next collective fails at once (no hang).
    let t = Instant::now();
    let res = on_ranks(3, opts("stopped"), |g| -> Result<()> {
        g.barrier()?;
        if g.rank() == 2 {
            return Ok(());
        }
        g.all_reduce_f32(&[1.0, 2.0]).map(|_| ())
    });
    assert!(res[0].as_ref().unwrap().is_err() || res[1].as_ref().unwrap().is_err(), "a lost rank is an error");
    assert!(t.elapsed() < Duration::from_secs(30), "no hang: {:?}", t.elapsed());
    // Rank 0 broadcasts while rank 1 calls a barrier: both fail with the mismatch.
    let res = on_ranks(2, opts("mismatch"), |g| -> Result<()> {
        if g.rank() == 0 {
            let mut b = vec![1u8];
            g.broadcast(0, &mut b)?;
            g.barrier()
        } else {
            g.barrier()
        }
    });
    let e = res[1].as_ref().unwrap().as_ref().unwrap_err();
    assert!(e.message().contains("different collectives"), "{e}");
}

#[test]
fn the_single_process_group_is_the_identity() {
    let mut g = ProcessGroup::single();
    assert_eq!((g.rank(), g.world_size(), g.is_distributed()), (0, 1, false));
    let x = order_sensitive(0, 9);
    assert_eq!(bits(&g.all_reduce_f32(&x).unwrap()), bits(&x));
    assert_eq!(g.all_gather(b"x").unwrap(), vec![b"x".to_vec()]);
    let mut b = b"y".to_vec();
    g.broadcast(0, &mut b).unwrap();
    assert_eq!(b, b"y");
    g.barrier().unwrap();
}

// ------------------------------------------------------- one process, distribution off

/// The gate's model: vocab 16, width 32, 4 heads, context 8, 2 blocks, MLP width 64, dropout 0.1
/// (so the step's random masks are part of what must match), 2 models on the seed axis.
fn gate_cfg() -> DecoderConfig {
    let mut c = DecoderConfig::new(16, 32, 4, 8, 2, 64);
    c.dropout = 0.1;
    c.dropout_root = 3;
    c
}
const GATE_SEEDS: usize = 2;

/// Item `i` of the gate's data: `a b c SEP a b c 0 0` (tokens, targets, answer mask), symbols
/// drawn from the item's own stream.
fn item(i: usize) -> (Vec<i64>, Vec<i64>, Vec<f32>) {
    use rand::Rng;
    let mut rng = crate::rng::derive(5, &format!("item {i}"));
    let a: Vec<i64> = (0..3).map(|_| rng.gen_range(2..16i64)).collect();
    let seq = [a.clone(), vec![1], a, vec![0, 0]].concat();
    let mask = (0..8).map(|t| if (3..6).contains(&t) { 1.0 } else { 0.0 }).collect();
    (seq[..8].to_vec(), seq[1..].to_vec(), mask)
}

/// The batch of `items` on the seed axis: (tokens, targets, mask), each `[S, B, 8]`.
fn batch(items: &[usize]) -> (IntTensor, IntTensor, Tensor) {
    let (mut t, mut y, mut m) = (vec![], vec![], vec![]);
    for _ in 0..GATE_SEEDS {
        for &i in items {
            let (a, b, c) = item(i);
            t.extend(a);
            y.extend(b);
            m.extend(c);
        }
    }
    let shape = [GATE_SEEDS, items.len(), 8];
    (IntTensor::from_data(t, shape), IntTensor::from_data(y, shape), Tensor::from_data(m, shape))
}

/// The gate's loss for one micro-batch at training step `train_step`.
fn gate_loss(cfg: &DecoderConfig, lifted: &VarMap, items: &[usize], train_step: u64) -> Tensor {
    let (tokens, targets, mask) = batch(items);
    let model = Decoder::load(cfg.clone(), lifted).expect("the vars fit the config");
    let out = model.forward_with(&tokens, &ForwardOptions { pad: None, train_step: Some(train_step) });
    masked_cross_entropy(out.logits, &targets, &mask).sum()
}

#[test]
fn one_process_with_one_micro_step_equals_the_plain_trainer() {
    // data_parallel_step on the single-process group with one micro-step against nn::train_step:
    // the same vars and moments, bit for bit, after 4 steps.
    let cfg = gate_cfg();
    let (_, vars) = Decoder::init(cfg.clone(), GATE_SEEDS, 9).unwrap();
    let (mut a, mut b) = (vars.clone(), vars);
    let mut oa = Adam::new(AdamConfig::default(), ParamGroups::new(&a), &a);
    let mut ob = Adam::new(AdamConfig::default(), ParamGroups::new(&b), &b);
    let mut g = ProcessGroup::single();
    for step in 0..4u64 {
        let items: Vec<usize> = (0..3).map(|q| (step as usize * 3 + q) % 7).collect();
        let (tokens, targets, mask) = batch(&items);
        train_step(&cfg, &mut a, &mut oa, &tokens, &targets, &mask, step, AuxWeights::default(), 1.0);
        let n = data_parallel_step(&mut g, &mut b, &mut ob, 1, Reduce::Mean, 1.0, |lifted, _| gate_loss(&cfg, lifted, &items, step)).unwrap();
        assert_eq!(n, 1);
    }
    assert_eq!(state_hash(&a).unwrap(), state_hash(&b).unwrap());
    let (ma, va, ta) = oa.state();
    let (mb, vb, tb) = ob.state();
    assert_eq!(ta, tb);
    for (x, y) in ma.iter().zip(mb).chain(va.iter().zip(vb)) {
        assert_eq!(bits(&x.to_vec()), bits(&y.to_vec()));
    }
}

#[test]
fn the_gate_can_see_a_changed_order() {
    // The same 4 micro-batches per step, added in another order, give other weights: identity
    // in the gates below is evidence of the order, not of insensitive arithmetic.
    let cfg = gate_cfg();
    let run = |order: [usize; 4]| {
        let (_, mut vars) = Decoder::init(cfg.clone(), GATE_SEEDS, 5).unwrap();
        let mut adam = Adam::new(AdamConfig { lr: 1e-2, ..AdamConfig::default() }, ParamGroups::new(&vars), &vars);
        let mut g = ProcessGroup::single();
        for step in 0..3u64 {
            data_parallel_step(&mut g, &mut vars, &mut adam, 4, Reduce::Mean, 1.0, |l, i| {
                let j = order[i];
                gate_loss(&cfg, l, &[(step as usize * 8 + 2 * j) % 24, (step as usize * 8 + 2 * j + 1) % 24], step * 4 + j as u64)
            })
            .unwrap();
        }
        state_hash(&vars).unwrap()
    };
    assert_eq!(run([0, 1, 2, 3]), run([0, 1, 2, 3]), "run to run");
    assert_ne!(run([0, 1, 2, 3]), run([0, 2, 1, 3]), "another order of the same micro-batches");
}

#[test]
fn a_checkpoint_round_trips_and_refuses_another_world_size() {
    let _guard = procs();
    let dir = scratch("checkpoint-unit");
    let cfg = gate_cfg();
    let (_, mut vars) = Decoder::init(cfg.clone(), GATE_SEEDS, 9).unwrap();
    let mut adam = Adam::new(AdamConfig::default(), ParamGroups::new(&vars), &vars);
    let mut g = ProcessGroup::single();
    data_parallel_step(&mut g, &mut vars, &mut adam, 2, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &[i, i + 1], i as u64)).unwrap();
    Checkpoint::save(&mut g, &dir, 1, 1, &vars, &adam).unwrap();
    let c = Checkpoint::load_latest(&g, &dir).unwrap().expect("a checkpoint");
    assert_eq!((c.meta.step, c.meta.world_size, c.meta.cursors.clone()), (1, 1, vec![1]));
    let (_, mut fresh) = Decoder::init(cfg.clone(), GATE_SEEDS, 10).unwrap();
    let mut fresh_adam = Adam::new(AdamConfig::default(), ParamGroups::new(&fresh), &fresh);
    assert_eq!(c.restore(&mut fresh, &mut fresh_adam).unwrap(), 1);
    assert_eq!(state_hash(&fresh).unwrap(), state_hash(&vars).unwrap());
    assert_eq!(fresh_adam.state().2, adam.state().2);
    // Another world size is another key: refused.
    let meta_path = Checkpoint::folder(&dir, 1).join("meta.json");
    let text = std::fs::read_to_string(&meta_path).unwrap().replace("\"world_size\": 1", "\"world_size\": 2");
    std::fs::write(&meta_path, text).unwrap();
    let e = Checkpoint::load_latest(&g, &dir).unwrap_err();
    assert!(e.message().contains("resume refused"), "{e}");
    // A changed byte in the vars fails the hash.
    let vp = Checkpoint::folder(&dir, 1).join("vars.json");
    let mut text = std::fs::read_to_string(&vp).unwrap();
    let at = text.rfind("\"data\":\"").unwrap() + 8;
    let c = if &text[at..at + 1] == "0" { "1" } else { "0" };
    text.replace_range(at..at + 1, c);
    std::fs::write(&vp, text).unwrap();
    assert!(Checkpoint::load(&Checkpoint::folder(&dir, 1)).unwrap_err().message().contains("sha256"));
    std::fs::remove_dir_all(&dir).unwrap();
}

// -------------------------------------------------------- the gates, in separate processes

/// One multi-process test at a time (the machine's other work comes first).
static PROCS: Mutex<()> = Mutex::new(());

fn procs() -> std::sync::MutexGuard<'static, ()> {
    PROCS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A fresh folder under the crate's target folder.
fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("dist-tests").join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A gate run: `nproc` processes per node on `nnodes` nodes (one launcher per node), `micro`
/// global micro-steps per step, `steps` steps; extra variables for the worker.
struct Gate {
    nproc: usize,
    nnodes: usize,
    micro: usize,
    steps: usize,
    env: Vec<(String, String)>,
}

impl Gate {
    fn new(nproc: usize, nnodes: usize, micro: usize, steps: usize) -> Gate {
        Gate { nproc, nnodes, micro, steps, env: vec![] }
    }

    fn with(mut self, k: &str, v: impl ToString) -> Gate {
        self.env.push((k.into(), v.to_string()));
        self
    }

    fn configs(&self, out: &Path) -> Vec<LaunchConfig> {
        let port = free_port().unwrap();
        let mut env = vec![("DIST_TEST_OUT".to_string(), out.display().to_string()), ("DIST_TEST_MICRO".into(), self.micro.to_string()), ("DIST_TEST_STEPS".into(), self.steps.to_string())];
        env.extend(self.env.iter().cloned());
        (0..self.nnodes)
            .map(|node| LaunchConfig {
                nproc_per_node: self.nproc,
                nnodes: self.nnodes,
                node_rank: node,
                master_addr: "127.0.0.1".into(),
                master_port: port,
                command: std::env::current_exe().unwrap(),
                args: worker_args("body_ddp_worker"),
                env: env.clone(),
                log_dir: Some(out.join("logs")),
            })
            .collect()
    }

    /// Run to the end; panics with the ranks' logs on failure.
    fn run(&self, out: &Path) {
        let jobs: Vec<Job> = self.configs(out).iter().map(|c| spawn(c).unwrap()).collect();
        for j in jobs {
            if let Err(e) = j.wait() {
                panic!("{e}\n{}", logs(out));
            }
        }
    }
}

fn worker_args(name: &str) -> Vec<String> {
    vec![format!("nn::dist::tests::{name}"), "--exact".into(), "--ignored".into(), "--nocapture".into(), "--test-threads=1".into()]
}

fn logs(out: &Path) -> String {
    let mut s = String::new();
    if let Ok(rd) = std::fs::read_dir(out.join("logs")) {
        let mut ps: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        ps.sort();
        for p in ps {
            s.push_str(&format!("--- {}\n{}\n", p.display(), std::fs::read_to_string(&p).unwrap_or_default()));
        }
    }
    s
}

/// The final weights (rank 0's saved vars) and every rank's per-step hashes (step → hash; a
/// step logged twice, before a kill and after the resume, must agree).
fn results(out: &Path, world: usize) -> (String, BTreeMap<u64, String>) {
    let fin = std::fs::read_to_string(out.join("final.json")).unwrap_or_else(|e| panic!("{}: {e}\n{}", out.display(), logs(out)));
    let mut steps = BTreeMap::new();
    for r in 0..world {
        let text = std::fs::read_to_string(out.join(format!("rank-{r}.steps"))).unwrap();
        for l in text.lines() {
            let (s, h) = l.split_once(' ').unwrap();
            let s: u64 = s.parse().unwrap();
            if let Some(prev) = steps.insert(s, h.to_string()) {
                assert_eq!(prev, h, "rank {r}, step {s}: two hashes");
            }
        }
    }
    (fin, steps)
}

/// The worker of the gates (run by them in separate processes; does nothing in a plain test
/// run). Trains the gate's model with `data_parallel_step` on its shard, checks after every
/// step that every rank holds the same vars, logs their hash, takes coordinated checkpoints,
/// and can pause (to be killed) or resume.
#[test]
#[ignore = "run by the gates in separate processes"]
fn body_ddp_worker() {
    let Ok(out) = std::env::var("DIST_TEST_OUT") else { return };
    let out = PathBuf::from(out);
    let num = |k: &str, d: u64| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    let (micro, steps, every) = (num("DIST_TEST_MICRO", 2) as usize, num("DIST_TEST_STEPS", 4), num("DIST_TEST_CKPT", 0));
    let pause = std::env::var("DIST_TEST_PAUSE").ok().map(|v| {
        let (r, s) = v.split_once(':').unwrap();
        (r.parse::<usize>().unwrap(), s.parse::<u64>().unwrap())
    });
    crate::tensor::pin_reference();
    let mut g = ProcessGroup::from_env(GroupOptions { job_key: format!("gate micro={micro}"), ..opts("") }).unwrap();
    let (rank, world) = (g.rank(), g.world_size());
    let cfg = gate_cfg();
    let (_, mut vars) = Decoder::init(cfg.clone(), GATE_SEEDS, 5).unwrap();
    let mut adam = Adam::new(AdamConfig { lr: 1e-2, ..AdamConfig::default() }, ParamGroups::new(&vars), &vars);
    check_in_sync(&mut g, &vars).unwrap();
    // The shards: every rank checks every rank's shard hash for this run's steps.
    let sp = ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: micro };
    let shard = Shard::new(sp.clone(), rank, world).unwrap();
    let mine = shard.hash(0..steps);
    for (q, h) in g.all_gather(mine.as_bytes()).unwrap().iter().enumerate() {
        assert_eq!(h.as_slice(), Shard::new(sp.clone(), q, world).unwrap().hash(0..steps).as_bytes(), "rank {q}'s shard");
    }
    let ckpt = out.join("ckpt");
    let mut start = 0;
    if std::env::var("DIST_TEST_RESUME").is_ok() {
        let c = Checkpoint::load_latest(&g, &ckpt).unwrap_or_else(|e| panic!("{e}")).expect("a checkpoint to resume from");
        assert_eq!(c.meta.cursors, vec![c.meta.step; world], "every rank's cursor");
        start = c.restore(&mut vars, &mut adam).unwrap();
        check_in_sync(&mut g, &vars).unwrap();
        eprintln!("rank {rank}: resumed at step {start}");
    }
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(out.join(format!("rank-{rank}.steps"))).unwrap();
    for step in start..steps {
        if pause == Some((rank, step)) {
            std::fs::write(out.join(format!("paused-{rank}")), step.to_string()).unwrap();
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        let k = shard.micro_steps_per_rank();
        data_parallel_step(&mut g, &mut vars, &mut adam, k, Reduce::Mean, 1.0, |lifted, i| {
            // Dropout masks from the global micro-step, never the rank's.
            let j = shard.global_micro(i) as u64;
            gate_loss(&cfg, lifted, &shard.batch(step, i), step * micro as u64 + j)
        })
        .unwrap();
        let h = check_in_sync(&mut g, &vars).unwrap();
        use std::io::Write;
        writeln!(log, "{} {h}", step + 1).unwrap();
        if every > 0 && (step + 1) % every == 0 {
            Checkpoint::save(&mut g, &ckpt, step + 1, step + 1, &vars, &adam).unwrap();
        }
    }
    if rank == 0 {
        vars.save(out.join("final.json")).unwrap();
        std::fs::write(out.join("key.txt"), format!("{}\n{}\n", g.key(), mine)).unwrap();
    }
    g.barrier().unwrap();
}

/// The launcher's failure worker: rank 1 fails at once, the others wait to be stopped.
#[test]
#[ignore = "run by its gate in separate processes"]
fn body_failing_worker() {
    let Ok(rank) = std::env::var("RANK") else { return };
    if rank == "1" {
        std::process::exit(3);
    }
    std::thread::sleep(Duration::from_secs(120));
}

#[test]
fn the_launcher_stops_the_job_when_a_rank_fails() {
    let _guard = procs();
    let out = scratch("launcher");
    let mut cfg = LaunchConfig::local(3, std::env::current_exe().unwrap()).unwrap();
    cfg.args = worker_args("body_failing_worker");
    cfg.log_dir = Some(out.join("logs"));
    let t = Instant::now();
    let e = run(&cfg).unwrap_err();
    assert!(e.message().contains("rank 1 failed"), "{e}");
    assert!(t.elapsed() < Duration::from_secs(60), "the others were stopped, not waited for: {:?}", t.elapsed());
    std::fs::remove_dir_all(&out).unwrap();
}

/// The gate: W ranks × k micro-steps give the bits of one process × W·k micro-steps.
#[test]
fn gate_world_sizes_1_2_4_equal_one_process_with_accumulation() {
    let _guard = procs();
    let steps = 12;
    let root = scratch("gate");
    // (name, processes per node, nodes, global micro-steps)
    let runs = [
        ("m2-one-process", 1, 1, 2),
        ("m2-one-node-two-processes", 2, 1, 2),
        ("m2-two-nodes-one-process-each", 1, 2, 2),
        ("m4-one-process", 1, 1, 4),
        ("m4-two-processes", 2, 1, 4),
        ("m4-four-processes", 4, 1, 4),
    ];
    let mut got: BTreeMap<&str, (String, BTreeMap<u64, String>)> = BTreeMap::new();
    for (name, nproc, nnodes, micro) in runs {
        let out = root.join(name);
        std::fs::create_dir_all(&out).unwrap();
        let t = Instant::now();
        Gate::new(nproc, nnodes, micro, steps).run(&out);
        let r = results(&out, nproc * nnodes);
        eprintln!("gate {name}: final weights sha256 {} ({:.1} s)", crate::hash::sha256_hex(r.0.as_bytes()), t.elapsed().as_secs_f64());
        got.insert(name, r);
    }
    // Distribution off (no rendezvous variables at all), 2 micro-steps.
    let off = root.join("m2-distribution-off");
    std::fs::create_dir_all(&off).unwrap();
    let st = std::process::Command::new(std::env::current_exe().unwrap())
        .args(worker_args("body_ddp_worker"))
        .env("DIST_TEST_OUT", &off)
        .env("DIST_TEST_MICRO", "2")
        .env("DIST_TEST_STEPS", steps.to_string())
        .env_remove("RANK")
        .env_remove("WORLD_SIZE")
        .output()
        .unwrap();
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let r = results(&off, 1);
    eprintln!("gate m2-distribution-off: final weights sha256 {}", crate::hash::sha256_hex(r.0.as_bytes()));
    got.insert("m2-distribution-off", r);
    for (group, names) in [("m2", ["m2-one-node-two-processes", "m2-two-nodes-one-process-each", "m2-distribution-off"].as_slice()), ("m4", ["m4-two-processes", "m4-four-processes"].as_slice())] {
        let base = &got[format!("{group}-one-process").as_str()];
        assert_eq!(base.1.len(), steps);
        for n in names {
            assert!(got[n].0 == base.0, "{n}: the final weights differ from one process with accumulation");
            assert_eq!(got[n].1, base.1, "{n}: the per-step hashes");
        }
    }
    assert_ne!(got["m2-one-process"].0, got["m4-one-process"].0, "2 and 4 micro-steps train differently");
    std::fs::remove_dir_all(&root).unwrap();
}

/// Run `gate` with `kill_rank` paused at step `at`, kill that rank, check the job stopped, then
/// resume; returns the out folder.
#[allow(clippy::too_many_arguments)]
fn killed_and_resumed(name: &str, root: &Path, nproc: usize, micro: usize, steps: usize, every: usize, kill_rank: usize, at: u64) -> PathBuf {
    let out = root.join(name);
    std::fs::create_dir_all(&out).unwrap();
    let gate = Gate::new(nproc, 1, micro, steps).with("DIST_TEST_CKPT", every).with("DIST_TEST_PAUSE", format!("{kill_rank}:{at}"));
    let cfg = gate.configs(&out).remove(0);
    let mut job = spawn(&cfg).unwrap();
    let marker = out.join(format!("paused-{kill_rank}"));
    let t = Instant::now();
    while !marker.exists() {
        assert!(t.elapsed() < Duration::from_secs(120), "rank {kill_rank} never reached step {at}\n{}", logs(&out));
        std::thread::sleep(Duration::from_millis(20));
    }
    job.kill_rank(kill_rank).unwrap();
    let e = job.wait().unwrap_err();
    assert!(e.message().contains("failed"), "{e}");
    eprintln!("{name}: rank {kill_rank} killed at step {at}: {e}");
    let latest = std::fs::read_to_string(out.join("ckpt").join("latest")).unwrap();
    let want = Checkpoint::folder(Path::new(""), at / every as u64 * every as u64).display().to_string();
    assert_eq!(latest.trim(), want, "the last coordinated checkpoint");
    // The resume: the same job, from the checkpoint.
    let resume = Gate::new(nproc, 1, micro, steps).with("DIST_TEST_CKPT", every).with("DIST_TEST_RESUME", 1);
    std::fs::remove_dir_all(out.join("logs")).unwrap();
    resume.run(&out);
    out
}

/// The gate: a rank killed mid-run, then a resume from the last coordinated checkpoint, gives
/// the uninterrupted run's weights; resuming at another world size is refused.
#[test]
fn gate_kill_and_resume_equals_the_uninterrupted_run() {
    let _guard = procs();
    let root = scratch("resume");
    for (nproc, micro, every, kill_rank, at) in [(2usize, 2usize, 3usize, 1usize, 7u64), (4, 4, 2, 2, 5)] {
        let steps = 12;
        let whole = root.join(format!("w{nproc}-uninterrupted"));
        std::fs::create_dir_all(&whole).unwrap();
        Gate::new(nproc, 1, micro, steps).run(&whole);
        let want = results(&whole, nproc);
        let out = killed_and_resumed(&format!("w{nproc}-killed"), &root, nproc, micro, steps, every, kill_rank, at);
        let got = results(&out, nproc);
        eprintln!("resume W {nproc}: final weights sha256 {} (uninterrupted {})", crate::hash::sha256_hex(got.0.as_bytes()), crate::hash::sha256_hex(want.0.as_bytes()));
        assert!(got.0 == want.0, "W {nproc}: the resumed run's final weights differ from the uninterrupted run's");
        assert_eq!(got.1, want.1, "W {nproc}: the per-step hashes");
        // One process resuming the checkpoint of this world size: refused.
        let refused = Gate::new(1, 1, micro, steps).with("DIST_TEST_RESUME", 1);
        let cfg = refused.configs(&out).remove(0);
        let e = run(&cfg).unwrap_err();
        assert!(e.message().contains("rank 0 failed"), "{e}");
        assert!(logs(&out).contains("resume refused"), "{}", logs(&out));
    }
    std::fs::remove_dir_all(&root).unwrap();
}
