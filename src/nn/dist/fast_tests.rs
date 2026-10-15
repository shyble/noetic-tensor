//! Tests of the fast mode (`DistMode::Fast`), a child of the distributed tests. Ranks run as
//! threads over localhost TCP.
//! - The backward pass's notification of finished gradients: every leaf once, with `backward`'s bits.
//! - The ring all-reduce in buckets: the sum of every rank's micro-steps within f32 rounding of
//!   the exact sum, the same bits on every rank, absent vars kept absent, at world sizes 1–4 and
//!   with buckets smaller than a var; overlap on and off give the same sum.
//! - Sharded training in the fast mode: each rank's pieces of the sum at every stage and partition.
//! - Training: the ranks stay in sync; the loss follows the deterministic mode's within a stated
//!   tolerance (no identity is claimed).
//! - Keys and refusals: the fast key and record, a mixed-mode job refused before step 0, a
//!   checkpoint of one mode refused by the other, the fast CPU backend in the fast mode only.

use super::*;

fn fast(bucket_bytes: usize, overlap: bool) -> DistMode {
    DistMode::Fast(FastOptions::new().with_bucket_bytes(bucket_bytes).with_overlap(overlap))
}

/// Micro-step j's loss on `small_vars`: `Σ_v sum(v · c_v)` over the vars present in
/// `micro_grads(j)`, so that each var's gradient is exactly `micro_grads(j)`'s (absent where it
/// is None).
fn small_loss(lifted: &VarMap, j: usize) -> Tensor {
    let mut loss: Option<Tensor> = None;
    for (v, c) in micro_grads(j).into_iter().enumerate() {
        if let Some(c) = c {
            let term = (lifted.tensors()[v].clone() * c).sum();
            loss = Some(match loss {
                None => term,
                Some(l) => l + term,
            });
        }
    }
    loss.expect("var a is in every micro-step")
}

/// The exact sum (f64) of `micro_grads(j)` over `js`, and the sum of the absolute values, per var
/// (None: absent in every micro-step).
fn exact(js: impl Iterator<Item = usize> + Clone) -> Vec<Option<(Vec<f64>, Vec<f64>)>> {
    (0..3)
        .map(|v| {
            let parts: Vec<Vec<f32>> = js.clone().filter_map(|j| micro_grads(j)[v].as_ref().map(|t| t.to_vec())).collect();
            (!parts.is_empty()).then(|| {
                let n = parts[0].len();
                let s = (0..n).map(|i| parts.iter().map(|p| p[i] as f64).sum()).collect();
                let a = (0..n).map(|i| parts.iter().map(|p| (p[i] as f64).abs()).sum()).collect();
                (s, a)
            })
        })
        .collect()
}

/// `got` (a sum of `terms` f32 values in some order) is within f32 rounding of the exact sum.
fn close(got: f32, want: f64, abs: f64, terms: usize) -> bool {
    (got as f64 - want).abs() <= terms as f64 * f32::EPSILON as f64 * abs + 1e-30
}

#[test]
fn backward_notify_names_every_leaf_once_with_backwards_bits() {
    let cfg = gate_cfg();
    let (vars, _) = gate_state(false);
    let lifted = vars.lifted();
    let loss = gate_loss(&cfg, &lifted, &[0, 1], 3);
    let plain = lifted.grads(&loss.backward());
    let ids: Vec<u64> = lifted.tensors().iter().map(|t| t.leaf_id().unwrap()).collect();
    let mut seen: Vec<(usize, Vec<u32>)> = vec![];
    let notified = loss.backward_notify(&mut |id, g| {
        let v = ids.iter().position(|x| *x == id).expect("a var's leaf");
        seen.push((v, bits(&g.to_vec())));
    });
    // Every var with a gradient is named once, with the gradient `backward` gives it.
    let mut named: Vec<usize> = seen.iter().map(|(v, _)| *v).collect();
    named.sort();
    let with_grad: Vec<usize> = plain.iter().enumerate().filter(|(_, g)| g.is_some()).map(|(v, _)| v).collect();
    assert_eq!(named, with_grad);
    for (v, b) in &seen {
        assert_eq!(Some(b.clone()), plain[*v].as_ref().map(|t| bits(&t.to_vec())), "var {}", vars.names()[*v]);
    }
    assert_eq!(grads_bits(&lifted.grads(&notified)), grads_bits(&plain));
    // The vars are named as the backward pass finishes them, not all at the end: the last
    // vars of the model (the head, the last block) before the first ones (the embedding).
    let first = seen.first().unwrap().0;
    let last = seen.last().unwrap().0;
    assert!(first > last, "first named {}, last named {}", vars.names()[first], vars.names()[last]);
}

#[test]
fn buckets_cover_every_element_once_and_segments_follow_the_target() {
    let sizes = [6usize, 5, 12, 0, 3];
    for bytes in [4usize, 20, 32, 1 << 20] {
        for w in 1..=4usize {
            let plan = fast::plan_segments_for_tests(&sizes, bytes, w, None);
            // Vars in reverse order, each in exactly one bucket.
            let order: Vec<usize> = plan.iter().flat_map(|(vars, _)| vars.clone()).collect();
            assert_eq!(order, vec![4, 3, 2, 1, 0], "bytes {bytes}");
            for (vars, segs) in &plan {
                let len: usize = vars.iter().map(|v| sizes[*v]).sum();
                // Over the cap only when one var is larger than it.
                assert!(4 * len <= bytes.max(4) || vars.iter().filter(|v| sizes[**v] > 0).count() == 1, "bytes {bytes}: bucket {vars:?}");
                assert_eq!(segs.len(), w);
                let flat: Vec<(usize, usize)> = segs.iter().flatten().copied().collect();
                assert_eq!(flat.first().map_or(0, |s| s.0), 0);
                assert!(flat.windows(2).all(|p| p[0].1 == p[1].0) && flat.last().is_none_or(|s| s.1 == len));
            }
        }
    }
    // Scatter: each rank's segment is its pieces within the bucket.
    let pieces: Vec<Vec<(usize, usize, usize)>> = vec![vec![(0, 0, 3), (1, 0, 5), (2, 0, 6)], vec![(0, 3, 6), (2, 6, 12)]];
    let plan = fast::plan_segments_for_tests(&[6, 5, 12], 1 << 20, 2, Some(&pieces));
    assert_eq!(plan.len(), 1);
    // Bucket order: var 2 at 0, var 1 at 12, var 0 at 17.
    assert_eq!(plan[0].1, vec![vec![(17, 20), (12, 17), (0, 6)], vec![(20, 23), (6, 12)]]);
}

#[test]
fn the_fast_all_reduce_is_the_sum_of_every_ranks_micro_steps() {
    // k = 2 micro-steps per rank, global micro-step j = rank·2 + i; var "b" is absent in odd
    // micro-steps and var "c" in all but micro-step 2 (so absent everywhere at W 1).
    for w in 1..=4usize {
        if !may_run(w, "a fast all-reduce on threads") {
            continue;
        }
        let want = exact(0..2 * w);
        let mut by_overlap = vec![];
        for overlap in [true, false] {
            for bytes in [8usize, 20, 1 << 20] {
                let res = on_ranks(w, opts("fast sum").with_mode(fast(bytes, overlap)), move |g| {
                    let vars = small_vars();
                    let r = g.rank();
                    let (grads, count) = data_parallel_grads(g, &vars, 2, Reduce::Sum, |l, i| small_loss(l, r * 2 + i)).unwrap();
                    (grads_bits(&grads), count)
                });
                let res: Vec<_> = res.into_iter().map(|x| x.unwrap()).collect();
                for (r, (gb, count)) in res.iter().enumerate() {
                    assert_eq!(*count, 2 * w);
                    assert_eq!(gb, &res[0].0, "W {w} bucket {bytes}: rank {r}'s sum differs from rank 0's");
                }
                for (v, (got, want)) in res[0].0.iter().zip(&want).enumerate() {
                    match (got, want) {
                        (None, None) => {}
                        (Some(got), Some((s, a))) => {
                            for (i, b) in got.iter().enumerate() {
                                assert!(close(f32::from_bits(*b), s[i], a[i], 2 * w), "W {w} bucket {bytes} var {v}[{i}]: {} against {}", f32::from_bits(*b), s[i]);
                            }
                        }
                        _ => panic!("W {w} bucket {bytes} var {v}: presence {} against {}", got.is_some(), want.is_some()),
                    }
                }
                by_overlap.push((overlap, bytes, res[0].0.clone()));
            }
        }
        // Overlap changes when a bucket is reduced, not what is added.
        for (o, b, x) in &by_overlap {
            let (_, _, y) = by_overlap.iter().find(|(o2, b2, _)| !*o2 == *o && b2 == b).unwrap();
            assert_eq!(x, y, "W {w} bucket {b}: overlap {o} against {}", !o);
        }
    }
}

#[test]
fn the_fast_mean_divides_by_every_ranks_micro_steps() {
    let res = on_ranks(2, opts("fast mean").with_mode(fast(20, true)), |g| {
        let vars = small_vars();
        let r = g.rank();
        let (sum, _) = data_parallel_grads(g, &vars, 2, Reduce::Sum, |l, i| small_loss(l, r * 2 + i)).unwrap();
        let (mean, count) = data_parallel_grads(g, &vars, 2, Reduce::Mean, |l, i| small_loss(l, r * 2 + i)).unwrap();
        (grads_bits(&sum), grads_bits(&mean), count)
    });
    for x in res {
        let (sum, mean, count) = x.unwrap();
        assert_eq!(count, 4);
        for (s, m) in sum.iter().zip(&mean) {
            assert_eq!(s.as_ref().map(|s| s.iter().map(|b| (f32::from_bits(*b) / 4.0).to_bits()).collect::<Vec<_>>()), m.clone());
        }
    }
}

#[test]
fn fast_sharded_pieces_are_the_sum_cut_into_pieces() {
    for (stage, partition) in [("1", "flat"), ("2", "flat"), ("3", "flat"), ("1", "by-var"), ("2", "by-var"), ("3", "by-var")] {
        for w in [2usize, 3] {
            if !may_run(w, "fast sharded pieces on threads") {
                continue;
            }
            let want = exact(0..2 * w);
            let o = zero_tests::shard_options(stage, partition);
            let res = on_ranks(w, opts("fast shard").with_mode(fast(20, true)), move |g| {
                let vars = small_vars();
                let r = g.rank();
                let mut sdp = ShardedDataParallel::new(g, vars, ParamGroups::new(&small_vars()), o.clone(), |v, gr| Adam::new(AdamConfig::default(), gr, v)).unwrap();
                let grads = sdp.grads(g, 2, Reduce::Sum, |l, i| small_loss(l, r * 2 + i)).unwrap();
                (grads.pieces().to_vec(), grads.grads().iter().map(|t| t.as_ref().map(|t| t.to_vec())).collect::<Vec<_>>(), grads.count())
            });
            let mut covered: Vec<Vec<u32>> = want.iter().map(|x| x.as_ref().map_or(vec![], |(s, _)| vec![0; s.len()])).collect();
            for (r, x) in res.into_iter().enumerate() {
                let (pieces, vals, count) = x.unwrap();
                assert_eq!(count, 2 * w);
                for (p, v) in pieces.iter().zip(vals) {
                    match (&v, &want[p.var]) {
                        (None, None) => {}
                        (Some(v), Some((s, a))) => {
                            for (i, x) in v.iter().enumerate() {
                                assert!(close(*x, s[p.lo + i], a[p.lo + i], 2 * w), "stage {stage} {partition} W {w} rank {r} var {}[{}]", p.var, p.lo + i);
                                covered[p.var][p.lo + i] += 1;
                            }
                        }
                        _ => panic!("stage {stage} {partition} W {w} rank {r} var {}: presence", p.var),
                    }
                }
            }
            assert!(covered.iter().flatten().all(|c| *c == 1), "stage {stage} {partition} W {w}: every element on one rank");
        }
    }
}

/// The gate's model trained `steps` steps at W `w` (threads), 2 micro-steps per rank, in `mode`:
/// rank 0's per-step mean losses (over the global micro-steps), the final vars' values, and each
/// rank's final hash.
fn train(w: usize, steps: u64, mode: DistMode, sharded: Option<(&'static str, &'static str)>) -> (Vec<f64>, Vec<f32>, Vec<String>) {
    let res = on_ranks(w, opts("fast train").with_mode(mode), move |g| {
        let cfg = gate_cfg();
        let (vars, adam) = gate_state(false);
        let micro = 2 * g.world_size();
        let shard = Shard::new(ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: micro }, g.rank(), g.world_size()).unwrap();
        let mut losses = vec![];
        let mut run = |g: &mut ProcessGroup, step: u64, f: &mut dyn FnMut(&mut ProcessGroup, &mut dyn FnMut(&VarMap, usize) -> Tensor)| {
            let mut seen = vec![];
            let mut micro_loss = |l: &VarMap, i: usize| {
                let t = gate_micro(&cfg, l, &shard.batch(step, i), step, shard.global_micro(i) as u64, micro, false);
                seen.push(t.to_vec_f64()[0]);
                t
            };
            f(g, &mut micro_loss);
            let all = g.all_gather(&seen.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            losses.push(all.iter().flat_map(|b| b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()))).sum::<f64>() / micro as f64);
        };
        let vars = match sharded {
            None => {
                let (mut vars, mut adam) = (vars, adam);
                for step in 0..steps {
                    run(g, step, &mut |g, f| {
                        data_parallel_step(g, &mut vars, &mut adam, 2, Reduce::Mean, 1.0, f).unwrap();
                    });
                    check_in_sync(g, &vars).unwrap();
                }
                vars
            }
            Some((stage, partition)) => {
                drop(adam);
                let groups = ParamGroups::new(&vars);
                let mut sdp = ShardedDataParallel::new(g, vars, groups, zero_tests::shard_options(stage, partition), |v, gr| Adam::new(AdamConfig { lr: 1e-2, ..AdamConfig::default() }, gr, v)).unwrap();
                for step in 0..steps {
                    run(g, step, &mut |g, f| {
                        sdp.step(g, 2, Reduce::Mean, 1.0, f).unwrap();
                    });
                    let full = sdp.full_params(g).unwrap();
                    check_in_sync(g, &full).unwrap();
                }
                sdp.full_params(g).unwrap()
            }
        };
        let values: Vec<f32> = vars.tensors().iter().flat_map(|t| t.to_vec()).collect();
        (losses, values, state_hash(&vars).unwrap())
    });
    let res: Vec<_> = res.into_iter().map(|x| x.unwrap()).collect();
    let hashes = res.iter().map(|r| r.2.clone()).collect();
    (res[0].0.clone(), res[0].1.clone(), hashes)
}

#[test]
fn fast_training_stays_in_sync_and_follows_the_deterministic_run() {
    let _guard = procs();
    // The gate's model, W 2, 2 micro-steps per rank, 6 steps; buckets of 16 KiB (several per
    // step) and of 25 MiB (one).
    let (det_loss, det_vals, det_hashes) = train(2, 6, DistMode::Deterministic, None);
    assert!(det_hashes.iter().all(|h| *h == det_hashes[0]));
    for bytes in [16 << 10, 25 << 20] {
        let (loss, vals, hashes) = train(2, 6, fast(bytes, true), None);
        assert!(hashes.iter().all(|h| *h == hashes[0]), "the ranks' vars differ");
        let dl = loss.iter().zip(&det_loss).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let dw = vals.iter().zip(&det_vals).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("fast mode, buckets of {bytes} bytes, W 2, 6 steps: largest loss difference {dl:.3e}, largest weight difference {dw:.3e}");
        assert!(dl < 1e-3 && dw < 1e-3, "loss {dl}, weights {dw}");
        assert!(loss.last().unwrap() < loss.first().unwrap(), "the loss falls: {loss:?}");
    }
    // Sharded, stage 3 flat and stage 2 by var (6 steps).
    for cfg in [("3", "flat"), ("2", "by-var")] {
        let (loss, vals, hashes) = train(2, 6, fast(16 << 10, true), Some(cfg));
        assert!(hashes.iter().all(|h| *h == hashes[0]), "the ranks' vars differ");
        let dl = loss.iter().zip(&det_loss).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let dw = vals.iter().zip(&det_vals).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("fast mode sharded {cfg:?}, W 2, 6 steps: largest loss difference {dl:.3e}, largest weight difference {dw:.3e}");
        assert!(dl < 1e-3 && dw < 1e-3, "loss {dl}, weights {dw}");
    }
}

#[test]
fn the_fast_mode_has_its_own_key_and_record() {
    let res = on_ranks(2, opts("keys").with_mode(fast(1 << 20, true)), |g| (g.key(), g.record(), g.require_deterministic("a byte-for-byte gate").map_err(|e| e.to_string())));
    let det = on_ranks(2, opts("keys"), |g| (g.key(), g.record(), g.require_deterministic("a byte-for-byte gate").map_err(|e| e.to_string())));
    for (f, d) in res.into_iter().zip(det) {
        let (fk, fr, fq) = f.unwrap();
        let (dk, dr, dq) = d.unwrap();
        assert!(fk.starts_with("mode=fast order=ring world=2 bucket=1048576 overlap=on backend=tcp platform="), "{fk}");
        assert!(dk.starts_with("mode=deterministic order=flat-rank backend=tcp platform="), "{dk}");
        assert_ne!(fk, dk);
        assert_eq!(fk.split_once(" platform=").unwrap().1, dk.split_once(" platform=").unwrap().1);
        assert!(fr.starts_with(&fk) && fr.ends_with(" (fast mode, not reproducible bit for bit)"), "{fr}");
        assert!(dr.starts_with(&dk) && !dr.contains("fast"), "{dr}");
        assert!(fq.unwrap_err().contains("needs the deterministic mode"));
        assert!(dq.is_ok());
    }
    // The world size and the buckets are in the fast key.
    let k1 = on_ranks(1, opts("keys").with_mode(fast(1 << 20, false)), |g| g.key()).pop().unwrap().unwrap();
    assert!(k1.starts_with("mode=fast order=ring world=1 bucket=1048576 overlap=off "), "{k1}");
}

#[test]
fn a_mixed_mode_job_is_refused_before_step_0() {
    let cases: Vec<(DistMode, DistMode, &str)> = vec![
        (DistMode::Deterministic, fast(1 << 20, true), "a mixed-mode job"),
        (fast(1 << 20, true), DistMode::Deterministic, "a mixed-mode job"),
        (fast(1 << 20, true), fast(1 << 10, true), "a mixed-mode job"),
        (fast(1 << 20, true), fast(1 << 20, false), "a mixed-mode job"),
    ];
    for (first, second, why) in cases {
        let res = on_ranks_with(2, move |r| opts("mixed").with_mode(if r == 0 { first.clone() } else { second.clone() }), |_| ());
        for (r, x) in res.into_iter().enumerate() {
            let e = x.expect_err("refused");
            assert!(e.to_string().contains(why), "rank {r}: {e}");
        }
    }
    // A bucket that cannot hold one f32 is refused.
    assert!(on_ranks(1, opts("tiny").with_mode(fast(2, true)), |_| ()).pop().unwrap().is_err());
}

#[test]
fn a_checkpoint_of_one_mode_is_refused_by_the_other() {
    let _guard = procs();
    let dir = scratch("fast-checkpoint");
    let cfg = gate_cfg();
    let (mut vars, mut adam) = gate_state(false);
    let one = |mode: DistMode| {
        let env = DistEnv { rank: 0, world_size: 1, rank_in_node: 0, node_size: 1, node_rank: 0, master_addr: "127.0.0.1".into(), master_port: 1 };
        ProcessGroup::init(&env, opts("ckpt").with_mode(mode)).unwrap()
    };
    let mut det = one(DistMode::Deterministic);
    let mut fst = one(fast(1 << 20, true));
    data_parallel_step(&mut det, &mut vars, &mut adam, 2, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &[i, i + 1], i as u64)).unwrap();
    Checkpoint::save(&mut det, &dir, 1, 1, &vars, &adam).unwrap();
    let e = Checkpoint::load_latest(&mut fst, &dir).unwrap_err();
    assert!(e.to_string().contains("resume refused") && e.to_string().contains("mode=fast"), "{e}");
    let dir2 = scratch("fast-checkpoint-2");
    data_parallel_step(&mut fst, &mut vars, &mut adam, 2, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &[i, i + 1], i as u64)).unwrap();
    Checkpoint::save(&mut fst, &dir2, 2, 2, &vars, &adam).unwrap();
    let e = Checkpoint::load_latest(&mut det, &dir2).unwrap_err();
    assert!(e.to_string().contains("resume refused"), "{e}");
    assert!(Checkpoint::load_latest(&mut fst, &dir2).unwrap().is_some());
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&dir2).unwrap();
}

#[test]
fn the_fast_cpu_backend_is_a_device_of_the_fast_mode_only() {
    let cpu_fast = crate::tensor::Device::Cpu(crate::tensor::CpuMode::Fast);
    let e = on_ranks(1, opts("cpu fast").with_device(cpu_fast), |g| g.key()).pop().unwrap().unwrap_err();
    assert!(e.to_string().contains("not a device of the deterministic mode"), "{e}");
    assert!(device_key(cpu_fast).is_err());
    let k = on_ranks(1, opts("cpu fast").with_device(cpu_fast).with_mode(fast(1 << 20, true)), |g| g.key()).pop().unwrap().unwrap();
    assert!(k.starts_with("mode=fast ") && k.ends_with(" device=cpu-fast"), "{k}");
}
