//! Tests of mixed precision (`mixed_*` steps, `LossScale`), a child of the distributed tests.
//! - The loss scale: a power of two, so scaling and unscaling are exact in f32; the finiteness
//!   check sees an infinity or a NaN anywhere.
//! - Refusals: a mixed step on a group made at f32; on a GPU, a step at another precision than the
//!   group was made at; a job whose ranks multiply at different precisions (another key).
//! - On a GPU (ranks as threads): W 2 in the deterministic mode gives one process's bits within the
//!   mixed key, data-parallel and sharded; a loss scale that overflows skips the step on every rank
//!   and leaves the vars alone; the mixed key differs from f32's.

use super::*;
use crate::nn::{all_finite, LossScale};

#[test]
fn the_loss_scale_is_a_power_of_two_and_exact() {
    assert!(LossScale::power_of_two(61).is_err() && LossScale::power_of_two(-61).is_err());
    let s = LossScale::power_of_two(12).unwrap();
    assert_eq!((s.log2(), s.factor(), s.name()), (12, 4096.0, "loss-scale=2^12".to_string()));
    // Values across the normal range: times 2^12, then 2^−12, the same bits.
    let v: Vec<f32> = (0..2000).map(|i| ((i as f32 - 1000.0) * 1.37e-3).powi(3) * 1e-20f32.powf((i % 7) as f32 / 7.0)).collect();
    let t = Tensor::from_data(v.clone(), [2000]);
    let scaled = s.scale(t.clone());
    let mut grads = vec![Some(scaled), None];
    s.unscale(&mut grads);
    assert_eq!(bits(&grads[0].as_ref().unwrap().to_vec()), bits(&v));
    assert!(grads[1].is_none());
    // No scaling is the identity, the tensor itself.
    let none = LossScale::none();
    assert_eq!(bits(&none.scale(t.clone()).to_vec()), bits(&v));
    // Finiteness.
    assert!(all_finite(&[Some(t.clone()), None]));
    for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let mut w = v.clone();
        w[1234] = bad;
        assert!(!all_finite(&[Some(t.clone()), Some(Tensor::from_data(w, [2000]))]), "{bad}");
    }
}

#[test]
fn a_mixed_step_needs_a_group_made_at_half() {
    // On the CPU a group is always made at f32 (the CPU multiplies in f32): refused.
    let mut g = ProcessGroup::single();
    let (mut vars, mut adam) = gate_state(false);
    let cfg = gate_cfg();
    let e = mixed_data_parallel_step(&mut g, &mut vars, &mut adam, 2, Reduce::Mean, 1.0, LossScale::none(), |l, i| gate_loss(&cfg, l, &[i, i + 1], i as u64)).unwrap_err();
    assert!(e.to_string().contains("mixed-precision step on a group made at matmul precision f32"), "{e}");
    assert_eq!(g.matmul_precision(), crate::tensor::MatmulPrecision::F32);
}

#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
mod gpu {
    use super::*;
    use crate::tensor::{set_matmul_precision, Device, MatmulPrecision};

    fn dev() -> Device {
        if gpu_kind() == "metal" { Device::Metal(0) } else { Device::Cuda(0) }
    }

    /// The gate's model on the GPU for `steps` steps: `w` ranks (threads, each setting Half before
    /// its group is made) with `k` micro-steps each, mixed precision with `scale`, data-parallel or
    /// sharded (stage 2, flat); per rank, the per-step hashes, whether each step stepped, the key.
    fn run(w: usize, k: usize, steps: u64, scale: LossScale, sharded: bool) -> Vec<(Vec<String>, Vec<bool>, String)> {
        let port = free_port().unwrap();
        let hs: Vec<_> = (0..w)
            .map(|r| {
                std::thread::spawn(move || {
                    set_matmul_precision(MatmulPrecision::Half);
                    let env = DistEnv { rank: r, world_size: w, rank_in_node: r, node_size: w, node_rank: 0, master_addr: "127.0.0.1".into(), master_port: port };
                    let mut g = ProcessGroup::init(&env, opts("mixed").with_device(dev())).unwrap();
                    let cfg = gate_cfg();
                    let (vars, adam) = gate_state_on(false, None, dev());
                    crate::tensor::set_default_device(dev()).unwrap();
                    let micro = w * k;
                    let shard = Shard::new(ShardSpec { seed: 11, items: 24, micro_batch: 2, micro_steps: micro }, r, w).unwrap();
                    let (mut hashes, mut stepped) = (vec![], vec![]);
                    let mut vars = vars;
                    let mut adam = adam;
                    let mut sdp = sharded.then(|| ShardedDataParallel::new(&mut g, vars.clone(), crate::nn::ParamGroups::new(&vars), ShardOptions::new(ShardStage::Gradients), |v, gr| Adam::new(AdamConfig { lr: 1e-2, ..AdamConfig::default() }, gr, v)).unwrap());
                    for step in 0..steps {
                        let f = |l: &VarMap, i: usize| gate_micro(&cfg, l, &shard.batch(step, i), step, shard.global_micro(i) as u64, micro, false);
                        let s = match &mut sdp {
                            None => mixed_data_parallel_step(&mut g, &mut vars, &mut adam, k, Reduce::Mean, 1.0, scale, f).unwrap(),
                            Some(z) => {
                                let s = mixed_sharded_step(z, &mut g, k, Reduce::Mean, 1.0, scale, f).unwrap();
                                vars = z.full_params(&mut g).unwrap();
                                s
                            }
                        };
                        stepped.push(s.stepped);
                        hashes.push(check_in_sync(&mut g, &vars).unwrap());
                    }
                    set_matmul_precision(MatmulPrecision::F32);
                    (hashes, stepped, g.key())
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("a rank thread")).collect()
    }

    #[test]
    fn gpu_mixed_world_size_2_equals_one_process_within_the_mixed_key() {
        let _guard = procs();
        let one = run(1, 2, 6, LossScale::none(), false);
        let two = run(2, 1, 6, LossScale::none(), false);
        let sharded = run(2, 1, 6, LossScale::none(), true);
        assert!(one[0].1.iter().all(|s| *s));
        for r in two.iter().chain(&sharded) {
            assert_eq!(r.0, one[0].0, "every per-step hash equals one process with 2 micro-steps");
        }
        assert!(one[0].2.contains("-gemm-"), "{}", one[0].2);
        eprintln!("{} mixed: W 2 (data-parallel and sharded stage 2) equal one process for 6 steps; final {}", gpu_kind(), one[0].0.last().unwrap());
    }

    #[test]
    fn gpu_mixed_overflow_skips_the_step_on_every_rank() {
        let _guard = procs();
        // A loss scale of 2^60 overflows the scaled backward pass: every step is skipped, on both
        // ranks, and the vars keep their initial bits.
        let (vars, _) = gate_state_on(false, None, dev());
        let initial = state_hash(&vars).unwrap();
        for sharded in [false, true] {
            let res = run(2, 1, 3, LossScale::power_of_two(60).unwrap(), sharded);
            for r in &res {
                assert_eq!(r.1, vec![false; 3], "sharded {sharded}");
                assert!(r.0.iter().all(|h| *h == initial), "sharded {sharded}");
            }
        }
        // A moderate scale steps.
        let res = run(2, 1, 2, LossScale::power_of_two(8).unwrap(), false);
        assert!(res.iter().all(|r| r.1 == vec![true; 2]));
    }

    #[test]
    fn gpu_a_job_mixing_precisions_or_a_step_at_another_precision_is_refused() {
        let _guard = procs();
        // Rank 1 multiplies at Half, rank 0 at f32: other platform keys, refused at the rendezvous.
        let port = free_port().unwrap();
        let hs: Vec<_> = (0..2)
            .map(|r| {
                std::thread::spawn(move || {
                    if r == 1 {
                        set_matmul_precision(MatmulPrecision::Half);
                    }
                    let env = DistEnv { rank: r, world_size: 2, rank_in_node: r, node_size: 2, node_rank: 0, master_addr: "127.0.0.1".into(), master_port: port };
                    let e = ProcessGroup::init(&env, opts("mixed precisions").with_device(dev())).map(|_| ()).map_err(|e| e.to_string());
                    set_matmul_precision(MatmulPrecision::F32);
                    e
                })
            })
            .collect();
        for h in hs {
            let e = h.join().unwrap().unwrap_err();
            assert!(e.contains("refused"), "{e}");
        }
        // A group made at f32 refuses a step taken at Half, and the other way round.
        let mut g = ProcessGroup::init(&DistEnv::single(), opts("one").with_device(dev())).unwrap();
        let (mut vars, mut adam) = gate_state_on(false, None, dev());
        crate::tensor::set_default_device(dev()).unwrap();
        let cfg = gate_cfg();
        set_matmul_precision(MatmulPrecision::Half);
        let e = data_parallel_step(&mut g, &mut vars, &mut adam, 1, Reduce::Mean, 1.0, |l, i| gate_loss(&cfg, l, &[i, i + 1], i as u64)).unwrap_err();
        set_matmul_precision(MatmulPrecision::F32);
        crate::tensor::set_default_device(crate::tensor::Device::Cpu(crate::tensor::CpuMode::Reference)).unwrap();
        assert!(e.to_string().contains("the group was made (and is keyed) at f32"), "{e}");
    }
}
