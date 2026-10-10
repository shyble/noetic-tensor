//! The copy task (`a b c SEP a b c`) trained data-parallel: each optimizer step takes 4 global
//! micro-batches of 8 items, split among the ranks, summed in rank order; every rank takes the
//! same step. Run alone it accumulates the 4 micro-batches itself, and gives the same weights,
//! bit for bit, as any world size that divides 4.
//!
//! ```sh
//! cargo build --release --examples
//! target/release/examples/ddp_copy_task                                                        # one process
//! target/release/examples/launch --nproc-per-node 2 -- target/release/examples/ddp_copy_task   # two
//! target/release/examples/launch --nproc-per-node 4 -- target/release/examples/ddp_copy_task   # four
//! ```

use noetic::nn::dist::{check_in_sync, data_parallel_step, GroupOptions, ProcessGroup, Reduce, Shard, ShardSpec};
use noetic::nn::{masked_cross_entropy, Adam, AdamConfig, Decoder, DecoderConfig, ForwardOptions, ParamGroups};
use noetic::tensor::{pin_reference, IntTensor, Tensor};
use rand::{Rng, SeedableRng};

const SEEDS: usize = 2;
const MICRO_STEPS: usize = 4;

/// Item i: `a b c SEP a b c 0` as (tokens, targets, answer mask), from the item's own stream.
fn item(i: usize) -> (Vec<i64>, Vec<i64>, Vec<f32>) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(i as u64);
    let a: Vec<i64> = (0..3).map(|_| rng.gen_range(2..16i64)).collect();
    let seq = [a.clone(), vec![1], a, vec![0]].concat();
    (seq[..7].to_vec(), seq[1..].to_vec(), (0..7).map(|t| if (3..6).contains(&t) { 1.0 } else { 0.0 }).collect())
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
    let shape = [SEEDS, items.len(), 7];
    (IntTensor::from_data(t, shape), IntTensor::from_data(y, shape), Tensor::from_data(m, shape))
}

fn main() -> noetic::Result<()> {
    pin_reference();
    let mut group = ProcessGroup::from_env(GroupOptions::new("copy task"))?;
    let (rank, world) = (group.rank(), group.world_size());
    let cfg = DecoderConfig::new(16, 32, 4, 7, 2, 96);
    let (_, mut vars) = Decoder::init(cfg.clone(), SEEDS, 1)?;
    let mut opt = Adam::new(AdamConfig::default(), ParamGroups::new(&vars), &vars);
    let shard = Shard::new(ShardSpec { seed: 7, items: 4096, micro_batch: 8, micro_steps: MICRO_STEPS }, rank, world)?;
    if rank == 0 {
        println!("{} parameters per model, {SEEDS} models; {} ({} micro-steps per rank)", vars.count_per_seed(), group.record(), shard.micro_steps_per_rank());
    }
    for step in 0..200u64 {
        let mut ce = vec![0.0f32; SEEDS];
        data_parallel_step(&mut group, &mut vars, &mut opt, shard.micro_steps_per_rank(), Reduce::Mean, 1.0, |lifted, i| {
            let (tokens, targets, mask) = batch(&shard.batch(step, i));
            let model = Decoder::load(cfg.clone(), lifted).expect("the vars fit the config");
            let out = model.forward_with(&tokens, &ForwardOptions { pad: None, train_step: Some(step * MICRO_STEPS as u64 + shard.global_micro(i) as u64) });
            let loss = masked_cross_entropy(out.logits, &targets, &mask);
            for (c, l) in ce.iter_mut().zip(loss.to_vec()) {
                *c += l;
            }
            loss.sum()
        })?;
        if rank == 0 && (step % 50 == 0 || step == 199) {
            let ce: Vec<String> = ce.iter().map(|c| format!("{:.3}", c / shard.micro_steps_per_rank() as f32)).collect();
            println!("step {step:>3}: rank 0's answer loss per model [{}]", ce.join(", "));
        }
    }
    let hash = check_in_sync(&mut group, &vars)?;
    println!("rank {rank} of {world}: final weights sha256 {hash}");
    Ok(())
}
