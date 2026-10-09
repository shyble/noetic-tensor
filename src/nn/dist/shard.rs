//! The distributed sampler: which items each rank trains on, as a function of the seed, the rank
//! and the world size only.
//!
//! The items form one global stream: epoch e is a permutation of `0..items` drawn from
//! `rng::derive(seed, "shard epoch e")` (Fisher–Yates on u64 draws, the same on every platform),
//! and the epochs follow each other. Optimizer step s takes the next `micro_steps` micro-batches
//! of `micro_batch` items from the stream; global micro-batch j of step s starts at item
//! `(s·micro_steps + j)·micro_batch`. Rank r of W takes micro-batches `r·k .. (r+1)·k`
//! (`k = micro_steps / W`), so the ranks' micro-batches, in rank order, are exactly the single
//! process's micro-batches of that step, and a resumed run needs only the step number (the
//! loader's cursor).

use crate::error::{NnError, Result};
use rand::Rng;
use std::cell::RefCell;

/// What the sampler draws from: the seed, the item count, the micro-batch size and the global
/// micro-steps per optimizer step (the same for every world size).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardSpec {
    pub seed: u64,
    pub items: usize,
    pub micro_batch: usize,
    pub micro_steps: usize,
}

impl ShardSpec {
    /// The spec as text, for keys and hashes.
    pub fn key(&self) -> String {
        format!("seed={} items={} micro_batch={} micro_steps={}", self.seed, self.items, self.micro_batch, self.micro_steps)
    }
}

/// One rank's share of the stream.
#[derive(Debug)]
pub struct Shard {
    spec: ShardSpec,
    rank: usize,
    world: usize,
    /// The last epoch's permutation.
    perm: RefCell<Option<(u64, Vec<usize>)>>,
}

impl Shard {
    /// Rank `rank` of `world`; the global micro-steps must divide evenly among the ranks.
    pub fn new(spec: ShardSpec, rank: usize, world: usize) -> Result<Shard> {
        if spec.items == 0 || spec.micro_batch == 0 || spec.micro_steps == 0 {
            return Err(NnError::Dist(format!("an empty sampler ({})", spec.key())));
        }
        if world == 0 || rank >= world || !spec.micro_steps.is_multiple_of(world) {
            return Err(NnError::Dist(format!("{} global micro-steps do not divide among {world} ranks (rank {rank})", spec.micro_steps)));
        }
        Ok(Shard { spec, rank, world, perm: RefCell::new(None) })
    }

    pub fn spec(&self) -> &ShardSpec {
        &self.spec
    }

    /// The micro-steps this rank runs per optimizer step.
    pub fn micro_steps_per_rank(&self) -> usize {
        self.spec.micro_steps / self.world
    }

    /// The global index of this rank's micro-step `i` within a step.
    pub fn global_micro(&self, i: usize) -> usize {
        self.rank * self.micro_steps_per_rank() + i
    }

    /// Item `p` of the global stream.
    fn item(&self, p: u64) -> usize {
        let n = self.spec.items as u64;
        let (epoch, at) = (p / n, (p % n) as usize);
        let mut cache = self.perm.borrow_mut();
        if cache.as_ref().map(|(e, _)| *e) != Some(epoch) {
            *cache = Some((epoch, permutation(self.spec.seed, epoch, self.spec.items)));
        }
        cache.as_ref().expect("filled above").1[at]
    }

    /// The items of this rank's micro-step `i` (of `micro_steps_per_rank`) in optimizer step `step`.
    pub fn batch(&self, step: u64, i: usize) -> Vec<usize> {
        assert!(i < self.micro_steps_per_rank(), "micro-step {i} of {}", self.micro_steps_per_rank());
        let s = &self.spec;
        let start = (step * s.micro_steps as u64 + self.global_micro(i) as u64) * s.micro_batch as u64;
        (0..s.micro_batch as u64).map(|q| self.item(start + q)).collect()
    }

    /// The shard's key: the spec, the rank and the world size.
    pub fn key(&self) -> String {
        format!("{} rank={} world={}", self.spec.key(), self.rank, self.world)
    }

    /// sha256 of the shard's key and every item it takes in `steps`, in order: equal hashes mean
    /// the same data in the same order.
    pub fn hash(&self, steps: std::ops::Range<u64>) -> String {
        let mut b = self.key().into_bytes();
        for s in steps {
            for i in 0..self.micro_steps_per_rank() {
                for x in self.batch(s, i) {
                    b.extend_from_slice(&(x as u64).to_le_bytes());
                }
            }
        }
        crate::hash::sha256_hex(&b)
    }
}

/// Epoch `epoch`'s permutation of `0..n`.
fn permutation(seed: u64, epoch: u64, n: usize) -> Vec<usize> {
    let mut rng = crate::rng::derive(seed, &format!("shard epoch {epoch}"));
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i as u64) as usize;
        p.swap(i, j);
    }
    p
}
