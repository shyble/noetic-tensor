//! Sharded data parallelism in the shape of ZeRO's stages and FSDP, in deterministic mode: the
//! optimizer's state, then the summed gradients, then the parameters are split among the ranks,
//! and every rank's weights stay bit for bit those of data-parallel training (`data_parallel_step`)
//! at the same world size and micro-steps.
//!
//! **Pieces.** A `ShardLayout` gives each rank a piece of every var: a range of its elements in
//! row-major order, the ranges increasing with the rank. `Partition::Flat` cuts every var into
//! world-size contiguous ranges (rank r of W takes `[n·r/W, n·(r+1)/W)`; FSDP's flat parameter, per
//! var); `Partition::ByVar` gives each var whole to one rank: the vars from the largest, each to the
//! least loaded rank so far (lowest rank on a tie), in the shape of torch's
//! `ZeroRedundancyOptimizer`, which also gives whole parameters to ranks. A piece that covers its
//! whole var keeps the var's shape; any other piece is a rank-1 tensor of its elements.
//!
//! **Stages** (`ShardStage`):
//! 1. `OptimizerState`: every rank keeps the whole vars; the gradients are all-reduced as in
//!    `data_parallel_step`; each rank steps its pieces with its own optimizer (whose state covers
//!    its pieces only), and the updated pieces are gathered back into every rank's vars.
//! 2. `Gradients`: as 1, but the summed gradients are reduce-scattered: each rank receives only its
//!    pieces of the sum.
//! 3. `Parameters`: as 2, and between steps each rank keeps only its pieces of the vars; the whole
//!    vars are gathered at the start of a step for its micro-steps and dropped before the
//!    reduction. (The whole model is gathered at once: there is no per-layer gather and free.)
//!
//! **Why the bits are data parallelism's.**
//! - The reduce-scatter is the all-reduce's chain (`GradSum`, the global micro-steps folded in flat
//!   (rank, micro-step) order along the ranks); only the last step differs: the last rank sends each
//!   rank its own pieces instead of the whole total. Every element is the same fold of the same
//!   values in the same order, and `Reduce::Mean` divides it by the same count.
//! - The optimizers (Adam, SGD) update every element on its own: the same operations on the same
//!   values give the same bits whether a tensor holds the whole var or a piece. Options that need a
//!   var's shape (lazy rows, multiplier tensors, per-seed rates) are refused unless every var they
//!   touch stays whole (`Partition::ByVar`).
//! - Gathering moves bits, never adds them.
//! - A var no micro-step on any rank used has no sum on any rank, so every piece of it is skipped by
//!   its optimizer, decay included, as in one process.
//! - Clipping (`ShardGrads::clip_grad_norm_per_seed`) gives `clip_grad_norm_per_seed`'s norms bit
//!   for bit: each (var, seed) partial sum of squares is one f64 fold over the var's elements in
//!   row-major order, continued from rank to rank in rank order, and the partials are added var by
//!   var in the vars' order, as one process does.
//!
//! The group's key is unchanged: in the flat order the weights do not depend on the sharding, and
//! a run's record names it (`ShardedDataParallel::record`).
//!
//! **Checkpoints.** `save_consolidated` gathers the whole vars and optimizer state and writes
//! `Checkpoint::save`'s checkpoint, byte for byte the one data parallelism writes at the same step
//! (and either kind of run resumes it). `save_sharded` has every rank write its own pieces (vars and
//! optimizer state) into one folder, with a meta file holding every file's sha256; a sharded resume
//! takes the same world size, partition, job key and group key.

use super::checkpoint::{io, set_map, write_atomic, Checkpoint, CheckpointOptimizer};
use super::ddp::state_hash;
use super::group::ProcessGroup;
use super::reduce::{scale, Reduce};
use super::wire::{Dec, Enc};
use crate::error::{NnError, Result};
use crate::nn::{Optimizer, ParamGroups, VarMap};
use crate::tensor::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// What is split among the ranks (ZeRO's stages; stage 3 is FSDP's full sharding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardStage {
    /// Stage 1: the optimizer's state.
    OptimizerState,
    /// Stage 2: the optimizer's state and the summed gradients.
    Gradients,
    /// Stage 3: the optimizer's state, the summed gradients and the vars between steps.
    Parameters,
}

impl ShardStage {
    pub fn number(self) -> u8 {
        match self {
            ShardStage::OptimizerState => 1,
            ShardStage::Gradients => 2,
            ShardStage::Parameters => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ShardStage::OptimizerState => "optimizer-state",
            ShardStage::Gradients => "gradients",
            ShardStage::Parameters => "parameters",
        }
    }
}

/// How the vars' elements are cut into the ranks' pieces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Partition {
    /// Every var cut into world-size contiguous ranges of its row-major elements.
    #[default]
    Flat,
    /// Every var whole on one rank: the vars from the largest (the earlier on a tie), each to the
    /// least loaded rank so far (the lowest on a tie).
    ByVar,
}

impl Partition {
    pub fn name(self) -> &'static str {
        match self {
            Partition::Flat => "flat",
            Partition::ByVar => "by-var",
        }
    }
}

/// Options of sharded data parallelism. Made with `ShardOptions::new` and the `with_*` methods;
/// non-exhaustive, so later fields do not break callers.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ShardOptions {
    pub stage: ShardStage,
    pub partition: Partition,
}

impl ShardOptions {
    /// Sharding at `stage`, flat pieces.
    pub fn new(stage: ShardStage) -> ShardOptions {
        ShardOptions { stage, partition: Partition::Flat }
    }

    pub fn with_partition(mut self, partition: Partition) -> ShardOptions {
        self.partition = partition;
        self
    }
}

/// One rank's piece of a var: elements `lo..hi` in row-major order; `whole` when it is the whole
/// var (the piece then keeps the var's shape).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub var: usize,
    pub lo: usize,
    pub hi: usize,
    pub whole: bool,
}

impl Piece {
    pub fn len(&self) -> usize {
        self.hi - self.lo
    }

    pub fn is_empty(&self) -> bool {
        self.hi == self.lo
    }
}

/// Which elements of which var each rank holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardLayout {
    names: Vec<String>,
    shapes: Vec<Vec<usize>>,
    world: usize,
    partition: Partition,
    /// `ranges[v][r]`: rank r's elements of var v, increasing with r.
    ranges: Vec<Vec<(usize, usize)>>,
}

impl ShardLayout {
    /// The layout of `vars` among `world` ranks. Every var must hold at least one element.
    pub fn new(vars: &VarMap, world: usize, partition: Partition) -> Result<ShardLayout> {
        if world == 0 {
            return Err(NnError::Dist("a layout for no ranks".into()));
        }
        let shapes: Vec<Vec<usize>> = vars.tensors().iter().map(|t| t.shape().to_vec()).collect();
        for (name, shape) in vars.names().iter().zip(&shapes) {
            if shape.iter().product::<usize>() == 0 {
                return Err(NnError::Dist(format!("var {name:?} has no elements (sharding needs at least one)")));
            }
        }
        let sizes: Vec<usize> = shapes.iter().map(|s| s.iter().product()).collect();
        // By var: the vars from the largest (the earlier on a tie), each to the least loaded rank
        // (the lowest on a tie).
        let mut holder = vec![0usize; sizes.len()];
        if partition == Partition::ByVar {
            let mut order: Vec<usize> = (0..sizes.len()).collect();
            order.sort_by_key(|v| (std::cmp::Reverse(sizes[*v]), *v));
            let mut loads = vec![0usize; world];
            for v in order {
                let r = (0..world).min_by_key(|r| (loads[*r], *r)).expect("a rank");
                loads[r] += sizes[v];
                holder[v] = r;
            }
        }
        let ranges = sizes
            .iter()
            .zip(&holder)
            .map(|(&n, &o)| match partition {
                Partition::Flat => (0..world).map(|r| (n * r / world, n * (r + 1) / world)).collect(),
                Partition::ByVar => (0..world).map(|r| if r < o { (0, 0) } else if r == o { (0, n) } else { (n, n) }).collect(),
            })
            .collect();
        Ok(ShardLayout { names: vars.names().to_vec(), shapes, world, partition, ranges })
    }

    pub fn world_size(&self) -> usize {
        self.world
    }

    pub fn partition(&self) -> Partition {
        self.partition
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn shapes(&self) -> &[Vec<usize>] {
        &self.shapes
    }

    fn numel(&self, var: usize) -> usize {
        self.shapes[var].iter().product()
    }

    /// Rank `rank`'s elements of var `var`.
    pub fn range(&self, var: usize, rank: usize) -> (usize, usize) {
        self.ranges[var][rank]
    }

    /// Rank `rank`'s non-empty pieces, in the vars' order.
    pub fn pieces(&self, rank: usize) -> Vec<Piece> {
        (0..self.names.len())
            .filter_map(|v| {
                let (lo, hi) = self.ranges[v][rank];
                (hi > lo).then(|| Piece { var: v, lo, hi, whole: lo == 0 && hi == self.numel(v) })
            })
            .collect()
    }

    /// Every rank's pieces as (var, first element, end), for the reduce-scatter.
    fn segments(&self) -> Vec<Vec<(usize, usize, usize)>> {
        (0..self.world).map(|r| self.pieces(r).iter().map(|p| (p.var, p.lo, p.hi)).collect()).collect()
    }

    /// Whether var `var` is cut into more than one piece.
    pub fn is_split(&self, var: usize) -> bool {
        let n = self.numel(var);
        !self.ranges[var].iter().any(|&(lo, hi)| lo == 0 && hi == n)
    }

    /// The shape of a piece: the var's when whole, else `[len]`.
    pub fn piece_shape(&self, p: &Piece) -> Vec<usize> {
        if p.whole {
            self.shapes[p.var].clone()
        } else {
            vec![p.len()]
        }
    }

    /// The bytes of the vars' f32 elements a rank holds.
    pub fn bytes(&self, rank: usize) -> usize {
        self.pieces(rank).iter().map(|p| 4 * p.len()).sum()
    }

    /// sha256 of the partition, the world size, the vars' names and shapes and every range.
    pub fn key(&self) -> String {
        let mut text = format!("partition={} world={};", self.partition.name(), self.world);
        for ((n, s), r) in self.names.iter().zip(&self.shapes).zip(&self.ranges) {
            text.push_str(&format!("{n}:{s:?}:{r:?};"));
        }
        crate::hash::sha256_hex(text.as_bytes())
    }

    /// Rank `rank`'s pieces of `vars` (every var of this layout), on `device`, as a var map under
    /// the vars' names.
    fn slice(&self, vars: &VarMap, rank: usize, device: Device) -> Result<VarMap> {
        let mut out = VarMap::new();
        for p in self.pieces(rank) {
            out.insert(self.names[p.var].clone(), piece_of(&vars.tensors()[p.var], &p, device))?;
        }
        Ok(out)
    }

    /// Whole vars on `device` from every rank's pieces gathered by `gather_values`.
    fn assemble(&self, vals: Vec<Option<Vec<f32>>>, device: Device) -> Result<VarMap> {
        let mut out = VarMap::new();
        for (v, x) in vals.into_iter().enumerate() {
            let x = x.ok_or_else(|| NnError::Dist(format!("var {:?}: a piece is missing", self.names[v])))?;
            out.insert(self.names[v].clone(), Tensor::from_data(x, self.shapes[v].clone()).to(device))?;
        }
        Ok(out)
    }
}

/// Piece `p` of a var's tensor `t`, on `device`: `t` itself when whole, else its elements
/// `lo..hi` as a rank-1 tensor.
fn piece_of(t: &Tensor, p: &Piece, device: Device) -> Tensor {
    if p.whole {
        t.clone().to(device)
    } else {
        Tensor::from_data(t.to_vec()[p.lo..p.hi].to_vec(), vec![p.len()]).to(device)
    }
}

/// The f32 values of a tensor (refusing other dtypes).
fn f32_values(t: &Tensor, what: &str) -> Result<Vec<f32>> {
    if t.dtype() != DType::F32 {
        return Err(NnError::Dist(format!("{what}: a {} tensor (sharding takes f32)", t.dtype().name())));
    }
    Ok(t.to_vec())
}

/// Every rank's pieces gathered into whole vars, on every rank: `mine` holds this rank's values,
/// one entry per piece of `layout.pieces(rank)` (None: no tensor for that var). A var is Some when
/// every piece of it is, None when none is; a mix is an error. Only bits move.
fn gather_values(group: &mut ProcessGroup, layout: &ShardLayout, mine: &[Option<Vec<f32>>]) -> Result<Vec<Option<Vec<f32>>>> {
    let own = layout.pieces(group.rank());
    if own.len() != mine.len() {
        return Err(NnError::Dist(format!("{} values for {} pieces", mine.len(), own.len())));
    }
    let mut e = Enc::default();
    for (p, x) in own.iter().zip(mine) {
        match x {
            Some(x) if x.len() == p.len() => {
                e.u8(1).f32s(x);
            }
            Some(x) => return Err(NnError::Dist(format!("var {:?}: {} values for a piece of {}", layout.names[p.var], x.len(), p.len()))),
            None => {
                e.u8(0);
            }
        }
    }
    let all = group.all_gather(&e.0)?;
    let n = layout.names.len();
    let mut vals: Vec<Option<Vec<f32>>> = vec![None; n];
    let mut present: Vec<Option<bool>> = vec![None; n];
    for (q, bytes) in all.iter().enumerate() {
        let mut d = Dec::new(bytes);
        for p in layout.pieces(q) {
            let x = match d.u8()? {
                0 => None,
                1 => Some(d.f32s(p.len())?),
                b => return Err(NnError::Dist(format!("rank {q}: presence byte {b}"))),
            };
            if present[p.var].is_some_and(|was| was != x.is_some()) {
                return Err(NnError::Dist(format!("var {:?}: some ranks hold a piece of it, others none", layout.names[p.var])));
            }
            present[p.var] = Some(x.is_some());
            if let Some(x) = x {
                vals[p.var].get_or_insert_with(|| vec![0.0; layout.numel(p.var)])[p.lo..p.hi].copy_from_slice(&x);
            }
        }
        d.done()?;
    }
    Ok(vals)
}

/// Every rank's result agreed: Ok on every rank, or the first rank's error on every rank.
fn agree(group: &mut ProcessGroup, what: &str, res: Result<()>) -> Result<()> {
    let mine = match &res {
        Ok(()) => "ok".to_string(),
        Err(e) => format!("rank {}: {e}", group.rank()),
    };
    let all = group.all_gather(mine.as_bytes())?;
    match all.iter().find(|a| a.as_slice() != b"ok") {
        None => Ok(()),
        Some(bad) => Err(NnError::Dist(format!("{what}: {}", String::from_utf8_lossy(bad)))),
    }
}

/// A rank's pieces of a step's summed gradients, scaled by the step's `Reduce`, on its device.
#[derive(Debug)]
pub struct ShardGrads {
    layout: ShardLayout,
    rank: usize,
    pieces: Vec<Piece>,
    grads: Vec<Option<Tensor>>,
    count: usize,
}

impl ShardGrads {
    /// The micro-steps summed over every rank.
    pub fn count(&self) -> usize {
        self.count
    }

    /// This rank's pieces, in order.
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// The gradient of each piece (None: no micro-step on any rank used the var).
    pub fn grads(&self) -> &[Option<Tensor>] {
        &self.grads
    }

    /// The gradients, for an elementwise transform every rank applies to its pieces.
    pub fn grads_mut(&mut self) -> &mut [Option<Tensor>] {
        &mut self.grads
    }

    /// The bytes of the gradients held.
    pub fn bytes(&self) -> usize {
        self.grads.iter().flatten().map(|g| g.numel() * g.dtype().size_in_bytes()).sum()
    }

    /// The L2 norm of each seed's whole gradient over every var, on every rank: bit for bit
    /// `grad_norms_per_seed` of the whole summed gradients. Each (var, seed) sum of squares is one
    /// f64 fold over the var's elements in row-major order, passed from rank to rank in rank order
    /// (a broadcast per rank); the sums are then added var by var. A collective.
    pub fn grad_norms_per_seed(&self, group: &mut ProcessGroup) -> Result<Vec<f64>> {
        let l = &self.layout;
        let n = l.names.len();
        let seeds: Vec<usize> = l.shapes.iter().map(|s| s[0]).collect();
        // The neutral element of f64's `Sum`, which the one-process norm folds from.
        let start: f64 = std::iter::empty::<f64>().sum();
        // Per var: 0 not yet seen, 1 no gradient, 2 a gradient; and its per-seed partial sums.
        let mut state = vec![0u8; n];
        let mut parts: Vec<Vec<f64>> = seeds.iter().map(|s| vec![start; *s]).collect();
        let me = group.rank();
        for r in 0..group.world_size() {
            let mut buf = vec![];
            if r == me {
                for (p, g) in self.pieces.iter().zip(&self.grads) {
                    match g {
                        None => state[p.var] = 1,
                        Some(g) => {
                            state[p.var] = 2;
                            let per = l.numel(p.var) / seeds[p.var];
                            let x = g.to_vec_f64();
                            for (k, v) in x.iter().enumerate() {
                                let i = (p.lo + k) / per;
                                parts[p.var][i] += v * v;
                            }
                        }
                    }
                }
                let mut e = Enc::default();
                for (s, ps) in state.iter().zip(&parts) {
                    e.u8(*s);
                    for x in ps {
                        e.u64(x.to_bits());
                    }
                }
                buf = e.0;
            }
            group.broadcast(r, &mut buf)?;
            if r != me {
                let mut d = Dec::new(&buf);
                for (s, ps) in state.iter_mut().zip(parts.iter_mut()) {
                    *s = d.u8()?;
                    for x in ps.iter_mut() {
                        *x = f64::from_bits(d.u64()?);
                    }
                }
                d.done()?;
            }
        }
        if let Some(v) = state.iter().position(|s| *s == 0) {
            return Err(NnError::Dist(format!("var {:?}: no rank holds a piece of it", l.names[v])));
        }
        let Some(first) = state.iter().position(|s| *s == 2) else { return Ok(vec![]) };
        let s = seeds[first];
        let mut sq = vec![0.0f64; s];
        for v in (0..n).filter(|v| state[*v] == 2) {
            if seeds[v] != s {
                return Err(NnError::Dist("every gradient has the same seed count".into()));
            }
            for (acc, x) in sq.iter_mut().zip(&parts[v]) {
                *acc += x;
            }
        }
        Ok(sq.into_iter().map(f64::sqrt).collect())
    }

    /// Scale each seed's gradient pieces by `min(1, max_norm / (norm + 1e-6))`, the norm being
    /// `grad_norms_per_seed`'s; returns the norms before clipping. Bit for bit
    /// `clip_grad_norm_per_seed` on the whole summed gradients, then cut into pieces. A collective.
    pub fn clip_grad_norm_per_seed(&mut self, group: &mut ProcessGroup, max_norm: f64) -> Result<Vec<f64>> {
        let norms = self.grad_norms_per_seed(group)?;
        let factors: Vec<f64> = norms.iter().map(|n| if *n > max_norm { max_norm / (n + 1e-6) } else { 1.0 }).collect();
        if factors.iter().all(|f| *f == 1.0) {
            return Ok(norms);
        }
        let s = factors.len();
        for (p, g) in self.pieces.iter().zip(self.grads.iter_mut()) {
            let Some(t) = g else { continue };
            let f = if p.whole {
                // As one process: the factors `[S]` viewed as `[S, 1, …, 1]`.
                let mut shape = vec![1; t.rank()];
                shape[0] = s;
                Tensor::from_f64s(factors.clone(), vec![s], t.dtype()).to(t.device()).reshape_dyn(shape)
            } else {
                // Each element's seed's factor.
                let per = self.layout.numel(p.var) / self.layout.shapes[p.var][0];
                Tensor::from_f64s((p.lo..p.hi).map(|e| factors[e / per]).collect(), vec![p.len()], t.dtype()).to(t.device())
            };
            *t = t.clone() * f;
        }
        Ok(norms)
    }
}

/// The bytes a rank holds (`ShardedDataParallel::memory`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ShardMemory {
    /// The vars held between steps (stages 1–2: every var; stage 3: this rank's pieces).
    pub params: usize,
    /// The summed gradient held after the last step's reduction (stage 1: the whole all-reduced
    /// sum; stages 2–3: this rank's pieces). Not counted: the micro-steps' own gradients, which
    /// are whole on every rank, and the whole sum the last rank holds during the reduction.
    pub grads: usize,
    /// The optimizer's state (this rank's pieces).
    pub optimizer_state: usize,
}

/// The vars a rank keeps between steps.
#[derive(Debug)]
enum Params {
    /// Stages 1–2: every var, whole.
    Full(VarMap),
    /// Stage 3: this rank's pieces.
    Shard(VarMap),
}

/// Data-parallel training with the optimizer's state, the summed gradients and (at stage 3) the
/// vars split among the ranks; the weights are `data_parallel_step`'s, bit for bit, at every
/// stage and partition. `O` is the optimizer of this rank's pieces.
pub struct ShardedDataParallel<O> {
    opts: ShardOptions,
    layout: ShardLayout,
    rank: usize,
    device: Device,
    params: Params,
    opt: O,
    last_grads: usize,
}

impl<O> std::fmt::Debug for ShardedDataParallel<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedDataParallel").field("opts", &self.opts).field("rank", &self.rank).field("device", &self.device).field("layout", &self.layout.key()).finish_non_exhaustive()
    }
}

impl<O> ShardedDataParallel<O> {
    /// Shard `vars` (the same on every rank, checked) among the group's ranks. `groups` holds the
    /// per-var options for the whole vars; `make` builds this rank's optimizer from its pieces (a
    /// var map under the vars' names) and their options, as one would build it for whole vars
    /// (`|vars, groups| Adam::new(cfg, groups, vars)`). Every rank calls it (a collective).
    ///
    /// Refused, with one error each: vars off the group's device or not f32, a var without
    /// elements, and, on a var cut into pieces (`Partition::Flat` at world size 2 or more), lazy
    /// rows ("lazy rows need its [S, R, C] shape"), a multiplier tensor ("its multiplier tensor
    /// broadcasts against its shape") and per-seed learning rates ("per-seed learning rates need
    /// its seed axis"). `Partition::ByVar` keeps every var whole, and takes all three.
    pub fn new(group: &mut ProcessGroup, vars: VarMap, groups: ParamGroups, opts: ShardOptions, make: impl FnOnce(&VarMap, ParamGroups) -> O) -> Result<Self> {
        let device = group.device();
        let check = (|| {
            for (i, t) in vars.tensors().iter().enumerate() {
                if t.device() != device {
                    return Err(NnError::Dist(format!("var {:?} is on {:?}, the group trains on {:?} (GroupOptions::device)", vars.names()[i], t.device(), device)));
                }
                f32_values_dtype(t, &vars.names()[i])?;
            }
            if groups.names() != vars.names() {
                return Err(NnError::Dist("the parameter groups are for another var map".into()));
            }
            let layout = ShardLayout::new(&vars, group.world_size(), opts.partition)?;
            // Options that need a var's shape are refused on a var cut into pieces (one error each).
            for i in (0..vars.len()).filter(|i| layout.is_split(*i)) {
                let name = &vars.names()[i];
                if groups.options(i).lazy_rows {
                    return Err(NnError::Dist(format!("var {name:?} is cut into pieces: lazy rows need its [S, R, C] shape (refused; Partition::ByVar keeps every var whole)")));
                }
                if groups.multiplier(i).is_some() {
                    return Err(NnError::Dist(format!("var {name:?} is cut into pieces: its multiplier tensor broadcasts against its shape (refused; Partition::ByVar keeps every var whole)")));
                }
                if groups.seed_lr().is_some() {
                    return Err(NnError::Dist(format!("var {name:?} is cut into pieces: per-seed learning rates need its seed axis (refused; Partition::ByVar keeps every var whole)")));
                }
            }
            Ok(layout)
        })();
        // Every rank starts from the same vars and the same sharding.
        let mine = match &check {
            Ok(layout) => format!("{} {} stage={} partition={}", state_hash(&vars)?, layout.key(), opts.stage.name(), opts.partition.name()),
            Err(e) => format!("error: {e}"),
        };
        let all = group.all_gather(mine.as_bytes())?;
        if let Some((q, a)) = all.iter().enumerate().find(|(_, a)| a.as_slice() != mine.as_bytes()) {
            let theirs = String::from_utf8_lossy(a);
            return Err(match check {
                Err(e) => e,
                Ok(_) => NnError::Dist(format!("rank {}: the vars or the sharding differ from rank {q}'s ({mine} against {theirs})", group.rank())),
            });
        }
        let layout = check?;
        let rank = group.rank();
        let pieces = layout.pieces(rank);
        let shard = layout.slice(&vars, rank, device)?;
        let mut sg = ParamGroups::new(&shard);
        for p in &pieces {
            let name = &layout.names[p.var];
            *sg.options_mut(name) = groups.options(p.var).clone();
            if let Some(m) = groups.multiplier(p.var) {
                sg.set_multiplier(name, m.clone());
            }
        }
        if let (Some(l), false) = (groups.seed_lr(), shard.is_empty()) {
            sg.set_seed_lr(l.to_vec());
        }
        let opt = make(&shard, sg);
        let params = if opts.stage == ShardStage::Parameters { Params::Shard(shard) } else { Params::Full(vars) };
        Ok(ShardedDataParallel { opts, layout, rank, device, params, opt, last_grads: 0 })
    }

    pub fn options(&self) -> &ShardOptions {
        &self.opts
    }

    pub fn layout(&self) -> &ShardLayout {
        &self.layout
    }

    /// This rank's optimizer (over its pieces).
    pub fn optimizer(&self) -> &O {
        &self.opt
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.opt
    }

    /// The run's record: the group's (`ProcessGroup::record`) and the sharding.
    pub fn record(&self, group: &ProcessGroup) -> String {
        format!("{} sharding=stage-{}-{} partition={}", group.record(), self.opts.stage.number(), self.opts.stage.name(), self.opts.partition.name())
    }

    fn check_group(&self, group: &ProcessGroup) -> Result<()> {
        if group.rank() != self.rank || group.world_size() != self.layout.world || group.device() != self.device {
            return Err(NnError::Dist(format!(
                "sharded for rank {} of {} on {:?}, called with rank {} of {} on {:?}",
                self.rank,
                self.layout.world,
                self.device,
                group.rank(),
                group.world_size(),
                group.device()
            )));
        }
        Ok(())
    }

    /// This rank's pieces of the vars, on its device.
    pub fn shard_params(&self) -> Result<VarMap> {
        match &self.params {
            Params::Full(v) => self.layout.slice(v, self.rank, self.device),
            Params::Shard(s) => Ok(s.clone()),
        }
    }

    /// The whole vars on every rank (FSDP's full parameters): at stage 3 gathered from every
    /// rank's pieces (a collective, every rank calls it), else the vars every rank keeps.
    pub fn full_params(&self, group: &mut ProcessGroup) -> Result<VarMap> {
        self.check_group(group)?;
        match &self.params {
            Params::Full(v) => Ok(v.clone()),
            Params::Shard(s) => {
                let mine: Vec<Option<Vec<f32>>> = s.tensors().iter().map(|t| Some(t.to_vec())).collect();
                let vals = gather_values(group, &self.layout, &mine)?;
                self.layout.assemble(vals, self.device)
            }
        }
    }

    /// One optimizer step: `grads`, then `apply`. Returns the number of micro-steps summed over
    /// every rank. The vars after it are `data_parallel_step`'s, bit for bit.
    pub fn step(&mut self, group: &mut ProcessGroup, micro_steps_per_rank: usize, how: Reduce, lr_scale: f64, micro: impl FnMut(&VarMap, usize) -> Tensor) -> Result<usize>
    where
        O: Optimizer,
    {
        let grads = self.grads(group, micro_steps_per_rank, how, micro)?;
        let n = grads.count();
        self.apply(group, grads, lr_scale)?;
        Ok(n)
    }

    /// This rank's pieces of a step's summed gradients: this rank's micro-steps on the whole vars
    /// (`micro(lifted, i)` builds micro-step i's loss, as for `data_parallel_grads`), then the
    /// all-reduce (stage 1) or the reduce-scatter (stages 2–3), in flat (rank, micro-step) order,
    /// scaled by `how`. For a caller that clips them (`ShardGrads::clip_grad_norm_per_seed`)
    /// before `apply`, as every rank must then do.
    pub fn grads(&mut self, group: &mut ProcessGroup, micro_steps_per_rank: usize, how: Reduce, mut micro: impl FnMut(&VarMap, usize) -> Tensor) -> Result<ShardGrads> {
        self.check_group(group)?;
        let mut sum = {
            let gathered;
            let full = match &self.params {
                Params::Full(v) => v,
                Params::Shard(_) => {
                    gathered = self.full_params(group)?;
                    &gathered
                }
            };
            let mut sum = group.grad_sum(full);
            for i in 0..micro_steps_per_rank {
                let lifted = full.lifted();
                let loss = micro(&lifted, i);
                let grads = loss.backward();
                sum.add(lifted.grads(&grads))?;
            }
            sum
            // Stage 3: the gathered vars are dropped here, before the reduction.
        };
        let pieces = self.layout.pieces(self.rank);
        let (vals, count, held) = match self.opts.stage {
            ShardStage::OptimizerState => {
                group.all_reduce_grads(&mut sum)?;
                let count = sum.count();
                let total = sum.finish_host(how);
                let held = total.iter().flatten().map(|v| 4 * v.len()).sum();
                (pieces.iter().map(|p| total[p.var].as_ref().map(|x| x[p.lo..p.hi].to_vec())).collect::<Vec<_>>(), count, held)
            }
            ShardStage::Gradients | ShardStage::Parameters => {
                let (mut vals, count) = group.reduce_scatter_grads(&mut sum, &self.layout.segments())?;
                for v in vals.iter_mut().flatten() {
                    scale(v, how, count);
                }
                let held = vals.iter().flatten().map(|v| 4 * v.len()).sum();
                (vals, count, held)
            }
        };
        if count == 0 {
            return Err(NnError::Dist("a step without micro-steps on any rank".into()));
        }
        self.last_grads = held;
        let grads = pieces.iter().zip(vals).map(|(p, v)| v.map(|v| Tensor::from_data(v, self.layout.piece_shape(p)).to(self.device))).collect();
        Ok(ShardGrads { layout: self.layout.clone(), rank: self.rank, pieces, grads, count })
    }

    /// This rank's optimizer steps its pieces with `grads`; at stages 1–2 the updated pieces are
    /// gathered back into every rank's whole vars (a collective).
    pub fn apply(&mut self, group: &mut ProcessGroup, grads: ShardGrads, lr_scale: f64) -> Result<()>
    where
        O: Optimizer,
    {
        self.check_group(group)?;
        if grads.rank != self.rank || grads.layout != self.layout {
            return Err(NnError::Dist("gradients of another rank or layout".into()));
        }
        let ShardedDataParallel { layout, rank, device, params, opt, .. } = self;
        match params {
            Params::Full(vars) => {
                let mut shard = layout.slice(vars, *rank, *device)?;
                opt.step(&mut shard, grads.grads, lr_scale);
                let mine: Vec<Option<Vec<f32>>> = shard.tensors().iter().map(|t| Some(t.to_vec())).collect();
                let vals = gather_values(group, layout, &mine)?;
                *vars = layout.assemble(vals, *device)?;
            }
            Params::Shard(shard) => opt.step(shard, grads.grads, lr_scale),
        }
        Ok(())
    }

    /// The bytes this rank holds: the vars between steps, the last step's summed gradient, the
    /// optimizer's state.
    pub fn memory(&self) -> ShardMemory
    where
        O: CheckpointOptimizer,
    {
        let params = match &self.params {
            Params::Full(v) => v.tensors().iter().map(|t| t.numel() * t.dtype().size_in_bytes()).sum(),
            Params::Shard(s) => s.tensors().iter().map(|t| t.numel() * t.dtype().size_in_bytes()).sum(),
        };
        let (sets, _) = self.opt.checkpoint_state();
        let optimizer_state = sets.iter().flat_map(|(_, s)| s.iter().flatten()).map(|t| t.numel() * t.dtype().size_in_bytes()).sum();
        ShardMemory { params, grads: self.last_grads, optimizer_state }
    }

    /// The whole optimizer state, gathered from every rank's pieces (a collective).
    fn consolidated_state(&self, group: &mut ProcessGroup) -> Result<Consolidated>
    where
        O: CheckpointOptimizer,
    {
        let (sets, t) = self.opt.checkpoint_state();
        let mut out = vec![];
        for (name, set) in sets {
            let mine: Vec<Option<Vec<f32>>> = set.iter().map(|x| x.as_ref().map(|t| f32_values(t, name)).transpose()).collect::<Result<_>>()?;
            let vals = gather_values(group, &self.layout, &mine)?;
            let whole = vals.into_iter().enumerate().map(|(v, x)| x.map(|x| Tensor::from_data(x, self.layout.shapes[v].clone()))).collect();
            out.push((name, whole));
        }
        Ok(Consolidated { kind: self.opt.checkpoint_kind(), sets: out, t })
    }

    /// A consolidated checkpoint: the whole vars and the whole optimizer state, gathered, written
    /// by `Checkpoint::save` (rank 0 writes, every rank verifies). Byte for byte the checkpoint
    /// data parallelism writes at the same step; either kind of run resumes it
    /// (`restore_consolidated`, `Checkpoint::restore`). Every rank calls it.
    pub fn save_consolidated(&self, group: &mut ProcessGroup, dir: &Path, step: u64, cursor: u64) -> Result<PathBuf>
    where
        O: CheckpointOptimizer,
    {
        let full = self.full_params(group)?;
        let state = self.consolidated_state(group)?;
        Checkpoint::save(group, dir, step, cursor, &full, &state)
    }

    /// Put a consolidated checkpoint (`Checkpoint::load_latest`; one written by data parallelism
    /// too) in place: this rank's pieces of its vars and optimizer state. Returns the steps taken.
    /// The optimizer must be of the checkpoint's kind (else refused).
    pub fn restore_consolidated(&mut self, c: Checkpoint) -> Result<u64>
    where
        O: CheckpointOptimizer,
    {
        let shapes: Vec<Vec<usize>> = c.vars.tensors().iter().map(|t| t.shape().to_vec()).collect();
        if c.vars.names() != self.layout.names || shapes != self.layout.shapes {
            return Err(NnError::Dist("the checkpoint holds other vars".into()));
        }
        if self.opt.checkpoint_kind() != c.meta.optimizer {
            return Err(NnError::Dist(format!(
                "the checkpoint holds {:?}'s state; this run's optimizer is {:?} (its state is not in the checkpoint: resume refused)",
                c.meta.optimizer,
                self.opt.checkpoint_kind()
            )));
        }
        let pieces = self.layout.pieces(self.rank);
        let sets = c.optimizer_state.iter().map(|set| pieces.iter().map(|p| set[p.var].as_ref().map(|t| piece_of(t, p, self.device))).collect()).collect();
        self.opt.restore_checkpoint_state(sets, c.meta.adam_t)?;
        let mut vars = VarMap::new();
        for (n, t) in c.vars.iter() {
            vars.insert(n, t.clone().to(self.device))?;
        }
        self.params = match self.opts.stage {
            ShardStage::Parameters => Params::Shard(self.layout.slice(&vars, self.rank, self.device)?),
            _ => Params::Full(vars),
        };
        Ok(c.meta.step)
    }

    /// A sharded checkpoint after `step` optimizer steps: every rank writes its own pieces of the
    /// vars and of the optimizer state into `step-<step>/` under `dir` (one folder every rank can
    /// write: one machine, or a filesystem the ranks share), and rank 0 the meta file with every
    /// file's sha256. Every rank calls it, with its loader cursor.
    ///
    /// 1. Every rank builds its files' bytes and their sha256, and the hash of its state; the
    ///    ranks all-gather them, so every rank builds the same meta file and its sha256.
    /// 2. Rank 0 makes a temporary folder; every rank writes its files there and reads them back.
    /// 3. Rank 0 writes the meta file, renames the folder into place, checks every file against
    ///    the meta file and broadcasts the meta file's sha256; every rank compares it with its own.
    /// 4. Rank 0 points `latest` at the folder; a barrier ends the checkpoint.
    ///
    /// A failure on any rank fails the checkpoint on every rank, and `latest` is not moved.
    pub fn save_sharded(&self, group: &mut ProcessGroup, dir: &Path, step: u64, cursor: u64) -> Result<PathBuf>
    where
        O: CheckpointOptimizer,
    {
        self.check_group(group)?;
        let r = self.rank;
        let kind = self.opt.checkpoint_kind();
        let (sets, t) = self.opt.checkpoint_state();
        let shard = self.shard_params()?;
        let maps: Vec<(&str, VarMap)> = sets.iter().map(|(n, s)| Ok((*n, set_map(&shard, s)?))).collect::<Result<_>>()?;
        let mut texts: Vec<(String, String)> = vec![(rank_file(r, "params"), shard.to_json()?)];
        for (n, m) in &maps {
            texts.push((rank_file(r, n), m.to_json()?));
        }
        let files: Vec<(String, String)> = texts.iter().map(|(f, x)| (f.clone(), crate::hash::sha256_hex(x.as_bytes()))).collect();
        let set_names: Vec<String> = maps.iter().map(|(n, _)| n.to_string()).collect();
        let st = rank_state(kind, &shard, &maps.iter().map(|(_, m)| m.clone()).collect::<Vec<_>>(), t)?;
        let rec = RankRecord { state: st, cursor, optimizer: kind.to_string(), adam_t: t, sets: set_names.clone(), files };
        let all = group.all_gather(serde_json::to_string(&rec).map_err(|e| NnError::Dist(format!("checkpoint record: {e}")))?.as_bytes())?;
        let mut ranks = vec![];
        let mut cursors = vec![];
        for (q, a) in all.iter().enumerate() {
            let x: RankRecord = serde_json::from_slice(a).map_err(|e| NnError::Dist(format!("rank {q}: a malformed checkpoint record: {e}")))?;
            if x.optimizer != kind || x.adam_t != t || x.sets != set_names {
                return Err(NnError::Dist(format!("checkpoint at step {step}: rank {q}'s optimizer ({} at t {}) differs from rank {r}'s ({kind} at t {t})", x.optimizer, x.adam_t)));
            }
            cursors.push(x.cursor);
            ranks.push(ShardRank { state: x.state, files: x.files });
        }
        let state = crate::hash::sha256_hex(ranks.iter().map(|x| x.state.as_str()).collect::<Vec<_>>().join(" ").as_bytes());
        let meta = ShardCheckpointMeta {
            format: SHARD_CHECKPOINT_FORMAT.into(),
            step,
            adam_t: t,
            optimizer: kind.to_string(),
            sets: set_names,
            world_size: group.world_size(),
            partition: self.opts.partition.name().to_string(),
            layout: self.layout.key(),
            group_key: group.key(),
            job_key: group.options().job_key.clone(),
            cursors,
            ranks,
            state,
        };
        let meta_text = serde_json::to_string_pretty(&meta).map_err(|e| NnError::Dist(format!("checkpoint meta: {e}")))?;
        let sha = crate::hash::sha256_hex(meta_text.as_bytes());
        let folder = Checkpoint::folder(dir, step);
        let tmp = folder.with_extension("tmp");
        let prep = if r == 0 { prepare(dir, &tmp) } else { Ok(()) };
        agree(group, &format!("checkpoint at step {step} not written"), prep)?;
        let wrote = (|| {
            for (f, x) in &texts {
                let p = tmp.join(f);
                write_atomic(&p, x.as_bytes())?;
                if std::fs::read(&p).map_err(io(&p))? != x.as_bytes() {
                    return Err(NnError::Dist(format!("{}: the bytes read back differ from those written", p.display())));
                }
            }
            Ok(())
        })();
        agree(group, &format!("checkpoint at step {step} not written"), wrote)?;
        let mut theirs = if r == 0 {
            match finish(&tmp, &folder, &meta_text) {
                Ok(h) => h.into_bytes(),
                Err(e) => format!("rank 0 could not write: {e}").into_bytes(),
            }
        } else {
            vec![]
        };
        group.broadcast(0, &mut theirs)?;
        let theirs = String::from_utf8_lossy(&theirs).to_string();
        let ack = if theirs == sha { Ok(()) } else { Err(NnError::Dist(format!("rank 0 wrote {theirs}, this rank's checkpoint is {sha}"))) };
        agree(group, &format!("checkpoint at step {step} not verified"), ack)?;
        if r == 0 {
            let name = folder.file_name().expect("a folder name").to_string_lossy().to_string();
            write_atomic(&dir.join("latest"), name.as_bytes())?;
        }
        group.barrier()?;
        Ok(folder)
    }

    /// The sharded checkpoint `latest` points at under `dir`, with this rank's files checked
    /// against the meta file; None when there is none. Every rank calls it. Refused: another world
    /// size, partition, layout, job key or group key, a meta file that differs between ranks, and
    /// any file whose sha256 or state hash is not the recorded one (on every rank).
    pub fn load_sharded_latest(&self, group: &mut ProcessGroup, dir: &Path) -> Result<Option<ShardCheckpoint>> {
        self.check_group(group)?;
        let latest = dir.join("latest");
        let mine: Result<Option<(PathBuf, String, String)>> = if latest.exists() {
            (|| {
                let name = std::fs::read_to_string(&latest).map_err(io(&latest))?;
                let folder = dir.join(name.trim());
                let mp = folder.join("meta.json");
                let text = std::fs::read_to_string(&mp).map_err(io(&mp))?;
                Ok(Some((folder, crate::hash::sha256_hex(text.as_bytes()), text)))
            })()
        } else {
            Ok(None)
        };
        let said = match &mine {
            Ok(Some((_, h, _))) => h.clone(),
            Ok(None) => "none".to_string(),
            Err(e) => format!("error: {e}"),
        };
        let all = group.all_gather(said.as_bytes())?;
        if let Some((q, a)) = all.iter().enumerate().find(|(_, a)| a.as_slice() != said.as_bytes()) {
            return Err(NnError::Dist(format!("rank {}: loaded checkpoint {said}, rank {q} {} (resume refused)", group.rank(), String::from_utf8_lossy(a))));
        }
        let Some((folder, sha256, text)) = mine? else { return Ok(None) };
        let meta: ShardCheckpointMeta = serde_json::from_str(&text).map_err(io(&folder.join("meta.json")))?;
        let refusal = if meta.format != SHARD_CHECKPOINT_FORMAT {
            Some(format!("format {:?}, expected {SHARD_CHECKPOINT_FORMAT:?}", meta.format))
        } else if meta.world_size != group.world_size() || meta.job_key != group.options().job_key {
            Some(format!(
                "the checkpoint at step {} was taken at world size {} with job key {:?}; this run has world size {} and job key {:?} (a resume keeps both: resume refused)",
                meta.step,
                meta.world_size,
                meta.job_key,
                group.world_size(),
                group.options().job_key
            ))
        } else if meta.group_key != group.key() {
            Some(format!("the checkpoint at step {} was taken under the key {:?}; this run's key is {:?} (another platform or device: resume refused)", meta.step, meta.group_key, group.key()))
        } else if meta.partition != self.opts.partition.name() || meta.layout != self.layout.key() {
            Some(format!("the checkpoint at step {} holds {} pieces of another layout ({}); this run's are {} ({}) (resume refused)", meta.step, meta.partition, meta.layout, self.opts.partition.name(), self.layout.key()))
        } else if meta.cursors.len() != group.world_size() || meta.ranks.len() != group.world_size() {
            Some(format!("the checkpoint holds {} cursors and {} ranks' files for {} ranks (resume refused)", meta.cursors.len(), meta.ranks.len(), group.world_size()))
        } else {
            None
        };
        if let Some(why) = refusal {
            return Err(NnError::Dist(why));
        }
        let loaded = read_rank(&folder, &meta, self.rank);
        agree(group, "the sharded checkpoint", loaded.as_ref().map(|_| ()).map_err(|e| NnError::Dist(e.message().to_string())))?;
        let (params, sets) = loaded?;
        Ok(Some(ShardCheckpoint { meta, sha256, rank: self.rank, params, sets }))
    }

    /// Put a sharded checkpoint (`load_sharded_latest`) in place; returns the steps taken. At
    /// stages 1–2 the whole vars are gathered from every rank's pieces (a collective: every rank
    /// calls it). The optimizer must be of the checkpoint's kind (else refused).
    pub fn restore_sharded(&mut self, group: &mut ProcessGroup, c: ShardCheckpoint) -> Result<u64>
    where
        O: CheckpointOptimizer,
    {
        self.check_group(group)?;
        let pieces = self.layout.pieces(self.rank);
        let want: Vec<(String, Vec<usize>)> = pieces.iter().map(|p| (self.layout.names[p.var].clone(), self.layout.piece_shape(p))).collect();
        let got: Vec<(String, Vec<usize>)> = c.params.iter().map(|(n, t)| (n.to_string(), t.shape().to_vec())).collect();
        let ok = if self.opt.checkpoint_kind() != c.meta.optimizer {
            Err(NnError::Dist(format!(
                "the checkpoint holds {:?}'s state; this run's optimizer is {:?} (its state is not in the checkpoint: resume refused)",
                c.meta.optimizer,
                self.opt.checkpoint_kind()
            )))
        } else if c.rank != self.rank || want != got {
            Err(NnError::Dist(format!("rank {}'s pieces given to rank {} (resume refused)", c.rank, self.rank)))
        } else {
            Ok(())
        };
        agree(group, "the sharded resume", ok)?;
        let sets = c.sets.iter().map(|m| pieces.iter().map(|p| m.get(&self.layout.names[p.var]).map(|t| t.clone().to(self.device))).collect()).collect();
        self.opt.restore_checkpoint_state(sets, c.meta.adam_t)?;
        let mut shard = VarMap::new();
        for (n, t) in c.params.iter() {
            shard.insert(n, t.clone().to(self.device))?;
        }
        self.params = match self.opts.stage {
            ShardStage::Parameters => Params::Shard(shard),
            _ => {
                let mine: Vec<Option<Vec<f32>>> = shard.tensors().iter().map(|t| Some(t.to_vec())).collect();
                let vals = gather_values(group, &self.layout, &mine)?;
                Params::Full(self.layout.assemble(vals, self.device)?)
            }
        };
        Ok(c.meta.step)
    }
}

/// `f32_values`'s dtype check alone.
fn f32_values_dtype(t: &Tensor, name: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(NnError::Dist(format!("var {name:?}: a {} var (sharding takes f32 vars)", t.dtype().name())));
    }
    Ok(())
}

/// The whole optimizer state gathered from every rank, for `Checkpoint::save`.
struct Consolidated {
    kind: &'static str,
    sets: Vec<(&'static str, Vec<Option<Tensor>>)>,
    t: u64,
}

impl CheckpointOptimizer for Consolidated {
    fn checkpoint_kind(&self) -> &'static str {
        self.kind
    }

    fn checkpoint_state(&self) -> (Vec<(&'static str, Vec<Option<Tensor>>)>, u64) {
        (self.sets.clone(), self.t)
    }

    fn restore_checkpoint_state(&mut self, _sets: Vec<Vec<Option<Tensor>>>, _t: u64) -> Result<()> {
        Err(NnError::Dist("a consolidated state is written, never restored into".into()))
    }
}

/// The sharded checkpoint format's name and version.
pub const SHARD_CHECKPOINT_FORMAT: &str = "dist-shard-checkpoint-1";

/// What a sharded checkpoint records besides the tensors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardCheckpointMeta {
    pub format: String,
    /// The optimizer steps taken.
    pub step: u64,
    /// The optimizer's step count (Adam's; 0 for SGD).
    pub adam_t: u64,
    /// The optimizer whose state the checkpoint holds ("adam", "sgd") and its sets' names.
    pub optimizer: String,
    pub sets: Vec<String>,
    pub world_size: usize,
    /// The partition's name and the layout's key (`ShardLayout::key`).
    pub partition: String,
    pub layout: String,
    pub group_key: String,
    pub job_key: String,
    /// Each rank's loader cursor, in rank order.
    pub cursors: Vec<u64>,
    /// Each rank's state hash and files, in rank order.
    pub ranks: Vec<ShardRank>,
    /// sha256 of the ranks' state hashes.
    pub state: String,
}

/// One rank's part of a sharded checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardRank {
    /// sha256 of its pieces of the vars, of each optimizer set and of the step count.
    pub state: String,
    /// (file, sha256) of each of its files.
    pub files: Vec<(String, String)>,
}

/// What a rank says about its part of a checkpoint (all-gathered).
#[derive(Serialize, Deserialize)]
struct RankRecord {
    state: String,
    cursor: u64,
    optimizer: String,
    adam_t: u64,
    sets: Vec<String>,
    files: Vec<(String, String)>,
}

/// A loaded sharded checkpoint: one rank's pieces.
#[derive(Debug)]
pub struct ShardCheckpoint {
    pub meta: ShardCheckpointMeta,
    /// sha256 of the meta file (which holds every other file's sha256).
    pub sha256: String,
    /// The rank whose pieces these are.
    pub rank: usize,
    /// Its pieces of the vars (under the vars' names).
    pub params: VarMap,
    /// Its pieces of each optimizer set, in `meta.sets`' order (a var without state is absent).
    pub sets: Vec<VarMap>,
}

impl ShardCheckpoint {
    /// Rank `rank`'s loader cursor.
    pub fn cursor(&self, rank: usize) -> Option<u64> {
        self.meta.cursors.get(rank).copied()
    }
}

fn rank_file(rank: usize, what: &str) -> String {
    format!("rank-{rank:05}-{what}.json")
}

/// The hash of a rank's state: its pieces of the vars, the kind, every set and the step count.
fn rank_state(kind: &str, vars: &VarMap, sets: &[VarMap], t: u64) -> Result<String> {
    let mut text = format!("{} {kind}", state_hash(vars)?);
    for m in sets {
        text.push_str(&format!(" {}", state_hash(m)?));
    }
    text.push_str(&format!(" {t}"));
    Ok(crate::hash::sha256_hex(text.as_bytes()))
}

/// A fresh temporary folder `tmp` under `dir`.
fn prepare(dir: &Path, tmp: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    if tmp.exists() {
        std::fs::remove_dir_all(tmp).map_err(io(tmp))?;
    }
    std::fs::create_dir_all(tmp).map_err(io(tmp))
}

/// Rank 0's last part of a sharded checkpoint: the meta file, the rename, and a check of every
/// rank's files against it; returns the meta file's sha256 as written.
fn finish(tmp: &Path, folder: &Path, meta_text: &str) -> Result<String> {
    write_atomic(&tmp.join("meta.json"), meta_text.as_bytes())?;
    if folder.exists() {
        std::fs::remove_dir_all(folder).map_err(io(folder))?;
    }
    std::fs::rename(tmp, folder).map_err(io(folder))?;
    let mp = folder.join("meta.json");
    let text = std::fs::read_to_string(&mp).map_err(io(&mp))?;
    let meta: ShardCheckpointMeta = serde_json::from_str(&text).map_err(io(&mp))?;
    for r in 0..meta.ranks.len() {
        read_rank(folder, &meta, r)?;
    }
    Ok(crate::hash::sha256_hex(text.as_bytes()))
}

/// Rank `rank`'s files of a sharded checkpoint, checked against the meta file: its pieces of the
/// vars and of each optimizer set.
fn read_rank(folder: &Path, meta: &ShardCheckpointMeta, rank: usize) -> Result<(VarMap, Vec<VarMap>)> {
    let rec = meta.ranks.get(rank).ok_or_else(|| NnError::Dist(format!("the checkpoint holds no files for rank {rank}")))?;
    let mut maps = vec![];
    for what in std::iter::once("params").chain(meta.sets.iter().map(String::as_str)) {
        let f = rank_file(rank, what);
        let p = folder.join(&f);
        let text = std::fs::read_to_string(&p).map_err(io(&p))?;
        let want = rec.files.iter().find(|(n, _)| *n == f).map(|(_, h)| h.as_str());
        let got = crate::hash::sha256_hex(text.as_bytes());
        if want != Some(got.as_str()) {
            return Err(NnError::Dist(format!("{}: sha256 {got}, the meta file records {want:?}", p.display())));
        }
        maps.push(VarMap::from_json(&text)?);
    }
    let params = maps.remove(0);
    let st = rank_state(&meta.optimizer, &params, &maps, meta.adam_t)?;
    if st != rec.state {
        return Err(NnError::Dist(format!("{}: rank {rank}'s state hashes to {st}, the meta file records {}", folder.display(), rec.state)));
    }
    Ok((params, maps))
}
