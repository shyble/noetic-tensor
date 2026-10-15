//! The fast mode: data-parallel and sharded training over the same TCP mesh, with the order of
//! the additions chosen for speed instead of for reproducibility.
//!
//! - **Local accumulation first.** Each rank adds its own micro-steps in order (per-rank-first),
//!   then the ranks' sums are reduced; the bits depend on the world size.
//! - **Buckets.** The gradients go in buckets of at most `FastOptions::bucket_bytes` bytes (torch
//!   DDP's `bucket_cap_mb`; a var larger than that is a bucket of its own), filled with the vars
//!   in reverse order (the order in which a backward pass usually finishes them). Every rank cuts
//!   the same buckets and reduces them in the same order.
//! - **A ring.** Each bucket is reduced by a ring: a reduce-scatter in W − 1 steps (each rank
//!   passes a segment's running sum to the next rank, which adds its own), then, for an
//!   all-reduce, an all-gather in W − 1 steps. Each element of the result is added once, by its
//!   segment's chain, so every rank holds the same bits and the vars stay in sync; but the order
//!   of the additions depends on the world size and on where the segments are cut, so it is not
//!   the deterministic mode's flat rank order. For sharded training the segments are the ranks'
//!   pieces, and the reduce-scatter alone gives each rank its own.
//! - **Overlap.** With `FastOptions::overlap`, the reduction of a bucket starts on a
//!   communication thread during the backward pass of the rank's last micro-step, as soon as
//!   every gradient in the bucket is final; the backward pass goes on meanwhile.
//!
//! A fast-mode group carries `mode=fast` in its key, with its world size and buckets, and its
//! record says "fast mode, not reproducible bit for bit". It is never pooled with, or compared
//! byte for byte against, the deterministic mode: a job mixing modes is refused at the
//! rendezvous, and a checkpoint keyed by one mode is refused by the other.

use super::group::ProcessGroup;
use super::reduce::{fold_var, scale, Part, Reduce};
use super::wire::{self, Dec, Enc, Op};
use crate::error::{NnError, Result};
use crate::nn::VarMap;
use crate::tensor::{DType, Tensor};
use std::collections::HashMap;
use std::net::{Shutdown, TcpStream};
use std::sync::{mpsc, Arc};

/// The fast mode's options. Non-exhaustive: made with `FastOptions::new` (or `default`) and the
/// `with_*` methods.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FastOptions {
    /// The largest bucket, in bytes of f32 gradients (25 MiB by default, torch DDP's default
    /// `bucket_cap_mb`); a var larger than this is a bucket of its own. Every rank must use the
    /// same value (it is part of the mode the rendezvous compares, and of the key).
    pub bucket_bytes: usize,
    /// Whether the buckets are reduced during the backward pass of the rank's last micro-step
    /// (on by default). The bits do not depend on it.
    pub overlap: bool,
}

impl Default for FastOptions {
    fn default() -> Self {
        FastOptions { bucket_bytes: 25 << 20, overlap: true }
    }
}

impl FastOptions {
    pub fn new() -> FastOptions {
        FastOptions::default()
    }

    pub fn with_bucket_bytes(mut self, bucket_bytes: usize) -> FastOptions {
        self.bucket_bytes = bucket_bytes;
        self
    }

    pub fn with_overlap(mut self, overlap: bool) -> FastOptions {
        self.overlap = overlap;
        self
    }

    /// The options' part of the key: `bucket=<bytes> overlap=on|off`.
    pub(crate) fn key_part(&self) -> String {
        format!("bucket={} overlap={}", self.bucket_bytes, if self.overlap { "on" } else { "off" })
    }
}

/// The mode of a process group (`GroupOptions::mode`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DistMode {
    /// The flat rank order: W ranks with k micro-steps give one process's W·k, bit for bit.
    #[default]
    Deterministic,
    /// The fast mode: per-rank sums, buckets, a ring, overlap with the backward pass; its own key;
    /// not reproducible bit for bit.
    Fast(FastOptions),
}

impl DistMode {
    /// The fast mode with the default options.
    pub fn fast() -> DistMode {
        DistMode::Fast(FastOptions::default())
    }

    pub fn is_fast(&self) -> bool {
        matches!(self, DistMode::Fast(_))
    }

    /// The mode's part of the hello (None for the deterministic mode, whose hello is 0.5.0's).
    pub(crate) fn hello(&self) -> Option<String> {
        match self {
            DistMode::Deterministic => None,
            DistMode::Fast(f) => Some(format!("mode=fast order=ring {}", f.key_part())),
        }
    }

    /// Refuse options that cannot work.
    pub(crate) fn check(&self) -> Result<()> {
        match self {
            DistMode::Fast(f) if f.bucket_bytes < 4 => Err(NnError::Dist(format!("a fast-mode bucket of {} bytes (at least one f32)", f.bucket_bytes))),
            _ => Ok(()),
        }
    }
}

/// What a fast reduction leaves on each rank.
pub(crate) enum Target<'a> {
    /// The whole sum, on every rank.
    AllReduce,
    /// Rank q's pieces (var, first element, end) of the sum, on rank q only: `pieces[q]`.
    Scatter(&'a [Vec<(usize, usize, usize)>]),
}

/// One bucket: its vars, their offsets in its flat buffer, and per rank the flat ranges of the
/// segment that rank owns at the end of the reduce-scatter.
struct Bucket {
    vars: Vec<usize>,
    offsets: Vec<usize>,
    len: usize,
    segs: Vec<Vec<(usize, usize)>>,
}

/// The buckets of a step, the same on every rank.
struct Plan {
    buckets: Vec<Bucket>,
    /// Per var: its bucket and its place in the bucket.
    of_var: Vec<(usize, usize)>,
}

impl Plan {
    fn new(sizes: &[usize], bucket_bytes: usize, world: usize, target: &Target) -> Result<Plan> {
        if let Target::Scatter(pieces) = target {
            if let Some(&(v, lo, hi)) = pieces.iter().flatten().find(|(v, lo, hi)| *v >= sizes.len() || lo > hi || *hi > sizes[*v]) {
                return Err(NnError::Dist(format!("a piece ({v}, {lo}..{hi}) outside the vars")));
            }
        }
        let cap = (bucket_bytes / 4).max(1);
        let mut buckets: Vec<Bucket> = vec![];
        let mut of_var = vec![(0, 0); sizes.len()];
        for v in (0..sizes.len()).rev() {
            let n = sizes[v];
            let fits = buckets.last().is_some_and(|b| b.len + n <= cap);
            if !fits {
                buckets.push(Bucket { vars: vec![], offsets: vec![], len: 0, segs: vec![] });
            }
            let bi = buckets.len() - 1;
            let b = &mut buckets[bi];
            of_var[v] = (bi, b.vars.len());
            b.vars.push(v);
            b.offsets.push(b.len);
            b.len += n;
        }
        for (bi, b) in buckets.iter_mut().enumerate() {
            b.segs = match target {
                Target::AllReduce => (0..world).map(|q| vec![(b.len * q / world, b.len * (q + 1) / world)]).collect(),
                Target::Scatter(pieces) => {
                    if pieces.len() != world {
                        return Err(NnError::Dist(format!("pieces for {} ranks in a world of {world}", pieces.len())));
                    }
                    pieces
                        .iter()
                        .map(|own| {
                            own.iter()
                                .filter(|(v, _, _)| of_var.get(*v).is_some_and(|(b2, _)| *b2 == bi))
                                .map(|&(v, lo, hi)| {
                                    let off = b.offsets[of_var[v].1];
                                    (off + lo, off + hi)
                                })
                                .collect()
                        })
                        .collect()
                }
            };
        }
        Ok(Plan { buckets, of_var })
    }

    /// Bucket `b`'s data from this rank's sums (taken out of `acc`), with this rank's count.
    fn pack(&self, b: usize, acc: &mut Part, count: usize) -> Data {
        let bk = &self.buckets[b];
        let mut buf = vec![0.0f32; bk.len];
        let mut present = vec![0u8; bk.vars.len()];
        for (i, &v) in bk.vars.iter().enumerate() {
            if let Some(x) = acc[v].take() {
                buf[bk.offsets[i]..bk.offsets[i] + x.len()].copy_from_slice(&x);
                present[i] = 1;
            }
        }
        Data { buf, present, count: count as u64 }
    }
}

/// A bucket's values (absent vars as zeros), which of its vars have a gradient on any rank so
/// far, and the micro-steps summed.
struct Data {
    buf: Vec<f32>,
    present: Vec<u8>,
    count: u64,
}

/// A gradient's host values, checked as `GradSum::add` checks them.
fn host_grad(t: &Tensor, shape: &[usize], v: usize) -> Result<Vec<f32>> {
    if t.dtype() != DType::F32 {
        return Err(NnError::Dist(format!("var {v}: a {} gradient (the fast mode takes f32 gradients)", t.dtype().name())));
    }
    if t.shape() != shape {
        return Err(NnError::Dist(format!("var {v}: a gradient of shape {:?} for a var of shape {shape:?}", t.shape())));
    }
    Ok(t.to_vec())
}

/// One step's gradients in the fast mode: this rank's `k` micro-steps (`micro(lifted, i)` builds
/// micro-step i's loss), summed locally in order, then reduced bucket by bucket by the ring,
/// overlapped with the last micro-step's backward pass when the options say so; scaled by `how`.
/// Returns, for `Target::AllReduce`, the sum per var (None: no micro-step on any rank used it);
/// for `Target::Scatter`, this rank's pieces in order; and the micro-steps summed over every rank.
pub(crate) fn fast_grads(group: &mut ProcessGroup, vars: &VarMap, k: usize, how: Reduce, target: Target, mut micro: impl FnMut(&VarMap, usize) -> Tensor) -> Result<(Part, usize)> {
    let DistMode::Fast(f) = group.mode().clone() else {
        return Err(NnError::Dist("a fast-mode step on a group in the deterministic mode".into()));
    };
    let shapes: Vec<Vec<usize>> = vars.tensors().iter().map(|t| t.shape().to_vec()).collect();
    let sizes: Vec<usize> = shapes.iter().map(|s| s.iter().product()).collect();
    if sizes.is_empty() {
        return Err(NnError::Dist("a fast-mode step without vars".into()));
    }
    let (rank, world) = (group.rank(), group.world_size());
    let plan = Arc::new(Plan::new(&sizes, f.bucket_bytes, world, &target)?);
    let gather = matches!(target, Target::AllReduce);
    let n = sizes.len();
    // The ring's links (handles on the group's connections to the previous and the next rank)
    // and a sequence number per bucket, taken from the group before the communication thread
    // starts; the group is not used again until that thread has ended.
    let links = if world > 1 {
        let (prev, next) = ((rank + world - 1) % world, (rank + 1) % world);
        let clone = |s: &std::net::TcpStream, q: usize| s.try_clone().map_err(|e| NnError::Dist(format!("rank {rank}: the link to rank {q}: {e}")));
        Some(Links { prev: clone(group.peer(prev), prev)?, next: clone(group.peer(next), next)? })
    } else {
        None
    };
    let seqs: Vec<u64> = (0..plan.buckets.len()).map(|_| if world > 1 { group.next_seq() } else { 0 }).collect();
    let reduced = {
        let (btx, brx) = mpsc::channel::<(usize, Data)>();
        let comm = {
            let plan = Arc::clone(&plan);
            std::thread::spawn(move || reduce_buckets(rank, world, links, seqs, &plan, gather, brx))
        };
        let plan = &plan;
        let computed = (|| -> Result<()> {
            let mut acc: Part = vec![None; n];
            let mut left: Vec<usize> = plan.buckets.iter().map(|b| b.vars.len()).collect();
            // Bucket `b` is complete on this rank: hand it to the communication thread (a closed
            // channel means that thread failed; its error is reported after the join).
            let send = |b: usize, acc: &mut Part| {
                let _ = btx.send((b, plan.pack(b, acc, k)));
            };
            for i in 0..k {
                let lifted = vars.lifted();
                let loss = micro(&lifted, i);
                let last = i + 1 == k;
                if last && f.overlap {
                    let ids: HashMap<u64, usize> = lifted.tensors().iter().enumerate().filter_map(|(v, t)| t.leaf_id().map(|id| (id, v))).collect();
                    let mut ready = vec![false; n];
                    let mut err: Option<NnError> = None;
                    let mut on_ready = |id: u64, g: &Tensor| {
                        let Some(&v) = ids.get(&id) else { return };
                        if err.is_some() || ready[v] {
                            return;
                        }
                        match host_grad(g, &shapes[v], v) {
                            Ok(x) => fold_var(&mut acc[v], Some(&x), i > 0),
                            Err(e) => {
                                err = Some(e);
                                return;
                            }
                        }
                        ready[v] = true;
                        let b = plan.of_var[v].0;
                        left[b] -= 1;
                        if left[b] == 0 {
                            send(b, &mut acc);
                        }
                    };
                    drop(loss.backward_notify(&mut on_ready));
                    if let Some(e) = err {
                        return Err(e);
                    }
                    // The vars this micro-step did not use count as zeros (as in the
                    // deterministic sum); their buckets go now.
                    for v in 0..n {
                        if !ready[v] {
                            fold_var(&mut acc[v], None, i > 0);
                            let b = plan.of_var[v].0;
                            left[b] -= 1;
                            if left[b] == 0 {
                                send(b, &mut acc);
                            }
                        }
                    }
                } else {
                    let grads = loss.backward();
                    for (v, g) in lifted.grads(&grads).into_iter().enumerate() {
                        let x = match g {
                            Some(g) => Some(host_grad(&g, &shapes[v], v)?),
                            None => None,
                        };
                        fold_var(&mut acc[v], x.as_deref(), i > 0);
                    }
                    if last {
                        for b in 0..plan.buckets.len() {
                            send(b, &mut acc);
                        }
                    }
                }
            }
            if k == 0 {
                for b in 0..plan.buckets.len() {
                    send(b, &mut acc);
                }
            }
            Ok(())
        })();
        drop(btx);
        let reduced = comm.join().expect("the fast mode's communication thread");
        computed?;
        reduced?
    };
    let count = reduced[0].count;
    if let Some(d) = reduced.iter().find(|d| d.count != count) {
        return Err(NnError::Dist(format!("rank {rank}: buckets summed {count} and {} micro-steps", d.count)));
    }
    let count = count as usize;
    let value = |v: usize, lo: usize, hi: usize| -> Option<Vec<f32>> {
        let (b, i) = plan.of_var[v];
        let d = &reduced[b];
        (d.present[i] != 0).then(|| {
            let off = plan.buckets[b].offsets[i];
            let mut x = d.buf[off + lo..off + hi].to_vec();
            scale(&mut x, how, count);
            x
        })
    };
    let out = match target {
        Target::AllReduce => (0..n).map(|v| value(v, 0, sizes[v])).collect(),
        Target::Scatter(pieces) => pieces[rank].iter().map(|&(v, lo, hi)| value(v, lo, hi)).collect(),
    };
    Ok((out, count))
}

/// Handles on the group's connections to the previous rank (received from) and the next rank
/// (sent to) of the ring.
struct Links {
    prev: TcpStream,
    next: TcpStream,
}

/// The communication thread: the buckets, as the backward pass hands them over, reduced by the
/// ring in bucket order (every rank in the same order), bucket b under sequence number `seqs[b]`.
fn reduce_buckets(rank: usize, world: usize, links: Option<Links>, seqs: Vec<u64>, plan: &Plan, gather: bool, rx: mpsc::Receiver<(usize, Data)>) -> Result<Vec<Data>> {
    let nb = plan.buckets.len();
    let mut waiting: Vec<Option<Data>> = (0..nb).map(|_| None).collect();
    let next_bucket = |b: usize, waiting: &mut Vec<Option<Data>>| -> Result<Data> {
        while waiting[b].is_none() {
            let (i, d) = rx.recv().map_err(|_| NnError::Dist(format!("rank {rank}: the backward pass stopped before bucket {b} was complete")))?;
            waiting[i] = Some(d);
        }
        Ok(waiting[b].take().expect("bucket"))
    };
    let Some(Links { mut prev, next }) = links else {
        return (0..nb).map(|b| next_bucket(b, &mut waiting)).collect();
    };
    let stop = next.try_clone().map_err(|e| NnError::Dist(format!("rank {rank}: the link to the next rank: {e}")))?;
    // The sending side of the ring: messages to the next rank, in order, on a thread of its own
    // while this thread receives from the previous rank (so no rank blocks on a full socket).
    let (stx, srx) = mpsc::channel::<(Op, u64, Vec<u8>)>();
    let sender = std::thread::spawn(move || -> Result<()> {
        let mut out = next;
        for (op, seq, p) in srx {
            wire::send(&mut out, op, rank, seq, &p)?;
        }
        Ok(())
    });
    let res = (|| -> Result<Vec<Data>> {
        let mut done = Vec::with_capacity(nb);
        for b in 0..nb {
            let mut d = next_bucket(b, &mut waiting)?;
            ring(rank, world, &mut prev, &stx, seqs[b], b, &plan.buckets[b], &mut d, gather)?;
            done.push(d);
        }
        Ok(done)
    })();
    drop(stx);
    if res.is_err() {
        // Unblock the sender (and tell the next rank at once).
        let _ = stop.shutdown(Shutdown::Both);
    }
    let sent = sender.join().expect("the fast mode's sending thread");
    let done = res?;
    sent?;
    Ok(done)
}

/// One bucket through the ring: the reduce-scatter (W − 1 steps; rank r ends with segment r
/// whole), then with `gather` the all-gather (W − 1 steps). The presence of each var and the
/// micro-step count travel with every segment's chain.
#[allow(clippy::too_many_arguments)]
fn ring(r: usize, w: usize, from: &mut TcpStream, out: &mpsc::Sender<(Op, u64, Vec<u8>)>, seq: u64, b: usize, bk: &Bucket, d: &mut Data, gather: bool) -> Result<()> {
    let prev = (r + w - 1) % w;
    let mut pres: Vec<Vec<u8>> = vec![d.present.clone(); w];
    let mut cnt: Vec<u64> = vec![d.count; w];
    let seg_len = |q: usize| -> usize { bk.segs[q].iter().map(|(lo, hi)| hi - lo).sum() };
    let encode = |t: usize, q: usize, cnt: u64, pres: &[u8], buf: &[f32]| -> Vec<u8> {
        let mut e = Enc::default();
        e.u64(b as u64).u64(t as u64).u64(q as u64).u64(cnt).bytes(pres);
        for &(lo, hi) in &bk.segs[q] {
            e.f32s(&buf[lo..hi]);
        }
        e.0
    };
    let gone = || NnError::Dist(format!("rank {r}: the fast mode's sending thread stopped (a link failed)"));
    // A received segment: (count, presence, values), checked against what is due.
    let decode = |payload: &[u8], phase: &str, t: usize, q: usize| -> Result<(u64, Vec<u8>, Vec<f32>)> {
        let mut dec = Dec::new(payload);
        let (gb, gt, gq) = (dec.u64()? as usize, dec.u64()? as usize, dec.u64()? as usize);
        if (gb, gt, gq) != (b, t, q) {
            return Err(NnError::Dist(format!("rank {r}: fast ring {phase}: bucket {gb} step {gt} segment {gq} where bucket {b} step {t} segment {q} was due")));
        }
        let c = dec.u64()?;
        let p = dec.bytes()?.to_vec();
        if p.len() != bk.vars.len() {
            return Err(NnError::Dist(format!("rank {r}: fast ring {phase}: {} presence flags for {} vars", p.len(), bk.vars.len())));
        }
        let x = dec.f32s(seg_len(q))?;
        dec.done()?;
        Ok((c, p, x))
    };
    for t in 0..w - 1 {
        let (qs, qr) = ((r + 2 * w - t - 1) % w, (r + 2 * w - t - 2) % w);
        out.send((Op::RingReduce, seq, encode(t, qs, cnt[qs], &pres[qs], &d.buf))).map_err(|_| gone())?;
        let payload = wire::recv(from, Op::RingReduce, r, prev, seq)?;
        let (c, p, x) = decode(&payload, "reduce", t, qr)?;
        let mut it = x.into_iter();
        for &(lo, hi) in &bk.segs[qr] {
            for (a, y) in d.buf[lo..hi].iter_mut().zip(it.by_ref()) {
                *a = y + *a;
            }
        }
        for (a, y) in pres[qr].iter_mut().zip(&p) {
            *a |= *y;
        }
        cnt[qr] += c;
    }
    d.present = pres[r].clone();
    d.count = cnt[r];
    if gather {
        for t in 0..w - 1 {
            let (qs, qr) = ((r + w - t) % w, (r + 2 * w - t - 1) % w);
            out.send((Op::RingGather, seq, encode(t, qs, cnt[qs], &pres[qs], &d.buf))).map_err(|_| gone())?;
            let payload = wire::recv(from, Op::RingGather, r, prev, seq)?;
            let (c, p, x) = decode(&payload, "gather", t, qr)?;
            let mut it = x.into_iter();
            for &(lo, hi) in &bk.segs[qr] {
                for (a, y) in d.buf[lo..hi].iter_mut().zip(it.by_ref()) {
                    *a = y;
                }
            }
            pres[qr] = p;
            cnt[qr] = c;
        }
        // Every segment's chain saw every rank: the presence and the count agree.
        if pres.iter().any(|p| *p != d.present) || cnt.iter().any(|c| *c != d.count) {
            return Err(NnError::Dist(format!("rank {r}: the fast ring's segments of bucket {b} disagree on presence or count")));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn plan_segments_for_tests(sizes: &[usize], bucket_bytes: usize, world: usize, pieces: Option<&[Vec<(usize, usize, usize)>]>) -> Vec<(Vec<usize>, Vec<Vec<(usize, usize)>>)> {
    let target = match pieces {
        None => Target::AllReduce,
        Some(p) => Target::Scatter(p),
    };
    Plan::new(sizes, bucket_bytes, world, &target).unwrap().buckets.into_iter().map(|b| (b.vars, b.segs)).collect()
}
