//! Coordinated checkpoints of data-parallel training: the vars, the optimizer's state and every
//! rank's loader cursor, taken by every rank at the same step.
//!
//! Every rank hashes its vars and optimizer state, and the hashes must agree (the ranks took
//! identical steps). Rank 0, the single writer, writes `step-<step>/` (vars, moments, a meta file
//! with each file's sha256); every rank builds the same bytes itself, and acknowledges only once
//! rank 0's broadcast sha256 of what it wrote equals its own. Then `latest` points at the folder
//! and a barrier ends the checkpoint: once any rank is past it, the checkpoint is on disk whole
//! and verified. Other nodes may keep copies, kept only if they hold the same bytes. A resume
//! checks every hash, that every rank loaded the same checkpoint, the cursors, and refuses
//! another world size, job key or group key (platform and device).

use super::ddp::state_hash;
use super::group::ProcessGroup;
use crate::error::{NnError, Result};
use crate::nn::{Adam, Sgd, VarMap};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The checkpoint format's name and version.
pub const CHECKPOINT_FORMAT: &str = "dist-checkpoint-1";

/// What a checkpoint records besides the tensors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub format: String,
    /// The optimizer steps taken.
    pub step: u64,
    /// The optimizer's step count (Adam's; 0 for SGD).
    pub adam_t: u64,
    /// The optimizer whose state the checkpoint holds (`CheckpointOptimizer::checkpoint_kind`):
    /// "adam" (written as in 0.3.0: the field is left out) or "sgd".
    #[serde(default = "adam_kind", skip_serializing_if = "is_adam")]
    pub optimizer: String,
    pub world_size: usize,
    /// The reproducibility key (`ProcessGroup::key`; the world size is recorded above) and the
    /// job key.
    pub group_key: String,
    pub job_key: String,
    /// Each rank's loader cursor, in rank order.
    pub cursors: Vec<u64>,
    /// sha256 of the vars, the optimizer's state and step count (`state_hash` of each, then of
    /// them all).
    pub state: String,
    /// (file, sha256) of every file in the folder but this one.
    pub files: Vec<(String, String)>,
}

/// A loaded checkpoint.
#[derive(Debug)]
pub struct Checkpoint {
    pub meta: CheckpointMeta,
    /// sha256 of the meta file (which holds every other file's sha256): the checkpoint's hash.
    pub sha256: String,
    pub vars: VarMap,
    /// The optimizer's state: its per-var tensor sets in `checkpoint_state`'s order (None: no
    /// state for that var, e.g. an SGD var that never took a step).
    pub optimizer_state: Vec<Vec<Option<Tensor>>>,
}

fn adam_kind() -> String {
    "adam".into()
}

fn is_adam(k: &str) -> bool {
    k == "adam"
}

/// An optimizer whose whole state a coordinated checkpoint holds, so a resume is exact: Adam
/// (both moments and the step count) and SGD (the momentum buffers). A resume into another kind
/// of optimizer is refused.
pub trait CheckpointOptimizer {
    /// The optimizer's name in the checkpoint ("adam", "sgd").
    fn checkpoint_kind(&self) -> &'static str;
    /// The state: named per-var tensor sets (each one file) and a step count.
    fn checkpoint_state(&self) -> (Vec<(&'static str, Vec<Option<Tensor>>)>, u64);
    /// Put a saved state in place: the sets in `checkpoint_state`'s order, on the vars' devices.
    fn restore_checkpoint_state(&mut self, sets: Vec<Vec<Option<Tensor>>>, t: u64) -> Result<()>;
}

impl CheckpointOptimizer for Adam {
    fn checkpoint_kind(&self) -> &'static str {
        "adam"
    }

    fn checkpoint_state(&self) -> (Vec<(&'static str, Vec<Option<Tensor>>)>, u64) {
        let (m, v, t) = self.state();
        let some = |x: &[Tensor]| x.iter().cloned().map(Some).collect();
        (vec![("adam_m", some(m)), ("adam_v", some(v))], t)
    }

    fn restore_checkpoint_state(&mut self, sets: Vec<Vec<Option<Tensor>>>, t: u64) -> Result<()> {
        let all = |x: Vec<Option<Tensor>>| x.into_iter().collect::<Option<Vec<Tensor>>>().ok_or_else(|| NnError::Dist("an Adam moment is missing from the checkpoint".into()));
        let mut it = sets.into_iter();
        let (Some(m), Some(v), None) = (it.next(), it.next(), it.next()) else {
            return Err(NnError::Dist("Adam's state is two moment sets".into()));
        };
        self.set_state(all(m)?, all(v)?, t);
        Ok(())
    }
}

impl CheckpointOptimizer for Sgd {
    fn checkpoint_kind(&self) -> &'static str {
        "sgd"
    }

    fn checkpoint_state(&self) -> (Vec<(&'static str, Vec<Option<Tensor>>)>, u64) {
        (vec![("sgd_momentum", self.buffers().to_vec())], 0)
    }

    fn restore_checkpoint_state(&mut self, sets: Vec<Vec<Option<Tensor>>>, _t: u64) -> Result<()> {
        let mut it = sets.into_iter();
        let (Some(b), None) = (it.next(), it.next()) else {
            return Err(NnError::Dist("SGD's state is one set of momentum buffers".into()));
        };
        self.set_buffers(b).map_err(NnError::Dist)
    }
}

/// A per-var tensor set as a var map: the vars that have a tensor, under their names.
fn set_map(vars: &VarMap, set: &[Option<Tensor>]) -> Result<VarMap> {
    let mut out = VarMap::new();
    for (n, t) in vars.names().iter().zip(set) {
        if let Some(t) = t {
            out.insert(n.clone(), t.clone())?;
        }
    }
    Ok(out)
}

/// The state hash: for Adam 0.3.0's (`vars moments t`), else the vars, the kind, every set and t.
fn state(kind: &str, vars: &VarMap, sets: &[VarMap], t: u64) -> Result<String> {
    let mut text = state_hash(vars)?;
    if kind != "adam" {
        text.push_str(&format!(" {kind}"));
    }
    for m in sets {
        text.push_str(&format!(" {}", state_hash(m)?));
    }
    text.push_str(&format!(" {t}"));
    Ok(crate::hash::sha256_hex(text.as_bytes()))
}

fn io<E: std::fmt::Display>(what: &Path) -> impl FnOnce(E) -> NnError + '_ {
    move |e| NnError::Dist(format!("{}: {e}", what.display()))
}

/// Write `bytes` to `path` durably (a temporary file, synced, then renamed).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp).map_err(io(&tmp))?;
    f.write_all(bytes).map_err(io(&tmp))?;
    f.sync_all().map_err(io(&tmp))?;
    std::fs::rename(&tmp, path).map_err(io(path))
}

/// Write the files `texts` into `folder` (through a temporary folder and a rename), read them back
/// and check them; returns the sha256 of the meta file as written.
fn write_folder(dir: &Path, folder: &Path, texts: &[(&str, String)]) -> Result<String> {
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    let tmp = folder.with_extension("tmp");
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).map_err(io(&tmp))?;
    }
    std::fs::create_dir_all(&tmp).map_err(io(&tmp))?;
    for (name, text) in texts {
        write_atomic(&tmp.join(name), text.as_bytes())?;
    }
    if folder.exists() {
        std::fs::remove_dir_all(folder).map_err(io(folder))?;
    }
    std::fs::rename(&tmp, folder).map_err(io(folder))?;
    for (name, text) in texts {
        let p = folder.join(name);
        let back = std::fs::read(&p).map_err(io(&p))?;
        if back != text.as_bytes() {
            return Err(NnError::Dist(format!("{}: the bytes read back differ from those written", p.display())));
        }
    }
    let c = Checkpoint::load(folder)?;
    Ok(c.sha256)
}

impl Checkpoint {
    /// The folder of step `step` under `dir`.
    pub fn folder(dir: &Path, step: u64) -> PathBuf {
        dir.join(format!("step-{step:010}"))
    }

    /// Take a coordinated checkpoint after `step` optimizer steps: every rank calls it, with its
    /// loader cursor; rank 0 writes it. Returns the folder.
    pub fn save(group: &mut ProcessGroup, dir: &Path, step: u64, cursor: u64, vars: &VarMap, opt: &dyn CheckpointOptimizer) -> Result<PathBuf> {
        Checkpoint::save_with(group, dir, step, cursor, vars, opt, false)
    }

    /// `save`, and with `node_copies` the first rank of every other node writes a copy under its
    /// own `dir` too (a node-local folder; never a folder another writer shares).
    ///
    /// 1. Every rank hashes its vars and optimizer state; the hashes must agree.
    /// 2. Every rank builds the checkpoint's bytes itself (the state is the same on every rank)
    ///    and its sha256: the sha256 of the meta file, which holds every other file's sha256.
    /// 3. The writers write through a temporary folder and a rename, read the files back and
    ///    check them against their hashes (a copy is kept only if it holds the same bytes).
    /// 4. Rank 0 broadcasts the sha256 of what it wrote; every rank compares it with its own
    ///    and acknowledges; any mismatch fails the checkpoint on every rank, and `latest` is not
    ///    moved.
    /// 5. The writers point `latest` at the folder (a rename); a barrier ends the checkpoint.
    pub fn save_with(group: &mut ProcessGroup, dir: &Path, step: u64, cursor: u64, vars: &VarMap, opt: &dyn CheckpointOptimizer, node_copies: bool) -> Result<PathBuf> {
        let kind = opt.checkpoint_kind();
        let (sets, t) = opt.checkpoint_state();
        let maps: Vec<(&str, VarMap)> = sets.iter().map(|(n, set)| Ok((*n, set_map(vars, set)?))).collect::<Result<_>>()?;
        let st = state(kind, vars, &maps.iter().map(|(_, m)| m.clone()).collect::<Vec<_>>(), t)?;
        let mine = format!("{st} {cursor}");
        let all = group.all_gather(mine.as_bytes())?;
        let mut cursors = Vec::with_capacity(all.len());
        for (q, a) in all.iter().enumerate() {
            let a = String::from_utf8_lossy(a);
            let (h, c) = a.split_once(' ').ok_or_else(|| NnError::Dist(format!("rank {q}: a malformed checkpoint record")))?;
            if h != st {
                return Err(NnError::Dist(format!("checkpoint at step {step}: rank {q}'s state ({h}) differs from rank {}'s ({st})", group.rank())));
            }
            cursors.push(c.parse().map_err(|_| NnError::Dist(format!("rank {q}: cursor {c:?}")))?);
        }
        // The checkpoint's bytes, built on every rank.
        let mut files = vec![];
        let mut texts = vec![];
        let names: Vec<String> = std::iter::once("vars".to_string()).chain(maps.iter().map(|(n, _)| n.to_string())).map(|n| format!("{n}.json")).collect();
        for (name, map) in names.iter().zip(std::iter::once(vars).chain(maps.iter().map(|(_, m)| m))) {
            let text = map.to_json()?;
            files.push((name.clone(), crate::hash::sha256_hex(text.as_bytes())));
            texts.push((name.as_str(), text));
        }
        let meta = CheckpointMeta {
            format: CHECKPOINT_FORMAT.into(),
            step,
            adam_t: t,
            optimizer: kind.to_string(),
            world_size: group.world_size(),
            group_key: group.key(),
            job_key: group.options().job_key.clone(),
            cursors,
            state: st,
            files,
        };
        let meta_text = serde_json::to_string_pretty(&meta).map_err(|e| NnError::Dist(format!("checkpoint meta: {e}")))?;
        let sha = crate::hash::sha256_hex(meta_text.as_bytes());
        texts.push(("meta.json", meta_text));
        let folder = Checkpoint::folder(dir, step);
        let env = group.env().clone();
        let writer = group.rank() == 0 || (node_copies && env.rank_in_node == 0 && env.node_rank != 0);
        let written = if writer { Some(write_folder(dir, &folder, &texts)) } else { None };
        // Rank 0's sha256 of what it wrote, against every rank's own.
        let mut theirs = match (&written, group.rank()) {
            (Some(Ok(h)), 0) => h.clone().into_bytes(),
            (Some(Err(e)), 0) => format!("rank 0 could not write: {e}").into_bytes(),
            _ => vec![],
        };
        group.broadcast(0, &mut theirs)?;
        let theirs = String::from_utf8_lossy(&theirs).to_string();
        let ack = match &written {
            Some(Err(e)) => format!("rank {}: {e}", group.rank()),
            Some(Ok(h)) if *h != sha => format!("rank {}: its copy hashes to {h}, the checkpoint to {sha}", group.rank()),
            _ if theirs != sha => format!("rank {}: rank 0 wrote {theirs}, this rank's checkpoint is {sha}", group.rank()),
            _ => "ok".to_string(),
        };
        let acks = group.all_gather(ack.as_bytes())?;
        if let Some(bad) = acks.iter().find(|a| a.as_slice() != b"ok") {
            return Err(NnError::Dist(format!("checkpoint at step {step} not verified: {}", String::from_utf8_lossy(bad))));
        }
        if writer {
            let name = folder.file_name().expect("a folder name").to_string_lossy().to_string();
            write_atomic(&dir.join("latest"), name.as_bytes())?;
        }
        group.barrier()?;
        Ok(folder)
    }

    /// The checkpoint `latest` points at under `dir` (every rank its own `dir`), checked
    /// against its hashes; None when no rank has one. Every rank calls it.
    /// - Every rank must have loaded the same checkpoint (the same sha256 of its meta file):
    ///   node copies are used only if they hold the same bytes.
    /// - The checkpoint must hold a cursor for every rank.
    /// - A checkpoint of another world size or job key is refused, and one taken under another
    ///   group key (another platform or device: a CPU checkpoint on a GPU, another GPU model or
    ///   driver) too.
    pub fn load_latest(group: &mut ProcessGroup, dir: &Path) -> Result<Option<Checkpoint>> {
        let latest = dir.join("latest");
        let mine: Result<Option<Checkpoint>> = if latest.exists() {
            std::fs::read_to_string(&latest).map_err(io(&latest)).and_then(|name| Checkpoint::load(&dir.join(name.trim()))).map(Some)
        } else {
            Ok(None)
        };
        let said = match &mine {
            Ok(Some(c)) => c.sha256.clone(),
            Ok(None) => "none".to_string(),
            Err(e) => format!("error: {e}"),
        };
        let all = group.all_gather(said.as_bytes())?;
        if let Some((q, a)) = all.iter().enumerate().find(|(_, a)| a.as_slice() != said.as_bytes()) {
            return Err(NnError::Dist(format!("rank {}: loaded checkpoint {said}, rank {q} {} (resume refused)", group.rank(), String::from_utf8_lossy(a))));
        }
        let Some(c) = mine? else { return Ok(None) };
        if c.meta.world_size != group.world_size() || c.meta.job_key != group.options().job_key {
            return Err(NnError::Dist(format!(
                "the checkpoint at step {} was taken at world size {} with job key {:?}; this run has world size {} and job key {:?} (a resume keeps both: resume refused)",
                c.meta.step,
                c.meta.world_size,
                c.meta.job_key,
                group.world_size(),
                group.options().job_key
            )));
        }
        if c.meta.group_key != group.key() {
            return Err(NnError::Dist(format!(
                "the checkpoint at step {} was taken under the key {:?}; this run's key is {:?} (another platform or device: resume refused)",
                c.meta.step,
                c.meta.group_key,
                group.key()
            )));
        }
        if c.meta.cursors.len() != group.world_size() {
            return Err(NnError::Dist(format!("the checkpoint holds {} cursors for {} ranks (resume refused)", c.meta.cursors.len(), group.world_size())));
        }
        Ok(Some(c))
    }

    /// Rank `rank`'s loader cursor.
    pub fn cursor(&self, rank: usize) -> Option<u64> {
        self.meta.cursors.get(rank).copied()
    }

    /// The checkpoint in `folder`, checked against its hashes.
    pub fn load(folder: &Path) -> Result<Checkpoint> {
        let mp = folder.join("meta.json");
        let text = std::fs::read_to_string(&mp).map_err(io(&mp))?;
        let sha256 = crate::hash::sha256_hex(text.as_bytes());
        let meta: CheckpointMeta = serde_json::from_str(&text).map_err(io(&mp))?;
        if meta.format != CHECKPOINT_FORMAT {
            return Err(NnError::Dist(format!("{}: format {:?}, expected {CHECKPOINT_FORMAT:?}", mp.display(), meta.format)));
        }
        let sets: &[&str] = match meta.optimizer.as_str() {
            "adam" => &["adam_m", "adam_v"],
            "sgd" => &["sgd_momentum"],
            k => return Err(NnError::Dist(format!("{}: an optimizer {k:?} this version cannot restore", mp.display()))),
        };
        let mut maps = vec![];
        for name in std::iter::once("vars").chain(sets.iter().copied()).map(|n| format!("{n}.json")) {
            let name = name.as_str();
            let p = folder.join(name);
            let text = std::fs::read_to_string(&p).map_err(io(&p))?;
            let want = meta.files.iter().find(|(f, _)| f == name).map(|(_, h)| h.as_str());
            let got = crate::hash::sha256_hex(text.as_bytes());
            if want != Some(got.as_str()) {
                return Err(NnError::Dist(format!("{}: sha256 {got}, the meta file records {want:?}", p.display())));
            }
            maps.push(VarMap::from_json(&text)?);
        }
        let vars = maps.remove(0);
        let st = state(&meta.optimizer, &vars, &maps, meta.adam_t)?;
        if st != meta.state {
            return Err(NnError::Dist(format!("{}: the state hashes to {st}, the meta file records {}", folder.display(), meta.state)));
        }
        let optimizer_state = maps.iter().map(|m| vars.names().iter().map(|n| m.get(n).cloned()).collect()).collect();
        Ok(Checkpoint { meta, sha256, vars, optimizer_state })
    }

    /// Put the checkpoint's vars and optimizer state in place; returns the steps taken. The vars
    /// and the state go to the devices of the vars they replace. The optimizer must be of the
    /// kind whose state the checkpoint holds (else the resume would not be exact: refused).
    pub fn restore(self, vars: &mut VarMap, opt: &mut dyn CheckpointOptimizer) -> Result<u64> {
        if self.vars.names() != vars.names() {
            return Err(NnError::Dist("the checkpoint holds other vars".into()));
        }
        if opt.checkpoint_kind() != self.meta.optimizer {
            return Err(NnError::Dist(format!(
                "the checkpoint holds {:?}'s state; this run's optimizer is {:?} (its state is not in the checkpoint: resume refused)",
                self.meta.optimizer,
                opt.checkpoint_kind()
            )));
        }
        let devices: Vec<_> = vars.tensors().iter().map(|t| t.device()).collect();
        for (i, (t, d)) in self.vars.tensors().iter().zip(&devices).enumerate() {
            vars.set_index(i, t.clone().to(*d))?;
        }
        let on = |ts: Vec<Option<Tensor>>| ts.into_iter().zip(&devices).map(|(t, d)| t.map(|t| t.to(*d))).collect::<Vec<_>>();
        opt.restore_checkpoint_state(self.optimizer_state.into_iter().map(on).collect(), self.meta.adam_t)?;
        Ok(self.meta.step)
    }
}
