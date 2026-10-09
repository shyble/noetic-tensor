//! Coordinated checkpoints of data-parallel training: the vars, the optimizer's state and every
//! rank's loader cursor, taken by every rank at the same step.
//!
//! Every rank hashes its vars and optimizer state, and the hashes must agree (the ranks took
//! identical steps). Rank 0 then writes `step-<step>/` (vars, moments, a meta file with each
//! file's sha256) through a temporary folder and a rename, then points `latest` at it (again by
//! a rename), and a barrier ends the checkpoint: once any rank is past it, the checkpoint is on
//! disk whole. Loading checks the hashes and refuses another world size or job key (another
//! run's key). Rank 0 writes on its own machine; across machines the folder must be shared, or
//! copied before a resume.

use super::ddp::state_hash;
use super::group::ProcessGroup;
use crate::error::{NnError, Result};
use crate::nn::{Adam, VarMap};
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
    /// Adam's step count.
    pub adam_t: u64,
    pub world_size: usize,
    /// The reproducibility key (`ProcessGroup::key`; the world size is recorded above) and the
    /// job key.
    pub group_key: String,
    pub job_key: String,
    /// Each rank's loader cursor, in rank order.
    pub cursors: Vec<u64>,
    /// sha256 of the vars, moments and step count (`state_hash` of each, then of the three).
    pub state: String,
    /// (file, sha256) of every file in the folder but this one.
    pub files: Vec<(String, String)>,
}

/// A loaded checkpoint.
#[derive(Debug)]
pub struct Checkpoint {
    pub meta: CheckpointMeta,
    pub vars: VarMap,
    pub adam_m: Vec<Tensor>,
    pub adam_v: Vec<Tensor>,
}

fn moments(vars: &VarMap, m: &[Tensor]) -> Result<VarMap> {
    let mut out = VarMap::new();
    for (n, t) in vars.names().iter().zip(m) {
        out.insert(n.clone(), t.clone())?;
    }
    Ok(out)
}

fn state(vars: &VarMap, m: &VarMap, v: &VarMap, t: u64) -> Result<String> {
    let text = format!("{} {} {} {t}", state_hash(vars)?, state_hash(m)?, state_hash(v)?);
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

impl Checkpoint {
    /// The folder of step `step` under `dir`.
    pub fn folder(dir: &Path, step: u64) -> PathBuf {
        dir.join(format!("step-{step:010}"))
    }

    /// Take a coordinated checkpoint after `step` optimizer steps: every rank calls it, with its
    /// loader cursor. Returns the folder.
    pub fn save(group: &mut ProcessGroup, dir: &Path, step: u64, cursor: u64, vars: &VarMap, adam: &Adam) -> Result<PathBuf> {
        let (m, v, t) = adam.state();
        let (m, v) = (moments(vars, m)?, moments(vars, v)?);
        let st = state(vars, &m, &v, t)?;
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
        let folder = Checkpoint::folder(dir, step);
        if group.rank() == 0 {
            std::fs::create_dir_all(dir).map_err(io(dir))?;
            let tmp = folder.with_extension("tmp");
            if tmp.exists() {
                std::fs::remove_dir_all(&tmp).map_err(io(&tmp))?;
            }
            std::fs::create_dir_all(&tmp).map_err(io(&tmp))?;
            let mut files = vec![];
            for (name, map) in [("vars.json", vars), ("adam_m.json", &m), ("adam_v.json", &v)] {
                let text = map.to_json()?;
                write_atomic(&tmp.join(name), text.as_bytes())?;
                files.push((name.to_string(), crate::hash::sha256_hex(text.as_bytes())));
            }
            let meta = CheckpointMeta {
                format: CHECKPOINT_FORMAT.into(),
                step,
                adam_t: t,
                world_size: group.world_size(),
                group_key: group.key(),
                job_key: group.options().job_key.clone(),
                cursors,
                state: st,
                files,
            };
            let text = serde_json::to_string_pretty(&meta).map_err(|e| NnError::Dist(format!("checkpoint meta: {e}")))?;
            write_atomic(&tmp.join("meta.json"), text.as_bytes())?;
            if folder.exists() {
                std::fs::remove_dir_all(&folder).map_err(io(&folder))?;
            }
            std::fs::rename(&tmp, &folder).map_err(io(&folder))?;
            let name = folder.file_name().expect("a folder name").to_string_lossy().to_string();
            write_atomic(&dir.join("latest"), name.as_bytes())?;
        }
        group.barrier()?;
        Ok(folder)
    }

    /// The checkpoint `latest` points at under `dir`, checked against its hashes; None when
    /// there is none. Refuses a checkpoint of another world size or job key.
    pub fn load_latest(group: &ProcessGroup, dir: &Path) -> Result<Option<Checkpoint>> {
        let latest = dir.join("latest");
        if !latest.exists() {
            return Ok(None);
        }
        let name = std::fs::read_to_string(&latest).map_err(io(&latest))?;
        let c = Checkpoint::load(&dir.join(name.trim()))?;
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
        Ok(Some(c))
    }

    /// The checkpoint in `folder`, checked against its hashes.
    pub fn load(folder: &Path) -> Result<Checkpoint> {
        let mp = folder.join("meta.json");
        let text = std::fs::read_to_string(&mp).map_err(io(&mp))?;
        let meta: CheckpointMeta = serde_json::from_str(&text).map_err(io(&mp))?;
        if meta.format != CHECKPOINT_FORMAT {
            return Err(NnError::Dist(format!("{}: format {:?}, expected {CHECKPOINT_FORMAT:?}", mp.display(), meta.format)));
        }
        let mut maps = vec![];
        for name in ["vars.json", "adam_m.json", "adam_v.json"] {
            let p = folder.join(name);
            let text = std::fs::read_to_string(&p).map_err(io(&p))?;
            let want = meta.files.iter().find(|(f, _)| f == name).map(|(_, h)| h.as_str());
            let got = crate::hash::sha256_hex(text.as_bytes());
            if want != Some(got.as_str()) {
                return Err(NnError::Dist(format!("{}: sha256 {got}, the meta file records {want:?}", p.display())));
            }
            maps.push(VarMap::from_json(&text)?);
        }
        let v = maps.pop().expect("three maps");
        let m = maps.pop().expect("three maps");
        let vars = maps.pop().expect("three maps");
        let st = state(&vars, &m, &v, meta.adam_t)?;
        if st != meta.state {
            return Err(NnError::Dist(format!("{}: the state hashes to {st}, the meta file records {}", folder.display(), meta.state)));
        }
        Ok(Checkpoint { meta, vars, adam_m: m.tensors().to_vec(), adam_v: v.tensors().to_vec() })
    }

    /// Put the checkpoint's vars and optimizer state in place; returns the steps taken. The vars
    /// go to the devices of the vars they replace.
    pub fn restore(self, vars: &mut VarMap, adam: &mut Adam) -> Result<u64> {
        if self.vars.names() != vars.names() {
            return Err(NnError::Dist("the checkpoint holds other vars".into()));
        }
        let devices: Vec<_> = vars.tensors().iter().map(|t| t.device()).collect();
        for (i, (t, d)) in self.vars.tensors().iter().zip(&devices).enumerate() {
            vars.set_index(i, t.clone().to(*d))?;
        }
        let on = |ts: Vec<Tensor>| ts.into_iter().zip(&devices).map(|(t, d)| t.to(*d)).collect::<Vec<_>>();
        adam.set_state(on(self.adam_m), on(self.adam_v), self.meta.adam_t);
        Ok(self.meta.step)
    }
}
