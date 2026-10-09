//! A launcher in the shape of torchrun: start `nproc_per_node` processes of a command on this
//! node with the rendezvous variables set, and supervise them. The first process that fails
//! (or is killed) stops the whole job: the others are killed, and the job reports which rank
//! failed. Each rank's output goes to its own log file when a log folder is given.
//!
//! Multi-node jobs start one launcher per node with the same `nnodes`, master address and port
//! and the node's `node_rank`; global ranks are `node_rank · nproc_per_node + rank_in_node`.

use super::env::DistEnv;
use crate::error::{NnError, Result};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Duration;

/// What to launch, and where the job meets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub nproc_per_node: usize,
    pub nnodes: usize,
    pub node_rank: usize,
    pub master_addr: String,
    pub master_port: u16,
    /// The executable and its arguments.
    pub command: PathBuf,
    pub args: Vec<String>,
    /// Extra environment variables for every rank.
    pub env: Vec<(String, String)>,
    /// A folder for one log per rank (`rank-<r>.log`, stdout and stderr); None inherits the
    /// launcher's.
    pub log_dir: Option<PathBuf>,
}

impl LaunchConfig {
    /// `nproc` processes of `command` on this machine, meeting on 127.0.0.1 at a free port.
    pub fn local(nproc: usize, command: impl Into<PathBuf>) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            nproc_per_node: nproc,
            nnodes: 1,
            node_rank: 0,
            master_addr: "127.0.0.1".into(),
            master_port: free_port()?,
            command: command.into(),
            args: vec![],
            env: vec![],
            log_dir: None,
        })
    }

    pub fn world_size(&self) -> usize {
        self.nproc_per_node * self.nnodes
    }

    /// The environment of local rank `local`.
    pub fn rank_env(&self, local: usize) -> DistEnv {
        DistEnv {
            rank: self.node_rank * self.nproc_per_node + local,
            world_size: self.world_size(),
            rank_in_node: local,
            node_size: self.nproc_per_node,
            node_rank: self.node_rank,
            master_addr: self.master_addr.clone(),
            master_port: self.master_port,
        }
    }
}

/// A port that was free a moment ago on 127.0.0.1 (the system's choice).
pub fn free_port() -> Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| NnError::Dist(format!("finding a free port: {e}")))?;
    Ok(std::net::TcpListener::local_addr(&l).map_err(|e| NnError::Dist(format!("finding a free port: {e}")))?.port())
}

/// A running job: this node's processes, by global rank. Dropping it kills what still runs.
#[derive(Debug)]
pub struct Job {
    children: Vec<(usize, Child)>,
}

/// Start this node's processes.
pub fn spawn(cfg: &LaunchConfig) -> Result<Job> {
    if cfg.nproc_per_node == 0 || cfg.nnodes == 0 || cfg.node_rank >= cfg.nnodes {
        return Err(NnError::Dist(format!("{} processes on node {} of {}", cfg.nproc_per_node, cfg.node_rank, cfg.nnodes)));
    }
    if let Some(d) = &cfg.log_dir {
        std::fs::create_dir_all(d).map_err(|e| NnError::Dist(format!("{}: {e}", d.display())))?;
    }
    let mut job = Job { children: vec![] };
    for local in 0..cfg.nproc_per_node {
        let env = cfg.rank_env(local);
        let mut c = Command::new(&cfg.command);
        c.args(&cfg.args).envs(env.to_vars()).envs(cfg.env.iter().map(|(k, v)| (k, v))).stdin(Stdio::null());
        if let Some(d) = &cfg.log_dir {
            let p = d.join(format!("rank-{}.log", env.rank));
            let f = std::fs::File::create(&p).map_err(|e| NnError::Dist(format!("{}: {e}", p.display())))?;
            let f2 = f.try_clone().map_err(|e| NnError::Dist(format!("{}: {e}", p.display())))?;
            c.stdout(f).stderr(f2);
        }
        let child = c.spawn().map_err(|e| NnError::Dist(format!("starting rank {} ({}): {e}", env.rank, cfg.command.display())))?;
        job.children.push((env.rank, child));
    }
    Ok(job)
}

/// Start the job and wait for it (`spawn`, then `Job::wait`).
pub fn run(cfg: &LaunchConfig) -> Result<()> {
    spawn(cfg)?.wait()
}

impl Job {
    /// The global ranks running here, and their process ids.
    pub fn pids(&self) -> Vec<(usize, u32)> {
        self.children.iter().map(|(r, c)| (*r, c.id())).collect()
    }

    /// Kill one rank's process (as a lost machine would stop it).
    pub fn kill_rank(&mut self, rank: usize) -> Result<()> {
        let (_, c) = self.children.iter_mut().find(|(r, _)| *r == rank).ok_or_else(|| NnError::Dist(format!("no rank {rank} in this job")))?;
        c.kill().map_err(|e| NnError::Dist(format!("killing rank {rank}: {e}")))
    }

    /// Kill every process that still runs and reap them.
    pub fn kill_all(&mut self) {
        for (_, c) in self.children.iter_mut() {
            if matches!(c.try_wait(), Ok(None)) {
                let _ = c.kill();
            }
            let _ = c.wait();
        }
    }

    /// Wait for every process. The first that fails stops the job: the others are killed and
    /// the error names the rank and its exit status.
    pub fn wait(mut self) -> Result<()> {
        let mut done: Vec<Option<ExitStatus>> = vec![None; self.children.len()];
        loop {
            for (k, (rank, c)) in self.children.iter_mut().enumerate() {
                if done[k].is_some() {
                    continue;
                }
                match c.try_wait() {
                    Ok(Some(st)) => {
                        done[k] = Some(st);
                        if !st.success() {
                            let rank = *rank;
                            self.kill_all();
                            return Err(NnError::Dist(format!("rank {rank} failed ({st}); the job was stopped")));
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        let rank = *rank;
                        self.kill_all();
                        return Err(NnError::Dist(format!("waiting for rank {rank}: {e}")));
                    }
                }
            }
            if done.iter().all(Option::is_some) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.kill_all();
    }
}
