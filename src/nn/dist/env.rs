//! The process's place in a job, from torchrun's environment variables.

use crate::error::{NnError, Result};

/// Rank, world size and the coordinator's address, as torchrun passes them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistEnv {
    /// This process's rank, 0..world_size.
    pub rank: usize,
    pub world_size: usize,
    /// The rank among the processes of this node, and their count.
    pub rank_in_node: usize,
    pub node_size: usize,
    /// The node's index among the nodes.
    pub node_rank: usize,
    /// Where rank 0 listens for the rendezvous.
    pub master_addr: String,
    pub master_port: u16,
}

pub(crate) const VARS: [&str; 7] = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"];

impl DistEnv {
    /// One process, no network.
    pub fn single() -> DistEnv {
        DistEnv { rank: 0, world_size: 1, rank_in_node: 0, node_size: 1, node_rank: 0, master_addr: "127.0.0.1".into(), master_port: 0 }
    }

    /// From the process environment: None when neither `RANK` nor `WORLD_SIZE` is set
    /// (distribution off); an error when the set is incomplete or malformed.
    pub fn from_env() -> Result<Option<DistEnv>> {
        DistEnv::from_lookup(|k| std::env::var(k).ok())
    }

    /// From any lookup of the variables (the environment, a test's map).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Option<DistEnv>> {
        let (rank, world) = (get("RANK"), get("WORLD_SIZE"));
        if rank.is_none() && world.is_none() {
            return Ok(None);
        }
        let num = |k: &str, v: Option<String>| -> Result<usize> {
            let v = v.ok_or_else(|| NnError::Dist(format!("{k} is not set (RANK and WORLD_SIZE need MASTER_ADDR and MASTER_PORT too)")))?;
            v.trim().parse().map_err(|_| NnError::Dist(format!("{k}={v:?} is not a number")))
        };
        let world_size = num("WORLD_SIZE", world)?;
        let rank = num("RANK", rank)?;
        if world_size == 0 || rank >= world_size {
            return Err(NnError::Dist(format!("RANK {rank} is not in 0..WORLD_SIZE {world_size}")));
        }
        let (master_addr, master_port) = if world_size == 1 && get("MASTER_ADDR").is_none() {
            ("127.0.0.1".to_string(), 0)
        } else {
            let addr = get("MASTER_ADDR").ok_or_else(|| NnError::Dist("MASTER_ADDR is not set".into()))?;
            let port = num("MASTER_PORT", get("MASTER_PORT"))?;
            let port = u16::try_from(port).map_err(|_| NnError::Dist(format!("MASTER_PORT {port} is not a port")))?;
            (addr, port)
        };
        let rank_in_node = match get("LOCAL_RANK") {
            Some(v) => num("LOCAL_RANK", Some(v))?,
            None => rank,
        };
        let node_size = match get("LOCAL_WORLD_SIZE") {
            Some(v) => num("LOCAL_WORLD_SIZE", Some(v))?,
            None => world_size,
        };
        let node_rank = match get("GROUP_RANK") {
            Some(v) => num("GROUP_RANK", Some(v))?,
            None => 0,
        };
        if rank_in_node >= node_size {
            return Err(NnError::Dist(format!("LOCAL_RANK {rank_in_node} is not below LOCAL_WORLD_SIZE {node_size}")));
        }
        Ok(Some(DistEnv { rank, world_size, rank_in_node, node_size, node_rank, master_addr, master_port }))
    }

    /// The variables that describe this process, as the launcher sets them.
    pub fn to_vars(&self) -> Vec<(String, String)> {
        let vals = [
            self.master_addr.clone(),
            self.master_port.to_string(),
            self.rank.to_string(),
            self.world_size.to_string(),
            self.rank_in_node.to_string(),
            self.node_size.to_string(),
            self.node_rank.to_string(),
        ];
        VARS.iter().map(|k| k.to_string()).zip(vals).collect()
    }
}
