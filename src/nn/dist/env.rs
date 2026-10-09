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

/// The platform key a job's ranks must share: the engine's platform key (OS, architecture, the
/// CPU features its kernels select on, the compiler, the gemm kernel) and a fingerprint of the
/// OS maths library (`maths_fingerprint`), since softmax, log_softmax and `powf` call it and its
/// last bits may differ between systems. The rendezvous refuses a job whose ranks differ in it.
pub fn platform_key() -> String {
    format!("{} libm={}", crate::tensor::platform_key(), maths_fingerprint())
}

/// sha256 of the bits of f32 and f64 `exp`, `ln`, `powf` and `tanh` on fixed inputs that span the
/// ranges the engine reaches:
/// - `exp` on 513 evenly spaced arguments over [−110, 89] (f32) and [−750, 710] (f64): the
///   large negative arguments of a softmax after its max is subtracted, the arguments whose
///   results are subnormal (f32 below about −87.3, f64 below about −708.4), and the arguments
///   next to overflow (f32 88.72, f64 709.78);
/// - `ln` and `powf` on 513 positive values: 64 spaced evenly over the subnormals' bit patterns,
///   then 449 spaced evenly in bit pattern from the smallest normal to the largest finite value
///   (so the whole normal range and values next to overflow), `powf` with exponents 2, 0.5, −1,
///   0.37, 3 and −2.5 in turn;
/// - `tanh` on 513 arguments over [−20, 20];
/// - and fixed edge arguments: ±0, the smallest subnormal and normal, the largest finite, −1e4,
///   −1e30, and the f32 and f64 overflow and subnormal thresholds.
pub(crate) fn maths_fingerprint() -> String {
    use std::hint::black_box;
    let mut b = Vec::new();
    let (x32, p32, y32, e32) = inputs32();
    for i in 0..x32.len() {
        let (x, pos, y) = (black_box(x32[i]), black_box(p32[i]), black_box(y32[i]));
        for v in [x.exp(), pos.ln(), pos.powf(black_box(POWERS[i % 6] as f32)), y.tanh()] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    for x in e32 {
        let x = black_box(x);
        for v in [x.exp(), x.abs().ln(), x.abs().powf(black_box(0.37)), x.tanh()] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    let (x64, p64, y64, e64) = inputs64();
    for i in 0..x64.len() {
        let (x, pos, y) = (black_box(x64[i]), black_box(p64[i]), black_box(y64[i]));
        for v in [x.exp(), pos.ln(), pos.powf(black_box(POWERS[i % 6])), y.tanh()] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    for x in e64 {
        let x = black_box(x);
        for v in [x.exp(), x.abs().ln(), x.abs().powf(black_box(0.37)), x.tanh()] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    crate::hash::sha256_hex(&b)
}

const POWERS: [f64; 6] = [2.0, 0.5, -1.0, 0.37, 3.0, -2.5];

/// The f32 inputs: `exp` arguments, positive values for `ln` and `powf`, `tanh` arguments (513
/// each), and the edge arguments.
pub(crate) fn inputs32() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let t = |i: u32| i as f32 / 512.0;
    (
        (0..513).map(|i| -110.0 + 199.0 * t(i)).collect(),
        (0..513).map(|i| f32::from_bits(if i < 64 { 1 + i * (0x007f_ffff / 64) } else { 0x0080_0000 + (i - 64) * ((0x7f7f_ffff - 0x0080_0000) / 448) })).collect(),
        (0..513).map(|i| -20.0 + 40.0 * t(i)).collect(),
        vec![0.0, -0.0, f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX, -1.0e4, -1.0e30, 88.722_83, 88.722_84, -87.336_54, -103.972_08, -103.972_09],
    )
}

/// The f64 inputs, as `inputs32`.
pub(crate) fn inputs64() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = |i: u64| i as f64 / 512.0;
    (
        (0..513).map(|i| -750.0 + 1460.0 * t(i)).collect(),
        (0..513).map(|i| f64::from_bits(if i < 64 { 1 + i * (0x000f_ffff_ffff_ffff / 64) } else { 0x0010_0000_0000_0000 + (i - 64) * ((0x7fef_ffff_ffff_ffff - 0x0010_0000_0000_0000) / 448) })).collect(),
        (0..513).map(|i| -20.0 + 40.0 * t(i)).collect(),
        vec![0.0, -0.0, f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX, -1.0e4, -1.0e30, 709.782_712_893_384, 709.782_712_893_385, -708.396_418_532_264, -745.133_219_101_941],
    )
}
