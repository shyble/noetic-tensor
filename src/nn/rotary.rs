//! Rotary position embedding. Pair i of a head (frequency θᵢ = base^(−2i/dh))
//! at position m is rotated by the angle m·θᵢ, so q_m·k_n depends only on m − n. Two pairings:
//! - `RotateHalf` (GPT-NeoX, Llama in HF form): pair i is (i, i + dh/2);
//! - `Interleaved` (the original paper, GPT-J): pair i is (2i, 2i + 1).
//!
//! The angles are computed on the host in f64 and stored at the input's dtype; the rotation is
//! `x·cos + rot(x)·sin` in tensor ops, so it carries gradient.

use crate::error::{NnError, Result};
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RopeStyle {
    #[default]
    RotateHalf,
    Interleaved,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RopeConfig {
    pub base: f64,
    pub style: RopeStyle,
}

impl Default for RopeConfig {
    fn default() -> Self {
        RopeConfig { base: 10_000.0, style: RopeStyle::RotateHalf }
    }
}

#[derive(Clone, Debug)]
pub struct Rope {
    dh: usize,
    cfg: RopeConfig,
}

impl Rope {
    /// RoPE over head width `dh` (even).
    pub fn new(dh: usize, cfg: RopeConfig) -> Result<Rope> {
        if dh == 0 || !dh.is_multiple_of(2) {
            return Err(NnError::Config(format!("RoPE needs an even head width, not {dh}")));
        }
        Ok(Rope { dh, cfg })
    }

    pub fn config(&self) -> RopeConfig {
        self.cfg
    }

    /// The angle of pair i at position m.
    pub fn angle(&self, m: usize, i: usize) -> f64 {
        m as f64 * self.cfg.base.powf(-2.0 * i as f64 / self.dh as f64)
    }

    /// cos and sin tables `[1, t, dh/2]` for positions offset..offset + t.
    fn tables(&self, t: usize, offset: usize, like: &Tensor) -> (Tensor, Tensor) {
        let h = self.dh / 2;
        let a: Vec<f64> = (0..t).flat_map(|m| (0..h).map(move |i| (m + offset, i))).map(|(m, i)| self.angle(m, i)).collect();
        let mk = |v: Vec<f64>| Tensor::from_f64s(v, [1, t, h], like.dtype()).to(like.device());
        (mk(a.iter().map(|x| x.cos()).collect()), mk(a.iter().map(|x| x.sin()).collect()))
    }

    /// Rotate `x` (`[N, t, dh]`, heads in the attention layout) at positions offset..offset + t.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Tensor {
        let [n, t, dh] = x.dims();
        assert_eq!(dh, self.dh, "RoPE for head width {} applied to {dh}", self.dh);
        let h = dh / 2;
        let (cos, sin) = self.tables(t, offset, x);
        match self.cfg.style {
            RopeStyle::RotateHalf => {
                let x1 = x.clone().slice([0..n, 0..t, 0..h]);
                let x2 = x.clone().slice([0..n, 0..t, h..dh]);
                let o1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
                let o2 = x2 * cos + x1 * sin;
                Tensor::cat(vec![o1, o2], 2)
            }
            RopeStyle::Interleaved => {
                let xr = x.clone().reshape([n, t, h, 2]);
                let a = xr.clone().slice([0..n, 0..t, 0..h, 0..1]).reshape([n, t, h]);
                let b = xr.slice([0..n, 0..t, 0..h, 1..2]).reshape([n, t, h]);
                let o1 = a.clone() * cos.clone() - b.clone() * sin.clone();
                let o2 = a * sin + b * cos;
                Tensor::cat(vec![o1.unsqueeze_dim(3), o2.unsqueeze_dim(3)], 3).reshape([n, t, dh])
            }
        }
    }
}
