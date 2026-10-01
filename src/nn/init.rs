//! Initialisation schemes. Every var is `[S, …]`, and seed slot j of a var named
//! `name` is drawn from its own stream `rng::derive(root, "init:{i}:{name}")`, where i is the
//! slot's seed index; so seed i's weights never depend on S or on the seeds beside it. The
//! normal draw is Box–Muller.

use crate::rng::derive;
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    /// Every element `value`.
    Const(f64),
    /// N(0, std²), one Box–Muller draw per element from the slot's stream.
    Normal { std: f64 },
}

impl Init {
    /// N(0, 1/fan_in): the projection scheme (std `1 / √fan_in`).
    pub fn fan_in(fan_in: usize) -> Init {
        Init::Normal { std: 1.0 / (fan_in as f64).sqrt() }
    }

    /// The `n` values of one seed slot, in row-major order, as f64 (`z · std` before any rounding
    /// to the var's dtype).
    pub fn draw(&self, root: u64, seed: usize, name: &str, n: usize) -> Vec<f64> {
        match *self {
            Init::Const(c) => vec![c; n],
            Init::Normal { std } => {
                let mut rng = derive(root, &stream_label(seed, name));
                (0..n)
                    .map(|_| {
                        let u1: f64 = rng.gen::<f64>().max(1e-300);
                        let u2: f64 = rng.gen();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        z * std
                    })
                    .collect()
            }
        }
    }
}

/// The label of seed `seed`'s init stream for the var `name`.
pub fn stream_label(seed: usize, name: &str) -> String {
    format!("init:{seed}:{name}")
}
