//! Learning-rate schedules: a schedule gives the scale the optimizer's base rate
//! is multiplied by at each step.

pub trait Schedule {
    fn scale(&self, step: u64) -> f64;
}

/// Linear warmup over the first `total / warmup_div` steps (at least one), then cosine decay to
/// `floor` × the peak at `total`. The default: `warmup_div` 100, `floor` 0.1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarmupCosine {
    pub total: u64,
    pub warmup_div: u64,
    pub floor: f64,
}

impl WarmupCosine {
    pub fn new(total: u64) -> Self {
        WarmupCosine { total, warmup_div: 100, floor: 0.1 }
    }
}

impl Schedule for WarmupCosine {
    fn scale(&self, step: u64) -> f64 {
        let warm = (self.total / self.warmup_div).max(1);
        if step < warm {
            return (step + 1) as f64 / warm as f64;
        }
        let frac = (step - warm) as f64 / (self.total - warm).max(1) as f64;
        self.floor + (1.0 - self.floor) * 0.5 * (1.0 + (std::f64::consts::PI * frac.min(1.0)).cos())
    }
}

/// A constant scale of 1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Constant;

impl Schedule for Constant {
    fn scale(&self, _step: u64) -> f64 {
        1.0
    }
}

/// Linear warmup to 1 over `warmup` steps (at least one), then constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearWarmup {
    pub warmup: u64,
}

impl Schedule for LinearWarmup {
    fn scale(&self, step: u64) -> f64 {
        let w = self.warmup.max(1);
        if step < w { (step + 1) as f64 / w as f64 } else { 1.0 }
    }
}

/// Step decay: `gamma^⌊step / step_size⌋`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepDecay {
    pub step_size: u64,
    pub gamma: f64,
}

impl Schedule for StepDecay {
    fn scale(&self, step: u64) -> f64 {
        self.gamma.powi((step / self.step_size.max(1)) as i32)
    }
}
