//! One training step on any decoder configuration: the masked per-seed
//! cross-entropy, plus the MoE blocks' auxiliary losses at their weights, at a training step
//! (dropout masks of that step, MoE capacity), and one optimizer step.

use super::decoder::{Decoder, DecoderConfig, ForwardOptions};
use super::loss::masked_cross_entropy;
use super::optim::Optimizer;
use super::var::VarMap;
use crate::tensor::{IntTensor, Tensor};

/// Weights of the MoE auxiliary losses (Switch uses 0.01 for the balance loss; ST-MoE 0.001 for
/// the z-loss).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxWeights {
    pub balance: f64,
    pub z_loss: f64,
}

impl Default for AuxWeights {
    fn default() -> Self {
        AuxWeights { balance: 0.01, z_loss: 0.001 }
    }
}

/// What a step reports, per seed.
#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    /// The cross-entropy (without the auxiliary terms).
    pub ce: Vec<f32>,
    pub balance: Option<Vec<f32>>,
    pub z_loss: Option<Vec<f32>>,
    /// Per MoE block, per seed, kept assignments per expert.
    pub load: Vec<Vec<Vec<usize>>>,
    pub dropped: Vec<Vec<usize>>,
}

/// One step at training step `step`: loss per seed = CE + wb·balance + wz·z (the auxiliary terms
/// only with MoE blocks); the seeds' losses are summed for the backward pass.
#[allow(clippy::too_many_arguments)]
pub fn train_step(cfg: &DecoderConfig, vars: &mut VarMap, opt: &mut dyn Optimizer, tokens: &IntTensor, targets: &IntTensor, mask: &Tensor, step: u64, aux: AuxWeights, lr_scale: f64) -> StepReport {
    let lifted = vars.lifted();
    let model = Decoder::load(cfg.clone(), &lifted).expect("the vars fit the config");
    let out = model.forward_with(tokens, &ForwardOptions { pad: None, train_step: Some(step) });
    let ce = masked_cross_entropy(out.logits, targets, mask);
    let report = StepReport {
        ce: ce.to_vec(),
        balance: out.balance.as_ref().map(|t| t.to_vec()),
        z_loss: out.z_loss.as_ref().map(|t| t.to_vec()),
        load: out.load,
        dropped: out.dropped,
    };
    let mut loss = ce;
    if let Some(b) = out.balance {
        loss = loss + b.mul_scalar(aux.balance);
    }
    if let Some(z) = out.z_loss {
        loss = loss + z.mul_scalar(aux.z_loss);
    }
    let grads = loss.sum().backward();
    opt.step(vars, lifted.grads(&grads), lr_scale);
    report
}
