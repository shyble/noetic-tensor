//! Neural-network components on the engine's tensors, structured like candle-nn.

pub mod activation;
pub mod adam;
pub mod attention;
pub mod block;
pub mod clip;
pub mod decoder;
pub mod dist;
pub mod dropout;
pub mod ema;
pub mod embedding;
pub mod init;
pub mod kv_cache;
pub mod linear;
pub mod loss;
pub mod mlp;
pub mod moe;
pub mod module;
pub mod norm;
pub mod optim;
pub mod rotary;
pub mod schedule;
pub mod sgd;
pub mod trainer;
pub mod var;
pub mod var_builder;

#[cfg(test)]
mod tests;

pub use activation::{erf, gelu_erf, gelu_tanh, relu, tanh, Activation};
pub use adam::{Adam, AdamConfig, DecayOrder};
pub use attention::{attention, causal_attention, AttentionConfig, MultiHeadAttention};
pub use block::{block, block_with, Block, BlockConfig, FeedForward, MlpKind, MoeSpec, Norm, NormKind};
pub use clip::{clip_grad_norm_per_seed, clip_grad_value, grad_norms_per_seed};
pub use decoder::{BlockNaming, Decoder, DecoderConfig, DecoderOutput, ForwardOptions, Positions};
pub use dropout::Dropout;
pub use ema::Ema;
pub use embedding::{embedding, Embedding, EmbeddingMode};
pub use init::Init;
pub use kv_cache::{DecoderCache, KvCache};
pub use linear::{linear, linear_b, linear_init, Linear};
pub use loss::{cross_entropy, cross_entropy_with, kl_div, masked_cross_entropy, mse, nll, CeOptions};
pub use mlp::{gated_mlp, mlp, swiglu, GatedMlp, GatedMlpConfig, Mlp, MlpExt, MlpExtension, MlpTransform};
pub use moe::{moe, Expert, ExpertKind, Moe, MoeConfig, MoeOutput};
pub use module::{Module, ModuleT};
pub use norm::{layer_norm, rms_norm, rms_normalize, LayerNorm, RmsNorm};
pub use optim::{Optimizer, ParamGroups, ParamOptions};
pub use rotary::{Rope, RopeConfig, RopeStyle};
pub use schedule::{Constant, LinearWarmup, Schedule, StepDecay, WarmupCosine};
pub use sgd::{Sgd, SgdConfig};
pub use trainer::{train_step, AuxWeights, StepReport};
pub use var::{VarMap, VarMapFile, VarRecord};
pub use var_builder::VarBuilder;
