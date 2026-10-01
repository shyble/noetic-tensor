//! Tensors and reverse-mode autodiff, on the CPU (Reference and Fast) and, behind features,
//! Metal and CUDA. Built first to reproduce burn 0.21 (`Autodiff<NdArray<f32>>`) bit for bit;
//! the numerics are now the standard formulations, and burn is not a dependency: its recorded
//! outputs are kept as test fixtures where the numerics did not change.

mod activation;
mod autodiff;
mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod metal;
pub mod device;
mod boolean;
mod dtype;
mod error;
mod infer;
mod layout;
mod storage;
mod float;
pub(crate) mod gpu;
pub mod half;
mod int;
mod kernels;
mod shape;

pub use activation::{log_softmax, logsumexp, sigmoid, silu, softmax};
pub use autodiff::Gradients;
pub use boolean::BoolTensor;
pub use backend::threads as backend_threads;
pub use device::{default_device, pin_reference, platform_key, set_default_device, CpuMode, Device};
pub use dtype::{DType, Element, FloatElem};
pub use error::{Result, TensorError};
pub use layout::Layout;
pub use storage::{CpuStorage, Storage};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{replay_trace as metal_replay_trace, set_trace as metal_trace, set_per_kernel_timing as metal_per_kernel_timing, take_profile as metal_profile, Profile as MetalProfile};
pub use gpu::{gpu_key, synchronize, transfer_counts, GpuStorage};
pub use float::Tensor;
pub use int::IntTensor;

#[cfg(test)]
pub(crate) mod tests;
