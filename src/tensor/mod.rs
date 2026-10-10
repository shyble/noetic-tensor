//! Tensors and reverse-mode autodiff, on the CPU (Reference and
//! Fast) and, behind features, Metal and CUDA. Built first to reproduce burn 0.21
//! (`Autodiff<NdArray<f32>>`) bit for bit; burn is no longer a dependency and
//! the numerics are the standard formulations, with burn's recorded
//! outputs kept as test fixtures where the numerics did not change.

mod activation;
pub mod api;
mod autodiff;
mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
// The kernel source alone, for `cuda_kernel_source` in a build without the `cuda` feature.
#[cfg(not(feature = "cuda"))]
#[path = "cuda/kernels.rs"]
mod cuda_kernels;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[doc(hidden)]
pub mod metal;
pub mod device;
pub(crate) mod nvml;
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
pub use gpu::{gpu_key, gpu_platform_key, synchronize, transfer_counts, GpuStorage};
pub use float::Tensor;

/// GPU memory on `device`: (bytes in use, the limit) — Metal: this process's buffers and the
/// recommended working set; CUDA: the device's used and total memory. None on a CPU device, or
/// before this process has created that GPU's context (it never creates one).
pub fn gpu_memory(device: Device) -> Option<(u64, u64)> {
    match device {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Device::Metal(_) => metal::memory(),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda::memory(),
        _ => None,
    }
}
pub use int::IntTensor;

/// The CUDA kernels' source text: what NVRTC compiles and what the CUDA backend's provenance key
/// hashes (sha256 of these bytes). Available with or without the `cuda` feature.
pub fn cuda_kernel_source() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        cuda::kernels::SOURCE
    }
    #[cfg(not(feature = "cuda"))]
    {
        cuda_kernels::SOURCE
    }
}

#[cfg(test)]
pub(crate) mod tests;
