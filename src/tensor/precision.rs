//! The precision of matrix products on a GPU: f32 (the default, every backend's ordered f32
//! product) or a reduced-precision fast path on the GPU's matrix units, opt-in.
//!
//! `MatmulPrecision::Half` rounds both operands to a 16-bit float as they are loaded and
//! accumulates the products in f32, on the GPU's matrix hardware:
//! - **CUDA**: bf16 operands (round to nearest, ties to even) through the tensor cores'
//!   `mma.sync.m16n8k16` (compute capability 8.0 or later), f32 accumulators, 64×64 output tiles
//!   with k in steps of 32, in order; no split over k, no atomics.
//! - **Metal**: f16 operands (IEEE half, round to nearest even; values beyond ±65504 become ±∞)
//!   through simdgroup 8×8 matrices, f32 accumulators, 64×64 output tiles with k in steps of 32,
//!   in order. (Metal takes f16, the 16-bit type the tested Apple GPU's simdgroup matrices
//!   take.)
//!
//! The configuration is fixed: one kernel per backend, its tile shape and k order chosen by
//! nothing but the shape (no timing, no autotuning), its source hashed into the key
//! (`half_gemm_key`). Its results are its own: they are never compared byte for byte with, or
//! pooled with, f32 results, and the key says which precision a run used.
//!
//! The precision is set per thread (`set_matmul_precision`; the backward pass of a product runs
//! with the precision of its forward product) and applies to GPU tensors only: CPU tensors always
//! multiply in f32. `Tensor::matmul_with` names the precision of one product.

use super::device::Device;
use super::error::{Result, TensorError};
use std::cell::Cell;

/// The precision of a matrix product on a GPU (see the module's documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MatmulPrecision {
    /// f32 operands and products, in the backend's fixed order: every result in this engine's
    /// keys and gates so far.
    #[default]
    F32,
    /// 16-bit operands (bf16 on CUDA, f16 on Metal), f32 accumulation, on the GPU's matrix
    /// units. Its own key; never compared byte for byte with, or pooled with, f32 results.
    Half,
}

impl MatmulPrecision {
    pub fn name(self) -> &'static str {
        match self {
            MatmulPrecision::F32 => "f32",
            MatmulPrecision::Half => "half",
        }
    }
}

thread_local! {
    static PRECISION: Cell<MatmulPrecision> = const { Cell::new(MatmulPrecision::F32) };
}

/// Set the precision of `Tensor::matmul` on GPU tensors for the calling thread (and of the
/// backward passes of the products made then). Returns the previous setting. Set it before the
/// run's keys are taken (`gpu_platform_key`, a process group): they read it.
pub fn set_matmul_precision(p: MatmulPrecision) -> MatmulPrecision {
    PRECISION.with(|c| c.replace(p))
}

/// The calling thread's matmul precision for GPU tensors (`MatmulPrecision::F32` unless set).
pub fn matmul_precision() -> MatmulPrecision {
    PRECISION.with(|c| c.get())
}

/// The reduced-precision product's part of a key on `device`: the operand type, the matrix
/// instruction, the accumulator, the tile and the kernel source's hash (16 hex digits). An error
/// on a CPU device, on a build without the device's backend, or on a GPU that lacks the matrix
/// units (CUDA before compute capability 8.0).
pub fn half_gemm_key(device: Device) -> Result<String> {
    super::gpu::backend_for(device)?.half_gemm_key()
}

/// `half_gemm_key` when the calling thread's precision is `Half`, else None: the part
/// `gpu_platform_key` appends.
pub(crate) fn key_part(device: Device) -> Result<Option<String>> {
    match matmul_precision() {
        MatmulPrecision::F32 => Ok(None),
        MatmulPrecision::Half => half_gemm_key(device).map(Some),
    }
}

/// The refusal of a reduced-precision product on a CPU device.
pub(crate) fn cpu_refusal(device: Device) -> TensorError {
    TensorError::Unsupported(format!("a reduced-precision (MatmulPrecision::Half) product on {device:?}: the CPU backends multiply in f32 only"))
}
