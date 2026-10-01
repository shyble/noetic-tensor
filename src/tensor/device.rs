//! Devices. `Cpu(Reference)` runs the reference kernels (bit-exact
//! and deterministic, in the standard formulations);
//! `Cpu(Fast)` runs the same operations with threads (std::thread::scope)
//! and SIMD (std::arch), checked against Reference within a tolerance. `Metal` and `Cuda`
//! need their feature; without it, selecting them is `TensorError::Unsupported`.
//!
//! A process-level pin keeps reproducible runs on Reference: after `pin_reference()`, Fast, Metal
//! and Cuda can never be selected (the default device and every `to` refuse them), and a pinned
//! process never initialises a GPU.

use super::error::{Result, TensorError};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum CpuMode {
    /// The reference numerics: bit-exact and deterministic, in the standard formulations.
    #[default]
    Reference,
    /// Threads and SIMD; equal to Reference within a per-op tolerance.
    Fast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu(CpuMode),
    Metal(usize),
    Cuda(usize),
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu(CpuMode::Reference)
    }
}

static PINNED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The calling thread's default device (Reference unless set on this thread).
    static DEFAULT: std::cell::Cell<Device> = const { std::cell::Cell::new(Device::Cpu(CpuMode::Reference)) };
}

/// Pin this process to the Reference backend (for bit-exact runs): Fast can no longer be selected,
/// and a Fast default already set falls back to Reference.
pub fn pin_reference() {
    PINNED.store(true, Ordering::SeqCst);
}

pub fn is_pinned() -> bool {
    PINNED.load(Ordering::SeqCst)
}

/// Whether `device` can hold tensors in this process.
pub fn check(device: Device) -> Result<()> {
    check_with(device, is_pinned())
}

pub(crate) fn check_with(device: Device, pinned: bool) -> Result<()> {
    match device {
        Device::Cpu(CpuMode::Reference) => Ok(()),
        Device::Cpu(CpuMode::Fast) if pinned => Err(TensorError::Unsupported("this process is pinned to the reference backend; the fast backend cannot be selected".into())),
        Device::Cpu(CpuMode::Fast) => Ok(()),
        Device::Metal(_) | Device::Cuda(_) if pinned => Err(TensorError::Unsupported(format!("this process is pinned to the reference backend; {device:?} cannot be selected"))),
        Device::Metal(_) | Device::Cuda(_) => super::gpu::check_device(device),
    }
}

/// The device new tensors are created on by this thread (Reference unless set; always
/// Reference in a pinned process).
pub fn default_device() -> Device {
    let d = DEFAULT.with(|d| d.get());
    if is_pinned() { Device::Cpu(CpuMode::Reference) } else { d }
}

/// Set the device this thread creates new tensors on.
pub fn set_default_device(device: Device) -> Result<()> {
    check(device)?;
    DEFAULT.with(|d| d.set(device));
    Ok(())
}

/// The key golden hashes are recorded under: OS, architecture, the CPU features the kernels
/// select on, the compiler and the gemm kernel's version (matrixmultiply picks NEON or
/// AVX/FMA at run time, and libm and rustc can change the last bits).
pub fn platform_key() -> String {
    let features: Vec<&str> = {
        #[cfg(target_arch = "aarch64")]
        {
            [("neon", std::arch::is_aarch64_feature_detected!("neon"))].iter().filter(|f| f.1).map(|f| f.0).collect()
        }
        #[cfg(target_arch = "x86_64")]
        {
            [("avx", std::arch::is_x86_feature_detected!("avx")), ("avx2", std::arch::is_x86_feature_detected!("avx2")), ("fma", std::arch::is_x86_feature_detected!("fma")), ("avx512f", std::arch::is_x86_feature_detected!("avx512f"))].iter().filter(|f| f.1).map(|f| f.0).collect()
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Vec::new()
        }
    };
    format!("{}-{}-{}-{}-matrixmultiply-0.3.11", std::env::consts::OS, std::env::consts::ARCH, if features.is_empty() { "none".into() } else { features.join("+") }, env!("NOETIC_RUSTC").replace(' ', "_"))
}
