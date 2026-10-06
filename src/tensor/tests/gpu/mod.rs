//! GPU backend tests: one set of test bodies for every backend. Each backend
//! module below defines its device `M`, its tag, and a few adapters (the raw kernels' entry
//! points, whether the driver is initialised, its launch counter, its provenance-key markers),
//! then includes the same files (raw, ops, backward, kernels); every file's wrappers run its `body_*` tests in a fresh
//! process (the reference pin is process-wide). Tests that exist for one backend only (its raw
//! device API, Metal's tuned matmul variants) are in `metal_only.rs` and `cuda_only.rs`.

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use crate::tensor::Device;

    pub(crate) const M: Device = Device::Metal(0);
    pub(crate) const TAG: &str = "metal";
    /// A device index this backend does not have.
    pub(crate) const OTHER: Device = Device::Metal(1);
    pub(crate) const KEY_PREFIX: &str = "metal-";
    pub(crate) const KEY_MARK: &str = "mathmode-safe";

    pub(crate) fn initialized() -> bool {
        crate::tensor::metal::initialized()
    }

    /// Kernels encoded since the device was created.
    pub(crate) fn launches() -> u64 {
        crate::tensor::metal::dispatches()
    }

    /// What a pinned process checks beyond the tensor API (nothing more on Metal).
    pub(crate) fn pinned_extra() {}

    /// c = a + b through the raw kernel.
    pub(crate) fn raw_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        crate::tensor::metal::raw_add(a, b).expect("Metal")
    }

    /// `batch` row-major products through the raw tiled sgemm (one call per batch).
    pub(crate) fn raw_matmul(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        (0..batch).flat_map(|t| crate::tensor::metal::raw_matmul(&a[t * m * k..(t + 1) * m * k], &b[t * k * n..(t + 1) * k * n], m, k, n).expect("Metal")).collect()
    }

    mod raw {
        include!("raw.rs");
    }
    mod ops {
        include!("ops.rs");
    }
    mod backward {
        include!("backward.rs");
    }
    mod kernels {
        include!("kernels.rs");
    }
    mod only {
        include!("metal_only.rs");
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::tensor::cuda::CudaDevice;
    use crate::tensor::{Device, TensorError};

    pub(crate) const M: Device = Device::Cuda(0);
    pub(crate) const TAG: &str = "cuda";
    /// A device index this backend does not have.
    pub(crate) const OTHER: Device = Device::Cuda(1);
    pub(crate) const KEY_PREFIX: &str = "cuda-";
    pub(crate) const KEY_MARK: &str = "nofastmath-fmadoff";

    pub(crate) fn initialized() -> bool {
        crate::tensor::cuda::backend::initialized()
    }

    /// Kernels launched since the device was created.
    pub(crate) fn launches() -> u64 {
        crate::tensor::cuda::backend::launches().unwrap()
    }

    /// A pinned process: the raw device refuses too.
    pub(crate) fn pinned_extra() {
        assert!(matches!(CudaDevice::new(0), Err(TensorError::Device(m)) if m.contains("pinned")), "the raw device refuses too");
    }

    fn dev() -> CudaDevice {
        CudaDevice::new(0).unwrap_or_else(|e| panic!("CUDA device 0: {e}"))
    }

    /// c = a + b through the raw NVRTC kernel.
    pub(crate) fn raw_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        let d = dev();
        let (ga, gb, gc) = (d.upload(a).unwrap(), d.upload(b).unwrap(), d.alloc(a.len()).unwrap());
        d.add(&ga, &gb, &gc).unwrap();
        d.download(&gc).unwrap()
    }

    /// `batch` row-major products through the raw tiled sgemm.
    pub(crate) fn raw_matmul(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        let d = dev();
        let (ga, gb, gc) = (d.upload(a).unwrap(), d.upload(b).unwrap(), d.alloc(batch * m * n).unwrap());
        d.sgemm_tiled(&ga, &gb, &gc, batch, m, k, n).unwrap();
        d.download(&gc).unwrap()
    }

    mod raw {
        include!("raw.rs");
    }
    mod ops {
        include!("ops.rs");
    }
    mod backward {
        include!("backward.rs");
    }
    mod kernels {
        include!("kernels.rs");
    }
    mod only {
        include!("cuda_only.rs");
    }
}
