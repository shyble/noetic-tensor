//! nn tests on GPU backends: one set of test bodies for every backend. Each
//! backend module defines its device `M`, its tag and its launch counter, then includes the same
//! forward (decoder forward, greedy, traffic) and training (gradients, Adam, trajectories) files; their
//! wrappers run the `body_*` tests in a fresh process.

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use crate::tensor::Device;

    pub(crate) const M: Device = Device::Metal(0);
    pub(crate) const TAG: &str = "metal";

    /// Kernels encoded since the device was created.
    pub(crate) fn launches() -> u64 {
        crate::tensor::metal::dispatches()
    }

    mod forward {
        include!("forward.rs");
    }
    mod training {
        include!("training.rs");
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use crate::tensor::Device;

    pub(crate) const M: Device = Device::Cuda(0);
    pub(crate) const TAG: &str = "cuda";

    /// Kernels launched since the device was created.
    pub(crate) fn launches() -> u64 {
        crate::tensor::cuda::backend::launches().unwrap()
    }

    mod forward {
        include!("forward.rs");
    }
    mod training {
        include!("training.rs");
    }
}
