//! The CUDA backend. `backend` implements `GpuBackend` (tensors on one
//! `Device::Cuda(i)` per process, `Cuda(0)` unless the first use names another); the rest of this file is the raw device API: a device with its primary context, one stream and one
//! cuBLAS handle; f32 buffers; kernels compiled at run time by NVRTC (vector add and a tiled
//! sgemm of our own) and cuBLAS sgemm. Behind the `cuda` feature (off by default); nothing here
//! is reachable from `Tensor` (tensors reach the GPU through `Storage::Gpu` and `backend`).
//!
//! Determinism: kernels are compiled without fast math and with `--fmad=false` (the only
//! contraction is the explicit `fmaf` in sgemm, one per k in order); cuBLAS runs in pedantic math
//! with TF32 off (`NVIDIA_TF32_OVERRIDE=0` is set before the handle is created if unset). A
//! pinned process (`pin_reference`) never initialises the driver.

#[doc(hidden)]
pub mod backend;
pub use backend::{memory, profile_start, profile_take, ProfileReport};
mod ffi;
pub(crate) mod kernels;
#[cfg(test)]
mod sweep;

use super::error::{Result, TensorError};
use ffi::*;
use std::ffi::{c_int, c_void, CStr, CString};
use std::sync::{Arc, Mutex};

/// The raw device's kernels. `sgemm_tiled`: 64×64 output tiles, k in steps of 16 through shared
/// memory, 256 threads with a 4×4 register block each; every output is one `fmaf` chain over k
/// in order from 0 (zero padding past the edges adds exact zeros), batched through blockIdx.z.
const KERNELS: &str = r#"
extern "C" __global__ void vadd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long step = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) c[i] = a[i] + b[i];
}

#define BM 64
#define BN 64
#define BK 16
extern "C" __global__ void __launch_bounds__(256) sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, long long sA, long long sB, long long sC) {
    __shared__ float As[BK][BM + 4];
    __shared__ float Bs[BK][BN + 4];
    A += blockIdx.z * sA; B += blockIdx.z * sB; C += blockIdx.z * sC;
    const int t = threadIdx.x, tx = t % 16, ty = t / 16;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int e = t + 256 * l;
            int r = e / BK, c = e % BK;
            int gr = row0 + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[(long long)gr * K + gc] : 0.0f;
            int rb = e / BN, cb = e % BN;
            int grb = k0 + rb, gcb = col0 + cb;
            Bs[rb][cb] = (grb < K && gcb < N) ? B[(long long)grb * N + gcb] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[kk][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[kk][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + ty + 16 * i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int c = col0 + tx + 16 * j;
            if (c < N) C[(long long)r * N + c] = acc[i][j];
        }
    }
}
"#;

/// What the device and its libraries are (for the provenance key).
#[derive(Clone, Debug)]
pub struct CudaInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessors: i32,
    pub total_mem: usize,
    /// cuDriverGetVersion: 1000 × major + 10 × minor.
    pub driver_version: i32,
    pub nvrtc_version: (i32, i32),
    pub cublas_version: i32,
}

struct Inner {
    info: CudaInfo,
    dev: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
    blas: cublasHandle_t,
    module: CUmodule,
    f_add: CUfunction,
    f_sgemm: CUfunction,
    /// Serialises every call on this device (the stream and the cuBLAS handle are not
    /// thread-safe) and makes the context current on the calling thread.
    lock: Mutex<()>,
}

// SAFETY: the raw handles are only used under `lock`, with the context made current first.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// A CUDA device: primary context, one non-blocking stream, one pedantic cuBLAS handle and
/// the NVRTC-compiled kernels. Cloning shares it.
#[derive(Clone)]
pub struct CudaDevice {
    inner: Arc<Inner>,
}

/// f32 elements in device memory; freed on drop.
pub struct CudaBuffer {
    ptr: CUdeviceptr,
    len: usize,
    dev: Arc<Inner>,
}

fn dev_err(what: &str, msg: impl std::fmt::Display) -> TensorError {
    TensorError::Device(format!("CUDA {what}: {msg}"))
}

fn cu(r: CUresult, what: &str) -> Result<()> {
    if r == CUDA_SUCCESS {
        return Ok(());
    }
    let (mut name, mut desc): (*const std::ffi::c_char, *const std::ffi::c_char) = (std::ptr::null(), std::ptr::null());
    // SAFETY: the driver returns static strings (or leaves the pointers null).
    let (n, d) = unsafe {
        cuGetErrorName(r, &mut name);
        cuGetErrorString(r, &mut desc);
        let s = |p: *const std::ffi::c_char| if p.is_null() { "?".to_string() } else { CStr::from_ptr(p).to_string_lossy().into_owned() };
        (s(name), s(desc))
    };
    Err(dev_err(what, format!("{n} ({r}): {d}")))
}

fn nvrtc(r: nvrtcResult, what: &str) -> Result<()> {
    if r == NVRTC_SUCCESS {
        return Ok(());
    }
    // SAFETY: nvrtcGetErrorString returns a static string.
    let s = unsafe { CStr::from_ptr(nvrtcGetErrorString(r)).to_string_lossy().into_owned() };
    Err(dev_err(what, format!("{s} ({r})")))
}

fn blas(r: cublasStatus_t, what: &str) -> Result<()> {
    if r == CUBLAS_STATUS_SUCCESS { Ok(()) } else { Err(dev_err(what, format!("cublasStatus {r}"))) }
}

/// Compile `src` to PTX for compute capability `cc` (no fast math, no FMA contraction).
fn compile(src: &str, cc: (i32, i32)) -> Result<CString> {
    let csrc = CString::new(src).unwrap();
    let name = CString::new("tensor_kernels.cu").unwrap();
    let opts: Vec<CString> = [format!("--gpu-architecture=compute_{}{}", cc.0, cc.1), "--fmad=false".into(), "--std=c++14".into()].into_iter().map(|o| CString::new(o).unwrap()).collect();
    let optp: Vec<*const std::ffi::c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    let mut prog: nvrtcProgram = std::ptr::null_mut();
    // SAFETY: valid NUL-terminated strings and a program handle destroyed below.
    unsafe {
        nvrtc(nvrtcCreateProgram(&mut prog, csrc.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()), "nvrtcCreateProgram")?;
        let r = nvrtcCompileProgram(prog, optp.len() as c_int, optp.as_ptr());
        if r != NVRTC_SUCCESS {
            let mut n = 0usize;
            nvrtcGetProgramLogSize(prog, &mut n);
            let mut log = vec![0u8; n.max(1)];
            nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut _);
            nvrtcDestroyProgram(&mut prog);
            let log = String::from_utf8_lossy(&log[..n.saturating_sub(1)]).into_owned();
            return Err(dev_err("nvrtcCompileProgram", log));
        }
        let mut n = 0usize;
        nvrtc(nvrtcGetPTXSize(prog, &mut n), "nvrtcGetPTXSize")?;
        let mut ptx = vec![0u8; n];
        nvrtc(nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut _), "nvrtcGetPTX")?;
        nvrtcDestroyProgram(&mut prog);
        ptx.truncate(n.saturating_sub(1));
        Ok(CString::new(ptx).map_err(|e| dev_err("PTX", e))?)
    }
}

impl CudaDevice {
    /// Open device `ordinal`. Refused (before the driver is touched) in a pinned process.
    pub fn new(ordinal: usize) -> Result<CudaDevice> {
        Self::new_with_pin(ordinal, super::device::is_pinned())
    }

    pub(crate) fn new_with_pin(ordinal: usize, pinned: bool) -> Result<CudaDevice> {
        if pinned {
            return Err(TensorError::Device(format!("this process is pinned to the reference backend; Cuda({ordinal}) is never initialised")));
        }
        if std::env::var_os("NVIDIA_TF32_OVERRIDE").is_none() {
            // TF32 off for every library in the process (cuBLAS reads it at handle creation).
            std::env::set_var("NVIDIA_TF32_OVERRIDE", "0");
        }
        // SAFETY: plain driver calls with out-pointers to locals; handles are owned by Inner.
        unsafe {
            cu(cuInit(0), "cuInit")?;
            let mut count = 0;
            cu(cuDeviceGetCount(&mut count), "cuDeviceGetCount")?;
            if ordinal >= count as usize {
                return Err(dev_err("device", format!("ordinal {ordinal} of {count} devices")));
            }
            let mut dev: CUdevice = 0;
            cu(cuDeviceGet(&mut dev, ordinal as c_int), "cuDeviceGet")?;
            let mut name = [0 as std::ffi::c_char; 256];
            cu(cuDeviceGetName(name.as_mut_ptr(), 256, dev), "cuDeviceGetName")?;
            let attr = |a| -> Result<i32> {
                let mut v = 0;
                cu(cuDeviceGetAttribute(&mut v, a, dev), "cuDeviceGetAttribute")?;
                Ok(v)
            };
            let cc = (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?, attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?);
            let sms = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
            let mut total = 0usize;
            cu(cuDeviceTotalMem_v2(&mut total, dev), "cuDeviceTotalMem")?;
            let mut drv = 0;
            cu(cuDriverGetVersion(&mut drv), "cuDriverGetVersion")?;
            let (mut vmaj, mut vmin) = (0, 0);
            nvrtc(nvrtcVersion(&mut vmaj, &mut vmin), "nvrtcVersion")?;

            let mut ctx: CUcontext = std::ptr::null_mut();
            cu(cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            // From here a failure releases what was acquired (Inner's Drop, fields null-checked).
            let mut inner = Inner {
                info: CudaInfo { ordinal, name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(), compute_capability: cc, multiprocessors: sms, total_mem: total, driver_version: drv, nvrtc_version: (vmaj, vmin), cublas_version: 0 },
                dev,
                ctx,
                stream: std::ptr::null_mut(),
                blas: std::ptr::null_mut(),
                module: std::ptr::null_mut(),
                f_add: std::ptr::null_mut(),
                f_sgemm: std::ptr::null_mut(),
                lock: Mutex::new(()),
            };
            cu(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            cu(cuStreamCreate(&mut inner.stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")?;
            let ptx = compile(KERNELS, cc)?;
            cu(cuModuleLoadData(&mut inner.module, ptx.as_ptr() as *const c_void), "cuModuleLoadData")?;
            let fname = |s: &str| CString::new(s).unwrap();
            cu(cuModuleGetFunction(&mut inner.f_add, inner.module, fname("vadd").as_ptr()), "cuModuleGetFunction(vadd)")?;
            cu(cuModuleGetFunction(&mut inner.f_sgemm, inner.module, fname("sgemm_tiled").as_ptr()), "cuModuleGetFunction(sgemm_tiled)")?;
            blas(cublasCreate_v2(&mut inner.blas), "cublasCreate")?;
            blas(cublasSetStream_v2(inner.blas, inner.stream), "cublasSetStream")?;
            blas(cublasSetMathMode(inner.blas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode")?;
            let mut mode = -1;
            blas(cublasGetMathMode(inner.blas, &mut mode), "cublasGetMathMode")?;
            if mode != CUBLAS_PEDANTIC_MATH {
                return Err(dev_err("cuBLAS", format!("math mode {mode}, expected pedantic")));
            }
            blas(cublasGetVersion_v2(inner.blas, &mut inner.info.cublas_version), "cublasGetVersion")?;
            Ok(CudaDevice { inner: Arc::new(inner) })
        }
    }

    pub fn info(&self) -> &CudaInfo {
        &self.inner.info
    }

    /// The provenance key of GPU results: device, compute capability, driver,
    /// NVRTC and cuBLAS versions, and the math settings.
    pub fn gpu_key(&self) -> String {
        let i = &self.inner.info;
        format!(
            "cuda-{}-sm{}{}-driver{}-nvrtc{}.{}-cublas{}-pedantic-tf32off-nofastmath",
            i.name.replace(' ', "_"),
            i.compute_capability.0,
            i.compute_capability.1,
            i.driver_version,
            i.nvrtc_version.0,
            i.nvrtc_version.1,
            i.cublas_version
        )
    }

    fn enter(&self) -> Result<std::sync::MutexGuard<'_, ()>> {
        let g = self.inner.lock.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: the context is retained for Inner's lifetime.
        cu(unsafe { cuCtxSetCurrent(self.inner.ctx) }, "cuCtxSetCurrent")?;
        Ok(g)
    }

    /// Uninitialised device memory for `len` f32s.
    pub fn alloc(&self, len: usize) -> Result<CudaBuffer> {
        let _g = self.enter()?;
        let mut ptr: CUdeviceptr = 0;
        if len > 0 {
            // SAFETY: out-pointer to a local.
            cu(unsafe { cuMemAlloc_v2(&mut ptr, len * 4) }, "cuMemAlloc")?;
        }
        Ok(CudaBuffer { ptr, len, dev: self.inner.clone() })
    }

    /// Copy `host` into new device memory (the copy is ordered on the device's stream; the host
    /// slice is staged before the call returns).
    pub fn upload(&self, host: &[f32]) -> Result<CudaBuffer> {
        let b = self.alloc(host.len())?;
        self.write(&b, host)?;
        Ok(b)
    }

    /// Overwrite `buf` with `host` (equal lengths).
    pub fn write(&self, buf: &CudaBuffer, host: &[f32]) -> Result<()> {
        self.same(buf)?;
        if buf.len != host.len() {
            return Err(TensorError::Shape(format!("write of {} elements into a buffer of {}", host.len(), buf.len)));
        }
        let _g = self.enter()?;
        if host.is_empty() {
            return Ok(());
        }
        // SAFETY: buf holds len f32s; the pageable host range is staged before the call returns,
        // and the stream is synchronised so the host slice may be dropped afterwards.
        unsafe {
            cu(cuMemcpyHtoDAsync_v2(buf.ptr, host.as_ptr() as *const c_void, host.len() * 4, self.inner.stream), "cuMemcpyHtoDAsync")?;
            cu(cuStreamSynchronize(self.inner.stream), "cuStreamSynchronize")
        }
    }

    /// Copy `buf` back to the host (waits for all work queued before it).
    pub fn download(&self, buf: &CudaBuffer) -> Result<Vec<f32>> {
        self.same(buf)?;
        let _g = self.enter()?;
        let mut v = vec![0f32; buf.len];
        if buf.len > 0 {
            // SAFETY: v has room for len f32s; the stream is synchronised before v is read.
            unsafe {
                cu(cuMemcpyDtoHAsync_v2(v.as_mut_ptr() as *mut c_void, buf.ptr, buf.len * 4, self.inner.stream), "cuMemcpyDtoHAsync")?;
                cu(cuStreamSynchronize(self.inner.stream), "cuStreamSynchronize")?;
            }
        }
        Ok(v)
    }

    /// Wait for every operation queued on the device's stream.
    pub fn synchronize(&self) -> Result<()> {
        let _g = self.enter()?;
        // SAFETY: the stream lives as long as Inner.
        cu(unsafe { cuStreamSynchronize(self.inner.stream) }, "cuStreamSynchronize")
    }

    fn same(&self, b: &CudaBuffer) -> Result<()> {
        if Arc::ptr_eq(&self.inner, &b.dev) { Ok(()) } else { Err(TensorError::Unsupported("a buffer of another CUDA device".into())) }
    }

    /// Queue `c = a + b` elementwise (NVRTC kernel; bit-exact against the CPU).
    pub fn add(&self, a: &CudaBuffer, b: &CudaBuffer, c: &CudaBuffer) -> Result<()> {
        for x in [a, b, c] {
            self.same(x)?;
        }
        if a.len != b.len || a.len != c.len {
            return Err(TensorError::Shape(format!("add of {}, {} into {}", a.len, b.len, c.len)));
        }
        if a.len == 0 {
            return Ok(());
        }
        let _g = self.enter()?;
        let n = a.len as u64;
        let (mut pa, mut pb, mut pc, mut pn) = (a.ptr, b.ptr, c.ptr, n);
        let mut params = [&mut pa as *mut _ as *mut c_void, &mut pb as *mut _ as *mut c_void, &mut pc as *mut _ as *mut c_void, &mut pn as *mut _ as *mut c_void];
        let blocks = n.div_ceil(256).min(65_535 * 8) as u32;
        // SAFETY: parameters match vadd's signature; buffers hold n f32s.
        cu(unsafe { cuLaunchKernel(self.inner.f_add, blocks, 1, 1, 256, 1, 1, 0, self.inner.stream, params.as_mut_ptr(), std::ptr::null_mut()) }, "cuLaunchKernel(vadd)")
    }

    fn check_gemm(&self, a: &CudaBuffer, b: &CudaBuffer, c: &CudaBuffer, batch: usize, m: usize, k: usize, n: usize) -> Result<()> {
        for x in [a, b, c] {
            self.same(x)?;
        }
        if a.len != batch * m * k || b.len != batch * k * n || c.len != batch * m * n {
            return Err(TensorError::Shape(format!("sgemm [{batch}, {m}, {k}] × [{batch}, {k}, {n}] with buffers of {}, {}, {}", a.len, b.len, c.len)));
        }
        if [batch, m, k, n].iter().any(|&d| d > i32::MAX as usize) || batch > 65_535 {
            return Err(TensorError::Shape(format!("sgemm dimensions too large: [{batch}, {m}, {k}, {n}]")));
        }
        Ok(())
    }

    /// Queue `c[t] = a[t] · b[t]` for `t < batch` (row-major m×k, k×n, m×n blocks) with our own
    /// tiled kernel.
    pub fn sgemm_tiled(&self, a: &CudaBuffer, b: &CudaBuffer, c: &CudaBuffer, batch: usize, m: usize, k: usize, n: usize) -> Result<()> {
        self.check_gemm(a, b, c, batch, m, k, n)?;
        if batch * m * n == 0 {
            return Ok(());
        }
        let _g = self.enter()?;
        let (mut pa, mut pb, mut pc) = (a.ptr, b.ptr, c.ptr);
        let (mut im, mut i_n, mut ik) = (m as i32, n as i32, k as i32);
        let (mut sa, mut sb, mut sc) = ((m * k) as i64, (k * n) as i64, (m * n) as i64);
        let mut params = [
            &mut pa as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut pc as *mut _ as *mut c_void,
            &mut im as *mut _ as *mut c_void,
            &mut i_n as *mut _ as *mut c_void,
            &mut ik as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut sb as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        // SAFETY: parameters match sgemm_tiled's signature; shapes checked against the buffers.
        cu(
            unsafe { cuLaunchKernel(self.inner.f_sgemm, n.div_ceil(64) as u32, m.div_ceil(64) as u32, batch as u32, 256, 1, 1, 0, self.inner.stream, params.as_mut_ptr(), std::ptr::null_mut()) },
            "cuLaunchKernel(sgemm_tiled)",
        )
    }

    /// The same product through cuBLAS (pedantic math, TF32 off). Row-major C = A·B is the
    /// column-major Cᵀ = Bᵀ·Aᵀ, so B and A are passed swapped without transposes.
    pub fn sgemm_cublas(&self, a: &CudaBuffer, b: &CudaBuffer, c: &CudaBuffer, batch: usize, m: usize, k: usize, n: usize) -> Result<()> {
        self.check_gemm(a, b, c, batch, m, k, n)?;
        if batch * m * n == 0 {
            return Ok(());
        }
        if k == 0 {
            return Err(TensorError::Unsupported("sgemm with k = 0 on cuBLAS".into()));
        }
        let _g = self.enter()?;
        let (alpha, beta) = (1f32, 0f32);
        // SAFETY: column-major views of the row-major buffers: B is n×k with ld n, A is k×m with
        // ld k, C is n×m with ld n; strides are the per-batch block sizes.
        blas(
            unsafe {
                cublasSgemmStridedBatched(
                    self.inner.blas,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha,
                    b.ptr as *const f32,
                    n as c_int,
                    (k * n) as i64,
                    a.ptr as *const f32,
                    k as c_int,
                    (m * k) as i64,
                    &beta,
                    c.ptr as *mut f32,
                    n as c_int,
                    (m * n) as i64,
                    batch as c_int,
                )
            },
            "cublasSgemmStridedBatched",
        )
    }
}

impl CudaBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _g = self.dev.lock.lock().unwrap_or_else(|p| p.into_inner());
            // SAFETY: ptr came from cuMemAlloc in this context; errors on teardown are ignored.
            unsafe {
                cuCtxSetCurrent(self.dev.ctx);
                cuMemFree_v2(self.ptr);
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: each handle is released once, in reverse order of creation; null ones skipped.
        unsafe {
            cuCtxSetCurrent(self.ctx);
            if !self.blas.is_null() {
                cublasDestroy_v2(self.blas);
            }
            if !self.module.is_null() {
                cuModuleUnload(self.module);
            }
            if !self.stream.is_null() {
                cuStreamSynchronize(self.stream);
                cuStreamDestroy_v2(self.stream);
            }
            cuDevicePrimaryCtxRelease_v2(self.dev);
        }
    }
}

/// The NVIDIA driver's release (e.g. "576.88"), from NVML (part of the driver), read once.
pub(crate) fn driver_release() -> Result<String> {
    static R: std::sync::OnceLock<std::result::Result<String, String>> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        let mut buf = [0 as std::ffi::c_char; 96];
        // SAFETY: NVML's init and shutdown are reference counted; the buffer outlives the call
        // and its length is passed.
        unsafe {
            let r = nvmlInit_v2();
            if r != NVML_SUCCESS {
                return Err(format!("nvmlInit: status {r}"));
            }
            let r = nvmlSystemGetDriverVersion(buf.as_mut_ptr(), buf.len() as std::ffi::c_uint);
            nvmlShutdown();
            if r != NVML_SUCCESS {
                return Err(format!("nvmlSystemGetDriverVersion: status {r}"));
            }
            Ok(CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())
        }
    })
    .clone()
    .map_err(|e| dev_err("NVML", e))
}
