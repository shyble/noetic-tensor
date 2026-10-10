//! Hand-written declarations for the three system libraries the CUDA backend links: the driver API (nvcuda / libcuda), the runtime compiler NVRTC and cuBLAS. Only the
//! functions the backend calls are declared; the signatures follow cuda.h, nvrtc.h and cublas_api.h of
//! CUDA 12 (the `_v2` names are the ABI the headers map the plain names to). NVML is loaded at run
//! time (`tensor::nvml`).

#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUmemoryPool = *mut c_void;
pub type CUevent = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: c_int = 115;
/// CUmemPool_attribute: bytes the pool keeps reserved across synchronisations.
pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: c_int = 4;
/// cuStreamCreate flag: the stream does not synchronise with the legacy default stream.
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;

extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDriverGetVersion(version: *mut c_int) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, stream: CUstream) -> CUresult;
    pub fn cuMemFreeAsync(dptr: CUdeviceptr, stream: CUstream) -> CUresult;
    pub fn cuDeviceGetDefaultMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;
    pub fn cuMemPoolSetAttribute(pool: CUmemoryPool, attr: c_int, value: *mut c_void) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize, stream: CUstream) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize, stream: CUstream) -> CUresult;
    pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(f: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult;
    pub fn cuLaunchKernel(f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared_bytes: c_uint, stream: CUstream, params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult;
    pub fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    pub fn cuEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;
    pub fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    pub fn cuGetErrorName(error: CUresult, pstr: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, pstr: *mut *const c_char) -> CUresult;
}

pub type nvrtcResult = c_int;
pub type nvrtcProgram = *mut c_void;
pub const NVRTC_SUCCESS: nvrtcResult = 0;

extern "C" {
    pub fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult;
    pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;
    pub fn nvrtcCreateProgram(prog: *mut nvrtcProgram, src: *const c_char, name: *const c_char, num_headers: c_int, headers: *const *const c_char, include_names: *const *const c_char) -> nvrtcResult;
    pub fn nvrtcCompileProgram(prog: nvrtcProgram, num_options: c_int, options: *const *const c_char) -> nvrtcResult;
    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetCUBINSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
}

pub type cublasStatus_t = c_int;
pub type cublasHandle_t = *mut c_void;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_OP_N: c_int = 0;
/// cublasMath_t: pedantic math (no reduced-precision or TF32 paths).
pub const CUBLAS_PEDANTIC_MATH: c_int = 2;

extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut c_int) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) -> cublasStatus_t;
    pub fn cublasSetMathMode(handle: cublasHandle_t, mode: c_int) -> cublasStatus_t;
    pub fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut c_int) -> cublasStatus_t;
    pub fn cublasSgemmStridedBatched(
        handle: cublasHandle_t,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        stride_a: i64,
        b: *const f32,
        ldb: c_int,
        stride_b: i64,
        beta: *const f32,
        c: *mut f32,
        ldc: c_int,
        stride_c: i64,
        batch: c_int,
    ) -> cublasStatus_t;
}
