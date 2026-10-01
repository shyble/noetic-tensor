//! `GpuBackend` on CUDA: device-resident storage, one non-blocking stream,
//! the NVRTC-compiled kernels of `kernels.rs` (fast math off, no FMA contraction). Each op
//! allocates its output (stream-ordered, from the device's memory pool), queues one kernel (or
//! a few) and returns; a host read synchronises the stream. Frees are stream-ordered too, so a
//! buffer dropped while a queued kernel still reads it stays valid until that kernel ran.
//!
//! The context is created on first use only, and refused in a pinned process (as Metal's), so a
//! pinned process never initialises the driver.

use super::ffi::*;
use super::{compile, cu, dev_err};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::error::{Result, TensorError};
use crate::tensor::gpu::{AdamScalars, BinaryOp, CmpOp, GpuBackend, GpuBuf, GpuStorage, ReduceOp, UnaryOp};
use crate::tensor::half::{BF16, F16};
use crate::tensor::layout::Layout;
use crate::tensor::storage::CpuStorage;
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Device memory (freed stream-ordered after the work queued before the drop).
pub(crate) struct Mem {
    ptr: CUdeviceptr,
    bytes: usize,
}

impl std::fmt::Debug for Mem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaMem({} bytes)", self.bytes)
    }
}

/// Pointers dropped since the context last ran; freed on the stream at the next op. (A drop can
/// happen while the context's lock is held, so it only records the pointer.)
static PENDING_FREE: Mutex<Vec<CUdeviceptr>> = Mutex::new(Vec::new());

impl Drop for Mem {
    fn drop(&mut self) {
        if self.ptr != 0 {
            PENDING_FREE.lock().unwrap_or_else(|e| e.into_inner()).push(self.ptr);
        }
    }
}

pub(crate) struct Ctx {
    ctx: CUcontext,
    stream: CUstream,
    module: CUmodule,
    funcs: HashMap<&'static str, CUfunction>,
    /// The stream-ordered allocator (memory pools) is available.
    pools: bool,
    name: String,
    cc: (i32, i32),
    driver: i32,
    nvrtc: (i32, i32),
    /// Kernels launched since start (for reports).
    pub(crate) launches: u64,
    /// Profiling, off unless `profile_start`.
    prof: Prof,
    /// The profile label of the next launch (an op within a kernel, e.g. `unary:exp`).
    tag: Option<&'static str>,
}

/// Per-launch GPU time from events around each kernel, and host time in launches, allocations
/// and frees (profiling; off by default, no cost when off).
#[derive(Default)]
struct Prof {
    on: bool,
    events: Vec<(&'static str, CUevent, CUevent)>,
    spare: Vec<CUevent>,
    launch_ns: u128,
    allocs: u64,
    alloc_bytes: u64,
    alloc_ns: u128,
    frees: u64,
    free_ns: u128,
}

/// A profile of the work queued between `profile_start` and `profile_take`.
#[derive(Debug, Default)]
pub struct ProfileReport {
    /// Per kernel: (name, launches, GPU ms), by GPU time, largest first.
    pub kernels: Vec<(String, u64, f64)>,
    pub launches: u64,
    /// Host time inside cuLaunchKernel (and the event records around it).
    pub launch_host_ms: f64,
    pub allocs: u64,
    pub alloc_bytes: u64,
    pub alloc_host_ms: f64,
    pub frees: u64,
    pub free_host_ms: f64,
    /// Sum of kernel times, and the span from the first kernel's start to the last one's end.
    pub gpu_busy_ms: f64,
    pub gpu_span_ms: f64,
}

// SAFETY: the handles are only used under CTX's mutex, with the context made current first.
unsafe impl Send for Ctx {}

static CTX: OnceLock<std::result::Result<Mutex<Ctx>, String>> = OnceLock::new();

#[cfg(test)]
/// Whether this process has created the CUDA context.
pub(crate) fn initialized() -> bool {
    CTX.get().is_some()
}

/// The CUDA context of device 0 (created on first use). Refused in a pinned process.
pub(crate) fn context() -> Result<MutexGuard<'static, Ctx>> {
    if crate::tensor::device::is_pinned() {
        return Err(TensorError::Device("this process is pinned to the reference backend; CUDA cannot be initialised".into()));
    }
    let m = match CTX.get_or_init(|| Ctx::create().map(Mutex::new).map_err(|e| e.message().to_string())) {
        Ok(m) => m,
        Err(e) => return Err(TensorError::Device(e.clone())),
    };
    let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
    g.enter()?;
    Ok(g)
}

impl Ctx {
    fn create() -> Result<Ctx> {
        // SAFETY: plain driver calls with out-pointers to locals.
        unsafe {
            cu(cuInit(0), "cuInit")?;
            let mut count = 0;
            cu(cuDeviceGetCount(&mut count), "cuDeviceGetCount")?;
            if count < 1 {
                return Err(dev_err("device", "no CUDA device on this machine"));
            }
            let mut dev: CUdevice = 0;
            cu(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
            let mut name = [0 as std::ffi::c_char; 256];
            cu(cuDeviceGetName(name.as_mut_ptr(), 256, dev), "cuDeviceGetName")?;
            let attr = |a| -> Result<i32> {
                let mut v = 0;
                cu(cuDeviceGetAttribute(&mut v, a, dev), "cuDeviceGetAttribute")?;
                Ok(v)
            };
            let cc = (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?, attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?);
            let pools = attr(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)? != 0;
            let mut driver = 0;
            cu(cuDriverGetVersion(&mut driver), "cuDriverGetVersion")?;
            let (mut vmaj, mut vmin) = (0, 0);
            super::nvrtc(nvrtcVersion(&mut vmaj, &mut vmin), "nvrtcVersion")?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            cu(cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            cu(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            let mut stream: CUstream = std::ptr::null_mut();
            cu(cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")?;
            if pools {
                // Keep freed memory in the pool across synchronisations (no return to the OS).
                let mut pool: CUmemoryPool = std::ptr::null_mut();
                cu(cuDeviceGetDefaultMemPool(&mut pool, dev), "cuDeviceGetDefaultMemPool")?;
                let mut threshold = u64::MAX;
                cu(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &mut threshold as *mut u64 as *mut c_void), "cuMemPoolSetAttribute")?;
            }
            let ptx = compile(super::kernels::SOURCE, cc)?;
            let mut module: CUmodule = std::ptr::null_mut();
            cu(cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void), "cuModuleLoadData")?;
            Ok(Ctx { ctx, stream, module, funcs: HashMap::new(), pools, name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(), cc, driver, nvrtc: (vmaj, vmin), launches: 0, prof: Prof::default(), tag: None })
        }
    }

    /// Make the context current on this thread and free what was dropped (stream-ordered).
    fn enter(&mut self) -> Result<()> {
        // SAFETY: the context is retained for the process's lifetime.
        cu(unsafe { cuCtxSetCurrent(self.ctx) }, "cuCtxSetCurrent")?;
        let freed: Vec<CUdeviceptr> = std::mem::take(&mut *PENDING_FREE.lock().unwrap_or_else(|e| e.into_inner()));
        let t0 = self.prof.on.then(std::time::Instant::now);
        let nfreed = freed.len() as u64;
        for p in freed {
            // SAFETY: p came from this context's allocator; queued work before it still runs.
            unsafe {
                if self.pools {
                    cu(cuMemFreeAsync(p, self.stream), "cuMemFreeAsync")?;
                } else {
                    cu(cuMemFree_v2(p), "cuMemFree")?;
                }
            }
        }
        if let Some(t0) = t0 {
            self.prof.frees += nfreed;
            self.prof.free_ns += t0.elapsed().as_nanos();
        }
        Ok(())
    }

    fn alloc(&mut self, bytes: usize) -> Result<Mem> {
        let mut ptr: CUdeviceptr = 0;
        let t0 = self.prof.on.then(std::time::Instant::now);
        if bytes > 0 {
            // SAFETY: out-pointer to a local.
            unsafe {
                if self.pools {
                    cu(cuMemAllocAsync(&mut ptr, bytes, self.stream), "cuMemAllocAsync")?;
                } else {
                    cu(cuMemAlloc_v2(&mut ptr, bytes), "cuMemAlloc")?;
                }
            }
        }
        if let Some(t0) = t0 {
            self.prof.allocs += 1;
            self.prof.alloc_bytes += bytes as u64;
            self.prof.alloc_ns += t0.elapsed().as_nanos();
        }
        Ok(Mem { ptr, bytes })
    }

    fn upload(&mut self, data: &[u8]) -> Result<Mem> {
        let m = self.alloc(data.len())?;
        if !data.is_empty() {
            // SAFETY: m holds data.len() bytes; a pageable source is staged before the call
            // returns, so `data` may be dropped afterwards.
            cu(unsafe { cuMemcpyHtoDAsync_v2(m.ptr, data.as_ptr() as *const c_void, data.len(), self.stream) }, "cuMemcpyHtoDAsync")?;
        }
        Ok(m)
    }

    fn sync(&mut self) -> Result<()> {
        // SAFETY: the stream lives for the process's lifetime.
        cu(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize")
    }

    fn func(&mut self, kernel: &'static str) -> Result<CUfunction> {
        if let Some(f) = self.funcs.get(kernel) {
            return Ok(*f);
        }
        let n = CString::new(kernel).unwrap();
        let mut f: CUfunction = std::ptr::null_mut();
        // SAFETY: a loaded module and a NUL-terminated name.
        cu(unsafe { cuModuleGetFunction(&mut f, self.module, n.as_ptr()) }, kernel)?;
        self.funcs.insert(kernel, f);
        Ok(f)
    }

    /// Queue `kernel` on the stream with `grid` blocks of `block` threads and `args` in order.
    fn launch(&mut self, kernel: &'static str, grid: (u32, u32, u32), block: u32, args: &[Arg]) -> Result<()> {
        self.launch_shared(kernel, grid, block, 0, args)
    }

    /// `launch` with `shared` bytes of dynamic shared memory per block.
    fn launch_shared(&mut self, kernel: &'static str, grid: (u32, u32, u32), block: u32, shared: u32, args: &[Arg]) -> Result<()> {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        let f = self.func(kernel)?;
        // Device pointers need a stable home for cuLaunchKernel to read them from.
        let mut ptrs: Vec<CUdeviceptr> = Vec::with_capacity(args.len());
        let mut params: Vec<*mut c_void> = Vec::with_capacity(args.len());
        for a in args {
            match a {
                Arg::Buf(m) => {
                    ptrs.push(m.ptr);
                    params.push(ptrs.last_mut().unwrap() as *mut u64 as *mut c_void);
                }
                Arg::Bytes(b) => params.push(b.as_ptr() as *mut c_void),
            }
        }
        let timed = if self.prof.on { Some((std::time::Instant::now(), self.event()?, self.event()?)) } else { None };
        if let Some((_, e0, _)) = timed {
            // SAFETY: a live event on this context's stream.
            cu(unsafe { cuEventRecord(e0, self.stream) }, "cuEventRecord")?;
        }
        // SAFETY: the parameters match the kernel's signature (by position and size); ptrs has
        // its final capacity, so the pointers into it stay valid.
        cu(unsafe { cuLaunchKernel(f, grid.0, grid.1, grid.2, block, 1, 1, shared, self.stream, params.as_mut_ptr(), std::ptr::null_mut()) }, kernel)?;
        self.launches += 1;
        if timed.is_none() {
            self.tag = None;
        }
        if let Some((t0, e0, e1)) = timed {
            // SAFETY: as above.
            cu(unsafe { cuEventRecord(e1, self.stream) }, "cuEventRecord")?;
            self.prof.events.push((self.tag.take().unwrap_or(kernel), e0, e1));
            self.prof.launch_ns += t0.elapsed().as_nanos();
        }
        Ok(())
    }

    fn event(&mut self) -> Result<CUevent> {
        if let Some(e) = self.prof.spare.pop() {
            return Ok(e);
        }
        let mut e: CUevent = std::ptr::null_mut();
        // SAFETY: out-pointer to a local; default flags keep timing on.
        cu(unsafe { cuEventCreate(&mut e, 0) }, "cuEventCreate")?;
        Ok(e)
    }

    fn run1(&mut self, kernel: &'static str, args: &[Arg], n: usize) -> Result<()> {
        let blocks = u32::try_from(n.div_ceil(256)).map_err(|_| TensorError::Unsupported(format!("{n} elements in one CUDA launch")))?;
        self.launch(kernel, (blocks, 1, 1), 256, args)
    }
}

/// A kernel argument: a device buffer, or plain data passed by value (scalars and structs).
enum Arg<'a> {
    Buf(&'a Mem),
    Bytes(&'a [u8]),
}

fn bytes_of<T: Copy>(v: &T) -> &[u8] {
    // SAFETY: T is plain Copy data read as bytes (kernel parameters).
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn raw_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain numeric element types without padding, read as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub(crate) struct CudaBackend;

pub(crate) static CUDA: CudaBackend = CudaBackend;

fn mem(g: &GpuStorage) -> &Mem {
    match &g.buf {
        GpuBuf::Cuda(b) => b,
        #[allow(unreachable_patterns)]
        _ => unreachable!("a CUDA op on another GPU's storage"),
    }
}

fn storage(device: Device, dtype: DType, len: usize, m: Mem) -> GpuStorage {
    GpuStorage { device, dtype, len, buf: GpuBuf::Cuda(Arc::new(m)) }
}

fn out(c: &mut Ctx, like: &GpuStorage, dtype: DType, len: usize) -> Result<GpuStorage> {
    Ok(storage(like.device, dtype, len, c.alloc(len * dtype.size_in_bytes())?))
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| TensorError::Unsupported(format!("CUDA kernels index with 32 bits: {what} {v} is too large")))
}

const MAXR: usize = 8;

/// The kernels' `Strided` parameters (the same layout as Metal's).
#[repr(C)]
#[derive(Clone, Copy)]
struct Strided {
    n: u32,
    rank: u32,
    off_a: u32,
    off_b: u32,
    off_c: u32,
    shape: [u32; MAXR],
    sa: [u32; MAXR],
    sb: [u32; MAXR],
    sc: [u32; MAXR],
}

/// Parameters for an iteration over `shape` reading a (and b) and writing c through layouts
/// (None: contiguous from 0). Dimensions of size 1 are dropped and adjacent dimensions that
/// are contiguous in every operand are merged, so a contiguous operand is read as one run (fewer
/// divisions per element); the element order and every address are unchanged.
fn strided(shape: &[usize], a: Option<&Layout>, b: Option<&Layout>, c: Option<&Layout>) -> Result<Strided> {
    let contiguous = crate::tensor::shape::strides(shape);
    let ops: [(&[usize], usize); 3] = [a, b, c].map(|l| match l {
        Some(l) => (l.strides.as_slice(), l.offset),
        None => (contiguous.as_slice(), 0),
    });
    // (size, strides per operand), outermost first.
    let mut dims: Vec<(usize, [usize; 3])> = Vec::with_capacity(shape.len());
    for d in (0..shape.len()).rev() {
        if shape[d] == 1 {
            continue;
        }
        let st = [ops[0].0[d], ops[1].0[d], ops[2].0[d]];
        if let Some(inner) = dims.last_mut() {
            if (0..3).all(|x| st[x] == inner.1[x] * inner.0) {
                inner.0 *= shape[d];
                continue;
            }
        }
        dims.push((shape[d], st));
    }
    dims.reverse();
    if dims.len() > MAXR {
        return Err(TensorError::Unsupported(format!("CUDA kernels take rank ≤ {MAXR}, not {}", dims.len())));
    }
    let mut p = Strided { n: u32_of(shape.iter().product(), "elements")?, rank: dims.len() as u32, off_a: u32_of(ops[0].1, "offset")?, off_b: u32_of(ops[1].1, "offset")?, off_c: u32_of(ops[2].1, "offset")?, shape: [1; MAXR], sa: [0; MAXR], sb: [0; MAXR], sc: [0; MAXR] };
    for (k, (size, st)) in dims.iter().enumerate() {
        p.shape[k] = u32_of(*size, "dimension")?;
        p.sa[k] = u32_of(st[0], "stride")?;
        p.sb[k] = u32_of(st[1], "stride")?;
        p.sc[k] = u32_of(st[2], "stride")?;
    }
    Ok(p)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScalarOp {
    op: u32,
    s: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Lanes {
    outer: u32,
    n: u32,
    inner: u32,
    m: u32,
}

fn lanes(l: [usize; 4]) -> Result<Lanes> {
    Ok(Lanes { outer: u32_of(l[0], "outer")?, n: u32_of(l[1], "n")?, inner: u32_of(l[2], "inner")?, m: u32_of(l[3], "m")? })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
    a_rs: u32,
    a_cs: u32,
    b_rs: u32,
    b_cs: u32,
    zbase: u32,
    /// KC blocks per product when split over k (0: not split).
    nkc: u32,
}

fn f32s(g: &GpuStorage) -> Result<()> {
    if g.dtype == DType::F32 { Ok(()) } else { Err(TensorError::DType(format!("a CUDA float kernel on {} storage", g.dtype.name()))) }
}

fn copy_kernel(dtype: DType) -> &'static str {
    match dtype.size_in_bytes() {
        1 => "copy_strided_8",
        2 => "copy_strided_16",
        4 => "copy_strided_32",
        _ => "copy_strided_64",
    }
}

/// Profile labels of the unary op codes (`UnaryOp::code`).
const UNARY_TAGS: [&str; 13] = ["unary:neg", "unary:exp", "unary:log", "unary:sqrt", "unary:recip", "unary:abs", "unary:sign", "unary:sigmoid", "unary:add_scalar", "unary:sub_scalar", "unary:mul_scalar", "unary:div_scalar", "unary:pow"];

/// The longest lane the device sort takes (keys and indices of the next power of two in 32 KB of
/// shared memory); longer lanes sort on the host.
const MAX_SORT: usize = 4096;

/// matrixmultiply's k block (the matmul kernels close their fma chains every KC).
const KC: usize = 256;

/// A 128-tile product with fewer output tiles than this splits over k (two waves of the RTX
/// 4060 Laptop's 24 SMs).
const SPLIT_BELOW_TILES: usize = 48;

/// Grid z is limited to 65 535: larger batches run as several launches.
const MAX_Z: usize = 65_535;

impl GpuBackend for CudaBackend {
    fn upload(&self, device: Device, s: &CpuStorage) -> Result<GpuStorage> {
        let mut c = context()?;
        let (m, dtype) = match s {
            CpuStorage::F32(v) => (c.upload(raw_bytes(v))?, DType::F32),
            CpuStorage::F16(v) => (c.upload(raw_bytes(v))?, DType::F16),
            CpuStorage::BF16(v) => (c.upload(raw_bytes(v))?, DType::BF16),
            CpuStorage::I64(v) => (c.upload(raw_bytes(v))?, DType::I64),
            CpuStorage::I32(v) => (c.upload(raw_bytes(v))?, DType::I32),
            CpuStorage::U8(v) => (c.upload(raw_bytes(v))?, DType::U8),
            CpuStorage::Bool(v) => (c.upload(raw_bytes(v))?, DType::Bool),
            CpuStorage::F64(_) => return Err(TensorError::Unsupported("f64 tensors on CUDA are not in the CUDA kernel set; cast to f32 first".into())),
        };
        Ok(storage(device, dtype, s.len(), m))
    }

    fn download(&self, g: &GpuStorage) -> Result<CpuStorage> {
        let mut c = context()?;
        let m = mem(g);
        let n = g.len;
        fn read<T: Copy + Default>(c: &mut Ctx, m: &Mem, n: usize) -> Result<Vec<T>> {
            let mut v = vec![T::default(); n];
            let bytes = std::mem::size_of_val(v.as_slice());
            assert!(bytes <= m.bytes);
            if bytes > 0 {
                // SAFETY: v has room for `bytes`; the stream is synchronised before v is read.
                cu(unsafe { cuMemcpyDtoHAsync_v2(v.as_mut_ptr() as *mut c_void, m.ptr, bytes, c.stream) }, "cuMemcpyDtoHAsync")?;
            }
            c.sync()?;
            Ok(v)
        }
        Ok(match g.dtype {
            DType::F32 => CpuStorage::F32(read(&mut c, m, n)?),
            DType::F16 => CpuStorage::F16(read::<u16>(&mut c, m, n)?.into_iter().map(F16).collect()),
            DType::BF16 => CpuStorage::BF16(read::<u16>(&mut c, m, n)?.into_iter().map(BF16).collect()),
            DType::I64 => CpuStorage::I64(read(&mut c, m, n)?),
            DType::I32 => CpuStorage::I32(read(&mut c, m, n)?),
            DType::U8 => CpuStorage::U8(read(&mut c, m, n)?),
            // The kernels write 0 or 1 only; uploads come from valid bools.
            DType::Bool => CpuStorage::Bool(read::<u8>(&mut c, m, n)?.into_iter().map(|x| x != 0).collect()),
            DType::F64 => unreachable!("no f64 on CUDA"),
        })
    }

    fn copy(&self, x: &GpuStorage, xl: &Layout) -> Result<GpuStorage> {
        let mut c = context()?;
        let n = xl.numel();
        let y = out(&mut c, x, x.dtype, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        c.run1(copy_kernel(x.dtype), &[Arg::Buf(mem(x)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn cast(&self, x: &GpuStorage, xl: &Layout, to: DType) -> Result<GpuStorage> {
        let kernel = match (x.dtype, to) {
            (a, b) if a == b => return self.copy(x, xl),
            (DType::F16, DType::F32) => "cast_f16_f32",
            (DType::BF16, DType::F32) => "cast_bf16_f32",
            (DType::F32, DType::F16) => "cast_f32_f16",
            (DType::F32, DType::BF16) => "cast_f32_bf16",
            (DType::Bool, DType::F32) => "cast_bool_f32",
            (DType::F16 | DType::BF16, DType::F16 | DType::BF16) => {
                let f = self.cast(x, xl, DType::F32)?;
                return self.cast(&f, &Layout::contiguous(xl.shape.clone()), to);
            }
            (a, b) => return Err(TensorError::Unsupported(format!("CUDA cast {} → {}", a.name(), b.name()))),
        };
        let mut c = context()?;
        let n = xl.numel();
        let y = out(&mut c, x, to, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        c.run1(kernel, &[Arg::Buf(mem(x)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn unary(&self, op: UnaryOp, x: &GpuStorage, xl: &Layout) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = context()?;
        let n = xl.numel();
        let y = out(&mut c, x, DType::F32, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        let (code, s) = op.code();
        let u = ScalarOp { op: code, s };
        c.tag = Some(UNARY_TAGS[code as usize]);
        c.run1("unary_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&u))], n)?;
        Ok(y)
    }

    fn compare(&self, op: CmpOp, x: &GpuStorage, xl: &Layout, s: f32) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = context()?;
        let n = xl.numel();
        let y = out(&mut c, x, DType::Bool, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        let code = match op {
            CmpOp::Gt => 0,
            CmpOp::Ge => 1,
            CmpOp::Lt => 2,
            CmpOp::Le => 3,
            CmpOp::Eq => 4,
        };
        let u = ScalarOp { op: code, s };
        c.run1("compare_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&u))], n)?;
        Ok(y)
    }

    fn binary(&self, op: BinaryOp, a: &GpuStorage, al: &Layout, b: &GpuStorage, bl: &Layout) -> Result<GpuStorage> {
        f32s(a)?;
        f32s(b)?;
        let mut c = context()?;
        let n = al.numel();
        let y = out(&mut c, a, DType::F32, n)?;
        let p = strided(&al.shape, Some(al), Some(bl), None)?;
        let code: u32 = match op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
        };
        c.tag = Some(["binary:add", "binary:sub", "binary:mul", "binary:div"][code as usize]);
        c.run1("binary_f32", &[Arg::Buf(mem(a)), Arg::Buf(mem(b)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&code))], n)?;
        Ok(y)
    }

    fn mask_fill(&self, x: &GpuStorage, xl: &Layout, mask: &GpuStorage, ml: &Layout, value: f32) -> Result<GpuStorage> {
        f32s(x)?;
        if mask.dtype != DType::Bool {
            return Err(TensorError::DType(format!("a mask of {} storage", mask.dtype.name())));
        }
        let mut c = context()?;
        let n = xl.numel();
        let y = out(&mut c, x, DType::F32, n)?;
        let p = strided(&xl.shape, Some(xl), Some(ml), None)?;
        c.run1("mask_fill_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(mask)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&value))], n)?;
        Ok(y)
    }

    fn reduce_dim(&self, op: ReduceOp, x: &GpuStorage, shape: &[usize], dim: usize) -> Result<GpuStorage> {
        f32s(x)?;
        let outer: usize = shape[..dim].iter().product();
        let n = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let mut c = context()?;
        let l = lanes([outer, n, inner, 0])?;
        match op {
            ReduceOp::Sum => {
                let y = out(&mut c, x, DType::F32, outer * inner)?;
                // CpuRef's rule: eight partial sums along the last axis only, in-order slice
                // additions along any other.
                if dim + 1 == shape.len() {
                    c.run1("sum_last_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&l))], outer)?;
                } else {
                    c.run1("sum_mid_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&l))], outer * inner)?;
                }
                Ok(y)
            }
            ReduceOp::Max | ReduceOp::ArgMax => {
                if n == 0 {
                    return Err(TensorError::Shape("max or argmax along an empty dimension".into()));
                }
                let idx = out(&mut c, x, DType::I32, outer * inner)?;
                let val = out(&mut c, x, DType::F32, outer * inner)?;
                c.run1("argmax_dim_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(&idx)), Arg::Buf(mem(&val)), Arg::Bytes(bytes_of(&l))], outer * inner)?;
                Ok(if op == ReduceOp::Max { val } else { idx })
            }
        }
    }

    fn reduce_all(&self, op: ReduceOp, x: &GpuStorage, n: usize) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = context()?;
        let y = out(&mut c, x, DType::F32, 1)?;
        let nn = u32_of(n, "elements")?;
        let kernel = match op {
            ReduceOp::Sum => "sum_all_f32",
            ReduceOp::Max if n > 0 => "max_all_f32",
            ReduceOp::Max => return Err(TensorError::Shape("max of an empty tensor".into())),
            ReduceOp::ArgMax => return Err(TensorError::Unsupported("argmax over all elements".into())),
        };
        c.run1(kernel, &[Arg::Buf(mem(x)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&nn))], 1)?;
        Ok(y)
    }

    fn matmul(&self, a: &GpuStorage, al: &Layout, b: &GpuStorage, bl: &Layout) -> Result<(GpuStorage, Vec<usize>)> {
        f32s(a)?;
        f32s(b)?;
        let (ash, bsh) = (&al.shape, &bl.shape);
        let r = ash.len();
        if r < 2 || bsh.len() != r || ash[r - 1] != bsh[r - 2] {
            return Err(TensorError::Shape(format!("matmul {ash:?} × {bsh:?}")));
        }
        let (m, k, n) = (ash[r - 2], ash[r - 1], bsh[r - 1]);
        let batch = crate::tensor::shape::broadcast(&ash[..r - 2], &bsh[..r - 2]);
        let nb: usize = batch.iter().product();
        // Per output batch (row-major), the element offsets of its A and B blocks.
        let mut offs: Vec<u32> = Vec::with_capacity(nb * 2);
        let mut idx = vec![0usize; batch.len()];
        for _ in 0..nb {
            let at = |l: &Layout| -> usize { l.offset + idx.iter().enumerate().map(|(i, &q)| if l.shape[i] == 1 { 0 } else { q * l.strides[i] }).sum::<usize>() };
            offs.push(u32_of(at(al), "offset")?);
            offs.push(u32_of(at(bl), "offset")?);
            for d in (0..batch.len()).rev() {
                idx[d] += 1;
                if idx[d] < batch[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        let mut shape = batch.clone();
        shape.extend([m, n]);
        let mut c = context()?;
        let y = out(&mut c, a, DType::F32, nb * m * n)?;
        if nb > 0 && m > 0 && n > 0 {
            u32_of(nb * m * n, "elements")?;
            let ob = c.upload(raw_bytes(&offs))?;
            let mut p = MatmulParams { m: u32_of(m, "m")?, n: u32_of(n, "n")?, k: u32_of(k, "k")?, a_rs: u32_of(al.strides[r - 2], "stride")?, a_cs: u32_of(al.strides[r - 1], "stride")?, b_rs: u32_of(bl.strides[r - 2], "stride")?, b_cs: u32_of(bl.strides[r - 1], "stride")?, zbase: 0, nkc: 0 };
            // 128×128 tiles for large outputs; the same per-output order, so the same bits.
            let tile = if m >= 128 && n >= 128 { 128 } else { 64 };
            let kernel = if tile == 128 { "matmul128_f32" } else { "matmul_f32" };
            let (gx, gy) = (u32_of(n.div_ceil(tile), "grid")?, u32_of(m.div_ceil(tile), "grid")?);
            // A long k with few output tiles splits over its KC blocks (each block's chain into
            // its own slice, then an in-order fold): the unsplit kernel's arithmetic, more blocks.
            let nkc = k.div_ceil(KC);
            let split = tile == 128 && nkc > 1 && (gx as usize) * (gy as usize) * nb < SPLIT_BELOW_TILES && nb * nkc <= MAX_Z;
            c.tag = Some(match (p.a_cs == 1, p.b_cs == 1, tile == 128, split) {
                (_, _, _, true) => "matmul128 split-k",
                (true, true, true, _) => "matmul128 A·B",
                (false, true, true, _) => "matmul128 Aᵀ·B",
                (true, false, true, _) => "matmul128 A·Bᵀ",
                (false, false, true, _) => "matmul128 Aᵀ·Bᵀ",
                _ => "matmul64",
            });
            if split {
                p.nkc = nkc as u32;
                let parts = c.alloc(nb * nkc * m * n * 4)?;
                c.launch(kernel, (gx, gy, (nb * nkc) as u32), 256, &[Arg::Buf(mem(a)), Arg::Buf(mem(b)), Arg::Buf(&parts), Arg::Buf(&ob), Arg::Bytes(bytes_of(&p))])?;
                let (nkc32, mn, nb32) = (nkc as u32, u32_of(m * n, "elements")?, nb as u32);
                c.run1("kc_reduce_f32", &[Arg::Buf(&parts), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&nkc32)), Arg::Bytes(bytes_of(&mn)), Arg::Bytes(bytes_of(&nb32))], nb * m * n)?;
            } else {
                let mut z0 = 0;
                while z0 < nb {
                    let nz = (nb - z0).min(MAX_Z);
                    p.zbase = z0 as u32;
                    c.launch(kernel, (gx, gy, nz as u32), 256, &[Arg::Buf(mem(a)), Arg::Buf(mem(b)), Arg::Buf(mem(&y)), Arg::Buf(&ob), Arg::Bytes(bytes_of(&p))])?;
                    z0 += nz;
                }
            }
        }
        Ok((y, shape))
    }

    fn gather(&self, x: &GpuStorage, l: [usize; 4], idx: &GpuStorage) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = context()?;
        let n = l[0] * l[3] * l[2];
        let y = out(&mut c, x, DType::F32, n)?;
        let p = lanes(l)?;
        c.run1("gather_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(idx)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn index_select(&self, x: &GpuStorage, l: [usize; 4], idx: &GpuStorage) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = context()?;
        let n = l[0] * l[3] * l[2];
        let y = out(&mut c, x, DType::F32, n)?;
        let p = lanes(l)?;
        c.run1("index_select_f32", &[Arg::Buf(mem(x)), Arg::Buf(mem(idx)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn one_hot(&self, idx: &GpuStorage, count: usize, n: usize) -> Result<GpuStorage> {
        let mut c = context()?;
        let y = out(&mut c, idx, DType::F32, count * n)?;
        let p = lanes([count, n, 1, 0])?;
        c.run1("one_hot_f32", &[Arg::Buf(mem(idx)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p))], count * n)?;
        Ok(y)
    }

    fn scatter_add(&self, idx: &GpuStorage, vals: &GpuStorage, l: [usize; 4]) -> Result<GpuStorage> {
        f32s(vals)?;
        let mut c = context()?;
        let n = l[0] * l[1] * l[2];
        let y = out(&mut c, vals, DType::F32, n)?;
        let p = lanes(l)?;
        c.run1("scatter_add_f32", &[Arg::Buf(mem(idx)), Arg::Buf(mem(vals)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn index_add(&self, idx: &GpuStorage, vals: &GpuStorage, l: [usize; 4]) -> Result<GpuStorage> {
        f32s(vals)?;
        let mut c = context()?;
        let n = l[0] * l[1] * l[2];
        let y = out(&mut c, vals, DType::F32, n)?;
        let p = lanes(l)?;
        c.run1("index_add_f32", &[Arg::Buf(mem(idx)), Arg::Buf(mem(vals)), Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn write(&self, dst: &GpuStorage, dl: &Layout, src: &GpuStorage, sl: &Layout) -> Result<()> {
        if dst.dtype != src.dtype || dl.shape != sl.shape {
            return Err(TensorError::Shape(format!("write of {} {:?} into {} {:?}", src.dtype.name(), sl.shape, dst.dtype.name(), dl.shape)));
        }
        let mut c = context()?;
        let p = strided(&sl.shape, Some(sl), None, Some(dl))?;
        c.run1(copy_kernel(src.dtype), &[Arg::Buf(mem(src)), Arg::Buf(mem(dst)), Arg::Bytes(bytes_of(&p))], sl.numel())
    }

    fn alloc(&self, device: Device, dtype: DType, len: usize) -> Result<GpuStorage> {
        let mut c = context()?;
        Ok(storage(device, dtype, len, c.alloc(len * dtype.size_in_bytes())?))
    }

    fn sync(&self) -> Result<()> {
        context()?.sync()
    }

    fn sort_desc(&self, x: &GpuStorage, l: [usize; 3]) -> Result<(GpuStorage, GpuStorage)> {
        f32s(x)?;
        let (outer, n, inner) = (l[0], l[1], l[2]);
        let np2 = n.max(1).next_power_of_two();
        if np2 > MAX_SORT {
            // Longer lanes: the trait's counted host round trip.
            return crate::tensor::gpu::sort_desc_on_host(self, x, l);
        }
        let mut c = context()?;
        let vals = out(&mut c, x, DType::F32, outer * n * inner)?;
        let idx = out(&mut c, x, DType::I32, outer * n * inner)?;
        let p = lanes([outer, n, inner, 0])?;
        let np2u = np2 as u32;
        let lanes_n = u32_of(outer * inner, "lanes")?;
        if n > 0 {
            c.launch_shared("sort_desc_f32", (lanes_n, 1, 1), np2.min(512) as u32, (np2 * 8) as u32, &[Arg::Buf(mem(x)), Arg::Buf(mem(&vals)), Arg::Buf(mem(&idx)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&np2u))])?;
        }
        Ok((vals, idx))
    }

    fn fill(&self, device: Device, len: usize, value: f32) -> Result<GpuStorage> {
        let mut c = context()?;
        let y = storage(device, DType::F32, len, c.alloc(len * 4)?);
        let n = u32_of(len, "elements")?;
        c.run1("fill_f32", &[Arg::Buf(mem(&y)), Arg::Bytes(bytes_of(&value)), Arg::Bytes(bytes_of(&n))], len)?;
        Ok(y)
    }

    fn adam_update(&self, p: &GpuStorage, g: &GpuStorage, m: &GpuStorage, v: &GpuStorage, n: usize, s: &AdamScalars) -> Result<(GpuStorage, GpuStorage, GpuStorage)> {
        for x in [p, g, m, v] {
            f32s(x)?;
        }
        let mut c = context()?;
        let (po, mo, vo) = (out(&mut c, p, DType::F32, n)?, out(&mut c, p, DType::F32, n)?, out(&mut c, p, DType::F32, n)?);
        let nn = u32_of(n, "elements")?;
        c.run1(
            "adam_f32",
            &[Arg::Buf(mem(p)), Arg::Buf(mem(g)), Arg::Buf(mem(m)), Arg::Buf(mem(v)), Arg::Buf(mem(&po)), Arg::Buf(mem(&mo)), Arg::Buf(mem(&vo)), Arg::Bytes(bytes_of(s)), Arg::Bytes(bytes_of(&nn))],
            n,
        )?;
        Ok((po, mo, vo))
    }

    fn key(&self) -> Result<String> {
        let c = context()?;
        let src = crate::hash::sha256_hex(super::kernels::SOURCE.as_bytes());
        Ok(format!("cuda-{}-sm{}{}-driver{}-nvrtc{}.{}-nofastmath-fmadoff-kernels-{}", c.name.replace(' ', "_"), c.cc.0, c.cc.1, c.driver, c.nvrtc.0, c.nvrtc.1, &src[..16]))
    }
}

/// Start profiling the CUDA device's work: events around every launch, host time in
/// launches, allocations and frees. Clears an earlier profile.
pub fn profile_start() -> Result<()> {
    let mut c = context()?;
    c.prof.on = true;
    let spent: Vec<CUevent> = c.prof.events.drain(..).flat_map(|(_, a, b)| [a, b]).collect();
    let spare = std::mem::take(&mut c.prof.spare);
    c.prof = Prof { on: true, spare: spare.into_iter().chain(spent).collect(), ..Prof::default() };
    Ok(())
}

/// Stop profiling and report (waits for the queued work).
pub fn profile_take() -> Result<ProfileReport> {
    let mut c = context()?;
    c.sync()?;
    let mut per: HashMap<&'static str, (u64, f64)> = HashMap::new();
    let mut busy = 0f64;
    let elapsed = |a: CUevent, b: CUevent| -> Result<f64> {
        let mut ms = 0f32;
        // SAFETY: both events were recorded on the stream, which is synchronised.
        cu(unsafe { cuEventElapsedTime(&mut ms, a, b) }, "cuEventElapsedTime")?;
        Ok(ms as f64)
    };
    for &(k, a, b) in &c.prof.events {
        let ms = elapsed(a, b)?;
        let e = per.entry(k).or_default();
        e.0 += 1;
        e.1 += ms;
        busy += ms;
    }
    let span = match (c.prof.events.first(), c.prof.events.last()) {
        (Some(f), Some(l)) => elapsed(f.1, l.2)?,
        _ => 0.0,
    };
    let mut kernels: Vec<(String, u64, f64)> = per.into_iter().map(|(k, (n, ms))| (k.to_string(), n, ms)).collect();
    kernels.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let p = &c.prof;
    let r = ProfileReport {
        kernels,
        launches: p.events.len() as u64,
        launch_host_ms: p.launch_ns as f64 / 1e6,
        allocs: p.allocs,
        alloc_bytes: p.alloc_bytes,
        alloc_host_ms: p.alloc_ns as f64 / 1e6,
        frees: p.frees,
        free_host_ms: p.free_ns as f64 / 1e6,
        gpu_busy_ms: busy,
        gpu_span_ms: span,
    };
    let spent: Vec<CUevent> = c.prof.events.drain(..).flat_map(|(_, a, b)| [a, b]).collect();
    c.prof.spare.extend(spent);
    c.prof.on = false;
    Ok(r)
}

#[cfg(test)]
/// Kernels launched on the CUDA device so far (for reports).
pub(crate) fn launches() -> Result<u64> {
    Ok(context()?.launches)
}

// The kernels' by-value structs: 37, 2, 4 and 8 32-bit words, as the CUDA source declares them.
const _: () = assert!(std::mem::size_of::<Strided>() == 148);
const _: () = assert!(std::mem::size_of::<ScalarOp>() == 8);
const _: () = assert!(std::mem::size_of::<Lanes>() == 16);
const _: () = assert!(std::mem::size_of::<MatmulParams>() == 36);
const _: () = assert!(std::mem::size_of::<AdamScalars>() == 48);
