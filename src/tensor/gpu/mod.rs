//! GPU backends, behind off-by-default cargo features: `metal` (macOS) and
//! `cuda` (the CUDA driver, NVRTC; Device::Cuda(0)). No
//! crate is added: the system frameworks are reached through hand-written FFI.
//!
//! `BackendStorage` (the CPU kernel set) takes host slices and Rust closures, which a GPU cannot
//! run, so a GPU backend implements `GpuBackend`: the same operations named by op codes
//! (`UnaryOp`, `CmpOp`, `BinaryOp`, `ReduceOp`) over `GpuStorage` and a `Layout` (views are read
//! in place through their strides). Every tensor op branches here before it reads host values
//! (`gpu_*` below return None for a CPU tensor, so the CPU path runs unchanged); CpuRef is not
//! touched. Autodiff stays above storage.
//!
//! Coverage (forward): elementwise ops, comparisons, mask_fill, lane reductions (CpuRef's order:
//! sums equal CpuRef's), batched matmul (matrixmultiply's order), gather, index_select,
//! one_hot, deterministic scatter_add and index_add, strided copies, casts. Sort (and so topk)
//! runs on the host through a counted round trip. Floats compute in f32 (f16 and bf16 are cast
//! to f32 first, as on the CPU); F64 is `TensorError::Unsupported` on Metal.

// Without a GPU feature `GpuStorage` is uninhabited, so every GPU branch below is dead code.
#![cfg_attr(not(any(all(feature = "metal", target_os = "macos"), feature = "cuda")), allow(unreachable_code, unused_variables, unused_assignments, unreachable_patterns, dead_code))]

use super::device::Device;
use super::dtype::DType;
use super::error::{Result, TensorError};
use super::layout::Layout;
use super::storage::{CpuStorage, Storage};
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

/// A device buffer. Without a GPU feature this enum is empty, so `GpuStorage` cannot exist and
/// every GPU branch is dead code.
#[derive(Clone, Debug)]
pub(crate) enum GpuBuf {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(std::sync::Arc<super::metal::Buffer>),
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<super::cuda::backend::Mem>),
}

/// Elements on a GPU: `len` elements of `dtype` (f32, f16/bf16 bits, bool and u8 as bytes,
/// i32, i64), row-major, read through the tensor's layout as CPU storage is.
#[derive(Clone, Debug)]
pub struct GpuStorage {
    pub(crate) device: Device,
    pub(crate) dtype: DType,
    pub(crate) len: usize,
    pub(crate) buf: GpuBuf,
}

impl GpuStorage {
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum UnaryOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Recip,
    Abs,
    Sign,
    /// The stable sigmoid in f32: `1 / (1 + e^−x)` for x ≥ 0, `e^x / (1 + e^x)` below.
    Sigmoid,
    AddScalar(f32),
    SubScalar(f32),
    MulScalar(f32),
    DivScalar(f32),
    PowScalar(f32),
}

impl UnaryOp {
    /// The kernel's op code and scalar.
    pub(crate) fn code(self) -> (u32, f32) {
        match self {
            UnaryOp::Neg => (0, 0.0),
            UnaryOp::Exp => (1, 0.0),
            UnaryOp::Log => (2, 0.0),
            UnaryOp::Sqrt => (3, 0.0),
            UnaryOp::Recip => (4, 0.0),
            UnaryOp::Abs => (5, 0.0),
            UnaryOp::Sign => (6, 0.0),
            UnaryOp::Sigmoid => (7, 0.0),
            UnaryOp::AddScalar(s) => (8, s),
            UnaryOp::SubScalar(s) => (9, s),
            UnaryOp::MulScalar(s) => (10, s),
            UnaryOp::DivScalar(s) => (11, s),
            UnaryOp::PowScalar(s) => (12, s),
        }
    }
}

/// A comparison with a scalar (a bool output).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A reduction along one dimension, or over all elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReduceOp {
    /// CpuRef's sum: eight partial sums along the last axis, in-order slices along any other.
    Sum,
    /// The value at the first maximum.
    Max,
    /// The index of the first maximum (I32).
    ArgMax,
}

/// The kernel set a GPU device runs. Float kernels take F32 storage (the
/// dispatch below casts f16/bf16 first); outputs are contiguous and row-major.
pub(crate) trait GpuBackend: Sync {
    fn upload(&self, device: Device, s: &CpuStorage) -> Result<GpuStorage>;
    fn download(&self, s: &GpuStorage) -> Result<CpuStorage>;
    /// The elements a view selects, contiguous (any dtype).
    fn copy(&self, x: &GpuStorage, xl: &Layout) -> Result<GpuStorage>;
    /// Convert between f32, f16 and bf16, or bool to f32 (1.0 / 0.0) (contiguous
    /// output).
    fn cast(&self, x: &GpuStorage, xl: &Layout, to: DType) -> Result<GpuStorage>;
    fn unary(&self, op: UnaryOp, x: &GpuStorage, xl: &Layout) -> Result<GpuStorage>;
    /// Bool storage.
    fn compare(&self, op: CmpOp, x: &GpuStorage, xl: &Layout, s: f32) -> Result<GpuStorage>;
    /// `a op b` over the broadcast shape `out` (layouts already broadcast to `out`).
    fn binary(&self, op: BinaryOp, a: &GpuStorage, al: &Layout, b: &GpuStorage, bl: &Layout) -> Result<GpuStorage>;
    /// `value` where the (bool) mask, broadcast to x's shape by its layout, is true.
    fn mask_fill(&self, x: &GpuStorage, xl: &Layout, mask: &GpuStorage, ml: &Layout, value: f32) -> Result<GpuStorage>;
    /// Along `dim` of a contiguous `[outer, n, inner]` reading, keeping the dimension (size 1).
    fn reduce_dim(&self, op: ReduceOp, x: &GpuStorage, shape: &[usize], dim: usize) -> Result<GpuStorage>;
    /// Sum or Max of every element of contiguous storage (one element).
    fn reduce_all(&self, op: ReduceOp, x: &GpuStorage, n: usize) -> Result<GpuStorage>;
    /// Batched matmul of two views (batch dimensions broadcast when 1); the output shape.
    fn matmul(&self, a: &GpuStorage, al: &Layout, b: &GpuStorage, bl: &Layout) -> Result<(GpuStorage, Vec<usize>)>;
    /// `gather` of contiguous x `[outer, n, inner]` at I32 indices `[outer, m, inner]`.
    fn gather(&self, x: &GpuStorage, lanes: [usize; 4], idx: &GpuStorage) -> Result<GpuStorage>;
    /// `index_select` of contiguous x `[outer, n, inner]` at `m` I32 indices.
    fn index_select(&self, x: &GpuStorage, lanes: [usize; 4], idx: &GpuStorage) -> Result<GpuStorage>;
    /// One-hot f32 rows of width `n` for `count` I32 indices.
    fn one_hot(&self, idx: &GpuStorage, count: usize, n: usize) -> Result<GpuStorage>;
    /// scatter_add of values `[outer, m, inner]` into zeros `[outer, n, inner]`.
    fn scatter_add(&self, idx: &GpuStorage, vals: &GpuStorage, lanes: [usize; 4]) -> Result<GpuStorage>;
    /// index_add of value slices `[outer, m, inner]` into zeros `[outer, n, inner]`.
    fn index_add(&self, idx: &GpuStorage, vals: &GpuStorage, lanes: [usize; 4]) -> Result<GpuStorage>;
    /// Copy the view `src` into the view `dst` of a buffer this op owns (slice_assign, cat).
    fn write(&self, dst: &GpuStorage, dl: &Layout, src: &GpuStorage, sl: &Layout) -> Result<()>;
    /// Uninitialised storage.
    fn alloc(&self, device: Device, dtype: DType, len: usize) -> Result<GpuStorage>;
    /// Wait for all encoded work.
    fn sync(&self) -> Result<()>;
    /// Descending sort along the middle of a contiguous f32 `[outer, n, inner]`: the values and
    /// their I32 source positions, ties in first-index order (the default is the
    /// counted host round trip, so a backend without a sort kernel keeps working).
    fn sort_desc(&self, x: &GpuStorage, lanes: [usize; 3]) -> Result<(GpuStorage, GpuStorage)> {
        sort_desc_on_host(self, x, lanes)
    }
    /// `len` f32 elements equal to `value` (the default uploads a host vector, a
    /// backend with a fill kernel overrides it).
    fn fill(&self, device: Device, len: usize, value: f32) -> Result<GpuStorage> {
        self.upload(device, &CpuStorage::F32(vec![value; len]))
    }
    /// The device's provenance key.
    fn key(&self) -> Result<String>;
    /// Optional: nn::Adam's plain update of contiguous f32 `p`, `g`, `m`, `v` (n elements)
    /// fused into one kernel with the op sequence's f32 arithmetic; returns (p, m, v). The default
    /// is Unsupported: the caller runs the op sequence.
    fn adam_update(&self, p: &GpuStorage, g: &GpuStorage, m: &GpuStorage, v: &GpuStorage, n: usize, s: &AdamScalars) -> Result<(GpuStorage, GpuStorage, GpuStorage)> {
        let _ = (p, g, m, v, n, s);
        Err(TensorError::Unsupported("no fused Adam on this backend".into()))
    }
}

/// The scalars of one fused Adam update, each the f32 value the op sequence uses (`s as f32` of
/// its f64 expression). `flags`: 1 coupled decay, 2 decay before the step, 4 decay after it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AdamScalars {
    pub wd: f32,
    pub b1: f32,
    pub omb1: f32,
    pub b2: f32,
    pub omb2: f32,
    pub bc1: f32,
    pub bc2: f32,
    pub eps: f32,
    pub lr: f32,
    pub pre: f32,
    pub post: f32,
    pub flags: u32,
}

/// The backend of a GPU device (Unsupported when not built or not a GPU).
pub(crate) fn backend_for(device: Device) -> Result<&'static dyn GpuBackend> {
    match device {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Device::Metal(0) => Ok(&super::metal::backend::METAL),
        Device::Metal(i) if cfg!(all(feature = "metal", target_os = "macos")) => Err(TensorError::Unsupported(format!("Metal({i}): this machine has one Metal device, Metal(0)"))),
        Device::Metal(i) => Err(TensorError::Unsupported(format!("Metal({i}): this build has no Metal backend (the `metal` feature, macOS)"))),
        #[cfg(feature = "cuda")]
        Device::Cuda(0) => Ok(&super::cuda::backend::CUDA),
        Device::Cuda(i) if cfg!(feature = "cuda") => Err(TensorError::Unsupported(format!("Cuda({i}): the CUDA backend runs on Cuda(0) only"))),
        Device::Cuda(i) => Err(TensorError::Unsupported(format!("Cuda({i}): this build has no CUDA backend (the `cuda` feature)"))),
        Device::Cpu(_) => Err(TensorError::Unsupported("the CPU is not a GPU backend".into())),
    }
}

/// Whether a GPU device can be used in this process (built, found, and the process unpinned).
pub(crate) fn check_device(device: Device) -> Result<()> {
    let b = backend_for(device)?;
    b.key().map(drop)
}

fn be(g: &GpuStorage) -> &'static dyn GpuBackend {
    match g.buf {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        GpuBuf::Metal(_) => &super::metal::backend::METAL,
        #[cfg(feature = "cuda")]
        GpuBuf::Cuda(_) => &super::cuda::backend::CUDA,
    }
}

// ---------------------------------------------------------------- counters

static UPLOADS: AtomicU64 = AtomicU64::new(0);
static DOWNLOADS: AtomicU64 = AtomicU64::new(0);
static ROUND_TRIPS: AtomicU64 = AtomicU64::new(0);

/// Host ↔ GPU traffic of this process: (uploads, downloads, counted host round trips of ops
/// without a GPU kernel, such as sort).
pub fn transfer_counts() -> (u64, u64, u64) {
    (UPLOADS.load(Ordering::Relaxed), DOWNLOADS.load(Ordering::Relaxed), ROUND_TRIPS.load(Ordering::Relaxed))
}

pub(crate) fn count_round_trip() {
    ROUND_TRIPS.fetch_add(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------- storage placement

/// Whether `storage` lives where `device` computes.
pub(crate) fn agrees(storage: &Storage, device: Device) -> bool {
    match (storage, device) {
        (Storage::Cpu(_), Device::Cpu(_)) => true,
        (Storage::Gpu(g), d) => g.device == d,
        _ => false,
    }
}

/// `storage` moved to `device` (the same storage when it is already there).
pub(crate) fn place(storage: &std::sync::Arc<Storage>, device: Device) -> Result<std::sync::Arc<Storage>> {
    if agrees(storage, device) {
        return Ok(storage.clone());
    }
    let host: Cow<'_, CpuStorage> = match &**storage {
        Storage::Cpu(c) => Cow::Borrowed(c),
        Storage::Gpu(g) => Cow::Owned(download(g)?),
    };
    match device {
        Device::Cpu(_) => Ok(std::sync::Arc::new(Storage::Cpu(host.into_owned()))),
        d => {
            if host.dtype() == DType::F64 && matches!(d, Device::Metal(_)) {
                return Err(TensorError::Unsupported("f64 tensors cannot live on Metal (Metal has no f64); cast to f32 first".into()));
            }
            if host.dtype() == DType::F64 && matches!(d, Device::Cuda(_)) {
                return Err(TensorError::Unsupported("f64 tensors are not supported on CUDA; cast to f32 first".into()));
            }
            let g = backend_for(d)?.upload(d, &host)?;
            UPLOADS.fetch_add(1, Ordering::Relaxed);
            Ok(std::sync::Arc::new(Storage::Gpu(g)))
        }
    }
}

/// Host storage from a GPU (waits for encoded work).
pub(crate) fn download(g: &GpuStorage) -> Result<CpuStorage> {
    DOWNLOADS.fetch_add(1, Ordering::Relaxed);
    be(g).download(g)
}

/// Upload host storage to `device`.
pub(crate) fn upload(device: Device, s: CpuStorage) -> Result<GpuStorage> {
    UPLOADS.fetch_add(1, Ordering::Relaxed);
    backend_for(device)?.upload(device, &s)
}

// ---------------------------------------------------------------- op dispatch (tensor level)

use super::error::ok;
use super::{BoolTensor, IntTensor, Tensor};

/// The GPU storage of a tensor, if it has one.
pub(crate) fn gpu_of(s: &Storage) -> Option<&GpuStorage> {
    match s {
        Storage::Gpu(g) => Some(g),
        Storage::Cpu(_) => None,
    }
}

/// An f32 operand: the storage itself, or f16/bf16 cast to f32 (contiguous); f64 is refused.
fn f32_operand<'a>(g: &'a GpuStorage, l: &'a Layout) -> Result<(Cow<'a, GpuStorage>, Cow<'a, Layout>)> {
    match g.dtype {
        DType::F32 => Ok((Cow::Borrowed(g), Cow::Borrowed(l))),
        DType::F16 | DType::BF16 => Ok((Cow::Owned(be(g).cast(g, l, DType::F32)?), Cow::Owned(Layout::contiguous(l.shape.clone())))),
        other => Err(TensorError::Unsupported(format!("{} storage in a float GPU kernel", other.name()))),
    }
}

/// A contiguous f32 operand (reductions and indexing read `[outer, n, inner]` blocks).
fn f32_contiguous(g: &GpuStorage, l: &Layout) -> Result<GpuStorage> {
    let (x, xl) = f32_operand(g, l)?;
    if xl.is_contiguous() && xl.offset == 0 && x.len == xl.numel() {
        Ok(x.into_owned())
    } else {
        be(g).copy(&x, &xl)
    }
}

/// Indices as I32 storage on `device` (values already checked by `infer`).
fn indices_on(device: Device, idx: &[i64]) -> Result<GpuStorage> {
    upload(device, CpuStorage::I32(idx.iter().map(|&j| j as i32).collect()))
}

/// Split `shape` at `dim` into (outer, n, inner).
fn lanes(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    (shape[..dim].iter().product(), shape[dim], shape[dim + 1..].iter().product())
}

pub(crate) fn gpu_unary(t: &Tensor, op: UnaryOp) -> Option<Storage> {
    let g = gpu_of(&t.storage)?;
    Some(Storage::Gpu(ok((|| {
        let (x, xl) = f32_operand(g, &t.layout)?;
        be(g).unary(op, &x, &xl)
    })())))
}

pub(crate) fn gpu_compare(t: &Tensor, op: CmpOp, s: f64) -> Option<BoolTensor> {
    let g = gpu_of(&t.storage)?;
    let st = ok((|| {
        let (x, xl) = f32_operand(g, &t.layout)?;
        be(g).compare(op, &x, &xl, s as f32)
    })());
    Some(BoolTensor { storage: std::sync::Arc::new(Storage::Gpu(st)), layout: Layout::contiguous(t.layout.shape.clone()) })
}

pub(crate) fn gpu_binary(a: &Tensor, b: &Tensor, op: BinaryOp) -> Option<(Storage, Vec<usize>)> {
    let ga = gpu_of(&a.storage)?;
    let gb = ok(gpu_of(&b.storage).ok_or_else(|| TensorError::Unsupported("a GPU tensor with a CPU operand".into())));
    let out = super::shape::broadcast(&a.layout.shape, &b.layout.shape);
    let st = ok((|| {
        let (x, xl) = f32_operand(ga, &a.layout)?;
        let (y, yl) = f32_operand(gb, &b.layout)?;
        be(ga).binary(op, &x, &xl.broadcast(&out), &y, &yl.broadcast(&out))
    })());
    Some((Storage::Gpu(st), out))
}

pub(crate) fn gpu_mask_fill(t: &Tensor, mask: &BoolTensor, value: f64) -> Option<Storage> {
    let g = gpu_of(&t.storage)?;
    Some(Storage::Gpu(ok((|| {
        let (x, xl) = f32_operand(g, &t.layout)?;
        let mask_dev = mask_on(g.device, mask)?;
        let ml = mask_dev.1.broadcast(&t.layout.shape);
        be(g).mask_fill(&x, &xl, &mask_dev.0, &ml, value as f32)
    })())))
}

/// A mask's storage on `device` (uploaded when it is on the host) and its layout.
pub(crate) fn mask_on(device: Device, mask: &BoolTensor) -> Result<(GpuStorage, Layout)> {
    match &*mask.storage {
        Storage::Gpu(m) if m.device == device => Ok((m.clone(), mask.layout.clone())),
        _ => Ok((upload(device, CpuStorage::Bool(mask.data().into_owned()))?, Layout::contiguous(mask.layout.shape.clone()))),
    }
}

/// A bool tensor moved to `device` (masks captured for backward stay where they were computed).
pub(crate) fn bool_on(device: Device, mask: &BoolTensor) -> BoolTensor {
    let (s, l) = ok(mask_on(device, mask));
    BoolTensor { storage: std::sync::Arc::new(Storage::Gpu(s)), layout: l }
}

pub(crate) fn gpu_sum_dim(t: &Tensor, dim: usize) -> Option<(Storage, Vec<usize>)> {
    let g = gpu_of(&t.storage)?;
    let x = ok(f32_contiguous(g, &t.layout));
    let st = ok(be(g).reduce_dim(ReduceOp::Sum, &x, &t.layout.shape, dim));
    let mut out = t.layout.shape.clone();
    out[dim] = 1;
    Some((Storage::Gpu(st), out))
}

/// Sum (or Max) of every element, one element.
pub(crate) fn gpu_reduce_all(t: &Tensor, op: ReduceOp) -> Option<GpuStorage> {
    let g = gpu_of(&t.storage)?;
    let x = ok(f32_contiguous(g, &t.layout));
    Some(ok(be(g).reduce_all(op, &x, t.layout.numel())))
}

/// max_dim: the values (Storage) and the I32 argmax indices (kept on the device for backward).
pub(crate) fn gpu_max_dim(t: &Tensor, dim: usize) -> Option<(Storage, GpuStorage, Vec<usize>)> {
    let g = gpu_of(&t.storage)?;
    let x = ok(f32_contiguous(g, &t.layout));
    let idx = ok(be(g).reduce_dim(ReduceOp::ArgMax, &x, &t.layout.shape, dim));
    let val = ok(be(g).reduce_dim(ReduceOp::Max, &x, &t.layout.shape, dim));
    let mut out = t.layout.shape.clone();
    out[dim] = 1;
    Some((Storage::Gpu(val), idx, out))
}

/// argmax along `dim` (read back to the host: int tensors live on the host).
pub(crate) fn gpu_argmax(t: &Tensor, dim: usize) -> Option<(Vec<i64>, Vec<usize>)> {
    let g = gpu_of(&t.storage)?;
    let x = ok(f32_contiguous(g, &t.layout));
    let idx = ok(be(g).reduce_dim(ReduceOp::ArgMax, &x, &t.layout.shape, dim));
    let v = match ok(download(&idx)) {
        CpuStorage::I32(v) => v.into_iter().map(|j| j as i64).collect(),
        _ => unreachable!("argmax gives I32"),
    };
    let mut out = t.layout.shape.clone();
    out[dim] = 1;
    Some((v, out))
}

pub(crate) fn gpu_matmul(a: &Tensor, b: &Tensor) -> Option<(Storage, Vec<usize>)> {
    let ga = gpu_of(&a.storage)?;
    let gb = ok(gpu_of(&b.storage).ok_or_else(|| TensorError::Unsupported("a GPU tensor with a CPU operand".into())));
    let (st, sh) = ok((|| {
        let (x, xl) = f32_operand(ga, &a.layout)?;
        let (y, yl) = f32_operand(gb, &b.layout)?;
        be(ga).matmul(&x, &xl, &y, &yl)
    })());
    Some((Storage::Gpu(st), sh))
}

pub(crate) fn gpu_gather(t: &Tensor, dim: usize, idx: &IntTensor) -> Option<Storage> {
    let g = gpu_of(&t.storage)?;
    Some(Storage::Gpu(ok((|| {
        let x = f32_contiguous(g, &t.layout)?;
        let (outer, n, inner) = lanes(&t.layout.shape, dim);
        let m = idx.layout.shape[dim];
        let i = indices_on(g.device, &idx.data())?;
        be(g).gather(&x, [outer, n, inner, m], &i)
    })())))
}

pub(crate) fn gpu_select(t: &Tensor, dim: usize, idx: &IntTensor) -> Option<(Storage, Vec<usize>)> {
    let g = gpu_of(&t.storage)?;
    let ids = idx.data();
    let st = ok((|| {
        let x = f32_contiguous(g, &t.layout)?;
        let (outer, n, inner) = lanes(&t.layout.shape, dim);
        let i = indices_on(g.device, &ids)?;
        be(g).index_select(&x, [outer, n, inner, ids.len()], &i)
    })());
    let mut out = t.layout.shape.clone();
    out[dim] = ids.len();
    Some((Storage::Gpu(st), out))
}

/// scatter_add of `g` (index shape `ish`) into zeros of `shape` at host or device indices.
pub(crate) fn gpu_scatter_add(g: &Tensor, shape: &[usize], dim: usize, idx: Indices<'_>, ish: &[usize]) -> Option<Storage> {
    let gs = gpu_of(&g.storage)?;
    Some(Storage::Gpu(ok((|| {
        let v = f32_contiguous(gs, &g.layout)?;
        let (outer, n, inner) = lanes(shape, dim);
        let i = match idx {
            Indices::Host(h) => indices_on(gs.device, h)?,
            Indices::Device(d) => d.clone(),
        };
        be(gs).scatter_add(&i, &v, [outer, n, inner, ish[dim]])
    })())))
}

/// index_add of `g`'s slices into zeros of `shape` (select backward).
pub(crate) fn gpu_index_add(g: &Tensor, shape: &[usize], dim: usize, idx: &[i64]) -> Option<Storage> {
    let gs = gpu_of(&g.storage)?;
    Some(Storage::Gpu(ok((|| {
        let v = f32_contiguous(gs, &g.layout)?;
        let (outer, n, inner) = lanes(shape, dim);
        let i = indices_on(gs.device, idx)?;
        be(gs).index_add(&i, &v, [outer, n, inner, idx.len()])
    })())))
}

/// Indices of a scatter: host values or I32 device storage.
pub(crate) enum Indices<'a> {
    Host(&'a [i64]),
    Device(&'a GpuStorage),
}

/// One-hot f32 rows `[.., n]` of host ids, on `device`.
pub(crate) fn gpu_one_hot(device: Device, ids: &IntTensor, n: usize) -> Result<Storage> {
    let data = ids.data();
    let i = indices_on(device, &data)?;
    Ok(Storage::Gpu(backend_for(device)?.one_hot(&i, data.len(), n)?))
}

/// A contiguous copy of a view (any dtype).
pub(crate) fn gpu_copy(t: &Tensor) -> Option<Storage> {
    let g = gpu_of(&t.storage)?;
    Some(Storage::Gpu(ok(be(g).copy(g, &t.layout))))
}

/// The sort by a host round trip (counted): read back, CpuRef's sort, upload values and indices.
pub(crate) fn sort_desc_on_host<B: GpuBackend + ?Sized>(b: &B, x: &GpuStorage, lanes: [usize; 3]) -> Result<(GpuStorage, GpuStorage)> {
    count_round_trip();
    let CpuStorage::F32(v) = b.download(x)? else { return Err(TensorError::DType("sort of non-f32 GPU storage".into())) };
    let (vals, idx) = super::kernels::sort_desc_with_indices(&v, &lanes, 1);
    Ok((b.upload(x.device, &CpuStorage::F32(vals))?, b.upload(x.device, &CpuStorage::I32(idx.into_iter().map(|j| j as i32).collect()))?))
}

/// Descending sort along `dim` on the device (rank ≥ 2, as CpuRef's): values and I32 indices.
pub(crate) fn gpu_sort_desc(t: &Tensor, dim: usize) -> Option<(Storage, GpuStorage)> {
    let g = gpu_of(&t.storage)?;
    if t.layout.shape.len() < 2 {
        return None;
    }
    let (v, i) = ok((|| {
        let x = f32_contiguous(g, &t.layout)?;
        let (outer, n, inner) = lanes(&t.layout.shape, dim);
        be(g).sort_desc(&x, [outer, n, inner])
    })());
    Some((Storage::Gpu(v), i))
}

/// `sort_descending_with_indices` of a GPU tensor: the values (tracked like the CPU op: their
/// gradient scatter-adds back by the device indices) and the indices, which stay on the device
/// as I32 until read.
pub(crate) fn gpu_sort_desc_tracked(t: &Tensor, dim: usize) -> Option<(Tensor, IntTensor)> {
    let (st, idx) = gpu_sort_desc(t, dim)?;
    let shape = t.layout.shape.clone();
    let indices = IntTensor { storage: std::sync::Arc::new(Storage::Gpu(idx.clone())), layout: Layout::contiguous(shape.clone()) };
    let out = super::autodiff::record_storage(st, shape.clone(), &[t], move || Box::new(move |g: Tensor, _: &[bool]| vec![Some(g.scatter_add_into_device(&shape, dim, &idx, &shape))]));
    Some((out, indices))
}

/// A constant f32 tensor on a GPU device, written there (no upload where the backend fills).
pub(crate) fn gpu_full(device: Device, shape: Vec<usize>, value: f64) -> Option<Tensor> {
    let b = backend_for(device).ok()?;
    let st = ok(b.fill(device, shape.iter().product(), value as f32));
    Some(Tensor::fresh(Storage::Gpu(st), shape, device))
}

/// A bool tensor on a GPU as f32 1.0 / 0.0 on the same device (a fresh tensor).
pub(crate) fn gpu_bool_float(mask: &BoolTensor) -> Option<Tensor> {
    let g = gpu_of(&mask.storage)?;
    let st = match be(g).cast(g, &mask.layout, DType::F32) {
        Ok(st) => st,
        // A backend without the Bool → F32 cast: convert on the host and upload
        // to the mask's device, a counted round trip.
        Err(TensorError::Unsupported(_)) => {
            count_round_trip();
            let v: Vec<f32> = mask.data().iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
            ok(upload(g.device, CpuStorage::F32(v)))
        }
        Err(e) => ok(Err(e)),
    };
    Some(Tensor::fresh(Storage::Gpu(st), mask.layout.shape.clone(), g.device))
}

/// Cast to f32, f16 or bf16 (f64 refused on Metal).
pub(crate) fn gpu_cast(t: &Tensor, to: DType) -> Option<Storage> {
    let g = gpu_of(&t.storage)?;
    Some(Storage::Gpu(ok((|| {
        if to == DType::F64 || g.dtype == DType::F64 {
            return Err(TensorError::Unsupported(format!("f64 on {:?}: the GPU kernels are f32 (Metal has no f64)", g.device)));
        }
        let (x, xl) = f32_operand(g, &t.layout)?;
        match to {
            DType::F32 if x.dtype == DType::F32 && xl.is_contiguous() && xl.offset == 0 && x.len == xl.numel() => Ok(x.into_owned()),
            DType::F32 => be(g).copy(&x, &xl),
            DType::F16 | DType::BF16 => be(g).cast(&x, &xl, to),
            other => Err(TensorError::DType(format!("cast of a float tensor to {}", other.name()))),
        }
    })())))
}

/// slice_assign: a copy of `t` with `value` written into `ranges`.
pub(crate) fn gpu_slice_assign(t: &Tensor, ranges: &[std::ops::Range<usize>], value: &Tensor) -> Option<Storage> {
    let g = gpu_of(&t.storage)?;
    let gv = ok(gpu_of(&value.storage).ok_or_else(|| TensorError::Unsupported("slice_assign of a CPU value into a GPU tensor".into())));
    Some(Storage::Gpu(ok((|| {
        let out = f32_contiguous(g, &t.layout)?;
        // The output must be a fresh buffer: `write` changes it in place.
        let out = if gpu_same_buffer(&out, g) { be(g).copy(g, &t.layout)? } else { out };
        let (v, vl) = f32_operand(gv, &value.layout)?;
        let dl = Layout::contiguous(t.layout.shape.clone()).narrowed(ranges);
        be(g).write(&out, &dl, &v, &vl)?;
        Ok(out)
    })())))
}

/// cat along `dim` (every part on the same GPU, one float dtype).
pub(crate) fn gpu_cat(parts: &[Tensor], dim: usize) -> Option<(Storage, Vec<usize>)> {
    let first = gpu_of(&parts[0].storage)?;
    let mut out_sh = parts[0].layout.shape.clone();
    out_sh[dim] = parts.iter().map(|p| p.layout.shape[dim]).sum();
    let st = ok((|| {
        let out = be(first).alloc(first.device, DType::F32, out_sh.iter().product())?;
        let base = Layout::contiguous(out_sh.clone());
        let mut at = 0;
        for p in parts {
            let gp = gpu_of(&p.storage).ok_or_else(|| TensorError::Unsupported("cat of CPU and GPU tensors".into()))?;
            let (v, vl) = f32_operand(gp, &p.layout)?;
            let mut r: Vec<std::ops::Range<usize>> = out_sh.iter().map(|&d| 0..d).collect();
            r[dim] = at..at + p.layout.shape[dim];
            at += p.layout.shape[dim];
            be(first).write(&out, &base.narrowed(&r), &v, &vl)?;
        }
        Ok(out)
    })());
    Some((Storage::Gpu(st), out_sh))
}

fn gpu_same_buffer(a: &GpuStorage, b: &GpuStorage) -> bool {
    match (&a.buf, &b.buf) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        (GpuBuf::Metal(x), GpuBuf::Metal(y)) => std::sync::Arc::ptr_eq(x, y),
        #[cfg(feature = "cuda")]
        (GpuBuf::Cuda(x), GpuBuf::Cuda(y)) => std::sync::Arc::ptr_eq(x, y),
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

/// nn::Adam's plain update fused on the parameter's GPU: (p, m, v) as fresh tensors, or None
/// (a CPU tensor, a backend without the fused kernel, or operands that are not contiguous f32 of
/// one shape), when the caller runs the op sequence.
pub(crate) fn gpu_adam(p: &Tensor, g: &Tensor, m: &Tensor, v: &Tensor, s: &AdamScalars) -> Option<(Tensor, Tensor, Tensor)> {
    let gp = gpu_of(&p.storage)?;
    fn plain<'a>(t: &'a Tensor, like: &Tensor, device: Device) -> Option<&'a GpuStorage> {
        let x = gpu_of(&t.storage)?;
        (x.device == device && x.dtype == DType::F32 && t.layout.shape == like.layout.shape && t.layout.is_contiguous() && t.layout.offset == 0 && x.len == t.layout.numel()).then_some(x)
    }
    let plain = |t| plain(t, p, gp.device);
    let (pp, gg, mm, vv) = (plain(p)?, plain(g)?, plain(m)?, plain(v)?);
    match be(gp).adam_update(pp, gg, mm, vv, p.layout.numel(), s) {
        Ok((a, b, c)) => {
            let sh = p.layout.shape.clone();
            let f = |st: GpuStorage| Tensor::fresh(Storage::Gpu(st), sh.clone(), gp.device);
            Some((f(a), f(b), f(c)))
        }
        Err(TensorError::Unsupported(_)) => None,
        Err(e) => ok(Err(e)),
    }
}

/// The provenance key of a GPU device (its name and the kernel options), e.g. for run records.
pub fn gpu_key(device: Device) -> Result<String> {
    backend_for(device)?.key()
}

/// Wait for every kernel encoded on `device`.
pub fn synchronize(device: Device) -> Result<()> {
    backend_for(device)?.sync()
}
