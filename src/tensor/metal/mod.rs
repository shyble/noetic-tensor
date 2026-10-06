//! The Metal backend: one system GPU, shared (unified-memory) buffers, kernels
//! compiled from MSL source with fast math off, and a command stream that batches encoded
//! kernels into one command buffer until a host read (`sync`).
//!
//! The device is created on first use only. A process pinned to the reference backend refuses
//! Metal before it gets here (`device::check_with`), and `context()` refuses again, so a pinned
//! process never initialises a GPU.

pub(crate) mod backend;
pub(crate) mod ffi;
pub(crate) mod shaders;

use ffi::{msg, Id, MTLSize, Pool};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// A shared Metal buffer. On drop it returns to the buffer pool when it has a pool size
/// class (`cap` > 0), else it is released; command buffers retain what they use.
pub(crate) struct Buffer {
    id: Id,
    bytes: usize,
    /// The buffer's size class in bytes (0: not pooled).
    cap: usize,
}

// SAFETY: MTLBuffer objects are thread-safe; host access goes through `Context::sync` first.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.cap > 0 && pool::give(self.id, self.cap) {
            return;
        }
        // SAFETY: a +1 reference from newBuffer… (or a retain).
        unsafe { ffi::release(self.id) }
    }
}

/// The buffer pool: freed buffers by size class, reused by later requests instead of
/// creating new MTLBuffers (≈ 5 µs each). A buffer freed while GPU work is still encoded or in
/// flight may be read by that work, so it is reused at once only as a kernel output (kernels
/// run in encoding order, so the earlier reader runs before the new writer); a host write
/// (upload) takes it only after the next `sync` has completed that work. Outputs are always
/// fully written by their kernel, so stale contents are never read.
pub(crate) mod pool {
    use super::ffi::{self, Id};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Mutex;

    struct Entry {
        id: Id,
        /// The sync count when freed, and whether GPU work was pending then.
        epoch: u64,
        pending: bool,
    }

    #[derive(Default)]
    struct State {
        free: HashMap<usize, Vec<Entry>>,
        bytes: usize,
    }

    // SAFETY: the ids are MTLBuffers (thread-safe objects) owned by the pool.
    unsafe impl Send for State {}

    static POOL: Mutex<Option<State>> = Mutex::new(None);
    /// Completed syncs, and whether any kernel is encoded or in flight.
    pub(super) static EPOCH: AtomicU64 = AtomicU64::new(0);
    pub(super) static PENDING: AtomicBool = AtomicBool::new(false);

    /// The most bytes the pool keeps (NOETIC_METAL_POOL_MB, default 1024 MB; 0 turns it off).
    fn limit() -> usize {
        static L: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *L.get_or_init(|| std::env::var("NOETIC_METAL_POOL_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024usize) << 20)
    }

    /// The size class of `bytes`: powers of two from 256 bytes to 1 MB, then whole MBs.
    pub(crate) fn class(bytes: usize) -> usize {
        let b = bytes.max(256);
        if b <= 1 << 20 { b.next_power_of_two() } else { b.div_ceil(1 << 20) << 20 }
    }

    /// Keep a freed buffer; false when the pool is full (the caller releases it).
    pub(super) fn give(id: Id, cap: usize) -> bool {
        let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let st = g.get_or_insert_with(State::default);
        if st.bytes + cap > limit() {
            return false;
        }
        let pending = PENDING.load(Ordering::SeqCst);
        let epoch = EPOCH.load(Ordering::SeqCst);
        st.free.entry(cap).or_default().push(Entry { id, epoch, pending });
        st.bytes += cap;
        true
    }

    /// A pooled buffer of class `cap`; `host_write` asks for one no pending work can read.
    pub(super) fn take(cap: usize, host_write: bool) -> Option<Id> {
        let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let st = g.as_mut()?;
        let list = st.free.get_mut(&cap)?;
        let now = EPOCH.load(Ordering::SeqCst);
        let pos = if host_write { list.iter().rposition(|e| !e.pending || e.epoch < now)? } else { list.len().checked_sub(1)? };
        let e = list.swap_remove(pos);
        st.bytes -= cap;
        Some(e.id)
    }

    /// Release every pooled buffer.
    #[allow(dead_code)]
    pub fn clear() {
        let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(st) = g.take() {
            for (_, l) in st.free {
                for e in l {
                    // SAFETY: the pool's own reference.
                    unsafe { ffi::release(e.id) };
                }
            }
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalBuffer({} bytes)", self.bytes)
    }
}

impl Buffer {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// The buffer's memory. Read or write it only after `sync` (no kernel in flight).
    pub(crate) fn contents(&self) -> *mut u8 {
        // SAFETY: a live MTLBuffer.
        unsafe { msg!(*mut c_void; self.id, "contents") as *mut u8 }
    }
}

/// An argument of a kernel.
pub(crate) enum Arg<'a> {
    Buf(&'a Buffer),
    /// Small constant data (`setBytes`, at most 4 KB).
    Bytes(&'a [u8]),
}

pub(crate) struct Context {
    device: Id,
    queue: Id,
    library: Id,
    pipelines: HashMap<&'static str, Id>,
    /// The open command buffer and its compute encoder (both retained), if any work is encoded.
    open: Option<(Id, Id)>,
    /// Kernels encoded into the open command buffer.
    open_count: usize,
    /// Command buffers committed but not yet waited for (retained), in commit order.
    inflight: Vec<Id>,
    name: String,
    /// Kernels encoded and command buffers committed since start (for reports).
    pub(crate) dispatches: u64,
    pub(crate) commits: u64,
    /// Where host and GPU time goes (profiling; always collected, cheap).
    pub(crate) prof: std::cell::RefCell<Profile>,
    /// Dispatches recorded for replay (`set_trace`), with their buffers retained.
    trace: Option<Vec<Traced>>,
}

/// One recorded dispatch: the kernel, its arguments (buffers retained, bytes copied), its grid.
struct Traced {
    kernel: &'static str,
    args: Vec<Result<Id, Vec<u8>>>,
    groups: MTLSize,
    threads: MTLSize,
}

impl Drop for Traced {
    fn drop(&mut self) {
        for a in self.args.iter().flatten() {
            // SAFETY: retained when recorded.
            unsafe { ffi::release(*a) };
        }
    }
}

/// Counters of the Metal context since the last `take_profile`.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    /// Kernels encoded, and host time spent encoding them.
    pub dispatches: u64,
    pub encode_ns: u64,
    /// Command buffers committed, and host time blocked waiting for them in `sync`.
    pub commits: u64,
    pub syncs: u64,
    pub wait_ns: u64,
    /// GPU execution time summed over command buffers (GPUEndTime − GPUStartTime).
    pub gpu_ns: u64,
    /// Buffers created (and host time), and buffers served from the pool.
    pub allocs: u64,
    pub alloc_ns: u64,
    pub pool_hits: u64,
    /// Host → GPU copies: count, bytes, host time.
    pub uploads: u64,
    pub upload_bytes: u64,
    pub upload_ns: u64,
    /// Per kernel: dispatches and (in per-kernel mode) GPU time.
    pub kernels: std::collections::BTreeMap<&'static str, (u64, u64)>,
}

/// Per-kernel GPU timing: commit and wait after every kernel (slow; profiling only).
static PER_KERNEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn per-kernel GPU timing on or off (profiling only: every kernel becomes its own command
/// buffer, so totals are inflated; the split between kernels is what it measures).
pub fn set_per_kernel_timing(on: bool) {
    PER_KERNEL.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// Record every dispatch from now on (for `replay_trace`), or stop and drop the record.
pub fn set_trace(on: bool) {
    if let Ok(mut c) = context() {
        c.trace = on.then(Vec::new);
    }
}

/// Steady-state GPU time of every recorded dispatch: each one runs `reps` times back to back
/// in one command buffer (warm clocks, no host gap), and its time is the command buffer's GPU
/// time divided by `reps`. Returns per kernel (dispatches, ns per step-equivalent) summed over
/// the record. The recorded kernels only write their own outputs, so replaying is idempotent.
pub fn replay_trace(reps: usize) -> std::collections::BTreeMap<&'static str, (u64, u64)> {
    let mut out: std::collections::BTreeMap<&'static str, (u64, u64)> = Default::default();
    let Ok(mut c) = context() else { return out };
    let Some(trace) = c.trace.take() else { return out };
    let _ = c.sync();
    for t in &trace {
        let gpu0 = c.prof.borrow().gpu_ns;
        for _ in 0..reps {
            // SAFETY: the recorded ids are retained live buffers.
            unsafe { c.dispatch_raw(t.kernel, &t.args, t.groups, t.threads) };
        }
        let _ = c.sync();
        let dt = c.prof.borrow().gpu_ns - gpu0;
        if std::env::var_os("NOETIC_METAL_TRACE").is_some() {
            let p = t.args.iter().rev().find_map(|a| a.as_ref().err()).map(|b| b.iter().take(28).copied().collect::<Vec<u8>>()).unwrap_or_default();
            let words: Vec<u32> = p.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            eprintln!("metal replay: {} {:.4} ms groups {:?} params {:?}", t.kernel, dt as f64 / reps as f64 / 1e6, (t.groups.width, t.groups.height, t.groups.depth), words);
        }
        let e = out.entry(t.kernel).or_default();
        e.0 += 1;
        e.1 += dt / reps as u64;
    }
    out
}

/// The counters since the last call, reset.
pub fn take_profile() -> Profile {
    match context() {
        Ok(c) => c.prof.take(),
        Err(_) => Profile::default(),
    }
}

// SAFETY: Metal devices, queues, libraries and pipeline states are thread-safe; the command
// buffer and encoder are only touched under the context's mutex.
unsafe impl Send for Context {}

static CTX: OnceLock<std::result::Result<Mutex<Context>, String>> = OnceLock::new();

/// Kernels encoded since the Metal device was created (0 before).
#[doc(hidden)]
pub fn dispatches() -> u64 {
    match CTX.get() {
        Some(Ok(m)) => m.lock().unwrap_or_else(|e| e.into_inner()).dispatches,
        _ => 0,
    }
}

#[cfg(test)]
/// Whether this process has created the Metal device.
pub(crate) fn initialized() -> bool {
    CTX.get().is_some()
}

/// Memory this process holds on the Metal device and the device's recommended working-set
/// limit, in bytes (`currentAllocatedSize`, `recommendedMaxWorkingSetSize`); None until this
/// process has created its context (it never creates one).
pub fn memory() -> Option<(u64, u64)> {
    match CTX.get() {
        Some(Ok(m)) => {
            let c = m.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: both are NSUInteger / uint64_t properties of MTLDevice, sent with that signature.
            unsafe { Some((msg!(usize; c.device, "currentAllocatedSize") as u64, msg!(u64; c.device, "recommendedMaxWorkingSetSize"))) }
        }
        _ => None,
    }
}

/// The Metal context (created on first use). Refused in a pinned process.
pub(crate) fn context() -> std::result::Result<MutexGuard<'static, Context>, String> {
    if crate::tensor::device::is_pinned() {
        return Err("this process is pinned to the reference backend; Metal cannot be initialised".into());
    }
    match CTX.get_or_init(|| Context::create().map(Mutex::new)) {
        Ok(m) => Ok(m.lock().unwrap_or_else(|e| e.into_inner())),
        Err(e) => Err(e.clone()),
    }
}

impl Context {
    fn create() -> std::result::Result<Context, String> {
        let _pool = Pool::new();
        // SAFETY: every message below is sent with its method's exact signature.
        unsafe {
            let device = ffi::MTLCreateSystemDefaultDevice();
            if device.is_null() {
                return Err("no Metal device on this machine".into());
            }
            let name = ffi::string_of(msg!(Id; device, "name"));
            let queue: Id = msg!(Id; device, "newCommandQueue");
            if queue.is_null() {
                return Err("Metal: newCommandQueue failed".into());
            }
            // Compile options: fast math off (safe math mode and precise functions where the OS
            // has them; fastMathEnabled = NO before macOS 15).
            let cls = ffi::class(c"MTLCompileOptions");
            let opts: Id = msg!(Id; msg!(Id; cls, "alloc"), "init");
            if ffi::responds(opts, ffi::sel!("setMathMode:")) {
                msg!((); opts, "setMathMode:", ffi::MTLMathModeSafe => isize);
                if ffi::responds(opts, ffi::sel!("setMathFloatingPointFunctions:")) {
                    msg!((); opts, "setMathFloatingPointFunctions:", ffi::MTLMathFloatingPointFunctionsPrecise => isize);
                }
            } else {
                msg!((); opts, "setFastMathEnabled:", false => bool);
            }
            let src = ffi::nsstring(shaders::SOURCE);
            let mut err: Id = std::ptr::null_mut();
            let library: Id = msg!(Id; device, "newLibraryWithSource:options:error:", src => Id, opts => Id, &mut err as *mut Id => *mut Id);
            ffi::release(src);
            ffi::release(opts);
            if library.is_null() {
                return Err(format!("Metal: the kernels did not compile: {}", ffi::error_text(err)));
            }
            Ok(Context { device, queue, library, pipelines: HashMap::new(), open: None, open_count: 0, inflight: Vec::new(), name, dispatches: 0, commits: 0, prof: Default::default(), trace: None })
        }
    }

    /// The GPU's name (for provenance keys).
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    fn pipeline(&mut self, kernel: &'static str) -> std::result::Result<Id, String> {
        if let Some(p) = self.pipelines.get(kernel) {
            return Ok(*p);
        }
        let _pool = Pool::new();
        // SAFETY: exact signatures; the name string and function are released after use.
        unsafe {
            let n = ffi::nsstring(kernel);
            let f: Id = msg!(Id; self.library, "newFunctionWithName:", n => Id);
            ffi::release(n);
            if f.is_null() {
                return Err(format!("Metal: no kernel named {kernel}"));
            }
            let mut err: Id = std::ptr::null_mut();
            let p: Id = msg!(Id; self.device, "newComputePipelineStateWithFunction:error:", f => Id, &mut err as *mut Id => *mut Id);
            ffi::release(f);
            if p.is_null() {
                return Err(format!("Metal: pipeline {kernel}: {}", ffi::error_text(err)));
            }
            self.pipelines.insert(kernel, p);
            Ok(p)
        }
    }

    /// A new zero-length-safe shared buffer of `bytes` bytes (contents undefined).
    pub(crate) fn alloc(&self, bytes: usize) -> std::result::Result<Buffer, String> {
        let t0 = std::time::Instant::now();
        let r = self.alloc_raw(bytes);
        let mut p = self.prof.borrow_mut();
        p.allocs += 1;
        p.alloc_ns += t0.elapsed().as_nanos() as u64;
        r
    }

    fn alloc_raw(&self, bytes: usize) -> std::result::Result<Buffer, String> {
        let cap = pool::class(bytes);
        if let Some(id) = pool::take(cap, false) {
            self.prof.borrow_mut().pool_hits += 1;
            return Ok(Buffer { id, bytes, cap });
        }
        self.new_buffer(bytes, cap)
    }

    /// A new MTLBuffer of `cap` bytes (at least 16) holding `bytes` logical bytes.
    fn new_buffer(&self, bytes: usize, cap: usize) -> std::result::Result<Buffer, String> {
        let len = cap.max(bytes).max(16);
        // SAFETY: exact signature; the result is +1.
        let id: Id = unsafe { msg!(Id; self.device, "newBufferWithLength:options:", len => usize, ffi::MTLResourceStorageModeShared => usize) };
        if id.is_null() {
            return Err(format!("Metal: could not allocate {bytes} bytes"));
        }
        Ok(Buffer { id, bytes, cap })
    }

    /// A shared buffer holding a copy of `data`.
    pub(crate) fn upload(&self, data: &[u8]) -> std::result::Result<Buffer, String> {
        let t0 = std::time::Instant::now();
        let r = self.upload_raw(data);
        let mut p = self.prof.borrow_mut();
        p.uploads += 1;
        p.upload_bytes += data.len() as u64;
        p.upload_ns += t0.elapsed().as_nanos() as u64;
        r
    }

    fn upload_raw(&self, data: &[u8]) -> std::result::Result<Buffer, String> {
        let cap = pool::class(data.len());
        let b = match pool::take(cap, true) {
            Some(id) => {
                self.prof.borrow_mut().pool_hits += 1;
                Buffer { id, bytes: data.len(), cap }
            }
            None => self.new_buffer(data.len(), cap)?,
        };
        // SAFETY: the buffer holds at least data.len() bytes, and no encoded or in-flight kernel
        // reads or writes it (a fresh buffer, or a pooled one freed before the last sync).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents(), data.len()) };
        Ok(b)
    }

    /// Encode one kernel: `groups` threadgroups of `threads` threads, arguments bound at
    /// indices 0, 1, … in order. Runs at the next `sync` (kernels run in encoding order: the
    /// encoder is serial).
    pub(crate) fn dispatch(&mut self, kernel: &'static str, args: &[Arg], groups: MTLSize, threads: MTLSize) -> std::result::Result<(), String> {
        if groups.width == 0 || groups.height == 0 || groups.depth == 0 {
            return Ok(());
        }
        let p = self.pipeline(kernel)?;
        if let Some(tr) = self.trace.as_mut() {
            let args = args
                .iter()
                .map(|a| match a {
                    // SAFETY: a live buffer, retained for the record.
                    Arg::Buf(b) => Ok(unsafe { ffi::retain(b.id) }),
                    Arg::Bytes(d) => Err(d.to_vec()),
                })
                .collect();
            tr.push(Traced { kernel, args, groups, threads });
        }
        let t0 = std::time::Instant::now();
        let _pool = Pool::new();
        // SAFETY: exact signatures; the command buffer and encoder are retained while open.
        unsafe {
            let enc = match self.open {
                Some((_, enc)) => enc,
                None => {
                    let cb = ffi::retain(msg!(Id; self.queue, "commandBuffer"));
                    let enc = ffi::retain(msg!(Id; cb, "computeCommandEncoder"));
                    if cb.is_null() || enc.is_null() {
                        return Err("Metal: could not open a command buffer".into());
                    }
                    self.open = Some((cb, enc));
                    enc
                }
            };
            msg!((); enc, "setComputePipelineState:", p => Id);
            for (i, a) in args.iter().enumerate() {
                match a {
                    Arg::Buf(b) => msg!((); enc, "setBuffer:offset:atIndex:", b.id => Id, 0usize => usize, i => usize),
                    Arg::Bytes(d) => msg!((); enc, "setBytes:length:atIndex:", d.as_ptr() as *const c_void => *const c_void, d.len() => usize, i => usize),
                }
            }
            msg!((); enc, "dispatchThreadgroups:threadsPerThreadgroup:", groups => MTLSize, threads => MTLSize);
        }
        self.dispatches += 1;
        self.open_count += 1;
        pool::PENDING.store(true, std::sync::atomic::Ordering::SeqCst);
        {
            let mut p = self.prof.borrow_mut();
            p.dispatches += 1;
            p.encode_ns += t0.elapsed().as_nanos() as u64;
            p.kernels.entry(kernel).or_default().0 += 1;
        }
        if PER_KERNEL.load(std::sync::atomic::Ordering::Relaxed) {
            let gpu0 = self.prof.borrow().gpu_ns;
            self.sync()?;
            let dt = self.prof.borrow().gpu_ns - gpu0;
            if std::env::var_os("NOETIC_METAL_TRACE").is_some() && dt > 1_000_000 {
                eprintln!("metal trace: {kernel} took {:.3} ms on the GPU (groups {:?}, threads {:?})", dt as f64 / 1e6, groups, threads);
            }
            self.prof.borrow_mut().kernels.entry(kernel).or_default().1 += dt;
        } else if self.open_count >= FLUSH.load(std::sync::atomic::Ordering::Relaxed) {
            self.commit_open();
        }
        Ok(())
    }

    /// Encode a recorded dispatch (profiling replay).
    ///
    /// # Safety
    /// The ids are live buffers.
    unsafe fn dispatch_raw(&mut self, kernel: &'static str, args: &[Result<Id, Vec<u8>>], groups: MTLSize, threads: MTLSize) {
        let bufs: Vec<Buffer> = args.iter().filter_map(|a| a.as_ref().ok()).map(|id| Buffer { id: ffi::retain(*id), bytes: 0, cap: 0 }).collect();
        let mut bi = bufs.iter();
        let a: Vec<Arg> = args.iter().map(|x| match x {
            Ok(_) => Arg::Buf(bi.next().unwrap()),
            Err(d) => Arg::Bytes(d),
        }).collect();
        let tr = self.trace.take();
        let _ = self.dispatch(kernel, &a, groups, threads);
        self.trace = tr;
    }

    /// Commit the open command buffer without waiting (the GPU starts on it while more work is
    /// encoded; the queue runs command buffers in commit order).
    fn commit_open(&mut self) {
        if let Some((cb, enc)) = self.open.take() {
            let _pool = Pool::new();
            // SAFETY: exact signatures; the encoder's retain is balanced here, the command
            // buffer's in `sync`.
            unsafe {
                msg!((); enc, "endEncoding");
                msg!((); cb, "commit");
                ffi::release(enc);
            }
            self.inflight.push(cb);
            self.open_count = 0;
            self.commits += 1;
            self.prof.borrow_mut().commits += 1;
        }
    }

    /// Run everything encoded and wait for it (before any host read of a buffer).
    pub(crate) fn sync(&mut self) -> std::result::Result<(), String> {
        self.commit_open();
        if self.inflight.is_empty() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let _pool = Pool::new();
        let mut err = None;
        let mut gpu = 0.0f64;
        for cb in self.inflight.drain(..) {
            // SAFETY: exact signatures; the release balances the retain in `dispatch`.
            unsafe {
                msg!((); cb, "waitUntilCompleted");
                gpu += msg!(f64; cb, "GPUEndTime") - msg!(f64; cb, "GPUStartTime");
                let status: isize = msg!(isize; cb, "status");
                if status != ffi::MTLCommandBufferStatusCompleted && err.is_none() {
                    err = Some(format!("Metal: a command buffer failed (status {status}): {}", ffi::error_text(msg!(Id; cb, "error"))));
                }
                ffi::release(cb);
            }
        }
        pool::EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        pool::PENDING.store(false, std::sync::atomic::Ordering::SeqCst);
        {
            let mut p = self.prof.borrow_mut();
            p.syncs += 1;
            p.wait_ns += t0.elapsed().as_nanos() as u64;
            p.gpu_ns += (gpu * 1e9) as u64;
        }
        err.map_or(Ok(()), Err)
    }
}

/// Kernels per command buffer before it is committed (without waiting).
#[doc(hidden)]
pub static FLUSH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(256);

/// Bytes of a plain-old-data value (kernel parameters).
pub(crate) fn bytes_of<T: Copy>(v: &T) -> &[u8] {
    // SAFETY: T is Copy plain data read as bytes for setBytes.
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Threadgroups spanning `n` threads in groups of `per`.
pub(crate) fn groups_1d(n: usize, per: usize) -> (MTLSize, MTLSize) {
    (MTLSize::new(n.div_ceil(per), 1, 1), MTLSize::new(per, 1, 1))
}

// ---------------------------------------------------------------- raw entry points

/// Matmul parameters (the kernel's `MatmulParams`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MatmulParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_rs: u32,
    pub a_cs: u32,
    pub b_rs: u32,
    pub b_cs: u32,
}

/// The tuned matmul variants: kernel name, BM, BN, TM, TN (threads (BN/TN)×(BM/TM)).
pub(crate) const MATMUL_VARIANTS: [(&str, usize, usize, usize, usize); 10] = [
    ("matmul_64x64_4x4", 64, 64, 4, 4),
    ("matmul_128x64_8x4", 128, 64, 8, 4),
    ("matmul_64x128_4x8", 64, 128, 4, 8),
    ("matmul_128x128_8x8", 128, 128, 8, 8),
    ("matmul_32x64_4x4", 32, 64, 4, 4),
    ("matmul_64x64_8x4", 64, 64, 8, 4),
    ("matmul_32x32_4x4", 32, 32, 4, 4),
    ("matmul_32x32_2x2", 32, 32, 2, 2),
    ("matmul_16x16_1x1", 16, 16, 1, 1),
    ("matmul_32x16_2x1", 32, 16, 2, 1),
];

/// A forced matmul variant (index into MATMUL_VARIANTS; out of range: the rule below). For
/// measurements only.
#[doc(hidden)]
pub static MATMUL_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// The variant for an m×k·k×n matmul over `batches` (measured on an Apple M-series GPU): 64×64 tiles
/// with 4×4 per thread, except when those give fewer than 32 threadgroups (two per GPU core)
/// over a long k, where 16×16 tiles of one output per thread finish the serial k chains about
/// twice as fast. Every variant gives the same bits.
pub(crate) fn choose_matmul(m: usize, n: usize, k: usize, batches: usize) -> (&'static str, usize, usize, usize, usize) {
    let o = MATMUL_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if o < MATMUL_VARIANTS.len() {
        return MATMUL_VARIANTS[o];
    }
    let g64 = m.div_ceil(64) * n.div_ceil(64) * batches;
    if g64 < 32 && k >= 128 { MATMUL_VARIANTS[8] } else { MATMUL_VARIANTS[0] }
}

/// Encode C = A·B with a named tuned variant (same bits as `encode_matmul`); `offs` holds the
/// per-batch (A, B) element offsets as u32 pairs (a buffer, or bytes up to 4 KB).
pub(crate) fn encode_matmul_variant(ctx: &mut Context, v: (&'static str, usize, usize, usize, usize), a: &Buffer, b: &Buffer, c: &Buffer, offs: Arg, batches: usize, p: MatmulParams) -> std::result::Result<(), String> {
    let (name, bm, bn, tm, tn) = v;
    let groups = MTLSize::new((p.n as usize).div_ceil(bn), (p.m as usize).div_ceil(bm), batches);
    ctx.dispatch(name, &[Arg::Buf(a), Arg::Buf(b), Arg::Buf(c), offs, Arg::Bytes(bytes_of(&p))], groups, MTLSize::new(bn / tn, bm / tm, 1))
}

#[cfg(test)]
/// Encode C = A·B over `offs` batches (element offsets of each batch's A and B blocks).
pub(crate) fn encode_matmul(ctx: &mut Context, a: &Buffer, b: &Buffer, c: &Buffer, offs: &Buffer, batches: usize, p: MatmulParams) -> std::result::Result<(), String> {
    let groups = MTLSize::new((p.n as usize).div_ceil(64), (p.m as usize).div_ceil(64), batches);
    ctx.dispatch("matmul_f32", &[Arg::Buf(a), Arg::Buf(b), Arg::Buf(c), Arg::Buf(offs), Arg::Bytes(bytes_of(&p))], groups, MTLSize::new(16, 16, 1))
}

#[cfg(test)]
/// The f32 values of a buffer (after `sync`).
pub(crate) fn read_f32(b: &Buffer, n: usize) -> Vec<f32> {
    assert!(n * 4 <= b.bytes.max(16));
    let mut v = vec![0f32; n];
    // SAFETY: n f32 inside the buffer; no kernel in flight (the caller synced).
    unsafe { std::ptr::copy_nonoverlapping(b.contents() as *const f32, v.as_mut_ptr(), n) };
    v
}

/// Bytes of a slice of plain numbers.
pub(crate) fn raw_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain numeric element types without padding, read as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Bytes of an f32 slice.
#[cfg(test)]
pub(crate) fn f32_bytes(v: &[f32]) -> &[u8] {
    raw_bytes(v)
}

#[cfg(test)]
/// c = a + b on the GPU (upload, kernel, download).
pub(crate) fn raw_add(a: &[f32], b: &[f32]) -> std::result::Result<Vec<f32>, String> {
    let mut ctx = context()?;
    let (ba, bb) = (ctx.upload(f32_bytes(a))?, ctx.upload(f32_bytes(b))?);
    let c = ctx.alloc(a.len() * 4)?;
    let n = a.len() as u32;
    let (g, t) = groups_1d(a.len(), 256);
    ctx.dispatch("add_f32", &[Arg::Buf(&ba), Arg::Buf(&bb), Arg::Buf(&c), Arg::Bytes(bytes_of(&n))], g, t)?;
    ctx.sync()?;
    Ok(read_f32(&c, a.len()))
}

#[cfg(test)]
/// C = A·B for row-major m×k and k×n (upload, kernel, download).
pub(crate) fn raw_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> std::result::Result<Vec<f32>, String> {
    let mut ctx = context()?;
    let (ba, bb) = (ctx.upload(f32_bytes(a))?, ctx.upload(f32_bytes(b))?);
    let c = ctx.alloc(m * n * 4)?;
    let offs = ctx.upload(&[0u8; 8])?;
    let p = MatmulParams { m: m as u32, n: n as u32, k: k as u32, a_rs: k as u32, a_cs: 1, b_rs: n as u32, b_cs: 1 };
    encode_matmul(&mut ctx, &ba, &bb, &c, &offs, 1, p)?;
    ctx.sync()?;
    Ok(read_f32(&c, m * n))
}
