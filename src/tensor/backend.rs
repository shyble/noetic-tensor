//! Backends: the kernel set a tensor's device runs. `CpuRef` is
//! the reference (the reference kernels: burn 0.21's arithmetic at first, the standard formulations now); `CpuFast` runs the same operations
//! with worker threads and, for reductions along a lane, 4-wide SIMD partial sums (std::arch on
//! aarch64, a portable 4-accumulator loop elsewhere). Elementwise maps, matmul (row blocks of
//! the same gemm) and reductions across lanes keep Reference's per-element order, so only the
//! lane reductions (`sum_dim` on the last axis, `sum`) differ, in the last bits.

use super::dtype::FloatElem;
use super::kernels as k;

/// The operations a backend implements (the hot ones; every other op is shared).
pub(crate) trait BackendStorage {
    fn map<T: Copy + Sync, U: Send, F: Fn(T) -> U + Sync>(x: &[T], f: F) -> Vec<U>;
    fn zip<T: Copy + Sync, U: Copy + Sync, V: Send, F: Fn(T, U) -> V + Sync>(a: &[T], ash: &[usize], b: &[U], bsh: &[usize], f: F) -> (Vec<V>, Vec<usize>);
    fn matmul<T: FloatElem>(a: &[T], ash: &[usize], b: &[T], bsh: &[usize]) -> (Vec<T>, Vec<usize>);
    fn sum_dim<T: FloatElem>(x: &[T], sh: &[usize], dim: usize) -> (Vec<T>, Vec<usize>);
    fn sum_all<T: FloatElem>(x: &[T]) -> T;
}

pub(crate) struct CpuRef;
pub(crate) struct CpuFast;

/// Run `$body` with `$B` the backend of `$device`.
macro_rules! with_backend {
    ($device:expr, $B:ident => $body:expr) => {
        match $device {
            $crate::tensor::device::Device::Cpu($crate::tensor::device::CpuMode::Fast) => {
                type $B = $crate::tensor::backend::CpuFast;
                $body
            }
            _ => {
                type $B = $crate::tensor::backend::CpuRef;
                $body
            }
        }
    };
}
pub(crate) use with_backend;

impl BackendStorage for CpuRef {
    fn map<T: Copy + Sync, U: Send, F: Fn(T) -> U + Sync>(x: &[T], f: F) -> Vec<U> {
        k::map(x, f)
    }
    fn zip<T: Copy + Sync, U: Copy + Sync, V: Send, F: Fn(T, U) -> V + Sync>(a: &[T], ash: &[usize], b: &[U], bsh: &[usize], f: F) -> (Vec<V>, Vec<usize>) {
        k::zip(a, ash, b, bsh, f)
    }
    fn matmul<T: FloatElem>(a: &[T], ash: &[usize], b: &[T], bsh: &[usize]) -> (Vec<T>, Vec<usize>) {
        k::matmul(a, ash, b, bsh)
    }
    fn sum_dim<T: FloatElem>(x: &[T], sh: &[usize], dim: usize) -> (Vec<T>, Vec<usize>) {
        k::sum_dim(x, sh, dim)
    }
    fn sum_all<T: FloatElem>(x: &[T]) -> T {
        k::sum_all(x)
    }
}

/// Worker threads for the fast backend (`NOETIC_THREADS`, else the machine's parallelism).
pub fn threads() -> usize {
    std::env::var("NOETIC_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)).max(1)
}

/// The thresholds: below them Fast runs the Reference kernels
/// themselves (bit-identical); measured on the M-series Mac at 4 threads under load.
/// Unary maps (exp, ln, sqrt: compute-bound) from 2^20 elements.
pub const FAST_MIN_MAP: usize = 1 << 20;
/// Binary elementwise ops (memory-bound) from 2^22 elements.
pub const FAST_MIN_ZIP: usize = 1 << 22;
/// Reductions (`sum_dim`, `sum`) from 2^22 elements.
pub const FAST_MIN_SUM: usize = 1 << 22;
/// Matmul from 2^27 multiply-adds (batches × m × k × n).
pub const FAST_MIN_MATMUL: usize = 1 << 27;

/// A vector of `n` elements, each written exactly once by `write(start, chunk)` on one of
/// `parts` threads (every chunk starts at `start` and is fully written).
fn fill<U: Send>(n: usize, parts: usize, write: impl Fn(usize, &mut [std::mem::MaybeUninit<U>]) + Sync) -> Vec<U> {
    let mut v: Vec<std::mem::MaybeUninit<U>> = Vec::with_capacity(n);
    // SAFETY: MaybeUninit needs no initialisation; every element is written below.
    unsafe { v.set_len(n) };
    let per = n.div_ceil(parts).max(1);
    std::thread::scope(|s| {
        for (i, c) in v.chunks_mut(per).enumerate() {
            let w = &write;
            s.spawn(move || w(i * per, c));
        }
    });
    let mut v = std::mem::ManuallyDrop::new(v);
    // SAFETY: all n elements are initialised; MaybeUninit<U> has U's layout.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut U, v.len(), v.capacity()) }
}

/// Sum of a lane with four SIMD lanes × four vector accumulators.
fn simd_sum<T: FloatElem>(xs: &[T]) -> T {
    #[cfg(target_arch = "aarch64")]
    {
        if T::DTYPE == super::DType::F32 {
            // SAFETY: T is f32 (checked by dtype); NEON is baseline on aarch64.
            let xs32: &[f32] = unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const f32, xs.len()) };
            let s = unsafe { neon_sum_f32(xs32) };
            return T::from_f32(s);
        }
    }
    let mut acc = [T::ZERO; 16];
    let mut chunks = xs.chunks_exact(16);
    for c in &mut chunks {
        for i in 0..16 {
            acc[i] = acc[i] + c[i];
        }
    }
    let mut total = T::ZERO;
    for a in acc {
        total = total + a;
    }
    for &x in chunks.remainder() {
        total = total + x;
    }
    total
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_sum_f32(xs: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut a0 = vdupq_n_f32(0.0);
    let mut a1 = vdupq_n_f32(0.0);
    let mut a2 = vdupq_n_f32(0.0);
    let mut a3 = vdupq_n_f32(0.0);
    let mut chunks = xs.chunks_exact(16);
    for c in &mut chunks {
        let p = c.as_ptr();
        a0 = vaddq_f32(a0, vld1q_f32(p));
        a1 = vaddq_f32(a1, vld1q_f32(p.add(4)));
        a2 = vaddq_f32(a2, vld1q_f32(p.add(8)));
        a3 = vaddq_f32(a3, vld1q_f32(p.add(12)));
    }
    let v = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    let mut total = vaddvq_f32(v);
    for &x in chunks.remainder() {
        total += x;
    }
    total
}

impl BackendStorage for CpuFast {
    fn map<T: Copy + Sync, U: Send, F: Fn(T) -> U + Sync>(x: &[T], f: F) -> Vec<U> {
        let parts = threads();
        if x.len() < FAST_MIN_MAP || parts == 1 {
            return k::map(x, f);
        }
        fill(x.len(), parts, |start, out| {
            for (o, &a) in out.iter_mut().zip(&x[start..]) {
                o.write(f(a));
            }
        })
    }

    fn zip<T: Copy + Sync, U: Copy + Sync, V: Send, F: Fn(T, U) -> V + Sync>(a: &[T], ash: &[usize], b: &[U], bsh: &[usize], f: F) -> (Vec<V>, Vec<usize>) {
        let parts = threads();
        if ash != bsh || a.len() < FAST_MIN_ZIP || parts == 1 {
            return k::zip(a, ash, b, bsh, f);
        }
        let v = fill(a.len(), parts, |start, out| {
            for ((o, &x), &y) in out.iter_mut().zip(&a[start..]).zip(&b[start..]) {
                o.write(f(x, y));
            }
        });
        (v, ash.to_vec())
    }

    /// Row blocks of the same gemm on separate threads: every output element is computed
    /// exactly as Reference computes it.
    fn matmul<T: FloatElem>(a: &[T], ash: &[usize], b: &[T], bsh: &[usize]) -> (Vec<T>, Vec<usize>) {
        let ((m, kk, n), out, plan) = k::matmul_plan(ash, bsh);
        let parts = threads();
        if plan.len() * m * kk * n < FAST_MIN_MATMUL || parts == 1 {
            return k::matmul(a, ash, b, bsh);
        }
        let mut c = vec![T::ZERO; plan.len() * m * n];
        // Tasks: (batch, first row, rows), sized so each thread gets a few.
        let rows_per = (plan.len() * m).div_ceil(parts * 2).clamp(1, m.max(1));
        std::thread::scope(|s| {
            for (t, cb) in c.chunks_mut(m * n).enumerate() {
                let (ao, bo) = plan[t];
                for (ri, cr) in cb.chunks_mut(rows_per * n).enumerate() {
                    let row0 = ri * rows_per;
                    let rows = cr.len() / n.max(1);
                    let (a, b) = (&a[ao + row0 * kk..], &b[bo..]);
                    s.spawn(move || unsafe {
                        // SAFETY: rows×k, k×n and rows×n row-major blocks inside the buffers.
                        T::gemm(rows, kk, n, a.as_ptr(), b.as_ptr(), cr.as_mut_ptr());
                    });
                }
            }
        });
        (c, out)
    }

    fn sum_dim<T: FloatElem>(x: &[T], sh: &[usize], dim: usize) -> (Vec<T>, Vec<usize>) {
        let parts = threads();
        if x.len() < FAST_MIN_SUM || parts == 1 {
            return k::sum_dim(x, sh, dim);
        }
        let mut out_sh = sh.to_vec();
        out_sh[dim] = 1;
        let n = sh[dim];
        if dim + 1 == sh.len() && n > 0 {
            // Lanes along the last axis: SIMD partial sums, lanes split across threads.
            let lanes = x.len() / n;
            let mut v = vec![T::ZERO; lanes];
            {
                let per = lanes.div_ceil(parts);
                std::thread::scope(|s| {
                    for (vo, xo) in v.chunks_mut(per).zip(x.chunks(per * n)) {
                        s.spawn(move || {
                            for (o, lane) in vo.iter_mut().zip(xo.chunks(n)) {
                                *o = simd_sum(lane);
                            }
                        });
                    }
                });
            }
            return (v, out_sh);
        }
        // Other axes: whole outer blocks per thread, each in Reference's order.
        let outer: usize = sh[..dim].iter().product();
        let inner: usize = sh[dim + 1..].iter().product();
        let mut v = vec![T::ZERO; outer * inner];
        let per = outer.div_ceil(parts).max(1);
        par_chunks_blocks(x, &mut v, per * n * inner, per * inner, |xo, vo| {
            let o_count = vo.len() / inner.max(1);
            let (part, _) = k::sum_dim(xo, &[o_count, n, inner], 1);
            vo.copy_from_slice(&part);
        });
        (v, out_sh)
    }

    fn sum_all<T: FloatElem>(x: &[T]) -> T {
        if x.len() < FAST_MIN_SUM { k::sum_all(x) } else { simd_sum(x) }
    }
}

/// Like `par_chunks`, with different chunk sizes for input and output.
fn par_chunks_blocks<T: Sync, U: Send>(x: &[T], out: &mut [U], xper: usize, oper: usize, f: impl Fn(&[T], &mut [U]) + Sync) {
    std::thread::scope(|s| {
        for (xi, oi) in x.chunks(xper.max(1)).zip(out.chunks_mut(oper.max(1))) {
            let f = &f;
            s.spawn(move || f(xi, oi));
        }
    });
}

