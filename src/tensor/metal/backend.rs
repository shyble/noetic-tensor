//! `GpuBackend` on Metal: each op allocates its output, encodes one kernel
//! (or a few) into the open command buffer and returns; host reads synchronise.

use super::ffi::MTLSize;
use super::{bytes_of, context, groups_1d, Arg, Buffer, Context, MatmulParams};
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::error::{Result, TensorError};
use crate::tensor::gpu::{GpuBackend, GpuBuf, GpuStorage, BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::tensor::half::{BF16, F16};
use crate::tensor::layout::Layout;
use crate::tensor::storage::CpuStorage;
use std::sync::Arc;

pub(crate) struct MetalBackend;

pub(crate) static METAL: MetalBackend = MetalBackend;

fn dev(e: String) -> TensorError {
    TensorError::Device(e)
}

fn ctx() -> Result<std::sync::MutexGuard<'static, Context>> {
    context().map_err(dev)
}

fn buf(g: &GpuStorage) -> &Buffer {
    match &g.buf {
        GpuBuf::Metal(b) => b,
        #[allow(unreachable_patterns)]
        _ => unreachable!("a Metal op on another GPU's storage"),
    }
}

fn storage(device: Device, dtype: DType, len: usize, b: Buffer) -> GpuStorage {
    GpuStorage { device, dtype, len, buf: GpuBuf::Metal(Arc::new(b)) }
}

fn out(c: &Context, like: &GpuStorage, dtype: DType, len: usize) -> Result<GpuStorage> {
    Ok(storage(like.device, dtype, len, c.alloc(len * dtype.size_in_bytes()).map_err(dev)?))
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| TensorError::Unsupported(format!("Metal kernels index with 32 bits: {what} {v} is too large")))
}

const MAXR: usize = 8;

/// The kernels' `Strided` parameters.
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
    /// Bit 0/1/2: operand a/b/c is row-major over `shape` (fast path).
    contig: u32,
}

/// Parameters for an iteration over `shape` reading a (and b) and writing c through layouts
/// (None: contiguous from 0).
fn strided(shape: &[usize], a: Option<&Layout>, b: Option<&Layout>, c: Option<&Layout>) -> Result<Strided> {
    if shape.len() > MAXR {
        return Err(TensorError::Unsupported(format!("Metal kernels take rank ≤ {MAXR}, not {}", shape.len())));
    }
    let mut p = Strided { n: u32_of(shape.iter().product(), "elements")?, rank: shape.len() as u32, off_a: 0, off_b: 0, off_c: 0, shape: [1; MAXR], sa: [0; MAXR], sb: [0; MAXR], sc: [0; MAXR], contig: 0 };
    let contiguous = crate::tensor::shape::strides(shape);
    let fill = |dst: &mut [u32; MAXR], off: &mut u32, l: Option<&Layout>| -> Result<()> {
        let (st, o) = match l {
            Some(l) => (l.strides.as_slice(), l.offset),
            None => (contiguous.as_slice(), 0),
        };
        *off = u32_of(o, "offset")?;
        for (d, s) in dst.iter_mut().zip(st) {
            *d = u32_of(*s, "stride")?;
        }
        Ok(())
    };
    for (d, s) in p.shape.iter_mut().zip(shape) {
        *d = *s as u32;
    }
    fill(&mut p.sa, &mut p.off_a, a)?;
    fill(&mut p.sb, &mut p.off_b, b)?;
    fill(&mut p.sc, &mut p.off_c, c)?;
    // Row-major: every dimension longer than 1 has its row-major stride.
    let row_major = |l: Option<&Layout>| l.is_none_or(|l| shape.iter().zip(&l.strides).zip(&contiguous).all(|((&d, &s), &c)| d <= 1 || s == c));
    p.contig = row_major(a) as u32 | (row_major(b) as u32) << 1 | (row_major(c) as u32) << 2;
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

fn run1(c: &mut Context, kernel: &'static str, args: &[Arg], n: usize) -> Result<()> {
    let (g, t) = groups_1d(n, 256);
    c.dispatch(kernel, args, g, t).map_err(dev)
}

fn f32s(g: &GpuStorage) -> Result<()> {
    if g.dtype == DType::F32 { Ok(()) } else { Err(TensorError::DType(format!("a Metal float kernel on {} storage", g.dtype.name()))) }
}

fn copy_kernel(dtype: DType) -> &'static str {
    match dtype.size_in_bytes() {
        1 => "copy_strided_8",
        2 => "copy_strided_16",
        4 => "copy_strided_32",
        _ => "copy_strided_64",
    }
}

impl GpuBackend for MetalBackend {
    fn upload(&self, device: Device, s: &CpuStorage) -> Result<GpuStorage> {
        use super::raw_bytes as raw;
        let c = ctx()?;
        let (bytes, dtype): (&[u8], DType) = match s {
            CpuStorage::F32(v) => (raw(v), DType::F32),
            CpuStorage::F16(v) => (raw(v), DType::F16),
            CpuStorage::BF16(v) => (raw(v), DType::BF16),
            CpuStorage::I64(v) => (raw(v), DType::I64),
            CpuStorage::I32(v) => (raw(v), DType::I32),
            CpuStorage::U8(v) => (raw(v), DType::U8),
            CpuStorage::Bool(v) => (raw(v), DType::Bool),
            CpuStorage::F64(_) => return Err(TensorError::Unsupported("f64 tensors cannot live on Metal (Metal has no f64); cast to f32 first".into())),
        };
        Ok(storage(device, dtype, s.len(), c.upload(bytes).map_err(dev)?))
    }

    fn download(&self, g: &GpuStorage) -> Result<CpuStorage> {
        let mut c = ctx()?;
        c.sync().map_err(dev)?;
        let b = buf(g);
        let n = g.len;
        fn read<T: Copy + Default>(b: &Buffer, n: usize) -> Vec<T> {
            let mut v = vec![T::default(); n];
            assert!(std::mem::size_of_val(v.as_slice()) <= b.bytes().max(16));
            // SAFETY: n elements inside the buffer; synced, so no kernel writes it.
            unsafe { std::ptr::copy_nonoverlapping(b.contents() as *const T, v.as_mut_ptr(), n) };
            v
        }
        Ok(match g.dtype {
            DType::F32 => CpuStorage::F32(read(b, n)),
            DType::F16 => CpuStorage::F16(read::<u16>(b, n).into_iter().map(F16).collect()),
            DType::BF16 => CpuStorage::BF16(read::<u16>(b, n).into_iter().map(BF16).collect()),
            DType::I64 => CpuStorage::I64(read(b, n)),
            DType::I32 => CpuStorage::I32(read(b, n)),
            DType::U8 => CpuStorage::U8(read(b, n)),
            // The kernels write 0 or 1 only, so every byte is a valid bool.
            DType::Bool => CpuStorage::Bool(read::<u8>(b, n).into_iter().map(|x| x != 0).collect()),
            DType::F64 => unreachable!("no f64 on Metal"),
        })
    }

    fn copy(&self, x: &GpuStorage, xl: &Layout) -> Result<GpuStorage> {
        let mut c = ctx()?;
        let n = xl.numel();
        let y = out(&c, x, x.dtype, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        run1(&mut c, copy_kernel(x.dtype), &[Arg::Buf(buf(x)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p))], n)?;
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
            (a, b) => return Err(TensorError::Unsupported(format!("Metal cast {} → {}", a.name(), b.name()))),
        };
        let mut c = ctx()?;
        let n = xl.numel();
        let y = out(&c, x, to, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        run1(&mut c, kernel, &[Arg::Buf(buf(x)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn unary(&self, op: UnaryOp, x: &GpuStorage, xl: &Layout) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = ctx()?;
        let n = xl.numel();
        let y = out(&c, x, DType::F32, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        let (code, s) = op.code();
        let u = ScalarOp { op: code, s };
        run1(&mut c, "unary_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&u))], n)?;
        Ok(y)
    }

    fn compare(&self, op: CmpOp, x: &GpuStorage, xl: &Layout, s: f32) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = ctx()?;
        let n = xl.numel();
        let y = out(&c, x, DType::Bool, n)?;
        let p = strided(&xl.shape, Some(xl), None, None)?;
        let code = match op {
            CmpOp::Gt => 0,
            CmpOp::Ge => 1,
            CmpOp::Lt => 2,
            CmpOp::Le => 3,
            CmpOp::Eq => 4,
        };
        let u = ScalarOp { op: code, s };
        run1(&mut c, "compare_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&u))], n)?;
        Ok(y)
    }

    fn binary(&self, op: BinaryOp, a: &GpuStorage, al: &Layout, b: &GpuStorage, bl: &Layout) -> Result<GpuStorage> {
        f32s(a)?;
        f32s(b)?;
        let mut c = ctx()?;
        let n = al.numel();
        let y = out(&c, a, DType::F32, n)?;
        let p = strided(&al.shape, Some(al), Some(bl), None)?;
        let code: u32 = match op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
        };
        run1(&mut c, "binary_f32", &[Arg::Buf(buf(a)), Arg::Buf(buf(b)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&code))], n)?;
        Ok(y)
    }

    fn mask_fill(&self, x: &GpuStorage, xl: &Layout, mask: &GpuStorage, ml: &Layout, value: f32) -> Result<GpuStorage> {
        f32s(x)?;
        if mask.dtype != DType::Bool {
            return Err(TensorError::DType(format!("a mask of {} storage", mask.dtype.name())));
        }
        let mut c = ctx()?;
        let n = xl.numel();
        let y = out(&c, x, DType::F32, n)?;
        let p = strided(&xl.shape, Some(xl), Some(ml), None)?;
        run1(&mut c, "mask_fill_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(mask)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p)), Arg::Bytes(bytes_of(&value))], n)?;
        Ok(y)
    }

    fn reduce_dim(&self, op: ReduceOp, x: &GpuStorage, shape: &[usize], dim: usize) -> Result<GpuStorage> {
        f32s(x)?;
        let outer: usize = shape[..dim].iter().product();
        let n = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let mut c = ctx()?;
        let l = lanes([outer, n, inner, 0])?;
        match op {
            ReduceOp::Sum => {
                let y = out(&c, x, DType::F32, outer * inner)?;
                // CpuRef's rule: eight partial sums along the last axis only (by position, not
                // by the inner size), in-order slice additions along any other.
                if dim + 1 == shape.len() {
                    run1(&mut c, "sum_last_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&l))], outer)?;
                } else {
                    run1(&mut c, "sum_mid_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&l))], outer * inner)?;
                }
                Ok(y)
            }
            ReduceOp::Max | ReduceOp::ArgMax => {
                if n == 0 {
                    return Err(TensorError::Shape("max or argmax along an empty dimension".into()));
                }
                let idx = out(&c, x, DType::I32, outer * inner)?;
                let val = out(&c, x, DType::F32, outer * inner)?;
                run1(&mut c, "argmax_dim_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(&idx)), Arg::Buf(buf(&val)), Arg::Bytes(bytes_of(&l))], outer * inner)?;
                Ok(if op == ReduceOp::Max { val } else { idx })
            }
        }
    }

    fn reduce_all(&self, op: ReduceOp, x: &GpuStorage, n: usize) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = ctx()?;
        let y = out(&c, x, DType::F32, 1)?;
        let nn = u32_of(n, "elements")?;
        let kernel = match op {
            ReduceOp::Sum => "sum_all_f32",
            ReduceOp::Max if n > 0 => "max_all_f32",
            ReduceOp::Max => return Err(TensorError::Shape("max of an empty tensor".into())),
            ReduceOp::ArgMax => return Err(TensorError::Unsupported("argmax over all elements".into())),
        };
        run1(&mut c, kernel, &[Arg::Buf(buf(x)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&nn))], 1)?;
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
        let mut c = ctx()?;
        let y = out(&c, a, DType::F32, nb * m * n)?;
        let p = MatmulParams { m: u32_of(m, "m")?, n: u32_of(n, "n")?, k: u32_of(k, "k")?, a_rs: u32_of(al.strides[r - 2], "stride")?, a_cs: u32_of(al.strides[r - 1], "stride")?, b_rs: u32_of(bl.strides[r - 2], "stride")?, b_cs: u32_of(bl.strides[r - 1], "stride")? };
        if nb > 0 && m > 0 && n > 0 {
            // The tuned kernel (the same bits); batch offsets as kernel bytes when they fit
            // (no buffer, no upload), else in a buffer.
            let bytes = super::raw_bytes(&offs);
            let v = super::choose_matmul(m, n, k, nb);
            if bytes.len() <= 4096 {
                super::encode_matmul_variant(&mut c, v, buf(a), buf(b), buf(&y), super::Arg::Bytes(bytes), nb, p).map_err(dev)?;
            } else {
                let ob = c.upload(bytes).map_err(dev)?;
                super::encode_matmul_variant(&mut c, v, buf(a), buf(b), buf(&y), super::Arg::Buf(&ob), nb, p).map_err(dev)?;
            }
        }
        Ok((y, shape))
    }

    fn gather(&self, x: &GpuStorage, l: [usize; 4], idx: &GpuStorage) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = ctx()?;
        let n = l[0] * l[3] * l[2];
        let y = out(&c, x, DType::F32, n)?;
        let p = lanes(l)?;
        run1(&mut c, "gather_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(idx)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn index_select(&self, x: &GpuStorage, l: [usize; 4], idx: &GpuStorage) -> Result<GpuStorage> {
        f32s(x)?;
        let mut c = ctx()?;
        let n = l[0] * l[3] * l[2];
        let y = out(&c, x, DType::F32, n)?;
        let p = lanes(l)?;
        run1(&mut c, "index_select_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(idx)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn one_hot(&self, idx: &GpuStorage, count: usize, n: usize) -> Result<GpuStorage> {
        let mut c = ctx()?;
        let y = out(&c, idx, DType::F32, count * n)?;
        let p = lanes([count, n, 1, 0])?;
        run1(&mut c, "one_hot_f32", &[Arg::Buf(buf(idx)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p))], count * n)?;
        Ok(y)
    }

    fn scatter_add(&self, idx: &GpuStorage, vals: &GpuStorage, l: [usize; 4]) -> Result<GpuStorage> {
        f32s(vals)?;
        let mut c = ctx()?;
        let n = l[0] * l[1] * l[2];
        let y = out(&c, vals, DType::F32, n)?;
        let p = lanes(l)?;
        run1(&mut c, "scatter_add_f32", &[Arg::Buf(buf(idx)), Arg::Buf(buf(vals)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn index_add(&self, idx: &GpuStorage, vals: &GpuStorage, l: [usize; 4]) -> Result<GpuStorage> {
        f32s(vals)?;
        let mut c = ctx()?;
        let n = l[0] * l[1] * l[2];
        let y = out(&c, vals, DType::F32, n)?;
        let p = lanes(l)?;
        run1(&mut c, "index_add_f32", &[Arg::Buf(buf(idx)), Arg::Buf(buf(vals)), Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&p))], n)?;
        Ok(y)
    }

    fn write(&self, dst: &GpuStorage, dl: &Layout, src: &GpuStorage, sl: &Layout) -> Result<()> {
        if dst.dtype != src.dtype || dl.shape != sl.shape {
            return Err(TensorError::Shape(format!("write of {} {:?} into {} {:?}", src.dtype.name(), sl.shape, dst.dtype.name(), dl.shape)));
        }
        let mut c = ctx()?;
        let p = strided(&sl.shape, Some(sl), None, Some(dl))?;
        run1(&mut c, copy_kernel(src.dtype), &[Arg::Buf(buf(src)), Arg::Buf(buf(dst)), Arg::Bytes(bytes_of(&p))], sl.numel())
    }

    fn alloc(&self, device: Device, dtype: DType, len: usize) -> Result<GpuStorage> {
        let c = ctx()?;
        Ok(storage(device, dtype, len, c.alloc(len * dtype.size_in_bytes()).map_err(dev)?))
    }

    fn adam_update(&self, p: &GpuStorage, g: &GpuStorage, m: &GpuStorage, v: &GpuStorage, n: usize, s: &crate::tensor::gpu::AdamScalars) -> Result<(GpuStorage, GpuStorage, GpuStorage)> {
        for x in [p, g, m, v] {
            f32s(x)?;
        }
        let mut c = ctx()?;
        let (po, mo, vo) = (out(&c, p, DType::F32, n)?, out(&c, p, DType::F32, n)?, out(&c, p, DType::F32, n)?);
        let nn = u32_of(n, "elements")?;
        run1(&mut c, "adam_f32", &[Arg::Buf(buf(p)), Arg::Buf(buf(g)), Arg::Buf(buf(m)), Arg::Buf(buf(v)), Arg::Buf(buf(&po)), Arg::Buf(buf(&mo)), Arg::Buf(buf(&vo)), Arg::Bytes(bytes_of(s)), Arg::Bytes(bytes_of(&nn))], n)?;
        Ok((po, mo, vo))
    }

    fn sort_desc(&self, x: &GpuStorage, l: [usize; 3]) -> Result<(GpuStorage, GpuStorage)> {
        f32s(x)?;
        let [outer, n, inner] = l;
        let np2 = n.max(1).next_power_of_two();
        if np2 > 4096 {
            // Lanes longer than the threadgroup memory holds: the counted host round trip.
            return crate::tensor::gpu::sort_desc_on_host(self, x, l);
        }
        let mut c = ctx()?;
        let vals = out(&c, x, DType::F32, outer * n * inner)?;
        let idx = out(&c, x, DType::I32, outer * n * inner)?;
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct SortP {
            outer: u32,
            n: u32,
            inner: u32,
            np2: u32,
        }
        let p = SortP { outer: u32_of(outer, "outer")?, n: u32_of(n, "n")?, inner: u32_of(inner, "inner")?, np2: np2 as u32 };
        let threads = (np2 / 2).clamp(1, 1024);
        c.dispatch("sort_desc_f32", &[Arg::Buf(buf(x)), Arg::Buf(buf(&vals)), Arg::Buf(buf(&idx)), Arg::Bytes(bytes_of(&p))], MTLSize::new(outer * inner, 1, 1), MTLSize::new(threads, 1, 1)).map_err(dev)?;
        Ok((vals, idx))
    }

    fn fill(&self, device: Device, len: usize, value: f32) -> Result<GpuStorage> {
        let mut c = ctx()?;
        let y = storage(device, DType::F32, len, c.alloc(len * 4).map_err(dev)?);
        let n = u32_of(len, "elements")?;
        run1(&mut c, "fill_f32", &[Arg::Buf(buf(&y)), Arg::Bytes(bytes_of(&value)), Arg::Bytes(bytes_of(&n))], len)?;
        Ok(y)
    }

    fn sync(&self) -> Result<()> {
        ctx()?.sync().map_err(dev)
    }

    fn key(&self) -> Result<String> {
        let c = ctx()?;
        let src = crate::hash::sha256_hex(super::shaders::SOURCE.as_bytes());
        Ok(format!("metal-{}-mathmode-safe-kernels-{}", c.name().replace(' ', "_"), &src[..16]))
    }

    fn platform_details(&self) -> Result<String> {
        Ok(ctx()?.platform_details())
    }
}
