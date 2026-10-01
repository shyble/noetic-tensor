//! Storage-level kernels over contiguous row-major buffers: CpuRef's arithmetic. The order is the
//! one burn 0.21's NdArray backend used (ndarray 0.17 without BLAS, no SIMD feature):
//! elementwise maps are exact IEEE operations; reductions follow ndarray's `sum_axis`
//! (eight partial sums along the last axis, in-order slice additions along any other axis);
//! matmul is matrixmultiply's sgemm/dgemm. Argmax propagates NaN and the sort is
//! stable (burn's skipped NaN and used `sort_unstable_by`).

use super::dtype::FloatElem;
use super::shape::{broadcast, broadcast_strides, numel, strides};
use std::ops::Range;

/// Visit the source offsets of an iteration in row-major order over `out`, reading a buffer
/// laid out with `src_strides` (0 on broadcast dimensions).
fn for_each_offset(out: &[usize], src_strides: &[usize], mut f: impl FnMut(usize)) {
    let rank = out.len();
    if rank == 0 {
        f(0);
        return;
    }
    let n = numel(out);
    if n == 0 {
        return;
    }
    let last = out[rank - 1];
    let ls = src_strides[rank - 1];
    let mut idx = vec![0usize; rank];
    let mut base = 0usize;
    loop {
        for j in 0..last {
            f(base + j * ls);
        }
        // Advance the outer odometer (all dims but the last).
        let mut d = rank - 1;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            base += src_strides[d];
            if idx[d] < out[d] {
                break;
            }
            base -= src_strides[d] * idx[d];
            idx[d] = 0;
        }
    }
}

/// The elements a layout selects from `storage`, in row-major order of its shape (a view made
/// contiguous).
pub(crate) fn materialize<T: Copy>(storage: &[T], layout: &super::layout::Layout) -> Vec<T> {
    let mut v = Vec::with_capacity(numel(&layout.shape));
    let off = layout.offset;
    for_each_offset(&layout.shape, &layout.strides, |o| v.push(storage[off + o]));
    v
}

/// Read `x` (shape `sh`) broadcast to `out`, in row-major order.
pub(crate) fn broadcast_to<T: Copy>(x: &[T], sh: &[usize], out: &[usize]) -> Vec<T> {
    if sh == out {
        return x.to_vec();
    }
    let st = broadcast_strides(sh, out);
    let mut v = Vec::with_capacity(numel(out));
    for_each_offset(out, &st, |o| v.push(x[o]));
    v
}

pub(crate) fn map<T: Copy, U>(x: &[T], f: impl Fn(T) -> U) -> Vec<U> {
    x.iter().map(|&a| f(a)).collect()
}

/// As `for_each_offset`, reading two buffers with their own strides at once.
fn for_each_offset2(out: &[usize], sa: &[usize], sb: &[usize], mut f: impl FnMut(usize, usize)) {
    let rank = out.len();
    if numel(out) == 0 {
        return;
    }
    let last = out[rank - 1];
    let (la, lb) = (sa[rank - 1], sb[rank - 1]);
    let mut idx = vec![0usize; rank];
    let (mut ba, mut bb) = (0usize, 0usize);
    loop {
        for j in 0..last {
            f(ba + j * la, bb + j * lb);
        }
        let mut d = rank - 1;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            ba += sa[d];
            bb += sb[d];
            if idx[d] < out[d] {
                break;
            }
            ba -= sa[d] * idx[d];
            bb -= sb[d] * idx[d];
            idx[d] = 0;
        }
    }
}

/// Elementwise binary op with burn's broadcasting (same rank, size-1 dimensions broadcast).
pub(crate) fn zip<T: Copy, U: Copy, V>(a: &[T], ash: &[usize], b: &[U], bsh: &[usize], f: impl Fn(T, U) -> V) -> (Vec<V>, Vec<usize>) {
    if ash == bsh {
        return (a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect(), ash.to_vec());
    }
    let out = broadcast(ash, bsh);
    let sa = broadcast_strides(ash, &out);
    let sb = broadcast_strides(bsh, &out);
    let mut v = Vec::with_capacity(numel(&out));
    for_each_offset2(&out, &sa, &sb, |oa, ob| v.push(f(a[oa], b[ob])));
    (v, out)
}

/// ndarray's `numeric_util::unrolled_fold` with `+` from zero: eight partial sums.
#[inline]
pub(crate) fn unrolled_sum<T: FloatElem>(mut xs: &[T]) -> T {
    let z = T::ZERO;
    let mut acc = z;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) = (z, z, z, z, z, z, z, z);
    while xs.len() >= 8 {
        p0 = p0 + xs[0];
        p1 = p1 + xs[1];
        p2 = p2 + xs[2];
        p3 = p3 + xs[3];
        p4 = p4 + xs[4];
        p5 = p5 + xs[5];
        p6 = p6 + xs[6];
        p7 = p7 + xs[7];
        xs = &xs[8..];
    }
    acc = acc + (p0 + p4);
    acc = acc + (p1 + p5);
    acc = acc + (p2 + p6);
    acc = acc + (p3 + p7);
    for (i, x) in xs.iter().enumerate() {
        if i >= 7 {
            break;
        }
        acc = acc + *x;
    }
    acc
}

/// `sum_axis` keeping the dimension (size 1), as ndarray computes it on a standard-layout array:
/// along the last axis each lane is folded with eight partial sums; along any other axis the
/// slices are added in order, starting from zero.
pub(crate) fn sum_dim<T: FloatElem>(x: &[T], sh: &[usize], dim: usize) -> (Vec<T>, Vec<usize>) {
    let mut out_sh = sh.to_vec();
    out_sh[dim] = 1;
    let n = sh[dim];
    let outer: usize = sh[..dim].iter().product();
    let inner: usize = sh[dim + 1..].iter().product();
    if dim + 1 == sh.len() {
        let v = (0..outer).map(|o| unrolled_sum(&x[o * n..(o + 1) * n])).collect();
        return (v, out_sh);
    }
    let mut v = vec![T::ZERO; outer * inner];
    for o in 0..outer {
        let acc = &mut v[o * inner..(o + 1) * inner];
        for k in 0..n {
            let row = &x[(o * n + k) * inner..(o * n + k + 1) * inner];
            for (a, &r) in acc.iter_mut().zip(row) {
                *a = *a + r;
            }
        }
    }
    (v, out_sh)
}

/// ndarray's `sum()` on a contiguous array.
pub(crate) fn sum_all<T: FloatElem>(x: &[T]) -> T {
    unrolled_sum(x)
}

/// `argmax` keeping the dimension: the first maximum, NaN-propagating (as PyTorch and
/// NumPy): the first NaN of a lane is its maximum; otherwise a strictly greater value replaces.
/// burn 0.21 skipped NaN (it never compares greater), so a lane with NaN reported a number.
#[allow(clippy::eq_op)] // `x != x` is the generic NaN test.
pub(crate) fn argmax_dim<T: Copy + PartialOrd>(x: &[T], sh: &[usize], dim: usize) -> (Vec<i64>, Vec<usize>) {
    let mut out_sh = sh.to_vec();
    out_sh[dim] = 1;
    let n = sh[dim];
    let outer: usize = sh[..dim].iter().product();
    let inner: usize = sh[dim + 1..].iter().product();
    let mut v = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let at = |k: usize| x[(o * n + k) * inner + i];
            let (mut best, mut idx) = (at(0), 0usize);
            for k in 0..n {
                let e = at(k);
                if best != best {
                    break;
                }
                if e != e || e > best {
                    best = e;
                    idx = k;
                }
            }
            v.push(idx as i64);
        }
    }
    (v, out_sh)
}

/// `gather(dim, indices)`: out[p] = x[p with coordinate `dim` replaced by indices[p]]. The index
/// shape is the output shape; its other dimensions match `x`'s.
pub(crate) fn gather<T: Copy>(x: &[T], sh: &[usize], dim: usize, idx: &[i64], ish: &[usize]) -> Vec<T> {
    check_index_shape(sh, dim, ish, "gather");
    let n = sh[dim];
    let m = ish[dim];
    let outer: usize = ish[..dim].iter().product();
    let inner: usize = ish[dim + 1..].iter().product();
    let mut v = Vec::with_capacity(idx.len());
    for o in 0..outer {
        for k in 0..m {
            for i in 0..inner {
                let j = idx[(o * m + k) * inner + i] as usize;
                assert!(j < n, "gather index {j} out of range {n}");
                v.push(x[(o * n + j) * inner + i]);
            }
        }
    }
    v
}

/// `scatter_add` into zeros of shape `sh` (burn's gather backward): every target accumulates
/// its contributions in ascending position along `dim`, starting from 0.
pub(crate) fn scatter_add_zeros<T: FloatElem>(sh: &[usize], dim: usize, idx: &[i64], ish: &[usize], vals: &[T]) -> Vec<T> {
    check_index_shape(sh, dim, ish, "scatter_add");
    assert_eq!(vals.len(), numel(ish), "scatter_add: values must have the index shape");
    assert!(idx.iter().all(|&j| j >= 0 && (j as usize) < sh[dim]), "scatter_add: index out of range {}", sh[dim]);
    let n = sh[dim];
    let m = ish[dim];
    let outer: usize = ish[..dim].iter().product();
    let inner: usize = ish[dim + 1..].iter().product();
    let mut out = vec![T::ZERO; numel(sh)];
    for o in 0..outer {
        for k in 0..m {
            for i in 0..inner {
                let p = (o * m + k) * inner + i;
                let j = idx[p] as usize;
                let t = &mut out[(o * n + j) * inner + i];
                *t = *t + vals[p];
            }
        }
    }
    out
}

/// `select(dim, indices)`: slices along `dim` in the order of `indices`.
pub(crate) fn select<T: Copy>(x: &[T], sh: &[usize], dim: usize, indices: &[i64]) -> (Vec<T>, Vec<usize>) {
    assert!(dim < sh.len(), "select: dim {dim} out of range for {sh:?}");
    assert!(indices.iter().all(|&j| j >= 0 && (j as usize) < sh[dim]), "select: index out of range {}", sh[dim]);
    let n = sh[dim];
    let outer: usize = sh[..dim].iter().product();
    let inner: usize = sh[dim + 1..].iter().product();
    let mut v = Vec::with_capacity(outer * indices.len() * inner);
    for o in 0..outer {
        for &j in indices {
            let j = j as usize;
            v.extend_from_slice(&x[(o * n + j) * inner..(o * n + j + 1) * inner]);
        }
    }
    let mut out = sh.to_vec();
    out[dim] = indices.len();
    (v, out)
}

/// `select_add` into zeros of shape `sh`: `value`'s slice t is added to slice indices[t], in order.
pub(crate) fn select_add_zeros<T: FloatElem>(sh: &[usize], dim: usize, indices: &[i64], value: &[T]) -> Vec<T> {
    assert!(dim < sh.len(), "select_add: dim {dim} out of range for {sh:?}");
    assert!(indices.iter().all(|&j| j >= 0 && (j as usize) < sh[dim]), "select_add: index out of range {}", sh[dim]);
    assert_eq!(value.len(), numel(sh) / sh[dim].max(1) * indices.len(), "select_add: value shape does not match");
    let n = sh[dim];
    let m = indices.len();
    let outer: usize = sh[..dim].iter().product();
    let inner: usize = sh[dim + 1..].iter().product();
    let mut out = vec![T::ZERO; numel(sh)];
    for (t, &j) in indices.iter().enumerate() {
        let j = j as usize;
        for o in 0..outer {
            let dst = &mut out[(o * n + j) * inner..(o * n + j + 1) * inner];
            let src = &value[(o * m + t) * inner..(o * m + t + 1) * inner];
            for (a, &b) in dst.iter_mut().zip(src) {
                *a = *a + b;
            }
        }
    }
    out
}

pub(crate) fn check_ranges(sh: &[usize], ranges: &[Range<usize>]) {
    assert_eq!(ranges.len(), sh.len(), "one range per dimension: {ranges:?} for {sh:?}");
    for (r, &d) in ranges.iter().zip(sh) {
        assert!(r.start <= r.end && r.end <= d, "slice {r:?} out of bounds for {d}");
    }
}

/// Every dimension of `ish` but `dim` must equal `sh`'s (gather, scatter-add).
fn check_index_shape(sh: &[usize], dim: usize, ish: &[usize], what: &str) {
    assert_eq!(sh.len(), ish.len(), "{what}: index rank {} vs tensor rank {}", ish.len(), sh.len());
    assert!(dim < sh.len(), "{what}: dim {dim} out of range for rank {}", sh.len());
    for (i, (&a, &b)) in sh.iter().zip(ish).enumerate() {
        assert!(i == dim || a == b, "{what}: index shape {ish:?} does not match {sh:?} outside dim {dim}");
    }
}

pub(crate) fn slice<T: Copy>(x: &[T], sh: &[usize], ranges: &[Range<usize>]) -> (Vec<T>, Vec<usize>) {
    check_ranges(sh, ranges);
    let out: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
    let st = strides(sh);
    let start: usize = ranges.iter().zip(&st).map(|(r, s)| r.start * s).sum();
    let mut v = Vec::with_capacity(numel(&out));
    for_each_offset(&out, &st, |o| v.push(x[start + o]));
    (v, out)
}

pub(crate) fn slice_assign<T: Copy>(x: &[T], sh: &[usize], ranges: &[Range<usize>], value: &[T]) -> Vec<T> {
    check_ranges(sh, ranges);
    let out: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
    assert_eq!(numel(&out), value.len(), "slice_assign: value shape must match the slice");
    let st = strides(sh);
    let start: usize = ranges.iter().zip(&st).map(|(r, s)| r.start * s).sum();
    let mut v = x.to_vec();
    let mut k = 0;
    for_each_offset(&out, &st, |o| {
        v[start + o] = value[k];
        k += 1;
    });
    v
}

/// Swap two dimensions, materialised in row-major order.
pub(crate) fn swap_dims<T: Copy>(x: &[T], sh: &[usize], a: usize, b: usize) -> (Vec<T>, Vec<usize>) {
    let mut out = sh.to_vec();
    out.swap(a, b);
    let mut st = strides(sh);
    st.swap(a, b);
    let mut v = Vec::with_capacity(x.len());
    for_each_offset(&out, &st, |o| v.push(x[o]));
    (v, out)
}

pub(crate) fn cat<T: Copy>(parts: &[(&[T], &[usize])], dim: usize) -> (Vec<T>, Vec<usize>) {
    assert!(!parts.is_empty(), "cat of no tensors");
    let first = parts[0].1;
    assert!(dim < first.len(), "cat: dim {dim} out of range for {first:?}");
    for (_, sh) in parts {
        assert!(sh.len() == first.len() && sh.iter().zip(first).enumerate().all(|(i, (a, b))| i == dim || a == b), "cat: shapes {sh:?} and {first:?} differ outside dim {dim}");
    }
    let mut out = first.to_vec();
    out[dim] = parts.iter().map(|p| p.1[dim]).sum();
    let outer: usize = first[..dim].iter().product();
    let inner: usize = first[dim + 1..].iter().product();
    let mut v = Vec::with_capacity(numel(&out));
    for o in 0..outer {
        for (x, sh) in parts {
            let w = sh[dim] * inner;
            v.extend_from_slice(&x[o * w..(o + 1) * w]);
        }
    }
    (v, out)
}

/// Descending sort with indices along `dim` for a rank ≥ 2 tensor: per lane a stable
/// sort in `sort_order` (descending), so ties keep ascending source position on every lane length
/// and toolchain (burn 0.21 used `sort_unstable_by`, whose tie order is unspecified past std's
/// small-slice insertion sort). The GPU sorts use the same order.
/// The order the sorts use: `total_cmp` with every NaN, whatever its sign bit and payload,
/// equal to every other NaN and above +inf, as PyTorch's sort treats NaN (x86's default NaN has
/// the sign bit set, aarch64's and CUDA's do not, so ordering NaN by its bits would send the same
/// computation's NaN to opposite ends on different machines). Unlike PyTorch, −0 and +0 are not
/// equal: +0 sorts above −0 (so, descending, +0 comes first), as `total_cmp` orders them.
pub(crate) fn sort_order<T: FloatElem>(a: T, b: T) -> std::cmp::Ordering {
    #[allow(clippy::eq_op)]
    match (a != a, b != b) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

pub(crate) fn sort_desc_with_indices<T: FloatElem>(x: &[T], sh: &[usize], dim: usize) -> (Vec<T>, Vec<i64>) {
    assert!(sh.len() >= 2, "sort needs a rank ≥ 2 tensor");
    let n = sh[dim];
    let outer: usize = sh[..dim].iter().product();
    let inner: usize = sh[dim + 1..].iter().product();
    let mut vals = vec![T::ZERO; x.len()];
    let mut idx = vec![0i64; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut elements: Vec<(usize, usize, T)> = (0..n)
                .map(|d| {
                    let flat = (o * n + d) * inner + i;
                    (d, flat, x[flat])
                })
                .collect();
            elements.sort_by(|&(_, _, a), &(_, _, b)| sort_order(b, a));
            for (p, (d, _, e)) in elements.into_iter().enumerate() {
                let at = (o * n + p) * inner + i;
                vals[at] = e;
                idx[at] = d as i64;
            }
        }
    }
    (vals, idx)
}

/// Batched matmul `[.., m, k] × [.., k, n]` with batch dimensions broadcast when 1, one
/// `matrixmultiply::sgemm` call per output batch (burn-ndarray's call; the same kernel gives
/// the same fused multiply–add order).
/// The batches of a matmul: (m, k, n), the output shape, and per output batch the offsets of
/// its lhs and rhs blocks (batch dimensions broadcast when 1).
pub(crate) fn matmul_plan(ash: &[usize], bsh: &[usize]) -> ((usize, usize, usize), Vec<usize>, Vec<(usize, usize)>) {
    let r = ash.len();
    assert!(r >= 2 && bsh.len() == r, "matmul needs equal ranks ≥ 2: {ash:?} × {bsh:?}");
    let (m, k) = (ash[r - 2], ash[r - 1]);
    let (k2, n) = (bsh[r - 2], bsh[r - 1]);
    assert_eq!(k, k2, "matmul inner dimensions: {ash:?} × {bsh:?}");
    let batch = broadcast(&ash[..r - 2], &bsh[..r - 2]);
    let mut out = batch.clone();
    out.extend([m, n]);
    let sa = broadcast_strides(&ash[..r - 2], &batch);
    let sb = broadcast_strides(&bsh[..r - 2], &batch);
    let mut ob = Vec::new();
    for_each_offset(&batch, &sa, |o| ob.push(o));
    let mut plan = Vec::with_capacity(ob.len());
    let mut t = 0;
    for_each_offset(&batch, &sb, |bo| {
        plan.push((ob[t] * m * k, bo * k * n));
        t += 1;
    });
    ((m, k, n), out, plan)
}

pub(crate) fn matmul<T: FloatElem>(a: &[T], ash: &[usize], b: &[T], bsh: &[usize]) -> (Vec<T>, Vec<usize>) {
    let ((m, k, n), out, plan) = matmul_plan(ash, bsh);
    let mut c = vec![T::ZERO; numel(&out)];
    for (t, (ao, bo)) in plan.into_iter().enumerate() {
        let c_slice = &mut c[t * m * n..(t + 1) * m * n];
        // SAFETY: the pointers cover m×k, k×n and m×n row-major blocks inside their buffers.
        unsafe {
            T::gemm(m, k, n, a[ao..].as_ptr(), b[bo..].as_ptr(), c_slice.as_mut_ptr());
        }
    }
    (c, out)
}
