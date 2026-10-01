//! Shape inference and checks: each operation's output shape is
//! computed here from its inputs' shapes, and every way the inputs can be wrong is an error
//! here, before any kernel runs. The kernels keep their own asserts as a second line.

use super::error::{Result, TensorError};
use std::ops::Range;

fn numel(s: &[usize]) -> usize {
    s.iter().product()
}

pub(crate) fn dim(shape: &[usize], d: usize, op: &str) -> Result<()> {
    if d < shape.len() { Ok(()) } else { Err(TensorError::Index(format!("{op}: dim {d} out of range for {shape:?}"))) }
}

/// burn's broadcasting of two shapes: equal ranks, each dimension equal or 1.
pub(crate) fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    if a.len() != b.len() {
        return Err(TensorError::Broadcast(format!("broadcasting needs equal ranks: {a:?} vs {b:?}")));
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| if x == y || x == 1 || y == 1 { Ok(x.max(y)) } else { Err(TensorError::Broadcast(format!("shapes {a:?} and {b:?} do not broadcast"))) })
        .collect()
}

/// `shape` read as `out` (equal ranks; each dimension equal or 1).
pub(crate) fn broadcast_to(shape: &[usize], out: &[usize]) -> Result<()> {
    if shape.len() != out.len() {
        return Err(TensorError::Broadcast(format!("cannot broadcast rank {} {shape:?} to rank {} {out:?}", shape.len(), out.len())));
    }
    if shape.iter().zip(out).all(|(&d, &o)| d == o || d == 1) { Ok(()) } else { Err(TensorError::Broadcast(format!("cannot broadcast {shape:?} to {out:?}"))) }
}

/// `expand`: the input right-aligned against the output, then broadcast.
pub(crate) fn expand(shape: &[usize], out: &[usize]) -> Result<Vec<usize>> {
    if out.len() < shape.len() {
        return Err(TensorError::Broadcast(format!("cannot broadcast {shape:?} to the lower rank {out:?}")));
    }
    let mut aligned = vec![1; out.len()];
    aligned[out.len() - shape.len()..].copy_from_slice(shape);
    broadcast_to(&aligned, out)?;
    Ok(aligned)
}

pub(crate) fn reshape(from: &[usize], to: &[usize]) -> Result<()> {
    if numel(from) == numel(to) { Ok(()) } else { Err(TensorError::Shape(format!("reshape {from:?} → {to:?}: {} elements vs {}", numel(from), numel(to)))) }
}

pub(crate) fn swap(shape: &[usize], a: usize, b: usize) -> Result<()> {
    dim(shape, a, "swap_dims")?;
    dim(shape, b, "swap_dims")
}

pub(crate) fn unsqueeze(shape: &[usize], d: usize) -> Result<()> {
    if d <= shape.len() { Ok(()) } else { Err(TensorError::Index(format!("unsqueeze_dim: dim {d} out of range for {shape:?}"))) }
}

pub(crate) fn slice(shape: &[usize], ranges: &[Range<usize>]) -> Result<Vec<usize>> {
    if ranges.len() != shape.len() {
        return Err(TensorError::Shape(format!("one range per dimension: {ranges:?} for {shape:?}")));
    }
    for (r, &d) in ranges.iter().zip(shape) {
        if !(r.start <= r.end && r.end <= d) {
            return Err(TensorError::Index(format!("slice {r:?} out of bounds for {d}")));
        }
    }
    Ok(ranges.iter().map(|r| r.end - r.start).collect())
}

pub(crate) fn slice_assign(shape: &[usize], ranges: &[Range<usize>], value: &[usize]) -> Result<()> {
    let out = slice(shape, ranges)?;
    if numel(&out) == numel(value) { Ok(()) } else { Err(TensorError::Shape(format!("slice_assign: value shape must match the slice ({value:?} into {out:?})"))) }
}

pub(crate) fn cat(shapes: &[&[usize]], d: usize) -> Result<Vec<usize>> {
    let first = shapes.first().ok_or_else(|| TensorError::Shape("cat of no tensors".into()))?;
    dim(first, d, "cat")?;
    for sh in shapes {
        if !(sh.len() == first.len() && sh.iter().zip(first.iter()).enumerate().all(|(i, (a, b))| i == d || a == b)) {
            return Err(TensorError::Shape(format!("cat: shapes {sh:?} and {first:?} differ outside dim {d}")));
        }
    }
    let mut out = first.to_vec();
    out[d] = shapes.iter().map(|s| s[d]).sum();
    Ok(out)
}

/// `[.., m, k] × [.., k, n]` with the batch dimensions broadcast.
pub(crate) fn matmul(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let r = a.len();
    if r < 2 || b.len() != r {
        return Err(TensorError::Shape(format!("matmul needs equal ranks ≥ 2: {a:?} × {b:?}")));
    }
    if a[r - 1] != b[r - 2] {
        return Err(TensorError::Shape(format!("matmul inner dimensions: {a:?} × {b:?}")));
    }
    let mut out = broadcast(&a[..r - 2], &b[..r - 2])?;
    out.extend([a[r - 2], b[r - 1]]);
    Ok(out)
}

/// gather and scatter-add: every index dimension but `d` equals the tensor's, and every index
/// is in range along `d`.
pub(crate) fn index(shape: &[usize], d: usize, ishape: &[usize], idx: &[i64], what: &str) -> Result<()> {
    if shape.len() != ishape.len() {
        return Err(TensorError::Shape(format!("{what}: index rank {} vs tensor rank {}", ishape.len(), shape.len())));
    }
    dim(shape, d, what)?;
    for (i, (&a, &b)) in shape.iter().zip(ishape).enumerate() {
        if i != d && a != b {
            return Err(TensorError::Shape(format!("{what}: index shape {ishape:?} does not match {shape:?} outside dim {d}")));
        }
    }
    match idx.iter().find(|&&j| j < 0 || j as usize >= shape[d]) {
        Some(j) => Err(TensorError::Index(format!("{what} index {j} out of range {}", shape[d]))),
        None => Ok(()),
    }
}

pub(crate) fn select(shape: &[usize], d: usize, indices: &[i64], what: &str) -> Result<()> {
    dim(shape, d, what)?;
    match indices.iter().find(|&&j| j < 0 || j as usize >= shape[d]) {
        Some(_) => Err(TensorError::Index(format!("{what}: index out of range {}", shape[d]))),
        None => Ok(()),
    }
}

pub(crate) fn mask(shape: &[usize], mask: &[usize]) -> Result<()> {
    if mask.len() != shape.len() {
        return Err(TensorError::Shape(format!("mask_fill: mask rank {mask:?} vs tensor rank {shape:?}")));
    }
    broadcast_to(mask, shape)
}

pub(crate) fn one_hot(values: &[i64], n: usize) -> Result<()> {
    match values.iter().find(|&&x| x < 0 || x as usize >= n) {
        Some(x) => Err(TensorError::Index(format!("one_hot: {x} is outside 0..{n}"))),
        None => Ok(()),
    }
}

pub(crate) fn backward_root(shape: &[usize], tracked: bool) -> Result<()> {
    if numel(shape) != 1 {
        return Err(TensorError::Shape(format!("backward needs a single-element root (a loss); got shape {shape:?} (sum it first)")));
    }
    if !tracked {
        return Err(TensorError::Unsupported("backward on a tensor that does not require gradients".into()));
    }
    Ok(())
}

pub(crate) fn same_device(a: super::device::Device, b: super::device::Device) -> Result<()> {
    if a == b { Ok(()) } else { Err(TensorError::Unsupported(format!("tensors on {a:?} and {b:?} (move one with `to`)"))) }
}

pub(crate) fn same_dtype(a: super::DType, b: super::DType, op: &str) -> Result<()> {
    if a == b { Ok(()) } else { Err(TensorError::DType(format!("{op}: {} and {} (casts are explicit)", a.name(), b.name()))) }
}

pub(crate) fn scalar(shape: &[usize]) -> Result<()> {
    if numel(shape) == 1 { Ok(()) } else { Err(TensorError::Shape(format!("into_scalar on a tensor of shape {shape:?}"))) }
}
