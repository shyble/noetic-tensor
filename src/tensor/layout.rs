//! Layout: how a tensor reads its storage — shape, strides and offset.
//! Reshape (of a row-major layout), unsqueeze, swap_dims, slice and expand are views: new
//! layouts over the same storage. A kernel reads a view through `materialize` (row-major order),
//! so values, and every arithmetic order, are those of a copied tensor.

use super::shape::{numel, strides};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

impl Layout {
    /// The contiguous row-major layout of `shape`.
    pub fn contiguous(shape: Vec<usize>) -> Layout {
        let strides = strides(&shape);
        Layout { shape, strides, offset: 0 }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    /// Whether the elements are one row-major block (from `offset`): a kernel can read them in
    /// place.
    pub fn is_contiguous(&self) -> bool {
        self.strides == strides(&self.shape)
    }

    /// The same elements under another shape, if the layout is one row-major block.
    pub(crate) fn reshaped(&self, shape: Vec<usize>) -> Option<Layout> {
        self.is_contiguous().then(|| Layout { strides: strides(&shape), shape, offset: self.offset })
    }

    pub(crate) fn swapped(&self, a: usize, b: usize) -> Layout {
        let mut l = self.clone();
        l.shape.swap(a, b);
        l.strides.swap(a, b);
        l
    }

    pub(crate) fn narrowed(&self, ranges: &[std::ops::Range<usize>]) -> Layout {
        let offset = self.offset + ranges.iter().zip(&self.strides).map(|(r, s)| r.start * s).sum::<usize>();
        Layout { shape: ranges.iter().map(|r| r.end - r.start).collect(), strides: self.strides.clone(), offset }
    }

    /// Broadcast to `out`: the shape right-aligned, stride 0 on every expanded dimension.
    pub(crate) fn broadcast(&self, out: &[usize]) -> Layout {
        let (ni, no) = (self.shape.len(), out.len());
        let mut st = vec![0; no];
        for i in 0..ni {
            let o = no - ni + i;
            st[o] = if self.shape[i] == out[o] { self.strides[i] } else { 0 };
        }
        Layout { shape: out.to_vec(), strides: st, offset: self.offset }
    }
}
