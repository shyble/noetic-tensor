//! The bool tensor (masks). Never tracked.

use super::kernels as k;
use super::shape::numel;
use super::Tensor;
use super::error::ok;
use super::infer;
use super::layout::Layout;
use super::storage::Storage;
use super::DType;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct BoolTensor {
    pub(crate) storage: Arc<Storage>,
    pub(crate) layout: Layout,
}

impl BoolTensor {
    /// The elements in row-major order.
    pub(crate) fn data(&self) -> std::borrow::Cow<'_, [bool]> {
        let Some(s) = self.storage.try_slice::<bool>() else {
            // A mask computed on a GPU: read back.
            let host = self.storage.host();
            let super::storage::CpuStorage::Bool(all) = &*host else { panic!("a bool tensor holds {}", host.dtype().name()) };
            return std::borrow::Cow::Owned(k::materialize(all, &self.layout));
        };
        if self.layout.is_contiguous() {
            std::borrow::Cow::Borrowed(&s[self.layout.offset..self.layout.offset + self.layout.numel()])
        } else {
            std::borrow::Cow::Owned(k::materialize(s, &self.layout))
        }
    }

    /// The same elements under `shape`: in place for a row-major layout, else copied.
    pub(crate) fn reshaped(self, shape: Vec<usize>) -> BoolTensor {
        match self.layout.reshaped(shape.clone()) {
            Some(l) => BoolTensor { storage: self.storage, layout: l },
            None => BoolTensor::raw(self.data().into_owned(), shape),
        }
    }

    /// Whether this tensor is a view that a kernel would have to copy.
    pub fn is_view(&self) -> bool {
        !self.layout.is_contiguous()
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(crate) fn raw(data: Vec<bool>, shape: Vec<usize>) -> BoolTensor {
        assert_eq!(data.len(), numel(&shape), "data length {} does not match shape {:?}", data.len(), shape);
        BoolTensor { storage: Arc::new(Storage::from_vec(data)), layout: Layout::contiguous(shape) }
    }

    pub fn from_data(values: Vec<bool>, shape: impl Into<Vec<usize>>) -> BoolTensor {
        BoolTensor::raw(values, shape.into())
    }

    /// burn `tril_mask([rows, cols], offset)`: true where `col > row + offset` (the part outside
    /// the lower triangle; with offset 0, the strict upper triangle).
    pub fn tril_mask(shape: [usize; 2], offset: i64) -> BoolTensor {
        let [h, w] = shape;
        let v = (0..h * w).map(|p| ((p / w) as i64 - ((p % w) as i64 - offset)) < 0).collect();
        BoolTensor::raw(v, vec![h, w])
    }

    pub fn dims<const D: usize>(&self) -> [usize; D] {
        self.layout.shape.as_slice().try_into().unwrap_or_else(|_| panic!("rank {} tensor read as rank {D}", self.layout.shape.len()))
    }

    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    pub fn to_vec(&self) -> Vec<bool> {
        self.data().to_vec()
    }

    #[track_caller]
    pub fn reshape<const D: usize>(self, shape: [usize; D]) -> BoolTensor {
        ok(infer::reshape(&self.layout.shape, &shape));
        assert_eq!(numel(&shape), self.data().len());
        self.reshaped(shape.to_vec())
    }

    #[track_caller]
    pub fn unsqueeze_dim(self, dim: usize) -> BoolTensor {
        ok(infer::unsqueeze(&self.layout.shape, dim));
        let mut s = self.layout.shape.clone();
        s.insert(dim, 1);
        self.reshaped(s)
    }

    pub fn expand<const D: usize>(self, shape: [usize; D]) -> BoolTensor {
        self.expand_dyn(shape.to_vec())
    }

    #[track_caller]
    pub(crate) fn expand_dyn(self, shape: Vec<usize>) -> BoolTensor {
        ok(infer::expand(&self.layout.shape, &shape).map(drop));
        let (ni, no) = (self.layout.shape.len(), shape.len());
        let mut aligned = vec![1; no];
        aligned[no - ni..].copy_from_slice(&self.layout.shape);
        BoolTensor::raw(k::broadcast_to(&self.data(), &aligned, &shape), shape)
    }

    pub fn bool_not(self) -> BoolTensor {
        BoolTensor::raw(k::map(&self.data(), |b| !b), self.layout.shape.clone())
    }

    /// Convert to a float dtype (f32 or f64): 1 and 0.
    pub fn float_dtype(self, dtype: DType) -> Tensor {
        match dtype {
            DType::F64 => Tensor::raw_t(k::map(&self.data(), |b| if b { 1.0f64 } else { 0.0 }), self.layout.shape.clone()),
            _ => self.float(),
        }
    }

    /// Convert to f32 (1.0 / 0.0; a fresh, order-0 tensor).
    pub fn float(self) -> Tensor {
        // A mask computed on a GPU converts there (Adam's lazy-row masks stay on the device).
        if let Some(t) = super::gpu::gpu_bool_float(&self) {
            return t;
        }
        Tensor::raw(k::map(&self.data(), |b| if b { 1.0 } else { 0.0 }), self.layout.shape.clone())
    }
}
