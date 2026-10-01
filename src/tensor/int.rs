//! The i64 tensor (tokens, targets, indices). Never tracked.

use super::kernels as k;
use super::shape::numel;
use super::{BoolTensor, Tensor};
use std::ops::Range;
use super::error::ok;
use super::infer;
use super::layout::Layout;
use super::storage::Storage;
use super::DType;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct IntTensor {
    pub(crate) storage: Arc<Storage>,
    pub(crate) layout: Layout,
}

impl IntTensor {
    /// The elements in row-major order.
    pub(crate) fn data(&self) -> std::borrow::Cow<'_, [i64]> {
        let Some(s) = self.storage.try_slice::<i64>() else {
            // I32 or U8 storage: read as i64 (int ops compute in i64).
            let all = self.storage.int_vec().expect("an integer storage");
            return std::borrow::Cow::Owned(k::materialize(&all, &self.layout));
        };
        if self.layout.is_contiguous() {
            std::borrow::Cow::Borrowed(&s[self.layout.offset..self.layout.offset + self.layout.numel()])
        } else {
            std::borrow::Cow::Owned(k::materialize(s, &self.layout))
        }
    }

    /// The same elements under `shape`: in place for a row-major layout, else copied.
    pub(crate) fn reshaped(self, shape: Vec<usize>) -> IntTensor {
        match self.layout.reshaped(shape.clone()) {
            Some(l) => IntTensor { storage: self.storage, layout: l },
            None => IntTensor::raw(self.data().into_owned(), shape),
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

    pub(crate) fn raw(data: Vec<i64>, shape: Vec<usize>) -> IntTensor {
        assert_eq!(data.len(), numel(&shape), "data length {} does not match shape {:?}", data.len(), shape);
        IntTensor { storage: Arc::new(Storage::from_vec(data)), layout: Layout::contiguous(shape) }
    }

    /// A rank-1 tensor of the values (burn `from_ints`).
    pub fn from_ints(values: &[i64]) -> IntTensor {
        IntTensor::raw(values.to_vec(), vec![values.len()])
    }

    pub fn from_data(values: Vec<i64>, shape: impl Into<Vec<usize>>) -> IntTensor {
        IntTensor::raw(values, shape.into())
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> IntTensor {
        let shape = shape.into();
        IntTensor::raw(vec![0; numel(&shape)], shape)
    }

    pub fn zeros_like(&self) -> IntTensor {
        IntTensor::zeros(self.layout.shape.clone())
    }

    pub fn arange(r: Range<i64>) -> IntTensor {
        let v: Vec<i64> = r.collect();
        let n = v.len();
        IntTensor::raw(v, vec![n])
    }

    pub fn dims<const D: usize>(&self) -> [usize; D] {
        self.layout.shape.as_slice().try_into().unwrap_or_else(|_| panic!("rank {} tensor read as rank {D}", self.layout.shape.len()))
    }

    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    pub fn to_vec(&self) -> Vec<i64> {
        self.data().to_vec()
    }

    /// The values in row-major order (borrowed when the layout is one block, else copied).
    pub fn as_slice(&self) -> std::borrow::Cow<'_, [i64]> {
        self.data()
    }

    #[track_caller]
    pub fn reshape<const D: usize>(self, shape: [usize; D]) -> IntTensor {
        ok(infer::reshape(&self.layout.shape, &shape));
        assert_eq!(numel(&shape), self.data().len(), "reshape {:?} → {:?}", self.layout.shape, shape);
        self.reshaped(shape.to_vec())
    }

    #[track_caller]
    pub fn unsqueeze_dim(self, dim: usize) -> IntTensor {
        ok(infer::unsqueeze(&self.layout.shape, dim));
        let mut s = self.layout.shape.clone();
        s.insert(dim, 1);
        self.reshaped(s)
    }

    pub fn expand<const D: usize>(self, shape: [usize; D]) -> IntTensor {
        self.expand_dyn(shape.to_vec())
    }

    #[track_caller]
    pub(crate) fn expand_dyn(self, shape: Vec<usize>) -> IntTensor {
        ok(infer::expand(&self.layout.shape, &shape).map(drop));
        let (ni, no) = (self.layout.shape.len(), shape.len());
        let mut aligned = vec![1; no];
        aligned[no - ni..].copy_from_slice(&self.layout.shape);
        IntTensor::raw(k::broadcast_to(&self.data(), &aligned, &shape), shape)
    }

    pub fn slice<const D: usize>(self, ranges: [Range<usize>; D]) -> IntTensor {
        self.slice_dyn(ranges.to_vec())
    }

    #[track_caller]
    pub(crate) fn slice_dyn(self, ranges: Vec<Range<usize>>) -> IntTensor {
        ok(infer::slice(&self.layout.shape, &ranges).map(drop));
        let (v, sh) = k::slice(&self.data(), &self.layout.shape, &ranges);
        IntTensor::raw(v, sh)
    }

    #[track_caller]
    pub fn swap_dims(self, a: usize, b: usize) -> IntTensor {
        ok(infer::swap(&self.layout.shape, a, b));
        let (v, sh) = k::swap_dims(&self.data(), &self.layout.shape, a, b);
        IntTensor::raw(v, sh)
    }

    #[track_caller]
    pub fn cat(tensors: Vec<IntTensor>, dim: usize) -> IntTensor {
        ok(infer::cat(&tensors.iter().map(|t| t.layout.shape.as_slice()).collect::<Vec<_>>(), dim).map(drop));
        let datas: Vec<std::borrow::Cow<'_, [i64]>> = tensors.iter().map(|t| t.data()).collect();
        let parts: Vec<(&[i64], &[usize])> = datas.iter().zip(&tensors).map(|(d, t)| (&**d, t.layout.shape.as_slice())).collect();
        let (v, sh) = k::cat(&parts, dim);
        IntTensor::raw(v, sh)
    }

    #[track_caller]
    pub fn gather(self, dim: usize, indices: IntTensor) -> IntTensor {
        ok(infer::index(&self.layout.shape, dim, &indices.layout.shape, &indices.data(), "gather"));
        IntTensor::raw(k::gather(&self.data(), &self.layout.shape, dim, &indices.data(), &indices.layout.shape), indices.layout.shape.clone())
    }

    #[track_caller]
    pub fn select(self, dim: usize, indices: IntTensor) -> IntTensor {
        ok(infer::select(&self.layout.shape, dim, &indices.data(), "select"));
        let (v, sh) = k::select(&self.data(), &self.layout.shape, dim, &indices.data());
        IntTensor::raw(v, sh)
    }

    /// burn-ndarray's integer remainder: `((x % r) + r) % r`.
    pub fn remainder_scalar(self, r: i64) -> IntTensor {
        IntTensor::raw(k::map(&self.data(), |x| ((x % r) + r) % r), self.layout.shape.clone())
    }

    /// Integer division (truncating, as Rust's `/`).
    pub fn div_scalar(self, d: i64) -> IntTensor {
        IntTensor::raw(k::map(&self.data(), |x| x / d), self.layout.shape.clone())
    }

    pub fn mul_scalar(self, m: i64) -> IntTensor {
        IntTensor::raw(k::map(&self.data(), |x| x * m), self.layout.shape.clone())
    }

    pub fn add_scalar(self, a: i64) -> IntTensor {
        IntTensor::raw(k::map(&self.data(), |x| x + a), self.layout.shape.clone())
    }

    #[track_caller]
    pub fn add(self, rhs: IntTensor) -> IntTensor {
        ok(infer::broadcast(&self.layout.shape, &rhs.layout.shape).map(drop));
        let (v, sh) = k::zip(&self.data(), &self.layout.shape, &rhs.data(), &rhs.layout.shape, |a, b| a + b);
        IntTensor::raw(v, sh)
    }

    /// One-hot along a new last dimension of size `n` (burn `one_hot`); values must be in 0..n.
    #[track_caller]
    pub fn one_hot(self, n: usize) -> IntTensor {
        ok(infer::one_hot(&self.data(), n));
        let mut v = vec![0i64; self.data().len() * n];
        for (i, &x) in self.data().iter().enumerate() {
            assert!(x >= 0 && (x as usize) < n, "one_hot: {x} is outside 0..{n}");
            v[i * n + x as usize] = 1;
        }
        let mut sh = self.layout.shape.clone();
        sh.push(n);
        IntTensor::raw(v, sh)
    }

    /// `one_hot(n)` as floats of `dtype` (f64, else f32, as `float_dtype`) on `device`: on a GPU
    /// the rows are written by a kernel from the uploaded ids, on the CPU this is
    /// `one_hot(n).float_dtype(dtype)` moved to `device`.
    #[track_caller]
    pub(crate) fn one_hot_float(self, n: usize, dtype: DType, device: super::Device) -> Tensor {
        if matches!(device, super::Device::Metal(_) | super::Device::Cuda(_)) && dtype != DType::F64 {
            ok(infer::one_hot(&self.data(), n));
            let mut sh = self.layout.shape.clone();
            sh.push(n);
            return Tensor::fresh(ok(super::gpu::gpu_one_hot(device, &self, n)), sh, device);
        }
        self.one_hot(n).float_dtype(dtype).on(device)
    }

    /// Store as another integer dtype (I64, I32 or U8; values wrap by `as`). Ops read any of them
    /// as i64.
    pub fn cast(self, to: DType) -> IntTensor {
        let v = self.data().into_owned();
        let storage = match to {
            DType::I64 => Storage::from_vec(v),
            DType::I32 => Storage::from_vec(v.iter().map(|x| *x as i32).collect::<Vec<_>>()),
            DType::U8 => Storage::from_vec(v.iter().map(|x| *x as u8).collect::<Vec<_>>()),
            other => panic!("cast of an int tensor to {}", other.name()),
        };
        IntTensor { storage: Arc::new(storage), layout: Layout::contiguous(self.layout.shape.clone()) }
    }

    /// Convert to a float dtype (f32 or f64) by `as` (burn's `int_into_float` with that dtype).
    pub fn float_dtype(self, dtype: DType) -> Tensor {
        match dtype {
            DType::F64 => Tensor::raw_t(k::map(&self.data(), |x| x as f64), self.layout.shape.clone()),
            _ => self.float(),
        }
    }

    /// Convert to f32 (a fresh, order-0 tensor, as burn's `int_into_float`).
    pub fn float(self) -> Tensor {
        Tensor::raw(k::map(&self.data(), |x| x as f32), self.layout.shape.clone())
    }

    pub fn equal_elem(self, s: i64) -> BoolTensor {
        BoolTensor::raw(k::map(&self.data(), |x| x == s), self.layout.shape.clone())
    }

    pub fn not_equal_elem(self, s: i64) -> BoolTensor {
        BoolTensor::raw(k::map(&self.data(), |x| x != s), self.layout.shape.clone())
    }

    pub fn greater_equal_elem(self, s: i64) -> BoolTensor {
        BoolTensor::raw(k::map(&self.data(), |x| x >= s), self.layout.shape.clone())
    }

    pub fn greater_elem(self, s: i64) -> BoolTensor {
        BoolTensor::raw(k::map(&self.data(), |x| x > s), self.layout.shape.clone())
    }

    pub fn lower_elem(self, s: i64) -> BoolTensor {
        BoolTensor::raw(k::map(&self.data(), |x| x < s), self.layout.shape.clone())
    }
}

impl std::ops::Add<IntTensor> for IntTensor {
    type Output = IntTensor;
    fn add(self, rhs: IntTensor) -> IntTensor {
        IntTensor::add(self, rhs)
    }
}
