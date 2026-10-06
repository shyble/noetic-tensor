//! A typed facade over `tensor` in burn 0.21's shape.
//!
//! It lets code written against burn's typed API move onto this engine line for line
//! (`burn::` → `tensor::api::`).
//! The typed wrapper `Tensor<B, D, K>` carries the backend (`NdArray` = untracked; `Autodiff<NdArray>`
//! = tracked), the rank and the kind (float, int, bool). No burn crate is involved.
//! Callers instantiate `CpuF32`/`CpuF32Grad` and cross to the tensor core's own types with
//! `into_raw`/`from_raw`.

use super::{BoolTensor, Gradients as RawGradients, IntTensor, Tensor as Raw};
use std::marker::PhantomData;
use std::ops::Range;

// ------------------------------------------------------------------------ backends

/// The CPU device (the only one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Cpu;

pub trait Backend: Clone + Copy + Default + std::fmt::Debug + Send + Sync + 'static {
    type Device: Clone + Default + std::fmt::Debug + Send + Sync;
    /// Whether tensors on this backend record gradients (`Autodiff`).
    const AUTODIFF: bool;
    /// The float dtype of this backend's float tensors (burn's `FloatElem`).
    const FLOAT: super::DType;
}

pub trait AutodiffBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device>;
}

/// Marker for the f32 element (the only one the facade holds).
pub trait FloatElement: Clone + Copy + Default + std::fmt::Debug + Send + Sync + 'static {
    const DTYPE: super::DType;
}
impl FloatElement for f32 {
    const DTYPE: super::DType = super::DType::F32;
}
impl FloatElement for f64 {
    const DTYPE: super::DType = super::DType::F64;
}

/// The inner (untracked) CPU backend, burn's `NdArray<f32>`.
#[derive(Clone, Copy, Default, Debug)]
pub struct NdArray<E: FloatElement = f32>(PhantomData<E>);

/// The tracked backend, burn's `Autodiff<B>`.
#[derive(Clone, Copy, Default, Debug)]
pub struct Autodiff<B: Backend>(PhantomData<B>);

impl<E: FloatElement> Backend for NdArray<E> {
    type Device = Cpu;
    const AUTODIFF: bool = false;
    const FLOAT: super::DType = E::DTYPE;
}

impl<B: Backend> Backend for Autodiff<B> {
    type Device = B::Device;
    const AUTODIFF: bool = true;
    const FLOAT: super::DType = B::FLOAT;
}

impl<B: Backend> AutodiffBackend for Autodiff<B> {
    type InnerBackend = B;
}

pub type Device<B> = <B as Backend>::Device;

/// The one backend the facade's callers instantiate: f32 on the CPU, untracked.
/// Functions written for it take this concrete type instead of a `B: Backend` parameter.
pub type CpuF32 = NdArray<f32>;

/// `CpuF32` with gradients recorded.
pub type CpuF32Grad = Autodiff<CpuF32>;

// ------------------------------------------------------------------------ kinds

pub trait TensorKind: Clone + Send + Sync + 'static {
    type Prim: Clone + Send + Sync + std::fmt::Debug;
    fn zeros(shape: Vec<usize>, float: super::DType) -> Self::Prim;
    fn shape(p: &Self::Prim) -> &[usize];
    fn values(p: &Self::Prim) -> Values;
    fn reshape(p: Self::Prim, shape: Vec<usize>) -> Self::Prim;
    fn expand(p: Self::Prim, shape: Vec<usize>) -> Self::Prim;
    fn slice(p: Self::Prim, ranges: Vec<Range<usize>>) -> Self::Prim;
    fn cat(ps: Vec<Self::Prim>, dim: usize) -> Self::Prim;
    fn gather(p: Self::Prim, dim: usize, idx: IntTensor) -> Self::Prim;
    fn select(p: Self::Prim, dim: usize, idx: IntTensor) -> Self::Prim;
}

#[derive(Clone, Debug)]
pub struct Float;
#[derive(Clone, Debug)]
pub struct Int;
#[derive(Clone, Debug)]
pub struct Bool;

impl TensorKind for Float {
    type Prim = Raw;
    fn zeros(shape: Vec<usize>, float: super::DType) -> Raw { Raw::full_dtype(shape, 0.0, float) }
    fn shape(p: &Raw) -> &[usize] { p.shape() }
    fn values(p: &Raw) -> Values { if p.dtype() == super::DType::F64 { Values::F64(p.to_vec_f64()) } else { Values::F32(p.to_vec()) } }
    fn reshape(p: Raw, shape: Vec<usize>) -> Raw { p.reshape_dyn(shape) }
    fn expand(p: Raw, shape: Vec<usize>) -> Raw { p.expand_dyn(shape) }
    fn slice(p: Raw, ranges: Vec<Range<usize>>) -> Raw { p.slice_dyn(ranges) }
    fn cat(ps: Vec<Raw>, dim: usize) -> Raw { Raw::cat(ps, dim) }
    fn gather(p: Raw, dim: usize, idx: IntTensor) -> Raw { p.gather(dim, idx) }
    fn select(p: Raw, dim: usize, idx: IntTensor) -> Raw { p.select(dim, idx) }
}
impl TensorKind for Int {
    type Prim = IntTensor;
    fn zeros(shape: Vec<usize>, _: super::DType) -> IntTensor { IntTensor::zeros(shape) }
    fn shape(p: &IntTensor) -> &[usize] { p.shape() }
    fn values(p: &IntTensor) -> Values { Values::I64(p.to_vec()) }
    fn reshape(p: IntTensor, shape: Vec<usize>) -> IntTensor { IntTensor::from_data(p.to_vec(), shape) }
    fn expand(p: IntTensor, shape: Vec<usize>) -> IntTensor { p.expand_dyn(shape) }
    fn slice(p: IntTensor, ranges: Vec<Range<usize>>) -> IntTensor { p.slice_dyn(ranges) }
    fn cat(ps: Vec<IntTensor>, dim: usize) -> IntTensor { IntTensor::cat(ps, dim) }
    fn gather(p: IntTensor, dim: usize, idx: IntTensor) -> IntTensor { p.gather(dim, idx) }
    fn select(p: IntTensor, dim: usize, idx: IntTensor) -> IntTensor { p.select(dim, idx) }
}
impl TensorKind for Bool {
    type Prim = BoolTensor;
    fn zeros(shape: Vec<usize>, _: super::DType) -> BoolTensor { let n = shape.iter().product(); BoolTensor::from_data(vec![false; n], shape) }
    fn shape(p: &BoolTensor) -> &[usize] { p.shape() }
    fn values(p: &BoolTensor) -> Values { Values::Bool(p.to_vec()) }
    fn reshape(p: BoolTensor, shape: Vec<usize>) -> BoolTensor { BoolTensor::from_data(p.to_vec(), shape) }
    fn expand(p: BoolTensor, shape: Vec<usize>) -> BoolTensor { p.expand_dyn(shape) }
    fn slice(_: BoolTensor, _: Vec<Range<usize>>) -> BoolTensor { unimplemented!("bool slice is not used") }
    fn cat(_: Vec<BoolTensor>, _: usize) -> BoolTensor { unimplemented!("bool cat is not used") }
    fn gather(_: BoolTensor, _: usize, _: IntTensor) -> BoolTensor { unimplemented!("bool gather is not used") }
    fn select(_: BoolTensor, _: usize, _: IntTensor) -> BoolTensor { unimplemented!("bool select is not used") }
}

// ------------------------------------------------------------------------ every kind

impl<B: Backend, const D: usize, K: TensorKind> Tensor<B, D, K> {
    /// The `tensor` value this wraps (`Tensor`, `IntTensor` or `BoolTensor`): the
    /// boundary from the typed facade to the tensor core.
    pub fn into_raw(self) -> K::Prim {
        self.p
    }

    /// Wrap a `tensor` value (its rank must be `D`).
    #[track_caller]
    pub fn from_raw(p: K::Prim) -> Self {
        check_rank::<D>(K::shape(&p));
        wrap(p)
    }

    pub fn zeros<S: Into<[usize; D]>>(shape: S, _device: &B::Device) -> Self {
        wrap(K::zeros(shape.into().to_vec(), B::FLOAT))
    }

    pub fn zeros_like(&self) -> Self {
        wrap(K::zeros(K::shape(&self.p).to_vec(), B::FLOAT))
    }

    pub fn dims(&self) -> [usize; D] {
        K::shape(&self.p).try_into().expect("rank")
    }

    pub fn device(&self) -> B::Device {
        Default::default()
    }

    pub fn into_data(self) -> TensorData {
        TensorData { values: K::values(&self.p), shape: K::shape(&self.p).to_vec() }
    }

    pub fn to_data(&self) -> TensorData {
        self.clone().into_data()
    }

    pub fn reshape<const D2: usize>(self, shape: [usize; D2]) -> Tensor<B, D2, K> {
        wrap(K::reshape(self.p, shape.to_vec()))
    }

    pub fn unsqueeze_dim<const D2: usize>(self, dim: usize) -> Tensor<B, D2, K> {
        let mut s = K::shape(&self.p).to_vec();
        s.insert(dim, 1);
        wrap(K::reshape(self.p, s))
    }

    pub fn expand<const D2: usize>(self, shape: [usize; D2]) -> Tensor<B, D2, K> {
        wrap(K::expand(self.p, shape.to_vec()))
    }

    pub fn slice(self, ranges: [Range<usize>; D]) -> Self {
        wrap(K::slice(self.p, ranges.to_vec()))
    }

    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        wrap(K::cat(tensors.into_iter().map(|t| t.p).collect(), dim))
    }

    pub fn gather(self, dim: usize, indices: Tensor<B, D, Int>) -> Self {
        wrap(K::gather(self.p, dim, indices.p))
    }

    pub fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self {
        wrap(K::select(self.p, dim, indices.p))
    }
}

pub struct Tensor<B: Backend, const D: usize, K: TensorKind = Float> {
    pub(crate) p: K::Prim,
    _b: PhantomData<B>,
}

impl<B: Backend, const D: usize, K: TensorKind> Clone for Tensor<B, D, K> {
    fn clone(&self) -> Self {
        Tensor { p: self.p.clone(), _b: PhantomData }
    }
}

impl<B: Backend, const D: usize, K: TensorKind> std::fmt::Debug for Tensor<B, D, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.p.fmt(f)
    }
}

fn wrap<B: Backend, const D: usize, K: TensorKind>(p: K::Prim) -> Tensor<B, D, K> {
    Tensor { p, _b: PhantomData }
}

// ------------------------------------------------------------------------ data

/// burn's `TensorData`, enough for readback: `into_data().convert::<E>().to_vec::<E>()`.
#[derive(Clone, Debug)]
pub struct TensorData {
    values: Values,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Values {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[derive(Debug)]
pub struct DataError;

pub trait Element: Copy + 'static {
    fn convert(v: &Values) -> Values;
    fn take(v: Values) -> Option<Vec<Self>>;
}

impl Element for f32 {
    fn convert(v: &Values) -> Values {
        Values::F32(match v {
            Values::F32(x) => x.clone(),
            Values::F64(x) => x.iter().map(|a| *a as f32).collect(),
            Values::I64(x) => x.iter().map(|a| *a as f32).collect(),
            Values::Bool(x) => x.iter().map(|a| if *a { 1.0 } else { 0.0 }).collect(),
        })
    }
    fn take(v: Values) -> Option<Vec<f32>> {
        if let Values::F32(x) = v { Some(x) } else { None }
    }
}

impl Element for f64 {
    fn convert(v: &Values) -> Values {
        Values::F64(match v {
            Values::F32(x) => x.iter().map(|a| *a as f64).collect(),
            Values::F64(x) => x.clone(),
            Values::I64(x) => x.iter().map(|a| *a as f64).collect(),
            Values::Bool(x) => x.iter().map(|a| if *a { 1.0 } else { 0.0 }).collect(),
        })
    }
    fn take(v: Values) -> Option<Vec<f64>> {
        if let Values::F64(x) = v { Some(x) } else { None }
    }
}

impl Element for i64 {
    fn convert(v: &Values) -> Values {
        Values::I64(match v {
            Values::F32(x) => x.iter().map(|a| *a as i64).collect(),
            Values::F64(x) => x.iter().map(|a| *a as i64).collect(),
            Values::I64(x) => x.clone(),
            Values::Bool(x) => x.iter().map(|a| *a as i64).collect(),
        })
    }
    fn take(v: Values) -> Option<Vec<i64>> {
        if let Values::I64(x) = v { Some(x) } else { None }
    }
}

impl Element for bool {
    fn convert(v: &Values) -> Values {
        Values::Bool(match v {
            Values::F32(x) => x.iter().map(|a| *a != 0.0).collect(),
            Values::F64(x) => x.iter().map(|a| *a != 0.0).collect(),
            Values::I64(x) => x.iter().map(|a| *a != 0).collect(),
            Values::Bool(x) => x.clone(),
        })
    }
    fn take(v: Values) -> Option<Vec<bool>> {
        if let Values::Bool(x) = v { Some(x) } else { None }
    }
}

impl TensorData {
    pub fn convert<E: Element>(self) -> TensorData {
        TensorData { values: E::convert(&self.values), shape: self.shape }
    }

    /// The values as `E`; an error if they are not stored as `E` (call `convert` first).
    pub fn to_vec<E: Element>(&self) -> Result<Vec<E>, DataError> {
        E::take(self.values.clone()).ok_or(DataError)
    }
}

/// Values accepted by `from_floats`: slices, vectors and nested arrays of f32 or f64.
pub trait FloatInput {
    fn flat(self) -> (Vec<f64>, Vec<usize>);
}

/// Values accepted by `from_ints`: slices, vectors and nested arrays of integers.
pub trait IntInput {
    fn flat(self) -> (Vec<i64>, Vec<usize>);
}

pub trait Scalar: Copy {
    fn to_f64(self) -> f64;
    fn to_i64(self) -> i64;
}
macro_rules! scalars {
    ($($t:ty),*) => {$(impl Scalar for $t {
        fn to_f64(self) -> f64 { self as f64 }
        fn to_i64(self) -> i64 { self as i64 }
    })*};
}
scalars!(f32, f64, i32, i64, u32, usize);

macro_rules! inputs {
    ($tr:ident, $out:ty, $conv:ident, $($t:ty),*) => {$(
        impl $tr for &[$t] {
            fn flat(self) -> (Vec<$out>, Vec<usize>) { (self.iter().map(|x| x.$conv() as $out).collect(), vec![self.len()]) }
        }
        impl $tr for &Vec<$t> {
            fn flat(self) -> (Vec<$out>, Vec<usize>) { self.as_slice().flat() }
        }
        impl $tr for Vec<$t> {
            fn flat(self) -> (Vec<$out>, Vec<usize>) { self.as_slice().flat() }
        }
        impl<const N: usize> $tr for [$t; N] {
            fn flat(self) -> (Vec<$out>, Vec<usize>) { self.as_slice().flat() }
        }
        impl<const N: usize, const M: usize> $tr for [[$t; N]; M] {
            fn flat(self) -> (Vec<$out>, Vec<usize>) {
                (self.iter().flat_map(|r| r.iter().map(|x| x.$conv() as $out)).collect(), vec![M, N])
            }
        }
        impl<const N: usize, const M: usize, const L: usize> $tr for [[[$t; N]; M]; L] {
            fn flat(self) -> (Vec<$out>, Vec<usize>) {
                (self.iter().flat_map(|m| m.iter().flat_map(|r| r.iter().map(|x| x.$conv() as $out))).collect(), vec![L, M, N])
            }
        }
    )*};
}
inputs!(FloatInput, f64, to_f64, f32, f64);
inputs!(IntInput, i64, to_i64, i32, i64);

fn check_rank<const D: usize>(shape: &[usize]) {
    assert_eq!(shape.len(), D, "data of rank {} for a rank-{D} tensor", shape.len());
}

// ------------------------------------------------------------------------ float tensors

impl<B: Backend, const D: usize> Tensor<B, D> {
    pub fn from_floats<T: FloatInput>(data: T, _device: &B::Device) -> Self {
        let (v, s) = data.flat();
        check_rank::<D>(&s);
        // The values in the backend's float dtype (burn converted through f32 first, so an
        // f64 backend lost precision here).
        wrap(Raw::from_f64s(v, s, B::FLOAT))
    }


    pub fn ones<S: Into<[usize; D]>>(shape: S, _device: &B::Device) -> Self {
        wrap(Raw::full_dtype(shape.into().to_vec(), 1.0, B::FLOAT))
    }

    pub fn full<S: Into<[usize; D]>, E: Scalar>(shape: S, value: E, _device: &B::Device) -> Self {
        wrap(Raw::full_dtype(shape.into().to_vec(), value.to_f64(), B::FLOAT))
    }


    pub fn ones_like(&self) -> Self {
        wrap(self.p.ones_like())
    }





    pub fn into_scalar(self) -> f32 {
        self.p.into_scalar()
    }




    pub fn swap_dims(self, a: usize, b: usize) -> Self {
        wrap(self.p.swap_dims(a, b))
    }

    pub fn transpose(self) -> Self {
        wrap(self.p.transpose())
    }


    pub fn slice_assign(self, ranges: [Range<usize>; D], value: Self) -> Self {
        wrap(self.p.slice_assign(ranges, value.p))
    }


    pub fn matmul(self, rhs: Self) -> Self {
        wrap(self.p.matmul(rhs.p))
    }

    pub fn add(self, rhs: Self) -> Self {
        wrap(self.p.add(rhs.p))
    }

    pub fn sub(self, rhs: Self) -> Self {
        wrap(self.p.sub(rhs.p))
    }

    pub fn mul(self, rhs: Self) -> Self {
        wrap(self.p.mul(rhs.p))
    }

    pub fn div(self, rhs: Self) -> Self {
        wrap(self.p.div(rhs.p))
    }

    pub fn add_scalar<E: Scalar>(self, s: E) -> Self {
        wrap(self.p.add_scalar(s.to_f64()))
    }

    pub fn sub_scalar<E: Scalar>(self, s: E) -> Self {
        wrap(self.p.sub_scalar(s.to_f64()))
    }

    pub fn mul_scalar<E: Scalar>(self, s: E) -> Self {
        wrap(self.p.mul_scalar(s.to_f64()))
    }

    pub fn div_scalar<E: Scalar>(self, s: E) -> Self {
        wrap(self.p.div_scalar(s.to_f64()))
    }

    pub fn neg(self) -> Self {
        wrap(self.p.neg())
    }

    pub fn exp(self) -> Self {
        wrap(self.p.exp())
    }

    pub fn log(self) -> Self {
        wrap(self.p.log())
    }

    pub fn sqrt(self) -> Self {
        wrap(self.p.sqrt())
    }

    pub fn recip(self) -> Self {
        wrap(self.p.recip())
    }

    pub fn abs(self) -> Self {
        wrap(self.p.abs())
    }

    pub fn powf_scalar<E: Scalar>(self, p: E) -> Self {
        wrap(self.p.powf_scalar(p.to_f64()))
    }

    pub fn clamp_min<E: Scalar>(self, min: E) -> Self {
        wrap(self.p.clamp_min(min.to_f64()))
    }

    pub fn clamp_max<E: Scalar>(self, max: E) -> Self {
        wrap(self.p.clamp_max(max.to_f64()))
    }

    pub fn clamp<E: Scalar>(self, min: E, max: E) -> Self {
        wrap(self.p.clamp(min.to_f64(), max.to_f64()))
    }

    pub fn sum(self) -> Tensor<B, 1> {
        wrap(self.p.sum())
    }

    pub fn mean(self) -> Tensor<B, 1> {
        wrap(self.p.mean())
    }

    pub fn max(self) -> Tensor<B, 1> {
        wrap(self.p.max())
    }

    pub fn sum_dim(self, dim: usize) -> Self {
        wrap(self.p.sum_dim(dim))
    }

    pub fn mean_dim(self, dim: usize) -> Self {
        wrap(self.p.mean_dim(dim))
    }

    pub fn max_dim(self, dim: usize) -> Self {
        wrap(self.p.max_dim(dim))
    }

    pub fn argmax(self, dim: usize) -> Tensor<B, D, Int> {
        wrap(self.p.argmax(dim))
    }

    pub fn mask_fill<E: Scalar>(self, mask: Tensor<B, D, Bool>, value: E) -> Self {
        wrap(self.p.mask_fill(mask.p, value.to_f64()))
    }



    pub fn topk_with_indices(self, k: usize, dim: usize) -> (Self, Tensor<B, D, Int>) {
        let (v, i) = self.p.topk_with_indices(k, dim);
        (wrap(v), wrap(i))
    }

    pub fn greater_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.greater_elem(s.to_f64()))
    }

    pub fn greater_equal_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.greater_equal_elem(s.to_f64()))
    }

    pub fn lower_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.lower_elem(s.to_f64()))
    }

    pub fn lower_equal_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.lower_equal_elem(s.to_f64()))
    }

    pub fn equal_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.equal_elem(s.to_f64()))
    }

    pub fn detach(self) -> Self {
        wrap(self.p.detach())
    }

    /// Mark as requiring gradients; a no-op on the untracked backend (as in burn).
    pub fn require_grad(self) -> Self {
        if B::AUTODIFF { wrap(self.p.require_grad()) } else { self }
    }

    pub fn is_require_grad(&self) -> bool {
        self.p.is_require_grad()
    }
}

impl<B: AutodiffBackend, const D: usize> Tensor<B, D> {
    /// Lift an inner tensor (untracked, order 0).
    pub fn from_inner(t: Tensor<B::InnerBackend, D>) -> Self {
        wrap(t.p.detach())
    }

    /// The values on the inner backend.
    pub fn inner(self) -> Tensor<B::InnerBackend, D> {
        wrap(self.p.untracked())
    }

    pub fn backward(&self) -> RawGradients {
        self.p.backward()
    }

    pub fn grad(&self, grads: &RawGradients) -> Option<Tensor<B::InnerBackend, D>> {
        self.p.grad(grads).map(wrap)
    }
}

// ------------------------------------------------------------------------ int tensors

impl<B: Backend, const D: usize> Tensor<B, D, Int> {
    pub fn from_ints<T: IntInput>(data: T, _device: &B::Device) -> Self {
        let (v, s) = data.flat();
        check_rank::<D>(&s);
        wrap(IntTensor::from_data(v, s))
    }














    pub fn remainder_scalar<E: Scalar>(self, r: E) -> Self {
        wrap(self.p.remainder_scalar(r.to_i64()))
    }

    pub fn div_scalar<E: Scalar>(self, d: E) -> Self {
        wrap(self.p.div_scalar(d.to_i64()))
    }

    pub fn mul_scalar<E: Scalar>(self, m: E) -> Self {
        wrap(self.p.mul_scalar(m.to_i64()))
    }

    pub fn add_scalar<E: Scalar>(self, a: E) -> Self {
        wrap(self.p.add_scalar(a.to_i64()))
    }

    pub fn add(self, rhs: Self) -> Self {
        wrap(self.p.add(rhs.p))
    }

    pub fn one_hot<const D2: usize>(self, n: usize) -> Tensor<B, D2, Int> {
        wrap(self.p.one_hot(n))
    }

    pub fn float(self) -> Tensor<B, D> {
        wrap(self.p.float_dtype(B::FLOAT))
    }

    pub fn equal_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.equal_elem(s.to_i64()))
    }

    pub fn not_equal_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.not_equal_elem(s.to_i64()))
    }

    pub fn greater_equal_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.greater_equal_elem(s.to_i64()))
    }

    pub fn greater_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.greater_elem(s.to_i64()))
    }

    pub fn lower_elem<E: Scalar>(self, s: E) -> Tensor<B, D, Bool> {
        wrap(self.p.lower_elem(s.to_i64()))
    }
}

impl<B: Backend> Tensor<B, 1, Int> {
    pub fn arange(r: Range<i64>, _device: &B::Device) -> Self {
        wrap(IntTensor::arange(r))
    }
}

impl<B: AutodiffBackend, const D: usize> Tensor<B, D, Int> {
    pub fn from_inner(t: Tensor<B::InnerBackend, D, Int>) -> Self {
        wrap(t.p)
    }

    pub fn inner(self) -> Tensor<B::InnerBackend, D, Int> {
        wrap(self.p)
    }
}

// ------------------------------------------------------------------------ bool tensors

impl<B: Backend, const D: usize> Tensor<B, D, Bool> {





    pub fn bool_not(self) -> Self {
        wrap(self.p.bool_not())
    }

    pub fn float(self) -> Tensor<B, D> {
        wrap(self.p.float_dtype(B::FLOAT))
    }
}

impl<B: Backend> Tensor<B, 2, Bool> {
    pub fn tril_mask(shape: [usize; 2], offset: i64, _device: &B::Device) -> Self {
        wrap(BoolTensor::tril_mask(shape, offset))
    }
}

// ------------------------------------------------------------------------ operators

impl<B: Backend, const D: usize> std::ops::Add for Tensor<B, D> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Tensor::<B, D>::add(self, rhs)
    }
}
impl<B: Backend, const D: usize> std::ops::Sub for Tensor<B, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Tensor::<B, D>::sub(self, rhs)
    }
}
impl<B: Backend, const D: usize> std::ops::Mul for Tensor<B, D> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Tensor::<B, D>::mul(self, rhs)
    }
}
impl<B: Backend, const D: usize> std::ops::Div for Tensor<B, D> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Tensor::<B, D>::div(self, rhs)
    }
}
impl<B: Backend, const D: usize> std::ops::Neg for Tensor<B, D> {
    type Output = Self;
    fn neg(self) -> Self {
        Tensor::<B, D>::neg(self)
    }
}
impl<B: Backend, const D: usize, E: Scalar> std::ops::Add<E> for Tensor<B, D> {
    type Output = Self;
    fn add(self, s: E) -> Self {
        self.add_scalar(s)
    }
}
impl<B: Backend, const D: usize, E: Scalar> std::ops::Sub<E> for Tensor<B, D> {
    type Output = Self;
    fn sub(self, s: E) -> Self {
        self.sub_scalar(s)
    }
}
impl<B: Backend, const D: usize, E: Scalar> std::ops::Mul<E> for Tensor<B, D> {
    type Output = Self;
    fn mul(self, s: E) -> Self {
        self.mul_scalar(s)
    }
}
impl<B: Backend, const D: usize, E: Scalar> std::ops::Div<E> for Tensor<B, D> {
    type Output = Self;
    fn div(self, s: E) -> Self {
        self.div_scalar(s)
    }
}
impl<B: Backend, const D: usize> std::ops::Add for Tensor<B, D, Int> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Tensor::<B, D, Int>::add(self, rhs)
    }
}

// ------------------------------------------------------------------------ burn's module paths

/// `burn::tensor::activation`.
pub mod activation {
    use super::{wrap, Backend, Tensor};
    pub fn softmax<B: Backend, const D: usize>(t: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
        wrap(crate::tensor::softmax(t.p, dim))
    }
    pub fn log_softmax<B: Backend, const D: usize>(t: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
        wrap(crate::tensor::log_softmax(t.p, dim))
    }
    pub fn sigmoid<B: Backend, const D: usize>(t: Tensor<B, D>) -> Tensor<B, D> {
        wrap(crate::tensor::sigmoid(t.p))
    }
    pub fn silu<B: Backend, const D: usize>(t: Tensor<B, D>) -> Tensor<B, D> {
        wrap(crate::tensor::silu(t.p))
    }
}

/// `burn::backend`.
pub mod backend {
    pub use super::{Autodiff, CpuF32, CpuF32Grad, NdArray};
}

/// `burn::tensor`.
pub mod tensor {
    pub use super::{activation, Bool, Device, Float, Int, Tensor, TensorData};
    /// `burn::tensor::backend`.
    pub mod backend {
        pub use super::super::{AutodiffBackend, Backend};
    }
}

/// `burn::prelude`.
pub mod prelude {
    pub use super::{Backend, Bool, Cpu, CpuF32, CpuF32Grad, Device, Float, Int, Tensor, TensorData};
}
