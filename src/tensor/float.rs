//! The float tensor: storage, a strided layout, and the autodiff bookkeeping (`order`, and a node
//! when the tensor is tracked). One type serves untracked and tracked (after `require_grad`).
//!
//! The methods began as one burn 0.21 call each on `Autodiff<NdArray<f32>>`, with burn's
//! decomposition and backward rules. The backward rules now
//! are the standard ones (PyTorch's formulas: `g / r` and `−g·((l / r) / r)` for div, `g / x` for
//! log, `g / (2y)` and `−g·y²` from the saved output for sqrt and recip, `g / n` for the means,
//! `g / s` for div_scalar), every op computes in the tensor's dtype, and a broadcast matmul
//! operand's gradient sums the per-batch products.

use super::autodiff::{record_storage, record_view, record_with_output, run_backward, BackwardFn, Gradients, Node};
use super::kernels as k;
use super::shape::numel;
use super::{BoolTensor, IntTensor};
use std::ops::Range;
use super::error::{ok, Result};
use super::infer;
use super::layout::Layout;
use super::storage::Storage;
use super::dtype::{Element, FloatElem};
use super::backend::{with_backend, BackendStorage};
use super::device::Device;
use super::DType;
use super::gpu::{self, BinaryOp, CmpOp, ReduceOp, UnaryOp};
use std::sync::Arc;

#[derive(Clone)]
pub struct Tensor {
    pub(crate) storage: Arc<Storage>,
    pub(crate) layout: Layout,
    pub(crate) order: usize,
    pub(crate) node: Option<Arc<Node>>,
    /// The device (and so the kernel set) this tensor's operations run on.
    pub(crate) device: Device,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor{:?}{}", self.layout.shape, if self.node.is_some() { " (tracked)" } else { "" })
    }
}

/// Run `$body` with `$T` the float type the tensor computes in: f64 for F64 storage, f32 for
/// F32 (and for F16/BF16, which are stored and converted but computed in f32).
macro_rules! fdisp {
    ($dt:expr, $T:ident => $body:expr) => {
        match compute_dtype($dt) {
            DType::F64 => {
                type $T = f64;
                $body
            }
            _ => {
                type $T = f32;
                $body
            }
        }
    };
}

#[track_caller]
pub(crate) fn compute_dtype(d: DType) -> DType {
    match d {
        DType::F64 => DType::F64,
        DType::F32 | DType::F16 | DType::BF16 => DType::F32,
        other => ok(Err(super::TensorError::DType(format!("a float operation on {} storage", other.name())))),
    }
}

fn boxed(f: impl Fn(Tensor, &[bool]) -> Vec<Option<Tensor>> + Send + Sync + 'static) -> BackwardFn {
    Box::new(f)
}

/// burn-autodiff's `broadcast_shape`: sum the gradient over the dimensions the forward broadcast.
fn unbroadcast(grad: Tensor, shape: &[usize]) -> Tensor {
    let gs = grad.layout.shape.clone();
    let mut grad = grad;
    for i in 0..gs.len() {
        if gs[i] != shape[i] {
            assert_eq!(shape[i], 1, "invalid broadcast in backward");
            grad = grad.sum_dim(i);
        }
    }
    grad
}

impl Tensor {
    /// The elements in row-major order.
    pub(crate) fn data(&self) -> std::borrow::Cow<'_, [f32]> {
        self.vals::<f32>()
    }

    /// The elements as `T` in row-major order: borrowed when the storage holds `T` in one block;
    /// otherwise materialised and, for another float dtype, converted (f16/bf16 exactly to f32).
    pub(crate) fn vals<T: FloatElem>(&self) -> std::borrow::Cow<'_, [T]> {
        match self.storage.try_slice::<T>() {
            Some(s) if self.layout.is_contiguous() => std::borrow::Cow::Borrowed(&s[self.layout.offset..self.layout.offset + self.layout.numel()]),
            Some(s) => std::borrow::Cow::Owned(k::materialize(s, &self.layout)),
            None => std::borrow::Cow::Owned(k::materialize(&self.storage.float_vec::<T>(), &self.layout)),
        }
    }

    /// The values as f64 (exact for every float dtype).
    pub fn to_vec_f64(&self) -> Vec<f64> {
        match self.dtype() {
            DType::F64 => self.vals::<f64>().into_owned(),
            _ => self.vals::<f32>().iter().map(|x| *x as f64).collect(),
        }
    }

    /// Convert to another float dtype (burn `cast`): f32 ↔ f64 by `as`, to f16/bf16 by
    /// round-to-nearest-even. The gradient is cast back to the input's dtype.
    #[track_caller]
    pub fn cast(self, to: DType) -> Tensor {
        let from = self.dtype();
        if from == to {
            return self;
        }
        let storage = match gpu::gpu_cast(&self, to) {
            Some(st) => st,
            None => self.cast_host(to),
        };
        let shape = self.layout.shape.clone();
        super::autodiff::record_storage(storage, shape, &[&self], move || boxed(move |g, _| vec![Some(g.cast(from))]))
    }

    fn cast_host(&self, to: DType) -> Storage {
        let v = self.to_vec_f64();
        match to {
            DType::F32 => Storage::from_vec(v.iter().map(|x| *x as f32).collect::<Vec<f32>>()),
            DType::F64 => Storage::from_vec(v),
            DType::F16 => Storage::from_vec(v.iter().map(|x| super::half::F16::from_f32(*x as f32)).collect::<Vec<_>>()),
            DType::BF16 => Storage::from_vec(v.iter().map(|x| super::half::BF16::from_f32(*x as f32)).collect::<Vec<_>>()),
            other => ok(Err(super::TensorError::DType(format!("cast of a float tensor to {}", other.name())))),
        }
    }

    /// Whether this tensor is a view that a kernel would have to copy.
    pub fn is_view(&self) -> bool {
        !self.layout.is_contiguous()
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// The device this tensor's operations run on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The same values on `device`: CPU modes share the storage; a GPU device gets a copy
    /// (uploaded, or downloaded back to the CPU), recorded on the tape when the tensor is
    /// tracked (the gradient flows back to the source device).
    #[track_caller]
    pub fn to(self, device: Device) -> Tensor {
        ok(self.try_to(device))
    }

    pub fn try_to(self, device: Device) -> Result<Tensor> {
        super::device::check(device)?;
        self.transfer(device)
    }

    /// Move to `device`. A copy between the CPU and a GPU is an operation on the tape: a
    /// tracked tensor's copy gets a node whose backward moves the gradient back to the source
    /// device. CPU modes relabel the same storage and keep the identity, as before.
    fn transfer(self, device: Device) -> Result<Tensor> {
        let moved = !gpu::agrees(&self.storage, device);
        let storage = gpu::place(&self.storage, device)?;
        let from = self.device;
        let t = match &self.node {
            Some(_) if moved => {
                let order = self.order + 1;
                let node = Arc::new(Node {
                    id: super::autodiff::next_id(),
                    order,
                    inputs: vec![self.node.clone()],
                    backward: Some(boxed(move |g, _| vec![Some(ok(g.transfer(from)))])),
                });
                Tensor { storage, layout: self.layout, order, node: Some(node), device }
            }
            _ => Tensor { storage, device, ..self },
        };
        debug_assert!(gpu::agrees(&t.storage, t.device), "storage and device disagree");
        Ok(t)
    }

    fn be_map<T: Copy + Sync, U: Send, F: Fn(T) -> U + Sync>(&self, x: &[T], f: F) -> Vec<U> {
        with_backend!(self.device, B => B::map(x, f))
    }

    fn be_zip<T: Copy + Sync, U: Copy + Sync, V: Send, F: Fn(T, U) -> V + Sync>(&self, a: &[T], ash: &[usize], b: &[U], bsh: &[usize], f: F) -> (Vec<V>, Vec<usize>) {
        with_backend!(self.device, B => B::zip(a, ash, b, bsh, f))
    }

    fn be_matmul<T: FloatElem>(&self, a: &[T], ash: &[usize], b: &[T], bsh: &[usize]) -> (Vec<T>, Vec<usize>) {
        with_backend!(self.device, B => B::matmul(a, ash, b, bsh))
    }

    fn be_sum_dim<T: FloatElem>(&self, x: &[T], sh: &[usize], dim: usize) -> (Vec<T>, Vec<usize>) {
        with_backend!(self.device, B => B::sum_dim(x, sh, dim))
    }

    fn be_sum_all<T: FloatElem>(&self, x: &[T]) -> T {
        with_backend!(self.device, B => B::sum_all(x))
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    // ---------------------------------------------------------------- construction and readback

    /// A fresh (order 0, untracked) tensor.
    pub(crate) fn raw(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::raw_t(data, shape)
    }

    pub(crate) fn raw_t<T: Element>(data: Vec<T>, shape: Vec<usize>) -> Tensor {
        assert_eq!(data.len(), numel(&shape), "data length {} does not match shape {:?}", data.len(), shape);
        let t = Tensor { storage: Arc::new(Storage::from_vec(data)), layout: Layout::contiguous(shape), order: 0, node: None, device: Device::default() };
        t.on(super::device::default_device())
    }

    /// A fresh (order 0, untracked) tensor over `storage` on `device` (they must agree).
    pub(crate) fn fresh(storage: Storage, shape: Vec<usize>, device: Device) -> Tensor {
        debug_assert!(gpu::agrees(&storage, device), "storage and device disagree");
        Tensor { storage: Arc::new(storage), layout: Layout::contiguous(shape), order: 0, node: None, device }
    }

    /// A constant tensor of `dtype` (f32 or f64).
    pub fn full_dtype(shape: impl Into<Vec<usize>>, value: f64, dtype: DType) -> Tensor {
        let shape = shape.into();
        fdisp!(dtype, T => Tensor::raw_t(vec![T::from_f64(value); numel(&shape)], shape))
    }

    /// A constant tensor with `like`'s dtype and device.
    pub(crate) fn full_like(shape: Vec<usize>, value: f64, like: &Tensor) -> Tensor {
        // A GPU tensor's constants are written on the device; f16/bf16 compute in f32, as
        // every GPU op returns f32.
        if gpu::gpu_of(&like.storage).is_some() {
            if let Some(t) = gpu::gpu_full(like.device, shape.clone(), value) {
                return t;
            }
        }
        Tensor::full_dtype(shape, value, like.dtype()).on(like.device)
    }

    /// Move a fresh tensor to `device` (no check: it follows an existing tensor's device); a GPU
    /// device gets the storage uploaded.
    #[track_caller]
    pub(crate) fn on(self, device: Device) -> Tensor {
        ok(self.transfer(device))
    }

    /// Row-major values of `dtype` given as f64 (f32 values convert by `as`).
    pub fn from_f64s(values: Vec<f64>, shape: impl Into<Vec<usize>>, dtype: DType) -> Tensor {
        let shape = shape.into();
        fdisp!(dtype, T => Tensor::raw_t(values.iter().map(|x| T::from_f64(*x)).collect::<Vec<T>>(), shape))
    }

    pub(crate) fn ones_shape(shape: Vec<usize>) -> Tensor {
        Tensor::raw(vec![1.0; numel(&shape)], shape)
    }

    /// A rank-1 tensor of the values (burn `from_floats`).
    pub fn from_floats(values: &[f32]) -> Tensor {
        Tensor::raw(values.to_vec(), vec![values.len()])
    }

    /// A tensor of the given shape from row-major values.
    pub fn from_data(values: Vec<f32>, shape: impl Into<Vec<usize>>) -> Tensor {
        Tensor::raw(values, shape.into())
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Tensor {
        let shape = shape.into();
        Tensor::raw(vec![0.0; numel(&shape)], shape)
    }

    pub fn ones(shape: impl Into<Vec<usize>>) -> Tensor {
        Tensor::ones_shape(shape.into())
    }

    pub fn full(shape: impl Into<Vec<usize>>, value: f64) -> Tensor {
        let shape = shape.into();
        Tensor::raw(vec![value as f32; numel(&shape)], shape)
    }

    pub fn zeros_like(&self) -> Tensor {
        Tensor::full_like(self.layout.shape.clone(), 0.0, self)
    }

    pub fn ones_like(&self) -> Tensor {
        Tensor::full_like(self.layout.shape.clone(), 1.0, self)
    }

    /// The dimensions as an array (`let [s, b, t] = x.dims();`); panics on a rank mismatch.
    pub fn dims<const D: usize>(&self) -> [usize; D] {
        self.layout.shape.as_slice().try_into().unwrap_or_else(|_| panic!("rank {} tensor read as rank {D}", self.layout.shape.len()))
    }

    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    pub fn rank(&self) -> usize {
        self.layout.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// The values in row-major order (burn `into_data().to_vec()`).
    pub fn to_vec(&self) -> Vec<f32> {
        self.data().to_vec()
    }

    /// The values in row-major order (borrowed when the layout is one block, else copied).
    pub fn as_slice(&self) -> std::borrow::Cow<'_, [f32]> {
        self.data()
    }

    /// The single value of a one-element tensor.
    #[track_caller]
    pub fn into_scalar(&self) -> f32 {
        ok(infer::scalar(&self.layout.shape));
        assert_eq!(self.layout.numel(), 1, "into_scalar on a tensor of shape {:?}", self.layout.shape);
        self.to_vec_f64()[0] as f32
    }

    // ---------------------------------------------------------------- autodiff

    /// Mark a leaf as requiring gradients (burn `require_grad`: order 0, same identity).
    pub fn require_grad(mut self) -> Tensor {
        match &self.node {
            Some(n) if n.backward.is_none() => self,
            Some(_) => panic!("Can't convert a non leaf tensor into a tracked tensor"),
            None => {
                self.node = Some(Node::leaf());
                self.order = 0;
                self
            }
        }
    }

    /// Stop the gradient: the same values as a fresh, untracked, order-0 tensor (the
    /// standard `detach`; burn 0.21 kept a `require_grad` leaf a leaf, as a new one).
    pub fn detach(self) -> Tensor {
        Tensor { storage: self.storage, layout: self.layout, order: 0, node: None, device: self.device }
    }

    pub fn is_tracked(&self) -> bool {
        self.node.is_some()
    }

    pub fn is_require_grad(&self) -> bool {
        matches!(&self.node, Some(n) if n.backward.is_none())
    }

    /// Reverse-mode gradients of this tensor (normally a scalar loss) with respect to every
    /// `require_grad` leaf it depends on.
    pub fn backward(&self) -> Gradients {
        run_backward(self)
    }

    /// This leaf's gradient in `grads` (burn `grad`), untracked.
    pub fn grad(&self, grads: &Gradients) -> Option<Tensor> {
        grads.get(self)
    }

    // ---------------------------------------------------------------- elementwise, binary

    /// Gradient accumulation (`B::float_add` on inner tensors), no bookkeeping.
    pub(crate) fn add_raw(&self, other: &Tensor) -> Tensor {
        if let Some((st, sh)) = gpu::gpu_binary(self, other, BinaryOp::Add) {
            return Tensor::fresh(st, sh, self.device);
        }
        fdisp!(self.dtype(), T => {
            let (v, sh) = self.be_zip(&self.vals::<T>(), &self.layout.shape, &other.vals::<T>(), &other.layout.shape, |a, b| a + b);
            Tensor::raw_t(v, sh).on(self.device)
        })
    }

    #[track_caller]
    pub fn add(self, rhs: Tensor) -> Tensor {
        ok(infer::broadcast(&self.layout.shape, &rhs.layout.shape).map(drop));
        ok(infer::same_device(self.device, rhs.device));
        ok(infer::same_dtype(self.dtype(), rhs.dtype(), "add"));
        let st = match gpu::gpu_binary(&self, &rhs, BinaryOp::Add) {
            Some(r) => r,
            None => fdisp!(self.dtype(), T => {
                let (v, sh) = self.be_zip(&self.vals::<T>(), &self.layout.shape, &rhs.vals::<T>(), &rhs.layout.shape, |a, b| a + b);
                (Storage::from_vec(v), sh)
            }),
        };
        let (ls, rs) = (self.layout.shape.clone(), rhs.layout.shape.clone());
        record_storage(st.0, st.1, &[&self, &rhs], move || boxed(move |g, n| vec![n[0].then(|| unbroadcast(g.clone(), &ls)), n[1].then(|| unbroadcast(g, &rs))]))
    }

    #[track_caller]
    pub fn sub(self, rhs: Tensor) -> Tensor {
        ok(infer::broadcast(&self.layout.shape, &rhs.layout.shape).map(drop));
        ok(infer::same_device(self.device, rhs.device));
        ok(infer::same_dtype(self.dtype(), rhs.dtype(), "sub"));
        let st = match gpu::gpu_binary(&self, &rhs, BinaryOp::Sub) {
            Some(r) => r,
            None => fdisp!(self.dtype(), T => {
                let (v, sh) = self.be_zip(&self.vals::<T>(), &self.layout.shape, &rhs.vals::<T>(), &rhs.layout.shape, |a, b| a - b);
                (Storage::from_vec(v), sh)
            }),
        };
        let (ls, rs) = (self.layout.shape.clone(), rhs.layout.shape.clone());
        record_storage(st.0, st.1, &[&self, &rhs], move || boxed(move |g, n| vec![n[0].then(|| unbroadcast(g.clone(), &ls)), n[1].then(|| unbroadcast(g.neg(), &rs))]))
    }

    #[track_caller]
    pub fn mul(self, rhs: Tensor) -> Tensor {
        ok(infer::broadcast(&self.layout.shape, &rhs.layout.shape).map(drop));
        ok(infer::same_device(self.device, rhs.device));
        ok(infer::same_dtype(self.dtype(), rhs.dtype(), "mul"));
        let st = match gpu::gpu_binary(&self, &rhs, BinaryOp::Mul) {
            Some(r) => r,
            None => fdisp!(self.dtype(), T => {
                let (v, sh) = self.be_zip(&self.vals::<T>(), &self.layout.shape, &rhs.vals::<T>(), &rhs.layout.shape, |a, b| a * b);
                (Storage::from_vec(v), sh)
            }),
        };
        let (ls, rs) = (self.layout.shape.clone(), rhs.layout.shape.clone());
        let (l, r) = (self.clone().untracked(), rhs.clone().untracked());
        record_storage(st.0, st.1, &[&self, &rhs], move || {
            boxed(move |g, n| vec![n[0].then(|| unbroadcast(g.clone().mul(r.clone()), &ls)), n[1].then(|| unbroadcast(g.mul(l.clone()), &rs))])
        })
    }

    #[track_caller]
    pub fn div(self, rhs: Tensor) -> Tensor {
        ok(infer::broadcast(&self.layout.shape, &rhs.layout.shape).map(drop));
        ok(infer::same_device(self.device, rhs.device));
        ok(infer::same_dtype(self.dtype(), rhs.dtype(), "div"));
        let st = match gpu::gpu_binary(&self, &rhs, BinaryOp::Div) {
            Some(r) => r,
            None => fdisp!(self.dtype(), T => {
                let (v, sh) = self.be_zip(&self.vals::<T>(), &self.layout.shape, &rhs.vals::<T>(), &rhs.layout.shape, |a, b| a / b);
                (Storage::from_vec(v), sh)
            }),
        };
        let (ls, rs) = (self.layout.shape.clone(), rhs.layout.shape.clone());
        let (l, r) = (self.clone().untracked(), rhs.clone().untracked());
        record_storage(st.0, st.1, &[&self, &rhs], move || {
            boxed(move |g, n| {
                // PyTorch's: g / r, and −g · ((l / r) / r).
                vec![
                    n[0].then(|| unbroadcast(g.clone().div(r.clone()), &ls)),
                    n[1].then(|| unbroadcast(g.neg().mul(l.clone().div(r.clone()).div(r.clone())), &rs)),
                ]
            })
        })
    }

    // ---------------------------------------------------------------- elementwise, scalar

    pub fn add_scalar(self, s: impl Into<f64>) -> Tensor {
        let s = s.into();
        let st = gpu::gpu_unary(&self, UnaryOp::AddScalar(s as f32)).unwrap_or_else(|| fdisp!(self.dtype(), T => { let s = T::from_f64(s); Storage::from_vec(self.be_map(&self.vals::<T>(), |a| a + s)) }));
        record_storage(st, self.layout.shape.clone(), &[&self], || boxed(|g, _| vec![Some(g)]))
    }

    pub fn sub_scalar(self, s: impl Into<f64>) -> Tensor {
        let s = s.into();
        let st = gpu::gpu_unary(&self, UnaryOp::SubScalar(s as f32)).unwrap_or_else(|| fdisp!(self.dtype(), T => { let s = T::from_f64(s); Storage::from_vec(self.be_map(&self.vals::<T>(), |a| a - s)) }));
        record_storage(st, self.layout.shape.clone(), &[&self], || boxed(|g, _| vec![Some(g)]))
    }

    pub fn mul_scalar(self, s: impl Into<f64>) -> Tensor {
        let s64 = s.into();
        let st = gpu::gpu_unary(&self, UnaryOp::MulScalar(s64 as f32)).unwrap_or_else(|| fdisp!(self.dtype(), T => { let s = T::from_f64(s64); Storage::from_vec(self.be_map(&self.vals::<T>(), |a| a * s)) }));
        record_storage(st, self.layout.shape.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.mul_scalar(s64))]))
    }

    pub fn div_scalar(self, s: impl Into<f64>) -> Tensor {
        let s64 = s.into();
        let st = gpu::gpu_unary(&self, UnaryOp::DivScalar(s64 as f32)).unwrap_or_else(|| fdisp!(self.dtype(), T => { let s = T::from_f64(s64); Storage::from_vec(self.be_map(&self.vals::<T>(), |a| a / s)) }));
        // g / s in the tensor's dtype.
        record_storage(st, self.layout.shape.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.div_scalar(s64))]))
    }

    // ---------------------------------------------------------------- elementwise, unary

    pub fn neg(self) -> Tensor {
        record_storage(gpu::gpu_unary(&self, UnaryOp::Neg).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| -a)))), self.layout.shape.clone(), &[&self], || boxed(|g, _| vec![Some(g.neg())]))
    }

    /// exp; the backward reads the saved output, `g · exp(x)`.
    pub fn exp(self) -> Tensor {
        record_with_output(gpu::gpu_unary(&self, UnaryOp::Exp).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| a.exp())))), self.layout.shape.clone(), &[&self], move |y| boxed(move |g, _| vec![Some(g.mul(y.clone()))]))
    }

    /// ln; backward `g / x`.
    pub fn log(self) -> Tensor {
        let x = self.clone().untracked();
        record_storage(gpu::gpu_unary(&self, UnaryOp::Log).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| a.ln())))), self.layout.shape.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.div(x.clone()))]))
    }

    /// sqrt; backward `g / (2·y)` from the saved output.
    pub fn sqrt(self) -> Tensor {
        record_with_output(gpu::gpu_unary(&self, UnaryOp::Sqrt).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| a.sqrt())))), self.layout.shape.clone(), &[&self], move |y| {
            boxed(move |g, _| vec![Some(g.div(y.clone().mul_scalar(2.0)))])
        })
    }

    /// 1/x; backward `−g · y²` from the saved output.
    pub fn recip(self) -> Tensor {
        record_with_output(gpu::gpu_unary(&self, UnaryOp::Recip).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| T::ONE / a)))), self.layout.shape.clone(), &[&self], move |y| {
            boxed(move |g, _| vec![Some(g.neg().mul(y.clone().mul(y.clone())))])
        })
    }

    pub fn abs(self) -> Tensor {
        let x = self.clone().untracked();
        record_storage(gpu::gpu_unary(&self, UnaryOp::Abs).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| a.abs())))), self.layout.shape.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.mul(x.clone().sign()))]))
    }

    /// burn-ndarray `sign_op`: 0 for ±0, 1 for positive, −1 otherwise.
    pub fn sign(self) -> Tensor {
        let st = gpu::gpu_unary(&self, UnaryOp::Sign).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| if a == T::ZERO { T::ZERO } else if a.is_sign_positive() { T::ONE } else { -T::ONE }))));
        record_storage(st, self.layout.shape.clone(), &[&self], || boxed(|g, _| vec![Some(g.zeros_like())]))
    }

    /// burn `powf_scalar`: an integer exponent goes through `powi` (2 is `x · x`), any other
    /// through `powf`.
    pub fn powf_scalar(self, p: impl Into<f64>) -> Tensor {
        let p = p.into();
        if p.floor() == p {
            self.powi_scalar(p as i64)
        } else {
            self.powf_scalar_impl(p)
        }
    }

    fn powi_scalar(self, p: i64) -> Tensor {
        match p {
            0 => self.ones_like(),
            1 => self,
            2 => self.clone().mul(self),
            -1 => self.recip(),
            -2 => self.clone().mul(self).recip(),
            _ => self.powf_scalar_impl(p as f64),
        }
    }

    fn powf_scalar_impl(self, p: f64) -> Tensor {
        // The exponent in the tensor's dtype (burn passed it as f32 for every dtype).
        let x = self.clone().untracked();
        record_storage(gpu::gpu_unary(&self, UnaryOp::PowScalar(p as f32)).unwrap_or_else(|| fdisp!(self.dtype(), T => { let pt = T::from_f64(p); Storage::from_vec(self.be_map(&self.vals::<T>(), |a: T| a.powf(pt))) })), self.layout.shape.clone(), &[&self], move || {
            boxed(move |g, _| {
                let tmp = x.clone().powf_scalar(p - 1.0);
                let value = tmp.mul_scalar(p);
                vec![Some(g.mul(value))]
            })
        })
    }

    /// burn `clamp_min` (on the autodiff backend: `mask_fill(x < min, min)`).
    pub fn clamp_min(self, min: impl Into<f64>) -> Tensor {
        let min = min.into();
        let mask = self.clone().lower_elem(min);
        self.mask_fill(mask, min)
    }

    /// burn `clamp_max` (`mask_fill(x > max, max)`).
    pub fn clamp_max(self, max: impl Into<f64>) -> Tensor {
        let max = max.into();
        let mask = self.clone().greater_elem(max);
        self.mask_fill(mask, max)
    }

    /// burn `clamp`: `clamp_min(clamp_max(x, max), min)`.
    pub fn clamp(self, min: impl Into<f64>, max: impl Into<f64>) -> Tensor {
        self.clamp_max(max).clamp_min(min)
    }

    // ---------------------------------------------------------------- comparisons (untracked)

    fn cmp_elem(&self, s: f64, op: CmpOp, f32f: impl Fn(f32, f32) -> bool, f64f: impl Fn(f64, f64) -> bool) -> BoolTensor {
        if let Some(b) = gpu::gpu_compare(self, op, s) {
            return b;
        }
        let v = match compute_dtype(self.dtype()) {
            DType::F64 => k::map(&self.vals::<f64>(), |a| f64f(a, s)),
            _ => {
                let s = s as f32;
                k::map(&self.vals::<f32>(), |a| f32f(a, s))
            }
        };
        BoolTensor::raw(v, self.layout.shape.clone())
    }

    pub fn greater_elem(self, s: impl Into<f64>) -> BoolTensor {
        self.cmp_elem(s.into(), CmpOp::Gt, |a, s| a > s, |a, s| a > s)
    }

    pub fn greater_equal_elem(self, s: impl Into<f64>) -> BoolTensor {
        self.cmp_elem(s.into(), CmpOp::Ge, |a, s| a >= s, |a, s| a >= s)
    }

    pub fn lower_elem(self, s: impl Into<f64>) -> BoolTensor {
        self.cmp_elem(s.into(), CmpOp::Lt, |a, s| a < s, |a, s| a < s)
    }

    pub fn lower_equal_elem(self, s: impl Into<f64>) -> BoolTensor {
        self.cmp_elem(s.into(), CmpOp::Le, |a, s| a <= s, |a, s| a <= s)
    }

    pub fn equal_elem(self, s: impl Into<f64>) -> BoolTensor {
        self.cmp_elem(s.into(), CmpOp::Eq, |a, s| a == s, |a, s| a == s)
    }

    // ---------------------------------------------------------------- reductions

    /// Sum of every element, shape [1].
    pub fn sum(self) -> Tensor {
        let shape = self.layout.shape.clone();
        let st = match gpu::gpu_reduce_all(&self, ReduceOp::Sum) {
            Some(g) => Storage::Gpu(g),
            None => fdisp!(self.dtype(), T => Storage::from_vec(vec![self.be_sum_all(&self.vals::<T>())])),
        };
        record_storage(st, vec![1], &[&self], move || {
            boxed(move |g, _| {
                let val = Tensor::full_like(shape.clone(), 1.0, &g);
                let g = g.reshape_vec(vec![1; shape.len()]);
                vec![Some(val.mul(g))]
            })
        })
    }

    /// Mean of every element, shape [1] (ndarray `mean`: the sum divided by n).
    pub fn mean(self) -> Tensor {
        let shape = self.layout.shape.clone();
        let st = match gpu::gpu_reduce_all(&self, ReduceOp::Sum) {
            // The sum divided by n, as the CPU divides it.
            Some(g) => gpu::gpu_unary(&Tensor::fresh(Storage::Gpu(g), vec![1], self.device), UnaryOp::DivScalar(self.layout.numel() as f32)).expect("a GPU tensor"),
            None => fdisp!(self.dtype(), T => { let x = self.vals::<T>(); let n = T::from_usize(x.len()); Storage::from_vec(vec![self.be_sum_all(&x) / n]) }),
        };
        // Backward: g / n over the input's shape (burn multiplied by 1/n).
        record_storage(st, vec![1], &[&self], move || {
            boxed(move |g, _| {
                let ones = Tensor::full_like(shape.clone(), 1.0, &g);
                let g = g.reshape_vec(vec![1; shape.len()]);
                vec![Some(ones.mul(g).div_scalar(numel(&shape) as f64))]
            })
        })
    }

    #[track_caller]
    pub fn sum_dim(self, dim: usize) -> Tensor {
        ok(infer::dim(&self.layout.shape, dim, "sum_dim"));
        let (st, sh) = gpu::gpu_sum_dim(&self, dim).unwrap_or_else(|| fdisp!(self.dtype(), T => { let (v, sh) = self.be_sum_dim(&self.vals::<T>(), &self.layout.shape, dim); (Storage::from_vec(v), sh) }));
        let shape = self.layout.shape.clone();
        record_storage(st, sh, &[&self], move || {
            boxed(move |g, _| {
                let ones = Tensor::full_like(shape.clone(), 1.0, &g);
                let g = g.sum_dim(dim);
                vec![Some(ones.mul(g))]
            })
        })
    }

    #[track_caller]
    pub fn mean_dim(self, dim: usize) -> Tensor {
        ok(infer::dim(&self.layout.shape, dim, "mean_dim"));
        let gpu_mean = gpu::gpu_sum_dim(&self, dim).map(|(st, sh)| {
            let n = self.layout.shape[dim] as f32;
            (gpu::gpu_unary(&Tensor::fresh(st, sh.clone(), self.device), UnaryOp::DivScalar(n)).expect("a GPU tensor"), sh)
        });
        let (st, sh) = if let Some(r) = gpu_mean { r } else { fdisp!(self.dtype(), T => {
            let (mut v, sh) = self.be_sum_dim(&self.vals::<T>(), &self.layout.shape, dim);
            let n = T::from_usize(self.layout.shape[dim]);
            for x in v.iter_mut() {
                *x = *x / n;
            }
            (Storage::from_vec(v), sh)
        }) };
        let shape = self.layout.shape.clone();
        record_storage(st, sh, &[&self], move || {
            // Backward: g / n along `dim` (burn multiplied by 1/n).
            boxed(move |g, _| {
                let ones = Tensor::full_like(shape.clone(), 1.0, &g);
                let g = g.sum_dim(dim);
                vec![Some(ones.mul(g).div_scalar(shape[dim] as f64))]
            })
        })
    }

    /// Maximum along `dim`, keeping it (argmax, then gather; NaN-propagating, as `argmax`).
    #[track_caller]
    pub fn max_dim(self, dim: usize) -> Tensor {
        ok(infer::dim(&self.layout.shape, dim, "max_dim"));
        if let Some((st, idx, ish)) = gpu::gpu_max_dim(&self, dim) {
            let shape = self.layout.shape.clone();
            return record_storage(st, ish.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.scatter_add_into_device(&shape, dim, &idx, &ish))]));
        }
        let (st, idx, ish) = fdisp!(self.dtype(), T => {
            let x = self.vals::<T>();
            let (idx, ish) = k::argmax_dim(&x, &self.layout.shape, dim);
            (Storage::from_vec(k::gather(&x, &self.layout.shape, dim, &idx, &ish)), idx, ish)
        });
        let shape = self.layout.shape.clone();
        record_storage(st, ish.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.scatter_add_into(&shape, dim, &idx, &ish))]))
    }

    /// Maximum of every element (untracked; tests and host checks).
    pub fn max(self) -> Tensor {
        if let Some(g) = gpu::gpu_reduce_all(&self, ReduceOp::Max) {
            return Tensor::fresh(Storage::Gpu(g), vec![1], self.device);
        }
        fdisp!(self.dtype(), T => {
            // NaN-propagating: any NaN makes the maximum NaN.
            let m = self.vals::<T>().iter().copied().reduce(|a, b| if a != a || a > b { a } else { b }).expect("max of an empty tensor");
            Tensor::raw_t(vec![m], vec![1]).on(self.device)
        })
    }

    #[track_caller]
    pub fn argmax(self, dim: usize) -> IntTensor {
        ok(infer::dim(&self.layout.shape, dim, "argmax"));
        let (v, sh) = gpu::gpu_argmax(&self, dim).unwrap_or_else(|| fdisp!(self.dtype(), T => k::argmax_dim(&self.vals::<T>(), &self.layout.shape, dim)));
        IntTensor::raw(v, sh)
    }

    // ---------------------------------------------------------------- layout

    /// The same values without autodiff tracking (no copy).
    pub fn untracked(self) -> Tensor {
        Tensor { storage: self.storage, layout: self.layout, order: 0, node: None, device: self.device }
    }

    #[track_caller]
    pub fn reshape_dyn(self, shape: Vec<usize>) -> Tensor {
        ok(infer::reshape(&self.layout.shape, &shape));
        self.reshape_vec(shape)
    }

    #[track_caller]
    fn reshape_vec(self, shape: Vec<usize>) -> Tensor {
        ok(infer::reshape(&self.layout.shape, &shape));
        let original = self.layout.shape.clone();
        let out = shape.clone();
        // A row-major layout reshapes in place; any other view is copied first.
        let (storage, layout) = match self.layout.reshaped(shape.clone()) {
            Some(l) => (self.storage.clone(), l),
            None => (Arc::new(gpu::gpu_copy(&self).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(self.vals::<T>().into_owned())))), Layout::contiguous(shape.clone())),
        };
        let order = self.order + 1;
        let node = self.node.as_ref().map(|_| {
            Arc::new(Node {
                id: super::autodiff::next_id(),
                order,
                inputs: vec![self.node.clone()],
                backward: Some(boxed(move |g, _| {
                    let gs = g.layout.shape.clone();
                    let mut g = g;
                    for i in 0..out.len() {
                        if out[i] == 1 && gs[i] != 1 {
                            g = g.sum_dim(i);
                        }
                    }
                    vec![Some(g.reshape_vec(original.clone()))]
                })),
            })
        });
        Tensor { storage, layout, order, node, device: self.device }
    }

    /// Reshape; `-1` is not supported (callers give explicit sizes).
    #[track_caller]
    pub fn reshape<const D: usize>(self, shape: [usize; D]) -> Tensor {
        ok(infer::reshape(&self.layout.shape, &shape));
        self.reshape_vec(shape.to_vec())
    }

    /// Insert a size-1 dimension at `dim` (a reshape, as in burn).
    #[track_caller]
    pub fn unsqueeze_dim(self, dim: usize) -> Tensor {
        ok(infer::unsqueeze(&self.layout.shape, dim));
        let mut s = self.layout.shape.clone();
        s.insert(dim, 1);
        self.reshape_vec(s)
    }

    #[track_caller]
    pub fn swap_dims(self, a: usize, b: usize) -> Tensor {
        ok(infer::swap(&self.layout.shape, a, b));
        if a == b {
            return self;
        }
        let layout = self.layout.swapped(a, b);
        record_view(&self, layout, move || boxed(move |g, _| vec![Some(g.swap_dims(b, a))]))
    }

    /// Broadcast to `shape` (size-1 dimensions repeat; leading dimensions may be added).
    #[track_caller]
    pub fn expand<const D: usize>(self, shape: [usize; D]) -> Tensor {
        ok(infer::expand(&self.layout.shape, &shape).map(drop));
        self.expand_dyn(shape.to_vec())
    }

    #[track_caller]
    pub(crate) fn expand_dyn(self, out: Vec<usize>) -> Tensor {
        ok(infer::expand(&self.layout.shape, &out).map(drop));
        let (ni, no) = (self.layout.shape.len(), out.len());
        assert!(no >= ni, "expand to a lower rank");
        let mut aligned = vec![1; no];
        aligned[no - ni..].copy_from_slice(&self.layout.shape);
        let layout = self.layout.broadcast(&out);
        let shape_in = self.layout.shape.clone();
        record_view(&self, layout, move || {
            boxed(move |g, _| {
                let gs = g.layout.shape.clone();
                let mut g = g;
                for i in 0..no {
                    if aligned[i] == 1 && gs[i] != 1 {
                        g = g.sum_dim(i);
                    }
                }
                vec![Some(g.reshape_vec(shape_in.clone()))]
            })
        })
    }

    #[track_caller]
    pub fn slice<const D: usize>(self, ranges: [Range<usize>; D]) -> Tensor {
        ok(infer::slice(&self.layout.shape, &ranges).map(drop));
        self.slice_ranges(ranges.to_vec())
    }

    pub fn slice_dyn(self, ranges: Vec<Range<usize>>) -> Tensor {
        self.slice_ranges(ranges)
    }

    #[track_caller]
    fn slice_ranges(self, ranges: Vec<Range<usize>>) -> Tensor {
        ok(infer::slice(&self.layout.shape, &ranges).map(drop));
        let layout = self.layout.narrowed(&ranges);
        let shape = self.layout.shape.clone();
        record_view(&self, layout, move || boxed(move |g, _| vec![Some(Tensor::full_like(shape.clone(), 0.0, &g).slice_assign_ranges(ranges.clone(), g))]))
    }

    #[track_caller]
    pub fn slice_assign<const D: usize>(self, ranges: [Range<usize>; D], value: Tensor) -> Tensor {
        ok(infer::slice_assign(&self.layout.shape, &ranges, &value.layout.shape));
        self.slice_assign_ranges(ranges.to_vec(), value)
    }

    #[track_caller]
    fn slice_assign_ranges(self, ranges: Vec<Range<usize>>, value: Tensor) -> Tensor {
        ok(infer::slice_assign(&self.layout.shape, &ranges, &value.layout.shape));
        ok(infer::same_dtype(self.dtype(), value.dtype(), "slice_assign"));
        let st = gpu::gpu_slice_assign(&self, &ranges, &value).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(k::slice_assign(&self.vals::<T>(), &self.layout.shape, &ranges, &value.vals::<T>()))));
        let vshape = value.layout.shape.clone();
        record_storage(st, self.layout.shape.clone(), &[&self, &value], move || {
            boxed(move |g, n| {
                vec![
                    n[0].then(|| g.clone().slice_assign_ranges(ranges.clone(), Tensor::full_like(vshape.clone(), 0.0, &g))),
                    n[1].then(|| g.clone().slice_ranges(ranges.clone())),
                ]
            })
        })
    }

    /// Concatenate along `dim` (burn `Tensor::cat`).
    #[track_caller]
    pub fn cat(tensors: Vec<Tensor>, dim: usize) -> Tensor {
        ok(infer::cat(&tensors.iter().map(|t| t.layout.shape.as_slice()).collect::<Vec<_>>(), dim).map(drop));
        for t in &tensors {
            ok(infer::same_dtype(tensors[0].dtype(), t.dtype(), "cat"));
        }
        let (st, sh) = if let Some(r) = gpu::gpu_cat(&tensors, dim) { r } else { fdisp!(tensors[0].dtype(), T => {
            let datas: Vec<std::borrow::Cow<'_, [T]>> = tensors.iter().map(|t| t.vals::<T>()).collect();
            let parts: Vec<(&[T], &[usize])> = datas.iter().zip(&tensors).map(|(d, t)| (&**d, t.layout.shape.as_slice())).collect();
            let (v, sh) = k::cat(&parts, dim);
            (Storage::from_vec(v), sh)
        }) };
        let sizes: Vec<usize> = tensors.iter().map(|t| t.layout.shape[dim]).collect();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        record_storage(st, sh, &refs, move || {
            boxed(move |g, n| {
                let mut off = 0;
                sizes
                    .iter()
                    .zip(n)
                    .map(|(&sz, &need)| {
                        let (start, end) = (off, off + sz);
                        off = end;
                        need.then(|| {
                            let mut r: Vec<Range<usize>> = g.layout.shape.iter().map(|&d| 0..d).collect();
                            r[dim] = start..end;
                            g.clone().slice_ranges(r)
                        })
                    })
                    .collect()
            })
        })
    }

    // ---------------------------------------------------------------- matmul

    /// Batched matmul `[.., m, k] × [.., k, n]`, batch dimensions broadcast when 1; a broadcast
    /// operand's gradient is the sum of the per-batch products (burn's rule that ran
    /// `[.., b, 1, k] × [.., 1, k, n]` as one matmul over the swapped batch is gone; the forward
    /// values are the same, the broadcast operand's gradient sums in another order).
    #[track_caller]
    pub fn matmul(self, rhs: Tensor) -> Tensor {
        ok(infer::matmul(&self.layout.shape, &rhs.layout.shape).map(drop));
        self.matmul_raw(rhs)
    }

    #[track_caller]
    fn matmul_raw(self, rhs: Tensor) -> Tensor {
        ok(infer::matmul(&self.layout.shape, &rhs.layout.shape).map(drop));
        ok(infer::same_device(self.device, rhs.device));
        ok(infer::same_dtype(self.dtype(), rhs.dtype(), "matmul"));
        let (st, sh) = gpu::gpu_matmul(&self, &rhs).unwrap_or_else(|| fdisp!(self.dtype(), T => {
            let (v, sh) = self.be_matmul(&self.vals::<T>(), &self.layout.shape, &rhs.vals::<T>(), &rhs.layout.shape);
            (Storage::from_vec(v), sh)
        }));
        let (ls, rs) = (self.layout.shape.clone(), rhs.layout.shape.clone());
        let (l, r) = (self.clone().untracked(), rhs.clone().untracked());
        let broadcast = ls != rs;
        record_storage(st, sh, &[&self, &rhs], move || {
            boxed(move |g, n| {
                let lhs_grad = n[0].then(|| {
                    let rt = r.clone().transpose();
                    let g = g.clone().matmul_raw(rt);
                    if broadcast { unbroadcast(g, &ls) } else { g }
                });
                let rhs_grad = n[1].then(|| {
                    let lt = l.clone().transpose();
                    let g = lt.matmul_raw(g.clone());
                    if broadcast { unbroadcast(g, &rs) } else { g }
                });
                vec![lhs_grad, rhs_grad]
            })
        })
    }

    /// Swap the last two dimensions.
    pub fn transpose(self) -> Tensor {
        let r = self.rank();
        self.swap_dims(r - 2, r - 1)
    }

    // ---------------------------------------------------------------- indexing

    /// Fill `value` where `mask` is true (the mask broadcasts to this shape).
    #[track_caller]
    pub fn mask_fill(self, mask: BoolTensor, value: impl Into<f64>) -> Tensor {
        ok(infer::mask(&self.layout.shape, &mask.layout.shape));
        let value = value.into();
        assert_eq!(mask.layout.shape.len(), self.layout.shape.len(), "mask_fill: mask rank {:?} vs tensor rank {:?}", mask.layout.shape, self.layout.shape);
        if let Some(st) = gpu::gpu_mask_fill(&self, &mask, value) {
            // The backward mask stays on the device (its layout broadcasts as the forward's).
            let mask = gpu::bool_on(self.device, &mask);
            return record_storage(st, self.layout.shape.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.mask_fill(mask.clone(), 0.0f32))]));
        }
        let m = k::broadcast_to(&mask.data(), &mask.layout.shape, &self.layout.shape);
        let st = fdisp!(self.dtype(), T => {
            let value = T::from_f64(value);
            Storage::from_vec(self.vals::<T>().iter().zip(&m).map(|(&x, &b)| if b { value } else { x }).collect::<Vec<T>>())
        });
        let mask = BoolTensor::raw(m, self.layout.shape.clone());
        record_storage(st, self.layout.shape.clone(), &[&self], move || boxed(move |g, _| vec![Some(g.mask_fill(mask.clone(), 0.0f32))]))
    }

    /// out[p] = self[p with coordinate `dim` replaced by indices[p]].
    #[track_caller]
    pub fn gather(self, dim: usize, indices: IntTensor) -> Tensor {
        ok(infer::index(&self.layout.shape, dim, &indices.layout.shape, &indices.data(), "gather"));
        let st = gpu::gpu_gather(&self, dim, &indices).unwrap_or_else(|| fdisp!(self.dtype(), T => Storage::from_vec(k::gather(&self.vals::<T>(), &self.layout.shape, dim, &indices.data(), &indices.layout.shape))));
        let shape = self.layout.shape.clone();
        record_storage(st, indices.layout.shape.clone(), &[&self], move || {
            boxed(move |g, _| vec![Some(g.scatter_add_into(&shape, dim, &indices.data(), &indices.layout.shape))])
        })
    }

    /// The slices along `dim` named by `indices`, in order.
    #[track_caller]
    pub fn select(self, dim: usize, indices: IntTensor) -> Tensor {
        ok(infer::select(&self.layout.shape, dim, &indices.data(), "select"));
        let (st, sh) = gpu::gpu_select(&self, dim, &indices).unwrap_or_else(|| fdisp!(self.dtype(), T => { let (v, sh) = k::select(&self.vals::<T>(), &self.layout.shape, dim, &indices.data()); (Storage::from_vec(v), sh) }));
        let shape = self.layout.shape.clone();
        record_storage(st, sh, &[&self], move || {
            boxed(move |g, _| {
                if let Some(st) = gpu::gpu_index_add(&g, &shape, dim, &indices.data()) {
                    return vec![Some(Tensor::fresh(st, shape.clone(), g.device))];
                }
                vec![Some(fdisp!(g.dtype(), T => Tensor::raw_t(k::select_add_zeros(&shape, dim, &indices.data(), &g.vals::<T>()), shape.clone()).on(g.device)))]
            })
        })
    }

    /// Sort descending along `dim` (rank ≥ 2) with the source positions; the values carry
    /// gradient back to where they came from.
    #[track_caller]
    pub fn sort_descending_with_indices(self, dim: usize) -> (Tensor, IntTensor) {
        ok(infer::dim(&self.layout.shape, dim, "sort"));
        // A GPU tensor sorts on the device (the indices stay there as I32 until read).
        if let Some(r) = gpu::gpu_sort_desc_tracked(&self, dim) {
            return r;
        }
        let (st, idx) = fdisp!(self.dtype(), T => { let (v, idx) = k::sort_desc_with_indices(&self.vals::<T>(), &self.layout.shape, dim); (Storage::from_vec(v), idx) });
        let shape = self.layout.shape.clone();
        let indices = IntTensor::raw(idx, shape.clone());
        let ind = indices.clone();
        let t = record_storage(st, shape.clone(), &[&self], move || {
            boxed(move |g, _| vec![Some(g.scatter_add_into(&shape, dim, &ind.data(), &ind.layout.shape))])
        });
        (t, indices)
    }

    /// Scatter-add this gradient into zeros of `shape` (gather, max_dim and sort backward).
    fn scatter_add_into(&self, shape: &[usize], dim: usize, idx: &[i64], ish: &[usize]) -> Tensor {
        if let Some(st) = gpu::gpu_scatter_add(self, shape, dim, gpu::Indices::Host(idx), ish) {
            return Tensor::fresh(st, shape.to_vec(), self.device);
        }
        fdisp!(self.dtype(), T => Tensor::raw_t(k::scatter_add_zeros(shape, dim, idx, ish, &self.vals::<T>()), shape.to_vec()).on(self.device))
    }

    /// As `scatter_add_into`, with I32 indices on this tensor's GPU (max_dim backward).
    pub(crate) fn scatter_add_into_device(&self, shape: &[usize], dim: usize, idx: &gpu::GpuStorage, ish: &[usize]) -> Tensor {
        match gpu::gpu_scatter_add(self, shape, dim, gpu::Indices::Device(idx), ish) {
            Some(st) => Tensor::fresh(st, shape.to_vec(), self.device),
            None => {
                // A CPU gradient for a GPU forward: read the indices back.
                let host = match ok(gpu::download(idx)) {
                    super::storage::CpuStorage::I32(v) => v.into_iter().map(|j| j as i64).collect::<Vec<_>>(),
                    _ => unreachable!("argmax indices are I32"),
                };
                self.scatter_add_into(shape, dim, &host, ish)
            }
        }
    }

    /// The k largest values along `dim` and their positions (burn: descending sort, then
    /// `select` of the first k).
    pub fn topk_with_indices(self, kk: usize, dim: usize) -> (Tensor, IntTensor) {
        let first = IntTensor::arange(0..kk as i64);
        let (values, indices) = self.sort_descending_with_indices(dim);
        if gpu::gpu_of(&indices.storage).is_some() {
            // The first k positions of device indices: a view, so nothing is read back.
            ok(infer::select(&indices.layout.shape, dim, &first.data(), "select"));
            let mut r: Vec<Range<usize>> = indices.layout.shape.iter().map(|&d| 0..d).collect();
            r[dim] = 0..kk;
            let view = IntTensor { layout: indices.layout.narrowed(&r), storage: indices.storage };
            return (values.select(dim, first), view);
        }
        (values.select(dim, first.clone()), indices.select(dim, first))
    }
}

// -------------------------------------------------------------------- operators (as burn's)

impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Tensor {
        Tensor::add(self, rhs)
    }
}

impl std::ops::Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Tensor {
        Tensor::sub(self, rhs)
    }
}

impl std::ops::Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Tensor {
        Tensor::mul(self, rhs)
    }
}

impl std::ops::Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Tensor {
        Tensor::div(self, rhs)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

macro_rules! scalar_ops {
    ($($t:ty),*) => {$(
        impl std::ops::Add<$t> for Tensor {
            type Output = Tensor;
            fn add(self, s: $t) -> Tensor { self.add_scalar(s) }
        }
        impl std::ops::Sub<$t> for Tensor {
            type Output = Tensor;
            fn sub(self, s: $t) -> Tensor { self.sub_scalar(s) }
        }
        impl std::ops::Mul<$t> for Tensor {
            type Output = Tensor;
            fn mul(self, s: $t) -> Tensor { self.mul_scalar(s) }
        }
        impl std::ops::Div<$t> for Tensor {
            type Output = Tensor;
            fn div(self, s: $t) -> Tensor { self.div_scalar(s) }
        }
    )*};
}
scalar_ops!(f64);

// -------------------------------------------------------------------- the fallible API

/// `try_*`: the same operations returning a `TensorError` instead of panicking on bad shapes,
/// dtypes or indices; the checks are `infer`'s, so the values are the panicking methods'.
impl Tensor {
    pub fn try_reshape<const D: usize>(self, shape: [usize; D]) -> Result<Tensor> {
        infer::reshape(&self.layout.shape, &shape)?;
        Ok(self.reshape(shape))
    }

    pub fn try_unsqueeze_dim(self, dim: usize) -> Result<Tensor> {
        infer::unsqueeze(&self.layout.shape, dim)?;
        Ok(self.unsqueeze_dim(dim))
    }

    pub fn try_swap_dims(self, a: usize, b: usize) -> Result<Tensor> {
        infer::swap(&self.layout.shape, a, b)?;
        Ok(self.swap_dims(a, b))
    }

    pub fn try_expand<const D: usize>(self, shape: [usize; D]) -> Result<Tensor> {
        infer::expand(&self.layout.shape, &shape)?;
        Ok(self.expand(shape))
    }

    pub fn try_slice<const D: usize>(self, ranges: [Range<usize>; D]) -> Result<Tensor> {
        infer::slice(&self.layout.shape, &ranges)?;
        Ok(self.slice(ranges))
    }

    pub fn try_slice_assign<const D: usize>(self, ranges: [Range<usize>; D], value: Tensor) -> Result<Tensor> {
        infer::slice_assign(&self.layout.shape, &ranges, &value.layout.shape)?;
        Ok(self.slice_assign(ranges, value))
    }

    pub fn try_cat(tensors: Vec<Tensor>, dim: usize) -> Result<Tensor> {
        infer::cat(&tensors.iter().map(|t| t.layout.shape.as_slice()).collect::<Vec<_>>(), dim)?;
        Ok(Tensor::cat(tensors, dim))
    }

    pub fn try_matmul(self, rhs: Tensor) -> Result<Tensor> {
        infer::matmul(&self.layout.shape, &rhs.layout.shape)?;
        infer::same_device(self.device, rhs.device)?;
        infer::same_dtype(self.dtype(), rhs.dtype(), "matmul")?;
        Ok(self.matmul(rhs))
    }

    pub fn try_add(self, rhs: Tensor) -> Result<Tensor> {
        infer::broadcast(&self.layout.shape, &rhs.layout.shape)?;
        infer::same_device(self.device, rhs.device)?;
        infer::same_dtype(self.dtype(), rhs.dtype(), "add")?;
        Ok(Tensor::add(self, rhs))
    }

    pub fn try_sub(self, rhs: Tensor) -> Result<Tensor> {
        infer::broadcast(&self.layout.shape, &rhs.layout.shape)?;
        infer::same_device(self.device, rhs.device)?;
        infer::same_dtype(self.dtype(), rhs.dtype(), "sub")?;
        Ok(Tensor::sub(self, rhs))
    }

    pub fn try_mul(self, rhs: Tensor) -> Result<Tensor> {
        infer::broadcast(&self.layout.shape, &rhs.layout.shape)?;
        infer::same_device(self.device, rhs.device)?;
        infer::same_dtype(self.dtype(), rhs.dtype(), "mul")?;
        Ok(Tensor::mul(self, rhs))
    }

    pub fn try_div(self, rhs: Tensor) -> Result<Tensor> {
        infer::broadcast(&self.layout.shape, &rhs.layout.shape)?;
        infer::same_device(self.device, rhs.device)?;
        infer::same_dtype(self.dtype(), rhs.dtype(), "div")?;
        Ok(Tensor::div(self, rhs))
    }

    pub fn try_sum_dim(self, dim: usize) -> Result<Tensor> {
        infer::dim(&self.layout.shape, dim, "sum_dim")?;
        Ok(self.sum_dim(dim))
    }

    pub fn try_mean_dim(self, dim: usize) -> Result<Tensor> {
        infer::dim(&self.layout.shape, dim, "mean_dim")?;
        Ok(self.mean_dim(dim))
    }

    pub fn try_max_dim(self, dim: usize) -> Result<Tensor> {
        infer::dim(&self.layout.shape, dim, "max_dim")?;
        Ok(self.max_dim(dim))
    }

    pub fn try_mask_fill(self, mask: BoolTensor, value: impl Into<f64>) -> Result<Tensor> {
        infer::mask(&self.layout.shape, &mask.layout.shape)?;
        Ok(self.mask_fill(mask, value))
    }

    pub fn try_gather(self, dim: usize, indices: IntTensor) -> Result<Tensor> {
        infer::index(&self.layout.shape, dim, &indices.layout.shape, &indices.data(), "gather")?;
        Ok(self.gather(dim, indices))
    }

    pub fn try_select(self, dim: usize, indices: IntTensor) -> Result<Tensor> {
        infer::select(&self.layout.shape, dim, &indices.data(), "select")?;
        Ok(self.select(dim, indices))
    }

    pub fn try_into_scalar(&self) -> Result<f32> {
        infer::scalar(&self.layout.shape)?;
        Ok(self.into_scalar())
    }

    pub fn try_backward(&self) -> Result<super::Gradients> {
        infer::backward_root(&self.layout.shape, self.node.is_some())?;
        Ok(self.backward())
    }
}
