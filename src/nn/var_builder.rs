//! `VarBuilder`: how components get their vars. In init mode it creates each var
//! the first time it is asked for, in the order asked (so a model's constructor fixes the
//! `VarMap`'s order), drawing seed slot j from seed index `indices[j]`'s own stream (`Init`). In
//! load mode it reads existing vars (a loaded or lifted map) and checks their shapes. Device and
//! dtype come only from here: components never choose them.
//!
//! Names are dotted paths: `vb.pp("b1").get(.., "wq", ..)` is the var "b1.wq"; an empty prefix
//! adds nothing (so an unprefixed block keeps plain names).

use super::init::Init;
use super::var::VarMap;
use crate::error::{NnError, Result};
use crate::tensor::{default_device, DType, Device, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
enum Source<'a> {
    Init { map: &'a RefCell<VarMap>, indices: Rc<[usize]>, root: u64 },
    Load { map: &'a VarMap },
}

#[derive(Clone)]
pub struct VarBuilder<'a> {
    src: Source<'a>,
    prefix: String,
    dtype: DType,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    /// Create vars into `map` for seeds 0..`seeds` from `root` (F32, on the thread's default
    /// device).
    pub fn init(map: &'a RefCell<VarMap>, seeds: usize, root: u64) -> Self {
        Self::init_indexed(map, &(0..seeds).collect::<Vec<_>>(), root)
    }

    /// As `init`, with slot j initialised as seed `indices[j]` (paired grids reuse indices).
    pub fn init_indexed(map: &'a RefCell<VarMap>, indices: &[usize], root: u64) -> Self {
        VarBuilder { src: Source::Init { map, indices: indices.into(), root }, prefix: String::new(), dtype: DType::F32, device: default_device() }
    }

    /// Read the vars of `map` (a loaded or lifted map); dtype and device are the map's.
    pub fn from_varmap(map: &'a VarMap) -> Self {
        let (dtype, device) = map.tensors().first().map(|t| (t.dtype(), t.device())).unwrap_or((DType::F32, default_device()));
        VarBuilder { src: Source::Load { map }, prefix: String::new(), dtype, device }
    }

    /// Created vars get this dtype (F32 or F64; F16 and BF16 as storage).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Created vars live on this device.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The seed count S.
    pub fn seeds(&self) -> usize {
        match &self.src {
            Source::Init { indices, .. } => indices.len(),
            Source::Load { map } => map.seeds().unwrap_or(0),
        }
    }

    /// A builder whose names are prefixed by `name` (an empty name adds nothing).
    pub fn pp(&self, name: &str) -> VarBuilder<'a> {
        let mut vb = self.clone();
        vb.prefix = self.path(name);
        vb
    }

    /// The full name of `name` under this builder's prefix.
    pub fn path(&self, name: &str) -> String {
        match (self.prefix.is_empty(), name.is_empty()) {
            (_, true) => self.prefix.clone(),
            (true, false) => name.to_string(),
            (false, false) => format!("{}.{name}", self.prefix),
        }
    }

    /// The var `name` of per-seed shape `shape` (the var is `[S, shape…]`). Init mode creates it
    /// with `init` if it does not exist yet; load mode reads it. The shape is checked either way.
    pub fn get(&self, shape: &[usize], name: &str, init: Init) -> Result<Tensor> {
        let full = self.path(name);
        let want: Vec<usize> = std::iter::once(self.seeds()).chain(shape.iter().copied()).collect();
        let check = |t: &Tensor| -> Result<Tensor> {
            if t.shape() != want.as_slice() {
                return Err(NnError::Tensor(format!("var {full:?} is {:?}, the model asks for {want:?}", t.shape())));
            }
            Ok(t.clone())
        };
        match &self.src {
            Source::Load { map } => check(map.get(&full).ok_or_else(|| NnError::Tensor(format!("no var named {full:?}")))?),
            Source::Init { map, indices, root } => {
                if let Some(t) = map.borrow().get(&full) {
                    return check(t);
                }
                let n: usize = shape.iter().product();
                let mut vals = Vec::with_capacity(indices.len() * n);
                for &i in indices.iter() {
                    vals.extend(init.draw(*root, i, &full, n));
                }
                let t = match self.dtype {
                    DType::F32 | DType::F64 => Tensor::from_f64s(vals, want.clone(), self.dtype),
                    // Storage-only types: drawn in f32, stored rounded to nearest even.
                    DType::F16 | DType::BF16 => Tensor::from_f64s(vals, want.clone(), DType::F32).cast(self.dtype),
                    other => return Err(NnError::Tensor(format!("var {full:?}: vars are float, not {}", other.name()))),
                };
                let t = t.try_to(self.device)?;
                map.borrow_mut().insert(full, t.clone())?;
                Ok(t)
            }
        }
    }
}
