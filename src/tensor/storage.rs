//! Storage: where a tensor's elements live. `Storage::Cpu` holds a
//! `CpuStorage`, one variant per dtype; `Storage::Gpu` a device buffer (; it exists
//! only in a build with a GPU feature). A tensor shares its storage (`Arc<Storage>`) and reads it
//! through its `Layout`; host reads of GPU storage download it (`host`).

use super::dtype::{DType, Element};
use super::gpu::GpuStorage;
use super::half::{BF16, F16};
use std::borrow::Cow;

#[derive(Clone, Debug)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<F16>),
    BF16(Vec<BF16>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
}

macro_rules! each {
    ($s:expr, $v:ident => $e:expr) => {
        match $s {
            CpuStorage::F32($v) => $e,
            CpuStorage::F64($v) => $e,
            CpuStorage::F16($v) => $e,
            CpuStorage::BF16($v) => $e,
            CpuStorage::I64($v) => $e,
            CpuStorage::I32($v) => $e,
            CpuStorage::U8($v) => $e,
            CpuStorage::Bool($v) => $e,
        }
    };
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F64(_) => DType::F64,
            CpuStorage::F16(_) => DType::F16,
            CpuStorage::BF16(_) => DType::BF16,
            CpuStorage::I64(_) => DType::I64,
            CpuStorage::I32(_) => DType::I32,
            CpuStorage::U8(_) => DType::U8,
            CpuStorage::Bool(_) => DType::Bool,
        }
    }

    pub fn len(&self) -> usize {
        each!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Gpu(GpuStorage),
}

impl Storage {
    pub fn from_vec<E: Element>(v: Vec<E>) -> Storage {
        Storage::Cpu(E::wrap(v))
    }

    pub fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(s) => s.dtype(),
            Storage::Gpu(g) => g.dtype,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(s) => s.len(),
            Storage::Gpu(g) => g.len,
        }
    }

    /// The elements on the host: CPU storage itself, or GPU storage downloaded (after the
    /// device finishes its encoded work).
    pub fn host(&self) -> Cow<'_, CpuStorage> {
        match self {
            Storage::Cpu(s) => Cow::Borrowed(s),
            Storage::Gpu(g) => Cow::Owned(super::error::ok(super::gpu::download(g))),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as `E`, if the storage holds `E`.
    pub fn try_slice<E: Element>(&self) -> Option<&[E]> {
        match self {
            Storage::Cpu(s) => E::slice(s),
            Storage::Gpu(_) => None,
        }
    }

    /// Every element converted to the float type `T` (f16/bf16 exactly through f32; ints and
    /// bools by `as`).
    pub(crate) fn float_vec<T: super::dtype::FloatElem>(&self) -> Vec<T> {
        let host = self.host();
        match &*host {
            CpuStorage::F32(v) => v.iter().map(|x| T::from_f32(*x)).collect(),
            CpuStorage::F64(v) => v.iter().map(|x| T::from_f64(*x)).collect(),
            CpuStorage::F16(v) => v.iter().map(|x| T::from_f32(x.to_f32())).collect(),
            CpuStorage::BF16(v) => v.iter().map(|x| T::from_f32(x.to_f32())).collect(),
            CpuStorage::I64(v) => v.iter().map(|x| T::from_i64(*x)).collect(),
            CpuStorage::I32(v) => v.iter().map(|x| T::from_i64(*x as i64)).collect(),
            CpuStorage::U8(v) => v.iter().map(|x| T::from_i64(*x as i64)).collect(),
            CpuStorage::Bool(v) => v.iter().map(|x| if *x { T::ONE } else { T::ZERO }).collect(),
        }
    }

    /// Every element as i64 (integer storages; bools as 0/1).
    pub(crate) fn int_vec(&self) -> Option<Vec<i64>> {
        let host = self.host();
        match &*host {
            CpuStorage::I64(v) => Some(v.clone()),
            CpuStorage::I32(v) => Some(v.iter().map(|x| *x as i64).collect()),
            CpuStorage::U8(v) => Some(v.iter().map(|x| *x as i64).collect()),
            CpuStorage::Bool(v) => Some(v.iter().map(|x| *x as i64).collect()),
            _ => None,
        }
    }

    /// The elements as `E`; panics if the storage holds another dtype.
    pub fn as_slice<E: Element>(&self) -> &[E] {
        match self {
            Storage::Cpu(s) => E::slice(s).unwrap_or_else(|| panic!("storage holds {}, read as {}", s.dtype().name(), E::DTYPE.name())),
            Storage::Gpu(g) => panic!("storage of {} is on {:?}; read it through `host`", g.dtype.name(), g.device),
        }
    }
}
