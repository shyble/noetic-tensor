//! `VarMap`: the named parameters of a model, in insertion order, every var
//! `[S, …]` with the seed axis first. There is no interior mutability: a training step lifts the
//! map (`lifted`: every var a fresh `require_grad` leaf), builds the model from the lifted map,
//! reads the gradients back by var index (`grads`) and an optimizer replaces the untracked vars.
//!
//! Save and load are exact to the bit: the format (`FORMAT`) stores each var's name, dtype
//! (f32, f64, f16 or bf16), shape and the little-endian bytes of its stored values in hex (so
//! NaN payloads, signalling NaNs of the 16-bit types included, and signed zeros survive),
//! through serde. f16 and bf16 are storage types: they are read and written as their bit
//! patterns, never through a conversion.

use crate::error::{NnError, Result};
use crate::tensor::half::{BF16, F16};
use crate::tensor::{CpuStorage, DType, Gradients, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The save format's name and version.
pub const FORMAT: &str = "noetic.nn.v1";

#[derive(Clone, Debug, Default)]
pub struct VarMap {
    names: Vec<String>,
    tensors: Vec<Tensor>,
    index: BTreeMap<String, usize>,
}

impl VarMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The names, in insertion order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The vars, in insertion order.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.names.iter().map(|n| n.as_str()).zip(self.tensors.iter())
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.index_of(name).map(|i| &self.tensors[i])
    }

    /// The var `name`; panics if absent.
    #[track_caller]
    pub fn var(&self, name: &str) -> &Tensor {
        self.get(name).unwrap_or_else(|| panic!("no var named {name:?}"))
    }

    /// Append a var; an error if the name is taken or the var has no seed axis.
    pub fn insert(&mut self, name: impl Into<String>, t: Tensor) -> Result<usize> {
        let name = name.into();
        if self.index.contains_key(&name) {
            return Err(NnError::Tensor(format!("var {name:?} already exists")));
        }
        if t.rank() == 0 {
            return Err(NnError::Tensor(format!("var {name:?} has no seed axis")));
        }
        let i = self.names.len();
        self.index.insert(name.clone(), i);
        self.names.push(name);
        self.tensors.push(t);
        Ok(i)
    }

    /// Replace var `name` with a tensor of the same shape and dtype.
    pub fn set(&mut self, name: &str, t: Tensor) -> Result<()> {
        let i = self.index_of(name).ok_or_else(|| NnError::Tensor(format!("no var named {name:?}")))?;
        self.set_index(i, t)
    }

    /// Replace var `i` with a tensor of the same shape and dtype.
    pub fn set_index(&mut self, i: usize, t: Tensor) -> Result<()> {
        let old = &self.tensors[i];
        if old.shape() != t.shape() || old.dtype() != t.dtype() {
            return Err(NnError::Tensor(format!("var {:?} is {:?} {}, not {:?} {}", self.names[i], old.shape(), old.dtype().name(), t.shape(), t.dtype().name())));
        }
        self.tensors[i] = t;
        Ok(())
    }

    /// The seed count S (the leading dimension), if the map holds any var.
    pub fn seeds(&self) -> Option<usize> {
        self.tensors.first().map(|t| t.shape()[0])
    }

    /// Elements per seed over every var.
    pub fn count_per_seed(&self) -> usize {
        self.tensors.iter().map(|t| t.shape()[1..].iter().product::<usize>()).sum()
    }

    /// The same vars, each a fresh `require_grad` leaf .
    pub fn lifted(&self) -> VarMap {
        VarMap { names: self.names.clone(), tensors: self.tensors.iter().map(|t| t.clone().detach().require_grad()).collect(), index: self.index.clone() }
    }

    /// The same vars, untracked.
    pub fn detached(&self) -> VarMap {
        VarMap { names: self.names.clone(), tensors: self.tensors.iter().map(|t| t.clone().untracked()).collect(), index: self.index.clone() }
    }

    /// Each var's gradient in `grads` (None for a var that took no part), by var index. The map
    /// must be the lifted one the loss was built from.
    pub fn grads(&self, grads: &Gradients) -> Vec<Option<Tensor>> {
        self.tensors.iter().map(|t| t.grad(grads)).collect()
    }

    /// The saved form.
    pub fn to_file(&self) -> Result<VarMapFile> {
        let mut vars = Vec::with_capacity(self.len());
        for (name, t) in self.iter() {
            let bytes: Vec<u8> = match t.dtype() {
                DType::F32 => t.to_vec().iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
                DType::F64 => t.to_vec_f64().iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
                DType::F16 | DType::BF16 => half_bits(t).iter().flat_map(|x| x.to_le_bytes()).collect(),
                other => return Err(NnError::Persist(format!("var {name:?}: saving {} vars is not supported", other.name()))),
            };
            vars.push(VarRecord { name: name.to_string(), dtype: t.dtype().name().to_string(), shape: t.shape().to_vec(), data: to_hex(&bytes) });
        }
        Ok(VarMapFile { format: FORMAT.to_string(), vars })
    }

    /// Rebuild from the saved form (on the calling thread's default device).
    pub fn from_file(f: &VarMapFile) -> Result<VarMap> {
        if f.format != FORMAT {
            return Err(NnError::Persist(format!("unknown var map format {:?} (expected {FORMAT:?})", f.format)));
        }
        let mut map = VarMap::new();
        for r in &f.vars {
            let n: usize = r.shape.iter().product();
            let bytes = from_hex(&r.data).ok_or_else(|| NnError::Persist(format!("var {:?}: malformed data", r.name)))?;
            let t = match r.dtype.as_str() {
                "f32" => {
                    if bytes.len() != 4 * n {
                        return Err(NnError::Persist(format!("var {:?}: {} bytes for {n} f32 values", r.name, bytes.len())));
                    }
                    Tensor::from_data(bytes.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect(), r.shape.clone())
                }
                "f64" => {
                    if bytes.len() != 8 * n {
                        return Err(NnError::Persist(format!("var {:?}: {} bytes for {n} f64 values", r.name, bytes.len())));
                    }
                    let v: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_bits(u64::from_le_bytes(c.try_into().expect("8 bytes")))).collect();
                    Tensor::from_f64s(v, r.shape.clone(), DType::F64)
                }
                "f16" | "bf16" => {
                    if bytes.len() != 2 * n {
                        return Err(NnError::Persist(format!("var {:?}: {} bytes for {n} {} values", r.name, bytes.len(), r.dtype)));
                    }
                    let b: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    if r.dtype == "f16" {
                        Tensor::raw_t(b.into_iter().map(F16).collect::<Vec<_>>(), r.shape.clone())
                    } else {
                        Tensor::raw_t(b.into_iter().map(BF16).collect::<Vec<_>>(), r.shape.clone())
                    }
                }
                other => return Err(NnError::Persist(format!("var {:?}: unknown dtype {other:?}", r.name))),
            };
            map.insert(r.name.clone(), t).map_err(|e| NnError::Persist(e.message().to_string()))?;
        }
        Ok(map)
    }

    /// The saved form as JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(&self.to_file()?).map_err(|e| NnError::Persist(format!("var map: {e}")))
    }

    pub fn from_json(s: &str) -> Result<VarMap> {
        let f: VarMapFile = serde_json::from_str(s).map_err(|e| NnError::Persist(format!("var map: {e}")))?;
        VarMap::from_file(&f)
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_json()?).map_err(|e| NnError::Persist(format!("{}: {e}", path.display())))
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<VarMap> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path).map_err(|e| NnError::Persist(format!("{}: {e}", path.display())))?;
        VarMap::from_json(&s)
    }
}

/// The saved form of a `VarMap` (`FORMAT`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VarMapFile {
    pub format: String,
    pub vars: Vec<VarRecord>,
}

/// One saved var: its values as the hex of their little-endian bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VarRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: String,
}

fn to_hex(bytes: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    bytes.iter().flat_map(|b| [D[(b >> 4) as usize] as char, D[(b & 15) as usize] as char]).collect()
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    let s = s.as_bytes();
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let nib = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        }
    };
    s.chunks_exact(2).map(|p| Some(nib(p[0])? << 4 | nib(p[1])?)).collect()
}

/// The stored bit patterns of an f16 or bf16 tensor, in row-major order.
fn half_bits(t: &Tensor) -> Vec<u16> {
    let host = t.storage.host();
    let cpu = &*host;
    let bits: Vec<u16> = match cpu {
        CpuStorage::F16(v) => v.iter().map(|x| x.0).collect(),
        CpuStorage::BF16(v) => v.iter().map(|x| x.0).collect(),
        _ => unreachable!("a 16-bit float tensor"),
    };
    let l = t.layout();
    if l.is_contiguous() {
        return bits[l.offset()..l.offset() + l.numel()].to_vec();
    }
    // A view: walk its strides.
    let shape = l.shape().to_vec();
    let mut out = Vec::with_capacity(l.numel());
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..l.numel() {
        out.push(bits[l.offset() + idx.iter().zip(l.strides()).map(|(i, s)| i * s).sum::<usize>()]);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}
