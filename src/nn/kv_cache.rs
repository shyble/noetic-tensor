//! KV-cache: the keys and values of the positions already read, per block,
//! in the attention's head layout `[S·B·H, L, dh]`, so decoding one more token computes only
//! that token's row. Caching is for inference: a tracked tensor is refused (a cache would keep
//! the graph alive across steps and mix gradients of different passes).

use crate::error::{NnError, Result};
use crate::tensor::Tensor;

#[derive(Clone, Debug, Default)]
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
    capacity: Option<usize>,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// A cache of fixed capacity: its tensors always span `capacity` positions.
    pub fn with_capacity(capacity: usize) -> Self {
        KvCache { capacity: Some(capacity), ..Self::default() }
    }

    /// The fixed capacity, if any.
    pub fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    /// Positions held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget every position.
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.len = 0;
    }

    /// The cached keys and values, if any.
    pub fn get(&self) -> Option<(&Tensor, &Tensor)> {
        Some((self.k.as_ref()?, self.v.as_ref()?))
    }

    /// Append the keys and values of new positions (`[S·B·H, t, dh]` each) and return the whole
    /// cache (`[S·B·H, L + t, dh]`; with a capacity, the `[S·B·H, T, dh]` buffer). An error for
    /// tracked tensors, shapes that do not continue the cache, or a full cache.
    pub fn append(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        if k.is_tracked() || v.is_tracked() {
            return Err(NnError::Tensor("the KV-cache holds untracked tensors only (decode on an untracked model)".into()));
        }
        if k.rank() != 3 || k.shape() != v.shape() {
            return Err(NnError::Tensor(format!("KV-cache: keys {:?} and values {:?} must be the same [S·B·H, t, dh]", k.shape(), v.shape())));
        }
        let [n, t, dh] = k.dims();
        if let Some(ck) = &self.k {
            let a = ck.shape();
            if a[0] != n || a[2] != dh {
                return Err(NnError::Tensor(format!("KV-cache holds {a:?}; cannot append {:?}", k.shape())));
            }
        }
        let (k, v) = match self.capacity {
            Some(cap) => {
                if self.len + t > cap {
                    return Err(NnError::Tensor(format!("KV-cache of capacity {cap} holds {}; cannot append {t}", self.len)));
                }
                let (ck, cv) = match (&self.k, &self.v) {
                    (Some(a), Some(b)) => (a.clone(), b.clone()),
                    _ => (Tensor::full_dtype([n, cap, dh], 0.0, k.dtype()).to(k.device()), Tensor::full_dtype([n, cap, dh], 0.0, v.dtype()).to(v.device())),
                };
                let r = self.len..self.len + t;
                (ck.slice_assign([0..n, r.clone(), 0..dh], k), cv.slice_assign([0..n, r, 0..dh], v))
            }
            None => match (&self.k, &self.v) {
                (Some(ck), Some(cv)) => (Tensor::cat(vec![ck.clone(), k], 1), Tensor::cat(vec![cv.clone(), v], 1)),
                _ => (k, v),
            },
        };
        self.len += t;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }
}

/// One `KvCache` per decoder block.
#[derive(Clone, Debug)]
pub struct DecoderCache {
    blocks: Vec<KvCache>,
}

impl DecoderCache {
    /// Empty, extendable caches.
    pub fn new(blocks: usize) -> Self {
        DecoderCache { blocks: vec![KvCache::new(); blocks] }
    }

    /// Caches of fixed capacity (the decoder's context: bit-equal to full recompute).
    pub fn with_capacity(blocks: usize, capacity: usize) -> Self {
        DecoderCache { blocks: vec![KvCache::with_capacity(capacity); blocks] }
    }

    /// Positions held (every block holds the same number).
    pub fn len(&self) -> usize {
        self.blocks.first().map_or(0, |c| c.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) {
        self.blocks.iter_mut().for_each(|c| c.reset());
    }

    pub fn blocks(&self) -> &[KvCache] {
        &self.blocks
    }

    pub fn block_mut(&mut self, j: usize) -> &mut KvCache {
        &mut self.blocks[j]
    }
}
