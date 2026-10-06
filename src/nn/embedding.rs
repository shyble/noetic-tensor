//! `Embedding`: a table `[S, V, d]` read at token ids `[S, N]`.
//! - `OneHot`: `one_hot(ids) · table`, the default embedding (its exact mode). Its gradient
//!   is the matmul's, `one_hotᵀ · g`.
//! - `Gather`: `gather` of the rows, O(N·d) instead of O(N·V·d). Its values equal OneHot's; its
//!   gradient is a scatter-add, which can differ from the matmul's in the last bits when a row
//!   is read more than once.

use super::init::Init;
use super::var_builder::VarBuilder;
use crate::error::Result;
use crate::tensor::{IntTensor, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum EmbeddingMode {
    #[default]
    OneHot,
    Gather,
}

#[derive(Clone, Debug)]
pub struct Embedding {
    table: Tensor,
    mode: EmbeddingMode,
}

impl Embedding {
    pub fn new(table: Tensor, mode: EmbeddingMode) -> Self {
        Embedding { table, mode }
    }

    pub fn table(&self) -> &Tensor {
        &self.table
    }

    pub fn mode(&self) -> EmbeddingMode {
        self.mode
    }

    /// Ids `[S, N]` → `[S, N, d]`.
    pub fn forward(&self, ids: &IntTensor) -> Tensor {
        let [s, v, d] = self.table.dims();
        let [si, n] = ids.dims();
        assert_eq!(si, s, "embedding: ids have {si} seeds, the table {s}");
        match self.mode {
            EmbeddingMode::OneHot => ids.clone().one_hot_float(v, self.table.dtype(), self.table.device()).matmul(self.table.clone()),
            EmbeddingMode::Gather => {
                let idx = ids.clone().reshape([s, n, 1]).expand([s, n, d]);
                self.table.clone().gather(1, idx)
            }
        }
    }
}

/// An embedding table named `name`, `[S, vocab, d]`, initialised N(0, 1).
pub fn embedding(vocab: usize, d: usize, name: &str, mode: EmbeddingMode, vb: &VarBuilder) -> Result<Embedding> {
    Ok(Embedding::new(vb.get(&[vocab, d], name, Init::Normal { std: 1.0 })?, mode))
}
