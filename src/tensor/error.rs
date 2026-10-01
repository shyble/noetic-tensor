//! Tensor errors: every shape, dtype and index check of the core
//! returns a `TensorError`; the panicking methods report the same message at the caller.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Shapes that do not broadcast (burn's rule: equal ranks, each dimension equal or 1).
    Broadcast(String),
    /// A value of another dtype than the operation needs.
    DType(String),
    /// An index or range outside a dimension.
    Index(String),
    /// Shapes or lengths that do not fit the operation.
    Shape(String),
    /// An operation this tensor cannot do (for example backward on an untracked tensor).
    Unsupported(String),
    /// A GPU device failed or refused (driver, runtime compiler or BLAS).
    Device(String),
}

impl TensorError {
    pub fn message(&self) -> &str {
        match self {
            Self::Broadcast(m) | Self::DType(m) | Self::Index(m) | Self::Shape(m) | Self::Unsupported(m) | Self::Device(m) => m,
        }
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for TensorError {}

impl From<TensorError> for crate::Error {
    fn from(e: TensorError) -> Self {
        crate::Error::Tensor(e.message().to_string())
    }
}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Unwrap for the panicking API: the error's message, reported at the caller's line.
#[track_caller]
pub(crate) fn ok<T>(r: Result<T>) -> T {
    r.unwrap_or_else(|e| panic!("{e}"))
}
