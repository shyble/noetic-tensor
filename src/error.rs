//! The nn library's error type (the tensor engine's own is `tensor::TensorError`): one enum with a
//! variant per area. `Display` is the message alone.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NnError {
    /// An invalid configuration: sizes or options that contradict each other.
    Config(String),
    /// An MLP extension was given to a feed-forward that is not the gated MLP.
    ExtensionNeedsGatedMlp,
    /// A saved var map: an unknown format, a missing part, a malformed field.
    Persist(String),
    /// A tensor whose shape, length or type does not match what it is loaded into.
    Tensor(String),
    /// Distributed training: the environment, the rendezvous, a collective, a launched process
    /// or a coordinated checkpoint.
    Dist(String),
}

/// The 0.1.0 name of `NnError`, kept for compatibility.
pub type Error = NnError;

/// The nn library's result type.
pub type Result<T, E = NnError> = std::result::Result<T, E>;

impl NnError {
    /// The message.
    pub fn message(&self) -> &str {
        match self {
            Self::Config(m) | Self::Persist(m) | Self::Tensor(m) | Self::Dist(m) => m,
            Self::ExtensionNeedsGatedMlp => "an MLP extension needs the gated MLP",
        }
    }

    /// The area, in lower case: "config", "persist", "tensor" or "dist".
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Config(_) | Self::ExtensionNeedsGatedMlp => "config",
            Self::Persist(_) => "persist",
            Self::Tensor(_) => "tensor",
            Self::Dist(_) => "dist",
        }
    }
}

impl fmt::Display for NnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for NnError {}

impl From<NnError> for String {
    fn from(e: NnError) -> Self {
        e.message().to_string()
    }
}

impl From<NnError> for std::io::Error {
    fn from(e: NnError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, String::from(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_the_message_and_every_variant_has_its_kind() {
        let all = [Error::Config("c".into()), Error::Persist("p".into()), Error::Tensor("t".into()), Error::ExtensionNeedsGatedMlp, Error::Dist("d".into())];
        let kinds: Vec<&str> = all.iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, ["config", "persist", "tensor", "config", "dist"]);
        for e in &all {
            assert_eq!(e.to_string(), e.message());
            assert_eq!(String::from(e.clone()), e.message());
        }
        let io: std::io::Error = Error::Persist("bad".into()).into();
        assert_eq!((io.kind(), io.to_string()), (std::io::ErrorKind::InvalidData, "bad".to_string()));
    }
}
