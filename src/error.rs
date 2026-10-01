//! The crate's error type: one enum with a variant per area. `Display` is the message alone.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// An invalid configuration: sizes or options that contradict each other.
    Config(String),
    /// A saved var map: an unknown format, a missing part, a malformed field.
    Persist(String),
    /// A tensor whose shape, length or type does not match what it is loaded into.
    Tensor(String),
}

/// The crate's result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// The message.
    pub fn message(&self) -> &str {
        match self {
            Self::Config(m) | Self::Persist(m) | Self::Tensor(m) => m,
        }
    }

    /// The area, in lower case: "config", "persist" or "tensor".
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Config(_) => "config",
            Self::Persist(_) => "persist",
            Self::Tensor(_) => "tensor",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

impl From<Error> for String {
    fn from(e: Error) -> Self {
        e.message().to_string()
    }
}

impl From<Error> for std::io::Error {
    fn from(e: Error) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, String::from(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_the_message_and_every_variant_has_its_kind() {
        let all = [Error::Config("c".into()), Error::Persist("p".into()), Error::Tensor("t".into())];
        let kinds: Vec<&str> = all.iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, ["config", "persist", "tensor"]);
        for e in &all {
            assert_eq!(e.to_string(), e.message());
            assert_eq!(String::from(e.clone()), e.message());
        }
        let io: std::io::Error = Error::Persist("bad".into()).into();
        assert_eq!((io.kind(), io.to_string()), (std::io::ErrorKind::InvalidData, "bad".to_string()));
    }
}
