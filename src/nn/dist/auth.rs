//! The authenticated handshake: a per-job secret, shared by the launcher with every rank, and
//! HMAC-SHA256 challenge-responses at the rendezvous and on every link of the mesh.
//!
//! Each side of a connection sends a fresh random challenge (32 bytes from the OS); the other
//! side answers with HMAC-SHA256(secret, label · challenge · its message), so a rank proves it
//! holds the job's secret without sending it, and an answer cannot be replayed on another
//! connection. Without a secret the handshake carries empty answers and a rank holding one
//! refuses it: unauthenticated jobs (the job key alone) are for an isolated development network
//! only. Only the handshake is authenticated; the collectives' traffic is neither authenticated
//! nor encrypted after it.

use crate::error::{NnError, Result};
use rand::RngCore;

/// A job's secret: at least 32 bytes. Its `Debug` form never shows the bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct JobSecret(Vec<u8>);

impl std::fmt::Debug for JobSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JobSecret({} bytes)", self.0.len())
    }
}

impl JobSecret {
    /// The variable the launcher passes the secret in (hex).
    pub const ENV: &'static str = "DIST_JOB_SECRET";

    /// 32 fresh random bytes from the OS.
    pub fn random() -> JobSecret {
        let mut b = vec![0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut b);
        JobSecret(b)
    }

    pub fn from_bytes(b: &[u8]) -> Result<JobSecret> {
        if b.len() < 32 {
            return Err(NnError::Dist(format!("a job secret of {} bytes (at least 32)", b.len())));
        }
        Ok(JobSecret(b.to_vec()))
    }

    pub fn from_hex(s: &str) -> Result<JobSecret> {
        let s = s.trim();
        let nib = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
        if !s.len().is_multiple_of(2) {
            return Err(NnError::Dist("a job secret with an odd number of hex digits".into()));
        }
        let b: Option<Vec<u8>> = s.as_bytes().chunks_exact(2).map(|p| Some(nib(p[0])? << 4 | nib(p[1])?)).collect();
        JobSecret::from_bytes(&b.ok_or_else(|| NnError::Dist("a job secret that is not hex".into()))?)
    }

    pub fn to_hex(&self) -> String {
        crate::hash::to_hex(&self.0)
    }

    /// The secret in `DIST_JOB_SECRET`, if set.
    pub fn from_env() -> Result<Option<JobSecret>> {
        std::env::var(Self::ENV).ok().map(|v| JobSecret::from_hex(&v)).transpose()
    }
}

/// A fresh 32-byte challenge.
pub(crate) fn challenge() -> Vec<u8> {
    let mut b = vec![0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut b);
    b
}

/// The answer to `challenge` for a message `fields` under `label`: HMAC-SHA256, or empty
/// without a secret.
pub(crate) fn answer(secret: Option<&JobSecret>, label: &str, challenge: &[u8], fields: &[u8]) -> Vec<u8> {
    let Some(s) = secret else { return vec![] };
    let mut m = Vec::with_capacity(label.len() + 1 + challenge.len() + fields.len());
    m.extend_from_slice(label.as_bytes());
    m.push(0);
    m.extend_from_slice(challenge);
    m.extend_from_slice(fields);
    crate::hash::hmac_sha256(&s.0, &m).to_vec()
}

/// Whether `got` answers `challenge` for `fields` (in constant time for equal lengths). Without
/// a secret only an empty answer passes; with one, only the right one.
pub(crate) fn verify(secret: Option<&JobSecret>, label: &str, challenge: &[u8], fields: &[u8], got: &[u8]) -> bool {
    let want = answer(secret, label, challenge, fields);
    want.len() == got.len() && want.iter().zip(got).fold(0u8, |a, (x, y)| a | (x ^ y)) == 0
}
