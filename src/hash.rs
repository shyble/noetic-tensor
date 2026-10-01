//! sha256 as lowercase hex, for the GPU kernel-source keys and the test fixtures.

use sha2::{Digest, Sha256};

/// sha256 of `data`, as lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    to_hex(&Sha256::digest(data))
}

/// Bytes as lowercase hex, two digits per byte.
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes.iter().flat_map(|b| [DIGITS[(b >> 4) as usize] as char, DIGITS[(b & 15) as usize] as char]).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(super::sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(super::sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
