//! Recorded reference outputs. The parity tests compare the tensors with burn
//! 0.21's outputs, recorded once from burn itself on each platform the tests run on
//! (`fixtures/burn021-<os>-<arch>.txt`); burn is not a dependency.
//!
//! A record is keyed by the test (its thread name, which libtest sets to the test path), the
//! check's label, and the occurrence of that label within the test (`#n` from the second one).
//! It holds the element type, the element count and the sha256 of the elements' little-endian
//! bit patterns, so a record is exact to the bit. `NOETIC_FIXTURE_RECORD=FILE` appends records
//! instead of checking them.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Every fixture file, by name and the platform (`os-arch`) it was recorded on: libm and the
/// gemm kernel differ between platforms, so a record is exact only on its own.
const FILES: &[(&str, &str, &str)] = &[
    ("burn021", "macos-aarch64", include_str!("fixtures/burn021-macos-aarch64.txt")),
    ("burn021", "windows-x86_64", include_str!("fixtures/burn021-windows-x86_64.txt")),
];

fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Whether this platform has records; without them every check is skipped with a notice.
pub(crate) fn available() -> bool {
    FILES.iter().any(|(_, p, _)| *p == platform())
}

struct Record {
    kind: String,
    len: usize,
    sha: String,
}

fn records() -> &'static HashMap<String, (String, Record)> {
    static R: OnceLock<HashMap<String, (String, Record)>> = OnceLock::new();
    R.get_or_init(|| {
        let mut m = HashMap::new();
        for (file, _, text) in FILES.iter().filter(|(_, p, _)| *p == platform()) {
            for line in text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
                let f: Vec<&str> = line.split('\t').collect();
                assert_eq!(f.len(), 4, "fixture {file}: malformed line {line:?}");
                let rec = Record { kind: f[1].into(), len: f[2].parse().expect("fixture length"), sha: f[3].into() };
                assert!(m.insert(f[0].to_string(), (file.to_string(), rec)).is_none(), "fixture key {} recorded twice", f[0]);
            }
        }
        m
    })
}

/// The key of the next check labelled `what` in the running test.
fn key(what: &str) -> String {
    static SEEN: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    let test = std::thread::current().name().unwrap_or("main").to_string();
    let base = format!("{test}::{what}");
    let mut seen = SEEN.get_or_init(Default::default).lock().unwrap();
    let n = seen.entry(base.clone()).or_insert(0);
    *n += 1;
    if *n == 1 { base } else { format!("{base}#{n}") }
}

fn check(what: &str, kind: &str, len: usize, bytes: impl Iterator<Item = u8>) {
    let mut h = Sha256::new();
    h.update(bytes.collect::<Vec<u8>>());
    let sha = crate::hash::to_hex(&h.finalize());
    let k = key(what);
    if let Ok(path) = std::env::var("NOETIC_FIXTURE_RECORD") {
        use std::io::Write;
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).expect("open the fixture record file");
        // One write per record, so records from concurrent processes never interleave.
        f.write_all(format!("{k}\t{kind}\t{len}\t{sha}\n").as_bytes()).expect("write a fixture record");
        return;
    }
    if !available() {
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| eprintln!("fixture: no records for {}; the recorded-reference checks are skipped", platform()));
        return;
    }
    let (file, rec) = records().get(&k).unwrap_or_else(|| panic!("no recorded fixture for {k:?}"));
    assert_eq!((rec.kind.as_str(), rec.len), (kind, len), "{what}: element type or count differs from the {file} record");
    assert!(rec.sha == sha, "{what}: the values differ from the {file} record (key {k})");
}

/// f32 values, bit for bit against the record.
pub(crate) fn f32s(what: &str, v: impl AsRef<[f32]>) {
    let v = v.as_ref();
    check(what, "f32", v.len(), v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
}

/// f64 values, bit for bit against the record.
pub(crate) fn f64s(what: &str, v: impl AsRef<[f64]>) {
    let v = v.as_ref();
    check(what, "f64", v.len(), v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
}

/// i64 values against the record.
pub(crate) fn i64s(what: &str, v: impl AsRef<[i64]>) {
    let v = v.as_ref();
    check(what, "i64", v.len(), v.iter().flat_map(|x| x.to_le_bytes()));
}

/// bool values against the record.
pub(crate) fn bools(what: &str, v: impl AsRef<[bool]>) {
    let v = v.as_ref();
    check(what, "bool", v.len(), v.iter().map(|x| *x as u8));
}
