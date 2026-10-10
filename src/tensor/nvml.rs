//! The NVIDIA driver's release (e.g. "576.88"), read through NVML, the driver's management
//! library, loaded at run time (`nvml.dll` on Windows, `libnvidia-ml.so.1` elsewhere) through the
//! system loader: no link-time dependency, so the CUDA backend builds and runs on one device
//! where NVML is missing (some containers lack it). Only distributed runs need it, for their
//! platform key, and there a missing NVML is an error, never a key without the release.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};

#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(name: *const c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}

// glibc before 2.34 keeps dlopen in libdl (a stub after it).
#[cfg(unix)]
#[cfg_attr(all(target_os = "linux", target_env = "gnu"), link(name = "dl"))]
extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}

/// NVML's library on this system.
pub(crate) const LIBRARY: &str = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };

/// Open a shared library by name; None when the loader cannot find or load it.
fn open(name: &str) -> Option<*mut c_void> {
    let c = CString::new(name).ok()?;
    // SAFETY: a NUL-terminated name; the loader's own search rules apply.
    #[cfg(windows)]
    let h = unsafe { LoadLibraryA(c.as_ptr()) };
    // RTLD_NOW (2) on Linux and macOS.
    #[cfg(unix)]
    let h = unsafe { dlopen(c.as_ptr(), 2) };
    #[cfg(not(any(windows, unix)))]
    let h: *mut c_void = {
        let _ = c;
        std::ptr::null_mut()
    };
    (!h.is_null()).then_some(h)
}

fn symbol(h: *mut c_void, name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `h` is a live handle from `open`; `name` is NUL-terminated.
    #[cfg(windows)]
    let p = unsafe { GetProcAddress(h, name.as_ptr()) };
    #[cfg(unix)]
    let p = unsafe { dlsym(h, name.as_ptr()) };
    #[cfg(not(any(windows, unix)))]
    let p: *mut c_void = {
        let _ = (h, name);
        std::ptr::null_mut()
    };
    (!p.is_null()).then_some(p)
}

/// The driver's release through the NVML library `library` (the handle stays open).
pub(crate) fn release_from(library: &str) -> Result<String, String> {
    let h = open(library).ok_or_else(|| format!("NVML ({library}) could not be loaded"))?;
    let get = |n: &CStr| symbol(h, n).ok_or_else(|| format!("NVML ({library}) has no {}", n.to_string_lossy()));
    let (init, version, shutdown) = (get(c"nvmlInit_v2")?, get(c"nvmlSystemGetDriverVersion")?, get(c"nvmlShutdown")?);
    // SAFETY: the symbols have these C signatures (nvml.h); init and shutdown are reference
    // counted; the buffer outlives the call and its length is passed.
    unsafe {
        let init: unsafe extern "C" fn() -> c_int = std::mem::transmute(init);
        let version: unsafe extern "C" fn(*mut c_char, c_uint) -> c_int = std::mem::transmute(version);
        let shutdown: unsafe extern "C" fn() -> c_int = std::mem::transmute(shutdown);
        let r = init();
        if r != 0 {
            return Err(format!("nvmlInit: status {r}"));
        }
        let mut buf = [0 as c_char; 96];
        let r = version(buf.as_mut_ptr(), buf.len() as c_uint);
        shutdown();
        if r != 0 {
            return Err(format!("nvmlSystemGetDriverVersion: status {r}"));
        }
        let s = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
        if s.is_empty() {
            return Err("NVML gave an empty driver release".into());
        }
        Ok(s)
    }
}

/// A library name used instead of NVML's, set before the first read (tests of the error path).
#[cfg(test)]
pub(crate) static OVERRIDE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// The driver's release, read once per process.
pub(crate) fn driver_release() -> Result<String, String> {
    static R: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        #[cfg(test)]
        if let Some(lib) = OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            return release_from(&lib);
        }
        release_from(LIBRARY)
    })
    .clone()
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_missing_nvml_is_an_error() {
        let e = super::release_from("libnvidia-ml-that-does-not-exist.so.0").unwrap_err();
        assert!(e.contains("could not be loaded"), "{e}");
    }
}
