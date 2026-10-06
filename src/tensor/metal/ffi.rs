//! Hand-written bindings to the Objective-C runtime and Metal: no crate. Every
//! message goes through `objc_msgSend`, transmuted to the exact C signature of the method
//! (arm64 has no variadic message send), with each selector looked up once per call site.
//! Metal.framework, Foundation and libobjc are linked by build.rs only with the `metal` feature.

#![allow(non_upper_case_globals)]

use std::ffi::{c_char, c_void, CStr};

/// An Objective-C object pointer (`id`).
pub type Id = *mut c_void;
/// A selector (`SEL`).
pub type Sel = *const c_void;

extern "C" {
    pub fn objc_getClass(name: *const c_char) -> Id;
    pub fn sel_registerName(name: *const c_char) -> Sel;
    /// Never called with this signature: always transmuted to the method's own.
    pub fn objc_msgSend();
    pub fn objc_autoreleasePoolPush() -> *mut c_void;
    pub fn objc_autoreleasePoolPop(pool: *mut c_void);
    pub fn MTLCreateSystemDefaultDevice() -> Id;
}

/// `MTLSize` (three NSUIntegers, passed by value).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MTLSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl MTLSize {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        MTLSize { width, height, depth }
    }
}

/// `MTLResourceStorageModeShared` (CPU and GPU see one buffer: unified memory) with the
/// default (tracked) hazard mode and default CPU cache mode.
pub const MTLResourceStorageModeShared: usize = 0;
/// `MTLCommandBufferStatusCompleted`.
pub const MTLCommandBufferStatusCompleted: isize = 4;
/// `NSUTF8StringEncoding`.
pub const NSUTF8StringEncoding: usize = 4;
/// `MTLMathModeSafe` and `MTLMathFloatingPointFunctionsPrecise` (macOS 15+).
pub const MTLMathModeSafe: isize = 0;
pub const MTLMathFloatingPointFunctionsPrecise: isize = 1;

/// A selector looked up once per call site.
macro_rules! sel {
    ($name:literal) => {{
        static S: std::sync::atomic::AtomicPtr<std::ffi::c_void> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
        let mut p = S.load(std::sync::atomic::Ordering::Relaxed);
        if p.is_null() {
            // SAFETY: a NUL-terminated literal.
            #[allow(unused_unsafe)]
            let q = unsafe { $crate::tensor::metal::ffi::sel_registerName(concat!($name, "\0").as_ptr() as *const std::ffi::c_char) } as *mut std::ffi::c_void;
            p = q;
            S.store(p, std::sync::atomic::Ordering::Relaxed);
        }
        p as $crate::tensor::metal::ffi::Sel
    }};
}
pub(crate) use sel;

/// `[obj sel arg…]` with the method's exact signature: `msg!(Ret; obj, "sel:", a: A, …)`.
macro_rules! msg {
    ($ret:ty; $obj:expr, $name:literal $(, $a:expr => $t:ty)* $(,)?) => {{
        let f: unsafe extern "C" fn($crate::tensor::metal::ffi::Id, $crate::tensor::metal::ffi::Sel $(, $t)*) -> $ret =
            std::mem::transmute($crate::tensor::metal::ffi::objc_msgSend as unsafe extern "C" fn());
        f($obj, $crate::tensor::metal::ffi::sel!($name) $(, $a)*)
    }};
}
pub(crate) use msg;

/// A class by name (`objc_getClass`); null if not loaded.
pub fn class(name: &CStr) -> Id {
    // SAFETY: a valid C string.
    unsafe { objc_getClass(name.as_ptr()) }
}

/// An autorelease pool until drop (popped on drop).
pub struct Pool(*mut c_void);

impl Pool {
    pub fn new() -> Pool {
        // SAFETY: push/pop are paired by this guard, on this thread.
        Pool(unsafe { objc_autoreleasePoolPush() })
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: the pool pushed by `new`, on the same thread (the guard is not Send).
        unsafe { objc_autoreleasePoolPop(self.0) }
    }
}

/// A +1 NSString of `s` (the caller releases it).
///
/// # Safety
/// Foundation must be loaded (it is linked with the feature).
pub unsafe fn nsstring(s: &str) -> Id {
    let cls = class(c"NSString");
    let alloc: Id = msg!(Id; cls, "alloc");
    msg!(Id; alloc, "initWithBytes:length:encoding:", s.as_ptr() as *const c_void => *const c_void, s.len() => usize, NSUTF8StringEncoding => usize)
}

/// The UTF-8 text of an NSString (empty for nil).
///
/// # Safety
/// `s` is nil or an NSString.
pub unsafe fn string_of(s: Id) -> String {
    if s.is_null() {
        return String::new();
    }
    let p: *const c_char = msg!(*const c_char; s, "UTF8String");
    if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
}

/// An NSError's localized description (empty for nil).
///
/// # Safety
/// `e` is nil or an NSError.
pub unsafe fn error_text(e: Id) -> String {
    if e.is_null() {
        return String::new();
    }
    string_of(msg!(Id; e, "localizedDescription"))
}

/// `[obj release]` (nil is ignored).
///
/// # Safety
/// `obj` is nil or an object the caller owns a reference to.
pub unsafe fn release(obj: Id) {
    if !obj.is_null() {
        msg!((); obj, "release");
    }
}

/// `[obj retain]`.
///
/// # Safety
/// `obj` is nil or a live object.
pub unsafe fn retain(obj: Id) -> Id {
    if obj.is_null() { obj } else { msg!(Id; obj, "retain") }
}

/// `[obj respondsToSelector:sel]`.
///
/// # Safety
/// `obj` is a live object.
pub unsafe fn responds(obj: Id, sel: Sel) -> bool {
    msg!(bool; obj, "respondsToSelector:", sel => Sel)
}
