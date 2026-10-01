//! Element types: the runtime `DType`, the `Element` trait that ties a Rust type
//! to its dtype and storage variant, and `FloatElem`, the float types the kernels compute in
//! (f32, f64). The dtypes: F32, F64, I64, I32, U8, F16 and BF16.

use super::half::{BF16, F16};
use super::storage::CpuStorage;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Bool,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }

    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::Bool => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }

    pub fn is_int(self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::U8)
    }
}

/// A Rust element type with its dtype and its CPU storage variant.
pub trait Element: Copy + Send + Sync + std::fmt::Debug + 'static {
    const DTYPE: DType;
    fn wrap(v: Vec<Self>) -> CpuStorage;
    fn slice(s: &CpuStorage) -> Option<&[Self]>;
}

macro_rules! element {
    ($t:ty, $d:ident) => {
        impl Element for $t {
            const DTYPE: DType = DType::$d;
            fn wrap(v: Vec<$t>) -> CpuStorage {
                CpuStorage::$d(v)
            }
            fn slice(s: &CpuStorage) -> Option<&[$t]> {
                if let CpuStorage::$d(v) = s { Some(v) } else { None }
            }
        }
    };
}
element!(f32, F32);
element!(f64, F64);
element!(F16, F16);
element!(BF16, BF16);
element!(i64, I64);
element!(i32, I32);
element!(u8, U8);
element!(bool, Bool);

/// The float types the kernels compute in. Every method is the type's own std operation, so an
/// f32 kernel written against this trait does exactly what the f32 code did.
pub trait FloatElem: Element + PartialOrd + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> + std::ops::Div<Output = Self> + std::ops::Neg<Output = Self> {
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
    fn from_f32(x: f32) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(n: usize) -> Self;
    fn from_i64(x: i64) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn is_sign_positive(self) -> bool;
    fn total_cmp(&self, other: &Self) -> std::cmp::Ordering;
    /// C = A·B for row-major m×k and k×n blocks (matrixmultiply's sgemm or dgemm).
    ///
    /// # Safety
    /// The pointers must cover m×k, k×n and m×n row-major blocks.
    unsafe fn gemm(m: usize, k: usize, n: usize, a: *const Self, b: *const Self, c: *mut Self);
}

macro_rules! float_elem {
    ($t:ty, $gemm:ident) => {
        impl FloatElem for $t {
            const ZERO: $t = 0.0;
            const ONE: $t = 1.0;
            fn from_f64(x: f64) -> $t { x as $t }
            fn from_f32(x: f32) -> $t { x as $t }
            fn to_f64(self) -> f64 { self as f64 }
            fn from_usize(n: usize) -> $t { n as $t }
            fn from_i64(x: i64) -> $t { x as $t }
            fn exp(self) -> $t { <$t>::exp(self) }
            fn ln(self) -> $t { <$t>::ln(self) }
            fn sqrt(self) -> $t { <$t>::sqrt(self) }
            fn abs(self) -> $t { <$t>::abs(self) }
            fn powf(self, p: $t) -> $t { <$t>::powf(self, p) }
            fn is_sign_positive(self) -> bool { <$t>::is_sign_positive(self) }
            fn total_cmp(&self, other: &$t) -> std::cmp::Ordering { <$t>::total_cmp(self, other) }
            unsafe fn gemm(m: usize, k: usize, n: usize, a: *const $t, b: *const $t, c: *mut $t) {
                matrixmultiply::$gemm(m, k, n, 1.0, a, k as isize, 1, b, n as isize, 1, 0.0, c, n as isize, 1);
            }
        }
    };
}
float_elem!(f32, sgemm);
float_elem!(f64, dgemm);
