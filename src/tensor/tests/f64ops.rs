//! f64 operations against burn's `NdArray<f64>` (inputs made as burn's from_floats makes
//! them: through f32), bit for bit, against the values recorded from burn (fixtures/burn021-<os>-<arch>.txt).
//! Without burn in the process, the test no longer has to run alone. sigmoid is not in the list
//! (burn computed it in f32 even for f64; it is checked in `standard`).

use super::*;
use crate::tensor::{self as nt, DType};
fn ours(v: &[f64], s: [usize; 3]) -> Tensor { Tensor::from_f64s(v.iter().map(|x| (*x as f32) as f64).collect(), s, DType::F64) }
fn cmp(what: &str, a: Tensor) {
    fixture::f64s(what, a.to_vec_f64());
}
#[test]
fn f64_ops_are_bit_equal_to_burn_f64() {
    let s = [2, 5, 13];
    let x: Vec<f64> = rnd(130, 1, -3.0, 3.0).iter().map(|v| *v as f64 * 1.37).collect();
    let y: Vec<f64> = rnd(130, 2, 0.5, 3.0).iter().map(|v| *v as f64 * 0.91).collect();
    cmp("exp", ours(&x, s).exp());
    cmp("log", ours(&y, s).log());
    cmp("sqrt", ours(&y, s).sqrt());
    cmp("recip", ours(&y, s).recip());
    cmp("sum_dim2", ours(&x, s).sum_dim(2));
    cmp("mean_dim2", ours(&x, s).mean_dim(2));
    cmp("mean_dim1", ours(&x, s).mean_dim(1));
    cmp("softmax", nt::softmax(ours(&x, s), 2));
    cmp("log_softmax", nt::log_softmax(ours(&x, s), 2));
    cmp("add_scalar", ours(&x, s) + 1e-6);
    cmp("mul_scalar", ours(&x, s).mul_scalar(0.3));
    cmp("powf2", ours(&x, s).powf_scalar(2.0));
    let w: Vec<f64> = rnd(13 * 7, 3, -1.0, 1.0).iter().map(|v| *v as f64).collect();
    cmp("matmul", ours(&x, s).matmul(ours(&w, [1, 13, 7])));
}
