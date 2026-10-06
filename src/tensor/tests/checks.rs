//! Must-hold checks: shape and bounds checks panic instead of
//! computing wrong numbers, backward needs a single-element root, NaN and all-(−∞) softmax
//! behaviour is pinned (bit for bit, as burn), and a long graph drops without deep recursion.

use super::*;
use crate::tensor::{self as nt, BoolTensor};

fn t(shape: &[usize]) -> Tensor {
    let n = numel(shape);
    Tensor::from_data(rnd(n, 1, -1.0, 1.0), shape.to_vec())
}

#[test]
#[should_panic(expected = "cannot broadcast")]
fn expand_rejects_a_non_broadcastable_shape() {
    let _ = t(&[2, 3]).expand([2, 4]);
}

#[test]
#[should_panic(expected = "cannot broadcast")]
fn int_expand_rejects_a_non_broadcastable_shape() {
    let _ = IntTensor::zeros([3]).expand([2, 4]);
}

#[test]
#[should_panic(expected = "cannot broadcast")]
fn bool_expand_rejects_a_non_broadcastable_shape() {
    let _ = BoolTensor::tril_mask([3, 3], 0).expand([2, 4, 4]);
}

#[test]
#[should_panic(expected = "cannot broadcast")]
fn mask_fill_rejects_a_mismatched_mask() {
    let _ = t(&[2, 3]).mask_fill(BoolTensor::tril_mask([2, 2], 0), 0.0);
}

#[test]
#[should_panic(expected = "mask_fill: mask rank")]
fn mask_fill_rejects_a_mask_of_another_rank() {
    let _ = t(&[1, 3, 3]).mask_fill(BoolTensor::tril_mask([3, 3], 0), 0.0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn slice_assign_rejects_out_of_bounds_ranges() {
    let _ = t(&[2, 3]).slice_assign([0..2, 2..4], t(&[2, 2]));
}

#[test]
#[should_panic(expected = "value shape must match")]
fn slice_assign_rejects_a_mismatched_value() {
    let _ = t(&[2, 3]).slice_assign([0..2, 0..2], t(&[2, 3]));
}

#[test]
#[should_panic(expected = "select: index out of range")]
fn select_rejects_out_of_range_indices() {
    let _ = t(&[2, 3]).select(1, IntTensor::from_ints(&[0, 3]));
}

#[test]
#[should_panic(expected = "select_add: index out of range")]
fn select_backward_rejects_out_of_range_indices() {
    let _ = super::super::kernels::select_add_zeros(&[2, 3], 1, &[5], &[0.0; 2]);
}

#[test]
#[should_panic(expected = "cat: shapes")]
fn cat_rejects_mismatched_shapes() {
    let _ = Tensor::cat(vec![t(&[2, 3]), t(&[3, 3])], 1);
}

#[test]
#[should_panic(expected = "gather: index shape")]
fn gather_rejects_a_mismatched_leading_dimension() {
    // Leading dimension 3 against 2: previously read past the rows (only a debug check on the
    // trailing dimensions existed).
    let _ = t(&[2, 4, 5]).gather(1, IntTensor::zeros([3, 4, 5]));
}

#[test]
#[should_panic(expected = "gather: index shape")]
fn gather_rejects_a_mismatched_trailing_dimension() {
    let _ = t(&[2, 4, 5]).gather(1, IntTensor::zeros([2, 4, 4]));
}

#[test]
#[should_panic(expected = "gather index")]
fn gather_rejects_out_of_range_indices() {
    let _ = t(&[2, 4]).gather(1, IntTensor::from_data(vec![0, 4, 1, 1], [2, 2]));
}

#[test]
#[should_panic(expected = "single-element root")]
fn backward_needs_a_scalar_root() {
    let x = t(&[2, 3]).require_grad();
    let _ = (x.clone() * x).backward();
}

/// Softmax and log-softmax of rows with NaN or with every entry −∞ (a fully masked attention
/// row): the current values, bit for bit as burn's recorded ones (NaN out, no panic).
#[test]
fn softmax_of_nan_and_fully_masked_rows_is_burns() {
    let ninf = f32::NEG_INFINITY;
    let v = vec![0.5, f32::NAN, -1.0, 2.0, ninf, ninf, ninf, ninf, 1.0, ninf, 3.0, ninf];
    for d in [1usize] {
        let ours = nt::softmax(Tensor::from_data(v.clone(), [3, 4]), d).to_vec();
        fixture::f32s("softmax with NaN / all −inf", &ours);
        assert!(ours[..8].iter().all(|x| x.is_nan()), "a NaN or all-(−∞) row gives NaN");
        assert!(ours[8..].iter().all(|x| !x.is_nan()));
        let ours = nt::log_softmax(Tensor::from_data(v.clone(), [3, 4]), d).to_vec();
        fixture::f32s("log_softmax with NaN / all −inf", &ours);
    }
}

/// A 200,000-op chain drops without recursing once per node.
#[test]
fn a_long_graph_drops_without_deep_recursion() {
    let h = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let x = Tensor::from_floats(&[1.0]).require_grad();
            let mut y = x.clone();
            for _ in 0..200_000 {
                y = y.add_scalar(1.0);
            }
            drop(y);
        })
        .unwrap();
    h.join().expect("dropping a long graph must not overflow the stack");
}

/// Containers: every tensor is a contiguous layout over shared CPU storage of its dtype;
/// reshape shares the storage.
#[test]
fn containers_hold_dtype_storage_and_a_contiguous_layout() {
    use crate::tensor::{DType, Storage};
    let x = t(&[2, 3, 4]);
    assert_eq!(x.dtype(), DType::F32);
    assert!(x.layout().is_contiguous());
    assert_eq!((x.layout().shape(), x.layout().strides(), x.layout().offset()), (&[2usize, 3, 4][..], &[12usize, 4, 1][..], 0));
    let y = x.clone().reshape([6, 4]);
    assert!(std::sync::Arc::ptr_eq(&x.storage, &y.storage), "reshape shares the storage");
    assert_eq!(IntTensor::zeros([2]).dtype(), DType::I64);
    assert_eq!(BoolTensor::tril_mask([2, 2], 0).dtype(), DType::Bool);
    let s = Storage::from_vec(vec![1i64, 2]);
    assert_eq!((s.dtype(), s.len(), s.as_slice::<i64>()), (DType::I64, 2, &[1i64, 2][..]));
}

#[test]
#[should_panic(expected = "storage holds i64, read as f32")]
fn storage_refuses_a_wrong_dtype_read() {
    let _ = crate::tensor::Storage::from_vec(vec![1i64]).as_slice::<f32>().len();
}

/// Every check is a `TensorError` on the `try_*` API (the panicking methods report the
/// same message), and a good call returns the same values as the panicking method.
#[test]
fn try_api_returns_typed_errors() {
    use crate::tensor::TensorError as E;
    let is = |r: crate::tensor::Result<Tensor>, want: fn(&E) -> bool, text: &str| {
        let e = r.err().expect("an error");
        assert!(want(&e) && e.to_string().contains(text), "{e:?}");
    };
    is(t(&[2, 3]).try_expand([2, 4]), |e| matches!(e, E::Broadcast(_)), "cannot broadcast");
    is(t(&[2, 3]).try_add(t(&[3, 2])), |e| matches!(e, E::Broadcast(_)), "do not broadcast");
    is(t(&[2, 3]).try_reshape([4, 2]), |e| matches!(e, E::Shape(_)), "reshape");
    is(t(&[2, 3]).try_slice([0..2, 2..4]), |e| matches!(e, E::Index(_)), "out of bounds");
    is(t(&[2, 3]).try_slice_assign([0..2, 0..2], t(&[2, 3])), |e| matches!(e, E::Shape(_)), "value shape must match");
    is(Tensor::try_cat(vec![t(&[2, 3]), t(&[3, 3])], 1), |e| matches!(e, E::Shape(_)), "cat: shapes");
    is(t(&[2, 3]).try_matmul(t(&[2, 3])), |e| matches!(e, E::Shape(_)), "inner dimensions");
    is(t(&[2, 4, 5]).try_gather(1, IntTensor::zeros([3, 4, 5])), |e| matches!(e, E::Shape(_)), "gather: index shape");
    is(t(&[2, 4]).try_gather(1, IntTensor::from_data(vec![0, 4, 1, 1], [2, 2])), |e| matches!(e, E::Index(_)), "gather index 4");
    is(t(&[2, 3]).try_select(1, IntTensor::from_ints(&[3])), |e| matches!(e, E::Index(_)), "select: index out of range");
    is(t(&[2, 3]).try_mask_fill(BoolTensor::tril_mask([2, 2], 0), 0.0), |e| matches!(e, E::Broadcast(_)), "cannot broadcast");
    is(t(&[2, 3]).try_sum_dim(2), |e| matches!(e, E::Index(_)), "sum_dim: dim 2");
    is(t(&[2, 3]).try_swap_dims(0, 3), |e| matches!(e, E::Index(_)), "swap_dims");
    let x = t(&[2, 3]).require_grad();
    assert!(matches!((x.clone() * x.clone()).try_backward(), Err(E::Shape(_))));
    assert!(matches!(t(&[1]).try_backward(), Err(E::Unsupported(_))));
    assert!(matches!(t(&[2]).try_into_scalar(), Err(E::Shape(_))));
    let a = t(&[1, 2, 3]).try_matmul(t(&[1, 3, 4])).unwrap();
    assert_bits("try_matmul = matmul", a.as_slice(), t(&[1, 2, 3]).matmul(t(&[1, 3, 4])).as_slice());
    let n: crate::NnError = E::Shape("s".into()).into();
    assert_eq!(n, crate::NnError::Tensor("s".into()));
}

/// Reshape (of a row-major layout), swap_dims, slice and expand are views over the same
/// storage; their values, and every result computed from them, equal the copying versions'.
#[test]
fn views_share_storage_and_read_like_copies() {
    let x = t(&[2, 3, 4]);
    let same = |a: &Tensor, b: &Tensor| std::sync::Arc::ptr_eq(&a.storage, &b.storage);
    let s = x.clone().swap_dims(0, 2);
    let n = x.clone().slice([0..2, 1..3, 1..4]);
    let e = x.clone().slice([0..2, 0..1, 0..4]).expand([2, 5, 4]);
    let r = x.clone().reshape([6, 4]);
    for v in [&s, &n, &e, &r] {
        assert!(same(&x, v), "a view shares the storage");
    }
    assert!(s.is_view() && n.is_view() && e.is_view() && !r.is_view());
    // Row-major element order of each view, against an explicit copy.
    let xv = x.to_vec();
    let at = |i: usize, j: usize, k: usize| xv[i * 12 + j * 4 + k];
    let want_s: Vec<f32> = (0..4).flat_map(|k| (0..3).flat_map(move |j| (0..2).map(move |i| (k, j, i)))).map(|(k, j, i)| at(i, j, k)).collect();
    assert_bits("swap view", s.as_slice(), &want_s);
    let want_n: Vec<f32> = (0..2).flat_map(|i| (1..3).flat_map(move |j| (1..4).map(move |k| (i, j, k)))).map(|(i, j, k)| at(i, j, k)).collect();
    assert_bits("slice view", n.as_slice(), &want_n);
    let want_e: Vec<f32> = (0..2).flat_map(|i| (0..5).flat_map(move |_| (0..4).map(move |k| (i, k)))).map(|(i, k)| at(i, 0, k)).collect();
    assert_bits("expand view", e.as_slice(), &want_e);
    // A view of a view, reshaped (copied once) and reduced.
    let chain = x.clone().swap_dims(1, 2).slice([0..2, 1..3, 0..3]).reshape([2, 6]).sum_dim(1);
    let copied = Tensor::from_data(x.clone().swap_dims(1, 2).to_vec(), [2, 4, 3]).slice([0..2, 1..3, 0..3]).to_vec();
    let copied = Tensor::from_data(copied, [2, 6]).sum_dim(1);
    assert_bits("view chain", chain.as_slice(), copied.as_slice());
    // Gradients through views equal the copy path's.
    let p = t(&[2, 3, 4]).require_grad();
    let w = t(&[4, 2, 2]);
    let y = (p.clone().swap_dims(0, 2).slice([0..4, 1..3, 0..2]) * w.clone()).sum();
    let g = p.grad(&y.backward()).unwrap();
    assert_eq!(g.shape(), &[2, 3, 4]);
    let manual: Vec<f32> = {
        let wv = w.to_vec();
        let mut m = vec![0.0f32; 24];
        for a in 0..4 {
            for b in 0..2 {
                for c in 0..2 {
                    // p.swap(0,2)[a, b + 1, c] = p[c, b + 1, a]
                    m[c * 12 + (b + 1) * 4 + a] += wv[a * 4 + b * 2 + c];
                }
            }
        }
        m
    };
    assert_bits("gradient through views", g.as_slice(), &manual);
}

/// f16 and bf16 conversions round to nearest (ties to even) against an exact reference
/// (the nearest of every finite value of the type), and every bit pattern round-trips.
#[test]
fn half_and_bfloat16_conversions_are_exact() {
    use crate::tensor::half::{BF16, F16};
    for h in 0..=u16::MAX {
        let x = F16(h).to_f32();
        if !x.is_nan() {
            assert_eq!(F16::from_f32(x).0, h, "f16 {h:#06x} round trip");
        }
        let y = BF16(h).to_f32();
        if !y.is_nan() {
            assert_eq!(BF16::from_f32(y).0, h, "bf16 {h:#06x} round trip");
        }
    }
    // Reference rounding: among the finite positive values of each type (sorted by value), the
    // nearest in f64, ties to the even bit pattern; beyond the largest finite value, round to it
    // unless at least half a step above (then ∞).
    let near = |vals: &[(u16, f32)], x: f32| -> u16 {
        let i = vals.partition_point(|v| v.1 < x);
        let cands = [i.checked_sub(1), (i < vals.len()).then_some(i)];
        let mut best: Option<(f64, u16)> = None;
        for c in cands.into_iter().flatten() {
            let d = (vals[c].1 as f64 - x as f64).abs();
            best = match best {
                None => Some((d, vals[c].0)),
                Some((bd, bh)) if d < bd || (d == bd && vals[c].0 % 2 == 0 && bh % 2 == 1) => Some((d, vals[c].0)),
                b => b,
            };
        }
        best.unwrap().1
    };
    let fvals: Vec<(u16, f32)> = (0..0x7c00u16).map(|h| (h, F16(h).to_f32())).collect();
    let bvals: Vec<(u16, f32)> = (0..0x7f80u16).map(|h| (h, BF16(h).to_f32())).collect();
    let mut r = 0x1234_5678u32;
    for i in 0..200_000u32 {
        r ^= r << 13;
        r ^= r >> 17;
        r ^= r << 5;
        // Positive f32s spanning the half range (and beyond for bf16), plus exact midpoints.
        let x = f32::from_bits((r & 0x7fff_ffff) % 0x4780_0000);
        if x < 65504.0 {
            assert_eq!(F16::from_f32(x).0, near(&fvals, x), "f16 of {x:e}");
            assert_eq!(F16::from_f32(-x).0, near(&fvals, x) | 0x8000, "f16 of {:e}", -x);
        }
        let y = f32::from_bits(r & 0x7f7f_ffff);
        if y <= BF16(0x7f7f).to_f32() {
            assert_eq!(BF16::from_f32(y).0, near(&bvals, y), "bf16 of {y:e} ({i})");
        }
    }
    assert_eq!(F16::from_f32(1e6).0, 0x7c00, "overflow to +inf");
    assert_eq!(F16::from_f32(65520.0).0, 0x7c00, "half a step above the largest finite value");
    assert!(F16::from_f32(f32::NAN).to_f32().is_nan() && BF16::from_f32(f32::NAN).to_f32().is_nan());
}

/// Dtypes and casts. f64 tensors compute in f64; mixed dtypes are an error (casts are
/// explicit); int storages I32/U8 read as i64; f16/bf16 store converted values.
#[test]
fn dtypes_and_casts() {
    use crate::tensor::{DType, TensorError};
    let x = t(&[2, 3]);
    let d = x.clone().cast(DType::F64);
    assert_eq!(d.dtype(), DType::F64);
    assert_eq!(d.to_vec_f64(), x.to_vec().iter().map(|v| *v as f64).collect::<Vec<_>>());
    let y = (d.clone() * d.clone()).sum().to_vec_f64()[0];
    let want: f64 = x.to_vec().iter().map(|v| (*v as f64) * (*v as f64)).fold([0.0f64; 8], |mut p, v| { p[0] += v; p })[0];
    assert!((y - want).abs() < 1e-12);
    assert!(matches!(x.clone().try_add(d.clone()), Err(TensorError::DType(_))));
    let h = x.clone().cast(DType::F16);
    assert_eq!(h.dtype(), DType::F16);
    let back = h.clone().cast(DType::F32).to_vec();
    assert!(back.iter().zip(x.to_vec()).all(|(a, b)| (a - b).abs() <= b.abs() * 1e-3));
    let hb = x.clone().cast(DType::BF16).cast(DType::F32).to_vec();
    assert!(hb.iter().zip(x.to_vec()).all(|(a, b)| (a - b).abs() <= b.abs() * 1e-2));
    // Gradient of a cast flows back in the input's dtype.
    let p = t(&[4]).require_grad();
    let g = p.grad(&p.clone().cast(DType::F64).mul_scalar(3.0).sum().backward()).unwrap();
    assert_eq!(g.dtype(), DType::F32);
    assert_eq!(g.to_vec(), vec![3.0f32; 4]);
    let i = IntTensor::from_ints(&[1, 200, -3]).cast(DType::I32);
    assert_eq!(i.dtype(), DType::I32);
    assert_eq!(i.to_vec(), vec![1, 200, -3]);
    let u = IntTensor::from_ints(&[1, 200, 300]).cast(DType::U8);
    assert_eq!((u.dtype(), u.to_vec()), (DType::U8, vec![1, 200, 44]));
    assert_eq!(IntTensor::from_ints(&[2, 5]).float_dtype(DType::F64).to_vec_f64(), vec![2.0, 5.0]);
}
