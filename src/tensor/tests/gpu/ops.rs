// Tensor ops, one set of bodies for every GPU backend: tensor ops on the device against
// CpuRef. Bit-exact for copies, views, masks, comparisons, elementwise ops without
// transcendentals, the reductions (CpuRef's order), matmul (matrixmultiply's order, also past one
// k block), indexing and casts; exp, ln, sqrt, pow and sigmoid within 4 ulp. The reference pin
// refuses the device in a fresh process and never initialises the GPU.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::tensor::tests::*;
use crate::tensor::{self as nt, BoolTensor, CpuMode, DType, Device, IntTensor, Tensor, TensorError};

const R: Device = Device::Cpu(CpuMode::Reference);

fn t(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_data(rnd(numel(shape), seed, -2.0, 2.0), shape.to_vec())
}

/// Distance in units in the last place (same-sign finite values; 0 for equal bits).
fn ulps(a: f32, b: f32) -> u64 {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN.wrapping_sub(i) as i64 } else { i as i64 }
    };
    (key(a) - key(b)).unsigned_abs()
}

fn max_ulps(a: &[f32], b: &[f32]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| ulps(*x, *y)).max().unwrap_or(0)
}

/// `f` on CPU (Reference) and on the device from the same inputs, read back.
fn both(name: &str, inputs: &[Tensor], f: impl Fn(&[Tensor]) -> Tensor) -> (Vec<f32>, Vec<f32>) {
    let cpu = f(inputs).to_vec();
    let on: Vec<Tensor> = inputs.iter().map(|x| x.clone().to(M)).collect();
    let out = f(&on);
    assert_eq!(out.device(), M, "{name}: results stay on the device");
    assert!(matches!(&*out.storage, crate::tensor::Storage::Gpu(_)), "{name}: the result lives on the GPU");
    (cpu, out.to_vec())
}

/// The op bodies, in a fresh process.
#[test]
fn ops_against_cpu_ref() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 7);
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_exact_ops_match_cpu_ref_bit_for_bit() {
    let a = t(&[3, 5, 7], 1);
    let b = t(&[3, 1, 7], 2);
    let c = t(&[3, 5, 7], 3).abs().add_scalar(0.5);
    let ins = [a, b, c];
    type F = fn(&[Tensor]) -> Tensor;
    let cases: Vec<(&str, F)> = vec![
        ("add broadcast", |x| x[0].clone() + x[1].clone()),
        ("sub broadcast", |x| x[1].clone() - x[0].clone()),
        ("mul", |x| x[0].clone() * x[2].clone()),
        ("div", |x| x[0].clone() / x[2].clone()),
        ("add of views", |x| x[0].clone().swap_dims(0, 2) + x[2].clone().swap_dims(0, 2)),
        ("scalars", |x| x[0].clone().add_scalar(0.3).mul_scalar(1.7).sub_scalar(0.1).div_scalar(3.0)),
        ("neg recip abs sign", |x| x[0].clone().neg().recip().abs() + x[0].clone().sign()),
        ("powi 2 and -2", |x| x[2].clone().powf_scalar(2.0) + x[2].clone().powf_scalar(-2.0)),
        ("clamp (compare + mask_fill)", |x| x[0].clone().clamp(-0.5, 0.7)),
        ("mask_fill broadcast mask", |x| x[0].clone().mask_fill(x[1].clone().greater_elem(0.0), -9.0)),
        ("mask_fill host mask", |x| x[0].clone().slice([0..3, 0..5, 0..5]).mask_fill(BoolTensor::tril_mask([5, 5], 0).unsqueeze_dim(0), -1e9)),
        ("sum", |x| x[0].clone().sum()),
        ("mean", |x| x[0].clone().mean()),
        ("sum_dim 0", |x| x[0].clone().sum_dim(0)),
        ("sum_dim 1", |x| x[0].clone().sum_dim(1)),
        ("sum_dim 2", |x| x[0].clone().sum_dim(2)),
        ("sum_dim of a view", |x| x[0].clone().swap_dims(1, 2).sum_dim(2)),
        ("sum_dim last, long lane", |x| t(&[4, 1037], 9).to(x[0].device()).sum_dim(1)),
        ("mean_dim", |x| x[0].clone().mean_dim(1) + x[0].clone().mean_dim(2).sum_dim(1)),
        ("max_dim", |x| x[0].clone().max_dim(1).add(x[0].clone().max_dim(2).sum_dim(1))),
        ("max", |x| x[0].clone().max()),
        ("argmax (as floats)", |x| x[0].clone().argmax(2).float().to(x[0].device())),
        ("reshape of a view", |x| x[0].clone().swap_dims(0, 1).reshape([15, 7])),
        ("slice", |x| x[0].clone().slice([1..3, 0..5, 2..6]).mul_scalar(1.0)),
        ("slice_assign", |x| x[0].clone().slice_assign([0..2, 1..3, 0..7], x[2].clone().slice([0..2, 0..2, 0..7]))),
        ("cat", |x| Tensor::cat(vec![x[0].clone(), x[2].clone().swap_dims(0, 0), x[0].clone()], 1)),
        ("expand", |x| x[1].clone().expand([3, 4, 7]).mul_scalar(2.0)),
        ("matmul batched", |x| x[0].clone().matmul(x[2].clone().swap_dims(1, 2))),
        ("matmul broadcast batch", |x| x[0].clone().matmul(x[2].clone().slice([0..1, 0..5, 0..7]).swap_dims(1, 2))),
        ("matmul rank 4 swapped", |x| x[0].clone().reshape([3, 5, 1, 7]).matmul(x[2].clone().reshape([3, 1, 7, 5]))),
        ("gather", |x| x[0].clone().gather(2, IntTensor::from_data((0..3 * 5 * 4).map(|i| (i * 5 % 7) as i64).collect(), [3, 5, 4]))),
        ("select", |x| x[0].clone().select(1, IntTensor::from_ints(&[4, 0, 0, 2]))),
        ("cast f16 round trip", |x| x[0].clone().cast(DType::F16).cast(DType::F32)),
        ("cast bf16 round trip", |x| x[0].clone().mul_scalar(1e3).cast(DType::BF16).cast(DType::F32)),
        ("f16 operand", |x| x[0].clone().cast(DType::F16) + x[2].clone().cast(DType::F16)),
    ];
    for (name, f) in cases {
        let (cpu, gpu) = both(name, &ins, f);
        assert_bits(name, &gpu, &cpu);
    }
}

/// Matmul in matrixmultiply's order (fma chains closed every 256 along k): bit-exact against
/// CpuRef also past one k block, where a single chain would differ.
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_matmul_bit_exact_past_one_k_block() {
    for (i, sh) in [[1, 64, 1000, 33], [1, 300, 1000, 17], [2, 129, 513, 65], [1, 512, 512, 512], [1, 1024, 1024, 1024]].into_iter().enumerate() {
        let a = t(&[sh[0], sh[1], sh[2]], 40 + i as u64);
        let b = t(&[sh[0], sh[2], sh[3]], 50 + i as u64);
        let name = format!("matmul {sh:?}");
        let (cpu, gpu) = both(&name, &[a, b], |x| x[0].clone().matmul(x[1].clone()));
        eprintln!("{TAG} vs cpu_ref {name}: {} of {} bit-equal", gpu.iter().zip(&cpu).filter(|(x, y)| x.to_bits() == y.to_bits()).count(), cpu.len());
        assert_bits(&name, &gpu, &cpu);
    }
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_transcendentals_within_4_ulp() {
    let x = t(&[64, 257], 5);
    let pos = x.clone().abs().add_scalar(1e-3);
    type F = fn(&[Tensor]) -> Tensor;
    let cases: Vec<(&str, F, u64)> = vec![
        ("exp", |x| x[0].clone().exp(), 4),
        ("log", |x| x[1].clone().log(), 4),
        ("sqrt", |x| x[1].clone().sqrt(), 4),
        ("powf 0.5", |x| x[1].clone().powf_scalar(0.5), 4),
        ("powf 3", |x| x[1].clone().powf_scalar(3.0), 4),
        ("sigmoid", |x| nt::sigmoid(x[0].clone()), 4),
    ];
    for (name, f, tol) in cases {
        let (cpu, gpu) = both(name, &[x.clone(), pos.clone()], f);
        let u = max_ulps(&gpu, &cpu);
        let same = gpu.iter().zip(&cpu).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        eprintln!("{TAG} vs cpu_ref {name:<9}: max {u} ulp, {same}/{} bit-equal", cpu.len());
        assert!(u <= tol, "{name}: {u} ulp");
    }
    for (name, f) in [("softmax", nt::softmax as fn(Tensor, usize) -> Tensor), ("log_softmax", nt::log_softmax)] {
        let (cpu, gpu) = both(name, &[x.clone()], |x| f(x[0].clone(), 1));
        let d = gpu.iter().zip(&cpu).map(|(a, b)| ((a - b).abs() / (1.0 + b.abs())) as f64).fold(0.0, f64::max);
        eprintln!("{TAG} vs cpu_ref {name:<11}: max rel {d:.2e}");
        assert!(d <= 1e-6, "{name}: {d}");
    }
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_sort_and_topk_on_the_device() {
    let x = t(&[4, 9], 7);
    let (vr, ir) = x.clone().topk_with_indices(3, 1);
    let before = nt::transfer_counts().2;
    let (vm, im) = x.to(M).topk_with_indices(3, 1);
    // The sort now runs on the device (no host round trip).
    assert_eq!(nt::transfer_counts().2, before, "the sort runs on the device");
    assert_eq!(vm.device(), M);
    assert_bits("topk values", &vm.to_vec(), &vr.to_vec());
    assert_eq!(im.to_vec(), ir.to_vec());
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_transfers_and_errors() {
    let x = t(&[2, 3], 1);
    let m = x.clone().to(M);
    assert_eq!(m.device(), M);
    assert_bits("round trip", &m.clone().to(R).to_vec(), &x.to_vec());
    assert_eq!(m.clone().to(R).device(), R);
    let f64t = Tensor::from_f64s(vec![1.0, 2.0], [2], DType::F64);
    assert!(matches!(f64t.try_to(M), Err(TensorError::Unsupported(e)) if e.contains("f64")), "F64 is Unsupported on {TAG}");
    assert!(matches!(x.clone().try_to(OTHER), Err(TensorError::Unsupported(_))));
    assert!(matches!(m.clone().try_add(x.clone()), Err(TensorError::Unsupported(_))), "mixed devices are an error");
    let key = nt::gpu_key(M).unwrap();
    assert!(key.starts_with(KEY_PREFIX) && key.contains(KEY_MARK), "{key}");
    // A fresh tensor made while the GPU is the default lives on it.
    nt::set_default_device(M).unwrap();
    let z = Tensor::zeros([4]);
    nt::set_default_device(R).unwrap();
    assert_eq!(z.device(), M);
    assert!(matches!(&*z.storage, crate::tensor::Storage::Gpu(_)));
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_deterministic_twice() {
    let run = || {
        let a = t(&[8, 64, 96], 11).to(M);
        let b = t(&[8, 96, 64], 12).to(M);
        let y = nt::softmax(a.clone().matmul(b), 2).sum_dim(1).log();
        let z = nt::sigmoid(a).mean_dim(2);
        (y.to_vec(), z.to_vec())
    };
    let (y1, z1) = run();
    let (y2, z2) = run();
    assert_bits("determinism y", &y1, &y2);
    assert_bits("determinism z", &z1, &z2);
}

#[test]
fn the_reference_pin_refuses_the_gpu_without_initialising_it() {
    crate::tensor::tests::isolated_bodies(&here!("pinned_body"), 1);
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn pinned_body() {
    use crate::tensor::device::check_with;
    assert!(!initialized(), "no GPU before the pin");
    crate::tensor::pin_reference();
    assert!(matches!(t(&[2], 1).try_to(M), Err(TensorError::Unsupported(m)) if m.contains("pinned")));
    assert!(nt::set_default_device(M).is_err());
    assert!(matches!(check_with(M, true), Err(TensorError::Unsupported(m)) if m.contains("pinned")));
    assert!(nt::gpu_key(M).is_err(), "the backend refuses too");
    assert!(!initialized(), "a pinned process never initialises a GPU");
    pinned_extra();
    assert!(!initialized());
}

/// NaN-propagating max and argmax, and the stable descending sort (ties by position, every
/// NaN first whatever its sign bit, +0 before −0) on short, device-sorted and host-sorted lanes,
/// exactly as CpuRef.
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_nan_max_argmax_and_stable_sort_match_cpu_ref() {
    let nan = f32::NAN;
    let v = vec![1.0, nan, 5.0, 2.0, nan, 3.0, 9.0, 0.0, 4.0, nan, nan, 1.0, 1.0, 7.0, 7.0, 2.0];
    let x = Tensor::from_data(v, [4, 4]);
    for d in 0..2 {
        assert_eq!(x.clone().to(M).argmax(d).to_vec(), x.clone().argmax(d).to_vec(), "argmax {d}");
        let (c, g) = (x.clone().max_dim(d).to_vec(), x.clone().to(M).max_dim(d).to_vec());
        assert_eq!(max_ulps(&c, &g), 0, "max_dim {d}");
    }
    for v in [vec![1.0, nan, 2.0], vec![nan, 1.0, 2.0], vec![1.0, 2.0, nan], vec![1.0, 3.0, 2.0]] {
        let t = Tensor::from_data(v, [3]);
        assert_eq!(max_ulps(&t.clone().max().to_vec(), &t.to(M).max().to_vec()), 0, "max");
    }
    let levels = [-1e9f32, -0.5, 0.0, -0.0, 0.25, 1.0, nan, f32::from_bits(0xFFC0_0000), f32::NEG_INFINITY, f32::from_bits(0x7F80_0001), f32::INFINITY];
    for n in [16usize, 300, 4096, 5000] {
        let v: Vec<f32> = (0..2 * n).map(|i| levels[(i * 7 + i / 3) % levels.len()]).collect();
        let x = Tensor::from_data(v, [2, n]);
        let (cv, ci) = x.clone().sort_descending_with_indices(1);
        let (gv, gi) = x.to(M).sort_descending_with_indices(1);
        assert_eq!(ci.to_vec(), gi.to_vec(), "sort indices, lanes of {n}");
        assert_eq!(max_ulps(&cv.to_vec(), &gv.to_vec()), 0, "sort values, lanes of {n}");
    }
}
