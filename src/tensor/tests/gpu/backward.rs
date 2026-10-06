// Backward: one set of bodies for every GPU backend: backward on the device against
// CpuRef. Gradients equal CpuRef's bit for bit where every forward and backward op is exact
// (elementwise without transcendentals, reductions in CpuRef's order, matmul in matrixmultiply's
// order, indexing, deterministic scatters), and within 1e-4 (|Δ|/(1+|ref|)) through exp, ln,
// sqrt, pow and sigmoid. Transfers are on the tape: a CPU leaf moved to the device gets its
// gradient back on the CPU.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::tensor::tests::*;
use crate::tensor::{self as nt, BoolTensor, CpuMode, Device, IntTensor, Tensor};

const R: Device = Device::Cpu(CpuMode::Reference);

fn t(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_data(rnd(numel(shape), seed, -2.0, 2.0), shape.to_vec())
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| ((*x as f64 - *y as f64).abs()) / (1.0 + (*y as f64).abs())).fold(0.0, f64::max)
}

/// The loss's value and every leaf's gradient, with the leaves on `device`.
fn grads_on(inputs: &[Tensor], device: Device, f: &dyn Fn(&[Tensor]) -> Tensor) -> (Vec<f32>, Vec<Vec<f32>>) {
    // Leaves the loss does not use have no gradient (an empty vector on both devices).
    let leaves: Vec<Tensor> = inputs.iter().map(|x| x.clone().to(device).require_grad()).collect();
    let loss = f(&leaves);
    assert_eq!(loss.device(), device);
    let g = loss.backward();
    let grads = leaves
        .iter()
        .map(|l| match l.grad(&g) {
            Some(gr) => {
                assert_eq!(gr.device(), device, "gradients live on the leaf's device");
                gr.to_vec()
            }
            None => Vec::new(),
        })
        .collect();
    (loss.to_vec(), grads)
}

type F = fn(&[Tensor]) -> Tensor;

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_exact_gradients_bit_for_bit() {
    let ins = [t(&[3, 5, 7], 1), t(&[3, 1, 7], 2), t(&[3, 5, 7], 3).abs().add_scalar(0.5)];
    let cases: Vec<(&str, F)> = vec![
        ("add sub broadcast", |x| (x[0].clone() + x[1].clone() - x[2].clone()).sum()),
        ("mul div", |x| (x[0].clone() * x[1].clone() / x[2].clone()).sum()),
        ("scalars neg recip abs", |x| x[0].clone().mul_scalar(1.5).add_scalar(0.2).div_scalar(3.0).neg().abs().sum() + x[2].clone().recip().sum()),
        ("powi 2", |x| x[0].clone().powf_scalar(2.0).mean()),
        ("sum_dim mean_dim", |x| (x[0].clone().sum_dim(1) * x[1].clone()).mean_dim(2).sum() + x[2].clone().sum_dim(0).sum()),
        ("max_dim", |x| x[0].clone().max_dim(2).sum() + x[2].clone().max_dim(1).sum()),
        ("clamp", |x| x[0].clone().clamp(-0.5, 0.7).sum()),
        ("mask_fill", |x| x[0].clone().slice([0..3, 0..5, 0..5]).mask_fill(BoolTensor::tril_mask([5, 5], 0).unsqueeze_dim(0), -1e9).max_dim(2).sum()),
        ("views: swap, slice, reshape, expand", |x| (x[0].clone().swap_dims(1, 2).slice([0..3, 1..6, 0..4]).reshape([3, 20]) * x[1].clone().expand([3, 4, 7]).slice([0..3, 0..4, 0..5]).reshape([3, 20])).sum()),
        ("slice_assign", |x| x[0].clone().slice_assign([0..2, 1..3, 0..7], x[2].clone().slice([0..2, 0..2, 0..7]).mul_scalar(2.0)).sum_dim(2).max_dim(1).sum()),
        ("cat", |x| Tensor::cat(vec![x[0].clone(), x[2].clone()], 1).mul_scalar(0.5).sum_dim(1).mul(x[1].clone()).sum()),
        ("matmul batched", |x| x[0].clone().matmul(x[2].clone().swap_dims(1, 2)).mean()),
        ("matmul broadcast batch", |x| x[0].clone().matmul(x[1].clone().reshape([3, 7, 1])).sum() + x[0].clone().matmul(x[2].clone().slice([0..1, 0..5, 0..7]).swap_dims(1, 2)).sum()),
        ("gather", |x| x[0].clone().gather(2, IntTensor::from_data((0..3 * 5 * 4).map(|i| (i * 5 % 7) as i64).collect(), [3, 5, 4])).powf_scalar(2.0).sum()),
        ("select", |x| x[0].clone().select(1, IntTensor::from_ints(&[4, 0, 0, 2])).mul(x[1].clone()).sum()),
        ("topk (sort round trip)", |x| x[0].clone().topk_with_indices(3, 2).0.sum()),
        ("cast f16", |x| x[0].clone().cast(crate::tensor::DType::F16).cast(crate::tensor::DType::F32).mul(x[2].clone()).sum()),
    ];
    for (name, f) in cases {
        let (lc, gc) = grads_on(&ins, R, &f);
        let (lm, gm) = grads_on(&ins, M, &f);
        assert_bits(&format!("{name} loss"), &lm, &lc);
        for (i, (a, b)) in gm.iter().zip(&gc).enumerate() {
            assert_bits(&format!("{name} gradient {i}"), a, b);
        }
        eprintln!("{TAG} {name:<36} loss and gradients bit-equal");
    }
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_transcendental_gradients_within_1e4() {
    let ins = [t(&[6, 40], 5), t(&[6, 40], 6).abs().add_scalar(1e-2)];
    let cases: Vec<(&str, F)> = vec![
        ("exp", |x| x[0].clone().exp().sum()),
        ("log", |x| x[1].clone().log().sum()),
        ("sqrt", |x| x[1].clone().sqrt().sum()),
        ("powf 0.5 and 3", |x| x[1].clone().powf_scalar(0.5).sum() + x[1].clone().powf_scalar(3.0).sum()),
        ("sigmoid silu", |x| nt::silu(x[0].clone()).sum()),
        ("softmax", |x| nt::softmax(x[0].clone(), 1).mul(x[1].clone()).sum()),
        ("log_softmax", |x| nt::log_softmax(x[0].clone(), 1).mul(x[1].clone()).sum()),
    ];
    for (name, f) in cases {
        let (lc, gc) = grads_on(&ins, R, &f);
        let (lm, gm) = grads_on(&ins, M, &f);
        let dl = rel(&lm, &lc);
        let dg = gm.iter().zip(&gc).map(|(a, b)| rel(a, b)).fold(0.0, f64::max);
        eprintln!("{TAG} {name:<16} loss rel {dl:.2e}, gradients rel {dg:.2e}");
        assert!(dl <= 1e-5 && dg <= 1e-4, "{name}: loss {dl:e}, gradients {dg:e}");
    }
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_transfers_are_on_the_tape() {
    let x = t(&[4, 6], 7);
    let w = t(&[6, 3], 8);
    // CPU leaves; the computation runs on the device; the gradients come back to the CPU.
    let (xc, wc) = (x.clone().require_grad(), w.clone().require_grad());
    let loss = xc.clone().to(M).matmul(wc.clone().to(M)).exp().sum().to(R);
    assert_eq!(loss.device(), R);
    let g = loss.backward();
    let (gx, gw) = (xc.grad(&g).unwrap(), wc.grad(&g).unwrap());
    assert_eq!((gx.device(), gw.device()), (R, R), "the gradient returns to the leaf's device");
    // Against the same computation on the CPU.
    let (xr, wr) = (x.require_grad(), w.require_grad());
    let lr = xr.clone().matmul(wr.clone()).exp().sum();
    let gr = lr.backward();
    let d = rel(&gx.to_vec(), &xr.grad(&gr).unwrap().to_vec()).max(rel(&gw.to_vec(), &wr.grad(&gr).unwrap().to_vec()));
    eprintln!("{TAG} transfer on the tape: gradients rel {d:.2e}");
    assert!(d <= 1e-4);
    // An untracked transfer keeps no node; a leaf moved and marked stays a leaf on its device.
    assert!(!t(&[2], 1).to(M).is_tracked());
    assert!(t(&[2], 1).to(M).require_grad().is_require_grad());
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_backward_twice_byte_identical() {
    let run = || {
        let (a, b) = (t(&[8, 32, 48], 11).to(M).require_grad(), t(&[8, 48, 32], 12).to(M).require_grad());
        let loss = nt::log_softmax(a.clone().matmul(b.clone()), 2).max_dim(2).mean() + nt::sigmoid(a.clone()).sum();
        let g = loss.backward();
        [a.grad(&g).unwrap().to_vec(), b.grad(&g).unwrap().to_vec()].concat()
    };
    assert_bits("backward twice", &run(), &run());
}

/// The backward op bodies, in a fresh process.
#[test]
fn backward_against_cpu_ref() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 4);
}
