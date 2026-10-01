//! Per-operation parity: forward values and gradients bit-equal to burn 0.21's, recorded from
//! burn (fixtures/burn021-<os>-<arch>.txt). Where the standard formulations changed an operation,
//! only what did not change is compared with burn: the forward
//! values (`check*_fwd`) of div, div_scalar, log, sqrt, recip, mean, mean_dim, softmax,
//! log_softmax and the broadcast matmul; their gradients, and sigmoid and silu, are checked in
//! `standard` against f64 and finite differences.

use super::*;
use crate::tensor::{self as nt, BoolTensor};

const S3: [usize; 3] = [2, 5, 13];

fn x3(seed: u64) -> Vec<f32> {
    rnd(numel(&S3), seed, -3.0, 3.0)
}

#[test]
fn binary_ops_with_broadcasting() {
    let full = x3(1);
    let other = x3(2);
    check2("add", S3, &full, S3, &other, |a, b| a + b);
    check2("sub", S3, &full, S3, &other, |a, b| a - b);
    check2("mul", S3, &full, S3, &other, |a, b| a * b);
    check2_fwd("div", S3, &full, S3, &other, |a, b| a / b);
    for bs in [[2, 1, 13], [2, 5, 1], [1, 5, 13], [1, 1, 1]] {
        let y = rnd(numel(&bs), 3, 0.5, 2.0);
        check2(&format!("add {bs:?}"), S3, &full, bs, &y, |a, b| a + b);
        check2(&format!("sub {bs:?}"), S3, &full, bs, &y, |a, b| a - b);
        check2(&format!("mul {bs:?}"), S3, &full, bs, &y, |a, b| a * b);
        check2_fwd(&format!("div {bs:?}"), S3, &full, bs, &y, |a, b| a / b);
        check2(&format!("mul rev {bs:?}"), bs, &y, S3, &full, |a, b| a * b);
    }
}

#[test]
fn scalar_and_unary_ops() {
    let x = x3(4);
    let pos = rnd(numel(&S3), 5, 0.01, 4.0);
    check1("add_scalar", S3, &x, |t| t + 1.0);
    check1("sub_scalar", S3, &x, |t| t - 1.0);
    check1("mul_scalar", S3, &x, |t| t.mul_scalar(0.3));
    check1_fwd("div_scalar", S3, &x, |t| t.div_scalar(7.0));
    check1("neg", S3, &x, |t| -t);
    check1("exp", S3, &x, |t| t.exp());
    check1_fwd("log", S3, &pos, |t| t.log());
    check1_fwd("sqrt", S3, &pos, |t| t.sqrt());
    check1_fwd("recip", S3, &pos, |t| t.recip());
    check1("abs", S3, &x, |t| t.abs());
    check1("powf 2", S3, &x, |t| t.powf_scalar(2.0));
    check1("powf 0.5", S3, &pos, |t| t.powf_scalar(0.5));
    check1("powf 3", S3, &x, |t| t.powf_scalar(3.0));
    check1("clamp_min", S3, &x, |t| t.clamp_min(0.5));
    check1("clamp_max", S3, &x, |t| t.clamp_max(0.0));
    check1("clamp", S3, &x, |t| t.clamp(-1.0, 1.0));
    check1_fwd("rms chain", S3, &x, |t| (t.clone().powf_scalar(2.0).mean_dim(2) + 1e-6).sqrt().recip() * t);
}

#[test]
fn reductions() {
    for (seed, shape) in [(6u64, S3), (7, [3, 1, 40]), (8, [4, 17, 1]), (9, [1, 9, 8])] {
        let x = rnd(numel(&shape), seed, -2.0, 2.0);
        check1("sum", shape, &x, |t| t.sum());
        check1_fwd("mean", shape, &x, |t| t.mean());
        for d in 0..3 {
            check1(&format!("sum_dim {d} {shape:?}"), shape, &x, |t| t.sum_dim(d));
            check1_fwd(&format!("mean_dim {d} {shape:?}"), shape, &x, |t| t.mean_dim(d));
            check1(&format!("max_dim {d} {shape:?}"), shape, &x, |t| t.max_dim(d));
            let ours = Tensor::from_data(x.clone(), shape).argmax(d).to_vec();
            fixture::i64s(&format!("argmax {d} {shape:?}"), &ours);
        }
    }
    // Long lanes exercise the eight partial sums and their tail.
    let long = rnd(3 * 1001, 10, -1.0, 1.0);
    check1("sum_dim long", [3, 1001], &long, |t| t.sum_dim(1));
    check1("sum long", [3, 1001], &long, |t| t.sum());
    // Ties: argmax/max_dim take the first maximum.
    let ties = vec![1.0, 3.0, 3.0, 2.0, 5.0, 5.0, 5.0, 0.0];
    fixture::i64s("argmax ties", Tensor::from_data(ties.clone(), [2, 4]).argmax(1).to_vec());
}

#[test]
fn layout_ops() {
    let x = x3(11);
    check1("reshape", S3, &x, |t| t.reshape([10, 13]));
    check1("swap_dims", S3, &x, |t| t.swap_dims(1, 2));
    check1("swap_dims 0 2", S3, &x, |t| t.swap_dims(0, 2));
    check1("unsqueeze+expand", S3, &x, |t| t.unsqueeze_dim(1).expand([2, 3, 5, 13]));
    check1("expand", [2, 1, 13], &x[..26], |t| t.expand([2, 5, 13]));
    check1("slice", S3, &x, |t| t.slice([0..2, 1..4, 3..13]));
    let v = rnd(2 * 3 * 4, 12, -1.0, 1.0);
    check2("slice_assign", S3, &x, [2, 3, 4], &v, |a, b| a.slice_assign([0..2, 1..4, 5..9], b));
    let y = rnd(2 * 5 * 3, 13, -1.0, 1.0);
    check2("cat", S3, &x, [2, 5, 3], &y, |a, b| Tensor::cat(vec![a, b], 2));
    let z = rnd(2 * 2 * 13, 14, -1.0, 1.0);
    check2("cat dim 1", S3, &x, [2, 2, 13], &z, |a, b| Tensor::cat(vec![a.clone(), b, a], 1));
}

#[test]
fn matmul() {
    let (a, b) = (rnd(3 * 7 * 64, 15, -1.0, 1.0), rnd(3 * 64 * 33, 16, -1.0, 1.0));
    check2("matmul rank 3", [3, 7, 64], &a, [3, 64, 33], &b, |x, y| x.matmul(y));
    let w = rnd(64 * 33, 17, -1.0, 1.0);
    check2("matmul broadcast batch", [3, 7, 64], &a, [1, 64, 33], &w, |x, y| x.matmul(y));
    let (c, d) = (rnd(2 * 3 * 5 * 300, 18, -1.0, 1.0), rnd(2 * 3 * 300 * 4, 19, -1.0, 1.0));
    check2("matmul rank 4, k > block", [2, 3, 5, 300], &c, [2, 3, 300, 4], &d, |x, y| x.matmul(y));
    let (e, f) = (rnd(2 * 4 * 1 * 6, 20, -1.0, 1.0), rnd(2 * 1 * 6 * 5, 21, -1.0, 1.0));
    check2_fwd("matmul [b,1,k]×[1,k,n] rule", [2, 4, 1, 6], &e, [2, 1, 6, 5], &f, |x, y| x.matmul(y));
    check2("matmul transposed rhs", [3, 7, 64], &a, [3, 33, 64], &b, |x, y| x.matmul(y.swap_dims(1, 2)));
}

#[test]
fn indexing_ops() {
    let x = x3(22);
    // Causal mask with −∞, then softmax (the attention pattern).
    let t = 13;
    let sq = rnd(4 * t * t, 23, -2.0, 2.0);
    check1_fwd(
        "mask_fill causal + softmax",
        [4, t, t],
        &sq,
        |z| nt::softmax(z.mask_fill(BoolTensor::tril_mask([t, t], 0).unsqueeze_dim(0).expand([4, t, t]), f64::NEG_INFINITY), 2),
    );
    fixture::bools("tril_mask", BoolTensor::tril_mask([5, 5], 0).to_vec());
    // gather along 1 and 2 with repeated indices.
    let idx1: Vec<i64> = (0..2 * 9 * 13).map(|i| ((i * 7 + i / 13) % 5) as i64).collect();
    check1("gather dim 1", S3, &x, |z| z.gather(1, our_ints(&idx1, [2, 9, 13])));
    let idx2: Vec<i64> = (0..2 * 5 * 4).map(|i| ((i * 5) % 13) as i64).collect();
    check1("gather dim 2", S3, &x, |z| z.gather(2, our_ints(&idx2, [2, 5, 4])));
    let sel: Vec<i64> = vec![3, 0, 3, 12];
    check1("select", S3, &x, |z| z.select(2, our_ints(&sel, [4])));
    // topk with ties (many equal −1e9 entries).
    let mut tk = rnd(3 * 4 * 16, 24, -1.0, 1.0);
    for i in (0..tk.len()).step_by(3) {
        tk[i] = -1e9;
    }
    for i in (1..tk.len()).step_by(7) {
        tk[i] = 0.25;
    }
    for kk in [1, 4, 16] {
        let (vo, io) = Tensor::from_data(tk.clone(), [3, 4, 16]).topk_with_indices(kk, 2);
        fixture::f32s(&format!("topk {kk} values"), vo.as_slice());
        fixture::i64s(&format!("topk {kk} indices"), io.to_vec());
        check1(&format!("topk {kk} gradient"), [3, 4, 16], &tk, |z| z.topk_with_indices(kk, 2).0);
    }
}

#[test]
fn activations() {
    let x = x3(25);
    for d in 0..3 {
        check1_fwd(&format!("softmax {d}"), S3, &x, |z| nt::softmax(z, d));
        check1_fwd(&format!("log_softmax {d}"), S3, &x, |z| nt::log_softmax(z, d));
    }
    let big = rnd(numel(&S3), 26, -60.0, 60.0);
    check1_fwd("log_softmax large", S3, &big, |z| nt::log_softmax(z, 2));
}

#[test]
fn detach_and_masked_stop_gradient() {
    // r·m + detach(r)·(1 − m): the gradient flows only where the mask is 1.
    let x = x3(27);
    let m: Vec<f32> = (0..2 * 5).map(|i| (i % 2) as f32).collect();
    check1(
        "masked",
        S3,
        &x,
        |r| {
            let m = Tensor::from_data(m.clone(), [2, 5, 1]);
            r.clone().exp() * m.clone() + r.exp().detach() * (m.neg() + 1.0)
        },
    );
}

#[test]
fn int_ops() {
    let v: Vec<i64> = (0..2 * 3 * 7).map(|i| (i * 13 % 40) as i64).collect();
    let o = our_ints(&v, [2, 3, 7]);
    fixture::i64s("int remainder_scalar", o.clone().remainder_scalar(9).to_vec());
    fixture::i64s("int div_scalar", o.clone().div_scalar(9).to_vec());
    fixture::i64s("int mul_scalar", o.clone().mul_scalar(3).to_vec());
    fixture::i64s("int add", (o.clone() + o.clone()).to_vec());
    fixture::f32s("one_hot", o.clone().reshape([6, 7]).one_hot(40).float().as_slice());
    fixture::f32s("not_equal_elem", o.clone().not_equal_elem(0).float().as_slice());
    fixture::f32s("greater_equal_elem", o.clone().greater_equal_elem(20).float().as_slice());
    fixture::i64s("int cat", IntTensor::cat(vec![o.clone().slice([0..2, 0..3, 0..1]), o.clone()], 2).to_vec());
    let gi: Vec<i64> = (0..2 * 3 * 4).map(|i| (i % 7) as i64).collect();
    fixture::i64s("int gather", o.clone().gather(2, our_ints(&gi, [2, 3, 4])).to_vec());
    fixture::i64s("int unsqueeze+expand", o.clone().unsqueeze_dim(3).expand([2, 3, 7, 5]).to_vec());
}

/// The checks above are meaningful: a plain left-to-right sum and an unfused matmul already
/// differ from the reference in the last bits, so matching it requires its exact order.
#[test]
fn bit_checks_detect_a_different_order() {
    let long = rnd(3 * 1001, 10, -1.0, 1.0);
    let naive: Vec<f32> = (0..3).map(|r| long[r * 1001..(r + 1) * 1001].iter().fold(0.0f32, |a, b| a + b)).collect();
    let b = Tensor::from_data(long.clone(), [3, 1001]).sum_dim(1).to_vec();
    assert!(naive.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()));
    let (a, w) = (rnd(7 * 64, 15, -1.0, 1.0), rnd(64 * 33, 16, -1.0, 1.0));
    let mut c = vec![0f32; 7 * 33];
    for i in 0..7 {
        for j in 0..33 {
            let mut s = 0f32;
            for k in 0..64 {
                s += a[i * 64 + k] * w[k * 33 + j];
            }
            c[i * 33 + j] = s;
        }
    }
    let cb = Tensor::from_data(a, [1, 7, 64]).matmul(Tensor::from_data(w, [1, 64, 33])).to_vec();
    assert!(c.iter().zip(&cb).filter(|(x, y)| x.to_bits() != y.to_bits()).count() > 0);
}
