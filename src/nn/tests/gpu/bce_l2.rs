// `binary_cross_entropy_with_logits` and `l2_normalize` on the device: values against CpuRef
// within 1e-5 and gradients within 1e-4 (|Δ|/(1+|ref|)), finite at |x| = 1e4, a zero row finite
// with the upstream gradient / ε, unit norms within 1e-6, byte-identical run to run.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::nn::tests::*;
use crate::nn::*;
use crate::tensor::{self as nt, CpuMode, DType, Device, Tensor};

const R: Device = Device::Cpu(CpuMode::Reference);

fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| ((*x as f64 - *y as f64).abs()) / (1.0 + (*y as f64).abs())).fold(0.0, f64::max)
}

fn t32(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v.to_vec(), shape.to_vec(), DType::F32)
}

/// Loss, logit gradient and target gradient of BCE with logits on `device`, every option and
/// reduction, in order.
fn bce_run(x: &[f64], y: &[f64], w: &[f64], p: &[f64], shape: [usize; 3], device: Device) -> Vec<Vec<f32>> {
    nt::set_default_device(device).unwrap();
    let mut out = Vec::new();
    for (weight, pos) in [(false, false), (true, false), (false, true), (true, true)] {
        let bw = BceWeights { weight: weight.then(|| t32(w, &shape)), pos_weight: pos.then(|| t32(p, &[shape[2]])) };
        for r in [Reduction::None, Reduction::Mean, Reduction::Sum] {
            let xt = t32(x, &shape).require_grad();
            let yt = t32(y, &shape).require_grad();
            let l = binary_cross_entropy_with_logits(xt.clone(), &yt, Some(&bw), r);
            assert_eq!(l.device(), device);
            let up = t32(&rnd(l.numel(), 7, -1.0, 1.0), l.shape());
            let g = (l.clone() * up).sum().backward();
            out.extend([l.to_vec(), xt.grad(&g).unwrap().to_vec(), yt.grad(&g).unwrap().to_vec()]);
        }
    }
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_bce_with_logits_on_the_device() {
    let shape = [3, 16, 8];
    let n = shape.iter().product();
    let mut x = rnd(n, 700, -12.0, 12.0);
    // Large logits and zeros.
    for (i, v) in [1e4, -1e4, 1e3, -1e3, 0.0, 0.0, 88.0, -88.0].iter().enumerate() {
        x[i * 37] = *v;
    }
    let y: Vec<f64> = rnd(n, 701, 0.0, 1.0).into_iter().enumerate().map(|(i, u)| if i % 3 == 0 { u.round() } else { u }).collect();
    let w = rnd(n, 702, 0.2, 2.0);
    let p = rnd(shape[2], 703, 0.5, 4.0);
    let cpu = bce_run(&x, &y, &w, &p, shape, R);
    let dev = bce_run(&x, &y, &w, &p, shape, M);
    let mut worst = [0.0f64; 2];
    for (i, (a, c)) in dev.iter().zip(&cpu).enumerate() {
        assert!(a.iter().all(|v| v.is_finite()), "{TAG} BCE output {i}: not finite");
        let (k, tol) = if i % 3 == 0 { (0, 1e-5) } else { (1, 1e-4) };
        let e = rel(a, c);
        assert!(e <= tol, "{TAG} BCE output {i}: {e:e} from CpuRef");
        worst[k] = worst[k].max(e);
    }
    let again = bce_run(&x, &y, &w, &p, shape, M);
    for (i, (a, c)) in dev.iter().zip(&again).enumerate() {
        assert_bits(&format!("{TAG} BCE run-to-run {i}"), a, c);
    }
    println!("{TAG} BCE with logits: losses within {:.1e} and gradients within {:.1e} of CpuRef (|x| up to 1e4, finite), byte-identical run to run", worst[0], worst[1]);
}

/// Output and input gradient of `l2_normalize` along `dim` on `device`.
fn l2_run(x: &[f64], shape: &[usize], dim: usize, eps: f64, device: Device) -> (Vec<f32>, Vec<f32>) {
    nt::set_default_device(device).unwrap();
    let xt = t32(x, shape).require_grad();
    let y = l2_normalize(&xt, dim, eps);
    assert_eq!(y.device(), device);
    let up = t32(&rnd(y.numel(), 8, -1.0, 1.0), y.shape());
    let g = (y.clone() * up).sum().backward();
    let out = (y.to_vec(), xt.grad(&g).unwrap().to_vec());
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_l2_normalize_on_the_device() {
    let shape = [3, 12, 64];
    let n: usize = shape.iter().product();
    let d = shape[2];
    let mut x: Vec<f64> = rnd(n, 710, -1.0, 1.0).iter().enumerate().map(|(j, v)| v * 10f64.powi((j / d) as i32 % 7 - 3)).collect();
    // Row 5 of seed 1 is zero.
    let z = (shape[1] + 5) * d;
    x[z..z + d].iter_mut().for_each(|v| *v = 0.0);
    let mut worst = [0.0f64; 2];
    for dim in 0..3 {
        let cpu = l2_run(&x, &shape, dim, L2_EPS, R);
        let dev = l2_run(&x, &shape, dim, L2_EPS, M);
        assert!(dev.0.iter().chain(&dev.1).all(|v| v.is_finite()), "{TAG} l2_normalize dim {dim}: not finite");
        let (e0, e1) = (rel(&dev.0, &cpu.0), rel(&dev.1, &cpu.1));
        assert!(e0 <= 1e-5 && e1 <= 1e-4, "{TAG} l2_normalize dim {dim}: output {e0:e}, gradient {e1:e} from CpuRef");
        worst = [worst[0].max(e0), worst[1].max(e1)];
        let again = l2_run(&x, &shape, dim, L2_EPS, M);
        assert_bits(&format!("{TAG} l2_normalize run-to-run output dim {dim}"), &dev.0, &again.0);
        assert_bits(&format!("{TAG} l2_normalize run-to-run gradient dim {dim}"), &dev.1, &again.1);
    }
    // Along the last axis: the zero row is zero with gradient upstream / ε; the others have unit norm.
    for eps in [L2_EPS, 1e-6] {
        let (y, g) = l2_run(&x, &shape, 2, eps, M);
        let up = rnd(n, 8, -1.0, 1.0);
        assert!(y[z..z + d].iter().all(|v| *v == 0.0), "{TAG}: zero row output");
        for k in z..z + d {
            let want = up[k] as f32 as f64 / eps;
            assert!((g[k] as f64 - want).abs() <= 1e-6 * want.abs(), "{TAG} ε {eps:e}: zero-row gradient {:e} vs {want:e}", g[k]);
        }
        let mut far: f64 = 0.0;
        for row in (0..n / d).filter(|&r| r * d != z) {
            let n2: f64 = y[row * d..(row + 1) * d].iter().map(|v| *v as f64 * *v as f64).sum();
            far = far.max((n2.sqrt() - 1.0).abs());
        }
        assert!(far <= 1e-6, "{TAG} ε {eps:e}: a non-zero row's norm is {far:e} from 1");
        println!("{TAG} l2_normalize ε {eps:e}: unit norms within {far:.1e}, zero row finite with gradient upstream/ε");
    }
    println!("{TAG} l2_normalize: outputs within {:.1e} and gradients within {:.1e} of CpuRef on every axis, byte-identical run to run", worst[0], worst[1]);
}

/// The BCE and l2_normalize bodies, in a fresh process.
#[test]
fn bce_l2_ops_on_the_device() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 2);
}
