// `nn::rl` on the device: every loss's value and logit gradient against CpuRef (values within
// 1e-5, gradients within 1e-4, |Δ|/(1+|ref|)) with ±1e4 and −∞ logits and masked positions,
// masked positions with exactly zero gradient, byte-identical run to run; the sampler's tokens
// against CpuRef's for the same stream and run to run; group advantages on device tensors.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::nn::tests::*;
use crate::nn::*;
use crate::tensor::{self as nt, CpuMode, DType, Device, IntTensor, Tensor};

const R: Device = Device::Cpu(CpuMode::Reference);

fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| if x == y { 0.0 } else { ((*x as f64 - *y as f64).abs()) / (1.0 + (*y as f64).abs()) }).fold(0.0, f64::max)
}

fn t32(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v.to_vec(), shape.to_vec(), DType::F32)
}

const SH: [usize; 3] = [2, 8, 12];
const V: usize = 24;

struct Data {
    logits: Vec<f64>,
    refl: Vec<f64>,
    tokens: Vec<i64>,
    mask: Vec<f64>,
    old: Vec<f64>,
    adv: Vec<f64>,
    lr: Vec<f64>,
}

fn data() -> Data {
    let n: usize = SH.iter().product();
    let mut logits = rnd(n * V, 950, -6.0, 6.0);
    let mut refl = rnd(n * V, 951, -6.0, 6.0);
    let tokens = rnd_ints(n, 952, V as i64);
    // ±1e4 at some entries and −∞ (in both distributions) at entries that are not the token.
    for i in (0..n).step_by(5) {
        let t = tokens[i] as usize;
        let (a, b) = ((t + 1) % V, (t + 2) % V);
        logits[i * V + a] = f64::NEG_INFINITY;
        refl[i * V + a] = f64::NEG_INFINITY;
        logits[i * V + b] = if i % 2 == 0 { 1e4 } else { -1e4 };
    }
    let mut mask: Vec<f64> = rnd(n, 953, 0.0, 1.0).into_iter().map(|u| if u < 0.75 { 1.0 } else { 0.0 }).collect();
    let t = SH[2];
    mask[3 * t..4 * t].iter_mut().for_each(|m| *m = 0.0);
    let old = rnd(n, 954, -8.0, -0.5);
    let adv = rnd(n, 955, -2.0, 2.0);
    // The reference log-probabilities of the tokens: the policy's (f64 host log-softmax) plus δ
    // in [−1.5, 1.5], as on policy (k3's exp(δ) overflows f32 for δ above about 88).
    let lr: Vec<f64> = (0..n)
        .zip(rnd(n, 956, -1.5, 1.5))
        .map(|(i, dl)| {
            let row = &logits[i * V..(i + 1) * V];
            let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = row.iter().map(|a| (a - mx).exp()).sum::<f64>().ln() + mx;
            row[tokens[i] as usize] - lse + dl
        })
        .collect();
    Data { logits, refl, tokens, mask, old, adv, lr }
}

/// Each loss's value and logit gradient on `device` (f32), in order, with the names.
fn losses_run(d: &Data, device: Device) -> Vec<(String, Vec<f32>)> {
    nt::set_default_device(device).unwrap();
    let lsh = [SH[0], SH[1], SH[2], V];
    let tok = IntTensor::from_data(d.tokens.clone(), SH);
    let m = t32(&d.mask, &SH);
    let (old, adv, lr, refl) = (t32(&d.old, &SH), t32(&d.adv, &SH), t32(&d.lr, &SH), t32(&d.refl, &lsh));
    let advs = t32(&d.adv[..SH[0] * SH[1]], &SH[..2]);
    let cases: Vec<(&str, Box<dyn Fn(Tensor) -> Tensor>)> = vec![
        ("token_log_probs", Box::new(|x| token_log_probs(x, &tok).mask_fill(m.clone().equal_elem(0.0), 0.0))),
        ("pg token-mean", Box::new(|x| policy_gradient_loss(token_log_probs(x, &tok), &adv, Some(&m), TokenReduction::TokenMean))),
        ("pg per-sequence weights, none", Box::new(|x| policy_gradient_loss(token_log_probs(x, &tok), &advs, Some(&m), TokenReduction::None))),
        ("ppo sum", Box::new(|x| clipped_policy_loss(token_log_probs(x, &tok), &old, &adv, Some(&m), Clip { low: 0.2, high: 0.28 }, TokenReduction::Sum))),
        ("kl sequence-mean", Box::new(|x| categorical_kl(x, &refl, Some(&m), TokenReduction::SequenceMean))),
        ("k1 per-sequence", Box::new(|x| kl_estimate(token_log_probs(x, &tok), &lr, KlEstimator::K1, Some(&m), TokenReduction::PerSequence))),
        ("k2 per-sequence", Box::new(|x| kl_estimate(token_log_probs(x, &tok), &lr, KlEstimator::K2, Some(&m), TokenReduction::PerSequence))),
        ("k3 per-sequence", Box::new(|x| kl_estimate(token_log_probs(x, &tok), &lr, KlEstimator::K3, Some(&m), TokenReduction::PerSequence))),
        ("entropy token-mean", Box::new(|x| categorical_entropy(x, Some(&m), TokenReduction::TokenMean))),
    ];
    let mut out = Vec::new();
    for (name, f) in &cases {
        let x = t32(&d.logits, &lsh).require_grad();
        let l = f(x.clone());
        assert_eq!(l.device(), device, "{name}");
        let up = t32(&rnd(l.numel(), 9, -1.0, 1.0), l.shape());
        let g = (l.clone() * up).sum().backward();
        out.push((format!("{name} value"), l.to_vec()));
        out.push((format!("{name} gradient"), x.grad(&g).unwrap().to_vec()));
    }
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_rl_losses_on_the_device() {
    let d = data();
    let cpu = losses_run(&d, R);
    let dev = losses_run(&d, M);
    let mut worst = [0.0f64; 2];
    for ((name, a), (_, c)) in dev.iter().zip(&cpu) {
        assert!(a.iter().all(|v| v.is_finite()), "{TAG} {name}: not finite");
        let (k, tol) = if name.ends_with("value") { (0, 1e-5) } else { (1, 1e-4) };
        let e = rel(a, c);
        assert!(e <= tol, "{TAG} {name}: {e:e} from CpuRef");
        worst[k] = worst[k].max(e);
        if name.ends_with("gradient") {
            for i in (0..d.mask.len()).filter(|&i| d.mask[i] == 0.0) {
                assert!(a[i * V..(i + 1) * V].iter().all(|g| *g == 0.0), "{TAG} {name}: masked position {i} has a gradient");
            }
        }
    }
    let again = losses_run(&d, M);
    for ((name, a), (_, c)) in dev.iter().zip(&again) {
        assert_bits(&format!("{TAG} {name} run-to-run"), a, c);
    }
    // Group advantages of a device tensor stay on the device, equal to the host's.
    nt::set_default_device(M).unwrap();
    let rw = t32(&rnd(4 * 8, 957, 0.0, 1.0), &[4, 8]);
    let a = group_advantages_tensor(&rw, GroupNorm::Std { eps: 1e-6, unbiased: true });
    assert_eq!(a.device(), M);
    let host = group_advantages(&rw.to_vec_f64(), 8, GroupNorm::Std { eps: 1e-6, unbiased: true });
    nt::set_default_device(R).unwrap();
    assert!(a.to_vec_f64().iter().zip(&host).all(|(x, y)| (x - y).abs() <= 1e-6 * (1.0 + y.abs())), "{TAG} group advantages");
    println!("{TAG} rl losses: values within {:.1e} and gradients within {:.1e} of CpuRef (±1e4 and −∞ logits, masked positions zero), byte-identical run to run", worst[0], worst[1]);
}

/// Tokens from `sample_categorical` on `device` for a fixed stream: temperatures 1 and 0.7 with
/// an additive mask, and greedy.
fn sample_run(logits: &[f64], mask: &[f64], shape: [usize; 3], device: Device) -> Vec<Vec<i64>> {
    nt::set_default_device(device).unwrap();
    let x = t32(logits, &shape);
    let mk = t32(mask, &[1, 1, shape[2]]);
    let mut out = Vec::new();
    for (tau, root) in [(1.0, 11), (0.7, 12), (0.0, 13)] {
        let t = sample_categorical(&x, tau, Some(&mk), &mut crate::rng::derive(root, "device sampler"));
        out.push(t.to_vec());
    }
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_rl_sampler_on_the_device() {
    let shape = [4, 128, 96];
    let n = shape[0] * shape[1];
    let logits = rnd(n * shape[2], 960, -3.0, 3.0);
    let mask: Vec<f64> = (0..shape[2]).map(|k| if k % 7 == 3 { f64::NEG_INFINITY } else { 0.0 }).collect();
    let cpu = sample_run(&logits, &mask, shape, R);
    let dev = sample_run(&logits, &mask, shape, M);
    let mut differ = 0;
    for (a, c) in dev.iter().zip(&cpu) {
        assert_eq!(a.len(), n);
        assert!(a.iter().all(|t| *t as usize % 7 != 3), "{TAG}: an excluded token was drawn");
        differ += a.iter().zip(c).filter(|(x, y)| x != y).count();
    }
    assert_eq!(dev[2], cpu[2], "{TAG}: greedy tokens differ from CpuRef");
    assert!(differ * 1000 <= 3 * n, "{TAG}: {differ} of {} sampled tokens differ from CpuRef", 3 * n);
    assert_eq!(sample_run(&logits, &mask, shape, M), dev, "{TAG}: sampler run to run");
    println!("{TAG} rl sampler: {differ} of {} tokens differ from CpuRef for the same stream (greedy equal), none excluded, identical run to run", 3 * n);
}

/// The rl bodies, in a fresh process.
#[test]
fn rl_ops_on_the_device() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 2);
}
