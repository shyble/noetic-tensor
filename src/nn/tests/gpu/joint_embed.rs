// The joint-embedding additions, one set of bodies for every GPU backend: set attention with the key padding
// mask and the pool, TargetEma, VICReg, masked group InfoNCE and the effective-rank monitor on the
// device. Values against CpuRef within 1e-5 and gradients within 1e-4 (|Δ|/(1+|ref|)); padding
// and permutation invariance within fp32 1e-6 on the device; masked entries get exactly zero
// gradient; everything is byte-identical run to run, and a small joint-embedding training run gives the same
// state hash twice.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::nn::tests::*;
use crate::nn::*;
use crate::tensor::{self as nt, BoolTensor, CpuMode, DType, Device, Tensor};
use std::cell::RefCell;

const R: Device = Device::Cpu(CpuMode::Reference);

fn rel(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| ((*x as f64 - *y as f64).abs()) / (1.0 + (*y as f64).abs())).fold(0.0, f64::max)
}

fn vars_on(vars: &VarMap, device: Device) -> VarMap {
    let mut m = VarMap::new();
    for (name, t) in vars.iter() {
        m.insert(name, t.clone().to(device)).unwrap();
    }
    m
}

fn t32(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v.to_vec(), shape.to_vec(), DType::F32)
}

// ------------------------------------------------------------------------- set attention

/// Sets of sizes `sizes` (same per seed) padded to t; pads at the end or spread; pads hold
/// `fill`-scaled garbage. Returns the input, the mask and per (seed, set) the example positions.
fn sets(s: usize, t: usize, d: usize, sizes: &[usize], interleave: bool, fill: f64, seed: u64) -> (Vec<f64>, Vec<bool>, Vec<Vec<usize>>) {
    let b = sizes.len();
    let mut x: Vec<f64> = rnd(s * b * t * d, seed, -1.0, 1.0).into_iter().map(|v| v * fill).collect();
    let mut pad = vec![true; s * b * t];
    let mut pos = Vec::new();
    for k in 0..s {
        for (r, &n) in sizes.iter().enumerate() {
            let p: Vec<usize> = if interleave { (0..n).map(|i| (i * t) / n + ((r + k) % ((t / n).max(1)))).collect() } else { (0..n).collect() };
            let ex = rnd(n * d, seed + 1 + (k * b + r) as u64, -1.0, 1.0);
            for (j, &q) in p.iter().enumerate() {
                let row = (k * b + r) * t + q;
                pad[row] = false;
                x[row * d..(row + 1) * d].copy_from_slice(&ex[j * d..(j + 1) * d]);
            }
            pos.push(p);
        }
    }
    (x, pad, pos)
}

/// Outputs, pooled, input gradient and var gradients of the padded set attention on `device`.
fn set_run(vars: &VarMap, d: usize, h: usize, x: &[f64], pad: &[bool], dims: [usize; 3], device: Device) -> Vec<Vec<f32>> {
    let [s, b, t] = dims;
    nt::set_default_device(device).unwrap();
    let lifted = vars_on(vars, device).lifted();
    let a = attention(AttentionConfig::non_causal(d, h), &VarBuilder::from_varmap(&lifted)).unwrap();
    let pad_t = BoolTensor::from_data(pad.to_vec(), [s, b, t]);
    let xt = t32(x, &[s, b * t, d]).require_grad();
    let y = a.forward_with(&xt, t, Some(&pad_t));
    assert_eq!(y.device(), device);
    let pooled = masked_mean_pool(&y, t, Some(&pad_t));
    let w = t32(&rnd(s * b * d, 7, -1.0, 1.0), &[s, b, d]);
    let g = (pooled.clone() * w).sum().backward();
    let mut out = vec![y.to_vec(), pooled.to_vec(), xt.grad(&g).unwrap().to_vec()];
    out.extend(lifted.grads(&g).into_iter().map(|g| g.unwrap().to_vec()));
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_set_attention_on_the_device() {
    let (s, t, d, h) = (3, 10, 16, 4);
    let sizes = [2, 5, 10, 3, 7];
    let b = sizes.len();
    let map = RefCell::new(VarMap::new());
    let _ = attention(AttentionConfig::non_causal(d, h), &VarBuilder::init(&map, s, 41)).unwrap();
    let vars = map.into_inner();
    let (x, pad, pos) = sets(s, t, d, &sizes, true, 2.0, 100);
    let dims = [s, b, t];
    // Against CpuRef.
    let cpu = set_run(&vars, d, h, &x, &pad, dims, R);
    let dev = set_run(&vars, d, h, &x, &pad, dims, M);
    let names = ["outputs", "pooled", "input gradient"];
    for (i, (a, c)) in dev.iter().zip(&cpu).enumerate() {
        let tol = if i < 2 { 1e-5 } else { 1e-4 };
        let e = rel(a, c);
        assert!(e <= tol, "{TAG} set attention {}: {e:e} from CpuRef", names.get(i).copied().unwrap_or("var gradient"));
    }
    // Masked rows: exactly zero gradient.
    let gx = &dev[2];
    for (row, p) in pad.iter().enumerate() {
        if *p {
            assert!(gx[row * d..(row + 1) * d].iter().all(|v| v.to_bits() == 0), "{TAG}: pad row {row} has a gradient");
        }
    }
    // Run to run, byte-identical.
    let again = set_run(&vars, d, h, &x, &pad, dims, M);
    for (i, (a, c)) in dev.iter().zip(&again).enumerate() {
        assert_bits(&format!("{TAG} set attention run-to-run {i}"), a, c);
    }
    // Padded equals unpadded on the device.
    nt::set_default_device(M).unwrap();
    let a = attention(AttentionConfig::non_causal(d, h), &VarBuilder::from_varmap(&vars_on(&vars, M))).unwrap();
    let mut worst: f64 = 0.0;
    for (r, &n) in sizes.iter().enumerate() {
        let alone: Vec<f64> = (0..s).flat_map(|k| pos[k * b + r].iter().flat_map(|&q| { let row = (k * b + r) * t + q; x[row * d..(row + 1) * d].to_vec() }).collect::<Vec<_>>()).collect();
        let ya = a.forward(&t32(&alone, &[s, n, d]), n);
        let pa = masked_mean_pool(&ya, n, None).to_vec();
        let ya = ya.to_vec();
        for k in 0..s {
            let i = k * b + r;
            let ours: Vec<f32> = pos[i].iter().flat_map(|&q| dev[0][(i * t + q) * d..(i * t + q + 1) * d].to_vec()).collect();
            worst = worst.max(rel(&ours, &ya[k * n * d..(k + 1) * n * d])).max(rel(&dev[1][i * d..(i + 1) * d], &pa[k * d..(k + 1) * d]));
        }
    }
    assert!(worst <= 1e-6, "{TAG}: padded vs unpadded differ by {worst:e}");
    // Permuting the examples (here: reversing each set) leaves the pooled output unchanged.
    let mut xp = x.clone();
    for (i, p) in pos.iter().enumerate() {
        for (j, &q) in p.iter().enumerate() {
            let (dst, src) = (i * t + q, i * t + p[p.len() - 1 - j]);
            xp[dst * d..(dst + 1) * d].copy_from_slice(&x[src * d..(src + 1) * d]);
        }
    }
    let pad_t = BoolTensor::from_data(pad.clone(), [s, b, t]);
    let pp = masked_mean_pool(&a.forward_with(&t32(&xp, &[s, b * t, d]), t, Some(&pad_t)), t, Some(&pad_t)).to_vec();
    let e = rel(&pp, &dev[1]);
    assert!(e <= 1e-6, "{TAG}: permuted sets pool {e:e} apart");
    nt::set_default_device(R).unwrap();
    println!("{TAG} set attention: within tolerance of CpuRef, pads zero-gradient, padded/unpadded {worst:.1e}, permuted {e:.1e}, byte-identical run to run");
}

// ------------------------------------------------------------------------------ TargetEma

fn online_map(root: u64) -> VarMap {
    let mut m = VarMap::new();
    m.insert("a.w", t32(&rnd(2 * 6 * 4, root, -1.0, 1.0), &[2, 6, 4])).unwrap();
    m.insert("b.w", t32(&rnd(2 * 9, root + 1, -1.0, 1.0), &[2, 1, 9])).unwrap();
    m
}

fn host(m: &VarMap) -> Vec<f32> {
    m.tensors().iter().flat_map(|t| t.to_vec()).collect()
}

/// Copy initialisation, τ endpoints, 20 scheduled updates, detached reads and save/restore on
/// the device.
fn ema_run(device: Device) -> (Vec<Vec<f32>>, String) {
    nt::set_default_device(device).unwrap();
    let a = vars_on(&online_map(1), device);
    let mut e = TargetEma::new(&a.lifted(), TauSchedule::cosine(16));
    assert!(e.target().tensors().iter().all(|t| t.device() == device && !t.is_tracked()));
    let mut out = vec![host(&e.target())];
    for k in 0..20u64 {
        e.update(&vars_on(&online_map(100 + k), device).lifted());
        out.push(host(&e.target()));
    }
    let json = e.to_json().unwrap();
    nt::set_default_device(R).unwrap();
    (out, json)
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_target_ema_on_the_device() {
    let (cpu, cpu_json) = ema_run(R);
    let (dev, dev_json) = ema_run(M);
    assert_bits(&format!("{TAG} copy initialisation"), &dev[0], &cpu[0]);
    let worst = dev.iter().zip(&cpu).map(|(a, c)| rel(a, c)).fold(0.0, f64::max);
    assert!(worst <= 1e-6, "{TAG} target EMA: {worst:e} from CpuRef");
    let bit_equal = dev.iter().zip(&cpu).all(|(a, c)| a.iter().zip(c).all(|(x, y)| x.to_bits() == y.to_bits()));
    let (again, again_json) = ema_run(M);
    for (i, (a, c)) in dev.iter().zip(&again).enumerate() {
        assert_bits(&format!("{TAG} target EMA run-to-run {i}"), a, c);
    }
    assert_eq!(dev_json, again_json, "{TAG}: saved state differs run to run");
    let _ = cpu_json;
    nt::set_default_device(M).unwrap();
    // τ = 1 and τ = 0, exactly.
    let a = vars_on(&online_map(2), M);
    let b = vars_on(&online_map(3), M);
    let mut e1 = TargetEma::new(&a, TauSchedule::Constant(1.0));
    e1.update(&b);
    assert_bits(&format!("{TAG} τ = 1"), &host(&e1.target()), &host(&a));
    let mut e0 = TargetEma::new(&a, TauSchedule::Constant(0.0));
    e0.update(&b);
    assert_bits(&format!("{TAG} τ = 0"), &host(&e0.target()), &host(&b));
    // No gradient reaches the target.
    let lifted = b.lifted();
    let tgt = e1.target();
    let x = t32(&rnd(2 * 3 * 6, 4, -1.0, 1.0), &[2, 3, 6]);
    let g = mse(x.clone().matmul(lifted.var("a.w").clone()), &x.clone().matmul(tgt.var("a.w").clone())).sum().backward();
    assert!(tgt.tensors().iter().all(|t| t.grad(&g).is_none()) && lifted.var("a.w").grad(&g).is_some());
    // Restore on the device and continue: as the original.
    let mut r = TargetEma::from_json(&dev_json).unwrap();
    assert_eq!(r.step(), 20);
    let mut o = TargetEma::from_json(&dev_json).unwrap();
    for k in 0..3 {
        let on = vars_on(&online_map(200 + k), M);
        r.update(&on);
        o.update(&on);
    }
    assert_eq!(r.to_json().unwrap(), o.to_json().unwrap());
    assert!(r.target().tensors().iter().all(|t| t.device() == M));
    nt::set_default_device(R).unwrap();
    println!("{TAG} target EMA: 20 scheduled updates within {worst:.1e} of CpuRef (bit-equal: {bit_equal}), τ = 1 and τ = 0 exact, no target gradient, restore exact, byte-identical run to run");
}

// --------------------------------------------------------------------------------- VICReg

/// VICReg's four outputs and both branches' gradients on `device`.
fn vicreg_run(a: &[f64], b: &[f64], dims: [usize; 3], device: Device) -> Vec<Vec<f32>> {
    nt::set_default_device(device).unwrap();
    let za = t32(a, &dims).require_grad();
    let zb = t32(b, &dims).require_grad();
    let v = vicreg(za.clone(), zb.clone(), VicregWeights::default());
    assert_eq!(v.loss.device(), device);
    let g = v.loss.clone().sum().backward();
    let out = vec![v.loss.to_vec(), v.invariance.to_vec(), v.variance.to_vec(), v.covariance.to_vec(), za.grad(&g).unwrap().to_vec(), zb.grad(&g).unwrap().to_vec()];
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_vicreg_on_the_device() {
    let dims = [3, 32, 16];
    let n = dims.iter().product();
    let scale: Vec<f64> = (0..16).map(|j| 0.2 + 0.15 * j as f64).collect();
    let a: Vec<f64> = rnd(n, 300, -1.0, 1.0).iter().enumerate().map(|(i, v)| v * scale[i % 16]).collect();
    let b: Vec<f64> = a.iter().zip(rnd(n, 301, -0.2, 0.2)).map(|(x, e)| x + e).collect();
    let cpu = vicreg_run(&a, &b, dims, R);
    let dev = vicreg_run(&a, &b, dims, M);
    let mut worst = [0.0f64; 2];
    for (i, (x, c)) in dev.iter().zip(&cpu).enumerate() {
        let (k, tol) = if i < 4 { (0, 1e-5) } else { (1, 1e-4) };
        let e = rel(x, c);
        assert!(e <= tol, "{TAG} VICReg output {i}: {e:e} from CpuRef");
        worst[k] = worst[k].max(e);
    }
    let again = vicreg_run(&a, &b, dims, M);
    for (i, (x, c)) in dev.iter().zip(&again).enumerate() {
        assert_bits(&format!("{TAG} VICReg run-to-run {i}"), x, c);
    }
    nt::set_default_device(M).unwrap();
    // Above γ: the hinge is 0 with no gradient.
    let big = t32(&rnd(2 * 16 * 5, 302, -4.0, 4.0), &[2, 16, 5]).require_grad();
    let v = vicreg_variance(&big, 1.0, 1e-4);
    assert!(v.to_vec().iter().all(|x| *x == 0.0));
    assert!(big.grad(&v.sum().backward()).unwrap().to_vec().iter().all(|x| *x == 0.0));
    // Whitened (Hadamard columns): covariance exactly 0.
    let h = |i: usize, j: usize| if (i & j).count_ones() % 2 == 0 { 1.5 } else { -1.5 };
    let wz: Vec<f64> = (0..8).flat_map(|i| (1..=4).map(move |j| h(i, j))).collect();
    assert_eq!(vicreg_covariance(&t32(&wz, &[1, 8, 4])).to_vec(), vec![0.0]);
    // Constant: the hinge 0.99.
    let c = vicreg_variance(&t32(&vec![0.75; 2 * 10 * 7], &[2, 10, 7]), 1.0, 1e-4).to_vec();
    assert!(c.iter().all(|x| (*x as f64 - 0.99).abs() <= 1e-6), "{TAG}: constant hinge {c:?}");
    nt::set_default_device(R).unwrap();
    println!("{TAG} VICReg: terms within {:.1e} and gradients within {:.1e} of CpuRef, zero hinge above 1, zero covariance whitened, constant hinge 0.99, byte-identical run to run", worst[0], worst[1]);
}

// ------------------------------------------------------------------ masked group InfoNCE

/// Loss and logit gradient of the masked group InfoNCE on `device`.
fn nce_run(x: &[f64], tg: &[i64], valid: &[bool], keep: Option<&[f64]>, dims: [usize; 3], device: Device) -> (Vec<f32>, Vec<f32>) {
    let [s, g, _] = dims;
    nt::set_default_device(device).unwrap();
    let xt = t32(x, &dims).require_grad();
    let keep = keep.map(|k| t32(k, &[s, g]));
    let l = masked_group_info_nce(xt.clone(), &nt::IntTensor::from_data(tg.to_vec(), [s, g]), &BoolTensor::from_data(valid.to_vec(), dims), keep.as_ref());
    assert_eq!(l.device(), device);
    let gx = xt.grad(&l.clone().sum().backward()).unwrap().to_vec();
    let out = (l.to_vec(), gx);
    nt::set_default_device(R).unwrap();
    out
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_group_info_nce_on_the_device() {
    use rand::Rng;
    let (s, g, k) = (3, 24, 12);
    let dims = [s, g, k];
    let mut r = rng(400);
    let sizes: Vec<usize> = (0..s * g).map(|_| r.gen_range(2..=10)).collect();
    let tg: Vec<i64> = sizes.iter().map(|&n| r.gen_range(0..n) as i64).collect();
    let mut x = rnd(s * g * k, 401, -3.0, 3.0);
    let valid: Vec<bool> = (0..s * g * k).map(|i| i % k < sizes[i / k]).collect();
    let keep: Vec<f64> = (0..s * g).map(|i| if i % 5 == 4 { 0.0 } else { 1.0 }).collect();
    let cpu = nce_run(&x, &tg, &valid, Some(&keep), dims, R);
    let dev = nce_run(&x, &tg, &valid, Some(&keep), dims, M);
    let (e0, e1) = (rel(&dev.0, &cpu.0), rel(&dev.1, &cpu.1));
    assert!(e0 <= 1e-5 && e1 <= 1e-4, "{TAG} group InfoNCE: loss {e0:e}, gradient {e1:e} from CpuRef");
    let again = nce_run(&x, &tg, &valid, Some(&keep), dims, M);
    assert_bits(&format!("{TAG} InfoNCE run-to-run loss"), &dev.0, &again.0);
    assert_bits(&format!("{TAG} InfoNCE run-to-run gradient"), &dev.1, &again.1);
    // Masked entries and dropped groups: exactly zero gradient.
    for (i, gv) in dev.1.iter().enumerate() {
        if !valid[i] || keep[i / k] == 0.0 {
            assert!(gv.to_bits() == 0, "{TAG}: masked entry {i} has a gradient");
        }
    }
    // The pads' values do not matter, to the bit.
    for (i, v) in x.iter_mut().enumerate() {
        if !valid[i] {
            *v = *v * 1e5 - 7.0;
        }
    }
    let other = nce_run(&x, &tg, &valid, Some(&keep), dims, M);
    assert_bits(&format!("{TAG} InfoNCE other pad values"), &dev.0, &other.0);
    // All valid, no group mask: cross_entropy on the device, to the bit.
    let full = vec![true; s * g * k];
    let tf: Vec<i64> = (0..s * g).map(|i| (i % k) as i64).collect();
    let a = nce_run(&x, &tf, &full, None, dims, M);
    nt::set_default_device(M).unwrap();
    let xt = t32(&x, &dims).require_grad();
    let ce = cross_entropy(xt.clone().reshape([s, g, 1, k]), &nt::IntTensor::from_data(tf, [s, g, 1]));
    let gce = xt.grad(&ce.clone().sum().backward()).unwrap().to_vec();
    assert_bits(&format!("{TAG} InfoNCE = cross_entropy (loss)"), &a.0, &ce.to_vec());
    assert_bits(&format!("{TAG} InfoNCE = cross_entropy (gradient)"), &a.1, &gce);
    nt::set_default_device(R).unwrap();
    println!("{TAG} group InfoNCE: loss within {e0:.1e} and gradient within {e1:.1e} of CpuRef, masked entries zero-gradient, pads' values irrelevant, equal to cross_entropy on full groups, byte-identical run to run");
}

/// The joint-embedding bodies, in a fresh process.
#[test]
fn joint_embed_ops_on_the_device() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 4);
}
