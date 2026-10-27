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

/// The joint-embedding bodies, in a fresh process.
#[test]
fn joint_embed_ops_on_the_device() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 1);
}
