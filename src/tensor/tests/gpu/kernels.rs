// Device kernels, one set of bodies for every GPU backend: the device sort, the fused Adam
// update and the large-tile and split-k matmul paths against CpuRef.
// - sort (`GpuBackend::sort_desc`, a bitonic kernel): values bit-equal to CpuRef's on every lane;
//   positions equal to CpuRef's wherever a value is unique in its lane (CpuRef gives first-index
//   order for ties only on lanes up to 32; the device sort always does, checked directly); along
//   every axis, of views, with ±0, infinities and NaNs; lanes up to the 4096 limit on the device,
//   longer lanes through the trait's counted host round trip; the positions stay on the device;
//   topk and its gradient bit-equal.
// - matmul with large tiles and split k: bit-exact.
// - fused Adam: the parameters and both moments bit-equal to CpuRef's op sequence after 5 steps,
//   with coupled decay and with each decoupled-decay order; one fused launch per var and step.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::tensor::tests::*;
use crate::nn::{Adam, AdamConfig, DecayOrder, Optimizer, ParamGroups, VarMap};
use crate::tensor::{self as nt, CpuMode, Device, Tensor};

const R: Device = Device::Cpu(CpuMode::Reference);

fn t(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_data(rnd(numel(shape), seed, -2.0, 2.0), shape.to_vec())
}

fn has_ties(lane: &[f32]) -> bool {
    let mut b: Vec<u32> = lane.iter().map(|x| x.to_bits()).collect();
    b.sort();
    b.windows(2).any(|w| w[0] == w[1])
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_device_sort_matches_cpu_ref() {
    // Distinct values: values and positions equal CpuRef's, along each axis.
    for (shape, dim) in [(vec![4usize, 9], 1usize), (vec![3, 7, 5], 1), (vec![6, 5, 4], 0), (vec![2, 3, 1000], 2), (vec![2, 4096], 1), (vec![1, 1], 1)] {
        let x = t(&shape, 11 + dim as u64);
        let (vr, ir) = x.clone().sort_descending_with_indices(dim);
        let xm = x.to(M);
        let (_, d0, r0) = nt::transfer_counts();
        let (vm, im) = xm.sort_descending_with_indices(dim);
        let (_, d1, r1) = nt::transfer_counts();
        assert_eq!(vm.device(), M);
        assert_eq!((r1 - r0, d1 - d0), (0, 0), "{shape:?}: no round trip and no read (the positions stay on the device)");
        let (vv, rv) = (vm.to_vec(), vr.to_vec());
        assert_bits(&format!("sort {shape:?} dim {dim} values"), &vv, &rv);
        // Positions equal CpuRef's for every value that is unique in its lane (random draws can
        // tie; CpuRef orders ties arbitrarily).
        let (outer, n, inner) = (shape[..dim].iter().product::<usize>(), shape[dim], shape[dim + 1..].iter().product::<usize>());
        let (pm, pr) = (im.to_vec(), ir.to_vec());
        let (mut same, mut tied) = (0, 0);
        for o in 0..outer {
            for i in 0..inner {
                let at = |p: usize| (o * n + p) * inner + i;
                for p in 0..n {
                    let unique = (p == 0 || rv[at(p - 1)].to_bits() != rv[at(p)].to_bits()) && (p + 1 == n || rv[at(p + 1)].to_bits() != rv[at(p)].to_bits());
                    if unique {
                        assert_eq!(pm[at(p)], pr[at(p)], "sort {shape:?} dim {dim}: the position of a unique value");
                        same += 1;
                    } else {
                        tied += 1;
                    }
                }
            }
        }
        eprintln!("{TAG} sort {shape:?} along {dim}: values bit-equal to CpuRef's; positions equal on {same} unique values ({tied} tied)");
    }
    // Ties, ±0 and infinities: values bit-equal; positions of tied values ascending (first index).
    let v: Vec<f32> = (0..8 * 33).map(|i| [1.0, -0.0, 0.0, 2.5, f32::INFINITY, -3.0, 1.0, f32::NEG_INFINITY, 2.5][(i * 7 + i / 33) % 9]).collect();
    let x = Tensor::from_data(v, [8, 33]);
    let (vr, _) = x.clone().sort_descending_with_indices(1);
    let (vm, im) = x.clone().to(M).sort_descending_with_indices(1);
    assert_bits("sort with ties: values", &vm.to_vec(), &vr.to_vec());
    let (xs, vs, is) = (x.to_vec(), vm.to_vec(), im.to_vec());
    for lane in 0..8 {
        assert!(has_ties(&xs[lane * 33..(lane + 1) * 33]));
        for p in 0..32 {
            let (a, b) = (lane * 33 + p, lane * 33 + p + 1);
            assert_eq!(xs[lane * 33 + is[a] as usize].to_bits(), vs[a].to_bits(), "a position points at its value");
            if vs[a].to_bits() == vs[b].to_bits() {
                assert!(is[a] < is[b], "ties in first-index order");
            }
        }
    }
    // topk through the device sort.
    let x = t(&[5, 40], 3);
    let (vr, ir) = x.clone().topk_with_indices(4, 1);
    let (vm, im) = x.to(M).topk_with_indices(4, 1);
    assert_bits("topk values", &vm.to_vec(), &vr.to_vec());
    assert_eq!(im.to_vec(), ir.to_vec());
    // Lanes past the limit sort on the host, a counted round trip.
    let x = t(&[2, 5000], 4);
    let (vr, _) = x.clone().sort_descending_with_indices(1);
    let r0 = nt::transfer_counts().2;
    let (vm, _) = x.to(M).sort_descending_with_indices(1);
    assert_eq!(nt::transfer_counts().2, r0 + 1, "a 5000-long lane goes through the host");
    assert_bits("long lanes", &vm.to_vec(), &vr.to_vec());
    eprintln!("{TAG} sort: ties in first-index order with values equal CpuRef's; topk equal; a 5000-long lane via the host");
}

/// Indices of a descending sort along `dim`: the device's are the stable order (ties by first
/// index; checked directly); they equal CpuRef's wherever CpuRef is stable too (lanes up to 32,
/// the threshold of std's small sort), and elsewhere at every position whose value has no exact
/// tie in its lane (CpuRef's order among ties is then std's unstable choice).
fn check_indices(what: &str, x: &Tensor, dim: usize, vals: &[f32], gpu: &[i64], cpu: &[i64]) {
    let sh = x.shape().to_vec();
    let n = sh[dim];
    let inner: usize = sh[dim + 1..].iter().product();
    let outer: usize = sh[..dim].iter().product();
    for o in 0..outer {
        for i in 0..inner {
            let at = |p: usize| (o * n + p) * inner + i;
            for p in 0..n {
                if p + 1 < n && vals[at(p)].to_bits() == vals[at(p + 1)].to_bits() {
                    assert!(gpu[at(p)] < gpu[at(p + 1)], "{what}: ties not in first-index order");
                }
                let tied = (0..n).any(|q| q != p && vals[at(q)].to_bits() == vals[at(p)].to_bits());
                if n <= 32 || !tied {
                    assert_eq!(gpu[at(p)], cpu[at(p)], "{what}: position {p} of lane ({o}, {i})");
                }
            }
        }
    }
}

/// The device sort against CpuRef: values bit-equal always; indices equal wherever CpuRef puts
/// ties in first-index order (lanes up to 32, or lanes without exact ties), along every axis,
/// for views, with ±0, infinities and NaNs; topk and its gradient; no host round trip.
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_device_sort_ties_views_and_topk_gradient() {
    let (m, r) = (M, R);
    let before = nt::transfer_counts().2;
    let cases: Vec<(Vec<usize>, usize, bool)> = vec![
        (vec![4, 9], 1, true),
        (vec![3, 32, 5], 1, true),
        (vec![2, 7, 6], 0, true),
        (vec![6, 4, 20], 2, true),
        (vec![2, 1000], 1, false),
        (vec![3, 2048], 1, false),
        (vec![5, 1], 1, true),
    ];
    for (shape, dim, ties) in cases {
        let n: usize = shape.iter().product();
        let mut v = rnd(n, 17 + n as u64, -3.0, 3.0);
        if ties {
            for x in v.iter_mut() {
                *x = x.round();
            }
            if n > 6 {
                v[0] = 0.0;
                v[1] = -0.0;
                v[2] = f32::INFINITY;
                v[3] = f32::NEG_INFINITY;
                v[4] = f32::NAN;
                v[5] = -f32::NAN;
            }
        }
        let x = Tensor::from_data(v, shape.clone());
        let (vc, ic) = x.clone().sort_descending_with_indices(dim);
        let (vg, ig) = x.clone().to(m).sort_descending_with_indices(dim);
        assert_eq!(vg.device(), m);
        assert_bits(&format!("sort values {shape:?} dim {dim}"), &vg.to_vec(), &vc.to_vec());
        check_indices(&format!("sort indices {shape:?} dim {dim}"), &x, dim, &vg.to_vec(), &ig.to_vec(), &ic.to_vec());
        // A view (swapped axes) sorts the same.
        let (vcv, icv) = x.clone().swap_dims(0, 1).sort_descending_with_indices(0);
        let (vgv, igv) = x.clone().to(m).swap_dims(0, 1).sort_descending_with_indices(0);
        assert_bits("sort of a view", &vgv.to_vec(), &vcv.to_vec());
        check_indices("sort of a view", &x.clone().swap_dims(0, 1), 0, &vgv.to_vec(), &igv.to_vec(), &icv.to_vec());
        let _ = r;
    }
    // topk and its gradient (scatter-add by the device indices).
    let x = Tensor::from_data(rnd(6 * 30, 5, -2.0, 2.0).into_iter().map(|a| (a * 2.0).round()).collect(), [6, 30]);
    let grad = |dev: Device| {
        let l = x.clone().to(dev).require_grad();
        let (v, i) = l.clone().topk_with_indices(4, 1);
        let g = v.clone().mul(v.clone()).sum().backward();
        (v.to_vec(), i.to_vec(), l.grad(&g).unwrap().to_vec())
    };
    let (a, b) = (grad(r), grad(m));
    assert_bits("topk values", &b.0, &a.0);
    assert_eq!(b.1, a.1, "topk indices");
    assert_bits("topk gradient", &b.2, &a.2);
    assert_eq!(nt::transfer_counts().2, before, "no host round trip");
    eprintln!("{TAG} device sort: values bit-equal, indices equal (ties first-index), topk gradient bit-equal, no round trip");
}

/// Five Adam steps of `cfg` on `device` from the same vars and gradients; the vars and moments,
/// and on the GPU the kernels launched by the steps.
fn adam_run(cfg: AdamConfig, device: Device) -> (Vec<Vec<f32>>, u64) {
    let mut vars = VarMap::new();
    vars.insert("a", t(&[3, 17, 9], 1).to(device)).unwrap();
    vars.insert("b", t(&[3, 1, 40], 2).to(device)).unwrap();
    let mut opt = Adam::new(cfg, ParamGroups::new(&vars), &vars);
    let grads: Vec<Vec<Option<Tensor>>> = (0..5u64).map(|step| vec![Some(t(&[3, 17, 9], 100 + step).to(device)), Some(t(&[3, 1, 40], 200 + step).to(device))]).collect();
    let count = || if device == M { launches() } else { 0 };
    let k0 = count();
    for g in grads {
        opt.step(&mut vars, g, 0.9);
    }
    let k = count() - k0;
    let (m, v, _) = opt.state();
    (vars.tensors().iter().chain(m).chain(v).map(|x| x.to_vec()).collect(), k)
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_fused_adam_matches_cpu_ref_bit_for_bit() {
    let base = AdamConfig { lr: 3e-3, ..AdamConfig::default() };
    let cases = [
        ("plain", base.clone()),
        ("coupled decay", AdamConfig { weight_decay: 0.01, ..base.clone() }),
        ("decoupled decay after", AdamConfig { decoupled_decay: 0.05, ..base.clone() }),
        ("decoupled decay torch order", AdamConfig { decoupled_decay: 0.05, decay_order: DecayOrder::Torch, ..base.clone() }),
    ];
    for (what, cfg) in cases {
        let (cpu, _) = adam_run(cfg.clone(), R);
        let (gpu, k) = adam_run(cfg, M);
        for (i, (a, b)) in gpu.iter().zip(&cpu).enumerate() {
            assert_bits(&format!("{what}: tensor {i}"), a, b);
        }
        eprintln!("{TAG} fused adam {what}: vars and moments bit-equal to CpuRef after 5 steps ({k} launches for 2 vars × 5 steps)");
        assert!(k <= 10, "{what}: one fused launch per var and step ({k})");
    }
}

/// The 128×128-tile matmul (large outputs) is bit-exact against CpuRef as the 64×64 one is,
/// including views read through strides and broadcast batches.
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_matmul_large_tiles_and_split_k_bit_exact() {
    type F = fn(&Tensor, &Tensor) -> Tensor;
    let cases: Vec<(&str, Vec<usize>, Vec<usize>, F)> = vec![
        ("[2,130,300]x[2,300,257]", vec![2, 130, 300], vec![2, 300, 257], |a, b| a.clone().matmul(b.clone())),
        ("[1,128,64]x[1,64,128]", vec![1, 128, 64], vec![1, 64, 128], |a, b| a.clone().matmul(b.clone())),
        ("[3,200,513]x[1,513,140] broadcast", vec![3, 200, 513], vec![1, 513, 140], |a, b| a.clone().matmul(b.clone())),
        ("transposed views", vec![2, 300, 150], vec![2, 260, 300], |a, b| a.clone().swap_dims(1, 2).matmul(b.clone().swap_dims(1, 2))),
        ("[1,1024,1024]x[1,1024,1024]", vec![1, 1024, 1024], vec![1, 1024, 1024], |a, b| a.clone().matmul(b.clone())),
        ("split-k [1,256,2048]x[1,2048,256]", vec![1, 256, 2048], vec![1, 2048, 256], |a, b| a.clone().matmul(b.clone())),
        ("split-k transposed rhs, k 1000", vec![2, 130, 1000], vec![2, 140, 1000], |a, b| a.clone().matmul(b.clone().swap_dims(1, 2))),
    ];
    for (i, (name, sa, sb, f)) in cases.into_iter().enumerate() {
        let (a, b) = (t(&sa, 60 + i as u64), t(&sb, 70 + i as u64));
        let cpu = f(&a, &b).to_vec();
        let gpu = f(&a.to(M), &b.to(M)).to_vec();
        assert_bits(name, &gpu, &cpu);
        eprintln!("{TAG} matmul {name}: {} outputs bit-equal to CpuRef", cpu.len());
    }
}

/// The kernel bodies, in a fresh process.
#[test]
fn sort_adam_and_matmul_against_cpu_ref() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 4);
}

/// Matmul throughput through the tensor API on resident data (ignored; median of 21 after a
/// warm-up, synchronised): row-major, transposed-lhs and transposed-rhs products at the d512
/// step's shapes and 1024³.
#[test]
#[ignore]
fn matmul_throughput() {
    type F = fn(&Tensor, &Tensor) -> Tensor;
    let cases: Vec<(&str, [usize; 3], F)> = vec![
        ("A·B 1024³", [1024, 1024, 1024], |a, b| a.clone().matmul(b.clone())),
        ("A·B 512×512×2048", [512, 512, 2048], |a, b| a.clone().matmul(b.clone())),
        ("A·Bᵀ 512×2048×512", [512, 2048, 512], |a, b| a.clone().matmul(b.clone().swap_dims(1, 2))),
        ("Aᵀ·B 512×512×2048", [512, 512, 2048], |a, b| a.clone().swap_dims(1, 2).matmul(b.clone())),
    ];
    for (name, [m, k, n], f) in cases {
        let transposed_a = name.starts_with("Aᵀ");
        let transposed_b = name.contains("Bᵀ");
        let a = t(&if transposed_a { [1, k, m] } else { [1, m, k] }, 1).to(M);
        let b = t(&if transposed_b { [1, n, k] } else { [1, k, n] }, 2).to(M);
        for _ in 0..3 {
            drop(f(&a, &b));
        }
        nt::synchronize(M).unwrap();
        let mut v: Vec<f64> = (0..21)
            .map(|_| {
                let t0 = std::time::Instant::now();
                let y = f(&a, &b);
                nt::synchronize(M).unwrap();
                drop(y);
                t0.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let ms = v[10];
        eprintln!("{TAG} matmul throughput {name}: {ms:.3} ms, {:.0} GFLOP/s", 2.0 * (m * k * n) as f64 / (ms * 1e-3) / 1e9);
    }
}
