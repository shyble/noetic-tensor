//! Devices and the fast CPU backend. Reference stays byte-identical (every other test); Fast is
//! checked against Reference within a tolerance per op, and its speed is reported.

use super::*;
use crate::tensor::device::check_with;
use crate::tensor::{self as nt, CpuMode, Device, TensorError};
use std::time::Instant;

const FAST: Device = Device::Cpu(CpuMode::Fast);

fn big(shape: &[usize], seed: u64) -> Tensor {
    Tensor::from_data(rnd(numel(shape), seed, -2.0, 2.0), shape.to_vec())
}

/// Largest |fast − reference| / (1 + |reference|).
fn rel(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| ((*x as f64 - *y as f64).abs()) / (1.0 + (*y as f64).abs())).fold(0.0, f64::max)
}

/// Run one ignored test body alone in a fresh process: the reference pin is process-wide, and
/// any library test that pins the reference backend pins the shared test process.
fn isolated(name: &str) {
    let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args([name, "--exact", "--ignored", "--test-threads=1", "--nocapture"])
        .output()
        .expect("run the test binary");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "{name} in a fresh process:\n{text}");
    for l in text.lines().filter(|l| l.starts_with("fast vs")) {
        eprintln!("{l}");
    }
}

#[test]
fn devices_placeholders_and_the_reference_pin() {
    isolated("tensor::tests::fast::devices_body");
}

#[test]
fn fast_matches_reference_within_tolerance() {
    isolated("tensor::tests::fast::tolerance_body");
}

#[test]
fn pin_reference_pins_the_process() {
    isolated("tensor::tests::fast::pin_body");
}

/// `pin_reference` pins the process: Fast is then refused by `to` and by the default device,
/// and a Fast default falls back to Reference.
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn pin_body() {
    assert!(!crate::tensor::device::is_pinned());
    assert!(big(&[2], 1).try_to(FAST).is_ok(), "Fast is selectable before the pin");
    crate::tensor::set_default_device(FAST).unwrap();
    crate::tensor::pin_reference();
    assert!(crate::tensor::device::is_pinned());
    assert_eq!(crate::tensor::default_device(), Device::Cpu(CpuMode::Reference), "a Fast default falls back to Reference");
    assert!(matches!(big(&[2], 1).try_to(FAST), Err(TensorError::Unsupported(m)) if m.contains("pinned")));
    assert!(crate::tensor::set_default_device(FAST).is_err());
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn devices_body() {
    let x = big(&[2, 3], 1);
    assert_eq!(x.device(), Device::Cpu(CpuMode::Reference), "Reference is the default");
    if !cfg!(all(feature = "metal", target_os = "macos")) {
        assert!(matches!(x.clone().try_to(Device::Metal(0)), Err(TensorError::Unsupported(_))), "no Metal backend in this build");
    }
    assert!(matches!(x.clone().try_to(Device::Cuda(1)), Err(TensorError::Unsupported(_))));
    assert!(check_with(FAST, false).is_ok());
    assert!(matches!(check_with(FAST, true), Err(TensorError::Unsupported(m)) if m.contains("pinned")), "a pinned process refuses Fast");
    assert!(check_with(Device::Cpu(CpuMode::Reference), true).is_ok());
    for g in [Device::Metal(0), Device::Cuda(0)] {
        assert!(matches!(check_with(g, true), Err(TensorError::Unsupported(m)) if m.contains("pinned")), "a pinned process refuses {g:?}");
    }
    let f = x.clone().to(FAST);
    assert_eq!(f.device(), FAST);
    assert!(matches!(f.clone().try_add(x.clone()), Err(TensorError::Unsupported(_))), "mixed devices are an error");
    assert_eq!((f.clone() + f).device(), FAST, "results stay on their device");
    let key = crate::tensor::platform_key();
    assert!(key.contains(std::env::consts::OS) && key.contains(std::env::consts::ARCH) && key.contains("rustc") && key.contains("matrixmultiply-0.3.11"), "{key}");
}

/// Fast equals Reference exactly where it keeps Reference's order (maps, zips, matmul row
/// blocks, reductions across lanes) and within 1e-5 (|Δ|/(1+|ref|)) where it reduces a lane with SIMD
/// partial sums; the loss and gradients of a whole decoder stay within 1e-5 and 1e-4.
#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn tolerance_body() {
    // Above the fast kernels' size thresholds (2^22 elements).
    let (a, b) = (big(&[64, 256, 257], 2), big(&[64, 256, 257], 3));
    let run = |dev: Device| {
        let (a, b) = (a.clone().to(dev), b.clone().to(dev));
        vec![
            ("add", (a.clone() + b.clone()).to_vec()),
            ("exp", a.clone().exp().to_vec()),
            ("sum_dim last", a.clone().sum_dim(2).to_vec()),
            ("sum_dim middle", a.clone().sum_dim(1).to_vec()),
            ("sum", a.clone().sum().to_vec()),
            ("matmul", a.clone().matmul(b.clone().swap_dims(1, 2)).to_vec()),
            ("softmax", nt::softmax(a.clone(), 2).to_vec()),
        ]
    };
    let (r, f) = (run(Device::Cpu(CpuMode::Reference)), run(FAST));
    for ((name, x), (_, y)) in r.iter().zip(&f) {
        let d = rel(y, x);
        let exact = matches!(*name, "add" | "exp" | "sum_dim middle" | "matmul");
        eprintln!("fast vs reference {name:<16} max rel diff {d:.2e}{}", if exact && d == 0.0 { " (identical)" } else { "" });
        if exact {
            assert_bits(name, y, x);
        } else {
            assert!(d < 1e-5, "{name}: {d}");
        }
    }
    // A whole decoder: loss and gradients.
    let cfg = crate::nn::DecoderConfig::new(32, 32, 4, 16, 2, 96);
    let (_, vars) = crate::nn::Decoder::init(cfg.clone(), 2, 4).unwrap();
    let (s, b, t) = (2, 8, cfg.context);
    let tk = our_ints(&rnd(s * b * t, 1, 0.0, 32.0).iter().map(|x| *x as i64).collect::<Vec<_>>(), [s, b, t]);
    let tg = our_ints(&rnd(s * b * t, 2, 0.0, 32.0).iter().map(|x| *x as i64).collect::<Vec<_>>(), [s, b, t]);
    let lg = |dev: Device| {
        crate::tensor::set_default_device(dev).unwrap();
        let mut on = crate::nn::VarMap::new();
        for (name, x) in vars.iter() {
            on.insert(name, Tensor::from_data(x.to_vec(), x.shape()).to(dev)).unwrap();
        }
        let lifted = on.lifted();
        let logits = crate::nn::Decoder::load(cfg.clone(), &lifted).unwrap().forward(&tk);
        let loss = crate::nn::cross_entropy(logits, &tg);
        let g = loss.clone().sum().backward();
        crate::tensor::set_default_device(Device::Cpu(CpuMode::Reference)).unwrap();
        (loss.to_vec(), lifted.grads(&g).into_iter().map(|x| x.map(|x| x.to_vec()).unwrap_or_default()).collect::<Vec<_>>())
    };
    let ((lr, gr), (lf, gf)) = (lg(Device::Cpu(CpuMode::Reference)), lg(FAST));
    let dl = rel(&lf, &lr);
    let dg = gr.iter().zip(&gf).map(|(x, y)| rel(y, x)).fold(0.0, f64::max);
    eprintln!("fast vs reference decoder: loss {dl:.2e}, gradients {dg:.2e}");
    assert!(dl < 1e-5 && dg < 1e-4, "decoder: loss {dl}, gradients {dg}");
}

/// Speed of Fast against Reference (ignored: `cargo test --release --lib tensor::tests::fast -- --ignored --nocapture`).
#[test]
#[ignore]
fn fast_speedups() {
    let time = |f: &dyn Fn()| {
        f();
        let mut v: Vec<f64> = (0..7).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() * 1e3 }).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[3]
    };
    let (a, b) = (big(&[16, 256, 256], 4), big(&[16, 256, 256], 5));
    eprintln!("threads: {}", crate::tensor::backend_threads());
    for (name, op) in [
        ("add 1M", &(|x: &Tensor, y: &Tensor| { let _ = (x.clone() + y.clone()).to_vec(); }) as &dyn Fn(&Tensor, &Tensor)),
        ("exp 1M", &|x: &Tensor, _: &Tensor| { let _ = x.clone().exp().to_vec(); }),
        ("sum_dim last 1M", &|x: &Tensor, _: &Tensor| { let _ = x.clone().sum_dim(2).to_vec(); }),
        ("softmax 1M", &|x: &Tensor, _: &Tensor| { let _ = nt::softmax(x.clone(), 2).to_vec(); }),
        ("matmul 16x256^3", &|x: &Tensor, y: &Tensor| { let _ = x.clone().matmul(y.clone()).to_vec(); }),
    ] {
        let r = time(&|| op(&a, &b));
        let (fa, fb) = (a.clone().to(FAST), b.clone().to(FAST));
        let f = time(&|| op(&fa, &fb));
        eprintln!("{name:<18} reference {r:8.2} ms  fast {f:8.2} ms  ({:.2}x)", r / f);
    }
}

/// Crossover sizes for the fast backend's thresholds (ignored; prints a table).
#[test]
#[ignore]
fn fast_crossover() {
    let time = |f: &dyn Fn()| {
        f();
        let mut v: Vec<f64> = (0..9).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() * 1e3 }).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[4]
    };
    for n in [1usize << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22] {
        let (a, b) = (big(&[n / 256, 256], 6), big(&[n / 256, 256], 7));
        let (fa, fb) = (a.clone().to(FAST), b.clone().to(FAST));
        let row = |name: &str, r: f64, f: f64| eprintln!("n {n:>8} {name:<8} ref {r:8.3} ms  fast {f:8.3} ms  {:.2}x", r / f);
        row("add", time(&|| { let _ = (a.clone() + b.clone()).to_vec(); }), time(&|| { let _ = (fa.clone() + fb.clone()).to_vec(); }));
        row("exp", time(&|| { let _ = a.clone().exp().to_vec(); }), time(&|| { let _ = fa.clone().exp().to_vec(); }));
        row("sum_dim", time(&|| { let _ = a.clone().sum_dim(1).to_vec(); }), time(&|| { let _ = fa.clone().sum_dim(1).to_vec(); }));
        let m = ((n as f64).sqrt() as usize).max(16);
        let (x, y) = (big(&[1, m, 64], 8), big(&[1, 64, m], 9));
        let (fx, fy) = (x.clone().to(FAST), y.clone().to(FAST));
        row("matmul", time(&|| { let _ = x.clone().matmul(y.clone()).to_vec(); }), time(&|| { let _ = fx.clone().matmul(fy.clone()).to_vec(); }));
    }
}
