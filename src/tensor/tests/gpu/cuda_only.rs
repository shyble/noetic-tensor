// CUDA-only tests: the raw device API (`CudaDevice`): the pin, device info and
// key, cuBLAS sgemm parity, shape errors, and the raw latency and crossover table. The raw
// vector add and tiled sgemm run in the generic raw bodies (raw.rs). Included by gpu/mod.rs.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::tensor::tests::*;
use crate::tensor::backend::{BackendStorage, CpuFast, CpuRef};
use crate::tensor::cuda::CudaDevice;
use crate::tensor::TensorError;
use std::time::Instant;

fn dev() -> CudaDevice {
    CudaDevice::new(0).unwrap_or_else(|e| panic!("CUDA device 0: {e}"))
}

#[test]
fn pinned_process_never_initialises_cuda() {
    assert!(matches!(CudaDevice::new_with_pin(0, true), Err(TensorError::Device(m)) if m.contains("pinned")));
}

#[test]
fn device_info_and_key() {
    let d = dev();
    let i = d.info();
    assert!(i.compute_capability.0 >= 5 && i.multiprocessors > 0 && i.total_mem > 0, "{i:?}");
    let key = d.gpu_key();
    assert!(key.starts_with("cuda-") && key.contains("pedantic-tf32off"), "{key}");
    eprintln!("{i:?}\n{key}");
}

/// Max of |Δ| / bound over the outputs, with bound = 2k·ε·(|A||B|)ᵢⱼ (ε = f32::EPSILON), and
/// the number of outputs equal bit for bit.
fn against_cpu(want: &[f32], got: &[f32], a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> (f64, usize) {
    let eps = f32::EPSILON as f64;
    let (mut worst, mut equal) = (0f64, 0usize);
    for t in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut abs = 0f64;
                for l in 0..k {
                    abs += (a[t * m * k + i * k + l] as f64).abs() * (b[t * k * n + l * n + j] as f64).abs();
                }
                let o = t * m * n + i * n + j;
                let bound = 2.0 * k as f64 * eps * abs;
                let delta = (want[o] as f64 - got[o] as f64).abs();
                if want[o].to_bits() == got[o].to_bits() {
                    equal += 1;
                } else {
                    worst = worst.max(if bound > 0.0 { delta / bound } else { f64::INFINITY });
                }
            }
        }
    }
    (worst, equal)
}

const SHAPES: [(usize, usize, usize, usize); 10] = [(1, 1, 1, 1), (1, 7, 13, 5), (1, 65, 33, 129), (3, 37, 50, 29), (1, 64, 64, 64), (32, 64, 64, 64), (1, 256, 256, 256), (1, 300, 1000, 17), (1, 512, 512, 512), (1, 1024, 1024, 1024)];

#[test]
fn sgemm_cublas_parity_with_cpu_ref() {
    let d = dev();
    for (s, &(batch, m, k, n)) in SHAPES.iter().enumerate() {
        let a = rnd(batch * m * k, 100 + s as u64, -1.0, 1.0);
        let b = rnd(batch * k * n, 200 + s as u64, -1.0, 1.0);
        let (want, sh) = CpuRef::matmul(&a, &[batch, m, k], &b, &[batch, k, n]);
        assert_eq!(sh, vec![batch, m, n]);
        let (ga, gb, gc) = (d.upload(&a).unwrap(), d.upload(&b).unwrap(), d.alloc(batch * m * n).unwrap());
        for (name, cublas) in [("cublas", true)] {
            let run = || {
                if cublas { d.sgemm_cublas(&ga, &gb, &gc, batch, m, k, n).unwrap() } else { d.sgemm_tiled(&ga, &gb, &gc, batch, m, k, n).unwrap() }
                d.download(&gc).unwrap()
            };
            let got = run();
            let again = run();
            assert!(got.iter().zip(&again).all(|(x, y)| x.to_bits() == y.to_bits()), "{name} [{batch},{m},{k},{n}] is not repeatable");
            let (worst, equal) = against_cpu(&want, &got, &a, &b, batch, m, k, n);
            eprintln!("{name:6} [{batch},{m},{k},{n}]: max |Δ|/bound {worst:.4}, bit-equal {equal}/{}", want.len());
            assert!(worst <= 1.0, "{name} [{batch},{m},{k},{n}] exceeds 2k·ε·|A||B| (ratio {worst})");
        }
    }
}

#[test]
fn shape_errors_are_reported() {
    let d = dev();
    let (a, c) = (d.alloc(6).unwrap(), d.alloc(5).unwrap());
    assert!(matches!(d.sgemm_tiled(&a, &a, &c, 1, 2, 3, 2), Err(TensorError::Shape(_))));
    assert!(matches!(d.add(&a, &a, &c), Err(TensorError::Shape(_))));
    assert!(matches!(d.write(&c, &[1.0; 3]), Err(TensorError::Shape(_))));
}

fn median_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut v: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Latency (median ms) of matmul and vector add on CpuRef (one thread), CpuFast (all threads)
/// and the GPU (kernel only: queue + synchronise; end to end: upload A and B, compute, download).
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn raw_latency_and_crossover() {
    let d = dev();
    eprintln!("{}\n{}\nCPU threads (Fast): {}", d.gpu_key(), crate::tensor::platform_key(), crate::tensor::backend_threads());
    eprintln!("\nmatmul (median ms; GFLOP/s of the kernel-only GPU time in brackets)");
    eprintln!("{:>16} {:>10} {:>10} {:>18} {:>18} {:>10} {:>10}", "shape", "cpu_ref", "cpu_fast", "tiled_kernel", "cublas_kernel", "tiled_e2e", "cublas_e2e");
    for (batch, dd) in [(1usize, 64usize), (32, 64), (1, 256), (1, 512), (1, 1024)] {
        let (m, k, n) = (dd, dd, dd);
        let a = rnd(batch * m * k, 1, -1.0, 1.0);
        let b = rnd(batch * k * n, 2, -1.0, 1.0);
        let runs = if dd >= 1024 { 7 } else if dd >= 512 { 15 } else { 51 };
        let (ash, bsh) = ([batch, m, k], [batch, k, n]);
        let cpu_ref = median_ms(runs, || {
            std::hint::black_box(CpuRef::matmul(&a, &ash, &b, &bsh));
        });
        let cpu_fast = median_ms(runs, || {
            std::hint::black_box(CpuFast::matmul(&a, &ash, &b, &bsh));
        });
        let (ga, gb, gc) = (d.upload(&a).unwrap(), d.upload(&b).unwrap(), d.alloc(batch * m * n).unwrap());
        let gpu_runs = 101;
        let tiled = median_ms(gpu_runs, || {
            d.sgemm_tiled(&ga, &gb, &gc, batch, m, k, n).unwrap();
            d.synchronize().unwrap();
        });
        let cublas = median_ms(gpu_runs, || {
            d.sgemm_cublas(&ga, &gb, &gc, batch, m, k, n).unwrap();
            d.synchronize().unwrap();
        });
        let e2e = |cub: bool| {
            median_ms(gpu_runs.min(31), || {
                d.write(&ga, &a).unwrap();
                d.write(&gb, &b).unwrap();
                if cub { d.sgemm_cublas(&ga, &gb, &gc, batch, m, k, n).unwrap() } else { d.sgemm_tiled(&ga, &gb, &gc, batch, m, k, n).unwrap() }
                std::hint::black_box(d.download(&gc).unwrap());
            })
        };
        let (te, ce) = (e2e(false), e2e(true));
        let gf = |ms: f64| 2.0 * (batch * m * k * n) as f64 / (ms * 1e-3) / 1e9;
        let shape = if batch == 1 { format!("{dd}³") } else { format!("{batch}×{dd}³") };
        eprintln!("{:>16} {:>10.4} {:>10.4} {:>9.4} [{:>6.0}] {:>9.4} [{:>6.0}] {:>10.4} {:>10.4}", shape, cpu_ref, cpu_fast, tiled, gf(tiled), cublas, gf(cublas), te, ce);
    }
    eprintln!("\nvector add of d² elements (median ms)");
    eprintln!("{:>10} {:>10} {:>10} {:>12} {:>10}", "n", "cpu_ref", "cpu_fast", "gpu_kernel", "gpu_e2e");
    for dd in [64usize, 256, 512, 1024, 4096] {
        let n = dd * dd;
        let (a, b) = (rnd(n, 3, -1.0, 1.0), rnd(n, 4, -1.0, 1.0));
        let cpu_ref = median_ms(51, || {
            std::hint::black_box(CpuRef::zip(&a, &[n], &b, &[n], |x: f32, y: f32| x + y));
        });
        let cpu_fast = median_ms(51, || {
            std::hint::black_box(CpuFast::zip(&a, &[n], &b, &[n], |x: f32, y: f32| x + y));
        });
        let (ga, gb, gc) = (d.upload(&a).unwrap(), d.upload(&b).unwrap(), d.alloc(n).unwrap());
        let k = median_ms(101, || {
            d.add(&ga, &gb, &gc).unwrap();
            d.synchronize().unwrap();
        });
        let e = median_ms(31, || {
            d.write(&ga, &a).unwrap();
            d.write(&gb, &b).unwrap();
            d.add(&ga, &gb, &gc).unwrap();
            std::hint::black_box(d.download(&gc).unwrap());
        });
        eprintln!("{:>10} {:>10.4} {:>10.4} {:>12.4} {:>10.4}", format!("{dd}²"), cpu_ref, cpu_fast, k, e);
    }
    // Launch overhead: an empty-ish add of one element, queue + synchronise.
    let (one, out) = (d.upload(&[1.0]).unwrap(), d.alloc(1).unwrap());
    let lat = median_ms(1001, || {
        d.add(&one, &one, &out).unwrap();
        d.synchronize().unwrap();
    });
    eprintln!("\nlaunch + synchronise (1 element): {lat:.4} ms");
}
