// Metal-only tests: the raw device's latency table, the tuned matmul variants
// (every variant bit-equal to CpuRef on every layout) and their benchmark, and a CPU-side
// fact the sort tests rely on (CpuRef's tie order by lane length). Included by gpu/mod.rs.

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use crate::tensor::tests::*;
use crate::tensor::kernels as k;
use crate::tensor::metal::{self, Arg, MatmulParams, MATMUL_VARIANTS};
use crate::tensor::Tensor;
use std::time::Instant;

/// Latency and crossover (`cargo test --release --features metal --lib raw_latency -- --ignored --nocapture`).
#[test]
#[ignore]
fn raw_latency() {
    use crate::tensor::{CpuMode, Device};
    let time = |f: &mut dyn FnMut()| {
        f();
        let mut v: Vec<f64> = (0..9)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[4]
    };
    eprintln!("threads {}", crate::tensor::backend_threads());
    eprintln!("{:>5} {:>7} | {:>10} {:>10} {:>10} {:>10} | {:>6} {:>6}", "d", "op", "cpu_ref", "cpu_fast", "metal_e2e", "metal_krn", "ref/e2e", "ref/krn");
    for d in [64usize, 256, 512, 1024] {
        let (a, b) = (rnd(d * d, 5, -1.0, 1.0), rnd(d * d, 6, -1.0, 1.0));
        let (ta, tb) = (Tensor::from_data(a.clone(), [d, d]), Tensor::from_data(b.clone(), [d, d]));
        let (fa, fb) = (ta.clone().to(Device::Cpu(CpuMode::Fast)), tb.clone().to(Device::Cpu(CpuMode::Fast)));
        // Resident buffers: kernel plus command-buffer round trip only.
        let ctx = metal::context().expect("Metal");
        let (ba, bb) = (ctx.upload(metal::f32_bytes(&a)).unwrap(), ctx.upload(metal::f32_bytes(&b)).unwrap());
        let c = ctx.alloc(d * d * 4).unwrap();
        let offs = ctx.upload(&[0u8; 8]).unwrap();
        drop(ctx);
        let p = metal::MatmulParams { m: d as u32, n: d as u32, k: d as u32, a_rs: d as u32, a_cs: 1, b_rs: d as u32, b_cs: 1 };
        let r_ref = time(&mut || drop(k::matmul(&a, &[d, d], &b, &[d, d])));
        let r_fast = time(&mut || drop(fa.clone().matmul(fb.clone()).to_vec()));
        let r_e2e = time(&mut || drop(metal::raw_matmul(&a, &b, d, d, d).unwrap()));
        let r_krn = time(&mut || {
            let mut ctx = metal::context().unwrap();
            metal::encode_matmul(&mut ctx, &ba, &bb, &c, &offs, 1, p).unwrap();
            ctx.sync().unwrap();
        });
        let gflops = |ms: f64| 2.0 * (d * d * d) as f64 / (ms * 1e6);
        eprintln!("{d:>5} {:>7} | {r_ref:>10.3} {r_fast:>10.3} {r_e2e:>10.3} {r_krn:>10.3} | {:>6.2} {:>6.2}   (GFLOP/s ref {:.1} fast {:.1} metal kernel {:.1})", "matmul", r_ref / r_e2e, r_ref / r_krn, gflops(r_ref), gflops(r_fast), gflops(r_krn));
        let a_ref = time(&mut || drop(ta.clone().add(tb.clone()).to_vec()));
        let a_fast = time(&mut || drop(fa.clone().add(fb.clone()).to_vec()));
        let a_e2e = time(&mut || drop(metal::raw_add(&a, &b).unwrap()));
        let a_krn = time(&mut || {
            let mut ctx = metal::context().unwrap();
            let n = (d * d) as u32;
            let (g, t) = metal::groups_1d(d * d, 256);
            ctx.dispatch("add_f32", &[metal::Arg::Buf(&ba), metal::Arg::Buf(&bb), metal::Arg::Buf(&c), metal::Arg::Bytes(metal::bytes_of(&n))], g, t).unwrap();
            ctx.sync().unwrap();
        });
        eprintln!("{d:>5} {:>7} | {a_ref:>10.3} {a_fast:>10.3} {a_e2e:>10.3} {a_krn:>10.3} | {:>6.2} {:>6.2}", "add d^2", a_ref / a_e2e, a_ref / a_krn);
    }
    eprintln!("GPU: {}", metal::context().unwrap().name());
}

/// Row-major m×k values `v` stored as given (`col` false) or transposed in memory (`col`
/// true), with the element strides of the logical m×k matrix.
fn store(v: &[f32], m: usize, kk: usize, col: bool) -> (Vec<f32>, usize, usize) {
    if !col {
        return (v.to_vec(), kk, 1);
    }
    let mut t = vec![0f32; v.len()];
    for i in 0..m {
        for j in 0..kk {
            t[j * m + i] = v[i * kk + j];
        }
    }
    (t, 1, m)
}

fn run_variant(v: (&'static str, usize, usize, usize, usize), a: &[f32], b: &[f32], m: usize, kk: usize, n: usize, ta: bool, tb: bool) -> Vec<f32> {
    let (sa, a_rs, a_cs) = store(a, m, kk, ta);
    let (sb, b_rs, b_cs) = store(b, kk, n, tb);
    let mut ctx = metal::context().unwrap();
    let (ba, bb) = (ctx.upload(metal::f32_bytes(&sa)).unwrap(), ctx.upload(metal::f32_bytes(&sb)).unwrap());
    let c = ctx.alloc(m * n * 4).unwrap();
    let p = MatmulParams { m: m as u32, n: n as u32, k: kk as u32, a_rs: a_rs as u32, a_cs: a_cs as u32, b_rs: b_rs as u32, b_cs: b_cs as u32 };
    metal::encode_matmul_variant(&mut ctx, v, &ba, &bb, &c, Arg::Bytes(&[0u8; 8]), 1, p).unwrap();
    ctx.sync().unwrap();
    metal::read_f32(&c, m * n)
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_tuned_matmul_bit_exact() {
    for &(m, kk, n) in &[(1, 1, 1), (3, 5, 7), (17, 33, 9), (65, 129, 63), (100, 300, 70), (130, 257, 131), (256, 512, 64), (64, 1000, 200)] {
        let (a, b) = (rnd(m * kk, 3, -1.0, 1.0), rnd(kk * n, 4, -1.0, 1.0));
        let (want, _) = k::matmul(&a, &[m, kk], &b, &[kk, n]);
        for v in MATMUL_VARIANTS {
            for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                assert_bits(&format!("{} {m}x{kk}x{n} ta {ta} tb {tb}", v.0), &run_variant(v, &a, &b, m, kk, n, ta, tb), &want);
            }
        }
    }
    eprintln!("metal tuned matmul: {} variants × 4 layouts × 8 shapes bit-equal to CpuRef", MATMUL_VARIANTS.len());
}

/// The Metal-only bodies, in a fresh process.
#[test]
fn metal_matmul_variants_against_cpu_ref() {
    crate::tensor::tests::isolated_bodies(&here!("body_"), 1);
}

/// Matmul throughput per variant on the d512 step's shapes (ignored).
#[test]
#[ignore]
fn metal_perf_matmul_bench() {
    let shapes = [
        ("x·W  512x512x512", 512, 512, 512, false, false),
        ("x·W1 512x512x2048", 512, 512, 2048, false, false),
        ("h·W2 512x2048x512", 512, 2048, 512, false, false),
        ("g·W1ᵀ 512x2048x512", 512, 2048, 512, false, true),
        ("xᵀ·g 512x512x2048", 512, 512, 2048, true, false),
        ("hᵀ·g 2048x512x512", 2048, 512, 512, true, false),
        ("head 512x512x256", 512, 512, 256, false, false),
        ("reg x·W 1024x32x96", 1024, 32, 96, false, false),
        ("reg xᵀ·g 32x1024x96", 32, 1024, 96, true, false),
        ("reg Wᵀ 32x256x32 (S 4)", 32, 256, 32, true, false),
        ("reg 32x256x415", 32, 256, 415, true, false),
        ("reg 415x256x32", 415, 256, 32, true, false),
        ("reg 256x32x415", 256, 32, 415, false, false),
    ];
    let mut all = vec![("matmul_f32", 64usize, 64usize, 4usize, 4usize)];
    all.extend(MATMUL_VARIANTS);
    for (what, m, kk, n, ta, tb) in shapes {
        let (a, b) = (rnd(m * kk, 5, -1.0, 1.0), rnd(kk * n, 6, -1.0, 1.0));
        let (sa, a_rs, a_cs) = store(&a, m, kk, ta);
        let (sb, b_rs, b_cs) = store(&b, kk, n, tb);
        let ctx = metal::context().unwrap();
        let (ba, bb) = (ctx.upload(metal::f32_bytes(&sa)).unwrap(), ctx.upload(metal::f32_bytes(&sb)).unwrap());
        let c = ctx.alloc(m * n * 4).unwrap();
        let offs = ctx.upload(&[0u8; 8]).unwrap();
        drop(ctx);
        let p = MatmulParams { m: m as u32, n: n as u32, k: kk as u32, a_rs: a_rs as u32, a_cs: a_cs as u32, b_rs: b_rs as u32, b_cs: b_cs as u32 };
        let mut row = Vec::new();
        for (i, v) in all.iter().enumerate() {
            // 20 back-to-back launches in one command buffer, GPU time per launch.
            let mut best = f64::MAX;
            for _ in 0..3 {
                let mut ctx = metal::context().unwrap();
                ctx.sync().unwrap();
                let _ = ctx.prof.take();
                for _ in 0..20 {
                    if i == 0 {
                        metal::encode_matmul(&mut ctx, &ba, &bb, &c, &offs, 1, p).unwrap();
                    } else {
                        metal::encode_matmul_variant(&mut ctx, *v, &ba, &bb, &c, Arg::Bytes(&[0u8; 8]), 1, p).unwrap();
                    }
                }
                ctx.sync().unwrap();
                best = best.min(ctx.prof.take().gpu_ns as f64 / 20.0 / 1e6);
            }
            row.push(format!("{} {:.3} ms {:.0} GF/s", v.0, best, 2.0 * (m * kk * n) as f64 / best / 1e6));
        }
        eprintln!("metal matmul {what:<22}: {}", row.join(" | "));
    }
}

/// Whether CpuRef's descending sort puts ties in first-index order, by lane length, printed for
/// the report (under burn's `sort_unstable_by` only short lanes did; the sort is now
/// stable, so every lane does).
#[test]
#[ignore]
fn metal_perf_cpu_ref_tie_order() {
    for n in [2usize, 4, 8, 16, 20, 21, 32, 64, 256, 1024] {
        let mut stable_lanes = 0;
        for seed in 0..50u64 {
            // Few distinct values: many ties.
            let v: Vec<f32> = rnd(n, 1000 + seed, 0.0, 4.0).into_iter().map(|x| x.floor()).collect();
            let (_, idx) = k::sort_desc_with_indices(&v, &[1, n], 1);
            let mut want: Vec<usize> = (0..n).collect();
            want.sort_by(|&a, &b| k::sort_order(v[b], v[a]));
            stable_lanes += (idx.iter().map(|&i| i as usize).collect::<Vec<_>>() == want) as usize;
        }
        eprintln!("metal cpu_ref sort ties n {n:>5}: {stable_lanes}/50 lanes in first-index order");
    }
}
