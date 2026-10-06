//! Sweeps of the CUDA f32 exp, ln, pow and sigmoid (kernels.rs) against CpuRef (Rust's f32
//! functions, which CpuRef's kernels call) and against the correctly rounded result (the f64
//! value rounded to f32; results within 2^-50 of an f32 midpoint are counted as undecided and
//! compared with CpuRef only). Reports per function: values, the largest ulp distance from
//! CpuRef, how many differ from CpuRef, from the correctly rounded value, and how many differ
//! where CpuRef itself is correctly rounded.
//!
//! `cuda_math_dense` (every 4099th f32 of each domain) runs in the normal suite, in a fresh
//! process (a test elsewhere may pin the shared one). `cuda_math_exhaustive` (every
//! f32 of the exp and ln domains; pow every 64th x per exponent) is ignored:
//! `cargo test --release --features cuda --lib cuda_math_exhaustive -- --ignored --nocapture`.

use crate::tensor::{self as nt, Device, Tensor};

const M: Device = Device::Cuda(0);

/// Distance in units in the last place (0 for equal bits or two NaNs; infinities count by bits).
fn ulps(a: f32, b: f32) -> u64 {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN.wrapping_sub(i) as i64 } else { i as i64 }
    };
    (key(a) - key(b)).unsigned_abs()
}

/// The correctly rounded f32 of `v` (an f64 accurate to about 2^-52), or None when `v` is within
/// 2^-50·|v| of an f32 rounding midpoint.
fn cr_of(v: f64) -> Option<f32> {
    if !v.is_finite() || v == 0.0 {
        return Some(v as f32);
    }
    let f = v as f32;
    let g = if (f as f64) < v { f32::from_bits(f.to_bits().wrapping_add(if f > 0.0 { 1 } else { u32::MAX })) } else { f32::from_bits(f.to_bits().wrapping_add(if f > 0.0 { u32::MAX } else { 1 })) };
    if !g.is_finite() || f.is_infinite() {
        return Some(f);
    }
    let mid = (f as f64 + g as f64) / 2.0;
    if (v - mid).abs() <= v.abs() * 2f64.powi(-50) { None } else { Some(f) }
}

#[derive(Default, Debug)]
struct Stats {
    n: u64,
    max_ulp: u64,
    ne_cpu: u64,
    ne_cr: u64,
    undecided: u64,
    cpu_ne_cr: u64,
    /// GPU ≠ CR where CpuRef = CR (the cases the aim is bit-equality on).
    ne_where_cpu_cr: u64,
    /// Undecided cases (near a midpoint) where the GPU differs from CpuRef.
    undecided_ne_cpu: u64,
    /// A few differing inputs (x, gpu, cpu, cr).
    examples: Vec<(f32, f32, f32, Option<f32>)>,
}

impl Stats {
    fn add(&mut self, x: f32, gpu: f32, cpu: f32, cr: Option<f32>) {
        self.n += 1;
        let u = ulps(gpu, cpu);
        self.max_ulp = self.max_ulp.max(u);
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        if !same(gpu, cpu) {
            self.ne_cpu += 1;
            if self.examples.len() < 6 {
                self.examples.push((x, gpu, cpu, cr));
            }
        }
        match cr {
            None => {
                self.undecided += 1;
                if !same(gpu, cpu) {
                    self.undecided_ne_cpu += 1;
                    if self.examples.len() < 12 {
                        self.examples.push((x, gpu, cpu, None));
                    }
                }
            }
            Some(c) => {
                if !same(gpu, c) {
                    self.ne_cr += 1;
                }
                if !same(cpu, c) {
                    self.cpu_ne_cr += 1;
                } else if !same(gpu, c) {
                    self.ne_where_cpu_cr += 1;
                }
            }
        }
    }

    fn merge(&mut self, o: Stats) {
        self.n += o.n;
        self.max_ulp = self.max_ulp.max(o.max_ulp);
        self.ne_cpu += o.ne_cpu;
        self.ne_cr += o.ne_cr;
        self.undecided += o.undecided;
        self.cpu_ne_cr += o.cpu_ne_cr;
        self.ne_where_cpu_cr += o.ne_where_cpu_cr;
        self.undecided_ne_cpu += o.undecided_ne_cpu;
        for e in o.examples {
            if self.examples.len() < 12 {
                self.examples.push(e);
            }
        }
    }

    fn report(&self, name: &str) {
        eprintln!(
            "cuda math {name:<14} {:>11} values: max {} ulp from CpuRef; ≠ CpuRef {} ; ≠ correctly rounded {} ; CpuRef ≠ correctly rounded {} ; ≠ where CpuRef is correctly rounded {} ; undecided {} (≠ CpuRef there {})",
            self.n, self.max_ulp, self.ne_cpu, self.ne_cr, self.cpu_ne_cr, self.ne_where_cpu_cr, self.undecided, self.undecided_ne_cpu
        );
        for (x, g, c, r) in self.examples.iter().filter(|e| e.3.is_none()).chain(self.examples.iter().filter(|e| e.3.is_some()).take(3)) {
            eprintln!("cuda math {name:<14}   x {x:e} ({:#010x}): cuda {g:e} ({:#010x}), cpu {c:e} ({:#010x}), cr {:?}", x.to_bits(), g.to_bits(), c.to_bits(), r.map(|v| format!("{:#010x}", v.to_bits())));
        }
    }
}

/// `gpu` of each chunk of `xs` on the device against `cpu` and `exact` on host threads.
fn sweep(xs: impl Iterator<Item = f32>, gpu: &dyn Fn(Tensor) -> Tensor, cpu: &(dyn Fn(f32) -> f32 + Sync), exact: &(dyn Fn(f32) -> f64 + Sync), decide: &(dyn Fn(f32) -> Option<f32> + Sync)) -> Stats {
    const CHUNK: usize = 1 << 23;
    let mut total = Stats::default();
    let mut xs = xs.peekable();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    while xs.peek().is_some() {
        let chunk: Vec<f32> = xs.by_ref().take(CHUNK).collect();
        let n = chunk.len();
        let g = gpu(Tensor::from_data(chunk.clone(), [n]).to(M)).to_vec();
        let per = n.div_ceil(threads);
        let parts: Vec<Stats> = std::thread::scope(|s| {
            let hs: Vec<_> = chunk
                .chunks(per)
                .zip(g.chunks(per))
                .map(|(xc, gc)| {
                    s.spawn(move || {
                        let mut st = Stats::default();
                        for (&x, &gv) in xc.iter().zip(gc) {
                            st.add(x, gv, cpu(x), cr_of(exact(x)).or_else(|| decide(x)));
                        }
                        st
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for p in parts {
            total.merge(p);
        }
    }
    total
}

/// Every `stride`-th f32 bit pattern in [lo, hi] (inclusive; both finite), plus both ends.
fn range(lo: f32, hi: f32, stride: u32) -> Vec<f32> {
    let mut v = Vec::new();
    let mut push_run = |a: u32, b: u32, neg: bool| {
        let mut i = a;
        while i <= b {
            v.push(f32::from_bits(if neg { i | 0x8000_0000 } else { i }));
            match i.checked_add(stride) {
                Some(n) => i = n,
                None => break,
            }
        }
        v.push(f32::from_bits(if neg { b | 0x8000_0000 } else { b }));
    };
    if lo < 0.0 {
        push_run(0, (-lo).to_bits(), true);
    }
    if hi > 0.0 {
        push_run(0, hi.to_bits(), false);
    }
    v
}

const SPECIALS: [f32; 12] = [0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MIN_POSITIVE, 1e-45, f32::MAX, 88.72283, -103.97208];

/// A big unsigned integer, little-endian 32-bit limbs (exact pow results).
#[derive(Clone)]
struct Big(Vec<u32>);

impl Big {
    fn mul(&self, k: u64) -> Big {
        let mut out = Vec::with_capacity(self.0.len() + 2);
        let mut carry: u128 = 0;
        for &l in &self.0 {
            let v = l as u128 * k as u128 + carry;
            out.push(v as u32);
            carry = v >> 32;
        }
        while carry > 0 {
            out.push(carry as u32);
            carry >>= 32;
        }
        Big(out)
    }
    fn bits(&self) -> u32 {
        let mut n = self.0.len();
        while n > 0 && self.0[n - 1] == 0 {
            n -= 1;
        }
        if n == 0 { 0 } else { 32 * (n as u32 - 1) + (32 - self.0[n - 1].leading_zeros()) }
    }
    fn bit(&self, i: u32) -> bool {
        self.0.get((i / 32) as usize).is_some_and(|l| (l >> (i % 32)) & 1 == 1)
    }
    /// The integer `self >> sh` (at most 64 bits wanted) and whether any shifted-out bit below
    /// the top one is set.
    fn shr(&self, sh: u32) -> (u64, bool, bool) {
        let mut q: u64 = 0;
        let top = self.bits();
        for i in (sh..top).rev() {
            q = (q << 1) | self.bit(i) as u64;
        }
        let half = sh > 0 && self.bit(sh - 1);
        let rest = (0..sh.saturating_sub(1)).any(|i| self.bit(i));
        (q, half, rest)
    }
}

/// The exact, correctly rounded (ties to even) f32 of x^y for x > 0 when x^y is a dyadic
/// rational: y a positive integer ≤ 16, or k + 1/2 with x a perfect square; None otherwise.
fn exact_pow(x: f32, y: f32) -> Option<f32> {
    if !(x > 0.0) || !x.is_finite() || !(y > 0.0) || y > 16.0 {
        return None;
    }
    // x = m·2^e with m odd.
    let b = x.to_bits();
    let (mut m, mut e) = if b >> 23 == 0 { ((b & 0x7fffff) as u64, -149i64) } else { (((b & 0x7fffff) | 0x800000) as u64, ((b >> 23) as i64) - 150) };
    while m % 2 == 0 {
        m /= 2;
        e += 1;
    }
    let (int, root) = if y == y.floor() {
        (y as u32, None)
    } else if (y - 0.5) == (y - 0.5).floor() {
        let s = (m as f64).sqrt().round() as u64;
        if s * s != m || e % 2 != 0 {
            return None;
        }
        ((y - 0.5) as u32, Some((s, e / 2)))
    } else {
        return None;
    };
    let mut big = Big(vec![1]);
    for _ in 0..int {
        big = big.mul(m);
    }
    let mut ex = e * int as i64;
    if let Some((s, h)) = root {
        big = big.mul(s);
        ex += h;
    }
    // Round big·2^ex to f32.
    let l = big.bits() as i64;
    let top = l - 1 + ex;
    if top >= 128 {
        return Some(f32::INFINITY);
    }
    let lsb = if top >= -126 { top - 23 } else { -149 };
    let sh = lsb - ex;
    if sh <= 0 {
        return Some(((big.shr(0).0 as f64) * 2f64.powi(ex as i32)) as f32);
    }
    let (mut q, half, rest) = big.shr(sh as u32);
    if half && (rest || q % 2 == 1) {
        q += 1;
    }
    Some(((q as f64) * 2f64.powi(lsb as i32)) as f32)
}

/// The stable sigmoid with its exp correctly rounded (the f64 value rounded to f32), the
/// other steps in f32 as the kernel does them.
fn sigmoid_cr(a: f32) -> f32 {
    if a >= 0.0 {
        let e = ((-a) as f64).exp() as f32;
        1.0 / (1.0 + e)
    } else {
        let e = (a as f64).exp() as f32;
        e / (1.0 + e)
    }
}

fn run(stride: u32, pow_stride: u32) {
    let exp_xs = range(-104.0, 89.0, stride).into_iter().chain(SPECIALS);
    let s = sweep(exp_xs, &|t| t.exp(), &|x| x.exp(), &|x| (x as f64).exp(), &|_| None);
    s.report("exp");
    assert!(s.max_ulp <= 1, "exp: {} ulp", s.max_ulp);
    let ln_xs = range(0.0, f32::MAX, stride).into_iter().chain(SPECIALS).chain([-2.0, -1e-30]);
    let s = sweep(ln_xs, &|t| t.log(), &|x| x.ln(), &|x| (x as f64).ln(), &|_| None);
    s.report("ln");
    assert!(s.max_ulp <= 1, "ln: {} ulp", s.max_ulp);
    for y in [0.5f32, -0.5, 1.5, 2.5, 3.0, 0.3, -1.7, 7.0] {
        let xs = range(0.0, f32::MAX, pow_stride).into_iter().chain(SPECIALS).chain([-2.0, -0.5, -3.25]);
        let s = sweep(xs, &|t| t.powf_scalar(y), &|x| x.powf(y), &|x| (x as f64).powf(y as f64), &|x| exact_pow(x, y));
        s.report(&format!("pow y={y}"));
        assert!(s.max_ulp <= 1, "pow y={y}: {} ulp", s.max_ulp);
    }
    // Sigmoid (the stable form: one exp, then f32 steps), against CpuRef's own sigmoid and
    // against the same formula with a correctly rounded exp, reported by band of |x|. CpuRef's
    // libm (not correctly rounded everywhere) makes CpuRef's sigmoid differ from the correctly
    // rounded step; the GPU's does not. (Under burn's exp(−ln(exp(−x) + 1)), the distance grew
    // with |x|; the stable form keeps it about one ulp of exp.)
    let xs: Vec<f32> = range(-104.0, 104.0, stride.max(64));
    let n = xs.len();
    let cpu = nt::sigmoid(Tensor::from_data(xs.clone(), [n])).to_vec();
    let gpu = nt::sigmoid(Tensor::from_data(xs.clone(), [n]).to(M)).to_vec();
    let mut st = Stats::default();
    let mut bands = [(2.0f32, 0u64), (8.0, 0), (16.0, 0), (32.0, 0), (104.0, 0)];
    for i in 0..n {
        st.add(xs[i], gpu[i], cpu[i], Some(sigmoid_cr(xs[i])));
        let u = ulps(gpu[i], cpu[i]);
        for b in bands.iter_mut() {
            if xs[i].abs() <= b.0 {
                b.1 = b.1.max(u);
            }
        }
    }
    st.report("sigmoid");
    eprintln!("cuda math sigmoid bands (|x| ≤ bound: max ulp from CpuRef): {}", bands.iter().map(|(b, u)| format!("{b}: {u}")).collect::<Vec<_>>().join(", "));
    // The GPU sigmoid is the correctly rounded-step formula (up to the few non-correctly-rounded
    // exp/ln results); against CpuRef, 4 ulp on |x| ≤ 2 (the forward bound's range).
    assert!(st.ne_cr * 1_000_000 <= st.n, "sigmoid: {} of {} differ from the correctly rounded steps", st.ne_cr, st.n);
    assert!(bands[0].1 <= 4, "sigmoid on |x| ≤ 2: {bands:?}");
}

#[test]
#[ignore = "run by its wrapper in a fresh process"]
fn body_cuda_math_dense() {
    run(4099, 4099 * 8);
}

#[test]
#[ignore = "exhaustive; minutes"]
fn cuda_math_exhaustive() {
    run(1, 64);
}

/// The dense sweep, in a fresh process (the reference pin is process-wide).
#[test]
fn cuda_math_dense() {
    let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args(["tensor::cuda::sweep::body_cuda_math_dense", "--exact", "--ignored", "--test-threads=1", "--nocapture"])
        .output()
        .expect("run the test binary");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success() && text.contains("1 passed"), "the dense sweep in a fresh process:\n{text}");
    for l in text.lines().filter(|l| l.starts_with("cuda math")) {
        eprintln!("{l}");
    }
}

/// The exact pow reference itself (host only).
#[test]
fn exact_pow_reference() {
    assert_eq!(exact_pow(3.0, 3.0), Some(27.0));
    assert_eq!(exact_pow(4.0, 1.5), Some(8.0));
    assert_eq!(exact_pow(2.0, 0.5), None, "√2 is not dyadic");
    assert_eq!(exact_pow(0.3, 0.3), None);
    // (1 + 2^-8)^3 = 1 + 3·2^-8 + 3·2^-16 + 2^-24: a midpoint, ties to even.
    let x = 1.0 + 2f32.powi(-8);
    let want = ((1.0f64 + 2f64.powi(-8)).powi(3)) as f32;
    assert_eq!(exact_pow(x, 3.0), Some(want));
    // A subnormal midpoint (13.5 · 2^-149 rounds to 14).
    assert_eq!(exact_pow(f32::from_bits(0x0f10_0000), 1.5).map(f32::to_bits), Some(0xe));
    for (x, y) in [(1.7f32, 3.0f32), (0.37, 7.0), (123.25, 2.0), (6.25, 2.5), (1e-20, 3.0)] {
        let e = exact_pow(x, y).unwrap();
        let f = (x as f64).powf(y as f64) as f32;
        assert!(ulps(e, f) <= 1, "{x}^{y}: {e} vs {f}");
    }
}
