//! `nn::rl` on the CPU reference backend: forward values against f64 host references and
//! hand-computed values, f64 central finite differences, masking and edge cases (every position
//! masked, one scored token, ±large and −∞ logits), group advantages and the sampler.

use super::joint_embed::fd_input;
use super::*;
use crate::nn::*;
use crate::tensor::DType;

fn t64(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F64)
}

fn t32(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_f64s(v, shape.to_vec(), DType::F32)
}

/// log-softmax of each row of `x` (rows of `v`) on the host, max-shifted.
fn log_softmax_host(x: &[f64], v: usize) -> Vec<f64> {
    x.chunks(v)
        .flat_map(|row| {
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = row.iter().map(|a| (a - m).exp()).sum::<f64>().ln() + m;
            row.iter().map(move |a| a - lse).collect::<Vec<_>>()
        })
        .collect()
}

/// The host reduction of per-token values `[S, B, T]` (zero at masked positions) under `mask`.
fn reduce_host(per: &[f64], mask: &[f64], [s, b, t]: [usize; 3], r: TokenReduction) -> Vec<f64> {
    let seq = |si: usize, bi: usize| -> (f64, f64) {
        let o = (si * b + bi) * t;
        (per[o..o + t].iter().sum(), mask[o..o + t].iter().sum())
    };
    match r {
        TokenReduction::None => per.to_vec(),
        TokenReduction::Sum => (0..s).map(|si| (0..b).map(|bi| seq(si, bi).0).sum()).collect(),
        TokenReduction::TokenMean => (0..s).map(|si| (0..b).map(|bi| seq(si, bi).0).sum::<f64>() / (0..b).map(|bi| seq(si, bi).1).sum::<f64>().max(1.0)).collect(),
        TokenReduction::PerSequence => (0..s).flat_map(|si| (0..b).map(move |bi| (si, bi))).map(|(si, bi)| { let (v, n) = seq(si, bi); v / n.max(1.0) }).collect(),
        TokenReduction::SequenceMean => (0..s)
            .map(|si| {
                let (mut tot, mut k) = (0.0, 0.0);
                for bi in 0..b {
                    let (v, n) = seq(si, bi);
                    if n > 0.0 {
                        tot += v / n;
                        k += 1.0;
                    }
                }
                tot / f64::max(k, 1.0)
            })
            .collect(),
    }
}

const REDUCTIONS: [TokenReduction; 5] = [TokenReduction::None, TokenReduction::Sum, TokenReduction::TokenMean, TokenReduction::PerSequence, TokenReduction::SequenceMean];

/// A batch: S 2, B 3, T 4, V 5; sequence 2 of seed 1 entirely masked, sequence 0 of seed 0 with
/// one scored token, the rest random.
struct Case {
    sh: [usize; 3],
    v: usize,
    logits: Vec<f64>,
    tokens: Vec<i64>,
    mask: Vec<f64>,
}

fn case(seed: u64, lo: f64, hi: f64) -> Case {
    let (s, b, t, v) = (2, 3, 4, 5);
    let n = s * b * t;
    let mut mask: Vec<f64> = rnd(n, seed + 2, 0.0, 1.0).into_iter().map(|u| if u < 0.7 { 1.0 } else { 0.0 }).collect();
    mask[0..t].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
    let z = (b + 2) * t;
    mask[z..z + t].iter_mut().for_each(|m| *m = 0.0);
    Case { sh: [s, b, t], v, logits: rnd(n * v, seed, lo, hi), tokens: rnd_ints(n, seed + 1, v as i64), mask }
}

impl Case {
    fn n(&self) -> usize {
        self.sh.iter().product()
    }
    fn lshape(&self) -> Vec<usize> {
        vec![self.sh[0], self.sh[1], self.sh[2], self.v]
    }
    fn tokens(&self) -> IntTensor {
        IntTensor::from_data(self.tokens.clone(), self.sh)
    }
    fn mask(&self, dtype: DType) -> Tensor {
        Tensor::from_f64s(self.mask.clone(), self.sh.to_vec(), dtype)
    }
    /// The host log-probabilities of the tokens.
    fn logp_host(&self) -> Vec<f64> {
        let lp = log_softmax_host(&self.logits, self.v);
        (0..self.n()).map(|i| lp[i * self.v + self.tokens[i] as usize]).collect()
    }
}

// ------------------------------------------------------------------------- token_log_probs

/// Against the host log-softmax in f64 (1e-14) and f32 (1e-6); equal to `−nll` to the bit; any
/// leading shape.
#[test]
fn token_log_probs_match_f64_reference() {
    let c = case(800, -6.0, 6.0);
    let want = c.logp_host();
    for (dtype, tol) in [(DType::F64, 1e-14), (DType::F32, 1e-6)] {
        let lp = token_log_probs(Tensor::from_f64s(c.logits.clone(), c.lshape(), dtype), &c.tokens());
        assert_eq!(lp.shape(), &c.sh);
        assert_close(&format!("token_log_probs {dtype:?}"), &lp.to_vec_f64(), &want, tol);
        let nll = nll(Tensor::from_f64s(c.logits.clone(), c.lshape(), dtype), &c.tokens()).to_vec_f64();
        assert_eq!(lp.to_vec_f64(), nll.iter().map(|x| -x).collect::<Vec<_>>(), "{dtype:?}: −nll to the bit");
    }
    // Rank 2 ([N, V] → [N]) and rank 1 ([V] → []).
    let lp2 = token_log_probs(t64(c.logits.clone(), &[c.n(), c.v]), &IntTensor::from_data(c.tokens.clone(), [c.n()]));
    assert_eq!(lp2.to_vec_f64(), token_log_probs(t64(c.logits.clone(), &c.lshape()), &c.tokens()).to_vec_f64());
    // The gradient is onehot − softmax.
    let x = t64(c.logits.clone(), &c.lshape()).require_grad();
    let g = token_log_probs(x.clone(), &c.tokens()).sum().backward();
    let lp = log_softmax_host(&c.logits, c.v);
    let want: Vec<f64> = (0..c.n() * c.v).map(|j| if c.tokens[j / c.v] as usize == j % c.v { 1.0 } else { 0.0 } - lp[j].exp()).collect();
    assert_close("token_log_probs gradient", &x.grad(&g).unwrap().to_vec_f64(), &want, 1e-14);
}

#[test]
fn token_log_probs_match_finite_differences() {
    let c = case(810, -3.0, 3.0);
    let tok = c.tokens();
    fd_input("token_log_probs", &c.logits, &c.lshape(), &|x| token_log_probs(x, &tok), 1e-6, 1e-7);
}

/// Logits of ±1e4 are finite in f32 and f64; a token excluded by −∞ gives −∞, the others stay
/// finite with finite gradients (zero at the excluded entry).
#[test]
fn token_log_probs_large_and_excluded_logits() {
    let v = 4;
    let logits = vec![1e4, -1e4, 0.0, 1e3, -1e4, -1e4, 1e4, 1e4];
    for dtype in [DType::F32, DType::F64] {
        let x = Tensor::from_f64s(logits.clone(), vec![2, v], dtype).require_grad();
        let lp = token_log_probs(x.clone(), &IntTensor::from_data(vec![0, 2], [2]));
        let lv = lp.to_vec_f64();
        assert!(lv[0].abs() < 1e-6 && (lv[1] - 0.5f64.ln()).abs() < 1e-6, "{dtype:?}: {lv:?}");
        let gv = x.grad(&lp.sum().backward()).unwrap().to_vec_f64();
        assert_close(&format!("{dtype:?} large-logit gradient"), &gv, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5], 1e-6);
        let row = vec![1.0, f64::NEG_INFINITY, 0.0, 2.0];
        let x = Tensor::from_f64s([row.clone(), row.clone()].concat(), vec![2, v], dtype).require_grad();
        let lp = token_log_probs(x.clone(), &IntTensor::from_data(vec![1, 3], [2]));
        let lv = lp.to_vec_f64();
        assert_eq!(lv[0], f64::NEG_INFINITY, "{dtype:?}: excluded token");
        let want = log_softmax_host(&[1.0, 0.0, 2.0], 3)[2];
        assert!((lv[1] - want).abs() < 1e-6, "{dtype:?}: {} vs {want}", lv[1]);
        // Only the finite row is differentiated (the −∞ one would give −∞ upstream).
        let g = (lp.slice([1..2])).sum().backward();
        let gv = x.grad(&g).unwrap().to_vec_f64();
        assert!(gv.iter().all(|v| v.is_finite()) && gv[5] == 0.0, "{dtype:?}: gradient {gv:?}");
    }
}

// ------------------------------------------------------------------- policy-gradient loss

/// Hand-computed reductions: logp `[-1 … -6]` as `[1, 2, 3]`, per-sequence weights `[1, −2]`,
/// mask `[1 1 0; 0 1 0]` give per-token `[1 2 0; 0 −10 0]`, sum −7, token mean −7/3, per
/// sequence `[3/2, −10]`, sequence mean −4.25.
#[test]
fn policy_gradient_reductions_by_hand() {
    let lp = t64(vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], &[1, 2, 3]);
    let w = t64(vec![1.0, -2.0], &[1, 2]);
    let m = t64(vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0], &[1, 2, 3]);
    let run = |r| policy_gradient_loss(lp.clone(), &w, Some(&m), r).to_vec_f64();
    assert_eq!(run(TokenReduction::None), vec![1.0, 2.0, 0.0, 0.0, -10.0, 0.0]);
    assert_eq!(run(TokenReduction::Sum), vec![-7.0]);
    assert_close("token mean", &run(TokenReduction::TokenMean), &[-7.0 / 3.0], 1e-15);
    assert_eq!(run(TokenReduction::PerSequence), vec![1.5, -10.0]);
    assert_eq!(run(TokenReduction::SequenceMean), vec![-4.25]);
    assert_eq!(TokenReduction::default(), TokenReduction::TokenMean);
    // No mask scores every token: per-token weights [S, B, T] equal to the broadcast ones.
    let wt = t64(vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0], &[1, 2, 3]);
    assert_eq!(policy_gradient_loss(lp.clone(), &wt, None, TokenReduction::None).to_vec_f64(), policy_gradient_loss(lp.clone(), &w, None, TokenReduction::None).to_vec_f64());
    assert_eq!(policy_gradient_loss(lp, &w, None, TokenReduction::Sum).to_vec_f64(), vec![6.0 - 30.0]);
}

/// Every reduction with per-token and per-sequence signed weights against the host, in f64 and
/// f32; the weights get no gradient.
#[test]
fn policy_gradient_matches_f64_reference() {
    let c = case(820, -4.0, 4.0);
    let [s, b, t] = c.sh;
    let wtok = rnd(c.n(), 821, -2.0, 2.0);
    let wseq = rnd(s * b, 822, -2.0, 2.0);
    let lp = c.logp_host();
    for per_seq in [false, true] {
        let w = |i: usize| if per_seq { wseq[i / t] } else { wtok[i] };
        let host: Vec<f64> = (0..c.n()).map(|i| if c.mask[i] == 1.0 { -w(i) * lp[i] } else { 0.0 }).collect();
        for r in REDUCTIONS {
            let want = reduce_host(&host, &c.mask, c.sh, r);
            for (dtype, tol) in [(DType::F64, 1e-13), (DType::F32, 1e-6)] {
                let wt = if per_seq { Tensor::from_f64s(wseq.clone(), vec![s, b], dtype) } else { Tensor::from_f64s(wtok.clone(), c.sh.to_vec(), dtype) };
                let x = Tensor::from_f64s(c.logits.clone(), c.lshape(), dtype);
                let got = policy_gradient_loss(token_log_probs(x, &c.tokens()), &wt, Some(&c.mask(dtype)), r).to_vec_f64();
                assert_close(&format!("PG per_seq {per_seq} {r:?} {dtype:?}"), &got, &want, tol);
            }
        }
    }
    let wt = t64(wtok, &c.sh).require_grad();
    let x = t64(c.logits.clone(), &c.lshape()).require_grad();
    let g = policy_gradient_loss(token_log_probs(x.clone(), &c.tokens()), &wt, Some(&c.mask(DType::F64)), TokenReduction::Sum).sum().backward();
    assert!(wt.grad(&g).is_none(), "weights are detached");
    assert!(x.grad(&g).is_some());
}

#[test]
fn policy_gradient_matches_finite_differences() {
    let c = case(830, -3.0, 3.0);
    let tok = c.tokens();
    let m = c.mask(DType::F64);
    let w = t64(rnd(c.n(), 831, -2.0, 2.0), &c.sh);
    let ws = t64(rnd(c.sh[0] * c.sh[1], 832, -2.0, 2.0), &c.sh[..2]);
    for r in REDUCTIONS {
        fd_input(&format!("PG per-token {r:?}"), &c.logits, &c.lshape(), &|x| policy_gradient_loss(token_log_probs(x, &tok), &w, Some(&m), r), 1e-6, 1e-7);
        fd_input(&format!("PG per-sequence {r:?}"), &c.logits, &c.lshape(), &|x| policy_gradient_loss(token_log_probs(x, &tok), &ws, Some(&m), r), 1e-6, 1e-7);
    }
}

// -------------------------------------------------------------------------------- PPO clip

/// Old log-probabilities `logp + o` with offsets o chosen so that every ratio exp(−o) is away
/// from the clip edges 0.8 and 1.2: 1.65, 1.105, 0.95, 0.67 (both sides of the range, inside
/// and outside), with advantages of both signs.
fn ppo_case(seed: u64) -> (Case, Vec<f64>, Vec<f64>) {
    let c = case(seed, -3.0, 3.0);
    let lp = c.logp_host();
    let offs = [-0.5, -0.1, 0.05, 0.4];
    let old: Vec<f64> = lp.iter().enumerate().map(|(i, l)| l + offs[(i * 7 + i / 3) % 4]).collect();
    let adv: Vec<f64> = rnd(c.n(), seed + 5, -2.0, 2.0).into_iter().map(|a| if a.abs() < 0.1 { 0.5 } else { a }).collect();
    (c, old, adv)
}

fn ppo_host(lp: &[f64], old: &[f64], adv: &[f64], mask: &[f64], clip: Clip) -> Vec<f64> {
    (0..lp.len())
        .map(|i| {
            if mask[i] == 0.0 {
                return 0.0;
            }
            let r = (lp[i] - old[i]).exp();
            let rc = r.clamp(1.0 - clip.low, 1.0 + clip.high);
            (-(r * adv[i])).max(-(rc * adv[i]))
        })
        .collect()
}

#[test]
fn clipped_policy_loss_matches_f64_reference() {
    let (c, old, adv) = ppo_case(840);
    let lp = c.logp_host();
    for clip in [Clip::symmetric(0.2), Clip { low: 0.2, high: 0.28 }] {
        let host = ppo_host(&lp, &old, &adv, &c.mask, clip);
        for r in REDUCTIONS {
            let want = reduce_host(&host, &c.mask, c.sh, r);
            for (dtype, tol) in [(DType::F64, 1e-13), (DType::F32, 1e-6)] {
                let x = Tensor::from_f64s(c.logits.clone(), c.lshape(), dtype);
                let got = clipped_policy_loss(token_log_probs(x, &c.tokens()), &Tensor::from_f64s(old.clone(), c.sh.to_vec(), dtype), &Tensor::from_f64s(adv.clone(), c.sh.to_vec(), dtype), Some(&c.mask(dtype)), clip, r).to_vec_f64();
                assert_close(&format!("PPO {clip:?} {r:?} {dtype:?}"), &got, &want, tol);
            }
        }
        // The clip fraction: scored tokens with A > 0, r > 1 + high or A < 0, r < 1 − low.
        let mut want = vec![0.0; c.sh[0]];
        let per = c.n() / c.sh[0];
        for i in 0..c.n() {
            let r = (lp[i] - old[i]).exp();
            if c.mask[i] == 1.0 && ((adv[i] > 0.0 && r > 1.0 + clip.high) || (adv[i] < 0.0 && r < 1.0 - clip.low)) {
                want[i / per] += 1.0;
            }
        }
        for (si, w) in want.iter_mut().enumerate() {
            *w /= c.mask[si * per..(si + 1) * per].iter().sum::<f64>();
        }
        assert!(want.iter().all(|w| *w > 0.0 && *w < 1.0), "the case clips some tokens: {want:?}");
        let lpt = token_log_probs(t64(c.logits.clone(), &c.lshape()), &c.tokens());
        let got = clip_fraction(&lpt, &t64(old.clone(), &c.sh), &t64(adv.clone(), &c.sh), Some(&c.mask(DType::F64)), clip).to_vec_f64();
        assert_close(&format!("clip fraction {clip:?}"), &got, &want, 1e-15);
    }
}

/// Finite differences away from the clip edges; at `old = logp` the gradient equals the
/// policy-gradient loss's with the advantages as weights (within 1e-14); clipped-and-active
/// tokens get zero gradient.
#[test]
fn clipped_policy_loss_gradients() {
    let (c, old, adv) = ppo_case(850);
    let tok = c.tokens();
    let m = c.mask(DType::F64);
    let (ot, at) = (t64(old.clone(), &c.sh), t64(adv.clone(), &c.sh));
    for r in REDUCTIONS {
        fd_input(&format!("PPO {r:?}"), &c.logits, &c.lshape(), &|x| clipped_policy_loss(token_log_probs(x, &tok), &ot, &at, Some(&m), Clip::symmetric(0.2), r), 1e-6, 1e-7);
    }
    let lp = t64(c.logp_host(), &c.sh).require_grad();
    let on = token_log_probs(t64(c.logits.clone(), &c.lshape()), &tok).detach();
    let grad_ppo = clipped_policy_loss(lp.clone(), &on, &at, Some(&m), Clip::symmetric(0.2), TokenReduction::TokenMean).sum().backward();
    let grad_pg = policy_gradient_loss(lp.clone(), &at, Some(&m), TokenReduction::TokenMean).sum().backward();
    assert_close("PPO at old = logp vs PG", &lp.grad(&grad_ppo).unwrap().to_vec_f64(), &lp.grad(&grad_pg).unwrap().to_vec_f64(), 1e-14);
    // Per token: −A·r inside the range or on the unclipped side, 0 where clipped and active.
    let lpv = c.logp_host();
    let lt = t64(lpv.clone(), &c.sh).require_grad();
    let ot2 = ot.clone().require_grad();
    let at2 = at.clone().require_grad();
    let g = clipped_policy_loss(lt.clone(), &ot2, &at2, Some(&m), Clip::symmetric(0.2), TokenReduction::Sum).sum().backward();
    assert!(ot2.grad(&g).is_none() && at2.grad(&g).is_none(), "old log-probabilities and advantages are detached");
    let gv = lt.grad(&g).unwrap().to_vec_f64();
    for i in 0..c.n() {
        let r = (lpv[i] - old[i]).exp();
        let clipped = (adv[i] > 0.0 && r > 1.2) || (adv[i] < 0.0 && r < 0.8);
        let want = if c.mask[i] == 0.0 || clipped { 0.0 } else { -adv[i] * r };
        assert!((gv[i] - want).abs() <= 1e-14 * (1.0 + want.abs()), "PPO d/dlogp[{i}]: {} vs {want}", gv[i]);
    }
}

/// Arbitrary old values at masked positions (±1e4, which would overflow exp) leave the loss and
/// the gradient finite and unchanged.
#[test]
fn clipped_policy_loss_ignores_garbage_at_masked_positions() {
    let (c, old, adv) = ppo_case(860);
    let m = c.mask(DType::F32);
    let garbage: Vec<f64> = old.iter().zip(&c.mask).enumerate().map(|(i, (o, k))| if *k == 0.0 { if i % 2 == 0 { -1e4 } else { 1e4 } } else { *o }).collect();
    let run = |o: &[f64]| {
        let x = t32(c.logits.clone(), &c.lshape()).require_grad();
        let l = clipped_policy_loss(token_log_probs(x.clone(), &c.tokens()), &t32(o.to_vec(), &c.sh), &t32(adv.clone(), &c.sh), Some(&m), Clip::symmetric(0.2), TokenReduction::TokenMean);
        let g = x.grad(&l.clone().sum().backward()).unwrap().to_vec_f64();
        (l.to_vec_f64(), g)
    };
    let (a, b) = (run(&old), run(&garbage));
    assert!(b.0.iter().chain(&b.1).all(|v| v.is_finite()));
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------------------- KL

fn kl_host(p: &[f64], q: &[f64], v: usize) -> Vec<f64> {
    let (lp, lq) = (log_softmax_host(p, v), log_softmax_host(q, v));
    lp.chunks(v).zip(lq.chunks(v)).map(|(a, b)| a.iter().zip(b).map(|(x, y)| if *x == f64::NEG_INFINITY { 0.0 } else { x.exp() * (x - y) }).sum()).collect()
}

fn entropy_host(x: &[f64], v: usize) -> Vec<f64> {
    log_softmax_host(x, v).chunks(v).map(|r| -r.iter().map(|l| if *l == f64::NEG_INFINITY { 0.0 } else { l.exp() * l }).sum::<f64>()).collect()
}

/// Exact KL and entropy against the host per reduction, f64 and f32; KL(π ‖ π) = 0; the
/// reference gets no gradient.
#[test]
fn kl_and_entropy_match_f64_reference() {
    let c = case(870, -4.0, 4.0);
    let refl = rnd(c.n() * c.v, 871, -4.0, 4.0);
    let mk = |v: Vec<f64>| v.iter().zip(&c.mask).map(|(a, m)| a * m).collect::<Vec<_>>();
    let kl = mk(kl_host(&c.logits, &refl, c.v));
    let h = mk(entropy_host(&c.logits, c.v));
    for r in REDUCTIONS {
        for (dtype, tol) in [(DType::F64, 1e-13), (DType::F32, 2e-6)] {
            let x = Tensor::from_f64s(c.logits.clone(), c.lshape(), dtype);
            let rt = Tensor::from_f64s(refl.clone(), c.lshape(), dtype);
            assert_close(&format!("KL {r:?} {dtype:?}"), &categorical_kl(x.clone(), &rt, Some(&c.mask(dtype)), r).to_vec_f64(), &reduce_host(&kl, &c.mask, c.sh, r), tol);
            assert_close(&format!("entropy {r:?} {dtype:?}"), &categorical_entropy(x, Some(&c.mask(dtype)), r).to_vec_f64(), &reduce_host(&h, &c.mask, c.sh, r), tol);
        }
    }
    let x = t64(c.logits.clone(), &c.lshape());
    assert!(categorical_kl(x.clone(), &x, None, TokenReduction::None).to_vec_f64().iter().all(|v| v.abs() < 1e-15));
    let xr = x.clone().require_grad();
    let rt = t64(refl, &c.lshape()).require_grad();
    let g = categorical_kl(xr.clone(), &rt, None, TokenReduction::Sum).sum().backward();
    assert!(rt.grad(&g).is_none() && xr.grad(&g).is_some(), "the reference is detached");
    // A uniform distribution's entropy is ln V; KL ≥ 0.
    let u = categorical_entropy(t64(vec![0.3; 2 * c.v], &[1, 1, 2, c.v]), None, TokenReduction::None).to_vec_f64();
    assert_close("uniform entropy", &u, &[(c.v as f64).ln(); 2], 1e-15);
    assert!(kl.iter().all(|k| *k >= 0.0));
}

#[test]
fn kl_and_entropy_match_finite_differences() {
    let c = case(880, -3.0, 3.0);
    let refl = t64(rnd(c.n() * c.v, 881, -3.0, 3.0), &c.lshape());
    let m = c.mask(DType::F64);
    for r in REDUCTIONS {
        fd_input(&format!("KL {r:?}"), &c.logits, &c.lshape(), &|x| categorical_kl(x, &refl, Some(&m), r), 1e-6, 1e-7);
        fd_input(&format!("entropy {r:?}"), &c.logits, &c.lshape(), &|x| categorical_entropy(x, Some(&m), r), 1e-6, 1e-7);
    }
}

/// −∞ logits (excluded tokens, in both distributions at the same entry) give finite KL and
/// entropy equal to the distributions over the remaining tokens, with finite gradients that are
/// zero at the excluded entries; logits of ±1e4 stay finite in f32 and f64; π > 0 where
/// π_ref = 0 gives +∞.
#[test]
fn kl_and_entropy_with_excluded_and_large_logits() {
    let ninf = f64::NEG_INFINITY;
    let p = vec![1.0, ninf, 0.5, -1.0, 2.0, 0.0, ninf, ninf];
    let q = vec![0.0, ninf, 1.5, 0.5, -1.0, 1.0, ninf, ninf];
    let pk = vec![1.0, 0.5, -1.0, 2.0, 0.0];
    let qk = vec![0.0, 1.5, 0.5, -1.0, 1.0];
    let want_kl = [kl_host(&pk[..3], &qk[..3], 3)[0], kl_host(&pk[3..], &qk[3..], 2)[0]];
    let want_h = [entropy_host(&pk[..3], 3)[0], entropy_host(&pk[3..], 2)[0]];
    for dtype in [DType::F32, DType::F64] {
        let tol = if dtype == DType::F64 { 1e-14 } else { 1e-6 };
        let x = Tensor::from_f64s(p.clone(), vec![1, 1, 2, 4], dtype).require_grad();
        let rt = Tensor::from_f64s(q.clone(), vec![1, 1, 2, 4], dtype);
        let kl = categorical_kl(x.clone(), &rt, None, TokenReduction::None);
        assert_close(&format!("{dtype:?} KL with −∞"), &kl.to_vec_f64(), &want_kl, tol);
        let g = x.grad(&kl.sum().backward()).unwrap().to_vec_f64();
        assert!(g.iter().all(|v| v.is_finite()) && g[1] == 0.0 && g[6] == 0.0 && g[7] == 0.0, "{dtype:?} KL gradient {g:?}");
        let h = categorical_entropy(x.clone(), None, TokenReduction::None);
        assert_close(&format!("{dtype:?} entropy with −∞"), &h.to_vec_f64(), &want_h, tol);
        let g = x.grad(&h.sum().backward()).unwrap().to_vec_f64();
        assert!(g.iter().all(|v| v.is_finite()) && g[1] == 0.0 && g[6] == 0.0 && g[7] == 0.0, "{dtype:?} entropy gradient {g:?}");
        // ±1e4.
        let big = Tensor::from_f64s(vec![1e4, -1e4, 0.0, 1e4], vec![1, 1, 1, 4], dtype).require_grad();
        let bref = Tensor::from_f64s(vec![-1e4, 1e4, 0.0, 0.0], vec![1, 1, 1, 4], dtype);
        let l = categorical_kl(big.clone(), &bref, None, TokenReduction::Sum) + categorical_entropy(big.clone(), None, TokenReduction::Sum);
        let lv = l.to_vec_f64()[0];
        // π = (1/2, 0, 0, 1/2): KL = ½(1e4 − (−2e4 − ln(…)))… computed on the host.
        let want = kl_host(&[1e4, -1e4, 0.0, 1e4], &[-1e4, 1e4, 0.0, 0.0], 4)[0] + entropy_host(&[1e4, -1e4, 0.0, 1e4], 4)[0];
        assert!(lv.is_finite() && (lv - want).abs() <= 1e-6 * (1.0 + want.abs()), "{dtype:?} large logits: {lv} vs {want}");
        assert!(big.grad(&l.backward()).unwrap().to_vec_f64().iter().all(|v| v.is_finite()));
        // π_v > 0 where π_ref,v = 0.
        let inf = categorical_kl(Tensor::from_f64s(vec![0.0, 0.0], vec![1, 1, 1, 2], dtype), &Tensor::from_f64s(vec![0.0, ninf], vec![1, 1, 1, 2], dtype), None, TokenReduction::Sum);
        assert_eq!(inf.to_vec_f64()[0], f64::INFINITY);
    }
}

// ------------------------------------------------------------------------- KL estimators

fn k_host(lp: f64, lr: f64, e: KlEstimator) -> f64 {
    let d = lr - lp;
    match e {
        KlEstimator::K1 => -d,
        KlEstimator::K2 => d * d / 2.0,
        KlEstimator::K3 => d.exp() - d - 1.0,
    }
}

const ESTIMATORS: [KlEstimator; 3] = [KlEstimator::K1, KlEstimator::K2, KlEstimator::K3];

#[test]
fn kl_estimators_match_reference_and_finite_differences() {
    let c = case(890, -3.0, 3.0);
    let lp = c.logp_host();
    let lr: Vec<f64> = lp.iter().zip(rnd(c.n(), 891, -1.5, 1.5)).map(|(a, d)| a + d).collect();
    let tok = c.tokens();
    for e in ESTIMATORS {
        let host: Vec<f64> = (0..c.n()).map(|i| if c.mask[i] == 1.0 { k_host(lp[i], lr[i], e) } else { 0.0 }).collect();
        if e != KlEstimator::K1 {
            assert!(host.iter().all(|v| *v >= 0.0));
        }
        for r in REDUCTIONS {
            let want = reduce_host(&host, &c.mask, c.sh, r);
            for (dtype, tol) in [(DType::F64, 1e-13), (DType::F32, 2e-6)] {
                let x = Tensor::from_f64s(c.logits.clone(), c.lshape(), dtype);
                let got = kl_estimate(token_log_probs(x, &tok), &Tensor::from_f64s(lr.clone(), c.sh.to_vec(), dtype), e, Some(&c.mask(dtype)), r).to_vec_f64();
                assert_close(&format!("{e:?} {r:?} {dtype:?}"), &got, &want, tol);
            }
            let (lrt, m) = (t64(lr.clone(), &c.sh), c.mask(DType::F64));
            fd_input(&format!("{e:?} {r:?}"), &c.logits, &c.lshape(), &|x| kl_estimate(token_log_probs(x, &tok), &lrt, e, Some(&m), r), 1e-6, 1e-7);
        }
    }
}

/// The documented expectations, computed exactly by enumerating every token of one
/// distribution (each token one position, weighted by its probability): the values of k1 and k3
/// are KL(π ‖ π_ref); the expected per-sample gradient is zero for k1, the gradient of
/// KL(π ‖ π_ref) (`categorical_kl`) for k2 and of the forward KL(π_ref ‖ π) (`kl_div`) for k3.
#[test]
fn kl_estimator_expectations_by_enumeration() {
    let v = 6;
    let pl = rnd(v, 900, -2.0, 2.0);
    let rl = rnd(v, 901, -2.0, 2.0);
    let pi: Vec<f64> = log_softmax_host(&pl, v).iter().map(|l| l.exp()).collect();
    let lr = log_softmax_host(&rl, v);
    let kl = kl_host(&pl, &rl, v)[0];
    let tokens = IntTensor::from_data((0..v as i64).collect(), [1, 1, v]);
    let weights = t64(pi.clone(), &[1, 1, v]);
    let expect = |e: KlEstimator| -> (f64, Vec<f64>) {
        let x = t64(pl.clone(), &[1, 1, 1, v]).require_grad();
        let rows = Tensor::cat(vec![x.clone(); v], 2); // [1, 1, V, V]: position t holds the same logits
        let k = kl_estimate(token_log_probs(rows, &tokens), &t64(lr.clone(), &[1, 1, v]), e, None, TokenReduction::None);
        let ex = (k * weights.clone()).sum();
        let val = ex.to_vec_f64()[0];
        (val, x.grad(&ex.backward()).unwrap().to_vec_f64())
    };
    let grad_of = |f: &dyn Fn(Tensor) -> Tensor| {
        let x = t64(pl.clone(), &[1, 1, 1, v]).require_grad();
        x.grad(&f(x.clone()).sum().backward()).unwrap().to_vec_f64()
    };
    let reverse = grad_of(&|x| categorical_kl(x, &t64(rl.clone(), &[1, 1, 1, v]), None, TokenReduction::Sum));
    let forward = grad_of(&|x| kl_div(x, t64(rl.clone(), &[1, 1, 1, v])));
    let (v1, dk1) = expect(KlEstimator::K1);
    let (_, dk2) = expect(KlEstimator::K2);
    let (v3, dk3) = expect(KlEstimator::K3);
    assert_close("E[k1] = KL", &[v1], &[kl], 1e-14);
    assert_close("E[k3] = KL", &[v3], &[kl], 1e-14);
    assert_close("E[∇k1] = 0", &dk1, &vec![0.0; v], 1e-14);
    assert_close("E[∇k2] = ∇KL(π ‖ ref)", &dk2, &reverse, 1e-14);
    assert_close("E[∇k3] = ∇KL(ref ‖ π)", &dk3, &forward, 1e-14);
    assert!(reverse.iter().zip(&forward).any(|(a, b)| (a - b).abs() > 1e-3), "the two gradients differ on this case");
}

// ------------------------------------------------------------------------- masking edges

/// Every position masked: each loss and reduction is 0 with exactly zero gradient; one scored
/// token: the token mean and the sequence mean are that token's loss.
#[test]
fn masking_edges_all_masked_and_one_token() {
    let c = case(910, -3.0, 3.0);
    let tok = c.tokens();
    let (old, adv) = (t64(c.logp_host().iter().map(|l| l + 0.3).collect(), &c.sh), t64(rnd(c.n(), 911, -1.0, 1.0), &c.sh));
    let refl = t64(rnd(c.n() * c.v, 912, -3.0, 3.0), &c.lshape());
    let lr = t64(rnd(c.n(), 913, -4.0, -1.0), &c.sh);
    let losses: Vec<(&str, Box<dyn Fn(Tensor, &Tensor, TokenReduction) -> Tensor>)> = vec![
        ("PG", Box::new(|x, m, r| policy_gradient_loss(token_log_probs(x, &tok), &adv, Some(m), r))),
        ("PPO", Box::new(|x, m, r| clipped_policy_loss(token_log_probs(x, &tok), &old, &adv, Some(m), Clip::symmetric(0.2), r))),
        ("KL", Box::new(|x, m, r| categorical_kl(x, &refl, Some(m), r))),
        ("k3", Box::new(|x, m, r| kl_estimate(token_log_probs(x, &tok), &lr, KlEstimator::K3, Some(m), r))),
        ("entropy", Box::new(|x, m, r| categorical_entropy(x, Some(m), r))),
    ];
    let none = t64(vec![0.0; c.n()], &c.sh);
    let mut one = vec![0.0; c.n()];
    one[c.n() - 2] = 1.0;
    let one = t64(one, &c.sh);
    for (name, f) in &losses {
        for r in REDUCTIONS {
            let x = t64(c.logits.clone(), &c.lshape()).require_grad();
            let l = f(x.clone(), &none, r);
            assert!(l.to_vec_f64().iter().all(|v| *v == 0.0), "{name} {r:?} all masked: {:?}", l.to_vec_f64());
            assert!(x.grad(&l.sum().backward()).unwrap().to_vec_f64().iter().all(|g| *g == 0.0), "{name} {r:?} all masked: gradient");
        }
        let per = f(t64(c.logits.clone(), &c.lshape()), &one, TokenReduction::None).to_vec_f64();
        let only = per[c.n() - 2];
        assert!(per.iter().enumerate().all(|(i, v)| i == c.n() - 2 || *v == 0.0));
        for r in [TokenReduction::TokenMean, TokenReduction::SequenceMean, TokenReduction::Sum] {
            let got = f(t64(c.logits.clone(), &c.lshape()), &one, r).to_vec_f64();
            assert_eq!(got, vec![0.0, only], "{name} {r:?} one token");
        }
        // Masked positions get exactly zero gradient under the case's mask.
        let x = t64(c.logits.clone(), &c.lshape()).require_grad();
        let g = x.grad(&f(x.clone(), &c.mask(DType::F64), TokenReduction::TokenMean).sum().backward()).unwrap().to_vec_f64();
        for i in (0..c.n()).filter(|&i| c.mask[i] == 0.0) {
            assert!(g[i * c.v..(i + 1) * c.v].iter().all(|v| *v == 0.0), "{name}: masked position {i} has a gradient");
        }
    }
}

// ------------------------------------------------------------------------ group advantages

/// Hand-computed: groups `[1 0 0 1]`, `[2 2 2 2]`, `[0 0 0 3]`.
#[test]
fn group_advantages_by_hand() {
    let r = [1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 3.0];
    let s3 = 0.5 / (1.0f64 / 3.0).sqrt(); // ±0.5 over the sample std √(1/3)
    let unbiased = group_advantages(&r, 4, GroupNorm::Std { eps: 0.0, unbiased: true });
    assert_close("unbiased", &unbiased, &[s3, -s3, -s3, s3, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, 1.5], 1e-15);
    let pop = group_advantages(&r, 4, GroupNorm::Std { eps: 0.0, unbiased: false });
    let p3 = 1.6875_f64.sqrt();
    assert_close("population", &pop, &[1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.75 / p3, -0.75 / p3, -0.75 / p3, 2.25 / p3], 1e-15);
    let mean = group_advantages(&r, 4, GroupNorm::MeanOnly);
    assert_eq!(mean, vec![0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, -0.75, -0.75, -0.75, 2.25]);
    let eps = group_advantages(&r, 4, GroupNorm::Std { eps: 1e-2, unbiased: true });
    assert_close("ε", &eps[8..], &[-0.75 / 1.51, -0.75 / 1.51, -0.75 / 1.51, 2.25 / 1.51], 1e-15);
    // The tensor form groups along the last axis and keeps shape and dtype.
    let t = group_advantages_tensor(&t32(r.to_vec(), &[3, 4]), GroupNorm::Std { eps: 0.0, unbiased: true });
    assert_eq!((t.shape(), t.dtype()), (&[3usize, 4][..], DType::F32));
    assert_close("tensor form", &t.to_vec_f64(), &unbiased, 1e-7);
    // Groups of one: population std gives 0, mean-only gives 0.
    assert_eq!(group_advantages(&[3.0, -1.0], 1, GroupNorm::Std { eps: 1e-6, unbiased: false }), vec![0.0, 0.0]);
}

#[test]
#[should_panic(expected = "at least 2")]
fn group_advantages_refuse_unbiased_groups_of_one() {
    let _ = group_advantages(&[1.0, 2.0], 1, GroupNorm::Std { eps: 1e-6, unbiased: true });
}

// --------------------------------------------------------------------------------- sampler

/// Frequencies over 40,000 draws match softmax(l/τ + mask) within 4.5 standard errors; an
/// excluded token is never drawn; the stream advances by one u64 per logit; the same stream
/// gives the same tokens; temperature 0 is the argmax and draws nothing.
#[test]
fn sampler_matches_its_distribution() {
    use rand::RngCore;
    let v = 6;
    let n = 40_000;
    let row = [0.4, -1.0, 1.2, 3.0, 0.0, -0.3];
    let maskrow = [0.0, 0.0, 0.0, f64::NEG_INFINITY, 0.5, 0.0];
    for (dtype, tau) in [(DType::F32, 0.7), (DType::F64, 1.0), (DType::F32, 2.5)] {
        let logits = Tensor::from_f64s(row.repeat(n), vec![n, v], dtype);
        let mask = Tensor::from_f64s(maskrow.to_vec(), vec![1, v], dtype);
        let mut rng = crate::rng::derive(5, "sampler test");
        let mut copy = rng.clone();
        let tok = sample_categorical(&logits, tau, Some(&mask), &mut rng);
        assert_eq!(tok.shape(), &[n]);
        for _ in 0..n * v {
            copy.next_u64();
        }
        assert_eq!(rng.next_u64(), copy.next_u64(), "one u64 per logit");
        let z: Vec<f64> = row.iter().zip(&maskrow).map(|(l, m)| l / tau + m).collect();
        let p: Vec<f64> = log_softmax_host(&z, v).iter().map(|l| l.exp()).collect();
        let mut count = vec![0usize; v];
        for t in tok.to_vec() {
            count[t as usize] += 1;
        }
        assert_eq!(count[3], 0, "excluded token drawn");
        for k in 0..v {
            let f = count[k] as f64 / n as f64;
            let se = (p[k] * (1.0 - p[k]) / n as f64).sqrt();
            assert!((f - p[k]).abs() <= 4.5 * se + 1e-12, "{dtype:?} τ {tau}: token {k} frequency {f} vs {}", p[k]);
        }
        let again = sample_categorical(&logits, tau, Some(&mask), &mut crate::rng::derive(5, "sampler test"));
        assert_eq!(again.to_vec(), tok.to_vec(), "same stream, same tokens");
        let other = sample_categorical(&logits, tau, Some(&mask), &mut crate::rng::derive(6, "sampler test"));
        assert_ne!(other.to_vec(), tok.to_vec());
    }
    // Temperature 0: the argmax of the masked logits (token 4: 0 + 2 beats 1.2; token 3 excluded),
    // and no draws.
    let logits = t32([row.to_vec(), row.to_vec()].concat(), &[1, 2, v]);
    let mask = t32(vec![0.0, 0.0, 0.0, f64::NEG_INFINITY, 2.0, 0.0], &[1, 1, v]);
    let mut rng = crate::rng::derive(1, "greedy");
    let mut copy = rng.clone();
    let tok = sample_categorical(&logits, 0.0, Some(&mask), &mut rng);
    assert_eq!((tok.shape(), tok.to_vec()), (&[1usize, 2][..], vec![4, 4]));
    assert_eq!(rng.next_u64(), copy.next_u64(), "greedy draws nothing");
    // The log-probabilities of samples under the sampling distribution are finite.
    let lp = token_log_probs(tempered_logits(t32(row.repeat(4), &[4, v]), 0.7, Some(&t32(maskrow.to_vec(), &[1, v]))), &sample_categorical(&t32(row.repeat(4), &[4, v]), 0.7, Some(&t32(maskrow.to_vec(), &[1, v])), &mut crate::rng::derive(2, "lp")));
    assert!(lp.to_vec_f64().iter().all(|l| l.is_finite() && *l <= 0.0));
}

#[test]
#[should_panic(expected = "no allowed token")]
fn sampler_refuses_a_position_with_no_allowed_token() {
    let logits = t32(vec![0.0; 6], &[2, 3]);
    let mask = t32(vec![0.0, 0.0, 0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], &[2, 3]);
    let _ = sample_categorical(&logits, 1.0, Some(&mask), &mut crate::rng::derive(0, "x"));
}
