//! A collapse monitor for embeddings, on the host in f64 and outside autograd: the covariance of a
//! batch of embeddings, its eigenvalues by the cyclic Jacobi method, and the effective rank of Roy
//! and Vetterli (2007), `exp(H(p))` with `p_i = λ_i / Σλ` over the non-negative eigenvalues.
//! The identity in d dimensions has effective rank d; k equal eigenvalues and the rest zero give
//! k. Everything here is a fixed sequence of f64 operations, so it is deterministic.

use crate::tensor::Tensor;

/// The eigenvalues of the symmetric `d × d` matrix `a` (row-major), descending, by cyclic Jacobi
/// rotations until the off-diagonal mass is below 1e-30 of the total (or 100 sweeps).
pub fn jacobi_eigenvalues(a: &[f64], d: usize) -> Vec<f64> {
    assert_eq!(a.len(), d * d, "Jacobi: {} values for a {d}×{d} matrix", a.len());
    let mut m = a.to_vec();
    // Symmetrise exactly (an input symmetric up to rounding).
    for p in 0..d {
        for q in p + 1..d {
            let v = 0.5 * (m[p * d + q] + m[q * d + p]);
            m[p * d + q] = v;
            m[q * d + p] = v;
        }
    }
    let total: f64 = m.iter().map(|x| x * x).sum();
    for _ in 0..100 {
        let off: f64 = (0..d).flat_map(|p| (p + 1..d).map(move |q| (p, q))).map(|(p, q)| m[p * d + q] * m[p * d + q]).sum();
        if off <= 1e-30 * total || off == 0.0 {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = m[p * d + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (m[q * d + q] - m[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..d {
                    if r != p && r != q {
                        let (arp, arq) = (m[r * d + p], m[r * d + q]);
                        let (np, nq) = (c * arp - s * arq, s * arp + c * arq);
                        m[r * d + p] = np;
                        m[p * d + r] = np;
                        m[r * d + q] = nq;
                        m[q * d + r] = nq;
                    }
                }
                m[p * d + p] -= t * apq;
                m[q * d + q] += t * apq;
                m[p * d + q] = 0.0;
                m[q * d + p] = 0.0;
            }
        }
    }
    let mut ev: Vec<f64> = (0..d).map(|i| m[i * d + i]).collect();
    ev.sort_by(|x, y| y.total_cmp(x));
    ev
}

/// The Roy–Vetterli effective rank of eigenvalues `ev`: negative ones (rounding) count as 0;
/// all zero gives 0.
pub fn effective_rank_of(ev: &[f64]) -> f64 {
    let sum: f64 = ev.iter().map(|x| x.max(0.0)).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let h: f64 = ev.iter().map(|x| x.max(0.0) / sum).filter(|p| *p > 0.0).map(|p| -p * p.ln()).sum();
    h.exp()
}

/// The effective rank of the symmetric `d × d` matrix `cov` (row-major).
pub fn effective_rank_of_covariance(cov: &[f64], d: usize) -> f64 {
    effective_rank_of(&jacobi_eigenvalues(cov, d))
}

/// The unbiased covariance `d × d` of `n` rows `z` (`[n, d]` row-major), n ≥ 2.
pub fn covariance(z: &[f64], n: usize, d: usize) -> Vec<f64> {
    assert!(n >= 2 && z.len() == n * d, "covariance: {} values for {n} rows of {d}", z.len());
    let mean: Vec<f64> = (0..d).map(|j| (0..n).map(|i| z[i * d + j]).sum::<f64>() / n as f64).collect();
    let mut c = vec![0.0; d * d];
    for p in 0..d {
        for q in p..d {
            let v = (0..n).map(|i| (z[i * d + p] - mean[p]) * (z[i * d + q] - mean[q])).sum::<f64>() / (n - 1) as f64;
            c[p * d + q] = v;
            c[q * d + p] = v;
        }
    }
    c
}

/// What the monitor reports for one seed's embeddings.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingStats {
    pub effective_rank: f64,
    /// The covariance's eigenvalues, descending.
    pub eigenvalues: Vec<f64>,
    /// Per dimension, the standard deviation over the rows.
    pub std: Vec<f64>,
}

/// Per seed, the monitor's statistics of embeddings `z` `[S, N, D]` (any device and float dtype;
/// read to the host as f64, outside autograd).
pub fn embedding_stats(z: &Tensor) -> Vec<EmbeddingStats> {
    let [s, n, d] = z.dims();
    let v = z.clone().untracked().to_vec_f64();
    (0..s)
        .map(|k| {
            let c = covariance(&v[k * n * d..(k + 1) * n * d], n, d);
            let eigenvalues = jacobi_eigenvalues(&c, d);
            EmbeddingStats { effective_rank: effective_rank_of(&eigenvalues), std: (0..d).map(|j| c[j * d + j].sqrt()).collect(), eigenvalues }
        })
        .collect()
}

/// Per seed, the effective rank of the covariance of `z` `[S, N, D]`.
pub fn effective_rank(z: &Tensor) -> Vec<f64> {
    embedding_stats(z).into_iter().map(|x| x.effective_rank).collect()
}
