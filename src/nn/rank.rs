//! A collapse monitor for embeddings, on the host in f64 and outside autograd: the covariance of a
//! batch of embeddings, its eigenvalues by the cyclic Jacobi method, and two effective ranks.
//!
//! - `effective_rank_of` / `effective_rank` (the covariance spectrum): Roy and Vetterli's (2007)
//!   entropy rank `exp(H(p))` over the **covariance eigenvalues**, `p_i = λ_i / Σλ`.
//! - `effective_rank_of_singular` / `effective_rank_singular` (RankMe, Garrido et al. 2023): the
//!   same entropy rank over the **singular values** of the embedding matrix, `p_i = σ_i / Σσ` with
//!   `σ_i = √λ_i(ZᵀZ)` (uncentred, as RankMe; RankMe's ε inside the log is left out, zero terms
//!   are skipped instead).
//!
//! The two agree on flat spectra (the identity in d dimensions gives d, k equal values and the
//! rest zero give k) and differ otherwise: √ flattens the spectrum, so the singular-value rank is
//! the larger. Which one a gate uses is the caller's choice. Everything here is a fixed
//! sequence of f64 operations, so it is deterministic.

use crate::tensor::Tensor;

/// The outcome of `jacobi`.
#[derive(Clone, Debug, PartialEq)]
pub struct Jacobi {
    /// Descending.
    pub eigenvalues: Vec<f64>,
    /// Sweeps run (a sweep rotates every off-diagonal pair once).
    pub sweeps: usize,
    /// Whether the off-diagonal mass fell below 1e-30 of the total within `MAX_SWEEPS`.
    pub converged: bool,
}

/// The sweep limit of `jacobi`.
pub const MAX_SWEEPS: usize = 100;

/// The eigenvalues of the symmetric `d × d` matrix `a` (row-major), descending; panics if Jacobi
/// did not converge (see `jacobi`).
pub fn jacobi_eigenvalues(a: &[f64], d: usize) -> Vec<f64> {
    let j = jacobi(a, d);
    assert!(j.converged, "Jacobi did not converge in {} sweeps (d {d})", j.sweeps);
    j.eigenvalues
}

/// Cyclic Jacobi rotations on the symmetric `d × d` matrix `a` (row-major) until the
/// off-diagonal mass is below 1e-30 of the total, at most `MAX_SWEEPS` sweeps; the eigenvalues,
/// the sweep count and whether it converged.
pub fn jacobi(a: &[f64], d: usize) -> Jacobi {
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
    let off_mass = |m: &[f64]| -> f64 { (0..d).flat_map(|p| (p + 1..d).map(move |q| (p, q))).map(|(p, q)| m[p * d + q] * m[p * d + q]).sum() };
    let mut sweeps = 0;
    let mut converged = false;
    loop {
        let off = off_mass(&m);
        if off <= 1e-30 * total || off == 0.0 {
            converged = true;
            break;
        }
        if sweeps == MAX_SWEEPS {
            break;
        }
        sweeps += 1;
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
    Jacobi { eigenvalues: ev, sweeps, converged }
}

/// The Roy–Vetterli effective rank of covariance eigenvalues `ev`: negative ones (rounding)
/// count as 0; all zero gives 0. (The entropy rank of any non-negative spectrum: given singular
/// values it is RankMe, `effective_rank_of_singular`.)
pub fn effective_rank_of(ev: &[f64]) -> f64 {
    let sum: f64 = ev.iter().map(|x| x.max(0.0)).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let h: f64 = ev.iter().map(|x| x.max(0.0) / sum).filter(|p| *p > 0.0).map(|p| -p * p.ln()).sum();
    h.exp()
}

/// The Roy–Vetterli effective rank of the symmetric `d × d` matrix `cov` (row-major), over its
/// eigenvalues.
pub fn effective_rank_of_covariance(cov: &[f64], d: usize) -> f64 {
    effective_rank_of(&jacobi_eigenvalues(cov, d))
}

/// RankMe's effective rank of singular values `sv` (non-negative; the entropy rank with
/// `p_i = σ_i / Σσ`).
pub fn effective_rank_of_singular(sv: &[f64]) -> f64 {
    effective_rank_of(sv)
}

/// The singular values of the `n × d` matrix `z` (row-major), descending: `√λ` of `zᵀz`'s
/// eigenvalues (negative rounding clipped to 0). Not centred, as RankMe.
pub fn singular_values(z: &[f64], n: usize, d: usize) -> Vec<f64> {
    assert_eq!(z.len(), n * d, "singular values: {} values for {n} rows of {d}", z.len());
    let mut g = vec![0.0; d * d];
    for p in 0..d {
        for q in p..d {
            let v: f64 = (0..n).map(|i| z[i * d + p] * z[i * d + q]).sum();
            g[p * d + q] = v;
            g[q * d + p] = v;
        }
    }
    jacobi_eigenvalues(&g, d).into_iter().map(|x| x.max(0.0).sqrt()).collect()
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
    /// Roy–Vetterli over the covariance eigenvalues.
    pub effective_rank: f64,
    /// RankMe: over the singular values of the (uncentred) embedding matrix.
    pub effective_rank_singular: f64,
    /// The covariance's eigenvalues, descending.
    pub eigenvalues: Vec<f64>,
    /// The embedding matrix's singular values, descending.
    pub singular_values: Vec<f64>,
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
            let rows = &v[k * n * d..(k + 1) * n * d];
            let c = covariance(rows, n, d);
            let eigenvalues = jacobi_eigenvalues(&c, d);
            let singular_values = singular_values(rows, n, d);
            EmbeddingStats {
                effective_rank: effective_rank_of(&eigenvalues),
                effective_rank_singular: effective_rank_of_singular(&singular_values),
                std: (0..d).map(|j| c[j * d + j].sqrt()).collect(),
                eigenvalues,
                singular_values,
            }
        })
        .collect()
}

/// Per seed, the Roy–Vetterli effective rank of the covariance eigenvalues of `z` `[S, N, D]`.
pub fn effective_rank(z: &Tensor) -> Vec<f64> {
    embedding_stats(z).into_iter().map(|x| x.effective_rank).collect()
}

/// Per seed, RankMe's effective rank of the singular values of `z` `[S, N, D]` (uncentred).
pub fn effective_rank_singular(z: &Tensor) -> Vec<f64> {
    embedding_stats(z).into_iter().map(|x| x.effective_rank_singular).collect()
}
