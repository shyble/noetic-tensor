//! Shapes: element counts, row-major strides and burn's broadcasting rule (same rank; a
//! dimension broadcasts only when it is 1).

pub(crate) fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major (C order) strides.
pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// The broadcast shape of two same-rank shapes.
pub(crate) fn broadcast(a: &[usize], b: &[usize]) -> Vec<usize> {
    assert_eq!(a.len(), b.len(), "broadcasting needs equal ranks: {a:?} vs {b:?}");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            assert!(x == y || x == 1 || y == 1, "shapes {a:?} and {b:?} do not broadcast");
            x.max(y)
        })
        .collect()
}

/// Strides of `shape` read as if broadcast to `out` (0 on broadcast dimensions).
/// Panics unless `shape` broadcasts to `out` (equal ranks; every dimension equal or 1).
pub(crate) fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    assert_eq!(shape.len(), out.len(), "cannot broadcast rank {} {shape:?} to rank {} {out:?}", shape.len(), out.len());
    for (&d, &o) in shape.iter().zip(out) {
        assert!(d == o || d == 1, "cannot broadcast {shape:?} to {out:?}");
    }
    let s = strides(shape);
    shape.iter().zip(out).zip(s).map(|((&d, &o), st)| if d == o { st } else { 0 }).collect()
}
