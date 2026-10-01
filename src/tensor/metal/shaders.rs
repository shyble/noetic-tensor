//! The Metal kernels, compiled from source at first use with fast math off
//! (`MTLMathModeSafe`, precise math functions). Every kernel is deterministic: one thread per
//! output element (or per output tile), reductions in a fixed order, no atomics.

pub(crate) const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- vector add

kernel void add_f32(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* c [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------- matmul
// C[z] = A[z] · B[z] for M×K and K×N blocks read through element strides (views need no copy),
// with per-batch element offsets (broadcast batches share a block). A 64×64 output tile per
// threadgroup of 16×16 threads, each thread a 4×4 register tile. The K loop runs in order and
// every product enters its accumulator by one fused multiply-add, in blocks of KC = 256 that are
// then added to the running total (block sum + total): the order of matrixmultiply's sgemm
// (kc = 256, the NEON kernel's fma chain from zero, then ab + c), so on aarch64 the result is
// CpuRef's to the bit, and it is fixed in any case.

struct MatmulParams {
    uint m, n, k;
    uint a_rs, a_cs, b_rs, b_cs;
};

#define BM 64
#define BN 64
#define BK 16
#define KC 256

kernel void matmul_f32(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device float* C [[buffer(2)]],
                       device const uint2* offs [[buffer(3)]],
                       constant MatmulParams& p [[buffer(4)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint3 tid [[thread_position_in_threadgroup]]) {
    threadgroup float As[BK][BM];
    threadgroup float Bs[BK][BN];
    const uint tx = tid.x, ty = tid.y;
    const uint lid = ty * 16 + tx;
    const uint row0 = tg.y * BM, col0 = tg.x * BN;
    const uint2 o = offs[tg.z];
    device const float* a = A + o.x;
    device const float* b = B + o.y;
    device float* c = C + (ulong)tg.z * p.m * p.n;
    float acc[4][4], tot[4][4];
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 4; ++j) { acc[i][j] = 0.0f; tot[i][j] = 0.0f; }
    for (uint k0 = 0; k0 < p.k; k0 += BK) {
        for (uint e = lid; e < BM * BK; e += 256) {
            uint r = e / BK, kk = e % BK;
            uint gr = row0 + r, gk = k0 + kk;
            As[kk][r] = (gr < p.m && gk < p.k) ? a[(ulong)gr * p.a_rs + (ulong)gk * p.a_cs] : 0.0f;
        }
        for (uint e = lid; e < BK * BN; e += 256) {
            uint kk = e / BN, cc = e % BN;
            uint gk = k0 + kk, gc = col0 + cc;
            Bs[kk][cc] = (gk < p.k && gc < p.n) ? b[(ulong)gk * p.b_rs + (ulong)gc * p.b_cs] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint kmax = min((uint)BK, p.k - k0);
        for (uint kk = 0; kk < kmax; ++kk) {
            float av[4], bv[4];
            for (uint i = 0; i < 4; ++i) av[i] = As[kk][ty + 16 * i];
            for (uint j = 0; j < 4; ++j) bv[j] = Bs[kk][tx + 16 * j];
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((k0 + BK) % KC == 0 || k0 + BK >= p.k) {
            // Close a KC block: the first is the total, later ones are added to it.
            const bool first = k0 < KC;
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) {
                    tot[i][j] = first ? acc[i][j] : acc[i][j] + tot[i][j];
                    acc[i][j] = 0.0f;
                }
        }
    }
    for (uint i = 0; i < 4; ++i) {
        uint r = row0 + ty + 16 * i;
        if (r >= p.m) continue;
        for (uint j = 0; j < 4; ++j) {
            uint cc = col0 + tx + 16 * j;
            if (cc < p.n) c[(ulong)r * p.n + cc] = tot[i][j];
        }
    }
}

// ---------------------------------------------------------------- tiled matmul, tuned
// The same arithmetic per output element as matmul_f32 (one fma chain over k in order, closed in
// KC = 256 blocks added to the running total), so the result is identical to the bit; only the
// work split changes: BM×BN output tiles, TM×TN outputs per thread (consecutive rows and
// columns, read from threadgroup memory as vectors), tile loads coalesced along whichever axis
// of A and B is contiguous (transposed views in backward), and padded threadgroup rows against
// bank conflicts.

template <uint TBM, uint TBN, uint TTM, uint TTN>
inline void matmul_tiled(device const float* A, device const float* B, device float* C,
                         device const uint2* offs, constant MatmulParams& p,
                         threadgroup float (*As)[TBM + 4], threadgroup float (*Bs)[TBN + 4],
                         uint3 tg, uint3 tid) {
    const uint NX = TBN / TTN, NT = (TBM / TTM) * NX;
    const uint tx = tid.x, ty = tid.y, lid = ty * NX + tx;
    const uint row0 = tg.y * TBM, col0 = tg.x * TBN;
    const uint2 o = offs[tg.z];
    device const float* a = A + o.x;
    device const float* b = B + o.y;
    device float* c = C + (ulong)tg.z * p.m * p.n;
    const bool a_col = p.a_rs == 1 && p.a_cs != 1;
    const bool b_col = p.b_cs != 1 && p.b_rs == 1;
    float acc[TTM][TTN], tot[TTM][TTN];
    for (uint i = 0; i < TTM; ++i) for (uint j = 0; j < TTN; ++j) { acc[i][j] = 0.0f; tot[i][j] = 0.0f; }
    for (uint k0 = 0; k0 < p.k; k0 += BK) {
        for (uint e = lid; e < TBM * BK; e += NT) {
            uint r, kk;
            if (a_col) { kk = e / TBM; r = e % TBM; } else { r = e / BK; kk = e % BK; }
            uint gr = row0 + r, gk = k0 + kk;
            As[kk][r] = (gr < p.m && gk < p.k) ? a[(ulong)gr * p.a_rs + (ulong)gk * p.a_cs] : 0.0f;
        }
        for (uint e = lid; e < BK * TBN; e += NT) {
            uint kk, cc;
            if (b_col) { cc = e / BK; kk = e % BK; } else { kk = e / TBN; cc = e % TBN; }
            uint gk = k0 + kk, gc = col0 + cc;
            Bs[kk][cc] = (gk < p.k && gc < p.n) ? b[(ulong)gk * p.b_rs + (ulong)gc * p.b_cs] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint kmax = min((uint)BK, p.k - k0);
        for (uint kk = 0; kk < kmax; ++kk) {
            float av[TTM], bv[TTN];
            if (TTM % 4 == 0) {
                for (uint i = 0; i < TTM; i += 4) {
                    float4 v = *(threadgroup const float4*)&As[kk][ty * TTM + i];
                    av[i] = v.x; av[i + 1] = v.y; av[i + 2] = v.z; av[i + 3] = v.w;
                }
            } else {
                for (uint i = 0; i < TTM; ++i) av[i] = As[kk][ty * TTM + i];
            }
            if (TTN % 4 == 0) {
                for (uint j = 0; j < TTN; j += 4) {
                    float4 v = *(threadgroup const float4*)&Bs[kk][tx * TTN + j];
                    bv[j] = v.x; bv[j + 1] = v.y; bv[j + 2] = v.z; bv[j + 3] = v.w;
                }
            } else {
                for (uint j = 0; j < TTN; ++j) bv[j] = Bs[kk][tx * TTN + j];
            }
            for (uint i = 0; i < TTM; ++i)
                for (uint j = 0; j < TTN; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((k0 + BK) % KC == 0 || k0 + BK >= p.k) {
            const bool first = k0 < KC;
            for (uint i = 0; i < TTM; ++i)
                for (uint j = 0; j < TTN; ++j) {
                    tot[i][j] = first ? acc[i][j] : acc[i][j] + tot[i][j];
                    acc[i][j] = 0.0f;
                }
        }
    }
    for (uint i = 0; i < TTM; ++i) {
        uint r = row0 + ty * TTM + i;
        if (r >= p.m) continue;
        for (uint j = 0; j < TTN; ++j) {
            uint cc = col0 + tx * TTN + j;
            if (cc < p.n) c[(ulong)r * p.n + cc] = tot[i][j];
        }
    }
}

#define MATMUL_VARIANT(NAME, BM_, BN_, TM_, TN_) \
kernel void NAME(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], \
                 device const uint2* offs [[buffer(3)]], constant MatmulParams& p [[buffer(4)]], \
                 uint3 tg [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) { \
    threadgroup float As[BK][BM_ + 4]; \
    threadgroup float Bs[BK][BN_ + 4]; \
    matmul_tiled<BM_, BN_, TM_, TN_>(A, B, C, offs, p, As, Bs, tg, tid); \
}
MATMUL_VARIANT(matmul_64x64_4x4, 64, 64, 4, 4)
MATMUL_VARIANT(matmul_128x64_8x4, 128, 64, 8, 4)
MATMUL_VARIANT(matmul_64x128_4x8, 64, 128, 4, 8)
MATMUL_VARIANT(matmul_128x128_8x8, 128, 128, 8, 8)
MATMUL_VARIANT(matmul_32x64_4x4, 32, 64, 4, 4)
MATMUL_VARIANT(matmul_64x64_8x4, 64, 64, 8, 4)
MATMUL_VARIANT(matmul_32x32_4x4, 32, 32, 4, 4)
MATMUL_VARIANT(matmul_32x32_2x2, 32, 32, 2, 2)
MATMUL_VARIANT(matmul_16x16_1x1, 16, 16, 1, 1)
MATMUL_VARIANT(matmul_32x16_2x1, 32, 16, 2, 1)

// ---------------------------------------------------------------- strided elementwise
// A view is read in place: element i of the output (row-major over `shape`) reads operand a at
// off_a + Σ coord·sa (0 strides broadcast), b likewise, and writes c at off_c + Σ coord·sc.

#define MAXR 8

struct Strided {
    uint n, rank, off_a, off_b, off_c;
    uint shape[MAXR];
    uint sa[MAXR];
    uint sb[MAXR];
    uint sc[MAXR];
    // Bit 0/1/2 set when a/b/c is row-major over `shape` (from its offset): element i is then
    // at offset + i, with no per-dimension index arithmetic. The same element is read either way.
    uint contig;
};

inline uint off_of(uint i, constant Strided& p, constant uint* st, uint base) {
    uint o = base;
    for (int d = int(p.rank) - 1; d >= 0; --d) {
        uint s = p.shape[d];
        o += (i % s) * st[d];
        i /= s;
    }
    return o;
}

#define COPY_KERNEL(NAME, T) \
kernel void NAME(device const T* x [[buffer(0)]], device T* y [[buffer(1)]], constant Strided& p [[buffer(2)]], uint i [[thread_position_in_grid]]) { \
    if (i >= p.n) return; \
    y[((p.contig & 4u) ? p.off_c + i : off_of(i, p, p.sc, p.off_c))] = x[((p.contig & 1u) ? p.off_a + i : off_of(i, p, p.sa, p.off_a))]; \
}
COPY_KERNEL(copy_strided_8, uchar)
COPY_KERNEL(copy_strided_16, ushort)
COPY_KERNEL(copy_strided_32, uint)
COPY_KERNEL(copy_strided_64, ulong)

// f16 and bf16 bit patterns, as tensor::half converts them (round to nearest, ties to even).
inline float bf16_to_f32(ushort h) { return as_type<float>(uint(h) << 16); }
inline ushort f32_to_bf16(float x) {
    uint b = as_type<uint>(x);
    if (isnan(x)) return ushort((b >> 16) | 0x0040);
    uint round = 0x7fff + ((b >> 16) & 1);
    return ushort((b + round) >> 16);
}
inline float f16_to_f32(ushort hh) {
    uint h = hh;
    uint sign = (h & 0x8000) << 16;
    uint e = (h >> 10) & 0x1f;
    uint man = h & 0x3ff;
    uint bits;
    if (e == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int ee = 0;
            uint m = man;
            while ((m & 0x400) == 0) { m <<= 1; ee -= 1; }
            bits = sign | (uint(ee + 127 - 14) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (e == 0x1f) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((e + 127 - 15) << 23) | (man << 13);
    }
    return as_type<float>(bits);
}
inline ushort f32_to_f16(float x) {
    uint b = as_type<uint>(x);
    uint sign = (b >> 16) & 0x8000;
    int ex = int((b >> 23) & 0xff);
    uint man = b & 0x7fffff;
    if (ex == 0xff) return ushort(sign | 0x7c00 | (man != 0 ? (0x0200 | (man >> 13)) : 0));
    int e = ex - 127 + 15;
    if (e >= 0x1f) return ushort(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return ushort(sign);
        uint m = man | 0x800000;
        uint shift = uint(14 - e);
        uint half_ = m >> shift;
        uint rem = m & ((1u << shift) - 1);
        uint mid = 1u << (shift - 1);
        bool up = rem > mid || (rem == mid && (half_ & 1) == 1);
        return ushort(sign | (half_ + (up ? 1 : 0)));
    }
    uint half_ = (uint(e) << 10) | (man >> 13);
    uint rem = man & 0x1fff;
    bool up = rem > 0x1000 || (rem == 0x1000 && (half_ & 1) == 1);
    return ushort(sign | (half_ + (up ? 1 : 0)));
}

#define CAST_KERNEL(NAME, TI, TO, EXPR) \
kernel void NAME(device const TI* x [[buffer(0)]], device TO* y [[buffer(1)]], constant Strided& p [[buffer(2)]], uint i [[thread_position_in_grid]]) { \
    if (i >= p.n) return; \
    TI v = x[((p.contig & 1u) ? p.off_a + i : off_of(i, p, p.sa, p.off_a))]; \
    y[i] = EXPR; \
}
CAST_KERNEL(cast_f16_f32, ushort, float, f16_to_f32(v))
CAST_KERNEL(cast_bf16_f32, ushort, float, bf16_to_f32(v))
CAST_KERNEL(cast_f32_f16, float, ushort, f32_to_f16(v))
CAST_KERNEL(cast_f32_bf16, float, ushort, f32_to_bf16(v))
CAST_KERNEL(cast_bool_f32, uchar, float, (v != 0 ? 1.0f : 0.0f))

struct ScalarOp { uint op; float s; };

// ---------------------------------------------------------------- nn::Adam, fused
// One thread per element, the same f32 operations in the same order as nn::Adam's op sequence
// (each op rounds; no contraction: the products are kept apart from the sums). flags: 1 coupled
// decay, 2 decoupled decay before the step (torch), 4 after it (the default).

struct AdamScalars { float wd, b1, omb1, b2, omb2, bc1, bc2, eps, lr, pre, post; uint flags; };

kernel void adam_f32(device const float* p [[buffer(0)]], device const float* g [[buffer(1)]],
                     device const float* m [[buffer(2)]], device const float* v [[buffer(3)]],
                     device float* po [[buffer(4)]], device float* mo [[buffer(5)]], device float* vo [[buffer(6)]],
                     constant AdamScalars& s [[buffer(7)]], constant uint& n [[buffer(8)]],
                     uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if (i >= n) return;
    float pi = p[i], gi = g[i];
    if (s.flags & 1u) gi = gi + pi * s.wd;
    float mn = m[i] * s.b1 + gi * s.omb1;
    float vn = v[i] * s.b2 + (gi * gi) * s.omb2;
    float mh = mn / s.bc1;
    float vh = vn / s.bc2;
    float u = mh / (precise::sqrt(vh) + s.eps);
    if (s.flags & 2u) pi = pi * s.pre;
    float pn = pi - u * s.lr;
    if (s.flags & 4u) pn = pn * s.post;
    po[i] = pn;
    mo[i] = mn;
    vo[i] = vn;
}

kernel void fill_f32(device float* y [[buffer(0)]], constant float& v [[buffer(1)]], constant uint& n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = v;
}

// The stable sigmoid in f32, as CpuRef's: 1 / (1 + e^−x) for x ≥ 0, e^x / (1 + e^x) below.
inline float sigmoid_std(float a) {
    if (a >= 0.0f) {
        return 1.0f / (1.0f + precise::exp(-a));
    }
    float e = precise::exp(a);
    return e / (1.0f + e);
}

kernel void unary_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant Strided& p [[buffer(2)]], constant ScalarOp& u [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float a = x[((p.contig & 1u) ? p.off_a + i : off_of(i, p, p.sa, p.off_a))];
    float s = u.s;
    float r;
    switch (u.op) {
        case 0: r = -a; break;
        case 1: r = precise::exp(a); break;
        case 2: r = precise::log(a); break;
        case 3: r = precise::sqrt(a); break;
        case 4: r = 1.0f / a; break;
        case 5: r = fabs(a); break;
        case 6: r = (a == 0.0f) ? 0.0f : (signbit(a) ? -1.0f : 1.0f); break;
        case 7: r = sigmoid_std(a); break;
        case 8: r = a + s; break;
        case 9: r = a - s; break;
        case 10: r = a * s; break;
        case 11: r = a / s; break;
        default: r = precise::pow(a, s); break;
    }
    y[i] = r;
}

kernel void compare_f32(device const float* x [[buffer(0)]], device uchar* y [[buffer(1)]],
                        constant Strided& p [[buffer(2)]], constant ScalarOp& u [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float a = x[((p.contig & 1u) ? p.off_a + i : off_of(i, p, p.sa, p.off_a))];
    float s = u.s;
    bool r;
    switch (u.op) {
        case 0: r = a > s; break;
        case 1: r = a >= s; break;
        case 2: r = a < s; break;
        case 3: r = a <= s; break;
        default: r = a == s; break;
    }
    y[i] = r ? 1 : 0;
}

kernel void binary_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* c [[buffer(2)]],
                       constant Strided& p [[buffer(3)]], constant uint& op [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float x = a[((p.contig & 1u) ? p.off_a + i : off_of(i, p, p.sa, p.off_a))];
    float y = b[((p.contig & 2u) ? p.off_b + i : off_of(i, p, p.sb, p.off_b))];
    float r;
    switch (op) {
        case 0: r = x + y; break;
        case 1: r = x - y; break;
        case 2: r = x * y; break;
        default: r = x / y; break;
    }
    c[i] = r;
}

kernel void mask_fill_f32(device const float* x [[buffer(0)]], device const uchar* m [[buffer(1)]], device float* y [[buffer(2)]],
                          constant Strided& p [[buffer(3)]], constant float& value [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float a = x[((p.contig & 1u) ? p.off_a + i : off_of(i, p, p.sa, p.off_a))];
    y[i] = m[((p.contig & 2u) ? p.off_b + i : off_of(i, p, p.sb, p.off_b))] != 0 ? value : a;
}

// ---------------------------------------------------------------- reductions
// One thread per output: CpuRef's order exactly (ndarray's sum_axis), so sums equal CpuRef's.

struct Lanes { uint outer, n, inner, m; };

// ndarray's unrolled_fold with + from zero: eight partial sums, combined as (p0+p4), (p1+p5),
// (p2+p6), (p3+p7), then the tail in order.
inline float unrolled_sum(device const float* x, uint len) {
    float acc = 0.0f;
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, p4 = 0.0f, p5 = 0.0f, p6 = 0.0f, p7 = 0.0f;
    uint j = 0;
    while (len - j >= 8) {
        p0 = p0 + x[j + 0];
        p1 = p1 + x[j + 1];
        p2 = p2 + x[j + 2];
        p3 = p3 + x[j + 3];
        p4 = p4 + x[j + 4];
        p5 = p5 + x[j + 5];
        p6 = p6 + x[j + 6];
        p7 = p7 + x[j + 7];
        j += 8;
    }
    acc = acc + (p0 + p4);
    acc = acc + (p1 + p5);
    acc = acc + (p2 + p6);
    acc = acc + (p3 + p7);
    for (; j < len; ++j) acc = acc + x[j];
    return acc;
}

kernel void sum_last_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                         constant Lanes& p [[buffer(2)]], uint lane [[thread_position_in_grid]]) {
    if (lane >= p.outer) return;
    y[lane] = unrolled_sum(x + (ulong)lane * p.n, p.n);
}

kernel void sum_mid_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant Lanes& p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.inner) return;
    uint o = t / p.inner, i = t % p.inner;
    float acc = 0.0f;
    for (uint k = 0; k < p.n; ++k) acc = acc + x[((ulong)o * p.n + k) * p.inner + i];
    y[t] = acc;
}

kernel void sum_all_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    y[0] = unrolled_sum(x, n);
}

// The fold `if a is NaN or a > b { a } else { b }` from the first element: NaN-propagating, as
// CpuRef's max.
kernel void max_all_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    float a = x[0];
    for (uint j = 1; j < n; ++j) { float b = x[j]; a = (isnan(a) || a > b) ? a : b; }
    y[0] = a;
}

// The first maximum along the lane, its index and value, NaN-propagating as CpuRef's argmax
//: the first NaN is the maximum; otherwise a strictly greater value replaces.
kernel void argmax_dim_f32(device const float* x [[buffer(0)]], device uint* idx [[buffer(1)]], device float* val [[buffer(2)]],
                           constant Lanes& p [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.inner) return;
    uint o = t / p.inner, i = t % p.inner;
    device const float* lane = x + (ulong)o * p.n * p.inner + i;
    float best = lane[0];
    uint bi = 0;
    for (uint k = 0; k < p.n; ++k) {
        float e = lane[(ulong)k * p.inner];
        if (isnan(best)) break;
        if (isnan(e) || e > best) { best = e; bi = k; }
    }
    idx[t] = bi;
    val[t] = best;
}

// ---------------------------------------------------------------- sort
// Descending sort of each lane of a contiguous [outer, n, inner] tensor, one threadgroup per lane:
// a bitonic network over n padded to a power of two (≤ 4096; 32 KB of threadgroup memory). The order
// is Rust's `total_cmp` reversed (the key maps the f32 bits to a signed integer with the same
// order), every NaN one key above +inf, ties by the smaller source index first; padding sorts last. The network is fixed, so
// the result is deterministic; the order is a total order, so it is the stable descending sort.

struct SortP { uint outer, n, inner, np2; };

inline int total_key(float x) {
    // Every NaN, whatever its sign and payload, is one key above +inf (CpuRef's sort_order).
    if (isnan(x)) return INT_MAX;
    int b = as_type<int>(x);
    return b ^ int(uint(b >> 31) >> 1);
}

inline bool sorts_before(int ka, uint ia, int kb, uint ib) {
    return ka > kb || (ka == kb && ia < ib);
}

kernel void sort_desc_f32(device const float* x [[buffer(0)]], device float* vals [[buffer(1)]], device int* idx [[buffer(2)]],
                          constant SortP& p [[buffer(3)]],
                          uint lane [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint nt [[threads_per_threadgroup]]) {
    threadgroup int key[4096];
    threadgroup uint ix[4096];
    const uint o = lane / p.inner, i = lane % p.inner;
    device const float* src = x + (ulong)o * p.n * p.inner + i;
    for (uint e = t; e < p.np2; e += nt) {
        if (e < p.n) { key[e] = total_key(src[(ulong)e * p.inner]); ix[e] = e; }
        else { key[e] = INT_MIN; ix[e] = 0xFFFFFFFFu; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= p.np2; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint e = t; e < p.np2; e += nt) {
                uint q = e ^ j;
                if (q > e) {
                    bool forward = (e & k) == 0;
                    bool in_order = sorts_before(key[e], ix[e], key[q], ix[q]);
                    if (forward != in_order) {
                        int tk = key[e]; key[e] = key[q]; key[q] = tk;
                        uint ti = ix[e]; ix[e] = ix[q]; ix[q] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    device float* dv = vals + (ulong)o * p.n * p.inner + i;
    device int* di = idx + (ulong)o * p.n * p.inner + i;
    for (uint e = t; e < p.n; e += nt) {
        dv[(ulong)e * p.inner] = src[(ulong)ix[e] * p.inner];
        di[(ulong)e * p.inner] = int(ix[e]);
    }
}

// ---------------------------------------------------------------- indexing
// Deterministic scatter and index adds: one thread per target, contributions in ascending
// source order (CpuRef's order), no atomics.

kernel void gather_f32(device const float* x [[buffer(0)]], device const uint* idx [[buffer(1)]], device float* y [[buffer(2)]],
                       constant Lanes& p [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.m * p.inner) return;
    uint i = t % p.inner, o = t / (p.m * p.inner);
    y[t] = x[((ulong)o * p.n + idx[t]) * p.inner + i];
}

kernel void index_select_f32(device const float* x [[buffer(0)]], device const uint* idx [[buffer(1)]], device float* y [[buffer(2)]],
                             constant Lanes& p [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.m * p.inner) return;
    uint i = t % p.inner, r = t / p.inner, k = r % p.m, o = r / p.m;
    y[t] = x[((ulong)o * p.n + idx[k]) * p.inner + i];
}

kernel void one_hot_f32(device const uint* idx [[buffer(0)]], device float* y [[buffer(1)]],
                        constant Lanes& p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.n) return;
    y[t] = (idx[t / p.n] == t % p.n) ? 1.0f : 0.0f;
}

// scatter_add into zeros of [outer, n, inner] from values [outer, m, inner] at idx (same shape).
kernel void scatter_add_f32(device const uint* idx [[buffer(0)]], device const float* v [[buffer(1)]], device float* y [[buffer(2)]],
                            constant Lanes& p [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.n * p.inner) return;
    uint i = t % p.inner, r = t / p.inner, j = r % p.n, o = r / p.n;
    float acc = 0.0f;
    for (uint k = 0; k < p.m; ++k) {
        ulong q = ((ulong)o * p.m + k) * p.inner + i;
        if (idx[q] == j) acc = acc + v[q];
    }
    y[t] = acc;
}

// index_add into zeros of [outer, n, inner]: value slice k goes to slice idx[k], in order.
kernel void index_add_f32(device const uint* idx [[buffer(0)]], device const float* v [[buffer(1)]], device float* y [[buffer(2)]],
                          constant Lanes& p [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.n * p.inner) return;
    uint i = t % p.inner, r = t / p.inner, j = r % p.n, o = r / p.n;
    float acc = 0.0f;
    for (uint k = 0; k < p.m; ++k) {
        if (idx[k] == j) acc = acc + v[((ulong)o * p.m + k) * p.inner + i];
    }
    y[t] = acc;
}
"#;
