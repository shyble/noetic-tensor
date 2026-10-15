//! The Metal kernel of `MatmulPrecision::Half`: f16 operands in simdgroup 8×8 matrices, f32
//! accumulators. A library of its own, compiled on first use only (with the same options as the
//! f32 kernels), so the f32 kernels' source, and with it the Metal key, is unchanged when the
//! option is off.

/// The kernel's source; its sha256 is in `half_gemm_key`.
pub(crate) const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// C[z] = A[z] · B[z] with A and B read as f32 through element strides and per-batch element
// offsets (as the f32 matmul reads them), each element converted to half (round to nearest
// even; beyond ±65504 it becomes ±inf) as it enters threadgroup memory. A 64×64 output tile per
// threadgroup of 128 threads (four simdgroups, 2×2, each a 32×32 tile of 4×4 simdgroup 8×8
// float accumulators); k in steps of 32 in order, each step four k8 products of half matrices
// into the same accumulators. Zero padding past the edges adds exact zeros. Interior tiles store
// their accumulators directly; edge tiles go through threadgroup memory and store only what is
// inside. The loads (four at a time or one at a time) and the stores move values; they do not
// change what is added, or in which order.

struct MatmulParams {
    uint m, n, k;
    uint a_rs, a_cs, b_rs, b_cs;
};

#define HM 64
#define HN 64
#define HK 32

kernel void hgemm_f16_f32acc(device const float* A [[buffer(0)]],
                             device const float* B [[buffer(1)]],
                             device float* C [[buffer(2)]],
                             device const uint2* offs [[buffer(3)]],
                             constant MatmulParams& p [[buffer(4)]],
                             uint3 tg [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half As[HM * HK];
    threadgroup half Bs[HK * HN];
    threadgroup float Cs[HM * HN];
    const uint2 o = offs[tg.z];
    device const float* a = A + o.x;
    device const float* b = B + o.y;
    device float* c = C + (ulong)tg.z * p.m * p.n;
    const uint row0 = tg.y * HM, col0 = tg.x * HN;
    const uint wm = (sg >> 1) * 32, wn = (sg & 1) * 32;
    const bool a_vec = p.a_cs == 1, b_vec = p.b_cs == 1;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.k; k0 += HK) {
        // Interior tiles of operands contiguous along the read axis load four values at a time
        // (the same values as the element loads below, which take edges and strided views).
        if (a_vec && row0 + HM <= p.m && k0 + HK <= p.k) {
            for (uint e = lid; e < HM * HK / 4; e += 128) {
                uint r = e / (HK / 4), kk = (e % (HK / 4)) * 4;
                float4 v = float4(*(device const packed_float4*)(a + (ulong)(row0 + r) * p.a_rs + k0 + kk));
                *(threadgroup half4*)(As + r * HK + kk) = half4(v);
            }
        } else {
            for (uint e = lid; e < HM * HK; e += 128) {
                uint r = e / HK, kk = e % HK;
                uint gr = row0 + r, gk = k0 + kk;
                As[e] = half((gr < p.m && gk < p.k) ? a[(ulong)gr * p.a_rs + (ulong)gk * p.a_cs] : 0.0f);
            }
        }
        if (b_vec && col0 + HN <= p.n && k0 + HK <= p.k) {
            for (uint e = lid; e < HK * HN / 4; e += 128) {
                uint kk = e / (HN / 4), cc = (e % (HN / 4)) * 4;
                float4 v = float4(*(device const packed_float4*)(b + (ulong)(k0 + kk) * p.b_rs + col0 + cc));
                *(threadgroup half4*)(Bs + kk * HN + cc) = half4(v);
            }
        } else {
            for (uint e = lid; e < HK * HN; e += 128) {
                uint kk = e / HN, cc = e % HN;
                uint gk = k0 + kk, gc = col0 + cc;
                Bs[e] = half((gk < p.k && gc < p.n) ? b[(ulong)gk * p.b_rs + (ulong)gc * p.b_cs] : 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0; ks < HK; ks += 8) {
            simdgroup_half8x8 ma[4], mb[4];
            for (uint i = 0; i < 4; ++i) simdgroup_load(ma[i], As + (wm + i * 8) * HK + ks, HK);
            for (uint j = 0; j < 4; ++j) simdgroup_load(mb[j], Bs + ks * HN + wn + j * 8, HN);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 + HM <= p.m && col0 + HN <= p.n) {
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], c + (ulong)(row0 + wm + i * 8) * p.n + col0 + wn + j * 8, p.n);
        return;
    }
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_store(acc[i][j], Cs + (wm + i * 8) * HN + wn + j * 8, HN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lid; e < HM * HN; e += 128) {
        uint r = row0 + e / HN, cc = col0 + e % HN;
        if (r < p.m && cc < p.n) c[(ulong)r * p.n + cc] = Cs[e];
    }
}
"#;

/// The kernel's configuration, for keys: operand type, instruction, accumulator, tile.
pub(crate) const CONFIG: &str = "f16-simdgroup8x8-f32acc-tile64x64x32";
