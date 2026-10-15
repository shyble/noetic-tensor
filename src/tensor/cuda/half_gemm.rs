//! The CUDA kernel of `MatmulPrecision::Half`: bf16 operands on the tensor cores, f32
//! accumulators. A source of its own, compiled by NVRTC on first use only (compute capability 8.0
//! or later), so the f32 kernels' source, and with it the CUDA key, is unchanged when the option is
//! off.

/// The kernel's source; its sha256 is in `half_gemm_key`.
pub(crate) const SOURCE: &str = r#"
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned long long ulong;

// C[z] = A[z] · B[z] with A and B read as f32 through element strides and per-batch element
// offsets (as the f32 matmul reads them), each element rounded to bf16 (nearest, ties to even;
// NaN stays NaN) as it enters shared memory. A 64×64 output tile per block of 128 threads (four
// warps, 2×2, each a 32×32 warp tile of 2×4 mma.sync.m16n8k16 tiles); k in steps of 32 in order,
// each step two k16 products into the same f32 accumulators. Zero padding past the edges adds
// exact zeros. No split over k, no atomics: one fixed order per output element.

struct MatmulParams { uint m, n, k, a_rs, a_cs, b_rs, b_cs, zbase, nkc; };

__device__ __forceinline__ ushort bf16_rne(float x) {
    uint u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (ushort)((u >> 16) | 0x40u);
    return (ushort)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

#define HM 64
#define HN 64
#define HK 32
#define HP 8

extern "C" __global__ void __launch_bounds__(128) hgemm_bf16_f32acc(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        const uint* __restrict__ offs, MatmulParams p) {
    __shared__ __align__(16) ushort As[HM][HK + HP];
    __shared__ __align__(16) ushort Bs[HN][HK + HP];
    const uint z = blockIdx.z + p.zbase;
    const float* a = A + offs[2 * z];
    const float* b = B + offs[2 * z + 1];
    float* c = C + (ulong)z * p.m * p.n;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int g = lane >> 2, q = lane & 3;
    const uint row0 = blockIdx.y * HM, col0 = blockIdx.x * HN;
    // Four-value loads need a row-contiguous operand whose rows start 16-byte aligned.
    const bool a_vec = p.a_cs == 1 && p.a_rs % 4 == 0 && offs[2 * z] % 4 == 0 && ((ulong)A & 15) == 0;
    const bool b_vec = p.b_cs == 1 && p.b_rs % 4 == 0 && offs[2 * z + 1] % 4 == 0 && ((ulong)B & 15) == 0;
    // Transposed views (the backward pass's Aᵀ and Bᵀ): contiguous along m for A, along k for B.
    const bool a_col = p.a_rs == 1 && p.a_cs != 1, b_col = p.b_rs == 1 && p.b_cs != 1;
    const bool a_colv = a_col && p.a_cs % 4 == 0 && offs[2 * z] % 4 == 0 && ((ulong)A & 15) == 0;
    const bool b_colv = b_col && p.b_cs % 4 == 0 && offs[2 * z + 1] % 4 == 0 && ((ulong)B & 15) == 0;
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.0f;
    for (uint k0 = 0; k0 < p.k; k0 += HK) {
        // Interior tiles of operands contiguous (and 16-byte aligned) along the read axis load four
        // values at a time; edges and strided views load one at a time. Both store the same bf16
        // values.
        if (a_vec && row0 + HM <= p.m && k0 + HK <= p.k) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int e = t + 128 * l;
                int r = e / (HK / 4), kk = (e % (HK / 4)) * 4;
                float4 v = *(const float4*)(a + (ulong)(row0 + r) * p.a_rs + k0 + kk);
                uint lo = (uint)bf16_rne(v.x) | ((uint)bf16_rne(v.y) << 16), hi = (uint)bf16_rne(v.z) | ((uint)bf16_rne(v.w) << 16);
                *(uint2*)&As[r][kk] = make_uint2(lo, hi);
            }
        } else if (a_colv && row0 + HM <= p.m && k0 + HK <= p.k) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int e = t + 128 * l;
                int kk = e / (HM / 4), r = (e % (HM / 4)) * 4;
                float4 v = *(const float4*)(a + row0 + r + (ulong)(k0 + kk) * p.a_cs);
                As[r][kk] = bf16_rne(v.x);
                As[r + 1][kk] = bf16_rne(v.y);
                As[r + 2][kk] = bf16_rne(v.z);
                As[r + 3][kk] = bf16_rne(v.w);
            }
        } else {
            // One at a time, consecutive threads along whichever axis of A is contiguous.
            #pragma unroll 4
            for (int l = 0; l < 16; l++) {
                int e = t + 128 * l;
                int r, kk;
                if (a_col) { kk = e / HM; r = e % HM; } else { r = e / HK; kk = e % HK; }
                uint gr = row0 + r, gk = k0 + kk;
                float v = (gr < p.m && gk < p.k) ? a[(ulong)gr * p.a_rs + (ulong)gk * p.a_cs] : 0.0f;
                As[r][kk] = bf16_rne(v);
            }
        }
        if (b_vec && col0 + HN <= p.n && k0 + HK <= p.k) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int e = t + 128 * l;
                int kk = e / (HN / 4), nn = (e % (HN / 4)) * 4;
                float4 v = *(const float4*)(b + (ulong)(k0 + kk) * p.b_rs + col0 + nn);
                Bs[nn][kk] = bf16_rne(v.x);
                Bs[nn + 1][kk] = bf16_rne(v.y);
                Bs[nn + 2][kk] = bf16_rne(v.z);
                Bs[nn + 3][kk] = bf16_rne(v.w);
            }
        } else if (b_colv && col0 + HN <= p.n && k0 + HK <= p.k) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int e = t + 128 * l;
                int nn = e / (HK / 4), kk = (e % (HK / 4)) * 4;
                float4 v = *(const float4*)(b + k0 + kk + (ulong)(col0 + nn) * p.b_cs);
                uint lo = (uint)bf16_rne(v.x) | ((uint)bf16_rne(v.y) << 16), hi = (uint)bf16_rne(v.z) | ((uint)bf16_rne(v.w) << 16);
                *(uint2*)&Bs[nn][kk] = make_uint2(lo, hi);
            }
        } else {
            #pragma unroll 4
            for (int l = 0; l < 16; l++) {
                int e = t + 128 * l;
                int kk, nn;
                if (b_col) { nn = e / HK; kk = e % HK; } else { kk = e / HN; nn = e % HN; }
                uint gk = k0 + kk, gn = col0 + nn;
                float v = (gk < p.k && gn < p.n) ? b[(ulong)gk * p.b_rs + (ulong)gn * p.b_cs] : 0.0f;
                Bs[nn][kk] = bf16_rne(v);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < HK; ks += 16) {
            uint af[2][4], bfr[4][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int r = wm + i * 16 + g;
                af[i][0] = *(const uint*)&As[r][ks + q * 2];
                af[i][1] = *(const uint*)&As[r + 8][ks + q * 2];
                af[i][2] = *(const uint*)&As[r][ks + q * 2 + 8];
                af[i][3] = *(const uint*)&As[r + 8][ks + q * 2 + 8];
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int nn = wn + j * 8 + g;
                bfr[j][0] = *(const uint*)&Bs[nn][ks + q * 2];
                bfr[j][1] = *(const uint*)&Bs[nn][ks + q * 2 + 8];
            }
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bfr[j][0]), "r"(bfr[j][1]));
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint r = row0 + wm + i * 16 + g, cc = col0 + wn + j * 8 + q * 2;
            if (r < p.m) {
                if (cc < p.n) c[(ulong)r * p.n + cc] = acc[i][j][0];
                if (cc + 1 < p.n) c[(ulong)r * p.n + cc + 1] = acc[i][j][1];
            }
            if (r + 8 < p.m) {
                if (cc < p.n) c[(ulong)(r + 8) * p.n + cc] = acc[i][j][2];
                if (cc + 1 < p.n) c[(ulong)(r + 8) * p.n + cc + 1] = acc[i][j][3];
            }
        }
}
"#;

/// The kernel's configuration, for keys: operand type, instruction, accumulator, tile.
pub(crate) const CONFIG: &str = "bf16-mma.m16n8k16-f32acc-tile64x64x32";
