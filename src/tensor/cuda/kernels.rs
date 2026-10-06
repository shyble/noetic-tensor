//! The CUDA kernels, compiled at first use by NVRTC without fast math and
//! with `--fmad=false` (no contraction; the only fused multiply-adds are matmul's explicit
//! `fmaf`). They mirror the Metal kernels one for one (`metal/shaders.rs`): one thread per
//! output element, reductions in CpuRef's order, deterministic scatter and index adds without
//! atomics, and a matmul in matrixmultiply's order (fma chains closed in k blocks of 256).

pub(crate) const SOURCE: &str = r#"
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned long long ulong;

// ---------------------------------------------------------------- matmul
// C[z] = A[z] · B[z] for M×K and K×N blocks read through element strides, with per-batch
// element offsets (broadcast batches share a block). A 64×64 output tile per block of 256
// threads, each a 4×4 register tile. K runs in order; every product enters its accumulator by one
// fmaf; the accumulator is closed every KC = 256 (the first block is the total, later blocks are
// added to it: block sum + total). That is matrixmultiply 0.3.11's sgemm order (kc = 256, an fma
// chain from zero per block, then ab + c), so the result equals CpuRef's to the bit.

struct MatmulParams { uint m, n, k, a_rs, a_cs, b_rs, b_cs, zbase, nkc; };

#define BM 64
#define BN 64
#define BK 16
#define KC 256

extern "C" __global__ void __launch_bounds__(256) matmul_f32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        const uint* __restrict__ offs, MatmulParams p) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    const uint lid = threadIdx.x, tx = lid % 16, ty = lid / 16;
    const uint row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const uint z = blockIdx.z + p.zbase;
    const float* a = A + offs[2 * z];
    const float* b = B + offs[2 * z + 1];
    float* c = C + (ulong)z * p.m * p.n;
    float acc[4][4], tot[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { acc[i][j] = 0.0f; tot[i][j] = 0.0f; }
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
        __syncthreads();
        const uint kmax = min((uint)BK, p.k - k0);
        for (uint kk = 0; kk < kmax; ++kk) {
            float av[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) av[i] = As[kk][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = Bs[kk][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
        if ((k0 + BK) % KC == 0 || k0 + BK >= p.k) {
            const bool first = k0 < KC;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    tot[i][j] = first ? acc[i][j] : __fadd_rn(acc[i][j], tot[i][j]);
                    acc[i][j] = 0.0f;
                }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint r = row0 + ty + 16 * i;
        if (r >= p.m) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint cc = col0 + tx + 16 * j;
            if (cc < p.n) c[(ulong)r * p.n + cc] = tot[i][j];
        }
    }
}

// The same product with a 128×128 output tile per block of 256 threads, each an 8×8
// register tile (rows ty + 16·i, columns tx + 16·j), for large outputs. Every output is the same
// fmaf chain over k in order, closed every KC = 256, so the result is bit-identical to
// matmul_f32's (and CpuRef's); only the tiling differs.
#define BM2 128
#define BN2 128
#define BK2 16
#define PAD2 4
extern "C" __global__ void __launch_bounds__(256) matmul128_f32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        const uint* __restrict__ offs, MatmulParams p) {
    // Rows padded by 4 floats: the transposed-operand stores (8 or 16 k values of one row or
    // column from neighbouring threads) fall in distinct banks.
    __shared__ float As[BK2][BM2 + PAD2];
    __shared__ float Bs[BK2][BN2 + PAD2];
    const uint lid = threadIdx.x, tx = lid % 16, ty = lid / 16;
    const uint row0 = blockIdx.y * BM2, col0 = blockIdx.x * BN2;
    // Split over k (nkc > 0): z runs over (batch, KC block); each block writes its KC block's
    // chain (from zero) to its own slice of C, and kc_reduce folds the slices in order.
    const uint zz = blockIdx.z + p.zbase;
    const uint z = p.nkc ? zz / p.nkc : zz;
    const uint kbeg = p.nkc ? (zz % p.nkc) * KC : 0;
    const uint kend = p.nkc ? min(p.k, kbeg + KC) : p.k;
    const float* a = A + offs[2 * z];
    const float* b = B + offs[2 * z + 1];
    float* c = C + (ulong)zz * p.m * p.n;
    const bool a_rows = p.a_cs == 1, b_rows = p.b_cs == 1 || p.b_rs != 1;
    float acc[8][8], tot[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; tot[i][j] = 0.0f; }
    for (uint k0 = kbeg; k0 < kend; k0 += BK2) {
        // Loads walk the operand's unit-stride axis across neighbouring threads (coalesced for
        // row-major and for transposed views alike). Past K the tile holds zeros: fmaf(0, 0, acc)
        // is acc exactly (an accumulator from +0 is never −0), so the k loop always runs BK2 steps.
        #pragma unroll
        for (int l = 0; l < (BM2 * BK2) / 256; ++l) {
            uint e = lid + 256 * l;
            uint r, kk;
            if (a_rows) { r = e / BK2; kk = e % BK2; } else { r = e % BM2; kk = e / BM2; }
            uint gr = row0 + r, gk = k0 + kk;
            As[kk][r] = (gr < p.m && gk < kend) ? a[(ulong)gr * p.a_rs + (ulong)gk * p.a_cs] : 0.0f;
            uint kb, cc;
            if (b_rows) { kb = e / BN2; cc = e % BN2; } else { kb = e % BK2; cc = e / BK2; }
            uint gkb = k0 + kb, gc = col0 + cc;
            Bs[kb][cc] = (gkb < kend && gc < p.n) ? b[(ulong)gkb * p.b_rs + (ulong)gc * p.b_cs] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK2; ++kk) {
            float av[8], bv[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) av[i] = As[kk][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 8; ++j) bv[j] = Bs[kk][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
        if ((k0 + BK2) % KC == 0 || k0 + BK2 >= kend) {
            const bool first = k0 < KC || p.nkc;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    tot[i][j] = first ? acc[i][j] : __fadd_rn(acc[i][j], tot[i][j]);
                    acc[i][j] = 0.0f;
                }
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint r = row0 + ty + 16 * i;
        if (r >= p.m) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint cc = col0 + tx + 16 * j;
            if (cc < p.n) c[(ulong)r * p.n + cc] = tot[i][j];
        }
    }
}

// The split-k fold: C[z] = the KC-block chains of slices (z, 0..nkc) folded in order as the
// unsplit kernel does (block sum + running total).
extern "C" __global__ void kc_reduce_f32(const float* __restrict__ parts, float* __restrict__ c, uint nkc, uint mn, uint batches) {
    uint t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= mn * batches) return;
    uint z = t / mn, i = t % mn;
    const float* s = parts + (ulong)z * nkc * mn + i;
    float tot = s[0];
    for (uint j = 1; j < nkc; ++j) tot = __fadd_rn(s[(ulong)j * mn], tot);
    c[t] = tot;
}

// ---------------------------------------------------------------- strided elementwise
// Element i of the output (row-major over `shape`) reads a at off_a + Σ coord·sa (0 strides
// broadcast), b likewise, and writes c at off_c + Σ coord·sc.

#define MAXR 8

struct Strided {
    uint n, rank, off_a, off_b, off_c;
    uint shape[MAXR];
    uint sa[MAXR];
    uint sb[MAXR];
    uint sc[MAXR];
};

__device__ __forceinline__ uint off_of(uint i, const Strided& p, const uint* st, uint base) {
    uint o = base;
    for (int d = int(p.rank) - 1; d >= 0; --d) {
        uint s = p.shape[d];
        o += (i % s) * st[d];
        i /= s;
    }
    return o;
}

#define TID (blockIdx.x * blockDim.x + threadIdx.x)

#define COPY_KERNEL(NAME, T) \
extern "C" __global__ void NAME(const T* __restrict__ x, T* __restrict__ y, Strided p) { \
    uint i = TID; \
    if (i >= p.n) return; \
    y[off_of(i, p, p.sc, p.off_c)] = x[off_of(i, p, p.sa, p.off_a)]; \
}
COPY_KERNEL(copy_strided_8, uchar)
COPY_KERNEL(copy_strided_16, ushort)
COPY_KERNEL(copy_strided_32, uint)
COPY_KERNEL(copy_strided_64, ulong)

// f16 and bf16 bit patterns, as tensor::half converts them (round to nearest, ties to even).
__device__ __forceinline__ float bf16_to_f32(ushort h) { return __uint_as_float(uint(h) << 16); }
__device__ __forceinline__ ushort f32_to_bf16(float x) {
    uint b = __float_as_uint(x);
    if (x != x) return ushort((b >> 16) | 0x0040);
    uint round = 0x7fff + ((b >> 16) & 1);
    return ushort((b + round) >> 16);
}
__device__ __forceinline__ float f16_to_f32(ushort hh) {
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
    return __uint_as_float(bits);
}
__device__ __forceinline__ ushort f32_to_f16(float x) {
    uint b = __float_as_uint(x);
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
extern "C" __global__ void NAME(const TI* __restrict__ x, TO* __restrict__ y, Strided p) { \
    uint i = TID; \
    if (i >= p.n) return; \
    TI v = x[off_of(i, p, p.sa, p.off_a)]; \
    y[i] = EXPR; \
}
CAST_KERNEL(cast_f16_f32, ushort, float, f16_to_f32(v))
CAST_KERNEL(cast_bf16_f32, ushort, float, bf16_to_f32(v))
CAST_KERNEL(cast_f32_f16, float, ushort, f32_to_f16(v))
CAST_KERNEL(cast_f32_bf16, float, ushort, f32_to_bf16(v))
CAST_KERNEL(cast_bool_f32, uchar, float, (v != 0 ? 1.0f : 0.0f))

struct ScalarOp { uint op; float s; };

// ---------------------------------------------------------------- tables for exp and ln
__device__ const uint EXP_T_HI[64] = {0x3f800000u, 0x3f8164d2u, 0x3f82cd87u, 0x3f843a29u, 0x3f85aac3u, 0x3f871f62u, 0x3f88980fu, 0x3f8a14d5u, 0x3f8b95c2u, 0x3f8d1adfu, 0x3f8ea43au, 0x3f9031dcu, 0x3f91c3d3u, 0x3f935a2bu, 0x3f94f4f0u, 0x3f96942du, 0x3f9837f0u, 0x3f99e046u, 0x3f9b8d3au, 0x3f9d3edau, 0x3f9ef532u, 0x3fa0b051u, 0x3fa27043u, 0x3fa43516u, 0x3fa5fed7u, 0x3fa7cd94u, 0x3fa9a15bu, 0x3fab7a3au, 0x3fad583fu, 0x3faf3b79u, 0x3fb123f6u, 0x3fb311c4u, 0x3fb504f3u, 0x3fb6fd92u, 0x3fb8fbafu, 0x3fbaff5bu, 0x3fbd08a4u, 0x3fbf179au, 0x3fc12c4du, 0x3fc346cdu, 0x3fc5672au, 0x3fc78d75u, 0x3fc9b9beu, 0x3fcbec15u, 0x3fce248cu, 0x3fd06334u, 0x3fd2a81eu, 0x3fd4f35bu, 0x3fd744fdu, 0x3fd99d16u, 0x3fdbfbb8u, 0x3fde60f5u, 0x3fe0ccdfu, 0x3fe33f89u, 0x3fe5b907u, 0x3fe8396au, 0x3feac0c7u, 0x3fed4f30u, 0x3fefe4bau, 0x3ff28177u, 0x3ff5257du, 0x3ff7d0dfu, 0x3ffa83b3u, 0x3ffd3e0cu};
__device__ const uint EXP_T_LO[64] = {0x00000000u, 0xb1c43fd0u, 0xb34ea7a9u, 0xb2f14c87u, 0x334f9891u, 0xb352c2e6u, 0xb37eda4bu, 0x336a92deu, 0xb260aba1u, 0x3336fcb7u, 0xb3697465u, 0x330628cdu, 0x33675624u, 0x32bc4f9cu, 0xb32e0212u, 0x32dc8061u, 0x33231b71u, 0xb359be90u, 0xb30c5563u, 0xb331a601u, 0x33412342u, 0x31fb9715u, 0x30c3125au, 0xb323ec33u, 0xb32c9d5eu, 0xb3162d36u, 0xb3162b08u, 0xb314ad82u, 0xb22deaf6u, 0xb3252debu, 0xb37c5aa8u, 0x32154889u, 0x32cfe77au, 0xb266b974u, 0x330ec5f7u, 0xb31bd983u, 0xb3414fe8u, 0xb3130b1au, 0xb2d6663eu, 0xb2976da2u, 0x320aa837u, 0xb2dd5119u, 0xb37323a2u, 0xb006c6c2u, 0x3228fc24u, 0xb2944353u, 0xb35c1daau, 0xb3286024u, 0xb2d4a58au, 0xb2f61d41u, 0xb3504a1cu, 0xb37b43e3u, 0xb21eab59u, 0x33657d15u, 0xb2441be6u, 0x33207898u, 0xb24116deu, 0x3276cca1u, 0xb348464au, 0x32f167ffu, 0x32292436u, 0x336615a2u, 0xb2923758u, 0x31cf486cu};
#define EXP_L1 __uint_as_float(0x3c318000u)
#define EXP_L2 __uint_as_float(0xb65e8083u)
#define EXP_L3 __uint_as_float(0x28e7bcd6u)
#define EXP_INV_L __uint_as_float(0x42b8aa3bu)
__device__ const uint LOG_C[128] = {0x3f800000u, 0x3f7d08e5u, 0x3f7b1885u, 0x3f792fb2u, 0x3f774e40u, 0x3f757404u, 0x3f73a0d5u, 0x3f71d48cu, 0x3f700f01u, 0x3f6e500fu, 0x3f6c9791u, 0x3f6ae564u, 0x3f693965u, 0x3f679373u, 0x3f65f36du, 0x3f645933u, 0x3f62c4a7u, 0x3f6135aau, 0x3f5fac1fu, 0x3f5e27ebu, 0x3f5ca8f1u, 0x3f5b2f17u, 0x3f59ba42u, 0x3f584a5au, 0x3f56df44u, 0x3f5578e9u, 0x3f541733u, 0x3f52ba08u, 0x3f516154u, 0x3f500d01u, 0x3f4ebcf9u, 0x3f4d7127u, 0x3f4c2978u, 0x3f4ae5d8u, 0x3f49a634u, 0x3f486a79u, 0x3f473294u, 0x3f45fe74u, 0x3f44ce08u, 0x3f43a13eu, 0x3f427806u, 0x3f415250u, 0x3f40300cu, 0x3f3f112bu, 0x3f3df59du, 0x3f3cdd53u, 0x3f3bc841u, 0x3f3ab656u, 0x3f39a786u, 0x3f389bc3u, 0x3f379301u, 0x3f368d31u, 0x3f358a48u, 0x3f348a3au, 0x3f338cfau, 0x3f32927cu, 0x3f319ab6u, 0x3f30a59bu, 0x3f2fb322u, 0x3f2ec33eu, 0x3f2dd5e6u, 0x3f2ceb10u, 0x3f2c02b0u, 0x3f2b1cbeu, 0x3f2a392fu, 0x3f2957fbu, 0x3f287917u, 0x3f279c7bu, 0x3f26c21eu, 0x3f25e9f7u, 0x3f2513fdu, 0x3f244029u, 0x3f236e72u, 0x3f229ecfu, 0x3f21d13au, 0x3f2105a9u, 0x3f203c17u, 0x3f1f747au, 0x3f1eaecdu, 0x3f1deb07u, 0x3f1d2922u, 0x3f1c6917u, 0x3f1baadfu, 0x3f1aee73u, 0x3f1a33cdu, 0x3f197ae7u, 0x3f18c3bbu, 0x3f180e41u, 0x3f175a75u, 0x3f16a850u, 0x3f15f7ccu, 0x3f1548e5u, 0x3f149b93u, 0x3f13efd2u, 0x3f13459cu, 0x3f129cecu, 0x3f11f5bdu, 0x3f115009u, 0x3f10abccu, 0x3f100901u, 0x3f0f67a2u, 0x3f0ec7abu, 0x3f0e2918u, 0x3f0d8be3u, 0x3f0cf009u, 0x3f0c5584u, 0x3f0bbc51u, 0x3f0b246bu, 0x3f0a8dcdu, 0x3f09f874u, 0x3f09645cu, 0x3f08d181u, 0x3f083fdeu, 0x3f07af70u, 0x3f072033u, 0x3f069223u, 0x3f06053cu, 0x3f05797cu, 0x3f04eeddu, 0x3f04655eu, 0x3f03dcf9u, 0x3f0355adu, 0x3f02cf75u, 0x3f024a4eu, 0x3f01c636u, 0x3f014328u, 0x3f00c122u, 0x3f000000u};
__device__ const uint LOG_T_HI[128] = {0x00000000u, 0x3c3ee24fu, 0x3c9e752fu, 0x3cdcfe05u, 0x3d0d86c8u, 0x3d2c52dbu, 0x3d4ae41bu, 0x3d693b59u, 0x3d83acc1u, 0x3d929fb1u, 0x3da176e7u, 0x3db032c5u, 0x3dbed3b5u, 0x3dcd5a11u, 0x3ddbc63fu, 0x3dea189du, 0x3df85182u, 0x3e0338a8u, 0x3e0a3c2eu, 0x3e11337au, 0x3e181ebau, 0x3e1efe16u, 0x3e25d1b9u, 0x3e2c99c5u, 0x3e33566au, 0x3e3a07cau, 0x3e40ae03u, 0x3e474948u, 0x3e4dd9b3u, 0x3e545f67u, 0x3e5ada8cu, 0x3e614b44u, 0x3e67b1adu, 0x3e6e0de7u, 0x3e746013u, 0x3e7aa850u, 0x3e807362u, 0x3e838dc3u, 0x3e86a35au, 0x3e89b438u, 0x3e8cc06au, 0x3e8fc7fdu, 0x3e92cb01u, 0x3e95c981u, 0x3e98c38du, 0x3e9bb934u, 0x3e9eaa7cu, 0x3ea19779u, 0x3ea48034u, 0x3ea764bbu, 0x3eaa4515u, 0x3ead2156u, 0x3eaff984u, 0xbeb01685u, 0xbead4658u, 0xbeaa7a18u, 0xbea7b1c0u, 0xbea4ed3fu, 0xbea22c90u, 0xbe9f6fa3u, 0xbe9cb671u, 0xbe9a00f2u, 0xbe974f16u, 0xbe94a0d8u, 0xbe91f62cu, 0xbe8f4f0cu, 0xbe8cab6au, 0xbe8a0b3fu, 0xbe876e83u, 0xbe84d52cu, 0xbe823f2fu, 0xbe7f5911u, 0xbe7a3a5bu, 0xbe752224u, 0xbe701069u, 0xbe6b050bu, 0xbe660009u, 0xbe610145u, 0xbe5c08bcu, 0xbe571655u, 0xbe522a06u, 0xbe4d43bfu, 0xbe486371u, 0xbe43890au, 0xbe3eb47fu, 0xbe39e5c5u, 0xbe351cd0u, 0xbe305986u, 0xbe2b9be6u, 0xbe26e3dcu, 0xbe22315au, 0xbe1d845eu, 0xbe18dccbu, 0xbe143a9fu, 0xbe0f9dcau, 0xbe0b0640u, 0xbe0673f8u, 0xbe01e6e0u, 0xbdfabde4u, 0xbdf1b846u, 0xbde8bcbeu, 0xbddfcb40u, 0xbdd6e3c1u, 0xbdce0616u, 0xbdc5323fu, 0xbdbc6810u, 0xbdb3a787u, 0xbdaaf087u, 0xbda242eeu, 0xbd999ebau, 0xbd9103d6u, 0xbd887231u, 0xbd7fd34au, 0xbd6ed45cu, 0xbd5de76au, 0xbd4d0c47u, 0xbd3c42c5u, 0xbd2b8af0u, 0xbd1ae459u, 0xbd0a4f2bu, 0xbcf395e6u, 0xbcd2afb1u, 0xbcb1eb0bu, 0xbc9147c1u, 0xbc618bbeu, 0xbc20c95eu, 0xbbc090deu, 0x00000000u};
__device__ const uint LOG_T_LO[128] = {0x00000000u, 0x2f70215cu, 0xae395a38u, 0x307af91au, 0x30923d99u, 0xb08371b2u, 0x30cc8e3fu, 0x2fb6c2d7u, 0x30931c8eu, 0xb15f5ebeu, 0xb0cb3722u, 0x3153750cu, 0xb1784ed3u, 0x30a40676u, 0xb154f125u, 0x3116cb36u, 0xb0ddd3a7u, 0xb1c35a69u, 0x315ce858u, 0xb1d75a53u, 0x31c647feu, 0x309b318bu, 0x31534abeu, 0xb1967df7u, 0x2f655aa2u, 0xaeff2116u, 0xb1479500u, 0x30a21b51u, 0xb1b82047u, 0xaff7d0e4u, 0xb1bf04a5u, 0x3190c18du, 0xb1df63a1u, 0xb0d504ddu, 0xb1f55a5au, 0x31ecb7b5u, 0xb1ea55fcu, 0x304d590du, 0xb119522cu, 0xb1bd8aeau, 0xb23abdc9u, 0x311228bdu, 0xb262089cu, 0xb248762du, 0x31828832u, 0x311ff619u, 0xb0bf0472u, 0x32784e81u, 0x326c009du, 0xb1fbffb1u, 0x3226dd26u, 0x314d33f9u, 0x326a793eu, 0x31b1edc3u, 0x31f8240du, 0xb247be1au, 0xb09d4c06u, 0x3180677du, 0xb134e7d0u, 0x31f7d2f8u, 0xb146efd4u, 0x30a89a37u, 0x31e3dbafu, 0xb23986b8u, 0xb19bd749u, 0xb172267cu, 0x31f487a1u, 0xb1cecef0u, 0xb186f010u, 0x31bb2726u, 0xb147a0ebu, 0xb16d40fau, 0x30ec837bu, 0xb1d84aafu, 0xb1646b52u, 0x318efcafu, 0xb195b3c3u, 0x313d00cau, 0xb1d3823bu, 0x31031a86u, 0x31c19550u, 0xaf57543cu, 0x31883435u, 0x3145e2cdu, 0xb165eb56u, 0xb1dcfd13u, 0xb1a3bd38u, 0x3179ed74u, 0x310fce64u, 0xb03de0e4u, 0x3106c989u, 0xafd7f28au, 0xb1aba4f8u, 0xb1f11abau, 0x3101b458u, 0xb1c4b27bu, 0xb1c4cc28u, 0x3182e13fu, 0x30cdff45u, 0x312cfb7du, 0x30bbcd94u, 0xb179fca0u, 0xb0cb5d51u, 0x30b65b5au, 0x3146e23eu, 0x315c27d3u, 0xb16365a6u, 0xb00380c9u, 0xb10dbdfau, 0x313e7c10u, 0xb1078547u, 0x3153517eu, 0xb0dab930u, 0x2fe60473u, 0x3032e188u, 0x2e8c21dbu, 0x30dca887u, 0x30bfc31cu, 0xb0ca1304u, 0xb08005a0u, 0xb0413cdau, 0xaf98b885u, 0x2f65bd0bu, 0xaffd45a2u, 0x2fb38a21u, 0x2f52f9a8u, 0x2f3a7979u, 0x00000000u};
#define LN2_HI __uint_as_float(0x3f317200u)
#define LN2_LO __uint_as_float(0x35bfbe8eu)
#define LN2_LO2 __uint_as_float(0x29779abdu)

// ---------------------------------------------------------------- f32 exp, ln and pow
// Our own f32 implementations: range reduction by lookup tables and short
// polynomials evaluated in float-float (pairs hi + lo) with explicit fmaf, compiled with
// --fmad=false and no fast math, so every operation rounds as written and results are
// deterministic. The float-float value carries about 2^-40 relative error, so the final rounding
// to f32 is correct except when the exact result lies within that distance of a rounding
// midpoint (then it is off by one ulp at most). Tables: exact values rounded to f32 (generated
// with exact decimal arithmetic; hi = nearest f32, lo = nearest f32 of the remainder).

__device__ __forceinline__ float fbits(uint b) { return __uint_as_float(b); }

// a + b = s + e exactly (Knuth), and the |a| >= |b| form (Dekker).
__device__ __forceinline__ void two_sum(float a, float b, float& s, float& e) {
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void fast_two_sum(float a, float b, float& s, float& e) {
    s = a + b;
    e = b - (s - a);
}
// a · b = p + e exactly.
__device__ __forceinline__ void two_prod(float a, float b, float& p, float& e) {
    p = a * b;
    e = fmaf(a, b, -p);
}

// exp(xh + xl) rounded to f32 (xl tiny against xh).
__device__ float exp_ff(float xh, float xl) {
    if (xh != xh) return xh + xh;
    if (xh > 89.0f) return __uint_as_float(0x7f800000u);
    if (xh < -104.0f) return 0.0f;
    // x = N·ln2/64 + r, N = 64k + j; exp(x) = 2^k · 2^(j/64) · exp(r), |r| ≤ ln2/128 (+ tiny).
    float nf = rintf(xh * EXP_INV_L);
    int n = (int)nf;
    int j = n & 63, k = n >> 6;
    float t1 = fmaf(-nf, EXP_L1, xh);  // exact: N·L1 has ≤ 24 bits and cancels against x
    float p, pe;
    two_prod(nf, EXP_L2, p, pe);
    float rh, rl;
    two_sum(t1, -p, rh, rl);
    rl = rl - pe - nf * EXP_L3 + xl;
    fast_two_sum(rh, rl, rh, rl);
    // exp(r) − 1 = r + r²/2 + r³/6 + r⁴/24 + r⁵/120 (r⁶/720 < 2^-50).
    float s, se;
    two_prod(rh, rh, s, se);
    float q = rh * s * (1.0f / 6.0f + rh * (1.0f / 24.0f + rh * (1.0f / 120.0f)));
    float ph, pl;
    fast_two_sum(rh, 0.5f * s, ph, pl);
    pl = pl + (rl + (0.5f * se + (rh * rl + q)));
    // 2^(j/64) · (1 + p) in float-float.
    float th = fbits(__ldg(&EXP_T_HI[j])), tl = fbits(__ldg(&EXP_T_LO[j]));
    float a, ae;
    two_prod(th, ph, a, ae);
    float hi, lo;
    fast_two_sum(th, a, hi, lo);
    lo = lo + (ae + (th * pl + (tl + tl * ph)));
    float f = hi + lo;
    if (k > -126 || (k == -126 && f >= 1.0f)) {
        // A normal result: one rounding of hi + lo, then exact scaling by 2^k (split in two so
        // k up to 128 overflows to inf exactly when it should).
        int k1 = k / 2;
        return f * __uint_as_float((uint)(127 + k1) << 23) * __uint_as_float((uint)(127 + k - k1) << 23);
    }
    // A subnormal result: round (hi + lo)·2^(k+149) to an integer (ties to even) in float-float,
    // then scale by 2^-149 (exact).
    int sh = k + 149;  // in [-2, 22]
    float sc = sh >= 0 ? __uint_as_float((uint)(127 + sh) << 23) : (sh == -1 ? 0.5f : 0.25f);
    float mh = hi * sc, ml = lo * sc;  // exact (normal range)
    float nn = rintf(mh);
    float d = (mh - nn) + ml;  // mh − nn is exact
    if (d > 0.5f || (d == 0.5f && fmodf(nn, 2.0f) != 0.0f)) nn += 1.0f;
    else if (d < -0.5f || (d == -0.5f && fmodf(nn, 2.0f) != 0.0f)) nn -= 1.0f;
    return nn * __uint_as_float(0x00000001u);
}

// ln(x) as float-float (x finite, > 0).
__device__ void log_ff(float x, float& rh_out, float& rl_out) {
    int e = 0;
    uint b = __float_as_uint(x);
    if (b < 0x00800000u) {  // subnormal: scale by 2^24
        x = x * 16777216.0f;
        b = __float_as_uint(x);
        e = -24;
    }
    e += (int)(b >> 23) - 127;
    uint i = (b >> 16) & 127u;
    float m = __uint_as_float((b & 0x007fffffu) | 0x3f800000u);  // [1, 2)
    if (i >= 53) e += 1;  // m ≥ √2: ln x = (e + 1)·ln2 + ln(m/2), folded into the table
    // r = m·c − 1 exactly as float-float; ln m = −ln c + log1p(r), |r| < 2^-7.
    float c = fbits(__ldg(&LOG_C[i]));
    float mc, mce;
    two_prod(m, c, mc, mce);
    float rh, rl;
    two_sum(mc - 1.0f, mce, rh, rl);  // mc − 1 is exact (Sterbenz)
    // log1p(r) = r − r²/2 + r³(1/3 − r/4 + r²/5 − r³/6 + r⁴/7) (the next term < 2^-49·|r|).
    float s, se;
    two_prod(rh, rh, s, se);
    float poly = rh * s * (1.0f / 3.0f + rh * (-0.25f + rh * (0.2f + rh * (-1.0f / 6.0f + rh * (1.0f / 7.0f)))));
    float a, ae;
    fast_two_sum(rh, -0.5f * s, a, ae);
    float alo = ae + (rl - (0.5f * se + (rh * rl - poly)));
    // e·ln2 + T + log1p(r).
    float ef = (float)e;
    float eh = ef * LN2_HI;  // exact
    float q, qe;
    two_prod(ef, LN2_LO, q, qe);
    float th = fbits(__ldg(&LOG_T_HI[i])), tl = fbits(__ldg(&LOG_T_LO[i]));
    float s1, s1e, s2, s2e, s3, s3e;
    two_sum(eh, th, s1, s1e);
    two_sum(s1, a, s2, s2e);
    two_sum(s2, q, s3, s3e);
    float lo = s1e + s2e + s3e + (tl + (alo + (qe + ef * LN2_LO2)));
    fast_two_sum(s3, lo, rh_out, rl_out);
}

__device__ float log_f(float x) {
    if (x != x) return x + x;
    if (x < 0.0f) return __uint_as_float(0x7fc00000u);
    if (x == 0.0f) return __uint_as_float(0xff800000u);
    if (x == __uint_as_float(0x7f800000u)) return x;
    float h, l;
    log_ff(x, h, l);
    return h + l;
}

__device__ __forceinline__ float exp_f(float x) { return exp_ff(x, 0.0f); }

// Whether y is an integer, and an odd one (|y| < 2^24 for odd).
__device__ __forceinline__ bool is_int(float y) { return floorf(y) == y; }
__device__ __forceinline__ bool is_odd(float y) { return is_int(y) && fabsf(y) < 16777216.0f && fmodf(y, 2.0f) != 0.0f; }

// x^y exactly when it is a dyadic rational of at most 27 bits: y > 0 an integer or an integer
// plus 1/2 (x then a perfect square), and the odd part of x's significand small enough. These
// are the only inputs whose result can be exactly an f32 rounding midpoint, which the
// float-float path cannot tell from a nearby value; they are rounded here with integers (ties to
// even). Other inputs return false.
__device__ bool pow_exact(float ax, float y, float& out) {
    float y2 = 2.0f * y;
    if (!(y > 0.0f) || y > 32.0f || floorf(y2) != y2) return false;
    uint b = __float_as_uint(ax);
    unsigned long long m;
    int e;
    if ((b >> 23) == 0) { m = b & 0x7fffffu; e = -149; } else { m = (b & 0x7fffffu) | 0x800000u; e = (int)(b >> 23) - 150; }
    int tz = __ffsll((long long)m) - 1;
    m >>= tz;
    e += tz;
    int n = (int)y;
    unsigned long long q = 1;
    int ex = 0;
    if (y != (float)n) {  // n + 1/2: needs √x dyadic
        if (e & 1) return false;
        unsigned long long s = (unsigned long long)rintf(sqrtf((float)m));
        if (s * s != m) return false;
        q = s;
        ex = e / 2;
    }
    int bm = 64 - __clzll((long long)m);
    if ((bm - 1) * n > 27) return false;  // m^n ≥ 2^((bm−1)·n): too many bits for a midpoint
    for (int i = 0; i < n; ++i) {
        q *= m;
        if (q >> 27) return false;
    }
    ex += n * e;
    // Round q·2^ex to f32, ties to even.
    int top = 63 - __clzll((long long)q) + ex;
    if (top > 127) { out = __uint_as_float(0x7f800000u); return true; }
    int lsb = top >= -126 ? top - 23 : -149;
    int sh = lsb - ex;
    if (sh > 0) {
        if (sh > 40) { out = 0.0f; return true; }
        unsigned long long half = (q >> (sh - 1)) & 1ull, rest = q & ((1ull << (sh - 1)) - 1ull);
        q >>= sh;
        if (half && (rest || (q & 1ull))) q += 1;
        ex = lsb;
    }
    // q ≤ 2^24 and q·2^ex representable: exact scaling (2^ex as a normal or subnormal f32).
    float scale1 = ex >= -126 ? __uint_as_float((uint)(127 + (ex >= 0 ? ex / 2 : ex)) << 23) : __uint_as_float(1u << (ex + 149));
    float scale2 = ex >= 0 ? __uint_as_float((uint)(127 + ex - ex / 2) << 23) : 1.0f;
    out = (float)q * scale1 * scale2;
    return true;
}

// C99 powf: the special cases, then exp(y·ln|x|) in float-float with the sign of x^y.
__device__ float pow_f(float x, float y) {
    const float inf = __uint_as_float(0x7f800000u);
    if (y == 0.0f || x == 1.0f) return 1.0f;
    if (x != x || y != y) return x + y;
    float ax = fabsf(x);
    if (fabsf(y) == inf) {
        if (ax == 1.0f) return 1.0f;
        return ((ax < 1.0f) == (y < 0.0f)) ? inf : 0.0f;
    }
    bool odd = is_odd(y);
    if (x == 0.0f) {
        float r = y < 0.0f ? inf : 0.0f;
        return odd ? copysignf(r, x) : r;
    }
    if (ax == inf) {
        float r = y < 0.0f ? 0.0f : inf;
        return (x < 0.0f && odd) ? -r : r;
    }
    if (x < 0.0f && !is_int(y)) return __uint_as_float(0x7fc00000u);
    float ex;
    if (pow_exact(ax, y, ex)) return (x < 0.0f && odd) ? -ex : ex;
    float lh, ll;
    log_ff(ax, lh, ll);
    float zh, ze;
    two_prod(y, lh, zh, ze);
    if (fabsf(zh) > 200.0f) return exp_ff(zh, 0.0f) * ((x < 0.0f && odd) ? -1.0f : 1.0f);  // over/underflow
    float zl = ze + y * ll;
    fast_two_sum(zh, zl, zh, zl);
    float r = exp_ff(zh, zl);
    return (x < 0.0f && odd) ? -r : r;
}


// The stable sigmoid in f32, as CpuRef's: 1 / (1 + e^−x) for x ≥ 0, e^x / (1 + e^x) below.
__device__ __forceinline__ float sigmoid_std(float a) {
    if (a >= 0.0f) {
        return 1.0f / (1.0f + exp_f(-a));
    }
    float e = exp_f(a);
    return e / (1.0f + e);
}

extern "C" __global__ void unary_f32(const float* __restrict__ x, float* __restrict__ y, Strided p, ScalarOp u) {
    uint i = TID;
    if (i >= p.n) return;
    float a = x[off_of(i, p, p.sa, p.off_a)];
    float s = u.s;
    float r;
    switch (u.op) {
        case 0: r = -a; break;
        case 1: r = exp_f(a); break;
        case 2: r = log_f(a); break;
        case 3: r = sqrtf(a); break;
        case 4: r = 1.0f / a; break;
        case 5: r = fabsf(a); break;
        case 6: r = (a == 0.0f) ? 0.0f : (signbit(a) ? -1.0f : 1.0f); break;
        case 7: r = sigmoid_std(a); break;
        case 8: r = a + s; break;
        case 9: r = a - s; break;
        case 10: r = a * s; break;
        case 11: r = a / s; break;
        default: r = pow_f(a, s); break;
    }
    y[i] = r;
}

extern "C" __global__ void compare_f32(const float* __restrict__ x, uchar* __restrict__ y, Strided p, ScalarOp u) {
    uint i = TID;
    if (i >= p.n) return;
    float a = x[off_of(i, p, p.sa, p.off_a)];
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

extern "C" __global__ void binary_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, Strided p, uint op) {
    uint i = TID;
    if (i >= p.n) return;
    float x = a[off_of(i, p, p.sa, p.off_a)];
    float y = b[off_of(i, p, p.sb, p.off_b)];
    float r;
    switch (op) {
        case 0: r = x + y; break;
        case 1: r = x - y; break;
        case 2: r = x * y; break;
        default: r = x / y; break;
    }
    c[i] = r;
}

extern "C" __global__ void mask_fill_f32(const float* __restrict__ x, const uchar* __restrict__ m, float* __restrict__ y, Strided p, float value) {
    uint i = TID;
    if (i >= p.n) return;
    float a = x[off_of(i, p, p.sa, p.off_a)];
    y[i] = m[off_of(i, p, p.sb, p.off_b)] != 0 ? value : a;
}

// ---------------------------------------------------------------- reductions
// One thread per output, CpuRef's order exactly (ndarray's sum_axis).

struct Lanes { uint outer, n, inner, m; };

// ndarray's unrolled_fold with + from zero: eight partial sums, combined as (p0+p4), (p1+p5),
// (p2+p6), (p3+p7), then the tail in order.
__device__ __forceinline__ float unrolled_sum(const float* x, uint len) {
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

extern "C" __global__ void sum_last_f32(const float* __restrict__ x, float* __restrict__ y, Lanes p) {
    uint lane = TID;
    if (lane >= p.outer) return;
    y[lane] = unrolled_sum(x + (ulong)lane * p.n, p.n);
}

extern "C" __global__ void sum_mid_f32(const float* __restrict__ x, float* __restrict__ y, Lanes p) {
    uint t = TID;
    if (t >= p.outer * p.inner) return;
    uint o = t / p.inner, i = t % p.inner;
    float acc = 0.0f;
    for (uint k = 0; k < p.n; ++k) acc = acc + x[((ulong)o * p.n + k) * p.inner + i];
    y[t] = acc;
}

extern "C" __global__ void sum_all_f32(const float* __restrict__ x, float* __restrict__ y, uint n) {
    if (TID != 0) return;
    y[0] = unrolled_sum(x, n);
}

// The fold `if a is NaN or a > b { a } else { b }` from the first element: NaN-propagating, as
// CpuRef's max.
extern "C" __global__ void max_all_f32(const float* __restrict__ x, float* __restrict__ y, uint n) {
    if (TID != 0) return;
    float a = x[0];
    for (uint j = 1; j < n; ++j) { float b = x[j]; a = (isnan(a) || a > b) ? a : b; }
    y[0] = a;
}

// The first maximum along the lane, its index and value, NaN-propagating as CpuRef's argmax:
// the first NaN is the maximum; otherwise a strictly greater value replaces.
extern "C" __global__ void argmax_dim_f32(const float* __restrict__ x, uint* __restrict__ idx, float* __restrict__ val, Lanes p) {
    uint t = TID;
    if (t >= p.outer * p.inner) return;
    uint o = t / p.inner, i = t % p.inner;
    const float* lane = x + (ulong)o * p.n * p.inner + i;
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

// ---------------------------------------------------------------- indexing
// Deterministic scatter and index adds: one thread per target, contributions in ascending source
// order (CpuRef's order), no atomics.

extern "C" __global__ void gather_f32(const float* __restrict__ x, const uint* __restrict__ idx, float* __restrict__ y, Lanes p) {
    uint t = TID;
    if (t >= p.outer * p.m * p.inner) return;
    uint i = t % p.inner, o = t / (p.m * p.inner);
    y[t] = x[((ulong)o * p.n + idx[t]) * p.inner + i];
}

extern "C" __global__ void index_select_f32(const float* __restrict__ x, const uint* __restrict__ idx, float* __restrict__ y, Lanes p) {
    uint t = TID;
    if (t >= p.outer * p.m * p.inner) return;
    uint i = t % p.inner, r = t / p.inner, k = r % p.m, o = r / p.m;
    y[t] = x[((ulong)o * p.n + idx[k]) * p.inner + i];
}

extern "C" __global__ void one_hot_f32(const uint* __restrict__ idx, float* __restrict__ y, Lanes p) {
    uint t = TID;
    if (t >= p.outer * p.n) return;
    y[t] = (idx[t / p.n] == t % p.n) ? 1.0f : 0.0f;
}

extern "C" __global__ void scatter_add_f32(const uint* __restrict__ idx, const float* __restrict__ v, float* __restrict__ y, Lanes p) {
    uint t = TID;
    if (t >= p.outer * p.n * p.inner) return;
    uint i = t % p.inner, r = t / p.inner, j = r % p.n, o = r / p.n;
    float acc = 0.0f;
    for (uint k = 0; k < p.m; ++k) {
        ulong q = ((ulong)o * p.m + k) * p.inner + i;
        if (idx[q] == j) acc = acc + v[q];
    }
    y[t] = acc;
}

extern "C" __global__ void index_add_f32(const uint* __restrict__ idx, const float* __restrict__ v, float* __restrict__ y, Lanes p) {
    uint t = TID;
    if (t >= p.outer * p.n * p.inner) return;
    uint i = t % p.inner, r = t / p.inner, j = r % p.n, o = r / p.n;
    float acc = 0.0f;
    for (uint k = 0; k < p.m; ++k) {
        if (idx[k] == j) acc = acc + v[((ulong)o * p.m + k) * p.inner + i];
    }
    y[t] = acc;
}

// ---------------------------------------------------------------- sort
// A descending sort of each lane of [outer, n, inner] by f32's total order (Rust's total_cmp:
// the bits mapped to an unsigned key), ties by the first index: a bitonic network over the next
// power of two (padding sorts last), one block per lane, keys and indices in shared memory. The
// order is total, so the result is unique and deterministic.

__device__ __forceinline__ uint total_key(float v) {
    // Every NaN, whatever its sign and payload, is one key above +inf (CpuRef's sort_order;
    // an earlier key put a sign-bit NaN below −inf, and 0xFFFFFFFF collided with the padding key 0).
    if (isnan(v)) return 0xffffffffu;
    uint b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

extern "C" __global__ void sort_desc_f32(const float* __restrict__ x, float* __restrict__ vals, uint* __restrict__ idx, Lanes p, uint np2) {
    extern __shared__ uint sh[];
    uint* key = sh;
    uint* id = sh + np2;
    const uint lane = blockIdx.x;
    const uint o = lane / p.inner, i = lane % p.inner;
    for (uint d = threadIdx.x; d < np2; d += blockDim.x) {
        if (d < p.n) {
            key[d] = total_key(x[((ulong)o * p.n + d) * p.inner + i]);
            id[d] = d;
        } else {
            key[d] = 0;
            id[d] = 0xffffffffu;
        }
    }
    __syncthreads();
    for (uint k = 2; k <= np2; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint t = threadIdx.x; t < np2; t += blockDim.x) {
                uint l = t ^ j;
                if (l > t) {
                    uint ka = key[t], kb = key[l], ia = id[t], ib = id[l];
                    // Whether t's element comes first in the final (descending, first-index) order.
                    bool a_first = ka > kb || (ka == kb && ia < ib);
                    bool up = (t & k) == 0;
                    if (up ? !a_first : a_first) {
                        key[t] = kb; key[l] = ka;
                        id[t] = ib; id[l] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (uint d = threadIdx.x; d < p.n; d += blockDim.x) {
        uint src = id[d];
        ulong at = ((ulong)o * p.n + d) * p.inner + i;
        vals[at] = x[((ulong)o * p.n + src) * p.inner + i];
        idx[at] = src;
    }
}

// ---------------------------------------------------------------- fill

extern "C" __global__ void fill_f32(float* __restrict__ y, float value, uint n) {
    uint i = TID;
    if (i < n) y[i] = value;
}

// ---------------------------------------------------------------- fused Adam
// nn::Adam's plain update (no lazy rows, per-seed rates or multipliers), the same f32 operations
// in the same order as its op sequence (each rounded; no contraction with --fmad=false):
// g' = g + p·wd (flag 1); m' = m·β1 + g'·(1−β1); v' = v·β2 + (g'·g')·(1−β2);
// u = (m'/bc1) / (sqrt(v'/bc2) + eps); p' = (p·pre (flag 2)) − u·lr; p' = p'·post (flag 4).

struct AdamScalars { float wd, b1, omb1, b2, omb2, bc1, bc2, eps, lr, pre, post; uint flags; };

extern "C" __global__ void adam_f32(const float* __restrict__ p, const float* __restrict__ g, const float* __restrict__ m, const float* __restrict__ v,
        float* __restrict__ po, float* __restrict__ mo, float* __restrict__ vo, AdamScalars s, uint n) {
    uint i = TID;
    if (i >= n) return;
    float pi = p[i], gi = g[i];
    if (s.flags & 1u) gi = gi + pi * s.wd;
    float mn = m[i] * s.b1 + gi * s.omb1;
    float vn = v[i] * s.b2 + (gi * gi) * s.omb2;
    float mh = mn / s.bc1;
    float vh = vn / s.bc2;
    float u = mh / (sqrtf(vh) + s.eps);
    if (s.flags & 2u) pi = pi * s.pre;
    float pn = pi - u * s.lr;
    if (s.flags & 4u) pn = pn * s.post;
    po[i] = pn;
    mo[i] = mn;
    vo[i] = vn;
}
"#;
