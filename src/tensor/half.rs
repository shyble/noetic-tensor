//! In-house 16-bit floats: IEEE 754 binary16
//! (`F16`) and bfloat16 (`BF16`), stored as their bit patterns, with round-to-nearest-even
//! conversions from f32 and exact conversions to f32. Tensors store them and convert; arithmetic
//! on them runs in f32 (16-bit types accumulate in f32).

/// IEEE 754 half precision (1 sign, 5 exponent, 10 mantissa bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct F16(pub u16);

/// bfloat16 (1 sign, 8 exponent, 7 mantissa bits): the top half of an f32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BF16(pub u16);

impl BF16 {
    /// Round to nearest, ties to even; NaN stays a (quiet) NaN.
    pub fn from_f32(x: f32) -> BF16 {
        let b = x.to_bits();
        if x.is_nan() {
            return BF16(((b >> 16) as u16) | 0x0040);
        }
        let round = 0x7fff + ((b >> 16) & 1);
        BF16((b.wrapping_add(round) >> 16) as u16)
    }

    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl F16 {
    /// Round to nearest, ties to even, with subnormals, overflow to ±∞ and NaN kept.
    pub fn from_f32(x: f32) -> F16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let man = b & 0x7f_ffff;
        if exp == 0xff {
            // ∞ or NaN (keep a quiet NaN).
            return F16(sign | 0x7c00 | if man != 0 { 0x0200 | (man >> 13) as u16 } else { 0 });
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return F16(sign | 0x7c00);
        }
        if e <= 0 {
            // Subnormal half (or zero): shift the full significand into place, then round.
            if e < -10 {
                return F16(sign);
            }
            let m = man | 0x80_0000;
            let shift = (14 - e) as u32;
            let half = m >> shift;
            let rem = m & ((1 << shift) - 1);
            let mid = 1 << (shift - 1);
            let up = rem > mid || (rem == mid && half & 1 == 1);
            return F16(sign | (half + up as u32) as u16);
        }
        let half = ((e as u32) << 10) | (man >> 13);
        let rem = man & 0x1fff;
        let up = rem > 0x1000 || (rem == 0x1000 && half & 1 == 1);
        // A carry out of the mantissa increments the exponent, and at the top gives ∞: both right.
        F16(sign | (half + up as u32) as u16)
    }

    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let man = h & 0x3ff;
        let bits = if exp == 0 {
            if man == 0 {
                sign
            } else {
                // Subnormal: normalise.
                let mut e = 0i32;
                let mut m = man;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (((e + 127 - 14) as u32) << 23) | ((m & 0x3ff) << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f80_0000 | (man << 13)
        } else {
            sign | ((exp + 127 - 15) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }
}
