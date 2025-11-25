// Copyright 2016 - 2023 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::kernel::GemmKernel;
use crate::kernel::GemmSelect;
use crate::kernel::{U4, U8};
use crate::archparam;

#[cfg(target_arch="x86")]
use core::arch::x86::*;
#[cfg(target_arch="x86_64")]
use core::arch::x86_64::*;

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelAvx2;
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelSse41;

#[cfg(target_arch="aarch64")]
struct KernelNeon;

struct KernelFallback;

type T = i32;

/// Detect which implementation to use and select it using the selector's
/// .select(Kernel) method.
///
/// This function is called one or more times during a whole program's
/// execution, it may be called for each gemm kernel invocation or fewer times.
#[inline]
pub(crate) fn detect<G>(selector: G) where G: GemmSelect<T> {
    #[cfg(any(target_arch="x86", target_arch="x86_64"))]
    {
        if is_x86_feature_detected_!("avx2") {
            return selector.select(KernelAvx2);
        }
        if is_x86_feature_detected_!("sse4.1") {
            return selector.select(KernelSse41);
        }
    }
    #[cfg(target_arch="aarch64")]
    {
        if is_aarch64_feature_detected_!("neon") {
            return selector.select(KernelNeon);
        }
    }
    return selector.select(KernelFallback);
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
impl GemmKernel for KernelAvx2 {
    type Elem = T;

    type MRTy = U8;
    type NRTy = U8;

    #[inline(always)]
    fn align_to() -> usize { 32 }

    #[inline(always)]
    fn always_masked() -> bool { false }

    #[inline(always)]
    fn nc() -> usize { archparam::S_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::S_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::S_MC }

    #[inline(always)]
    unsafe fn kernel(
        k: usize,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *mut T, rsc: isize, csc: isize) {
        kernel_target_avx2(k, alpha, a, b, beta, c, rsc, csc)
    }
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
impl GemmKernel for KernelSse41 {
    type Elem = T;

    type MRTy = U4;
    type NRTy = U4;

    #[inline(always)]
    fn align_to() -> usize { 16 }

    #[inline(always)]
    fn always_masked() -> bool { false }

    #[inline(always)]
    fn nc() -> usize { archparam::S_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::S_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::S_MC }

    #[inline(always)]
    unsafe fn kernel(
        k: usize,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *mut T, rsc: isize, csc: isize) {
        kernel_target_sse41(k, alpha, a, b, beta, c, rsc, csc)
    }
}

#[cfg(target_arch="aarch64")]
impl GemmKernel for KernelNeon {
    type Elem = T;

    type MRTy = U4;
    type NRTy = U4;

    #[inline(always)]
    fn align_to() -> usize { 16 }

    #[inline(always)]
    fn always_masked() -> bool { false }

    #[inline(always)]
    fn nc() -> usize { archparam::S_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::S_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::S_MC }

    #[inline(always)]
    unsafe fn kernel(
        k: usize,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *mut T, rsc: isize, csc: isize) {
        kernel_target_neon(k, alpha, a, b, beta, c, rsc, csc)
    }
}

impl GemmKernel for KernelFallback {
    type Elem = T;

    type MRTy = U8;
    type NRTy = U4;

    #[inline(always)]
    fn align_to() -> usize { 0 }

    #[inline(always)]
    fn always_masked() -> bool { true }

    #[inline(always)]
    fn nc() -> usize { archparam::S_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::S_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::S_MC }

    #[inline(always)]
    unsafe fn kernel(
        k: usize,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *mut T, rsc: isize, csc: isize) {
        kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
    }
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="avx2")]
unsafe fn kernel_target_avx2(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    const MR: usize = KernelAvx2::MR;
    const NR: usize = KernelAvx2::NR;

    let mut ab = [_mm256_setzero_si256(); MR];
    let mut a = a;
    let mut b = b;

    // Compute A B
    //
    // We compute ab[i][j] += a[i] * b[j]
    //
    // ab[i] holds the row i of the accumulator block (size NR)
    //
    // In each step of k:
    // 1. Load b as a vector (size NR)
    // 2. For each row i in 0..MR:
    //    Broadcast a[i] to a vector
    //    Multiply broadcasted a[i] with b vector
    //    Accumulate into ab[i]
    unroll_by!(4 => k, {
        let bv = _mm256_loadu_si256(b as *const _);
        
        ab[0] = _mm256_add_epi32(ab[0], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 0)), bv));
        ab[1] = _mm256_add_epi32(ab[1], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 1)), bv));
        ab[2] = _mm256_add_epi32(ab[2], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 2)), bv));
        ab[3] = _mm256_add_epi32(ab[3], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 3)), bv));
        ab[4] = _mm256_add_epi32(ab[4], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 4)), bv));
        ab[5] = _mm256_add_epi32(ab[5], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 5)), bv));
        ab[6] = _mm256_add_epi32(ab[6], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 6)), bv));
        ab[7] = _mm256_add_epi32(ab[7], _mm256_mullo_epi32(_mm256_set1_epi32(at(a, 7)), bv));

        a = a.add(MR);
        b = b.add(NR);
    });

    let alphav = _mm256_set1_epi32(alpha);
    let betav = _mm256_set1_epi32(beta);

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    if alpha != 1 {
        for i in 0..MR {
            ab[i] = _mm256_mullo_epi32(alphav, ab[i]);
        }
    }


    if beta != 0 {
        if csc == 1 {
            for i in 0..MR {
                let c_ptr = c![i, 0];
                let cv = _mm256_loadu_si256(c_ptr as *const _);
                let cv = _mm256_mullo_epi32(cv, betav);
                ab[i] = _mm256_add_epi32(ab[i], cv);
            }
        } else {
            for i in 0..MR {
                let cv = _mm256_setr_epi32(
                    *c![i, 0], *c![i, 1], *c![i, 2], *c![i, 3],
                    *c![i, 4], *c![i, 5], *c![i, 6], *c![i, 7]
                );
                let cv = _mm256_mullo_epi32(cv, betav);
                ab[i] = _mm256_add_epi32(ab[i], cv);
            }
        }
    }

    if csc == 1 {
        for i in 0..MR {
            _mm256_storeu_si256(c![i, 0] as *mut _, ab[i]);
        }
    } else {
        for i in 0..MR {
            *c![i, 0] = _mm256_extract_epi32(ab[i], 0);
            *c![i, 1] = _mm256_extract_epi32(ab[i], 1);
            *c![i, 2] = _mm256_extract_epi32(ab[i], 2);
            *c![i, 3] = _mm256_extract_epi32(ab[i], 3);
            *c![i, 4] = _mm256_extract_epi32(ab[i], 4);
            *c![i, 5] = _mm256_extract_epi32(ab[i], 5);
            *c![i, 6] = _mm256_extract_epi32(ab[i], 6);
            *c![i, 7] = _mm256_extract_epi32(ab[i], 7);
        }
    }
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="sse4.1")]
unsafe fn kernel_target_sse41(k: usize, alpha: T, a: *const T, b: *const T,
                              beta: T, c: *mut T, rsc: isize, csc: isize)
{
    const MR: usize = KernelSse41::MR;
    const NR: usize = KernelSse41::NR;

    let mut ab = [_mm_setzero_si128(); MR];
    let mut a = a;
    let mut b = b;

    // Compute A B
    //
    // We compute ab[i][j] += a[i] * b[j]
    //
    // ab[i] holds the row i of the accumulator block (size NR)
    //
    // In each step of k:
    // 1. Load b as a vector (size NR)
    // 2. For each row i in 0..MR:
    //    Broadcast a[i] to a vector
    //    Multiply broadcasted a[i] with b vector
    //    Accumulate into ab[i]
    unroll_by!(4 => k, {
        let bv = _mm_loadu_si128(b as *const _);
        
        ab[0] = _mm_add_epi32(ab[0], _mm_mullo_epi32(_mm_set1_epi32(at(a, 0)), bv));
        ab[1] = _mm_add_epi32(ab[1], _mm_mullo_epi32(_mm_set1_epi32(at(a, 1)), bv));
        ab[2] = _mm_add_epi32(ab[2], _mm_mullo_epi32(_mm_set1_epi32(at(a, 2)), bv));
        ab[3] = _mm_add_epi32(ab[3], _mm_mullo_epi32(_mm_set1_epi32(at(a, 3)), bv));

        a = a.add(MR);
        b = b.add(NR);
    });

    let alphav = _mm_set1_epi32(alpha);
    let betav = _mm_set1_epi32(beta);

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    if alpha != 1 {
        for i in 0..MR {
            ab[i] = _mm_mullo_epi32(alphav, ab[i]);
        }
    }

    if beta != 0 {
        if csc == 1 {
            for i in 0..MR {
                let c_ptr = c![i, 0];
                let cv = _mm_loadu_si128(c_ptr as *const _);
                let cv = _mm_mullo_epi32(cv, betav);
                ab[i] = _mm_add_epi32(ab[i], cv);
            }
        } else {
            for i in 0..MR {
                let cv = _mm_setr_epi32(
                    *c![i, 0], *c![i, 1], *c![i, 2], *c![i, 3]
                );
                let cv = _mm_mullo_epi32(cv, betav);
                ab[i] = _mm_add_epi32(ab[i], cv);
            }
        }
    }

    if csc == 1 {
        for i in 0..MR {
            _mm_storeu_si128(c![i, 0] as *mut _, ab[i]);
        }
    } else {
        for i in 0..MR {
            *c![i, 0] = _mm_extract_epi32(ab[i], 0);
            *c![i, 1] = _mm_extract_epi32(ab[i], 1);
            *c![i, 2] = _mm_extract_epi32(ab[i], 2);
            *c![i, 3] = _mm_extract_epi32(ab[i], 3);
        }
    }
}

#[cfg(target_arch="aarch64")]
#[target_feature(enable="neon")]
unsafe fn kernel_target_neon(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    use core::arch::aarch64::*;
    const MR: usize = KernelNeon::MR;
    const NR: usize = KernelNeon::NR;

    let mut ab = [vdupq_n_s32(0); MR];
    let mut a = a;
    let mut b = b;

    // Compute A B
    //
    // We compute ab[i][j] += a[i] * b[j]
    //
    // ab[i] holds the row i of the accumulator block (size NR)
    //
    // In each step of k:
    // 1. Load b as a vector (size NR)
    // 2. For each row i in 0..MR:
    //    Load a[i] (scalar)
    //    Multiply-accumulate a[i] * b vector into ab[i]
    unroll_by!(4 => k, {
        let bv = vld1q_s32(b);
        let av = vld1q_s32(a);

        ab[0] = vmlaq_n_s32(ab[0], bv, vgetq_lane_s32(av, 0));
        ab[1] = vmlaq_n_s32(ab[1], bv, vgetq_lane_s32(av, 1));
        ab[2] = vmlaq_n_s32(ab[2], bv, vgetq_lane_s32(av, 2));
        ab[3] = vmlaq_n_s32(ab[3], bv, vgetq_lane_s32(av, 3));

        a = a.add(MR);
        b = b.add(NR);
    });

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    if alpha != 1 {
        for i in 0..MR {
            ab[i] = vmulq_n_s32(ab[i], alpha);
        }
    }

    if beta != 0 {
        if csc == 1 {
            for i in 0..MR {
                let c_ptr = c![i, 0];
                let cv = vld1q_s32(c_ptr);
                ab[i] = vmlaq_n_s32(ab[i], cv, beta);
            }
        } else {
            for i in 0..MR {
                let mut cv = vdupq_n_s32(0);
                cv = vld1q_lane_s32(c![i, 0], cv, 0);
                cv = vld1q_lane_s32(c![i, 1], cv, 1);
                cv = vld1q_lane_s32(c![i, 2], cv, 2);
                cv = vld1q_lane_s32(c![i, 3], cv, 3);
                ab[i] = vmlaq_n_s32(ab[i], cv, beta);
            }
        }
    }

    if csc == 1 {
        for i in 0..MR {
            vst1q_s32(c![i, 0], ab[i]);
        }
    } else {
        for i in 0..MR {
            vst1q_lane_s32(c![i, 0], ab[i], 0);
            vst1q_lane_s32(c![i, 1], ab[i], 1);
            vst1q_lane_s32(c![i, 2], ab[i], 2);
            vst1q_lane_s32(c![i, 3], ab[i], 3);
        }
    }
}

#[inline]
unsafe fn kernel_fallback_impl(k: usize, alpha: T, a: *const T, b: *const T,
                               beta: T, c: *mut T, rsc: isize, csc: isize)
{
    const MR: usize = KernelFallback::MR;
    const NR: usize = KernelFallback::NR;
    let mut ab: [[T; NR]; MR] = [[0; NR]; MR];
    let mut a = a;
    let mut b = b;
    debug_assert_eq!(beta, 0, "Beta must be 0 or is not masked");

    // Compute A B into ab[i][j]
    unroll_by!(4 => k, {
        loop8!(i, loop4!(j, ab[i][j] += at(a, i) * at(b, j)));

        a = a.offset(MR as isize);
        b = b.offset(NR as isize);
    });

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // set C = α A B
    loop4!(j, loop8!(i, *c![i, j] = alpha * ab[i][j]));
}

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
    *ptr.offset(i as isize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::gemm_loop;
    use std::vec::Vec;

    // Helper to run gemm with a specific kernel
    fn run_gemm<K: GemmKernel<Elem=i32>>(m: usize, k: usize, n: usize, 
                                         alpha: i32, a: &[i32], b: &[i32], 
                                         beta: i32, c: &mut [i32]) {
        unsafe {
            gemm_loop::<K>(
                m, k, n,
                alpha,
                a.as_ptr(), k as isize, 1,
                b.as_ptr(), n as isize, 1,
                beta,
                c.as_mut_ptr(), n as isize, 1
            );
        }
    }

    #[test]
    fn test_simd_vs_fallback() {
        let sizes = vec![
            (32, 32, 32),
            (1, 1, 1),
            (16, 16, 16),
            (17, 10, 5),
            (4, 4, 4),
            (8, 8, 8),
            (33, 33, 33),
            (6, 6, 6), // Test partial blocks for AVX2 (MR=8) and SSE4.1 (MR=4)
        ];

        for (m, k, n) in sizes {
            let alpha = 2;
            let beta = 3;

            let a: Vec<i32> = (0..m*k).map(|x| (x % 10) as i32).collect();
            let b: Vec<i32> = (0..k*n).map(|x| (x % 10) as i32).collect();
            let mut c_fallback = vec![0; m*n];
            
            // Initialize C with some data to test beta
            for i in 0..m*n { c_fallback[i] = i as i32; }
            
            let c_init = c_fallback.clone();

            // Run fallback
            run_gemm::<KernelFallback>(m, k, n, alpha, &a, &b, beta, &mut c_fallback);

            #[cfg(any(target_arch="x86", target_arch="x86_64"))]
            {
                if is_x86_feature_detected!("avx2") {
                    let mut c_avx2 = c_init.clone();
                    run_gemm::<KernelAvx2>(m, k, n, alpha, &a, &b, beta, &mut c_avx2);
                    assert_eq!(c_fallback, c_avx2, "AVX2 result mismatch for m={}, k={}, n={}", m, k, n);
                }
                
                if is_x86_feature_detected!("sse4.1") {
                    let mut c_sse41 = c_init.clone();
                    run_gemm::<KernelSse41>(m, k, n, alpha, &a, &b, beta, &mut c_sse41);
                    assert_eq!(c_fallback, c_sse41, "SSE4.1 result mismatch for m={}, k={}, n={}", m, k, n);
                }
            }

            #[cfg(target_arch="aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    let mut c_neon = c_init.clone();
                    run_gemm::<KernelNeon>(m, k, n, alpha, &a, &b, beta, &mut c_neon);
                    assert_eq!(c_fallback, c_neon, "NEON result mismatch for m={}, k={}, n={}", m, k, n);
                }
            }
        }
    }
}
