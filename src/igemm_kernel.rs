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

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    if alpha != 1 {
        let alphav = _mm256_set1_epi32(alpha);
        for i in 0..MR {
            ab[i] = _mm256_mullo_epi32(alphav, ab[i]);
        }
    }

    if beta != 0 {
        let betav = _mm256_set1_epi32(beta);
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

    let (mut a, mut b, rsc, csc) = if rsc == 1 { (b, a, csc, rsc) } else { (a, b, rsc, csc) };

    // Kernel 8 x 8 (a x b)
    // Four quadrants of 4 x 4
    let mut ab11 = [vdupq_n_s32(0); 4];
    let mut ab12 = [vdupq_n_s32(0); 4];
    let mut ab21 = [vdupq_n_s32(0); 4];
    let mut ab22 = [vdupq_n_s32(0); 4];

    // Compute
    // ab_ij = a_i * b_j for all i, j
    macro_rules! ab_ij_equals_ai_bj {
        ($dest:ident, $av:expr, $bv:expr) => {
            $dest[0] = vmlaq_laneq_s32($dest[0], $bv, $av, 0);
            $dest[1] = vmlaq_laneq_s32($dest[1], $bv, $av, 1);
            $dest[2] = vmlaq_laneq_s32($dest[2], $bv, $av, 2);
            $dest[3] = vmlaq_laneq_s32($dest[3], $bv, $av, 3);
        }
    }

    for _ in 0..k {
        let a1 = vld1q_s32(a);
        let b1 = vld1q_s32(b);
        let a2 = vld1q_s32(a.add(4));
        let b2 = vld1q_s32(b.add(4));

        // compute an outer product ab = a (*) b in four quadrants ab11, ab12, ab21, ab22

        // ab11: [a1 a2 a3 a4] (*) [b1 b2 b3 b4]
        // ab11: a1b1 a1b2 a1b3 a1b4
        //       a2b1 a2b2 a2b3 a2b4
        //       a3b1 a3b2 a3b3 a3b4
        //       a4b1 a4b2 a4b3 a4b4
        //  etc
        ab_ij_equals_ai_bj!(ab11, a1, b1);
        ab_ij_equals_ai_bj!(ab12, a1, b2);
        ab_ij_equals_ai_bj!(ab21, a2, b1);
        ab_ij_equals_ai_bj!(ab22, a2, b2);

        a = a.add(MR);
        b = b.add(NR);
    }

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // ab *= alpha
    if alpha != 1 {
        loop4!(i, ab11[i] = vmulq_n_s32(ab11[i], alpha));
        loop4!(i, ab12[i] = vmulq_n_s32(ab12[i], alpha));
        loop4!(i, ab21[i] = vmulq_n_s32(ab21[i], alpha));
        loop4!(i, ab22[i] = vmulq_n_s32(ab22[i], alpha));
    }

    // load one int32x4_t from four pointers
    macro_rules! loadq_from_pointers {
        ($p0:expr, $p1:expr, $p2:expr, $p3:expr) => (
            {
                let v = vld1q_dup_s32($p0);
                let v = vld1q_lane_s32($p1, v, 1);
                let v = vld1q_lane_s32($p2, v, 2);
                let v = vld1q_lane_s32($p3, v, 3);
                v
            }
        );
    }

    if beta != 0 {
        // load existing value in C
        let mut c11 = [vdupq_n_s32(0); 4];
        let mut c12 = [vdupq_n_s32(0); 4];
        let mut c21 = [vdupq_n_s32(0); 4];
        let mut c22 = [vdupq_n_s32(0); 4];

        if csc == 1 {
            loop4!(i, c11[i] = vld1q_s32(c![i + 0, 0]));
            loop4!(i, c12[i] = vld1q_s32(c![i + 0, 4]));
            loop4!(i, c21[i] = vld1q_s32(c![i + 4, 0]));
            loop4!(i, c22[i] = vld1q_s32(c![i + 4, 4]));
        } else {
            loop4!(i, c11[i] = loadq_from_pointers!(c![i + 0, 0], c![i + 0, 1], c![i + 0, 2], c![i + 0, 3]));
            loop4!(i, c12[i] = loadq_from_pointers!(c![i + 0, 4], c![i + 0, 5], c![i + 0, 6], c![i + 0, 7]));
            loop4!(i, c21[i] = loadq_from_pointers!(c![i + 4, 0], c![i + 4, 1], c![i + 4, 2], c![i + 4, 3]));
            loop4!(i, c22[i] = loadq_from_pointers!(c![i + 4, 4], c![i + 4, 5], c![i + 4, 6], c![i + 4, 7]));
        }

        let betav = vdupq_n_s32(beta);

        // ab += β C
        loop4!(i, ab11[i] = vmlaq_s32(ab11[i], c11[i], betav));
        loop4!(i, ab12[i] = vmlaq_s32(ab12[i], c12[i], betav));
        loop4!(i, ab21[i] = vmlaq_s32(ab21[i], c21[i], betav));
        loop4!(i, ab22[i] = vmlaq_s32(ab22[i], c22[i], betav));
    }

    // c <- ab
    // which is in full
    //   C <- α A B (+ β C)
    if csc == 1 {
        loop4!(i, vst1q_s32(c![i + 0, 0], ab11[i]));
        loop4!(i, vst1q_s32(c![i + 0, 4], ab12[i]));
        loop4!(i, vst1q_s32(c![i + 4, 0], ab21[i]));
        loop4!(i, vst1q_s32(c![i + 4, 4], ab22[i]));
    } else {
        loop4!(i, vst1q_lane_s32(c![i + 0, 0], ab11[i], 0));
        loop4!(i, vst1q_lane_s32(c![i + 0, 1], ab11[i], 1));
        loop4!(i, vst1q_lane_s32(c![i + 0, 2], ab11[i], 2));
        loop4!(i, vst1q_lane_s32(c![i + 0, 3], ab11[i], 3));

        loop4!(i, vst1q_lane_s32(c![i + 0, 4], ab12[i], 0));
        loop4!(i, vst1q_lane_s32(c![i + 0, 5], ab12[i], 1));
        loop4!(i, vst1q_lane_s32(c![i + 0, 6], ab12[i], 2));
        loop4!(i, vst1q_lane_s32(c![i + 0, 7], ab12[i], 3));

        loop4!(i, vst1q_lane_s32(c![i + 4, 0], ab21[i], 0));
        loop4!(i, vst1q_lane_s32(c![i + 4, 1], ab21[i], 1));
        loop4!(i, vst1q_lane_s32(c![i + 4, 2], ab21[i], 2));
        loop4!(i, vst1q_lane_s32(c![i + 4, 3], ab21[i], 3));

        loop4!(i, vst1q_lane_s32(c![i + 4, 4], ab22[i], 0));
        loop4!(i, vst1q_lane_s32(c![i + 4, 5], ab22[i], 1));
        loop4!(i, vst1q_lane_s32(c![i + 4, 6], ab22[i], 2));
        loop4!(i, vst1q_lane_s32(c![i + 4, 7], ab22[i], 3));
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
