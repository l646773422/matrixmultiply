// Copyright 2016 - 2023 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rawpointer::PointerExt;

use core::ptr::copy_nonoverlapping;

use crate::kernel::ConstNum;
use crate::kernel::Element;

/// Pack matrix into `pack`
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + pack: packing buffer
/// + a: matrix,
/// + rsa: row stride
/// + csa: column stride
///
/// + MR: kernel rows/columns that we round up to
// If one of pack and a is of a reference type, it gets a noalias annotation which
// gives benefits to optimization. The packing buffer is contiguous so it can be passed as a slice
// here.
pub(crate) unsafe fn pack<MR, T>(kc: usize, mc: usize, pack: &mut [T],
                                 a: *const T, rsa: isize, csa: isize)
    where T: Element,
          MR: ConstNum,
{
    pack_impl::<MR, T>(kc, mc, pack, a, rsa, csa)
}

/// Specialized for AVX2
/// Safety: Requires AVX2
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="avx2")]
pub(crate) unsafe fn pack_avx2<MR, T>(kc: usize, mc: usize, pack: &mut [T],
                                     a: *const T, rsa: isize, csa: isize)
    where T: Element,
          MR: ConstNum,
{
    pack_impl::<MR, T>(kc, mc, pack, a, rsa, csa)
}

#[cfg(target_arch="aarch64")]
pub(crate) trait PackNeon : Element {
    unsafe fn pack_neon_impl<MR>(kc: usize, mc: usize, pack: &mut [Self],
                                 a: *const Self, rsa: isize, csa: isize)
    where MR: ConstNum;
}

#[cfg(target_arch="aarch64")]
impl PackNeon for f32 {
    #[inline(always)]
    unsafe fn pack_neon_impl<MR>(kc: usize, mc: usize, pack: &mut [Self],
                                 a: *const Self, rsa: isize, csa: isize)
    where MR: ConstNum
    {
        let pack = pack.as_mut_ptr();
        let mr = MR::VALUE;
        let mut p = 0; // offset into pack

        // Optimized loop for MR=4 and MR=8
        if mr == 4 || mr == 8 {
            for ir in 0..mc/mr {
                let row_offset = ir * mr;
                for j in 0..kc {
                    let a_ptr = a.stride_offset(rsa, row_offset).stride_offset(csa, j);
                    
                    if rsa == 1 {
                        if mr == 4 {
                            let v = vld1q_f32(a_ptr);
                            vst1q_f32(pack.add(p), v);
                        } else { // mr == 8
                            let v0 = vld1q_f32(a_ptr);
                            let v1 = vld1q_f32(a_ptr.add(4));
                            vst1q_f32(pack.add(p), v0);
                            vst1q_f32(pack.add(p+4), v1);
                        }
                    } else {
                        if mr == 4 {
                            let mut v = vmovq_n_f32(0.);
                            v = vld1q_lane_f32(a_ptr, v, 0);
                            v = vld1q_lane_f32(a_ptr.stride_offset(rsa, 1), v, 1);
                            v = vld1q_lane_f32(a_ptr.stride_offset(rsa, 2), v, 2);
                            v = vld1q_lane_f32(a_ptr.stride_offset(rsa, 3), v, 3);
                            vst1q_f32(pack.add(p), v);
                        } else { // mr == 8
                            let mut v0 = vmovq_n_f32(0.);
                            v0 = vld1q_lane_f32(a_ptr, v0, 0);
                            v0 = vld1q_lane_f32(a_ptr.stride_offset(rsa, 1), v0, 1);
                            v0 = vld1q_lane_f32(a_ptr.stride_offset(rsa, 2), v0, 2);
                            v0 = vld1q_lane_f32(a_ptr.stride_offset(rsa, 3), v0, 3);
                            vst1q_f32(pack.add(p), v0);
                            
                            let mut v1 = vmovq_n_f32(0.);
                            let a_ptr2 = a_ptr.stride_offset(rsa, 4);
                            v1 = vld1q_lane_f32(a_ptr2, v1, 0);
                            v1 = vld1q_lane_f32(a_ptr2.stride_offset(rsa, 1), v1, 1);
                            v1 = vld1q_lane_f32(a_ptr2.stride_offset(rsa, 2), v1, 2);
                            v1 = vld1q_lane_f32(a_ptr2.stride_offset(rsa, 3), v1, 3);
                            vst1q_f32(pack.add(p+4), v1);
                        }
                    }
                    p += mr;
                }
            }
        } else {
            // Fallback for other MR
             for ir in 0..mc/mr {
                let row_offset = ir * mr;
                for j in 0..kc {
                    for i in 0..mr {
                        let a_elt = a.stride_offset(rsa, i + row_offset)
                                     .stride_offset(csa, j);
                        copy_nonoverlapping(a_elt, pack.add(p), 1);
                        p += 1;
                    }
                }
            }
        }

        // Padding (same as generic)
        let zero = <_>::zero();
        let rest = mc % mr;
        if rest > 0 {
            let row_offset = (mc/mr) * mr;
            for j in 0..kc {
                for i in 0..mr {
                    if i < rest {
                        let a_elt = a.stride_offset(rsa, i + row_offset)
                                     .stride_offset(csa, j);
                        copy_nonoverlapping(a_elt, pack.add(p), 1);
                    } else {
                        *pack.add(p) = zero;
                    }
                    p += 1;
                }
            }
        }
    }
}

#[cfg(target_arch="aarch64")]
impl PackNeon for f64 {
    #[inline(always)]
    unsafe fn pack_neon_impl<MR>(kc: usize, mc: usize, pack: &mut [Self],
                                 a: *const Self, rsa: isize, csa: isize)
    where MR: ConstNum
    {
        pack_impl::<MR, Self>(kc, mc, pack, a, rsa, csa)
    }
}

#[cfg(target_arch="aarch64")]
impl PackNeon for i32 {
    #[inline(always)]
    unsafe fn pack_neon_impl<MR>(kc: usize, mc: usize, pack: &mut [Self],
                                 a: *const Self, rsa: isize, csa: isize)
    where MR: ConstNum
    {
        let pack = pack.as_mut_ptr();
        let mr = MR::VALUE;
        let mut p = 0; // offset into pack

        // Optimized loop for MR=4 and MR=8
        if mr == 4 || mr == 8 {
            for ir in 0..mc/mr {
                let row_offset = ir * mr;
                for j in 0..kc {
                    let a_ptr = a.stride_offset(rsa, row_offset).stride_offset(csa, j);
                    
                    if rsa == 1 {
                        if mr == 4 {
                            let v = vld1q_s32(a_ptr);
                            vst1q_s32(pack.add(p), v);
                        } else { // mr == 8
                            let v0 = vld1q_s32(a_ptr);
                            let v1 = vld1q_s32(a_ptr.add(4));
                            vst1q_s32(pack.add(p), v0);
                            vst1q_s32(pack.add(p+4), v1);
                        }
                    } else {
                        if mr == 4 {
                            let mut v = vmovq_n_s32(0);
                            v = vld1q_lane_s32(a_ptr, v, 0);
                            v = vld1q_lane_s32(a_ptr.stride_offset(rsa, 1), v, 1);
                            v = vld1q_lane_s32(a_ptr.stride_offset(rsa, 2), v, 2);
                            v = vld1q_lane_s32(a_ptr.stride_offset(rsa, 3), v, 3);
                            vst1q_s32(pack.add(p), v);
                        } else { // mr == 8
                            let mut v0 = vmovq_n_s32(0);
                            v0 = vld1q_lane_s32(a_ptr, v0, 0);
                            v0 = vld1q_lane_s32(a_ptr.stride_offset(rsa, 1), v0, 1);
                            v0 = vld1q_lane_s32(a_ptr.stride_offset(rsa, 2), v0, 2);
                            v0 = vld1q_lane_s32(a_ptr.stride_offset(rsa, 3), v0, 3);
                            vst1q_s32(pack.add(p), v0);
                            
                            let mut v1 = vmovq_n_s32(0);
                            let a_ptr2 = a_ptr.stride_offset(rsa, 4);
                            v1 = vld1q_lane_s32(a_ptr2, v1, 0);
                            v1 = vld1q_lane_s32(a_ptr2.stride_offset(rsa, 1), v1, 1);
                            v1 = vld1q_lane_s32(a_ptr2.stride_offset(rsa, 2), v1, 2);
                            v1 = vld1q_lane_s32(a_ptr2.stride_offset(rsa, 3), v1, 3);
                            vst1q_s32(pack.add(p+4), v1);
                        }
                    }
                    p += mr;
                }
            }
        } else {
            // Fallback for other MR
             for ir in 0..mc/mr {
                let row_offset = ir * mr;
                for j in 0..kc {
                    for i in 0..mr {
                        let a_elt = a.stride_offset(rsa, i + row_offset)
                                     .stride_offset(csa, j);
                        copy_nonoverlapping(a_elt, pack.add(p), 1);
                        p += 1;
                    }
                }
            }
        }

        // Padding (same as generic)
        let zero = <_>::zero();
        let rest = mc % mr;
        if rest > 0 {
            let row_offset = (mc/mr) * mr;
            for j in 0..kc {
                for i in 0..mr {
                    if i < rest {
                        let a_elt = a.stride_offset(rsa, i + row_offset)
                                     .stride_offset(csa, j);
                        copy_nonoverlapping(a_elt, pack.add(p), 1);
                    } else {
                        *pack.add(p) = zero;
                    }
                    p += 1;
                }
            }
        }
    }
}

#[cfg(all(target_arch="aarch64", feature="cgemm"))]
impl PackNeon for c32 {
    unsafe fn pack_neon_impl<MR>(kc: usize, mc: usize, pack: &mut [Self],
                                 a: *const Self, rsa: isize, csa: isize)
    where MR: ConstNum
    {
        pack_impl::<MR, Self>(kc, mc, pack, a, rsa, csa)
    }
}

#[cfg(all(target_arch="aarch64", feature="cgemm"))]
impl PackNeon for c64 {
    unsafe fn pack_neon_impl<MR>(kc: usize, mc: usize, pack: &mut [Self],
                                 a: *const Self, rsa: isize, csa: isize)
    where MR: ConstNum
    {
        pack_impl::<MR, Self>(kc, mc, pack, a, rsa, csa)
    }
}

/// Specialized for NEON
/// Safety: Requires NEON
#[cfg(target_arch="aarch64")]
#[target_feature(enable="neon")]
pub(crate) unsafe fn pack_neon<MR, T>(kc: usize, mc: usize, pack: &mut [T],
                                     a: *const T, rsa: isize, csa: isize)
    where T: Element + PackNeon,
          MR: ConstNum,
{
    T::pack_neon_impl::<MR>(kc, mc, pack, a, rsa, csa)
}

/// Pack implementation, see pack above for docs.
///
/// Uses inline(always) so that it can be instantiated for different target features.
#[inline(always)]
unsafe fn pack_impl<MR, T>(kc: usize, mc: usize, pack: &mut [T],
                           a: *const T, rsa: isize, csa: isize)
    where T: Element,
          MR: ConstNum,
{
    let pack = pack.as_mut_ptr();
    let mr = MR::VALUE;
    let mut p = 0; // offset into pack

    if rsa == 1 {
        // if the matrix is contiguous in the same direction we are packing,
        // copy a kernel row at a time.
        for ir in 0..mc/mr {
            let row_offset = ir * mr;
            for j in 0..kc {
                let a_row = a.stride_offset(rsa, row_offset)
                             .stride_offset(csa, j);
                copy_nonoverlapping(a_row, pack.add(p), mr);
                p += mr;
            }
        }
    } else {
        // general layout case
        for ir in 0..mc/mr {
            let row_offset = ir * mr;
            for j in 0..kc {
                for i in 0..mr {
                    let a_elt = a.stride_offset(rsa, i + row_offset)
                                 .stride_offset(csa, j);
                    copy_nonoverlapping(a_elt, pack.add(p), 1);
                    p += 1;
                }
            }
        }
    }

    let zero = <_>::zero();

    // Pad with zeros to multiple of kernel size (uneven mc)
    let rest = mc % mr;
    if rest > 0 {
        let row_offset = (mc/mr) * mr;
        for j in 0..kc {
            for i in 0..mr {
                if i < rest {
                    let a_elt = a.stride_offset(rsa, i + row_offset)
                                 .stride_offset(csa, j);
                    copy_nonoverlapping(a_elt, pack.add(p), 1);
                } else {
                    *pack.add(p) = zero;
                }
                p += 1;
            }
        }
    }
}

#[cfg(target_arch="aarch64")]
use core::arch::aarch64::*;

#[cfg(all(target_arch="aarch64", feature="cgemm"))]
use crate::kernel::{c32, c64};

