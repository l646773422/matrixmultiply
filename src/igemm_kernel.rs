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

struct KernelFallback;

type T = i32;

/// Detect which implementation to use and select it using the selector's
/// .select(Kernel) method.
#[inline]
pub(crate) fn detect<G>(selector: G) where G: GemmSelect<T> {
    return selector.select(KernelFallback);
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
