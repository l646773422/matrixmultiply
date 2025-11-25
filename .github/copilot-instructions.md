# Copilot Instructions for matrixmultiply

This repository implements general matrix multiplication (GEMM) for f32, f64, and complex matrices in Rust. It uses a BLIS-like microkernel approach for high performance and portability.

## Architectural Overview

- **Microkernel Strategy**: The core logic splits matrix multiplication into smaller blocks to fit in CPU caches. The innermost loop uses architecture-specific microkernels (AVX, FMA, NEON) or a generic fallback.
- **Entry Points**: `sgemm`, `dgemm`, `cgemm`, `zgemm` in `src/lib.rs` are the public APIs. They delegate to `gemm_loop` in `src/gemm.rs`.
- **Kernel Abstraction**: The `GemmKernel` trait (`src/kernel.rs`) defines the interface for microkernels. Implementations must specify register blocking factors (`MR`, `NR`) and alignment requirements.
- **Feature Detection**: Runtime CPU feature detection selects the best kernel at runtime (e.g., `src/dgemm_kernel.rs` uses `detect` to choose between AVX, FMA, or fallback).
- **Packing**: Matrices are packed into contiguous buffers (`src/packing.rs`) to optimize memory access patterns for the kernels.

## Key Files & Directories

- `src/gemm.rs`: Implements the high-level loops (packing, threading, and calling kernels).
- `src/kernel.rs`: Defines the `GemmKernel` and `Element` traits.
- `src/*gemm_kernel.rs`: Contains kernel selection logic and fallback implementations.
- `src/x86/` & `src/aarch64/`: Architecture-specific SIMD microkernels.
- `examples/benchmark.rs`: The primary benchmarking tool.
- `benches/benchmarks.rs`: Cargo benchmarks for regression testing.

## Developer Workflows

### Building
- **Standard**: `cargo build`
- **No Std**: `cargo build --no-default-features` (ensure code remains `no_std` compatible).
- **Threading**: `cargo build --features threading` (enabled by default in `Cargo.toml` via `std` usually, but check feature flags).

### Benchmarking (Critical)
Performance is the primary goal. Always verify changes with benchmarks.
- **Quick Check**: `cargo bench`
- **Comprehensive**: Use the benchmark example for custom sizes and CSV output:
  ```bash
  cargo run --release --example benchmark -- --n 1024 --csv
  ```
- **Parameter Sweeps**: Use `benches/benchloop.py` to run ranges of benchmarks.

### Testing
- **Correctness**: `cargo test` covers various matrix layouts (row-major, col-major, arbitrary strides).
- **Layouts**: Tests must ensure correctness for non-contiguous strides and edge cases (e.g., `k=0`).

## Coding Conventions

- **Unsafe Code**: This crate relies heavily on `unsafe` for pointer arithmetic and SIMD.
  - Use `rawpointer` methods for pointer offsets.
  - Validate pointers and strides before unsafe blocks where possible.
  - Ensure `MaskBuffer` and packing buffers are correctly aligned.
- **Performance**:
  - **No Allocations in Loops**: Memory for packing is allocated once per GEMM call (or per thread).
  - **Inlining**: Use `#[inline(always)]` for small helper functions used in inner loops.
- **Portability**:
  - Code must compile on stable Rust.
  - Architecture-specific code must be guarded by `#[cfg(target_arch = "...")]` and runtime feature detection.
- **Formatting**: Follow existing patterns.

## Common Patterns

### Kernel Selection
When adding a new kernel:
1. Implement `GemmKernel` for the new type.
2. Update the `detect` function in the corresponding `*gemm_kernel.rs` to check for the required CPU feature.
3. Ensure a fallback exists.

### Packing
Packing functions (`pack_mr`, `pack_nr`) rearrange data into `KC x MC` or `KC x NC` blocks. If a kernel requires a specific layout, override the default packing methods in the `GemmKernel` implementation.
