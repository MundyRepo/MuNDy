// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
//
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_UTILS_SUPPRESS_WARNINGS_HPP_
#define MUNDY_UTILS_SUPPRESS_WARNINGS_HPP_

//==================================================================================================
// Basic pragma helpers
//==================================================================================================

#define MUNDY_PRAGMA_IMPL(x) _Pragma(#x)
#define MUNDY_PRAGMA(x) MUNDY_PRAGMA_IMPL(x)

//==================================================================================================
// Compiler-family push/pop
//==================================================================================================

// MSVC host compiler
#if defined(_MSC_VER)
#define MUNDY_MSVC_DIAG_PUSH __pragma(warning(push))
#define MUNDY_MSVC_DIAG_POP __pragma(warning(pop))
#define MUNDY_MSVC_DIAG_DISABLE(num) __pragma(warning(disable : num))
#else
#define MUNDY_MSVC_DIAG_PUSH
#define MUNDY_MSVC_DIAG_POP
#define MUNDY_MSVC_DIAG_DISABLE(num)
#endif

// Clang family
// Covers:
//   - Clang proper
//   - clang-cuda
//   - HIP-Clang / amdclang++
//   - Intel icx/icpx/dpcpp (Clang frontend)
// ROCm and Intel docs both line up with this model.
#if defined(__clang__)
#define MUNDY_CLANG_DIAG_PUSH MUNDY_PRAGMA(clang diagnostic push)
#define MUNDY_CLANG_DIAG_POP MUNDY_PRAGMA(clang diagnostic pop)
#define MUNDY_CLANG_DIAG_IGNORE(w) MUNDY_PRAGMA(clang diagnostic ignored w)
#else
#define MUNDY_CLANG_DIAG_PUSH
#define MUNDY_CLANG_DIAG_POP
#define MUNDY_CLANG_DIAG_IGNORE(w)
#endif

// GCC family
#if defined(__GNUC__) && !defined(__clang__)
#define MUNDY_GCC_DIAG_PUSH MUNDY_PRAGMA(GCC diagnostic push)
#define MUNDY_GCC_DIAG_POP MUNDY_PRAGMA(GCC diagnostic pop)
#define MUNDY_GCC_DIAG_IGNORE(w) MUNDY_PRAGMA(GCC diagnostic ignored w)
#else
#define MUNDY_GCC_DIAG_PUSH
#define MUNDY_GCC_DIAG_POP
#define MUNDY_GCC_DIAG_IGNORE(w)
#endif

// NVCC and compatible (e.g. NVCC in Clang mode for CUDA/HIP, AMD's hipcc/amdclang++ in CUDA mode)
#if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#define MUNDY_NV_DIAG_PUSH MUNDY_PRAGMA(nv_diagnostic push)
#define MUNDY_NV_DIAG_POP MUNDY_PRAGMA(nv_diagnostic pop)
#define MUNDY_NV_DIAG_SUPPRESS(num) MUNDY_PRAGMA(nv_diag_suppress num)
#else
#define MUNDY_NV_DIAG_PUSH
#define MUNDY_NV_DIAG_POP
#define MUNDY_NV_DIAG_SUPPRESS(num)
#endif

//==================================================================================================
// Convenience regions
//==================================================================================================

// Generic push/pop for "whatever compiler is driving this translation unit"
#define MUNDY_DIAG_PUSH \
  MUNDY_MSVC_DIAG_PUSH  \
  MUNDY_CLANG_DIAG_PUSH \
  MUNDY_GCC_DIAG_PUSH   \
  MUNDY_NV_DIAG_PUSH

#define MUNDY_DIAG_POP \
  MUNDY_NV_DIAG_POP    \
  MUNDY_GCC_DIAG_POP   \
  MUNDY_CLANG_DIAG_POP \
  MUNDY_MSVC_DIAG_POP

// Suppress the CUDA/HIP/SYCL-ish cross-space call warning family.
//
// Notes:
// - NVCC numeric warning IDs are NVCC-specific.
// - Clang-family warnings use named groups, but the exact warning group can vary
//   by frontend/version. -Wcudacall-from-host is the important CUDA one.
// - HIP and SYCL often surface similar issues through Clang-style warnings, but
//   not always under the same warning group.
//
// This macro is intentionally conservative: it suppresses what is known, and
// leaves room for you to add site-local extra ignores.
#define MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH \
  MUNDY_DIAG_PUSH                                       \
  /* NVCC */                                            \
  MUNDY_NV_DIAG_SUPPRESS(20011)                         \
  MUNDY_NV_DIAG_SUPPRESS(20013)                         \
  MUNDY_NV_DIAG_SUPPRESS(20014)                         \
  MUNDY_NV_DIAG_SUPPRESS(20015)                         \
  /* clang-cuda */                                      \
  MUNDY_CLANG_DIAG_IGNORE("-Wcudacall-from-host")

#define MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_POP MUNDY_DIAG_POP

#endif  // MUNDY_UTILS_SUPPRESS_WARNINGS_HPP_
