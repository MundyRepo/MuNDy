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

#ifndef MUNDY_MATH_INVERT_HPP_
#define MUNDY_MATH_INVERT_HPP_

/// \file invert.hpp
/// \brief Invert a dense square matrix held in a Kokkos view, by LU factorization with partial pivoting.
///
/// invert is the Kokkos-view counterpart of inverse(mat) for AMatrix. It solves with KokkosLapack::gesv, so it needs
/// the KokkosKernels TPL with LAPACK for host views (MAGMA, cuSOLVER, or rocSOLVER for device views); without
/// KokkosKernels this header declares nothing.

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS

#if defined(HAVE_MUNDYMATH_KOKKOSKERNELS)

// External
#include <KokkosKernels_config.h>  // for KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include <KokkosLapack_gesv.hpp>   // for KokkosLapack::gesv
#include <Kokkos_Core.hpp>         // for Kokkos::View, Kokkos::parallel_for, Kokkos::parallel_reduce, Kokkos::isfinite

// C++ core
#include <stdexcept>    // for std::invalid_argument, std::runtime_error
#include <type_traits>  // for std::is_same_v

// Mundy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace impl {
template <class MemorySpace>
constexpr bool invert_is_available() {
  // KokkosLapack::gesv's backends: LAPACK for host memory, cuSOLVER or MAGMA for CUDA memory, rocSOLVER or MAGMA for
  // HIP memory.
  if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
#if defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK)
    return true;
#endif
  }
#if defined(KOKKOS_ENABLE_CUDA) && \
    (defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER) || defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA))
  if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
    return true;
  }
#endif
#if defined(KOKKOS_ENABLE_HIP) && \
    (defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER) || defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA))
  if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>) {
    return true;
  }
#endif
  return false;
}
}  // namespace impl

/// \brief Whether this build can invert matrices held in MemorySpace. Where it cannot, invert throws
/// std::runtime_error.
template <class MemorySpace>
inline constexpr bool invert_is_available_v = impl::invert_is_available<MemorySpace>();

/// \brief matrix_inverse = matrix^{-1} for a dense square matrix, by LU factorization with partial pivoting.
///
/// matrix is overwritten by its LU factors, which saves a copy of an n x n matrix. Throws std::runtime_error if matrix
/// is singular: if a pivot of its factorization is zero (LAPACK's info > 0) or not finite; and if this build cannot
/// invert matrices held in the views' memory space (see invert_is_available_v).
///
/// \param[in] space The execution space instance to run on.
/// \param[in,out] matrix On entry, the n x n matrix; on exit, its LU factors.
/// \param[out] matrix_inverse The n x n inverse.
template <class ExecSpace, class MatrixView, class InverseView>
void invert(const ExecSpace& space, const MatrixView& matrix, const InverseView& matrix_inverse) {
  using scalar_t = typename MatrixView::value_type;
  using memory_space = typename MatrixView::memory_space;
  static_assert(Kokkos::is_view_v<MatrixView> && Kokkos::is_view_v<InverseView>,
                "invert: matrix and matrix_inverse must be Kokkos views.");
  static_assert(MatrixView::rank() == 2 && InverseView::rank() == 2, "invert: views must be rank 2.");
  static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                "invert: matrix must be a writable view of float or double.");
  static_assert(std::is_same_v<scalar_t, typename InverseView::value_type>,
                "invert: matrix_inverse must be a writable view of the same scalar type as matrix.");
  static_assert(std::is_same_v<typename MatrixView::array_layout, Kokkos::LayoutLeft> &&
                    std::is_same_v<typename InverseView::array_layout, Kokkos::LayoutLeft>,
                "invert: LAPACK factors column-major matrices, so the views must be LayoutLeft.");
  static_assert(std::is_same_v<memory_space, typename InverseView::memory_space>,
                "invert: matrix and matrix_inverse must share a memory space.");
  static_assert(Kokkos::SpaceAccessibility<ExecSpace, memory_space>::accessible,
                "invert: the views must be accessible from the execution space.");

  const size_t n = matrix.extent(0);
  MUNDY_THROW_REQUIRE(matrix.extent(1) == n, std::invalid_argument,
                      mundy::sink() << "invert: matrix is " << matrix.extent(0) << " x " << matrix.extent(1)
                                    << ", not square.");
  MUNDY_THROW_REQUIRE(matrix_inverse.extent(0) == n && matrix_inverse.extent(1) == n, std::invalid_argument,
                      mundy::sink() << "invert: matrix_inverse is " << matrix_inverse.extent(0) << " x "
                                    << matrix_inverse.extent(1) << ", but matrix is " << n << " x " << n << ".");
  if (n == 0) {
    return;
  }

  // KokkosKernels reaches LAPACK only for views on Device<ExecSpace, memory_space>, so alias the arguments as such: the
  // solve then runs on space, not just on the views' default execution space.
  using device_t = Kokkos::Device<ExecSpace, memory_space>;
  const Kokkos::View<scalar_t**, Kokkos::LayoutLeft, device_t> lu = matrix;
  const Kokkos::View<scalar_t**, Kokkos::LayoutLeft, device_t> solution = matrix_inverse;
#if defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::HostSpace> pivots("mundy::invert::pivots", n);  // MAGMA's are host
#else
  Kokkos::View<int*, Kokkos::LayoutLeft, device_t> pivots("mundy::invert::pivots", n);
#endif

  // Solve matrix X = I, which leaves X = matrix^{-1} in matrix_inverse and the LU factors in matrix.
  Kokkos::deep_copy(space, solution, scalar_t(0));
  Kokkos::parallel_for(
      "mundy::invert::identity", Kokkos::RangePolicy<ExecSpace>(space, 0, n),
      KOKKOS_LAMBDA(const size_t i) { solution(i, i) = scalar_t(1); });
  KokkosLapack::gesv(space, lu, solution, pivots);

  // KokkosKernels discards LAPACK's info, so detect a singular matrix from the diagonal of U ourselves.
  size_t num_bad_pivots = 0;
  Kokkos::parallel_reduce(
      "mundy::invert::check_pivots", Kokkos::RangePolicy<ExecSpace>(space, 0, n),
      KOKKOS_LAMBDA(const size_t i, size_t& count) {
        const scalar_t pivot = lu(i, i);
        count += !(pivot != scalar_t(0) && Kokkos::isfinite(pivot));
      },
      num_bad_pivots);
  MUNDY_THROW_REQUIRE(num_bad_pivots == 0, std::runtime_error,
                      mundy::sink() << "invert: the matrix is singular: " << num_bad_pivots << " of the " << n
                                    << " pivots of its LU factorization are zero or not finite.");
}

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

#endif  // MUNDY_MATH_INVERT_HPP_
