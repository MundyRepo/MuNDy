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

#ifndef MUNDY_MATH_IMPL_DIRECT_SUM_IMPL_HPP_
#define MUNDY_MATH_IMPL_DIRECT_SUM_IMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for Kokkos::reduction_identity, Kokkos::SpaceAccessibility, KOKKOS_INLINE_FUNCTION

// C++ core
#include <cstddef>      // for size_t
#include <type_traits>  // for std::integral_constant, std::remove_cv_t

// Mundy
#include <mundy_math/Matrix.hpp>     // for mundy::AMatrix, mundy::Matrix
#include <mundy_math/NumTraits.hpp>  // for mundy::is_passive_scalar_v
#include <mundy_math/Vector.hpp>     // for mundy::AVector, mundy::Vector

namespace mundy {

namespace impl {

//! \name The values a direct sum adds up
//@{

/// \brief A value of a direct sum as num_components scalars: a passive scalar is one component.
///
/// direct_sum keeps a panel's sums component by component, so the arithmetic across the panel's targets vectorizes.
/// sum_type is what accumulate receives.
template <class Value>
struct DirectSumValue {
  static constexpr bool is_supported = is_passive_scalar_v<Value>;
  using scalar_type = Value;
  using sum_type = Value;
  static constexpr size_t num_components = 1;

  KOKKOS_INLINE_FUNCTION static constexpr scalar_type component(const Value& value, size_t) {
    return value;
  }
  KOKKOS_INLINE_FUNCTION static constexpr scalar_type& component(sum_type& sum, size_t) {
    return sum;
  }
};

/// \brief An N-vector as N components.
template <typename T, size_t N, class Accessor>
struct DirectSumValue<AVector<T, N, Accessor>> {
  static constexpr bool is_supported = is_passive_scalar_v<std::remove_cv_t<T>>;
  using scalar_type = std::remove_cv_t<T>;
  using sum_type = Vector<scalar_type, N>;
  static constexpr size_t num_components = N;

  KOKKOS_INLINE_FUNCTION static constexpr scalar_type component(const AVector<T, N, Accessor>& value, size_t c) {
    return value[c];
  }
  KOKKOS_INLINE_FUNCTION static constexpr scalar_type& component(sum_type& sum, size_t c) {
    return sum[c];
  }
};

/// \brief An N x M matrix as its N M components, row by row.
template <typename T, size_t N, size_t M, class Accessor>
struct DirectSumValue<AMatrix<T, N, M, Accessor>> {
  static constexpr bool is_supported = is_passive_scalar_v<std::remove_cv_t<T>>;
  using scalar_type = std::remove_cv_t<T>;
  using sum_type = Matrix<scalar_type, N, M>;
  static constexpr size_t num_components = N * M;

  KOKKOS_INLINE_FUNCTION static constexpr scalar_type component(const AMatrix<T, N, M, Accessor>& value, size_t c) {
    return value(c / M, c % M);
  }
  KOKKOS_INLINE_FUNCTION static constexpr scalar_type& component(sum_type& sum, size_t c) {
    return sum(c / M, c % M);
  }
};
//@}

//! \name The panel kernel
//@{

/// \brief The panel size direct_sum uses on ExecSpace: 4 targets per thread on the host, 1 on a device.
///
/// On the host, a panel of 4 vectorizes with AVX2 and keeps its sums in registers (fastest measured for the
/// Stokeslet). On a device, one target per thread is the classic N-body layout.
template <class ExecSpace>
inline constexpr size_t default_direct_sum_panel_size =
    Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible ? 4 : 1;

/// \brief One thread of direct_sum: the panel of PanelSize targets starting at target panel * PanelSize.
template <size_t PanelSize, class Value, class Interaction, class Accumulate>
struct DirectSumPanel {
  using traits = DirectSumValue<Value>;
  using scalar_type = typename traits::scalar_type;
  static constexpr size_t num_components = traits::num_components;

  Interaction interaction;
  Accumulate accumulate;
  size_t num_targets;
  size_t num_sources;

  KOKKOS_INLINE_FUNCTION void operator()(const size_t panel) const {
    const size_t first = panel * PanelSize;
    const size_t size = num_targets - first < PanelSize ? num_targets - first : PanelSize;

    scalar_type sums[num_components][PanelSize];
    for (size_t c = 0; c < num_components; ++c) {
      for (size_t k = 0; k < PanelSize; ++k) {
        sums[c][k] = Kokkos::reduction_identity<scalar_type>::sum();
      }
    }

    // A full panel has a compile-time size, so the loop over its targets vectorizes.
    if (size == PanelSize) {
      add_sources(first, std::integral_constant<size_t, PanelSize>{}, sums);
    } else {
      add_sources(first, size, sums);
    }

    for (size_t k = 0; k < size; ++k) {
      typename traits::sum_type sum;
      for (size_t c = 0; c < num_components; ++c) {
        traits::component(sum, c) = sums[c][k];
      }
      accumulate(first + k, sum);
    }
  }

  /// \brief Add the contribution of every source, in order, to each of the first size targets of the panel.
  ///
  /// Force-inlined so the sums stay in registers: otherwise each addition round-trips through the stack, which halves
  /// the speed of a one-target panel.
  template <class Size>
  KOKKOS_FORCEINLINE_FUNCTION void add_sources(const size_t first, const Size size,
                                               scalar_type (&sums)[num_components][PanelSize]) const {
    for (size_t s = 0; s < num_sources; ++s) {
      for (size_t k = 0; k < size; ++k) {
        const Value contribution = interaction(first + k, s);
        for (size_t c = 0; c < num_components; ++c) {
          sums[c][k] += traits::component(contribution, c);
        }
      }
    }
  }
};
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_DIRECT_SUM_IMPL_HPP_
