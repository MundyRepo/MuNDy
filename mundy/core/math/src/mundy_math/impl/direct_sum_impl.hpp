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
#include <Kokkos_Core.hpp>  // for Kokkos::TeamPolicy, Kokkos::RangePolicy, Kokkos::ThreadVectorRange, ...

// C++ core
#include <algorithm>    // for std::min
#include <climits>      // for INT_MAX
#include <cstddef>      // for size_t
#include <stdexcept>    // for std::invalid_argument
#include <type_traits>  // for std::integral_constant, std::invoke_result_t, std::is_invocable_v, std::remove_cvref_t

// Mundy
#include <mundy_math/Matrix.hpp>         // for mundy::AMatrix, mundy::Matrix
#include <mundy_math/NumTraits.hpp>      // for mundy::is_passive_scalar_v
#include <mundy_math/Vector.hpp>         // for mundy::AVector, mundy::Vector
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

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

/// \brief The value interaction returns and its traits, once direct_sum's compile-time checks pass.
template <class ExecSpace, class Interaction, class Accumulate>
struct DirectSumTypes {
  using value_type = std::remove_cvref_t<std::invoke_result_t<const Interaction&, size_t, size_t>>;
  using traits = DirectSumValue<value_type>;

  static_assert(Kokkos::is_execution_space_v<ExecSpace>, "direct_sum: space must be a Kokkos execution space.");
  static_assert(traits::is_supported,
                "direct_sum: interaction(t, s) must return a passive scalar, a mundy::Vector, or a mundy::Matrix.");
  static_assert(std::is_invocable_v<const Accumulate&, size_t, const typename traits::sum_type&>,
                "direct_sum: accumulate(t, sum) must accept a target index and the target's total.");
};
//@}

//! \name The panel kernel
//@{

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

/// \brief direct_sum with the panel kernel on any space: a thread per panel of PanelSize targets.
template <size_t PanelSize, class ExecSpace, class Interaction, class Accumulate>
void direct_sum_with_panels(const ExecSpace& space, const size_t num_targets, const size_t num_sources,
                            const Interaction& interaction, const Accumulate& accumulate) {
  using types = DirectSumTypes<ExecSpace, Interaction, Accumulate>;
  static_assert(PanelSize >= 1, "direct_sum: a panel needs at least one target.");
  if (num_targets == 0) {
    return;
  }
  const size_t num_panels = (num_targets + PanelSize - 1) / PanelSize;
  Kokkos::parallel_for("mundy::direct_sum", Kokkos::RangePolicy<ExecSpace>(space, 0, num_panels),
                       DirectSumPanel<PanelSize, typename types::value_type, Interaction, Accumulate>{
                           interaction, accumulate, num_targets, num_sources});
}
//@}

//! \name The lane kernel
//@{

/// \brief One team of the lane kernel: each of its threads owns a target, whose sources its vector lanes split.
///
/// Lane l adds sources l, l + L, l + 2 L, ... in order, and a fixed tree combines the L lanes, so the result is the
/// same from run to run.
template <class Value, class Interaction, class Accumulate>
struct DirectSumLanes {
  using traits = DirectSumValue<Value>;
  using sum_type = typename traits::sum_type;

  Interaction interaction;
  Accumulate accumulate;
  size_t num_targets;
  size_t num_sources;

  template <class TeamMember>
  KOKKOS_INLINE_FUNCTION void operator()(const TeamMember& team) const {
    const size_t first = static_cast<size_t>(team.league_rank()) * static_cast<size_t>(team.team_size());
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team.team_size()), [&](const int i) {
      const size_t t = first + static_cast<size_t>(i);
      if (t >= num_targets) {
        return;
      }
      sum_type total;
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(team, num_sources),
          [&](const size_t s, sum_type& partial) {
            const Value contribution = interaction(t, s);
            for (size_t c = 0; c < traits::num_components; ++c) {
              traits::component(partial, c) += traits::component(contribution, c);
            }
          },
          Kokkos::Sum<sum_type>(total));
      Kokkos::single(Kokkos::PerThread(team), [&]() { accumulate(t, total); });
    });
  }
};

/// \brief direct_sum with the lane kernel on any space: Lanes vector lanes per target, TargetsPerTeam targets per team.
///
/// Both are capped at what the space allows: on a host space the lanes run one after another, and Serial allows one
/// thread per team.
template <int Lanes, int TargetsPerTeam, class ExecSpace, class Interaction, class Accumulate>
void direct_sum_with_lanes(const ExecSpace& space, const size_t num_targets, const size_t num_sources,
                           const Interaction& interaction, const Accumulate& accumulate) {
  using types = DirectSumTypes<ExecSpace, Interaction, Accumulate>;
  using policy_t = Kokkos::TeamPolicy<ExecSpace>;
  static_assert(Lanes >= 1 && (Lanes & (Lanes - 1)) == 0, "direct_sum: the lanes per target must be a power of 2.");
  static_assert(TargetsPerTeam >= 1, "direct_sum: a team needs at least one target.");
  if (num_targets == 0) {
    return;
  }
  const DirectSumLanes<typename types::value_type, Interaction, Accumulate> kernel{interaction, accumulate,
                                                                                 num_targets, num_sources};
  const int vector_length = std::min(Lanes, policy_t::vector_length_max());
  const int team_size =
      std::min(TargetsPerTeam, policy_t(space, 1, 1, vector_length).team_size_max(kernel, Kokkos::ParallelForTag{}));
  const size_t num_teams = (num_targets + static_cast<size_t>(team_size) - 1) / static_cast<size_t>(team_size);
  MUNDY_THROW_REQUIRE(num_teams <= static_cast<size_t>(INT_MAX), std::invalid_argument,
                      "direct_sum: too many targets for one launch of the lane kernel.");
  Kokkos::parallel_for("mundy::direct_sum", policy_t(space, static_cast<int>(num_teams), team_size, vector_length),
                       kernel);
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_DIRECT_SUM_IMPL_HPP_
