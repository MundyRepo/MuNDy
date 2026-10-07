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

#ifndef MUNDY_MATH_DIRECT_SUM_HPP_
#define MUNDY_MATH_DIRECT_SUM_HPP_

/// \file direct_sum.hpp
/// \brief The dense direct sum of a pairwise interaction: for every target, the sum over every source.
///
/// direct_sum is the exact O(targets x sources) evaluation that a fast multipole method approximates, for any dense
/// long-range kernel (Stokeslet, RPY, double layer, Coulomb, ...) on one process. The caller supplies two functions:
/// interaction(t, s), the contribution of source s to target t as a scalar, a mundy::Vector, or a mundy::Matrix; and
/// accumulate(t, sum), which receives each target's total exactly once and decides where it goes.
///
/// Each thread owns a panel of targets and sweeps the sources once for the whole panel, so each source is loaded once
/// per panel and the arithmetic vectorizes across the panel's targets. Each target still adds its sources in order
/// 0, 1, 2, ..., so the result does not depend on the number of threads or the panel size, up to FMA contraction. The
/// default panel is 4 targets on the host and 1 on a device.
///
/// To vectorize on the host, keep interaction free of branches: select with ?: rather than branch around a division
/// or a square root, e.g. near ? 0 : rsqrt(near ? 1 : r2) instead of if (near) return 0. MundyMath builds with
/// -fno-math-errno, without which every sqrt keeps a branch that blocks vectorization.

// External
#include <Kokkos_Core.hpp>  // for Kokkos::parallel_for, Kokkos::RangePolicy

// C++ core
#include <cstddef>      // for size_t
#include <type_traits>  // for std::invoke_result_t, std::is_invocable_v, std::remove_cvref_t

// Mundy
#include <mundy_math/impl/direct_sum_impl.hpp>  // for mundy::impl::{DirectSumPanel, DirectSumValue, ...}

namespace mundy {

/// \brief accumulate(t, sum over s of interaction(t, s)) for every target t, with PanelSize targets per thread.
///
/// Runs asynchronously on space, like Kokkos::parallel_for. With no sources, every target receives zero.
///
/// \param[in] space The execution space instance to run on.
/// \param[in] num_targets The number of targets t.
/// \param[in] num_sources The number of sources s.
/// \param[in] interaction interaction(t, s) returns the contribution of source s to target t.
/// \param[in] accumulate accumulate(t, sum) receives the total for target t, once.
template <size_t PanelSize, class ExecSpace, class Interaction, class Accumulate>
void direct_sum(const ExecSpace& space, const size_t num_targets, const size_t num_sources,
                const Interaction& interaction, const Accumulate& accumulate) {
  using value_t = std::remove_cvref_t<std::invoke_result_t<const Interaction&, size_t, size_t>>;
  using traits = impl::DirectSumValue<value_t>;
  static_assert(PanelSize >= 1, "direct_sum: a panel needs at least one target.");
  static_assert(Kokkos::is_execution_space_v<ExecSpace>, "direct_sum: space must be a Kokkos execution space.");
  static_assert(traits::is_supported,
                "direct_sum: interaction(t, s) must return a passive scalar, a mundy::Vector, or a mundy::Matrix.");
  static_assert(std::is_invocable_v<const Accumulate&, size_t, const typename traits::sum_type&>,
                "direct_sum: accumulate(t, sum) must accept a target index and the target's total.");
  if (num_targets == 0) {
    return;
  }
  const size_t num_panels = (num_targets + PanelSize - 1) / PanelSize;
  Kokkos::parallel_for("mundy::direct_sum", Kokkos::RangePolicy<ExecSpace>(space, 0, num_panels),
                       impl::DirectSumPanel<PanelSize, value_t, Interaction, Accumulate>{interaction, accumulate,
                                                                                         num_targets, num_sources});
}

/// \brief accumulate(t, sum over s of interaction(t, s)) for every target t, with the default panel for space.
///
/// See the PanelSize overload for the parameters.
template <class ExecSpace, class Interaction, class Accumulate>
void direct_sum(const ExecSpace& space, const size_t num_targets, const size_t num_sources,
                const Interaction& interaction, const Accumulate& accumulate) {
  direct_sum<impl::default_direct_sum_panel_size<ExecSpace>>(space, num_targets, num_sources, interaction,
                                                             accumulate);
}

}  // namespace mundy

#endif  // MUNDY_MATH_DIRECT_SUM_HPP_
