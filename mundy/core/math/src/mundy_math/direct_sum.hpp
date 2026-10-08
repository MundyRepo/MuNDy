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
/// How the work is split depends on the execution space:
///   - On the host, each thread owns a panel of 4 targets and sweeps the sources once for the whole panel, so each
///     source is loaded once per panel and the arithmetic vectorizes across the panel's targets. Each target adds its
///     sources in order 0, 1, 2, ..., so the result does not depend on the number of threads, up to FMA contraction.
///   - On a device, each target gets a warp: its 32 vector lanes split the sources and one reduction combines them.
///     Splitting every target's sum keeps a GPU busy from about a thousand targets, where one thread per target needs
///     about a hundred thousand. The result is the same from run to run, but it adds in a different order than the
///     host, so the two agree to rounding. With fewer sources than lanes, lanes idle; with many targets and only a few
///     sources (under about 100), one thread per target, is faster.
///
/// To vectorize on the host, keep interaction free of branches: select with ?: rather than branch around a division
/// or a square root, e.g. near ? 0 : rsqrt(near ? 1 : r2) instead of if (near) return 0. MundyMath builds with
/// -fno-math-errno, without which every sqrt keeps a branch that blocks vectorization.

// External
#include <Kokkos_Core.hpp>  // for Kokkos::SpaceAccessibility, Kokkos::HostSpace

// C++ core
#include <cstddef>  // for size_t

// Mundy
#include <mundy_math/impl/direct_sum_impl.hpp>  // for mundy::impl::{direct_sum_with_panels, direct_sum_with_lanes, ...}

namespace mundy {

/// \brief accumulate(t, sum over s of interaction(t, s)) for every target t, split the best way for space.
///
/// Panels of 4 targets per thread on the host and a warp of vector lanes per target on a device (see the file
/// documentation). Runs asynchronously on space, like Kokkos::parallel_for. With no sources, every target receives
/// zero.
///
/// \param[in] space The execution space instance to run on.
/// \param[in] num_targets The number of targets t.
/// \param[in] num_sources The number of sources s.
/// \param[in] interaction interaction(t, s) returns the contribution of source s to target t.
/// \param[in] accumulate accumulate(t, sum) receives the total for target t, once.
template <class ExecSpace, class Interaction, class Accumulate>
void direct_sum(const ExecSpace& space, const size_t num_targets, const size_t num_sources,
                const Interaction& interaction, const Accumulate& accumulate) {
  if constexpr (Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible) {
    // A panel of 4 targets vectorizes with AVX2 and keeps its sums in registers (fastest measured for the Stokeslet).
    impl::direct_sum_with_panels<4>(space, num_targets, num_sources, interaction, accumulate);
  } else {
    // A warp of 32 lanes per target, 4 targets (128 threads) per team: fastest measured on an RTX 6000 and an A100.
    // Teams of one warp would hit a streaming multiprocessor's limit on resident blocks before its thread limit.
    impl::direct_sum_with_lanes<32, 4>(space, num_targets, num_sources, interaction, accumulate);
  }
}

}  // namespace mundy

#endif  // MUNDY_MATH_DIRECT_SUM_HPP_
