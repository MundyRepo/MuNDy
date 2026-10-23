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

#ifndef MUNDY_UTILS_RNG_HPP_
#define MUNDY_UTILS_RNG_HPP_

// C++ core
#include <stdexcept>

// External
#include <openrand/philox.h>  // for openrand::Philox

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

KOKKOS_INLINE_FUNCTION
openrand::Philox make_philox(size_t seed, size_t counter, uint32_t global_seed, uint32_t ctr1 = 0x12345) {
  // Philox uses uint64_t seed, uint32_t counter
  MUNDY_THROW_ASSERT(seed <= Kokkos::Experimental::finite_max_v<uint64_t>, std::out_of_range,
                     "Seed value is too large for Philox.");
  MUNDY_THROW_ASSERT(counter <= Kokkos::Experimental::finite_max_v<uint32_t>, std::out_of_range,
                     "Counter value is too large for Philox.");
  return openrand::Philox(static_cast<uint64_t>(seed), static_cast<uint32_t>(counter), global_seed, ctr1);
}

// A separate overload rather than default arguments, so that make_philox still binds to a (size_t, size_t) function
// pointer (e.g., the default RNG factory in NgpAccessorExpr).
KOKKOS_INLINE_FUNCTION
openrand::Philox make_philox(size_t seed, size_t counter) {
  return make_philox(seed, counter, openrand::DEFAULT_GLOBAL_SEED);
}

}  // namespace mundy

#endif  // MUNDY_UTILS_RNG_HPP_
