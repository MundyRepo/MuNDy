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

/// \brief Make a Philox generator from a 64-bit seed and a 64-bit counter.
///
/// The low 32 bits of `counter` step through a stream (Philox ctr0) and the high 32 bits pick the stream (Philox ctr1).
/// The stream is XORed into OpenRAND's default ctr1 (0x12345), so stream 0 reproduces
/// `openrand::Philox(seed, counter)` exactly. See mundy::mesh::field_start_rng_stream for handing out streams.
KOKKOS_INLINE_FUNCTION
openrand::Philox make_philox(size_t seed, size_t counter) {
  // Philox uses uint64_t seed, uint32_t counters
  MUNDY_THROW_ASSERT(seed <= Kokkos::Experimental::finite_max_v<uint64_t>, std::out_of_range,
                     "Seed value is too large for Philox.");
  const uint32_t ctr = static_cast<uint32_t>(counter);
  const uint32_t stream = static_cast<uint32_t>(counter >> 32);
  return openrand::Philox(static_cast<uint64_t>(seed), ctr, openrand::DEFAULT_GLOBAL_SEED, 0x12345u ^ stream);
}

}  // namespace mundy

#endif  // MUNDY_UTILS_RNG_HPP_
