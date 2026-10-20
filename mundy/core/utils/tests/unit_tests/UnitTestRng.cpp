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

// External libs
#include <gtest/gtest.h>      // for TEST, EXPECT_EQ, etc
#include <openrand/philox.h>  // for openrand::Philox, openrand::DEFAULT_GLOBAL_SEED

// C++ core libs
#include <cstdint>  // for uint32_t, uint64_t

// Mundy libs
#include <mundy_utils/rng.hpp>  // for mundy::make_philox

namespace mundy {

namespace {

/// Expect two generators to produce the same next few draws
void expect_same_draws(openrand::Philox actual, openrand::Philox expected, size_t seed, size_t counter) {
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(actual.draw<uint64_t>(), expected.draw<uint64_t>()) << "seed=" << seed << ", counter=" << counter;
  }
}

TEST(UnitTestRng, make_philox_with_stream_zero_matches_openrand) {
  for (const size_t seed : {size_t{0}, size_t{1}, size_t{12345678901234}}) {
    for (const size_t counter : {size_t{0}, size_t{1}, size_t{4294967295}}) {
      expect_same_draws(make_philox(seed, counter), openrand::Philox(seed, static_cast<uint32_t>(counter)), seed,
                        counter);
    }
  }
}

TEST(UnitTestRng, make_philox_passes_high_counter_bits_through_as_stream) {
  for (const size_t seed : {size_t{0}, size_t{12345678901234}}) {
    for (const uint32_t low : {uint32_t{0}, uint32_t{3}, uint32_t{4294967295}}) {
      for (const uint32_t stream : {uint32_t{0}, uint32_t{1}, uint32_t{2}, uint32_t{4294967295}}) {
        const size_t counter = (static_cast<size_t>(stream) << 32) | low;
        expect_same_draws(make_philox(seed, counter),
                          openrand::Philox(seed, low, openrand::DEFAULT_GLOBAL_SEED, 0x12345u ^ stream), seed, counter);
      }
    }
  }
}

}  // namespace

}  // namespace mundy
