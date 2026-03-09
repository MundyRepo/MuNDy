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

//! \file ReferenceWrapperVsExplicitReference.cpp
/// \brief Performance test the use of mundy::reference_wrapper vs explicit references in a workspace struct.

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cstddef>   // for size_t
#include <cstdint>   // for uint64_t
#include <iostream>  // for std::cout, std::endl
#include <string>    // for std::string
#include <vector>    // for std::vector

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize

// Mundy
#include <mundy_utils/reference_wrapper.hpp>  // for mundy::reference_wrapper, mundy::ref

using scalar_t = double;
using vec_t = std::vector<scalar_t>;

struct WorkspaceWrapper {
  mundy::reference_wrapper<scalar_t> x;
  mundy::reference_wrapper<scalar_t> y;
  mundy::reference_wrapper<scalar_t> z;

  KOKKOS_INLINE_FUNCTION
  constexpr void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
    scalar_t& x_ref = x;  // intentionally exercise implicit conversion
    scalar_t& y_ref = y;
    scalar_t& z_ref = z;

    const scalar_t wave = static_cast<scalar_t>((i % 11) + 1) + static_cast<scalar_t>((round % 7) + 1);
    const scalar_t t0 = x_ref + beta * y_ref;
    const scalar_t t1 = y_ref - alpha * z_ref + 0.125 * wave;
    z_ref = t0 * t1 + 0.01 * x_ref;
    x_ref = z_ref - 0.25 * y_ref + 0.5 * alpha;
    y_ref = y_ref + alpha * x_ref - beta * z_ref + 0.001 * wave;
  }
};

struct WorkspaceExplicit {
  scalar_t& x;
  scalar_t& y;
  scalar_t& z;

  KOKKOS_INLINE_FUNCTION
  constexpr void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
    const scalar_t wave = static_cast<scalar_t>((i % 11) + 1) + static_cast<scalar_t>((round % 7) + 1);
    const scalar_t t0 = x + beta * y;
    const scalar_t t1 = y - alpha * z + 0.125 * wave;
    z = t0 * t1 + 0.01 * x;
    x = z - 0.25 * y + 0.5 * alpha;
    y = y + alpha * x - beta * z + 0.001 * wave;
  }
};

void fill_deterministic(vec_t& x, vec_t& y, vec_t& z) {
  std::uint64_t seed = 0xC0FFEEULL;
  auto next_unit = [&seed]() {
    seed = seed * 1664525ULL + 1013904223ULL;
    return static_cast<scalar_t>(seed & 0xFFFFULL) / static_cast<scalar_t>(0x10000ULL);
  };

  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = 0.1 + next_unit();
    y[i] = 0.2 + next_unit();
    z[i] = 0.3 + next_unit();
  }
}

scalar_t compute_checksum(const vec_t& x, const vec_t& y, const vec_t& z) {
  scalar_t checksum = 0.0;
  for (size_t i = 0; i < x.size(); i += 7) {
    checksum += x[i] * 0.5 + y[i] * 0.25 + z[i] * 0.125;
  }
  return checksum;
}

scalar_t run_with_wrapper(vec_t& x, vec_t& y, vec_t& z, const scalar_t alpha, const scalar_t beta,
                          const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceWrapper workspace{mundy::ref(x[i]), mundy::ref(y[i]), mundy::ref(z[i])};
      workspace.step(alpha, beta, i, round);
    }
  }
  return compute_checksum(x, y, z);
}

scalar_t run_with_explicit_ref(vec_t& x, vec_t& y, vec_t& z, const scalar_t alpha, const scalar_t beta,
                               const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceExplicit workspace{x[i], y[i], z[i]};
      workspace.step(alpha, beta, i, round);
    }
  }
  return compute_checksum(x, y, z);
}

scalar_t run_direct(vec_t& x, vec_t& y, vec_t& z, const scalar_t alpha, const scalar_t beta, const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      const scalar_t wave = static_cast<scalar_t>((i % 11) + 1) + static_cast<scalar_t>((round % 7) + 1);
      const scalar_t t0 = x[i] + beta * y[i];
      const scalar_t t1 = y[i] - alpha * z[i] + 0.125 * wave;
      z[i] = t0 * t1 + 0.01 * x[i];
      x[i] = z[i] - 0.25 * y[i] + 0.5 * alpha;
      y[i] = y[i] + alpha * x[i] - beta * z[i] + 0.001 * wave;
    }
  }
  return compute_checksum(x, y, z);
}

template <typename Func>
void run_case(ankerl::nanobench::Bench& bench, const std::string& name, const vec_t& x0, const vec_t& y0,
              const vec_t& z0, Func&& func) {
  bench.run(name, [&] {
    vec_t x = x0;
    vec_t y = y0;
    vec_t z = z0;
    const scalar_t checksum = func(x, y, z);
    ankerl::nanobench::doNotOptimizeAway(checksum);
    ankerl::nanobench::doNotOptimizeAway(x);
    ankerl::nanobench::doNotOptimizeAway(y);
    ankerl::nanobench::doNotOptimizeAway(z);
  });
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr size_t num_entries = 200000;
    constexpr size_t rounds = 8;
    constexpr scalar_t alpha = 1.75;
    constexpr scalar_t beta = 0.65;

    vec_t x0(num_entries);
    vec_t y0(num_entries);
    vec_t z0(num_entries);
    fill_deterministic(x0, y0, z0);

    ankerl::nanobench::Bench bench;
    bench.relative(true)
        .title("reference_wrapper vs explicit reference workspace")
        .unit("iteration")
        .performanceCounters(true)
        .minEpochIterations(60);

    run_case(bench, "workspace/reference_wrapper", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_with_wrapper(x, y, z, alpha, beta, rounds); });

    run_case(bench, "workspace/explicit-reference", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_with_explicit_ref(x, y, z, alpha, beta, rounds); });

    run_case(bench, "direct-update", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_direct(x, y, z, alpha, beta, rounds); });

    std::cout << "Reference wrapper benchmark completed." << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
