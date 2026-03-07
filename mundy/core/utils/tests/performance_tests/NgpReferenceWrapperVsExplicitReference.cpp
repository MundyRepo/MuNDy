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

//! \file NgpReferenceWrapperVsExplicitReference.cpp
/// \brief Performance test for mundy::utils::reference_wrapper vs explicit references using Kokkos views/kernels.
#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cstddef>   // for size_t
#include <iostream>  // for std::cout, std::endl
#include <string>    // for std::string

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize

// Mundy
#include <mundy_utils/reference_wrapper.hpp>  // for mundy::utils::reference_wrapper, mundy::utils::ref

using scalar_t = double;
using View1D = Kokkos::View<scalar_t*, Kokkos::DefaultExecutionSpace>;

struct WorkspaceWrapper {
  mundy::utils::reference_wrapper<scalar_t> x;
  mundy::utils::reference_wrapper<scalar_t> y;
  mundy::utils::reference_wrapper<scalar_t> z;

  KOKKOS_INLINE_FUNCTION
  void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
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
  void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
    const scalar_t wave = static_cast<scalar_t>((i % 11) + 1) + static_cast<scalar_t>((round % 7) + 1);
    const scalar_t t0 = x + beta * y;
    const scalar_t t1 = y - alpha * z + 0.125 * wave;
    z = t0 * t1 + 0.01 * x;
    x = z - 0.25 * y + 0.5 * alpha;
    y = y + alpha * x - beta * z + 0.001 * wave;
  }
};

void fill_deterministic(View1D x, View1D y, View1D z) {
  const size_t n = x.extent(0);
  Kokkos::parallel_for(
      "fill_deterministic", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
        x(i) = 0.1 + static_cast<scalar_t>((17 * i + 13) % 1024) / 1024.0;
        y(i) = 0.2 + static_cast<scalar_t>((31 * i + 7) % 1024) / 1024.0;
        z(i) = 0.3 + static_cast<scalar_t>((43 * i + 19) % 1024) / 1024.0;
      });
  Kokkos::fence();
}

scalar_t compute_checksum(const View1D x, const View1D y, const View1D z) {
  scalar_t checksum = 0.0;
  const size_t n = x.extent(0);
  Kokkos::parallel_reduce(
      "checksum", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const size_t i, scalar_t& local_sum) {
        if ((i % 7) == 0) {
          local_sum += x(i) * 0.5 + y(i) * 0.25 + z(i) * 0.125;
        }
      },
      checksum);
  Kokkos::fence();
  return checksum;
}

scalar_t run_with_wrapper(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                          const size_t rounds) {
  const size_t n = x.extent(0);
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_with_wrapper", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceWrapper workspace{mundy::utils::ref(x(i)), mundy::utils::ref(y(i)), mundy::utils::ref(z(i))};
          workspace.step(alpha, beta, i, round);
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

scalar_t run_with_explicit_ref(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                               const size_t rounds) {
  const size_t n = x.extent(0);
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_with_explicit_ref", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceExplicit workspace{x(i), y(i), z(i)};
          workspace.step(alpha, beta, i, round);
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

scalar_t run_direct(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta, const size_t rounds) {
  const size_t n = x.extent(0);
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_direct", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          const scalar_t wave = static_cast<scalar_t>((i % 11) + 1) + static_cast<scalar_t>((round % 7) + 1);
          const scalar_t t0 = x(i) + beta * y(i);
          const scalar_t t1 = y(i) - alpha * z(i) + 0.125 * wave;
          z(i) = t0 * t1 + 0.01 * x(i);
          x(i) = z(i) - 0.25 * y(i) + 0.5 * alpha;
          y(i) = y(i) + alpha * x(i) - beta * z(i) + 0.001 * wave;
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

template <typename Func>
void run_case(ankerl::nanobench::Bench& bench, const std::string& name, const View1D x0, const View1D y0,
              const View1D z0, const View1D x, const View1D y, const View1D z, Func&& func) {
  bench.run(name, [&] {
    Kokkos::deep_copy(x, x0);
    Kokkos::deep_copy(y, y0);
    Kokkos::deep_copy(z, z0);

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

    View1D x0("x0", num_entries);
    View1D y0("y0", num_entries);
    View1D z0("z0", num_entries);
    View1D x("x", num_entries);
    View1D y("y", num_entries);
    View1D z("z", num_entries);

    fill_deterministic(x0, y0, z0);

    ankerl::nanobench::Bench bench;
    bench.relative(true)
        .title("NGP reference_wrapper vs explicit reference workspace")
        .unit("iteration")
        .performanceCounters(true)
        .minEpochIterations(60);

    run_case(bench, "workspace/reference_wrapper", x0, y0, z0, x, y, z,
             [&](View1D x_view, View1D y_view, View1D z_view) {
               return run_with_wrapper(x_view, y_view, z_view, alpha, beta, rounds);
             });

    run_case(bench, "workspace/explicit-reference", x0, y0, z0, x, y, z,
             [&](View1D x_view, View1D y_view, View1D z_view) {
               return run_with_explicit_ref(x_view, y_view, z_view, alpha, beta, rounds);
             });

    run_case(bench, "direct-update", x0, y0, z0, x, y, z, [&](View1D x_view, View1D y_view, View1D z_view) {
      return run_direct(x_view, y_view, z_view, alpha, beta, rounds);
    });

    std::cout << "NGP reference wrapper benchmark completed." << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
