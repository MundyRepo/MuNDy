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

//! \file DirectSum.cpp
/// \brief Runtime benchmark: how direct_sum scales with N, and what its panels buy.
///
/// For N = 1024, 2048, ..., 65536 points in the unit cube, each both a source and a target, times the Stokeslet direct
/// sum with the default panel, with panels of 1, 2, 4, and 8 targets, and with a flat loop (one target per thread, no
/// panel) written here as a baseline. Each variant fits its run times to O(1), O(n), O(n log n), O(n^2), ... and to
/// a power law; the program exits nonzero unless direct_sum's best fit is O(n^2).
///
/// Usage: DirectSum [--simple] [--max-n N]
///   --simple   Suppress the per-size nanobench tables and print one compact table (Gpair/s per variant and N).
///   --max-n    The largest N (default 65536).

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cmath>     // for std::log, std::floor
#include <cstddef>   // for size_t
#include <cstdio>    // for std::printf
#include <iostream>  // for std::cout
#include <string>    // for std::string, std::stoul
#include <vector>    // for std::vector

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Vector.hpp>      // for mundy::Vector, mundy::dot
#include <mundy_math/cmath.hpp>       // for mundy::rsqrt
#include <mundy_math/direct_sum.hpp>  // for mundy::direct_sum

namespace {

using exec_space = Kokkos::DefaultExecutionSpace;
using view_t = Kokkos::View<double*, exec_space::memory_space>;
using vector3_t = mundy::Vector<double, 3>;

/// \brief The Stokeslet velocity (f + (f . r) r / r^2) / (8 pi r) at target t from the force at source s.
struct Stokeslet {
  static constexpr double one_over_eight_pi = 1.0 / (8.0 * Kokkos::numbers::pi_v<double>);

  view_t points;
  view_t forces;
  KOKKOS_INLINE_FUNCTION vector3_t operator()(const size_t t, const size_t s) const {
    const vector3_t r{points(3 * t) - points(3 * s), points(3 * t + 1) - points(3 * s + 1),
                      points(3 * t + 2) - points(3 * s + 2)};
    const vector3_t f{forces(3 * s), forces(3 * s + 1), forces(3 * s + 2)};
    const double r2 = mundy::dot(r, r);
    const bool coincident = r2 < 1e-24;
    const double rinv = coincident ? 0.0 : mundy::rsqrt(coincident ? 1.0 : r2);
    const double scale = one_over_eight_pi * rinv;
    return scale * f + (scale * mundy::dot(f, r) * rinv * rinv) * r;
  }
};

/// \brief Adds target t's sum to entries 3 t, 3 t + 1, 3 t + 2 of a flat view.
struct AddTo {
  view_t out;
  KOKKOS_INLINE_FUNCTION void operator()(const size_t t, const vector3_t& sum) const {
    out(3 * t) += sum[0];
    out(3 * t + 1) += sum[1];
    out(3 * t + 2) += sum[2];
  }
};

/// \brief The baseline: one target per thread, which sums every source itself (no panel).
struct FlatSum {
  Stokeslet interaction;
  AddTo accumulate;
  size_t num_sources;
  KOKKOS_INLINE_FUNCTION void operator()(const size_t t) const {
    vector3_t sum{0.0, 0.0, 0.0};
    for (size_t s = 0; s < num_sources; ++s) {
      sum += interaction(t, s);
    }
    accumulate(t, sum);
  }
};

/// \brief n points (or forces) as a flat (x, y, z) view, deterministic and spread over the unit cube.
view_t make_points(const size_t n, const double phase) {
  view_t points("points", 3 * n);
  auto host = Kokkos::create_mirror_view(points);
  for (size_t i = 0; i < 3 * n; ++i) {
    const double x = 0.6180339887498949 * static_cast<double>(i) + phase;
    host(i) = x - std::floor(x);
  }
  Kokkos::deep_copy(points, host);
  return points;
}

/// \brief The benchmarked variants, in table order.
enum Variant { kDefault, kPanel1, kPanel2, kPanel4, kPanel8, kFlat, kNumVariants };
const char* const kVariantNames[kNumVariants] = {"direct_sum (default panel)", "direct_sum<1>", "direct_sum<2>",
                                                 "direct_sum<4>", "direct_sum<8>", "flat (no panel)"};

/// \brief Run variant v once on n points: the full O(n^2) Stokeslet sum, finished before returning.
void run_variant(const Variant v, const size_t n, const Stokeslet& interaction, const AddTo& accumulate) {
  const exec_space space;
  switch (v) {
    case kDefault: mundy::direct_sum(space, n, n, interaction, accumulate); break;
    case kPanel1: mundy::direct_sum<1>(space, n, n, interaction, accumulate); break;
    case kPanel2: mundy::direct_sum<2>(space, n, n, interaction, accumulate); break;
    case kPanel4: mundy::direct_sum<4>(space, n, n, interaction, accumulate); break;
    case kPanel8: mundy::direct_sum<8>(space, n, n, interaction, accumulate); break;
    default:
      Kokkos::parallel_for("DirectSum::flat", Kokkos::RangePolicy<exec_space>(space, 0, n),
                           FlatSum{interaction, accumulate, n});
  }
  space.fence();
}

/// \brief The exponent p of the least-squares power law t = c n^p through (n, t).
double fitted_exponent(const std::vector<double>& ns, const std::vector<double>& ts) {
  double mean_x = 0.0;
  double mean_y = 0.0;
  for (size_t i = 0; i < ns.size(); ++i) {
    mean_x += std::log(ns[i]) / ns.size();
    mean_y += std::log(ts[i]) / ns.size();
  }
  double sxy = 0.0;
  double sxx = 0.0;
  for (size_t i = 0; i < ns.size(); ++i) {
    sxy += (std::log(ns[i]) - mean_x) * (std::log(ts[i]) - mean_y);
    sxx += (std::log(ns[i]) - mean_x) * (std::log(ns[i]) - mean_x);
  }
  return sxy / sxx;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    bool simple = false;
    size_t max_n = 65536;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--simple") {
        simple = true;
      } else if (arg == "--max-n" && i + 1 < argc) {
        max_n = std::stoul(argv[++i]);
      }
    }
    std::vector<size_t> sizes;
    for (size_t n = 1024; n <= max_n; n *= 2) {
      sizes.push_back(n);
    }

    // One bench per variant, so each fits its own complexity over the sizes.
    std::vector<ankerl::nanobench::Bench> benches(kNumVariants);
    for (int v = 0; v < kNumVariants; ++v) {
      benches[v]
          .output(simple ? nullptr : &std::cout)
          .title(kVariantNames[v])
          .unit("sum")
          .epochs(5)
          .minEpochIterations(1)
          .performanceCounters(!simple);
    }
    for (const size_t n : sizes) {
      const Stokeslet interaction{make_points(n, 0.1), make_points(n, 0.7)};
      const AddTo accumulate{view_t("velocities", 3 * n)};
      for (int v = 0; v < kNumVariants; ++v) {
        benches[v].complexityN(n).run(kVariantNames[v] + std::string(", N=") + std::to_string(n),
                 [&] { run_variant(static_cast<Variant>(v), n, interaction, accumulate); });
      }
    }

    // Throughput per variant and size, then the fitted scaling.
    std::printf("\n%-28s", "Gpair/s");
    for (const size_t n : sizes) {
      std::printf("%10zu", n);
    }
    std::printf("%12s  %s\n", "exponent", "best fit");
    for (int v = 0; v < kNumVariants; ++v) {
      std::vector<double> ns;
      std::vector<double> seconds;
      std::printf("%-28s", kVariantNames[v]);
      for (const auto& result : benches[v].results()) {
        const double n = result.config().mComplexityN;
        const double t = result.median(ankerl::nanobench::Result::Measure::elapsed);
        ns.push_back(n);
        seconds.push_back(t);
        std::printf("%10.2f", n * n / t * 1e-9);
      }
      const auto fits = benches[v].complexityBigO();
      std::printf("%12.3f  %s (rms %.3f)\n", fitted_exponent(ns, seconds), fits.front().name().c_str(),
                  fits.front().normalizedRootMeanSquare());
      if (v != kFlat && fits.front().name() != "O(n^2)") {
        status = 1;
      }
    }
    std::printf(status == 0 ? "\nPASS: every direct_sum variant scales as O(n^2).\n"
                            : "\nFAIL: a direct_sum variant does not scale as O(n^2).\n");
  }
  Kokkos::finalize();
  return status;
}
