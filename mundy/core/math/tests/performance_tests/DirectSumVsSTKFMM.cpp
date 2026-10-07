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

//! \file DirectSumVsSTKFMM.cpp
/// \brief Runtime benchmark: direct_sum against STKFMM on the same Stokeslet sum.
///
/// For N = 1024, 2048, ..., 131072 points in the unit cube, each both a source and a target, times
///   - direct_sum, the exact sum;
///   - STKFMM's own direct sum (evaluateKernel), pvfmm's hand-vectorized kernel, up to N = 32768; and
///   - STKFMM's FMM (KERNEL::Stokes, open boundaries) at multipole orders 6, 8, 10, and 12, split into its tree setup
///     (setPoints and setupTree), which moving points pay every step, and its evaluation (evaluateFMM).
/// It reports the FMM's relative L2 error against direct_sum and, per order, the smallest N at which the FMM's setup
/// plus evaluation beats direct_sum.
///
/// Everything runs on the host with OpenMP, so set OMP_NUM_THREADS once for both. STKFMM needs MPI; run one rank.
///
/// Usage: DirectSumVsSTKFMM [--max-n N]

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <algorithm>  // for std::fill
#include <cmath>      // for std::floor, std::sqrt
#include <cstddef>    // for size_t
#include <cstdio>     // for std::printf
#include <map>        // for std::map
#include <memory>     // for std::unique_ptr, std::make_unique
#include <string>     // for std::string, std::stoul
#include <vector>     // for std::vector

// External
#include <mpi.h>
#include <omp.h>

#include <STKFMM/STKFMM.hpp>

#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Vector.hpp>      // for mundy::Vector, mundy::dot
#include <mundy_math/cmath.hpp>       // for mundy::rsqrt
#include <mundy_math/direct_sum.hpp>  // for mundy::direct_sum

namespace {

using exec_space = Kokkos::DefaultHostExecutionSpace;
using view_t = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using vector3_t = mundy::Vector<double, 3>;

/// \brief The Stokeslet velocity at target t from the force at source s, as STKFMM's KERNEL::Stokes defines it.
///
/// That is (f + (f . r) r / r^2) / (8 pi r): viscosity 1.
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

/// \brief Writes target t's sum to entries 3 t, 3 t + 1, 3 t + 2 of a flat view.
struct Store {
  view_t out;
  KOKKOS_INLINE_FUNCTION void operator()(const size_t t, const vector3_t& sum) const {
    out(3 * t) = sum[0];
    out(3 * t + 1) = sum[1];
    out(3 * t + 2) = sum[2];
  }
};

/// \brief 3 n values in [0.01, 0.99): points well inside STKFMM's unit box, or forces.
std::vector<double> make_values(const size_t n, const double phase) {
  std::vector<double> values(3 * n);
  for (size_t i = 0; i < values.size(); ++i) {
    const double x = 0.6180339887498949 * static_cast<double>(i) + phase;
    values[i] = 0.01 + 0.98 * (x - std::floor(x));
  }
  return values;
}

/// \brief ||a - b|| / ||b|| in the 2-norm.
double relative_error(const std::vector<double>& a, const std::vector<double>& b) {
  double difference = 0.0;
  double norm = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    difference += (a[i] - b[i]) * (a[i] - b[i]);
    norm += b[i] * b[i];
  }
  return std::sqrt(difference / norm);
}

/// \brief The median seconds per run of fn, over 3 epochs of one run each.
template <class Function>
double median_seconds(const std::string& name, Function&& fn) {
  ankerl::nanobench::Bench bench;
  bench.output(nullptr).epochs(3).minEpochIterations(1).warmup(1).run(name, fn);
  return bench.results().front().median(ankerl::nanobench::Result::Measure::elapsed);
}

/// \brief The timings and error of one FMM order at one N.
struct FmmRow {
  double setup_seconds;
  double evaluate_seconds;
  double relative_error;
};

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    size_t max_n = 131072;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--max-n" && i + 1 < argc) {
        max_n = std::stoul(argv[++i]);
      }
    }
    const std::vector<int> orders = {6, 8, 10, 12};
    const auto kernel = stkfmm::KERNEL::Stokes;
    std::map<int, std::unique_ptr<stkfmm::Stk3DFMM>> fmms;  // built once per order: the precomputation is not timed
    for (const int order : orders) {
      fmms[order] = std::make_unique<stkfmm::Stk3DFMM>(order, 2000, stkfmm::PAXIS::NONE, stkfmm::asInteger(kernel));
      double origin[3] = {0.0, 0.0, 0.0};
      fmms[order]->setBox(origin, 1.0);
    }

    std::printf("%8s %12s %12s", "N", "direct_sum", "STKFMM P2P");
    for (const int order : orders) {
      std::printf(" | p=%-2d %9s %9s %9s", order, "setup", "evaluate", "rel err");
    }
    std::printf("\n");

    std::map<int, size_t> crossover;
    for (size_t n = 1024; n <= max_n; n *= 2) {
      std::vector<double> points = make_values(n, 0.1);
      std::vector<double> forces = make_values(n, 0.7);
      for (double& f : forces) {
        f -= 0.5;
      }

      // The exact sum.
      std::vector<double> exact(3 * n, 0.0);
      const Stokeslet interaction{view_t(points.data(), 3 * n), view_t(forces.data(), 3 * n)};
      const Store store{view_t(exact.data(), 3 * n)};
      const double direct_seconds = median_seconds("direct_sum", [&] {
        mundy::direct_sum(exec_space(), n, n, interaction, store);
        exec_space().fence();
      });

      // STKFMM's own direct sum, while it stays affordable.
      double p2p_seconds = 0.0;
      if (n <= 32768) {
        std::vector<double> p2p(3 * n, 0.0);
        p2p_seconds = median_seconds("STKFMM P2P", [&] {
          std::fill(p2p.begin(), p2p.end(), 0.0);
          fmms.begin()->second->evaluateKernel(kernel, omp_get_max_threads(), stkfmm::PPKERNEL::SLS2T, n,
                                               points.data(), forces.data(), n, points.data(), p2p.data());
        });
      }
      std::printf("%8zu %12.4g %12.4g", n, direct_seconds, p2p_seconds);

      for (const int order : orders) {
        stkfmm::Stk3DFMM& fmm = *fmms[order];
        FmmRow row{};
        row.setup_seconds = median_seconds("STKFMM setup", [&] {
          fmm.setPoints(n, points.data(), n, points.data());
          fmm.setupTree(kernel);
        });
        std::vector<double> velocities(3 * n, 0.0);
        row.evaluate_seconds = median_seconds("STKFMM evaluate", [&] {
          std::fill(velocities.begin(), velocities.end(), 0.0);  // evaluateFMM adds to its targets
          fmm.clearFMM(kernel);
          fmm.evaluateFMM(kernel, n, forces.data(), n, velocities.data());
        });
        row.relative_error = relative_error(velocities, exact);
        std::printf(" | %14.4g %9.4g %9.2e", row.setup_seconds, row.evaluate_seconds, row.relative_error);
        if (crossover.count(order) == 0 && row.setup_seconds + row.evaluate_seconds < direct_seconds) {
          crossover[order] = n;
        }
      }
      std::printf("\n");
    }

    std::printf("\nSmallest N at which the FMM (setup + evaluate) beats direct_sum:\n");
    for (const int order : orders) {
      if (crossover.count(order) != 0) {
        std::printf("  order %2d: N = %zu\n", order, crossover[order]);
      } else {
        std::printf("  order %2d: none up to N = %zu\n", order, max_n);
      }
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
