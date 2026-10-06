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

//! \file PeripheryInverse.cpp
/// \brief Runtime benchmark: the periphery's two ways of inverting the second-kind operator M.
///
/// For each sphere-quadrature size it times, on the same geometry:
///   - direct build : fill the dense self-interaction matrix and invert it (KokkosBlas::gesv)
///   - direct solve : apply the stored inverse (gemv)
///   - matrix-free  : solve M f = u with Belos GMRES over the matrix-free apply_skfie (no dense matrix)
/// and reports the GMRES iteration count. The two regimes the numbers speak to:
///   - repeated solves, fixed geometry: direct amortizes its one-time build, so "direct solve" (a gemv) is the
///     per-solve cost to beat.
///   - mobile geometry (M changes every step): direct must redo "direct build" + "direct solve" each step
///     (O(N^3) inversion), while matrix-free pays only "matrix-free" (iters x O(N^2)) each step.
///
/// Usage: PeripheryInverse [--simple]
///   --simple   Suppress the per-size nanobench tables and print one compact table (median time per operation).

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy
#include <MundyMath_config.hpp>       // for HAVE_MUNDYMATH_{BELOS,TPETRA,KOKKOSKERNELS}
#include <mundy_mbody/Periphery.hpp>  // for gen_sphere_quadrature, PeripheryT, InverseMethod

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
#include <mundy_math/belos_solver.hpp>  // for mundy::{BelosConfig, BelosSolver}

namespace mundy {

namespace mbody {

namespace {

using dev_view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Periphery::DeviceMemorySpace>;
using host_view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;

constexpr double kViscosity = 1.0;
constexpr double kSphereRadius = 1.0;
constexpr double kGmresTol = 1.0e-10;
constexpr unsigned kGmresMaxIters = 1000;
constexpr double kNoData = -1.0;

struct Options {
  bool simple = false;
};

// Operation columns, in run order.
enum Op { kDirectBuild = 0, kDirectSolve = 1, kMatrixFree = 2, kNumOps = 3 };
constexpr std::array<const char*, kNumOps> kOpLabels = {"direct build (fill+inv)", "direct solve (gemv)",
                                                        "matrix-free (GMRES)"};

struct Row {
  std::array<double, kNumOps> ns{kNoData, kNoData, kNoData};  // median ns per op
  size_t dof = 0;
  unsigned gmres_iters = 0;
};
using RowMap = std::map<size_t, Row>;  // num_nodes -> row

// Sphere quadrature (points/normals size 3N, weights size N) as host views.
struct SphereGeometry {
  size_t num_nodes = 0;
  host_view_t points, normals, weights;
};

SphereGeometry make_sphere_geometry(int order) {
  std::vector<double> points_vec, weights_vec, normals_vec;
  gen_sphere_quadrature(order, kSphereRadius, &points_vec, &weights_vec, &normals_vec);
  SphereGeometry g;
  g.num_nodes = weights_vec.size();
  g.points = host_view_t("points", 3 * g.num_nodes);
  g.normals = host_view_t("normals", 3 * g.num_nodes);
  g.weights = host_view_t("weights", g.num_nodes);
  for (size_t i = 0; i < g.num_nodes; ++i) {
    g.weights(i) = weights_vec[i];
    for (int j = 0; j < 3; ++j) {
      g.points(3 * i + j) = points_vec[3 * i + j];
      g.normals(3 * i + j) = normals_vec[3 * i + j];
    }
  }
  return g;
}

dev_view_t make_random_slip(size_t num_nodes, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  host_view_t u_host("u_host", 3 * num_nodes);
  for (size_t i = 0; i < 3 * num_nodes; ++i) {
    u_host(i) = dist(gen);
  }
  dev_view_t u(Kokkos::view_alloc(Kokkos::WithoutInitializing, "u"), 3 * num_nodes);
  Kokkos::deep_copy(u, u_host);
  return u;
}

ankerl::nanobench::Bench make_bench(size_t num_nodes, const Options& opts) {
  ankerl::nanobench::Bench bench;
  bench.output(opts.simple ? nullptr : &std::cout)
      .title("periphery inverse, N=" + std::to_string(num_nodes) + " (dof=" + std::to_string(3 * num_nodes) + ")")
      .unit("op")
      .relative(true)
      .performanceCounters(!opts.simple)
      .minEpochIterations(2)
      .epochs(5)
      .warmup(1);
  return bench;
}

double median_ns(const ankerl::nanobench::Bench& bench, size_t run_index) {
  const auto& results = bench.results();
  if (run_index >= results.size()) {
    return kNoData;
  }
  return results[run_index].median(ankerl::nanobench::Result::Measure::elapsed) * 1e9;
}

void bench_size(int order, const Options& opts, RowMap& rows) {
  const SphereGeometry g = make_sphere_geometry(order);
  const size_t num_nodes = g.num_nodes;

  // make_sphere_geometry uses gen_sphere_quadrature's default (outward) normals.
  Periphery periphery(num_nodes, kViscosity);
  periphery.set_surface_positions(g.points)
      .set_surface_normals(g.normals, /*outward_normal=*/true)
      .set_quadrature_weights(g.weights);

  // Build both inverses once so the solve timings do not include setup.
  periphery.build_inverse_self_interaction_matrix(/*write_to_file=*/false);
  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = kGmresTol;
  cfg.max_iters = kGmresMaxIters;
  cfg.num_blocks = 100;
  cfg.max_restarts = 20;
  periphery.set_belos_config(cfg).build_matrix_free_inverse();

  dev_view_t u = make_random_slip(num_nodes, 20260721u + static_cast<unsigned>(num_nodes));
  dev_view_t f(Kokkos::view_alloc(Kokkos::WithoutInitializing, "f"), 3 * num_nodes);

  // One matrix-free solve outside timing to record the iteration count.
  periphery.set_inverse_method(InverseMethod::MatrixFreeGMRES);
  Kokkos::deep_copy(f, 0.0);
  periphery.compute_surface_forces(u, f);
  const unsigned gmres_iters = periphery.get_last_matrix_free_result().num_iters;

  auto bench = make_bench(num_nodes, opts);

  bench.run(kOpLabels[kDirectBuild], [&] {
    periphery.build_inverse_self_interaction_matrix(/*write_to_file=*/false);
    ankerl::nanobench::doNotOptimizeAway(periphery.get_M_inv().data());
  });

  periphery.set_inverse_method(InverseMethod::Direct);
  bench.run(kOpLabels[kDirectSolve], [&] {
    Kokkos::deep_copy(f, 0.0);
    periphery.compute_surface_forces(u, f);
    Kokkos::fence();
    ankerl::nanobench::doNotOptimizeAway(f);
  });

  periphery.set_inverse_method(InverseMethod::MatrixFreeGMRES);
  bench.run(kOpLabels[kMatrixFree], [&] {
    Kokkos::deep_copy(f, 0.0);
    periphery.compute_surface_forces(u, f);
    Kokkos::fence();
    ankerl::nanobench::doNotOptimizeAway(f);
  });

  if (opts.simple) {
    Row row;
    row.dof = 3 * num_nodes;
    row.gmres_iters = gmres_iters;
    row.ns = {median_ns(bench, kDirectBuild), median_ns(bench, kDirectSolve), median_ns(bench, kMatrixFree)};
    rows[num_nodes] = row;
  }
}

std::string fmt_cell(double ns, double divisor) {
  if (ns < 0.0) {
    return "---";
  }
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%.3f", ns / divisor);
  return buf;
}

void print_simple_table(const RowMap& rows) {
  // One time unit for the whole table, from the median finite time, so fast and slow ops stay legible.
  std::vector<double> finite;
  for (const auto& [n, row] : rows) {
    for (double ns : row.ns) {
      if (ns >= 0.0) {
        finite.push_back(ns);
      }
    }
  }
  double ref_ns = 0.0;
  if (!finite.empty()) {
    std::sort(finite.begin(), finite.end());
    ref_ns = finite[finite.size() / 2];
  }
  const char* unit = "ns";
  double divisor = 1.0;
  if (ref_ns >= 1e6) {
    unit = "ms";
    divisor = 1e6;
  } else if (ref_ns >= 1e3) {
    unit = "us";
    divisor = 1e3;
  }

  constexpr int kNW = 6, kDofW = 7, kColW = 26, kIterW = 8;
  std::cout << "\n[PeripheryInverse: invert the second-kind operator]  (median " << unit << " per op)\n";
  std::cout << "  " << std::left << std::setw(kNW) << "N" << std::setw(kDofW) << "dof";
  for (const char* label : kOpLabels) {
    std::cout << std::right << std::setw(kColW) << label;
  }
  std::cout << std::right << std::setw(kIterW) << "iters";
  std::cout << "\n  " << std::string(kNW + kDofW + kNumOps * kColW + kIterW, '-') << "\n";
  for (const auto& [n, row] : rows) {
    std::cout << "  " << std::left << std::setw(kNW) << n << std::setw(kDofW) << row.dof;
    for (double ns : row.ns) {
      std::cout << std::right << std::setw(kColW) << fmt_cell(ns, divisor);
    }
    std::cout << std::right << std::setw(kIterW) << row.gmres_iters << "\n";
  }
  std::cout << "\n  Mobile-geometry per-step cost = (direct build + direct solve) for the direct method vs\n"
            << "  (matrix-free) for GMRES; fixed-geometry repeated-solve cost = (direct solve) vs (matrix-free).\n";
}

}  // namespace

}  // namespace mbody

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA && HAVE_MUNDYMATH_KOKKOSKERNELS

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
  {
     mundy::mbody::Options opts;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--simple") {
        opts.simple = true;
      }
    }

    mundy::mbody::RowMap rows;
    // gen_sphere_quadrature uses order+1 Gauss-Legendre points; the table supports up to 128 points, so every
    // order here (up to ~126) is available. Order p gives 2(p+1)^2 nodes, i.e. 6(p+1)^2 dof.
    for (int order : {4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}) {
      bench_size(order, opts, rows);
    }
    if (opts.simple) {
      print_simple_table(rows);
    }
  }
#else
  std::cout << "PeripheryInverse skipped: requires the Belos, Tpetra, and KokkosKernels TPLs.\n";
#endif

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
