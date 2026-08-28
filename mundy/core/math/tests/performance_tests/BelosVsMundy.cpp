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

//! \file BelosVsMundy.cpp
/// \brief Runtime benchmark: solve A x = b for a random SPD A via Belos and via Mundy.
///
/// For each size, times four solves on the same system: Mundy CG (KokkosBackend), Belos CG and Belos GMRES (both
/// over Tpetra/Kokkos), and a Mundy direct inverse (MundyMathBackend, fixed-size mundy::Matrix). The direct inverse
/// uses a compile-time size, so it runs for N = 2..12; the View-based Krylov paths run for N > 12 too.
///
/// The direct inverse can also run for larger sizes and is faster than the Krylov paths: N=12 was ~500x faster than
/// either Belos or CG and N=24 was 30-50x. Both its runtime and compile cost grow steeply with N: N=24 takes
/// minutes to compile and N=30 exceeds the maximum template instantiation depth of 900.
///
/// Usage: BelosVsMundy [--simple]
///   --simple   Suppress the per-size nanobench tables and print one compact table (median time per solve).

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cstddef>
#include <iostream>
#include <string>

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{BELOS,TPETRA,KOKKOSKERNELS}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)

#include <Tpetra_Map.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <iomanip>
#include <map>
#include <mundy_math/Matrix.hpp>
#include <mundy_math/Vector.hpp>
#include <mundy_math/belos_solver.hpp>
#include <mundy_math/linear_system.hpp>
#include <mundy_math/solver_backends.hpp>
#include <random>
#include <utility>
#include <vector>

namespace {

// Tie the Kokkos spaces to Tpetra's default Node so the Belos path's exec-space-match holds.
using exec_space = Tpetra::Map<>::node_type::execution_space;
using mem_space = Tpetra::Map<>::node_type::memory_space;
using kokkos_backend_t = mundy::KokkosBackend<exec_space>;
using vec_t = Kokkos::View<double*, mem_space>;
using mat_t = Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>;

constexpr double kTol = 1.0e-8;
constexpr unsigned kMaxIters = 500;
constexpr double kNoData = -1.0;

struct Options {
  bool simple = false;
};

// Method columns of the compact --simple table, in run order.
enum Method { kMundyKokkosCG = 0, kBelosCG = 1, kBelosGmres = 2, kMundyMathDirect = 3, kNumMethods = 4 };
constexpr std::array<const char*, kNumMethods> kMethodLabels = {"mundy (kokkos/CG)", "belos (kokkos/CG)",
                                                                "belos (kokkos/GMRES)", "mundy (math/Direct)"};
using RowMap = std::map<size_t, std::array<double, kNumMethods>>;  // N -> median ns per method

// A symmetric, strongly diagonally dominant (hence SPD and well-conditioned) N x N matrix, row-major.
std::vector<double> make_spd(size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> a(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i; j < n; ++j) {
      const double v = dist(gen);
      a[i * n + j] = v;
      a[j * n + i] = v;
    }
    a[i * n + i] += 5.0 * static_cast<double>(n);
  }
  return a;
}

std::vector<double> make_rhs(size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> b(n);
  for (auto& e : b) {
    e = dist(gen);
  }
  return b;
}

mat_t make_mat_view(const std::vector<double>& a, size_t n) {
  mat_t mat(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), n, n);
  auto host = Kokkos::create_mirror_view(mat);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      host(i, j) = a[i * n + j];
    }
  }
  Kokkos::deep_copy(mat, host);
  return mat;
}

vec_t make_vec_view(const std::vector<double>& b, const std::string& name) {
  vec_t v(Kokkos::view_alloc(Kokkos::WithoutInitializing, name), b.size());
  auto host = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < b.size(); ++i) {
    host(i) = b[i];
  }
  Kokkos::deep_copy(v, host);
  return v;
}

// Mundy CG and Belos CG/GMRES (KokkosBackend) over Kokkos Views; runtime size.
void bench_view_methods(ankerl::nanobench::Bench& bench, const std::vector<double>& a, const std::vector<double>& b) {
  const size_t n = b.size();
  mat_t A = make_mat_view(a, n);
  vec_t bview = make_vec_view(b, "b");
  vec_t xview(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), n);

  mundy::CGConfig<double> cg_cfg;
  cg_cfg.tol = kTol;
  cg_cfg.max_iters = kMaxIters;
  auto mundy_inv = mundy::make_cg_inv_op<kokkos_backend_t>(mat_t(A), cg_cfg);

  mundy::BelosConfig<double> belos_cg_cfg;
  belos_cg_cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_CG;
  belos_cg_cfg.tol = kTol;
  belos_cg_cfg.max_iters = kMaxIters;
  auto belos_cg_inv = mundy::make_belos_inv_op<kokkos_backend_t>(mat_t(A), belos_cg_cfg);

  mundy::BelosConfig<double> belos_gmres_cfg;
  belos_gmres_cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  belos_gmres_cfg.tol = kTol;
  belos_gmres_cfg.max_iters = kMaxIters;
  auto belos_gmres_inv = mundy::make_belos_inv_op<kokkos_backend_t>(mat_t(A), belos_gmres_cfg);

  bench.run(kMethodLabels[kMundyKokkosCG], [&] {
    mundy_inv.apply(bview, xview);
    Kokkos::fence();
    ankerl::nanobench::doNotOptimizeAway(xview);
  });
  bench.run(kMethodLabels[kBelosCG], [&] {
    belos_cg_inv.apply(bview, xview);
    Kokkos::fence();
    ankerl::nanobench::doNotOptimizeAway(xview);
  });
  bench.run(kMethodLabels[kBelosGmres], [&] {
    belos_gmres_inv.apply(bview, xview);
    Kokkos::fence();
    ankerl::nanobench::doNotOptimizeAway(xview);
  });
}

// Mundy direct inverse (MundyMathBackend) over a fixed-size mundy::Matrix; compile-time size.
template <int N>
void bench_mundymath_method(ankerl::nanobench::Bench& bench, const std::vector<double>& a,
                            const std::vector<double>& b) {
  std::vector<double> a_local(a);  // get_matrix aliases mutable storage
  auto A = mundy::get_matrix<double, N, N>(a_local.data());
  mundy::Vector<double, N> bvec;
  for (int i = 0; i < N; ++i) {
    bvec[i] = b[static_cast<size_t>(i)];
  }
  mundy::Vector<double, N> xvec;

  bench.run(kMethodLabels[kMundyMathDirect], [&] {
    xvec = mundy::inverse(A) * bvec;
    ankerl::nanobench::doNotOptimizeAway(xvec);
  });
}

ankerl::nanobench::Bench make_bench(size_t n, const Options& opts) {
  ankerl::nanobench::Bench bench;
  bench.output(opts.simple ? nullptr : &std::cout)
      .title("invert random SPD, N=" + std::to_string(n))
      .unit("solve")
      .relative(true)
      .performanceCounters(!opts.simple);
  return bench;
}

double median_ns(const ankerl::nanobench::Bench& bench, size_t run_index) {
  const auto& results = bench.results();
  if (run_index >= results.size()) {
    return kNoData;
  }
  return results[run_index].median(ankerl::nanobench::Result::Measure::elapsed) * 1e9;
}

// N = 2..12: all four methods.
template <int N>
void bench_small(const Options& opts, RowMap& rows) {
  const auto a = make_spd(N, 20260720u + static_cast<unsigned>(N));
  const auto b = make_rhs(N, 970707u + static_cast<unsigned>(N));
  auto bench = make_bench(N, opts);
  bench_view_methods(bench, a, b);
  bench_mundymath_method<N>(bench, a, b);
  if (opts.simple) {
    rows[N] = {median_ns(bench, kMundyKokkosCG), median_ns(bench, kBelosCG), median_ns(bench, kBelosGmres),
               median_ns(bench, kMundyMathDirect)};
  }
}

template <int... Ns>
void bench_small_sizes(std::integer_sequence<int, Ns...>, const Options& opts, RowMap& rows) {
  (bench_small<Ns + 2>(opts, rows), ...);  // Ns = 0..10 -> N = 2..12
}

// N > 12: View-based Krylov methods only.
void bench_large(size_t n, const Options& opts, RowMap& rows) {
  const auto a = make_spd(n, 20260720u + static_cast<unsigned>(n));
  const auto b = make_rhs(n, 970707u + static_cast<unsigned>(n));
  auto bench = make_bench(n, opts);
  bench_view_methods(bench, a, b);
  if (opts.simple) {
    rows[n] = {median_ns(bench, kMundyKokkosCG), median_ns(bench, kBelosCG), median_ns(bench, kBelosGmres), kNoData};
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
  // One time unit for the whole table, chosen from the median finite time so both the fastest and slowest
  // methods stay legible across a wide size range.
  std::vector<double> finite;
  for (const auto& [n, cells] : rows) {
    for (double ns : cells) {
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

  constexpr int kNW = 6, kColW = 22;
  std::cout << "\n[BelosVsMundy: invert random SPD]  (median " << unit << " per solve)\n";
  std::cout << "  " << std::left << std::setw(kNW) << "N";
  for (const char* label : kMethodLabels) {
    std::cout << std::right << std::setw(kColW) << label;
  }
  std::cout << "\n  " << std::string(kNW + kNumMethods * kColW, '-') << "\n";
  for (const auto& [n, cells] : rows) {
    std::cout << "  " << std::left << std::setw(kNW) << n;
    for (double ns : cells) {
      std::cout << std::right << std::setw(kColW) << fmt_cell(ns, divisor);
    }
    std::cout << "\n";
  }
}

}  // namespace

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA && HAVE_MUNDYMATH_KOKKOSKERNELS

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--simple") {
      opts.simple = true;
    }
  }

  RowMap rows;
  bench_small_sizes(std::make_integer_sequence<int, 12>{}, opts, rows);
  // increase by 8s until 128, then by 32s until 512
  for (size_t n : {16u,  24u,  32u,  40u,  48u,  56u,  64u,  72u,  80u,   //
                   88u,  96u,  104u, 112u, 120u, 128u, 160u, 192u, 224u,  //
                   256u, 288u, 320u, 352u, 384u, 416u, 448u, 480u, 512u}) {
    bench_large(n, opts, rows);
  }
  if (opts.simple) {
    print_simple_table(rows);
  }
#else
  std::cout << "BelosVsMundy skipped: requires the Belos, Tpetra, and KokkosKernels TPLs.\n";
#endif

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
