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

/// \file MPRGPNormEstimate.cpp
/// \brief MPRGP's estimate of lambda_max(A) and what it costs: applies, iterations, and the expansion step against
/// MPRGP's bound 2 / lambda_max, on LCPs over x >= 0 and box-constrained QPs.
///
/// Workloads:
///   - W1: single solves of the Convex suite's analytical problems, a random LCP, and a contact-like LCP.
///   - W2: sequences of contact-like LCPs, A = D^T M D + delta I with D's columns contact normals between bodies; the
///     contacts drift, gain rows among the existing bodies, or gain stiffer rows among new bodies that couple to no
///     old row. Each solve starts from the previous solution, padded with zeros on new rows.
///   - W3: warm starts from an exact eigenvector of an interior eigenvalue: blockdiag(L_8, 2 KMS_m(r)), started from
///     L_8's top eigenvector.
///
/// Configurations: the cold estimate (C), the cold estimate at tolerance 5e-2 (C05), and warm starts with a basis of K
/// columns (WK). An MPRGP without the warm factory runs only C, as H.
///
/// Each solve prints one CSV row: workload, config, sequence, solve, n, lambda_max, Gershgorin bound, the expansion
/// step at the first step and at the end (times lambda_max), applies in the estimate (and the estimate vector's
/// formation) and in the iterations, iterations, converged, residual, f(x), and max |x - x*| where x* is known.
///
/// Usage: MPRGPNormEstimate [--workloads W1,W2,W3] [--sizes 1000,10000,100000] [--steps 50]

// External
#include <Kokkos_Core.hpp>
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS
#include <stk_util/parallel/Parallel.hpp>

// C++ core
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
#include <KokkosSparse_CrsMatrix.hpp>

// Mundy
#include <mundy_math/convex_spaces.hpp>
#include <mundy_math/cqpp.hpp>
#include <mundy_math/eigenvalues.hpp>
#include <mundy_math/pgd.hpp>
#include <mundy_math/residuals.hpp>
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>

namespace {

using exec_t = Kokkos::DefaultExecutionSpace;
using mem_t = exec_t::memory_space;
using backend_t = mundy::KokkosBackend<exec_t>;
using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, mem_t>;
using sparse_t = KokkosSparse::CrsMatrix<double, int, Kokkos::Device<exec_t, mem_t>, void, size_t>;
using entry_t = mundy::impl::SparseEntry<double>;
template <size_t K>
using basis_t = Kokkos::View<double* [K], Kokkos::LayoutLeft, mem_t>;

/// \brief Whether pgd.hpp offers MPRGP's warm start.
template <class V>
constexpr bool has_warm_mprgp =
    requires(V& p, V& e, basis_t<1>& basis) { mundy::make_mprgp_step_strategy(p, e, basis); };

constexpr double fraction = 1.8;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

/// \brief A, counting its applies.
struct CountingOp {
  sparse_t A;
  size_t* count;

  size_t domain_size() const {
    return A.numCols();
  }
  size_t range_size() const {
    return A.numRows();
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    ++*count;
    backend_t::apply(A, x, y);
  }
};

/// \brief The box lower <= x_i <= upper, either end possibly infinite.
struct Box {
  double lower = 0.0;
  double upper = std::numeric_limits<double>::infinity();
};

/// \brief f(space) for the convex space of box.
template <class F>
auto with_space(const Box& box, F&& f) {
  constexpr double inf = std::numeric_limits<double>::infinity();
  if (box.lower == -inf && box.upper == inf) {
    return f(mundy::UnconstrainedSpace<double>{});
  } else if (box.upper == inf) {
    return f(mundy::LowerBoundSpace<double>(box.lower));
  } else if (box.lower == -inf) {
    return f(mundy::UpperBoundSpace<double>(box.upper));
  }
  return f(mundy::BoundedSpace<double>(box.lower, box.upper));
}

/// \brief A host-side problem: A's entries (row-major, sorted), q, its box, and x* where known.
struct Problem {
  std::string name;
  size_t n = 0;
  std::vector<entry_t> entries;
  std::vector<double> q;
  Box box;
  std::vector<double> x_star;

  sparse_t matrix() const {
    return mundy::impl::make_sparse_matrix<sparse_t>(n, n, entries);
  }

  double gershgorin() const {
    std::vector<double> row(n, 0.0);
    for (const auto& e : entries) {
      row[e.row] += e.row == e.col ? e.value : std::abs(e.value);
    }
    return *std::max_element(row.begin(), row.end());
  }

  std::vector<double> times(const std::vector<double>& x) const {
    std::vector<double> y(n, 0.0);
    for (const auto& e : entries) {
      y[e.row] += e.value * x[e.col];
    }
    return y;
  }
};

/// \brief A's entries from a dense row-major host matrix.
std::vector<entry_t> dense_entries(size_t n, const std::vector<double>& dense) {
  std::vector<entry_t> entries;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (dense[i * n + j] != 0.0) {
        entries.push_back({i, j, dense[i * n + j]});
      }
    }
  }
  return entries;
}

/// \brief q = g* - A x*.
void set_q(Problem& prob, const std::vector<double>& grad_star) {
  const auto Ax = prob.times(prob.x_star);
  prob.q.resize(prob.n);
  for (size_t i = 0; i < prob.n; ++i) {
    prob.q[i] = grad_star[i] - Ax[i];
  }
}

view_t to_device(const std::vector<double>& v, const char* label) {
  view_t d(std::string(label), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < v.size(); ++i) {
    h(i) = v[i];
  }
  Kokkos::deep_copy(d, h);
  return d;
}

std::vector<double> to_host(const view_t& d) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

/// \brief lambda_max(A) by Lanczos to the rounding level, from the cold start.
double reference_lambda_max(const sparse_t& A) {
  const size_t n = A.numRows();
  const unsigned max_iters = static_cast<unsigned>(std::min<size_t>(std::max<size_t>(n, 2), 1000));
  view_t v("v", n), Bv_prev("Bv_prev", n), Av("Av", n);
  Kokkos::View<double*, Kokkos::HostSpace> scalars("scalars", 4 * max_iters);
  auto state = mundy::make_lanczos_state(v, Bv_prev, Av, mundy::NoVector{}, scalars);
  const auto eigen_problem = mundy::make_eigen_problem<backend_t>(A);
  const auto strategy = mundy::make_lanczos_strategy(
      mundy::LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-14, .largest = true, .smallest = false});
  return mundy::solve_eigen_problem(eigen_problem, strategy, state).largest.eigenvalue;
}

struct Record {
  std::string workload, config, sequence;
  int solve = 0;
  size_t n = 0;
  double lambda_max = nan, gershgorin = nan, alpha0_lmax = nan, alpha_end_lmax = nan;
  size_t estimate_applies = 0, step_applies = 0;
  unsigned iters = 0;
  bool converged = false;
  double residual = nan, f = nan, x_error = nan;
};

void print(const Record& r) {
  std::printf("REC,%s,%s,%s,%d,%zu,%.17g,%.17g,%.17g,%.17g,%zu,%zu,%u,%d,%.6e,%.17g,%.6e\n", r.workload.c_str(),
              r.config.c_str(), r.sequence.c_str(), r.solve, r.n, r.lambda_max, r.gershgorin, r.alpha0_lmax,
              r.alpha_end_lmax, r.estimate_applies, r.step_applies, r.iters, int(r.converged), r.residual, r.f,
              r.x_error);
  std::fflush(stdout);
}

/// \brief One MPRGP solve of prob from x (overwritten with the result), with steps from make_steps(p).
template <class MakeSteps>
Record solve(const Problem& prob, const sparse_t& A, view_t& x, double lambda_max, MakeSteps&& make_steps,
             double tol = 1e-8, unsigned max_iters = 20000) {
  Record rec;
  rec.n = prob.n;
  rec.lambda_max = lambda_max;
  rec.gershgorin = prob.gershgorin();
  size_t count = 0;
  const view_t q = to_device(prob.q, "q");
  view_t grad("grad", prob.n), x_tmp("x_tmp", prob.n), grad_tmp("grad_tmp", prob.n), p("p", prob.n);
  with_space(prob.box, [&](const auto& space) {
    const auto cqpp = mundy::make_cqpp<backend_t>(CountingOp{A, &count}, q, space);
    const auto pgd = mundy::make_pgd_solution_strategy(make_steps(p), mundy::LinfNormProjectedGradientResidual{},
                                                       mundy::PGDConfig<double>{.max_iters = max_iters, .tol = tol});
    auto state = mundy::make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
    pgd.initialize(cqpp, state);
    const bool stepped = !state.converged() && max_iters > 0;
    rec.estimate_applies = stepped ? count - 1 : 0;
    rec.alpha0_lmax = stepped ? state.step_state().expansion_step * lambda_max : nan;
    count = 0;
    while (!pgd.done(state)) {
      if (pgd.iterate(cqpp, state)) {
        break;
      }
    }
    const auto result = pgd.result(state);
    rec.step_applies = count;
    rec.alpha_end_lmax = stepped ? state.step_state().expansion_step * lambda_max : nan;
    rec.iters = result.num_iters;
    rec.converged = result.converged;
    rec.residual = result.residual;
  });
  const auto x_host = to_host(x);
  const auto Ax = prob.times(x_host);
  rec.f = 0.0;
  for (size_t i = 0; i < prob.n; ++i) {
    rec.f += x_host[i] * (0.5 * Ax[i] + prob.q[i]);
  }
  if (!prob.x_star.empty()) {
    rec.x_error = 0.0;
    for (size_t i = 0; i < prob.n; ++i) {
      rec.x_error = std::max(rec.x_error, std::abs(x_host[i] - prob.x_star[i]));
    }
  }
  return rec;
}

/// \brief A configuration of MPRGP's estimate: cold at a tolerance, or warm with K basis columns (K = 0: cold).
struct Config {
  std::string name;
  size_t K = 0;
  double tol = 1e-2;
};

std::vector<Config> configs() {
  if constexpr (has_warm_mprgp<view_t>) {
    return {{"C", 0, 1e-2}, {"C05", 0, 5e-2}, {"W1", 1, 1e-2}, {"W8", 8, 1e-2}, {"W32", 32, 1e-2}};
  } else {
    return {{"H", 0, 1e-2}};
  }
}

/// \brief One warm solve with a basis of config.K columns; V is view_t, a parameter so that the call is checked only
/// where pgd.hpp offers the warm start.
template <class V>
Record solve_warm(const Config& config, const Problem& prob, const sparse_t& A, V& x, V& e, double lambda_max) {
  if constexpr (has_warm_mprgp<V>) {
    const auto warm = [&](auto basis) {
      return solve(prob, A, x, lambda_max, [&](V& p) {
        auto steps = mundy::make_mprgp_step_strategy(p, e, basis);
        steps.norm_estimate.tol = config.tol;
        return steps;
      });
    };
    if (config.K == 1) {
      return warm(basis_t<1>("basis", prob.n));
    } else if (config.K == 8) {
      return warm(basis_t<8>("basis", prob.n));
    }
    return warm(basis_t<32>("basis", prob.n));
  } else {
    return Record{};
  }
}

/// \brief One solve under config: e is the warm start's estimate vector, of prob's size, and is ignored when cold.
Record solve_with(const Config& config, const Problem& prob, const sparse_t& A, view_t& x, view_t& e,
                  double lambda_max) {
  if (config.K == 0) {
    return solve(prob, A, x, lambda_max, [&](view_t& p) {
      auto steps = mundy::make_mprgp_step_strategy(p);
      steps.norm_estimate.tol = config.tol;
      return steps;
    });
  }
  return solve_warm(config, prob, A, x, e, lambda_max);
}

//! \name Problems
//@{

/// \brief tridiag(-1, 2, -1) of size 3 over box, with x*: lambda_max = 2 + sqrt(2).
Problem dirichlet3(const std::string& name, const Box& box, std::vector<double> x_star, std::vector<double> grad_star) {
  Problem prob{name, 3, dense_entries(3, {2, -1, 0, -1, 2, -1, 0, -1, 2}), {}, box, std::move(x_star)};
  set_q(prob, grad_star);
  return prob;
}

/// \brief The periodic Laplacian LCP of UnitTestConvex: A = L + I / 100, x* = 1 on even entries, g* = 1 on odd ones.
Problem periodic_laplacian(size_t n) {
  Problem prob{"PeriodicLaplacian" + std::to_string(n), n};
  for (size_t i = 0; i < n; ++i) {
    std::map<size_t, double> row{{i, 2.01}};
    row[(i + 1) % n] -= 1.0;
    row[(i + n - 1) % n] -= 1.0;
    for (const auto& [j, a] : row) {
      prob.entries.push_back({i, j, a});
    }
  }
  prob.x_star.resize(n);
  std::vector<double> grad_star(n);
  for (size_t i = 0; i < n; ++i) {
    prob.x_star[i] = i % 2 == 0 ? 1.0 : 0.0;
    grad_star[i] = i % 2 == 0 ? 0.0 : 1.0;
  }
  set_q(prob, grad_star);
  return prob;
}

/// \brief x*, g* with disjoint supports: x*_i in (0.1, 1) on about 40% of the rows, g*_i in (0.1, 1) on the rest.
std::vector<double> lcp_solution(Problem& prob, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  prob.x_star.assign(prob.n, 0.0);
  std::vector<double> grad_star(prob.n, 0.0);
  for (size_t i = 0; i < prob.n; ++i) {
    const double value = 0.1 + 0.9 * u01(rng);
    (u01(rng) < 0.4 ? prob.x_star[i] : grad_star[i]) = value;
  }
  return grad_star;
}

/// \brief The random LCP of UnitTestConvex: symmetric entries in (-1, 1), diagonal 10 plus the row's off-diagonal sum.
Problem random_lcp(size_t n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> dense(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      dense[i * n + j] = dense[j * n + i] = u(rng);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    double off = 0.0;
    for (size_t j = 0; j < n; ++j) {
      off += i == j ? 0.0 : std::abs(dense[i * n + j]);
    }
    dense[i * n + i] = off + 10.0;
  }
  Problem prob{"RandomLCP" + std::to_string(n), n, dense_entries(n, dense)};
  set_q(prob, lcp_solution(prob, rng));
  return prob;
}

/// \brief blockdiag(L_8, 2 KMS_m(r)), L_8 the periodic Laplacian + I / 100 and KMS_m(r)_ij = r^|i - j|; x* is that of
/// the periodic Laplacian on the first block and 0.5 + (i mod 5) / 10 on every third row of the second.
Problem blocked_kms(size_t m, double r) {
  const size_t n = 8 + m;
  Problem prob{"BlockedKMS" + std::to_string(m) + "_" + std::to_string(r).substr(0, 4), n};
  const Problem laplacian = periodic_laplacian(8);
  prob.entries = laplacian.entries;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      prob.entries.push_back({8 + i, 8 + j, 2.0 * std::pow(r, std::abs(double(i) - double(j)))});
    }
  }
  prob.x_star.assign(n, 0.0);
  std::vector<double> grad_star(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const bool active = i < 8 ? i % 2 == 0 : (i - 8) % 3 == 0;
    (active ? prob.x_star[i] : grad_star[i]) = i < 8 ? 1.0 : (active ? 0.5 + ((i - 8) % 5) / 10.0 : 1.0);
  }
  set_q(prob, grad_star);
  return prob;
}

/// \brief Bodies with mobilities and contacts between them, each with a unit normal; A = D^T M D + delta I, for D
/// mapping contact c's force to +normal on its first body and -normal on its second.
struct Contacts {
  std::vector<double> mobility;  // per body
  std::vector<size_t> body_i, body_j;
  std::vector<std::array<double, 3>> normal;

  void add(size_t i, size_t j, std::mt19937_64& rng) {
    std::normal_distribution<double> g;
    body_i.push_back(i);
    body_j.push_back(j);
    normal.push_back({g(rng), g(rng), g(rng)});
    normalize(normal.back());
  }

  static void normalize(std::array<double, 3>& v) {
    const double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (auto& c : v) {
      c /= norm;
    }
  }

  /// \brief Each normal turned by about eta.
  void drift(double eta, std::mt19937_64& rng) {
    std::normal_distribution<double> g;
    for (auto& v : normal) {
      for (auto& c : v) {
        c += eta * g(rng);
      }
      normalize(v);
    }
  }

  Problem lcp(const std::string& name, std::mt19937_64& rng) const {
    constexpr double delta = 1e-3;
    const size_t n = body_i.size();
    std::vector<std::vector<std::pair<size_t, double>>> at_body(mobility.size());  // (contact, sign)
    for (size_t c = 0; c < n; ++c) {
      at_body[body_i[c]].push_back({c, 1.0});
      at_body[body_j[c]].push_back({c, -1.0});
    }
    std::vector<std::map<size_t, double>> rows(n);
    for (size_t c = 0; c < n; ++c) {
      rows[c][c] += delta;
    }
    for (size_t b = 0; b < mobility.size(); ++b) {
      for (const auto& [c, sc] : at_body[b]) {
        for (const auto& [d, sd] : at_body[b]) {
          const auto& u = normal[c];
          const auto& v = normal[d];
          rows[c][d] += mobility[b] * sc * sd * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]);
        }
      }
    }
    Problem prob{name, n};
    for (size_t c = 0; c < n; ++c) {
      for (const auto& [d, a] : rows[c]) {
        prob.entries.push_back({c, d, a});
      }
    }
    set_q(prob, lcp_solution(prob, rng));
    return prob;
  }
};

/// \brief n contacts among n bodies, each between a body and one of its next three, with mobility in (0.5, 1.5).
Contacts initial_contacts(size_t n, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  Contacts contacts;
  for (size_t b = 0; b < n; ++b) {
    contacts.mobility.push_back(0.5 + u01(rng));
  }
  for (size_t c = 0; c < n; ++c) {
    const size_t i = rng() % n;
    contacts.add(i, (i + 1 + rng() % 3) % n, rng);
  }
  return contacts;
}
//@}

//! \name Workloads
//@{

/// \brief Single cold solves of each problem, from x_0 = 99.99 projected onto the box.
void run_w1() {
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<Problem> problems{dirichlet3("Unconstrained3", {-inf, inf}, {1, 0, 1}, {0, 0, 0}),
                                dirichlet3("InactiveBox3", {0, 2}, {1, 0, 1}, {0, 0, 0}),
                                dirichlet3("ActiveBox3", {9, 10}, {9, 9, 9}, {0, 0, 0}),
                                dirichlet3("ActiveUpperBound3", {-inf, 2}, {2, 1, 2}, {-1, 0, -1}),
                                random_lcp(3, 3),
                                random_lcp(7, 7),
                                random_lcp(200, 200),
                                periodic_laplacian(8),
                                periodic_laplacian(64),
                                periodic_laplacian(1024),
                                blocked_kms(16, 0.9)};
  std::mt19937_64 rng(2000);
  auto contacts = initial_contacts(2000, rng);
  problems.push_back(contacts.lcp("Contacts2000", rng));
  for (const auto& prob : problems) {
    const sparse_t A = prob.matrix();
    const double lambda_max = reference_lambda_max(A);
    for (const auto& config : configs()) {
      if (config.K != 0) {
        continue;
      }
      std::vector<double> start(prob.n, 99.99);
      for (auto& s : start) {
        s = std::clamp(s, prob.box.lower, prob.box.upper);
      }
      view_t x = to_device(start, "x");
      view_t e("e", prob.n);
      Record rec = solve_with(config, prob, A, x, e, lambda_max);
      rec.workload = "W1";
      rec.config = config.name;
      rec.sequence = prob.name;
      print(rec);
    }
  }
}

/// \brief Sequences of contact LCPs: "drift" (normals turn by 1e-3), "append" (by 1e-2, and 1% more contacts among
/// the existing bodies each solve), and "decoupled" (by 1e-3, and every 10th solve 20 new bodies of four times the
/// mobility with 30 contacts among them alone).
void run_w2(const std::vector<size_t>& sizes, int steps) {
  for (const size_t n0 : sizes) {
    for (const std::string variant : {"drift", "append", "decoupled"}) {
      std::mt19937_64 rng(n0 * 7 + variant.size());
      Contacts contacts = initial_contacts(n0, rng);
      std::map<std::string, std::vector<double>> x_prev;
      std::map<std::string, std::vector<double>> e_prev;
      const std::string sequence = variant + std::to_string(n0);
      for (int t = 0; t < steps; ++t) {
        if (t > 0) {
          contacts.drift(variant == "append" ? 1e-2 : 1e-3, rng);
          if (variant == "append") {
            const size_t nb = contacts.mobility.size();
            for (size_t c = 0; c < std::max<size_t>(1, n0 / 100); ++c) {
              const size_t i = rng() % nb;
              contacts.add(i, (i + 1 + rng() % 3) % nb, rng);
            }
          } else if (variant == "decoupled" && t % 10 == 0) {
            const size_t first = contacts.mobility.size();
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            for (size_t b = 0; b < 20; ++b) {
              contacts.mobility.push_back(4.0 * (0.5 + u01(rng)));
            }
            for (size_t c = 0; c < 30; ++c) {
              const size_t i = first + rng() % 20;
              contacts.add(i, first + (i - first + 1 + rng() % 3) % 20, rng);
            }
          }
        }
        std::mt19937_64 lcp_rng(t * 1000003ull + n0);
        const Problem prob = contacts.lcp(sequence, lcp_rng);
        const sparse_t A = prob.matrix();
        const double lambda_max = reference_lambda_max(A);
        for (const auto& config : configs()) {
          auto& xp = x_prev[config.name];
          auto& ep = e_prev[config.name];
          xp.resize(prob.n, 0.0);
          ep.resize(prob.n, 0.0);
          view_t x = to_device(xp, "x");
          view_t e = to_device(ep, "e");
          Record rec = solve_with(config, prob, A, x, e, lambda_max);
          xp = to_host(x);
          ep = to_host(e);
          rec.workload = "W2";
          rec.config = config.name;
          rec.sequence = sequence;
          rec.solve = t;
          print(rec);
        }
      }
    }
  }
}

/// \brief Warm starts from L_8's top eigenvector, an exact eigenvector of blockdiag(L_8, 2 KMS_m(r)) for 4.01, from
/// x_0 = 99.99 and x_0 = 0; the cold estimate runs alongside.
void run_w3() {
  for (const size_t m : {16, 32, 64}) {
    for (const double r : {0.8, 0.9, 0.95}) {
      const Problem prob = blocked_kms(m, r);
      const sparse_t A = prob.matrix();
      const double lambda_max = reference_lambda_max(A);
      for (const double start : {99.99, 0.0}) {
        for (const auto& config : configs()) {
          std::vector<double> e_host(prob.n, 0.0);
          for (size_t i = 0; i < 8; ++i) {
            e_host[i] = (i % 2 == 0 ? 1.0 : -1.0) / std::sqrt(8.0);
          }
          view_t x = to_device(std::vector<double>(prob.n, start), "x");
          view_t e = to_device(e_host, "e");
          Record rec = solve_with(config, prob, A, x, e, lambda_max);
          rec.workload = "W3";
          rec.config = config.name;
          rec.sequence = prob.name + (start == 0.0 ? "_from0" : "_from99");
          print(rec);
        }
      }
    }
  }
}
//@}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) {
    out.push_back(item);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    std::vector<std::string> workloads{"W1", "W2", "W3"};
    std::vector<size_t> sizes{1000, 10000, 100000};
    int steps = 50;
    for (int i = 1; i + 1 < argc; i += 2) {
      const std::string key = argv[i];
      if (key == "--workloads") {
        workloads = split(argv[i + 1]);
      } else if (key == "--sizes") {
        sizes.clear();
        for (const auto& s : split(argv[i + 1])) {
          sizes.push_back(std::stoul(s));
        }
      } else if (key == "--steps") {
        steps = std::stoi(argv[i + 1]);
      }
    }
    std::printf(
        "REC,workload,config,sequence,solve,n,lambda_max,gershgorin,alpha0_lmax,alpha_end_lmax,"
        "estimate_applies,step_applies,iters,converged,residual,f,x_error\n");
    for (const auto& w : workloads) {
      if (w == "W1") {
        run_w1();
      } else if (w == "W2") {
        run_w2(sizes, steps);
      } else if (w == "W3") {
        run_w3();
      }
    }
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}

#else
int main() {
  std::cout << "MPRGPNormEstimate needs KokkosKernels." << std::endl;
  return 0;
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
