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

/// \file UnitTestEigenvalues.cpp
/// \brief Power method and shifted power method (eigenvalues.hpp) against closed-form spectra.
///
/// Anchors:
///   - A diagonal operator, whose power iterates and Rayleigh quotients are known exactly at every step.
///   - The tridiagonal T = [-1, 2, -1] of size n: eigenvalues 4 sin^2(k pi / (2n + 2)) and unit eigenvectors
///     sqrt(2 / (n + 1)) sin(j k pi / (n + 1)), k, j = 1..n.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Mundy
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS
#include <mundy_math/Matrix.hpp>
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Vector.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_math/eigenvalues.hpp>
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix
#include <mundy_utils/rng.hpp>           // for mundy::make_philox

namespace mundy {

namespace {

using mm_backend_t = MundyMathBackend;
using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using view_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;

//! \name Group 0: compile-time checks
//@{

static_assert(std::is_same_v<decltype(make_power_strategy(PowerConfig<double>{})),
                             PowerStrategy<L2Residual, PowerConfig<double>>>,
              "The default power strategy measures ||A q - lambda q||_2.");
static_assert(ChangeResidualPolicy<ChangeResidual, double>);
static_assert(!ChangeResidualPolicy<L2Residual, double>, "The power strategy dispatches on disjoint contracts.");
static_assert(!VectorResidualPolicy<ChangeResidual, mm_backend_t, Vector3d, Vector3d>,
              "The power strategy dispatches on disjoint contracts.");
static_assert(LinearOperator<mm_backend_t, ShiftedOp<mm_backend_t, double, const Matrix3d&>, Vector3d, Vector3d>,
              "A shifted operator is itself a linear operator.");
//@}

//! \name Analytic spectrum of T = [-1, 2, -1]
//@{

double tridiagonal_eigenvalue(size_t n, size_t k) {
  const double np1 = static_cast<double>(n + 1);
  const double s = std::sin(static_cast<double>(k) * Kokkos::numbers::pi_v<double> / (2.0 * np1));
  return 4.0 * s * s;
}

double tridiagonal_eigenvector(size_t n, size_t k, size_t j) {
  const double np1 = static_cast<double>(n + 1);
  return std::sqrt(2.0 / np1) *
         std::sin(static_cast<double>(j) * static_cast<double>(k) * Kokkos::numbers::pi_v<double> / np1);
}

// scale * T
template <size_t N>
KOKKOS_INLINE_FUNCTION Matrix<double, N, N> tridiagonal(double scale) {
  auto T = Matrix<double, N, N>::zeros();
  for (size_t i = 0; i < N; ++i) {
    T(i, i) = scale * 2.0;
    if (i > 0) T(i, i - 1) = -scale;
    if (i + 1 < N) T(i, i + 1) = -scale;
  }
  return T;
}

template <size_t N>
Vector<double, N> tridiagonal_eigenvector(size_t k) {
  Vector<double, N> v;
  for (size_t j = 0; j < N; ++j) {
    v[j] = tridiagonal_eigenvector(N, k, j + 1);
  }
  return v;
}

// A generic start vector: every eigenvector component is nonzero with probability one.
template <size_t N>
KOKKOS_INLINE_FUNCTION Vector<double, N> random_start(size_t seed) {
  Vector<double, N> q;
  for (size_t i = 0; i < N; ++i) {
    openrand::Philox rng = make_philox(seed, i);
    q[i] = rng.uniform<double>(-1.0, 1.0);
  }
  return q;
}
//@}

//! \name Group 1: the iteration itself
//@{

// With A = diag(d) and q0 = (1, 1, 1), the k-th iterate is q_{k-1} proportional to d^{k-1}, so every Rayleigh quotient
// has the closed form sum d^{2k-1} / sum d^{2k-2} and every eigen residual is computable directly.
TEST(Eigenvalues, PowerIterates) {
  const Vector3d d{1.0, 0.5, 0.25};
  const Matrix3d A{d[0], 0.0,  0.0,  //
                   0.0,  d[1], 0.0,  //
                   0.0,  0.0,  d[2]};
  auto prob = make_eigen_problem<mm_backend_t>(Matrix3d(A));
  auto state = make_power_state(Vector3d{1.0, 1.0, 1.0}, Vector3d{}, Vector3d{});
  const auto strat = make_power_strategy(PowerConfig<double>{.max_iters = 20, .tol = 0.0});

  strat.initialize(prob, state);
  for (unsigned k = 1; k <= 20; ++k) {
    double numerator = 0.0;
    double denominator = 0.0;
    Vector3d q_prev;
    for (size_t i = 0; i < 3; ++i) {
      numerator += std::pow(d[i], 2 * k - 1);
      denominator += std::pow(d[i], 2 * k - 2);
      q_prev[i] = std::pow(d[i], k - 1);
    }
    q_prev = q_prev / norm(q_prev);
    const double lambda = numerator / denominator;
    const double residual = norm(A * q_prev - lambda * q_prev);

    ASSERT_FALSE(strat.iterate(prob, state)) << "iteration " << k;
    EXPECT_NEAR(state.eigenvalue(), lambda, 1e-14) << "iteration " << k;
    EXPECT_NEAR(state.residual(), residual, 1e-10 * residual) << "iteration " << k;
  }
}

// Once the third mode has died out, the eigen residual is proportional to (lambda_{n-1} / lambda_n)^k, so successive
// residuals contract by exactly that ratio.
TEST(Eigenvalues, ContractionRate) {
  constexpr size_t N = 8;
  auto prob = make_eigen_problem<mm_backend_t>(tridiagonal<N>(1.0));
  auto state = make_power_state(random_start<N>(7), Vector<double, N>{}, Vector<double, N>{});
  const auto strat = make_power_strategy(PowerConfig<double>{.max_iters = 150, .tol = 0.0});
  const double rate = tridiagonal_eigenvalue(N, N - 1) / tridiagonal_eigenvalue(N, N);

  strat.initialize(prob, state);
  std::vector<double> residuals;
  for (unsigned k = 0; k < 150; ++k) {
    strat.iterate(prob, state);
    residuals.push_back(state.residual());
  }
  for (size_t k = 100; k + 1 < residuals.size(); ++k) {
    EXPECT_NEAR(residuals[k + 1] / residuals[k], rate, 1e-5) << "iteration " << k;
  }
}

TEST(Eigenvalues, Failures) {
  // A zero start vector has no direction.
  auto prob = make_eigen_problem<mm_backend_t>(tridiagonal<3>(1.0));
  auto zero_start = make_power_state(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{});
  EXPECT_THROW(make_power_strategy(PowerConfig<double>{}).initialize(prob, zero_start), std::invalid_argument);

  // An unconverged dominant eigenvalue cannot set the shift.
  auto dominant = make_power_state(Vector3d{1.0, 0.3, -0.2}, Vector3d{}, Vector3d{});
  auto opposite = make_power_state(Vector3d{0.2, -0.7, 0.4}, Vector3d{}, Vector3d{});
  EXPECT_THROW(solve_eigen_bounds(prob, make_power_strategy(PowerConfig<double>{.max_iters = 1}), dominant, opposite),
               std::runtime_error);

  // A q0 = 0 leaves nothing to normalize once the eigenvalue-change policy asks for another iterate.
  const Matrix3d nilpotent{0.0, 1.0, 0.0,  //
                           0.0, 0.0, 0.0,  //
                           0.0, 0.0, 0.0};
  auto nilpotent_prob = make_eigen_problem<mm_backend_t>(Matrix3d(nilpotent));
  auto null_start = make_power_state(Vector3d{1.0, 0.0, 0.0}, Vector3d{}, Vector3d{});
  EXPECT_THROW(
      solve_eigen_problem(nilpotent_prob, make_power_strategy(ChangeResidual{}, PowerConfig<double>{}), null_start),
      std::runtime_error);
}
//@}

//! \name Group 2: both ends of the spectrum
//@{

// For sign * T the dominant end is sign * lambda_n and the opposite end sign * lambda_1. The residual tolerance bounds
// each eigenvector angle by tol / gap, so both eigenvectors match to round-off.
template <size_t N>
void expect_tridiagonal_bounds(double sign) {
  auto prob = make_eigen_problem<mm_backend_t>(tridiagonal<N>(sign));
  auto dominant = make_power_state(random_start<N>(1), Vector<double, N>{}, Vector<double, N>{});
  auto opposite = make_power_state(random_start<N>(2), Vector<double, N>{}, Vector<double, N>{});
  const auto strat = make_power_strategy(PowerConfig<double>{.max_iters = 20000, .tol = 1e-12});

  const auto bounds = solve_eigen_bounds(prob, strat, dominant, opposite);
  ASSERT_TRUE(bounds.opposite.converged) << "n=" << N << " sign=" << sign;

  EXPECT_NEAR(bounds.dominant.eigenvalue, sign * tridiagonal_eigenvalue(N, N), 1e-12) << "n=" << N << " sign=" << sign;
  EXPECT_NEAR(bounds.opposite.eigenvalue, sign * tridiagonal_eigenvalue(N, 1), 1e-12) << "n=" << N << " sign=" << sign;
  EXPECT_NEAR(std::abs(dot(dominant.q(), tridiagonal_eigenvector<N>(N))), 1.0, 1e-12) << "n=" << N << " sign=" << sign;
  EXPECT_NEAR(std::abs(dot(opposite.q(), tridiagonal_eigenvector<N>(1))), 1.0, 1e-12) << "n=" << N << " sign=" << sign;
}

TEST(Eigenvalues, Bounds) {
  for (const double sign : {1.0, -1.0}) {
    expect_tridiagonal_bounds<4>(sign);
    expect_tridiagonal_bounds<8>(sign);
    expect_tridiagonal_bounds<16>(sign);
  }
}

// Both policies find the same eigenvalue, but a small eigenvalue change certifies only lambda: it converges like
// rate^{2k} while the eigenvector error, and with it the eigen residual, converges like rate^k.
TEST(Eigenvalues, Policies) {
  constexpr size_t N = 8;
  constexpr double tol = 1e-12;
  auto prob = make_eigen_problem<mm_backend_t>(tridiagonal<N>(1.0));
  auto by_residual = make_power_state(random_start<N>(4), Vector<double, N>{}, Vector<double, N>{});
  auto by_change = make_power_state(random_start<N>(4), Vector<double, N>{}, Vector<double, N>{});
  const PowerConfig<double> cfg{.max_iters = 20000, .tol = tol};

  const auto residual_result = solve_eigen_problem(prob, make_power_strategy(L2Residual{}, cfg), by_residual);
  const auto change_result = solve_eigen_problem(prob, make_power_strategy(ChangeResidual{}, cfg), by_change);
  ASSERT_TRUE(residual_result.converged);
  ASSERT_TRUE(change_result.converged);

  EXPECT_NEAR(residual_result.eigenvalue, tridiagonal_eigenvalue(N, N), 1e-12);
  EXPECT_NEAR(change_result.eigenvalue, tridiagonal_eigenvalue(N, N), 1e-10);
  EXPECT_LE(norm(by_residual.r()), tol);
  EXPECT_GT(norm(by_change.r()), 1e3 * tol);
}

// Entity e holds (e + 1) T and computes its own spectral bounds inside a kernel.
void solve_bounds_in_kernel(const Kokkos::View<double* [2], Kokkos::DefaultExecutionSpace::memory_space>& eigenvalues,
                            const Kokkos::View<bool*, Kokkos::DefaultExecutionSpace::memory_space>& converged) {
  constexpr size_t N = 4;
  Kokkos::parallel_for(
      "solve_bounds_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, eigenvalues.extent(0)),
      KOKKOS_LAMBDA(const size_t e) {
        auto prob = make_eigen_problem<mm_backend_t>(tridiagonal<N>(static_cast<double>(e + 1)));
        auto dominant = make_power_state(random_start<N>(2 * e + 1), Vector<double, N>{}, Vector<double, N>{});
        auto opposite = make_power_state(random_start<N>(2 * e + 2), Vector<double, N>{}, Vector<double, N>{});
        const auto strat = make_power_strategy(PowerConfig<double>{.max_iters = 20000, .tol = 1e-12});
        const auto bounds = solve_eigen_bounds(prob, strat, dominant, opposite);
        eigenvalues(e, 0) = bounds.dominant.eigenvalue;
        eigenvalues(e, 1) = bounds.opposite.eigenvalue;
        converged(e) = bounds.opposite.converged;
      });
}

TEST(Eigenvalues, BoundsInKernel) {
  constexpr size_t N = 4;
  constexpr size_t num_entities = 8;
  Kokkos::View<double* [2], Kokkos::DefaultExecutionSpace::memory_space> eigenvalues("eigenvalues", num_entities);
  Kokkos::View<bool*, Kokkos::DefaultExecutionSpace::memory_space> converged("converged", num_entities);
  solve_bounds_in_kernel(eigenvalues, converged);

  const auto eigenvalues_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, eigenvalues);
  const auto converged_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, converged);
  for (size_t e = 0; e < num_entities; ++e) {
    const double scale = static_cast<double>(e + 1);
    EXPECT_TRUE(converged_host(e)) << "entity " << e;
    EXPECT_NEAR(eigenvalues_host(e, 0), scale * tridiagonal_eigenvalue(N, N), 1e-12 * scale) << "entity " << e;
    EXPECT_NEAR(eigenvalues_host(e, 1), scale * tridiagonal_eigenvalue(N, 1), 1e-12 * scale) << "entity " << e;
  }
}
//@}

//! \name Group 3: KokkosBackend
//@{

// y := sign * T x over Kokkos::View, independent of KokkosKernels.
struct TridiagonalKokkosOp {
  size_t n;
  double sign;

  size_t domain_size() const {
    return n;
  }
  size_t range_size() const {
    return n;
  }
  view_t make_domain_vector() const {
    return view_t("tridiagonal_domain", n);
  }
  view_t make_range_vector() const {
    return view_t("tridiagonal_range", n);
  }
  void apply(const view_t& x, view_t& y) const {
    const size_t size = n;
    const double s = sign;
    Kokkos::parallel_for(
        "TridiagonalKokkosOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, size),
        KOKKOS_LAMBDA(const size_t i) {
          double yi = 2.0 * x(i);
          if (i > 0) yi -= x(i - 1);
          if (i + 1 < size) yi -= x(i + 1);
          y(i) = s * yi;
        });
  }
};

view_t random_start_view(size_t n, size_t seed) {
  view_t q("q0", n);
  Kokkos::parallel_for(
      "random_start_view", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n), KOKKOS_LAMBDA(const size_t i) {
        openrand::Philox rng = make_philox(seed, i);
        q(i) = rng.uniform<double>(-1.0, 1.0);
      });
  return q;
}

double abs_overlap_with_tridiagonal_eigenvector(const view_t& q, size_t k) {
  const auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, q);
  double overlap = 0.0;
  for (size_t j = 0; j < q.extent(0); ++j) {
    overlap += q_host(j) * tridiagonal_eigenvector(q.extent(0), k, j + 1);
  }
  return std::abs(overlap);
}

// sign * T as an operator that applies itself and, with KokkosKernels, as a dense and a sparse matrix.
TEST(Eigenvalues, KokkosBounds) {
  constexpr size_t n = 16;
  for (const double sign : {1.0, -1.0}) {
    const auto check = [&](const auto& A, const char* kind) {
      using op_t = std::remove_cvref_t<decltype(A)>;
      auto prob = make_eigen_problem<kokkos_backend_t>(op_t(A));
      auto dominant = make_power_state(random_start_view(n, 5), kokkos_backend_t::make_range_vector(A),
                                       kokkos_backend_t::make_range_vector(A));
      auto opposite = make_power_state(random_start_view(n, 6), kokkos_backend_t::make_range_vector(A),
                                       kokkos_backend_t::make_range_vector(A));
      const auto strat = make_power_strategy(PowerConfig<double>{.max_iters = 20000, .tol = 1e-12});

      const auto bounds = solve_eigen_bounds(prob, strat, dominant, opposite);
      ASSERT_TRUE(bounds.dominant.converged) << kind << " sign=" << sign;
      ASSERT_TRUE(bounds.opposite.converged) << kind << " sign=" << sign;

      EXPECT_NEAR(bounds.dominant.eigenvalue, sign * tridiagonal_eigenvalue(n, n), 1e-12) << kind << " sign=" << sign;
      EXPECT_NEAR(bounds.opposite.eigenvalue, sign * tridiagonal_eigenvalue(n, 1), 1e-12) << kind << " sign=" << sign;
      EXPECT_NEAR(abs_overlap_with_tridiagonal_eigenvector(dominant.q(), n), 1.0, 1e-12) << kind << " sign=" << sign;
      EXPECT_NEAR(abs_overlap_with_tridiagonal_eigenvector(opposite.q(), 1), 1.0, 1e-12) << kind << " sign=" << sign;
    };
    check(TridiagonalKokkosOp{n, sign}, "applies itself");

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
    using dense_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
    using sparse_matrix_t = KokkosSparse::CrsMatrix<
        double, int, Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>, void,
        size_t>;
    const dense_matrix_t dense("dense", n, n);
    const auto dense_host = Kokkos::create_mirror_view(dense);
    for (size_t i = 0; i < n; ++i) {
      dense_host(i, i) = 2.0 * sign;
      if (i > 0) dense_host(i, i - 1) = -sign;
      if (i + 1 < n) dense_host(i, i + 1) = -sign;
    }
    Kokkos::deep_copy(dense, dense_host);
    check(dense, "dense");
    check(make_sparse_matrix<sparse_matrix_t>(dense), "sparse");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
  }
}
//@}

}  // namespace

}  // namespace mundy
