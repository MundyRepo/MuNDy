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
/// \brief Power method, shifted power method and Lanczos method (eigenvalues.hpp) against closed-form spectra.
///
/// Reference problems:
///   - A diagonal operator, whose power iterates and Rayleigh quotients are known exactly at every step.
///   - The tridiagonal T = [-1, 2, -1] of size n: eigenvalues 4 sin^2(k pi / (2n + 2)) and unit eigenvectors
///     sqrt(2 / (n + 1)) sin(j k pi / (n + 1)), k, j = 1..n.
///   - The congruence D T D for a positive diagonal D: with the diagonal preconditioner P = D^-2, P D T D = D^-1 T D is
///     similar to T, so its spectrum is T's.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <cstddef>
#include <limits>
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
#include <mundy_math/preconditioners.hpp>  // for mundy::JacobiPreconditioner
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix
#include <mundy_utils/rng.hpp>           // for mundy::make_philox

namespace mundy {

namespace {

using mm_backend_t = MundyMathBackend;
using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using view_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;
using host_array_t = Kokkos::View<double*, Kokkos::HostSpace>;
using eigenvalue_pairs_t = Kokkos::View<double* [2], Kokkos::DefaultExecutionSpace::memory_space>;
using convergence_flags_t = Kokkos::View<bool*, Kokkos::DefaultExecutionSpace::memory_space>;

//! \name Compile-time contracts
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
static_assert(
    std::is_same_v<decltype(make_lanczos_strategy(LanczosConfig<double>{})), LanczosStrategy<LanczosConfig<double>>>,
    "The default Lanczos strategy is unpreconditioned.");
//@}

//! \name Reference matrices, operators, and start vectors
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

/// \brief The symmetric tridiagonal matrix scale * T, with T = [-1, 2, -1].
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

// d_i = 1 + i / n, the diagonal of D.
KOKKOS_INLINE_FUNCTION double congruence_scale(size_t n, size_t i) {
  return 1.0 + static_cast<double>(i) / static_cast<double>(n);
}

// Reproducible starts for the convergence tests; each seed selects a different direction.
template <size_t N>
KOKKOS_INLINE_FUNCTION Vector<double, N> random_start(size_t seed) {
  Vector<double, N> q;
  for (size_t i = 0; i < N; ++i) {
    openrand::Philox rng = make_philox(seed, i);
    q[i] = rng.uniform<double>(-1.0, 1.0);
  }
  return q;
}

/// \brief D (sign T) D, where D has diagonal 1 + i / N.
template <size_t N>
Matrix<double, N, N> congruent_tridiagonal(double sign) {
  auto A = tridiagonal<N>(sign);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      A(i, j) *= congruence_scale(N, i) * congruence_scale(N, j);
    }
  }
  return A;
}

// Matrix-free forms of the same problems, for each backend.
template <size_t N>
struct TridiagonalMundyOp {
  double sign;
  bool congruent;

  KOKKOS_INLINE_FUNCTION size_t domain_size() const {
    return N;
  }
  KOKKOS_INLINE_FUNCTION size_t range_size() const {
    return N;
  }
  KOKKOS_INLINE_FUNCTION double scale(size_t i) const {
    return congruent ? congruence_scale(N, i) : 1.0;
  }
  KOKKOS_INLINE_FUNCTION void apply(const Vector<double, N>& x, Vector<double, N>& y) const {
    for (size_t i = 0; i < N; ++i) {
      double yi = 2.0 * scale(i) * x[i];
      if (i > 0) yi -= scale(i - 1) * x[i - 1];
      if (i + 1 < N) yi -= scale(i + 1) * x[i + 1];
      y[i] = sign * scale(i) * yi;
    }
  }
};

struct TridiagonalKokkosOp {
  size_t n;
  double sign;
  bool congruent = false;

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
    const bool scaled = congruent;
    Kokkos::parallel_for(
        "TridiagonalKokkosOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, size),
        KOKKOS_LAMBDA(const size_t i) {
          const auto d = [=](size_t j) { return scaled ? congruence_scale(size, j) : 1.0; };
          double yi = 2.0 * d(i) * x(i);
          if (i > 0) yi -= d(i - 1) * x(i - 1);
          if (i + 1 < size) yi -= d(i + 1) * x(i + 1);
          y(i) = s * d(i) * yi;
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

// P = D^-2 makes P D (sign T) D similar to sign T. Supply D^2 to the componentwise division operator;
// this is a chosen positive preconditioner, not the actual diagonal of D (sign T) D.
template <size_t N>
auto make_congruence_preconditioner() {
  Vector<double, N> diagonal;
  for (size_t i = 0; i < N; ++i) {
    const double d = congruence_scale(N, i);
    diagonal[i] = d * d;
  }
  return JacobiPreconditioner<mm_backend_t, Vector<double, N>>(mm_backend_t{}, Vector<double, N>(diagonal));
}

auto make_congruence_preconditioner(size_t n) {
  view_t diagonal("d_squared", n);
  const auto host = Kokkos::create_mirror_view(diagonal);
  for (size_t i = 0; i < n; ++i) {
    const double d = congruence_scale(n, i);
    host(i) = d * d;
  }
  Kokkos::deep_copy(diagonal, host);
  return JacobiPreconditioner<kokkos_backend_t, view_t>(kokkos_backend_t{}, view_t(diagonal));
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
using dense_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
using sparse_matrix_t =
    KokkosSparse::CrsMatrix<double, int,
                            Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>,
                            void, size_t>;

/// \brief A dense representation of the matrix-free tridiagonal problem.
dense_matrix_t make_dense_tridiagonal(const TridiagonalKokkosOp& op) {
  dense_matrix_t dense("tridiagonal", op.n, op.n);
  const auto host = Kokkos::create_mirror_view(dense);
  const auto d = [&](size_t i) { return op.congruent ? congruence_scale(op.n, i) : 1.0; };
  for (size_t i = 0; i < op.n; ++i) {
    host(i, i) = 2.0 * op.sign * d(i) * d(i);
    if (i > 0) host(i, i - 1) = -op.sign * d(i) * d(i - 1);
    if (i + 1 < op.n) host(i, i + 1) = -op.sign * d(i) * d(i + 1);
  }
  Kokkos::deep_copy(dense, host);
  return dense;
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name Running and checking eigenvalue solves
//@{

double abs_overlap_with_tridiagonal_eigenvector(const view_t& q, size_t k) {
  const auto q_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, q);
  double overlap = 0.0;
  for (size_t j = 0; j < q.extent(0); ++j) {
    overlap += q_host(j) * tridiagonal_eigenvector(q.extent(0), k, j + 1);
  }
  return std::abs(overlap);
}

// Allow for accumulated rounding in these tridiagonal examples, whose spectral norm is at most 4.
// The factor 10 is a test allowance, not a general error bound for Lanczos.
double tridiagonal_roundoff_tolerance(size_t n) {
  return 10.0 * static_cast<double>(n) * std::numeric_limits<double>::epsilon() * 4.0;
}

/// \brief Require convergence and compare both Lanczos endpoints with the known spectrum.
void expect_lanczos_endpoints(const LanczosResult<double>& result, size_t n, double sign, double tol) {
  SCOPED_TRACE(::testing::Message() << "n=" << n << " sign=" << sign);
  const double smallest = sign > 0.0 ? tridiagonal_eigenvalue(n, 1) : -tridiagonal_eigenvalue(n, n);
  const double largest = sign > 0.0 ? tridiagonal_eigenvalue(n, n) : -tridiagonal_eigenvalue(n, 1);
  const double rounding = tridiagonal_roundoff_tolerance(n);
  ASSERT_TRUE(result.converged) << result;
  EXPECT_NEAR(result.smallest_eigenvalue, smallest, tol * std::abs(smallest) + rounding);
  EXPECT_NEAR(result.largest_eigenvalue, largest, tol * std::abs(largest) + rounding);
}

// Power finds the dominant-by-magnitude endpoint first, even when sign is negative.
template <size_t N>
void expect_mundy_power_representations(double sign) {
  SCOPED_TRACE(::testing::Message() << "n=" << N << " sign=" << sign);
  const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 20000, .tol = 1e-12});
  const auto check = [&](const auto& A, const char* representation) {
    SCOPED_TRACE(representation);
    using op_t = std::remove_cvref_t<decltype(A)>;
    auto problem = make_eigen_problem<mm_backend_t>(op_t(A));
    auto dominant = make_power_state(random_start<N>(1), Vector<double, N>{}, Vector<double, N>{});
    auto opposite = make_power_state(random_start<N>(2), Vector<double, N>{}, Vector<double, N>{});

    const auto bounds = solve_eigen_bounds(problem, strategy, dominant, opposite);
    ASSERT_TRUE(bounds.dominant.converged);
    ASSERT_TRUE(bounds.opposite.converged);
    EXPECT_NEAR(bounds.dominant.eigenvalue, sign * tridiagonal_eigenvalue(N, N), 1e-12);
    EXPECT_NEAR(bounds.opposite.eigenvalue, sign * tridiagonal_eigenvalue(N, 1), 1e-12);
    EXPECT_NEAR(std::abs(dot(dominant.q(), tridiagonal_eigenvector<N>(N))), 1.0, 1e-12);
    EXPECT_NEAR(std::abs(dot(opposite.q(), tridiagonal_eigenvector<N>(1))), 1.0, 1e-12);
  };
  check(tridiagonal<N>(sign), "matrix");
  check(TridiagonalMundyOp<N>{sign, false}, "matrix-free operator");
}

/// \brief Solve the same small Lanczos problem as a matrix and as a matrix-free operator.
template <size_t N, class Strategy>
void expect_mundy_lanczos_representations(double sign, bool congruent, const Strategy& strategy) {
  constexpr size_t max_iters = 4 * N;
  const auto check = [&](const auto& A, const char* representation) {
    SCOPED_TRACE(representation);
    using op_t = std::remove_cvref_t<decltype(A)>;
    auto state = make_lanczos_state(random_start<N>(3), Vector<double, N>{}, Vector<double, N>{}, Vector<double, N>{},
                                    Vector<double, max_iters>{}, Vector<double, max_iters>{});
    const auto result = solve_eigen_problem(make_eigen_problem<mm_backend_t>(op_t(A)), strategy, state);
    expect_lanczos_endpoints(result, N, sign, 1e-12);
  };
  check(congruent ? congruent_tridiagonal<N>(sign) : tridiagonal<N>(sign), "matrix");
  check(TridiagonalMundyOp<N>{sign, congruent}, "matrix-free operator");
}

template <size_t N>
void expect_unpreconditioned_lanczos_bounds(double sign) {
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 4 * N, .tol = 1e-12});
  expect_mundy_lanczos_representations<N>(sign, false, strategy);
}

template <size_t N>
void expect_preconditioned_lanczos_bounds(double sign) {
  const auto preconditioner = make_congruence_preconditioner<N>();
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 4 * N, .tol = 1e-12}, preconditioner);
  expect_mundy_lanczos_representations<N>(sign, true, strategy);
}

/// \brief Solve a Kokkos power problem through every available representation.
void expect_kokkos_power_representations(const TridiagonalKokkosOp& op, const PowerConfig<double>& config) {
  SCOPED_TRACE(::testing::Message() << "n=" << op.n << " sign=" << op.sign);
  const auto strategy = make_power_strategy(config);
  const auto check = [&](const auto& A, const char* representation) {
    SCOPED_TRACE(representation);
    using op_t = std::remove_cvref_t<decltype(A)>;
    auto problem = make_eigen_problem<kokkos_backend_t>(op_t(A));
    auto dominant = make_power_state(random_start_view(op.n, 5), kokkos_backend_t::make_range_vector(A),
                                     kokkos_backend_t::make_range_vector(A));
    auto opposite = make_power_state(random_start_view(op.n, 6), kokkos_backend_t::make_range_vector(A),
                                     kokkos_backend_t::make_range_vector(A));

    const auto bounds = solve_eigen_bounds(problem, strategy, dominant, opposite);
    ASSERT_TRUE(bounds.dominant.converged);
    ASSERT_TRUE(bounds.opposite.converged);
    EXPECT_NEAR(bounds.dominant.eigenvalue, op.sign * tridiagonal_eigenvalue(op.n, op.n), config.tol);
    EXPECT_NEAR(bounds.opposite.eigenvalue, op.sign * tridiagonal_eigenvalue(op.n, 1), config.tol);
    EXPECT_NEAR(abs_overlap_with_tridiagonal_eigenvector(dominant.q(), op.n), 1.0, 1e-12);
    EXPECT_NEAR(abs_overlap_with_tridiagonal_eigenvector(opposite.q(), 1), 1.0, 1e-12);
  };
  check(op, "matrix-free operator");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  const auto dense = make_dense_tridiagonal(op);
  check(dense, "dense matrix");
  check(make_sparse_matrix<sparse_matrix_t>(dense), "sparse matrix");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
}

/// \brief Solve a Kokkos Lanczos problem through every available representation.
template <class Strategy>
void expect_kokkos_lanczos_representations(const TridiagonalKokkosOp& op, const Strategy& strategy,
                                           const LanczosConfig<double>& config) {
  const auto check = [&](const auto& A, const char* representation) {
    SCOPED_TRACE(representation);
    using op_t = std::remove_cvref_t<decltype(A)>;
    auto state =
        make_lanczos_state(random_start_view(op.n, 9), view_t("v_prev", op.n), view_t("z", op.n), view_t("Az", op.n),
                           host_array_t("alpha", config.max_iters), host_array_t("beta", config.max_iters));
    const auto result = solve_eigen_problem(make_eigen_problem<kokkos_backend_t>(op_t(A)), strategy, state);
    expect_lanczos_endpoints(result, op.n, op.sign, config.tol);
  };
  check(op, "matrix-free operator");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  const auto dense = make_dense_tridiagonal(op);
  check(dense, "dense matrix");
  check(make_sparse_matrix<sparse_matrix_t>(dense), "sparse matrix");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
}

// Keep KOKKOS_LAMBDA outside GTest bodies for CUDA. Entity e solves its own scaled matrix (e + 1) T.
void solve_power_bounds_in_kernel(const eigenvalue_pairs_t& eigenvalues, const convergence_flags_t& converged) {
  constexpr size_t N = 4;
  Kokkos::parallel_for(
      "solve_power_bounds_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, eigenvalues.extent(0)),
      KOKKOS_LAMBDA(const size_t e) {
        auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<N>(static_cast<double>(e + 1)));
        auto dominant = make_power_state(random_start<N>(2 * e + 1), Vector<double, N>{}, Vector<double, N>{});
        auto opposite = make_power_state(random_start<N>(2 * e + 2), Vector<double, N>{}, Vector<double, N>{});
        const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 20000, .tol = 1e-12});
        const auto bounds = solve_eigen_bounds(problem, strategy, dominant, opposite);
        eigenvalues(e, 0) = bounds.dominant.eigenvalue;
        eigenvalues(e, 1) = bounds.opposite.eigenvalue;
        converged(e) = bounds.dominant.converged && bounds.opposite.converged;
      });
}

void solve_lanczos_bounds_in_kernel(const eigenvalue_pairs_t& eigenvalues, const convergence_flags_t& converged) {
  constexpr size_t N = 4;
  constexpr size_t K = 16;
  Kokkos::parallel_for(
      "solve_lanczos_bounds_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, eigenvalues.extent(0)),
      KOKKOS_LAMBDA(const size_t e) {
        auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<N>(static_cast<double>(e + 1)));
        auto state = make_lanczos_state(random_start<N>(e + 1), Vector<double, N>{}, Vector<double, N>{},
                                        Vector<double, N>{}, Vector<double, K>{}, Vector<double, K>{});
        const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = K, .tol = 1e-12});
        const auto result = solve_eigen_problem(problem, strategy, state);
        eigenvalues(e, 0) = result.smallest_eigenvalue;
        eigenvalues(e, 1) = result.largest_eigenvalue;
        converged(e) = result.converged;
      });
}
//@}

//! \name Iteration values and convergence
//@{

TEST(Eigenvalues, PowerIteratesMatchTheDiagonalClosedForm) {
  // For q0 = (1, 1, 1), q before step k is proportional to d^(k-1).
  const Vector3d d{1.0, 0.5, 0.25};
  const Matrix3d A{d[0], 0.0,  0.0,  //
                   0.0,  d[1], 0.0,  //
                   0.0,  0.0,  d[2]};
  auto problem = make_eigen_problem<mm_backend_t>(Matrix3d(A));
  auto state = make_power_state(Vector3d{1.0, 1.0, 1.0}, Vector3d{}, Vector3d{});
  const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 20, .tol = 0.0});

  strategy.initialize(problem, state);
  for (unsigned k = 1; k <= 20; ++k) {
    SCOPED_TRACE(::testing::Message() << "iteration " << k);
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

    ASSERT_FALSE(strategy.iterate(problem, state));
    EXPECT_EQ(state.iter(), k);
    EXPECT_NEAR(state.eigenvalue(), lambda, 1e-14);
    EXPECT_NEAR(state.residual(), residual, 1e-10 * residual);
    Vector3d expected_next;
    for (size_t i = 0; i < 3; ++i) {
      expected_next[i] = std::pow(d[i], k);
    }
    expected_next = expected_next / norm(expected_next);
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_NEAR(state.q()[i], expected_next[i], 1e-14) << "component " << i;
    }
  }
}

TEST(Eigenvalues, LanczosFirstStepMatchesTheRayleighQuotientAndResidual) {
  // The first projected matrix has one entry, q^T A q, so both Ritz values equal the Rayleigh quotient.
  const Matrix3d A{1.0, 0.0, 0.0,  //
                   0.0, 2.0, 0.0,  //
                   0.0, 0.0, 4.0};
  const Vector3d start{1.0, 2.0, 1.0};
  const Vector3d q = start / norm(start);
  const double expected_eigenvalue = dot(q, A * q);
  const double expected_residual = norm(A * q - expected_eigenvalue * q);
  auto problem = make_eigen_problem<mm_backend_t>(Matrix3d(A));
  auto state = make_lanczos_state(Vector3d(start), Vector3d{}, Vector3d{}, Vector3d{}, Vector3d{}, Vector3d{});
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 3, .tol = 0.0});

  strategy.initialize(problem, state);
  ASSERT_FALSE(strategy.iterate(problem, state));
  EXPECT_EQ(state.iter(), 1u);
  EXPECT_NEAR(state.smallest_eigenvalue(), expected_eigenvalue, 1e-14);
  EXPECT_NEAR(state.largest_eigenvalue(), expected_eigenvalue, 1e-14);
  EXPECT_NEAR(state.smallest_ritz_residual(), expected_residual, 1e-14);
  EXPECT_NEAR(state.largest_ritz_residual(), expected_residual, 1e-14);
}

TEST(Eigenvalues, PowerResidualContractsAtTheSpectralRatio) {
  // After the faster modes decay, successive residuals approach lambda_(N-1) / lambda_N.
  constexpr size_t N = 8;
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<N>(1.0));
  auto state = make_power_state(random_start<N>(7), Vector<double, N>{}, Vector<double, N>{});
  const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 150, .tol = 0.0});
  const double rate = tridiagonal_eigenvalue(N, N - 1) / tridiagonal_eigenvalue(N, N);

  strategy.initialize(problem, state);
  std::vector<double> residuals;
  for (unsigned k = 0; k < 150; ++k) {
    ASSERT_FALSE(strategy.iterate(problem, state)) << "iteration " << k;
    ASSERT_EQ(state.iter(), k + 1);
    residuals.push_back(state.residual());
  }
  for (size_t k = 100; k + 1 < residuals.size(); ++k) {
    EXPECT_NEAR(residuals[k + 1] / residuals[k], rate, 1e-5) << "iteration " << k;
  }
}

TEST(Eigenvalues, LanczosRitzValuesMoveTowardTheSpectrumEndpoints) {
  constexpr size_t N = 16;
  constexpr size_t max_iters = 4 * N;
  const double rounding = tridiagonal_roundoff_tolerance(N);
  const double smallest = tridiagonal_eigenvalue(N, 1);
  const double largest = tridiagonal_eigenvalue(N, N);
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<N>(1.0));
  auto state = make_lanczos_state(random_start<N>(5), Vector<double, N>{}, Vector<double, N>{}, Vector<double, N>{},
                                  Vector<double, max_iters>{}, Vector<double, max_iters>{});
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 0.0});

  strategy.initialize(problem, state);
  double previous_smallest = std::numeric_limits<double>::infinity();
  double previous_largest = -std::numeric_limits<double>::infinity();
  // Include steps beyond N if rounding prevents exact termination. Only check steps that actually execute.
  for (unsigned k = 1; k <= max_iters; ++k) {
    SCOPED_TRACE(::testing::Message() << "iteration " << k);
    const bool stopped = strategy.iterate(problem, state);
    ASSERT_EQ(state.iter(), k);
    EXPECT_GE(state.smallest_eigenvalue(), smallest - rounding);
    EXPECT_LE(state.largest_eigenvalue(), largest + rounding);
    EXPECT_LE(state.smallest_eigenvalue(), previous_smallest + rounding);
    EXPECT_GE(state.largest_eigenvalue(), previous_largest - rounding);
    previous_smallest = state.smallest_eigenvalue();
    previous_largest = state.largest_eigenvalue();
    if (stopped) break;
  }
  EXPECT_NEAR(state.smallest_eigenvalue(), smallest, rounding);
  EXPECT_NEAR(state.largest_eigenvalue(), largest, rounding);
}
//@}

//! \name Iteration limits
//@{

TEST(Eigenvalues, PowerIterationLimitReturnsTheLastIterate) {
  constexpr size_t N = 16;
  constexpr unsigned max_iters = 3;
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<N>(1.0));
  auto manual = make_power_state(random_start<N>(5), Vector<double, N>{}, Vector<double, N>{});
  auto solved = make_power_state(random_start<N>(5), Vector<double, N>{}, Vector<double, N>{});
  const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = max_iters, .tol = 1e-12});

  strategy.initialize(problem, manual);
  for (unsigned k = 1; k <= max_iters; ++k) {
    ASSERT_FALSE(strategy.iterate(problem, manual));
    ASSERT_EQ(manual.iter(), k);
  }
  const auto result = solve_eigen_problem(problem, strategy, solved);

  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.num_iters, max_iters);
  EXPECT_EQ(solved.iter(), manual.iter());
  EXPECT_DOUBLE_EQ(result.eigenvalue, manual.eigenvalue());
  EXPECT_DOUBLE_EQ(result.residual, manual.residual());
  EXPECT_GT(result.residual, 1e-12);
  EXPECT_LT(result.eigenvalue, tridiagonal_eigenvalue(N, N));
}

TEST(Eigenvalues, LanczosIterationLimitReturnsTheLastIterate) {
  constexpr size_t N = 16;
  constexpr unsigned max_iters = 3;
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<N>(1.0));
  auto manual = make_lanczos_state(random_start<N>(5), Vector<double, N>{}, Vector<double, N>{}, Vector<double, N>{},
                                   Vector<double, N>{}, Vector<double, N>{});
  auto solved = make_lanczos_state(random_start<N>(5), Vector<double, N>{}, Vector<double, N>{}, Vector<double, N>{},
                                   Vector<double, N>{}, Vector<double, N>{});
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-12});

  strategy.initialize(problem, manual);
  for (unsigned k = 1; k <= max_iters; ++k) {
    ASSERT_FALSE(strategy.iterate(problem, manual));
    ASSERT_EQ(manual.iter(), k);
  }
  const auto result = solve_eigen_problem(problem, strategy, solved);

  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.num_iters, max_iters);
  EXPECT_EQ(solved.iter(), manual.iter());
  EXPECT_DOUBLE_EQ(result.smallest_eigenvalue, manual.smallest_eigenvalue());
  EXPECT_DOUBLE_EQ(result.largest_eigenvalue, manual.largest_eigenvalue());
  EXPECT_DOUBLE_EQ(result.residual, manual.residual());
  EXPECT_GT(result.residual, 1e-12);
  EXPECT_GT(result.smallest_eigenvalue, tridiagonal_eigenvalue(N, 1));
  EXPECT_LT(result.largest_eigenvalue, tridiagonal_eigenvalue(N, N));
}
//@}

//! \name Spectrum endpoints
//@{

TEST(Eigenvalues, PowerAndShiftedPowerFindBothSpectrumEndpoints) {
  for (const double sign : {1.0, -1.0}) {
    expect_mundy_power_representations<4>(sign);
    expect_mundy_power_representations<8>(sign);
    expect_mundy_power_representations<16>(sign);
  }
}

TEST(Eigenvalues, LanczosFindsBothSpectrumEndpoints) {
  for (const double sign : {1.0, -1.0}) {
    expect_unpreconditioned_lanczos_bounds<4>(sign);
    expect_unpreconditioned_lanczos_bounds<8>(sign);
    expect_unpreconditioned_lanczos_bounds<16>(sign);
  }
}
//@}

//! \name Zero start vectors
//@{

TEST(Eigenvalues, PowerRejectsZeroStartVector) {
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<3>(1.0));
  auto state = make_power_state(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{});
  EXPECT_THROW(make_power_strategy(PowerConfig<double>{}).initialize(problem, state), std::invalid_argument);
}

TEST(Eigenvalues, LanczosRejectsZeroStartVector) {
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<3>(1.0));
  auto state = make_lanczos_state(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{}, Vector<double, 6>{},
                                  Vector<double, 6>{});
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6});
  EXPECT_THROW(strategy.initialize(problem, state), std::invalid_argument);
}
//@}

//! \name Backend representations and device execution
//@{

TEST(Eigenvalues, KokkosPowerFindsBoundsForEveryRepresentation) {
  constexpr size_t n = 16;
  const PowerConfig<double> config{.max_iters = 20000, .tol = 1e-12};
  for (const double sign : {1.0, -1.0}) {
    expect_kokkos_power_representations(TridiagonalKokkosOp{n, sign}, config);
  }
}

TEST(Eigenvalues, KokkosLanczosFindsEndpointsForEveryRepresentation) {
  for (const size_t n : {16, 200}) {
    // The n = 200 problem has a small lowest eigenvalue; keep the relative tolerance above its rounding floor.
    const LanczosConfig<double> config{.max_iters = static_cast<unsigned>(10 * n), .tol = 1e-10};
    const auto strategy = make_lanczos_strategy(config);
    for (const double sign : {1.0, -1.0}) {
      expect_kokkos_lanczos_representations(TridiagonalKokkosOp{n, sign, false}, strategy, config);
    }
  }
}

TEST(Eigenvalues, PowerFindsIndependentBoundsInsideAKernel) {
  constexpr size_t N = 4;
  constexpr size_t num_entities = 8;
  eigenvalue_pairs_t eigenvalues("eigenvalues", num_entities);
  convergence_flags_t converged("converged", num_entities);
  solve_power_bounds_in_kernel(eigenvalues, converged);

  const auto eigenvalues_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, eigenvalues);
  const auto converged_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, converged);
  for (size_t e = 0; e < num_entities; ++e) {
    const double scale = static_cast<double>(e + 1);
    EXPECT_TRUE(converged_host(e)) << "entity " << e;
    EXPECT_NEAR(eigenvalues_host(e, 0), scale * tridiagonal_eigenvalue(N, N), 1e-12 * scale) << "entity " << e;
    EXPECT_NEAR(eigenvalues_host(e, 1), scale * tridiagonal_eigenvalue(N, 1), 1e-12 * scale) << "entity " << e;
  }
}

TEST(Eigenvalues, LanczosFindsIndependentEndpointsInsideAKernel) {
  constexpr size_t N = 4;
  constexpr size_t num_entities = 8;
  const double rounding = tridiagonal_roundoff_tolerance(N);
  eigenvalue_pairs_t eigenvalues("eigenvalues", num_entities);
  convergence_flags_t converged("converged", num_entities);
  solve_lanczos_bounds_in_kernel(eigenvalues, converged);

  const auto eigenvalues_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, eigenvalues);
  const auto converged_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, converged);
  for (size_t e = 0; e < num_entities; ++e) {
    const double scale = static_cast<double>(e + 1);
    const double smallest = scale * tridiagonal_eigenvalue(N, 1);
    const double largest = scale * tridiagonal_eigenvalue(N, N);
    EXPECT_TRUE(converged_host(e)) << "entity " << e;
    EXPECT_NEAR(eigenvalues_host(e, 0), smallest, 1e-12 * smallest + scale * rounding) << "entity " << e;
    EXPECT_NEAR(eigenvalues_host(e, 1), largest, 1e-12 * largest + scale * rounding) << "entity " << e;
  }
}
//@}

//! \name Power residual policies and shifting
//@{

TEST(Eigenvalues, EigenvalueChangeCanConvergeBeforeTheEigenvector) {
  // On this problem the eigenvalue change becomes small while the eigenvector residual is still larger.
  constexpr size_t N = 8;
  constexpr double tol = 1e-12;
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<N>(1.0));
  auto by_residual = make_power_state(random_start<N>(4), Vector<double, N>{}, Vector<double, N>{});
  auto by_change = make_power_state(random_start<N>(4), Vector<double, N>{}, Vector<double, N>{});
  const PowerConfig<double> config{.max_iters = 20000, .tol = tol};

  const auto residual_result = solve_eigen_problem(problem, make_power_strategy(L2Residual{}, config), by_residual);
  const auto change_result = solve_eigen_problem(problem, make_power_strategy(ChangeResidual{}, config), by_change);
  ASSERT_TRUE(residual_result.converged);
  ASSERT_TRUE(change_result.converged);

  EXPECT_NEAR(residual_result.eigenvalue, tridiagonal_eigenvalue(N, N), 1e-12);
  EXPECT_NEAR(change_result.eigenvalue, tridiagonal_eigenvalue(N, N), 1e-10);
  EXPECT_LE(norm(by_residual.r()), tol);
  EXPECT_GT(norm(by_change.r()), 1e3 * tol);
}

TEST(Eigenvalues, PowerChangePolicyRejectsAZeroImage) {
  // A q0 = 0 leaves nothing to normalize when the change policy requests another step.
  const Matrix3d nilpotent{0.0, 1.0, 0.0,  //
                           0.0, 0.0, 0.0,  //
                           0.0, 0.0, 0.0};
  auto problem = make_eigen_problem<mm_backend_t>(Matrix3d(nilpotent));
  auto state = make_power_state(Vector3d{1.0, 0.0, 0.0}, Vector3d{}, Vector3d{});
  EXPECT_THROW(solve_eigen_problem(problem, make_power_strategy(ChangeResidual{}, PowerConfig<double>{}), state),
               std::runtime_error);
}

TEST(Eigenvalues, ShiftedPowerRejectsAnUnconvergedDominantEigenvalue) {
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<3>(1.0));
  auto dominant = make_power_state(Vector3d{1.0, 0.3, -0.2}, Vector3d{}, Vector3d{});
  auto opposite = make_power_state(Vector3d{0.2, -0.7, 0.4}, Vector3d{}, Vector3d{});
  const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 1});
  EXPECT_THROW(solve_eigen_bounds(problem, strategy, dominant, opposite), std::runtime_error);
}
//@}

//! \name Lanczos coefficient storage and preconditioning
//@{

TEST(Eigenvalues, LanczosRejectsInsufficientAlphaStorage) {
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<3>(1.0));
  auto state = make_lanczos_state(Vector3d{1.0, 0.3, -0.2}, Vector3d{}, Vector3d{}, Vector3d{}, Vector<double, 5>{},
                                  Vector<double, 6>{});
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6});
  EXPECT_THROW(strategy.initialize(problem, state), std::invalid_argument);
}

TEST(Eigenvalues, LanczosRejectsInsufficientBetaStorage) {
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<3>(1.0));
  auto state = make_lanczos_state(Vector3d{1.0, 0.3, -0.2}, Vector3d{}, Vector3d{}, Vector3d{}, Vector<double, 6>{},
                                  Vector<double, 5>{});
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6});
  EXPECT_THROW(strategy.initialize(problem, state), std::invalid_argument);
}

TEST(Eigenvalues, PreconditionedLanczosRecoversTheOriginalSpectrum) {
  for (const double sign : {1.0, -1.0}) {
    expect_preconditioned_lanczos_bounds<4>(sign);
    expect_preconditioned_lanczos_bounds<8>(sign);
    expect_preconditioned_lanczos_bounds<16>(sign);
  }
}

TEST(Eigenvalues, KokkosPreconditionedLanczosRecoversTheSpectrumForEveryRepresentation) {
  for (const size_t n : {16, 100}) {
    const LanczosConfig<double> config{.max_iters = static_cast<unsigned>(10 * n), .tol = 1e-10};
    const auto preconditioner = make_congruence_preconditioner(n);
    const auto strategy = make_lanczos_strategy(config, preconditioner);
    for (const double sign : {1.0, -1.0}) {
      expect_kokkos_lanczos_representations(TridiagonalKokkosOp{n, sign, true}, strategy, config);
    }
  }
}

TEST(Eigenvalues, LanczosRejectsANegativeDefinitePreconditioner) {
  auto problem = make_eigen_problem<mm_backend_t>(tridiagonal<3>(1.0));
  auto state = make_lanczos_state(Vector3d{1.0, 0.3, -0.2}, Vector3d{}, Vector3d{}, Vector3d{}, Vector<double, 6>{},
                                  Vector<double, 6>{});
  const JacobiPreconditioner<mm_backend_t, Vector3d> negative(mm_backend_t{}, Vector3d{-1.0, -1.0, -1.0});
  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6}, negative);
  EXPECT_THROW(strategy.initialize(problem, state), std::invalid_argument);
}
//@}

}  // namespace

}  // namespace mundy
