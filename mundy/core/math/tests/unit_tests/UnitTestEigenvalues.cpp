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
///   - diag(1, 1/2, 1/4), whose power iterates and Rayleigh quotients are known exactly at every step.
///   - The 1-D Laplacian L = [-1, 2, -1] of size n: eigenvalues 4 sin^2(k pi / (2n + 2)) and unit eigenvectors
///     sqrt(2 / (n + 1)) sin(j k pi / (n + 1)), k, j = 1..n; ||L|| <= 4.
///   - The congruence D A D for D = diag(1 + i / n): with the preconditioner P = D^-2, P D A D = D^-1 A D is similar
///     to A, so its spectrum is A's, and D^-1 e is the unit eigenvector estimate of A for a P-unit estimate e.
///   - H diag(lambda) H for a Householder reflection H, eigenvalues lambda and eigenvectors H e_i, built in long double
///     and rounded once: diag(1, 2, 3) of size 3, and spectra of size 32: a separated top (1, 31/30, ..., 2, 10); the
///     same with a singular bottom (0, 1/30, ..., 1, 10); two separated ends (0.1, 1, 30/29, ..., 2, 10); and the
///     Chebyshev points 2 - cos((i + 1/2) pi / 32), clustered at both ends.
///   - diag(1, 1 + 1/n, ..., 2 - 2/n, top): top 10 at n = 1000, and top 2.1 at n = 10^5.
///   - The diagonal of the 19 Chebyshev points 2 - cos((i + 1/2) pi / 19), on which Lanczos from all ones gives the
///     Chebyshev Jacobi matrix.
///   - I + 1 1^T / n, whose top eigenvector is the all-ones vector.
///   - The Jacobi matrix J with diagonal (10, 2, ..., 2) and off-diagonal (0.1, 1, ..., 1), on which Lanczos from e_1
///     reproduces J exactly, one unit vector e_k per iteration.
///   - The quadratic form D^T M D of PowerLimits, a nilpotent 3 x 3 matrix, and Jacobi preconditioners that are
///     indefinite or negative.
///   - Lanczos's own tridiagonal T, whose eigenvalues and eigenvectors an implicit-QL solve gives independently.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

// Mundy
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS
#include <mundy_math/Matrix.hpp>
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Vector.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_math/eigenvalues.hpp>
#include <mundy_math/linear_ops.hpp>       // for mundy::make_quadratic_form
#include <mundy_math/preconditioners.hpp>  // for mundy::JacobiPreconditioner
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix
#include <mundy_utils/rng.hpp>           // for mundy::make_philox

namespace mundy {

namespace {

using mm_backend_t = MundyMathBackend;
using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using memory_space_t = Kokkos::DefaultExecutionSpace::memory_space;
using view_t = Kokkos::View<double*, memory_space_t>;
using host_array_t = Kokkos::View<double*, Kokkos::HostSpace>;
template <size_t K>
using view_basis_t = Kokkos::View<double* [K], Kokkos::LayoutLeft, memory_space_t>;
template <size_t N, size_t K>
using mundy_basis_t = Kokkos::Array<Vector<double, N>, K>;

constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double norm_L_bound = 4.0;     // ||L||_2 <= 4, by Gershgorin
constexpr double norm_DLD_bound = 16.0;  // ||D L D||_2 <= ||D||^2 ||L||_2 <= 4 * 4 for D <= 2; also >= ||L||_2

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
static_assert(kokkos_backend_t::is_scalar_array<host_array_t> &&
                  kokkos_backend_t::is_scalar_array<Kokkos::Array<double, 4>>,
              "KokkosBackend's driver runs on the host: host views and Kokkos::Arrays hold its scalars.");
static_assert(kokkos_backend_t::is_scalar_array<view_t> ==
                  Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space_t>::accessible,
              "A view the host cannot access cannot hold KokkosBackend's scalars.");
static_assert(!kokkos_backend_t::is_scalar_array<Kokkos::View<double*, Kokkos::LayoutStride, Kokkos::HostSpace>>,
              "A strided view does not hold contiguous scalars.");
static_assert(mm_backend_t::is_scalar_array<Kokkos::Array<double, 4>> &&
                  !mm_backend_t::is_scalar_array<Vector<double, 4>> && !mm_backend_t::is_scalar_array<host_array_t>,
              "MundyMathBackend's driver may run in a kernel: a Kokkos::Array holds its scalars.");
static_assert(kokkos_backend_t::num_columns<view_basis_t<5>>() == 5 &&
                  mm_backend_t::num_columns<mundy_basis_t<3, 5>>() == 5,
              "A multivector's column count is a compile-time constant.");
//@}

//! \name Reference matrices, operators, and start vectors
//@{

double laplacian_eigenvalue(size_t n, size_t k) {
  const double np1 = static_cast<double>(n + 1);
  const double s = std::sin(static_cast<double>(k) * Kokkos::numbers::pi_v<double> / (2.0 * np1));
  return 4.0 * s * s;
}

/// \brief L's unit eigenvector of eigenvalue laplacian_eigenvalue(N, k).
template <size_t N>
KOKKOS_INLINE_FUNCTION Vector<double, N> laplacian_eigenvector(size_t k) {
  constexpr double np1 = static_cast<double>(N + 1);
  Vector<double, N> v;
  for (size_t j = 0; j < N; ++j) {
    v[j] = Kokkos::sqrt(2.0 / np1) *
           Kokkos::sin(static_cast<double>(j + 1) * static_cast<double>(k) * Kokkos::numbers::pi_v<double> / np1);
  }
  return v;
}

/// \brief scale * L, for the 1-D Laplacian L = [-1, 2, -1].
template <size_t N>
KOKKOS_INLINE_FUNCTION Matrix<double, N, N> laplacian(double scale) {
  auto L = Matrix<double, N, N>::zeros();
  for (size_t i = 0; i < N; ++i) {
    L(i, i) = scale * 2.0;
    if (i > 0) L(i, i - 1) = -scale;
    if (i + 1 < N) L(i, i + 1) = -scale;
  }
  return L;
}

// d_i = 1 + i / n, the diagonal of D.
KOKKOS_INLINE_FUNCTION double congruence_scale(size_t n, size_t i) {
  return 1.0 + static_cast<double>(i) / static_cast<double>(n);
}

// Reproducible random vectors; each seed selects a different direction.
template <size_t N>
KOKKOS_INLINE_FUNCTION Vector<double, N> random_start(size_t seed) {
  Vector<double, N> q;
  for (size_t i = 0; i < N; ++i) {
    openrand::Philox rng = make_philox(seed, i);
    q[i] = rng.uniform<double>(-1.0, 1.0);
  }
  return q;
}

/// \brief D A D, where D has diagonal 1 + i / N.
template <size_t N>
Matrix<double, N, N> congruent(const Matrix<double, N, N>& A) {
  auto C = A;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      C(i, j) *= congruence_scale(N, i) * congruence_scale(N, j);
    }
  }
  return C;
}

/// \brief x / d_i, entry by entry: P^(1/2) x for P = D^-2.
template <size_t N>
Vector<double, N> divide_by_congruence_scale(const Vector<double, N>& x) {
  Vector<double, N> y;
  for (size_t i = 0; i < N; ++i) {
    y[i] = x[i] / congruence_scale(N, i);
  }
  return y;
}

/// \brief The separated top: bottom, bottom + 1/30, ..., bottom + 1, then 10.
Vector<double, 32> separated_top_spectrum(double bottom) {
  Vector<double, 32> lambda;
  for (size_t i = 0; i < 31; ++i) {
    lambda[i] = bottom + static_cast<double>(i) / 30.0;
  }
  lambda[31] = 10.0;
  return lambda;
}

/// \brief Two separated ends: 0.1, then 1, 30/29, ..., 2, then 10.
Vector<double, 32> separated_ends_spectrum() {
  Vector<double, 32> lambda;
  lambda[0] = 0.1;
  for (size_t i = 0; i < 30; ++i) {
    lambda[i + 1] = 1.0 + static_cast<double>(i) / 29.0;
  }
  lambda[31] = 10.0;
  return lambda;
}

/// \brief The Chebyshev points 2 - cos((i + 1/2) pi / 32), ascending.
Vector<double, 32> chebyshev_spectrum() {
  Vector<double, 32> lambda;
  for (size_t i = 0; i < 32; ++i) {
    lambda[i] = 2.0 - std::cos((static_cast<double>(i) + 0.5) * Kokkos::numbers::pi_v<double> / 32.0);
  }
  return lambda;
}

/// \brief The fixtures whose certificates are checked, each a spectrum in ascending order.
std::vector<std::pair<const char*, Vector<double, 32>>> rotated_spectra() {
  return {{"separated top", separated_top_spectrum(1.0)},
          {"singular separated top", separated_top_spectrum(0.0)},
          {"Chebyshev", chebyshev_spectrum()}};
}

/// \brief H e_i in long double, with H = I - 2 u u^T / u^T u for a fixed u.
template <size_t N>
std::array<long double, N> householder_column(size_t i) {
  const auto u = random_start<N>(11);
  long double u_squared = 0.0L;
  for (size_t j = 0; j < N; ++j) {
    u_squared += static_cast<long double>(u[j]) * u[j];
  }
  const long double scale = -2.0L * u[i] / u_squared;
  std::array<long double, N> h;
  for (size_t j = 0; j < N; ++j) {
    h[j] = scale * u[j];
  }
  h[i] += 1.0L;
  return h;
}

/// \brief The unit eigenvector H e_i of rotated(lambda) for lambda_i.
template <size_t N>
Vector<double, N> rotated_eigenvector(size_t i) {
  const auto h = householder_column<N>(i);
  Vector<double, N> v;
  for (size_t j = 0; j < N; ++j) {
    v[j] = static_cast<double>(h[j]);
  }
  return v;
}

/// \brief The eigenvector of 2 on the separated top, plus 10^-3 of the top eigenvector and of a random vector.
template <size_t N>
Vector<double, N> near_second_eigenvector() {
  return rotated_eigenvector<N>(N - 2) + 1e-3 * rotated_eigenvector<N>(N - 1) + 1e-3 * random_start<N>(13);
}

/// \brief H diag(lambda) H, eigenvalues lambda and eigenvectors H e_i along no axis, rounded once from long double.
template <size_t N>
Matrix<double, N, N> rotated(const Vector<double, N>& lambda) {
  std::vector<long double> sum(N * N, 0.0L);
  for (size_t m = 0; m < N; ++m) {
    const auto h = householder_column<N>(m);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        sum[i * N + j] += lambda[m] * h[i] * h[j];
      }
    }
  }
  Matrix<double, N, N> A;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      A(i, j) = static_cast<double>(sum[i * N + j]);
    }
  }
  return A;
}

/// \brief How far the rounded rotated(lambda) can move an eigenvalue: N eps max |lambda| bounds ||A - fl(A)||_F.
template <size_t N>
double construction_error(const Vector<double, N>& lambda) {
  return N * eps * std::max(std::abs(lambda[0]), std::abs(lambda[N - 1]));
}

/// \brief I + 1 1^T / N: eigenvalue 2 on the all-ones vector and 1 on its complement.
template <size_t N>
Matrix<double, N, N> ones_rank_one_update() {
  auto A = Matrix<double, N, N>::identity();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      A(i, j) += 1.0 / static_cast<double>(N);
    }
  }
  return A;
}

/// \brief y = J x for the Jacobi matrix J, diagonal (10, 2, ..., 2) and off-diagonal (0.1, 1, ..., 1), with the last
/// entry of y set to +infinity whenever x reaches it; a function of x alone.
template <size_t N>
struct BlowupMundyOp {
  KOKKOS_INLINE_FUNCTION size_t domain_size() const {
    return N;
  }
  KOKKOS_INLINE_FUNCTION size_t range_size() const {
    return N;
  }
  KOKKOS_INLINE_FUNCTION void apply(const Vector<double, N>& x, Vector<double, N>& y) const {
    for (size_t i = 0; i < N; ++i) {
      const double below = i == 0 ? 0.0 : (i == 1 ? 0.1 : 1.0);
      const double above = i + 1 == N ? 0.0 : (i == 0 ? 0.1 : 1.0);
      y[i] = (i == 0 ? 10.0 : 2.0) * x[i];
      if (i > 0) y[i] += below * x[i - 1];
      if (i + 1 < N) y[i] += above * x[i + 1];
    }
    if (x[N - 1] != 0.0) y[N - 1] = std::numeric_limits<double>::infinity();
  }
};

// Matrix-free sign L, or sign D L D when congruent, for each backend.
template <size_t N>
struct LaplacianMundyOp {
  double sign;
  bool congruent = false;

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

struct LaplacianKokkosOp {
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
    return view_t("laplacian_domain", n);
  }
  view_t make_range_vector() const {
    return view_t("laplacian_range", n);
  }
  void apply(const view_t& x, view_t& y) const {
    const size_t size = n;
    const double s = sign;
    const bool scaled = congruent;
    Kokkos::parallel_for(
        "LaplacianKokkosOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, size),
        KOKKOS_LAMBDA(const size_t i) {
          const auto d = [=](size_t j) { return scaled ? congruence_scale(size, j) : 1.0; };
          double yi = 2.0 * d(i) * x(i);
          if (i > 0) yi -= d(i - 1) * x(i - 1);
          if (i + 1 < size) yi -= d(i + 1) * x(i + 1);
          y(i) = s * d(i) * yi;
        });
  }
};

/// \brief diag(d) for KokkosBackend.
struct DiagonalKokkosOp {
  view_t d;

  size_t domain_size() const {
    return d.extent(0);
  }
  size_t range_size() const {
    return d.extent(0);
  }
  void apply(const view_t& x, view_t& y) const {
    const view_t diagonal = d;
    Kokkos::parallel_for(
        "DiagonalKokkosOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, diagonal.extent(0)),
        KOKKOS_LAMBDA(const size_t i) { y(i) = diagonal(i) * x(i); });
  }
};

/// \brief A dense matrix applied entry by entry, for KokkosBackend.
struct DenseKokkosOp {
  Kokkos::View<double**, Kokkos::LayoutRight, memory_space_t> A;

  size_t domain_size() const {
    return A.extent(1);
  }
  size_t range_size() const {
    return A.extent(0);
  }
  void apply(const view_t& x, view_t& y) const {
    const auto matrix = A;
    Kokkos::parallel_for(
        "DenseKokkosOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, matrix.extent(0)),
        KOKKOS_LAMBDA(const size_t i) {
          double sum = 0.0;
          for (size_t j = 0; j < matrix.extent(1); ++j) {
            sum += matrix(i, j) * x(j);
          }
          y(i) = sum;
        });
  }
};

template <size_t N>
DenseKokkosOp to_kokkos(const Matrix<double, N, N>& M) {
  DenseKokkosOp op{Kokkos::View<double**, Kokkos::LayoutRight, memory_space_t>("dense", N, N)};
  const auto host = Kokkos::create_mirror_view(op.A);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      host(i, j) = M(i, j);
    }
  }
  Kokkos::deep_copy(op.A, host);
  return op;
}

/// \brief The diagonal 1, 1 + 1/n, ..., 2 - 2/n, then top.
view_t separated_top_diagonal(size_t n, double top) {
  view_t d("d", n);
  Kokkos::parallel_for(
      "separated_top_diagonal", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const size_t i) {
        d(i) = i + 1 == n ? top : 1.0 + static_cast<double>(i) / static_cast<double>(n);
      });
  return d;
}

view_t random_start_view(size_t n, size_t seed) {
  view_t q("q0", n);
  Kokkos::parallel_for(
      "random_start_view", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n), KOKKOS_LAMBDA(const size_t i) {
        openrand::Philox rng = make_philox(seed, i);
        q(i) = rng.uniform<double>(-1.0, 1.0);
      });
  return q;
}

/// \brief x as a device view.
template <size_t N>
view_t to_view(const Vector<double, N>& x) {
  view_t v("v", N);
  const auto host = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < N; ++i) {
    host(i) = x[i];
  }
  Kokkos::deep_copy(v, host);
  return v;
}

/// \brief x on the host.
std::vector<double> to_host(const view_t& x) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  return std::vector<double>(host.data(), host.data() + host.extent(0));
}

template <size_t N>
Vector<double, N> to_vector(const view_t& x) {
  const auto host = to_host(x);
  Vector<double, N> v;
  for (size_t i = 0; i < N; ++i) {
    v[i] = host[i];
  }
  return v;
}

// P = D^-2 makes P D A D similar to A. Supply D^2 to the componentwise division operator; this is a chosen positive
// preconditioner, not the diagonal of D A D.
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
using dense_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, memory_space_t>;
using sparse_matrix_t =
    KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::DefaultExecutionSpace, memory_space_t>, void, size_t>;

/// \brief A dense representation of the matrix-free Laplacian problem.
dense_matrix_t make_dense_laplacian(const LaplacianKokkosOp& op) {
  dense_matrix_t dense("laplacian", op.n, op.n);
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

//! \name Independent references for Lanczos's T
//@{

/// \brief The eigenvalues, ascending, of the leading k x k block of the symmetric tridiagonal (alpha, beta).
///
/// By the implicit QL method in long double, so that its rounding lies below double's; with last, also the last entry
/// of each unit eigenvector, in the same order.
template <class AlphaArray, class BetaArray>
std::vector<double> eigen_of_t(const AlphaArray& alpha, const BetaArray& beta, size_t k,
                                      std::vector<double>* last = nullptr) {
  using real = long double;
  constexpr real precision = std::numeric_limits<real>::epsilon();
  std::vector<real> d(k), e(k, 0.0L), z(k, 0.0L);  // z: the last row of the eigenvector matrix
  for (size_t i = 0; i < k; ++i) {
    d[i] = alpha[i];
  }
  for (size_t i = 0; i + 1 < k; ++i) {
    e[i] = beta[i];
  }
  z[k - 1] = 1.0L;
  for (size_t l = 0; l < k; ++l) {
    for (int sweep = 0;; ++sweep) {
      size_t m = l;
      for (; m + 1 < k; ++m) {
        if (std::abs(e[m]) <= precision * (std::abs(d[m]) + std::abs(d[m + 1]))) break;
      }
      if (m == l) break;
      if (sweep == 100) throw std::runtime_error("eigen_of_t: the QL sweeps did not converge.");
      real g = (d[l + 1] - d[l]) / (2.0L * e[l]);
      real r = std::hypot(g, 1.0L);
      g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
      real s = 1.0L, c = 1.0L, p = 0.0L;
      bool deflated = false;
      for (size_t i = m; i-- > l;) {
        const real f = s * e[i];
        const real b = c * e[i];
        r = std::hypot(f, g);
        e[i + 1] = r;
        if (r == 0.0L) {
          d[i + 1] -= p;
          e[m] = 0.0L;
          deflated = true;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2.0L * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;
        const real t = z[i + 1];
        z[i + 1] = s * z[i] + c * t;
        z[i] = c * z[i] - s * t;
      }
      if (deflated) continue;
      d[l] -= p;
      e[l] = g;
      e[m] = 0.0L;
    }
  }
  std::vector<size_t> order(k);
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return d[a] < d[b]; });
  std::vector<double> sorted(k);
  if (last != nullptr) last->assign(k, 0.0);
  for (size_t j = 0; j < k; ++j) {
    sorted[j] = static_cast<double>(d[order[j]]);
    if (last != nullptr) (*last)[j] = static_cast<double>(z[order[j]]);
  }
  return sorted;
}

/// \brief An end of the leading k x k block of a state's T by the dense reference: its extreme eigenvalue, the Ritz
/// residual beta_k |s_k| of that eigenvalue's unit eigenvector s, and the distance to the next eigenvalue.
struct ReferenceEnd {
  double eigenvalue;
  double residual;
  double gap;
};

template <class State>
ReferenceEnd reference_end(const State& state, unsigned k, bool largest) {
  std::vector<double> last;
  const auto theta = eigen_of_t(state.alpha(), state.beta(), k, &last);
  const size_t j = largest ? k - 1 : 0;
  const double gap = k == 1 ? std::numeric_limits<double>::infinity() : std::abs(theta[j] - theta[largest ? k - 2 : 1]);
  return {theta[j], state.beta()[k - 1] * std::abs(last[j]), gap};
}

/// \brief Gershgorin's bound max_i |alpha_i| + beta_{i-1} + beta_i on the norm of T_k, the leading k x k block of T,
/// for beta_0 = beta_k = 0.
double norm_of_t(const double* alpha, const double* beta, unsigned k) {
  double norm = 0.0;
  for (unsigned i = 0; i < k; ++i) {
    norm = std::max(norm, std::abs(alpha[i]) + (i > 0 ? beta[i - 1] : 0.0) + (i + 1 < k ? beta[i] : 0.0));
  }
  return norm;
}

//@}

//! \name Comparisons
//@{

/// \brief The distance from theta to the nearest of the eigenvalues lambda.
template <class Spectrum>
double distance_to_spectrum(double theta, const Spectrum& lambda, size_t n) {
  double nearest = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < n; ++i) {
    nearest = std::min(nearest, std::abs(theta - lambda[i]));
  }
  return nearest;
}

/// \brief The iterations of a solve, its state's iter() at the stop: the larger num_iters of its ends, since the solve
/// stops at the measurement that settles its last end (an unwanted end has none).
unsigned stop_of(const EigenBounds<double>& result) {
  return std::max(result.largest.num_iters, result.smallest.num_iters);
}

/// \brief Whether every entry of x equals the same entry of y, bit for bit or both NaN.
template <class X, class Y>
bool same_entries(const X& x, const Y& y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!(x[i] == y[i] || (std::isnan(x[i]) && std::isnan(y[i])))) return false;
  }
  return true;
}
//@}

//! \name Building Lanczos states and strategies
//@{

/// \brief A MundyMathBackend state's Bv: a vector of size N with a preconditioner, NoVector without one.
template <size_t N, bool Preconditioned>
KOKKOS_INLINE_FUNCTION auto mundy_Bv() {
  if constexpr (Preconditioned) {
    return Vector<double, N>{};
  } else {
    return NoVector{};
  }
}

/// \brief A KokkosBackend state's Bv: a view of size n with a preconditioner, NoVector without one.
template <bool Preconditioned>
auto view_Bv(size_t n) {
  if constexpr (Preconditioned) {
    return view_t("Bv", n);
  } else {
    return NoVector{};
  }
}

/// \brief A MundyMathBackend state of size N with room for K iterations, without estimate vectors.
template <size_t N, size_t K, bool Preconditioned = false>
KOKKOS_INLINE_FUNCTION auto mundy_state() {
  return make_lanczos_state(Vector<double, N>{}, Vector<double, N>{}, Vector<double, N>{},
                            mundy_Bv<N, Preconditioned>(),
                            Kokkos::Array<double, 4 * K>{});
}

/// \brief The same with an estimate vector, or NoVector, for each end, and a basis; all three are held by reference.
template <size_t N, size_t K, bool Preconditioned = false, class Largest, class Smallest, class Basis>
KOKKOS_INLINE_FUNCTION auto mundy_state(Largest&& largest, Smallest&& smallest, Basis&& basis) {
  constexpr size_t e = !is_no_vector_v<Largest> + !is_no_vector_v<Smallest>;
  return make_lanczos_state(Vector<double, N>{}, Vector<double, N>{}, Vector<double, N>{},
                            mundy_Bv<N, Preconditioned>(),
                            Kokkos::Array<double, (4 + e) * K>{}, std::forward<Largest>(largest),
                            std::forward<Smallest>(smallest), std::forward<Basis>(basis));
}

/// \brief A KokkosBackend state of size n with room for k iterations, without estimate vectors.
template <bool Preconditioned = false>
auto view_state(size_t n, size_t k) {
  return make_lanczos_state(view_t("v", n), view_t("Bv_prev", n), view_t("Av", n), view_Bv<Preconditioned>(n),
                            host_array_t("scalars", 4 * k));
}

/// \brief The same with an estimate vector, or NoVector, for each end, and a basis; all three are held by reference.
template <bool Preconditioned = false, class Largest, class Smallest, class Basis>
auto view_state(size_t n, size_t k, Largest&& largest, Smallest&& smallest, Basis&& basis) {
  constexpr size_t e = !is_no_vector_v<Largest> + !is_no_vector_v<Smallest>;
  return make_lanczos_state(view_t("v", n), view_t("Bv_prev", n), view_t("Av", n), view_Bv<Preconditioned>(n),
                            host_array_t("scalars", (4 + e) * k), std::forward<Largest>(largest),
                            std::forward<Smallest>(smallest), std::forward<Basis>(basis));
}

/// \brief The Lanczos strategy of config for MundyMathBackend vectors of size N, with P = D^-2 when Preconditioned.
template <size_t N, bool Preconditioned>
auto mundy_lanczos_strategy(const LanczosConfig<double>& config) {
  if constexpr (Preconditioned) {
    return make_lanczos_strategy(config, make_congruence_preconditioner<N>());
  } else {
    return make_lanczos_strategy(config);
  }
}

/// \brief The same for KokkosBackend vectors of size n.
template <bool Preconditioned>
auto view_lanczos_strategy(const LanczosConfig<double>& config, size_t n) {
  if constexpr (Preconditioned) {
    return make_lanczos_strategy(config, make_congruence_preconditioner(n));
  } else {
    return make_lanczos_strategy(config);
  }
}

/// \brief The cold start c for vectors of size N: a solve of no iterations replaces a zero estimate by c and, with no
/// T to measure, returns it there.
template <size_t N, class Problem>
Vector<double, N> cold_start_of(const Problem& problem) {
  auto cold = Vector<double, N>::zeros();
  mundy_basis_t<N, 1> basis{};
  auto state = mundy_state<N, 1>(cold, NoVector{}, basis);
  solve_eigen_problem(problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = 0, .smallest = false}), state);
  return cold;
}
//@}

//! \name Device kernels and allocation counts
//@{
// KOKKOS_LAMBDA cannot sit in a gtest body under CUDA; each kernel writes what its test then checks on the host.

using bounds_t = Kokkos::View<double* [2], memory_space_t>;
using flags_t = Kokkos::View<bool*, memory_space_t>;

/// \brief Entity e's shifted-power bounds of (e + 1) L: (largest, smallest) eigenvalues and whether both converged.
void solve_power_bounds_in_kernel(const bounds_t& eigenvalues, const flags_t& converged) {
  constexpr size_t N = 4;
  Kokkos::parallel_for(
      "solve_power_bounds_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, eigenvalues.extent(0)),
      KOKKOS_LAMBDA(const size_t e) {
        auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(static_cast<double>(e + 1)));
        auto dominant = make_power_state(random_start<N>(2 * e + 1), Vector<double, N>{}, Vector<double, N>{});
        auto opposite = make_power_state(random_start<N>(2 * e + 2), Vector<double, N>{}, Vector<double, N>{});
        const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 20000, .tol = 1e-12});
        const auto bounds = solve_eigen_bounds(problem, strategy, dominant, opposite);
        eigenvalues(e, 0) = bounds.largest.eigenvalue;
        eigenvalues(e, 1) = bounds.smallest.eigenvalue;
        converged(e) = bounds.largest.converged && bounds.smallest.converged;
      });
}

/// \brief The columns of solve_lanczos_ends_in_kernel's output, one row per entity.
struct LanczosKernelColumn {
  enum : size_t {
    largest_eigenvalue,
    largest_residual,
    largest_num_iters,
    smallest_eigenvalue,
    smallest_residual,
    both_converged,  // 1 or 0
    top_overlap,     // |e^T q_N| for the estimate e and the top eigenvector q_N
    count
  };
};

/// \brief Entity e's two-end Lanczos solve of (e + 1) L from the cold start, with Kokkos::Array scalars and an
/// estimate of the largest end formed over a basis of 8.
void solve_lanczos_ends_in_kernel(const Kokkos::View<double* [LanczosKernelColumn::count], memory_space_t>& out) {
  using column = LanczosKernelColumn;
  constexpr size_t N = 4;
  constexpr size_t K = 16;
  Kokkos::parallel_for(
      "solve_lanczos_ends_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, out.extent(0)),
      KOKKOS_LAMBDA(const size_t e) {
        auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(static_cast<double>(e + 1)));
        auto estimate = Vector<double, N>::zeros();
        mundy_basis_t<N, 8> basis{};
        auto state = mundy_state<N, K>(estimate, NoVector{}, basis);
        const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = K, .tol = 1e-12});
        const auto result = solve_eigen_problem(problem, strategy, state);
        out(e, column::largest_eigenvalue) = result.largest.eigenvalue;
        out(e, column::largest_residual) = result.largest.residual;
        out(e, column::largest_num_iters) = result.largest.num_iters;
        out(e, column::smallest_eigenvalue) = result.smallest.eigenvalue;
        out(e, column::smallest_residual) = result.smallest.residual;
        out(e, column::both_converged) = result.largest.converged && result.smallest.converged ? 1.0 : 0.0;
        out(e, column::top_overlap) = Kokkos::abs(dot(estimate, laplacian_eigenvector<N>(N)));
      });
}

/// \brief The Kokkos allocations made while f runs.
template <class F>
size_t count_allocations(F&& f) {
  static size_t count = 0;
  count = 0;
  Kokkos::Tools::Experimental::set_init_callback(
      [](const int, const uint64_t, const uint32_t, Kokkos_Profiling_KokkosPDeviceInfo*) {});
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](const Kokkos_Profiling_SpaceHandle, const char*, const void*, const uint64_t) { ++count; });
  f();
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_init_callback(nullptr);
  return count;
}
//@}

//! \name Power method
//@{

TEST(Eigenvalues, PowerIterates) {
  {
    SCOPED_TRACE("diagonal closed form");
    // Setup: for q0 = (1, 1, 1) on diag(d), q before step k is proportional to d^(k-1); A q is exact for d a power of 2
    constexpr size_t n = 3;
    const Vector3d d{1.0, 0.5, 0.25};
    const Matrix3d A{d[0], 0.0,  0.0,  //
                     0.0,  d[1], 0.0,  //
                     0.0,  0.0,  d[2]};
    auto problem = make_eigen_problem<mm_backend_t>(Matrix3d(A));
    auto state = make_power_state(Vector3d{1.0, 1.0, 1.0}, Vector3d{}, Vector3d{});
    const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 21, .tol = 0.0});

    // Iterate and compare
    strategy.initialize(problem, state);
    for (unsigned k = 1; k <= 20; ++k) {
      SCOPED_TRACE(::testing::Message() << "iteration " << k);
      double numerator = 0.0;
      double denominator = 0.0;
      Vector3d q_prev;
      Vector3d expected_next;
      for (size_t i = 0; i < 3; ++i) {
        numerator += std::pow(d[i], 2 * k - 1);
        denominator += std::pow(d[i], 2 * k - 2);
        q_prev[i] = std::pow(d[i], k - 1);
        expected_next[i] = std::pow(d[i], k);
      }
      q_prev = q_prev / norm(q_prev);
      expected_next = expected_next / norm(expected_next);
      const double lambda = numerator / denominator;
      const Vector3d r = A * q_prev - lambda * q_prev;
      const double residual = norm(r);
      double q_dot_r = 0.0;  // sum_i |q_i| |r_i|
      for (size_t i = 0; i < 3; ++i) {
        q_dot_r += std::abs(q_prev[i]) * std::abs(r[i]);
      }

      // To first order, for u = eps / 2. The entries of q and of A q are powers of 2 times one significand, so q's
      // direction is exact and only norms round: q's by (n/2 + 3) u (n-term dot, sqrt, reciprocal, scaling), the
      // reference's by ((n - 1)/2 + 3) u (its squares are exact). lambda = q^T A q: the n-term dot, n u, q's squared
      // norm, (n + 6) u, and the reference's two sums and quotient, (2n - 1) u. The residual: r is orthogonal to q, so
      // lambda's error moves it only to second order. Relative to |r|, the norms of q round by (n + 5.5) u together,
      // each side's subtraction by u and its norm by (n/2 + 1) u; the product lambda q rounds by u lambda per side
      // along r, sum |q_i| |r_i| / |r|.
      const double u = 0.5 * eps;
      const double q_floor = (n / 2.0 + 3 + (n - 1) / 2.0 + 3) * u;
      const double lambda_bound = (4 * n + 5) * u * lambda;
      const double residual_bound = (2 * n + 9.5) * u * residual + 2 * u * lambda * q_dot_r / residual;
      ASSERT_FALSE(strategy.iterate(problem, state));
      EXPECT_EQ(state.iter(), k);
      EXPECT_NEAR(state.eigenvalue(), lambda, lambda_bound);
      EXPECT_NEAR(state.residual(), residual, residual_bound);
      for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(state.q()[i], expected_next[i], q_floor) << "component " << i;
      }
    }
  }
  {
    SCOPED_TRACE("contraction at the spectral ratio");
    // Setup: after the faster modes decay, successive residuals approach lambda_(N-1) / lambda_N
    constexpr size_t N = 8;
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(1.0));
    auto state = make_power_state(random_start<N>(7), Vector<double, N>{}, Vector<double, N>{});
    const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 150, .tol = 0.0});
    const double rate = laplacian_eigenvalue(N, N - 1) / laplacian_eigenvalue(N, N);

    // Iterate
    strategy.initialize(problem, state);
    std::vector<double> residuals;
    for (unsigned k = 0; k < 150; ++k) {
      ASSERT_FALSE(strategy.iterate(problem, state)) << "iteration " << k;
      ASSERT_EQ(state.iter(), k + 1);
      residuals.push_back(state.residual());
    }

    // Compare the late ratios. With q_k proportional to sum_j a_j lambda_j^k u_j, the ratio is rate to within
    // rate (x^2 / 2 + e^2 (1 + x^2)), for e = |a_{N-1} / a_N| rate^k the second mode's weight and x^2 the lower modes'
    // weights, relative to the second's, scaled by their distance to lambda_N. Rounding, to first order for
    // u = eps / 2: a residual is measured to within (2 ||L|| + lambda_N) u (the apply's 2 additions per row, the
    // product lambda q) plus (N + 5) u of itself (q's norm, the subtraction, the norm); a step turns q by at most
    // phi = (1 + 2 ||L|| / lambda_N) u (the apply, the scaling), which moves the next residual by ||L|| phi; the
    // reference rate rounds by 23 u and the ratio by u. Noise accumulated in q contracts at the signal's rates, so it
    // cancels from the ratio: along the second mode it changes e by a relative u / e, inside the e^2 term's slack over
    // its leading e^2 (1 - rate^2); along lower modes it adds terms below the injection's by factors of order x and
    // u / e.
    const auto start = random_start<N>(7);
    const double lambda_top = laplacian_eigenvalue(N, N);
    const double gap = lambda_top - laplacian_eigenvalue(N, N - 1);
    const double a_top = dot(start, laplacian_eigenvector<N>(N));
    const double a_second = dot(start, laplacian_eigenvector<N>(N - 1));
    const double u = 0.5 * eps;
    const double measurement = (2 * norm_L_bound + lambda_top) * u;
    const double injection = norm_L_bound * (1 + 2 * norm_L_bound / lambda_top) * u;
    const double relative_rounding = (2 * (N + 5) + 23 + 1) * u;
    for (size_t k = 100; k + 1 < residuals.size(); ++k) {
      const double e = std::abs(a_second / a_top) * std::pow(rate, static_cast<double>(k));
      double x_squared = 0.0;
      for (size_t j = 1; j + 1 < N; ++j) {
        const double lambda_j = laplacian_eigenvalue(N, j);
        const double weight = dot(start, laplacian_eigenvector<N>(j)) / a_second *
                              std::pow(lambda_j / laplacian_eigenvalue(N, N - 1), static_cast<double>(k)) *
                              (lambda_top - lambda_j) / gap;
        x_squared += weight * weight;
      }
      const double ratio_bound = rate * (x_squared / 2 + e * e * (1 + x_squared) +
                                         (2 * measurement + injection) / residuals[k + 1] + relative_rounding);
      EXPECT_NEAR(residuals[k + 1] / residuals[k], rate, ratio_bound) << "iteration " << k;
    }
  }
}

TEST(Eigenvalues, PowerLimits) {
  {
    SCOPED_TRACE("iteration limit");
    constexpr size_t N = 16;
    constexpr unsigned max_iters = 3;
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(1.0));
    auto manual = make_power_state(random_start<N>(5), Vector<double, N>{}, Vector<double, N>{});
    auto solved = make_power_state(random_start<N>(5), Vector<double, N>{}, Vector<double, N>{});
    const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = max_iters, .tol = 1e-12});

    // Step by hand, then solve
    strategy.initialize(problem, manual);
    for (unsigned k = 1; k <= max_iters; ++k) {
      ASSERT_FALSE(strategy.iterate(problem, manual));
      ASSERT_EQ(manual.iter(), k);
    }
    const auto result = solve_eigen_problem(problem, strategy, solved);

    // The result is the last iterate, measured at the returned q
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.num_iters, max_iters);
    EXPECT_EQ(solved.iter(), manual.iter());
    EXPECT_DOUBLE_EQ(result.eigenvalue, manual.eigenvalue());
    EXPECT_DOUBLE_EQ(result.residual, manual.residual());
    EXPECT_GT(result.residual, 1e-12);
    EXPECT_LT(result.eigenvalue, laplacian_eigenvalue(N, N));
    const auto A_q = laplacian<N>(1.0) * solved.q();
    const double lambda = dot(solved.q(), A_q);
    EXPECT_DOUBLE_EQ(result.eigenvalue, lambda);
    EXPECT_DOUBLE_EQ(result.residual, norm(A_q - lambda * solved.q()));
  }
  {
    SCOPED_TRACE("no iterations");
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<3>(1.0));
    auto state = make_power_state(Vector3d{1.0, 0.3, -0.2}, Vector3d{}, Vector3d{});
    const auto result =
        solve_eigen_problem(problem, make_power_strategy(PowerConfig<double>{.max_iters = 0}), state);
    EXPECT_EQ(result.num_iters, 0u);
    EXPECT_TRUE(std::isnan(result.residual));
    EXPECT_TRUE(std::isnan(result.eigenvalue));
    EXPECT_FALSE(result.converged);
  }
  {
    SCOPED_TRACE("ends on an apply at its eigenvector");
    // Setup: D^T M D, whose workspace holds f = D q and u = M f
    const Matrix3d D{2.0, 0.5, 0.0,  //
                     0.0, 1.0, 0.5,  //
                     0.0, 0.0, 1.5};
    const Matrix3d M{2.0, -1.0, 0.0,   //
                     -1.0, 2.0, -1.0,  //
                     0.0, -1.0, 2.0};
    const auto A = make_quadratic_form<mm_backend_t>(transpose(D), M, D);
    for (const unsigned max_iters : {3u, 2000u}) {
      SCOPED_TRACE(::testing::Message() << "max_iters " << max_iters);
      Vector3d f, u;
      const auto problem = EigenProblem(mm_backend_t{}, A, A.make_workspace(f, u));
      auto state = make_power_state(Vector3d{1.0, 1.0, 1.0}, Vector3d{}, Vector3d{});
      const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = max_iters, .tol = 1e-10});
      const auto result = solve_eigen_problem(problem, strategy, state);
      EXPECT_EQ(result.converged, max_iters == 2000u);

      // The workspace holds an apply at the returned q
      Vector3d f_at_q, u_at_q, A_q;
      auto at_q = A.make_workspace(f_at_q, u_at_q);
      mm_backend_t::apply(A, state.q(), A_q, at_q);
      for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(f[i], f_at_q[i]) << "component " << i;
        EXPECT_EQ(u[i], u_at_q[i]) << "component " << i;
      }
    }
  }
}

// Both ends of sign L, as a matrix and matrix-free on each backend and inside a kernel: the dominant end comes first
// from the power method, and the bounds order the ends by sign. A converged end's Rayleigh quotient is within
// ||r||^2 / gap of an eigenvalue (Kato-Temple), negligible at ||r|| <= tol, so its error is the rounding of q^T A q,
// far below tol.
TEST(Eigenvalues, PowerEndpoints) {
  {
    SCOPED_TRACE("MundyMathBackend");
    const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 20000, .tol = 1e-12});
    const auto check = [&]<size_t N>(const auto& A) {
      using op_t = std::remove_cvref_t<decltype(A)>;
      auto problem = make_eigen_problem<mm_backend_t>(op_t(A));
      auto dominant = make_power_state(random_start<N>(1), Vector<double, N>{}, Vector<double, N>{});
      auto opposite = make_power_state(random_start<N>(2), Vector<double, N>{}, Vector<double, N>{});
      const auto bounds = solve_eigen_bounds(problem, strategy, dominant, opposite);
      return std::tuple{bounds, std::abs(dot(dominant.q(), laplacian_eigenvector<N>(N))),
                        std::abs(dot(opposite.q(), laplacian_eigenvector<N>(1)))};
    };
    const auto expect_ends = [](const auto& solved, size_t n, double sign) {
      const auto& [bounds, dominant_overlap, opposite_overlap] = solved;
      // 1 - |q^T u| <= sin^2(q, u) <= (||r|| / gap)^2, with ||r|| at most tol plus its rounding on the shifted operator
      // (norm <= 2 ||L||), and the rounding of q's norm, of the n-term dot, and of u's sines, whose arguments reach
      // n pi
      const double tol = 1e-12;
      const double gap = laplacian_eigenvalue(n, n) - laplacian_eigenvalue(n, n - 1);
      const double residual = tol + (1.5 * n + 3) * eps * 2.0 * norm_L_bound;
      const double overlap_floor =
          (residual / gap) * (residual / gap) +
          (1.5 * n + 3 + std::sqrt(2.0) * (3.0 * Kokkos::numbers::pi_v<double> * n + 4)) * eps;
      ASSERT_TRUE(bounds.largest.converged);
      ASSERT_TRUE(bounds.smallest.converged);
      EXPECT_NEAR(bounds.largest.eigenvalue,
                  sign > 0.0 ? laplacian_eigenvalue(n, n) : -laplacian_eigenvalue(n, 1), tol);
      EXPECT_NEAR(bounds.smallest.eigenvalue,
                  sign > 0.0 ? laplacian_eigenvalue(n, 1) : -laplacian_eigenvalue(n, n), tol);
      EXPECT_NEAR(dominant_overlap, 1.0, overlap_floor);
      EXPECT_NEAR(opposite_overlap, 1.0, overlap_floor);
    };
    for (const double sign : {1.0, -1.0}) {
      SCOPED_TRACE(::testing::Message() << "sign " << sign);
      expect_ends(check.template operator()<4>(laplacian<4>(sign)), 4, sign);
      expect_ends(check.template operator()<4>(LaplacianMundyOp<4>{sign}), 4, sign);
      expect_ends(check.template operator()<8>(laplacian<8>(sign)), 8, sign);
      expect_ends(check.template operator()<8>(LaplacianMundyOp<8>{sign}), 8, sign);
      expect_ends(check.template operator()<16>(laplacian<16>(sign)), 16, sign);
      expect_ends(check.template operator()<16>(LaplacianMundyOp<16>{sign}), 16, sign);
    }
  }
  {
    SCOPED_TRACE("KokkosBackend");
    constexpr size_t n = 16;
    const PowerConfig<double> config{.max_iters = 20000, .tol = 1e-12};
    for (const double sign : {1.0, -1.0}) {
      SCOPED_TRACE(::testing::Message() << "sign " << sign);
      const LaplacianKokkosOp op{n, sign};
      const auto check = [&](const auto& A, const char* representation) {
        SCOPED_TRACE(representation);
        using op_t = std::remove_cvref_t<decltype(A)>;
        auto problem = make_eigen_problem<kokkos_backend_t>(op_t(A));
        auto dominant = make_power_state(random_start_view(n, 5), view_t("z", n), view_t("r", n));
        auto opposite = make_power_state(random_start_view(n, 6), view_t("z", n), view_t("r", n));
        const auto bounds = solve_eigen_bounds(problem, make_power_strategy(config), dominant, opposite);
        ASSERT_TRUE(bounds.largest.converged);
        ASSERT_TRUE(bounds.smallest.converged);
        EXPECT_NEAR(bounds.largest.eigenvalue,
                    sign > 0.0 ? laplacian_eigenvalue(n, n) : -laplacian_eigenvalue(n, 1), config.tol);
        EXPECT_NEAR(bounds.smallest.eigenvalue,
                    sign > 0.0 ? laplacian_eigenvalue(n, 1) : -laplacian_eigenvalue(n, n), config.tol);
      };
      check(op, "matrix-free operator");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
      const auto dense = make_dense_laplacian(op);
      check(dense, "dense matrix");
      check(make_sparse_matrix<sparse_matrix_t>(dense), "sparse matrix");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
    }
  }
  {
    SCOPED_TRACE("inside a kernel");
    constexpr size_t N = 4;
    constexpr size_t num_entities = 8;
    bounds_t eigenvalues("eigenvalues", num_entities);
    flags_t converged("converged", num_entities);
    solve_power_bounds_in_kernel(eigenvalues, converged);
    const auto eigenvalues_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, eigenvalues);
    const auto converged_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, converged);
    for (size_t e = 0; e < num_entities; ++e) {
      // To first order, for u = eps / 2. A residual within tol moves a Rayleigh quotient only by (tol + m)^2 / gap
      // (Kato-Temple), for m <= the rounding below. The dominant end: the apply's product and 2 additions per row,
      // 3 u ||(e + 1) L||, as e + 1 need not be a power of 2; the N-term dot and q's squared norm, (2N + 6) u lambda;
      // the reference's 4 sin^2 and scaling, 12 u lambda. The opposite end rounds on its own: the shift sigma is an
      // exact double that is added back, so the dominant end's error cancels. The shift adds u sigma and u |mu| to the
      // apply, for mu = lambda - sigma the shifted eigenvalue; the dot and q's squared norm add (2N + 6) u |mu|, adding
      // sigma back u lambda, and the reference 12 u lambda.
      const double scale = static_cast<double>(e + 1);
      const double tol = 1e-12;
      const double u = 0.5 * eps;
      const double largest = scale * laplacian_eigenvalue(N, N);
      const double smallest = scale * laplacian_eigenvalue(N, 1);
      const double largest_gap = scale * (laplacian_eigenvalue(N, N) - laplacian_eigenvalue(N, N - 1));
      const double smallest_gap = scale * (laplacian_eigenvalue(N, 2) - laplacian_eigenvalue(N, 1));
      const double apply = 3 * norm_L_bound * scale * u;
      const double largest_rounding = apply + (2.0 * N + 6) * u * largest + 12 * u * largest;
      const double smallest_rounding =
          apply + u * largest + (2.0 * N + 7) * u * (largest - smallest) + u * smallest + 12 * u * smallest;
      const double largest_bound = largest_rounding + (tol + largest_rounding) * (tol + largest_rounding) / largest_gap;
      const double smallest_bound =
          smallest_rounding + (tol + smallest_rounding) * (tol + smallest_rounding) / smallest_gap;
      EXPECT_TRUE(converged_host(e)) << "entity " << e;
      EXPECT_NEAR(eigenvalues_host(e, 0), largest, largest_bound) << "entity " << e;
      EXPECT_NEAR(eigenvalues_host(e, 1), smallest, smallest_bound) << "entity " << e;
    }
  }
}

TEST(Eigenvalues, PowerPolicies) {
  {
    SCOPED_TRACE("the change policy converges before the eigenvector");
    constexpr size_t N = 8;
    constexpr double tol = 1e-12;
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(1.0));
    auto by_residual = make_power_state(random_start<N>(4), Vector<double, N>{}, Vector<double, N>{});
    auto by_change = make_power_state(random_start<N>(4), Vector<double, N>{}, Vector<double, N>{});
    const PowerConfig<double> config{.max_iters = 20000, .tol = tol};
    const auto residual_result = solve_eigen_problem(problem, make_power_strategy(L2Residual{}, config), by_residual);
    const auto change_result = solve_eigen_problem(problem, make_power_strategy(ChangeResidual{}, config), by_change);
    ASSERT_TRUE(residual_result.converged);
    ASSERT_TRUE(change_result.converged);
    // The Rayleigh quotient's error e_k contracts by rho^2 = (lambda_{N-1} / lambda_N)^2 per step. Each measured
    // lambda rounds by R (L's apply, the N-term dot, q's normalization), so a measured change <= tol gives
    // e_{k-1} (1 - rho^2) <= tol + 2 R, and the returned lambda is within a tol + (1 + 2 a) R of lambda_N, for
    // a = rho^2 / (1 - rho^2), plus the reference's 4 sin^2 rounding
    const double rho = laplacian_eigenvalue(N, N - 1) / laplacian_eigenvalue(N, N);
    const double lambda_top = laplacian_eigenvalue(N, N);
    const double amplification = rho * rho / (1.0 - rho * rho);
    const double lambda_rounding = (norm_L_bound + (N + 3) * lambda_top) * eps;
    const double change_bound =
        amplification * tol + (1.0 + 2.0 * amplification) * lambda_rounding + 5.5 * eps * lambda_top;
    EXPECT_NEAR(residual_result.eigenvalue, laplacian_eigenvalue(N, N), 1e-12);
    EXPECT_NEAR(change_result.eigenvalue, laplacian_eigenvalue(N, N), change_bound);
    EXPECT_LE(norm(by_residual.r()), tol);
    EXPECT_GT(norm(by_change.r()), 1e3 * tol);
  }
  {
    SCOPED_TRACE("a zero image under the change policy");
    // A e_1 = 0: e_1 is an eigenvector the residual policy would accept, but the change policy measures no residual
    // and needs A q != 0 to take the next step
    const Matrix3d nilpotent{0.0, 1.0, 0.0,  //
                             0.0, 0.0, 0.0,  //
                             0.0, 0.0, 0.0};
    auto problem = make_eigen_problem<mm_backend_t>(Matrix3d(nilpotent));
    auto state = make_power_state(Vector3d{1.0, 0.0, 0.0}, Vector3d{}, Vector3d{});
    EXPECT_THROW(solve_eigen_problem(problem, make_power_strategy(ChangeResidual{}, PowerConfig<double>{}), state),
                 std::runtime_error);
  }
  {
    SCOPED_TRACE("an unconverged dominant end cannot shift");
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<3>(1.0));
    auto dominant = make_power_state(Vector3d{1.0, 0.3, -0.2}, Vector3d{}, Vector3d{});
    auto opposite = make_power_state(Vector3d{0.2, -0.7, 0.4}, Vector3d{}, Vector3d{});
    const auto strategy = make_power_strategy(PowerConfig<double>{.max_iters = 1});
    EXPECT_THROW(solve_eigen_bounds(problem, strategy, dominant, opposite), std::runtime_error);
  }
}
//@}

//! \name The cold start
//@{

// A zero start is the cold start c: c read back from a solve of no iterations lies strictly inside (-1, 1); power and
// Lanczos from a zero start find the top eigenvalue 2 of the all-ones vector of I + 1 1^T / n at n = 8 and 16, which
// a start of +-1 entries summing to zero misses, being orthogonal to it; and a zero start runs bit for bit as one at
// c, for power and for Lanczos on each backend, as both rescale a start by powers of two only, which is exact.
TEST(Eigenvalues, ColdStart) {
  const auto check = [&]<size_t N>() {
    SCOPED_TRACE(::testing::Message() << "n=" << N);
    auto problem = make_eigen_problem<mm_backend_t>(ones_rank_one_update<N>());

    // Read back c
    const auto cold = cold_start_of<N>(problem);
    for (size_t i = 0; i < N; ++i) {
      EXPECT_GT(cold[i], -1.0) << "entry " << i;
      EXPECT_LT(cold[i], 1.0) << "entry " << i;
    }

    // Power from zero
    auto power = make_power_state(Vector<double, N>::zeros(), Vector<double, N>{}, Vector<double, N>{});
    const auto power_result = solve_eigen_problem(
        problem, make_power_strategy(PowerConfig<double>{.max_iters = 200, .tol = 1e-12}), power);
    EXPECT_TRUE(power_result.converged) << power_result;
    EXPECT_NEAR(power_result.eigenvalue, 2.0, 1e-12);

    // Lanczos from zero
    auto lanczos = mundy_state<N, N>();
    const auto lanczos_result = solve_eigen_problem(
        problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = N, .tol = 1e-12}), lanczos);
    const unsigned k_largest = lanczos_result.largest.num_iters;
    const unsigned k_smallest = lanczos_result.smallest.num_iters;
    EXPECT_TRUE(lanczos_result.largest.converged) << lanczos_result;
    EXPECT_TRUE(lanczos_result.smallest.converged) << lanczos_result;
    EXPECT_NEAR(lanczos_result.largest.eigenvalue, 2.0,
                lanczos_result.largest.residual +
                    k_largest * eps * norm_of_t(lanczos.alpha(), lanczos.beta(), k_largest));
    EXPECT_NEAR(lanczos_result.smallest.eigenvalue, 1.0,
                lanczos_result.smallest.residual +
                    k_smallest * eps * norm_of_t(lanczos.alpha(), lanczos.beta(), k_smallest));

    // Control: from +-1 entries summing to zero, orthogonal to the all-ones vector, both find the eigenvalue 1
    Vector<double, N> alternating;
    for (size_t i = 0; i < N; ++i) {
      alternating[i] = i % 2 == 0 ? 1.0 : -1.0;
    }
    auto power_alternating = make_power_state(Vector<double, N>(alternating), Vector<double, N>{}, Vector<double, N>{});
    const auto power_control = solve_eigen_problem(
        problem, make_power_strategy(PowerConfig<double>{.max_iters = 200, .tol = 1e-12}), power_alternating);
    EXPECT_TRUE(power_control.converged) << power_control;
    EXPECT_NEAR(power_control.eigenvalue, 1.0, 1e-12 + (1.5 * N + 3) * eps * 2.0);
    Vector<double, N> lanczos_alternating = alternating;
    mundy_basis_t<N, 1> control_basis{};
    auto control_state = mundy_state<N, N>(lanczos_alternating, NoVector{}, control_basis);
    const auto lanczos_control = solve_eigen_problem(
        problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = N, .tol = 1e-12, .smallest = false}),
        control_state);
    const unsigned k_control = lanczos_control.largest.num_iters;
    EXPECT_TRUE(lanczos_control.largest.converged) << lanczos_control;
    EXPECT_NEAR(lanczos_control.largest.eigenvalue, 1.0,
                lanczos_control.largest.residual +
                    k_control * eps * norm_of_t(control_state.alpha(), control_state.beta(), k_control));
  };
  check.template operator()<8>();
  check.template operator()<16>();

  {
    SCOPED_TRACE("a zero start is the cold start");
    constexpr size_t N = 8;
    constexpr unsigned max_iters = 16;
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(1.0));
    const auto cold = cold_start_of<N>(problem);
    const auto expect_same = [](const EigenBounds<double>& a, const EigenBounds<double>& b) {
      for (const bool largest : {true, false}) {
        const auto& x = largest ? a.largest : a.smallest;
        const auto& y = largest ? b.largest : b.smallest;
        EXPECT_EQ(x.num_iters, y.num_iters);
        EXPECT_EQ(x.eigenvalue, y.eigenvalue);
        EXPECT_EQ(x.residual, y.residual);
        EXPECT_EQ(x.converged, y.converged);
      }
    };
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-10});

    // Power
    const auto power_strategy = make_power_strategy(PowerConfig<double>{.max_iters = 50, .tol = 1e-10});
    auto power_zero = make_power_state(Vector<double, N>::zeros(), Vector<double, N>{}, Vector<double, N>{});
    auto power_cold = make_power_state(Vector<double, N>(cold), Vector<double, N>{}, Vector<double, N>{});
    const auto power_zero_result = solve_eigen_problem(problem, power_strategy, power_zero);
    const auto power_cold_result = solve_eigen_problem(problem, power_strategy, power_cold);
    EXPECT_EQ(power_zero_result.num_iters, power_cold_result.num_iters);
    EXPECT_EQ(power_zero_result.eigenvalue, power_cold_result.eigenvalue);
    EXPECT_EQ(power_zero_result.residual, power_cold_result.residual);
    EXPECT_TRUE(same_entries(power_zero.q(), power_cold.q(), N));

    // Lanczos, MundyMathBackend
    auto zero_estimate = Vector<double, N>::zeros();
    Vector<double, N> cold_estimate = cold;
    mundy_basis_t<N, 4> zero_basis{}, cold_basis{};
    auto zero = mundy_state<N, max_iters>(zero_estimate, NoVector{}, zero_basis);
    auto at_cold = mundy_state<N, max_iters>(cold_estimate, NoVector{}, cold_basis);
    expect_same(solve_eigen_problem(problem, strategy, zero), solve_eigen_problem(problem, strategy, at_cold));
    EXPECT_TRUE(same_entries(zero_estimate, cold_estimate, N));

    // Lanczos, KokkosBackend
    auto kokkos_problem = make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{N, 1.0});
    view_t zero_view("zero", N);
    view_t cold_view = to_view(cold);
    view_basis_t<4> zero_view_basis("basis", N), cold_view_basis("basis", N);
    auto kokkos_zero = view_state(N, max_iters, zero_view, NoVector{}, zero_view_basis);
    auto kokkos_cold = view_state(N, max_iters, cold_view, NoVector{}, cold_view_basis);
    expect_same(solve_eigen_problem(kokkos_problem, strategy, kokkos_zero),
                solve_eigen_problem(kokkos_problem, strategy, kokkos_cold));
    EXPECT_TRUE(same_entries(to_host(zero_view), to_host(cold_view), N));
  }
}
//@}

//! \name Lanczos iterations and certificates
//@{

TEST(Eigenvalues, LanczosIterates) {
  {
    SCOPED_TRACE("the first step");
    // The first T has one entry, q^T A q, so both ends are the Rayleigh quotient; its eigenvector is s = 1, so
    // rho = beta_1 = ||A q - theta q||
    const Matrix3d A{1.0, 0.0, 0.0,  //
                     0.0, 2.0, 0.0,  //
                     0.0, 0.0, 4.0};
    Vector3d start{1.0, 2.0, 1.0};
    const Vector3d q = start / norm(start);
    const double expected_eigenvalue = dot(q, A * q);
    const double expected_residual = norm(A * q - expected_eigenvalue * q);
    auto problem = make_eigen_problem<mm_backend_t>(Matrix3d(A));
    Vector3d largest = start;
    Vector3d smallest = start;
    mundy_basis_t<3, 3> basis{};
    auto state = mundy_state<3, 3>(largest, smallest, basis);
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 3, .tol = 0.0});

    strategy.initialize(problem, state);
    ASSERT_FALSE(strategy.iterate(problem, state));
    EXPECT_EQ(state.iter(), 1u);
    EXPECT_NEAR(state.largest().eigenvalue, expected_eigenvalue, 1e-14);
    EXPECT_NEAR(state.smallest().eigenvalue, expected_eigenvalue, 1e-14);
    EXPECT_NEAR(state.largest().residual, expected_residual, 1e-14);
    EXPECT_NEAR(state.smallest().residual, expected_residual, 1e-14);
  }
  {
    SCOPED_TRACE("monotone toward the ends");
    // Both ends move monotonically toward the spectrum's ends from inside (interlacing, exact for the stored T); each
    // theta is within 4 eps ||T|| of T's extreme eigenvalue, so consecutive thetas may step back by 8 eps ||T||.
    // max_iters = 16 keeps every iteration within the solve's measurement of every step up to 16
    constexpr size_t N = 16;
    constexpr unsigned max_iters = 16;
    const double smallest = laplacian_eigenvalue(N, 1);
    const double largest = laplacian_eigenvalue(N, N);
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(1.0));
    auto state = mundy_state<N, max_iters>();
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 0.0});

    strategy.initialize(problem, state);
    double previous_smallest = std::numeric_limits<double>::infinity();
    double previous_largest = -std::numeric_limits<double>::infinity();
    for (unsigned k = 1; k <= max_iters; ++k) {
      SCOPED_TRACE(::testing::Message() << "iteration " << k);
      const bool stopped = strategy.iterate(problem, state);
      ASSERT_EQ(state.iter(), k);
      const double norm_T = norm_of_t(state.alpha(), state.beta(), k);
      const double floor = k * eps * norm_T;
      const double step_back = 8.0 * eps * norm_T;
      EXPECT_GE(state.smallest().eigenvalue, smallest - floor);
      EXPECT_LE(state.largest().eigenvalue, largest + floor);
      EXPECT_LE(state.smallest().eigenvalue, previous_smallest + step_back);
      EXPECT_GE(state.largest().eigenvalue, previous_largest - step_back);
      previous_smallest = state.smallest().eigenvalue;
      previous_largest = state.largest().eigenvalue;
      if (stopped) break;
    }
    const double floor = state.iter() * eps * norm_of_t(state.alpha(), state.beta(), state.iter());
    EXPECT_NEAR(state.smallest().eigenvalue, smallest, state.smallest().residual + floor);
    EXPECT_NEAR(state.largest().eigenvalue, largest, state.largest().residual + floor);
  }
}

// At the stop of every max_iters = 1, 2, ... from the cold start, until the solve stops on its own: each end's theta
// is the extreme eigenvalue of its T to 4 eps ||T||, the bisection's accuracy; its rho matches the dense reference
// while unconverged; a converged rho meets the convergence rule; and theta lies within rho + k eps ||T|| of an
// eigenvalue of A, of the extreme one once converged. With tol = 0 the rotated spectra stop only on an invariant T,
// beta_k at rounding, which the loss of orthogonality delays past N; 4N bounds it. On the n = 1000 diagonal with a
// separated top, over max_iters 1, 2, 4, ..., 512 with tol 1e-12, the top end converges early, and T then holds
// duplicate Ritz values, two eigenvalues within rounding of the top end's theta, while the bottom end is still
// measured.
TEST(Eigenvalues, LanczosCertificates) {
  const auto expect_certified = [](const auto& state, const EigenBounds<double>& result, const auto& spectrum,
                                   size_t n, double construction, double tol) {
    for (const bool largest : {true, false}) {
      SCOPED_TRACE(largest ? "largest end" : "smallest end");
      const auto& end = largest ? result.largest : result.smallest;
      const unsigned k = end.num_iters;
      const double norm_T = norm_of_t(state.alpha(), state.beta(), k);
      const double floor = k * eps * norm_T;
      const auto reference = reference_end(state, k, largest);
      EXPECT_NEAR(end.eigenvalue, reference.eigenvalue, 4.0 * eps * norm_T);
      if (end.converged) {
        EXPECT_LE(end.residual, std::max(tol * std::abs(end.eigenvalue), floor));
        EXPECT_LE(std::abs(end.eigenvalue - (largest ? spectrum[n - 1] : spectrum[0])),
                  end.residual + floor + construction);
      } else {
        // perturbing T by floor moves s_k by at most floor / gap
        EXPECT_NEAR(end.residual, reference.residual, state.beta()[k - 1] * floor / reference.gap);
      }
      EXPECT_LE(distance_to_spectrum(end.eigenvalue, spectrum, n), end.residual + floor + construction);
    }
  };

  for (const auto& [name, spectrum] : rotated_spectra()) {
    SCOPED_TRACE(name);
    constexpr size_t N = 32;
    auto problem = make_eigen_problem<mm_backend_t>(rotated<N>(spectrum));
    unsigned natural_stop = 0;
    for (unsigned max_iters = 1; natural_stop == 0 || max_iters <= natural_stop + 1; ++max_iters) {
      SCOPED_TRACE(::testing::Message() << "max_iters " << max_iters);
      ASSERT_LE(max_iters, 4 * N) << "the solve must stop on its own";
      auto state = mundy_state<N, 4 * N>();
      const auto result = solve_eigen_problem(
          problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 0.0}), state);
      if (natural_stop == 0 && stop_of(result) < max_iters) natural_stop = stop_of(result);
      EXPECT_EQ(stop_of(result), natural_stop == 0 ? max_iters : natural_stop);
      expect_certified(state, result, spectrum, N, construction_error(spectrum), 0.0);
    }
  }

  {
    SCOPED_TRACE("n = 1000 diagonal");
    constexpr size_t n = 1000;
    const view_t d = separated_top_diagonal(n, 10.0);
    const auto spectrum = to_host(d);
    auto problem = make_eigen_problem<kokkos_backend_t>(DiagonalKokkosOp{d});
    unsigned natural_stop = 0;
    EigenResult<double> top;
    std::vector<double> alpha, beta;
    for (const unsigned max_iters : {1u, 2u, 4u, 8u, 12u, 16u, 24u, 32u, 48u, 64u, 96u, 128u, 192u, 256u, 384u, 512u}) {
      SCOPED_TRACE(::testing::Message() << "max_iters " << max_iters);
      auto state = view_state(n, max_iters);
      const auto result = solve_eigen_problem(
          problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-12}), state);
      if (natural_stop == 0 && stop_of(result) < max_iters) natural_stop = stop_of(result);
      EXPECT_EQ(stop_of(result), natural_stop == 0 ? max_iters : natural_stop);
      expect_certified(state, result, spectrum, n, 0.0, 1e-12);
      top = result.largest;
      alpha.assign(state.alpha(), state.alpha() + stop_of(result));
      beta.assign(state.beta(), state.beta() + stop_of(result));
    }
    ASSERT_GT(natural_stop, 0u) << "the solve must stop on its own";
    EXPECT_LT(4 * top.num_iters, natural_stop) << "the top end must converge long before the bottom end";

    // Control: duplicate Ritz values, the two largest eigenvalues of the final T both within rho + k eps ||T|| of the
    // top end's theta
    const auto theta = eigen_of_t(alpha, beta, natural_stop);
    const double floor = natural_stop * eps * norm_of_t(alpha.data(), beta.data(), natural_stop);
    EXPECT_LE(std::abs(theta[natural_stop - 1] - top.eigenvalue), top.residual + floor);
    EXPECT_LE(std::abs(theta[natural_stop - 2] - top.eigenvalue), top.residual + floor);
  }
}

// Both ends to tol 1e-8 on each rotated spectrum: the solve stops no earlier than the first k at which the dense
// reference certifies both ends, and at most num_iters / 8 iterations after it, the measurement spacing the solve
// documents.
TEST(Eigenvalues, LanczosStopRule) {
  constexpr size_t N = 32;
  constexpr unsigned max_iters = 400;
  constexpr double tol = 1e-8;
  for (const auto& [name, spectrum] : rotated_spectra()) {
    SCOPED_TRACE(name);
    auto problem = make_eigen_problem<mm_backend_t>(rotated<N>(spectrum));
    auto state = mundy_state<N, max_iters>();
    const auto result = solve_eigen_problem(
        problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = tol}), state);
    ASSERT_TRUE(result.largest.converged && result.smallest.converged) << result;

    // The first k each end is certified at
    const unsigned stop = stop_of(result);
    unsigned certified[2] = {0, 0};  // largest, smallest
    for (unsigned k = 1; k <= stop; ++k) {
      const double norm_T = norm_of_t(state.alpha(), state.beta(), k);
      for (const bool largest : {true, false}) {
        unsigned& first = certified[largest ? 0 : 1];
        const auto reference = reference_end(state, k, largest);
        if (first == 0 && reference.residual <= std::max(tol * std::abs(reference.eigenvalue), k * eps * norm_T)) {
          first = k;
        }
      }
    }
    ASSERT_GT(certified[0], 0u);
    ASSERT_GT(certified[1], 0u);
    const unsigned both = std::max(certified[0], certified[1]);
    EXPECT_GE(stop, both);
    EXPECT_LE(stop - both, stop / 8);
  }
}

// Each choice of ends, on the singular and the nonsingular separated top: a wanted end converges to its extreme
// eigenvalue within its certificate, an unwanted end is a default result, and a one-end solve stops where that end
// stopped in the two-end solve, so the end converging first costs fewer iterations alone. The two-end cold start is
// 2 c / ||c||, a power-of-two multiple of the one-end start c / ||c||, so all three solves build the same T.
TEST(Eigenvalues, LanczosEnds) {
  constexpr size_t N = 32;
  constexpr unsigned max_iters = 400;
  for (const double bottom : {0.0, 1.0}) {
    SCOPED_TRACE(::testing::Message() << "bottom " << bottom);
    const auto spectrum = separated_top_spectrum(bottom);
    auto problem = make_eigen_problem<mm_backend_t>(rotated<N>(spectrum));
    // Each end's certificate uses the norm of the T it was measured on
    const auto solve = [&](bool largest, bool smallest) {
      auto state = mundy_state<N, max_iters>();
      const LanczosConfig<double> config{
          .max_iters = max_iters, .tol = 1e-10, .largest = largest, .smallest = smallest};
      const auto result = solve_eigen_problem(problem, make_lanczos_strategy(config), state);
      return std::tuple{result, norm_of_t(state.alpha(), state.beta(), result.largest.num_iters),
                        norm_of_t(state.alpha(), state.beta(), result.smallest.num_iters)};
    };
    const auto both = solve(true, true);
    const auto largest_only = solve(true, false);
    const auto smallest_only = solve(false, true);

    // Wanted ends converge within their certificates
    for (const auto& [result, largest_norm, smallest_norm] : {both, largest_only, smallest_only}) {
      for (const bool largest : {true, false}) {
        const auto& end = largest ? result.largest : result.smallest;
        if (end.num_iters == 0) continue;
        EXPECT_TRUE(end.converged) << result;
        EXPECT_LE(std::abs(end.eigenvalue - (largest ? spectrum[N - 1] : spectrum[0])),
                  end.residual + end.num_iters * eps * (largest ? largest_norm : smallest_norm) +
                      construction_error(spectrum));
      }
    }

    // Unwanted ends are default results
    for (const auto& end : {std::get<0>(largest_only).smallest, std::get<0>(smallest_only).largest}) {
      EXPECT_EQ(end.num_iters, 0u);
      EXPECT_TRUE(std::isnan(end.eigenvalue));
      EXPECT_TRUE(std::isnan(end.residual));
      EXPECT_FALSE(end.converged);
    }

    // The ends are independent
    const auto& both_result = std::get<0>(both);
    ASSERT_LT(both_result.largest.num_iters, both_result.smallest.num_iters) << "the top end converges first";
    EXPECT_EQ(std::get<0>(largest_only).largest.num_iters, both_result.largest.num_iters);
    EXPECT_EQ(std::get<0>(smallest_only).smallest.num_iters, both_result.smallest.num_iters);
    EXPECT_LT(stop_of(std::get<0>(largest_only)), stop_of(both_result));
  }
}

// The Krylov space becomes invariant at m = 2, an iteration where the ends are measured anyway, and at m = 19, one
// where they are not (every step is measured up to 16, then 18, 18 - 16 >= 18 / 8, and next 21): H diag(1, 2, 3) H
// from the eigenvectors of 1 and 3, and diag of the 19 Chebyshev points 2 - cos((i + 1/2) pi / 19) from all ones,
// whose T is the Chebyshev Jacobi matrix: alpha_j = 2, beta_1 = 1/sqrt(2), beta_j = 1/2. The solve stops at m, both
// ends within their certificates of the span's extreme eigenvalues and each estimate its end's eigenvector to
// rounding over the gap to the next eigenvalue.
//
// For a unit estimate y at angle phi to its eigenvector, 1 - |y^T u| <= sin^2 phi, with sin phi at most the
// residual (rho + floor) over the gap; the rest is the rounding of y's norm, (N/2 + 3) eps, and of the N-term dot.
TEST(Eigenvalues, LanczosInvariantStop) {
  const auto solve = [&]<size_t N>(const Matrix<double, N, N>& A, const Vector<double, N>& start) {
    auto problem = make_eigen_problem<mm_backend_t>(Matrix<double, N, N>(A));
    Vector<double, N> largest = start;
    Vector<double, N> smallest = start;
    mundy_basis_t<N, 2 * N> basis{};
    auto state = mundy_state<N, 2 * N>(largest, smallest, basis);
    const auto result = solve_eigen_problem(
        problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = 2 * N, .tol = 0.0}), state);
    std::vector<double> alpha(state.alpha(), state.alpha() + stop_of(result));
    std::vector<double> beta(state.beta(), state.beta() + stop_of(result));
    return std::tuple{result, norm_of_t(state.alpha(), state.beta(), stop_of(result)), largest, smallest, alpha, beta};
  };

  {
    SCOPED_TRACE("m = 2");
    const auto [result, norm_T, largest, smallest, alpha, beta] =
        solve.template operator()<3>(rotated<3>(Vector3d{1.0, 2.0, 3.0}),
                                     rotated_eigenvector<3>(0) + rotated_eigenvector<3>(2));
    constexpr size_t N = 3;
    const double floor = 2 * eps * norm_T;
    const double gap = 1.0;  // to the eigenvalue 2
    const double construction = construction_error(Vector3d{1.0, 2.0, 3.0});
    const auto overlap_floor = [&](double rho) {
      const double sine = (rho + floor + construction) / gap;
      return std::min(3 * floor, sine * sine + (1.5 * N + 4) * eps);
    };
    EXPECT_EQ(stop_of(result), 2u);
    EXPECT_TRUE(result.largest.converged && result.smallest.converged) << result;
    EXPECT_LE(std::abs(result.smallest.eigenvalue - 1.0), result.smallest.residual + floor);
    EXPECT_LE(std::abs(result.largest.eigenvalue - 3.0), result.largest.residual + floor);
    EXPECT_NEAR(std::abs(dot(smallest, rotated_eigenvector<3>(0))), 1.0, overlap_floor(result.smallest.residual));
    EXPECT_NEAR(std::abs(dot(largest, rotated_eigenvector<3>(2))), 1.0, overlap_floor(result.largest.residual));
  }
  {
    SCOPED_TRACE("m = 19");
    constexpr size_t N = 19;
    auto A = Matrix<double, N, N>::zeros();
    for (size_t i = 0; i < N; ++i) {
      A(i, i) = 2.0 - std::cos((static_cast<double>(i) + 0.5) * Kokkos::numbers::pi_v<double> / N);
    }
    const auto [result, norm_T, largest, smallest, alpha, beta] =
        solve.template operator()<N>(A, Vector<double, N>::ones());
    const double floor = N * eps * norm_T;
    ASSERT_EQ(stop_of(result), N);
    EXPECT_TRUE(result.largest.converged && result.smallest.converged) << result;
    for (size_t j = 0; j < N; ++j) {
      // the rounding of an N-term inner product, then of j steps of the recurrence
      const double coefficient_floor = (N + j + 1) * eps * norm_T;
      const double closed_form = j == 0 ? 1.0 / std::sqrt(2.0) : (j + 1 < N ? 0.5 : 0.0);
      EXPECT_NEAR(alpha[j], 2.0, coefficient_floor) << "alpha_" << j + 1;
      EXPECT_NEAR(beta[j], closed_form, coefficient_floor) << "beta_" << j + 1;
    }
    EXPECT_LE(std::abs(result.smallest.eigenvalue - A(0, 0)), result.smallest.residual + floor);
    EXPECT_LE(std::abs(result.largest.eigenvalue - A(N - 1, N - 1)), result.largest.residual + floor);
    // u is a unit vector e_i here, so no dot rounds
    const double gap = A(1, 1) - A(0, 0);
    const auto entry_floor = [&](double rho) {
      const double sine = (rho + floor) / gap;
      return std::min(N * floor / gap, sine * sine + (N / 2.0 + 3) * eps);
    };
    EXPECT_NEAR(std::abs(largest[N - 1]), 1.0, entry_floor(result.largest.residual));
    EXPECT_NEAR(std::abs(smallest[0]), 1.0, entry_floor(result.smallest.residual));
  }
}

// Lanczos from e_1 on the Jacobi matrix J reproduces J, one unit vector per iteration, the top end converging within a
// few. A step that is not finite (A gives +infinity once the Lanczos vector reaches the last entry) ends the solve
// unconverged with the results and estimates of a solve capped one iteration earlier. A preconditioner negative on that
// entry ends it with no end converged, and otherwise the results and estimates of a solve capped before the step. A
// start that is not finite, or on which P is not positive, ends it at its first iteration with default results.
TEST(Eigenvalues, LanczosBreakdown) {
  constexpr size_t N = 16;
  constexpr unsigned max_iters = 2 * N;
  auto first = Vector<double, N>::zeros();
  first[0] = 1.0;
  const auto solve = [&](const auto& strategy, auto preconditioned_tag) {
    constexpr bool preconditioned = decltype(preconditioned_tag)::value;
    auto problem = make_eigen_problem<mm_backend_t>(BlowupMundyOp<N>{});
    Vector<double, N> largest = first;
    Vector<double, N> smallest = first;
    mundy_basis_t<N, 4> basis{};
    auto state = mundy_state<N, max_iters, preconditioned>(largest, smallest, basis);
    const auto result = solve_eigen_problem(problem, strategy, state);
    return std::tuple{result, largest, smallest};
  };
  const auto config = [](unsigned cap) { return LanczosConfig<double>{.max_iters = cap, .tol = 1e-10}; };

  {
    SCOPED_TRACE("a step that is not finite");
    const auto [result, largest, smallest] = solve(make_lanczos_strategy(config(max_iters)), std::false_type{});
    const auto [capped, capped_largest, capped_smallest] =
        solve(make_lanczos_strategy(config(N - 1)), std::false_type{});

    // Premise: the top end converged well before the bad step, the bottom end did not
    ASSERT_TRUE(result.largest.converged) << result;
    ASSERT_LT(result.largest.num_iters, N - 1);
    ASSERT_FALSE(capped.smallest.converged) << capped;

    // Stopped one iteration short of the bad step, as the capped solve
    EXPECT_EQ(stop_of(result), N - 1);
    for (const bool is_largest : {true, false}) {
      const auto& a = is_largest ? result.largest : result.smallest;
      const auto& b = is_largest ? capped.largest : capped.smallest;
      EXPECT_EQ(a.num_iters, b.num_iters);
      EXPECT_EQ(a.eigenvalue, b.eigenvalue);
      EXPECT_EQ(a.residual, b.residual);
      EXPECT_EQ(a.converged, b.converged);
    }
    EXPECT_TRUE(same_entries(largest, capped_largest, N));
    EXPECT_TRUE(same_entries(smallest, capped_smallest, N));
  }
  {
    SCOPED_TRACE("a preconditioner not positive definite");
    auto diagonal = Vector<double, N>::ones();
    diagonal[N - 1] = -1e-6;
    const JacobiPreconditioner<mm_backend_t, Vector<double, N>> indefinite(mm_backend_t{}, Vector<double, N>(diagonal));
    const auto [result, largest, smallest] =
        solve(make_lanczos_strategy(config(max_iters), indefinite), std::true_type{});
    const auto [capped, capped_largest, capped_smallest] =
        solve(make_lanczos_strategy(config(N - 2), indefinite), std::true_type{});

    // Premise: before P shows itself, the top end converged
    ASSERT_TRUE(capped.largest.converged) << capped;

    // Stopped where w^T P w < 0, with no end converged: w_{N-1} lies along e_N, where P is -1e-6, far below rounding,
    // so step N - 1 ends the solve on T_{N-2}, with the results and estimates of the solve capped there
    EXPECT_EQ(stop_of(result), N - 2);
    EXPECT_FALSE(result.largest.converged);
    EXPECT_FALSE(result.smallest.converged);
    for (const bool is_largest : {true, false}) {
      const auto& a = is_largest ? result.largest : result.smallest;
      const auto& b = is_largest ? capped.largest : capped.smallest;
      EXPECT_EQ(a.num_iters, b.num_iters);
      EXPECT_EQ(a.eigenvalue, b.eigenvalue);
      EXPECT_EQ(a.residual, b.residual);
    }
    EXPECT_TRUE(same_entries(largest, capped_largest, N));
    EXPECT_TRUE(same_entries(smallest, capped_smallest, N));
  }
  {
    SCOPED_TRACE("starts that are not finite or on which P is not positive");
    constexpr size_t M = 8;
    auto problem = make_eigen_problem<mm_backend_t>(laplacian<M>(1.0));
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 16, .tol = 1e-10});
    const auto expect_default = [](const EigenBounds<double>& result) {
      for (const auto& end : {result.largest, result.smallest}) {
        EXPECT_EQ(end.num_iters, 0u);
        EXPECT_TRUE(std::isnan(end.eigenvalue));
        EXPECT_TRUE(std::isnan(end.residual));
        EXPECT_FALSE(end.converged);
      }
    };
    const auto e = random_start<M>(3);

    // A NaN entry
    Vector<double, M> with_nan = e;
    with_nan[3] = nan;
    const Vector<double, M> with_nan_before = with_nan;
    mundy_basis_t<M, 4> basis{};
    auto nan_state = mundy_state<M, 16>(with_nan, NoVector{}, basis);
    expect_default(solve_eigen_problem(problem, strategy, nan_state));
    EXPECT_TRUE(same_entries(with_nan, with_nan_before, M));

    // A negative-definite preconditioner
    const JacobiPreconditioner<mm_backend_t, Vector<double, M>> negative(mm_backend_t{}, -Vector<double, M>::ones());
    auto negative_state = mundy_state<M, 16, true>();
    expect_default(solve_eigen_problem(
        problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = 16, .tol = 1e-10}, negative),
        negative_state));
  }
}

// From a start along the eigenvector of 2 with components of 10^-3 along that of 10 and along a random vector, on the
// separated top, the largest end's theta sits near 2 with a small rho, crosses the gap, and reaches 10: at every
// iteration the reported theta and rho place an eigenvalue of A within rho + k eps ||T||, in the gap too. max_iters =
// 16 keeps every iteration within the solve's measurement of every step up to 16.
TEST(Eigenvalues, LanczosWarmCertificate) {
  constexpr size_t N = 32;
  constexpr unsigned max_iters = 16;
  const auto spectrum = separated_top_spectrum(1.0);
  auto problem = make_eigen_problem<mm_backend_t>(rotated<N>(spectrum));
  Vector<double, N> estimate = near_second_eigenvector<N>();
  mundy_basis_t<N, 4> basis{};
  auto state = mundy_state<N, max_iters>(estimate, NoVector{}, basis);
  const auto strategy = make_lanczos_strategy(
      LanczosConfig<double>{.max_iters = max_iters, .tol = 0.0, .largest = true, .smallest = false});

  strategy.initialize(problem, state);
  bool moved = false;
  for (unsigned k = 1; k <= max_iters; ++k) {
    SCOPED_TRACE(::testing::Message() << "iteration " << k);
    const bool stopped = strategy.iterate(problem, state);
    const auto& end = state.largest();
    ASSERT_EQ(end.num_iters, k);
    EXPECT_LE(distance_to_spectrum(end.eigenvalue, spectrum, N),
              end.residual + k * eps * norm_of_t(state.alpha(), state.beta(), k) + construction_error(spectrum));
    moved = moved || end.eigenvalue > 9.0;
    if (stopped) break;
  }
  EXPECT_TRUE(moved) << "the fixture must carry the largest end from 2 to 10";
}
//@}

//! \name Lanczos estimates
//@{

// Two-end solves to tol 1e-6 on the two separated ends, unpreconditioned on R = H diag(lambda) H and preconditioned on
// D R D with P = D^-2, on each backend: the ends stop at different iterations with rho far above rounding, and each
// end's unit estimate u (D^-1 e with P) has the residual ||R u - theta u|| = rho / ||y||, for y = V s the estimate
// before its scaling to unit P-norm, with ||y|| = 1 to the orthogonality the Lanczos vectors keep, k eps ||T|| / rho;
// u is R's eigenvector to (residual / gap)^2, as 1 - |u^T q| <= sin^2(u, q) <= (residual / gap)^2. The rounding
// allows N k eps ||T|| for k steps of N-term products.
TEST(Eigenvalues, LanczosRitzVectors) {
  constexpr size_t N = 32;
  constexpr size_t K = 64;
  const auto spectrum = separated_ends_spectrum();
  const auto R = rotated<N>(spectrum);
  const LanczosConfig<double> config{.max_iters = K, .tol = 1e-6};
  const auto expect_ritz_vectors = [&](const EigenBounds<double>& result, double norm_T,
                                       const Vector<double, N>& largest_u, const Vector<double, N>& smallest_u) {
    ASSERT_TRUE(result.largest.converged && result.smallest.converged) << result;
    ASSERT_NE(result.largest.num_iters, result.smallest.num_iters) << "the ends must stop at different iterations";
    const double rounding = N * stop_of(result) * eps * norm_T + construction_error(spectrum);
    const double orthogonality = stop_of(result) * eps * norm_T /
                                 std::min(result.largest.residual, result.smallest.residual);
    for (const bool largest : {true, false}) {
      SCOPED_TRACE(largest ? "largest end" : "smallest end");
      const auto& end = largest ? result.largest : result.smallest;
      const auto& u = largest ? largest_u : smallest_u;
      const double residual = norm(R * u - end.eigenvalue * u);
      const double gap = largest ? spectrum[N - 1] - spectrum[N - 2] : spectrum[1] - spectrum[0];
      ASSERT_GT(end.residual, 1e3 * rounding) << "rho must lie far above rounding";
      EXPECT_NEAR(residual, end.residual, end.residual * orthogonality + rounding);
      EXPECT_NEAR(std::abs(dot(u, rotated_eigenvector<N>(largest ? N - 1 : 0))), 1.0,
                  (residual / gap) * (residual / gap) + rounding);
    }
  };

  {
    SCOPED_TRACE("MundyMathBackend, unpreconditioned");
    auto largest = Vector<double, N>::zeros();
    auto smallest = Vector<double, N>::zeros();
    mundy_basis_t<N, K> basis{};
    auto state = mundy_state<N, K>(largest, smallest, basis);
    const auto result =
        solve_eigen_problem(make_eigen_problem<mm_backend_t>(Matrix<double, N, N>(R)), make_lanczos_strategy(config),
                            state);
    expect_ritz_vectors(result, norm_of_t(state.alpha(), state.beta(), stop_of(result)), largest, smallest);
  }
  {
    SCOPED_TRACE("MundyMathBackend, preconditioned");
    auto largest = Vector<double, N>::zeros();
    auto smallest = Vector<double, N>::zeros();
    mundy_basis_t<N, K> basis{};
    auto state = mundy_state<N, K, true>(largest, smallest, basis);
    const auto result = solve_eigen_problem(make_eigen_problem<mm_backend_t>(congruent<N>(R)),
                                            make_lanczos_strategy(config, make_congruence_preconditioner<N>()), state);
    const double norm_T = norm_of_t(state.alpha(), state.beta(), stop_of(result));
    expect_ritz_vectors(result, norm_T, divide_by_congruence_scale(largest), divide_by_congruence_scale(smallest));
  }
  {
    SCOPED_TRACE("KokkosBackend, unpreconditioned");
    view_t largest("largest", N), smallest("smallest", N);
    view_basis_t<K> basis("basis", N);
    auto state = view_state(N, K, largest, smallest, basis);
    const auto result =
        solve_eigen_problem(make_eigen_problem<kokkos_backend_t>(to_kokkos(R)), make_lanczos_strategy(config), state);
    const double norm_T = norm_of_t(state.alpha(), state.beta(), stop_of(result));
    expect_ritz_vectors(result, norm_T, to_vector<N>(largest), to_vector<N>(smallest));
  }
  {
    SCOPED_TRACE("KokkosBackend, preconditioned");
    view_t largest("largest", N), smallest("smallest", N);
    view_basis_t<K> basis("basis", N);
    auto state = view_state<true>(N, K, largest, smallest, basis);
    const auto result = solve_eigen_problem(make_eigen_problem<kokkos_backend_t>(to_kokkos(congruent<N>(R))),
                                            make_lanczos_strategy(config, make_congruence_preconditioner(N)), state);
    const double norm_T = norm_of_t(state.alpha(), state.beta(), stop_of(result));
    expect_ritz_vectors(result, norm_T, divide_by_congruence_scale(to_vector<N>(largest)),
                        divide_by_congruence_scale(to_vector<N>(smallest)));
  }
}

// The basis's capacity changes nothing the solve measures, nor, beyond rounding, its estimates. With the same warm
// estimate, bases of 1 and 16 give the same T and both ends bit for bit. After k = 12 iterations on L (on D L D with
// P = D^-2), from the cold start and from a warm one, bases of 1, 2, k - 1, k and more than k columns give the same
// estimates to rounding, on each backend: k - 1 columns repeat one step and k or more repeat none, and with no
// preconditioner the repeated steps add the same terms in the same order as the combination, so only the rounding of
// k-term sums and of the normalization differs. The backends agree. After k = 256 iterations on the n = 1000 diagonal
// with a separated top, where T holds duplicate Ritz values of the converged top end, bases of 1, 2 and 4 give the
// estimates of a basis of 256.
TEST(Eigenvalues, LanczosBasisCapacity) {
  constexpr size_t N = 16;
  {
    SCOPED_TRACE("the solve");
    auto problem = make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{N, 1.0});
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 32, .tol = 1e-10});
    const auto solve = [&]<size_t K>() {
      view_t largest = to_view(random_start<N>(21));
      view_t smallest("smallest", N);
      view_basis_t<K> basis("basis", N);
      auto state = view_state(N, 32, largest, smallest, basis);
      const auto result = solve_eigen_problem(problem, strategy, state);
      std::vector<double> T(state.alpha(), state.alpha() + stop_of(result));
      T.insert(T.end(), state.beta(), state.beta() + stop_of(result));
      return std::pair{result, T};
    };
    const auto [one, one_T] = solve.template operator()<1>();
    const auto [sixteen, sixteen_T] = solve.template operator()<16>();
    ASSERT_EQ(stop_of(one), stop_of(sixteen));
    EXPECT_EQ(one_T, sixteen_T);
    for (const bool largest : {true, false}) {
      const auto& a = largest ? one.largest : one.smallest;
      const auto& b = largest ? sixteen.largest : sixteen.smallest;
      EXPECT_EQ(a.num_iters, b.num_iters);
      EXPECT_EQ(a.eigenvalue, b.eigenvalue);
      EXPECT_EQ(a.residual, b.residual);
      EXPECT_EQ(a.converged, b.converged);
    }
  }

  constexpr unsigned k = 12;
  const LanczosConfig<double> config{.max_iters = k, .tol = 0.0};
  const auto warm = laplacian_eigenvector<N>(N) + 0.1 * random_start<N>(9);
  const auto mundy_estimates = [&]<size_t K, bool Preconditioned>(bool is_warm) {
    Vector<double, N> largest = is_warm ? warm : Vector<double, N>::zeros();
    auto smallest = Vector<double, N>::zeros();
    mundy_basis_t<N, K> basis{};
    auto state = mundy_state<N, k, Preconditioned>(largest, smallest, basis);
    const auto result = solve_eigen_problem(make_eigen_problem<mm_backend_t>(LaplacianMundyOp<N>{1.0, Preconditioned}),
                                            mundy_lanczos_strategy<N, Preconditioned>(config), state);
    return std::tuple{result, largest, smallest};
  };
  const auto kokkos_estimates = [&]<size_t K, bool Preconditioned>(bool is_warm) {
    view_t largest = is_warm ? to_view(warm) : view_t("largest", N);
    view_t smallest("smallest", N);
    view_basis_t<K> basis("basis", N);
    auto state = view_state<Preconditioned>(N, k, largest, smallest, basis);
    auto problem = make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{N, 1.0, Preconditioned});
    const auto result = solve_eigen_problem(problem, view_lanczos_strategy<Preconditioned>(config, N), state);
    return std::tuple{result, to_vector<N>(largest), to_vector<N>(smallest)};
  };
  const auto expect_close = [](const auto& a, const auto& b, double tolerance) {
    for (size_t i = 0; i < N; ++i) {
      EXPECT_NEAR(std::get<1>(a)[i], std::get<1>(b)[i], tolerance) << "largest, entry " << i;
      EXPECT_NEAR(std::get<2>(a)[i], std::get<2>(b)[i], tolerance) << "smallest, entry " << i;
    }
  };
  const auto check = [&]<bool Preconditioned>(bool is_warm) {
    SCOPED_TRACE(::testing::Message() << (Preconditioned ? "preconditioned" : "unpreconditioned") << ", "
                                      << (is_warm ? "warm" : "cold"));
    // Unpreconditioned: each k-term sum rounds by k eps (|s| = 1, rows of V near unit), on both sides, and the
    // normalizations differ by the sums' difference over all N entries and by their own rounding, (N + 6) eps;
    // N k eps otherwise
    const double tolerance =
        Preconditioned ? N * k * eps : std::min(N * k * eps, (2 * k * (1 + std::sqrt(N)) + N + 6) * eps);
    const auto reference = mundy_estimates.template operator()<16, Preconditioned>(is_warm);
    ASSERT_EQ(std::get<0>(reference).largest.num_iters, k);
    ASSERT_EQ(std::get<0>(reference).smallest.num_iters, k);
    expect_close(mundy_estimates.template operator()<1, Preconditioned>(is_warm), reference, tolerance);
    expect_close(mundy_estimates.template operator()<2, Preconditioned>(is_warm), reference, tolerance);
    expect_close(mundy_estimates.template operator()<k - 1, Preconditioned>(is_warm), reference, tolerance);
    expect_close(mundy_estimates.template operator()<k, Preconditioned>(is_warm), reference, tolerance);

    const auto kokkos_reference = kokkos_estimates.template operator()<16, Preconditioned>(is_warm);
    expect_close(kokkos_estimates.template operator()<1, Preconditioned>(is_warm), kokkos_reference, tolerance);
    expect_close(kokkos_estimates.template operator()<2, Preconditioned>(is_warm), kokkos_reference, tolerance);
    expect_close(kokkos_estimates.template operator()<k - 1, Preconditioned>(is_warm), kokkos_reference, tolerance);
    expect_close(kokkos_estimates.template operator()<k, Preconditioned>(is_warm), kokkos_reference, tolerance);

    // The backends' T differ by rounding, which moves an estimate by a term of order k eps ||A|| / rho over the gap
    // between T's Ritz values (Paige), with no explicit constant, so this allowance is empirical; and they agree up to
    // sign: an eigenvector's sign is arbitrary
    const auto& result = std::get<0>(reference);
    const double drift = N * k * eps * norm_DLD_bound / std::min(result.largest.residual, result.smallest.residual);
    const auto signed_like = [](const auto& x, const auto& y) {
      using vector_t = std::remove_cvref_t<decltype(x)>;
      return dot(x, y) < 0.0 ? vector_t(-x) : vector_t(x);
    };
    expect_close(std::tuple{std::get<0>(kokkos_reference),
                            signed_like(std::get<1>(kokkos_reference), std::get<1>(reference)),
                            signed_like(std::get<2>(kokkos_reference), std::get<2>(reference))},
                 reference, drift);
  };
  check.template operator()<false>(false);
  check.template operator()<false>(true);
  check.template operator()<true>(false);
  check.template operator()<true>(true);

  {
    SCOPED_TRACE("a long solve");
    constexpr size_t n = 1000;
    constexpr unsigned long_k = 256;
    const view_t d = separated_top_diagonal(n, 10.0);
    auto problem = make_eigen_problem<kokkos_backend_t>(DiagonalKokkosOp{d});
    const LanczosConfig<double> long_config{.max_iters = long_k, .tol = 0.0};
    const auto estimates = [&]<size_t K, bool Preconditioned>() {
      view_t largest("largest", n), smallest("smallest", n);
      view_basis_t<K> basis("basis", n);
      auto state = view_state<Preconditioned>(n, long_k, largest, smallest, basis);
      const auto result = solve_eigen_problem(problem, view_lanczos_strategy<Preconditioned>(long_config, n), state);
      return std::tuple{result, to_host(largest), to_host(smallest),
                        eigen_of_t(state.alpha(), state.beta(), stop_of(result)),
                        stop_of(result) * eps * norm_of_t(state.alpha(), state.beta(), stop_of(result))};
    };
    const auto expect_same_estimates = [&](const auto& a, const auto& b) {
      ASSERT_EQ(stop_of(std::get<0>(a)), long_k);
      ASSERT_TRUE(std::get<0>(a).largest.converged) << "the top end must converge, so its eigenvalue repeats";
      const double tolerance = 4.0 * long_k * std::sqrt(static_cast<double>(long_k)) * eps;
      for (size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(std::get<1>(a)[i], std::get<1>(b)[i], tolerance) << "largest, entry " << i;
        ASSERT_NEAR(std::get<2>(a)[i], std::get<2>(b)[i], tolerance) << "smallest, entry " << i;
      }
    };
    // A basis of 1 or 2 resumes from the start's column, one of 4 from later Lanczos vectors
    const auto check = [&]<bool Preconditioned>() {
      SCOPED_TRACE(Preconditioned ? "preconditioned" : "unpreconditioned");
      const auto reference = estimates.template operator()<long_k, Preconditioned>();
      expect_same_estimates(estimates.template operator()<1, Preconditioned>(), reference);
      expect_same_estimates(estimates.template operator()<2, Preconditioned>(), reference);
      expect_same_estimates(estimates.template operator()<4, Preconditioned>(), reference);

      // Control: duplicate Ritz values, the two largest eigenvalues of T both within rho + k eps ||T|| of the top
      // end's theta
      const auto& top = std::get<0>(reference).largest;
      const auto& theta = std::get<3>(reference);
      const double floor = std::get<4>(reference);
      EXPECT_LE(std::abs(theta[long_k - 1] - top.eigenvalue), top.residual + floor);
      EXPECT_LE(std::abs(theta[long_k - 2] - top.eigenvalue), top.residual + floor);
    };
    check.template operator()<false>();
    check.template operator()<true>();
  }
}

// Columns past the solve's Lanczos vectors are never read: a basis filled with NaN beforehand gives the estimates, bit
// for bit, of one filled with zeros, on each backend.
TEST(Eigenvalues, LanczosBasisInvisible) {
  constexpr size_t N = 16;
  const LanczosConfig<double> config{.max_iters = 5, .tol = 0.0};
  const auto mundy_solve = [&](double fill) {
    auto largest = Vector<double, N>::zeros();
    auto smallest = Vector<double, N>::zeros();
    mundy_basis_t<N, 16> basis{};
    for (size_t j = 0; j < 16; ++j) {
      basis[j] = Vector<double, N>(fill);
    }
    auto state = mundy_state<N, 5>(largest, smallest, basis);
    solve_eigen_problem(make_eigen_problem<mm_backend_t>(laplacian<N>(1.0)), make_lanczos_strategy(config), state);
    return std::pair{largest, smallest};
  };
  const auto kokkos_solve = [&](double fill) {
    view_t largest("largest", N), smallest("smallest", N);
    view_basis_t<16> basis("basis", N);
    Kokkos::deep_copy(basis, fill);
    auto state = view_state(N, 5, largest, smallest, basis);
    solve_eigen_problem(make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{N, 1.0}), make_lanczos_strategy(config),
                        state);
    return std::pair{to_host(largest), to_host(smallest)};
  };
  const auto mundy_zero = mundy_solve(0.0);
  const auto mundy_nan = mundy_solve(nan);
  const auto kokkos_zero = kokkos_solve(0.0);
  const auto kokkos_nan = kokkos_solve(nan);
  EXPECT_TRUE(same_entries(mundy_nan.first, mundy_zero.first, N));
  EXPECT_TRUE(same_entries(mundy_nan.second, mundy_zero.second, N));
  EXPECT_TRUE(same_entries(kokkos_nan.first, kokkos_zero.first, N));
  EXPECT_TRUE(same_entries(kokkos_nan.second, kokkos_zero.second, N));
}

// An estimate's scale does not matter: e 2^560 and e 2^-560, whose squared norms overflow and underflow, give the
// results and estimates of e bit for bit, as rescaling by a power of two is exact; and e 2^-1060, whose entries are
// subnormal and keep about 14 significant bits, still starts the solve, which then meets its certificate.
TEST(Eigenvalues, LanczosEstimateScale) {
  constexpr size_t N = 32;
  constexpr unsigned max_iters = 64;
  const auto spectrum = separated_top_spectrum(1.0);
  auto problem = make_eigen_problem<mm_backend_t>(rotated<N>(spectrum));
  const auto strategy = make_lanczos_strategy(
      LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-10, .largest = true, .smallest = false});
  const Vector<double, N> e = near_second_eigenvector<N>();
  const auto solve = [&](int exponent) {
    Vector<double, N> estimate;
    for (size_t i = 0; i < N; ++i) {
      estimate[i] = std::ldexp(e[i], exponent);
    }
    mundy_basis_t<N, 8> basis{};
    auto state = mundy_state<N, max_iters>(estimate, NoVector{}, basis);
    const auto result = solve_eigen_problem(problem, strategy, state);
    return std::tuple{result, estimate, norm_of_t(state.alpha(), state.beta(), stop_of(result))};
  };
  const auto [unit, unit_estimate, unit_norm] = solve(0);
  ASSERT_TRUE(unit.largest.converged) << unit;
  for (const int exponent : {560, -560}) {
    SCOPED_TRACE(::testing::Message() << "2^" << exponent);
    const auto [scaled, scaled_estimate, scaled_norm] = solve(exponent);
    EXPECT_EQ(scaled.largest.num_iters, unit.largest.num_iters);
    EXPECT_EQ(scaled.largest.eigenvalue, unit.largest.eigenvalue);
    EXPECT_EQ(scaled.largest.residual, unit.largest.residual);
    EXPECT_TRUE(same_entries(scaled_estimate, unit_estimate, N));
  }
  {
    SCOPED_TRACE("subnormal");
    const auto [subnormal, subnormal_estimate, subnormal_norm] = solve(-1060);
    ASSERT_TRUE(subnormal.largest.converged) << subnormal;
    EXPECT_LE(std::abs(subnormal.largest.eigenvalue - spectrum[N - 1]),
              subnormal.largest.residual + subnormal.largest.num_iters * eps * subnormal_norm +
                  construction_error(spectrum));
  }
}

// Started at an eigenvector, a one-end solve converges at the first iteration to that eigenvector's eigenvalue and
// returns the eigenvector: at the end's own eigenvector, and at the opposite end's, whose eigenvalue the certificate
// then reports. A solve of no iterations leaves a nonzero estimate as it was.
TEST(Eigenvalues, LanczosEigenvectorStart) {
  constexpr size_t N = 16;
  auto problem = make_eigen_problem<mm_backend_t>(laplacian<N>(1.0));
  // A one-end solve from estimate, with a basis of 8, which holds that end's estimate on return
  const auto solve_one_end = [&](bool largest, Vector<double, N>& estimate, const LanczosConfig<double>& config) {
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{
        .max_iters = config.max_iters, .tol = config.tol, .largest = largest, .smallest = !largest});
    mundy_basis_t<N, 8> basis{};
    if (largest) {
      auto state = mundy_state<N, N>(estimate, NoVector{}, basis);
      return solve_eigen_problem(problem, strategy, state);
    }
    auto state = mundy_state<N, N>(NoVector{}, estimate, basis);
    return solve_eigen_problem(problem, strategy, state);
  };

  for (const bool largest : {true, false}) {
    for (const size_t k : {N, size_t{1}}) {
      SCOPED_TRACE(::testing::Message() << (largest ? "largest" : "smallest") << " end, eigenvector " << k);
      const auto eigenvector = laplacian_eigenvector<N>(k);
      Vector<double, N> estimate = eigenvector;
      const auto result = solve_one_end(largest, estimate, LanczosConfig<double>{.max_iters = N, .tol = 1e-10});
      const auto& end = largest ? result.largest : result.smallest;
      const double floor = N * eps * norm_L_bound;
      // theta to first order, for u = eps / 2; the start's distance from the eigenvector moves the Rayleigh quotient
      // only to second order. alpha = v^T A v rounds by the apply's 2 additions per row, 2 u ||L||, by the N-term
      // dot, N u lambda, and by the start's unit scaling, ||v||^2 = 1 + (N + 6) u. Bisection on the 1 x 1 T_1 ends
      // within its half-width eps |alpha| of alpha, plus u |alpha| for the midpoint. The reference 4 sin^2 rounds by
      // 11 u lambda: 3 u on the argument, 1 ulp on sin, 1 u on the product.
      const double lambda = laplacian_eigenvalue(N, k);
      const double eigenvalue_bound = (norm_L_bound + (N + 3) * lambda + 1.5 * lambda + 5.5 * lambda) * eps;
      // The estimate is the eigenvector rescaled to unit norm: its entries, at most sqrt(2 / (N + 1)), move by the
      // eigenvector's norm error, from sines whose arguments reach k pi, and by the normalizations' (N/2 + 6) eps
      const double entry_bound = std::sqrt(2.0 / (N + 1)) *
                                 (std::sqrt(2.0) * (3.0 * Kokkos::numbers::pi_v<double> * k + 4) + N / 2.0 + 6) * eps;
      EXPECT_TRUE(end.converged) << result;
      EXPECT_EQ(end.num_iters, 1u);
      EXPECT_NEAR(end.eigenvalue, lambda, eigenvalue_bound);
      for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(estimate[i], eigenvector[i], std::min(floor, entry_bound)) << "entry " << i;
      }
    }
  }
  {
    SCOPED_TRACE("no iterations");
    Vector<double, N> estimate = random_start<N>(17);
    const Vector<double, N> before = estimate;
    solve_one_end(true, estimate, LanczosConfig<double>{.max_iters = 0});
    EXPECT_TRUE(same_entries(estimate, before, N));
  }
}

// A warm estimate for one end speeds that end in a two-end solve, whose start also carries the cold start for the
// other end: on diag(1, 1 + 1/n, ..., 2.1), n = 10^5, the largest end started from the estimate of an earlier solve
// converges in fewer iterations than from the cold start, whose component along the top eigenvector is about
// 1 / sqrt(n).
TEST(Eigenvalues, LanczosWarmTwoEnd) {
  constexpr size_t n = 100000;
  constexpr unsigned max_iters = 100;
  auto problem = make_eigen_problem<kokkos_backend_t>(DiagonalKokkosOp{separated_top_diagonal(n, 2.1)});
  view_t estimate("estimate", n);
  view_basis_t<8> basis("basis", n);
  auto earlier = view_state(n, max_iters, estimate, NoVector{}, basis);
  const auto earlier_result = solve_eigen_problem(
      problem, make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-6, .smallest = false}),
      earlier);
  ASSERT_TRUE(earlier_result.largest.converged) << earlier_result;

  const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-8});
  auto cold = view_state(n, max_iters);
  auto warm = view_state(n, max_iters, estimate, NoVector{}, basis);
  const auto cold_result = solve_eigen_problem(problem, strategy, cold);
  const auto warm_result = solve_eigen_problem(problem, strategy, warm);
  ASSERT_TRUE(cold_result.largest.converged) << cold_result;
  ASSERT_TRUE(warm_result.largest.converged) << warm_result;
  EXPECT_LT(warm_result.largest.num_iters, cold_result.largest.num_iters);
}
//@}

//! \name Representations, kernels and allocations
//@{

// sign L and D (sign L) D with P = D^-2, as matrices and matrix-free on each backend: both ends converge within their
// certificates; and a Kokkos::Array scalar array runs as a host view does, results and estimates bit for bit. With P,
// the operator rounds relative to ||D L D|| <= 16 rather than ||T|| ~ ||L||, which n eps ||D L D|| allows for.
TEST(Eigenvalues, LanczosRepresentations) {
  // Each end's certificate uses the norm of the T it was measured on
  const auto expect_ends = [](const EigenBounds<double>& result, double largest_norm, double smallest_norm, size_t n,
                              double sign, double construction) {
    SCOPED_TRACE(::testing::Message() << "n=" << n << " sign=" << sign);
    ASSERT_TRUE(result.largest.converged && result.smallest.converged) << result;
    const double smallest = sign > 0.0 ? laplacian_eigenvalue(n, 1) : -laplacian_eigenvalue(n, n);
    const double largest = sign > 0.0 ? laplacian_eigenvalue(n, n) : -laplacian_eigenvalue(n, 1);
    EXPECT_NEAR(result.smallest.eigenvalue, smallest,
                result.smallest.residual + result.smallest.num_iters * eps * smallest_norm + construction);
    EXPECT_NEAR(result.largest.eigenvalue, largest,
                result.largest.residual + result.largest.num_iters * eps * largest_norm + construction);
  };
  const auto end_norms = [](const auto& state, const EigenBounds<double>& result) {
    return std::pair{norm_of_t(state.alpha(), state.beta(), result.largest.num_iters),
                     norm_of_t(state.alpha(), state.beta(), result.smallest.num_iters)};
  };

  {
    SCOPED_TRACE("MundyMathBackend");
    const auto check = [&]<size_t N, bool Preconditioned>(double sign) {
      const LanczosConfig<double> config{.max_iters = 4 * N, .tol = 1e-12};
      const double construction = Preconditioned ? norm_DLD_bound * N * eps : 0.0;
      const auto solve = [&](const auto& A) {
        using op_t = std::remove_cvref_t<decltype(A)>;
        auto state = mundy_state<N, 4 * N, Preconditioned>();
        auto problem = make_eigen_problem<mm_backend_t>(op_t(A));
        const auto result = solve_eigen_problem(problem, mundy_lanczos_strategy<N, Preconditioned>(config), state);
        return std::pair{result, end_norms(state, result)};
      };
      const auto matrix = Preconditioned ? congruent<N>(laplacian<N>(sign)) : laplacian<N>(sign);
      const auto [matrix_result, matrix_norms] = solve(matrix);
      const auto [free_result, free_norms] = solve(LaplacianMundyOp<N>{sign, Preconditioned});
      return std::tuple{matrix_result, matrix_norms, free_result, free_norms, construction};
    };
    const auto expect_both = [&](const auto& solved, size_t n, double sign) {
      const auto& [matrix_result, matrix_norms, free_result, free_norms, construction] = solved;
      expect_ends(matrix_result, matrix_norms.first, matrix_norms.second, n, sign, construction);
      expect_ends(free_result, free_norms.first, free_norms.second, n, sign, construction);
    };
    for (const double sign : {1.0, -1.0}) {
      expect_both(check.template operator()<4, false>(sign), 4, sign);
      expect_both(check.template operator()<8, false>(sign), 8, sign);
      expect_both(check.template operator()<16, false>(sign), 16, sign);
      expect_both(check.template operator()<4, true>(sign), 4, sign);
      expect_both(check.template operator()<8, true>(sign), 8, sign);
      expect_both(check.template operator()<16, true>(sign), 16, sign);
    }
  }
  {
    SCOPED_TRACE("KokkosBackend");
    const auto check = [&]<bool Preconditioned>(size_t n, double sign) {
      const LanczosConfig<double> config{.max_iters = static_cast<unsigned>(10 * n), .tol = 1e-10};
      const double construction = Preconditioned ? norm_DLD_bound * n * eps : 0.0;
      const LaplacianKokkosOp op{n, sign, Preconditioned};
      const auto solve = [&](const auto& A, const char* representation) {
        SCOPED_TRACE(representation);
        using op_t = std::remove_cvref_t<decltype(A)>;
        auto state = view_state<Preconditioned>(n, config.max_iters);
        auto problem = make_eigen_problem<kokkos_backend_t>(op_t(A));
        const auto result = solve_eigen_problem(problem, view_lanczos_strategy<Preconditioned>(config, n), state);
        const auto [largest_norm, smallest_norm] = end_norms(state, result);
        expect_ends(result, largest_norm, smallest_norm, n, sign, construction);
      };
      solve(op, "matrix-free operator");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
      const auto dense = make_dense_laplacian(op);
      solve(dense, "dense matrix");
      solve(make_sparse_matrix<sparse_matrix_t>(dense), "sparse matrix");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
    };
    for (const double sign : {1.0, -1.0}) {
      check.template operator()<false>(16, sign);
      check.template operator()<false>(200, sign);
      check.template operator()<true>(16, sign);
      check.template operator()<true>(100, sign);
    }
  }
  {
    SCOPED_TRACE("scalar arrays");
    constexpr size_t N = 8;
    constexpr unsigned max_iters = 16;
    auto problem = make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{N, 1.0});
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = max_iters, .tol = 1e-12});
    const auto solve = [&](auto scalars) {
      view_t largest("largest", N), smallest("smallest", N);
      view_basis_t<4> basis("basis", N);
      auto state = make_lanczos_state(view_t("v", N), view_t("Bv_prev", N), view_t("Av", N), NoVector{},
                                      std::move(scalars), largest, smallest, basis);
      const auto result = solve_eigen_problem(problem, strategy, state);
      return std::tuple{result, to_host(largest), to_host(smallest)};
    };
    // (4 + e) max_iters scalars for e = 2 estimate vectors
    const auto [view_result, view_largest, view_smallest] = solve(host_array_t("scalars", (4 + 2) * max_iters));
    const auto [array_result, array_largest, array_smallest] = solve(Kokkos::Array<double, (4 + 2) * max_iters>{});
    for (const bool largest : {true, false}) {
      const auto& a = largest ? array_result.largest : array_result.smallest;
      const auto& b = largest ? view_result.largest : view_result.smallest;
      EXPECT_EQ(a.num_iters, b.num_iters);
      EXPECT_EQ(a.eigenvalue, b.eigenvalue);
      EXPECT_EQ(a.residual, b.residual);
      EXPECT_EQ(a.converged, b.converged);
    }
    EXPECT_TRUE(same_entries(array_largest, view_largest, N));
    EXPECT_TRUE(same_entries(array_smallest, view_smallest, N));
  }
}

// Inside a kernel, with Kokkos::Array scalars and basis: both ends of (e + 1) L within their certificates, with
// ||(e + 1) L|| <= 4 (e + 1) for ||T||, and the largest end's estimate its eigenvector to (rho / gap)^2.
TEST(Eigenvalues, LanczosInKernel) {
  using column = LanczosKernelColumn;
  constexpr size_t N = 4;
  constexpr size_t num_entities = 8;
  Kokkos::View<double* [column::count], memory_space_t> out("out", num_entities);
  solve_lanczos_ends_in_kernel(out);
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  for (size_t e = 0; e < num_entities; ++e) {
    SCOPED_TRACE(::testing::Message() << "entity " << e);
    const double scale = static_cast<double>(e + 1);
    const double norm_bound = norm_L_bound * scale;
    const double floor = host(e, column::largest_num_iters) * eps * norm_bound;
    const double gap = scale * (laplacian_eigenvalue(N, N) - laplacian_eigenvalue(N, N - 1));
    const double rho = host(e, column::largest_residual);
    // 1 - |e^T q_N| <= sin^2 <= ((rho + floor) / gap)^2, plus the rounding of e's norm, of the N-term dot, and of q_N's
    // sines, whose arguments reach N pi
    const double sine = (rho + floor) / gap;
    const double overlap_bound =
        sine * sine + (1.5 * N + 3 + std::sqrt(2.0) * (3.0 * Kokkos::numbers::pi_v<double> * N + 4)) * eps;
    EXPECT_EQ(host(e, column::both_converged), 1.0);
    EXPECT_NEAR(host(e, column::largest_eigenvalue), scale * laplacian_eigenvalue(N, N), rho + floor);
    EXPECT_NEAR(host(e, column::smallest_eigenvalue), scale * laplacian_eigenvalue(N, 1),
                host(e, column::smallest_residual) + floor);
    EXPECT_NEAR(host(e, column::top_overlap), 1.0, std::min((rho / gap) * (rho / gap) + N * floor, overlap_bound));
  }
}

// After a first solve, a second one on the same state, estimates, and basis, initialize included, allocates nothing,
// with and without a preconditioner; an allocation is counted.
TEST(Eigenvalues, LanczosAllocatesNothing) {
  constexpr size_t n = 64;
  constexpr unsigned max_iters = 32;
  EXPECT_GT(count_allocations([] { view_t probe("probe", 1); }), 0u);

  auto problem = make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{n, 1.0});
  const LanczosConfig<double> config{.max_iters = max_iters, .tol = 1e-8};
  view_t largest("largest", n), smallest("smallest", n);
  view_basis_t<8> basis("basis", n);
  {
    SCOPED_TRACE("unpreconditioned");
    auto state = view_state(n, max_iters, largest, smallest, basis);
    const auto strategy = make_lanczos_strategy(config);
    ASSERT_GT(stop_of(solve_eigen_problem(problem, strategy, state)), 8u) << "the solve must outrun the basis";
    EXPECT_EQ(count_allocations([&] { solve_eigen_problem(problem, strategy, state); }), 0u);
  }
  {
    SCOPED_TRACE("preconditioned");
    auto congruent_problem = make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{n, 1.0, true});
    auto state = view_state<true>(n, max_iters, largest, smallest, basis);
    const auto strategy = make_lanczos_strategy(config, make_congruence_preconditioner(n));
    ASSERT_GT(stop_of(solve_eigen_problem(congruent_problem, strategy, state)), 8u);
    EXPECT_EQ(count_allocations([&] { solve_eigen_problem(congruent_problem, strategy, state); }), 0u);
  }
}

TEST(Eigenvalues, LanczosRejections) {
  auto problem = make_eigen_problem<mm_backend_t>(laplacian<3>(1.0));
  {
    SCOPED_TRACE("no end wanted");
    auto state = mundy_state<3, 6>();
    EXPECT_THROW(make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6, .largest = false, .smallest = false})
                     .initialize(problem, state),
                 std::invalid_argument);
  }
  {
    SCOPED_TRACE("an estimate for an unwanted end");
    Vector3d estimate{1.0, 0.3, -0.2};
    mundy_basis_t<3, 2> basis{};
    auto with_largest = mundy_state<3, 6>(estimate, NoVector{}, basis);
    auto with_smallest = mundy_state<3, 6>(NoVector{}, estimate, basis);
    EXPECT_THROW(make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6, .largest = false})
                     .initialize(problem, with_largest),
                 std::invalid_argument);
    EXPECT_THROW(make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6, .smallest = false})
                     .initialize(problem, with_smallest),
                 std::invalid_argument);
  }
  {
    SCOPED_TRACE("a scalar array short of max_iters");
    const auto strategy = make_lanczos_strategy(LanczosConfig<double>{.max_iters = 6});
    auto short_array = make_lanczos_state(Vector3d{}, Vector3d{}, Vector3d{}, NoVector{},
                                          Kokkos::Array<double, 4 * 6 - 1>{});
    EXPECT_THROW(strategy.initialize(problem, short_array), std::invalid_argument);
    auto kokkos_problem = make_eigen_problem<kokkos_backend_t>(LaplacianKokkosOp{3, 1.0});
    auto short_view = make_lanczos_state(view_t("v", 3), view_t("Bv_prev", 3), view_t("Av", 3), NoVector{},
                                         host_array_t("scalars", 4 * 6 - 1));
    EXPECT_THROW(strategy.initialize(kokkos_problem, short_view), std::invalid_argument);
  }
}
//@}

}  // namespace

}  // namespace mundy
