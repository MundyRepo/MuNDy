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

/// \file UnitTestMueLuPreconditioner.cpp
/// \brief The multigrid preconditioner (muelu_preconditioner.hpp) on the 2D five-point Dirichlet Laplacian: exact on
/// one level and on its near-null space, a fixed symmetric positive definite linear map, update equals construction,
/// and preconditioned CG.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{MUELU,TPETRA,KOKKOSKERNELS}

#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)

#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <Tpetra_Map.hpp>             // for Tpetra::Map<>::node_type

// C++ core
#include <cmath>     // for std::sin, std::tan, std::pow, std::abs, std::sqrt
#include <iostream>  // for std::cout
#include <limits>    // for std::numeric_limits
#include <string>    // for std::string
#include <vector>    // for std::vector

// Mundy
#include <mundy_math/linear_system.hpp>         // for mundy::{LinearSystem, CGStrategy, CGState, solve_linear_system}
#include <mundy_math/muelu_preconditioner.hpp>  // for mundy::{MueLuConfig, make_muelu_preconditioner}
#include <mundy_math/preconditioners.hpp>       // for mundy::Preconditioner
#include <mundy_math/residuals.hpp>             // for mundy::L2Residual
#include <mundy_math/solver_backends.hpp>       // for mundy::KokkosBackend
#include <mundy_math/sparse_matrix.hpp>         // for mundy::impl::{SparseEntry, make_sparse_matrix}

namespace mundy {

namespace {

//! \name Shared helpers
//@{

// The Kokkos spaces of Tpetra's default Node, which the preconditioner's backend must share.
using exec_space = Tpetra::Map<>::node_type::execution_space;
using mem_space = Tpetra::Map<>::node_type::memory_space;
using backend_t = KokkosBackend<exec_space>;
using vector_t = Kokkos::View<double*, mem_space>;
using sparse_matrix_t = KokkosSparse::CrsMatrix<double, int, Kokkos::Device<exec_space, mem_space>, void, size_t>;
using int_offset_matrix_t = KokkosSparse::CrsMatrix<double, int, Kokkos::Device<exec_space, mem_space>, void, int>;
using preconditioner_t = MueLuPreconditioner<backend_t, sparse_matrix_t>;

static_assert(Preconditioner<preconditioner_t, backend_t, vector_t>,
              "The MueLu preconditioner must satisfy Preconditioner under KokkosBackend");

constexpr double eps = std::numeric_limits<double>::epsilon();

/// \brief scale times the five-point Dirichlet Laplacian on a k x k grid: 4 on the diagonal, -1 for each neighbor.
template <class Matrix = sparse_matrix_t>
Matrix laplacian(size_t k, double scale = 1.0) {
  std::vector<impl::SparseEntry<double>> entries;
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j < k; ++j) {
      const size_t row = i * k + j;
      if (i > 0) entries.push_back({row, row - k, -scale});
      if (j > 0) entries.push_back({row, row - 1, -scale});
      entries.push_back({row, row, 4.0 * scale});
      if (j + 1 < k) entries.push_back({row, row + 1, -scale});
      if (i + 1 < k) entries.push_back({row, row + k, -scale});
    }
  }
  return impl::make_sparse_matrix<Matrix>(k * k, k * k, entries);
}

/// \brief The smallest eigenvalue of the k x k five-point Laplacian, 8 sin^2(pi / (2 (k + 1))).
double laplacian_min_eigenvalue(size_t k) {
  const double s = std::sin(M_PI / (2.0 * (k + 1)));
  return 8.0 * s * s;
}

/// \brief The near-null space {1, x} of the k x k grid, with x = j / k at the point (i, j) of row i k + j.
Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> constant_and_linear(size_t k) {
  const size_t n = k * k;
  Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> vectors("constant_and_linear", n, 2);
  const auto vectors_host = Kokkos::create_mirror_view(vectors);
  for (size_t row = 0; row < n; ++row) {
    vectors_host(row, 0) = 1.0;
    vectors_host(row, 1) = static_cast<double>(row % k) / static_cast<double>(k);
  }
  Kokkos::deep_copy(vectors, vectors_host);
  return vectors;
}

/// \brief A vector of n entries sin(seed + 0.37 i): fixed, and rich in every mode of the Laplacian.
vector_t fixed_vector(size_t n, double seed) {
  vector_t v("v", n);
  const auto v_host = Kokkos::create_mirror_view(v);
  for (size_t i = 0; i < n; ++i) {
    v_host(i) = std::sin(seed + 0.37 * static_cast<double>(i));
  }
  Kokkos::deep_copy(v, v_host);
  return v;
}

/// \brief The entries of a vector, on the host.
std::vector<double> to_host(const vector_t& v) {
  const auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  return std::vector<double>(v_host.data(), v_host.data() + v_host.extent(0));
}

/// \brief The entries of op applied to x, on the host.
template <class Op>
std::vector<double> applied(const Op& op, const vector_t& x) {
  vector_t y("y", x.extent(0));
  backend_t::apply(op, x, y);
  return to_host(y);
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

double norm_inf(const std::vector<double>& a) {
  double max = 0.0;
  for (const double value : a) {
    max = std::max(max, std::abs(value));
  }
  return max;
}
//@}

//! \name The preconditioner as a linear map
//@{

// With the coarse limit at least the matrix size, the hierarchy is one level solved directly, so P = A^-1: P r solves
// A z = r to the backward error of an LU solve, |A z - r| <= c n eps |A| |z| with c = 10 covering pivot growth.
TEST(MueLu, ExactOnOneLevel) {
  const size_t k = 8;
  const size_t n = k * k;
  const sparse_matrix_t A = laplacian(k);
  const auto P = make_muelu_preconditioner<backend_t>(A, MueLuConfig<double>{});

  // z = P r, then A z
  const vector_t r = fixed_vector(n, 1.0);
  vector_t z("z", n);
  backend_t::apply(P, r, z);
  const std::vector<double> az = applied(A, z);

  // A z = r
  const std::vector<double> r_host = to_host(r);
  const double bound = 10.0 * n * eps * 8.0 * norm_inf(to_host(z));  // |A|_inf = 8
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(az[i], r_host[i], bound) << "entry " << i;
  }
}

// On two levels without smoothing, with the unsmoothed prolongation Q, whose orthonormal columns span the near-null
// space on each aggregate, P = Q (Q^T A Q)^-1 Q^T is the A-orthogonal projection onto the coarse space: P A b = b for
// every near-null vector b. The coarse matrix's condition number is at most A's, kappa = cot^2(pi / (2 (k + 1))), so
// |P A b - b|_2 <= c n eps kappa |b|_2, with c = 10 covering the products and pivot growth.
TEST(MueLu, ExactOnNearNullSpace) {
  const size_t k = 16;
  const size_t n = k * k;
  const sparse_matrix_t A = laplacian(k);
  const auto near_null_space = constant_and_linear(k);
  MueLuConfig<double> cfg;
  cfg.max_levels = 2;
  cfg.coarse_max_size = 10;
  cfg.extra = Teuchos::rcp(new Teuchos::ParameterList());
  cfg.extra->set("sa: damping factor", 0.0);
  cfg.extra->set("smoother: pre or post", std::string("none"));
  const auto P = make_muelu_preconditioner<backend_t>(A, near_null_space, cfg);

  const double kappa = 1.0 / std::pow(std::tan(M_PI / (2.0 * (k + 1))), 2);
  for (size_t column = 0; column < 2; ++column) {
    // P A b
    vector_t b("b", n);
    Kokkos::deep_copy(b, Kokkos::subview(near_null_space, Kokkos::ALL(), column));
    vector_t ab("ab", n);
    backend_t::apply(A, b, ab);
    const std::vector<double> pab = applied(P, ab);

    // P A b = b
    const std::vector<double> b_host = to_host(b);
    double error_squared = 0.0;
    for (size_t i = 0; i < n; ++i) {
      error_squared += (pab[i] - b_host[i]) * (pab[i] - b_host[i]);
    }
    EXPECT_LE(std::sqrt(error_squared), 10.0 * n * eps * kappa * std::sqrt(dot(b_host, b_host)))
        << "near-null vector " << column;
  }
}

// On several levels, P is a fixed linear map, symmetric and positive definite: applying it twice gives the same bits,
// it is linear, s^T P r = r^T P s, and r^T P r > 0. Each comparison is bounded by the rounding of the cycle's sums, c n
// eps times the size of the terms compared, with c = 10.
TEST(MueLu, SymmetricFixedLinearMap) {
  const size_t k = 16;
  const size_t n = k * k;
  MueLuConfig<double> cfg;
  cfg.coarse_max_size = 10;
  const auto P = make_muelu_preconditioner<backend_t>(laplacian(k), cfg);
  const vector_t r = fixed_vector(n, 1.0);
  const vector_t s = fixed_vector(n, 2.0);

  // Twice
  const std::vector<double> pr = applied(P, r);
  EXPECT_EQ(applied(P, r), pr);

  // Linear: P (2 r - 3 s) = 2 P r - 3 P s
  const std::vector<double> ps = applied(P, s);
  vector_t combination("combination", n);
  backend_t::deep_copy(combination, r);
  backend_t::axpby(-3.0, s, 2.0, combination);
  const std::vector<double> p_combination = applied(P, combination);
  const double linearity_bound = 10.0 * n * eps * (2.0 * norm_inf(pr) + 3.0 * norm_inf(ps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(p_combination[i], 2.0 * pr[i] - 3.0 * ps[i], linearity_bound) << "entry " << i;
  }

  // Symmetric and positive definite
  const std::vector<double> r_host = to_host(r);
  const std::vector<double> s_host = to_host(s);
  const double symmetry_bound =
      10.0 * n * eps * (std::sqrt(dot(s_host, s_host) * dot(pr, pr)) + std::sqrt(dot(r_host, r_host) * dot(ps, ps)));
  EXPECT_NEAR(dot(s_host, pr), dot(r_host, ps), symmetry_bound);
  EXPECT_GT(dot(r_host, pr), 0.0);
  EXPECT_GT(dot(s_host, ps), 0.0);
}
//@}

//! \name Construction and update
//@{

// With deterministic aggregation, the preconditioner updated to B is the one made from B, with the default and with a
// given near-null space, and the preconditioner of 2 A is half that of A. Setup sums in an order that can vary from
// run to run, so each comparison holds to rounding: c n eps times the size of the terms compared, with c = 10.
TEST(MueLu, UpdateEqualsConstruction) {
  const size_t k = 16;
  const size_t n = k * k;
  MueLuConfig<double> cfg;
  cfg.coarse_max_size = 10;
  cfg.deterministic = true;
  const sparse_matrix_t A = laplacian(k);
  const sparse_matrix_t B = laplacian(k, 2.0);
  const vector_t r = fixed_vector(n, 1.0);

  // Default near-null space
  auto updated = make_muelu_preconditioner<backend_t>(A, cfg);
  const std::vector<double> pa = applied(updated, r);
  updated.update(B);
  const std::vector<double> pb_updated = applied(updated, r);
  const std::vector<double> pb = applied(make_muelu_preconditioner<backend_t>(B, cfg), r);
  const double bound = 10.0 * n * eps * norm_inf(pa);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(pb_updated[i], pb[i], bound) << "entry " << i;
    EXPECT_NEAR(pb[i], 0.5 * pa[i], bound) << "entry " << i;
  }

  // Constant and linear near-null space
  const auto near_null_space = constant_and_linear(k);
  auto updated_with_space = make_muelu_preconditioner<backend_t>(A, near_null_space, cfg);
  updated_with_space.update(B, near_null_space);
  const std::vector<double> pb_space_updated = applied(updated_with_space, r);
  const std::vector<double> pb_space = applied(make_muelu_preconditioner<backend_t>(B, near_null_space, cfg), r);
  const double space_bound = 10.0 * n * eps * norm_inf(pb_space);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(pb_space_updated[i], pb_space[i], space_bound) << "entry " << i;
  }
}

// The constant vector is the default near-null space, and a matrix whose index types differ from Tpetra's gives the
// same preconditioner; both with deterministic aggregation and to rounding, c n eps times the size of the terms
// compared, with c = 10.
TEST(MueLu, DefaultsAndMatrixTypes) {
  const size_t k = 16;
  const size_t n = k * k;
  MueLuConfig<double> cfg;
  cfg.coarse_max_size = 10;
  cfg.deterministic = true;
  const sparse_matrix_t A = laplacian(k);
  const vector_t r = fixed_vector(n, 1.0);
  const std::vector<double> pr = applied(make_muelu_preconditioner<backend_t>(A, cfg), r);
  const double bound = 10.0 * n * eps * norm_inf(pr);

  // The constant near-null space
  Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> constant("constant", n, 1);
  Kokkos::deep_copy(constant, 1.0);
  const std::vector<double> p_constant = applied(make_muelu_preconditioner<backend_t>(A, constant, cfg), r);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(p_constant[i], pr[i], bound) << "entry " << i;
  }

  // int row offsets
  const std::vector<double> p_int_offsets =
      applied(make_muelu_preconditioner<backend_t>(laplacian<int_offset_matrix_t>(k), cfg), r);
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(p_int_offsets[i], pr[i], bound) << "entry " << i;
  }
}
//@}

//! \name Preconditioned CG
//@{

// CG preconditioned by P solves A u = b for b = A u_exact on refined grids. CG stops at |b - A u|_2 <= tol, so
// |u - u_exact|_2 <= tol / lambda_min, with lambda_min the Laplacian's exact smallest eigenvalue. tol = 1e-8 is above
// the true-residual floor eps kappa |A| |u| at these sizes. Iteration counts are reported, not asserted.
TEST(MueLu, PreconditionedCGSolvesPoisson) {
  const double tol = 1e-8;
  for (const size_t k : {16, 32, 64}) {
    const size_t n = k * k;
    MueLuConfig<double> cfg;
    cfg.coarse_max_size = 10;
    const sparse_matrix_t A = laplacian(k);
    const auto P = make_muelu_preconditioner<backend_t>(A, cfg);

    // b = A u_exact
    const vector_t u_exact = fixed_vector(n, 3.0);
    vector_t b("b", n);
    backend_t::apply(A, u_exact, b);

    // Solve from zero
    auto state = make_cg_state(vector_t("u", n), vector_t("r", n), vector_t("p", n), vector_t("Ap", n));
    const auto result =
        solve_linear_system(make_linear_system<backend_t>(A, b),
                            make_cg_solution_strategy(L2Residual{}, CGConfig<double>{1000, tol}, P), state);
    std::cout << "k = " << k << ": " << result.num_iters << " preconditioned CG iterations" << std::endl;

    // u = u_exact
    EXPECT_TRUE(result.converged) << "k = " << k;
    const std::vector<double> u = to_host(state.x());
    const std::vector<double> u_exact_host = to_host(u_exact);
    double error_squared = 0.0;
    for (size_t i = 0; i < n; ++i) {
      error_squared += (u[i] - u_exact_host[i]) * (u[i] - u_exact_host[i]);
    }
    EXPECT_LE(std::sqrt(error_squared), tol / laplacian_min_eigenvalue(k)) << "k = " << k;
  }
}
//@}

}  // namespace

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_MUELU && HAVE_MUNDYMATH_TPETRA && HAVE_MUNDYMATH_KOKKOSKERNELS
