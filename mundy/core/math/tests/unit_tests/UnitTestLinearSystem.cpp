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

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>      // for std::ldexp
#include <cstdint>    // for uint64_t
#include <limits>     // for std::numeric_limits
#include <stdexcept>  // for std::runtime_error
#include <type_traits>  // for std::remove_cvref_t

// Mundy
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS
#include <mundy_math/Matrix.hpp>
#include <mundy_math/Vector.hpp>
#include <mundy_math/linear_system.hpp>
#include <mundy_math/preconditioners.hpp>
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix

namespace mundy {

namespace {

// A small, fixed, well-conditioned SPD system solved via mundy::MundyMathBackend: A is the classic
// tridiagonal [2,-1,0;-1,2,-1;0,-1,2], b chosen so the exact solution is (1, 0, 1).
Matrix3d spd_matrix() {
  return Matrix3d{2.0,  -1.0, 0.0,   //
                  -1.0, 2.0,  -1.0,  //
                  0.0,  -1.0, 2.0};
}

Vector3d spd_rhs() {
  return spd_matrix() * Vector3d{1.0, 0.0, 1.0};
}

using mm_backend_t = MundyMathBackend;

static_assert(VectorResidualPolicy<L2Residual, mm_backend_t, Vector3d, Vector3d>);
static_assert(VectorResidualPolicy<RelativeL2Residual, mm_backend_t, Vector3d, Vector3d>);
static_assert(VectorResidualPolicy<LinfResidual, mm_backend_t, Vector3d, Vector3d>);
static_assert(Preconditioner<NoPreconditioner, mm_backend_t, Vector3d>);
static_assert(Preconditioner<JacobiPreconditioner<mm_backend_t, Vector3d>, mm_backend_t, Vector3d>);
static_assert(!Preconditioner<bool, mm_backend_t, Vector3d>);

using Vector7d = Vector<double, 7>;
using Matrix7d = Matrix<double, 7, 7>;

/// \brief A symmetric, strictly diagonally dominant (so SPD) 7x7 matrix with exactly representable entries.
Matrix7d spd_matrix7() {
  Matrix7d A = Matrix7d();
  for (size_t i = 0; i < 7; ++i) {
    A(i, i) = 4.0 + static_cast<double>(i);
  }
  for (size_t i = 0; i + 1 < 7; ++i) {
    A(i, i + 1) = -1.0 - 0.25 * static_cast<double>(i);
    A(i + 1, i) = A(i, i + 1);
  }
  return A;
}

/// \brief The diagonal of a 7x7 matrix.
Vector7d diagonal7(const Matrix7d& A) {
  Vector7d d = Vector7d();
  for (size_t i = 0; i < 7; ++i) {
    d[i] = A(i, i);
  }
  return d;
}

// Solves spd_matrix() x = spd_rhs() inside a kernel, constructing the LinearSystem there.
void solve_spd_in_kernel(const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>& x,
                         const Kokkos::View<bool, Kokkos::DefaultExecutionSpace::memory_space>& converged) {
  Kokkos::parallel_for(
      "solve_spd_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int) {
        const Matrix3d A{2.0,  -1.0, 0.0,   //
                         -1.0, 2.0,  -1.0,  //
                         0.0,  -1.0, 2.0};
        auto prob = make_linear_system<mm_backend_t>(Matrix3d(A), Vector3d(A * Vector3d{1.0, 0.0, 1.0}));
        auto state = make_cg_state(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{});
        const auto result = solve_linear_system(prob, make_cg_solution_strategy(CGConfig<double>{}), state);
        for (size_t i = 0; i < 3; ++i) {
          x(i) = state.x()[i];
        }
        converged() = result.converged;
      });
}

// Applies plain and Jacobi-preconditioned inverses of spd_matrix() to one rhs inside a kernel, the plain one twice:
// from zero, then from its solution. The inverses are const, so all their state changes go through mutable members.
// Solution k is x(3k), ..., x(3k + 2) and took iters(k) iterations.
void apply_cg_inv_in_kernel(const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>& x,
                            const Kokkos::View<unsigned*, Kokkos::DefaultExecutionSpace::memory_space>& iters) {
  Kokkos::parallel_for(
      "apply_cg_inv_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int) {
        const Matrix3d A{2.0,  -1.0, 0.0,   //
                         -1.0, 2.0,  -1.0,  //
                         0.0,  -1.0, 2.0};
        const Vector3d rhs{0.3, -0.1, 0.7};
        const auto plain = make_cg_inv_op<mm_backend_t>(Matrix3d(A), CGConfig<double>{});
        const auto jacobi = make_cg_inv_op<mm_backend_t>(
            Matrix3d(A), CGConfig<double>{}, make_jacobi_preconditioner<mm_backend_t>(Vector3d{2.0, 2.0, 2.0}));
        Vector3d out[3] = {Vector3d{0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 0.0}, Vector3d{0.0, 0.0, 0.0}};
        plain.apply(rhs, out[0]);
        iters(0) = plain.last_result().num_iters;
        out[1] = out[0];
        plain.apply(rhs, out[1]);
        iters(1) = plain.last_result().num_iters;
        jacobi.apply(rhs, out[2]);
        iters(2) = jacobi.last_result().num_iters;
        for (int k = 0; k < 3; ++k) {
          for (int i = 0; i < 3; ++i) {
            x(3 * k + i) = out[k][i];
          }
        }
      });
}

TEST(LinearSystem, MundyMathBackendConvergesToKnownSolution) {
  const Matrix3d A = spd_matrix();
  const Vector3d b = spd_rhs();

  auto prob = LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(b));
  Vector3d x{0.0, 0.0, 0.0};
  auto state = CGState(Vector3d(x), Vector3d{}, Vector3d{}, Vector3d{});
  auto strat = CGStrategy(L2Residual{}, CGConfig<double>{});

  const auto result = solve_linear_system(prob, strat, state);
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, 3u);  // exact CG property: dim(A) = 3
  EXPECT_NEAR(state.x()[0], 1.0, 1e-8);
  EXPECT_NEAR(state.x()[1], 0.0, 1e-8);
  EXPECT_NEAR(state.x()[2], 1.0, 1e-8);
}

TEST(LinearSystem, DiagonalSystemConvergesWithinDimensionIterations) {
  const Matrix3d A{3.0, 0.0, 0.0,  //
                   0.0, 5.0, 0.0,  //
                   0.0, 0.0, 7.0};
  const Vector3d x_exact{1.0, -2.0, 0.5};
  const Vector3d b = A * x_exact;

  auto prob = LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(b));
  auto state = CGState(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{});
  auto strat = CGStrategy(L2Residual{}, CGConfig<double>{});

  const auto result = solve_linear_system(prob, strat, state);
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, 3u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(state.x()[i], x_exact[i], 1e-8);
  }
}

TEST(LinearSystem, WarmStartFromNearbySolutionConvergesFaster) {
  const Matrix3d A = spd_matrix();
  const Vector3d b1 = spd_rhs();
  // A small perturbation to the rhs -- the exact solution barely moves, so starting from the previous solution
  // (warm start) should need fewer iterations than starting cold (x0 = 0) on the same perturbed system.
  const Vector3d b2 = b1 + Vector3d{1e-6, -1e-6, 1e-6};

  auto prob1 = LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(b1));
  auto state1 = CGState(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{});
  auto strat = CGStrategy(L2Residual{}, CGConfig<double>{});
  const auto result1 = solve_linear_system(prob1, strat, state1);
  ASSERT_TRUE(result1.converged);

  // Warm start: reuse state1.x() (the converged solution to b1) as the initial guess for b2.
  auto prob_warm = LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(b2));
  auto state_warm = CGState(Vector3d(state1.x()), Vector3d{}, Vector3d{}, Vector3d{});
  const auto result_warm = solve_linear_system(prob_warm, strat, state_warm);
  ASSERT_TRUE(result_warm.converged);

  // Cold start: solve the same perturbed system from x0 = 0.
  auto prob_cold = LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(b2));
  auto state_cold = CGState(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{});
  const auto result_cold = solve_linear_system(prob_cold, strat, state_cold);
  ASSERT_TRUE(result_cold.converged);

  EXPECT_LE(result_warm.num_iters, result_cold.num_iters);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(state_warm.x()[i], state_cold.x()[i], 1e-6);
  }
}

TEST(LinearSystem, ResidualPolicyChoiceDoesNotChangeIterates) {
  const Matrix3d A = spd_matrix();
  const Vector3d b = spd_rhs();
  CGConfig<double> cfg;

  auto solve_with = [&](auto residual_policy) {
    auto prob = LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(b));
    auto state = CGState(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{});
    auto strat = CGStrategy(residual_policy, cfg);
    const auto result = solve_linear_system(prob, strat, state);
    return std::make_pair(result, state.x());
  };

  const auto [result_l2, x_l2] = solve_with(L2Residual{});
  const auto [result_rel, x_rel] = solve_with(RelativeL2Residual{});
  const auto [result_linf, x_linf] = solve_with(LinfResidual{});

  ASSERT_TRUE(result_l2.converged);
  ASSERT_TRUE(result_rel.converged);
  ASSERT_TRUE(result_linf.converged);

  // The recurrence's own alpha/beta are driven by the exact dot(r,r), never by whichever residual policy is
  // plugged in, so the converged iterate must be identical (to solver tolerance) regardless of policy choice.
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(x_l2[i], x_rel[i], 1e-8);
    EXPECT_NEAR(x_l2[i], x_linf[i], 1e-8);
  }
}

TEST(LinearSystem, CGInvOpMatchesDenseInverse) {
  const Matrix3d A = spd_matrix();
  const Vector3d rhs{0.3, -0.1, 0.7};
  const Vector3d expected = inverse(A) * rhs;

  // Unpreconditioned
  auto cg_inv = CGInvOp(mm_backend_t{}, Matrix3d(A), CGConfig<double>{});
  Vector3d out{0.0, 0.0, 0.0};
  cg_inv.apply(rhs, out);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-6);
  }
  EXPECT_TRUE(cg_inv.last_result().converged);

  // Jacobi-preconditioned
  auto pcg_inv = make_cg_inv_op<mm_backend_t>(Matrix3d(A), CGConfig<double>{},
                                              make_jacobi_preconditioner<mm_backend_t>(Vector3d{2.0, 2.0, 2.0}));
  Vector3d pcg_out{0.0, 0.0, 0.0};
  pcg_inv.apply(rhs, pcg_out);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(pcg_out[i], expected[i], 1e-6);
  }
  EXPECT_TRUE(pcg_inv.last_result().converged);
}

// With P = I, preconditioned CG is plain CG: z = r ./ 1 is r itself, so the solves agree bit for bit.
TEST(LinearSystem, UnitJacobiIsPlainCG) {
  const Matrix7d A = spd_matrix7();
  const Vector7d b = A * Vector7d{1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25};
  const Vector7d ones{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const CGConfig<double> cfg{.max_iters = 100, .tol = 1e-12};

  // Solve
  auto plain_state = CGState(Vector7d(), Vector7d(), Vector7d(), Vector7d());
  const auto plain = solve_linear_system(LinearSystem(mm_backend_t{}, Matrix7d(A), Vector7d(b)),
                                         CGStrategy(L2Residual{}, cfg), plain_state);
  auto unit_state = CGState(Vector7d(), Vector7d(), Vector7d(), Vector7d());
  const auto unit = solve_linear_system(
      LinearSystem(mm_backend_t{}, Matrix7d(A), Vector7d(b)),
      CGStrategy(L2Residual{}, cfg, make_jacobi_preconditioner<mm_backend_t>(Vector7d(ones))), unit_state);

  // Bit for bit
  ASSERT_TRUE(plain.converged);
  EXPECT_EQ(unit.converged, plain.converged);
  EXPECT_EQ(unit.num_iters, plain.num_iters);
  EXPECT_EQ(unit.residual, plain.residual);
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(unit_state.x()[i], plain_state.x()[i]) << "entry " << i;
  }
}

// Jacobi-preconditioned CG is invariant under diagonal scaling: its iterates on D A D y = D b are D^-1 those on
// A x = b. Powers of two scale exactly, so the two runs agree bit for bit, and n = 7 iterations reach the solution.
TEST(LinearSystem, JacobiIsDiagonalScalingInvariant) {
  const Matrix7d A = spd_matrix7();
  const Vector7d x_exact{1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25};
  const Vector7d b = A * x_exact;
  const int exponents[7] = {-6, 3, 0, 5, -2, 7, -4};
  Matrix7d DAD = Matrix7d();
  Vector7d Db = Vector7d();
  for (size_t i = 0; i < 7; ++i) {
    Db[i] = std::ldexp(b[i], exponents[i]);
    for (size_t j = 0; j < 7; ++j) {
      DAD(i, j) = std::ldexp(A(i, j), exponents[i] + exponents[j]);
    }
  }
  const CGConfig<double> cfg{.max_iters = 7, .tol = 0.0};

  // Solve
  auto state = CGState(Vector7d(), Vector7d(), Vector7d(), Vector7d());
  solve_linear_system(LinearSystem(mm_backend_t{}, Matrix7d(A), Vector7d(b)),
                      CGStrategy(L2Residual{}, cfg, make_jacobi_preconditioner<mm_backend_t>(diagonal7(A))), state);
  auto scaled_state = CGState(Vector7d(), Vector7d(), Vector7d(), Vector7d());
  solve_linear_system(LinearSystem(mm_backend_t{}, Matrix7d(DAD), Vector7d(Db)),
                      CGStrategy(L2Residual{}, cfg, make_jacobi_preconditioner<mm_backend_t>(diagonal7(DAD))),
                      scaled_state);

  // D^-1 x, bit for bit, at the solution
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(std::ldexp(scaled_state.x()[i], exponents[i]), state.x()[i]) << "entry " << i;
    EXPECT_NEAR(state.x()[i], x_exact[i], 1e-12) << "entry " << i;
  }
}

// CG starts from alpha g for a guess g, alpha = g^T b / g^T A g: a guess along the solution starts on it, and a zero
// right-hand side starts at zero. Every quantity here is exact.
TEST(LinearSystem, WarmStartRescalesGuess) {
  const Matrix3d A = spd_matrix();
  const Vector3d x_exact{1.0, 0.0, 1.0};
  const CGConfig<double> cfg;

  // A guess along the solution: g = 2 x*, alpha = 1/2
  auto along = CGState(Vector3d(2.0 * x_exact), Vector3d{}, Vector3d{}, Vector3d{});
  const auto along_result = solve_linear_system(LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(A * x_exact)),
                                                CGStrategy(L2Residual{}, cfg), along);
  EXPECT_TRUE(along_result.converged);
  EXPECT_EQ(along_result.num_iters, 0u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(along.x()[i], x_exact[i]);
  }

  // A zero right-hand side: alpha = 0
  auto zero_rhs = CGState(Vector3d{0.3, -0.7, 1.1}, Vector3d{}, Vector3d{}, Vector3d{});
  const auto zero_rhs_result = solve_linear_system(LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d{0.0, 0.0, 0.0}),
                                                   CGStrategy(L2Residual{}, cfg), zero_rhs);
  EXPECT_TRUE(zero_rhs_result.converged);
  EXPECT_EQ(zero_rhs_result.num_iters, 0u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(zero_rhs.x()[i], 0.0);
  }
}

// A = diag(1, 1, 0) with b = (1, 1, 1) off its range. The first step is exact, alpha = 3/2 and x = 3/2 b; the
// second direction (0, 0, 3/2) lies in A's null space, so p^T A p = 0 and CG stops there, unconverged.
TEST(LinearSystem, NotPositiveDefiniteStopsEarly) {
  const Matrix3d A{1.0, 0.0, 0.0,  //
                   0.0, 1.0, 0.0,  //
                   0.0, 0.0, 0.0};
  const Vector3d b{1.0, 1.0, 1.0};
  const CGConfig<double> cfg;

  // Strategy
  auto prob = LinearSystem(mm_backend_t{}, Matrix3d(A), Vector3d(b));
  auto state = CGState(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{});
  const auto result = solve_linear_system(prob, CGStrategy(L2Residual{}, cfg), state);
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.num_iters, 1u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(state.x()[i], 1.5);
  }

  // Inverse operator
  auto cg_inv = CGInvOp(mm_backend_t{}, Matrix3d(A), cfg);
  Vector3d out{0.0, 0.0, 0.0};
  EXPECT_THROW(cg_inv.apply(b, out), std::runtime_error);
}

// CGInvOps constructed and applied inside a kernel. A solve from the solution takes no iteration.
TEST(LinearSystem, CGInvOpInKernel) {
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> x("x", 9);
  Kokkos::View<unsigned*, Kokkos::DefaultExecutionSpace::memory_space> iters("iters", 3);
  apply_cg_inv_in_kernel(x, iters);

  const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  const auto iters_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, iters);
  const Vector3d expected = inverse(spd_matrix()) * Vector3d{0.3, -0.1, 0.7};
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(x_host(3 * k + i), expected[i], 1e-6) << "solution " << k << ", entry " << i;
    }
  }
  EXPECT_GT(iters_host(0), 0u);
  EXPECT_EQ(iters_host(1), 0u);
}

// A LinearSystem constructed and solved inside a kernel.
TEST(LinearSystem, MundyMathBackendInKernel) {
  Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> x("x", 3);
  Kokkos::View<bool, Kokkos::DefaultExecutionSpace::memory_space> converged("converged");
  solve_spd_in_kernel(x, converged);

  const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  EXPECT_TRUE(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, converged)());
  EXPECT_NEAR(x_host(0), 1.0, 1e-8);
  EXPECT_NEAR(x_host(1), 0.0, 1e-8);
  EXPECT_NEAR(x_host(2), 1.0, 1e-8);
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
//! \name KokkosBackend coverage
//@{

using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using view_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;

/// \brief How many Kokkos allocations f makes.
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

// A hand-rolled 3x3 SPD tridiagonal operator over Kokkos::View, avoiding any dependence on
// KokkosBlas/KokkosLapack (which may not have a usable LAPACK backend in a given build environment) -- this
// proves the CG solver itself works against a duck-typed, View-backed operator under KokkosBackend, independent
// of whatever BLAS/LAPACK support happens to be configured. Its vectors start NaN-filled, as uninitialized storage may.
struct TridiagKokkosOp {
  size_t domain_size() const {
    return 3;
  }
  size_t range_size() const {
    return 3;
  }
  view_t make_domain_vector() const {
    view_t v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "tridiag_domain"), 3);
    Kokkos::deep_copy(v, std::numeric_limits<double>::quiet_NaN());
    return v;
  }
  view_t make_range_vector() const {
    view_t v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "tridiag_range"), 3);
    Kokkos::deep_copy(v, std::numeric_limits<double>::quiet_NaN());
    return v;
  }
  void apply(const view_t& x, view_t& y) const {
    Kokkos::parallel_for(
        "TridiagKokkosOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int) {
          y(0) = 2.0 * x(0) - x(1);
          y(1) = -x(0) + 2.0 * x(1) - x(2);
          y(2) = -x(1) + 2.0 * x(2);
        });
  }
};

// The tridiagonal [2, -1, 0; -1, 2, -1; 0, -1, 2] as an operator that applies itself and, with KokkosKernels, as a
// dense and a sparse matrix.
TEST(LinearSystem, KokkosBackendConvergesToKnownSolution) {
  view_t b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"), 3);
  auto b_host = Kokkos::create_mirror_view(b);
  // b = A * (1, 0, 1)
  b_host(0) = 2.0 * 1.0 - 0.0;
  b_host(1) = -1.0 + 0.0 - 1.0;
  b_host(2) = -0.0 + 2.0 * 1.0;
  Kokkos::deep_copy(b, b_host);

  const auto check = [&](const auto& A, const char* kind) {
    using op_t = std::remove_cvref_t<decltype(A)>;
    view_t x0(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x0"), 3);
    Kokkos::deep_copy(x0, 0.0);

    auto prob = LinearSystem(kokkos_backend_t{}, op_t(A), view_t(b));
    auto state = CGState(view_t(x0), kokkos_backend_t::make_range_vector(A), kokkos_backend_t::make_range_vector(A),
                         kokkos_backend_t::make_range_vector(A));
    auto strat = CGStrategy(L2Residual{}, CGConfig<double>{});

    const auto result = solve_linear_system(prob, strat, state);
    EXPECT_TRUE(result.converged) << kind;
    EXPECT_LE(result.num_iters, 3u) << kind;

    auto x_host = Kokkos::create_mirror_view(state.x());
    Kokkos::deep_copy(x_host, state.x());
    EXPECT_NEAR(x_host(0), 1.0, 1e-8) << kind;
    EXPECT_NEAR(x_host(1), 0.0, 1e-8) << kind;
    EXPECT_NEAR(x_host(2), 1.0, 1e-8) << kind;

    // Jacobi-preconditioned, with A's diagonal (2, 2, 2)
    view_t d("d", 3);
    Kokkos::deep_copy(d, 2.0);
    auto pcg_state = CGState(view_t("x", 3), kokkos_backend_t::make_range_vector(A),
                             kokkos_backend_t::make_range_vector(A), kokkos_backend_t::make_range_vector(A));
    const auto pcg_result = solve_linear_system(
        LinearSystem(kokkos_backend_t{}, op_t(A), view_t(b)),
        CGStrategy(L2Residual{}, CGConfig<double>{}, make_jacobi_preconditioner<kokkos_backend_t>(d)), pcg_state);
    EXPECT_TRUE(pcg_result.converged) << kind;
    EXPECT_LE(pcg_result.num_iters, 3u) << kind;

    const auto pcg_x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pcg_state.x());
    EXPECT_NEAR(pcg_x_host(0), 1.0, 1e-8) << kind;
    EXPECT_NEAR(pcg_x_host(1), 0.0, 1e-8) << kind;
    EXPECT_NEAR(pcg_x_host(2), 1.0, 1e-8) << kind;
  };
  check(TridiagKokkosOp{}, "applies itself");

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  using dense_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
  using sparse_matrix_t = KokkosSparse::CrsMatrix<
      double, int, Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>, void,
      size_t>;
  const dense_matrix_t dense("dense", 3, 3);
  const auto dense_host = Kokkos::create_mirror_view(dense);
  for (size_t i = 0; i < 3; ++i) {
    dense_host(i, i) = 2.0;
    if (i > 0) dense_host(i, i - 1) = -1.0;
    if (i + 1 < 3) dense_host(i, i + 1) = -1.0;
  }
  Kokkos::deep_copy(dense, dense_host);
  check(dense, "dense");
  check(make_sparse_matrix<sparse_matrix_t>(dense), "sparse");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
}

// An inverse holds no solve history: each solve iterates in its out from the guess it holds. Into a zeroed out it is a
// fresh inverse's solve, bit for bit, whatever it solved before; re-solving a right-hand side into the out holding its
// solution takes no iteration.
TEST(LinearSystem, CGInvOpStartsFromOut) {
  view_t rhs_a("rhs_a", 3), rhs_b("rhs_b", 3);
  auto rhs_a_host = Kokkos::create_mirror_view(rhs_a);
  auto rhs_b_host = Kokkos::create_mirror_view(rhs_b);
  rhs_a_host(0) = 0.3;
  rhs_a_host(1) = -0.1;
  rhs_a_host(2) = 0.7;
  rhs_b_host(0) = -0.5;
  rhs_b_host(1) = 0.9;
  rhs_b_host(2) = 0.2;
  Kokkos::deep_copy(rhs_a, rhs_a_host);
  Kokkos::deep_copy(rhs_b, rhs_b_host);
  const CGConfig<double> cfg;

  // Solve
  auto fresh = make_cg_inv_op<kokkos_backend_t>(TridiagKokkosOp{}, cfg);
  view_t fresh_out("fresh_out", 3);
  fresh.apply(rhs_a, fresh_out);
  auto inv = make_cg_inv_op<kokkos_backend_t>(TridiagKokkosOp{}, cfg);
  view_t out_a("out_a", 3), out_b("out_b", 3);
  inv.apply(rhs_b, out_b);
  inv.apply(rhs_a, out_a);
  const unsigned zeroed_iters = inv.last_result().num_iters;
  const auto zeroed = Kokkos::create_mirror(Kokkos::HostSpace{}, out_a);  // a copy, as the next solve rewrites out_a
  Kokkos::deep_copy(zeroed, out_a);
  inv.apply(rhs_a, out_a);

  // Into a zeroed out
  ASSERT_GT(fresh.last_result().num_iters, 0u);
  EXPECT_EQ(zeroed_iters, fresh.last_result().num_iters);
  const auto fresh_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fresh_out);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(zeroed(i), fresh_host(i)) << "entry " << i;
  }

  // Into the out holding the solution
  EXPECT_EQ(inv.last_result().num_iters, 0u);
  EXPECT_TRUE(inv.last_result().converged);
}

// An inverse's scratch lives in its workspace: constructing the inverse allocates nothing, and solves through one
// workspace allocate nothing. A solve through a held workspace is the solve through a temporary one, bit for bit.
TEST(LinearSystem, CGInvOpScratchLivesInItsWorkspace) {
  view_t rhs("rhs", 3);
  auto rhs_host = Kokkos::create_mirror_view(rhs);
  rhs_host(0) = 0.3;
  rhs_host(1) = -0.1;
  rhs_host(2) = 0.7;
  Kokkos::deep_copy(rhs, rhs_host);
  const CGConfig<double> cfg;

  // Solve
  const size_t construct_allocations =
      count_allocations([&] { (void)make_cg_inv_op<kokkos_backend_t>(TridiagKokkosOp{}, cfg); });
  const auto inv = make_cg_inv_op<kokkos_backend_t>(TridiagKokkosOp{}, cfg);
  view_t temp_out("temp_out", 3), held_out("held_out", 3);
  inv.apply(rhs, temp_out);
  auto workspace = inv.make_workspace();
  const size_t solve_allocations = count_allocations([&] {
    inv.apply(rhs, held_out, workspace);
    Kokkos::deep_copy(held_out, 0.0);
    inv.apply(rhs, held_out, workspace);
  });

  // Allocations
  EXPECT_EQ(construct_allocations, 0u);
  EXPECT_EQ(solve_allocations, 0u);

  // Bit for bit
  const auto temp_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, temp_out);
  const auto held_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, held_out);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(held_host(i), temp_host(i)) << "entry " << i;
  }
}

// Every residual is a norm, so that of a zero-length vector is exactly zero.
TEST(LinearSystem, EmptyResidualsAreZero) {
  const view_t r("r", 0), b("b", 0);
  EXPECT_EQ(L2Residual{}(kokkos_backend_t{}, r, b), 0.0);
  EXPECT_EQ(RelativeL2Residual{}(kokkos_backend_t{}, r, b), 0.0);
  EXPECT_EQ(LinfResidual{}(kokkos_backend_t{}, r, b), 0.0);
}
//@}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

}  // namespace

}  // namespace mundy
