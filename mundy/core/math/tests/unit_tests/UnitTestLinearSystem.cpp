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

// Mundy
#include <mundy_math/Matrix.hpp>
#include <mundy_math/Vector.hpp>
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/linear_system.hpp>

namespace mundy {

namespace {

// A small, fixed, well-conditioned SPD system solved via mundy::MundyMathBackend: A is the classic
// tridiagonal [2,-1,0;-1,2,-1;0,-1,2], b chosen so the exact solution is (1, 0, 1).
Matrix3d spd_matrix() {
  return Matrix3d{2.0, -1.0, 0.0,   //
                  -1.0, 2.0, -1.0,  //
                  0.0, -1.0, 2.0};
}

Vector3d spd_rhs() {
  return spd_matrix() * Vector3d{1.0, 0.0, 1.0};
}

using mm_backend_t = MundyMathBackend;

TEST(LinearSystem, MundyMathBackendConvergesToKnownSolution) {
  const Matrix3d A = spd_matrix();
  const Vector3d b = spd_rhs();

  auto prob = LinearSystemProblem(mm_backend_t{}, Matrix3d(A), Vector3d(b));
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

  auto prob = LinearSystemProblem(mm_backend_t{}, Matrix3d(A), Vector3d(b));
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

  auto prob1 = LinearSystemProblem(mm_backend_t{}, Matrix3d(A), Vector3d(b1));
  auto state1 = CGState(Vector3d{0.0, 0.0, 0.0}, Vector3d{}, Vector3d{}, Vector3d{});
  auto strat = CGStrategy(L2Residual{}, CGConfig<double>{});
  const auto result1 = solve_linear_system(prob1, strat, state1);
  ASSERT_TRUE(result1.converged);

  // Warm start: reuse state1.x() (the converged solution to b1) as the initial guess for b2.
  auto prob_warm = LinearSystemProblem(mm_backend_t{}, Matrix3d(A), Vector3d(b2));
  auto state_warm = CGState(Vector3d(state1.x()), Vector3d{}, Vector3d{}, Vector3d{});
  const auto result_warm = solve_linear_system(prob_warm, strat, state_warm);
  ASSERT_TRUE(result_warm.converged);

  // Cold start: solve the same perturbed system from x0 = 0.
  auto prob_cold = LinearSystemProblem(mm_backend_t{}, Matrix3d(A), Vector3d(b2));
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
    auto prob = LinearSystemProblem(mm_backend_t{}, Matrix3d(A), Vector3d(b));
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

  auto cg_inv = CGInvOp(mm_backend_t{}, Matrix3d(A), CGConfig<double>{});
  Vector3d out{0.0, 0.0, 0.0};
  cg_inv.apply(rhs, out);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-6);
  }
  EXPECT_TRUE(cg_inv.last_result().converged);
}

TEST(LinearSystem, CGInvOpReusedAcrossMultipleRhsAlwaysColdStarts) {
  // CGInvOp's constructor default is warm_start = false: every apply() call must solve from x0 = 0, regardless
  // of what a previous apply() call left behind in the persistent x_ buffer.
  const Matrix3d A = spd_matrix();
  auto cg_inv = CGInvOp(mm_backend_t{}, Matrix3d(A), CGConfig<double>{});

  Vector3d out1{0.0, 0.0, 0.0};
  cg_inv.apply(Vector3d{1.0, 0.0, 0.0}, out1);
  Vector3d out2{0.0, 0.0, 0.0};
  cg_inv.apply(Vector3d{0.0, 0.0, 1.0}, out2);

  const Vector3d expected2 = inverse(A) * Vector3d{0.0, 0.0, 1.0};
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(out2[i], expected2[i], 1e-6);
  }
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
//! \name KokkosBackend coverage
//@{

using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using view_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;

// A hand-rolled 3x3 SPD tridiagonal operator over Kokkos::View, avoiding any dependence on
// KokkosBlas/KokkosLapack (which may not have a usable LAPACK backend in a given build environment) -- this
// proves the CG solver itself works against a duck-typed, View-backed operator under KokkosBackend, independent
// of whatever BLAS/LAPACK support happens to be configured.
struct TridiagKokkosOp {
  size_t domain_size() const {
    return 3;
  }
  size_t range_size() const {
    return 3;
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "tridiag_domain"), 3);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "tridiag_range"), 3);
  }
  void apply(const view_t& x, view_t& y) const {
    Kokkos::parallel_for(
        "TridiagKokkosOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
        KOKKOS_LAMBDA(const int) {
          y(0) = 2.0 * x(0) - x(1);
          y(1) = -x(0) + 2.0 * x(1) - x(2);
          y(2) = -x(1) + 2.0 * x(2);
        });
  }
};

TEST(LinearSystem, KokkosBackendConvergesToKnownSolution) {
  const TridiagKokkosOp A;
  view_t b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"), 3);
  auto b_host = Kokkos::create_mirror_view(b);
  // b = A * (1, 0, 1)
  b_host(0) = 2.0 * 1.0 - 0.0;
  b_host(1) = -1.0 + 0.0 - 1.0;
  b_host(2) = -0.0 + 2.0 * 1.0;
  Kokkos::deep_copy(b, b_host);

  view_t x0(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x0"), 3);
  Kokkos::deep_copy(x0, 0.0);

  auto prob = LinearSystemProblem(kokkos_backend_t{}, TridiagKokkosOp(A), view_t(b));
  auto state = CGState(view_t(x0), A.make_range_vector(), A.make_range_vector(), A.make_range_vector());
  auto strat = CGStrategy(L2Residual{}, CGConfig<double>{});

  const auto result = solve_linear_system(prob, strat, state);
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, 3u);

  auto x_host = Kokkos::create_mirror_view(state.x());
  Kokkos::deep_copy(x_host, state.x());
  EXPECT_NEAR(x_host(0), 1.0, 1e-8);
  EXPECT_NEAR(x_host(1), 0.0, 1e-8);
  EXPECT_NEAR(x_host(2), 1.0, 1e-8);
}
//@}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

}  // namespace

}  // namespace mundy
