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
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{BELOS,TPETRA}
#include <mundy_math/linear_system.hpp>
#include <mundy_math/solver_backends.hpp>

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)

#include <Tpetra_Map.hpp>
#include <cmath>
#include <initializer_list>
#include <mundy_math/belos_solver.hpp>
#include <stdexcept>

namespace mundy {

namespace {

// Tie the Kokkos spaces to Tpetra's default Node so belos_solve's exec-space-match static_assert holds by
// construction (Mundy and Tpetra must share one Kokkos execution space).
using exec_space = Tpetra::Map<>::node_type::execution_space;
using mem_space = Tpetra::Map<>::node_type::memory_space;
using kokkos_backend_t = KokkosBackend<exec_space>;
using view_t = Kokkos::View<double*, mem_space>;
using host_view_t = Kokkos::View<double*, Kokkos::HostSpace>;

// =============================================================================
// Operator / preconditioner fixtures
// =============================================================================

// A non-symmetric, strictly diagonally dominant tridiagonal operator: diagonal 4, subdiagonal -2, superdiagonal
// -1. The asymmetric off-diagonals (-2 vs -1) make CG invalid, so it exercises a genuinely non-symmetric solve;
// diagonal dominance guarantees nonsingularity and fast convergence.
struct NonsymTridiagOp {
  int n{5};
  size_t domain_size() const {
    return static_cast<size_t>(n);
  }
  size_t range_size() const {
    return static_cast<size_t>(n);
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nonsym_dom"), n);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nonsym_rng"), n);
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    const int nn = n;
    Kokkos::parallel_for(
        "NonsymTridiagOp::apply", Kokkos::RangePolicy<exec_space>(0, nn), KOKKOS_LAMBDA(const int i) {
          double v = 4.0 * x(i);
          if (i > 0) v += -2.0 * x(i - 1);
          if (i < nn - 1) v += -1.0 * x(i + 1);
          y(i) = v;
        });
  }
};

// A symmetric positive definite tridiagonal (diagonal 4, both off-diagonals -1) for CG / MINRES.
struct SymTridiagOp {
  int n{12};
  size_t domain_size() const {
    return static_cast<size_t>(n);
  }
  size_t range_size() const {
    return static_cast<size_t>(n);
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sym_dom"), n);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "sym_rng"), n);
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    const int nn = n;
    Kokkos::parallel_for(
        "SymTridiagOp::apply", Kokkos::RangePolicy<exec_space>(0, nn), KOKKOS_LAMBDA(const int i) {
          double v = 4.0 * x(i);
          if (i > 0) v += -1.0 * x(i - 1);
          if (i < nn - 1) v += -1.0 * x(i + 1);
          y(i) = v;
        });
  }
};

// A non-symmetric tridiagonal with a caller-supplied diagonal (off-diagonals -0.4 / -0.3). Paired with a widely
// varying diagonal so a Jacobi preconditioner measurably changes the iteration count.
struct VaryingDiagTridiagOp {
  view_t diag;  // A_ii
  int n;
  size_t domain_size() const {
    return static_cast<size_t>(n);
  }
  size_t range_size() const {
    return static_cast<size_t>(n);
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vd_dom"), n);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vd_rng"), n);
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    const int nn = n;
    const view_t d = diag;
    Kokkos::parallel_for(
        "VaryingDiagTridiagOp::apply", Kokkos::RangePolicy<exec_space>(0, nn), KOKKOS_LAMBDA(const int i) {
          double v = d(i) * x(i);
          if (i > 0) v += -0.4 * x(i - 1);
          if (i < nn - 1) v += -0.3 * x(i + 1);
          y(i) = v;
        });
  }
};

// Jacobi (diagonal) preconditioner: apply() computes the approximate-inverse action y_i = x_i / A_ii, as a right
// preconditioner operator must.
struct JacobiPrecond {
  view_t inv_diag;  // 1 / A_ii
  int n;
  size_t domain_size() const {
    return static_cast<size_t>(n);
  }
  size_t range_size() const {
    return static_cast<size_t>(n);
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "jac_dom"), n);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "jac_rng"), n);
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    const int nn = n;
    const view_t m = inv_diag;
    Kokkos::parallel_for(
        "JacobiPrecond::apply", Kokkos::RangePolicy<exec_space>(0, nn),
        KOKKOS_LAMBDA(const int i) { y(i) = m(i) * x(i); });
  }
};

// A workspace for WorkspacedScaledDiagOp: a scratch vector plus the standard commit/invalidate flag.
struct ScaledDiagWorkspace {
  view_t scratch;
  bool committed{false};
  void commit() {
    committed = true;
  }
  void invalidate() {
    committed = false;
  }
  bool is_committed() const {
    return committed;
  }
};

// A workspace-only operator (apply(x, y, workspace) with no plain apply(x, y)): A = scale * diag(d), staged through
// the workspace's scratch vector. This is the case the workspace-threaded scaled apply exists for -- routed through
// the plain unworkspaced apply it would hit the "op must provide apply(x,y)" fallback.
struct WorkspacedScaledDiagOp {
  view_t diag;
  double scale;
  int n;
  size_t domain_size() const {
    return static_cast<size_t>(n);
  }
  size_t range_size() const {
    return static_cast<size_t>(n);
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "wsd_dom"), n);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "wsd_rng"), n);
  }
  ScaledDiagWorkspace make_workspace() const {
    return ScaledDiagWorkspace{view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "wsd_scratch"), n), false};
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y, ScaledDiagWorkspace& ws) const {
    const int nn = n;
    const view_t d = diag;
    const view_t s = ws.scratch;
    const double c = scale;
    Kokkos::parallel_for(
        "wsd::stage", Kokkos::RangePolicy<exec_space>(0, nn), KOKKOS_LAMBDA(const int i) { s(i) = d(i) * x(i); });
    Kokkos::parallel_for(
        "wsd::scale", Kokkos::RangePolicy<exec_space>(0, nn), KOKKOS_LAMBDA(const int i) { y(i) = c * s(i); });
  }
};

// A widely-varying diagonal (1 .. 1e3, i.e. three orders of magnitude) and its reciprocal, so a Jacobi
// preconditioner has a clear effect. The spread sets the condition number (~1e3), which bounds solution accuracy
// at ~cond * residual_tol; the solves below use residual tol 1e-10, good to ~1e-7, inside the 1e-6 checks.
struct VaryingSystem {
  view_t diag;
  view_t inv_diag;
  int n;
};
VaryingSystem make_varying_system(int n) {
  view_t diag("diag", n);
  view_t inv_diag("inv_diag", n);
  auto diag_h = Kokkos::create_mirror_view(diag);
  auto inv_h = Kokkos::create_mirror_view(inv_diag);
  for (int i = 0; i < n; ++i) {
    const double di = std::pow(10.0, 3.0 * i / (n - 1));  // 1 .. 1e3
    diag_h(i) = di;
    inv_h(i) = 1.0 / di;
  }
  Kokkos::deep_copy(diag, diag_h);
  Kokkos::deep_copy(inv_diag, inv_h);
  return {diag, inv_diag, n};
}

// =============================================================================
// Test helpers
// =============================================================================

// A host reference solution x_i = 1 + step*i.
host_view_t ramp(int n, double step = 0.5) {
  host_view_t x("ramp", n);
  for (int i = 0; i < n; ++i) {
    x(i) = 1.0 + step * i;
  }
  return x;
}

// rhs = A * x_exact (x_exact given on host).
template <class Op>
view_t known_rhs(const Op& A, const host_view_t& x_exact_h) {
  const int n = static_cast<int>(A.domain_size());
  view_t x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "known_x"), n);
  Kokkos::deep_copy(x, x_exact_h);
  view_t b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "known_b"), n);
  A.apply(x, b);
  return b;
}

// Solve A x = rhs from a cold start via belos_solve; return the result, leaving the solution in x_out.
template <class Op, class Precond = impl::NoPreconditioner>
BelosResult<double> solve_into(const Op& A, const view_t& rhs, view_t& x_out, BelosConfig<double> cfg,
                               const Precond& precond = {}) {
  Kokkos::deep_copy(x_out, 0.0);
  auto prob = LinearSystem(kokkos_backend_t{}, Op(A), view_t(rhs));
  return belos_solve(prob, x_out, cfg, precond);
}

// Check every entry of a device solution against the host reference.
void expect_matches(const view_t& x_out, const host_view_t& x_exact_h, double atol = 1e-6) {
  auto xh = Kokkos::create_mirror_view(x_out);
  Kokkos::deep_copy(xh, x_out);
  ASSERT_EQ(xh.extent(0), x_exact_h.extent(0));
  for (int i = 0; i < static_cast<int>(x_exact_h.extent(0)); ++i) {
    EXPECT_NEAR(xh(i), x_exact_h(i), atol) << "mismatch at entry " << i;
  }
}

// Apply `op` (wrapped as a Tpetra::Operator) as Y = alpha*(A X) + beta*Y on a size-n system with X_i = 1+i and
// Y_i = 10+i, and check each entry against an independent host reference: ax(i) must return (A X)_i for X_i = 1+i.
template <class Op, class AxFn>
void expect_scaled_apply(const Op& op, int n, double alpha, double beta, AxFn ax) {
  using LO = Tpetra::Map<>::local_ordinal_type;
  using GO = Tpetra::Map<>::global_ordinal_type;
  using NO = Tpetra::Map<>::node_type;
  using map_type = Tpetra::Map<LO, GO, NO>;
  using mv_type = Tpetra::MultiVector<double, LO, GO, NO>;

  const Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::SerialComm<int>());
  const Teuchos::RCP<const map_type> map = Teuchos::rcp(new map_type(static_cast<Tpetra::global_size_t>(n), 0, comm));
  mundy::impl::MundyTpetraOperator<kokkos_backend_t, Op, double, LO, GO, NO> tpetra_op(op, map);

  mv_type X(map, 1);
  mv_type Y(map, 1);
  {
    auto xd = X.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    auto yd = Y.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    Kokkos::parallel_for(
        "fill_xy", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(const int i) {
          xd(i, 0) = 1.0 + i;
          yd(i, 0) = 10.0 + i;
        });
  }

  tpetra_op.apply(X, Y, Teuchos::NO_TRANS, alpha, beta);

  auto y_d = Y.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto y_h = Kokkos::create_mirror_view(y_d);
  Kokkos::deep_copy(y_h, y_d);
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(y_h(i, 0), alpha * ax(i) + beta * (10.0 + i), 1e-10) << "entry " << i;
  }
}

// =============================================================================
// Tests
// =============================================================================

// BelosInvOp must be usable as a matrix-free operator in operator algebra (e.g. a Schur complement).
static_assert(LinearOperator<kokkos_backend_t, BelosInvOp<kokkos_backend_t, NonsymTridiagOp>, view_t, view_t>,
              "BelosInvOp must satisfy the LinearOperator concept.");

// ---- Solver-menu coverage: every BelosSolver enumerator is exercised through a real solve. ----

TEST(BelosSolver, NonsymmetricSolversConverge) {
  const NonsymTridiagOp A{12};
  const host_view_t xe = ramp(A.n);
  const view_t rhs = known_rhs(A, xe);

  for (const BelosSolver s : {BelosSolver::PSEUDOBLOCK_GMRES, BelosSolver::BICGSTAB, BelosSolver::TFQMR}) {
    BelosConfig<double> cfg;
    cfg.solver = s;
    cfg.tol = 1e-10;
    cfg.max_iters = 200;
    cfg.num_blocks = A.n;
    view_t x("x", A.n);
    const auto res = solve_into(A, rhs, x, cfg);
    EXPECT_TRUE(res.converged) << impl::solver_name_string(s) << " did not converge: " << res;
    expect_matches(x, xe);
  }
}

TEST(BelosSolver, SymmetricSolversConvergeOnSpdSystem) {
  const SymTridiagOp A{12};
  const host_view_t xe = ramp(A.n);
  const view_t rhs = known_rhs(A, xe);

  for (const BelosSolver s : {BelosSolver::PSEUDOBLOCK_CG, BelosSolver::MINRES}) {
    BelosConfig<double> cfg;
    cfg.solver = s;
    cfg.tol = 1e-10;
    cfg.max_iters = 200;
    view_t x("x", A.n);
    const auto res = solve_into(A, rhs, x, cfg);
    EXPECT_TRUE(res.converged) << impl::solver_name_string(s) << " did not converge: " << res;
    expect_matches(x, xe);
  }
}

TEST(BelosSolver, GcrodrConvergesWithRecycleParams) {
  const NonsymTridiagOp A{12};
  const host_view_t xe = ramp(A.n);
  const view_t rhs = known_rhs(A, xe);

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::GCRODR;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;
  cfg.num_blocks = A.n;
  cfg.extra = Teuchos::rcp(new Teuchos::ParameterList());
  cfg.extra->set("Num Recycled Blocks", 2);  // GCRO-DR needs a recycle-subspace size

  view_t x("x", A.n);
  const auto res = solve_into(A, rhs, x, cfg);
  EXPECT_TRUE(res.converged) << "GCRODR did not converge: " << res;
  expect_matches(x, xe);
}

// ---- BelosConfig::extra escape hatch ----

TEST(BelosSolver, ExtraParameterListOverridesTypedFields) {
  // extra is merged after the typed fields, so it wins: capping iterations to 1 via extra on a system that needs
  // more must produce a non-converged result. This fails if extra were dropped or merged before the typed fields.
  const NonsymTridiagOp A{12};
  const view_t rhs = known_rhs(A, ramp(A.n));

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;
  cfg.num_blocks = A.n;
  cfg.extra = Teuchos::rcp(new Teuchos::ParameterList());
  cfg.extra->set("Maximum Iterations", 1);

  view_t x("x", A.n);
  const auto res = solve_into(A, rhs, x, cfg);
  EXPECT_FALSE(res.converged) << res;
  EXPECT_LE(res.num_iters, 1u);
}

// ---- Failure contracts ----

TEST(BelosSolver, NonConvergenceReturnsFlagWithoutThrowing) {
  // belos_solve reports failure through the result, never by throwing.
  const NonsymTridiagOp A{12};
  const view_t rhs = known_rhs(A, ramp(A.n));

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1e-12;
  cfg.max_iters = 1;  // far too few for a 12-dimensional system
  cfg.num_blocks = A.n;

  view_t x("x", A.n);
  BelosResult<double> res;
  EXPECT_NO_THROW({ res = solve_into(A, rhs, x, cfg); });
  EXPECT_FALSE(res.converged) << res;
}

TEST(BelosSolver, BelosInvOpThrowsOnNonConvergence) {
  // BelosInvOp cannot hand back a wrong inverse, so a non-converged inner solve throws.
  const NonsymTridiagOp A{12};
  const view_t rhs = known_rhs(A, ramp(A.n));

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1e-12;
  cfg.max_iters = 1;
  cfg.num_blocks = A.n;
  auto inv = make_belos_inv_op<kokkos_backend_t>(NonsymTridiagOp(A), cfg);

  view_t out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out"), A.n);
  EXPECT_THROW(inv.apply(rhs, out), std::runtime_error);
}

// ---- BelosInvOp (inverse-as-operator) ----

TEST(BelosSolver, BelosInvOpRecoversKnownSolution) {
  const NonsymTridiagOp A{5};
  const host_view_t xe = ramp(A.n);
  const view_t rhs = known_rhs(A, xe);

  BelosConfig<double> cfg;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;
  auto inv = make_belos_inv_op<kokkos_backend_t>(NonsymTridiagOp(A), cfg);

  view_t out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out"), A.n);
  inv.apply(rhs, out);
  EXPECT_TRUE(inv.last_result().converged) << "inner solve did not converge: " << inv.last_result();
  expect_matches(out, xe);
}

TEST(BelosSolver, BelosInvOpReusedAcrossRhsColdStarts) {
  // Default warm_start = false: each apply() cold-starts, so a second solve of a different system is correct
  // regardless of the internal buffer left by the first.
  const NonsymTridiagOp A{12};
  BelosConfig<double> cfg;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;
  auto inv = make_belos_inv_op<kokkos_backend_t>(NonsymTridiagOp(A), cfg);

  const host_view_t x1 = ramp(A.n, 0.5);
  const view_t rhs1 = known_rhs(A, x1);
  view_t out1(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out1"), A.n);
  inv.apply(rhs1, out1);
  expect_matches(out1, x1);

  const host_view_t x2 = ramp(A.n, -0.3);  // a different solution
  const view_t rhs2 = known_rhs(A, x2);
  view_t out2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out2"), A.n);
  inv.apply(rhs2, out2);
  EXPECT_TRUE(inv.last_result().converged);
  expect_matches(out2, x2);
}

TEST(BelosSolver, BelosInvOpWarmStartConvergesFaster) {
  // warm_start = true feeds the previous solve's solution as the next initial guess. With the stopping tolerance
  // measured against the right-hand-side norm (the regime in which a better guess is worth fewer iterations),
  // priming the warm operator on one system and then solving a nearby one (same operator, slightly shifted rhs)
  // takes fewer iterations than an otherwise-identical cold operator solving that same nearby system from zero.
  // A clustered, well-conditioned spectrum makes convergence residual-driven, so the guess's head start shows up
  // as a clear iteration reduction (a spread spectrum would need ~one iteration per eigenvalue regardless).
  const NonsymTridiagOp A{40};
  const int n = A.n;

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1e-8;
  cfg.max_iters = 200;
  cfg.num_blocks = n;  // full GMRES (no restart within a cycle)
  cfg.extra = Teuchos::rcp(new Teuchos::ParameterList());
  cfg.extra->set("Implicit Residual Scaling", std::string("Norm of RHS"));
  cfg.extra->set("Explicit Residual Scaling", std::string("Norm of RHS"));

  auto warm = make_belos_inv_op<kokkos_backend_t>(NonsymTridiagOp(A), cfg, impl::NoPreconditioner{},
                                                  /*warm_start=*/true);
  auto cold = make_belos_inv_op<kokkos_backend_t>(NonsymTridiagOp(A), cfg, impl::NoPreconditioner{},
                                                  /*warm_start=*/false);

  const host_view_t x1 = ramp(n, 0.1);
  const view_t rhs1 = known_rhs(A, x1);
  view_t out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out"), n);
  warm.apply(rhs1, out);  // prime the warm operator's internal guess with solution 1

  host_view_t x2("x2", n);  // a nearby system: solution shifted by a small constant
  for (int i = 0; i < n; ++i) {
    x2(i) = x1(i) + 1e-4;
  }
  const view_t rhs2 = known_rhs(A, x2);

  warm.apply(rhs2, out);
  const unsigned warm_iters = warm.last_result().num_iters;
  expect_matches(out, x2, 1e-4);
  cold.apply(rhs2, out);
  const unsigned cold_iters = cold.last_result().num_iters;
  expect_matches(out, x2, 1e-4);

  EXPECT_LT(warm_iters, cold_iters) << "warm=" << warm_iters << " cold=" << cold_iters;
}

// ---- Preconditioning ----

TEST(BelosSolver, BelosInvOpWithPreconditionerRecoversSolution) {
  const int n = 30;
  const VaryingSystem sys = make_varying_system(n);
  const VaryingDiagTridiagOp A{sys.diag, n};

  const host_view_t xe = ramp(n, 0.1);
  const view_t rhs = known_rhs(A, xe);

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1e-10;
  cfg.max_iters = n;
  cfg.num_blocks = n;
  auto inv = make_belos_inv_op<kokkos_backend_t>(VaryingDiagTridiagOp(A), cfg, JacobiPrecond{sys.inv_diag, n});

  view_t out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out"), n);
  inv.apply(rhs, out);
  EXPECT_TRUE(inv.last_result().converged) << inv.last_result();
  expect_matches(out, xe);
}

TEST(BelosSolver, RightPreconditionerReducesIterations) {
  const int n = 30;  // large enough that the spread-diagonal system takes many unpreconditioned iterations
  const VaryingSystem sys = make_varying_system(n);
  const VaryingDiagTridiagOp A{sys.diag, n};
  const JacobiPrecond M{sys.inv_diag, n};

  const host_view_t xe = ramp(n, 0.1);
  const view_t rhs = known_rhs(A, xe);

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1e-10;
  cfg.max_iters = n;
  cfg.num_blocks = n;  // full GMRES (no restart) -> iteration counts are directly comparable

  view_t x_unprec("x_unprec", n);
  const auto res_unprec = solve_into(A, rhs, x_unprec, cfg);
  view_t x_prec("x_prec", n);
  const auto res_prec = solve_into(A, rhs, x_prec, cfg, M);

  ASSERT_TRUE(res_unprec.converged) << res_unprec;
  ASSERT_TRUE(res_prec.converged) << res_prec;
  // Jacobi equilibrates a diagonal spanning three orders of magnitude, so it must cut the iteration count. Were
  // the preconditioner silently ignored, these counts would be equal.
  EXPECT_LT(res_prec.num_iters, res_unprec.num_iters)
      << "prec=" << res_prec.num_iters << " unprec=" << res_unprec.num_iters;
  expect_matches(x_unprec, xe);
  expect_matches(x_prec, xe);
}

// ---- Tpetra::Operator adapter: general alpha/beta apply ----

TEST(BelosSolver, TpetraOperatorHonorsAlphaBeta) {
  constexpr int n = 5;
  // SymTridiagOp: (A x)_i = 4 x_i - x_{i-1} - x_{i+1}, with x_i = 1 + i.
  expect_scaled_apply(SymTridiagOp{n}, n, /*alpha=*/2.0, /*beta=*/3.0, [n](int i) {
    const double x_im1 = (i > 0) ? static_cast<double>(i) : 0.0;          // x_{i-1} = 1 + (i-1)
    const double x_ip1 = (i < n - 1) ? static_cast<double>(i + 2) : 0.0;  // x_{i+1} = 1 + (i+1)
    return 4.0 * (1.0 + i) - x_im1 - x_ip1;
  });
}

TEST(BelosSolver, TpetraOperatorHonorsAlphaBetaWithWorkspacedOp) {
  constexpr int n = 5;
  const double scale = 1.5;
  view_t d("d", n);
  {
    auto dh = Kokkos::create_mirror_view(d);
    for (int i = 0; i < n; ++i) {
      dh(i) = 2.0 + i;
    }
    Kokkos::deep_copy(d, dh);
  }
  // Workspace-only op A = scale * diag(d): (A x)_i = scale * d_i * x_i, with x_i = 1 + i, d_i = 2 + i.
  expect_scaled_apply(WorkspacedScaledDiagOp{d, scale, n}, n, /*alpha=*/2.0, /*beta=*/3.0,
                      [scale](int i) { return scale * (2.0 + i) * (1.0 + i); });
}

}  // namespace

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA
