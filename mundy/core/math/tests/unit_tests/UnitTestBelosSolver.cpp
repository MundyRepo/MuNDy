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

#include <cmath>
#include <initializer_list>

#include <Tpetra_Map.hpp>

#include <mundy_math/belos_solver.hpp>

namespace mundy {

namespace {

// Tie the Kokkos spaces to Tpetra's default Node so belos_solve's exec-space-match static_assert holds by
// construction (Mundy and Tpetra must share one Kokkos execution space).
using exec_space = Tpetra::Map<>::node_type::execution_space;
using mem_space = Tpetra::Map<>::node_type::memory_space;
using kokkos_backend_t = KokkosBackend<exec_space>;
using view_t = Kokkos::View<double*, mem_space>;
using host_view_t = Kokkos::View<double*, Kokkos::HostSpace>;

// A non-symmetric, strictly diagonally dominant tridiagonal operator over Kokkos::View: diagonal 4, subdiagonal
// -2, superdiagonal -1. The asymmetric off-diagonals (-2 vs -1) make CG invalid, so this exercises a genuinely
// non-symmetric Krylov solve. Diagonal dominance guarantees nonsingularity and convergence.
struct NonsymTridiagOp {
  int n{5};

  size_t domain_size() const {
    return static_cast<size_t>(n);
  }
  size_t range_size() const {
    return static_cast<size_t>(n);
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nonsym_domain"), n);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nonsym_range"), n);
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    const int nn = n;
    Kokkos::parallel_for(
        "NonsymTridiagOp::apply", Kokkos::RangePolicy<exec_space>(0, nn), KOKKOS_LAMBDA(const int i) {
          double val = 4.0 * x(i);
          if (i > 0) val += -2.0 * x(i - 1);
          if (i < nn - 1) val += -1.0 * x(i + 1);
          y(i) = val;
        });
  }
};

// Build b = A * x_exact for x_exact = (1, 2, ..., n); return (b, x_exact_host) for checking.
std::pair<view_t, Kokkos::View<double*, Kokkos::HostSpace>> make_known_system(const NonsymTridiagOp& A) {
  const int n = A.n;
  view_t x_exact(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_exact"), n);
  auto x_exact_h = Kokkos::create_mirror_view(x_exact);
  for (int i = 0; i < n; ++i) {
    x_exact_h(i) = static_cast<double>(i + 1);
  }
  Kokkos::deep_copy(x_exact, x_exact_h);

  view_t b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"), n);
  A.apply(x_exact, b);

  Kokkos::View<double*, Kokkos::HostSpace> x_exact_copy("x_exact_copy", n);
  Kokkos::deep_copy(x_exact_copy, x_exact_h);
  return {b, x_exact_copy};
}

void expect_solves_known_system(BelosSolver solver_choice) {
  const NonsymTridiagOp A;
  auto [b, x_exact_h] = make_known_system(A);

  view_t x0(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x0"), A.n);
  Kokkos::deep_copy(x0, 0.0);

  auto prob = LinearSystem(kokkos_backend_t{}, NonsymTridiagOp(A), view_t(b));
  BelosConfig<double> cfg;
  cfg.solver = solver_choice;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;

  const auto result = belos_solve(prob, x0, cfg);
  EXPECT_TRUE(result.converged) << "solver did not converge: " << result;

  auto x_host = Kokkos::create_mirror_view(x0);
  Kokkos::deep_copy(x_host, x0);
  for (int i = 0; i < A.n; ++i) {
    EXPECT_NEAR(x_host(i), x_exact_h(i), 1e-6) << "mismatch at entry " << i;
  }
}

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

// A non-symmetric tridiagonal with a caller-supplied diagonal (off-diagonals -0.4 / -0.3). Used with a widely
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

// Jacobi (diagonal) preconditioner: apply() computes the approximate-inverse action y_i = x_i / A_ii by storing
// 1/A_ii and multiplying -- what a right-preconditioner operator must compute.
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
  for (int i = 0; i < static_cast<int>(x_exact_h.extent(0)); ++i) {
    EXPECT_NEAR(xh(i), x_exact_h(i), atol) << "mismatch at entry " << i;
  }
}

// A linearly increasing host reference solution x_i = 1 + step*i.
host_view_t ramp(int n, double step = 0.5) {
  host_view_t x("ramp", n);
  for (int i = 0; i < n; ++i) {
    x(i) = 1.0 + step * i;
  }
  return x;
}

TEST(BelosSolver, GmresConvergesToKnownNonsymmetricSolution) {
  expect_solves_known_system(BelosSolver::PSEUDOBLOCK_GMRES);
}

TEST(BelosSolver, BiCGStabConvergesToKnownNonsymmetricSolution) {
  expect_solves_known_system(BelosSolver::BICGSTAB);
}

// BelosInvOp must be usable as a matrix-free operator in operator algebra (e.g. a Schur complement).
static_assert(LinearOperator<kokkos_backend_t, BelosInvOp<kokkos_backend_t, NonsymTridiagOp>, view_t, view_t>,
              "BelosInvOp must satisfy the LinearOperator concept.");

TEST(BelosSolver, BelosInvOpRecoversKnownSolution) {
  const NonsymTridiagOp A;
  auto [rhs, x_exact_h] = make_known_system(A);  // rhs = A * (1..n)

  BelosConfig<double> cfg;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;
  auto inv = make_belos_inv_op<kokkos_backend_t>(NonsymTridiagOp(A), cfg);

  view_t out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out"), A.n);
  Kokkos::deep_copy(out, 0.0);
  inv.apply(rhs, out);

  EXPECT_TRUE(inv.last_result().converged) << "inner solve did not converge: " << inv.last_result();
  auto out_h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(out_h, out);
  for (int i = 0; i < A.n; ++i) {
    EXPECT_NEAR(out_h(i), x_exact_h(i), 1e-6) << "mismatch at entry " << i;
  }
}

TEST(BelosSolver, BelosInvOpReusedAcrossRhsColdStarts) {
  // Default warm_start = false: a second apply() must be correct despite the x_ buffer left by the first.
  const NonsymTridiagOp A;
  BelosConfig<double> cfg;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;
  auto inv = make_belos_inv_op<kokkos_backend_t>(NonsymTridiagOp(A), cfg);

  // First solve populates the persistent x_ buffer.
  auto rhs1 = make_known_system(A).first;
  view_t out1(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out1"), A.n);
  Kokkos::deep_copy(out1, 0.0);
  inv.apply(rhs1, out1);

  // Second solve: a different solution (n..1); rhs2 = A * x2.
  view_t x2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x2"), A.n);
  auto x2_h = Kokkos::create_mirror_view(x2);
  for (int i = 0; i < A.n; ++i) {
    x2_h(i) = static_cast<double>(A.n - i);
  }
  Kokkos::deep_copy(x2, x2_h);
  view_t rhs2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rhs2"), A.n);
  A.apply(x2, rhs2);

  view_t out2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "out2"), A.n);
  Kokkos::deep_copy(out2, 0.0);
  inv.apply(rhs2, out2);

  EXPECT_TRUE(inv.last_result().converged);
  auto out2_h = Kokkos::create_mirror_view(out2);
  Kokkos::deep_copy(out2_h, out2);
  for (int i = 0; i < A.n; ++i) {
    EXPECT_NEAR(out2_h(i), x2_h(i), 1e-6) << "mismatch at entry " << i;
  }
}

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
    EXPECT_TRUE(res.converged) << "solver " << static_cast<int>(s) << " did not converge: " << res;
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
    EXPECT_TRUE(res.converged) << "solver " << static_cast<int>(s) << " did not converge: " << res;
    expect_matches(x, xe);
  }
}

TEST(BelosSolver, GcrodrConvergesWithRecycleParamsViaEscapeHatch) {
  const NonsymTridiagOp A{12};
  const host_view_t xe = ramp(A.n);
  const view_t rhs = known_rhs(A, xe);

  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::GCRODR;
  cfg.tol = 1e-10;
  cfg.max_iters = 200;
  cfg.num_blocks = A.n;
  cfg.extra = Teuchos::rcp(new Teuchos::ParameterList());
  cfg.extra->set("Num Recycled Blocks", 2);  // also exercises the ParameterList escape hatch

  view_t x("x", A.n);
  const auto res = solve_into(A, rhs, x, cfg);
  EXPECT_TRUE(res.converged) << "GCRODR did not converge: " << res;
  expect_matches(x, xe);
}

// ---- Preconditioning ----

// Build a widely-varying diagonal (1 .. 1e3) so a Jacobi preconditioner must reduce the iteration count. The
// spread sets the condition number (~1e3), which bounds solution accuracy at ~cond * residual_tol; the solves
// below use residual tol 1e-10 so the recovered solution is good to ~1e-7, well inside the 1e-6 checks.
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

TEST(BelosSolver, RightPreconditionerReducesIterations) {
  const int n = 30;
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
  // Jacobi equilibrates a diagonal spanning four orders of magnitude, so it must cut the iteration count. Were the
  // preconditioner silently ignored, these counts would be equal.
  EXPECT_LT(res_prec.num_iters, res_unprec.num_iters)
      << "prec=" << res_prec.num_iters << " unprec=" << res_unprec.num_iters;
  expect_matches(x_unprec, xe);
  expect_matches(x_prec, xe);
}

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

  view_t out("out", n);
  Kokkos::deep_copy(out, 0.0);
  inv.apply(rhs, out);
  EXPECT_TRUE(inv.last_result().converged) << inv.last_result();
  expect_matches(out, xe);
}

TEST(BelosSolver, TpetraOperatorHonorsAlphaBeta) {
  using LO = Tpetra::Map<>::local_ordinal_type;
  using GO = Tpetra::Map<>::global_ordinal_type;
  using NO = Tpetra::Map<>::node_type;
  using map_type = Tpetra::Map<LO, GO, NO>;
  using mv_type = Tpetra::MultiVector<double, LO, GO, NO>;

  constexpr int n = 5;
  const SymTridiagOp A{n};  // symmetric tridiagonal (diag 4, off-diag -1); templated apply
  const double alpha = 2.0;
  const double beta = 3.0;

  const Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::SerialComm<int>());
  const Teuchos::RCP<const map_type> map = Teuchos::rcp(new map_type(static_cast<Tpetra::global_size_t>(n), 0, comm));
  mundy::impl::MundyTpetraOperator<kokkos_backend_t, SymTridiagOp, double, LO, GO, NO> op(A, map);

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

  op.apply(X, Y, Teuchos::NO_TRANS, alpha, beta);

  // Reference y_new = alpha * (A x) + beta * y_old, computed independently on host.
  auto y_d = Y.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto y_h = Kokkos::create_mirror_view(y_d);
  Kokkos::deep_copy(y_h, y_d);
  for (int i = 0; i < n; ++i) {
    const double x_im1 = (i > 0) ? static_cast<double>(i) : 0.0;         // 1 + (i-1)
    const double x_ip1 = (i < n - 1) ? static_cast<double>(i + 2) : 0.0;  // 1 + (i+1)
    const double ax = 4.0 * (1.0 + i) - x_im1 - x_ip1;
    const double y_old = 10.0 + i;
    EXPECT_NEAR(y_h(i, 0), alpha * ax + beta * y_old, 1e-10) << "entry " << i;
  }
}

}  // namespace

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA
