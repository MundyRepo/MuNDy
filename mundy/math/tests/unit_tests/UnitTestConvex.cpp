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

// External libs
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

#include <Kokkos_Core.hpp>  // for Kokkos::Array

// C++ core libs
#include <ostream>  // for std::cout

// Mundy libs
#include <MundyMath_config.hpp>   // for HAVE_MUNDYMATH_*
#include <mundy_math/Matrix.hpp>  // for mundy::math::Matrix
#include <mundy_math/Vector.hpp>  // for mundy::math::Vector
#include <mundy_math/convex.hpp>  // for mundy::math::solve_lcp/solve_cqpp

namespace mundy {

namespace math {

namespace {

//! \name MundyMath backend test problems
//@{

namespace math_backend {

struct UnconstrainedSPD1Problem {
  using scalar_t = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = convex::MundyMathBackend;

  std::string name() const {
    return "UnconstrainedSPD1Problem";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return convex::space::Unconstrained<scalar_t>();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_exact_solution() const {
    return Vector3d{1.0, 0.0, 1.0};
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_A() const {
    return Matrix3d{2.0,  -1.0, 0.0,   //
                    -1.0, 2.0,  -1.0,  //
                    0.0,  -1.0, 2.0};
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    return -get_A() * get_exact_solution();
  }
};

struct InactiveBoxConstrainedSPDProblem {
  using scalar_t = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = convex::MundyMathBackend;

  std::string name() const {
    return "InactiveBoxConstrainedSPDProblem";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return convex::space::Bounded<scalar_t>(0.0, 2.0);
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_exact_solution() const {
    return Vector3d{1.0, 0.0, 1.0};
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_A() const {
    return Matrix3d{2.0,  -1.0, 0.0,   //
                    -1.0, 2.0,  -1.0,  //
                    0.0,  -1.0, 2.0};
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    return -get_A() * get_exact_solution();
  }
};

struct ActiveBoxConstrainedSPDProblem {
  using scalar_t = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = convex::MundyMathBackend;

  std::string name() const {
    return "ActiveBoxConstrainedSPDProblem";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return convex::space::Bounded<scalar_t>(9.0, 10.0);
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_exact_solution() const {
    return Vector3d{9.0, 9.0, 9.0};
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_A() const {
    return Matrix3d{2.0,  -1.0, 0.0,   //
                    -1.0, 2.0,  -1.0,  //
                    0.0,  -1.0, 2.0};
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    return -get_A() * get_exact_solution();
  }
};

template <size_t N>
struct RandomLCP {
  using scalar_t = double;
  using vector_t = Vector<scalar_t, N>;
  using linear_op_t = Matrix<scalar_t, N, N>;
  using backend_t = convex::MundyMathBackend;

  std::string name() const {
    return "RandomLCP" + std::to_string(N);
  }

  KOKKOS_INLINE_FUNCTION
  RandomLCP() {
    // 1. Build M
    A_ = gen_random_p_matrix();

    // 2. Choose disjoint supports for z* and w*
    for (size_t i = 0; i < N; ++i) {
      double u01 = static_cast<double>(rand()) / RAND_MAX;
      bool is_active = static_cast<double>(rand()) / RAND_MAX < 0.5;
      if (is_active) {
        x_star_[i] = u01 * 0.9 + 0.1;
        grad_star_[i] = 0.0;
      } else {
        x_star_[i] = 0.0;
        grad_star_[i] = u01 * 0.9 + 0.1;
      }
    }

    // 3. q that makes (z*, w*) solve the LCP
    q_ = grad_star_ - A_ * x_star_;
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return convex::space::LowerBound<scalar_t>(0.0);
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_exact_solution() const {
    return x_star_;
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_A() const {
    return A_;
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    return q_;
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t gen_random_matrix() {
    linear_op_t mat;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        mat(i, j) = 1.0 - 2 * static_cast<double>(rand()) / RAND_MAX;
      }
    }
    return mat;
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t gen_random_p_matrix() {
    // Strictly diagonally dominant with positive diagonal
    linear_op_t mat = gen_random_matrix();
    for (size_t i = 0; i < N; ++i) {
      scalar_t off_diag_abs_row_sum = 0;
      for (size_t j = 0; j < N; ++j) {
        off_diag_abs_row_sum += Kokkos::abs(mat(i, j)) * (i != j);
      }
      mat(i, i) = off_diag_abs_row_sum + 10.0;
    }

    return mat;
  }

 private:
  linear_op_t A_;
  vector_t q_;
  vector_t x_star_;
  vector_t grad_star_;
};

namespace congruent {

template <class NonCongruentProblem>
struct CongruentLCPProblemWrapper {
  // D = I
  using scalar_t = typename NonCongruentProblem::scalar_t;
  using vector_t = typename NonCongruentProblem::vector_t;
  using linear_op_t = typename NonCongruentProblem::linear_op_t;
  using backend_t = typename NonCongruentProblem::backend_t;

  std::string name() const {
    return "CongruentLCPProblemWrapper<" + cdp.name() + ">";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return cdp.get_space();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_x_exact() const {
    return cdp.get_exact_solution();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_f_exact() const {
    return cdp.get_exact_solution();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_u_exact() const {
    return get_M() * get_x_exact();
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_D() const {
    return linear_op_t::identity();
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_M() const {
    return cdp.get_A();
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_DT() const {
    return linear_op_t::identity();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    return cdp.get_q();
  }

  NonCongruentProblem cdp;
};

}  // namespace congruent

}  // namespace math_backend
//@}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
//! \name Kokkos backend test problems (single process)
//@{
namespace kokkos_backend {

struct UnconstrainedSPD1Problem {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using scalar_t = double;
  using layout_t = Kokkos::View<scalar_t*, mem_space>::array_layout;
  using vector_t = Kokkos::View<scalar_t*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<scalar_t**, layout_t, mem_space>;
  using backend_t = convex::KokkosBackend<exec_space>;

  std::string name() const {
    return "UnconstrainedSPD1Problem";
  }

  unsigned size() const {
    return 3;
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  auto get_space() const {
    return convex::space::Unconstrained<scalar_t>();
  }

  vector_t get_exact_solution() const {
    vector_t x_exact(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_exact"), 3);

    auto x_exact_host = Kokkos::create_mirror_view(x_exact);
    x_exact_host(0) = 1.0;
    x_exact_host(1) = 0.0;
    x_exact_host(2) = 1.0;
    Kokkos::deep_copy(x_exact, x_exact_host);

    return x_exact;
  }

  linear_op_t get_A() const {
    linear_op_t A(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), 3, 3);

    // clang-format off
    auto A_host = Kokkos::create_mirror_view(A);
    A_host(0, 0) = 2.0;  A_host(0, 1) = -1.0; A_host(0, 2) = 0.0;
    A_host(1, 0) = -1.0; A_host(1, 1) = 2.0;  A_host(1, 2) = -1.0;
    A_host(2, 0) = 0.0;  A_host(2, 1) = -1.0; A_host(2, 2) = 2.0;
    // clang-format on

    Kokkos::deep_copy(A, A_host);
    return A;
  }

  vector_t get_q() const {
    vector_t q(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q"), 3);
    backend_t::apply(-1.0, get_A(), get_exact_solution(), 0.0, q);
    return q;
  }
};

struct InactiveBoxConstrainedSPDProblem {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using scalar_t = double;
  using layout_t = Kokkos::View<scalar_t*, mem_space>::array_layout;
  using vector_t = Kokkos::View<scalar_t*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<scalar_t**, layout_t, mem_space>;
  using backend_t = convex::KokkosBackend<exec_space>;

  std::string name() const {
    return "InactiveBoxConstrainedSPDProblem";
  }

  unsigned size() const {
    return 3;
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  auto get_space() const {
    return convex::space::Bounded<scalar_t>(0.0, 2.0);
  }

  vector_t get_exact_solution() const {
    vector_t x_exact(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_exact"), 3);

    auto x_exact_host = Kokkos::create_mirror_view(x_exact);
    x_exact_host(0) = 1.0;
    x_exact_host(1) = 0.0;
    x_exact_host(2) = 1.0;
    Kokkos::deep_copy(x_exact, x_exact_host);

    return x_exact;
  }

  linear_op_t get_A() const {
    linear_op_t A(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), 3, 3);

    // clang-format off
    auto A_host = Kokkos::create_mirror_view(A);
    A_host(0, 0) = 2.0;  A_host(0, 1) = -1.0; A_host(0, 2) = 0.0;
    A_host(1, 0) = -1.0; A_host(1, 1) = 2.0;  A_host(1, 2) = -1.0;
    A_host(2, 0) = 0.0;  A_host(2, 1) = -1.0; A_host(2, 2) = 2.0;
    // clang-format on

    Kokkos::deep_copy(A, A_host);
    return A;
  }

  vector_t get_q() const {
    vector_t q(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q"), 3);
    backend_t::apply(-1.0, get_A(), get_exact_solution(), 0.0, q);
    return q;
  }
};

struct ActiveBoxConstrainedSPDProblem {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using scalar_t = double;
  using layout_t = Kokkos::View<scalar_t*, mem_space>::array_layout;
  using vector_t = Kokkos::View<scalar_t*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<scalar_t**, layout_t, mem_space>;
  using backend_t = convex::KokkosBackend<exec_space>;

  std::string name() const {
    return "ActiveBoxConstrainedSPDProblem";
  }

  unsigned size() const {
    return 3;
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  auto get_space() const {
    return convex::space::Bounded<scalar_t>(9.0, 10.0);
  }

  vector_t get_exact_solution() const {
    vector_t x_exact(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_exact"), 3);

    auto x_exact_host = Kokkos::create_mirror_view(x_exact);
    x_exact_host(0) = 9.0;
    x_exact_host(1) = 9.0;
    x_exact_host(2) = 9.0;
    Kokkos::deep_copy(x_exact, x_exact_host);

    return x_exact;
  }

  linear_op_t get_A() const {
    linear_op_t A(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), 3, 3);

    // clang-format off
    auto A_host = Kokkos::create_mirror_view(A);
    A_host(0, 0) = 2.0;  A_host(0, 1) = -1.0; A_host(0, 2) = 0.0;
    A_host(1, 0) = -1.0; A_host(1, 1) = 2.0;  A_host(1, 2) = -1.0;
    A_host(2, 0) = 0.0;  A_host(2, 1) = -1.0; A_host(2, 2) = 2.0;
    // clang-format on

    Kokkos::deep_copy(A, A_host);
    return A;
  }

  vector_t get_q() const {
    vector_t q(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q"), 3);
    backend_t::apply(-1.0, get_A(), get_exact_solution(), 0.0, q);
    return q;
  }
};

struct RandomLCP {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using scalar_t = double;
  using layout_t = Kokkos::View<scalar_t*, mem_space>::array_layout;
  using vector_t = Kokkos::View<scalar_t*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<scalar_t**, layout_t, mem_space>;
  using backend_t = convex::KokkosBackend<exec_space>;

  std::string name() const {
    return "RandomLCP" + std::to_string(size_);
  }

  unsigned size() const {
    return size_;
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  RandomLCP(unsigned size) : size_(size) {
    // 1. Build M
    A_ = gen_random_p_matrix(size_);

    // 2. Choose disjoint supports for z* and w*
    x_star_ = vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_star"), size_);
    grad_star_ = vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "grad_star"), size_);
    auto x_star_host = Kokkos::create_mirror_view(x_star_);
    auto grad_star_host = Kokkos::create_mirror_view(grad_star_);
    for (size_t i = 0; i < size_; ++i) {
      double u01 = static_cast<double>(rand()) / RAND_MAX;
      bool is_active = static_cast<double>(rand()) / RAND_MAX < 0.5;
      if (is_active) {
        x_star_host[i] = u01 * 0.9 + 0.1;
        grad_star_host[i] = 0.0;
      } else {
        x_star_host[i] = 0.0;
        grad_star_host[i] = u01 * 0.9 + 0.1;
      }
    }
    Kokkos::deep_copy(x_star_, x_star_host);
    Kokkos::deep_copy(grad_star_, grad_star_host);

    // 3. q that makes (z*, w*) solve the LCP
    // q_ = grad_star_ - A_ * x_star_;
    q_ = vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q"), size_);
    backend_t::axpby(1.0, grad_star_, 0.0, q_);
    backend_t::apply(-1.0, A_, x_star_, 1.0, q_);
  }

  auto get_space() const {
    return convex::space::LowerBound<scalar_t>(0.0);
  }

  vector_t get_exact_solution() const {
    return x_star_;
  }

  linear_op_t get_A() const {
    return A_;
  }

  vector_t get_q() const {
    return q_;
  }

  linear_op_t gen_random_matrix(unsigned size) {
    linear_op_t mat(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mat"), size, size);

    // Fill with random values in [-1, 1] (not a statistically random matrix but this is a test)
    Kokkos::parallel_for(
        "gen_random_matrix", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {size, size}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          openrand::Philox rng(i, j);
          mat(i, j) = rng.uniform<double>(-1.0, 1.0);
        });

    return mat;
  }

  linear_op_t gen_random_p_matrix(unsigned size) {
    // Strictly diagonally dominant with positive diagonal
    linear_op_t mat = gen_random_matrix(size);

    // Team loop over each row, thread reduce over columns
    using team_policy = Kokkos::TeamPolicy<exec_space>;
    using team_member = typename team_policy::member_type;
    Kokkos::parallel_for(
        "gen_random_p_matrix", team_policy(size, Kokkos::AUTO()), KOKKOS_LAMBDA(const team_member& team) {
          size_t i = team.league_rank();
          scalar_t off_diag_abs_row_sum = 0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 0, size),
              [&](const size_t j, scalar_t& sum) { sum += Kokkos::abs(mat(i, j)) * (j != i); }, off_diag_abs_row_sum);
          mat(i, i) = off_diag_abs_row_sum + 10.0;
        });

    return mat;
  }

 private:
  unsigned size_;
  linear_op_t A_;
  vector_t q_;
  vector_t x_star_;
  vector_t grad_star_;
};

namespace congruent {

template <class NonCongruentProblem>
struct CongruentLCPProblemWrapper {
  // D = I
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using scalar_t = typename NonCongruentProblem::scalar_t;
  using layout_t = typename NonCongruentProblem::layout_t;
  using vector_t = typename NonCongruentProblem::vector_t;
  using linear_op_t = typename NonCongruentProblem::linear_op_t;
  using backend_t = typename NonCongruentProblem::backend_t;

  std::string name() const {
    return "CongruentLCPProblemWrapper<" + cdp.name() + ">";
  }

  unsigned size() const {
    return cdp.size();
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return cdp.get_space();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_x_exact() const {
    return cdp.get_exact_solution();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_f_exact() const {
    return cdp.get_exact_solution();
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_u_exact() const {
    return get_M() * get_x_exact();
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_D() const {
    return gen_identity(size());
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_M() const {
    return cdp.get_A();
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_DT() const {
    return gen_identity(size());
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    return cdp.get_q();
  }

  linear_op_t gen_identity(unsigned size) {
    linear_op_t mat(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mat"), size, size);
    Kokkos::deep_copy(mat, 0.0);
    Kokkos::parallel_for(
        "gen_identity", Kokkos::RangePolicy<exec_space>(0, size), KOKKOS_LAMBDA(const size_t i) { mat(i, i) = 1.0; });
    return mat;
  }

  NonCongruentProblem cdp;
};

}  // namespace congruent

}  // namespace kokkos_backend
//@}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

void run_mundy_math_test(const auto& test) {
  std::cout << "Running test: " << test.name() << std::endl;
  // Problem setup
  auto A = test.get_A();
  auto q = test.get_q();
  auto space = test.get_space();
  auto x_exact = test.get_exact_solution();

  using vector_t = decltype(x_exact);
  vector_t x{}, grad{}, x_tmp{}, grad_tmp{};

  x.fill(99.99);  // use a bad initial guess to force more iterations

  // Build the problem
  const auto cqpp = make_cqpp<convex::MundyMathBackend>(A, q, space);

  // Strategy + state
  convex::PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd_solution_strategy(cfg);
  auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, cfg.max_iters);
  for (size_t i = 0; i < vector_t::size; ++i) {
    EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
  }
}

void run_mundy_math_congruent_test(const auto& test) {
  std::cout << "Running congruent test: " << test.name() << std::endl;
  // Problem setup
  auto DT = test.get_DT();
  auto M = test.get_M();
  auto D = test.get_D();
  auto q = test.get_q();
  auto space = test.get_space();
  auto x_exact = test.get_x_exact();
  auto f_exact = test.get_f_exact();
  auto u_exact = test.get_u_exact();

  std::cout << "DT:\n" << DT << "\nM:\n" << M << "\nD:\n" << D << "\nq:\n" << q << std::endl;

  using vector_t = decltype(x_exact);
  vector_t x{}, grad{}, x_tmp{}, grad_tmp{};
  vector_t f{}, u{};

  x.fill(99.99);  // use a bad initial guess to force more iterations

  // Build quadratic form operator + user-owned workspace, then the problem
  const auto A = convex::make_quadratic_form<convex::MundyMathBackend>(DT, M, D);
  auto workspace = A.make_workspace(f, u);  // intermediate variables f = D x, u = M f
  const auto cqpp = make_cqpp<convex::MundyMathBackend>(A, q, space, workspace);

  // Strategy + state
  convex::PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd_solution_strategy(cfg);
  auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, cfg.max_iters);
  EXPECT_TRUE(cqpp.workspace().is_committed());
  for (size_t i = 0; i < vector_t::size; ++i) {
    EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
    EXPECT_NEAR(f[i], f_exact[i], 10 * cfg.tol);
    EXPECT_NEAR(u[i], u_exact[i], 10 * cfg.tol);
  }
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
void run_kokkos_test(const auto& test) {
  // Problem setup
  auto exec_space = test.get_exec_space();
  auto A = test.get_A();
  auto q = test.get_q();
  auto space = test.get_space();
  auto x_exact = test.get_exact_solution();
  unsigned size = test.size();

  using vector_t = decltype(x_exact);
  vector_t x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), size);
  vector_t grad(Kokkos::view_alloc(Kokkos::WithoutInitializing, "grad"), size);
  vector_t x_tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_tmp"), size);
  vector_t grad_tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "grad_tmp"), size);

  Kokkos::deep_copy(x, 99.99);  // use a bad initial guess to force more iterations

  // Build the problem
  const auto cqpp = make_cqpp<convex::KokkosBackend<decltype(exec_space)>>(A, q, space);

  // Strategy + state
  convex::PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd_solution_strategy(cfg);
  auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, cfg.max_iters);

  // Copy x to host for comparison
  auto x_host = Kokkos::create_mirror_view(x);
  auto x_exact_host = Kokkos::create_mirror_view(x_exact);
  Kokkos::deep_copy(x_host, x);
  Kokkos::deep_copy(x_exact_host, x_exact);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(x_host[i], x_exact_host[i], 10 * cfg.tol);
  }
}

void run_kokkos_congruent_test(const auto& test) {
  // Problem setup
  auto exec_space = test.get_exec_space();
  auto D = test.get_D();
  auto M = test.get_M();
  auto DT = test.get_DT();
  auto q = test.get_q();
  auto space = test.get_space();
  auto x_exact = test.get_x_exact();
  auto f_exact = test.get_f_exact();
  auto u_exact = test.get_u_exact();
  unsigned size = test.size();

  using vector_t = decltype(x_exact);
  vector_t x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), size);
  vector_t grad(Kokkos::view_alloc(Kokkos::WithoutInitializing, "grad"), size);
  vector_t x_tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_tmp"), size);
  vector_t grad_tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "grad_tmp"), size);
  vector_t f(Kokkos::view_alloc(Kokkos::WithoutInitializing, "f"), size);
  vector_t u(Kokkos::view_alloc(Kokkos::WithoutInitializing, "u"), size);

  Kokkos::deep_copy(x, 99.99);  // use a bad initial guess to force more iterations

  // Build quadratic form operator + user-owned workspace, then the problem
  const auto A = convex::make_quadratic_form<convex::KokkosBackend<decltype(exec_space)>>(DT, M, D);
  auto workspace = A.make_workspace(f, u);  // intermediate variables f = D x, u = M f
  const auto cqpp = make_cqpp<convex::KokkosBackend<decltype(exec_space)>>(A, q, space, workspace);

  // Strategy + state
  convex::PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd_solution_strategy(cfg);
  auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, cfg.max_iters);
  EXPECT_TRUE(cqpp.workspace().is_committed());

  // Copy x to host for comparison
  auto x_host = Kokkos::create_mirror_view(x);
  auto f_host = Kokkos::create_mirror_view(f);
  auto u_host = Kokkos::create_mirror_view(u);
  auto x_exact_host = Kokkos::create_mirror_view(x_exact);
  auto f_exact_host = Kokkos::create_mirror_view(f_exact);
  auto u_exact_host = Kokkos::create_mirror_view(u_exact);
  Kokkos::deep_copy(x_host, x);
  Kokkos::deep_copy(f_host, f);
  Kokkos::deep_copy(u_host, u);
  Kokkos::deep_copy(x_exact_host, x_exact);
  Kokkos::deep_copy(f_exact_host, f_exact);
  Kokkos::deep_copy(u_exact_host, u_exact);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(x_host[i], x_exact_host[i], 10 * cfg.tol);
    EXPECT_NEAR(f_host[i], f_exact_host[i], 10 * cfg.tol);
    EXPECT_NEAR(u_host[i], u_exact_host[i], 10 * cfg.tol);
  }
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

TEST(Convex, MundyMathAnalyticalSolutions) {
  auto test_cases = std::make_tuple(math_backend::UnconstrainedSPD1Problem{},          //
                                    math_backend::InactiveBoxConstrainedSPDProblem{},  //
                                    math_backend::ActiveBoxConstrainedSPDProblem{},    //
                                    math_backend::RandomLCP<3>{},                      //
                                    math_backend::RandomLCP<7>{});
  std::apply([](auto&&... test_case) { (run_mundy_math_test(test_case), ...); }, test_cases);
}

TEST(Convex, MundyMathCongruentAnalyticalSolutions) {
  using math_backend::congruent::CongruentLCPProblemWrapper;
  auto test_cases = std::make_tuple(CongruentLCPProblemWrapper{math_backend::UnconstrainedSPD1Problem{}},          //
                                    CongruentLCPProblemWrapper{math_backend::InactiveBoxConstrainedSPDProblem{}},  //
                                    CongruentLCPProblemWrapper{math_backend::ActiveBoxConstrainedSPDProblem{}},    //
                                    CongruentLCPProblemWrapper{math_backend::RandomLCP<3>{}},                      //
                                    CongruentLCPProblemWrapper{math_backend::RandomLCP<7>{}});
  std::apply([](auto&&... test_case) { (run_mundy_math_congruent_test(test_case), ...); }, test_cases);
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
TEST(Convex, KokkosAnalyticalSolutions) {
  auto test_cases = std::make_tuple(kokkos_backend::UnconstrainedSPD1Problem{},          //
                                    kokkos_backend::InactiveBoxConstrainedSPDProblem{},  //
                                    kokkos_backend::ActiveBoxConstrainedSPDProblem{},    //
                                    kokkos_backend::RandomLCP{3},                        //
                                    kokkos_backend::RandomLCP{7},                        //
                                    kokkos_backend::RandomLCP{200});
  std::apply([](auto&&... test_case) { (run_kokkos_test(test_case), ...); }, test_cases);
}

TEST(Convex, KokkosCongruentAnalyticalSolutions) {
  using kokkos_backend::congruent::CongruentLCPProblemWrapper;
  auto test_cases = std::make_tuple(CongruentLCPProblemWrapper{kokkos_backend::UnconstrainedSPD1Problem{}},          //
                                    CongruentLCPProblemWrapper{kokkos_backend::InactiveBoxConstrainedSPDProblem{}},  //
                                    CongruentLCPProblemWrapper{kokkos_backend::ActiveBoxConstrainedSPDProblem{}},    //
                                    CongruentLCPProblemWrapper{kokkos_backend::RandomLCP{3}},                        //
                                    CongruentLCPProblemWrapper{kokkos_backend::RandomLCP{7}},                        //
                                    CongruentLCPProblemWrapper{kokkos_backend::RandomLCP{200}});
  std::apply([](auto&&... test_case) { (run_kokkos_congruent_test(test_case), ...); }, test_cases);
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

}  // namespace

}  // namespace math

}  // namespace mundy
