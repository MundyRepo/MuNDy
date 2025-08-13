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
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

#include <Kokkos_Core.hpp>  // for Kokkos::Array

// C++ core libs
#include <ostream>  // for std::cout

// Mundy libs
#include <mundy_math/Matrix.hpp>  // for mundy::math::Matrix
#include <mundy_math/Vector.hpp>  // for mundy::math::Vector
#include <mundy_math/convex.hpp>  // for mundy::math::solve_lcp/solve_cqpp

namespace mundy {

namespace math {

namespace {

//! \name MundyMath backend test problems
//@{

struct UnconstrainedSPD1Problem {
  using scalar_t = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = convex::MundyMathBackend<scalar_t, 3>;

  std::string name() const {
    return "UnconstrainedSPD1Problem";
  }

  auto get_space() const {
    return convex::space::Unconstrained<scalar_t>();
  }

  vector_t get_exact_solution() const {
    return Vector3d{1.0, 0.0, 1.0};
  }

  linear_op_t get_A() const {
    return Matrix3d{2.0,  -1.0, 0.0,   //
                    -1.0, 2.0,  -1.0,  //
                    0.0,  -1.0, 2.0};
  }

  vector_t get_q() const {
    return -get_A() * get_exact_solution();
  }
};

struct InactiveBoxConstrainedSPDProblem {
  using scalar_t = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = convex::MundyMathBackend<scalar_t, 3>;

  std::string name() const {
    return "InactiveBoxConstrainedSPDProblem";
  }

  auto get_space() const {
    return convex::space::Bounded<scalar_t>(0.0, 2.0);
  }

  vector_t get_exact_solution() const {
    return Vector3d{1.0, 0.0, 1.0};
  }

  linear_op_t get_A() const {
    return Matrix3d{2.0,  -1.0, 0.0,   //
                    -1.0, 2.0,  -1.0,  //
                    0.0,  -1.0, 2.0};
  }

  vector_t get_q() const {
    return -get_A() * get_exact_solution();
  }
};

struct ActiveBoxConstrainedSPDProblem {
  using scalar_t = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = convex::MundyMathBackend<scalar_t, 3>;

  std::string name() const {
    return "ActiveBoxConstrainedSPDProblem";
  }

  auto get_space() const {
    return convex::space::Bounded<scalar_t>(9.0, 10.0);
  }

  vector_t get_exact_solution() const {
    return Vector3d{9.0, 9.0, 9.0};
  }

  linear_op_t get_A() const {
    return Matrix3d{2.0,  -1.0, 0.0,   //
                    -1.0, 2.0,  -1.0,  //
                    0.0,  -1.0, 2.0};
  }

  vector_t get_q() const {
    return -get_A() * get_exact_solution();
  }
};

template <size_t N>
struct RandomLCP {
  using scalar_t = double;
  using vector_t = Vector<scalar_t, N>;
  using linear_op_t = Matrix<scalar_t, N, N>;
  using backend_t = convex::MundyMathBackend<scalar_t, N>;

  std::string name() const {
    return "RandomLCP" + std::to_string(N);
  }

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

 private:
  linear_op_t gen_random_matrix() {
    linear_op_t mat;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        mat(i, j) = 1.0 - 2 * static_cast<double>(rand()) / RAND_MAX;
      }
    }
    return mat;
  }

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

  linear_op_t A_;
  vector_t q_;
  vector_t x_star_;
  vector_t grad_star_;
};
//@}

void run_mundy_math_test(const auto& test) {
  // Problem setup
  auto A = test.get_A();
  auto q = test.get_q();
  auto space = test.get_space();
  auto x_exact = test.get_exact_solution();

  using vector_t = decltype(x_exact);
  vector_t x{}, grad{}, x_tmp{}, grad_tmp{};

  x.fill(99.99);  // use a bad initial guess to force more iterations

  // Build the problem
  const auto cqpp = make_mundy_math_cqpp(A, q, space);

  // Reuse the backend token from the problem
  const auto backend = cqpp.backend();

  // Strategy + state
  convex::PGDConfig cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd_solution_strategy(backend, cfg);
  auto pgd_state = make_pgd_state(backend, x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, cfg.max_iters);
  for (size_t i = 0; i < vector_t::size; ++i) {
    EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
  }
}

TEST(Convex, AnalyticalSolutions) {
  auto test_cases = std::make_tuple(UnconstrainedSPD1Problem{},          //
                                    InactiveBoxConstrainedSPDProblem{},  //
                                    ActiveBoxConstrainedSPDProblem{},    //
                                    RandomLCP<3>{},                      //
                                    RandomLCP<7>{});
  std::apply([](auto&&... test_case) { (run_mundy_math_test(test_case), ...); }, test_cases);
}

}  // namespace

}  // namespace math

}  // namespace mundy
