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
#include <ostream>      // for std::cout 

// Mundy libs
#include <mundy_math/Vector.hpp>  // for mundy::math::Vector
#include <mundy_math/Matrix.hpp>  // for mundy::math::Matrix
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
//@}

TEST(Convex, AnalyticalSolutions) {
  UnconstrainedSPD1Problem test1;

  // Problem setup
  auto A = test1.get_A();
  auto q = test1.get_q();
  auto space = test1.get_space();
  auto x_exact = test1.get_exact_solution();
  Vector3d x{0.0, 0.0, 0.0}, grad{}, x_tmp{}, grad_tmp{};

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
  std::cout << result << std::endl;

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, cfg.max_iters);

  std::cout << "x: " << x << std::endl;
  std::cout << "x_exact: " << x_exact << std::endl;
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(x[i], x_exact[i], cfg.tol);
  }
}

}  // namespace

}  // namespace math

}  // namespace mundy
