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

/// \file UnitTestConvex.cpp
/// \brief Projected-gradient descent (pgd.hpp) with Barzilai-Borwein, nonmonotone Barzilai-Borwein, and MPRGP steps on
/// constrained quadratic programs (cqpp.hpp) with known minimizers, and the projected residuals (residuals.hpp).
///
/// Reference problems, each on MundyMathBackend and KokkosBackend:
///   - A 3 x 3 SPD problem without constraints, and inside a box [0, 2] that its minimizer does not touch and a box
///     [9, 10] that it does.
///   - The same SPD problem under the upper bound x <= 2, which its minimizer touches.
///   - A random LCP, 0 <= A x + q _|_ x >= 0, built from a chosen x* and A x* + q with disjoint supports.
///   - A periodic Laplacian LCP, A = periodic [-1, 2.01, -1], with x* = 1 on even entries and A x* + q = 1 on odd
///     ones; plain Barzilai-Borwein steps fall into a five-step limit cycle on it.
///   - Each of the above with A congruent to D^T M D (D = I), and a random mixed problem whose unilateral block is
///     reduced through the Schur complement of its bilateral one.
///   - Two nearly parallel rows, A = [[1, cos delta], [cos delta, 1]] and q = -(1, 1 + eps), whose minimizer
///     x* = (0, 1 + eps) leaves the first row slack by (1 + eps) cos delta - 1.

// External
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

#include <Kokkos_Core.hpp>

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
#include <KokkosBlas.hpp>
#include <KokkosBlas_gesv.hpp>
#endif

// C++ core
#include <cmath>        // for std::cos, std::ldexp, std::sqrt
#include <cstdlib>      // for std::srand
#include <limits>       // for std::numeric_limits
#include <ostream>      // for std::cout
#include <string_view>  // for std::string_view
#include <type_traits>  // for std::is_same_v

// Mundy
#include <mundy_math/Matrix.hpp>  // for mundy::Matrix
#include <mundy_math/Vector.hpp>  // for mundy::Vector
#include <mundy_math/cmath.hpp>
#include <mundy_math/convex_spaces.hpp>
#include <mundy_math/cqpp.hpp>
#include <mundy_math/lcp.hpp>
#include <mundy_math/linear_system.hpp>  // for mundy::{CGConfig, make_cg_inv_op}
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix
#include <mundy_utils/rng.hpp>           // for mundy::make_philox

namespace mundy {

namespace {

//! \name Compile-time contracts
//@{

static_assert(ProjectedResidualPolicy<LinfNormProjectedGradientResidual, MundyMathBackend, Vector3d, Vector3d,
                                      LowerBoundSpace<double>>,
              "The projected-gradient residual measures an iterate against a convex space.");
static_assert(ProjectedResidualPolicy<LinfNormProjectedDiffResidual, MundyMathBackend, Vector3d, Vector3d,
                                      LowerBoundSpace<double>>,
              "The projected-difference residual measures an iterate against a convex space.");
using cqpp3_t = decltype(make_cqpp<MundyMathBackend>(std::declval<Matrix3d>(), std::declval<Vector3d>(),
                                                     std::declval<LowerBoundSpace<double>>()));
using mprgp_state3_t = PGDState<double, Vector3d&, Vector3d&, Vector3d&, Vector3d&, MPRGPStepState<double>>;
static_assert(
    PGDStepStrategy<MPRGPStepStrategy<Vector3d&>, cqpp3_t, mprgp_state3_t> &&
        PGDStepStrategy<MPRGPStepStrategy<Vector3d&, Vector3d&, Kokkos::Array<Vector3d, 2>&>, cqpp3_t, mprgp_state3_t>,
    "MPRGP steps, from the cold start or from an estimate vector and a basis, are PGD step strategies.");
static_assert(
    std::is_same_v<MPRGPStepStrategy<Vector3d&>::step_state_t<double>, MPRGPStepState<double>> &&
        std::is_same_v<MPRGPStepStrategy<Vector3d&, Vector3d&, Kokkos::Array<Vector3d, 2>&>::step_state_t<double>,
                       MPRGPStepState<double>>,
    "MPRGP's memory between steps is its expansion step, from either start.");
static_assert(!ProjectedResidualPolicy<L2Residual, MundyMathBackend, Vector3d, Vector3d, LowerBoundSpace<double>>,
              "A vector residual is not a projected residual.");
//@}

//! \name MundyMath backend test problems
//@{

namespace math_backend {

struct UnconstrainedSPD1Problem {
  using value_type = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = MundyMathBackend;

  std::string name() const {
    return "UnconstrainedSPD1Problem";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return UnconstrainedSpace<value_type>();
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
  using value_type = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = MundyMathBackend;

  std::string name() const {
    return "InactiveBoxConstrainedSPDProblem";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return BoundedSpace<value_type>(0.0, 2.0);
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
  using value_type = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = MundyMathBackend;

  std::string name() const {
    return "ActiveBoxConstrainedSPDProblem";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return BoundedSpace<value_type>(9.0, 10.0);
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

/// \brief x <= 2 with two entries pressed against it: x* = (2, 1, 2) and A x* + q = (-1, 0, -1).
struct ActiveUpperBoundSPDProblem {
  using value_type = double;
  using vector_t = Vector3d;
  using linear_op_t = Matrix3d;
  using backend_t = MundyMathBackend;

  std::string name() const {
    return "ActiveUpperBoundSPDProblem";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return UpperBoundSpace<value_type>(2.0);
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_exact_solution() const {
    return Vector3d{2.0, 1.0, 2.0};
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_A() const {
    return Matrix3d{2.0,  -1.0, 0.0,   //
                    -1.0, 2.0,  -1.0,  //
                    0.0,  -1.0, 2.0};
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    return Vector3d{-1.0, 0.0, -1.0} - get_A() * get_exact_solution();
  }
};

template <size_t N>
struct RandomLCP {
  using value_type = double;
  using vector_t = Vector<value_type, N>;
  using linear_op_t = Matrix<value_type, N, N>;
  using backend_t = MundyMathBackend;

  std::string name() const {
    return "RandomLCP" + std::to_string(N);
  }

  KOKKOS_INLINE_FUNCTION
  RandomLCP() {
    // 1. Build A
    A_ = gen_random_spd_matrix();

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
    return LowerBoundSpace<value_type>(0.0);
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
  linear_op_t gen_random_symmetric_matrix() {
    linear_op_t mat;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = i; j < N; ++j) {
        mat(i, j) = 1.0 - 2 * static_cast<double>(rand()) / RAND_MAX;
        mat(j, i) = mat(i, j);
      }
    }
    return mat;
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t gen_random_spd_matrix() {
    // Symmetric and strictly diagonally dominant with a positive diagonal, so SPD
    linear_op_t mat = gen_random_symmetric_matrix();
    for (size_t i = 0; i < N; ++i) {
      value_type off_diag_abs_row_sum = 0;
      for (size_t j = 0; j < N; ++j) {
        off_diag_abs_row_sum += abs(mat(i, j)) * (i != j);
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

/// \brief 0 <= A x + q _|_ x >= 0 with A = L + I / 100 for the periodic Laplacian L = [-1, 2, -1], whose bottom
/// eigenvector is the all-ones vector: x* is 1 on the even entries, and A x* + q is 1 on the odd ones.
template <size_t N>
struct PeriodicLaplacianLCP {
  using value_type = double;
  using vector_t = Vector<value_type, N>;
  using linear_op_t = Matrix<value_type, N, N>;
  using backend_t = MundyMathBackend;

  std::string name() const {
    return "PeriodicLaplacianLCP" + std::to_string(N);
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return LowerBoundSpace<value_type>(0.0);
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_exact_solution() const {
    vector_t x;
    for (size_t i = 0; i < N; ++i) {
      x[i] = i % 2 == 0 ? 1.0 : 0.0;
    }
    return x;
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_A() const {
    auto A = linear_op_t::zeros();
    for (size_t i = 0; i < N; ++i) {
      A(i, i) = 2.0 + 0.01;
      A(i, (i + 1) % N) = -1.0;
      A(i, (i + N - 1) % N) = -1.0;
    }
    return A;
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    vector_t grad_star;
    for (size_t i = 0; i < N; ++i) {
      grad_star[i] = i % 2 == 0 ? 0.0 : 1.0;
    }
    return grad_star - get_A() * get_exact_solution();
  }
};

/// \brief 0 <= A x + q _|_ x >= 0 with A = blockdiag(L + I / 100, 2 K) for the 8 x 8 periodic Laplacian L and the
/// 16 x 16 Kac-Murdock-Szego matrix K_ij = 0.9^|i - j|, whose rows couple to none of L's. x* is PeriodicLaplacianLCP's
/// on the first block and 0.5 + (i mod 5) / 10 on every third row i of the second; A x* + q is 1 off x*'s support.
///
/// L's top eigenvector (1, -1, ...) / sqrt(8), padded with zeros, is an eigenvector of A for 4.01, while the second
/// block's curvature along its all-ones vector exceeds 19.
struct BlockedKMSLCP {
  static constexpr size_t N = 24;
  using value_type = double;
  using vector_t = Vector<value_type, N>;
  using linear_op_t = Matrix<value_type, N, N>;
  using backend_t = MundyMathBackend;

  std::string name() const {
    return "BlockedKMSLCP";
  }

  KOKKOS_INLINE_FUNCTION
  auto get_space() const {
    return LowerBoundSpace<value_type>(0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static bool on_support(size_t i) {
    return i < 8 ? i % 2 == 0 : (i - 8) % 3 == 0;
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_exact_solution() const {
    vector_t x;
    for (size_t i = 0; i < N; ++i) {
      x[i] = !on_support(i) ? 0.0 : (i < 8 ? 1.0 : 0.5 + static_cast<double>((i - 8) % 5) / 10.0);
    }
    return x;
  }

  KOKKOS_INLINE_FUNCTION
  linear_op_t get_A() const {
    auto A = linear_op_t::zeros();
    for (size_t i = 0; i < 8; ++i) {
      A(i, i) = 2.0 + 0.01;
      A(i, (i + 1) % 8) = -1.0;
      A(i, (i + 7) % 8) = -1.0;
    }
    for (size_t i = 0; i < 16; ++i) {
      double power = 2.0;  // 2 * 0.9^|i - j|, for j moving away from i
      for (size_t d = 0; i + d < 16; ++d, power *= 0.9) {
        A(8 + i, 8 + i + d) = power;
        A(8 + i + d, 8 + i) = power;
      }
    }
    return A;
  }

  KOKKOS_INLINE_FUNCTION
  vector_t get_q() const {
    vector_t grad_star;
    for (size_t i = 0; i < N; ++i) {
      grad_star[i] = on_support(i) ? 0.0 : 1.0;
    }
    return grad_star - get_A() * get_exact_solution();
  }

  /// \brief L's top eigenvector, padded with zeros.
  KOKKOS_INLINE_FUNCTION
  vector_t get_first_block_top_eigenvector() const {
    vector_t e;
    for (size_t i = 0; i < N; ++i) {
      e[i] = i < 8 ? (i % 2 == 0 ? 1.0 : -1.0) / sqrt(8.0) : 0.0;
    }
    return e;
  }
};

namespace congruent {

template <class NonCongruentProblem>
struct CongruentLCPWrapper {
  // D = I
  using value_type = typename NonCongruentProblem::value_type;
  using vector_t = typename NonCongruentProblem::vector_t;
  using linear_op_t = typename NonCongruentProblem::linear_op_t;
  using backend_t = typename NonCongruentProblem::backend_t;

  std::string name() const {
    return "CongruentLCPWrapper<" + cdp.name() + ">";
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

namespace mixed {

/// The following problem is a randomly generated instance of the following mixed CCQP:
///   x^*, y^* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
///
/// We refactor this into the desired form by defining:
///   S := (B^T M B + K^{-1})^{-1} (symmetric positive definite)
///
///  M is size NZ x NZ,
///  B is NZ x NY,
///  Kinv is NY x NY,
///  D is NZ x NX,
///  q is NX,
///  b is NY,
///  x* in R^NX,
///  y* in R^NY.
///
/// NX: num unilateral constraints
/// NY: num bilateral constraints
/// NZ: num configurational variables (in the intermediate space)
template <size_t NX, size_t NY, size_t NZ>
struct RandomMixedCongruentCCQP {
  using value_type = double;
  using vecx_t = Vector<value_type, NX>;
  using vecy_t = Vector<value_type, NY>;
  using vecz_t = Vector<value_type, NZ>;
  using matxx_t = Matrix<value_type, NX, NX>;
  using matxy_t = Matrix<value_type, NX, NY>;
  using matxz_t = Matrix<value_type, NX, NZ>;
  using matyx_t = Matrix<value_type, NY, NX>;
  using matyy_t = Matrix<value_type, NY, NY>;
  using matyz_t = Matrix<value_type, NY, NZ>;
  using matzx_t = Matrix<value_type, NZ, NX>;
  using matzy_t = Matrix<value_type, NZ, NY>;
  using matzz_t = Matrix<value_type, NZ, NZ>;

  RandomMixedCongruentCCQP(unsigned seed = 1) {
    srand(seed);
    build();
  }

  std::string name() const {
    return "RandomMixedCongruentCCQP<" + std::to_string(NX) + "," + std::to_string(NY) + "," + std::to_string(NZ) + ">";
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION auto get_space_x() const { return LowerBoundSpace<value_type>(0.0); }
  KOKKOS_INLINE_FUNCTION vecx_t get_exact_x() const { return x_star_; }
  KOKKOS_INLINE_FUNCTION vecy_t get_exact_y() const { return y_star_; }
  KOKKOS_INLINE_FUNCTION matxz_t get_DT() const { return transpose(get_D()); }
  KOKKOS_INLINE_FUNCTION matzz_t get_M() const { return M_; }
  KOKKOS_INLINE_FUNCTION matzx_t get_D() const { return D_; }
  KOKKOS_INLINE_FUNCTION vecx_t get_q() const { return q_; }
  KOKKOS_INLINE_FUNCTION matyz_t get_BT() const { return transpose(B_); }
  KOKKOS_INLINE_FUNCTION matyy_t get_S() const { return S_; }
  KOKKOS_INLINE_FUNCTION matzy_t get_B() const { return B_; }
  KOKKOS_INLINE_FUNCTION vecy_t get_b() const { return b_; }
  // clang-format on

  // Helper: random in [-1,1]
  static value_type urand() {
    return value_type(1.0) - value_type(2.0) * (value_type(rand()) / RAND_MAX);
  }

  matzz_t gen_spd_zz(value_type diag_boost = 5.0) {
    matzz_t R;
    for (size_t i = 0; i < NZ; ++i) {
      for (size_t j = 0; j < NZ; ++j) {
        R(i, j) = urand();
      }
    }
    matzz_t A = transpose(R) * R;
    for (size_t i = 0; i < NZ; ++i) {
      A(i, i) += diag_boost;
    }
    return A;
  }

  void make_D_full_rank() {
    // NZ >= NX assumed. Force full column rank by embedding I.
    for (size_t i = 0; i < NZ; ++i) {
      for (size_t j = 0; j < NX; ++j) {
        D_(i, j) = urand();
      }
    }
    for (size_t j = 0; j < NX; ++j) {
      D_(j, j) += 3.0;  // strong diagonal injection
    }
  }

  void build() {
    // 1) Choose M SPD, Kinv SPD, and random B,D (with D full rank)
    M_ = gen_spd_zz(/*diag_boost=*/10.0);

    make_D_full_rank();

    for (size_t i = 0; i < NZ; ++i) {
      for (size_t j = 0; j < NY; ++j) {
        B_(i, j) = 0.2 * urand();  // small-ish coupling
      }
    }

    Kinv_ = matyy_t{};
    for (size_t i = 0; i < NY; ++i) {
      Kinv_(i, i) = 10.0 + std::abs(urand());  // big diag -> A well-conditioned
    }

    // 2) Build S = (B^T M B + Kinv)^{-1} (SPD)
    S_ = inverse(transpose(B_) * M_ * B_ + Kinv_);

    // 3) Choose x_star with random active set, and choose slack s_star
    vecx_t s_star;
    for (size_t i = 0; i < NX; ++i) {
      bool active = (double(rand()) / RAND_MAX) < 0.5;
      if (active) {
        x_star_[i] = 0.1 + 0.9 * (double(rand()) / RAND_MAX);
        s_star[i] = 0.0;
      } else {
        x_star_[i] = 0.0;
        s_star[i] = 0.1 + 0.9 * (double(rand()) / RAND_MAX);
      }
    }

    // 4) Optionally choose nonzero b
    for (size_t i = 0; i < NY; ++i) {
      b_[i] = 0.3 * urand();
    }

    // 5) Compute H and g via solves
    //    T := S * (B^T M D)  => NY x NX
    matyx_t BTC = transpose(B_) * M_ * D_;
    matyx_t T = S_ * BTC;

    matxx_t H = transpose(D_) * M_ * D_ - (transpose(D_) * M_ * B_) * T;

    // g := s_star - H x_star
    vecx_t g = s_star - H * x_star_;

    // 6) Set c so that reduced gradient is g:
    // g = c - D^T M B S b  => c = g + D^T M B S b
    vecy_t u = S_ * b_;  // mundy-math problems are small enough to use a direct inverse
    q_ = g + (transpose(D_) * M_ * B_) * u;

    // 7) y_star = -S (b + B^T M D x_star)
    vecy_t rhs = b_ + BTC * x_star_;
    y_star_ = -S_ * rhs;
  }

 private:
  // Problem data
  matzz_t M_;  // SPD
  matzx_t D_;
  matzy_t B_;
  matyy_t Kinv_;  // SPD
  vecx_t q_;
  vecy_t b_;

  // Refactored form data
  matyy_t S_;

  // Planted solution
  vecx_t x_star_;
  vecy_t y_star_;
};

}  // namespace mixed

}  // namespace math_backend
//@}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
//! \name Kokkos backend test problems (single process)
//@{
namespace kokkos_backend {

struct UnconstrainedSPD1Problem {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

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
    return UnconstrainedSpace<value_type>();
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

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

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
    return BoundedSpace<value_type>(0.0, 2.0);
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

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

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
    return BoundedSpace<value_type>(9.0, 10.0);
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

/// \brief x <= 2 with two entries pressed against it: x* = (2, 1, 2) and A x* + q = (-1, 0, -1).
struct ActiveUpperBoundSPDProblem {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

  std::string name() const {
    return "ActiveUpperBoundSPDProblem";
  }

  unsigned size() const {
    return 3;
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  auto get_space() const {
    return UpperBoundSpace<value_type>(2.0);
  }

  vector_t get_exact_solution() const {
    vector_t x_exact(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_exact"), 3);

    auto x_exact_host = Kokkos::create_mirror_view(x_exact);
    x_exact_host(0) = 2.0;
    x_exact_host(1) = 1.0;
    x_exact_host(2) = 2.0;
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
    auto q_host = Kokkos::create_mirror_view(q);
    q_host(0) = -1.0;
    q_host(1) = 0.0;
    q_host(2) = -1.0;
    Kokkos::deep_copy(q, q_host);
    backend_t::apply(-1.0, get_A(), get_exact_solution(), 1.0, q);
    return q;
  }
};

struct RandomLCP {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

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
    // 1. Build A
    A_ = gen_random_spd_matrix(size_);

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
    axpby(backend_t{}, 1.0, grad_star_, 0.0, q_);
    backend_t::apply(-1.0, A_, x_star_, 1.0, q_);
  }

  auto get_space() const {
    return LowerBoundSpace<value_type>(0.0);
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

  linear_op_t gen_random_symmetric_matrix(unsigned size) {
    linear_op_t mat(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mat"), size, size);

    // Fill with random values in [-1, 1] (not a statistically random matrix but this is a test), one draw per {i, j}
    Kokkos::parallel_for(
        "gen_random_symmetric_matrix", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {size, size}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          openrand::Philox rng = make_philox(Kokkos::min(i, j), Kokkos::max(i, j));
          mat(i, j) = rng.uniform<double>(-1.0, 1.0);
        });

    return mat;
  }

  linear_op_t gen_random_spd_matrix(unsigned size) {
    // Symmetric and strictly diagonally dominant with a positive diagonal, so SPD
    linear_op_t mat = gen_random_symmetric_matrix(size);

    // Team loop over each row, thread reduce over columns
    using team_policy = Kokkos::TeamPolicy<exec_space>;
    using team_member = typename team_policy::member_type;
    Kokkos::parallel_for(
        "gen_random_spd_matrix", team_policy(size, Kokkos::AUTO()), KOKKOS_LAMBDA(const team_member& team) {
          size_t i = team.league_rank();
          value_type off_diag_abs_row_sum = 0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 0, size),
              [&](const size_t j, value_type& sum) { sum += abs(mat(i, j)) * (j != i); }, off_diag_abs_row_sum);
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

/// \brief 0 <= A x + q _|_ x >= 0 with A = L + I / 100 for the periodic Laplacian L = [-1, 2, -1], whose bottom
/// eigenvector is the all-ones vector: x* is 1 on the even entries, and A x* + q is 1 on the odd ones.
struct PeriodicLaplacianLCP {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

  std::string name() const {
    return "PeriodicLaplacianLCP" + std::to_string(size_);
  }

  explicit PeriodicLaplacianLCP(unsigned size) : size_(size) {
  }

  unsigned size() const {
    return size_;
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  auto get_space() const {
    return LowerBoundSpace<value_type>(0.0);
  }

  vector_t get_exact_solution() const {
    vector_t x_exact(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_exact"), size_);
    auto x_exact_host = Kokkos::create_mirror_view(x_exact);
    for (size_t i = 0; i < size_; ++i) {
      x_exact_host(i) = i % 2 == 0 ? 1.0 : 0.0;
    }
    Kokkos::deep_copy(x_exact, x_exact_host);
    return x_exact;
  }

  linear_op_t get_A() const {
    linear_op_t A("A", size_, size_);
    auto A_host = Kokkos::create_mirror_view(A);
    for (size_t i = 0; i < size_; ++i) {
      A_host(i, i) = 2.0 + 0.01;
      A_host(i, (i + 1) % size_) = -1.0;
      A_host(i, (i + size_ - 1) % size_) = -1.0;
    }
    Kokkos::deep_copy(A, A_host);
    return A;
  }

  vector_t get_q() const {
    vector_t q(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q"), size_);
    auto q_host = Kokkos::create_mirror_view(q);
    for (size_t i = 0; i < size_; ++i) {
      q_host(i) = i % 2 == 0 ? 0.0 : 1.0;
    }
    Kokkos::deep_copy(q, q_host);
    backend_t::apply(-1.0, get_A(), get_exact_solution(), 1.0, q);
    return q;
  }

 private:
  unsigned size_;
};

/// \brief math_backend::BlockedKMSLCP over views.
struct BlockedKMSLCP {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using linear_op_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

  std::string name() const {
    return "BlockedKMSLCP";
  }

  unsigned size() const {
    return math_backend::BlockedKMSLCP::N;
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  auto get_space() const {
    return LowerBoundSpace<value_type>(0.0);
  }

  vector_t get_exact_solution() const {
    return to_view(math_backend::BlockedKMSLCP{}.get_exact_solution(), "x_exact");
  }

  vector_t get_q() const {
    return to_view(math_backend::BlockedKMSLCP{}.get_q(), "q");
  }

  vector_t get_first_block_top_eigenvector() const {
    return to_view(math_backend::BlockedKMSLCP{}.get_first_block_top_eigenvector(), "e");
  }

  linear_op_t get_A() const {
    const auto A_math = math_backend::BlockedKMSLCP{}.get_A();
    linear_op_t A("A", size(), size());
    auto A_host = Kokkos::create_mirror_view(A);
    for (size_t i = 0; i < size(); ++i) {
      for (size_t j = 0; j < size(); ++j) {
        A_host(i, j) = A_math(i, j);
      }
    }
    Kokkos::deep_copy(A, A_host);
    return A;
  }

 private:
  static vector_t to_view(const math_backend::BlockedKMSLCP::vector_t& v, const char* label) {
    vector_t view(std::string(label), math_backend::BlockedKMSLCP::N);
    auto host = Kokkos::create_mirror_view(view);
    for (size_t i = 0; i < math_backend::BlockedKMSLCP::N; ++i) {
      host(i) = v[i];
    }
    Kokkos::deep_copy(view, host);
    return view;
  }
};

namespace congruent {

template <class NonCongruentProblem>
struct CongruentLCPWrapper {
  // D = I
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using value_type = typename NonCongruentProblem::value_type;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vector_t = typename NonCongruentProblem::vector_t;
  using linear_op_t = typename NonCongruentProblem::linear_op_t;
  using backend_t = typename NonCongruentProblem::backend_t;

  std::string name() const {
    return "CongruentLCPWrapper<" + cdp.name() + ">";
  }

  unsigned size() const {
    return cdp.size();
  }

  auto get_exec_space() const {
    return exec_space{};
  }

  auto get_space() const {
    return cdp.get_space();
  }

  vector_t get_x_exact() const {
    return cdp.get_exact_solution();
  }

  vector_t get_f_exact() const {
    return cdp.get_exact_solution();
  }

  vector_t get_u_exact() const {
    vector_t u_exact(Kokkos::view_alloc(Kokkos::WithoutInitializing, "u_exact"), size());
    Kokkos::deep_copy(u_exact, 0.0);
    backend_t::apply(get_M(), get_x_exact(), u_exact);
    return u_exact;
  }

  linear_op_t get_D() const {
    return gen_identity(size());
  }

  linear_op_t get_M() const {
    return cdp.get_A();
  }

  linear_op_t get_DT() const {
    return gen_identity(size());
  }

  vector_t get_q() const {
    return cdp.get_q();
  }

  linear_op_t gen_identity(unsigned size) const {
    linear_op_t mat(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mat"), size, size);
    Kokkos::deep_copy(mat, 0.0);
    Kokkos::parallel_for(
        "gen_identity", Kokkos::RangePolicy<exec_space>(0, size), KOKKOS_LAMBDA(const size_t i) { mat(i, i) = 1.0; });
    return mat;
  }

  NonCongruentProblem cdp;
};

}  // namespace congruent

namespace mixed {

template <size_t NX, size_t NY, size_t NZ>
struct RandomMixedCongruentCCQP {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;

  using value_type = double;
  using layout_t = Kokkos::LayoutLeft;  // For compatibility with KokkosKernels
  using vecx_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using vecy_t = Kokkos::View<value_type*, layout_t, mem_space>;
  using matxx_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matxy_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matxz_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matyx_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matyy_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matyz_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matzx_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matzy_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using matzz_t = Kokkos::View<value_type**, layout_t, mem_space>;
  using backend_t = KokkosBackend<exec_space>;

  RandomMixedCongruentCCQP(unsigned seed = 1) {
    seed_ = seed;

    M_ = matzz_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "M"), NZ, NZ);
    D_ = matzx_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "D"), NZ, NX);
    DT_ = matxz_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "DT"), NX, NZ);
    q_ = vecx_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q"), NX);
    B_ = matzy_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "B"), NZ, NY);
    Kinv_ = matyy_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Kinv"), NY, NY);
    S_ = matyy_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "S"), NY, NY);
    BT_ = matyz_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "BT"), NY, NZ);
    b_ = vecy_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"), NY);
    x_star_ = vecx_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_star"), NX);
    y_star_ = vecy_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "y_star"), NY);

    build();
  }

  std::string name() const {
    return "RandomMixedCongruentCCQP<" + std::to_string(NX) + "," + std::to_string(NY) + "," + std::to_string(NZ) + ">";
  }

  // clang-format off
  auto get_exec_space() const { return exec_space{}; }
  auto get_space_x() const { return LowerBoundSpace<value_type>(0.0); }
  vecx_t get_exact_x() const { return x_star_; }
  vecy_t get_exact_y() const { return y_star_; }
  matxz_t get_DT() const { return DT_; }
  matzz_t get_M() const { return M_; }
  matzx_t get_D() const { return D_; }
  vecx_t get_q() const { return q_; }
  matyz_t get_BT() const { return BT_; }
  matyy_t get_S() const { return S_; }
  matyy_t get_Kinv() const { return Kinv_; }
  matzy_t get_B() const { return B_; }
  vecy_t get_b() const { return b_; }
  // clang-format on

  KOKKOS_INLINE_FUNCTION
  static value_type urand(uint64_t a, uint64_t b) {
    openrand::Philox rng = make_philox(a, b);
    return rng.uniform<double>(-1.0, 1.0);
  }

  static void fill_identity(const matyy_t& mat) {
    Kokkos::deep_copy(mat, 0.0);
    Kokkos::parallel_for(
        "fill_identity_yy", Kokkos::RangePolicy<exec_space>(0, NY), KOKKOS_LAMBDA(const size_t i) { mat(i, i) = 1.0; });
  }

  static void transpose_zy_to_yz(const matzy_t& src, const matyz_t& dst) {
    Kokkos::parallel_for(
        "transpose_zy_to_yz", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NY, NZ}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) { dst(i, j) = src(j, i); });
  }

  static void transpose_zx_to_xz(const matzx_t& src, const matxz_t& dst) {
    Kokkos::parallel_for(
        "transpose_zx_to_xz", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NZ}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) { dst(i, j) = src(j, i); });
  }

  matzz_t gen_spd_zz(value_type diag_boost = 5.0) {
    matzz_t R(Kokkos::view_alloc(Kokkos::WithoutInitializing, "R"), NZ, NZ);
    const auto seed = seed_;
    Kokkos::parallel_for(
        "fill_random_matrix_zz", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NZ, NZ}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          R(i, j) = urand(static_cast<uint64_t>(i + seed + 101), static_cast<uint64_t>(j + 17 * (seed + 101)));
        });

    matzz_t A(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), NZ, NZ);
    KokkosBlas::gemm("T", "N", 1.0, R, R, 0.0, A);
    Kokkos::parallel_for(
        "spd_diag_boost", Kokkos::RangePolicy<exec_space>(0, NZ),
        KOKKOS_LAMBDA(const size_t i) { A(i, i) += diag_boost; });
    return A;
  }

  void make_D_full_rank() {
    const auto seed = seed_;
    auto D = D_;
    auto DT = DT_;
    Kokkos::parallel_for(
        "fill_random_matrix_zx", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NZ, NX}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          D(i, j) = urand(static_cast<uint64_t>(i + seed + 211), static_cast<uint64_t>(j + 11 * (seed + 211)));
        });

    constexpr size_t diag_n = (NZ < NX ? NZ : NX);
    Kokkos::parallel_for(
        "inject_d_diag", Kokkos::RangePolicy<exec_space>(0, diag_n), KOKKOS_LAMBDA(const size_t i) { D(i, i) += 3.0; });

    transpose_zx_to_xz(D, DT);
  }

  void fill_B() {
    const auto seed = seed_;
    auto B = B_;
    auto BT = BT_;
    Kokkos::parallel_for(
        "fill_random_matrix_zy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NZ, NY}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          B(i, j) = 0.2 * urand(static_cast<uint64_t>(i + seed + 307), static_cast<uint64_t>(j + 13 * (seed + 307)));
        });
    transpose_zy_to_yz(B, BT);
  }

  void build() {
    const auto seed = seed_;
    matzz_t R(Kokkos::view_alloc(Kokkos::WithoutInitializing, "R"), NZ, NZ);
    auto Kinv = Kinv_;
    auto x_star = x_star_;
    auto b = b_;
    M_ = gen_spd_zz(/*diag_boost=*/10.0);
    make_D_full_rank();
    fill_B();
    Kokkos::deep_copy(Kinv, 0.0);
    Kokkos::parallel_for(
        "init_kinv_diag", Kokkos::RangePolicy<exec_space>(0, NY), KOKKOS_LAMBDA(const size_t i) {
          Kinv(i, i) = 10.0 + abs(urand(static_cast<uint64_t>(i + seed + 401), static_cast<uint64_t>(911)));
        });

    matzy_t MB(Kokkos::view_alloc(Kokkos::WithoutInitializing, "MB"), NZ, NY);
    matyy_t BTMB(Kokkos::view_alloc(Kokkos::WithoutInitializing, "BTMB"), NY, NY);
    KokkosBlas::gemm("N", "N", 1.0, M_, B_, 0.0, MB);
    KokkosBlas::gemm("N", "N", 1.0, BT_, MB, 0.0, BTMB);
    Kokkos::deep_copy(S_, BTMB);
    KokkosBlas::axpy(1.0, Kinv_, S_);

    matyy_t S_lu(Kokkos::view_alloc(Kokkos::WithoutInitializing, "S_lu"), NY, NY);
    Kokkos::deep_copy(S_lu, S_);
    fill_identity(S_);
    Kokkos::View<int*, layout_t, mem_space> pivots(Kokkos::view_alloc(Kokkos::WithoutInitializing, "pivots"), NY);
    KokkosBlas::gesv(S_lu, S_, pivots);

    vecx_t s_star(Kokkos::view_alloc(Kokkos::WithoutInitializing, "s_star"), NX);
    Kokkos::parallel_for(
        "init_xs", Kokkos::RangePolicy<exec_space>(0, NX), KOKKOS_LAMBDA(const size_t i) {
          openrand::Philox rng = make_philox(static_cast<uint64_t>(i + seed + 503), static_cast<uint64_t>(1337));
          const double u0 = rng.uniform<double>(0.0, 1.0);
          const double u1 = rng.uniform<double>(0.0, 1.0);
          const bool active = u0 < 0.5;
          if (active) {
            x_star(i) = 0.1 + 0.9 * u1;
            s_star(i) = 0.0;
          } else {
            x_star(i) = 0.0;
            s_star(i) = 0.1 + 0.9 * u1;
          }
        });

    Kokkos::parallel_for(
        "init_b", Kokkos::RangePolicy<exec_space>(0, NY), KOKKOS_LAMBDA(const size_t i) {
          b(i) = 0.3 * urand(static_cast<uint64_t>(i + seed + 607), static_cast<uint64_t>(2027));
        });

    matzx_t MD(Kokkos::view_alloc(Kokkos::WithoutInitializing, "MD"), NZ, NX);
    matyx_t BTC(Kokkos::view_alloc(Kokkos::WithoutInitializing, "BTC"), NY, NX);
    matyx_t T(Kokkos::view_alloc(Kokkos::WithoutInitializing, "T"), NY, NX);
    KokkosBlas::gemm("N", "N", 1.0, M_, D_, 0.0, MD);
    KokkosBlas::gemm("N", "N", 1.0, BT_, MD, 0.0, BTC);
    KokkosBlas::gemm("N", "N", 1.0, S_, BTC, 0.0, T);

    matxx_t DtMD(Kokkos::view_alloc(Kokkos::WithoutInitializing, "DtMD"), NX, NX);
    matzx_t MB2(Kokkos::view_alloc(Kokkos::WithoutInitializing, "MB2"), NZ, NY);
    matxy_t DtMB(Kokkos::view_alloc(Kokkos::WithoutInitializing, "DtMB"), NX, NY);
    matxx_t Htmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Htmp"), NX, NX);
    KokkosBlas::gemm("N", "N", 1.0, M_, D_, 0.0, MD);
    KokkosBlas::gemm("N", "N", 1.0, DT_, MD, 0.0, DtMD);
    KokkosBlas::gemm("N", "N", 1.0, M_, B_, 0.0, MB2);
    KokkosBlas::gemm("N", "N", 1.0, DT_, MB2, 0.0, DtMB);
    KokkosBlas::gemm("N", "N", 1.0, DtMB, T, 0.0, Htmp);

    vecx_t Hx(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Hx"), NX);
    KokkosBlas::gemv("N", 1.0, DtMD, x_star_, 0.0, Hx);
    KokkosBlas::gemv("N", -1.0, Htmp, x_star_, 1.0, Hx);
    Kokkos::deep_copy(q_, s_star);
    KokkosBlas::axpy(-1.0, Hx, q_);

    vecy_t u(Kokkos::view_alloc(Kokkos::WithoutInitializing, "u"), NY);
    vecx_t DtMBu(Kokkos::view_alloc(Kokkos::WithoutInitializing, "DtMBu"), NX);
    KokkosBlas::gemv("N", 1.0, S_, b_, 0.0, u);
    KokkosBlas::gemv("N", 1.0, DtMB, u, 0.0, DtMBu);
    KokkosBlas::axpy(1.0, DtMBu, q_);

    vecy_t rhs(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rhs"), NY);
    Kokkos::deep_copy(rhs, b_);
    KokkosBlas::gemv("N", 1.0, BTC, x_star_, 1.0, rhs);
    KokkosBlas::gemv("N", -1.0, S_, rhs, 0.0, y_star_);
  }

 private:
  unsigned seed_ = 1;
  matzz_t M_;
  matzx_t D_;
  matyy_t Kinv_;
  matxz_t DT_;
  vecx_t q_;
  matzy_t B_;
  matyy_t S_;
  matyz_t BT_;
  vecy_t b_;
  vecx_t x_star_;
  vecy_t y_star_;
};

}  // namespace mixed

/// \brief Problem with each of its matrices given as the sparse matrix holding the same entries.
template <class Problem>
struct SparseMatrices : Problem {
  using sparse_t =
      KokkosSparse::CrsMatrix<typename Problem::value_type, int,
                              Kokkos::Device<typename Problem::exec_space, typename Problem::exec_space::memory_space>,
                              void, size_t>;

  std::string name() const {
    return "SparseMatrices<" + Problem::name() + ">";
  }

  // clang-format off
  sparse_t get_A()    const requires requires(const Problem& p) { p.get_A(); }    { return sparse(Problem::get_A()); }
  sparse_t get_D()    const requires requires(const Problem& p) { p.get_D(); }    { return sparse(Problem::get_D()); }
  sparse_t get_M()    const requires requires(const Problem& p) { p.get_M(); }    { return sparse(Problem::get_M()); }
  sparse_t get_DT()   const requires requires(const Problem& p) { p.get_DT(); }   { return sparse(Problem::get_DT()); }
  sparse_t get_B()    const requires requires(const Problem& p) { p.get_B(); }    { return sparse(Problem::get_B()); }
  sparse_t get_S()    const requires requires(const Problem& p) { p.get_S(); }    { return sparse(Problem::get_S()); }
  sparse_t get_BT()   const requires requires(const Problem& p) { p.get_BT(); }   { return sparse(Problem::get_BT()); }
  sparse_t get_Kinv() const requires requires(const Problem& p) { p.get_Kinv(); } { return sparse(Problem::get_Kinv()); }
  // clang-format on

 private:
  template <class DenseMatrix>
  static sparse_t sparse(const DenseMatrix& dense) {
    return make_sparse_matrix<sparse_t>(dense);
  }
};

/// \brief problem with its matrices sparse.
template <class Problem>
SparseMatrices<Problem> as_sparse(const Problem& problem) {
  return SparseMatrices<Problem>{problem};
}

}  // namespace kokkos_backend
//@}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

/// \brief PGD from make_pgd_solution_strategy's defaults; its steps need no direction vector, so p goes unused.
struct DefaultPGD {
  KOKKOS_INLINE_FUNCTION auto operator()(const PGDConfig<double>& cfg, auto& /*p*/) const {
    return make_pgd_solution_strategy(cfg);
  }
};

/// \brief PGD with Barzilai-Borwein steps and the given residual; the steps need no direction vector, so p goes unused.
template <class Residual = LinfNormProjectedDiffResidual>
struct BBPGD {
  KOKKOS_INLINE_FUNCTION auto operator()(const PGDConfig<double>& cfg, auto& /*p*/) const {
    return make_pgd_solution_strategy(BBStepStrategy{}, Residual{}, cfg);
  }
};

/// \brief PGD with Barzilai-Borwein steps under the nonmonotone line search and the given residual; p goes unused.
template <class Residual = LinfNormProjectedDiffResidual>
struct NonmonotoneBBPGD {
  KOKKOS_INLINE_FUNCTION auto operator()(const PGDConfig<double>& cfg, auto& /*p*/) const {
    return make_pgd_solution_strategy(NonmonotoneBBStepStrategy<>{}, Residual{}, cfg);
  }
};

/// \brief PGD with MPRGP steps, carrying their direction in p, and the given residual.
template <class Residual = LinfNormProjectedDiffResidual>
struct MPRGPPGD {
  KOKKOS_INLINE_FUNCTION auto operator()(const PGDConfig<double>& cfg, auto& p) const {
    return make_pgd_solution_strategy(make_mprgp_step_strategy(p), Residual{}, cfg);
  }
};

/// \brief Whether a run is expected to converge to x* or to fail to converge.
enum class Expect { converges, fails };

template <class Test, class MakePGD>
void run_mundy_math_test(const Test& test, const MakePGD& make_pgd, Expect expect = Expect::converges) {
  std::cout << "Running test: " << test.name() << std::endl;

  // Problem setup
  auto A = test.get_A();
  auto q = test.get_q();
  auto space = test.get_space();
  auto x_exact = test.get_exact_solution();

  using vector_t = decltype(x_exact);
  vector_t x{}, grad{}, x_tmp{}, grad_tmp{}, p{};

  // A bad initial guess, to force more iterations, in the space
  x.fill(99.99);
  MundyMathBackend::assign("start", x, projection(MundyMathBackend{}, space, x));

  // Build the problem
  const auto cqpp = make_cqpp<MundyMathBackend>(A, q, space);

  // Strategy + state
  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd(cfg, p);
  auto pgd_state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results
  EXPECT_EQ(result.converged, expect == Expect::converges);
  if (expect == Expect::converges) {
    for (size_t i = 0; i < vector_t::size; ++i) {
      EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
    }
  }
}

template <class Test, class MakePGD>
void run_mundy_math_congruent_test(const Test& test, const MakePGD& make_pgd, Expect expect = Expect::converges) {
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

  // Solving via explicit quadratic form
  {
    using vector_t = decltype(x_exact);
    vector_t x{}, grad{}, x_tmp{}, grad_tmp{}, p{};
    vector_t f{}, u{};

    // A bad initial guess, to force more iterations, in the space
    x.fill(99.99);
    MundyMathBackend::assign("start", x, projection(MundyMathBackend{}, space, x));

    // Build quadratic form operator + user-owned workspace, then the problem
    const auto A = make_quadratic_form<MundyMathBackend>(DT, M, D);
    auto workspace = A.make_workspace(f, u);  // intermediate variables f = D x, u = M f
    const auto cqpp = make_cqpp<MundyMathBackend>(A, q, space, workspace);

    // Strategy + state
    PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
    auto pgd = make_pgd(cfg, p);
    auto pgd_state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);

    // Solve (can reuse "cqpp" and "pgd" across many states)
    auto result = solve_cqpp(cqpp, pgd, pgd_state);

    // Check results; f and u are those of an apply at the returned x
    EXPECT_EQ(result.converged, expect == Expect::converges);
    vector_t f_at_x{}, u_at_x{}, A_x{};
    auto at_x = A.make_workspace(f_at_x, u_at_x);
    MundyMathBackend::apply(A, x, A_x, at_x);
    for (size_t i = 0; i < vector_t::size; ++i) {
      EXPECT_EQ(f[i], f_at_x[i]);
      EXPECT_EQ(u[i], u_at_x[i]);
    }
    if (expect == Expect::converges) {
      for (size_t i = 0; i < vector_t::size; ++i) {
        EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
        EXPECT_NEAR(f[i], f_exact[i], 10 * cfg.tol);
        EXPECT_NEAR(u[i], u_exact[i], 10 * cfg.tol);
      }
    }
  }

  // Solving via quadratic form helper
  {
    using vector_t = decltype(x_exact);
    vector_t x{}, grad{}, x_tmp{}, grad_tmp{}, p{};
    vector_t f{}, u{};

    // A bad initial guess, to force more iterations, in the space
    x.fill(99.99);
    MundyMathBackend::assign("start", x, projection(MundyMathBackend{}, space, x));

    // Build the cqpp directly
    // Use f = D x and u = M f as intermediate variables to avoid redundant computations in the PGD iterations
    // On return, f and u are those of an apply at the returned x.
    const auto cqpp = make_cqpp<MundyMathBackend>(DT, M, D, q, f, u, space);

    // Strategy + state
    PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
    auto pgd = make_pgd(cfg, p);
    auto pgd_state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);

    // Solve (can reuse "cqpp" and "pgd" across many states)
    auto result = solve_cqpp(cqpp, pgd, pgd_state);

    // Check results; f and u are those of an apply at the returned x
    EXPECT_EQ(result.converged, expect == Expect::converges);
    const auto A = make_quadratic_form<MundyMathBackend>(DT, M, D);
    vector_t f_at_x{}, u_at_x{}, A_x{};
    auto at_x = A.make_workspace(f_at_x, u_at_x);
    MundyMathBackend::apply(A, x, A_x, at_x);
    for (size_t i = 0; i < vector_t::size; ++i) {
      EXPECT_EQ(f[i], f_at_x[i]);
      EXPECT_EQ(u[i], u_at_x[i]);
    }
    if (expect == Expect::converges) {
      for (size_t i = 0; i < vector_t::size; ++i) {
        EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
        EXPECT_NEAR(f[i], f_exact[i], 10 * cfg.tol);
        EXPECT_NEAR(u[i], u_exact[i], 10 * cfg.tol);
      }
    }
  }
}

template <class Test, class MakePGD>
void run_mundy_math_mixed_congruent_test(const Test& test, const MakePGD& make_pgd) {
  std::cout << "Running test: " << test.name() << std::endl;

  // Problem setup
  auto DT = test.get_DT();
  auto M = test.get_M();
  auto D = test.get_D();
  auto q = test.get_q();
  auto B = test.get_B();
  auto S = test.get_S();
  auto BT = test.get_BT();
  auto b = test.get_b();
  auto space = test.get_space_x();
  auto x_exact = test.get_exact_x();
  auto y_exact = test.get_exact_y();

  using vector_t = decltype(x_exact);
  vector_t x{}, grad{}, x_tmp{}, grad_tmp{}, p{};

  x.fill(99.99);  // use a bad initial guess to force more iterations

  // Double check sizes:
  ASSERT_EQ(M.num_rows, M.num_cols) << "M should be square";
  ASSERT_EQ(DT.num_rows, D.num_cols) << "DT and D are supposed to be transposes of each other";
  ASSERT_EQ(DT.num_cols, D.num_rows) << "DT and D are supposed to be transposes of each other";
  ASSERT_EQ(DT.num_cols, M.num_rows) << "DT * M should be well-defined";
  ASSERT_EQ(M.num_cols, D.num_rows) << "M * D should be well-defined";
  ASSERT_EQ(M.num_cols, B.num_rows) << "M * B should be well-defined";
  ASSERT_EQ(BT.num_cols, M.num_rows) << "B^T * M should be well-defined";

  ASSERT_EQ(D.num_cols, x_exact.size) << "D * x should be well-defined";
  ASSERT_EQ(B.num_cols, y_exact.size) << "B * y should be well-defined";
  ASSERT_EQ(S.num_cols, BT.num_rows) << "S * BT should be well-defined";
  ASSERT_EQ(B.num_cols, S.num_rows) << "B * S should be well-defined";
  ASSERT_EQ(q.size, x_exact.size) << "q should be same size as x_exact";
  ASSERT_EQ(b.size, y_exact.size) << "b should be same size as y_exact";

  // Build the problem
  const auto mixed_cqpp = make_mixed_cqpp<MundyMathBackend>(DT, M, D, q, B, S, BT, b, space);

  auto DT_op = mixed_cqpp.DT();
  auto M_op = mixed_cqpp.M();
  auto f_b = mixed_cqpp.f_b();

  ASSERT_EQ(MundyMathBackend::domain_size(M_op), MundyMathBackend::size(f_b))
      << "M and f_b should be compatible for multiplication";
  ASSERT_EQ(MundyMathBackend::domain_size(DT_op), MundyMathBackend::range_size(M_op))
      << "DT and M should be compatible for DT * M";

  // Strategy + state
  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd(cfg, p);
  auto pgd_state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);

  // Solve
  auto result = solve_mixed_cqpp(mixed_cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  for (size_t i = 0; i < vector_t::size; ++i) {
    EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
  }
}

// The full mixed-CQPP chain (make_mixed_cqpp through solve_mixed_cqpp) constructed and solved inside a kernel.
template <class Test, class MakePGD>
void run_mundy_math_mixed_congruent_test_in_kernel(const Test& test, const MakePGD& make_pgd) {
  constexpr double tol = 1e-6;
  Kokkos::View<double, Kokkos::DefaultExecutionSpace::memory_space> max_error("max_error");
  Kokkos::View<bool, Kokkos::DefaultExecutionSpace::memory_space> converged("converged");
  Kokkos::parallel_for(
      "mixed_cqpp_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int) {
        const auto mixed_cqpp =
            make_mixed_cqpp<MundyMathBackend>(test.get_DT(), test.get_M(), test.get_D(), test.get_q(), test.get_B(),
                                              test.get_S(), test.get_BT(), test.get_b(), test.get_space_x());
        const auto x_exact = test.get_exact_x();
        auto x = x_exact;
        auto grad = x_exact;
        auto x_tmp = x_exact;
        auto grad_tmp = x_exact;
        auto p = x_exact;
        for (size_t i = 0; i < x.size; ++i) {
          x[i] = 99.99;
        }
        auto pgd = make_pgd(PGDConfig<double>{.max_iters = 1000, .tol = tol}, p);
        auto state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
        const auto result = solve_mixed_cqpp(mixed_cqpp, pgd, state);

        double error = 0.0;
        for (size_t i = 0; i < x.size; ++i) {
          error = max(error, abs(x[i] - x_exact[i]));
        }
        max_error() = error;
        converged() = result.converged;
      });
  EXPECT_TRUE(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, converged)()) << test.name();
  EXPECT_LE(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, max_error)(), 10 * tol) << test.name();
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
template <class Test, class MakePGD>
void run_kokkos_test(const Test& test, const MakePGD& make_pgd, Expect expect = Expect::converges) {
  std::cout << "Running test: " << test.name() << std::endl;

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
  vector_t p(Kokkos::view_alloc(Kokkos::WithoutInitializing, "p"), size);

  // A bad initial guess, to force more iterations, in the space
  Kokkos::deep_copy(x, 99.99);
  KokkosBackend<decltype(exec_space)>::assign("start", x, projection(KokkosBackend<decltype(exec_space)>{}, space, x));

  // Build the problem
  const auto cqpp = make_cqpp<KokkosBackend<decltype(exec_space)>>(A, q, space);

  // Strategy + state
  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd(cfg, p);
  auto pgd_state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results
  EXPECT_EQ(result.converged, expect == Expect::converges);
  if (expect == Expect::converges) {
    const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
    const auto x_exact_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_exact);
    for (size_t i = 0; i < size; ++i) {
      EXPECT_NEAR(x_host[i], x_exact_host[i], 10 * cfg.tol);
    }
  }
}

template <class Test, class MakePGD>
void run_kokkos_congruent_test(const Test& test, const MakePGD& make_pgd, Expect expect = Expect::converges) {
  std::cout << "Running test: " << test.name() << std::endl;

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
  vector_t p(Kokkos::view_alloc(Kokkos::WithoutInitializing, "p"), size);

  // A bad initial guess, to force more iterations, in the space
  Kokkos::deep_copy(x, 99.99);
  KokkosBackend<decltype(exec_space)>::assign("start", x, projection(KokkosBackend<decltype(exec_space)>{}, space, x));

  // Build quadratic form operator + user-owned workspace, then the problem
  const auto A = make_quadratic_form<KokkosBackend<decltype(exec_space)>>(DT, M, D);
  auto workspace = A.make_workspace(f, u);  // intermediate variables f = D x, u = M f
  const auto cqpp = make_cqpp<KokkosBackend<decltype(exec_space)>>(A, q, space, workspace);

  // Strategy + state
  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd(cfg, p);
  auto pgd_state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);

  // Solve (can reuse "cqpp" and "pgd" across many states)
  auto result = solve_cqpp(cqpp, pgd, pgd_state);

  // Check results; f and u are those of an apply at the returned x
  EXPECT_EQ(result.converged, expect == Expect::converges);
  vector_t f_at_x("f_at_x", size), u_at_x("u_at_x", size), A_x("A_x", size);
  auto at_x = A.make_workspace(f_at_x, u_at_x);
  KokkosBackend<decltype(exec_space)>::apply(A, x, A_x, at_x);
  {
    const auto f_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);
    const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
    const auto f_at_x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f_at_x);
    const auto u_at_x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_at_x);
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(f_host[i], f_at_x_host[i]);
      EXPECT_EQ(u_host[i], u_at_x_host[i]);
    }
  }
  if (expect == Expect::converges) {
    const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
    const auto f_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);
    const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
    const auto x_exact_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_exact);
    const auto f_exact_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f_exact);
    const auto u_exact_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_exact);
    for (size_t i = 0; i < size; ++i) {
      EXPECT_NEAR(x_host[i], x_exact_host[i], 10 * cfg.tol);
      EXPECT_NEAR(f_host[i], f_exact_host[i], 10 * cfg.tol);
      EXPECT_NEAR(u_host[i], u_exact_host[i], 10 * cfg.tol);
    }
  }
}

template <class Test, class MakePGD>
void run_kokkos_mixed_congruent_test(const Test& test, const MakePGD& make_pgd) {
  std::cout << "Running test: " << test.name() << std::endl;

  auto exec_space = test.get_exec_space();
  using backend_t = KokkosBackend<decltype(exec_space)>;

  // Problem setup
  auto DT = test.get_DT();
  auto M = test.get_M();
  auto D = test.get_D();
  auto q = test.get_q();
  auto B = test.get_B();
  auto S = test.get_S();
  auto BT = test.get_BT();
  auto b = test.get_b();
  auto space = test.get_space_x();
  auto x_exact = test.get_exact_x();
  auto y_exact = test.get_exact_y();

  // Double check sizes:
  ASSERT_EQ(backend_t::range_size(M), backend_t::domain_size(M)) << "M should be square";
  ASSERT_EQ(backend_t::range_size(DT), backend_t::domain_size(D)) << "DT and D are supposed to be transposes";
  ASSERT_EQ(backend_t::domain_size(DT), backend_t::range_size(D)) << "DT and D are supposed to be transposes";
  ASSERT_EQ(backend_t::domain_size(DT), backend_t::range_size(M)) << "DT * M should be well-defined";
  ASSERT_EQ(backend_t::domain_size(M), backend_t::range_size(D)) << "M * D should be well-defined";
  ASSERT_EQ(backend_t::domain_size(M), backend_t::range_size(B)) << "M * B should be well-defined";
  ASSERT_EQ(backend_t::domain_size(BT), backend_t::range_size(M)) << "B^T * M should be well-defined";

  ASSERT_EQ(backend_t::domain_size(D), x_exact.extent(0)) << "D * x should be well-defined";
  ASSERT_EQ(backend_t::domain_size(B), y_exact.extent(0)) << "B * y should be well-defined";
  ASSERT_EQ(backend_t::domain_size(S), backend_t::range_size(BT)) << "S * BT should be well-defined";
  ASSERT_EQ(backend_t::domain_size(B), backend_t::range_size(S)) << "B * S should be well-defined";
  ASSERT_EQ(q.extent(0), x_exact.extent(0)) << "q should be same size as x_exact";
  ASSERT_EQ(b.extent(0), y_exact.extent(0)) << "b should be same size as y_exact";

  // Build the problem
  const auto mixed_cqpp = make_mixed_cqpp<KokkosBackend<decltype(exec_space)>>(DT, M, D, q, B, S, BT, b, space);

  auto DT_op = mixed_cqpp.DT();
  auto M_op = mixed_cqpp.M();
  auto f_b = mixed_cqpp.f_b();

  ASSERT_EQ(backend_t::domain_size(M_op), backend_t::size(f_b)) << "M and f_b should be compatible for multiplication";
  ASSERT_EQ(backend_t::domain_size(DT_op), backend_t::range_size(M_op)) << "DT and M should be compatible for DT * M";

  // Strategy + state
  using vector_t = decltype(x_exact);
  const size_t size = x_exact.extent(0);
  vector_t x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), size);
  vector_t grad(Kokkos::view_alloc(Kokkos::WithoutInitializing, "grad"), size);
  vector_t x_tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x_tmp"), size);
  vector_t grad_tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "grad_tmp"), size);
  vector_t p(Kokkos::view_alloc(Kokkos::WithoutInitializing, "p"), size);
  Kokkos::deep_copy(x, 99.99);  // use a bad initial guess to force more iterations

  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
  auto pgd = make_pgd(cfg, p);
  auto pgd_state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);

  // Solve
  auto result = solve_mixed_cqpp(mixed_cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  const auto x_exact_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_exact);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(x_host(i), x_exact_host(i), 10 * cfg.tol);
  }
}

/// \brief The Kokkos allocations one sparse apply makes on ExecSpace.
///
/// On CUDA its team launch takes a team-scratch slot, an allocation even when the kernel asks for no scratch; host
/// launches make none.
template <class ExecSpace>
constexpr size_t allocations_per_sparse_apply() {
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
    return 1;
  }
#endif
  return 0;
}

/// \brief The Kokkos allocations made while some code runs, and its sparse applies (KokkosSparse::spmv launches).
struct AllocationCount {
  size_t allocations = 0;
  size_t sparse_applies = 0;
};

/// \brief The Kokkos allocations made, and sparse applies launched, while f runs.
template <class F>
AllocationCount count_allocations(F&& f) {
  static AllocationCount count;
  count = AllocationCount{};
  Kokkos::Tools::Experimental::set_init_callback(
      [](const int, const uint64_t, const uint32_t, Kokkos_Profiling_KokkosPDeviceInfo*) {});
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](const Kokkos_Profiling_SpaceHandle, const char*, const void*, const uint64_t) { ++count.allocations; });
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback([](const char* name, const uint32_t, uint64_t*) {
    count.sparse_applies += std::string_view(name).starts_with("KokkosSparse::spmv");
  });
  f();
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_init_callback(nullptr);
  return count;
}

// A mixed CQPP formed in a reused workspace solves exactly as one formed in fresh storage, and forming and solving it
// through that workspace allocates nothing but the allocations its sparse applies make.
template <class Test, class MakePGD>
void run_kokkos_mixed_congruent_workspace_test(const Test& test, const MakePGD& make_pgd) {
  using exec_space = decltype(test.get_exec_space());
  using backend_t = KokkosBackend<exec_space>;
  const auto DT = test.get_DT();
  const auto M = test.get_M();
  const auto D = test.get_D();
  const auto q = test.get_q();
  const auto B = test.get_B();
  const auto S = test.get_S();
  const auto BT = test.get_BT();
  const auto b = test.get_b();
  const auto space = test.get_space_x();
  using vector_t = decltype(test.get_exact_x());
  const size_t size = test.get_exact_x().extent(0);

  vector_t grad("grad", size), x_tmp("x_tmp", size), grad_tmp("grad_tmp", size), p("p", size);
  const auto pgd = make_pgd(PGDConfig<double>{.max_iters = 1000, .tol = 1e-6}, p);

  // Fresh storage
  vector_t x_fresh("x_fresh", size);
  Kokkos::deep_copy(x_fresh, 99.99);
  auto fresh_state = make_pgd_state(pgd, x_fresh, grad, x_tmp, grad_tmp);
  const auto fresh = solve_mixed_cqpp(make_mixed_cqpp<backend_t>(DT, M, D, q, B, S, BT, b, space), pgd, fresh_state);
  ASSERT_TRUE(fresh.converged) << test.name();
  const auto x_fresh_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_fresh);

  // One workspace, twice
  auto workspace = make_mixed_cqpp_workspace<backend_t>(DT, M, D, q, B, S, BT);
  vector_t x_reused("x_reused", size);
  for (int pass = 0; pass < 2; ++pass) {
    Kokkos::deep_copy(x_reused, 99.99);
    auto state = make_pgd_state(pgd, x_reused, grad, x_tmp, grad_tmp);
    PGDResult<double> result;
    const AllocationCount counted = count_allocations([&] {
      result = solve_mixed_cqpp(make_mixed_cqpp<backend_t>(DT, M, D, q, B, S, BT, b, space, workspace), pgd, state);
    });
    const auto x_reused_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_reused);
    EXPECT_EQ(result.num_iters, fresh.num_iters) << test.name() << " pass " << pass;
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(x_reused_host(i), x_fresh_host(i)) << test.name() << " pass " << pass << " entry " << i;
    }
    EXPECT_EQ(counted.allocations, allocations_per_sparse_apply<exec_space>() * counted.sparse_applies)
        << test.name() << " pass " << pass;
  }

  // l_workspace.u is that of an apply at the returned x
  auto at_x = make_mixed_cqpp_workspace<backend_t>(DT, M, D, q, B, S, BT);
  const auto reduced = to_cqpp(make_mixed_cqpp<backend_t>(DT, M, D, q, B, S, BT, b, space, at_x));
  vector_t H_x("H_x", size);
  backend_t::apply(reduced.A(), x_reused, H_x, reduced.workspace());
  const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, workspace.l_workspace.u());
  const auto u_at_x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, at_x.l_workspace.u());
  for (size_t i = 0; i < u_host.extent(0); ++i) {
    EXPECT_EQ(u_host(i), u_at_x_host(i)) << test.name() << " entry " << i;
  }
}

/// \brief op, whose new vectors start NaN-filled, as uninitialized storage may.
template <class Op>
struct NaNVectorsOp {
  Op op;

  size_t domain_size() const {
    return op.domain_size();
  }
  size_t range_size() const {
    return op.range_size();
  }
  auto make_domain_vector() const {
    auto v = op.make_domain_vector();
    Kokkos::deep_copy(v, std::numeric_limits<double>::quiet_NaN());
    return v;
  }
  auto make_range_vector() const {
    auto v = op.make_range_vector();
    Kokkos::deep_copy(v, std::numeric_limits<double>::quiet_NaN());
    return v;
  }
  auto make_workspace() const {
    return op.make_workspace();
  }
  template <class XVector, class YVector, class Workspace>
  void apply(const XVector& x, YVector& y, Workspace& workspace) const {
    op.apply(x, y, workspace);
  }
  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    op.apply(x, y);
  }
};

// A mixed CQPP whose applies of S, for S b and within L = B S B^T, share one given workspace of S solves exactly as one
// whose workspace holds its own, and both run in the given storage. S is CG on B^T M B + K^-1, so it has scratch of its
// own: CG's vectors and the quadratic form's. Its vectors start NaN-filled, so either workspace must zero S's guesses.
template <class Test, class MakePGD>
void run_kokkos_mixed_congruent_shared_s_workspace_test(const Test& test, const MakePGD& make_pgd) {
  using backend_t = KokkosBackend<decltype(test.get_exec_space())>;
  const auto DT = test.get_DT();
  const auto M = test.get_M();
  const auto D = test.get_D();
  const auto q = test.get_q();
  const auto B = test.get_B();
  const auto S = make_cg_inv_op<backend_t>(
      NaNVectorsOp{make_sum_op<backend_t>(make_quadratic_form<backend_t>(test.get_BT(), test.get_M(), test.get_B()),
                                          test.get_Kinv())},
      CGConfig<double>{});
  const auto BT = test.get_BT();
  const auto b = test.get_b();
  const auto space = test.get_space_x();
  using vector_t = decltype(test.get_exact_x());
  const size_t size = test.get_exact_x().extent(0);
  vector_t grad("grad", size), x_tmp("x_tmp", size), grad_tmp("grad_tmp", size), p("p", size);
  const auto pgd = make_pgd(PGDConfig<double>{.max_iters = 1000, .tol = 1e-6}, p);

  // Solve
  auto own = make_mixed_cqpp_workspace<backend_t>(DT, M, D, q, B, S, BT);
  vector_t x_own("x_own", size);
  auto own_state = make_pgd_state(pgd, x_own, grad, x_tmp, grad_tmp);
  const auto own_result =
      solve_mixed_cqpp(make_mixed_cqpp<backend_t>(DT, M, D, q, B, S, BT, b, space, own), pgd, own_state);
  const auto s_workspace = S.make_workspace();
  auto shared = make_mixed_cqpp_workspace<backend_t>(DT, M, D, q, B, S, BT, s_workspace);
  vector_t x_shared("x_shared", size);
  auto shared_state = make_pgd_state(pgd, x_shared, grad, x_tmp, grad_tmp);
  const auto shared_result =
      solve_mixed_cqpp(make_mixed_cqpp<backend_t>(DT, M, D, q, B, S, BT, b, space, shared), pgd, shared_state);
  ASSERT_TRUE(own_result.converged) << test.name();

  // Bit for bit
  EXPECT_EQ(shared_result.num_iters, own_result.num_iters) << test.name();
  const auto x_own_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_own);
  const auto x_shared_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_shared);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(x_shared_host(i), x_own_host(i)) << test.name() << " entry " << i;
  }

  // In the given storage
  EXPECT_EQ(shared.s_workspace.r().data(), s_workspace.r().data()) << test.name();
  EXPECT_EQ(shared.l_workspace.m_workspace().r().data(), s_workspace.r().data()) << test.name();
  EXPECT_NE(own.s_workspace.r().data(), own.l_workspace.m_workspace().r().data()) << test.name();

  // l_workspace.u is S w, w = B^T M D x, to within the inner solve: u - S w = S r with ||r||_2 <= cg_tol, so
  // ||u - S w||_2 <= ||S||_F cg_tol
  const auto S_exact = test.get_S();
  const size_t num_y = backend_t::range_size(BT);
  vector_t dx("dx", backend_t::range_size(D)), mdx("mdx", backend_t::range_size(M)), w("w", num_y);
  vector_t S_w("S_w", num_y), e("e", num_y), S_e("S_e", num_y);
  backend_t::apply(D, x_shared, dx);
  backend_t::apply(M, dx, mdx);
  backend_t::apply(BT, mdx, w);
  backend_t::apply(S_exact, w, S_w);
  double S_frobenius_squared = 0.0;
  for (size_t j = 0; j < num_y; ++j) {
    Kokkos::deep_copy(e, 0.0);
    Kokkos::deep_copy(Kokkos::subview(e, j), 1.0);
    backend_t::apply(S_exact, e, S_e);
    S_frobenius_squared += dot(backend_t{}, S_e, S_e);
  }
  const auto u_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, shared.l_workspace.u());
  const auto S_w_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, S_w);
  double error_squared = 0.0;
  for (size_t i = 0; i < num_y; ++i) {
    error_squared += (u_host(i) - S_w_host(i)) * (u_host(i) - S_w_host(i));
  }
  EXPECT_LE(std::sqrt(error_squared), std::sqrt(S_frobenius_squared) * CGConfig<double>{}.tol) << test.name();
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

/// \brief num_iters counts every step make_pgd's strategy takes, on a random LCP: a solve capped at the steps an
/// uncapped one took still converges, one capped a step earlier does not, and one started at x* takes none.
///
/// rand() is seeded here, so the LCP does not depend on the tests run before.
template <class MakePGD>
void expect_iterations_count_every_step(const MakePGD& make_pgd) {
  std::srand(7);
  const math_backend::RandomLCP<7> test;
  using vector_t = decltype(test.get_exact_solution());
  const auto cqpp = make_cqpp<MundyMathBackend>(test.get_A(), test.get_q(), test.get_space());
  const auto solve = [&](unsigned max_iters, const vector_t& start) {
    vector_t x = start, grad{}, x_tmp{}, grad_tmp{}, p{};
    const auto pgd = make_pgd(PGDConfig<double>{.max_iters = max_iters, .tol = 1e-12}, p);
    auto state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
    return solve_cqpp(cqpp, pgd, state);
  };
  vector_t start;
  start.fill(99.99);

  const PGDResult<double> uncapped = solve(1000, start);
  ASSERT_TRUE(uncapped.converged) << uncapped;
  ASSERT_GT(uncapped.num_iters, 1u);

  const PGDResult<double> capped = solve(uncapped.num_iters, start);
  EXPECT_TRUE(capped.converged) << capped;
  EXPECT_EQ(capped.num_iters, uncapped.num_iters);

  const PGDResult<double> a_step_short = solve(uncapped.num_iters - 1, start);
  EXPECT_FALSE(a_step_short.converged) << a_step_short;
  EXPECT_EQ(a_step_short.num_iters, uncapped.num_iters - 1);

  const PGDResult<double> at_solution = solve(1000, test.get_exact_solution());
  EXPECT_TRUE(at_solution.converged) << at_solution;
  EXPECT_EQ(at_solution.num_iters, 0u);
}

/// \brief Every solve make_pgd's strategy ends, capped at any k from 0 to the steps an uncapped one takes, leaves f and
/// u those of an apply at the returned x and reports the projected-difference residual of A x + q formed by that apply.
template <class MakePGD>
void expect_solves_end_on_an_apply_at_their_iterate(const MakePGD& make_pgd) {
  const math_backend::congruent::CongruentLCPWrapper test{math_backend::PeriodicLaplacianLCP<8>{}};
  using vector_t = decltype(test.get_x_exact());
  const auto q = test.get_q();
  const auto space = test.get_space();
  const auto A = make_quadratic_form<MundyMathBackend>(test.get_DT(), test.get_M(), test.get_D());
  const auto solve = [&](unsigned max_iters, vector_t& x, vector_t& f, vector_t& u) {
    x.fill(99.99);
    vector_t grad{}, x_tmp{}, grad_tmp{}, p{};
    const auto cqpp = make_cqpp<MundyMathBackend>(A, q, space, A.make_workspace(f, u));
    const auto pgd = make_pgd(PGDConfig<double>{.max_iters = max_iters, .tol = 1e-6}, p);
    auto state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
    return solve_cqpp(cqpp, pgd, state);
  };

  vector_t x{}, f{}, u{};
  const PGDResult<double> uncapped = solve(1000, x, f, u);
  for (unsigned k = 0; k <= uncapped.num_iters; ++k) {
    const PGDResult<double> result = solve(k, x, f, u);
    vector_t f_at_x{}, u_at_x{}, grad_at_x{};
    auto at_x = A.make_workspace(f_at_x, u_at_x);
    MundyMathBackend::apply(A, x, grad_at_x, at_x);
    axpby(MundyMathBackend{}, 1.0, q, 1.0, grad_at_x);
    ASSERT_EQ(result.residual, LinfNormProjectedDiffResidual{}(MundyMathBackend{}, x, grad_at_x, space)) << "k = " << k;
    for (size_t i = 0; i < vector_t::size; ++i) {
      ASSERT_EQ(f[i], f_at_x[i]) << "k = " << k << ", i = " << i;
      ASSERT_EQ(u[i], u_at_x[i]) << "k = " << k << ", i = " << i;
    }
  }
}

//! \name MPRGP runs
//@{

/// \brief u = eps / 2, the unit roundoff.
constexpr double unit_roundoff = std::numeric_limits<double>::epsilon() / 2;

/// \brief gamma_m = m u / (1 - m u), which bounds the relative rounding of an m-term sum or dot product (Higham,
/// Accuracy and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1).
constexpr double rounding_gamma(double m) {
  return m * unit_roundoff / (1.0 - m * unit_roundoff);
}

/// \brief An MPRGP solve: its expansion step once initialized, which is its longest, and at its end, its result, and x.
template <class XVector>
struct MPRGPRun {
  double first_expansion_step = 0.0;
  double last_expansion_step = 0.0;
  PGDResult<double> result;
  XVector x;
};

/// \brief The steps make_steps(p) on test's problem from start, a value or a vector, projected onto its space, with the
/// projected-gradient residual to tol. grad, x_tmp, grad_tmp, and p start NaN-filled, so the solve reads none before
/// writing it.
template <class Test, class Start, class MakeSteps>
auto run_mundy_math_mprgp(const Test& test, const Start& start, unsigned max_iters, double tol,
                          const MakeSteps& make_steps) {
  using vector_t = decltype(test.get_exact_solution());
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  MPRGPRun<vector_t> run;
  vector_t grad, x_tmp, grad_tmp, p;
  grad.fill(nan);
  x_tmp.fill(nan);
  grad_tmp.fill(nan);
  p.fill(nan);
  const auto space = test.get_space();
  if constexpr (std::is_arithmetic_v<Start>) {
    run.x.fill(start);
  } else {
    run.x = start;
  }
  MundyMathBackend::assign("start", run.x, projection(MundyMathBackend{}, space, run.x));

  const auto A = test.get_A();
  const auto q = test.get_q();
  const auto cqpp = make_cqpp<MundyMathBackend>(A, q, space);
  const auto pgd = make_pgd_solution_strategy(make_steps(p), LinfNormProjectedGradientResidual{},
                                              PGDConfig<double>{.max_iters = max_iters, .tol = tol});
  auto state = make_pgd_state(pgd, run.x, grad, x_tmp, grad_tmp);
  pgd.initialize(cqpp, state);
  run.first_expansion_step = state.step_state().expansion_step;
  while (!pgd.done(state)) {
    if (pgd.iterate(cqpp, state)) {
      break;
    }
  }
  run.result = pgd.result(state);
  run.last_expansion_step = state.step_state().expansion_step;
  return run;
}

/// \brief MPRGP on math_backend::BlockedKMSLCP inside a kernel, from x_0 = start, warm-started from its first block's
/// top eigenvector with a basis of 8 columns, to the projected-gradient residual tol.
inline MPRGPRun<math_backend::BlockedKMSLCP::vector_t> run_blocked_kms_in_kernel(double start, double tol) {
  using vector_t = math_backend::BlockedKMSLCP::vector_t;
  constexpr size_t n = math_backend::BlockedKMSLCP::N;
  Kokkos::View<double[n], Kokkos::DefaultExecutionSpace::memory_space> x("x");
  Kokkos::View<double[2], Kokkos::DefaultExecutionSpace::memory_space> steps("steps");
  Kokkos::View<unsigned, Kokkos::DefaultExecutionSpace::memory_space> num_iters("num_iters");
  Kokkos::View<bool, Kokkos::DefaultExecutionSpace::memory_space> converged("converged");
  Kokkos::parallel_for(
      "run_blocked_kms_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int) {
        const math_backend::BlockedKMSLCP problem;
        const auto A = problem.get_A();
        const auto q = problem.get_q();
        const auto cqpp = make_cqpp<MundyMathBackend>(A, q, problem.get_space());
        vector_t x_k, grad, x_tmp, grad_tmp, p;
        x_k.fill(start);
        auto e = problem.get_first_block_top_eigenvector();
        Kokkos::Array<vector_t, 8> basis{};
        const auto pgd =
            make_pgd_solution_strategy(make_mprgp_step_strategy(p, e, basis), LinfNormProjectedGradientResidual{},
                                       PGDConfig<double>{.max_iters = 1000, .tol = tol});
        auto state = make_pgd_state(pgd, x_k, grad, x_tmp, grad_tmp);
        pgd.initialize(cqpp, state);
        steps(0) = state.step_state().expansion_step;
        while (!pgd.done(state)) {
          if (pgd.iterate(cqpp, state)) {
            break;
          }
        }
        steps(1) = state.step_state().expansion_step;
        num_iters() = pgd.result(state).num_iters;
        converged() = pgd.result(state).converged;
        for (size_t i = 0; i < n; ++i) {
          x(i) = x_k[i];
        }
      });
  MPRGPRun<vector_t> run;
  const auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  const auto steps_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, steps);
  for (size_t i = 0; i < n; ++i) {
    run.x[i] = x_host(i);
  }
  run.first_expansion_step = steps_host(0);
  run.last_expansion_step = steps_host(1);
  run.result.num_iters = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, num_iters)();
  run.result.converged = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, converged)();
  return run;
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
/// \brief run_mundy_math_mprgp over views; x is returned on the host.
template <class Test, class Start, class MakeSteps>
auto run_kokkos_mprgp(const Test& test, const Start& start, unsigned max_iters, double tol,
                      const MakeSteps& make_steps) {
  using backend_t = KokkosBackend<decltype(test.get_exec_space())>;
  using vector_t = decltype(test.get_exact_solution());
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  const size_t size = test.size();
  vector_t x("x", size), grad("grad", size), x_tmp("x_tmp", size), grad_tmp("grad_tmp", size), p("p", size);
  Kokkos::deep_copy(grad, nan);
  Kokkos::deep_copy(x_tmp, nan);
  Kokkos::deep_copy(grad_tmp, nan);
  Kokkos::deep_copy(p, nan);
  const auto space = test.get_space();
  Kokkos::deep_copy(x, start);
  backend_t::assign("start", x, projection(backend_t{}, space, x));

  const auto A = test.get_A();
  const auto q = test.get_q();
  const auto cqpp = make_cqpp<backend_t>(A, q, space);
  const auto pgd = make_pgd_solution_strategy(make_steps(p), LinfNormProjectedGradientResidual{},
                                              PGDConfig<double>{.max_iters = max_iters, .tol = tol});
  auto state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
  MPRGPRun<decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x))> run;
  pgd.initialize(cqpp, state);
  run.first_expansion_step = state.step_state().expansion_step;
  while (!pgd.done(state)) {
    if (pgd.iterate(cqpp, state)) {
      break;
    }
  }
  run.result = pgd.result(state);
  run.last_expansion_step = state.step_state().expansion_step;
  run.x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  return run;
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name Analytical solutions
//@{

TEST(Convex, MundyMathAnalyticalSolutions) {
  auto test_cases = std::make_tuple(math_backend::UnconstrainedSPD1Problem{},          //
                                    math_backend::InactiveBoxConstrainedSPDProblem{},  //
                                    math_backend::ActiveBoxConstrainedSPDProblem{},    //
                                    math_backend::ActiveUpperBoundSPDProblem{},        //
                                    math_backend::RandomLCP<3>{},                      //
                                    math_backend::RandomLCP<7>{});
  std::apply(
      [](auto&&... test_case) {
        (run_mundy_math_test(test_case, DefaultPGD{}), ...);
        (run_mundy_math_test(test_case, MPRGPPGD{}), ...);
        (run_mundy_math_test(test_case, NonmonotoneBBPGD{}), ...);
      },
      test_cases);

  // Plain Barzilai-Borwein steps fall into a limit cycle on these
  auto cycling_cases = std::make_tuple(math_backend::PeriodicLaplacianLCP<8>{});
  std::apply(
      [](auto&&... test_case) {
        (run_mundy_math_test(test_case, DefaultPGD{}, Expect::fails), ...);
        (run_mundy_math_test(test_case, MPRGPPGD{}), ...);
        (run_mundy_math_test(test_case, NonmonotoneBBPGD{}), ...);
      },
      cycling_cases);
}

TEST(Convex, MundyMathCongruentAnalyticalSolutions) {
  using math_backend::congruent::CongruentLCPWrapper;
  auto test_cases = std::make_tuple(CongruentLCPWrapper{math_backend::UnconstrainedSPD1Problem{}},          //
                                    CongruentLCPWrapper{math_backend::InactiveBoxConstrainedSPDProblem{}},  //
                                    CongruentLCPWrapper{math_backend::ActiveBoxConstrainedSPDProblem{}},    //
                                    CongruentLCPWrapper{math_backend::ActiveUpperBoundSPDProblem{}},        //
                                    CongruentLCPWrapper{math_backend::RandomLCP<3>{}},                      //
                                    CongruentLCPWrapper{math_backend::RandomLCP<7>{}});
  std::apply(
      [](auto&&... test_case) {
        (run_mundy_math_congruent_test(test_case, DefaultPGD{}), ...);
        (run_mundy_math_congruent_test(test_case, MPRGPPGD{}), ...);
        (run_mundy_math_congruent_test(test_case, NonmonotoneBBPGD{}), ...);
      },
      test_cases);

  // Plain Barzilai-Borwein steps fall into a limit cycle on these
  auto cycling_cases = std::make_tuple(CongruentLCPWrapper{math_backend::PeriodicLaplacianLCP<8>{}});
  std::apply(
      [](auto&&... test_case) {
        (run_mundy_math_congruent_test(test_case, DefaultPGD{}, Expect::fails), ...);
        (run_mundy_math_congruent_test(test_case, MPRGPPGD{}), ...);
        (run_mundy_math_congruent_test(test_case, NonmonotoneBBPGD{}), ...);
      },
      cycling_cases);
}

TEST(Convex, MundyMathMixedCongruentAnalyticalSolutions) {
  // Template params are <NX, NY, NZ>; make_D_full_rank requires NZ >= NX.
  auto test_cases = std::make_tuple(math_backend::mixed::RandomMixedCongruentCCQP<4, 3, 5>{},  //
                                    math_backend::mixed::RandomMixedCongruentCCQP<3, 4, 5>{});
  std::apply(
      [](auto&&... test_case) {
        (run_mundy_math_mixed_congruent_test(test_case, DefaultPGD{}), ...);
        (run_mundy_math_mixed_congruent_test_in_kernel(test_case, DefaultPGD{}), ...);
        (run_mundy_math_mixed_congruent_test(test_case, MPRGPPGD{}), ...);
        (run_mundy_math_mixed_congruent_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_mundy_math_mixed_congruent_test_in_kernel(test_case, MPRGPPGD{}), ...);
        (run_mundy_math_mixed_congruent_test_in_kernel(test_case, NonmonotoneBBPGD{}), ...);
      },
      test_cases);
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
TEST(Convex, KokkosAnalyticalSolutions) {
#if !defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) && !defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER) && !defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  GTEST_SKIP() << "KokkosLapack::gesv requires LAPACK, CUSOLVER, ROCSOLVER, or MAGMA.";
#endif
  auto test_cases = std::make_tuple(kokkos_backend::UnconstrainedSPD1Problem{},          //
                                    kokkos_backend::InactiveBoxConstrainedSPDProblem{},  //
                                    kokkos_backend::ActiveBoxConstrainedSPDProblem{},    //
                                    kokkos_backend::ActiveUpperBoundSPDProblem{},        //
                                    kokkos_backend::RandomLCP{3},                        //
                                    kokkos_backend::RandomLCP{7},                        //
                                    kokkos_backend::RandomLCP{200});
  std::apply(
      [](auto&&... test_case) {
        (run_kokkos_test(test_case, DefaultPGD{}), ...);
        (run_kokkos_test(kokkos_backend::as_sparse(test_case), DefaultPGD{}), ...);
        (run_kokkos_test(test_case, MPRGPPGD{}), ...);
        (run_kokkos_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_kokkos_test(kokkos_backend::as_sparse(test_case), MPRGPPGD{}), ...);
        (run_kokkos_test(kokkos_backend::as_sparse(test_case), NonmonotoneBBPGD{}), ...);
      },
      test_cases);

  // Plain Barzilai-Borwein steps fall into a limit cycle on these
  auto cycling_cases = std::make_tuple(kokkos_backend::PeriodicLaplacianLCP{8},  //
                                       kokkos_backend::PeriodicLaplacianLCP{64});
  std::apply(
      [](auto&&... test_case) {
        (run_kokkos_test(test_case, DefaultPGD{}, Expect::fails), ...);
        (run_kokkos_test(kokkos_backend::as_sparse(test_case), DefaultPGD{}, Expect::fails), ...);
        (run_kokkos_test(test_case, MPRGPPGD{}), ...);
        (run_kokkos_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_kokkos_test(kokkos_backend::as_sparse(test_case), MPRGPPGD{}), ...);
        (run_kokkos_test(kokkos_backend::as_sparse(test_case), NonmonotoneBBPGD{}), ...);
      },
      cycling_cases);
}

TEST(Convex, KokkosCongruentAnalyticalSolutions) {
#if !defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) && !defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER) && !defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  GTEST_SKIP() << "KokkosLapack::gesv requires LAPACK, CUSOLVER, ROCSOLVER, or MAGMA.";
#endif
  using kokkos_backend::congruent::CongruentLCPWrapper;
  auto test_cases = std::make_tuple(CongruentLCPWrapper{kokkos_backend::UnconstrainedSPD1Problem{}},          //
                                    CongruentLCPWrapper{kokkos_backend::InactiveBoxConstrainedSPDProblem{}},  //
                                    CongruentLCPWrapper{kokkos_backend::ActiveBoxConstrainedSPDProblem{}},    //
                                    CongruentLCPWrapper{kokkos_backend::ActiveUpperBoundSPDProblem{}},        //
                                    CongruentLCPWrapper{kokkos_backend::RandomLCP{3}},                        //
                                    CongruentLCPWrapper{kokkos_backend::RandomLCP{7}},                        //
                                    CongruentLCPWrapper{kokkos_backend::RandomLCP{200}});
  std::apply(
      [](auto&&... test_case) {
        (run_kokkos_congruent_test(test_case, DefaultPGD{}), ...);
        (run_kokkos_congruent_test(kokkos_backend::as_sparse(test_case), DefaultPGD{}), ...);
        (run_kokkos_congruent_test(test_case, MPRGPPGD{}), ...);
        (run_kokkos_congruent_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_kokkos_congruent_test(kokkos_backend::as_sparse(test_case), MPRGPPGD{}), ...);
        (run_kokkos_congruent_test(kokkos_backend::as_sparse(test_case), NonmonotoneBBPGD{}), ...);
      },
      test_cases);

  // Plain Barzilai-Borwein steps fall into a limit cycle on these
  auto cycling_cases = std::make_tuple(CongruentLCPWrapper{kokkos_backend::PeriodicLaplacianLCP{8}},  //
                                       CongruentLCPWrapper{kokkos_backend::PeriodicLaplacianLCP{64}});
  std::apply(
      [](auto&&... test_case) {
        (run_kokkos_congruent_test(test_case, DefaultPGD{}, Expect::fails), ...);
        (run_kokkos_congruent_test(kokkos_backend::as_sparse(test_case), DefaultPGD{}, Expect::fails), ...);
        (run_kokkos_congruent_test(test_case, MPRGPPGD{}), ...);
        (run_kokkos_congruent_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_kokkos_congruent_test(kokkos_backend::as_sparse(test_case), MPRGPPGD{}), ...);
        (run_kokkos_congruent_test(kokkos_backend::as_sparse(test_case), NonmonotoneBBPGD{}), ...);
      },
      cycling_cases);
}

TEST(Convex, KokkosMixedCongruentAnalyticalSolutions) {
#if !defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) && !defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER) && !defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  GTEST_SKIP() << "KokkosLapack::gesv requires LAPACK, CUSOLVER, ROCSOLVER, or MAGMA.";
#endif
  // NX <= NZ, so the reduced Hessian D^T (M - M B S B^T M) D can be nonsingular and x* is the unique solution.
  auto test_cases = std::make_tuple(kokkos_backend::mixed::RandomMixedCongruentCCQP<4, 3, 5>{},  //
                                    kokkos_backend::mixed::RandomMixedCongruentCCQP<3, 4, 5>{});
  std::apply(
      [](auto&&... test_case) {
        (run_kokkos_mixed_congruent_test(test_case, DefaultPGD{}), ...);
        (run_kokkos_mixed_congruent_test(kokkos_backend::as_sparse(test_case), DefaultPGD{}), ...);
        (run_kokkos_mixed_congruent_test(test_case, MPRGPPGD{}), ...);
        (run_kokkos_mixed_congruent_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_kokkos_mixed_congruent_test(kokkos_backend::as_sparse(test_case), MPRGPPGD{}), ...);
        (run_kokkos_mixed_congruent_test(kokkos_backend::as_sparse(test_case), NonmonotoneBBPGD{}), ...);
      },
      test_cases);
}

TEST(Convex, KokkosMixedCongruentWorkspaceReuse) {
#if !defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) && !defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER) && !defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  GTEST_SKIP() << "KokkosLapack::gesv requires LAPACK, CUSOLVER, ROCSOLVER, or MAGMA.";
#endif
  auto test_cases = std::make_tuple(kokkos_backend::mixed::RandomMixedCongruentCCQP<5, 4, 3>{},  //
                                    kokkos_backend::mixed::RandomMixedCongruentCCQP<3, 4, 5>{});
  std::apply(
      [](auto&&... test_case) {
        (run_kokkos_mixed_congruent_workspace_test(test_case, DefaultPGD{}), ...);
        (run_kokkos_mixed_congruent_shared_s_workspace_test(test_case, DefaultPGD{}), ...);
        (run_kokkos_mixed_congruent_workspace_test(kokkos_backend::as_sparse(test_case), DefaultPGD{}), ...);
        (run_kokkos_mixed_congruent_shared_s_workspace_test(kokkos_backend::as_sparse(test_case), DefaultPGD{}), ...);
        (run_kokkos_mixed_congruent_shared_s_workspace_test(test_case, MPRGPPGD{}), ...);
        (run_kokkos_mixed_congruent_shared_s_workspace_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_kokkos_mixed_congruent_workspace_test(test_case, MPRGPPGD{}), ...);
        (run_kokkos_mixed_congruent_workspace_test(test_case, NonmonotoneBBPGD{}), ...);
        (run_kokkos_mixed_congruent_workspace_test(kokkos_backend::as_sparse(test_case), MPRGPPGD{}), ...);
        (run_kokkos_mixed_congruent_workspace_test(kokkos_backend::as_sparse(test_case), NonmonotoneBBPGD{}), ...);
      },
      test_cases);
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name Scale invariance
//@{

// PGD commutes with scaling the problem: (c A, c q) to tolerance c tol has the same minimizer, its steps scale as 1/c
// and its gradients as c. A power-of-two c scales exactly, so the scaled solves match the unscaled one bit for bit,
// under Barzilai-Borwein, nonmonotone Barzilai-Borwein, and MPRGP steps. Each measures the projected gradient, which
// scales with c; the projected difference's fixed step does not.
TEST(Convex, PGDIsScaleInvariant) {
  const math_backend::RandomLCP<7> test;
  using vector_t = decltype(test.get_exact_solution());
  const auto expect_scale_invariant = [&](const auto& make_pgd, std::string_view steps) {
    // Unscaled
    vector_t x{}, grad{}, x_tmp{}, grad_tmp{}, p{};
    x.fill(99.99);
    const auto pgd = make_pgd(PGDConfig<double>{.max_iters = 1000, .tol = 1e-12}, p);
    auto state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
    const auto unscaled =
        solve_cqpp(make_cqpp<MundyMathBackend>(test.get_A(), test.get_q(), test.get_space()), pgd, state);
    ASSERT_TRUE(unscaled.converged) << steps << ": " << unscaled;

    // Scaled by 2^-20 and 2^-40
    for (const int exponent : {-20, -40}) {
      const double c = std::ldexp(1.0, exponent);
      vector_t x_scaled{}, grad_scaled{}, x_tmp_scaled{}, grad_tmp_scaled{}, p_scaled{};
      x_scaled.fill(99.99);
      const auto scaled_pgd = make_pgd(PGDConfig<double>{.max_iters = 1000, .tol = c * 1e-12}, p_scaled);
      auto scaled_state = make_pgd_state(scaled_pgd, x_scaled, grad_scaled, x_tmp_scaled, grad_tmp_scaled);
      const auto scaled = solve_cqpp(make_cqpp<MundyMathBackend>(test.get_A() * c, test.get_q() * c, test.get_space()),
                                     scaled_pgd, scaled_state);
      EXPECT_TRUE(scaled.converged) << steps << ", c = 2^" << exponent << ": " << scaled;
      EXPECT_EQ(scaled.num_iters, unscaled.num_iters) << steps << ", c = 2^" << exponent;
      for (size_t i = 0; i < vector_t::size; ++i) {
        EXPECT_EQ(x_scaled[i], x[i]) << steps << ", c = 2^" << exponent << ", entry " << i;
      }
    }
  };
  expect_scale_invariant(BBPGD<LinfNormProjectedGradientResidual>{}, "Barzilai-Borwein");
  expect_scale_invariant(NonmonotoneBBPGD<LinfNormProjectedGradientResidual>{}, "nonmonotone Barzilai-Borwein");
  expect_scale_invariant(MPRGPPGD<LinfNormProjectedGradientResidual>{}, "MPRGP");
}
//@}

//! \name Iteration counts
//@{

TEST(Convex, IterationsCountEveryStepTaken) {
  {
    SCOPED_TRACE("Barzilai-Borwein");
    expect_iterations_count_every_step(DefaultPGD{});
  }
  {
    SCOPED_TRACE("nonmonotone Barzilai-Borwein");
    expect_iterations_count_every_step(NonmonotoneBBPGD{});
  }
  {
    SCOPED_TRACE("MPRGP");
    expect_iterations_count_every_step(MPRGPPGD{});
  }
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
// A problem with no unknowns starts converged, so every step strategy takes no step.
TEST(Convex, EmptyProblemsTakeNoStep) {
  using backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
  using vector_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
  using matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
  const matrix_t A("A", 0, 0);
  const vector_t q("q", 0);
  const auto cqpp = make_cqpp<backend_t>(A, q, LowerBoundSpace<double>(0.0));
  const auto expect_no_step = [&](const auto& make_pgd, std::string_view steps) {
    vector_t x("x", 0), grad("grad", 0), x_tmp("x_tmp", 0), grad_tmp("grad_tmp", 0), p("p", 0);
    const auto pgd = make_pgd(PGDConfig<double>{.max_iters = 1000, .tol = 1e-12}, p);
    auto state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
    const PGDResult<double> result = solve_cqpp(cqpp, pgd, state);
    EXPECT_TRUE(result.converged) << steps << ": " << result;
    EXPECT_EQ(result.num_iters, 0u) << steps;
  };
  expect_no_step(DefaultPGD{}, "Barzilai-Borwein");
  expect_no_step(NonmonotoneBBPGD{}, "nonmonotone Barzilai-Borwein");
  expect_no_step(MPRGPPGD{}, "MPRGP");
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name Returned iterates
//@{

TEST(Convex, SolvesEndOnAnApplyAtTheirIterate) {
  {
    SCOPED_TRACE("Barzilai-Borwein");
    expect_solves_end_on_an_apply_at_their_iterate(BBPGD<LinfNormProjectedDiffResidual>{});
  }
  {
    SCOPED_TRACE("nonmonotone Barzilai-Borwein");
    expect_solves_end_on_an_apply_at_their_iterate(NonmonotoneBBPGD<LinfNormProjectedDiffResidual>{});
  }
  {
    SCOPED_TRACE("MPRGP");
    expect_solves_end_on_an_apply_at_their_iterate(MPRGPPGD<LinfNormProjectedDiffResidual>{});
  }
}
//@}

//! \name Nearly parallel rows
//@{

// Two rows at an angle delta = 0.1, the second asking eps = 0.01 more: 0 <= A x + q _|_ x >= 0 with
// A = [[1, cos delta], [cos delta, 1]] and q = -(1, 1 + eps) is solved by x* = (0, 1 + eps), the first row slack by
// (1 + eps) cos delta - 1 > 0. From (1/2, 0), projected Barzilai-Borwein steps alone do not reach x* within 1000 steps,
// and under the nonmonotone line search they do.
TEST(Convex, NonmonotonePGDEndsBBCycle) {
  constexpr double delta = 0.1;
  constexpr double eps = 0.01;
  using vector_t = Vector<double, 2>;
  const Matrix<double, 2, 2> A{1.0, std::cos(delta), std::cos(delta), 1.0};
  const vector_t q{-1.0, -1.0 - eps};
  const LowerBoundSpace<double> space(0.0);
  const auto problem = make_cqpp<MundyMathBackend>(A, q, space);
  const PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-12};

  // Plain Barzilai-Borwein steps
  vector_t x_bb{0.5, 0.0}, grad{}, x_tmp{}, grad_tmp{}, p{};
  const auto bb = BBPGD<LinfNormProjectedGradientResidual>{}(cfg, p);
  auto bb_state = make_pgd_state(bb, x_bb, grad, x_tmp, grad_tmp);
  const PGDResult<double> bb_result = solve_cqpp(problem, bb, bb_state);
  EXPECT_FALSE(bb_result.converged) << bb_result;

  // Under the line search
  vector_t x{0.5, 0.0};
  const auto nonmonotone = NonmonotoneBBPGD<LinfNormProjectedGradientResidual>{}(cfg, p);
  auto state = make_pgd_state(nonmonotone, x, grad, x_tmp, grad_tmp);
  const PGDResult<double> result = solve_cqpp(problem, nonmonotone, state);
  ASSERT_TRUE(result.converged) << result;

  // ||g^P||_inf <= tol at x, and the first row is slack by far more than tol, so x_0 lies on its bound; |g_1| <= tol
  // then puts x_1 within tol + cos(delta) x_0 of 1 + eps.
  EXPECT_GE(x[0], 0.0);
  EXPECT_LE(x[0], std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(x[1], 1.0 + eps, cfg.tol + std::cos(delta) * x[0]);
  EXPECT_LE(LinfNormProjectedGradientResidual{}(MundyMathBackend{}, x, A * x + q, space), cfg.tol);
}

// Two rows at an angle delta = 1e-3, the second asking eps = 1e-6 more: 0 <= A x + q _|_ x >= 0 with
// A = [[1, cos delta], [cos delta, 1]], of condition number about 4 / delta^2 = 4e6, and q = -(1, 1 + eps) is solved by
// x* = (0, 1 + eps), the first row slack by (1 + eps) cos delta - 1 = 5e-7. From (1, 0), Barzilai-Borwein steps under
// the nonmonotone line search do not reach x* within 1000 steps, and MPRGP's steps do.
TEST(Convex, MPRGPSettlesNearDuplicateRows) {
  constexpr double delta = 1e-3;
  constexpr double eps = 1e-6;
  using vector_t = Vector<double, 2>;
  const Matrix<double, 2, 2> A{1.0, std::cos(delta), std::cos(delta), 1.0};
  const vector_t q{-1.0, -1.0 - eps};
  const LowerBoundSpace<double> space(0.0);
  const auto problem = make_cqpp<MundyMathBackend>(A, q, space);
  const PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-12};

  // Barzilai-Borwein steps under the line search
  vector_t x_nonmonotone{1.0, 0.0}, grad{}, x_tmp{}, grad_tmp{}, p{};
  const auto nonmonotone = NonmonotoneBBPGD<LinfNormProjectedGradientResidual>{}(cfg, p);
  auto nonmonotone_state = make_pgd_state(nonmonotone, x_nonmonotone, grad, x_tmp, grad_tmp);
  const PGDResult<double> nonmonotone_result = solve_cqpp(problem, nonmonotone, nonmonotone_state);
  EXPECT_FALSE(nonmonotone_result.converged) << nonmonotone_result;

  // MPRGP steps
  vector_t x{1.0, 0.0};
  const auto mprgp = MPRGPPGD<LinfNormProjectedGradientResidual>{}(cfg, p);
  auto state = make_pgd_state(mprgp, x, grad, x_tmp, grad_tmp);
  const PGDResult<double> result = solve_cqpp(problem, mprgp, state);
  ASSERT_TRUE(result.converged) << result;

  // ||g^P||_inf <= tol at x, and the first row is slack by far more than tol, so x_0 lies on its bound; |g_1| <= tol
  // then puts x_1 within tol + cos(delta) x_0 of 1 + eps.
  EXPECT_GE(x[0], 0.0);
  EXPECT_LE(x[0], std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(x[1], 1.0 + eps, cfg.tol + std::cos(delta) * x[0]);
  EXPECT_LE(LinfNormProjectedGradientResidual{}(MundyMathBackend{}, x, A * x + q, space), cfg.tol);
}
//@}

//! \name MPRGP's estimate of lambda_max
//@{

// MPRGP's convergence theory bounds its expansion step by 2 / lambda_max. From the cold start the first expansion step,
// the solve's longest, is within it on problems of known lambda_max: 4.01 for PeriodicLaplacianLCP, along
// (1, -1, ...), and 2 + sqrt(2) for tridiag(-1, 2, -1) of size 3. So is that of a solve started from the estimate
// vector the cold one leaves. The step is 1.8 / lambda, within the bound once lambda >= 0.9 lambda_max: a 10% margin
// against Paige's rounding term 3 sqrt(k) eps_1 ||A|| + 4 eps ||T||, below 1e-12 lambda_max here, so the comparisons
// are exact.
TEST(Convex, MPRGPStepMeetsTheory) {
  const double sqrt2 = std::sqrt(2.0);
  const auto expect_mundy_math = [](const auto& test, double lambda_max) {
    using vector_t = decltype(test.get_exact_solution());
    vector_t e{};
    Kokkos::Array<vector_t, 8> basis{};
    const auto cold =
        run_mundy_math_mprgp(test, 99.99, 1000, 1e-10, [](auto& p) { return make_mprgp_step_strategy(p); });
    const auto warm = [&](auto& p) { return make_mprgp_step_strategy(p, e, basis); };
    run_mundy_math_mprgp(test, 99.99, 1000, 1e-10, warm);
    const auto from_estimate = run_mundy_math_mprgp(test, 99.99, 1000, 1e-10, warm);
    EXPECT_TRUE(cold.result.converged) << test.name() << ": " << cold.result;
    EXPECT_TRUE(from_estimate.result.converged) << test.name() << ": " << from_estimate.result;
    EXPECT_LE(cold.first_expansion_step, 2.0 / lambda_max) << test.name();
    EXPECT_LE(from_estimate.first_expansion_step, 2.0 / lambda_max) << test.name();
  };
  expect_mundy_math(math_backend::PeriodicLaplacianLCP<8>{}, 4.01);
  expect_mundy_math(math_backend::UnconstrainedSPD1Problem{}, 2.0 + sqrt2);
  expect_mundy_math(math_backend::InactiveBoxConstrainedSPDProblem{}, 2.0 + sqrt2);
  expect_mundy_math(math_backend::ActiveBoxConstrainedSPDProblem{}, 2.0 + sqrt2);
  expect_mundy_math(math_backend::ActiveUpperBoundSPDProblem{}, 2.0 + sqrt2);

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  const auto expect_kokkos = [](const auto& test, double lambda_max) {
    using vector_t = decltype(test.get_exact_solution());
    vector_t e("e", test.size());
    Kokkos::View<double* [8], Kokkos::LayoutLeft, typename vector_t::memory_space> basis("basis", test.size());
    const auto cold = run_kokkos_mprgp(test, 99.99, 1000, 1e-10, [](auto& p) { return make_mprgp_step_strategy(p); });
    const auto warm = [&](auto& p) { return make_mprgp_step_strategy(p, e, basis); };
    run_kokkos_mprgp(test, 99.99, 1000, 1e-10, warm);
    const auto from_estimate = run_kokkos_mprgp(test, 99.99, 1000, 1e-10, warm);
    EXPECT_TRUE(cold.result.converged) << test.name() << ": " << cold.result;
    EXPECT_TRUE(from_estimate.result.converged) << test.name() << ": " << from_estimate.result;
    EXPECT_LE(cold.first_expansion_step, 2.0 / lambda_max) << test.name();
    EXPECT_LE(from_estimate.first_expansion_step, 2.0 / lambda_max) << test.name();
  };
  for (const unsigned size : {8u, 64u}) {
    expect_kokkos(kokkos_backend::PeriodicLaplacianLCP{size}, 4.01);
    expect_kokkos(kokkos_backend::as_sparse(kokkos_backend::PeriodicLaplacianLCP{size}), 4.01);
  }
  const auto three_by_three =
      std::make_tuple(kokkos_backend::UnconstrainedSPD1Problem{}, kokkos_backend::InactiveBoxConstrainedSPDProblem{},
                      kokkos_backend::ActiveBoxConstrainedSPDProblem{}, kokkos_backend::ActiveUpperBoundSPDProblem{});
  std::apply(
      [&](const auto&... test) {
        (expect_kokkos(test, 2.0 + sqrt2), ...);
        (expect_kokkos(kokkos_backend::as_sparse(test), 2.0 + sqrt2), ...);
      },
      three_by_three);
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
// The estimate vector e: a zero e is the cold start, bit for bit, and the basis's number of columns changes neither the
// solve nor, beyond rounding, the estimate it leaves in e. e on exit is a unit vector whose Rayleigh quotient is
// lambda_max to within the estimate's tolerance, and a solve that takes no step leaves e as it was. A warm solve
// allocates nothing but its sparse applies' allocations.
TEST(Convex, MPRGPWarmStartContract) {
  constexpr double tol = 1e-10;
  const auto expect_contract = [](const auto& test) {
    using vector_t = decltype(test.get_exact_solution());
    using memory_space = typename vector_t::memory_space;
    const size_t size = test.size();
    const auto cold_steps = [](auto& p) { return make_mprgp_step_strategy(p); };
    const auto warm_steps = [](auto& e, auto& basis) {
      return [&](auto& p) { return make_mprgp_step_strategy(p, e, basis); };
    };
    const auto expect_same_solve = [&](const auto& run, const auto& reference, const std::string& what) {
      EXPECT_EQ(run.result.num_iters, reference.result.num_iters) << test.name() << ", " << what;
      EXPECT_EQ(run.first_expansion_step, reference.first_expansion_step) << test.name() << ", " << what;
      EXPECT_EQ(run.last_expansion_step, reference.last_expansion_step) << test.name() << ", " << what;
      for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(run.x(i), reference.x(i)) << test.name() << ", " << what << ", entry " << i;
      }
    };

    // A zero e is the cold start
    vector_t e("e", size);
    Kokkos::View<double* [32], Kokkos::LayoutLeft, memory_space> basis("basis", size);
    const auto cold = run_kokkos_mprgp(test, 99.99, 1000, tol, cold_steps);
    ASSERT_TRUE(cold.result.converged) << test.name() << ": " << cold.result;
    expect_same_solve(run_kokkos_mprgp(test, 99.99, 1000, tol, warm_steps(e, basis)), cold, "zero e");

    // One basis column or 32, from e = 0: the same solve, and the same estimate to rounding. As in
    // Eigenvalues.LanczosBasisCapacity, each of the estimate's k-term sums rounds by k eps on either side and the
    // normalizations by (n + 6) eps beyond the sums' difference, for k at most norm_estimate.max_iters = 32.
    vector_t e_1("e_1", size), e_32("e_32", size);
    Kokkos::View<double* [1], Kokkos::LayoutLeft, memory_space> basis_1("basis_1", size);
    Kokkos::View<double* [32], Kokkos::LayoutLeft, memory_space> basis_32("basis_32", size);
    const auto with_1 = run_kokkos_mprgp(test, 99.99, 1000, tol, warm_steps(e_1, basis_1));
    const auto with_32 = run_kokkos_mprgp(test, 99.99, 1000, tol, warm_steps(e_32, basis_32));
    expect_same_solve(with_1, with_32, "1 or 32 columns");
    const double eps = std::numeric_limits<double>::epsilon();
    const double k = make_mprgp_step_strategy(e).norm_estimate.max_iters;
    const double estimate_tol = std::min(size * k * eps, (2 * k * (1 + std::sqrt(double(size))) + size + 6) * eps);
    const auto e_1_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, e_1);
    const auto e_32_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, e_32);
    for (size_t i = 0; i < size; ++i) {
      EXPECT_NEAR(e_1_host(i), e_32_host(i), estimate_tol) << test.name() << ", entry " << i;
    }

    // A solve that takes no step, from x* or capped at 0 iterations, leaves e as it was
    vector_t e_kept = test.get_first_block_top_eigenvector();
    const auto e_before = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, e_kept);
    const auto at_solution = run_kokkos_mprgp(test, test.get_exact_solution(), 1000, tol, warm_steps(e_kept, basis));
    const auto capped = run_kokkos_mprgp(test, 99.99, 0, tol, warm_steps(e_kept, basis));
    EXPECT_EQ(at_solution.result.num_iters, 0u) << test.name();
    EXPECT_EQ(capped.result.num_iters, 0u) << test.name();
    const auto e_after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, e_kept);
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(e_after(i), e_before(i)) << test.name() << ", entry " << i;
    }
  };
  expect_contract(kokkos_backend::BlockedKMSLCP{});
  expect_contract(kokkos_backend::as_sparse(kokkos_backend::BlockedKMSLCP{}));

  // e on exit, on PeriodicLaplacianLCP of size 8: lambda_max = 4.01 is 0.586 from the next eigenvalue, and the cold
  // start reaches every one of the five distinct eigenvalues.
  //   - |e^T e - 1| <= 2 gamma_{n+3}: gamma_n for e's normalizing sum, 2u each for its square root, reciprocal, and
  //     entrywise product (each doubled by squaring), and gamma_n for this test's sum.
  //   - |R(e) - lambda_max| <= lambda_max (tol / (1 - tol) + 3 sqrt(k) eps_1 + 4 eps + gamma_{2n+1}): Paige's bound
  //     puts theta within rho <= tol theta of lambda_max in exact arithmetic, so within tol lambda_max / (1 - tol), and
  //     adds 3 sqrt(k) eps_1 ||A|| + 4 eps ||T|| in floating point, for k <= 32 iterations, ||T|| <= ||A|| = 4.01 and
  //     eps_1 = 2 (7 + m beta) eps with m = 3 nonzeros per row and beta = || |A| || / ||A|| = 1 (Paige, Linear Algebra
  //     Appl. 34, 1980); R(e) = theta for the Ritz vector in exact arithmetic; gamma_{2n+1} bounds this test's apply
  //     and dot products and its quotient.
  {
    const kokkos_backend::PeriodicLaplacianLCP test{8};
    using backend_t = KokkosBackend<decltype(test.get_exec_space())>;
    using vector_t = decltype(test.get_exact_solution());
    const size_t n = test.size();
    const double lambda_max = 4.01;
    vector_t e("e", n), A_e("A_e", n);
    const double estimate_tol = make_mprgp_step_strategy(e).norm_estimate.tol;
    Kokkos::View<double* [8], Kokkos::LayoutLeft, typename vector_t::memory_space> basis("basis", n);
    run_kokkos_mprgp(test, 99.99, 1000, tol, [&](auto& p) { return make_mprgp_step_strategy(p, e, basis); });
    backend_t::apply(test.get_A(), e, A_e);
    const double e_e = dot(backend_t{}, e, e);
    const double eps = std::numeric_limits<double>::epsilon();
    EXPECT_LE(std::abs(e_e - 1.0), 2.0 * rounding_gamma(n + 3));
    const double eps_1 = 2.0 * (7.0 + 3.0) * eps;
    const double rayleigh_tol = lambda_max * (estimate_tol / (1.0 - estimate_tol) + 3.0 * std::sqrt(32.0) * eps_1 +
                                              4.0 * eps + rounding_gamma(2 * n + 1));
    EXPECT_NEAR(dot(backend_t{}, e, A_e) / e_e, lambda_max, rayleigh_tol);
  }

  // Nothing allocated but the sparse applies' allocations
  {
    const auto test = kokkos_backend::as_sparse(kokkos_backend::BlockedKMSLCP{});
    using exec_space = decltype(test.get_exec_space());
    using backend_t = KokkosBackend<exec_space>;
    using vector_t = decltype(test.get_exact_solution());
    const size_t size = test.size();
    const auto A = test.get_A();
    const auto q = test.get_q();
    vector_t x("x", size), grad("grad", size), x_tmp("x_tmp", size), grad_tmp("grad_tmp", size), p("p", size),
        e("e", size);
    Kokkos::View<double* [8], Kokkos::LayoutLeft, typename vector_t::memory_space> basis("basis", size);
    Kokkos::deep_copy(x, 99.99);
    PGDResult<double> result;
    const AllocationCount counted = count_allocations([&] {
      const auto cqpp = make_cqpp<backend_t>(A, q, test.get_space());
      const auto pgd =
          make_pgd_solution_strategy(make_mprgp_step_strategy(p, e, basis), LinfNormProjectedGradientResidual{},
                                     PGDConfig<double>{.max_iters = 1000, .tol = tol});
      auto state = make_pgd_state(pgd, x, grad, x_tmp, grad_tmp);
      result = solve_cqpp(cqpp, pgd, state);
    });
    EXPECT_TRUE(result.converged) << result;
    EXPECT_GT(counted.sparse_applies, 0u);
    EXPECT_EQ(counted.allocations, allocations_per_sparse_apply<exec_space>() * counted.sparse_applies);
  }
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

// Started from an exact eigenvector of A for 4.01, BlockedKMSLCP's first block's top one, the estimate stops at once
// and the first expansion step is beyond 2 / lambda_max: 2 / R(v) for v = (0, 1, ..., 1), whose curvature R(v) > 19
// bounds lambda_max from below. The steps' curvature shortens the step, and the solve converges, from x_0 = 99.99 and
// from x_0 = 0, on both backends, the MundyMath one in a kernel. The step stays above 1.8 / lambda_max:
//   - The step's curvatures are Rayleigh quotients, each at most lambda_max <= G, Gershgorin's max_i sum_j |A_ij|, as
//     computed with at most gamma_{2n} relative error in v^T A v (an apply of up to n terms per row, then a dot),
//     gamma_n in v^T v, 2u in forming fraction v^T v / v^T A v, and gamma_n in G: step >= (1.8 / G) (1 - gamma_{4n+2}).
//   - x is within (1 + lambda_max) / lambda_min ||x - P(x - (A x + q))||_2 of x* (the error bound of a strongly
//     monotone affine variational inequality; Facchinei & Pang, Finite-Dimensional Variational Inequalities, 2003,
//     Prop. 6.3.1). The natural residual's entries are at most the projected gradient's, which the solve holds to tol
//     up to the rounding of A x + q, gamma_{n+1} (G ||x||_inf + ||q||_inf). lambda_max <= G, and lambda_min >= 0.01,
//     the least of L's 0.01 and 2 K's 2 (1 - 0.9) / (1 + 0.9), the lower bound of a Kac-Murdock-Szego spectrum.
TEST(Convex, MPRGPRecoversFromWarmUndershoot) {
  constexpr double tol = 1e-10;
  constexpr size_t n = math_backend::BlockedKMSLCP::N;
  const math_backend::BlockedKMSLCP test;
  const auto A = test.get_A();
  const auto q = test.get_q();
  const auto x_star = test.get_exact_solution();
  using vector_t = math_backend::BlockedKMSLCP::vector_t;

  // G and R(v)
  double gershgorin = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double row = 0.0;
    for (size_t j = 0; j < n; ++j) {
      row += abs(A(i, j));
    }
    gershgorin = max(gershgorin, row);
  }
  vector_t v;
  for (size_t i = 0; i < n; ++i) {
    v[i] = i < 8 ? 0.0 : 1.0;
  }
  const double curvature = dot(v, A * v) / dot(v, v);
  const double fraction = make_mprgp_step_strategy(v).expansion_step_fraction;
  const double q_max = max(abs(q));

  const auto expect_recovers = [&](const auto& run, const std::string& what) {
    // Hardness: the first step is beyond 2 / lambda_max, as R(v) <= lambda_max
    EXPECT_GT(run.first_expansion_step * curvature, 2.0) << what;

    // Recovery
    ASSERT_TRUE(run.result.converged) << what << ": " << run.result;
    EXPECT_LT(run.last_expansion_step, run.first_expansion_step) << what;
    EXPECT_GE(run.last_expansion_step, fraction / gershgorin * (1.0 - rounding_gamma(4 * n + 2))) << what;
    double x_max = 0.0;
    double error = 0.0;
    for (size_t i = 0; i < n; ++i) {
      x_max = max(x_max, abs(run.x[i]));
      error = max(error, abs(run.x[i] - x_star[i]));
    }
    const double residual_bound = tol + rounding_gamma(n + 1) * (gershgorin * x_max + q_max);
    EXPECT_LE(error, (1.0 + gershgorin) / 0.01 * std::sqrt(double(n)) * residual_bound) << what;
  };

  // MundyMath, in a kernel
  for (const double x_0 : {99.99, 0.0}) {
    expect_recovers(run_blocked_kms_in_kernel(x_0, tol), "MundyMath in a kernel from x_0 = " + std::to_string(x_0));
  }

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // Kokkos, dense and sparse
  const auto expect_kokkos = [&](const auto& kokkos_test) {
    using kokkos_vector_t = decltype(kokkos_test.get_exact_solution());
    Kokkos::View<double* [8], Kokkos::LayoutLeft, typename kokkos_vector_t::memory_space> basis("basis", n);
    for (const double x_0 : {99.99, 0.0}) {
      kokkos_vector_t e = kokkos_test.get_first_block_top_eigenvector();
      const auto kokkos_run =
          run_kokkos_mprgp(kokkos_test, x_0, 1000, tol, [&](auto& p) { return make_mprgp_step_strategy(p, e, basis); });
      MPRGPRun<vector_t> run{kokkos_run.first_expansion_step, kokkos_run.last_expansion_step, kokkos_run.result, {}};
      for (size_t i = 0; i < n; ++i) {
        run.x[i] = kokkos_run.x(i);
      }
      expect_recovers(run, kokkos_test.name() + " from x_0 = " + std::to_string(x_0));
    }
  };
  expect_kokkos(kokkos_backend::BlockedKMSLCP{});
  expect_kokkos(kokkos_backend::as_sparse(kokkos_backend::BlockedKMSLCP{}));
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
}
//@}

//! \name Residuals
//@{

// Over x >= 0, x is optimal when grad_i = 0 where x_i > 0 and grad_i >= 0 where x_i = 0. Both residuals measure the
// violation: |grad_i| off the bound and max(0, -grad_i) on it. The projected difference resolves its step h = 1e-6
// only to ulp(x) / h ~ 1e-10.
TEST(Convex, ResidualsMeasureOptimalityViolation) {
  const LowerBoundSpace<double> space{.lower_bound = 0.0};
  const Vector3d x{0.0, 0.0, 1.0};
  const Vector3d optimal{0.5, 0.0, 0.0};            // x_0 pressed onto its bound, x_2 stationary
  const Vector3d pulled_off_bound{-0.5, 0.0, 0.0};  // x_0 would decrease the objective by leaving its bound
  const Vector3d not_stationary{0.5, 0.0, 0.3};     // x_2 off its bound with a nonzero gradient

  EXPECT_EQ(LinfNormProjectedGradientResidual{}(MundyMathBackend{}, x, optimal, space), 0.0);
  EXPECT_EQ(LinfNormProjectedGradientResidual{}(MundyMathBackend{}, x, pulled_off_bound, space), 0.5);
  EXPECT_EQ(LinfNormProjectedGradientResidual{}(MundyMathBackend{}, x, not_stationary, space), 0.3);

  EXPECT_EQ(LinfNormProjectedDiffResidual{}(MundyMathBackend{}, x, optimal, space), 0.0);
  EXPECT_NEAR(LinfNormProjectedDiffResidual{}(MundyMathBackend{}, x, pulled_off_bound, space), 0.5, 1e-9);
  EXPECT_NEAR(LinfNormProjectedDiffResidual{}(MundyMathBackend{}, x, not_stationary, space), 0.3, 1e-9);
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
// Every residual is a norm, so that of a zero-length iterate is exactly zero.
TEST(Convex, EmptyResidualsAreZero) {
  using backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
  const Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> x("x", 0), grad("grad", 0);
  const LowerBoundSpace<double> space{.lower_bound = 0.0};
  EXPECT_EQ(LinfNormProjectedGradientResidual{}(backend_t{}, x, grad, space), 0.0);
  EXPECT_EQ(LinfNormProjectedDiffResidual{}(backend_t{}, x, grad, space), 0.0);
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

}  // namespace

}  // namespace mundy
