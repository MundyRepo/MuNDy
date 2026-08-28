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
#include <ostream>  // for std::cout

// Mundy
#include <mundy_math/Matrix.hpp>  // for mundy::Matrix
#include <mundy_math/Vector.hpp>  // for mundy::Vector
#include <mundy_math/cmath.hpp>
#include <mundy_math/convex_spaces.hpp>
#include <mundy_math/cqpp.hpp>
#include <mundy_math/lcp.hpp>
#include <mundy_utils/rng.hpp>  // for mundy::make_philox

namespace mundy {

namespace {

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

  linear_op_t gen_random_matrix(unsigned size) {
    linear_op_t mat(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mat"), size, size);

    // Fill with random values in [-1, 1] (not a statistically random matrix but this is a test)
    Kokkos::parallel_for(
        "gen_random_matrix", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {size, size}),
        KOKKOS_LAMBDA(const size_t i, const size_t j) {
          openrand::Philox rng = make_philox(i, j);
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
  const auto cqpp = make_cqpp<MundyMathBackend>(A, q, space);

  // Strategy + state
  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
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

  // Solving via explicit quadratic form
  {
    using vector_t = decltype(x_exact);
    vector_t x{}, grad{}, x_tmp{}, grad_tmp{};
    vector_t f{}, u{};

    x.fill(99.99);  // use a bad initial guess to force more iterations

    // Build quadratic form operator + user-owned workspace, then the problem
    const auto A = make_quadratic_form<MundyMathBackend>(DT, M, D);
    auto workspace = A.make_workspace(f, u);  // intermediate variables f = D x, u = M f
    const auto cqpp = make_cqpp<MundyMathBackend>(A, q, space, workspace);

    // Strategy + state
    PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
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

  // Solving via quadratic form helper
  {
    using vector_t = decltype(x_exact);
    vector_t x{}, grad{}, x_tmp{}, grad_tmp{};
    vector_t f{}, u{};

    x.fill(99.99);  // use a bad initial guess to force more iterations

    // Build the cqpp directly
    // Use f = D x and u = M f as intermediate variables to avoid redundant computations in the PGD iterations
    // The final values of f and u post-solve are guaranteed to be f(x^*) and u(x^*).
    const auto cqpp = make_cqpp<MundyMathBackend>(DT, M, D, q, f, u, space);

    // Strategy + state
    PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
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
}

void run_mundy_math_mixed_congruent_test(const auto& test) {
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
  vector_t x{}, grad{}, x_tmp{}, grad_tmp{};

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
  auto pgd = make_pgd_solution_strategy(cfg);
  auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);

  // Solve
  auto result = solve_mixed_cqpp(mixed_cqpp, pgd, pgd_state);

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LE(result.num_iters, cfg.max_iters);
  for (size_t i = 0; i < vector_t::size; ++i) {
    EXPECT_NEAR(x[i], x_exact[i], 10 * cfg.tol);
  }
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
void run_kokkos_test(const auto& test) {
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

  Kokkos::deep_copy(x, 99.99);  // use a bad initial guess to force more iterations

  // Build the problem
  const auto cqpp = make_cqpp<KokkosBackend<decltype(exec_space)>>(A, q, space);

  // Strategy + state
  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
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

  Kokkos::deep_copy(x, 99.99);  // use a bad initial guess to force more iterations

  // Build quadratic form operator + user-owned workspace, then the problem
  const auto A = make_quadratic_form<KokkosBackend<decltype(exec_space)>>(DT, M, D);
  auto workspace = A.make_workspace(f, u);  // intermediate variables f = D x, u = M f
  const auto cqpp = make_cqpp<KokkosBackend<decltype(exec_space)>>(A, q, space, workspace);

  // Strategy + state
  PGDConfig<double> cfg{.max_iters = 1000, .tol = 1e-6};
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

void run_kokkos_mixed_congruent_test(const auto& test) {
  std::cout << "Running test: " << test.name() << std::endl;

  auto exec_space = test.get_exec_space();

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
  ASSERT_EQ(M.extent(0), M.extent(1)) << "M should be square";
  ASSERT_EQ(DT.extent(0), D.extent(1)) << "DT and D are supposed to be transposes of each other";
  ASSERT_EQ(DT.extent(1), D.extent(0)) << "DT and D are supposed to be transposes of each other";
  ASSERT_EQ(DT.extent(1), M.extent(0)) << "DT * M should be well-defined";
  ASSERT_EQ(M.extent(1), D.extent(0)) << "M * D should be well-defined";
  ASSERT_EQ(M.extent(1), B.extent(0)) << "M * B should be well-defined";
  ASSERT_EQ(BT.extent(1), M.extent(0)) << "B^T * M should be well-defined";

  ASSERT_EQ(D.extent(1), x_exact.extent(0)) << "D * x should be well-defined";
  ASSERT_EQ(B.extent(1), y_exact.extent(0)) << "B * y should be well-defined";
  ASSERT_EQ(S.extent(1), BT.extent(0)) << "S * BT should be well-defined";
  ASSERT_EQ(B.extent(1), S.extent(0)) << "B * S should be well-defined";
  ASSERT_EQ(q.extent(0), x_exact.extent(0)) << "q should be same size as x_exact";
  ASSERT_EQ(b.extent(0), y_exact.extent(0)) << "b should be same size as y_exact";

  // Build the problem
  const auto mixed_cqpp = make_mixed_cqpp<KokkosBackend<decltype(exec_space)>>(DT, M, D, q, B, S, BT, b, space);

  auto DT_op = mixed_cqpp.DT();
  auto M_op = mixed_cqpp.M();
  auto f_b = mixed_cqpp.f_b();

  using backend_t = KokkosBackend<decltype(exec_space)>;
  ASSERT_EQ(backend_t::domain_size(M_op), backend_t::size(f_b)) << "M and f_b should be compatible for multiplication";
  ASSERT_EQ(backend_t::domain_size(DT_op), backend_t::range_size(M_op)) << "DT and M should be compatible for DT * M";
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
  using math_backend::congruent::CongruentLCPWrapper;
  auto test_cases = std::make_tuple(CongruentLCPWrapper{math_backend::UnconstrainedSPD1Problem{}},          //
                                    CongruentLCPWrapper{math_backend::InactiveBoxConstrainedSPDProblem{}},  //
                                    CongruentLCPWrapper{math_backend::ActiveBoxConstrainedSPDProblem{}},    //
                                    CongruentLCPWrapper{math_backend::RandomLCP<3>{}},                      //
                                    CongruentLCPWrapper{math_backend::RandomLCP<7>{}});
  std::apply([](auto&&... test_case) { (run_mundy_math_congruent_test(test_case), ...); }, test_cases);
}

TEST(Convex, MundyMathMixedCongruentAnalyticalSolutions) {
  // Template params are <NX, NY, NZ>; make_D_full_rank requires NZ >= NX.
  auto test_cases = std::make_tuple(math_backend::mixed::RandomMixedCongruentCCQP<4, 3, 5>{},  //
                                    math_backend::mixed::RandomMixedCongruentCCQP<3, 4, 5>{});
  std::apply([](auto&&... test_case) { (run_mundy_math_mixed_congruent_test(test_case), ...); }, test_cases);
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
                                    kokkos_backend::RandomLCP{3},                        //
                                    kokkos_backend::RandomLCP{7},                        //
                                    kokkos_backend::RandomLCP{200});
  std::apply([](auto&&... test_case) { (run_kokkos_test(test_case), ...); }, test_cases);
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
                                    CongruentLCPWrapper{kokkos_backend::RandomLCP{3}},                        //
                                    CongruentLCPWrapper{kokkos_backend::RandomLCP{7}},                        //
                                    CongruentLCPWrapper{kokkos_backend::RandomLCP{200}});
  std::apply([](auto&&... test_case) { (run_kokkos_congruent_test(test_case), ...); }, test_cases);
}

TEST(Convex, KokkosMixedCongruentAnalyticalSolutions) {
#if !defined(KOKKOSKERNELS_ENABLE_TPL_LAPACK) && !defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER) && !defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  GTEST_SKIP() << "KokkosLapack::gesv requires LAPACK, CUSOLVER, ROCSOLVER, or MAGMA.";
#endif
  auto test_cases = std::make_tuple(kokkos_backend::mixed::RandomMixedCongruentCCQP<5, 4, 3>{},  //
                                    kokkos_backend::mixed::RandomMixedCongruentCCQP<3, 4, 5>{});
  std::apply([](auto&&... test_case) { (run_kokkos_mixed_congruent_test(test_case), ...); }, test_cases);
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

}  // namespace

}  // namespace mundy
