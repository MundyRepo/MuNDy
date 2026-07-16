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

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*

// C++ core
#include <sstream>
#include <utility>

// Mundy
#include <mundy_math/Matrix.hpp>  // for mundy::Matrix3d
#include <mundy_math/Vector.hpp>  // for mundy::Vector3d
#include <mundy_math/solver_backends.hpp>

namespace mundy {

namespace {

// A mock "vector" type that does not satisfy VectorBackend against any real backend (no size/axpby/dot/deep_copy
// defined for it anywhere) -- used to prove the concept actually rejects bad input, not just accepts good input.
struct NotAVector {};

// A mock operator with no apply/domain_size/range_size member and no operator*, so it does not satisfy
// LinearOperator against any real backend.
struct NotAnOperator {};

// A mock operator that provides the fused scaled-apply member (apply(alpha, x, beta, y)).
struct MockScaledOp {
  void apply(double alpha, const Vector3d& x, double beta, Vector3d& y) const {
    y = alpha * x + beta * y;
  }
};

// A mock operator that only provides the plain apply(x, y) -- no scaled-apply member.
struct MockUnscaledOp {
  void apply(const Vector3d& x, Vector3d& y) const {
    y = x;
  }
};

//! \name VectorBackend concept
//@{
static_assert(VectorBackend<MundyMathBackend, Vector3d>,
             "Vector3d must satisfy VectorBackend under MundyMathBackend");
static_assert(!VectorBackend<MundyMathBackend, NotAVector>,
             "NotAVector must NOT satisfy VectorBackend under MundyMathBackend");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
static_assert(VectorBackend<KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                    Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
             "Kokkos::View<double*> must satisfy VectorBackend under KokkosBackend");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name LinearOperator concept
//@{
static_assert(LinearOperator<MundyMathBackend, Matrix3d, Vector3d, Vector3d>,
             "Matrix3d must satisfy LinearOperator under MundyMathBackend");
static_assert(!LinearOperator<MundyMathBackend, NotAnOperator, Vector3d, Vector3d>,
             "NotAnOperator must NOT satisfy LinearOperator under MundyMathBackend");
//@}

//! \name HasScaledApplyMember concept
//@{
static_assert(HasScaledApplyMember<MockScaledOp, double, Vector3d, Vector3d>,
             "MockScaledOp must satisfy HasScaledApplyMember");
static_assert(!HasScaledApplyMember<MockUnscaledOp, double, Vector3d, Vector3d>,
             "MockUnscaledOp must NOT satisfy HasScaledApplyMember");
//@}

}  // namespace

TEST(SolverBackends, MundyMathBackendDotMatchesFreeDot) {
  const Vector3d a{1.0, 2.0, 3.0};
  const Vector3d b{4.0, -5.0, 6.0};
  const double expected = dot(a, b);
  const double actual = MundyMathBackend::dot<double>(a, b);
  EXPECT_DOUBLE_EQ(actual, expected);
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
TEST(SolverBackends, KokkosBackendDotMatchesHandComputed) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;
  using view_t = Kokkos::View<double*, mem_space>;

  view_t x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), 3);
  view_t y(Kokkos::view_alloc(Kokkos::WithoutInitializing, "y"), 3);
  auto x_host = Kokkos::create_mirror_view(x);
  auto y_host = Kokkos::create_mirror_view(y);
  x_host(0) = 1.0;
  x_host(1) = 2.0;
  x_host(2) = 3.0;
  y_host(0) = 4.0;
  y_host(1) = -5.0;
  y_host(2) = 6.0;
  Kokkos::deep_copy(x, x_host);
  Kokkos::deep_copy(y, y_host);

  using backend_t = KokkosBackend<exec_space>;
  const double actual = backend_t::dot<double>(x, y);
  EXPECT_DOUBLE_EQ(actual, 1.0 * 4.0 + 2.0 * -5.0 + 3.0 * 6.0);
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

}  // namespace mundy
