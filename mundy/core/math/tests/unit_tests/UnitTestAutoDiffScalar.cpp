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
#include <gtest/gtest.h>  // for TEST, EXPECT_*

// C++ core libs
#include <cmath>        // for std::sqrt, std::sin, ...
#include <type_traits>  // for std::is_same_v
#include <utility>      // for std::declval

// Mundy libs
#include <mundy_math/AutoDiffScalar.hpp>        // for mundy::AutoDiffScalar
#include <mundy_math/Matrix.hpp>                // for mundy::Matrix2, mundy::determinant, mundy::inverse
#include <mundy_math/Matrix3.hpp>               // for mundy::Matrix3, mundy::cholesky
#include <mundy_math/NumTraits.hpp>             // for mundy::NumTraits
#include <mundy_math/Quaternion.hpp>            // for mundy::Quaternion, mundy::slerp, mundy::axis_angle_to_quaternion
#include <mundy_math/Scalar.hpp>                // for mundy::Scalar, mundy::AScalar, mundy::is_close
#include <mundy_math/ScalarBinaryOpTraits.hpp>  // for mundy::scalar_*_result_t, mundy::ScalarBinaryOpSupported
#include <mundy_math/Vector.hpp>                // for mundy::Vector, mundy::dot, mundy::two_norm
#include <mundy_math/Vector3.hpp>               // for mundy::Vector3, mundy::cross
#include <mundy_math/cmath.hpp>                 // for mundy::passive_scalar_t

namespace mundy {

namespace {

// Two independent variables, derivative seeded in slot 0 (x) and slot 1 (y).
using AD = AutoDiffScalar<double, 2>;

// Compile-time: AutoDiffScalar satisfies the MundyMath scalar contract without touching namespace std.
static_assert(std::is_same_v<passive_scalar_t<AD>, double>);
static_assert(std::is_same_v<scalar_product_result_t<AD, double>, AD>);
static_assert(std::is_same_v<scalar_product_result_t<double, AD>, AD>);
static_assert(std::is_same_v<scalar_product_result_t<AD, AD>, AD>);
static_assert(std::is_same_v<scalar_quotient_result_t<AD, double>, AD>);
static_assert(ScalarBinaryOpSupported<AD, double, scalar_binary_op::product>);
static_assert(ScalarBinaryOpSupported<AD, AD, scalar_binary_op::product>);
static_assert(NumTraits<AD>::IsInteger == false);
static_assert(NumTraits<AD>::RequireInitialization == true);
static_assert(std::is_same_v<NumTraits<AD>::Real, AD>);

constexpr double kTol = 1e-12;

TEST(AutoDiffScalar, NumTraitsUsePassiveValues) {
  // Numeric facts about an AD scalar are the passive (double) facts.
  EXPECT_EQ(NumTraits<AD>::epsilon(), NumTraits<double>::epsilon());
  EXPECT_EQ(NumTraits<AD>::dummy_precision(), NumTraits<double>::dummy_precision());
}

TEST(AutoDiffScalar, ArithmeticChainRules) {
  const double xv = 3.0, yv = 4.0;
  AD x(xv, 0), y(yv, 1);

  AD prod = x * y;  // d/dx = y, d/dy = x
  EXPECT_NEAR(prod.value(), xv * yv, kTol);
  EXPECT_NEAR(prod.derivatives()[0], yv, kTol);
  EXPECT_NEAR(prod.derivatives()[1], xv, kTol);

  AD quot = x / y;  // d/dx = 1/y, d/dy = -x/y^2
  EXPECT_NEAR(quot.value(), xv / yv, kTol);
  EXPECT_NEAR(quot.derivatives()[0], 1.0 / yv, kTol);
  EXPECT_NEAR(quot.derivatives()[1], -xv / (yv * yv), kTol);

  AD diff = x + y - x;  // value yv, d/dx = 0, d/dy = 1
  EXPECT_NEAR(diff.value(), yv, kTol);
  EXPECT_NEAR(diff.derivatives()[0], 0.0, kTol);
  EXPECT_NEAR(diff.derivatives()[1], 1.0, kTol);
}

TEST(AutoDiffScalar, MixedWithPassive) {
  const double xv = 3.0;
  AD x(xv, 0);

  AD a = 2.0 * x;  // d/dx = 2
  EXPECT_NEAR(a.value(), 2.0 * xv, kTol);
  EXPECT_NEAR(a.derivatives()[0], 2.0, kTol);

  AD b = 2.0 / x;  // d/dx = -2/x^2
  EXPECT_NEAR(b.value(), 2.0 / xv, kTol);
  EXPECT_NEAR(b.derivatives()[0], -2.0 / (xv * xv), kTol);

  AD c = x + 5.0;  // derivative unchanged
  EXPECT_NEAR(c.value(), xv + 5.0, kTol);
  EXPECT_NEAR(c.derivatives()[0], 1.0, kTol);
}

TEST(AutoDiffScalar, CompoundAssignment) {
  AD f(3.0, 0);
  f += AD(4.0, 1);  // value 7, d/dx 1, d/dy 1
  f *= 2.0;         // value 14, d/dx 2, d/dy 2
  EXPECT_NEAR(f.value(), 14.0, kTol);
  EXPECT_NEAR(f.derivatives()[0], 2.0, kTol);
  EXPECT_NEAR(f.derivatives()[1], 2.0, kTol);
}

TEST(AutoDiffScalar, MathFunctionDerivatives) {
  const double xv = 3.0, yv = 4.0;
  AD x(xv, 0), y(yv, 1);

  AD s = sqrt(x);
  EXPECT_NEAR(s.value(), std::sqrt(xv), kTol);
  EXPECT_NEAR(s.derivatives()[0], 0.5 / std::sqrt(xv), kTol);
  EXPECT_NEAR(s.derivatives()[1], 0.0, kTol);

  AD e = exp(x);
  EXPECT_NEAR(e.derivatives()[0], std::exp(xv), 1e-9);

  AD l = log(x);
  EXPECT_NEAR(l.derivatives()[0], 1.0 / xv, kTol);

  AD sn = sin(x);
  EXPECT_NEAR(sn.derivatives()[0], std::cos(xv), kTol);

  AD cs = cos(x);
  EXPECT_NEAR(cs.derivatives()[0], -std::sin(xv), kTol);

  AD p = pow(x, 3.0);  // d/dx = 3 x^2
  EXPECT_NEAR(p.value(), std::pow(xv, 3.0), 1e-9);
  EXPECT_NEAR(p.derivatives()[0], 3.0 * xv * xv, 1e-9);

  AD at = atan2(y, x);  // d/dx = -y/(x^2+y^2), d/dy = x/(x^2+y^2)
  const double r2 = xv * xv + yv * yv;
  EXPECT_NEAR(at.value(), std::atan2(yv, xv), kTol);
  EXPECT_NEAR(at.derivatives()[0], -yv / r2, kTol);
  EXPECT_NEAR(at.derivatives()[1], xv / r2, kTol);
}

TEST(AutoDiffScalar, AbsBranchesOnValueSign) {
  AD pos(3.0, 0);
  AD ap = abs(pos);  // x > 0 -> +1
  EXPECT_NEAR(ap.value(), 3.0, kTol);
  EXPECT_NEAR(ap.derivatives()[0], 1.0, kTol);

  AD neg(-3.0, 0);
  AD an = abs(neg);  // x < 0 -> -1
  EXPECT_NEAR(an.value(), 3.0, kTol);
  EXPECT_NEAR(an.derivatives()[0], -1.0, kTol);
}

TEST(AutoDiffScalar, MinMaxSelectOperand) {
  AD x(3.0, 0), y(4.0, 1);
  AD m = min(x, y);  // x smaller -> carries x's derivatives
  EXPECT_NEAR(m.value(), 3.0, kTol);
  EXPECT_NEAR(m.derivatives()[0], 1.0, kTol);
  EXPECT_NEAR(m.derivatives()[1], 0.0, kTol);

  AD c = max(x, 10.0);  // passive larger -> constant (zero derivative)
  EXPECT_NEAR(c.value(), 10.0, kTol);
  EXPECT_NEAR(c.derivatives()[0], 0.0, kTol);
}

TEST(AutoDiffScalar, CompositeMatchesFiniteDifference) {
  const double xv = 3.0;
  auto g = [](double t) { return std::sin(t) * std::sqrt(t) + std::exp(t / 5.0); };

  AD t(xv, 0);
  AD f = sin(t) * sqrt(t) + exp(t / 5.0);
  const double h = 1e-6;
  const double fd = (g(xv + h) - g(xv - h)) / (2.0 * h);

  EXPECT_NEAR(f.value(), g(xv), 1e-12);
  EXPECT_NEAR(f.derivatives()[0], fd, 1e-5);
}

// ----------------------------------------------------------------------------------------------------
// Scalar<AutoDiffScalar> — the owning scalar wrapper carries an AD value and preserves derivatives
// through its operators. value() returns the AD; value().value() / value().derivatives() reach the
// underlying number and its gradient.
// ----------------------------------------------------------------------------------------------------

using ADScalar = Scalar<AD>;

// AScalar admits AD (ValidScalarType) and its operators yield AD results via ScalarBinaryOpTraits.
static_assert(ValidScalarType<AD>);
static_assert(std::is_same_v<decltype(std::declval<ADScalar>() + std::declval<ADScalar>()), ADScalar>);
static_assert(std::is_same_v<decltype(std::declval<ADScalar>() * std::declval<AD>()), ADScalar>);
static_assert(std::is_same_v<decltype(std::declval<Scalar<double>>() + std::declval<ADScalar>()), ADScalar>);

// Atomics are constrained to arithmetic scalars, so they are unavailable for AD.
template <typename T>
concept HasAtomicAdd = requires(T* p, T v) { mundy::atomic_add(p, v); };
static_assert(HasAtomicAdd<double>);
static_assert(!HasAtomicAdd<AD>);

TEST(AutoDiffScalarInScalar, ArithmeticPreservesDerivatives) {
  ADScalar sx(AD(3.0, 0));
  ADScalar sy(AD(4.0, 1));

  auto prod = sx * sy;  // value 12, d/dx = 4, d/dy = 3
  EXPECT_NEAR(prod.value().value(), 12.0, kTol);
  EXPECT_NEAR(prod.value().derivatives()[0], 4.0, kTol);
  EXPECT_NEAR(prod.value().derivatives()[1], 3.0, kTol);

  auto quot = sx / sy;  // value 0.75, d/dx = 1/4, d/dy = -3/16
  EXPECT_NEAR(quot.value().value(), 0.75, kTol);
  EXPECT_NEAR(quot.value().derivatives()[0], 0.25, kTol);
  EXPECT_NEAR(quot.value().derivatives()[1], -3.0 / 16.0, kTol);
}

TEST(AutoDiffScalarInScalar, MixedWithPassiveAndPrimitiveScalar) {
  ADScalar sx(AD(3.0, 0));

  auto a = 2.0 * sx;  // d/dx = 2
  EXPECT_NEAR(a.value().value(), 6.0, kTol);
  EXPECT_NEAR(a.value().derivatives()[0], 2.0, kTol);

  auto b = sx + 1.0;  // derivative unchanged
  EXPECT_NEAR(b.value().value(), 4.0, kTol);
  EXPECT_NEAR(b.value().derivatives()[0], 1.0, kTol);

  auto c = Scalar<double>(2.0) + sx;  // Scalar<double> + Scalar<AD> -> Scalar<AD>
  EXPECT_NEAR(c.value().value(), 5.0, kTol);
  EXPECT_NEAR(c.value().derivatives()[0], 1.0, kTol);
}

TEST(AutoDiffScalarInScalar, IsCloseComparesValues) {
  ADScalar sx(AD(3.0, 0));
  ADScalar near_sx(AD(3.0 + 1e-16, 1));  // same value, different derivative
  ADScalar far_sx(AD(4.0, 1));
  EXPECT_TRUE(is_close(sx, near_sx));  // closeness is about value, not derivative
  EXPECT_FALSE(is_close(sx, far_sx));
}

// ----------------------------------------------------------------------------------------------------
// Vector<AutoDiffScalar> — components carry derivatives through vector algebra (dot, norm, cross, ...).
// ----------------------------------------------------------------------------------------------------

TEST(AutoDiffScalarInVector, DotAndNorm) {
  AD x(2.0, 0), y(3.0, 1);
  Vector<AD, 2> u(x, y);

  auto d = dot(u, u);  // x^2 + y^2 = 13; gradient (2x, 2y)
  EXPECT_NEAR(d.value(), 13.0, kTol);
  EXPECT_NEAR(d.derivatives()[0], 4.0, kTol);
  EXPECT_NEAR(d.derivatives()[1], 6.0, kTol);

  auto nrm = two_norm(u);  // sqrt(13); gradient u / |u|
  EXPECT_NEAR(nrm.value(), std::sqrt(13.0), 1e-9);
  EXPECT_NEAR(nrm.derivatives()[0], 2.0 / std::sqrt(13.0), 1e-9);
  EXPECT_NEAR(nrm.derivatives()[1], 3.0 / std::sqrt(13.0), 1e-9);
}

TEST(AutoDiffScalarInVector, ScalarBroadcastAndCloseness) {
  AD x(2.0, 0), y(3.0, 1);
  Vector<AD, 2> u(x, y);

  auto a = u * 2.0;  // passive scale: [0] = 2x
  EXPECT_NEAR(a[0].value(), 4.0, kTol);
  EXPECT_NEAR(a[0].derivatives()[0], 2.0, kTol);

  auto b = u * x;  // AD scale: [0] = x^2, d/dx = 2x
  EXPECT_NEAR(b[0].value(), 4.0, kTol);
  EXPECT_NEAR(b[0].derivatives()[0], 4.0, kTol);

  EXPECT_TRUE(is_close(u, u));
}

TEST(AutoDiffScalarInVector, CrossProduct) {
  AD x(2.0, 0), y(3.0, 1);
  Vector3<AD> p(x, y, AD(0.0)), q(AD(0.0), AD(0.0), AD(1.0));
  auto c = cross(p, q);  // (y, -x, 0)
  EXPECT_NEAR(c[0].value(), 3.0, kTol);
  EXPECT_NEAR(c[0].derivatives()[1], 1.0, kTol);  // d(y)/dy
  EXPECT_NEAR(c[1].value(), -2.0, kTol);
  EXPECT_NEAR(c[1].derivatives()[0], -1.0, kTol);  // d(-x)/dx
}

// Reductions promote through NumTraits: NonInteger of AD is AD, so they return AD (no spurious double).
static_assert(std::is_same_v<decltype(mean(std::declval<Vector3<AD>>())), AD>);
static_assert(std::is_same_v<decltype(variance(std::declval<Vector3<AD>>())), AD>);
static_assert(std::is_same_v<decltype(stddev(std::declval<Vector3<AD>>())), AD>);

TEST(AutoDiffScalarInVector, ReductionsCarryDerivatives) {
  AD x(2.0, 0);
  Vector3<AD> u(x, 2.0 * x, 3.0 * x);  // elements x, 2x, 3x

  auto m = mean(u);  // (x + 2x + 3x)/3 = 2x
  EXPECT_NEAR(m.value(), 4.0, kTol);
  EXPECT_NEAR(m.derivatives()[0], 2.0, kTol);

  auto v = variance(u);  // population variance = 2 x^2 / 3, d/dx = 4x/3
  EXPECT_NEAR(v.value(), 8.0 / 3.0, kTol);
  EXPECT_NEAR(v.derivatives()[0], 8.0 / 3.0, kTol);

  auto sd = stddev(u);  // sqrt(2/3) * x, d/dx = sqrt(2/3)
  EXPECT_NEAR(sd.value(), std::sqrt(8.0 / 3.0), 1e-9);
  EXPECT_NEAR(sd.derivatives()[0], std::sqrt(2.0 / 3.0), 1e-9);

  // Finite-difference cross-check of the full reduction pipeline.
  auto stddev_of = [](double t) {
    const double a = t, b = 2.0 * t, c = 3.0 * t;
    const double mu = (a + b + c) / 3.0;
    return std::sqrt(((a - mu) * (a - mu) + (b - mu) * (b - mu) + (c - mu) * (c - mu)) / 3.0);
  };
  const double h = 1e-6;
  const double fd = (stddev_of(2.0 + h) - stddev_of(2.0 - h)) / (2.0 * h);
  EXPECT_NEAR(sd.derivatives()[0], fd, 1e-5);
}

// ----------------------------------------------------------------------------------------------------
// Matrix<AutoDiffScalar> — entries carry derivatives through matrix algebra (matmul, matrix-vector,
// determinant, inverse, reductions) and through the Matrix3 Cholesky factorization.
// ----------------------------------------------------------------------------------------------------

// AMatrix admits AD; reductions promote through NumTraits (NonInteger of AD is AD).
static_assert(is_matrix_v<AMatrix<AD, 2, 2>>);
static_assert(std::is_same_v<decltype(mean(std::declval<Matrix2<AD>>())), AD>);
static_assert(std::is_same_v<decltype(variance(std::declval<Matrix2<AD>>())), AD>);
static_assert(std::is_same_v<decltype(stddev(std::declval<Matrix2<AD>>())), AD>);
// More-permissive-than-Eigen mixed primitives still resolve: Matrix3f * Vector3d -> Vector3d.
static_assert(
    std::is_same_v<decltype(std::declval<Matrix3<float>>() * std::declval<Vector3<double>>()), Vector3<double>>);

TEST(AutoDiffScalarInMatrix, MatMulCarriesDerivatives) {
  AD x(2.0, 0), y(5.0, 1);
  Matrix2<AD> A(x, AD(2.0), AD(3.0), AD(4.0));  // [[x, 2], [3, 4]]
  Matrix2<AD> B(y, AD(1.0), AD(0.0), AD(5.0));  // [[y, 1], [0, 5]]
  auto C = A * B;                               // [[x*y, x + 10], [3y, 23]]
  EXPECT_NEAR(C(0, 0).value(), 10.0, kTol);
  EXPECT_NEAR(C(0, 0).derivatives()[0], 5.0, kTol);  // d/dx = y
  EXPECT_NEAR(C(0, 0).derivatives()[1], 2.0, kTol);  // d/dy = x
  EXPECT_NEAR(C(0, 1).value(), 12.0, kTol);
  EXPECT_NEAR(C(0, 1).derivatives()[0], 1.0, kTol);
  EXPECT_NEAR(C(1, 0).value(), 15.0, kTol);
  EXPECT_NEAR(C(1, 0).derivatives()[1], 3.0, kTol);
  EXPECT_NEAR(C(1, 1).value(), 23.0, kTol);
  EXPECT_NEAR(C(1, 1).derivatives()[0], 0.0, kTol);
}

TEST(AutoDiffScalarInMatrix, MatrixVectorProduct) {
  AD x(2.0, 0), y(5.0, 1);
  Matrix2<AD> A(x, AD(2.0), AD(3.0), AD(4.0));  // [[x, 2], [3, 4]]
  Vector2<AD> v(y, AD(1.0));                    // [y, 1]
  auto w = A * v;                               // [x*y + 2, 3y + 4]
  EXPECT_NEAR(w[0].value(), 12.0, kTol);
  EXPECT_NEAR(w[0].derivatives()[0], 5.0, kTol);  // d/dx = y
  EXPECT_NEAR(w[0].derivatives()[1], 2.0, kTol);  // d/dy = x
  EXPECT_NEAR(w[1].value(), 19.0, kTol);
  EXPECT_NEAR(w[1].derivatives()[1], 3.0, kTol);  // d/dy = 3
}

TEST(AutoDiffScalarInMatrix, ScalarMultiplyADAndPassive) {
  AD x(2.0, 0);
  Matrix2<AD> A(x, AD(1.0), AD(2.0), AD(3.0));
  auto P = A * 2.0;  // passive scale keeps the derivative
  EXPECT_NEAR(P(0, 0).value(), 4.0, kTol);
  EXPECT_NEAR(P(0, 0).derivatives()[0], 2.0, kTol);
  auto Q = A * x;  // AD scale: Q(0,0) = x^2
  EXPECT_NEAR(Q(0, 0).value(), 4.0, kTol);
  EXPECT_NEAR(Q(0, 0).derivatives()[0], 4.0, kTol);  // d(x^2)/dx = 2x
}

TEST(AutoDiffScalarInMatrix, DeterminantAndInverse) {
  AD x(3.0, 0), y(4.0, 1);
  Matrix2<AD> A(x, AD(1.0), AD(2.0), y);  // [[x, 1], [2, y]], det = x*y - 2
  auto d = determinant(A);
  EXPECT_NEAR(d.value(), 10.0, kTol);
  EXPECT_NEAR(d.derivatives()[0], 4.0, kTol);  // d/dx = y
  EXPECT_NEAR(d.derivatives()[1], 3.0, kTol);  // d/dy = x

  // inverse(A) * A is the identity, which is constant -> zero derivatives.
  auto Id = inverse(A) * A;
  EXPECT_NEAR(Id(0, 0).value(), 1.0, 1e-9);
  EXPECT_NEAR(Id(0, 0).derivatives()[0], 0.0, 1e-9);
  EXPECT_NEAR(Id(0, 0).derivatives()[1], 0.0, 1e-9);
  EXPECT_NEAR(Id(0, 1).value(), 0.0, 1e-9);
  EXPECT_NEAR(Id(1, 0).value(), 0.0, 1e-9);
  EXPECT_NEAR(Id(1, 1).value(), 1.0, 1e-9);
}

TEST(AutoDiffScalarInMatrix, ReductionsAndFrobenius) {
  AD x(2.0, 0);
  Matrix2<AD> A(x, 2.0 * x, 3.0 * x, 4.0 * x);  // entries x, 2x, 3x, 4x
  auto m = mean(A);                             // 10x/4 = 2.5x
  EXPECT_NEAR(m.value(), 5.0, kTol);
  EXPECT_NEAR(m.derivatives()[0], 2.5, kTol);
  auto fr = frobenius_inner_product(A, A);  // (1 + 4 + 9 + 16) x^2 = 30 x^2
  EXPECT_NEAR(fr.value(), 120.0, kTol);
  EXPECT_NEAR(fr.derivatives()[0], 120.0, kTol);  // d/dx = 60x
}

TEST(AutoDiffScalarInMatrix, Matrix3CholeskyDerivative) {
  AD x(4.0, 0);
  // SPD matrix; only the (2,2) corner depends on x.
  Matrix3<AD> A(AD(2.0), AD(1.0), AD(0.0),  //
                AD(1.0), AD(2.0), AD(0.0),  //
                AD(0.0), AD(0.0), x);
  auto L = cholesky(A);                               // L(2,2) = sqrt(x)
  EXPECT_NEAR(L(2, 2).value(), 2.0, 1e-9);            // sqrt(4)
  EXPECT_NEAR(L(2, 2).derivatives()[0], 0.25, 1e-9);  // 0.5 / sqrt(x)
  EXPECT_NEAR(L(0, 0).derivatives()[0], 0.0, 1e-12);  // constant block
}

// ----------------------------------------------------------------------------------------------------
// Quaternion<AutoDiffScalar> — components carry derivatives through quaternion algebra (Hamilton product,
// rotation, dot/norm, inverse, slerp, axis-angle).
// ----------------------------------------------------------------------------------------------------

static_assert(is_quaternion_v<AQuaternion<AD>>);
// More-permissive-than-Eigen mixed primitives still resolve: Quaterniond * Quaternionf -> Quaternion<double>.
static_assert(std::is_same_v<decltype(std::declval<Quaterniond>() * std::declval<Quaternionf>()), Quaternion<double>>);

TEST(AutoDiffScalarInQuaternion, HamiltonProductCarriesDerivatives) {
  AD x(2.0, 0), y(3.0, 1);
  Quaternion<AD> q1(x, AD(1.0), AD(0.0), AD(0.0));  // (x, 1, 0, 0)
  Quaternion<AD> q2(y, AD(0.0), AD(1.0), AD(0.0));  // (y, 0, 1, 0)
  auto p = q1 * q2;                                 // (x*y, y, x, 1)
  EXPECT_NEAR(p.w().value(), 6.0, kTol);
  EXPECT_NEAR(p.w().derivatives()[0], 3.0, kTol);  // d/dx = y
  EXPECT_NEAR(p.w().derivatives()[1], 2.0, kTol);  // d/dy = x
  EXPECT_NEAR(p.x().value(), 3.0, kTol);
  EXPECT_NEAR(p.x().derivatives()[1], 1.0, kTol);
  EXPECT_NEAR(p.y().value(), 2.0, kTol);
  EXPECT_NEAR(p.y().derivatives()[0], 1.0, kTol);
  EXPECT_NEAR(p.z().value(), 1.0, kTol);
  EXPECT_NEAR(p.z().derivatives()[0], 0.0, kTol);
}

TEST(AutoDiffScalarInQuaternion, ScalarDotNorm) {
  AD x(2.0, 0);
  Quaternion<AD> q(x, 2.0 * x, AD(0.0), AD(0.0));  // (x, 2x, 0, 0)
  auto qs = q * 3.0;                               // passive scale keeps derivative
  EXPECT_NEAR(qs.w().value(), 6.0, kTol);
  EXPECT_NEAR(qs.w().derivatives()[0], 3.0, kTol);
  auto qa = q * x;  // AD scale: w = x^2
  EXPECT_NEAR(qa.w().value(), 4.0, kTol);
  EXPECT_NEAR(qa.w().derivatives()[0], 4.0, kTol);  // d(x^2)/dx = 2x
  auto d = dot(q, q);                               // 5 x^2
  EXPECT_NEAR(d.value(), 20.0, kTol);
  EXPECT_NEAR(d.derivatives()[0], 20.0, kTol);  // d/dx = 10x
  auto n = norm(q);                             // sqrt(5) x
  EXPECT_NEAR(n.value(), std::sqrt(20.0), 1e-9);
  EXPECT_NEAR(n.derivatives()[0], std::sqrt(5.0), 1e-9);
}

TEST(AutoDiffScalarInQuaternion, InverseIdentityAndNormalize) {
  AD x(2.0, 0);
  Quaternion<AD> q(x, AD(1.0), AD(0.5), AD(0.25));
  auto id = q * inverse(q);  // identity (1,0,0,0), constant -> zero derivatives
  EXPECT_NEAR(id.w().value(), 1.0, 1e-9);
  EXPECT_NEAR(id.w().derivatives()[0], 0.0, 1e-9);
  EXPECT_NEAR(id.x().value(), 0.0, 1e-9);
  EXPECT_NEAR(id.y().value(), 0.0, 1e-9);
  EXPECT_NEAR(id.z().value(), 0.0, 1e-9);

  auto n_unit = norm(normalize(q));  // unit norm is identically 1 -> zero derivative
  EXPECT_NEAR(n_unit.value(), 1.0, 1e-9);
  EXPECT_NEAR(n_unit.derivatives()[0], 0.0, 1e-9);

  EXPECT_TRUE(is_close(q, q));  // closeness compares values
  EXPECT_FALSE(is_close(q, Quaternion<AD>(x + AD(1.0), AD(1.0), AD(0.5), AD(0.25))));
}

TEST(AutoDiffScalarInQuaternion, AxisAngleRotationMatchesFiniteDifference) {
  // q(theta) = rotation about z; rotate (1,0,0). Derivative checked against a finite difference of the
  // same double pipeline (exercises the gateway-routed sin/cos; independent of rotation convention).
  const double th = 0.7;
  AD theta(th, 0);
  Vector3<AD> axis(AD(0.0), AD(0.0), AD(1.0));
  Vector3<AD> v(AD(1.0), AD(0.0), AD(0.0));
  auto r = axis_angle_to_quaternion(axis, theta) * v;

  auto rot_x = [](double t) {
    Vector3<double> a(0.0, 0.0, 1.0), w(1.0, 0.0, 0.0);
    return (axis_angle_to_quaternion(a, t) * w)[0];
  };
  const double h = 1e-6;
  const double fd = (rot_x(th + h) - rot_x(th - h)) / (2.0 * h);
  EXPECT_NEAR(r[0].derivatives()[0], fd, 1e-6);
}

TEST(AutoDiffScalarInQuaternion, SlerpEndpointCarriesDerivative) {
  Quaternion<AD> qa(AD(1.0, 0), AD(0.0), AD(0.0), AD(0.0));
  Quaternion<AD> qb(AD(0.0), AD(1.0, 1), AD(0.0), AD(0.0));
  auto s0 = slerp(qa, qb, AD(0.0));  // = qa
  EXPECT_NEAR(s0.w().value(), 1.0, 1e-9);
  EXPECT_NEAR(s0.w().derivatives()[0], 1.0, 1e-9);  // carries qa's seeded derivative
  EXPECT_NEAR(s0.x().value(), 0.0, 1e-9);
}

}  // namespace

}  // namespace mundy
