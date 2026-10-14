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

/// \file UnitTestDoubleDouble.cpp
/// \brief DoubleDouble (DoubleDouble.hpp) against 70-digit references, and as a scalar throughout MundyMath.
///
/// Anchors: reference values are the exact (hi, lo) pairs nearest to 70-digit decimal evaluations, so a correct
/// result differs from them by a few units of 2^-104 (about 5e-32) relative.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>        // for std::isnan, std::isinf, std::signbit
#include <cstdint>      // for std::int64_t, std::uint64_t
#include <type_traits>  // for std::is_same_v, std::is_trivially_copyable_v

// Mundy
#include <mundy_math/AutoDiffScalar.hpp>        // for mundy::AutoDiffScalar
#include <mundy_math/DoubleDouble.hpp>          // for mundy::DoubleDouble
#include <mundy_math/Matrix.hpp>                // for mundy::Matrix, mundy::inverse
#include <mundy_math/NumTraits.hpp>             // for mundy::NumTraits, mundy::is_passive_scalar_v
#include <mundy_math/Quaternion.hpp>            // for mundy::Quaternion, mundy::axis_angle_to_quaternion
#include <mundy_math/Scalar.hpp>                // for mundy::Scalar, mundy::atomic_add
#include <mundy_math/ScalarBinaryOpTraits.hpp>  // for mundy::scalar_*_result_t
#include <mundy_math/Tolerance.hpp>             // for mundy::get_zero_tolerance, mundy::get_comparison_tolerance
#include <mundy_math/Vector.hpp>                // for mundy::Vector, mundy::dot, mundy::two_norm
#include <mundy_math/Vector3.hpp>               // for mundy::Vector3, mundy::cross
#include <mundy_math/cmath.hpp>                 // for mundy::passive_scalar_t

namespace mundy {

namespace {

using DD = DoubleDouble;

//! \name Group 0: compile-time checks
//@{

static_assert(is_passive_scalar_v<DD> && !is_autodiff_scalar_v<DD>);
static_assert(std::is_same_v<passive_scalar_t<DD>, DD>);
static_assert(std::is_same_v<passive_scalar_t<AutoDiffScalar<DD, 1>>, DD>);
static_assert(is_autodiff_scalar_v<AutoDiffScalar<DD, 1>> && !is_passive_scalar_v<AutoDiffScalar<DD, 1>>);
static_assert(ValidScalarType<DD>);
static_assert(std::is_trivially_copyable_v<DD>);
static_assert(std::is_same_v<scalar_product_result_t<DD, double>, DD>);
static_assert(std::is_same_v<scalar_sum_result_t<int, DD>, DD>);
static_assert(std::is_same_v<scalar_quotient_result_t<DD, DD>, DD>);
static_assert(std::is_same_v<NumTraits<DD>::Real, DD> && !NumTraits<DD>::IsInteger);

// Arithmetic is constexpr: 1/3 * 3 - 1 vanishes to double-double precision at compile time.
static_assert(abs((DD(1.0) / 3.0 * 3.0 - 1.0).hi()) < 1e-31);
static_assert(DD(1.0) + DD(0x1p-60) - 1.0 == DD(0x1p-60), "Cancellation is exact.");

// So is sqrt: sqrt(2)^2 - 2 vanishes to double-double precision at compile time.
static_assert(abs((sqrt(DD(2.0)) * sqrt(DD(2.0)) - 2.0).hi()) < 1e-30);

template <typename T>
concept HasAtomicAdd = requires(T* p, T v) { mundy::atomic_add(p, v); };
static_assert(HasAtomicAdd<DD>, "Atomics accept passive scalars.");
static_assert(!HasAtomicAdd<AutoDiffScalar<DD, 1>>, "Atomics still exclude autodiff scalars.");
//@}

//! \name Helpers
//@{

/// \brief Expect |actual - expected| <= rel_tol * |expected|, measured in double-double.
void expect_dd_near(const DD& actual, const DD& expected, double rel_tol, const char* what) {
  const double error = static_cast<double>(abs(actual - expected));
  EXPECT_LE(error, rel_tol * static_cast<double>(abs(expected)))
      << what << ": got " << actual.hi() << " + " << actual.lo() << ", expected " << expected.hi() << " + "
      << expected.lo();
}

// A correct double-double result is within a few units of 2^-104 (~5e-32) of the reference; allow 20.
constexpr double kTol = 1e-30;
//@}

TEST(DoubleDouble, ArithmeticMatchesReferences) {
  expect_dd_near(DD(1.0) / DD(3.0), {0x1.5555555555555p-2, 0x1.5555555555555p-56}, kTol, "1/3");
  expect_dd_near(DD(1.0) / 3.0, {0x1.5555555555555p-2, 0x1.5555555555555p-56}, kTol, "1/3, double divisor");
  expect_dd_near(sqrt(DD(2.0)), {0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54}, kTol, "sqrt(2)");
  expect_dd_near(cbrt(DD(2.0)), {0x1.428a2f98d728bp+0, -0x1.ddc22548ea41ep-56}, kTol, "cbrt(2)");
  expect_dd_near(sqrt(DD(2.0)) * sqrt(DD(2.0)), DD(2.0), kTol, "sqrt(2)^2");

  DD x = 1.0;
  x += 0.5;
  x *= 4.0;
  x -= 1.0;
  x /= 5.0;
  EXPECT_EQ(x, DD(1.0));
}

TEST(DoubleDouble, ConstructsIntegersExactly) {
  const DD big = std::int64_t{(1LL << 53) + 1};  // not representable in a double
  EXPECT_EQ(big.hi(), 0x1p53);
  EXPECT_EQ(big.lo(), 1.0);
  const DD max_u64 = std::uint64_t{0xFFFFFFFFFFFFFFFF};
  EXPECT_EQ(max_u64.hi(), 0x1p64);
  EXPECT_EQ(max_u64.lo(), -1.0);
  EXPECT_EQ(DD(-7), DD(-7.0));
}

TEST(DoubleDouble, ComparesLexicographically) {
  const DD one_plus = DD(1.0, 0x1p-80);
  EXPECT_LT(DD(1.0), one_plus);
  EXPECT_GT(one_plus, 1.0);
  EXPECT_LE(DD(1.0), 1);
  EXPECT_NE(one_plus, DD(1.0));
  EXPECT_EQ(min(one_plus, DD(1.0)), DD(1.0));
  EXPECT_EQ(max(one_plus, DD(1.0)), one_plus);
  EXPECT_EQ(static_cast<double>(one_plus), 1.0);
}

TEST(DoubleDouble, RoundsUsingTheLowPart) {
  EXPECT_EQ(floor(DD(1.0, -1e-17)), DD(0.0));  // just below 1
  EXPECT_EQ(ceil(DD(1.0, 1e-17)), DD(2.0));    // just above 1
  EXPECT_EQ(round(DD(2.5)), DD(3.0));          // halfway: away from zero
  EXPECT_EQ(round(DD(-2.5)), DD(-3.0));
  EXPECT_EQ(round(DD(2.5, -1e-17)), DD(2.0));  // just below halfway
  EXPECT_EQ(round(DD(-2.5, 1e-17)), DD(-2.0));
  EXPECT_EQ(copysign(DD(3.0), DD(-0.0)), DD(-3.0));
  EXPECT_EQ(abs(DD(-1.0, 0x1p-60)), DD(1.0, -0x1p-60));
}

TEST(DoubleDouble, ExponentialsAndLogarithmsMatchReferences) {
  expect_dd_near(exp(DD(1.0)), {0x1.5bf0a8b145769p+1, 0x1.4d57ee2b1013ap-53}, kTol, "exp(1)");
  expect_dd_near(exp(DD(-0.5)), {0x1.368b2fc6f960ap-1, -0x1.85314b9559e64p-61}, kTol, "exp(-0.5)");
  expect_dd_near(exp(DD(700.0)), {0x1.d945df4f8ec8ep+1009, 0x1.183392684a46ep+954}, kTol, "exp(700)");
  expect_dd_near(log(DD(2.0)), {0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56}, kTol, "log(2)");
  expect_dd_near(log(DD(10.0)), {0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53}, kTol, "log(10)");
  expect_dd_near(log(DD(0.75)), {-0x1.269621134db92p-2, -0x1.e0efadd9db02bp-56}, kTol, "log(0.75)");
  expect_dd_near(log(DD(1e-300)), {-0x1.5963447f87fb5p+9, -0x1.aa670d35324e6p-46}, kTol, "log(1e-300)");
  expect_dd_near(log10(DD(7.0)), {0x1.b0b0b0b78cc3fp-1, 0x1.4b692ff8a8060p-56}, kTol, "log10(7)");
  expect_dd_near(pow(DD(1.5), DD(-2.5)), {0x1.7398bf1d1ee70p-2, -0x1.c5417107a035cp-61}, kTol, "1.5^-2.5");
  EXPECT_EQ(pow(DD(-2.0), DD(3.0)), DD(-8.0));
  EXPECT_EQ(log(exp(DD(0.0))), DD(0.0));
}

TEST(DoubleDouble, TrigonometricFunctionsMatchReferences) {
  expect_dd_near(sin(DD(1.0)), {0x1.aed548f090ceep-1, 0x1.06374f484e288p-59}, kTol, "sin(1)");
  expect_dd_near(cos(DD(1.0)), {0x1.14a280fb5068cp-1, -0x1.b71edcc9344bcp-55}, kTol, "cos(1)");
  expect_dd_near(tan(DD(1.0)), {0x1.8eb245cbee3a6p+0, -0x1.1d4ce0afb373bp-54}, kTol, "tan(1)");
  expect_dd_near(sin(DD(1e5)), {0x1.24daa9c527e96p-5, 0x1.c767d8e3e1ca8p-60}, kTol, "sin(1e5)");
  expect_dd_near(cos(DD(-3.0)), {-0x1.fae04be85e5d2p-1, -0x1.83effc17efb54p-55}, kTol, "cos(-3)");
  expect_dd_near(atan(DD(0.5)), {0x1.dac670561bb4fp-2, 0x1.a2b7f222f65e2p-56}, kTol, "atan(0.5)");
  expect_dd_near(asin(DD(0.5)), {0x1.0c152382d7366p-1, -0x1.ee6913347c2a6p-55}, kTol, "asin(0.5) = pi/6");
  expect_dd_near(acos(DD(0.5)), {0x1.0c152382d7366p+0, -0x1.ee6913347c2a6p-54}, kTol, "acos(0.5) = pi/3");
  EXPECT_EQ(atan(DD(1.0)), DD(0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55)) << "atan(1) is pi/4 exactly";
  expect_dd_near(atan2(DD(1.0), DD(-1.0)), {0x1.2d97c7f3321d2p+1, 0x1.a79394c9e8a0ap-54}, kTol, "atan2(1, -1)");
  EXPECT_EQ(acos(DD(-1.0)), DD(0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53)) << "acos(-1) is pi exactly";
}

TEST(DoubleDouble, SpecialValuesFollowIeee) {
  EXPECT_TRUE(std::isnan(sqrt(DD(-1.0)).hi()));
  EXPECT_TRUE(std::isinf(log(DD(0.0)).hi()) && log(DD(0.0)) < 0.0);
  EXPECT_TRUE(std::isinf(exp(DD(1000.0)).hi()));
  EXPECT_EQ(exp(DD(-1000.0)), DD(0.0));
  EXPECT_TRUE(std::isinf((DD(1.0) / DD(0.0)).hi()));
  EXPECT_EQ(DD(1.0) / NumTraits<DD>::infinity(), DD(0.0));
  EXPECT_TRUE(std::isnan(asin(DD(2.0)).hi()));
  EXPECT_TRUE(std::signbit(sin(DD(-0.0)).hi()));
  EXPECT_TRUE(std::signbit(atan2(DD(-0.0), DD(1.0)).hi()));
  EXPECT_TRUE(std::isnan((NumTraits<DD>::quiet_NaN() + 1.0).hi()));
}

TEST(DoubleDouble, SqrtCoversTheWholeDoubleRange) {
  // Near the largest double, and subnormals: their roots are normal, but subnormals carry at most double precision.
  expect_dd_near(sqrt(DD(0x1.ffffffp+1023)), {0x1.ffffff7ffffffp+511, -0x1.0000005000002p+433}, kTol,
                 "sqrt(0x1.ffffffp+1023)");
  EXPECT_EQ(sqrt(DD(0x1p-1074)), DD(0x1p-537)) << "The smallest subnormal is a perfect square.";
  EXPECT_EQ(sqrt(DD(0x1p-1073)).hi(), 0x1.6a09e667f3bcdp-537) << "sqrt(2) * 2^-537, correctly rounded";

  // Special values: +-0 and +infinity are their own roots; negative numbers and NaN give NaN.
  EXPECT_TRUE(sqrt(DD(-0.0)) == 0.0 && std::signbit(sqrt(DD(-0.0)).hi()));
  EXPECT_EQ(sqrt(NumTraits<DD>::infinity()), NumTraits<DD>::infinity());
  EXPECT_TRUE(std::isnan(sqrt(-NumTraits<DD>::infinity()).hi()));
  EXPECT_TRUE(std::isnan(sqrt(NumTraits<DD>::quiet_NaN()).hi()));
}

TEST(DoubleDouble, SqrtIsTheSameAtCompileTimeAndRunTime) {
  // The compile-time root starts from the correctly rounded constexpr double root, the run-time root from the
  // hardware's; IEEE 754 makes the two equal, so the double-double roots match bit for bit. That holds wherever
  // double-double arithmetic is exact, |a| >= NumTraits<DD>::norm_min(): below it, x^2 has a subnormal low part.
  constexpr DD inputs[] = {DD(2.0), DD(1.0) / 3.0, DD(1e-280), DD(1e300), DD(0x1.ffffffp+1023)};
  constexpr DD roots[] = {sqrt(inputs[0]), sqrt(inputs[1]), sqrt(inputs[2]), sqrt(inputs[3]), sqrt(inputs[4])};
  for (int i = 0; i < 5; ++i) {
    const volatile double hi = inputs[i].hi();  // opaque to the optimizer, so the root is computed at run time
    const volatile double lo = inputs[i].lo();
    const DD root = sqrt(DD(hi, lo));
    EXPECT_EQ(root.hi(), roots[i].hi()) << "input " << i;
    EXPECT_EQ(root.lo(), roots[i].lo()) << "input " << i;
  }
}

TEST(DoubleDouble, TolerancesUseDoubleDoublePrecision) {
  EXPECT_EQ(get_zero_tolerance<DD>(), 10.0 * NumTraits<DD>::epsilon());
  EXPECT_EQ(get_relaxed_zero_tolerance<DD>(), NumTraits<DD>::dummy_precision());
  static_assert(std::is_same_v<decltype(get_comparison_tolerance<DD, double>()), double>,
                "Comparing with a double uses the coarser, double, tolerance.");
  static_assert(std::is_same_v<decltype(get_comparison_tolerance<DD, int>()), DD>);
  const float float_double_tolerance = get_comparison_tolerance<float, double>();
  EXPECT_EQ(float_double_tolerance, 1e-6f) << "Unchanged for primitives.";
}

TEST(DoubleDoubleInMundyMath, ScalarAndVector) {
  const Scalar<DD> s(DD(1.0) / 3.0);
  EXPECT_TRUE(is_close(s * 3.0, Scalar<DD>(1.0)));
  EXPECT_FALSE(is_close(s * 3.0, Scalar<DD>(DD(1.0, 1e-25))));

  const Vector3<DD> v(DD(3.0), DD(4.0), DD(12.0));
  EXPECT_EQ(two_norm(v), DD(13.0));
  const Vector3<DD> u(DD(1.0) / 3.0, sqrt(DD(2.0)), DD(0.5));
  EXPECT_LT(static_cast<double>(abs(dot(cross(u, v), u))), 1e-30) << "(u x v) . u vanishes";
  EXPECT_TRUE(is_close(u, u + Vector3<DD>(DD(1e-31), DD(0.0), DD(0.0))));
  EXPECT_FALSE(is_close(u, u + Vector3<DD>(DD(1e-25), DD(0.0), DD(0.0))));
}

TEST(DoubleDoubleInMundyMath, IllConditionedInverseKeepsItsDigits) {
  // cond(A) ~ 1e10: double keeps ~6 digits of the inverse, DoubleDouble ~22.
  const double delta = 1e-10;
  const Matrix<DD, 3, 3> a(1.0, 1.0, 1.0, 1.0, 1.0 + delta, 1.0, 1.0, 1.0, 1.0 + delta);
  const Matrix<double, 3, 3> a_double(1.0, 1.0, 1.0, 1.0, 1.0 + delta, 1.0, 1.0, 1.0, 1.0 + delta);
  const Matrix<DD, 3, 3> residual = a * inverse(a) - Matrix<DD, 3, 3>::identity();
  const Matrix<double, 3, 3> residual_double = a_double * inverse(a_double) - Matrix<double, 3, 3>::identity();
  double worst = 0.0;
  double worst_double = 0.0;
  for (size_t i = 0; i < 9; ++i) {
    worst = std::max(worst, std::abs(residual[i].hi()));
    worst_double = std::max(worst_double, std::abs(residual_double[i]));
  }
  EXPECT_LT(worst, 1e-20);
  EXPECT_GT(worst_double, 1e-12) << "The matrix must be ill-conditioned enough for double to lose digits.";
}

TEST(DoubleDoubleInMundyMath, QuaternionRotation) {
  // Rotating x by 0.7 about z gives (cos 0.7, sin 0.7, 0).
  const DD theta = 0.7;
  const Vector3<DD> axis(DD(0.0), DD(0.0), DD(1.0));
  const Quaternion<DD> q = axis_angle_to_quaternion(axis, theta);
  const Vector3<DD> r = q * Vector3<DD>(DD(1.0), DD(0.0), DD(0.0));
  expect_dd_near(r[0], {0x1.87996529f9d93p-1, -0x1.7234b60138711p-55}, kTol, "cos(0.7)");
  expect_dd_near(r[1], {0x1.49d6e694619b8p-1, 0x1.a822cbb5cf8f0p-59}, kTol, "sin(0.7)");
  EXPECT_LT(static_cast<double>(abs(r[2])), 1e-31);

  const Quaternion<DD> id = q * inverse(q);
  expect_dd_near(id.w(), DD(1.0), kTol, "q q^-1");
  EXPECT_LT(static_cast<double>(abs(id.z())), 1e-31);
}

TEST(DoubleDoubleInMundyMath, AutoDiffOverDoubleDouble) {
  // d/dx (x^2 + sqrt(x)) at x = 2 is 4 + sqrt(2)/4, to double-double precision.
  const AutoDiffScalar<DD, 1> x(DD(2.0), 0);
  const auto f = x * x + sqrt(x);
  expect_dd_near(f.derivatives()[0], {0x1.16a09e667f3bdp+2, -0x1.b7ba682764c8bp-53}, kTol, "f'(2)");
  EXPECT_EQ(impl::passive_value(f), f.value());
}

//! \name Device kernels, as free functions: CUDA forbids KOKKOS_LAMBDA in a test body (a private member function)
//@{

/// \brief sum_{i=1}^{n} 1/(i (i + 1)), reduced with Kokkos::Sum on the default execution space.
DD telescoping_sum_on_device(const int n) {
  DD sum = 0.0;
  Kokkos::parallel_reduce(
      "DoubleDouble::sum", Kokkos::RangePolicy<>(1, n + 1),
      KOKKOS_LAMBDA(const int i, DD& local) { local += DD(1.0) / (DD(i) * DD(i + 1)); }, sum);
  return sum;
}

/// \brief max_{i < n} (1 + 2^-80) i, reduced with Kokkos::Max on the default execution space.
DD largest_on_device(const int n) {
  DD largest = 0.0;
  Kokkos::parallel_reduce(
      "DoubleDouble::max", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i, DD& local) { local = max(local, DD(1.0, 0x1p-80) * DD(i)); },
      Kokkos::Max<DD>(largest));
  return largest;
}

/// \brief n atomic additions of 1 + 2^-80 to one value on the default execution space.
DD atomic_total_on_device(const int n) {
  Kokkos::View<DD> total("total");
  Kokkos::parallel_for(
      "DoubleDouble::atomic_add", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int) { mundy::atomic_add(&total(), DD(1.0, 0x1p-80)); });
  DD host_total;
  Kokkos::deep_copy(host_total, total);
  return host_total;
}
//@}

TEST(DoubleDoubleInKokkos, ReductionsAndAtomics) {
  // sum_{i=1}^{n} 1/(i (i + 1)) telescopes to n / (n + 1). Each addition rounds at 2^-104 relative, so the total
  // error stays below n * 2^-103; in double it would be about 1e-16.
  constexpr int n = 1000;
  expect_dd_near(telescoping_sum_on_device(n), DD(n) / DD(n + 1), 1e-27, "telescoping sum");
  EXPECT_EQ(largest_on_device(n), DD(1.0, 0x1p-80) * DD(n - 1));

  // Every thread adds 1 + 2^-80; the exact total n + n 2^-80 needs the low part.
  EXPECT_EQ(atomic_total_on_device(n), DD(static_cast<double>(n), n * 0x1p-80));
}

}  // namespace

}  // namespace mundy
