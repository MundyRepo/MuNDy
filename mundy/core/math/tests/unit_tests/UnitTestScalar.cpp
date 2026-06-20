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

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::is_arithmetic_v, std::common_type_t

// Mundy libs
#include <mundy_math/Matrix3.hpp>  // for mundy::Matrix3, mundy::AMatrix3
#include <mundy_math/Scalar.hpp>   // for mundy::AScalar, mundy::Scalar
#include <mundy_math/Tolerance.hpp>
#include <mundy_math/Vector3.hpp>  // for mundy::Vector3, mundy::AVector3

// Note: these tests are meant to look like real use cases for AScalar. As a result, we use implicit type conversions
// rather than being fully explicit about types. Compiling with -Wdouble-promotion or -Wconversion will produce
// warnings; we do not locally suppress them.

namespace mundy {

namespace {

//! \name Helper functions
//@{

template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_close_debug(const AScalar<U, Accessor1>& s1, const AScalar<T, Accessor2>& s2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(s1, s2)) {
    std::cout << "s1 = " << s1 << std::endl;
    std::cout << "s2 = " << s2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(s1, s2)) << message_if_fail;
}

template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_close_debug(const AVector3<U, Accessor1>& v1, const AVector3<T, Accessor2>& v2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(v1, v2)) << message_if_fail;
}

template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_close_debug(const AMatrix3<U, Accessor1>& m1, const AMatrix3<T, Accessor2>& m2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(m1, m2)) {
    std::cout << "m1 = " << m1 << std::endl;
    std::cout << "m2 = " << m2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(m1, m2)) << message_if_fail;
}
//@}

//! \name GTEST typed test fixtures
//@{

template <typename U>
class ScalarSingleTypeTest : public ::testing::Test {
  using T = U;
};
using SingleTypes = ::testing::Types<int, float, double>;
TYPED_TEST_SUITE(ScalarSingleTypeTest, SingleTypes);

template <typename U1, typename U2>
struct TypePair {
  using T1 = U1;
  using T2 = U2;
};

template <typename Pair>
class ScalarPairwiseTypeTest : public ::testing::Test {};
using PairwiseTypes = ::testing::Types<TypePair<int, float>, TypePair<int, double>, TypePair<float, double>,
                                       TypePair<int, int>, TypePair<float, float>, TypePair<double, double>>;
TYPED_TEST_SUITE(ScalarPairwiseTypeTest, PairwiseTypes);
//@}

//! \name Type trait checks
//@{

TEST(AScalarTest, TypeTraits) {
  static_assert(is_scalar_v<Scalar<int>>);
  static_assert(is_scalar_v<Scalar<float>>);
  static_assert(is_scalar_v<Scalar<double>>);
  static_assert(is_scalar_v<AScalar<double, Array<double, 1>>>);
  static_assert(!is_scalar_v<int>);
  static_assert(!is_scalar_v<double>);
  static_assert(!is_scalar_v<Vector3<double>>);
  static_assert(!is_scalar_v<Matrix3<double>>);
  // A 1-element AVector is NOT an AScalar — clean separation
  static_assert(!is_scalar_v<AVector<double, 1>>);
}
//@}

//! \name Construction and access
//@{

TYPED_TEST(ScalarSingleTypeTest, ConstructionAndAccess) {
  using T = TypeParam;

  Scalar<T> s{static_cast<T>(7)};
  EXPECT_EQ(s.value(), static_cast<T>(7));
  EXPECT_EQ(s[0], static_cast<T>(7));
  EXPECT_EQ(s(0), static_cast<T>(7));
  EXPECT_EQ(s(), static_cast<T>(7));
  EXPECT_EQ(static_cast<T>(s), static_cast<T>(7));  // implicit conversion
}

TYPED_TEST(ScalarSingleTypeTest, StaticFactories) {
  using T = TypeParam;
  EXPECT_EQ(static_cast<T>(Scalar<T>::zero()), static_cast<T>(0));
  EXPECT_EQ(static_cast<T>(Scalar<T>::one()), static_cast<T>(1));
}

TYPED_TEST(ScalarSingleTypeTest, UnaryOperators) {
  using T = TypeParam;
  Scalar<T> s{static_cast<T>(5)};
  is_close_debug(+s, Scalar<T>{static_cast<T>(5)}, "Unary + failed.");
  is_close_debug(-s, Scalar<T>{static_cast<T>(-5)}, "Unary - failed.");
}

TYPED_TEST(ScalarSingleTypeTest, CopyAndCast) {
  using T = TypeParam;
  Scalar<T> s{static_cast<T>(3)};
  auto c = s.copy();
  is_close_debug(c, s, "copy() failed.");
  auto d = cast<double>(s);
  EXPECT_DOUBLE_EQ(static_cast<double>(d), static_cast<double>(static_cast<T>(3)));
}
//@}

//! \name Scalar * Scalar and Scalar / Scalar
//@{

TYPED_TEST(ScalarPairwiseTypeTest, ScalarScalarMultiplication) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using Common = std::common_type_t<T1, T2>;

  Scalar<T1> a{static_cast<T1>(3)};
  Scalar<T2> b{static_cast<T2>(4)};
  is_close_debug(a * b, Scalar<Common>{static_cast<Common>(12)}, "Scalar * Scalar failed.");
}

TYPED_TEST(ScalarPairwiseTypeTest, ScalarScalarDivision) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Use values that divide evenly so integer-type tests are valid.
  Scalar<T1> a{static_cast<T1>(12)};
  Scalar<T2> b{static_cast<T2>(4)};
  auto result = a / b;

  // AScalar / AScalar promotes integral/integral to double, matching AVector semantics.
  using Common = std::common_type_t<T1, T2>;
  using Promoted = std::conditional_t<std::is_integral_v<T1> && std::is_integral_v<T2>, double, Common>;
  is_close_debug(result, Scalar<Promoted>{static_cast<Promoted>(3)}, "Scalar / Scalar failed.");
}

TYPED_TEST(ScalarSingleTypeTest, ScalarScalarAddSub) {
  using T = TypeParam;
  Scalar<T> a{static_cast<T>(10)};
  Scalar<T> b{static_cast<T>(4)};
  is_close_debug(a + b, Scalar<T>{static_cast<T>(14)}, "Scalar + Scalar failed.");
  is_close_debug(a - b, Scalar<T>{static_cast<T>(6)}, "Scalar - Scalar failed.");
}

TYPED_TEST(ScalarSingleTypeTest, ScalarScalarRoundTrip) {
  using T = TypeParam;
  Scalar<T> a{static_cast<T>(6)};
  Scalar<T> b{static_cast<T>(2)};
  // (a * b) / b == a
  auto product = a * b;
  auto quotient = product / b;
  is_close_debug(quotient, cast<typename decltype(quotient)::value_type>(a), "Round-trip (a*b)/b != a.");
}
//@}

//! \name Scalar op arithmetic and arithmetic op Scalar
//@{

TYPED_TEST(ScalarSingleTypeTest, ScalarArithmeticOps) {
  using T = TypeParam;
  Scalar<T> s{static_cast<T>(4)};

  is_close_debug(s + static_cast<T>(2), Scalar<T>{static_cast<T>(6)}, "Scalar + T failed.");
  is_close_debug(s - static_cast<T>(2), Scalar<T>{static_cast<T>(2)}, "Scalar - T failed.");
  is_close_debug(s * static_cast<T>(3), Scalar<T>{static_cast<T>(12)}, "Scalar * T failed.");

  is_close_debug(static_cast<T>(2) + s, Scalar<T>{static_cast<T>(6)}, "T + Scalar failed.");
  is_close_debug(static_cast<T>(10) - s, Scalar<T>{static_cast<T>(6)}, "T - Scalar failed.");
  is_close_debug(static_cast<T>(3) * s, Scalar<T>{static_cast<T>(12)}, "T * Scalar failed.");
}
//@}

//! \name Self-modification operators
//@{

TYPED_TEST(ScalarSingleTypeTest, SelfModification) {
  using T = TypeParam;

  Scalar<T> s{static_cast<T>(10)};
  s += Scalar<T>{static_cast<T>(5)};
  EXPECT_EQ(static_cast<T>(s), static_cast<T>(15));

  s -= Scalar<T>{static_cast<T>(3)};
  EXPECT_EQ(static_cast<T>(s), static_cast<T>(12));

  s *= Scalar<T>{static_cast<T>(2)};
  EXPECT_EQ(static_cast<T>(s), static_cast<T>(24));

  s /= Scalar<T>{static_cast<T>(4)};
  EXPECT_EQ(static_cast<T>(s), static_cast<T>(6));
}
//@}

//! \name Vector3 * Scalar and Scalar * Vector3
//@{

TYPED_TEST(ScalarPairwiseTypeTest, VectorScalarMultiplication) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using Common = std::common_type_t<T1, T2>;

  Vector3<T1> v{static_cast<T1>(1), static_cast<T1>(2), static_cast<T1>(3)};
  Scalar<T2> s{static_cast<T2>(2)};
  Vector3<Common> expected{static_cast<Common>(2), static_cast<Common>(4), static_cast<Common>(6)};

  is_close_debug(v * s, expected, "Vector3 * Scalar failed.");
  is_close_debug(s * v, expected, "Scalar * Vector3 failed.");
  is_close_debug(v * s, s * v, "Vector3 * Scalar != Scalar * Vector3.");
}
//@}

//! \name Vector3 / Scalar
//@{

TYPED_TEST(ScalarPairwiseTypeTest, VectorScalarDivision) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Vector3<T1> v{static_cast<T1>(2), static_cast<T1>(4), static_cast<T1>(6)};
  Scalar<T2> s{static_cast<T2>(2)};
  auto result = v / s;

  // AVector::operator/ promotes integral/integral to double.
  using Common = std::common_type_t<T1, T2>;
  using Promoted = std::conditional_t<std::is_integral_v<T1> && std::is_integral_v<T2>, double, Common>;
  Vector3<Promoted> expected{static_cast<Promoted>(1), static_cast<Promoted>(2), static_cast<Promoted>(3)};
  is_close_debug(result, expected, "Vector3 / Scalar failed.");
}
//@}

//! \name Matrix3 * Scalar and Scalar * Matrix3
//@{

TYPED_TEST(ScalarPairwiseTypeTest, MatrixScalarMultiplication) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using Common = std::common_type_t<T1, T2>;

  // clang-format off
  Matrix3<T1> mat{static_cast<T1>(1), static_cast<T1>(2), static_cast<T1>(3),
                  static_cast<T1>(4), static_cast<T1>(5), static_cast<T1>(6),
                  static_cast<T1>(7), static_cast<T1>(8), static_cast<T1>(9)};
  // clang-format on
  Scalar<T2> s{static_cast<T2>(2)};

  // clang-format off
  Matrix3<Common> expected{static_cast<Common>(2),  static_cast<Common>(4),  static_cast<Common>(6),
                           static_cast<Common>(8),  static_cast<Common>(10), static_cast<Common>(12),
                           static_cast<Common>(14), static_cast<Common>(16), static_cast<Common>(18)};
  // clang-format on

  is_close_debug(mat * s, expected, "Matrix3 * Scalar failed.");
  is_close_debug(s * mat, expected, "Scalar * Matrix3 failed.");
  is_close_debug(mat * s, s * mat, "Matrix3 * Scalar != Scalar * Matrix3.");
}
//@}

//! \name Matrix3 / Scalar
//@{

TYPED_TEST(ScalarPairwiseTypeTest, MatrixScalarDivision) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // clang-format off
  Matrix3<T1> mat{static_cast<T1>(2),  static_cast<T1>(4),  static_cast<T1>(6),
                  static_cast<T1>(8),  static_cast<T1>(10), static_cast<T1>(12),
                  static_cast<T1>(14), static_cast<T1>(16), static_cast<T1>(18)};
  // clang-format on
  Scalar<T2> s{static_cast<T2>(2)};
  auto result = mat / s;

  using Common = std::common_type_t<T1, T2>;
  using Promoted = std::conditional_t<std::is_integral_v<T1> && std::is_integral_v<T2>, double, Common>;
  // clang-format off
  Matrix3<Promoted> expected{static_cast<Promoted>(1), static_cast<Promoted>(2), static_cast<Promoted>(3),
                             static_cast<Promoted>(4), static_cast<Promoted>(5), static_cast<Promoted>(6),
                             static_cast<Promoted>(7), static_cast<Promoted>(8), static_cast<Promoted>(9)};
  // clang-format on
  is_close_debug(result, expected, "Matrix3 / Scalar failed.");
}
//@}

//! \name View semantics
//@{

TEST(AScalarTest, ViewSemantics) {
  double raw = 42.0;
  auto view = get_scalar_view<double>(&raw);

  EXPECT_DOUBLE_EQ(view.value(), 42.0);

  // Writing through the view modifies the original
  view.value() = 99.0;
  EXPECT_DOUBLE_EQ(raw, 99.0);
}
//@}

}  // namespace

}  // namespace mundy
