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
#include <algorithm>  // for std::max
#include <atomic>
#include <barrier>
#include <future>
#include <map>        // for std::map
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <thread>
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Mundy libs
#include <mundy_math/Array.hpp>      // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_relaxed_tolerance
#include <mundy_math/Vector.hpp>     // for mundy::math::Vector

// Note, these tests are meant to look like real use cases for the Vector class. As a result, we use implicit type
// conversions rather than being explicit about types. This is to ensure that the Vector class can be used in a
// natural way. This choice means that compiling this test with -Wdouble-promotion or -Wconversion will result in many
// warnings. We will not however, locally disable these warnings.

namespace mundy {

namespace math {

namespace {

//! \name Helper functions
//@{

/// \brief Test that two algebraic types are close
/// \param[in] a The first algebraic type
/// \param[in] b The second algebraic type
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T>
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
void is_close_debug(const U& a, const T& b, const std::string& message_if_fail = "") {
  if (!is_approx_close(a, b)) {
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    using CommonType = std::common_type_t<U, T>;
    std::cout << "diff = " << static_cast<CommonType>(a) - static_cast<CommonType>(b) << std::endl;
  }

  EXPECT_TRUE(is_approx_close(a, b)) << message_if_fail;
}

/// \brief Test that two Vectors are close
/// \param[in] v1 The first Vector
/// \param[in] v2 The second Vector
/// \param[in] message_if_fail The message to print if the test fails
template <size_t N, typename U, typename OtherAccessor, typename OtherOwnershipType, typename T, typename Accessor,
          typename OwnershipType>
void is_close_debug(const AVector<U, N, OtherAccessor, OtherOwnershipType>& v1,
                    const AVector<T, N, Accessor, OwnershipType>& v2, const std::string& message_if_fail = "") {
  if (!is_approx_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(v1, v2)) << message_if_fail;
}

/// \brief Test that two Vectors are different
/// \param[in] v1 The first Vector
/// \param[in] v2 The second Vector
/// \param[in] message_if_fail The message to print if the test fails
template <size_t N, typename U, typename OtherAccessor, typename OtherOwnershipType, typename T, typename Accessor,
          typename OwnershipType>
void is_different_debug(const AVector<U, N, OtherAccessor, OtherOwnershipType>& v1,
                        const AVector<T, N, Accessor, OwnershipType>& v2, const std::string& message_if_fail = "") {
  if (is_approx_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(!is_approx_close(v1, v2)) << message_if_fail;
}
//@}

//! \name GTEST typed test fixtures
//@{

/// \brief GTEST typed test fixture so we can run tests on multiple types
/// \tparam U The type to run the tests on
template <typename U>
class VectorSingleTypeTest : public ::testing::Test {
  using T = U;
};  // VectorSingleTypeTest

/// \brief List of types to run the tests on
using MyTypes = ::testing::Types<int, float, double>;

/// \brief Tell GTEST to run the tests on the types in MyTypes
TYPED_TEST_SUITE(VectorSingleTypeTest, MyTypes);

/// \brief A helper class for a pair of types
/// \tparam U1 The first type
/// \tparam U2 The second type
template <typename U1, typename U2>
struct TypePair {
  using T1 = U1;
  using T2 = U2;
};

/// \brief GTETS typed test fixture so we can run tests on multiple pairs of types
/// \tparam Pair The pair of types to run the tests on
template <typename Pair>
class VectorPairwiseTypeTest : public ::testing::Test {};  // VectorPairwiseTypeTest

/// \brief List of pairs of types to run the tests on
using MyTypePairs = ::testing::Types<TypePair<int, float>, TypePair<int, double>, TypePair<float, double>,
                                     TypePair<int, int>, TypePair<float, float>, TypePair<double, double>>;

/// \brief Tell GTEST to run the tests on the types in MyTypePairs
TYPED_TEST_SUITE(VectorPairwiseTypeTest, MyTypePairs);
//@}

//! \name Vector Constructors and Destructor (in dims 1, 2, and 3)
//@{

TYPED_TEST(VectorSingleTypeTest, DefaultConstructor) {
  ASSERT_NO_THROW(Vector1<TypeParam>());
  ASSERT_NO_THROW(Vector2<TypeParam>());
  ASSERT_NO_THROW(Vector3<TypeParam>());
}

TYPED_TEST(VectorSingleTypeTest, LiteralConstruction) {
  Vector1<TypeParam>{1};
  Vector2<TypeParam>{1, 2};
  Vector3<TypeParam>{1, 2, 3};
  constexpr Vector1<TypeParam> v1{1};
  constexpr Vector2<TypeParam> v2{1, 2};
  constexpr Vector3<TypeParam> v3{1, 2, 3};
}

TYPED_TEST(VectorSingleTypeTest, ConstructorFromNScalars) {
  ASSERT_NO_THROW(Vector1<TypeParam>(1));
  ASSERT_NO_THROW(Vector2<TypeParam>(1, 2));
  ASSERT_NO_THROW(Vector3<TypeParam>(1, 2, 3));
  Vector1<TypeParam> v1(1);
  Vector2<TypeParam> v2(1, 2);
  Vector3<TypeParam> v3(1, 2, 3);
  is_close_debug(v1[0], 1);
  is_close_debug(v2[0], 1);
  is_close_debug(v2[1], 2);
  is_close_debug(v3[0], 1);
  is_close_debug(v3[1], 2);
  is_close_debug(v3[2], 3);
}

TYPED_TEST(VectorSingleTypeTest, Comparison) {
  Vector1<TypeParam> v1{1};
  Vector1<TypeParam> v2{2};
  EXPECT_TRUE(is_close(v1, v1));
  EXPECT_FALSE(is_close(v1, v2));
  is_close_debug(v1, v1);
  is_close_debug(v1, Vector1<TypeParam>{1});

  Vector2<TypeParam> v3{1, 2};
  Vector2<TypeParam> v4{2, 3};
  EXPECT_TRUE(is_close(v3, v3));
  EXPECT_FALSE(is_close(v3, v4));
  is_close_debug(v3, v3);
  is_close_debug(v3, Vector2<TypeParam>{1, 2});

  Vector3<TypeParam> v5{1, 2, 3};
  Vector3<TypeParam> v6{2, 3, 4};
  EXPECT_TRUE(is_close(v5, v5));
  EXPECT_FALSE(is_close(v5, v6));
  is_close_debug(v5, v5);
  is_close_debug(v5, Vector3<TypeParam>{1, 2, 3});
}

TYPED_TEST(VectorSingleTypeTest, CopyConstructor) {
  Vector1<TypeParam> v1{1};
  Vector1<TypeParam> v2(v1);
  is_close_debug(v1, v2, "Copy constructor failed.");
  v1.set(2);
  is_different_debug(v1, v2, "Copy constructor failed, somehow the data is shared.");

  Vector2<TypeParam> v3{1, 2};
  Vector2<TypeParam> v4(v3);
  is_close_debug(v3, v4, "Copy constructor failed.");
  v3.set(3, 4);
  is_different_debug(v3, v4, "Copy constructor failed, somehow the data is shared.");

  Vector3<TypeParam> v5{1, 2, 3};
  Vector3<TypeParam> v6(v5);
  is_close_debug(v5, v6, "Copy constructor failed.");
  v5.set(4, 5, 6);
  is_different_debug(v5, v6, "Copy constructor failed, somehow the data is shared.");
}

TYPED_TEST(VectorSingleTypeTest, MoveConstructor) {
  Vector1<TypeParam> v1{1};
  Vector1<TypeParam> v2(std::move(v1));
  is_close_debug(v2, Vector1<TypeParam>{1}, "Move constructor failed.");

  Vector2<TypeParam> v3{1, 2};
  Vector2<TypeParam> v4(std::move(v3));
  is_close_debug(v4, Vector2<TypeParam>{1, 2}, "Move constructor failed.");

  Vector3<TypeParam> v5{1, 2, 3};
  Vector3<TypeParam> v6(std::move(v5));
  is_close_debug(v6, Vector3<TypeParam>{1, 2, 3}, "Move constructor failed.");
}

TYPED_TEST(VectorSingleTypeTest, CopyAssignment) {
  Vector1<TypeParam> v1{1};
  Vector1<TypeParam> v2{2};
  ASSERT_NO_THROW(v2 = v1);
  is_close_debug(v1, v2, "Copy assignment failed.");

  Vector2<TypeParam> v3{1, 2};
  Vector2<TypeParam> v4{3, 4};
  ASSERT_NO_THROW(v4 = v3);
  is_close_debug(v3, v4, "Copy assignment failed.");

  Vector3<TypeParam> v5{1, 2, 3};
  Vector3<TypeParam> v6{4, 5, 6};
  ASSERT_NO_THROW(v6 = v5);
  is_close_debug(v5, v6, "Copy assignment failed.");
}

TYPED_TEST(VectorSingleTypeTest, MoveAssignment) {
  Vector1<TypeParam> v1{1};
  Vector1<TypeParam> v2{2};
  ASSERT_NO_THROW(v2 = std::move(v1));
  is_close_debug(v2, Vector1<TypeParam>{1}, "Move assignment failed.");

  Vector2<TypeParam> v3{1, 2};
  Vector2<TypeParam> v4{3, 4};
  ASSERT_NO_THROW(v4 = std::move(v3));
  is_close_debug(v4, Vector2<TypeParam>{1, 2}, "Move assignment failed.");

  Vector3<TypeParam> v5{1, 2, 3};
  Vector3<TypeParam> v6{4, 5, 6};
  ASSERT_NO_THROW(v6 = std::move(v5));
  is_close_debug(v6, Vector3<TypeParam>{1, 2, 3}, "Move assignment failed.");
}

TYPED_TEST(VectorSingleTypeTest, Destructor) {
  ASSERT_NO_THROW(Vector1<TypeParam>());
  ASSERT_NO_THROW(Vector2<TypeParam>());
  ASSERT_NO_THROW(Vector3<TypeParam>());
}
//@}

//! \name Vector Accessors (in dims 1, 2, and 3)
//@{

TYPED_TEST(VectorSingleTypeTest, Accessors) {
  Vector1<TypeParam> v1(1);
  is_close_debug(v1[0], 1);
  v1[0] = 2;
  is_close_debug(v1[0], 2);

  Vector2<TypeParam> v2(1, 2);
  is_close_debug(v2[0], 1);
  is_close_debug(v2[1], 2);
  v2[0] = 3;
  v2[1] = 4;
  is_close_debug(v2[0], 3);
  is_close_debug(v2[1], 4);

  Vector3<TypeParam> v3(1, 2, 3);
  is_close_debug(v3[0], 1);
  is_close_debug(v3[1], 2);
  is_close_debug(v3[2], 3);
  v3[0] = 4;
  v3[1] = 5;
  v3[2] = 6;
  is_close_debug(v3[0], 4);
  is_close_debug(v3[1], 5);
  is_close_debug(v3[2], 6);
}
//@}

//! \name Vector Setters
//@{

TYPED_TEST(VectorSingleTypeTest, Setters) {
  // Dim 1
  Vector1<TypeParam> v1;
  v1.set(1);
  is_close_debug(v1, Vector1<TypeParam>{1}, "Set by scalar failed.");

  v1.set(Vector1<TypeParam>{2});
  is_close_debug(v1, Vector1<TypeParam>{2}, "Set by vector failed.");

  v1.fill(3);
  is_close_debug(v1, Vector1<TypeParam>{3}, "Fill failed.");

  // Dim 2
  Vector2<TypeParam> v2;
  v2.set(1, 2);
  is_close_debug(v2, Vector2<TypeParam>{1, 2}, "Set by two scalars failed.");

  v2.set(Vector2<TypeParam>{3, 4});
  is_close_debug(v2, Vector2<TypeParam>{3, 4}, "Set by vector failed.");

  v2.fill(5);
  is_close_debug(v2, Vector2<TypeParam>{5, 5}, "Fill failed.");

  // Dim 3
  Vector3<TypeParam> v3;
  v3.set(1, 2, 3);
  is_close_debug(v3, Vector3<TypeParam>{1, 2, 3}, "Set by three scalars failed.");

  v3.set(Vector3<TypeParam>{4, 5, 6});
  is_close_debug(v3, Vector3<TypeParam>{4, 5, 6}, "Set by vector failed.");

  v3.fill(7);
  is_close_debug(v3, Vector3<TypeParam>{7, 7, 7}, "Fill failed.");
}
//@}

//! \name Vector Special vectors
//@{

TYPED_TEST(VectorSingleTypeTest, SpecialVectors) {
  ASSERT_NO_THROW(Vector1<TypeParam>::zeros());
  ASSERT_NO_THROW(Vector1<TypeParam>::ones());
  auto ones1 = Vector1<TypeParam>::ones();
  auto zeros1 = Vector1<TypeParam>::zeros();

  is_close_debug(ones1, Vector1<TypeParam>{1}, "Ones failed.");
  is_close_debug(zeros1, Vector1<TypeParam>{0}, "Zeros failed.");

  ASSERT_NO_THROW(Vector2<TypeParam>::zeros());
  ASSERT_NO_THROW(Vector2<TypeParam>::ones());
  auto ones2 = Vector2<TypeParam>::ones();
  auto zeros2 = Vector2<TypeParam>::zeros();

  is_close_debug(ones2, Vector2<TypeParam>{1, 1}, "Ones failed.");
  is_close_debug(zeros2, Vector2<TypeParam>{0, 0}, "Zeros failed.");

  ASSERT_NO_THROW(Vector3<TypeParam>::zeros());
  ASSERT_NO_THROW(Vector3<TypeParam>::ones());
  auto ones3 = Vector3<TypeParam>::ones();
  auto zeros3 = Vector3<TypeParam>::zeros();

  is_close_debug(ones3, Vector3<TypeParam>{1, 1, 1}, "Ones failed.");
  is_close_debug(zeros3, Vector3<TypeParam>{0, 0, 0}, "Zeros failed.");
}
//@}

//! \name Vector Addition and subtraction
//@{

TYPED_TEST(VectorPairwiseTypeTest, AdditionAndSubtractionWithVector) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  Vector1<T1> v1(1);
  Vector1<T2> v2(2);
  auto v3 = v1 + v2;
  using T3 = decltype(v3)::scalar_t;
  is_close_debug(v3, Vector1<T3>{3}, "Vector-vector addition failed.");

  v1 += v2;
  is_close_debug(v1, Vector1<T1>{3}, "Vector-vector addition assignment failed.");

  v3 = v1 - v2;
  is_close_debug(v3, Vector1<T3>{1}, "Vector-vector subtraction failed.");

  v1 -= v2;
  is_close_debug(v1, Vector1<T1>{1}, "Vector-vector subtraction assignment failed.");

  // Dim 2
  Vector2<T1> v4(1, 2);
  Vector2<T2> v5(3, 4);
  auto v6 = v4 + v5;
  using T4 = decltype(v6)::scalar_t;
  is_close_debug(v6, Vector2<T4>{4, 6}, "Vector-vector addition failed.");

  v4 += v5;
  is_close_debug(v4, Vector2<T1>{4, 6}, "Vector-vector addition assignment failed.");

  v6 = v4 - v5;
  is_close_debug(v6, Vector2<T4>{1, 2}, "Vector-vector subtraction failed.");

  v4 -= v5;
  is_close_debug(v4, Vector2<T1>{1, 2}, "Vector-vector subtraction assignment failed.");

  // Dim 3
  Vector3<T1> v7(1, 2, 3);
  Vector3<T2> v8(4, 5, 6);
  auto v9 = v7 + v8;
  using T5 = decltype(v9)::scalar_t;
  is_close_debug(v9, Vector3<T5>{5, 7, 9}, "Vector-vector addition failed.");

  v7 += v8;
  is_close_debug(v7, Vector3<T1>{5, 7, 9}, "Vector-vector addition assignment failed.");

  v9 = v7 - v8;
  is_close_debug(v9, Vector3<T5>{1, 2, 3}, "Vector-vector subtraction failed.");

  v7 -= v8;
  is_close_debug(v7, Vector3<T1>{1, 2, 3}, "Vector-vector subtraction assignment failed.");
}
TYPED_TEST(VectorPairwiseTypeTest, AdditionAndSubtractionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  Vector1<T1> v1(1);
  auto v2 = v1 + T2(1);
  using T3 = decltype(v2)::scalar_t;
  is_close_debug(v1 + T2(1), Vector1<T3>{2}, "Vector-scalar addition failed.");
  is_close_debug(T2(1) + v1, Vector1<T3>{2}, "Scalar-vector addition failed.");
  is_close_debug(v1 - T2(1), Vector1<T3>{0}, "Vector-scalar subtraction failed.");
  is_close_debug(T2(1) - v1, Vector1<T3>{0}, "Scalar-vector subtraction failed.");

  // Dim 2
  Vector2<T1> v3(1, 2);
  auto v4 = v3 + T2(1);
  using T4 = decltype(v4)::scalar_t;
  is_close_debug(v4, Vector2<T4>{2, 3}, "Vector-scalar addition failed.");
  is_close_debug(T2(1) + v3, Vector2<T4>{2, 3}, "Scalar-vector addition failed.");
  is_close_debug(v3 - T2(1), Vector2<T4>{0, 1}, "Vector-scalar subtraction failed.");
  is_close_debug(T2(1) - v3, Vector2<T4>{0, -1}, "Scalar-vector subtraction failed.");

  // Dim 3
  Vector3<T1> v5(1, 2, 3);
  auto v6 = v5 + T2(1);
  using T5 = decltype(v6)::scalar_t;
  is_close_debug(v6, Vector3<T5>{2, 3, 4}, "Vector-scalar addition failed.");
  is_close_debug(T2(1) + v5, Vector3<T5>{2, 3, 4}, "Scalar-vector addition failed.");
  is_close_debug(v5 - T2(1), Vector3<T5>{0, 1, 2}, "Vector-scalar subtraction failed.");
  is_close_debug(T2(1) - v5, Vector3<T5>{0, -1, -2}, "Scalar-vector subtraction failed.");
}

TYPED_TEST(VectorPairwiseTypeTest, AdditionAndSubtractionRValues) {
  // Test that the addition and subtraction operators work with rvalues
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  Vector1<T1> v1(1);
  is_close_debug(v1 + Vector1<T2>{2}, Vector1<T1>{3}, "Vector-vector addition failed.");
  is_close_debug(Vector1<T2>{2} + v1, Vector1<T1>{3}, "Vector-vector addition failed.");
  is_close_debug(v1 - Vector1<T2>{2}, Vector1<T1>{-1}, "Vector-vector subtraction failed.");
  is_close_debug(Vector1<T2>{2} - v1, Vector1<T1>{1}, "Vector-vector subtraction failed.");

  // Dim 2
  Vector2<T1> v2(1, 2);
  is_close_debug(v2 + Vector2<T2>{3, 4}, Vector2<T1>{4, 6}, "Vector-vector addition failed.");
  is_close_debug(Vector2<T2>{3, 4} + v2, Vector2<T1>{4, 6}, "Vector-vector addition failed.");
  is_close_debug(v2 - Vector2<T2>{3, 4}, Vector2<T1>{-2, -2}, "Vector-vector subtraction failed.");
  is_close_debug(Vector2<T2>{3, 4} - v2, Vector2<T1>{2, 2}, "Vector-vector subtraction failed.");

  // Dim 3
  Vector3<T1> v3(1, 2, 3);
  is_close_debug(v3 + Vector3<T2>{4, 5, 6}, Vector3<T1>{5, 7, 9}, "Vector-vector addition failed.");
  is_close_debug(Vector3<T2>{4, 5, 6} + v3, Vector3<T1>{5, 7, 9}, "Vector-vector addition failed.");
  is_close_debug(v3 - Vector3<T2>{4, 5, 6}, Vector3<T1>{-3, -3, -3}, "Vector-vector subtraction failed.");
  is_close_debug(Vector3<T2>{4, 5, 6} - v3, Vector3<T1>{3, 3, 3}, "Vector-vector subtraction failed.");
}
//@}

//! \name Vector Multiplication and division (in dims 1, 2, and 3)
//@{

TYPED_TEST(VectorPairwiseTypeTest, MultiplicationAndDivisionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  Vector1<T1> v1(1);
  auto v2 = v1 * T2(2);
  using T3 = decltype(v2)::scalar_t;
  is_close_debug(v2, Vector1<T3>{2}, "Vector-scalar multiplication failed.");
  is_close_debug(T2(2) * v1, Vector1<T3>{2}, "Scalar-vector multiplication failed.");
  is_close_debug(v2 / T2(2), Vector1<T3>{1}, "Vector-scalar division failed.");

  // Dim 2
  Vector2<T1> v3(1, 2);
  auto v4 = v3 * T2(2);
  using T4 = decltype(v4)::scalar_t;
  is_close_debug(v4, Vector2<T4>{2, 4}, "Vector-scalar multiplication failed.");
  is_close_debug(T2(2) * v3, Vector2<T4>{2, 4}, "Scalar-vector multiplication failed.");
  is_close_debug(v4 / T2(2), Vector2<T4>{1, 2}, "Vector-scalar division failed.");

  // Dim 3
  Vector3<T1> v5(1, 2, 3);
  auto v6 = v5 * T2(2);
  using T5 = decltype(v6)::scalar_t;
  is_close_debug(v6, Vector3<T5>{2, 4, 6}, "Vector-scalar multiplication failed.");
  is_close_debug(T2(2) * v5, Vector3<T5>{2, 4, 6}, "Scalar-vector multiplication failed.");
  is_close_debug(v6 / T2(2), Vector3<T5>{1, 2, 3}, "Vector-scalar division failed.");
}
//@}

//! \name Vector Basic arithmetic reduction operations
//@{

TYPED_TEST(VectorSingleTypeTest, BasicArithmeticReductionOperations) {
  // Dim 1
  Vector1<TypeParam> v1(1);
  is_close_debug(sum(v1), 1, "Sum failed.");
  is_close_debug(product(v1), 1, "Product failed.");
  is_close_debug(min(v1), 1, "Min failed.");
  is_close_debug(max(v1), 1, "Max failed.");
  is_close_debug(mean(v1), 1, "Mean failed.");
  is_close_debug(variance(v1), 0, "Variance failed.");
  is_close_debug(stddev(v1), 0, "Stddev failed.");

  // Dim 2
  Vector2<TypeParam> v2(1, 2);
  is_close_debug(sum(v2), 3, "Sum failed.");
  is_close_debug(product(v2), 2, "Product failed.");
  is_close_debug(min(v2), 1, "Min failed.");
  is_close_debug(max(v2), 2, "Max failed.");
  is_close_debug(mean(v2), 1.5, "Mean failed.");
  is_close_debug(variance(v2), 0.25, "Variance failed.");
  is_close_debug(stddev(v2), 0.5, "Stddev failed.");

  // Dim 3
  Vector3<TypeParam> v3(1, 2, 3);
  is_close_debug(sum(v3), 6, "Sum failed.");
  is_close_debug(product(v3), 6, "Product failed.");
  is_close_debug(min(v3), 1, "Min failed.");
  is_close_debug(max(v3), 3, "Max failed.");
  is_close_debug(mean(v3), 2, "Mean failed.");
  is_close_debug(variance(v3), 2.0 / 3.0, "Variance failed.");
  is_close_debug(stddev(v3), std::sqrt(2.0 / 3.0), "Stddev failed.");
}
//@}

//! \name Vector Special vector operations (in dims 1, 2, and 3)
//@{

TYPED_TEST(VectorPairwiseTypeTest, SpecialOperations) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // Dim 1
  Vector1<T1> v1(1);
  Vector1<T2> v2(2);
  is_close_debug(dot(v1, v2), static_cast<C>(2.0), "Dot product failed.");
  is_close_debug(norm(v1), static_cast<C>(1.0), "Norm failed.");
  is_close_debug(norm_squared(v1), static_cast<C>(1.0), "Norm squared failed.");
  is_close_debug(inf_norm(v1), static_cast<C>(1.0), "Infinity norm failed.");
  is_close_debug(one_norm(v1), static_cast<C>(1.0), "One norm failed.");
  is_close_debug(two_norm(v1), static_cast<C>(1.0), "Two norm failed.");
  is_close_debug(two_norm_squared(v1), static_cast<C>(1.0), "Two norm squared failed.");
  is_close_debug(minor_angle(v1, v2), static_cast<C>(0.0), "Minor angle failed.");
  is_close_debug(major_angle(v1, v2), static_cast<C>(M_PI), "Major angle failed.");

  // Dim 2
  Vector2<T1> v3(1, 2);
  Vector2<T2> v4(3, 4);
  is_close_debug(dot(v3, v4), static_cast<C>(11.0), "Dot product failed.");
  is_close_debug(norm(v3), static_cast<C>(std::sqrt(5.0)), "Norm failed.");
  is_close_debug(norm_squared(v3), static_cast<C>(5.0), "Norm squared failed.");
  is_close_debug(inf_norm(v3), static_cast<C>(2.0), "Infinity norm failed.");
  is_close_debug(one_norm(v3), static_cast<C>(3.0), "One norm failed.");
  is_close_debug(two_norm(v3), static_cast<C>(std::sqrt(5.0)), "Two norm failed.");
  is_close_debug(two_norm_squared(v3), static_cast<C>(5.0), "Two norm squared failed.");
  is_close_debug(minor_angle(v3, v4), static_cast<C>(std::acos(11.0 / (std::sqrt(5.0) * std::sqrt(25.0)))),
                 "Minor angle failed.");
  is_close_debug(major_angle(v3, v4), static_cast<C>(M_PI - std::acos(11.0 / (std::sqrt(5.0) * std::sqrt(25.0)))),
                 "Major angle failed.");

  // Dim 3
  Vector3<T1> v5(1, 2, 3);
  Vector3<T2> v6(4, 5, 6);
  is_close_debug(dot(v5, v6), static_cast<C>(32.0), "Dot product failed.");
  is_close_debug(norm(v5), static_cast<C>(std::sqrt(14.0)), "Norm failed.");
  is_close_debug(norm_squared(v5), static_cast<C>(14.0), "Norm squared failed.");
  is_close_debug(inf_norm(v5), static_cast<C>(3.0), "Infinity norm failed.");
  is_close_debug(one_norm(v5), static_cast<C>(6.0), "One norm failed.");
  is_close_debug(two_norm(v5), static_cast<C>(std::sqrt(14.0)), "Two norm failed.");
  is_close_debug(two_norm_squared(v5), static_cast<C>(14.0), "Two norm squared failed.");
  is_close_debug(minor_angle(v5, v6), static_cast<C>(std::acos(32.0 / (std::sqrt(14.0) * std::sqrt(77.0)))),
                 "Minor angle failed.");
  is_close_debug(major_angle(v5, v6), static_cast<C>(M_PI - std::acos(32.0 / (std::sqrt(14.0) * std::sqrt(77.0)))),
                 "Major angle failed.");
}
//@}

//! \name Apply
//@{

struct an_external_functor {
  template <typename T>
  KOKKOS_INLINE_FUNCTION T operator()(const T& x) const {
    return x + 1;
  }
};

TYPED_TEST(VectorSingleTypeTest, Apply) {
  // Using a lambda function
  // Dim 1
  Vector1<TypeParam> v1(1);
  auto v2 = apply([](auto x) { return x + 1; }, v1);
  is_close_debug(v2, Vector1<TypeParam>{2}, "Apply failed.");

  // Dim 2
  Vector2<TypeParam> v3(1, 2);
  auto v4 = apply([](auto x) { return x + 1; }, v3);
  is_close_debug(v4, Vector2<TypeParam>{2, 3}, "Apply failed.");

  // Dim 3
  Vector3<TypeParam> v5(1, 2, 3);
  auto v6 = apply([](auto x) { return x + 1; }, v5);
  is_close_debug(v6, Vector3<TypeParam>{2, 3, 4}, "Apply failed.");

  // Using an external function
  // Dim 1
  Vector1<TypeParam> v7(1);
  auto v8 = apply(an_external_functor{}, v7);
  is_close_debug(v8, Vector1<TypeParam>{2}, "Apply failed.");

  // Dim 2
  Vector2<TypeParam> v9(1, 2);
  auto v10 = apply(an_external_functor{}, v9);
  is_close_debug(v10, Vector2<TypeParam>{2, 3}, "Apply failed.");

  // Dim 3
  Vector3<TypeParam> v11(1, 2, 3);
  auto v12 = apply(an_external_functor{}, v11);
  is_close_debug(v12, Vector3<TypeParam>{2, 3, 4}, "Apply failed.");

  // Using a temporary vector
  auto v13 = apply(an_external_functor{}, Vector1<TypeParam>{1});
  is_close_debug(v13, Vector1<TypeParam>{2}, "Apply failed.");
}

struct an_external_constexpr_functor {
  template <typename T>
  constexpr T operator()(const T& x) const {
    return x + 1;
  }
};

TYPED_TEST(VectorSingleTypeTest, ConstexprApply) {
  // Using a lambda function
  // Dim 1
  constexpr Vector1<TypeParam> v1(1);
  constexpr auto v2 = apply([](auto x) { return x + 1; }, v1);
  static_assert(std::abs(v2[0] - 2) < 1e-6, "Constexpr apply failed.");

  // Dim 2
  constexpr Vector2<TypeParam> v3(1, 2);
  constexpr auto v4 = apply([](auto x) { return x + 1; }, v3);
  static_assert(std::abs(v4[0] - 2) < 1e-6 && std::abs(v4[1] - 3) < 1e-6, "Constexpr apply failed.");

  // Dim 3
  constexpr Vector3<TypeParam> v5(1, 2, 3);
  constexpr auto v6 = apply([](auto x) { return x + 1; }, v5);
  static_assert(std::abs(v6[0] - 2) < 1e-6 && std::abs(v6[1] - 3) < 1e-6 && std::abs(v6[2] - 4) < 1e-6,
                "Constexpr apply failed.");

  // Using an external function
  // Dim 1
  constexpr Vector1<TypeParam> v7(1);
  constexpr auto v8 = apply(an_external_constexpr_functor{}, v7);
  static_assert(std::abs(v8[0] - 2) < 1e-6, "Constexpr apply failed.");

  // Dim 2
  constexpr Vector2<TypeParam> v9(1, 2);
  constexpr auto v10 = apply(an_external_constexpr_functor{}, v9);
  static_assert(std::abs(v10[0] - 2) < 1e-6 && std::abs(v10[1] - 3) < 1e-6, "Constexpr apply failed.");

  // Dim 3
  constexpr Vector3<TypeParam> v11(1, 2, 3);
  constexpr auto v12 = apply(an_external_constexpr_functor{}, v11);
  static_assert(std::abs(v12[0] - 2) < 1e-6 && std::abs(v12[1] - 3) < 1e-6 && std::abs(v12[2] - 4) < 1e-6,
                "Constexpr apply failed.");
}
//@}

//! \name Atomic operations
//@{

template <typename TypeParam>
bool check_vector_atomic_op_load_store_test_for_false_positive_1d() {
  Vector1<TypeParam> finished(false);
  auto func_no_atomic = [&finished]() {
    bool hit_max_loops = false;
    size_t max_loops = 1'000'000'000;
    size_t i = 0;
    while (!(finished[0])) {
      if (i > max_loops) {
        hit_max_loops = true;
        break;
      }
      ++i;
    }
    return hit_max_loops;
  };
  auto result = std::async(std::launch::async, func_no_atomic);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  finished[0] = true;
  bool false_positive = !result.get();
  return false_positive;
}

template <typename TypeParam>
bool check_vector_atomic_op_load_store_test_for_false_positive_3d() {
  Vector3<TypeParam> finished(false);
  auto func_no_atomic = [&finished]() {
    bool hit_max_loops = false;
    size_t max_loops = 1'000'000'000;
    size_t i = 0;
    while (!(finished[2])) {
      if (i > max_loops) {
        hit_max_loops = true;
        break;
      }
      ++i;
    }
    return hit_max_loops;
  };
  auto result = std::async(std::launch::async, func_no_atomic);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  finished[2] = true;
  bool false_positive = !result.get();
  return false_positive;
}

TYPED_TEST(VectorSingleTypeTest, AtomicOpTestLoadStore1D) {
  if (check_vector_atomic_op_load_store_test_for_false_positive_1d<TypeParam>()) {
    GTEST_SKIP() << "Skipping atomic load/store test due to false positive in non-atomic test.\n"
                 << "This typically occurs if you compile with -O0.";
  }

  Vector1<TypeParam> finished(false);
  auto func_atomic = [&finished]() {
    bool hit_max_loops = false;
    size_t max_loops = 1'000'000'000;
    size_t i = 0;
    while (!(atomic_load(&finished)[0])) {
      if (i > max_loops) {
        hit_max_loops = true;
        break;
      }
      ++i;
    }
    return hit_max_loops;
  };
  auto result = std::async(std::launch::async, func_atomic);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  atomic_store(&finished, true);
  EXPECT_FALSE(result.get()) << "Atomic load/store test failed.";
}

TYPED_TEST(VectorSingleTypeTest, AtomicOpTestLoadStore3D) {
  if (check_vector_atomic_op_load_store_test_for_false_positive_3d<TypeParam>()) {
    GTEST_SKIP() << "Skipping atomic load/store test due to false positive in non-atomic test.\n"
                 << "This typically occurs if you compile with -O0.";
  }

  Vector3<TypeParam> finished(false);
  auto func_atomic = [&finished]() {
    bool hit_max_loops = false;
    size_t max_loops = 1'000'000'000;
    size_t i = 0;
    while (!(atomic_load(&finished)[2])) {
      if (i > max_loops) {
        hit_max_loops = true;
        break;
      }
      ++i;
    }
    return hit_max_loops;
  };
  auto result = std::async(std::launch::async, func_atomic);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  atomic_store(&finished, true);
  EXPECT_FALSE(result.get()) << "Atomic load/store test failed.";
}

template <typename TypeParam>
bool check_vector_atomic_op_add_sub_mul_div_or_false_positive_1d() {
  int num_threads = 8;
  int num_iterations = 10001;  // Must be odd
  int num_steps_of_mul_div = 2;

  // Naming convention
  // vs: Vector-scalar operation
  // vv: Vector-vector operation
  // pos: Positive result using atomic operations
  // neg: Negative result without atomic operations
  Vector1<TypeParam> vs_add_neg(0);
  Vector1<TypeParam> vv_add_neg(0);
  Vector1<TypeParam> vs_sub_neg(0);
  Vector1<TypeParam> vv_sub_neg(0);
  Vector1<TypeParam> vs_mul_div_neg(1);
  Vector1<TypeParam> vv_mul_div_neg(1);

  // Thread function to perform atomic_add
  std::atomic<long long int> thread_id_counter(0);
  auto thread_func = [&]() {
    for (int i = 0; i < num_iterations; ++i) {
      vs_add_neg += 1;
      vv_add_neg += Vector1<TypeParam>(1);
      vs_sub_neg -= 1;
      vv_sub_neg -= Vector1<TypeParam>(1);

      if (i % num_steps_of_mul_div == 0) {
        vs_mul_div_neg *= static_cast<TypeParam>(2);
        vv_mul_div_neg = elementwise_mul(vv_mul_div_neg, Vector1<TypeParam>(2));
      } else {
        vs_mul_div_neg /= static_cast<TypeParam>(2);
        vv_mul_div_neg = elementwise_div(vv_mul_div_neg, Vector1<TypeParam>(2));
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify the result
  bool vs_add_false_positive = (vs_add_neg[0] == num_threads * num_iterations);
  bool vv_add_false_positive = (vv_add_neg[0] == num_threads * num_iterations);
  bool vs_sub_false_positive = (vs_sub_neg[0] == -num_threads * num_iterations);
  bool vv_sub_false_positive = (vv_sub_neg[0] == -num_threads * num_iterations);
  bool vs_mul_div_false_positive = (vs_mul_div_neg[0] == std::pow(2, num_threads));
  bool vv_mul_div_false_positive = (vv_mul_div_neg[0] == std::pow(2, num_threads));
  return vs_add_false_positive || vv_add_false_positive || vs_sub_false_positive || vv_sub_false_positive ||
         vs_mul_div_false_positive || vv_mul_div_false_positive;
}

TYPED_TEST(VectorSingleTypeTest, AtomicOpTestAddSubMulDiv1D) {
  if (check_vector_atomic_op_add_sub_mul_div_or_false_positive_1d<TypeParam>()) {
    GTEST_SKIP() << "Skipping atomic add/sub/mul/div test due to false positive in non-atomic test.\n"
                 << "This typically occurs if you compile with -O0.";
  }

  int num_threads = 8;
  int num_iterations = 10001;  // Must be odd
  int num_steps_of_mul_div = 2;

  // Naming convention
  // vs: Vector-scalar operation
  // vv: Vector-vector operation
  // pos: Positive result using atomic operations
  // neg: Negative result without atomic operations
  Vector1<TypeParam> vs_add_pos(0);
  Vector1<TypeParam> vv_add_pos(0);
  Vector1<TypeParam> vs_sub_pos(0);
  Vector1<TypeParam> vv_sub_pos(0);
  Vector1<TypeParam> vs_mul_div_pos(1);
  Vector1<TypeParam> vv_mul_div_pos(1);

  // Thread function to perform atomic_add
  std::atomic<long long int> thread_id_counter(0);
  auto thread_func = [&]() {
    for (int i = 0; i < num_iterations; ++i) {
      atomic_add(&vs_add_pos, 1);
      atomic_add(&vv_add_pos, Vector1<TypeParam>(1));
      atomic_sub(&vs_sub_pos, 1);
      atomic_sub(&vv_sub_pos, Vector1<TypeParam>(1));

      if (i % num_steps_of_mul_div == 0) {
        atomic_mul(&vs_mul_div_pos, static_cast<TypeParam>(2));
        atomic_elementwise_mul(&vv_mul_div_pos, Vector1<TypeParam>(2));
      } else {
        atomic_div(&vs_mul_div_pos, static_cast<TypeParam>(2));
        atomic_elementwise_div(&vv_mul_div_pos, Vector1<TypeParam>(2));
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify the result
  EXPECT_EQ(vs_add_pos[0], num_threads * num_iterations) << "Atomic add failed.";
  EXPECT_EQ(vv_add_pos[0], num_threads * num_iterations) << "Atomic add failed.";
  EXPECT_EQ(vs_sub_pos[0], -num_threads * num_iterations) << "Atomic sub failed.";
  EXPECT_EQ(vv_sub_pos[0], -num_threads * num_iterations) << "Atomic sub failed.";
  EXPECT_EQ(vs_mul_div_pos[0], std::pow(2, num_threads)) << "Atomic mul/div failed.";
  EXPECT_EQ(vv_mul_div_pos[0], std::pow(2, num_threads)) << "Atomic mul/div failed.";
}

TYPED_TEST(VectorSingleTypeTest, AtomicOpTestAddSubMulDiv3D) {
  if (check_vector_atomic_op_add_sub_mul_div_or_false_positive_1d<TypeParam>()) {
    GTEST_SKIP() << "Skipping atomic add/sub/mul/div test due to false positive in non-atomic test.\n"
                 << "This typically occurs if you compile with -O0.";
  }

  int num_threads = 8;
  int num_iterations = 100001;  // Must be odd
  int num_steps_of_mul_div = 2;

  // Naming convention
  // vs: Vector-scalar operation
  // vv: Vector-vector operation
  // pos: Positive result using atomic operations
  // neg: Negative result without atomic operations
  Vector3<TypeParam> vs_add_pos(0, 1, 2);
  Vector3<TypeParam> vv_add_pos(0, 1, 2);
  Vector3<TypeParam> vs_sub_pos(0, 1, 2);
  Vector3<TypeParam> vv_sub_pos(0, 1, 2);
  Vector3<TypeParam> vs_mul_div_pos(1, 2, 3);
  Vector3<TypeParam> vv_mul_div_pos(1, 2, 3);

  // Thread function to perform atomic_add
  std::atomic<long long int> thread_id_counter(0);
  auto thread_func = [&]() {
    for (int i = 0; i < num_iterations; ++i) {
      atomic_add(&vs_add_pos, 1);
      atomic_add(&vv_add_pos, Vector3<TypeParam>(1, 2, 3));
      atomic_sub(&vs_sub_pos, 1);
      atomic_sub(&vv_sub_pos, Vector3<TypeParam>(1, 2, 3));

      if (i % num_steps_of_mul_div == 0) {
        atomic_mul(&vs_mul_div_pos, static_cast<TypeParam>(2));
        atomic_elementwise_mul(&vv_mul_div_pos, Vector3<TypeParam>(2, 3, 4));
      } else {
        atomic_div(&vs_mul_div_pos, static_cast<TypeParam>(2));
        atomic_elementwise_div(&vv_mul_div_pos, Vector3<TypeParam>(2, 3, 4));
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify the result
  EXPECT_EQ(vs_add_pos[0], 0 + num_threads * num_iterations) << "Atomic add failed.";
  EXPECT_EQ(vs_add_pos[1], 1 + num_threads * num_iterations) << "Atomic add failed.";
  EXPECT_EQ(vs_add_pos[2], 2 + num_threads * num_iterations) << "Atomic add failed.";

  EXPECT_EQ(vv_add_pos[0], 0 + 1 * num_threads * num_iterations) << "Atomic add failed.";
  EXPECT_EQ(vv_add_pos[1], 1 + 2 * num_threads * num_iterations) << "Atomic add failed.";
  EXPECT_EQ(vv_add_pos[2], 2 + 3 * num_threads * num_iterations) << "Atomic add failed.";

  EXPECT_EQ(vs_sub_pos[0], 0 - num_threads * num_iterations) << "Atomic sub failed.";
  EXPECT_EQ(vs_sub_pos[1], 1 - num_threads * num_iterations) << "Atomic sub failed.";
  EXPECT_EQ(vs_sub_pos[2], 2 - num_threads * num_iterations) << "Atomic sub failed.";

  EXPECT_EQ(vv_sub_pos[0], 0 - 1 * num_threads * num_iterations) << "Atomic sub failed.";
  EXPECT_EQ(vv_sub_pos[1], 1 - 2 * num_threads * num_iterations) << "Atomic sub failed.";
  EXPECT_EQ(vv_sub_pos[2], 2 - 3 * num_threads * num_iterations) << "Atomic sub failed.";

  EXPECT_EQ(vs_mul_div_pos[0], 1 * std::pow(2, num_threads)) << "Atomic mul/div failed.";
  EXPECT_EQ(vs_mul_div_pos[1], 2 * std::pow(2, num_threads)) << "Atomic mul/div failed.";
  EXPECT_EQ(vs_mul_div_pos[2], 3 * std::pow(2, num_threads)) << "Atomic mul/div failed.";

  EXPECT_EQ(vv_mul_div_pos[0], 1 * std::pow(2, num_threads)) << "Atomic mul/div failed.";
  EXPECT_EQ(vv_mul_div_pos[1], 2 * std::pow(3, num_threads)) << "Atomic mul/div failed.";
  EXPECT_EQ(vv_mul_div_pos[2], 3 * std::pow(4, num_threads)) << "Atomic mul/div failed.";
}

TYPED_TEST(VectorSingleTypeTest, AtomicFetchOpTestAddSubMulDiv1D) {
  if (check_vector_atomic_op_add_sub_mul_div_or_false_positive_1d<TypeParam>()) {
    GTEST_SKIP() << "Skipping atomic add/sub/mul/div test due to false positive in non-atomic test.\n"
                 << "This typically occurs if you compile with -O0.";
  }

  int num_threads = 8;
  int num_iterations = 100001;  // Must be odd
  int num_steps_of_mul_div = 2;

  // Naming convention
  // vs: Vector-scalar operation
  // vv: Vector-vector operation
  // pos: Positive result using atomic operations
  // neg: Negative result without atomic operations
  Vector1<TypeParam> vs_add_pos(0);
  Vector1<TypeParam> vv_add_pos(0);
  Vector1<TypeParam> vs_sub_pos(0);
  Vector1<TypeParam> vv_sub_pos(0);
  Vector1<TypeParam> vs_mul_div_pos(1);
  Vector1<TypeParam> vv_mul_div_pos(1);

  std::vector<int> vs_add_pos_counts(num_iterations * num_threads, 0);
  std::vector<int> vv_add_pos_counts(num_iterations * num_threads, 0);
  std::vector<int> vs_sub_pos_counts(num_iterations * num_threads, 0);
  std::vector<int> vv_sub_pos_counts(num_iterations * num_threads, 0);

  int total_vs_mul_div_pos_count = 0;
  int total_vv_mul_div_pos_count = 0;

  // Thread function to perform atomic_add
  std::atomic<long long int> thread_id_counter(0);
  auto thread_func = [&]() {
    for (int i = 0; i < num_iterations; ++i) {
      Vector1<TypeParam> old_vs_add_pos = atomic_fetch_add(&vs_add_pos, 1);
      Vector1<TypeParam> old_vv_add_pos = atomic_fetch_add(&vv_add_pos, Vector1<TypeParam>(1));

      int old_vs_add_pos_index = static_cast<int>(old_vs_add_pos[0]);
      int old_vv_add_pos_index = static_cast<int>(old_vv_add_pos[0]);
      if (old_vs_add_pos_index >= 0 && old_vs_add_pos_index < num_iterations * num_threads) {
        vs_add_pos_counts[old_vs_add_pos_index] += 1;
      }
      if (old_vv_add_pos_index >= 0 && old_vv_add_pos_index < num_iterations * num_threads) {
        vv_add_pos_counts[old_vv_add_pos_index] += 1;
      }

      Vector1<TypeParam> old_vs_sub_pos = atomic_fetch_sub(&vs_sub_pos, 1);
      Vector1<TypeParam> old_vv_sub_pos = atomic_fetch_sub(&vv_sub_pos, Vector1<TypeParam>(1));

      int old_vs_sub_pos_index = -static_cast<int>(old_vs_sub_pos[0]);
      int old_vv_sub_pos_index = -static_cast<int>(old_vv_sub_pos[0]);
      if (old_vs_sub_pos_index >= 0 && old_vs_sub_pos_index < num_iterations * num_threads) {
        vs_sub_pos_counts[old_vs_sub_pos_index] += 1;
      }
      if (old_vv_sub_pos_index >= 0 && old_vv_sub_pos_index < num_iterations * num_threads) {
        vv_sub_pos_counts[old_vv_sub_pos_index] += 1;
      }

      Vector1<TypeParam> old_vs_mul_div_pos;
      Vector1<TypeParam> old_vv_mul_div_pos;
      if (i % num_steps_of_mul_div == 0) {
        old_vs_mul_div_pos = atomic_fetch_mul(&vs_mul_div_pos, static_cast<TypeParam>(2));
        old_vv_mul_div_pos = atomic_fetch_elementwise_mul(&vv_mul_div_pos, Vector1<TypeParam>(2));
      } else {
        old_vs_mul_div_pos = atomic_fetch_div(&vs_mul_div_pos, static_cast<TypeParam>(2));
        old_vv_mul_div_pos = atomic_fetch_elementwise_div(&vv_mul_div_pos, Vector1<TypeParam>(2));
      }

      int old_vs_mul_div_pos_index = static_cast<int>(std::log2(old_vs_mul_div_pos[0]));
      int old_vv_mul_div_pos_index = static_cast<int>(std::log2(old_vv_mul_div_pos[0]));
      if (old_vs_mul_div_pos_index >= 0 && old_vs_mul_div_pos_index <= num_threads) {
        Kokkos::atomic_add(&total_vs_mul_div_pos_count, 1);
      }
      if (old_vv_mul_div_pos_index >= 0 && old_vv_mul_div_pos_index <= num_threads) {
        Kokkos::atomic_add(&total_vv_mul_div_pos_count, 1);
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify the results
  EXPECT_EQ(vs_add_pos[0], num_threads * num_iterations) << "Atomic fetch add failed.";
  EXPECT_EQ(vv_add_pos[0], num_threads * num_iterations) << "Atomic fetch add failed.";
  EXPECT_EQ(vs_sub_pos[0], -num_threads * num_iterations) << "Atomic fetch sub failed.";
  EXPECT_EQ(vv_sub_pos[0], -num_threads * num_iterations) << "Atomic fetch sub failed.";
  EXPECT_EQ(vs_mul_div_pos[0], std::pow(2, num_threads)) << "Atomic fetch mul/div failed.";
  EXPECT_EQ(vv_mul_div_pos[0], std::pow(2, num_threads)) << "Atomic fetch mul/div failed.";

  // Verify the fetch counts
  bool vs_add_pos_fetch_passed = true;
  bool vv_add_pos_fetch_passed = true;
  bool vs_sub_pos_fetch_passed = true;
  bool vv_sub_pos_fetch_passed = true;

  for (int i = 0; i < num_iterations * num_threads; ++i) {
    // Add
    if (vs_add_pos_counts[i] != 1) {
      vs_add_pos_fetch_passed = false;
    }
    if (vv_add_pos_counts[i] != 1) {
      vv_add_pos_fetch_passed = false;
    }

    // Sub
    if (vs_sub_pos_counts[i] != 1) {
      vs_sub_pos_fetch_passed = false;
    }
    if (vv_sub_pos_counts[i] != 1) {
      vv_sub_pos_fetch_passed = false;
    }
  }

  EXPECT_TRUE(vs_add_pos_fetch_passed) << "Atomic fetch add failed to properly return old value.";
  EXPECT_TRUE(vv_add_pos_fetch_passed) << "Atomic fetch add failed to properly return old value.";
  EXPECT_TRUE(vs_sub_pos_fetch_passed) << "Atomic fetch sub failed to properly return old value.";
  EXPECT_TRUE(vv_sub_pos_fetch_passed) << "Atomic fetch sub failed to properly return old value.";

  // Addition and multiplication are communicative operations, so the best we can do is check the total number of
  // operations.
  int expected_num_occurrences = num_threads * num_iterations;
  EXPECT_EQ(total_vs_mul_div_pos_count, expected_num_occurrences)
      << "Atomic fetch mul/div failed to properly return old value.";
  EXPECT_EQ(total_vv_mul_div_pos_count, expected_num_occurrences)
      << "Atomic fetch mul/div failed to properly return old value.";
}

TYPED_TEST(VectorSingleTypeTest, AtomicFetchOpTestAddSubMulDiv3D) {
  if (check_vector_atomic_op_add_sub_mul_div_or_false_positive_1d<TypeParam>()) {
    GTEST_SKIP() << "Skipping atomic add/sub/mul/div test due to false positive in non-atomic test.\n"
                 << "This typically occurs if you compile with -O0.";
  }

  int num_threads = 8;
  int num_iterations = 10001;  // Must be odd
  int num_steps_of_mul_div = 2;

  // Naming convention
  // vs: Vector-scalar operation
  // vv: Vector-vector operation
  // pos: Positive result using atomic operations
  // neg: Negative result without atomic operations
  Vector3<TypeParam> vs_add_pos(0, 1, 2);
  Vector3<TypeParam> vv_add_pos(0, 1, 2);
  Vector3<TypeParam> vs_sub_pos(0, 1, 2);
  Vector3<TypeParam> vv_sub_pos(0, 1, 2);
  Vector3<TypeParam> vs_mul_div_pos(1, 2, 3);
  Vector3<TypeParam> vv_mul_div_pos(1, 2, 3);

  // Thread function to perform atomic_add
  std::atomic<long long int> thread_id_counter(0);
  auto thread_func = [&]() {
    for (int i = 0; i < num_iterations; ++i) {
      atomic_fetch_add(&vs_add_pos, 1);
      atomic_fetch_add(&vv_add_pos, Vector3<TypeParam>(1, 2, 3));
      atomic_fetch_sub(&vs_sub_pos, 1);
      atomic_fetch_sub(&vv_sub_pos, Vector3<TypeParam>(1, 2, 3));

      if (i % num_steps_of_mul_div == 0) {
        atomic_fetch_mul(&vs_mul_div_pos, static_cast<TypeParam>(2));
        atomic_fetch_elementwise_mul(&vv_mul_div_pos, Vector3<TypeParam>(2, 3, 4));
      } else {
        atomic_fetch_div(&vs_mul_div_pos, static_cast<TypeParam>(2));
        atomic_fetch_elementwise_div(&vv_mul_div_pos, Vector3<TypeParam>(2, 3, 4));
      }
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }

  // Join threads
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify the result
  EXPECT_EQ(vs_add_pos[0], 0 + num_threads * num_iterations) << "Atomic fetch add failed.";
  EXPECT_EQ(vs_add_pos[1], 1 + num_threads * num_iterations) << "Atomic fetch add failed.";
  EXPECT_EQ(vs_add_pos[2], 2 + num_threads * num_iterations) << "Atomic fetch add failed.";

  EXPECT_EQ(vv_add_pos[0], 0 + 1 * num_threads * num_iterations) << "Atomic fetch add failed.";
  EXPECT_EQ(vv_add_pos[1], 1 + 2 * num_threads * num_iterations) << "Atomic fetch add failed.";
  EXPECT_EQ(vv_add_pos[2], 2 + 3 * num_threads * num_iterations) << "Atomic fetch add failed.";

  EXPECT_EQ(vs_sub_pos[0], 0 - num_threads * num_iterations) << "Atomic fetch sub failed.";
  EXPECT_EQ(vs_sub_pos[1], 1 - num_threads * num_iterations) << "Atomic fetch sub failed.";
  EXPECT_EQ(vs_sub_pos[2], 2 - num_threads * num_iterations) << "Atomic fetch sub failed.";

  EXPECT_EQ(vv_sub_pos[0], 0 - 1 * num_threads * num_iterations) << "Atomic fetch sub failed.";
  EXPECT_EQ(vv_sub_pos[1], 1 - 2 * num_threads * num_iterations) << "Atomic fetch sub failed.";
  EXPECT_EQ(vv_sub_pos[2], 2 - 3 * num_threads * num_iterations) << "Atomic fetch sub failed.";

  EXPECT_EQ(vs_mul_div_pos[0], 1 * std::pow(2, num_threads)) << "Atomic fetch mul/div failed.";
  EXPECT_EQ(vs_mul_div_pos[1], 2 * std::pow(2, num_threads)) << "Atomic fetch mul/div failed.";
  EXPECT_EQ(vs_mul_div_pos[2], 3 * std::pow(2, num_threads)) << "Atomic fetch mul/div failed.";

  EXPECT_EQ(vv_mul_div_pos[0], 1 * std::pow(2, num_threads)) << "Atomic fetch mul/div failed.";
  EXPECT_EQ(vv_mul_div_pos[1], 2 * std::pow(3, num_threads)) << "Atomic fetch mul/div failed.";
  EXPECT_EQ(vv_mul_div_pos[2], 3 * std::pow(4, num_threads)) << "Atomic fetch mul/div failed.";
}
//@}

//! \name Vector Views
//@{

TYPED_TEST(VectorSingleTypeTest, Views) {
  // Pointers are valid for views, as their copy constructor performs a shallow copy
  {
    std::vector<TypeParam> std_vec1{0, 0, 1, 2, 3, 0, 0};
    // Dim 1
    auto v2 = get_vector_view<TypeParam, 1>(std_vec1.data() + 2);
    is_close_debug(v2[0], 1, "1D pointer view failed.");
    auto ptr_before = std_vec1.data();
    std_vec1 = {0, 0, 2, 3, 4, 0, 0};
    auto ptr_after = std_vec1.data();
    ASSERT_EQ(ptr_before, ptr_after);
    is_close_debug(v2[0], 2, "1D pointer view not a view.");

    // Dim 2
    auto v3 = get_vector_view<TypeParam, 2>(std_vec1.data() + 2);
    is_close_debug(v3[0], 2, "2D pointer view failed.");
    is_close_debug(v3[1], 3, "2D pointer view failed.");
    ptr_before = std_vec1.data();
    std_vec1 = {0, 0, 3, 4, 5, 0, 0};
    ptr_after = std_vec1.data();
    ASSERT_EQ(ptr_before, ptr_after);
    is_close_debug(v3[0], 3, "2D pointer view not a view.");
    is_close_debug(v3[1], 4, "2D pointer view not a view.");

    // Dim 3
    auto v4 = get_vector_view<TypeParam, 3>(std_vec1.data() + 2);
    is_close_debug(v4[0], 3, "3D pointer view failed.");
    is_close_debug(v4[1], 4, "3D pointer view failed.");
    is_close_debug(v4[2], 5, "3D pointer view failed.");
    ptr_before = std_vec1.data();
    std_vec1 = {0, 0, 4, 5, 6, 0, 0};
    ptr_after = std_vec1.data();
    ASSERT_EQ(ptr_before, ptr_after);
    is_close_debug(v4[0], 4, "3D pointer view not a view.");
    is_close_debug(v4[1], 5, "3D pointer view not a view.");
    is_close_debug(v4[2], 6, "3D pointer view not a view.");
  }

  // Array's are also valid for views, as we store a reference to the original data rather than performing a copy.
  // We will illustrate this with std::array<TypeParam, N>
  {
    // Dim 1
    Kokkos::Array<TypeParam, 1> array1{1};
    auto v1 = get_vector_view<TypeParam, 1>(array1);
    is_close_debug(v1[0], 1, "1D array view failed.");
    array1[0] = 2;
    is_close_debug(v1[0], 2, "1D array view somehow not not a view.");

    // Dim 2
    Kokkos::Array<TypeParam, 2> array2{1, 2};
    auto v2 = get_vector_view<TypeParam, 2>(array2);
    is_close_debug(v2[0], 1, "2D array view failed.");
    is_close_debug(v2[1], 2, "2D array view failed.");
    array2[0] = 3;
    array2[1] = 4;
    is_close_debug(v2[0], 3, "2D array view somehow not a view.");
    is_close_debug(v2[1], 4, "2D array view somehow not a view.");

    // Dim 3
    Kokkos::Array<TypeParam, 3> array3{1, 2, 3};
    auto v3 = get_vector_view<TypeParam, 3>(array3);
    is_close_debug(v3[0], 1, "3D array view failed.");
    is_close_debug(v3[1], 2, "3D array view failed.");
    is_close_debug(v3[2], 3, "3D array view failed.");
    array3[0] = 4;
    array3[1] = 5;
    array3[2] = 6;
    is_close_debug(v3[0], 4, "3D array view somehow not a view.");
    is_close_debug(v3[1], 5, "3D array view somehow not a view.");
    is_close_debug(v3[2], 6, "3D array view somehow not a view.");
  }
}
//@}

}  // namespace

}  // namespace math

}  // namespace mundy
