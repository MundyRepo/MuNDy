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
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
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

//! \name Helper typedefs
//@{

// \note The following typedefs are used because GTEST's macros interpret the comma in the template as separating
// arguments
template <typename T>
using OurVector1 = Vector<T, 1>;

template <typename T>
using OurVector2 = Vector<T, 2>;

template <typename T>
using OurVector3 = Vector<T, 3>;
//@}

//! \name Vector Constructors and Destructor (in dims 1, 2, and 3)
//@{

TYPED_TEST(VectorSingleTypeTest, DefaultConstructor) {
  ASSERT_NO_THROW(OurVector1<TypeParam>());
  ASSERT_NO_THROW(OurVector2<TypeParam>());
  ASSERT_NO_THROW(OurVector3<TypeParam>());
}

TYPED_TEST(VectorSingleTypeTest, LiteralConstruction) {
  OurVector1<TypeParam>{1};
  OurVector2<TypeParam>{1, 2};
  OurVector3<TypeParam>{1, 2, 3};
  constexpr OurVector1<TypeParam> v1{1};
  constexpr OurVector2<TypeParam> v2{1, 2};
  constexpr OurVector3<TypeParam> v3{1, 2, 3};
}

TYPED_TEST(VectorSingleTypeTest, ConstructorFromNScalars) {
  ASSERT_NO_THROW(OurVector1<TypeParam>(1));
  ASSERT_NO_THROW(OurVector2<TypeParam>(1, 2));
  ASSERT_NO_THROW(OurVector3<TypeParam>(1, 2, 3));
  OurVector1<TypeParam> v1(1);
  OurVector2<TypeParam> v2(1, 2);
  OurVector3<TypeParam> v3(1, 2, 3);
  is_close_debug(v1[0], 1);
  is_close_debug(v2[0], 1);
  is_close_debug(v2[1], 2);
  is_close_debug(v3[0], 1);
  is_close_debug(v3[1], 2);
  is_close_debug(v3[2], 3);
}

TYPED_TEST(VectorSingleTypeTest, Comparison) {
  OurVector1<TypeParam> v1{1};
  OurVector1<TypeParam> v2{2};
  EXPECT_TRUE(is_close(v1, v1));
  EXPECT_FALSE(is_close(v1, v2));
  is_close_debug(v1, v1);
  is_close_debug(v1, OurVector1<TypeParam>{1});

  OurVector2<TypeParam> v3{1, 2};
  OurVector2<TypeParam> v4{2, 3};
  EXPECT_TRUE(is_close(v3, v3));
  EXPECT_FALSE(is_close(v3, v4));
  is_close_debug(v3, v3);
  is_close_debug(v3, OurVector2<TypeParam>{1, 2});

  OurVector3<TypeParam> v5{1, 2, 3};
  OurVector3<TypeParam> v6{2, 3, 4};
  EXPECT_TRUE(is_close(v5, v5));
  EXPECT_FALSE(is_close(v5, v6));
  is_close_debug(v5, v5);
  is_close_debug(v5, OurVector3<TypeParam>{1, 2, 3});
}

TYPED_TEST(VectorSingleTypeTest, CopyConstructor) {
  OurVector1<TypeParam> v1{1};
  OurVector1<TypeParam> v2(v1);
  is_close_debug(v1, v2, "Copy constructor failed.");
  v1.set(2);
  is_different_debug(v1, v2, "Copy constructor failed, somehow the data is shared.");

  OurVector2<TypeParam> v3{1, 2};
  OurVector2<TypeParam> v4(v3);
  is_close_debug(v3, v4, "Copy constructor failed.");
  v3.set(3, 4);
  is_different_debug(v3, v4, "Copy constructor failed, somehow the data is shared.");

  OurVector3<TypeParam> v5{1, 2, 3};
  OurVector3<TypeParam> v6(v5);
  is_close_debug(v5, v6, "Copy constructor failed.");
  v5.set(4, 5, 6);
  is_different_debug(v5, v6, "Copy constructor failed, somehow the data is shared.");
}

TYPED_TEST(VectorSingleTypeTest, MoveConstructor) {
  OurVector1<TypeParam> v1{1};
  OurVector1<TypeParam> v2(std::move(v1));
  is_close_debug(v2, OurVector1<TypeParam>{1}, "Move constructor failed.");

  OurVector2<TypeParam> v3{1, 2};
  OurVector2<TypeParam> v4(std::move(v3));
  is_close_debug(v4, OurVector2<TypeParam>{1, 2}, "Move constructor failed.");

  OurVector3<TypeParam> v5{1, 2, 3};
  OurVector3<TypeParam> v6(std::move(v5));
  is_close_debug(v6, OurVector3<TypeParam>{1, 2, 3}, "Move constructor failed.");
}

TYPED_TEST(VectorSingleTypeTest, CopyAssignment) {
  OurVector1<TypeParam> v1{1};
  OurVector1<TypeParam> v2{2};
  ASSERT_NO_THROW(v2 = v1);
  is_close_debug(v1, v2, "Copy assignment failed.");

  OurVector2<TypeParam> v3{1, 2};
  OurVector2<TypeParam> v4{3, 4};
  ASSERT_NO_THROW(v4 = v3);
  is_close_debug(v3, v4, "Copy assignment failed.");

  OurVector3<TypeParam> v5{1, 2, 3};
  OurVector3<TypeParam> v6{4, 5, 6};
  ASSERT_NO_THROW(v6 = v5);
  is_close_debug(v5, v6, "Copy assignment failed.");
}

TYPED_TEST(VectorSingleTypeTest, MoveAssignment) {
  OurVector1<TypeParam> v1{1};
  OurVector1<TypeParam> v2{2};
  ASSERT_NO_THROW(v2 = std::move(v1));
  is_close_debug(v2, OurVector1<TypeParam>{1}, "Move assignment failed.");

  OurVector2<TypeParam> v3{1, 2};
  OurVector2<TypeParam> v4{3, 4};
  ASSERT_NO_THROW(v4 = std::move(v3));
  is_close_debug(v4, OurVector2<TypeParam>{1, 2}, "Move assignment failed.");

  OurVector3<TypeParam> v5{1, 2, 3};
  OurVector3<TypeParam> v6{4, 5, 6};
  ASSERT_NO_THROW(v6 = std::move(v5));
  is_close_debug(v6, OurVector3<TypeParam>{1, 2, 3}, "Move assignment failed.");
}

TYPED_TEST(VectorSingleTypeTest, Destructor) {
  ASSERT_NO_THROW(OurVector1<TypeParam>());
  ASSERT_NO_THROW(OurVector2<TypeParam>());
  ASSERT_NO_THROW(OurVector3<TypeParam>());
}
//@}

//! \name Vector Accessors (in dims 1, 2, and 3)
//@{

TYPED_TEST(VectorSingleTypeTest, Accessors) {
  OurVector1<TypeParam> v1(1);
  is_close_debug(v1[0], 1);
  v1[0] = 2;
  is_close_debug(v1[0], 2);

  OurVector2<TypeParam> v2(1, 2);
  is_close_debug(v2[0], 1);
  is_close_debug(v2[1], 2);
  v2[0] = 3;
  v2[1] = 4;
  is_close_debug(v2[0], 3);
  is_close_debug(v2[1], 4);

  OurVector3<TypeParam> v3(1, 2, 3);
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
  OurVector1<TypeParam> v1;
  v1.set(1);
  is_close_debug(v1, OurVector1<TypeParam>{1}, "Set by scalar failed.");

  v1.set(OurVector1<TypeParam>{2});
  is_close_debug(v1, OurVector1<TypeParam>{2}, "Set by vector failed.");

  v1.fill(3);
  is_close_debug(v1, OurVector1<TypeParam>{3}, "Fill failed.");

  // Dim 2
  OurVector2<TypeParam> v2;
  v2.set(1, 2);
  is_close_debug(v2, OurVector2<TypeParam>{1, 2}, "Set by two scalars failed.");

  v2.set(OurVector2<TypeParam>{3, 4});
  is_close_debug(v2, OurVector2<TypeParam>{3, 4}, "Set by vector failed.");

  v2.fill(5);
  is_close_debug(v2, OurVector2<TypeParam>{5, 5}, "Fill failed.");

  // Dim 3
  OurVector3<TypeParam> v3;
  v3.set(1, 2, 3);
  is_close_debug(v3, OurVector3<TypeParam>{1, 2, 3}, "Set by three scalars failed.");

  v3.set(OurVector3<TypeParam>{4, 5, 6});
  is_close_debug(v3, OurVector3<TypeParam>{4, 5, 6}, "Set by vector failed.");

  v3.fill(7);
  is_close_debug(v3, OurVector3<TypeParam>{7, 7, 7}, "Fill failed.");
}
//@}

//! \name Vector Special vectors
//@{

TYPED_TEST(VectorSingleTypeTest, SpecialVectors) {
  ASSERT_NO_THROW(OurVector1<TypeParam>::zeros());
  ASSERT_NO_THROW(OurVector1<TypeParam>::ones());
  auto ones1 = OurVector1<TypeParam>::ones();
  auto zeros1 = OurVector1<TypeParam>::zeros();

  is_close_debug(ones1, OurVector1<TypeParam>{1}, "Ones failed.");
  is_close_debug(zeros1, OurVector1<TypeParam>{0}, "Zeros failed.");

  ASSERT_NO_THROW(OurVector2<TypeParam>::zeros());
  ASSERT_NO_THROW(OurVector2<TypeParam>::ones());
  auto ones2 = OurVector2<TypeParam>::ones();
  auto zeros2 = OurVector2<TypeParam>::zeros();

  is_close_debug(ones2, OurVector2<TypeParam>{1, 1}, "Ones failed.");
  is_close_debug(zeros2, OurVector2<TypeParam>{0, 0}, "Zeros failed.");

  ASSERT_NO_THROW(OurVector3<TypeParam>::zeros());
  ASSERT_NO_THROW(OurVector3<TypeParam>::ones());
  auto ones3 = OurVector3<TypeParam>::ones();
  auto zeros3 = OurVector3<TypeParam>::zeros();

  is_close_debug(ones3, OurVector3<TypeParam>{1, 1, 1}, "Ones failed.");
  is_close_debug(zeros3, OurVector3<TypeParam>{0, 0, 0}, "Zeros failed.");
}
//@}

//! \name Vector Addition and subtraction
//@{

TYPED_TEST(VectorPairwiseTypeTest, AdditionAndSubtractionWithVector) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  OurVector1<T1> v1(1);
  OurVector1<T2> v2(2);
  auto v3 = v1 + v2;
  using T3 = decltype(v3)::scalar_t;
  is_close_debug(v3, OurVector1<T3>{3}, "Vector-vector addition failed.");

  v1 += v2;
  is_close_debug(v1, OurVector1<T1>{3}, "Vector-vector addition assignment failed.");

  v3 = v1 - v2;
  is_close_debug(v3, OurVector1<T3>{1}, "Vector-vector subtraction failed.");

  v1 -= v2;
  is_close_debug(v1, OurVector1<T1>{1}, "Vector-vector subtraction assignment failed.");

  // Dim 2
  OurVector2<T1> v4(1, 2);
  OurVector2<T2> v5(3, 4);
  auto v6 = v4 + v5;
  using T4 = decltype(v6)::scalar_t;
  is_close_debug(v6, OurVector2<T4>{4, 6}, "Vector-vector addition failed.");

  v4 += v5;
  is_close_debug(v4, OurVector2<T1>{4, 6}, "Vector-vector addition assignment failed.");

  v6 = v4 - v5;
  is_close_debug(v6, OurVector2<T4>{1, 2}, "Vector-vector subtraction failed.");

  v4 -= v5;
  is_close_debug(v4, OurVector2<T1>{1, 2}, "Vector-vector subtraction assignment failed.");

  // Dim 3
  OurVector3<T1> v7(1, 2, 3);
  OurVector3<T2> v8(4, 5, 6);
  auto v9 = v7 + v8;
  using T5 = decltype(v9)::scalar_t;
  is_close_debug(v9, OurVector3<T5>{5, 7, 9}, "Vector-vector addition failed.");

  v7 += v8;
  is_close_debug(v7, OurVector3<T1>{5, 7, 9}, "Vector-vector addition assignment failed.");

  v9 = v7 - v8;
  is_close_debug(v9, OurVector3<T5>{1, 2, 3}, "Vector-vector subtraction failed.");

  v7 -= v8;
  is_close_debug(v7, OurVector3<T1>{1, 2, 3}, "Vector-vector subtraction assignment failed.");
}
TYPED_TEST(VectorPairwiseTypeTest, AdditionAndSubtractionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  OurVector1<T1> v1(1);
  auto v2 = v1 + T2(1);
  using T3 = decltype(v2)::scalar_t;
  is_close_debug(v1 + T2(1), OurVector1<T3>{2}, "Vector-scalar addition failed.");
  is_close_debug(T2(1) + v1, OurVector1<T3>{2}, "Scalar-vector addition failed.");
  is_close_debug(v1 - T2(1), OurVector1<T3>{0}, "Vector-scalar subtraction failed.");
  is_close_debug(T2(1) - v1, OurVector1<T3>{0}, "Scalar-vector subtraction failed.");

  // Dim 2
  OurVector2<T1> v3(1, 2);
  auto v4 = v3 + T2(1);
  using T4 = decltype(v4)::scalar_t;
  is_close_debug(v4, OurVector2<T4>{2, 3}, "Vector-scalar addition failed.");
  is_close_debug(T2(1) + v3, OurVector2<T4>{2, 3}, "Scalar-vector addition failed.");
  is_close_debug(v3 - T2(1), OurVector2<T4>{0, 1}, "Vector-scalar subtraction failed.");
  is_close_debug(T2(1) - v3, OurVector2<T4>{0, -1}, "Scalar-vector subtraction failed.");

  // Dim 3
  OurVector3<T1> v5(1, 2, 3);
  auto v6 = v5 + T2(1);
  using T5 = decltype(v6)::scalar_t;
  is_close_debug(v6, OurVector3<T5>{2, 3, 4}, "Vector-scalar addition failed.");
  is_close_debug(T2(1) + v5, OurVector3<T5>{2, 3, 4}, "Scalar-vector addition failed.");
  is_close_debug(v5 - T2(1), OurVector3<T5>{0, 1, 2}, "Vector-scalar subtraction failed.");
  is_close_debug(T2(1) - v5, OurVector3<T5>{0, -1, -2}, "Scalar-vector subtraction failed.");
}

TYPED_TEST(VectorPairwiseTypeTest, AdditionAndSubtractionRValues) {
  // Test that the addition and subtraction operators work with rvalues
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  OurVector1<T1> v1(1);
  is_close_debug(v1 + OurVector1<T2>{2}, OurVector1<T1>{3}, "Vector-vector addition failed.");
  is_close_debug(OurVector1<T2>{2} + v1, OurVector1<T1>{3}, "Vector-vector addition failed.");
  is_close_debug(v1 - OurVector1<T2>{2}, OurVector1<T1>{-1}, "Vector-vector subtraction failed.");
  is_close_debug(OurVector1<T2>{2} - v1, OurVector1<T1>{1}, "Vector-vector subtraction failed.");

  // Dim 2
  OurVector2<T1> v2(1, 2);
  is_close_debug(v2 + OurVector2<T2>{3, 4}, OurVector2<T1>{4, 6}, "Vector-vector addition failed.");
  is_close_debug(OurVector2<T2>{3, 4} + v2, OurVector2<T1>{4, 6}, "Vector-vector addition failed.");
  is_close_debug(v2 - OurVector2<T2>{3, 4}, OurVector2<T1>{-2, -2}, "Vector-vector subtraction failed.");
  is_close_debug(OurVector2<T2>{3, 4} - v2, OurVector2<T1>{2, 2}, "Vector-vector subtraction failed.");

  // Dim 3
  OurVector3<T1> v3(1, 2, 3);
  is_close_debug(v3 + OurVector3<T2>{4, 5, 6}, OurVector3<T1>{5, 7, 9}, "Vector-vector addition failed.");
  is_close_debug(OurVector3<T2>{4, 5, 6} + v3, OurVector3<T1>{5, 7, 9}, "Vector-vector addition failed.");
  is_close_debug(v3 - OurVector3<T2>{4, 5, 6}, OurVector3<T1>{-3, -3, -3}, "Vector-vector subtraction failed.");
  is_close_debug(OurVector3<T2>{4, 5, 6} - v3, OurVector3<T1>{3, 3, 3}, "Vector-vector subtraction failed.");
}
//@}

//! \name Vector Multiplication and division (in dims 1, 2, and 3)
//@{

TYPED_TEST(VectorPairwiseTypeTest, MultiplicationAndDivisionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Dim 1
  OurVector1<T1> v1(1);
  auto v2 = v1 * T2(2);
  using T3 = decltype(v2)::scalar_t;
  is_close_debug(v2, OurVector1<T3>{2}, "Vector-scalar multiplication failed.");
  is_close_debug(T2(2) * v1, OurVector1<T3>{2}, "Scalar-vector multiplication failed.");
  is_close_debug(v2 / T2(2), OurVector1<T3>{1}, "Vector-scalar division failed.");

  // Dim 2
  OurVector2<T1> v3(1, 2);
  auto v4 = v3 * T2(2);
  using T4 = decltype(v4)::scalar_t;
  is_close_debug(v4, OurVector2<T4>{2, 4}, "Vector-scalar multiplication failed.");
  is_close_debug(T2(2) * v3, OurVector2<T4>{2, 4}, "Scalar-vector multiplication failed.");
  is_close_debug(v4 / T2(2), OurVector2<T4>{1, 2}, "Vector-scalar division failed.");

  // Dim 3
  OurVector3<T1> v5(1, 2, 3);
  auto v6 = v5 * T2(2);
  using T5 = decltype(v6)::scalar_t;
  is_close_debug(v6, OurVector3<T5>{2, 4, 6}, "Vector-scalar multiplication failed.");
  is_close_debug(T2(2) * v5, OurVector3<T5>{2, 4, 6}, "Scalar-vector multiplication failed.");
  is_close_debug(v6 / T2(2), OurVector3<T5>{1, 2, 3}, "Vector-scalar division failed.");
}
//@}

//! \name Vector Basic arithmetic reduction operations
//@{

TYPED_TEST(VectorSingleTypeTest, BasicArithmeticReductionOperations) {
  // Dim 1
  OurVector1<TypeParam> v1(1);
  is_close_debug(sum(v1), 1, "Sum failed.");
  is_close_debug(product(v1), 1, "Product failed.");
  is_close_debug(min(v1), 1, "Min failed.");
  is_close_debug(max(v1), 1, "Max failed.");
  is_close_debug(mean(v1), 1, "Mean failed.");
  is_close_debug(variance(v1), 0, "Variance failed.");
  is_close_debug(stddev(v1), 0, "Stddev failed.");

  // Dim 2
  OurVector2<TypeParam> v2(1, 2);
  is_close_debug(sum(v2), 3, "Sum failed.");
  is_close_debug(product(v2), 2, "Product failed.");
  is_close_debug(min(v2), 1, "Min failed.");
  is_close_debug(max(v2), 2, "Max failed.");
  is_close_debug(mean(v2), 1.5, "Mean failed.");
  is_close_debug(variance(v2), 0.25, "Variance failed.");
  is_close_debug(stddev(v2), 0.5, "Stddev failed.");

  // Dim 3
  OurVector3<TypeParam> v3(1, 2, 3);
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

  // Dim 1
  OurVector1<T1> v1(1);
  OurVector1<T2> v2(2);
  is_close_debug(dot(v1, v2), 2.0, "Dot product failed.");
  is_close_debug(norm(v1), 1.0, "Norm failed.");
  is_close_debug(norm_squared(v1), 1.0, "Norm squared failed.");
  is_close_debug(infinity_norm(v1), 1.0, "Infinity norm failed.");
  is_close_debug(one_norm(v1), 1.0, "One norm failed.");
  is_close_debug(two_norm(v1), 1.0, "Two norm failed.");
  is_close_debug(two_norm_squared(v1), 1.0, "Two norm squared failed.");
  is_close_debug(minor_angle(v1, v2), 0.0, "Minor angle failed.");
  is_close_debug(major_angle(v1, v2), M_PI, "Major angle failed.");

  // Dim 2
  OurVector2<T1> v3(1, 2);
  OurVector2<T2> v4(3, 4);
  is_close_debug(dot(v3, v4), 11.0, "Dot product failed.");
  is_close_debug(norm(v3), std::sqrt(5.0), "Norm failed.");
  is_close_debug(norm_squared(v3), 5.0, "Norm squared failed.");
  is_close_debug(infinity_norm(v3), 2.0, "Infinity norm failed.");
  is_close_debug(one_norm(v3), 3.0, "One norm failed.");
  is_close_debug(two_norm(v3), std::sqrt(5.0), "Two norm failed.");
  is_close_debug(two_norm_squared(v3), 5.0, "Two norm squared failed.");
  is_close_debug(minor_angle(v3, v4), std::acos(11.0 / (std::sqrt(5.0) * std::sqrt(25.0))), "Minor angle failed.");
  is_close_debug(major_angle(v3, v4), M_PI - std::acos(11.0 / (std::sqrt(5.0) * std::sqrt(25.0))),
                 "Major angle failed.");

  // Dim 3
  OurVector3<T1> v5(1, 2, 3);
  OurVector3<T2> v6(4, 5, 6);
  is_close_debug(dot(v5, v6), 32.0, "Dot product failed.");
  is_close_debug(norm(v5), std::sqrt(14.0), "Norm failed.");
  is_close_debug(norm_squared(v5), 14.0, "Norm squared failed.");
  is_close_debug(infinity_norm(v5), 3.0, "Infinity norm failed.");
  is_close_debug(one_norm(v5), 6.0, "One norm failed.");
  is_close_debug(two_norm(v5), std::sqrt(14.0), "Two norm failed.");
  is_close_debug(two_norm_squared(v5), 14.0, "Two norm squared failed.");
  is_close_debug(minor_angle(v5, v6), std::acos(32.0 / (std::sqrt(14.0) * std::sqrt(77.0))), "Minor angle failed.");
  is_close_debug(major_angle(v5, v6), M_PI - std::acos(32.0 / (std::sqrt(14.0) * std::sqrt(77.0))),
                 "Major angle failed.");
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
