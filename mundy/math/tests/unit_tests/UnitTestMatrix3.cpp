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
#include <mundy_math/Matrix3.hpp>    // for mundy::math::Matrix3
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_relaxed_zero_tolerance
#include <mundy_math/Vector3.hpp>    // for mundy::math::Vector3

// Note, these tests are meant to look like real use cases for the Matrix3 class. As a result, we use implicit type
// conversions rather than being explicit about types. This is to ensure that the Matrix3 class can be used in a
// natural way. This choice means that compiling this test with -Wdouble-promotion or -Wconversion will result in many
// warnings. We will not however, locally disable these warning.

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
void is_close_debug(const U& a, const T& b, const std::string& message_if_fail = "")
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
{
  if (!is_approx_close(a, b)) {
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    using CommonType = std::common_type_t<U, T>;
    std::cout << "diff = " << static_cast<CommonType>(a) - static_cast<CommonType>(b) << std::endl;
  }

  EXPECT_TRUE(is_approx_close(a, b)) << message_if_fail;
}

/// \brief Test that two Matrix3s are close
/// \param[in] m1 The first Matrix3
/// \param[in] m2 The second Matrix3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
void is_close_debug(const AMatrix3<U, Accessor1, Ownership1>& m1, const AMatrix3<T, Accessor2, Ownership2>& m2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(m1, m2)) {
    std::cout << "m1 = " << m1 << std::endl;
    std::cout << "m2 = " << m2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(m1, m2)) << message_if_fail;
}

/// \brief Test that two Vector3s are close
/// \param[in] v1 The first Vector3
/// \param[in] v2 The second Vector3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
void is_close_debug(const AVector3<U, Accessor1, Ownership1>& v1, const AVector3<T, Accessor2, Ownership2>& v2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(v1, v2)) << message_if_fail;
}

/// \brief Test that two Matrix3s are different
/// \param[in] m1 The first Matrix3
/// \param[in] m2 The second Matrix3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
void is_different_debug(const AMatrix3<U, Accessor1, Ownership1>& m1, const AMatrix3<T, Accessor2, Ownership2>& m2,
                        const std::string& message_if_fail = "") {
  if (is_approx_close(m1, m2)) {
    std::cout << "m1 = " << m1 << std::endl;
    std::cout << "m2 = " << m2 << std::endl;
  }
  EXPECT_TRUE(!is_approx_close(m1, m2)) << message_if_fail;
}

/// \brief Test that two Vector3s are different
/// \param[in] v1 The first Vector3
/// \param[in] v2 The second Vector3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
void is_different_debug(const AVector3<U, Accessor1, Ownership1>& v1, const AVector3<T, Accessor2, Ownership2>& v2,
                        const std::string& message_if_fail = "") {
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
class Matrix3SingleTypeTest : public ::testing::Test {
  using T = U;
};  // Matrix3SingleTypeTest

/// \brief List of types to run the tests on
using MyTypes = ::testing::Types<int, float, double>;

/// \brief Tell GTEST to run the tests on the types in MyTypes
TYPED_TEST_SUITE(Matrix3SingleTypeTest, MyTypes);

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
class Matrix3PairwiseTypeTest : public ::testing::Test {};  // Matrix3PairwiseTypeTest

/// \brief List of pairs of types to run the tests on
using MyTypePairs = ::testing::Types<TypePair<int, float>, TypePair<int, double>, TypePair<float, double>,
                                     TypePair<int, int>, TypePair<float, float>, TypePair<double, double>>;

/// \brief Tell GTEST to run the tests on the types in MyTypePairs
TYPED_TEST_SUITE(Matrix3PairwiseTypeTest, MyTypePairs);
//@}

//! \name Matrix3 Constructors and Destructor
//@{

TYPED_TEST(Matrix3SingleTypeTest, DefaultConstructor) {
  ASSERT_NO_THROW(Matrix3<TypeParam>());
}

TYPED_TEST(Matrix3SingleTypeTest, ConstructorFromNineScalars) {
  ASSERT_NO_THROW(Matrix3<TypeParam>(1, 2, 3, 4, 5, 6, -7, -8, -9));
  Matrix3<TypeParam> m(1, 2, 3, 4, 5, 6, -7, -8, -9);
  is_close_debug(m(0), 1);
  is_close_debug(m(1), 2);
  is_close_debug(m(2), 3);
  is_close_debug(m(3), 4);
  is_close_debug(m(4), 5);
  is_close_debug(m(5), 6);
  is_close_debug(m(6), -7);
  is_close_debug(m(7), -8);
  is_close_debug(m(8), -9);
}

TYPED_TEST(Matrix3SingleTypeTest, ConstructorFromInitializerList) {
  Matrix3<TypeParam> R_x = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  is_close_debug(R_x(0), 1);
  is_close_debug(R_x(1), 0);
  is_close_debug(R_x(2), 0);
  is_close_debug(R_x(3), 0);
  is_close_debug(R_x(4), 0);
  is_close_debug(R_x(5), -1);
  is_close_debug(R_x(6), 0);
  is_close_debug(R_x(7), 1);
  is_close_debug(R_x(8), 0);
}

TYPED_TEST(Matrix3SingleTypeTest, Comparison) {
  Matrix3<TypeParam> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Matrix3<TypeParam> m2(4, 5, 6, -7, -8, -9, 10, 11, 12);
  EXPECT_TRUE(is_close(m1, m1));
  EXPECT_FALSE(is_close(m1, m2));

  is_close_debug(m1, m1);
  is_close_debug(m1, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9});
}

TYPED_TEST(Matrix3SingleTypeTest, CopyConstructor) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  Matrix3<TypeParam> m2(m1);
  is_close_debug(m1, m2, "Copy constructor failed.");

  // The copy owns its own data since m1 is not a view
  m1 = {4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_different_debug(m1, m2, "Copy constructor failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, MoveConstructor) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  Matrix3<TypeParam> m2(std::move(m1));
  is_close_debug(m2, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Move constructor failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, CopyAssignment) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  Matrix3<TypeParam> m2{4, 5, 6, -7, -8, -9, 10, 11, 12};
  ASSERT_NO_THROW(m2 = m1);
  is_close_debug(m1, m2, "Copy assignment failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, MoveAssignment) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  Matrix3<TypeParam> m2{4, 5, 6, -7, -8, -9, 10, 11, 12};
  ASSERT_NO_THROW(m2 = std::move(m1));
  is_close_debug(m2, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Move assignment failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, Destructor) {
  ASSERT_NO_THROW(Matrix3<TypeParam>());
}
//@}

//! \name Matrix3 Accessors
//@{

TYPED_TEST(Matrix3SingleTypeTest, Accessors) {
  Matrix3<TypeParam> m(1, 2, 3, 4, 5, 6, -7, -8, -9);

  // By row-major flattened index
  is_close_debug(m(0), 1);
  is_close_debug(m(1), 2);
  is_close_debug(m(2), 3);
  is_close_debug(m(3), 4);
  is_close_debug(m(4), 5);
  is_close_debug(m(5), 6);
  is_close_debug(m(6), -7);
  is_close_debug(m(7), -8);
  is_close_debug(m(8), -9);

  // By row and column indices, version 1
  is_close_debug(m(0, 0), 1);
  is_close_debug(m(0, 1), 2);
  is_close_debug(m(0, 2), 3);
  is_close_debug(m(1, 0), 4);
  is_close_debug(m(1, 1), 5);
  is_close_debug(m(1, 2), 6);
  is_close_debug(m(2, 0), -7);
  is_close_debug(m(2, 1), -8);
  is_close_debug(m(2, 2), -9);

  // Fetch a row (deep copy)
  Vector3<TypeParam> row0 = m.copy_row(0);
  is_close_debug(row0[0], 1);
  is_close_debug(row0[1], 2);
  is_close_debug(row0[2], 3);
  row0 = {4, 5, 6};
  is_close_debug(m(0, 0), 1);
  is_close_debug(m(0, 1), 2);
  is_close_debug(m(0, 2), 3);

  // Fetch a column (deep copy)
  Vector3<TypeParam> col0 = m.copy_column(0);
  is_close_debug(col0[0], 1);
  is_close_debug(col0[1], 4);
  is_close_debug(col0[2], -7);
  col0 = {4, 5, 6};
  is_close_debug(m(0, 0), 1);
  is_close_debug(m(1, 0), 4);
  is_close_debug(m(2, 0), -7);
}

TYPED_TEST(Matrix3SingleTypeTest, RowViewAccessors) {
  Matrix3<TypeParam> m(1, 2, 3, 4, 5, 6, -7, -8, -9);

  // View each row, modify the view and check that the matrix is modified
  auto row0 = m.template view_row<0>();
  EXPECT_EQ(row0[0], 1);
  EXPECT_EQ(row0[1], 2);
  EXPECT_EQ(row0[2], 3);
  is_close_debug(row0, Vector3<TypeParam>{1, 2, 3});

  row0(0) = 4;
  row0(1) = 5;
  row0(2) = 6;
  is_close_debug(m(0, 0), 4, "Row0 view not a view.");
  is_close_debug(m(0, 1), 5, "Row0 view not a view.");
  is_close_debug(m(0, 2), 6, "Row0 view not a view.");

  auto row1 = m.template view_row<1>();
  is_close_debug(row1, Vector3<TypeParam>{4, 5, 6});

  row1(0) = 7;
  row1(1) = 8;
  row1(2) = 9;
  is_close_debug(m(1, 0), 7, "Row1 view not a view.");
  is_close_debug(m(1, 1), 8, "Row1 view not a view.");
  is_close_debug(m(1, 2), 9, "Row1 view not a view.");

  auto row2 = m.template view_row<2>();
  is_close_debug(row2, Vector3<TypeParam>{-7, -8, -9});

  row2(0) = -1;
  row2(1) = -2;
  row2(2) = -3;
  is_close_debug(m(2, 0), -1, "Row2 view not a view.");
  is_close_debug(m(2, 1), -2, "Row2 view not a view.");
  is_close_debug(m(2, 2), -3, "Row2 view not a view.");
}

TYPED_TEST(Matrix3SingleTypeTest, ColumnViewAccessors) {
  Matrix3<TypeParam> m(1, 2, 3, 4, 5, 6, -7, -8, -9);

  // View each column, modify the view and check that the matrix is modified
  auto col0 = m.template view_column<0>();
  ASSERT_EQ(col0(0), 1);
  ASSERT_EQ(col0(1), 4);
  ASSERT_EQ(col0(2), -7);

  col0(0) = 4;
  col0(1) = 5;
  col0(2) = 6;
  ASSERT_EQ(m(0, 0), 4);
  ASSERT_EQ(m(1, 0), 5);
  ASSERT_EQ(m(2, 0), 6);

  auto col1 = m.template view_column<1>();
  ASSERT_EQ(col1(0), 2);
  ASSERT_EQ(col1(1), 5);
  ASSERT_EQ(col1(2), -8);

  col1(0) = 7;
  col1(1) = 8;
  col1(2) = 9;
  ASSERT_EQ(m(0, 1), 7);
  ASSERT_EQ(m(1, 1), 8);
  ASSERT_EQ(m(2, 1), 9);

  auto col2 = m.template view_column<2>();
  ASSERT_EQ(col2(0), 3);
  ASSERT_EQ(col2(1), 6);
  ASSERT_EQ(col2(2), -9);

  col2(0) = -1;
  col2(1) = -2;
  col2(2) = -3;
  ASSERT_EQ(m(0, 2), -1);
  ASSERT_EQ(m(1, 2), -2);
  ASSERT_EQ(m(2, 2), -3);
}

// Tranposed view

TYPED_TEST(Matrix3SingleTypeTest, TransposedView) {
  Matrix3<TypeParam> m(1, 2, 3, 4, 5, 6, -7, -8, -9);

  // Transposed view
  auto m_t_view = m.view_transpose();
  is_close_debug(m_t_view, Matrix3<TypeParam>{1, 4, -7, 2, 5, -8, 3, 6, -9});
  m_t_view = Matrix3<TypeParam>{12, 13, 14, 15, 16, 17, 18, 19, 20};
  is_close_debug(m, Matrix3<TypeParam>{12, 15, 18, 13, 16, 19, 14, 17, 20});
}
//@}

//! \name Matrix3 Setters
//@{

TYPED_TEST(Matrix3SingleTypeTest, Setters) {
  // Set entire matrix by row-major flattened index
  Matrix3<TypeParam> m;
  m.set(1, 2, 3, 4, 5, 6, -7, -8, -9);
  is_close_debug(m, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Set by row-major flattened index failed.");

  // Set entire matrix by another Matrix3
  m.set(Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9});
  is_close_debug(m, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Set by Matrix3 failed.");

  // Set a column by a vector
  m.set_column(0, Vector3<TypeParam>{-1, -2, -3});
  is_close_debug(m.copy_column(0), Vector3<TypeParam>{-1, -2, -3}, "Set column by vector failed.");

  m.set_column(1, Vector3<TypeParam>{2, 4, 6});
  is_close_debug(m.copy_column(1), Vector3<TypeParam>{2, 4, 6}, "Set column by vector failed.");

  m.set_column(2, Vector3<TypeParam>{-3, -6, -9});
  is_close_debug(m.copy_column(2), Vector3<TypeParam>{-3, -6, -9}, "Set column by vector failed.");

  // Set a column by three scalars
  m.set_column(0, -4, -5, -6);
  is_close_debug(m.copy_column(0), Vector3<TypeParam>{-4, -5, -6}, "Set column by scalars failed.");

  m.set_column(1, 1, 2, 3);
  is_close_debug(m.copy_column(1), Vector3<TypeParam>{1, 2, 3}, "Set column by scalars failed.");

  m.set_column(2, -1, -2, -3);
  is_close_debug(m.copy_column(2), Vector3<TypeParam>{-1, -2, -3}, "Set column by scalars failed.");

  // Set a row by a vector
  m.set_row(0, Vector3<TypeParam>{-4, -5, -6});
  is_close_debug(m.copy_row(0), Vector3<TypeParam>{-4, -5, -6}, "Set row by vector failed.");

  m.set_row(1, Vector3<TypeParam>{1, 2, 3});
  is_close_debug(m.copy_row(1), Vector3<TypeParam>{1, 2, 3}, "Set row by vector failed.");

  m.set_row(2, Vector3<TypeParam>{-1, -2, -3});
  is_close_debug(m.copy_row(2), Vector3<TypeParam>{-1, -2, -3}, "Set row by vector failed.");

  // Set a row by three scalars
  m.set_row(1, -1, -2, -3);
  is_close_debug(m.copy_row(1), Vector3<TypeParam>{-1, -2, -3}, "Set row by scalars failed.");

  // Fill entire matrix with a scalar
  m.fill(7);
  is_close_debug(m, Matrix3<TypeParam>{7, 7, 7, 7, 7, 7, 7, 7, 7}, "Fill failed.");
}
//@}

//! \name Matrix3 Special vectors
//@{

TYPED_TEST(Matrix3SingleTypeTest, SpecialVectors) {
  ASSERT_NO_THROW(Matrix3<TypeParam>::identity());
  ASSERT_NO_THROW(Matrix3<TypeParam>::ones());
  ASSERT_NO_THROW(Matrix3<TypeParam>::zeros());

  auto identity = Matrix3<TypeParam>::identity();
  auto ones = Matrix3<TypeParam>::ones();
  auto zeros = Matrix3<TypeParam>::zeros();

  is_close_debug(identity, Matrix3<TypeParam>{1, 0, 0, 0, 1, 0, 0, 0, 1}, "Identity failed.");
  is_close_debug(ones, Matrix3<TypeParam>{1, 1, 1, 1, 1, 1, 1, 1, 1}, "Ones failed.");
  is_close_debug(zeros, Matrix3<TypeParam>{0, 0, 0, 0, 0, 0, 0, 0, 0}, "Zeros failed.");
}
//@}

//! \name Matrix3 Addition and subtraction
//@{

TYPED_TEST(Matrix3PairwiseTypeTest, AdditionAndSubtractionWithMatrix3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Matrix3<T2> m2(4, 5, 6, -7, -8, -9, 10, 11, 12);
  auto m3 = m1 + m2;
  using T3 = decltype(m3)::scalar_t;
  is_close_debug(m3, Matrix3<T3>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition failed.");

  m1 += m2;
  is_close_debug(m1, Matrix3<T1>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition assignment failed.");

  m3 = m1 - m2;
  is_close_debug(m3, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction failed.");

  m1 -= m2;
  is_close_debug(m1, Matrix3<T1>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction assignment failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, AdditionAndSubtractionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = m1 + T2(1);
  using T3 = decltype(m2)::scalar_t;
  is_close_debug(m2, Matrix3<T3>{2, 3, 4, 5, 6, 7, -6, -7, -8}, "Right addition failed.");

  m2 = T2(1) + m1;
  is_close_debug(m2, Matrix3<T3>{2, 3, 4, 5, 6, 7, -6, -7, -8}, "Left addition failed.");

  m2 = m1 - T2(1);
  is_close_debug(m2, Matrix3<T3>{0, 1, 2, 3, 4, 5, -8, -9, -10}, "Right subtraction failed.");

  m2 = T2(1) - m1;
  is_close_debug(m2, Matrix3<T3>{0, -1, -2, -3, -4, -5, 8, 9, 10}, "Left subtraction failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, AdditionAndSubtractionEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m3 = m1 + Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  using T3 = decltype(m3)::scalar_t;
  is_close_debug(m3, Matrix3<T3>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition failed.");

  m1 += Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_close_debug(m1, Matrix3<T1>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition assignment failed.");

  m3 = m1 - Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_close_debug(m3, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction failed.");

  m1 -= Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_close_debug(m1, Matrix3<T1>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction assignment failed.");
}
//@}

//! \name Matrix3 Multiplication and division
//@{

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionWithMatrix3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Matrix3<T2> m2(4, 5, 6, -7, -8, -9, 10, 11, 12);
  auto m3 = m1 * m2;
  using T3 = decltype(m3)::scalar_t;
  is_close_debug(m3, Matrix3<T3>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Multiplication failed.");

  m1 *= m2;
  is_close_debug(m1, Matrix3<T1>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Multiplication assignment failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionWithVector3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Vector3<T2> v1(4, 5, 6);
  auto v2 = m1 * v1;
  using T3 = decltype(v2)::scalar_t;
  is_close_debug(v2, Vector3<T3>{32, 77, -122}, "Multiplication failed.");

  // Left multiplication by a vector: v^T * m
  auto v3 = v1 * m1;
  using T4 = decltype(v3)::scalar_t;
  is_close_debug(v3, Vector3<T4>{-18, -15, -12}, "Left multiplication failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = m1 * T2(2);
  using T3 = decltype(m2)::scalar_t;
  is_close_debug(m2, Matrix3<T3>{2, 4, 6, 8, 10, 12, -14, -16, -18}, "Right multiplication failed.");

  m2 /= T2(2);
  is_close_debug(m2, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Division assignment failed.");

  m2 = T2(2) * m1;
  is_close_debug(m2, Matrix3<T3>{2, 4, 6, 8, 10, 12, -14, -16, -18}, "Left multiplication failed.");

  m2 = m2 / T2(2);
  is_close_debug(m2, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Right division failed.");

  m2 *= T2(2);
  is_close_debug(m2, Matrix3<T3>{2, 4, 6, 8, 10, 12, -14, -16, -18}, "Multiplication assignment failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionEdgeCases) {
  // Test that the multiplication and division operators work with rvalues
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = m1 * Matrix3<T2>(4, 5, 6, -7, -8, -9, 10, 11, 12);
  using T3 = decltype(m2)::scalar_t;
  is_close_debug(m2, Matrix3<T3>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Right rvalue multiplication failed.");

  m1 *= Matrix3<T2>(4, 5, 6, -7, -8, -9, 10, 11, 12);
  is_close_debug(m2, Matrix3<T3>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Left rvalue multiplication failed.");
}
//@}

//! \name Matrix3 Basic arithmetic reduction operations
//@{

TYPED_TEST(Matrix3SingleTypeTest, BasicArithmeticReductionOperations) {
  Matrix3<TypeParam> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  is_close_debug(determinant(m1), 0.0, "Determinant failed.");
  is_close_debug(trace(m1), -3.0, "Trace failed.");
  is_close_debug(sum(m1), -3.0, "Sum failed.");
  is_close_debug(product(m1), -362880.0, "Product failed.");
  is_close_debug(min(m1), -9.0, "Min failed.");
  is_close_debug(max(m1), 6.0, "Max failed.");
  is_close_debug(mean(m1), -1.0 / 3.0, "Mean failed.");
  is_close_debug(variance(m1), 284.0 / 9.0, "Variance failed.");
}
//@}

//! \name Matrix3 Special matrix operations
//@{

TYPED_TEST(Matrix3SingleTypeTest, SpecialOperations) {
  // Test the transpose
  Matrix3<TypeParam> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = transpose(m1);
  using T3 = decltype(m2)::scalar_t;
  is_close_debug(m2, Matrix3<T3>{1, 4, -7, 2, 5, -8, 3, 6, -9}, "Transpose failed.");

  // Test the inverse of a singular matrix
  // Won't throw in release
#ifndef NDEBUG
  EXPECT_ANY_THROW(inverse(m1));
#endif

  // Test the inverse of a non-singular matrix
  m1 = {1, 2, 3, 0, 1, 4, 5, 6, 0};
  auto m3 = inverse(m1);
  using T4 = decltype(m3)::scalar_t;
  is_close_debug(m3, Matrix3<T4>{-24, 18, 5, 20, -15, -4, -5, 4, 1}, "Inverse failed.");

  // Test the frobenius_inner_product
  m1 = {1, 2, 3, 2, 5, 6, 3, 6, 9};
  m2 = {4, 5, 6, 7, 8, 9, -10, -11, -12};
  is_close_debug(frobenius_inner_product(m1, m2), -64, "Frobenius inner product failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, SpecialOperationsEdgeCases) {
  // Test that the special vector operations work with rvalues
  Matrix3<TypeParam> m2 = transpose(Matrix3<TypeParam>(1, 2, 3, 4, 5, 6, -7, -8, -9));
  is_close_debug(m2, Matrix3<TypeParam>{1, 4, -7, 2, 5, -8, 3, 6, -9}, "Transpose failed.");

  // Test the inverse of a singular matrix
  // Won't throw in release
#ifndef NDEBUG
  EXPECT_ANY_THROW(inverse(Matrix3<TypeParam>(1, 2, 3, 2, 4, 6, 3, 6, 9)));
#endif

  // Test the inverse of a non-singular matrix. Notice, that even though the input matrix may have integer entries, the
  // inverse will have floating point entries.
  auto m3 = inverse(Matrix3<TypeParam>(1, 2, 3, 0, 1, 4, 5, 6, 0));
  is_close_debug(m3, Matrix3<TypeParam>{-24, 18, 5, 20, -15, -4, -5, 4, 1}, "Inverse failed.");

  // Test the frobenius_inner_product
  is_close_debug(frobenius_inner_product(Matrix3<TypeParam>(1, 2, 3, 2, 5, 6, 3, 6, 9),
                                         Matrix3<TypeParam>(4, 5, 6, 7, 8, 9, -10, -11, -12)),
                 -64, "Frobenius inner product failed.");
}
//@}

//! \name Apply
//@{

struct an_external_functor {
  template <typename T>
  KOKKOS_FUNCTION T operator()(const T& x) const {
    return x + 1;
  }
};

struct an_vector_external_functor {
  template <typename T>
  KOKKOS_FUNCTION auto operator()(const T& x) const {
    // Set each element of the row or column to the sum of its elements
    return sum(x) * Vector3<typename T::scalar_t>{1, 1, 1};
  }
};

TYPED_TEST(Matrix3SingleTypeTest, Apply) {
  // Apply to each element of the matrix
  Matrix3<TypeParam> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = apply(an_external_functor{}, m1);
  using T3 = decltype(m2)::scalar_t;
  is_close_debug(m2, Matrix3<T3>{2, 3, 4, 5, 6, 7, -6, -7, -8}, "Apply to elements failed.");

  // Apply to each row
  auto m3 = apply_row(an_vector_external_functor{}, m1);
  using T4 = decltype(m3)::scalar_t;
  is_close_debug(m3, Matrix3<T4>{6, 6, 6, 15, 15, 15, -24, -24, -24}, "Apply to rows failed.");

  // Apply to each column
  auto m4 = apply_column(an_vector_external_functor{}, m1);
  is_close_debug(m4, Matrix3<T4>{-2, -1, 0, -2, -1, 0, -2, -1, 0}, "Apply to columns failed.");
}
//@}

//! \name Matrix3 Views
//@{

TYPED_TEST(Matrix3SingleTypeTest, Views) {
  // Create a view from a subset of an std::vector<TypeParam>
  std::vector<TypeParam> m1{0, 0, 1, 2, 3, 4, 5, 6, -7, -8, -9, 0, 0};
  auto m2 = get_matrix3_view<TypeParam>(m1.data() + 2);
  is_close_debug(m2, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "View failed.");
  auto ptr_before = m1.data();
  m1 = {1, 2, 4, 5, 6, -7, -8, -9, 10, 11, 12, 13, 14};
  auto ptr_after = m1.data();
  ASSERT_EQ(ptr_before, ptr_after);
  is_close_debug(m2, Matrix3<TypeParam>{4, 5, 6, -7, -8, -9, 10, 11, 12}, "View isn't shallow.");

  // Create a view from a TypeParam*
  TypeParam m3[9] = {1, 2, 3, 4, 5, 6, -7, -8, -9};
  auto m4 = get_matrix3_view<TypeParam>(&m3[0]);
  is_close_debug(m4, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "View failed.");
  m3[0] = 4;
  m3[1] = 5;
  m3[2] = 6;
  m3[3] = -7;
  m3[4] = -8;
  m3[5] = -9;
  m3[6] = 10;
  m3[7] = 11;
  m3[8] = 12;
  is_close_debug(m4, Matrix3<TypeParam>{4, 5, 6, -7, -8, -9, 10, 11, 12}, "View isn't shallow.");

  // Create a const view from an std::vector<TypeParam>
  const std::vector<TypeParam> m5{1, 2, 3, 4, 5, 6, -7, -8, -9};
  auto m6 = get_matrix3_view<TypeParam>(m5.data());
  is_close_debug(m6, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Const view failed.");
}
//@}

}  // namespace

}  // namespace math

}  // namespace mundy
