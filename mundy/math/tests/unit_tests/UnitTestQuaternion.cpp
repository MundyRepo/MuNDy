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
#include <mundy_math/Matrix3.hpp>     // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy_math::Quaternion
#include <mundy_math/Tolerance.hpp>   // for mundy::math::get_relaxed_tolerance
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3

// Note, these tests are meant to look like real use cases for the Quaternion class. As a result, we use implicit type
// conversions rather than being explicit about types. This is to ensure that the Quaternion class can be used in a
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
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
void is_close_debug(const U& a, const T& b, const std::string& message_if_fail = "") {
  using CommonType = std::common_type_t<T, U>;
  if (!is_approx_close(a, b)) {
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "diff = " << static_cast<CommonType>(a) - static_cast<CommonType>(b) << std::endl;
  }

  EXPECT_TRUE(is_approx_close(a, b)) << message_if_fail;
}

/// \brief Test that two Matrix3s are close
/// \param[in] m1 The first Matrix3
/// \param[in] m2 The second Matrix3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
void is_close_debug(const AMatrix3<U, Accessor1, OwnershipType1>& m1, const AMatrix3<T, Accessor2, OwnershipType2>& m2,
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
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
void is_close_debug(const AVector3<U, Accessor1, OwnershipType1>& v1, const AVector3<T, Accessor2, OwnershipType2>& v2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(v1, v2)) << message_if_fail;
}

/// \brief Test that two Quaternions are close
/// \param[in] q1 The first Quaternion
/// \param[in] q2 The second Quaternion
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
void is_close_debug(const AQuaternion<U, Accessor1, OwnershipType1>& q1,
                    const AQuaternion<T, Accessor2, OwnershipType2>& q2, const std::string& message_if_fail = "") {
  if (!is_approx_close(q1, q2)) {
    std::cout << "q1 = " << q1 << std::endl;
    std::cout << "q2 = " << q2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(q1, q2)) << message_if_fail;
}

//// \brief Test that two Matrix3s are different
/// \param[in] m1 The first Matrix3
/// \param[in] m2 The second Matrix3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
void is_different_debug(const AMatrix3<U, Accessor1, OwnershipType1>& m1,
                        const AMatrix3<T, Accessor2, OwnershipType2>& m2, const std::string& message_if_fail = "") {
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
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
void is_different_debug(const AVector3<U, Accessor1, OwnershipType1>& v1,
                        const AVector3<T, Accessor2, OwnershipType2>& v2, const std::string& message_if_fail = "") {
  if (is_approx_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(!is_approx_close(v1, v2)) << message_if_fail;
}

/// \brief Test that two Quaternions are different
/// \param[in] q1 The first Quaternion
/// \param[in] q2 The second Quaternion
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
void is_different_debug(const AQuaternion<U, Accessor1, OwnershipType1>& q1,
                        const AQuaternion<T, Accessor2, OwnershipType2>& q2, const std::string& message_if_fail = "") {
  if (is_approx_close(q1, q2)) {
    std::cout << "q1 = " << q1 << std::endl;
    std::cout << "q2 = " << q2 << std::endl;
  }
  EXPECT_TRUE(!is_approx_close(q1, q2)) << message_if_fail;
}

/// \brief Get the quaternion corresponding to a 90 deg rotation about the x-axis
template <typename T>
Quaternion<T> get_quaternion_x_90() {
  return Quaternion<T>(static_cast<T>(1.0 / std::sqrt(2.0)), static_cast<T>(1.0 / std::sqrt(2.0)), static_cast<T>(0.0),
                       static_cast<T>(0.0));
}

/// \brief Get the quaternion corresponding to a 90 deg rotation about the y-axis
template <typename T>
Quaternion<T> get_quaternion_y_90() {
  return Quaternion<T>(static_cast<T>(1.0 / std::sqrt(2.0)), static_cast<T>(0.0), static_cast<T>(1.0 / std::sqrt(2.0)),
                       static_cast<T>(0.0));
}

/// \brief Get the quaternion corresponding to a 90 deg rotation about the z-axis
template <typename T>
Quaternion<T> get_quaternion_z_90() {
  return Quaternion<T>(static_cast<T>(1.0 / std::sqrt(2.0)), static_cast<T>(0.0), static_cast<T>(0.0),
                       static_cast<T>(1.0 / std::sqrt(2.0)));
}

//@}

//! \name GTEST typed test fixtures
//@{

/// \brief GTEST typed test fixture so we can run tests on multiple types
/// \tparam U The type to run the tests on
template <typename U>
class QuaternionSingleTypeTest : public ::testing::Test {
  using T = U;
};  // Vector3SingleTypeTest

/// \brief List of types to run the tests on
using MyTypes = ::testing::Types<float, double>;

/// \brief Tell GTEST to run the tests on the types in MyTypes
TYPED_TEST_SUITE(QuaternionSingleTypeTest, MyTypes);

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
class QuaternionPairwiseTypeTest : public ::testing::Test {};  // Vector3PairwiseTypeTest

/// \brief List of pairs of types to run the tests on
using MyTypePairs = ::testing::Types<TypePair<float, double>, TypePair<float, float>, TypePair<double, double>>;

/// \brief Tell GTEST to run the tests on the types in MyTypePairs
TYPED_TEST_SUITE(QuaternionPairwiseTypeTest, MyTypePairs);
//@}

//! \name Quaternion Constructors and Destructor
//@{

TYPED_TEST(QuaternionSingleTypeTest, DefaultConstructor) {
  ASSERT_NO_THROW(Quaternion<TypeParam>());
}

TYPED_TEST(QuaternionSingleTypeTest, ConstructorFromFourScalars) {
  ASSERT_NO_THROW(Quaternion<TypeParam>(1, 2, 3, 4));
  Quaternion<TypeParam> q(1, 2, 3, 4);
  is_close_debug(q[0], 1);
  is_close_debug(q[1], 2);
  is_close_debug(q[2], 3);
  is_close_debug(q[3], 4);
}

TYPED_TEST(QuaternionSingleTypeTest, Comparison) {
  Quaternion<TypeParam> q1(1, 2, 3, 4);
  Quaternion<TypeParam> q2(4, 10, 11, 12);
  EXPECT_TRUE(is_close(q1, q1));
  EXPECT_FALSE(is_close(q1, q2));

  is_close_debug(q1, q1);
  is_close_debug(q1, Quaternion<TypeParam>{1, 2, 3, 4});
}

TYPED_TEST(QuaternionSingleTypeTest, CopyConstructor) {
  Quaternion<TypeParam> q1{1, 2, 3, 4};
  Quaternion<TypeParam> q2(q1);
  is_close_debug(q1, q2, "Copy constructor failed.");

  // The copy owns its own data since 11 is not a view
  q1 = {4, 10, 11, 12};
  is_different_debug(q1, q2, "Copy constructor failed.");
}

TYPED_TEST(QuaternionSingleTypeTest, MoveConstructor) {
  Quaternion<TypeParam> q1{1, 2, 3, 4};
  Quaternion<TypeParam> q2(std::move(q1));
  is_close_debug(q2, Quaternion<TypeParam>{1, 2, 3, 4}, "Move constructor failed.");
}

TYPED_TEST(QuaternionSingleTypeTest, CopyAssignment) {
  Quaternion<TypeParam> q1{1, 2, 3, 4};
  Quaternion<TypeParam> q2{4, 10, 11, 12};
  ASSERT_NO_THROW(q2 = q1);
  is_close_debug(q1, q2, "Copy assignment failed.");
}

TYPED_TEST(QuaternionSingleTypeTest, MoveAssignment) {
  Quaternion<TypeParam> q1{1, 2, 3, 4};
  Quaternion<TypeParam> q2{4, 10, 11, 12};
  ASSERT_NO_THROW(q2 = std::move(q1));
  is_close_debug(q2, Quaternion<TypeParam>{1, 2, 3, 4}, "Move assignment failed.");
}

TYPED_TEST(QuaternionSingleTypeTest, Destructor) {
  ASSERT_NO_THROW(Quaternion<TypeParam>());
}
//@}

//! \name Quaternion Accessors
//@{

TYPED_TEST(QuaternionSingleTypeTest, Accessors) {
  Quaternion<TypeParam> q(1, 2, 3, 4);

  // By index
  is_close_debug(q[0], 1);
  is_close_debug(q[1], 2);
  is_close_debug(q[2], 3);
  is_close_debug(q[3], 4);

  // By w, x, y, z
  is_close_debug(q.w(), 1);
  is_close_debug(q.x(), 2);
  is_close_debug(q.y(), 3);
  is_close_debug(q.z(), 4);

  // Fetch the vector component of the quaternion
  Vector3<TypeParam> v = q.vector();
  is_close_debug(v, Vector3<TypeParam>{2, 3, 4}, "Get vector failed.");

  // Accessors return references
  q[0] = 4;
  is_close_debug(q[0], 4);
  q.w() = 5;
  is_close_debug(q.w(), 5);
}
//@}

//! \name Quaternion Setters
//@{

TYPED_TEST(QuaternionSingleTypeTest, Setters) {
  // Set entire quaternion by four scalars
  Quaternion<TypeParam> q;
  q.set(1, 2, 3, 4);
  is_close_debug(q, Quaternion<TypeParam>{1, 2, 3, 4}, "Set by scalar failed.");

  // Set entire quaternion by a scalar and a vector
  q.set(1, Vector3<TypeParam>{2, 3, 4});
  is_close_debug(q, Quaternion<TypeParam>{1, 2, 3, 4}, "Set by scalar and vector failed.");

  // Set entire quaternion by another quaternion
  q.set(Quaternion<TypeParam>{1, 2, 3, 4});
  is_close_debug(q, Quaternion<TypeParam>{1, 2, 3, 4}, "Set by quaternion failed.");

  // Set the vector component of the quaternion
  q.set_vector(Vector3<TypeParam>{2, 3, 4});
  is_close_debug(q, Quaternion<TypeParam>{1, 2, 3, 4}, "Set vector failed.");
}
//@}

//! \name Quaternion Special vectors
//@{

TYPED_TEST(QuaternionSingleTypeTest, SpecialVectors) {
  ASSERT_NO_THROW(Quaternion<TypeParam>::identity());

  auto identity = Quaternion<TypeParam>::identity();
  is_close_debug(identity, Quaternion<TypeParam>{1, 0, 0, 0}, "Identity failed.");
}
//@}

//! \name Quaternion Addition and subtraction
//@{

TYPED_TEST(QuaternionPairwiseTypeTest, AdditionAndSubtractionWithQuaternion) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Quaternion<T1> q1(1, 2, 3, 4);
  Quaternion<T2> q2(4, 10, 11, 12);
  auto q3 = q1 + q2;
  using T3 = decltype(q3)::scalar_t;
  is_close_debug(q3, Quaternion<T3>{5, 12, 14, 16}, "Addition failed.");

  q1 += q2;
  is_close_debug(q1, Quaternion<T1>{5, 12, 14, 16}, "Addition assignment failed.");

  q3 = q1 - q2;
  is_close_debug(q3, Quaternion<T3>{1, 2, 3, 4}, "Subtraction failed.");

  q1 -= q2;
  is_close_debug(q1, Quaternion<T1>{1, 2, 3, 4}, "Subtraction assignment failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, AdditionAndSubtractionEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Test that the addition and subtraction operators work with rvalues
  Quaternion<T1> q1(1, 2, 3, 4);
  auto q3 = q1 + Quaternion<T2>(4, 10, 11, 12);
  using T3 = decltype(q3)::scalar_t;
  is_close_debug(q3, Quaternion<T3>{5, 12, 14, 16}, "Right rvalue addition failed.");

  q1 += Quaternion<T2>(4, 10, 11, 12);
  is_close_debug(q1, Quaternion<T3>{5, 12, 14, 16}, "Right rvalue addition assignment failed.");

  q3 = q1 - Quaternion<T2>(4, 10, 11, 12);
  is_close_debug(q3, Quaternion<T3>{1, 2, 3, 4}, "Right rvalue subtraction failed.");

  q1 -= Quaternion<T2>(4, 10, 11, 12);
  is_close_debug(q1, Quaternion<T3>{1, 2, 3, 4}, "Right rvalue subtraction assignment failed.");
}
//@}

//! \name Quaternion Multiplication and division
//@{

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithQuaternion) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using T3 = decltype(Quaternion<T1>() * Quaternion<T2>())::scalar_t;

  // 90 degrees rotation around Z-axis
  Quaternion<T1> q1_z = get_quaternion_z_90<T1>();
  Quaternion<T2> q2_z = get_quaternion_y_90<T2>();
  Quaternion<T3> expected_quat_z = {0.5, -0.5, 0.5, 0.5};
  is_close_debug(q1_z * q2_z, expected_quat_z, "90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis
  Quaternion<T1> q1_y = get_quaternion_y_90<T1>();
  Quaternion<T2> q2_y = get_quaternion_x_90<T2>();
  Quaternion<T3> expected_quat_y = {0.5, 0.5, 0.5, -0.5};
  is_close_debug(q1_y * q2_y, expected_quat_y, "90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis
  Quaternion<T1> q1_x = get_quaternion_x_90<T1>();
  Quaternion<T2> q2_x = get_quaternion_z_90<T2>();
  Quaternion<T3> expected_quat_x = {0.5, 0.5, -0.5, 0.5};
  is_close_debug(q1_x * q2_x, expected_quat_x, "90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithMatrix3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Choose a random matrix to rotate
  Matrix3<T2> m(1, 2, 3, 4, 5, 6, -7, -8, -9);

  // Left multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: R_z m
  Quaternion<T1> q1_z = get_quaternion_z_90<T1>();
  Matrix3<T1> R_z = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  is_close_debug(R_z, quaternion_to_rotation_matrix(q1_z), "Rotation matrix-quaternion mismatch.");
  is_close_debug(R_z * m, Matrix3<T1>{-4, -5, -6, 1, 2, 3, -7, -8, -9},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug(q1_z * m, R_z * m, "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y m
  Quaternion<T1> q1_y = get_quaternion_y_90<T1>();
  Matrix3<T1> R_y = {0, 0, 1, 0, 1, 0, -1, 0, 0};
  is_close_debug(R_y, quaternion_to_rotation_matrix(q1_y), "Rotation matrix-quaternion mismatch.");
  is_close_debug(R_y * m, Matrix3<T1>{-7, -8, -9, 4, 5, 6, -1, -2, -3},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug(q1_y * m, R_y * m, "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x m
  Quaternion<T1> q1_x = get_quaternion_x_90<T1>();
  Matrix3<T1> R_x = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  is_close_debug(R_x, quaternion_to_rotation_matrix(q1_x), "Rotation matrix-quaternion mismatch.");
  is_close_debug(R_x * m, Matrix3<T1>{1, 2, 3, 7, 8, 9, 4, 5, 6}, "Matrix-matrix multiplication sanity check failed.");
  is_close_debug(q1_x * m, R_x * m, "Left 90 degrees rotation around X-axis failed.");

  // Right multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: m R_z
  is_close_debug(m * R_z, Matrix3<T1>{2, -1, 3, 5, -4, 6, -8, 7, -9},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug(m * q1_z, m * R_z, "Right 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: m R_y
  is_close_debug(m * R_y, Matrix3<T1>{-3, 2, 1, -6, 5, 4, 9, -8, -7},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug(m * q1_y, m * R_y, "Right 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: m R_x
  is_close_debug(m * R_x, Matrix3<T1>{1, 3, -2, 4, 6, -5, -7, -9, 8},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug(m * q1_x, m * R_x, "Right 90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithMatrix3sEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Test that the multiplication and division operators work with rvalues

  // Left multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: R_z m
  Quaternion<T1> q1_z = get_quaternion_z_90<T1>();
  Matrix3<T1> R_z = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  is_close_debug(R_z, quaternion_to_rotation_matrix(q1_z), "Rotation matrix-quaternion mismatch.");
  is_close_debug(q1_z * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}, R_z * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9},
                 "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y m
  Quaternion<T1> q1_y = get_quaternion_y_90<T1>();
  Matrix3<T1> R_y = {0, 0, 1, 0, 1, 0, -1, 0, 0};
  is_close_debug(R_y, quaternion_to_rotation_matrix(q1_y), "Rotation matrix-quaternion mismatch.");
  is_close_debug(q1_y * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}, R_y * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9},
                 "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x m
  Quaternion<T1> q1_x = get_quaternion_x_90<T1>();
  Matrix3<T1> R_x = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  is_close_debug(R_x, quaternion_to_rotation_matrix(q1_x), "Rotation matrix-quaternion mismatch.");
  is_close_debug(q1_x * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}, R_x * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9},
                 "Left 90 degrees rotation around X-axis failed.");

  // Right multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: m R_z
  is_close_debug(Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * q1_z, Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * R_z,
                 "Right 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: m R_y
  is_close_debug(Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * q1_y, Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * R_y,
                 "Right 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: m R_x
  is_close_debug(Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * q1_x, Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * R_x,
                 "Right 90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithVector3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using T3 = decltype(Quaternion<T1>() * Vector3<T2>())::scalar_t;

  // Choose a random vector to rotate
  Vector3<T2> v(1, 2, 3);

  // Left multiplication of a vector by a quaternion
  // 90 degrees rotation around Z-axis: R_z v
  Quaternion<T1> q_z = get_quaternion_z_90<T1>();
  Vector3<T3> expected_v_z = {-2, 1, 3};
  is_close_debug(q_z * v, expected_v_z, "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y v
  Quaternion<T1> q_y = get_quaternion_y_90<T1>();
  Vector3<T3> expected_v_y = {3, 2, -1};
  is_close_debug(q_y * v, expected_v_y, "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x v
  Quaternion<T1> q_x = get_quaternion_x_90<T1>();
  Vector3<T3> expected_v_x = {1, -3, 2};
  is_close_debug(q_x * v, expected_v_x, "Left 90 degrees rotation around X-axis failed.");

  // Right multiplication of a vector by a quaternion
  // 90 degrees rotation around Z-axis: v^T R_z = (R_z^T v)^T
  expected_v_z = {2, -1, 3};
  is_close_debug(v * q_z, expected_v_z, "Right 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: v^T R_y = (R_y^T v)^T
  expected_v_y = {-3, 2, 1};
  is_close_debug(v * q_y, expected_v_y, "Right 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: v^T R_x = (R_x^T v)^T
  expected_v_x = {1, 3, -2};
  is_close_debug(v * q_x, expected_v_x, "Right 90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithVector3sEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using T3 = decltype(Quaternion<T1>() * Vector3<T2>())::scalar_t;

  // Test that the multiplication and division operators work with rvalues

  // Left multiplication of a vector by a quaternion
  // 90 degrees rotation around Z-axis: R_z v
  Quaternion<T1> q_z = get_quaternion_z_90<T1>();
  Vector3<T3> expected_v_z = {-2, 1, 3};
  is_close_debug(q_z * Vector3<T2>{1, 2, 3}, expected_v_z, "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y v
  Quaternion<T1> q_y = get_quaternion_y_90<T1>();
  Vector3<T3> expected_v_y = {3, 2, -1};
  is_close_debug(q_y * Vector3<T2>{1, 2, 3}, expected_v_y, "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x v
  Quaternion<T1> q_x = get_quaternion_x_90<T1>();
  Vector3<T3> expected_v_x = {1, -3, 2};
  is_close_debug(q_x * Vector3<T2>{1, 2, 3}, expected_v_x, "Left 90 degrees rotation around X-axis failed.");

  // Right multiplication of a vector by a quaternion
  // 90 degrees rotation around Z-axis: v^T R_z = (R_z^T v)^T
  expected_v_z = {2, -1, 3};
  is_close_debug(Vector3<T2>{1, 2, 3} * q_z, expected_v_z, "Right 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: v^T R_y = (R_y^T v)^T
  expected_v_y = {-3, 2, 1};
  is_close_debug(Vector3<T2>{1, 2, 3} * q_y, expected_v_y, "Right 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: v^T R_x = (R_x^T v)^T
  expected_v_x = {1, 3, -2};
  is_close_debug(Vector3<T2>{1, 2, 3} * q_x, expected_v_x, "Right 90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Quaternion<T1> q1(1, 2, 3, 4);
  auto q2 = q1 * T2(2);
  using T3 = decltype(q2)::scalar_t;
  is_close_debug(q2, Quaternion<T3>{2, 4, 6, 8}, "Right multiplication failed.");

  q2 = T2(2) * q1;
  is_close_debug(q2, Quaternion<T3>{2, 4, 6, 8}, "Left multiplication failed.");

  q2 = q1 / T2(2);
  is_close_debug(q2, Quaternion<T3>{0.5, 1, 1.5, 2}, "Right division failed.");

  q1 /= T2(2);
  is_close_debug(q1, Quaternion<T3>{0.5, 1, 1.5, 2}, "Division assignment failed.");

  q1 *= T2(2);
  is_close_debug(q1, Quaternion<T3>{1, 2, 3, 4}, "Multiplication assignment failed.");
}
//@}

//! \name Quaternion Special quaternion operations
//@{

TYPED_TEST(QuaternionPairwiseTypeTest, SpecialOperations) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // dot
  Quaternion<T1> q1(1, 2, 3, 4);
  Quaternion<T2> q2(4, 10, 11, 12);
  is_close_debug(dot(q1, q2), 105, "Dot failed.");

  // conjugate
  auto q3 = conjugate(q1);
  using T3 = decltype(q3)::scalar_t;
  is_close_debug(q3, Quaternion<T3>{1, -2, -3, -4}, "Conjugate failed.");

  // conjugate in place
  q1.conjugate();
  is_close_debug(q1, Quaternion<T1>{1, -2, -3, -4}, "Conjugate assignment failed.");
  q1 = {1, 2, 3, 4};

  // norm
  is_close_debug(norm(q1), std::sqrt(30.0), "Norm failed.");

  // norm_squared
  is_close_debug(norm_squared(q1), 30.0, "Norm squared failed.");

  // normalize
  auto q4 = normalize(q1);
  using T4 = decltype(q4)::scalar_t;
  is_close_debug(q4,
                 Quaternion<T4>{static_cast<T4>(1.0 / std::sqrt(30.0)), static_cast<T4>(2.0 / std::sqrt(30.0)),
                                static_cast<T4>(3.0 / std::sqrt(30.0)), static_cast<T4>(4.0 / std::sqrt(30.0))},
                 "Normalize assignment failed.");

  // inverse
  auto q5 = inverse(q1);
  using T5 = decltype(q5)::scalar_t;
  is_close_debug(q5,
                 Quaternion<T5>{static_cast<T5>(1.0 / 30.0), static_cast<T5>(-2.0 / 30.0), static_cast<T5>(-3.0 / 30.0),
                                static_cast<T5>(-4.0 / 30.0)},
                 "Inverse failed.");

  // normalize in place
  q1.normalize();
  q2.normalize();
  is_close_debug(q1,
                 Quaternion<T4>{static_cast<T4>(1.0 / std::sqrt(30.0)), static_cast<T4>(2.0 / std::sqrt(30.0)),
                                static_cast<T4>(3.0 / std::sqrt(30.0)), static_cast<T4>(4.0 / std::sqrt(30.0))},
                 "Normalize failed.");

  // slerp (only applicable to unit quaternions)
  auto q6 = slerp(q1, q2, 0.5);
  using T6 = decltype(q5)::scalar_t;
  is_close_debug(q6,
                 Quaternion<T6>{static_cast<T6>(0.1946219299433149), static_cast<T6>(0.4407059160784743),
                                static_cast<T6>(0.5581347617390449), static_cast<T6>(0.6755636074046377)},
                 "Slerp failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, SpecialOperationsEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  // Test that the special vector operations work with rvalues

  // dot
  is_close_debug(dot(Quaternion<T1>(1, 2, 3, 4), Quaternion<T2>(4, 10, 11, 12)), 105, "Dot failed.");

  // conjugate
  is_close_debug(conjugate(Quaternion<T1>(1, 2, 3, 4)), Quaternion<T2>{1, -2, -3, -4}, "Conjugate failed.");

  // norm
  is_close_debug(norm(Quaternion<T1>(1, 2, 3, 4)), std::sqrt(30.0), "Norm failed.");

  // norm_squared
  is_close_debug(norm_squared(Quaternion<T1>(1, 2, 3, 4)), 30.0, "Norm squared failed.");

  // normalize
  auto q4 = normalize(Quaternion<T1>(1, 2, 3, 4));
  using T4 = decltype(q4)::scalar_t;
  is_close_debug(q4,
                 Quaternion<T4>{static_cast<T4>(1.0 / std::sqrt(30.0)), static_cast<T4>(2.0 / std::sqrt(30.0)),
                                static_cast<T4>(3.0 / std::sqrt(30.0)), static_cast<T4>(4.0 / std::sqrt(30.0))},
                 "Normalize failed.");

  // slerp
  auto q5 = slerp(normalize(Quaternion<T1>(1, 2, 3, 4)), normalize(Quaternion<T2>(4, 10, 11, 12)), 0.5);
  using T5 = decltype(q5)::scalar_t;
  is_close_debug(q5,
                 Quaternion<T5>{static_cast<T5>(0.1946219299433149), static_cast<T5>(0.4407059160784743),
                                static_cast<T5>(0.5581347617390449), static_cast<T5>(0.6755636074046377)},
                 "Slerp failed.");
}
//@}

//! \name Quaternion Views
//@{

TYPED_TEST(QuaternionSingleTypeTest, Views) {
  // Create a view from a subset of an std::vector<TypeParam>
  std::vector<TypeParam> q1{0, 0, 1, 2, 3, 4, 0, 0};
  auto q2 = get_quaternion_view<TypeParam>(q1.data() + 2);
  is_close_debug(q2, Quaternion<TypeParam>{1, 2, 3, 4}, "View failed.");
  q1 = {1, 2, 4, 10, 11, 12, 13, 14};
  is_close_debug(q2, Quaternion<TypeParam>{4, 10, 11, 12}, "View isn't shallow.");

  // Create a view from a TypeParam*
  TypeParam q3[4] = {1, 2, 3, 4};
  auto q4 = get_quaternion_view<TypeParam>(&q3[0]);
  is_close_debug(q4, Quaternion<TypeParam>{1, 2, 3, 4}, "View failed.");
  q3[0] = 4;
  q3[1] = 10;
  q3[2] = 11;
  q3[3] = 12;
  is_close_debug(q4, Quaternion<TypeParam>{4, 10, 11, 12}, "View isn't shallow.");

  // Create a const view from an std::vector<TypeParam>
  const std::vector<TypeParam> q5{1, 2, 3, 4};
  auto q6 = get_quaternion_view<TypeParam>(q5.data());
  is_close_debug(q6, Quaternion<TypeParam>{1, 2, 3, 4}, "Const view failed.");
}
//@}

}  // namespace

}  // namespace math

}  // namespace mundy
