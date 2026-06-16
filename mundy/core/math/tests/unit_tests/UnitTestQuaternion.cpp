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
#include <mundy_math/Matrix3.hpp>     // for mundy::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy_math::Quaternion
#include <mundy_math/Tolerance.hpp>   // for mundy::get_relaxed_tolerance
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3
#include <mundy_utils/requires.hpp>

// Note, these tests are meant to look like real use cases for the Quaternion class. As a result, we use implicit type
// conversions rather than being explicit about types. This is to ensure that the Quaternion class can be used in a
// natural way. This choice means that compiling this test with -Wdouble-promotion or -Wconversion will result in many
// warnings. We will not however, locally disable these warning.

namespace mundy {

namespace {

//! \name Helper functions
//@{

/// \brief Test that two algebraic types are close
/// \param[in] a The first algebraic type
/// \param[in] b The second algebraic type
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T>
MUNDY_REQUIRES(std::is_arithmetic_v<T>&& std::is_arithmetic_v<U>)
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
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_close_debug(const AMatrix3<U, Accessor1>& m1, const AMatrix3<T, Accessor2>& m2,
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
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_close_debug(const AVector3<U, Accessor1>& v1, const AVector3<T, Accessor2>& v2,
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
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_close_debug(const AQuaternion<U, Accessor1>& q1, const AQuaternion<T, Accessor2>& q2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(q1, q2)) {
    std::cout << "q1 = " << q1 << std::endl;
    std::cout << "q2 = " << q2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(q1, q2)) << message_if_fail;
}

//// \brief Test that two Matrix3s are different
/// \param[in] m1 The first AMatrix3
/// \param[in] m2 The second AMatrix3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_different_debug(const AMatrix3<U, Accessor1>& m1, const AMatrix3<T, Accessor2>& m2,
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
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_different_debug(const AVector3<U, Accessor1>& v1, const AVector3<T, Accessor2>& v2,
                        const std::string& message_if_fail = "") {
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
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
void is_different_debug(const AQuaternion<U, Accessor1>& q1, const AQuaternion<T, Accessor2>& q2,
                        const std::string& message_if_fail = "") {
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

/// \brief GTEST typed test fixture so we can run tests on multiple pairs of types
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
  // Following eigen: Construction order is w, x, y, z but underlying storage order is x, y, z, w.
  // So q[0] is x, q[1] is y, q[2] is z, and q[3] is w.
  Quaternion<TypeParam> q(1, 2, 3, 4);
  is_close_debug(q[0], 2);
  is_close_debug(q[1], 3);
  is_close_debug(q[2], 4);
  is_close_debug(q[3], 1);
  is_close_debug(q.w(), q[3]);
  is_close_debug(q.x(), q[0]);
  is_close_debug(q.y(), q[1]);
  is_close_debug(q.z(), q[2]);
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
  is_close_debug(q[0], 2);
  is_close_debug(q[1], 3);
  is_close_debug(q[2], 4);
  is_close_debug(q[3], 1);

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
  is_close_debug(q.x(), 4);
  q.w() = 5;
  is_close_debug(q.w(), 5);
  is_close_debug(q[3], 5);
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
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  Quaternion<T1> q1(1, 2, 3, 4);
  Quaternion<T2> q2(4, 10, 11, 12);
  auto q3 = q1 + q2;
  is_close_debug(q3, Quaternion<C>{5, 12, 14, 16}, "Addition failed.");

  q1 += q2;
  is_close_debug(q1, Quaternion<C>{5, 12, 14, 16}, "Addition assignment failed.");

  q3 = q1 - q2;
  is_close_debug(q3, Quaternion<C>{1, 2, 3, 4}, "Subtraction failed.");

  q1 -= q2;
  is_close_debug(q1, Quaternion<C>{1, 2, 3, 4}, "Subtraction assignment failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, AdditionAndSubtractionEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // Test that the addition and subtraction operators work with rvalues
  Quaternion<T1> q1(1, 2, 3, 4);
  auto q3 = q1 + Quaternion<T2>(4, 10, 11, 12);
  is_close_debug(q3, Quaternion<C>{5, 12, 14, 16}, "Right rvalue addition failed.");

  q1 += Quaternion<T2>(4, 10, 11, 12);
  is_close_debug(q1, Quaternion<C>{5, 12, 14, 16}, "Right rvalue addition assignment failed.");

  q3 = q1 - Quaternion<T2>(4, 10, 11, 12);
  is_close_debug(q3, Quaternion<C>{1, 2, 3, 4}, "Right rvalue subtraction failed.");

  q1 -= Quaternion<T2>(4, 10, 11, 12);
  is_close_debug(q1, Quaternion<C>{1, 2, 3, 4}, "Right rvalue subtraction assignment failed.");
}
//@}

//! \name Quaternion Multiplication and division
//@{

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithQuaternion) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // 90 degrees rotation around Z-axis
  Quaternion<T1> q1_z = get_quaternion_z_90<T1>();
  Quaternion<T2> q2_z = get_quaternion_y_90<T2>();
  Quaternion<C> expected_quat_z = {0.5, -0.5, 0.5, 0.5};
  is_close_debug(q1_z * q2_z, expected_quat_z, "90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis
  Quaternion<T1> q1_y = get_quaternion_y_90<T1>();
  Quaternion<T2> q2_y = get_quaternion_x_90<T2>();
  Quaternion<C> expected_quat_y = {0.5, 0.5, 0.5, -0.5};
  is_close_debug(q1_y * q2_y, expected_quat_y, "90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis
  Quaternion<T1> q1_x = get_quaternion_x_90<T1>();
  Quaternion<T2> q2_x = get_quaternion_z_90<T2>();
  Quaternion<C> expected_quat_x = {0.5, 0.5, -0.5, 0.5};
  is_close_debug(q1_x * q2_x, expected_quat_x, "90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithMatrix3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // Choose a random matrix to rotate
  Matrix3<T2> m(1, 2, 3, 4, 5, 6, -7, -8, -9);

  // Left multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: R_z m
  Quaternion<T1> q1_z = get_quaternion_z_90<T1>();
  Matrix3<T1> R_z = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  is_close_debug(R_z, quaternion_to_rotation_matrix(q1_z), "Rotation matrix-quaternion mismatch.");
  is_close_debug(R_z * m, Matrix3<C>{-4, -5, -6, 1, 2, 3, -7, -8, -9},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug((q1_z * m).template cast<C>(), (R_z * m).template cast<C>(),
                 "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y m
  Quaternion<T1> q1_y = get_quaternion_y_90<T1>();
  Matrix3<T1> R_y = {0, 0, 1, 0, 1, 0, -1, 0, 0};
  is_close_debug(R_y, quaternion_to_rotation_matrix(q1_y), "Rotation matrix-quaternion mismatch.");
  is_close_debug(R_y * m, Matrix3<C>{-7, -8, -9, 4, 5, 6, -1, -2, -3},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug((q1_y * m).template cast<C>(), (R_y * m).template cast<C>(),
                 "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x m
  Quaternion<T1> q1_x = get_quaternion_x_90<T1>();
  Matrix3<T1> R_x = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  is_close_debug(R_x, quaternion_to_rotation_matrix(q1_x), "Rotation matrix-quaternion mismatch.");
  is_close_debug(R_x * m, Matrix3<C>{1, 2, 3, 7, 8, 9, 4, 5, 6}, "Matrix-matrix multiplication sanity check failed.");
  is_close_debug((q1_x * m).template cast<C>(), (R_x * m).template cast<C>(),
                 "Left 90 degrees rotation around X-axis failed.");

  // Right multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: m R_z
  is_close_debug(m * R_z, Matrix3<C>{2, -1, 3, 5, -4, 6, -8, 7, -9},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug((m * q1_z).template cast<C>(), (m * R_z).template cast<C>(),
                 "Right 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: m R_y
  is_close_debug(m * R_y, Matrix3<C>{-3, 2, 1, -6, 5, 4, 9, -8, -7},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug((m * q1_y).template cast<C>(), (m * R_y).template cast<C>(),
                 "Right 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: m R_x
  is_close_debug(m * R_x, Matrix3<C>{1, 3, -2, 4, 6, -5, -7, -9, 8},
                 "Matrix-matrix multiplication sanity check failed.");
  is_close_debug((m * q1_x).template cast<C>(), (m * R_x).template cast<C>(),
                 "Right 90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithMatrix3sEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // Test that the multiplication and division operators work with rvalues

  // Left multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: R_z m
  Quaternion<T1> q1_z = get_quaternion_z_90<T1>();
  Matrix3<T1> R_z = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  is_close_debug(R_z, quaternion_to_rotation_matrix(q1_z), "Rotation matrix-quaternion mismatch.");
  is_close_debug((q1_z * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}).template cast<C>(),  //
                 (R_z * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}).template cast<C>(),   //
                 "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y m
  Quaternion<T1> q1_y = get_quaternion_y_90<T1>();
  Matrix3<T1> R_y = {0, 0, 1, 0, 1, 0, -1, 0, 0};
  is_close_debug(R_y, quaternion_to_rotation_matrix(q1_y), "Rotation matrix-quaternion mismatch.");
  is_close_debug((q1_y * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}).template cast<C>(),  //
                 (R_y * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}).template cast<C>(),   //
                 "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x m
  Quaternion<T1> q1_x = get_quaternion_x_90<T1>();
  Matrix3<T1> R_x = {1, 0, 0, 0, 0, -1, 0, 1, 0};
  is_close_debug(R_x, quaternion_to_rotation_matrix(q1_x), "Rotation matrix-quaternion mismatch.");
  is_close_debug((q1_x * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}).template cast<C>(),  //
                 (R_x * Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9}).template cast<C>(),   //
                 "Left 90 degrees rotation around X-axis failed.");

  // Right multiplication of a matrix by a quaternion
  // 90 degrees rotation around Z-axis: m R_z
  is_close_debug((Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * q1_z).template cast<C>(),  //
                 (Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * R_z).template cast<C>(),   //
                 "Right 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: m R_y
  is_close_debug((Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * q1_y).template cast<C>(),  //
                 (Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * R_y).template cast<C>(),   //
                 "Right 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: m R_x
  is_close_debug((Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * q1_x).template cast<C>(),  //
                 (Matrix3<T2>{1, 2, 3, 4, 5, 6, -7, -8, -9} * R_x).template cast<C>(),   //
                 "Right 90 degrees rotation around X-axis failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, MultiplicationAndDivisionWithVector3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // Choose a random vector to rotate
  Vector3<T2> v(1, 2, 3);

  // Left multiplication of a vector by a quaternion
  // 90 degrees rotation around Z-axis: R_z v
  Quaternion<T1> q_z = get_quaternion_z_90<T1>();
  Vector3<C> expected_v_z = {-2, 1, 3};
  is_close_debug(q_z * v, expected_v_z, "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y v
  Quaternion<T1> q_y = get_quaternion_y_90<T1>();
  Vector3<C> expected_v_y = {3, 2, -1};
  is_close_debug(q_y * v, expected_v_y, "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x v
  Quaternion<T1> q_x = get_quaternion_x_90<T1>();
  Vector3<C> expected_v_x = {1, -3, 2};
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
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // Test that the multiplication and division operators work with rvalues

  // Left multiplication of a vector by a quaternion
  // 90 degrees rotation around Z-axis: R_z v
  Quaternion<T1> q_z = get_quaternion_z_90<T1>();
  Vector3<C> expected_v_z = {-2, 1, 3};
  is_close_debug(q_z * Vector3<T2>{1, 2, 3}, expected_v_z, "Left 90 degrees rotation around Z-axis failed.");

  // 90 degrees rotation around Y-axis: R_y v
  Quaternion<T1> q_y = get_quaternion_y_90<T1>();
  Vector3<C> expected_v_y = {3, 2, -1};
  is_close_debug(q_y * Vector3<T2>{1, 2, 3}, expected_v_y, "Left 90 degrees rotation around Y-axis failed.");

  // 90 degrees rotation around X-axis: R_x v
  Quaternion<T1> q_x = get_quaternion_x_90<T1>();
  Vector3<C> expected_v_x = {1, -3, 2};
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
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  Quaternion<T1> q1(1, 2, 3, 4);
  auto q2 = q1 * T2(2);
  is_close_debug(q2, Quaternion<C>{2, 4, 6, 8}, "Right multiplication failed.");

  q2 = T2(2) * q1;
  is_close_debug(q2, Quaternion<C>{2, 4, 6, 8}, "Left multiplication failed.");

  q2 = q1 / T2(2);
  is_close_debug(q2, Quaternion<C>{0.5, 1, 1.5, 2}, "Right division failed.");

  q1 /= T2(2);
  is_close_debug(q1, Quaternion<C>{0.5, 1, 1.5, 2}, "Division assignment failed.");

  q1 *= T2(2);
  is_close_debug(q1, Quaternion<C>{1, 2, 3, 4}, "Multiplication assignment failed.");
}
//@}

//! \name Quaternion Special quaternion operations
//@{

TYPED_TEST(QuaternionPairwiseTypeTest, SpecialOperations) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // dot
  Quaternion<T1> q1(1, 2, 3, 4);
  Quaternion<T2> q2(4, 10, 11, 12);
  is_close_debug(dot(q1, q2), static_cast<C>(105), "Dot failed.");

  // conjugate
  auto q3 = conjugate(q1);
  is_close_debug(q3, Quaternion<C>{1, -2, -3, -4}, "Conjugate failed.");

  // conjugate in place
  q1.conjugate();
  is_close_debug(q1, Quaternion<C>{1, -2, -3, -4}, "Conjugate assignment failed.");
  q1 = {1, 2, 3, 4};

  // norm
  is_close_debug(norm(q1), static_cast<C>(std::sqrt(30.0)), "Norm failed.");

  // norm_squared
  is_close_debug(norm_squared(q1), static_cast<C>(30.0), "Norm squared failed.");

  // normalize
  auto q4 = normalize(q1);
  is_close_debug(q4,
                 Quaternion<C>{static_cast<C>(1.0 / std::sqrt(30.0)), static_cast<C>(2.0 / std::sqrt(30.0)),
                               static_cast<C>(3.0 / std::sqrt(30.0)), static_cast<C>(4.0 / std::sqrt(30.0))},
                 "Normalize assignment failed.");

  // inverse
  auto q5 = inverse(q1);
  is_close_debug(q5,
                 Quaternion<C>{static_cast<C>(1.0 / 30.0), static_cast<C>(-2.0 / 30.0), static_cast<C>(-3.0 / 30.0),
                               static_cast<C>(-4.0 / 30.0)},
                 "Inverse failed.");

  // normalize in place
  q1.normalize();
  q2.normalize();
  is_close_debug(q1,
                 Quaternion<C>{static_cast<C>(1.0 / std::sqrt(30.0)), static_cast<C>(2.0 / std::sqrt(30.0)),
                               static_cast<C>(3.0 / std::sqrt(30.0)), static_cast<C>(4.0 / std::sqrt(30.0))},
                 "Normalize failed.");

  // slerp (only applicable to unit quaternions)
  auto q6 = slerp(q1, q2, static_cast<C>(0.5));
  is_close_debug(q6,
                 Quaternion<C>{static_cast<C>(0.1946219299433149), static_cast<C>(0.4407059160784743),
                               static_cast<C>(0.5581347617390449), static_cast<C>(0.6755636074046377)},
                 "Slerp failed.");
}

TYPED_TEST(QuaternionPairwiseTypeTest, SpecialOperationsEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using C = decltype(get_comparison_tolerance_promote_ints<T1, T2>());

  // Test that the special vector operations work with rvalues

  // dot
  is_close_debug(dot(Quaternion<T1>(1, 2, 3, 4), Quaternion<T2>(4, 10, 11, 12)), static_cast<C>(105), "Dot failed.");

  // conjugate
  is_close_debug(conjugate(Quaternion<T1>(1, 2, 3, 4)), Quaternion<C>{1, -2, -3, -4}, "Conjugate failed.");

  // norm
  is_close_debug(norm(Quaternion<T1>(1, 2, 3, 4)), static_cast<C>(std::sqrt(30.0)), "Norm failed.");

  // norm_squared
  is_close_debug(norm_squared(Quaternion<T1>(1, 2, 3, 4)), static_cast<C>(30.0), "Norm squared failed.");

  // normalize
  auto q4 = normalize(Quaternion<T1>(1, 2, 3, 4));
  is_close_debug(q4,
                 Quaternion<C>{static_cast<C>(1.0 / std::sqrt(30.0)), static_cast<C>(2.0 / std::sqrt(30.0)),
                               static_cast<C>(3.0 / std::sqrt(30.0)), static_cast<C>(4.0 / std::sqrt(30.0))},
                 "Normalize failed.");

  // slerp
  auto q5 = slerp(normalize(Quaternion<T1>(1, 2, 3, 4)), normalize(Quaternion<T2>(4, 10, 11, 12)), static_cast<C>(0.5));
  is_close_debug(q5,
                 Quaternion<C>{static_cast<C>(0.1946219299433149), static_cast<C>(0.4407059160784743),
                               static_cast<C>(0.5581347617390449), static_cast<C>(0.6755636074046377)},
                 "Slerp failed.");
}
//@}

//! \name Quaternion Views
//@{

TYPED_TEST(QuaternionSingleTypeTest, Views) {
  // Create a view from a subset of an std::vector<TypeParam>
  std::vector<TypeParam> q1{0, 0, 2, 3, 4, 1, 0, 0};
  auto q2 = get_quaternion_view<TypeParam>(q1.data() + 2);
  is_close_debug(q2, Quaternion<TypeParam>{1, 2, 3, 4}, "View failed.");
  q1 = {1, 2, 10, 11, 12, 4, 13, 14};
  is_close_debug(q2, Quaternion<TypeParam>{4, 10, 11, 12}, "View isn't shallow.");

  // Create a view from a TypeParam*
  TypeParam q3[4] = {2, 3, 4, 1};
  auto q4 = get_quaternion_view<TypeParam>(&q3[0]);
  is_close_debug(q4, Quaternion<TypeParam>{1, 2, 3, 4}, "View failed.");
  q3[0] = 10;
  q3[1] = 11;
  q3[2] = 12;
  q3[3] = 4;
  is_close_debug(q4, Quaternion<TypeParam>{4, 10, 11, 12}, "View isn't shallow.");

  // Create a const view from an std::vector<TypeParam>
  const std::vector<TypeParam> q5{2, 3, 4, 1};
  auto q6 = get_quaternion_view<TypeParam>(q5.data());
  is_close_debug(q6, Quaternion<TypeParam>{1, 2, 3, 4}, "Const view failed.");
}
//@}

//! \name rotate_quaternion tests
//@{

// All quaternion equality checks in these tests use EXPECT_NEAR with tight tolerance
// rather than is_close_debug / is_approx_close (~1e-8).  The relaxed tolerance is far too
// loose: a bug that introduces 1e-9 drift per call would require ~10 million iterations to
// be detectable, while a tight tolerance catches it on the first call.

TEST(RotateQuaternion, ZeroOmegaIsNoOp) {
  // When omega == 0, the early-return branch must leave q unchanged exactly.
  constexpr double tol = 1e-15;
  const Quaternion<double> q_orig{0.5, 0.5, 0.5, 0.5};
  Quaternion<double> q = q_orig;
  rotate_quaternion(q, Vector3<double>{0.0, 0.0, 0.0}, 1.0);
  EXPECT_NEAR(q.w(), q_orig.w(), tol);
  EXPECT_NEAR(q.x(), q_orig.x(), tol);
  EXPECT_NEAR(q.y(), q_orig.y(), tol);
  EXPECT_NEAR(q.z(), q_orig.z(), tol);
}

TEST(RotateQuaternion, UnitNormPreserved) {
  // normalize() is called at the end of every non-trivial path, so ||q|| must stay at 1.
  // Using tight tolerance (1e-14) rather than is_approx_close (~1e-8): a broken normalize()
  // would show up after a single call, not only after millions.
  constexpr double tol = 1e-14;
  auto q = normalize(Quaternion<double>{1.0, 2.0, 3.0, 4.0});
  rotate_quaternion(q, Vector3<double>{1.0, -2.0, 3.0}, 0.37);
  EXPECT_NEAR(norm(q), 1.0, tol);

  // Large rotation angle — worst-case numerics for sin/cos.
  auto q2 = Quaternion<double>::identity();
  rotate_quaternion(q2, Vector3<double>{0.0, 0.0, 5.0}, 100.0);
  EXPECT_NEAR(norm(q2), 1.0, tol);
}

TEST(RotateQuaternion, FromIdentityAgreesWithAxisAngle) {
  // Starting from identity, rotating by omega for time dt must give the same quaternion
  // as axis_angle_to_quaternion(omega_hat, |omega|*dt).  Checked with tight tolerance
  // because both methods compute the same mathematical object; any deviation signals a bug.
  constexpr double tol = 1e-13;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  auto check = [&](Quaternion<double> q, const Quaternion<double>& expected, const char* label) {
    EXPECT_NEAR(q.w(), expected.w(), tol) << label;
    EXPECT_NEAR(q.x(), expected.x(), tol) << label;
    EXPECT_NEAR(q.y(), expected.y(), tol) << label;
    EXPECT_NEAR(q.z(), expected.z(), tol) << label;
  };

  // Case: rotation around z-axis
  {
    const double w = 2.0;
    const double dt = pi / 4.0;
    Quaternion<double> q = Quaternion<double>::identity();
    rotate_quaternion(q, Vector3<double>{0.0, 0.0, w}, dt);
    check(q, axis_angle_to_quaternion(Vector3<double>{0.0, 0.0, 1.0}, w * dt), "z-axis");
  }

  // Case: rotation around x-axis
  {
    const double w = 3.0;
    const double dt = pi / 6.0;
    Quaternion<double> q = Quaternion<double>::identity();
    rotate_quaternion(q, Vector3<double>{w, 0.0, 0.0}, dt);
    check(q, axis_angle_to_quaternion(Vector3<double>{1.0, 0.0, 0.0}, w * dt), "x-axis");
  }

  // Case: oblique axis
  {
    const Vector3<double> omega{1.0, 1.0, 1.0};
    const double dt = 0.5;
    const double w = norm(omega);
    Quaternion<double> q = Quaternion<double>::identity();
    rotate_quaternion(q, omega, dt);
    check(q, axis_angle_to_quaternion(omega / w, w * dt), "oblique axis");
  }
}

TEST(RotateQuaternion, TwoHalfStepsEqualOneFullStep) {
  // The Delong formula is the exact solution to dq/dt = (0,omega)*q/2 for constant omega,
  // so two half-steps must give the same result as one full step — not merely approximately,
  // but to floating-point precision.
  constexpr double tol = 1e-13;
  const auto q0 = normalize(Quaternion<double>{1.0, 2.0, -1.0, 0.5});
  const Vector3<double> omega{0.4, -1.2, 0.8};
  const double dt = 0.6;

  Quaternion<double> q_full = q0;
  rotate_quaternion(q_full, omega, dt);

  Quaternion<double> q_half = q0;
  rotate_quaternion(q_half, omega, dt / 2.0);
  rotate_quaternion(q_half, omega, dt / 2.0);

  EXPECT_NEAR(q_full.w(), q_half.w(), tol);
  EXPECT_NEAR(q_full.x(), q_half.x(), tol);
  EXPECT_NEAR(q_full.y(), q_half.y(), tol);
  EXPECT_NEAR(q_full.z(), q_half.z(), tol);
}

TEST(RotateQuaternion, Reversibility) {
  // Rotating by omega for dt then by omega for -dt must return to the original quaternion.
  constexpr double tol = 1e-13;
  const auto q0 = normalize(Quaternion<double>{0.6, -0.4, 0.5, 0.1});
  const Vector3<double> omega{1.0, 0.5, -0.7};
  const double dt = 0.8;

  Quaternion<double> q = q0;
  rotate_quaternion(q, omega, dt);
  rotate_quaternion(q, omega, -dt);

  EXPECT_NEAR(q.w(), q0.w(), tol);
  EXPECT_NEAR(q.x(), q0.x(), tol);
  EXPECT_NEAR(q.y(), q0.y(), tol);
  EXPECT_NEAR(q.z(), q0.z(), tol);
}

TEST(RotateQuaternion, FullRotationReturnsToCoveredQuaternion) {
  // Rotating by |omega|*dt = 2π is a full rotation: q must map back to ±q (double cover).
  constexpr double tol = 1e-13;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  const auto q0 = normalize(Quaternion<double>{0.3, 0.7, -0.5, 0.4});
  const Vector3<double> omega{2.0, 0.0, 0.0};
  const double dt = 2.0 * pi / norm(omega);  // |omega|*dt = 2pi

  Quaternion<double> q = q0;
  rotate_quaternion(q, omega, dt);

  // q must equal q0 or -q0 (both represent the same rotation).
  const bool same = (std::abs(q.w() - q0.w()) < tol && std::abs(q.x() - q0.x()) < tol &&
                     std::abs(q.y() - q0.y()) < tol && std::abs(q.z() - q0.z()) < tol);
  const bool neg = (std::abs(q.w() + q0.w()) < tol && std::abs(q.x() + q0.x()) < tol &&
                    std::abs(q.y() + q0.y()) < tol && std::abs(q.z() + q0.z()) < tol);
  EXPECT_TRUE(same || neg) << "Full rotation (2π) must return q to ±q0.";
}

TEST(RotateQuaternion, RotatedVectorAgreesWithComposedRotation) {
  // Geometric meaning: if q maps body→world, then after rotate_quaternion(q, omega, dt)
  // any body vector v transforms to (R*q)*v in world frame, which is the same as
  // first applying q's rotation then applying R's rotation.
  //
  // In other words: q_new * v == R * (q * v), where R = axis_angle(omega_hat, |omega|*dt).
  constexpr double tol = 1e-13;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  // Initial orientation: 90° around z
  const auto q0 = axis_angle_to_quaternion(Vector3<double>{0.0, 0.0, 1.0}, pi / 2.0);
  // Body vector along x-axis
  const Vector3<double> v_body{1.0, 0.0, 0.0};
  // Angular velocity: omega around z
  const Vector3<double> omega{0.0, 0.0, 2.0};
  const double dt = pi / 4.0;  // additional 90° rotation

  // Apply rotate_quaternion
  Quaternion<double> q_new = q0;
  rotate_quaternion(q_new, omega, dt);

  // Method 1: rotate v via q_new directly
  const auto v_via_q_new = q_new * v_body;

  // Method 2: rotate via q, then apply R
  const auto v_via_q = q0 * v_body;
  const auto R = axis_angle_to_quaternion(Vector3<double>{0.0, 0.0, 1.0}, norm(omega) * dt);
  const auto v_via_R = R * v_via_q;

  EXPECT_NEAR(v_via_q_new[0], v_via_R[0], tol);
  EXPECT_NEAR(v_via_q_new[1], v_via_R[1], tol);
  EXPECT_NEAR(v_via_q_new[2], v_via_R[2], tol);
}

TEST(RotateQuaternion, WorksWithFloat) {
  // Ensures rotate_quaternion is not hardcoded to double.
  // Float tolerance is loose (~1e-5) since float has ~7 significant digits.
  constexpr float tol = 1e-5f;
  constexpr float pi = Kokkos::numbers::pi_v<float>;

  Quaternion<float> q = Quaternion<float>::identity();
  rotate_quaternion(q, Vector3<float>{0.0f, 0.0f, 2.0f}, pi / 4.0f);
  const auto expected = axis_angle_to_quaternion(Vector3<float>{0.0f, 0.0f, 1.0f}, 2.0f * pi / 4.0f);

  EXPECT_NEAR(q.w(), expected.w(), tol);
  EXPECT_NEAR(q.x(), expected.x(), tol);
  EXPECT_NEAR(q.y(), expected.y(), tol);
  EXPECT_NEAR(q.z(), expected.z(), tol);
}

//@}

}  // namespace

}  // namespace mundy
