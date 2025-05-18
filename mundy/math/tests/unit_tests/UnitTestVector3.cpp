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
#include <mundy_math/Vector3.hpp>    // for mundy::math::Vector3

// Note, these tests are meant to look like real use cases for the Vector3 class. As a result, we use implicit type
// conversions rather than being explicit about types. This is to ensure that the Vector3 class can be used in a
// natural way. This choice means that compiling this test with -Wdouble-promotion or -Wconversion will result in many
// warnings. We will not however, locally disable these warnings.

namespace mundy {

namespace math {

namespace {

//! \name Helper functions
//@{

/// \brief Test that two Vector3s are close
/// \param[in] v1 The first Vector3
/// \param[in] v2 The second Vector3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename OtherAccessor, typename T, typename Accessor>
void is_close_debug(const AVector3<U, OtherAccessor>& v1, const AVector3<T, Accessor>& v2,
                    const std::string& message_if_fail = "") {
  if (!is_approx_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(is_approx_close(v1, v2)) << message_if_fail;
}

/// \brief Test that two Vector3s are different
/// \param[in] v1 The first Vector3
/// \param[in] v2 The second Vector3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename OtherAccessor, typename T, typename Accessor>
void is_different_debug(const AVector3<U, OtherAccessor>& v1, const AVector3<T, Accessor>& v2,
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
class Vector3SingleTypeTest : public ::testing::Test {
  using T = U;
};  // Vector3SingleTypeTest

/// \brief List of types to run the tests on
using MyTypes = ::testing::Types<int, float, double>;

/// \brief Tell GTEST to run the tests on the types in MyTypes
TYPED_TEST_SUITE(Vector3SingleTypeTest, MyTypes);

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
class Vector3PairwiseTypeTest : public ::testing::Test {};  // Vector3PairwiseTypeTest

/// \brief List of pairs of types to run the tests on
using MyTypePairs = ::testing::Types<TypePair<int, float>, TypePair<int, double>, TypePair<float, double>,
                                     TypePair<int, int>, TypePair<float, float>, TypePair<double, double>>;

/// \brief Tell GTEST to run the tests on the types in MyTypePairs
TYPED_TEST_SUITE(Vector3PairwiseTypeTest, MyTypePairs);
//@}

//! \name Vector3 Special vector operations
//@{

TYPED_TEST(Vector3PairwiseTypeTest, SpecialOperations) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Vector3<T1> v1(1, 2, 3);
  Vector3<T2> v2(4, 5, 6);
  is_close_debug(cross(v1, v2), Vector3<double>{-3.0, 6.0, -3.0}, "Cross product failed.");
  is_close_debug(element_multiply(v1, v2), Vector3<double>{4.0, 10.0, 18.0}, "Element-wise product failed.");
}
//@}

}  // namespace

}  // namespace math

}  // namespace mundy
