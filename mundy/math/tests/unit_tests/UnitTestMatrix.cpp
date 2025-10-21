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
#include <mundy_math/Matrix.hpp>    // for mundy::math::Matrix
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_relaxed_zero_tolerance
#include <mundy_math/Vector.hpp>    // for mundy::math::Vector

namespace mundy {

namespace math {

namespace {

// Some quick tests of non-square matrices
TEST(Matrix, NonSquareMatrices) {
  // Create a 2x3 matrix
  Matrix<double, 2, 3> m1(1., 2., 3., //
                          4., 5., 6.);
  EXPECT_DOUBLE_EQ(m1[0], 1.0);
  EXPECT_DOUBLE_EQ(m1[1], 2.0);
  EXPECT_DOUBLE_EQ(m1[2], 3.0);
  EXPECT_DOUBLE_EQ(m1[3], 4.0);
  EXPECT_DOUBLE_EQ(m1[4], 5.0);
  EXPECT_DOUBLE_EQ(m1[5], 6.0);

  EXPECT_DOUBLE_EQ(m1(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m1(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(m1(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(m1(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(m1(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(m1(1, 2), 6.0);
  
  // Transpose
  Matrix<double, 3, 2> m2 = transpose(m1);
  EXPECT_DOUBLE_EQ(m2(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m2(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(m2(1, 0), 2.0);
  EXPECT_DOUBLE_EQ(m2(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(m2(2, 0), 3.0);
  EXPECT_DOUBLE_EQ(m2(2, 1), 6.0);

  // Multiplication
  Matrix<double, 2, 2> m3 = m1 * m2;
  EXPECT_DOUBLE_EQ(m3(0, 0), 14.0);
  EXPECT_DOUBLE_EQ(m3(0, 1), 32.0);
  EXPECT_DOUBLE_EQ(m3(1, 0), 32.0);
  EXPECT_DOUBLE_EQ(m3(1, 1), 77.0);

  // Set row
  m1.set_row(0, Vector<double, 3>(7., 8., 9.));
  EXPECT_DOUBLE_EQ(m1(0, 0), 7.0);
  EXPECT_DOUBLE_EQ(m1(0, 1), 8.0);
  EXPECT_DOUBLE_EQ(m1(0, 2), 9.0);
  EXPECT_DOUBLE_EQ(m1(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(m1(1, 1), 5.0);
  EXPECT_DOUBLE_EQ(m1(1, 2), 6.0);
  m1.set_row(1, Vector<double, 3>(10., 11., 12.));
  EXPECT_DOUBLE_EQ(m1(0, 0), 7.0);
  EXPECT_DOUBLE_EQ(m1(0, 1), 8.0);
  EXPECT_DOUBLE_EQ(m1(0, 2), 9.0);
  EXPECT_DOUBLE_EQ(m1(1, 0), 10.0);
  EXPECT_DOUBLE_EQ(m1(1, 1), 11.0);
  EXPECT_DOUBLE_EQ(m1(1, 2), 12.0);

  // Set column
  m1.set_column(0, Vector<double, 2>(13., 14.));
  EXPECT_DOUBLE_EQ(m1(0, 0), 13.0);
  EXPECT_DOUBLE_EQ(m1(0, 1), 8.0);
  EXPECT_DOUBLE_EQ(m1(0, 2), 9.0);
  EXPECT_DOUBLE_EQ(m1(1, 0), 14.0);
  EXPECT_DOUBLE_EQ(m1(1, 1), 11.0);
  EXPECT_DOUBLE_EQ(m1(1, 2), 12.0);
  m1.set_column(1, Vector<double, 2>(15., 16.));
  EXPECT_DOUBLE_EQ(m1(0, 0), 13.0);
  EXPECT_DOUBLE_EQ(m1(0, 1), 15.0);
  EXPECT_DOUBLE_EQ(m1(0, 2), 9.0);
  EXPECT_DOUBLE_EQ(m1(1, 0), 14.0);
  EXPECT_DOUBLE_EQ(m1(1, 1), 16.0);
  EXPECT_DOUBLE_EQ(m1(1, 2), 12.0);
}


}  // namespace

}  // namespace math

}  // namespace mundy
