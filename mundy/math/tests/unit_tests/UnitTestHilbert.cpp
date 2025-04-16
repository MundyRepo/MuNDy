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
#include <mundy_math/Hilbert.hpp>  // for mundy::math::hilbert_3d

// Test hilbert3d space filling curves

namespace mundy {

namespace math {

namespace {

//! \name Hilbert3D CubeN tests
//@{

TEST(Hilbert3D, Cube2) {
  size_t s = 2;
  size_t ind = 0;
  std::vector<mundy::math::Vector3<double>> position_array(s * s * s);
  mundy::math::Vector3<double> current_position(0.0, 0.0, 0.0);
  mundy::math::Vector3<double> dr1(1.0, 0.0, 0.0);
  mundy::math::Vector3<double> dr2(0.0, 1.0, 0.0);
  mundy::math::Vector3<double> dr3(0.0, 0.0, 1.0);

  ind = hilbert_3d(s, ind, position_array, current_position, dr1, dr2, dr3);

  std::vector<mundy::math::Vector3<double>> expected_position_array = {
      mundy::math::Vector3<double>(0.0, 0.0, 0.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(1.0, 1.0, 0.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 1.0, 1.0), mundy::math::Vector3<double>(1.0, 1.0, 1.0),
      mundy::math::Vector3<double>(1.0, 0.0, 1.0), mundy::math::Vector3<double>(0.0, 0.0, 1.0)};
  for (size_t i = 0; i < s * s * s; ++i) {
    ASSERT_TRUE(is_close(position_array[i], expected_position_array[i]));
  }
}

TEST(Hilbert3D, Cube4) {
  size_t s = 4;
  size_t ind = 0;
  std::vector<mundy::math::Vector3<double>> position_array(s * s * s);
  mundy::math::Vector3<double> current_position(0.0, 0.0, 0.0);
  mundy::math::Vector3<double> dr1(1.0, 0.0, 0.0);
  mundy::math::Vector3<double> dr2(0.0, 1.0, 0.0);
  mundy::math::Vector3<double> dr3(0.0, 0.0, 1.0);

  ind = hilbert_3d(s, ind, position_array, current_position, dr1, dr2, dr3);

  std::vector<mundy::math::Vector3<double>> expected_position_array = {
      mundy::math::Vector3<double>(0.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 1.0, 1.0), mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(1.0, 0.0, 1.0), mundy::math::Vector3<double>(1.0, 1.0, 1.0),
      mundy::math::Vector3<double>(1.0, 1.0, 0.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(2.0, 0.0, 0.0), mundy::math::Vector3<double>(2.0, 0.0, 1.0),
      mundy::math::Vector3<double>(3.0, 0.0, 1.0), mundy::math::Vector3<double>(3.0, 0.0, 0.0),
      mundy::math::Vector3<double>(3.0, 1.0, 0.0), mundy::math::Vector3<double>(3.0, 1.0, 1.0),
      mundy::math::Vector3<double>(2.0, 1.0, 1.0), mundy::math::Vector3<double>(2.0, 1.0, 0.0),
      mundy::math::Vector3<double>(2.0, 2.0, 0.0), mundy::math::Vector3<double>(2.0, 2.0, 1.0),
      mundy::math::Vector3<double>(3.0, 2.0, 1.0), mundy::math::Vector3<double>(3.0, 2.0, 0.0),
      mundy::math::Vector3<double>(3.0, 3.0, 0.0), mundy::math::Vector3<double>(3.0, 3.0, 1.0),
      mundy::math::Vector3<double>(2.0, 3.0, 1.0), mundy::math::Vector3<double>(2.0, 3.0, 0.0),
      mundy::math::Vector3<double>(1.0, 3.0, 0.0), mundy::math::Vector3<double>(0.0, 3.0, 0.0),
      mundy::math::Vector3<double>(0.0, 2.0, 0.0), mundy::math::Vector3<double>(1.0, 2.0, 0.0),
      mundy::math::Vector3<double>(1.0, 2.0, 1.0), mundy::math::Vector3<double>(0.0, 2.0, 1.0),
      mundy::math::Vector3<double>(0.0, 3.0, 1.0), mundy::math::Vector3<double>(1.0, 3.0, 1.0),
      mundy::math::Vector3<double>(1.0, 3.0, 2.0), mundy::math::Vector3<double>(0.0, 3.0, 2.0),
      mundy::math::Vector3<double>(0.0, 2.0, 2.0), mundy::math::Vector3<double>(1.0, 2.0, 2.0),
      mundy::math::Vector3<double>(1.0, 2.0, 3.0), mundy::math::Vector3<double>(0.0, 2.0, 3.0),
      mundy::math::Vector3<double>(0.0, 3.0, 3.0), mundy::math::Vector3<double>(1.0, 3.0, 3.0),
      mundy::math::Vector3<double>(2.0, 3.0, 3.0), mundy::math::Vector3<double>(2.0, 3.0, 2.0),
      mundy::math::Vector3<double>(3.0, 3.0, 2.0), mundy::math::Vector3<double>(3.0, 3.0, 3.0),
      mundy::math::Vector3<double>(3.0, 2.0, 3.0), mundy::math::Vector3<double>(3.0, 2.0, 2.0),
      mundy::math::Vector3<double>(2.0, 2.0, 2.0), mundy::math::Vector3<double>(2.0, 2.0, 3.0),
      mundy::math::Vector3<double>(2.0, 1.0, 3.0), mundy::math::Vector3<double>(2.0, 1.0, 2.0),
      mundy::math::Vector3<double>(3.0, 1.0, 2.0), mundy::math::Vector3<double>(3.0, 1.0, 3.0),
      mundy::math::Vector3<double>(3.0, 0.0, 3.0), mundy::math::Vector3<double>(3.0, 0.0, 2.0),
      mundy::math::Vector3<double>(2.0, 0.0, 2.0), mundy::math::Vector3<double>(2.0, 0.0, 3.0),
      mundy::math::Vector3<double>(1.0, 0.0, 3.0), mundy::math::Vector3<double>(1.0, 1.0, 3.0),
      mundy::math::Vector3<double>(1.0, 1.0, 2.0), mundy::math::Vector3<double>(1.0, 0.0, 2.0),
      mundy::math::Vector3<double>(0.0, 0.0, 2.0), mundy::math::Vector3<double>(0.0, 1.0, 2.0),
      mundy::math::Vector3<double>(0.0, 1.0, 3.0), mundy::math::Vector3<double>(0.0, 0.0, 3.0)};

  for (size_t i = 0; i < expected_position_array.size(); ++i) {
    ASSERT_TRUE(is_close(position_array[i], expected_position_array[i]));
  }
}

TEST(Hilbert3D, Cube8) {
  size_t s = 8;
  size_t ind = 0;
  std::vector<mundy::math::Vector3<double>> position_array(s * s * s);
  mundy::math::Vector3<double> current_position(0.0, 0.0, 0.0);
  mundy::math::Vector3<double> dr1(1.0, 0.0, 0.0);
  mundy::math::Vector3<double> dr2(0.0, 1.0, 0.0);
  mundy::math::Vector3<double> dr3(0.0, 0.0, 1.0);

  ind = hilbert_3d(s, ind, position_array, current_position, dr1, dr2, dr3);

  std::vector<mundy::math::Vector3<double>> expected_position_array = {
      mundy::math::Vector3<double>(0.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(1.0, 0.0, 1.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(1.0, 1.0, 0.0), mundy::math::Vector3<double>(1.0, 1.0, 1.0),
      mundy::math::Vector3<double>(0.0, 1.0, 1.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 2.0, 0.0), mundy::math::Vector3<double>(1.0, 2.0, 0.0),
      mundy::math::Vector3<double>(1.0, 3.0, 0.0), mundy::math::Vector3<double>(0.0, 3.0, 0.0),
      mundy::math::Vector3<double>(0.0, 3.0, 1.0), mundy::math::Vector3<double>(1.0, 3.0, 1.0),
      mundy::math::Vector3<double>(1.0, 2.0, 1.0), mundy::math::Vector3<double>(0.0, 2.0, 1.0),
      mundy::math::Vector3<double>(0.0, 2.0, 2.0), mundy::math::Vector3<double>(1.0, 2.0, 2.0),
      mundy::math::Vector3<double>(1.0, 3.0, 2.0), mundy::math::Vector3<double>(0.0, 3.0, 2.0),
      mundy::math::Vector3<double>(0.0, 3.0, 3.0), mundy::math::Vector3<double>(1.0, 3.0, 3.0),
      mundy::math::Vector3<double>(1.0, 2.0, 3.0), mundy::math::Vector3<double>(0.0, 2.0, 3.0),
      mundy::math::Vector3<double>(0.0, 1.0, 3.0), mundy::math::Vector3<double>(0.0, 0.0, 3.0),
      mundy::math::Vector3<double>(0.0, 0.0, 2.0), mundy::math::Vector3<double>(0.0, 1.0, 2.0),
      mundy::math::Vector3<double>(1.0, 1.0, 2.0), mundy::math::Vector3<double>(1.0, 0.0, 2.0),
      mundy::math::Vector3<double>(1.0, 0.0, 3.0), mundy::math::Vector3<double>(1.0, 1.0, 3.0),
      mundy::math::Vector3<double>(2.0, 1.0, 3.0), mundy::math::Vector3<double>(2.0, 0.0, 3.0),
      mundy::math::Vector3<double>(2.0, 0.0, 2.0), mundy::math::Vector3<double>(2.0, 1.0, 2.0),
      mundy::math::Vector3<double>(3.0, 1.0, 2.0), mundy::math::Vector3<double>(3.0, 0.0, 2.0),
      mundy::math::Vector3<double>(3.0, 0.0, 3.0), mundy::math::Vector3<double>(3.0, 1.0, 3.0),
      mundy::math::Vector3<double>(3.0, 2.0, 3.0), mundy::math::Vector3<double>(2.0, 2.0, 3.0),
      mundy::math::Vector3<double>(2.0, 3.0, 3.0), mundy::math::Vector3<double>(3.0, 3.0, 3.0),
      mundy::math::Vector3<double>(3.0, 3.0, 2.0), mundy::math::Vector3<double>(2.0, 3.0, 2.0),
      mundy::math::Vector3<double>(2.0, 2.0, 2.0), mundy::math::Vector3<double>(3.0, 2.0, 2.0),
      mundy::math::Vector3<double>(3.0, 2.0, 1.0), mundy::math::Vector3<double>(2.0, 2.0, 1.0),
      mundy::math::Vector3<double>(2.0, 3.0, 1.0), mundy::math::Vector3<double>(3.0, 3.0, 1.0),
      mundy::math::Vector3<double>(3.0, 3.0, 0.0), mundy::math::Vector3<double>(2.0, 3.0, 0.0),
      mundy::math::Vector3<double>(2.0, 2.0, 0.0), mundy::math::Vector3<double>(3.0, 2.0, 0.0),
      mundy::math::Vector3<double>(3.0, 1.0, 0.0), mundy::math::Vector3<double>(3.0, 1.0, 1.0),
      mundy::math::Vector3<double>(2.0, 1.0, 1.0), mundy::math::Vector3<double>(2.0, 1.0, 0.0),
      mundy::math::Vector3<double>(2.0, 0.0, 0.0), mundy::math::Vector3<double>(2.0, 0.0, 1.0),
      mundy::math::Vector3<double>(3.0, 0.0, 1.0), mundy::math::Vector3<double>(3.0, 0.0, 0.0),
      mundy::math::Vector3<double>(4.0, 0.0, 0.0), mundy::math::Vector3<double>(5.0, 0.0, 0.0),
      mundy::math::Vector3<double>(5.0, 1.0, 0.0), mundy::math::Vector3<double>(4.0, 1.0, 0.0),
      mundy::math::Vector3<double>(4.0, 1.0, 1.0), mundy::math::Vector3<double>(5.0, 1.0, 1.0),
      mundy::math::Vector3<double>(5.0, 0.0, 1.0), mundy::math::Vector3<double>(4.0, 0.0, 1.0),
      mundy::math::Vector3<double>(4.0, 0.0, 2.0), mundy::math::Vector3<double>(4.0, 1.0, 2.0),
      mundy::math::Vector3<double>(4.0, 1.0, 3.0), mundy::math::Vector3<double>(4.0, 0.0, 3.0),
      mundy::math::Vector3<double>(5.0, 0.0, 3.0), mundy::math::Vector3<double>(5.0, 1.0, 3.0),
      mundy::math::Vector3<double>(5.0, 1.0, 2.0), mundy::math::Vector3<double>(5.0, 0.0, 2.0),
      mundy::math::Vector3<double>(6.0, 0.0, 2.0), mundy::math::Vector3<double>(6.0, 1.0, 2.0),
      mundy::math::Vector3<double>(6.0, 1.0, 3.0), mundy::math::Vector3<double>(6.0, 0.0, 3.0),
      mundy::math::Vector3<double>(7.0, 0.0, 3.0), mundy::math::Vector3<double>(7.0, 1.0, 3.0),
      mundy::math::Vector3<double>(7.0, 1.0, 2.0), mundy::math::Vector3<double>(7.0, 0.0, 2.0),
      mundy::math::Vector3<double>(7.0, 0.0, 1.0), mundy::math::Vector3<double>(7.0, 0.0, 0.0),
      mundy::math::Vector3<double>(6.0, 0.0, 0.0), mundy::math::Vector3<double>(6.0, 0.0, 1.0),
      mundy::math::Vector3<double>(6.0, 1.0, 1.0), mundy::math::Vector3<double>(6.0, 1.0, 0.0),
      mundy::math::Vector3<double>(7.0, 1.0, 0.0), mundy::math::Vector3<double>(7.0, 1.0, 1.0),
      mundy::math::Vector3<double>(7.0, 2.0, 1.0), mundy::math::Vector3<double>(7.0, 2.0, 0.0),
      mundy::math::Vector3<double>(6.0, 2.0, 0.0), mundy::math::Vector3<double>(6.0, 2.0, 1.0),
      mundy::math::Vector3<double>(6.0, 3.0, 1.0), mundy::math::Vector3<double>(6.0, 3.0, 0.0),
      mundy::math::Vector3<double>(7.0, 3.0, 0.0), mundy::math::Vector3<double>(7.0, 3.0, 1.0),
      mundy::math::Vector3<double>(7.0, 3.0, 2.0), mundy::math::Vector3<double>(7.0, 2.0, 2.0),
      mundy::math::Vector3<double>(7.0, 2.0, 3.0), mundy::math::Vector3<double>(7.0, 3.0, 3.0),
      mundy::math::Vector3<double>(6.0, 3.0, 3.0), mundy::math::Vector3<double>(6.0, 2.0, 3.0),
      mundy::math::Vector3<double>(6.0, 2.0, 2.0), mundy::math::Vector3<double>(6.0, 3.0, 2.0),
      mundy::math::Vector3<double>(5.0, 3.0, 2.0), mundy::math::Vector3<double>(5.0, 2.0, 2.0),
      mundy::math::Vector3<double>(5.0, 2.0, 3.0), mundy::math::Vector3<double>(5.0, 3.0, 3.0),
      mundy::math::Vector3<double>(4.0, 3.0, 3.0), mundy::math::Vector3<double>(4.0, 2.0, 3.0),
      mundy::math::Vector3<double>(4.0, 2.0, 2.0), mundy::math::Vector3<double>(4.0, 3.0, 2.0),
      mundy::math::Vector3<double>(4.0, 3.0, 1.0), mundy::math::Vector3<double>(5.0, 3.0, 1.0),
      mundy::math::Vector3<double>(5.0, 2.0, 1.0), mundy::math::Vector3<double>(4.0, 2.0, 1.0),
      mundy::math::Vector3<double>(4.0, 2.0, 0.0), mundy::math::Vector3<double>(5.0, 2.0, 0.0),
      mundy::math::Vector3<double>(5.0, 3.0, 0.0), mundy::math::Vector3<double>(4.0, 3.0, 0.0),
      mundy::math::Vector3<double>(4.0, 4.0, 0.0), mundy::math::Vector3<double>(5.0, 4.0, 0.0),
      mundy::math::Vector3<double>(5.0, 5.0, 0.0), mundy::math::Vector3<double>(4.0, 5.0, 0.0),
      mundy::math::Vector3<double>(4.0, 5.0, 1.0), mundy::math::Vector3<double>(5.0, 5.0, 1.0),
      mundy::math::Vector3<double>(5.0, 4.0, 1.0), mundy::math::Vector3<double>(4.0, 4.0, 1.0),
      mundy::math::Vector3<double>(4.0, 4.0, 2.0), mundy::math::Vector3<double>(4.0, 5.0, 2.0),
      mundy::math::Vector3<double>(4.0, 5.0, 3.0), mundy::math::Vector3<double>(4.0, 4.0, 3.0),
      mundy::math::Vector3<double>(5.0, 4.0, 3.0), mundy::math::Vector3<double>(5.0, 5.0, 3.0),
      mundy::math::Vector3<double>(5.0, 5.0, 2.0), mundy::math::Vector3<double>(5.0, 4.0, 2.0),
      mundy::math::Vector3<double>(6.0, 4.0, 2.0), mundy::math::Vector3<double>(6.0, 5.0, 2.0),
      mundy::math::Vector3<double>(6.0, 5.0, 3.0), mundy::math::Vector3<double>(6.0, 4.0, 3.0),
      mundy::math::Vector3<double>(7.0, 4.0, 3.0), mundy::math::Vector3<double>(7.0, 5.0, 3.0),
      mundy::math::Vector3<double>(7.0, 5.0, 2.0), mundy::math::Vector3<double>(7.0, 4.0, 2.0),
      mundy::math::Vector3<double>(7.0, 4.0, 1.0), mundy::math::Vector3<double>(7.0, 4.0, 0.0),
      mundy::math::Vector3<double>(6.0, 4.0, 0.0), mundy::math::Vector3<double>(6.0, 4.0, 1.0),
      mundy::math::Vector3<double>(6.0, 5.0, 1.0), mundy::math::Vector3<double>(6.0, 5.0, 0.0),
      mundy::math::Vector3<double>(7.0, 5.0, 0.0), mundy::math::Vector3<double>(7.0, 5.0, 1.0),
      mundy::math::Vector3<double>(7.0, 6.0, 1.0), mundy::math::Vector3<double>(7.0, 6.0, 0.0),
      mundy::math::Vector3<double>(6.0, 6.0, 0.0), mundy::math::Vector3<double>(6.0, 6.0, 1.0),
      mundy::math::Vector3<double>(6.0, 7.0, 1.0), mundy::math::Vector3<double>(6.0, 7.0, 0.0),
      mundy::math::Vector3<double>(7.0, 7.0, 0.0), mundy::math::Vector3<double>(7.0, 7.0, 1.0),
      mundy::math::Vector3<double>(7.0, 7.0, 2.0), mundy::math::Vector3<double>(7.0, 6.0, 2.0),
      mundy::math::Vector3<double>(7.0, 6.0, 3.0), mundy::math::Vector3<double>(7.0, 7.0, 3.0),
      mundy::math::Vector3<double>(6.0, 7.0, 3.0), mundy::math::Vector3<double>(6.0, 6.0, 3.0),
      mundy::math::Vector3<double>(6.0, 6.0, 2.0), mundy::math::Vector3<double>(6.0, 7.0, 2.0),
      mundy::math::Vector3<double>(5.0, 7.0, 2.0), mundy::math::Vector3<double>(5.0, 6.0, 2.0),
      mundy::math::Vector3<double>(5.0, 6.0, 3.0), mundy::math::Vector3<double>(5.0, 7.0, 3.0),
      mundy::math::Vector3<double>(4.0, 7.0, 3.0), mundy::math::Vector3<double>(4.0, 6.0, 3.0),
      mundy::math::Vector3<double>(4.0, 6.0, 2.0), mundy::math::Vector3<double>(4.0, 7.0, 2.0),
      mundy::math::Vector3<double>(4.0, 7.0, 1.0), mundy::math::Vector3<double>(5.0, 7.0, 1.0),
      mundy::math::Vector3<double>(5.0, 6.0, 1.0), mundy::math::Vector3<double>(4.0, 6.0, 1.0),
      mundy::math::Vector3<double>(4.0, 6.0, 0.0), mundy::math::Vector3<double>(5.0, 6.0, 0.0),
      mundy::math::Vector3<double>(5.0, 7.0, 0.0), mundy::math::Vector3<double>(4.0, 7.0, 0.0),
      mundy::math::Vector3<double>(3.0, 7.0, 0.0), mundy::math::Vector3<double>(3.0, 6.0, 0.0),
      mundy::math::Vector3<double>(3.0, 6.0, 1.0), mundy::math::Vector3<double>(3.0, 7.0, 1.0),
      mundy::math::Vector3<double>(2.0, 7.0, 1.0), mundy::math::Vector3<double>(2.0, 6.0, 1.0),
      mundy::math::Vector3<double>(2.0, 6.0, 0.0), mundy::math::Vector3<double>(2.0, 7.0, 0.0),
      mundy::math::Vector3<double>(1.0, 7.0, 0.0), mundy::math::Vector3<double>(1.0, 7.0, 1.0),
      mundy::math::Vector3<double>(0.0, 7.0, 1.0), mundy::math::Vector3<double>(0.0, 7.0, 0.0),
      mundy::math::Vector3<double>(0.0, 6.0, 0.0), mundy::math::Vector3<double>(0.0, 6.0, 1.0),
      mundy::math::Vector3<double>(1.0, 6.0, 1.0), mundy::math::Vector3<double>(1.0, 6.0, 0.0),
      mundy::math::Vector3<double>(1.0, 5.0, 0.0), mundy::math::Vector3<double>(1.0, 5.0, 1.0),
      mundy::math::Vector3<double>(0.0, 5.0, 1.0), mundy::math::Vector3<double>(0.0, 5.0, 0.0),
      mundy::math::Vector3<double>(0.0, 4.0, 0.0), mundy::math::Vector3<double>(0.0, 4.0, 1.0),
      mundy::math::Vector3<double>(1.0, 4.0, 1.0), mundy::math::Vector3<double>(1.0, 4.0, 0.0),
      mundy::math::Vector3<double>(2.0, 4.0, 0.0), mundy::math::Vector3<double>(3.0, 4.0, 0.0),
      mundy::math::Vector3<double>(3.0, 5.0, 0.0), mundy::math::Vector3<double>(2.0, 5.0, 0.0),
      mundy::math::Vector3<double>(2.0, 5.0, 1.0), mundy::math::Vector3<double>(3.0, 5.0, 1.0),
      mundy::math::Vector3<double>(3.0, 4.0, 1.0), mundy::math::Vector3<double>(2.0, 4.0, 1.0),
      mundy::math::Vector3<double>(2.0, 4.0, 2.0), mundy::math::Vector3<double>(3.0, 4.0, 2.0),
      mundy::math::Vector3<double>(3.0, 5.0, 2.0), mundy::math::Vector3<double>(2.0, 5.0, 2.0),
      mundy::math::Vector3<double>(2.0, 5.0, 3.0), mundy::math::Vector3<double>(3.0, 5.0, 3.0),
      mundy::math::Vector3<double>(3.0, 4.0, 3.0), mundy::math::Vector3<double>(2.0, 4.0, 3.0),
      mundy::math::Vector3<double>(1.0, 4.0, 3.0), mundy::math::Vector3<double>(1.0, 4.0, 2.0),
      mundy::math::Vector3<double>(0.0, 4.0, 2.0), mundy::math::Vector3<double>(0.0, 4.0, 3.0),
      mundy::math::Vector3<double>(0.0, 5.0, 3.0), mundy::math::Vector3<double>(0.0, 5.0, 2.0),
      mundy::math::Vector3<double>(1.0, 5.0, 2.0), mundy::math::Vector3<double>(1.0, 5.0, 3.0),
      mundy::math::Vector3<double>(1.0, 6.0, 3.0), mundy::math::Vector3<double>(1.0, 6.0, 2.0),
      mundy::math::Vector3<double>(0.0, 6.0, 2.0), mundy::math::Vector3<double>(0.0, 6.0, 3.0),
      mundy::math::Vector3<double>(0.0, 7.0, 3.0), mundy::math::Vector3<double>(0.0, 7.0, 2.0),
      mundy::math::Vector3<double>(1.0, 7.0, 2.0), mundy::math::Vector3<double>(1.0, 7.0, 3.0),
      mundy::math::Vector3<double>(2.0, 7.0, 3.0), mundy::math::Vector3<double>(2.0, 6.0, 3.0),
      mundy::math::Vector3<double>(2.0, 6.0, 2.0), mundy::math::Vector3<double>(2.0, 7.0, 2.0),
      mundy::math::Vector3<double>(3.0, 7.0, 2.0), mundy::math::Vector3<double>(3.0, 6.0, 2.0),
      mundy::math::Vector3<double>(3.0, 6.0, 3.0), mundy::math::Vector3<double>(3.0, 7.0, 3.0),
      mundy::math::Vector3<double>(3.0, 7.0, 4.0), mundy::math::Vector3<double>(3.0, 6.0, 4.0),
      mundy::math::Vector3<double>(3.0, 6.0, 5.0), mundy::math::Vector3<double>(3.0, 7.0, 5.0),
      mundy::math::Vector3<double>(2.0, 7.0, 5.0), mundy::math::Vector3<double>(2.0, 6.0, 5.0),
      mundy::math::Vector3<double>(2.0, 6.0, 4.0), mundy::math::Vector3<double>(2.0, 7.0, 4.0),
      mundy::math::Vector3<double>(1.0, 7.0, 4.0), mundy::math::Vector3<double>(1.0, 7.0, 5.0),
      mundy::math::Vector3<double>(0.0, 7.0, 5.0), mundy::math::Vector3<double>(0.0, 7.0, 4.0),
      mundy::math::Vector3<double>(0.0, 6.0, 4.0), mundy::math::Vector3<double>(0.0, 6.0, 5.0),
      mundy::math::Vector3<double>(1.0, 6.0, 5.0), mundy::math::Vector3<double>(1.0, 6.0, 4.0),
      mundy::math::Vector3<double>(1.0, 5.0, 4.0), mundy::math::Vector3<double>(1.0, 5.0, 5.0),
      mundy::math::Vector3<double>(0.0, 5.0, 5.0), mundy::math::Vector3<double>(0.0, 5.0, 4.0),
      mundy::math::Vector3<double>(0.0, 4.0, 4.0), mundy::math::Vector3<double>(0.0, 4.0, 5.0),
      mundy::math::Vector3<double>(1.0, 4.0, 5.0), mundy::math::Vector3<double>(1.0, 4.0, 4.0),
      mundy::math::Vector3<double>(2.0, 4.0, 4.0), mundy::math::Vector3<double>(3.0, 4.0, 4.0),
      mundy::math::Vector3<double>(3.0, 5.0, 4.0), mundy::math::Vector3<double>(2.0, 5.0, 4.0),
      mundy::math::Vector3<double>(2.0, 5.0, 5.0), mundy::math::Vector3<double>(3.0, 5.0, 5.0),
      mundy::math::Vector3<double>(3.0, 4.0, 5.0), mundy::math::Vector3<double>(2.0, 4.0, 5.0),
      mundy::math::Vector3<double>(2.0, 4.0, 6.0), mundy::math::Vector3<double>(3.0, 4.0, 6.0),
      mundy::math::Vector3<double>(3.0, 5.0, 6.0), mundy::math::Vector3<double>(2.0, 5.0, 6.0),
      mundy::math::Vector3<double>(2.0, 5.0, 7.0), mundy::math::Vector3<double>(3.0, 5.0, 7.0),
      mundy::math::Vector3<double>(3.0, 4.0, 7.0), mundy::math::Vector3<double>(2.0, 4.0, 7.0),
      mundy::math::Vector3<double>(1.0, 4.0, 7.0), mundy::math::Vector3<double>(1.0, 4.0, 6.0),
      mundy::math::Vector3<double>(0.0, 4.0, 6.0), mundy::math::Vector3<double>(0.0, 4.0, 7.0),
      mundy::math::Vector3<double>(0.0, 5.0, 7.0), mundy::math::Vector3<double>(0.0, 5.0, 6.0),
      mundy::math::Vector3<double>(1.0, 5.0, 6.0), mundy::math::Vector3<double>(1.0, 5.0, 7.0),
      mundy::math::Vector3<double>(1.0, 6.0, 7.0), mundy::math::Vector3<double>(1.0, 6.0, 6.0),
      mundy::math::Vector3<double>(0.0, 6.0, 6.0), mundy::math::Vector3<double>(0.0, 6.0, 7.0),
      mundy::math::Vector3<double>(0.0, 7.0, 7.0), mundy::math::Vector3<double>(0.0, 7.0, 6.0),
      mundy::math::Vector3<double>(1.0, 7.0, 6.0), mundy::math::Vector3<double>(1.0, 7.0, 7.0),
      mundy::math::Vector3<double>(2.0, 7.0, 7.0), mundy::math::Vector3<double>(2.0, 6.0, 7.0),
      mundy::math::Vector3<double>(2.0, 6.0, 6.0), mundy::math::Vector3<double>(2.0, 7.0, 6.0),
      mundy::math::Vector3<double>(3.0, 7.0, 6.0), mundy::math::Vector3<double>(3.0, 6.0, 6.0),
      mundy::math::Vector3<double>(3.0, 6.0, 7.0), mundy::math::Vector3<double>(3.0, 7.0, 7.0),
      mundy::math::Vector3<double>(4.0, 7.0, 7.0), mundy::math::Vector3<double>(5.0, 7.0, 7.0),
      mundy::math::Vector3<double>(5.0, 6.0, 7.0), mundy::math::Vector3<double>(4.0, 6.0, 7.0),
      mundy::math::Vector3<double>(4.0, 6.0, 6.0), mundy::math::Vector3<double>(5.0, 6.0, 6.0),
      mundy::math::Vector3<double>(5.0, 7.0, 6.0), mundy::math::Vector3<double>(4.0, 7.0, 6.0),
      mundy::math::Vector3<double>(4.0, 7.0, 5.0), mundy::math::Vector3<double>(4.0, 6.0, 5.0),
      mundy::math::Vector3<double>(4.0, 6.0, 4.0), mundy::math::Vector3<double>(4.0, 7.0, 4.0),
      mundy::math::Vector3<double>(5.0, 7.0, 4.0), mundy::math::Vector3<double>(5.0, 6.0, 4.0),
      mundy::math::Vector3<double>(5.0, 6.0, 5.0), mundy::math::Vector3<double>(5.0, 7.0, 5.0),
      mundy::math::Vector3<double>(6.0, 7.0, 5.0), mundy::math::Vector3<double>(6.0, 6.0, 5.0),
      mundy::math::Vector3<double>(6.0, 6.0, 4.0), mundy::math::Vector3<double>(6.0, 7.0, 4.0),
      mundy::math::Vector3<double>(7.0, 7.0, 4.0), mundy::math::Vector3<double>(7.0, 6.0, 4.0),
      mundy::math::Vector3<double>(7.0, 6.0, 5.0), mundy::math::Vector3<double>(7.0, 7.0, 5.0),
      mundy::math::Vector3<double>(7.0, 7.0, 6.0), mundy::math::Vector3<double>(7.0, 7.0, 7.0),
      mundy::math::Vector3<double>(6.0, 7.0, 7.0), mundy::math::Vector3<double>(6.0, 7.0, 6.0),
      mundy::math::Vector3<double>(6.0, 6.0, 6.0), mundy::math::Vector3<double>(6.0, 6.0, 7.0),
      mundy::math::Vector3<double>(7.0, 6.0, 7.0), mundy::math::Vector3<double>(7.0, 6.0, 6.0),
      mundy::math::Vector3<double>(7.0, 5.0, 6.0), mundy::math::Vector3<double>(7.0, 5.0, 7.0),
      mundy::math::Vector3<double>(6.0, 5.0, 7.0), mundy::math::Vector3<double>(6.0, 5.0, 6.0),
      mundy::math::Vector3<double>(6.0, 4.0, 6.0), mundy::math::Vector3<double>(6.0, 4.0, 7.0),
      mundy::math::Vector3<double>(7.0, 4.0, 7.0), mundy::math::Vector3<double>(7.0, 4.0, 6.0),
      mundy::math::Vector3<double>(7.0, 4.0, 5.0), mundy::math::Vector3<double>(7.0, 5.0, 5.0),
      mundy::math::Vector3<double>(7.0, 5.0, 4.0), mundy::math::Vector3<double>(7.0, 4.0, 4.0),
      mundy::math::Vector3<double>(6.0, 4.0, 4.0), mundy::math::Vector3<double>(6.0, 5.0, 4.0),
      mundy::math::Vector3<double>(6.0, 5.0, 5.0), mundy::math::Vector3<double>(6.0, 4.0, 5.0),
      mundy::math::Vector3<double>(5.0, 4.0, 5.0), mundy::math::Vector3<double>(5.0, 5.0, 5.0),
      mundy::math::Vector3<double>(5.0, 5.0, 4.0), mundy::math::Vector3<double>(5.0, 4.0, 4.0),
      mundy::math::Vector3<double>(4.0, 4.0, 4.0), mundy::math::Vector3<double>(4.0, 5.0, 4.0),
      mundy::math::Vector3<double>(4.0, 5.0, 5.0), mundy::math::Vector3<double>(4.0, 4.0, 5.0),
      mundy::math::Vector3<double>(4.0, 4.0, 6.0), mundy::math::Vector3<double>(5.0, 4.0, 6.0),
      mundy::math::Vector3<double>(5.0, 5.0, 6.0), mundy::math::Vector3<double>(4.0, 5.0, 6.0),
      mundy::math::Vector3<double>(4.0, 5.0, 7.0), mundy::math::Vector3<double>(5.0, 5.0, 7.0),
      mundy::math::Vector3<double>(5.0, 4.0, 7.0), mundy::math::Vector3<double>(4.0, 4.0, 7.0),
      mundy::math::Vector3<double>(4.0, 3.0, 7.0), mundy::math::Vector3<double>(5.0, 3.0, 7.0),
      mundy::math::Vector3<double>(5.0, 2.0, 7.0), mundy::math::Vector3<double>(4.0, 2.0, 7.0),
      mundy::math::Vector3<double>(4.0, 2.0, 6.0), mundy::math::Vector3<double>(5.0, 2.0, 6.0),
      mundy::math::Vector3<double>(5.0, 3.0, 6.0), mundy::math::Vector3<double>(4.0, 3.0, 6.0),
      mundy::math::Vector3<double>(4.0, 3.0, 5.0), mundy::math::Vector3<double>(4.0, 2.0, 5.0),
      mundy::math::Vector3<double>(4.0, 2.0, 4.0), mundy::math::Vector3<double>(4.0, 3.0, 4.0),
      mundy::math::Vector3<double>(5.0, 3.0, 4.0), mundy::math::Vector3<double>(5.0, 2.0, 4.0),
      mundy::math::Vector3<double>(5.0, 2.0, 5.0), mundy::math::Vector3<double>(5.0, 3.0, 5.0),
      mundy::math::Vector3<double>(6.0, 3.0, 5.0), mundy::math::Vector3<double>(6.0, 2.0, 5.0),
      mundy::math::Vector3<double>(6.0, 2.0, 4.0), mundy::math::Vector3<double>(6.0, 3.0, 4.0),
      mundy::math::Vector3<double>(7.0, 3.0, 4.0), mundy::math::Vector3<double>(7.0, 2.0, 4.0),
      mundy::math::Vector3<double>(7.0, 2.0, 5.0), mundy::math::Vector3<double>(7.0, 3.0, 5.0),
      mundy::math::Vector3<double>(7.0, 3.0, 6.0), mundy::math::Vector3<double>(7.0, 3.0, 7.0),
      mundy::math::Vector3<double>(6.0, 3.0, 7.0), mundy::math::Vector3<double>(6.0, 3.0, 6.0),
      mundy::math::Vector3<double>(6.0, 2.0, 6.0), mundy::math::Vector3<double>(6.0, 2.0, 7.0),
      mundy::math::Vector3<double>(7.0, 2.0, 7.0), mundy::math::Vector3<double>(7.0, 2.0, 6.0),
      mundy::math::Vector3<double>(7.0, 1.0, 6.0), mundy::math::Vector3<double>(7.0, 1.0, 7.0),
      mundy::math::Vector3<double>(6.0, 1.0, 7.0), mundy::math::Vector3<double>(6.0, 1.0, 6.0),
      mundy::math::Vector3<double>(6.0, 0.0, 6.0), mundy::math::Vector3<double>(6.0, 0.0, 7.0),
      mundy::math::Vector3<double>(7.0, 0.0, 7.0), mundy::math::Vector3<double>(7.0, 0.0, 6.0),
      mundy::math::Vector3<double>(7.0, 0.0, 5.0), mundy::math::Vector3<double>(7.0, 1.0, 5.0),
      mundy::math::Vector3<double>(7.0, 1.0, 4.0), mundy::math::Vector3<double>(7.0, 0.0, 4.0),
      mundy::math::Vector3<double>(6.0, 0.0, 4.0), mundy::math::Vector3<double>(6.0, 1.0, 4.0),
      mundy::math::Vector3<double>(6.0, 1.0, 5.0), mundy::math::Vector3<double>(6.0, 0.0, 5.0),
      mundy::math::Vector3<double>(5.0, 0.0, 5.0), mundy::math::Vector3<double>(5.0, 1.0, 5.0),
      mundy::math::Vector3<double>(5.0, 1.0, 4.0), mundy::math::Vector3<double>(5.0, 0.0, 4.0),
      mundy::math::Vector3<double>(4.0, 0.0, 4.0), mundy::math::Vector3<double>(4.0, 1.0, 4.0),
      mundy::math::Vector3<double>(4.0, 1.0, 5.0), mundy::math::Vector3<double>(4.0, 0.0, 5.0),
      mundy::math::Vector3<double>(4.0, 0.0, 6.0), mundy::math::Vector3<double>(5.0, 0.0, 6.0),
      mundy::math::Vector3<double>(5.0, 1.0, 6.0), mundy::math::Vector3<double>(4.0, 1.0, 6.0),
      mundy::math::Vector3<double>(4.0, 1.0, 7.0), mundy::math::Vector3<double>(5.0, 1.0, 7.0),
      mundy::math::Vector3<double>(5.0, 0.0, 7.0), mundy::math::Vector3<double>(4.0, 0.0, 7.0),
      mundy::math::Vector3<double>(3.0, 0.0, 7.0), mundy::math::Vector3<double>(3.0, 0.0, 6.0),
      mundy::math::Vector3<double>(2.0, 0.0, 6.0), mundy::math::Vector3<double>(2.0, 0.0, 7.0),
      mundy::math::Vector3<double>(2.0, 1.0, 7.0), mundy::math::Vector3<double>(2.0, 1.0, 6.0),
      mundy::math::Vector3<double>(3.0, 1.0, 6.0), mundy::math::Vector3<double>(3.0, 1.0, 7.0),
      mundy::math::Vector3<double>(3.0, 2.0, 7.0), mundy::math::Vector3<double>(2.0, 2.0, 7.0),
      mundy::math::Vector3<double>(2.0, 3.0, 7.0), mundy::math::Vector3<double>(3.0, 3.0, 7.0),
      mundy::math::Vector3<double>(3.0, 3.0, 6.0), mundy::math::Vector3<double>(2.0, 3.0, 6.0),
      mundy::math::Vector3<double>(2.0, 2.0, 6.0), mundy::math::Vector3<double>(3.0, 2.0, 6.0),
      mundy::math::Vector3<double>(3.0, 2.0, 5.0), mundy::math::Vector3<double>(2.0, 2.0, 5.0),
      mundy::math::Vector3<double>(2.0, 3.0, 5.0), mundy::math::Vector3<double>(3.0, 3.0, 5.0),
      mundy::math::Vector3<double>(3.0, 3.0, 4.0), mundy::math::Vector3<double>(2.0, 3.0, 4.0),
      mundy::math::Vector3<double>(2.0, 2.0, 4.0), mundy::math::Vector3<double>(3.0, 2.0, 4.0),
      mundy::math::Vector3<double>(3.0, 1.0, 4.0), mundy::math::Vector3<double>(3.0, 0.0, 4.0),
      mundy::math::Vector3<double>(3.0, 0.0, 5.0), mundy::math::Vector3<double>(3.0, 1.0, 5.0),
      mundy::math::Vector3<double>(2.0, 1.0, 5.0), mundy::math::Vector3<double>(2.0, 0.0, 5.0),
      mundy::math::Vector3<double>(2.0, 0.0, 4.0), mundy::math::Vector3<double>(2.0, 1.0, 4.0),
      mundy::math::Vector3<double>(1.0, 1.0, 4.0), mundy::math::Vector3<double>(1.0, 0.0, 4.0),
      mundy::math::Vector3<double>(1.0, 0.0, 5.0), mundy::math::Vector3<double>(1.0, 1.0, 5.0),
      mundy::math::Vector3<double>(0.0, 1.0, 5.0), mundy::math::Vector3<double>(0.0, 0.0, 5.0),
      mundy::math::Vector3<double>(0.0, 0.0, 4.0), mundy::math::Vector3<double>(0.0, 1.0, 4.0),
      mundy::math::Vector3<double>(0.0, 2.0, 4.0), mundy::math::Vector3<double>(1.0, 2.0, 4.0),
      mundy::math::Vector3<double>(1.0, 3.0, 4.0), mundy::math::Vector3<double>(0.0, 3.0, 4.0),
      mundy::math::Vector3<double>(0.0, 3.0, 5.0), mundy::math::Vector3<double>(1.0, 3.0, 5.0),
      mundy::math::Vector3<double>(1.0, 2.0, 5.0), mundy::math::Vector3<double>(0.0, 2.0, 5.0),
      mundy::math::Vector3<double>(0.0, 2.0, 6.0), mundy::math::Vector3<double>(1.0, 2.0, 6.0),
      mundy::math::Vector3<double>(1.0, 3.0, 6.0), mundy::math::Vector3<double>(0.0, 3.0, 6.0),
      mundy::math::Vector3<double>(0.0, 3.0, 7.0), mundy::math::Vector3<double>(1.0, 3.0, 7.0),
      mundy::math::Vector3<double>(1.0, 2.0, 7.0), mundy::math::Vector3<double>(0.0, 2.0, 7.0),
      mundy::math::Vector3<double>(0.0, 1.0, 7.0), mundy::math::Vector3<double>(0.0, 1.0, 6.0),
      mundy::math::Vector3<double>(1.0, 1.0, 6.0), mundy::math::Vector3<double>(1.0, 1.0, 7.0),
      mundy::math::Vector3<double>(1.0, 0.0, 7.0), mundy::math::Vector3<double>(1.0, 0.0, 6.0),
      mundy::math::Vector3<double>(0.0, 0.0, 6.0), mundy::math::Vector3<double>(0.0, 0.0, 7.0)};

  for (size_t i = 0; i < expected_position_array.size(); ++i) {
    ASSERT_TRUE(is_close(position_array[i], expected_position_array[i]));
  }
}

TEST(Hilbert3D, DirectorLinks8) {
  size_t num_links = 8;

  // Use structureed bindings to make our life easier
  auto [position_array, directors] = mundy::math::create_hilbert_positions_and_directors(num_links);

  std::vector<mundy::math::Vector3<double>> expected_position_array = {
      mundy::math::Vector3<double>(0.0, 0.0, 0.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(1.0, 1.0, 0.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 1.0, 1.0), mundy::math::Vector3<double>(1.0, 1.0, 1.0),
      mundy::math::Vector3<double>(1.0, 0.0, 1.0), mundy::math::Vector3<double>(0.0, 0.0, 1.0)};
  std::vector<mundy::math::Vector3<double>> expected_directors = {
      mundy::math::Vector3<double>(1.0, 0.0, 0.0),  mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(-1.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(1.0, 0.0, 0.0),  mundy::math::Vector3<double>(0.0, -1.0, 0.0),
      mundy::math::Vector3<double>(-1.0, 0.0, 0.0)};
  for (size_t i = 0; i < expected_position_array.size(); ++i) {
    ASSERT_TRUE(is_close(position_array[i], expected_position_array[i]));
  }
  for (size_t i = 0; i < expected_directors.size(); ++i) {
    ASSERT_TRUE(is_close(directors[i], expected_directors[i]));
  }
}

TEST(Hilbert3D, DirectorLinks9) {
  size_t num_links = 9;

  // Use structureed bindings to make our life easier
  auto [position_array, directors] = mundy::math::create_hilbert_positions_and_directors(num_links);

  std::vector<mundy::math::Vector3<double>> expected_position_array = {
      mundy::math::Vector3<double>(0.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 1.0, 1.0), mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(1.0, 0.0, 1.0), mundy::math::Vector3<double>(1.0, 1.0, 1.0),
      mundy::math::Vector3<double>(1.0, 1.0, 0.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(2.0, 0.0, 0.0), mundy::math::Vector3<double>(2.0, 0.0, 1.0),
      mundy::math::Vector3<double>(3.0, 0.0, 1.0), mundy::math::Vector3<double>(3.0, 0.0, 0.0),
      mundy::math::Vector3<double>(3.0, 1.0, 0.0), mundy::math::Vector3<double>(3.0, 1.0, 1.0),
      mundy::math::Vector3<double>(2.0, 1.0, 1.0), mundy::math::Vector3<double>(2.0, 1.0, 0.0),
      mundy::math::Vector3<double>(2.0, 2.0, 0.0), mundy::math::Vector3<double>(2.0, 2.0, 1.0),
      mundy::math::Vector3<double>(3.0, 2.0, 1.0), mundy::math::Vector3<double>(3.0, 2.0, 0.0),
      mundy::math::Vector3<double>(3.0, 3.0, 0.0), mundy::math::Vector3<double>(3.0, 3.0, 1.0),
      mundy::math::Vector3<double>(2.0, 3.0, 1.0), mundy::math::Vector3<double>(2.0, 3.0, 0.0),
      mundy::math::Vector3<double>(1.0, 3.0, 0.0), mundy::math::Vector3<double>(0.0, 3.0, 0.0),
      mundy::math::Vector3<double>(0.0, 2.0, 0.0), mundy::math::Vector3<double>(1.0, 2.0, 0.0),
      mundy::math::Vector3<double>(1.0, 2.0, 1.0), mundy::math::Vector3<double>(0.0, 2.0, 1.0),
      mundy::math::Vector3<double>(0.0, 3.0, 1.0), mundy::math::Vector3<double>(1.0, 3.0, 1.0),
      mundy::math::Vector3<double>(1.0, 3.0, 2.0), mundy::math::Vector3<double>(0.0, 3.0, 2.0),
      mundy::math::Vector3<double>(0.0, 2.0, 2.0), mundy::math::Vector3<double>(1.0, 2.0, 2.0),
      mundy::math::Vector3<double>(1.0, 2.0, 3.0), mundy::math::Vector3<double>(0.0, 2.0, 3.0),
      mundy::math::Vector3<double>(0.0, 3.0, 3.0), mundy::math::Vector3<double>(1.0, 3.0, 3.0),
      mundy::math::Vector3<double>(2.0, 3.0, 3.0), mundy::math::Vector3<double>(2.0, 3.0, 2.0),
      mundy::math::Vector3<double>(3.0, 3.0, 2.0), mundy::math::Vector3<double>(3.0, 3.0, 3.0),
      mundy::math::Vector3<double>(3.0, 2.0, 3.0), mundy::math::Vector3<double>(3.0, 2.0, 2.0),
      mundy::math::Vector3<double>(2.0, 2.0, 2.0), mundy::math::Vector3<double>(2.0, 2.0, 3.0),
      mundy::math::Vector3<double>(2.0, 1.0, 3.0), mundy::math::Vector3<double>(2.0, 1.0, 2.0),
      mundy::math::Vector3<double>(3.0, 1.0, 2.0), mundy::math::Vector3<double>(3.0, 1.0, 3.0),
      mundy::math::Vector3<double>(3.0, 0.0, 3.0), mundy::math::Vector3<double>(3.0, 0.0, 2.0),
      mundy::math::Vector3<double>(2.0, 0.0, 2.0), mundy::math::Vector3<double>(2.0, 0.0, 3.0),
      mundy::math::Vector3<double>(1.0, 0.0, 3.0), mundy::math::Vector3<double>(1.0, 1.0, 3.0),
      mundy::math::Vector3<double>(1.0, 1.0, 2.0), mundy::math::Vector3<double>(1.0, 0.0, 2.0),
      mundy::math::Vector3<double>(0.0, 0.0, 2.0), mundy::math::Vector3<double>(0.0, 1.0, 2.0),
      mundy::math::Vector3<double>(0.0, 1.0, 3.0), mundy::math::Vector3<double>(0.0, 0.0, 3.0)};
  std::vector<mundy::math::Vector3<double>> expected_directors = {
      mundy::math::Vector3<double>(0.0, 1.0, 0.0),  mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(0.0, -1.0, 0.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 1.0, 0.0),  mundy::math::Vector3<double>(0.0, 0.0, -1.0),
      mundy::math::Vector3<double>(0.0, -1.0, 0.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(-1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(-1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(-1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(-1.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, -1.0, 0.0),
      mundy::math::Vector3<double>(1.0, 0.0, 0.0),  mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(-1.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(1.0, 0.0, 0.0),  mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(-1.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, -1.0, 0.0),
      mundy::math::Vector3<double>(1.0, 0.0, 0.0),  mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(-1.0, 0.0, 0.0), mundy::math::Vector3<double>(0.0, 1.0, 0.0),
      mundy::math::Vector3<double>(1.0, 0.0, 0.0),  mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(0.0, -1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(-1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(0.0, -1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(0.0, -1.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, -1.0), mundy::math::Vector3<double>(-1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 0.0, 1.0),  mundy::math::Vector3<double>(-1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 1.0, 0.0),  mundy::math::Vector3<double>(0.0, 0.0, -1.0),
      mundy::math::Vector3<double>(0.0, -1.0, 0.0), mundy::math::Vector3<double>(-1.0, 0.0, 0.0),
      mundy::math::Vector3<double>(0.0, 1.0, 0.0),  mundy::math::Vector3<double>(0.0, 0.0, 1.0),
      mundy::math::Vector3<double>(0.0, -1.0, 0.0)};
  for (size_t i = 0; i < expected_position_array.size(); ++i) {
    ASSERT_TRUE(is_close(position_array[i], expected_position_array[i]));
  }
  for (size_t i = 0; i < expected_directors.size(); ++i) {
    ASSERT_TRUE(is_close(directors[i], expected_directors[i]));
  }
}
//@}

}  // namespace

}  // namespace math

}  // namespace mundy
