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

#ifndef MUNDY_MATH_HILBERT_HPP_
#define MUNDY_MATH_HILBERT_HPP_

// External

// C++ core
#include <cmath>
#include <cstddef>
#include <vector>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace math {

/// \brief Functions for generating and using 3D Hilbert space-filling curves.
///
/// This defines functions for converting between 3D coordinates and a 1D index along the curve.

/// \brief Recursive function to create a 3D hilbert curve given a number of lattice points per side
/// \param[in] s The index of the element.
///
/// The non-reference passing of the current_position, dr1, dr2, and dr3 is intentional. This is because the
/// we do some math when passing the values to the next recursive call.
size_t hilbert_3d(size_t s, size_t i, std::vector<mundy::math::Vector3d> &position_array,
                  mundy::math::Vector3d current_position, mundy::math::Vector3d dr1, mundy::math::Vector3d dr2,
                  mundy::math::Vector3d dr3) {
  // Check to make sure we've been passed in a power of two
  MUNDY_THROW_REQUIRE(s != 0 && (s & (s - 1)) == 0, std::logic_error, "s must be a power of 2");
  if (s == 1) {
    position_array[i] = current_position;
    return i + 1;
  }

  size_t snew = static_cast<size_t>(std::floor(s / 2));
  MUNDY_THROW_REQUIRE(snew > 0, std::logic_error, "snew must be greater than 0");

  mundy::math::Vector3 current_position_new = current_position;
  mundy::math::Vector3 dr1_new = dr1;
  mundy::math::Vector3 dr2_new = dr2;
  mundy::math::Vector3 dr3_new = dr3;

  for (auto &dr : {dr1_new, dr2_new, dr3_new}) {
    mundy::math::Vector3d dr_stencil = {dr[0] < 0.0 ? 1.0 : 0.0, dr[1] < 0.0 ? 1.0 : 0.0, dr[2] < 0.0 ? 1.0 : 0.0};
    current_position_new -= static_cast<double>(snew) * elementwise_mul(dr_stencil, dr);
  }

  i = hilbert_3d(snew, i, position_array, current_position_new, dr2_new, dr3_new, dr1_new);
  i = hilbert_3d(snew, i, position_array, current_position_new + snew * dr1_new, dr3_new, dr1_new, dr2_new);
  i = hilbert_3d(snew, i, position_array, current_position_new + snew * (dr1_new + dr2_new), dr3_new, dr1_new, dr2_new);
  i = hilbert_3d(snew, i, position_array, current_position_new + snew * dr2_new, -1.0 * dr1_new, -1.0 * dr2_new,
                 dr3_new);
  i = hilbert_3d(snew, i, position_array, current_position_new + snew * (dr2_new + dr3_new), -1.0 * dr1_new,
                 -1.0 * dr2_new, dr3_new);
  i = hilbert_3d(snew, i, position_array, current_position_new + snew * (dr1_new + dr2_new + dr3_new), -1.0 * dr3_new,
                 dr1_new, -1.0 * dr2_new);
  i = hilbert_3d(snew, i, position_array, current_position_new + snew * (dr1_new + dr3_new), -1.0 * dr3_new, dr1_new,
                 -1.0 * dr2_new);
  i = hilbert_3d(snew, i, position_array, current_position_new + snew * dr3_new, dr2_new, -1.0 * dr3_new,
                 -1.0 * dr1_new);

  return i;
}

/// \brief Create a 3D Hilbert curve with a given number of links.
std::tuple<std::vector<mundy::math::Vector3d>, std::vector<mundy::math::Vector3d>>
create_hilbert_positions_and_directors(size_t num_points,
                                       mundy::math::Vector3d orientation = mundy::math::Vector3d(1.0, 0.0, 0.0),
                                       double side_length = 1.0) {
  MUNDY_THROW_ASSERT(num_points > 0, std::logic_error, "num_points must be greater than 0");
  MUNDY_THROW_ASSERT(side_length > 0.0, std::logic_error, "side_length must be greater than 0");

  size_t num_side_points = 2;
  while (num_side_points * num_side_points * num_side_points < num_points) {
    num_side_points *= 2;
  }
  size_t ind = 0;
  // Create a vector of 3D vectors to store the positions, and the default position
  std::vector<mundy::math::Vector3d> position_array(num_side_points * num_side_points * num_side_points);
  mundy::math::Vector3d current_position(0.0, 0.0, 0.0);
  // Create a orthogonal right handed coordinate system for the cell vectors
  // TODO(palmerb4): This is not valid for all orientations :(
  mundy::math::Vector3d zhat(0.0, 0.0, 1.0);
  mundy::math::Vector3d dr1_hat = orientation;
  dr1_hat = dr1_hat / two_norm(dr1_hat);
  mundy::math::Vector3d dr2_hat = cross(zhat, dr1_hat);
  dr2_hat = dr2_hat / two_norm(dr2_hat);
  mundy::math::Vector3d dr3_hat = cross(dr1_hat, dr2_hat);
  dr3_hat = dr3_hat / two_norm(dr3_hat);

  mundy::math::Vector3d dr1 = side_length * dr1_hat;
  mundy::math::Vector3d dr2 = side_length * dr2_hat;
  mundy::math::Vector3d dr3 = side_length * dr3_hat;

  ind = hilbert_3d(num_side_points, ind, position_array, current_position, dr1, dr2, dr3);

  // Now create the directors array
  std::vector<mundy::math::Vector3d> directors(position_array.size() - 1);
  for (size_t i = 0; i < directors.size(); i++) {
    directors[i] = position_array[(i + 1) % position_array.size()] - position_array[i];
    directors[i] = directors[i] / two_norm(directors[i]);
  }

  return std::make_tuple(position_array, directors);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_HILBERT_HPP_
