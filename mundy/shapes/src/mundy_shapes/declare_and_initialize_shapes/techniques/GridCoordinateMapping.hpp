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

#ifndef MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_GRIDCOORDINATEMAPPING_HPP_
#define MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_GRIDCOORDINATEMAPPING_HPP_

/// \file GridCoordinateMapping.hpp
/// \brief An interface and some concrete implementations for mapping grid indices to coordinates.

// C++ core libs
#include <array>  // for std::array
#include <cmath>  // for std::sin, std::cos, std::sqrt

namespace mundy {

namespace shapes {

namespace declare_and_initialize_shapes {

namespace techniques {

/// \brief An interface for mapping grid indices to coordinates.
class GridCoordinateMapping {
 public:
  /// \brief Virtual destructor.
  virtual ~GridCoordinateMapping() = default;

  /// \brief Get the grid coordinate corresponding to a given grid index.
  /// \param grid_index The grid index.
  /// \return The grid coordinate.
  virtual std::array<double, 3> get_grid_coordinate(const std::array<size_t, 3> &grid_index) const = 0;
};  // class GridCoordinateMapping

/// \brief The identity map, which maps grid indices to themselves.
class IdentityMap : public GridCoordinateMapping {
 public:
  /// \brief Get the grid coordinate corresponding to a given grid index.
  /// \param grid_index The grid index.
  /// \return The grid coordinate.
  std::array<double, 3> get_grid_coordinate(const std::array<size_t, 3> &grid_index) const override {
    return {static_cast<double>(grid_index[0]), static_cast<double>(grid_index[1]), static_cast<double>(grid_index[2])};
  }
};  // class IdentityMap

/// \brief Scaled grid coordinate mapping.
/// \details Given a grid index $(i,j,k)$, we map it to the coordinate
/// $(x,y,z)$\in[min_x,max_x]\times[min_x,max_x]\times[min_z,max_z]$ using the formula $$x = \frac{i}{N_x} (max_x -
/// min_x) + min_x, \quad y = \frac{j}{N_y} (max_y - min_y) + min_y, \quad z = \frac{k}{N_z} (max_z - min_z) + min_z$$
class ScaledGridCoordinateMapping : public GridCoordinateMapping {
 public:
  /// Constructor
  ScaledGridCoordinateMapping(const size_t num_grid_points_x, const size_t num_grid_points_y,
                              const size_t num_grid_points_z, const double min_x = 0, const double max_x = 1,
                              const double min_y = 0, const double max_y = 1, const double min_z = 0,
                              const double max_z = 1)
      : num_grid_points_x_(num_grid_points_x),
        num_grid_points_y_(num_grid_points_y),
        num_grid_points_z_(num_grid_points_z),
        min_x_(min_x),
        max_x_(max_x),
        min_y_(min_y),
        max_y_(max_y),
        min_z_(min_z),
        max_z_(max_z) {
  }

  /// \brief Get the grid coordinate corresponding to a given grid index.
  /// \param grid_index The grid index.
  /// \return The grid coordinate.
  std::array<double, 3> get_grid_coordinate(const std::array<size_t, 3> &grid_index) const override {
    return {static_cast<double>(grid_index[0]) / static_cast<double>(num_grid_points_x_) * (max_x_ - min_x_) + min_x_,
            static_cast<double>(grid_index[1]) / static_cast<double>(num_grid_points_y_) * (max_y_ - min_y_) + min_y_,
            static_cast<double>(grid_index[2]) / static_cast<double>(num_grid_points_z_) * (max_z_ - min_z_) + min_z_};
  }

 private:
  size_t num_grid_points_x_;
  size_t num_grid_points_y_;
  size_t num_grid_points_z_;
  double min_x_;
  double max_x_;
  double min_y_;
  double max_y_;
  double min_z_;
  double max_z_;
};  // class ScaledGridCoordinateMapping

/// \brief Levi's function in 3D applied to a 2D grid with Nx x Ny points. We ignore the k-th index.
/// @details Given a grid index $(i,j)$, we map it to the coordinate $(x,y)$\in[0,2]\times[0,2]$ using the formula
/// $$x = \frac{2i}{N_x}, \quad y = \frac{2j}{N_y}$$. We then compute their z coordinate using
/// $$z(x,y)= \sin^2(3 \pi x) + (x - 1)^2 \left(1 + \sin^2(3 \pi y) \right) + (y - 1)^2 \left(1 + \sin^2(2 \pi x)
/// \right)$$
/// We then scale and shift the coordinates to fall within the range
/// [min_x,max_x]\times[min_x,max_x]\times[min_z,max_z].
class LevisFunction2DTo3D : public GridCoordinateMapping {
 public:
  /// Constructor
  LevisFunction2DTo3D(const size_t num_grid_points_x, const size_t num_grid_points_y, const double min_x = 0,
                      const double max_x = 1, const double min_y = 0, const double max_y = 1, const double min_z = 0,
                      const double max_z = 1)
      : num_grid_points_x_(num_grid_points_x),
        num_grid_points_y_(num_grid_points_y),
        min_x_(min_x),
        max_x_(max_x),
        min_y_(min_y),
        max_y_(max_y),
        min_z_(min_z),
        max_z_(max_z) {
  }

  /// \brief Get the grid coordinate corresponding to a given grid index.
  /// \param grid_index The grid index.
  /// \return The grid coordinate.
  std::array<double, 3> get_grid_coordinate(const std::array<size_t, 3> &grid_index) const override {
    const double x = 2.0 * static_cast<double>(grid_index[0]) / static_cast<double>(num_grid_points_x_);
    const double y = 2.0 * static_cast<double>(grid_index[1]) / static_cast<double>(num_grid_points_y_);
    const double z = std::sin(3.0 * M_PI * x) * std::sin(3.0 * M_PI * x) +
                     (x - 1.0) * (x - 1.0) * (1.0 + std::sin(3.0 * M_PI * y) * std::sin(3.0 * M_PI * y)) +
                     (y - 1.0) * (y - 1.0) * (1.0 + std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * x));
    return {x / 2.0 * (max_x_ - min_x_) + min_x_, y / 2.0 * (max_y_ - min_y_) + min_y_,
            z / 4.0 * (max_z_ - min_z_) + min_z_};
  }

 private:
  size_t num_grid_points_x_;
  size_t num_grid_points_y_;
  double min_x_;
  double max_x_;
  double min_y_;
  double max_y_;
  double min_z_;
  double max_z_;
};  // class LevisFunction2DTo3D

}  // namespace techniques

}  // namespace declare_and_initialize_shapes

}  // namespace shapes

}  // namespace mundy

#endif  // MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_GRIDCOORDINATEMAPPING_HPP_