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
#include <gtest/gtest.h>

// Mundy
#include <mundy_geom/distance.hpp>
#include <mundy_geom/primitives.hpp>
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/Tolerance.hpp>
#include <mundy_math/Vector3.hpp>

namespace mundy {

namespace {

TEST(DistanceViews, AcceptsViewBackedPointLineSegmentAndSphere) {
  const double tol = mundy::get_relaxed_zero_tolerance<double>();

  double origin_data[3] = {0.0, 0.0, 0.0};
  double point_x_data[3] = {1.0, 0.0, 0.0};
  double point_above_segment_data[3] = {0.5, 1.0, 0.0};
  double dir_y_data[3] = {0.0, 1.0, 0.0};
  double dir_z_data[3] = {0.0, 0.0, 1.0};
  double line2_center_data[3] = {1.0, 0.0, 0.0};
  double segment_end_data[3] = {1.0, 0.0, 0.0};
  double segment2_start_data[3] = {0.0, 2.0, 0.0};
  double segment2_end_data[3] = {1.0, 2.0, 0.0};
  double sphere_center_data[3] = {3.0, 0.0, 0.0};
  double sphere2_center_data[3] = {5.0, 0.0, 0.0};

  auto origin = get_vector3_view<double>(origin_data);
  auto point_x = get_vector3_view<double>(point_x_data);
  auto point_above_segment = get_vector3_view<double>(point_above_segment_data);
  auto dir_y = get_vector3_view<double>(dir_y_data);
  auto dir_z = get_vector3_view<double>(dir_z_data);
  auto line2_center = get_vector3_view<double>(line2_center_data);
  auto segment_end = get_vector3_view<double>(segment_end_data);
  auto segment2_start = get_vector3_view<double>(segment2_start_data);
  auto segment2_end = get_vector3_view<double>(segment2_end_data);
  auto sphere_center = get_vector3_view<double>(sphere_center_data);
  auto sphere2_center = get_vector3_view<double>(sphere2_center_data);

  Line<double, decltype(origin)> line_y{origin, dir_y};
  Line<double, decltype(line2_center)> line_z{line2_center, dir_z};
  LineSegment<double, decltype(origin), decltype(segment_end)> segment{origin, segment_end};
  LineSegment<double, decltype(segment2_start), decltype(segment2_end)> segment2{segment2_start, segment2_end};
  Sphere<double, decltype(sphere_center)> sphere{sphere_center, 0.5};
  Sphere<double, decltype(sphere2_center)> sphere2{sphere2_center, 0.25};

  EXPECT_NEAR(distance(origin, point_x), 1.0, tol);
  EXPECT_NEAR(distance(point_x, line_y), 1.0, tol);
  EXPECT_NEAR(distance(point_above_segment, segment), 1.0, tol);
  EXPECT_NEAR(distance(line_y, line_z), 1.0, tol);
  EXPECT_NEAR(distance(segment, segment2), 2.0, tol);
  EXPECT_NEAR(distance(point_x, sphere), 1.5, tol);
  EXPECT_NEAR(distance(line_y, sphere), 2.5, tol);
  EXPECT_NEAR(distance(segment, sphere), 1.5, tol);
  EXPECT_NEAR(distance(sphere, sphere2), 1.25, tol);

  Point<double> closest;
  double arch_length = 0.0;
  Vector3d sep;
  EXPECT_NEAR(distance(point_above_segment, segment, closest, arch_length, sep), 1.0, tol);
  EXPECT_NEAR(arch_length, 0.5, tol);

  Point<double> closest1;
  Point<double> closest2;
  double arch_length1 = 0.0;
  double arch_length2 = 0.0;
  EXPECT_NEAR(distance(segment, segment2, closest1, closest2, arch_length1, arch_length2, sep), 2.0, tol);

  FreeSpaceMetric free_metric;
  const auto free_sep = free_metric(origin, point_x);
  EXPECT_NEAR(free_sep[0], 1.0, tol);

  PeriodicScaledSpaceMetric<double> periodic_metric{Vector3d{10.0, 10.0, 10.0}};
  double near_right_data[3] = {9.8, 0.0, 0.0};
  double near_left_data[3] = {0.2, 0.0, 0.0};
  auto near_right = get_vector3_view<double>(near_right_data);
  auto near_left = get_vector3_view<double>(near_left_data);
  const auto periodic_sep = periodic_metric(near_right, near_left);
  EXPECT_NEAR(periodic_sep[0], 0.4, tol);
}

TEST(DistanceViews, AcceptsViewBackedEllipsoidAndCircle) {
  const double tol = 1.0e-4;

  double point_data[3] = {2.0, 0.0, 0.0};
  double center0_data[3] = {0.0, 0.0, 0.0};
  double center1_data[3] = {3.0, 0.0, 0.0};
  double radii0_data[3] = {1.0, 1.0, 1.0};
  double radii1_data[3] = {0.5, 0.5, 0.5};
  double quat0_data[4] = {0.0, 0.0, 0.0, 1.0};
  double quat1_data[4] = {0.0, 0.0, 0.0, 1.0};

  auto point = get_vector3_view<double>(point_data);
  auto center0 = get_vector3_view<double>(center0_data);
  auto center1 = get_vector3_view<double>(center1_data);
  auto radii0 = get_vector3_view<double>(radii0_data);
  auto radii1 = get_vector3_view<double>(radii1_data);
  auto quat0 = get_quaternion_view<double>(quat0_data);
  auto quat1 = get_quaternion_view<double>(quat1_data);

  Ellipsoid<double, decltype(center0), decltype(quat0)> ellipsoid0{center0, quat0, radii0};
  Ellipsoid<double, decltype(center1), decltype(quat1)> ellipsoid1{center1, quat1, radii1};

  EXPECT_NEAR(distance(point, ellipsoid0), 1.0, tol);
  EXPECT_NEAR(distance(ellipsoid0, ellipsoid1), 1.5, tol);

  Circle3D<double, decltype(center0), decltype(quat0)> circle0{center0, quat0, 1.0};
  Circle3D<double, decltype(center1), decltype(quat1)> circle1{center1, quat1, 1.0};
  static_cast<void>(distance(circle0, circle1));
}

}  // namespace

}  // namespace mundy
