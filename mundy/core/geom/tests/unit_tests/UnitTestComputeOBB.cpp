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
#include <gtest/gtest.h>  // for TEST, EXPECT_*

// C++ core
#include <cmath>  // for std::sqrt

// Mundy
#include <mundy_geom/compute_obb.hpp>  // for mundy::compute_obb
#include <mundy_geom/primitives.hpp>   // for mundy::OBB, mundy::Sphere, mundy::Ellipsoid, mundy::is_close
#include <mundy_math/Quaternion.hpp>   // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>    // for mundy::get_relaxed_zero_tolerance
#include <mundy_math/Vector3.hpp>      // for mundy::Vector3, mundy::norm

namespace mundy {

namespace {

// A point is a degenerate box: on the point, zero half-extents, identity orientation.
TEST(ComputeOBB, Point) {
  const auto obb = compute_obb(Point<double>{1.0, -2.0, 3.0});
  const OBB<double> expected{Point<double>{1.0, -2.0, 3.0}, Quaternion<double>::identity(), 0.0, 0.0, 0.0};
  EXPECT_TRUE(is_close(obb, expected, get_relaxed_zero_tolerance<double>()));
}

// A sphere's tightest box is the axis-aligned cube of side 2r.
TEST(ComputeOBB, Sphere) {
  const auto obb = compute_obb(Sphere<double>{Point<double>{1.0, -2.0, 3.0}, 4.0});
  const OBB<double> expected{Point<double>{1.0, -2.0, 3.0}, Quaternion<double>::identity(), 4.0, 4.0, 4.0};
  EXPECT_TRUE(is_close(obb, expected, get_relaxed_zero_tolerance<double>()));
}

// An AABB maps to an axis-aligned OBB centered at the box center with half-extents = half the widths.
TEST(ComputeOBB, AABB) {
  const auto obb = compute_obb(AABB<double>{1.0, -2.0, 3.0, 5.0, 4.0, 9.0});
  const OBB<double> expected{Point<double>{3.0, 1.0, 6.0}, Quaternion<double>::identity(), 2.0, 3.0, 3.0};
  EXPECT_TRUE(is_close(obb, expected, get_relaxed_zero_tolerance<double>()));
}

// Ellipsoid principal axes are the box axes: orientation forwarded (non-identity here), half-extents = radii.
TEST(ComputeOBB, Ellipsoid) {
  const double s = 1.0 / std::sqrt(2.0);
  const Quaternion<double> orient{s, s, 0.0, 0.0};  // 90 deg about x
  const auto obb =
      compute_obb(Ellipsoid<double>{Point<double>{1.0, -2.0, 3.0}, orient, Vector3<double>{4.0, 5.0, 6.0}});
  const OBB<double> expected{Point<double>{1.0, -2.0, 3.0}, orient, 4.0, 5.0, 6.0};
  EXPECT_TRUE(is_close(obb, expected, get_relaxed_zero_tolerance<double>()));
}

// Spherocylinder(center, orientation, radius, length): radial half-extents = radius, long half-extent =
// length/2 + radius, orientation forwarded.
TEST(ComputeOBB, Spherocylinder) {
  const double s = 1.0 / std::sqrt(2.0);
  const Quaternion<double> orient{s, s, 0.0, 0.0};  // 90 deg about x
  const auto obb = compute_obb(Spherocylinder<double>{Point<double>{1.0, -2.0, 3.0}, orient, 2.0, 4.0});
  const OBB<double> expected{Point<double>{1.0, -2.0, 3.0}, orient, 2.0, 2.0, 4.0};
  EXPECT_TRUE(is_close(obb, expected, get_relaxed_zero_tolerance<double>()));
}

// A degenerate spherocylinder segment (start == end) falls back to the identity orientation and a sphere-like box
// of half-extent = radius.
TEST(ComputeOBB, SpherocylinderSegmentDegenerate) {
  const auto obb =
      compute_obb(SpherocylinderSegment<double>{Point<double>{1.0, -2.0, 3.0}, Point<double>{1.0, -2.0, 3.0}, 2.0});
  const OBB<double> expected{Point<double>{1.0, -2.0, 3.0}, Quaternion<double>::identity(), 2.0, 2.0, 2.0};
  EXPECT_TRUE(is_close(obb, expected, get_relaxed_zero_tolerance<double>()));
}

// Spherocylinder segment along +x: center at midpoint, long half-extent = |e - s|/2 + radius, radial = radius,
// long (local-z) axis points start -> end. Radial axes are a gauge freedom, so only the long axis is asserted.
TEST(ComputeOBB, SpherocylinderSegmentAlongX) {
  const double tol = get_relaxed_zero_tolerance<double>();
  const auto obb =
      compute_obb(SpherocylinderSegment<double>{Point<double>{0.0, 0.0, 0.0}, Point<double>{4.0, 0.0, 0.0}, 1.0});
  EXPECT_LT(norm(obb.center() - Vector3<double>{2.0, 0.0, 0.0}), tol);
  EXPECT_LT(norm(obb.half_extents() - Vector3<double>{1.0, 1.0, 3.0}), tol);
  EXPECT_LT(norm(obb.orientation() * Vector3<double>{0.0, 0.0, 1.0} - Vector3<double>{1.0, 0.0, 0.0}), tol);
}

// A line segment has zero radius: radial half-extents vanish, long half-extent = |e - s|/2, long axis from start
// to end.
TEST(ComputeOBB, LineSegmentAlongY) {
  const double tol = get_relaxed_zero_tolerance<double>();
  const auto obb = compute_obb(LineSegment<double>{Point<double>{1.0, 0.0, 0.0}, Point<double>{1.0, 4.0, 0.0}});
  EXPECT_LT(norm(obb.center() - Vector3<double>{1.0, 2.0, 0.0}), tol);
  EXPECT_LT(norm(obb.half_extents() - Vector3<double>{0.0, 0.0, 2.0}), tol);
  EXPECT_LT(norm(obb.orientation() * Vector3<double>{0.0, 0.0, 1.0} - Vector3<double>{0.0, 1.0, 0.0}), tol);
}

}  // namespace

}  // namespace mundy
