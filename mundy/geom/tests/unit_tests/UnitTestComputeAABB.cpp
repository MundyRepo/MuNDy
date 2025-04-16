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
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

// C++ core
#include <algorithm>   // for std::max
#include <concepts>    // for std::convertible_to
#include <functional>  // for std::hash
#include <string>      // for std::string

// Trilinos includes
#include <Kokkos_Core.hpp>  // for Kokkos::numbers::pi

// Mundy
#include <mundy_geom/compute_aabb.hpp>  // for mundy::geom::compute_aabb
#include <mundy_geom/primitives.hpp>    // for mundy::geom::Ellipsoid...
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance

namespace mundy {

namespace geom {

namespace {

//   - Point
//   - LineSegment
//   - Sphere
//   - Ellipsoid
//   - Spherocylinder
//   - SpherocylinderSegment

// A custom macro that wraps GTEST for checking if two AABBs are nearly equal
#define ASSERT_NEAR_AABB(expected, actual, tol)                                   \
  ASSERT_NEAR(mundy::math::norm(expected.min_corner() - actual.min_corner()) +    \
                  mundy::math::norm(expected.max_corner() - actual.max_corner()), \
              0.0, tol)

/// \brief Get the quaternion corresponding to a 90 deg rotation about the x-axis
template <typename T>
mundy::math::Quaternion<T> get_quaternion_x_90() {
  return mundy::math::Quaternion<T>(static_cast<T>(1.0 / std::sqrt(2.0)), static_cast<T>(1.0 / std::sqrt(2.0)),
                                    static_cast<T>(0.0), static_cast<T>(0.0));
}

template <typename Scalar>
struct PointTestCase {
  std::string name;
  Point<Scalar> point;
  AABB<Scalar> expected_aabb;
  void check() const {
    const auto actual_aabb = compute_aabb(point);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR_AABB(expected_aabb, actual_aabb, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct LineSegmentTestCase {
  std::string name;
  LineSegment<Scalar> segment;
  AABB<Scalar> expected_aabb;
  void check() const {
    const auto actual_aabb = compute_aabb(segment);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR_AABB(expected_aabb, actual_aabb, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct SphereTestCase {
  std::string name;
  Sphere<Scalar> sphere;
  AABB<Scalar> expected_aabb;
  void check() const {
    const auto actual_aabb = compute_aabb(sphere);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR_AABB(expected_aabb, actual_aabb, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct EllipsoidTestCase {
  std::string name;
  Ellipsoid<Scalar> ellipsoid;
  AABB<Scalar> expected_aabb;
  void check() const {
    const auto actual_aabb = compute_aabb(ellipsoid);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR_AABB(expected_aabb, actual_aabb, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct SpherocylinderTestCase {
  std::string name;
  Spherocylinder<Scalar> spherocylinder;
  AABB<Scalar> expected_aabb;
  void check() const {
    const auto actual_aabb = compute_aabb(spherocylinder);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR_AABB(expected_aabb, actual_aabb, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct SpherocylinderSegmentTestCase {
  std::string name;
  SpherocylinderSegment<Scalar> spherocylinder_segment;
  AABB<Scalar> expected_aabb;
  void check() const {
    const auto actual_aabb = compute_aabb(spherocylinder_segment);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR_AABB(expected_aabb, actual_aabb, tol) << "Failed test case: " << name;
  }
};

std::vector<PointTestCase<double>> point_test_cases() {
  // The AABB for a point is just the point itself
  std::vector<PointTestCase<double>> test_cases;
  test_cases.push_back(PointTestCase<double>{.name = std::string("trivial"),   //
                                             .point = Point<double>{0, 0, 0},  //
                                             .expected_aabb = AABB<double>{0, 0, 0, 0, 0, 0}});
  test_cases.push_back(PointTestCase<double>{.name = std::string("+/-"),        //
                                             .point = Point<double>{1, -2, 3},  //
                                             .expected_aabb = AABB<double>{1, -2, 3, 1, -2, 3}});
  return test_cases;
}

std::vector<LineSegmentTestCase<double>> line_segment_test_cases() {
  // The AABB for a line segment, given by two its start and end points, is just the
  // min_x/y/z and max_x/y/z of those points
  using mundy::math::Vector3;
  std::vector<LineSegmentTestCase<double>> test_cases;
  test_cases.push_back(
      LineSegmentTestCase{.name = std::string("length 0"),                                                   //
                          .segment = LineSegment<double>{Point<double>{1, -2, 3}, Point<double>{1, -2, 3}},  //
                          .expected_aabb = AABB<double>(1, -2, 3, 1, -2, 3)});
  test_cases.push_back(
      LineSegmentTestCase{.name = std::string("regular"),                                                   //
                          .segment = LineSegment<double>{Point<double>{1, -2, 3}, Point<double>{4, 5, 6}},  //
                          .expected_aabb = AABB<double>{1, -2, 3, 4, 5, 6}});
  test_cases.push_back(
      LineSegmentTestCase{.name = std::string("regular flipped"),                                           //
                          .segment = LineSegment<double>{Point<double>{4, 5, 6}, Point<double>{1, -2, 3}},  //
                          .expected_aabb = AABB<double>{1, -2, 3, 4, 5, 6}});
  return test_cases;
}

std::vector<SphereTestCase<double>> sphere_test_cases() {
  // The min/max corners of an AABB for a sphere is just the center -/+ radius in each direction
  std::vector<SphereTestCase<double>> test_cases;
  test_cases.push_back(SphereTestCase{.name = std::string("trivial"),                       //
                                      .sphere = Sphere<double>{Point<double>{0, 0, 0}, 1},  //
                                      .expected_aabb = AABB<double>{-1, -1, -1, 1, 1, 1}});
  test_cases.push_back(SphereTestCase{.name = std::string("regular"),                        //
                                      .sphere = Sphere<double>{Point<double>{1, -2, 3}, 4},  //
                                      .expected_aabb = AABB<double>{-3, -6, -1, 5, 2, 7}});
  return test_cases;
}

std::vector<EllipsoidTestCase<double>> ellipsoid_test_cases() {
  // Our ellipsoids have a center, 3 radii, and a quaternion orientation
  // Lets start by testing unit orientation
  using mundy::math::Quaternion;
  using mundy::math::Vector3;
  std::vector<EllipsoidTestCase<double>> test_cases;

  // Rotate 90 degrees about the x-axis
  const Quaternion<double> x_90_rot = get_quaternion_x_90<double>();

  test_cases.push_back(EllipsoidTestCase{
      .name = std::string("spherical"),  //
      .ellipsoid = Ellipsoid<double>{Point<double>{1, -2, 3}, Quaternion<double>{1, 0, 0, 0}, Vector3<double>{4, 4, 4}},
      .expected_aabb = AABB<double>{-3, -6, -1, 5, 2, 7}});
  test_cases.push_back(EllipsoidTestCase{
      .name = std::string("ellipsoidal"),  //
      .ellipsoid = Ellipsoid<double>{Point<double>{1, -2, 3}, Quaternion<double>{1, 0, 0, 0}, Vector3<double>{4, 5, 6}},
      .expected_aabb = AABB<double>{-3, -7, -3, 5, 3, 9}});
  test_cases.push_back(
      EllipsoidTestCase{.name = std::string("rotated 90 degrees about x"),  // y goes to -z, z goes to y
                        .ellipsoid = Ellipsoid<double>{Point<double>{0, 0, 0}, x_90_rot, Vector3<double>{4, 5, 6}},
                        .expected_aabb = AABB<double>{-4, -6, -5, 4, 6, 5}});
  test_cases.push_back(
      EllipsoidTestCase{.name = std::string("rotated 90 degrees about x + shift"),
                        .ellipsoid = Ellipsoid<double>{Point<double>{1, -2, 3}, x_90_rot, Vector3<double>{4, 5, 6}},
                        .expected_aabb = AABB<double>{-3, -8, -2, 5, 4, 8}});
  return test_cases;
}

std::vector<SpherocylinderTestCase<double>> spherocylinder_test_cases() {
  // Our spherocylinders have a center, a radius, a length, and a quaternion orientation
  // Lets start by testing unit orientation
  using mundy::math::Quaternion;
  using mundy::math::Vector3;
  std::vector<SpherocylinderTestCase<double>> test_cases;

  const Quaternion<double> x_90_rot = get_quaternion_x_90<double>();

  test_cases.push_back(SpherocylinderTestCase{
      .name = std::string("spherical"),  //
      .spherocylinder = Spherocylinder<double>{Point<double>{1, -2, 3}, Quaternion<double>{1, 0, 0, 0}, 4, 0},
      .expected_aabb = AABB<double>{-3, -6, -1, 5, 2, 7}});
  test_cases.push_back(SpherocylinderTestCase{
      .name = std::string("line segment"),  //
      .spherocylinder = Spherocylinder<double>{Point<double>{1, -2, 3}, Quaternion<double>{1, 0, 0, 0}, 0, 4},
      .expected_aabb = AABB<double>{1, -2, 1, 1, -2, 5}});
  test_cases.push_back(SpherocylinderTestCase{
      .name = std::string("regular"),  //
      .spherocylinder = Spherocylinder<double>{Point<double>{1, -2, 3}, Quaternion<double>{1, 0, 0, 0}, 2, 4},
      .expected_aabb = AABB<double>{-1, -4, -1, 3, 0, 7}});
  test_cases.push_back(
      SpherocylinderTestCase{.name = std::string("rotated 90 degrees about x"),  // y goes to -z, z goes to y
                             .spherocylinder = Spherocylinder<double>{Point<double>{1, -2, 3}, x_90_rot, 2, 3},
                             .expected_aabb = AABB<double>{-1, -5.5, 1, 3, 1.5, 5}});
  return test_cases;
}

std::vector<SpherocylinderSegmentTestCase<double>> spherocylinder_segment_test_cases() {
  // Our spherocylinder segments have two endpoints and a radius
  // Same test cases as a spherocylinder except with endpoints
  using mundy::math::Vector3;
  std::vector<SpherocylinderSegmentTestCase<double>> test_cases;

  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("spherical"),  //
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -2, 3}, Point<double>{1, -2, 3}, 4},
      .expected_aabb = AABB<double>{-3, -6, -1, 5, 2, 7}});
  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("line segment"),  //
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -2, 1}, Point<double>{1, -2, 5}, 0},
      .expected_aabb = AABB<double>{1, -2, 1, 1, -2, 5}});
  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("regular"),  //
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -2, 1}, Point<double>{1, -2, 5}, 2},
      .expected_aabb = AABB<double>{-1, -4, -1, 3, 0, 7}});
  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("aligned with y"),  // same as regular but rotated 90 degrees about x
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -3.5, 3}, Point<double>{1, -0.5, 3}, 2},
      .expected_aabb = AABB<double>{-1, -5.5, 1, 3, 1.5, 5}});
  return test_cases;
}

TEST(ComputeAABB, HardCodedTestCases) {
  for (const auto& point_test_case : point_test_cases()) {
    point_test_case.check();
  }
  for (const auto& line_segment_test_case : line_segment_test_cases()) {
    line_segment_test_case.check();
  }
  for (const auto& sphere_test_case : sphere_test_cases()) {
    sphere_test_case.check();
  }
  for (const auto& ellipsoid_test_case : ellipsoid_test_cases()) {
    ellipsoid_test_case.check();
  }
  for (const auto& spherocylinder_test_case : spherocylinder_test_cases()) {
    spherocylinder_test_case.check();
  }
  for (const auto& spherocylinder_segment_test_case : spherocylinder_segment_test_cases()) {
    spherocylinder_segment_test_case.check();
  }
}

}  // namespace

}  // namespace geom

}  // namespace mundy
