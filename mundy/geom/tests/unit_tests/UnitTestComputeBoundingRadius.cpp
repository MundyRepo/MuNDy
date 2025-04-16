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
#include <mundy_geom/compute_bounding_radius.hpp>  // for mundy::geom::compute_bounding_radius
#include <mundy_geom/primitives.hpp>               // for mundy::geom::Ellipsoid...
#include <mundy_math/Tolerance.hpp>                // for mundy::math::get_zero_tolerance

namespace mundy {

namespace geom {

namespace {

//   - Point
//   - LineSegment
//   - Sphere
//   - Ellipsoid
//   - Spherocylinder
//   - SpherocylinderSegment

template <typename Scalar>
struct PointTestCase {
  std::string name;
  Point<Scalar> point;
  Scalar expected_bounding_radius;
  void check() const {
    const auto actual_bounding_radius = compute_bounding_radius(point);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR(expected_bounding_radius, actual_bounding_radius, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct LineSegmentTestCase {
  std::string name;
  LineSegment<Scalar> segment;
  Scalar expected_bounding_radius;
  void check() const {
    const auto actual_bounding_radius = compute_bounding_radius(segment);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR(expected_bounding_radius, actual_bounding_radius, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct SphereTestCase {
  std::string name;
  Sphere<Scalar> sphere;
  Scalar expected_bounding_radius;
  void check() const {
    const auto actual_bounding_radius = compute_bounding_radius(sphere);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR(expected_bounding_radius, actual_bounding_radius, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct EllipsoidTestCase {
  std::string name;
  Ellipsoid<Scalar> ellipsoid;
  Scalar expected_bounding_radius;
  void check() const {
    const auto actual_bounding_radius = compute_bounding_radius(ellipsoid);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR(expected_bounding_radius, actual_bounding_radius, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct SpherocylinderTestCase {
  std::string name;
  Spherocylinder<Scalar> spherocylinder;
  Scalar expected_bounding_radius;
  void check() const {
    const auto actual_bounding_radius = compute_bounding_radius(spherocylinder);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR(expected_bounding_radius, actual_bounding_radius, tol) << "Failed test case: " << name;
  }
};

template <typename Scalar>
struct SpherocylinderSegmentTestCase {
  std::string name;
  SpherocylinderSegment<Scalar> spherocylinder_segment;
  Scalar expected_bounding_radius;
  void check() const {
    const auto actual_bounding_radius = compute_bounding_radius(spherocylinder_segment);
    const double tol = mundy::math::get_relaxed_zero_tolerance<Scalar>();
    ASSERT_NEAR(expected_bounding_radius, actual_bounding_radius, tol) << "Failed test case: " << name;
  }
};

std::vector<PointTestCase<double>> point_test_cases() {
  // The bounding radius for a point is 0.
  std::vector<PointTestCase<double>> test_cases;
  test_cases.push_back(PointTestCase{.name = std::string("trivial"),   //
                                     .point = Point<double>{0, 0, 0},  //
                                     .expected_bounding_radius = 0.0});
  test_cases.push_back(PointTestCase{.name = std::string("+/-"),        //
                                     .point = Point<double>{1, -2, 3},  //
                                     .expected_bounding_radius = 0.0});
  return test_cases;
}

std::vector<LineSegmentTestCase<double>> line_segment_test_cases() {
  // The bounding radius for a line segment, given by two its start and end points, is just the
  // half-length of the line segment.
  using mundy::math::Vector3;
  std::vector<LineSegmentTestCase<double>> test_cases;
  test_cases.push_back(
      LineSegmentTestCase{.name = std::string("length 0"),                                                   //
                          .segment = LineSegment<double>{Point<double>{1, -2, 3}, Point<double>{1, -2, 3}},  //
                          .expected_bounding_radius = 0.0});
  test_cases.push_back(
      LineSegmentTestCase{.name = std::string("regular"),                                                   //
                          .segment = LineSegment<double>{Point<double>{1, -2, 3}, Point<double>{4, 5, 6}},  //
                          .expected_bounding_radius = 0.5 * std::sqrt(3.0 * 3.0 + 7.0 * 7.0 + 3.0 * 3.0)});
  test_cases.push_back(
      LineSegmentTestCase{.name = std::string("regular flipped"),                                           //
                          .segment = LineSegment<double>{Point<double>{4, 5, 6}, Point<double>{1, -2, 3}},  //
                          .expected_bounding_radius = 0.5 * std::sqrt(3.0 * 3.0 + 7.0 * 7.0 + 3.0 * 3.0)});
  return test_cases;
}

std::vector<SphereTestCase<double>> sphere_test_cases() {
  // The bounding radius of a sphere is its radius.
  std::vector<SphereTestCase<double>> test_cases;
  test_cases.push_back(SphereTestCase{.name = std::string("trivial"),                       //
                                      .sphere = Sphere<double>{Point<double>{0, 0, 0}, 1},  //
                                      .expected_bounding_radius = 1.0});
  test_cases.push_back(SphereTestCase{.name = std::string("regular"),                        //
                                      .sphere = Sphere<double>{Point<double>{1, -2, 3}, 4},  //
                                      .expected_bounding_radius = 4.0});
  return test_cases;
}

std::vector<EllipsoidTestCase<double>> ellipsoid_test_cases() {
  // The bounding radius of an ellipsoid is the maximum of its radii.
  using mundy::math::Quaternion;
  using mundy::math::Vector3;
  std::vector<EllipsoidTestCase<double>> test_cases;

  // Bounding radius is independent of orientation
  Quaternion<double> random_quat = Quaternion<double>{static_cast<double>(rand()), static_cast<double>(rand()),
                                                      static_cast<double>(rand()), static_cast<double>(rand())};
  random_quat.normalize();

  test_cases.push_back(
      EllipsoidTestCase{.name = std::string("spherical"),  //
                        .ellipsoid = Ellipsoid<double>{Point<double>{1, -2, 3}, random_quat, Vector3<double>{4, 4, 4}},
                        .expected_bounding_radius = 4.0});
  test_cases.push_back(
      EllipsoidTestCase{.name = std::string("ellipsoidal"),  //
                        .ellipsoid = Ellipsoid<double>{Point<double>{1, -2, 3}, random_quat, Vector3<double>{4, 5, 6}},
                        .expected_bounding_radius = 6.0});
  return test_cases;
}

std::vector<SpherocylinderTestCase<double>> spherocylinder_test_cases() {
  // Our spherocylinders have a center, a radius, a length, and a quaternion orientation
  // Their bounding radius is just the 0.5 length + radius
  using mundy::math::Quaternion;
  using mundy::math::Vector3;
  std::vector<SpherocylinderTestCase<double>> test_cases;

  // Bounding radius is independent of orientation
  Quaternion<double> random_quat = Quaternion<double>{static_cast<double>(rand()), static_cast<double>(rand()),
                                                      static_cast<double>(rand()), static_cast<double>(rand())};
  random_quat.normalize();

  test_cases.push_back(
      SpherocylinderTestCase{.name = std::string("spherical"),  //
                             .spherocylinder = Spherocylinder<double>{Point<double>{1, -2, 3}, random_quat, 4, 0},
                             .expected_bounding_radius = 4.0});
  test_cases.push_back(
      SpherocylinderTestCase{.name = std::string("line segment"),  //
                             .spherocylinder = Spherocylinder<double>{Point<double>{1, -2, 3}, random_quat, 0, 4},
                             .expected_bounding_radius = 2.0});
  test_cases.push_back(
      SpherocylinderTestCase{.name = std::string("regular"),  //
                             .spherocylinder = Spherocylinder<double>{Point<double>{1, -2, 3}, random_quat, 2, 4},
                             .expected_bounding_radius = 4.0});
  return test_cases;
}

std::vector<SpherocylinderSegmentTestCase<double>> spherocylinder_segment_test_cases() {
  // Our spherocylinder segments have two endpoints and a radius.
  // Their bounding radius is just the 0.5 length + radius
  using mundy::math::Vector3;
  std::vector<SpherocylinderSegmentTestCase<double>> test_cases;

  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("spherical"),  //
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -2, 3}, Point<double>{1, -2, 3}, 4},
      .expected_bounding_radius = 4.0});
  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("line segment"),  //
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -2, 1}, Point<double>{1, -2, 5}, 0},
      .expected_bounding_radius = 2.0});
  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("regular"),  //
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -2, 1}, Point<double>{1, -2, 5}, 2},
      .expected_bounding_radius = 4.0});
  test_cases.push_back(SpherocylinderSegmentTestCase{
      .name = std::string("aligned with y"),  // same as regular but rotated 90 degrees about x
      .spherocylinder_segment = SpherocylinderSegment<double>{Point<double>{1, -3.5, 3}, Point<double>{1, -0.5, 3}, 2},
      .expected_bounding_radius = 3.5});
  return test_cases;
}

TEST(ComputeBoundingRadius, HardCodedTestCases) {
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
