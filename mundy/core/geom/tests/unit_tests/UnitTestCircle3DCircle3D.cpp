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
#include <gtest/gtest.h>  // for TEST, EXPECT_NEAR, etc

// C++ core
#include <cmath>  // for std::sqrt

// Trilinos includes
#include <Kokkos_Core.hpp>  // for Kokkos::numbers::pi

// Mundy
#include <mundy_geom/distance.hpp>        // for mundy::distance(Circle3D, Circle3D)
#include <mundy_geom/primitives.hpp>      // for mundy::Circle3D
#include <mundy_math/AutoDiffScalar.hpp>  // for mundy::AutoDiffScalar
#include <mundy_math/Quaternion.hpp>      // for mundy::Quaternion

namespace mundy {

namespace {

constexpr double TEST_DOUBLE_EPSILON = 1.0e-4;

// Coaxial unit circles in parallel planes z=0 and z=5: the minimum separation is the plane gap, 5.
TEST(EuclideanDistanceBetweenCircles, CoaxialParallelCircles) {
  const Circle3D<double> c1(Point<double>(0, 0, 0), Quaternion<double>(1, 0, 0, 0), 1.0);
  const Circle3D<double> c2(Point<double>(0, 0, 5), Quaternion<double>(1, 0, 0, 0), 1.0);
  EXPECT_NEAR(distance(c1, c2), 5.0, TEST_DOUBLE_EPSILON);
}

// The Euclidean separation distance is differentiable w.r.t. the circle parameters. AD gradient
// cross-checked against central finite differences.
TEST(EuclideanDistanceBetweenCircles, AutoDiffGradients) {
  using AD = AutoDiffScalar<double, 3>;  // derivatives seeded in slots 0,1,2

  auto dist_of_center2 = [](double cx, double cy, double cz) {
    const Circle3D<double> c1(Point<double>(0, 0, 0), Quaternion<double>(1, 0, 0, 0), 1.0);
    const Circle3D<double> c2(Point<double>(cx, cy, cz), Quaternion<double>(1, 0, 0, 0), 1.0);
    return distance(c1, c2);
  };
  const double cx = 3.0, cy = 0.5, cz = 5.0, h = 1.0e-5;
  const double fx = (dist_of_center2(cx + h, cy, cz) - dist_of_center2(cx - h, cy, cz)) / (2 * h);
  const double fy = (dist_of_center2(cx, cy + h, cz) - dist_of_center2(cx, cy - h, cz)) / (2 * h);
  const double fz = (dist_of_center2(cx, cy, cz + h) - dist_of_center2(cx, cy, cz - h)) / (2 * h);

  const Circle3D<AD> c1(Point<AD>(AD(0), AD(0), AD(0)), Quaternion<AD>(AD(1), AD(0), AD(0), AD(0)), AD(1));
  const Circle3D<AD> c2(Point<AD>(AD(cx, 0), AD(cy, 1), AD(cz, 2)), Quaternion<AD>(AD(1), AD(0), AD(0), AD(0)), AD(1));
  const AD d = distance(c1, c2);
  EXPECT_NEAR(d.value(), dist_of_center2(cx, cy, cz), TEST_DOUBLE_EPSILON);
  EXPECT_NEAR(d.derivatives()[0], fx, 1.0e-3);
  EXPECT_NEAR(d.derivatives()[1], fy, 1.0e-3);
  EXPECT_NEAR(d.derivatives()[2], fz, 1.0e-3);
}

}  // namespace

}  // namespace mundy
