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
#include <mundy_geom/distance.hpp>                  // for mundy::geom::distance
#include <mundy_geom/distance/DistanceMetrics.hpp>  // for mundy::geom::FreeSpaceMetric, mundy::geom::PeriodicSpaceMetric
#include <mundy_geom/primitives.hpp>                // for mundy::geom::Point, mundy::geom::LineSegment
#include <mundy_math/Tolerance.hpp>                 // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>                   // for mundy::math::Vector3

/// \brief The following global is used to control the number of samples per test.
/// For unit tests, this number should be kept low to ensure fast test times, but to still give an immediate warning if
/// something went very wrong. For integration tests, we recommend setting this number to 10,000 or more.
#ifndef MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST
#define MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST 1000000
#endif

namespace mundy {

namespace geom {

namespace {

//! \brief Unit tests
//@{

TEST(FreeAndPeriodicMetricConstruction, PositiveResult) {
  Point<double> point1{1.0, 2.0, 3.0};
  Point<double> point2{4.0, 5.0, 6.0};
  math::Vector3<double> cell_size{10.0, 10.0, 10.0};

  FreeSpaceMetric<double> free_space_metric{};
  PeriodicSpaceMetric<double> periodic_space_metric = periodic_metric_from_unit_cell(cell_size);

  // Calculate the separation vectors in this case, which should be equal
  auto sep_free = free_space_metric(point1, point2);
  auto sep_periodic = periodic_space_metric(point1, point2);

  EXPECT_DOUBLE_EQ(sep_free[0], point2[0] - point1[0]);
  EXPECT_DOUBLE_EQ(sep_free[1], point2[1] - point1[1]);
  EXPECT_DOUBLE_EQ(sep_free[2], point2[2] - point1[2]);
  EXPECT_DOUBLE_EQ(sep_periodic[0], point2[0] - point1[0]);
  EXPECT_DOUBLE_EQ(sep_periodic[1], point2[1] - point1[1]);
  EXPECT_DOUBLE_EQ(sep_periodic[2], point2[2] - point1[2]);
}

TEST(FreeAndPeriodicMetricDistances, PositiveResult) {
  Point<double> point1{1.0, 1.0, 1.0};
  math::Vector3<double> cell_size{100.0, 30.0, 20.0};

  FreeSpaceMetric<double> free_space_metric{};
  PeriodicSpaceMetric<double> periodic_space_metric = periodic_metric_from_unit_cell(cell_size);

  // Calculate the separation vectors in the near case not crossing a periodic boundary
  Point<double> point2_near{2.0, 2.0, 2.0};
  auto sep_free = free_space_metric(point1, point2_near);
  auto sep_periodic = periodic_space_metric(point1, point2_near);

  EXPECT_DOUBLE_EQ(sep_free[0], point2_near[0] - point1[0]);
  EXPECT_DOUBLE_EQ(sep_free[1], point2_near[1] - point1[1]);
  EXPECT_DOUBLE_EQ(sep_free[2], point2_near[2] - point1[2]);
  EXPECT_DOUBLE_EQ(sep_periodic[0], point2_near[0] - point1[0]);
  EXPECT_DOUBLE_EQ(sep_periodic[1], point2_near[1] - point1[1]);
  EXPECT_DOUBLE_EQ(sep_periodic[2], point2_near[2] - point1[2]);

  // Calculate when point2 has crossed the periodic boundary in x
  Point<double> point2_crossed_x{99.0, 2.0, 2.0};
  sep_free = free_space_metric(point1, point2_crossed_x);
  sep_periodic = periodic_space_metric(point1, point2_crossed_x);

  std::cout << "Real space separation vector crossing x boundary: " << sep_free << std::endl;
  std::cout << "Periodic space separation vector crossing x boundary: " << sep_periodic << std::endl;

  EXPECT_DOUBLE_EQ(sep_free[0], point2_crossed_x[0] - point1[0]);
  EXPECT_DOUBLE_EQ(sep_free[1], point2_crossed_x[1] - point1[1]);
  EXPECT_DOUBLE_EQ(sep_free[2], point2_crossed_x[2] - point1[2]);
  EXPECT_NE(sep_periodic[0], point2_crossed_x[0] - point1[0]);
  EXPECT_DOUBLE_EQ(sep_periodic[1], point2_crossed_x[1] - point1[1]);
  EXPECT_DOUBLE_EQ(sep_periodic[2], point2_crossed_x[2] - point1[2]);
}

TEST(FreeAndPeriodicMetricPointPoint, PositiveResult) {
  // Origin point
  Point<double> point1{1.0, 1.0, 1.0};
  math::Vector3<double> cell_size{100.0, 30.0, 20.0};
  
  PeriodicSpaceMetric<double> periodic_space_metric = periodic_metric_from_unit_cell(cell_size);

  // Test point that should not be a periodic image
  Point<double> point2{2.0, 2.0, 2.0};

  // Get the distance measures without separation vectors
  auto free_distance = distance(point1, point2);
  auto periodic_distance = distance_pbc(point1, point2, periodic_space_metric);

  EXPECT_DOUBLE_EQ(free_distance, periodic_distance);

  // Get the separation vectors
  math::Vector3<double> sep_free;
  math::Vector3<double> sep_periodic;
  free_distance = distance(point1, point2, sep_free);
  periodic_distance = distance_pbc(point1, point2, periodic_space_metric, sep_periodic);
  EXPECT_DOUBLE_EQ(sep_free[0], point2[0] - point1[0]);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(sep_free[i], sep_periodic[i]);
  }

  // Test point that is wrapped in x
  Point<double> point2_crossed_x{99.0, 2.0, 2.0};
  free_distance = distance(point1, point2_crossed_x, sep_free);
  periodic_distance = distance_pbc(point1, point2_crossed_x, periodic_space_metric, sep_periodic);
  EXPECT_NE(free_distance, periodic_distance);
  EXPECT_DOUBLE_EQ(sep_periodic[0], -2.0);
  EXPECT_DOUBLE_EQ(sep_periodic[1], sep_free[1]);
  EXPECT_DOUBLE_EQ(sep_periodic[2], sep_free[2]);
}

TEST(FreeAndPeriodicMetricPointLineSegment, PositiveResult) {
  // Need to test both moving the line segment to wrap around the periodic boundary and moving the point to wrap
  // around the periodic boundary.

  // Moving the line segment around, using a single test point
  Point<double> point1{1.0, 1.0, 1.0};
  math::Vector3<double> cell_size{100.0, 100.0, 100.0};

  PeriodicSpaceMetric<double> periodic_space_metric = periodic_metric_from_unit_cell(cell_size);

  // Test segment that should not be a periodic image
  LineSegment<double> line_segment{Point<double>{2.0, 1.0, 1.0},
                                                Point<double>{4.0, 1.0, 1.0}};

  // Get the distance measures without separation vectors
  auto free_distance = distance(point1, line_segment);
  auto periodic_distance = distance_pbc(point1, line_segment, periodic_space_metric);
  EXPECT_DOUBLE_EQ(free_distance, periodic_distance);
  // Now do with separation vectors
  math::Vector3<double> sep_free;
  math::Vector3<double> closest_free;
  double arc_free;
  math::Vector3<double> sep_periodic;
  math::Vector3<double> closest_periodic;
  double arc_periodic;
  free_distance = distance(point1, line_segment, closest_free, arc_free, sep_free);
  periodic_distance = distance_pbc(point1, line_segment, periodic_space_metric, closest_periodic,
                                                arc_periodic, sep_periodic);
  EXPECT_DOUBLE_EQ(sep_free[0], 1.0);
  EXPECT_DOUBLE_EQ(sep_free[1], 0.0);
  EXPECT_DOUBLE_EQ(sep_free[2], 0.0);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(sep_free[i], sep_periodic[i]);
  }

  // Place a segment that is fully in the periodic image
  LineSegment<double> line_segment_crossed_x{Point<double>{97.0, 1.0, 1.0},
                                                          Point<double>{99.0, 1.0, 1.0}};
  free_distance = distance(point1, line_segment_crossed_x);
  periodic_distance = distance_pbc(point1, line_segment_crossed_x, periodic_space_metric);
  EXPECT_NE(free_distance, periodic_distance);
  EXPECT_DOUBLE_EQ(periodic_distance, 2.0);
  // separation distance too
  free_distance = distance(point1, line_segment_crossed_x, closest_free, arc_free, sep_free);
  periodic_distance = distance_pbc(point1, line_segment_crossed_x, periodic_space_metric, closest_periodic,
                                                arc_periodic, sep_periodic);
  EXPECT_DOUBLE_EQ(sep_free[0], 96.0);
  EXPECT_DOUBLE_EQ(sep_free[1], 0.0);
  EXPECT_DOUBLE_EQ(sep_free[2], 0.0);
  EXPECT_DOUBLE_EQ(closest_free[0], 97.0);
  EXPECT_DOUBLE_EQ(closest_free[1], 1.0);
  EXPECT_DOUBLE_EQ(closest_free[2], 1.0);
  // Periodic
  EXPECT_DOUBLE_EQ(sep_periodic[0], -2.0);
  EXPECT_DOUBLE_EQ(sep_periodic[1], 0.0);
  EXPECT_DOUBLE_EQ(sep_periodic[2], 0.0);
  EXPECT_DOUBLE_EQ(closest_periodic[0], 99.0);
  EXPECT_DOUBLE_EQ(closest_periodic[1], 1.0);
  EXPECT_DOUBLE_EQ(closest_periodic[2], 1.0);

  // Place a segment that is fully in the periodic image has the point orthogonal
  LineSegment<double> line_segment_crossed_y_orthogonal{Point<double>{0.0, 99.0, 1.0},
                                                                     Point<double>{2.0, 99.0, 1.0}};
  free_distance = distance(point1, line_segment_crossed_y_orthogonal);
  periodic_distance = distance_pbc(point1, line_segment_crossed_y_orthogonal, periodic_space_metric);
  EXPECT_NE(free_distance, periodic_distance);
  EXPECT_DOUBLE_EQ(periodic_distance, 2.0);
  // seaparation distance too
  free_distance = distance(point1, line_segment_crossed_y_orthogonal, closest_free, arc_free, sep_free);
  periodic_distance = distance_pbc(
      point1, line_segment_crossed_y_orthogonal, periodic_space_metric, closest_periodic, arc_periodic,
      sep_periodic);
  EXPECT_DOUBLE_EQ(sep_free[0], 0.0);
  EXPECT_DOUBLE_EQ(sep_free[1], 98.0);
  EXPECT_DOUBLE_EQ(sep_free[2], 0.0);
  EXPECT_DOUBLE_EQ(closest_free[0], 1.0);
  EXPECT_DOUBLE_EQ(closest_free[1], 99.0);
  EXPECT_DOUBLE_EQ(closest_free[2], 1.0);
  // Periodic
  EXPECT_DOUBLE_EQ(sep_periodic[0], 0.0);
  EXPECT_DOUBLE_EQ(sep_periodic[1], -2.0);
  EXPECT_DOUBLE_EQ(sep_periodic[2], 0.0);
  EXPECT_DOUBLE_EQ(closest_periodic[0], 1.0);
  EXPECT_DOUBLE_EQ(closest_periodic[1], 99.0);
  EXPECT_DOUBLE_EQ(closest_periodic[2], 1.0);

  // Now the tricky one, split the segment across the periodic boundary, also use a point that is further into the
  // domain
  Point<double> point2{2.0, 2.0, 1.0};
  LineSegment<double> line_segment_crossed_x_y{Point<double>{99.0, 99.0, 1.0},
                                                            Point<double>{1.0, 1.0, 1.0}};
  free_distance = distance(point2, line_segment_crossed_x_y);
  periodic_distance = distance_pbc(point2, line_segment_crossed_x_y, periodic_space_metric);
  EXPECT_NE(free_distance, periodic_distance);
  EXPECT_DOUBLE_EQ(periodic_distance, Kokkos::sqrt(2.0));
  // separation distance too
  free_distance = distance(point2, line_segment_crossed_x_y, closest_free, arc_free, sep_free);
  periodic_distance = distance_pbc(
      point2, line_segment_crossed_x_y, periodic_space_metric, closest_periodic, arc_periodic, sep_periodic);
  EXPECT_NEAR(sep_free[0], 0.0, 1e-6);
  EXPECT_NEAR(sep_free[1], 0.0, 1e-6);
  EXPECT_NEAR(sep_free[2], 0.0, 1e-6);
  EXPECT_DOUBLE_EQ(closest_free[0], 2.0);
  EXPECT_DOUBLE_EQ(closest_free[1], 2.0);
  EXPECT_DOUBLE_EQ(closest_free[2], 1.0);
  // Periodic
  EXPECT_DOUBLE_EQ(sep_periodic[0], -1.0);
  EXPECT_DOUBLE_EQ(sep_periodic[1], -1.0);
  EXPECT_DOUBLE_EQ(sep_periodic[2], 0.0);
  EXPECT_DOUBLE_EQ(closest_periodic[0], 1.0);
  EXPECT_DOUBLE_EQ(closest_periodic[1], 1.0);
  EXPECT_DOUBLE_EQ(closest_periodic[2], 1.0);
}

}  // namespace

}  // namespace geom

}  // namespace mundy
