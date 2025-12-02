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
#include <mundy_geom/distance.hpp>    // for mundy::geom::distance
#include <mundy_geom/primitives.hpp>  // for mundy::geom::Point, mundy::geom::LineSegment
#include <mundy_math/Tolerance.hpp>   // for mundy::math::get_zero_tolerance

/// \brief The following global is used to control the number of samples per test.
/// For unit tests, this number should be kept low to ensure fast test times, but to still give an immediate warning if
/// something went very wrong. For integration tests, we recommend setting this number to 10,000 or more.
#ifndef MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST
#define MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST 1000000
#endif

namespace mundy {

namespace geom {

namespace {

// The following test differ from VTK, which erroneously throws a failure only when BOTH u and v are off in many of the
// tests.

const double TEST_DOUBLE_EPSILON = 1.e-6;

//! \brief Helper functions
//@{

// Utility function to generate a unique seed for each test based on its GTEST name.
size_t generate_test_seed() {
  const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string test_identifier = std::string(test_info->test_suite_name()) + "." + test_info->name();
  return std::hash<std::string>{}(test_identifier);
}

// A concept for RNG types that offer a rand<T>() function.
template <typename T>
concept RandomNumberGenerator = requires(T rng) {
  { rng.template rand<double>() } -> std::convertible_to<double>;
};

template <RandomNumberGenerator RngType>
void generate_intersecting_line_segments(RngType& rng, Point<double>& a1, Point<double>& a2, Point<double>& b1,
                                         Point<double>& b2, double& u, double& v) {
  // Generate two line segments ((a1,a2) and (b1,b2)) that intersect, and set
  // u and v as the parametric points of intersection on the two respective
  // lines.
  Point<double> intersection;
  for (unsigned i = 0; i < 3; i++) {
    intersection[i] = rng.template rand<double>();
    a1[i] = rng.template rand<double>();
    b1[i] = rng.template rand<double>();
  }

  // Note, we cannot generate u, v directly as getting a2 and b2 for degenerate cases is difficult. Instead, we can
  // generate a random ratio for the distance between the intersection and the endpoints of the line segments. This can
  // still generate degenerate rods (as desired), but without dividing by zero.
  const double ratio1 = rng.template rand<double>();
  const double ratio2 = rng.template rand<double>();
  a2 = a1 + (intersection - a1) * (1. + ratio1);
  b2 = b1 + (intersection - b1) * (1. + ratio2);

  const double len_a = mundy::math::norm(a2 - a1);
  const double len_b = mundy::math::norm(b2 - b1);

  const double len_a1_to_intersection = mundy::math::norm(a1 - intersection);
  const double len_b1_to_intersection = mundy::math::norm(b1 - intersection);

  u = len_a1_to_intersection / len_a;
  v = len_b1_to_intersection / len_b;
}

template <RandomNumberGenerator RngType>
void random_sphere(RngType& rng, const double radius, const Point<double>& offset, Point<double>& value) {
  // Generate a point within a sphere centered at offset with a given radius.
  double theta = 2. * Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  double phi = Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  value[0] = radius * cos(theta) * sin(phi) + offset[0];
  value[1] = radius * sin(theta) * sin(phi) + offset[1];
  value[2] = radius * cos(phi) + offset[2];
}

template <RandomNumberGenerator RngType>
void generate_non_intersecting_line_segments(RngType& rng, Point<double>& a1, Point<double>& a2, Point<double>& b1,
                                             Point<double>& b2) {
  // Generate two line segments ((a1,a2) and (b1,b2)) that do not intersect.
  // The endpoints of each line segment are generated from two non-overlapping
  // spheres, and the two spheres for each line segment are physically displaced
  // as well.

  static const double radius = 0.5 - 1.e-6;
  random_sphere(rng, radius, Point<double>(0., 0., 0.), a1);
  random_sphere(rng, radius, Point<double>(1., 0., 0.), a2);
  random_sphere(rng, radius, Point<double>(0., 1., 0.), b1);
  random_sphere(rng, radius, Point<double>(1., 1., 0.), b2);
}

template <RandomNumberGenerator RngType>
void generate_colinear_line_segments(RngType& rng, Point<double>& a1, Point<double>& a2, Point<double>& b1,
                                     Point<double>& b2) {
  // Generate two line segments ((a1,a2) and (b1,b2)) that are colinear.
  for (unsigned i = 0; i < 3; i++) {
    a1[i] = rng.template rand<double>();
    a2[i] = rng.template rand<double>();
    double tmp = rng.template rand<double>();
    b1[i] = a1[i] + tmp;
    b2[i] = a2[i] + tmp;
  }
}

// Use a class enum to specify the type of degeneracy (if any) to force upon the lines.
enum class DegeneracyType {
  NONE,
  A1_EQUALS_A12,
  A2_EQUALS_A12,
  B1_EQUALS_B12,
  B2_EQUALS_B12,
  A1_EQUALS_A2,
  B1_EQUALS_B2,
  RANDOM
};

template <RandomNumberGenerator RngType>
void generate_lines_at_known_distance(RngType& rng, double& line_dist, Point<double>& a1, Point<double>& a2,
                                      Point<double>& b1, Point<double>& b2, Point<double>& a12, Point<double>& b12,
                                      double& u, double& v, DegeneracyType degeneracy = DegeneracyType::RANDOM) {
  // Generate two lines ((a1,a2) and (b1,b2)) set a known distance (line_dist)
  // apart. the parameter and value of the closest points for lines a and b are
  // a12, u and b12, v, respectively.

  // Importantly, lines require that the distance between endpoints on the same line is non-zero. Otherwise, the
  // orientation of the line is not well-defined.
  if (degeneracy == DegeneracyType::RANDOM) {
    // Randomly choose a degeneracy for the lines. Only using the first five degeneracies (0, 1, 2, 3, 4)
    degeneracy = static_cast<DegeneracyType>(static_cast<int>(rng.template rand<double>() * 4));
  }
  MUNDY_THROW_ASSERT(degeneracy != DegeneracyType::A1_EQUALS_A2, std::logic_error,
                     "A1_EQUALS_A2 is not a valid degeneracy for lines.");
  MUNDY_THROW_ASSERT(degeneracy != DegeneracyType::B1_EQUALS_B2, std::logic_error,
                     "B1_EQUALS_B2 is not a valid degeneracy for lines.");

  // Generate two unit vectors v1 and v2, and their cross product v3.
  // v1 and v2 will represent the orientations of the two lines, and v3 will be the direction of the line connecting
  // them.
  //
  // Properly generating unit vectors can be done by choosing phi in [0, 2pi] and z = cos(theta) in [-1, 1], and then
  // converting to cartesian coordinates.
  const double phi1 = 2. * Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  const double phi2 = 2. * Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  const double theta1 = acos(2. * rng.template rand<double>() - 1.);
  const double theta2 = acos(2. * rng.template rand<double>() - 1.);

  Point<double> v1 = {
      cos(phi1) * sin(theta1),
      sin(phi1) * sin(theta1),
      cos(theta1),
  };
  Point<double> v2 = {
      cos(phi2) * sin(theta2),
      sin(phi2) * sin(theta2),
      cos(theta2),
  };

  // Because v1 and v2 may be equal, care needs to be taken when determining the line distance along v3.
  // Instead of normalizing v3, we'll scale it randomly by [0, 1] such that its norm is the desired line distance.
  const Point<double> v3 = mundy::math::cross(v1, v2) * rng.template rand<double>();
  const double norm_v3 = mundy::math::norm(v3);
  line_dist = norm_v3;

  // Now that we have the unit orientations, the separation vector, and the separation distance, we need to decide the
  // placements of the ends of the lines. The key here is to allow degenerate lines where the endpoints are the same or
  // where one of the endpoints is the same as the contact point. We'll choose a the left and right distance from the
  // contact point from U01 with a small shift to guarantee that the endpoints can never be the same.
  double a1_to_a12 = (degeneracy == DegeneracyType::A1_EQUALS_A12) ? 0.0 : 0.1 + rng.template rand<double>();
  double a12_to_a2 = (degeneracy == DegeneracyType::A2_EQUALS_A12) ? 0.0 : 0.1 + rng.template rand<double>();
  double b1_to_b12 = (degeneracy == DegeneracyType::B1_EQUALS_B12) ? 0.0 : 0.1 + rng.template rand<double>();
  double b12_to_b2 = (degeneracy == DegeneracyType::B2_EQUALS_B12) ? 0.0 : 0.1 + rng.template rand<double>();

  for (unsigned i = 0; i < 3; i++) {
    a12[i] = rng.template rand<double>();
    b12[i] = a12[i] + v3[i];
    a1[i] = a12[i] - a1_to_a12 * v1[i];
    a2[i] = a12[i] + a12_to_a2 * v1[i];
    b1[i] = b12[i] - b1_to_b12 * v2[i];
    b2[i] = b12[i] + b12_to_b2 * v2[i];
  }

  u = a1_to_a12 / (a1_to_a12 + a12_to_a2);
  v = b1_to_b12 / (b1_to_b12 + b12_to_b2);
}

template <RandomNumberGenerator RngType>
void generate_line_segments_at_known_distance(RngType& rng, double& line_dist, Point<double>& a1, Point<double>& a2,
                                              Point<double>& b1, Point<double>& b2, Point<double>& a12,
                                              Point<double>& b12, double& u, double& v,
                                              DegeneracyType degeneracy = DegeneracyType::RANDOM) {
  // Generate two line segments ((a1,a2) and (b1,b2)) set a known distance (line_dist)
  // apart. The parameter and value of the closest points for lines a and b are
  // a12, u and b12, v, respectively.

  if (degeneracy == DegeneracyType::RANDOM) {
    // Randomly choose a degeneracy for the line segments. Only using the first seven degeneracies (0, 1, 2, 3, 4, 5,
    // 6).
    degeneracy = static_cast<DegeneracyType>(static_cast<int>(rng.template rand<double>() * 6));
  }

  // Generate two unit vectors v1 and v2, and their cross product v3.
  // v1 and v2 will represent the orientations of the two lines, and v3 will be the direction of the line connecting
  // them.
  //
  // Properly generating unit vectors can be done by choosing phi in [0, 2pi] and z = cos(theta) in [-1, 1], and then
  // converting to cartesian coordinates.
  const double phi1 = 2. * Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  const double phi2 = 2. * Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  const double theta1 = acos(2. * rng.template rand<double>() - 1.);
  const double theta2 = acos(2. * rng.template rand<double>() - 1.);

  Point<double> v1 = {
      cos(phi1) * sin(theta1),
      sin(phi1) * sin(theta1),
      cos(theta1),
  };
  Point<double> v2 = {
      cos(phi2) * sin(theta2),
      sin(phi2) * sin(theta2),
      cos(theta2),
  };

  // Because v1 and v2 may be equal, care needs to be taken when determining the line distance along v3.
  // Instead of normalizing v3, we'll scale it randomly by [0, 1] such that its norm is the desired line distance.
  const Point<double> v3 = mundy::math::cross(v1, v2) * rng.template rand<double>();
  const double norm_v3 = mundy::math::norm(v3);
  line_dist = norm_v3;

  // Now that we have the unit orientations, the separation vector, and the separation distance, we need to decide the
  // placements of the ends of the lines. The key here is to allow degenerate lines where the endpoints are the same or
  // where one of the endpoints is the same as the contact point. We'll choose a the left and right distance from the
  // contact point from U01 and then check for degeneracies.
  double a1_to_a12 = ((degeneracy == DegeneracyType::A1_EQUALS_A12) || (degeneracy == DegeneracyType::A1_EQUALS_A2))
                         ? 0.0
                         : rng.template rand<double>();
  double a12_to_a2 = ((degeneracy == DegeneracyType::A2_EQUALS_A12) || (degeneracy == DegeneracyType::A1_EQUALS_A2))
                         ? 0.0
                         : rng.template rand<double>();
  double b1_to_b12 = ((degeneracy == DegeneracyType::B1_EQUALS_B12) || (degeneracy == DegeneracyType::B1_EQUALS_B2))
                         ? 0.0
                         : rng.template rand<double>();
  double b12_to_b2 = ((degeneracy == DegeneracyType::B2_EQUALS_B12) || (degeneracy == DegeneracyType::B1_EQUALS_B2))
                         ? 0.0
                         : rng.template rand<double>();

  for (unsigned i = 0; i < 3; i++) {
    a12[i] = rng.template rand<double>();
    b12[i] = a12[i] + v3[i];
    a1[i] = a12[i] - a1_to_a12 * v1[i];
    a2[i] = a12[i] + a12_to_a2 * v1[i];
    b1[i] = b12[i] - b1_to_b12 * v2[i];
    b2[i] = b12[i] + b12_to_b2 * v2[i];
  }

  const bool is_a_degenerate = (a1_to_a12 + a12_to_a2) < mundy::math::get_zero_tolerance<double>();
  const bool is_b_degenerate = (b1_to_b12 + b12_to_b2) < mundy::math::get_zero_tolerance<double>();
  u = is_a_degenerate ? 0. : a1_to_a12 / (a1_to_a12 + a12_to_a2);
  v = is_b_degenerate ? 0. : b1_to_b12 / (b1_to_b12 + b12_to_b2);
}

template <RandomNumberGenerator RngType>
void generate_line_at_known_distance(RngType& rng, Point<double>& a1, Point<double>& a2, Point<double>& a12,
                                     Point<double>& p, double& dist) {
  // Generate a line (a1,a2) set a known distance (dist) from a generated point p.

  // Generate a random point p
  for (unsigned i = 0; i < 3; i++) {
    p[i] = rng.template rand<double>();
  }

  // Generate a random unit vector v1 to describe the orientation of the line and a random unit vector v2 orthogonal to
  // v1 to describe the direction of the line from the point to the nearest point on the rod.
  //
  // Properly generating unit vectors can be done by choosing phi in [0, 2pi] and z = cos(theta) in [-1, 1], and then
  // converting to cartesian coordinates.
  const double phi1 = 2. * Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  const double phi2 = 2. * Kokkos::numbers::pi_v<double> * rng.template rand<double>();
  const double theta1 = acos(2. * rng.template rand<double>() - 1.);
  const double theta2 = acos(2. * rng.template rand<double>() - 1.);
  Point<double> v1 = {
      cos(phi1) * sin(theta1),
      sin(phi1) * sin(theta1),
      cos(theta1),
  };
  Point<double> v2 = {
      cos(phi2) * sin(theta2),
      sin(phi2) * sin(theta2),
      cos(theta2),
  };

  // At this point, v2 is not orthogonal to v1. We'll use the orthogonal projection formula to make it so.
  // Specifically, v2 = v2 - (v2 dot v1) * v1.
  // Instead of normalizing v3, we'll scale it randomly by [0, 1] such that its norm is the desired line distance.
  v2 -= mundy::math::dot(v2, v1) * v1;
  v2 *= rng.template rand<double>();
  dist = mundy::math::norm(v2);

  // Hardcode one degeneracy at a time.
  bool force_a1_equals_a12 = false;
  bool force_a2_equals_a12 = false;

  // Use a random distance from the endpoints to the nearest point on the line.
  double a1_to_a12 = force_a1_equals_a12 ? 0.0 : rng.template rand<double>();
  double a12_to_a2 = force_a2_equals_a12 ? 0.0 : rng.template rand<double>();
  for (unsigned i = 0; i < 3; i++) {
    a12[i] = p[i] + v2[i];
    a1[i] = a12[i] - a1_to_a12 * v1[i];
    a2[i] = a12[i] + a12_to_a2 * v1[i];
  }
}
//@}

//! \brief Unit tests
//@{

TEST(DistanceBetweenLines, PositiveResult) {
  openrand::Philox rng(generate_test_seed(), 0);
  unsigned nTests = MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST;

  Point<double> a1, a2, b1, b2, a12_expected, a12_actual, b12_expected, b12_actual;
  mundy::math::Vector3d sep_actual;
  double u_expected, v_expected, u_actual, v_actual, dist_expected, dist_actual;
  for (unsigned i = 0; i < nTests; i++) {
    // (a1, a2) and (b1, b2) are points on the lines and u and v are the archlength measured from the left endpoint.
    // The archlength we use is measured from the center.
    generate_lines_at_known_distance(rng, dist_expected, a1, a2, b1, b2, a12_expected, b12_expected, u_expected,
                                     v_expected);

    const auto center_a = 0.5 * (a1 + a2);
    const auto center_b = 0.5 * (b1 + b2);
    const double len_a = mundy::math::norm(a2 - a1);
    const double len_b = mundy::math::norm(b2 - b1);
    const auto dir_a = (a2 - a1) / len_a;
    const auto dir_b = (b2 - b1) / len_b;
    const Line<double> line_a(center_a, dir_a);
    const Line<double> line_b(center_b, dir_b);
    u_expected *= len_a;
    v_expected *= len_b;
    u_expected -= 0.5 * len_a;
    v_expected -= 0.5 * len_b;

    dist_actual = distance(line_a, line_b, a12_actual, b12_actual, u_actual, v_actual, sep_actual);
    ASSERT_NEAR(dist_expected, dist_actual, TEST_DOUBLE_EPSILON);

    for (unsigned j = 0; j < 3; j++) {
      ASSERT_NEAR(a12_expected[j], a12_actual[j], TEST_DOUBLE_EPSILON);
      ASSERT_NEAR(b12_expected[j], b12_actual[j], TEST_DOUBLE_EPSILON);
    }

    ASSERT_NEAR(u_expected, u_actual, TEST_DOUBLE_EPSILON);
    ASSERT_NEAR(v_expected, v_actual, TEST_DOUBLE_EPSILON);
  }
}

TEST(DistanceBetweenLineSegments, PositiveResult) {
  openrand::Philox rng(generate_test_seed(), 0);
  unsigned nTests = MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST;

  Point<double> a1, a2, b1, b2, a12_expected, a12_actual, b12_expected, b12_actual;
  mundy::math::Vector3d sep_actual;
  double u_expected, v_expected, u_actual, v_actual, dist_expected, dist_actual;
  for (unsigned i = 0; i < nTests; i++) {
    generate_line_segments_at_known_distance(rng, dist_expected, a1, a2, b1, b2, a12_expected, b12_expected, u_expected,
                                             v_expected);
    const LineSegment<double> line_segment_a(a1, a2);
    const LineSegment<double> line_segment_b(b1, b2);
    dist_actual = distance(line_segment_a, line_segment_b, a12_actual, b12_actual, u_actual, v_actual, sep_actual);
    ASSERT_NEAR(dist_expected, dist_actual, TEST_DOUBLE_EPSILON);

    for (unsigned j = 0; j < 3; j++) {
      ASSERT_NEAR(a12_expected[j], a12_actual[j], TEST_DOUBLE_EPSILON);
      ASSERT_NEAR(b12_expected[j], b12_actual[j], TEST_DOUBLE_EPSILON);
    }

    ASSERT_NEAR(u_expected, u_actual, TEST_DOUBLE_EPSILON);
    ASSERT_NEAR(v_expected, v_actual, TEST_DOUBLE_EPSILON);
  }
}

TEST(DistanceBetweenLineSegments, APeskyEdgeCase) {
  // The following pesky edge case is for a colinear rod that caused an untested edge case.
  // TODO(palmerb4): We'll need colinear rods that give each of the 4 possible cases.
  openrand::Philox rng(generate_test_seed(), 0);
  Point<double> a1, a2, b1, b2, a12_expected, a12_actual, b12_expected, b12_actual;
  mundy::math::Vector3d sep_actual;

  double u_expected, v_expected, u_actual, v_actual, dist_expected, dist_actual;
  // Hardcoding a case that I know is wrong.
  a1 = {0.2257294191072674, 0.30159862841764695, 0.12784820133135649};
  b1 = {0.5220039935659887, 0.88764831847472003, -0.2219484914838093};
  a2 = {0.22572948671663273, 0.30159858045792487, 0.1278481814714105};
  b2 = {0.50288066060587278, 0.66779290982621586, -0.5723507723323677};
  a12_expected = {0.22572948671663273, 0.30159858045792487, 0.1278481814714105};
  b12_expected = {0.52067221426302679, 0.87233723836682309, -0.24635106326970288};
  u_expected = 1.0;
  v_expected = 0.069641589451982497;
  dist_expected = 0.74347757392471259;

  const LineSegment<double> line_segment_a(a1, a2);
  const LineSegment<double> line_segment_b(b1, b2);
  dist_actual = distance(line_segment_a, line_segment_b, a12_actual, b12_actual, u_actual, v_actual, sep_actual);
  EXPECT_NEAR(dist_expected, dist_actual, TEST_DOUBLE_EPSILON);

  for (unsigned j = 0; j < 3; j++) {
    EXPECT_NEAR(a12_expected[j], a12_actual[j], TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(b12_expected[j], b12_actual[j], TEST_DOUBLE_EPSILON);
  }

  EXPECT_NEAR(u_expected, u_actual, TEST_DOUBLE_EPSILON);
  EXPECT_NEAR(v_expected, v_actual, TEST_DOUBLE_EPSILON);
}

TEST(DistanceBetweenLineSegments, APeskyEdgeCaseCollinear) {
  // The following pesky edge case is for a colinear rod that caused an untested edge case.
  // TODO(palmerb4): We'll need colinear rods that give each of the 4 possible cases.
  openrand::Philox rng(generate_test_seed(), 0);
  Point<double> a1, a2, b1, b2;
  mundy::math::Vector3d sep_actual;

  double u_expected, v_expected, u_actual, v_actual, dist_expected, dist_actual;
  // Hardcoding a case that I know is wrong.
  a1 = {9.64101615137754, 6, 3.18961417478521};
  b1 = {10.39230484541326, 6, 0.6472696138825587};
  a2 = {9.64101615137754, 6, 8.189614174785209};
  b2 = {10.39230484541326, 6, 5.647269613882559};
  dist_expected = 0.7512886940357237;

  const LineSegment<double> line_segment_a(a1, a2);
  const LineSegment<double> line_segment_b(b1, b2);
  dist_actual = distance(line_segment_a, line_segment_b);
  EXPECT_NEAR(dist_expected, dist_actual, TEST_DOUBLE_EPSILON);

  auto dist_actual_rev = distance(line_segment_b, line_segment_a);
  EXPECT_NEAR(dist_actual_rev, dist_actual, TEST_DOUBLE_EPSILON);
}

TEST(DistanceToLine, PositiveResult) {
  openrand::Philox rng(generate_test_seed(), 0);
  unsigned nTests = MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST;

  Point<double> a1, a2, a12_actual, a12_expected, p;
  double dist_expected, dist_actual, u_actual;
  mundy::math::Vector3d sep_actual;
  for (unsigned i = 0; i < nTests; i++) {
    generate_line_at_known_distance<openrand::Philox>(rng, a1, a2, a12_expected, p, dist_expected);
    const LineSegment<double> line_segment(a1, a2);
    dist_actual = distance(p, line_segment, a12_actual, u_actual, sep_actual);

    ASSERT_NEAR(dist_expected, dist_actual, TEST_DOUBLE_EPSILON);

    for (unsigned j = 0; j < 3; j++) {
      ASSERT_NEAR(a12_expected[j], a12_actual[j], TEST_DOUBLE_EPSILON);
    }
  }
}

}  // namespace

}  // namespace geom

}  // namespace mundy
