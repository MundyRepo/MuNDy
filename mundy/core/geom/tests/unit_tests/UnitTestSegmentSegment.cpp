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
#include <cmath>       // for std::ldexp, std::sqrt, std::hypot, std::abs, std::isfinite
#include <concepts>    // for std::convertible_to
#include <functional>  // for std::hash
#include <limits>      // for std::numeric_limits
#include <string>      // for std::string

// Trilinos includes
#include <Kokkos_Core.hpp>  // for Kokkos::numbers::pi

// Mundy
#include <mundy_geom/distance.hpp>    // for mundy::distance
#include <mundy_geom/primitives.hpp>  // for mundy::Point, mundy::LineSegment
#include <mundy_math/Tolerance.hpp>   // for mundy::get_zero_tolerance
#include <mundy_utils/rng.hpp>        // for mundy::make_philox

/// \brief The following global is used to control the number of samples per test.
/// For unit tests, this number should be kept low to ensure fast test times, but to still give an immediate warning if
/// something went very wrong. For integration tests, we recommend setting this number to 10,000 or more.
#ifndef MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST
#define MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST 10000
#endif

namespace mundy {

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

  const double len_a = mundy::norm(a2 - a1);
  const double len_b = mundy::norm(b2 - b1);

  const double len_a1_to_intersection = mundy::norm(a1 - intersection);
  const double len_b1_to_intersection = mundy::norm(b1 - intersection);

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
  const Point<double> v3 = mundy::cross(v1, v2) * rng.template rand<double>();
  const double norm_v3 = mundy::norm(v3);
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
  const Point<double> v3 = mundy::cross(v1, v2) * rng.template rand<double>();
  const double norm_v3 = mundy::norm(v3);
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

  const bool is_a_degenerate = (a1_to_a12 + a12_to_a2) < mundy::get_zero_tolerance<double>();
  const bool is_b_degenerate = (b1_to_b12 + b12_to_b2) < mundy::get_zero_tolerance<double>();
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
  v2 -= mundy::dot(v2, v1) * v1;
  v2 *= rng.template rand<double>();
  dist = mundy::norm(v2);

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
  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  unsigned nTests = MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST;

  Point<double> a1, a2, b1, b2, a12_expected, a12_actual, b12_expected, b12_actual;
  mundy::Vector3d sep_actual;
  double u_expected, v_expected, u_actual, v_actual, dist_expected, dist_actual;
  for (unsigned i = 0; i < nTests; i++) {
    // (a1, a2) and (b1, b2) are points on the lines and u and v are the archlength measured from the left endpoint.
    // The archlength we use is measured from the center.
    generate_lines_at_known_distance(rng, dist_expected, a1, a2, b1, b2, a12_expected, b12_expected, u_expected,
                                     v_expected);

    const auto center_a = 0.5 * (a1 + a2);
    const auto center_b = 0.5 * (b1 + b2);
    const double len_a = mundy::norm(a2 - a1);
    const double len_b = mundy::norm(b2 - b1);
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
  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  unsigned nTests = MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST;

  Point<double> a1, a2, b1, b2, a12_expected, a12_actual, b12_expected, b12_actual;
  mundy::Vector3d sep_actual;
  double u_expected, v_expected, u_actual, v_actual, dist_expected, dist_actual;
  for (unsigned i = 0; i < nTests; i++) {
    generate_line_segments_at_known_distance(rng, dist_expected, a1, a2, b1, b2, a12_expected, b12_expected, u_expected,
                                             v_expected);
    const LineSegment<double> line_segment_a(a1, a2);
    const LineSegment<double> line_segment_b(b1, b2);
    dist_actual = distance(line_segment_a, line_segment_b, a12_actual, b12_actual, u_actual, v_actual, sep_actual);
    ASSERT_NEAR(dist_expected, dist_actual, TEST_DOUBLE_EPSILON)
        << "Failure for lines :" << line_segment_a << " " << line_segment_b;

    for (unsigned j = 0; j < 3; j++) {
      ASSERT_NEAR(a12_expected[j], a12_actual[j], TEST_DOUBLE_EPSILON)
          << "Failure for lines :" << line_segment_a << " " << line_segment_b;
      ASSERT_NEAR(b12_expected[j], b12_actual[j], TEST_DOUBLE_EPSILON)
          << "Failure for lines :" << line_segment_a << " " << line_segment_b;
    }

    ASSERT_NEAR(u_expected, u_actual, TEST_DOUBLE_EPSILON)
        << "Failure for lines :" << line_segment_a << " " << line_segment_b;
    ASSERT_NEAR(v_expected, v_actual, TEST_DOUBLE_EPSILON)
        << "Failure for lines :" << line_segment_a << " " << line_segment_b;
  }
}

TEST(DistanceBetweenLineSegments, APeskyEdgeCase) {
  // A segment of length 8.5e-8 crossing a long one (sin^2 of the angle 0.79), which must not be classified as parallel.
  Point<double> a1, a2, b1, b2, a12_expected, a12_actual, b12_expected, b12_actual;
  mundy::Vector3d sep_actual;

  double u_expected, v_expected, u_actual, v_actual, dist_expected, dist_actual;
  // The expected parameters are the unique exact minimizer (rational arithmetic on these inputs); the expected closest
  // points and distance lie within TEST_DOUBLE_EPSILON of it.
  a1 = {0.2257294191072674, 0.30159862841764695, 0.12784820133135649};
  b1 = {0.5220039935659887, 0.88764831847472003, -0.2219484914838093};
  a2 = {0.22572948671663273, 0.30159858045792487, 0.1278481814714105};
  b2 = {0.50288066060587278, 0.66779290982621586, -0.5723507723323677};
  a12_expected = {0.22572948671663273, 0.30159858045792487, 0.1278481814714105};
  b12_expected = {0.52067221426302679, 0.87233723836682309, -0.24635106326970288};
  u_expected = 0.9989252263656248;
  v_expected = 0.06964158935038495;
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
  Point<double> a1, a2, b1, b2;

  double dist_expected, dist_actual;
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
  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  unsigned nTests = MUNDY_GEOM_TESTS_UNIT_TESTS_SEGMENT_SEGMENT_DISTANCE_NUM_SAMPLES_PER_TEST;

  Point<double> a1, a2, a12_actual, a12_expected, p;
  double dist_expected, dist_actual, u_actual;
  mundy::Vector3d sep_actual;
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

TEST(DistanceToLineSegment, EveryBranch) {
  // Each case drives one branch to a closest point known exactly on the segment (0,0,0)-(2,0,0), or on a degenerate
  // segment at the origin.
  struct Case {
    const char* branch;
    Point<double> p, p1, p2;  // the point and the segment (p1, p2)
    Point<double> closest;    // closest point on the segment
    double s;                 // its arch-length parameter
  };
  const Case cases[] = {
      {"interior", {0.5, 1, 0}, {0, 0, 0}, {2, 0, 0}, {0.5, 0, 0}, 0.25},
      {"projects onto p1", {0, 1, 0}, {0, 0, 0}, {2, 0, 0}, {0, 0, 0}, 0.0},
      {"before p1", {-1, 1, 0}, {0, 0, 0}, {2, 0, 0}, {0, 0, 0}, 0.0},
      {"past p2", {3, 1, 0}, {0, 0, 0}, {2, 0, 0}, {2, 0, 0}, 1.0},
      {"degenerate segment", {1, 1, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 0.0},
      {"nearly degenerate segment", {1, 1, 0}, {0, 0, 0}, {1e-8, 0, 0}, {1e-8, 0, 0}, 1.0},
  };

  const double tol = 1e-12;
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  for (const Case& c : cases) {
    SCOPED_TRACE(c.branch);
    const LineSegment<double> segment(c.p1, c.p2);
    const mundy::Vector3d sep_expected = c.closest - c.p;
    Point<double> closest{nan, nan, nan};
    double s = nan;
    mundy::Vector3d sep{nan, nan, nan};

    // sep runs from the point to the segment.
    const double dist = distance(c.p, segment, closest, s, sep);
    EXPECT_NEAR(dist, mundy::norm(sep_expected), tol);
    EXPECT_NEAR(distance(c.p, segment), mundy::norm(sep_expected), tol);
    EXPECT_NEAR(s, c.s, tol);
    for (unsigned j = 0; j < 3; j++) {
      EXPECT_NEAR(closest[j], c.closest[j], tol);
      EXPECT_NEAR(sep[j], sep_expected[j], tol);
    }
  }
}

// Two perpendicular segments of length L = 1e-3 a gap 1e-4 apart: small segments must not be classified as parallel.

TEST(DistanceBetweenLineSegments, ScaleInvariantParallelThreshold) {
  const double L = 1e-3;
  const double gap = 1e-4;

  // S1 along x, S2 along y, both centered near the origin, separated by gap.
  // True closest approach: midpoint of each segment, distance = gap.
  const Point<double> l0{-L / 2, 0.0, 0.0};
  const Point<double> l1{L / 2, 0.0, 0.0};
  const Point<double> m0{0.0, -L / 2, gap};
  const Point<double> m1{0.0, L / 2, gap};

  const LineSegment<double> S1(l0, l1);
  const LineSegment<double> S2(m0, m1);

  Point<double> cp1, cp2;
  double al1, al2;
  mundy::Vector3d sep;
  const double dist = distance(S1, S2, cp1, cp2, al1, al2, sep);

  EXPECT_NEAR(dist, gap, 1e-12);
  EXPECT_NEAR(al1, 0.5, 1e-10);
  EXPECT_NEAR(al2, 0.5, 1e-10);
}

TEST(DistanceBetweenLineSegments, EveryBranch) {
  // Each case drives one branch of the closest-point search to a closest pair known exactly. Segment 2 sits a gap h
  // above segment 1; a tilt delta is small enough that the segments count as parallel.
  struct Case {
    const char* branch;
    Point<double> l0, l1, m0, m1;  // segment 1 = (l0, l1), segment 2 = (m0, m1)
    Point<double> cp1, cp2;        // closest points on segments 1 and 2
    double s, t;                   // their arch-length parameters
  };
  const double h = 0.25;
  const double delta = 1e-6;
  const Case cases[] = {
      // Parallel: one of the four endpoints is closest.
      {"parallel, l0 closest",
       {-0.5, 0, 0},
       {0.5, 0, -delta},
       {-1, 0, h},
       {1, 0, h},  //
       {-0.5, 0, 0},
       {-0.5, 0, h},
       0.0,
       0.25},
      {"parallel, l1 closest",
       {-0.5, 0, -delta},
       {0.5, 0, 0},
       {-1, 0, h},
       {1, 0, h},  //
       {0.5, 0, 0},
       {0.5, 0, h},
       1.0,
       0.75},
      {"parallel, m0 closest",
       {-1, 0, 0},
       {1, 0, 0},
       {-0.5, 0, h},
       {0.5, 0, h + delta},  //
       {-0.5, 0, 0},
       {-0.5, 0, h},
       0.25,
       0.0},
      {"parallel, m1 closest",
       {-1, 0, 0},
       {1, 0, 0},
       {-0.5, 0, h + delta},
       {0.5, 0, h},  //
       {0.5, 0, 0},
       {0.5, 0, h},
       0.75,
       1.0},
      {"exactly parallel, disjoint",
       {0, 0, 0},
       {1, 0, 0},
       {1.5, 0, h},
       {2.5, 0, h},  //
       {1, 0, 0},
       {1.5, 0, h},
       1.0,
       0.0},

      // Skew: segment 1 along x, segment 2 along y.
      {"skew, interior",
       {-1, 0, 0},
       {1, 0, 0},
       {0, -1, h},
       {0, 1, h},  //
       {0, 0, 0},
       {0, 0, h},
       0.5,
       0.5},
      {"skew, s = 0",
       {0, 0, 0},
       {1, 0, 0},
       {-0.5, -1, h},
       {-0.5, 1, h},  //
       {0, 0, 0},
       {-0.5, 0, h},
       0.0,
       0.5},
      {"skew, s = 1",
       {0, 0, 0},
       {1, 0, 0},
       {1.5, -1, h},
       {1.5, 1, h},  //
       {1, 0, 0},
       {1.5, 0, h},
       1.0,
       0.5},
      {"skew, t = 0, s interior",
       {-1, 0, 0},
       {1, 0, 0},
       {0, 0.5, h},
       {0, 1.5, h},  //
       {0, 0, 0},
       {0, 0.5, h},
       0.5,
       0.0},
      {"skew, t = 0, s = 0",
       {0, 0, 0},
       {1, 0, 0},
       {-0.5, 0.5, h},
       {-0.5, 1.5, h},  //
       {0, 0, 0},
       {-0.5, 0.5, h},
       0.0,
       0.0},
      {"skew, t = 0, s = 1",
       {0, 0, 0},
       {1, 0, 0},
       {1.5, 0.5, h},
       {1.5, 1.5, h},  //
       {1, 0, 0},
       {1.5, 0.5, h},
       1.0,
       0.0},
      {"skew, t = 1, s interior",
       {-1, 0, 0},
       {1, 0, 0},
       {0, -1.5, h},
       {0, -0.5, h},  //
       {0, 0, 0},
       {0, -0.5, h},
       0.5,
       1.0},
      {"skew, t = 1, s = 0",
       {0, 0, 0},
       {1, 0, 0},
       {-0.5, -1.5, h},
       {-0.5, -0.5, h},  //
       {0, 0, 0},
       {-0.5, -0.5, h},
       0.0,
       1.0},
      {"skew, t = 1, s = 1",
       {0, 0, 0},
       {1, 0, 0},
       {1.5, -1.5, h},
       {1.5, -0.5, h},  //
       {1, 0, 0},
       {1.5, -0.5, h},
       1.0,
       1.0},

      // Oblique: segment 1 along x, segment 2 along (1, 1, 0), so clamping one parameter moves the other.
      {"oblique, t = 0 moves s",
       {0, 0, 0},
       {2, 0, 0},
       {1.5, 0.5, h},
       {2.5, 1.5, h},  //
       {1.5, 0, 0},
       {1.5, 0.5, h},
       0.75,
       0.0},
      {"oblique, t = 1 moves s",
       {0, 0, 0},
       {2, 0, 0},
       {0, -1.5, h},
       {1, -0.5, h},  //
       {1, 0, 0},
       {1, -0.5, h},
       0.5,
       1.0},
      {"oblique, s = 0 moves t",
       {0, 0, 0},
       {2, 0, 0},
       {-1, 0, h},
       {0, 1, h},  //
       {0, 0, 0},
       {-0.5, 0.5, h},
       0.0,
       0.5},
      {"oblique, s = 1 moves t",
       {0, 0, 0},
       {2, 0, 0},
       {2, -1, h},
       {3, 0, h},  //
       {2, 0, 0},
       {2.5, -0.5, h},
       1.0,
       0.5},
      {"oblique, s = 0 then t = 0 moves s",
       {0, 0, 0},
       {2, 0, 0},
       {0.5, 1, h},
       {1.5, 2, h},  //
       {0.5, 0, 0},
       {0.5, 1, h},
       0.25,
       0.0},
      {"oblique, s = 1 then t = 1 moves s",
       {0, 0, 0},
       {2, 0, 0},
       {0.5, -2, h},
       {1.5, -1, h},  //
       {1.5, 0, 0},
       {1.5, -1, h},
       0.75,
       1.0},
  };

  const double tol = 1e-12;
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  for (const Case& c : cases) {
    SCOPED_TRACE(c.branch);
    const LineSegment<double> segment1(c.l0, c.l1);
    const LineSegment<double> segment2(c.m0, c.m1);
    const mundy::Vector3d sep_expected = c.cp2 - c.cp1;
    Point<double> cp1{nan, nan, nan}, cp2{nan, nan, nan};
    double s = nan, t = nan;
    mundy::Vector3d sep{nan, nan, nan};

    // sep runs from segment 1 to segment 2.
    const double dist = distance(segment1, segment2, cp1, cp2, s, t, sep);
    EXPECT_NEAR(dist, mundy::norm(sep_expected), tol);
    EXPECT_NEAR(distance(segment1, segment2), mundy::norm(sep_expected), tol);
    EXPECT_NEAR(s, c.s, tol);
    EXPECT_NEAR(t, c.t, tol);
    for (unsigned j = 0; j < 3; j++) {
      EXPECT_NEAR(cp1[j], c.cp1[j], tol);
      EXPECT_NEAR(cp2[j], c.cp2[j], tol);
      EXPECT_NEAR(sep[j], sep_expected[j], tol);
    }

    // Swapping the segments swaps the closest points and reverses sep.
    cp1 = cp2 = Point<double>{nan, nan, nan};
    s = t = nan;
    sep = mundy::Vector3d{nan, nan, nan};
    const double dist_swapped = distance(segment2, segment1, cp2, cp1, t, s, sep);
    EXPECT_NEAR(dist_swapped, mundy::norm(sep_expected), tol);
    EXPECT_NEAR(distance(segment2, segment1), mundy::norm(sep_expected), tol);
    EXPECT_NEAR(s, c.s, tol);
    EXPECT_NEAR(t, c.t, tol);
    for (unsigned j = 0; j < 3; j++) {
      EXPECT_NEAR(cp1[j], c.cp1[j], tol);
      EXPECT_NEAR(cp2[j], c.cp2[j], tol);
      EXPECT_NEAR(sep[j], -sep_expected[j], tol);
    }
  }
}

//! \name Exact anchors
//@{

// Tolerances. eps = 2^-52 and X is the largest coordinate magnitude among the inputs.
//  - Closest points and separation: start + s (end - start) rounds each coordinate at most twice with s fixed
//    (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1: gamma_2 ~ 2 eps), so each closest
//    point lies within 2 eps X per component of the exact point for its parameter; sep = cp2 - cp1 adds one rounding
//    of at most eps |sep_i| <= eps d. Over three components this is sqrt(3) (4 eps X + eps d) <= 7 eps X + 2 eps d, and
//    |sep| adds at most (n/2 + 1) eps d = 2.5 eps d for n = 3 (Higham, Section 3.1): 7 eps X + 5 eps d in all.
//  - Parameters (empirical): the computed parameter pair minimizes R(s, t) = |l(s) - m(t)|^2 exactly along one
//    coordinate direction each, so its own rounding moves the pair along the valley of R. No bound on that contribution
//    is derived; adversarial testing (angle scans to 1e-20, 150k cases, both FMA builds) measured it at <= 1.4 eps X,
//    and 2 eps X is charged for it.
//  Hence dist_tol(X, d) = 9 eps X + 5 eps d, of which 2 eps X is empirical.
//  - Where the closest points are exactly representable and the parameters are exact (powers of two and their
//    sums), only |sep| rounds: exact_dist_tol(d) = 3 eps d + 2^-1074, the last term being the spacing of subnormals.
//  - A pair within tau of the minimal distance d lies, in (s, t), within sqrt(((d + tau)^2 - d^2) / lambda) of the
//    minimizer, where lambda is the smallest eigenvalue of [[a, -b], [-b, c]], the Hessian of R / 2 (strong
//    convexity of R). Parameters are checked against that bound, and only where the minimizer is unique.

/// \brief The distance tolerance dist_tol(X, d) = 9 eps X + 5 eps d (2 eps X of it empirical)
double dist_tol(const double x, const double d) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  return 9.0 * eps * x + 5.0 * eps * d;
}

/// \brief The distance tolerance for exactly representable closest points, 3 eps d + 2^-1074
double exact_dist_tol(const double d) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  return 3.0 * eps * d + std::numeric_limits<double>::denorm_min();
}

/// \brief The largest coordinate magnitude among four points
double max_coordinate(const Point<double>& p0, const Point<double>& p1, const Point<double>& p2,
                      const Point<double>& p3) {
  double x = 0.0;
  for (unsigned i = 0; i < 3; ++i) {
    x = std::max({x, std::abs(p0[i]), std::abs(p1[i]), std::abs(p2[i]), std::abs(p3[i])});
  }
  return x;
}

/// \brief |v| without overflow or underflow in the squares, for checks at the edges of the double range
double length(const mundy::Vector3d& v) {
  return std::hypot(v[0], v[1], v[2]);
}

/// \brief The smallest eigenvalue of [[u.u, -u.v], [-u.v, v.v]], the curvature of R / 2 along its valley, formed as
/// |u x v|^2 / (largest eigenvalue) so it does not cancel
double valley_curvature(const mundy::Vector3d& u, const mundy::Vector3d& v) {
  const double a = mundy::dot(u, u);
  const double b = mundy::dot(u, v);
  const double c = mundy::dot(v, v);
  const double largest = 0.5 * (a + c) + std::sqrt(0.25 * (a - c) * (a - c) + b * b);
  const mundy::Vector3d n = mundy::cross(u, v);
  return largest > 0.0 ? mundy::dot(n, n) / largest : 0.0;
}

/// \brief The parameter-space radius within which a pair at distance <= d + tau lies, sqrt(((d + tau)^2 - d^2) /
/// lambda), capped at 1 (the parameter range)
double parameter_tol(const double lambda, const double d, const double tau) {
  const double r2 = (2.0 * d * tau + tau * tau) / lambda;
  return (lambda > 0.0 && r2 < 1.0) ? std::sqrt(r2) : 1.0;
}

TEST(DistanceBetweenLineSegments, AngleRefinement) {
  // Segment 1 runs along x from (-1, 0, 0) to (1, 0, 0). Segment 2 runs from (x0 - 1, -delta, h) to (x0 + 1, delta, h),
  // so its line crosses above (x0, 0, 0) at height h, at an angle atan(delta). The minimizer is s = (x0 + 1) / 2,
  // t = 1 / 2, at distance exactly h, for every delta > 0; delta = 0 makes the segments parallel at distance h with a
  // line of minimizers. All coordinates are exactly representable.
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  for (const double x0 : {0.0, 0.5}) {
    for (const double h : {0.0, 0.25}) {
      for (const bool reversed : {false, true}) {
        for (int k = -1; k <= 60; ++k) {
          const double delta = k < 0 ? 0.0 : std::ldexp(1.0, -k);
          SCOPED_TRACE(::testing::Message()
                       << "x0 " << x0 << " h " << h << " reversed " << reversed << " delta 2^-" << k);
          const Point<double> l0{-1.0, 0.0, 0.0};
          const Point<double> l1{1.0, 0.0, 0.0};
          const Point<double> a{x0 - 1.0, -delta, h};
          const Point<double> b{x0 + 1.0, delta, h};
          const Point<double> m0 = reversed ? b : a;
          const Point<double> m1 = reversed ? a : b;
          const LineSegment<double> segment1(l0, l1);
          const LineSegment<double> segment2(m0, m1);

          Point<double> cp1{nan, nan, nan}, cp2{nan, nan, nan};
          double s = nan, t = nan;
          mundy::Vector3d sep{nan, nan, nan};
          const double dist = distance(segment1, segment2, cp1, cp2, s, t, sep);

          // Distance and contract.
          const double x = max_coordinate(l0, l1, m0, m1);
          const double tau = dist_tol(x, h);
          EXPECT_NEAR(dist, h, tau);
          EXPECT_GE(s, 0.0);
          EXPECT_LE(s, 1.0);
          EXPECT_GE(t, 0.0);
          EXPECT_LE(t, 1.0);
          for (unsigned j = 0; j < 3; ++j) {
            EXPECT_EQ(sep[j], cp2[j] - cp1[j]);
            EXPECT_NEAR(cp1[j], l0[j] + s * (l1[j] - l0[j]), 2.0 * std::numeric_limits<double>::epsilon() * x);
            EXPECT_NEAR(cp2[j], m0[j] + t * (m1[j] - m0[j]), 2.0 * std::numeric_limits<double>::epsilon() * x);
          }
          EXPECT_NEAR(dist, mundy::norm(sep), 3.0 * std::numeric_limits<double>::epsilon() * dist);

          // The returned pair attains the distance.
          const Point<double> p = l0 + s * (l1 - l0);
          const Point<double> q = m0 + t * (m1 - m0);
          EXPECT_NEAR(length(q - p), h, tau);

          // Parameters, where the minimizer is unique.
          if (delta > 0.0) {
            const double r = parameter_tol(valley_curvature(l1 - l0, m1 - m0), h, tau);
            EXPECT_NEAR(s, 0.5 * (x0 + 1.0), r);
            EXPECT_NEAR(t, 0.5, r);
          }
        }
      }
    }
  }
}

TEST(DistanceBetweenLineSegments, ScaleInvariance) {
  // Scaling every coordinate by 2^k leaves the parameters unchanged and scales the closest points, separation, and
  // distance by exactly 2^k, as long as nothing is subnormal.
  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  const unsigned num_samples = 2000;
  for (unsigned n = 0; n < num_samples; ++n) {
    Point<double> l0, l1, m0, m1;
    for (unsigned i = 0; i < 3; ++i) {
      l0[i] = rng.template rand<double>();
      l1[i] = rng.template rand<double>();
      m0[i] = rng.template rand<double>();
      m1[i] = rng.template rand<double>();
    }
    if (n % 4 == 1) {
      m1 = m0 + (l1 - l0) * 0.5;  // parallel
    } else if (n % 4 == 2) {
      m0 = l1;  // shared endpoint
    } else if (n % 4 == 3) {
      m1 = m0;  // degenerate
    }
    Point<double> cp1, cp2;
    double s, t;
    mundy::Vector3d sep;
    const double dist = distance(LineSegment<double>(l0, l1), LineSegment<double>(m0, m1), cp1, cp2, s, t, sep);

    for (const int k : {-900, -300, -1, 1, 300, 900}) {
      SCOPED_TRACE(::testing::Message() << "sample " << n << " k " << k);
      const double f = std::ldexp(1.0, k);
      Point<double> cp1k, cp2k;
      double sk, tk;
      mundy::Vector3d sepk;
      const double distk =
          distance(LineSegment<double>(l0 * f, l1 * f), LineSegment<double>(m0 * f, m1 * f), cp1k, cp2k, sk, tk, sepk);
      EXPECT_EQ(distk, dist * f);
      EXPECT_EQ(sk, s);
      EXPECT_EQ(tk, t);
      for (unsigned j = 0; j < 3; ++j) {
        EXPECT_EQ(cp1k[j], cp1[j] * f);
        EXPECT_EQ(cp2k[j], cp2[j] * f);
        EXPECT_EQ(sepk[j], sep[j] * f);
      }
    }
  }
}

TEST(DistanceBetweenLineSegments, Translation) {
  // Translating by T = 2^30 L keeps the exact answer, since every translated coordinate stays exactly representable
  // (all coordinates are multiples of 2^-22 with magnitude below 2^31). The closest points are rounded to the spacing
  // of coordinates near T, so the distance is accurate to dist_tol(T + L, d).
  struct Case {
    const char* name;
    Point<double> l0, l1, m0, m1;
    double s, t, d;
  };
  const double h = 0.25;
  const double delta = std::ldexp(1.0, -20);
  const double d_end = std::sqrt(0.25 + h * h);     // from (1, 0, 0) to (1.5, 0, h)
  const double d_oblique = std::sqrt(1.0 + h * h);  // from (0.5, 0, 0) to (0.5, 1, h)
  const Case cases[] = {
      {"skew, interior", {-1, 0, 0}, {1, 0, 0}, {0, -1, h}, {0, 1, h}, 0.5, 0.5, h},
      {"skew, s = 1", {0, 0, 0}, {1, 0, 0}, {1.5, -1, h}, {1.5, 1, h}, 1.0, 0.5, d_end},
      {"oblique, s = 0 then t = 0 moves s", {0, 0, 0}, {2, 0, 0}, {0.5, 1, h}, {1.5, 2, h}, 0.25, 0.0, d_oblique},
      {"near parallel, interior", {-1, 0, 0}, {1, 0, 0}, {-0.5, -delta, h}, {1.5, delta, h}, 0.75, 0.5, h},
      {"near parallel, crossing", {-1, 0, 0}, {1, 0, 0}, {-1, -delta, 0}, {1, delta, 0}, 0.5, 0.5, 0.0},
  };
  const double scale_l = 2.0;
  const mundy::Vector3d shift = std::ldexp(1.0, 30) * scale_l * mundy::Vector3d{1.0, -1.0, 0.5};
  for (const Case& c : cases) {
    for (const double sign : {1.0, -1.0}) {
      SCOPED_TRACE(::testing::Message() << c.name << " sign " << sign);
      const Point<double> l0 = c.l0 + sign * shift;
      const Point<double> l1 = c.l1 + sign * shift;
      const Point<double> m0 = c.m0 + sign * shift;
      const Point<double> m1 = c.m1 + sign * shift;
      Point<double> cp1, cp2;
      double s, t;
      mundy::Vector3d sep;
      const double dist = distance(LineSegment<double>(l0, l1), LineSegment<double>(m0, m1), cp1, cp2, s, t, sep);

      const double tau = dist_tol(max_coordinate(l0, l1, m0, m1), c.d);
      EXPECT_NEAR(dist, c.d, tau);
      for (unsigned j = 0; j < 3; ++j) {
        EXPECT_EQ(sep[j], cp2[j] - cp1[j]);
      }
      const double r = parameter_tol(valley_curvature(l1 - l0, m1 - m0), c.d, tau);
      EXPECT_NEAR(s, c.s, r);
      EXPECT_NEAR(t, c.t, r);
    }
  }
}

TEST(DistanceBetweenLineSegments, ParallelLargeLength) {
  // Exactly parallel segments of length L, a height L / 8 apart, for L up to 2^1022. Overlapping projections have a
  // line of minimizers at distance L / 8; disjoint ones have the unique minimizer at the facing endpoints, at distance
  // L sqrt(1/16 + 1/64) = L sqrt(5) / 8.
  for (const int k : {0, 100, 511, 600, 900, 1000, 1022}) {
    const double L = std::ldexp(1.0, k);
    const double h = 0.125 * L;
    for (const bool reversed : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "L 2^" << k << " reversed " << reversed);
      const Point<double> l0{0.0, 0.0, 0.0};
      const Point<double> l1{L, 0.0, 0.0};

      // Overlapping: distance only, and that the returned pair attains it.
      {
        const Point<double> a{0.5 * L, h, 0.0};
        const Point<double> b{1.5 * L, h, 0.0};
        const Point<double> m0 = reversed ? b : a;
        const Point<double> m1 = reversed ? a : b;
        Point<double> cp1, cp2;
        double s, t;
        mundy::Vector3d sep;
        const double dist = distance(LineSegment<double>(l0, l1), LineSegment<double>(m0, m1), cp1, cp2, s, t, sep);
        const double tau = dist_tol(max_coordinate(l0, l1, m0, m1), h);
        EXPECT_NEAR(dist, h, tau);
        EXPECT_NEAR(length((m0 + t * (m1 - m0)) - (l0 + s * (l1 - l0))), h, tau);
      }

      // Disjoint: unique minimizer (l1, near endpoint of segment 2).
      {
        const Point<double> a{1.25 * L, h, 0.0};
        const Point<double> b{1.75 * L, h, 0.0};
        const Point<double> m0 = reversed ? b : a;
        const Point<double> m1 = reversed ? a : b;
        Point<double> cp1, cp2;
        double s, t;
        mundy::Vector3d sep;
        const double dist = distance(LineSegment<double>(l0, l1), LineSegment<double>(m0, m1), cp1, cp2, s, t, sep);
        const double d = L * (std::sqrt(5.0) / 8.0);
        const double tau = dist_tol(max_coordinate(l0, l1, m0, m1), d);
        EXPECT_NEAR(dist, d, tau);
        EXPECT_EQ(s, 1.0);
        EXPECT_EQ(t, reversed ? 1.0 : 0.0);
      }
    }
  }
}

TEST(DistanceBetweenLineSegments, SharedEndpoint) {
  // Two segments that share an endpoint exactly are at distance 0, attained only at that endpoint unless they are
  // collinear. Directions run from perpendicular to within 2^-52 of collinear, continuing or folding back, at lengths
  // 2^-20 to 2^20, in all four endpoint pairings.
  const Point<double> P{0.5, -0.25, 0.125};
  for (int k = 0; k <= 52; k += 4) {
    const double delta = std::ldexp(1.0, -k);
    const mundy::Vector3d directions[] = {{-1.0, delta, 0.0}, {1.0, delta, 0.0}, {0.0, 1.0, delta}};
    for (const mundy::Vector3d& dir : directions) {
      for (const double len : {std::ldexp(1.0, -20), 1.0, std::ldexp(1.0, 20)}) {
        const Point<double> A = P + mundy::Vector3d{1.0, 0.0, 0.0};
        const Point<double> B = P + len * dir;
        for (int pairing = 0; pairing < 4; ++pairing) {
          SCOPED_TRACE(::testing::Message()
                       << "delta 2^-" << k << " dir " << dir << " len " << len << " pairing " << pairing);
          const bool flip1 = pairing & 1;
          const bool flip2 = pairing & 2;
          const Point<double> l0 = flip1 ? A : P;
          const Point<double> l1 = flip1 ? P : A;
          const Point<double> m0 = flip2 ? B : P;
          const Point<double> m1 = flip2 ? P : B;
          Point<double> cp1, cp2;
          double s, t;
          mundy::Vector3d sep;
          const double dist = distance(LineSegment<double>(l0, l1), LineSegment<double>(m0, m1), cp1, cp2, s, t, sep);

          const double tau = dist_tol(max_coordinate(l0, l1, m0, m1), 0.0);
          EXPECT_LE(dist, tau);
          for (unsigned j = 0; j < 3; ++j) {
            EXPECT_EQ(sep[j], cp2[j] - cp1[j]);
          }
          const double r = parameter_tol(valley_curvature(l1 - l0, m1 - m0), 0.0, tau);
          EXPECT_NEAR(s, flip1 ? 1.0 : 0.0, r);
          EXPECT_NEAR(t, flip2 ? 1.0 : 0.0, r);
        }
      }
    }
  }
}

TEST(DistanceBetweenLineSegments, ExtremeScaleSpecialCases) {
  // Degenerate, collinear, and T-junction configurations with exact answers, scaled by 2^k up to 2^+-1000. Where the
  // minimizer is a line, only the distance and that the returned pair attains it are checked.
  struct Case {
    const char* name;
    Point<double> l0, l1, m0, m1;
    double d;
    bool unique;
    double s, t;
  };
  const Case cases[] = {
      {"point beside a segment", {0.5, 1, 0}, {0.5, 1, 0}, {0, 0, 0}, {2, 0, 0}, 1.0, true, 0.0, 0.25},
      {"segment beside a point", {0, 0, 0}, {2, 0, 0}, {0.5, 1, 0}, {0.5, 1, 0}, 1.0, true, 0.25, 0.0},
      {"two points", {1, 2, 3}, {1, 2, 3}, {4, 6, 3}, {4, 6, 3}, 5.0, true, 0.0, 0.0},
      {"collinear, overlapping", {0, 0, 0}, {2, 0, 0}, {1, 0, 0}, {3, 0, 0}, 0.0, false, 0.0, 0.0},
      {"collinear, gap", {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, 1.0, true, 1.0, 0.0},
      {"collinear, gap, reversed", {0, 0, 0}, {1, 0, 0}, {3, 0, 0}, {2, 0, 0}, 1.0, true, 1.0, 1.0},
      {"collinear, touching", {0, 0, 0}, {1, 0, 0}, {1, 0, 0}, {2, 0, 0}, 0.0, true, 1.0, 0.0},
      {"identical", {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, 0.0, false, 0.0, 0.0},
      {"identical, reversed", {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 0.0, false, 0.0, 0.0},
      {"T-junction", {0, 0, 0}, {2, 0, 0}, {1, 0, 0}, {1, 1, 0}, 0.0, true, 0.5, 0.0},
      {"T-junction, gap", {0, 0, 0}, {2, 0, 0}, {1, 0.5, 0}, {1, 1, 0}, 0.5, true, 0.5, 0.0},
  };
  for (const Case& c : cases) {
    for (const int k : {-1000, -600, -1, 0, 1, 600, 1000}) {
      SCOPED_TRACE(::testing::Message() << c.name << " k " << k);
      const double f = std::ldexp(1.0, k);
      const Point<double> l0 = c.l0 * f;
      const Point<double> l1 = c.l1 * f;
      const Point<double> m0 = c.m0 * f;
      const Point<double> m1 = c.m1 * f;
      Point<double> cp1, cp2;
      double s, t;
      mundy::Vector3d sep;
      const double dist = distance(LineSegment<double>(l0, l1), LineSegment<double>(m0, m1), cp1, cp2, s, t, sep);

      const double d = c.d * f;
      const double tau = dist_tol(max_coordinate(l0, l1, m0, m1), d);
      EXPECT_NEAR(dist, d, tau);
      for (unsigned j = 0; j < 3; ++j) {
        EXPECT_EQ(sep[j], cp2[j] - cp1[j]);
      }
      EXPECT_NEAR(length((m0 + t * (m1 - m0)) - (l0 + s * (l1 - l0))), d, tau);
      if (c.unique) {
        EXPECT_EQ(s, c.s);
        EXPECT_EQ(t, c.t);
      }
    }
  }
}

TEST(DistanceBetweenLineSegments, RangeLimits) {
  // Exactly representable closest points at the edges of the double range: the distance is exact to the rounding of
  // |sep| (exact_dist_tol).
  constexpr double max = std::numeric_limits<double>::max();
  struct Case {
    const char* name;
    Point<double> l0, l1, m0, m1;
    double d;
  };
  const double big = std::ldexp(1.0, 1000);
  const double small = std::ldexp(1.0, -1000);
  const double big_h = 0.25 * big;
  const double small_h = 0.25 * small;
  const double lo = 0.4 * max;
  const double hi = 0.5 * max;
  const double near_max = 0.9 * max;
  const double a = 0.9 * max;               // endpoint differences overflow
  const double g = std::ldexp(1.0, -1060);  // a subnormal gap
  const Case cases[] = {
      {"skew X at 2^1000", {-big, 0, 0}, {big, 0, 0}, {0, -big, big_h}, {0, big, big_h}, big_h},
      {"skew X at 2^-1000", {-small, 0, 0}, {small, 0, 0}, {0, -small, small_h}, {0, small, small_h}, small_h},
      {"differences overflow", {-a, 0, 0}, {a, 0, 0}, {0, -a, 0.5 * max}, {0, a, 0.5 * max}, 0.5 * max},
      {"distance near the maximum", {-lo, 0, 0}, {-lo, 0, 0}, {hi, 0, 0}, {hi, 0, 0}, near_max},
      {"subnormal gap", {-1, 0, 0}, {1, 0, 0}, {0, -1, g}, {0, 1, g}, g},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Point<double> cp1, cp2;
    double s, t;
    mundy::Vector3d sep;
    const double dist = distance(LineSegment<double>(c.l0, c.l1), LineSegment<double>(c.m0, c.m1), cp1, cp2, s, t, sep);
    EXPECT_NEAR(dist, c.d, exact_dist_tol(c.d));
    for (unsigned j = 0; j < 3; ++j) {
      EXPECT_TRUE(std::isfinite(cp1[j]) && std::isfinite(cp2[j]) && std::isfinite(sep[j]));
    }
  }
}

TEST(DistanceBetweenLineSegments, SmallSegmentRatioLimit) {
  // A segment of half-length 2^-q crossing the midpoint of one of half-length 2^p, a gap 2^(-q-10) above it, at
  // length ratio 2^(p+q). Up to 2^1020 the small segment is resolved and the exactly representable answer is returned
  // (exact_dist_tol). Beyond, the small segment may be treated as the point at its start, which costs at most its
  // length 2^(1-q).
  for (const int r : {0, 100, 537, 538, 1000, 1020, 1021, 1060, 2000}) {
    const int p = r / 2;
    const int q = r - p;
    const double big = std::ldexp(1.0, p);
    const double b = std::ldexp(1.0, -q);
    const double g = std::ldexp(1.0, -q - 10);
    for (const bool small_first : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "ratio 2^" << r << " small first " << small_first);
      const Point<double> L0{-big, 0.0, 0.0}, L1{big, 0.0, 0.0}, S0{0.0, -b, g}, S1{0.0, b, g};
      const Point<double>& l0 = small_first ? S0 : L0;
      const Point<double>& l1 = small_first ? S1 : L1;
      const Point<double>& m0 = small_first ? L0 : S0;
      const Point<double>& m1 = small_first ? L1 : S1;
      Point<double> cp1, cp2;
      double s, t;
      mundy::Vector3d sep;
      const double dist = distance(LineSegment<double>(l0, l1), LineSegment<double>(m0, m1), cp1, cp2, s, t, sep);
      if (r <= 1020) {
        EXPECT_NEAR(dist, g, exact_dist_tol(g));
        EXPECT_EQ(s, 0.5);
        EXPECT_EQ(t, 0.5);
      } else {
        EXPECT_GE(dist, g - exact_dist_tol(g));
        EXPECT_LE(dist, g + 2.0 * b + exact_dist_tol(g + 2.0 * b));
      }
    }
  }
}

TEST(DistanceBetweenLines, ParallelAndNonUnitDirections) {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();

  // Parallel distinct lines are at their true distance, with every output filled and the returned pair attaining it
  // (any pair of facing points minimizes, so only that is checked). Coincident lines are at distance 0.
  {
    struct Case {
      const char* name;
      Point<double> c1;
      mundy::Vector3d d1;
      Point<double> c2;
      mundy::Vector3d d2;
      double d;
    };
    const double big = std::ldexp(1.0, 700);
    const double big_dir = std::ldexp(1.0, 600);
    // A direction (ox, oy, 0) with full-precision components and its half: u x v is zero, but a fused multiply-add can
    // round it to a tiny nonzero value. The offset (-oy, ox, 1/2) is perpendicular to it. Reading the components
    // through volatile keeps the products at run time, where they may be fused, rather than folded at compile time.
    const volatile double ox_runtime = 0x1.7f817c2d0d048p-1;
    const volatile double oy_runtime = -0x1.11fd50266e476p-1;
    const double ox = ox_runtime;
    const double oy = oy_runtime;
    const double d_oblique = std::sqrt(ox * ox + oy * oy + 0.25);
    const Case cases[] = {
        {"parallel", {0, 0, 0}, {1, 0, 0}, {5, 0.5, 0}, {1, 0, 0}, 0.5},
        {"antiparallel, non-unit", {0, 0, 0}, {0.25, 0, 0}, {5, 0.5, 0.5}, {-8, 0, 0}, std::sqrt(0.5)},
        {"parallel, large", {0, 0, 0}, {0, 0, big_dir}, {big, 0, 3}, {0, 0, 1}, big},
        {"coincident", {0, 0, 0}, {1, 1, 0}, {2, 2, 0}, {-3, -3, 0}, 0.0},
        {"parallel, oblique", {0, 0, 0}, {ox, oy, 0}, {-oy, ox, 0.5}, {0.5 * ox, 0.5 * oy, 0}, d_oblique},
    };
    for (const Case& c : cases) {
      SCOPED_TRACE(c.name);
      Point<double> cp1{nan, nan, nan}, cp2{nan, nan, nan};
      double s = nan, t = nan;
      mundy::Vector3d sep{nan, nan, nan};
      const double dist = distance(Line<double>(c.c1, c.d1), Line<double>(c.c2, c.d2), cp1, cp2, s, t, sep);
      // The distance is accurate to the inputs' scale, |c1| + |c2|. Any pair of facing points minimizes, so the
      // returned pair's own coordinates set the rounding of the checks made on it.
      const double tau = dist_tol(length(c.c1) + length(c.c2), c.d);
      EXPECT_NEAR(dist, c.d, tau);
      const double x = std::max({length(c.c1), length(c.c2), length(cp1), length(cp2)});
      EXPECT_NEAR(length((c.c2 + t * c.d2) - (c.c1 + s * c.d1)), c.d, dist_tol(x, c.d));
      for (unsigned j = 0; j < 3; ++j) {
        EXPECT_EQ(sep[j], cp2[j] - cp1[j]);
        EXPECT_NEAR(cp1[j], c.c1[j] + s * c.d1[j], 2.0 * std::numeric_limits<double>::epsilon() * x);
        EXPECT_NEAR(cp2[j], c.c2[j] + t * c.d2[j], 2.0 * std::numeric_limits<double>::epsilon() * x);
      }
    }
  }

  // Skew lines with non-unit directions alpha e_x and beta (1, 1, 0) or beta e_y: the closest points are (x0, 0, 0) and
  // (x0, 0, h) for line 2 through (x0, 0, h), so s = (x0 - 1/4) / alpha and t is the offset of the center along line 2
  // over beta. Powers of two keep every quantity exact. A zero direction makes line 2 the point at its center.
  for (const double alpha : {std::ldexp(1.0, -40), 2.0, std::ldexp(1.0, 40)}) {
    for (const double beta : {std::ldexp(1.0, -30), 0.5, std::ldexp(1.0, 30)}) {
      for (const bool oblique : {false, true}) {
        for (const double h : {0.0, 0.25}) {
          SCOPED_TRACE(::testing::Message()
                       << "alpha " << alpha << " beta " << beta << " oblique " << oblique << " h " << h);
          const double x0 = 1.5;
          const Point<double> c1{0.25, 0.0, 0.0};
          const mundy::Vector3d d1{alpha, 0.0, 0.0};
          const mundy::Vector3d d2 = oblique ? mundy::Vector3d{beta, beta, 0.0} : mundy::Vector3d{0.0, beta, 0.0};
          const Point<double> c2 = Point<double>{x0, 0.0, h} + 3.0 * d2;  // the closest point is at t = -3
          Point<double> cp1, cp2;
          double s, t;
          mundy::Vector3d sep;
          const double dist = distance(Line<double>(c1, d1), Line<double>(c2, d2), cp1, cp2, s, t, sep);
          const double s_exact = (x0 - 0.25) / alpha;
          const double x = std::max({length(c1), length(c2), x0});
          const double tau = dist_tol(x, h);
          EXPECT_NEAR(dist, h, tau);
          for (unsigned j = 0; j < 3; ++j) {
            EXPECT_EQ(sep[j], cp2[j] - cp1[j]);
          }
          // Parameters in units of length along each line, where the curvature is that of R in those units.
          const double r = parameter_tol(valley_curvature(d1 / alpha, d2 / length(d2)), h, tau);
          EXPECT_NEAR(s * alpha, s_exact * alpha, r);
          EXPECT_NEAR(t * length(d2), -3.0 * length(d2), r);
        }
      }
      {
        SCOPED_TRACE(::testing::Message() << "point line, alpha " << alpha);
        const Point<double> c1{0.25, 0.0, 0.0};
        const Point<double> c2{1.5, 2.0, 0.0};
        Point<double> cp1, cp2;
        double s, t;
        mundy::Vector3d sep;
        const double dist = distance(Line<double>(c1, mundy::Vector3d{alpha, 0.0, 0.0}),
                                     Line<double>(c2, mundy::Vector3d{0.0, 0.0, 0.0}), cp1, cp2, s, t, sep);
        EXPECT_NEAR(dist, 2.0, dist_tol(2.0, 2.0));
        EXPECT_EQ(t, 0.0);
        EXPECT_NEAR(s * alpha, 1.25, dist_tol(2.0, 2.0));
      }
    }
  }

  // Near-parallel lines in the planes z = 0 and z = h, at angle atan(delta), crossing above (x0, 0, 0) with x0 = 2^20:
  // the distance is h for every delta > 0.
  for (int k = 0; k <= 60; ++k) {
    const double delta = std::ldexp(1.0, -k);
    SCOPED_TRACE(::testing::Message() << "near parallel, delta 2^-" << k);
    const double x0 = std::ldexp(1.0, 20);
    const double h = 0.25;
    const Point<double> c1{0.0, 0.0, 0.0};
    const mundy::Vector3d d1{1.0, 0.0, 0.0};
    const mundy::Vector3d d2{1.0, delta, 0.0};
    const Point<double> c2{x0 - 1.0, -delta, h};
    Point<double> cp1, cp2;
    double s, t;
    mundy::Vector3d sep;
    const double dist = distance(Line<double>(c1, d1), Line<double>(c2, d2), cp1, cp2, s, t, sep);
    EXPECT_NEAR(dist, h, dist_tol(x0 + 1.0, h));
  }
}

TEST(DistanceToLineSegment, SmallScale) {
  // The point-segment cases with exact answers, scaled by 2^k down to 2^-1000 and up to 2^1000: the parameter is
  // invariant, the closest point and distance scale by exactly 2^k, and the distance is within dist_tol of exact. The
  // point (2^-60, 1, 0) projects to s = 2^-61 on (0, 0, 0)-(2, 0, 0), and the near-degenerate segment resolves too.
  struct Case {
    const char* name;
    Point<double> p, p1, p2;
    Point<double> closest;
    double s;
  };
  const double x = std::ldexp(1.0, -60);
  const Case cases[] = {
      {"interior", {0.5, 1, 0}, {0, 0, 0}, {2, 0, 0}, {0.5, 0, 0}, 0.25},
      {"tiny projection", {x, 1, 0}, {0, 0, 0}, {2, 0, 0}, {x, 0, 0}, 0.5 * x},
      {"projects onto p1", {0, 1, 0}, {0, 0, 0}, {2, 0, 0}, {0, 0, 0}, 0.0},
      {"before p1", {-1, 1, 0}, {0, 0, 0}, {2, 0, 0}, {0, 0, 0}, 0.0},
      {"past p2", {3, 1, 0}, {0, 0, 0}, {2, 0, 0}, {2, 0, 0}, 1.0},
      {"on the segment", {1.5, 0, 0}, {0, 0, 0}, {2, 0, 0}, {1.5, 0, 0}, 0.75},
      {"degenerate segment", {1, 1, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 0.0},
      {"nearly degenerate segment", {1, 1, 0}, {0, 0, 0}, {1e-8, 0, 0}, {1e-8, 0, 0}, 1.0},
  };
  for (const Case& c : cases) {
    Point<double> closest0;
    double s0;
    mundy::Vector3d sep0;
    const double dist0 = distance(c.p, LineSegment<double>(c.p1, c.p2), closest0, s0, sep0);
    for (const int k : {-1000, -600, -100, -1, 0, 1, 100, 600, 1000}) {
      SCOPED_TRACE(::testing::Message() << c.name << " k " << k);
      const double f = std::ldexp(1.0, k);
      Point<double> closest;
      double s;
      mundy::Vector3d sep;
      const double dist = distance(c.p * f, LineSegment<double>(c.p1 * f, c.p2 * f), closest, s, sep);
      const double d = length(c.closest - c.p) * f;
      EXPECT_NEAR(dist, d, dist_tol(max_coordinate(c.p * f, c.p1 * f, c.p2 * f, c.p2 * f), d));
      EXPECT_EQ(s, c.s);
      EXPECT_EQ(s, s0);
      EXPECT_EQ(dist, dist0 * f);
      for (unsigned j = 0; j < 3; ++j) {
        EXPECT_EQ(closest[j], c.closest[j] * f);
        EXPECT_EQ(sep[j], closest[j] - c.p[j] * f);
      }
    }
  }
}
//@}

}  // namespace

}  // namespace mundy
