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
#include <chrono>      // for std::chrono
#include <cmath>       // for std::sqrt, std::isnan
#include <concepts>    // for std::convertible_to
#include <functional>  // for std::hash
#include <string>      // for std::string

// Trilinos includes
#include <Kokkos_Core.hpp>  // for Kokkos::numbers::pi

// Mundy
#include <mundy_geom/distance.hpp>        // for mundy::distance(ellipsoid, ellipsoid)
#include <mundy_geom/primitives.hpp>      // for mundy::Ellipsoid
#include <mundy_math/AutoDiffScalar.hpp>  // for mundy::AutoDiffScalar
#include <mundy_math/Tolerance.hpp>       // for mundy::get_zero_tolerance
#include <mundy_utils/rng.hpp>            // for make_philox

/// \brief The following global is used to control the number of samples per test.
/// For unit tests, this number should be kept low to ensure fast test times, but to still give an immediate warning if
/// something went very wrong. For integration tests, we recommend setting this number to 10,000 or more.
#ifndef MUNDY_GEOM_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST
#define MUNDY_GEOM_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST 10000
#endif

namespace mundy {

namespace {

// After numerous tests, the best precision we can get for the random sphere test is 1.e-4
const double TEST_DOUBLE_EPSILON = 1.e-4;

// Utility function to generate a unique seed for each test based on its GTEST name.
size_t generate_test_seed() {
  const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string test_identifier = std::string(test_info->test_suite_name()) + "." + test_info->name();
  return std::hash<std::string>{}(test_identifier);
}

TEST(SharedNormalDistanceBetweenEllipsoidAndPoint, AnalyticalSphereTestCases) {
  // Spheres admit an analytical signed separation distance. We can generate N random spheres with random positions,
  // radii, and orientations and check that the numerical signed separation distance matches the analytical result.

  auto perform_test_for_given_spheres = [](const Point<double>& center, const mundy::Quaterniond& orientation,
                                           const double r, const Point<double>& point) {
    const Ellipsoid<double> ellipsoid(center, orientation, r, r, r);
    const double shared_normal_ssd = distance(SharedNormalSigned{}, point, ellipsoid);
    const double expected_ssd = mundy::norm(point - center) - r;

    // Assert used to avoid 10 million throws
    ASSERT_NEAR(shared_normal_ssd, expected_ssd, TEST_DOUBLE_EPSILON);
  };

  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  const double min_xyz = -10.0;
  const double max_xyz = 10.0;
  const double range_xyz = max_xyz - min_xyz;
  const double min_r = 0.1;
  const double max_r = 10.0;
  const double range_r = max_r - min_r;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  for (size_t i = 0; i < MUNDY_GEOM_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST; ++i) {
    const Point<double> center = {rng.rand<double>() * range_xyz + min_xyz,  //
                                  rng.rand<double>() * range_xyz + min_xyz,  //
                                  rng.rand<double>() * range_xyz + min_xyz};
    const auto orientation = mundy::euler_to_quat(rng.rand<double>() * 2.0 * pi,  //
                                                  rng.rand<double>() * 2.0 * pi,  //
                                                  rng.rand<double>() * 2.0 * pi);
    const double r = rng.rand<double>() * range_r + min_r;

    const Point<double> point = {rng.rand<double>() * range_xyz + min_xyz,  //
                                 rng.rand<double>() * range_xyz + min_xyz,  //
                                 rng.rand<double>() * range_xyz + min_xyz};
    perform_test_for_given_spheres(center, orientation, r, point);
  }
}

TEST(SharedNormalDistanceBetweenEllipsoids, AnalyticalSphereTestCases) {
  // Spheres admit an analytical signed separation distance. We can generate N random spheres with random positions,
  // radii, and orientations and check that the numerical signed separation distance matches the analytical result.

  auto perform_test_for_given_spheres = [](const Point<double>& center0, const mundy::Quaterniond& orientation0,
                                           const double r0, const Point<double>& center1,
                                           const mundy::Quaterniond& orientation1, const double r1) {
    const Ellipsoid<double> ellipsoid0(center0, orientation0, r0, r0, r0);
    const Ellipsoid<double> ellipsoid1(center1, orientation1, r1, r1, r1);
    const double shared_normal_ssd = distance(SharedNormalSigned{}, ellipsoid0, ellipsoid1);
    const double expected_ssd = mundy::norm(center1 - center0) - r0 - r1;

    // Assert used to avoid 10 million throws
    ASSERT_NEAR(shared_normal_ssd, expected_ssd, TEST_DOUBLE_EPSILON);
  };

  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  const double min_xyz = -10.0;
  const double max_xyz = 10.0;
  const double range_xyz = max_xyz - min_xyz;
  const double min_r = 0.1;
  const double max_r = 10.0;
  const double range_r = max_r - min_r;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  for (size_t i = 0; i < MUNDY_GEOM_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST; ++i) {
    const Point<double> center0 = {rng.rand<double>() * range_xyz + min_xyz,  //
                                   rng.rand<double>() * range_xyz + min_xyz,  //
                                   rng.rand<double>() * range_xyz + min_xyz};
    const auto orientation0 = mundy::euler_to_quat(rng.rand<double>() * 2.0 * pi,  //
                                                   rng.rand<double>() * 2.0 * pi,  //
                                                   rng.rand<double>() * 2.0 * pi);
    const double r0 = rng.rand<double>() * range_r + min_r;

    const Point<double> center1 = {rng.rand<double>() * range_xyz + min_xyz,  //
                                   rng.rand<double>() * range_xyz + min_xyz,  //
                                   rng.rand<double>() * range_xyz + min_xyz};
    const auto orientation1 = mundy::euler_to_quat(rng.rand<double>() * 2.0 * pi,  //
                                                   rng.rand<double>() * 2.0 * pi,  //
                                                   rng.rand<double>() * 2.0 * pi);
    const double r1 = rng.rand<double>() * range_r + min_r;

    perform_test_for_given_spheres(center0, orientation0, r0, center1, orientation1, r1);
  }
}

TEST(SharedNormalDistanceBetweenEllipsoids, AnalyticalEllipsoidTestCases) {
  // There are a few cases where we can analytically compute the shared normal signed separation distance between two
  // ellipsoids.

  // Case 1: Perfect overlap
  {
    const auto center0 = Point<double>(0.0, 0.0, 0.0);
    const auto orientation0 = mundy::Quaterniond::identity();
    const double r1_0 = 3.0;
    const double r2_0 = 1.0;
    const double r3_0 = 2.0;
    const Ellipsoid<double> ellipsoid0(center0, orientation0, r1_0, r2_0, r3_0);

    const auto center1 = Point<double>(0.0, 0.0, 0.0);
    const auto orientation1 = mundy::Quaterniond::identity();
    const double r1_1 = r1_0;
    const double r2_1 = r2_0;
    const double r3_1 = r3_0;
    const Ellipsoid<double> ellipsoid1(center1, orientation1, r1_1, r2_1, r3_1);

    const double shared_normal_ssd = distance(SharedNormalSigned{}, ellipsoid0, ellipsoid1);
    EXPECT_NEAR(shared_normal_ssd, -2 * r2_0, TEST_DOUBLE_EPSILON);
  }

  // Case 2: Same centers/orientations but one scaled up by a factor of 2
  {
    const auto center0 = Point<double>(0.0, 0.0, 0.0);
    const auto orientation0 = mundy::Quaterniond::identity();
    const double r1_0 = 3.0;
    const double r2_0 = 1.0;
    const double r3_0 = 2.0;
    const Ellipsoid<double> ellipsoid0(center0, orientation0, r1_0, r2_0, r3_0);

    const auto center1 = Point<double>(0.0, 0.0, 0.0);
    const auto orientation1 = mundy::Quaterniond::identity();
    const double r1_1 = 2 * r1_0;
    const double r2_1 = 2 * r2_0;
    const double r3_1 = 2 * r3_0;
    const Ellipsoid<double> ellipsoid1(center1, orientation1, r1_1, r2_1, r3_1);

    const double shared_normal_ssd = distance(SharedNormalSigned{}, ellipsoid0, ellipsoid1);
    EXPECT_NEAR(shared_normal_ssd, -3 * r2_0, TEST_DOUBLE_EPSILON);
  }

  // Case 3: Same radii colinear along their major axis with known overlap (positive, negative, and zero)
  {
    auto run_case_3 = [](const double expected_ssd) {
      const double r1_0 = 3.0;
      const double r2_0 = 1.0;
      const double r3_0 = 2.0;
      const auto center0 = Point<double>(-r1_0 - 0.5 * expected_ssd, 0.0, 0.0);
      const auto orientation0 = mundy::Quaterniond::identity();  // Aligned with the x-axis
      const Ellipsoid<double> ellipsoid0(center0, orientation0, r1_0, r2_0, r3_0);

      const double r1_1 = r1_0;
      const double r2_1 = r2_0;
      const double r3_1 = r3_0;
      const auto center1 = -center0;
      const auto orientation1 = orientation0;
      const Ellipsoid<double> ellipsoid1(center1, orientation1, r1_1, r2_1, r3_1);

      const double shared_normal_ssd = distance(SharedNormalSigned{}, ellipsoid0, ellipsoid1);
      EXPECT_NEAR(shared_normal_ssd, expected_ssd, TEST_DOUBLE_EPSILON);
    };

    run_case_3(0.2);
    run_case_3(-0.2);
    run_case_3(0.0);
  }

  // Case 4: Perpendicular along major and minor axes with known overlap (positive, negative, and zero)
  {
    auto run_case_4 = [](const double expected_ssd) {
      const double r1_0 = 3.0;
      const double r2_0 = 1.0;
      const double r3_0 = 2.0;
      const auto center0 = Point<double>(0.0, r1_0 + r2_0 + expected_ssd, 0.0);
      const auto orientation0 = quat_from_parallel_transport(Point<double>(1.0, 0.0, 0.0),
                                                             Point<double>(0.0, 1.0, 0.0));  // Aligned with the y-axis
      const Ellipsoid<double> ellipsoid0(center0, orientation0, r1_0, r2_0, r3_0);

      const double r1_1 = r1_0;
      const double r2_1 = r2_0;
      const double r3_1 = r3_0;
      const auto center1 = Point<double>(0.0, 0.0, 0.0);
      const auto orientation1 = mundy::Quaterniond::identity();  // Aligned with the x-axis
      const Ellipsoid<double> ellipsoid1(center1, orientation1, r1_1, r2_1, r3_1);

      const double shared_normal_ssd = distance(SharedNormalSigned{}, ellipsoid0, ellipsoid1);
      EXPECT_NEAR(shared_normal_ssd, expected_ssd, TEST_DOUBLE_EPSILON);
    };

    run_case_4(0.2);
    run_case_4(-0.2);
    run_case_4(0.0);
  }
}

// ============================================================
//! \name Head-to-head: finite-diff baseline vs. default (FDF)
//@{
// ============================================================

/// \brief Generate a random true ellipsoid (not a sphere) within the given bounds.
static Ellipsoid<double> random_ellipsoid(openrand::Philox& rng, double xyz_range, double r_min, double r_max,
                                          double min_aspect) {
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  Point<double> center{rng.rand<double>() * xyz_range - xyz_range * 0.5,
                       rng.rand<double>() * xyz_range - xyz_range * 0.5,
                       rng.rand<double>() * xyz_range - xyz_range * 0.5};
  const auto orient =
      mundy::euler_to_quat(rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi);
  const double r = rng.rand<double>() * (r_max - r_min) + r_min;
  const double a1 = r;
  const double a2 = r * (rng.rand<double>() * (1.0 / min_aspect - min_aspect) + min_aspect);
  const double a3 = r * (rng.rand<double>() * (1.0 / min_aspect - min_aspect) + min_aspect);
  return Ellipsoid<double>(center, orient, a1, a2, a3);
}

TEST(SharedNormalDistanceBetweenEllipsoids_HeadToHead, AccuracyAgreement) {
  // SharedNormalSigned (FDF default) must agree with SharedNormalSignedFiniteDiff (baseline)
  // on the signed separation distance for a wide variety of random ellipsoid pairs.
  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  constexpr size_t n = MUNDY_GEOM_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST;

  size_t disagreements = 0;
  for (size_t i = 0; i < n; ++i) {
    const auto e0 = random_ellipsoid(rng, 20.0, 0.5, 3.0, 0.3);
    const auto e1 = random_ellipsoid(rng, 20.0, 0.5, 3.0, 0.3);
    const double d_fd = distance(SharedNormalSignedFiniteDiff{}, e0, e1);
    const double d_fdf = distance(SharedNormalSigned{}, e0, e1);
    if (std::abs(d_fd - d_fdf) > TEST_DOUBLE_EPSILON) {
      ++disagreements;
      EXPECT_NEAR(d_fd, d_fdf, TEST_DOUBLE_EPSILON) << "pair " << i;
    }
  }
  EXPECT_LE(disagreements, n / 1000) << disagreements << " / " << n << " pairs disagreed";
}

TEST(SharedNormalDistanceBetweenEllipsoids_HeadToHead, TimingComparison) {
  openrand::Philox rng = make_philox(generate_test_seed(), 0);
  constexpr size_t n = MUNDY_GEOM_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST;

  std::vector<std::pair<Ellipsoid<double>, Ellipsoid<double>>> pairs;
  pairs.reserve(n);
  for (size_t i = 0; i < n; ++i)
    pairs.emplace_back(random_ellipsoid(rng, 20.0, 0.5, 3.0, 0.3), random_ellipsoid(rng, 20.0, 0.5, 3.0, 0.3));

  double sink_fd = 0.0, sink_fdf = 0.0;

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (const auto& [e0, e1] : pairs) sink_fd += distance(SharedNormalSignedFiniteDiff{}, e0, e1);
  const auto t1 = std::chrono::high_resolution_clock::now();
  for (const auto& [e0, e1] : pairs) sink_fdf += distance(SharedNormalSigned{}, e0, e1);
  const auto t2 = std::chrono::high_resolution_clock::now();

  ASSERT_FALSE(std::isnan(sink_fd));
  ASSERT_FALSE(std::isnan(sink_fdf));

  const double ms_fd = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_fdf = std::chrono::duration<double, std::milli>(t2 - t1).count();

  std::cout << "  [EllipsoidEllipsoid distance, n=" << n << "]\n"
            << "    FiniteDiff (baseline): " << ms_fd << " ms  (1.00x)\n"
            << "    SharedNormalSigned:    " << ms_fdf << " ms  (" << ms_fd / ms_fdf << "x)\n";

  EXPECT_GT(ms_fd / ms_fdf, 0.8) << "default (FDF) unexpectedly slower than finite-diff baseline";
}

// The signed distance is differentiable w.r.t. the point and ellipsoid parameters. AD gradient
// checked analytically and against central finite differences.
TEST(SharedNormalDistanceBetweenEllipsoidAndPoint, AutoDiffGradients) {
  using AD = AutoDiffScalar<double, 3>;  // derivatives seeded in slots 0,1,2

  // (1) Point vs unit sphere, off-axis: d(dist)/d(point) = (p - c) / ||p - c||.
  {
    const double px = 3.0, py = 1.0, pz = 0.5;
    const double nrm = std::sqrt(px * px + py * py + pz * pz);
    const Point<AD> p(AD(px, 0), AD(py, 1), AD(pz, 2));
    const Ellipsoid<AD> e(Point<AD>(AD(0), AD(0), AD(0)), AD(1), AD(1), AD(1));
    const AD d = distance(p, e);
    EXPECT_NEAR(d.value(), nrm - 1.0, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[0], px / nrm, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[1], py / nrm, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[2], pz / nrm, TEST_DOUBLE_EPSILON);
  }

  // (2) Point vs genuine ellipsoid: AD gradient vs central finite differences of the double distance.
  {
    auto dist_of_point = [](double px, double py, double pz) {
      const Point<double> p(px, py, pz);
      const Ellipsoid<double> e(Point<double>(0, 0, 0), 1.0, 2.0, 0.5);
      return distance(p, e);
    };
    const double px = 3.0, py = 1.2, pz = 0.4, h = 1.0e-5;
    const double fx = (dist_of_point(px + h, py, pz) - dist_of_point(px - h, py, pz)) / (2 * h);
    const double fy = (dist_of_point(px, py + h, pz) - dist_of_point(px, py - h, pz)) / (2 * h);
    const double fz = (dist_of_point(px, py, pz + h) - dist_of_point(px, py, pz - h)) / (2 * h);

    const Point<AD> p(AD(px, 0), AD(py, 1), AD(pz, 2));
    const Ellipsoid<AD> e(Point<AD>(AD(0), AD(0), AD(0)), AD(1.0), AD(2.0), AD(0.5));
    const AD d = distance(p, e);
    EXPECT_NEAR(d.value(), dist_of_point(px, py, pz), TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[0], fx, 1.0e-3);
    EXPECT_NEAR(d.derivatives()[1], fy, 1.0e-3);
    EXPECT_NEAR(d.derivatives()[2], fz, 1.0e-3);
  }
}

// The shared-normal signed distance is differentiable w.r.t. the ellipsoid parameters. AD gradient
// checked analytically, against central finite differences, at a degenerate axis-aligned
// configuration, and for agreement between the analytical- and finite-difference-gradient variants.
TEST(SharedNormalDistanceBetweenEllipsoids, AutoDiffGradients) {
  using AD = AutoDiffScalar<double, 3>;  // derivatives seeded in slots 0,1,2

  // (1) Two unit spheres, off-axis: d(dist)/d(center2) = (c2 - c1) / ||c2 - c1||.
  {
    const double cx = 5.0, cy = 1.0, cz = 0.5;
    const double nrm = std::sqrt(cx * cx + cy * cy + cz * cz);
    const Ellipsoid<AD> e1(Point<AD>(AD(0), AD(0), AD(0)), AD(1), AD(1), AD(1));
    const Ellipsoid<AD> e2(Point<AD>(AD(cx, 0), AD(cy, 1), AD(cz, 2)), AD(1), AD(1), AD(1));
    const AD d = distance(e1, e2);
    EXPECT_NEAR(d.value(), nrm - 2.0, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[0], cx / nrm, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[1], cy / nrm, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[2], cz / nrm, TEST_DOUBLE_EPSILON);
  }

  // (2) Genuinely ellipsoidal pair: AD gradient vs central finite differences of the double distance.
  {
    auto dist_of_center = [](double cx, double cy, double cz) {
      const Ellipsoid<double> e1(Point<double>(0, 0, 0), 1.0, 2.0, 0.5);
      const Ellipsoid<double> e2(Point<double>(cx, cy, cz), 1.5, 0.8, 1.2);
      return distance(e1, e2);
    };
    const double cx = 4.0, cy = 1.0, cz = 0.3, h = 1.0e-5;
    const double fd_x = (dist_of_center(cx + h, cy, cz) - dist_of_center(cx - h, cy, cz)) / (2 * h);
    const double fd_y = (dist_of_center(cx, cy + h, cz) - dist_of_center(cx, cy - h, cz)) / (2 * h);
    const double fd_z = (dist_of_center(cx, cy, cz + h) - dist_of_center(cx, cy, cz - h)) / (2 * h);

    const Ellipsoid<AD> e1(Point<AD>(AD(0), AD(0), AD(0)), AD(1.0), AD(2.0), AD(0.5));
    const Ellipsoid<AD> e2(Point<AD>(AD(cx, 0), AD(cy, 1), AD(cz, 2)), AD(1.5), AD(0.8), AD(1.2));
    const AD d = distance(e1, e2);
    EXPECT_NEAR(d.value(), dist_of_center(cx, cy, cz), TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[0], fd_x, 1.0e-3);
    EXPECT_NEAR(d.derivatives()[1], fd_y, 1.0e-3);
    EXPECT_NEAR(d.derivatives()[2], fd_z, 1.0e-3);
  }

  // (3) Degenerate axis-aligned spheres: the gradient must stay finite and analytic,
  // d(dist)/d(center2) = (1, 0, 0).
  {
    const Ellipsoid<AD> e1(Point<AD>(AD(0), AD(0), AD(0)), AD(1), AD(1), AD(1));
    const Ellipsoid<AD> e2(Point<AD>(AD(5.0, 0), AD(0.0, 1), AD(0.0, 2)), AD(1), AD(1), AD(1));
    const AD d = distance(e1, e2);
    ASSERT_FALSE(std::isnan(d.derivatives()[0]));
    ASSERT_FALSE(std::isnan(d.derivatives()[1]));
    ASSERT_FALSE(std::isnan(d.derivatives()[2]));
    EXPECT_NEAR(d.value(), 3.0, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[0], 1.0, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[1], 0.0, TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d.derivatives()[2], 0.0, TEST_DOUBLE_EPSILON);
  }

  // (4) The finite-difference-gradient variant agrees with the analytical-gradient default in both
  // value and gradient.
  {
    const double cx = 4.0, cy = 1.0, cz = 0.3;
    const Ellipsoid<AD> e1(Point<AD>(AD(0), AD(0), AD(0)), AD(1.0), AD(2.0), AD(0.5));
    const Ellipsoid<AD> e2(Point<AD>(AD(cx, 0), AD(cy, 1), AD(cz, 2)), AD(1.5), AD(0.8), AD(1.2));
    const AD d_fd = distance(SharedNormalSignedFiniteDiff{}, e1, e2);
    const AD d_default = distance(e1, e2);
    EXPECT_NEAR(d_fd.value(), d_default.value(), TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d_fd.derivatives()[0], d_default.derivatives()[0], TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d_fd.derivatives()[1], d_default.derivatives()[1], TEST_DOUBLE_EPSILON);
    EXPECT_NEAR(d_fd.derivatives()[2], d_default.derivatives()[2], TEST_DOUBLE_EPSILON);
  }
}

//@}

}  // namespace

}  // namespace mundy
