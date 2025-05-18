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

// C++ core libs
#include <algorithm>   // for std::max
#include <concepts>    // for std::convertible_to
#include <functional>  // for std::hash
#include <string>      // for std::string

// Trilinos includes
#include <Kokkos_Core.hpp>  // for Kokkos::numbers::pi

// Mundy libs
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>    // for mundy::math::Vector3
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>

/// \brief The following global is used to control the number of samples per test.
/// For unit tests, this number should be kept low to ensure fast test times, but to still give an immediate warning if
/// something went very wrong. For integration tests, we recommend setting this number to 10,000 or more.
#ifndef MUNDY_MATH_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST
#define MUNDY_MATH_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST 10000
#endif

namespace mundy {

namespace math {

namespace distance {

namespace {

const double TEST_DOUBLE_EPSILON =
    1.e-4;  // After numerous tests, the best precision we can get for the random sphere test is 1.e-4

// Utility function to generate a unique seed for each test based on its GTEST name.
size_t generate_test_seed() {
  const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string test_identifier = std::string(test_info->test_suite_name()) + "." + test_info->name();
  return std::hash<std::string>{}(test_identifier);
}

TEST(SharedNormalDistanceBetweenEllipsoidAndPoint, AnalyticalSphereTestCases) {
  // Spheres admit an analytical signed separation distance. We can generate N random spheres with random positions,
  // radii, and orientations and check that the numerical signed separation distance matches the analytical result.

  auto perform_test_for_given_spheres = [](const Vector3d& center, const Quaterniond& orientation, const double r,
                                           const Vector3d& point) {
    const double shared_normal_ssd_numeric =
        shared_normal_ssd_between_ellipsoid_and_point(center, orientation, r, r, r, point);
    const double shared_normal_ssd_analytic =
        shared_normal_ssd_between_ellipsoid_and_point(center, orientation, r, r, r, point);
    const double expected_ssd = mundy::math::norm(point - center) - r;

    // Assert used to avoid 10 million throws
    ASSERT_NEAR(shared_normal_ssd_numeric, expected_ssd, TEST_DOUBLE_EPSILON);
    ASSERT_NEAR(shared_normal_ssd_analytic, expected_ssd, TEST_DOUBLE_EPSILON);
  };

  openrand::Philox rng(generate_test_seed(), 0);
  const double min_xyz = -10.0;
  const double max_xyz = 10.0;
  const double range_xyz = max_xyz - min_xyz;
  const double min_r = 0.1;
  const double max_r = 10.0;
  const double range_r = max_r - min_r;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  for (size_t i = 0; i < MUNDY_MATH_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST; ++i) {
    const Vector3d center = {rng.rand<double>() * range_xyz + min_xyz, rng.rand<double>() * range_xyz + min_xyz,
                             rng.rand<double>() * range_xyz + min_xyz};
    const auto orientation =
        euler_to_quat(rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi);
    const double r = rng.rand<double>() * range_r + min_r;

    const Vector3d point = {rng.rand<double>() * range_xyz + min_xyz, rng.rand<double>() * range_xyz + min_xyz,
                            rng.rand<double>() * range_xyz + min_xyz};
    perform_test_for_given_spheres(center, orientation, r, point);
  }
}

TEST(SharedNormalDistanceBetweenEllipsoids, AnalyticalSphereTestCases) {
  // Spheres admit an analytical signed separation distance. We can generate N random spheres with random positions,
  // radii, and orientations and check that the numerical signed separation distance matches the analytical result.

  auto perform_test_for_given_spheres = [](const Vector3d& center0, const Quaterniond& orientation0, const double r0,
                                           const Vector3d& center1, const Quaterniond& orientation1, const double r1) {
    const double shared_normal_ssd =
        shared_normal_ssd_between_ellipsoids(center0, orientation0, r0, r0, r0, center1, orientation1, r1, r1, r1);

    const double expected_ssd = mundy::math::norm(center1 - center0) - r0 - r1;

    // Assert used to avoud 10 million throws
    ASSERT_NEAR(shared_normal_ssd, expected_ssd, TEST_DOUBLE_EPSILON);
  };

  openrand::Philox rng(generate_test_seed(), 0);
  const double min_xyz = -10.0;
  const double max_xyz = 10.0;
  const double range_xyz = max_xyz - min_xyz;
  const double min_r = 0.1;
  const double max_r = 10.0;
  const double range_r = max_r - min_r;
  constexpr double pi = Kokkos::numbers::pi_v<double>;

  for (size_t i = 0; i < MUNDY_MATH_TESTS_UNIT_TESTS_ELLIPSOID_ELLIPSOID_DISTANCE_NUM_SAMPLES_PER_TEST; ++i) {
    const Vector3d center0 = {rng.rand<double>() * range_xyz + min_xyz, rng.rand<double>() * range_xyz + min_xyz,
                              rng.rand<double>() * range_xyz + min_xyz};
    const auto orientation0 =
        euler_to_quat(rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi);
    const double r0 = rng.rand<double>() * range_r + min_r;

    const Vector3d center1 = {rng.rand<double>() * range_xyz + min_xyz, rng.rand<double>() * range_xyz + min_xyz,
                              rng.rand<double>() * range_xyz + min_xyz};
    const auto orientation1 =
        euler_to_quat(rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi, rng.rand<double>() * 2.0 * pi);
    const double r1 = rng.rand<double>() * range_r + min_r;

    perform_test_for_given_spheres(center0, orientation0, r0, center1, orientation1, r1);
  }
}

TEST(SharedNormalDistanceBetweenEllipsoids, AnalyticalEllipsoidTestCases) {
  // There are a few cases where we can analytically compute the shared normal signed separation distance between two
  // ellipsoids.

  // Case 1: Perfect overlap
  {
    const auto center0 = Vector3d(0.0, 0.0, 0.0);
    const auto orientation0 = Quaterniond::identity();
    const double r1_0 = 3.0;
    const double r2_0 = 1.0;
    const double r3_0 = 2.0;

    const auto center1 = Vector3d(0.0, 0.0, 0.0);
    const auto orientation1 = Quaterniond::identity();
    const double r1_1 = r1_0;
    const double r2_1 = r2_0;
    const double r3_1 = r3_0;

    const double shared_normal_ssd = shared_normal_ssd_between_ellipsoids(center0, orientation0, r1_0, r2_0, r3_0,
                                                                          center1, orientation1, r1_1, r2_1, r3_1);

    EXPECT_NEAR(shared_normal_ssd, -2 * r2_0, TEST_DOUBLE_EPSILON);
  }

  // Case 2: Same centers/orientations but one scaled up by a factor of 2
  {
    const auto center0 = Vector3d(0.0, 0.0, 0.0);
    const auto orientation0 = Quaterniond::identity();
    const double r1_0 = 3.0;
    const double r2_0 = 1.0;
    const double r3_0 = 2.0;

    const auto center1 = Vector3d(0.0, 0.0, 0.0);
    const auto orientation1 = Quaterniond::identity();
    const double r1_1 = 2 * r1_0;
    const double r2_1 = 2 * r2_0;
    const double r3_1 = 2 * r3_0;

    const double shared_normal_ssd = shared_normal_ssd_between_ellipsoids(center0, orientation0, r1_0, r2_0, r3_0,
                                                                          center1, orientation1, r1_1, r2_1, r3_1);

    EXPECT_NEAR(shared_normal_ssd, -3 * r2_0, TEST_DOUBLE_EPSILON);
  }

  // Case 3: Same radii colinear along their major axis with known overlap (positive, negative, and zero)
  {
    auto run_case_3 = [](const double expected_ssd) {
      const double r1_0 = 3.0;
      const double r2_0 = 1.0;
      const double r3_0 = 2.0;
      const auto center0 = Vector3d(-r1_0 - 0.5 * expected_ssd, 0.0, 0.0);
      const auto orientation0 = Quaterniond::identity();  // Aligned with the x-axis

      const double r1_1 = r1_0;
      const double r2_1 = r2_0;
      const double r3_1 = r3_0;
      const auto center1 = -center0;
      const auto orientation1 = orientation0;

      const double shared_normal_ssd = shared_normal_ssd_between_ellipsoids(center0, orientation0, r1_0, r2_0, r3_0,
                                                                            center1, orientation1, r1_1, r2_1, r3_1);

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
      const auto center0 = Vector3d(0.0, r1_0 + r2_0 + expected_ssd, 0.0);
      const auto orientation0 =
          quat_from_parallel_transport(Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 1.0, 0.0));  // Aligned with the y-axis

      const double r1_1 = r1_0;
      const double r2_1 = r2_0;
      const double r3_1 = r3_0;
      const auto center1 = Vector3d(0.0, 0.0, 0.0);
      const auto orientation1 = Quaterniond::identity();  // Aligned with the x-axis

      const double shared_normal_ssd = shared_normal_ssd_between_ellipsoids(center0, orientation0, r1_0, r2_0, r3_0,
                                                                            center1, orientation1, r1_1, r2_1, r3_1);

      EXPECT_NEAR(shared_normal_ssd, expected_ssd, TEST_DOUBLE_EPSILON);
    };

    run_case_4(0.2);
    run_case_4(-0.2);
    run_case_4(0.0);
  }
}

}  // namespace

}  // namespace distance

}  // namespace math

}  // namespace mundy
