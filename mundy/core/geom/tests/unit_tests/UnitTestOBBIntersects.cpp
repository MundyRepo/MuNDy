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
#include <gtest/gtest.h>      // for TEST, EXPECT_TRUE, EXPECT_FALSE
#include <openrand/philox.h>  // for openrand::Philox

// C++ core
#include <cmath>       // for std::sqrt, std::sin, std::cos
#include <concepts>    // for std::convertible_to
#include <functional>  // for std::hash
#include <string>      // for std::string

// Trilinos
#include <Kokkos_Core.hpp>  // for Kokkos::numbers::pi_v

// Mundy
#include <mundy_geom/primitives.hpp>  // for mundy::OBB, mundy::Point, mundy::intersects
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaternion
#include <mundy_math/Tolerance.hpp>   // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3
#include <mundy_utils/rng.hpp>        // for mundy::make_philox

/// \brief Number of random samples per test.
/// Keep low for fast unit-test runs; increase to 10,000+ for integration testing.
#ifndef MUNDY_GEOM_TESTS_UNIT_TESTS_OBB_INTERSECTS_NUM_SAMPLES_PER_TEST
#define MUNDY_GEOM_TESTS_UNIT_TESTS_OBB_INTERSECTS_NUM_SAMPLES_PER_TEST 1000000
#endif

namespace mundy {

namespace {

size_t generate_test_seed() {
  const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string test_identifier = std::string(test_info->test_suite_name()) + "." + test_info->name();
  return std::hash<std::string>{}(test_identifier);
}

template <typename T>
concept RandomNumberGenerator = requires(T rng) {
  { rng.template rand<double>() } -> std::convertible_to<double>;
};

// ============================================================
// random_unit_quaternion
//
// Uniformly distributed unit quaternion via Shoemake's method
// (Graphics Gems III, 1992). Returns Quaternion<double>{w,x,y,z}.
// ============================================================
template <RandomNumberGenerator RngType>
Quaternion<double> random_unit_quaternion(RngType& rng) {
  const double u1 = rng.template rand<double>();
  const double u2 = rng.template rand<double>();
  const double u3 = rng.template rand<double>();
  const double two_pi = 2.0 * Kokkos::numbers::pi_v<double>;
  return Quaternion<double>{std::sqrt(1.0 - u1) * std::sin(two_pi * u2), std::sqrt(1.0 - u1) * std::cos(two_pi * u2),
                            std::sqrt(u1) * std::sin(two_pi * u3), std::sqrt(u1) * std::cos(two_pi * u3)};
}

// ============================================================
// generate_globally_rotated_obbs
//
// Construct two OBBs with a known face-normal gap along A's local x-axis:
//
//   A: center (0,0,0), orientation Q, half-extents (hax, hay, haz)
//   B: center Q*(hax+hbx+gap, 0, 0), orientation Q, half-extents (hbx, hby, hbz)
//
// Applying the same rotation Q to both boxes preserves the intersection status:
// the relative rotation R = mat(conj(Q)*Q) = I, so the face-normal test on A's
// local x-axis reduces to |hax + hbx + gap| > hax + hbx, which fires iff gap > 0.
//
// Oracle: intersects(a, b)  <=>  gap <= 0
// ============================================================
template <RandomNumberGenerator RngType>
void generate_globally_rotated_obbs(RngType& rng, double gap, OBB<double>& a, OBB<double>& b) {
  const double hax = 0.1 + 0.8 * rng.template rand<double>();
  const double hay = 0.1 + 0.8 * rng.template rand<double>();
  const double haz = 0.1 + 0.8 * rng.template rand<double>();
  const double hbx = 0.1 + 0.8 * rng.template rand<double>();
  const double hby = 0.1 + 0.8 * rng.template rand<double>();
  const double hbz = 0.1 + 0.8 * rng.template rand<double>();

  const Quaternion<double> Q = random_unit_quaternion(rng);

  // B's center in A's pre-rotation local frame: (hax+hbx+gap, 0, 0).
  const double sep = hax + hbx + gap;
  const Vector3<double> b_local{sep, 0.0, 0.0};
  const auto b_world = Q * b_local;

  a = OBB<double>{Point<double>{0.0, 0.0, 0.0}, Q, hax, hay, haz};
  b = OBB<double>{Point<double>{b_world[0], b_world[1], b_world[2]}, Q, hbx, hby, hbz};
}

// ============================================================
// Tests
// ============================================================

// ============================================================
// SeparatedAlongFaceNormal
//
// Random half-extents, random global rotation, gap strictly positive.
// Oracle: intersects() must return false.
// ============================================================
TEST(OBBIntersects, SeparatedAlongFaceNormal) {
  auto rng = mundy::make_philox(generate_test_seed(), 0.);
  const int N = MUNDY_GEOM_TESTS_UNIT_TESTS_OBB_INTERSECTS_NUM_SAMPLES_PER_TEST;
  for (int i = 0; i < N; ++i) {
    // gap in [0.02, 0.5]: clearly separated, not within the eps buffer
    const double gap = 0.02 + 0.48 * rng.template rand<double>();
    OBB<double> a, b;
    generate_globally_rotated_obbs(rng, gap, a, b);
    EXPECT_FALSE(intersects(a, b)) << "gap=" << gap << " i=" << i;
  }
}

// ============================================================
// OverlappingAlongFaceNormal
//
// Random half-extents, random global rotation, gap strictly negative.
// Oracle: intersects() must return true.
// ============================================================
TEST(OBBIntersects, OverlappingAlongFaceNormal) {
  auto rng = mundy::make_philox(generate_test_seed(), 0.);
  const int N = MUNDY_GEOM_TESTS_UNIT_TESTS_OBB_INTERSECTS_NUM_SAMPLES_PER_TEST;
  for (int i = 0; i < N; ++i) {
    // gap in [-0.1, -0.02]: clearly overlapping.
    // |gap| is bounded by 0.1 < 0.1 (min half-extent) to keep B from passing
    // completely through A along x.
    const double gap = -0.02 - 0.08 * rng.template rand<double>();
    OBB<double> a, b;
    generate_globally_rotated_obbs(rng, gap, a, b);
    EXPECT_TRUE(intersects(a, b)) << "gap=" << gap << " i=" << i;
  }
}

// ============================================================
// AxisAlignedGapSweep
//
// Deterministic sweep with identity orientation along each principal axis.
// Verifies the separated/overlapping transition at gap = 0.
// ============================================================
TEST(OBBIntersects, AxisAlignedGapSweep) {
  const Quaternion<double> I = Quaternion<double>::identity();
  const double hax = 0.3, hay = 0.4, haz = 0.5;
  const double hbx = 0.2, hby = 0.3, hbz = 0.4;
  const OBB<double> a{Point<double>{0.0, 0.0, 0.0}, I, hax, hay, haz};

  // Along A's local x-axis
  for (const double gap : {0.5, 0.1, 0.01, 0.001, -0.001, -0.01, -0.1, -0.5}) {
    const double sep = hax + hbx + gap;
    const OBB<double> b{Point<double>{sep, 0.0, 0.0}, I, hbx, hby, hbz};
    EXPECT_EQ(intersects(a, b), gap <= 0.0) << "x-axis gap=" << gap;
  }

  // Along A's local y-axis
  for (const double gap : {0.5, 0.1, 0.001, -0.001, -0.1}) {
    const double sep = hay + hby + gap;
    const OBB<double> b{Point<double>{0.0, sep, 0.0}, I, hbx, hby, hbz};
    EXPECT_EQ(intersects(a, b), gap <= 0.0) << "y-axis gap=" << gap;
  }

  // Along A's local z-axis
  for (const double gap : {0.5, 0.1, 0.001, -0.001, -0.1}) {
    const double sep = haz + hbz + gap;
    const OBB<double> b{Point<double>{0.0, 0.0, sep}, I, hbx, hby, hbz};
    EXPECT_EQ(intersects(a, b), gap <= 0.0) << "z-axis gap=" << gap;
  }
}

// ============================================================
// SameCenter_AlwaysIntersects
//
// A and B at the same center with independent random orientations.
// Both boxes contain their own center, so they always intersect
// (B's center is inside A, and vice versa).
// ============================================================
TEST(OBBIntersects, SameCenter_AlwaysIntersects) {
  auto rng = mundy::make_philox(generate_test_seed(), 0.);
  const int N = MUNDY_GEOM_TESTS_UNIT_TESTS_OBB_INTERSECTS_NUM_SAMPLES_PER_TEST;
  for (int i = 0; i < N; ++i) {
    const double hax = 0.1 + 0.9 * rng.template rand<double>();
    const double hay = 0.1 + 0.9 * rng.template rand<double>();
    const double haz = 0.1 + 0.9 * rng.template rand<double>();
    const double hbx = 0.1 + 0.9 * rng.template rand<double>();
    const double hby = 0.1 + 0.9 * rng.template rand<double>();
    const double hbz = 0.1 + 0.9 * rng.template rand<double>();
    const Quaternion<double> Q_a = random_unit_quaternion(rng);
    const Quaternion<double> Q_b = random_unit_quaternion(rng);
    const OBB<double> a{Point<double>{0.0, 0.0, 0.0}, Q_a, hax, hay, haz};
    const OBB<double> b{Point<double>{0.0, 0.0, 0.0}, Q_b, hbx, hby, hbz};
    EXPECT_TRUE(intersects(a, b)) << "i=" << i;
  }
}

// ============================================================
// BoundingSphereOracle_Separated
//
// Place B's center at distance > bounding_radius_A + bounding_radius_B
// from A's center (bounding_radius = sqrt(hx²+hy²+hz²) = half-diagonal).
// With independent random orientations. Oracle: must NOT intersect.
// ============================================================
TEST(OBBIntersects, BoundingSphereOracle_Separated) {
  auto rng = mundy::make_philox(generate_test_seed(), 0.);
  const int N = MUNDY_GEOM_TESTS_UNIT_TESTS_OBB_INTERSECTS_NUM_SAMPLES_PER_TEST;
  const double two_pi = 2.0 * Kokkos::numbers::pi_v<double>;
  for (int i = 0; i < N; ++i) {
    const double hax = 0.1 + 0.4 * rng.rand<double>();
    const double hay = 0.1 + 0.4 * rng.rand<double>();
    const double haz = 0.1 + 0.4 * rng.rand<double>();
    const double hbx = 0.1 + 0.4 * rng.rand<double>();
    const double hby = 0.1 + 0.4 * rng.rand<double>();
    const double hbz = 0.1 + 0.4 * rng.rand<double>();

    const double r_a = std::sqrt(hax * hax + hay * hay + haz * haz);
    const double r_b = std::sqrt(hbx * hbx + hby * hby + hbz * hbz);

    // B center at distance r_a + r_b + 0.1 from origin, in a random direction.
    const double dist = r_a + r_b + 0.1;
    const double phi = two_pi * rng.template rand<double>();
    const double theta = std::acos(2.0 * rng.template rand<double>() - 1.0);
    const Point<double> b_center{dist * std::sin(theta) * std::cos(phi), dist * std::sin(theta) * std::sin(phi),
                                 dist * std::cos(theta)};

    const Quaternion<double> Q_a = random_unit_quaternion(rng);
    const Quaternion<double> Q_b = random_unit_quaternion(rng);
    const OBB<double> a{Point<double>{0.0, 0.0, 0.0}, Q_a, hax, hay, haz};
    const OBB<double> b{b_center, Q_b, hbx, hby, hbz};
    EXPECT_FALSE(intersects(a, b)) << "i=" << i;
  }
}

}  // namespace
}  // namespace mundy
