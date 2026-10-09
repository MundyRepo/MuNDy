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

/// \file UnitTestQuasiUniformSpherocylinder.cpp
/// \brief Spherocylinder quadrature against surface geometry, exact area, and axial moments.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

// Mundy
#include <mundy_math/QuasiUniformSpherocylinder.hpp>
#include <mundy_math/Vector3.hpp>

namespace mundy {

namespace {

constexpr double pi = Kokkos::numbers::pi_v<double>;
using SmallRule = QuasiUniformSpherocylinder<double, 8, 3, 12>;
using SphereRule = QuasiUniformSpherocylinder<double, 8, 0, 12>;

struct One {
  KOKKOS_INLINE_FUNCTION constexpr double operator()(double, double, double) const {
    return 1.0;
  }
};

struct AxialMoments {
  KOKKOS_INLINE_FUNCTION Vector3d operator()(double, double, double z) const {
    return Vector3d{1.0, z, z * z};
  }
};

static_assert(SmallRule::num_points == 48);
static_assert(SmallRule::points(2.0, 3.0)[2] == -1.0);
static_assert(SmallRule::weights(2.0, 3.0)[0] > 0.0);
static_assert(SphereRule::num_points == 24);
static_assert(SphereRule::points(1.0, 0.0)[2] > 0.0);
static_assert(SmallRule::integrate(2.0, 3.0, One{}) > 28.0 * pi - 1e-12 &&
              SmallRule::integrate(2.0, 3.0, One{}) < 28.0 * pi + 1e-12);

/// \brief Check surface membership, patch ordering, weights, and the two cap reflections.
void expect_surface(const std::vector<double>& points, const std::vector<double>& weights, unsigned n_theta,
                    unsigned n_z, unsigned n_cap, double radius, double length) {
  const unsigned cylinder_points = n_theta * n_z;
  ASSERT_EQ(weights.size(), cylinder_points + 2 * n_cap);
  ASSERT_EQ(points.size(), 3 * weights.size());
  double cylinder_area = 0.0;
  double cap_area = 0.0;
  for (unsigned i = 0; i < weights.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "point " << i);
    const double x = points[3 * i];
    const double y = points[3 * i + 1];
    const double z = points[3 * i + 2];
    if (i < cylinder_points) {
      EXPECT_NEAR(std::hypot(x, y), radius, 4e-15 * radius);
      const double expected_z = length * ((static_cast<double>(i / n_theta) + 0.5) / n_z - 0.5);
      EXPECT_NEAR(z, expected_z, 2e-15 * length);
      EXPECT_NEAR(weights[i], 2 * pi * radius * length / cylinder_points, 2e-15 * radius * length);
      cylinder_area += weights[i];
    } else {
      const bool north = i < cylinder_points + n_cap;
      const double center = north ? length / 2 : -length / 2;
      EXPECT_NEAR(std::hypot(std::hypot(x, y), z - center), radius, 4e-15 * (radius + length));
      EXPECT_GT(north ? z - center : center - z, 0.0);
      EXPECT_NEAR(weights[i], 2 * pi * radius * radius / n_cap, 2e-15 * radius * radius);
      if (north) {
        EXPECT_EQ(x, points[3 * (i + n_cap)]);
        EXPECT_EQ(y, points[3 * (i + n_cap) + 1]);
        EXPECT_EQ(z, -points[3 * (i + n_cap) + 2]);
      }
      cap_area += weights[i];
    }
    EXPECT_GE(weights[i], 0.0);
  }
  EXPECT_NEAR(cylinder_area, 2 * pi * radius * length, 1e-13 * radius * length);
  EXPECT_NEAR(cap_area, 4 * pi * radius * radius, 1e-13 * radius * radius);
}

/// \brief Compare compile-time construction, host construction, and the view builder for one rule.
template <class Scalar, unsigned NTheta, unsigned NZ, unsigned NCap>
void expect_builders_agree(Scalar radius, Scalar length) {
  using Rule = QuasiUniformSpherocylinder<Scalar, NTheta, NZ, NCap>;
  const auto fixed_points = Rule::points(radius, length);
  const auto fixed_weights = Rule::weights(radius, length);
  std::vector<Scalar> points, weights;
  quasi_uniform_spherocylinder_rule(NTheta, NZ, NCap, radius, length, points, weights);
  Kokkos::View<Scalar*> device_points("points", 0), device_weights("weights", 0);
  quasi_uniform_spherocylinder_rule(NTheta, NZ, NCap, radius, length, device_points, device_weights);
  const auto host_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_points);
  const auto host_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_weights);
  ASSERT_EQ(points.size(), 3 * Rule::num_points);
  ASSERT_EQ(weights.size(), Rule::num_points);
  ASSERT_EQ(host_points.extent(0), points.size());
  ASSERT_EQ(host_weights.extent(0), weights.size());
  const double eps = std::numeric_limits<Scalar>::epsilon();
  for (unsigned i = 0; i < points.size(); ++i) {
    EXPECT_EQ(points[i], fixed_points[i]) << "point component " << i;
    EXPECT_NEAR(host_points(i), points[i], 4 * eps * (radius + length)) << "point component " << i;
  }
  for (unsigned i = 0; i < weights.size(); ++i) {
    EXPECT_EQ(weights[i], fixed_weights[i]) << "weight " << i;
    EXPECT_NEAR(host_weights(i), weights[i], 4 * eps * std::abs(weights[i])) << "weight " << i;
  }
}

// Keep device lambdas outside GTest bodies. Each thread evaluates a fixed rule with its own geometry.
void integrate_in_kernel(const Kokkos::View<double* [3]>& result) {
  Kokkos::parallel_for(
      "spherocylinder_integrals", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, result.extent(0)),
      KOKKOS_LAMBDA(const unsigned i) {
        const double radius = 1.0 + i;
        const double length = 0.5 * i;
        const auto moments = SmallRule::integrate(radius, length, AxialMoments{});
        const auto points = SmallRule::points(radius, length);
        const auto weights = SmallRule::weights(radius, length);
        double explicit_sum = 0.0;
        for (unsigned j = 0; j < SmallRule::num_points; ++j) {
          explicit_sum += weights[j] * points[3 * j + 2] * points[3 * j + 2];
        }
        result(i, 0) = moments[0];
        result(i, 1) = moments[1];
        result(i, 2) = moments[2] - explicit_sum;
      });
}

TEST(QuasiUniformSpherocylinder, PointsLieOnTheirSurfacePatches) {
  for (const auto& [radius, length] : {std::pair{1.0, 2.0}, std::pair{0.3, 4.7}, std::pair{2.0, 0.0}}) {
    std::vector<double> points, weights;
    quasi_uniform_spherocylinder_rule(9, 5, 31, radius, length, points, weights);
    expect_surface(points, weights, 9, 5, 31, radius, length);
  }
}

TEST(QuasiUniformSpherocylinder, CylinderRingsAlternateHalfALongitudeStep) {
  constexpr unsigned n_theta = 8;
  const auto points = SmallRule::points(1.0, 3.0);
  for (unsigned ring = 0; ring < 3; ++ring) {
    for (unsigned k = 0; k < n_theta; ++k) {
      const double angle = 2 * pi * (k + 0.5 + 0.5 * (ring % 2)) / n_theta;
      const unsigned i = ring * n_theta + k;
      EXPECT_NEAR(points[3 * i], std::cos(angle), 1e-15);
      EXPECT_NEAR(points[3 * i + 1], std::sin(angle), 1e-15);
    }
  }
}

TEST(QuasiUniformSpherocylinder, SphereLimitNeedsNoCylinderRings) {
  std::vector<double> points, weights;
  quasi_uniform_spherocylinder_rule(8, 0, 31, 2.0, 0.0, points, weights);
  expect_surface(points, weights, 8, 0, 31, 2.0, 0.0);
  EXPECT_NEAR(SphereRule::integrate(2.0, 0.0, One{}), 16 * pi, 1e-13);
}

TEST(QuasiUniformSpherocylinder, BuildersAgreeForFloatDoubleAndTheSphereLimit) {
  expect_builders_agree<double, 3, 1, 1>(2.0, 3.0);
  expect_builders_agree<double, 8, 3, 12>(1.3, 2.7);
  expect_builders_agree<double, 9, 0, 17>(0.7, 0.0);
  expect_builders_agree<double, 9, 3, 17>(0.7, 0.0);
  expect_builders_agree<float, 8, 3, 12>(1.3f, 2.7f);
  expect_builders_agree<float, 9, 0, 17>(0.7f, 0.0f);
}

TEST(QuasiUniformSpherocylinder, IntegratesAreaAndOddAxialMomentsInsideAKernel) {
  Kokkos::View<double* [3]> result("integrals", 4);
  integrate_in_kernel(result);
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  for (unsigned i = 0; i < 4; ++i) {
    const double radius = 1.0 + i;
    const double length = 0.5 * i;
    const double area = 2 * pi * radius * (length + 2 * radius);
    EXPECT_NEAR(host(i, 0), area, 2e-14 * area);
    EXPECT_NEAR(host(i, 1), 0.0, 2e-14 * area * (radius + length));
    EXPECT_NEAR(host(i, 2), 0.0, 2e-14 * area * (radius + length) * (radius + length));
  }
}

TEST(QuasiUniformSpherocylinder, AxialSecondMomentConvergesUnderRefinement) {
  const double radius = 1.3;
  const double length = 2.7;
  const double exact = 2 * pi * radius * length * length * length / 12 +
                       4 * pi * radius * radius * (length * length / 4 + length * radius / 2 + radius * radius / 3);
  double previous_error = 0.0;
  for (unsigned rings : {4u, 8u, 16u, 32u}) {
    std::vector<double> points, weights;
    quasi_uniform_spherocylinder_rule(8, rings, 2 * rings, radius, length, points, weights);
    double integral = 0.0;
    for (unsigned i = 0; i < weights.size(); ++i) {
      integral += weights[i] * points[3 * i + 2] * points[3 * i + 2];
    }
    const double error = exact - integral;
    EXPECT_GT(error, 0.0);
    if (previous_error > 0.0) {
      EXPECT_NEAR(error / previous_error, 0.25, 2e-11);
    }
    previous_error = error;
  }
}

TEST(QuasiUniformSpherocylinder, FibonacciAzimuthsRemainAccurateAtLargeIndices) {
  // Independent 80-digit evaluation of k*pi*(3-sqrt(5)), followed by sine and cosine.
  for (const auto& [k, sine, cosine] :
       {std::tuple{7u, -0.88744842924525484320286906199945055, -0.46090702471336874638397801770087125},
        std::tuple{100000000u, 0.70715349735499724205552482700352751, 0.70706006193151364316448047230295974}}) {
    const auto angle = impl::fibonacci_sin_cos(k);
    EXPECT_NEAR(angle.sin.hi(), sine, 2e-16);
    EXPECT_NEAR(angle.cos.hi(), cosine, 2e-16);
    EXPECT_NEAR(std::hypot(angle.sin.hi(), angle.cos.hi()), 1.0, 2e-16);
  }
}

TEST(QuasiUniformSpherocylinder, TargetCountDependsOnAspectRatioAndAgreesAcrossBuilders) {
  for (const double length : {0.0, 0.5, 4.0, 20.0}) {
    for (const unsigned target : {128u, 512u, 2048u}) {
      std::vector<double> points, weights, scaled_points, scaled_weights;
      quasi_uniform_spherocylinder_rule(target, 2.0, length, points, weights);
      quasi_uniform_spherocylinder_rule(target, 4.0, 2 * length, scaled_points, scaled_weights);
      ASSERT_EQ(points.size(), scaled_points.size());
      ASSERT_EQ(weights.size(), scaled_weights.size());
      EXPECT_NEAR(static_cast<double>(weights.size()), target, 0.2 * target);
      for (unsigned i = 0; i < points.size(); ++i) {
        EXPECT_EQ(scaled_points[i], 2 * points[i]);
      }
      double area = 0.0;
      for (unsigned i = 0; i < weights.size(); ++i) {
        EXPECT_EQ(scaled_weights[i], 4 * weights[i]);
        area += weights[i];
      }
      EXPECT_NEAR(area, 4 * pi * (length + 4), 1e-12 * area);
      Kokkos::View<double*> device_points("points", 0), device_weights("weights", 0);
      quasi_uniform_spherocylinder_rule(target, 2.0, length, device_points, device_weights);
      const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_points);
      const auto host_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_weights);
      ASSERT_EQ(host.extent(0), points.size());
      ASSERT_EQ(host_weights.extent(0), weights.size());
      for (unsigned i = 0; i < points.size(); ++i) {
        EXPECT_NEAR(host(i), points[i], 4e-15 * (2 + length));
      }
      for (unsigned i = 0; i < weights.size(); ++i) {
        EXPECT_NEAR(host_weights(i), weights[i], 4e-15 * weights[i]);
      }
    }
  }
}

TEST(QuasiUniformSpherocylinder, SmallTargetsRetainNonemptyCaps) {
  std::vector<double> points, weights;
  quasi_uniform_spherocylinder_rule(1, 1.0, 0.0, points, weights);
  EXPECT_EQ(weights.size(), 16u);
  quasi_uniform_spherocylinder_rule(1, 1.0, 1.0, points, weights);
  EXPECT_EQ(weights.size(), 24u);
}

TEST(QuasiUniformSpherocylinder, RejectsInvalidGeometryInEveryBuilder) {
  std::vector<double> points, weights;
  Kokkos::View<double*> device_points("points", 0), device_weights("weights", 0);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto& [radius, length] :
       {std::pair{0.0, 1.0}, std::pair{-1.0, 1.0}, std::pair{1.0, -1.0}, std::pair{inf, 1.0}, std::pair{1.0, inf},
        std::pair{nan, 1.0}, std::pair{1.0, nan}}) {
    EXPECT_THROW(SmallRule::points(radius, length), std::invalid_argument);
    EXPECT_THROW(SmallRule::weights(radius, length), std::invalid_argument);
    EXPECT_THROW(SmallRule::integrate(radius, length, One{}), std::invalid_argument);
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(8, 3, 12, radius, length, points, weights), std::invalid_argument);
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(8, 3, 12, radius, length, device_points, device_weights),
                 std::invalid_argument);
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(128, radius, length, points, weights), std::invalid_argument);
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(128, radius, length, device_points, device_weights),
                 std::invalid_argument);
  }
}

TEST(QuasiUniformSpherocylinder, RejectsMissingCylinderRingsForPositiveLength) {
  std::vector<double> points, weights;
  Kokkos::View<double*> device_points("points", 0), device_weights("weights", 0);
  EXPECT_THROW(SphereRule::points(1.0, 2.0), std::invalid_argument);
  EXPECT_THROW(SphereRule::weights(1.0, 2.0), std::invalid_argument);
  EXPECT_THROW(SphereRule::integrate(1.0, 2.0, One{}), std::invalid_argument);
  EXPECT_THROW(quasi_uniform_spherocylinder_rule(8, 0, 12, 1.0, 2.0, points, weights), std::invalid_argument);
  EXPECT_THROW(quasi_uniform_spherocylinder_rule(8, 0, 12, 1.0, 2.0, device_points, device_weights),
               std::invalid_argument);
}

TEST(QuasiUniformSpherocylinder, RejectsInvalidCountsBeforeAllocating) {
  std::vector<double> points, weights;
  Kokkos::View<double*> device_points("points", 0), device_weights("weights", 0);
  const unsigned max = std::numeric_limits<unsigned>::max();
  for (const auto& [n_theta, n_z, n_cap] : {std::tuple{2u, 1u, 8u}, std::tuple{8u, 1u, 0u}, std::tuple{max, 1u, 8u},
                                            std::tuple{8u, max, 8u}, std::tuple{8u, 1u, max}}) {
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(n_theta, n_z, n_cap, 1.0, 2.0, points, weights),
                 std::invalid_argument);
    EXPECT_THROW(quasi_uniform_spherocylinder_rule(n_theta, n_z, n_cap, 1.0, 2.0, device_points, device_weights),
                 std::invalid_argument);
  }
  EXPECT_THROW(quasi_uniform_spherocylinder_rule(0, 1.0, 2.0, points, weights), std::invalid_argument);
  EXPECT_THROW(quasi_uniform_spherocylinder_rule(0, 1.0, 2.0, device_points, device_weights), std::invalid_argument);
  EXPECT_TRUE(points.empty());
  EXPECT_TRUE(weights.empty());
  EXPECT_EQ(device_points.extent(0), 0u);
  EXPECT_EQ(device_weights.extent(0), 0u);
}

}  // namespace

}  // namespace mundy
