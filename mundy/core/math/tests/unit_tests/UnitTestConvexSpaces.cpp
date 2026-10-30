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

/// \file UnitTestConvexSpaces.cpp
/// \brief The geometry of the convex spaces (convex_spaces.hpp): projection, projected gradient, free part, and step to
/// the boundary, over vectors on MundyMathBackend and KokkosBackend.
///
/// Reference problem:
///   - The bounds -1 and 4, and the feasible point x = (-1, -1, 4, 4, 1, 1) with g = (2, -2, 2, -2, 2, -2), used both
///     as a gradient and as a direction: each entry of x lies on the lower bound, on the upper bound, or between them,
///     with g pointing each way. y = (-3, -1, 1/2, 4, 6, 1) is projected. Every value is small and dyadic, so every
///     result is exact and is compared against its value worked out by hand.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// Mundy
#include <mundy_math/Vector.hpp>  // for mundy::Vector
#include <mundy_math/convex_spaces.hpp>
#include <mundy_math/solver_backends.hpp>

#include "TestBackendVectors.hpp"  // for mundy::test::{make_backend_vector, host_entries}

namespace mundy {

namespace {

using mm_backend_t = MundyMathBackend;
using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using kokkos_vector_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;
using test::host_entries;
using test::make_backend_vector;

//! \name Compile-time contracts
//@{

static_assert(ConvexSpace<UnconstrainedSpace<double>, mm_backend_t, Vector<double, 6>>);
static_assert(ConvexSpace<LowerBoundSpace<double>, mm_backend_t, Vector<double, 6>>);
static_assert(ConvexSpace<UpperBoundSpace<double>, mm_backend_t, Vector<double, 6>>);
static_assert(ConvexSpace<BoundedSpace<double>, mm_backend_t, Vector<double, 6>>);
static_assert(ConvexSpace<UnconstrainedSpace<double>, kokkos_backend_t, kokkos_vector_t>);
static_assert(ConvexSpace<LowerBoundSpace<double>, kokkos_backend_t, kokkos_vector_t>);
static_assert(ConvexSpace<UpperBoundSpace<double>, kokkos_backend_t, kokkos_vector_t>);
static_assert(ConvexSpace<BoundedSpace<double>, kokkos_backend_t, kokkos_vector_t>);

static_assert(projected_gradient(LowerBoundSpace<double>{0.0}, 0.0, 1.0) == 0.0,
              "On a lower bound, a gradient pushing below it has no feasible part.");
static_assert(projected_gradient(UpperBoundSpace<double>{0.0}, 0.0, -1.0) == 0.0,
              "On an upper bound, a gradient pushing above it has no feasible part.");
//@}

//! \name Reference problem
//@{

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr std::array<double, 6> x_ref{-1.0, -1.0, 4.0, 4.0, 1.0, 1.0};
constexpr std::array<double, 6> g_ref{2.0, -2.0, 2.0, -2.0, 2.0, -2.0};
constexpr std::array<double, 6> y_ref{-3.0, -1.0, 0.5, 4.0, 6.0, 1.0};

const UnconstrainedSpace<double> unconstrained{};
const LowerBoundSpace<double> lower{-1.0};
const UpperBoundSpace<double> upper{4.0};
const BoundedSpace<double> bounded{-1.0, 4.0};

/// \brief The entries of e, on the host, once Backend assigns it to a vector of N entries.
template <class Backend, size_t N, class E>
std::vector<double> assigned_entries(const E& e) {
  auto out = make_backend_vector<Backend>(std::array<double, N>{});
  Backend::assign("assigned_entries", out, e);
  return host_entries(out);
}
//@}

//! \name Running and checking space geometry
//@{

template <class Backend>
void expect_projection_is_the_nearest_feasible_point() {
  const Backend backend;
  const auto y = make_backend_vector<Backend>(y_ref);
  EXPECT_EQ((assigned_entries<Backend, 6>(projection(backend, unconstrained, y))),
            (std::vector<double>{-3.0, -1.0, 0.5, 4.0, 6.0, 1.0}))
      << "unconstrained";
  EXPECT_EQ((assigned_entries<Backend, 6>(projection(backend, lower, y))),
            (std::vector<double>{-1.0, -1.0, 0.5, 4.0, 6.0, 1.0}))
      << "x >= -1";
  EXPECT_EQ((assigned_entries<Backend, 6>(projection(backend, upper, y))),
            (std::vector<double>{-3.0, -1.0, 0.5, 4.0, 4.0, 1.0}))
      << "x <= 4";
  EXPECT_EQ((assigned_entries<Backend, 6>(projection(backend, bounded, y))),
            (std::vector<double>{-1.0, -1.0, 0.5, 4.0, 4.0, 1.0}))
      << "-1 <= x <= 4";
}

template <class Backend>
void expect_projected_gradient_drops_the_infeasible_part() {
  const Backend backend;
  const auto x = make_backend_vector<Backend>(x_ref);
  const auto g = make_backend_vector<Backend>(g_ref);
  EXPECT_EQ((assigned_entries<Backend, 6>(projected_gradient(backend, unconstrained, x, g))),
            (std::vector<double>{2.0, -2.0, 2.0, -2.0, 2.0, -2.0}))
      << "unconstrained";
  EXPECT_EQ((assigned_entries<Backend, 6>(projected_gradient(backend, lower, x, g))),
            (std::vector<double>{0.0, -2.0, 2.0, -2.0, 2.0, -2.0}))
      << "x >= -1";
  EXPECT_EQ((assigned_entries<Backend, 6>(projected_gradient(backend, upper, x, g))),
            (std::vector<double>{2.0, -2.0, 2.0, 0.0, 2.0, -2.0}))
      << "x <= 4";
  EXPECT_EQ((assigned_entries<Backend, 6>(projected_gradient(backend, bounded, x, g))),
            (std::vector<double>{0.0, -2.0, 2.0, 0.0, 2.0, -2.0}))
      << "-1 <= x <= 4";
}

template <class Backend>
void expect_free_part_zeroes_the_entries_on_a_bound() {
  const Backend backend;
  const auto x = make_backend_vector<Backend>(x_ref);
  const auto v = make_backend_vector<Backend>(g_ref);
  EXPECT_EQ((assigned_entries<Backend, 6>(free_part(backend, unconstrained, x, v))),
            (std::vector<double>{2.0, -2.0, 2.0, -2.0, 2.0, -2.0}))
      << "unconstrained";
  EXPECT_EQ((assigned_entries<Backend, 6>(free_part(backend, lower, x, v))),
            (std::vector<double>{0.0, 0.0, 2.0, -2.0, 2.0, -2.0}))
      << "x >= -1";
  EXPECT_EQ((assigned_entries<Backend, 6>(free_part(backend, upper, x, v))),
            (std::vector<double>{2.0, -2.0, 0.0, 0.0, 2.0, -2.0}))
      << "x <= 4";
  EXPECT_EQ((assigned_entries<Backend, 6>(free_part(backend, bounded, x, v))),
            (std::vector<double>{0.0, 0.0, 0.0, 0.0, 2.0, -2.0}))
      << "-1 <= x <= 4";
}

template <class Backend>
void expect_step_to_boundary_is_the_largest_feasible_step() {
  const Backend backend;
  const auto x = make_backend_vector<Backend>(x_ref);
  const auto d = make_backend_vector<Backend>(g_ref);
  EXPECT_EQ((assigned_entries<Backend, 6>(step_to_boundary(backend, unconstrained, x, d))),
            (std::vector<double>{inf, inf, inf, inf, inf, inf}))
      << "unconstrained";
  EXPECT_EQ((assigned_entries<Backend, 6>(step_to_boundary(backend, lower, x, d))),
            (std::vector<double>{inf, 0.0, inf, 2.5, inf, 1.0}))
      << "x >= -1";
  EXPECT_EQ((assigned_entries<Backend, 6>(step_to_boundary(backend, upper, x, d))),
            (std::vector<double>{2.5, inf, 0.0, inf, 1.5, inf}))
      << "x <= 4";
  EXPECT_EQ((assigned_entries<Backend, 6>(step_to_boundary(backend, bounded, x, d))),
            (std::vector<double>{2.5, 0.0, 0.0, 2.5, 1.5, 1.0}))
      << "-1 <= x <= 4";
}

// On the interval [2, 2] every entry lies on both bounds: the projection is 2, and no direction is feasible.
template <class Backend>
void expect_a_point_interval_admits_no_step() {
  const Backend backend;
  const BoundedSpace<double> point{2.0, 2.0};
  const auto x = make_backend_vector<Backend>(std::array{2.0, 2.0});
  const auto g = make_backend_vector<Backend>(std::array{1.0, -1.0});
  const auto y = make_backend_vector<Backend>(std::array{-3.0, 5.0});
  EXPECT_EQ((assigned_entries<Backend, 2>(projection(backend, point, y))), (std::vector<double>{2.0, 2.0}));
  EXPECT_EQ((assigned_entries<Backend, 2>(projected_gradient(backend, point, x, g))), (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ((assigned_entries<Backend, 2>(free_part(backend, point, x, g))), (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ((assigned_entries<Backend, 2>(step_to_boundary(backend, point, x, g))), (std::vector<double>{0.0, 0.0}));
}

// Within epsilon of a bound counts as on it, wherever the bound lies: -1 + eps/2 is on -1 and -1 + 2 eps is not, and
// the double next below 4, 4 - 2 eps, is not on 4.
template <class Backend>
void expect_a_bound_is_within_epsilon() {
  const Backend backend;
  const double eps = std::numeric_limits<double>::epsilon();
  const auto x =
      make_backend_vector<Backend>(std::array{-1.0 + eps / 2.0, -1.0 + 2.0 * eps, std::nextafter(4.0, 0.0), 4.0});
  const auto v = make_backend_vector<Backend>(std::array{1.0, 1.0, 1.0, 1.0});
  EXPECT_EQ((assigned_entries<Backend, 4>(free_part(backend, bounded, x, v))),
            (std::vector<double>{0.0, 1.0, 1.0, 0.0}));
}

// Keep KOKKOS_LAMBDA outside GTest bodies for CUDA. The kernel holds the reference vectors as MundyMath vectors and
// writes back the geometry of x >= -1 computed from them: the projection of y, and the projected gradient, free part,
// and step to the boundary at x for g.
void evaluate_lower_bound_geometry_in_kernel(const kokkos_vector_t& projected, const kokkos_vector_t& gradient,
                                             const kokkos_vector_t& free, const kokkos_vector_t& steps) {
  Kokkos::parallel_for(
      "evaluate_lower_bound_geometry_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(const size_t) {
        const mm_backend_t backend;
        const LowerBoundSpace<double> space{-1.0};
        const Vector<double, 6> x{-1.0, -1.0, 4.0, 4.0, 1.0, 1.0};
        const Vector<double, 6> g{2.0, -2.0, 2.0, -2.0, 2.0, -2.0};
        const Vector<double, 6> y{-3.0, -1.0, 0.5, 4.0, 6.0, 1.0};
        Vector<double, 6> p;
        Vector<double, 6> q;
        Vector<double, 6> r;
        Vector<double, 6> s;
        mm_backend_t::assign("geometry", p, projection(backend, space, y), q, projected_gradient(backend, space, x, g),
                             r, free_part(backend, space, x, g), s, step_to_boundary(backend, space, x, g));
        for (size_t i = 0; i < 6; ++i) {
          projected(i) = p[i];
          gradient(i) = q[i];
          free(i) = r[i];
          steps(i) = s[i];
        }
      });
}
//@}

//! \name Geometry of the four spaces
//@{

TEST(ConvexSpaces, ProjectionIsTheNearestFeasiblePoint) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_projection_is_the_nearest_feasible_point<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_projection_is_the_nearest_feasible_point<kokkos_backend_t>();
  }
}

TEST(ConvexSpaces, ProjectedGradientDropsTheInfeasiblePart) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_projected_gradient_drops_the_infeasible_part<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_projected_gradient_drops_the_infeasible_part<kokkos_backend_t>();
  }
}

TEST(ConvexSpaces, FreePartZeroesTheEntriesOnABound) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_free_part_zeroes_the_entries_on_a_bound<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_free_part_zeroes_the_entries_on_a_bound<kokkos_backend_t>();
  }
}

TEST(ConvexSpaces, StepToBoundaryIsTheLargestFeasibleStep) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_step_to_boundary_is_the_largest_feasible_step<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_step_to_boundary_is_the_largest_feasible_step<kokkos_backend_t>();
  }
}
//@}

//! \name Bounds
//@{

TEST(ConvexSpaces, APointIntervalAdmitsNoStep) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_a_point_interval_admits_no_step<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_a_point_interval_admits_no_step<kokkos_backend_t>();
  }
}

TEST(ConvexSpaces, ABoundIsWithinEpsilon) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_a_bound_is_within_epsilon<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_a_bound_is_within_epsilon<kokkos_backend_t>();
  }
}
//@}

//! \name Device execution
//@{

TEST(ConvexSpaces, GeometryEvaluatesInsideAKernel) {
  const kokkos_vector_t projected("projected", 6);
  const kokkos_vector_t gradient("gradient", 6);
  const kokkos_vector_t free("free", 6);
  const kokkos_vector_t steps("steps", 6);
  evaluate_lower_bound_geometry_in_kernel(projected, gradient, free, steps);
  EXPECT_EQ(host_entries(projected), (std::vector<double>{-1.0, -1.0, 0.5, 4.0, 6.0, 1.0}));
  EXPECT_EQ(host_entries(gradient), (std::vector<double>{0.0, -2.0, 2.0, -2.0, 2.0, -2.0}));
  EXPECT_EQ(host_entries(free), (std::vector<double>{0.0, 0.0, 2.0, -2.0, 2.0, -2.0}));
  EXPECT_EQ(host_entries(steps), (std::vector<double>{inf, 0.0, inf, 2.5, inf, 1.0}));
}
//@}

}  // namespace

}  // namespace mundy
