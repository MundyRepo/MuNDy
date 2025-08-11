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
#include <mundy_geom/periodicity.hpp>  // for mundy::geom::PeriodicMetric, ...
#include <mundy_geom/primitives.hpp>   // for mundy::geom::Point, mundy::geom::LineSegment
#include <mundy_math/Tolerance.hpp>    // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>      // for mundy::math::Vector3

namespace mundy {

namespace geom {

namespace {

//! \name Random primitive generation
//@{

template <typename Scalar, typename RNG>
Point<Scalar> generate_random_point(const AABB<Scalar>& box, RNG& rng) {
  // Generate a random point within the bounding box
  Scalar x = rng.uniform(box.x_min(), box.x_max());
  Scalar y = rng.uniform(box.y_min(), box.y_max());
  Scalar z = rng.uniform(box.z_min(), box.z_max());
  return Point<Scalar>(x, y, z);
}

template <typename Scalar, typename RNG>
Point<Scalar> generate_random_unit_vector(const AABB<Scalar>& box, RNG& rng) {
  constexpr Scalar two_pi = 2.0 * Kokkos::numbers::pi_v<Scalar>;

  const Scalar zrand = rng.template rand<Scalar>() - static_cast<Scalar>(1);
  const Scalar wrand = Kokkos::sqrt(static_cast<Scalar>(1) - zrand * zrand);
  const Scalar trand = two_pi * rng.template rand<Scalar>();

  return mundy::math::Vector3<Scalar>{wrand * Kokkos::cos(trand), wrand * Kokkos::sin(trand), zrand};
}

template <typename Scalar, typename RNG>
Line<Scalar> generate_random_line(const AABB<Scalar>& box, RNG& rng) {
  // Generate random point in the domain and a random direction
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);
  mundy::math::Vector3<Scalar> direction = generate_random_unit_vector<Scalar>(box, rng);
  return Line<Scalar>(center, direction);
}

template <typename Scalar, typename RNG>
LineSegment<Scalar> generate_random_line_segment(const AABB<Scalar>& box, RNG& rng) {
  // Unknown if the following is necessary or not. We'll keep it until debugging is over.
  // // Generate two random points within the bounding box with lengths less than 0.5 times the smallest box size
  // double min_width =
  //     Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());

  // while (true) {
  //   Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
  //   Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);
  //   if (distance(p1, p2) < 0.5 * min_width) {
  //     return LineSegment<Scalar>(p1, p2);
  //   }
  // }

  // Generate two random points within the bounding box
  Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
  Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);
  return LineSegment<Scalar>(p1, p2);
}

template <typename Scalar, typename RNG>
Sphere<Scalar> generate_random_sphere(const AABB<Scalar>& box, RNG& rng) {
  // Generate a random center point within the bounding box
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);

  // Generate a random radius (between 0 and 0.25 times the smallest box size)
  Scalar min_width =
      Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
  Scalar radius = rng.uniform(0.0, 0.25 * min_width);
  return Sphere<Scalar>(center, radius);
}

template <typename Scalar, typename RNG>
Ellipsoid<Scalar> generate_random_ellipsoid(const AABB<Scalar>& box, RNG& rng) {
  // Generate a random center point within the bounding box
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);

  // Generate random semi-axis radii (between 0 and 0.25 times the smallest box size)
  Scalar min_width =
      Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
  Scalar r0 = rng.uniform(0.0, 0.25 * min_width);
  Scalar r1 = rng.uniform(0.0, 0.25 * min_width);
  Scalar r2 = rng.uniform(0.0, 0.25 * min_width);

  // Random orientation
  mundy::math::Vector3<Scalar> z_hat{0.0, 0.0, 1.0};
  mundy::math::Vector3<Scalar> u_hat = generate_random_unit_vector<Scalar>(box, rng);
  auto random_quaternion = mundy::math::quat_from_parallel_transport(z_hat, u_hat);

  return Ellipsoid<Scalar>(center, random_quaternion, r0, r1, r2);
}

template <typename Scalar, typename RNG>
Circle3D<Scalar> generate_random_circle3D(const AABB<Scalar>& box, RNG& rng) {
  // Generate a random center point within the bounding box
  Point<Scalar> center = generate_random_point<Scalar>(box, rng);

  // Generate a random radius (between 0 and 0.5 times the smallest box size)
  Scalar min_width =
      Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
  Scalar radius = rng.uniform(0.0, 0.25 * min_width);

  // Generate a random quaternion orientation rotating the circle's normal from z-axis to a random unit vector
  mundy::math::Vector3<Scalar> z_hat{0.0, 0.0, 1.0};
  mundy::math::Vector3<Scalar> u_hat = generate_random_unit_vector<Scalar>(box, rng);
  auto random_quaternion = mundy::math::quat_from_parallel_transport(z_hat, u_hat);

  return Circle3D<Scalar>(center, random_quaternion, radius);
}
//@}

//! \name Translate a primitive
//@{

template <typename Scalar>
Point<Scalar> translate(const Point<Scalar>& point, const mundy::math::Vector3<Scalar>& disp) {
  return Point<Scalar>(point[0] + disp[0], point[1] + disp[1], point[2] + disp[2]);
}

template <typename Scalar>
Line<Scalar> translate(const Line<Scalar>& line, const mundy::math::Vector3<Scalar>& disp) {
  return Line<Scalar>(translate(line.center(), disp), line.direction());
}

template <typename Scalar>
LineSegment<Scalar> translate(const LineSegment<Scalar>& line_segment, const mundy::math::Vector3<Scalar>& disp) {
  return LineSegment<Scalar>(translate(line_segment.start(), disp), translate(line_segment.end(), disp));
}

template <typename Scalar>
Sphere<Scalar> translate(const Sphere<Scalar>& sphere, const mundy::math::Vector3<Scalar>& disp) {
  return Sphere<Scalar>(translate(sphere.center(), disp), sphere.radius());
}

template <typename Scalar>
Ellipsoid<Scalar> translate(const Ellipsoid<Scalar>& ellipsoid, const mundy::math::Vector3<Scalar>& disp) {
  return Ellipsoid<Scalar>(translate(ellipsoid.center(), disp), ellipsoid.orientation(), ellipsoid.radii());
}

template <typename Scalar>
Circle3D<Scalar> translate(const Circle3D<Scalar>& circle, const mundy::math::Vector3<Scalar>& disp) {
  return Circle3D<Scalar>(translate(circle.center(), disp), circle.orientation(), circle.radius());
}
//@}

//! \name Runtime to compile-time dispatch
//@{

enum class TestObjectType : std::uint8_t {
  POINT = 0,
  LINE,
  LINE_SEGMENT,
  SPHERE,
  ELLIPSOID,
  CIRCLE_3D,
  NUM_TYPES,
  INVALID = std::numeric_limits<std::uint8_t>::max()
};

/// \brief Ostream overload to print TestObjectType
inline std::ostream& operator<<(std::ostream& os, const TestObjectType& type) {
  switch (type) {
    case TestObjectType::POINT:
      os << "POINT";
      break;
    case TestObjectType::LINE:
      os << "LINE";
      break;
    case TestObjectType::LINE_SEGMENT:
      os << "LINE_SEGMENT";
      break;
    case TestObjectType::SPHERE:
      os << "SPHERE";
      break;
    case TestObjectType::ELLIPSOID:
      os << "ELLIPSOID";
      break;
    case TestObjectType::CIRCLE_3D:
      os << "CIRCLE_3D";
      break;
    default:
      os << "INVALID";
  }
  return os;
}

template <TestObjectType Type>
struct TestObjectTraits;

template <>
struct TestObjectTraits<TestObjectType::POINT> {
  using type = Point<double>;
  static constexpr TestObjectType object_type = TestObjectType::POINT;

  // Function to generate a random point within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_point<double>(box, rng);
  }

  // Apply a function to each point
  template <typename Functor>
  KOKKOS_FUNCTION static void for_each_point(const type& point, const Functor& functor) {
    functor(point);
  }
};

template <>
struct TestObjectTraits<TestObjectType::LINE> {
  using type = Line<double>;
  static constexpr TestObjectType object_type = TestObjectType::LINE;

  // Function to generate a random line within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_line<double>(box, rng);
  }

  // Apply a function to each point on the line
  template <typename Functor>
  KOKKOS_FUNCTION static void for_each_point(const type& line, const Functor& functor) {
    functor(line.center());
  }
};

template <>
struct TestObjectTraits<TestObjectType::LINE_SEGMENT> {
  using type = LineSegment<double>;
  static constexpr TestObjectType object_type = TestObjectType::LINE_SEGMENT;

  // Function to generate a random line segment within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_line_segment<double>(box, rng);
  }

  // Apply a function to each point on the line segment
  template <typename Functor>
  KOKKOS_FUNCTION static void for_each_point(const type& line_segment, const Functor& functor) {
    functor(line_segment.start());
    functor(line_segment.end());
  }
};

template <>
struct TestObjectTraits<TestObjectType::SPHERE> {
  using type = Sphere<double>;
  static constexpr TestObjectType object_type = TestObjectType::SPHERE;

  // Function to generate a random sphere within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_sphere<double>(box, rng);
  }

  // Apply a function to each point on the sphere
  template <typename Functor>
  KOKKOS_FUNCTION static void for_each_point(const type& sphere, const Functor& functor) {
    // For a sphere, we can apply the function to the center point
    functor(sphere.center());
  }
};

template <>
struct TestObjectTraits<TestObjectType::ELLIPSOID> {
  using type = Ellipsoid<double>;
  static constexpr TestObjectType object_type = TestObjectType::ELLIPSOID;

  // Function to generate a random ellipsoid within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_ellipsoid<double>(box, rng);
  }

  // Apply a function to each point on the ellipsoid
  template <typename Functor>
  KOKKOS_FUNCTION static void for_each_point(const type& ellipsoid, const Functor& functor) {
    // For an ellipsoid, we can apply the function to the center point
    functor(ellipsoid.center());
  }
};

template <>
struct TestObjectTraits<TestObjectType::CIRCLE_3D> {
  using type = Circle3D<double>;
  static constexpr TestObjectType object_type = TestObjectType::CIRCLE_3D;

  // Function to generate a random circle3D within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_circle3D<double>(box, rng);
  }

  // Apply a function to each point on the circle3D
  template <typename Functor>
  KOKKOS_FUNCTION static void for_each_point(const type& circle, const Functor& functor) {
    // For a circle3D, we can apply the function to the center point
    functor(circle.center());
  }
};

template <typename Functor>
struct apply_functor {
  using return_type = typename Functor::return_type;

  KOKKOS_DEFAULTED_FUNCTION
  apply_functor() = default;

  KOKKOS_FUNCTION
  apply_functor(const Functor& functor) : functor_(functor) {
  }

  template <typename... Args>
  KOKKOS_FUNCTION return_type operator()(TestObjectType runtime_type, Args&&... args) const {
    MUNDY_THROW_ASSERT(runtime_type != TestObjectType::INVALID, std::invalid_argument,
                       "Invalid test object type provided.");

    if (runtime_type1 == TestObjectType::POINT) {
      return functor_(TestObjectTraits<TestObjectType::POINT>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::LINE) {
      return functor_(TestObjectTraits<TestObjectType::LINE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::LINE_SEGMENT) {
      return functor_(TestObjectTraits<TestObjectType::LINE_SEGMENT>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::SPHERE) {
      return functor_(TestObjectTraits<TestObjectType::SPHERE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::ELLIPSOID) {
      return functor_(TestObjectTraits<TestObjectType::ELLIPSOID>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::CIRCLE_3D) {
      return functor_(TestObjectTraits<TestObjectType::CIRCLE_3D>{},  //
                      std::forward<Args>(args)...);
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unsupported combination of test object types.");
    }

    return return_type{};  // Fallback return value, should never be reached
  }

  Functor functor_;
};  // apply_functor
//@}

KOKKOS_INLINE_FUNCTION bool is_point_in_box(const Point<double>& point, const AABB<double>& box) {
  return (point[0] >= box.x_min() && point[0] <= box.x_max() && point[1] >= box.y_min() && point[1] <= box.y_max() &&
          point[2] >= box.z_min() && point[2] <= box.z_max());
}

struct test_wrap_points_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& box, RNG& rng,
                                                const Metric& metric) const {
    auto s = ShapeTraits::generate(box, rng);
    wrap_points(s, metric);
    bool all_in_primary_domain = true;
    ShapeTraits::for_each_point(s, [&](const auto& point) {
      if (!is_point_in_box(point, box)) {
        all_in_primary_domain = false;
      }
    });
    return all_in_primary_domain;
  }
};

KOKKOS_INLINE_FUNCTION double test_wrap_points(TestObjectType type, const AABB<double>& box, size_t seed,
                                               size_t counter) {
  openrand::Philox rng(seed, counter);

  using Functor = test_wrap_points_impl;
  apply_functor<Functor> apply;
  return apply(type, box, rng);
}

//! \brief Unit tests
//@{

TEST(PeriodicMetric, MinImageDirectVsPeriodic) {
  size_t seed = 1234;
  size_t num_samples = 100000;

  math::Vector3<double> cell_size{100.0, 100.0, 100.0};
  AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  auto periodic_metric = periodic_metric_from_unit_cell(cell_size);
  auto periodic_metric_scale_only = periodic_scaled_metric_from_unit_cell(cell_size);
  EuclideanMetric euclidean_metric{};

  for (size_t t = 0; t < num_samples; ++t) {
    openrand::Philox rng(seed, t);

    // Generate two random points within the bounding box
    Point<double> point1 = generate_random_point<double>(box, rng);
    Point<double> point2 = generate_random_point<double>(box, rng);

    // Compute the minimum image distance using the free-space metric and the 27 periodic images
    mundy::math::Vector<double, 27> min_image_distances;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          // Shift obj2 by the box dimensions in each direction
          mundy::math::Vector3<double> disp((i - 1) * cell_size[0], (j - 1) * cell_size[1], (k - 1) * cell_size[2]);
          min_image_distances(i * 9 + j * 3 + k) = norm(euclidean_metric.sep(point1, point2 + disp));
        }
      }
    }

    double min_image_distance = min(min_image_distances);
    double periodic_distance = norm(periodic_metric.sep(point1, point2));
    double periodic_distance_scale_only = norm(periodic_metric_scale_only.sep(point1, point2));
    ASSERT_NEAR(min_image_distance, periodic_distance, mundy::math::get_relaxed_zero_tolerance<double>())
        << "Minimum image distance does not match periodic distance.";
    ASSERT_NEAR(min_image_distance, periodic_distance_scale_only, mundy::math::get_relaxed_zero_tolerance<double>())
        << "Minimum image distance does not match periodic distance (scale only).";
  }
}

TEST(PeriodicMetric, WrapRigid) {
}

}  // namespace

}  // namespace geom

}  // namespace mundy
