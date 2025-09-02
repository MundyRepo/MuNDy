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
#include <mundy_geom/randomize.hpp>    // for mundy::geom::generate_random_point, ...
#include <mundy_geom/transform.hpp>    // for mundy::geom::translate
#include <mundy_math/Tolerance.hpp>    // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>      // for mundy::math::Vector3

namespace mundy {

namespace geom {

namespace {

// //! \name Random primitive generation
// //@{

// template <typename Scalar, typename RNG>
// Point<Scalar> generate_random_point(const AABB<Scalar>& box, RNG& rng) {
//   // Generate a random point within the bounding box
//   Scalar x = rng.uniform(box.x_min(), box.x_max());
//   Scalar y = rng.uniform(box.y_min(), box.y_max());
//   Scalar z = rng.uniform(box.z_min(), box.z_max());
//   return Point<Scalar>(x, y, z);
// }

// template <typename Scalar, typename RNG>
// Point<Scalar> generate_random_unit_vector(const AABB<Scalar>& box, RNG& rng) {
//   constexpr Scalar two_pi = 2.0 * Kokkos::numbers::pi_v<Scalar>;

//   const Scalar zrand = rng.template rand<Scalar>() - static_cast<Scalar>(1);
//   const Scalar wrand = Kokkos::sqrt(static_cast<Scalar>(1) - zrand * zrand);
//   const Scalar trand = two_pi * rng.template rand<Scalar>();

//   return math::Vector3<Scalar>{wrand * Kokkos::cos(trand), wrand * Kokkos::sin(trand), zrand};
// }

// template <typename Scalar, typename RNG>
// Line<Scalar> generate_random_line(const AABB<Scalar>& box, RNG& rng) {
//   // Generate random point in the domain and a random direction
//   Point<Scalar> center = generate_random_point<Scalar>(box, rng);
//   math::Vector3<Scalar> direction = generate_random_unit_vector<Scalar>(box, rng);
//   return Line<Scalar>(center, direction);
// }

// template <typename Scalar, typename RNG>
// LineSegment<Scalar> generate_random_line_segment(const AABB<Scalar>& box, RNG& rng) {
//   // Unknown if the following is necessary or not. We'll keep it until debugging is over.
//   // // Generate two random points within the bounding box with lengths less than 0.5 times the smallest box size
//   // double min_width =
//   //     Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());

//   // while (true) {
//   //   Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
//   //   Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);
//   //   if (distance(p1, p2) < 0.5 * min_width) {
//   //     return LineSegment<Scalar>(p1, p2);
//   //   }
//   // }

//   // Generate two random points within the bounding box
//   Point<Scalar> p1 = generate_random_point<Scalar>(box, rng);
//   Point<Scalar> p2 = generate_random_point<Scalar>(box, rng);
//   return LineSegment<Scalar>(p1, p2);
// }

// template <typename Scalar, typename RNG>
// Sphere<Scalar> generate_random_sphere(const AABB<Scalar>& box, RNG& rng) {
//   // Generate a random center point within the bounding box
//   Point<Scalar> center = generate_random_point<Scalar>(box, rng);

//   // Generate a random radius (between 0 and 0.25 times the smallest box size)
//   Scalar min_width =
//       Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
//   Scalar radius = rng.uniform(0.0, 0.25 * min_width);
//   return Sphere<Scalar>(center, radius);
// }

// template <typename Scalar, typename RNG>
// Ellipsoid<Scalar> generate_random_ellipsoid(const AABB<Scalar>& box, RNG& rng) {
//   // Generate a random center point within the bounding box
//   Point<Scalar> center = generate_random_point<Scalar>(box, rng);

//   // Generate random semi-axis radii (between 0 and 0.25 times the smallest box size)
//   Scalar min_width =
//       Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
//   Scalar r0 = rng.uniform(0.0, 0.25 * min_width);
//   Scalar r1 = rng.uniform(0.0, 0.25 * min_width);
//   Scalar r2 = rng.uniform(0.0, 0.25 * min_width);

//   // Random orientation
//   math::Vector3<Scalar> z_hat{0.0, 0.0, 1.0};
//   math::Vector3<Scalar> u_hat = generate_random_unit_vector<Scalar>(box, rng);
//   auto random_quaternion = math::quat_from_parallel_transport(z_hat, u_hat);

//   return Ellipsoid<Scalar>(center, random_quaternion, r0, r1, r2);
// }

// template <typename Scalar, typename RNG>
// Circle3D<Scalar> generate_random_circle3D(const AABB<Scalar>& box, RNG& rng) {
//   // Generate a random center point within the bounding box
//   Point<Scalar> center = generate_random_point<Scalar>(box, rng);

//   // Generate a random radius (between 0 and 0.5 times the smallest box size)
//   Scalar min_width =
//       Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
//   Scalar radius = rng.uniform(0.0, 0.25 * min_width);

//   // Generate a random quaternion orientation rotating the circle's normal from z-axis to a random unit vector
//   math::Vector3<Scalar> z_hat{0.0, 0.0, 1.0};
//   math::Vector3<Scalar> u_hat = generate_random_unit_vector<Scalar>(box, rng);
//   auto random_quaternion = math::quat_from_parallel_transport(z_hat, u_hat);

//   return Circle3D<Scalar>(center, random_quaternion, radius);
// }
// //@}

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
  static constexpr unsigned num_points = 1;

  // Function to generate a random point within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_point<double>(box, rng);
  }

  // Function to fetch the reference point
  static KOKKOS_FUNCTION Point<double> reference_point(const type& point) {
    return point;
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
  static constexpr unsigned num_points = 1;

  // Function to generate a random line within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_line<double>(box, rng);
  }

  // Function to fetch the reference point
  static KOKKOS_FUNCTION Point<double> reference_point(const type& line) {
    return line.center();
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
  static constexpr unsigned num_points = 2;

  // Function to generate a random line segment within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_line_segment<double>(box, rng);
  }

  // Function to fetch the reference point
  static KOKKOS_FUNCTION Point<double> reference_point(const type& line_segment) {
    return line_segment.start();
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
  static constexpr unsigned num_points = 1;

  // Function to generate a random sphere within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    double min_box_width =
        Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
    double min_radius = 0.0;
    double max_radius = 0.25 * min_box_width;
    return generate_random_sphere<double>(box, min_radius, max_radius, rng);
  }

  // Function to fetch the reference point
  static KOKKOS_FUNCTION Point<double> reference_point(const type& sphere) {
    return sphere.center();
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
  static constexpr unsigned num_points = 1;

  // Function to generate a random ellipsoid within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    double min_box_width =
        Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
    double min_radius = 0.0;
    double max_radius = 0.25 * min_box_width;
    return generate_random_ellipsoid<double>(box, math::Vector3<double>{min_radius, min_radius, min_radius},
                                             math::Vector3<double>{max_radius, max_radius, max_radius}, rng);
  }

  // Function to fetch the reference point
  static KOKKOS_FUNCTION Point<double> reference_point(const type& ellipsoid) {
    return ellipsoid.center();
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
  static constexpr unsigned num_points = 1;

  // Function to generate a random circle3D within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    double min_box_width =
        Kokkos::min(Kokkos::min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
    double min_radius = 0.0;
    double max_radius = 0.25 * min_box_width;
    return generate_random_circle3D<double>(box, min_radius, max_radius, rng);
  }

  // Function to fetch the reference point
  static KOKKOS_FUNCTION Point<double> reference_point(const type& circle) {
    return circle.center();
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

    if (runtime_type == TestObjectType::POINT) {
      return functor_(TestObjectTraits<TestObjectType::POINT>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type == TestObjectType::LINE) {
      return functor_(TestObjectTraits<TestObjectType::LINE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type == TestObjectType::LINE_SEGMENT) {
      return functor_(TestObjectTraits<TestObjectType::LINE_SEGMENT>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type == TestObjectType::SPHERE) {
      return functor_(TestObjectTraits<TestObjectType::SPHERE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type == TestObjectType::ELLIPSOID) {
      return functor_(TestObjectTraits<TestObjectType::ELLIPSOID>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type == TestObjectType::CIRCLE_3D) {
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

struct test_wrap_rigid_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& primary_box,
                                                const AABB<double>& other_box, RNG& rng, const Metric& metric) const {
    // Generate a random shape within other_box and wrap it to the primary box (rigidly)
    //
    // The reference point should end up within the primary box and all others should maintain the same relative
    // positions
    constexpr unsigned num_points = ShapeTraits::num_points;
    Kokkos::Array<math::Vector3<double>, num_points> original_displacements;
    Kokkos::Array<math::Vector3<double>, num_points> wrapped_displacements;

    auto s = ShapeTraits::generate(other_box, rng);
    Point<double> ref_point = ShapeTraits::reference_point(s);

    unsigned i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& point) {
      original_displacements[i] = point - ref_point;
      ++i;
    });

    s = wrap_rigid(s, metric);
    Point<double> wrapped_ref_point = ShapeTraits::reference_point(s);

    bool ref_in_primary_domain = is_point_in_box(wrapped_ref_point, primary_box);

    i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& point) {
      wrapped_displacements[i] = point - wrapped_ref_point;
      ++i;
    });

    // Check that all displacements are the same before and after wrapping
    bool all_displacements_equal = true;
    for (unsigned j = 0; j < num_points; ++j) {
      if (math::norm(original_displacements[j] - wrapped_displacements[j]) >
          math::get_relaxed_zero_tolerance<double>()) {
        all_displacements_equal = false;
      }
    }

    return ref_in_primary_domain && all_displacements_equal;
  }
};

struct test_wrap_points_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& primary_box,
                                                const AABB<double>& disjoint_box, RNG& rng,
                                                const Metric& metric) const {
    // Generate a random shape outside the primary box and wrap it to the primary box
    auto s = ShapeTraits::generate(disjoint_box, rng);
    s = wrap_points(s, metric);

    bool all_in_primary_domain = true;
    ShapeTraits::for_each_point(s, [&](const auto& point) {
      if (!is_point_in_box(point, primary_box)) {
        all_in_primary_domain = false;
      }
    });
    return all_in_primary_domain;
  }
};

struct test_unwrap_to_ref_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& primary_box,
                                                const AABB<double>& disjoint_box, RNG& rng,
                                                const Metric& metric) const {
    // Generate a random shape within the primary box and unwrap it to a reference point
    // outside the primary box
    auto s = ShapeTraits::generate(primary_box, rng);
    Point<double> ref_point = generate_random_point<double>(disjoint_box, rng);
    s = unwrap_points_to_ref(s, metric, ref_point);

    // The shifted object should have all points within one image of the reference point
    auto primary_center = 0.5 * (primary_box.min_corner() + primary_box.max_corner());
    auto shifted_box = translate(primary_box, ref_point - primary_center);
    bool all_in_primary_domain = true;
    ShapeTraits::for_each_point(s, [&](const auto& point) {
      if (!is_point_in_box(point, shifted_box)) {
        all_in_primary_domain = false;
      }
    });
    return all_in_primary_domain;
  }
};

struct test_shift_image_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits,                               //
                                                const AABB<double>& box,                   //
                                                const math::Vector3<int>& lattice_vector,  //
                                                RNG& rng,                                  //
                                                const Metric& metric) const {
    constexpr unsigned num_points = ShapeTraits::num_points;
    Kokkos::Array<math::Vector3<double>, num_points> original_displacements;
    Kokkos::Array<math::Vector3<double>, num_points> shifted_displacements;

    auto s = ShapeTraits::generate(box, rng);
    Point<double> ref_point = ShapeTraits::reference_point(s);

    unsigned i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& point) {
      original_displacements[i] = point - ref_point;
      ++i;
    });

    s = shift_image(s, lattice_vector, metric);
    Point<double> shifted_ref_point = ShapeTraits::reference_point(s);

    auto shifted_ref_displacement = shifted_ref_point - ref_point;
    auto expected_displacement = metric.shift_image(ref_point, lattice_vector) - ref_point;
    bool displacements_match =
        math::norm(shifted_ref_displacement - expected_displacement) < math::get_relaxed_zero_tolerance<double>();

    i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& point) {
      shifted_displacements[i] = point - shifted_ref_point;
      ++i;
    });

    // Check that all displacements are the same before and after wrapping
    bool all_displacements_equal = true;
    for (unsigned j = 0; j < num_points; ++j) {
      if (math::norm(original_displacements[j] - shifted_displacements[j]) >
          math::get_relaxed_zero_tolerance<double>()) {
        all_displacements_equal = false;
      }
    }

    return displacements_match && all_displacements_equal;
  }
};

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_wrap_rigid(TestObjectType type, const AABB<double>& primary_box,
                                            const AABB<double>& other_box, size_t seed, size_t counter,
                                            const Metric& metric) {
  openrand::Philox rng(seed, counter);

  using Functor = test_wrap_rigid_impl;
  apply_functor<Functor> apply;
  return apply(type, primary_box, other_box, rng, metric);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_wrap_points(TestObjectType type, const AABB<double>& primary_box,
                                             const AABB<double>& disjoint_box, size_t seed, size_t counter,
                                             const Metric& metric) {
  openrand::Philox rng(seed, counter);

  using Functor = test_wrap_points_impl;
  apply_functor<Functor> apply;
  return apply(type, primary_box, disjoint_box, rng, metric);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_unwrap_to_ref(TestObjectType type, const AABB<double>& primary_box,
                                               const AABB<double>& disjoint_box, size_t seed, size_t counter,
                                               const Metric& metric) {
  openrand::Philox rng(seed, counter);

  using Functor = test_unwrap_to_ref_impl;
  apply_functor<Functor> apply;
  return apply(type, primary_box, disjoint_box, rng, metric);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_shift_image(TestObjectType type, const AABB<double>& box,
                                             const math::Vector3<int>& lattice_vector,  //
                                             size_t seed, size_t counter, const Metric& metric) {
  openrand::Philox rng(seed, counter);

  using Functor = test_shift_image_impl;
  apply_functor<Functor> apply;
  return apply(type, box, lattice_vector, rng, metric);
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
    math::Vector<double, 27> min_image_distances;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          // Shift obj2 by the box dimensions in each direction
          math::Vector3<double> disp((i - 1) * cell_size[0], (j - 1) * cell_size[1], (k - 1) * cell_size[2]);
          min_image_distances(i * 9 + j * 3 + k) = norm(euclidean_metric.sep(point1, point2 + disp));
        }
      }
    }

    double min_image_distance = min(min_image_distances);
    double periodic_distance = norm(periodic_metric.sep(point1, point2));
    double periodic_distance_scale_only = norm(periodic_metric_scale_only.sep(point1, point2));
    ASSERT_NEAR(min_image_distance, periodic_distance, math::get_relaxed_zero_tolerance<double>())
        << "Minimum image distance does not match periodic distance.";
    ASSERT_NEAR(min_image_distance, periodic_distance_scale_only, math::get_relaxed_zero_tolerance<double>())
        << "Minimum image distance does not match periodic distance (scale only).";
  }
}

TEST(PeriodicMetric, WrapRigid) {
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 1000;  // Number of trials for each pair

  math::Vector3<double> cell_size{100.0, 100.0, 100.0};
  AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  auto overlapping_box = translate(box, math::Vector3<double>{50.0, 50.0, 50.0});

  auto periodic_metric = periodic_metric_from_unit_cell(cell_size);
  auto periodic_metric_scale_only = periodic_scaled_metric_from_unit_cell(cell_size);
  EuclideanMetric euclidean_metric{};

  std::vector<TestObjectType> test_types = {TestObjectType::POINT,        TestObjectType::LINE,
                                            TestObjectType::LINE_SEGMENT, TestObjectType::SPHERE,
                                            TestObjectType::ELLIPSOID,    TestObjectType::CIRCLE_3D};

  for (const auto& type : test_types) {
    for (size_t t = 0; t < num_trials; ++t) {
      if (type != TestObjectType::LINE) {
        EXPECT_TRUE(test_wrap_rigid(type, box, overlapping_box, seed, counter, periodic_metric))
            << "Rigid wrapping for type " << type << " failed. For the periodic metric.";
        EXPECT_TRUE(test_wrap_rigid(type, box, overlapping_box, seed, counter, periodic_metric_scale_only))
            << "Rigid wrapping for type " << type << " failed. For the periodic metric (scale only).";

        // For the aperiodic metric, rigid wrapping should just return the original point
        EXPECT_TRUE(test_wrap_rigid(type, box, box, seed, counter, euclidean_metric))
            << "Rigid wrapping for type " << type << " failed. For the free space metric.";
      } else {
        // Expect a "Not implemented error"
        EXPECT_THROW(test_wrap_rigid(type, box, overlapping_box, seed, counter, periodic_metric), std::invalid_argument)
            << "Rigid wrapping for type " << type << " should throw an error for the periodic metric.";
      }

      ++counter;
    }
  }
}

TEST(PeriodicMetric, WrapPoints) {
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 1000;  // Number of trials for each pair

  math::Vector3<double> cell_size{100.0, 100.0, 100.0};
  AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  AABB<double> disjoint_box{900.0, 900.0, 900.0, 1000.0, 1000.0, 1000.0};

  auto periodic_metric = periodic_metric_from_unit_cell(cell_size);
  auto periodic_metric_scale_only = periodic_scaled_metric_from_unit_cell(cell_size);
  EuclideanMetric euclidean_metric{};

  std::vector<TestObjectType> test_types = {TestObjectType::POINT,        TestObjectType::LINE,
                                            TestObjectType::LINE_SEGMENT, TestObjectType::SPHERE,
                                            TestObjectType::ELLIPSOID,    TestObjectType::CIRCLE_3D};

  for (const auto& type : test_types) {
    for (size_t t = 0; t < num_trials; ++t) {
      if (type != TestObjectType::LINE) {
        EXPECT_TRUE(test_wrap_points(type, box, disjoint_box, seed, counter, periodic_metric))
            << "Wrapping points for type " << type << " failed. For the periodic metric.";
        EXPECT_TRUE(test_wrap_points(type, box, disjoint_box, seed, counter, periodic_metric_scale_only))
            << "Wrapping points for type " << type << " failed. For the periodic metric (scale only).";

        // For the aperiodic metric, wrapping should just return the original point
        EXPECT_TRUE(test_wrap_points(type, disjoint_box, disjoint_box, seed, counter, euclidean_metric))
            << "Wrapping points for type " << type << " failed. For the free space metric.";
      } else {
        // Expect a "Not implemented error"
        EXPECT_THROW(test_wrap_points(type, box, disjoint_box, seed, counter, periodic_metric), std::invalid_argument)
            << "Wrapping points for type " << type << " should throw an error for the periodic metric.";
      }

      ++counter;
    }
  }
}

TEST(PeriodicMetric, UnwrapPointsToRef) {
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 1000;  // Number of trials for each pair

  math::Vector3<double> cell_size{100.0, 100.0, 100.0};
  AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  AABB<double> disjoint_box{900.0, 900.0, 900.0, 1000.0, 1000.0, 1000.0};

  auto periodic_metric = periodic_metric_from_unit_cell(cell_size);
  auto periodic_metric_scale_only = periodic_scaled_metric_from_unit_cell(cell_size);
  EuclideanMetric euclidean_metric{};

  std::vector<TestObjectType> test_types = {TestObjectType::POINT,        TestObjectType::LINE,
                                            TestObjectType::LINE_SEGMENT, TestObjectType::SPHERE,
                                            TestObjectType::ELLIPSOID,    TestObjectType::CIRCLE_3D};

  for (const auto& type : test_types) {
    for (size_t t = 0; t < num_trials; ++t) {
      if (type != TestObjectType::LINE) {
        EXPECT_TRUE(test_unwrap_to_ref(type, box, disjoint_box, seed, counter, periodic_metric))
            << "Unwrapping points for type " << type << " failed. For the periodic metric.";
        EXPECT_TRUE(test_unwrap_to_ref(type, box, disjoint_box, seed, counter, periodic_metric_scale_only))
            << "Unwrapping points for type " << type << " failed. For the periodic metric (scale only).";

        // For the aperiodic metric, unwrapping should just return the original point
        EXPECT_TRUE(test_unwrap_to_ref(type, disjoint_box, disjoint_box, seed, counter, euclidean_metric))
            << "Unwrapping points for type " << type << " failed. For the free space metric.";
      } else {
        // Expect a "Not implemented error"
        EXPECT_THROW(test_unwrap_to_ref(type, box, disjoint_box, seed, counter, periodic_metric), std::invalid_argument)
            << "Unwrapping points for type " << type << " should throw an error for the periodic metric.";
      }

      ++counter;
    }
  }
}

LineSegment<double> generate_spanning_line_segment_with_length_limit(const AABB<double>& box1, const AABB<double>& box2,
                                                                     openrand::Philox& rng, double max_length) {
  // Generate a random line segment with one end in box1 and the other in box2
  // Ensure that the length of the segment is less than max_length
  while (true) {
    Point<double> p1 = generate_random_point<double>(box1, rng);
    Point<double> p2 = generate_random_point<double>(box2, rng);
    LineSegment<double> segment(p1, p2);
    if (math::norm(p1 - p2) <= max_length) {
      return segment;
    }
  }
}

TEST(PeriodicMetric, WrapPointsSpanning) {
  // Test the special case of a line segment that crosses the domain boundary
  // Generate a random segment whose length is less then 0.5 times the smallest box size
  // with one end in the primary box and the other in the disjoint box
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 1000;  // Number of trials for each pair

  double domain_width = 100.0;
  math::Vector3<double> cell_size{domain_width, domain_width, domain_width};
  AABB<double> box{0.0, 0.0, 0.0, domain_width, domain_width, domain_width};
  auto disjoint_box = translate(box, math::Vector3<double>{domain_width, domain_width, domain_width});

  auto periodic_metric = periodic_metric_from_unit_cell(cell_size);
  auto periodic_metric_scale_only = periodic_scaled_metric_from_unit_cell(cell_size);
  EuclideanMetric euclidean_metric{};

  for (size_t t = 0; t < num_trials; ++t) {
    openrand::Philox rng(seed, counter);

    // Generate a random line segment with one end in the primary box and the other in the disjoint box
    LineSegment<double> line_segment =
        generate_spanning_line_segment_with_length_limit(box, disjoint_box, rng, 0.5 * domain_width);

    // Expected results
    auto expected_periodic_start = wrap_points(line_segment.start(), periodic_metric);
    auto expected_periodic_end = wrap_points(line_segment.end(), periodic_metric);

    ASSERT_TRUE(math::norm(expected_periodic_start - expected_periodic_end) >
                math::norm(line_segment.start() - line_segment.end()))
        << "Test setup failure: Wrapping the segment should have caused it to have a MUCH longer euclidean length.";

    // Periodic check
    auto line_segment_periodic = wrap_points(line_segment, periodic_metric);
    EXPECT_NEAR(math::norm(line_segment_periodic.start() - expected_periodic_start), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Wrapped line segment start point does not match expected value.";
    EXPECT_NEAR(math::norm(line_segment_periodic.end() - expected_periodic_end), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Wrapped line segment end point does not match expected value.";

    // Periodic check (scale only)
    auto line_segment_periodic_scale_only = wrap_points(line_segment, periodic_metric_scale_only);
    EXPECT_NEAR(math::norm(line_segment_periodic_scale_only.start() - expected_periodic_start), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Wrapped line segment start point does not match expected value (scale only).";
    EXPECT_NEAR(math::norm(line_segment_periodic_scale_only.end() - expected_periodic_end), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Wrapped line segment end point does not match expected value (scale only).";

    // Aperiodic check (should be unchanged)
    auto line_segment_aperiodic = wrap_points(line_segment, euclidean_metric);
    EXPECT_NEAR(math::norm(line_segment_aperiodic.start() - line_segment.start()), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Wrapped line segment start point does not match original value for the free space metric.";
    EXPECT_NEAR(math::norm(line_segment_aperiodic.end() - line_segment.end()), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Wrapped line segment end point does not match original value for the free space metric.";

    ++counter;
  }
}

TEST(PeriodicMetric, UnwrapPointsSpanning) {
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 1000;  // Number of trials for each pair

  double domain_width = 100.0;
  math::Vector3<double> cell_size{domain_width, domain_width, domain_width};
  AABB<double> box{0.0, 0.0, 0.0, domain_width, domain_width, domain_width};
  auto disjoint_box = translate(box, math::Vector3<double>{domain_width, domain_width, domain_width});

  auto periodic_metric = periodic_metric_from_unit_cell(cell_size);
  auto periodic_metric_scale_only = periodic_scaled_metric_from_unit_cell(cell_size);
  EuclideanMetric euclidean_metric{};

  for (size_t t = 0; t < num_trials; ++t) {
    openrand::Philox rng(seed, counter);

    // Generate a random line segment with one end in the primary box and the other in the disjoint box
    LineSegment<double> line_segment =
        generate_spanning_line_segment_with_length_limit(box, disjoint_box, rng, 0.5 * domain_width);

    // No-op check (should be unchanged)
    auto line_segment_unwrapped = unwrap_points_to_ref(line_segment, periodic_metric, line_segment.start());
    EXPECT_NEAR(math::norm(line_segment_unwrapped.start() - line_segment.start()), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Unwrapped line segment start point should not change.";
    EXPECT_NEAR(math::norm(line_segment_unwrapped.end() - line_segment.end()), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Unwrapped line segment end point should not change.";

    // Wrap points to break the length. Then unwrap to get back to the original segment
    double expected_length = math::norm(line_segment.start() - line_segment.end());
    auto line_segment_wrapped = wrap_points(line_segment, periodic_metric);
    double wrapped_length = math::norm(line_segment_wrapped.start() - line_segment_wrapped.end());
    auto line_segment_unwrapped_again =
        unwrap_points_to_ref(line_segment_wrapped, periodic_metric, line_segment.start());
    double unwrapped_length = math::norm(line_segment_unwrapped_again.start() - line_segment_unwrapped_again.end());
    EXPECT_TRUE(expected_length < wrapped_length)
        << "Test setup failure: Wrapping the segment should have caused it to have a MUCH longer euclidean length.";
    EXPECT_NEAR(expected_length, unwrapped_length, math::get_relaxed_zero_tolerance<double>())
        << "Unwrapped line segment length does not match the original length.";

    // Because our reference point starts in the domain our initial and final segments should match
    EXPECT_NEAR(math::norm(line_segment_unwrapped_again.start() - line_segment.start()), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Unwrapped line segment start point does not match original value.";
    EXPECT_NEAR(math::norm(line_segment_unwrapped_again.end() - line_segment.end()), 0.0,
                math::get_relaxed_zero_tolerance<double>())
        << "Unwrapped line segment end point does not match original value.";

    ++counter;
  }
}

TEST(PeriodicMetric, ShiftImage) {
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 1;  // Number of trials for each pair

  math::Vector3<double> cell_size{100.0, 100.0, 100.0};
  AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  auto disjoint_box = translate(box, math::Vector3<double>{100.0, 100.0, 100.0});

  auto periodic_metric = periodic_metric_from_unit_cell(cell_size);
  auto periodic_metric_scale_only = periodic_scaled_metric_from_unit_cell(cell_size);
  EuclideanMetric euclidean_metric{};

  std::vector<TestObjectType> test_types = {TestObjectType::POINT,        TestObjectType::LINE,
                                            TestObjectType::LINE_SEGMENT, TestObjectType::SPHERE,
                                            TestObjectType::ELLIPSOID,    TestObjectType::CIRCLE_3D};

  for (const auto& type : test_types) {
    for (size_t t = 0; t < num_trials; ++t) {
      // Generate a random lattice vector between -10 and 10 in each direction
      openrand::Philox rng(seed, counter);
      math::Vector3<int> lattice_vector{rng.uniform<int>(-10, 10),
                                        rng.uniform<int>(-10, 10),
                                        rng.uniform<int>(-10, 10)};
      EXPECT_TRUE(test_shift_image(type, disjoint_box, lattice_vector, seed, counter, periodic_metric))
          << "Shift image for type " << type << " failed. For the periodic metric.";
      EXPECT_TRUE(test_shift_image(type, disjoint_box, lattice_vector, seed, counter, periodic_metric_scale_only))
          << "Shift image for type " << type << " failed. For the periodic metric (scale only).";
      EXPECT_TRUE(test_shift_image(type, disjoint_box, lattice_vector, seed, counter, euclidean_metric))
          << "Shift image for type " << type << " failed. For the free space metric.";

      ++counter;
    }
  }
}

}  // namespace

}  // namespace geom

}  // namespace mundy
