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
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion, mundy::math::quat_from_parallel_transport
#include <mundy_math/Tolerance.hpp>   // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3

namespace mundy {

namespace geom {

namespace {

/*
The goal of this test is to ensure that distance_pbc and distance agree for free space metric and periodic space metric
for all supported distance measures and their augmented returns.

Supported distance measures:
  (point, point)
  (point, line)
  (point, line segment)
  (point, sphere)
  (line, line)
  (line, sphere)
  (line segment, line segment)
  (line segment, sphere)
  (sphere, sphere)
  (ellipsoid, ellipsoid)
  (circle3D, circle3D)

I envision that we'll need:
  - The ability to generate a random instance of each primitive in the unit cell
  - The ability to shift each primitive by a given amount in each direction

With these two capabilities, we can generate pairs of primitives within an octant of the unit cell (to avoid
interactions with periodic images), compute their free space and periodic distances, and compare the results (they
should match). We can then shift one of the primitives by the unit cell size in each direction (ensuring that its
periodic image remains in the same location), and compute the updated periodic distance. It should still match the free
space result.
*/

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
};

template <>
struct TestObjectTraits<TestObjectType::LINE> {
  using type = Line<double>;
  static constexpr TestObjectType object_type = TestObjectType::LINE;

  // Function to generate a random line within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_line<double>(box, rng);
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
};

template <>
struct TestObjectTraits<TestObjectType::SPHERE> {
  using type = Sphere<double>;
  static constexpr TestObjectType object_type = TestObjectType::SPHERE;

  // Function to generate a random sphere within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_sphere<double>(box, rng);
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
};

template <>
struct TestObjectTraits<TestObjectType::CIRCLE_3D> {
  using type = Circle3D<double>;
  static constexpr TestObjectType object_type = TestObjectType::CIRCLE_3D;

  // Function to generate a random circle3D within a given bounding box
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_circle3D<double>(box, rng);
  }
};

template <typename Functor>
struct apply_pairwise_functor {
  using return_type = typename Functor::return_type;

  KOKKOS_DEFAULTED_FUNCTION
  apply_pairwise_functor() = default;

  KOKKOS_FUNCTION
  apply_pairwise_functor(const Functor& functor) : functor_(functor) {
  }

  template <typename... Args>
  KOKKOS_FUNCTION return_type operator()(TestObjectType runtime_type1, TestObjectType runtime_type2,
                                         Args&&... args) const {
    MUNDY_THROW_ASSERT(runtime_type1 != TestObjectType::INVALID && runtime_type2 != TestObjectType::INVALID,
                       std::invalid_argument, "Invalid test object type provided.");

    if (runtime_type1 == TestObjectType::POINT && runtime_type2 == TestObjectType::POINT) {
      return functor_(TestObjectTraits<TestObjectType::POINT>{},  //
                      TestObjectTraits<TestObjectType::POINT>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::POINT && runtime_type2 == TestObjectType::LINE) {
      return functor_(TestObjectTraits<TestObjectType::POINT>{},  //
                      TestObjectTraits<TestObjectType::LINE>{},   //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::POINT && runtime_type2 == TestObjectType::LINE_SEGMENT) {
      return functor_(TestObjectTraits<TestObjectType::POINT>{},         //
                      TestObjectTraits<TestObjectType::LINE_SEGMENT>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::POINT && runtime_type2 == TestObjectType::SPHERE) {
      return functor_(TestObjectTraits<TestObjectType::POINT>{},   //
                      TestObjectTraits<TestObjectType::SPHERE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::LINE && runtime_type2 == TestObjectType::LINE) {
      return functor_(TestObjectTraits<TestObjectType::LINE>{},  //
                      TestObjectTraits<TestObjectType::LINE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::LINE && runtime_type2 == TestObjectType::SPHERE) {
      return functor_(TestObjectTraits<TestObjectType::LINE>{},    //
                      TestObjectTraits<TestObjectType::SPHERE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::LINE_SEGMENT && runtime_type2 == TestObjectType::LINE_SEGMENT) {
      return functor_(TestObjectTraits<TestObjectType::LINE_SEGMENT>{},  //
                      TestObjectTraits<TestObjectType::LINE_SEGMENT>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::LINE_SEGMENT && runtime_type2 == TestObjectType::SPHERE) {
      return functor_(TestObjectTraits<TestObjectType::LINE_SEGMENT>{},  //
                      TestObjectTraits<TestObjectType::SPHERE>{},        //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::SPHERE && runtime_type2 == TestObjectType::SPHERE) {
      return functor_(TestObjectTraits<TestObjectType::SPHERE>{},  //
                      TestObjectTraits<TestObjectType::SPHERE>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::ELLIPSOID && runtime_type2 == TestObjectType::ELLIPSOID) {
      return functor_(TestObjectTraits<TestObjectType::ELLIPSOID>{},  //
                      TestObjectTraits<TestObjectType::ELLIPSOID>{},  //
                      std::forward<Args>(args)...);
    } else if (runtime_type1 == TestObjectType::CIRCLE_3D && runtime_type2 == TestObjectType::CIRCLE_3D) {
      return functor_(TestObjectTraits<TestObjectType::CIRCLE_3D>{},  //
                      TestObjectTraits<TestObjectType::CIRCLE_3D>{},  //
                      std::forward<Args>(args)...);
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unsupported combination of test object types.");
    }

    return return_type{};  // Fallback return value, should never be reached
  }

  Functor functor_;
};  // apply_pairwise_functor
//@}

struct compute_distance_impl {
  using return_type = double;
  template <typename ShapeTraits1, typename ShapeTraits2, typename RNG>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits1, ShapeTraits2, const AABB<double>& box, RNG& rng) const {
    auto obj1 = ShapeTraits1::generate(box, rng);
    auto obj2 = ShapeTraits2::generate(box, rng);
    return distance(obj1, obj2);
  }
};

struct compute_distance_min_image_impl {
  using return_type = double;
  template <typename ShapeTraits1, typename ShapeTraits2, typename RNG>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits1, ShapeTraits2, const AABB<double>& box, RNG& rng) const {
    auto obj1 = ShapeTraits1::generate(box, rng);
    auto obj2 = ShapeTraits2::generate(box, rng);

    // Assume that obj1 and obj2 are both within the bounding box
    auto box_size = mundy::math::Vector3<double>{box.x_max() - box.x_min(),  //
                                                 box.y_max() - box.y_min(),  //
                                                 box.z_max() - box.z_min()};

    mundy::math::Vector<double, 27> min_image_distance;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          // Shift obj2 by the box dimensions in each direction
          mundy::math::Vector3<double> disp((i - 1) * box_size[0], (j - 1) * box_size[1], (k - 1) * box_size[2]);
          auto shifted_obj2 = translate(obj2, disp);
          min_image_distance(i * 9 + j * 3 + k) = distance(obj1, shifted_obj2);
        }
      }
    }

    return min(min_image_distance);
  }
};

struct compute_distance_pbc_impl {
  using return_type = double;
  template <typename ShapeTraits1, typename ShapeTraits2, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits1, ShapeTraits2, const AABB<double>& box, RNG& rng,
                                                const Metric& metric) const {
    auto obj1 = ShapeTraits1::generate(box, rng);
    auto obj2 = ShapeTraits2::generate(box, rng);
    return distance_pbc(obj1, obj2, metric);
  }
};

struct compute_distance_pbc_shifted_impl {
  using return_type = double;
  template <typename ShapeTraits1, typename ShapeTraits2, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits1, ShapeTraits2, const AABB<double>& box, RNG& rng,
                                                const Metric& metric) const {
    auto obj1 = ShapeTraits1::generate(box, rng);
    auto obj2 = ShapeTraits2::generate(box, rng);

    // Shift obj2 by the box dimensions in each direction
    mundy::math::Vector3<double> box_disp{box.x_max() - box.x_min(),  //
                                          box.y_max() - box.y_min(),  //
                                          box.z_max() - box.z_min()};
    obj2 = translate(obj2, box_disp);
    return distance_pbc(obj1, obj2, metric);
  }
};

KOKKOS_INLINE_FUNCTION double compute_distance(TestObjectType type1, TestObjectType type2, const AABB<double>& box,
                                               size_t seed, size_t counter) {
  openrand::Philox rng(seed, counter);

  using Functor = compute_distance_impl;
  apply_pairwise_functor<Functor> apply;
  return apply(type1, type2, box, rng);
}

KOKKOS_INLINE_FUNCTION double compute_distance_min_image(TestObjectType type1, TestObjectType type2,
                                                         const AABB<double>& box, size_t seed, size_t counter) {
  openrand::Philox rng(seed, counter);

  using Functor = compute_distance_min_image_impl;
  apply_pairwise_functor<Functor> apply;
  return apply(type1, type2, box, rng);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION double compute_distance_pbc(TestObjectType type1, TestObjectType type2, const AABB<double>& box,
                                                   size_t seed, size_t counter, const Metric& metric) {
  openrand::Philox rng(seed, counter);

  using Functor = compute_distance_pbc_impl;
  apply_pairwise_functor<Functor> apply;
  return apply(type1, type2, box, rng, metric);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION double compute_distance_pbc_shifted(TestObjectType type1, TestObjectType type2,
                                                           const AABB<double>& box, size_t seed, size_t counter,
                                                           const Metric& metric) {
  openrand::Philox rng(seed, counter);

  using Functor = compute_distance_pbc_shifted_impl;
  apply_pairwise_functor<Functor> apply;
  return apply(type1, type2, box, rng, metric);
}

TEST(UnitTestPeriodicDistance, PeriodicFailureCase) {
  // Two line segments
  //   a: (0.0126464, 0.547307, 0.833031) -> (0.0576377, 0.682308, 0.364761)
  //   b: (0.708635, 0.396194, 0.328567) -> (0.264517, 0.277178, 0.142539)
  // In free-space the min distance solution (from vtk's python interface) is 0.5062726330273937
  //
  // from vtkmodules.vtkCommonDataModel import vtkLine
  // from vtkmodules.vtkCommonCore import reference
  //
  // l0 = [0.0126464, 0.547307, 0.833031]
  // l1 = [0.0576377, 0.682308, 0.364761]
  // m0 = [0.708635 , 0.396194, 0.328567]
  // m1 = [0.264517 , 0.277178, 0.142539]
  //
  // closest1 = [0.0, 0.0, 0.0]
  // closest2 = [0.0, 0.0, 0.0]
  // t1 = reference(0.0)
  // t2 = reference(0.0)
  //
  // d2 = vtkLine.DistanceBetweenLineSegments(l0, l1, m0, m1, closest1, closest2, t1, t2)
  // dist = d2 ** 0.5
  // print("min distance:", dist)
  double expected_distance = 0.5062726330273937;
  double expected_min_image_distance = 0.44625207297733677;
  LineSegment<double> a(Point<double>(0.0126464, 0.547307, 0.833031), Point<double>(0.0576377, 0.682308, 0.364761));
  LineSegment<double> b(Point<double>(0.708635, 0.396194, 0.328567), Point<double>(0.264517, 0.277178, 0.142539));

  // Free space distance
  Point<double> free_space_closest1;
  Point<double> free_space_closest2;
  double free_space_arch_length1 = 0.0;
  double free_space_arch_length2 = 0.0;
  mundy::math::Vector3<double> free_space_sep;
  double free_space_distance = distance(a, b, free_space_closest1, free_space_closest2, free_space_arch_length1,
                                        free_space_arch_length2, free_space_sep);

  // Periodic distance (with a free space metric)
  Point<double> periodic_closest1;
  Point<double> periodic_closest2;
  double periodic_arch_length1 = 0.0;
  double periodic_arch_length2 = 0.0;
  mundy::math::Vector3<double> periodic_sep;
  double free_space_distance2 = distance_pbc(a, b, FreeSpaceMetric{}, 
                                             periodic_closest1, periodic_closest2, periodic_arch_length1,
                                             periodic_arch_length2, periodic_sep);

  EXPECT_NEAR(free_space_distance, expected_distance, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(free_space_distance2, expected_distance, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(norm(free_space_closest1 - free_space_closest2), free_space_distance, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(norm(free_space_sep), free_space_distance, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(free_space_arch_length1, periodic_arch_length1, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(free_space_arch_length2, periodic_arch_length2, mundy::math::get_relaxed_zero_tolerance<double>());

  // Periodic distance (direct min image calculation)
  mundy::math::Vector3<double> box_size{1.0, 1.0, 1.0};
  double min_image_distance = std::numeric_limits<double>::max();
  Point<double> min_image_closest1;
  Point<double> min_image_closest2;
  double min_image_arch_length1 = 0.0;
  double min_image_arch_length2 = 0.0;
  mundy::math::Vector3<double> min_image_sep;
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      for (int k = -1; k <= 1; ++k) {
        mundy::math::Vector3<double> disp(static_cast<double>(i) * box_size[0],  //
                                          static_cast<double>(j) * box_size[1],  //
                                          static_cast<double>(k) * box_size[2]);
        LineSegment<double> b_shifted = translate(b, disp);
        
        Point<double> temp_closest1;
        Point<double> temp_closest2;
        double temp_arch_length1 = 0.0;
        double temp_arch_length2 = 0.0;
        mundy::math::Vector3<double> temp_sep;
        double d = distance(a, b_shifted, 
                            temp_closest1, temp_closest2, 
                            temp_arch_length1, temp_arch_length2, 
                            temp_sep);
        if (d < min_image_distance) {
          min_image_distance = d;
          min_image_closest1 = temp_closest1;
          min_image_closest2 = temp_closest2;
          min_image_arch_length1 = temp_arch_length1;
          min_image_arch_length2 = temp_arch_length2; 
          min_image_sep = temp_sep;         
        }
      }
    }
  }

  EXPECT_NEAR(min_image_distance, expected_min_image_distance, mundy::math::get_relaxed_zero_tolerance<double>())
      << "Min image distance did not match expected value.";
  
  // Periodic distance (with a periodic space metric)
  auto periodic_metric = periodic_scaled_metric_from_unit_cell(box_size);

  Point<double> periodic2_closest1;
  Point<double> periodic2_closest2;
  double periodic2_arch_length1 = 0.0;
  double periodic2_arch_length2 = 0.0;
  mundy::math::Vector3<double> periodic2_sep;
  double periodic_distance = distance_pbc(a, b, periodic_metric, 
                                          periodic2_closest1, periodic2_closest2, 
                                          periodic2_arch_length1, periodic2_arch_length2, 
                                          periodic2_sep);
  
  EXPECT_NEAR(periodic_distance, expected_min_image_distance, mundy::math::get_relaxed_zero_tolerance<double>())
      << "Periodic distance did not match expected value.";
  EXPECT_NEAR(distance_pbc(periodic2_closest1, periodic2_closest2, periodic_metric), periodic_distance,
              mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(norm(periodic2_sep), periodic_distance, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(periodic2_arch_length1, min_image_arch_length1, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(periodic2_arch_length2, min_image_arch_length2, mundy::math::get_relaxed_zero_tolerance<double>());
  EXPECT_NEAR(norm(periodic2_sep-min_image_sep), 0.0, mundy::math::get_relaxed_zero_tolerance<double>());
}

TEST(UnitTestPeriodicDistance, TestAperiodic) {
  // Validate that we can actually compute the distance for each pair of test objects
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 100;  // Number of trials for each pair

  using ShapePair = Kokkos::pair<TestObjectType, TestObjectType>;
  std::vector<ShapePair> test_pairs = {{TestObjectType::POINT, TestObjectType::POINT},
                                       {TestObjectType::POINT, TestObjectType::LINE},
                                       {TestObjectType::POINT, TestObjectType::LINE_SEGMENT},
                                       {TestObjectType::POINT, TestObjectType::SPHERE},
                                       {TestObjectType::LINE, TestObjectType::LINE},
                                       {TestObjectType::LINE, TestObjectType::SPHERE},
                                       {TestObjectType::LINE_SEGMENT, TestObjectType::LINE_SEGMENT},
                                       {TestObjectType::LINE_SEGMENT, TestObjectType::SPHERE},
                                       {TestObjectType::SPHERE, TestObjectType::SPHERE},
                                       {TestObjectType::ELLIPSOID, TestObjectType::ELLIPSOID},
                                       {TestObjectType::CIRCLE_3D, TestObjectType::CIRCLE_3D}};

  AABB<double> box{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};  // Unit cube bounding box
  for (const auto& pair : test_pairs) {
    for (size_t t = 0; t < num_trials; ++t) {
      TestObjectType type1 = pair.first;
      TestObjectType type2 = pair.second;

      // Must use the same seed and counter for each
      double free_space_distance = compute_distance(type1, type2, box, seed, counter);
      double periodic_distance = compute_distance_pbc(type1, type2, box, seed, counter, FreeSpaceMetric{});
      EXPECT_NEAR(free_space_distance, periodic_distance, mundy::math::get_relaxed_zero_tolerance<double>())
          << " for types " << type1 << " and " << type2;

      ++counter;
    }
  }
}

TEST(UnitTestPeriodicDistance, TestPeriodic) {
  // Validate that we can actually compute the distance for each pair of test objects
  size_t seed = 1234;
  size_t counter = 0;
  size_t num_trials = 100;  // Number of trials for each pair

  using ShapePair = Kokkos::pair<TestObjectType, TestObjectType>;
  std::vector<ShapePair> test_pairs = {{TestObjectType::POINT, TestObjectType::POINT},
                                       {TestObjectType::POINT, TestObjectType::LINE},
                                       {TestObjectType::POINT, TestObjectType::LINE_SEGMENT},
                                       {TestObjectType::POINT, TestObjectType::SPHERE},
                                       {TestObjectType::LINE, TestObjectType::LINE},
                                       {TestObjectType::LINE, TestObjectType::SPHERE},
                                       {TestObjectType::LINE_SEGMENT, TestObjectType::LINE_SEGMENT},
                                       {TestObjectType::LINE_SEGMENT, TestObjectType::SPHERE},
                                       {TestObjectType::SPHERE, TestObjectType::SPHERE},
                                       {TestObjectType::ELLIPSOID, TestObjectType::ELLIPSOID},
                                       {TestObjectType::CIRCLE_3D, TestObjectType::CIRCLE_3D}};

  mundy::math::Vector3<double> cell_size{1.0, 1.0, 1.0};
  AABB<double> box{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};  // Unit cube bounding box
  auto periodic_metric = periodic_scaled_metric_from_unit_cell(cell_size);

  for (const auto& pair : test_pairs) {
    for (size_t t = 0; t < num_trials; ++t) {
      TestObjectType type1 = pair.first;
      TestObjectType type2 = pair.second;

      // Must use the same seed and counter for each
      double min_free_space_distance = compute_distance_min_image(type1, type2, box, seed, counter);
      double periodic_distance = compute_distance_pbc(type1, type2, box, seed, counter, periodic_metric);
      double periodic_distance_shifted =
          compute_distance_pbc_shifted(type1, type2, box, seed, counter, periodic_metric);

      ASSERT_NEAR(min_free_space_distance, periodic_distance, mundy::math::get_relaxed_zero_tolerance<double>())
          << " for types " << type1 << " and " << type2;
      ASSERT_NEAR(min_free_space_distance, periodic_distance_shifted, mundy::math::get_relaxed_zero_tolerance<double>())
          << " for types " << type1 << " and " << type2;

      ++counter;
    }
  }
}

}  // namespace

}  // namespace geom

}  // namespace mundy
