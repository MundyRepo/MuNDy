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

#ifndef MUNDY_GEOM_PERIODICITY_HPP_
#define MUNDY_GEOM_PERIODICITY_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives.hpp>    // for mundy::geom::Point, mundy::geom::LineSegment, ...
#include <mundy_geom/transform.hpp>     // for mundy::geom::translate
#include <mundy_math/Matrix3.hpp>       // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>    // for mundy::math::Quaternion
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace geom {

namespace impl {

template <typename Integer, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar safe_unit_mod1(Scalar s) {
  // Map to [0,1); guard s ≈ 1 to avoid returning 0 due to FP noise.
  const Scalar tol = math::get_zero_tolerance<Scalar>();
  const Scalar k = static_cast<Scalar>(static_cast<Integer>(Kokkos::floor(s)));
  Scalar t = s - k;
  if (Kokkos::fabs(t - Scalar(1)) < tol) {
    t = Scalar(0);  // // t in [0,1) ideally: Guard against numerical errors that may cause t to be exactly 1.0
  }
  return t;
}

template <typename Integer, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr math::Vector3<Scalar> frac_minimum_image(const math::Vector3<Scalar>& fractional_vec) {
  return apply([](Scalar x) { return x - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(x))); },
               fractional_vec);
}

template <typename Integer, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr math::Vector3<Scalar> frac_wrap_to_unit_cell(
    const math::Vector3<Scalar>& fractional_vec) {
  return apply([](Scalar x) { return safe_unit_mod1<Integer>(x); }, fractional_vec);
}

}  // namespace impl

class EuclideanMetric {
 public:
  /// \brief Map a point into fractional coordinates
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION constexpr Point<Scalar> to_fractional(const Point<Scalar>& point) const {
    return point;
  }

  /// \brief Map a point from fractional coordinates to real space
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION constexpr Point<Scalar> from_fractional(const Point<Scalar>& point_frac) const {
    return point_frac;
  }

  /// \brief Distance vector between two points in free space (from point1 to point2)
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION constexpr Point<Scalar> sep(const Point<Scalar>& point1, const Point<Scalar>& point2) const {
    return point2 - point1;
  }

  /// \brief Wrap a point into the free space
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION constexpr Point<Scalar> wrap(const Point<Scalar>& point) const {
    return point;
  }

  /// \brief Direct lattice vectors (return as the columns of a matrix)
  KOKKOS_INLINE_FUNCTION constexpr math::Matrix3<double> direct_lattice_vectors() const {
    return math::Matrix3<double>::identity();
  }

  /// \brief Shift a point by a given number of lattice images in each direction (free space does nothing)
  template <typename Scalar, typename Integer>
  KOKKOS_INLINE_FUNCTION constexpr Point<Scalar> shift_image(const Point<Scalar>& point,
                                                             [[maybe_unused]] const math::Vector3<Integer>& num_images) const {
    return point;
  }
};  // EuclideanMetric

template <typename Scalar>
class PeriodicMetric {
 public:
  /// \brief Type aliases
  using scalar_t = Scalar;  ///< The scalar type
  using OurVector3 = math::Vector3<Scalar>;
  using OurMatrix3 = math::Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  constexpr PeriodicMetric() = default;

  /// \brief Constructor with unit cell matrix
  explicit constexpr PeriodicMetric(const OurMatrix3& h) : h_(h), h_inv_(math::inverse(h)) {
  }

  /// \brief Set the unit cell matrix
  void constexpr set_unit_cell_matrix(const OurMatrix3& h) {
    h_ = h;
    h_inv_ = math::inverse(h_);
  }

  /// \brief Map a point into fractional coordinates
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint to_fractional(const OurPoint& point) const {
    return h_inv_ * point;
  }

  /// \brief Map a point from fractional coordinates to real space
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint from_fractional(const OurPoint& point_frac) const {
    return h_ * point_frac;
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint sep(const OurPoint& point1, const OurPoint& point2) const {
    // Assumes linearity of to_fractional
    return from_fractional(impl::frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint wrap(const OurPoint& point) const {
    return from_fractional(impl::frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// \brief Direct lattice vectors (return as the columns of a matrix)
  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return h_;
  }

  /// \brief Shift a point by a given number of lattice images in each direction
  template<typename Integer>
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint shift_image(const OurPoint& point, const math::Vector3<Integer>& num_images) const {
    return  translate(point, h_ * num_images);
  }

 private:
  OurMatrix3 h_;      ///< Unit cell matrix
  OurMatrix3 h_inv_;  ///< Inverse of the unit cell matrix
};  // PeriodicMetric

template <typename Scalar>
class PeriodicScaledMetric {
 public:
  /// \brief Type aliases
  using OurVector3 = math::Vector3<Scalar>;
  using OurMatrix3 = math::Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  constexpr PeriodicScaledMetric() = default;

  /// \brief Constructor with unit cell matrix
  explicit constexpr PeriodicScaledMetric(const OurVector3& cell_size)
      : scale_(cell_size), scale_inv_(Scalar(1.0) / scale_[0], Scalar(1.0) / scale_[1], Scalar(1.0) / scale_[2]) {
  }

  /// \brief Set the cell size
  void set_cell_size(const OurVector3& cell_size) {
    scale_ = cell_size;
    scale_inv_.set(Scalar(1.0) / scale_[0], Scalar(1.0) / scale_[1], Scalar(1.0) / scale_[2]);
  }

  /// \brief Map a point into fractional coordinates
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint to_fractional(const OurPoint& point) const {
    return math::elementwise_multiply(scale_inv_, point);
  }

  /// \brief Map a point from fractional coordinates to real space
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint from_fractional(const OurPoint& point_frac) const {
    return math::elementwise_multiply(scale_, point_frac);
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint sep(const OurPoint& point1, const OurPoint& point2) const {
    return from_fractional(impl::frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint wrap(const OurPoint& point) const {
    return from_fractional(impl::frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// \brief Direct lattice vectors (return as the columns of a matrix)
  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return OurMatrix3::diagonal(scale_);
  }

  /// \brief Shift a point by a given number of lattice images in each direction
  template<typename Integer>
  KOKKOS_INLINE_FUNCTION
  constexpr OurPoint shift_image(const OurPoint& point, const math::Vector3<Integer>& num_images) const {
    return  translate(point, math::elementwise_multiply(scale_, num_images));
  }

 private:
  OurVector3 scale_;      ///< Unit cell scaling factors
  OurVector3 scale_inv_;  ///< Inverse of the scaling factors
};  // PeriodicScaledMetric

//! \name Non-member constructors
//@{

/// \brief Create a periodic space metric from a unit cell size
template <typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr PeriodicMetric<Scalar> periodic_metric_from_unit_cell(
    const math::Vector3<Scalar>& cell_size) {
  auto h = math::Matrix3<Scalar>::identity();
  h(0, 0) = cell_size[0];
  h(1, 1) = cell_size[1];
  h(2, 2) = cell_size[2];
  return PeriodicMetric<Scalar>{std::move(h)};
}

/// \brief Create a periodic scaled space metric from a unit cell size
template <typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr PeriodicScaledMetric<Scalar> periodic_scaled_metric_from_unit_cell(
    const math::Vector3<Scalar>& cell_size) {
  return PeriodicScaledMetric<Scalar>{cell_size};
}
//@}

//! \name Get the periodic reference point for an object
//@{
/*


The set of reference points:
                Point -> center
          LineSegment -> start
                 Line -> center
             Circle3D -> center
             VSegment -> start
                 AABB -> min_corner
               Sphere -> center
       Spherocylinder -> center
SpherocylinderSegment -> start
                 Ring -> center
            Ellipsoid -> center
*/

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const Point<Scalar>& point) {
  return point;
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const Line<Scalar>& line) {
  return line.center();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const LineSegment<Scalar>& line_segment) {
  return line_segment.start();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const Circle3D<Scalar>& circle) {
  return circle.center();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const VSegment<Scalar>& v_segment) {
  return v_segment.start();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const AABB<Scalar>& aabb) {
  return aabb.min_corner();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const Sphere<Scalar>& sphere) {
  return sphere.center();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const Spherocylinder<Scalar>& spherocylinder) {
  return spherocylinder.center();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const SpherocylinderSegment<Scalar>& spherocylinder_segment) {
  return spherocylinder_segment.start();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const Ring<Scalar>& ring) {
  return ring.center();
}

template <typename Scalar>
KOKKOS_INLINE_FUNCTION Point<Scalar> reference_point(const Ellipsoid<Scalar>& ellipsoid) {
  return ellipsoid.center();
}
//@}

//! \name Shift image functions to take an object and shift it by a lattice vector
//@{

/// \brief Shift an object by a lattice vector
template <typename Integer, typename Object, typename Metric>
KOKKOS_INLINE_FUNCTION Object shift_image(const Object& obj,                             //
                                          const math::Vector3<Integer>& lattice_vector,  //
                                          const Metric& metric) {
  return translate(obj, metric.shift_image(reference_point(obj), lattice_vector));
}

//@}

//! \name Rigid wrapping functions (based on a consistent reference point)
//@{

/*

The set of reference points:
                Point -> center
          LineSegment -> start
                 Line -> (not valid)
             Circle3D -> center
             VSegment -> start
                 AABB -> min_corner
               Sphere -> center
       Spherocylinder -> center
SpherocylinderSegment -> start
                 Ring -> center
            Ellipsoid -> center

Re-imagining the names for these functions:
wrap_points: wraps each point of the object independently using the metric's wrap function.
unwrap_points_to_ref: unwraps each point of the object into the same image as the reference
point. wrap_rigid: rigid translation so the internal reference point ends up wrapped into the primary cell; all other
points move with it.
*/

/// \brief Rigidly wrap a point into a given space (based on its center)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Point<Scalar> wrap_rigid(const Point<Scalar>& point, const Metric& metric) {
  return metric.wrap(point);
}

/// \brief Rigidly wrap a point into a given space (based on its center | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Point<Scalar>& point, const Metric& metric) {
  metric.wrap_inplace(point);
}

/// \brief Rigidly wrap a line into a given space (invalid operation)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Line<Scalar> wrap_rigid(const Line<Scalar>& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
  return line;
}

/// brief Rigidly wrap a line into a given space (invalid operation | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Line<Scalar>& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
}

/// \brief Rigidly wrap a line segment into a given space (based on its start point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION LineSegment<Scalar> wrap_rigid(const LineSegment<Scalar>& line_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(line_segment.start());
  auto wrapped_end = line_segment.end() + (wrapped_start - line_segment.start());
  return LineSegment<Scalar>(wrapped_start, wrapped_end);
}

/// \brief Rigidly wrap a line segment into a given space (based on its start point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(LineSegment<Scalar>& line_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(line_segment.start());
  auto disp = wrapped_start - line_segment.start();
  line_segment.set_start(wrapped_start);
  line_segment.set_end(line_segment.end() + disp);
}

/// \brief Rigidly wrap a circle3D into a given space (based on its center point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Circle3D<Scalar> wrap_rigid(const Circle3D<Scalar>& circle, const Metric& metric) {
  return Circle3D<Scalar>(metric.wrap(circle.center()), circle.orientation(), circle.radius());
}

/// \brief Rigidly wrap a circle3D into a given space (based on its center point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Circle3D<Scalar>& circle, const Metric& metric) {
  metric.wrap_inplace(circle.center());
}

/// \brief Rigidly wrap a v-segment into a given space (based on its start point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION VSegment<Scalar> wrap_rigid(const VSegment<Scalar>& v_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(v_segment.start());
  auto disp = wrapped_start - v_segment.start();
  auto wrapped_middle = v_segment.middle() + disp;
  auto wrapped_end = v_segment.end() + disp;
  return VSegment<Scalar>(wrapped_start, wrapped_middle, wrapped_end);
}

/// \brief Rigidly wrap a v-segment into a given space (based on its start point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(VSegment<Scalar>& v_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(v_segment.start());
  auto disp = wrapped_start - v_segment.start();
  v_segment.set_start(wrapped_start);
  v_segment.set_middle(v_segment.middle() + disp);
  v_segment.set_end(v_segment.end() + disp);
}

/// \brief Rigidly wrap an AABB into a given space (based on its min_corner)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION AABB<Scalar> wrap_rigid(const AABB<Scalar>& aabb, const Metric& metric) {
  auto wrapped_min = metric.wrap(aabb.min_corner());
  auto wrapped_max = aabb.max_corner() + (wrapped_min - aabb.min_corner());
  return AABB<Scalar>(wrapped_min, wrapped_max);
}

/// \brief Rigidly wrap an AABB into a given space (based on its min_corner | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(AABB<Scalar>& aabb, const Metric& metric) {
  auto wrapped_min = metric.wrap(aabb.min_corner());
  auto disp = wrapped_min - aabb.min_corner();
  aabb.set_min_corner(wrapped_min);
  aabb.set_max_corner(aabb.max_corner() + disp);
}

/// \brief Rigidly wrap a sphere into a given space (based on its center point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Sphere<Scalar> wrap_rigid(const Sphere<Scalar>& sphere, const Metric& metric) {
  return Sphere<Scalar>(metric.wrap(sphere.center()), sphere.radius());
}

/// \brief Rigidly wrap a sphere into a given space (based on its center point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Sphere<Scalar>& sphere, const Metric& metric) {
  metric.wrap_inplace(sphere.center());
}

/// \brief Rigidly wrap a spherocylinder into a given space (based on its center point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Spherocylinder<Scalar> wrap_rigid(const Spherocylinder<Scalar>& spherocylinder,
                                                         const Metric& metric) {
  return Spherocylinder<Scalar>(metric.wrap(spherocylinder.center()), spherocylinder.orientation(),
                                spherocylinder.radius(), spherocylinder.length());
}

/// \brief Rigidly wrap a spherocylinder into a given space (based on its center point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Spherocylinder<Scalar>& spherocylinder, const Metric& metric) {
  metric.wrap_inplace(spherocylinder.center());
}

/// \brief Rigidly wrap a spherocylinder segment into a given space (based on its start point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<Scalar> wrap_rigid(
    const SpherocylinderSegment<Scalar>& spherocylinder_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(spherocylinder_segment.start());
  auto wrapped_end = spherocylinder_segment.end() + (wrapped_start - spherocylinder_segment.start());
  return SpherocylinderSegment<Scalar>(wrapped_start, wrapped_end, spherocylinder_segment.orientation(),
                                       spherocylinder_segment.radius(), spherocylinder_segment.length());
}

/// \brief Rigidly wrap a spherocylinder segment into a given space (based on its start point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(SpherocylinderSegment<Scalar>& spherocylinder_segment,
                                               const Metric& metric) {
  auto wrapped_start = metric.wrap(spherocylinder_segment.start());
  auto disp = wrapped_start - spherocylinder_segment.start();
  spherocylinder_segment.set_start(wrapped_start);
  spherocylinder_segment.set_end(spherocylinder_segment.end() + disp);
}

/// \brief Rigidly wrap a ring into a given space (based on its center point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Ring<Scalar> wrap_rigid(const Ring<Scalar>& ring, const Metric& metric) {
  return Ring<Scalar>(metric.wrap(ring.center()), ring.orientation(), ring.major_radius(), ring.minor_radius());
}

/// \brief Rigidly wrap a ring into a given space (based on its center point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Ring<Scalar>& ring, const Metric& metric) {
  metric.wrap_inplace(ring.center());
}

/// \brief Rigidly wrap an ellipsoid into a given space (based on its center point)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Ellipsoid<Scalar> wrap_rigid(const Ellipsoid<Scalar>& ellipsoid, const Metric& metric) {
  return Ellipsoid<Scalar>(metric.wrap(ellipsoid.center()), ellipsoid.orientation(), ellipsoid.radii());
}

/// \brief Rigidly wrap an ellipsoid into a given space (based on its center point | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Ellipsoid<Scalar>& ellipsoid, const Metric& metric) {
  metric.wrap_inplace(ellipsoid.center());
}
//@}

//! \name Wrap all points of an object into a periodic space
//@{

/// \brief Wrap all points of a point into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Point<Scalar> wrap_points(const Point<Scalar>& point, const Metric& metric) {
  return metric.wrap(point);
}

/// \brief Wrap all points of a point into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Point<Scalar>& point, const Metric& metric) {
  metric.wrap_inplace(point);
}

/// \brief Wrap all points of a line into a periodic space (invalid operation)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Line<Scalar> wrap_points(const Line<Scalar>& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
  return line;
}

/// \brief Wrap all points of a line into a periodic space (invalid operation | inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Line<Scalar>& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
}

/// \brief Wrap all points of a line segment into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION LineSegment<Scalar> wrap_points(const LineSegment<Scalar>& line_segment, const Metric& metric) {
  return LineSegment<Scalar>(metric.wrap(line_segment.start()), metric.wrap(line_segment.end()));
}

/// \brief Wrap all points of a line segment into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(LineSegment<Scalar>& line_segment, const Metric& metric) {
  metric.wrap_inplace(line_segment.start());
  metric.wrap_inplace(line_segment.end());
}

/// \brief Wrap all points of a circle3D into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Circle3D<Scalar> wrap_points(const Circle3D<Scalar>& circle, const Metric& metric) {
  return Circle3D<Scalar>(metric.wrap(circle.center()), circle.orientation(), circle.radius());
}

/// \brief Wrap all points of a circle3D into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Circle3D<Scalar>& circle, const Metric& metric) {
  metric.wrap_inplace(circle.center());
}

/// \brief Wrap all points of a v-segment into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION VSegment<Scalar> wrap_points(const VSegment<Scalar>& v_segment, const Metric& metric) {
  return VSegment<Scalar>(metric.wrap(v_segment.start()), metric.wrap(v_segment.middle()),
                          metric.wrap(v_segment.end()));
}

/// \brief Wrap all points of a v-segment into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(VSegment<Scalar>& v_segment, const Metric& metric) {
  metric.wrap_inplace(v_segment.start());
  metric.wrap_inplace(v_segment.middle());
  metric.wrap_inplace(v_segment.end());
}

/// \brief Wrap all points of an AABB into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION AABB<Scalar> wrap_points(const AABB<Scalar>& aabb, const Metric& metric) {
  return AABB<Scalar>(metric.wrap(aabb.min_corner()), metric.wrap(aabb.max_corner()));
}

/// \brief Wrap all points of an AABB into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(AABB<Scalar>& aabb, const Metric& metric) {
  metric.wrap_inplace(aabb.min_corner());
  metric.wrap_inplace(aabb.max_corner());
}

/// \brief Wrap all points of a sphere into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Sphere<Scalar> wrap_points(const Sphere<Scalar>& sphere, const Metric& metric) {
  return Sphere<Scalar>(metric.wrap(sphere.center()), sphere.radius());
}

/// \brief Wrap all points of a sphere into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Sphere<Scalar>& sphere, const Metric& metric) {
  metric.wrap_inplace(sphere.center());
}

/// \brief Wrap all points of a spherocylinder into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Spherocylinder<Scalar> wrap_points(const Spherocylinder<Scalar>& spherocylinder,
                                                          const Metric& metric) {
  return Spherocylinder<Scalar>(metric.wrap(spherocylinder.center()), spherocylinder.orientation(),
                                spherocylinder.radius(), spherocylinder.length());
}

/// \brief Wrap all points of a spherocylinder into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Spherocylinder<Scalar>& spherocylinder, const Metric& metric) {
  metric.wrap_inplace(spherocylinder.center());
}

/// \brief Wrap all points of a spherocylinder segment into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<Scalar> wrap_points(
    const SpherocylinderSegment<Scalar>& spherocylinder_segment, const Metric& metric) {
  return SpherocylinderSegment<Scalar>(metric.wrap(spherocylinder_segment.start()),
                                       metric.wrap(spherocylinder_segment.end()), spherocylinder_segment.orientation(),
                                       spherocylinder_segment.radius(), spherocylinder_segment.length());
}

/// \brief Wrap all points of a spherocylinder segment into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(SpherocylinderSegment<Scalar>& spherocylinder_segment,
                                                const Metric& metric) {
  metric.wrap_inplace(spherocylinder_segment.start());
  metric.wrap_inplace(spherocylinder_segment.end());
}

/// \brief Wrap all points of a ring into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Ring<Scalar> wrap_points(const Ring<Scalar>& ring, const Metric& metric) {
  return Ring<Scalar>(metric.wrap(ring.center()), ring.orientation(), ring.major_radius(), ring.minor_radius());
}

/// \brief Wrap all points of a ring into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Ring<Scalar>& ring, const Metric& metric) {
  metric.wrap_inplace(ring.center());
}

/// \brief Wrap all points of an ellipsoid into a periodic space
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Ellipsoid<Scalar> wrap_points(const Ellipsoid<Scalar>& ellipsoid, const Metric& metric) {
  return Ellipsoid<Scalar>(metric.wrap(ellipsoid.center()), ellipsoid.orientation(), ellipsoid.radii());
}

/// \brief Wrap all points of an ellipsoid into a periodic space (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Ellipsoid<Scalar>& ellipsoid, const Metric& metric) {
  metric.wrap_inplace(ellipsoid.center());
}
//@}

//! \name Unwrap all points of an object into the same image as the reference point
//@{

/*
We must emphasize that unwrap_points_to_ref is only the inverse of wrap_points if the reference point already lies
within the primary cell. Otherwise, they are the same up to wrap_rigid applied to the original shape. That is:
  wrap_rigid(s) == unwrap_points_to_ref(wrap_points(s, metric)) == unwrap_points_to_ref(s, s.ref()).

Note. The following are valid even if the ref point is one of the given points.
*/

/// \brief Unwrap all points of a point into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Point<Scalar> unwrap_points_to_ref(const Point<Scalar>& point, const Metric& metric,
                                                          const Point<Scalar>& ref_point) {
  // Map reference point to fractional coordinates
  const auto s = metric.to_fractional(point);
  const auto sr = metric.to_fractional(ref_point);

  // Minimum-image convention
  const auto d = impl::frac_minimum_image<int64_t>(s - sr);

  // Map the fractional coordinates back to real space
  return metric.from_fractional(sr + d);
}

/// \brief Unwrap all points of a point into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Point<Scalar>& point, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  const auto s = metric.to_fractional(point);
  const auto sr = metric.to_fractional(ref_point);

  const auto d = impl::frac_minimum_image<int64_t>(s - sr);

  point = metric.from_fractional(sr + d);
}

/// \brief Unwrap all points of a line into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Line<Scalar> unwrap_points_to_ref(const Line<Scalar>& line, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: unwrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
  return line;
}

/// \brief Unwrap all points of a line into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Line<Scalar>& line, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: unwrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
}

/// \brief Unwrap all points of a line segment into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION LineSegment<Scalar> unwrap_points_to_ref(const LineSegment<Scalar>& line_segment,
                                                                const Metric& metric, const Point<Scalar>& ref_point) {
  const auto s_start = metric.to_fractional(line_segment.start());
  const auto s_end = metric.to_fractional(line_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = impl::frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = impl::frac_minimum_image<int64_t>(s_end - sr);

  return LineSegment<Scalar>(metric.from_fractional(sr + d_start), metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a line segment into the same image as the reference point
/// (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(LineSegment<Scalar>& line_segment, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  const auto s_start = metric.to_fractional(line_segment.start());
  const auto s_end = metric.to_fractional(line_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = impl::frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = impl::frac_minimum_image<int64_t>(s_end - sr);

  line_segment.set_start(metric.from_fractional(sr + d_start));
  line_segment.set_end(metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a circle3D into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Circle3D<Scalar> unwrap_points_to_ref(const Circle3D<Scalar>& circle, const Metric& metric,
                                                             const Point<Scalar>& ref_point) {
  auto new_center = unwrap_points_to_ref(circle.center(), metric, ref_point);
  return Circle3D<Scalar>(new_center, circle.orientation(), circle.radius());
}

/// \brief Unwrap all points of a circle3D into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Circle3D<Scalar>& circle, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  unwrap_points_to_ref_inplace(circle.center(), metric, ref_point);
}

/// \brief Unwrap all points of a v-segment into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION VSegment<Scalar> unwrap_points_to_ref(const VSegment<Scalar>& v_segment, const Metric& metric,
                                                             const Point<Scalar>& ref_point) {
  const auto s_start = metric.to_fractional(v_segment.start());
  const auto s_middle = metric.to_fractional(v_segment.middle());
  const auto s_end = metric.to_fractional(v_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = impl::frac_minimum_image<int64_t>(s_start - sr);
  const auto d_middle = impl::frac_minimum_image<int64_t>(s_middle - sr);
  const auto d_end = impl::frac_minimum_image<int64_t>(s_end - sr);

  return VSegment<Scalar>(metric.from_fractional(sr + d_start), metric.from_fractional(sr + d_middle),
                          metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a v-segment into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(VSegment<Scalar>& v_segment, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  const auto s_start = metric.to_fractional(v_segment.start());
  const auto s_middle = metric.to_fractional(v_segment.middle());
  const auto s_end = metric.to_fractional(v_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = impl::frac_minimum_image<int64_t>(s_start - sr);
  const auto d_middle = impl::frac_minimum_image<int64_t>(s_middle - sr);
  const auto d_end = impl::frac_minimum_image<int64_t>(s_end - sr);

  v_segment.set_start(metric.from_fractional(sr + d_start));
  v_segment.set_middle(metric.from_fractional(sr + d_middle));
  v_segment.set_end(metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of an AABB into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION AABB<Scalar> unwrap_points_to_ref(const AABB<Scalar>& aabb, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  const auto s_min = metric.to_fractional(aabb.min_corner());
  const auto s_max = metric.to_fractional(aabb.max_corner());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_min = impl::frac_minimum_image<int64_t>(s_min - sr);
  const auto d_max = impl::frac_minimum_image<int64_t>(s_max - sr);

  return AABB<Scalar>(metric.from_fractional(sr + d_min), metric.from_fractional(sr + d_max));
}

/// \brief Unwrap all points of an AABB into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(AABB<Scalar>& aabb, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  const auto s_min = metric.to_fractional(aabb.min_corner());
  const auto s_max = metric.to_fractional(aabb.max_corner());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_min = impl::frac_minimum_image<int64_t>(s_min - sr);
  const auto d_max = impl::frac_minimum_image<int64_t>(s_max - sr);

  aabb.set_min_corner(metric.from_fractional(sr + d_min));
  aabb.set_max_corner(metric.from_fractional(sr + d_max));
}

/// \brief Unwrap all points of a sphere into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Sphere<Scalar> unwrap_points_to_ref(const Sphere<Scalar>& sphere, const Metric& metric,
                                                           const Point<Scalar>& ref_point) {
  auto new_center = unwrap_points_to_ref(sphere.center(), metric, ref_point);
  return Sphere<Scalar>(new_center, sphere.radius());
}

/// \brief Unwrap all points of a sphere into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Sphere<Scalar>& sphere, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  unwrap_points_to_ref_inplace(sphere.center(), metric, ref_point);
}

/// \brief Unwrap all points of a spherocylinder into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Spherocylinder<Scalar> unwrap_points_to_ref(const Spherocylinder<Scalar>& spherocylinder,
                                                                   const Metric& metric,
                                                                   const Point<Scalar>& ref_point) {
  auto new_center = unwrap_points_to_ref(spherocylinder.center(), metric, ref_point);
  return Spherocylinder<Scalar>(new_center, spherocylinder.orientation(), spherocylinder.radius(),
                                spherocylinder.length());
}

/// \brief Unwrap all points of a spherocylinder into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Spherocylinder<Scalar>& spherocylinder, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  unwrap_points_to_ref_inplace(spherocylinder.center(), metric, ref_point);
}

/// \brief Unwrap all points of a spherocylinder segment into the same image as the reference
/// point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<Scalar> unwrap_points_to_ref(
    const SpherocylinderSegment<Scalar>& spherocylinder_segment, const Metric& metric, const Point<Scalar>& ref_point) {
  const auto s_start = metric.to_fractional(spherocylinder_segment.start());
  const auto s_end = metric.to_fractional(spherocylinder_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = impl::frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = impl::frac_minimum_image<int64_t>(s_end - sr);

  return SpherocylinderSegment<Scalar>(metric.from_fractional(sr + d_start), metric.from_fractional(sr + d_end),
                                       spherocylinder_segment.orientation(), spherocylinder_segment.radius(),
                                       spherocylinder_segment.length());
}

/// \brief Unwrap all points of a spherocylinder segment into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(SpherocylinderSegment<Scalar>& spherocylinder_segment,
                                                         const Metric& metric, const Point<Scalar>& ref_point) {
  const auto s_start = metric.to_fractional(spherocylinder_segment.start());
  const auto s_end = metric.to_fractional(spherocylinder_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = impl::frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = impl::frac_minimum_image<int64_t>(s_end - sr);

  spherocylinder_segment.set_start(metric.from_fractional(sr + d_start));
  spherocylinder_segment.set_end(metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a ring into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Ring<Scalar> unwrap_points_to_ref(const Ring<Scalar>& ring, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  auto new_center = unwrap_points_to_ref(ring.center(), metric, ref_point);
  return Ring<Scalar>(new_center, ring.orientation(), ring.major_radius(), ring.minor_radius());
}

/// \brief Unwrap all points of a ring into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Ring<Scalar>& ring, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  unwrap_points_to_ref_inplace(ring.center(), metric, ref_point);
}

/// \brief Unwrap all points of an ellipsoid into the same image as the reference point
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION Ellipsoid<Scalar> unwrap_points_to_ref(const Ellipsoid<Scalar>& ellipsoid, const Metric& metric,
                                                              const Point<Scalar>& ref_point) {
  auto new_center = unwrap_points_to_ref(ellipsoid.center(), metric, ref_point);
  return Ellipsoid<Scalar>(new_center, ellipsoid.orientation(), ellipsoid.radii());
}

/// \brief Unwrap all points of an ellipsoid into the same image as the reference point (inplace)
template <typename Scalar, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Ellipsoid<Scalar>& ellipsoid, const Metric& metric,
                                                         const Point<Scalar>& ref_point) {
  unwrap_points_to_ref_inplace(ellipsoid.center(), metric, ref_point);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PERIODICITY_HPP_
