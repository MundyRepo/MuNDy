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

/// \file periodicity.hpp
/// \defgroup MundyGeomPeriodicity mundy::periodicity
/// \brief Periodic metric classes and geometric wrapping helpers.
///
/// ## Design overview
///
/// Three concrete metric classes cover the full range of boundary conditions:
///
///   FreeSpaceMetric<Scalar>                  -- non-periodic open boundaries
///   OrthorhombicMetric<PeriodicAxes, Scalar> -- axis-aligned periodic cell
///   TriclinicMetric<PeriodicAxes, Scalar>    -- general tilted periodic cell
///
/// Both OrthorhombicMetric and TriclinicMetric are parameterised by a compile-time
/// bitmask of periodic axes:
///
///   AXIS_X = 0b001,  AXIS_Y = 0b010,  AXIS_Z = 0b100
///
/// Combine with | to select subsets (e.g. AXIS_XY = AXIS_X | AXIS_Y). "Cell shape"
/// (orthorhombic vs triclinic) and "which axes are periodic" are orthogonal concepts;
/// the bitmask captures the latter for both geometries. For TriclinicMetric, AXIS_X/Y/Z
/// refer to the first, second, and third lattice vector directions (fractional axes),
/// not Cartesian x/y/z. All branching on PeriodicAxes is resolved at compile time via
/// if constexpr, so unused axes impose zero runtime cost.
///
/// All concrete metrics are zero-overhead and Kokkos device-compatible.
///
/// For configuration-file-driven code where the metric type is unknown at compile
/// time, use the runtime `metric` class (stk::topology-style). Its visit() method
/// dispatches via a plain switch to a concrete metric, so the called code sees a
/// fully concrete type and the compiler inlines it.
///
/// Geometric wrapping functions (wrap_rigid, wrap_points, unwrap_points_to_ref)
/// are single function templates driven by the for_each_point_mutable free-function
/// protocol. Each primitive type must provide an explicit overload of
/// for_each_point_mutable alongside its definition. There is no default
/// implementation: a missing overload is a compile error, not silent wrong behavior.

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_geom/primitives.hpp>  // for mundy::Point, mundy::LineSegment, ...
#include <mundy_geom/transform.hpp>   // for mundy::translate, mundy::translate_inplace
#include <mundy_math/Matrix3.hpp>     // for mundy::Matrix3
#include <mundy_math/Tolerance.hpp>   // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>
#include <mundy_math/cmath.hpp>

namespace mundy {

namespace impl {

/// Map s into [0, 1). Uses floor() rather than integer truncation to avoid UB
/// when |s| exceeds the representable range of any integer type.
/// The guard is required: for a tiny negative s (e.g. -1e-300), floor(s) = -1
/// and (s + 1) rounds to exactly 1.0 in IEEE 754, so without the clamp the
/// return value would be 1.0 instead of 0.0.
template <typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar safe_unit_mod1(Scalar s) {
  const Scalar tol = get_zero_tolerance<Scalar>();
  Scalar t = s - floor(s);
  if (fabs(t - Scalar(1)) < tol) t = Scalar(0);
  return t;
}

}  // namespace impl

//! \name Axis bitmask constants
//@{

inline constexpr unsigned AXIS_X = 0b001u;  ///< Bitmask selecting the X axis (or first lattice vector)
inline constexpr unsigned AXIS_Y = 0b010u;  ///< Bitmask selecting the Y axis (or second lattice vector)
inline constexpr unsigned AXIS_Z = 0b100u;  ///< Bitmask selecting the Z axis (or third lattice vector)
inline constexpr unsigned AXIS_XY = AXIS_X | AXIS_Y;
inline constexpr unsigned AXIS_XZ = AXIS_X | AXIS_Z;
inline constexpr unsigned AXIS_YZ = AXIS_Y | AXIS_Z;
inline constexpr unsigned AXIS_XYZ = AXIS_X | AXIS_Y | AXIS_Z;
//@}

/// \brief Non-periodic (open-boundary) metric. All operations are identities.
template <typename Scalar>
class FreeSpaceMetric {
 public:
  //! \name Type aliases
  //@{

  using value_type = Scalar;
  using OurVector3 = Vector3<Scalar>;
  using OurMatrix3 = Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;
  //@}

  //! \name Periodicity queries
  //@{

  template <unsigned dim>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return false;
  }
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned /*dim*/) const {
    return false;
  }
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 0;
  }
  //@}

  //! \name Metric operations
  //@{

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& p) const {
    return p;
  }

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& p) const {
    return p;
  }

  template <ValidVector3Type Vector3T>
  MUNDY_REQUIRES(std::is_same_v<typename Vector3T::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 frac_minimum_image(const Vector3T& fv) const {
    return OurVector3{fv[0], fv[1], fv[2]};
  }

  template <ValidVector3Type Vector3T>
  MUNDY_REQUIRES(std::is_same_v<typename Vector3T::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 frac_wrap_to_unit_cell(const Vector3T& fv) const {
    return OurVector3{fv[0], fv[1], fv[2]};
  }

  template <ValidPointType PointT1, ValidPointType PointT2>
  MUNDY_REQUIRES(
      std::is_same_v<typename PointT1::value_type, Scalar>&& std::is_same_v<typename PointT2::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 sep(const PointT1& p1, const PointT2& p2) const {
    return p2 - p1;
  }

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& p) const {
    return p;
  }

  /// \brief Returns the identity matrix by convention; FreeSpaceMetric has no physical lattice.
  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return OurMatrix3::identity();
  }

  template <ValidPointType PointT, typename Integer>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& p, const Vector3<Integer>& /*n*/) const {
    return p;  // No periodic images in free space; shift is a no-op for any n.
  }
  //@}
};

/// \brief Axis-aligned periodic metric parameterised by a compile-time axis bitmask.
///
/// \tparam PeriodicAxes Bitmask of periodic axes. Combine AXIS_X (0b001),
///   AXIS_Y (0b010), AXIS_Z (0b100) with | for multi-axis periodicity.
/// \tparam Scalar Floating-point scalar type.
///
/// Cell widths for non-periodic axes are set to 1 internally and never participate
/// in wrapping or distance calculations. All branching on PeriodicAxes is resolved
/// at compile time via if constexpr, so unused axes impose zero runtime cost.
template <unsigned PeriodicAxes, typename Scalar>
class OrthorhombicMetric {
  static_assert(PeriodicAxes > 0 && PeriodicAxes <= AXIS_XYZ,
                "PeriodicAxes must be a non-zero combination of AXIS_X, AXIS_Y, AXIS_Z");

 public:
  //! \name Type aliases
  //@{

  using value_type = Scalar;
  using OurVector3 = Vector3<Scalar>;
  using OurMatrix3 = Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;
  //@}

  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION
  constexpr OrthorhombicMetric() = default;

  /// \brief Construct from cell widths.
  ///
  /// \p cell_widths entries for non-periodic axes are ignored (those axes are
  /// treated as having width 1). Only the widths of periodic axes must be positive.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr OrthorhombicMetric(const OurVector3& cell_widths) {
    MUNDY_THROW_ASSERT((!(PeriodicAxes & AXIS_X) || cell_widths[0] > Scalar(0)) &&
                           (!(PeriodicAxes & AXIS_Y) || cell_widths[1] > Scalar(0)) &&
                           (!(PeriodicAxes & AXIS_Z) || cell_widths[2] > Scalar(0)),
                       std::invalid_argument, "Periodic cell widths must be positive");
    scale_[0] = (PeriodicAxes & AXIS_X) ? cell_widths[0] : Scalar(1);
    scale_[1] = (PeriodicAxes & AXIS_Y) ? cell_widths[1] : Scalar(1);
    scale_[2] = (PeriodicAxes & AXIS_Z) ? cell_widths[2] : Scalar(1);
    scale_inv_[0] = Scalar(1) / scale_[0];
    scale_inv_[1] = Scalar(1) / scale_[1];
    scale_inv_[2] = Scalar(1) / scale_[2];
  }

  KOKKOS_INLINE_FUNCTION
  void set_cell_widths(const OurVector3& cell_widths) {
    *this = OrthorhombicMetric(cell_widths);
  }
  //@}

  //! \name Periodicity queries
  //@{

  template <unsigned dim>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return (PeriodicAxes >> dim) & 1u;
  }
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned dim) const {
    return (PeriodicAxes >> dim) & 1u;
  }
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return ((PeriodicAxes & 1u) + ((PeriodicAxes >> 1) & 1u) + ((PeriodicAxes >> 2) & 1u));
  }
  //@}

  //! \name Metric operations
  //@{

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& p) const {
    return elementwise_mul(scale_inv_, p);
  }

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& frac) const {
    return elementwise_mul(scale_, frac);
  }

  /// \brief Minimum-image displacement in fractional coordinates.
  ///
  /// Maps each periodic component to [-0.5, 0.5) by subtracting round().
  /// Non-periodic components are passed through unchanged.
  template <ValidVector3Type Vector3T>
  MUNDY_REQUIRES(std::is_same_v<typename Vector3T::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 frac_minimum_image(const Vector3T& fv) const {
    OurVector3 r{fv[0], fv[1], fv[2]};
    if constexpr (PeriodicAxes & AXIS_X) r[0] -= round(r[0]);
    if constexpr (PeriodicAxes & AXIS_Y) r[1] -= round(r[1]);
    if constexpr (PeriodicAxes & AXIS_Z) r[2] -= round(r[2]);
    return r;
  }

  template <ValidVector3Type Vector3T>
  MUNDY_REQUIRES(std::is_same_v<typename Vector3T::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 frac_wrap_to_unit_cell(const Vector3T& fv) const {
    OurVector3 r{fv[0], fv[1], fv[2]};
    if constexpr (PeriodicAxes & AXIS_X) r[0] = impl::safe_unit_mod1(r[0]);
    if constexpr (PeriodicAxes & AXIS_Y) r[1] = impl::safe_unit_mod1(r[1]);
    if constexpr (PeriodicAxes & AXIS_Z) r[2] = impl::safe_unit_mod1(r[2]);
    return r;
  }

  /// \brief Minimum-image displacement vector from p1 to p2.
  template <ValidPointType PointT1, ValidPointType PointT2>
  MUNDY_REQUIRES(
      std::is_same_v<typename PointT1::value_type, Scalar>&& std::is_same_v<typename PointT2::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 sep(const PointT1& p1, const PointT2& p2) const {
    return from_fractional(frac_minimum_image(to_fractional(p2 - p1)));
  }

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& p) const {
    return from_fractional(frac_wrap_to_unit_cell(to_fractional(p)));
  }

  /// \brief Lattice vectors as columns of a diagonal matrix.
  /// Non-periodic entries appear as 1 by convention.
  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return OurMatrix3::diagonal(scale_);
  }

  template <ValidPointType PointT, typename Integer>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& p, const Vector3<Integer>& n) const {
    return translate(p, elementwise_mul(scale_, n.template cast<Scalar>()));
  }
  //@}

 private:
  OurVector3 scale_{Scalar(1), Scalar(1), Scalar(1)};
  OurVector3 scale_inv_{Scalar(1), Scalar(1), Scalar(1)};
};

/// \brief General periodic metric for a tilted unit cell.
///
/// \tparam PeriodicAxes Bitmask of periodic lattice-vector directions. Combine
///   AXIS_X (0b001), AXIS_Y (0b010), AXIS_Z (0b100) with | for multi-axis
///   periodicity. AXIS_X/Y/Z refer to the first, second, and third lattice vector
///   directions (fractional axes), not Cartesian x/y/z.
/// \tparam Scalar Floating-point scalar type.
///
/// The cell is described by a 3x3 matrix h whose columns are the lattice vectors.
/// For axis-aligned cells prefer OrthorhombicMetric: it avoids the matrix
/// multiplications in to_fractional and from_fractional.
template <unsigned PeriodicAxes, typename Scalar>
class TriclinicMetric {
  static_assert(PeriodicAxes > 0 && PeriodicAxes <= AXIS_XYZ,
                "PeriodicAxes must be a non-zero combination of AXIS_X, AXIS_Y, AXIS_Z");

 public:
  //! \name Type aliases
  //@{

  using value_type = Scalar;
  using OurVector3 = Vector3<Scalar>;
  using OurMatrix3 = Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;
  //@}

  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TriclinicMetric() = default;

  KOKKOS_INLINE_FUNCTION
  explicit constexpr TriclinicMetric(const OurMatrix3& h) : h_(h), h_inv_(inverse(h)) {
  }

  KOKKOS_INLINE_FUNCTION
  void set_cell_matrix(const OurMatrix3& h) {
    h_ = h;
    h_inv_ = inverse(h);
  }
  //@}

  //! \name Periodicity queries
  //@{

  template <unsigned dim>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return (PeriodicAxes >> dim) & 1u;
  }
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned dim) const {
    return (PeriodicAxes >> dim) & 1u;
  }
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return ((PeriodicAxes & 1u) + ((PeriodicAxes >> 1) & 1u) + ((PeriodicAxes >> 2) & 1u));
  }
  //@}

  //! \name Metric operations
  //@{

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& p) const {
    return h_inv_ * p;
  }

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& frac) const {
    return h_ * frac;
  }

  /// \brief Minimum-image displacement in fractional coordinates.
  ///
  /// Maps each periodic fractional component to [-0.5, 0.5) by subtracting round().
  /// Non-periodic fractional components are passed through unchanged.
  template <ValidVector3Type Vector3T>
  MUNDY_REQUIRES(std::is_same_v<typename Vector3T::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 frac_minimum_image(const Vector3T& fv) const {
    OurVector3 r{fv[0], fv[1], fv[2]};
    if constexpr (PeriodicAxes & AXIS_X) r[0] -= round(r[0]);
    if constexpr (PeriodicAxes & AXIS_Y) r[1] -= round(r[1]);
    if constexpr (PeriodicAxes & AXIS_Z) r[2] -= round(r[2]);
    return r;
  }

  template <ValidVector3Type Vector3T>
  MUNDY_REQUIRES(std::is_same_v<typename Vector3T::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 frac_wrap_to_unit_cell(const Vector3T& fv) const {
    OurVector3 r{fv[0], fv[1], fv[2]};
    if constexpr (PeriodicAxes & AXIS_X) r[0] = impl::safe_unit_mod1(r[0]);
    if constexpr (PeriodicAxes & AXIS_Y) r[1] = impl::safe_unit_mod1(r[1]);
    if constexpr (PeriodicAxes & AXIS_Z) r[2] = impl::safe_unit_mod1(r[2]);
    return r;
  }

  template <ValidPointType PointT1, ValidPointType PointT2>
  MUNDY_REQUIRES(
      std::is_same_v<typename PointT1::value_type, Scalar>&& std::is_same_v<typename PointT2::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurVector3 sep(const PointT1& p1, const PointT2& p2) const {
    return from_fractional(frac_minimum_image(to_fractional(p2 - p1)));
  }

  template <ValidPointType PointT>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& p) const {
    return from_fractional(frac_wrap_to_unit_cell(to_fractional(p)));
  }

  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return h_;
  }

  template <ValidPointType PointT, typename Integer>
  MUNDY_REQUIRES(std::is_same_v<typename PointT::value_type, Scalar>)
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& p, const Vector3<Integer>& n) const {
    return translate(p, h_ * n.template cast<Scalar>());
  }
  //@}

 private:
  OurMatrix3 h_{Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1)};
  OurMatrix3 h_inv_{Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1)};
};

//! \name Non-member metric constructors
//@{

/// \brief Orthorhombic metric from axis-aligned cell widths.
///
/// \tparam PeriodicAxes Bitmask of periodic axes; defaults to AXIS_XYZ (all periodic).
template <unsigned PeriodicAxes = AXIS_XYZ, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr OrthorhombicMetric<PeriodicAxes, Scalar> make_orthorhombic_metric(
    const Vector3<Scalar>& cell_widths) {
  return OrthorhombicMetric<PeriodicAxes, Scalar>{cell_widths};
}

/// \brief Orthorhombic metric from domain corners.
///
/// \tparam PeriodicAxes Bitmask of periodic axes; defaults to AXIS_XYZ (all periodic).
template <unsigned PeriodicAxes = AXIS_XYZ, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr OrthorhombicMetric<PeriodicAxes, Scalar> make_orthorhombic_metric(
    const Vector3<Scalar>& domain_min, const Vector3<Scalar>& domain_max) {
  return OrthorhombicMetric<PeriodicAxes, Scalar>{domain_max - domain_min};
}

/// \brief Triclinic metric from a cell matrix.
///
/// \tparam PeriodicAxes Bitmask of periodic lattice-vector directions; defaults to
///   AXIS_XYZ (all three lattice directions periodic).
template <unsigned PeriodicAxes = AXIS_XYZ, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr TriclinicMetric<PeriodicAxes, Scalar> make_triclinic_metric(const Matrix3<Scalar>& h) {
  return TriclinicMetric<PeriodicAxes, Scalar>{h};
}

/// \brief Triclinic metric from axis-aligned cell widths (diagonal h).
///
/// \tparam PeriodicAxes Bitmask of periodic lattice-vector directions; defaults to
///   AXIS_XYZ (all three lattice directions periodic).
template <unsigned PeriodicAxes = AXIS_XYZ, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr TriclinicMetric<PeriodicAxes, Scalar> make_triclinic_metric(
    const Vector3<Scalar>& cell_widths) {
  return TriclinicMetric<PeriodicAxes, Scalar>{Matrix3<Scalar>::diagonal(cell_widths)};
}

//@}

/// \brief Runtime-polymorphic metric (stk::topology-style value class).
///
/// Holds a metric_t type tag plus cell parameters as an inline flat array.
/// Construct from an enum or a string, inspect with is_periodic() / name(),
/// stream with <<, and dispatch to a concrete metric type with visit().
///
///   mundy::metric m(mundy::metric::ORTHORHOMBIC_XYZ);
///   m.set_cell_widths({10.0, 10.0, 10.0});
///   std::cout << m << "\n";                 // prints "ORTHORHOMBIC_XYZ"
///   m.visit([&](auto&& concrete) {          // concrete is OrthorhombicMetric<AXIS_XYZ, double>
///       Kokkos::parallel_for(n, MyFunctor{positions, concrete});
///   });
///
/// All cell parameters and concrete metrics produced by visit() are double-precision.
/// Scalar-typed concrete metrics are not supported at the runtime level; obtain them
/// directly from OrthorhombicMetric<> or TriclinicMetric<> if float precision is needed.
///
/// This class is host-only. Device kernels should receive concrete metric types
/// directly so the compiler can inline all operations.
class metric {
 public:
  //! \name Type aliases
  //@{

  enum metric_t : unsigned {
    INVALID_METRIC = 0,  ///< Default-constructed sentinel; metric is not yet configured.
    FREE_SPACE,          ///< Non-periodic, Euclidean metric
    ORTHORHOMBIC_X,      ///< Axis-aligned cell, periodic along X only
    ORTHORHOMBIC_Y,      ///< Axis-aligned cell, periodic along Y only
    ORTHORHOMBIC_Z,      ///< Axis-aligned cell, periodic along Z only
    ORTHORHOMBIC_XY,     ///< Axis-aligned cell, periodic along X and Y
    ORTHORHOMBIC_XZ,     ///< Axis-aligned cell, periodic along X and Z
    ORTHORHOMBIC_YZ,     ///< Axis-aligned cell, periodic along Y and Z
    ORTHORHOMBIC_XYZ,    ///< Axis-aligned cell, periodic along all three axes
    TRICLINIC_X,         ///< Tilted cell, periodic along first lattice vector only
    TRICLINIC_Y,         ///< Tilted cell, periodic along second lattice vector only
    TRICLINIC_Z,         ///< Tilted cell, periodic along third lattice vector only
    TRICLINIC_XY,        ///< Tilted cell, periodic along first and second lattice vectors
    TRICLINIC_XZ,        ///< Tilted cell, periodic along first and third lattice vectors
    TRICLINIC_YZ,        ///< Tilted cell, periodic along second and third lattice vectors
    TRICLINIC_XYZ,       ///< Tilted cell, periodic along all three lattice vectors
    NUM_METRIC_TYPES,
  };

  static constexpr metric_t ORTHORHOMBIC = ORTHORHOMBIC_XYZ;      ///< Alias for the common fully-periodic case
  static constexpr metric_t TRICLINIC = TRICLINIC_XYZ;            ///< Alias for the common fully-periodic case
  static constexpr metric_t ORTHORHOMBIC_START = ORTHORHOMBIC_X;  ///< First orthorhombic enum value (range sentinel)
  static constexpr metric_t ORTHORHOMBIC_END = ORTHORHOMBIC_XYZ;  ///< Last orthorhombic enum value (range sentinel)
  static constexpr metric_t TRICLINIC_START = TRICLINIC_X;        ///< First triclinic enum value (range sentinel)
  static constexpr metric_t TRICLINIC_END = TRICLINIC_XYZ;        ///< Last triclinic enum value (range sentinel)
  //@}

  //! \name Constructors
  //@{

  metric() = default;
  explicit metric(metric_t m) : value_(m) {
  }

  /// \brief Construct from a string name. Throws std::invalid_argument on unrecognized input.
  /// Recognised names are the enum spellings plus "ORTHORHOMBIC" / "TRICLINIC" as aliases for
  /// ORTHORHOMBIC_XYZ and TRICLINIC_XYZ respectively.
  static metric from_string(std::string_view name);
  //@}

  //! \name Queries
  //@{

  bool is_valid() const {
    return value_ != INVALID_METRIC && value_ < NUM_METRIC_TYPES;
  }
  metric_t value() const {
    return value_;
  }

  /// \brief Canonical string name. Round-trips through from_string.
  std::string_view name() const;

  /// \brief True if the given dimension is periodic under this metric.
  /// For orthorhombic metrics, dim maps to Cartesian x/y/z.
  /// For triclinic metrics, dim maps to the first/second/third lattice vector direction.
  bool is_periodic(unsigned dim) const;

  friend std::ostream& operator<<(std::ostream& os, const metric& m) {
    return os << m.name();
  }
  //@}

  //! \name Configuration
  //@{

  /// \brief Set cell widths. Only valid for ORTHORHOMBIC_* types.
  void set_cell_widths(const Vector3<double>& cell_widths);

  /// \brief Set cell matrix. Only valid for TRICLINIC_* types.
  void set_cell_matrix(const Matrix3<double>& h);
  //@}

  //! \name Visitation
  //@{

  /// \brief Dispatch to a concrete metric type and invoke f with it.
  ///
  /// f receives one of: FreeSpaceMetric<double>,
  ///                    OrthorhombicMetric<axes, double>,
  ///                    TriclinicMetric<axes, double>.
  /// The switch is a one-time cost; the compiler inlines f for each arm.
  /// Asserts is_valid() before dispatching.
  template <typename Functor>
  void visit(Functor&& f) const;
  //@}

 private:
  // Assumes enum values within each family are contiguous (ORTHORHOMBIC_START..END, TRICLINIC_START..END).
  static bool is_orthorhombic_type(metric_t m) {
    return m >= ORTHORHOMBIC_START && m <= ORTHORHOMBIC_END;
  }
  static bool is_triclinic_type(metric_t m) {
    return m >= TRICLINIC_START && m <= TRICLINIC_END;
  }
  static unsigned periodic_axes_of(metric_t m);

  metric_t value_ = INVALID_METRIC;
  // data_[0..2] = cell widths for ORTHORHOMBIC_*
  // data_[0..8] = cell matrix (row-major) for TRICLINIC_*
  double data_[9] = {};
};

inline std::string_view metric::name() const {
  switch (value_) {
    case INVALID_METRIC:
      return "INVALID_METRIC";
    case FREE_SPACE:
      return "FREE_SPACE";
    case ORTHORHOMBIC_X:
      return "ORTHORHOMBIC_X";
    case ORTHORHOMBIC_Y:
      return "ORTHORHOMBIC_Y";
    case ORTHORHOMBIC_Z:
      return "ORTHORHOMBIC_Z";
    case ORTHORHOMBIC_XY:
      return "ORTHORHOMBIC_XY";
    case ORTHORHOMBIC_XZ:
      return "ORTHORHOMBIC_XZ";
    case ORTHORHOMBIC_YZ:
      return "ORTHORHOMBIC_YZ";
    case ORTHORHOMBIC_XYZ:
      return "ORTHORHOMBIC_XYZ";
    case TRICLINIC_X:
      return "TRICLINIC_X";
    case TRICLINIC_Y:
      return "TRICLINIC_Y";
    case TRICLINIC_Z:
      return "TRICLINIC_Z";
    case TRICLINIC_XY:
      return "TRICLINIC_XY";
    case TRICLINIC_XZ:
      return "TRICLINIC_XZ";
    case TRICLINIC_YZ:
      return "TRICLINIC_YZ";
    case TRICLINIC_XYZ:
      return "TRICLINIC_XYZ";
    default:
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "metric::name: unhandled metric_t value");
      return "";  // unreachable; silences -Wreturn-type
  }
}

inline metric metric::from_string(std::string_view name) {
  if (name == "FREE_SPACE") return metric{FREE_SPACE};
  if (name == "ORTHORHOMBIC_X") return metric{ORTHORHOMBIC_X};
  if (name == "ORTHORHOMBIC_Y") return metric{ORTHORHOMBIC_Y};
  if (name == "ORTHORHOMBIC_Z") return metric{ORTHORHOMBIC_Z};
  if (name == "ORTHORHOMBIC_XY") return metric{ORTHORHOMBIC_XY};
  if (name == "ORTHORHOMBIC_XZ") return metric{ORTHORHOMBIC_XZ};
  if (name == "ORTHORHOMBIC_YZ") return metric{ORTHORHOMBIC_YZ};
  if (name == "ORTHORHOMBIC_XYZ") return metric{ORTHORHOMBIC_XYZ};
  if (name == "ORTHORHOMBIC") return metric{ORTHORHOMBIC_XYZ};
  if (name == "TRICLINIC_X") return metric{TRICLINIC_X};
  if (name == "TRICLINIC_Y") return metric{TRICLINIC_Y};
  if (name == "TRICLINIC_Z") return metric{TRICLINIC_Z};
  if (name == "TRICLINIC_XY") return metric{TRICLINIC_XY};
  if (name == "TRICLINIC_XZ") return metric{TRICLINIC_XZ};
  if (name == "TRICLINIC_YZ") return metric{TRICLINIC_YZ};
  if (name == "TRICLINIC_XYZ") return metric{TRICLINIC_XYZ};
  if (name == "TRICLINIC") return metric{TRICLINIC_XYZ};
  MUNDY_THROW_ASSERT(false, std::invalid_argument,
                     "metric::from_string: unrecognized name. "
                     "Valid names: FREE_SPACE, ORTHORHOMBIC_X/Y/Z/XY/XZ/YZ/XYZ, TRICLINIC_X/Y/Z/XY/XZ/YZ/XYZ.");
  return metric{};
}

inline unsigned metric::periodic_axes_of(metric_t m) {
  switch (m) {
    case FREE_SPACE:
      return 0u;
    case ORTHORHOMBIC_X:
      return AXIS_X;
    case ORTHORHOMBIC_Y:
      return AXIS_Y;
    case ORTHORHOMBIC_Z:
      return AXIS_Z;
    case ORTHORHOMBIC_XY:
      return AXIS_XY;
    case ORTHORHOMBIC_XZ:
      return AXIS_XZ;
    case ORTHORHOMBIC_YZ:
      return AXIS_YZ;
    case ORTHORHOMBIC_XYZ:
      return AXIS_XYZ;
    case TRICLINIC_X:
      return AXIS_X;
    case TRICLINIC_Y:
      return AXIS_Y;
    case TRICLINIC_Z:
      return AXIS_Z;
    case TRICLINIC_XY:
      return AXIS_XY;
    case TRICLINIC_XZ:
      return AXIS_XZ;
    case TRICLINIC_YZ:
      return AXIS_YZ;
    case TRICLINIC_XYZ:
      return AXIS_XYZ;
    default:
      return 0u;
  }
}

inline bool metric::is_periodic(unsigned dim) const {
  return (periodic_axes_of(value_) >> dim) & 1u;
}

inline void metric::set_cell_widths(const Vector3<double>& cell_widths) {
  MUNDY_THROW_ASSERT(is_orthorhombic_type(value_), std::invalid_argument,
                     "set_cell_widths is only valid for ORTHORHOMBIC_* metric types");
  data_[0] = cell_widths[0];
  data_[1] = cell_widths[1];
  data_[2] = cell_widths[2];
}

inline void metric::set_cell_matrix(const Matrix3<double>& h) {
  MUNDY_THROW_ASSERT(is_triclinic_type(value_), std::invalid_argument,
                     "set_cell_matrix is only valid for TRICLINIC_* metric types");
  data_[0] = h(0, 0);
  data_[1] = h(0, 1);
  data_[2] = h(0, 2);
  data_[3] = h(1, 0);
  data_[4] = h(1, 1);
  data_[5] = h(1, 2);
  data_[6] = h(2, 0);
  data_[7] = h(2, 1);
  data_[8] = h(2, 2);
}

template <typename Functor>
void metric::visit(Functor&& f) const {
  MUNDY_THROW_ASSERT(is_valid(), std::invalid_argument, "metric::visit called on an invalid metric");
  const Vector3<double> scale{data_[0], data_[1], data_[2]};
  const Matrix3<double> h{data_[0], data_[1], data_[2], data_[3], data_[4], data_[5], data_[6], data_[7], data_[8]};
  switch (value_) {
    case FREE_SPACE:
      f(FreeSpaceMetric<double>{});
      return;
    case ORTHORHOMBIC_X:
      f(OrthorhombicMetric<AXIS_X, double>{scale});
      return;
    case ORTHORHOMBIC_Y:
      f(OrthorhombicMetric<AXIS_Y, double>{scale});
      return;
    case ORTHORHOMBIC_Z:
      f(OrthorhombicMetric<AXIS_Z, double>{scale});
      return;
    case ORTHORHOMBIC_XY:
      f(OrthorhombicMetric<AXIS_XY, double>{scale});
      return;
    case ORTHORHOMBIC_XZ:
      f(OrthorhombicMetric<AXIS_XZ, double>{scale});
      return;
    case ORTHORHOMBIC_YZ:
      f(OrthorhombicMetric<AXIS_YZ, double>{scale});
      return;
    case ORTHORHOMBIC_XYZ:
      f(OrthorhombicMetric<AXIS_XYZ, double>{scale});
      return;
    case TRICLINIC_X:
      f(TriclinicMetric<AXIS_X, double>{h});
      return;
    case TRICLINIC_Y:
      f(TriclinicMetric<AXIS_Y, double>{h});
      return;
    case TRICLINIC_Z:
      f(TriclinicMetric<AXIS_Z, double>{h});
      return;
    case TRICLINIC_XY:
      f(TriclinicMetric<AXIS_XY, double>{h});
      return;
    case TRICLINIC_XZ:
      f(TriclinicMetric<AXIS_XZ, double>{h});
      return;
    case TRICLINIC_YZ:
      f(TriclinicMetric<AXIS_YZ, double>{h});
      return;
    case TRICLINIC_XYZ:
      f(TriclinicMetric<AXIS_XYZ, double>{h});
      return;
    default:
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "metric::visit: unhandled metric_t value");
  }
}

//! \name reference_point protocol
//@{

/// \brief Returns the canonical reference point for an object.
///
/// wrap_rigid and shift_image translate the whole object so this point reaches
/// the target location; all other constituent points move with it.
///
/// Mapping:
///   Point                  -> the point itself
///   LineSegment            -> start
///   VSegment               -> start
///   SpherocylinderSegment  -> start
///   AABB                   -> min_corner
///   All others             -> center
template <typename Object>
KOKKOS_INLINE_FUNCTION auto reference_point(const Object& obj);

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)

template <ValidPointType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& p) {
  return p;
}

template <ValidLineType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& l) {
  return l.center();
}

template <ValidLineSegmentType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& ls) {
  return ls.start();
}

template <ValidCircle3DType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& c) {
  return c.center();
}

template <ValidVSegmentType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& vs) {
  return vs.start();
}

template <ValidAABBType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& aabb) {
  return aabb.min_corner();
}

template <ValidSphereType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& s) {
  return s.center();
}

template <ValidSpherocylinderType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& sc) {
  return sc.center();
}

template <ValidSpherocylinderSegmentType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& scs) {
  return scs.start();
}

template <ValidRingType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& r) {
  return r.center();
}

template <ValidEllipsoidType T>
KOKKOS_INLINE_FUNCTION Point<typename T::value_type> reference_point(const T& e) {
  return e.center();
}

#endif  // DOXYGEN_SHOULD_SKIP_THIS

//@}

//! \name Wrapping utilities
//@{

/// \brief Translate an object by an integer number of lattice images.
///
/// The displacement is determined by shifting the reference point and applied
/// rigidly to the whole object via translate().
template <typename Integer, typename Object, typename Metric>
KOKKOS_INLINE_FUNCTION auto shift_image(const Object& obj, const Vector3<Integer>& lattice_vector,
                                        const Metric& metric) {
  const auto ref = reference_point(obj);
  return translate(obj, metric.shift_image(ref, lattice_vector) - ref);
}

/// \brief Translate an object so its reference point lies in the primary cell.
///
/// The whole object moves as a rigid body: orientation, shape, and relative
/// positions of all constituent points are preserved. The reference point is
/// defined by reference_point(obj).
template <typename Object, typename Metric>
KOKKOS_INLINE_FUNCTION auto wrap_rigid(const Object& obj, const Metric& metric) {
  const auto ref = reference_point(obj);
  return translate(obj, metric.wrap(ref) - ref);
}

/// \brief In-place variant of wrap_rigid.
template <typename Object, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Object& obj, const Metric& metric) {
  const auto ref = reference_point(obj);
  translate_inplace(obj, metric.wrap(ref) - ref);
}

/// \brief Wrap each geometric point of an object independently into the primary cell.
///
/// Each point is wrapped in isolation via metric.wrap(). For multi-point objects
/// (LineSegment, VSegment, AABB, etc.) this does NOT preserve relative positions
/// between points — use wrap_rigid when shape must be maintained.
///
/// Requires FinitePrimitive<Object>. Infinite primitives (e.g. Line) do not satisfy
/// this — use wrap_rigid for those instead.
template <FinitePrimitive Object, typename Metric>
KOKKOS_INLINE_FUNCTION auto wrap_points(const Object& obj, const Metric& metric) {
  auto result = obj;
  for_each_point_mutable(result, [&](auto& p) { p = metric.wrap(p); });
  return result;
}

/// \brief In-place variant of wrap_points.
template <FinitePrimitive Object, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Object& obj, const Metric& metric) {
  for_each_point_mutable(obj, [&](auto& p) { p = metric.wrap(p); });
}

/// \brief Move each point to the periodic image closest to ref_point.
///
/// For each constituent point p, computes:
///   p' = from_fractional(frac(ref) + frac_minimum_image(frac(p) - frac(ref)))
///
/// This is the inverse of wrap_points only when ref_point is already inside the
/// primary cell. Otherwise the result equals wrap_rigid applied to the original
/// object:
///   wrap_rigid(s, m) == unwrap_points_to_ref(wrap_points(s, m), m, reference_point(s))
///
/// Requires FinitePrimitive<Object>.
template <FinitePrimitive Object, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION auto unwrap_points_to_ref(const Object& obj, const Metric& metric, const PointT& ref_point) {
  const auto sr = metric.to_fractional(ref_point);
  auto result = obj;
  for_each_point_mutable(result, [&](auto& p) {
    p = metric.from_fractional(sr + metric.frac_minimum_image(metric.to_fractional(p) - sr));
  });
  return result;
}

/// \brief In-place variant of unwrap_points_to_ref.
template <FinitePrimitive Object, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Object& obj, const Metric& metric, const PointT& ref_point) {
  const auto sr = metric.to_fractional(ref_point);
  for_each_point_mutable(
      obj, [&](auto& p) { p = metric.from_fractional(sr + metric.frac_minimum_image(metric.to_fractional(p) - sr)); });
}

/// \brief The integer periodic image of a point: the lattice cell `k` such that the point lies in cell `k`.
///
/// Computed in fractional coordinates, where wrapping is genuinely per-axis, so it is exact and correct for any
/// lattice — orthorhombic or tilted. A non-periodic axis wraps to itself, giving `k = 0` there. The displacement that
/// wraps the point into the primary cell is `-lattice_displacement(image_index(p, m), m)`; the integer reconstruction
/// avoids the sub-ULP noise of the metric's `wrap`/`from_fractional` round-trip.
template <ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Vector3<int> image_index(const PointT& p, const Metric& metric) {
  const auto f = metric.to_fractional(p);
  const auto fw = metric.frac_wrap_to_unit_cell(f);
  return Vector3<int>{static_cast<int>(round(f[0] - fw[0])), static_cast<int>(round(f[1] - fw[1])),
                      static_cast<int>(round(f[2] - fw[2]))};
}

/// \brief The Cartesian displacement of an integer lattice combination `n`, i.e. `Σ nᵢ·aᵢ` over the lattice vectors.
///
/// Applies the metric's lattice vectors to the integer fractional offset via `from_fractional`, so it is exact and
/// correct for tilted cells. The result is in the metric's scalar type.
template <typename Integer, typename Metric>
KOKKOS_INLINE_FUNCTION Vector3<typename Metric::value_type> lattice_displacement(const Vector3<Integer>& n,
                                                                                 const Metric& metric) {
  using Scalar = typename Metric::value_type;
  const auto d = metric.from_fractional(
      Point<Scalar>{static_cast<Scalar>(n[0]), static_cast<Scalar>(n[1]), static_cast<Scalar>(n[2])});
  return Vector3<Scalar>{d[0], d[1], d[2]};
}

//@}

// =============================================================================
//! \name Metric type traits
//!
//! Compile-time predicates that classify concrete metric types.  These traits
//! are defined here alongside the metric classes so that any code working with
//! metrics can branch on their structural properties without inspecting member
//! names or relying on ad-hoc partial specialisations elsewhere.
//!
//! Primary templates evaluate to `false_type`; explicit specialisations below
//! opt each metric family in to the appropriate trait.
//@{
// =============================================================================

/// \brief True when T is any instantiation of `FreeSpaceMetric`.
///
/// A `FreeSpaceMetric` represents unbounded Euclidean space: no periodic
/// images, identity wrapping, and direct Cartesian displacements.  This trait
/// distinguishes it from every periodic metric family.
template <typename T>
struct is_free_space_metric : std::false_type {};
template <typename Scalar>
struct is_free_space_metric<FreeSpaceMetric<Scalar>> : std::true_type {};
template <typename T>
inline constexpr bool is_free_space_metric_v = is_free_space_metric<T>::value;

/// \brief True when T is any instantiation of `OrthorhombicMetric`.
///
/// An `OrthorhombicMetric` represents an axis-aligned periodic cell.  The
/// bitmask of periodic axes is a template parameter; a metric that is periodic
/// along any subset of axes satisfies this trait.
template <typename T>
struct is_orthorhombic_metric : std::false_type {};
template <unsigned PeriodicAxes, typename Scalar>
struct is_orthorhombic_metric<OrthorhombicMetric<PeriodicAxes, Scalar>> : std::true_type {};
template <typename T>
inline constexpr bool is_orthorhombic_metric_v = is_orthorhombic_metric<T>::value;

/// \brief True when T is any instantiation of `TriclinicMetric`.
///
/// A `TriclinicMetric` represents a general (tilted) periodic cell described
/// by an arbitrary 3×3 lattice matrix.
template <typename T>
struct is_triclinic_metric : std::false_type {};
template <unsigned PeriodicAxes, typename Scalar>
struct is_triclinic_metric<TriclinicMetric<PeriodicAxes, Scalar>> : std::true_type {};
template <typename T>
inline constexpr bool is_triclinic_metric_v = is_triclinic_metric<T>::value;

/// \brief True when T is any periodic metric (orthorhombic or triclinic).
///
/// Evaluates to true for any `OrthorhombicMetric` or `TriclinicMetric`
/// instantiation, regardless of which axes are marked periodic.
template <typename T>
struct is_periodic_metric
    : std::bool_constant<is_orthorhombic_metric_v<T> || is_triclinic_metric_v<T>> {};
template <typename T>
inline constexpr bool is_periodic_metric_v = is_periodic_metric<T>::value;

//@}

}  // namespace mundy

#endif  // MUNDY_GEOM_PERIODICITY_HPP_
