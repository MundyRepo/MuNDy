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

}  // namespace impl

template <typename Scalar>
class EuclideanMetric {
 public:
  /// \brief Type aliases
  using scalar_t = Scalar;
  using OurVector3 = math::Vector3<Scalar>;
  using OurMatrix3 = math::Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Get if the given dimension is periodic
  template<unsigned dimension>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return false;
  }

  /// \brief Get if the given dimension is periodic
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned /*dimension*/) const {
    return false;
  }

  /// \brief Get the number of periodic dimensions
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 0;
  }

  /// \brief Map a point into fractional coordinates
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& point) const {
    return point;
  }

  /// \brief Map a point from fractional coordinates to real space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& point_frac) const {
    return point_frac;
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_minimum_image(
      const Vector3T& fractional_vec) const {
    return fractional_vec;
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_wrap_to_unit_cell(
      const Vector3T& fractional_vec) const {
    return fractional_vec;
  }

  /// \brief Distance vector between two points in free space (from point1 to point2)
  template <ValidPointType PointT1, ValidPointType PointT2>
    requires std::is_same_v<typename PointT1::scalar_t, Scalar> && std::is_same_v<typename PointT2::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint sep(const PointT1& point1, const PointT2& point2) const {
    return point2 - point1;
  }

  /// \brief Wrap a point into the free space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& point) const {
    return point;
  }

  /// \brief Direct lattice vectors (return as the columns of a matrix)
  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return OurMatrix3::identity();
  }

  /// \brief Shift a point by a given number of lattice images in each direction (free space does nothing)
  template <ValidPointType PointT, typename Integer>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(
      const PointT& point, [[maybe_unused]] const math::Vector3<Integer>& num_images) const {
    return point;
  }
};  // EuclideanMetric

template <typename Scalar>
class PeriodicMetric {
 public:
  /// \brief Type aliases
  using scalar_t = Scalar;
  using OurVector3 = math::Vector3<Scalar>;
  using OurMatrix3 = math::Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicMetric() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicMetric(const OurMatrix3& h) : h_(h), h_inv_(math::inverse(h)) {
  }

  /// \brief Set the unit cell matrix
  KOKKOS_INLINE_FUNCTION
  void constexpr set_unit_cell_matrix(const OurMatrix3& h) {
    h_ = h;
    h_inv_ = math::inverse(h_);
  }

  /// \brief Get if the given dimension is periodic
  template<unsigned dimension>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return true;
  }

  /// \brief Get if the given dimension is periodic
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned /*dimension*/) const {
    return true;
  }

  /// \brief Get the number of periodic dimensions
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 3;
  }

  /// \brief Map a point into fractional coordinates
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& point) const {
    return h_inv_ * point;
  }

  /// \brief Map a point from fractional coordinates to real space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& point_frac) const {
    return h_ * point_frac;
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_minimum_image(
      const Vector3T& fractional_vec) const {
    return apply([](Scalar x) { return x - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(x))); },
                 fractional_vec);
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_wrap_to_unit_cell(
      const Vector3T& fractional_vec) const {
    return apply([](Scalar x) { return impl::safe_unit_mod1<Integer>(x); }, fractional_vec);
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  template <ValidPointType PointT1, ValidPointType PointT2>
    requires std::is_same_v<typename PointT1::scalar_t, Scalar> && std::is_same_v<typename PointT2::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint sep(const PointT1& point1, const PointT2& point2) const {
    // Assumes linearity of to_fractional
    return from_fractional(frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& point) const {
    return from_fractional(frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// \brief Direct lattice vectors (return as the columns of a matrix)
  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return h_;
  }

  /// \brief Shift a point by a given number of lattice images in each direction
  template <ValidPointType PointT, typename Integer>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& point,
                                                        const math::Vector3<Integer>& num_images) const {
    return translate(point, h_ * num_images);
  }

 private:
  OurMatrix3 h_;      ///< Unit cell matrix
  OurMatrix3 h_inv_;  ///< Inverse of the unit cell matrix
};  // PeriodicMetric


template <typename Scalar>
class PeriodicMetricX {
 public:
  /// \brief Type aliases
  using scalar_t = Scalar;
  using OurVector3 = math::Vector3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicMetricX() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicMetricX(const double width_x) : scale_{width_x, 1.0, 1.0}, inv_scale_{1.0 / width_x, 1.0, 1.0} {
    MUNDY_THROW_ASSERT(width_x > 0, std::invalid_argument, "Cell dimensions must be positive");
  }

  /// \brief Get if the given dimension is periodic
  template<unsigned dimension>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return dimension == 0;
  }

  /// \brief Get if the given dimension is periodic
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned dimension) const {
    return dimension == 0;
  }

  /// \brief Get the number of periodic dimensions
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 1;
  }

  /// \brief Map a point into fractional coordinates
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& point) const {
    return math::elementwise_mul(inv_scale_, point);
  }

  /// \brief Map a point from fractional coordinates to real space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& point_frac) const {
    return math::elementwise_mul(scale_, point_frac);
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_minimum_image(
      const Vector3T& fractional_vec) const {
    OurVector3 min_image{
      fractional_vec[0] - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(fractional_vec[0]))),
      fractional_vec[1],
      fractional_vec[2]};
    return min_image;
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_wrap_to_unit_cell(
      const Vector3T& fractional_vec) const {
    OurVector3 wrapped{
      impl::safe_unit_mod1<Integer>(fractional_vec[0]),
      fractional_vec[1],
      fractional_vec[2]};
    return wrapped;
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  template <ValidPointType PointT1, ValidPointType PointT2>
    requires std::is_same_v<typename PointT1::scalar_t, Scalar> && std::is_same_v<typename PointT2::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint sep(const PointT1& point1, const PointT2& point2) const {
    // Assumes linearity of to_fractional
    return from_fractional(frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& point) const {
    return from_fractional(frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// TODO(palmerb4): I don't think this should be offered.
  // /// \brief Direct lattice vectors (return as the columns of a matrix)
  // KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
  //   return h_;
  // }

  /// \brief Shift a point by a given number of lattice images in each direction
  template <ValidPointType PointT, typename Integer>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& point,
                                                        const math::Vector3<Integer>& num_images) const {
    return translate(point, from_fractional(num_images.template cast<Scalar>()));
  }

 private:
  OurVector3 scale_;      ///< Unit cell scaling factors
  OurVector3 inv_scale_;  ///< Inverse of the scaling factors
};  // PeriodicMetricX



template <typename Scalar>
class PeriodicMetricY {
 public:
  /// \brief Type aliases
  using scalar_t = Scalar;
  using OurVector3 = math::Vector3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicMetricY() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicMetricY(const double width_y) : scale_{1.0, width_y, 1.0}, inv_scale_{1.0, 1.0 / width_y, 1.0} {
    MUNDY_THROW_ASSERT(width_y > 0, std::invalid_argument, "Cell dimensions must be positive");
  }

  /// \brief Get if the given dimension is periodic
  template<unsigned dimension>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return dimension == 1;
  }

  /// \brief Get if the given dimension is periodic
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned dimension) const {
    return dimension == 1;
  }

  /// \brief Get the number of periodic dimensions
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 1;
  }

  /// \brief Map a point into fractional coordinates
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& point) const {
    return math::elementwise_mul(inv_scale_, point);
  }

  /// \brief Map a point from fractional coordinates to real space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& point_frac) const {
    return math::elementwise_mul(scale_, point_frac);
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_minimum_image(
      const Vector3T& fractional_vec) const {
    OurVector3 min_image{
      fractional_vec[0],
      fractional_vec[1] - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(fractional_vec[1]))),
      fractional_vec[2]};
    return min_image;
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_wrap_to_unit_cell(
      const Vector3T& fractional_vec) const {
    OurVector3 wrapped{
      fractional_vec[0],
      impl::safe_unit_mod1<Integer>(fractional_vec[1]),
      fractional_vec[2]};
    return wrapped;
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  template <ValidPointType PointT1, ValidPointType PointT2>
    requires std::is_same_v<typename PointT1::scalar_t, Scalar> && std::is_same_v<typename PointT2::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint sep(const PointT1& point1, const PointT2& point2) const {
    // Assumes linearity of to_fractional
    return from_fractional(frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& point) const {
    return from_fractional(frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// TODO(palmerb4): I don't think this should be offered.
  // /// \brief Direct lattice vectors (return as the columns of a matrix)
  // KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
  //   return h_;
  // }

  /// \brief Shift a point by a given number of lattice images in each direction
  template <ValidPointType PointT, typename Integer>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& point,
                                                        const math::Vector3<Integer>& num_images) const {
    return translate(point, from_fractional(num_images.template cast<Scalar>()));
  }

 private:
  OurVector3 scale_;      ///< Unit cell scaling factors
  OurVector3 inv_scale_;  ///< Inverse of the scaling factors
};  // PeriodicMetricY

template <typename Scalar>
class PeriodicMetricXY {
 public:
  /// \brief Type aliases
  using scalar_t = Scalar;
  using OurVector3 = math::Vector3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicMetricXY() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicMetricXY(const double width_x, const double width_y) : scale_{width_x, width_y, 1.0}, inv_scale_{1.0 / width_x, 1.0 / width_y, 1.0} {
    MUNDY_THROW_ASSERT(width_x > 0 && width_y > 0, std::invalid_argument, "Cell dimensions must be positive");
  }

  /// \brief Get if the given dimension is periodic
  template<unsigned dimension>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return dimension == 0 || dimension == 1;
  }

  /// \brief Get if the given dimension is periodic
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned dimension) const {
    return dimension == 0 || dimension == 1;
  }

  /// \brief Get the number of periodic dimensions
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 2;
  }

  /// \brief Map a point into fractional coordinates
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& point) const {
    return math::elementwise_mul(inv_scale_, point);
  }

  /// \brief Map a point from fractional coordinates to real space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& point_frac) const {
    return math::elementwise_mul(scale_, point_frac);
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_minimum_image(
      const Vector3T& fractional_vec) const {
    OurVector3 min_image{
      fractional_vec[0] - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(fractional_vec[0]))),
      fractional_vec[1] - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(fractional_vec[1]))),
      fractional_vec[2]};
    return min_image;
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_wrap_to_unit_cell(
      const Vector3T& fractional_vec) const {
    OurVector3 wrapped{
      impl::safe_unit_mod1<Integer>(fractional_vec[0]),
      impl::safe_unit_mod1<Integer>(fractional_vec[1]),
      fractional_vec[2]};
    return wrapped;
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  template <ValidPointType PointT1, ValidPointType PointT2>
    requires std::is_same_v<typename PointT1::scalar_t, Scalar> && std::is_same_v<typename PointT2::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint sep(const PointT1& point1, const PointT2& point2) const {
    // Assumes linearity of to_fractional
    return from_fractional(frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& point) const {
    return from_fractional(frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// TODO(palmerb4): I don't think this should be offered.
  // /// \brief Direct lattice vectors (return as the columns of a matrix)
  // KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
  //   return h_;
  // }

  /// \brief Shift a point by a given number of lattice images in each direction
  template <ValidPointType PointT, typename Integer>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& point,
                                                        const math::Vector3<Integer>& num_images) const {
    return translate(point, from_fractional(num_images.template cast<Scalar>()));
  }

 private:
  OurVector3 scale_;      ///< Unit cell scaling factors
  OurVector3 inv_scale_;  ///< Inverse of the scaling factors
};  // PeriodicMetricXY

template <typename Scalar>
class PeriodicMetricYZ {
 public:
  /// \brief Type aliases
  using scalar_t = Scalar;
  using OurVector3 = math::Vector3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicMetricYZ() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicMetricYZ(const double width_y, const double width_z) : scale_{1.0, width_y, width_z}, inv_scale_{1.0, 1.0 / width_y, 1.0 / width_z} {
    MUNDY_THROW_ASSERT(width_y > 0 && width_z > 0, std::invalid_argument, "Cell dimensions must be positive");
  }

  /// \brief Get if the given dimension is periodic
  template<unsigned dimension>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return dimension == 1 || dimension == 2;
  }

  /// \brief Get if the given dimension is periodic
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned dimension) const {
    return dimension == 1 || dimension == 2;
  }

  /// \brief Get the number of periodic dimensions
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 2;
  }

  /// \brief Map a point into fractional coordinates
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& point) const {
    return math::elementwise_mul(inv_scale_, point);
  }

  /// \brief Map a point from fractional coordinates to real space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& point_frac) const {
    return math::elementwise_mul(scale_, point_frac);
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_minimum_image(
      const Vector3T& fractional_vec) const {
    OurVector3 min_image{
      fractional_vec[0],
      fractional_vec[1] - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(fractional_vec[1]))),
      fractional_vec[2] - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(fractional_vec[2])))};
    return min_image;
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_wrap_to_unit_cell(
      const Vector3T& fractional_vec) const {
    OurVector3 wrapped{
      fractional_vec[0],
      impl::safe_unit_mod1<Integer>(fractional_vec[1]),
     impl::safe_unit_mod1<Integer>(fractional_vec[2])};
    return wrapped;
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  template <ValidPointType PointT1, ValidPointType PointT2>
    requires std::is_same_v<typename PointT1::scalar_t, Scalar> && std::is_same_v<typename PointT2::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint sep(const PointT1& point1, const PointT2& point2) const {
    // Assumes linearity of to_fractional
    return from_fractional(frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& point) const {
    return from_fractional(frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// TODO(palmerb4): I don't think this should be offered.
  // /// \brief Direct lattice vectors (return as the columns of a matrix)
  // KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
  //   return h_;
  // }

  /// \brief Shift a point by a given number of lattice images in each direction
  template <ValidPointType PointT, typename Integer>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& point,
                                                        const math::Vector3<Integer>& num_images) const {
    return translate(point, from_fractional(num_images.template cast<Scalar>()));
  }

 private:
  OurVector3 scale_;      ///< Unit cell scaling factors
  OurVector3 inv_scale_;  ///< Inverse of the scaling factors
};  // PeriodicMetricYZ

template <typename Scalar>
class PeriodicScaledMetric {
 public:
  /// \brief Type aliases
  using OurVector3 = math::Vector3<Scalar>;
  using OurMatrix3 = math::Matrix3<Scalar>;
  using OurPoint = Point<Scalar>;

  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicScaledMetric() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicScaledMetric(const OurVector3& cell_size)
      : scale_(cell_size), scale_inv_(Scalar(1.0) / scale_[0], Scalar(1.0) / scale_[1], Scalar(1.0) / scale_[2]) {
  }

  /// \brief Set the cell size
  KOKKOS_INLINE_FUNCTION
  void set_cell_size(const OurVector3& cell_size) {
    scale_ = cell_size;
    scale_inv_.set(Scalar(1.0) / scale_[0], Scalar(1.0) / scale_[1], Scalar(1.0) / scale_[2]);
  }

  /// \brief Get if the given dimension is periodic
  template<unsigned dimension>
  KOKKOS_INLINE_FUNCTION static constexpr bool is_periodic() {
    return true;
  }

  /// \brief Get if the given dimension is periodic
  KOKKOS_INLINE_FUNCTION constexpr bool is_periodic(unsigned /*dimension*/) const {
    return true;
  }

  /// \brief Get the number of periodic dimensions
  KOKKOS_INLINE_FUNCTION constexpr unsigned num_periodic_dimensions() const {
    return 3;
  }

  /// \brief Map a point into fractional coordinates
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint to_fractional(const PointT& point) const {
    return math::elementwise_mul(scale_inv_, point);
  }

  /// \brief Map a point from fractional coordinates to real space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint from_fractional(const PointT& point_frac) const {
    return math::elementwise_mul(scale_, point_frac);
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_minimum_image(
      const Vector3T& fractional_vec) const {
    return apply([](Scalar x) { return x - static_cast<Scalar>(static_cast<Integer>(Kokkos::round(x))); },
                 fractional_vec);
  }

  template <typename Integer, math::ValidVector3Type Vector3T>
  KOKKOS_INLINE_FUNCTION constexpr math::Vector3<typename Vector3T::scalar_t> frac_wrap_to_unit_cell(
      const Vector3T& fractional_vec) const {
    return apply([](Scalar x) { return impl::safe_unit_mod1<Integer>(x); }, fractional_vec);
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  template <ValidPointType PointT1, ValidPointType PointT2>
    requires std::is_same_v<typename PointT1::scalar_t, Scalar> && std::is_same_v<typename PointT2::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint sep(const PointT1& point1, const PointT2& point2) const {
    return from_fractional(frac_minimum_image<int64_t>(to_fractional(point2 - point1)));
  }

  /// \brief Wrap a point into the periodic space
  template <ValidPointType PointT>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint wrap(const PointT& point) const {
    return from_fractional(frac_wrap_to_unit_cell<int64_t>(to_fractional(point)));
  }

  /// \brief Direct lattice vectors (return as the columns of a matrix)
  KOKKOS_INLINE_FUNCTION constexpr OurMatrix3 direct_lattice_vectors() const {
    return OurMatrix3::diagonal(scale_);
  }

  /// \brief Shift a point by a given number of lattice images in each direction
  template <ValidPointType PointT, typename Integer>
    requires std::is_same_v<typename PointT::scalar_t, Scalar>
  KOKKOS_INLINE_FUNCTION constexpr OurPoint shift_image(const PointT& point,
                                                        const math::Vector3<Integer>& num_images) const {
    return translate(point, math::elementwise_mul(scale_, num_images));
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

template <ValidPointType PointT>
KOKKOS_INLINE_FUNCTION Point<typename PointT::scalar_t> reference_point(const PointT& point) {
  return point;
}

template <ValidLineType LineT>
KOKKOS_INLINE_FUNCTION Point<typename LineT::scalar_t> reference_point(const LineT& line) {
  return line.center();
}

template <ValidLineSegmentType LineSegmentT>
KOKKOS_INLINE_FUNCTION Point<typename LineSegmentT::scalar_t> reference_point(const LineSegmentT& line_segment) {
  return line_segment.start();
}

template <ValidCircle3DType Circle3DT>
KOKKOS_INLINE_FUNCTION Point<typename Circle3DT::scalar_t> reference_point(const Circle3DT& circle) {
  return circle.center();
}

template <ValidVSegmentType VSegmentT>
KOKKOS_INLINE_FUNCTION Point<typename VSegmentT::scalar_t> reference_point(const VSegmentT& v_segment) {
  return v_segment.start();
}

template <ValidAABBType AABBT>
KOKKOS_INLINE_FUNCTION Point<typename AABBT::scalar_t> reference_point(const AABBT& aabb) {
  return aabb.min_corner();
}

template <ValidSphereType SphereT>
KOKKOS_INLINE_FUNCTION Point<typename SphereT::scalar_t> reference_point(const SphereT& sphere) {
  return sphere.center();
}

template <ValidSpherocylinderType SpherocylinderT>
KOKKOS_INLINE_FUNCTION Point<typename SpherocylinderT::scalar_t> reference_point(
    const SpherocylinderT& spherocylinder) {
  return spherocylinder.center();
}

template <ValidSpherocylinderSegmentType SpherocylinderSegmentT>
KOKKOS_INLINE_FUNCTION Point<typename SpherocylinderSegmentT::scalar_t> reference_point(
    const SpherocylinderSegmentT& spherocylinder_segment) {
  return spherocylinder_segment.start();
}

template <ValidRingType RingT>
KOKKOS_INLINE_FUNCTION Point<typename RingT::scalar_t> reference_point(const RingT& ring) {
  return ring.center();
}

template <ValidEllipsoidType EllipsoidT>
KOKKOS_INLINE_FUNCTION Point<typename EllipsoidT::scalar_t> reference_point(const EllipsoidT& ellipsoid) {
  return ellipsoid.center();
}
//@}

//! \name Shift image functions to take an object and shift it by a lattice vector
//@{

/// \brief Shift an object by a lattice vector (returns a new object, owning its own memory)
template <typename Integer, typename Object, typename Metric>
KOKKOS_INLINE_FUNCTION auto shift_image(const Object& obj,                             //
                                        const math::Vector3<Integer>& lattice_vector,  //
                                        const Metric& metric) {
  auto shift_disp = metric.shift_image(reference_point(obj), lattice_vector) - reference_point(obj);
  return translate(obj, shift_disp);
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
 -wrap_points: wraps each point of the object independently using the metric's wrap function.
 -wrap_rigid: rigid translation so the internal reference point ends up wrapped into the primary cell; all other
points move with it.
*/

/// \brief Rigidly wrap a point into a given space (based on its center)
template <ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Point<typename PointT::scalar_t> wrap_rigid(const PointT& point, const Metric& metric) {
  return metric.wrap(point);
}

/// \brief Rigidly wrap a point into a given space (based on its center | inplace)
template <ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(PointT& point, const Metric& metric) {
  metric.wrap_inplace(point);
}

/// \brief Rigidly wrap a line into a given space (invalid operation)
template <ValidLineType LineT, typename Metric>
KOKKOS_INLINE_FUNCTION Line<typename LineT::scalar_t> wrap_rigid(const LineT& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
  return line;
}

/// brief Rigidly wrap a line into a given space (invalid operation | inplace)
template <ValidLineType LineT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(LineT& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
}

/// \brief Rigidly wrap a line segment into a given space (based on its start point)
template <ValidLineSegmentType LineSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION LineSegment<typename LineSegmentT::scalar_t> wrap_rigid(const LineSegmentT& line_segment,
                                                                               const Metric& metric) {
  auto wrapped_start = metric.wrap(line_segment.start());
  auto wrapped_end = line_segment.end() + (wrapped_start - line_segment.start());
  return LineSegment<typename LineSegmentT::scalar_t>(wrapped_start, wrapped_end);
}

/// \brief Rigidly wrap a line segment into a given space (based on its start point | inplace)
template <ValidLineSegmentType LineSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(LineSegmentT& line_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(line_segment.start());
  auto disp = wrapped_start - line_segment.start();
  line_segment.set_start(wrapped_start);
  line_segment.set_end(line_segment.end() + disp);
}

/// \brief Rigidly wrap a circle3D into a given space (based on its center point)
template <ValidCircle3DType Circle3DT, typename Metric>
KOKKOS_INLINE_FUNCTION Circle3D<typename Circle3DT::scalar_t> wrap_rigid(const Circle3DT& circle,
                                                                         const Metric& metric) {
  return Circle3D<typename Circle3DT::scalar_t>(metric.wrap(circle.center()), circle.orientation(), circle.radius());
}

/// \brief Rigidly wrap a circle3D into a given space (based on its center point | inplace)
template <ValidCircle3DType Circle3DT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(Circle3DT& circle, const Metric& metric) {
  metric.wrap_inplace(circle.center());
}

/// \brief Rigidly wrap a v-segment into a given space (based on its start point)
template <ValidVSegmentType VSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION VSegment<typename VSegmentT::scalar_t> wrap_rigid(const VSegmentT& v_segment,
                                                                         const Metric& metric) {
  auto wrapped_start = metric.wrap(v_segment.start());
  auto disp = wrapped_start - v_segment.start();
  auto wrapped_middle = v_segment.middle() + disp;
  auto wrapped_end = v_segment.end() + disp;
  return VSegment<typename VSegmentT::scalar_t>(wrapped_start, wrapped_middle, wrapped_end);
}

/// \brief Rigidly wrap a v-segment into a given space (based on its start point | inplace)
template <ValidVSegmentType VSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(VSegmentT& v_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(v_segment.start());
  auto disp = wrapped_start - v_segment.start();
  v_segment.set_start(wrapped_start);
  v_segment.set_middle(v_segment.middle() + disp);
  v_segment.set_end(v_segment.end() + disp);
}

/// \brief Rigidly wrap an AABB into a given space (based on its min_corner)
template <ValidAABBType AABBT, typename Metric>
KOKKOS_INLINE_FUNCTION AABB<typename AABBT::scalar_t> wrap_rigid(const AABBT& aabb, const Metric& metric) {
  auto wrapped_min = metric.wrap(aabb.min_corner());
  auto wrapped_max = aabb.max_corner() + (wrapped_min - aabb.min_corner());
  return AABB<typename AABBT::scalar_t>(wrapped_min, wrapped_max);
}

/// \brief Rigidly wrap an AABB into a given space (based on its min_corner | inplace)
template <ValidAABBType AABBT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(AABBT& aabb, const Metric& metric) {
  auto wrapped_min = metric.wrap(aabb.min_corner());
  auto disp = wrapped_min - aabb.min_corner();
  aabb.set_min_corner(wrapped_min);
  aabb.set_max_corner(aabb.max_corner() + disp);
}

/// \brief Rigidly wrap a sphere into a given space (based on its center point)
template <ValidSphereType SphereT, typename Metric>
KOKKOS_INLINE_FUNCTION Sphere<typename SphereT::scalar_t> wrap_rigid(const SphereT& sphere, const Metric& metric) {
  return Sphere<typename SphereT::scalar_t>(metric.wrap(sphere.center()), sphere.radius());
}

/// \brief Rigidly wrap a sphere into a given space (based on its center point | inplace)
template <ValidSphereType SphereT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(SphereT& sphere, const Metric& metric) {
  metric.wrap_inplace(sphere.center());
}

/// \brief Rigidly wrap a spherocylinder into a given space (based on its center point)
template <ValidSpherocylinderType SpherocylinderT, typename Metric>
KOKKOS_INLINE_FUNCTION Spherocylinder<typename SpherocylinderT::scalar_t> wrap_rigid(
    const SpherocylinderT& spherocylinder, const Metric& metric) {
  return Spherocylinder<typename SpherocylinderT::scalar_t>(metric.wrap(spherocylinder.center()),
                                                            spherocylinder.orientation(), spherocylinder.radius(),
                                                            spherocylinder.length());
}

/// \brief Rigidly wrap a spherocylinder into a given space (based on its center point | inplace)
template <ValidSpherocylinderType SpherocylinderT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(SpherocylinderT& spherocylinder, const Metric& metric) {
  metric.wrap_inplace(spherocylinder.center());
}

/// \brief Rigidly wrap a spherocylinder segment into a given space (based on its start point)
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<typename SpherocylinderSegmentT::scalar_t> wrap_rigid(
    const SpherocylinderSegmentT& spherocylinder_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(spherocylinder_segment.start());
  auto wrapped_end = spherocylinder_segment.end() + (wrapped_start - spherocylinder_segment.start());
  return SpherocylinderSegment<typename SpherocylinderSegmentT::scalar_t>(
      wrapped_start, wrapped_end, spherocylinder_segment.orientation(), spherocylinder_segment.radius(),
      spherocylinder_segment.length());
}

/// \brief Rigidly wrap a spherocylinder segment into a given space (based on its start point | inplace)
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(SpherocylinderSegmentT& spherocylinder_segment, const Metric& metric) {
  auto wrapped_start = metric.wrap(spherocylinder_segment.start());
  auto disp = wrapped_start - spherocylinder_segment.start();
  spherocylinder_segment.set_start(wrapped_start);
  spherocylinder_segment.set_end(spherocylinder_segment.end() + disp);
}

/// \brief Rigidly wrap a ring into a given space (based on its center point)
template <ValidRingType RingT, typename Metric>
KOKKOS_INLINE_FUNCTION Ring<typename RingT::scalar_t> wrap_rigid(const RingT& ring, const Metric& metric) {
  return Ring<typename RingT::scalar_t>(metric.wrap(ring.center()), ring.orientation(), ring.major_radius(),
                                        ring.minor_radius());
}

/// \brief Rigidly wrap a ring into a given space (based on its center point | inplace)
template <ValidRingType RingT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(RingT& ring, const Metric& metric) {
  metric.wrap_inplace(ring.center());
}

/// \brief Rigidly wrap an ellipsoid into a given space (based on its center point)
template <ValidEllipsoidType EllipsoidT, typename Metric>
KOKKOS_INLINE_FUNCTION Ellipsoid<typename EllipsoidT::scalar_t> wrap_rigid(const EllipsoidT& ellipsoid,
                                                                           const Metric& metric) {
  return Ellipsoid<typename EllipsoidT::scalar_t>(metric.wrap(ellipsoid.center()), ellipsoid.orientation(),
                                                  ellipsoid.radii());
}

/// \brief Rigidly wrap an ellipsoid into a given space (based on its center point | inplace)
template <ValidEllipsoidType EllipsoidT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_rigid_inplace(EllipsoidT& ellipsoid, const Metric& metric) {
  metric.wrap_inplace(ellipsoid.center());
}
//@}

//! \name Wrap all points of an object into a periodic space
//@{

/// \brief Wrap all points of a point into a periodic space
template <ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Point<typename PointT::scalar_t> wrap_points(const PointT& point, const Metric& metric) {
  return metric.wrap(point);
}

/// \brief Wrap all points of a point into a periodic space (inplace)
template <ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(PointT& point, const Metric& metric) {
  metric.wrap_inplace(point);
}

/// \brief Wrap all points of a line into a periodic space (invalid operation)
template <ValidLineType LineT, typename Metric>
KOKKOS_INLINE_FUNCTION Line<typename LineT::scalar_t> wrap_points(const LineT& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
  return line;
}

/// \brief Wrap all points of a line into a periodic space (invalid operation | inplace)
template <ValidLineType LineT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(LineT& line, const Metric& metric) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: wrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
}

/// \brief Wrap all points of a line segment into a periodic space
template <ValidLineSegmentType LineSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION LineSegment<typename LineSegmentT::scalar_t> wrap_points(const LineSegmentT& line_segment,
                                                                                const Metric& metric) {
  return LineSegment<typename LineSegmentT::scalar_t>(metric.wrap(line_segment.start()),
                                                      metric.wrap(line_segment.end()));
}

/// \brief Wrap all points of a line segment into a periodic space (inplace)
template <ValidLineSegmentType LineSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(LineSegmentT& line_segment, const Metric& metric) {
  metric.wrap_inplace(line_segment.start());
  metric.wrap_inplace(line_segment.end());
}

/// \brief Wrap all points of a circle3D into a periodic space
template <ValidCircle3DType Circle3DT, typename Metric>
KOKKOS_INLINE_FUNCTION Circle3D<typename Circle3DT::scalar_t> wrap_points(const Circle3DT& circle,
                                                                          const Metric& metric) {
  return Circle3D<typename Circle3DT::scalar_t>(metric.wrap(circle.center()), circle.orientation(), circle.radius());
}

/// \brief Wrap all points of a circle3D into a periodic space (inplace)
template <ValidCircle3DType Circle3DT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(Circle3DT& circle, const Metric& metric) {
  metric.wrap_inplace(circle.center());
}

/// \brief Wrap all points of a v-segment into a periodic space
template <ValidVSegmentType VSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION VSegment<typename VSegmentT::scalar_t> wrap_points(const VSegmentT& v_segment,
                                                                          const Metric& metric) {
  return VSegment<typename VSegmentT::scalar_t>(metric.wrap(v_segment.start()), metric.wrap(v_segment.middle()),
                                                metric.wrap(v_segment.end()));
}

/// \brief Wrap all points of a v-segment into a periodic space (inplace)
template <ValidVSegmentType VSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(VSegmentT& v_segment, const Metric& metric) {
  metric.wrap_inplace(v_segment.start());
  metric.wrap_inplace(v_segment.middle());
  metric.wrap_inplace(v_segment.end());
}

/// \brief Wrap all points of an AABB into a periodic space
template <ValidAABBType AABBT, typename Metric>
KOKKOS_INLINE_FUNCTION AABB<typename AABBT::scalar_t> wrap_points(const AABBT& aabb, const Metric& metric) {
  return AABB<typename AABBT::scalar_t>(metric.wrap(aabb.min_corner()), metric.wrap(aabb.max_corner()));
}

/// \brief Wrap all points of an AABB into a periodic space (inplace)
template <ValidAABBType AABBT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(AABBT& aabb, const Metric& metric) {
  metric.wrap_inplace(aabb.min_corner());
  metric.wrap_inplace(aabb.max_corner());
}

/// \brief Wrap all points of a sphere into a periodic space
template <ValidSphereType SphereT, typename Metric>
KOKKOS_INLINE_FUNCTION Sphere<typename SphereT::scalar_t> wrap_points(const SphereT& sphere, const Metric& metric) {
  return Sphere<typename SphereT::scalar_t>(metric.wrap(sphere.center()), sphere.radius());
}

/// \brief Wrap all points of a sphere into a periodic space (inplace)
template <ValidSphereType SphereT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(SphereT& sphere, const Metric& metric) {
  metric.wrap_inplace(sphere.center());
}

/// \brief Wrap all points of a spherocylinder into a periodic space
template <ValidSpherocylinderType SpherocylinderT, typename Metric>
KOKKOS_INLINE_FUNCTION Spherocylinder<typename SpherocylinderT::scalar_t> wrap_points(
    const SpherocylinderT& spherocylinder, const Metric& metric) {
  return Spherocylinder<typename SpherocylinderT::scalar_t>(metric.wrap(spherocylinder.center()),
                                                            spherocylinder.orientation(), spherocylinder.radius(),
                                                            spherocylinder.length());
}

/// \brief Wrap all points of a spherocylinder into a periodic space (inplace)
template <ValidSpherocylinderType SpherocylinderT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(SpherocylinderT& spherocylinder, const Metric& metric) {
  metric.wrap_inplace(spherocylinder.center());
}

/// \brief Wrap all points of a spherocylinder segment into a periodic space
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<typename SpherocylinderSegmentT::scalar_t> wrap_points(
    const SpherocylinderSegmentT& spherocylinder_segment, const Metric& metric) {
  return SpherocylinderSegment<typename SpherocylinderSegmentT::scalar_t>(
      metric.wrap(spherocylinder_segment.start()), metric.wrap(spherocylinder_segment.end()),
      spherocylinder_segment.orientation(), spherocylinder_segment.radius(), spherocylinder_segment.length());
}

/// \brief Wrap all points of a spherocylinder segment into a periodic space (inplace)
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(SpherocylinderSegmentT& spherocylinder_segment, const Metric& metric) {
  metric.wrap_inplace(spherocylinder_segment.start());
  metric.wrap_inplace(spherocylinder_segment.end());
}

/// \brief Wrap all points of a ring into a periodic space
template <ValidRingType RingT, typename Metric>
KOKKOS_INLINE_FUNCTION Ring<typename RingT::scalar_t> wrap_points(const RingT& ring, const Metric& metric) {
  return Ring<typename RingT::scalar_t>(metric.wrap(ring.center()), ring.orientation(), ring.major_radius(),
                                        ring.minor_radius());
}

/// \brief Wrap all points of a ring into a periodic space (inplace)
template <ValidRingType RingT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(RingT& ring, const Metric& metric) {
  metric.wrap_inplace(ring.center());
}

/// \brief Wrap all points of an ellipsoid into a periodic space
template <ValidEllipsoidType EllipsoidT, typename Metric>
KOKKOS_INLINE_FUNCTION Ellipsoid<typename EllipsoidT::scalar_t> wrap_points(const EllipsoidT& ellipsoid,
                                                                            const Metric& metric) {
  return Ellipsoid<typename EllipsoidT::scalar_t>(metric.wrap(ellipsoid.center()), ellipsoid.orientation(),
                                                  ellipsoid.radii());
}

/// \brief Wrap all points of an ellipsoid into a periodic space (inplace)
template <ValidEllipsoidType EllipsoidT, typename Metric>
KOKKOS_INLINE_FUNCTION void wrap_points_inplace(EllipsoidT& ellipsoid, const Metric& metric) {
  metric.wrap_inplace(ellipsoid.center());
}
//@}

//! \name Unwrap all points of an object to be within one image of the reference point
//@{

/*
We must emphasize that unwrap_points_to_ref is only the inverse of wrap_points if the reference point already lies
within the primary cell. Otherwise, they are the same up to wrap_rigid applied to the original shape. That is:
  wrap_rigid(s) == unwrap_points_to_ref(wrap_points(s, metric)) == unwrap_points_to_ref(s, s.ref()).

Note. The following are valid even if the ref point is one of the given points.
*/

/// \brief Unwrap all points of a point to be within one image of the reference point
template <ValidPointType PointT1, ValidPointType PointT2, typename Metric>
KOKKOS_INLINE_FUNCTION Point<typename PointT1::scalar_t> unwrap_points_to_ref(const PointT1& point,
                                                                              const Metric& metric,
                                                                              const PointT2& ref_point) {
  return ref_point + metric.sep(ref_point, point);
}

/// \brief Unwrap all points of a point to be within one image of the reference point (inplace)
template <ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(PointT& point, const Metric& metric, const PointT& ref_point) {
  point = ref_point + metric.sep(ref_point, point);
}

/// \brief Unwrap all points of a line to be within one image of the reference point
template <ValidLineType LineT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Line<typename PointT::scalar_t> unwrap_points_to_ref(const LineT& line, const Metric& metric,
                                                                            const PointT& ref_point) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: unwrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
  return line;
}

/// \brief Unwrap all points of a line to be within one image of the reference point (inplace)
template <ValidLineType LineT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(LineT& line, const Metric& metric, const PointT& ref_point) {
  MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                      "Not implemented error: unwrapping a line into a periodic space does not make sense, as it is "
                      "infinite in length and could fill the space.")
}

/// \brief Unwrap all points of a line segment to be within one image of the reference point
template <ValidLineSegmentType LineSegmentT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION LineSegment<typename PointT::scalar_t> unwrap_points_to_ref(const LineSegmentT& line_segment,
                                                                                   const Metric& metric,
                                                                                   const PointT& ref_point) {
  const auto s_start = metric.to_fractional(line_segment.start());
  const auto s_end = metric.to_fractional(line_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = metric.template frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = metric.template frac_minimum_image<int64_t>(s_end - sr);

  return LineSegment<typename PointT::scalar_t>(metric.from_fractional(sr + d_start),
                                                metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a line segment to be within one image of the reference point
/// (inplace)
template <ValidLineSegmentType LineSegmentT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(LineSegmentT& line_segment, const Metric& metric,
                                                         const PointT& ref_point) {
  const auto s_start = metric.to_fractional(line_segment.start());
  const auto s_end = metric.to_fractional(line_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = metric.template frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = metric.template frac_minimum_image<int64_t>(s_end - sr);

  line_segment.set_start(metric.from_fractional(sr + d_start));
  line_segment.set_end(metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a circle3D to be within one image of the reference point
template <ValidCircle3DType Circle3DT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Circle3D<typename PointT::scalar_t> unwrap_points_to_ref(const Circle3DT& circle,
                                                                                const Metric& metric,
                                                                                const PointT& ref_point) {
  auto new_center = unwrap_points_to_ref(circle.center(), metric, ref_point);
  return Circle3D<typename PointT::scalar_t>(new_center, circle.orientation(), circle.radius());
}

/// \brief Unwrap all points of a circle3D to be within one image of the reference point (inplace)
template <ValidCircle3DType Circle3DT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(Circle3DT& circle, const Metric& metric,
                                                         const PointT& ref_point) {
  unwrap_points_to_ref_inplace(circle.center(), metric, ref_point);
}

/// \brief Unwrap all points of a v-segment to be within one image of the reference point
template <ValidVSegmentType VSegmentT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION VSegment<typename PointT::scalar_t> unwrap_points_to_ref(const VSegmentT& v_segment,
                                                                                const Metric& metric,
                                                                                const PointT& ref_point) {
  const auto s_start = metric.to_fractional(v_segment.start());
  const auto s_middle = metric.to_fractional(v_segment.middle());
  const auto s_end = metric.to_fractional(v_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = metric.template frac_minimum_image<int64_t>(s_start - sr);
  const auto d_middle = metric.template frac_minimum_image<int64_t>(s_middle - sr);
  const auto d_end = metric.template frac_minimum_image<int64_t>(s_end - sr);

  return VSegment<typename PointT::scalar_t>(metric.from_fractional(sr + d_start),
                                             metric.from_fractional(sr + d_middle), metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a v-segment to be within one image of the reference point (inplace)
template <ValidVSegmentType VSegmentT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(VSegmentT& v_segment, const Metric& metric,
                                                         const PointT& ref_point) {
  const auto s_start = metric.to_fractional(v_segment.start());
  const auto s_middle = metric.to_fractional(v_segment.middle());
  const auto s_end = metric.to_fractional(v_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = metric.template frac_minimum_image<int64_t>(s_start - sr);
  const auto d_middle = metric.template frac_minimum_image<int64_t>(s_middle - sr);
  const auto d_end = metric.template frac_minimum_image<int64_t>(s_end - sr);

  v_segment.set_start(metric.from_fractional(sr + d_start));
  v_segment.set_middle(metric.from_fractional(sr + d_middle));
  v_segment.set_end(metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of an AABB to be within one image of the reference point
template <ValidAABBType AABBT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION AABB<typename PointT::scalar_t> unwrap_points_to_ref(const AABBT& aabb, const Metric& metric,
                                                                            const PointT& ref_point) {
  const auto s_min = metric.to_fractional(aabb.min_corner());
  const auto s_max = metric.to_fractional(aabb.max_corner());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_min = metric.template frac_minimum_image<int64_t>(s_min - sr);
  const auto d_max = metric.template frac_minimum_image<int64_t>(s_max - sr);

  return AABB<typename PointT::scalar_t>(metric.from_fractional(sr + d_min), metric.from_fractional(sr + d_max));
}

/// \brief Unwrap all points of an AABB to be within one image of the reference point (inplace)
template <ValidAABBType AABBT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(AABBT& aabb, const Metric& metric, const PointT& ref_point) {
  const auto s_min = metric.to_fractional(aabb.min_corner());
  const auto s_max = metric.to_fractional(aabb.max_corner());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_min = metric.template frac_minimum_image<int64_t>(s_min - sr);
  const auto d_max = metric.template frac_minimum_image<int64_t>(s_max - sr);

  aabb.set_min_corner(metric.from_fractional(sr + d_min));
  aabb.set_max_corner(metric.from_fractional(sr + d_max));
}

/// \brief Unwrap all points of a sphere to be within one image of the reference point
template <ValidSphereType SphereT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Sphere<typename PointT::scalar_t> unwrap_points_to_ref(const SphereT& sphere,
                                                                              const Metric& metric,
                                                                              const PointT& ref_point) {
  auto new_center = unwrap_points_to_ref(sphere.center(), metric, ref_point);
  return Sphere<typename PointT::scalar_t>(new_center, sphere.radius());
}

/// \brief Unwrap all points of a sphere to be within one image of the reference point (inplace)
template <ValidSphereType SphereT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(SphereT& sphere, const Metric& metric,
                                                         const PointT& ref_point) {
  unwrap_points_to_ref_inplace(sphere.center(), metric, ref_point);
}

/// \brief Unwrap all points of a spherocylinder to be within one image of the reference point
template <ValidSpherocylinderType SpherocylinderT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Spherocylinder<typename PointT::scalar_t> unwrap_points_to_ref(
    const SpherocylinderT& spherocylinder, const Metric& metric, const PointT& ref_point) {
  auto new_center = unwrap_points_to_ref(spherocylinder.center(), metric, ref_point);
  return Spherocylinder<typename PointT::scalar_t>(new_center, spherocylinder.orientation(), spherocylinder.radius(),
                                                   spherocylinder.length());
}

/// \brief Unwrap all points of a spherocylinder to be within one image of the reference point (inplace)
template <ValidSpherocylinderType SpherocylinderT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(SpherocylinderT& spherocylinder, const Metric& metric,
                                                         const PointT& ref_point) {
  unwrap_points_to_ref_inplace(spherocylinder.center(), metric, ref_point);
}

/// \brief Unwrap all points of a spherocylinder segment to be within one image of the reference
/// point
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION SpherocylinderSegment<typename PointT::scalar_t> unwrap_points_to_ref(
    const SpherocylinderSegmentT& spherocylinder_segment, const Metric& metric, const PointT& ref_point) {
  const auto s_start = metric.to_fractional(spherocylinder_segment.start());
  const auto s_end = metric.to_fractional(spherocylinder_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = metric.template frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = metric.template frac_minimum_image<int64_t>(s_end - sr);

  return SpherocylinderSegment<typename PointT::scalar_t>(metric.from_fractional(sr + d_start),  //
                                                          metric.from_fractional(sr + d_end),    //
                                                          spherocylinder_segment.orientation(),  //
                                                          spherocylinder_segment.radius(),       //
                                                          spherocylinder_segment.length());
}

/// \brief Unwrap all points of a spherocylinder segment to be within one image of the reference point (inplace)
template <ValidSpherocylinderSegmentType SpherocylinderSegmentT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(SpherocylinderSegmentT& spherocylinder_segment,
                                                         const Metric& metric, const PointT& ref_point) {
  const auto s_start = metric.to_fractional(spherocylinder_segment.start());
  const auto s_end = metric.to_fractional(spherocylinder_segment.end());
  const auto sr = metric.to_fractional(ref_point);

  const auto d_start = metric.template frac_minimum_image<int64_t>(s_start - sr);
  const auto d_end = metric.template frac_minimum_image<int64_t>(s_end - sr);

  spherocylinder_segment.set_start(metric.from_fractional(sr + d_start));
  spherocylinder_segment.set_end(metric.from_fractional(sr + d_end));
}

/// \brief Unwrap all points of a ring to be within one image of the reference point
template <ValidRingType RingT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Ring<typename PointT::scalar_t> unwrap_points_to_ref(const RingT& ring, const Metric& metric,
                                                                            const PointT& ref_point) {
  auto new_center = unwrap_points_to_ref(ring.center(), metric, ref_point);
  return Ring<typename PointT::scalar_t>(new_center, ring.orientation(), ring.major_radius(), ring.minor_radius());
}

/// \brief Unwrap all points of a ring to be within one image of the reference point (inplace)
template <ValidRingType RingT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(RingT& ring, const Metric& metric, const PointT& ref_point) {
  unwrap_points_to_ref_inplace(ring.center(), metric, ref_point);
}

/// \brief Unwrap all points of an ellipsoid to be within one image of the reference point
template <ValidEllipsoidType EllipsoidT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION Ellipsoid<typename PointT::scalar_t> unwrap_points_to_ref(const EllipsoidT& ellipsoid,
                                                                                 const Metric& metric,
                                                                                 const PointT& ref_point) {
  auto new_center = unwrap_points_to_ref(ellipsoid.center(), metric, ref_point);
  return Ellipsoid<typename PointT::scalar_t>(new_center, ellipsoid.orientation(), ellipsoid.radii());
}

/// \brief Unwrap all points of an ellipsoid to be within one image of the reference point (inplace)
template <ValidEllipsoidType EllipsoidT, ValidPointType PointT, typename Metric>
KOKKOS_INLINE_FUNCTION void unwrap_points_to_ref_inplace(EllipsoidT& ellipsoid, const Metric& metric,
                                                         const PointT& ref_point) {
  unwrap_points_to_ref_inplace(ellipsoid.center(), metric, ref_point);
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PERIODICITY_HPP_
