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

#ifndef MUNDY_MATH_IMPL_QUATERNIONIMPL_HPP_
#define MUNDY_MATH_IMPL_QUATERNIONIMPL_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <initializer_list>  // for std::initializer_list
#include <type_traits>       // for std::decay_t
#include <utility>

// Mundy
#include <mundy_math/Accessor.hpp>       // for mundy::ValidAccessor
#include <mundy_math/Array.hpp>          // for mundy::Array
#include <mundy_math/Matrix3.hpp>        // for mundy::Matrix3
#include <mundy_math/Tolerance.hpp>      // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>        // for mundy::Vector3
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_utils/requires.hpp>

namespace mundy {

template <typename T, ValidAccessor<T> Accessor = Array<T, 4>>
  MUNDY_REQUIRES(std::is_floating_point_v<T>)
class AQuaternion;

//! \name Forward declare AQuaternion functions that also require AQuaternion to be defined
//@{

/// \brief Get the inverse of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> inverse(const AQuaternion<T, Accessor>& quat);

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto norm(const AQuaternion<T, Accessor>& quat);
//@}

namespace impl {
//! \name Helper functions for generic quaternion operators applied to an abstract accessor.
//@{

/// \brief Deep copy assignment operator with (potentially) different accessor
/// \details Copies the data from the other quaternion to our data. This is only enabled if T is not const.
template <typename T, ValidAccessor<T> Accessor, typename U, ValidAccessor<U> OtherAccessor>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T> && std::is_convertible_v<U, T>)
KOKKOS_INLINE_FUNCTION constexpr void deep_copy_impl(AQuaternion<T, Accessor>& quat,
                                                     const AQuaternion<U, OtherAccessor>& other) {
  quat[0] = static_cast<T>(other[0]);
  quat[1] = static_cast<T>(other[1]);
  quat[2] = static_cast<T>(other[2]);
  quat[3] = static_cast<T>(other[3]);
}

/// \brief AQuaternion-quaternion addition
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
KOKKOS_INLINE_FUNCTION constexpr auto quat_quat_addition_impl(const AQuaternion<T, Accessor>& quat,
                                                              const AQuaternion<U, OtherAccessor>& other)
    -> AQuaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  AQuaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) + static_cast<CommonType>(other[0]);
  result[1] = static_cast<CommonType>(quat[1]) + static_cast<CommonType>(other[1]);
  result[2] = static_cast<CommonType>(quat[2]) + static_cast<CommonType>(other[2]);
  result[3] = static_cast<CommonType>(quat[3]) + static_cast<CommonType>(other[3]);
  return result;
}

/// \brief Self-quaternion addition
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>)
KOKKOS_INLINE_FUNCTION constexpr void self_quat_addition_impl(AQuaternion<T, Accessor>& quat,
                                                              const AQuaternion<U, OtherAccessor>& other) {
  quat[0] += static_cast<T>(other[0]);
  quat[1] += static_cast<T>(other[1]);
  quat[2] += static_cast<T>(other[2]);
  quat[3] += static_cast<T>(other[3]);
}

/// \brief AQuaternion-quaternion subtraction
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
KOKKOS_INLINE_FUNCTION constexpr auto quat_quat_subtraction_impl(const AQuaternion<T, Accessor>& quat,
                                                                 const AQuaternion<U, OtherAccessor>& other)
    -> AQuaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  AQuaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) - static_cast<CommonType>(other[0]);
  result[1] = static_cast<CommonType>(quat[1]) - static_cast<CommonType>(other[1]);
  result[2] = static_cast<CommonType>(quat[2]) - static_cast<CommonType>(other[2]);
  result[3] = static_cast<CommonType>(quat[3]) - static_cast<CommonType>(other[3]);
  return result;
}

/// \brief Self-quaternion subtraction
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>)
KOKKOS_INLINE_FUNCTION constexpr void self_quat_subtraction_impl(AQuaternion<T, Accessor>& quat,
                                                                 const AQuaternion<U, OtherAccessor>& other) {
  quat[0] -= static_cast<T>(other[0]);
  quat[1] -= static_cast<T>(other[1]);
  quat[2] -= static_cast<T>(other[2]);
  quat[3] -= static_cast<T>(other[3]);
}

/// \brief AQuaternion-quaternion multiplication
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
KOKKOS_INLINE_FUNCTION constexpr auto quat_quat_multiplication_impl(const AQuaternion<T, Accessor>& quat,
                                                                    const AQuaternion<U, OtherAccessor>& other)
    -> AQuaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  const CommonType lhs_w = static_cast<CommonType>(quat.w());
  const CommonType lhs_x = static_cast<CommonType>(quat.x());
  const CommonType lhs_y = static_cast<CommonType>(quat.y());
  const CommonType lhs_z = static_cast<CommonType>(quat.z());
  const CommonType rhs_w = static_cast<CommonType>(other.w());
  const CommonType rhs_x = static_cast<CommonType>(other.x());
  const CommonType rhs_y = static_cast<CommonType>(other.y());
  const CommonType rhs_z = static_cast<CommonType>(other.z());
  return AQuaternion<CommonType>(lhs_w * rhs_w - lhs_x * rhs_x - lhs_y * rhs_y - lhs_z * rhs_z,
                                 lhs_w * rhs_x + lhs_x * rhs_w + lhs_y * rhs_z - lhs_z * rhs_y,
                                 lhs_w * rhs_y - lhs_x * rhs_z + lhs_y * rhs_w + lhs_z * rhs_x,
                                 lhs_w * rhs_z + lhs_x * rhs_y - lhs_y * rhs_x + lhs_z * rhs_w);
}

/// \brief Self-quaternion multiplication
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>)
KOKKOS_INLINE_FUNCTION constexpr void self_quat_multiplication_impl(AQuaternion<T, Accessor>& quat,
                                                                    const AQuaternion<U, OtherAccessor>& other) {
  const T w = quat.w() * static_cast<T>(other.w()) - quat.x() * static_cast<T>(other.x()) -
              quat.y() * static_cast<T>(other.y()) - quat.z() * static_cast<T>(other.z());
  const T x = quat.w() * static_cast<T>(other.x()) + quat.x() * static_cast<T>(other.w()) +
              quat.y() * static_cast<T>(other.z()) - quat.z() * static_cast<T>(other.y());
  const T y = quat.w() * static_cast<T>(other.y()) - quat.x() * static_cast<T>(other.z()) +
              quat.y() * static_cast<T>(other.w()) + quat.z() * static_cast<T>(other.x());
  const T z = quat.w() * static_cast<T>(other.z()) + quat.x() * static_cast<T>(other.y()) -
              quat.y() * static_cast<T>(other.x()) + quat.z() * static_cast<T>(other.w());
  quat.set(w, x, y, z);
}

/// \brief AQuaternion-vector multiplication (same as R * v)
/// \param[in] vec The vector.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
KOKKOS_INLINE_FUNCTION constexpr auto quat_vec_multiplication_impl(const AQuaternion<T, Accessor>& quat,
                                                                   const AVector3<U, OtherAccessor>& vec)
    -> AVector3<std::common_type_t<T, U>> {
  // AQuaternion-vector multiplication consists of three parts:
  // 1. The vector is converted to a quaternion with a scalar component of 0
  // 2. The quaternion-quaternion multiplication is performed
  // 3. The quaternion is converted back to a vector
  const AQuaternion<U> vec_quat(U(0), vec[0], vec[1], vec[2]);
  const auto quat_inv = inverse(quat);
  const auto quat_result = quat * vec_quat * quat_inv;
  return AVector3<std::common_type_t<T, U>>(quat_result.x(), quat_result.y(), quat_result.z());
}

/// \brief AAVector-quaternion multiplication (same as v^T * R = transpose(R^T * v))
/// \param[in] vec The vector.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
KOKKOS_INLINE_FUNCTION constexpr auto vec_quat_multiplication_impl(const AVector3<T, Accessor>& vec,
                                                                   const AQuaternion<U, OtherAccessor>& quat)
    -> AVector3<std::common_type_t<T, U>> {
  // AAVector-quaternion multiplication consists of three parts:
  // 1. The vector is converted to a quaternion with a scalar component of 0
  // 2. The quaternion-quaternion multiplication is performed
  // 3. The quaternion is converted back to a vector
  const AQuaternion<T> vec_quat(T(0), vec[0], vec[1], vec[2]);
  const auto quat_inv = inverse(quat);
  const auto quat_result = quat_inv * vec_quat * quat;
  return AVector3<std::common_type_t<T, U>>(quat_result.x(), quat_result.y(), quat_result.z());
}

/// \param[in] other The other matrix.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
KOKKOS_INLINE_FUNCTION constexpr auto quat_mat_multiplication_impl(const AQuaternion<T, Accessor>& quat,
                                                                   const AMatrix3<U, OtherAccessor>& mat)
    -> AMatrix3<std::common_type_t<T, U>> {
  // AQuaternion-matrix multiplication consists of applying the quaternion to each column of the matrix
  using CommonType = std::common_type_t<T, U>;
  AMatrix3<CommonType> result;
  result.set_column(0, quat * mat.template view_column<0>());
  result.set_column(1, quat * mat.template view_column<1>());
  result.set_column(2, quat * mat.template view_column<2>());
  return result;
}

/// \brief AAMatrix-quaternion multiplication
/// \param[in] other The other matrix.
template <typename T, typename U, ValidAccessor<T> Accessor, ValidAccessor<U> OtherAccessor>
KOKKOS_INLINE_FUNCTION constexpr auto mat_quat_multiplication_impl(const AMatrix3<T, Accessor>& mat,
                                                                   const AQuaternion<U, OtherAccessor>& quat)
    -> AMatrix3<std::common_type_t<T, U>> {
  // AAMatrix-quaternion multiplication consists of applying the quaternion to each row of the matrix
  using CommonType = std::common_type_t<T, U>;
  AMatrix3<CommonType> result;
  result.set_row(0, mat.template view_row<0>() * quat);
  result.set_row(1, mat.template view_row<1>() * quat);
  result.set_row(2, mat.template view_row<2>() * quat);
  return result;
}

/// \brief AQuaternion-scalar multiplication
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto quat_scalar_multiplication_impl(const AQuaternion<T, Accessor>& quat,
                                                                      const U& scalar)
    -> AQuaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  AQuaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) * static_cast<CommonType>(scalar);
  result[1] = static_cast<CommonType>(quat[1]) * static_cast<CommonType>(scalar);
  result[2] = static_cast<CommonType>(quat[2]) * static_cast<CommonType>(scalar);
  result[3] = static_cast<CommonType>(quat[3]) * static_cast<CommonType>(scalar);
  return result;
}

/// \brief Self-scalar multiplication
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>)
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_multiplication_impl(AQuaternion<T, Accessor>& quat, const U& scalar) {
  quat[0] *= static_cast<T>(scalar);
  quat[1] *= static_cast<T>(scalar);
  quat[2] *= static_cast<T>(scalar);
  quat[3] *= static_cast<T>(scalar);
}

/// \brief AQuaternion-scalar division
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto quat_scalar_division_impl(const AQuaternion<T, Accessor>& quat, const U& scalar)
    -> AQuaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  AQuaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) / static_cast<CommonType>(scalar);
  result[1] = static_cast<CommonType>(quat[1]) / static_cast<CommonType>(scalar);
  result[2] = static_cast<CommonType>(quat[2]) / static_cast<CommonType>(scalar);
  result[3] = static_cast<CommonType>(quat[3]) / static_cast<CommonType>(scalar);
  return result;
}

/// \brief Self-scalar division
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>)
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_division_impl(AQuaternion<T, Accessor>& quat, const U& scalar) {
  quat[0] /= static_cast<T>(scalar);
  quat[1] /= static_cast<T>(scalar);
  quat[2] /= static_cast<T>(scalar);
  quat[3] /= static_cast<T>(scalar);
}
//@}
}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_QUATERNIONIMPL_HPP_
