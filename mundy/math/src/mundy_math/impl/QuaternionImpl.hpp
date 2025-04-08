// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
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
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Matrix3.hpp>       // for mundy::math::Matrix3
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace math {

template <typename T, ValidAccessor<T> Accessor = Array<T, 4>, typename OwnershipType = Ownership::Owns>
  requires std::is_floating_point_v<T>
class Quaternion;

//! \name Forward declare Quaternion functions that also require Quaternion to be defined
//@{

/// \brief Get the inverse of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr Quaternion<std::remove_const_t<T>> inverse(
    const Quaternion<T, Accessor, OwnershipType> &quat);

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto norm(const Quaternion<T, Accessor, OwnershipType> &quat);
//@}

namespace impl {
//! \name Helper functions for generic quaternion operators applied to an abstract accessor.
//@{

/// \brief Deep copy assignment operator with (potentially) different accessor
/// \details Copies the data from the other quaternion to our data. This is only enabled if T is not const.
template <size_t... Is, typename T, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<T> OtherAccessor,
          typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void deep_copy_impl(Quaternion<T, Accessor, OwnershipType> &quat,
                                                     const Quaternion<T, OtherAccessor, OtherOwnershipType> &other) {
  quat[0] = other[0];
  quat[1] = other[1];
  quat[2] = other[2];
  quat[3] = other[3];
}

/// \brief Quaternion-quaternion addition
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto quat_quat_addition_impl(
    const Quaternion<T, Accessor, OwnershipType> &quat, const Quaternion<U, OtherAccessor, OtherOwnershipType> &other)
    -> Quaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Quaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) + static_cast<CommonType>(other[0]);
  result[1] = static_cast<CommonType>(quat[1]) + static_cast<CommonType>(other[1]);
  result[2] = static_cast<CommonType>(quat[2]) + static_cast<CommonType>(other[2]);
  result[3] = static_cast<CommonType>(quat[3]) + static_cast<CommonType>(other[3]);
  return result;
}

/// \brief Self-quaternion addition
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void self_quat_addition_impl(
    Quaternion<T, Accessor, OwnershipType> &quat, const Quaternion<U, OtherAccessor, OtherOwnershipType> &other) {
  quat[0] += static_cast<T>(other[0]);
  quat[1] += static_cast<T>(other[1]);
  quat[2] += static_cast<T>(other[2]);
  quat[3] += static_cast<T>(other[3]);
}

/// \brief Quaternion-quaternion subtraction
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto quat_quat_subtraction_impl(
    const Quaternion<T, Accessor, OwnershipType> &quat, const Quaternion<U, OtherAccessor, OtherOwnershipType> &other)
    -> Quaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Quaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) - static_cast<CommonType>(other[0]);
  result[1] = static_cast<CommonType>(quat[1]) - static_cast<CommonType>(other[1]);
  result[2] = static_cast<CommonType>(quat[2]) - static_cast<CommonType>(other[2]);
  result[3] = static_cast<CommonType>(quat[3]) - static_cast<CommonType>(other[3]);
  return result;
}

/// \brief Self-quaternion subtraction
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void self_quat_subtraction_impl(
    Quaternion<T, Accessor, OwnershipType> &quat, const Quaternion<U, OtherAccessor, OtherOwnershipType> &other) {
  quat[0] -= static_cast<T>(other[0]);
  quat[1] -= static_cast<T>(other[1]);
  quat[2] -= static_cast<T>(other[2]);
  quat[3] -= static_cast<T>(other[3]);
}

/// \brief Quaternion-quaternion multiplication
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto quat_quat_multiplication_impl(
    const Quaternion<T, Accessor, OwnershipType> &quat, const Quaternion<U, OtherAccessor, OtherOwnershipType> &other)
    -> Quaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Quaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) * static_cast<CommonType>(other[0]) -
              static_cast<CommonType>(quat[1]) * static_cast<CommonType>(other[1]) -
              static_cast<CommonType>(quat[2]) * static_cast<CommonType>(other[2]) -
              static_cast<CommonType>(quat[3]) * static_cast<CommonType>(other[3]);
  result[1] = static_cast<CommonType>(quat[0]) * static_cast<CommonType>(other[1]) +
              static_cast<CommonType>(quat[1]) * static_cast<CommonType>(other[0]) +
              static_cast<CommonType>(quat[2]) * static_cast<CommonType>(other[3]) -
              static_cast<CommonType>(quat[3]) * static_cast<CommonType>(other[2]);
  result[2] = static_cast<CommonType>(quat[0]) * static_cast<CommonType>(other[2]) -
              static_cast<CommonType>(quat[1]) * static_cast<CommonType>(other[3]) +
              static_cast<CommonType>(quat[2]) * static_cast<CommonType>(other[0]) +
              static_cast<CommonType>(quat[3]) * static_cast<CommonType>(other[1]);
  result[3] = static_cast<CommonType>(quat[0]) * static_cast<CommonType>(other[3]) +
              static_cast<CommonType>(quat[1]) * static_cast<CommonType>(other[2]) -
              static_cast<CommonType>(quat[2]) * static_cast<CommonType>(other[1]) +
              static_cast<CommonType>(quat[3]) * static_cast<CommonType>(other[0]);
  return result;
}

/// \brief Self-quaternion multiplication
/// \param[in] other The other quaternion.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void self_quat_multiplication_impl(
    Quaternion<T, Accessor, OwnershipType> &quat, const Quaternion<U, OtherAccessor, OtherOwnershipType> &other) {
  const T w = quat[0] * static_cast<T>(other[0]) - quat[1] * static_cast<T>(other[1]) -
              quat[2] * static_cast<T>(other[2]) - quat[3] * static_cast<T>(other[3]);
  const T x = quat[0] * static_cast<T>(other[1]) + quat[1] * static_cast<T>(other[0]) +
              quat[2] * static_cast<T>(other[3]) - quat[3] * static_cast<T>(other[2]);
  const T y = quat[0] * static_cast<T>(other[2]) - quat[1] * static_cast<T>(other[3]) +
              quat[2] * static_cast<T>(other[0]) + quat[3] * static_cast<T>(other[1]);
  const T z = quat[0] * static_cast<T>(other[3]) + quat[1] * static_cast<T>(other[2]) -
              quat[2] * static_cast<T>(other[1]) + quat[3] * static_cast<T>(other[0]);
  quat[0] = w;
  quat[1] = x;
  quat[2] = y;
  quat[3] = z;
}

/// \brief Quaternion-vector multiplication (same as R * v)
/// \param[in] vec The vector.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto quat_vec_multiplication_impl(
    const Quaternion<T, Accessor, OwnershipType> &quat, const Vector3<U, OtherAccessor, OtherOwnershipType> &vec)
    -> Vector3<std::common_type_t<T, U>> {
  // Quaternion-vector multiplication consists of three parts:
  // 1. The vector is converted to a quaternion with a scalar component of 0
  // 2. The quaternion-quaternion multiplication is performed
  // 3. The quaternion is converted back to a vector
  const Quaternion<U> vec_quat(0.0, vec[0], vec[1], vec[2]);
  const auto quat_inv = inverse(quat);
  const auto quat_result = quat * vec_quat * quat_inv;
  return quat_result.vector();
}

/// \brief Vector-quaternion multiplication (same as v^T * R = transpose(R^T * v))
/// \param[in] vec The vector.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto vec_quat_multiplication_impl(
    const Vector3<T, Accessor, OwnershipType> &vec, const Quaternion<U, OtherAccessor, OtherOwnershipType> &quat)
    -> Vector3<std::common_type_t<T, U>> {
  // Vector-quaternion multiplication consists of three parts:
  // 1. The vector is converted to a quaternion with a scalar component of 0
  // 2. The quaternion-quaternion multiplication is performed
  // 3. The quaternion is converted back to a vector
  const Quaternion<T> vec_quat(0.0, vec[0], vec[1], vec[2]);
  const auto quat_inv = inverse(quat);
  const auto quat_result = quat_inv * vec_quat * quat;
  return quat_result.vector();
}

/// \brief Quaternion-matrix multiplication
/// \param[in] other The other matrix.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto quat_mat_multiplication_impl(
    const Quaternion<T, Accessor, OwnershipType> &quat, const Matrix3<U, OtherAccessor, OtherOwnershipType> &mat)
    -> Matrix3<std::common_type_t<T, U>> {
  // Quaternion-matrix multiplication consists of applying the quaternion to each column of the matrix
  using CommonType = std::common_type_t<T, U>;
  Matrix3<CommonType> result;
  result.set_column(0, quat * mat.template view_column<0>());
  result.set_column(1, quat * mat.template view_column<1>());
  result.set_column(2, quat * mat.template view_column<2>());
  return result;
}

/// \brief Matrix-quaternion multiplication
/// \param[in] other The other matrix.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType, ValidAccessor<U> OtherAccessor,
          typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto mat_quat_multiplication_impl(
    const Matrix3<T, Accessor, OwnershipType> &mat, const Quaternion<U, OtherAccessor, OtherOwnershipType> &quat)
    -> Matrix3<std::common_type_t<T, U>> {
  // Matrix-quaternion multiplication consists of applying the quaternion to each row of the matrix
  using CommonType = std::common_type_t<T, U>;
  Matrix3<CommonType> result;
  result.set_row(0, mat.template view_row<0>() * quat);
  result.set_row(1, mat.template view_row<1>() * quat);
  result.set_row(2, mat.template view_row<2>() * quat);
  return result;
}

/// \brief Quaternion-scalar multiplication
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto quat_scalar_multiplication_impl(
    const Quaternion<T, Accessor, OwnershipType> &quat, const U &scalar) -> Quaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Quaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) * static_cast<CommonType>(scalar);
  result[1] = static_cast<CommonType>(quat[1]) * static_cast<CommonType>(scalar);
  result[2] = static_cast<CommonType>(quat[2]) * static_cast<CommonType>(scalar);
  result[3] = static_cast<CommonType>(quat[3]) * static_cast<CommonType>(scalar);
  return result;
}

/// \brief Self-scalar multiplication
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_multiplication_impl(Quaternion<T, Accessor, OwnershipType> &quat,
                                                                      const U &scalar) {
  quat[0] *= static_cast<T>(scalar);
  quat[1] *= static_cast<T>(scalar);
  quat[2] *= static_cast<T>(scalar);
  quat[3] *= static_cast<T>(scalar);
}

/// \brief Quaternion-scalar division
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr auto quat_scalar_division_impl(const Quaternion<T, Accessor, OwnershipType> &quat,
                                                                const U &scalar)
    -> Quaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Quaternion<CommonType> result;
  result[0] = static_cast<CommonType>(quat[0]) / static_cast<CommonType>(scalar);
  result[1] = static_cast<CommonType>(quat[1]) / static_cast<CommonType>(scalar);
  result[2] = static_cast<CommonType>(quat[2]) / static_cast<CommonType>(scalar);
  result[3] = static_cast<CommonType>(quat[3]) / static_cast<CommonType>(scalar);
  return result;
}

/// \brief Self-scalar division
/// \param[in] scalar The scalar.
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_division_impl(Quaternion<T, Accessor, OwnershipType> &quat,
                                                                const U &scalar) {
  quat[0] /= static_cast<T>(scalar);
  quat[1] /= static_cast<T>(scalar);
  quat[2] /= static_cast<T>(scalar);
  quat[3] /= static_cast<T>(scalar);
}
//@}
}  // namespace impl

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_QUATERNIONIMPL_HPP_
