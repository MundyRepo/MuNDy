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

#ifndef MUNDY_MATH_IMPL_MATRIXIMPL_HPP_
#define MUNDY_MATH_IMPL_MATRIXIMPL_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>  // for std::decay_t
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>    // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>        // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>           // for mundy::math::Array
#include <mundy_math/MaskedView.hpp>      // for mundy::math::MaskedView
#include <mundy_math/ShiftedView.hpp>     // for mundy::math::ShiftedView
#include <mundy_math/StridedView.hpp>     // for mundy::math::StridedView
#include <mundy_math/Tolerance.hpp>       // for mundy::math::get_zero_tolerance
#include <mundy_math/TransposedView.hpp>  // for mundy::math::TransposedView
#include <mundy_math/Vector.hpp>          // for mundy::math::AVector

namespace mundy {

namespace math {

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor = Array<T, N * M>,
          typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
class AMatrix;

namespace impl {
//! \name Helper functions for generic matrix operators applied to an abstract accessor.
//@{

/// \brief Deep copy assignment operator with (potentially) different accessor
/// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void deep_copy_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                           const AMatrix<T, N, M, OtherAccessor, OtherOwnershipType>& other) {
  ((mat[Is] = other[Is]), ...);
}

/// \brief Move assignment operator with same accessor
/// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void move_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                      AMatrix<T, N, M, OtherAccessor, OtherOwnershipType>&& other) {
  ((mat[Is] = std::move(other[Is])), ...);
}

/// \brief Get a deep copy of a certain column of the matrix
/// \param[in] col The column index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION AVector<std::remove_const_t<T>, N> copy_column_impl(
    std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat, size_t col) {
  return {mat[col + Is * N]...};
}

/// \brief Get a deep copy of a certain row of the matrix
/// \param[in] row The row index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION AVector<std::remove_const_t<T>, M> copy_row_impl(
    std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat, size_t row) {
  return {mat[M * row + Is]...};
}

/// \brief Create a mask that excludes a specific row and column
template <size_t N, size_t M, size_t excluded_row, size_t excluded_col>
KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<bool, N * M> create_row_and_col_mask() {
  Kokkos::Array<bool, N * M> mask{};
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      mask[i * M + j] = (i != excluded_row) && (j != excluded_col);
    }
  }
  return mask;
}

/// \brief Cast (and copy) the matrix to a different type
template <typename U, size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto cast_impl(std::index_sequence<Is...>,
                                      const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  return AMatrix<U, N, M>{static_cast<U>(mat[Is])...};
}

/// \brief Set all elements of the matrix
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename... Args>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                     Args&&... args) {
  ((mat[Is] = std::forward<Args>(args)), ...);
}

/// \brief Set all elements of the matrix using an accessor
/// \param[in] accessor A valid accessor.
/// \note A AMatrix is also a valid accessor.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                     const auto& accessor) {
  ((mat[Is] = accessor[Is]), ...);
}

/// \brief Set a certain row of the matrix
/// \param[in] i The row index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename... Args>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_row_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                         const size_t& i, Args&&... args) {
  ((mat[Is + M * i] = std::forward<Args>(args)), ...);
}

/// \brief Set a certain row of the matrix
/// \param[in] i The row index.
/// \param[in] row The row vector.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_row_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                         const size_t& i, const AVector<T, M, OtherAccessor, OtherOwnershipType>& row) {
  ((mat[Is + M * i] = row[Is]), ...);
}

/// \brief Set a certain column of the matrix
/// \param[in] j The column index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename... Args>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_column_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                            const size_t& j, Args&&... args) {
  ((mat[j + Is * N] = std::forward<Args>(args)), ...);
}

/// \brief Set a certain column of the matrix
/// \param[in] j The column index.
/// \param[in] col The column vector.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_column_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                            const size_t& j,
                                            const AVector<T, N, OtherAccessor, OtherOwnershipType>& col) {
  ((mat[j + Is * N] = col[Is]), ...);
}

/// \brief Set all elements of the matrix to a single value
/// \param[in] value The value to set all elements to.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void fill_impl(std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                      const T& value) {
  ((mat[Is] = value), ...);
}

/// \brief Unary minus operator
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION AMatrix<T, N, M> unary_minus_impl(std::index_sequence<Is...>,
                                                         const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  AMatrix<T, N, M> result;
  ((result[Is] = -mat[Is]), ...);
  return result;
}

/// \brief AMatrix-matrix addition
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_matrix_addition_impl(
    std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
    const AMatrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) -> AMatrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) + static_cast<CommonType>(other[Is])), ...);
  return result;
}

/// \brief Self-matrix addition
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void self_matrix_addition_impl(std::index_sequence<Is...>,
                                                      AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                                      const AMatrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((mat[Is] += static_cast<T>(other[Is])), ...);
}

/// \brief AMatrix-matrix subtraction
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_matrix_subtraction_impl(
    std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
    const AMatrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) -> AMatrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) - static_cast<CommonType>(other[Is])), ...);
  return result;
}

/// \brief AMatrix-matrix subtraction
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void self_matrix_subtraction_impl(
    std::index_sequence<Is...>, AMatrix<T, N, M, Accessor, OwnershipType>& mat,
    const AMatrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((mat[Is] -= static_cast<T>(other[Is])), ...);
}

/// \brief AMatrix-scalar addition
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_addition_impl(std::index_sequence<Is...>,
                                                        const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                                        const U& scalar) -> AMatrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) + static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief AMatrix-scalar addition
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION void self_scalar_addition_impl(std::index_sequence<Is...>,
                                                      AMatrix<T, N, M, Accessor, OwnershipType>& mat, const U& scalar) {
  ((mat[Is] += static_cast<T>(scalar)), ...);
}

/// \brief AMatrix-scalar subtraction
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_subtraction_impl(std::index_sequence<Is...>,
                                                           const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                                           const U& scalar) -> AMatrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) - static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar subtraction
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION void self_scalar_subtraction_impl(std::index_sequence<Is...>,
                                                         AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                                         const U& scalar) {
  ((mat[Is] -= static_cast<T>(scalar)), ...);
}

/// \brief AMatrix-matrix multiplication
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_matrix_multiplication_impl(
    std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
    const AMatrix<U, OtherN, OtherM, OtherAccessor, OtherOwnershipType>& other)
    -> AMatrix<std::common_type_t<T, U>, OtherN, OtherM> {
  static_assert(M == OtherN,
                "AMatrix-matrix multiplication requires the number of columns in the first matrix to be equal to the "
                "number of rows in the second matrix.");

  // We need use a fold expressions to compute the dot product of each row of the first matrix
  // with each column of the second matrix via view symmantics.
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, OtherM> result;
  (...,
   (result(Is / M, Is % M) = mundy::math::dot(mat.template view_row<Is / M>(), other.template view_column<Is % M>())));
  return result;
}

/// \brief Self-matrix multiplication
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void self_matrix_multiplication_impl(
    std::index_sequence<Is...>, AMatrix<T, N, N, Accessor, OwnershipType>& mat,
    const AMatrix<U, N, N, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  // When writing to self, it's important to not write over data before we use it.
  AMatrix<T, N, N> tmp;
  (...,
   (tmp(Is / N, Is % N) = static_cast<T>(dot(mat.template view_row<Is / N>(), other.template view_column<Is % N>()))));
  mat = tmp;
}

/// \brief AMatrix-vector multiplication
/// \param[in] other The other vector.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_vector_multiplication_impl(
    std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
    const AVector<U, M, OtherAccessor, OtherOwnershipType>& other) -> AVector<std::common_type_t<T, U>, N> {
  using CommonType = std::common_type_t<T, U>;
  AVector<CommonType, N> result;
  (..., (result[Is] = dot(mat.template view_row<Is>(), other)));
  return result;
}

/// \brief AMatrix-scalar multiplication
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_multiplication_impl(std::index_sequence<Is...>,
                                                              const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                                              const U& scalar)
    -> AMatrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) * static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar multiplication
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires(HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>)
KOKKOS_INLINE_FUNCTION void self_scalar_multiplication_impl(std::index_sequence<Is...>,
                                                            AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                                            const U& scalar)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((mat[Is] *= static_cast<T>(scalar)), ...);
}

/// \brief AMatrix-scalar division
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_division_impl(std::index_sequence<Is...>,
                                                        const AMatrix<T, N, M, Accessor, OwnershipType>& mat,
                                                        const U& scalar) -> AMatrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) / static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar division
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION void self_scalar_division_impl(std::index_sequence<Is...>,
                                                      AMatrix<T, N, M, Accessor, OwnershipType>& mat, const U& scalar) {
  ((mat[Is] /= static_cast<T>(scalar)), ...);
}

/// \brief AMatrix-matrix equality (element-wise within a tolerance)
/// \param[in] mat1 The first matrix.
/// \param[in] mat2 The second matrix.
/// \param[in] tol The tolerance.
template <size_t... Is, typename T, size_t N, size_t M, typename U, typename V, ValidAccessor<T> Accessor,
          typename OwnershipType, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  requires std::is_arithmetic_v<V>
KOKKOS_INLINE_FUNCTION bool is_close_impl(std::index_sequence<Is...>,
                                          const AMatrix<U, N, M, Accessor, OwnershipType>& mat1,
                                          const AMatrix<T, N, M, OtherAccessor, OtherOwnershipType>& mat2,
                                          const V& tol) {
  // Use the type of the tolerance to determine the comparison type
  return ((Kokkos::abs(static_cast<V>(mat1[Is]) - static_cast<V>(mat2[Is])) <= tol) && ...);
}

/// \brief Sum of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T sum_impl(std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  return (mat[Is] + ...);
}

/// \brief Product of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T product_impl(std::index_sequence<Is...>,
                                      const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  return (mat[Is] * ...);
}

/// \brief Min of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T min_impl(std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  // Initialize min_value with the first element
  T min_value = mat[0];
  ((min_value = (mat[Is] < min_value ? mat[Is] : min_value)), ...);
  return min_value;
}

/// \brief Max of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T max_impl(std::index_sequence<Is...>, const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  // Initialize max_value with the first element
  T max_value = mat[0];
  ((max_value = (mat[Is] > max_value ? mat[Is] : max_value)), ...);
  return max_value;
}

/// \brief Variance of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION OutputType variance_impl(std::index_sequence<Is...>,
                                                const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  OutputType inv_NM = static_cast<OutputType>(1.0) / static_cast<OutputType>(N * M);
  OutputType mat_mean = inv_NM * sum_impl(std::make_index_sequence<N * M>{}, mat);
  return (((static_cast<OutputType>(mat[Is]) - mat_mean) * (static_cast<OutputType>(mat[Is]) - mat_mean)) + ...) *
         inv_NM;
}

/// \brief Standard deviation of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION OutputType standard_deviation_impl(std::index_sequence<Is...>,
                                                          const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  return std::sqrt(variance_impl(std::make_index_sequence<N * M>{}, mat));
}

/// \brief AMatrix determinant (specialized for size 1 matrices)
template <size_t N, size_t... Is, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
  requires(N == 1)
KOKKOS_INLINE_FUNCTION auto determinant_impl(std::index_sequence<Is...>,
                                             const AMatrix<T, N, N, Accessor, OwnershipType>& mat) {
  return mat(0, 0);
}

/// \brief AMatrix determinant
template <size_t N, size_t... Is, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
  requires(N != 1)
KOKKOS_INLINE_FUNCTION auto determinant_impl(std::index_sequence<Is...>,
                                             const AMatrix<T, N, N, Accessor, OwnershipType>& mat) {
  // Recursively compute the determinant using the Laplace expansion
  // Use views to avoid copying the matrix
  return ((mat(0, Is) * determinant_impl<N - 1>(std::make_index_sequence<N - 1>{}, mat.template view_minor<0, Is>()) *
           ((Is % 2 == 0) ? 1 : -1)) +
          ...);
}

/// \brief AMatrix transpose
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto transpose_impl(std::index_sequence<Is...>,
                                           const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  AMatrix<T, M, N> result;
  ((result(Is % M, Is / M) = mat(Is / N, Is % N)), ...);
  return result;
}

/// \brief AMatrix cofactors
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto cofactors_impl(std::index_sequence<Is...>,
                                           const AMatrix<T, N, N, Accessor, OwnershipType>& mat) {
  AMatrix<T, N, N> result;
  ((result[Is] = determinant(mat.template view_minor<Is / N, Is % N>()) * ((Is % 2 == 0) ? 1 : -1)), ...);
  return result;
}

/// \brief Frobenius inner product of two matrices
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto frobenius_inner_product_impl(
    std::index_sequence<Is...>, const AMatrix<U, N, M, Accessor, OwnershipType>& mat1,
    const AMatrix<T, N, M, OtherAccessor, OtherOwnershipType>& mat2) {
  using CommonType = std::common_type_t<T, U>;
  return ((static_cast<CommonType>(mat1[Is]) * static_cast<CommonType>(mat2[Is])) + ...);
}

/// \brief Outer product of two vectors (result is a matrix)
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto outer_product_impl(std::index_sequence<Is...>,
                                               const AVector<U, N, Accessor, OwnershipType>& vec1,
                                               const AVector<T, M, OtherAccessor, OtherOwnershipType>& vec2) {
  using CommonType = std::common_type_t<T, U>;
  AMatrix<CommonType, N, M> result;
  ((result(Is / M, Is % M) = static_cast<CommonType>(vec1[Is / M]) * static_cast<CommonType>(vec2[Is % M])), ...);
  return result;
}

/// \brief Infinity norm
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T infinity_norm_impl(std::index_sequence<Is...>,
                                            const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  T max_value = Kokkos::abs(sum(mat.template view_row<0>()));
  ((max_value = Kokkos::max(max_value, Kokkos::abs(sum(mat.template view_row<Is>())))), ...);
  return max_value;
}

/// \brief One norm
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T one_norm_impl(std::index_sequence<Is...>,
                                       const AMatrix<T, N, M, Accessor, OwnershipType>& mat) {
  // Max absolute column sum
  T max_value = Kokkos::abs(sum(mat.template view_column<0>()));
  ((max_value = Kokkos::max(max_value, Kokkos::abs(sum(mat.template view_column<Is>())))), ...);
  return max_value;
}

/// \brief Element-wise multiplication
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto matrix_matrix_elementwise_mul_impl(
    std::index_sequence<Is...>, const AMatrix<U, N, M, Accessor, OwnershipType>& a,
    const AMatrix<T, N, M, OtherAccessor, OtherOwnershipType>& b) {
  using CommonType = std::common_type_t<U, T>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(a[Is]) * static_cast<CommonType>(b[Is])), ...);
  return result;
}

/// \brief Element-wise division
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto matrix_matrix_elementwise_div_impl(
    std::index_sequence<Is...>, const AMatrix<U, N, M, Accessor, OwnershipType>& a,
    const AMatrix<T, N, M, OtherAccessor, OtherOwnershipType>& b) {
  using CommonType = std::common_type_t<U, T>;
  AMatrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(a[Is]) / static_cast<CommonType>(b[Is])), ...);
  return result;
}

/// \brief Apply a function to each element of the matrix
template <size_t... Is, typename Func, typename T, size_t N, size_t M, ValidAccessor<T> Accessor,
          typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto apply_impl(std::index_sequence<Is...>, const Func& func,
                                       const AMatrix<T, N, M, Accessor, OwnershipType>& mat)
    -> AMatrix<std::invoke_result_t<Func, T>, N, M> {
  using ResultType = std::invoke_result_t<Func, T>;
  AMatrix<ResultType, N, M> result;
  ((result[Is] = func(mat[Is])), ...);
  return result;
}

/// \brief Apply a function to each row of the matrix
template <size_t... Is, typename Func, typename T, size_t N, size_t M, ValidAccessor<T> Accessor,
          typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto apply_row_impl(std::index_sequence<Is...>, const Func& func,
                                           const AMatrix<T, N, M, Accessor, OwnershipType>& mat)
    -> AMatrix<typename std::invoke_result_t<Func, Vector<T, M>>::scalar_t, N, M> {
  using ResultType = typename std::invoke_result_t<Func, Vector<T, M>>::scalar_t;
  AMatrix<ResultType, N, M> result;
  ((result.template view_row<Is>() = func(mat.template view_row<Is>())), ...);
  return result;
}

/// \brief Apply a function to each column of the matrix
template <size_t... Is, typename Func, typename T, size_t N, size_t M, ValidAccessor<T> Accessor,
          typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto apply_column_impl(std::index_sequence<Is...>, const Func& func,
                                              const AMatrix<T, N, M, Accessor, OwnershipType>& mat)
    -> AMatrix<typename std::invoke_result_t<Func, Vector<T, N>>::scalar_t, N, M> {
  using ResultType = typename std::invoke_result_t<Func, Vector<T, M>>::scalar_t;
  AMatrix<ResultType, N, M> result;
  ((result.template view_column<Is>() = func(mat.template view_column<Is>())), ...);
  return result;
}

/// \brief Atomic m_copy = m.
///
/// Note: Even if the input is a view, the return is a plain owning matrix.
template <size_t... Is, size_t N, size_t M, typename T, ValidAccessor<T> A, typename OT>
KOKKOS_INLINE_FUNCTION AMatrix<T, N, M> atomic_matrix_load_impl(std::index_sequence<Is...>,
                                                                AMatrix<T, N, M, A, OT>* const m) {
  AMatrix<T, N, M> result;
  ((result[Is] = Kokkos::atomic_load(&((*m)[Is]))), ...);
  return result;
}

/// \brief Atomic m[i] = s.
template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A, typename OT, typename T2>
KOKKOS_INLINE_FUNCTION void atomic_matrix_scalar_store_impl(std::index_sequence<Is...>,
                                                            AMatrix<T1, N, M, A, OT>* const m, const T2& s) {
  ((Kokkos::atomic_store(&((*m)[Is]), static_cast<T1>(s))), ...);
}

/// \brief Atomic m1[i] = m2[i].
template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2,
          ValidAccessor<T2> A2, typename OT2>
KOKKOS_INLINE_FUNCTION void atomic_matrix_matrix_store_impl(std::index_sequence<Is...>,
                                                            AMatrix<T1, N, M, A1, OT1>* const m1,
                                                            const AMatrix<T2, N, M, A2, OT2>& m2) {
  ((Kokkos::atomic_store(&((*m1)[Is]), m2[Is])), ...);
}

#define MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_IMPL(op_name, atomic_op)                                         \
  template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2> \
  KOKKOS_INLINE_FUNCTION void atomic_matrix_scalar_##op_name##_impl(                                        \
      std::index_sequence<Is...>, AMatrix<T1, N, M, A1, OT1>* const m, const T2& s) {                       \
    ((atomic_op(&((*m)[Is]), static_cast<T1>(s))), ...);                                                    \
  }

#define MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_IMPL(op_name, atomic_op)                                             \
  template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2,     \
            ValidAccessor<T2> A2, typename OT2>                                                                 \
  KOKKOS_INLINE_FUNCTION void atomic_matrix_matrix_##op_name##_impl(                                            \
      std::index_sequence<Is...>, AMatrix<T1, N, M, A1, OT1>* const m1, const AMatrix<T2, N, M, A2, OT2>& m2) { \
    ((atomic_op(&((*m1)[Is]), static_cast<T1>(m2[Is]))), ...);                                                  \
  }

/// \brief Atomic m[i, j] += s
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_IMPL(add, Kokkos::atomic_add)

/// \brief Atomic m[i, j] -= s
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_IMPL(sub, Kokkos::atomic_sub)

/// \brief Atomic m[i, j] *= s
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_IMPL(mul, Kokkos::atomic_mul)

/// \brief Atomic m[i, j] /= s
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_IMPL(div, Kokkos::atomic_div)

/// \brief Atomic m1[i, j] += m2[i, j]
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_IMPL(add, Kokkos::atomic_add)

/// \brief Atomic m1[i, j] -= m2[i, j]
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_IMPL(sub, Kokkos::atomic_sub)

/// \brief Atomic m1[i, j] *= m2[i, j]
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_IMPL(elementwise_mul, Kokkos::atomic_mul)

/// \brief Atomic m1[i, j] /= m2[i, j]
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_IMPL(elementwise_div, Kokkos::atomic_div)

#define MUNDY_MATH_MATRIX_SCALAR_ATOMIC_FETCH_OP_IMPL(op_name, atomic_fetch_op)                             \
  template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2> \
  KOKKOS_INLINE_FUNCTION auto matrix_scalar_atomic_fetch_##op_name##_impl(                                  \
      std::index_sequence<Is...>, AMatrix<T1, N, M, A1, OT1>* const m, const T2& s) {                       \
    AMatrix<T1, N, M> result;                                                                               \
    ((result[Is] = atomic_fetch_op(&((*m)[Is]), static_cast<T1>(s))), ...);                                 \
    return result;                                                                                          \
  }

#define MUNDY_MATH_MATRIX_MATRIX_ATOMIC_FETCH_OP_IMPL(op_name, atomic_fetch_op)                                 \
  template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2,     \
            ValidAccessor<T2> A2, typename OT2>                                                                 \
  KOKKOS_INLINE_FUNCTION auto matrix_matrix_atomic_fetch_##op_name##_impl(                                      \
      std::index_sequence<Is...>, AMatrix<T1, N, M, A1, OT1>* const m1, const AMatrix<T2, N, M, A2, OT2>& m2) { \
    AMatrix<T1, N, M> result;                                                                                   \
    ((result[Is] = atomic_fetch_op(&((*m1)[Is]), static_cast<T1>(m2[Is]))), ...);                               \
    return result;                                                                                              \
  }

#define MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_FETCH_IMPL(op_name, atomic_op_fetch)                             \
  template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2> \
  KOKKOS_INLINE_FUNCTION auto matrix_scalar_atomic_##op_name##_fetch_impl(                                  \
      std::index_sequence<Is...>, AMatrix<T1, N, M, A1, OT1>* const m, const T2& s) {                       \
    AMatrix<T1, N, M> result;                                                                               \
    ((result[Is] = atomic_op_fetch(&((*m)[Is]), static_cast<T1>(s))), ...);                                 \
    return result;                                                                                          \
  }

#define MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_FETCH_IMPL(op_name, atomic_op_fetch)                                 \
  template <size_t... Is, size_t N, size_t M, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2,     \
            ValidAccessor<T2> A2, typename OT2>                                                                 \
  KOKKOS_INLINE_FUNCTION auto matrix_matrix_atomic_##op_name##_fetch_impl(                                      \
      std::index_sequence<Is...>, AMatrix<T1, N, M, A1, OT1>* const m1, const AMatrix<T2, N, M, A2, OT2>& m2) { \
    AMatrix<T1, N, M> result;                                                                                   \
    ((result[Is] = atomic_op_fetch(&((*m1)[Is]), static_cast<T1>(m2[Is]))), ...);                               \
    return result;                                                                                              \
  }

/// \brief Atomic m[i, j] += s (returns old/new m)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_FETCH_OP_IMPL(add, Kokkos::atomic_fetch_add)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_FETCH_IMPL(add, Kokkos::atomic_add_fetch)

/// \brief Atomic m[i, j] -= s (returns old/new m)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_FETCH_OP_IMPL(sub, Kokkos::atomic_fetch_sub)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_FETCH_IMPL(sub, Kokkos::atomic_sub_fetch)

/// \brief Atomic m[i, j] *= s (returns old/new m)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_FETCH_OP_IMPL(mul, Kokkos::atomic_fetch_mul)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_FETCH_IMPL(mul, Kokkos::atomic_mul_fetch)

/// \brief Atomic m[i, j] /= s (returns old/new m)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_FETCH_OP_IMPL(div, Kokkos::atomic_fetch_div)
MUNDY_MATH_MATRIX_SCALAR_ATOMIC_OP_FETCH_IMPL(div, Kokkos::atomic_div_fetch)

/// \brief Atomic m1[i, j] += m2[i, j] (returns old/new m1)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_FETCH_OP_IMPL(add, Kokkos::atomic_fetch_add)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_FETCH_IMPL(add, Kokkos::atomic_add_fetch)

/// \brief Atomic m1[i, j] -= m2[i, j] (returns old/new m1)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_FETCH_OP_IMPL(sub, Kokkos::atomic_fetch_sub)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_FETCH_IMPL(sub, Kokkos::atomic_sub_fetch)

/// \brief Atomic m1[i, j] *= m2[i, j] (returns old/new m1)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_FETCH_OP_IMPL(elementwise_mul, Kokkos::atomic_fetch_mul)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_FETCH_IMPL(elementwise_mul, Kokkos::atomic_mul_fetch)

/// \brief Atomic m1[i, j] /= m2[i, j] (returns old/new m1)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_FETCH_OP_IMPL(elementwise_div, Kokkos::atomic_fetch_div)
MUNDY_MATH_MATRIX_MATRIX_ATOMIC_OP_FETCH_IMPL(elementwise_div, Kokkos::atomic_div_fetch)
//@}

}  // namespace impl

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_MATRIXIMPL_HPP_
