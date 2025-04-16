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
#include <mundy_math/Vector.hpp>          // for mundy::math::Vector

namespace mundy {

namespace math {

template <typename T, size_t N, size_t M, ValidAccessor<T> Accessor = Array<T, N * M>,
          typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
class Matrix;

namespace impl {
//! \name Helper functions for generic matrix operators applied to an abstract accessor.
//@{

/// \brief Deep copy assignment operator with (potentially) different accessor
/// \details Copies the data from the other matrix to our data. This is only enabled if T is not const.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void deep_copy_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                           const Matrix<T, N, M, OtherAccessor, OtherOwnershipType>& other) {
  ((mat[Is] = other[Is]), ...);
}

/// \brief Move assignment operator with same accessor
/// \details Moves the data from the other matrix to our data. This is only enabled if T is not const.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void move_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                      Matrix<T, N, M, OtherAccessor, OtherOwnershipType>&& other) {
  ((mat[Is] = std::move(other[Is])), ...);
}

/// \brief Get a deep copy of a certain column of the matrix
/// \param[in] col The column index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION Vector<std::remove_const_t<T>, N> copy_column_impl(
    std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat, size_t col) {
  return {mat[col + Is * N]...};
}

/// \brief Get a deep copy of a certain row of the matrix
/// \param[in] row The row index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION Vector<std::remove_const_t<T>, M> copy_row_impl(
    std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat, size_t row) {
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
KOKKOS_INLINE_FUNCTION auto cast_impl(std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return Matrix<U, N, M>{static_cast<U>(mat[Is])...};
}

/// \brief Set all elements of the matrix
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename... Args>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                     Args&&... args) {
  ((mat[Is] = std::forward<Args>(args)), ...);
}

/// \brief Set all elements of the matrix using an accessor
/// \param[in] accessor A valid accessor.
/// \note A Matrix is also a valid accessor.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                     const auto& accessor) {
  ((mat[Is] = accessor[Is]), ...);
}

/// \brief Set a certain row of the matrix
/// \param[in] i The row index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename... Args>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_row_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                         const size_t& i, Args&&... args) {
  ((mat[Is + M * i] = std::forward<Args>(args)), ...);
}

/// \brief Set a certain row of the matrix
/// \param[in] i The row index.
/// \param[in] row The row vector.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_row_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                         const size_t& i, const Vector<T, M, OtherAccessor, OtherOwnershipType>& row) {
  ((mat[Is + M * i] = row[Is]), ...);
}

/// \brief Set a certain column of the matrix
/// \param[in] j The column index.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename... Args>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_column_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                            const size_t& j, Args&&... args) {
  ((mat[j + Is * N] = std::forward<Args>(args)), ...);
}

/// \brief Set a certain column of the matrix
/// \param[in] j The column index.
/// \param[in] col The column vector.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void set_column_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                            const size_t& j,
                                            const Vector<T, N, OtherAccessor, OtherOwnershipType>& col) {
  ((mat[j + Is * N] = col[Is]), ...);
}

/// \brief Set all elements of the matrix to a single value
/// \param[in] value The value to set all elements to.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void fill_impl(std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                      const T& value) {
  ((mat[Is] = value), ...);
}

/// \brief Unary minus operator
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION Matrix<T, N, M> unary_minus_impl(std::index_sequence<Is...>,
                                                        const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  Matrix<T, N, M> result;
  ((result[Is] = -mat[Is]), ...);
  return result;
}

/// \brief Matrix-matrix addition
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_matrix_addition_impl(std::index_sequence<Is...>,
                                                        const Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                        const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
    -> Matrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) + static_cast<CommonType>(other[Is])), ...);
  return result;
}

/// \brief Self-matrix addition
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void self_matrix_addition_impl(std::index_sequence<Is...>,
                                                      Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                      const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((mat[Is] += static_cast<T>(other[Is])), ...);
}

/// \brief Matrix-matrix subtraction
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_matrix_subtraction_impl(
    std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat,
    const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other) -> Matrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) - static_cast<CommonType>(other[Is])), ...);
  return result;
}

/// \brief Matrix-matrix subtraction
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION void self_matrix_subtraction_impl(
    std::index_sequence<Is...>, Matrix<T, N, M, Accessor, OwnershipType>& mat,
    const Matrix<U, N, M, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((mat[Is] -= static_cast<T>(other[Is])), ...);
}

/// \brief Matrix-scalar addition
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_addition_impl(std::index_sequence<Is...>,
                                                        const Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                        const U& scalar) -> Matrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) + static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Matrix-scalar addition
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION void self_scalar_addition_impl(std::index_sequence<Is...>,
                                                      Matrix<T, N, M, Accessor, OwnershipType>& mat, const U& scalar) {
  ((mat[Is] += static_cast<T>(scalar)), ...);
}

/// \brief Matrix-scalar subtraction
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_subtraction_impl(std::index_sequence<Is...>,
                                                           const Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                           const U& scalar) -> Matrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) - static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar subtraction
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION void self_scalar_subtraction_impl(std::index_sequence<Is...>,
                                                         Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                         const U& scalar) {
  ((mat[Is] -= static_cast<T>(scalar)), ...);
}

/// \brief Matrix-matrix multiplication
/// \param[in] other The other matrix.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          size_t OtherN, size_t OtherM, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_matrix_multiplication_impl(
    std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat,
    const Matrix<U, OtherN, OtherM, OtherAccessor, OtherOwnershipType>& other)
    -> Matrix<std::common_type_t<T, U>, OtherN, OtherM> {
  static_assert(M == OtherN,
                "Matrix-matrix multiplication requires the number of columns in the first matrix to be equal to the "
                "number of rows in the second matrix.");

  // We need use a fold expressions to compute the dot product of each row of the first matrix
  // with each column of the second matrix via view symmantics.
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, OtherM> result;
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
    std::index_sequence<Is...>, Matrix<T, N, N, Accessor, OwnershipType>& mat,
    const Matrix<U, N, N, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  // When writing to self, it's important to not write over data before we use it.
  Matrix<T, N, N> tmp;
  (...,
   (tmp(Is / N, Is % N) = static_cast<T>(dot(mat.template view_row<Is / N>(), other.template view_column<Is % N>()))));
  mat = tmp;
}

/// \brief Matrix-vector multiplication
/// \param[in] other The other vector.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto matrix_vector_multiplication_impl(
    std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat,
    const Vector<U, M, OtherAccessor, OtherOwnershipType>& other) -> Vector<std::common_type_t<T, U>, N> {
  using CommonType = std::common_type_t<T, U>;
  Vector<CommonType, N> result;
  (..., (result[Is] = dot(mat.template view_row<Is>(), other)));
  return result;
}

/// \brief Matrix-scalar multiplication
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_multiplication_impl(std::index_sequence<Is...>,
                                                              const Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                              const U& scalar)
    -> Matrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) * static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar multiplication
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires(HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>)
KOKKOS_INLINE_FUNCTION void self_scalar_multiplication_impl(std::index_sequence<Is...>,
                                                            Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                            const U& scalar)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((mat[Is] *= static_cast<T>(scalar)), ...);
}

/// \brief Matrix-scalar division
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION auto matrix_scalar_division_impl(std::index_sequence<Is...>,
                                                        const Matrix<T, N, M, Accessor, OwnershipType>& mat,
                                                        const U& scalar) -> Matrix<std::common_type_t<T, U>, N, M> {
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, M> result;
  ((result[Is] = static_cast<CommonType>(mat[Is]) / static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar division
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType, typename U>
  requires HasNonConstAccessOperator<Accessor, T> && std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION void self_scalar_division_impl(std::index_sequence<Is...>,
                                                      Matrix<T, N, M, Accessor, OwnershipType>& mat, const U& scalar) {
  ((mat[Is] /= static_cast<T>(scalar)), ...);
}

/// \brief Matrix-matrix equality (element-wise within a tolerance)
/// \param[in] mat1 The first matrix.
/// \param[in] mat2 The second matrix.
/// \param[in] tol The tolerance.
template <size_t... Is, typename T, size_t N, size_t M, typename U, typename V, ValidAccessor<T> Accessor,
          typename OwnershipType, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  requires std::is_arithmetic_v<V>
KOKKOS_INLINE_FUNCTION bool is_close_impl(std::index_sequence<Is...>,
                                          const Matrix<U, N, M, Accessor, OwnershipType>& mat1,
                                          const Matrix<T, N, M, OtherAccessor, OtherOwnershipType>& mat2,
                                          const V& tol) {
  // Use the type of the tolerance to determine the comparison type
  return ((Kokkos::abs(static_cast<V>(mat1[Is]) - static_cast<V>(mat2[Is])) <= tol) && ...);
}

/// \brief Sum of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T sum_impl(std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return (mat[Is] + ...);
}

/// \brief Product of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T product_impl(std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return (mat[Is] * ...);
}

/// \brief Min of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T min_impl(std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  // Initialize min_value with the first element
  T min_value = mat[0];
  ((min_value = (mat[Is] < min_value ? mat[Is] : min_value)), ...);
  return min_value;
}

/// \brief Max of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T max_impl(std::index_sequence<Is...>, const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  // Initialize max_value with the first element
  T max_value = mat[0];
  ((max_value = (mat[Is] > max_value ? mat[Is] : max_value)), ...);
  return max_value;
}

/// \brief Variance of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION OutputType variance_impl(std::index_sequence<Is...>,
                                                const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  OutputType inv_NM = static_cast<OutputType>(1.0) / static_cast<OutputType>(N * M);
  OutputType mat_mean = inv_NM * sum_impl(std::make_index_sequence<N * M>{}, mat);
  return (((static_cast<OutputType>(mat[Is]) - mat_mean) * (static_cast<OutputType>(mat[Is]) - mat_mean)) + ...) *
         inv_NM;
}

/// \brief Standard deviation of all elements
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION OutputType standard_deviation_impl(std::index_sequence<Is...>,
                                                          const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  return std::sqrt(variance_impl(std::make_index_sequence<N * M>{}, mat));
}

/// \brief Matrix determinant (specialized for size 1 matrices)
template <size_t N, size_t... Is, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
  requires(N == 1)
KOKKOS_INLINE_FUNCTION auto determinant_impl(std::index_sequence<Is...>,
                                             const Matrix<T, N, N, Accessor, OwnershipType>& mat) {
  return mat(0, 0);
}

/// \brief Matrix determinant
template <size_t N, size_t... Is, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
  requires(N != 1)
KOKKOS_INLINE_FUNCTION auto determinant_impl(std::index_sequence<Is...>,
                                             const Matrix<T, N, N, Accessor, OwnershipType>& mat) {
  // Recursively compute the determinant using the Laplace expansion
  // Use views to avoid copying the matrix
  return ((mat(0, Is) * determinant_impl<N - 1>(std::make_index_sequence<N - 1>{}, mat.template view_minor<0, Is>()) *
           ((Is % 2 == 0) ? 1 : -1)) +
          ...);
}

/// \brief Matrix transpose
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto transpose_impl(std::index_sequence<Is...>,
                                           const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  Matrix<T, M, N> result;
  ((result(Is % M, Is / M) = mat(Is / N, Is % N)), ...);
  return result;
}

/// \brief Matrix cofactors
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION auto cofactors_impl(std::index_sequence<Is...>,
                                           const Matrix<T, N, N, Accessor, OwnershipType>& mat) {
  Matrix<T, N, N> result;
  ((result[Is] = determinant(mat.template view_minor<Is / N, Is % N>()) * ((Is % 2 == 0) ? 1 : -1)), ...);
  return result;
}

/// \brief Frobenius inner product of two matrices
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto frobenius_inner_product_impl(
    std::index_sequence<Is...>, const Matrix<U, N, M, Accessor, OwnershipType>& mat1,
    const Matrix<T, N, M, OtherAccessor, OtherOwnershipType>& mat2) {
  using CommonType = std::common_type_t<T, U>;
  return ((static_cast<CommonType>(mat1[Is]) * static_cast<CommonType>(mat2[Is])) + ...);
}

/// \brief Outer product of two vectors (result is a matrix)
template <size_t... Is, typename T, size_t N, size_t M, typename U, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION auto outer_product_impl(std::index_sequence<Is...>,
                                               const Vector<U, N, Accessor, OwnershipType>& vec1,
                                               const Vector<T, M, OtherAccessor, OtherOwnershipType>& vec2) {
  using CommonType = std::common_type_t<T, U>;
  Matrix<CommonType, N, M> result;
  ((result(Is / M, Is % M) = static_cast<CommonType>(vec1[Is / M]) * static_cast<CommonType>(vec2[Is % M])), ...);
  return result;
}

/// \brief Infinity norm
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T infinity_norm_impl(std::index_sequence<Is...>,
                                            const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  T max_value = Kokkos::abs(sum(mat.template view_row<0>()));
  ((max_value = Kokkos::max(max_value, Kokkos::abs(sum(mat.template view_row<Is>())))), ...);
  return max_value;
}

/// \brief One norm
template <size_t... Is, typename T, size_t N, size_t M, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T one_norm_impl(std::index_sequence<Is...>,
                                       const Matrix<T, N, M, Accessor, OwnershipType>& mat) {
  // Max absolute column sum
  T max_value = Kokkos::abs(sum(mat.template view_column<0>()));
  ((max_value = Kokkos::max(max_value, Kokkos::abs(sum(mat.template view_column<Is>())))), ...);
  return max_value;
}
//@}
}  // namespace impl

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_MATRIXIMPL_HPP_
