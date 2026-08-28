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

#ifndef MUNDY_MATH_MATRIX3_HPP_
#define MUNDY_MATH_MATRIX3_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>  // for std::decay_t

// Our libs
#include <mundy_math/Accessor.hpp>   // for mundy::ValidAccessor
#include <mundy_math/Array.hpp>      // for mundy::Array
#include <mundy_math/Matrix.hpp>     // for mundy::Matrix
#include <mundy_math/Tolerance.hpp>  // for mundy::get_zero_tolerance
#include <mundy_math/cmath.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \brief Get the lower triangular matrix of the Cholesky decomposition of a symmetric positive definite matrix
/// \param A The symmetric positive definite matrix
/// \return The lower triangular matrix of the Cholesky decomposition
template <typename T, ValidAccessor<T> Accessor, typename OutputType = typename NumTraits<T>::NonInteger>
KOKKOS_INLINE_FUNCTION auto cholesky(const AMatrix3<T, Accessor>& A) -> Matrix3<OutputType> {
  const passive_scalar_t<OutputType> tol = get_zero_tolerance<OutputType>();
  const OutputType a00 = static_cast<OutputType>(A(0, 0));
  const OutputType a10 = static_cast<OutputType>(A(1, 0));
  const OutputType a11 = static_cast<OutputType>(A(1, 1));
  const OutputType a20 = static_cast<OutputType>(A(2, 0));
  const OutputType a21 = static_cast<OutputType>(A(2, 1));
  const OutputType a22 = static_cast<OutputType>(A(2, 2));

  MUNDY_THROW_ASSERT(a00 > tol, std::invalid_argument, "Matrix3 must be positive definite");
  const OutputType l11 = sqrt(a00);

  const OutputType l21 = a10 / l11;
  const OutputType s22 = a11 - l21 * l21;
  MUNDY_THROW_ASSERT(s22 > tol, std::invalid_argument, "Matrix3 must be positive definite");
  const OutputType l22 = sqrt(s22);

  const OutputType l31 = a20 / l11;
  const OutputType l32 = (a21 - l31 * l21) / l22;
  const OutputType s33 = a22 - l31 * l31 - l32 * l32;
  MUNDY_THROW_ASSERT(s33 > tol, std::invalid_argument, "Matrix3 must be positive definite");
  const OutputType l33 = sqrt(s33);

  return Matrix3<OutputType>(l11, OutputType(0), OutputType(0), l21, l22, OutputType(0), l31, l32, l33);
}
//
template <typename T, ValidAccessor<T> Accessor,
          typename OutputType = std::conditional_t<NumTraits<T>::IsInteger, float, T>>
KOKKOS_INLINE_FUNCTION auto cholesky_f(const AMatrix3<T, Accessor>& A) -> Matrix3<OutputType> {
  return cholesky<T, Accessor, OutputType>(A);
}

/// \brief A temporary concept to check if a type is a valid Matrix3 type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our quaternions.
template <typename Matrix3Type>
concept ValidMatrix3Type = is_matrix3_v<std::decay_t<Matrix3Type>> &&
                           requires(std::decay_t<Matrix3Type> matrix3, const std::decay_t<Matrix3Type> const_matrix3) {
                             typename std::decay_t<Matrix3Type>::value_type;
                             { matrix3[0] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[1] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[2] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[3] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[4] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[5] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[6] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[7] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3[8] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;

                             { matrix3(0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(3) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(4) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(5) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(6) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(7) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(8) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;

                             { matrix3(0, 0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(0, 1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(0, 2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(1, 0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(1, 1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(1, 2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(2, 0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(2, 1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;
                             { matrix3(2, 2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::value_type>;

                             {
                               const_matrix3[0]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[1]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[2]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[3]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[4]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[5]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[6]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[7]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3[8]
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;

                             {
                               const_matrix3(0)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(1)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(2)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(3)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(4)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(5)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(6)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(7)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(8)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;

                             {
                               const_matrix3(0, 0)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(0, 1)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(0, 2)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(1, 0)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(1, 1)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(1, 2)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(2, 0)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(2, 1)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                             {
                               const_matrix3(2, 2)
                             } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::value_type>;
                           };  // ValidMatrix3Type

static_assert(is_matrix3_v<Matrix3<int>>, "Odd, default matrix3 is not a matrix3.");
static_assert(is_matrix3_v<AMatrix3<int, Array<int, 9>>>, "Odd, default matrix3 with Array accessor is not a matrix3.");

//@}

}  // namespace mundy

#endif  // MUNDY_MATH_MATRIX3_HPP_
