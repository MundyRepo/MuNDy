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

// ScalarWrapper.hpp is superseded by Scalar.hpp (AScalar / Scalar).
// This header exists only for backward compatibility and will be removed in a future release.
// Please migrate all uses of ScalarWrapper<T, Acc> to AScalar<T, Acc> and include <mundy_math/Scalar.hpp>.

#ifndef MUNDY_MATH_SCALARWRAPPER_HPP_
#define MUNDY_MATH_SCALARWRAPPER_HPP_

#include <mundy_math/Scalar.hpp>  // full AScalar / Scalar definition

namespace mundy {

// ---------------------------------------------------------------------------
// Deprecated aliases — migrate to AScalar<T, Acc> / Scalar<T>
// ---------------------------------------------------------------------------

template <typename T, ValidAccessor<T> Accessor = Array<T, 1>>
using ScalarWrapper = AScalar<T, Accessor>;

template <typename TypeToCheck>
struct is_scalar_wrapper_impl : is_scalar_impl<TypeToCheck> {};

template <typename TypeToCheck>
struct is_scalar_wrapper : is_scalar<TypeToCheck> {};

template <typename TypeToCheck>
constexpr bool is_scalar_wrapper_v = is_scalar_v<TypeToCheck>;

template <typename ScalarWrapperType>
concept ValidScalarWrapperType = ValidScalarType<ScalarWrapperType>;

}  // namespace mundy

#endif  // MUNDY_MATH_SCALARWRAPPER_HPP_
