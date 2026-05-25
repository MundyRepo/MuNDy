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

#ifndef MUNDY_MESH_COMPONENTACCESS_HPP_
#define MUNDY_MESH_COMPONENTACCESS_HPP_

/// \file ComponentAccess.hpp
/// \brief Storage-independent component access shape tags and traits.
///
/// This header is intentionally minimal. It may be included by restriction-layer headers
/// (DeclarePart, DeclareClass) without pulling in field or shared component implementations.
///
/// Do NOT include FieldComponent.hpp, SharedComponent.hpp, or DeclareComponent.hpp from here.

// C++ core
#include <type_traits>

// Mundy math / geometry
#include <mundy_geom/primitives/AABB.hpp>  // for mundy::AABB, is_aabb_v
#include <mundy_math/Matrix3.hpp>          // for mundy::Matrix3, is_matrix3_v
#include <mundy_math/Quaternion.hpp>       // for mundy::Quaternion, is_quaternion_v
#include <mundy_math/Vector.hpp>           // for mundy::Vector, is_vector_v

namespace mundy {

namespace mesh {

// ======================================================================================================================
// Access shape tag types
// ======================================================================================================================

namespace access {

/// Raw access: exposes entity field data as a flat array / EntityFieldData view.
template <typename ValueType>
struct raw {
  using scalar_type = ValueType;
};

/// Scalar access: one arithmetic value per entity.
template <typename ScalarType>
struct scalar {
  using scalar_type = ScalarType;
};

/// Fixed-size vector access: N scalars per entity.
template <typename ScalarType, size_t N>
struct vector {
  using scalar_type = ScalarType;
  static constexpr size_t size = N;
};

/// 3x3 matrix access: 9 scalars per entity.
template <typename ScalarType>
struct matrix3 {
  using scalar_type = ScalarType;
};

/// Quaternion access: 4 scalars per entity.
template <typename ScalarType>
struct quaternion {
  using scalar_type = ScalarType;
};

/// Axis-aligned bounding box access: 6 scalars per entity.
template <typename ScalarType>
struct aabb {
  using scalar_type = ScalarType;
};

}  // namespace access

// ======================================================================================================================
// canonical_component_access — maps user-facing types to canonical access tags
// ======================================================================================================================

template <typename AccessLike, typename Enable = void>
struct canonical_component_access {
  using type = access::raw<std::remove_cvref_t<AccessLike>>;
};

template <typename ValueType>
struct canonical_component_access<access::raw<ValueType>, void> {
  using type = access::raw<std::remove_cvref_t<ValueType>>;
};

template <typename ScalarType>
struct canonical_component_access<access::scalar<ScalarType>, void> {
  using type = access::scalar<std::remove_cvref_t<ScalarType>>;
};

template <typename ScalarType, size_t N>
struct canonical_component_access<access::vector<ScalarType, N>, void> {
  using type = access::vector<std::remove_cvref_t<ScalarType>, N>;
};

template <typename ScalarType>
struct canonical_component_access<access::matrix3<ScalarType>, void> {
  using type = access::matrix3<std::remove_cvref_t<ScalarType>>;
};

template <typename ScalarType>
struct canonical_component_access<access::quaternion<ScalarType>, void> {
  using type = access::quaternion<std::remove_cvref_t<ScalarType>>;
};

template <typename ScalarType>
struct canonical_component_access<access::aabb<ScalarType>, void> {
  using type = access::aabb<std::remove_cvref_t<ScalarType>>;
};

// Arithmetic scalars map to access::scalar
template <typename ScalarType>
struct canonical_component_access<ScalarType,
                                   std::enable_if_t<std::is_arithmetic_v<std::remove_cvref_t<ScalarType>>>> {
  using type = access::scalar<std::remove_cvref_t<ScalarType>>;
};

// Mundy Vector<> types map to access::vector
template <typename VectorType>
struct canonical_component_access<VectorType,
                                   std::enable_if_t<is_vector_v<std::remove_cvref_t<VectorType>>>> {
  using decayed = std::remove_cvref_t<VectorType>;
  using type    = access::vector<typename decayed::scalar_t, decayed::size>;
};

// Mundy Matrix3<> types map to access::matrix3
template <typename Matrix3Type>
struct canonical_component_access<Matrix3Type,
                                   std::enable_if_t<is_matrix3_v<std::remove_cvref_t<Matrix3Type>>>> {
  using decayed = std::remove_cvref_t<Matrix3Type>;
  using type    = access::matrix3<typename decayed::scalar_t>;
};

// Mundy Quaternion<> types map to access::quaternion
template <typename QuaternionType>
struct canonical_component_access<QuaternionType,
                                   std::enable_if_t<is_quaternion_v<std::remove_cvref_t<QuaternionType>>>> {
  using decayed = std::remove_cvref_t<QuaternionType>;
  using type    = access::quaternion<typename decayed::scalar_t>;
};

// Mundy AABB<> types map to access::aabb
template <typename AABBType>
struct canonical_component_access<AABBType,
                                   std::enable_if_t<is_aabb_v<std::remove_cvref_t<AABBType>>>> {
  using decayed = std::remove_cvref_t<AABBType>;
  using type    = access::aabb<typename decayed::scalar_t>;
};

template <typename AccessLike>
using canonical_component_access_t = typename canonical_component_access<AccessLike>::type;

// ======================================================================================================================
// component_access_shape — storage-independent shape facts for a canonical access tag
//
// Provides:
//   field_scalar_type       — the STK field scalar type this access shape uses
//   shared_value_type       — the C++ value type stored in a shared component
//   has_fixed_field_scalars — true if the scalar count per entity is statically known
//   field_scalars           — scalar count per entity (valid only if has_fixed_field_scalars)
//
// IO output type defaults are NOT part of this trait — they depend on stk::io and live in
// impl/DeclareComponentImpl.hpp as component_default_output_type<CanonicalAccess>.
// ======================================================================================================================

template <typename CanonicalAccess>
struct component_access_shape;

template <typename ValueType>
struct component_access_shape<access::raw<ValueType>> {
  using field_scalar_type  = ValueType;
  using shared_value_type  = ValueType;
  static constexpr bool has_fixed_field_scalars = false;
  static constexpr bool has_default_output_type  = false;
};

template <typename ScalarType>
struct component_access_shape<access::scalar<ScalarType>> {
  using field_scalar_type  = ScalarType;
  using shared_value_type  = ScalarType;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 1;
};

template <typename ScalarType, size_t N>
struct component_access_shape<access::vector<ScalarType, N>> {
  using field_scalar_type  = ScalarType;
  using shared_value_type  = Vector<ScalarType, N>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = static_cast<unsigned>(N);
};

template <typename ScalarType>
struct component_access_shape<access::matrix3<ScalarType>> {
  using field_scalar_type  = ScalarType;
  using shared_value_type  = Matrix3<ScalarType>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 9;
};

template <typename ScalarType>
struct component_access_shape<access::quaternion<ScalarType>> {
  using field_scalar_type  = ScalarType;
  using shared_value_type  = Quaternion<ScalarType>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 4;
};

template <typename ScalarType>
struct component_access_shape<access::aabb<ScalarType>> {
  using field_scalar_type  = ScalarType;
  using shared_value_type  = AABB<ScalarType>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 6;
  static constexpr bool     has_default_output_type  = false;
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_COMPONENTACCESS_HPP_
