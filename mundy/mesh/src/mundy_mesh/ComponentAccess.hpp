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
#include <mundy_geom/primitives/OBB.hpp>   // for mundy::OBB, is_obb_v
#include <mundy_math/Matrix.hpp>           // for mundy::Matrix, is_matrix_v
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
  using value_typeype = ValueType;
};

/// Scalar access: one arithmetic value per entity.
template <typename ScalarType>
struct scalar {
  using value_typeype = ScalarType;
};

/// Fixed-size vector access: N scalars per entity.
template <typename ScalarType, size_t N>
struct vector {
  using value_typeype = ScalarType;
  static constexpr size_t size = N;
};

template <typename ScalarType>
using vector1 = vector<ScalarType, 1>;
template <typename ScalarType>
using vector2 = vector<ScalarType, 2>;
template <typename ScalarType>
using vector3 = vector<ScalarType, 3>;
template <typename ScalarType>
using vector4 = vector<ScalarType, 4>;
template <typename ScalarType>
using vector5 = vector<ScalarType, 5>;
template <typename ScalarType>
using vector6 = vector<ScalarType, 6>;

using vector1d = vector1<double>;
using vector2d = vector2<double>;
using vector3d = vector3<double>;
using vector4d = vector4<double>;
using vector5d = vector5<double>;
using vector6d = vector6<double>;

using vector1f = vector1<float>;
using vector2f = vector2<float>;
using vector3f = vector3<float>;
using vector4f = vector4<float>;
using vector5f = vector5<float>;
using vector6f = vector6<float>;

using vector1i = vector1<int>;
using vector2i = vector2<int>;
using vector3i = vector3<int>;
using vector4i = vector4<int>;
using vector5i = vector5<int>;
using vector6i = vector6<int>;

/// Fixed-size matrix access: N * M scalars per entity.
template <typename ScalarType, size_t N, size_t M>
struct matrix {
  using value_typeype = ScalarType;
  static constexpr size_t num_rows = N;
  static constexpr size_t num_cols = M;
};

#define MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(N, M) \
  template <typename ScalarType>                       \
  using matrix##N####M = matrix<ScalarType, N, M>;     \
  using matrix##N####M##d = matrix##N####M<double>;    \
  using matrix##N####M##f = matrix##N####M<float>;

MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(1, 1)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(1, 2)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(1, 3)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(1, 4)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(1, 5)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(1, 6)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(2, 1)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(2, 2)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(2, 3)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(2, 4)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(2, 5)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(2, 6)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(3, 1)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(3, 2)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(3, 3)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(3, 4)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(3, 5)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(3, 6)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(4, 1)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(4, 2)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(4, 3)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(4, 4)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(4, 5)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(4, 6)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(5, 1)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(5, 2)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(5, 3)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(5, 4)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(5, 5)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(5, 6)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(6, 1)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(6, 2)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(6, 3)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(6, 4)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(6, 5)
MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS(6, 6)
#undef MUNDY_MESH_COMPONENT_ACCESS_MATRIX_ALIAS

template <typename ScalarType>
using matrix1 = matrix11<ScalarType>;
template <typename ScalarType>
using matrix2 = matrix22<ScalarType>;
template <typename ScalarType>
using matrix3 = matrix33<ScalarType>;
template <typename ScalarType>
using matrix4 = matrix44<ScalarType>;
template <typename ScalarType>
using matrix5 = matrix55<ScalarType>;
template <typename ScalarType>
using matrix6 = matrix66<ScalarType>;

using matrix1d = matrix1<double>;
using matrix2d = matrix2<double>;
using matrix3d = matrix3<double>;
using matrix4d = matrix4<double>;
using matrix5d = matrix5<double>;
using matrix6d = matrix6<double>;

using matrix1f = matrix1<float>;
using matrix2f = matrix2<float>;
using matrix3f = matrix3<float>;
using matrix4f = matrix4<float>;
using matrix5f = matrix5<float>;
using matrix6f = matrix6<float>;

using matrix1i = matrix1<int>;
using matrix2i = matrix2<int>;
using matrix3i = matrix3<int>;
using matrix4i = matrix4<int>;
using matrix5i = matrix5<int>;
using matrix6i = matrix6<int>;

/// Quaternion access: 4 scalars per entity.
template <typename ScalarType>
struct quaternion {
  using value_typeype = ScalarType;
};

/// Axis-aligned bounding box access: 6 scalars per entity.
template <typename ScalarType>
struct aabb {
  using value_typeype = ScalarType;
};

/// Oriented bounding box access: 10 scalars per entity.
/// Layout: center xyz (0-2), orientation quaternion wxyz (3-6), half-extents xyz (7-9).
template <typename ScalarType>
struct obb {
  using value_typeype = ScalarType;
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

template <typename ScalarType, size_t N, size_t M>
struct canonical_component_access<access::matrix<ScalarType, N, M>, void> {
  using type = access::matrix<std::remove_cvref_t<ScalarType>, N, M>;
};

template <typename ScalarType>
struct canonical_component_access<access::quaternion<ScalarType>, void> {
  using type = access::quaternion<std::remove_cvref_t<ScalarType>>;
};

template <typename ScalarType>
struct canonical_component_access<access::aabb<ScalarType>, void> {
  using type = access::aabb<std::remove_cvref_t<ScalarType>>;
};

template <typename ScalarType>
struct canonical_component_access<access::obb<ScalarType>, void> {
  using type = access::obb<std::remove_cvref_t<ScalarType>>;
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
  using type    = access::vector<typename decayed::value_type, decayed::size>;
};

// Mundy Matrix3<> types map to access::matrix3
template <typename Matrix3Type>
struct canonical_component_access<Matrix3Type,
                                   std::enable_if_t<is_matrix3_v<std::remove_cvref_t<Matrix3Type>>>> {
  using decayed = std::remove_cvref_t<Matrix3Type>;
  using type    = access::matrix3<typename decayed::value_type>;
};

// Mundy Matrix<> types map to access::matrix
template <typename MatrixType>
struct canonical_component_access<
    MatrixType, std::enable_if_t<is_matrix_v<std::remove_cvref_t<MatrixType>> &&
                                 !is_matrix3_v<std::remove_cvref_t<MatrixType>>>> {
  using decayed = std::remove_cvref_t<MatrixType>;
  using type    = access::matrix<typename decayed::value_type, decayed::num_rows, decayed::num_cols>;
};

// Mundy Quaternion<> types map to access::quaternion
template <typename QuaternionType>
struct canonical_component_access<QuaternionType,
                                   std::enable_if_t<is_quaternion_v<std::remove_cvref_t<QuaternionType>>>> {
  using decayed = std::remove_cvref_t<QuaternionType>;
  using type    = access::quaternion<typename decayed::value_type>;
};

// Mundy AABB<> types map to access::aabb
template <typename AABBType>
struct canonical_component_access<AABBType,
                                   std::enable_if_t<is_aabb_v<std::remove_cvref_t<AABBType>>>> {
  using decayed = std::remove_cvref_t<AABBType>;
  using type    = access::aabb<typename decayed::value_type>;
};

// Mundy OBB<> types map to access::obb
template <typename OBBType>
struct canonical_component_access<OBBType,
                                   std::enable_if_t<is_obb_v<std::remove_cvref_t<OBBType>>>> {
  using decayed = std::remove_cvref_t<OBBType>;
  using type    = access::obb<typename decayed::value_type>;
};

template <typename AccessLike>
using canonical_component_access_t = typename canonical_component_access<AccessLike>::type;

// ======================================================================================================================
// component_access_shape — storage-independent shape facts for a canonical access tag
//
// Provides:
//   field_value_typeype       — the STK field scalar type this access shape uses
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
  using field_value_typeype  = ValueType;
  using shared_value_type  = ValueType;
  static constexpr bool has_fixed_field_scalars = false;
  static constexpr bool has_default_output_type  = false;
};

template <typename ScalarType>
struct component_access_shape<access::scalar<ScalarType>> {
  using field_value_typeype  = ScalarType;
  using shared_value_type  = ScalarType;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 1;
};

template <typename ScalarType, size_t N>
struct component_access_shape<access::vector<ScalarType, N>> {
  using field_value_typeype  = ScalarType;
  using shared_value_type  = Vector<ScalarType, N>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = static_cast<unsigned>(N);
};

template <typename ScalarType, size_t N, size_t M>
struct component_access_shape<access::matrix<ScalarType, N, M>> {
  using field_value_typeype  = ScalarType;
  using shared_value_type  = Matrix<ScalarType, N, M>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = static_cast<unsigned>(N * M);
};

template <typename ScalarType>
struct component_access_shape<access::quaternion<ScalarType>> {
  using field_value_typeype  = ScalarType;
  using shared_value_type  = Quaternion<ScalarType>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 4;
};

template <typename ScalarType>
struct component_access_shape<access::aabb<ScalarType>> {
  using field_value_typeype  = ScalarType;
  using shared_value_type  = AABB<ScalarType>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 6;
  static constexpr bool     has_default_output_type  = false;
};

template <typename ScalarType>
struct component_access_shape<access::obb<ScalarType>> {
  using field_value_typeype  = ScalarType;
  using shared_value_type  = OBB<ScalarType>;
  static constexpr bool     has_fixed_field_scalars = true;
  static constexpr unsigned field_scalars            = 10;
  static constexpr bool     has_default_output_type  = false;
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_COMPONENTACCESS_HPP_
