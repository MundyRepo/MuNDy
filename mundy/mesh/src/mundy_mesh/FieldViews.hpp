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

#ifndef MUNDY_MESH_FIELDVIEWS_HPP_
#define MUNDY_MESH_FIELDVIEWS_HPP_

/// \file FieldViews.hpp
/// \brief Declaration of the field view helper functions

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/FieldBase.hpp>  // for stk::mesh::FieldBase, stk::mesh::field_data
#include <stk_mesh/base/NgpField.hpp>   // for stk::mesh::NgpField

// Mundy
#include <mundy_geom/primitives/AABB.hpp>  // for mundy::geom::AABB
#include <mundy_math/Matrix3.hpp>          // for mundy::math::get_matrix3_view and mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>       // for mundy::math::get_quaternion_view and mundy::math::Quaternion
#include <mundy_math/ScalarWrapper.hpp>    // for mundy::math::get_scalar_view and mundy::math::ScalarView
#include <mundy_math/Vector3.hpp>          // for mundy::math::get_vector3_view and mundy::math::Vector3
namespace mundy {

namespace mesh {

//! \name stk::mesh::Field data views
///@{

/// \brief A helper function for getting a view of a field's data as a scalar. 1 scalar per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto scalar_field_data(const FieldType& f, stk::mesh::Entity e,
                              stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                              const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_scalar_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief A helper function for getting a view of a field's data as a Vector3. 3 scalars per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto vector3_field_data(const FieldType& f, stk::mesh::Entity e,
                               stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                               const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_vector3_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief A helper function for getting a view of a field's data as a Quaternion. 4 scalars per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto quaternion_field_data(const FieldType& f, stk::mesh::Entity e,
                                  stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                                  const char* fileName = HOST_DEBUG_FILE_NAME,
                                  int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_quaternion_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief A helper function for getting a view of a field's data as a Matrix3. 9 scalars per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto matrix3_field_data(const FieldType& f, stk::mesh::Entity e,
                               stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                               const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_matrix3_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief A helper function for getting a view of a field's data as an AABB. 6 scalars per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto aabb_field_data(const FieldType& f, stk::mesh::Entity e,
                            stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                            const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  constexpr size_t shift = 3;
  using scalar_t = typename FieldType::value_type;
  auto shifted_data_accessor =
      math::get_shifted_view<scalar_t, shift>(stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
  auto max_corner = math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  auto min_corner = math::get_vector3_view<scalar_t>(stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));

  using min_point_t = decltype(min_corner);
  using max_point_t = decltype(max_corner);
  return geom::AABB<scalar_t, min_point_t, max_point_t>(min_corner, max_corner);
}
//@}

//! \name stk::mesh::NgpField data views
///@{

/// \brief A helper function for getting a view of a field's data as a ScalarWrapper.
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto scalar_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_scalar<typename FieldType::value_type>(f(i));
}

/// \brief A helper function for getting a view of a field's data as a Vector3
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto vector3_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_vector3<typename FieldType::value_type>(f(i));
}

/// \brief A helper function for getting a view of a field's data as a Quaternion
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto quaternion_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_quaternion<typename FieldType::value_type>(f(i));
}

/// \brief A helper function for getting a view of a field's data as a Matrix3
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto matrix3_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_matrix3<typename FieldType::value_type>(f(i));
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FIELDVIEWS_HPP_
