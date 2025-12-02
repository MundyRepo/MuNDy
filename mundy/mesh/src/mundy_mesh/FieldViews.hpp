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

/// \brief Get a view of a field's data as a scalar. 1 scalar per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto scalar_field_data(const FieldType& f, stk::mesh::Entity e,
                              stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                              const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_scalar_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief Get a view of a field's data as a Vector<N>. N scalars per entity.
template <size_t N, class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto vector_field_data(const FieldType& f, stk::mesh::Entity e,
                              stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                              const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_vector_view<typename FieldType::value_type, N>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

#define MUNDY_IMPL_VECTOR_FIELD_DATA_N(N)                                                                      \
  template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>                    \
  inline auto vector##N##_field_data(                                                                          \
      const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(), \
      const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {                  \
    return math::get_vector_view<typename FieldType::value_type, N>(                                           \
        stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));                                          \
  }

MUNDY_IMPL_VECTOR_FIELD_DATA_N(1)  // vector1_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(2)  // vector2_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(3)  // vector3_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(4)  // vector4_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(5)  // vector5_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(6)  // vector6_field_data
#undef MUNDY_IMPL_VECTOR_FIELD_DATA_N

/// \brief Get a view of a field's data as a Matrix<N, M>. N * M scalars per entity.
template <size_t N, size_t M, class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto matrix_field_data(const FieldType& f, stk::mesh::Entity e,
                              stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                              const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_matrix_view<typename FieldType::value_type, N, M>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief Get a view of a field's data as a Matrix<N, M>. N * M scalars per entity. (explicit naming)
#define MUNDY_IMPL_MATRIX_FIELD_DATA_NM(N, M)                                                                  \
  template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>                    \
  inline auto matrix##N####M##_field_data(                                                                     \
      const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(), \
      const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {                  \
    return math::get_matrix_view<typename FieldType::value_type, N, M>(                                        \
        stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));                                          \
  }

MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 1)  // matrix11_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 2)  // matrix12_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 3)  // matrix13_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 4)  // matrix14_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 5)  // matrix15_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 6)  // matrix16_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 1)  // matrix21_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 2)  // matrix22_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 3)  // matrix23_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 4)  // matrix24_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 5)  // matrix25_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 6)  // matrix26_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 1)  // matrix31_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 2)  // matrix32_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 3)  // matrix33_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 4)  // matrix34_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 5)  // matrix35_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 6)  // matrix36_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 1)  // matrix41_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 2)  // matrix42_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 3)  // matrix43_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 4)  // matrix44_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 5)  // matrix45_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 6)  // matrix46_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 1)  // matrix51_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 2)  // matrix52_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 3)  // matrix53_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 4)  // matrix54_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 5)  // matrix55_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 6)  // matrix56_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 1)  // matrix61_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 2)  // matrix62_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 3)  // matrix63_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 4)  // matrix64_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 5)  // matrix65_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 6)  // matrix66_field_data
#undef MUNDY_IMPL_MATRIX_FIELD_DATA_NM

/// \brief Get a view of a field's data as a Matrix<N, N>. N * N scalars per entity. (explicit naming square)
#define MUNDY_IMPL_MATRIX_FIELD_DATA_NN(N)                                                                     \
  template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>                    \
  inline auto matrix##N##_field_data(                                                                          \
      const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(), \
      const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {                  \
    return math::get_matrix_view<typename FieldType::value_type, N, N>(                                        \
        stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));                                          \
  }

MUNDY_IMPL_MATRIX_FIELD_DATA_NN(1)  // matrix1_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(2)  // matrix2_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(3)  // matrix3_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(4)  // matrix4_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(5)  // matrix5_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(6)  // matrix6_field_data
#undef MUNDY_IMPL_MATRIX_FIELD_DATA_NN

/// \brief Get a view of a field's data as a Quaternion. 4 scalars per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto quaternion_field_data(const FieldType& f, stk::mesh::Entity e,
                                  stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                                  const char* fileName = HOST_DEBUG_FILE_NAME,
                                  int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return math::get_quaternion_view<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief Get a view of a field's data as an AABB. 6 scalars per entity.
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

/// \brief Get a view of a field's data as a ScalarWrapper.
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto scalar_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_scalar<typename FieldType::value_type>(f(i));
}

/// \brief Get a view of a field's data as a Vector<N>
template <size_t N, class FieldType>
KOKKOS_INLINE_FUNCTION auto vector_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_vector<typename FieldType::value_type, N>(f(i));
}

#define MUNDY_IMPL_VECTOR_FIELD_DATA_N(N)                                                               \
  template <class FieldType>                                                                            \
  KOKKOS_INLINE_FUNCTION auto vector##N##_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) { \
    return math::get_owning_vector<typename FieldType::value_type, N>(f(i));                            \
  }

MUNDY_IMPL_VECTOR_FIELD_DATA_N(1)  // vector1_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(2)  // vector2_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(3)  // vector3_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(4)  // vector4_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(5)  // vector5_field_data
MUNDY_IMPL_VECTOR_FIELD_DATA_N(6)  // vector6_field_data
#undef MUNDY_IMPL_VECTOR_FIELD_DATA_N

/// \brief Get a view of a field's data as a Matrix<N, M>
template <size_t N, size_t M, class FieldType>
KOKKOS_INLINE_FUNCTION auto matrix_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_matrix<typename FieldType::value_type, N, M>(f(i));
}

#define MUNDY_IMPL_MATRIX_FIELD_DATA_NM(N, M)                                                                \
  template <class FieldType>                                                                                 \
  KOKKOS_INLINE_FUNCTION auto matrix##N####M##_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) { \
    return math::get_owning_matrix<typename FieldType::value_type, N, M>(f(i));                              \
  }

MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 1)  // matrix11_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 2)  // matrix12_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 3)  // matrix13_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 4)  // matrix14_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 5)  // matrix15_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(1, 6)  // matrix16_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 1)  // matrix21_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 2)  // matrix22_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 3)  // matrix23_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 4)  // matrix24_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 5)  // matrix25_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(2, 6)  // matrix26_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 1)  // matrix31_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 2)  // matrix32_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 3)  // matrix33_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 4)  // matrix34_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 5)  // matrix35_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(3, 6)  // matrix36_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 1)  // matrix41_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 2)  // matrix42_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 3)  // matrix43_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 4)  // matrix44_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 5)  // matrix45_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(4, 6)  // matrix46_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 1)  // matrix51_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 2)  // matrix52_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 3)  // matrix53_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 4)  // matrix54_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 5)  // matrix55_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(5, 6)  // matrix56_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 1)  // matrix61_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 2)  // matrix62_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 3)  // matrix63_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 4)  // matrix64_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 5)  // matrix65_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NM(6, 6)  // matrix66_field_data
#undef MUNDY_IMPL_MATRIX_FIELD_DATA_NM

#define MUNDY_IMPL_MATRIX_FIELD_DATA_NN(N)                                                              \
  template <class FieldType>                                                                            \
  KOKKOS_INLINE_FUNCTION auto matrix##N##_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) { \
    return math::get_owning_matrix<typename FieldType::value_type, N, N>(f(i));                         \
  }

MUNDY_IMPL_MATRIX_FIELD_DATA_NN(1)  // matrix1_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(2)  // matrix2_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(3)  // matrix3_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(4)  // matrix4_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(5)  // matrix5_field_data
MUNDY_IMPL_MATRIX_FIELD_DATA_NN(6)  // matrix6_field_data
#undef MUNDY_IMPL_MATRIX_FIELD_DATA_NN

/// \brief Get a view of a field's data as a Quaternion
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto quaternion_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return math::get_owning_quaternion<typename FieldType::value_type>(f(i));
}

/// \brief Get a view of a field's data as a Matrix3
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto aabb_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  constexpr size_t shift = 3;
  using scalar_t = typename FieldType::value_type;
  auto shifted_data_accessor = math::get_owning_shifted_accessor<scalar_t, shift>(f(i));
  auto max_corner = math::get_owning_vector3<scalar_t>(std::move(shifted_data_accessor));
  auto min_corner = math::get_owning_vector3<scalar_t>(f(i));

  using min_point_t = decltype(min_corner);
  using max_point_t = decltype(max_corner);
  return geom::AABB<scalar_t, min_point_t, max_point_t>(min_corner, max_corner);
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FIELDVIEWS_HPP_
