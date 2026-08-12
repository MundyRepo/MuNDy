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
/// \defgroup MundyMeshFieldViews mundy::mesh::FieldViews
/// \brief Host and device helpers for viewing STK field data as Mundy math and geometry types.
/// \details These free functions adapt raw STK field storage into scalar, vector, matrix, quaternion, and bounding-box
/// accessors used by Mundy's math and geometry kernels.

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
#include <mundy_geom/primitives/AABB.hpp>  // for mundy::AABB
#include <mundy_geom/primitives/OBB.hpp>   // for mundy::OBB
#include <mundy_math/Matrix3.hpp>          // for mundy::get_matrix3 and mundy::Matrix3
#include <mundy_math/Quaternion.hpp>       // for mundy::get_quaternion and mundy::Quaternion
#include <mundy_math/ScalarWrapper.hpp>    // for mundy::get_scalar and mundy::ScalarView
#include <mundy_math/Vector3.hpp>          // for mundy::get_vector3 and mundy::Vector3
#include <mundy_math/ShiftedAccessor.hpp>      // for mundy::get_shifted_accessor and mundy::get_shifted_accessor

namespace mundy {

namespace mesh {

/// \addtogroup MundyMeshFieldViews
/// @{

//! \name stk::mesh::Field data views
///@{

/// \brief Get a view of a field's data as a scalar. 1 scalar per entity.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto scalar_field_data(const FieldType& f, stk::mesh::Entity e,
                              stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                              const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return get_scalar<typename FieldType::value_type>(stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief Get a view of a field's data as a Vector<N>. N scalars per entity.
template <size_t N, class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto vector_field_data(const FieldType& f, stk::mesh::Entity e,
                              stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                              const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return get_vector<typename FieldType::value_type, N>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

#define MUNDY_IMPL_VECTOR_FIELD_DATA_N(N)                                                                      \
  template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>                    \
  inline auto vector##N##_field_data(                                                                          \
      const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(), \
      const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {                  \
    return get_vector<typename FieldType::value_type, N>(                                                 \
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
  return get_matrix<typename FieldType::value_type, N, M>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief Get a view of a field's data as a Matrix<N, M>. N * M scalars per entity. (explicit naming)
#define MUNDY_IMPL_MATRIX_FIELD_DATA_NM(N, M)                                                                  \
  template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>                    \
  inline auto matrix##N####M##_field_data(                                                                     \
      const FieldType& f, stk::mesh::Entity e, stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(), \
      const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {                  \
    return get_matrix<typename FieldType::value_type, N, M>(                                              \
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
    return get_matrix<typename FieldType::value_type, N, N>(                                              \
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
/// Layout: x (0), y (1), z (2), w (3). Same as Eigen's internal storage order for quaternions.
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto quaternion_field_data(const FieldType& f, stk::mesh::Entity e,
                                  stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                                  const char* fileName = HOST_DEBUG_FILE_NAME,
                                  int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  return get_quaternion<typename FieldType::value_type>(
      stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
}

/// \brief Get a view of a field's data as an AABB. 6 scalars per entity.
/// Layout: min corner xyz (0-2), max corner xyz (3-5).
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto aabb_field_data(const FieldType& f, stk::mesh::Entity e,
                            stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                            const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  constexpr size_t shift = 3;
  using value_type = typename FieldType::value_type;
  auto shifted_data_accessor =
      get_shifted_accessor<value_type, shift>(stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));
  auto max_corner = get_vector3<value_type>(std::move(shifted_data_accessor));
  auto min_corner = get_vector3<value_type>(stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber));

  using min_point_t = decltype(min_corner);
  using max_point_t = decltype(max_corner);
  return AABB<value_type, min_point_t, max_point_t>(min_corner, max_corner);
}

/// \brief Get a view of a field's data as an OBB. 10 scalars per entity.
/// Layout: center xyz (0-2), orientation quaternion wxyz (3-6), half-extents xyz (7-9).
template <class FieldType, typename StkDebugger = stk::mesh::DefaultStkFieldSyncDebugger>
inline auto obb_field_data(const FieldType& f, stk::mesh::Entity e,
                           stk::mesh::DummyOverload dummyArg = stk::mesh::DummyOverload(),
                           const char* fileName = HOST_DEBUG_FILE_NAME, int lineNumber = HOST_DEBUG_LINE_NUMBER) {
  using value_type  = typename FieldType::value_type;
  value_type* base  = stk::mesh::field_data(f, e, dummyArg, fileName, lineNumber);
  auto center       = get_vector3<value_type>(base);
  auto orientation  = get_quaternion<value_type>(get_shifted_accessor<value_type, 3>(base));
  auto half_extents = get_vector3<value_type>(get_shifted_accessor<value_type, 7>(base));
  using center_t    = decltype(center);
  using orient_t    = decltype(orientation);
  using he_t        = decltype(half_extents);
  return OBB<value_type, center_t, orient_t, he_t>(center, orientation, half_extents);
}
//@}

//! \name stk::mesh::NgpField data views
///@{

/// \brief Get a view of a field's data as a ScalarWrapper.
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto scalar_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return get_scalar<typename FieldType::value_type>(f(i));
}

/// \brief Get a view of a field's data as a Vector<N>
template <size_t N, class FieldType>
KOKKOS_INLINE_FUNCTION auto vector_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  return get_vector<typename FieldType::value_type, N>(f(i));
}

#define MUNDY_IMPL_VECTOR_FIELD_DATA_N(N)                                                               \
  template <class FieldType>                                                                            \
  KOKKOS_INLINE_FUNCTION auto vector##N##_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) { \
    return get_vector<typename FieldType::value_type, N>(f(i));                                  \
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
  return get_matrix<typename FieldType::value_type, N, M>(f(i));
}

#define MUNDY_IMPL_MATRIX_FIELD_DATA_NM(N, M)                                                                \
  template <class FieldType>                                                                                 \
  KOKKOS_INLINE_FUNCTION auto matrix##N####M##_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) { \
    return get_matrix<typename FieldType::value_type, N, M>(f(i));                                    \
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
    return get_matrix<typename FieldType::value_type, N, N>(f(i));                               \
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
  return get_quaternion<typename FieldType::value_type>(f(i));
}

/// \brief Get a view of a field's data as an AABB. 6 scalars per entity.
/// Layout: min corner xyz (0-2), max corner xyz (3-5).
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto aabb_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  constexpr size_t shift = 3;
  using value_type = typename FieldType::value_type;
  auto shifted_data_accessor = get_shifted_accessor<value_type, shift>(f(i));
  auto max_corner = get_vector3<value_type>(std::move(shifted_data_accessor));
  auto min_corner = get_vector3<value_type>(f(i));

  using min_point_t = decltype(min_corner);
  using max_point_t = decltype(max_corner);
  return AABB<value_type, min_point_t, max_point_t>(min_corner, max_corner);
}

/// \brief Get a view of a field's NGP data as an OBB. 10 scalars per entity.
/// Layout: center xyz (0-2), orientation quaternion wxyz (3-6), half-extents xyz (7-9).
template <class FieldType>
KOKKOS_INLINE_FUNCTION auto obb_field_data(FieldType& f, const stk::mesh::FastMeshIndex& i) {
  using value_type  = typename FieldType::value_type;
  auto center       = get_vector3<value_type>(f(i));
  auto orientation  = get_quaternion<value_type>(get_shifted_accessor<value_type, 3>(f(i)));
  auto half_extents = get_vector3<value_type>(get_shifted_accessor<value_type, 7>(f(i)));
  using center_t    = decltype(center);
  using orient_t    = decltype(orientation);
  using he_t        = decltype(half_extents);
  return OBB<value_type, center_t, orient_t, he_t>(center, orientation, half_extents);
}
//@}

/// @}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FIELDVIEWS_HPP_
