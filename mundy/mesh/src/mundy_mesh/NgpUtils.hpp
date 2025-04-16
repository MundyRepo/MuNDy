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

#ifndef MUNDY_MESH_NGPUTILS_HPP_
#define MUNDY_MESH_NGPUTILS_HPP_

/// \file NgpUtils.hpp
/// \brief A set of utilities for working with stk::mesh::NgpField objects

// C++ core
#include <type_traits>

// Trilinos
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

namespace mundy {

namespace mesh {

template <typename Field>
constexpr bool is_device_field = false;

template <typename Field>
constexpr bool is_host_field = false;

#if TRILINOS_MAJOR_MINOR_VERSION > 160000
template <typename T, typename NgpMemSpace, template <typename, typename> class NgpDebugger>
constexpr bool is_device_field<stk::mesh::DeviceField<T, NgpMemSpace, NgpDebugger>> = true;

template <typename T, typename NgpMemSpace, template <typename, typename> class NgpDebugger>
constexpr bool is_device_field<const stk::mesh::DeviceField<T, NgpMemSpace, NgpDebugger>> = true;

template <typename T, typename NgpMemSpace, template <typename, typename> class NgpDebugger>
constexpr bool is_host_field<stk::mesh::HostField<T, NgpMemSpace, NgpDebugger>> = true;

template <typename T, typename NgpMemSpace, template <typename, typename> class NgpDebugger>
constexpr bool is_host_field<const stk::mesh::HostField<T, NgpMemSpace, NgpDebugger>> = true;
#else 
template <typename T, template <typename> class NgpDebugger>
constexpr bool is_device_field<stk::mesh::DeviceField<T, NgpDebugger>> = true;

template <typename T, template <typename> class NgpDebugger>
constexpr bool is_device_field<const stk::mesh::DeviceField<T, NgpDebugger>> = true;

template <typename T, template <typename> class NgpDebugger>
constexpr bool is_host_field<stk::mesh::HostField<T, NgpDebugger>> = true;

template <typename T, template <typename> class NgpDebugger>

constexpr bool is_host_field<const stk::mesh::HostField<T, NgpDebugger>> = true;
#endif

template <typename Field>
constexpr bool is_ngp_field = is_device_field<Field> || is_host_field<Field>;

template <typename Mesh>
constexpr bool is_device_mesh = std::is_base_of_v<stk::mesh::DeviceMesh, Mesh>;

template <typename Mesh>
constexpr bool is_host_mesh = std::is_base_of_v<stk::mesh::HostMesh, Mesh>;

template <typename Mesh>
constexpr bool is_ngp_mesh = is_device_mesh<Mesh> || is_host_mesh<Mesh>;

// For a field nd mesh to be compatible they must match in host/device status
template <typename Mesh, typename Field>
constexpr bool ngp_field_and_mesh_compatible =
    (is_host_field<Field> == is_host_mesh<Mesh>) && (is_device_field<Field> == is_device_mesh<Mesh>);

template <typename Field>
void sync_field_to_owning_space(Field& field) {
  static_assert(is_ngp_field<Field>, "Field must be an stk::mesh::NgpField");
  if constexpr (is_device_field<Field>) {
    field.sync_to_device();
  } else {
    field.sync_to_host();
  }
}

template <typename ExecSpace>
  requires Kokkos::is_execution_space<ExecSpace>::value
void sync_field_to_space(const stk::mesh::FieldBase& field, [[maybe_unused]] const ExecSpace& exec_space) {
  constexpr bool is_device_exec_space =
      !Kokkos::SpaceAccessibility<ExecSpace, stk::ngp::HostExecSpace::memory_space>::accessible;
  if constexpr (is_device_exec_space) {
    field.sync_to_device();
  } else {
    field.sync_to_host();
  }
}

template <typename Field>
void mark_field_modified_on_owning_space(Field& field) {
  static_assert(is_ngp_field<Field>, "Field must be an stk::mesh::NgpField");
  field.clear_sync_state();
  if constexpr (is_device_field<Field>) {
    field.modify_on_device();
  } else {
    field.modify_on_host();
  }
}

template <typename ExecSpace>
  requires Kokkos::is_execution_space<ExecSpace>::value
void mark_field_modified_on_space(const stk::mesh::FieldBase& field, [[maybe_unused]] const ExecSpace& exec_space) {
  constexpr bool is_device_exec_space =
      !Kokkos::SpaceAccessibility<ExecSpace, stk::ngp::HostExecSpace::memory_space>::accessible;
  field.clear_sync_state();
  if constexpr (is_device_exec_space) {
    field.modify_on_device();
  } else {
    field.modify_on_host();
  }
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPUTILS_HPP_
