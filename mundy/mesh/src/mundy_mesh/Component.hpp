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

#ifndef MUNDY_MESH_COMPONENT_HPP_
#define MUNDY_MESH_COMPONENT_HPP_

// C++ core
#include <any>
#include <concepts>
#include <memory>
#include <tuple>
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type
#include <utility>      // for std::declval

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// Trilinos
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

// STK mesh
#include <stk_io/StkMeshIoBroker.hpp>     // for stk::io::FieldOutputType
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_topology/topology.hpp>      // for stk::topology::topology_t
#include <stk_util/ngp/NgpSpaces.hpp>     // for stk::ngp::MemSpace

// Mundy
#include <mundy_mesh/BulkData.hpp>            // for mundy::mesh::BulkData
#include <mundy_mesh/ComponentAccess.hpp>     // for access::*, canonical_component_access_t, component_access_shape
#include <mundy_mesh/FieldViews.hpp>          // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/ForEachEntity.hpp>       // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/NgpAccessorExpr.hpp>     // for mundy::mesh::AccessorExpr and EntityExprBase
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/throw_assert.hpp>       // for MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>              // for mundy::tuple

namespace mundy {

namespace mesh {

//! \name Our Tags (types never need to be complete)
//@{

struct CENTER;
struct POSITION;

struct RADIUS;
struct COLLISION_RADIUS;
struct HYDRO_RADIUS;

struct ORIENT;
struct DIRECTION;

struct LIN_VEL;
struct ANG_VEL;
struct VELOCITY;
struct OMEGA;

struct FORCE;
struct TORQUE;
struct MASS;
struct DENSITY;

struct RNG_COUNTER;
struct LINKED_ENTITIES;
//@}

template <typename SharedType, typename NgpMemSpace>
class NgpSharedComponent;

/// \brief A small helper type for tying a Tag to an underlying component
template <typename Tag, typename ComponentType>
class TaggedComponent {
 public:
  using our_t = TaggedComponent<Tag, ComponentType>;
  using view_t = typename ComponentType::view_t;
  using tag_type = Tag;
  using component_type = ComponentType;
  using canonical_access = typename component_type::canonical_access;

  TaggedComponent() = default;
  TaggedComponent(component_type component) : component_(component) {
  }

  /// \brief Default copy/move/assign constructors
  TaggedComponent(const TaggedComponent&) = default;
  TaggedComponent(TaggedComponent&&) = default;
  TaggedComponent& operator=(const TaggedComponent&) = default;
  TaggedComponent& operator=(TaggedComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return component_(entity);
  }

  /// \brief Calling operator()(entity_expr) on any accessor will return an AccessorExpr
  /// Example:
  ///   auto v3_accessor = Vector3FieldComponent(v3_field);
  ///   EntityExpr all_nodes(node_selector, stk::topology::NODE_RANK);
  ///   auto get_v3_expr = v3_accessor(all_nodes);
  template <class EntityExpr>
  auto operator()(const impl::EntityExprBase<EntityExpr>& e) const;

  inline const component_type& component() const {
    // Our lifetime should be at least as long as the component's
    return component_;
  }

  inline component_type& component() {
    return component_;
  }

  void sync_to_device() {
    component_.sync_to_device();
  }

  void sync_to_host() {
    component_.sync_to_host();
  }

  void modify_on_device() {
    component_.modify_on_device();
  }

  void modify_on_host() {
    component_.modify_on_host();
  }

  void clear_host_sync_state() {
    component_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    component_.clear_device_sync_state();
  }

 private:
  component_type component_;
};  // TaggedComponent

template <typename Tag, typename ComponentType>
auto make_tagged_component(ComponentType component) {
  return TaggedComponent<Tag, ComponentType>(component);
}

/// \brief A small helper type for tying a Tag to an underlying ngp-compatible component
template <typename Tag, typename NgpComponentType>
class NgpTaggedComponent {
 public:
  using our_t = NgpTaggedComponent<Tag, NgpComponentType>;
  using view_t = typename NgpComponentType::view_t;
  using tag_type = Tag;
  using component_type = NgpComponentType;

  NgpTaggedComponent() = default;
  NgpTaggedComponent(component_type component) : component_(component) {
  }

  /// \brief Default copy/move/assign constructors
  NgpTaggedComponent(const NgpTaggedComponent&) = default;
  NgpTaggedComponent(NgpTaggedComponent&&) = default;
  NgpTaggedComponent& operator=(const NgpTaggedComponent&) = default;
  NgpTaggedComponent& operator=(NgpTaggedComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return component_(entity_index);
  }

  /// \brief Calling operator()(entity_expr) on any accessor will return an AccessorExpr
  /// Example:
  ///   auto v3_accessor = Vector3FieldComponent(v3_field);
  ///   EntityExpr all_nodes(node_selector, stk::topology::NODE_RANK);
  ///   auto get_v3_expr = v3_accessor(all_nodes);
  template <class EntityExpr>
  auto operator()(const impl::EntityExprBase<EntityExpr>& e) const {
    return impl::AccessorExpr<our_t, EntityExpr>(*this, e.self());
  }

  KOKKOS_INLINE_FUNCTION
  const component_type& component() const {
    return component_;
  }

  KOKKOS_INLINE_FUNCTION
  component_type& component() {
    return component_;
  }

  void sync_to_device() {
    component_.sync_to_device();
  }

  void sync_to_host() {
    component_.sync_to_host();
  }

  void modify_on_device() {
    component_.modify_on_device();
  }

  void modify_on_host() {
    component_.modify_on_host();
  }

  void clear_host_sync_state() {
    component_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    component_.clear_device_sync_state();
  }

 private:
  component_type component_;
};  // NgpTaggedComponent

template <typename Tag, typename ComponentType>
decltype(auto) get_updated_ngp_component(const TaggedComponent<Tag, ComponentType>& tagged_component) {
  auto ngp_component = get_updated_ngp_component(tagged_component.component());
  using ngp_component_type = std::remove_reference_t<decltype(ngp_component)>;
  return NgpTaggedComponent<Tag, ngp_component_type>(ngp_component);
}

template <typename Tag, typename ComponentType>
template <class EntityExpr>
auto TaggedComponent<Tag, ComponentType>::operator()(const impl::EntityExprBase<EntityExpr>& e) const {
  // Entity expressions are (currently) always on the device, so we need to get the NGP tagged component
  // TODO(palmerb4): Allow for exec_spaces that aren't simply the default execution space (need Tril 16.1+)
  auto ngp_this = get_updated_ngp_component(*this);
  return ngp_this(e.self());
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_COMPONENT_HPP_
