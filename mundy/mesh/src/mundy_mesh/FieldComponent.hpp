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

#ifndef MUNDY_MESH_FIELDCOMPONENT_HPP_
#define MUNDY_MESH_FIELDCOMPONENT_HPP_

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
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_mesh/Component.hpp>
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/ForEachEntity.hpp>    // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/NgpAccessorExpr.hpp>  // for mundy::mesh::AccessorExpr and EntityExprBase
#include <mundy_mesh/fmt_stk_types.hpp>    // for STK-compatible fmt::format
#include <mundy_mesh/impl/HostDeviceSynchronizer.hpp>
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/throw_assert.hpp>       // for MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>              // for mundy::tuple

namespace mundy {

namespace mesh {

class FieldComponentBase {
 public:
  FieldComponentBase(const stk::mesh::FieldBase& field_base) : field_base_(field_base) {
  }

  /// \brief Default copy/move/assign constructors
  FieldComponentBase(const FieldComponentBase&) = default;
  FieldComponentBase(FieldComponentBase&&) = default;
  FieldComponentBase& operator=(const FieldComponentBase&) = default;
  FieldComponentBase& operator=(FieldComponentBase&&) = default;

  // clang-format off
  void sync_to_device() { field_base_.sync_to_device(); }
  void sync_to_host() { field_base_.sync_to_host(); }
  void modify_on_device() { field_base_.modify_on_device(); }
  void modify_on_host() { field_base_.modify_on_host(); }
  void clear_host_sync_state() { field_base_.clear_host_sync_state(); }
  void clear_device_sync_state() { field_base_.clear_device_sync_state(); }
  const stk::mesh::FieldBase& field_base() const { return field_base_; }
  // clang-format on

 private:
  const stk::mesh::FieldBase& field_base_;
};  // FieldComponentBase

class NgpFieldComponentBase {
 public:
  NgpFieldComponentBase() = default;

#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  NgpFieldComponentBase(const stk::mesh::FieldBase& field_base) : host_field_base_(&field_base) {
  }

  /// \brief Default copy/move/assign constructors
  NgpFieldComponentBase(const NgpFieldComponentBase&) = default;
  NgpFieldComponentBase(NgpFieldComponentBase&&) = default;
  NgpFieldComponentBase& operator=(const NgpFieldComponentBase&) = default;
  NgpFieldComponentBase& operator=(NgpFieldComponentBase&&) = default;

  // clang-format off
  void sync_to_device() { host_field_base().sync_to_device(); }
  void sync_to_host() { host_field_base().sync_to_host(); }
  void modify_on_device() { host_field_base().modify_on_device(); }
  void modify_on_host() { host_field_base().modify_on_host(); }
  void clear_host_sync_state() { host_field_base().clear_host_sync_state(); }
  void clear_device_sync_state() { host_field_base().clear_device_sync_state(); }
  // clang-format on

  const stk::mesh::FieldBase& host_field_base() const {
    MUNDY_THROW_ASSERT(host_field_base_, std::runtime_error, "host_field_base_ is null");
    return *host_field_base_;
  }

 private:
  const stk::mesh::FieldBase* host_field_base_ = nullptr;
#endif
};  // NgpFieldComponentBase

namespace impl {

struct FieldDataAccessPolicy {
  template <typename FieldType>
  static decltype(auto) host_access(FieldType& field, stk::mesh::Entity entity) {
    using value_type = typename std::remove_cv_t<FieldType>::value_type;
    value_type* data_ptr = stk::mesh::field_data(field, entity);
    MUNDY_THROW_ASSERT(data_ptr, std::runtime_error, "Field data is null");
    unsigned num_scalars = stk::mesh::field_scalars_per_entity(field, entity);
    return stk::mesh::EntityFieldData<value_type>(data_ptr, num_scalars);
  }

  template <typename FieldType>
  KOKKOS_INLINE_FUNCTION static decltype(auto) ngp_access(FieldType& field, stk::mesh::FastMeshIndex entity_index) {
    return field(entity_index);
  }
};

struct ScalarFieldAccessPolicy {
  template <typename FieldType>
  static decltype(auto) host_access(FieldType& field, stk::mesh::Entity entity) {
    return scalar_field_data(field, entity);
  }

  template <typename FieldType>
  KOKKOS_INLINE_FUNCTION static decltype(auto) ngp_access(FieldType& field, stk::mesh::FastMeshIndex entity_index) {
    return scalar_field_data(field, entity_index);
  }
};

template <size_t N>
struct VectorFieldAccessPolicy {
  template <typename FieldType>
  static decltype(auto) host_access(FieldType& field, stk::mesh::Entity entity) {
    return vector_field_data<N>(field, entity);
  }

  template <typename FieldType>
  KOKKOS_INLINE_FUNCTION static decltype(auto) ngp_access(FieldType& field, stk::mesh::FastMeshIndex entity_index) {
    return vector_field_data<N>(field, entity_index);
  }
};

struct Matrix3FieldAccessPolicy {
  template <typename FieldType>
  static decltype(auto) host_access(FieldType& field, stk::mesh::Entity entity) {
    return matrix3_field_data(field, entity);
  }

  template <typename FieldType>
  KOKKOS_INLINE_FUNCTION static decltype(auto) ngp_access(FieldType& field, stk::mesh::FastMeshIndex entity_index) {
    return matrix3_field_data(field, entity_index);
  }
};

struct QuaternionFieldAccessPolicy {
  template <typename FieldType>
  static decltype(auto) host_access(FieldType& field, stk::mesh::Entity entity) {
    return quaternion_field_data(field, entity);
  }

  template <typename FieldType>
  KOKKOS_INLINE_FUNCTION static decltype(auto) ngp_access(FieldType& field, stk::mesh::FastMeshIndex entity_index) {
    return quaternion_field_data(field, entity_index);
  }
};

struct AABBFieldAccessPolicy {
  template <typename FieldType>
  static decltype(auto) host_access(FieldType& field, stk::mesh::Entity entity) {
    return aabb_field_data(field, entity);
  }

  template <typename FieldType>
  KOKKOS_INLINE_FUNCTION static decltype(auto) ngp_access(FieldType& field, stk::mesh::FastMeshIndex entity_index) {
    return aabb_field_data(field, entity_index);
  }
};

template <typename ScalarType, typename AccessPolicy>
class FieldComponent : public FieldComponentBase {
 public:
  using field_type = stk::mesh::Field<ScalarType>;
  using access_policy = AccessPolicy;
  using view_t = decltype(access_policy::host_access(std::declval<field_type&>(), std::declval<stk::mesh::Entity>()));

  explicit FieldComponent(field_type& field) : FieldComponentBase(field), field_(field) {
  }

  FieldComponent(const FieldComponent&) = default;
  FieldComponent(FieldComponent&&) = default;
  FieldComponent& operator=(const FieldComponent&) = delete;
  FieldComponent& operator=(FieldComponent&&) = delete;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return access_policy::host_access(field_, entity);
  }

  // clang-format off
  inline       field_type& field()       { return field_; }
  inline const field_type& field() const { return field_; }
  // clang-format on

 private:
  field_type& field_;
};  // FieldComponent

template <typename NgpFieldType, typename AccessPolicy>
class NgpFieldComponent : public NgpFieldComponentBase {
 public:
  using field_type = NgpFieldType;
  using access_policy = AccessPolicy;
  using view_t =
      decltype(access_policy::ngp_access(std::declval<field_type&>(), std::declval<stk::mesh::FastMeshIndex>()));

  NgpFieldComponent() = default;
  explicit NgpFieldComponent(field_type ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),
#else
      : NgpFieldComponentBase(),
#endif
        ngp_field_(ngp_field) {
  }

  NgpFieldComponent(const NgpFieldComponent&) = default;
  NgpFieldComponent(NgpFieldComponent&&) = default;
  NgpFieldComponent& operator=(const NgpFieldComponent&) = default;
  NgpFieldComponent& operator=(NgpFieldComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return access_policy::ngp_access(ngp_field_, entity_index);
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION       field_type& ngp_field()       { return ngp_field_; }
  KOKKOS_INLINE_FUNCTION const field_type& ngp_field() const { return ngp_field_; }

#if TRILINOS_MAJOR_MINOR_VERSION < 160000
  void sync_to_device() { ngp_field_.sync_to_device(); }
  void sync_to_host() { ngp_field_.sync_to_host(); }
  void modify_on_device() { ngp_field_.modify_on_device(); }
  void modify_on_host() { ngp_field_.modify_on_host(); }
  void clear_host_sync_state() { ngp_field_.clear_host_sync_state(); }
  void clear_device_sync_state() { ngp_field_.clear_device_sync_state(); }
#endif
  // clang-format on

 private:
  field_type ngp_field_;
};  // NgpFieldComponent

template <typename ComponentType>
  requires requires(ComponentType& component) { component.field(); }
inline decltype(auto) component_backing_field(ComponentType& component) {
  return component.field();
}

template <typename ComponentType>
  requires requires(const ComponentType& component) { component.field(); }
inline decltype(auto) component_backing_field(const ComponentType& component) {
  return component.field();
}

template <typename Tag, typename ComponentType>
  requires requires(ComponentType& component) { component_backing_field(component); }
inline decltype(auto) component_backing_field(TaggedComponent<Tag, ComponentType>& tagged_component) {
  return component_backing_field(tagged_component.component());
}

template <typename Tag, typename ComponentType>
  requires requires(const ComponentType& component) { component_backing_field(component); }
inline decltype(auto) component_backing_field(const TaggedComponent<Tag, ComponentType>& tagged_component) {
  return component_backing_field(tagged_component.component());
}

}  // namespace impl

template <typename ValueType>
class FieldComponent : public impl::FieldComponent<ValueType, impl::FieldDataAccessPolicy> {
 public:
  using our_t = FieldComponent<ValueType>;
  using base_t = impl::FieldComponent<ValueType, impl::FieldDataAccessPolicy>;
  using canonical_access = access::raw<ValueType>;
  using view_t = typename base_t::view_t;

  explicit FieldComponent(stk::mesh::Field<ValueType>& field) : base_t(field) {
  }
};  // FieldComponent

template <typename NgpFieldType>
class NgpFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::FieldDataAccessPolicy> {
 public:
  using our_t = NgpFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::FieldDataAccessPolicy>;
  using canonical_access = access::raw<typename NgpFieldType::value_type>;
  using view_t = typename base_t::view_t;

  NgpFieldComponent() = default;
  explicit NgpFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpFieldComponent

template <typename ScalarType>
class ScalarFieldComponent : public impl::FieldComponent<ScalarType, impl::ScalarFieldAccessPolicy> {
 public:
  using our_t = ScalarFieldComponent<ScalarType>;
  using base_t = impl::FieldComponent<ScalarType, impl::ScalarFieldAccessPolicy>;
  using canonical_access = access::scalar<ScalarType>;
  using view_t = typename base_t::view_t;

  explicit ScalarFieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // ScalarFieldComponent

template <typename NgpFieldType>
class NgpScalarFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::ScalarFieldAccessPolicy> {
 public:
  using our_t = NgpScalarFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::ScalarFieldAccessPolicy>;
  using canonical_access = access::scalar<typename NgpFieldType::value_type>;
  using view_t = typename base_t::view_t;

  NgpScalarFieldComponent() = default;
  explicit NgpScalarFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpScalarFieldComponent

template <typename ScalarType, size_t N>
class VectorFieldComponent : public impl::FieldComponent<ScalarType, impl::VectorFieldAccessPolicy<N>> {
 public:
  using our_t = VectorFieldComponent<ScalarType, N>;
  using base_t = impl::FieldComponent<ScalarType, impl::VectorFieldAccessPolicy<N>>;
  using canonical_access = access::vector<ScalarType, N>;
  using view_t = typename base_t::view_t;

  explicit VectorFieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // VectorFieldComponent

template <typename ScalarType>
using Vector1FieldComponent = VectorFieldComponent<ScalarType, 1>;

template <typename ScalarType>
using Vector2FieldComponent = VectorFieldComponent<ScalarType, 2>;

template <typename ScalarType>
using Vector3FieldComponent = VectorFieldComponent<ScalarType, 3>;

template <typename ScalarType>
using Vector4FieldComponent = VectorFieldComponent<ScalarType, 4>;

template <typename ScalarType>
using Vector5FieldComponent = VectorFieldComponent<ScalarType, 5>;

template <typename ScalarType>
using Vector6FieldComponent = VectorFieldComponent<ScalarType, 6>;

template <typename NgpFieldType, size_t N>
class NgpVectorFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::VectorFieldAccessPolicy<N>> {
 public:
  using our_t = NgpVectorFieldComponent<NgpFieldType, N>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::VectorFieldAccessPolicy<N>>;
  using canonical_access = access::vector<typename NgpFieldType::value_type, N>;
  using view_t = typename base_t::view_t;

  NgpVectorFieldComponent() = default;
  explicit NgpVectorFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpVectorFieldComponent

template <typename NgpFieldType>
using NgpVector1FieldComponent = NgpVectorFieldComponent<NgpFieldType, 1>;

template <typename NgpFieldType>
using NgpVector2FieldComponent = NgpVectorFieldComponent<NgpFieldType, 2>;

template <typename NgpFieldType>
using NgpVector3FieldComponent = NgpVectorFieldComponent<NgpFieldType, 3>;

template <typename NgpFieldType>
using NgpVector4FieldComponent = NgpVectorFieldComponent<NgpFieldType, 4>;

template <typename NgpFieldType>
using NgpVector5FieldComponent = NgpVectorFieldComponent<NgpFieldType, 5>;

template <typename NgpFieldType>
using NgpVector6FieldComponent = NgpVectorFieldComponent<NgpFieldType, 6>;

template <typename ScalarType>
class Matrix3FieldComponent : public impl::FieldComponent<ScalarType, impl::Matrix3FieldAccessPolicy> {
 public:
  using base_t = impl::FieldComponent<ScalarType, impl::Matrix3FieldAccessPolicy>;
  using canonical_access = access::matrix3<ScalarType>;
  using view_t = typename base_t::view_t;

  explicit Matrix3FieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // Matrix3FieldComponent

template <typename NgpFieldType>
class NgpMatrix3FieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::Matrix3FieldAccessPolicy> {
 public:
  using our_t = NgpMatrix3FieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::Matrix3FieldAccessPolicy>;
  using canonical_access = access::matrix3<typename NgpFieldType::value_type>;
  using view_t = typename base_t::view_t;

  NgpMatrix3FieldComponent() = default;
  explicit NgpMatrix3FieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpMatrix3FieldComponent

template <typename ScalarType>
class QuaternionFieldComponent : public impl::FieldComponent<ScalarType, impl::QuaternionFieldAccessPolicy> {
 public:
  using base_t = impl::FieldComponent<ScalarType, impl::QuaternionFieldAccessPolicy>;
  using canonical_access = access::quaternion<ScalarType>;
  using view_t = typename base_t::view_t;

  explicit QuaternionFieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // QuaternionFieldComponent

template <typename NgpFieldType>
class NgpQuaternionFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::QuaternionFieldAccessPolicy> {
 public:
  using our_t = NgpQuaternionFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::QuaternionFieldAccessPolicy>;
  using canonical_access = access::quaternion<typename NgpFieldType::value_type>;
  using view_t = typename base_t::view_t;

  NgpQuaternionFieldComponent() = default;
  explicit NgpQuaternionFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpQuaternionFieldComponent

template <typename ScalarType>
class AABBFieldComponent : public impl::FieldComponent<ScalarType, impl::AABBFieldAccessPolicy> {
 public:
  using base_t = impl::FieldComponent<ScalarType, impl::AABBFieldAccessPolicy>;
  using canonical_access = access::aabb<ScalarType>;
  using view_t = typename base_t::view_t;

  explicit AABBFieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // AABBFieldComponent

template <typename NgpFieldType>
class NgpAABBFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::AABBFieldAccessPolicy> {
 public:
  using our_t = NgpAABBFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::AABBFieldAccessPolicy>;
  using canonical_access = access::aabb<typename NgpFieldType::value_type>;
  using view_t = typename base_t::view_t;

  NgpAABBFieldComponent() = default;
  explicit NgpAABBFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpAABBFieldComponent

/// \brief A helper function for getting the NGP component from a regular component.
///
/// Field-backed components simply wrap STK's updated ngp field and return the wrapper by value.
/// SharedComponent follows the LinkData pattern instead, lazily materializing and caching a
/// memspace-specific NgpSharedComponent in a std::any owned by the host component state.
///
/// TODO(palmerb4): This function has type loss. We want the true object to map to its corresponding NGP counterpart, so
/// we can't just allow it to slice to its Base and then map to a general NgpFieldComponent. We need one overload per
/// component type. OR, we need a map from given Component to its NGP counterpart. I prefer that actually. That's
/// generally useful. Add such a map to Component.hpp itself and specialize it for field and shared components to allow
/// them to all have a single unified get_updated_ngp_component function.
template <typename ScalarType, typename AccessPolicy>
auto get_updated_ngp_component(const impl::FieldComponent<ScalarType, AccessPolicy>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ScalarType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return impl::NgpFieldComponent<ngp_field_type, AccessPolicy>(ngp_field);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FIELDCOMPONENT_HPP_
