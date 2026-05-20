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

#ifndef MUNDY_MESH_SHAREDCOMPONENT_HPP_
#define MUNDY_MESH_SHAREDCOMPONENT_HPP_

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
#include <mundy_mesh/impl/HostDeviceSynchronizer.hpp>
#include <mundy_mesh/impl/SharedComponentImpl.hpp>
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/throw_assert.hpp>       // for MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>              // for mundy::tuple
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

/// \brief A component that returns the same shared value for every entity.
///
/// Construct either from:
///   - a raw `SharedType`, which is copied into owned HostSpace storage
///   - a rank-1 Kokkos::View in HostSpace with extent 1, which is aliased exactly as given whether that view is
///   managed or unmanaged
template <typename SharedType>
class SharedComponent {
 public:
  static_assert(!std::is_reference_v<SharedType>, "SharedComponent may not store a reference type.");

  using our_t = SharedComponent<SharedType>;
  using canonical_access = access::raw<SharedType>;
  using shared_type = SharedType;
  using view_t = shared_type&;

  SharedComponent() = default;
  explicit SharedComponent(shared_type shared_value) : state_(std::make_shared<state_type>(std::move(shared_value))) {
  }

  template <typename HostViewType>
    MUNDY_REQUIRES(impl::CompatibleSharedComponentHostView<HostViewType, shared_type>)
  explicit SharedComponent(HostViewType host_view) : state_(std::make_shared<state_type>(std::move(host_view))) {
  }

  SharedComponent(const SharedComponent&) = default;
  SharedComponent(SharedComponent&&) = default;
  SharedComponent& operator=(const SharedComponent&) = default;
  SharedComponent& operator=(SharedComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity /*entity*/) const {
    return state().host_value();
  }

  // clang-format off
  inline       shared_type& shared_value()       { return state().host_value(); }
  inline const shared_type& shared_value() const { return state().host_value(); }
  // clang-format on

  void sync_to_device() {
    state().sync_to_device();
  }

  void sync_to_host() {
    state().sync_to_host();
  }

  void modify_on_device() {
    state().modify_on_device();
  }

  void modify_on_host() {
    state().modify_on_host();
  }

  void clear_host_sync_state() {
    state().clear_host_sync_state();
  }

  void clear_device_sync_state() {
    state().clear_device_sync_state();
  }

 private:
  using state_type = impl::SharedComponentState<shared_type>;
  using host_view_type = typename state_type::host_view_type;

  state_type& state() const {
    MUNDY_THROW_ASSERT(state_, std::runtime_error, "SharedComponent state is null");
    return *state_;
  }

  host_view_type host_view() const {
    return state().host_view();
  }

  std::any& any_ngp_component() const {
    return state().any_ngp_component();
  }

  void set_synchronizer(std::shared_ptr<impl::HostDeviceSynchronizer> synchronizer) const {
    state().set_synchronizer(std::move(synchronizer));
  }

  std::shared_ptr<state_type> state_;

  template <typename NgpMemSpace, typename OtherSharedType>
  friend NgpSharedComponent<OtherSharedType, NgpMemSpace>& get_updated_ngp_component(
      const SharedComponent<OtherSharedType>& component);
};  // SharedComponent

template <typename ScalarType>
class SharedScalarComponent : public SharedComponent<ScalarType> {
 public:
  using our_t = SharedScalarComponent<ScalarType>;
  using base_t = SharedComponent<ScalarType>;
  using canonical_access = access::scalar<ScalarType>;
  using view_t = decltype(get_scalar_view<ScalarType>(std::declval<ScalarType*>()));

  SharedScalarComponent() = default;
  explicit SharedScalarComponent(ScalarType shared_value) : base_t(std::move(shared_value)) {
  }

  template <typename HostViewType>
    MUNDY_REQUIRES(impl::CompatibleSharedComponentHostView<HostViewType, ScalarType>)
  explicit SharedScalarComponent(HostViewType host_view) : base_t(std::move(host_view)) {
  }

  SharedScalarComponent(const SharedScalarComponent&) = default;
  SharedScalarComponent(SharedScalarComponent&&) = default;
  SharedScalarComponent& operator=(const SharedScalarComponent&) = default;
  SharedScalarComponent& operator=(SharedScalarComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity /*entity*/) const {
    auto& value = static_cast<our_t&>(*const_cast<our_t*>(this)).shared_value();
    return get_scalar_view<ScalarType>(&value);
  }
};  // SharedScalarComponent

template <typename ScalarType, size_t N>
class SharedVectorComponent : public SharedComponent<Vector<ScalarType, N>> {
 public:
  using shared_value_type = Vector<ScalarType, N>;
  using our_t = SharedVectorComponent<ScalarType, N>;
  using base_t = SharedComponent<shared_value_type>;
  using canonical_access = access::vector<ScalarType, N>;
  using view_t = typename base_t::view_t;

  SharedVectorComponent() = default;
  explicit SharedVectorComponent(shared_value_type shared_value) : base_t(std::move(shared_value)) {
  }

  template <typename HostViewType>
    MUNDY_REQUIRES(impl::CompatibleSharedComponentHostView<HostViewType, shared_value_type>)
  explicit SharedVectorComponent(HostViewType host_view) : base_t(std::move(host_view)) {
  }

  SharedVectorComponent(const SharedVectorComponent&) = default;
  SharedVectorComponent(SharedVectorComponent&&) = default;
  SharedVectorComponent& operator=(const SharedVectorComponent&) = default;
  SharedVectorComponent& operator=(SharedVectorComponent&&) = default;
};  // SharedVectorComponent

template <typename ScalarType>
using SharedVector1Component = SharedVectorComponent<ScalarType, 1>;

template <typename ScalarType>
using SharedVector2Component = SharedVectorComponent<ScalarType, 2>;

template <typename ScalarType>
using SharedVector3Component = SharedVectorComponent<ScalarType, 3>;

template <typename ScalarType>
using SharedVector4Component = SharedVectorComponent<ScalarType, 4>;

template <typename ScalarType>
using SharedVector5Component = SharedVectorComponent<ScalarType, 5>;

template <typename ScalarType>
using SharedVector6Component = SharedVectorComponent<ScalarType, 6>;

template <typename ScalarType>
class SharedMatrix3Component : public SharedComponent<Matrix3<ScalarType>> {
 public:
  using shared_value_type = Matrix3<ScalarType>;
  using our_t = SharedMatrix3Component<ScalarType>;
  using base_t = SharedComponent<shared_value_type>;
  using canonical_access = access::matrix3<ScalarType>;
  using view_t = typename base_t::view_t;

  SharedMatrix3Component() = default;
  explicit SharedMatrix3Component(shared_value_type shared_value) : base_t(std::move(shared_value)) {
  }

  template <typename HostViewType>
    MUNDY_REQUIRES(impl::CompatibleSharedComponentHostView<HostViewType, shared_value_type>)
  explicit SharedMatrix3Component(HostViewType host_view) : base_t(std::move(host_view)) {
  }

  SharedMatrix3Component(const SharedMatrix3Component&) = default;
  SharedMatrix3Component(SharedMatrix3Component&&) = default;
  SharedMatrix3Component& operator=(const SharedMatrix3Component&) = default;
  SharedMatrix3Component& operator=(SharedMatrix3Component&&) = default;
};  // SharedMatrix3Component

template <typename ScalarType>
class SharedQuaternionComponent : public SharedComponent<Quaternion<ScalarType>> {
 public:
  using shared_value_type = Quaternion<ScalarType>;
  using our_t = SharedQuaternionComponent<ScalarType>;
  using base_t = SharedComponent<shared_value_type>;
  using canonical_access = access::quaternion<ScalarType>;
  using view_t = typename base_t::view_t;

  SharedQuaternionComponent() = default;
  explicit SharedQuaternionComponent(shared_value_type shared_value) : base_t(std::move(shared_value)) {
  }

  template <typename HostViewType>
    MUNDY_REQUIRES(impl::CompatibleSharedComponentHostView<HostViewType, shared_value_type>)
  explicit SharedQuaternionComponent(HostViewType host_view) : base_t(std::move(host_view)) {
  }

  SharedQuaternionComponent(const SharedQuaternionComponent&) = default;
  SharedQuaternionComponent(SharedQuaternionComponent&&) = default;
  SharedQuaternionComponent& operator=(const SharedQuaternionComponent&) = default;
  SharedQuaternionComponent& operator=(SharedQuaternionComponent&&) = default;
};  // SharedQuaternionComponent

template <typename ScalarType>
class SharedAABBComponent : public SharedComponent<AABB<ScalarType>> {
 public:
  using shared_value_type = AABB<ScalarType>;
  using our_t = SharedAABBComponent<ScalarType>;
  using base_t = SharedComponent<shared_value_type>;
  using canonical_access = access::aabb<ScalarType>;
  using view_t = typename base_t::view_t;

  SharedAABBComponent() = default;
  explicit SharedAABBComponent(shared_value_type shared_value) : base_t(std::move(shared_value)) {
  }

  template <typename HostViewType>
    MUNDY_REQUIRES(impl::CompatibleSharedComponentHostView<HostViewType, shared_value_type>)
  explicit SharedAABBComponent(HostViewType host_view) : base_t(std::move(host_view)) {
  }

  SharedAABBComponent(const SharedAABBComponent&) = default;
  SharedAABBComponent(SharedAABBComponent&&) = default;
  SharedAABBComponent& operator=(const SharedAABBComponent&) = default;
  SharedAABBComponent& operator=(SharedAABBComponent&&) = default;
};  // SharedAABBComponent

template <typename SharedType, typename NgpMemSpace>
class NgpSharedComponent {
 public:
  static_assert(!std::is_reference_v<SharedType>, "NgpSharedComponent may not store a reference type.");
  static_assert(Kokkos::is_memory_space_v<NgpMemSpace>,
                "NgpSharedComponent requires NgpMemSpace to be a Kokkos memory space.");

  using our_t = NgpSharedComponent<SharedType, NgpMemSpace>;
  using canonical_access = access::raw<SharedType>;
  using shared_type = SharedType;
  using view_t = typename SharedComponent<shared_type>::view_t;

  NgpSharedComponent() = default;
  NgpSharedComponent(const NgpSharedComponent&) = default;
  NgpSharedComponent(NgpSharedComponent&&) = default;
  NgpSharedComponent& operator=(const NgpSharedComponent&) = default;
  NgpSharedComponent& operator=(NgpSharedComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex /*entity_index*/) const {
    return ngp_view_(0);
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& ngp_view()       { return ngp_view_; }
  KOKKOS_INLINE_FUNCTION const auto& ngp_view() const { return ngp_view_; }
  // clang-format on

  void sync_to_device() {
    host_component().sync_to_device();
  }

  void sync_to_host() {
    host_component().sync_to_host();
  }

  void modify_on_device() {
    host_component().modify_on_device();
  }

  void modify_on_host() {
    host_component().modify_on_host();
  }

  void clear_host_sync_state() {
    host_component().clear_host_sync_state();
  }

  void clear_device_sync_state() {
    host_component().clear_device_sync_state();
  }

 private:
  using host_component_type = SharedComponent<shared_type>;
  using host_view_type = impl::shared_component_host_view_t<shared_type>;
  static constexpr bool aliases_host_storage = Kokkos::SpaceAccessibility<NgpMemSpace, Kokkos::HostSpace>::accessible;
  using ngp_view_type =
      std::conditional_t<aliases_host_storage, host_view_type, Kokkos::View<shared_type*, NgpMemSpace>>;

  NgpSharedComponent(host_component_type& host_component, ngp_view_type ngp_view)
      : host_component_(&host_component), ngp_view_(ngp_view) {
  }

  host_component_type& host_component() const {
    MUNDY_THROW_ASSERT(host_component_ != nullptr, std::runtime_error, "NgpSharedComponent host component is null");
    return *host_component_;
  }

  host_component_type* host_component_ = nullptr;
  ngp_view_type ngp_view_;

  template <typename OtherNgpMemSpace, typename OtherSharedType>
  friend NgpSharedComponent<OtherSharedType, OtherNgpMemSpace>& get_updated_ngp_component(
      const SharedComponent<OtherSharedType>& component);
};  // NgpSharedComponent

template <typename SharedType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpRawSharedComponent = NgpSharedComponent<SharedType, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
class NgpSharedScalarComponent : public NgpSharedComponent<ScalarType, NgpMemSpace> {
 public:
  using our_t = NgpSharedScalarComponent<ScalarType, NgpMemSpace>;
  using base_t = NgpSharedComponent<ScalarType, NgpMemSpace>;
  using canonical_access = access::scalar<ScalarType>;
  using view_t =
      decltype(get_scalar_view<ScalarType>(std::declval<decltype(std::declval<base_t&>().ngp_view().data())>()));

  NgpSharedScalarComponent() = default;
  explicit NgpSharedScalarComponent(base_t base_component) : base_t(std::move(base_component)) {
  }

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex /*entity_index*/) const {
    auto& ngp_view = static_cast<our_t&>(*const_cast<our_t*>(this)).ngp_view();
    return get_scalar_view<ScalarType>(ngp_view.data());
  }
};

template <typename ScalarType, size_t N, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedVectorComponent = NgpSharedComponent<Vector<ScalarType, N>, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedVector1Component = NgpSharedVectorComponent<ScalarType, 1, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedVector2Component = NgpSharedVectorComponent<ScalarType, 2, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedVector3Component = NgpSharedVectorComponent<ScalarType, 3, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedVector4Component = NgpSharedVectorComponent<ScalarType, 4, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedVector5Component = NgpSharedVectorComponent<ScalarType, 5, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedVector6Component = NgpSharedVectorComponent<ScalarType, 6, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedMatrix3Component = NgpSharedComponent<Matrix3<ScalarType>, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedQuaternionComponent = NgpSharedComponent<Quaternion<ScalarType>, NgpMemSpace>;

template <typename ScalarType, typename NgpMemSpace = stk::ngp::MemSpace>
using NgpSharedAABBComponent = NgpSharedComponent<AABB<ScalarType>, NgpMemSpace>;

template <typename NgpMemSpace = stk::ngp::MemSpace, typename SharedType>
NgpSharedComponent<SharedType, NgpMemSpace>& get_updated_ngp_component(const SharedComponent<SharedType>& component) {
  static_assert(Kokkos::SpaceAccessibility<NgpMemSpace, stk::ngp::MemSpace>::accessible,
                "get_updated_ngp_component requires a device-accessible memory space.");

  using ngp_component_type = NgpSharedComponent<SharedType, NgpMemSpace>;
  using host_view_type = impl::shared_component_host_view_t<SharedType>;
  constexpr bool aliases_host_storage = Kokkos::SpaceAccessibility<NgpMemSpace, Kokkos::HostSpace>::accessible;
  using ngp_view_type =
      std::conditional_t<aliases_host_storage, host_view_type, Kokkos::View<SharedType*, NgpMemSpace>>;
  using synchronizer_t = impl::SharedComponentSynchronizerT<host_view_type, ngp_view_type, aliases_host_storage>;

  std::any& any_ngp_component = component.any_ngp_component();

  if (!any_ngp_component.has_value()) {
    ngp_view_type ngp_view = [&component]() {
      if constexpr (aliases_host_storage) {
        return component.host_view();
      } else {
        return ngp_view_type(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ngp_shared_component_value"), 1);
      }
    }();
    if constexpr (!aliases_host_storage) {
      Kokkos::deep_copy(ngp_view, component.host_view());
    }

    any_ngp_component = ngp_component_type(const_cast<SharedComponent<SharedType>&>(component), ngp_view);
    ngp_component_type& ngp_component = std::any_cast<ngp_component_type&>(any_ngp_component);
    component.set_synchronizer(std::make_shared<synchronizer_t>(component.host_view(), ngp_component.ngp_view()));
  }

  return std::any_cast<NgpSharedComponent<SharedType, NgpMemSpace>&>(any_ngp_component);
}

template <typename NgpMemSpace = stk::ngp::MemSpace, typename ScalarType>
auto get_updated_ngp_component(const SharedScalarComponent<ScalarType>& component) {
  auto& ngp_component =
      get_updated_ngp_component<NgpMemSpace>(static_cast<const SharedComponent<ScalarType>&>(component));
  return NgpSharedScalarComponent<ScalarType, NgpMemSpace>(ngp_component);
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Host Deduction Guides
//@{

// **********************************************************************************************************************
/// \brief Class template argument deduction guides for SharedComponent
template <typename SharedType>
  MUNDY_REQUIRES(!Kokkos::is_view_v<std::remove_cvref_t<SharedType>>)
SharedComponent(SharedType) -> SharedComponent<std::remove_cvref_t<SharedType>>;

template <impl::SharedComponentHostView HostViewType>
SharedComponent(HostViewType) -> SharedComponent<impl::shared_component_host_view_value_t<HostViewType>>;

// **********************************************************************************************************************
/// \brief Class template argument deduction guides for SharedScalarComponent
template <typename ScalarType>
  MUNDY_REQUIRES(!Kokkos::is_view_v<std::remove_cvref_t<ScalarType>>)
SharedScalarComponent(ScalarType) -> SharedScalarComponent<std::remove_cvref_t<ScalarType>>;

template <impl::SharedComponentHostView HostViewType>
  MUNDY_REQUIRES(std::is_arithmetic_v<impl::shared_component_host_view_value_t<HostViewType>>)
SharedScalarComponent(HostViewType) -> SharedScalarComponent<impl::shared_component_host_view_value_t<HostViewType>>;

// **********************************************************************************************************************
/// \brief Class template argument deduction guides for SharedVectorComponent
template <typename ScalarType, size_t N>
SharedVectorComponent(Vector<ScalarType, N>) -> SharedVectorComponent<ScalarType, N>;

template <impl::SharedComponentHostView HostViewType>
  MUNDY_REQUIRES(is_vector_v<impl::shared_component_host_view_value_t<HostViewType>>)
SharedVectorComponent(HostViewType)
    -> SharedVectorComponent<typename impl::shared_component_host_view_value_t<HostViewType>::scalar_t,
                             impl::shared_component_host_view_value_t<HostViewType>::size>;

// **********************************************************************************************************************
/// \brief Class template argument deduction guides for SharedMatrix3Component
template <typename ScalarType>
SharedMatrix3Component(Matrix3<ScalarType>) -> SharedMatrix3Component<ScalarType>;

template <impl::SharedComponentHostView HostViewType>
  MUNDY_REQUIRES(is_matrix3_v<impl::shared_component_host_view_value_t<HostViewType>>)
SharedMatrix3Component(HostViewType)
    -> SharedMatrix3Component<typename impl::shared_component_host_view_value_t<HostViewType>::scalar_t>;

// **********************************************************************************************************************
/// \brief Class template argument deduction guides for SharedQuaternionComponent
template <typename ScalarType>
SharedQuaternionComponent(Quaternion<ScalarType>) -> SharedQuaternionComponent<ScalarType>;

template <impl::SharedComponentHostView HostViewType>
  MUNDY_REQUIRES(is_quaternion_v<impl::shared_component_host_view_value_t<HostViewType>>)
SharedQuaternionComponent(HostViewType)
    -> SharedQuaternionComponent<typename impl::shared_component_host_view_value_t<HostViewType>::scalar_t>;

// **********************************************************************************************************************
/// \brief Class template argument deduction guides for SharedAABBComponent
template <typename ScalarType>
SharedAABBComponent(AABB<ScalarType>) -> SharedAABBComponent<ScalarType>;

template <impl::SharedComponentHostView HostViewType>
  MUNDY_REQUIRES(is_aabb_v<impl::shared_component_host_view_value_t<HostViewType>>)
SharedAABBComponent(HostViewType)
    -> SharedAABBComponent<typename impl::shared_component_host_view_value_t<HostViewType>::scalar_t>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_SHAREDCOMPONENT_HPP_
