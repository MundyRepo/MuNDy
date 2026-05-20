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

#ifndef MUNDY_MESH_IMPL_SHAREDCOMPONENTSIMPL_HPP_
#define MUNDY_MESH_IMPL_SHAREDCOMPONENTSIMPL_HPP_

/// \file SharedComponentImpl.hpp
/// \brief A set of helpers for working with shared components with reduced boilerplate code.

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
#include <mundy_mesh/FieldViews.hpp>          // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/ForEachEntity.hpp>       // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/NgpAccessorExpr.hpp>     // for mundy::mesh::AccessorExpr and EntityExprBase
#include <mundy_mesh/impl/ComponentImpl.hpp>  // for mundy::mesh::impl::component_backing_field
#include <mundy_mesh/impl/HostDeviceSynchronizer.hpp>
#include <mundy_utils/requires.hpp>
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/throw_assert.hpp>       // for MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>              // for mundy::tuple

namespace mundy {

namespace mesh {

namespace impl {

template <typename SharedType>
using shared_component_host_view_t =
    Kokkos::View<SharedType*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename SharedType>
using owned_shared_component_host_view_t = Kokkos::View<SharedType*, Kokkos::HostSpace>;

template <typename HostViewType>
concept SharedComponentHostView = requires {
  requires Kokkos::is_view_v<std::remove_cvref_t<HostViewType>>;
  typename std::remove_cvref_t<HostViewType>::value_type;
  typename std::remove_cvref_t<HostViewType>::memory_space;
  requires(std::remove_cvref_t<HostViewType>::rank == 1);
  requires(std::is_same_v<typename std::remove_cvref_t<HostViewType>::memory_space, Kokkos::HostSpace>);
};

template <SharedComponentHostView HostViewType>
using shared_component_host_view_value_t = std::remove_cv_t<typename std::remove_cvref_t<HostViewType>::value_type>;

template <typename HostViewType, typename SharedType>
concept CompatibleSharedComponentHostView =
    SharedComponentHostView<HostViewType> &&
    std::same_as<shared_component_host_view_value_t<HostViewType>, std::remove_cv_t<SharedType>>;

template <typename HostViewType, typename DeviceViewType, bool AliasesStorage>
class SharedComponentSynchronizerT : public HostDeviceSynchronizer {
 public:
  SharedComponentSynchronizerT(HostViewType host_view, DeviceViewType device_view)
      : host_view_(host_view), device_view_(device_view) {
  }

  void sync_to_device() override {
    if constexpr (!AliasesStorage) {
      Kokkos::deep_copy(device_view_, host_view_);
    }
  }

  void sync_to_host() override {
    if constexpr (!AliasesStorage) {
      Kokkos::deep_copy(host_view_, device_view_);
    }
  }

  void modify_on_host() override {
  }

  void modify_on_device() override {
  }

  void update_post_mesh_mod() override {
  }

 private:
  HostViewType host_view_;
  DeviceViewType device_view_;
};  // SharedComponentSynchronizerT

// TODO(palmerb4): The only place that even uses SharedComponentState is the SharedComponent itself.
// That means we should just merge SharedComponentState into SharedComponent and eliminate the indirection. This class
// is entirely a feature of an outdated design that required a shared state. We'll need to clean up the doc of it to
// remove implementation details since this class will no longer be itself an implementation detail. But the comments
// about management of storage are important to tell the user for the sake of clarity, so we should preserve those. Just
// don't mention internal members like host_view_ in the doc.

/// \brief Internal state shared by all shallow copies of a SharedComponent.
///
/// The public SharedComponent API intentionally hides this type. It exists so the host-side accessor can behave
/// like a cheap view while still centralizing:
///   - the canonical host representation of the shared value
///   - lazy ownership/aliasing of the host storage
///   - the cached ngp component stored in a std::any
///   - the synchronizer and sync-state bookkeeping
///
/// Host-side access always goes through `host_view_`, which is a rank-1 unmanaged HostSpace view of extent 1. The
/// underlying storage that `host_view_` aliases is kept alive by `host_owner_`:
///   - if constructed from a raw value, we allocate an owned HostSpace view, stash it in `host_owner_`, and then point
///   `host_view_` at that allocation
///   - if constructed from a HostSpace Kokkos::View, we stash that exact view object in `host_owner_` and then point
///   `host_view_` at its memory regardless of whether the given view was managed or unmanaged
///
/// This gives us one stable, non-owning HostSpace view for all downstream logic while preserving the lifetime semantics
/// of the original input.
template <typename SharedType>
class SharedComponentState {
 public:
  static_assert(!std::is_reference_v<SharedType>, "SharedComponentState may not store a reference type.");

  using shared_type = SharedType;
  using host_view_type = shared_component_host_view_t<shared_type>;
  using owned_host_view_type = owned_shared_component_host_view_t<shared_type>;

  explicit SharedComponentState(shared_type host_value)
      : host_view_(),
        host_owner_(owned_host_view_type("host_shared_component_value", 1)),
        any_ngp_component_(),
        synchronizer_(nullptr),
        modified_on_host_(false),
        modified_on_device_(false) {
    auto& owned_view = std::any_cast<owned_host_view_type&>(host_owner_);
    owned_view(0) = std::move(host_value);
    host_view_ = host_view_type(owned_view.data(), 1);
  }

  template <typename HostViewType>
  MUNDY_REQUIRES(CompatibleSharedComponentHostView<HostViewType, shared_type>)
  explicit SharedComponentState(HostViewType host_view)
      : host_view_(),
        host_owner_(std::move(host_view)),
        any_ngp_component_(),
        synchronizer_(nullptr),
        modified_on_host_(false),
        modified_on_device_(false) {
    auto& host_owner_view = std::any_cast<std::remove_cvref_t<HostViewType>&>(host_owner_);
    MUNDY_THROW_REQUIRE(host_owner_view.extent(0) == 1, std::invalid_argument,
                        "SharedComponent requires a rank-1 HostSpace view with extent 1.");
    host_view_ = host_view_type(host_owner_view.data(), 1);
  }

  inline shared_type& host_value() {
    return host_view_(0);
  }

  inline const shared_type& host_value() const {
    return host_view_(0);
  }

  inline host_view_type host_view() {
    return host_view_;
  }

  inline host_view_type host_value_view() {
    return host_view_;
  }

  void modify_on_host() {
    MUNDY_THROW_REQUIRE(modified_on_device_ == false, std::invalid_argument,
                        "The host shared value may not be modified while the device shared value is also modified. "
                        "Either sync the device value to host or clear the device modification state.");
    modified_on_host_ = true;
    if (has_device_data()) {
      synchronizer_->modify_on_host();
    }
  }

  void modify_on_device() {
    MUNDY_THROW_REQUIRE(modified_on_host_ == false, std::invalid_argument,
                        "The device shared value may not be modified while the host shared value is also modified. "
                        "Either sync the host value to device or clear the host modification state.");
    modified_on_device_ = true;
    if (has_device_data()) {
      synchronizer_->modify_on_device();
    }
  }

  bool need_sync_to_host() const {
    return modified_on_device_;
  }

  bool need_sync_to_device() const {
    return modified_on_host_;
  }

  void sync_to_host() {
    if (need_sync_to_host()) {
      if (has_device_data()) {
        synchronizer_->sync_to_host();
      } else {
        MUNDY_THROW_REQUIRE(false, std::logic_error, "sync_to_host called on a SharedComponent with no device data.");
      }
      clear_device_sync_state();
    }
  }

  void sync_to_device() {
    if (need_sync_to_device()) {
      if (has_device_data()) {
        synchronizer_->sync_to_device();
      } else {
        MUNDY_THROW_REQUIRE(false, std::logic_error, "sync_to_device called on a SharedComponent with no device data.");
      }
      clear_host_sync_state();
    }
  }

  void clear_host_sync_state() {
    modified_on_host_ = false;
  }

  void clear_device_sync_state() {
    modified_on_device_ = false;
  }

  bool has_device_data() const {
    return synchronizer_ != nullptr;
  }

  std::any& any_ngp_component() {
    return any_ngp_component_;
  }

  void set_synchronizer(std::shared_ptr<HostDeviceSynchronizer> synchronizer) {
    synchronizer_ = std::move(synchronizer);
  }

 private:
  host_view_type host_view_;
  std::any host_owner_;
  std::any any_ngp_component_;
  std::shared_ptr<HostDeviceSynchronizer> synchronizer_;
  bool modified_on_host_;
  bool modified_on_device_;
};  // SharedComponentState

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_SHAREDCOMPONENTSIMPL_HPP_
