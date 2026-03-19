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

#ifndef MUNDY_MESH_COMPONENTS_HPP_
#define MUNDY_MESH_COMPONENTS_HPP_

// C++ core
#include <any>
#include <memory>
#include <tuple>
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type
#include <utility>      // for std::declval

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// Trilinos
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_topology/topology.hpp>      // for stk::topology::topology_t
#include <stk_util/ngp/NgpSpaces.hpp>     // for stk::ngp::MemSpace

// Mundy
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
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

template <typename SharedType>
class HostSharedComponent;

template <typename SharedType, typename NgpMemSpace>
class NgpSharedComponent;

//! \name Components
//@{

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

template <typename SharedType>
class SharedComponentState;

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

template <typename SharedType>
using shared_component_host_view_t =
    Kokkos::View<SharedType*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename SharedType>
using owned_shared_component_host_view_t = Kokkos::View<SharedType*, Kokkos::HostSpace>;

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

/// \brief Internal state shared by all shallow copies of a HostSharedComponent.
///
/// The public HostSharedComponent API intentionally hides this type. It exists so the host-side accessor can behave
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
    requires requires {
      requires Kokkos::is_view_v<std::remove_cvref_t<HostViewType>>;
      typename std::remove_cvref_t<HostViewType>::value_type;
      typename std::remove_cvref_t<HostViewType>::memory_space;
      requires(std::remove_cvref_t<HostViewType>::rank == 1);
      requires(std::is_same_v<typename std::remove_cvref_t<HostViewType>::memory_space, Kokkos::HostSpace>);
      requires(std::is_same_v<typename std::remove_cvref_t<HostViewType>::value_type, shared_type>);
    }
  explicit SharedComponentState(HostViewType host_view)
      : host_view_(),
        host_owner_(std::move(host_view)),
        any_ngp_component_(),
        synchronizer_(nullptr),
        modified_on_host_(false),
        modified_on_device_(false) {
    auto& host_owner_view = std::any_cast<std::remove_cvref_t<HostViewType>&>(host_owner_);
    MUNDY_THROW_REQUIRE(host_owner_view.extent(0) == 1, std::invalid_argument,
                        "HostSharedComponent requires a rank-1 HostSpace view with extent 1.");
    host_view_ = host_view_type(host_owner_view.data(), 1);
  }

  shared_type& host_value() {
    return host_view_(0);
  }

  const shared_type& host_value() const {
    return host_view_(0);
  }

  host_view_type host_view() {
    return host_view_;
  }

  host_view_type host_value_view() {
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
        MUNDY_THROW_REQUIRE(false, std::logic_error,
                            "sync_to_host called on a HostSharedComponent with no device data.");
      }
      clear_device_sync_state();
    }
  }

  void sync_to_device() {
    if (need_sync_to_device()) {
      if (has_device_data()) {
        synchronizer_->sync_to_device();
      } else {
        MUNDY_THROW_REQUIRE(false, std::logic_error,
                            "sync_to_device called on a HostSharedComponent with no device data.");
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

template <typename ScalarType, typename AccessPolicy>
class HostFieldComponent : public FieldComponentBase {
 public:
  using field_type = stk::mesh::Field<ScalarType>;
  using access_policy = AccessPolicy;
  using view_t = decltype(access_policy::host_access(std::declval<field_type&>(), std::declval<stk::mesh::Entity>()));

  explicit HostFieldComponent(field_type& field) : FieldComponentBase(field), field_(field) {
  }

  HostFieldComponent(const HostFieldComponent&) = default;
  HostFieldComponent(HostFieldComponent&&) = default;
  HostFieldComponent& operator=(const HostFieldComponent&) = delete;
  HostFieldComponent& operator=(HostFieldComponent&&) = delete;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return access_policy::host_access(field_, entity);
  }

  // clang-format off
  inline       field_type& field()       { return field_; }
  inline const field_type& field() const { return field_; }
  // clang-format on

 private:
  field_type& field_;
};  // HostFieldComponent

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

}  // namespace impl

template <typename ValueType>
class FieldComponent : public impl::HostFieldComponent<ValueType, impl::FieldDataAccessPolicy> {
 public:
  using our_t = FieldComponent<ValueType>;
  using base_t = impl::HostFieldComponent<ValueType, impl::FieldDataAccessPolicy>;
  using view_t = typename base_t::view_t;

  explicit FieldComponent(stk::mesh::Field<ValueType>& field) : base_t(field) {
  }
};  // FieldComponent

template <typename NgpFieldType>
class NgpFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::FieldDataAccessPolicy> {
 public:
  using our_t = NgpFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::FieldDataAccessPolicy>;
  using view_t = typename base_t::view_t;

  NgpFieldComponent() = default;
  explicit NgpFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpFieldComponent

template <typename ScalarType>
class ScalarFieldComponent : public impl::HostFieldComponent<ScalarType, impl::ScalarFieldAccessPolicy> {
 public:
  using our_t = ScalarFieldComponent<ScalarType>;
  using base_t = impl::HostFieldComponent<ScalarType, impl::ScalarFieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  explicit ScalarFieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // ScalarFieldComponent

template <typename NgpFieldType>
class NgpScalarFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::ScalarFieldAccessPolicy> {
 public:
  using our_t = NgpScalarFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::ScalarFieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  NgpScalarFieldComponent() = default;
  explicit NgpScalarFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpScalarFieldComponent

template <typename ScalarType, size_t N>
class VectorFieldComponent : public impl::HostFieldComponent<ScalarType, impl::VectorFieldAccessPolicy<N>> {
 public:
  using our_t = VectorFieldComponent<ScalarType, N>;
  using base_t = impl::HostFieldComponent<ScalarType, impl::VectorFieldAccessPolicy<N>>;
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
class Matrix3FieldComponent : public impl::HostFieldComponent<ScalarType, impl::Matrix3FieldAccessPolicy> {
 public:
  using base_t = impl::HostFieldComponent<ScalarType, impl::Matrix3FieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  explicit Matrix3FieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // Matrix3FieldComponent

template <typename NgpFieldType>
class NgpMatrix3FieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::Matrix3FieldAccessPolicy> {
 public:
  using our_t = NgpMatrix3FieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::Matrix3FieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  NgpMatrix3FieldComponent() = default;
  explicit NgpMatrix3FieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpMatrix3FieldComponent

template <typename ScalarType>
class QuaternionFieldComponent : public impl::HostFieldComponent<ScalarType, impl::QuaternionFieldAccessPolicy> {
 public:
  using base_t = impl::HostFieldComponent<ScalarType, impl::QuaternionFieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  explicit QuaternionFieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // QuaternionFieldComponent

template <typename NgpFieldType>
class NgpQuaternionFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::QuaternionFieldAccessPolicy> {
 public:
  using our_t = NgpQuaternionFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::QuaternionFieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  NgpQuaternionFieldComponent() = default;
  explicit NgpQuaternionFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpQuaternionFieldComponent

template <typename ScalarType>
class AABBFieldComponent : public impl::HostFieldComponent<ScalarType, impl::AABBFieldAccessPolicy> {
 public:
  using base_t = impl::HostFieldComponent<ScalarType, impl::AABBFieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  explicit AABBFieldComponent(stk::mesh::Field<ScalarType>& field) : base_t(field) {
  }
};  // AABBFieldComponent

template <typename NgpFieldType>
class NgpAABBFieldComponent : public impl::NgpFieldComponent<NgpFieldType, impl::AABBFieldAccessPolicy> {
 public:
  using our_t = NgpAABBFieldComponent<NgpFieldType>;
  using base_t = impl::NgpFieldComponent<NgpFieldType, impl::AABBFieldAccessPolicy>;
  using view_t = typename base_t::view_t;

  NgpAABBFieldComponent() = default;
  explicit NgpAABBFieldComponent(NgpFieldType ngp_field) : base_t(ngp_field) {
  }
};  // NgpAABBFieldComponent

/// \brief A component that returns the same shared value for every entity.
///
/// Construct either from:
///   - a raw `SharedType`, which is copied into owned HostSpace storage
///   - a rank-1 Kokkos::View in HostSpace with extent 1, which is aliased exactly as given whether that view is
///   managed or unmanaged
template <typename SharedType>
class HostSharedComponent {
 public:
  static_assert(!std::is_reference_v<SharedType>, "HostSharedComponent may not store a reference type.");

  using our_t = HostSharedComponent<SharedType>;
  using shared_type = SharedType;
  using view_t = shared_type&;

  HostSharedComponent() = default;
  explicit HostSharedComponent(shared_type shared_value)
      : state_(std::make_shared<state_type>(std::move(shared_value))) {
  }

  template <typename HostViewType>
    requires requires {
      requires Kokkos::is_view_v<std::remove_cvref_t<HostViewType>>;
      typename std::remove_cvref_t<HostViewType>::value_type;
      typename std::remove_cvref_t<HostViewType>::memory_space;
      requires(std::remove_cvref_t<HostViewType>::rank == 1);
      requires(std::is_same_v<typename std::remove_cvref_t<HostViewType>::memory_space, Kokkos::HostSpace>);
      requires(std::is_same_v<typename std::remove_cvref_t<HostViewType>::value_type, shared_type>);
    }
  explicit HostSharedComponent(HostViewType host_view) : state_(std::make_shared<state_type>(std::move(host_view))) {
  }

  HostSharedComponent(const HostSharedComponent&) = default;
  HostSharedComponent(HostSharedComponent&&) = default;
  HostSharedComponent& operator=(const HostSharedComponent&) = default;
  HostSharedComponent& operator=(HostSharedComponent&&) = default;

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
    MUNDY_THROW_ASSERT(state_, std::runtime_error, "HostSharedComponent state is null");
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
      const HostSharedComponent<OtherSharedType>& component);
};  // HostSharedComponent

template <typename SharedType>
HostSharedComponent(SharedType) -> HostSharedComponent<SharedType>;

template <typename HostViewType>
  requires requires {
    requires Kokkos::is_view_v<std::remove_cvref_t<HostViewType>>;
    typename std::remove_cvref_t<HostViewType>::value_type;
    typename std::remove_cvref_t<HostViewType>::memory_space;
    requires(std::remove_cvref_t<HostViewType>::rank == 1);
    requires(std::is_same_v<typename std::remove_cvref_t<HostViewType>::memory_space, Kokkos::HostSpace>);
  }
HostSharedComponent(HostViewType) -> HostSharedComponent<typename std::remove_cvref_t<HostViewType>::value_type>;

template <typename SharedType, typename NgpMemSpace>
class NgpSharedComponent {
 public:
  static_assert(!std::is_reference_v<SharedType>, "NgpSharedComponent may not store a reference type.");
  static_assert(Kokkos::is_memory_space_v<NgpMemSpace>,
                "NgpSharedComponent requires NgpMemSpace to be a Kokkos memory space.");

  using our_t = NgpSharedComponent<SharedType, NgpMemSpace>;
  using shared_type = SharedType;
  using view_t = typename HostSharedComponent<shared_type>::view_t;

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
  using host_component_type = HostSharedComponent<shared_type>;
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
      const HostSharedComponent<OtherSharedType>& component);
};  // NgpSharedComponent

/// \brief Create a HostSharedComponent by copying a raw value into owned HostSpace storage.
template <typename SharedType>
  requires(!requires {
    requires Kokkos::is_view_v<std::remove_cvref_t<SharedType>>;
    typename std::remove_cvref_t<SharedType>::value_type;
    typename std::remove_cvref_t<SharedType>::memory_space;
    requires(std::remove_cvref_t<SharedType>::rank == 1);
    requires(std::is_same_v<typename std::remove_cvref_t<SharedType>::memory_space, Kokkos::HostSpace>);
  })
auto make_shared_view_accessor(SharedType&& shared_value) {
  using component_type = HostSharedComponent<std::decay_t<SharedType>>;
  return component_type(std::forward<SharedType>(shared_value));
}

/// \brief Create a HostSharedComponent that aliases a rank-1 HostSpace view of extent 1.
template <typename HostViewType>
  requires requires {
    requires Kokkos::is_view_v<std::remove_cvref_t<HostViewType>>;
    typename std::remove_cvref_t<HostViewType>::value_type;
    typename std::remove_cvref_t<HostViewType>::memory_space;
    requires(std::remove_cvref_t<HostViewType>::rank == 1);
    requires(std::is_same_v<typename std::remove_cvref_t<HostViewType>::memory_space, Kokkos::HostSpace>);
  }
auto make_shared_view_accessor(HostViewType host_view) {
  using component_type = HostSharedComponent<typename std::remove_cvref_t<HostViewType>::value_type>;
  return component_type(std::move(host_view));
}

/// \brief A small helper type for tying a Tag to an underlying component
template <typename Tag, stk::topology::rank_t our_rank, typename ComponentType>
class TaggedComponent {
 public:
  using our_t = TaggedComponent<Tag, our_rank, ComponentType>;
  using view_t = typename ComponentType::view_t;
  using tag_type = Tag;
  using component_type = ComponentType;
  static constexpr stk::topology::rank_t rank = our_rank;

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
  auto operator()(const EntityExprBase<EntityExpr>& e) const;

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

template <typename Tag, stk::topology::rank_t our_rank, typename ComponentType>
TaggedComponent<Tag, our_rank, ComponentType> make_tagged_component(ComponentType component) {
  return TaggedComponent<Tag, our_rank, ComponentType>(component);
}

/// \brief A small helper type for tying a Tag to an underlying ngp-compatible component
template <typename Tag, stk::topology::rank_t our_rank, typename NgpComponentType>
class NgpTaggedComponent {
 public:
  using our_t = NgpTaggedComponent<Tag, our_rank, NgpComponentType>;
  using view_t = typename NgpComponentType::view_t;
  using tag_type = Tag;
  using component_type = NgpComponentType;
  static constexpr stk::topology::rank_t rank = our_rank;

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
  auto operator()(const EntityExprBase<EntityExpr>& e) const {
    MUNDY_THROW_REQUIRE(
        e.rank() == rank, std::runtime_error,
        fmt::format("Attempting to access field of rank {} on entity expression of rank {}", rank, e.rank()));
    return AccessorExpr<our_t, EntityExpr>(*this, e.self());
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

/// \brief A helper function for getting the NGP component from a regular component.
///
/// Field-backed components simply wrap STK's updated ngp field and return the wrapper by value.
/// HostSharedComponent follows the LinkData pattern instead, lazily materializing and caching a
/// memspace-specific NgpSharedComponent in a std::any owned by the host component state.
template <typename ScalarType, typename AccessPolicy>
auto get_updated_ngp_component(const impl::HostFieldComponent<ScalarType, AccessPolicy>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ScalarType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return impl::NgpFieldComponent<ngp_field_type, AccessPolicy>(ngp_field);
}

template <typename NgpMemSpace = stk::ngp::MemSpace, typename SharedType>
NgpSharedComponent<SharedType, NgpMemSpace>& get_updated_ngp_component(
    const HostSharedComponent<SharedType>& component) {
  static_assert(Kokkos::SpaceAccessibility<NgpMemSpace, stk::ngp::MemSpace>::accessible,
                "get_updated_ngp_component requires a device-accessible memory space.");

  using ngp_component_type = NgpSharedComponent<SharedType, NgpMemSpace>;
  using host_view_type = impl::shared_component_host_view_t<SharedType>;
  constexpr bool aliases_host_storage = Kokkos::SpaceAccessibility<NgpMemSpace, Kokkos::HostSpace>::accessible;
  using ngp_view_type =
      std::conditional_t<aliases_host_storage, host_view_type, Kokkos::View<SharedType*, NgpMemSpace>>;
  using synchronizer_t =
      impl::SharedComponentSynchronizerT<host_view_type, ngp_view_type, aliases_host_storage>;

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

    any_ngp_component = ngp_component_type(const_cast<HostSharedComponent<SharedType>&>(component), ngp_view);
    ngp_component_type& ngp_component = std::any_cast<ngp_component_type&>(any_ngp_component);
    component.set_synchronizer(std::make_shared<synchronizer_t>(component.host_view(), ngp_component.ngp_view()));
  }

  return std::any_cast<NgpSharedComponent<SharedType, NgpMemSpace>&>(any_ngp_component);
}
//
template <typename Tag, stk::topology::rank_t our_rank, typename ComponentType>
decltype(auto) get_updated_ngp_component(const TaggedComponent<Tag, our_rank, ComponentType>& tagged_component) {
  auto ngp_component = get_updated_ngp_component(tagged_component.component());
  using ngp_component_type = std::remove_reference_t<decltype(ngp_component)>;
  return NgpTaggedComponent<Tag, our_rank, ngp_component_type>(ngp_component);
}

template <typename Tag, stk::topology::rank_t our_rank, typename ComponentType>
template <class EntityExpr>
auto TaggedComponent<Tag, our_rank, ComponentType>::operator()(const EntityExprBase<EntityExpr>& e) const {
  MUNDY_THROW_REQUIRE(
      e.rank() == rank, std::runtime_error,
      fmt::format("Attempting to access field of rank {} on entity expression of rank {}", rank, e.rank()));

  // Entity expressions are (currently) always on the device, so we need to get the NGP tagged component
  // TODO(palmerb4): Allow for exec_spaces that aren't simply the default execution space (need Tril 16.1+)
  auto ngp_this = get_updated_ngp_component(*this);
  return ngp_this(e.self());
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_COMPONENTS_HPP_
