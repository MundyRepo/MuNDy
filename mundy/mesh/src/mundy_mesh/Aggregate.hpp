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

#ifndef MUNDY_MESH_AGGREGATES_HPP_
#define MUNDY_MESH_AGGREGATES_HPP_

// C++ core
#include <tuple>
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

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

// Mundy
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_core/tuple.hpp>          // for mundy::core::tuple
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/fmt_stk_types.hpp>  // for STK-compatible fmt::format

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

  void sync_to_device() {
    field_base_.sync_to_device();
  }

  void sync_to_host() {
    field_base_.sync_to_host();
  }

  void modify_on_device() {
    field_base_.modify_on_device();
  }

  void modify_on_host() {
    field_base_.modify_on_host();
  }

  const stk::mesh::FieldBase& field_base() {
    return field_base_;
  }

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

  void sync_to_device() {
    host_field_base_->sync_to_device();
  }

  void sync_to_host() {
    host_field_base_->sync_to_host();
  }

  void modify_on_device() {
    host_field_base_->modify_on_device();
  }

  void modify_on_host() {
    host_field_base_->modify_on_host();
  }

  const stk::mesh::FieldBase& host_field_base() {
    return *host_field_base_;
  }

 private:
  const stk::mesh::FieldBase* host_field_base_;
#endif
};  // NgpFieldComponentBase

template <typename ValueType>
class FieldComponent : public FieldComponentBase {
 public:
  FieldComponent(stk::mesh::Field<ValueType>& field) : FieldComponentBase(field), field_(field) {
  }

  /// \brief Default copy/move/assign constructors
  FieldComponent(const FieldComponent&) = default;
  FieldComponent(FieldComponent&&) = default;
  FieldComponent& operator=(const FieldComponent&) = default;
  FieldComponent& operator=(FieldComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    ValueType* data_ptr = stk::mesh::field_data(*field_, entity);
    MUNDY_THROW_ASSERT(data_ptr, std::runtime_error, "Field data is null");
    unsigned num_scalars = stk::mesh::field_scalars_per_entity(*field_, entity);
    return stk::mesh::EntityFieldData<ValueType>(data_ptr, num_scalars);
  }

  inline stk::mesh::Field<ValueType>& field() {
    return field_;
  }

  inline const stk::mesh::Field<ValueType>& field() const {
    return field_;
  }

 private:
  stk::mesh::Field<ValueType>& field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<FieldComponent<ValueType>>().operator()(std::declval<stk::mesh::Entity>()));
};  // FieldComponent

template <typename NgpFieldType>
class NgpFieldComponent : public NgpFieldComponentBase {
 public:
  NgpFieldComponent() = default;
  NgpFieldComponent(NgpFieldType ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),
#else
      : NgpFieldComponentBase(),
#endif
        ngp_field_(ngp_field) {
  }

  /// \brief Default copy/move/assign constructors
  NgpFieldComponent(const NgpFieldComponent&) = default;
  NgpFieldComponent(NgpFieldComponent&&) = default;
  NgpFieldComponent& operator=(const NgpFieldComponent&) = default;
  NgpFieldComponent& operator=(NgpFieldComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return ngp_field_(entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  NgpFieldType& ngp_field() {
    return ngp_field_;
  }

  KOKKOS_INLINE_FUNCTION
  const NgpFieldType& ngp_field() const {
    return ngp_field_;
  }

#if TRILINOS_MAJOR_MINOR_VERSION < 160000
  void sync_to_device() {
    ngp_field_.sync_to_device();
  }

  void sync_to_host() {
    ngp_field_.sync_to_host();
  }

  void modify_on_device() {
    ngp_field_.modify_on_device();
  }

  void modify_on_host() {
    ngp_field_.modify_on_host();
  }
#endif

 private:
  NgpFieldType ngp_field_;
 
 public:
  /// \brief The view type returned by operator()
  using view_t =
      decltype(std::declval<NgpFieldComponent<NgpFieldType>>().operator()(std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpFieldComponent

template <typename ScalarType>
class ScalarFieldComponent : public FieldComponentBase {
 public:
  ScalarFieldComponent(stk::mesh::Field<ScalarType>& field) : FieldComponentBase(field), field_(field) {
  }

  /// \brief Default copy/move/assign constructors
  ScalarFieldComponent(const ScalarFieldComponent&) = default;
  ScalarFieldComponent(ScalarFieldComponent&&) = default;
  ScalarFieldComponent& operator=(const ScalarFieldComponent&) = default;
  ScalarFieldComponent& operator=(ScalarFieldComponent&&) = default;

  /// \brief Fetch the value of the field at the given entity
  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return scalar_field_data(field_, entity);
  }

  inline stk::mesh::Field<ScalarType>& field() {
    return field_;
  }

  inline const stk::mesh::Field<ScalarType>& field() const {
    return field_;
  }

 private:
  stk::mesh::Field<ScalarType>& field_;

 public:
  /// \brief The view type returned by operator()
  using view_t =
      decltype(std::declval<ScalarFieldComponent<ScalarType>>().operator()(std::declval<stk::mesh::Entity>()));
};  // ScalarFieldComponent

template <typename NgpFieldType>
class NgpScalarFieldComponent : public NgpFieldComponentBase {
 public:
  NgpScalarFieldComponent() = default;
  NgpScalarFieldComponent(NgpFieldType ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),
#else
      : NgpFieldComponentBase(),
#endif
        ngp_field_(ngp_field) {
  }

  /// \brief Default copy/move/assign constructors
  NgpScalarFieldComponent(const NgpScalarFieldComponent&) = default;
  NgpScalarFieldComponent(NgpScalarFieldComponent&&) = default;
  NgpScalarFieldComponent& operator=(const NgpScalarFieldComponent&) = default;
  NgpScalarFieldComponent& operator=(NgpScalarFieldComponent&&) = default;

  /// \brief Fetch the value of the field at the given entity index
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return scalar_field_data(ngp_field_, entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  NgpFieldType& ngp_field() {
    return ngp_field_;
  }

  KOKKOS_INLINE_FUNCTION
  const NgpFieldType& ngp_field() const {
    return ngp_field_;
  }

#if TRILINOS_MAJOR_MINOR_VERSION < 160000
  void sync_to_device() {
    ngp_field_.sync_to_device();
  }

  void sync_to_host() {
    ngp_field_.sync_to_host();
  }

  void modify_on_device() {
    ngp_field_.modify_on_device();
  }

  void modify_on_host() {
    ngp_field_.modify_on_host();
  }
#endif

 private:
  NgpFieldType ngp_field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpScalarFieldComponent<NgpFieldType>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpScalarFieldComponent

template <typename ScalarType>
class Vector3FieldComponent : public FieldComponentBase {
 public:
  Vector3FieldComponent(stk::mesh::Field<ScalarType>& field) : FieldComponentBase(field), field_(field) {
  }

  /// \brief Default copy/move/assign constructors
  Vector3FieldComponent(const Vector3FieldComponent&) = default;
  Vector3FieldComponent(Vector3FieldComponent&&) = default;
  Vector3FieldComponent& operator=(const Vector3FieldComponent&) = default;
  Vector3FieldComponent& operator=(Vector3FieldComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return vector3_field_data(field_, entity);
  }

  inline stk::mesh::Field<ScalarType>& field() {
    return field_;
  }

  inline const stk::mesh::Field<ScalarType>& field() const {
    return field_;
  }

 private:
  stk::mesh::Field<ScalarType>& field_;

 public:
  /// \brief The view type returned by operator()
  using view_t =
      decltype(std::declval<Vector3FieldComponent<ScalarType>>().operator()(std::declval<stk::mesh::Entity>()));
};  // Vector3FieldComponent

template <typename NgpFieldType>
class NgpVector3FieldComponent : public NgpFieldComponentBase {
 public:
  NgpVector3FieldComponent() = default;
  NgpVector3FieldComponent(NgpFieldType ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),  // Directly store the field base
#else
      : NgpFieldComponentBase(),
#endif
        ngp_field_(ngp_field) {
  }

  /// \brief Default copy/move/assign constructors
  NgpVector3FieldComponent(const NgpVector3FieldComponent&) = default;
  NgpVector3FieldComponent(NgpVector3FieldComponent&&) = default;
  NgpVector3FieldComponent& operator=(const NgpVector3FieldComponent&) = default;
  NgpVector3FieldComponent& operator=(NgpVector3FieldComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return vector3_field_data(ngp_field_, entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  NgpFieldType& ngp_field() {
    return ngp_field_;
  }

  KOKKOS_INLINE_FUNCTION
  const NgpFieldType& ngp_field() const {
    return ngp_field_;
  }

#if TRILINOS_MAJOR_MINOR_VERSION < 160000
  void sync_to_device() {
    ngp_field_.sync_to_device();
  }

  void sync_to_host() {
    ngp_field_.sync_to_host();
  }

  void modify_on_device() {
    ngp_field_.modify_on_device();
  }

  void modify_on_host() {
    ngp_field_.modify_on_host();
  }
#endif

 private:
  NgpFieldType ngp_field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpVector3FieldComponent<NgpFieldType>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpVector3FieldComponent

template <typename ScalarType>
class Matrix3FieldComponent : public FieldComponentBase {
 public:
  Matrix3FieldComponent(stk::mesh::Field<ScalarType>& field) : FieldComponentBase(field), field_(field) {
  }

  /// \brief Default copy/move/assign constructors
  Matrix3FieldComponent(const Matrix3FieldComponent&) = default;
  Matrix3FieldComponent(Matrix3FieldComponent&&) = default;
  Matrix3FieldComponent& operator=(const Matrix3FieldComponent&) = default;
  Matrix3FieldComponent& operator=(Matrix3FieldComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return matrix3_field_data(field_, entity);
  }

  inline stk::mesh::Field<ScalarType>& field() {
    return field_;
  }

  inline const stk::mesh::Field<ScalarType>& field() const {
    return field_;
  }

 private:
  stk::mesh::Field<ScalarType>& field_;

 public:
  /// \brief The view type returned by operator()
  using view_t =
      decltype(std::declval<Matrix3FieldComponent<ScalarType>>().operator()(std::declval<stk::mesh::Entity>()));
};  // Matrix3FieldComponent

template <typename NgpFieldType>
class NgpMatrix3FieldComponent : public NgpFieldComponentBase {
 public:
  NgpMatrix3FieldComponent() = default;
  NgpMatrix3FieldComponent(NgpFieldType ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),  // Directly store the field base
#else
      : NgpFieldComponentBase(),
#endif
        ngp_field_(ngp_field) {
  }

  /// \brief Default copy/move/assign constructors
  NgpMatrix3FieldComponent(const NgpMatrix3FieldComponent&) = default;
  NgpMatrix3FieldComponent(NgpMatrix3FieldComponent&&) = default;
  NgpMatrix3FieldComponent& operator=(const NgpMatrix3FieldComponent&) = default;
  NgpMatrix3FieldComponent& operator=(NgpMatrix3FieldComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return matrix3_field_data(ngp_field_, entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  NgpFieldType& ngp_field() {
    return ngp_field_;
  }

  KOKKOS_INLINE_FUNCTION
  const NgpFieldType& ngp_field() const {
    return ngp_field_;
  }

#if TRILINOS_MAJOR_MINOR_VERSION < 160000
  void sync_to_device() {
    ngp_field_.sync_to_device();
  }

  void sync_to_host() {
    ngp_field_.sync_to_host();
  }

  void modify_on_device() {
    ngp_field_.modify_on_device();
  }

  void modify_on_host() {
    ngp_field_.modify_on_host();
  }
#endif

 private:
  NgpFieldType ngp_field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpMatrix3FieldComponent<NgpFieldType>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpMatrix3FieldComponent

template <typename ScalarType>
class QuaternionFieldComponent : public FieldComponentBase {
 public:
  QuaternionFieldComponent(stk::mesh::Field<ScalarType>& field) : FieldComponentBase(field), field_(field) {
  }

  /// \brief Default copy/move/assign constructors
  QuaternionFieldComponent(const QuaternionFieldComponent&) = default;
  QuaternionFieldComponent(QuaternionFieldComponent&&) = default;
  QuaternionFieldComponent& operator=(const QuaternionFieldComponent&) = default;
  QuaternionFieldComponent& operator=(QuaternionFieldComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return quaternion_field_data(field_, entity);
  }

  inline stk::mesh::Field<ScalarType>& field() {
    return field_;
  }

  inline const stk::mesh::Field<ScalarType>& field() const {
    return field_;
  }

 private:
  stk::mesh::Field<ScalarType>& field_;

 public:
  /// \brief The view type returned by operator()
  using view_t =
      decltype(std::declval<QuaternionFieldComponent<ScalarType>>().operator()(std::declval<stk::mesh::Entity>()));
};  // QuaternionFieldComponent

template <typename NgpFieldType>
class NgpQuaternionFieldComponent : public NgpFieldComponentBase {
 public:
  NgpQuaternionFieldComponent() = default;
  NgpQuaternionFieldComponent(NgpFieldType ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),
#else
      : NgpFieldComponentBase(),
#endif
        ngp_field_(ngp_field) {
  }

  /// \brief Default copy/move/assign constructors
  NgpQuaternionFieldComponent(const NgpQuaternionFieldComponent&) = default;
  NgpQuaternionFieldComponent(NgpQuaternionFieldComponent&&) = default;
  NgpQuaternionFieldComponent& operator=(const NgpQuaternionFieldComponent&) = default;
  NgpQuaternionFieldComponent& operator=(NgpQuaternionFieldComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return quaternion_field_data(ngp_field_, entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  NgpFieldType& ngp_field() {
    return ngp_field_;
  }

#if TRILINOS_MAJOR_MINOR_VERSION < 160000
  void sync_to_device() {
    ngp_field_.sync_to_device();
  }

  void sync_to_host() {
    ngp_field_.sync_to_host();
  }

  void modify_on_device() {
    ngp_field_.modify_on_device();
  }

  void modify_on_host() {
    ngp_field_.modify_on_host();
  }
#endif

 private:
  NgpFieldType ngp_field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpQuaternionFieldComponent<NgpFieldType>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpQuaternionFieldComponent

template <typename ScalarType>
class AABBFieldComponent : public FieldComponentBase {
 public:
  AABBFieldComponent(stk::mesh::Field<ScalarType>& field) : FieldComponentBase(field), field_(field) {
  }

  /// \brief Default copy/move/assign constructors
  AABBFieldComponent(const AABBFieldComponent&) = default;
  AABBFieldComponent(AABBFieldComponent&&) = default;
  AABBFieldComponent& operator=(const AABBFieldComponent&) = default;
  AABBFieldComponent& operator=(AABBFieldComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return aabb_field_data(field_, entity);
  }

  inline stk::mesh::Field<ScalarType>& field() {
    return field_;
  }

  inline const stk::mesh::Field<ScalarType>& field() const {
    return field_;
  }

 private:
  stk::mesh::Field<ScalarType>& field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<AABBFieldComponent<ScalarType>>().operator()(std::declval<stk::mesh::Entity>()));
};  // AABBFieldComponent

template <typename NgpFieldType>
class NgpAABBFieldComponent : public NgpFieldComponentBase {
 public:
  NgpAABBFieldComponent() = default;
  NgpAABBFieldComponent(NgpFieldType ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),
#else
      : NgpFieldComponentBase(),
#endif

        ngp_field_(ngp_field) {
  }

  /// \brief Default copy/move/assign constructors
  NgpAABBFieldComponent(const NgpAABBFieldComponent&) = default;
  NgpAABBFieldComponent(NgpAABBFieldComponent&&) = default;
  NgpAABBFieldComponent& operator=(const NgpAABBFieldComponent&) = default;
  NgpAABBFieldComponent& operator=(NgpAABBFieldComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return aabb_field_data(ngp_field_, entity_index);
  }

  KOKKOS_INLINE_FUNCTION
  NgpFieldType& ngp_field() {
    return ngp_field_;
  }

#if TRILINOS_MAJOR_MINOR_VERSION < 160000
  void sync_to_device() {
    ngp_field_.sync_to_device();
  }

  void sync_to_host() {
    ngp_field_.sync_to_host();
  }

  void modify_on_device() {
    ngp_field_.modify_on_device();
  }

  void modify_on_host() {
    ngp_field_.modify_on_host();
  }
#endif

 private:
  NgpFieldType ngp_field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpAABBFieldComponent<NgpFieldType>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpAABBFieldComponent

/// \brief A small helper type for tying a Tag to an underlying component
template <typename Tag, stk::topology::rank_t our_rank, typename ComponentType>
class TaggedComponent {
 public:
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

 private:
  component_type component_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<TaggedComponent<Tag, our_rank, ComponentType>>().operator()(
      std::declval<stk::mesh::Entity>()));
};  // TaggedComponent

/// \brief A small helper type for tying a Tag to an underlying ngp-compatible component
template <typename Tag, stk::topology::rank_t our_rank, typename NgpComponentType>
class NgpTaggedComponent {
 public:
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

 private:
  component_type component_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpTaggedComponent<Tag, our_rank, NgpComponentType>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpTaggedComponent

/// \brief A helper function for getting the NGP component from a regular component
///
/// For now, we just create an NGP component here and return it by value. We'll need to test
/// if we should do as STK does and store a pointer to the NGP component in the regular component
/// and use this function to fetch it. If the pointer is nullptr, this function would create said
/// NGP component, store it in the regular component. We could then fetch the NGP component and return it
/// as a reference.
///
/// Overload this function for each type of component with NGP compatibility
template <typename ScalarType>
decltype(auto) get_updated_ngp_component(const ScalarFieldComponent<ScalarType>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ScalarType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return NgpScalarFieldComponent<ngp_field_type>(ngp_field);
}
//
template <typename ScalarType>
decltype(auto) get_updated_ngp_component(const Vector3FieldComponent<ScalarType>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ScalarType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return NgpVector3FieldComponent<ngp_field_type>(ngp_field);
}
//
template <typename ScalarType>
decltype(auto) get_updated_ngp_component(const QuaternionFieldComponent<ScalarType>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ScalarType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return NgpQuaternionFieldComponent<ngp_field_type>(ngp_field);
}
//
template <typename ScalarType>
decltype(auto) get_updated_ngp_component(const AABBFieldComponent<ScalarType>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ScalarType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return NgpAABBFieldComponent<ngp_field_type>(ngp_field);
}
//
template <typename ValueType>
decltype(auto) get_updated_ngp_component(const FieldComponent<ValueType>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ValueType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return NgpFieldComponent<ngp_field_type>(ngp_field);
}
//
template <typename Tag, stk::topology::rank_t our_rank, typename ComponentType>
decltype(auto) get_updated_ngp_component(const TaggedComponent<Tag, our_rank, ComponentType>& tagged_component) {
  auto ngp_component = get_updated_ngp_component(tagged_component.component());
  using ngp_component_type = std::remove_reference_t<decltype(ngp_component)>;
  return NgpTaggedComponent<Tag, our_rank, ngp_component_type>(ngp_component);
}

namespace impl {

/// \brief Helper function to locate the component that matches a Tag
/// We assume each tag occurs only once and perform a simple linear search.
template <typename Tag, typename First, typename... Rest>
KOKKOS_FUNCTION static constexpr const auto& find_const_component_recurse_impl(const First& first,
                                                                               const Rest&... rest) {
  if constexpr (std::is_same_v<typename First::tag_type, Tag>) {
    return first;
  } else {
    return find_const_component_recurse_impl<Tag>(rest...);
  }
}

/// \brief Fetch the component corresponding to the given Tag using an index sequence
template <typename Tag, typename... Components, std::size_t... Is>
KOKKOS_FUNCTION static constexpr auto& find_const_component_impl(const core::tuple<Components...>& tuple,
                                                                 std::index_sequence<Is...>) {
  // Unpack into the
  return find_const_component_recurse_impl<Tag>(core::get<Is>(tuple)...);
}

/// \brief Helper function to locate the component that matches a Tag
/// We assume each tag occurs only once and perform a simple linear search.
template <typename Tag, typename First, typename... Rest>
KOKKOS_FUNCTION static constexpr auto& find_component_recurse_impl(First& first, Rest&... rest) {
  if constexpr (std::is_same_v<typename First::tag_type, Tag>) {
    return first;
  } else {
    return find_component_recurse_impl<Tag>(rest...);
  }
}

/// \brief Fetch the component corresponding to the given Tag using an index sequence
template <typename Tag, typename... Components, std::size_t... Is>
KOKKOS_FUNCTION static constexpr auto& find_component_impl(core::tuple<Components...>& tuple,
                                                           std::index_sequence<Is...>) {
  // Unpack into the
  return find_component_recurse_impl<Tag>(core::get<Is>(tuple)...);
}

/// \brief Helper function to determine if any components in a tuple have a given rank
template <stk::topology::rank_t rank, typename First, typename... Rest>
KOKKOS_FUNCTION static constexpr bool has_rank_recurse_impl(const First& first, const Rest&... rest) {
  if constexpr (First::rank == rank) {
    return true;
  } else {
    return has_rank_recurse_impl<rank>(rest...);
  }
}

/// \brief Determine if any components in a tuple have a given rank using an index sequence
template <stk::topology::rank_t rank, typename... Components, std::size_t... Is>
KOKKOS_FUNCTION static constexpr bool has_rank_impl(const core::tuple<Components...>& tuple,
                                                    std::index_sequence<Is...>) {
  return has_rank_recurse_impl<rank>(core::get<Is>(tuple)...);
}

/// \brief Helper function to determine if ~all~ components in a tuple have a given rank
template <stk::topology::rank_t rank, typename First, typename... Rest>
KOKKOS_FUNCTION static constexpr bool all_have_rank_recurse_impl(const First& first, const Rest&... rest) {
  if constexpr (First::rank == rank) {
    return all_have_rank_recurse_impl<rank>(rest...);
  } else {
    return false;
  }
}

/// \brief Determine if ~all~ components in a tuple have a given rank
template <stk::topology::rank_t rank, typename... Components, std::size_t... Is>
KOKKOS_FUNCTION static constexpr bool all_have_rank_impl(const core::tuple<Components...>& tuple,
                                                         std::index_sequence<Is...>) {
  return all_have_rank_recurse_impl<rank>(core::get<Components>(tuple)...);
}

/// \brief A helper class for wrapping a functor(view) with an operator()(FastMeshIndex)
template <typename NgpAggregateType, typename FunctorType>
class NgpFunctorWrapper {
 public:
  NgpFunctorWrapper(NgpAggregateType agg, const FunctorType& functor) : agg_(agg), functor_{functor} {
  }

  KOKKOS_FUNCTION
  void operator()(stk::mesh::FastMeshIndex entity_index) const {
    auto view = agg_.get_view(entity_index);
    functor_(view);
  }

 private:
  NgpAggregateType agg_;
  FunctorType functor_;
};  // NgpFunctorWrapper

}  // namespace impl

// A concept to check if a single component has a tag_type
template <typename T>
concept has_tag_type = requires { typename T::tag_type; };

template <typename T>
static constexpr bool has_tag_type_v = has_tag_type<T>;

// A concept to check if all components in a variadic list have a tag_type
template <typename... Components>
concept all_have_tags = (has_tag_type<Components> && ...);

/// \brief Helper type trait for determining if a list of tagged type contains a component with the given tag
template <typename Tag, typename... Components>
struct has_component : std::false_type {};
//
template <typename Tag, typename First, typename... Rest>
struct has_component<Tag, First, Rest...> {
  static_assert(all_have_tags<First, Rest...>, "All of the given components must have tags.");
  static constexpr bool value = std::is_same_v<typename First::tag_type, Tag> || has_component<Tag, Rest...>::value;
};
//
template <typename Tag, typename... Components>
static constexpr bool has_component_v = has_component<Tag, Components...>::value;

/// \brief Fetch the component corresponding to the given Tag (returns a const reference since the tuple is const)
template <typename Tag, typename... Components>
KOKKOS_FUNCTION static constexpr const auto& find_component(const core::tuple<Components...>& tuple) {
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  static_assert(has_component_v<Tag, Components...>,
                "Attempting to find a component that does not exist in the given tuple");
  return impl::find_const_component_impl<Tag>(tuple, std::make_index_sequence<sizeof...(Components)>{});
}

/// \brief Fetch the component corresponding to the given Tag
template <typename Tag, typename... Components>
KOKKOS_FUNCTION static constexpr auto& find_component(core::tuple<Components...>& tuple) {
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  static_assert(has_component_v<Tag, Components...>,
                "Attempting to find a component that does not exist in the given tuple");
  return impl::find_component_impl<Tag>(tuple, std::make_index_sequence<sizeof...(Components)>{});
}

/// \brief Determine if any components in a tuple have a given rank
template <stk::topology::rank_t rank, typename... Components>
KOKKOS_FUNCTION static constexpr bool has_rank(const core::tuple<Components...>& tuple) {
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  return impl::has_rank_impl<rank>(tuple, std::make_index_sequence<sizeof...(Components)>{});
}
/// \brief Determine if ~all~ components in a tuple have a given rank
template <stk::topology::rank_t rank, typename... Components>
KOKKOS_FUNCTION static constexpr bool all_have_rank(const core::tuple<Components...>& tuple) {
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  return impl::all_have_rank_impl<rank>(tuple, std::make_index_sequence<sizeof...(Components)>{});
}

/// Forward declarations:
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
class EntityView;

template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... NgpComponents>
class NgpEntityView;

/// \brief An aggregator of components
///
/// # ECS Overview
/// This class is the main entry point for the user to interact with that we refer to as components, in accordance with
/// the Entity Component System (ECS). If you aren't familiar with this pattern, ECS is an architectural pattern
/// designed to decouple data from behavior, enabling flexibility, performance, and scalability in software systems.
/// ECS's has increased in popularity since the late 2000s and is now gaining widespread use within the gaming industry,
/// with addoption by Minecraft (via the elegant EnTT library) and engines like Unity ECS and Unreal ECS.
///
/// Unity states: "ECS (Entity Component System) is a data-oriented framework [that] scales processing performance,
/// enabling experienced creators to build more ambitious games with an unprecedented level of control and determinism."
///
///
/// # ECS Core Concepts
///   *Entities* are simple, unique identifiers—purely conceptual representations of "things" in your application.
///   They have no data or behavior themselves and are rather lightweight, often consisting of nothing more than a
///   unique ID. This makes getting and manipulating lists of entities far faster then getting and manipulating lists
///   of objects.
///
///   *Components* are data-only structures that can be assigned to entities. They contain no behavior, only data.
///   They typically represent a single aspect of an entity's state, such as position, velocity, mass, acceleration.
///   As with most ECS designs, components may be added or removed from entities at runtime.
///
///   Importantly, ECS typically discourages hard-coding collections of component at compile-time. Unlike
///   polymorphism, we do not offer a RigidBody class or Sphere class; rather, any entity that ~looks~ like a sphere
///   (i.e. has a component that ~looks~ like a radius and a component that ~looks~ like a center) ~is~ a sphere. This
///   does not require an explicit Sphere class and attempting to create one would be counter to the ECS design. This
///   gives a level of flexibility that is difficult to achieve with traditional object-oriented programming and allows
///   for more optimized memory access patterns and cache coherency.
///
///   *Systems* are functions or processes that operate on entities possessing specific sets of components. They are
///   typically "free" functions that are outside of any class hierarchy or inheritance chain. Unlike functions in a
///   class, users can add "free" functions meant to operate on entities with specific components without needing to
///   modify some hard-coded class definition. This is more flexible to extension, as users can add new free functions
///   without needing to modify existing classes. In many ways, we like to think of ECS as a runtime-extensible
///   deconstructed class hierarchy.
///
///
/// # STK's flavor of ECS
/// STK's domain model can be seen as an extension of ECS, adding to it the concept of connections between entities of
/// different *ranks* and the ability for entities to possess a graph *topology*, statically defining its connectivity.
/// This concept of rank and topology is common in mesh-based or molecular dynamics simulations, where we have nodes,
/// edges, faces, and elements that connect to each other in a hierarchical manner. Unlike simplistic ECS systems, like
/// EnTT, adding ranks and topologies complicates the design, necessitating additional features and care.
///
/// STK introduces the concept of Parts, Fields, and Selectors.
/// data.
///
///   *Parts* are collections of entities that share the same properties. They may possess a topology, requiring that
///   all entities in that part of the same rank as the topology, have said topology. They may instead possess only a
///   rank, allowing them to hold any entities of that rank. Or they may possess neither, allowing them to be used as
///   Assemblies of any entities. Importantly, Parts may be subsets of other Parts, allowing for hierarchical
///   organization at runtime. Parts (by default) have inherited part membership, meaning that if an entity within the
///   part of its primary rank connects to an entity of lower rank, that entity is also considered to be within the
///   part.
///
///   *Fields* are collections of ranked data that can be assigned to any number of Parts. These Fields can be seen as
///   one type of component. They have a rank, a name, and a type. If a Part has a Field, then all entities in that
///   Part of the same rank as the Field pickup that Field.
///
///   *Selectors* (also called groups or views in other ECS systems) are a means of identifying a subset of entities. In
///   STK, Selectors are formed using set arithmetic applied to the Parts and Fields. For example, a Selector might
///   abstractly represent "all entities of Part A that have Field B but are not in Part C". Selectors are used to
///   define the scope of Systems such as "for all entities in Selector X, do Y". In practice, the smallest unit of work
///   in STK is defined by a Selector (Parts themselves may act as Selectors). If there is ever a need to iterate over a
///   specific subset set of entities that cannot be fetched by set arithmetic applied to your current Parts and Fields,
///   you likely need to create a new Part. That said, sometimes it's more efficient to iterate over a larger set of
///   entities and use a conditional to filter out the entities you don't want. If the subset of entities you wish to
///   iterate over is large, then using a new Part is likely more efficient.
///
/// # Accessors
/// We extend STK's domain model to include the concept of Aggregates and Accessors, as an organizational layer above
/// Parts, Fields, and Selectors, meant to abstract away access patterns into an entity's data while reducing
/// boilerplate code. Similar to Field and Parts, we ~assemble~ Aggregates at runtime.
///
/// Notably, Accessors overcome the following limitation of STK's domain model:
///   It's common to have a collection of entities within some set of Parts for which we want to store a shared value.
///   This might include a collection of spheres with the same radius and material properties. Similarly, one might want
///   to have a single shared material per part. Simply because a collection of spheres share a radius rather than
///   having a radius stored within a Field shouldn't impact the design of systems meant to operate on spheres, and yet
///   STK offers no means to abstract away shared vs non-shared data. This is a direct consequence of the lack of
///   separation of concerns with regard to data storage and access patterns.
///
/// Accessors provide an interface through which data beyond just Fields may be treated as components and accessed in a
/// unified manner. If a user starts with an aggregate for spheres that have a shared radius and then decides later to
/// switch to using a Field of radii, they need only update the aggregate's definition. The systems that act on the
/// aggregate will remain unchanged, as how the data is accesses does not concern them. Notably, accessors are ~views~
/// into data, i.e. they should not ~hold~ the data they access. They are meant to be cheap to construct and trivial to
/// copy (just like Kokkos::Views). Interface-wise, Accessors must provide a sync_to_host, sync_to_device,
/// modify_on_host, and modify_on_device method the same as STK's NgpField. When called, these methods should
/// synchronize the data to the appropriate space/mark the data as modified. Synchronization should be a no-op if the
/// data is up-to-date on the requested space.
///
/// Our Accessors, the data they access, and the return type of the Accessor's get_view method are as follows:
///          Component Name                :              Data it accesses             ->         Return Type
///   ScalarFieldComponent                 :  Field<scalar_t>                          ->  scalar_t&
///   Vector3FieldComponent                :  Field<scalar_t>                          ->  Vector3View<scalar_t>
///   Matrix3FieldComponent                :  Field<scalar_t>                          ->  Matrix3View<scalar_t>
///   QuaternionFieldComponent             :  Field<scalar_t>                          ->  QuaternionView<scalar_t>
///   AABBFieldComponent                   :  Field<scalar_t>                          ->  AABBView<scalar_t>
///   SharedViewComponent (size 1)         :  mundy::NgpVector<SharedType>             ->  SharedType&
///   PartMappedComponent<OtherComponent>  :  Kokkos::Map<PartOrdinal, OtherComponent> -> OtherComponent's return type
///
/// \note mundy::NgpVector is just a wrapper for a Kokkos::View and its host mirror with the necessary sync/modify
/// methods. Just like Kokkos::Views, it takes ownership of the data it accesses. We cannot simply store a reference to
/// the given SharedType, as we have no guarantee that the data will remain valid for the lifetime of the Accessor nor
/// are we able to copy a reference to host memory to the device.
///
///
/// # Aggregates
/// Aggregates are just collections of accessors(components) that can be accessed via a unified EntityView. The
/// components are "tagged". That is, they are associated with some type that is used to fetch the data. That type is
/// often nothing more than an emtpy struct, but it can be used to differentiate between components that have the same
/// underlying type. This is opposed to creating a SphereAggregate which stores a radius_field and a center_field.
/// Instead, we can access the radius and center accessors via their tags.
///
/// The Aggregate class can be constructed directly "Aggregate<topology, rank>(bulk_data, selector)", but we also offer
/// some non-member helper functions to streamline this process.
///   - Use "make_aggregate<topology>(bulk_data, selector)" to avoid the need to specify the rank redundantly.
///   - Use "make_ranked_aggregate<rank>(bulk_data, selector)" if the entities don't have a topology.
///
/// Adding components to an aggregate is done via the fluent interface "add_component<Tag>(accessor)",
/// which returns a new aggregate with the added component. This allows for easy chaining of components,
/// as seen in the below example.
///
/// Aggregates can then be used to fetch an EntityView for a given entity, which can be used to access the entity's
/// components. It offers basic information like the rank and topology of the entity, as well as the entity itself
/// and access to its connections. To then fetch the components, use the get<Tag>() method for components of the
/// same rank as the entity, or get<Tag>(connectivity_ordinal) for components of a lower rank then the entity. To then
/// apply functors to all entities in the aggregate, use the for_each method. This method runs the given functor on the
/// EntityView of each entity in the aggregate.
///
/// \note Accessors and aggregates are not a replacement for STK's Field and Part system. They are an abstraction layer
///   that sits above and beside it. We choose to use Aggregates to organize our data and Accessors to access it, but
///   you may also directly act on accessors or directly on fields and parts. The choice is yours.
///
///
/// # Example Usage
/// \code {.cpp}
///    // We'll assume that there exists an elem1 within a Spheres part of PARTICLE topology connected to a node1
///    //   with a NODE_RANK center_field and a shared ELEM_RANK radius. Both have double type.
///
///    // Create the accessors
///    auto center_accessor = make_scalar_field_accessor(center_field);
///    auto radius_accessor = make_shared_view_accessor(mundy::NgpVector<float>{radius});  // Radius is copied
///
///    // Fetch the data for the entity via the accessor's operator()
///    Vector3View<double> center = center_accessor(elem1);
///    double& radius = radius_accessor(node1);
///
///    // Create an aggregate for the spheres
///    auto collision_sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
///            .add_component<CENTER>(center_accessor)
///            .add_component<COLLISION_RADIUS>(radius_accessor);
///
///    // Sync the data to the device and mark it as modified
///    collision_sphere_data.sync_to_device<CENTER, COLLISION_RADIUS>();
///    collision_sphere_data.modify_on_device<CENTER, COLLISION_RADIUS>();
///
///    // Do the same directly via the accessors
///    center_accessor.sync_to_device();
///    radius_accessor.modify_on_device();
///    collision_sphere_data.get_component<CENTER>().sync_to_device();
///    collision_sphere_data.get_component<COLLISION_RADIUS>().modify_on_device();
///
///    // Get an accessor-independent view of the entity's data
///    auto sphere_view = collision_sphere_data.get_view(entity1);
///    sphere_view.entity();    // Returns entity1
///    sphere_view.rank();      // Returns ELEM_RANK
///    sphere_view.topology();  // Returns PARTICLE
///    unsigned center_node_con_ordinal = 0;
///    Vector3View<double> also_center = sphere_view.get<CENTER>(center_node_con_ordinal);
///    double& also_radius = sphere_view.get<COLLISION_RADIUS>();
///
///    // Apply a functor to all entities in the aggregate
///    collision_sphere_data.for_each([](auto& other_sphere_view) {
///        Vector3View<double> c = other_sphere_view.get<CENTER>();
///        double& r = other_sphere_view.get<COLLISION_RADIUS>();
///        std::cout << "Center = " << c << ", Radius = " << r << std::endl;
///    });
///
///    // Run a functor move_spheres<CenterTag, RadiusTag> on all entities in the aggregate
///    collision_sphere_data.for_each(move_spheres<CENTER, COLLISION_RADIUS>{});
///
///   // Use a functor to move all spheres
///   move_spheres<CENTER, COLLISION_RADIUS>{}.apply_to(collision_sphere_data);
///
///    // Directly use accessors without an aggregate
///    stk::mesh::for_each_entity_run(bulk_data, stk::topology::ELEM_RANK, selector,
///       [center_accessor, radius_accessor](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &entity) {
///           Vector3View<double> c = center_accessor(entity);
///           double& r = radius_accessor(entity);
///           std::cout << "Center = " << c << ", Radius = " << r << std::endl;
///       });
///
///   // Directly pass accessors to a free-function move_spheres2 templated by the accessor types
///   move_spheres2(center_accessor, radius_accessor);
/// \endcode
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
class Aggregate {
 public:
  static_assert(all_have_tags<Components...>, "All of the given components must have tags.");
  using ComponentsTuple = core::tuple<Components...>;

  //! \name Constructors
  //@{

  /// \brief Construct an Aggregate that has no components
  Aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector)
    requires(sizeof...(Components) == 0)
      : bulk_data_(bulk_data), selector_(std::move(selector)), components_{} {
  }

  /// \brief Construct an Aggregate that has the given components
  Aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector, ComponentsTuple components)
    requires(sizeof...(Components) > 0)
      : bulk_data_(bulk_data), selector_(std::move(selector)), components_(std::move(components)) {
  }

  /// \brief Default copy/move/assign constructors
  Aggregate(const Aggregate&) = default;
  Aggregate(Aggregate&&) = default;
  Aggregate& operator=(const Aggregate&) = default;
  Aggregate& operator=(Aggregate&&) = default;
  //@}

  //! \name Accessors
  //@{

  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }
  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }
  const stk::mesh::MetaData& mesh_meta_data() const {
    return bulk_data_.mesh_meta_data();
  }
  const stk::mesh::Selector& selector() const {
    return selector_;
  }
  //@}

  /// \brief Add a component (fluent interface):
  template <typename Tag, stk::topology::rank_t component_rank, typename NewComponent>
  auto add_component(NewComponent new_component) const {
    auto new_tagged_comp = TaggedComponent<Tag, component_rank, NewComponent>{std::move(new_component)};
    auto new_tuple = core::tuple_cat(components_, core::make_tuple(new_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = Aggregate<OurTopology, OurRank, Components..., decltype(new_tagged_comp)>;
    return NewType(bulk_data_, selector_, new_tuple);
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  const auto& get_component() const {
    return find_component<Tag>(components_);
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  auto& get_component() {
    return find_component<Tag>(components_);
  }

  /// \brief Synchronize the components marked by the given tags to the device
  template <typename... TagsToSync>
  void sync_to_device() {
    (get_component<TagsToSync>().sync_to_device(), ...);
  }

  /// \brief Synchronize the components marked by the given tags to the host
  template <typename... TagsToSync>
  void sync_to_host() {
    (get_component<TagsToSync>().sync_to_host(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the device
  template <typename... TagsToModify>
  void modify_on_device() {
    (get_component<TagsToModify>().modify_on_device(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the host
  template <typename... TagsToModify>
  void modify_on_host() {
    (get_component<TagsToModify>().modify_on_host(), ...);
  }

  /// \brief Get an EntityView for the given entity
  EntityView<OurTopology, OurRank, Components...> get_view(stk::mesh::Entity entity) const {
    return EntityView<OurTopology, OurRank, Components...>(*this, entity);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate that is also in the given
  /// subset selector
  template <typename Functor>
  void for_each(const stk::mesh::Selector& subset_selector, const Functor& f) const {
    // Only loop over the set intersection of the agg's selector and the subset selector
    auto agg = *this;
    stk::mesh::Selector sel = agg.selector() & subset_selector;

    auto wrapped_functor = [&agg, &f](const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& entity) {
      auto view = agg.get_view(entity);
      f(view);
    };

    ::mundy::mesh::for_each_entity_run(agg.bulk_data(), agg.rank(), sel, wrapped_functor);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate
  template <typename Functor>
  void for_each(const Functor& f) const {
    auto agg = *this;

    auto wrapped_functor = [&agg, &f]([[maybe_unused]] const stk::mesh::BulkData& bulk_data,
                                      const stk::mesh::Entity& entity) {
      auto view = agg.get_view(entity);
      f(view);
    };

    ::mundy::mesh::for_each_entity_run(agg.bulk_data(), agg.rank(), agg.selector(), wrapped_functor);
  }

 private:
  //! \name Private members
  //@{

  const stk::mesh::BulkData& bulk_data_;
  stk::mesh::Selector selector_;
  ComponentsTuple components_;
  //@}
};  // Aggregate

template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... NgpComponents>
class NgpAggregate {
 public:
  using NgpComponentsTuple = core::tuple<NgpComponents...>;
  static_assert(all_have_tags<NgpComponents...>, "All of the given components must have tags.");

  //! \name Constructors
  //@{

  /// \brief Default constructor
  NgpAggregate() : ngp_mesh_{}, host_selector_{}, ngp_components_{} {
  }

  /// \brief Construct an Aggregate that has no components
  NgpAggregate(stk::mesh::NgpMesh ngp_mesh, stk::mesh::Selector selector)
    requires(sizeof...(NgpComponents) == 0)
      : ngp_mesh_(ngp_mesh), host_selector_(std::move(selector)), ngp_components_{} {
  }

  /// \brief Construct an Aggregate that has the given components
  NgpAggregate(stk::mesh::NgpMesh ngp_mesh, stk::mesh::Selector selector, NgpComponentsTuple ngp_components)
    requires(sizeof...(NgpComponents) > 0)
      : ngp_mesh_(ngp_mesh), host_selector_(std::move(selector)), ngp_components_(std::move(ngp_components)) {
  }

  /// \brief Default move/copy/assign constructors
  NgpAggregate(NgpAggregate&& other)
      : ngp_mesh_(other.ngp_mesh_), host_selector_(other.host_selector_), ngp_components_(other.ngp_components_) {
  }
  NgpAggregate(const NgpAggregate& other)
      : ngp_mesh_(other.ngp_mesh_), host_selector_(other.host_selector_), ngp_components_(other.ngp_components_) {
  }
  NgpAggregate& operator=(NgpAggregate&& other) {
    ngp_mesh_ = other.ngp_mesh_;
    host_selector_ = other.host_selector_;
    ngp_components_ = other.ngp_components_;
    return *this;
  }
  NgpAggregate& operator=(const NgpAggregate& other) {
    ngp_mesh_ = other.ngp_mesh_;
    host_selector_ = other.host_selector_;
    ngp_components_ = other.ngp_components_;
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  const stk::mesh::BulkData& bulk_data() const {
    return ngp_mesh_.get_bulk_on_host();
  }

  const stk::mesh::MetaData& mesh_meta_data() const {
    return ngp_mesh_.get_bulk_on_host().mesh_meta_data();
  }

  const stk::mesh::Selector& selector() const {
    return host_selector_;
  }
  //@}

  /// \brief Add a component (fluent interface):
  /// TODO(palmerb4): If we do decide to use get_updated_ngp_aggregate with references, then this function will
  ///   need removed, as Aggregates managing the lifetime of NGP components means users should construct them
  template <typename Tag, stk::topology::rank_t component_rank, typename NewNgpComponent>
  auto add_component(NewNgpComponent new_ngp_component) const {
    auto new_ngp_tagged_comp = NgpTaggedComponent<Tag, component_rank, NewNgpComponent>{std::move(new_ngp_component)};
    auto new_tuple = core::tuple_cat(ngp_components_, core::make_tuple(new_ngp_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = NgpAggregate<OurTopology, OurRank, NgpComponents..., decltype(new_ngp_tagged_comp)>;
    return NewType(ngp_mesh_, host_selector_, new_tuple);
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION const auto& get_component() const {
    return find_component<Tag>(ngp_components_);
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION auto& get_component() {
    return find_component<Tag>(ngp_components_);
  }

  /// \brief Synchronize the components marked by the given tags to the device
  template <typename... TagsToSync>
  void sync_to_device() {
    (get_component<TagsToSync>().sync_to_device(), ...);
  }

  /// \brief Synchronize the components marked by the given tags to the host
  template <typename... TagsToSync>
  void sync_to_host() {
    (get_component<TagsToSync>().sync_to_host(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the device
  template <typename... TagsToModify>
  void modify_on_device() {
    (get_component<TagsToModify>().modify_on_device(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the host
  template <typename... TagsToModify>
  void modify_on_host() {
    (get_component<TagsToModify>().modify_on_host(), ...);
  }

  /// \brief Get an EntityView for the given entity
  KOKKOS_INLINE_FUNCTION
  NgpEntityView<OurTopology, OurRank, NgpComponents...> get_view(stk::mesh::FastMeshIndex entity_index) const {
    return NgpEntityView<OurTopology, OurRank, NgpComponents...>(ngp_mesh_, ngp_components_, entity_index);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate that is also in the given
  /// subset selector
  template <typename Functor>
  inline void for_each(const stk::mesh::Selector& subset_selector, const Functor& f) const {
    // Only loop over the set intersection of the agg's selector and the subset selector
    using our_t = NgpAggregate<OurTopology, OurRank, NgpComponents...>;
    our_t agg = *this;

    auto local_ngp_mesh = agg.ngp_mesh();
    impl::NgpFunctorWrapper<our_t, Functor> wrapper(agg, f);
    stk::mesh::Selector sel = agg.selector() & subset_selector;
    ::mundy::mesh::for_each_entity_run(local_ngp_mesh, agg.rank(), sel, wrapper);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate
  template <typename Functor>
  inline void for_each(const Functor& f) const {
    using our_t = NgpAggregate<OurTopology, OurRank, NgpComponents...>;
    our_t agg = *this;

    auto local_ngp_mesh = agg.ngp_mesh();
    impl::NgpFunctorWrapper<our_t, Functor> wrapper(agg, f);
    ::mundy::mesh::for_each_entity_run(local_ngp_mesh, agg.rank(), agg.selector(), wrapper);
  }

 private:
  //! \name Private members
  //@{

  stk::mesh::NgpMesh ngp_mesh_;
  stk::mesh::Selector host_selector_;
  NgpComponentsTuple ngp_components_;
  //@}
};  // NgpAggregate

/// \brief A view into the components, connections, and meta-information for an entity
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
class EntityView {
 public:
  /// \brief Construct an EntityView for the given entity
  /// TODO(palmerb4) Optimize for reuse of connectivity.
  EntityView(const Aggregate<OurTopology, OurRank, Components...>& parent, stk::mesh::Entity entity)
      : parent_(parent), entity_(entity) {
    MUNDY_THROW_ASSERT(parent.bulk_data().is_valid(entity), std::runtime_error,
                       "EntityView created with invalid entity");
    MUNDY_THROW_ASSERT(parent.bulk_data().entity_rank(entity) == OurRank, std::runtime_error,
                       fmt::format("EntityView created with entity of rank {} but aggregate has rank {}",
                                   parent.bulk_data().entity_rank(entity), OurRank));
  }

 private:
  const Aggregate<OurTopology, OurRank, Components...>& parent_;
  stk::mesh::Entity entity_;

 public:
  /// \brief Fetch the entity that we view
  stk::mesh::Entity entity() const {
    return entity_;
  }

  /// \brief Fetch the rank of the entity that we view
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }

  /// \brief Fetch the topology of the entity that we view
  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
  decltype(auto) get() {
    static_assert(
        std::decay_t<decltype(parent_.template get_component<Tag>())>::rank == OurRank,
        "EntityView::get() called with a tag that does not correspond to a component of the same rank as the entity");
    auto& comp = parent_.template get_component<Tag>();
    return comp(entity_);
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
  decltype(auto) get() const {
    static_assert(
        std::decay_t<decltype(parent_.template get_component<Tag>())>::rank == OurRank,
        "EntityView::get() called with a tag that does not correspond to a component of the same rank as the entity");
    auto& comp = parent_.template get_component<Tag>();
    return comp(entity_);
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
  decltype(auto) get(unsigned connectivity_ordinal) {
    static_assert(std::decay_t<decltype(parent_.template get_component<Tag>())>::rank != OurRank,
                  "EntityView::get(ordinal) called with a tag that does not correspond to a component of a different "
                  "rank than the entity");

    using TaggedComponentType = std::decay_t<decltype(parent_.template get_component<Tag>())>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = parent_.template get_component<Tag>();

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().num_connectivity(entity_, comp_rank) > connectivity_ordinal, std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but entity has only {} "
                    "connectivities of rank {}",
                    connectivity_ordinal, parent_.bulk_data().num_connectivity(entity_, comp_rank), comp_rank));

    // TODO: The following assumes that the entity is connected to all of its possible lower rank entities
    // such that the list of connected entities can be indexed by the connectivity_ordinal.
    const stk::mesh::Entity& connected_entity = parent_.bulk_data().begin(entity_, comp_rank)[connectivity_ordinal];

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().is_valid(connected_entity), std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but connected entity is invalid",
                    connectivity_ordinal));

    return comp(connected_entity);
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
  decltype(auto) get(unsigned connectivity_ordinal) const {
    static_assert(std::decay_t<decltype(parent_.template get_component<Tag>())>::rank != OurRank,
                  "EntityView::get(ordinal) called with a tag that does not correspond to a component of a different "
                  "rank than the entity");

    using TaggedComponentType = std::decay_t<decltype(parent_.template get_component<Tag>())>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = parent_.template get_component<Tag>();

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().num_connectivity(entity_, comp_rank) > connectivity_ordinal, std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but entity has only {} "
                    "connectivities of rank {}",
                    connectivity_ordinal, parent_.bulk_data().num_connectivity(entity_, comp_rank), comp_rank));

    const stk::mesh::Entity& connected_entity = parent_.bulk_data().begin(entity_, comp_rank)[connectivity_ordinal];

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().is_valid(connected_entity), std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but connected entity is invalid",
                    connectivity_ordinal));

    return comp(connected_entity);
  }
};  // EntityView

/// \brief A view into the components, connections, and meta-information for an entity
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... NgpComponents>
class NgpEntityView {
 public:
  using NgpComponentsTuple = core::tuple<NgpComponents...>;

  /// \brief Construct an EntityView for the given entity
  /// TODO(palmerb4) Optimize for reuse of connectivity.
  KOKKOS_INLINE_FUNCTION
  NgpEntityView(const stk::mesh::NgpMesh& ngp_mesh, const NgpComponentsTuple& components,
                stk::mesh::FastMeshIndex entity_index)
      : ngp_mesh_(ngp_mesh), ngp_components_(components), entity_index_(entity_index) {
  }

 private:
  const stk::mesh::NgpMesh& ngp_mesh_;
  const NgpComponentsTuple& ngp_components_;
  stk::mesh::FastMeshIndex entity_index_;

 public:
  /// \brief Fetch the entity that we view
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex entity_index() const {
    return entity_index_;
  }

  /// \brief Fetch the identifier of the entity that we view
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityId entity_id() const {
    return ngp_mesh_.identifier(ngp_mesh_.get_entity(OurRank, entity_index_));
  }

  /// \brief Fetch the rank of the entity that we view
  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }

  /// \brief Fetch the topology of the entity that we view
  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION decltype(auto) get() {
    static_assert(
        std::decay_t<decltype(find_component<Tag>(ngp_components_))>::rank == OurRank,
        "EntityView::get() called with a tag that does not correspond to a component of the same rank as the entity");
    auto& comp = find_component<Tag>(ngp_components_);
    return comp(entity_index_);
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION decltype(auto) get() const {
    static_assert(
        std::decay_t<decltype(find_component<Tag>(ngp_components_))>::rank == OurRank,
        "EntityView::get() called with a tag that does not correspond to a component of the same rank as the entity");
    auto& comp = find_component<Tag>(ngp_components_);
    return comp(entity_index_);
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION decltype(auto) get(unsigned connectivity_ordinal) {
    static_assert(std::decay_t<decltype(find_component<Tag>(ngp_components_))>::rank != OurRank,
                  "EntityView::get(ordinal) called with a tag that does not correspond to a component of a different "
                  "rank than the entity");

    using TaggedComponentType = std::decay_t<decltype(find_component<Tag>(ngp_components_))>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = find_component<Tag>(ngp_components_);

    // TODO: The following assumes that the entity is connected to all of its possible lower rank entities
    // such that the list of connected entities can be indexed by the connectivity_ordinal.
    const auto connected_entities = ngp_mesh_.get_connected_entities(OurRank, entity_index_, comp_rank);

    MUNDY_THROW_ASSERT(connected_entities.size() > connectivity_ordinal, std::runtime_error,
                       "EntityView::get() called with a connectivity ordinal that exceeds the number of connected "
                       "entities of the tag rank");

    const stk::mesh::FastMeshIndex connected_entity_index =
        ngp_mesh_.fast_mesh_index(connected_entities[connectivity_ordinal]);
    return comp(connected_entity_index);
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
  KOKKOS_INLINE_FUNCTION decltype(auto) get(unsigned connectivity_ordinal) const {
    static_assert(std::decay_t<decltype(find_component<Tag>(ngp_components_))>::rank != OurRank,
                  "EntityView::get(ordinal) called with a tag that does not correspond to a component of a different "
                  "rank than the entity");

    using TaggedComponentType = std::decay_t<decltype(find_component<Tag>(ngp_components_))>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = find_component<Tag>(ngp_components_);
    const auto connected_entities = ngp_mesh_.get_connected_entities(OurRank, entity_index_, comp_rank);

    MUNDY_THROW_ASSERT(connected_entities.size() > connectivity_ordinal, std::runtime_error,
                       "EntityView::get() called with a connectivity ordinal that exceeds the number of connected "
                       "entities of the tag rank");

    const stk::mesh::FastMeshIndex connected_entity_index =
        ngp_mesh_.fast_mesh_index(connected_entities[connectivity_ordinal]);
    return comp(connected_entity_index);
  }
};  // NgpEntityView

/// \brief Make an aggregate for the given topology
template <stk::topology::topology_t OurTopology>
auto make_aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector) {
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;
  return Aggregate<OurTopology, rank>(bulk_data, selector);
}

/// \brief Make an aggregate for the given rank
template <stk::topology::rank_t OurRank>
auto make_ranked_aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector) {
  return Aggregate<stk::topology::INVALID_TOPOLOGY, OurRank>(bulk_data, selector);
}

/// \brief Get a component of the given aggregate (const)
/// This simply calls the get_component method of the given aggregate and solely exists so you don't need to write
///  "aggregate. template get_component<Tag>()" every time you want to fetch a component. Instead,
/// you use "get_component<Tag>(aggregate)". Same concept as std::get<N>(tuple).
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
const auto& get_component(const Aggregate<OurTopology, OurRank, Components...>& aggregate) {
  return aggregate.template get_component<Tag>();
}

/// \brief Get a component of the given aggregate
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
auto& get_component(Aggregate<OurTopology, OurRank, Components...>& aggregate) {
  return aggregate.template get_component<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view (const)
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(const EntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(EntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(const EntityView<OurTopology, OurRank, Components...>& entity_view, unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(EntityView<OurTopology, OurRank, Components...>& entity_view, unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief Get the data tagged by the given tag from the given entity view (const)
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(const NgpEntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(NgpEntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(const NgpEntityView<OurTopology, OurRank, Components...>& entity_view,
                                          unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(NgpEntityView<OurTopology, OurRank, Components...>& entity_view,
                                          unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief A helper function for getting the NGP aggregate from a regular aggregate
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... TaggedComponents>
auto get_updated_ngp_aggregate(const Aggregate<OurTopology, OurRank, TaggedComponents...>& aggregate) {
  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(aggregate.bulk_data());

  auto ngp_components = core::make_tuple(
      get_updated_ngp_component(aggregate.template get_component<typename TaggedComponents::tag_type>())...);

  return NgpAggregate<OurTopology, OurRank,
                      std::decay_t<decltype(get_updated_ngp_component(
                          aggregate.template get_component<typename TaggedComponents::tag_type>()))>...>(
      ngp_mesh, aggregate.selector(), ngp_components);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_AGGREGATES_HPP_
