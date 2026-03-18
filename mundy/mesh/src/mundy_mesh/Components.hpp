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
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/ForEachEntity.hpp>    // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/NgpAccessorExpr.hpp>  // for mundy::mesh::AccessorExpr and EntityExprBase
#include <mundy_mesh/fmt_stk_types.hpp>    // for STK-compatible fmt::format
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

  void clear_host_sync_state() {
    field_base_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    field_base_.clear_device_sync_state();
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
    host_field_base().sync_to_device();
  }

  void sync_to_host() {
    host_field_base().sync_to_host();
  }

  void modify_on_device() {
    host_field_base().modify_on_device();
  }

  void modify_on_host() {
    host_field_base().modify_on_host();
  }

  void clear_host_sync_state() {
    host_field_base().clear_host_sync_state();
  }

  void clear_device_sync_state() {
    host_field_base().clear_device_sync_state();
  }

  const stk::mesh::FieldBase& host_field_base() {
    MUNDY_THROW_ASSERT(host_field_base_, std::runtime_error, "host_field_base_ is null");
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
    ValueType* data_ptr = stk::mesh::field_data(field_, entity);
    MUNDY_THROW_ASSERT(data_ptr, std::runtime_error, "Field data is null");
    unsigned num_scalars = stk::mesh::field_scalars_per_entity(field_, entity);
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
  using our_t = NgpFieldComponent<NgpFieldType>;

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

  /// \brief Calling operator()(entity_expr) on any accessor will return an AccessorExpr
  /// Example:
  ///   auto v3_accessor = Vector3FieldComponent(v3_field);
  ///   EntityExpr all_nodes(node_selector, stk::topology::NODE_RANK);
  ///   auto get_v3_expr = v3_accessor(all_nodes);
  // template <class EntityExpr>
  // KOKKOS_INLINE_FUNCTION auto operator()(const EntityExprBase<EntityExpr>& e) const {
  //   MUNDY_THROW_REQUIRE(e.rank() == ngp_field_.get_rank(), std::runtime_error,
  //                       fmt::format("Attempting to access field of rank {} on entity expression of rank {}",
  //                                   ngp_field_.get_rank(), e.rank()));

  //   return AccessorExpr<our_t, EntityExpr>(*this, e.self());
  // }

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

  void clear_host_sync_state() {
    ngp_field_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    ngp_field_.clear_device_sync_state();
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
  using our_t = NgpScalarFieldComponent<NgpFieldType>;

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

  /// \brief Calling operator()(entity_expr) on any accessor will return an AccessorExpr
  /// Example:
  ///   auto v3_accessor = Vector3FieldComponent(v3_field);
  ///   EntityExpr all_nodes(node_selector, stk::topology::NODE_RANK);
  ///   auto get_v3_expr = v3_accessor(all_nodes);
  // template <class EntityExpr>
  // KOKKOS_INLINE_FUNCTION auto operator()(const EntityExprBase<EntityExpr>& e) const {
  //   MUNDY_THROW_REQUIRE(e.rank() == ngp_field_.get_rank(), std::runtime_error,
  //                       fmt::format("Attempting to access field of rank {} on entity expression of rank {}",
  //                                   ngp_field_.get_rank(), e.rank()));
  //   return AccessorExpr<our_t, EntityExpr>(*this, e.self());
  // }

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

  void clear_host_sync_state() {
    ngp_field_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    ngp_field_.clear_device_sync_state();
  }
#endif

 private:
  NgpFieldType ngp_field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpScalarFieldComponent<NgpFieldType>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
};  // NgpScalarFieldComponent

template <typename ScalarType, size_t N>
class VectorFieldComponent : public FieldComponentBase {
 public:
  VectorFieldComponent(stk::mesh::Field<ScalarType>& field) : FieldComponentBase(field), field_(field) {
  }

  /// \brief Default copy/move/assign constructors
  VectorFieldComponent(const VectorFieldComponent&) = default;
  VectorFieldComponent(VectorFieldComponent&&) = default;
  VectorFieldComponent& operator=(const VectorFieldComponent&) = default;
  VectorFieldComponent& operator=(VectorFieldComponent&&) = default;

  inline decltype(auto) operator()(stk::mesh::Entity entity) const {
    return vector_field_data<N>(field_, entity);
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
  decltype(std::declval<VectorFieldComponent<ScalarType, N>>().operator()(std::declval<stk::mesh::Entity>()));
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
class NgpVectorFieldComponent : public NgpFieldComponentBase {
 public:
  using our_t = NgpVectorFieldComponent<NgpFieldType, N>;

  NgpVectorFieldComponent() = default;
  NgpVectorFieldComponent(NgpFieldType ngp_field)
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
      : NgpFieldComponentBase(*ngp_field.get_field_base()),  // Directly store the field base
#else
      : NgpFieldComponentBase(),
#endif
        ngp_field_(ngp_field) {
  }

  /// \brief Default copy/move/assign constructors
  NgpVectorFieldComponent(const NgpVectorFieldComponent&) = default;
  NgpVectorFieldComponent(NgpVectorFieldComponent&&) = default;
  NgpVectorFieldComponent& operator=(const NgpVectorFieldComponent&) = default;
  NgpVectorFieldComponent& operator=(NgpVectorFieldComponent&&) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(stk::mesh::FastMeshIndex entity_index) const {
    return vector_field_data<N>(ngp_field_, entity_index);
  }

  /// \brief Calling operator()(entity_expr) on any accessor will return an AccessorExpr
  /// Example:
  ///   auto v3_accessor = Vector3FieldComponent(v3_field);
  ///   EntityExpr all_nodes(node_selector, stk::topology::NODE_RANK);
  ///   auto get_v3_expr = v3_accessor(all_nodes);
  // template <class EntityExpr>
  // KOKKOS_INLINE_FUNCTION auto operator()(const EntityExprBase<EntityExpr>& e) const {
  //   MUNDY_THROW_REQUIRE(e.rank() == ngp_field_.get_rank(), std::runtime_error,
  //                       fmt::format("Attempting to access field of rank {} on entity expression of rank {}",
  //                                   ngp_field_.get_rank(), e.rank()));
  //   return AccessorExpr<our_t, EntityExpr>(*this, e.self());
  // }

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

  void clear_host_sync_state() {
    ngp_field_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    ngp_field_.clear_device_sync_state();
  }
#endif

 private:
  NgpFieldType ngp_field_;

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<NgpVectorFieldComponent<NgpFieldType, N>>().operator()(
      std::declval<stk::mesh::FastMeshIndex>()));
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
  using our_t = NgpMatrix3FieldComponent<NgpFieldType>;

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

  /// \brief Calling operator()(entity_expr) on any accessor will return an AccessorExpr
  /// Example:
  ///   auto v3_accessor = Vector3FieldComponent(v3_field);
  ///   EntityExpr all_nodes(node_selector, stk::topology::NODE_RANK);
  ///   auto get_v3_expr = v3_accessor(all_nodes);
  // template <class EntityExpr>
  // KOKKOS_INLINE_FUNCTION auto operator()(const EntityExprBase<EntityExpr>& e) const {
  //   MUNDY_THROW_REQUIRE(e.rank() == ngp_field_.get_rank(), std::runtime_error,
  //                       fmt::format("Attempting to access field of rank {} on entity expression of rank {}",
  //                                   ngp_field_.get_rank(), e.rank()));
  //   return AccessorExpr<our_t, EntityExpr>(*this, e.self());
  // }

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

  void clear_host_sync_state() {
    ngp_field_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    ngp_field_.clear_device_sync_state();
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
  using our_t = NgpQuaternionFieldComponent<NgpFieldType>;

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

  void clear_host_sync_state() {
    ngp_field_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    ngp_field_.clear_device_sync_state();
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
  using our_t = NgpAABBFieldComponent<NgpFieldType>;

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

  void clear_host_sync_state() {
    ngp_field_.clear_host_sync_state();
  }

  void clear_device_sync_state() {
    ngp_field_.clear_device_sync_state();
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

 public:
  /// \brief The view type returned by operator()
  using view_t = decltype(std::declval<TaggedComponent<Tag, our_rank, ComponentType>>().operator()(
      std::declval<stk::mesh::Entity>()));
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
template <typename ScalarType, size_t N>
decltype(auto) get_updated_ngp_component(const VectorFieldComponent<ScalarType, N>& component) {
  auto& ngp_field = stk::mesh::get_updated_ngp_field<ScalarType>(component.field());
  using ngp_field_type = std::remove_reference_t<decltype(ngp_field)>;
  return NgpVectorFieldComponent<ngp_field_type, N>(ngp_field);
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
