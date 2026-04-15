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

#ifndef MUNDY_MESH_DECLARECOMPONENT_HPP_
#define MUNDY_MESH_DECLARECOMPONENT_HPP_

/// \file DeclareComponent.hpp
/// \brief Helpers for fluently declaring field-backed and shared components.

// C++ core
#include <string>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_mesh/Component.hpp>
#include <mundy_mesh/DeclareField.hpp>
#include <mundy_mesh/FieldComponent.hpp>
#include <mundy_mesh/SharedComponent.hpp>

namespace mundy {

namespace mesh {

template <typename CanonicalAccess>
struct component_access_traits;

template <typename ValueType>
struct component_access_traits<access::raw<ValueType>> {
  using canonical_access = access::raw<ValueType>;
  using value_type = ValueType;
  using field_scalar_type = ValueType;
  using shared_value_type = ValueType;

  template <typename FieldScalarType>
  using field_component_t = FieldComponent<FieldScalarType>;

  template <typename NgpFieldType>
  using ngp_field_component_t = NgpFieldComponent<NgpFieldType>;

  using shared_component_t = SharedComponent<shared_value_type>;

  template <typename NgpMemSpace>
  using ngp_shared_component_t = NgpRawSharedComponent<shared_value_type, NgpMemSpace>;

  static constexpr bool has_default_io_output_type = false;
  static constexpr stk::io::FieldOutputType default_io_output_type =
      stk::io::FieldOutputType::CUSTOM;  // TODO(palmerb4): Is the CUSTOM default the same as STK's default style or is
                                         // it something different?
  static constexpr bool has_known_num_field_scalars = false;
  static constexpr unsigned num_field_scalars = 0;
};

template <typename ScalarType>
struct component_access_traits<access::scalar<ScalarType>> {
  using canonical_access = access::scalar<ScalarType>;
  using scalar_type = ScalarType;
  using field_scalar_type = ScalarType;
  using shared_value_type = ScalarType;

  template <typename FieldScalarType>
  using field_component_t = ScalarFieldComponent<FieldScalarType>;

  template <typename NgpFieldType>
  using ngp_field_component_t = NgpScalarFieldComponent<NgpFieldType>;

  using shared_component_t = SharedScalarComponent<shared_value_type>;

  template <typename NgpMemSpace>
  using ngp_shared_component_t = NgpSharedScalarComponent<shared_value_type, NgpMemSpace>;

  static constexpr bool has_default_io_output_type = true;
  static constexpr stk::io::FieldOutputType default_io_output_type = stk::io::FieldOutputType::SCALAR;
  static constexpr bool has_known_num_field_scalars = true;
  static constexpr unsigned num_field_scalars = 1;
};

template <typename ScalarType, size_t N>
struct component_access_traits<access::vector<ScalarType, N>> {
  using canonical_access = access::vector<ScalarType, N>;
  using scalar_type = ScalarType;
  using field_scalar_type = ScalarType;
  using shared_value_type = Vector<ScalarType, N>;

  template <typename FieldScalarType>
  using field_component_t = VectorFieldComponent<FieldScalarType, N>;

  template <typename NgpFieldType>
  using ngp_field_component_t = NgpVectorFieldComponent<NgpFieldType, N>;

  using shared_component_t = SharedVectorComponent<scalar_type, N>;

  template <typename NgpMemSpace>
  using ngp_shared_component_t = NgpSharedVectorComponent<scalar_type, N, NgpMemSpace>;

  static constexpr bool has_default_io_output_type = (N == 2 || N == 3);
  static constexpr stk::io::FieldOutputType default_io_output_type =
      N == 2 ? stk::io::FieldOutputType::VECTOR_2D
             : (N == 3 ? stk::io::FieldOutputType::VECTOR_3D : stk::io::FieldOutputType::CUSTOM);
  static constexpr bool has_known_num_field_scalars = true;
  static constexpr unsigned num_field_scalars = N;
};

template <typename ScalarType>
struct component_access_traits<access::matrix3<ScalarType>> {
  using canonical_access = access::matrix3<ScalarType>;
  using scalar_type = ScalarType;
  using field_scalar_type = ScalarType;
  using shared_value_type = Matrix3<ScalarType>;

  template <typename FieldScalarType>
  using field_component_t = Matrix3FieldComponent<FieldScalarType>;

  template <typename NgpFieldType>
  using ngp_field_component_t = NgpMatrix3FieldComponent<NgpFieldType>;

  using shared_component_t = SharedMatrix3Component<scalar_type>;

  template <typename NgpMemSpace>
  using ngp_shared_component_t = NgpSharedMatrix3Component<scalar_type, NgpMemSpace>;

  static constexpr bool has_default_io_output_type = true;
  static constexpr stk::io::FieldOutputType default_io_output_type = stk::io::FieldOutputType::MATRIX_33;
  static constexpr bool has_known_num_field_scalars = true;
  static constexpr unsigned num_field_scalars = 9;
};

template <typename ScalarType>
struct component_access_traits<access::quaternion<ScalarType>> {
  using canonical_access = access::quaternion<ScalarType>;
  using scalar_type = ScalarType;
  using field_scalar_type = ScalarType;
  using shared_value_type = Quaternion<ScalarType>;

  template <typename FieldScalarType>
  using field_component_t = QuaternionFieldComponent<FieldScalarType>;

  template <typename NgpFieldType>
  using ngp_field_component_t = NgpQuaternionFieldComponent<NgpFieldType>;

  using shared_component_t = SharedQuaternionComponent<scalar_type>;

  template <typename NgpMemSpace>
  using ngp_shared_component_t = NgpSharedQuaternionComponent<scalar_type, NgpMemSpace>;

  static constexpr bool has_default_io_output_type = true;
  // TODO(palmerb4): STK uses the opposite quaternion storage as us (its the same storage layout as Eigen's quaternions)
  //                 We should update our quaternion order for consistency with dependent libraries and standard
  //                 conventions.
  static constexpr stk::io::FieldOutputType default_io_output_type = stk::io::FieldOutputType::QUATERNION_3D;
  static constexpr bool has_known_num_field_scalars = true;
  static constexpr unsigned num_field_scalars = 4;
};

template <typename ScalarType>
struct component_access_traits<access::aabb<ScalarType>> {
  using canonical_access = access::aabb<ScalarType>;
  using scalar_type = ScalarType;
  using field_scalar_type = ScalarType;
  using shared_value_type = AABB<ScalarType>;

  template <typename FieldScalarType>
  using field_component_t = AABBFieldComponent<FieldScalarType>;

  template <typename NgpFieldType>
  using ngp_field_component_t = NgpAABBFieldComponent<NgpFieldType>;

  using shared_component_t = SharedAABBComponent<scalar_type>;

  template <typename NgpMemSpace>
  using ngp_shared_component_t = NgpSharedAABBComponent<scalar_type, NgpMemSpace>;

  static constexpr bool has_default_io_output_type = false;
  static constexpr stk::io::FieldOutputType default_io_output_type = stk::io::FieldOutputType::CUSTOM;
  static constexpr bool has_known_num_field_scalars = true;
  static constexpr unsigned num_field_scalars = 6;
};

namespace impl {

template <typename ValueType>
std::string component_access_name(access::raw<ValueType>) {
  return "raw";
}

template <typename ScalarType>
std::string component_access_name(access::scalar<ScalarType>) {
  return "scalar";
}

template <typename ScalarType, size_t N>
std::string component_access_name(access::vector<ScalarType, N>) {
  return (sink() << "vector" << N).to_string();
}

template <typename ScalarType>
std::string component_access_name(access::matrix3<ScalarType>) {
  return "matrix3";
}

template <typename ScalarType>
std::string component_access_name(access::quaternion<ScalarType>) {
  return "quaternion";
}

template <typename ScalarType>
std::string component_access_name(access::aabb<ScalarType>) {
  return "aabb";
}

template <typename CanonicalAccess>
std::string component_access_name() {
  return component_access_name(CanonicalAccess{});
}

template <typename CanonicalAccess>
void apply_default_output_type_if_needed(FieldDeclarationSnapshot& snapshot) {
  using access_traits = component_access_traits<CanonicalAccess>;

  if constexpr (access_traits::has_default_io_output_type) {
    if (!snapshot.has_output_type) {
      snapshot.has_output_type = true;
      snapshot.output_type = access_traits::default_io_output_type;
    } else {
      MUNDY_THROW_REQUIRE(snapshot.output_type == access_traits::default_io_output_type, std::invalid_argument,
                          sink() << "Field declaration for '" << snapshot.field_name << "' uses output type "
                                 << static_cast<int>(snapshot.output_type) << " but access '"
                                 << component_access_name<CanonicalAccess>() << "' expects "
                                 << static_cast<int>(access_traits::default_io_output_type));
    }
  }
}

template <typename SharedSource, typename Enable = void>
struct shared_component_source_value {
  using type = std::remove_cvref_t<SharedSource>;
};

template <typename SharedSource>
struct shared_component_source_value<SharedSource,
                                     std::enable_if_t<Kokkos::is_view_v<std::remove_cvref_t<SharedSource>>>> {
  using type = std::remove_cv_t<typename std::remove_cvref_t<SharedSource>::value_type>;
};

template <typename SharedSource>
using shared_component_source_value_t = typename shared_component_source_value<SharedSource>::type;

}  // namespace impl

template <typename FieldScalarType, typename Tag>
class TaggedFieldDeclarationHelperT {
 public:
  using our_t = TaggedFieldDeclarationHelperT<FieldScalarType, Tag>;
  using field_scalar_type = std::remove_cvref_t<FieldScalarType>;

  TaggedFieldDeclarationHelperT(const TaggedFieldDeclarationHelperT&) = default;
  TaggedFieldDeclarationHelperT(TaggedFieldDeclarationHelperT&&) = default;
  TaggedFieldDeclarationHelperT& operator=(const TaggedFieldDeclarationHelperT&) = default;
  TaggedFieldDeclarationHelperT& operator=(TaggedFieldDeclarationHelperT&&) = default;

  our_t rank(stk::mesh::EntityRank rank) {
    snapshot_.has_rank = true;
    snapshot_.rank = rank;
    return *this;
  }

  our_t name(const std::string& field_name) {
    snapshot_.has_name = true;
    snapshot_.field_name = field_name;
    return *this;
  }

  our_t role(Ioss::Field::RoleType field_role) {
    snapshot_.has_role = true;
    snapshot_.field_role = field_role;
    return *this;
  }

  our_t output_type(stk::io::FieldOutputType output_type) {
    snapshot_.has_output_type = true;
    snapshot_.output_type = output_type;
    return *this;
  }

  template <typename T>
  auto type() const {
    using new_field_scalar_type = std::remove_cvref_t<T>;
    return TaggedFieldDeclarationHelperT<new_field_scalar_type, Tag>(snapshot_);
  }

  template <typename AccessLike>
  auto access() const {
    return TaggedFieldComponentDeclarationHelperT<FieldScalarType, AccessLike, Tag>(snapshot_);
  }

  template <typename NewTag>
  auto tag() const {
    auto next = snapshot_;
    return TaggedFieldDeclarationHelperT<FieldScalarType, NewTag>(next);
  }

  void declare() const {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "Component access must be set before declaring a tagged field component.");
  }

 private:
  explicit TaggedFieldDeclarationHelperT(impl::FieldDeclarationSnapshot snapshot) : snapshot_(std::move(snapshot)) {
  }

  impl::FieldDeclarationSnapshot snapshot_;

  friend class FieldDeclarationHelper;
  template <typename T>
  friend class FieldDeclarationHelperT;
  template <typename OtherFieldScalarType, typename OtherTag>
  friend class TaggedFieldDeclarationHelperT;
};  // TaggedFieldDeclarationHelperT

template <typename FieldScalarType, typename AccessLike, typename Tag>
class TaggedFieldComponentDeclarationHelperT {
 public:
  using our_t = TaggedFieldComponentDeclarationHelperT<FieldScalarType, AccessLike, Tag>;
  using access_like = AccessLike;
  using canonical_access = canonical_component_access_t<AccessLike>;
  using access_traits = component_access_traits<canonical_access>;
  using field_scalar_type =
      std::conditional_t<std::is_void_v<FieldScalarType>, typename access_traits::field_scalar_type,
                         std::remove_cvref_t<FieldScalarType>>;

  static_assert(std::is_void_v<FieldScalarType> ||
                    std::is_same_v<std::remove_cvref_t<FieldScalarType>, typename access_traits::field_scalar_type>,
                "The chosen field scalar type is incompatible with the chosen component access.");

  TaggedFieldComponentDeclarationHelperT(const TaggedFieldComponentDeclarationHelperT&) = default;
  TaggedFieldComponentDeclarationHelperT(TaggedFieldComponentDeclarationHelperT&&) = default;
  TaggedFieldComponentDeclarationHelperT& operator=(const TaggedFieldComponentDeclarationHelperT&) = default;
  TaggedFieldComponentDeclarationHelperT& operator=(TaggedFieldComponentDeclarationHelperT&&) = default;

  our_t rank(stk::mesh::EntityRank rank) {
    snapshot_.has_rank = true;
    snapshot_.rank = rank;
    return *this;
  }

  our_t name(const std::string& field_name) {
    snapshot_.has_name = true;
    snapshot_.field_name = field_name;
    return *this;
  }

  our_t role(Ioss::Field::RoleType field_role) {
    snapshot_.has_role = true;
    snapshot_.field_role = field_role;
    return *this;
  }

  our_t output_type(stk::io::FieldOutputType output_type) {
    snapshot_.has_output_type = true;
    snapshot_.output_type = output_type;
    return *this;
  }

  template <typename T>
  auto type() const {
    using new_field_scalar_type = std::remove_cvref_t<T>;
    static_assert(std::is_same_v<new_field_scalar_type, typename access_traits::field_scalar_type>,
                  "The chosen field scalar type is incompatible with the chosen component access.");
    return TaggedFieldComponentDeclarationHelperT<new_field_scalar_type, AccessLike, Tag>(snapshot_);
  }

  template <typename NewAccessLike>
  auto access() const {
    return TaggedFieldComponentDeclarationHelperT<FieldScalarType, NewAccessLike, Tag>(snapshot_);
  }

  template <typename NewTag>
  auto tag() const {
    auto next = snapshot_;
    return TaggedFieldComponentDeclarationHelperT<FieldScalarType, AccessLike, NewTag>(next);
  }

  auto declare() const {
    auto snapshot = snapshot_;
    impl::apply_default_output_type_if_needed<canonical_access>(snapshot);

    stk::mesh::Field<field_scalar_type>& field = impl::declare_field_from_snapshot<field_scalar_type>(snapshot);
    using component_type = typename access_traits::template field_component_t<field_scalar_type>;
    component_type component(field);

    if constexpr (std::is_void_v<Tag>) {
      return component;
    } else {
      return make_tagged_component<Tag>(component);
    }
  }

 private:
  explicit TaggedFieldComponentDeclarationHelperT(impl::FieldDeclarationSnapshot snapshot)
      : snapshot_(std::move(snapshot)) {
  }

  impl::FieldDeclarationSnapshot snapshot_;

  friend class FieldDeclarationHelper;
  template <typename T>
  friend class FieldDeclarationHelperT;
  template <typename OtherFieldScalarType, typename OtherTag>
  friend class TaggedFieldDeclarationHelperT;
  template <typename OtherFieldScalarType, typename OtherAccessLike, typename OtherTag>
  friend class TaggedFieldComponentDeclarationHelperT;
};  // TaggedFieldComponentDeclarationHelperT

// TODO(palmerb4): The following needs FAR better documentation. It was not clear, for example, what SharedSource even
// was or why it exists It must exist because it may either be a raw value that needs to be copied into host storage or
// a rank-1 Kokkos::View in HostSpace of extent 1 that can be aliased. But this is not at all clear from the current doc
// (which currently doesn't exist)
template <typename SharedSource, typename AccessLike, typename Tag = void>
class TaggedSharedComponentDeclarationHelperT {
 public:
  using our_t = TaggedSharedComponentDeclarationHelperT<SharedSource, AccessLike, Tag>;
  using shared_source_type = SharedSource;
  using access_like = AccessLike;
  using canonical_access = canonical_component_access_t<AccessLike>;
  using access_traits = component_access_traits<canonical_access>;
  using shared_value_type = impl::shared_component_source_value_t<shared_source_type>;

  static_assert(std::is_same_v<shared_value_type, typename access_traits::shared_value_type>,
                "The shared source value is incompatible with the chosen component access.");

  explicit TaggedSharedComponentDeclarationHelperT(shared_source_type shared_source)
      : shared_source_(std::move(shared_source)), has_rank_(false), rank_(stk::topology::INVALID_RANK) {
  }

  TaggedSharedComponentDeclarationHelperT(const TaggedSharedComponentDeclarationHelperT&) = default;
  TaggedSharedComponentDeclarationHelperT(TaggedSharedComponentDeclarationHelperT&&) = default;
  TaggedSharedComponentDeclarationHelperT& operator=(const TaggedSharedComponentDeclarationHelperT&) = default;
  TaggedSharedComponentDeclarationHelperT& operator=(TaggedSharedComponentDeclarationHelperT&&) = default;

  our_t rank(stk::mesh::EntityRank rank) {
    has_rank_ = true;
    rank_ = rank;
    return *this;
  }

  template <typename NewAccessLike>
  auto access() const {
    auto next = TaggedSharedComponentDeclarationHelperT<shared_source_type, NewAccessLike, Tag>(shared_source_);
    next.has_rank_ = has_rank_;
    next.rank_ = rank_;
    return next;
  }

  template <typename NewTag>
  auto tag() const {
    auto next =
        TaggedSharedComponentDeclarationHelperT<shared_source_type, AccessLike, NewTag>(shared_source_);
    next.has_rank_ = has_rank_;
    next.rank_ = rank_;
    return next;
  }

  auto declare() const {
    MUNDY_THROW_REQUIRE(has_rank_, std::logic_error, "Component rank must be set before declaring a shared component.");

    using component_type = typename access_traits::shared_component_t;
    component_type component(shared_source_);

    if constexpr (std::is_void_v<Tag>) {
      return component;
    } else {
      return make_tagged_component<Tag>(component);
    }
  }

 private:
  shared_source_type shared_source_;
  bool has_rank_;
  stk::mesh::EntityRank rank_;

  template <typename OtherSharedSource, typename OtherAccessLike, typename OtherTag>
  friend class TaggedSharedComponentDeclarationHelperT;
};  // TaggedSharedComponentDeclarationHelperT

class ComponentDeclarationHelper {
 public:
  template <typename SharedSource>
  auto shared(SharedSource&& shared_source) const {
    using shared_source_type = std::decay_t<SharedSource>;
    using shared_value_type = impl::shared_component_source_value_t<shared_source_type>;
    using canonical_access = canonical_component_access_t<shared_value_type>;
    return TaggedSharedComponentDeclarationHelperT<shared_source_type, canonical_access>(
        std::forward<SharedSource>(shared_source));
  }
};  // ComponentDeclarationHelper

template <typename T>
template <typename AccessLike>
TaggedFieldComponentDeclarationHelperT<T, AccessLike> FieldDeclarationHelperT<T>::access() const {
  return TaggedFieldComponentDeclarationHelperT<T, AccessLike>(impl::make_field_declaration_snapshot(*this));
}

template <typename T>
template <typename Tag>
TaggedFieldDeclarationHelperT<T, Tag> FieldDeclarationHelperT<T>::tag() const {
  return TaggedFieldDeclarationHelperT<T, Tag>(impl::make_field_declaration_snapshot(*this));
}

template <typename AccessLike>
TaggedFieldComponentDeclarationHelperT<FieldDeclarationHelper::invalid_field_scalar_type, AccessLike>
FieldDeclarationHelper::access() const {
  return TaggedFieldComponentDeclarationHelperT<invalid_field_scalar_type, AccessLike>(
      impl::make_field_declaration_snapshot(*this));
}

template <typename Tag>
TaggedFieldDeclarationHelperT<FieldDeclarationHelper::invalid_field_scalar_type, Tag> FieldDeclarationHelper::tag()
    const {
  return TaggedFieldDeclarationHelperT<invalid_field_scalar_type, Tag>(impl::make_field_declaration_snapshot(*this));
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_DECLARECOMPONENT_HPP_
