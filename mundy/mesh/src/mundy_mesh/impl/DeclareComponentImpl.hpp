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

#ifndef MUNDY_MESH_IMPL_DECLARECOMPONENTIMPL_HPP_
#define MUNDY_MESH_IMPL_DECLARECOMPONENTIMPL_HPP_

/// \file DeclareComponentImpl.hpp
/// \brief Implementation helpers and intermediate builder types for component declaration.
///
/// This header provides:
///   - Storage type mappings (\c field_component_for, \c shared_component_for) from canonical
///     access tags to concrete component types.
///   - The intermediate fluent builder types (\c TaggedFieldDeclarationT,
///     \c TaggedFieldBackedDeclarationHelperT, \c TaggedSharedComponentDeclarationT)
///     that arise as a result of the expanding-type fluent API.  These types are implementation
///     details; callers should use \c auto to hold them.
///
/// Dependency order:
///   DeclareFieldImpl.hpp -> DeclareField.hpp -> DeclareComponentImpl.hpp -> DeclareComponent.hpp

// C++ core
#include <string>
#include <type_traits>
#include <utility>

// STK
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::FieldOutputType

// Mundy
#include <mundy_mesh/Component.hpp>        // for make_tagged_component
#include <mundy_mesh/ComponentAccess.hpp>  // for component_access_shape, canonical_component_access_t
#include <mundy_mesh/DeclareField.hpp>     // for FieldDeclaration, FieldDeclarationT, impl::FieldDeclarationSnapshot
#include <mundy_mesh/FieldComponent.hpp>   // for FieldComponent, ScalarFieldComponent, VectorFieldComponent, ...
#include <mundy_mesh/SharedComponent.hpp>  // for SharedComponent, SharedScalarComponent, ...
#include <mundy_utils/throw_assert.hpp>    // for MUNDY_THROW_REQUIRE, sink

namespace mundy {

namespace mesh {

namespace impl {

// ======================================================================================================================
// field_component_for — maps a canonical access tag to the concrete field-backed component type
// ======================================================================================================================

template <typename CanonicalAccess>
struct field_component_for;

template <typename ValueType>
struct field_component_for<access::raw<ValueType>> {
  template <typename FieldScalarType>
  using type = mesh::FieldComponent<FieldScalarType>;
};

template <typename ScalarType>
struct field_component_for<access::scalar<ScalarType>> {
  template <typename FieldScalarType>
  using type = ScalarFieldComponent<FieldScalarType>;
};

template <typename ScalarType, size_t N>
struct field_component_for<access::vector<ScalarType, N>> {
  template <typename FieldScalarType>
  using type = VectorFieldComponent<FieldScalarType, N>;
};

template <typename ScalarType, size_t N, size_t M>
struct field_component_for<access::matrix<ScalarType, N, M>> {
  template <typename FieldScalarType>
  using type = MatrixFieldComponent<FieldScalarType, N, M>;
};

template <typename ScalarType>
struct field_component_for<access::quaternion<ScalarType>> {
  template <typename FieldScalarType>
  using type = QuaternionFieldComponent<FieldScalarType>;
};

template <typename ScalarType>
struct field_component_for<access::aabb<ScalarType>> {
  template <typename FieldScalarType>
  using type = AABBFieldComponent<FieldScalarType>;
};

template <typename ScalarType>
struct field_component_for<access::obb<ScalarType>> {
  template <typename FieldScalarType>
  using type = OBBFieldComponent<FieldScalarType>;
};

// ======================================================================================================================
// shared_component_for — maps a canonical access tag to the concrete shared-backed component type
// ======================================================================================================================

template <typename CanonicalAccess>
struct shared_component_for;

template <typename ValueType>
struct shared_component_for<access::raw<ValueType>> {
  using type = SharedComponent<ValueType>;
};

template <typename ScalarType>
struct shared_component_for<access::scalar<ScalarType>> {
  using type = SharedScalarComponent<ScalarType>;
};

template <typename ScalarType, size_t N>
struct shared_component_for<access::vector<ScalarType, N>> {
  using type = SharedVectorComponent<ScalarType, N>;
};

template <typename ScalarType, size_t N, size_t M>
struct shared_component_for<access::matrix<ScalarType, N, M>> {
  using type = SharedMatrixComponent<ScalarType, N, M>;
};

template <typename ScalarType>
struct shared_component_for<access::quaternion<ScalarType>> {
  using type = SharedQuaternionComponent<ScalarType>;
};

template <typename ScalarType>
struct shared_component_for<access::aabb<ScalarType>> {
  using type = SharedAABBComponent<ScalarType>;
};

// ======================================================================================================================
// io_output_type_scalar_count — returns the per-entity scalar count implied by a FieldOutputType.
// Returns 0 for CUSTOM or any type whose scalar count is not statically known (always acceptable).
// ======================================================================================================================

inline unsigned io_output_type_scalar_count(stk::io::FieldOutputType type) {
  switch (type) {
    case stk::io::FieldOutputType::SCALAR:
      return 1;
    case stk::io::FieldOutputType::VECTOR_2D:
      return 2;
    case stk::io::FieldOutputType::VECTOR_3D:
      return 3;
    case stk::io::FieldOutputType::FULL_TENSOR_36:
      return 9;
    case stk::io::FieldOutputType::FULL_TENSOR_32:
      return 5;
    case stk::io::FieldOutputType::FULL_TENSOR_22:
      return 4;
    case stk::io::FieldOutputType::FULL_TENSOR_16:
      return 7;
    case stk::io::FieldOutputType::FULL_TENSOR_12:
      return 3;
    case stk::io::FieldOutputType::SYM_TENSOR_33:
      return 6;
    case stk::io::FieldOutputType::SYM_TENSOR_31:
      return 4;
    case stk::io::FieldOutputType::SYM_TENSOR_21:
      return 3;
    case stk::io::FieldOutputType::SYM_TENSOR_13:
      return 4;
    case stk::io::FieldOutputType::SYM_TENSOR_11:
      return 2;
    case stk::io::FieldOutputType::SYM_TENSOR_10:
      return 1;
    case stk::io::FieldOutputType::ASYM_TENSOR_03:
      return 3;
    case stk::io::FieldOutputType::ASYM_TENSOR_02:
      return 2;
    case stk::io::FieldOutputType::ASYM_TENSOR_01:
      return 1;
    case stk::io::FieldOutputType::MATRIX_22:
      return 4;
    case stk::io::FieldOutputType::MATRIX_33:
      return 9;
    case stk::io::FieldOutputType::QUATERNION_2D:
      return 2;
    case stk::io::FieldOutputType::QUATERNION_3D:
      return 4;
    case stk::io::FieldOutputType::CUSTOM:
    default:
      return 0;
  }
}

// ======================================================================================================================
// component_access_name — returns a human-readable name for a canonical access type (for diagnostics)
// ======================================================================================================================

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

template <typename ScalarType, size_t N, size_t M>
std::string component_access_name(access::matrix<ScalarType, N, M>) {
  return (sink() << "matrix" << N << "x" << M).to_string();
}

template <typename ScalarType>
std::string component_access_name(access::quaternion<ScalarType>) {
  return "quaternion";
}

template <typename ScalarType>
std::string component_access_name(access::aabb<ScalarType>) {
  return "aabb";
}

template <typename ScalarType>
std::string component_access_name(access::obb<ScalarType>) {
  return "obb";
}

template <typename CanonicalAccess>
std::string component_access_name() {
  return component_access_name(CanonicalAccess{});
}

// ======================================================================================================================
// component_default_output_type — maps a canonical access tag to its STK IO output type (if one exists).
//
// Provides:
//   has_default — true if a canonical stk::io::FieldOutputType exists for this access shape
//   value       — the canonical output type (only valid when has_default is true)
// ======================================================================================================================

template <typename CanonicalAccess>
struct component_default_output_type {
  static constexpr bool has_default = false;
};

template <typename ScalarType>
struct component_default_output_type<access::scalar<ScalarType>> {
  static constexpr bool has_default = true;
  static constexpr stk::io::FieldOutputType value = stk::io::FieldOutputType::SCALAR;
};

template <typename ScalarType, size_t N>
struct component_default_output_type<access::vector<ScalarType, N>> {
  static constexpr bool has_default = (N == 2 || N == 3);
  static constexpr stk::io::FieldOutputType value =
      N == 2 ? stk::io::FieldOutputType::VECTOR_2D : stk::io::FieldOutputType::VECTOR_3D;
};

template <typename ScalarType>
struct component_default_output_type<access::matrix<ScalarType, 3, 3>> {
  static constexpr bool has_default = true;
  static constexpr stk::io::FieldOutputType value = stk::io::FieldOutputType::MATRIX_33;
};

template <typename ScalarType>
struct component_default_output_type<access::quaternion<ScalarType>> {
  static constexpr bool has_default = true;
  static constexpr stk::io::FieldOutputType value = stk::io::FieldOutputType::QUATERNION_3D;
};

// ======================================================================================================================
// apply_default_output_type_if_needed — applies shape-default output type or validates user-supplied type.
//
// Rules:
//   - If no output type was set and the access shape has a default, apply it.
//   - If an output type was set and the access shape has a known scalar count, validate the scalar count matches.
//     io_output_type_scalar_count returns 0 for output types whose scalar count is not statically known
//     (e.g., CUSTOM), which bypasses the count check and allows them through.
// ======================================================================================================================

template <typename CanonicalAccess>
void apply_default_output_type_if_needed(FieldDeclarationSnapshot& snapshot) {
  using shape = component_access_shape<CanonicalAccess>;
  using output_type = component_default_output_type<CanonicalAccess>;

  if (!snapshot.has_output_type) {
    if constexpr (output_type::has_default) {
      snapshot.has_output_type = true;
      snapshot.output_type = output_type::value;
    }
  } else if constexpr (shape::has_fixed_field_scalars) {
    const unsigned user_count = io_output_type_scalar_count(snapshot.output_type);
    if (user_count != 0) {
      MUNDY_THROW_REQUIRE(user_count == shape::field_scalars, std::invalid_argument,
                          sink() << "Component declaration for '" << snapshot.field_name
                                 << "' specifies an output type with " << user_count << " scalars, but access '"
                                 << component_access_name<CanonicalAccess>() << "' requires " << shape::field_scalars
                                 << " scalars. "
                                 << "Use CUSTOM output type for non-standard subscripting.");
    }
  }
}

// ======================================================================================================================
// shared_component_source_value — deduces the stored value type from a shared source.
//   - For a plain value: the value type itself.
//   - For a rank-1 HostSpace Kokkos::View: the view's value_type.
// ======================================================================================================================

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

// ======================================================================================================================
// Forward declarations (class definitions below)
// ======================================================================================================================

template <typename FieldScalarType, typename AccessLike, typename Tag>
class TaggedFieldBackedDeclarationHelperT;

template <typename SharedSource, typename AccessLike, typename Tag>
class TaggedSharedComponentDeclarationT;

class ComponentDeclaration;

// ======================================================================================================================
// TaggedFieldDeclarationT
// ======================================================================================================================

/// \class TaggedFieldDeclarationT
/// \ingroup MundyMeshDeclareComponent
/// \brief Intermediate fluent builder: field scalar type and semantic tag are known; access shape is not yet chosen.
///
/// \tparam FieldScalarType  Scalar type for the underlying STK field, or \c void if not yet fixed.
/// \tparam Tag              Semantic tag to attach to the resulting component, or \c void for untagged.
///
/// This type is an implementation detail of the component declaration fluent API.
/// Users should not name it explicitly; use \c auto to capture the result of \c .tag<Tag>().
///
/// \code{.cpp}
///   auto c = ComponentDeclaration(meta)
///              .rank(NODE_RANK).name("velocity")
///              .tag<VelocityTag>()
///              .field<mundy::math::Vector3<double>>()
///              .declare();
/// \endcode
template <typename FieldScalarType, typename Tag>
class TaggedFieldDeclarationT {
 public:
  using our_t = TaggedFieldDeclarationT<FieldScalarType, Tag>;
  using field_value_typeype = std::remove_cvref_t<FieldScalarType>;

  //! \name Constructors and Assignment Operators
  //@{

  TaggedFieldDeclarationT(const TaggedFieldDeclarationT&) = default;
  TaggedFieldDeclarationT(TaggedFieldDeclarationT&&) = default;
  TaggedFieldDeclarationT& operator=(const TaggedFieldDeclarationT&) = default;
  TaggedFieldDeclarationT& operator=(TaggedFieldDeclarationT&&) = default;

  //@}

  //! \name Fluent setters
  //@{

  /// \brief Set the entity rank of the component.
  our_t rank(stk::mesh::EntityRank rank) const {
    our_t copy = *this;
    copy.snapshot_.has_rank = true;
    copy.snapshot_.rank = rank;
    return copy;
  }

  /// \brief Set the name of the component.
  our_t name(const std::string& field_name) const {
    our_t copy = *this;
    copy.snapshot_.has_name = true;
    copy.snapshot_.field_name = field_name;
    return copy;
  }

  /// \brief Set the I/O role for the component.
  our_t role(Ioss::Field::RoleType field_role) const {
    our_t copy = *this;
    copy.snapshot_.has_role = true;
    copy.snapshot_.field_role = field_role;
    return copy;
  }

  /// \brief Set the STK output type for the component.
  our_t output_type(stk::io::FieldOutputType output_type) const {
    our_t copy = *this;
    copy.snapshot_.has_output_type = true;
    copy.snapshot_.output_type = output_type;
    return copy;
  }

  /// \brief Fix the field scalar type, returning a builder with the scalar type locked in.
  template <typename T>
  auto type() const {
    using new_fst = std::remove_cvref_t<T>;
    return TaggedFieldDeclarationT<new_fst, Tag>(snapshot_);
  }

  /// \brief Commit to a field-backed component with the given access shape.
  ///
  /// \tparam AccessLike  Access shape: arithmetic type, Mundy math type, or explicit \c access:: tag.
  template <typename AccessLike>
  auto field() const {
    return TaggedFieldBackedDeclarationHelperT<FieldScalarType, AccessLike, Tag>(snapshot_);
  }

  /// \brief Commit to a shared-backed component with the given access shape and source.
  ///
  /// \param [in] source  Shared value or Kokkos::View to back the component.
  template <typename AccessLike, typename SharedSource>
  auto shared(SharedSource&& source) const {
    using canonical_access = canonical_component_access_t<AccessLike>;
    using shape = component_access_shape<canonical_access>;
    using source_type = std::decay_t<SharedSource>;
    using shared_value_t = impl::shared_component_source_value_t<source_type>;
    static_assert(std::is_same_v<shared_value_t, typename shape::shared_value_type>,
                  "Shared source value type is incompatible with the chosen component access.");
    return TaggedSharedComponentDeclarationT<source_type, AccessLike, Tag>(std::forward<SharedSource>(source),
                                                                           snapshot_);
  }

  /// \brief Replace the current tag, returning a new builder with the new tag.
  template <typename NewTag>
  auto tag() const {
    return TaggedFieldDeclarationT<FieldScalarType, NewTag>(snapshot_);
  }

  //@}

 private:
  explicit TaggedFieldDeclarationT(impl::FieldDeclarationSnapshot snapshot) : snapshot_(std::move(snapshot)) {
  }

  impl::FieldDeclarationSnapshot snapshot_;

  friend class ComponentDeclaration;
  template <typename OtherFST, typename OtherTag>
  friend class TaggedFieldDeclarationT;
};  // TaggedFieldDeclarationT

// ======================================================================================================================
// TaggedFieldBackedDeclarationHelperT
// ======================================================================================================================

/// \class TaggedFieldBackedDeclarationHelperT
/// \ingroup MundyMeshDeclareComponent
/// \brief Terminal fluent builder committed to a field-backed component.
///
/// \tparam FieldScalarType  Scalar type for the underlying STK field, or \c void to infer from access shape.
/// \tparam AccessLike       Access shape (arithmetic type, Mundy math type, or explicit \c access:: tag).
/// \tparam Tag              Semantic tag to attach to the resulting component, or \c void for untagged.
///
/// This type is an implementation detail of the component declaration fluent API.
/// Users should not name it explicitly; use \c auto to capture the result of \c .field().
///
/// Calling \c .declare() materializes a \c FieldComponent backed by a newly declared STK field.
template <typename FieldScalarType, typename AccessLike, typename Tag>
class TaggedFieldBackedDeclarationHelperT {
 public:
  using our_t = TaggedFieldBackedDeclarationHelperT<FieldScalarType, AccessLike, Tag>;
  using access_like = AccessLike;
  using canonical_access = canonical_component_access_t<AccessLike>;
  using shape = component_access_shape<canonical_access>;
  using field_value_typeype = std::conditional_t<std::is_void_v<FieldScalarType>, typename shape::field_value_typeype,
                                                 std::remove_cvref_t<FieldScalarType>>;

  static_assert(std::is_void_v<FieldScalarType> ||
                    std::is_same_v<std::remove_cvref_t<FieldScalarType>, typename shape::field_value_typeype>,
                "The chosen field scalar type is incompatible with the chosen component access.");

  //! \name Constructors and Assignment Operators
  //@{

  TaggedFieldBackedDeclarationHelperT(const TaggedFieldBackedDeclarationHelperT&) = default;
  TaggedFieldBackedDeclarationHelperT(TaggedFieldBackedDeclarationHelperT&&) = default;
  TaggedFieldBackedDeclarationHelperT& operator=(const TaggedFieldBackedDeclarationHelperT&) = default;
  TaggedFieldBackedDeclarationHelperT& operator=(TaggedFieldBackedDeclarationHelperT&&) = default;

  //@}

  //! \name Fluent setters
  //@{

  /// \brief Set the entity rank of the component.
  our_t rank(stk::mesh::EntityRank rank) const {
    our_t copy = *this;
    copy.snapshot_.has_rank = true;
    copy.snapshot_.rank = rank;
    return copy;
  }

  /// \brief Set the name of the component.
  our_t name(const std::string& field_name) const {
    our_t copy = *this;
    copy.snapshot_.has_name = true;
    copy.snapshot_.field_name = field_name;
    return copy;
  }

  /// \brief Set the I/O role for the component.
  our_t role(Ioss::Field::RoleType field_role) const {
    our_t copy = *this;
    copy.snapshot_.has_role = true;
    copy.snapshot_.field_role = field_role;
    return copy;
  }

  /// \brief Set the STK output type for the component.
  our_t output_type(stk::io::FieldOutputType output_type) const {
    our_t copy = *this;
    copy.snapshot_.has_output_type = true;
    copy.snapshot_.output_type = output_type;
    return copy;
  }

  /// \brief Fix the field scalar type, validating compatibility with the chosen access shape.
  template <typename T>
  auto type() const {
    using new_fst = std::remove_cvref_t<T>;
    static_assert(std::is_same_v<new_fst, typename shape::field_value_typeype>,
                  "The chosen field scalar type is incompatible with the chosen component access.");
    return TaggedFieldBackedDeclarationHelperT<new_fst, AccessLike, Tag>(snapshot_);
  }

  /// \brief Replace the semantic tag.
  template <typename NewTag>
  auto tag() const {
    return TaggedFieldBackedDeclarationHelperT<FieldScalarType, AccessLike, NewTag>(snapshot_);
  }

  //@}

  //! \name Terminal
  //@{

  /// \brief Declare a field-backed component and return it.
  auto declare() const {
    MUNDY_THROW_REQUIRE(snapshot_.meta_data != nullptr, std::logic_error,
                        "Field component declaration requires a MetaData reference. "
                        "Use ComponentDeclaration(meta_data) or FieldDeclaration(meta_data).");

    auto snapshot = snapshot_;
    impl::apply_default_output_type_if_needed<canonical_access>(snapshot);

    stk::mesh::Field<field_value_typeype>& field = impl::declare_field_from_snapshot<field_value_typeype>(snapshot);

    using component_type = typename impl::field_component_for<canonical_access>::template type<field_value_typeype>;
    component_type component(field);

    if constexpr (std::is_void_v<Tag>) {
      return component;
    } else {
      return make_tagged_component<Tag>(component);
    }
  }

  //@}

 private:
  explicit TaggedFieldBackedDeclarationHelperT(impl::FieldDeclarationSnapshot snapshot)
      : snapshot_(std::move(snapshot)) {
  }

  impl::FieldDeclarationSnapshot snapshot_;

  friend class ComponentDeclaration;
  template <typename OtherFST, typename OtherTag>
  friend class TaggedFieldDeclarationT;
  template <typename OtherFST, typename OtherAL, typename OtherTag>
  friend class TaggedFieldBackedDeclarationHelperT;
};  // TaggedFieldBackedDeclarationHelperT

// ======================================================================================================================
// TaggedSharedComponentDeclarationT
// ======================================================================================================================

/// \class TaggedSharedComponentDeclarationT
/// \ingroup MundyMeshDeclareComponent
/// \brief Intermediate fluent builder: shared source, access shape, and semantic tag are all known.
///
/// \tparam SharedSource  Type of the shared value source (plain value or Kokkos::View).
/// \tparam AccessLike    Access shape (arithmetic type, Mundy math type, or explicit \c access:: tag).
/// \tparam Tag           Semantic tag to attach to the resulting component, or \c void for untagged.
///
/// This type is an implementation detail of the component declaration fluent API.
/// Users should not name it explicitly; use \c auto to capture the result of \c .shared(source).
///
/// \c rank() must be called before \c declare().
template <typename SharedSource, typename AccessLike, typename Tag = void>
class TaggedSharedComponentDeclarationT {
 public:
  using our_t = TaggedSharedComponentDeclarationT<SharedSource, AccessLike, Tag>;
  using shared_source_type = SharedSource;
  using access_like = AccessLike;
  using canonical_access = canonical_component_access_t<AccessLike>;
  using shape = component_access_shape<canonical_access>;
  using shared_value_type = impl::shared_component_source_value_t<shared_source_type>;

  static_assert(std::is_same_v<shared_value_type, typename shape::shared_value_type>,
                "The shared source value is incompatible with the chosen component access.");

  //! \name Constructors and Assignment Operators
  //@{

  /// \brief Construct from a shared source value; snapshot carries optional metadata from the builder chain.
  explicit TaggedSharedComponentDeclarationT(shared_source_type shared_source,
                                             impl::FieldDeclarationSnapshot snapshot = {})
      : shared_source_(std::move(shared_source)), snapshot_(std::move(snapshot)) {
  }

  TaggedSharedComponentDeclarationT(const TaggedSharedComponentDeclarationT&) = default;
  TaggedSharedComponentDeclarationT(TaggedSharedComponentDeclarationT&&) = default;
  TaggedSharedComponentDeclarationT& operator=(const TaggedSharedComponentDeclarationT&) = default;
  TaggedSharedComponentDeclarationT& operator=(TaggedSharedComponentDeclarationT&&) = default;

  //@}

  //! \name Fluent setters
  //@{

  /// \brief Set the entity rank of the component (required before declare()).
  our_t rank(stk::mesh::EntityRank rank) const {
    our_t copy = *this;
    copy.snapshot_.has_rank = true;
    copy.snapshot_.rank = rank;
    return copy;
  }

  /// \brief Set the name of the component.
  our_t name(const std::string& component_name) const {
    our_t copy = *this;
    copy.snapshot_.has_name = true;
    copy.snapshot_.field_name = component_name;
    return copy;
  }

  /// \brief Replace the semantic tag.
  template <typename NewTag>
  auto tag() const {
    return TaggedSharedComponentDeclarationT<shared_source_type, AccessLike, NewTag>(shared_source_, snapshot_);
  }

  //@}

  //! \name Terminal transitions
  //@{

  /// \brief Declare a shared-backed component and return it.
  ///
  /// \c rank() must have been called before invoking this method.
  auto declare() const {
    MUNDY_THROW_REQUIRE(snapshot_.has_rank, std::logic_error,
                        "Component rank must be set before declaring a shared component.");

    auto snapshot = snapshot_;
    impl::apply_default_output_type_if_needed<canonical_access>(snapshot);

    using component_type = typename impl::shared_component_for<canonical_access>::type;
    component_type component(shared_source_);

    if (snapshot.has_name) {
      component.set_declaration_metadata(snapshot.field_name, snapshot.rank);
    } else {
      component.set_declaration_metadata("", snapshot.rank);
    }

    if constexpr (std::is_void_v<Tag>) {
      return component;
    } else {
      return make_tagged_component<Tag>(component);
    }
  }

  //@}

 private:
  shared_source_type shared_source_;
  impl::FieldDeclarationSnapshot snapshot_;

  template <typename OtherSrc, typename OtherAL, typename OtherTag>
  friend class TaggedSharedComponentDeclarationT;
};  // TaggedSharedComponentDeclarationT

// ======================================================================================================================
// ======================================================================================================================

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_DECLARECOMPONENTIMPL_HPP_
