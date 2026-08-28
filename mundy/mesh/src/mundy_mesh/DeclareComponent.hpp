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
/// \brief Unified fluent builder for declaring field-backed and shared components.
///
/// The preferred entry point for component declarations is \c ComponentDeclaration.
/// Use \c FieldDeclaration (DeclareField.hpp) for raw STK field declarations only.

// C++ core
#include <type_traits>
#include <utility>

// STK
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData

// Mundy
#include <Mundy_config.hpp>                          // for MUNDY_DEPRECATED_MSG
#include <mundy_mesh/impl/DeclareComponentImpl.hpp>  // for impl::FieldDeclarationSnapshot, tagged builder types

namespace mundy {

namespace mesh {

/// \class ComponentDeclaration
/// \brief Unified fluent builder for declaring both field-backed and shared-backed components.
///
/// \c ComponentDeclaration is the preferred entry point for all component declarations.
/// It accumulates declaration metadata through a fluent chain of setters and then materializes
/// either a field-backed component (via \c .field<A>().declare()) or a shared-backed component
/// (via \c .shared<A>(source).declare()).
///
/// The builder transitions through implementation-detail intermediate types as the fluent chain
/// accumulates type information; use \c auto to hold these intermediate results.
///
/// The backend is selected explicitly by calling \c .field<A>() for a field-backed component or
/// \c .shared<A>(source) for a shared-backed component, then \c .declare().
///
/// \par Field-backed example:
/// \code{.cpp}
///   ComponentDeclaration decl(meta_data);
///   auto velocity = decl.rank(NODE_RANK)
///                       .name("velocity")
///                       .role(Ioss::Field::TRANSIENT)
///                       .field<mundy::math::Vector3<double>>()
///                       .declare();
/// \endcode
///
/// \par Shared-backed example:
/// \code{.cpp}
///   ComponentDeclaration decl;
///   auto stiffness = decl.rank(ELEMENT_RANK)
///                        .name("stiffness")
///                        .shared<double>(1.0)
///                        .declare();
/// \endcode
///
/// \par Tagged field component:
/// \code{.cpp}
///   ComponentDeclaration decl(meta_data);
///   auto tagged = decl.rank(NODE_RANK)
///                     .name("velocity")
///                     .tag<VelocityTag>()
///                     .field<mundy::math::Vector3<double>>()
///                     .declare();
/// \endcode
///
/// \par Snapshot reuse:
/// Intermediate builder values are copyable and may be reused to declare multiple components
/// that share common properties:
/// \code{.cpp}
///   ComponentDeclaration decl(meta_data);
///   auto node_vec3 = decl.rank(NODE_RANK)
///                        .role(Ioss::Field::TRANSIENT)
///                        .field<mundy::math::Vector3<double>>();
///   auto velocity = node_vec3.name("velocity").declare();
///   auto force    = node_vec3.name("force").declare();
/// \endcode
class ComponentDeclaration {
 public:
  //! \name Constructors and Assignment Operators
  //@{

  /// \brief Construct with a MetaData reference (required for field-backed declarations; optional for shared).
  explicit ComponentDeclaration(stk::mesh::MetaData& meta_data) : meta_data_(&meta_data) {
  }

  /// \brief Default constructor for shared-backed declarations that do not require MetaData.
  ComponentDeclaration() : meta_data_(nullptr) {
  }

  ComponentDeclaration(const ComponentDeclaration&) = default;
  ComponentDeclaration(ComponentDeclaration&&) = default;
  ComponentDeclaration& operator=(const ComponentDeclaration&) = default;
  ComponentDeclaration& operator=(ComponentDeclaration&&) = default;

  //@}

  //! \name Fluent setters
  //@{

  /// \brief Set the entity rank of the component.
  ComponentDeclaration rank(stk::mesh::EntityRank rank) const {
    ComponentDeclaration copy = *this;
    copy.snapshot_.has_rank = true;
    copy.snapshot_.rank = rank;
    return copy;
  }

  /// \brief Set the name of the component.
  ComponentDeclaration name(const std::string& component_name) const {
    ComponentDeclaration copy = *this;
    copy.snapshot_.has_name = true;
    copy.snapshot_.field_name = component_name;
    return copy;
  }

  /// \brief Set the I/O role for field-backed components.
  ///
  /// The typical Mundy application will label fields as \c TRANSIENT or \c MESH.
  ComponentDeclaration role(Ioss::Field::RoleType field_role) const {
    ComponentDeclaration copy = *this;
    copy.snapshot_.has_role = true;
    copy.snapshot_.field_role = field_role;
    return copy;
  }

  /// \brief Set the STK output type for field-backed components.
  ComponentDeclaration output_type(stk::io::FieldOutputType output_type) const {
    ComponentDeclaration copy = *this;
    copy.snapshot_.has_output_type = true;
    copy.snapshot_.output_type = output_type;
    return copy;
  }

  //@}

  //! \name Terminal transitions
  //@{

  /// \brief Fix the field scalar type before choosing a component backend.
  template <typename T>
  auto type() const {
    impl::FieldDeclarationSnapshot snap = snapshot_;
    snap.meta_data = meta_data_;
    return TaggedFieldDeclarationT<std::remove_cvref_t<T>, void>(snap);
  }

  /// \brief Commit to a field-backed component with the given access shape.
  ///
  /// \tparam AccessLike  Access shape: arithmetic type, Mundy math type, or explicit \c access:: tag.
  template <typename AccessLike>
  auto field() const {
    impl::FieldDeclarationSnapshot snap = snapshot_;
    snap.meta_data = meta_data_;
    return TaggedFieldBackedDeclarationHelperT<void, AccessLike, void>(snap);
  }

  /// \brief Commit to a shared-backed component with the given access shape and source.
  template <typename AccessLike, typename SharedSource>
  auto shared(SharedSource&& source) const {
    using canonical_access = canonical_component_access_t<AccessLike>;
    using shape = component_access_shape<canonical_access>;
    using source_type = std::decay_t<SharedSource>;
    using shared_value_t = impl::shared_component_source_value_t<source_type>;
    static_assert(std::is_same_v<shared_value_t, typename shape::shared_value_type>,
                  "Shared source value type is incompatible with the chosen component access.");

    impl::FieldDeclarationSnapshot snap = snapshot_;
    snap.meta_data = meta_data_;
    return TaggedSharedComponentDeclarationT<source_type, AccessLike, void>(std::forward<SharedSource>(source), snap);
  }

  /// \brief Attach a semantic tag before choosing a component backend.
  ///
  /// \tparam Tag  Tag type to attach to the resulting component.
  template <typename Tag>
  auto tag() const {
    impl::FieldDeclarationSnapshot snap = snapshot_;
    snap.meta_data = meta_data_;
    return TaggedFieldDeclarationT<void, Tag>(snap);
  }

  //@}

 private:
  stk::mesh::MetaData* meta_data_ = nullptr;
  impl::FieldDeclarationSnapshot snapshot_;
};  // ComponentDeclaration

using ComponentDeclarationHelper MUNDY_DEPRECATED_MSG("use ComponentDeclaration") = ComponentDeclaration;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_DECLARECOMPONENT_HPP_
