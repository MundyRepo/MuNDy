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
/// \defgroup MundyMeshDeclareComponent mundy::mesh::ComponentDeclarationHelper
/// \brief Unified fluent builder for declaring field-backed and shared components.
///
/// The preferred entry point for component declarations is \c ComponentDeclarationHelper.
/// Use \c FieldDeclarationHelper (DeclareField.hpp) for raw STK field declarations only.

// STK
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData

// Mundy
#include <mundy_mesh/impl/DeclareComponentImpl.hpp>  // for impl::FieldDeclarationSnapshot, tagged builder types

namespace mundy {

namespace mesh {

/// \class ComponentDeclarationHelper
/// \ingroup MundyMeshDeclareComponent
/// \brief Unified fluent builder for declaring both field-backed and shared-backed components.
///
/// \c ComponentDeclarationHelper is the preferred entry point for all component declarations.
/// It accumulates declaration metadata through a fluent chain of setters and then materializes
/// either a field-backed component (via \c .access<A>().declare()) or a shared-backed component
/// (via \c .shared(source).declare()).
///
/// The builder transitions through implementation-detail intermediate types as the fluent chain
/// accumulates type information; use \c auto to hold these intermediate results.
///
/// The backend is selected explicitly after \c .access<A>(): call \c .field() for a field-backed
/// component or \c .shared(source) for a shared-backed component, then \c .declare().
///
/// \par Field-backed example:
/// \code{.cpp}
///   ComponentDeclarationHelper decl(meta_data);
///   auto velocity = decl.rank(NODE_RANK)
///                       .name("velocity")
///                       .role(Ioss::Field::TRANSIENT)
///                       .access<mundy::math::Vector3<double>>()
///                       .field()
///                       .declare();
/// \endcode
///
/// \par Shared-backed example:
/// \code{.cpp}
///   ComponentDeclarationHelper decl;
///   auto stiffness = decl.rank(ELEMENT_RANK)
///                        .name("stiffness")
///                        .access<double>()
///                        .shared(1.0)
///                        .declare();
/// \endcode
///
/// \par Tagged field component:
/// \code{.cpp}
///   ComponentDeclarationHelper decl(meta_data);
///   auto tagged = decl.rank(NODE_RANK)
///                     .name("velocity")
///                     .tag<VelocityTag>()
///                     .access<mundy::math::Vector3<double>>()
///                     .field()
///                     .declare();
/// \endcode
///
/// \par Snapshot reuse:
/// Intermediate builder values are copyable and may be reused to declare multiple components
/// that share common properties:
/// \code{.cpp}
///   ComponentDeclarationHelper decl(meta_data);
///   auto node_vec3 = decl.rank(NODE_RANK)
///                        .role(Ioss::Field::TRANSIENT)
///                        .access<mundy::math::Vector3<double>>();
///   auto velocity = node_vec3.name("velocity").field().declare();
///   auto force    = node_vec3.name("force").field().declare();
/// \endcode
class ComponentDeclarationHelper {
 public:
  //! \name Constructors and Assignment Operators
  //@{

  /// \brief Construct with a MetaData reference (required for field-backed declarations; optional for shared).
  explicit ComponentDeclarationHelper(stk::mesh::MetaData& meta_data) : meta_data_(&meta_data) {}

  /// \brief Default constructor for shared-backed declarations that do not require MetaData.
  ComponentDeclarationHelper() : meta_data_(nullptr) {}

  ComponentDeclarationHelper(const ComponentDeclarationHelper&) = default;
  ComponentDeclarationHelper(ComponentDeclarationHelper&&)      = default;
  ComponentDeclarationHelper& operator=(const ComponentDeclarationHelper&) = default;
  ComponentDeclarationHelper& operator=(ComponentDeclarationHelper&&)      = default;

  //@}

  //! \name Fluent setters
  //@{

  /// \brief Set the entity rank of the component.
  ComponentDeclarationHelper rank(stk::mesh::EntityRank rank) const {
    ComponentDeclarationHelper copy = *this;
    copy.snapshot_.has_rank = true;
    copy.snapshot_.rank     = rank;
    return copy;
  }

  /// \brief Set the name of the component.
  ComponentDeclarationHelper name(const std::string& component_name) const {
    ComponentDeclarationHelper copy = *this;
    copy.snapshot_.has_name   = true;
    copy.snapshot_.field_name = component_name;
    return copy;
  }

  /// \brief Set the I/O role for field-backed components.
  ///
  /// The typical Mundy application will label fields as \c TRANSIENT or \c MESH.
  ComponentDeclarationHelper role(Ioss::Field::RoleType field_role) const {
    ComponentDeclarationHelper copy = *this;
    copy.snapshot_.has_role   = true;
    copy.snapshot_.field_role = field_role;
    return copy;
  }

  /// \brief Set the STK output type for field-backed components.
  ComponentDeclarationHelper output_type(stk::io::FieldOutputType output_type) const {
    ComponentDeclarationHelper copy = *this;
    copy.snapshot_.has_output_type = true;
    copy.snapshot_.output_type     = output_type;
    return copy;
  }

  //@}

  //! \name Terminal transitions
  //@{

  /// \brief Set the access shape, returning an access-typed builder that supports both \c .declare() and \c .shared().
  ///
  /// \tparam AccessLike  Access shape: arithmetic type, Mundy math type, or explicit \c access:: tag.
  template <typename AccessLike>
  auto access() const {
    impl::FieldDeclarationSnapshot snap = snapshot_;
    snap.meta_data                      = meta_data_;
    return TaggedFieldComponentDeclarationHelperT<void, AccessLike, void>(snap);
  }

  /// \brief Attach a semantic tag before access is chosen.
  ///
  /// \tparam Tag  Tag type to attach to the resulting component.
  template <typename Tag>
  auto tag() const {
    impl::FieldDeclarationSnapshot snap = snapshot_;
    snap.meta_data                      = meta_data_;
    return TaggedFieldDeclarationHelperT<void, Tag>(snap);
  }

  //@}

 private:
  stk::mesh::MetaData*           meta_data_ = nullptr;
  impl::FieldDeclarationSnapshot snapshot_;
};  // ComponentDeclarationHelper

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_DECLARECOMPONENT_HPP_
