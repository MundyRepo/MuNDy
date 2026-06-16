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

#ifndef MUNDY_MESH_DECLAREFIELD_HPP_
#define MUNDY_MESH_DECLAREFIELD_HPP_

/// \file DeclareField.hpp
/// \brief A set of helpers for declaring fields with reduced boilerplate code.

// C++ core
#include <iostream>   // for std::ostream
#include <stdexcept>  // for std::runtime_error
#include <type_traits>
#include <utility>
#include <vector>  // for std::vector

// Trilinos
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData

// Mundy
#include <mundy_mesh/impl/DeclareFieldImpl.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE
#include <Mundy_config.hpp>  // for MUNDY_DEPRECATED_MSG

namespace mundy {

namespace mesh {

template <typename FieldScalarType, typename Tag = void>
class TaggedFieldDeclarationT;

/// \brief Helper class for declaring a field
///
/// This class is used to aid the declaration of a field on the mesh with reduced boilerplate.
/// It uses a fluent interface to set the field properties and then declare the field.
///
/// For example, to create a transient vector3 field on nodes called "velocity":
/// \code{.cpp}
///   FieldDeclaration field_decl(meta_data);
///   stk::mesh::Field<double> &node_velocity_field =
///      field_decl.type<double>().role(TRANSIENT).output_type(VECTOR_3D).rank(NODE_RANK).name("velocity").declare();
/// \endcode
///
/// These setters may be called in any order. Role and output type are optional, but type(), rank(), and name() must be
/// called before declare().
///
/// You may also reuse the same FieldDeclaration to declare multiple fields with similar properties:
/// \code{.cpp}
///   FieldDeclaration field_decl(meta_data);
///   auto vec3d_io_field_decl = field_decl.type<double>().role(TRANSIENT).output_type(VECTOR_3D);
///   stk::mesh::Field<double> &node_velocity_field = vec3d_io_field_decl.rank(NODE_RANK).name("velocity").declare();
///   stk::mesh::Field<double> &elem_force_field    = vec3d_io_field_decl.rank(ELEMENT_RANK).name("force").declare();
/// \endcode
class FieldDeclaration;

template <typename T>
class FieldDeclarationT {
 public:
  using field_value_typeype = T;

  //! \name Constructors and Assignment Operators
  //@{

  /// \brief Canonical constructor
  FieldDeclarationT(stk::mesh::MetaData& meta_data)
      : meta_data_(meta_data),
        field_has_rank_(false),
        field_has_name_(false),
        field_has_role_(false),
        field_has_output_type_(false),
        rank_(stk::topology::INVALID_RANK),
        field_name_(),
        field_role_(Ioss::Field::RoleType{}),
        output_type_(stk::io::FieldOutputType::CUSTOM) {
  }

  /// \brief Copy/Move constructors and assignment operators
  FieldDeclarationT(const FieldDeclarationT&) = default;
  FieldDeclarationT(FieldDeclarationT&&) = default;
  FieldDeclarationT& operator=(const FieldDeclarationT&) = default;
  FieldDeclarationT& operator=(FieldDeclarationT&&) = default;
  //@}

  //! \name Fluent interface
  //@{

  /// \brief Set the entity rank of the field (must be called before declare())
  FieldDeclarationT rank(stk::mesh::EntityRank rank) {
    field_has_rank_ = true;
    rank_ = rank;
    return *this;
  }

  /// \brief Set the name of the field (must be called before declare())
  FieldDeclarationT name(const std::string& field_name) {
    field_has_name_ = true;
    field_name_ = field_name;
    return *this;
  }

  /// \brief Set the io role of the field (optional)
  ///
  /// The typical Mundy application will label fields as TRANSIENT or MESH.
  /// Note, the NODE_COORDINATES field is special and is automatically assigned the MESH role by stk.
  /// If you attempt to give it a different role, an error will be thrown.
  ///
  /// Possible roles include:
  ///    INTERNAL,
  ///    MESH,      /**< A field which is used to define the basic geometry
  ///                    or topology of the model and is not normally transient
  ///                    in nature. Examples would be element connectivity or
  ///                    nodal coordinates. */
  ///    ATTRIBUTE, /**< A field which is used to define an attribute on an
  ///                    EntityBlock derived class. Examples would be thickness
  ///                    of the elements in a shell element block or the radius
  ///                    of particles in a particle element block. */
  ///    MAP,
  ///    COMMUNICATION,
  ///    MESH_REDUCTION, /**< A field which summarizes some non-transient data
  ///                       about an entity (\sa REDUCTION). This could be an
  ///                       offset applied to an element block, or the units
  ///                       system of a model or the name of the solid model
  ///                       which this entity is modelling... */
  ///    INFORMATION = MESH_REDUCTION,
  ///    REDUCTION, /**< A field which typically summarizes some transient data
  ///                    about an entity. The size of this field is typically not
  ///                    proportional to the number of entities in a GroupingEntity.
  ///                    An example would be average displacement over a group of
  ///                    nodes or the kinetic energy of a model. This data is also
  ///                    transient. */
  ///    TRANSIENT  /**< A field which is typically calculated at multiple steps
  ///                    or times in an analysis. These are typically "results"
  ///                    data. Examples would be nodal displacement or element
  ///                    stress. */
  FieldDeclarationT role(Ioss::Field::RoleType field_role) {
    field_has_role_ = true;
    field_role_ = field_role;
    return *this;
  }

  /// \brief Set the stk output type of the field (optional)
  ///
  /// The output type for a field defines how its individual components are subscripted.
  /// For example a vector2 field with name "velocity" will have components velocity_x, velocity_y.
  ///
  /// The possible output types and their resulting subscripting are:
  ///  SCALAR,           //  []
  ///  VECTOR_2D,        //  [x, y]
  ///  VECTOR_3D,        //  [x, y, z]
  ///  FULL_TENSOR_36,   //  [xx, yy, zz, xy, yz, zx, yx, zy, xz]
  ///  FULL_TENSOR_32,   //  [xx, yy, zz, xy, yx]
  ///  FULL_TENSOR_22,   //  [xx, yy, xy, yx]
  ///  FULL_TENSOR_16,   //  [xx, xy, yz, zx, yx, zy, xz]
  ///  FULL_TENSOR_12,   //  [xx, xy, yx]
  ///  SYM_TENSOR_33,    //  [xx, yy, zz, xy, yz, zx]
  ///  SYM_TENSOR_31,    //  [xx, yy, zz, xy]
  ///  SYM_TENSOR_21,    //  [xx, yy, xy]
  ///  SYM_TENSOR_13,    //  [xx, xy, yz, zx]
  ///  SYM_TENSOR_11,    //  [xx, xy]
  ///  SYM_TENSOR_10,    //  [xx]
  ///  ASYM_TENSOR_03,   //  [xy, yz, zx]
  ///  ASYM_TENSOR_02,   //  [xy, yz]
  ///  ASYM_TENSOR_01,   //  [xy]
  ///  MATRIX_22,        //  [xx, xy, yx, yy]
  ///  MATRIX_33,        //  [xx, xy, xz, yx, yy, yz, zx, zy, zz]
  ///  QUATERNION_2D,    //  [s, q]
  ///  QUATERNION_3D,    //  [x, y, z, q]
  ///  CUSTOM            //  User-defined subscripting
  FieldDeclarationT output_type(stk::io::FieldOutputType output_type) {
    field_has_output_type_ = true;
    output_type_ = output_type;
    return *this;
  }

  /// \brief Declare a field with the given stk output type and role.
  stk::mesh::Field<T>& declare() {
    // Validate that required parameters have been set
    MUNDY_THROW_REQUIRE(field_has_name_, std::logic_error, "Field name must be set before declaring a field.");
    MUNDY_THROW_REQUIRE(field_has_rank_, std::logic_error, "Field rank must be set before declaring a field.");

    // Declare the field
    stk::mesh::Field<T>& field = meta_data_.declare_field<T>(rank_, field_name_);

    // Set optional role and output type
    if (field_has_role_) {
      stk::io::set_field_role(field, field_role_);
    }
    if (field_has_output_type_) {
      stk::io::set_field_output_type(field, output_type_);
    }

    return field;
  }

  // clang-format off
  stk::mesh::MetaData& meta_data() const { return meta_data_; }
  bool has_rank() const { return field_has_rank_; }
  bool has_name() const { return field_has_name_; }
  bool has_role() const { return field_has_role_; }
  bool has_output_type() const { return field_has_output_type_; }
  stk::mesh::EntityRank rank_value() const { return rank_; }
  const std::string& name_value() const { return field_name_; }
  Ioss::Field::RoleType role_value() const { return field_role_; }
  stk::io::FieldOutputType output_type_value() const { return output_type_; }
  // clang-format on

 private:
  static stk::mesh::MetaData& get_meta_data_from_snapshot(const impl::FieldDeclarationSnapshot& snapshot) {
    MUNDY_THROW_ASSERT(snapshot.meta_data != nullptr, std::logic_error, "Field declaration metadata is null.");
    return *snapshot.meta_data;
  }

  explicit FieldDeclarationT(const impl::FieldDeclarationSnapshot& snapshot)
      : meta_data_(get_meta_data_from_snapshot(snapshot)),
        field_has_rank_(snapshot.has_rank),
        field_has_name_(snapshot.has_name),
        field_has_role_(snapshot.has_role),
        field_has_output_type_(snapshot.has_output_type),
        rank_(snapshot.rank),
        field_name_(snapshot.field_name),
        field_role_(snapshot.field_role),
        output_type_(snapshot.output_type) {
  }

  stk::mesh::MetaData& meta_data_;

  bool field_has_rank_;
  bool field_has_name_;
  bool field_has_role_;
  bool field_has_output_type_;

  stk::mesh::EntityRank rank_;
  std::string field_name_;
  Ioss::Field::RoleType field_role_;
  stk::io::FieldOutputType output_type_;

  friend class FieldDeclaration;
};

class FieldDeclaration {
 public:
  //! \name Constructors and Assignment Operators
  //@{

  using invalid_field_value_typeype = void;

  /// \brief Canonical constructor
  FieldDeclaration(stk::mesh::MetaData& meta_data)
      : meta_data_(meta_data),
        field_has_rank_(false),
        field_has_name_(false),
        field_has_role_(false),
        field_has_output_type_(false),
        rank_(stk::topology::INVALID_RANK),
        field_name_(),
        field_role_(Ioss::Field::RoleType{}),
        output_type_(stk::io::FieldOutputType::CUSTOM) {
  }

  /// \brief Copy/Move constructors and assignment operators
  FieldDeclaration(const FieldDeclaration&) = default;
  FieldDeclaration(FieldDeclaration&&) = default;
  FieldDeclaration& operator=(const FieldDeclaration&) = default;
  FieldDeclaration& operator=(FieldDeclaration&&) = default;

  //! \name Fluent interface
  //@{

  /// \brief Set the type of the field (must be called before declare())
  template <typename T>
  FieldDeclarationT<T> type() {
    return FieldDeclarationT<T>(impl::make_field_declaration_snapshot(*this));
  }

  /// \brief Set the entity rank of the field (must be called before declare())
  FieldDeclaration rank(stk::mesh::EntityRank rank) {
    field_has_rank_ = true;
    rank_ = rank;
    return *this;
  }

  /// \brief Set the name of the field (must be called before declare())
  FieldDeclaration name(const std::string& field_name) {
    field_has_name_ = true;
    field_name_ = field_name;
    return *this;
  }

  /// \brief Set the io role of the field (optional)
  FieldDeclaration role(Ioss::Field::RoleType field_role) {
    field_has_role_ = true;
    field_role_ = field_role;
    return *this;
  }

  /// \brief Set the stk output type of the field (optional)
  FieldDeclaration output_type(stk::io::FieldOutputType output_type) {
    field_has_output_type_ = true;
    output_type_ = output_type;
    return *this;
  }

  // clang-format off
  stk::mesh::MetaData& meta_data() const { return meta_data_; }
  bool has_rank() const { return field_has_rank_; }
  bool has_name() const { return field_has_name_; }
  bool has_role() const { return field_has_role_; }
  bool has_output_type() const { return field_has_output_type_; }
  stk::mesh::EntityRank rank_value() const { return rank_; }
  const std::string& name_value() const { return field_name_; }
  Ioss::Field::RoleType role_value() const { return field_role_; }
  stk::io::FieldOutputType output_type_value() const { return output_type_; }
  // clang-format on

 private:
  stk::mesh::MetaData& meta_data_;

  bool field_has_rank_;
  bool field_has_name_;
  bool field_has_role_;
  bool field_has_output_type_;

  stk::mesh::EntityRank rank_;
  std::string field_name_;
  Ioss::Field::RoleType field_role_;
  stk::io::FieldOutputType output_type_;
};

using FieldDeclarationHelper MUNDY_DEPRECATED_MSG("use FieldDeclaration") = FieldDeclaration;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_DECLAREFIELD_HPP_
