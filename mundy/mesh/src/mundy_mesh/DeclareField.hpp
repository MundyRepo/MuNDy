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

// External
#include <fmt/format.h>  // for fmt::format

// C++ core
#include <iostream>       // for std::ostream
#include <stdexcept>      // for std::runtime_error
#include <vector>         // for std::vector

// Trilinos
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::StkMeshIoBroker

// Mundy
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace mesh {

/// \brief Helper class for declaring a field
///
/// This class is used to aid the declaration of a field on the mesh with reduced boilerplate.
/// It uses a fluent interface to set the field properties and then declare the field.
/// 
/// For example, to create a transient vector3 field on nodes called "velocity":
/// \code{.cpp}
///   FieldDeclarationHelper field_decl(meta_data);
///   stk::mesh::Field<double> &node_velocity_field = 
///      field_decl.type<double>().role(TRANSIENT).output_type(VECTOR_3D).rank(NODE_RANK).name("velocity").declare();
/// \endcode
///
/// These setters may be called in any order. Role and output type are optional, but type(), rank(), and name() must be called before declare().
///
/// You may also reuse the same FieldDeclarationHelper to declare multiple fields with similar properties:
/// \code{.cpp}
///   FieldDeclarationHelper field_decl(meta_data);
///   auto vec3d_io_field_decl = field_decl.type<double>().role(TRANSIENT).output_type(VECTOR_3D);
///   stk::mesh::Field<double> &node_velocity_field = vec3d_io_field_decl.rank(NODE_RANK).name("velocity").declare();
///   stk::mesh::Field<double> &elem_force_field    = vec3d_io_field_decl.rank(ELEMENT_RANK).name("force").declare();
class FieldDeclarationHelper;

template <typename T>
class FieldDeclarationHelperT {
 public:
  //! \name Constructors and Assignment Operators
  //@{

  /// \brief Canonical constructor
  FieldDeclarationHelperT(stk::mesh::MetaData &meta_data)
      : meta_data_(meta_data),
        field_has_rank_(false),
        field_has_name_(false),
        field_has_role_(false),
        field_has_output_type_(false) {
  }

  /// \brief Copy/Move constructors and assignment operators
  FieldDeclarationHelperT(const FieldDeclarationHelperT &) = default;
  FieldDeclarationHelperT(FieldDeclarationHelperT &&) = default;
  FieldDeclarationHelperT &operator=(const FieldDeclarationHelperT &) = default;
  FieldDeclarationHelperT &operator=(FieldDeclarationHelperT &&) = default;
  //@}

  //! \name Fluent interface
  //@{

  /// \brief Set the entity rank of the field (must be called before declare())
  FieldDeclarationHelperT rank(stk::mesh::EntityRank rank) {
    field_has_rank_ = true;
    rank_ = rank;
    return *this;
  }

  /// \brief Set the name of the field (must be called before declare())
  FieldDeclarationHelperT name(const std::string &field_name) {
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
  FieldDeclarationHelperT role(Ioss::Field::RoleType field_role) {
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
  FieldDeclarationHelperT output_type(stk::io::FieldOutputType output_type) {
    field_has_output_type_ = true;
    output_type_ = output_type;
    return *this;
  }

  /// \brief Declare a field with the given stk output type and role.
  stk::mesh::Field<T> &declare() {
    // Validate that required parameters have been set
    MUNDY_THROW_REQUIRE(field_has_name_, std::logic_error, "Field name must be set before declaring a field.");
    MUNDY_THROW_REQUIRE(field_has_rank_, std::logic_error, "Field rank must be set before declaring a field.");

    // Declare the field
    stk::mesh::Field<T> &field = meta_data_.declare_field<T>(rank_, field_name_);

    // Set optional role and output type
    if (field_has_role_) {
      stk::io::set_field_role(field, field_role_);
    }
    if (field_has_output_type_) {
      stk::io::set_field_output_type(field, output_type_);
    }

    return field;
  }

 private:
  stk::mesh::MetaData &meta_data_;

  bool field_has_rank_;
  bool field_has_name_;
  bool field_has_role_;
  bool field_has_output_type_;

  stk::mesh::EntityRank rank_;
  std::string field_name_;
  Ioss::Field::RoleType field_role_;
  stk::io::FieldOutputType output_type_;
};

class FieldDeclarationHelper {
 public:
  //! \name Constructors and Assignment Operators
  //@{

  /// \brief Canonical constructor
  FieldDeclarationHelper(stk::mesh::MetaData &meta_data)
      : meta_data_(meta_data),
        field_has_rank_(false),
        field_has_name_(false),
        field_has_role_(false),
        field_has_output_type_(false) {
  }

  /// \brief Copy/Move constructors and assignment operators
  FieldDeclarationHelper(const FieldDeclarationHelper &) = default;
  FieldDeclarationHelper(FieldDeclarationHelper &&) = default;
  FieldDeclarationHelper &operator=(const FieldDeclarationHelper &) = default;
  FieldDeclarationHelper &operator=(FieldDeclarationHelper &&) = default;


  //! \name Fluent interface
  //@{

  /// \brief Set the type of the field (must be called before declare())
  template <typename T>
  FieldDeclarationHelperT<T> type() {
    FieldDeclarationHelperT<T> typed_builder(meta_data_);
    if (field_has_rank_) {
      typed_builder.rank(rank_);
    }
    if (field_has_name_) {
      typed_builder.name(field_name_);
    }
    if (field_has_role_) {
      typed_builder.role(field_role_);
    }
    if (field_has_output_type_) {
      typed_builder.output_type(output_type_);
    }
    return typed_builder;
  }

  /// \brief Set the entity rank of the field (must be called before declare())
  FieldDeclarationHelper rank(stk::mesh::EntityRank rank) {
    field_has_rank_ = true;
    rank_ = rank;
    return *this;
  }

  /// \brief Set the name of the field (must be called before declare())
  FieldDeclarationHelper name(const std::string &field_name) {
    field_has_name_ = true;
    field_name_ = field_name;
    return *this;
  }

  /// \brief Set the io role of the field (optional)
  FieldDeclarationHelper role(Ioss::Field::RoleType field_role) {
    field_has_role_ = true;
    field_role_ = field_role;
    return *this;
  }

  /// \brief Set the stk output type of the field (optional)
  FieldDeclarationHelper output_type(stk::io::FieldOutputType output_type) {
    field_has_output_type_ = true;
    output_type_ = output_type;
    return *this;
  }

  /// \brief Declare a field with the given stk output type and role.
  void declare() {
    // Validate that required parameters have been set
    MUNDY_THROW_REQUIRE(field_has_name_, std::logic_error, "Field name must be set before declaring a field.");
    MUNDY_THROW_REQUIRE(field_has_rank_, std::logic_error, "Field rank must be set before declaring a field.");
    MUNDY_THROW_REQUIRE(false, std::logic_error, "Field type must be set before declaring a field.");
  }

 private:
  stk::mesh::MetaData &meta_data_;

  bool field_has_rank_;
  bool field_has_name_;
  bool field_has_role_;
  bool field_has_output_type_;

  stk::mesh::EntityRank rank_;
  std::string field_name_;
  Ioss::Field::RoleType field_role_;
  stk::io::FieldOutputType output_type_;
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_DECLAREFIELD_HPP_
