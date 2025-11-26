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

#ifndef MUNDY_MESH_DECLAREPART_HPP_
#define MUNDY_MESH_DECLAREPART_HPP_

/// \file DeclarePart.hpp
/// \brief A set of helpers for declaring parts with reduced boilerplate code.

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

enum class IOPartRole { NONE, IO, ASSEMBLY, EDGE_BLOCK };

/// \brief Helper class for declaring a part
///
/// This class is used to aid the declaration of a part on the mesh with reduced boilerplate.
/// It uses a fluent interface to set the part properties and then declare the part.
/// 
/// There are three types of parts that may be declared:
///   1. Named parts (name, but no rank or topology)
///   2. Ranked parts (name and rank, but no topology)
///   3. Topological parts (name and topology, but no rank)
///
/// You may not specify both a rank and a topology for the same part.
///
/// For example, to create an element rank assembly part that contains all beams and spheres:
/// \code{.cpp}
///   PartDeclarationHelper part_decl(meta_data);
///   stk::mesh::Part &spheres      = part_decl.name("spheres").topology(stk::topology::PARTICLE).role(IO).declare();
///   stk::mesh::Part &beams        = part_decl.name("beams").topology(stk::topology::BEAM_2).role(IO).declare();
///   stk::mesh::Part &rigid_bodies = part_decl.name("rigid_bodies").rank(ELEM_RANK).role(ASSEMBLY).subpart(spheres).subpart(beams).declare();
/// \endcode
///
/// These setters may be called in any order. Role and subparts are optional, but you must call a valid combination of name, rank, and topology before declare().
///
/// You may also reuse the same PartDeclarationHelper to declare multiple parts with similar properties:
/// \code{.cpp}
///   PartDeclarationHelper part_decl(meta_data);
///   auto io_particle_part_decl = part_decl.topology(stk::topology::PARTICLE).role(IO).declare();
///   stk::mesh::Part &spheres      = io_particle_part_decl.name("spheres").declare();
///   stk::mesh::Part &points       = io_particle_part_decl.name("points").declare();
/// \endcode
class PartDeclarationHelper {
 public:
  //! \name Constructors and Assignment Operators

  /// \brief Canonical constructor
  PartDeclarationHelper(stk::mesh::MetaData &meta_data)
      : meta_data_(meta_data),
        part_has_name_(false),
        part_has_rank_(false),
        part_has_topology_(false),
        part_has_subparts_(false),
        part_has_role_(false) {
  }

  /// \brief Copy/Move constructors and assignment operators
  PartDeclarationHelper(const PartDeclarationHelper &) = default;
  PartDeclarationHelper(PartDeclarationHelper &&) = default;
  PartDeclarationHelper &operator=(const PartDeclarationHelper &) = default;
  PartDeclarationHelper &operator=(PartDeclarationHelper &&) = default;
  //@}

  //! \name Fluent interface
  //@{

  /// \brief Set the name of the part (must be called before declare())
  PartDeclarationHelper name(const std::string &part_name) {
    part_has_name_ = true;
    part_name_ = part_name;
    return *this;
  }

  /// \brief Set the entity rank of the part
  PartDeclarationHelper rank(stk::mesh::EntityRank part_rank) {
    part_has_rank_ = true;
    part_rank_ = part_rank;
    return *this;
  }

  /// \brief Set the topology of the part
  PartDeclarationHelper topology(stk::topology::topology_t part_topology) {
    part_has_topology_ = true;
    part_topology_ = part_topology;
    return *this;
  }

  /// \brief Set the io role of the part (optional)
  PartDeclarationHelper role(IOPartRole io_part_role) {
    part_has_role_ = true;
    part_role_ = io_part_role;
    return *this;
  }

  /// \brief Add a subpart to the part (i.e, declare the given part as a subset of this part)
  PartDeclarationHelper subpart(const stk::mesh::Part &subpart) {
    part_has_subparts_ = true;
    subset_part_ids_.push_back(subpart.mesh_meta_data_ordinal());
    return *this;
  }

  /// \brief Declare a part with the given properties.
  stk::mesh::Part &declare() {
    // Validate that required parameters have been set
    MUNDY_THROW_REQUIRE(part_has_name_, std::logic_error, "Part name must be set before declaring a part.");

    bool is_named_part = part_has_name_ && !part_has_rank_ && !part_has_topology_;
    bool is_ranked_part = part_has_name_ && part_has_rank_ && !part_has_topology_;
    bool is_topological_part = part_has_name_ && !part_has_rank_ && part_has_topology_;
    MUNDY_THROW_REQUIRE(
        is_named_part || is_ranked_part || is_topological_part, std::logic_error,
        fmt::format(
            "Part with name ('{}') is not properly specified. You may either specify:\n"
            "   1. A name (but no rank or topology)    -> meta_data.declare_part('name')\n"
            "   2. A name and a rank (but no topology) -> meta_data.declare_part('name', rank)\n"
            "   3. A name and a topology (but no rank) -> meta_data.declare_part_with_topology('name', topology)\n"
            "However, you have specified both a rank and a topology.",
            part_name_));

    if (is_named_part) {
      return internal_declare_named_part();
    } else if (is_ranked_part) {
      return internal_declare_ranked_part();
    } else {  // is_topological_part
      return internal_declare_topological_part();
    }
  }

  /// \brief Print the part declaration information to the output stream.
  void print(std::ostream &os = std::cout) const {
    os << "PartDeclarationHelper:" << std::endl;
    if (part_has_name_) {
      os << "  Name: " << part_name_ << std::endl;
    }
    if (part_has_rank_) {
      os << "  Rank: " << part_rank_ << std::endl;
    }
    if (part_has_topology_) {
      os << "  Topology: " << stk::topology(part_topology_) << std::endl;
    }
    if (part_has_subparts_) {
      os << "  Subparts: ";
      for (unsigned subpart_id : subset_part_ids_) {
        os << subpart_id << " ";
      }
      os << std::endl;
    }
    if (part_has_role_) {
      os << "  Role: ";
      switch (part_role_) {
        case IOPartRole::IO:
          os << "IO";
          break;
        case IOPartRole::ASSEMBLY:
          os << "ASSEMBLY";
          break;
        case IOPartRole::EDGE_BLOCK:
          os << "EDGE_BLOCK";
          break;
        case IOPartRole::NONE:
        default:
          os << "NONE";
          break;
      }
      os << std::endl;
    }
  }
  //@}

 private:
  void apply_optional_properties(stk::mesh::Part &part) {
    // Apply optional subparts
    if (part_has_subparts_) {
      for (unsigned subpart_id : subset_part_ids_) {
        stk::mesh::Part &subpart = meta_data_.get_part(subpart_id);
        meta_data_.declare_part_subset(part, subpart);
      }
    }

    // Apply optional role
    if (part_has_role_) {
      switch (part_role_) {
        case IOPartRole::IO:
          stk::io::put_io_part_attribute(part);
          break;
        case IOPartRole::ASSEMBLY:
          stk::io::put_assembly_io_part_attribute(part);
          break;
        case IOPartRole::EDGE_BLOCK:
          stk::io::put_edge_block_io_part_attribute(part);
          break;
        case IOPartRole::NONE:
        default:
          // Do nothing
          break;
      }
    }
  }

  stk::mesh::Part &internal_declare_named_part() {
    stk::mesh::Part &part = meta_data_.declare_part(part_name_);
    apply_optional_properties(part);
    return part;
  }

  stk::mesh::Part &internal_declare_ranked_part() {
    stk::mesh::Part &part = meta_data_.declare_part(part_name_, part_rank_);
    apply_optional_properties(part);
    return part;
  }

  stk::mesh::Part &internal_declare_topological_part() {
    stk::mesh::Part &part = meta_data_.declare_part_with_topology(part_name_, part_topology_);
    apply_optional_properties(part);
    return part;
  }

  stk::mesh::MetaData &meta_data_;

  // Part properties
  bool part_has_name_;
  bool part_has_rank_;
  bool part_has_topology_;
  bool part_has_subparts_;
  bool part_has_role_;

  std::string part_name_;
  stk::mesh::EntityRank part_rank_;
  stk::topology::topology_t part_topology_;
  std::vector<unsigned> subset_part_ids_;
  IOPartRole part_role_;
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_DECLAREPART_HPP_
