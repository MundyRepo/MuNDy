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

#ifndef MUNDY_MESH_IMPL_DECLAREFIELDIMPL_HPP_
#define MUNDY_MESH_IMPL_DECLAREFIELDIMPL_HPP_

/// \file DeclareFieldImpl.hpp
/// \brief A set of helpers for declaring fields with reduced boilerplate code.

// External
#include <fmt/format.h>  // for fmt::format

// C++ core
#include <stdexcept>  // for std::runtime_error
#include <utility>

// Trilinos
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::FieldOutputType
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData

// Mundy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace mesh {

namespace impl {

struct FieldDeclarationSnapshot {
  stk::mesh::MetaData* meta_data = nullptr;
  bool has_rank = false;
  bool has_name = false;
  bool has_role = false;
  bool has_output_type = false;
  stk::mesh::EntityRank rank = stk::topology::INVALID_RANK;
  std::string field_name;
  Ioss::Field::RoleType field_role = Ioss::Field::RoleType{};
  stk::io::FieldOutputType output_type = stk::io::FieldOutputType::CUSTOM;
};

template <typename FieldHelper>
FieldDeclarationSnapshot make_field_declaration_snapshot(const FieldHelper& field_helper) {
  FieldDeclarationSnapshot snapshot;
  snapshot.meta_data = &field_helper.meta_data();
  snapshot.has_rank = field_helper.has_rank();
  snapshot.has_name = field_helper.has_name();
  snapshot.has_role = field_helper.has_role();
  snapshot.has_output_type = field_helper.has_output_type();
  if (snapshot.has_rank) {
    snapshot.rank = field_helper.rank_value();
  }
  if (snapshot.has_name) {
    snapshot.field_name = field_helper.name_value();
  }
  if (snapshot.has_role) {
    snapshot.field_role = field_helper.role_value();
  }
  if (snapshot.has_output_type) {
    snapshot.output_type = field_helper.output_type_value();
  }
  return snapshot;
}

template <typename T>
stk::mesh::Field<T>& declare_field_from_snapshot(const FieldDeclarationSnapshot& snapshot) {
  MUNDY_THROW_ASSERT(snapshot.meta_data != nullptr, std::logic_error, "Field declaration metadata is null.");
  MUNDY_THROW_REQUIRE(snapshot.has_name, std::logic_error, "Field name must be set before declaring a component.");
  MUNDY_THROW_REQUIRE(snapshot.has_rank, std::logic_error, "Field rank must be set before declaring a component.");

  stk::mesh::Field<T>& field = snapshot.meta_data->declare_field<T>(snapshot.rank, snapshot.field_name);

  if (snapshot.has_role) {
    stk::io::set_field_role(field, snapshot.field_role);
  }
  if (snapshot.has_output_type) {
    stk::io::set_field_output_type(field, snapshot.output_type);
  }

  return field;
}

template <stk::topology::rank_t Rank>
void apply_tag_rank_default(bool& has_rank, stk::mesh::EntityRank& rank) {
  if constexpr (Rank != stk::topology::INVALID_RANK) {
    if (!has_rank) {
      has_rank = true;
      rank = Rank;
    }
  }
}

template <stk::topology::rank_t Rank>
void require_rank_matches_tag(stk::mesh::EntityRank rank) {
  if constexpr (Rank != stk::topology::INVALID_RANK) {
    MUNDY_THROW_REQUIRE(rank == Rank, std::invalid_argument,
                        fmt::format("Component declaration rank {} does not match tag rank {}.", rank, Rank));
  }
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_DECLAREFIELDIMPL_HPP_
