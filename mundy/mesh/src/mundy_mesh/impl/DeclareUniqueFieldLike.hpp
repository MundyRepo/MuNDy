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

#ifndef MUNDY_MESH_IMPL_DECLAREUNIQUEFIELDLIKE_HPP_
#define MUNDY_MESH_IMPL_DECLAREUNIQUEFIELDLIKE_HPP_

/// \file impl/DeclareUniqueFieldLike.hpp
/// \brief Helper to declare a new, uniquely-named field shaped like an existing field.

// C++ core
#include <cstddef>  // for size_t
#include <map>      // for std::map
#include <string>   // for std::string, std::to_string

// Trilinos
#include <stk_mesh/base/Field.hpp>             // for stk::mesh::Field, put_field_on_mesh
#include <stk_mesh/base/FieldBase.hpp>         // for stk::mesh::FieldBase
#include <stk_mesh/base/FieldRestriction.hpp>  // for stk::mesh::FieldRestriction
#include <stk_mesh/base/MetaData.hpp>          // for stk::mesh::MetaData, declare_field, put_field_on_mesh
#include <stk_mesh/base/Types.hpp>             // for stk::mesh::EntityRank

namespace mundy {

namespace mesh {

namespace impl {

/// \struct UniqueFieldLikeCounter
/// \brief Per-`MetaData` map: source field ordinal -> number of unique fields already derived from it.
///
/// Keyed by field *ordinal* (not name) to avoid any name-casing ambiguity, and stored as a `MetaData` attribute so
/// the counter persists with the mesh.
struct UniqueFieldLikeCounter {
  std::map<unsigned, size_t> next_count;
};

/// \brief Fetch (creating if needed) the per-`MetaData` unique-field-like counter attribute.
inline UniqueFieldLikeCounter& get_or_create_unique_field_like_counter(stk::mesh::MetaData& meta_data) {
  auto* counter = const_cast<UniqueFieldLikeCounter*>(meta_data.get_attribute<UniqueFieldLikeCounter>());
  if (counter == nullptr) {
    const UniqueFieldLikeCounter* fresh = new UniqueFieldLikeCounter();
    counter = const_cast<UniqueFieldLikeCounter*>(meta_data.declare_attribute_with_delete(fresh));
  }
  return *counter;
}

/// \brief Declare a NEW field with a unique name, shaped like `like_field`.
///
/// The new field has the same scalar type, entity rank, and per-partition sizing (it replays `like_field`'s
/// restrictions, so it lands on the same parts with the same components-per-entity). Its name is
/// `like_field.name() + "_" + post_script + "_" + <n>`, where `<n>` is a per-(MetaData, source-field-ordinal)
/// counter — so repeated calls for the same source field yield *distinct* fields and independent consumers (e.g.
/// multiple rebuilder instances) never collide on a shared name.
///
/// This does NOT enable late fields; if the mesh is committed the caller must scope `enable_late_fields()` around
/// the call.
/// \tparam T Field scalar type (must match `like_field`'s value type).
/// \param meta_data [in] Mesh metadata that will own the new field.
/// \param like_field [in] Field whose value type, rank, and restrictions are mirrored.
/// \param post_script [in] Human-readable tag embedded in the unique name (default `"scratch"`).
template <typename T>
stk::mesh::Field<T>& declare_unique_field_like(stk::mesh::MetaData& meta_data, const stk::mesh::FieldBase& like_field,
                                               const std::string& post_script = "scratch") {
  UniqueFieldLikeCounter& counter = get_or_create_unique_field_like_counter(meta_data);
  const size_t n = counter.next_count[like_field.mesh_meta_data_ordinal()]++;
  const std::string name = like_field.name() + "_" + post_script + "_" + std::to_string(n);

  stk::mesh::Field<T>& field = meta_data.declare_field<T>(like_field.entity_rank(), name);
  for (const stk::mesh::FieldRestriction& restriction : like_field.restrictions()) {
    stk::mesh::put_field_on_mesh(field, restriction.selector(),
                                 static_cast<unsigned>(restriction.num_scalars_per_entity()), nullptr);
  }
  return field;
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_DECLAREUNIQUEFIELDLIKE_HPP_
