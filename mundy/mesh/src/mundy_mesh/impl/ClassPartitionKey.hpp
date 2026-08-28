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

#ifndef MUNDY_MESH_IMPL_CLASSPARTITIONKEY_HPP_
#define MUNDY_MESH_IMPL_CLASSPARTITIONKEY_HPP_

/// \file ClassPartitionKey.hpp
///
/// Analogous to PartitionKey.hpp for STK parts: a ClassPartitionKey is a sorted, deduplicated
/// vector of class ordinals that uniquely identifies a set of Classes.  It serves the same two
/// roles as PartitionKey:
///
///  1. Memoisation key — two ClassVectors that contain the same set of classes (regardless of
///     order or duplicates) produce the same ClassPartitionKey and will share a request helper.
///
///  2. Reconstruction — get_classes_for_class_partition_key recovers the ClassVector from the
///     key and a MetaData reference, mirroring get_parts_for_partition_key.

// C++ core
#include <algorithm>
#include <stdexcept>
#include <vector>

// STK
#include <stk_mesh/base/MetaData.hpp>

// Mundy
#include <mundy_mesh/Class.hpp>  // for mundy::mesh::Class, mundy::mesh::ClassVector
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace mesh {

namespace impl {

/// \brief Sorted, deduplicated vector of class ordinals — the canonical key for a set of Classes.
using ClassPartitionKey = std::vector<Class::class_ordinal_t>;

/// \brief Build a ClassPartitionKey from a ClassVector.
///
/// The result is independent of the order of \p classes and suppresses duplicates, so any two
/// ClassVectors that contain the same set of classes produce the same key.
ClassPartitionKey get_class_partition_key(const ClassVector& classes);

/// \brief Reconstruct a ClassVector from a ClassPartitionKey and the mesh MetaData.
///
/// Iterates the registered classes on \p meta_data and returns those whose class_ordinal()
/// appears in \p key, in ordinal order.  Throws if any ordinal in \p key has no matching class.
ClassVector get_classes_for_class_partition_key(const ClassPartitionKey& key, stk::mesh::MetaData& meta_data);

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_CLASSPARTITIONKEY_HPP_
