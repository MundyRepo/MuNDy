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

// C++ core
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Mundy
#include <mundy_mesh/impl/ClassPartitionKey.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace mesh {

namespace impl {

ClassPartitionKey get_class_partition_key(const ClassVector& classes) {
  ClassPartitionKey key;
  key.reserve(classes.size());
  for (const Class* cls : classes) {
    key.push_back(cls->class_ordinal());
  }
  std::sort(key.begin(), key.end());
  key.erase(std::unique(key.begin(), key.end()), key.end());
  return key;
}

ClassVector get_classes_for_class_partition_key(const ClassPartitionKey& key, stk::mesh::MetaData& meta_data) {
  // Build a map from ordinal to Class* for all registered classes.
  const ClassVector& all_classes = get_classes(meta_data);
  std::map<Class::class_ordinal_t, Class*> ordinal_map;
  for (Class* cls : all_classes) {
    ordinal_map[cls->class_ordinal()] = cls;
  }

  ClassVector result;
  result.reserve(key.size());
  for (Class::class_ordinal_t ordinal : key) {
    auto it = ordinal_map.find(ordinal);
    MUNDY_THROW_REQUIRE(it != ordinal_map.end(), std::logic_error,
                        sink() << "get_classes_for_class_partition_key: class ordinal " << ordinal
                               << " not found in MetaData — the class may not have been declared on this MetaData.");
    result.push_back(it->second);
  }
  return result;
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy
