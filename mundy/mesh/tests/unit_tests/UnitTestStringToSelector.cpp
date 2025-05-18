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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// Mundy libs
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>       // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_mesh/StringToSelector.hpp>  // for mundy::mesh::string_to_selector

namespace mundy {

namespace mesh {

namespace {

std::unique_ptr<BulkData> create_bulk_data() {
  // Create a dummy mesh.
  MeshBuilder builder(MPI_COMM_WORLD);
  return builder.create_bulk_data();
}

std::pair<std::unique_ptr<BulkData>, std::vector<stk::mesh::Part *>> create_bulk_data_and_parts_with_names(
    const std::vector<std::string> &part_names = {}) {
  // Create a dummy mesh.
  std::unique_ptr<BulkData> bulk_data_ptr = create_bulk_data();
  MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Declare the parts and store them in a vector.
  std::vector<stk::mesh::Part *> parts;
  for (const std::string &part_name : part_names) {
    stk::mesh::Part &part = meta_data.declare_part(part_name);
    parts.push_back(&part);
  }
  return std::make_pair(std::move(bulk_data_ptr), parts);
}

TEST(StringToSelector, SpecialParts) {
  // We offer the following special part names: "UNIVERSAL", "LOCALLY_OWNED", "GLOBALLY_SHARED", "AURA"
  // Can we correctly fetch them from the bulk data?

  // Declare a mesh with only default parts.
  auto bulk_data_ptr = create_bulk_data();
  BulkData &bulk_data = *bulk_data_ptr.get();
  MetaData &meta_data = bulk_data.mesh_meta_data();

  // Fetch the special parts using selector strings
  auto universal_selector = string_to_selector(bulk_data, "UNIVERSAL");
  auto locally_owned_selector = string_to_selector(bulk_data, "LOCALLY_OWNED");
  auto globally_shared_selector = string_to_selector(bulk_data, "GLOBALLY_SHARED");
  auto aura_selector = string_to_selector(bulk_data, "AURA");

  // Fetch the expected selectors
  auto expected_universal_selector = stk::mesh::Selector(meta_data.universal_part());
  auto expected_locally_owned_selector = stk::mesh::Selector(meta_data.locally_owned_part());
  auto expected_globally_shared_selector = stk::mesh::Selector(meta_data.globally_shared_part());
  auto expected_aura_selector = stk::mesh::Selector(meta_data.aura_part());

  // Check that the selectors are correct
  EXPECT_EQ(universal_selector, expected_universal_selector);
  EXPECT_EQ(locally_owned_selector, expected_locally_owned_selector);
  EXPECT_EQ(globally_shared_selector, expected_globally_shared_selector);
  EXPECT_EQ(aura_selector, expected_aura_selector);
}

TEST(StringToSelector, ValidOperations) {
  // Declare a mesh with a variety of named parts.
  std::vector<std::string> part_names = {"A", "B", "C", "A1", "A_X", "A_", "weird_yet_1_valid..."};
  auto [bulk_data_ptr, parts] = create_bulk_data_and_parts_with_names(part_names);
  BulkData &bulk_data = *bulk_data_ptr.get();

  // Name the parts and their corresponding selectors for easy access
  stk::mesh::Part &p_A = *parts[0];
  stk::mesh::Part &p_B = *parts[1];
  stk::mesh::Part &p_C = *parts[2];
  stk::mesh::Part &p_A1 = *parts[3];
  stk::mesh::Part &p_A_X = *parts[4];
  stk::mesh::Part &p_A_ = *parts[5];
  stk::mesh::Part &p_weird_yet_valid = *parts[6];

  stk::mesh::Selector s_A = stk::mesh::Selector(p_A);
  stk::mesh::Selector s_B = stk::mesh::Selector(p_B);
  stk::mesh::Selector s_C = stk::mesh::Selector(p_C);
  stk::mesh::Selector s_A1 = stk::mesh::Selector(p_A1);
  stk::mesh::Selector s_A_X = stk::mesh::Selector(p_A_X);
  stk::mesh::Selector s_A_ = stk::mesh::Selector(p_A_);
  stk::mesh::Selector s_weird_yet_valid = stk::mesh::Selector(p_weird_yet_valid);

  // Selectors are allowed to be combined using the following operators:
  //  - Subtraction:    -
  //  - Arythmetic and: &
  //  - Arythmetic or:  |
  //  - Unary not:      !
  //  - Parentheses:   ( )

  // Check that we can select the parts individually
  EXPECT_EQ(string_to_selector(bulk_data, "A"), s_A);
  EXPECT_EQ(string_to_selector(bulk_data, "B"), s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "C"), s_C);
  EXPECT_EQ(string_to_selector(bulk_data, "A1"), s_A1);
  EXPECT_EQ(string_to_selector(bulk_data, "A_X"), s_A_X);
  EXPECT_EQ(string_to_selector(bulk_data, "A_"), s_A_);
  EXPECT_EQ(string_to_selector(bulk_data, "weird_yet_1_valid..."), s_weird_yet_valid);

  // Check basic operations (with and without spaces)
  EXPECT_EQ(string_to_selector(bulk_data, "A & B"), s_A & s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "A|B"), s_A | s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "A - B"), s_A - s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "A & !B"), s_A & !s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "A | !B"), s_A | !s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "A - !B"), s_A - !s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "(A & B)"), s_A & s_B);
  EXPECT_EQ(string_to_selector(bulk_data, "A & (B | C)"), s_A & (s_B | s_C));
  EXPECT_EQ(string_to_selector(bulk_data, "A & !(B | C)"), s_A & !(s_B | s_C));

  // Check that non-trivial part names don't interfere with the selector math
  EXPECT_EQ(string_to_selector(bulk_data, "A1& weird_yet_1_valid... & weird_yet_1_valid... & C & A_X & A_"),
            s_A1 & s_weird_yet_valid & s_weird_yet_valid & s_C & s_A_X & s_A_);

  // Try out some more complex operations
  EXPECT_EQ(string_to_selector(bulk_data, "(A & B) | C"), (s_A & s_B) | s_C);
  EXPECT_EQ(string_to_selector(bulk_data, "A & (B | C)"), s_A & (s_B | s_C));
  EXPECT_EQ(string_to_selector(bulk_data, "A & !(B | C)"), s_A & !(s_B | s_C));
  EXPECT_EQ(string_to_selector(bulk_data, "(A & !(B | C)) | (A1 & A_X)"), (s_A & !(s_B | s_C)) | (s_A1 & s_A_X));
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
