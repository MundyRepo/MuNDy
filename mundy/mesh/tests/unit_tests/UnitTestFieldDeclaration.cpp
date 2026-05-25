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
#include <gtest/gtest.h>

// C++ core
#include <string>

// STK mesh
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_topology/topology.hpp>

// Mundy libs
#include <mundy_mesh/DeclareField.hpp>
#include <mundy_mesh/MeshBuilder.hpp>

namespace mundy {

namespace mesh {

namespace {

TEST(UnitTestFieldDeclaration, DeclaresFieldsWhenSettersAreReorderedAroundType) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using Ioss::Field::TRANSIENT;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  FieldDeclarationHelper field_decl(meta_data);
  stk::mesh::Field<double>& field = field_decl.output_type(stk::io::FieldOutputType::VECTOR_3D)
                                        .name("DECLARED_FIELD")
                                        .type<double>()
                                        .role(TRANSIENT)
                                        .rank(stk::topology::ELEM_RANK)
                                        .declare();

  EXPECT_EQ(field.entity_rank(), stk::topology::ELEM_RANK);
  EXPECT_EQ(std::string(field.name()), "DECLARED_FIELD");
}

TEST(UnitTestFieldDeclaration, RequiresTypeRankAndNameBeforeDeclare) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  // Missing type: FieldDeclarationHelper has no .declare() — calling .type<T>() is required first (compile error).
  EXPECT_THROW((void)FieldDeclarationHelper(meta_data).type<double>().name("MISSING_RANK").declare(), std::logic_error);
  EXPECT_THROW(
      (void)FieldDeclarationHelper(meta_data).type<double>().rank(stk::topology::ELEM_RANK).declare(),  // Missing name
      std::logic_error);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
