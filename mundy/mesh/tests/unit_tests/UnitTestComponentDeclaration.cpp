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

// STK mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_topology/topology.hpp>

// Mundy libs
#include <mundy_math/Vector3.hpp>
#include <mundy_mesh/Aggregate.hpp>
#include <mundy_mesh/Component.hpp>
#include <mundy_mesh/DeclareComponent.hpp>
#include <mundy_mesh/DeclarePart.hpp>
#include <mundy_mesh/FieldComponent.hpp>
#include <mundy_mesh/FieldViews.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/SharedComponent.hpp>

namespace mundy {

namespace mesh {

namespace {

struct DECLARED_COORDS;
struct DECLARED_SPEED;
struct ORDERED_FIELD_1;
struct ORDERED_FIELD_2;
struct ORDERED_FIELD_3;
struct ORDERED_FIELD_4;
struct ORDERED_FIELD_5;

}  // namespace

namespace {

TEST(UnitTestComponentDeclaration, CanonicalUse) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using Ioss::Field::TRANSIENT;
  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  FieldDeclarationHelper field_decl(meta_data);
  ComponentDeclarationHelper component_decl;

  auto coords = field_decl.type<double>()
                    .role(TRANSIENT)
                    .rank(ELEM_RANK)
                    .name("DECLARED_COORDS")
                    .access<Vector3d>()
                    .tag<DECLARED_COORDS>()
                    .declare();

  auto speed = component_decl.shared(2.5).rank(ELEM_RANK).access<double>().tag<DECLARED_SPEED>().declare();

  PartDeclarationHelper part_decl(meta_data);
  stk::mesh::Part& particle_part =
      part_decl.name("DECLARED_PARTICLES").topology(stk::topology::PARTICLE).put_component(coords, nullptr).declare();

  auto bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  meta_data.commit();
  bulk_data.modification_begin();
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&particle_part});
  bulk_data.modification_end();

  vector3_field_data(coords.component().field(), elem1).set(1.0, 2.0, 3.0);
  coords.modify_on_host();

  auto agg = Aggregate(bulk_data, particle_part).add_component(coords).add_component(speed);

  EXPECT_EQ(agg.get_component<DECLARED_COORDS>().component().field().mesh_meta_data_ordinal(),
            coords.component().field().mesh_meta_data_ordinal());
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(coords.component().field(), elem1), 3u);

  auto coords_view = agg.get<DECLARED_COORDS>(elem1);
  auto speed_view = agg.get<DECLARED_SPEED>(elem1);

  EXPECT_DOUBLE_EQ(coords_view[0], 1.0);
  EXPECT_DOUBLE_EQ(coords_view[1], 2.0);
  EXPECT_DOUBLE_EQ(coords_view[2], 3.0);
  EXPECT_DOUBLE_EQ(speed_view[0], 2.5);
}

TEST(UnitTestComponentDeclaration, FieldComponentDeclarationIsInvariantToFluentCallOrder) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using Ioss::Field::TRANSIENT;
  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  FieldDeclarationHelper field_decl(meta_data);
  auto field1 = field_decl.type<double>()
                    .role(TRANSIENT)
                    .rank(ELEM_RANK)
                    .name("ORDERED_FIELD_1")
                    .access<Vector3d>()
                    .tag<ORDERED_FIELD_1>()
                    .declare();

  auto field2 = field_decl.tag<ORDERED_FIELD_2>()
                    .access<Vector3d>()
                    .name("ORDERED_FIELD_2")
                    .role(TRANSIENT)
                    .type<double>()
                    .rank(ELEM_RANK)
                    .declare();

  auto field3 = field_decl.name("ORDERED_FIELD_3")
                    .rank(ELEM_RANK)
                    .type<double>()
                    .tag<ORDERED_FIELD_3>()
                    .role(TRANSIENT)
                    .access<Vector3d>()
                    .declare();

  auto field4 = field_decl.access<Vector3d>()
                    .tag<ORDERED_FIELD_4>()
                    .rank(ELEM_RANK)
                    .role(TRANSIENT)
                    .name("ORDERED_FIELD_4")
                    .type<double>()
                    .declare();

  auto field5 = field_decl.rank(ELEM_RANK)
                    .name("ORDERED_FIELD_5")
                    .access<Vector3d>()
                    .type<double>()
                    .tag<ORDERED_FIELD_5>()
                    .role(TRANSIENT)
                    .declare();

  PartDeclarationHelper part_decl(meta_data);
  stk::mesh::Part& particle_part = part_decl.name("ORDERED_PARTICLES")
                                       .topology(stk::topology::PARTICLE)
                                       .put_component(field1, nullptr)
                                       .put_component(field2, nullptr)
                                       .put_component(field3, nullptr)
                                       .put_component(field4, nullptr)
                                       .put_component(field5, nullptr)
                                       .declare();

  auto bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  meta_data.commit();
  bulk_data.modification_begin();
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&particle_part});
  bulk_data.modification_end();

  EXPECT_EQ(field1.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field2.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field3.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field4.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field5.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field1.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field2.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field3.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field4.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field5.component().field(), elem1), 3u);

  vector3_field_data(field1.component().field(), elem1).set(1.0, 2.0, 3.0);
  vector3_field_data(field2.component().field(), elem1).set(4.0, 5.0, 6.0);
  vector3_field_data(field3.component().field(), elem1).set(7.0, 8.0, 9.0);
  vector3_field_data(field4.component().field(), elem1).set(10.0, 11.0, 12.0);
  vector3_field_data(field5.component().field(), elem1).set(13.0, 14.0, 15.0);

  auto field1_view = field1(elem1);
  auto field2_view = field2(elem1);
  auto field3_view = field3(elem1);
  auto field4_view = field4(elem1);
  auto field5_view = field5(elem1);

  EXPECT_DOUBLE_EQ(field1_view[0], 1.0);
  EXPECT_DOUBLE_EQ(field1_view[1], 2.0);
  EXPECT_DOUBLE_EQ(field1_view[2], 3.0);

  EXPECT_DOUBLE_EQ(field2_view[0], 4.0);
  EXPECT_DOUBLE_EQ(field2_view[1], 5.0);
  EXPECT_DOUBLE_EQ(field2_view[2], 6.0);

  EXPECT_DOUBLE_EQ(field3_view[0], 7.0);
  EXPECT_DOUBLE_EQ(field3_view[1], 8.0);
  EXPECT_DOUBLE_EQ(field3_view[2], 9.0);

  EXPECT_DOUBLE_EQ(field4_view[0], 10.0);
  EXPECT_DOUBLE_EQ(field4_view[1], 11.0);
  EXPECT_DOUBLE_EQ(field4_view[2], 12.0);

  EXPECT_DOUBLE_EQ(field5_view[0], 13.0);
  EXPECT_DOUBLE_EQ(field5_view[1], 14.0);
  EXPECT_DOUBLE_EQ(field5_view[2], 15.0);
}

TEST(UnitTestComponentDeclaration, SharedComponentDeclarationUsesCanonicalAccessByDefault) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  ComponentDeclarationHelper component_decl;
  auto speed = component_decl.shared(3.5).tag<DECLARED_SPEED>().rank(stk::topology::ELEM_RANK).declare();

  stk::mesh::Entity entity = stk::mesh::Entity();
  auto speed_view = speed(entity);
  EXPECT_DOUBLE_EQ(speed_view[0], 3.5);
}

TEST(UnitTestComponentDeclaration, FieldComponentDeclarationRejectsOutputTypeMismatch) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  FieldDeclarationHelper field_decl(meta_data);

  EXPECT_THROW((void)field_decl.type<double>()
                   .output_type(stk::io::FieldOutputType::SCALAR)
                   .rank(stk::topology::ELEM_RANK)
                   .name("BAD_VECTOR_OUTPUT")
                   .access<Vector3d>()
                   .declare(),
               std::invalid_argument);
}

TEST(UnitTestComponentDeclaration, ExpectedFailureModes) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  FieldDeclarationHelper field_decl(meta_data);
  ComponentDeclarationHelper component_decl;

  EXPECT_THROW((void)field_decl.type<double>().rank(ELEM_RANK).access<double>().declare(), std::logic_error);
  EXPECT_THROW((void)field_decl.type<double>().name("MISSING_FIELD_COMPONENT_RANK").access<double>().declare(),
               std::logic_error);
  EXPECT_THROW((void)field_decl.type<double>()
                   .rank(ELEM_RANK)
                   .name("MISSING_FIELD_COMPONENT_ACCESS")
                   .tag<ORDERED_FIELD_1>()
                   .declare(),
               std::logic_error);
  EXPECT_THROW((void)component_decl.shared(3.5).tag<DECLARED_SPEED>().declare(), std::logic_error);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
