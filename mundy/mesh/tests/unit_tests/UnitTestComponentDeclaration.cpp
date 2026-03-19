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
struct INFERRED_FORCE;

}  // namespace

template <>
struct component_tag_traits<DECLARED_COORDS> {
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;
};

template <>
struct component_tag_traits<DECLARED_SPEED> {
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;
};

template <>
struct component_tag_traits<INFERRED_FORCE> {
  static constexpr stk::topology::rank_t rank = stk::topology::ELEM_RANK;
};

namespace {

TEST(UnitTestComponentDeclaration, FieldAndSharedDeclarationsIntegrateWithAggregateAndParts) {
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

  auto agg = make_aggregate(bulk_data, particle_part).add_component(coords).add_component(speed);

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

TEST(UnitTestComponentDeclaration, FieldComponentDeclarationAllowsAccessBeforeTypeAndTagDrivenRank) {
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
  auto force = field_decl.access<Vector3d>().name("INFERRED_FORCE").type<double>().tag<INFERRED_FORCE>().declare();

  PartDeclarationHelper part_decl(meta_data);
  stk::mesh::Part& particle_part =
      part_decl.name("INFERRED_PARTICLES").topology(stk::topology::PARTICLE).put_component(force, nullptr).declare();

  auto bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  meta_data.commit();
  bulk_data.modification_begin();
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&particle_part});
  bulk_data.modification_end();

  EXPECT_EQ(force.component().field().entity_rank(), stk::topology::ELEM_RANK);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(force.component().field(), elem1), 3u);

  vector3_field_data(force.component().field(), elem1).set(4.0, 5.0, 6.0);
  auto force_view = force(elem1);
  EXPECT_DOUBLE_EQ(force_view[0], 4.0);
  EXPECT_DOUBLE_EQ(force_view[1], 5.0);
  EXPECT_DOUBLE_EQ(force_view[2], 6.0);
}

TEST(UnitTestComponentDeclaration, SharedComponentDeclarationUsesCanonicalAccessByDefault) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  ComponentDeclarationHelper component_decl;
  auto speed = component_decl.shared(3.5).tag<DECLARED_SPEED>().declare();

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

}  // namespace

}  // namespace mesh

}  // namespace mundy
