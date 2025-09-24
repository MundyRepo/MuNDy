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
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace {

// Note, these are the same tests as stk's MeshBuilder except with our wrapper.

void verify_comm(stk::ParallelMachine expected_comm, const stk::mesh::BulkData& bulk_data) {
  int result_of_comm_compare = 0;
  MPI_Comm_compare(expected_comm, bulk_data.parallel(), &result_of_comm_compare);
  EXPECT_EQ(MPI_IDENT, result_of_comm_compare);

  EXPECT_EQ(stk::parallel_machine_size(expected_comm), bulk_data.parallel_size());
  EXPECT_EQ(stk::parallel_machine_rank(expected_comm), bulk_data.parallel_rank());
}

TEST(MeshBuilderTest, CreateMetaNoComm) {
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = MeshBuilder().create_meta_data();
  EXPECT_TRUE(nullptr != meta_data_ptr);
}

TEST(MeshBuilderTest, CreateBulkdataNoCommThrows) {
  EXPECT_ANY_THROW(MeshBuilder().create_bulk_data());
}

TEST(MeshBuilderTest, ConstructBuilderThenSetComm) {
  stk::ParallelMachine comm = MPI_COMM_WORLD;
  MeshBuilder builder;
  builder.set_communicator(comm);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_TRUE(nullptr != bulk_data_ptr);
  verify_comm(comm, *bulk_data_ptr);
}

TEST(MeshBuilderTest, CreateSimplestCommWorld) {
  stk::ParallelMachine comm = MPI_COMM_WORLD;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = MeshBuilder(comm).create_bulk_data();

  EXPECT_TRUE(nullptr != bulk_data_ptr);
  verify_comm(comm, *bulk_data_ptr);
}

TEST(MeshBuilderTest, CreateSimplestCommSelf) {
  stk::ParallelMachine comm = MPI_COMM_SELF;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = MeshBuilder(comm).create_bulk_data();

  EXPECT_TRUE(nullptr != bulk_data_ptr);
  verify_comm(comm, *bulk_data_ptr);
}

TEST(MeshBuilderTest, BulkDataAndMetaDataOutliveBuilder) {
  stk::ParallelMachine comm = MPI_COMM_WORLD;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = MeshBuilder(comm).create_bulk_data();

  EXPECT_TRUE(nullptr != bulk_data_ptr);
  verify_comm(comm, *bulk_data_ptr);

  const stk::mesh::MetaData& meta = bulk_data_ptr->mesh_meta_data();
  const stk::mesh::BulkData& meta_bulk_data = meta.mesh_bulk_data();
  EXPECT_EQ(&meta_bulk_data, bulk_data_ptr.get());
}

TEST(MeshBuilderTest, BulkDataAuraDefault) {
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = MeshBuilder(MPI_COMM_WORLD).create_bulk_data();

  EXPECT_TRUE(bulk_data_ptr->is_automatic_aura_on());
}

TEST(MeshBuilderTest, BulkDataAuraOn) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_auto_aura_option(stk::mesh::BulkData::AUTO_AURA);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_TRUE(bulk_data_ptr->is_automatic_aura_on());
}

TEST(MeshBuilderTest, BulkDataAuraOff) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_auto_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_FALSE(bulk_data_ptr->is_automatic_aura_on());
}

TEST(MeshBuilderTest, SetSpatialDimension) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned expected_spatial_dim = 2;
  builder.set_spatial_dimension(expected_spatial_dim);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(expected_spatial_dim, bulk_data_ptr->mesh_meta_data().spatial_dimension());
}

TEST(MeshBuilderTest, SpatialDimensionDefaultThenInitialize) {
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = MeshBuilder(MPI_COMM_WORLD).create_bulk_data();
  EXPECT_EQ(0u, bulk_data_ptr->mesh_meta_data().spatial_dimension());
  const unsigned expected_spatial_dim = 2;
  bulk_data_ptr->mesh_meta_data().initialize(expected_spatial_dim);

  EXPECT_EQ(expected_spatial_dim, bulk_data_ptr->mesh_meta_data().spatial_dimension());
}

TEST(MeshBuilderTest, SetEntityRankNamesWithoutSpatialDimension) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  EXPECT_ANY_THROW(builder.create_bulk_data());
}

TEST(MeshBuilderTest, SetSpatialDimensionZeroAndEmptyEntityRankNames) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned expected_spatial_dim = 0;
  builder.set_spatial_dimension(expected_spatial_dim);
  std::vector<std::string> expected_rank_names = {};
  builder.set_entity_rank_names(expected_rank_names);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(expected_spatial_dim, bulk_data_ptr->mesh_meta_data().spatial_dimension());
  EXPECT_EQ(expected_rank_names, bulk_data_ptr->mesh_meta_data().entity_rank_names());
}

TEST(MeshBuilderTest, SetSpatialDimensionAndEntityRankNames) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned expected_spatial_dim = 3;
  builder.set_spatial_dimension(expected_spatial_dim);
  std::vector<std::string> expected_rank_names = {"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"};
  builder.set_entity_rank_names(expected_rank_names);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(expected_spatial_dim, bulk_data_ptr->mesh_meta_data().spatial_dimension());
  EXPECT_EQ(expected_rank_names, bulk_data_ptr->mesh_meta_data().entity_rank_names());
}

TEST(MeshBuilderTest, BulkDataAddFmwkData) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_add_fmwk_data_flag(true);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

#ifdef SIERRA_MIGRATION
  EXPECT_TRUE(bulk_data_ptr->add_fmwk_data());
#else
  EXPECT_FALSE(bulk_data_ptr->add_fmwk_data());
#endif
}

TEST(MeshBuilderTest, BulkdataAddFmwkDataFalse) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_add_fmwk_data_flag(false);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_FALSE(bulk_data_ptr->add_fmwk_data());
}

TEST(MeshBuilderTest, DefaultBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(bulk_data_ptr->get_initial_bucket_capacity(), stk::mesh::get_default_initial_bucket_capacity());
  EXPECT_EQ(bulk_data_ptr->get_maximum_bucket_capacity(), stk::mesh::get_default_maximum_bucket_capacity());
}

TEST(MeshBuilderTest, SetInvalidBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_bucket_capacity(0);
  EXPECT_ANY_THROW(builder.create_bulk_data());
}

TEST(MeshBuilderTest, SetBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned set_bucket_capacity = 256;
  builder.set_bucket_capacity(set_bucket_capacity);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(bulk_data_ptr->get_initial_bucket_capacity(), set_bucket_capacity);
  EXPECT_EQ(bulk_data_ptr->get_maximum_bucket_capacity(), set_bucket_capacity);
}

TEST(MeshBuilderTest, SetInvalidInitialBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_initial_bucket_capacity(0);
  EXPECT_ANY_THROW(builder.create_bulk_data());
}

TEST(MeshBuilderTest, SetInitialBucketCapacityBiggerThanMaximum) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned initial_bucket_capacity = stk::mesh::get_default_maximum_bucket_capacity() * 2;
  builder.set_initial_bucket_capacity(initial_bucket_capacity);
  EXPECT_ANY_THROW(builder.create_bulk_data());
}

TEST(MeshBuilderTest, SetInitialBucketCapacitySmallerThanMaximum) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned initial_bucket_capacity = stk::mesh::get_default_maximum_bucket_capacity() / 2;
  builder.set_initial_bucket_capacity(initial_bucket_capacity);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(bulk_data_ptr->get_initial_bucket_capacity(), initial_bucket_capacity);
  EXPECT_EQ(bulk_data_ptr->get_maximum_bucket_capacity(), stk::mesh::get_default_maximum_bucket_capacity());
}

TEST(MeshBuilderTest, SetInvalidMaximumBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_maximum_bucket_capacity(0);
  EXPECT_ANY_THROW(builder.create_bulk_data());
}

TEST(MeshBuilderTest, SetMaximumBucketCapacityBiggerThanInitial) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned maximum_bucket_capacity = stk::mesh::get_default_initial_bucket_capacity() * 2;
  builder.set_maximum_bucket_capacity(maximum_bucket_capacity);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(bulk_data_ptr->get_initial_bucket_capacity(), stk::mesh::get_default_initial_bucket_capacity());
  EXPECT_EQ(bulk_data_ptr->get_maximum_bucket_capacity(), maximum_bucket_capacity);
}

TEST(MeshBuilderTest, SetMaximumBucketCapacitySmallerThanInitial) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned maximum_bucket_capacity = stk::mesh::get_default_initial_bucket_capacity() / 2;
  builder.set_maximum_bucket_capacity(maximum_bucket_capacity);
  EXPECT_ANY_THROW(builder.create_bulk_data());
}

TEST(MeshBuilderTest, SetIdenticalInitialAndMaximumBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned initial_bucket_capacity = 256;
  const unsigned maximum_bucket_capacity = 256;
  builder.set_initial_bucket_capacity(initial_bucket_capacity);
  builder.set_maximum_bucket_capacity(maximum_bucket_capacity);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(bulk_data_ptr->get_initial_bucket_capacity(), initial_bucket_capacity);
  EXPECT_EQ(bulk_data_ptr->get_maximum_bucket_capacity(), maximum_bucket_capacity);
}

TEST(MeshBuilderTest, SetDifferentInitialAndMaximumBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned initial_bucket_capacity = 32;
  const unsigned maximum_bucket_capacity = 256;
  builder.set_initial_bucket_capacity(initial_bucket_capacity);
  builder.set_maximum_bucket_capacity(maximum_bucket_capacity);
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();

  EXPECT_EQ(bulk_data_ptr->get_initial_bucket_capacity(), initial_bucket_capacity);
  EXPECT_EQ(bulk_data_ptr->get_maximum_bucket_capacity(), maximum_bucket_capacity);
}

TEST(MeshBuilderTest, SetFaultyInitialAndMaximumBucketCapacity) {
  MeshBuilder builder(MPI_COMM_WORLD);
  const unsigned initial_bucket_capacity = 64;
  const unsigned maximum_bucket_capacity = 32;
  builder.set_initial_bucket_capacity(initial_bucket_capacity);
  builder.set_maximum_bucket_capacity(maximum_bucket_capacity);
  EXPECT_ANY_THROW(builder.create_bulk_data());
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
