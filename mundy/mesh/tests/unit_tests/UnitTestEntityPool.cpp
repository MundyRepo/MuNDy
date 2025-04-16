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
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_core/NgpView.hpp>        // for mundy::core::NgpView
#include <mundy_mesh/NgpEntityPool.hpp>  // for mundy::mesh::NgpEntityPool

namespace mundy {

namespace mesh {

namespace {

void basic_usage_test() {
  // Setup
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity node1 = bulk_data.declare_node(1);
  stk::mesh::Entity node2 = bulk_data.declare_node(2);
  stk::mesh::Entity node3 = bulk_data.declare_node(3);
  bulk_data.modification_end();

  ///////////////////////////////
  // Declare and reserve pools //
  ///////////////////////////////
  // Reserve does not change the size of the pool, only the capacity.
  NgpEntityPool node_pool(bulk_data, stk::topology::NODE_RANK);       // Size 0. Capacity 0.
  NgpEntityPool elem_pool(bulk_data, stk::topology::ELEM_RANK, 100);  // Size 0. Capacity 100.
  node_pool.reserve(100);                                             // Size 0. Capacity 100.
  EXPECT_EQ(node_pool.size(), 0);
  EXPECT_EQ(node_pool.capacity(), 100);
  EXPECT_EQ(elem_pool.size(), 0);
  EXPECT_EQ(elem_pool.capacity(), 100);

  // Reserving less than the current capacity does nothing
  node_pool.reserve(50);  // Size 0. Capacity 100.
  EXPECT_EQ(node_pool.size(), 0);
  EXPECT_EQ(node_pool.capacity(), 100);

  //////////////////////////////
  // Add entities to the pool //
  //////////////////////////////

  // To populate the pool with entities, you can
  //  a. add entities to the pool one at a time via add(Entity) or add_host(Entity)
  node_pool.add(node1);  // Size 1. Capacity 100.
  EXPECT_EQ(node_pool.size(), 1);
  EXPECT_EQ(node_pool.capacity(), 100);

  //  b. add entities to the pool in bulk via add(NgpView<Entity*>) or add_host(std::vector<Entity>)
  node_pool.batch_add_host(stk::mesh::EntityVector{node2, node3});  // Size 3. Capacity 100.
  EXPECT_EQ(node_pool.size(), 3);
  EXPECT_EQ(node_pool.capacity(), 100);

  //  c. request that the pool declare N new entities via reserve_and_declare(N)
  //       For this to work, we must be in a modification cycle and this function be called parallel synchronously.
  bulk_data.modification_begin();
  node_pool.reserve_and_declare(50);  // Size 50. Capacity 100.
  EXPECT_EQ(node_pool.size(), 50);
  EXPECT_EQ(node_pool.capacity(), 100);
  bulk_data.modification_end();

  ////////////////////////////////////
  // Acquire entities from the pool //
  ////////////////////////////////////

  // Getting entities from the pool is similar to adding them except it removes them from the pool.
  // The order nodes are fetched is not guaranteed to be the same as the order they were added.
  //
  //  a. acquire entities from the pool one at a time via acquire() or acquire_host()
  stk::mesh::Entity node = node_pool.acquire();  // Size 49. Capacity 100.
  EXPECT_EQ(node_pool.size(), 49);
  EXPECT_EQ(node_pool.capacity(), 100);

  //  b. acquire entities from the pool in bulk via acquire(N) or acquire_host(N)
  auto ten_nodes = node_pool.batch_acquire(10);  // Size 39. Capacity 100.
  EXPECT_EQ(node_pool.size(), 39);
  EXPECT_EQ(node_pool.capacity(), 100);
  stk::mesh::EntityVector ten_nodes_vec = node_pool.batch_acquire_host(10);  // Size 29. Capacity 100.
  EXPECT_EQ(node_pool.size(), 29);

  // Attempting to over-draw or over-flow the pool will throw an exception ONLY in debug mode.
  stk::mesh::EntityVector too_many_entities(102);
  for (size_t i = 0; i < 102; ++i) {
    too_many_entities[i] = stk::mesh::Entity();
  }
#ifndef NDEBUG                                                   // In debug mode
  EXPECT_THROW(node_pool.batch_acquire(34), std::runtime_error)  // Size -5. Capacity 100?!
      << "Overfetching from the pool should throw in debug.";
  EXPECT_THROW(elem_pool.batch_add_host(too_many_entities), std::runtime_error)  // Size 102. Capacity 100?!
      << "Overfilling the pool should throw in debug";
#else  // Not in debug mode
  EXPECT_NO_THROW(node_pool.batch_acquire(34))  // Size -5. Capacity 100?!
      << "Overfetching from the pool should not throw in release. We are now in an undefined state.";
  EXPECT_NO_THROW(elem_pool.batch_add_host(too_many_entities))  // Size 102. Capacity 100?!
      << "Overfilling the pool should not throw in release";
#endif
}

TEST(UnitTestEntityPool, BasicUsage) {
  basic_usage_test();
}

void thread_safety_test() {
  // Setup
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;
  meta_data.commit();

  // Use enough entities for parallel contention to be possible
  size_t num_entities = 100000;
  bulk_data.modification_begin();
  core::NgpView<stk::mesh::Entity*> nodes("nodes", num_entities);
  for (size_t i = 0; i < num_entities; ++i) {
    nodes.view_host()(i) = bulk_data.declare_node(i + 1);  // IDs are 1 indexed
  }
  nodes.modify_on_host();
  nodes.sync_to_device();
  bulk_data.modification_end();

  NgpEntityPool node_pool(bulk_data, stk::topology::NODE_RANK, num_entities);
  EXPECT_EQ(node_pool.size(), 0);

  // Add entities to the pool in parallel (it's better to use a batch add, but sometimes using single adds is necessary)
  // Note, we use a lambda to avoid capturing the bulk_data, which doesn't have a copy constructor
  auto perform_add = [&node_pool, &nodes, num_entities]() {
    Kokkos::parallel_for(
        "UnitTestEntityPool:ThreadSafety", num_entities, KOKKOS_LAMBDA(const size_t i) {
          stk::mesh::Entity node = nodes.view_device()(i);
          node_pool.add(node);
        });
    node_pool.modify_on_device();
    EXPECT_EQ(node_pool.size(), num_entities);
    EXPECT_EQ(node_pool.capacity(), num_entities);

// Notice that we modified on the device, but we never synced to the host. This is because we didn't really need to.
// We can perform a sync and see that the host is updated correctly.
#ifdef STK_ENABLE_GPU
    EXPECT_EQ(node_pool.size_host(), 0) << "If GPU is enabled then the host and device are different.";
#else
    EXPECT_EQ(node_pool.size_host(), num_entities) << "If GPU is not enabled then the host and device are the same.";
#endif
    node_pool.sync_to_host();
    EXPECT_EQ(node_pool.size_host(), num_entities);
  };
  perform_add();

  // Fetch each of the entities from the pool in parallel and assert that they are not default constructed
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  auto perform_fetch = [&node_pool, &ngp_mesh, num_entities]() {
    core::NgpView<bool*> node_exists("node_exists", num_entities);
    Kokkos::deep_copy(node_exists.view_device(), false);

    Kokkos::parallel_for(
        "UnitTestEntityPool:ThreadSafety", num_entities, KOKKOS_LAMBDA(const size_t i) {
          stk::mesh::Entity node = node_pool.acquire();
          ASSERT_TRUE(node != stk::mesh::Entity());

          stk::mesh::EntityId id = ngp_mesh.identifier(node);
          node_exists.view_device()(id - 1) = true;
        });
    node_pool.modify_on_device();
    EXPECT_EQ(node_pool.size(), 0);
    EXPECT_EQ(node_pool.capacity(), num_entities);

    // Sum the node_exists array to ensure that all nodes were acquired
    size_t sum = 0;
    Kokkos::parallel_reduce(
        "UnitTestEntityPool:ThreadSafety reduce", num_entities,
        KOKKOS_LAMBDA(const size_t i, size_t& lsum) { lsum += node_exists.view_device()(i); }, sum);
    EXPECT_EQ(sum, num_entities) << "If this fails, we have a race condition for inserting the entities into the pool.";

#ifdef STK_ENABLE_GPU
    EXPECT_EQ(node_pool.size_host(), num_entities) << "If GPU is enabled then the host and device are different.";
#else
    EXPECT_EQ(node_pool.size_host(), 0) << "If GPU is not enabled then the host and device are the same.";
#endif
    node_pool.sync_to_host();
    EXPECT_EQ(node_pool.size_host(), 0);
  };
  perform_fetch();
}

TEST(UnitTestEntityPool, ThreadSafety) {
  if (stk::ngp::ExecSpace::concurrency() == 1) {
    GTEST_SKIP() << "Test is only valid when multiple threads are available. Otherwise, it could be a false positive.";
  }

  thread_safety_test();
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
