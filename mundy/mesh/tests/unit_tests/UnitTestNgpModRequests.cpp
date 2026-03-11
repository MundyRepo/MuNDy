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
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/NgpModRequests.hpp>  // for mundy::mesh::NgpModRequests
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>

namespace mundy {

namespace mesh {

namespace {

TEST(NgpModRequests, BasicUsage) {
  // Setup
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = builder.create_meta_data();
  MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data(meta_data_ptr);
  BulkData& bulk_data = *bulk_data_ptr;
  stk::mesh::Part& elem_part = meta_data.declare_part("PART_1", stk::topology::ELEM_RANK);
  stk::mesh::Part& particle_part = meta_data.declare_part_with_topology("PART_2", stk::topology::PARTICLE);
  meta_data.commit();

  // Declare two particles
  // bulk_data.modification_begin();
  // Entity sphere1 = bulk_data.declare_element(1, stk::mesh::ConstPartVector{&spheres_part});
  // Entity sphere2 = bulk_data.declare_element(2, stk::mesh::ConstPartVector{&spheres_part});
  // Entity node1 = bulk_data.declare_node(1);
  // Entity node2 = bulk_data.declare_node(2);
  // bulk_data.declare_relation(sphere1, node1, 0);
  // bulk_data.declare_relation(sphere1, node2, 1);
  // bulk_data.modification_end();

  // Run the test
  NgpModRequests reqs;



}

/*
Stuff to test:
  - TicketIssuer
    initialize(activate_device)
    activate_host()
    activate_device()
    reset()
    finalize_count()
    claim(N)
    claim()
  - NgpModRequests should be a "view"-like class (cheap to copy where modifying a copy modifies the underlying data)
  - request_entities_new_ids and request_entities_known_ids should memoize their return based on the sorted and uniqued set of input parts
  - 
*/



}  // namespace

}  // namespace mesh

}  // namespace mundy
