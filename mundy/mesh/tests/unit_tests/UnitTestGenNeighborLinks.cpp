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
#include <mundy_mesh/Aggregate.hpp>
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/GenNeighborLinkers.hpp>  // for mundy::mesh::GenNeighborLinkers
#include <mundy_mesh/LinkData.hpp>            // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>

namespace mundy {

namespace mesh {

namespace {

template <typename NgpSphereAgg>
class BoundingSphereGen {
 public:
  BoundingSphereGen(NgpSphereAgg agg) : ngp_sphere_agg_(agg) {
  }

  KOKKOS_INLINE_FUNCTION
  geom::Sphere<double> operator()(const stk::mesh::FastMeshIndex& sphere_index) const {
    auto sphere_view = ngp_sphere_agg_.get_view(sphere_index);
    auto center = get<CENTER>(sphere_view, 0);
    double radius = get<RADIUS>(sphere_view)[0];
    return geom::Sphere<double>(center, radius);
  }

 private:
  NgpSphereAgg ngp_sphere_agg_;
};

TEST(GenNeighborLinks, BasicUsage) {
  using stk::mesh::Entity;
  using stk::mesh::EntityId;
  using stk::mesh::EntityRank;
  using stk::mesh::FastMeshIndex;
  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::PartVector;
  using stk::topology::EDGE_RANK;
  using stk::topology::ELEM_RANK;
  using stk::topology::FACE_RANK;
  using stk::topology::NODE_RANK;

  // Setup
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = builder.create_meta_data();
  MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data(meta_data_ptr);
  BulkData& bulk_data = *bulk_data_ptr;

  // Setup the link meta data and link data
  EntityRank linker_rank = NODE_RANK;
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", linker_rank);
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);
  Part& sp_sp_links_part = link_meta_data.declare_link_part("SPHERE_SPHERE_LINKS", 2 /* our dimensionality */);
  Part& spheres_part = meta_data.declare_part_with_topology("SPHERES_PART", stk::topology::PARTICLE);
  Field<double>& node_coords_field = meta_data.declare_field<double>(NODE_RANK, "NODE_COORDS");
  Field<double>& elem_radius_field = meta_data.declare_field<double>(ELEM_RANK, "RADIUS");
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, spheres_part, 1, nullptr);
  meta_data.commit();

  // Declare two overlapping spheres
  bulk_data.modification_begin();
  Entity sphere1 = bulk_data.declare_element(1, stk::mesh::ConstPartVector{&spheres_part});
  Entity sphere2 = bulk_data.declare_element(2, stk::mesh::ConstPartVector{&spheres_part});
  Entity node1 = bulk_data.declare_node(1);
  Entity node2 = bulk_data.declare_node(2);
  bulk_data.declare_relation(sphere1, node1, 0);
  bulk_data.declare_relation(sphere1, node2, 1);
  bulk_data.modification_end();

  // Create a sphere aggregate
  auto coords_accessor = Vector3FieldComponent(node_coords_field);
  auto radius_accessor = ScalarFieldComponent(elem_radius_field);
  auto spheres = make_aggregate<stk::topology::PARTICLE>(bulk_data, spheres_part)
                     .add_component<CENTER, NODE_RANK>(coords_accessor)
                     .add_component<RADIUS, ELEM_RANK>(radius_accessor);

  // Create the generator
  auto bounding_sphere_gen = BoundingSphereGen(get_updated_ngp_aggregate(spheres));
  GenNeighborLinks generator(link_data, stk::ngp::HostExecSpace{});
  generator.set_source_target_rank(ELEM_RANK, ELEM_RANK)
      .set_enforce_source_target_symmetry(true)
      .set_search_buffer(0.0)
      .set_search_filter(make_search_filter(stk::ngp::HostExecSpace{}, search_filters::ExcludeSelfInteractions(ELEM_RANK, ELEM_RANK)))
      .acts_on(spheres_part, spheres_part, bounding_sphere_gen, bounding_sphere_gen,
               stk::mesh::PartVector{&sp_sp_links_part})
      .concretize();
  ASSERT_TRUE(generator.is_concretized());
  ASSERT_EQ(generator.get_source_rank(), ELEM_RANK);
  ASSERT_EQ(generator.get_target_rank(), ELEM_RANK);
  ASSERT_TRUE(generator.get_enforce_source_target_symmetry());

  // Perform the initial neighbor link generation
  generator.generate();

  // Check that the linkers were created
  std::vector<stk::mesh::Entity> linkers;
  stk::mesh::get_entities(bulk_data, linker_rank, sp_sp_links_part, linkers);
  ASSERT_EQ(linkers.size(), 1) << "Expected one linker to be created.";
  ASSERT_TRUE(bulk_data.is_valid(linkers[0])) << "Linker is invalid.";
  auto source = link_data.get_linked_entity(linkers[0], 0);
  auto target = link_data.get_linked_entity(linkers[0], 1);
  ASSERT_TRUE((source == sphere1 && target == sphere2) || (source == sphere2 && target == sphere1));
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
