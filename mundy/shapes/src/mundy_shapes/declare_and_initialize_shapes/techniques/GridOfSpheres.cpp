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

/// \file GridOfSpheres.cpp
/// \brief Definition of DeclareAndInitShapes's GridOfSpheres technique.

// C++ core libs
#include <iostream>  // for std::cout, std::endl
#include <memory>    // for std::shared_ptr, std::unique_ptr
#include <string>    // for std::string
#include <vector>    // for std::vector

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_search/BoundingBox.hpp>     // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>    // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>    // for stk::search::KDTREE

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_shapes/Spheres.hpp>     // for mundy::shapes::Spheres
#include <mundy_shapes/declare_and_initialize_shapes/techniques/GridCoordinateMapping.hpp>  // for mundy::shapes::...::GridCoordinateMapping
#include <mundy_shapes/declare_and_initialize_shapes/techniques/GridOfSpheres.hpp>  // for mundy::shapes::...::GridOfSpheres

namespace mundy {

namespace shapes {

namespace declare_and_initialize_shapes {

namespace techniques {

namespace {

// \name Helper functions for shuffling code
//{

/// \brief Shuffle the input index to a new index in the range [0, max_size).
/// \details The shuffle function is a one-to-one mapping from the input index to the output index.
/// Tested for max_size up to 10,000,000. This is not a perfect shuffle, but it is good enough for our purposes.
///
/// \param input The input index to be shuffled.
/// \param max_size The maximum size of the index space.
size_t shuffle(size_t input, size_t max_size) {
  return input * 833 % max_size;
}

//}

// \name Helper functions for Morton code
//{

// From https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
// Method to separate bits from a given integer 3 positions apart
inline uint64_t splitBy3(uint32_t a) {
  uint64_t x = a & 0x1fffff;              // we only look at the first 21 bits
  x = (x | x << 32) & 0x1f00000000ffff;   // shift left 32 bits, OR with self, and
                                          // 00011111000000000000000000000000000000001111111111111111
  x = (x | x << 16) & 0x1f0000ff0000ff;   // shift left 32 bits, OR with self, and
                                          // 00011111000000000000000011111111000000000000000011111111
  x = (x | x << 8) & 0x100f00f00f00f00f;  // shift left 32 bits, OR with self, and
                                          // 0001000000001111000000001111000000001111000000001111000000000000
  x = (x | x << 4) & 0x10c30c30c30c30c3;  // shift left 32 bits, OR with self, and
                                          // 0001000011000011000011000011000011000011000011000011000100000000
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

inline uint64_t interleave_magic(uint32_t x, uint32_t y, uint32_t z) {
  uint64_t answer = 0;
  answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
  return answer;
}

inline uint64_t interleave_for(unsigned int x, unsigned int y, unsigned int z) {
  uint64_t answer = 0;
  for (uint64_t i = 0; i < (sizeof(uint64_t) * CHAR_BIT) / 3; ++i) {
    answer |= ((x & ((uint64_t)1 << i)) << 2 * i) | ((y & ((uint64_t)1 << i)) << (2 * i + 1)) |
              ((z & ((uint64_t)1 << i)) << (2 * i + 2));
  }
  return answer;
}

/// \brief Get the Morton code for the given i, j, k.
inline uint64_t get_morton_code(uint32_t i, uint32_t j, uint32_t k) {
  return interleave_for(i, j, k);
}
//}

}  // namespace

// \name Constructors and destructor
//{

GridOfSpheres::GridOfSpheres(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "GridOfSpheres: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(GridOfSpheres::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::Spheres::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);

  MUNDY_THROW_REQUIRE(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                      "GridOfSpheres: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                      "GridOfSpheres: radius_field_ptr cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  const Teuchos::Array<std::string> sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("sphere_part_names");
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(
          part != nullptr, std::invalid_argument,
          std::string("GridOfSpheres: Expected a part with name '") + part_name + "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };

  sphere_part_ptrs_ = parts_from_names(*meta_data_ptr_, sphere_part_names);
}
//}

// \name Setters
//{

void GridOfSpheres::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(GridOfSpheres::get_valid_mutable_params());

  num_spheres_x_ = valid_mutable_params.get<size_t>("num_spheres_x");
  num_spheres_y_ = valid_mutable_params.get<size_t>("num_spheres_y");
  num_spheres_z_ = valid_mutable_params.get<size_t>("num_spheres_z");
  coordinate_map_ptr_ = valid_mutable_params.get<std::shared_ptr<GridCoordinateMapping>>("coordinate_mapping");
  sphere_radius_lower_bound_ = valid_mutable_params.get<double>("sphere_radius_lower_bound");
  sphere_radius_upper_bound_ = valid_mutable_params.get<double>("sphere_radius_upper_bound");
  zmorton_ = valid_mutable_params.get<bool>("zmorton");
  shuffle_ = valid_mutable_params.get<bool>("shuffle");

  if (zmorton_) {
    build_zmorton_map();
  }
}
//}

// \name Getters
//{

stk::mesh::EntityId GridOfSpheres::node_id(size_t i, size_t j, size_t k) const {
  const size_t linearized_id =
      i * num_spheres_y_ * num_spheres_z_ + j * num_spheres_z_ + k;  // i slowest, j next, k fastest
  const size_t unshuffled_id = zmorton_ ? zmorton_map_[linearized_id] : linearized_id;
  return shuffle_ ? shuffle(unshuffled_id, num_spheres_x_ * num_spheres_y_ * num_spheres_z_) + sphere_node_id_start_
                  : unshuffled_id + sphere_node_id_start_;
}

stk::mesh::EntityId GridOfSpheres::element_id(size_t i, size_t j, size_t k) const {
  // The entity id of the spheres is the same as that of their connected nodes minus their node ID shift.
  return node_id(i, j, k) - sphere_node_id_start_ + sphere_element_id_start_;
}

stk::mesh::Entity GridOfSpheres::node(size_t i, size_t j, size_t k) const {
  return bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, node_id(i, j, k));
}

stk::mesh::Entity GridOfSpheres::element(size_t i, size_t j, size_t k) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, element_id(i, j, k));
}
//}

// \name Actions
//{

void GridOfSpheres::build_zmorton_map() {
  // The ZMorton codes for an arbitrary 3D grid are not, in general, sequentially ordered.
  // We need to loop over our entire grid and build a map from the linear index defined by
  // linear_index = i * num_spheres_y_ * num_spheres_z_ + j * num_spheres_z_ + k;  // i slowest, j next, k fastest
  // to the sequential ZMorton code. We can then use this map to create the spheres in the correct order.
  //
  // This map is just a vector of size num_spheres_x_ * num_spheres_y_ * num_spheres_z_ where the index is the linear
  // index and the value is the sequential ZMorton code.
  //
  // To create this map, we first create a map from linear_index to ZMorton code. This code is not sequential
  // but it contains the correct sorting property. We then create a sequentially ordered vector using iota and sort it
  // using the sort of the ZMorton code. This vector acts as a map from linear index to the index of the ZMordon codes
  // in sorted order, allowing us to replace the ZMorton codes in the ZMorton map with their sequential versions.

  // Create the map from linearized linear_index to ZMorton code.
  // This map is, unfortunitely, of the same size as the number of GLOBAL spheres, a potentially massive number.
  // TODO(palmerb4): Find a way to generate the Zmorton sequential sequence without needing to store the entire map.
  zmorton_map_.resize(num_spheres_x_ * num_spheres_y_ * num_spheres_z_);
  for (size_t i = 0; i < num_spheres_x_; ++i) {
    for (size_t j = 0; j < num_spheres_y_; ++j) {
      for (size_t k = 0; k < num_spheres_z_; ++k) {
        const size_t linear_index =
            i * num_spheres_y_ * num_spheres_z_ + j * num_spheres_z_ + k;  // i slowest, j next, k fastest
        const size_t morton_code = get_morton_code(static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                                                   static_cast<uint32_t>(k));  // Note, the current morton code requires
                                                                               // that i, j, and k are unsigned int or
                                                                               // uint32_t. This limits us to ints
                                                                               // between 0 and 4,294,967,295.
        zmorton_map_[linear_index] = morton_code;
      }
    }
  }

  // Create the sequentially ordered map using iota and sort it using the sort of the ZMorton code.
  std::vector<size_t> sequential_map(num_spheres_x_ * num_spheres_y_ * num_spheres_z_);
  std::iota(sequential_map.begin(), sequential_map.end(), 0);

  std::sort(sequential_map.begin(), sequential_map.end(),
            [this](size_t a, size_t b) { return zmorton_map_[a] < zmorton_map_[b]; });

  // Replace the ZMorton codes in the ZMorton map with their sequential versions.
  for (size_t i = 0; i < num_spheres_x_ * num_spheres_y_ * num_spheres_z_; ++i) {
    zmorton_map_[sequential_map[i]] = i;
  }
}

void GridOfSpheres::execute() {
  // Create the spheres and their connected nodes, distributing the work across the ranks.
  const size_t rank = bulk_data_ptr_->parallel_rank();
  const size_t columns_per_rank = num_spheres_x_ / bulk_data_ptr_->parallel_size();
  const size_t remainder = num_spheres_x_ % bulk_data_ptr_->parallel_size();
  const size_t start_column = rank * columns_per_rank + std::min(rank, remainder);
  const size_t end_column = start_column + columns_per_rank + (rank < remainder ? 1 : 0);
  bulk_data_ptr_->modification_begin();
  openrand::Philox rng(1, 0);
  for (size_t i = start_column; i < end_column; ++i) {
    for (size_t j = 0; j < num_spheres_y_; ++j) {
      for (size_t k = 0; k < num_spheres_z_; ++k) {
        // Create the sphere.
        stk::mesh::EntityId our_sphere_id = element_id(i, j, k);
        stk::mesh::Entity sphere = bulk_data_ptr_->declare_element(our_sphere_id);
        bulk_data_ptr_->change_entity_parts(sphere, sphere_part_ptrs_);

        // Create the node and connect it to the sphere.
        stk::mesh::EntityId our_node_id = node_id(i, j, k);
        stk::mesh::Entity node = bulk_data_ptr_->declare_node(our_node_id);
        bulk_data_ptr_->declare_relation(sphere, node, 0);

        // Set the node's coordinates using the given coordinate map.
        double *const node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node);
        const auto [coord_x, coord_y, coord_z] = coordinate_map_ptr_->get_grid_coordinate({i, j, k});
        node_coords[0] = coord_x;
        node_coords[1] = coord_y;
        node_coords[2] = coord_z;

        // Set the sphere's radius.
        double *const sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere);
        sphere_radius[0] =
            rng.rand<double>() * (sphere_radius_upper_bound_ - sphere_radius_lower_bound_) + sphere_radius_lower_bound_;
      }
    }
  }
  bulk_data_ptr_->modification_end();
}
//}

}  // namespace techniques

}  // namespace declare_and_initialize_shapes

}  // namespace shapes

}  // namespace mundy
