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

// STK mesh
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_io/FillMesh.hpp>         // for stk::io::fill_mesh_with_auto_decomp
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/FieldDataManager.hpp>
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/NgpField.hpp>  // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>   // for stk::mesh::NgpMesh
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::FastMeshIndex
#include <stk_topology/topology.hpp>

// Mundy libs
#include <mundy_mesh/Aggregate.hpp>        // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/NgpAccessorExpr.hpp>  // for accessor expressions
#include <mundy_mesh/NgpFieldBLAS.hpp>     // for mundy::mesh::field_axpby, etc.

namespace mundy {

namespace mesh {

namespace {

/*
Shared setup:
 - Create 3 fields (x, y, z, scratch1, scratch2) for each accessor type (scalar, vector3, matrix3, quaternion)
 - Create 3 additional fields (f1, f2, f3) of scalar type
 - Create 3 parts (block1, block2, block3) each with 1/3 of the entities
 - All parts should contain the (x, y, z, scratch1, scratch2) fields for all accessor types
 - Part 1 should contain f1, part 2 should contain f2, part 3 should contain f3 to create heterogeneity
 - Randomize the data in each field
*/

template <typename T>
void check_field_data_on_host(const std::string& message_to_throw, const stk::mesh::BulkData& stk_mesh,
                              const stk::mesh::FieldBase& stk_field, const stk::mesh::Selector& selector,
                              T expected_value, int component = -1, T component_value = 0) {
  stk_field.sync_to_host();

  stk::mesh::for_each_entity_run(
      stk_mesh, stk_field.entity_rank(), selector,
      [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
        const int num_components = stk::mesh::field_scalars_per_entity(stk_field, entity);
        const T* raw_field_data = reinterpret_cast<const T*>(stk::mesh::field_data(stk_field, entity));
        for (int i = 0; i < num_components; ++i) {
          if (i == component) {
            EXPECT_DOUBLE_EQ(raw_field_data[i], component_value) << "; i==" << i << ", component==" << component << "\n"
                                                                 << message_to_throw;
          } else {
            EXPECT_DOUBLE_EQ(raw_field_data[i], expected_value)
                << "; i==" << i << ", entity=" << bulk.entity_key(entity) << "\n"
                << message_to_throw;
          }
        }
      });
}

inline void set_field_data_on_host(const stk::mesh::BulkData& stk_mesh, const stk::mesh::FieldBase& stk_field,
                                   const stk::mesh::Selector& selector,
                                   std::function<std::vector<double>(const double*)> func) {
  const stk::mesh::FieldBase& coord_field = *stk_mesh.mesh_meta_data().coordinate_field();

  stk_field.clear_host_sync_state();
  coord_field.sync_to_host();

  stk::mesh::for_each_entity_run(
      stk_mesh, stk_field.entity_rank(), selector,
      [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
        double* entity_coords = static_cast<double*>(stk::mesh::field_data(coord_field, entity));
        auto expected_values = func(entity_coords);
        const int num_components = stk::mesh::field_scalars_per_entity(stk_field, entity);
        double* raw_field_data = static_cast<double*>(stk::mesh::field_data(stk_field, entity));
        for (int i = 0; i < num_components; ++i) {
          raw_field_data[i] = expected_values[i];
        }
      });

  stk_field.modify_on_host();
}

template <size_t NumComponents>
inline void check_field_data_on_host_func(const std::string& message_to_throw, const stk::mesh::BulkData& stk_mesh,
                                          const stk::mesh::FieldBase& stk_field, const stk::mesh::Selector& selector,
                                          const std::vector<const stk::mesh::FieldBase*>& other_fields,
                                          std::function<std::vector<double>(const double*)> func) {
  const stk::mesh::FieldBase& coord_field = *stk_mesh.mesh_meta_data().coordinate_field();

  stk_field.sync_to_host();
  coord_field.sync_to_host();

  stk::mesh::for_each_entity_run(
      stk_mesh, stk_field.entity_rank(), selector,
      [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
        double* entity_coords = static_cast<double*>(stk::mesh::field_data(coord_field, entity));
        auto expected_values = func(entity_coords);
        const double* raw_field_data = reinterpret_cast<const double*>(stk::mesh::field_data(stk_field, entity));
        for (unsigned int i = 0; i < NumComponents; ++i) {
          ASSERT_NEAR(raw_field_data[i], expected_values[i], 1e-12) << message_to_throw;
        }
      });
}

void randomize_coordinates(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& node_coord_field,
                           const unsigned spatial_dimension) {
  // Dereference values in *this
  const stk::mesh::Selector universal = bulk_data.mesh_meta_data().universal_part();
  node_coord_field.clear_host_sync_state();
  stk::mesh::for_each_entity_run(bulk_data, stk::topology::NODE_RANK, universal,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   double* raw_field_data =
                                       static_cast<double*>(stk::mesh::field_data(node_coord_field, entity));
                                   for (unsigned int i = 0; i < spatial_dimension; ++i) {
                                     raw_field_data[i] = static_cast<double>(rand()) / RAND_MAX;
                                   }
                                 });
  node_coord_field.modify_on_host();
}

class UnitTestAccessorExprFixture : public ::testing::Test {
 public:
  using DoubleField = stk::mesh::Field<double>;
  using CoordinateFunc = std::function<std::vector<double>(const double*)>;
  static constexpr double initial_value[9] = {-1, 2, -0.3, 4, -5, 6, -7, 8, -9};

  UnitTestAccessorExprFixture()
      : communicator_(MPI_COMM_WORLD),
        spatial_dimension_(3),
        entity_rank_names_({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"}) {
  }

  virtual ~UnitTestAccessorExprFixture() {
    reset_mesh();
  }

  void reset_mesh() {
    bulk_data_ptr_.reset();
    meta_data_ptr_.reset();
  }

  virtual stk::mesh::BulkData& get_bulk() {
    EXPECT_NE(bulk_data_ptr_, nullptr) << "Trying to get bulk data before it has been initialized.";
    return *bulk_data_ptr_;
  }

  DoubleField* create_field_on_parts(const std::string& field_name, const stk::mesh::EntityRank& entity_rank,
                                     const int& num_components, const stk::mesh::PartVector& parts) {
    DoubleField& field = meta_data_ptr_->declare_field<double>(entity_rank, field_name);
    for (stk::mesh::Part* part : parts) {
      stk::mesh::put_field_on_mesh(field, *part, num_components, initial_value);
    }
    return &field;
  }

  CoordinateFunc get_field_x_func() const {
    return [](const double* coords) {
      return std::vector<double>{coords[0] * coords[1],      3 * coords[1] * coords[2],  5 * coords[2] * coords[0],
                                 7 * coords[0] * coords[0],  11 * coords[1] * coords[1], 13 * coords[2] * coords[2],
                                 17 * coords[0] * coords[1], 19 * coords[1] * coords[2], 23 * coords[2] * coords[0]};
    };
  }

  CoordinateFunc get_field_y_func() const {
    return [](const double* coords) {
      return std::vector<double>{-coords[0] * coords[0],
                                 2 * coords[1],
                                 -3 * coords[2],
                                 -4 * coords[0] * coords[0] * coords[1],
                                 5 * coords[1] * coords[1],
                                 -6 * coords[2] * coords[2],  //
                                 -7 * coords[0] * coords[0] * coords[2],
                                 8 * coords[1] * coords[1] * coords[2],
                                 -9 * coords[2] * coords[2] * coords[0]};
    };
  }

  CoordinateFunc get_field_z_func() const {
    return [](const double* coords) {
      return std::vector<double>{23 * coords[0] + 29 * coords[1],
                                 31 * coords[2],
                                 2 * coords[0] * coords[1] * coords[2],
                                 37 * coords[0] * coords[0] + 41 * coords[1] * coords[1],
                                 43 * coords[2] * coords[2],
                                 5 * coords[0] * coords[1] * coords[2],
                                 47 * coords[0] * coords[0] + 53 * coords[1] * coords[1],
                                 59 * coords[2] * coords[2],
                                 7 * coords[0] * coords[1] * coords[2]};
    };
  }

  void reset_field_values() {
    randomize_coordinates(*bulk_data_ptr_, *node_coord_field_ptr_, spatial_dimension_);

    stk::mesh::Selector all_blocks = block1_selector_ | block2_selector_ | block3_selector_;
    set_field_data_on_host(*bulk_data_ptr_, *field_x_ptr_, all_blocks, get_field_x_func());
    set_field_data_on_host(*bulk_data_ptr_, *field_y_ptr_, all_blocks, get_field_y_func());
    set_field_data_on_host(*bulk_data_ptr_, *field_z_ptr_, all_blocks, get_field_z_func());
    set_field_data_on_host(*bulk_data_ptr_, *field_xs_ptr_, all_blocks, get_field_x_func());
    set_field_data_on_host(*bulk_data_ptr_, *field_ys_ptr_, all_blocks, get_field_y_func());
    set_field_data_on_host(*bulk_data_ptr_, *field_zs_ptr_, all_blocks, get_field_z_func());
  }

  void validate_initial_five_hex_mesh() {
    const stk::mesh::Entity hex1 = get_bulk().get_entity(stk::topology::ELEMENT_RANK, 1);
    const stk::mesh::Entity hex2 = get_bulk().get_entity(stk::topology::ELEMENT_RANK, 2);
    const stk::mesh::Entity hex3 = get_bulk().get_entity(stk::topology::ELEMENT_RANK, 3);
    const stk::mesh::Entity hex4 = get_bulk().get_entity(stk::topology::ELEMENT_RANK, 4);
    const stk::mesh::Entity hex5 = get_bulk().get_entity(stk::topology::ELEMENT_RANK, 5);

    // Check that the hexes are valid
    EXPECT_TRUE(get_bulk().is_valid(hex1));
    EXPECT_TRUE(get_bulk().is_valid(hex2));
    EXPECT_TRUE(get_bulk().is_valid(hex3));
    EXPECT_TRUE(get_bulk().is_valid(hex4));
    EXPECT_TRUE(get_bulk().is_valid(hex5));

    // Check that the hexes are in the correct blocks
    EXPECT_TRUE(get_bulk().bucket(hex1).member(*block1_part_ptr_));
    EXPECT_TRUE(get_bulk().bucket(hex2).member(*block1_part_ptr_));
    EXPECT_TRUE(get_bulk().bucket(hex3).member(*block2_part_ptr_));
    EXPECT_TRUE(get_bulk().bucket(hex4).member(*block2_part_ptr_));
    EXPECT_TRUE(get_bulk().bucket(hex5).member(*block3_part_ptr_));

    // Check that the hexes connect to the correct nodes
    const std::vector<int> hex1_node_ids = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int> hex2_node_ids = {5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<int> hex3_node_ids = {9, 13, 14, 15, 16, 17, 18, 19};
    const std::vector<int> hex4_node_ids = {9, 20, 21, 22, 23, 24, 25, 26};
    const std::vector<int> hex5_node_ids = {9, 27, 28, 29, 30, 31, 32, 33};

    auto check_hex_node_connectivity = [&](const stk::mesh::Entity hex, const std::vector<int>& node_ids) {
      const stk::mesh::Entity* hex_nodes = get_bulk().begin_nodes(hex);
      for (unsigned int i = 0; i < node_ids.size(); ++i) {
        EXPECT_EQ(get_bulk().identifier(hex_nodes[i]), node_ids[i]);
      }
    };

    check_hex_node_connectivity(hex1, hex1_node_ids);
    check_hex_node_connectivity(hex2, hex2_node_ids);
    check_hex_node_connectivity(hex3, hex3_node_ids);
    check_hex_node_connectivity(hex4, hex4_node_ids);
    check_hex_node_connectivity(hex5, hex5_node_ids);

    // Check that the nodes have inherited part membership
    auto check_hex_inherited_part_membership = [&](const stk::mesh::Entity hex, const stk::mesh::Part& part) {
      const stk::mesh::Entity* hex_nodes = get_bulk().begin_nodes(hex);
      for (unsigned int i = 0; i < 8; ++i) {
        const stk::mesh::Entity node = hex_nodes[i];
        EXPECT_TRUE(get_bulk().bucket(node).member(part));
      }
    };

    check_hex_inherited_part_membership(hex1, *block1_part_ptr_);
    check_hex_inherited_part_membership(hex2, *block1_part_ptr_);
    check_hex_inherited_part_membership(hex3, *block2_part_ptr_);
    check_hex_inherited_part_membership(hex4, *block2_part_ptr_);
    check_hex_inherited_part_membership(hex5, *block3_part_ptr_);
  }

  void declare_five_hexes() {
    const int parallel_size = get_bulk().parallel_size();
    ASSERT_TRUE(parallel_size == 1 || parallel_size == 2) << "This test is only designed to run with 1 or 2 MPI ranks.";
    std::string mesh_desc;
    if (parallel_size == 1) {
      mesh_desc =
          "textmesh:"
          "0,1,HEX_8,1,2,3,4,5,6,7,8,block_1\n"
          "0,2,HEX_8,5,6,7,8,9,10,11,12,block_1\n"
          "0,3,HEX_8,9,13,14,15,16,17,18,19,block_2\n"
          "0,4,HEX_8,9,20,21,22,23,24,25,26,block_2\n"
          "0,5,HEX_8,9,27,28,29,30,31,32,33,block_3";
    } else {
      mesh_desc =
          "textmesh:"
          "0,1,HEX_8,1,2,3,4,5,6,7,8,block_1\n"
          "1,2,HEX_8,5,6,7,8,9,10,11,12,block_1\n"
          "0,3,HEX_8,9,13,14,15,16,17,18,19,block_2\n"
          "1,4,HEX_8,9,20,21,22,23,24,25,26,block_2\n"
          "0,5,HEX_8,9,27,28,29,30,31,32,33,block_3";
    }

    stk::io::fill_mesh_with_auto_decomp(mesh_desc, *bulk_data_ptr_);
    validate_initial_five_hex_mesh();
  }

  void declare_N_hexes_per_dimension(const size_t num_hexes_per_dim) {
    const std::string mesh_desc = "generated:" + std::to_string(num_hexes_per_dim) + "x" +
                                  std::to_string(num_hexes_per_dim) + "x" + std::to_string(num_hexes_per_dim);
    stk::io::fill_mesh(mesh_desc, *bulk_data_ptr_);


    // All of the hexes start in block_1. Move a third of them to block_2 and a third of them to block_3, removing them
    // from block_1.
    bulk_data_ptr_->modification_begin();
    stk::mesh::EntityVector entities_to_move_to_block_2;
    stk::mesh::EntityVector entities_to_move_to_block_3;

    const stk::mesh::BucketVector& buckets =
        bulk_data_ptr_->get_buckets(stk::topology::ELEM_RANK, *block1_part_ptr_ &
        meta_data_ptr_->locally_owned_part());
    for (size_t bucket_count = 0, bucket_end = buckets.size(); bucket_count < bucket_end; ++bucket_count) {
      stk::mesh::Bucket& bucket = *buckets[bucket_count];
      for (size_t elem_count = 0, elem_end = bucket.size(); elem_count < elem_end; ++elem_count) {
        stk::mesh::Entity elem = bucket[elem_count];
        MUNDY_THROW_REQUIRE(bulk_data_ptr_->is_valid(elem), std::runtime_error, "Attempted to move an invalid entity."); 
        if (elem_count % 3 == 0) {
          entities_to_move_to_block_2.push_back(elem);
        } else if (elem_count % 3 == 1) {
          entities_to_move_to_block_3.push_back(elem);
        }
      }
    }

    bulk_data_ptr_->change_entity_parts(entities_to_move_to_block_2, stk::mesh::ConstPartVector{block2_part_ptr_},
                                        stk::mesh::ConstPartVector{block1_part_ptr_});
    bulk_data_ptr_->change_entity_parts(entities_to_move_to_block_3, stk::mesh::ConstPartVector{block3_part_ptr_},
                                        stk::mesh::ConstPartVector{block1_part_ptr_});
    bulk_data_ptr_->modification_end();
  }

  void setup_hex_mesh(const stk::mesh::EntityRank& entity_rank, stk::mesh::BulkData::AutomaticAuraOption aura_option,
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
                      std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager,
#else
                      stk::mesh::FieldDataManager* field_data_manager,
#endif
                      unsigned initial_bucket_capacity = stk::mesh::get_default_initial_bucket_capacity(),
                      unsigned maximum_bucket_capacity = stk::mesh::get_default_maximum_bucket_capacity()) {
    stk::mesh::MeshBuilder builder(communicator_);
    builder.set_spatial_dimension(spatial_dimension_);
    builder.set_entity_rank_names(entity_rank_names_);
    builder.set_aura_option(aura_option);
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
    builder.set_field_data_manager(std::move(field_data_manager));
#else
    builder.set_field_data_manager(field_data_manager);  // STK takes ownership. Bad practice but legacy.
#endif
    builder.set_initial_bucket_capacity(initial_bucket_capacity);
    builder.set_maximum_bucket_capacity(maximum_bucket_capacity);

    if (meta_data_ptr_ == nullptr) {
      meta_data_ptr_ = builder.create_meta_data();
      meta_data_ptr_->use_simple_fields();  // TODO(palmerb4): This is supposedly depreciated but still necessary, as
                                            // stk::io::fill_mesh_with_auto_decomp will throw without it.
      meta_data_ptr_->set_coordinate_field_name("coordinates");
    }

    if (bulk_data_ptr_ == nullptr) {
      bulk_data_ptr_ = builder.create(meta_data_ptr_);
      aura_option_ = aura_option;
      initial_bucket_capacity_ = initial_bucket_capacity;
      maximum_bucket_capacity_ = maximum_bucket_capacity;
    }

    block1_part_ptr_ = &meta_data_ptr_->declare_part_with_topology("block_1", stk::topology::HEX_8);
    block2_part_ptr_ = &meta_data_ptr_->declare_part_with_topology("block_2", stk::topology::HEX_8);
    block3_part_ptr_ = &meta_data_ptr_->declare_part_with_topology("block_3", stk::topology::HEX_8);
    block1_selector_ = *block1_part_ptr_;
    block2_selector_ = *block2_part_ptr_;
    block3_selector_ = *block3_part_ptr_;

    node_coord_field_ptr_ = &meta_data_ptr_->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    
    unsigned scalars_per_entity = 1;
    field_x_ptr_ = create_field_on_parts("field_x", stk::topology::NODE_RANK, scalars_per_entity,
                                         {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_y_ptr_ = create_field_on_parts("field_y", stk::topology::NODE_RANK, scalars_per_entity,
                                         {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_z_ptr_ = create_field_on_parts("field_z", stk::topology::NODE_RANK, scalars_per_entity,
                                         {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_xs_ptr_ = create_field_on_parts("field_x_scratch", stk::topology::NODE_RANK, scalars_per_entity,
                                          {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_ys_ptr_ = create_field_on_parts("field_y_scratch", stk::topology::NODE_RANK, scalars_per_entity,
                                          {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_zs_ptr_ = create_field_on_parts("field_z_scratch", stk::topology::NODE_RANK, scalars_per_entity,
                                          {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field1_ptr_ = create_field_on_parts("field1", stk::topology::NODE_RANK, scalars_per_entity, {block1_part_ptr_});
    field2_ptr_ = create_field_on_parts("field2", stk::topology::NODE_RANK, scalars_per_entity, {block2_part_ptr_});
    field3_ptr_ = create_field_on_parts("field3", stk::topology::NODE_RANK, scalars_per_entity, {block3_part_ptr_});

    declare_five_hexes();
    // declare_N_hexes_per_dimension(100);
    reset_field_values();
  }

 protected:
  MPI_Comm communicator_;
  unsigned spatial_dimension_;
  std::vector<std::string> entity_rank_names_;
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr_;
  stk::mesh::BulkData::AutomaticAuraOption aura_option_{stk::mesh::BulkData::AUTO_AURA};
  unsigned initial_bucket_capacity_ = 0;
  unsigned maximum_bucket_capacity_ = 0;

  DoubleField* node_coord_field_ptr_;

  DoubleField* field_x_ptr_;
  DoubleField* field_y_ptr_;
  DoubleField* field_z_ptr_;

  DoubleField* field_xs_ptr_;
  DoubleField* field_ys_ptr_;
  DoubleField* field_zs_ptr_;

  DoubleField* field1_ptr_;
  DoubleField* field2_ptr_;
  DoubleField* field3_ptr_;

  stk::mesh::Part* block1_part_ptr_;
  stk::mesh::Part* block2_part_ptr_;
  stk::mesh::Part* block3_part_ptr_;

  stk::mesh::Selector block1_selector_;
  stk::mesh::Selector block2_selector_;
  stk::mesh::Selector block3_selector_;
};  // class UnitTestAccessorExprFixture

struct XTag;
struct YTag;
struct ZTag;

TEST_F(UnitTestAccessorExprFixture, field_fill) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  const double fill_value = 3.14159;
  auto expected_value_func = [fill_value](const double* entity_coords) { return std::vector<double>{fill_value}; };

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));

  {
    auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);
    x(es) = fill_value;
  }

  check_field_data_on_host_func<1>("fill_field does not fill.", get_bulk(), *field_x_ptr_, b1_not_b2, {},
                                   expected_value_func);
  check_field_data_on_host_func<1>("fill_field does not respect selector.", get_bulk(), *field_x_ptr_, !b1_not_b2, {},
                                   get_field_x_func());
}

TEST_F(UnitTestAccessorExprFixture, field_copy) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));
  auto y = make_tagged_component<YTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_y_ptr_));

  {
    auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);
    x(es) = y(es);
  }

  check_field_data_on_host_func<1>("field copy error. x", get_bulk(), *field_x_ptr_, b1_not_b2, {}, get_field_y_func());
  check_field_data_on_host_func<1>("field copy error. y", get_bulk(), *field_y_ptr_, b1_not_b2, {}, get_field_y_func());

  check_field_data_on_host_func<1>("field subset error. x", get_bulk(), *field_x_ptr_, !b1_not_b2, {},
                                   get_field_x_func());
  check_field_data_on_host_func<1>("field subset error. y", get_bulk(), *field_y_ptr_, !b1_not_b2, {},
                                   get_field_y_func());
}

TEST_F(UnitTestAccessorExprFixture, field_swap) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));
  auto y = make_tagged_component<YTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_y_ptr_));

  {
    auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);

    // fused_assign evaluates all right hand sides before assigning them to the left hand sides.
    // This is the same as python's syntax: x, y = y, x.
    //
    // You must still use copy because since the result of x(es) is a view, so the stashed rhs is a copy of a view.
    // y_view_copy = y_view
    // x_view_copy = x_view
    // x_view[0] = y_view_copy[0]
    // y_view[0] = x_view_copy[0]
    //
    // If you use copy, then the logic becomes
    // y_copy = copy(y_view)
    // x_copy = copy(x_view)
    // x_view[0] = y_copy[0]
    // y_view[0] = x_copy[0]
    //
    // Should we do this automatically in fused_assign if the rhs is an AccessorExpr?
    fused_assign(x(es), /*=*/copy(y(es)),  //
                 y(es), /*=*/copy(x(es)));
  }

  check_field_data_on_host_func<1>("field_swap error. x", get_bulk(), *field_x_ptr_, b1_not_b2, {}, get_field_y_func());
  check_field_data_on_host_func<1>("field_swap error. y", get_bulk(), *field_y_ptr_, b1_not_b2, {}, get_field_x_func());

  check_field_data_on_host_func<1>("field subset error. x", get_bulk(), *field_x_ptr_, !b1_not_b2, {},
                                   get_field_x_func());
  check_field_data_on_host_func<1>("field subset error. y", get_bulk(), *field_y_ptr_, !b1_not_b2, {},
                                   get_field_y_func());
}

TEST_F(UnitTestAccessorExprFixture, field_scale) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  const double alpha = 3.14159;
  auto field_x_func = get_field_x_func();
  auto expected_value_func = [&alpha, &field_x_func](const double* entity_coords) {
    return std::vector<double>{alpha * field_x_func(entity_coords)[0]};
  };

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));

  {
    auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);
    x(es) *= alpha;
  }

  check_field_data_on_host_func<1>("field_scale does not fill.", get_bulk(), *field_x_ptr_, b1_not_b2, {},
                                   expected_value_func);
  check_field_data_on_host_func<1>("field_scale does not respect selector.", get_bulk(), *field_x_ptr_, !b1_not_b2, {},
                                   get_field_x_func());
}

TEST_F(UnitTestAccessorExprFixture, field_product) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  auto field_x_func = get_field_x_func();
  auto field_y_func = get_field_y_func();
  auto expected_value_func = [&field_x_func, &field_y_func](const double* entity_coords) {
    return std::vector<double>{field_x_func(entity_coords)[0] * field_y_func(entity_coords)[0]};
  };

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));
  auto y = make_tagged_component<YTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_y_ptr_));
  auto z = make_tagged_component<ZTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_z_ptr_));

  {
    auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);
    z(es) = x(es) * y(es);
  }

  check_field_data_on_host_func<1>("field_product error. x", get_bulk(), *field_x_ptr_, b1_not_b2, {},
                                   get_field_x_func());
  check_field_data_on_host_func<1>("field_product error. y", get_bulk(), *field_y_ptr_, b1_not_b2, {},
                                   get_field_y_func());
  check_field_data_on_host_func<1>("field_product error. z", get_bulk(), *field_z_ptr_, b1_not_b2, {},
                                   expected_value_func);

  check_field_data_on_host_func<1>("field subset error. x", get_bulk(), *field_x_ptr_, !b1_not_b2, {},
                                   get_field_x_func());
  check_field_data_on_host_func<1>("field subset error. y", get_bulk(), *field_y_ptr_, !b1_not_b2, {},
                                   get_field_y_func());
  check_field_data_on_host_func<1>("field subset error. z", get_bulk(), *field_z_ptr_, !b1_not_b2, {},
                                   get_field_z_func());
}

TEST_F(UnitTestAccessorExprFixture, field_axpby) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  const double alpha = 3.14159;
  const double beta = 2.71828;
  auto field_x_func = get_field_x_func();
  auto field_y_func = get_field_y_func();
  auto expected_value_func = [alpha, beta, &field_x_func, &field_y_func](const double* entity_coords) {
    return std::vector<double>{alpha * field_x_func(entity_coords)[0] + beta * field_y_func(entity_coords)[0]};
  };

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));
  auto y = make_tagged_component<YTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_y_ptr_));

  {
    auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);
    y(es) = alpha * x(es) + beta * y(es);
  }

  check_field_data_on_host_func<1>("field_axpby error. x", get_bulk(), *field_x_ptr_, b1_not_b2, {},
                                   get_field_x_func());
  check_field_data_on_host_func<1>("field_axpby error. y", get_bulk(), *field_y_ptr_, b1_not_b2, {},
                                   expected_value_func);

  check_field_data_on_host_func<1>("field subset error. x", get_bulk(), *field_x_ptr_, !b1_not_b2, {},
                                   get_field_x_func());
  check_field_data_on_host_func<1>("field subset error. y", get_bulk(), *field_y_ptr_, !b1_not_b2, {},
                                   get_field_y_func());
}

template <size_t NumComponents>
double host_direct_field_dot(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field1,
                             const stk::mesh::FieldBase& field2, const stk::mesh::Selector& selector) {
  double local_dot = 0.0;
  stk::mesh::for_each_entity_run(
      bulk_data, field1.entity_rank(), selector,
      [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
        const double* raw_field1_data = reinterpret_cast<const double*>(stk::mesh::field_data(field1, entity));
        const double* raw_field2_data = reinterpret_cast<const double*>(stk::mesh::field_data(field2, entity));
        for (size_t i = 0; i < NumComponents; ++i) {
#pragma omp atomic
          local_dot += raw_field1_data[i] * raw_field2_data[i];
        }
      });

  double global_dot = 0.0;
  stk::all_reduce_sum(bulk_data.parallel(), &local_dot, &global_dot, 1);
  return global_dot;
}

template <size_t NumComponents>
double host_direct_field_nrm2(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  return std::sqrt(host_direct_field_dot<NumComponents>(bulk_data, field, field, selector));
}

template <size_t NumComponents>
double host_direct_field_sum(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                             const stk::mesh::Selector& selector) {
  double local_sum = 0.0;
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (size_t i = 0; i < NumComponents; ++i) {
#pragma omp atomic
                                     local_sum += raw_field_data[i];
                                   }
                                 });

  double global_sum = 0.0;
  stk::all_reduce_sum(bulk_data.parallel(), &local_sum, &global_sum, 1);
  return global_sum;
}

template <size_t NumComponents>
double host_direct_field_asum(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  double local_asum = 0.0;
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (size_t i = 0; i < NumComponents; ++i) {
#pragma omp critical
                                     {
                                       local_asum += Kokkos::abs(raw_field_data[i]);
                                     }
                                   }
                                 });

  double global_asum = 0.0;
  stk::all_reduce_sum(bulk_data.parallel(), &local_asum, &global_asum, 1);
  return global_asum;
}

template <size_t NumComponents>
double host_direct_field_max(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                             const stk::mesh::Selector& selector) {
  double local_max = -std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (size_t i = 0; i < NumComponents; ++i) {
#pragma omp critical
                                     {
                                       local_max = Kokkos::max(local_max, raw_field_data[i]);
                                     }
                                   }
                                 });

  double global_max = 0.0;
  stk::all_reduce_max(bulk_data.parallel(), &local_max, &global_max, 1);
  return global_max;
}

template <size_t NumComponents>
double host_direct_field_amax(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  double local_amax = -std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (size_t i = 0; i < NumComponents; ++i) {
#pragma omp critical
                                     {
                                       local_amax = Kokkos::max(local_amax, Kokkos::abs(raw_field_data[i]));
                                     }
                                   }
                                 });

  double global_amax = 0.0;
  stk::all_reduce_max(bulk_data.parallel(), &local_amax, &global_amax, 1);
  return global_amax;
}

template <size_t NumComponents>
double host_direct_field_min(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                             const stk::mesh::Selector& selector) {
  double local_min = std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (size_t i = 0; i < NumComponents; ++i) {
#pragma omp critical
                                     {
                                       local_min = Kokkos::min(local_min, raw_field_data[i]);
                                     }
                                   }
                                 });

  double global_min = 0.0;
  stk::all_reduce_min(bulk_data.parallel(), &local_min, &global_min, 1);
  return global_min;
}

template <size_t NumComponents>
double host_direct_field_amin(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  double local_amin = std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (size_t i = 0; i < NumComponents; ++i) {
#pragma omp critical
                                     {
                                       local_amin = Kokkos::min(local_amin, Kokkos::abs(raw_field_data[i]));
                                     }
                                   }
                                 });

  double global_amin = 0.0;
  stk::all_reduce_min(bulk_data.parallel(), &local_amin, &global_amin, 1);
  return global_amin;
}

TEST_F(UnitTestAccessorExprFixture, field_dot) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));
  auto y = make_tagged_component<YTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_y_ptr_));

  auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);
  double actual_dot = all_reduce_sum<double>(x(es) * y(es));
  double expected_dot = host_direct_field_dot<1>(get_bulk(), *field_x_ptr_, *field_y_ptr_, b1_not_b2);
  EXPECT_NEAR(actual_dot, expected_dot, 1.0e-12);
}

TEST_F(UnitTestAccessorExprFixture, quick_perf_test_against_blas) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
#else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr =
      new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
#endif

  stk::mesh::Selector b1_not_b2 = block1_selector_ - block2_selector_;
  auto x = make_tagged_component<XTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_x_ptr_));
  auto y = make_tagged_component<YTag, stk::topology::NODE_RANK>(ScalarFieldComponent(*field_y_ptr_));

  auto es = make_entity_expr(get_bulk(), b1_not_b2, stk::topology::NODE_RANK);

  unsigned num_warmups = 10;
  unsigned num_replicates = 100;
  Kokkos::Timer timer;

  double elapsed_expr = 0;
  double elapsed_blas = 0;
  for (unsigned i = 0; i < num_replicates + num_warmups; ++i) {
    if (i < num_warmups) {
      double actual_dot = all_reduce_sum<double>(x(es) * y(es));
      double expected_dot = field_dot<double>(*field_x_ptr_, *field_y_ptr_, b1_not_b2, stk::ngp::ExecSpace());
      EXPECT_NEAR(actual_dot, expected_dot, 1.0e-10);
      timer.reset();
    } else {
      double actual_dot = all_reduce_sum<double>(x(es) * y(es));
      Kokkos::fence();
      elapsed_expr += timer.seconds();
      timer.reset();
      double expected_dot = field_dot<double>(*field_x_ptr_, *field_y_ptr_, b1_not_b2, stk::ngp::ExecSpace());
      Kokkos::fence();
      elapsed_blas += timer.seconds();
      EXPECT_NEAR(actual_dot, expected_dot, 1.0e-10);
      timer.reset();
    }
  }
  elapsed_expr /= num_replicates;
  elapsed_blas /= num_replicates;
  std::cout << "AccessorExpr dot time: " << elapsed_expr << std::endl;
  std::cout << "BLAS dot time: " << elapsed_blas << std::endl;
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
