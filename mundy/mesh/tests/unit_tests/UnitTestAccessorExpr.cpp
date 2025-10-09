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

        unsigned int num_components = stk::mesh::field_scalars_per_entity(stk_field, entity);
        for (const stk::mesh::FieldBase* other_field : other_fields) {
          num_components = std::min(num_components, stk::mesh::field_scalars_per_entity(*other_field, entity));
        }
        const double* raw_field_data = reinterpret_cast<const double*>(stk::mesh::field_data(stk_field, entity));
        for (unsigned int i = 0; i < num_components; ++i) {
          EXPECT_DOUBLE_EQ(raw_field_data[i], expected_values[i]) << message_to_throw;
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
      return std::vector<double>{coords[0] * coords[1],
                                 3 * coords[1] * coords[2],
                                 5 * coords[2] * coords[0],
                                 7 * coords[0] * coords[0],
                                 11 * coords[1] * coords[1],
                                 13 * coords[2] * coords[2],
                                 17 * coords[0] * coords[1],
                                 19 * coords[1] * coords[2],
                                 23 * coords[2] * coords[0]};
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
  }

  void validate_initial_mesh() {
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

  void setup_hex_mesh(
      const stk::mesh::EntityRank& entity_rank, stk::mesh::BulkData::AutomaticAuraOption aura_option,
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
    field_x_ptr_ = create_field_on_parts("field_x", stk::topology::NODE_RANK, 9,
                                         {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_y_ptr_ = create_field_on_parts("field_y", stk::topology::NODE_RANK, 9,
                                         {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_z_ptr_ = create_field_on_parts("field_z", stk::topology::NODE_RANK, 9,
                                         {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_scratch1_ptr_ = create_field_on_parts("field_scratch1", stk::topology::NODE_RANK, 9,
                                                {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field_scratch2_ptr_ = create_field_on_parts("field_scratch2", stk::topology::NODE_RANK, 9,
                                                {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});
    field1_ptr_ = create_field_on_parts("field1", stk::topology::NODE_RANK, 9, {block1_part_ptr_});
    field2_ptr_ = create_field_on_parts("field2", stk::topology::NODE_RANK, 9, {block2_part_ptr_});
    field3_ptr_ = create_field_on_parts("field3", stk::topology::NODE_RANK, 9, {block3_part_ptr_});

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
    validate_initial_mesh();
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

  DoubleField* field_scratch1_ptr_;
  DoubleField* field_scratch2_ptr_;

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

TEST_F(UnitTestAccessorExprFixture, Construction) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  const int we_know_there_are_five_ranks = 5;
  #if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
  #else
  stk::mesh::DefaultFieldDataManager* field_data_manager_ptr = new stk::mesh::DefaultFieldDataManager(we_know_there_are_five_ranks);
  setup_hex_mesh(stk::topology::NODE_RANK, stk::mesh::BulkData::AUTO_AURA, field_data_manager_ptr);
  #endif

  auto x_accessor = ScalarFieldComponent(*field_x_ptr_);
  auto y_accessor = ScalarFieldComponent(*field_y_ptr_);

  auto ngp_x_accessor = get_updated_ngp_component(x_accessor);
  auto ngp_y_accessor = get_updated_ngp_component(y_accessor);

  auto ngp_mesh = get_updated_ngp_mesh(get_bulk());
  auto eblock1 = make_entity_expr(ngp_mesh, block1_selector_, stk::topology::NODE_RANK);

  ngp_x_accessor(eblock1) *= ngp_y_accessor(eblock1);
}


// TEST_F(UnitTestAccessorExprFixture, Construction) {
//   if (stk::parallel_machine_size(communicator_) > 2) {
//     GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
//   }

//   for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
//     const int we_know_there_are_five_ranks = 5;
//     auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
//     setup_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

//     auto field_x_func = get_field_x_func();
//     auto field_y_func = get_field_y_func();
//     auto field_z_func = get_field_z_func();
//     auto expected_value_func = [&field_x_func, &field_y_func, &field_z_func](const double* entity_coords) {
//       std::vector<double> field_x_values = field_x_func(entity_coords);
//       std::vector<double> field_y_values = field_y_func(entity_coords);
//       std::vector<double> field_z_values = field_z_func(entity_coords);
//       const unsigned min_size = std::min({field_x_values.size(), field_y_values.size(), field_z_values.size()});
//       for (unsigned i = 0; i < min_size; ++i) {
//         field_z_values[i] = field_x_values[i] * field_y_values[i];
//       }
//       return field_z_values;
//     };

//     {



//     }

//     // field_product<double>(*field_x_ptr_, *field_y_ptr_, *field_z_ptr_, block1_selector_ - block2_selector_,
//     //                       stk::ngp::ExecSpace());

//     check_field_data_on_host_func("product_field does not multiply.", get_bulk(), *field3_ptr_,
//                                   block1_selector_ - block2_selector_, {}, expected_value_func);
//     check_field_data_on_host_func("product_field does not respect selector.", get_bulk(), *field3_ptr_,
//                                   block2_selector_ - block1_selector_, {}, get_field_z_func());

//     reset_mesh();
//   }
// }

}  // namespace

}  // namespace mesh

}  // namespace mundy
