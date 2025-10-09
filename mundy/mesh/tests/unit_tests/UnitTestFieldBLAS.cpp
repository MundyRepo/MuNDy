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

// C++ core
#include <functional>  // for std::function
#include <memory>      // for std::unique_ptr
#include <vector>      // for std::vector

// STK
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_io/FillMesh.hpp>  // for stk::io::fill_mesh_with_auto_decomp
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/FieldDataManager.hpp>
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>

// Mundy
#include <mundy_mesh/NgpFieldBLAS.hpp>  // for mundy::mesh::field_fill, mundy::mesh::field_copy, etc

namespace mundy {

namespace mesh {

namespace {

/*
The full set of field operations that we need to test are
  - field_fill
  - field_copy
  - field_swap
  - field_scale
  - field_product
  - field_axpy
  - field_axpby
  - field_axpbyz
  - field_dot
  - field_nrm2
  - field_sum
  - field_asum
  - field_max
  - field_amax
  - field_min
  - field_amin
Both with and without a selector.

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
  const stk::mesh::FieldBase &coord_field = *stk_mesh.mesh_meta_data().coordinate_field();
  
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
  const stk::mesh::FieldBase &coord_field = *stk_mesh.mesh_meta_data().coordinate_field();

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

class UnitTestFieldBLAS : public ::testing::Test {
 public:
  using DoubleField = stk::mesh::Field<double>;
  using CoordinateFunc = std::function<std::vector<double>(const double*)>;
  static constexpr double initial_value[3] = {-1, 2, -0.3};

  UnitTestFieldBLAS()
      : communicator_(MPI_COMM_WORLD),
        spatial_dimension_(3),
        entity_rank_names_({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"}) {
  }

  virtual ~UnitTestFieldBLAS() {
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

  CoordinateFunc get_field1_func() const {
    return [](const double* entity_coords) {
      return std::vector<double>{-entity_coords[0] * entity_coords[0], 2 * entity_coords[1]};
    };
  }

  CoordinateFunc get_field2_func() const {
    return [](const double* entity_coords) {
      return std::vector<double>{entity_coords[0] * entity_coords[1], 3 * entity_coords[1] * entity_coords[1]};
    };
  }

  CoordinateFunc get_field3_func() const {
    return [](const double* entity_coords) {
      return std::vector<double>{entity_coords[0] * entity_coords[1], 3 * entity_coords[1] * entity_coords[2],
                                 5 * entity_coords[2] * entity_coords[0]};
    };
  }

  void reset_field_values() {
    randomize_coordinates(*bulk_data_ptr_, *node_coord_field_ptr_, spatial_dimension_);
    set_field_data_on_host(*bulk_data_ptr_, *field1_ptr_, block1_selector_ | block2_selector_, get_field1_func());
    set_field_data_on_host(*bulk_data_ptr_, *field2_ptr_, block1_selector_ | block2_selector_, get_field2_func());
    set_field_data_on_host(*bulk_data_ptr_, *field3_ptr_, block1_selector_ | block2_selector_ | block3_selector_,
                           get_field3_func());
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

  void setup_three_field_five_hex_mesh(
      const stk::mesh::EntityRank& entity_rank, stk::mesh::BulkData::AutomaticAuraOption aura_option,
      std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager,
      unsigned initial_bucket_capacity = stk::mesh::get_default_initial_bucket_capacity(),
      unsigned maximum_bucket_capacity = stk::mesh::get_default_maximum_bucket_capacity()) {
    stk::mesh::MeshBuilder builder(communicator_);
    builder.set_spatial_dimension(spatial_dimension_);
    builder.set_entity_rank_names(entity_rank_names_);
    builder.set_aura_option(aura_option);
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
    builder.set_field_data_manager(std::move(field_data_manager));
#else
    builder.set_field_data_manager(field_data_manager.get());
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
    field1_ptr_ = create_field_on_parts("field1", entity_rank, 2, {block1_part_ptr_, block2_part_ptr_});
    field2_ptr_ = create_field_on_parts("field2", entity_rank, 2, {block1_part_ptr_, block2_part_ptr_});
    field3_ptr_ =
        create_field_on_parts("field3", entity_rank, 3, {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});

    // Sanity check field rank and part membership
    EXPECT_EQ(field1_ptr_->entity_rank(), entity_rank);
    EXPECT_EQ(field2_ptr_->entity_rank(), entity_rank);
    EXPECT_EQ(field3_ptr_->entity_rank(), entity_rank);
    EXPECT_TRUE(field1_ptr_->defined_on(*block1_part_ptr_));
    EXPECT_TRUE(field1_ptr_->defined_on(*block2_part_ptr_));
    EXPECT_TRUE(field2_ptr_->defined_on(*block1_part_ptr_));
    EXPECT_TRUE(field2_ptr_->defined_on(*block2_part_ptr_));
    EXPECT_TRUE(field3_ptr_->defined_on(*block1_part_ptr_));
    EXPECT_TRUE(field3_ptr_->defined_on(*block2_part_ptr_));
    EXPECT_TRUE(field3_ptr_->defined_on(*block3_part_ptr_));

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
  DoubleField* field1_ptr_;
  DoubleField* field2_ptr_;
  DoubleField* field3_ptr_;

  stk::mesh::Part* block1_part_ptr_;
  stk::mesh::Part* block2_part_ptr_;
  stk::mesh::Part* block3_part_ptr_;

  stk::mesh::Selector block1_selector_;
  stk::mesh::Selector block2_selector_;
  stk::mesh::Selector block3_selector_;
};  // class UnitTestFieldBLAS

TEST_F(UnitTestFieldBLAS, Fixture) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  EXPECT_NO_THROW(setup_three_field_five_hex_mesh(stk::topology::ELEMENT_RANK, stk::mesh::BulkData::AUTO_AURA,
                                                  std::make_unique<stk::mesh::DefaultFieldDataManager>(5)));
}

TEST_F(UnitTestFieldBLAS, field_fill) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    const double fill_value = 3.14159;
    auto expected_value_func = [fill_value](const double* entity_coords) {
      return std::vector<double>{fill_value, fill_value};
    };

    field_fill(fill_value, *field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    check_field_data_on_host_func("fill_field does not fill.", get_bulk(), *field1_ptr_,
                                  block1_selector_ - block2_selector_, {}, expected_value_func);
    check_field_data_on_host_func("fill_field does not respect selector.", get_bulk(), *field1_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field1_func());

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_copy) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    field_copy<double>(*field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    check_field_data_on_host_func("copy_field does not copy.", get_bulk(), *field2_ptr_,
                                  block1_selector_ - block2_selector_, {}, get_field1_func());
    check_field_data_on_host_func("copy_field does not respect selector.", get_bulk(), *field2_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field2_func());

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_swap) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    field_swap<double>(*field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    check_field_data_on_host_func("swap_field does not swap.", get_bulk(), *field1_ptr_,
                                  block1_selector_ - block2_selector_, {}, get_field2_func());
    check_field_data_on_host_func("swap_field does not swap.", get_bulk(), *field2_ptr_,
                                  block1_selector_ - block2_selector_, {}, get_field1_func());
    check_field_data_on_host_func("swap_field does not respect selector.", get_bulk(), *field1_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field1_func());
    check_field_data_on_host_func("swap_field does not respect selector.", get_bulk(), *field2_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field2_func());

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_scale) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    const double alpha = 3.14159;
    auto field1_func = get_field1_func();
    auto expected_value_func = [&alpha, &field1_func](const double* entity_coords) {
      std::vector<double> field_values = field1_func(entity_coords);
      for (double& value : field_values) {
        value *= alpha;
      }
      return field_values;
    };

    field_scale(alpha, *field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    check_field_data_on_host_func("scale_field does not scale.", get_bulk(), *field1_ptr_,
                                  block1_selector_ - block2_selector_, {}, expected_value_func);
    check_field_data_on_host_func("scale_field does not respect selector.", get_bulk(), *field1_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field1_func());

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_product) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    auto field1_func = get_field1_func();
    auto field2_func = get_field2_func();
    auto field3_func = get_field3_func();
    auto expected_value_func = [&field1_func, &field2_func, &field3_func](const double* entity_coords) {
      std::vector<double> field1_values = field1_func(entity_coords);
      std::vector<double> field2_values = field2_func(entity_coords);
      std::vector<double> field3_values = field3_func(entity_coords);
      const unsigned min_size = std::min({field1_values.size(), field2_values.size(), field3_values.size()});
      for (unsigned i = 0; i < min_size; ++i) {
        field3_values[i] = field1_values[i] * field2_values[i];
      }
      return field3_values;
    };

    field_product<double>(*field1_ptr_, *field2_ptr_, *field3_ptr_, block1_selector_ - block2_selector_,
                          stk::ngp::ExecSpace());

    check_field_data_on_host_func("product_field does not multiply.", get_bulk(), *field3_ptr_,
                                  block1_selector_ - block2_selector_, {}, expected_value_func);
    check_field_data_on_host_func("product_field does not respect selector.", get_bulk(), *field3_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field3_func());

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_axpy) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    const double alpha = 3.14159;
    auto field1_func = get_field1_func();
    auto field2_func = get_field2_func();
    auto expected_value_func = [&alpha, &field1_func, &field2_func](const double* entity_coords) {
      std::vector<double> field1_values = field1_func(entity_coords);
      std::vector<double> field2_values = field2_func(entity_coords);
      const unsigned min_size = std::min(field1_values.size(), field2_values.size());
      for (unsigned i = 0; i < min_size; ++i) {
        field2_values[i] += alpha * field1_values[i];
      }
      return field2_values;
    };

    field_axpy(alpha, *field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    check_field_data_on_host_func("axpy_field does not axpy.", get_bulk(), *field2_ptr_,
                                  block1_selector_ - block2_selector_, {}, expected_value_func);
    check_field_data_on_host_func("axpy_field does not respect selector.", get_bulk(), *field2_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field2_func());

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_axpby) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    const double alpha = 3.14159;
    const double beta = 0.3333333;
    auto field1_func = get_field1_func();
    auto field2_func = get_field2_func();
    auto expected_value_func = [&alpha, &beta, &field1_func, &field2_func](const double* entity_coords) {
      std::vector<double> field1_values = field1_func(entity_coords);
      std::vector<double> field2_values = field2_func(entity_coords);
      const unsigned min_size = std::min(field1_values.size(), field2_values.size());
      for (unsigned i = 0; i < min_size; ++i) {
        field2_values[i] = alpha * field1_values[i] + beta * field2_values[i];
      }
      return field2_values;
    };

    field_axpby(alpha, *field1_ptr_, beta, *field2_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    check_field_data_on_host_func("axpby_field does not axpby.", get_bulk(), *field2_ptr_,
                                  block1_selector_ - block2_selector_, {}, expected_value_func);
    check_field_data_on_host_func("axpby_field does not respect selector.", get_bulk(), *field2_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field2_func());

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_axpbyz) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    const double alpha = 3.14159;
    const double beta = 0.3333333;
    auto field1_func = get_field1_func();
    auto field2_func = get_field2_func();
    auto field3_func = get_field3_func();
    auto expected_value_func = [&alpha, &beta, &field1_func, &field2_func, &field3_func](const double* entity_coords) {
      std::vector<double> field1_values = field1_func(entity_coords);
      std::vector<double> field2_values = field2_func(entity_coords);
      std::vector<double> field3_values = field3_func(entity_coords);
      const unsigned min_size = std::min({field1_values.size(), field2_values.size(), field3_values.size()});
      for (unsigned i = 0; i < min_size; ++i) {
        field3_values[i] = alpha * field1_values[i] + beta * field2_values[i];
      }
      return field3_values;
    };

    field_axpbyz(alpha, *field1_ptr_, beta, *field2_ptr_, *field3_ptr_, block1_selector_ - block2_selector_,
                 stk::ngp::ExecSpace());

    check_field_data_on_host_func("axpbyz_field does not axpbyz.", get_bulk(), *field3_ptr_,
                                  block1_selector_ - block2_selector_, {}, expected_value_func);
    check_field_data_on_host_func("axpbyz_field does not respect selector.", get_bulk(), *field3_ptr_,
                                  block2_selector_ - block1_selector_, {}, get_field3_func());

    reset_mesh();
  }
}

double host_direct_field_dot(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field1,
                             const stk::mesh::FieldBase& field2, const stk::mesh::Selector& selector) {
  double local_dot = 0.0;
  stk::mesh::for_each_entity_run(
      bulk_data, field1.entity_rank(), selector,
      [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
        int num_components = Kokkos::min(stk::mesh::field_scalars_per_entity(field1, entity),
                                         stk::mesh::field_scalars_per_entity(field2, entity));
        const double* raw_field1_data = reinterpret_cast<const double*>(stk::mesh::field_data(field1, entity));
        const double* raw_field2_data = reinterpret_cast<const double*>(stk::mesh::field_data(field2, entity));
        for (int i = 0; i < num_components; ++i) {
#pragma omp atomic
          local_dot += raw_field1_data[i] * raw_field2_data[i];
        }
      });

  double global_dot = 0.0;
  stk::all_reduce_sum(bulk_data.parallel(), &local_dot, &global_dot, 1);
  return global_dot;
}

double host_direct_field_nrm2(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  return std::sqrt(host_direct_field_dot(bulk_data, field, field, selector));
}

double host_direct_field_sum(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                             const stk::mesh::Selector& selector) {
  double local_sum = 0.0;
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (int i = 0; i < num_components; ++i) {
#pragma omp atomic
                                     local_sum += raw_field_data[i];
                                   }
                                 });

  double global_sum = 0.0;
  stk::all_reduce_sum(bulk_data.parallel(), &local_sum, &global_sum, 1);
  return global_sum;
}

double host_direct_field_asum(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  double local_asum = 0.0;
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (int i = 0; i < num_components; ++i) {
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

double host_direct_field_max(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                             const stk::mesh::Selector& selector) {
  double local_max = -std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (int i = 0; i < num_components; ++i) {
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

double host_direct_field_amax(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  double local_amax = -std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (int i = 0; i < num_components; ++i) {
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

double host_direct_field_min(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                             const stk::mesh::Selector& selector) {
  double local_min = std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (int i = 0; i < num_components; ++i) {
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

double host_direct_field_amin(const stk::mesh::BulkData& bulk_data, const stk::mesh::FieldBase& field,
                              const stk::mesh::Selector& selector) {
  double local_amin = std::numeric_limits<double>::max();
  stk::mesh::for_each_entity_run(bulk_data, field.entity_rank(), selector,
                                 [&]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
                                   const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
                                   const double* raw_field_data =
                                       reinterpret_cast<const double*>(stk::mesh::field_data(field, entity));
                                   for (int i = 0; i < num_components; ++i) {
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

TEST_F(UnitTestFieldBLAS, field_dot) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_dot =
        field_dot<double>(*field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_dot =
        host_direct_field_dot(get_bulk(), *field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_dot, expected_dot, 1.0e-12);

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_nrm2) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_nrm2 = field_nrm2<double>(*field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_nrm2 = host_direct_field_nrm2(get_bulk(), *field1_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_nrm2, expected_nrm2, 1.0e-12);

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_sum) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_sum = field_sum<double>(*field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_sum = host_direct_field_sum(get_bulk(), *field1_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_sum, expected_sum, 1.0e-12);

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_asum) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_asum = field_asum<double>(*field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_asum = host_direct_field_asum(get_bulk(), *field1_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_asum, expected_asum, 1.0e-12);

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_max) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_max = field_max<double>(*field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_max = host_direct_field_max(get_bulk(), *field1_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_max, expected_max, 1.0e-12);

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_amax) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_amax = field_amax<double>(*field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_amax = host_direct_field_amax(get_bulk(), *field1_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_amax, expected_amax, 1.0e-12);

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_min) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_min = field_min<double>(*field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_min = host_direct_field_min(get_bulk(), *field1_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_min, expected_min, 1.0e-12);

    reset_mesh();
  }
}

TEST_F(UnitTestFieldBLAS, field_amin) {
  if (stk::parallel_machine_size(communicator_) > 2) {
    GTEST_SKIP() << "This test is only designed to run with 1 or 2 MPI ranks.";
  }

  for (const stk::mesh::EntityRank& entity_rank : {stk::topology::ELEMENT_RANK, stk::topology::NODE_RANK}) {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_five_hex_mesh(entity_rank, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));

    double ngp_amin = field_amin<double>(*field1_ptr_, block1_selector_ - block2_selector_, stk::ngp::ExecSpace());

    double expected_amin = host_direct_field_amin(get_bulk(), *field1_ptr_, block1_selector_ - block2_selector_);
    EXPECT_NEAR(ngp_amin, expected_amin, 1.0e-12);

    reset_mesh();
  }
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
