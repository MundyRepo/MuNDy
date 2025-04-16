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

// C++ core
#include <functional>  // for std::function
#include <iostream>    // for std::cout, std::endl
#include <memory>      // for std::unique_ptr
#include <vector>      // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#if TRILINOS_MAJOR_MINOR_VERSION >= 160000

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, etc

// STK
#include <stk_io/FillMesh.hpp>  // for stk::io::fill_mesh_with_auto_decomp
#include <stk_io/IossBridge.hpp>
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
#include <stk_mesh/base/NgpFieldBLAS.hpp>  // for stk::mesh::field_fill, stk::mesh::field_copy, etc
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>  // for MPI_Comm, MPI_COMM_WORLD

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE
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

inline void rough_randomize_field(const stk::mesh::BulkData& bulk_data, stk::mesh::FieldBase& field,
                                  const stk::mesh::Selector& selector) {
  stk::mesh::for_each_entity_run(
      bulk_data, field.entity_rank(), selector,
      [&field]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
        const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
        double* raw_field_data = reinterpret_cast<double*>(stk::mesh::field_data(field, entity));
        for (int i = 0; i < num_components; ++i) {
          raw_field_data[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        }
      });
}

class PerfTestFieldBLAS {
 public:
  using DoubleField = stk::mesh::Field<double>;
  using CoordinateFunc = std::function<std::vector<double>(const double*)>;
  static constexpr double initial_value[3] = {-1, 2, -0.3};

  PerfTestFieldBLAS(stk::mesh::EntityRank entity_rank)
      : entity_rank_(entity_rank),
        communicator_(MPI_COMM_WORLD),
        spatial_dimension_(3),
        entity_rank_names_({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"}) {
  }

  virtual ~PerfTestFieldBLAS() {
    reset();
  }

  virtual stk::mesh::BulkData& get_bulk() {
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::runtime_error,
                        "Trying to get bulk data before it has been initialized.");
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

  void setup_three_field_N_hex_mesh(
      const size_t num_hexes_per_dim, stk::mesh::BulkData::AutomaticAuraOption aura_option,
      std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager,
      unsigned initial_bucket_capacity = stk::mesh::get_default_initial_bucket_capacity(),
      unsigned maximum_bucket_capacity = stk::mesh::get_default_maximum_bucket_capacity()) {
    num_hexes_per_dim_ = num_hexes_per_dim;
    stk::mesh::MeshBuilder builder(communicator_);
    builder.set_spatial_dimension(spatial_dimension_);
    builder.set_entity_rank_names(entity_rank_names_);
    builder.set_aura_option(aura_option);
    builder.set_field_data_manager(std::move(field_data_manager));
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
    meta_data_ptr_->set_part_id(*block1_part_ptr_, 1);
    meta_data_ptr_->set_part_id(*block2_part_ptr_, 2);
    meta_data_ptr_->set_part_id(*block3_part_ptr_, 3);
    stk::io::put_io_part_attribute(*block1_part_ptr_);
    stk::io::put_io_part_attribute(*block2_part_ptr_);
    stk::io::put_io_part_attribute(*block3_part_ptr_);

    node_coord_field_ptr_ = &meta_data_ptr_->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    field1_ptr_ = create_field_on_parts("field1", entity_rank_, 2, {block1_part_ptr_, block2_part_ptr_});
    field2_ptr_ = create_field_on_parts("field2", entity_rank_, 2, {block1_part_ptr_, block2_part_ptr_});
    field3_ptr_ =
        create_field_on_parts("field3", entity_rank_, 3, {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_});

    field1_base_ptr_ = field1_ptr_;
    field2_base_ptr_ = field2_ptr_;
    field3_base_ptr_ = field3_ptr_;

    const std::string mesh_desc = "generated:" + std::to_string(num_hexes_per_dim_) + "x" +
                                  std::to_string(num_hexes_per_dim_) + "x" + std::to_string(num_hexes_per_dim_);
    stk::io::fill_mesh(mesh_desc, *bulk_data_ptr_);

    const size_t num_lo_elem_in_block_1 = stk::mesh::count_selected_entities(
        meta_data_ptr_->locally_owned_part() & *block1_part_ptr_, bulk_data_ptr_->buckets(stk::topology::ELEM_RANK));
    const size_t num_lo_elem_in_block_2 = stk::mesh::count_selected_entities(
        meta_data_ptr_->locally_owned_part() & *block2_part_ptr_, bulk_data_ptr_->buckets(stk::topology::ELEM_RANK));
    const size_t num_lo_elem_in_block_3 = stk::mesh::count_selected_entities(
        meta_data_ptr_->locally_owned_part() & *block3_part_ptr_, bulk_data_ptr_->buckets(stk::topology::ELEM_RANK));

    size_t num_elem_in_block_1 = 0;
    size_t num_elem_in_block_2 = 0;
    size_t num_elem_in_block_3 = 0;
    stk::all_reduce_sum(communicator_, &num_lo_elem_in_block_1, &num_elem_in_block_1, 1);
    stk::all_reduce_sum(communicator_, &num_lo_elem_in_block_2, &num_elem_in_block_2, 1);
    stk::all_reduce_sum(communicator_, &num_lo_elem_in_block_3, &num_elem_in_block_3, 1);
    MUNDY_THROW_REQUIRE(num_elem_in_block_1 == (num_hexes_per_dim * num_hexes_per_dim * num_hexes_per_dim),
                        std::runtime_error, "Block 1 should contain all of the elements.");
    MUNDY_THROW_REQUIRE(num_elem_in_block_2 == 0, std::runtime_error, "Block 2 should be empty.");
    MUNDY_THROW_REQUIRE(num_elem_in_block_3 == 0, std::runtime_error, "Block 3 should be empty.");

    // All of the hexes start in block_1. Move a third of them to block_2 and a third of them to block_3, removing them
    // from block_1.
    // bulk_data_ptr_->modification_begin();
    // stk::mesh::EntityVector entities_to_move_to_block_2;
    // stk::mesh::EntityVector entities_to_move_to_block_3;

    // const stk::mesh::BucketVector& buckets =
    //     bulk_data_ptr_->get_buckets(stk::topology::ELEM_RANK, *block1_part_ptr_ &
    //     meta_data_ptr_->locally_owned_part());
    // for (size_t bucket_count = 0, bucket_end = buckets.size(); bucket_count < bucket_end; ++bucket_count) {
    //   stk::mesh::Bucket& bucket = *buckets[bucket_count];
    //   for (size_t elem_count = 0, elem_end = bucket.size(); elem_count < elem_end; ++elem_count) {
    //     stk::mesh::Entity elem = bucket[elem_count];
    //     MUNDY_THROW_REQUIRE(bulk_data_ptr_->is_valid(elem), std::runtime_error, "Attempted to move an invalid
    //     entity."); if (elem_count % 3 == 0) {
    //       entities_to_move_to_block_2.push_back(elem);
    //     } else if (elem_count % 3 == 1) {
    //       entities_to_move_to_block_3.push_back(elem);
    //     }
    //   }
    // }

    // bulk_data_ptr_->change_entity_parts(entities_to_move_to_block_2, stk::mesh::ConstPartVector{block2_part_ptr_},
    //                                     stk::mesh::ConstPartVector{block1_part_ptr_});
    // bulk_data_ptr_->change_entity_parts(entities_to_move_to_block_3, stk::mesh::ConstPartVector{block3_part_ptr_},
    //                                     stk::mesh::ConstPartVector{block1_part_ptr_});
    // bulk_data_ptr_->modification_end();

    rough_randomize_field(*bulk_data_ptr_, *field1_ptr_, block1_selector_ | block2_selector_);
    rough_randomize_field(*bulk_data_ptr_, *field2_ptr_, block1_selector_ | block2_selector_);
    rough_randomize_field(*bulk_data_ptr_, *field3_ptr_, block1_selector_ | block2_selector_ | block3_selector_);
  }

  void setup() {
    const size_t num_hexes_per_dim = 50;
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_three_field_N_hex_mesh(num_hexes_per_dim, stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
  }

  void reset() {
    bulk_data_ptr_.reset();
    meta_data_ptr_.reset();
  }

 protected:
  stk::mesh::EntityRank entity_rank_;
  size_t num_hexes_per_dim_;
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

  const stk::mesh::FieldBase* field1_base_ptr_;
  const stk::mesh::FieldBase* field2_base_ptr_;
  const stk::mesh::FieldBase* field3_base_ptr_;

  stk::mesh::Part* block1_part_ptr_;
  stk::mesh::Part* block2_part_ptr_;
  stk::mesh::Part* block3_part_ptr_;

  stk::mesh::Selector block1_selector_;
  stk::mesh::Selector block2_selector_;
  stk::mesh::Selector block3_selector_;
};  // class PerfTestFieldBLAS

class FetchNgpObjTest : public PerfTestFieldBLAS {
 public:
  explicit FetchNgpObjTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = false;
  static constexpr bool has_our_stk_test = false;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      [[maybe_unused]] auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(get_bulk());
      [[maybe_unused]] stk::mesh::NgpField<double>& ngp_field1 = stk::mesh::get_updated_ngp_field<double>(*field1_ptr_);
      [[maybe_unused]] stk::mesh::NgpField<double>& ngp_field2 = stk::mesh::get_updated_ngp_field<double>(*field2_ptr_);
      [[maybe_unused]] stk::mesh::NgpField<double>& ngp_field3 = stk::mesh::get_updated_ngp_field<double>(*field3_ptr_);
    }
  }
};  // class FetchNgpObjTest

class FieldFillTest : public PerfTestFieldBLAS {
 public:
  explicit FieldFillTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = true;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  static constexpr double fill_value = 3.14159;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_fill(fill_value, *field1_ptr_, selector, exec_space);
    }
  }

  void run_stk_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_fill(fill_value, *field1_base_ptr_, selector, exec_space);
    }
  }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_fill(fill_value, *field1_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    double fill_value_local = fill_value;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_fill, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &fill_value_local]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
              raw_field_data[i] = fill_value_local;
            }
          });
    }
  }
};  // class FieldFillTest

class FieldCopyTest : public PerfTestFieldBLAS {
 public:
  explicit FieldCopyTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = true;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_copy<double>(*field1_ptr_, *field2_ptr_, selector, exec_space);
    }
  }

  void run_stk_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_copy(*field1_base_ptr_, *field2_base_ptr_, selector, exec_space);
    }
  }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_copy(*field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field1 = *field1_ptr_;
    DoubleField& field2 = *field2_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_copy, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(
          bulk, field1.entity_rank(), selector,
          [&field1, &field2]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field1, entity);
            double* raw_field1_data = stk::mesh::field_data(field1, entity);
            double* raw_field2_data = stk::mesh::field_data(field2, entity);
            for (int i = 0; i < num_components; ++i) {
              raw_field2_data[i] = raw_field1_data[i];
            }
          });
    }
  }
};  // class FieldCopyTest

class FieldSwapTest : public PerfTestFieldBLAS {
 public:
  explicit FieldSwapTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = false;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_swap<double>(*field1_ptr_, *field2_ptr_, selector, exec_space);
    }
  }

  // TODO(palmerb4): Implemented after 16.0.0. Uncomment when available.
  // void run_stk_ngp(const size_t num_iterations = 1) {
  //   const auto& exec_space = stk::ngp::ExecSpace();
  //   const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
  //   for (size_t i = 0; i < num_iterations; ++i) {
  //     stk::mesh::field_swap(get_bulk(), *field1_base_ptr_, *field2_base_ptr_, selector, exec_space);
  //   }
  // }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_swap(*field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field1 = *field1_ptr_;
    DoubleField& field2 = *field2_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_swap, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(
          bulk, field1.entity_rank(), selector,
          [&field1, &field2]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field1, entity);
            double* raw_field1_data = stk::mesh::field_data(field1, entity);
            double* raw_field2_data = stk::mesh::field_data(field2, entity);
            for (int i = 0; i < num_components; ++i) {
              const double temp = raw_field1_data[i];
              raw_field1_data[i] = raw_field2_data[i];
              raw_field2_data[i] = temp;
            }
          });
    }
  }
};  // class FieldSwapTest

class FieldScaleTest : public PerfTestFieldBLAS {
 public:
  explicit FieldScaleTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  static constexpr double alpha = 3.14159;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_scale(alpha, *field1_ptr_, selector, exec_space);
    }
  }

  // TODO(palmerb4): Implemented after 16.0.0. Uncomment when available.
  // void run_stk_ngp(const size_t num_iterations = 1) {
  //   const auto& exec_space = stk::ngp::ExecSpace();
  //   const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
  //   for (size_t i = 0; i < num_iterations; ++i) {
  //     stk::mesh::field_scale(get_bulk(), alpha, *field1_base_ptr_, selector, exec_space);
  //   }
  // }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_scale(alpha, *field1_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_scale, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
              raw_field_data[i] *= alpha;
            }
          });
    }
  }
};  // class FieldScaleTest

class FieldProductTest : public PerfTestFieldBLAS {
 public:
  explicit FieldProductTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_product<double>(*field1_ptr_, *field2_ptr_, *field3_ptr_, selector, exec_space);
    }
  }

  // TODO(palmerb4): Implemented after 16.0.0. Uncomment when available.
  // void run_stk_ngp(const size_t num_iterations = 1) {
  //   const auto& exec_space = stk::ngp::ExecSpace();
  //   const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
  //   for (size_t i = 0; i < num_iterations; ++i) {
  //     stk::mesh::field_product(get_bulk(), *field1_base_ptr_, *field2_base_ptr_, *field3_ptr_, selector,
  //     exec_space);
  //   }
  // }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_product(*field1_ptr_, *field2_ptr_, *field3_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field1 = *field1_ptr_;
    DoubleField& field2 = *field2_ptr_;
    DoubleField& field3 = *field3_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_product, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(bulk, field1.entity_rank(), selector,
                                     [&field1, &field2, &field3]([[maybe_unused]] const stk::mesh::BulkData& bulk,
                                                                 const stk::mesh::Entity entity) {
                                       const int num_components = stk::mesh::field_scalars_per_entity(field1, entity);
                                       double* raw_field1_data = stk::mesh::field_data(field1, entity);
                                       double* raw_field2_data = stk::mesh::field_data(field2, entity);
                                       double* raw_field3_data = stk::mesh::field_data(field3, entity);
                                       for (int i = 0; i < num_components; ++i) {
                                         raw_field3_data[i] = raw_field1_data[i] * raw_field2_data[i];
                                       }
                                     });
    }
  }
};  // class FieldProductTest

class FieldAxpyTest : public PerfTestFieldBLAS {
 public:
  explicit FieldAxpyTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  static constexpr double alpha = 3.14159;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_axpy(alpha, *field1_ptr_, *field2_ptr_, selector, exec_space);
    }
  }

  // TODO(palmerb4): Implemented after 16.0.0. Uncomment when available.
  // void run_stk_ngp(const size_t num_iterations = 1) {
  //   const auto& exec_space = stk::ngp::ExecSpace();
  //   const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
  //   for (size_t i = 0; i < num_iterations; ++i) {
  //     stk::mesh::field_axpy(get_bulk(), alpha, *field1_base_ptr_, *field2_base_ptr_, selector, exec_space);
  //   }
  // }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_axpy(alpha, *field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field1 = *field1_ptr_;
    DoubleField& field2 = *field2_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_axpy, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(
          bulk, field1.entity_rank(), selector,
          [&field1, &field2]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field1, entity);
            double* raw_field1_data = stk::mesh::field_data(field1, entity);
            double* raw_field2_data = stk::mesh::field_data(field2, entity);
            for (int i = 0; i < num_components; ++i) {
              raw_field2_data[i] += alpha * raw_field1_data[i];
            }
          });
    }
  }
};  // class FieldAxpyTest

class FieldAxpbyTest : public PerfTestFieldBLAS {
 public:
  explicit FieldAxpbyTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  static constexpr double alpha = 3.14159;
  static constexpr double beta = 0.3333333;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_axpby(alpha, *field1_ptr_, beta, *field2_ptr_, selector, exec_space);
    }
  }

  // TODO(palmerb4): Implemented after 16.0.0. Uncomment when available.
  // void run_stk_ngp(const size_t num_iterations = 1) {
  //   const auto& exec_space = stk::ngp::ExecSpace();
  //   const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
  //   for (size_t i = 0; i < num_iterations; ++i) {
  //     stk::mesh::field_axpby(get_bulk(), alpha, *field1_base_ptr_, beta, *field2_base_ptr_, selector, exec_space);
  //   }
  // }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      stk::mesh::field_axpby(alpha, *field1_ptr_, beta, *field2_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field1 = *field1_ptr_;
    DoubleField& field2 = *field2_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_axpby, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(
          bulk, field1.entity_rank(), selector,
          [&field1, &field2]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field1, entity);
            double* raw_field1_data = stk::mesh::field_data(field1, entity);
            double* raw_field2_data = stk::mesh::field_data(field2, entity);
            for (int i = 0; i < num_components; ++i) {
              raw_field2_data[i] = alpha * raw_field1_data[i] + beta * raw_field2_data[i];
            }
          });
    }
  }
};  // class FieldAxpbyTest

class FieldAxpbyzTest : public PerfTestFieldBLAS {
 public:
  explicit FieldAxpbyzTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = false;
  static constexpr bool has_our_stk_test = true;

  static constexpr double alpha = 3.14159;
  static constexpr double beta = 0.3333333;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      field_axpbyz(alpha, *field1_ptr_, beta, *field2_ptr_, *field3_ptr_, selector, exec_space);
    }
  }

  // TODO(palmerb4): Implemented after 16.0.0. Uncomment when available.
  // void run_stk_ngp(const size_t num_iterations = 1) {
  //   const auto& exec_space = stk::ngp::ExecSpace();
  //   const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
  //   for (size_t i = 0; i < num_iterations; ++i) {
  //     stk::mesh::field_axpbyz(get_bulk(), alpha, *field1_base_ptr_, beta, *field2_base_ptr_, *field3_ptr_,
  //     selector,
  //                           exec_space);
  //   }
  // }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field1 = *field1_ptr_;
    DoubleField& field2 = *field2_ptr_;
    DoubleField& field3 = *field3_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_axpbyz, write if from scratch using a host for_each_entity_run loop
      stk::mesh::for_each_entity_run(bulk, field1.entity_rank(), selector,
                                     [&field1, &field2, &field3]([[maybe_unused]] const stk::mesh::BulkData& bulk,
                                                                 const stk::mesh::Entity entity) {
                                       const int num_components = stk::mesh::field_scalars_per_entity(field1, entity);
                                       double* raw_field1_data = stk::mesh::field_data(field1, entity);
                                       double* raw_field2_data = stk::mesh::field_data(field2, entity);
                                       double* raw_field3_data = stk::mesh::field_data(field3, entity);
                                       for (int i = 0; i < num_components; ++i) {
                                         raw_field3_data[i] = alpha * raw_field1_data[i] + beta * raw_field2_data[i];
                                       }
                                     });
    }
  }
};  // class FieldAxpbyzTest

class FieldDotTest : public PerfTestFieldBLAS {
 public:
  explicit FieldDotTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_dot = field_dot<double>(*field1_ptr_, *field2_ptr_, selector, exec_space);
    }
  }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_dot = stk::mesh::field_dot(*field1_ptr_, *field2_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field1 = *field1_ptr_;
    DoubleField& field2 = *field2_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_dot, write if from scratch using a host for_each_entity_run loop
      double local_dot = 0.0;
      stk::mesh::for_each_entity_run(bulk, field1.entity_rank(), selector,
                                     [&field1, &field2, &local_dot]([[maybe_unused]] const stk::mesh::BulkData& bulk,
                                                                    const stk::mesh::Entity entity) {
                                       const int num_components = stk::mesh::field_scalars_per_entity(field1, entity);
                                       const double* raw_field1_data = stk::mesh::field_data(field1, entity);
                                       const double* raw_field2_data = stk::mesh::field_data(field2, entity);
                                       for (int i = 0; i < num_components; ++i) {
#pragma omp atomic
                                         local_dot += raw_field1_data[i] * raw_field2_data[i];
                                       }
                                     });

      // MPI reduction to get the global dot product
      double global_dot = 0;
      stk::all_reduce_sum(bulk.parallel(), &local_dot, &global_dot, 1);
    }
  }

};  // class FieldDotTest

class FieldNorm2Test : public PerfTestFieldBLAS {
 public:
  explicit FieldNorm2Test(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_nrm2 = field_nrm2<double>(*field1_ptr_, selector, exec_space);
    }
  }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_nrm2 = stk::mesh::field_nrm2(*field1_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_nrm2, write if from scratch using a host for_each_entity_run loop
      double local_nrm2 = 0.0;
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &local_nrm2]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            const double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
#pragma omp atomic
              local_nrm2 += raw_field_data[i] * raw_field_data[i];
            }
          });

      // MPI reduction to get the global norm2
      double global_nrm2 = 0;
      stk::all_reduce_sum(bulk.parallel(), &local_nrm2, &global_nrm2, 1);
      global_nrm2 = std::sqrt(global_nrm2);
    }
  }
};  // class FieldNorm2Test

class FieldSumTest : public PerfTestFieldBLAS {
 public:
  explicit FieldSumTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = true;
  static constexpr bool has_stk_test = false;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_sum = field_sum<double>(*field1_ptr_, selector, exec_space);
    }
  }

  void run_stk_ngp(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(get_bulk());
    stk::mesh::NgpField<double>& ngp_field1 = stk::mesh::get_updated_ngp_field<double>(*field1_ptr_);
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_sum = stk::mesh::get_field_sum(ngp_mesh, ngp_field1, selector);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_sum, write if from scratch using a host for_each_entity_run loop
      double local_sum = 0.0;
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &local_sum]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            const double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
#pragma omp atomic
              local_sum += raw_field_data[i];
            }
          });

      // MPI reduction to get the global sum
      double global_sum = 0;
      stk::all_reduce_sum(bulk.parallel(), &local_sum, &global_sum, 1);
    }
  }
};  // class FieldSumTest

class FieldAbsSumTest : public PerfTestFieldBLAS {
 public:
  explicit FieldAbsSumTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_asum = field_asum<double>(*field1_ptr_, selector, exec_space);
    }
  }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_asum = stk::mesh::field_asum<double>(*field1_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_asum, write if from scratch using a host for_each_entity_run loop
      double local_asum = 0.0;
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &local_asum]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            const double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
#pragma omp atomic
              local_asum += std::abs(raw_field_data[i]);
            }
          });

      // MPI reduction to get the global abs sum
      double global_asum = 0;
      stk::all_reduce_sum(bulk.parallel(), &local_asum, &global_asum, 1);
    }
  }
};  // class FieldAbsSumTest

class FieldMaxTest : public PerfTestFieldBLAS {
 public:
  explicit FieldMaxTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = true;
  static constexpr bool has_stk_test = false;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_max = field_max<double>(*field1_ptr_, selector, exec_space);
    }
  }

  void run_stk_ngp(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(get_bulk());
    stk::mesh::NgpField<double>& ngp_field1 = stk::mesh::get_updated_ngp_field<double>(*field1_ptr_);
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_max = stk::mesh::get_field_max(ngp_mesh, ngp_field1, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      // Instead of using stk's field_max, write if from scratch using a host for_each_entity_run loop
      double local_max = -std::numeric_limits<double>::max();
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &local_max]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            const double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
              local_max = std::max(local_max, raw_field_data[i]);
            }
          });

      // MPI reduction to get the global max
      double global_max = 0;
      stk::all_reduce_max(bulk.parallel(), &local_max, &global_max, 1);
    }
  }
};  // class FieldMaxTest

class FieldAbsMaxTest : public PerfTestFieldBLAS {
 public:
  explicit FieldAbsMaxTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_amax = field_amax<double>(*field1_ptr_, selector, exec_space);
    }
  }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_amax = stk::mesh::field_amax(*field1_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double local_amax = 0.0;
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &local_amax]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            const double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
              local_amax = std::max(local_amax, std::abs(raw_field_data[i]));
            }
          });

      // MPI reduction to get the global abs max
      double global_amax = 0;
      stk::all_reduce_max(bulk.parallel(), &local_amax, &global_amax, 1);
    }
  }
};  // class FieldAbsMaxTest

class FieldMinTest : public PerfTestFieldBLAS {
 public:
  explicit FieldMinTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = true;
  static constexpr bool has_stk_test = false;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_min = field_min<double>(*field1_ptr_, selector, exec_space);
    }
  }

  void run_stk_ngp(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(get_bulk());
    stk::mesh::NgpField<double>& ngp_field1 = stk::mesh::get_updated_ngp_field<double>(*field1_ptr_);
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_min = stk::mesh::get_field_min(ngp_mesh, ngp_field1, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double local_min = std::numeric_limits<double>::max();
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &local_min]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            const double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
              local_min = std::min(local_min, raw_field_data[i]);
            }
          });

      // MPI reduction to get the global min
      double global_min = 0;
      stk::all_reduce_min(bulk.parallel(), &local_min, &global_min, 1);
    }
  }
};  // class FieldMinTest

class FieldAbsMinTest : public PerfTestFieldBLAS {
 public:
  FieldAbsMinTest(const stk::mesh::EntityRank& entity_rank) : PerfTestFieldBLAS(entity_rank) {
  }

  static constexpr bool has_mundy_ngp_test = true;
  static constexpr bool has_stk_ngp_test = false;
  static constexpr bool has_stk_test = true;
  static constexpr bool has_our_stk_test = true;

  void run_mundy_ngp(const size_t num_iterations = 1) {
    const auto& exec_space = stk::ngp::ExecSpace();
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double ngp_amin = field_amin<double>(*field1_ptr_, selector, exec_space);
    }
  }

  void run_stk(const size_t num_iterations = 1) {
    for (size_t i = 0; i < num_iterations; ++i) {
      double stk_amin = stk::mesh::field_amin(*field1_ptr_, block1_selector_ - block2_selector_);
    }
  }

  void run_our_stk_test(const size_t num_iterations = 1) {
    const stk::mesh::Selector selector = block1_selector_ - block2_selector_;
    const stk::mesh::BulkData& bulk = get_bulk();
    DoubleField& field = *field1_ptr_;
    for (size_t i = 0; i < num_iterations; ++i) {
      double local_amin = std::numeric_limits<double>::max();
      stk::mesh::for_each_entity_run(
          bulk, field.entity_rank(), selector,
          [&field, &local_amin]([[maybe_unused]] const stk::mesh::BulkData& bulk, const stk::mesh::Entity entity) {
            const int num_components = stk::mesh::field_scalars_per_entity(field, entity);
            const double* raw_field_data = stk::mesh::field_data(field, entity);
            for (int i = 0; i < num_components; ++i) {
              local_amin = std::min(local_amin, std::abs(raw_field_data[i]));
            }
          });

      // MPI reduction to get the global abs min
      double global_amin = 0;
      stk::all_reduce_min(bulk.parallel(), &local_amin, &global_amin, 1);
    }
  }
};  // class FieldAbsMinTest

template <typename TestType>
inline void time_test(TestType& test, const std::string& test_name, const size_t num_iterations = 1000) {
  // Only print time to 3 digits
  std::cout.precision(3);
  std::cout << test_name << ":   \t (STK, STK NGP, Mundy NGP, Our STK) = ";
  test.setup();

  if constexpr (TestType::has_stk_test) {
    Kokkos::Timer timer;
    test.run_stk(num_iterations);
    const double average_time = static_cast<double>(timer.seconds()) / static_cast<double>(num_iterations);
    std::cout << "(" << average_time << ", ";
  } else {
    std::cout << "(        , ";
  }

  if constexpr (TestType::has_stk_ngp_test) {
    Kokkos::Timer timer;
    test.run_stk_ngp(num_iterations);
    const double average_time = static_cast<double>(timer.seconds()) / static_cast<double>(num_iterations);
    std::cout << average_time << ", ";
  } else {
    std::cout << "        , ";
  }

  if constexpr (TestType::has_mundy_ngp_test) {
    Kokkos::Timer timer;
    test.run_mundy_ngp(num_iterations);
    const double average_time = static_cast<double>(timer.seconds()) / static_cast<double>(num_iterations);
    std::cout << average_time << ", ";
  } else {
    std::cout << "        , ";
  }

  if constexpr (TestType::has_our_stk_test) {
    Kokkos::Timer timer;
    test.run_our_stk_test(num_iterations);
    const double average_time = static_cast<double>(timer.seconds()) / static_cast<double>(num_iterations);
    std::cout << average_time << ") seconds" << std::endl;
  } else {
    std::cout << "         ) seconds" << std::endl;
  }

  // Reset precision
  std::cout.precision(6);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {
    for (auto& [entity_rank, entity_rank_name] : {std::pair{stk::topology::NODE_RANK, std::string("NODE")},
                                                  std::pair{stk::topology::ELEM_RANK, std::string("ELEM")}}) {
      mundy::mesh::FetchNgpObjTest t1(entity_rank);
      mundy::mesh::time_test(t1, std::string("FetchNgpObj") + entity_rank_name);

      mundy::mesh::FieldFillTest t2(entity_rank);
      mundy::mesh::time_test(t2, std::string("FieldFill") + entity_rank_name);

      mundy::mesh::FieldCopyTest t3(entity_rank);
      mundy::mesh::time_test(t3, std::string("FieldCopy") + entity_rank_name);

      mundy::mesh::FieldSwapTest t4(entity_rank);
      mundy::mesh::time_test(t4, std::string("FieldSwap") + entity_rank_name);

      mundy::mesh::FieldScaleTest t5(entity_rank);
      mundy::mesh::time_test(t5, std::string("FieldScale") + entity_rank_name);

      mundy::mesh::FieldProductTest t6(entity_rank);
      mundy::mesh::time_test(t6, std::string("FieldProduct") + entity_rank_name);

      mundy::mesh::FieldAxpyTest t7(entity_rank);
      mundy::mesh::time_test(t7, std::string("FieldAxpy") + entity_rank_name);

      mundy::mesh::FieldAxpbyTest t8(entity_rank);
      mundy::mesh::time_test(t8, std::string("FieldAxpby") + entity_rank_name);

      mundy::mesh::FieldAxpbyzTest t9(entity_rank);
      mundy::mesh::time_test(t9, std::string("FieldAxpbyz") + entity_rank_name);

      mundy::mesh::FieldDotTest t10(entity_rank);
      mundy::mesh::time_test(t10, std::string("FieldDot") + entity_rank_name);

      mundy::mesh::FieldNorm2Test t11(entity_rank);
      mundy::mesh::time_test(t11, std::string("FieldNorm2") + entity_rank_name);

      mundy::mesh::FieldSumTest t12(entity_rank);
      mundy::mesh::time_test(t12, std::string("FieldSum") + entity_rank_name);

      mundy::mesh::FieldAbsSumTest t13(entity_rank);
      mundy::mesh::time_test(t13, std::string("FieldAbsSum") + entity_rank_name);

      mundy::mesh::FieldMaxTest t14(entity_rank);
      mundy::mesh::time_test(t14, std::string("FieldMax") + entity_rank_name);

      mundy::mesh::FieldAbsMaxTest t15(entity_rank);
      mundy::mesh::time_test(t15, std::string("FieldAbsMax") + entity_rank_name);

      mundy::mesh::FieldMinTest t16(entity_rank);
      mundy::mesh::time_test(t16, std::string("FieldMin") + entity_rank_name);

      mundy::mesh::FieldAbsMinTest t17(entity_rank);
      mundy::mesh::time_test(t17, std::string("FieldAbsMin") + entity_rank_name);
    }
  }

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}

#else

int main() {
  std::cout << "TEST DISABLED. Trilinos version must be at least 16.0.0." << std::endl;
  return 0;
}

#endif  // TRILINOS_MAJOR_MINOR_VERSION