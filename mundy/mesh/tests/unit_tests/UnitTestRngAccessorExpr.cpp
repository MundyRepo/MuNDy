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
#include <mundy_mesh/ForEachEntity.hpp>

namespace mundy {

namespace mesh {

namespace {


void seed_by_entity_id(const stk::mesh::BulkData& bulk_data, const stk::mesh::Field<size_t> &seed_field, const stk::mesh::EntityRank& rank, const stk::mesh::Selector &selector) {
  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  auto ngp_seed_field = stk::mesh::get_updated_ngp_field<size_t>(seed_field);
  ::mundy::mesh::for_each_entity_run(ngp_mesh, rank, selector,
                                 KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& fmi) {
                                   stk::mesh::Entity e = ngp_mesh.get_entity(rank, fmi);
                                   ngp_seed_field(fmi, 0) = static_cast<size_t>(ngp_mesh.identifier(e));
                                 });
  ngp_seed_field.modify_on_device();
}

class UnitTestRngAccessorExprFixture : public ::testing::Test {
 public:
  using DoubleField = stk::mesh::Field<double>;
  using CoordinateFunc = std::function<std::vector<double>(const double*)>;

  UnitTestRngAccessorExprFixture()
      : communicator_(MPI_COMM_WORLD),
        spatial_dimension_(3),
        entity_rank_names_({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"}) {
  }

  virtual ~UnitTestRngAccessorExprFixture() {
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

  template<typename Scalar>
  stk::mesh::Field<Scalar>* create_field_on_parts(const std::string& field_name, const stk::mesh::EntityRank& entity_rank,
                                     const int& num_components, const stk::mesh::PartVector& parts) {
    stk::mesh::Field<Scalar>& field = meta_data_ptr_->declare_field<Scalar>(entity_rank, field_name);
    for (stk::mesh::Part* part : parts) {
      stk::mesh::put_field_on_mesh(field, *part, num_components, nullptr);
    }
    return &field;
  }

  void reset_field_values() {
    stk::mesh::Selector all_blocks = block1_selector_ | block2_selector_ | block3_selector_;
    seed_by_entity_id(get_bulk(), *seed_field_ptr_, stk::topology::NODE_RANK, all_blocks);
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
   
    unsigned scalars_per_entity = 1;

    stk::mesh::PartVector all_blocks = {block1_part_ptr_, block2_part_ptr_, block3_part_ptr_};
    seed_field_ptr_ = create_field_on_parts<size_t>("seed_field", stk::topology::NODE_RANK, scalars_per_entity,
                                         all_blocks);
    counter_field_ptr_ = create_field_on_parts<size_t>("counter_field", stk::topology::NODE_RANK, scalars_per_entity,
                                         all_blocks);
    lower_bound_field_ptr_ = create_field_on_parts<double>("lower_bound_field", stk::topology::NODE_RANK, scalars_per_entity,
                                         all_blocks);
    upper_bound_field_ptr_ = create_field_on_parts<double>("upper_bound_field", stk::topology::NODE_RANK, scalars_per_entity,
                                          all_blocks);
    
    double_field_ptr_ = create_field_on_parts<double>("double_field", stk::topology::NODE_RANK, scalars_per_entity,
                                          all_blocks);
    float_field_ptr_ = create_field_on_parts<float>("float_field", stk::topology::NODE_RANK, scalars_per_entity,
                                          all_blocks);
    int_field_ptr_ = create_field_on_parts<int>("int_field", stk::topology::NODE_RANK, scalars_per_entity,
                                          all_blocks);      

    scratch_double_field_ptr_ = create_field_on_parts<double>("scratch_double_field", stk::topology::NODE_RANK, scalars_per_entity,
                                          all_blocks);
    scratch_float_field_ptr_ = create_field_on_parts<float>("scratch_float_field", stk::topology::NODE_RANK, scalars_per_entity,
                                          all_blocks);  
    scratch_int_field_ptr_ = create_field_on_parts<int>("scratch_int_field", stk::topology::NODE_RANK, scalars_per_entity,
                                          all_blocks);    
   
    // Extra fields to add heterogeneity
    field1_ptr_ = create_field_on_parts<double>("field1", stk::topology::NODE_RANK, scalars_per_entity, {block1_part_ptr_});
    field2_ptr_ = create_field_on_parts<double>("field2", stk::topology::NODE_RANK, scalars_per_entity, {block2_part_ptr_});
    field3_ptr_ = create_field_on_parts<double>("field3", stk::topology::NODE_RANK, scalars_per_entity, {block3_part_ptr_});

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

  stk::mesh::Field<size_t>* seed_field_ptr_;
  stk::mesh::Field<size_t>* counter_field_ptr_;
  stk::mesh::Field<double>* lower_bound_field_ptr_;
  stk::mesh::Field<double>* upper_bound_field_ptr_;
  stk::mesh::Field<double>* double_field_ptr_;
  stk::mesh::Field<float>* float_field_ptr_;
  stk::mesh::Field<int>* int_field_ptr_;
  stk::mesh::Field<double>* scratch_double_field_ptr_;
  stk::mesh::Field<float>* scratch_float_field_ptr_;
  stk::mesh::Field<int>* scratch_int_field_ptr_;
  stk::mesh::Field<double>* field1_ptr_;
  stk::mesh::Field<double>* field2_ptr_;
  stk::mesh::Field<double>* field3_ptr_;

  stk::mesh::Part* block1_part_ptr_;
  stk::mesh::Part* block2_part_ptr_;
  stk::mesh::Part* block3_part_ptr_;

  stk::mesh::Selector block1_selector_;
  stk::mesh::Selector block2_selector_;
  stk::mesh::Selector block3_selector_;
};  // class UnitTestRngAccessorExprFixture

struct XTag;
struct YTag;
struct ZTag;
struct DoubleFieldTag;
struct FloatFieldTag;
struct IntFieldTag;
struct SeedFieldTag;
struct CounterFieldTag;
struct LowerBoundFieldTag;
struct UpperBoundFieldTag;
struct ScratchDoubleFieldTag;
struct ScratchFloatFieldTag;
struct ScratchIntFieldTag;

template<bool use_seed_expr, bool use_counter_expr>
void randomize_test(stk::mesh::BulkData& bulk_data,
                    stk::mesh::Field<size_t>& seed_field,
                    stk::mesh::Field<size_t>& counter_field,
                    stk::mesh::Field<double>& double_field,
                    stk::mesh::Field<float>& float_field,
                    stk::mesh::Field<int>& int_field,
                    stk::mesh::Selector& selector) {
  auto seed = make_tagged_component<SeedFieldTag, stk::topology::NODE_RANK>(ScalarFieldComponent(seed_field));
  auto counter = make_tagged_component<CounterFieldTag, stk::topology::NODE_RANK>(ScalarFieldComponent(counter_field));
  auto double_accessor = make_tagged_component<DoubleFieldTag, stk::topology::NODE_RANK>(ScalarFieldComponent(double_field));
  auto float_accessor = make_tagged_component<FloatFieldTag, stk::topology::NODE_RANK>(ScalarFieldComponent(float_field));
  auto int_accessor = make_tagged_component<IntFieldTag, stk::topology::NODE_RANK>(ScalarFieldComponent(int_field));

  
  size_t fixed_seed = 11235;
  size_t fixed_counter = 98765;
  auto es = make_entity_expr(bulk_data, selector, stk::topology::NODE_RANK);
  if constexpr (use_seed_expr && use_counter_expr) {
    auto our_rng = rng(seed(es), counter(es));
    double_accessor(es) = our_rng.template rand<double>();
    float_accessor(es) = our_rng.template rand<float>();
    int_accessor(es) = our_rng.template rand<int>();
  } else if constexpr (use_seed_expr && !use_counter_expr) {
    auto our_rng = rng(seed(es), fixed_counter);
    double_accessor(es) = our_rng.template rand<double>();
    float_accessor(es) = our_rng.template rand<float>();
    int_accessor(es) = our_rng.template rand<int>();
  } else if constexpr (!use_seed_expr && use_counter_expr) {
    auto our_rng = rng(fixed_seed, counter(es));
    double_accessor(es) = our_rng.template rand<double>();
    float_accessor(es) = our_rng.template rand<float>();
    int_accessor(es) = our_rng.template rand<int>();
  } else {
    static_assert(use_seed_expr != false && use_counter_expr != false, "Test will not compile unless at least one of use_seed_expr or use_counter_expr is true.");
  }
  
  double_accessor.sync_to_host();
  float_accessor.sync_to_host();
  int_accessor.sync_to_host();

  ::mundy::mesh::for_each_entity_run(bulk_data, stk::topology::NODE_RANK, selector, 
    [&double_accessor, &float_accessor, &int_accessor, &seed, &counter, fixed_seed, fixed_counter](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity& e) {
      size_t local_seed = use_seed_expr ? seed(e) : fixed_seed;
      size_t local_counter = use_counter_expr ? counter(e) : fixed_counter;

      openrand::Philox rng_d(local_seed, local_counter);
      double actual_value_d = double_accessor(e);
      double expected_value_d = rng_d.rand<double>();
      EXPECT_DOUBLE_EQ(actual_value_d, expected_value_d); 

      openrand::Philox rng_f(local_seed, local_counter);
      float actual_value_f = float_accessor(e);
      float expected_value_f = rng_f.rand<float>();
      EXPECT_FLOAT_EQ(actual_value_f, expected_value_f);

      openrand::Philox rng_i(local_seed, local_counter);
      int actual_value_i = int_accessor(e);
      int expected_value_i = rng_i.rand<int>();
      EXPECT_EQ(actual_value_i, expected_value_i);
    });
}

TEST_F(UnitTestRngAccessorExprFixture, randomize_field) {
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
  randomize_test<true, true>(get_bulk(), *seed_field_ptr_, *counter_field_ptr_,
                             *double_field_ptr_, *float_field_ptr_, *int_field_ptr_,
                             b1_not_b2);

  randomize_test<true, false>(get_bulk(), *seed_field_ptr_, *counter_field_ptr_,
                              *double_field_ptr_, *float_field_ptr_, *int_field_ptr_,
                              b1_not_b2);

  randomize_test<false, true>(get_bulk(), *seed_field_ptr_, *counter_field_ptr_,
                              *double_field_ptr_, *float_field_ptr_, *int_field_ptr_,
                              b1_not_b2);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
