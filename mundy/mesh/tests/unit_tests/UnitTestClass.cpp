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

// External
#include <gtest/gtest.h>
#include <mpi.h>

// C++ core
#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// STK
#include <stk_io/FillMesh.hpp>
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

// Mundy
#include <mundy_mesh/Class.hpp>
#include <mundy_mesh/Component.hpp>
#include <mundy_mesh/FieldComponent.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/SharedComponent.hpp>

namespace mundy {

namespace mesh {

namespace {

struct ParticleClass {
  Class* class_ptr = nullptr;
  stk::mesh::Field<int>* elem_int_field = nullptr;

  void initialize(stk::mesh::MetaData& meta_data, const std::string& class_name) {
    class_ptr = &declare_class(meta_data, class_name, stk::topology::PARTICLE);

    elem_int_field = &meta_data.declare_field<int>(stk::topology::ELEM_RANK, class_name + "_elem_field0");
    stk::io::set_field_role(*elem_int_field, Ioss::Field::TRANSIENT);

    put_field_on_mesh(*elem_int_field, class_instance(), 1, nullptr);
  }

  Class& class_instance() {
    MUNDY_THROW_REQUIRE(class_ptr != nullptr, std::logic_error, "Class pointer is not initialized.");
    return *class_ptr;
  }

  const Class& class_instance() const {
    MUNDY_THROW_REQUIRE(class_ptr != nullptr, std::logic_error, "Class pointer is not initialized.");
    return *class_ptr;
  }
};

struct NodeClass {
  Class* class_ptr = nullptr;
  stk::mesh::Field<int>* node_int_field = nullptr;

  void initialize(stk::mesh::MetaData& meta_data, const std::string& class_name) {
    class_ptr = &declare_class(meta_data, class_name, stk::topology::NODE_RANK);

    node_int_field = &meta_data.declare_field<int>(stk::topology::NODE_RANK, class_name + "_node_field0");
    stk::io::set_field_role(*node_int_field, Ioss::Field::TRANSIENT);

    put_field_on_mesh(*node_int_field, class_instance(), 1, nullptr);
  }

  Class& class_instance() {
    MUNDY_THROW_REQUIRE(class_ptr != nullptr, std::logic_error, "Class pointer is not initialized.");
    return *class_ptr;
  }

  const Class& class_instance() const {
    MUNDY_THROW_REQUIRE(class_ptr != nullptr, std::logic_error, "Class pointer is not initialized.");
    return *class_ptr;
  }
};

struct ThreeClassTreeRestartMeta {
  // class0 supersets class1 and class2.
  // class2_nodes is a subset of class2
  // Nodes within class2 are components of class2's nodeset as well as class2_nodes class.
  std::shared_ptr<MetaData> meta_data;
  ParticleClass class0;
  ParticleClass class1;
  ParticleClass class2;
  NodeClass class2_nodes;

  stk::mesh::Field<double>* coords_field = nullptr;
  std::vector<Class*> particle_classes;
  std::vector<Class*> classes;
  std::vector<stk::mesh::Field<int>*> elem_int_fields;
  stk::mesh::Field<int>* node_int_field = nullptr;

  explicit ThreeClassTreeRestartMeta(MeshBuilder& mesh_builder, bool assign_assembly_ids = false,
                                     bool commit_meta_data = true) {
    meta_data = mesh_builder.create_meta_data();
    meta_data->use_simple_fields();
    meta_data->set_coordinate_field_name("coords");

    coords_field = &meta_data->declare_field<double>(stk::topology::NODE_RANK, "coords");
    stk::mesh::put_field_on_mesh(*coords_field, meta_data->universal_part(), 3, nullptr);

    class0.initialize(*meta_data, "class0");
    class1.initialize(*meta_data, "class1");
    class2.initialize(*meta_data, "class2");
    class2_nodes.initialize(*meta_data, "class2_nodes");

    class0.class_instance().declare_subset(class1.class_instance());
    class0.class_instance().declare_subset(class2.class_instance());

    particle_classes = {
        &class0.class_instance(),
        &class1.class_instance(),
        &class2.class_instance(),
    };

    classes = {
        &class0.class_instance(),
        &class1.class_instance(),
        &class2.class_instance(),
        &class2_nodes.class_instance(),
    };

    node_int_field = class2_nodes.node_int_field;
    elem_int_fields = {
        class0.elem_int_field,
        class1.elem_int_field,
        class2.elem_int_field,
    };

    if (assign_assembly_ids) {
      class0.class_instance().set_assembly_part_ids(100, 1100);
      class1.class_instance().set_assembly_part_ids(200, 1200);
      class2.class_instance().set_assembly_part_ids(201, 1201);
    }

    if (commit_meta_data) {
      meta_data->commit();
    }
  }
};

struct ElementClassBinding {
  Class* class_ptr = nullptr;

  void initialize(stk::mesh::MetaData& meta_data, const std::string& class_name) {
    class_ptr = &declare_class(meta_data, class_name, stk::topology::PARTICLE);
  }

  void declare_subset(ElementClassBinding& sub_class_binding) {
    class_instance().declare_subset(sub_class_binding.class_instance());
  }

  Class& class_instance() {
    MUNDY_THROW_REQUIRE(class_ptr != nullptr, std::logic_error, "Element class pointer is not initialized.");
    return *class_ptr;
  }

  const Class& class_instance() const {
    MUNDY_THROW_REQUIRE(class_ptr != nullptr, std::logic_error, "Element class pointer is not initialized.");
    return *class_ptr;
  }
};

struct FiveClassDiamondRestartMeta {
  std::shared_ptr<MetaData> meta_data;
  ElementClassBinding class0;
  ElementClassBinding class1;
  ElementClassBinding class2;
  ElementClassBinding class3;
  ElementClassBinding class4;

  stk::mesh::Field<double>* coords_field = nullptr;
  stk::mesh::Field<int>* node_int_field = nullptr;
  stk::mesh::Field<int>* elem_int_field = nullptr;
  stk::mesh::Field<unsigned>* node_unsigned_field = nullptr;
  std::vector<ElementClassBinding*> class_bindings;
  std::vector<Class*> classes;
  std::vector<stk::mesh::Field<int>*> node_int_fields;
  std::vector<stk::mesh::Field<int>*> elem_int_fields;
  std::vector<stk::mesh::Field<unsigned>*> node_unsigned_fields;

  explicit FiveClassDiamondRestartMeta(MeshBuilder& mesh_builder, bool commit_meta_data = true) {
    meta_data = mesh_builder.create_meta_data();
    meta_data->use_simple_fields();
    meta_data->set_coordinate_field_name("coords");

    coords_field = &meta_data->declare_field<double>(stk::topology::NODE_RANK, "coords");
    stk::mesh::put_field_on_mesh(*coords_field, meta_data->universal_part(), 3, nullptr);

    class0.initialize(*meta_data, "class0");
    class1.initialize(*meta_data, "class1");
    class2.initialize(*meta_data, "class2");
    class3.initialize(*meta_data, "class3");
    class4.initialize(*meta_data, "class4");

    class0.declare_subset(class2);
    class1.declare_subset(class2);
    class2.declare_subset(class3);
    class2.declare_subset(class4);

    node_int_field = &meta_data->declare_field<int>(stk::topology::NODE_RANK, "class2_node_field0");
    elem_int_field = &meta_data->declare_field<int>(stk::topology::ELEM_RANK, "class2_elem_field0");
    node_unsigned_field = &meta_data->declare_field<unsigned>(stk::topology::NODE_RANK, "class2_node_field2");
    stk::io::set_field_role(*elem_int_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(*node_int_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(*node_unsigned_field, Ioss::Field::TRANSIENT);

    put_field_on_mesh(*node_int_field, class2.class_instance(), 1, nullptr);
    put_field_on_mesh(*elem_int_field, class2.class_instance(), 1, nullptr);
    put_field_on_mesh(*node_unsigned_field, class2.class_instance(), 1, nullptr);
    class_bindings = {
        &class0, &class1, &class2, &class3, &class4,
    };

    for (ElementClassBinding* class_binding : class_bindings) {
      classes.push_back(&class_binding->class_instance());
    }

    elem_int_fields = {elem_int_field};
    node_int_fields = {node_int_field};
    node_unsigned_fields = {node_unsigned_field};

    if (commit_meta_data) {
      meta_data->commit();
    }
  }
};

struct BulkIoContext {
  std::unique_ptr<BulkData> bulk_data;
  stk::io::StkMeshIoBroker io_broker;

  BulkIoContext(MPI_Comm comm, MeshBuilder& mesh_builder, const std::shared_ptr<MetaData>& meta_data)
      : io_broker(comm) {
    bulk_data = mesh_builder.create_bulk_data(meta_data);
    io_broker.use_simple_fields();
    io_broker.set_bulk_data(*bulk_data);
  }
};

struct NumberedClass {
  std::string name;
  Class* class_instance = nullptr;
  stk::mesh::Field<int>* elem_field = nullptr;
  stk::mesh::Field<int>* node_field = nullptr;
  std::vector<size_t> subclasses;
};

struct CLASS_IO_COMPONENT;

template <typename FieldValueType>
void set_scalar_field_values(stk::mesh::BulkData& bulk_data, const stk::mesh::EntityRank rank,
                             const std::vector<stk::mesh::Field<FieldValueType>*>& all_fields,
                             const unsigned value_offset) {
  const stk::mesh::Selector locally_owned = bulk_data.mesh_meta_data().locally_owned_part();

  for (size_t field_idx = 0; field_idx < all_fields.size(); ++field_idx) {
    stk::mesh::Field<FieldValueType>& field = *all_fields[field_idx];
    for (const stk::mesh::Bucket* bucket : bulk_data.get_buckets(rank, locally_owned)) {
      for (const stk::mesh::Entity entity : *bucket) {
        FieldValueType* data = stk::mesh::field_data(field, entity);
        if (data != nullptr) {
          data[0] = static_cast<FieldValueType>(value_offset + (field_idx + 1) * 1000 + bulk_data.identifier(entity));
        }
      }
    }
  }
}

template <typename FieldValueType>
void expect_scalar_field_values(stk::mesh::BulkData& bulk_data, const stk::mesh::EntityRank rank,
                                const std::vector<stk::mesh::Field<FieldValueType>*>& all_fields,
                                const unsigned value_offset) {
  const stk::mesh::Selector locally_owned = bulk_data.mesh_meta_data().locally_owned_part();

  for (size_t field_idx = 0; field_idx < all_fields.size(); ++field_idx) {
    stk::mesh::Field<FieldValueType>& field = *all_fields[field_idx];
    for (const stk::mesh::Bucket* bucket : bulk_data.get_buckets(rank, locally_owned)) {
      for (const stk::mesh::Entity entity : *bucket) {
        FieldValueType* data = stk::mesh::field_data(field, entity);
        if (data != nullptr) {
          const FieldValueType expected =
              static_cast<FieldValueType>(value_offset + (field_idx + 1) * 1000 + bulk_data.identifier(entity));
          if constexpr (std::is_floating_point_v<FieldValueType>) {
            EXPECT_DOUBLE_EQ(data[0], expected);
          } else {
            EXPECT_EQ(data[0], expected);
          }
        }
      }
    }
  }
}

void declare_entity_on_class_leaf_data(stk::mesh::BulkData& bulk_data, stk::mesh::Field<double>& coords_field,
                                       const Class& class_instance, stk::mesh::EntityId& elem_count,
                                       stk::mesh::EntityId& node_count) {
  // Callers track counts from 0; STK entity IDs are 1-based.
  const stk::mesh::EntityId elem_id = elem_count + 1;
  const stk::mesh::EntityId node_id = node_count + 1;

  const stk::mesh::Entity node = class_interface(bulk_data).declare_node(node_id, class_instance);
  const stk::mesh::Entity elem = class_interface(bulk_data).declare_element(elem_id, class_instance);
  bulk_data.declare_relation(elem, node, 0u);

  double* coords = stk::mesh::field_data(coords_field, node);
  MUNDY_THROW_REQUIRE(coords != nullptr, std::runtime_error,
                      sink() << "Failed to get field data for coords field on node " << node_id);
  coords[0] = static_cast<double>(elem_id);
  coords[1] = static_cast<double>(node_id);
  coords[2] = 0.0;

  ++elem_count;
  ++node_count;
}

bool part_vector_contains(const stk::mesh::PartVector& parts, const stk::mesh::Part& part) {
  return std::find(parts.begin(), parts.end(), &part) != parts.end();
}

bool class_vector_contains(const ClassVector& classes, const Class& class_instance) {
  return std::find(classes.begin(), classes.end(), &class_instance) != classes.end();
}

class UnitTestClassFixture : public ::testing::Test {
 protected:
  void initialize_mesh_builder(MeshBuilder& mesh_builder) const {
    mesh_builder.set_spatial_dimension(3).set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  }

  std::filesystem::path prepare_output_dir(const std::string& directory_name) const {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::filesystem::path output_dir =
        std::filesystem::current_path() / ("mpi_size_" + std::to_string(size)) / directory_name;
    if (rank == 0) {
      std::filesystem::remove_all(output_dir);
      std::filesystem::create_directories(output_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return output_dir;
  }

  static std::vector<NumberedClass> declare_numbered_class_hierarchy(stk::mesh::MetaData& meta_data) {
    std::vector<NumberedClass> hierarchy;
    hierarchy.reserve(11u);

    for (size_t class_idx = 0; class_idx < 11u; ++class_idx) {
      NumberedClass class_node;
      class_node.name = "class" + std::to_string(class_idx);
      class_node.class_instance = &declare_class(meta_data, class_node.name, stk::topology::PARTICLE);
      EXPECT_EQ(class_node.class_instance->primary_entity_rank(), stk::topology::ELEM_RANK);
      hierarchy.push_back(class_node);
    }

    declare_subset(hierarchy, 0u, 1u);
    declare_subset(hierarchy, 0u, 2u);
    declare_subset(hierarchy, 1u, 3u);
    declare_subset(hierarchy, 1u, 4u);
    declare_subset(hierarchy, 2u, 5u);
    declare_subset(hierarchy, 2u, 6u);
    declare_subset(hierarchy, 3u, 7u);
    declare_subset(hierarchy, 3u, 8u);
    declare_subset(hierarchy, 7u, 9u);
    declare_subset(hierarchy, 8u, 9u);
    declare_subset(hierarchy, 4u, 10u);
    declare_subset(hierarchy, 5u, 10u);
    declare_subset(hierarchy, 6u, 10u);
    declare_subset(hierarchy, 9u, 10u);

    return hierarchy;
  }

  static void declare_scalar_fields_on_hierarchy(stk::mesh::MetaData& meta_data,
                                                 std::vector<NumberedClass>& hierarchy) {
    for (NumberedClass& class_node : hierarchy) {
      class_node.elem_field = &meta_data.declare_field<int>(stk::topology::ELEM_RANK, class_node.name + "_elem_field0");
      class_node.node_field = &meta_data.declare_field<int>(stk::topology::NODE_RANK, class_node.name + "_node_field0");

      put_field_on_mesh(*class_node.elem_field, *class_node.class_instance, 1, nullptr);
      put_field_on_mesh(*class_node.node_field, *class_node.class_instance, 1, nullptr);

      ASSERT_TRUE(class_node.class_instance->has_induced_set(stk::topology::NODE_RANK)) << class_node.name;
    }
  }

  static bool is_descendant_or_self(const std::vector<NumberedClass>& hierarchy, const size_t ancestor_idx,
                                    const size_t maybe_descendant_idx) {
    std::vector<bool> visited(hierarchy.size(), false);
    std::vector<size_t> stack{ancestor_idx};
    while (!stack.empty()) {
      const size_t current_idx = stack.back();
      stack.pop_back();
      if (visited[current_idx]) {
        continue;
      }
      visited[current_idx] = true;
      if (current_idx == maybe_descendant_idx) {
        return true;
      }
      for (const size_t subclass_idx : hierarchy[current_idx].subclasses) {
        stack.push_back(subclass_idx);
      }
    }
    return false;
  }

  static void expect_field_on_part(stk::mesh::FieldBase& field, const stk::mesh::Part& part, const bool expected) {
    const bool actual = stk::io::is_field_on_part(&field, field.entity_rank(), part);
    EXPECT_EQ(actual, expected) << "field='" << field.name() << "' part='" << part.name() << "'";
  }

  static void expect_entity_part_membership(stk::mesh::BulkData& bulk_data, const stk::mesh::Entity entity,
                                            const std::string& entity_label, const stk::mesh::Part& part,
                                            const bool expected) {
    ASSERT_TRUE(bulk_data.is_valid(entity)) << entity_label;
    const bool actual = bulk_data.bucket(entity).member(part);
    EXPECT_EQ(actual, expected) << entity_label << " part='" << part.name() << "'";
  }

  static void expect_three_class_tree_declarations(ThreeClassTreeRestartMeta& test_meta) {
    Class& class0 = test_meta.class0.class_instance();
    Class& class1 = test_meta.class1.class_instance();
    Class& class2 = test_meta.class2.class_instance();

    ASSERT_TRUE(class0.contains(class1));
    ASSERT_TRUE(class0.contains(class2));
    ASSERT_FALSE(class1.contains(class0));

    ASSERT_EQ(class0.subclasses().size(), 2u);
    EXPECT_TRUE(class_vector_contains(class0.subclasses(), class1));
    EXPECT_TRUE(class_vector_contains(class0.subclasses(), class2));
    EXPECT_TRUE(class_vector_contains(class1.superclasses(), class0));
    EXPECT_TRUE(class_vector_contains(class2.superclasses(), class0));

    EXPECT_TRUE(part_vector_contains(class0.assembly_part_subsets(), class1.assembly_part()));
    EXPECT_TRUE(part_vector_contains(class0.assembly_part_subsets(), class2.assembly_part()));
    EXPECT_TRUE(part_vector_contains(class0.data_part_subsets(), class1.data_part()));
    EXPECT_TRUE(part_vector_contains(class0.data_part_subsets(), class2.data_part()));

    for (Class* class_instance : test_meta.classes) {
      EXPECT_TRUE(class_instance->assembly_part().contains(class_instance->leaf_assembly_part()));
      EXPECT_TRUE(class_instance->leaf_assembly_part().contains(class_instance->leaf_part()));

      EXPECT_FALSE(stk::io::is_part_io_part(class_instance->data_part()));
      EXPECT_FALSE(stk::io::is_part_assembly_io_part(class_instance->data_part()));

      EXPECT_TRUE(stk::io::is_part_io_part(class_instance->leaf_part()));
      EXPECT_FALSE(stk::io::is_part_assembly_io_part(class_instance->leaf_part()));

      EXPECT_TRUE(stk::io::is_part_io_part(class_instance->assembly_part()));
      EXPECT_TRUE(stk::io::is_part_assembly_io_part(class_instance->assembly_part()));
      EXPECT_TRUE(stk::io::is_part_io_part(class_instance->leaf_assembly_part()));
      EXPECT_TRUE(stk::io::is_part_assembly_io_part(class_instance->leaf_assembly_part()));
    }

    ASSERT_EQ(class0.class_ordinal(), 0u);
    ASSERT_EQ(class1.class_ordinal(), 1u);
    ASSERT_EQ(class2.class_ordinal(), 2u);
    EXPECT_EQ(get_class(*test_meta.meta_data, 0u).name(), "class0");
    EXPECT_EQ(get_class(*test_meta.meta_data, 1u).name(), "class1");
    EXPECT_EQ(get_class(*test_meta.meta_data, 2u).name(), "class2");
  }

  static void expect_induced_sets_follow_hierarchy(std::vector<NumberedClass>& hierarchy) {
    for (const NumberedClass& parent_class : hierarchy) {
      for (const size_t child_idx : parent_class.subclasses) {
        const Class& parent_induced_set = parent_class.class_instance->get_induced_set(stk::topology::NODE_RANK);
        const Class& child_induced_set = hierarchy[child_idx].class_instance->get_induced_set(stk::topology::NODE_RANK);
        EXPECT_TRUE(parent_induced_set.contains(child_induced_set))
            << parent_induced_set.name() << " should contain " << child_induced_set.name();
      }
    }
  }

  static void expect_fields_follow_hierarchy(std::vector<NumberedClass>& hierarchy) {
    for (size_t field_owner_idx = 0; field_owner_idx < hierarchy.size(); ++field_owner_idx) {
      NumberedClass& field_owner = hierarchy[field_owner_idx];

      for (size_t target_idx = 0; target_idx < hierarchy.size(); ++target_idx) {
        NumberedClass& target = hierarchy[target_idx];
        const bool target_is_owner_or_subclass = is_descendant_or_self(hierarchy, field_owner_idx, target_idx);

        expect_field_on_part(*field_owner.elem_field, target.class_instance->data_part(), target_is_owner_or_subclass);
        expect_field_on_part(*field_owner.elem_field, target.class_instance->leaf_part(), target_is_owner_or_subclass);
        expect_field_on_part(*field_owner.node_field, target.class_instance->data_part(), false);
        expect_field_on_part(*field_owner.node_field, target.class_instance->leaf_part(), false);

        ASSERT_TRUE(target.class_instance->has_induced_set(stk::topology::NODE_RANK)) << target.name;
        Class& target_induced_set = target.class_instance->get_induced_set(stk::topology::NODE_RANK);
        ASSERT_EQ(target_induced_set.primary_entity_rank(), stk::topology::NODE_RANK);
        ASSERT_TRUE(target_induced_set.is_set());
        expect_field_on_part(*field_owner.node_field, target_induced_set.data_part(), target_is_owner_or_subclass);
        expect_field_on_part(*field_owner.node_field, target_induced_set.leaf_part(), target_is_owner_or_subclass);
      }
    }
  }

  static void expect_class_part_membership(stk::mesh::BulkData& bulk_data, const stk::mesh::Entity entity,
                                           const std::string& entity_label, const std::vector<NumberedClass>& hierarchy,
                                           const size_t declared_class_idx) {
    for (size_t class_idx = 0; class_idx < hierarchy.size(); ++class_idx) {
      const NumberedClass& class_node = hierarchy[class_idx];
      const bool in_ancestor_or_self = is_descendant_or_self(hierarchy, class_idx, declared_class_idx);
      const bool in_exact_class = class_idx == declared_class_idx;

      expect_entity_part_membership(bulk_data, entity, entity_label, class_node.class_instance->data_part(),
                                    in_ancestor_or_self);
      expect_entity_part_membership(bulk_data, entity, entity_label, class_node.class_instance->assembly_part(),
                                    in_ancestor_or_self);
      expect_entity_part_membership(bulk_data, entity, entity_label, class_node.class_instance->leaf_part(),
                                    in_exact_class);
      expect_entity_part_membership(bulk_data, entity, entity_label, class_node.class_instance->leaf_assembly_part(),
                                    in_exact_class);
    }
  }

  static void expect_induced_set_part_membership(stk::mesh::BulkData& bulk_data, const stk::mesh::Entity entity,
                                                 const std::string& entity_label, std::vector<NumberedClass>& hierarchy,
                                                 const size_t declared_class_idx,
                                                 const bool should_be_in_induced_sets) {
    for (size_t class_idx = 0; class_idx < hierarchy.size(); ++class_idx) {
      NumberedClass& class_node = hierarchy[class_idx];
      Class& induced_set = class_node.class_instance->get_induced_set(stk::topology::NODE_RANK);
      const bool in_ancestor_or_self =
          should_be_in_induced_sets && is_descendant_or_self(hierarchy, class_idx, declared_class_idx);
      const bool in_exact_induced_set = should_be_in_induced_sets && class_idx == declared_class_idx;

      expect_entity_part_membership(bulk_data, entity, entity_label, induced_set.data_part(), in_ancestor_or_self);
      expect_entity_part_membership(bulk_data, entity, entity_label, induced_set.assembly_part(), in_ancestor_or_self);
      expect_entity_part_membership(bulk_data, entity, entity_label, induced_set.leaf_part(), in_exact_induced_set);
      expect_entity_part_membership(bulk_data, entity, entity_label, induced_set.leaf_assembly_part(),
                                    in_exact_induced_set);
    }
  }

 private:
  static void declare_subset(std::vector<NumberedClass>& hierarchy, const size_t parent_idx, const size_t child_idx) {
    hierarchy[parent_idx].class_instance->declare_subset(*hierarchy[child_idx].class_instance);
    hierarchy[parent_idx].subclasses.push_back(child_idx);
  }
};

TEST_F(UnitTestClassFixture, ClassApiRejectsMismatchedRedeclarationAndPreservesConstClassView) {
  static_assert(
      std::is_same_v<decltype(get_classes(std::declval<const stk::mesh::MetaData&>())), const ConstClassVector&>);

  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();

  Class& particle_class = declare_class(*meta_data, "redeclaration_probe", stk::topology::PARTICLE);
  Class& repeated_particle_class = declare_class(*meta_data, "redeclaration_probe", stk::topology::PARTICLE);
  EXPECT_EQ(&particle_class, &repeated_particle_class);

  EXPECT_THROW(declare_class(*meta_data, "redeclaration_probe", stk::topology::NODE_RANK), std::logic_error);
  EXPECT_THROW(declare_class(*meta_data, "redeclaration_probe"), std::logic_error);
  EXPECT_THROW(declare_class(*meta_data, "redeclaration_probe", stk::topology::BEAM_2), std::logic_error);

  const stk::mesh::MetaData& const_meta_data = *meta_data;
  const ConstClassVector& const_classes = get_classes(const_meta_data);
  ASSERT_EQ(const_classes.size(), 1u);
  EXPECT_EQ(const_classes[0], &particle_class);
}

TEST_F(UnitTestClassFixture, AddClassComponentSupportsFieldBackedComponentsAndSkipsSharedComponents) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);

  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();
  meta_data->use_simple_fields();

  Class& particle_class = declare_class(*meta_data, "class_component_io_particle", stk::topology::PARTICLE);

  stk::mesh::Field<double>& direct_field =
      meta_data->declare_field<double>(stk::topology::ELEM_RANK, "class_component_direct_field");
  stk::io::set_field_role(direct_field, Ioss::Field::TRANSIENT);
  put_field_on_mesh(direct_field, particle_class, 1, nullptr);

  stk::mesh::Field<double>& tagged_field =
      meta_data->declare_field<double>(stk::topology::ELEM_RANK, "class_component_tagged_field");
  stk::io::set_field_role(tagged_field, Ioss::Field::TRANSIENT);
  put_field_on_mesh(tagged_field, particle_class, 1, nullptr);

  meta_data->commit();

  BulkIoContext writer(MPI_COMM_WORLD, mesh_builder, meta_data);
  const std::filesystem::path output_dir = prepare_output_dir("unit_test_class_component_io");
  const size_t output_index =
      writer.io_broker.create_output_mesh((output_dir / "class_component_io.exo").string(), stk::io::WRITE_RESULTS);

  ScalarFieldComponent<double> direct_component(direct_field);
  auto tagged_component = make_tagged_component<CLASS_IO_COMPONENT>(ScalarFieldComponent<double>(tagged_field));
  SharedScalarComponent<double> shared_component(1.0);

  EXPECT_NO_THROW(add_class_component(writer.io_broker, output_index, direct_component, "DIRECT_COMPONENT"));
  EXPECT_NO_THROW(add_class_component(writer.io_broker, output_index, tagged_component));
  EXPECT_NO_THROW(add_class_component(writer.io_broker, output_index, shared_component));
}

TEST_F(UnitTestClassFixture, RestartRoundTripPreservesTreeAssemblyFields) {
  const std::filesystem::path output_dir = prepare_output_dir("unit_test_class_tree_restart");
  const std::filesystem::path restart_file = output_dir / "tree_restart.e-s.0";

  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  const bool assign_assembly_ids = false;
  ThreeClassTreeRestartMeta writer_meta(mesh_builder, assign_assembly_ids, true);
  expect_three_class_tree_declarations(writer_meta);

  BulkIoContext writer(MPI_COMM_WORLD, mesh_builder, writer_meta.meta_data);
  stk::mesh::EntityId elem_count = 0;
  stk::mesh::EntityId node_count = 0;

  writer.bulk_data->modification_begin();
  for (Class* class_instance : writer_meta.particle_classes) {
    declare_entity_on_class_leaf_data(*writer.bulk_data, *writer_meta.coords_field, *class_instance, elem_count,
                                      node_count);
  }

  // Move nodes in class2 into class2_nodes
  stk::mesh::Entity class2_node = writer.bulk_data->get_entity(stk::topology::NODE_RANK, 3);
  ASSERT_TRUE(writer.bulk_data->is_valid(class2_node));
  class_interface(*writer.bulk_data)
      .change_entity_classes(class2_node, ClassVector{&writer_meta.class2_nodes.class_instance()});

  // Make a new node that is only in class2_nodes to test what happens when a node only resides in a nodeset and not in
  // a block
  const stk::mesh::EntityId new_node_id = node_count + 1;
  const stk::mesh::Entity new_node =
      class_interface(*writer.bulk_data).declare_node(new_node_id, writer_meta.class2_nodes.class_instance());
  double* coords = stk::mesh::field_data(*writer_meta.coords_field, new_node);
  MUNDY_THROW_REQUIRE(coords != nullptr, std::runtime_error,
                      sink() << "Failed to get field data for coords field on node " << new_node_id);
  coords[0] = 0.0;
  coords[1] = static_cast<double>(new_node_id);
  coords[2] = 0.0;

  writer.bulk_data->modification_end();
  ASSERT_EQ(elem_count, static_cast<stk::mesh::EntityId>(writer_meta.particle_classes.size()));
  ASSERT_EQ(node_count, static_cast<stk::mesh::EntityId>(writer_meta.particle_classes.size()));

  set_scalar_field_values(*writer.bulk_data, stk::topology::ELEM_RANK, writer_meta.elem_int_fields, 20000);

  const size_t output_index = writer.io_broker.create_output_mesh(restart_file.string(), stk::io::WRITE_RESULTS);
  for (stk::mesh::Field<int>* field : writer_meta.elem_int_fields) {
    writer.io_broker.add_field(output_index, *field);
  }

  // node_int_field is only defined on class2_nodes, so register it using the explicit-class overload.
  add_class_field(writer.io_broker, output_index, *writer_meta.node_int_field,
                  ClassVector{&writer_meta.class2_nodes.class_instance()});
  writer.io_broker.begin_output_step(output_index, 1.0);
  writer.io_broker.write_defined_output_fields(output_index);
  writer.io_broker.end_output_step(output_index);
  writer.io_broker.flush_output();
  writer.io_broker.close_output_mesh(output_index);

  MPI_Barrier(MPI_COMM_WORLD);

  ThreeClassTreeRestartMeta reader_meta(mesh_builder, assign_assembly_ids, false);
  ASSERT_FALSE(reader_meta.meta_data->is_commit());
  BulkIoContext reader(MPI_COMM_WORLD, mesh_builder, reader_meta.meta_data);
  stk::io::fill_mesh_with_fields(restart_file.string(), reader.io_broker, *reader.bulk_data, stk::io::READ_RESTART);
  expect_scalar_field_values(*reader.bulk_data, stk::topology::ELEM_RANK, reader_meta.elem_int_fields, 20000);
}

TEST_F(UnitTestClassFixture, RestartRoundTripReadsTypedFieldsFromInducedNodeSets) {
  const std::filesystem::path output_dir = prepare_output_dir("unit_test_class_induced_nodeset_restart");
  const std::filesystem::path restart_file = output_dir / "induced_nodeset_restart.e-s.0";

  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  FiveClassDiamondRestartMeta writer_meta(mesh_builder, true);

  BulkIoContext writer(MPI_COMM_WORLD, mesh_builder, writer_meta.meta_data);
  stk::mesh::EntityId elem_count = 0;
  stk::mesh::EntityId node_count = 0;

  writer.bulk_data->modification_begin();
  for (ElementClassBinding* class_binding : writer_meta.class_bindings) {
    declare_entity_on_class_leaf_data(*writer.bulk_data, *writer_meta.coords_field, class_binding->class_instance(),
                                      elem_count, node_count);
  }
  writer.bulk_data->modification_end();

  ASSERT_EQ(elem_count, static_cast<stk::mesh::EntityId>(writer_meta.class_bindings.size()));
  ASSERT_EQ(node_count, static_cast<stk::mesh::EntityId>(writer_meta.class_bindings.size()));

  set_scalar_field_values(*writer.bulk_data, stk::topology::NODE_RANK, writer_meta.node_int_fields, 10000);
  set_scalar_field_values(*writer.bulk_data, stk::topology::ELEM_RANK, writer_meta.elem_int_fields, 20000);
  set_scalar_field_values(*writer.bulk_data, stk::topology::NODE_RANK, writer_meta.node_unsigned_fields, 30000);

  const size_t output_index = writer.io_broker.create_output_mesh(restart_file.string(), stk::io::WRITE_RESULTS);
  add_class_field(writer.io_broker, output_index, *writer_meta.elem_int_field);
  add_class_field(writer.io_broker, output_index, *writer_meta.node_int_field);
  add_class_field(writer.io_broker, output_index, *writer_meta.node_unsigned_field);

  writer.io_broker.begin_output_step(output_index, 1.0);
  writer.io_broker.write_defined_output_fields(output_index);
  writer.io_broker.end_output_step(output_index);
  writer.io_broker.flush_output();
  writer.io_broker.close_output_mesh(output_index);

  MPI_Barrier(MPI_COMM_WORLD);

  FiveClassDiamondRestartMeta reader_meta(mesh_builder, false);
  ASSERT_FALSE(reader_meta.meta_data->is_commit());
  BulkIoContext reader(MPI_COMM_WORLD, mesh_builder, reader_meta.meta_data);
  stk::io::fill_mesh_with_fields(restart_file.string(), reader.io_broker, *reader.bulk_data, stk::io::READ_RESTART);
  expect_scalar_field_values(*reader.bulk_data, stk::topology::NODE_RANK, reader_meta.node_int_fields, 10000);
  expect_scalar_field_values(*reader.bulk_data, stk::topology::ELEM_RANK, reader_meta.elem_int_fields, 20000);
  expect_scalar_field_values(*reader.bulk_data, stk::topology::NODE_RANK, reader_meta.node_unsigned_fields, 30000);
}

TEST_F(UnitTestClassFixture, ClassHierarchyPropagatesPrimaryFieldsAndInducedSetFields) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();
  meta_data->use_simple_fields();

  std::vector<NumberedClass> hierarchy = declare_numbered_class_hierarchy(*meta_data);
  declare_scalar_fields_on_hierarchy(*meta_data, hierarchy);

  expect_induced_sets_follow_hierarchy(hierarchy);
  expect_fields_follow_hierarchy(hierarchy);
}

TEST_F(UnitTestClassFixture, ClassHierarchyPropagatesExistingInducedSetsToLateSubclasses) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();
  meta_data->use_simple_fields();

  Class& parent_class = declare_class(*meta_data, "class0_late_parent", stk::topology::PARTICLE);
  stk::mesh::Field<int>& field0 = meta_data->declare_field<int>(stk::topology::NODE_RANK, "late_subclass_field0");
  put_field_on_mesh(field0, parent_class, 3, nullptr);

  ASSERT_TRUE(parent_class.has_induced_set(stk::topology::NODE_RANK));
  Class& parent_node_set = parent_class.get_induced_set(stk::topology::NODE_RANK);
  ASSERT_TRUE(field0.defined_on(parent_node_set.data_part()));
  ASSERT_TRUE(field0.defined_on(parent_node_set.leaf_part()));

  Class& child_class0 = declare_class(*meta_data, "class1_late_child", stk::topology::PARTICLE);
  Class& child_class1 = declare_class(*meta_data, "class2_late_child", stk::topology::PARTICLE);
  parent_class.declare_subset(child_class0);
  parent_class.declare_subset(child_class1);

  ASSERT_TRUE(child_class0.has_induced_set(stk::topology::NODE_RANK));
  ASSERT_TRUE(child_class1.has_induced_set(stk::topology::NODE_RANK));
  Class& child_node_set0 = child_class0.get_induced_set(stk::topology::NODE_RANK);
  Class& child_node_set1 = child_class1.get_induced_set(stk::topology::NODE_RANK);
  EXPECT_TRUE(parent_node_set.contains(child_node_set0));
  EXPECT_TRUE(parent_node_set.contains(child_node_set1));
  EXPECT_TRUE(field0.defined_on(child_node_set0.data_part()));
  EXPECT_TRUE(field0.defined_on(child_node_set0.leaf_part()));
  EXPECT_TRUE(field0.defined_on(child_node_set1.data_part()));
  EXPECT_TRUE(field0.defined_on(child_node_set1.leaf_part()));

  meta_data->commit();
  std::unique_ptr<BulkData> bulk_data = mesh_builder.create_bulk_data(meta_data);
  BulkDataClassInterface cbulk_data = class_interface(*bulk_data);

  bulk_data->modification_begin();
  const stk::mesh::Entity node0 = cbulk_data.declare_node(1u, child_class0);
  const stk::mesh::Entity node1 = cbulk_data.declare_node(2u, child_class1);
  bulk_data->modification_end();

  ASSERT_TRUE(bulk_data->bucket(node0).member(child_node_set0.leaf_part()));
  ASSERT_TRUE(bulk_data->bucket(node0).member(parent_node_set.data_part()));
  ASSERT_NE(stk::mesh::field_data(field0, node0), nullptr);

  ASSERT_TRUE(bulk_data->bucket(node1).member(child_node_set1.leaf_part()));
  ASSERT_TRUE(bulk_data->bucket(node1).member(parent_node_set.data_part()));
  ASSERT_NE(stk::mesh::field_data(field0, node1), nullptr);
}

TEST_F(UnitTestClassFixture, ClassHierarchyDeclareElementAndNodeAssignsExpectedPartMembership) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();
  meta_data->use_simple_fields();

  std::vector<NumberedClass> hierarchy = declare_numbered_class_hierarchy(*meta_data);
  declare_scalar_fields_on_hierarchy(*meta_data, hierarchy);

  meta_data->commit();
  std::unique_ptr<BulkData> bulk_data = mesh_builder.create_bulk_data(meta_data);
  BulkDataClassInterface cbulk_data = class_interface(*bulk_data);

  std::vector<stk::mesh::Entity> elements(hierarchy.size());
  std::vector<stk::mesh::Entity> nodes(hierarchy.size());

  bulk_data->modification_begin();
  for (size_t class_idx = 0; class_idx < hierarchy.size(); ++class_idx) {
    const stk::mesh::EntityId node_id = static_cast<stk::mesh::EntityId>(class_idx + 101);
    nodes[class_idx] = cbulk_data.declare_node(node_id, *hierarchy[class_idx].class_instance);
  }
  bulk_data->modification_end();

  for (size_t class_idx = 0; class_idx < hierarchy.size(); ++class_idx) {
    const std::string node_label = hierarchy[class_idx].name + " node before relation";

    ASSERT_EQ(bulk_data->entity_rank(nodes[class_idx]), stk::topology::NODE_RANK) << node_label;
    expect_class_part_membership(*bulk_data, nodes[class_idx], node_label, hierarchy, class_idx);
    expect_induced_set_part_membership(*bulk_data, nodes[class_idx], node_label, hierarchy, class_idx, true);
  }

  bulk_data->modification_begin();
  for (size_t class_idx = 0; class_idx < hierarchy.size(); ++class_idx) {
    const stk::mesh::EntityId elem_id = static_cast<stk::mesh::EntityId>(class_idx + 1);
    elements[class_idx] = cbulk_data.declare_element(elem_id, *hierarchy[class_idx].class_instance);
    bulk_data->declare_relation(elements[class_idx], nodes[class_idx], 0u);
  }
  bulk_data->modification_end();

  for (size_t class_idx = 0; class_idx < hierarchy.size(); ++class_idx) {
    const std::string elem_label = hierarchy[class_idx].name + " particle";
    const std::string node_label = hierarchy[class_idx].name + " node after relation";

    ASSERT_EQ(bulk_data->entity_rank(elements[class_idx]), stk::topology::ELEM_RANK) << elem_label;
    ASSERT_EQ(bulk_data->entity_rank(nodes[class_idx]), stk::topology::NODE_RANK) << node_label;
    ASSERT_EQ(bulk_data->num_nodes(elements[class_idx]), 1u) << elem_label;
    EXPECT_EQ(bulk_data->begin_nodes(elements[class_idx])[0], nodes[class_idx]) << elem_label;

    expect_class_part_membership(*bulk_data, elements[class_idx], elem_label, hierarchy, class_idx);
    expect_induced_set_part_membership(*bulk_data, elements[class_idx], elem_label, hierarchy, class_idx, false);

    expect_class_part_membership(*bulk_data, nodes[class_idx], node_label, hierarchy, class_idx);
    expect_induced_set_part_membership(*bulk_data, nodes[class_idx], node_label, hierarchy, class_idx, true);
  }
}

TEST_F(UnitTestClassFixture, ChangeEntityClassesSupportsPrimarySetAndInducedMembershipSwaps) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();
  meta_data->use_simple_fields();

  Class& elem_class0 = declare_class(*meta_data, "elem_class0", stk::topology::PARTICLE);
  Class& elem_class1 = declare_class(*meta_data, "elem_class1", stk::topology::PARTICLE);
  Class& elem_class2 = declare_class(*meta_data, "elem_class2", stk::topology::PARTICLE);
  Class& node_class0 = declare_class(*meta_data, "node_class0", stk::topology::NODE_RANK);
  Class& node_class1 = declare_class(*meta_data, "node_class1", stk::topology::NODE_RANK);
  Class& node_set0 = declare_class(*meta_data, "node_set0", stk::topology::NODE_RANK);
  Class& node_set1 = declare_class(*meta_data, "node_set1", stk::topology::NODE_RANK);

  Class& elem_class0_nodeset = elem_class0.get_or_create_induced_set(stk::topology::NODE_RANK);
  Class& elem_class1_nodeset = elem_class1.get_or_create_induced_set(stk::topology::NODE_RANK);

  meta_data->commit();
  std::unique_ptr<BulkData> bulk_data = mesh_builder.create_bulk_data(meta_data);
  BulkDataClassInterface cbulk_data = class_interface(*bulk_data);

  bulk_data->modification_begin();
  const stk::mesh::Entity node = cbulk_data.declare_node(1u, node_class0);
  bulk_data->modification_end();

  bulk_data->modification_begin();
  cbulk_data.change_entity_classes(node, ClassVector{&node_set0, &elem_class0});
  bulk_data->modification_end();

  EXPECT_TRUE(bulk_data->bucket(node).member(node_class0.leaf_part()));
  EXPECT_TRUE(bulk_data->bucket(node).member(node_set0.leaf_part()));
  EXPECT_TRUE(bulk_data->bucket(node).member(elem_class0_nodeset.leaf_part()));

  bulk_data->modification_begin();
  cbulk_data.change_entity_classes(node, ClassVector{&node_class1, &node_set1, &elem_class1},
                                   ClassVector{&node_class0, &node_set0, &elem_class0});
  bulk_data->modification_end();

  EXPECT_FALSE(bulk_data->bucket(node).member(node_class0.leaf_part()));
  EXPECT_TRUE(bulk_data->bucket(node).member(node_class1.leaf_part()));

  EXPECT_FALSE(bulk_data->bucket(node).member(node_set0.leaf_part()));
  EXPECT_TRUE(bulk_data->bucket(node).member(node_set1.leaf_part()));

  EXPECT_FALSE(bulk_data->bucket(node).member(elem_class0_nodeset.leaf_part()));
  EXPECT_TRUE(bulk_data->bucket(node).member(elem_class1_nodeset.leaf_part()));

  bulk_data->modification_begin();
  EXPECT_NO_THROW(cbulk_data.change_entity_classes(node, ClassVector{&elem_class2}));
  bulk_data->modification_end();
}

TEST_F(UnitTestClassFixture, ChangeEntityClassesRejectsInvalidClassConfigurations) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();
  meta_data->use_simple_fields();

  Class& elem_class = declare_class(*meta_data, "elem_class_invalid", stk::topology::PARTICLE);
  Class& elem_class_alt = declare_class(*meta_data, "elem_class_alt_invalid", stk::topology::PARTICLE);
  Class& node_class0 = declare_class(*meta_data, "node_class0_invalid", stk::topology::NODE_RANK);
  Class& node_class1 = declare_class(*meta_data, "node_class1_invalid", stk::topology::NODE_RANK);
  Class& node_set = declare_class(*meta_data, "node_set_invalid", stk::topology::NODE_RANK);
  EXPECT_THROW(declare_class(*meta_data, "elem_set_invalid", stk::topology::ELEMENT_RANK), std::logic_error);

  elem_class.get_or_create_induced_set(stk::topology::NODE_RANK);

  meta_data->commit();
  std::unique_ptr<BulkData> bulk_data = mesh_builder.create_bulk_data(meta_data);
  BulkDataClassInterface cbulk_data = class_interface(*bulk_data);

  bulk_data->modification_begin();
  const stk::mesh::Entity node = cbulk_data.declare_node(1u, node_class0);
  const stk::mesh::Entity elem = cbulk_data.declare_element(1u, elem_class);
  bulk_data->modification_end();

  bulk_data->modification_begin();
  EXPECT_NO_THROW(cbulk_data.change_entity_classes(node, ClassVector{&node_class0, &node_class1}));
  EXPECT_NO_THROW(cbulk_data.change_entity_classes(node, ClassVector{}, ClassVector{&node_class0, &node_class1}));
  EXPECT_THROW(cbulk_data.change_entity_classes(elem, ClassVector{&elem_class, &elem_class_alt}), std::logic_error);
  EXPECT_THROW(cbulk_data.change_entity_classes(elem, ClassVector{&node_class0}), std::logic_error);
  EXPECT_THROW(cbulk_data.change_entity_classes(node, ClassVector{nullptr}), std::logic_error);
  EXPECT_NO_THROW(cbulk_data.change_entity_classes(node, ClassVector{&node_set, &elem_class}));
  bulk_data->modification_end();
}

TEST_F(UnitTestClassFixture, ClassApiAllowsTemporarilyUnsupportedCasesWhenIoIsDisabled) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();

  EXPECT_THROW(declare_class(*meta_data, "elem_set_io", stk::topology::ELEMENT_RANK), std::logic_error);
  EXPECT_THROW(declare_class(*meta_data, "node_primary_io", stk::topology::NODE), std::logic_error);

  Class& elem_set_non_io = declare_class(*meta_data, "elem_set_non_io", stk::topology::ELEMENT_RANK, true);
  Class& node_primary_non_io = declare_class(*meta_data, "node_primary_non_io", stk::topology::NODE, true);

  EXPECT_TRUE(elem_set_non_io.is_set());
  EXPECT_FALSE(elem_set_non_io.has_io_support());
  EXPECT_EQ(elem_set_non_io.primary_entity_rank(), stk::topology::ELEMENT_RANK);

  EXPECT_TRUE(node_primary_non_io.is_primary());
  EXPECT_FALSE(node_primary_non_io.has_io_support());
  EXPECT_EQ(node_primary_non_io.primary_entity_rank(), stk::topology::NODE_RANK);
}

TEST_F(UnitTestClassFixture, ClassAssemblySelectorsComposeLikeStkParts) {
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  initialize_mesh_builder(mesh_builder);
  std::shared_ptr<MetaData> meta_data = mesh_builder.create_meta_data();
  meta_data->use_simple_fields();

  Class& class0 = declare_class(*meta_data, "class0_selector", stk::topology::PARTICLE);
  Class& class1 = declare_class(*meta_data, "class1_selector", stk::topology::PARTICLE);
  Class& class2 = declare_class(*meta_data, "class2_selector", stk::topology::PARTICLE);
  class0.declare_subset(class1);

  meta_data->commit();
  std::unique_ptr<BulkData> bulk_data = mesh_builder.create_bulk_data(meta_data);
  BulkDataClassInterface cbulk_data = class_interface(*bulk_data);

  bulk_data->modification_begin();
  const stk::mesh::Entity elem0 = cbulk_data.declare_element(1u, class0);
  const stk::mesh::Entity elem1 = cbulk_data.declare_element(2u, class1);
  const stk::mesh::Entity elem2 = cbulk_data.declare_element(3u, class2);
  bulk_data->modification_end();

  const stk::mesh::Part& class0_as_part = class0;
  EXPECT_EQ(&class0_as_part, &class0.assembly_part());

  const stk::mesh::Selector class0_selector = class0;
  EXPECT_TRUE(class0_selector(bulk_data->bucket(elem0)));
  EXPECT_TRUE(class0_selector(bulk_data->bucket(elem1)));
  EXPECT_FALSE(class0_selector(bulk_data->bucket(elem2)));

  const stk::mesh::Selector child_only_selector = class0 & class1;
  EXPECT_FALSE(child_only_selector(bulk_data->bucket(elem0)));
  EXPECT_TRUE(child_only_selector(bulk_data->bucket(elem1)));
  EXPECT_FALSE(child_only_selector(bulk_data->bucket(elem2)));

  const stk::mesh::Selector parent_without_child_selector = class0 - class1;
  EXPECT_TRUE(parent_without_child_selector(bulk_data->bucket(elem0)));
  EXPECT_FALSE(parent_without_child_selector(bulk_data->bucket(elem1)));
  EXPECT_FALSE(parent_without_child_selector(bulk_data->bucket(elem2)));

  const stk::mesh::Selector class0_or_class2_selector = class0 | class2;
  EXPECT_TRUE(class0_or_class2_selector(bulk_data->bucket(elem0)));
  EXPECT_TRUE(class0_or_class2_selector(bulk_data->bucket(elem1)));
  EXPECT_TRUE(class0_or_class2_selector(bulk_data->bucket(elem2)));

  const stk::mesh::Selector not_class1_selector = !class1;
  EXPECT_TRUE(not_class1_selector(bulk_data->bucket(elem0)));
  EXPECT_FALSE(not_class1_selector(bulk_data->bucket(elem1)));
  EXPECT_TRUE(not_class1_selector(bulk_data->bucket(elem2)));
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
