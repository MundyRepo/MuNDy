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

#ifndef MUNDY_MESH_DECLAREENTITIES_HPP_
#define MUNDY_MESH_DECLAREENTITIES_HPP_

/// \file DeclareEntities.hpp
/// \brief A set of helper methods for declaring entities without worrying about parallel ownership and sharing.

// External
#include <fmt/format.h>  // for fmt::format

// C++ core
#include <iostream>       // for std::ostream
#include <memory>         // for std::shared_ptr
#include <stdexcept>      // for std::runtime_error
#include <tuple>          // for std::tuple, std::make_tuple
#include <typeindex>      // for std::type_index
#include <unordered_map>  // for std::unordered_map
#include <unordered_set>  // for std::unordered_set
#include <utility>        // for std::pair, std::make_pair
#include <vector>         // for std::vector

// Trilinos
#include <stk_mesh/base/BulkData.hpp>    // for stk::mesh::BulkData
#include <stk_mesh/base/FEMHelpers.hpp>  // for stk::mesh::declare_element

// Mundy
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_REQUIRE
#include <mundy_mesh/LinkData.hpp>       // for mundy::mesh::LinkData
#include <mundy_mesh/fmt_stk_types.hpp>  // adds fmt::format for stk types

namespace mundy {

namespace mesh {

/// \brief Helper class for declaring entities.
///
/// This class is used to aid the declaration of entities in a mesh. Use it to build up a parallel synchonous list of
/// nodes and elements that should be declared in the mesh. Once complete, use it to perform the declaration, sharing,
/// and setting of field data automatically.
///
/// \note We emphasize that all processes should own exact copies of the same DeclareEntitiesHelper. This choice means
/// that DeclareEntitiesHelper is not optimally performant but the cost of duplicating entity information is cheap in
/// comparison to the burden of determining parallel ownership and sharing.
///
/// \note The create_* methods within this class are not thread safe. To perform parallel construction of entities, use
/// the create_nodes and create_elements methods to create a vector of entities to populate and then populate them in
/// parallel.
///
/// Example usage declaring a pearl necklace (a chain of spheres connected by springs):
/* n1       n3        n5        n7
///  \      /  \      /  \      /
///   s1   s2   s3   s4   s5   s6
///    \  /      \  /      \  /
///     n2        n4        n6
*/
/// \code{.cpp}
///   const int num_nodes = 7;
///   const int num_edges = 6;
///   DeclareEntitiesHelper builder;
///   for (int i = 0; i < num_nodes; ++i) {
///     builder.create_node().owning_proc(0).id(i + 1);
///   }
///   for (int i = 0; i < num_edges; ++i) {
///     auto& spring = builder.create_element();
///     spring.owning_proc(0).id(i + 1).topology(stk::topology::BEAM_2).nodes({i + 1, i + 2});
///   }
///   for (int i = 0; i < num_nodes; ++i) {
///     auto& sphere = builder.create_element();
///     sphere.owning_proc(0).id(i + 1 + num_edges).topology(stk::topology::PARTICLE).nodes({i + 1});
///   }
///   builder.declare_entities(bulk_data);
/// \endcode
class DeclareEntitiesHelper {
 private:
  //! \name Private Helpers
  //@{

  class FieldDataBase {
   public:
    virtual ~FieldDataBase() = default;
    virtual void set_field_data(const stk::mesh::Entity& entity) = 0;
    virtual std::type_index type() const = 0;
    virtual std::string name() const = 0;
    virtual stk::mesh::FieldBase* field() const = 0;
  };

  template <typename T>
  class FieldData : public FieldDataBase {
   public:
    FieldData(stk::mesh::FieldBase* field, const std::vector<T>& data) : field_(field), data_(data) {
    }

    void set_field_data(const stk::mesh::Entity& entity) override {
      T* raw_field_data = static_cast<T*>(stk::mesh::field_data(*field_, entity));
      for (size_t i = 0; i < data_.size(); ++i) {
        raw_field_data[i] = data_[i];
      }
    }

    std::type_index type() const override {
      return typeid(T);
    }

    std::string name() const override {
      return field_->name();
    }

    std::vector<T>& data() {
      return data_;
    }

    const std::vector<T>& data() const {
      return data_;
    }

    stk::mesh::FieldBase* field() const override {
      return field_;
    }

   private:
    stk::mesh::FieldBase* const field_;
    std::vector<T> data_;
  };

  struct DeclareLinksInfo {
    stk::mesh::EntityIdVector linked_entity_ids;
    std::vector<stk::mesh::EntityRank> linked_entity_ranks;
  };

  struct DeclareNodeInfo {
    int owning_proc = 0;
    std::vector<int> non_owning_shared_procs;
    stk::mesh::EntityId id = stk::mesh::InvalidEntityId;
    stk::mesh::PartVector parts;
    std::vector<std::shared_ptr<FieldDataBase>> field_data;
    std::map<std::string, DeclareLinksInfo> link_info_map;
  };

  struct DeclareElementInfo {
    int owning_proc = 0;
    stk::mesh::EntityId id = stk::mesh::InvalidEntityId;
    stk::topology topology = stk::topology::INVALID_TOPOLOGY;
    stk::mesh::EntityIdVector node_ids;
    stk::mesh::PartVector parts;
    std::vector<std::shared_ptr<FieldDataBase>> field_data;
    std::map<std::string, DeclareLinksInfo> link_info_map;
  };

  // Overload the operator<<
  friend std::ostream& operator<<(std::ostream& os, const DeclareLinksInfo& info) {
    size_t num_links = info.linked_entity_ids.size();
    os << "  Linked Entities: ";
    if (num_links == 0) {
      os << "None";
    } else {
      os << "{";
      for (size_t i = 0; i < num_links; ++i) {
        os << "(" << info.linked_entity_ranks[i] << ", " << info.linked_entity_ids[i] << ")";
        if (i < num_links - 1) {
          os << ", ";
        } else {
          os << "}";
        }
      }
    }
    os << "\n";
    return os;
  }

  /// Overload the operator<<
  friend std::ostream& operator<<(std::ostream& os, const DeclareNodeInfo& info) {
    os << "  Owning Processor: " << info.owning_proc << "\n";
    os << "  Node ID: " << info.id << "\n";
    os << "  Number of Parts: " << info.parts.size() << "\n";

    // Print Parts
    os << "  Parts: ";
    size_t part_counter = 0;
    size_t num_parts = info.parts.size();
    if (num_parts == 0) {
      os << "None";
    } else {
      os << "{";
      for (const auto& part : info.parts) {
        if (part != nullptr) {
          os << part->name();
        } else {
          os << "nullptr";
        }
        if (part_counter < num_parts - 1) {
          os << ", ";
        } else {
          os << "}";
        }
        ++part_counter;
      }
    }
    os << "\n";

    // Print Field Data
    os << "  Field Data: ";
    size_t field_data_counter = 0;
    size_t num_field_data = info.field_data.size();
    if (num_field_data == 0) {
      os << "None";
    } else {
      os << "{";
      for (const auto& field_data : info.field_data) {
        if (field_data != nullptr) {
          os << field_data->name() << ": " << field_data->type().name();
        } else {
          os << "nullptr";
        }
        if (field_data_counter < num_field_data - 1) {
          os << ", ";
        } else {
          os << "}";
        }
        ++field_data_counter;
      }
    }
    os << "\n";

    // Print the links
    if (!info.link_info_map.empty()) {
      os << "  Links Info:\n";
      for (const auto& link_pair : info.link_info_map) {
        os << "    LinkMetaData Name: " << link_pair.first << "\n";
        os << link_pair.second;
      }
    }

    return os;
  }

  /// Overload the operator<<
  friend std::ostream& operator<<(std::ostream& os, const DeclareElementInfo& info) {
    os << "  Owning Processor: " << info.owning_proc << "\n";
    os << "  Element ID: " << info.id << "\n";
    os << "  Topology: " << info.topology.name() << "\n";

    // Print Node IDs
    os << "  Number of Node IDs: " << info.node_ids.size() << "\n";
    os << "  Node IDs: ";
    size_t node_id_counter = 0;
    size_t num_node_ids = info.node_ids.size();
    if (num_node_ids == 0) {
      os << "None";
    } else {
      os << "{";
      for (const auto& node_id : info.node_ids) {
        os << node_id;
        if (node_id_counter < num_node_ids - 1) {
          os << ", ";
        } else {
          os << "}";
        }
        ++node_id_counter;
      }
    }
    os << "\n";

    // Print Parts
    os << "  Number of Parts: " << info.parts.size() << "\n";
    os << "  Parts: ";
    size_t part_counter = 0;
    size_t num_parts = info.parts.size();
    if (num_parts == 0) {
      os << "None";
    } else {
      os << "{";
      for (const auto& part : info.parts) {
        if (part != nullptr) {
          os << part->name();
        } else {
          os << "nullptr";
        }
        if (part_counter < num_parts - 1) {
          os << ", ";
        } else {
          os << "}";
        }
        ++part_counter;
      }
    }
    os << "\n";

    // Print Field Data
    os << "  Number of Field Data: " << info.field_data.size() << "\n";
    os << "  Field Data: ";
    size_t field_data_counter = 0;
    size_t num_field_data = info.field_data.size();
    if (num_field_data == 0) {
      os << "None";
    } else {
      os << "{";
      for (const auto& field_data : info.field_data) {
        if (field_data != nullptr) {
          os << field_data->name() << ": " << field_data->type().name();
        } else {
          os << "nullptr";
        }
        if (field_data_counter < num_field_data - 1) {
          os << ", ";
        } else {
          os << "}";
        }
        ++field_data_counter;
      }
    }

    // Print the links
    if (!info.link_info_map.empty()) {
      os << "  Links Info:\n";
      for (const auto& link_pair : info.link_info_map) {
        os << "    LinkMetaData Name: " << link_pair.first << "\n";
        os << link_pair.second;
      }
    }

    os << "\n";
    return os;
  }
  //@}

 public:
  //! \name Nested Builders
  //@{

  /// \brief Nested builder for nodes.
  class NodeBuilder {
   public:
    /// \brief Set the processor that owns the node.
    NodeBuilder& owning_proc(const int proc) {
      node_info_.owning_proc = proc;
      return *this;
    }

    /// \brief Set the entity id of the node (indexed from 1)
    NodeBuilder& id(const stk::mesh::EntityId entity_id) {
      node_info_.id = entity_id;
      return *this;
    }

    /// \brief Add a part to the node (i.e, a part that the node belongs to)
    /// \note You do not need to add parts that will be automatically added such as the universal part.
    ///  This includes parts that the node will become a member of due to automatic part inheritance.
    NodeBuilder& add_part(stk::mesh::Part* part_ptr) {
      node_info_.parts.push_back(part_ptr);
      return *this;
    }

    /// \brief Add a vector of parts to the node (i.e, parts that the node belongs to)
    NodeBuilder& add_parts(const stk::mesh::PartVector& parts) {
      node_info_.parts.insert(node_info_.parts.end(), parts.begin(), parts.end());
      return *this;
    }

    /// \brief Add a vector of parts to the node (i.e, parts that the node belongs to)
    NodeBuilder& add_parts(std::initializer_list<stk::mesh::Part*> parts) {
      stk::mesh::PartVector part_vector(parts);
      return add_parts(part_vector);
    }

    /// \brief Add field data to the node.
    /// \param field The field to set data for
    /// \param data The data to set
    template <typename T>
    NodeBuilder& add_field_data(stk::mesh::FieldBase* const field, const std::vector<T>& data) {
      node_info_.field_data.push_back(std::make_shared<FieldData<T>>(field, data));
      return *this;
    }

    /// \brief Add field data to the node.
    /// \param field The field to set data for
    /// \param data The data to set
    template <typename T>
    NodeBuilder& add_field_data(stk::mesh::FieldBase* const field, const T& data) {
      node_info_.field_data.push_back(std::make_shared<FieldData<T>>(field, std::vector<T>{data}));
      return *this;
    }

    /// \brief This node is a valid linker. Link it to a given entity at the desired ordinal.
    /// \param link_data_ptr The link data to link under.
    /// \param linked_entity_id The entity id of the entity to link to.
    /// \param linked_entity_rank The entity rank of the entity to link to.
    /// \param ordinal The slot/ordinal to link the entity at.
    NodeBuilder& links_to(LinkData* const link_data_ptr, const stk::mesh::EntityId linked_entity_id,
                          const stk::mesh::EntityRank linked_entity_rank, const unsigned ordinal) {
      MUNDY_THROW_REQUIRE(link_data_ptr != nullptr, std::invalid_argument, "LinkData pointer is null.");
      const std::string& link_meta_data_name = link_data_ptr->link_meta_data().name();
      auto& link_info = node_info_.link_info_map[link_meta_data_name];
      if (link_info.linked_entity_ids.size() <= ordinal) {
        link_info.linked_entity_ids.resize(ordinal + 1, stk::mesh::InvalidEntityId);
        link_info.linked_entity_ranks.resize(ordinal + 1, stk::topology::INVALID_RANK);
      }
      link_info.linked_entity_ids[ordinal] = linked_entity_id;
      link_info.linked_entity_ranks[ordinal] = linked_entity_rank;
      return *this;
    }

    /// \brief Get the owner of the builder.
    DeclareEntitiesHelper& owner() {
      return owner_;
    }

    /// Overload the operator<<
    friend std::ostream& operator<<(std::ostream& os, const NodeBuilder& builder) {
      const auto& info = builder.node_info_;
      os << info;
      return os;
    }

   private:
    /// \brief Private constructor for the NodeBuilder.
    NodeBuilder(DeclareEntitiesHelper& owner, DeclareNodeInfo& node_info) : owner_(owner), node_info_(node_info) {
    }

    //! \name Internal Data
    //@{

    DeclareEntitiesHelper& owner_;
    DeclareNodeInfo& node_info_;
    friend class DeclareEntitiesHelper;
    //@}
  };  // class NodeBuilder

  // Nested Builder for Elements
  class ElementBuilder {
   public:
    /// \brief Set the processor that owns the element.
    ElementBuilder& owning_proc(const int proc) {
      elem_info_.owning_proc = proc;
      return *this;
    }

    /// \brief Set the entity id of the element (indexed from 1)
    ElementBuilder& id(const stk::mesh::EntityId entity_id) {
      elem_info_.id = entity_id;
      return *this;
    }

    /// \brief Set the topology of the element.
    ElementBuilder& topology(const stk::topology topology) {
      elem_info_.topology = topology;
      return *this;
    }

    /// \brief Set the nodes that the element is connected to.
    ElementBuilder& nodes(const std::vector<stk::mesh::EntityId>& node_ids) {
      elem_info_.node_ids = node_ids;
      return *this;
    }

    /// \brief Add a part to the element (i.e, a part that the element belongs to)
    ElementBuilder& add_part(stk::mesh::Part* part_ptr) {
      elem_info_.parts.push_back(part_ptr);
      return *this;
    }

    /// \brief Add a vector of parts to the element (i.e, parts that the element belongs to)
    ElementBuilder& add_parts(const stk::mesh::PartVector& parts) {
      elem_info_.parts.insert(elem_info_.parts.end(), parts.begin(), parts.end());
      return *this;
    }

    /// \brief Add a vector of parts to the element (i.e, parts that the element belongs to)
    ElementBuilder& add_parts(std::initializer_list<stk::mesh::Part*> parts) {
      stk::mesh::PartVector part_vector(parts);
      return add_parts(part_vector);
    }

    /// \brief Add field data to the element.
    /// \param field The field to set data for
    /// \param data The data to set
    template <typename T>
    ElementBuilder& add_field_data(stk::mesh::FieldBase* const field, const std::vector<T>& data) {
      elem_info_.field_data.push_back(std::make_shared<FieldData<T>>(field, data));
      return *this;
    }

    /// \brief Add field data to the element.
    /// \param field The field to set data for
    /// \param data The data to set
    template <typename T>
    ElementBuilder& add_field_data(stk::mesh::FieldBase* const field, const T& data) {
      elem_info_.field_data.push_back(std::make_shared<FieldData<T>>(field, std::vector<T>{data}));
      return *this;
    }

    /// \brief This element is a valid linker. Link it to a given entity at the desired ordinal.
    /// \param link_meta_data_name The name of the LinkMetaData to link under.
    /// \param linked_entity_id The entity id of the entity to link to.
    /// \param linked_entity_rank The entity rank of the entity to link to.
    /// \param ordinal The slot/ordinal to link the entity at.
    ElementBuilder& links_to(LinkData* const link_data_ptr, const stk::mesh::EntityId linked_entity_id,
                          const stk::mesh::EntityRank linked_entity_rank, const unsigned ordinal) {
      MUNDY_THROW_REQUIRE(link_data_ptr != nullptr, std::invalid_argument, "LinkData pointer is null.");
      const std::string& link_meta_data_name = link_data_ptr->link_meta_data().name();
      auto& link_info = elem_info_.link_info_map[link_meta_data_name];
      if (link_info.linked_entity_ids.size() <= ordinal) {
        link_info.linked_entity_ids.resize(ordinal + 1, stk::mesh::InvalidEntityId);
        link_info.linked_entity_ranks.resize(ordinal + 1, stk::topology::INVALID_RANK);
      }
      link_info.linked_entity_ids[ordinal] = linked_entity_id;
      link_info.linked_entity_ranks[ordinal] = linked_entity_rank;
      return *this;
    }

    /// \brief Get the owner of the builder.
    DeclareEntitiesHelper& owner() {
      return owner_;
    }

    /// Overload the operator<<
    friend std::ostream& operator<<(std::ostream& os, const ElementBuilder& builder) {
      const auto& info = builder.elem_info_;
      os << info;
      return os;
    }

   private:
    /// \brief Private constructor for the ElementBuilder.
    ElementBuilder(DeclareEntitiesHelper& owner, DeclareElementInfo& element_info)
        : owner_(owner), elem_info_(element_info) {
    }

    //! \name Internal Data
    //@{

    DeclareEntitiesHelper& owner_;
    DeclareElementInfo& elem_info_;
    //@}

    //! \name Friends <3
    //@{

    friend class DeclareEntitiesHelper;
    //@}
  };  // class ElementBuilder
  //@}

  //! \name Actions
  //@{

  /// \brief Create a new NodeBuilder for hierarchical construction of a node (not thread safe).
  /// NodeBuilder views our internal data, so we will automatically know about modifications to the node.
  ///
  /// \return The NodeBuilder for the new node.
  NodeBuilder create_node() {
    node_info_vec_.emplace_back();
    return NodeBuilder(*this, node_info_vec_.back());
  }

  /// \brief Create a new ElementBuilder for hierarchical construction of an element (not thread safe).
  /// ElementBuilder views our internal data, so we will automatically know about modifications to the element.
  ///
  /// \return The ElementBuilder for the new element.
  ElementBuilder create_element() {
    elem_info_vec_.emplace_back();
    return ElementBuilder(*this, elem_info_vec_.back());
  }

  /// \brief Create a vector of NodeBuilders for parallel construction of nodes (not thread safe).
  /// Each NodeBuilder views our internal data, so we will automatically know about modifications to the nodes.
  ///
  /// If you want to create nodes in parallel, use this method to create a vector of NodeBuilders to populate and then
  /// populate them in parallel.
  ///
  /// \param num_nodes The number of nodes to create
  /// \return A vector of NodeBuilders for the new nodes.
  std::vector<NodeBuilder> create_nodes(size_t num_nodes) {
    std::vector<NodeBuilder> builders;
    builders.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      node_info_vec_.emplace_back();
      builders.push_back(NodeBuilder(*this, node_info_vec_.back()));
    }
    return builders;
  }

  /// \brief Create a vector of ElementBuilders for parallel construction of elements (not thread safe).
  /// Each ElementBuilder views our internal data, so we will automatically know about modifications to the elements.
  ///
  /// If you want to create elements in parallel, use this method to create a vector of ElementBuilders to populate and
  /// then populate them in parallel.
  ///
  /// \param num_elements The number of elements to create
  /// \return A vector of ElementBuilders for the new elements.
  std::vector<ElementBuilder> create_elements(size_t num_elements) {
    std::vector<ElementBuilder> builders;
    builders.reserve(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      elem_info_vec_.emplace_back();
      builders.push_back(ElementBuilder(*this, elem_info_vec_.back()));
    }
    return builders;
  }

  /// \brief Print the builder information to the output stream.
  friend std::ostream& operator<<(std::ostream& os, const DeclareEntitiesHelper& builder) {
    os << "Number of Nodes: " << builder.node_info_vec_.size() << "\n";
    size_t node_count = 0;
    for (const auto& node_info : builder.node_info_vec_) {
      os << "Node " << ++node_count << ":\n";
      os << node_info;
    }
    os << "Number of Elements: " << builder.elem_info_vec_.size() << "\n";
    size_t element_count = 0;
    for (const auto& element_info : builder.elem_info_vec_) {
      os << "Element " << ++element_count << ":\n";
      os << element_info;
    }

    return os;
  }

  /// \brief Check the consistency of the builder.
  /// 1. Check that the number of nodes that an element connects to matches its topology.
  /// 2. Check for elements connected to non-existent nodes that aren't marked as Invalid.
  ///    It's perfectly ok for elements to connect to an invalid node (aka, a default constructed stk::mesh::Entity).
  ///    You just need to mark the node as invalid. This allows us to identify when someone attempts to connect an
  ///    element to a node that either doesn't exist in the mesh or is in our list of nodes to declare.
  /// 3. Check for duplicate entity ids in the nodes and elements that are requested for creation and in the entities
  /// that are already exist on the mesh.
  void check_consistency(const stk::mesh::BulkData& bulk_data) const;

  /// \brief Declare the entities in the mesh.
  /// This method will declare the entities in the mesh, share them, and set the field data according to the
  /// information already provided to the builder.
  ///
  /// We will not open a new modification cycle within declare entities and will instead assert that the function
  /// is called in a mod cycle. This helps with performance, as it avoid repeatedly opening and closing modification
  /// cycles, but it comes with a higher burden on the user. For example, entity sharing and aura generation will not
  /// occur until the next call to modification_end.
  ///
  /// \param bulk_data The bulk data
  DeclareEntitiesHelper& declare_entities(stk::mesh::BulkData& bulk_data);
  //@}

 private:
  //! \name Internal Helpers
  //@{

  void fill_key_ptr(const stk::mesh::OrdinalVector& parts, stk::mesh::PartOrdinal** key_ptr,
                    stk::mesh::PartOrdinal** key_end, const unsigned max_key_tmp_buffer_size,
                    stk::mesh::PartOrdinal* key_tmp_buffer, stk::mesh::OrdinalVector& key_tmp_vec) const {
    const size_t part_count = parts.size();
    const size_t key_len = 2 + part_count;

    *key_ptr = key_tmp_buffer;
    *key_end = *key_ptr + key_len;

    if (key_len >= max_key_tmp_buffer_size) {
      key_tmp_vec.resize(key_len);
      *key_ptr = key_tmp_vec.data();
      *key_end = *key_ptr + key_len;
    }

    //----------------------------------
    // Key layout:
    // { part_count + 1 , { part_ordinals } , partition_count }
    // Thus partition_count = key[ key[0] ]
    //
    // for upper bound search use the maximum key for a bucket in the partition.
    const unsigned max = static_cast<unsigned>(-1);
    (*key_ptr)[0] = static_cast<stk::mesh::PartOrdinal>(part_count + 1);
    (*key_ptr)[(*key_ptr)[0]] = static_cast<stk::mesh::PartOrdinal>(max);

    {
      for (unsigned i = 0; i < part_count; ++i) {
        (*key_ptr)[i + 1] = parts[i];
      }
    }
  }

  stk::topology get_topology(const stk::mesh::MetaData& meta_data, stk::mesh::EntityRank entity_rank,
                             const stk::mesh::OrdinalVector& part_ordinals) const {
    // This code is shamelessly copied from stk::mesh::impl::BucketRepository
    constexpr unsigned max_key_tmp_buffer_size = 64;
    stk::mesh::PartOrdinal key_tmp_buffer[max_key_tmp_buffer_size];
    stk::mesh::OrdinalVector key_tmp_vec;

    stk::mesh::PartOrdinal* key_ptr = nullptr;
    stk::mesh::PartOrdinal* key_end = nullptr;
    fill_key_ptr(part_ordinals, &key_ptr, &key_end, max_key_tmp_buffer_size, key_tmp_buffer, key_tmp_vec);

    std::vector<stk::mesh::PartOrdinal> key(key_ptr, key_end);
    const std::pair<const unsigned*, const unsigned*>& part_ordinals_begin_end =
        std::make_pair(key.data() + 1, key.data() + key[0]);
    return stk::mesh::get_topology(meta_data, entity_rank, part_ordinals_begin_end);
  }

  stk::topology get_topology(const stk::mesh::MetaData& meta_data, stk::mesh::EntityRank entity_rank,
                             const stk::mesh::PartVector& parts) const {
    stk::mesh::OrdinalVector part_ordinals;
    for (const stk::mesh::Part* part : parts) {
      part_ordinals.push_back(part->mesh_meta_data_ordinal());
    }
    return get_topology(meta_data, entity_rank, part_ordinals);
  }

  //! \name Internal Data
  //@{

  std::vector<DeclareNodeInfo> node_info_vec_;
  std::vector<DeclareElementInfo> elem_info_vec_;
  //@}
};

}  // namespace mesh

}  // namespace mundy

// Add a fmt::format for DeclareEntitiesHelper
template <>
struct fmt::formatter<mundy::mesh::DeclareEntitiesHelper> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::mesh::DeclareEntitiesHelper::NodeBuilder> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::mesh::DeclareEntitiesHelper::ElementBuilder> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::mesh::DeclareEntitiesHelper::DeclareNodeInfo> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::mesh::DeclareEntitiesHelper::DeclareElementInfo> : fmt::ostream_formatter {};

#endif  // MUNDY_MESH_DECLAREENTITIES_HPP_
