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

#ifndef MUNDY_MESH_CLASS_HPP_
#define MUNDY_MESH_CLASS_HPP_

/// \file Class.hpp
/// \defgroup MundyMeshClasses mundy::mesh::Classes
/// \brief Turning STK's IO support from a hierarchy of disjoint parts to a polymorphic class hierarchy.

// C++ core
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// STK
#include <stk_io/FieldAndName.hpp>
#include <stk_io/IossBridge.hpp>
#include <stk_io/OutputVariableParams.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>

// Mundy
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace mesh {

class Class;
using ClassVector = std::vector<Class*>;
using ConstClassVector = std::vector<const Class*>;

Class& declare_class(stk::mesh::MetaData& meta_data, const std::string& class_name);
Class& declare_class(stk::mesh::MetaData& meta_data, const std::string& class_name, stk::mesh::EntityRank class_rank,
                     bool disable_io_support = false);
Class& declare_class(stk::mesh::MetaData& meta_data, const std::string& class_name,
                     stk::topology::topology_t class_topology, bool disable_io_support = false);
const ClassVector& get_classes(stk::mesh::MetaData& meta_data);
const ConstClassVector& get_classes(const stk::mesh::MetaData& meta_data);

namespace impl {
class ClassFactory;
}

/// \class Class
/// \brief Semantic mesh class backed by synchronized data and assembly part hierarchies.
///
/// Classes handle the map between class hierarchy logic and STK's io system. We don't set the rules, we just work
/// around them and try to make them easier to use. The main rules are:
///  - There are two types of Classes: primary classes and sets.
///
///  - A primary class is one that was assigned a topology upon declaration; it is the only type of class that can have
///  primary entity members. Importantly, an entity can reside in one and only one primary class of the same rank.
///
///  - Entities may be members of zero or more primary classes of higher rank. These entities will automatically acquire
///  the data and selector membership of these primary classes.
///
///  - Primary classes may have subclasses (of either primary or set type), which inherit the data and selector
///  membership of their parent class. These subclasses must have the same rank as the parent class.
///
///  - A set is a class that has an assigned rank but no topology; it is used for giving data to specific groups of
///  entities. An entity can reside in zero or more sets independent of primary class membership.
///
///  - Sets do not induce membership to lower ranks but may have subclasses of the same rank.
///
/// Unsupported features and limitations that break the above rules when IO support is enabled:
///  - No ELEMENT_RANK sets
///  - No primary classes of NODE_RANK
///
/// TODO(palmerb4): Introduce a runtime flag for disabling Class IO rules, thereby allowing for ELEMENT_RANK sets and
/// NODE_RANK primary classes.
/// TODO(palmerb4): Induced sets should be an implementation detail. Mostly because they are inextensible. It's not that
/// they always exist and always contain all lower rank entities, it's that they contain all data that you wish to store
/// on lower rank entities so if you decide to remove all the node fields for a class, then no induced node rank class
/// will be created for that class. This minimizes the number of parts created per class.
///
/// Classes handle the STK IO part hierarchy by introducing multiple parts per class + optional induced set parts. They
/// handle the interface logic between a (top down) polymorphic class hierarchy and a (bottom up) disjoint part
/// hierarchy. The set of parts changes based on the class type:
///   (1/2) Primary classes + Sets:
///     - `data_part()` contains all data for this class; it is not an IO part and is a superset to the data parts of
///     all subsets.
///
///     - `leaf_part()` contains all entities that reside within this primary; it is the concrete IO part and will never
///     have subsets. It is a subset of the data part.
///
///     - `assembly_part()` contains all entities in this class and all of its subsets; it is an IO assembly part and
///     superset to the assembly parts of all subsets. Think of this part as synonymous with the class itself.
///
///     - `leaf_assembly_part()` contains all entities within the leaf part; it is an IO assembly part and exists solely
///     to keep assembly membership homogeneous. It is a subset of the assembly part.
///
class Class {
 public:
  //! \name Type aliases
  //@{
  enum { INVALID_ID = -1 };
  using class_ordinal_t = unsigned;

  /// \brief How the class data/leaf-data parts were declared.
  enum class DeclarationKind : unsigned { NAMED = 0u, RANKED = 1u, TOPOLOGICAL = 2u };

  /// \brief The type of class this is.
  /// Class rules:
  ///   - PRIMARY: An entity may reside in one and only one primary class.
  ///   - SET: An entity may reside in zero or more set classes independent of primary class membership.
  enum class Type : unsigned { PRIMARY = 0u, SET = 1u };

  /// \brief Canonical declaration signature used to validate repeated declarations by name.
  struct DeclarationSignature {
    DeclarationKind declaration_kind = DeclarationKind::NAMED;
    stk::mesh::EntityRank class_rank = stk::topology::INVALID_RANK;
    stk::topology::topology_t class_topology = stk::topology::INVALID_TOPOLOGY;
    bool disable_io_support = false;

    bool operator==(const DeclarationSignature& rhs) const noexcept {
      return declaration_kind == rhs.declaration_kind && class_rank == rhs.class_rank &&
             class_topology == rhs.class_topology && disable_io_support == rhs.disable_io_support;
    }

    bool operator!=(const DeclarationSignature& rhs) const noexcept {
      return !(*this == rhs);
    }
  };
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor.
  Class() = delete;

  /// \brief Non-copyable and non-movable, mirroring STK object semantics.
  Class(const Class&) = delete;
  Class(Class&&) = delete;
  Class& operator=(const Class&) = delete;
  Class& operator=(Class&&) = delete;

  /// \brief Destructor.
  ~Class() = default;
  //@}

  //! \name Part-like API
  //@{

  /// \brief Fetch the owning MetaData manager.
  stk::mesh::MetaData& mesh_meta_data() noexcept {
    return meta_data_;
  }

  /// \brief Fetch the owning MetaData manager.
  const stk::mesh::MetaData& mesh_meta_data() const noexcept {
    return meta_data_;
  }

  /// \brief Backward-compatible alias for mesh_meta_data().
  stk::mesh::MetaData& meta_data() noexcept {
    return mesh_meta_data();
  }

  /// \brief Backward-compatible alias for mesh_meta_data().
  const stk::mesh::MetaData& meta_data() const noexcept {
    return mesh_meta_data();
  }

  /// \brief Fetch the primary entity rank for this class.
  stk::mesh::EntityRank primary_entity_rank() const {
    return data_part_.primary_entity_rank();
  }

  /// \brief Fetch the topology for this class.
  stk::topology topology() const {
    return data_part_.topology();
  }

  /// \brief Fetch the class name.
  const std::string& name() const noexcept {
    return class_name_;
  }

  /// \brief Fetch this class ordinal.
  class_ordinal_t class_ordinal() const noexcept {
    return class_ordinal_;
  }

  /// \brief Fetch the declaration signature used to create this class.
  const DeclarationSignature& declaration_signature() const noexcept {
    return declaration_signature_;
  }

  /// \brief Return whether this class suppresses induction.
  bool force_no_induce() const {
    return data_part_.force_no_induce();
  }

  /// \brief Return whether this class has IO hierarchy support enabled.
  bool has_io_support() const noexcept {
    return !declaration_signature_.disable_io_support;
  }

  /// \brief Should this class induce membership from \p from_rank?
  bool should_induce(stk::mesh::EntityRank from_rank) const {
    return data_part_.should_induce(from_rank);
  }

  /// \brief Could entities of \p rank be induced into this class?
  bool was_induced(stk::mesh::EntityRank rank) const {
    return data_part_.was_induced(rank);
  }

  /// \brief Whether class membership must be parallel consistent.
  bool entity_membership_is_parallel_consistent() const {
    return data_part_.entity_membership_is_parallel_consistent();
  }

  /// \brief Set whether class membership must be parallel consistent.
  void entity_membership_is_parallel_consistent(bool is_parallel_consistent) {
    data_part_.entity_membership_is_parallel_consistent(is_parallel_consistent);
  }

  /// \brief Fetch the class type.
  Type class_type() const noexcept {
    return class_type_;
  }

  /// \brief Fetch whether this class is a set.
  bool is_set() const noexcept {
    return class_type() == Type::SET;
  }

  /// \brief Fetch whether this class is a primary class.
  bool is_primary() const noexcept {
    return class_type() == Type::PRIMARY;
  }

  /// \brief Fetch the data-part id.
  int64_t data_part_id() const {
    return data_part_.id();
  }

  /// \brief Fetch the assembly-part id.
  int64_t assembly_part_id() const {
    return assembly_part_.id();
  }

  /// \brief Check if \p other is contained by this class in both hierarchy channels.
  bool contains(const Class& other) const {
    return assembly_part_.contains(other.assembly_part_) && data_part_.contains(other.data_part_);
  }

  /// \brief Direct subclasses declared through the Class API.
  const ClassVector& subclasses() noexcept {
    return subclasses_;
  }

  /// \brief Direct subclasses declared through the Class API.
  const ConstClassVector& subclasses() const noexcept {
    return const_subclasses_;
  }

  /// \brief Direct superclasses declared through the Class API.
  const ClassVector& superclasses() noexcept {
    return superclasses_;
  }

  /// \brief Direct superclasses declared through the Class API.
  const ConstClassVector& superclasses() const noexcept {
    return const_superclasses_;
  }

  /// \brief Parts that are supersets of this class's data part.
  const stk::mesh::PartVector& data_part_supersets() const {
    return data_part_.supersets();
  }

  /// \brief Parts that are subsets of this class's data part.
  const stk::mesh::PartVector& data_part_subsets() const {
    return data_part_.subsets();
  }

  /// \brief Parts that are supersets of this class's assembly part.
  const stk::mesh::PartVector& assembly_part_supersets() const {
    return assembly_part_.supersets();
  }

  /// \brief Parts that are subsets of this class's assembly part.
  const stk::mesh::PartVector& assembly_part_subsets() const {
    return assembly_part_.subsets();
  }

  /// \brief Create an induced set for this class at \p rank if it doesn't already exist and return it.
  Class& get_or_create_induced_set(stk::mesh::EntityRank rank) {
    MUNDY_THROW_REQUIRE(
        is_primary(), std::logic_error,
        sink() << "Attempting to create an induced set of a class ('" << name() << "') that is not a primary class.");
    MUNDY_THROW_REQUIRE(rank < primary_entity_rank(), std::logic_error,
                        sink() << "Attempting to create an induced set for class '" << name() << "' at rank " << rank
                               << " that is greater than the class's primary entity rank of " << primary_entity_rank()
                               << ".");

    if (induced_sets_[rank] == nullptr) {
      const std::string set_name = name() + induced_set_suffix(rank);
      Class& induced_set = declare_class(mesh_meta_data(), set_name, rank);
      induced_sets_[rank] = &induced_set;

      // When a nodeset is created, it must be declared a subset of our supersets nodesets and (similarly) our subsets
      // nodesets must be declared subsets of it. This maintains the invariant that the nodeset hierarchy is always
      // synchronized with the data/assembly hierarchy.
      for (Class* subclass : subclasses()) {
        MUNDY_THROW_REQUIRE(subclass != nullptr, std::logic_error,
                            sink() << "Class '" << name() << "' has a null subclass pointer, which is invalid.");
        Class& sub_induced_set = subclass->get_or_create_induced_set(rank);
        induced_set.declare_subset(sub_induced_set);
      }
      for (Class* superclass : superclasses_) {
        MUNDY_THROW_REQUIRE(superclass != nullptr, std::logic_error,
                            sink() << "Class '" << name() << "' has a null superclass pointer, which is invalid.");
        // Supersets might not have nodesets and us having one doesn't require them to.
        if (superclass->has_induced_set(rank)) {
          Class& super_induced_set = superclass->get_or_create_induced_set(rank);
          super_induced_set.declare_subset(induced_set);
        }
      }
    }
    return *induced_sets_[rank];
  }

  /// \brief Get an induced set for this class at \p rank if it exists, otherwise throw
  Class& get_induced_set(stk::mesh::EntityRank rank) {
    MUNDY_THROW_REQUIRE(induced_sets_[rank] != nullptr, std::logic_error,
                        sink() << "Class '" << name() << "' does not have an induced set for the given rank.");
    return *induced_sets_[rank];
  }
  const Class& get_induced_set(stk::mesh::EntityRank rank) const {
    MUNDY_THROW_REQUIRE(induced_sets_[rank] != nullptr, std::logic_error,
                        sink() << "Class '" << name() << "' does not have an induced set for the given rank.");
    return *induced_sets_[rank];
  }

  bool has_induced_set(stk::mesh::EntityRank rank) const {
    return induced_sets_[rank] != nullptr;
  }

  /// \brief Equality comparison.
  bool operator==(const Class& rhs) const {
    return this == &rhs;
  }

  /// \brief Inequality comparison.
  bool operator!=(const Class& rhs) const {
    return this != &rhs;
  }

  /// \brief Query attribute attached to the class data part.
  template <class A>
  const A* attribute() const {
    return data_part_.template attribute<A>();
  }

  /// \brief Fetch the non-io data hierarchy part.
  stk::mesh::Part& data_part() noexcept {
    return data_part_;
  }

  /// \brief Fetch the non-io data hierarchy part.
  const stk::mesh::Part& data_part() const noexcept {
    return data_part_;
  }

  /// \brief Fetch the io leaf data part.
  stk::mesh::Part& leaf_part() noexcept {
    return leaf_part_;
  }

  /// \brief Fetch the io leaf data part.
  const stk::mesh::Part& leaf_part() const noexcept {
    return leaf_part_;
  }

  /// \brief Fetch the assembly hierarchy part.
  stk::mesh::Part& assembly_part() noexcept {
    return assembly_part_;
  }

  /// \brief Fetch the assembly hierarchy part.
  const stk::mesh::Part& assembly_part() const noexcept {
    return assembly_part_;
  }

  /// \brief Fetch the leaf assembly part.
  stk::mesh::Part& leaf_assembly_part() noexcept {
    return leaf_assembly_part_;
  }

  /// \brief Fetch the leaf assembly part.
  const stk::mesh::Part& leaf_assembly_part() const noexcept {
    return leaf_assembly_part_;
  }

  /// \brief Treat this class as the selector for its assembly hierarchy.
  ///
  /// The assembly part is the semantic membership channel for a Class: it selects all entities that are in this class
  /// or any of its subclasses. Field restrictions should still use `put_field_on_mesh(field, class_instance, ...)`,
  /// which writes to the data hierarchy instead of this selector channel.
  operator stk::mesh::Selector() const {
    return stk::mesh::Selector(assembly_part_);
  }

  /// \brief Treat this class as its assembly part for read-only STK APIs such as `Bucket::member(...)`.
  operator const stk::mesh::Part&() const noexcept {
    return assembly_part_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare \p sub_class as a direct subclass in both hierarchy channels.
  /// sub_class must have the same primary entity rank as this class.
  ///
  /// If sub_class has the same primary entity rank as this class, then its members are a subset of this class's
  /// members:
  ///  - `this.data_part      -> sub_class.data_part`
  ///  - `this.assembly_part  -> sub_class.assembly_part`
  void declare_subset(Class& sub_class) {
    MUNDY_THROW_REQUIRE(&meta_data_ == &sub_class.mesh_meta_data(), std::invalid_argument,
                        "Cannot declare subset relation across different MetaData instances.");
    MUNDY_THROW_REQUIRE(this != &sub_class, std::logic_error,
                        sink() << "Cannot declare class '" << name() << "' as a subclass of itself.");
    MUNDY_THROW_REQUIRE(
        primary_entity_rank() == sub_class.primary_entity_rank(), std::logic_error,
        sink() << "Cannot declare class '" << sub_class.name() << "' as a subclass of class '" << name()
               << "' because its primary entity rank is different than the primary entity rank of this class.");

    const bool already_direct_subclass =
        std::find(subclasses_.begin(), subclasses_.end(), &sub_class) != subclasses_.end();
    if (already_direct_subclass) {
      return;  // Subset already declared, nothing to do.
    }

    MUNDY_THROW_REQUIRE(!sub_class.contains(*this), std::logic_error,
                        sink() << "Declaring class '" << sub_class.name() << "' as a subclass of class '" << name()
                               << "' would create a class hierarchy cycle.");

    meta_data_.declare_part_subset(data_part_, sub_class.data_part_);
    meta_data_.declare_part_subset(assembly_part_, sub_class.assembly_part_);

    subclasses_.push_back(&sub_class);
    const_subclasses_.push_back(&sub_class);
    sub_class.superclasses_.push_back(this);
    sub_class.const_superclasses_.push_back(this);

    for (stk::mesh::EntityRank rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      if (has_induced_set(rank)) {
        Class& parent_induced_set = get_induced_set(rank);
        Class& sub_induced_set = sub_class.get_or_create_induced_set(rank);
        parent_induced_set.declare_subset(sub_induced_set);
      }
    }
  }

  /// \brief Assign deterministic ids to assembly-type parts for IO/visualization stability.
  void set_assembly_part_ids(unsigned assembly_part_id, unsigned leaf_assembly_part_id) {
    MUNDY_THROW_REQUIRE(assembly_part_id != leaf_assembly_part_id, std::invalid_argument,
                        sink() << "Assembly and leaf assembly part ids for class '" << name() << "' must be distinct.");
    meta_data_.set_part_id(assembly_part_, assembly_part_id);
    meta_data_.set_part_id(leaf_assembly_part_, leaf_assembly_part_id);
  }
  //@}

 private:
  //! \name Constructor
  //@{

  /// \brief Canonical constructor used by free declaration helpers.
  Class(stk::mesh::MetaData& meta_data, Type class_type, std::string class_name, class_ordinal_t class_ordinal,
        DeclarationSignature declaration_signature)
      : class_type_(class_type),
        class_name_(std::move(class_name)),
        class_ordinal_(class_ordinal),
        declaration_signature_(declaration_signature),
        meta_data_(meta_data),
        data_part_(declare_data_or_leaf_part(meta_data, class_name_ + data_part_suffix(), declaration_signature_)),
        leaf_part_(declare_data_or_leaf_part(meta_data, class_name_ + leaf_part_suffix(), declaration_signature_)),
        assembly_part_(meta_data.declare_part(class_name_ + assembly_part_suffix())),
        leaf_assembly_part_(meta_data.declare_part(class_name_ + leaf_assembly_part_suffix())) {
    MUNDY_THROW_REQUIRE(class_type_ != Type::SET || declaration_signature.declaration_kind == DeclarationKind::RANKED,
                        std::logic_error,
                        sink() << "Attempting to declare a set with name '" << class_name_
                               << "' using a non-ranked declaration signature, which is invalid.");

    for (stk::mesh::EntityRank rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
      induced_sets_[rank] = nullptr;
    }

    // Local four-part structure.
    meta_data_.declare_part_subset(data_part_, leaf_part_);
    meta_data_.declare_part_subset(assembly_part_, leaf_assembly_part_);
    meta_data_.declare_part_subset(leaf_assembly_part_, leaf_part_);

    // IO attributes: data is visible only through leafs/assemblies when IO support is enabled.
    if (has_io_support()) {
      stk::io::put_io_part_attribute(leaf_part_);
      stk::io::put_assembly_io_part_attribute(assembly_part_);
      stk::io::put_assembly_io_part_attribute(leaf_assembly_part_);
    }
  }
  //@}

  //! \name Helpers
  //@{

  static constexpr const char* data_part_suffix() noexcept {
    return "_DATA";
  }

  static constexpr const char* leaf_part_suffix() noexcept {
    return "_LEAF";
  }

  static constexpr const char* assembly_part_suffix() noexcept {
    return "";  // The assembly part contains all entities and connected entities of this class, so we identify it with
                // the class name itself.
  }

  static constexpr const char* leaf_assembly_part_suffix() noexcept {
    return "_LEAF_ASSEMBLY";
  }

  static const char* induced_set_suffix(stk::mesh::EntityRank rank) noexcept {
    if (rank == stk::topology::NODE_RANK) {
      return "_NODESET";
    } else if (rank == stk::topology::EDGE_RANK) {
      return "_EDGESET";
    } else if (rank == stk::topology::FACE_RANK) {
      return "_FACESET";
    } else if (rank == stk::topology::ELEMENT_RANK) {
      return "_ELEMSET";
    }

    return "_UNKNOWNSET";
  }

  static stk::mesh::Part& declare_data_or_leaf_part(stk::mesh::MetaData& meta_data, const std::string& part_name,
                                                    const DeclarationSignature& declaration_signature,
                                                    bool arg_force_no_induce = false) {
    switch (declaration_signature.declaration_kind) {
      case DeclarationKind::NAMED:
        return meta_data.declare_part(part_name);
      case DeclarationKind::RANKED:
        return meta_data.declare_part(part_name, declaration_signature.class_rank, arg_force_no_induce);
      case DeclarationKind::TOPOLOGICAL:
        return meta_data.declare_part_with_topology(part_name, declaration_signature.class_topology,
                                                    arg_force_no_induce);
      default:
        MUNDY_THROW_REQUIRE(false, std::logic_error, "Unexpected Class declaration kind.");
    }
    throw std::logic_error("Unreachable Class declaration kind.");
  }
  //@}

  //! \name Internal members
  //@{
  Type class_type_;
  std::string class_name_;
  class_ordinal_t class_ordinal_;
  DeclarationSignature declaration_signature_;
  stk::mesh::MetaData& meta_data_;
  stk::mesh::Part& data_part_;
  stk::mesh::Part& leaf_part_;
  stk::mesh::Part& assembly_part_;
  stk::mesh::Part& leaf_assembly_part_;

  Class* induced_sets_[stk::topology::NUM_RANKS];
  // TODO(palmerb4): Add support for edge/face sideset classes

  ClassVector subclasses_;
  ConstClassVector const_subclasses_;
  ClassVector superclasses_;
  ConstClassVector const_superclasses_;
  //@}

  //! \name Friends
  //@{
  friend struct impl::ClassFactory;
  //@}
};

namespace impl {

/// \brief Key used to track helper-managed output field registrations.
struct ClassFieldOutputKey {
  const stk::io::StkMeshIoBroker* io_broker = nullptr;
  size_t output_file_index = 0;
  const stk::mesh::FieldBase* field = nullptr;
  std::string db_name;

  /// \brief Strict weak ordering for use in std::map.
  bool operator<(const ClassFieldOutputKey& rhs) const {
    return std::tie(io_broker, output_file_index, field, db_name) <
           std::tie(rhs.io_broker, rhs.output_file_index, rhs.field, rhs.db_name);
  }
};

/// \brief Snapshot of Class-aware output registration decisions.
struct ClassFieldOutputSnapshot {
  bool is_nodeset_variable = false;
  std::vector<std::string> leaf_part_names;
};

/// \brief Output-registration state for Class-aware field output.
struct ClassIoRegistry {
  std::map<ClassFieldOutputKey, ClassFieldOutputSnapshot> snapshots;
};

/// \brief Fetch the Class IO registry from MetaData, creating it if needed.
inline ClassIoRegistry& get_or_create_class_io_registry(stk::mesh::MetaData& meta_data) {
  ClassIoRegistry* registry = const_cast<ClassIoRegistry*>(meta_data.get_attribute<ClassIoRegistry>());
  if (registry == nullptr) {
    const ClassIoRegistry* new_registry = new ClassIoRegistry();
    registry = const_cast<ClassIoRegistry*>(meta_data.declare_attribute_with_delete(new_registry));
  }
  return *registry;
}

/// \brief Internal storage for declared classes.
struct ClassMap {
  std::vector<std::unique_ptr<Class>> by_ordinal;
  std::map<std::string, Class*> by_name;
  ClassVector classes;
  ConstClassVector const_classes;

  /// \brief Append a freshly declared class and update all views atomically.
  Class& append(std::unique_ptr<Class> class_ptr) {
    MUNDY_THROW_REQUIRE(class_ptr != nullptr, std::logic_error, "Cannot append a null Class pointer.");
    Class* raw_class = class_ptr.get();
    MUNDY_THROW_REQUIRE(raw_class->class_ordinal() == by_ordinal.size(), std::logic_error,
                        sink() << "Class ordinal mismatch while appending class '" << raw_class->name() << "'.");
    MUNDY_THROW_REQUIRE(by_name.find(raw_class->name()) == by_name.end(), std::logic_error,
                        sink() << "Class '" << raw_class->name() << "' already exists in the ClassMap.");

    by_name.emplace(raw_class->name(), raw_class);
    classes.push_back(raw_class);
    const_classes.push_back(raw_class);
    by_ordinal.push_back(std::move(class_ptr));
    return *raw_class;
  }
};

/// \brief Attempt to fetch the class map from MetaData attributes.
inline ClassMap* try_get_class_map(const stk::mesh::MetaData& meta_data) {
  return const_cast<ClassMap*>(meta_data.get_attribute<ClassMap>());
}

/// \brief Fetch the class map from MetaData attributes, creating it if needed.
inline ClassMap& get_or_create_class_map(stk::mesh::MetaData& meta_data) {
  get_or_create_class_io_registry(meta_data);

  ClassMap* class_map = const_cast<ClassMap*>(meta_data.get_attribute<ClassMap>());
  if (class_map == nullptr) {
    const ClassMap* new_class_map = new ClassMap();
    class_map = const_cast<ClassMap*>(meta_data.declare_attribute_with_delete(new_class_map));
  }
  return *class_map;
}

/// \brief Internal privileged creator for Class instances.
struct ClassFactory {
  static std::unique_ptr<Class> create(stk::mesh::MetaData& meta_data, const Class::Type class_type,
                                       const std::string& class_name, Class::class_ordinal_t class_ordinal,
                                       Class::DeclarationSignature declaration_signature) {
    return std::unique_ptr<Class>(new Class(meta_data, class_type, class_name, class_ordinal, declaration_signature));
  }
};

/// \brief Human-readable declaration-kind name for diagnostics.
inline const char* declaration_kind_name(Class::DeclarationKind declaration_kind) {
  switch (declaration_kind) {
    case Class::DeclarationKind::NAMED:
      return "named";
    case Class::DeclarationKind::RANKED:
      return "ranked";
    case Class::DeclarationKind::TOPOLOGICAL:
      return "topological";
    default:
      return "unknown";
  }
}

inline const char* class_type_name(Class::Type class_type) {
  switch (class_type) {
    case Class::Type::PRIMARY:
      return "PRIMARY";
    case Class::Type::SET:
      return "SET";
    default:
      return "UNKNOWN";
  }
}

/// \brief Validate that an existing class matches a repeated declaration request.
inline void require_matching_declaration(const Class& class_instance, const Class::Type requested_class_type,
                                         const Class::DeclarationSignature& requested_signature) {
  const Class::DeclarationSignature& existing_signature = class_instance.declaration_signature();
  MUNDY_THROW_REQUIRE(
      class_instance.class_type() == requested_class_type, std::logic_error,
      sink() << "Repeated declaration of class '" << class_instance.name()
             << "' used incompatible class type. Existing type=" << class_type_name(class_instance.class_type())
             << ", requested type=" << class_type_name(requested_class_type) << '.');
  MUNDY_THROW_REQUIRE(
      existing_signature == requested_signature, std::logic_error,
      sink() << "Repeated declaration of class '" << class_instance.name()
             << "' used an incompatible signature. Existing signature is kind="
             << declaration_kind_name(existing_signature.declaration_kind) << ", rank=" << existing_signature.class_rank
             << ", topology=" << static_cast<unsigned>(existing_signature.class_topology)
             << ", disable_io_support=" << existing_signature.disable_io_support
             << "; requested signature is kind=" << declaration_kind_name(requested_signature.declaration_kind)
             << ", rank=" << requested_signature.class_rank
             << ", topology=" << static_cast<unsigned>(requested_signature.class_topology)
             << ", disable_io_support=" << requested_signature.disable_io_support << '.');
}

/// \brief Common internal declaration/get routine.
inline Class& declare_class_impl(stk::mesh::MetaData& meta_data, Class::Type class_type, const std::string& class_name,
                                 Class::DeclarationSignature declaration_signature) {
  if (class_type == Class::Type::PRIMARY) {
    const bool declares_node_rank_primary =
        (declaration_signature.declaration_kind == Class::DeclarationKind::RANKED &&
         declaration_signature.class_rank == stk::topology::NODE_RANK) ||
        (declaration_signature.declaration_kind == Class::DeclarationKind::TOPOLOGICAL &&
         stk::topology(declaration_signature.class_topology).rank() == stk::topology::NODE_RANK);
    MUNDY_THROW_REQUIRE(!declares_node_rank_primary || declaration_signature.disable_io_support, std::logic_error,
                        sink() << "Primary class declaration for NODE_RANK is unsupported for class '" << class_name
                               << "' unless IO support is disabled.");
  }
  if (class_type == Class::Type::SET) {
    MUNDY_THROW_REQUIRE(declaration_signature.declaration_kind == Class::DeclarationKind::RANKED, std::logic_error,
                        sink() << "Set class '" << class_name << "' must use ranked declaration kind.");
    MUNDY_THROW_REQUIRE(
        declaration_signature.class_rank != stk::topology::ELEMENT_RANK || declaration_signature.disable_io_support,
        std::logic_error,
        sink() << "Set class declaration for ELEMENT_RANK is unsupported for class '" << class_name
               << "' unless IO support is disabled.");
  }

  ClassMap& class_map = get_or_create_class_map(meta_data);

  auto name_it = class_map.by_name.find(class_name);
  if (name_it != class_map.by_name.end()) {
    require_matching_declaration(*name_it->second, class_type, declaration_signature);
    return *name_it->second;
  }

  MUNDY_THROW_REQUIRE(class_map.by_ordinal.size() <= std::numeric_limits<Class::class_ordinal_t>::max(),
                      std::overflow_error, "Class ordinal overflow while declaring a new Class.");
  const Class::class_ordinal_t class_ordinal = static_cast<Class::class_ordinal_t>(class_map.by_ordinal.size());

  std::unique_ptr<Class> class_ptr =
      ClassFactory::create(meta_data, class_type, class_name, class_ordinal, declaration_signature);
  return class_map.append(std::move(class_ptr));
}

/// \brief Return whether the broker already has the given field/name pair registered.
inline bool is_output_field_registered(const stk::io::StkMeshIoBroker& io_broker, size_t output_file_index,
                                       const stk::mesh::FieldBase& field, const std::string& db_name) {
  const std::vector<stk::io::FieldAndName>& named_fields = io_broker.get_defined_output_fields(output_file_index);
  for (const stk::io::FieldAndName& named_field : named_fields) {
    if (named_field.field() == &field && named_field.db_name() == db_name) {
      return true;
    }
  }
  return false;
}

/// \brief Collect IO leaf part names for the given classes whose leaf parts contain \p field.
inline std::vector<std::string> collect_field_leaf_part_names(const ClassVector& classes, stk::mesh::FieldBase& field) {
  std::vector<std::string> leaf_part_names;
  leaf_part_names.reserve(classes.size());
  for (const Class* class_instance : classes) {
    MUNDY_THROW_REQUIRE(
        class_instance != nullptr, std::logic_error,
        sink() << "Cannot collect field leaf part names from a null Class pointer for field '" << field.name() << "'.");
    const stk::mesh::Part& leaf_part = class_instance->leaf_part();
    if (stk::io::is_field_on_part(&field, stk::topology::NODE_RANK, leaf_part)) {
      const std::string leaf_part_name = stk::io::getPartName(leaf_part);
      const bool is_unique =
          std::find(leaf_part_names.begin(), leaf_part_names.end(), leaf_part_name) == leaf_part_names.end();
      MUNDY_THROW_REQUIRE(is_unique, std::logic_error,
                          sink() << "Class field output found duplicate IO leaf part name '" << leaf_part_name
                                 << "' while registering field '" << field.name() << "'.");
      leaf_part_names.push_back(leaf_part_name);
    }
  }

  std::sort(leaf_part_names.begin(), leaf_part_names.end());

  return leaf_part_names;
}

inline ClassVector filter_io_supported_classes(const ClassVector& classes) {
  ClassVector io_supported_classes;
  io_supported_classes.reserve(classes.size());
  for (Class* class_instance : classes) {
    MUNDY_THROW_REQUIRE(class_instance != nullptr, std::logic_error,
                        "Cannot filter IO-supported classes from a vector containing null Class pointers.");
    if (class_instance->has_io_support()) {
      io_supported_classes.push_back(class_instance);
    }
  }
  return io_supported_classes;
}

inline stk::mesh::ConstPartVector populate_entity_rank_parts(stk::mesh::EntityRank entity_rank,
                                                             const ClassVector& classes, const char* vector_name) {
  stk::mesh::ConstPartVector parts;
  bool found_primary_class = false;
  for (Class* class_instance : classes) {
    MUNDY_THROW_REQUIRE(class_instance != nullptr, std::logic_error,
                        sink() << "All classes in " << vector_name << " must be non-null.");
    if (class_instance->is_set()) {
      MUNDY_THROW_REQUIRE(class_instance->primary_entity_rank() == entity_rank, std::logic_error,
                          sink() << "Class set '" << class_instance->name() << "' in " << vector_name
                                 << " has primary entity rank " << class_instance->primary_entity_rank()
                                 << " that does not match the entity's rank " << entity_rank << '.');
      parts.push_back(&class_instance->leaf_part());
    } else {
      MUNDY_THROW_REQUIRE(class_instance->primary_entity_rank() >= entity_rank, std::logic_error,
                          sink() << "Primary class '" << class_instance->name() << "' in " << vector_name
                                 << " has primary entity rank " << class_instance->primary_entity_rank()
                                 << " that is less than the entity's rank " << entity_rank << '.');
      if (class_instance->primary_entity_rank() == entity_rank) {
        MUNDY_THROW_REQUIRE(!found_primary_class, std::logic_error,
                            sink() << "Multiple primary classes in " << vector_name
                                   << " have primary entity rank matching the entity's rank " << entity_rank << ": '"
                                   << class_instance->name() << "'.");
        found_primary_class = true;
      }

      // Implicit inherited part membership
      parts.push_back(&class_instance->leaf_part());

      // Explicit induced part membership
      if (class_instance->has_induced_set(entity_rank)) {
        parts.push_back(&class_instance->get_induced_set(entity_rank).leaf_part());
      }
    }
  }

  return parts;
}

}  // namespace impl

//! \name Class declaration helpers
//@{

/// \brief Declare (or fetch) a named class on the given MetaData.
inline Class& declare_class(stk::mesh::MetaData& meta_data, const std::string& class_name) {
  return impl::declare_class_impl(
      meta_data, Class::Type::PRIMARY, class_name,
      Class::DeclarationSignature{Class::DeclarationKind::NAMED, stk::topology::INVALID_RANK,
                                  stk::topology::INVALID_TOPOLOGY, /* disable_io_support */ false});
}

/// \brief Declare (or fetch) a ranked class-set on the given MetaData.
inline Class& declare_class(stk::mesh::MetaData& meta_data, const std::string& class_name,
                            stk::mesh::EntityRank class_rank, bool disable_io_support) {
  return impl::declare_class_impl(meta_data, Class::Type::SET, class_name,
                                  Class::DeclarationSignature{Class::DeclarationKind::RANKED, class_rank,
                                                              stk::topology::INVALID_TOPOLOGY, disable_io_support});
}

/// \brief Declare (or fetch) a topological primary class on the given MetaData.
inline Class& declare_class(stk::mesh::MetaData& meta_data, const std::string& class_name,
                            stk::topology::topology_t class_topology, bool disable_io_support) {
  return impl::declare_class_impl(
      meta_data, Class::Type::PRIMARY, class_name,
      Class::DeclarationSignature{Class::DeclarationKind::TOPOLOGICAL, stk::topology::INVALID_RANK, class_topology,
                                  disable_io_support});
}
//@}

//! \name Query helpers
//@{

/// \brief Fetch an existing class by ordinal from MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline Class& get_class(stk::mesh::MetaData& meta_data, Class::class_ordinal_t class_ordinal) {
  impl::ClassMap* class_map = impl::try_get_class_map(meta_data);
  MUNDY_THROW_REQUIRE(class_map != nullptr, std::invalid_argument,
                      "No ClassMap found on MetaData. Declare classes before calling get_class.");

  MUNDY_THROW_REQUIRE(class_ordinal < class_map->by_ordinal.size(), std::invalid_argument,
                      sink() << "No class exists with ordinal " << class_ordinal << '.');
  return *class_map->by_ordinal[class_ordinal];
}

/// \brief Fetch an existing class by ordinal from MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline const Class& get_class(const stk::mesh::MetaData& meta_data, Class::class_ordinal_t class_ordinal) {
  impl::ClassMap* class_map = impl::try_get_class_map(meta_data);
  MUNDY_THROW_REQUIRE(class_map != nullptr, std::invalid_argument,
                      "No ClassMap found on MetaData. Declare classes before calling get_class.");

  MUNDY_THROW_REQUIRE(class_ordinal < class_map->by_ordinal.size(), std::invalid_argument,
                      sink() << "No class exists with ordinal " << class_ordinal << '.');
  return *class_map->by_ordinal[class_ordinal];
}

/// \brief Fetch an existing class by name from MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline Class& get_class(stk::mesh::MetaData& meta_data, const std::string& class_name) {
  impl::ClassMap* class_map = impl::try_get_class_map(meta_data);
  MUNDY_THROW_REQUIRE(class_map != nullptr, std::invalid_argument,
                      "No ClassMap found on MetaData. Declare classes before calling get_class.");

  auto name_it = class_map->by_name.find(class_name);
  MUNDY_THROW_REQUIRE(name_it != class_map->by_name.end(), std::invalid_argument,
                      sink() << "No class exists with name '" << class_name << "'.");
  return *name_it->second;
}

/// \brief Fetch an existing class by name from MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline const Class& get_class(const stk::mesh::MetaData& meta_data, const std::string& class_name) {
  impl::ClassMap* class_map = impl::try_get_class_map(meta_data);
  MUNDY_THROW_REQUIRE(class_map != nullptr, std::invalid_argument,
                      "No ClassMap found on MetaData. Declare classes before calling get_class.");

  auto name_it = class_map->by_name.find(class_name);
  MUNDY_THROW_REQUIRE(name_it != class_map->by_name.end(), std::invalid_argument,
                      sink() << "No class exists with name '" << class_name << "'.");
  return *name_it->second;
}

/// \brief Fetch all declared classes from MetaData in class-ordinal order.
///
/// The returned pointers are non-owning. Class instances are owned by the class map attached to \p meta_data. No
/// per-call allocation is performed.
///
/// \throws std::invalid_argument if no class map exists.
inline const ClassVector& get_classes(stk::mesh::MetaData& meta_data) {
  impl::ClassMap* class_map = impl::try_get_class_map(meta_data);
  MUNDY_THROW_REQUIRE(class_map != nullptr, std::invalid_argument,
                      "No ClassMap found on MetaData. Declare classes before calling get_classes.");
  return class_map->classes;
}

/// \brief Fetch all declared classes from MetaData in class-ordinal order.
///
/// The returned pointers are non-owning. Class instances are owned by the class map attached to \p meta_data. No
/// per-call allocation is performed.
///
/// \throws std::invalid_argument if no class map exists.
inline const ConstClassVector& get_classes(const stk::mesh::MetaData& meta_data) {
  impl::ClassMap* class_map = impl::try_get_class_map(meta_data);
  MUNDY_THROW_REQUIRE(class_map != nullptr, std::invalid_argument,
                      "No ClassMap found on MetaData. Declare classes before calling get_classes.");
  return class_map->const_classes;
}

/// \brief Fetch an existing class by ordinal through the given BulkData's MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline Class& get_class(stk::mesh::BulkData& bulk_data, Class::class_ordinal_t class_ordinal) {
  return get_class(bulk_data.mesh_meta_data(), class_ordinal);
}

/// \brief Fetch an existing class by ordinal through the given BulkData's MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline const Class& get_class(const stk::mesh::BulkData& bulk_data, Class::class_ordinal_t class_ordinal) {
  return get_class(bulk_data.mesh_meta_data(), class_ordinal);
}

/// \brief Fetch an existing class by name through the given BulkData's MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline Class& get_class(stk::mesh::BulkData& bulk_data, const std::string& class_name) {
  return get_class(bulk_data.mesh_meta_data(), class_name);
}

/// \brief Fetch an existing class by name through the given BulkData's MetaData.
/// \throws std::invalid_argument if no class map exists or no matching class is found.
inline const Class& get_class(const stk::mesh::BulkData& bulk_data, const std::string& class_name) {
  return get_class(bulk_data.mesh_meta_data(), class_name);
}

/// \brief Fetch all classes visible from BulkData in class-ordinal order.
///
/// The returned vector is the canonical ordinal-ordered view stored on `bulk_data.mesh_meta_data()`. No per-call
/// allocation is performed.
///
/// \throws std::invalid_argument if no class map exists.
inline const ClassVector& get_classes(stk::mesh::BulkData& bulk_data) {
  return get_classes(bulk_data.mesh_meta_data());
}

/// \brief Fetch all classes visible from BulkData in class-ordinal order.
///
/// The returned vector is the canonical ordinal-ordered view stored on `bulk_data.mesh_meta_data()`. No per-call
/// allocation is performed.
///
/// \throws std::invalid_argument if no class map exists.
inline const ConstClassVector& get_classes(const stk::mesh::BulkData& bulk_data) {
  return get_classes(bulk_data.mesh_meta_data());
}
//@}

//! \name Non-member actions
//@{

/// \brief Declare \p sub_class as a subclass of \p parent_class in both hierarchy channels.
inline void declare_subset(Class& parent_class, Class& sub_class) {
  parent_class.declare_subset(sub_class);
}

/// \brief Intersection of a class assembly selector and another class assembly selector.
inline stk::mesh::Selector operator&(const Class& lhs, const Class& rhs) {
  stk::mesh::Selector selector(lhs.assembly_part());
  selector &= stk::mesh::Selector(rhs.assembly_part());
  return selector;
}

/// \brief Intersection of a class assembly selector and an STK selector.
inline stk::mesh::Selector operator&(const Class& lhs, const stk::mesh::Selector& rhs) {
  stk::mesh::Selector selector(lhs.assembly_part());
  selector &= rhs;
  return selector;
}

/// \brief Intersection of an STK selector and a class assembly selector.
inline stk::mesh::Selector operator&(const stk::mesh::Selector& lhs, const Class& rhs) {
  stk::mesh::Selector selector(lhs);
  selector &= stk::mesh::Selector(rhs.assembly_part());
  return selector;
}

/// \brief Union of a class assembly selector and another class assembly selector.
inline stk::mesh::Selector operator|(const Class& lhs, const Class& rhs) {
  stk::mesh::Selector selector(lhs.assembly_part());
  selector |= stk::mesh::Selector(rhs.assembly_part());
  return selector;
}

/// \brief Union of a class assembly selector and an STK selector.
inline stk::mesh::Selector operator|(const Class& lhs, const stk::mesh::Selector& rhs) {
  stk::mesh::Selector selector(lhs.assembly_part());
  selector |= rhs;
  return selector;
}

/// \brief Union of an STK selector and a class assembly selector.
inline stk::mesh::Selector operator|(const stk::mesh::Selector& lhs, const Class& rhs) {
  stk::mesh::Selector selector(lhs);
  selector |= stk::mesh::Selector(rhs.assembly_part());
  return selector;
}

/// \brief Difference of a class assembly selector and another class assembly selector.
inline stk::mesh::Selector operator-(const Class& lhs, const Class& rhs) {
  stk::mesh::Selector selector(lhs.assembly_part());
  selector -= stk::mesh::Selector(rhs.assembly_part());
  return selector;
}

/// \brief Difference of a class assembly selector and an STK selector.
inline stk::mesh::Selector operator-(const Class& lhs, const stk::mesh::Selector& rhs) {
  stk::mesh::Selector selector(lhs.assembly_part());
  selector -= rhs;
  return selector;
}

/// \brief Difference of an STK selector and a class assembly selector.
inline stk::mesh::Selector operator-(const stk::mesh::Selector& lhs, const Class& rhs) {
  stk::mesh::Selector selector(lhs);
  selector -= stk::mesh::Selector(rhs.assembly_part());
  return selector;
}

/// \brief Complement of a class assembly selector.
inline stk::mesh::Selector operator!(const Class& class_instance) {
  stk::mesh::Selector selector(class_instance.assembly_part());
  return selector.complement();
}

/// \brief Register a field for output using Class-aware IO rules over an explicit class set.
///
/// Node-rank fields are written as nodeset variables on the class leaf data parts in \p classes that contain the
/// field. Other field ranks are passed through the normal `StkMeshIoBroker::add_field(...)` path. Repeated calls
/// through this helper are idempotent only when the rediscovered Class-aware output parameters match the parameters
/// recorded by the first call.
///
/// Throws std::logic_error if \p field is already registered by another output-registration path.
inline void add_class_field(stk::io::StkMeshIoBroker& io_broker, size_t output_file_index, stk::mesh::FieldBase& field,
                            const ClassVector& classes, const std::string& db_name) {
  stk::mesh::BulkData& bulk_data = io_broker.bulk_data();
  for (const Class* class_instance : classes) {
    MUNDY_THROW_REQUIRE(
        class_instance != nullptr, std::logic_error,
        sink() << "Cannot register field '" << field.name() << "' with a class vector containing null Class pointers.");
    MUNDY_THROW_REQUIRE(class_instance->has_io_support(), std::logic_error,
                        sink() << "Cannot register field '" << field.name() << "' with class '"
                               << class_instance->name() << "' because that class does not support IO.");
  }

  const bool is_node_rank_field = field.entity_rank() == stk::topology::NODE_RANK;
  std::vector<std::string> leaf_part_names;
  if (is_node_rank_field) {
    leaf_part_names = impl::collect_field_leaf_part_names(classes, field);
    std::cout << "add_class_field: nodeset field '" << field.name() << "' is on leaves: ";
    for (const std::string& leaf_part_name : leaf_part_names) {
      std::cout << leaf_part_name << ' ';
    }
    std::cout << std::endl;
  }

  const bool already_registered = impl::is_output_field_registered(io_broker, output_file_index, field, db_name);
  impl::ClassIoRegistry& output_registry = impl::get_or_create_class_io_registry(bulk_data.mesh_meta_data());
  const impl::ClassFieldOutputKey output_key{&io_broker, output_file_index, &field, db_name};
  auto snapshot_it = output_registry.snapshots.find(output_key);

  if (snapshot_it != output_registry.snapshots.end()) {
    MUNDY_THROW_REQUIRE(snapshot_it->second.is_nodeset_variable == is_node_rank_field, std::logic_error,
                        sink() << "Repeated class field output registration for field '" << field.name()
                               << "' changed nodeset-variable mode.");
    MUNDY_THROW_REQUIRE(snapshot_it->second.leaf_part_names == leaf_part_names, std::logic_error,
                        sink() << "Repeated class field output registration for field '" << field.name()
                               << "' generated inconsistent class leaf subset information.");
    MUNDY_THROW_REQUIRE(already_registered, std::logic_error,
                        sink() << "Class field output registry contains field '" << field.name()
                               << "', but the IO broker does not have a matching output field registration.");
    return;
  }

  if (already_registered) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        sink() << "Field '" << field.name()
                               << "' is already registered on this output file, but not by add_class_field. "
                                  "Use add_class_field for the first registration so Class-aware output parameters are "
                                  "known and verifiable.");
  }

  if (is_node_rank_field) {
    stk::io::OutputVariableParams params(db_name);
    params.is_nodeset_variable(true);
    params.set_subset_info(/*isInclude=*/true, leaf_part_names);
    io_broker.add_field(output_file_index, field, stk::topology::NODE_RANK, params);
  } else if (db_name == field.name()) {
    io_broker.add_field(output_file_index, field);
  } else {
    io_broker.add_field(output_file_index, field, db_name);
  }

  output_registry.snapshots.emplace(output_key, impl::ClassFieldOutputSnapshot{is_node_rank_field, leaf_part_names});
}

/// \brief Register a field for output using Class-aware IO rules over an explicit class set.
inline void add_class_field(stk::io::StkMeshIoBroker& io_broker, size_t output_file_index, stk::mesh::FieldBase& field,
                            const ClassVector& classes) {
  add_class_field(io_broker, output_file_index, field, classes, field.name());
}

/// \brief Register a field for output using Class-aware IO rules.
///
/// This overload discovers all declared classes on the broker bulk data and forwards to the explicit-classes overload.
inline void add_class_field(stk::io::StkMeshIoBroker& io_broker, size_t output_file_index, stk::mesh::FieldBase& field,
                            const std::string& db_name) {
  add_class_field(io_broker, output_file_index, field,
                  impl::filter_io_supported_classes(get_classes(io_broker.bulk_data().mesh_meta_data())), db_name);
}

/// \brief Register a field for output using Class-aware IO rules.
inline void add_class_field(stk::io::StkMeshIoBroker& io_broker, size_t output_file_index,
                            stk::mesh::FieldBase& field) {
  add_class_field(io_broker, output_file_index, field, field.name());
}

/// \brief Put a rank-0/1 field restriction on a class data part.
template <typename FieldType>
inline void put_field_on_mesh(FieldType& field, Class& class_instance,
                              const typename FieldType::value_type* init_value) {
  if (field.entity_rank() != class_instance.primary_entity_rank()) {
    MUNDY_THROW_REQUIRE(class_instance.is_primary(), std::logic_error,
                        sink() << "Cannot put rank-" << field.entity_rank() << " field '" << field.name()
                               << "' on set class '" << class_instance.name()
                               << "' via induced-set path because sets do not induce.");
    Class& induced_set = class_instance.get_or_create_induced_set(field.entity_rank());
    stk::mesh::put_field_on_mesh(field, induced_set.data_part() | induced_set.leaf_part(), init_value);
  } else {
    stk::mesh::put_field_on_mesh(field, class_instance.data_part() | class_instance.leaf_part(), init_value);
  }
}

/// \brief Put a rank-1 field restriction on a class data part.
template <typename FieldType>
inline void put_field_on_mesh(FieldType& field, Class& class_instance, unsigned n1,
                              const typename FieldType::value_type* init_value) {
  if (field.entity_rank() != class_instance.primary_entity_rank()) {
    MUNDY_THROW_REQUIRE(class_instance.is_primary(), std::logic_error,
                        sink() << "Cannot put rank-" << field.entity_rank() << " field '" << field.name()
                               << "' on set class '" << class_instance.name()
                               << "' via induced-set path because sets do not induce.");
    Class& induced_set = class_instance.get_or_create_induced_set(field.entity_rank());
    stk::mesh::put_field_on_mesh(field, induced_set.data_part() | induced_set.leaf_part(), n1, init_value);
  } else {
    stk::mesh::put_field_on_mesh(field, class_instance.data_part() | class_instance.leaf_part(), n1, init_value);
  }
}

/// \brief Put a rank-2 field restriction on a class data part.
template <typename FieldType>
inline void put_field_on_mesh(FieldType& field, Class& class_instance, unsigned n1, unsigned n2,
                              const typename FieldType::value_type* init_value) {
  if (field.entity_rank() != class_instance.primary_entity_rank()) {
    MUNDY_THROW_REQUIRE(class_instance.is_primary(), std::logic_error,
                        sink() << "Cannot put rank-" << field.entity_rank() << " field '" << field.name()
                               << "' on set class '" << class_instance.name()
                               << "' via induced-set path because sets do not induce.");
    Class& induced_set = class_instance.get_or_create_induced_set(field.entity_rank());
    stk::mesh::put_field_on_mesh(field, induced_set.data_part() | induced_set.leaf_part(), n1, n2, init_value);
  } else {
    stk::mesh::put_field_on_mesh(field, class_instance.data_part() | class_instance.leaf_part(), n1, n2, init_value);
  }
}
//@}

//! \name Entity declaration interface
//@{

struct BulkDataClassInterface {
  stk::mesh::BulkData& bulk_data;

  inline ClassVector get_matching_rank_primary_classes(const stk::mesh::Entity entity) const {
    MUNDY_THROW_REQUIRE(bulk_data.is_valid(entity), std::invalid_argument,
                        "Cannot fetch matching-rank primary classes for invalid entity.");
    const stk::mesh::EntityRank entity_rank = bulk_data.entity_rank(entity);
    const stk::mesh::Bucket& bucket = bulk_data.bucket(entity);

    ClassVector matching_primary_classes;
    const ClassVector& all_classes = get_classes(bulk_data);
    for (Class* class_instance : all_classes) {
      MUNDY_THROW_REQUIRE(class_instance != nullptr, std::logic_error, "Encountered null class pointer.");
      if (class_instance->is_primary() && class_instance->primary_entity_rank() == entity_rank &&
          bucket.member(class_instance->leaf_part())) {
        matching_primary_classes.push_back(class_instance);
      }
    }
    return matching_primary_classes;
  }

  /// \brief Create or retrieve a locally owned entity of a given rank and id.
  ///
  ///  A parallel-local operation.
  ///
  ///  The entity is created as locally owned and a member of the input
  ///  mesh parts. The entity a member of the meta data's locally owned
  ///  mesh part and the entity's owner_rank() == parallel_rank().
  ///
  ///  If two or more processes create an entity of the same rank
  ///  and identifier then the sharing and ownership of these entities
  ///  will be resolved by the call to 'modification_end'.
  ///
  /// The vector of classes may contain primary classes or class sets; they must satisfy the following constraints:
  ///   - All class sets must have a primary entity rank equal to the entity's rank.
  ///   - At most one of the primary classes may have a primary entity rank matching the entity's rank.
  ///   - All other primary classes must have a primary entity rank greater than the entity's rank.
  inline stk::mesh::Entity declare_entity(stk::mesh::EntityRank rank, stk::mesh::EntityId id,
                                          const ClassVector& class_vector) {
    MUNDY_THROW_REQUIRE(!class_vector.empty(), std::logic_error,
                        "Cannot declare an entity with empty class membership.");
    stk::mesh::ConstPartVector part_vector =
        impl::populate_entity_rank_parts(rank, class_vector, "declare_entity class_vector");
    return bulk_data.declare_entity(rank, id, part_vector);
  }
  inline stk::mesh::Entity declare_node(stk::mesh::EntityId id, const ClassVector& class_vector) {
    return declare_entity(stk::topology::NODE_RANK, id, class_vector);
  }
  inline stk::mesh::Entity declare_edge(stk::mesh::EntityId id, const ClassVector& class_vector) {
    return declare_entity(stk::topology::EDGE_RANK, id, class_vector);
  }
  inline stk::mesh::Entity declare_element(stk::mesh::EntityId id, const ClassVector& class_vector) {
    return declare_entity(stk::topology::ELEMENT_RANK, id, class_vector);
  }

  /// \brief Same as declare_entity but with a single class instance
  inline stk::mesh::Entity declare_entity(stk::mesh::EntityRank rank, stk::mesh::EntityId id,
                                          const Class& class_instance) {
    // TODO(palmerb4): Long term, this has small performance implications due to the extra vector construction and
    // destruction; add a populate_entity_rank_parts for a single class but with duplicative, albeit somewhat
    // simplified, logic.
    return declare_entity(rank, id, ClassVector{const_cast<Class*>(&class_instance)});
  }
  inline stk::mesh::Entity declare_node(stk::mesh::EntityId id, const Class& class_instance) {
    return declare_entity(stk::topology::NODE_RANK, id, class_instance);
  }
  inline stk::mesh::Entity declare_edge(stk::mesh::EntityId id, const Class& class_instance) {
    return declare_entity(stk::topology::EDGE_RANK, id, class_instance);
  }
  inline stk::mesh::Entity declare_element(stk::mesh::EntityId id, const Class& class_instance) {
    return declare_entity(stk::topology::ELEMENT_RANK, id, class_instance);
  }

  /// \brief Change the parallel-locally-owned entity's class membership by swapping class membership (only valid for
  /// entities of non-primary rank for all involved classes).
  ///
  /// The vector of classes may contain primary classes or class sets; they must satisfy the following constraints:
  ///   - All class sets must have a primary entity rank equal to the entity's rank.
  ///   - At most one of the primary classes may have a primary entity rank matching the entity's rank.
  ///   - All other primary classes must have a primary entity rank greater than the entity's rank.
  ///
  /// The entity will be added (or removed) from the leaf parts of all given class sets, all induced sets of classes
  /// with primary entity rank greater than the entity's rank, and all primary classes with equal primary entity rank.
  inline void change_entity_classes(const stk::mesh::Entity entity, const ClassVector& add_classes,
                                    const ClassVector& remove_classes = ClassVector()) {
    MUNDY_THROW_REQUIRE(bulk_data.is_valid(entity), std::invalid_argument,
                        "Cannot change class membership for invalid entity.");
    stk::mesh::EntityRank entity_rank = bulk_data.entity_rank(entity);

    auto find_matching_primary = [entity_rank](const ClassVector& classes) {
      Class* matching_primary = nullptr;
      for (Class* class_instance : classes) {
        MUNDY_THROW_REQUIRE(class_instance != nullptr, std::logic_error,
                            "Class vectors must not contain null pointers.");
        if (class_instance->is_primary() && class_instance->primary_entity_rank() == entity_rank) {
          MUNDY_THROW_REQUIRE(matching_primary == nullptr, std::logic_error,
                              sink() << "Multiple matching-rank primary classes found for rank " << entity_rank << '.');
          matching_primary = class_instance;
        }
      }
      return matching_primary;
    };

    const ClassVector current_matching_primary_classes = get_matching_rank_primary_classes(entity);
    MUNDY_THROW_REQUIRE(
        current_matching_primary_classes.size() <= 1u, std::logic_error,
        sink() << "Entity id " << bulk_data.identifier(entity)
               << " currently belongs to multiple matching-rank primary classes, which violates class invariants.");

    Class* current_matching_primary =
        current_matching_primary_classes.empty() ? nullptr : current_matching_primary_classes.front();
    Class* add_matching_primary = find_matching_primary(add_classes);
    Class* remove_matching_primary = find_matching_primary(remove_classes);

    if (add_matching_primary != nullptr) {
      if (current_matching_primary == nullptr) {
        // allowed
      } else if (remove_matching_primary == current_matching_primary &&
                 add_matching_primary != current_matching_primary) {
        // atomic swap allowed
      } else {
        MUNDY_THROW_REQUIRE(false, std::logic_error,
                            sink() << "Cannot add matching-rank primary class '" << add_matching_primary->name()
                                   << "' for entity id " << bulk_data.identifier(entity)
                                   << " because it already belongs to matching-rank primary class '"
                                   << current_matching_primary->name()
                                   << "'. Remove the current primary in the same call to perform an atomic swap.");
      }
    }

    if (remove_matching_primary != nullptr && current_matching_primary != nullptr) {
      MUNDY_THROW_REQUIRE(
          remove_matching_primary == current_matching_primary || add_matching_primary != nullptr, std::logic_error,
          sink() << "Cannot remove matching-rank primary class '" << remove_matching_primary->name()
                 << "' from entity id " << bulk_data.identifier(entity)
                 << " because current matching-rank primary class is '" << current_matching_primary->name() << "'.");
    }

    stk::mesh::ConstPartVector add_parts = impl::populate_entity_rank_parts(entity_rank, add_classes, "add_classes");
    stk::mesh::ConstPartVector remove_parts =
        impl::populate_entity_rank_parts(entity_rank, remove_classes, "remove_classes");
    bulk_data.change_entity_parts(entity, add_parts, remove_parts);

    const ClassVector final_matching_primary_classes = get_matching_rank_primary_classes(entity);
    MUNDY_THROW_REQUIRE(final_matching_primary_classes.size() <= 1u, std::logic_error,
                        sink() << "Entity id " << bulk_data.identifier(entity)
                               << " has multiple matching-rank primary class memberships after change_entity_classes.");
  }

  inline void change_entity_classes(const stk::mesh::Entity entity, const Class& add_class,
                                    const ClassVector& remove_classes = ClassVector()) {
    return change_entity_classes(entity, ClassVector{const_cast<Class*>(&add_class)}, remove_classes);
  }

  inline void change_entity_classes(const stk::mesh::Entity entity, const ClassVector& add_classes,
                                    const Class& remove_class) {
    return change_entity_classes(entity, add_classes, ClassVector{const_cast<Class*>(&remove_class)});
  }
};

inline BulkDataClassInterface class_interface(stk::mesh::BulkData& bulk_data) {
  return BulkDataClassInterface{bulk_data};
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_CLASS_HPP_
