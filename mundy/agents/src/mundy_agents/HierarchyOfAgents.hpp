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

#ifndef MUNDY_AGENTS_HIERARCHYOFAGENTS_HPP_
#define MUNDY_AGENTS_HIERARCHYOFAGENTS_HPP_

/// \file HierarchyOfAgents.hpp
/// \brief Declaration of the HierarchyOfAgents class

// C++ core libs
#include <functional>     // for std::function
#include <iostream>       // for std::cout, std::endl
#include <map>            // for std::map
#include <memory>         // for std::shared_ptr
#include <stdexcept>      // for std::invalid_argument
#include <string>         // for std::string
#include <unordered_map>  // for std::unordered_map
#include <utility>        // for std::make_pair
#include <vector>         // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agents/IsAgentType.hpp>  // for mundy::agents::IsAgentType
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT

namespace mundy {

namespace agents {

/// \brief Unique agent identifier
/// In the current design, an "agent" is a Part endowed with a set of default requirements. Each agent can be
/// uniquely identified by either the agent's part or a unique uint, namely agent_t.
///
/// \note Agent_t is not a class enum, so comparing two agent_t's equates to comparing two unsigned ints (same as
/// regular enums). However, agent_t is unique in that no two agents will share the same agent_t. It is important to
/// note that, while agent_t is unique, the agent_t assigned to an agent need not be the same between consecutive
/// compilations of the code (due to the static initialization order fiasco).
using agent_t = unsigned;

/// \class HierarchyOfAgents
/// \brief A factory containing generation routines for different Agent types that form a hierarchical structure.
///
/// In the current design, it's best to directly interact with the Agent classes themselves. This factory is used to
/// fetch the static methods associated with an agent based on the agent's unique name or agent_t and requires special
/// care to ensure that all agent types are registered before they are used.
///
/// Agents are used to streamline the requirements generation process. Typically, an agent will hardcode only the bare
/// minimum requirements (e.g., all spheres have a radius and a center). All other requirements should be added at
/// run-time. Either one uses an agent directly, modifies the requirements of an agent, or inherits from/subsets an
/// agent. Typically, subsetting is more appropriate than direct modification since it's unlikely that you want ALL
/// agents of a type to have a certain field other than their default fields. If a subset occurs repeatedly in your
/// problem, consider creating and registering your own agent.
///
/// \code{.cpp}
/// class ExampleAgent {
///  public:
///   //! \name Getters
///   //@{
///
///   /// \brief Get the ExampleAgent's name.
///   static constexpr inline std::string get_name();
///
///   /// \brief Get the names of ExampleAgent's parents.
///   static constexpr inline std::vector<std::string> get_parent_names();
///
///   /// \brief Get the ExampleAgent's topology (throws if the part doesn't constrain topology)
///   static constexpr inline stk::topology::topology_t get_topology();
///
///   /// \brief Get the ExampleAgent's rank (throws if the part doesn't constrain rank)
///   static constexpr inline stk::topology::rank_t get_rank();
///
///   /// \brief Get if the ExampleAgent constrains the part's topology.
///   static constexpr inline bool has_topology();
///
///   /// \brief Get if the ExampleAgent constrains the part's rank.
///   static constexpr inline bool has_rank();
///
///   /// \brief Add new part requirements to ALL members of this agent part.
///   /// These modifications are reflected in our mesh requirements.
///   static inline void add_and_sync_part_reqs(std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr);
///
///   /// \brief Add sub-part requirements.
///   /// These modifications are reflected in our mesh requirements.
///   static inline void add_and_sync_subpart_reqs(std::shared_ptr<mundy::meta::PartReqs> subpart_reqs_ptr);
///
///   /// \brief Get the mesh requirements for the ExampleAgent.
///   static inline std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements();
/// };  // ExampleAgent
/// \endcode
///
/// This class is a tree of AgentFactories that can be used to generate the different types of agents. For example,
/// imagine a simple hierarchy of agents:
///                             Agents
///                  |                           |
///               Shapes                    Constraints
///    |         |           |             |       |      |
/// Spheres Ellipsoids Spherocylinders  Springs Hinges Joints
///
/// In this example, every node in the tree has a unique name and knows the names of its parents. The uniqueness of part
/// names means that we can create a on-to-one map from name to agent_t and use either to access the internal member
/// functions of the registered agent in the hierarchy.
///
/// Because this factory is constructed at static initialization time, the order in which nodes in the tree
/// are registered is not guaranteed (due to the static initialization order fiasco). As a result, the id's of the nodes
/// in the tree are not guaranteed to be in the order you would expect and they need not be consistent between runs. If
/// this is every required, tell the developers and we can try to use post-registration sorting to ensure that the id's
/// are consistent.
///
/// It's important to note that the topology/rank of each part in this hierarchy take some consideration when building
/// your agents. For example, if certain shapes have different topologies than others, how should one construct the
/// Shapes part to allow for proper subsetting. Because topologies are inherited and changing the topology of subparts
/// is invalid, the proper way to think of parts like Shapes, Constraints, and Agents is as "assemblies." Assemblies are
/// specialized parts that are meant to contain any number of sub-parts with arbitrary topologies/ranks. STK identifies
/// assemblies as ranked parts with INVALID_RANK.
class HierarchyOfAgents {
 public:
  //! \name Getters
  //@{

  /// \brief Get the number of classes this factory recognizes.
  static unsigned get_number_of_registered_types();

  /// \brief Get if the provided name is valid or not
  /// \param name [in] A string name that may or may not correspond to a registered class.
  static bool is_valid(const std::string& name);

  /// \brief Get if the provided agent_type is valid or not
  /// \param agent_type [in] A agent_type that may or may not correspond to a registered class.
  static bool is_valid(const agent_t agent_type);

  /// \brief Throw if the provided name is invalid
  /// \param name [in] A string name that may or may not correspond to a registered class.
  static void assert_is_valid(const std::string& name);

  /// \brief Throw if the provided agent_type is invalid
  /// \param agent_type [in] A agent_type that may or may not correspond to a registered class.
  static void assert_is_valid(const agent_t agent_type);

  /// \brief Get the agent_type corresponding to a registered class with the given name.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static agent_t get_agent_type(const std::string& name);

  /// \brief Get the name corresponding to the registered class with the given agent_type.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static std::string get_name(const agent_t agent_type);

  /// \brief Get the names corresponding to the parents of the registered class with the given name.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static std::vector<std::string> get_parent_names(const std::string& name);

  /// \brief Get the names corresponding to the parents of the registered class with the given agent_type.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static std::vector<std::string> get_parent_names(const agent_t agent_type);

  /// \brief Get the topology corresponding to a registered class with the given agent_type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static stk::topology::topology_t get_topology(const std::string& name);

  /// \brief Get the topology corresponding to a registered class with the given agent_type.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static stk::topology::topology_t get_topology(const agent_t agent_type);

  /// \brief Get the rank corresponding to a registered class with the given agent_type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static stk::topology::rank_t get_rank(const std::string& name);

  /// \brief Get the rank corresponding to a registered class with the given agent_type.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static stk::topology::rank_t get_rank(const agent_t agent_type);

  /// \brief Add new part requirements to ALL members of the specified agent part.
  /// \param name [in] A string name that correspond to a registered class.
  /// \param part_reqs_ptr [in] A pointer to the part requirements to add.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static void add_and_sync_part_reqs(std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr, const std::string& name);

  /// \brief Add new part requirements to ALL members of the specified agent part.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static void add_and_sync_part_reqs(std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr, const agent_t agent_type);

  /// \brief Add new sub-part requirements to ALL members of the specified agent part.
  /// \param name [in] A string name that correspond to a registered class.
  /// \param subpart_reqs_ptr [in] A pointer to the sub-part requirements to add.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static void add_and_sync_subpart_reqs(std::shared_ptr<mundy::meta::PartReqs> subpart_reqs_ptr,
                                        const std::string& name);

  /// \brief Add new sub-part requirements to ALL members of the specified agent part.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static void add_and_sync_subpart_reqs(std::shared_ptr<mundy::meta::PartReqs> subpart_reqs_ptr,
                                        const agent_t agent_type);

  /// \brief Get the mesh requirements for the specified agent.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(const std::string& name);

  /// \brief Get the mesh requirements for the specified agent.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(const agent_t agent_type);
  //@}

  //! \name Actions
  //@{

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  /// \return True if the class was registered successfully, false if registration failed.
  template <typename ClassToRegister>
  static inline bool register_new_class() {
    // Check that the ClassToRegister has the desired interface.
    using Checker = IsAgentType<ClassToRegister>;
    static_assert(Checker::has_get_name,
                  "HierarchyOfAgents: The class to register doesn't have the correct get_name function.\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_get_parent_names,
                  "HierarchyOfAgents: The class to register doesn't have the correct get_parent_names function.\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_get_topology,
                  "HierarchyOfAgents: The class to register doesn't have the correct get_topology function.\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_get_rank,
                  "HierarchyOfAgents: The class to register doesn't have the correct get_rank function.\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_has_topology,
                  "HierarchyOfAgents: The class to register doesn't have the correct has_topology function\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_has_rank,
                  "HierarchyOfAgents: The class to register doesn't have the correct has_rank function\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_add_part_reqs,
                  "HierarchyOfAgents: The class to register doesn't have the correct add_and_sync_part_reqs function.\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_add_subpart_reqs,
                  "HierarchyOfAgents: The class to register doesn't have the correct add_and_sync_subpart_reqs "
                  "function.\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");
    static_assert(Checker::has_get_mesh_requirements,
                  "HierarchyOfAgents: The class to register doesn't have the correct get_mesh_requirements function.\n"
                  "See the documentation of HierarchyOfAgents for more information about the expected interface.");

    // Ensure that the class name/parent name combo is unique.
    const std::string name = ClassToRegister::get_name();
    const bool already_registered = is_valid(name);
    if (already_registered) {
      std::cout << "HierarchyOfAgents: Skipping registration for class with name '" << name
                << "' because it is already registered." << std::endl;
      return false;
    }

    // Get the agent type for this class.
    number_of_registered_types_++;
    const agent_t agent_type = number_of_registered_types_ - 1;

    // Create the node in the hierarchy.
    StringTreeManager::create_node(agent_type, name, ClassToRegister::get_parent_names());

    // Register the class.
    get_name_to_type_map().insert(std::make_pair(name, agent_type));
    get_name_map().insert(std::make_pair(agent_type, name));
    get_parent_names_generator_map().insert(std::make_pair(agent_type, ClassToRegister::get_parent_names));
    get_topology_generator_map().insert(std::make_pair(agent_type, ClassToRegister::get_topology));
    get_rank_generator_map().insert(std::make_pair(agent_type, ClassToRegister::get_rank));
    get_has_topology_generator_map().insert(std::make_pair(agent_type, ClassToRegister::has_topology));
    get_has_rank_generator_map().insert(std::make_pair(agent_type, ClassToRegister::has_rank));
    get_add_part_reqs_generator_map().insert(std::make_pair(agent_type, ClassToRegister::add_and_sync_part_reqs));
    get_add_subpart_reqs_generator_map().insert(std::make_pair(agent_type, ClassToRegister::add_and_sync_subpart_reqs));
    get_mesh_requirements_generator_map().insert(std::make_pair(agent_type, ClassToRegister::get_mesh_requirements));

    std::cout << "HierarchyOfAgents: Class " << ClassToRegister::get_name() << " registered with id "
              << number_of_registered_types_ - 1 << std::endl;
    return true;
  }

  /// \brief Print the hierarchy.
  static void print_hierarchy(std::ostream& os = std::cout);

  // \brief Get the hierarchy as a string.
  static std::string get_hierarchy_as_a_string();
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The number of registered agent types.
  static inline agent_t number_of_registered_types_ = 0;
  //@}

  //! \name Internal hierarchy of agent names
  //@{

  /// \brief A helper class for generating a hierarchy of string names.
  class StringTreeNode {
   public:
    /// \brief Construct a new StringTreeNode object
    /// \param name [in] The name of this node.
    StringTreeNode(const unsigned& id, const std::string& name, const std::vector<std::string>& parent_names);

    /// \brief Add a child to the current node.
    /// Nodes shared so they can be assigned multiple parents.
    void add_child(std::shared_ptr<StringTreeNode> child);

    /// \brief Get the id of the node.
    unsigned get_id() const;

    /// \brief Get the name of the node.
    std::string get_name() const;

    /// \brief Get the names of the parents of the node.
    std::vector<std::string> get_parent_names() const;

    /// \brief Get the child of the node with the given name.
    std::shared_ptr<StringTreeNode> get_child(const std::string& name) const;

    /// \brief Get the children of the node.
    std::vector<std::shared_ptr<StringTreeNode>> get_children() const;

   private:
    const unsigned id_;
    const std::string name_;
    const std::vector<std::string> parent_names_;
    std::vector<std::shared_ptr<StringTreeNode>> children_;
  };  // StringTreeNode

  /// \brief A non-member builder class for \c StringTreeNode's that accounts for the static initialization order.
  class StringTreeManager {
   public:
    static std::shared_ptr<StringTreeNode> create_node(const unsigned& id, const std::string& name,
                                                       const std::vector<std::string>& parent_names);

    /// \brief Print the tree.
    static void print_tree(std::ostream& os = std::cout);

   private:
    /// \brief Print the tree.
    /// \param node The node in the tree to print.
    /// \param depth The depth of the node in the tree.
    static void print_tree(std::shared_ptr<StringTreeNode> node, std::ostream& os = std::cout, int depth = 0);

    static inline std::map<std::string, std::shared_ptr<StringTreeNode>> root_node_map_;
    static inline std::map<std::string, std::shared_ptr<StringTreeNode>> node_map_;
  };  // StringTreeManager

  //@}

  //! \name Typedefs
  //@{

  /// \brief A function that returns an std::string.
  using StringGenerator = std::function<std::string()>;

  /// \brief A function that returns a vector of strings.
  using VectorOfStringsGenerator = std::function<std::vector<std::string>()>;

  /// \brief A function that returns an stk::topology::topology_t.
  using TopologyGenerator = std::function<stk::topology::topology_t()>;

  /// \brief A function that returns an stk::topology::rank_t.
  using RankGenerator = std::function<stk::topology::rank_t()>;

  /// \brief A function type that takes a parameter list and produces a vector of shared pointers to PartReqs
  /// instances.
  using NewRequirementsGenerator = std::function<std::shared_ptr<mundy::meta::MeshReqs>()>;

  /// \brief A function that takes in a part requirements and returns a void.
  using AddRequirementsGenerator = std::function<void(std::shared_ptr<mundy::meta::PartReqs>)>;

  /// \brief A function that returns a bool.
  using BoolGenerator = std::function<bool()>;

  /// \brief A map from a string to agent_type.
  using NameToTypeMap = std::map<std::string, agent_t>;

  /// \brief A map from a string to agent_type.
  using TypeToNameMap = std::map<agent_t, std::string>;

  /// \brief A map from agent_type to a function that returns a vector of strings.
  using TypeToVectorOfStringsGeneratorMap = std::map<agent_t, VectorOfStringsGenerator>;

  /// \brief A map from agent_type to a function that returns a string view.
  using TypeToStringViewGeneratorMap = std::map<agent_t, StringGenerator>;

  /// \brief A map from agent_type to a function that returns an stk::topology::topology_t.
  using TypeToTopologyGeneratorMap = std::map<agent_t, TopologyGenerator>;

  /// \brief A map from agent_type to a function that returns an stk::topology::rank_t.
  using TypeToRankGeneratorMap = std::map<agent_t, RankGenerator>;

  /// \brief A map from agent_type to a function that takes in a part requirements and returns a void.
  using TypeToAddRequirementsGeneratorMap = std::map<agent_t, AddRequirementsGenerator>;

  /// \brief A map from agent_type to a function that returns a class's mesh requirements.
  using TypeToNewRequirementsGeneratorMap = std::map<agent_t, NewRequirementsGenerator>;

  /// \brief A map from agent_type to a function that returns a bool.
  using TypeToBoolGeneratorMap = std::map<agent_t, BoolGenerator>;
  //@}

  //! \name Attributes
  //@{
  static NameToTypeMap& get_name_to_type_map() {
    // Static: One and the same instance for all function calls.
    static NameToTypeMap name_to_id_map;
    return name_to_id_map;
  }

  static TypeToNameMap& get_name_map() {
    // Static: One and the same instance for all function calls.
    static TypeToNameMap name_map;
    return name_map;
  }

  static TypeToVectorOfStringsGeneratorMap& get_parent_names_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToVectorOfStringsGeneratorMap parent_names_generator_map;
    return parent_names_generator_map;
  }

  static TypeToTopologyGeneratorMap& get_topology_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToTopologyGeneratorMap topology_generator_map;
    return topology_generator_map;
  }

  static TypeToRankGeneratorMap& get_rank_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToRankGeneratorMap rank_generator_map;
    return rank_generator_map;
  }

  static TypeToBoolGeneratorMap& get_has_topology_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToBoolGeneratorMap has_topology_generator_map;
    return has_topology_generator_map;
  }

  static TypeToBoolGeneratorMap& get_has_rank_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToBoolGeneratorMap has_rank_generator_map;
    return has_rank_generator_map;
  }

  static TypeToAddRequirementsGeneratorMap& get_add_part_reqs_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToAddRequirementsGeneratorMap add_part_reqs_generator_map;
    return add_part_reqs_generator_map;
  }

  static TypeToAddRequirementsGeneratorMap& get_add_subpart_reqs_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToAddRequirementsGeneratorMap add_subpart_reqs_generator_map;
    return add_subpart_reqs_generator_map;
  }

  static TypeToNewRequirementsGeneratorMap& get_mesh_requirements_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToNewRequirementsGeneratorMap mesh_requirements_generator_map;
    return mesh_requirements_generator_map;
  }
  //@}
};  // HierarchyOfAgents

}  // namespace agents

}  // namespace mundy

#endif  // MUNDY_AGENTS_HIERARCHYOFAGENTS_HPP_
