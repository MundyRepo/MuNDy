// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
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

#ifndef MUNDY_SHAPES_LINKERS_HPP_
#define MUNDY_SHAPES_LINKERS_HPP_

/// \file Linkers.hpp
/// \brief Declaration of the Linkers class

// C++ core libs
#include <memory>         // for std::shared_ptr, std::unique_ptr
#include <string>         // for std::string
#include <type_traits>    // for std::enable_if, std::is_base_of
#include <unordered_set>  // for std::unordered_set
#include <vector>         // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agents/Agents.hpp>          // for mundy::agents::Agents
#include <mundy_agents/RankedAssembly.hpp>  // for mundy::agents::RankedAssembly
#include <mundy_core/StringLiteral.hpp>     // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_meta/FieldReqs.hpp>         // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>          // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>          // for mundy::meta::PartReqs

namespace mundy {

namespace linkers {

/// \class Linkers
/// \brief The static interface for all of Mundy's Linkers.
///
/// In the current design, "linkers" are a constraint-rank Part with some set of requirements that endow the entities of
/// that part with the properties of a linker. By linker, we mean a constraint rank entity that connects a static number
/// of non-constraint rank entities and is used to communicate information between them or perform actions on them.
/// Think of linkers as mediators between objects; a linker knows why two objects are connected
/// and how to use their information to perform actions. In this regard, linkers encode structural information that
/// cannot otherwise be expressed by static topological connections. A linker could be, for example, a means of
/// connecting a smooth surface to a node that lives on the surface, or a means of connecting two entities that are
/// physical neighbors. Neither of these connections can be easily expressed as a static topological connection. While
/// one could encode this information in a different data structure, such as a map or a neighbor list, using linker
/// entities allows us to leverage STK's fields and parallel mesh infrastructure. This is particularly useful when we
/// want to load balance a mesh without throwing away the neighbor list or when we want to store invariants on the
/// linkers entities themselves. As well, it allows us to use STK's infrastructure for parallel communication and field
/// reduction.
///
/// Linkers cannot connect directly to elements due to STK not supporting element sharing. Instead, they must connect to
/// the nodes of the entities they are connecting and store a constraint-rank field on the linkers containing the
/// EntityIds of the "connected" entities. If an entity's nodes are shared, then STK's aura guarentees that the entity
/// is at least ghosted to the process that owns the linker and the linker will be locally owned or ghosted to the
/// process that owns the entity. Hence, when declaring a linker, we declare a relation between the linker and each node
/// of the entity and declare each of these nodes shared between all processes that own any of them. This is how we
/// ensure that the entities connected by a linker are either locally owned or ghosted to any process that owns or
/// auras the linker.
///
/// Fetching a linker connected to an entity requires one level of indirection: first use the linker field to access the
/// desired entity, then fetch any node connected to the entity and use it to fetch the linkers connected to that node.
/// This is a small price to pay for the benefits of using STK's infrastructure and is fully within the design of STK
/// (they call these non-topological connections).
///
/// For completeness, we will review the nomenclature and core concepts as they relate to linkers.
///
/// As a reminder: A connection from a higher-rank entity to a lower-rank entity is referred to as a downward relation.
/// When a downward relation is declared (e.g., between an element and a node), STK Mesh, by default, creates the
/// corresponding upward relation (e.g., from the node to the element).
///
///   - Aura: Linkers with a downward connection to a shared (but not locally owned) entity are ghosted to the owning
///   process.
///   - Sharing: Nodes, edges, and faces that have a downward connection from an Entities are shared by the process that
///   owns that entity. Elements and Constraints cannot be shared.
///   - Upward connections: A downward connection from a linker to an entity automatically results in the creation of
///   the corresponding upward connection from the entity to the linker.
///
/// One of the principle results of the sharing concept is that nodes, edges, and faces are the only entities that
/// support fast reductions. Reductions over higher rank entitiy's requies more expensive communication. This is why
/// it's often better to store reduction fields on nodes, edges, and faces rather than elements or constraints. We tend
/// to think of these higher-rank entities more as figure-heads that connect and organize lower rank entities and their
/// data than as entities that store data themselves.
///
/// The following is an example of a linker that connects a spring to the surface of a sphere partitioned over two
/// ranks.
///                     RANK0               |                    RANK1
///                        CONSTRAINT1(G)   |                         CONSTRAINT1(LO)
///                       |             |   |                        |              |
///     SPRING1(LO)      |   SPHERE1(G)  |  |       SPRING1(G)      |   SPHERE1(LO)  |
///    |          |     |        |      |   |       |        |     |        |       |
/// NODE1(LO)  NODE2(LO,S)       NODE3(S)   |   NODE1(G)   NODE2(LO)       NODE3(LO,S)
///
/// LO: Locally owned
/// G: Ghosted
/// S: Shared
///
/// \note Linkers require invariants to be maintained. For example, if a SurfaceLinker is only supposed to link an
/// sphere to a node that resides on its surface, then the first EntityId in the linker id field should be the sphere
/// and the second the surface node. This is how one can achieve aggragate entities such two spheres connected by a
/// spring that, for some reason, need to communicate information between one another.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agents::AgentFactory.
///
/// \note This class is a constraint rank assembly part containing all linkers. It is a subset of the AGENTS part.
class Linkers : public mundy::agents::RankedAssembly<mundy::core::make_string_literal("LINKERS"),
                                                     stk::topology::CONSTRAINT_RANK, mundy::agents::Agents> {
};  // Linkers

/// \brief Declare the relations and their converse between a constraint-rank entity and the family tree of any number
/// of entities within the same mesh.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a modification
/// cycle.
template <typename... Entities>
  requires(std::is_same_v<std::remove_cv_t<std::remove_reference_t<Entities>>, stk::mesh::Entity> && ...)
inline void connect_linker_to_entitys_nodes(stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& linker,
                                     const Entities&... to_entities) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "declare_relation: The mesh must be in a modification cycle.");
  MUNDY_THROW_ASSERT(bulk_data.entity_rank(linker) == stk::topology::CONSTRAINT_RANK, std::logic_error,
                     "declare_relation: The from entity must be constraint rank.");

  // For each entity to connect, we will declare a relation between the constraint rank entity and that entity's nodes.
  // The order of these connections should not matter and do not consider it as fixed.
  unsigned relation_counter = 0;
  auto declare_relations_to_nodes = [&](const stk::mesh::Entity& to_entity) {
    const unsigned num_nodes = bulk_data.num_connectivity(to_entity, stk::topology::NODE_RANK);
    MUNDY_THROW_ASSERT(num_nodes > 0, std::logic_error,
                       "declare_relation: The to entity must have at least one node to connect a linker to it.");
    const stk::mesh::Entity* nodes = bulk_data.begin(to_entity, stk::topology::NODE_RANK);
    for (unsigned i = 0; i < num_nodes; ++i) {
      bulk_data.declare_relation(linker, nodes[i], relation_counter);
      ++relation_counter;
    }
  };
  (declare_relations_to_nodes(to_entities), ...);
}

template <typename... Entities>
  requires(std::is_same_v<std::remove_cv_t<std::remove_reference_t<Entities>>, stk::mesh::Entity> && ...)
inline void connect_linker_to_entitys_nodes(stk::mesh::BulkData& bulk_data, stk::mesh::Permutation permut,
                                            stk::mesh::OrdinalVector& scratch1, stk::mesh::OrdinalVector& scratch2,
                                            stk::mesh::OrdinalVector& scratch3, const stk::mesh::Entity& linker,
                                            const Entities&... to_entities) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "declare_relation: The mesh must be in a modification cycle.");
  MUNDY_THROW_ASSERT(bulk_data.entity_rank(linker) == stk::topology::CONSTRAINT_RANK, std::logic_error,
                     "declare_relation: The from entity must be constraint rank.");

  // For each entity to connect, we will declare a relation between the constraint rank entity and that entity's nodes.
  // The order of these connections should not matter and do not consider it as fixed.
  unsigned relation_counter = 0;
  auto declare_relations_to_nodes = [&](const stk::mesh::Entity& to_entity) {
    const unsigned num_nodes = bulk_data.num_connectivity(to_entity, stk::topology::NODE_RANK);
    MUNDY_THROW_ASSERT(num_nodes > 0, std::runtime_error,
                       "declare_relation: The to entity must have at least one node to connect a linker to it.");
    const stk::mesh::Entity* nodes = bulk_data.begin(to_entity, stk::topology::NODE_RANK);
    for (unsigned i = 0; i < num_nodes; ++i) {
      bulk_data.declare_relation(linker, nodes[i], relation_counter, permut, scratch1, scratch2, scratch3);
      ++relation_counter;
    }
  };
  (declare_relations_to_nodes(to_entities), ...);
}

using LinkedEntitiesFieldType = stk::mesh::Field<std::underlying_type_t<stk::mesh::EntityKey::entity_key_t>>;

void fixup_linker_entity_ghosting(stk::mesh::BulkData& bulk_data, const LinkedEntitiesFieldType& linked_entities_field,
                                  stk::mesh::Field<int>& linked_entity_owners_field,
                                  const stk::mesh::Selector& linker_selector);

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_SHAPES_LINKERS_HPP_
