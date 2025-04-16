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

/// \file HierarchyOfAgents.cpp
/// \brief Definition of the HierarchyOfAgents class

// External
#include <fmt/format.h>  // for fmt::format

// C++ core libs
#include <iostream>   // for std::cout
#include <map>        // for std::map
#include <memory>     // for std::shared_ptr
#include <sstream>    // for std::stringstream
#include <stdexcept>  // for std::invalid_argument
#include <string>     // for std::string
#include <utility>    // for std::make_pair
#include <vector>     // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agents/HierarchyOfAgents.hpp>  // for mundy::agents::HierarchyOfAgents
#include <mundy_core/throw_assert.hpp>         // for MUNDY_THROW_ASSERT

namespace mundy {

namespace agents {

//! \name Getters
//@{

unsigned HierarchyOfAgents::get_number_of_registered_types() {
  return number_of_registered_types_;
}

bool HierarchyOfAgents::is_valid(const std::string& name) {
  return get_name_to_type_map().count(name) != 0;
}

bool HierarchyOfAgents::is_valid(const agent_t agent_type) {
  return get_name_map().count(agent_type) != 0;
}

void HierarchyOfAgents::assert_is_valid(const std::string& name) {
  MUNDY_THROW_REQUIRE(is_valid(name), std::invalid_argument,
                      fmt::format("HierarchyOfAgents: The provided class's name '{}' is not valid.\n"
                                  "There are currently {} registered classes.\n"
                                  "The hierarchy is:\n{}",
                                  name, get_number_of_registered_types(), get_hierarchy_as_a_string()));
}

void HierarchyOfAgents::assert_is_valid(const agent_t agent_type) {
  MUNDY_THROW_REQUIRE(is_valid(agent_type), std::invalid_argument,
                      fmt::format("HierarchyOfAgents: The provided class's id '{}' is not valid.\n"
                                  "There are currently {} registered classes.\n"
                                  "The hierarchy is:\n{}",
                                  agent_type, get_number_of_registered_types(), get_hierarchy_as_a_string()));
}

agent_t HierarchyOfAgents::get_agent_type(const std::string& name) {
  assert_is_valid(name);
  return get_name_to_type_map()[name];
}

std::string HierarchyOfAgents::get_name(const agent_t agent_type) {
  assert_is_valid(agent_type);
  return get_name_map()[agent_type];
}

std::vector<std::string> HierarchyOfAgents::get_parent_names(const std::string& name) {
  assert_is_valid(name);
  const agent_t agent_type = get_agent_type(name);
  return get_parent_names(agent_type);
}

std::vector<std::string> HierarchyOfAgents::get_parent_names(const agent_t agent_type) {
  assert_is_valid(agent_type);
  return get_parent_names_generator_map()[agent_type]();
}

stk::topology::topology_t HierarchyOfAgents::get_topology(const std::string& name) {
  assert_is_valid(name);
  const agent_t agent_type = get_agent_type(name);
  return get_topology(agent_type);
}

stk::topology::topology_t HierarchyOfAgents::get_topology(const agent_t agent_type) {
  assert_is_valid(agent_type);
  return get_topology_generator_map()[agent_type]();
}

stk::topology::rank_t HierarchyOfAgents::get_rank(const std::string& name) {
  assert_is_valid(name);
  const agent_t agent_type = get_agent_type(name);
  return get_rank(agent_type);
}

stk::topology::rank_t HierarchyOfAgents::get_rank(const agent_t agent_type) {
  assert_is_valid(agent_type);
  return get_rank_generator_map()[agent_type]();
}

void HierarchyOfAgents::add_and_sync_part_reqs(std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr,
                                               const std::string& name) {
  assert_is_valid(name);
  const agent_t agent_type = get_agent_type(name);
  add_and_sync_part_reqs(part_reqs_ptr, agent_type);
}

void HierarchyOfAgents::add_and_sync_part_reqs(std::shared_ptr<mundy::meta::PartReqs> part_reqs_ptr,
                                               const agent_t agent_type) {
  assert_is_valid(agent_type);
  get_add_part_reqs_generator_map()[agent_type](part_reqs_ptr);
}

void HierarchyOfAgents::add_and_sync_subpart_reqs(std::shared_ptr<mundy::meta::PartReqs> subpart_reqs_ptr,
                                                  const std::string& name) {
  assert_is_valid(name);
  const agent_t agent_type = get_agent_type(name);
  add_and_sync_subpart_reqs(subpart_reqs_ptr, agent_type);
}

void HierarchyOfAgents::add_and_sync_subpart_reqs(std::shared_ptr<mundy::meta::PartReqs> subpart_reqs_ptr,
                                                  const agent_t agent_type) {
  assert_is_valid(agent_type);
  get_add_subpart_reqs_generator_map()[agent_type](subpart_reqs_ptr);
}

std::shared_ptr<mundy::meta::MeshReqs> HierarchyOfAgents::get_mesh_requirements(const std::string& name) {
  assert_is_valid(name);
  const agent_t agent_type = get_agent_type(name);
  return get_mesh_requirements(agent_type);
}

std::shared_ptr<mundy::meta::MeshReqs> HierarchyOfAgents::get_mesh_requirements(const agent_t agent_type) {
  assert_is_valid(agent_type);
  return get_mesh_requirements_generator_map()[agent_type]();
}
//@}

//! \name Actions
//@{

void HierarchyOfAgents::print_hierarchy(std::ostream& os) {
  StringTreeManager::print_tree(os);
}

std::string HierarchyOfAgents::get_hierarchy_as_a_string() {
  std::stringstream ss;
  print_hierarchy(ss);
  return ss.str();
}
//@}

//! \name Internal hierarchy of agent names
//@{

HierarchyOfAgents::StringTreeNode::StringTreeNode(const unsigned& id, const std::string& name,
                                                  const std::vector<std::string>& parent_names)
    : id_(id), name_(name), parent_names_(parent_names) {
}

void HierarchyOfAgents::StringTreeNode::add_child(std::shared_ptr<StringTreeNode> child) {
  // If the child already exists, don't add it.
  for (const auto& existing_child : children_) {
    if (existing_child->get_name() == child->get_name()) {
      return;
    }
  }

  children_.push_back(child);
}

unsigned HierarchyOfAgents::StringTreeNode::get_id() const {
  return id_;
}

std::string HierarchyOfAgents::StringTreeNode::get_name() const {
  return name_;
}

std::vector<std::string> HierarchyOfAgents::StringTreeNode::get_parent_names() const {
  return parent_names_;
}

std::shared_ptr<HierarchyOfAgents::StringTreeNode> HierarchyOfAgents::StringTreeNode::get_child(
    const std::string& name) const {
  for (const auto& child : children_) {
    if (child->get_name() == name) {
      return child;
    }
  }
  return nullptr;
}

std::vector<std::shared_ptr<HierarchyOfAgents::StringTreeNode>> HierarchyOfAgents::StringTreeNode::get_children()
    const {
  return children_;
}

std::shared_ptr<HierarchyOfAgents::StringTreeNode> HierarchyOfAgents::StringTreeManager::create_node(
    const unsigned& id, const std::string& name, const std::vector<std::string>& parent_names) {
  // Check if the node already exists.
  const auto named_node_iter = node_map_.find(name);
  if (named_node_iter != node_map_.end()) {
    // The node exists. Return it.
    return named_node_iter->second;
  }

  const auto named_root_node_iter = root_node_map_.find(name);
  if (named_root_node_iter != root_node_map_.end()) {
    // The root node exists. Return it.
    return named_root_node_iter->second;
  }

  // Create the node.
  auto node = std::make_shared<HierarchyOfAgents::StringTreeNode>(id, name, parent_names);
  node_map_.insert({name, node});

  // For each parent, if parent name exists in the map, attach the current node to it.
  // This handles the case where the child is declared after the parent.
  for (const auto& parent_name : parent_names) {
    const auto parent_iter = node_map_.find(parent_name);
    if (parent_iter != node_map_.end()) {
      parent_iter->second->add_child(node);
    }
  }

  // If the node has no parents, it is a root node.
  if (parent_names.size() == 0) {
    root_node_map_.insert({name, node});
  }

  // For each node, if the current node is one of its parents, attach the current node to it.
  // This handles the case where the parent is declared after the child.
  for (auto node_iter = node_map_.begin(); node_iter != node_map_.end();) {
    std::shared_ptr<StringTreeNode> potential_child_node = node_iter->second;
    for (const auto& parent_name : potential_child_node->get_parent_names()) {
      if (parent_name == name) {
        // This node is a child of the new node. Congrats!
        node->add_child(potential_child_node);
      } else {
        // This node is not a child of the new node.
        ++node_iter;
      }
    }
  }

  return node;
}

void HierarchyOfAgents::StringTreeManager::print_tree(std::ostream& os) {
  for (const auto& pair : root_node_map_) {
    print_tree(pair.second, os);
  }
}

void HierarchyOfAgents::StringTreeManager::print_tree(std::shared_ptr<StringTreeNode> node, std::ostream& os,
                                                      int depth) {
  for (int i = 0; i < depth; ++i) {
    os << "  ";
  }
  os << "Name: " << node->get_name() << " | " << "ID: " << node->get_id() << std::endl;

  for (const auto& child : node->get_children()) {
    print_tree(child, os, depth + 1);
  }
}
//@}

}  // namespace agents

}  // namespace mundy
