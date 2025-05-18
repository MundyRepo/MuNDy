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

#ifndef MUNDY_MESH_UTILS_SELECTOREVAL_HPP_
#define MUNDY_MESH_UTILS_SELECTOREVAL_HPP_

/// \file SelectorEval.hpp
/// \brief A class for evaluating selector expressions

// C++ core
#include <memory>
#include <set>
#include <string>
#include <vector>

// Trilinos
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// Mundy
#include <mundy_mesh/BulkData.hpp>            // for mundy::mesh::BulkData
#include <mundy_mesh/utils/SelectorNode.hpp>  // for mundy::mesh::utils::SelectorNode

namespace mundy {

namespace mesh {

namespace utils {

// Forward declarations
class SelectorNode;

/// \brief A class for evaluating selector expressions
///
/// Here, a selector expression is the string representation of combining parts to form a selector. For example,
/// Given the parts A, B, and C, one would typically create a selector using something like
///  auto our_selector = stk::mesh::Selector(A) & B & !C;
/// We want to allow users to perform such operations using a string. For the above example, the string would be
///  "A & B & !C"
///
/// The full set of mathematical operations that can be performed on selectors are:
///  - Subtraction:    -
///  - Arythmetic and: &
///  - Arythmetic or:  |
///  - Unary not:      !
///  - Parentheses:   ( )
///
/// Our selector expressions satisfy the following properties:
///   1. Spaces are allowed in the selector string, but are not required.
///   2. Names may contain any combination of letters, numbers, underscores, and periods, so long as they start with a
///   letter.
///   3. The names of the parts are fetched from the BulkData object. If a part name is not found, an exception is
///   thrown.
///   4. We offer the following special/reserved part names:
///     - "UNIVERSAL"        -> The universal part, which contains all entities.
///     - "LOCALLY_OWNED"    -> The locally owned part, which contains all entities owned by the current process.
///     - "GLOBALLY_SHARED"  -> The globally shared part, which contains all entities shared from another processes.
///     - "AURA"             -> The automatically generated auto part, which contains all entities ghosted from another
///     process.
///
/// This class is used to evaluate a selector expression. The expression is checked for syntax errors, parsed into a
/// weighted expression tree, and then the various nodes of that graph are evaluated to produce a final selector.
class SelectorEval {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  ///
  /// \param bulk_data [in] The bulk data object used to fetch the part names.
  /// \param expr [in] The selector expression to evaluate.
  SelectorEval(const BulkData &bulk_data, const std::string &expr = "");

  /// \brief Copy constructor
  SelectorEval(const SelectorEval &);

  /// \brief Default destructor
  ~SelectorEval() = default;
  //@}

  //! \name Getters
  //@{
  const std::string &get_expression() const {
    return expression_;
  }

  int get_node_count() const {
    return static_cast<int>(node_ptrs_.size());
  }

  int get_result_buffer_size() const {
    return static_cast<int>(result_buffer_.size());
  }

  int get_head_node_index() const;

  int get_first_node_index() const;

  int get_last_node_index() const;

  SelectorNode *get_node(int i) const {
    return node_ptrs_[i].get();
  }

  bool get_syntax_status() const {
    return is_syntax_valid_;
  }

  bool get_parse_status() const {
    return did_parse_succeed_;
  }

  stk::mesh::Selector &get_result_buffer_value(const int idx) {
    return result_buffer_[idx];
  }
  //@}

  //! \name Setters
  //@{

  SelectorEval &set_expression(const std::string &expression);
  //@}

  //! \name Actions
  //@{

  /// \brief Assign the result buffer indices to the nodes
  ///
  /// This function assigns the result buffer indices to the nodes in the evaluation tree. This is necessary
  /// because the nodes are evaluated in a specific order, and the result of each node is stored in the result buffer so
  /// that it can be used by other nodes.
  int assign_result_buffer_indices();

  /// \brief Create a new node that performs some operation
  ///
  /// \param op [in] The operation it performs
  SelectorNode *new_node(const int &op);

  /// \brief Create a new node that fetches a part given a name
  ///
  /// \param op [in] The operation it performs
  /// \param part_name [in] The name of the part to fetch
  SelectorNode *new_node(const int &op, const std::string &part_name);

  /// \brief Check the syntax of the expression
  void syntax();

  /// \brief Parse the expression
  void parse();

  /// \brief Evaluate the expression
  stk::mesh::Selector evaluate() const;
  //@}

 private:
  //! \name Private constructors and operators
  //@{

  /// \brief Assignment operator (hidden)
  SelectorEval &operator=(const SelectorEval &);
  //@}

  //! \name Private data
  //@{

  /// \brief The bulk data object used to fetch the part names
  const BulkData &bulk_data_;

  /// \brief The selector expression to evaluate
  std::string expression_;

  /// \brief A flag indicating if the syntax of the expression is valid
  bool is_syntax_valid_;

  /// \brief A flag indicating if the parsing of the expression was successful
  bool did_parse_succeed_;

  /// \brief The head node of the evaluation tree
  SelectorNode *head_node_ptr_;

  /// \brief The nodes in the evaluation tree
  std::vector<std::shared_ptr<SelectorNode>> node_ptrs_;

  /// \brief The nodes in the evaluation tree that are not constant expressions
  EvalNodesType evaluation_nodes_;

  /// \brief The result buffer that stores the result of each node
  std::vector<stk::mesh::Selector> result_buffer_;
  //@}

  //! \name Friends <3
  //@{

  friend void check_node_order(const std::string &expression);
  friend void check_evaluation_node_order(const std::string &expression);
  //@}
};

}  // namespace utils

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_UTILS_SELECTOREVAL_HPP_
