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

#ifndef MUNDY_MESH_UTILS_SELECTORNODE_HPP_
#define MUNDY_MESH_UTILS_SELECTORNODE_HPP_

// C++ core
#include <exception>  // for std::exception
#include <map>        // for std::map
#include <ostream>    // for std::ostream (operator<<)
#include <vector>     // for std::vector

// Trilinos
#include <stk_mesh/base/Selector.hpp>

struct expression_evaluation_exception : public virtual std::exception {
  virtual const char* what() const throw() {
    return "Error evaluating expressions";
  }
};

struct expression_undefined_exception : public virtual std::exception {
  virtual const char* what() const throw() {
    static std::string rtnMsg;
    rtnMsg = "Found undefined function with name: " + m_msg + " and " + std::to_string(m_numArgs) + " argument(s)";
    return rtnMsg.c_str();
  }

  expression_undefined_exception(const char* msg, std::size_t numArgs) : m_msg(msg), m_numArgs(numArgs) {
  }

  std::string m_msg;
  std::size_t m_numArgs = 0;
};

namespace mundy {

namespace mesh {

namespace utils {

// Forward declarations
class SelectorNode;
class SelectorEval;
class SelectorParser;

/// \brief Enumerated type for the different types of operations that can be performed on selectors
///
/// Think of these operations as one step above lexical operations. For example, a lexicographical minus sign
/// could be a subtraction operation, but it could also be a unary negation operation.
enum Opcode {
  OPCODE_UNDEFINED,
  OPCODE_CONSTANT,
  OPCODE_STATEMENT,
  OPCODE_SUBTRACT,
  OPCODE_UNARY_NOT,
  OPCODE_ARITHMETIC_AND,
  OPCODE_ARITHMETIC_OR
};

/// \brief A helper function for printing Opcode enums
inline std::ostream& operator<<(std::ostream& stream, Opcode opcode) {
  static std::vector<std::string> opcodeNames{"OPCODE_UNDEFINED",    "OPCODE_CONSTANT",  "OPCODE_STATEMENT",
                                              "OPCODE_SUBTRACT",     "OPCODE_UNARY_NOT", "OPCODE_ARITHMETIC_AND",
                                              "OPCODE_ARITHMETIC_OR"};
  return stream << opcodeNames[opcode];
}

using EvalNodesType = std::vector<SelectorNode*>;
using NodeWeightMap = std::map<SelectorNode*, int>;

/// \brief A node in the selector evaluation tree
class SelectorNode {
 public:
  //! \name Constructors and operators
  //@{

  /// \brief Constructor
  ///
  /// \param opcode The operation to be performed by this node
  /// \param owner The SelectorEval object that owns this node
  /// \param data The data associated with this node (if any)
  SelectorNode(Opcode opcode, SelectorEval* owner, const stk::mesh::Selector& data = stk::mesh::Selector());

  /// \brief Destructor
  ~SelectorNode() = default;
  //@}

  //! \name Getters
  //@{
  int get_next_node_index();

  stk::mesh::Selector get_result() const;
  //@}

  //! \name Setters
  //@{

  stk::mesh::Selector& set_result();
  //@}

  //! \name Actions
  //@{
  void eval();

  void compute_node_weight(NodeWeightMap& node_weights);

  void eval_trace(const NodeWeightMap& node_weights, EvalNodesType& evaluation_nodes);
  //@}

  //  private: // STK didn't design this class with encapsulation in mind. Sadly, we need to make our data public.
  // TODO(palmerb4): If we make SelectorParser into a class and then declare it as a friend, we can make this private
  //! \name Private constructors and operators
  //@{

  /// \brief Copy constructor (hidden)
  explicit SelectorNode(const SelectorNode&);

  /// \brief Assignment operator (hidden)
  SelectorNode& operator=(const SelectorNode&);
  //@}

  //! \name Private data
  //@{

  const Opcode opcode_;
  stk::mesh::Selector data_;
  int result_idx_;
  int current_node_index_;
  int next_node_index_;
  SelectorNode* left_node_ptr_;
  SelectorNode* right_node_ptr_;
  SelectorEval* owner_ptr_;
  bool has_been_evaluated_;
  //@}
};

}  // namespace utils

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_UTILS_SELECTORNODE_HPP_
