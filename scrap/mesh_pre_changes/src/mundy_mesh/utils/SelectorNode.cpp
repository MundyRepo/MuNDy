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
#include <string>

// Mundy
#include <mundy_mesh/utils/SelectorEval.hpp>
#include <mundy_mesh/utils/SelectorNode.hpp>

namespace mundy {

namespace mesh {

namespace utils {

SelectorNode::SelectorNode(Opcode opcode, SelectorEval *owner, const stk::mesh::Selector &data)
    : opcode_(opcode),
      data_(data),
      result_idx_(-1),
      current_node_index_(-1),
      next_node_index_(-1),
      left_node_ptr_(nullptr),
      right_node_ptr_(nullptr),
      owner_ptr_(owner),
      has_been_evaluated_(false) {
}

int SelectorNode::get_next_node_index() {
  return next_node_index_;
}

stk::mesh::Selector SelectorNode::get_result() const {
  STK_ThrowAssertMsg(has_been_evaluated_, "Requesting node result before it has been computed.");
  return owner_ptr_->get_result_buffer_value(result_idx_);
}

stk::mesh::Selector &SelectorNode::set_result() {
  return owner_ptr_->get_result_buffer_value(result_idx_);
}

void SelectorNode::eval() {
  switch (opcode_) {
    case OPCODE_STATEMENT: {
      set_result() = left_node_ptr_->get_result();
      break;
    }
    case OPCODE_CONSTANT: {
      set_result() = data_;
      break;
    }
    case OPCODE_SUBTRACT: {
      set_result() = left_node_ptr_->get_result() - right_node_ptr_->get_result();
      break;
    }
    case OPCODE_ARITHMETIC_AND: {
      set_result() = left_node_ptr_->get_result() & right_node_ptr_->get_result();
      break;
    }
    case OPCODE_ARITHMETIC_OR: {
      set_result() = left_node_ptr_->get_result() | right_node_ptr_->get_result();
      break;
    }
    case OPCODE_UNARY_NOT: {
      set_result() = !right_node_ptr_->get_result();
      break;
    }
    default: {
      STK_ThrowErrorMsg("Unknown OpCode (" + std::to_string(opcode_) + ")");
    }
  }

  has_been_evaluated_ = true;
}

void SelectorNode::compute_node_weight(NodeWeightMap &node_weights) {
  switch (opcode_) {
    case OPCODE_STATEMENT: {
      for (SelectorNode *statement = this; statement; statement = statement->right_node_ptr_) {
        statement->left_node_ptr_->compute_node_weight(node_weights);
        node_weights[statement] = node_weights[left_node_ptr_];
      }
      break;
    }

    case OPCODE_CONSTANT: {
      node_weights[this] = 1;
      break;
    }

    case OPCODE_SUBTRACT:
    case OPCODE_ARITHMETIC_AND:
    case OPCODE_ARITHMETIC_OR: {
      left_node_ptr_->compute_node_weight(node_weights);
      right_node_ptr_->compute_node_weight(node_weights);
      node_weights[this] = node_weights.at(left_node_ptr_) + node_weights.at(right_node_ptr_);
      break;
    }

    case OPCODE_UNARY_NOT: {
      right_node_ptr_->compute_node_weight(node_weights);
      node_weights[this] = node_weights[right_node_ptr_];
      break;
    }

    default: {  // Unknown opcode
      throw expression_evaluation_exception();
    }
  }
}

void SelectorNode::eval_trace(const NodeWeightMap &node_weights, EvalNodesType &evaluation_nodes) {
  switch (opcode_) {
    case OPCODE_STATEMENT: {
      for (SelectorNode *statement = this; statement; statement = statement->right_node_ptr_) {
        statement->left_node_ptr_->eval_trace(node_weights, evaluation_nodes);
        evaluation_nodes.back()->next_node_index_ = statement->current_node_index_;
        evaluation_nodes.push_back(statement);
      }
      break;
    }

    case OPCODE_CONSTANT: {
      break;
    }

    case OPCODE_SUBTRACT:
    case OPCODE_ARITHMETIC_AND:
    case OPCODE_ARITHMETIC_OR: {
      if (node_weights.at(left_node_ptr_) >= node_weights.at(right_node_ptr_)) {
        left_node_ptr_->eval_trace(node_weights, evaluation_nodes);
        right_node_ptr_->eval_trace(node_weights, evaluation_nodes);
      } else {
        right_node_ptr_->eval_trace(node_weights, evaluation_nodes);
        left_node_ptr_->eval_trace(node_weights, evaluation_nodes);
      }
      break;
    }

    case OPCODE_UNARY_NOT: {
      right_node_ptr_->eval_trace(node_weights, evaluation_nodes);
      break;
    }

    default: {  // Unknown opcode
      throw expression_evaluation_exception();
    }
  }

  if (opcode_ != OPCODE_STATEMENT) {
    if (!evaluation_nodes.empty()) {
      evaluation_nodes.back()->next_node_index_ = current_node_index_;
    }
    evaluation_nodes.push_back(this);
  }
}

}  // namespace utils

}  // namespace mesh

}  // namespace mundy