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
#include <queue>  // for std::queue

// Mundy
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_mesh/utils/SelectorEval.hpp>    // for mundy::mesh::utils::SelectorEval
#include <mundy_mesh/utils/SelectorLexem.hpp>   // for mundy::mesh::utils::SelectorLexem
#include <mundy_mesh/utils/SelectorParser.hpp>  // for mundy::mesh::utils::SelectorParser

namespace mundy {

namespace mesh {

namespace utils {

SelectorEval::SelectorEval(const BulkData &bulk_data, const std::string &expression)
    : bulk_data_(bulk_data),
      expression_(expression),
      is_syntax_valid_(false),
      did_parse_succeed_(false),
      head_node_ptr_(nullptr) {
}

SelectorEval::SelectorEval(const SelectorEval &otherEval)
    : bulk_data_(otherEval.bulk_data_),
      expression_(otherEval.expression_),
      is_syntax_valid_(otherEval.is_syntax_valid_),
      did_parse_succeed_(otherEval.did_parse_succeed_),
      head_node_ptr_(otherEval.head_node_ptr_),
      node_ptrs_(otherEval.node_ptrs_),
      evaluation_nodes_(otherEval.evaluation_nodes_),
      result_buffer_(otherEval.result_buffer_) {
}

SelectorNode *SelectorEval::new_node(const int &opcode) {
  node_ptrs_.push_back(std::make_shared<SelectorNode>(static_cast<Opcode>(opcode), this));
  node_ptrs_.back().get()->current_node_index_ = static_cast<int>(node_ptrs_.size()) - 1;
  return node_ptrs_.back().get();
}

SelectorNode *SelectorEval::new_node(const int &opcode, const std::string &part_name) {
  // Check if the part is a special, reserved part; otherwise, get the part from the meta data
  stk::mesh::Selector data;
  if (part_name == "UNIVERSAL") {
    data = bulk_data_.mesh_meta_data().universal_part();
  } else if (part_name == "LOCALLY_OWNED") {
    data = bulk_data_.mesh_meta_data().locally_owned_part();
  } else if (part_name == "GLOBALLY_SHARED") {
    data = bulk_data_.mesh_meta_data().globally_shared_part();
  } else if (part_name == "AURA") {
    data = bulk_data_.mesh_meta_data().aura_part();
  } else {
    stk::mesh::Part *part_ptr = bulk_data_.mesh_meta_data().get_part(part_name);
    MUNDY_THROW_REQUIRE(part_ptr != nullptr, std::invalid_argument,
                        std::string("Could not find a part with name '") + part_name +
                            "' while parsing the expression\n" + expression_);
    data = *part_ptr;
  }

  node_ptrs_.push_back(std::make_shared<SelectorNode>(static_cast<Opcode>(opcode), this, data));
  node_ptrs_.back().get()->current_node_index_ = static_cast<int>(node_ptrs_.size()) - 1;
  return node_ptrs_.back().get();
}

void SelectorEval::syntax() {
  is_syntax_valid_ = false;
  did_parse_succeed_ = false;

  try {
    // Validate the characters
    SelectorLexemVector lex_vector = tokenize(expression_);

    // Call the multiparse routine to parse subexpressions
    head_node_ptr_ = parse_statements(*this, lex_vector.begin(), lex_vector.end());

    is_syntax_valid_ = true;
  } catch (std::runtime_error &) {
  }
}

void SelectorEval::parse() {
  try {
    syntax();

    if (is_syntax_valid_) {
      if (head_node_ptr_) {
        NodeWeightMap node_weights;
        head_node_ptr_->compute_node_weight(node_weights);
        head_node_ptr_->eval_trace(node_weights, evaluation_nodes_);
        assign_result_buffer_indices();
      }

      did_parse_succeed_ = true;
    } else {
      throw std::runtime_error("The following expression has a syntax error in it.\n" + expression_);
    }
  } catch (std::runtime_error &) {
    throw;
  }
}

stk::mesh::Selector SelectorEval::evaluate() const {
  if (!did_parse_succeed_) {
    throw std::runtime_error(std::string("Expression '") + expression_ +
                             "' did not parse successfully or has yet to be parsed.");
  }

  stk::mesh::Selector return_value;
  try {
    if (head_node_ptr_) {
      int nodeIndex = evaluation_nodes_.front()->current_node_index_;
      while (nodeIndex >= 0) {
        SelectorNode *node = node_ptrs_[nodeIndex].get();
        node->eval();
        nodeIndex = node->get_next_node_index();
      }
      return_value = evaluation_nodes_.back()->get_result();
    }

  } catch (expression_evaluation_exception &) {
    throw std::runtime_error(std::string("Expression '") + expression_ + "' did not evaluate successfully.");
  }
  return return_value;
}

int SelectorEval::get_head_node_index() const {
  return (head_node_ptr_) ? head_node_ptr_->current_node_index_ : -1;
}

int SelectorEval::get_first_node_index() const {
  return (!evaluation_nodes_.empty()) ? evaluation_nodes_.front()->current_node_index_ : -1;
}

int SelectorEval::get_last_node_index() const {
  return (!evaluation_nodes_.empty()) ? evaluation_nodes_.back()->current_node_index_ : -1;
}

SelectorEval &SelectorEval::set_expression(const std::string &expression) {
  expression_ = expression;
  did_parse_succeed_ = false;
  return *this;
}

class ResultBufferIndices {
 public:
  int get_free_index() {
    if (m_freeList.size() == 0) {
      release_index(m_resultBufferSize++);
    }

    auto idx = m_freeList.front();
    m_freeList.pop();
    return idx;
  }

  void release_index(const int idx) {
    STK_ThrowRequireMsg(idx >= 0, "Attempting to free negative index");
    m_freeList.push(idx);
  }

  int get_result_buffer_size() const {
    return m_resultBufferSize;
  }

 private:
  std::queue<int> m_freeList;
  int m_resultBufferSize = 0;
};

int SelectorEval::assign_result_buffer_indices() {
  ResultBufferIndices indexAssigner;

  for (auto node : evaluation_nodes_) {
    if (node->left_node_ptr_ && node->left_node_ptr_->result_idx_ >= 0) {
      indexAssigner.release_index(node->left_node_ptr_->result_idx_);
    }

    if (node->right_node_ptr_ && node->right_node_ptr_->result_idx_ >= 0) {
      indexAssigner.release_index(node->right_node_ptr_->result_idx_);
    }

    node->result_idx_ = indexAssigner.get_free_index();
  }

  result_buffer_.resize(indexAssigner.get_result_buffer_size());
  return indexAssigner.get_result_buffer_size();
}

}  // namespace utils

}  // namespace mesh

}  // namespace mundy