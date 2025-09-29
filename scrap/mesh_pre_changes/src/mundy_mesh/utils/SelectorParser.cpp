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

// Mundy
#include <mundy_mesh/utils/SelectorEval.hpp>
#include <mundy_mesh/utils/SelectorNode.hpp>
#include <mundy_mesh/utils/SelectorParser.hpp>

namespace mundy {

namespace mesh {

namespace utils {

using LexemIterator = SelectorLexemVector::const_iterator;

SelectorNode *parse_statements(SelectorEval &eval, LexemIterator from, LexemIterator to);
SelectorNode *parse_statement(SelectorEval &eval, LexemIterator from, LexemIterator to);
SelectorNode *parse_expression(SelectorEval &eval, LexemIterator from, LexemIterator to);
SelectorNode *parse_term(SelectorEval &eval, LexemIterator from, LexemIterator term, LexemIterator to);
SelectorNode *parse_arythmatic(SelectorEval &eval, LexemIterator from, LexemIterator term, LexemIterator to);
SelectorNode *parse_unary(SelectorEval &eval, LexemIterator from, LexemIterator unary, LexemIterator to);
SelectorNode *parse_rvalue(SelectorEval &eval, LexemIterator from, LexemIterator to);

SelectorNode *parse_statements(SelectorEval &eval, LexemIterator from, LexemIterator to) {
  if ((*from).getToken() == TOKEN_END) {
    std::cout << "Nothing to parse" << std::endl;
    return nullptr;
  }

  LexemIterator it;
  for (it = from; (*it).getToken() != TOKEN_END; ++it) {
  }

  SelectorNode *statement = eval.new_node(OPCODE_STATEMENT);
  statement->left_node_ptr_ = parse_statement(eval, from, it);

  if ((*it).getToken() != TOKEN_END) {
    statement->right_node_ptr_ = parse_statements(eval, it + 1, to);
  }

  return statement;
}

SelectorNode *parse_statement(SelectorEval &eval, LexemIterator from, LexemIterator to) {
  return parse_expression(eval, from, to);
}

SelectorNode *parse_expression(SelectorEval &eval, LexemIterator from, LexemIterator to) {
  int paren_level = 0;                 // Paren level
  LexemIterator lparen_open_it = to;   // First open paren
  LexemIterator lparen_close_it = to;  // Corresponding close paren
  LexemIterator term_it = to;          // Last - at paren_level 0 for subtracting
  LexemIterator arythmetic_it = to;    // Last arithmetic at paren_level 0 for (and/or) operator

  LexemIterator unary_it = to;       // First +,- at plevel 0 for positive,negative
  LexemIterator last_unary_it = to;  // Last +,- found at plevel for for positive,negative

  // Scan the expression for the instances of the above tokens
  for (LexemIterator it = from; it != to; ++it) {
    switch ((*it).getToken()) {
      case TOKEN_LPAREN: {
        if (paren_level == 0 && lparen_open_it == to) {
          lparen_open_it = it;
        }
        paren_level++;
        break;
      }

      case TOKEN_RPAREN: {
        paren_level--;

        if (paren_level == 0 && lparen_close_it == to) {
          lparen_close_it = it;
        }

        if (paren_level < 0) {
          throw std::runtime_error("mismatched parenthesis");
        }
        break;
      }

      case TOKEN_ARITHMETIC_AND:

      case TOKEN_ARITHMETIC_OR: {
        if (paren_level == 0 && arythmetic_it == to) {
          arythmetic_it = it;
        }
        break;
      }

      case TOKEN_MINUS: {
        if (paren_level == 0) {
          // After any of these, we are a unary operator, not a term
          if (it == from || it == term_it + 1 || it == last_unary_it + 1) {  // Unary operator
            if (unary_it == to) {                                            // First unary operator?
              unary_it = it;
            }
            last_unary_it = it;
          } else {  // Term
            term_it = it;
          }
        }
        break;
      }

      case TOKEN_NOT: {
        if (paren_level == 0) {
          if (unary_it == to) {  /// First unary operator
            unary_it = it;
          }
          last_unary_it = it;
        }
        break;
      }

      default: {
        break;
      }
    }
  }

  if (paren_level != 0) {  // paren_level should now be zero
    throw std::runtime_error("mismatched parenthesis");
  }

  // This implements the operator hierarchy
  // Arythmatic
  if (arythmetic_it != to) {
    return parse_arythmatic(eval, from, arythmetic_it, to);
  }

  // Term
  if (term_it != to) {
    return parse_term(eval, from, term_it, to);
  }

  // Unary
  if (unary_it != to) {
    return parse_unary(eval, from, unary_it, to);
  }

  // Parenthetical
  if (lparen_open_it != to) {
    if (lparen_open_it == from) {
      if (lparen_close_it == to - 1 && lparen_close_it - lparen_open_it > 1) {
        return parse_expression(eval, lparen_open_it + 1, lparen_close_it);
      } else {
        throw std::runtime_error("syntax error parsing parentheses");
      }
    }

    throw std::runtime_error("syntax error 3");
  }

  // R-Value
  return parse_rvalue(eval, from, to);
}

SelectorNode *parse_arythmatic(SelectorEval &eval, LexemIterator from, LexemIterator arythmetic_it, LexemIterator to) {
  SelectorNode *arythmetic = eval.new_node(
      ((*arythmetic_it).getToken() == TOKEN_ARITHMETIC_AND ? OPCODE_ARITHMETIC_AND : OPCODE_ARITHMETIC_OR));

  arythmetic->left_node_ptr_ = parse_expression(eval, from, arythmetic_it);
  arythmetic->right_node_ptr_ = parse_expression(eval, arythmetic_it + 1, to);

  return arythmetic;
}

SelectorNode *parse_term(SelectorEval &eval, LexemIterator from, LexemIterator term_it, LexemIterator to) {
  SelectorNode *term = eval.new_node(OPCODE_SUBTRACT);

  term->left_node_ptr_ = parse_expression(eval, from, term_it);
  term->right_node_ptr_ = parse_expression(eval, term_it + 1, to);

  return term;
}

SelectorNode *parse_unary(SelectorEval &eval, [[maybe_unused]] LexemIterator from, LexemIterator unary_it,
                          LexemIterator to) {
  /* If it is a positive, just parse the internal of it */
  if ((*unary_it).getToken() == TOKEN_NOT) {
    SelectorNode *unary = eval.new_node(OPCODE_UNARY_NOT);
    unary->right_node_ptr_ = parse_expression(eval, unary_it + 1, to);
    return unary;
  } else {
    throw std::runtime_error("syntax error parsing unary operator");
  }
}

SelectorNode *parse_rvalue(SelectorEval &eval, LexemIterator from, LexemIterator to) {
  if (from + 1 != to) {
    throw std::runtime_error(std::string("r-value not allowed following ") + (*from).getString());
  }

  switch ((*from).getToken()) {
    case TOKEN_IDENTIFIER: {
      // Define a constant using some string data.
      SelectorNode *constant = eval.new_node(OPCODE_CONSTANT, (*from).getString());
      return constant;
    }

    default: {
      throw std::runtime_error("invalid rvalue");
    }
  }
}

}  // namespace utils

}  // namespace mesh

}  // namespace mundy
