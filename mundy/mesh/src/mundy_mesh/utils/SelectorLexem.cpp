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
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

// Mundy
#include <mundy_mesh/utils/SelectorLexem.hpp>

namespace mundy {

namespace mesh {

namespace utils {

template <class T>
T convert_cast(const std::string &s) {
  std::istringstream is(s.c_str());
  T t = 0;

  is >> t;

  if (!is) {
    std::ostringstream msg;
    msg << "Unable to convert \"" << s << "\" to type " << typeid(T).name();
    throw std::runtime_error(msg.str().c_str());
  }

  return t;
}

template double convert_cast<double>(const std::string &);
template float convert_cast<float>(const std::string &);
template int convert_cast<int>(const std::string &);
template unsigned convert_cast<unsigned>(const std::string &);
template long convert_cast<long>(const std::string &);
template unsigned long convert_cast<unsigned long>(const std::string &);

SelectorLexemVector tokenize(const std::string &expression) {
  struct Graph {
    char ch;
    SelectorStringToken token;
  };

  static constexpr Graph graph[]{{'-', TOKEN_MINUS},         {'(', TOKEN_LPAREN},         {')', TOKEN_RPAREN},
                                 {'|', TOKEN_ARITHMETIC_OR}, {'&', TOKEN_ARITHMETIC_AND}, {'!', TOKEN_NOT}};

  SelectorLexemVector lex_vector;

  const char *it = expression.c_str();

  while (*it != '\0') {
    if (std::isspace(*it) || ::iscntrl(*it)) {
      ++it;
    } else if (std::isalpha(*it)) {
      // Parse identifier [a-zA-Z][a-zA-Z0-9_.]*
      const char *from = it;
      while (std::isalpha(*it) || std::isdigit(*it) || *it == '_' || *it == '.') {
        ++it;
      }
      lex_vector.push_back(SelectorLexem(TOKEN_IDENTIFIER, from, it));
    } else if (ispunct(*it)) {
      // Parse graphs
      const char *from = it;
      for (size_t i = 0; i < sizeof(graph) / sizeof(graph[0]); ++i) {
        if (*it == graph[i].ch) {
          ++it;
          lex_vector.push_back(SelectorLexem(graph[i].token, from, it));
          goto next_token;
        }
      }

      throw std::runtime_error(std::string("std::expreval::tokenize: Invalid graphic character '") + *it + "'");
    } else {
      throw std::runtime_error("Impossible expression parse error");
    }

  next_token:
    continue;
  }

  lex_vector.push_back(SelectorLexem(TOKEN_END, ""));

  return lex_vector;
}

}  // namespace utils

}  // namespace mesh

}  // namespace mundy
