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

#ifndef MUNDY_MESH_IMPL_SELECTORLEXEM_HPP_
#define MUNDY_MESH_IMPL_SELECTORLEXEM_HPP_

// C++ core
#include <string>  // for std::string
#include <vector>  // for std::vector

namespace mundy {

namespace mesh {

namespace impl {

template <class T>
T convert_cast(const std::string &s);

/// \brief Valid token types within our Selector lexicon.
enum SelectorStringToken {
  TOKEN_MINUS,
  TOKEN_LPAREN,
  TOKEN_RPAREN,
  TOKEN_ARITHMETIC_OR,
  TOKEN_ARITHMETIC_AND,
  TOKEN_NOT,
  TOKEN_IDENTIFIER,
  TOKEN_END
};

/// \brief A selector lexem is the smallest unit of meaning in a selector string expression.
///
/// From Wikipedia, "[a] lexeme is a unit of lexical meaning that underlies a set of words that
/// are related through inflection. It is a basic abstract unit of meaning, a unit of morphological analysis in
/// linguistics that roughly corresponds to a set of forms taken by a single root word."
///
/// In our case, we are using the term to refer to the smallest unit of meaning in our selector string. This includes
/// string names (consisting of letters, numbers, and underscores) and various operators. The lexem doesn't "know"
/// about concepts like order of operations or the meaning of the operators, it just knows how to parse the string
/// into the desired tokens.
class SelectorLexem {
 public:
  //! \name Constructors
  //@{
  SelectorLexem(SelectorStringToken token, const char *from, const char *to) : m_token(token), m_value(from, to) {
  }

  SelectorLexem(SelectorStringToken token, const char *value) : m_token(token), m_value(value) {
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the token type of the lexem.
  SelectorStringToken getToken() const {
    return m_token;
  }

  /// \brief Get the string value of the lexem.
  const std::string &getString() const {
    return m_value;
  }

  /// \brief Get the value of the lexem as a specific type.
  ///
  /// @tparam T The type to convert the value to.
  template <class T>
  T getValue() const {
    return convert_cast<T>(m_value);
  }
  //@}

 private:
  //! \name Private data
  //@{

  /// \brief The token type of the lexem.
  SelectorStringToken m_token;

  /// \brief The string value of the lexem.
  std::string m_value;
  //@}
};  // SelectorLexem

/// \brief A vector of selector lexems.
typedef std::vector<SelectorLexem> SelectorLexemVector;

/// \brief Tokenize a selector string expression into a vector of lexems.
///
/// \param expression The expression to tokenize.
SelectorLexemVector tokenize(const std::string &expression);

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif /* MUNDY_MESH_IMPL_SELECTORLEXEM_HPP_ */
