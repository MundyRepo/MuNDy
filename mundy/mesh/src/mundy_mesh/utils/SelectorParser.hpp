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

#ifndef MUNDY_MESH_UTILS_SELECTORPARSER_HPP_
#define MUNDY_MESH_UTILS_SELECTORPARSER_HPP_

// Mundy
#include <mundy_mesh/utils/SelectorLexem.hpp>  // for mundy::mesh::utils::SelectorLexem

namespace mundy {

namespace mesh {

namespace utils {

class SelectorNode;
class SelectorEval;

/// \brief Parse a set of statements
///
/// \param eval The SelectorEval object that we will populate with the parsed statements
/// \param from The beginning of the lexem vector which we should parse
/// \param to The end of the lexem vector which we should parse
SelectorNode *parse_statements(SelectorEval &eval, SelectorLexemVector::const_iterator from,
                               SelectorLexemVector::const_iterator to);

}  // namespace utils

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_UTILS_SELECTORPARSER_HPP_
