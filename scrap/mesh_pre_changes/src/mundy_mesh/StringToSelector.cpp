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

/// \file StringToSelector.cpp
/// \brief Definition of the string to selector helper function

// C++ core libs
#include <string>  // for std::string

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// Mundy libs
#include <mundy_mesh/utils/SelectorEval.hpp>  // for mundy::mesh::utils::SelectorEval

namespace mundy {

namespace mesh {

stk::mesh::Selector string_to_selector(const BulkData &bulk_data, const std::string &selector_string) {
  auto eval = utils::SelectorEval(bulk_data, selector_string);
  eval.parse();
  return eval.evaluate();
}

}  // namespace mesh

}  // namespace mundy
