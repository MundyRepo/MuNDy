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

/// \file ResolveConstraints.cpp
/// \brief Definition of the ResolveConstraints class

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>            // for mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_meta/MetaTechniqueDispatcher.hpp>  // for mundy::meta::MetaMethodSubsetExecutionDispatcher
#include <mundy_motion/ResolveConstraints.hpp>     // for mundy::motion::ResolveConstraints

namespace mundy {

namespace motion {

// \name Constructors and destructor
//{

ResolveConstraints::ResolveConstraints(mundy::mesh::BulkData *const bulk_data_ptr,
                                       const Teuchos::ParameterList &fixed_params)
    : mundy::meta::MetaMethodSubsetExecutionDispatcher<ResolveConstraints,
                                                       mundy::meta::make_registration_string("NON_SMOOTH_LCP")>(
          bulk_data_ptr, fixed_params) {
}
//}

}  // namespace motion

}  // namespace mundy
