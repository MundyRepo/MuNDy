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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc


// STK mesh
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/NgpField.hpp>  // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>   // for stk::mesh::NgpMesh
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>     // for stk::mesh::FastMeshIndex

// Mundy libs
#include <mundy_mesh/Aggregate.hpp>    // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>  // for mundy::mesh::field_axpby, etc.
#include <mundy_mesh/NgpAccessorExpr.hpp>  // for accessor expressions

namespace mundy {

namespace mesh {

namespace {

/*
Shared setup: 
 - Create 3 fields (x, y, z, scratch1, scratch2) for each accessor type (scalar, vector3, matrix3, quaternion)
 - Create 3 additional fields (f1, f2, f3) of scalar type
 - Create 3 parts (p1, p2, p3) each with 1/3 of the entities
 - All parts should contain the (x, y, z, scratch1, scratch2) fields for all accessor types
 - Part 1 should contain f1, part 2 should contain f2, part 3 should contain f3 to create heterogeneity
 - Randomize the data in each field
*/










}  // namespace

}  // namespace mesh

}  // namespace mundy
