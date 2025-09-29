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

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Mundy libs
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace {

// Note, STK's BulkData is thoughly tested, so we don't need to test it here. We just need to test our extensions to it.
// Our only extension is the ability to return our MetaData wrapper, so that's all we need to test.

TEST(BulkDataWrapperTest, FetchMetaData) {
  MeshBuilder builder(MPI_COMM_WORLD);
  ASSERT_NO_THROW(std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data());
  std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data();
  ASSERT_NO_THROW([[maybe_unused]] MetaData &meta_data = bulk_data_ptr->mesh_meta_data());
  ASSERT_NO_THROW(std::shared_ptr<MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr());
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
