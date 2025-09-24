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

/// \file MeshBuilder.cpp
/// \brief Definition of the MeshBuilder class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Trilinos_version.h>     // for TRILINOS_MAJOR_MINOR_VERSION
#include <stk_util/stk_config.h>  // for MPI_COMM_NULL (MPI_COMM_NULL is defined by STK even if MPI is not enabled.)

#include <stk_mesh/base/BulkData.hpp>                              // for stk::mesh::BulkData
#include <stk_mesh/base/FieldDataManager.hpp>                      // for stk::mesh::FieldDataManager
#include <stk_mesh/base/MeshBuilder.hpp>                           // for stk::mesh::MeshBuilder
#include <stk_mesh/baseImpl/AuraGhostingDownwardConnectivity.hpp>  // for stk::mesh::impl::AuraGhostingDownwardConnectivity
#include <stk_mesh/baseImpl/BucketRepository.hpp>                  // stk::impl::BucketRepository
#include <stk_util/parallel/Parallel.hpp>                          // for stk::ParallelMachine

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>      // for BulkData
#include <mundy_mesh/MeshBuilder.hpp>   // for MeshBuilder
#include <mundy_mesh/MetaData.hpp>      // for MetaData

namespace mundy {

namespace mesh {

// \name Constructors and destructor
//{

MeshBuilder::MeshBuilder()
    : builder_(MPI_COMM_NULL),
      comm_(MPI_COMM_NULL),
      has_comm_(false),
      auto_aura_option_(stk::mesh::BulkData::AUTO_AURA),
      field_data_manager_ptr_(nullptr),
      initial_bucket_capacity_(stk::mesh::get_default_initial_bucket_capacity()),
      maximum_bucket_capacity_(stk::mesh::get_default_maximum_bucket_capacity()),
      spatial_dimension_(0),
      entity_rank_names_(),
      upward_connectivity_flag_(true) {
}

MeshBuilder::MeshBuilder(stk::ParallelMachine comm)
    : builder_(comm),
      comm_(comm),
      has_comm_(true),
      auto_aura_option_(stk::mesh::BulkData::AUTO_AURA),
      field_data_manager_ptr_(nullptr),
      initial_bucket_capacity_(stk::mesh::get_default_initial_bucket_capacity()),
      maximum_bucket_capacity_(stk::mesh::get_default_maximum_bucket_capacity()),
      spatial_dimension_(0),
      entity_rank_names_(),
      upward_connectivity_flag_(true) {
}
//}

// @name Setters
//{

MeshBuilder &MeshBuilder::set_spatial_dimension(const unsigned spatial_dimension) {
  spatial_dimension_ = spatial_dimension;
  return *this;
}

MeshBuilder &MeshBuilder::set_entity_rank_names(const std::vector<std::string> &entity_rank_names) {
  entity_rank_names_ = entity_rank_names;
  return *this;
}

MeshBuilder &MeshBuilder::set_communicator(const stk::ParallelMachine &comm) {
  comm_ = comm;
  has_comm_ = true;
  return *this;
}

MeshBuilder &MeshBuilder::set_auto_aura_option(const BulkData::AutomaticAuraOption &auto_aura_option) {
  auto_aura_option_ = auto_aura_option;
  return *this;
}

MeshBuilder &MeshBuilder::set_add_fmwk_data_flag(bool add_fmwk_data_flag) {
  add_fmwk_data_flag_ = add_fmwk_data_flag;
  return *this;
}

#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
MeshBuilder &MeshBuilder::set_field_data_manager(std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager_ptr) {
  field_data_manager_ptr_ = std::move(field_data_manager_ptr);
  return *this;
}
#else
MeshBuilder &MeshBuilder::set_field_data_manager(stk::mesh::FieldDataManager *const field_data_manager_ptr) {
  field_data_manager_ptr_ = field_data_manager_ptr;
  return *this;
}
#endif

MeshBuilder &MeshBuilder::set_bucket_capacity(const unsigned bucket_capacity) {
  initial_bucket_capacity_ = bucket_capacity;
  maximum_bucket_capacity_ = bucket_capacity;
  return *this;
}

MeshBuilder &MeshBuilder::set_initial_bucket_capacity(const unsigned initial_bucket_capacity) {
  initial_bucket_capacity_ = initial_bucket_capacity;
  return *this;
}

MeshBuilder &MeshBuilder::set_maximum_bucket_capacity(const unsigned maximum_bucket_capacity) {
  maximum_bucket_capacity_ = maximum_bucket_capacity;
  return *this;
}

MeshBuilder &MeshBuilder::set_upward_connectivity_flag(const bool enable_upward_connectivity) {
  upward_connectivity_flag_ = enable_upward_connectivity;
  return *this;
}
//}

// @name Actions
//{

std::shared_ptr<stk::mesh::impl::AuraGhosting> MeshBuilder::create_aura_ghosting() {
  if (upward_connectivity_flag_) {
    return std::make_shared<stk::mesh::impl::AuraGhosting>();
  }
  return std::make_shared<stk::mesh::impl::AuraGhostingDownwardConnectivity>();
}

std::shared_ptr<MetaData> MeshBuilder::create_meta_data() {
  if (spatial_dimension_ > 0 || !entity_rank_names_.empty()) {
    return std::make_shared<MetaData>(spatial_dimension_, entity_rank_names_);
  } else {
    return std::make_shared<MetaData>();
  }
}

std::unique_ptr<BulkData> MeshBuilder::create_bulk_data() {
  return this->create_bulk_data(this->create_meta_data());
}

std::unique_ptr<BulkData> MeshBuilder::create_bulk_data(std::shared_ptr<MetaData> meta_data_ptr) {
  MUNDY_THROW_REQUIRE(has_comm_, std::logic_error,
                      "MeshBuilder: Must be given an MPI communicator before creating BulkData.");

  return std::unique_ptr<BulkData>(new BulkData(meta_data_ptr, comm_, auto_aura_option_,
#ifdef SIERRA_MIGRATION
                                                add_fmwk_data_flag_,
#endif
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
                                                std::move(field_data_manager_ptr_),
#else
                                                field_data_manager_ptr_,
#endif
                                                initial_bucket_capacity_, maximum_bucket_capacity_,
                                                create_aura_ghosting(), upward_connectivity_flag_));
}
//}

}  // namespace mesh

}  // namespace mundy
