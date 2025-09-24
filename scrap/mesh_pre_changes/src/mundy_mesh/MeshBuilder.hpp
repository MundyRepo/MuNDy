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

#ifndef MUNDY_MESH_MESHBUILDER_HPP_
#define MUNDY_MESH_MESHBUILDER_HPP_

/// \file MeshBuilder.cpp
/// \brief Declaration of the MeshBuilder class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/FieldDataManager.hpp>  // for stl::mesh::FieldDataManager
#include <stk_util/parallel/Parallel.hpp>      // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>  // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

/// \class MeshBuilder
/// \brief A helper class for building an STK BulkData entity.
///
/// This class is a duplicate of STK's \c MeshBuilder with our extended BulkData and MetaMesh in place of STK's.
class MeshBuilder {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor
  MeshBuilder();

  /// \brief Constructor with given given communicator.
  explicit MeshBuilder(stk::ParallelMachine comm);
  //@}

  //! @name Setters
  //@{

  /// \brief Set the spatial dimension of the mash.
  /// \param spatial_dimension [in] The dimension of the space within which the parts and entities reside.
  MeshBuilder &set_spatial_dimension(const unsigned spatial_dimension);

  /// \brief Set the names assigned to each rank.
  /// \param entity_rank_names [in] The names assigned to each rank.
  MeshBuilder &set_entity_rank_names(const std::vector<std::string> &entity_rank_names);

  /// \brief Set the MPI communicator to be used by STK.
  /// \param comm [in] The MPI communicator.
  MeshBuilder &set_communicator(const stk::ParallelMachine &comm);

  /// \brief Set the chosen Aura option. For example, stk::mesh::BulkData::AUTO_AURA.
  /// \param auto_aura_option [in] The chosen Aura option.
  MeshBuilder &set_auto_aura_option(const stk::mesh::BulkData::AutomaticAuraOption &auto_aura_option);

  /// \brief Set the add framework data flag.
  /// \param add_fmwk_data_flag [in] A Siera-specific flag, whose purpose is unbeknownst to me.
  MeshBuilder &set_add_fmwk_data_flag(bool add_fmwk_data_flag);

#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  /// \brief Set the field data manager.
  /// \param field_data_manager_ptr [in] Pointer to an existing field data manager.
  MeshBuilder &set_field_data_manager(std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager_ptr);
#else
  /// \brief Set the field data manager.
  /// \param field_data_manager_ptr [in] Pointer to an existing field data manager.
  MeshBuilder &set_field_data_manager(stk::mesh::FieldDataManager *const field_data_manager_ptr);
#endif

  /// \brief Set the upper bound on the number of mesh entities that may be associated with a single bucket.
  ///
  /// Although subject to change, the maximum bucket capacity is currently 1024 and the default capacity is 512.
  ///
  /// \param bucket_capacity [in] The bucket capacity.
  MeshBuilder &set_bucket_capacity(const unsigned bucket_capacity);

  /// \brief Set the initial upper bound on the number of mesh entities that may be associated with a single bucket.
  ///
  /// Although subject to change, the maximum bucket capacity is currently 1024 and the default capacity is 512.
  ///
  /// \param initial_bucket_capacity [in] The initial bucket capacity.
  MeshBuilder &set_initial_bucket_capacity(const unsigned initial_bucket_capacity);

  /// \brief Set the maximum upper bound on the number of mesh entities that may be associated with a single bucket.
  ///
  /// Although subject to change, the maximum bucket capacity is currently 1024 and the default capacity is 512.
  ///
  /// \param bucket_capacity [in] The maximum bucket capacity.
  MeshBuilder &set_maximum_bucket_capacity(const unsigned maximum_bucket_capacity);

  /// \brief Set the flag specifying if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  MeshBuilder &set_upward_connectivity_flag(const bool enable_upward_connectivity);
  //@}

  //! @name Actions
  //@{

  /// \brief Create a new aura ghosting instance.
  std::shared_ptr<stk::mesh::impl::AuraGhosting> create_aura_ghosting();

  /// \brief Create a new MetaData instance.
  std::shared_ptr<MetaData> create_meta_data();

  /// \brief Create a new BulkData instance.
  std::unique_ptr<BulkData> create_bulk_data();

  /// \brief Create a new BulkData instance using an existing MetaData instance.
  std::unique_ptr<BulkData> create_bulk_data(std::shared_ptr<MetaData> meta_data);
  //@}

 private:
  //! \name Mesh settings
  //@{

  /// \brief An instance of STK's MeshBuilder class
  stk::mesh::MeshBuilder builder_;

  /// \brief MPI communicator to be used by STK.
  /// This must be set before BulkData can be created.
  stk::ParallelMachine comm_;

  /// \brief Flag specifying if comm has been set or not.
  bool has_comm_;

  /// \brief Chosen Aura option. For example, stk::mesh::BulkData::AUTO_AURA.
  BulkData::AutomaticAuraOption auto_aura_option_;

  /// \brief A Siera-specific flag, whose purpose is unbeknownst to me.
  bool add_fmwk_data_flag_;

  /// \brief Pointer to an existing field data manager.
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
  std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager_ptr_;
#else
  stk::mesh::FieldDataManager *field_data_manager_ptr_;
#endif

  /// \brief Initial upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned initial_bucket_capacity_;

  /// \brief Maximum upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned maximum_bucket_capacity_;

  /// \brief Spatial dimension of the mash.
  unsigned spatial_dimension_;

  /// \brief Names assigned to each rank.
  std::vector<std::string> entity_rank_names_;

  /// \brief Flag specifying if upward connectivity will be enabled or not.
  bool upward_connectivity_flag_;
  //@}
};  // MeshBuilder

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_MESHBUILDER_HPP_
