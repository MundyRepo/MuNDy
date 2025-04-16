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

#ifndef MUNDY_MESH_BULKDATA_HPP_
#define MUNDY_MESH_BULKDATA_HPP_

/// \file BulkData.hpp
/// \brief Declaration of the BulkData class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/Bucket.hpp>            // stk::mesh::get_default_maximum_bucket_capacity
#include <stk_mesh/base/BulkData.hpp>          // for stk::mesh::BulkData
#include <stk_mesh/base/FieldDataManager.hpp>  // for stl::mesh::FieldDataManager
#include <stk_mesh/base/MeshBuilder.hpp>       // for stk::mesh::MeshBuilder
#include <stk_util/parallel/Parallel.hpp>      // for stk::ParallelMachine

// Mundy libs
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

//! \name Forward declarations
//@{
// clang-format off

namespace mundy { namespace mesh { class MeshBuilder; } }
// clang-format on
//@}

/// \class BulkData
/// \brief A extension of STK's BulkData, with streamlined access to Mundy's stk wrappers.
///
/// For now, this extension simply stores and returns our MetaData wrapper
class BulkData : public stk::mesh::BulkData {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Destructor.
  virtual ~BulkData() {
    meta_data_ptr_ = nullptr;
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Fetch the meta data manager for this bulk data manager.
  const MetaData &mesh_meta_data() const {
    return *meta_data_ptr_;
  }

  /// \brief Fetch the meta data manager for this bulk data manager.
  MetaData &mesh_meta_data() {
    return *meta_data_ptr_;
  }

  /// \brief Fetch the pointer to the meta data manager for this bulk data manager.
  const std::shared_ptr<MetaData> mesh_meta_data_ptr() const {
    return meta_data_ptr_;
  }

  /// \brief Fetch the pointer to the meta data manager for this bulk data manager.
  std::shared_ptr<MetaData> mesh_meta_data_ptr() {
    return meta_data_ptr_;
  }
  //@}

 protected:
  //! \name Constructor
  //@{

  /// \brief This constructor wraps and extends that of stk's BulkData.
  /// \param meta_data_ptr [in] A pointer to this mesh's meta data manager.
  /// \param comm [in] The MPI communicator.
  /// \param auto_aura_option [in] The chosen automatic Aura option.
  /// \param field_data_manager_ptr [in] A pointer to an existing field data manager.
  /// \param initial_bucket_capacity [in] The initial bucket capacity.
  /// \param bucket_capacity [in] The maximum bucket capacity.
  /// \param aura_ghosting_ptr [in] A pointer to this mesh's aura ghosting manager.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  BulkData(std::shared_ptr<MetaData> meta_data_ptr, stk::ParallelMachine comm,
           enum stk::mesh::BulkData::AutomaticAuraOption auto_aura_option = stk::mesh::BulkData::AUTO_AURA,
#ifdef SIERRA_MIGRATION
           bool add_fmwk_data_flag = false,
#endif
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
           std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager_ptr = nullptr,
#else
           stk::mesh::FieldDataManager *const field_data_manager_ptr = nullptr,
#endif
           unsigned initial_bucket_capacity = stk::mesh::get_default_initial_bucket_capacity(),
           unsigned maximum_bucket_capacity = stk::mesh::get_default_maximum_bucket_capacity(),
           std::shared_ptr<stk::mesh::impl::AuraGhosting> aura_ghosting_ptr =
               std::shared_ptr<stk::mesh::impl::AuraGhosting>(),
           bool upward_connectivity_flag = true)
      : stk::mesh::BulkData(std::static_pointer_cast<stk::mesh::MetaData>(meta_data_ptr), comm, auto_aura_option,
#ifdef SIERRA_MIGRATION
                            add_fmwk_data_flag,
#endif
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
                            std::move(field_data_manager_ptr),
#else
                            field_data_manager_ptr,
#endif
                            initial_bucket_capacity, maximum_bucket_capacity, aura_ghosting_ptr,
                            upward_connectivity_flag),
        meta_data_ptr_(meta_data_ptr) {
  }
  //@}

 private:
  //! \name Internal members
  //@{

  std::shared_ptr<MetaData> meta_data_ptr_;
  //@}

  //! \name Friends!
  //@{

  friend class MeshBuilder;
  //@}
};  // BulkData

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_BULKDATA_HPP_
