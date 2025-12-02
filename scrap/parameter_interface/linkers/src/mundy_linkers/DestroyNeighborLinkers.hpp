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

#ifndef MUNDY_LINKERS_DESTROYNEIGHBORLINKERS_HPP_
#define MUNDY_LINKERS_DESTROYNEIGHBORLINKERS_HPP_

/// \file DestroyNeighborLinkers.hpp
/// \brief Declaration of the DestroyNeighborLinkers class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::make_string_literal
#include <mundy_linkers/destroy_neighbor_linkers/techniques/DestroyBoundNeighbors.hpp>  // for mundy::linkers::destroy_neighbor_linkers::techniques::DestroyBoundNeighbors
#include <mundy_linkers/destroy_neighbor_linkers/techniques/DestroyDistantNeighbors.hpp>  // for mundy::linkers::destroy_neighbor_linkers::techniques::DestroyDistantNeighbors
#include <mundy_mesh/BulkData.hpp>                                                        // for mundy::mesh::BulkData
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_meta/MetaTechniqueDispatcher.hpp>             // for mundy::meta::MetaMethodSubsetExecutionDispatcher

namespace mundy {

namespace linkers {

/// \class DestroyNeighborLinkers
/// \brief Method for generating neighbor linkers between source-target entity pairs.
class DestroyNeighborLinkers
    : public mundy::meta::MetaMethodSubsetExecutionDispatcher<
          DestroyNeighborLinkers, void, mundy::meta::make_registration_string("DESTROY_NEIGHBOR_LINKERS"),
          mundy::meta::make_registration_string("NO_DEFAULT_TECHNIQUE")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  DestroyNeighborLinkers() = delete;

  /// \brief Constructor
  DestroyNeighborLinkers(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaMethodSubsetExecutionDispatcher<
            DestroyNeighborLinkers, void, mundy::meta::make_registration_string("DESTROY_NEIGHBOR_LINKERS"),
            mundy::meta::make_registration_string("NO_DEFAULT_TECHNIQUE")>(bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaMethodSubsetExecutionDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we will forward to the techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to the techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we require all techniques registered with our technique factory to
  /// have.
  static Teuchos::ParameterList get_valid_required_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require all techniques registered with our technique factory to
  /// have.
  static Teuchos::ParameterList get_valid_required_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_neighbor_linkers_part_name_ = "NEIGHBOR_LINKERS";
  //@}
};  // DestroyNeighborLinkers

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_destroy_neighbor_linkers_kernels_ = []() {
  // Register our default kernels
  mundy::linkers::DestroyNeighborLinkers::OurTechniqueFactory::register_new_class<
      mundy::linkers::destroy_neighbor_linkers::techniques::DestroyDistantNeighbors>("DESTROY_DISTANT_NEIGHBORS");
  mundy::linkers::DestroyNeighborLinkers::OurTechniqueFactory::register_new_class<
      mundy::linkers::destroy_neighbor_linkers::techniques::DestroyBoundNeighbors>("DESTROY_BOUND_NEIGHBORS");
  return true;
}();

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_DESTROYNEIGHBORLINKERS_HPP_
