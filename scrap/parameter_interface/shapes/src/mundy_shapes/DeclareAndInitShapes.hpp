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

#ifndef MUNDY_SHAPES_DECLAREANDINITSHAPES_HPP_
#define MUNDY_SHAPES_DECLAREANDINITSHAPES_HPP_

/// \file DeclareAndInitShapes.hpp
/// \brief Declaration of the DeclareAndInitShapes class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_meta/MetaTechniqueDispatcher.hpp>             // for mundy::meta::MetaMethodExecutionDispatcher
#include <mundy_shapes/declare_and_initialize_shapes/techniques/GridOfSpheres.hpp>  // for mundy::shapes::declare_and_initialize_shapes::techniques::GridOfSpheres

namespace mundy {

namespace shapes {

/// \class DeclareAndInitShapes
/// \brief Method for generating neighbor linkers between source-target entity pairs.
class DeclareAndInitShapes
    : public mundy::meta::MetaMethodExecutionDispatcher<
          DeclareAndInitShapes, void, mundy::meta::make_registration_string("DECLARE_AND_INIT_SHAPES"),
          mundy::meta::make_registration_string("NO_DEFAULT_TECHNIQUE")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  DeclareAndInitShapes() = delete;

  /// \brief Constructor
  DeclareAndInitShapes(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaMethodExecutionDispatcher<DeclareAndInitShapes, void,
                                                   mundy::meta::make_registration_string("DECLARE_AND_INIT_SHAPES"),
                                                   mundy::meta::make_registration_string("NO_DEFAULT_TECHNIQUE")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaMethodExecutionDispatcher static interface implementation
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
};  // DeclareAndInitShapes

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_declare_and_init_shapes_kernels_ = []() {
  // Register our default kernels
  mundy::shapes::DeclareAndInitShapes::OurTechniqueFactory::register_new_class<
      mundy::shapes::declare_and_initialize_shapes::techniques::GridOfSpheres>("GRID_OF_SPHERES");
  return true;
}();

}  // namespace shapes

}  // namespace mundy

#endif  // MUNDY_SHAPES_DECLAREANDINITSHAPES_HPP_
