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

/// \file Driver.cpp
/// \brief Definition of the Driver class

// C++ core libs

// Trilinos libs
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <stk_mesh/base/DumpMeshInfo.hpp>  // for stk::mesh::impl::dump_all_mesh_info

// Mundy libs
#include <mundy_driver/Driver.hpp>  // for mundy::driver::Configurator

namespace mundy {

namespace driver {

/// \name Using directives
//@{

// Shorthand names for the different meta method factories we use later
using FactoryMM = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodExecutionInterface<void>>;
using FactoryMMS = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>;
using FactoryMMPS =
    mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>;

//@}

/// \name Constructors and destructors
//@{

Driver::Driver() {
}

Driver::Driver(const stk::ParallelMachine& comm) {
  comm_ = comm;
  has_comm_ = true;
}

//@}

//! \name Queries of registered "methods"
//@{

std::string Driver::get_registered_meta_method_execution_interface() {
  return FactoryMM::get_keys_as_string();
}

std::string Driver::get_registered_meta_method_subset_execution_interface() {
  return FactoryMMS::get_keys_as_string();
}

std::string Driver::get_registered_meta_method_pairwise_subset_execution_interface() {
  return FactoryMMPS::get_keys_as_string();
}

std::string Driver::get_registered_classes() {
  return get_registered_meta_method_execution_interface() + get_registered_meta_method_subset_execution_interface() +
         get_registered_meta_method_pairwise_subset_execution_interface();
}

//@}

//! \name Setters and Getters
//@{

Driver& Driver::set_spatial_dimension(const unsigned spatial_dimension) {
  spatial_dimension_ = spatial_dimension;
  return *this;
}

Driver& Driver::set_communicator(const stk::ParallelMachine& comm) {
  comm_ = comm;
  return *this;
}

Driver& Driver::set_node_coordinate_field_name(const std::string& node_coordinate_field_name) {
  node_coordinate_field_name_ = node_coordinate_field_name;
  return *this;
}

//@}

}  // namespace driver

}  // namespace mundy
