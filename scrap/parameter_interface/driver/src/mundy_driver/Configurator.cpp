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

/// \file Configurator.cpp
/// \brief Definition of the Configurator class

// C++ core libs

// Trilinos libs
#include <Teuchos_YamlParameterListHelpers.hpp>

// Mundy libs
#include <mundy_driver/Configurator.hpp>  // for mundy::driver::Configurator
#include <mundy_driver/Driver.hpp>        // for mundy::driver::Driver

namespace mundy {

namespace driver {

/// \name Using directives
//@{

// Shorthand names for the different meta method factories we use later
// TODO(cje): There must be a better way to get the MetaMethod factories and check/create from them in a general way.
// Right now I have ugly if-else blocks both here and the Driver, which are a pain, and make the code more fragile if we
// add things.
using FactoryMM = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodExecutionInterface<void>>;
using FactoryMMS = mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>;
using FactoryMMPS =
    mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>;

//@}

/// \name Constructors and destructors
//@{

Configurator::Configurator() {
}

Configurator::Configurator(stk::ParallelMachine comm) {
  comm_ = comm;
  has_comm_ = true;
}

//@}

/// \name Setters
//@{

Configurator &Configurator::set_configuration_version(const unsigned configuration_version) {
  configuration_version_ = configuration_version;
  return *this;
}

Configurator &Configurator::set_spatial_dimension(const unsigned spatial_dimension) {
  spatial_dimension_ = spatial_dimension;
  return *this;
}

Configurator &Configurator::set_entity_rank_names(const std::vector<std::string> &entity_rank_names) {
  entity_rank_names_ = entity_rank_names;
  return *this;
}

Configurator &Configurator::set_communicator(const stk::ParallelMachine &comm) {
  comm_ = comm;
  return *this;
}

Configurator &Configurator::set_param_list(const Teuchos::ParameterList &param_list) {
  param_list_ = param_list;
  return *this;
}

Configurator &Configurator::set_input_file_name(const std::string &input_file_name) {
  input_file_name_ = input_file_name;
  return *this;
}

Configurator &Configurator::set_input_file_type(const std::string &input_file_type) {
  input_file_type_ = input_file_type;
  return *this;
}

Configurator &Configurator::set_input_file(const std::string &input_file_name, const std::string &input_file_type) {
  input_file_name_ = input_file_name;
  input_file_type_ = input_file_type;
  return *this;
}

Configurator &Configurator::set_node_coordinate_field_name(const std::string &node_coordinate_field_name) {
  node_coordinate_field_name_ = node_coordinate_field_name;
  return *this;
}

Configurator &Configurator::set_driver(std::shared_ptr<Driver> driver) {
  // Check if this is the same driver we already know about
  {
    const bool same_driver_already_set = driver_ptr_.get() == driver.get();
    if (same_driver_already_set) {
      return *this;
    }
  }

  // If this isn't the same driver, and we already have a driver, throw an exception
  MUNDY_THROW_REQUIRE(driver_ptr_ == nullptr, std::logic_error,
                      "mundy::driver::Configurator Driver already initialized.");

  // Assign driver at this point
  driver_ptr_ = driver;

  return *this;
}

//@}

//! @name Getters
//@{

std::shared_ptr<mundy::meta::MeshReqs> Configurator::get_mesh_requirements() {
  return mesh_reqs_ptr_;
}

//@}

//! @name Actions
//@{

Configurator &Configurator::parse_parameters() {
  // First determine what kind of file we are reading
  if (input_file_type_ == "yaml") {
    param_list_ = *Teuchos::getParametersFromYamlFile(input_file_name_);
  } else if (input_file_type_ == "xml") {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                        "mundy::driver::Configurator XML files are not implemented for reading yet.");
  } else {
    MUNDY_THROW_REQUIRE(
        false, std::invalid_argument,
        std::string("mundy::driver::Configurator file_format ") + input_file_type_ + " not recognized.");
  }

  // At this point we are expecting to have a valid param_list_. Get into the Configuration section first, and configure
  // the MetaMethod* that we need
  MUNDY_THROW_REQUIRE(param_list_.isSublist("configuration"), std::invalid_argument,
                      "mundy::driver::ParseParameters parameters does not contain an 'configuration' sublist.");

  const Teuchos::ParameterList config_params = param_list_.sublist("configuration");
  parse_configuration(config_params);

  // Parse the action list
  MUNDY_THROW_REQUIRE(param_list_.isSublist("actions"), std::invalid_argument,
                      "mundy::driver::ParseParameters parameters does not contain an 'actions' sublist.");
  const Teuchos::ParameterList action_params = param_list_.sublist("actions");
  parse_actions(action_params);

  return *this;
}

Configurator &Configurator::parse_configuration(const Teuchos::ParameterList &config_params) {
  // Get simulation variables that don't belong to a specific Meta*
  int configuration_version = config_params.get<int>("configuration_version");
  set_configuration_version(configuration_version);
  int n_dim = config_params.get<int>("n_dim");
  set_spatial_dimension(n_dim);

  // Look for the node_coordinates name
  std::string node_coordinate_field_name(default_node_coordinate_field_name_);
  if (config_params.isParameter("node_coordinates_field_name")) {
    node_coordinate_field_name = config_params.get<std::string>("node_coordiantes_field_name");
  }
  set_node_coordinate_field_name(node_coordinate_field_name);

  // Look for restart condition
  if (config_params.isParameter("restart")) {
    is_restart_ = config_params.get<bool>("restart");
    if (is_restart_) {
      restart_filename_ = config_params.get<std::string>("restart_filename");
    }
  }

  // Loop over known MetaMethod types and parse_and_configure them
  for (auto metamethod_type = metamethod_types_.begin(); metamethod_type != metamethod_types_.end();
       ++metamethod_type) {
    std::string metamethod_str(*metamethod_type);
    // Configure the MetaMethod interface
    if (config_params.isSublist(metamethod_str)) {
      const Teuchos::ParameterList metamethod_params = config_params.sublist(metamethod_str);
      parse_meta_method_type(metamethod_str, metamethod_params);
    }
  }
  return *this;
}

Configurator &Configurator::parse_actions(const Teuchos::ParameterList &action_params) {
  // Loop over the different phases and set them up
  std::vector<std::string> phase_types{"setup", "run", "finalize"};
  for (auto phase = phase_types.begin(); phase != phase_types.end(); phase++) {
    // Check if we have information for this phase
    if (action_params.isSublist(*phase)) {
      const Teuchos::ParameterList phase_sublist = action_params.sublist(*phase);

      for (auto pit = phase_sublist.begin(); pit != phase_sublist.end(); ++pit) {
        // Get the name and check that it exists in the unordered map for enabled meta methods
        const std::string &method_name = pit->first;
        const Teuchos::ParameterEntry &entry = pit->second;

        MUNDY_THROW_REQUIRE(
            enabled_meta_methods_.contains(method_name), std::invalid_argument,
            std::string("Configurator did not find enabled meta method ") + method_name + " when parsing actions.");

        // Get the parameters for the type of trigger we are going to use
        const Teuchos::ParameterList action_sublist = Teuchos::getValue<Teuchos::ParameterList>(entry);

        all_actions_[*phase].push_back(std::make_tuple(method_name, action_sublist));
      }
    }
  }

  // Look for the special n_steps variable in configuration version 0
  int n_steps = action_params.get<int>("n_steps");
  n_steps_ = n_steps;
  return *this;
}

Configurator &Configurator::parse_meta_method_type(const std::string &method_type,
                                                   const Teuchos::ParameterList &method_params) {
  // Loop over MetaMethod sublist
  for (auto pit = method_params.begin(); pit != method_params.end(); ++pit) {
    const std::string &param_name = pit->first;
    const Teuchos::ParameterEntry &entry = pit->second;

    // Dive into the sublist of this method and get the method it's trying to call
    MUNDY_THROW_REQUIRE(
        entry.isList(), std::invalid_argument,
        std::string("mundy::driver::Configurator::parse_meta_method_type Invalid specification of method ") +
            param_name);
    const Teuchos::ParameterList method_sublist = Teuchos::getValue<Teuchos::ParameterList>(entry);

    // Check to see if the method is a valid one (registered with MetaFactory). Unfortunately, without reflection, we
    // have to make this an ugly if/else statement
    const std::string method_name = method_sublist.get<std::string>("method");
    if (method_type == "meta_method_execution_interface") {
      bool is_valid_metamethod = FactoryMM::is_valid_key(method_name);
      MUNDY_THROW_REQUIRE(
          is_valid_metamethod, std::invalid_argument,
          std::string("mundy::driver::Configurator::parse_meta_method_type Could not find MetaMethod ") + method_name +
              " for name " + param_name);
    } else if (method_type == "meta_method_subset_execution_interface") {
      bool is_valid_metamethod = FactoryMMS::is_valid_key(method_name);
      MUNDY_THROW_REQUIRE(
          is_valid_metamethod, std::invalid_argument,
          std::string("mundy::driver::Configurator::parse_meta_method_type Could not find MetaMethod ") + method_name +
              " for name " + param_name);
    } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
      bool is_valid_metamethod = FactoryMMPS::is_valid_key(method_name);
      MUNDY_THROW_REQUIRE(
          is_valid_metamethod, std::invalid_argument,
          std::string("mundy::driver::Configurator::parse_meta_method_type Could not find MetaMethod ") + method_name +
              " for name " + param_name);
    }

    // Now that we have the name of the method, etc, we can slice off the fixed/mutable parameters and validate them. Do
    // both together, as we are going to have an icky if block again I believe. Also, create a default structure to
    // load, as sometimes we won't have both types of paramters specified.
    Teuchos::ParameterList fixed_params;
    Teuchos::ParameterList mutable_params;
    if (method_sublist.isSublist("fixed_params")) {
      fixed_params = method_sublist.sublist("fixed_params");
    }
    if (method_sublist.isSublist("mutable_params")) {
      mutable_params = method_sublist.sublist("mutable_params");
    }
    // Create a mirrored copy of the params to validate against
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    Teuchos::ParameterList valid_mutable_params = mutable_params;

    // Validate based on the type of the factory
    if (method_type == "meta_method_execution_interface") {
      valid_fixed_params.validateParametersAndSetDefaults(FactoryMM::get_valid_fixed_params(method_name));
      valid_mutable_params.validateParametersAndSetDefaults(FactoryMM::get_valid_mutable_params(method_name));
    } else if (method_type == "meta_method_subset_execution_interface") {
      valid_fixed_params.validateParametersAndSetDefaults(FactoryMMS::get_valid_fixed_params(method_name));
      valid_mutable_params.validateParametersAndSetDefaults(FactoryMMS::get_valid_mutable_params(method_name));
    } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
      valid_fixed_params.validateParametersAndSetDefaults(FactoryMMPS::get_valid_fixed_params(method_name));
      valid_mutable_params.validateParametersAndSetDefaults(FactoryMMPS::get_valid_mutable_params(method_name));
    }

    // Build the enabled package to pass to Driver later. First, check to see if we already have a unique user name for
    // this action, and if so, throw an error up that the user cannot do this, each name they give to a meta method must
    // be unique.
    if (enabled_meta_methods_.find(param_name) != enabled_meta_methods_.end()) {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("User-defined MetaMethod name ") + param_name +
                              " already exist, check configuration for duplicate MetaMethod names");
    }
    // Now that we have a unique name, create the structure for the map for later (this is two commands because getting
    // the type coersion for std::make_tuple doesn't seem to like things for some reason...)
    std::tuple mtuple(method_type, method_name, valid_fixed_params, valid_mutable_params);
    enabled_meta_methods_[param_name] = mtuple;
  }  // for loop over user-given method names
  return *this;
}

std::shared_ptr<mundy::meta::MeshReqs> Configurator::create_mesh_requirements() {
  mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshReqs>(comm_);
  mesh_reqs_ptr_->set_spatial_dimension(spatial_dimension_);
  mesh_reqs_ptr_->set_entity_rank_names(entity_rank_names_);

  // Loop over the elements in the enabled_meta_methods_ map and add them to the requirements
  for (const auto &[key, value] : enabled_meta_methods_) {
    auto [method_type, method_name, fixed_params, mutable_params] = value;

    // Add a single requirement to the mesh, taking into account what factory it came from
    // TODO(cje): At some point having this be a single factory, or making this easier, would be nice
    if (method_type == "meta_method_execution_interface") {
      mesh_reqs_ptr_->sync(FactoryMM::get_mesh_requirements(method_name, fixed_params));
    } else if (method_type == "meta_method_subset_execution_interface") {
      mesh_reqs_ptr_->sync(FactoryMMS::get_mesh_requirements(method_name, fixed_params));
    } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
      mesh_reqs_ptr_->sync(FactoryMMPS::get_mesh_requirements(method_name, fixed_params));
    }
  }

  return mesh_reqs_ptr_;
}

std::shared_ptr<Driver> Configurator::generate_driver() {
  // If we don't have a driver, create an instance
  if (driver_ptr_ == nullptr) {
    driver_ptr_ = std::make_shared<Driver>(comm_);
  }

  // Check to make sure we've already called build_mesh_requirements (mesh_reqs_ptr_ isn't null)
  MUNDY_THROW_REQUIRE(mesh_reqs_ptr_ != nullptr, std::invalid_argument,
                      "Cannot create a Driver without mesh requirements.");

  // This does not commit the mesh as it is supposed to play nicely with IO, which sometimes commits the mesh on a
  // restart.
  bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
  meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
  // Also set to using simple fields
  meta_data_ptr_->use_simple_fields();
  // Set the coordinate field name
  meta_data_ptr_->set_coordinate_field_name(node_coordinate_field_name_);

  // TODO(cje): Remove later
  // stk::mesh::impl::dump_all_mesh_info(*bulk_data_ptr_, std::cout);

  // Here is where we might read in a restart mesh file, for now, throw an error if this is the case
  if (is_restart_) {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Mundy currently does not enable restarts.");
  } else {
    // Commit the mesh if not a restart
    meta_data_ptr_->commit();
  }

  // Now create instances of the meta methods that we want to configure. Loop over the elements in the
  // enabled_meta_methods_ map, created a shared version and a map, and then hand them to the driver as well.
  for (const auto &[key, value] : enabled_meta_methods_) {
    auto [method_type, method_name, fixed_params, mutable_params] = value;

    if (method_type == "meta_method_execution_interface") {
      // Create a new class instance
      std::shared_ptr<mundy::meta::MetaMethodExecutionInterface<void>> new_meta_method =
          FactoryMM::create_new_instance(method_name, bulk_data_ptr_.get(), fixed_params);
      new_meta_method->set_mutable_params(mutable_params);
      meta_methods_map_[method_name] = new_meta_method;
    } else if (method_type == "meta_method_subset_execution_interface") {
      // Create a new class instance
      std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> new_meta_method =
          FactoryMMS::create_new_instance(method_name, bulk_data_ptr_.get(), fixed_params);
      new_meta_method->set_mutable_params(mutable_params);
      meta_methods_subset_map_[method_name] = new_meta_method;
    } else if (method_type == "meta_method_pairwise_subset_execution_interface") {
      // Create a new class instance
      std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> new_meta_method =
          FactoryMMPS::create_new_instance(method_name, bulk_data_ptr_.get(), fixed_params);
      new_meta_method->set_mutable_params(mutable_params);
      meta_methods_pairwise_subset_map_[method_name] = new_meta_method;
    }
  }

  return driver_ptr_;
}

//@}

}  // namespace driver

}  // namespace mundy
