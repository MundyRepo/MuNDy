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

#ifndef MUNDY_DRIVER_CONFIGURATOR_HPP_
#define MUNDY_DRIVER_CONFIGURATOR_HPP_

/// \file Configurator.hpp
/// \brief Declaration of the Configurator class

// C++ core libs
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy includes
#include <MundyDriver_config.hpp>                                     // for HAVE_MUNDYDRIVER_*
#include <mundy_meta/MeshReqs.hpp>                                    // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                                 // for mundy::meta::StringBasedMetaFactory
#include <mundy_meta/MetaMethodExecutionInterface.hpp>                // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionInterface
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS

#ifdef HAVE_MUNDYDRIVER_MUNDYAGENTS
#include <mundy_agents/Agents.hpp>             // for mundy::agents::Agents
#include <mundy_agents/HierarchyOfAgents.hpp>  // for mundy::agents::HierarchyOfAgents
#include <mundy_agents/RegisterAgents.hpp>     // MUNDY_REGISTER_AGENTS
#endif                                         // HAVE_MUNDYDRIVER_MUNDYAGENTS

#ifdef HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/Constraints.hpp>                // for mundy::constraints::Constraints
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#endif                                                      // HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYDRIVER_MUNDYIO
#include <mundy_io/IOBroker.hpp>  // for mundy::io::IOBroker
#endif                            // HAVE_MUNDYDRIVER_MUNDYIO

#ifdef HAVE_MUNDYDRIVER_MUNDYLINKERS
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>                // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>              // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>               // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>         // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/Linkers.hpp>                               // for mundy::linkers::Linkers
#include <mundy_linkers/NeighborLinkers.hpp>                       // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSphereLinkers
#endif                                                             // HAVE_MUNDYDRIVER_MUNDYLINKERS

#ifdef HAVE_MUNDYDRIVER_MUNDYSHAPES
#include <mundy_shapes/ComputeAABB.hpp>             // for mundy::shapes::ComputeAABB
#include <mundy_shapes/ComputeBoundingRadius.hpp>   // for mundy::shapes::ComputeBoundingRadius
#include <mundy_shapes/ComputeOBB.hpp>              // for mundy::shapes::ComputeOBB
#include <mundy_shapes/DeclareAndInitShapes.hpp>    // for mundy::shapes::DeclareAndInitShapes
#include <mundy_shapes/Spheres.hpp>                 // for mundy::shapes::Spheres
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments
#include <mundy_shapes/Spherocylinders.hpp>         // for mundy::shapes::Spherocylinders
#endif                                              // HAVE_MUNDYDRIVER_MUNDYSHAPES

// Class forward definitions (if we don't want the header)
// clang-format off
namespace mundy { namespace driver { class Driver; } }
// clang-format on

namespace mundy {

namespace driver {

/// \class Configurator
/// \brief Class for reading in a configuration file and configuring a driver (simulation context)
///
/// The Configurator is responsible for parsing the input configuration file (YAML, XML, etc) and either passing it onto
/// a Driver, or creating a temporary Driver that can be used to query enabled/registered metamethods.
class Configurator {
 public:
  //! \name Constructors and destructors
  //@{

  /// \brief Default constructor
  Configurator();

  /// \brief Constructor with given communicator
  explicit Configurator(stk::ParallelMachine comm);

  //@}

  //! @name Setters
  //@{

  /// \brief Set the configuration version
  /// \param [in] configuration_version
  /// \return
  Configurator &set_configuration_version(const unsigned configuration_version);

  /// \brief Set the number of spatial dimensions
  /// \param [in] spatial_dimension
  /// \return
  Configurator &set_spatial_dimension(const unsigned spatial_dimension);

  /// \brief Set the entity rank names for the STK mesh
  /// \param [in] entity_rank_names
  /// \return
  Configurator &set_entity_rank_names(const std::vector<std::string> &entity_rank_names);

  /// \brief Set the MPI communicator
  /// \param [in] comm
  /// \return
  Configurator &set_communicator(const stk::ParallelMachine &comm);

  /// \brief Set the mastet configuration parameter list
  /// \param [in] param_list
  /// \return
  Configurator &set_param_list(const Teuchos::ParameterList &param_list);

  /// \brief Set the input configuration file name
  /// \param [in] input_file_name
  /// \return
  Configurator &set_input_file_name(const std::string &input_file_name);

  /// \brief Set the input configuration file type [YAML]
  /// \param [in] input_file_type
  /// \return
  Configurator &set_input_file_type(const std::string &input_file_type);

  /// \brief Set the input configuration file name and type
  /// \param [in] input_file_name
  /// \param [in] input_file_type
  /// \return
  Configurator &set_input_file(const std::string &input_file_name, const std::string &input_file_type);

  /// \brief Set the coordinate field name
  /// \param [in] node_coordinate_field_name
  /// \return
  Configurator &set_node_coordinate_field_name(const std::string &node_coordinate_field_name);

  /// \brief Set the driver if it already exists
  /// \param [in] driver
  /// \return
  Configurator &set_driver(std::shared_ptr<Driver> driver);

  //@}

  //! @name Getters
  //@{

  /// \brief Get the mesh requirements
  /// \return std::shared_ptr<mundy::meta::MeshReqs>
  std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements();

  //@}

  //! @name Actions
  //@{

  /// \brief Parse the parameters in internal ParameterList
  ///
  /// Parses and stores configuration information for entire simulation. This includes both the configuration (enabled
  /// MetaMethods) and their ordering (Actions).
  /// \return
  Configurator &parse_parameters();

  /// \brief Parse configuration section of parameters
  /// \param [in] config_params
  /// \return
  Configurator &parse_configuration(const Teuchos::ParameterList &config_params);

  /// \brief Parse actions section of parameters
  /// \param [in] action_params
  /// \return
  Configurator &parse_actions(const Teuchos::ParameterList &action_params);

  /// \brief Parse a MetaMethod type
  /// \param [in] method_type
  /// \param [in] method_params
  /// \return
  Configurator &parse_meta_method_type(const std::string &method_type, const Teuchos::ParameterList &method_params);

  /// \brief Create mesh requirements from enabled meta methods
  /// \return std::shared_ptr<mundy::meta::MeshReqs>
  std::shared_ptr<mundy::meta::MeshReqs> create_mesh_requirements();

  /// \brief Generate complete simulation driver to run simulation
  ///
  /// Creates the driver if it doesn't exist. If it does exist, populate it from the enabled MetaMethods and action
  /// ordering.
  /// \return
  std::shared_ptr<Driver> generate_driver();

  //@}

  //! @name Print/format
  //@{

  friend auto operator<<(std::ostream &os, const Configurator &m) -> std::ostream & {
    os << "Configuration Version: " << m.configuration_version_ << std::endl;
    os << "Enabled MetaMethods\n";
    for (const auto &[key, value] : m.enabled_meta_methods_) {
      os << "................................\n";
      os << ".MetaMethod: " << key << std::endl;
      auto [method_type, method_name, fixed_params, mutable_params] = value;
      os << "...Method type: " << method_type << std::endl;
      os << "...Method NAME: " << method_name << std::endl;
      os << "...Fixed Params:\n" << fixed_params;
      os << "...Mutable Params:\n" << mutable_params;
    }
    os << "Actions\n";
    os << ".Number of steps" << m.n_steps_ << std::endl;
    std::vector<std::string> phase_types{"setup", "run", "finalize"};
    for (const auto &phase : phase_types) {
      os << "................................\n";
      os << ".Phase: " << phase << std::endl;
      const auto action_iter = m.all_actions_.find(phase);
      for (const auto &[action, trigger_params] : action_iter->second) {
        os << "...Action name: " << action << std::endl;
        os << "...Trigger:\n" << trigger_params;
      }
    }
    return os;
  }

  //@}

 private:
  //! \name Default parameters
  //@{

  /// \brief MetaMethod type names
  static constexpr std::array<std::string_view, 3> metamethod_types_ = {
      "meta_method_execution_interface", "meta_method_subset_execution_interface",
      "meta_method_pairwise_subset_execution_interface"};

  static constexpr std::string_view default_node_coordinate_field_name_ = "NODE_COORDS";

  //@}

  //! \name Internal members
  //@{

  /// \brief Teuchos ParameterList for global information
  Teuchos::ParameterList param_list_;

  /// \brief MPI communicator to use
  stk::ParallelMachine comm_ = MPI_COMM_NULL;

  /// \brief Has the communicator been set?
  bool has_comm_ = false;

  /// \brief Is this a restart?
  bool is_restart_ = false;

  /// \brief Configuration version
  unsigned configuration_version_ = 0;

  /// \brief Number of dimensions
  unsigned spatial_dimension_ = 0;

  /// \brief Number of steps for run
  unsigned n_steps_ = 0;

  /// \brief Entity rank names
  std::vector<std::string> entity_rank_names_;

  /// \brief Node coordinate field name
  std::string node_coordinate_field_name_ = "";

  /// \brief Input file name
  std::string input_file_name_ = "";

  /// \brief Input file type
  std::string input_file_type_ = "";

  /// \brief Restart filename (if applicable)
  std::string restart_filename_ = "";

  /// \brief Enabled MetaMethods [User-defined name](MetaMethod type, MetaMethod name, fixed parameters, mutable
  /// parameters)
  std::unordered_map<std::string, std::tuple<std::string, std::string, Teuchos::ParameterList, Teuchos::ParameterList>>
      enabled_meta_methods_;

  // TODO(cje): This will be replaced with a DAG construct eventually. For now, we expect a simulation to have a singly
  // executed setup, some number of steps in the main loop, and then a finalize that does post-processing and tears down
  // the system.
  /// \brief Actions for differe phases [phase](Vector<User-defined name, Trigger Parameters>)
  std::unordered_map<std::string, std::vector<std::tuple<std::string, Teuchos::ParameterList>>> all_actions_;

  /// \brief Associated Driver instance
  //
  // TODO(cje): Think about if we want to hold a driver pointer ourselves, as the Configurator does not own the driver
  // in any way, or have it pass into all of the configuration methods. The reason to keep it on hand is so that we
  // aren't constantly writing down the Driver arguments, as well as incrementing the shared_ptr. The downside is that
  // it requires setting it on the inside of the class, and then not changing it, leading to undefined behavior.
  std::shared_ptr<Driver> driver_ptr_ = nullptr;

  /// Associated mesh requirements
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr_ = nullptr;

  /// \brief Mundy bulk data pointer
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_ = nullptr;

  /// \brief Mundy meta data pointer
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_ = nullptr;

  //! \name MetaMethod instance lookups
  //@{

  std::unordered_map<std::string, std::shared_ptr<mundy::meta::MetaMethodExecutionInterface<void>>> meta_methods_map_;
  std::unordered_map<std::string, std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>>>
      meta_methods_subset_map_;
  std::unordered_map<std::string, std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>>
      meta_methods_pairwise_subset_map_;

  //@}

  //}
};  // Configurator

}  // namespace driver

}  // namespace mundy

#endif  // MUNDY_DRIVER_CONFIGURATOR_HPP_
