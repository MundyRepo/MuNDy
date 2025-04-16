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

#pragma TODO "Add a brief description of the file here."
/* Notes:

Brief note:
We are trying to simulate the HP1 project with Hossein.

Here is the generic form of a single chromatin chain. In this example, we have 3 chromatin 'repeats', which means 2
heterochromatin sections, and 1 euchromatin section in the middle.

E : euchromatin spheres
H : heterochromatin spheres
| : HP1 crosslinker
--- : backbone spring

|   |                           |   |
H---H---E---E---E---E---E---E---H---H

Chromatin backbone:
Backbone segments are modeled as spherocylinder segments, and can have a different spring constants that determine the
separation between adjacent spheres.

Interactions:

1. Backbone segment --- Backbone segment
   - Interactions are via a hertzian contact potential.
2. Backbone segment -- itself
   - Interactions are via a harmonic spring between adjacent spheres, but done along the spherocylinder segment
(backbone).


*/

// External libs
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <openrand/philox.h>

// C++ core
#include <algorithm>   // for std::transform
#include <filesystem>  // for std::filesystem::path
#include <fstream>     // for std::ofstream
#include <iostream>    // for std::cout, std::endl
#include <memory>      // for std::shared_ptr, std::unique_ptr
#include <numeric>     // for std::accumulate
#include <regex>       // for std::regex
#include <string>      // for std::string
#include <vector>      // for std::vector

// Trilinos libs
#include <Kokkos_Core.hpp>                     // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>    // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>           // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>             // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>          // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Comm.hpp>              // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>      // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>            // for stk::mesh::Entity
#include <stk_mesh/base/EntitySorterBase.hpp>  // for stk::mesh::EntitySorterBase
#include <stk_mesh/base/FieldParallel.hpp>     // for stk::mesh::parallel_sum

// Mundy core
#include <mundy_core/MakeStringArray.hpp>                      // for mundy::core::make_string_array
#include <mundy_core/OurAnyNumberParameterEntryValidator.hpp>  // for mundy::core::OurAnyNumberParameterEntryValidator
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT

// Mundy math
#include <mundy_math/Hilbert.hpp>                      // for mundy::math::create_hilbert_positions_and_directors
#include <mundy_math/Vector3.hpp>                      // for mundy::math::Vector3
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>  // for mundy::math::distance::ellipsoid_ellipsoid
#include <mundy_math/zmort.hpp>                        // for mundy::math::zmorton_less(Vector3, Vector3)

// Mundy meta
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements

// Mundy mesh
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>  // adds fmt::format for stk types
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>  // for mundy::mesh::utils::destroy_flagged_entities
#include <mundy_mesh/utils/FillFieldWithValue.hpp>      // for mundy::mesh::utils::fill_field_with_value

// Mundy shapes
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

// Mundy constraint
#include <mundy_constraints/HookeanSprings.hpp>  // for mundy::constraints::HookeanSprings

// Mundy linkers
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>         // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>       // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>        // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                // for mundy::linkers::NeighborLinkers

// Mundy IO
#include <mundy_io/IOBroker.hpp>  // for mundy::io::IOBroker

// Mundy alens
#include <mundy_alens/actions_crosslinkers.hpp>             // for mundy::alens::crosslinkers...
#include <mundy_alens/periphery/Periphery.hpp>              // for gen_sphere_quadrature
#include <mundy_constraints/AngularSprings.hpp>             // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints

namespace mundy {

namespace alens {

namespace hp1 {

enum class BINDING_STATE_CHANGE : unsigned {
  NONE = 0u,
  LEFT_TO_DOUBLY,
  RIGHT_TO_DOUBLY,
  DOUBLY_TO_LEFT,
  DOUBLY_TO_RIGHT
};
enum class INITIALIZATION_TYPE : unsigned {
  GRID = 0u,
  RANDOM_UNIT_CELL,
  OVERLAP_TEST,
  HILBERT_RANDOM_UNIT_CELL,
  HILBERT_LATTICE,
  HILBERT_FILE,
  USHAPE_TEST,
  FROM_EXO,
  FROM_DAT
};

enum class BOND_TYPE : unsigned { HARMONIC = 0u, FENE, FENEWCA };
enum class PERIPHERY_BIND_SITES_TYPE : unsigned { RANDOM = 0u, FROM_FILE };
enum class PERIPHERY_SHAPE : unsigned { SPHERE = 0u, ELLIPSOID };
enum class PERIPHERY_QUADRATURE : unsigned { GAUSS_LEGENDRE = 0u, FROM_FILE };
enum class COLLISION_TYPE : unsigned { HERTZIAN = 0u, WCA };
enum class HYDRO_TYPE : unsigned { RPYC = 0u, RPY, STOKES };

std::ostream &operator<<(std::ostream &os, const BINDING_STATE_CHANGE &state) {
  switch (state) {
    case BINDING_STATE_CHANGE::NONE:
      os << "NONE";
      break;
    case BINDING_STATE_CHANGE::LEFT_TO_DOUBLY:
      os << "LEFT_TO_DOUBLY";
      break;
    case BINDING_STATE_CHANGE::RIGHT_TO_DOUBLY:
      os << "RIGHT_TO_DOUBLY";
      break;
    case BINDING_STATE_CHANGE::DOUBLY_TO_LEFT:
      os << "DOUBLY_TO_LEFT";
      break;
    case BINDING_STATE_CHANGE::DOUBLY_TO_RIGHT:
      os << "DOUBLY_TO_RIGHT";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const INITIALIZATION_TYPE &init_type) {
  switch (init_type) {
    case INITIALIZATION_TYPE::GRID:
      os << "GRID";
      break;
    case INITIALIZATION_TYPE::RANDOM_UNIT_CELL:
      os << "RANDOM_UNIT_CELL";
      break;
    case INITIALIZATION_TYPE::OVERLAP_TEST:
      os << "OVERLAP_TEST";
      break;
    case INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL:
      os << "HILBERT_RANDOM_UNIT_CELL";
      break;
    case INITIALIZATION_TYPE::HILBERT_LATTICE:
      os << "HILBERT_LATTICE";
      break;
    case INITIALIZATION_TYPE::HILBERT_FILE:
      os << "HILBERT_FILE";
      break;
    case INITIALIZATION_TYPE::USHAPE_TEST:
      os << "USHAPE_TEST";
      break;
    case INITIALIZATION_TYPE::FROM_EXO:
      os << "FROM_EXO";
      break;
    case INITIALIZATION_TYPE::FROM_DAT:
      os << "FROM_DAT";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const BOND_TYPE &bond_type) {
  switch (bond_type) {
    case BOND_TYPE::HARMONIC:
      os << "HARMONIC";
      break;
    case BOND_TYPE::FENE:
      os << "FENE";
      break;
    case BOND_TYPE::FENEWCA:
      os << "FENEWCA";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const PERIPHERY_BIND_SITES_TYPE &bind_sites_type) {
  switch (bind_sites_type) {
    case PERIPHERY_BIND_SITES_TYPE::RANDOM:
      os << "RANDOM";
      break;
    case PERIPHERY_BIND_SITES_TYPE::FROM_FILE:
      os << "FROM_FILE";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const PERIPHERY_SHAPE &periphery_shape) {
  switch (periphery_shape) {
    case PERIPHERY_SHAPE::SPHERE:
      os << "SPHERE";
      break;
    case PERIPHERY_SHAPE::ELLIPSOID:
      os << "ELLIPSOID";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const PERIPHERY_QUADRATURE &periphery_quadrature) {
  switch (periphery_quadrature) {
    case PERIPHERY_QUADRATURE::GAUSS_LEGENDRE:
      os << "GAUSS_LEGENDRE";
      break;
    case PERIPHERY_QUADRATURE::FROM_FILE:
      os << "FROM_FILE";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const COLLISION_TYPE &collision_type) {
  switch (collision_type) {
    case COLLISION_TYPE::HERTZIAN:
      os << "HERTZIAN";
      break;
    case COLLISION_TYPE::WCA:
      os << "WCA";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const HYDRO_TYPE &hydro_type) {
  switch (hydro_type) {
    case HYDRO_TYPE::RPYC:
      os << "RPYC";
      break;
    case HYDRO_TYPE::RPY:
      os << "RPY";
      break;
    case HYDRO_TYPE::STOKES:
      os << "STOKES";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

}  // namespace hp1

}  // namespace alens

}  // namespace mundy

template <>
struct fmt::formatter<mundy::alens::hp1::BINDING_STATE_CHANGE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::INITIALIZATION_TYPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::BOND_TYPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::PERIPHERY_BIND_SITES_TYPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::PERIPHERY_SHAPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::PERIPHERY_QUADRATURE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::COLLISION_TYPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::HYDRO_TYPE> : fmt::ostream_formatter {};

namespace mundy {

namespace alens {

namespace hp1 {

// Forward declare the enums and declare their fmt formatters.
class HP1 {
 public:
  using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
  using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

  HP1() = default;

  void print_rank0(auto thing_to_print, int indent_level = 0) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::string indent(indent_level * 2, ' ');
      std::cout << indent << thing_to_print << std::endl;
    }
  }

  void parse_user_inputs(int argc, char **argv) {
    // Parse the command line options to find the input filename
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("params", &input_parameter_filename_, "The name of the input file.");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_result = cmdp.parse(argc, argv);
    if (parse_result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      print_help_message();

      // Safely exit the program
      // If we print the help message, we don't need to do anything else.
      exit(0);
    } else if (parse_result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      throw std::invalid_argument("Failed to parse the command line arguments.");
    }

    // Read, validate, and parse in the parameters from the parameter list.
    try {
      Teuchos::ParameterList param_list = *Teuchos::getParametersFromYamlFile(input_parameter_filename_);
      set_params(param_list);
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Failed to read the input parameter file." << std::endl;
      std::cerr << "During read, the following error occurred: " << e.what() << std::endl;
      std::cerr << "NOTE: This can happen for any number of reasons. Check that the file exists and contains the "
                   "expected parameters."
                << std::endl;
      throw e;
    }
  }

  void print_help_message() {
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "To run this code, please pass in --params=<input.yaml> as a command line argument." << std::endl;
    std::cout << std::endl;
    std::cout << "Note, all parameters and sublists in input.yaml must be contained in a single top-level list."
              << std::endl;
    std::cout << "Such as:" << std::endl;
    std::cout << std::endl;
    std::cout << "HP1:" << std::endl;
    std::cout << "  num_time_steps: 1000" << std::endl;
    std::cout << "  timestep_size: 1e-6" << std::endl;
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "The valid parameters that can be set in the input file are:" << std::endl;
    Teuchos::ParameterList valid_params = HP1::get_valid_params();

    auto print_options =
        Teuchos::ParameterList::PrintOptions().showTypes(false).showDoc(true).showDefault(true).showFlags(false).indent(
            1);
    valid_params.print(std::cout, print_options);
    std::cout << "#############################################################################################"
              << std::endl;
  }

  void set_params(const Teuchos::ParameterList &param_list) {
    // Validate the parameters and set the defaults.
    Teuchos::ParameterList valid_param_list = param_list;
    valid_param_list.validateParametersAndSetDefaults(HP1::get_valid_params());

    // Simulation parameters:
    Teuchos::ParameterList &simulation_params = valid_param_list.sublist("simulation");
    num_time_steps_ = simulation_params.get<size_t>("num_time_steps");
    timestep_size_ = simulation_params.get<double>("timestep_size");
    viscosity_ = simulation_params.get<double>("viscosity");
    num_chromosomes_ = simulation_params.get<size_t>("num_chromosomes");
    num_chromatin_repeats_ = simulation_params.get<size_t>("num_chromatin_repeats");
    num_euchromatin_per_repeat_ = simulation_params.get<size_t>("num_euchromatin_per_repeat");
    num_heterochromatin_per_repeat_ = simulation_params.get<size_t>("num_heterochromatin_per_repeat");
    backbone_sphere_hydrodynamic_radius_ = simulation_params.get<double>("backbone_sphere_hydrodynamic_radius");
    std::string hydro_type_string = simulation_params.get<std::string>("hydro_type");
    if (hydro_type_string == "RPYC") {
      hydro_type_ = HYDRO_TYPE::RPYC;
    } else if (hydro_type_string == "RPY") {
      hydro_type_ = HYDRO_TYPE::RPY;
    } else if (hydro_type_string == "STOKES") {
      hydro_type_ = HYDRO_TYPE::STOKES;
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid hydro type. Received '") + hydro_type_string +
                              "' but expected 'RPYC', 'RPY', or 'STOKES'.");
    }
    initial_chromosome_separation_ = simulation_params.get<double>("initial_chromosome_separation");
    MUNDY_THROW_REQUIRE(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(viscosity_ > 0, std::invalid_argument, "viscosity_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(initial_chromosome_separation_ >= 0, std::invalid_argument,
                        "initial_chromosome_separation_ must be greater than or equal to 0.");

    io_frequency_ = simulation_params.get<size_t>("io_frequency");
    log_frequency_ = simulation_params.get<size_t>("log_frequency");
    output_filename_ = simulation_params.get<std::string>("output_filename");
    enable_continuation_if_available_ = simulation_params.get<bool>("enable_continuation_if_available");
    std::string initiliazation_type_string = simulation_params.get<std::string>("initialization_type");
    if (initiliazation_type_string == "GRID") {
      initialization_type_ = INITIALIZATION_TYPE::GRID;
    } else if (initiliazation_type_string == "RANDOM_UNIT_CELL") {
      initialization_type_ = INITIALIZATION_TYPE::RANDOM_UNIT_CELL;
    } else if (initiliazation_type_string == "OVERLAP_TEST") {
      initialization_type_ = INITIALIZATION_TYPE::OVERLAP_TEST;
    } else if (initiliazation_type_string == "HILBERT_RANDOM_UNIT_CELL") {
      initialization_type_ = INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL;
    } else if (initiliazation_type_string == "HILBERT_LATTICE") {
      initialization_type_ = INITIALIZATION_TYPE::HILBERT_LATTICE;
    } else if (initiliazation_type_string == "HILBERT_FILE") {
      initialization_type_ = INITIALIZATION_TYPE::HILBERT_FILE;
      hilbert_centers_filename_ = simulation_params.get<std::string>("hilbert_centers_filename");
    } else if (initiliazation_type_string == "FROM_EXO") {
      initialization_type_ = INITIALIZATION_TYPE::FROM_EXO;
      initialize_from_exo_filename_ = simulation_params.get<std::string>("initialize_from_exo_filename");
    } else if (initiliazation_type_string == "FROM_DAT") {
      initialization_type_ = INITIALIZATION_TYPE::FROM_DAT;
      initialize_from_dat_filename_ = simulation_params.get<std::string>("initialize_from_dat_filename");
    } else if (initiliazation_type_string == "USHAPE_TEST") {
      initialization_type_ = INITIALIZATION_TYPE::USHAPE_TEST;
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid initialization type. Received '") + initiliazation_type_string +
                              "' but expected 'GRID', 'RANDOM_UNIT_CELL', "
                              "'OVERLAP_TEST', 'HILBERT_RANDOM_UNIT_CELL', 'HILBERT_LATTICE', 'HILBERT_FILE', or "
                              "'FROM_FILE'.");
    }
    if (initialization_type_ == INITIALIZATION_TYPE::RANDOM_UNIT_CELL ||
        initialization_type_ == INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL ||
        initialization_type_ == INITIALIZATION_TYPE::HILBERT_LATTICE) {
      Teuchos::Array<double> unit_cell_size = simulation_params.get<Teuchos::Array<double>>("unit_cell_size");
      unit_cell_size_[0] = unit_cell_size[0];
      unit_cell_size_[1] = unit_cell_size[1];
      unit_cell_size_[2] = unit_cell_size[2];
    }
    loadbalance_post_initialization_ = simulation_params.get<bool>("loadbalance_post_initialization");
    check_maximum_speed_pre_position_update_ = simulation_params.get<bool>("check_maximum_speed_pre_position_update");
    if (check_maximum_speed_pre_position_update_) {
      max_allowable_speed_ = simulation_params.get<double>("max_allowable_speed");
    }
    hydro_update_frequency_ = simulation_params.get<size_t>("hydro_update_frequency");
    MUNDY_THROW_REQUIRE(hydro_update_frequency_ > 0, std::invalid_argument,
                        "hydro_update_frequency_ must be greater than 0.");

    enable_chromatin_brownian_motion_ = simulation_params.get<bool>("enable_chromatin_brownian_motion");
    enable_backbone_springs_ = simulation_params.get<bool>("enable_backbone_springs");
    enable_backbone_collision_ = simulation_params.get<bool>("enable_backbone_collision");
    enable_backbone_n_body_hydrodynamics_ = simulation_params.get<bool>("enable_backbone_n_body_hydrodynamics");
    enable_crosslinkers_ = simulation_params.get<bool>("enable_crosslinkers");
    enable_periphery_collision_ = simulation_params.get<bool>("enable_periphery_collision");
    enable_periphery_hydrodynamics_ = simulation_params.get<bool>("enable_periphery_hydrodynamics");
    enable_periphery_binding_ = simulation_params.get<bool>("enable_periphery_binding");
    MUNDY_THROW_REQUIRE(enable_periphery_hydrodynamics_ ? enable_backbone_n_body_hydrodynamics_ : true,
                        std::invalid_argument,
                        "Logically periphery hydrodynamics requires backbone hydrodynamics to be enabled.");
    enable_active_euchromatin_forces_ = simulation_params.get<bool>("enable_active_euchromatin_forces");

    if (enable_chromatin_brownian_motion_) {
      set_brownian_motion_params(valid_param_list.sublist("brownian_motion"));
    }
    if (enable_backbone_springs_) {
      set_backbone_springs_params(valid_param_list.sublist("backbone_springs"));
    }
    if (enable_backbone_collision_) {
      set_backbone_collision_params(valid_param_list.sublist("backbone_collision"));
    }
    if (enable_crosslinkers_) {
      set_crosslinker_params(valid_param_list.sublist("crosslinker"));
    }
    if (enable_periphery_collision_) {
      set_periphery_collision_params(valid_param_list.sublist("periphery_collision"));
    }
    if (enable_periphery_hydrodynamics_) {
      set_periphery_hydrodynamic_params(valid_param_list.sublist("periphery_hydro"));
    }
    if (enable_periphery_binding_) {
      set_periphery_binding_params(valid_param_list.sublist("periphery_binding"));
    }
    if (enable_active_euchromatin_forces_) {
      set_active_euchromatin_forces_params(valid_param_list.sublist("active_euchromatin_forces"));
    }

    set_neighbor_list_params(valid_param_list.sublist("neighbor_list"));
  }

  void set_brownian_motion_params(const Teuchos::ParameterList &param_list) {
    brownian_kt_ = param_list.get<double>("kt");
  }

  void set_backbone_springs_params(const Teuchos::ParameterList &param_list) {
    const std::string backbone_spring_type_string = param_list.get<std::string>("spring_type");
    if (backbone_spring_type_string == "HARMONIC") {
      backbone_spring_type_ = BOND_TYPE::HARMONIC;
      backbone_spring_constant_ = param_list.get<double>("spring_constant");
      backbone_spring_r0_ = param_list.get<double>("spring_r0");
    } else if (backbone_spring_type_string == "FENE") {
      backbone_spring_type_ = BOND_TYPE::FENE;
      backbone_spring_constant_ = param_list.get<double>("spring_constant");
      backbone_spring_r0_ = param_list.get<double>("spring_r0");
    } else if (backbone_spring_type_string == "FENEWCA") {
      backbone_spring_type_ = BOND_TYPE::FENEWCA;
      backbone_spring_constant_ = param_list.get<double>("spring_constant");
      backbone_spring_r0_ = param_list.get<double>("spring_r0");
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid backbone spring type. Received '") + backbone_spring_type_string +
                              "' but expected 'HARMONIC' or 'FENE' or 'FENEWCA'.");
    }
  }

  void set_backbone_collision_params(const Teuchos::ParameterList &param_list) {
    const std::string backbone_collision_type_string = param_list.get<std::string>("backbone_collision_type");
    backbone_excluded_volume_radius_ = param_list.get<double>("backbone_excluded_volume_radius");
    if (backbone_collision_type_string == "HERTZIAN") {
      backbone_collision_type_ = COLLISION_TYPE::HERTZIAN;
      backbone_youngs_modulus_ = param_list.get<double>("backbone_youngs_modulus");
      backbone_poissons_ratio_ = param_list.get<double>("backbone_poissons_ratio");
    } else if (backbone_collision_type_string == "WCA") {
      backbone_collision_type_ = COLLISION_TYPE::WCA;
      backbone_wca_epsilon_ = param_list.get<double>("backbone_wca_epsilon");
      backbone_wca_sigma_ = param_list.get<double>("backbone_wca_sigma");
      backbone_wca_cutoff_ = param_list.get<double>("backbone_wca_cutoff");
    }
  }

  void set_crosslinker_params(const Teuchos::ParameterList &param_list) {
    const std::string crosslinker_spring_type_string = param_list.get<std::string>("spring_type");
    if (crosslinker_spring_type_string == "HARMONIC") {
      crosslinker_spring_type_ = BOND_TYPE::HARMONIC;
    } else if (crosslinker_spring_type_string == "FENE") {
      crosslinker_spring_type_ = BOND_TYPE::FENE;
    } else if (crosslinker_spring_type_string == "FENEWCA") {
      crosslinker_spring_type_ = BOND_TYPE::FENEWCA;
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid crosslinker spring type. Received '") + crosslinker_spring_type_string +
                              "' but expected 'HARMONIC' or 'FENE' or 'FENEWCA'.");
    }

    crosslinker_kt_ = param_list.get<double>("kt");
    crosslinker_spring_constant_ = param_list.get<double>("spring_constant");
    crosslinker_r0_ = param_list.get<double>("spring_r0");
    crosslinker_left_binding_rate_ = param_list.get<double>("left_binding_rate");
    crosslinker_right_binding_rate_ = param_list.get<double>("right_binding_rate");
    crosslinker_left_unbinding_rate_ = param_list.get<double>("left_unbinding_rate");
    crosslinker_right_unbinding_rate_ = param_list.get<double>("right_unbinding_rate");
    if (crosslinker_spring_type_ == BOND_TYPE::HARMONIC) {
      crosslinker_cutoff_radius_ =
          crosslinker_r0_ + 5.0 * std::sqrt(1.0 / (crosslinker_kt_ * crosslinker_spring_constant_));
    } else if (crosslinker_spring_type_ == BOND_TYPE::FENE) {
      // The r0 quantity for FENE bonds is the rmax at which force goes to infinity, so anything beyond this in invalid!
      crosslinker_cutoff_radius_ = crosslinker_r0_;
    } else if (crosslinker_spring_type_ == BOND_TYPE::FENEWCA) {
      // The r0 quantity for FENE bonds is the rmax at which force goes to infinity, so anything beyond this in invalid!
      crosslinker_cutoff_radius_ = crosslinker_r0_;
    }
  }

  void set_periphery_hydrodynamic_params(const Teuchos::ParameterList &param_list) {
    check_maximum_periphery_overlap_ = param_list.get<bool>("check_maximum_periphery_overlap");
    if (check_maximum_periphery_overlap_) {
      maximum_allowed_periphery_overlap_ = param_list.get<double>("maximum_allowed_periphery_overlap");
    }

    std::string periphery_hydro_shape_string = param_list.get<std::string>("shape");
    if (periphery_hydro_shape_string == "SPHERE") {
      periphery_hydro_shape_ = PERIPHERY_SHAPE::SPHERE;
      periphery_hydro_radius_ = param_list.get<double>("radius");
    } else if (periphery_hydro_shape_string == "ELLIPSOID") {
      periphery_hydro_shape_ = PERIPHERY_SHAPE::ELLIPSOID;
      periphery_hydro_axis_radius1_ = param_list.get<double>("axis_radius1");
      periphery_hydro_axis_radius2_ = param_list.get<double>("axis_radius2");
      periphery_hydro_axis_radius3_ = param_list.get<double>("axis_radius3");
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid hydrodynamic periphery shape. Received '") +
                              periphery_hydro_shape_string + "' but expected 'SPHERE' or 'ELLIPSOID'.");
    }
    std::string periphery_hydro_quadrature_string = param_list.get<std::string>("quadrature");
    if (periphery_hydro_quadrature_string == "GAUSS_LEGENDRE") {
      MUNDY_THROW_REQUIRE((periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) ||
                              ((periphery_hydro_shape_ == PERIPHERY_SHAPE::ELLIPSOID) &&
                               (periphery_hydro_axis_radius1_ == periphery_hydro_axis_radius2_) &&
                               (periphery_hydro_axis_radius2_ == periphery_hydro_axis_radius3_) &&
                               (periphery_hydro_axis_radius3_ == periphery_hydro_axis_radius1_)),
                          std::invalid_argument, "Gauss-Legendre quadrature is only valid for spherical peripheries.");
      periphery_hydro_quadrature_ = PERIPHERY_QUADRATURE::GAUSS_LEGENDRE;
      periphery_hydro_spectral_order_ = param_list.get<size_t>("spectral_order");
    } else if (periphery_hydro_quadrature_string == "FROM_FILE") {
      periphery_hydro_quadrature_ = PERIPHERY_QUADRATURE::FROM_FILE;
      periphery_hydro_num_quadrature_points_ = param_list.get<size_t>("num_quadrature_points");
      periphery_hydro_quadrature_points_filename_ = param_list.get<std::string>("quadrature_points_filename");
      periphery_hydro_quadrature_weights_filename_ = param_list.get<std::string>("quadrature_weights_filename");
      periphery_hydro_quadrature_normals_filename_ = param_list.get<std::string>("quadrature_normals_filename");
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid periphery quadrature. Received '") + periphery_hydro_quadrature_string +
                              "' but expected 'GAUSS_LEGENDRE' or 'FROM_FILE'.");
    }
  }

  void set_periphery_collision_params(const Teuchos::ParameterList &param_list) {
    std::string periphery_collision_shape_string = param_list.get<std::string>("shape");
    periphery_collision_spring_constant_ = param_list.get<double>("collision_spring_constant");
    if (periphery_collision_shape_string == "SPHERE") {
      periphery_collision_shape_ = PERIPHERY_SHAPE::SPHERE;
      periphery_collision_radius_ = param_list.get<double>("radius");
    } else if (periphery_collision_shape_string == "ELLIPSOID") {
      periphery_collision_shape_ = PERIPHERY_SHAPE::ELLIPSOID;
      periphery_collision_axis_radius1_ = param_list.get<double>("axis_radius1");
      periphery_collision_axis_radius2_ = param_list.get<double>("axis_radius2");
      periphery_collision_axis_radius3_ = param_list.get<double>("axis_radius3");
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid collision periphery shape. Received '") +
                              periphery_collision_shape_string + "' but expected 'SPHERE' or 'ELLIPSOID'.");
    }

    periphery_collision_use_fast_approx_ = param_list.get<bool>("use_fast_approx");
    shrink_periphery_over_time_ = param_list.get<bool>("shrink_periphery_over_time");
    if (shrink_periphery_over_time_) {
      set_periphery_collision_shrinkage_params(param_list.sublist("shrinkage"));
    }
  }

  void set_periphery_collision_shrinkage_params(const Teuchos::ParameterList &param_list) {
    periphery_collision_shrinkage_num_steps_ = param_list.get<size_t>("num_shrinkage_steps");
    periphery_collision_scale_factor_before_shrinking_ = param_list.get<double>("scale_factor_before_shrinking");
  }

  void set_periphery_binding_params(const Teuchos::ParameterList &param_list) {
    periphery_binding_rate_ = param_list.get<double>("binding_rate");
    periphery_unbinding_rate_ = param_list.get<double>("unbinding_rate");
    // NOTE: Binding to the periphery will piggyback off of the type of crosslinking bonds (HARMONIC or FENE) used.
    periphery_spring_constant_ = param_list.get<double>("spring_constant");
    periphery_spring_r0_ = param_list.get<double>("spring_r0");
    std::string periphery_bind_sites_type_string = param_list.get<std::string>("bind_sites_type");
    if (periphery_bind_sites_type_string == "RANDOM") {
      periphery_bind_sites_type_ = PERIPHERY_BIND_SITES_TYPE::RANDOM;
      periphery_num_bind_sites_ = param_list.get<size_t>("num_bind_sites");
    } else if (periphery_bind_sites_type_string == "FROM_FILE") {
      periphery_bind_sites_type_ = PERIPHERY_BIND_SITES_TYPE::FROM_FILE;
      periphery_num_bind_sites_ = param_list.get<size_t>("num_bind_sites");
      periphery_bind_site_locations_filename_ = param_list.get<std::string>("bind_site_locations_filename");
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Invalid periphery binding sites type. Received '") +
                              periphery_bind_sites_type_string + "' but expected 'RANDOM' or 'FROM_FILE'.");
    }
  }

  void set_neighbor_list_params(const Teuchos::ParameterList &param_list) {
    skin_distance_ = param_list.get<double>("skin_distance");
    force_neighborlist_update_ = param_list.get<bool>("force_neighborlist_update");
    force_neighborlist_update_nsteps_ = param_list.get<size_t>("force_neighborlist_update_nsteps");
    print_neighborlist_statistics_ = param_list.get<bool>("print_neighborlist_statistics");
  }

  void set_active_euchromatin_forces_params(const Teuchos::ParameterList &param_list) {
    active_euchromatin_force_sigma_ = param_list.get<double>("force_sigma");
    active_euchromatin_force_kon_ = param_list.get<double>("kon");
    active_euchromatin_force_koff_ = param_list.get<double>("koff");
  }

  static Teuchos::ParameterList get_valid_params() {
    // Create a paramater entity validator for our large integers to allow for both int and long long.
    auto prefer_size_t = []() {
      if (std::is_same_v<size_t, unsigned short>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_SHORT;
      } else if (std::is_same_v<size_t, unsigned int>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      } else if (std::is_same_v<size_t, unsigned long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG;
      } else if (std::is_same_v<size_t, unsigned long long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG_LONG;
      } else {
        throw std::runtime_error("Unknown size_t type.");
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      }
    }();
    const bool allow_all_types_by_default = false;
    mundy::core::OurAnyNumberParameterEntryValidator::AcceptedTypes accept_int(allow_all_types_by_default);
    accept_int.allow_all_integer_types(true);
    auto make_new_validator = [](const auto &preferred_type, const auto &accepted_types) {
      return Teuchos::rcp(new mundy::core::OurAnyNumberParameterEntryValidator(preferred_type, accepted_types));
    };

    static Teuchos::ParameterList valid_parameter_list;

    valid_parameter_list.sublist("simulation")
        .set("num_time_steps", default_num_time_steps_, "Number of time steps.",
             make_new_validator(prefer_size_t, accept_int))
        .set("timestep_size", default_timestep_size_, "Time step size.")
        .set("viscosity", default_viscosity_, "Viscosity.")
        .set("num_chromosomes", default_num_chromosomes_, "Number of chromosomes.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_chromatin_repeats", default_num_chromatin_repeats_, "Number of chromatin repeats per chain.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_euchromatin_per_repeat", default_num_euchromatin_per_repeat_,
             "Number of euchromatin beads per repeat.", make_new_validator(prefer_size_t, accept_int))
        .set("num_heterochromatin_per_repeat", default_num_heterochromatin_per_repeat_,
             "Number of heterochromatin beads per repeat.", make_new_validator(prefer_size_t, accept_int))
        .set("backbone_sphere_hydrodynamic_radius", default_backbone_sphere_hydrodynamic_radius_,
             "Backbone sphere hydrodynamic radius. Even if n-body hydrodynamics is disabled, we still have "
             "self-interaction.")
        .set("hydro_type", std::string(default_backbone_hydro_type_), "Hydrodynamic kernel type.")
        .set("initial_chromosome_separation", default_initial_chromosome_separation_, "Initial chromosome separation.")
        .set("initialization_type", std::string(default_initialization_type_string_), "Initialization_type.")
        .set("initialize_from_exo_filename", std::string(default_initialize_from_exo_filename_),
             "Exo file to initialize from if initialization_type is FROM_EXO.")
        .set("initialize_from_dat_filename", std::string(default_initialize_from_dat_filename_),
             "Dat file to initialize from if initialization_type is FROM_DAT.")
        .set("hilbert_centers_filename", std::string(default_hilbert_centers_filename_),
             "Dat file containing the centers of the Hilbert curve if initialization_type is HILBERT_FILE.")
        .set<Teuchos::Array<double>>(
            "unit_cell_size",
            Teuchos::tuple<double>(default_unit_cell_size_[0], default_unit_cell_size_[1], default_unit_cell_size_[2]),
            "Unit cell size in each dimension. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set("check_maximum_speed_pre_position_update", default_check_maximum_speed_pre_position_update_,
             "Check maximum speed before updating positions.")
        .set("max_allowable_speed", default_max_allowable_speed_,
             "Maximum allowable speed (only used if "
             "check_maximum_speed_pre_position_update is true).")
        .set("hydro_update_frequency", default_hydro_update_frequency_,
             "The frequency at which we should update the background fluid velocity field. If not 1, the old velocity "
             "will be reused for the next update_frequency steps.",
             make_new_validator(prefer_size_t, accept_int))
        // IO
        .set("loadbalance_post_initialization", default_loadbalance_post_initialization_,
             "If we should load balance post-initialization or not.")
        .set("io_frequency", default_io_frequency_, "Number of timesteps between writing output.",
             make_new_validator(prefer_size_t, accept_int))
        .set("log_frequency", default_log_frequency_, "Number of timesteps between logging.",
             make_new_validator(prefer_size_t, accept_int))
        .set("output_filename", std::string(default_output_filename_), "Output filename.")
        .set("enable_continuation_if_available", default_enable_continuation_if_available_,
             "Enable continuing a previous simulation if an output file already exists.")
        // Control flags
        .set("enable_chromatin_brownian_motion", default_enable_chromatin_brownian_motion_,
             "Enable chromatin Brownian motion.")
        .set("enable_backbone_springs", default_enable_backbone_springs_, "Enable backbone springs.")
        .set("enable_backbone_collision", default_enable_backbone_collision_, "Enable backbone collision.")
        .set("enable_backbone_n_body_hydrodynamics", default_enable_backbone_n_body_hydrodynamics_,
             "Enable backbone N-body hydrodynamics.")
        .set("enable_crosslinkers", default_enable_crosslinkers_, "Enable crosslinkers.")
        .set("enable_periphery_collision", default_enable_periphery_collision_, "Enable periphery collision.")
        .set("enable_periphery_hydrodynamics", default_enable_periphery_hydrodynamics_,
             "Enable periphery hydrodynamics.")
        .set("enable_periphery_binding", default_enable_periphery_binding_, "Enable periphery binding.")
        .set("enable_active_euchromatin_forces", default_enable_active_euchromatin_forces_,
             "Enable active euchromatin forces.");

    valid_parameter_list.sublist("brownian_motion")
        .set("kt", default_brownian_kt_, "Temperature kT for Brownian Motion.");

    valid_parameter_list.sublist("backbone_springs")
        .set("spring_type", std::string(default_backbone_spring_type_string_), "Chromatin spring type.")
        .set("spring_constant", default_backbone_spring_constant_, "Chromatin spring constant.")
        .set("spring_r0", default_backbone_spring_r0_, "Chromatin rest length (HARMONIC) or rmax (FENE, FENEWCA).");

    valid_parameter_list.sublist("backbone_collision")
        .set("backbone_excluded_volume_radius", default_backbone_excluded_volume_radius_,
             "Backbone excluded volume radius.")
        .set("backbone_collision_type", std::string(default_backbone_collision_type_string_),
             "Backbone collision type.")
        .set("backbone_youngs_modulus", default_backbone_youngs_modulus_, "Backbone Young's modulus.")
        .set("backbone_poissons_ratio", default_backbone_poissons_ratio_, "Backbone Poisson's ratio.")
        .set("backbone_wca_epsilon", default_backbone_wca_epsilon_, "Backbone WCA epsilon.")
        .set("backbone_wca_sigma", default_backbone_wca_sigma_, "Backbone WCA sigma.")
        .set("backbone_wca_cutoff", default_backbone_wca_cutoff_, "Backbone WCA cutoff.");

    valid_parameter_list.sublist("crosslinker")
        .set("spring_type", std::string(default_crosslinker_spring_type_string_), "Crosslinker spring type.")
        .set("kt", default_crosslinker_kt_, "Temperature kT for crosslinkers.")
        .set("spring_constant", default_crosslinker_spring_constant_, "Crosslinker spring constant.")
        .set("spring_r0", default_crosslinker_r0_, "Crosslinker rest length.")
        .set("left_binding_rate", default_crosslinker_left_binding_rate_, "Crosslinker left binding rate.")
        .set("right_binding_rate", default_crosslinker_right_binding_rate_, "Crosslinker right binding rate.")
        .set("left_unbinding_rate", default_crosslinker_left_unbinding_rate_, "Crosslinker left unbinding rate.")
        .set("right_unbinding_rate", default_crosslinker_right_unbinding_rate_, "Crosslinker right unbinding rate.");

    valid_parameter_list.sublist("periphery_hydro")
        .set("check_maximum_periphery_overlap", default_check_maximum_periphery_overlap_,
             "Check maximum periphery overlap.")
        .set("maximum_allowed_periphery_overlap", default_maximum_allowed_periphery_overlap_,
             "Maximum allowed periphery overlap.")
        .set("shape", std::string(default_periphery_hydro_shape_string_), "Periphery hydrodynamic shape.")
        .set("radius", default_periphery_hydro_radius_, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", default_periphery_hydro_axis_radius1_,
             "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", default_periphery_hydro_axis_radius2_,
             "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", default_periphery_hydro_axis_radius3_,
             "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("quadrature", std::string(default_periphery_hydro_quadrature_string_), "Periphery quadrature.")
        .set("spectral_order", default_periphery_hydro_spectral_order_,
             "Periphery spectral order (only used if periphery is spherical is Gauss-Legendre quadrature).",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_quadrature_points", default_periphery_hydro_num_quadrature_points_,
             "Periphery number of quadrature points (only used if quadrature type is FROM_FILE). Number of points in "
             "the files must match this quantity.",
             make_new_validator(prefer_size_t, accept_int))
        .set("quadrature_points_filename", std::string(default_periphery_hydro_quadrature_points_filename_),
             "Periphery quadrature points filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_weights_filename", std::string(default_periphery_hydro_quadrature_weights_filename_),
             "Periphery quadrature weights filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_normals_filename", std::string(default_periphery_hydro_quadrature_normals_filename_),
             "Periphery quadrature normals filename (only used if quadrature type is FROM_FILE).");

    valid_parameter_list.sublist("periphery_collision")
        .set("shape", std::string(default_periphery_collision_shape_string_), "Periphery collision shape.")
        .set("radius", default_periphery_collision_radius_,
             "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("collision_spring_constant", default_periphery_collision_spring_constant_,
             "Periphery collision spring constant.")
        .set("axis_radius1", default_periphery_collision_axis_radius1_,
             "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", default_periphery_collision_axis_radius2_,
             "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", default_periphery_collision_axis_radius3_,
             "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("use_fast_approx", default_periphery_collision_use_fast_approx_, "Use fast periphery collision.")
        .set("shrink_periphery_over_time", default_shrink_periphery_over_time_, "Shrink periphery over time.")
        .sublist("shrinkage")
        .set("num_shrinkage_steps", default_periphery_collision_shrinkage_num_steps_,
             "Number of steps over which to perform the shrinking process (should not exceed num_time_steps).",
             make_new_validator(prefer_size_t, accept_int))
        .set("scale_factor_before_shrinking", default_periphery_collision_scale_factor_before_shrinking_,
             "Scale factor before shrinking.");

    valid_parameter_list.sublist("periphery_binding")
        .set("binding_rate", default_periphery_binding_rate_, "Periphery binding rate.")
        .set("unbinding_rate", default_periphery_unbinding_rate_, "Periphery unbinding rate.")
        .set("spring_constant", default_periphery_spring_constant_, "Periphery spring constant.")
        .set("spring_r0", default_periphery_spring_r0_, "Periphery spring rest length.")
        .set("bind_sites_type", std::string(default_periphery_bind_sites_type_string_), "Periphery bind sites type.")
        .set("num_bind_sites", default_periphery_num_bind_sites_,
             "Periphery number of binding sites (only used if periphery_binding_sites_type is RANDOM and periphery "
             "has spherical or ellipsoidal shape).",
             make_new_validator(prefer_size_t, accept_int))
        .set("bind_site_locations_filename", std::string(default_periphery_bind_site_locations_filename_),
             "Periphery binding sites filename (only used if periphery_binding_sites_type is FROM_FILE).");

    valid_parameter_list.sublist("active_euchromatin_forces")
        .set("force_sigma", default_active_euchromatin_force_sigma_, "Active euchromatin force sigma.")
        .set("kon", default_active_euchromatin_force_kon_, "Active euchromatin force kon.")
        .set("koff", default_active_euchromatin_force_koff_, "Active euchromatin force koff.");

    valid_parameter_list.sublist("neighbor_list")
        .set("skin_distance", default_skin_distance_, "Neighbor list skin distance.")
        .set("force_neighborlist_update", default_force_neighborlist_update_, "Force update of the neighbor list.")
        .set("force_neighborlist_update_nsteps", default_force_neighborlist_update_nsteps_,
             "Number of timesteps between force update of the neighbor list.",
             make_new_validator(prefer_size_t, accept_int))
        .set("print_neighborlist_statistics", default_print_neighborlist_statistics_,
             "Print neighbor list statistics.");

    return valid_parameter_list;
  }

  void dump_user_inputs() {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;

      std::cout << std::endl;
      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  num_time_steps:  " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size:   " << timestep_size_ << std::endl;
      std::cout << "  viscosity:       " << viscosity_ << std::endl;
      std::cout << "  num_chromosomes: " << num_chromosomes_ << std::endl;
      std::cout << "  num_chromatin_repeats:      " << num_chromatin_repeats_ << std::endl;
      std::cout << "  num_euchromatin_per_repeat: " << num_euchromatin_per_repeat_ << std::endl;
      std::cout << "  num_heterochromatin_per_repeat:  " << num_heterochromatin_per_repeat_ << std::endl;
      std::cout << "  backbone_sphere_hydrodynamic_radius: " << backbone_sphere_hydrodynamic_radius_ << std::endl;
      std::cout << "  hydro_type: " << hydro_type_ << std::endl;
      std::cout << "  initial_chromosome_separation:   " << initial_chromosome_separation_ << std::endl;
      std::cout << "  initialization_type:             " << initialization_type_ << std::endl;
      if (initialization_type_ == INITIALIZATION_TYPE::FROM_EXO) {
        std::cout << "  initialize_from_file_filename: " << initialize_from_exo_filename_ << std::endl;
      }
      if (initialization_type_ == INITIALIZATION_TYPE::FROM_DAT) {
        std::cout << "  initialize_from_file_filename: " << initialize_from_dat_filename_ << std::endl;
      }

      if ((initialization_type_ == INITIALIZATION_TYPE::RANDOM_UNIT_CELL) ||
          (initialization_type_ == INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL)) {
        std::cout << "  unit_cell_size: {" << unit_cell_size_[0] << ", " << unit_cell_size_[1] << ", "
                  << unit_cell_size_[2] << "}" << std::endl;
      }
      std::cout << "  loadbalance_post_initialization: " << loadbalance_post_initialization_ << std::endl;
      std::cout << "  check_maximum_speed_pre_position_update: " << check_maximum_speed_pre_position_update_
                << std::endl;
      if (check_maximum_speed_pre_position_update_) {
        std::cout << "  max_allowable_speed: " << max_allowable_speed_ << std::endl;
      }
      std::cout << "  hydro_update_frequency: " << hydro_update_frequency_ << std::endl;

      std::cout << std::endl;
      std::cout << "IO:" << std::endl;
      std::cout << "  io_frequency:    " << io_frequency_ << std::endl;
      std::cout << "  log_frequency:   " << log_frequency_ << std::endl;
      std::cout << "  output_filename: " << output_filename_ << std::endl;
      std::cout << "  enable_continuation_if_available: " << enable_continuation_if_available_ << std::endl;

      std::cout << std::endl;
      std::cout << "CONTROL FLAGS:" << std::endl;
      std::cout << "  enable_chromatin_brownian_motion: " << enable_chromatin_brownian_motion_ << std::endl;
      std::cout << "  enable_backbone_springs:          " << enable_backbone_springs_ << std::endl;
      std::cout << "  enable_backbone_collision:        " << enable_backbone_collision_ << std::endl;
      std::cout << "  enable_backbone_n_body_hydrodynamics:    " << enable_backbone_n_body_hydrodynamics_ << std::endl;
      std::cout << "  enable_crosslinkers:              " << enable_crosslinkers_ << std::endl;
      std::cout << "  enable_periphery_hydrodynamics:   " << enable_periphery_hydrodynamics_ << std::endl;
      std::cout << "  enable_periphery_collision:       " << enable_periphery_collision_ << std::endl;
      std::cout << "  enable_periphery_binding:         " << enable_periphery_binding_ << std::endl;
      std::cout << "  enable_active_euchromatin_forces: " << enable_active_euchromatin_forces_ << std::endl;

      if (enable_chromatin_brownian_motion_) {
        std::cout << std::endl;
        std::cout << "BROWNIAN MOTION:" << std::endl;
        std::cout << "  kt: " << brownian_kt_ << std::endl;
      }

      if (enable_backbone_springs_) {
        std::cout << std::endl;
        std::cout << "BACKBONE SPRINGS:" << std::endl;
        std::cout << "  spring_type:      " << backbone_spring_type_ << std::endl;
        std::cout << "  spring_constant:  " << backbone_spring_constant_ << std::endl;
        if (backbone_spring_type_ == BOND_TYPE::HARMONIC) {
          std::cout << "  spring_rest_length: " << backbone_spring_r0_ << std::endl;
        } else if (backbone_spring_type_ == BOND_TYPE::FENE) {
          std::cout << "  spring_rmax:        " << backbone_spring_r0_ << std::endl;
        } else if (backbone_spring_type_ == BOND_TYPE::FENEWCA) {
          std::cout << "  spring_rmax:        " << backbone_spring_r0_ << std::endl;
        }
      }

      if (enable_backbone_collision_) {
        std::cout << std::endl;
        std::cout << "BACKBONE COLLISION:" << std::endl;
        std::cout << "  backbone_collision_type: " << backbone_collision_type_ << std::endl;
        std::cout << "  excluded_volume_radius (if used): " << backbone_excluded_volume_radius_ << std::endl;
        if (backbone_collision_type_ == COLLISION_TYPE::HERTZIAN) {
          std::cout << "  youngs_modulus: " << backbone_youngs_modulus_ << std::endl;
          std::cout << "  poissons_ratio: " << backbone_poissons_ratio_ << std::endl;
        } else if (backbone_collision_type_ == COLLISION_TYPE::WCA) {
          std::cout << "  wca_epsilon: " << backbone_wca_epsilon_ << std::endl;
          std::cout << "  wca_sigma: " << backbone_wca_sigma_ << std::endl;
          std::cout << "  wca_cutoff: " << backbone_wca_cutoff_ << std::endl;
        }
      }

      if (enable_crosslinkers_) {
        std::cout << std::endl;
        std::cout << "CROSSLINKERS:" << std::endl;
        std::cout << "  spring_type: " << crosslinker_spring_type_ << std::endl;
        std::cout << "  kt: " << crosslinker_kt_ << std::endl;
        std::cout << "  spring_constant: " << crosslinker_spring_constant_ << std::endl;
        std::cout << "  r0: " << crosslinker_r0_ << std::endl;
        std::cout << "  left_binding_rate: " << crosslinker_left_binding_rate_ << std::endl;
        std::cout << "  right_binding_rate: " << crosslinker_right_binding_rate_ << std::endl;
        std::cout << "  left_unbinding_rate: " << crosslinker_left_unbinding_rate_ << std::endl;
        std::cout << "  right_unbinding_rate: " << crosslinker_right_unbinding_rate_ << std::endl;
        std::cout << "  cutoff_radius: " << crosslinker_cutoff_radius_ << std::endl;
      }

      if (enable_periphery_hydrodynamics_) {
        std::cout << std::endl;
        std::cout << "PERIPHERY HYDRODYNAMICS:" << std::endl;
        std::cout << "  check_maximum_periphery_overlap: " << check_maximum_periphery_overlap_ << std::endl;
        if (check_maximum_periphery_overlap_) {
          std::cout << "  maximum_allowed_periphery_overlap: " << maximum_allowed_periphery_overlap_ << std::endl;
        }
        if (periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_hydro_radius_ << std::endl;
        } else if (periphery_hydro_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_hydro_axis_radius1_ << std::endl;
          std::cout << "  axis_radius2: " << periphery_hydro_axis_radius2_ << std::endl;
          std::cout << "  axis_radius3: " << periphery_hydro_axis_radius3_ << std::endl;
        }
        if (periphery_hydro_quadrature_ == PERIPHERY_QUADRATURE::GAUSS_LEGENDRE) {
          std::cout << "  quadrature: GAUSS_LEGENDRE" << std::endl;
          std::cout << "  spectral_order: " << periphery_hydro_spectral_order_ << std::endl;
        } else if (periphery_hydro_quadrature_ == PERIPHERY_QUADRATURE::FROM_FILE) {
          std::cout << "  quadrature: FROM_FILE" << std::endl;
          std::cout << "  num_quadrature_points: " << periphery_hydro_num_quadrature_points_ << std::endl;
          std::cout << "  quadrature_points_filename: " << periphery_hydro_quadrature_points_filename_ << std::endl;
          std::cout << "  quadrature_weights_filename: " << periphery_hydro_quadrature_weights_filename_ << std::endl;
          std::cout << "  quadrature_normals_filename: " << periphery_hydro_quadrature_normals_filename_ << std::endl;
        }
      }

      if (enable_periphery_collision_) {
        std::cout << std::endl;
        std::cout << "PERIPHERY COLLISION:" << std::endl;
        if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_collision_radius_ << std::endl;
        } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_collision_axis_radius1_ << std::endl;
          std::cout << "  axis_radius2: " << periphery_collision_axis_radius2_ << std::endl;
          std::cout << "  axis_radius3: " << periphery_collision_axis_radius3_ << std::endl;
        }
        std::cout << "  collision_spring_constant: " << periphery_collision_spring_constant_ << std::endl;
        std::cout << "  periphery_collision_use_fast_approx: " << periphery_collision_use_fast_approx_ << std::endl;
        std::cout << "  shrink_periphery_over_time: " << shrink_periphery_over_time_ << std::endl;
        if (shrink_periphery_over_time_) {
          std::cout << "  SHRINKAGE:" << std::endl;
          std::cout << "    num_shrinkage_steps: " << periphery_collision_shrinkage_num_steps_ << std::endl;
          std::cout << "    scale_factor_before_shrinking: " << periphery_collision_scale_factor_before_shrinking_
                    << std::endl;
        }
      }

      if (enable_periphery_binding_) {
        std::cout << std::endl;
        std::cout << "PERIPHERY BINDING:" << std::endl;
        std::cout << "  binding_rate: " << periphery_binding_rate_ << std::endl;
        std::cout << "  unbinding_rate: " << periphery_unbinding_rate_ << std::endl;
        std::cout << "  spring_constant: " << periphery_spring_constant_ << std::endl;
        std::cout << "  r0: " << periphery_spring_r0_ << std::endl;
        if (periphery_bind_sites_type_ == PERIPHERY_BIND_SITES_TYPE::RANDOM) {
          std::cout << "  bind_sites_type: RANDOM" << std::endl;
          std::cout << "  num_bind_sites: " << periphery_num_bind_sites_ << std::endl;
        } else if (periphery_bind_sites_type_ == PERIPHERY_BIND_SITES_TYPE::FROM_FILE) {
          std::cout << "  bind_sites_type: FROM_FILE" << std::endl;
          std::cout << "  num_bind_sites: " << periphery_num_bind_sites_ << std::endl;
          std::cout << "  bind_site_locations_filename: " << periphery_bind_site_locations_filename_ << std::endl;
        }
      }

      if (enable_active_euchromatin_forces_) {
        std::cout << std::endl;
        std::cout << "ACTIVE EUCHROMATIN FORCES:" << std::endl;
        std::cout << "  force_sigma: " << active_euchromatin_force_sigma_ << std::endl;
        std::cout << "  kon: " << active_euchromatin_force_kon_ << std::endl;
        std::cout << "  koff: " << active_euchromatin_force_koff_ << std::endl;
      }

      std::cout << std::endl;
      std::cout << "NEIGHBOR LIST:" << std::endl;
      std::cout << "  skin_distance: " << skin_distance_ << std::endl;
      std::cout << "  force_neighborlist_update: " << force_neighborlist_update_ << std::endl;
      std::cout << "  force_neighborlist_update_nsteps: " << force_neighborlist_update_nsteps_ << std::endl;
      std::cout << "  print_neighborlist_statistics: " << print_neighborlist_statistics_ << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  template <typename FieldDataType, size_t field_size>
  void print_field(const stk::mesh::Field<FieldDataType> &field) {
    stk::mesh::BulkData &bulk_data = field.get_mesh();
    stk::mesh::Selector selector = stk::mesh::Selector(field);

    stk::mesh::EntityVector entities;
    stk::mesh::get_selected_entities(selector, bulk_data_ptr_->buckets(field.entity_rank()), entities);

    for (const stk::mesh::Entity &entity : entities) {
      const FieldDataType *field_data = stk::mesh::field_data(field, entity);
      std::cout << "Entity " << bulk_data.identifier(entity) << " field data: ";
      for (size_t i = 0; i < field_size; ++i) {
        std::cout << field_data[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  void setup_mundy_io() {
    // IO fixed parameters
    auto fixed_params_iobroker =
        Teuchos::ParameterList()
            .set("enabled_io_parts",
                 mundy::core::make_string_array("E", "H", "BS", "EESPRINGS", "EHSPRINGS", "HHSPRINGS", "LEFT_HP1",
                                                "DOUBLY_HP1_H", "DOUBLY_HP1_BS"))
            .set("enabled_io_fields_node_rank",
                 mundy::core::make_string_array("NODE_VELOCITY", "NODE_FORCE", "NODE_RNG_COUNTER"))
            .set("enabled_io_fields_element_rank",
                 mundy::core::make_string_array(
                     "ELEMENT_RADIUS", "ELEMENT_RNG_COUNTER", "ELEMENT_REALIZED_BINDING_RATES",
                     "ELEMENT_REALIZED_UNBINDING_RATES", "ELEMENT_PERFORM_STATE_CHANGE", "EUCHROMATIN_STATE",
                     "EUCHROMATIN_PERFORM_STATE_CHANGE", "EUCHROMATIN_STATE_CHANGE_NEXT_TIME",
                     "EUCHROMATIN_STATE_CHANGE_ELAPSED_TIME", "ELEMENT_CHAINID", "REQUIRES_ENDPOINT_CORRECTION"))
            .set("coordinate_field_name", "NODE_COORDS")
            .set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES")
            .set("exodus_database_output_filename", output_filename_)
            .set("parallel_io_mode", "hdf5")
            .set("database_purpose", "restart");

    // Check if we are continuing a previous sim or if we are initializing from a file.
    // In either case, we must perform a restart.
    if (enable_continuation_if_available_) {
      // The filename pattern is stem + ".e-s." + timestep_number
      // We want to determine if if such a file exist, and if so, the file with the largest timestep number
      auto find_file_with_largest_timestep_number = [](const std::string &stem) {
        // Pattern to match files like stem.e-s.*
        std::string pattern = stem + R"(\.e-s\.(\d+))";
        std::regex regex_pattern(pattern);
        int largest_number = -1;
        std::string largest_file;

        // Iterate through the directory of the stem (or the current directory if the stem doesn't provide a filepath)
        std::filesystem::path filepath(stem);
        if (!std::filesystem::exists(filepath)) {
          filepath = std::filesystem::current_path();
        }
        for (const auto &entry : std::filesystem::directory_iterator(filepath)) {
          std::string filename = entry.path().filename().string();
          std::smatch match;
          if (std::regex_match(filename, match, regex_pattern)) {
            int number = std::stoi(match[1].str());
            if (number > largest_number) {
              largest_number = number;
              largest_file = entry.path().string();
            }
          }
        }

        const double file_found = largest_number != -1;
        return std::make_tuple(file_found, largest_file, largest_number);
      };

      auto [file_found, restart_filename, largest_number] = find_file_with_largest_timestep_number(output_filename_);
      if (file_found) {
        std::cout << "Restarting from file: " << restart_filename << " at step " << largest_number << std::endl;
        fixed_params_iobroker.set("exodus_database_input_filename", restart_filename);
        fixed_params_iobroker.set("enable_restart", "true");
        restart_performed_ = true;
        restart_timestep_index_ = largest_number;
      }
    }

    // Continuing a previous simulation takes priority over initializing from a file.
    // Initialization should have already been performed in the previous simulation.
    if (initialization_type_ == INITIALIZATION_TYPE::FROM_EXO && !restart_performed_) {
      fixed_params_iobroker.set("exodus_database_input_filename", initialize_from_exo_filename_);
      fixed_params_iobroker.set("enable_restart", "false");
    }

    io_broker_ptr_ = mundy::io::IOBroker::create_new_instance(bulk_data_ptr_.get(), fixed_params_iobroker);
  }

  void build_our_mesh_and_method_instances() {
    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshReqs>(MPI_COMM_WORLD);
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements to the sphere part. These are requirements that exceed those of the
    // enabled methods and allow us to extend the functionality offered natively by Mundy.

    // Spheres need to be modified to contain the subparts for E, H, and BindSite.
    auto custom_sphere_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_sphere_part_reqs->add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_VELOCITY_HYDRO", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_VELOCITY_HYDRO_OLD", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<unsigned>("NODE_RNG_COUNTER", node_rank_, 1, 1)
        .add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", node_rank_, 3, 1)
        .add_subpart_reqs("E", stk::topology::PARTICLE)
        .add_subpart_reqs("H", stk::topology::PARTICLE)
        .add_subpart_reqs("BS", stk::topology::PARTICLE);
    mundy::shapes::Spheres::add_and_sync_part_reqs(custom_sphere_part_reqs);
    mesh_reqs_ptr_->sync(mundy::shapes::Spheres::get_mesh_requirements());

    // Add a chain ID to the E and H entities.
    auto custom_e_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    auto custom_h_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_e_part_reqs->set_part_name("E").add_field_reqs<unsigned>("ELEMENT_CHAINID", element_rank_, 1, 1);
    custom_h_part_reqs->set_part_name("H").add_field_reqs<unsigned>("ELEMENT_CHAINID", element_rank_, 1, 1);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_e_part_reqs);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_h_part_reqs);

    // HP1 needs to be added to the mesh. This includes the subparts for the states of HP1. It will be added to the
    // SpherocylinderSegment part the same was as StickySettings.
    auto custom_hp1_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_part_reqs->set_part_name("HP1S")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ELEMENT_REALIZED_UNBINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<double>("ELEMENT_REALIZED_BINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<unsigned>("ELEMENT_RNG_COUNTER", element_rank_, 1, 1)
        .add_field_reqs<unsigned>("ELEMENT_PERFORM_STATE_CHANGE", element_rank_, 1, 1)
        .add_field_reqs<unsigned>("ELEMENT_CHAINID", element_rank_, 1, 1)
        .add_subpart_reqs("LEFT_HP1", stk::topology::BEAM_2)
        .add_subpart_reqs("DOUBLY_HP1_H", stk::topology::BEAM_2)
        .add_subpart_reqs("DOUBLY_HP1_BS", stk::topology::BEAM_2);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_hp1_part_reqs);

    // Create the backbone segments.
    auto custom_backbone_segments_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_backbone_segments_part_reqs->set_part_name("BACKBONE_SEGMENTS")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<unsigned>("ELEMENT_CHAINID", element_rank_, 1, 1)
        .add_field_reqs<double>("REQUIRES_ENDPOINT_CORRECTION", element_rank_, 1, 1)
        .add_subpart_reqs("EESPRINGS", stk::topology::BEAM_2)
        .add_subpart_reqs("EHSPRINGS", stk::topology::BEAM_2)
        .add_subpart_reqs("HHSPRINGS", stk::topology::BEAM_2);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_backbone_segments_part_reqs);

    // Create the force-dipole information on just the EESPRINGS (euchromatin springs), in active and inactive states
    auto custom_euchromatin_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_euchromatin_part_reqs->set_part_name("EESPRINGS")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<unsigned>("ELEMENT_RNG_COUNTER", element_rank_, 1, 1)
        .add_field_reqs<unsigned>("EUCHROMATIN_STATE", element_rank_, 1, 1)
        .add_field_reqs<unsigned>("EUCHROMATIN_PERFORM_STATE_CHANGE", element_rank_, 1, 1)
        .add_field_reqs<double>("EUCHROMATIN_STATE_CHANGE_NEXT_TIME", element_rank_, 1, 1)
        .add_field_reqs<double>("EUCHROMATIN_STATE_CHANGE_ELAPSED_TIME", element_rank_, 1, 1);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_euchromatin_part_reqs);

    // Create the generalized interaction entities that connect HP1 and (H)eterochromatin
    //   This entity "knows" how to compute the binding probability between a crosslinker and a H and how to
    //   perform binding between a crosslinker and a H. It is a constraint rank entitiy because it must connect
    //   element rank entities.
    auto custom_hp1_h_genx_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_h_genx_part_reqs->set_part_name("HP1_H_NEIGHBOR_GENXS")
        .set_part_rank(constraint_rank_)
        .add_field_reqs<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_, 1, 1)
        .add_field_reqs<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_, 1, 1);
    mundy::linkers::NeighborLinkers::add_and_sync_subpart_reqs(custom_hp1_h_genx_part_reqs);
    mesh_reqs_ptr_->sync(mundy::linkers::NeighborLinkers::get_mesh_requirements());
    // Create the same type of entity to the binding sites on the periphery
    auto custom_hp1_bs_genx_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_bs_genx_part_reqs->set_part_name("HP1_BS_NEIGHBOR_GENXS")
        .set_part_rank(constraint_rank_)
        .add_field_reqs<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_, 1, 1)
        .add_field_reqs<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_, 1, 1);
    mundy::linkers::NeighborLinkers::add_and_sync_subpart_reqs(custom_hp1_bs_genx_part_reqs);
    mesh_reqs_ptr_->sync(mundy::linkers::NeighborLinkers::get_mesh_requirements());

    // Add the custom neighbor list implementation to the axis-aligned bounding box. In theory, we only care if the
    // corners of the AABB move more than skin distance/2. Set the AABB field to have an additional state that it keeps
    // track of at the end so that we can do dr calculations trivially. This create the two element aabb fields, but
    // also adds a second entire set of fields to the mesh.
    //
    // We also need an accumulator for summing up the total distance traveled by the AABB corners.
#pragma TODO right now this puts the ELEMENT_AABB on the entire mesh(elements)
    mesh_reqs_ptr_->add_field_reqs<double>("ELEMENT_AABB", element_rank_, 6, 2);
    auto custom_sphere_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_sphere_aabb_accumulator_reqs->set_part_name("SPHERES")
        .set_part_topology(stk::topology::PARTICLE)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    auto custom_backbone_segment_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_backbone_segment_aabb_accumulator_reqs->set_part_name("BACKBONE_SEGMENTS")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    auto custom_hp1_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_hp1_aabb_accumulator_reqs->set_part_name("HP1S")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_sphere_aabb_accumulator_reqs);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_backbone_segment_aabb_accumulator_reqs);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_hp1_aabb_accumulator_reqs);

    // Setup our fixed parameters for any of methods that we intend to use
    // When we eventually switch to the configurator, these individual fixed params will become sublists within a single
    // master parameter list. Note, sublist will return a reference to the sublist with the given name.
    //
    // Compute constraint (bonded) forces for the the BACKBONE_SEGMENTS and HP1S parts
    if (backbone_spring_type_ == BOND_TYPE::HARMONIC && crosslinker_spring_type_ == BOND_TYPE::HARMONIC) {
      compute_constraint_forcing_fixed_params_ =
          Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"));
      compute_constraint_forcing_fixed_params_.sublist("HOOKEAN_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS", "HP1S"));
    } else if (backbone_spring_type_ == BOND_TYPE::FENE && crosslinker_spring_type_ == BOND_TYPE::HARMONIC) {
      compute_constraint_forcing_fixed_params_ = Teuchos::ParameterList().set(
          "enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS", "FENE_SPRINGS"));
      compute_constraint_forcing_fixed_params_.sublist("HOOKEAN_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("HP1S"));
      compute_constraint_forcing_fixed_params_.sublist("FENE_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));
    } else if (backbone_spring_type_ == BOND_TYPE::HARMONIC && crosslinker_spring_type_ == BOND_TYPE::FENE) {
      compute_constraint_forcing_fixed_params_ = Teuchos::ParameterList().set(
          "enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS", "FENE_SPRINGS"));
      compute_constraint_forcing_fixed_params_.sublist("HOOKEAN_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));
      compute_constraint_forcing_fixed_params_.sublist("FENE_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("HP1S"));
    } else if (backbone_spring_type_ == BOND_TYPE::FENE && crosslinker_spring_type_ == BOND_TYPE::FENE) {
      compute_constraint_forcing_fixed_params_ =
          Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("FENE_SPRINGS"));
      compute_constraint_forcing_fixed_params_.sublist("FENE_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS", "HP1S"));
    } else if (backbone_spring_type_ == BOND_TYPE::FENEWCA && crosslinker_spring_type_ == BOND_TYPE::FENEWCA) {
      compute_constraint_forcing_fixed_params_ =
          Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("FENEWCA_SPRINGS"));
      compute_constraint_forcing_fixed_params_.sublist("FENEWCA_SPRINGS")
          .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS", "HP1S"));
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Something went wrong with the combination of backbone and crosslinker spring "
                                      "types, check the code for more details."));
    }

    // Compute the minimum distance for the SCS-SCS, HP1-H, HP1-BS interactions (SCS-SCS, S-SCS, S-SCS)
    // Try to be as explicit as possible with the parts that are associated with each of the interactions.
    compute_ssd_and_cn_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER",
                                                               "SPHERE_SPHEROCYLINDER_SEGMENT_LINKER"));
    compute_ssd_and_cn_fixed_params_.sublist("SPHERE_SPHEROCYLINDER_SEGMENT_LINKER")
        .set("valid_entity_part_names", mundy::core::make_string_array("HP1_H_NEIGHBOR_GENXS", "HP1_BS_NEIGHBOR_GENXS"))
        .set("valid_sphere_part_names", mundy::core::make_string_array("H", "BS"))
        .set("valid_spherocylinder_segment_part_names", mundy::core::make_string_array("HP1S"));
    compute_ssd_and_cn_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER")
        .set("valid_entity_part_names", mundy::core::make_string_array("BACKBONE_BACKBONE_NEIGHBOR_GENXS"))
        .set("valid_spherocylinder_segment_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));

    // Set up the AABB for the system
    compute_aabb_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE", "SPHEROCYLINDER_SEGMENT"));
    compute_aabb_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT")
        .set("valid_entity_part_names", mundy::core::make_string_array("HP1S", "BACKBONE_SEGMENTS"));

    // Generate the GENX neighbor linkers between spherocylinder segments
    generate_scs_scs_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("BACKBONE_BACKBONE_NEIGHBOR_GENXS"));
    generate_scs_scs_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));

    // Generate the GENX neighbor linkers between HP1 and H
    generate_hp1_h_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("HP1_H_NEIGHBOR_GENXS"));
    generate_hp1_h_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("HP1S")))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("H"));

    // Generate the GENX neighbor linkers between HP1 and BS
    generate_hp1_bs_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("HP1_BS_NEIGHBOR_GENXS"));
    generate_hp1_bs_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("HP1S")))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("BS"));

    // Evaluate the scs-scs contacts (hertzian or WCA)
    if (backbone_collision_type_ == COLLISION_TYPE::HERTZIAN) {
      evaluate_linker_potentials_fixed_params_ = Teuchos::ParameterList().set(
          "enabled_kernel_names",
          mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));
      evaluate_linker_potentials_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT")
          .set("valid_spherocylinder_segment_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));
    } else if (backbone_collision_type_ == COLLISION_TYPE::WCA) {
      evaluate_linker_potentials_fixed_params_ = Teuchos::ParameterList().set(
          "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_WCA"));
      evaluate_linker_potentials_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_WCA")
          .set("valid_spherocylinder_segment_part_names", mundy::core::make_string_array("BACKBONE_SEGMENTS"));
    }

    // Reduce the forces on the spherocylinder segments
    linker_potential_force_reduction_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"))
            .set("name_of_linker_part_to_reduce_over", "BACKBONE_BACKBONE_NEIGHBOR_GENXS");

    // Destroy the distant neighbors over time
    destroy_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");
    destroy_neighbor_linkers_fixed_params_.sublist("DESTROY_DISTANT_NEIGHBORS")
        .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
        .set("valid_connected_source_and_target_part_names",
             mundy::core::make_string_array(std::string("SPHEROCYLINDER_SEGMENTS"), std::string("HP1S")));

    // Destroy bound linkers to prevent pathological behavior along a chain
    destroy_bound_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_BOUND_NEIGHBORS");

    // Synchronize (merge and rectify differences) the requirements for each method based on the fixed parameters.
    // For now, we will directly use the types that each method corresponds to. The configurator will
    // fetch the static members of these methods using the configurable method factory.
    mesh_reqs_ptr_->sync(
        mundy::constraints::ComputeConstraintForcing::get_mesh_requirements(compute_constraint_forcing_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_scs_scs_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(generate_hp1_h_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(generate_hp1_bs_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::LinkerPotentialForceReduction::get_mesh_requirements(
        linker_potential_force_reduction_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_bound_neighbor_linkers_fixed_params_));

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->use_simple_fields();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDS");

    // IO is the only method that must come after the mesh is declared but before the mesh is committed.
    setup_mundy_io();
    if (!restart_performed_) {
      // Commit the mesh
      meta_data_ptr_->commit();
    } else {
      // The mesh better be committed already
      MUNDY_THROW_REQUIRE(meta_data_ptr_->is_commit(), std::runtime_error,
                          "The restart should have already committed the mesh. Something went wrong.");
    }
  }

  template <typename FieldType>
  stk::mesh::Field<FieldType> *fetch_field(const std::string &field_name, stk::topology::rank_t rank) {
    auto field_ptr = meta_data_ptr_->get_field<FieldType>(rank, field_name);
    MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                        std::string("Field ") + field_name + " not found in the mesh meta data.");
    return field_ptr;
  }

  stk::mesh::Part *fetch_part(const std::string &part_name) {
    auto part_ptr = meta_data_ptr_->get_part(part_name);
    MUNDY_THROW_REQUIRE(part_ptr != nullptr, std::invalid_argument,
                        std::string("Part ") + part_name + " not found in the mesh meta data.");
    return part_ptr;
  }

  void fetch_fields_and_parts() {
    // Fetch the fields
    node_coord_field_ptr_ = fetch_field<double>("NODE_COORDS", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_velocity_hydro_field_ptr_ = fetch_field<double>("NODE_VELOCITY_HYDRO", node_rank_);
    node_velocity_hydro_old_field_ptr_ = fetch_field<double>("NODE_VELOCITY_HYDRO_OLD", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_rng_field_ptr_ = fetch_field<unsigned>("NODE_RNG_COUNTER", node_rank_);

    element_rng_field_ptr_ = fetch_field<unsigned>("ELEMENT_RNG_COUNTER", element_rank_);
    element_spring_constant_field_ptr_ = fetch_field<double>("ELEMENT_SPRING_CONSTANT", element_rank_);
    element_spring_r0_field_ptr_ = fetch_field<double>("ELEMENT_SPRING_R0", element_rank_);
    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    if (backbone_collision_type_ == COLLISION_TYPE::HERTZIAN) {
      element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
      element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    }
    element_aabb_field_ptr_ = fetch_field<double>("ELEMENT_AABB", element_rank_);
    element_corner_displacement_field_ptr_ = fetch_field<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_);
    element_binding_rates_field_ptr_ = fetch_field<double>("ELEMENT_REALIZED_BINDING_RATES", element_rank_);
    element_unbinding_rates_field_ptr_ = fetch_field<double>("ELEMENT_REALIZED_UNBINDING_RATES", element_rank_);
    element_perform_state_change_field_ptr_ = fetch_field<unsigned>("ELEMENT_PERFORM_STATE_CHANGE", element_rank_);
    element_chainid_field_ptr_ = fetch_field<unsigned>("ELEMENT_CHAINID", element_rank_);
    element_requires_endpoint_correction_field_ptr_ =
        fetch_field<double>("REQUIRES_ENDPOINT_CORRECTION", element_rank_);

    euchromatin_state_field_ptr_ = fetch_field<unsigned>("EUCHROMATIN_STATE", element_rank_);
    euchromatin_perform_state_change_field_ptr_ =
        fetch_field<unsigned>("EUCHROMATIN_PERFORM_STATE_CHANGE", element_rank_);
    euchromatin_state_change_next_time_field_ptr_ =
        fetch_field<double>("EUCHROMATIN_STATE_CHANGE_NEXT_TIME", element_rank_);
    euchromatin_state_change_elapsed_time_field_ptr_ =
        fetch_field<double>("EUCHROMATIN_STATE_CHANGE_ELAPSED_TIME", element_rank_);

    constraint_potential_force_field_ptr_ = fetch_field<double>("LINKER_POTENTIAL_FORCE", constraint_rank_);
    constraint_state_change_rate_field_ptr_ =
        fetch_field<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_);
    constraint_perform_state_change_field_ptr_ =
        fetch_field<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_);
    constraint_linked_entities_field_ptr_ =
        fetch_field<mundy::linkers::LinkedEntitiesFieldType::value_type>("LINKED_NEIGHBOR_ENTITIES", constraint_rank_);
    constraint_linked_entity_owners_field_ptr_ = fetch_field<int>("LINKED_NEIGHBOR_ENTITY_OWNERS", constraint_rank_);

    // Fetch the parts
    spheres_part_ptr_ = fetch_part("SPHERES");
    e_part_ptr_ = fetch_part("E");
    h_part_ptr_ = fetch_part("H");
    bs_part_ptr_ = fetch_part("BS");

    hp1_part_ptr_ = fetch_part("HP1S");
    left_hp1_part_ptr_ = fetch_part("LEFT_HP1");
    doubly_hp1_h_part_ptr_ = fetch_part("DOUBLY_HP1_H");
    doubly_hp1_bs_part_ptr_ = fetch_part("DOUBLY_HP1_BS");

    backbone_segments_part_ptr_ = fetch_part("BACKBONE_SEGMENTS");
    ee_springs_part_ptr_ = fetch_part("EESPRINGS");
    eh_springs_part_ptr_ = fetch_part("EHSPRINGS");
    hh_springs_part_ptr_ = fetch_part("HHSPRINGS");

    backbone_backbone_neighbor_genx_part_ptr_ = fetch_part("BACKBONE_BACKBONE_NEIGHBOR_GENXS");
    hp1_h_neighbor_genx_part_ptr_ = fetch_part("HP1_H_NEIGHBOR_GENXS");
    hp1_bs_neighbor_genx_part_ptr_ = fetch_part("HP1_BS_NEIGHBOR_GENXS");
  }

  void instantiate_metamethods() {
    // Create the non-custom MetaMethods
    // MetaMethodExecutionInterface

    // MetaMethodSubsetExecutionInterface
    compute_constraint_forcing_ptr_ = mundy::constraints::ComputeConstraintForcing::create_new_instance(
        bulk_data_ptr_.get(), compute_constraint_forcing_fixed_params_);
    compute_ssd_and_cn_ptr_ = mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::create_new_instance(
        bulk_data_ptr_.get(), compute_ssd_and_cn_fixed_params_);
    compute_aabb_ptr_ =
        mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr_.get(), compute_aabb_fixed_params_);
    evaluate_linker_potentials_ptr_ = mundy::linkers::EvaluateLinkerPotentials::create_new_instance(
        bulk_data_ptr_.get(), evaluate_linker_potentials_fixed_params_);
    linker_potential_force_reduction_ptr_ = mundy::linkers::LinkerPotentialForceReduction::create_new_instance(
        bulk_data_ptr_.get(), linker_potential_force_reduction_fixed_params_);
    destroy_neighbor_linkers_ptr_ = mundy::linkers::DestroyNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), destroy_neighbor_linkers_fixed_params_);
    destroy_bound_neighbor_linkers_ptr_ = mundy::linkers::DestroyNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), destroy_bound_neighbor_linkers_fixed_params_);

    // MetaMethodPairwiseSubsetExecutionInterface
    generate_scs_scs_genx_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_scs_scs_neighbor_linkers_fixed_params_);
    generate_hp1_h_genx_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_hp1_h_neighbor_linkers_fixed_params_);
    generate_hp1_bs_genx_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_hp1_bs_neighbor_linkers_fixed_params_);
  }

  void set_mutable_parameters() {
    // ComputeAABB mutable parameters
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", skin_distance_);
    compute_aabb_ptr_->set_mutable_params(compute_aabb_mutable_params);

    // Generate the GENX neighbor linkers between spherocylinder segments mutable params
    auto generate_scs_scs_genx_mutable_params = Teuchos::ParameterList();
    generate_scs_scs_genx_mutable_params.sublist("STK_SEARCH").set("enforce_symmetry", true);
    generate_scs_scs_genx_ptr_->set_mutable_params(generate_scs_scs_genx_mutable_params);

    // Generate the GENX neighbor linkers between HP1 and H mutable params
    auto generate_hp1_h_genx_mutable_params = Teuchos::ParameterList();
    generate_hp1_h_genx_mutable_params.sublist("STK_SEARCH").set("enforce_symmetry", false);
    generate_hp1_h_genx_ptr_->set_mutable_params(generate_hp1_h_genx_mutable_params);

    // Generate the GENX neighbor linkers between HP1 and BS mutable params
    auto generate_hp1_bs_genx_mutable_params = Teuchos::ParameterList();
    generate_hp1_bs_genx_mutable_params.sublist("STK_SEARCH").set("enforce_symmetry", false);
    generate_hp1_bs_genx_ptr_->set_mutable_params(generate_hp1_bs_genx_mutable_params);

    // If we are using WCA potentials, set the mutable parameters that are attached to the class here
    if (backbone_collision_type_ == COLLISION_TYPE::WCA) {
      auto evaluate_linker_potentials_mutable_params = Teuchos::ParameterList();
      evaluate_linker_potentials_mutable_params.sublist("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_WCA")
          .set("epsilon", backbone_wca_epsilon_)
          .set("sigma", backbone_wca_sigma_)
          .set("lj_cutoff", backbone_wca_cutoff_);
      evaluate_linker_potentials_ptr_->set_mutable_params(evaluate_linker_potentials_mutable_params);
    }
  }

  void ghost_linked_entities() {
    bulk_data_ptr_->modification_begin();
    const stk::mesh::Selector linker_parts_selector = stk::mesh::selectUnion(stk::mesh::ConstPartVector{
        backbone_backbone_neighbor_genx_part_ptr_, hp1_h_neighbor_genx_part_ptr_, hp1_bs_neighbor_genx_part_ptr_});
    mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr_, *constraint_linked_entities_field_ptr_,
                                                 *constraint_linked_entity_owners_field_ptr_, linker_parts_selector);
    bulk_data_ptr_->modification_end();
  }

  void loadbalance() {
    stk::balance::balanceStkMesh(balance_settings_, *bulk_data_ptr_);
    ghost_linked_entities();
  }

  void rotate_field_states() {
    bulk_data_ptr_->update_field_data_states();
  }

  // Create the chromatin backbone and HP1 crosslinkers
  void create_chromatin_backbone_and_hp1() {
    // Calculate some constants, like the total number of spheres or segments per chromosome
    const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                               num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
    const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
    const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
    const size_t num_spheres_per_chromosome = num_nodes_per_chromosome;
    const size_t num_segments_per_chromosome = num_nodes_per_chromosome - 1;
    const bool enable_backbone_collision = enable_backbone_collision_;
    const bool enable_backbone_springs = enable_backbone_springs_;
    const bool enable_crosslinkers = enable_crosslinkers_;
    const size_t num_elements_created_per_chromosome =
        num_spheres_per_chromosome +
        (enable_backbone_springs || enable_backbone_collision) * num_segments_per_chromosome +
        enable_crosslinkers * num_heterochromatin_spheres;

    std::cout << "Per chromosome:\n";
    std::cout << "num_heterochromatin_spheres: " << num_heterochromatin_spheres << std::endl;
    std::cout << "num_euchromatin_spheres:     " << num_euchromatin_spheres << std::endl;
    std::cout << "num_nodes_per_chromosome:    " << num_nodes_per_chromosome << std::endl;
    std::cout << "num_spheres_per_chromosome:  " << num_spheres_per_chromosome << std::endl;
    std::cout << "num_segments_per_chromosome: " << num_segments_per_chromosome << std::endl;

    bulk_data_ptr_->modification_begin();

    // Rank 0: Declare N chromatin chains randomly in space
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        std::cout << "Creating chromosome " << j << std::endl;

        // Some notes on what will and will not be created
        // There are three enables that matter: enble_crosslinkers, enable_backbone_springs, and
        // enable_backbone_collision
        //   The backbone spheres will always be created.
        //   The HP1 crosslinkers will be created if enable_crosslinkers_ is true.
        //   The backbone segments will be created if EITHER enable_backbone_springs_ or enable_backbone_collision_ is
        //   true, but the segment will only be placed in the corresponding part if it is enabled.

        // Figure out the starting indices of the nodes and elements
        const size_t start_node_id = num_nodes_per_chromosome * j + 1u;
        const size_t start_element_id = num_elements_created_per_chromosome * j + 1u;

        // Helper functions for getting the IDs of various objects
        auto get_node_id = [start_node_id](const size_t &seq_node_index) { return start_node_id + seq_node_index; };

        auto get_sphere_id = [start_element_id](const size_t &seq_sphere_index) {
          return start_element_id + seq_sphere_index;
        };

        auto get_segment_id = [start_element_id, num_spheres_per_chromosome](const size_t &seq_segment_index) {
          return start_element_id + num_spheres_per_chromosome + seq_segment_index;
        };

        auto get_crosslinker_id = [start_element_id, num_spheres_per_chromosome, num_segments_per_chromosome,
                                   enable_backbone_collision,
                                   enable_backbone_springs](const size_t &seq_crosslinker_index) {
          return start_element_id + num_spheres_per_chromosome +
                 (enable_backbone_springs || enable_backbone_collision) * num_segments_per_chromosome +
                 seq_crosslinker_index;
        };

        auto is_first_or_last_element_in_chain = [start_element_id,
                                                  num_segments_per_chromosome](const size_t &seq_segment_index) {
          return seq_segment_index == 0 || seq_segment_index == num_segments_per_chromosome - 1;  // 0-based indexing
        };

        // Try to use modulo math to determine region
        const size_t num_heterochromatin_per_repeat = num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_per_repeat = num_euchromatin_per_repeat_;
        auto get_region_by_id = [num_heterochromatin_per_repeat,
                                 num_euchromatin_per_repeat](const size_t &seq_sphere_id) {
          auto local_idx = seq_sphere_id % (num_heterochromatin_per_repeat + num_euchromatin_per_repeat);
          return local_idx < num_heterochromatin_per_repeat ? std::string("H") : std::string("E");
        };

        // Show what a single chromatin chain would be in terms of membership
        std::cout << "Regional map:" << std::endl;
        for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
          std::cout << get_region_by_id(i);
        }
        std::cout << std::endl;

        // Temporary/scratch variables
        stk::mesh::PartVector empty;

        // Logically, it makes the most sense to march down the segments in a single chromosome, adjusting their part
        // membership as we go. Do this across the elements of the chromatin backbone.
        // Initialize the backbone such that we have different sphere types.
        //  E : euchromatin spheres
        //  H : heterochromatin spheres
        // ---: backbone springs (EE, EH, or HH depending on attached spheres)
        //
        //  H---H---E---E---E---E---E---E---H---H
        //
        std::cout << "  Building backbone segments" << std::endl;
        for (size_t segment_local_idx = 0; segment_local_idx < num_segments_per_chromosome; segment_local_idx++) {
          // Keep track of the vertex IDs for part memebership (local index into array)
          const size_t vertex_left_idx = segment_local_idx;
          const size_t vertex_right_idx = segment_local_idx + 1;
          // Process the nodes for this segment
          stk::mesh::EntityId left_node_id = get_node_id(segment_local_idx);
          stk::mesh::EntityId right_node_id = get_node_id(segment_local_idx + 1);

          stk::mesh::Entity left_node = bulk_data_ptr_->get_entity(node_rank_, left_node_id);
          stk::mesh::Entity right_node = bulk_data_ptr_->get_entity(node_rank_, right_node_id);
          if (!bulk_data_ptr_->is_valid(left_node)) {
            left_node = bulk_data_ptr_->declare_node(left_node_id, empty);
          }
          if (!bulk_data_ptr_->is_valid(right_node)) {
            right_node = bulk_data_ptr_->declare_node(right_node_id, empty);
          }

          // Each node is attached to a sphere that is (H)eterochromatin, (E)uchromatin, or (BS)BindingSite
          stk::mesh::EntityId left_sphere_id = get_sphere_id(segment_local_idx);
          stk::mesh::EntityId right_sphere_id = get_sphere_id(segment_local_idx + 1);
          stk::mesh::Entity left_sphere = bulk_data_ptr_->get_entity(element_rank_, left_sphere_id);
          stk::mesh::Entity right_sphere = bulk_data_ptr_->get_entity(element_rank_, right_sphere_id);
          if (!bulk_data_ptr_->is_valid(left_sphere)) {
            // Figure out the part we belong to
            stk::mesh::PartVector pvector;
            if (get_region_by_id(vertex_left_idx) == "H") {
              pvector.push_back(h_part_ptr_);
            } else if (get_region_by_id(vertex_left_idx) == "E") {
              pvector.push_back(e_part_ptr_);
            }
            // Declare the sphere and connect to it's node
            left_sphere = bulk_data_ptr_->declare_element(left_sphere_id, pvector);
            bulk_data_ptr_->declare_relation(left_sphere, left_node, 0);
            // Assign the chainID
            stk::mesh::field_data(*element_chainid_field_ptr_, left_sphere)[0] = j;
          }
          if (!bulk_data_ptr_->is_valid(right_sphere)) {
            // Figure out the part we belong to
            stk::mesh::PartVector pvector;
            if (get_region_by_id(vertex_right_idx) == "H") {
              pvector.push_back(h_part_ptr_);
            } else if (get_region_by_id(vertex_right_idx) == "E") {
              pvector.push_back(e_part_ptr_);
            }
            // Declare the sphere and connect to it's node
            right_sphere = bulk_data_ptr_->declare_element(right_sphere_id, pvector);
            bulk_data_ptr_->declare_relation(right_sphere, right_node, 0);
            // Assign the chainID
            stk::mesh::field_data(*element_chainid_field_ptr_, right_sphere)[0] = j;
          }

          // Figure out how to do the spherocylinder segments along the edges now
          if (enable_backbone_springs || enable_backbone_collision) {
            stk::mesh::Entity segment = bulk_data_ptr_->get_entity(element_rank_, get_segment_id(segment_local_idx));
            if (!bulk_data_ptr_->is_valid(segment)) {
              stk::mesh::PartVector pvector;
              pvector.push_back(backbone_segments_part_ptr_);
              if (enable_backbone_springs) {
                if (get_region_by_id(vertex_left_idx) == "E" && get_region_by_id(vertex_right_idx) == "E") {
                  pvector.push_back(ee_springs_part_ptr_);
                } else if (get_region_by_id(vertex_left_idx) == "E" && get_region_by_id(vertex_right_idx) == "H") {
                  pvector.push_back(eh_springs_part_ptr_);
                } else if (get_region_by_id(vertex_left_idx) == "H" && get_region_by_id(vertex_right_idx) == "E") {
                  pvector.push_back(eh_springs_part_ptr_);
                } else if (get_region_by_id(vertex_left_idx) == "H" && get_region_by_id(vertex_right_idx) == "H") {
                  pvector.push_back(hh_springs_part_ptr_);
                }
              }
              segment = bulk_data_ptr_->declare_element(get_segment_id(segment_local_idx), pvector);
              bulk_data_ptr_->declare_relation(segment, left_node, 0);
              bulk_data_ptr_->declare_relation(segment, right_node, 1);
              // Assign the chainID
              stk::mesh::field_data(*element_chainid_field_ptr_, segment)[0] = j;

              // Assign if the segment is an endpoint or not
              stk::mesh::field_data(*element_requires_endpoint_correction_field_ptr_, segment)[0] =
                  is_first_or_last_element_in_chain(segment_local_idx);
            }
          }
        }
        std::cout << "  ...finished building backbone segments" << std::endl;

        // Declare the crosslinkers along the backbone
        // Every sphere gets a left bound crosslinker
        //  E : euchromatin spheres
        //  H : heterochromatin spheres
        //  | : crosslinkers
        // ---: backbone springs
        //
        //  |   |                           |   |
        //  H---H---E---E---E---E---E---E---H---H

        // March down the chain of spheres, adding crosslinkers as we go. We just want to add to the heterochromatin
        // spheres, and so keep track of a running hp1_sphere_index.
        if (enable_crosslinkers_) {
          std::cout << "  Building hp1 segments" << std::endl;
          size_t hp1_sphere_index = 0;
          for (size_t sphere_local_idx = 0; sphere_local_idx < num_spheres_per_chromosome; sphere_local_idx++) {
            stk::mesh::Entity sphere_node = bulk_data_ptr_->get_entity(node_rank_, get_node_id(sphere_local_idx));
            MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(sphere_node), std::invalid_argument,
                               fmt::format("Node {} is not valid", sphere_local_idx));

            // Check if we are a heterochromatin sphere
            if (get_region_by_id(sphere_local_idx) == "H") {
              // Bind left and right nodes to the same node to start simulation (everybody is left bound)
              // Create the HP1 crosslinker
              auto left_bound_hp1_part_vector = stk::mesh::PartVector{left_hp1_part_ptr_};
              stk::mesh::EntityId hp1_crosslinker_id = get_crosslinker_id(hp1_sphere_index);
              stk::mesh::Entity hp1_crosslinker =
                  bulk_data_ptr_->declare_element(hp1_crosslinker_id, left_bound_hp1_part_vector);
              stk::mesh::Permutation invalid_perm = stk::mesh::Permutation::INVALID_PERMUTATION;
              bulk_data_ptr_->declare_relation(hp1_crosslinker, sphere_node, 0, invalid_perm);
              bulk_data_ptr_->declare_relation(hp1_crosslinker, sphere_node, 1, invalid_perm);
              MUNDY_THROW_ASSERT(
                  bulk_data_ptr_->bucket(hp1_crosslinker).topology() != stk::topology::INVALID_TOPOLOGY,
                  std::logic_error,
                  fmt::format("The crosslinker with id {} has an invalid topology.", hp1_crosslinker_id));
              // Assign the chainID
              stk::mesh::field_data(*element_chainid_field_ptr_, hp1_crosslinker)[0] = j;

              hp1_sphere_index++;
            }
          }
          std::cout << "  ...finished building hp1 segments" << std::endl;
        }
      }
    }
    bulk_data_ptr_->modification_end();
    std::cout << "...finished declaring system\n";
  }

  void initialize_chromosome_positions_from_file() {
    // The filename is in initialize_from_dat_filename_
    //
    // The file should be formatted as follows:
    // chromosome_id x y z
    // 0 x1 y1 z1
    // 0 x2 y2 z2
    // ...
    // 1 x1 y1 z1
    // 1 x2 y2 z2
    //
    // chromosome_id should start at 1
    //
    // And so on for each chromosome. The total number of nodes per chromosome should match the expected number of
    // nodes, as should the total number of chromosomes, lest we throw an exception.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                 num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
      const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
      const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;

      // Open the file
      std::ifstream infile(initialize_from_dat_filename_);
      MUNDY_THROW_REQUIRE(infile.is_open(), std::invalid_argument,
                          fmt::format("Could not open file {}", initialize_from_dat_filename_));

      // Read each line. While the chromosome_id is the same, keep adding nodes to the chromosome.
      size_t current_chromosome_id = 1;
      size_t current_node_id = 1;
      size_t num_nodes_per_this_chromosome = 0;
      std::string line;
      while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int chromosome_id;
        double x, y, z;
        if (!(iss >> chromosome_id >> x >> y >> z)) {
          MUNDY_THROW_REQUIRE(false, std::invalid_argument, fmt::format("Could not parse line {}", line));
        }
        if (chromosome_id != current_chromosome_id) {
          // We are starting a new chromosome
          MUNDY_THROW_REQUIRE(
              num_nodes_per_this_chromosome == num_nodes_per_chromosome, std::invalid_argument,
              fmt::format("Chromosome {} has {} nodes, but we expected {} nodes.", current_chromosome_id,
                          num_nodes_per_this_chromosome, num_nodes_per_chromosome));
          MUNDY_THROW_REQUIRE(chromosome_id == current_chromosome_id + 1, std::invalid_argument,
                              "Chromosome IDs should be sequential.");
          MUNDY_THROW_REQUIRE(
              chromosome_id <= num_chromosomes_, std::invalid_argument,
              fmt::format("Chromosome ID {} is greater than the number of chromosomes.", chromosome_id));
          current_chromosome_id = chromosome_id;
          num_nodes_per_this_chromosome = 0;
        }
        // Add the node to the chromosome
        stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, current_node_id);
        MUNDY_THROW_REQUIRE(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                            fmt::format("Node {} is not valid for chromosome {} out of {} chromosomes.",
                                        current_node_id, current_chromosome_id, num_chromosomes_));
        stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = x;
        stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = y;
        stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = z;
        current_node_id++;
        num_nodes_per_this_chromosome++;
      }
    }

    std::cout << "Finished initializing from file" << std::endl;
  }

  // Initialize the chromsomes on a grid
  void initialize_chromosome_positions_grid() {
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        openrand::Philox rng(j, 0);
        double jdouble = static_cast<double>(j);
        mundy::math::Vector3<double> r_start(2.0 * jdouble, 0.0, 0.0);
        // Add a tiny random change in X to make sure we don't wind up in perfectly parallel pathological states
        mundy::math::Vector3<double> u_hat(rng.uniform<double>(0.0, 0.001), 0.0, 1.0);
        u_hat = u_hat / mundy::math::two_norm(u_hat);

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                             fmt::format("Node {} is not valid", i));

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_chromosome_separation_ * u_hat;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = r[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = r[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = r[2];
        }
      }
    }
  }

  // Initialize the chromosomes randomly in the unit cell
  void initialize_chromosome_positions_random_unit_cell() {
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        // Find a random place within the unit cell with a random orientation for the chain.
        openrand::Philox rng(j, 0);
        mundy::math::Vector3<double> r_start(rng.uniform<double>(-0.5 * unit_cell_size_[0], 0.5 * unit_cell_size_[0]),
                                             rng.uniform<double>(-0.5 * unit_cell_size_[1], 0.5 * unit_cell_size_[1]),
                                             rng.uniform<double>(-0.5 * unit_cell_size_[2], 0.5 * unit_cell_size_[2]));
        // Find a random unit vector direction
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                             fmt::format("Node {} is not valid", i));

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_chromosome_separation_ * u_hat;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = r[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = r[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = r[2];
        }
      }
    }
  }

  // Initialize for the overlap test
  void initialize_chromosome_positions_overlap_test() {
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        // Start like we are pretending to be on a grid
        double jdouble = static_cast<double>(j);
        mundy::math::Vector3<double> r_start(2.0 * jdouble, 0.0, 0.0);
        mundy::math::Vector3<double> u_hat(0.0, 0.0, 1.0);

        // If num_chromosomes == 2, then try to do the crosshatch for a timestep?
        if (num_chromosomes_ == 2) {
          if (j == 0) {
            r_start = mundy::math::Vector3<double>(0.0, 0.0, 0.0);
            u_hat = mundy::math::Vector3<double>(0.0, 0.0, 1.0);
          } else if (j == 1) {
            r_start = mundy::math::Vector3<double>(-5.0, 0.25, 5.0);
            u_hat = mundy::math::Vector3<double>(1.0, 0.0, 0.0);
          }
        }

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                             fmt::format("Node {} is not valid", i));

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_chromosome_separation_ * u_hat;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = r[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = r[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = r[2];
        }
      }
    }
  }

  // Initialize for the U-shaped test
  void initialize_chromosome_positions_ushape_test() {
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      MUNDY_THROW_ASSERT(num_chromosomes_ == 1, std::invalid_argument, "This test is only for 1 chromosome.");
      MUNDY_THROW_ASSERT(num_chromatin_repeats_ == 3, std::invalid_argument, "This test is only for 3 repeats.");
      MUNDY_THROW_ASSERT(num_heterochromatin_per_repeat_ == 1, std::invalid_argument,
                         "This test is only for 1 H sphere per repeat.");
      MUNDY_THROW_ASSERT(num_euchromatin_per_repeat_ == 2, std::invalid_argument,
                         "This test is only for 2 E spheres per repeat.");

      // Start at Y = initial_chromosome_separation, X = 0.0, Z = 0.0, then move down in a u-shape and back up again in
      // Y
      const size_t j = 0u;
      const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                 num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
      const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
      const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
      size_t start_node_index = num_nodes_per_chromosome * j + 1u;
      size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;

      for (size_t i = start_node_index; i < end_node_index; ++i) {
        stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                           fmt::format("Node {} is not valid.", i));

        // Assign the node coordinates
        mundy::math::Vector3<double> r(0.0, 0.0, 0.0);
        if (i == start_node_index) {
          r[0] = -0.5 * initial_chromosome_separation_;
          r[1] = initial_chromosome_separation_;
        } else if (i == start_node_index + 1) {
          r[0] = -0.5 * initial_chromosome_separation_;
          r[1] = 0.0;
        } else if (i == start_node_index + 2) {
          r[0] = 0.5 * initial_chromosome_separation_;
          r[1] = 0.0;
        } else if (i == start_node_index + 3) {
          r[0] = 0.5 * initial_chromosome_separation_;
          r[1] = initial_chromosome_separation_;
        }
        stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = r[0];
        stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = r[1];
        stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = r[2];
      }
    }
  }

  // Initialize the chromosomes randomly in the unit cell
  //
  // If we want to initialize uniformly inside a sphere packing, here are the coordinates for a given number of spheres
  // within a bigger sphere.
  // http://hydra.nat.uni-magdeburg.de/packing/ssp/ssp.html
  void initialize_chromosome_positions_hilbert_random_unit_cell() {
    // We need to get which chromosome this rank is responsible for initializing, luckily, should follow what was done
    // for the creation step. Do this inside a modification loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      std::vector<mundy::math::Vector3<double>> chromosome_centers_array;
      std::vector<double> chromosome_radii_array;
      for (size_t ichromosome = 0; ichromosome < num_chromosomes_; ichromosome++) {
        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
        const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * ichromosome + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (ichromosome + 1) + 1u;

        // Generate a random unit vector (will be used for creating the locatino of the nodes, the random position in
        // the unit cell will be handled later).
        openrand::Philox rng(ichromosome, 0);
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

        // Once we have the number of chromosome spheres we can get the hilbert curve set up. This will be at some
        // orientation and then have sides with a length of initial_chromosome_separation.
        auto [hilbert_position_array, hilbert_directors] = mundy::math::create_hilbert_positions_and_directors(
            num_nodes_per_chromosome, u_hat, initial_chromosome_separation_);

        // Create the local positions of the spheres
        std::vector<mundy::math::Vector3<double>> sphere_position_array;
        for (size_t isphere = 0; isphere < num_nodes_per_chromosome; isphere++) {
          sphere_position_array.push_back(hilbert_position_array[isphere]);
        }

        // Figure out where the center of the chromosome is, and its radius, in its own local space
        mundy::math::Vector3<double> r_chromosome_center_local(0.0, 0.0, 0.0);
        double r_max = 0.0;
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          r_chromosome_center_local += sphere_position_array[i];
        }
        r_chromosome_center_local /= static_cast<double>(sphere_position_array.size());
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          r_max = std::max(r_max, mundy::math::two_norm(r_chromosome_center_local - sphere_position_array[i]));
        }

        // Do max_trials number of insertion attempts to get a random position and orientation within the unit cell that
        // doesn't overlap with exiting chromosomes.
        const size_t max_trials = 1000;
        size_t itrial = 0;
        bool chromosome_inserted = false;
        while (itrial <= max_trials) {
          // Generate a random position within the unit cell.
          mundy::math::Vector3<double> r_start(
              rng.uniform<double>(-0.5 * unit_cell_size_[0], 0.5 * unit_cell_size_[0]),
              rng.uniform<double>(-0.5 * unit_cell_size_[1], 0.5 * unit_cell_size_[1]),
              rng.uniform<double>(-0.5 * unit_cell_size_[2], 0.5 * unit_cell_size_[2]));

          // Check for overlaps with existing chromosomes
          bool found_overlap = false;
          for (size_t jchromosome = 0; jchromosome < chromosome_centers_array.size(); ++jchromosome) {
            double r_chromosome_distance = mundy::math::two_norm(chromosome_centers_array[jchromosome] - r_start);
            if (r_chromosome_distance < (r_max + chromosome_radii_array[jchromosome])) {
              found_overlap = true;
              break;
            }
          }
          if (found_overlap) {
            itrial++;
          } else {
            chromosome_inserted = true;
            chromosome_centers_array.push_back(r_start);
            chromosome_radii_array.push_back(r_max);
            break;
          }
        }
        MUNDY_THROW_REQUIRE(chromosome_inserted, std::runtime_error,
                            fmt::format("Failed to insert chromosome after {} trials.", max_trials));

        // Generate all the positions along the curve due to the placement in the global space
        std::vector<mundy::math::Vector3<double>> new_position_array;
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          new_position_array.push_back(chromosome_centers_array.back() + r_chromosome_center_local -
                                       sphere_position_array[i]);
        }

        // Update the coordinates for this chromosome
        for (size_t i = start_node_index, idx = 0; i < end_node_index; ++i, ++idx) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                             fmt::format("Node {} is not valid", i));

          // Assign the node coordinates
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = new_position_array[idx][0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = new_position_array[idx][1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = new_position_array[idx][2];
        }
      }
    }
  }

  void initialize_chromosome_positions_hilbert_lattice() {
    // Initialize the chromosomes (hilbert curves) on a lattice inscribed within the collision periphery.
    // If no periphery is specified, the lattice will be inscribed within the unit cell.
    //
    // If the periphery is a sphere, then the lattice can have a side length of 2 * r_sphere * sqrt(3) / 3.
    // The same is true for the ellipsoid but with r_sphere replaced by the smallest axis radius of the ellipsoid.
    //
    // Each chromosome is initialized as a hilbert curve with a random orientation and segment length
    // initial_chromosome_separation_. All chromosomes are considered to have the same number of nodes.

    if (bulk_data_ptr_->parallel_rank() == 0) {
      const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                 num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
      const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
      const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;

      // Because all chromosomes are the same, we can create a single hilbert curve and then place it in the lattice
      // for each chromosome.
      const mundy::math::Vector3<double> x_hat(1.0, 0.0, 0.0);
      auto [hilbert_position_array, hilbert_directors] = mundy::math::create_hilbert_positions_and_directors(
          num_nodes_per_chromosome, x_hat, initial_chromosome_separation_);

      // Determine the center and bounding radius of the hilbert curve
      mundy::math::Vector3<double> hilbert_center(0.0, 0.0, 0.0);
      for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
        hilbert_center += hilbert_position_array[i];
      }
      hilbert_center /= static_cast<double>(num_nodes_per_chromosome);
      double hilbert_bounding_radius = 0.0;
      for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
        hilbert_bounding_radius =
            std::max(hilbert_bounding_radius, mundy::math::two_norm(hilbert_center - hilbert_position_array[i]));
      }
      std::cout << "Hilbert curve center: " << hilbert_center << " bounding radius: " << hilbert_bounding_radius
                << std::endl;

      // Assert that the periphery is large enough to fit the lattice.
      double max_lattice_side_length;
      if (enable_periphery_collision_) {
        if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
          max_lattice_side_length = 2.0 * periphery_collision_radius_ * std::sqrt(3.0) / 3.0;
        } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
          const double min_axis_radius = std::min({periphery_collision_axis_radius1_, periphery_collision_axis_radius2_,
                                                   periphery_collision_axis_radius3_});
          max_lattice_side_length = 2.0 * min_axis_radius * std::sqrt(3.0) / 3.0;
        } else {
          MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Invalid periphery shape.");
        }
      } else {
        max_lattice_side_length = std::min({unit_cell_size_[0], unit_cell_size_[1], unit_cell_size_[2]});
      }

      const size_t num_chromosomes_per_side = static_cast<size_t>(std::ceil(std::pow(num_chromosomes_, 1.0 / 3.0)));
      const double lattice_spacing = max_lattice_side_length / num_chromosomes_per_side;
      MUNDY_THROW_REQUIRE(max_lattice_side_length >= num_chromosomes_per_side * hilbert_bounding_radius,
                          std::invalid_argument,
                          fmt::format("The lattice side length is too small to fit the chromosomes. The lattice side "
                                      "length is {} and the bounding radius is {}.",
                                      max_lattice_side_length, num_chromosomes_per_side * hilbert_bounding_radius));

      std::vector<mundy::math::Vector3<double>> chromosome_centers_array;
      std::vector<double> chromosome_radii_array;
      for (size_t ichromosome = 0; ichromosome < num_chromosomes_; ichromosome++) {
        // Figure out which nodes we are doing
        size_t start_node_index = num_nodes_per_chromosome * ichromosome + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (ichromosome + 1) + 1u;

        // Get the lattice position of this chromosome
        const size_t ix = ichromosome % num_chromosomes_per_side;
        const size_t iy = (ichromosome / num_chromosomes_per_side) % num_chromosomes_per_side;
        const size_t iz = ichromosome / (num_chromosomes_per_side * num_chromosomes_per_side);
        const mundy::math::Vector3<double> center =
            mundy::math::Vector3<double>((ix + 0.5) * lattice_spacing - 0.5 * max_lattice_side_length,
                                         (iy + 0.5) * lattice_spacing - 0.5 * max_lattice_side_length,
                                         (iz + 0.5) * lattice_spacing - 0.5 * max_lattice_side_length);

        // Generate a random orientation for the chromosome
        openrand::Philox rng(ichromosome, 0);
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);
        mundy::math::Quaternion<double> quat = mundy::math::quat_from_parallel_transport(x_hat, u_hat);

        std::cout << "Chromosome " << ichromosome << " center: " << center << " orientation: " << quat
                  << " radius: " << hilbert_bounding_radius << std::endl;

        // Update the coordinates for this chromosome
        for (size_t i = start_node_index, idx = 0; i < end_node_index; ++i, ++idx) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                             fmt::format("Node {} is not valid", i));

          // Assign the node coordinates
          auto pos = quat * (hilbert_position_array[idx] - hilbert_center) + center;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = pos[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = pos[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = pos[2];
        }
      }
    }
  }

  void initialize_chromosome_positions_hilbert_file() {
    // Initialize the chromosomes (hilbert curves) with centers given in a file.
    //
    // Each chromosome is initialized as a hilbert curve with a random orientation and segment length
    // initial_chromosome_separation_. All chromosomes are considered to have the same number of nodes.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      // Read in the chromosome centers from the file
      auto read_centers = [](const std::string &filename, const size_t expected_num_elements,
                             std::vector<double> &vector) {
        vector.resize(expected_num_elements);
        std::ifstream infile(filename);
        if (!infile) {
          MUNDY_THROW_REQUIRE(false, std::invalid_argument, fmt::format("Failed to open file: {}", filename));
        }
        // Parse the input
        size_t num_elements;
        if (!(infile >> num_elements)) {
          MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                              fmt::format("Failed to read vector size from file: {}", filename));
          return;
        }
        if (num_elements != expected_num_elements) {
          MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                              fmt::format("Vector size mismatch: expected {} elements, got {} elements.",
                                          expected_num_elements, num_elements));
          return;
        }
        for (size_t i = 0; i < num_elements; ++i) {
          if (!(infile >> vector[i])) {
            std::cerr << "Failed to read vector element at index " << i << std::endl;
            return;
          }
        }
        // Close the file
        infile.close();
      };
      std::string hilbert_centers_filename = hilbert_centers_filename_;
      std::vector<double> chromosome_centers;
      read_centers(hilbert_centers_filename, 3 * num_chromosomes_, chromosome_centers);
      const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                                 num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
      const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
      const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
      // Because all chromosomes are the same, we can create a single hilbert curve and then place it in the lattice
      // for each chromosome.
      const mundy::math::Vector3<double> x_hat(1.0, 0.0, 0.0);
      auto [hilbert_position_array, hilbert_directors] = mundy::math::create_hilbert_positions_and_directors(
          num_nodes_per_chromosome, x_hat, initial_chromosome_separation_);
      // Determine the center and bounding radius of the hilbert curve
      mundy::math::Vector3<double> hilbert_center(0.0, 0.0, 0.0);
      for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
        hilbert_center += hilbert_position_array[i];
      }
      hilbert_center /= static_cast<double>(num_nodes_per_chromosome);
      double hilbert_bounding_radius = 0.0;
      for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
        hilbert_bounding_radius =
            std::max(hilbert_bounding_radius, mundy::math::two_norm(hilbert_center - hilbert_position_array[i]));
      }
      std::cout << "Hilbert curve center: " << hilbert_center << " bounding radius: " << hilbert_bounding_radius
                << std::endl;
      for (size_t ichromosome = 0; ichromosome < num_chromosomes_; ichromosome++) {
        // Figure out which nodes we are doing
        size_t start_node_index = num_nodes_per_chromosome * ichromosome + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (ichromosome + 1) + 1u;
        const mundy::math::Vector3<double> center{chromosome_centers[3 * ichromosome],
                                                  chromosome_centers[3 * ichromosome + 1],
                                                  chromosome_centers[3 * ichromosome + 2]};
        // Generate a random orientation for the chromosome
        openrand::Philox rng(ichromosome, 0);
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);
        mundy::math::Quaternion<double> quat = mundy::math::quat_from_parallel_transport(x_hat, u_hat);
        std::cout << "Chromosome " << ichromosome << " center: " << center << " orientation: " << quat
                  << " radius: " << hilbert_bounding_radius << std::endl;
        // Update the coordinates for this chromosome
        for (size_t i = start_node_index, idx = 0; i < end_node_index; ++i, ++idx) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::invalid_argument,
                             fmt::format("Node {} is not valid", i));
          // Assign the node coordinates
          auto pos = quat * (hilbert_position_array[idx] - hilbert_center) + center;
          stk::mesh::field_data(*node_coord_field_ptr_, node)[0] = pos[0];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[1] = pos[1];
          stk::mesh::field_data(*node_coord_field_ptr_, node)[2] = pos[2];
        }
      }
    }
  }

  // Initialize the chromatin backbone and HP1 linkers based on part membership
  //
  // The part membership should already be set up, which makes this much easier to do, as we can just loop over the
  // parts.
  void initialize_chromatin_backbone_and_hp1() {
    // Note, the positions are potentially read from the restart file, the following fields are not.

    // Assert that all pointers are non-null
    MUNDY_THROW_REQUIRE(node_coord_field_ptr_ != nullptr, std::invalid_argument, "Node coordinate field is null.");
    MUNDY_THROW_REQUIRE(element_chainid_field_ptr_ != nullptr, std::invalid_argument, "Element chainID field is null.");
    MUNDY_THROW_REQUIRE(element_radius_field_ptr_ != nullptr, std::invalid_argument, "Element radius field is null.");
    if (backbone_collision_type_ == COLLISION_TYPE::HERTZIAN) {
      MUNDY_THROW_REQUIRE(element_youngs_modulus_field_ptr_ != nullptr, std::invalid_argument,
                          "Element youngs modulus field is null.");
      MUNDY_THROW_REQUIRE(element_poissons_ratio_field_ptr_ != nullptr, std::invalid_argument,
                          "Element poisson's ratio field is null.");
    }
    MUNDY_THROW_REQUIRE(element_spring_constant_field_ptr_ != nullptr, std::invalid_argument,
                        "Element spring constant field is null.");
    MUNDY_THROW_REQUIRE(element_spring_r0_field_ptr_ != nullptr, std::invalid_argument,
                        "Element spring r0 field is null.");
    MUNDY_THROW_REQUIRE(element_rng_field_ptr_ != nullptr, std::invalid_argument, "Element rng field is null.");
    MUNDY_THROW_REQUIRE(euchromatin_state_field_ptr_ != nullptr, std::invalid_argument,
                        "Euchromatin state field is null.");
    MUNDY_THROW_REQUIRE(euchromatin_perform_state_change_field_ptr_ != nullptr, std::invalid_argument,
                        "Euchromatin perform state change field is null.");
    MUNDY_THROW_REQUIRE(euchromatin_state_change_next_time_field_ptr_ != nullptr, std::invalid_argument,
                        "Euchromatin state change next time field is null.");
    MUNDY_THROW_REQUIRE(euchromatin_state_change_elapsed_time_field_ptr_ != nullptr, std::invalid_argument,
                        "Euchromatin state change elapsed time field is null.");

    // Initialize the backbone springs (EE, EH, HH)
    const stk::mesh::Selector backbone_segments = *ee_springs_part_ptr_ | *eh_springs_part_ptr_ | *hh_springs_part_ptr_;
    if (backbone_collision_type_ == COLLISION_TYPE::HERTZIAN) {
      mundy::mesh::utils::fill_field_with_value(backbone_segments, *element_youngs_modulus_field_ptr_,
                                                std::array<double, 1>{backbone_youngs_modulus_});
      mundy::mesh::utils::fill_field_with_value(backbone_segments, *element_poissons_ratio_field_ptr_,
                                                std::array<double, 1>{backbone_poissons_ratio_});
    }
    mundy::mesh::utils::fill_field_with_value(backbone_segments, *element_radius_field_ptr_,
                                              std::array<double, 1>{backbone_excluded_volume_radius_});
    mundy::mesh::utils::fill_field_with_value(backbone_segments, *element_spring_constant_field_ptr_,
                                              std::array<double, 1>{backbone_spring_constant_});
    mundy::mesh::utils::fill_field_with_value(backbone_segments, *element_spring_r0_field_ptr_,
                                              std::array<double, 1>{backbone_spring_r0_});

    // Initialize the EE springs (euchromatin activity)
    if (!restart_performed_) {
      mundy::mesh::utils::fill_field_with_value(*ee_springs_part_ptr_, *element_rng_field_ptr_,
                                                std::array<unsigned, 1>{0});
      mundy::mesh::utils::fill_field_with_value(*ee_springs_part_ptr_, *euchromatin_state_field_ptr_,
                                                std::array<unsigned, 1>{0});
      mundy::mesh::utils::fill_field_with_value(*ee_springs_part_ptr_, *euchromatin_perform_state_change_field_ptr_,
                                                std::array<unsigned, 1>{0});
      mundy::mesh::utils::fill_field_with_value(*ee_springs_part_ptr_, *euchromatin_state_change_next_time_field_ptr_,
                                                std::array<double, 1>{0});
      mundy::mesh::utils::fill_field_with_value(
          *ee_springs_part_ptr_, *euchromatin_state_change_elapsed_time_field_ptr_, std::array<double, 1>{0});
    }

    // Initialize HP1 springs
    mundy::mesh::utils::fill_field_with_value(*hp1_part_ptr_, *element_spring_constant_field_ptr_,
                                              std::array<double, 1>{crosslinker_spring_constant_});
    mundy::mesh::utils::fill_field_with_value(*hp1_part_ptr_, *element_spring_r0_field_ptr_,
                                              std::array<double, 1>{crosslinker_r0_});
    if (!restart_performed_) {
      mundy::mesh::utils::fill_field_with_value(*hp1_part_ptr_, *element_rng_field_ptr_, std::array<unsigned, 1>{0});
    }
    mundy::mesh::utils::fill_field_with_value(*hp1_part_ptr_, *element_radius_field_ptr_,
                                              std::array<double, 1>{crosslinker_cutoff_radius_});

    // Initialize the hydrodynamic spheres
    const stk::mesh::Selector chromatin_spheres = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::utils::fill_field_with_value(chromatin_spheres, *element_radius_field_ptr_,
                                              std::array<double, 1>{backbone_sphere_hydrodynamic_radius_});

    // Initialize node positions for each chromosome
    if (!restart_performed_) {
      if (initialization_type_ == INITIALIZATION_TYPE::GRID) {
        std::cout << "Initializing chromosomes on a grid" << std::endl;
        initialize_chromosome_positions_grid();
      } else if (initialization_type_ == INITIALIZATION_TYPE::FROM_DAT) {
        std::cout << "Initializing chromosomes from a dat file" << std::endl;
        initialize_chromosome_positions_from_file();
      } else if (initialization_type_ == INITIALIZATION_TYPE::RANDOM_UNIT_CELL) {
        std::cout << "Initializing chromosomes in a random unit cell" << std::endl;
        initialize_chromosome_positions_random_unit_cell();
      } else if (initialization_type_ == INITIALIZATION_TYPE::OVERLAP_TEST) {
        std::cout << "Initializing chromosomes as an overlap test" << std::endl;
        initialize_chromosome_positions_overlap_test();
      } else if (initialization_type_ == INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL) {
        std::cout << "Initializing chromosomes in a hilbert random unit cell" << std::endl;
        initialize_chromosome_positions_hilbert_random_unit_cell();
      } else if (initialization_type_ == INITIALIZATION_TYPE::HILBERT_FILE) {
        std::cout << "Initializing chromosomes as a hilbert curve with given centers " << std::endl;
        initialize_chromosome_positions_hilbert_file();
      } else if (initialization_type_ == INITIALIZATION_TYPE::USHAPE_TEST) {
        std::cout << "Initializing chromosomes as a U-shaped test" << std::endl;
        initialize_chromosome_positions_ushape_test();
      } else if (initialization_type_ == INITIALIZATION_TYPE::HILBERT_LATTICE) {
        std::cout << "Initializing chromosomes in a hilbert lattice" << std::endl;
        initialize_chromosome_positions_hilbert_lattice();
      } else {
        MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                            fmt::format("Unknown initialization type: {}", initialization_type_));
      }
    }
  }

  void initialize_euchromatin() {
    // Set the initial time lag for the active euchromatin forces

    // Selectors and aliases
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &euchromatin_state = *euchromatin_state_field_ptr_;
    stk::mesh::Field<double> &euchromatin_state_change_next_time = *euchromatin_state_change_next_time_field_ptr_;
    const double kon_inv = 1.0 / active_euchromatin_force_kon_;

    // Loop over the ee_springs and set the first time we would see a transition
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, ee_springs_part,
        [&element_rng_field, &euchromatin_state, &euchromatin_state_change_next_time, &kon_inv](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &euchromatin_spring) {
          // Get the fields we need
          unsigned *local_euchromatin_state = stk::mesh::field_data(euchromatin_state, euchromatin_spring);
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, euchromatin_spring);
          double *next_time = stk::mesh::field_data(euchromatin_state_change_next_time, euchromatin_spring);

          // Set them all to inactive to start
          local_euchromatin_state[0] = 0u;

          const stk::mesh::EntityId euchromatin_spring_gid = bulk_data.identifier(euchromatin_spring);
          openrand::Philox rng(euchromatin_spring_gid, element_rng_counter[0]);
          const double randu01 = rng.rand<double>();
          element_rng_counter[0]++;

          next_time[0] = -1.0 * kon_inv * std::log(randu01);
        });
  }

  void declare_and_initialize_hp1() {
    if (!restart_performed_) {
      create_chromatin_backbone_and_hp1();
    }
    initialize_chromatin_backbone_and_hp1();
    std::cout << "Done initializing!\n";
  }

  void initialize_hydrodynamic_periphery() {
    if ((periphery_hydro_quadrature_ == PERIPHERY_QUADRATURE::GAUSS_LEGENDRE) &&
        ((periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) ||
         ((periphery_hydro_shape_ == PERIPHERY_SHAPE::ELLIPSOID) &&
          (periphery_hydro_axis_radius1_ == periphery_hydro_axis_radius2_) &&
          (periphery_hydro_axis_radius2_ == periphery_hydro_axis_radius3_) &&
          (periphery_hydro_axis_radius3_ == periphery_hydro_axis_radius1_)))) {
      // Generate the quadrature points and weights for the sphere
      std::vector<double> points_vec;
      std::vector<double> weights_vec;
      std::vector<double> normals_vec;
      const bool invert = true;
      const bool include_poles = false;
      const size_t spectral_order = periphery_hydro_spectral_order_;
      const double radius =
          (periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) ? periphery_hydro_radius_ : periphery_hydro_axis_radius1_;
      mundy::alens::periphery::gen_sphere_quadrature(spectral_order, radius, &points_vec, &weights_vec, &normals_vec,
                                                     include_poles, invert);

      // Create the periphery object
      const size_t num_surface_nodes = weights_vec.size();
      periphery_ptr_ = std::make_shared<mundy::alens::periphery::Periphery>(num_surface_nodes, viscosity_);
      periphery_ptr_->set_surface_positions(points_vec.data())
          .set_quadrature_weights(weights_vec.data())
          .set_surface_normals(normals_vec.data());
    } else if (periphery_hydro_quadrature_ == PERIPHERY_QUADRATURE::FROM_FILE) {
      periphery_ptr_ =
          std::make_shared<mundy::alens::periphery::Periphery>(periphery_hydro_num_quadrature_points_, viscosity_);
      periphery_ptr_->set_surface_positions(periphery_hydro_quadrature_points_filename_)
          .set_quadrature_weights(periphery_hydro_quadrature_weights_filename_)
          .set_surface_normals(periphery_hydro_quadrature_normals_filename_);
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          "We currently only support GAUSS_LEGENDRE quadrature for "
                          "spheres and ellipsoids with equal radii or direct specification of the quadrature from a "
                          "file using FROM_FILE.");
    }

    // Run the precomputation for the inverse self-interaction matrix
    const bool write_to_file = false;
    periphery_ptr_->build_inverse_self_interaction_matrix(write_to_file);
  }

  void declare_and_initialize_periphery_bind_sites() {
    // Declare first
    std::cout << "Creating periphery bind site locations\n";
    bulk_data_ptr_->modification_begin();
    std::vector<std::size_t> requests(meta_data_ptr_->entity_rank_count(), 0);
    if (bulk_data_ptr_->parallel_rank() == 0) {
      requests[stk::topology::NODE_RANK] = periphery_num_bind_sites_;
      requests[stk::topology::ELEMENT_RANK] = periphery_num_bind_sites_;
    }
    std::vector<stk::mesh::Entity> requested_entities;
    bulk_data_ptr_->generate_new_entities(requests, requested_entities);
    bulk_data_ptr_->change_entity_parts(requested_entities, stk::mesh::PartVector{bs_part_ptr_},
                                        stk::mesh::PartVector{});
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t i = 0; i < periphery_num_bind_sites_; i++) {
        const stk::mesh::Entity &node_i = requested_entities[i];
        const stk::mesh::Entity &sphere_i = requested_entities[periphery_num_bind_sites_ + i];
        bulk_data_ptr_->declare_relation(sphere_i, node_i, 0);
      }
    }
    bulk_data_ptr_->modification_end();
    std::cout << "..Finished creating periphery bind site locations\n";

    // Initialize second
    if (periphery_bind_sites_type_ == PERIPHERY_BIND_SITES_TYPE::RANDOM) {
      // Sample the bind sites randomly on the surface of the periphery
      openrand::Philox rng(1234, 0);
      if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
        if (bulk_data_ptr_->parallel_rank() == 0) {
          for (size_t i = 0; i < periphery_num_bind_sites_; i++) {
            const stk::mesh::Entity &node_i = requested_entities[i];
            const stk::mesh::Entity &sphere_i = requested_entities[periphery_num_bind_sites_ + i];
            double *node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node_i);

            const double u1 = rng.rand<double>();
            const double u2 = rng.rand<double>();
            const double theta = 2.0 * M_PI * u1;
            const double phi = std::acos(2.0 * u2 - 1.0);
            node_coords[0] = periphery_collision_radius_ * std::sin(phi) * std::cos(theta);
            node_coords[1] = periphery_collision_radius_ * std::sin(phi) * std::sin(theta);
            node_coords[2] = periphery_collision_radius_ * std::cos(phi);
          }
        }
      } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
        if (bulk_data_ptr_->parallel_rank() == 0) {
          const double a = periphery_collision_axis_radius1_;
          const double b = periphery_collision_axis_radius2_;
          const double c = periphery_collision_axis_radius3_;
          const double inv_mu_max = 1.0 / std::max({b * c, a * c, a * b});
          auto keep = [&a, &b, &c, &inv_mu_max, &rng](double x, double y, double z) {
            const double mu_xyz =
                std::sqrt((b * c * x) * (b * c * x) + (a * c * y) * (a * c * y) + (a * b * z) * (a * b * z));
            return inv_mu_max * mu_xyz > rng.rand<double>();
          };

          for (size_t i = 0; i < periphery_num_bind_sites_; i++) {
            const stk::mesh::Entity &node_i = requested_entities[i];
            const stk::mesh::Entity &sphere_i = requested_entities[periphery_num_bind_sites_ + i];
            double *node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node_i);

            while (true) {
              // Generate a random point on the unit sphere
              const double u1 = rng.rand<double>();
              const double u2 = rng.rand<double>();
              const double theta = 2.0 * M_PI * u1;
              const double phi = std::acos(2.0 * u2 - 1.0);
              node_coords[0] = std::sin(phi) * std::cos(theta);
              node_coords[1] = std::sin(phi) * std::sin(theta);
              node_coords[2] = std::cos(phi);

              // Keep this point with probability proportional to the surface area element
              if (keep(node_coords[0], node_coords[1], node_coords[2])) {
                // Pushforward the point to the ellipsoid
                node_coords[0] *= a;
                node_coords[1] *= b;
                node_coords[2] *= c;
                break;
              }
            }
          }
        }
      } else {
        MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                            fmt::format("Unknown periphery shape. Recieved: {} but expected SPHERE or ELLIPSOID.",
                                        periphery_collision_shape_));
      }
    } else if (periphery_bind_sites_type_ == PERIPHERY_BIND_SITES_TYPE::FROM_FILE) {
      if (bulk_data_ptr_->parallel_rank() == 0) {
        // std::ifstream infile(periphery_bind_site_locations_filename_, std::ios::binary);
        // if (!infile) {
        //   std::cerr << "Failed to open file: " << periphery_bind_site_locations_filename_ << std::endl;
        //   return;
        // }

        // // Parse the input
        // size_t num_elements;
        // infile.read(reinterpret_cast<char *>(&num_elements), sizeof(size_t));
        // MUNDY_THROW_REQUIRE(num_elements == 3 * periphery_num_bind_sites_, std::invalid_argument,
        //                     fmt::format("Num bind sites mismatch: expected {} but got {}",
        //                                 3 * periphery_num_bind_sites_, num_elements));
        // for (size_t i = 0; i < periphery_num_bind_sites_; ++i) {
        //   const stk::mesh::Entity &node_i = requested_entities[i];
        //   const stk::mesh::Entity &sphere_i = requested_entities[periphery_num_bind_sites_ + i];
        //   double *node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node_i);
        //   for (size_t j = 0; j < 3; ++j) {
        //     infile.read(reinterpret_cast<char *>(&node_coords[3 * i + j]), sizeof(double));
        //   }
        // }

        // // Close the file
        // infile.close();
        std::cout << "..Reading " << periphery_num_bind_sites_
                  << " bind site locations from ASCII file : " << periphery_bind_site_locations_filename_ << std::endl;
        std::ifstream infile(periphery_bind_site_locations_filename_);
        MUNDY_THROW_REQUIRE(infile.is_open(), std::invalid_argument,
                            fmt::format("Could not open file {}", periphery_bind_site_locations_filename_));
        std::string line;
        while (std::getline(infile, line)) {
          std::istringstream iss(line);
          int bind_site_id;
          double x, y, z;
          if (!(iss >> bind_site_id >> x >> y >> z)) {
            MUNDY_THROW_REQUIRE(false, std::invalid_argument, fmt::format("Could not parse line {}", line));
          }
          const stk::mesh::Entity &node_i = requested_entities[bind_site_id];
          double *node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node_i);
          node_coords[0] = x;
          node_coords[1] = y;
          node_coords[2] = z;
        }
        infile.close();
      }
    } else {
      MUNDY_THROW_REQUIRE(
          false, std::invalid_argument,
          fmt::format("Unknown periphery bind sites type. Recieved: {} but expected RANDOM or FROM_FILE.",
                      periphery_bind_sites_type_));
    }
  }

  void zero_out_transient_node_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
  }

  void zero_out_transient_element_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*element_binding_rates_field_ptr_,
                                                      std::array<double, 2>{0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*element_unbinding_rates_field_ptr_,
                                                      std::array<double, 2>{0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<unsigned>(*element_perform_state_change_field_ptr_,
                                                        std::array<unsigned, 1>{0u});
    mundy::mesh::utils::fill_field_with_value<unsigned>(*euchromatin_perform_state_change_field_ptr_,
                                                        std::array<unsigned, 1>{0u});
  }

  void zero_out_transient_constraint_fields() {
    mundy::mesh::utils::fill_field_with_value<unsigned>(*constraint_perform_state_change_field_ptr_,
                                                        std::array<unsigned, 1>{0u});
    mundy::mesh::utils::fill_field_with_value<double>(*constraint_state_change_rate_field_ptr_,
                                                      std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*constraint_potential_force_field_ptr_,
                                                      std::array<double, 3>{0.0, 0.0, 0.0});
  }

  void zero_out_accumulator_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*element_corner_displacement_field_ptr_,
                                                      std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  }

  void detect_neighbors_initial() {
    Kokkos::Profiling::pushRegion("HP1::detect_neighbors_initial");

    last_neighborlist_update_step_ = 0;
    neighborlist_update_timer_.reset();

    // ComputeAABB for everyone (assume same buffer distance)
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto h_selector = stk::mesh::Selector(*h_part_ptr_);
    auto bs_selector = stk::mesh::Selector(*bs_part_ptr_);

    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);
    auto hp1_h_neighbor_genx_selector = stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_);
    auto hp1_bs_neighbor_genx_selector = stk::mesh::Selector(*hp1_bs_neighbor_genx_part_ptr_);

    compute_aabb_ptr_->execute(backbone_segments_selector | hp1_selector | h_selector | bs_selector);
    if (enable_backbone_collision_ || enable_crosslinkers_ || enable_periphery_binding_) {
      std::cout << "Destroying neighbor linkers" << std::endl;
      destroy_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector | hp1_h_neighbor_genx_selector |
                                             hp1_bs_neighbor_genx_selector);
      std::cout << "Done destroying neighbor linkers" << std::endl;
    }

    bool reghost = false;

    // Generate the GENX neighbor linkers
    if (enable_backbone_collision_) {
      generate_scs_scs_genx_ptr_->execute(backbone_segments_selector, backbone_segments_selector);
      reghost = true;
    }
    if (enable_crosslinkers_) {
      generate_hp1_h_genx_ptr_->execute(hp1_selector, h_selector);
      reghost = true;
    }
    if (enable_periphery_binding_) {
      generate_hp1_bs_genx_ptr_->execute(hp1_selector, bs_selector);
      reghost = true;
    }

    // Destroy linkers along backbone chains
    if (enable_backbone_collision_) {
      destroy_bound_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector);
      reghost = true;
    }

    if (reghost) {
      ghost_linked_entities();
    }

    Kokkos::Profiling::popRegion();
  }

  void detect_neighbors() {
    Kokkos::Profiling::pushRegion("HP1::detect_neighbors");

    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto h_selector = stk::mesh::Selector(*h_part_ptr_);
    auto bs_selector = stk::mesh::Selector(*bs_part_ptr_);

    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);
    auto hp1_h_neighbor_genx_selector = stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_);
    auto hp1_bs_neighbor_genx_selector = stk::mesh::Selector(*hp1_bs_neighbor_genx_part_ptr_);

    // ComputeAABB for everybody at each time step. The accumulator uses this updated information to
    // calculate if we need to update the entire neighbor list.
    compute_aabb_ptr_->execute(backbone_segments_selector | hp1_selector | h_selector | bs_selector);
    update_accumulators();

    // Check if we need to update the neighbor list. Eventually this will be replaced with a mesh attribute to
    // synchronize across multiple tasks. For now, make sure that the default is to not update neighbor lists.
    check_update_neighbor_list();

    // Now do a check to see if we need to update the neighbor list.
    if (((force_neighborlist_update_) && (timestep_index_ % force_neighborlist_update_nsteps_ == 0)) ||
        update_neighbor_list_) {
      // Read off the timing information before doing anything else and reset it
      auto elapsed_steps = timestep_index_ - last_neighborlist_update_step_;
      auto elapsed_time = neighborlist_update_timer_.seconds();
      neighborlist_update_steps_times_.push_back(std::make_tuple(timestep_index_, elapsed_steps, elapsed_time));
      last_neighborlist_update_step_ = timestep_index_;
      neighborlist_update_timer_.reset();

      // Reset the accumulators
      zero_out_accumulator_fields();

      bool reghost = false;

      // Update the neighbor list
      if (enable_backbone_collision_ || enable_crosslinkers_ || enable_periphery_binding_) {
        destroy_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector | hp1_h_neighbor_genx_selector |
                                               hp1_bs_neighbor_genx_selector);
        reghost = true;
      }

      // Generate the GENX neighbor linkers
      if (enable_backbone_collision_) {
        generate_scs_scs_genx_ptr_->execute(backbone_segments_selector, backbone_segments_selector);
        reghost = true;
      }
      if (enable_crosslinkers_) {
        generate_hp1_h_genx_ptr_->execute(hp1_selector, h_selector);
        reghost = true;
      }
      if (enable_periphery_binding_) {
        generate_hp1_bs_genx_ptr_->execute(hp1_selector, bs_selector);
        reghost = true;
      }

      // Destroy linkers along backbone chains
      if (enable_backbone_collision_) {
        destroy_bound_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector);
        reghost = true;
      }

      if (reghost) {
        ghost_linked_entities();
      }
    }

    Kokkos::Profiling::popRegion();
  }

  void update_accumulators() {
    Kokkos::Profiling::pushRegion("HP1::update_accumulators");

    // Selectors and aliases
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_aabb_field_old = element_aabb_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;

    stk::mesh::Selector combined_selector = spheres_selector | backbone_segments_selector | hp1_selector;

    // Update the accumulators based on the difference to the previous state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, combined_selector,
        [&element_aabb_field, &element_aabb_field_old, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_aabb = stk::mesh::field_data(element_aabb_field, aabb_entity);
          double *element_aabb_old = stk::mesh::field_data(element_aabb_field_old, aabb_entity);
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Add the (new_aabb - old_aabb) to the corner displacement
          element_corner_displacement[0] += element_aabb[0] - element_aabb_old[0];
          element_corner_displacement[1] += element_aabb[1] - element_aabb_old[1];
          element_corner_displacement[2] += element_aabb[2] - element_aabb_old[2];
          element_corner_displacement[3] += element_aabb[3] - element_aabb_old[3];
          element_corner_displacement[4] += element_aabb[4] - element_aabb_old[4];
          element_corner_displacement[5] += element_aabb[5] - element_aabb_old[5];
        });

    Kokkos::Profiling::popRegion();
  }

  void check_update_neighbor_list() {
    Kokkos::Profiling::pushRegion("HP1::check_update_neighbor_list");

    // Local variable for if we should update the neighbor list (do as an integer for now because MPI doesn't like
    // bools)
    int local_update_neighbor_list_int = 0;

    // Selectors and aliases
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);

    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;
    const double skin_distance2_over4 = 0.25 * skin_distance_ * skin_distance_;

    stk::mesh::Selector combined_selector = spheres_selector | backbone_segments_selector | hp1_selector;

    // Check if each corner has moved skin_distance/2. Or, if dr_mag2 >= skin_distance^2/4
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, combined_selector,
        [&local_update_neighbor_list_int, &skin_distance2_over4, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Compute dr2 for each corner
          double dr2_corner0 = element_corner_displacement[0] * element_corner_displacement[0] +
                               element_corner_displacement[1] * element_corner_displacement[1] +
                               element_corner_displacement[2] * element_corner_displacement[2];
          double dr2_corner1 = element_corner_displacement[3] * element_corner_displacement[3] +
                               element_corner_displacement[4] * element_corner_displacement[4] +
                               element_corner_displacement[5] * element_corner_displacement[5];

          if (dr2_corner0 >= skin_distance2_over4 || dr2_corner1 >= skin_distance2_over4) {
            local_update_neighbor_list_int =
                1;  // TODO(palmerb4): This is not thread safe. Use a reduction over a logical or
          }
        });

    // Communicate local_update_neighbor_list to all ranks. Convert to an integer first (MPI doesn't handle booleans
    // well).
    int global_update_neighbor_list_int = 0;
    MPI_Allreduce(&local_update_neighbor_list_int, &global_update_neighbor_list_int, 1, MPI_INT, MPI_LOR,
                  MPI_COMM_WORLD);
    // Convert back to the boolean for the global version and or it with the original value (in case somebody else set
    // the neighbor list update 'signal').
    update_neighbor_list_ = update_neighbor_list_ || (global_update_neighbor_list_int == 1);

    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for left-bound crosslinkers binding to a sphere
  void compute_z_partition_left_bound() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition_left_bound");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &constraint_state_change_probability = *constraint_state_change_rate_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_constant = *element_spring_constant_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_r0 = *element_spring_r0_field_ptr_;
    const mundy::linkers::LinkedEntitiesFieldType &constraint_linked_entities_field =
        *constraint_linked_entities_field_ptr_;
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;
    stk::mesh::Part &hp1_h_neighbor_genx_part = *hp1_h_neighbor_genx_part_ptr_;
    const double crosslinker_right_binding_rate = crosslinker_right_binding_rate_;
    const double inv_kt = 1.0 / crosslinker_kt_;

    const auto &crosslinker_spring_type = crosslinker_spring_type_;

    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, hp1_h_neighbor_genx_part,
        [&node_coord_field, &constraint_linked_entities_field, &constraint_state_change_probability,
         &crosslinker_spring_constant, &crosslinker_spring_r0, &left_hp1_part, &inv_kt, &crosslinker_right_binding_rate,
         &crosslinker_spring_type]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                   const stk::mesh::Entity &neighbor_genx) {
          // Get the sphere and crosslinker attached to the linker.
          const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
              stk::mesh::field_data(constraint_linked_entities_field, neighbor_genx));
          const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
          const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);

          MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker), std::invalid_argument,
                             "Encountered invalid crosslinker entity in compute_z_partition_left_bound_harmonic.");
          MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere), std::invalid_argument,
                             "Encountered invalid sphere entity in compute_z_partition_left_bound_harmonic.");

          // We need to figure out if this is a self-interaction or not. Since we are a left-bound crosslinker.
          const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];
          bool is_self_interaction = false;
          if (bulk_data.bucket(crosslinker).member(left_hp1_part)) {
            is_self_interaction = bulk_data.begin_nodes(crosslinker)[0] == sphere_node;
          }

          // Only act on the left-bound crosslinkers
          if (bulk_data.bucket(crosslinker).member(left_hp1_part) && !is_self_interaction) {
            const auto dr = mundy::mesh::vector3_field_data(node_coord_field, sphere_node) -
                            mundy::mesh::vector3_field_data(node_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
            const double dr_mag = mundy::math::norm(dr);

            // Compute the Z-partition score
            if (crosslinker_spring_type == BOND_TYPE::HARMONIC) {
              // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
              // A = crosslinker_binding_rates
              // k = crosslinker_spring_constant
              // r0 = crosslinker_spring_rest_length
              const double A = crosslinker_right_binding_rate;
              const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
              const double r0 = stk::mesh::field_data(crosslinker_spring_r0, crosslinker)[0];
              double Z = A * std::exp(-0.5 * inv_kt * k * (dr_mag - r0) * (dr_mag - r0));
              stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
            } else if (crosslinker_spring_type == BOND_TYPE::FENE) {
              // Z = A * (1 - (r/r0)^2)^(0.5 * 1/kt * k * r0^2)
              // A = crosslinker_binding_rates
              // k = crosslinker_spring_constant
              // r0 = crosslinker_spring_max_length (FENE)
              // R = crosslinker_fene_max_distance
              const double A = crosslinker_right_binding_rate;
              const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
              const double r0 = stk::mesh::field_data(crosslinker_spring_r0, crosslinker)[0];
              double Z = 0.0;
              if (dr_mag < r0) {
                Z = A * std::pow(1.0 - (dr_mag / r0) * (dr_mag / r0), 0.5 * inv_kt * k * r0 * r0);
              }
              stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
            }
          }
        });

    if (enable_periphery_binding_) {
      const double periphery_binding_rate = periphery_binding_rate_;
      const double periphery_spring_constant = periphery_spring_constant_;
      const double periphery_spring_r0 = periphery_spring_r0_;
      stk::mesh::Part &hp1_bs_neighbor_genx_part = *hp1_bs_neighbor_genx_part_ptr_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, hp1_bs_neighbor_genx_part,
          [&node_coord_field, &constraint_linked_entities_field, &constraint_state_change_probability,
           &periphery_spring_constant, &periphery_spring_r0, &left_hp1_part, &inv_kt, &periphery_binding_rate,
           &crosslinker_spring_type]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                     const stk::mesh::Entity &neighbor_genx) {
            // Get the sphere and crosslinker attached to the linker.
            const stk::mesh::EntityKey::entity_key_t *key_t_ptr =
                reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
                    stk::mesh::field_data(constraint_linked_entities_field, neighbor_genx));
            const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
            const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);
            const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];

            MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker), std::invalid_argument,
                               "Encountered invalid crosslinker entity in compute_z_partition_left_bound_harmonic.");
            MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere), std::invalid_argument,
                               "Encountered invalid sphere entity in compute_z_partition_left_bound_harmonic.");

            // Only act on the left-bound crosslinkers
            if (bulk_data.bucket(crosslinker).member(left_hp1_part)) {
              const auto dr = mundy::mesh::vector3_field_data(node_coord_field, sphere_node) -
                              mundy::mesh::vector3_field_data(node_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
              const double dr_mag = mundy::math::norm(dr);

              // Compute the Z-partition score
              if (crosslinker_spring_type == BOND_TYPE::HARMONIC) {
                // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
                // A = periphery_binding_rate
                // k = periphery_spring_constant
                // r0 = periphery_spring_r0
                const double A = periphery_binding_rate;
                const double k = periphery_spring_constant;
                const double r0 = periphery_spring_r0;
                double Z = A * std::exp(-0.5 * inv_kt * k * (dr_mag - r0) * (dr_mag - r0));
                stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
              } else if (crosslinker_spring_type == BOND_TYPE::FENE) {
                // Z = A * (1 - (r/r0)^2)^(0.5 * 1/kt * k * r0^2)
                // A = periphery_binding_rate
                // k = periphery_spring_constant
                // r0 = periphery_spring_r0
                const double A = periphery_binding_rate;
                const double k = periphery_spring_constant;
                const double r0 = periphery_spring_r0;
                double Z = 0.0;
                if (dr_mag < r0) {
                  Z = A * std::pow(1.0 - (dr_mag / r0) * (dr_mag / r0), 0.5 * inv_kt * k * r0 * r0);
                }
                stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
              }
            }
          });
    }
    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for doubly_bound crosslinkers
  void compute_z_partition_doubly_bound() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition_doubly_bound");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;
    const double &crosslinker_right_unbinding_rate = crosslinker_right_unbinding_rate_;

    // Loop over the neighbor list of the crosslinkers, then select down to the ones that are left-bound only.
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_hp1_h_part,
        [&node_coord_field, &crosslinker_unbinding_rates, &doubly_hp1_h_part, &crosslinker_right_unbinding_rate](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
// This is a left-bound crosslinker, so just calculate the right unbinding rate and store on the crosslinker
// itself in the correct position.
#pragma todo This needs to have a different rate for the periphery versus hp1 type bindings
          stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1] = crosslinker_right_unbinding_rate;
        });

    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function for everybody
  void compute_z_partition() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition");

    // Compute the left-bound to doubly-bound score
    // Works for both binding to an h-sphere and binding to a bs-sphere
    compute_z_partition_left_bound();

    // Compute the doubly-bound to left-bound score
    compute_z_partition_doubly_bound();

    Kokkos::Profiling::popRegion();
  }

  void kmc_crosslinker_left_to_doubly() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_left_to_doubly");

    // Selectors and aliases
    stk::mesh::Part &hp1_h_neighbor_genx_part = *hp1_h_neighbor_genx_part_ptr_;
    stk::mesh::Part &hp1_bs_neighbor_genx_part = *hp1_bs_neighbor_genx_part_ptr_;
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    stk::mesh::Field<unsigned> &constraint_perform_state_change_field = *constraint_perform_state_change_field_ptr_;
    stk::mesh::Field<double> &constraint_state_change_rate_field = *constraint_state_change_rate_field_ptr_;
    const mundy::linkers::LinkedEntitiesFieldType &constraint_linked_entities_field =
        *constraint_linked_entities_field_ptr_;
    const double timestep_size = timestep_size_;
    const double enable_periphery_binding = enable_periphery_binding_;
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;

    // Loop over left-bound crosslinkers and decide if they bind or not
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, left_hp1_part,
        [&hp1_h_neighbor_genx_part, &hp1_bs_neighbor_genx_part, &element_rng_field,
         &constraint_perform_state_change_field, &element_perform_state_change_field,
         &constraint_state_change_rate_field, &constraint_linked_entities_field, &timestep_size,
         &enable_periphery_binding]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                    const stk::mesh::Entity &crosslinker) {
          // Get all of my associated crosslinker_sphere_linkers
          const stk::mesh::Entity &any_arbitrary_crosslinker_node = bulk_data.begin_nodes(crosslinker)[0];
          const stk::mesh::Entity *neighbor_genx_linkers =
              bulk_data.begin(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);
          const unsigned num_neighbor_genx_linkers =
              bulk_data.num_connectivity(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);

          // Loop over the attached crosslinker_sphere_linkers and bind if the rqng falls in their range.
          double z_tot = 0.0;
          for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
            const auto &constraint_rank_entity = neighbor_genx_linkers[j];
            const bool is_hp1_h_neighbor_genx =
                bulk_data.bucket(constraint_rank_entity).member(hp1_h_neighbor_genx_part);
            const bool is_hp1_bs_neighbor_genx =
                bulk_data.bucket(constraint_rank_entity).member(hp1_bs_neighbor_genx_part);
            if (is_hp1_h_neighbor_genx || (enable_periphery_binding && is_hp1_bs_neighbor_genx)) {
              const double z_i =
                  timestep_size * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
              z_tot += z_i;
            }
          }

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, crosslinker);
          const stk::mesh::EntityId crosslinker_gid = bulk_data.identifier(crosslinker);
          openrand::Philox rng(crosslinker_gid, element_rng_counter[0]);
          const double randu01 = rng.rand<double>();
          element_rng_counter[0]++;

          // Notice that the sum of all probabilities is 1.
          // The probability of nothing happening is
          //   std::exp(-z_tot)
          // The probability of an individual event happening is
          //   z_i / z_tot * (1 - std::exp(-z_tot))
          //
          // This is (by construction) true since
          //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
          //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
          //
          // This means that binding only happens if randu01 < (1 - std::exp(-z_tot))
          const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
          const double scale_factor = probability_of_no_state_change * timestep_size / z_tot;
          if (randu01 < (1.0 - std::exp(-z_tot))) {
            // Binding occurs.
            // Loop back over the neighbor linkers to see if one of them binds in the running sum
            double cumsum = 0.0;
            for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
              auto &constraint_rank_entity = neighbor_genx_linkers[j];
              bool is_hp1_h_neighbor_genx = bulk_data.bucket(constraint_rank_entity).member(hp1_h_neighbor_genx_part);
              bool is_hp1_bs_neighbor_genx = bulk_data.bucket(constraint_rank_entity).member(hp1_bs_neighbor_genx_part);
              // Check in both the periphery binding and hp1 binding parts
              if (is_hp1_h_neighbor_genx || (enable_periphery_binding && is_hp1_bs_neighbor_genx)) {
                const double binding_probability =
                    scale_factor * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
                cumsum += binding_probability;
                if (randu01 < cumsum) {
                  // We have a binding event, set this, then bail on the for loop
                  // Store the state change on both the genx and the crosslinker
                  stk::mesh::field_data(constraint_perform_state_change_field, constraint_rank_entity)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  stk::mesh::field_data(element_perform_state_change_field, crosslinker)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  break;
                }
              }
            }
          }
        });

    Kokkos::Profiling::popRegion();
  }

  /// \brief Perform the binding of a crosslinker to a sphere (doubly to left)
  void kmc_crosslinker_doubly_to_left() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_doubly_to_left");

    // Selectors and aliases
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    const double &timestep_size = timestep_size_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;

    // This is just a loop over the doubly bound crosslinkers, since we know that the right head in is [1].
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_hp1_h_part,
        [&element_rng_field, &element_perform_state_change_field, &crosslinker_unbinding_rates, &timestep_size](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          // We only have a single node, our right node, that is bound that we can unbind.
          // TODO(cje): Right now this is coded to have a loop wrapping it, maybe not needed?
          const double unbinding_probability =
              timestep_size * stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1];
          double Z_tot = unbinding_probability;
          const double unbind_scale_factor = (1.0 - exp(-Z_tot)) * timestep_size;

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, crosslinker);
          const stk::mesh::EntityId crosslinker_gid = bulk_data.identifier(crosslinker);
          openrand::Philox rng(crosslinker_gid, element_rng_counter[0]);
          double randZ = rng.rand<double>() * Z_tot;
          double cumsum = 0.0;
          element_rng_counter[0]++;

          // Now check the cummulative sum and if less than perform the unbinding
          cumsum += unbind_scale_factor * stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1];
          if (randZ < cumsum) {
            // Set the state change on the element
            stk::mesh::field_data(element_perform_state_change_field, crosslinker)[0] =
                static_cast<unsigned>(BINDING_STATE_CHANGE::DOUBLY_TO_LEFT);
          }
        });

    Kokkos::Profiling::popRegion();
  }

  void kmc_crosslinker_sphere_linker_sampling() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_sphere_linker_sampling");

    // Perform the left to doubly bound crosslinker binding calc
    kmc_crosslinker_left_to_doubly();

    // Perform the doubly to left bound crosslinker binding calc
    kmc_crosslinker_doubly_to_left();

    // At this point, constraint_state_change_rate_field is only up-to-date for locally-owned entities. We need
    // to communicate this information to all other processors.
    stk::mesh::communicate_field_data(
        *bulk_data_ptr_, {element_perform_state_change_field_ptr_, constraint_perform_state_change_field_ptr_});

    Kokkos::Profiling::popRegion();
  }

  /// \brief Perform the state change of the crosslinkers
  void state_change_crosslinkers() {
    Kokkos::Profiling::pushRegion("HP1::state_change_crosslinkers");

    // Loop over both the CROSSLINKER_SPHERE_LINKERS and the CROSSLINKERS to perform the state changes.
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;

    // Get the vector of entities to modify
    stk::mesh::EntityVector hp1_h_neighbor_genxs;
    stk::mesh::EntityVector hp1_bs_neighbor_genxs;
    stk::mesh::EntityVector doubly_bound_hp1s;
    stk::mesh::get_selected_entities(stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_),
                                     bulk_data_ptr_->buckets(constraint_rank_), hp1_h_neighbor_genxs);
    stk::mesh::get_selected_entities(stk::mesh::Selector(*hp1_bs_neighbor_genx_part_ptr_),
                                     bulk_data_ptr_->buckets(constraint_rank_), hp1_bs_neighbor_genxs);
    stk::mesh::get_selected_entities(stk::mesh::Selector(*doubly_hp1_h_part_ptr_),
                                     bulk_data_ptr_->buckets(element_rank_), doubly_bound_hp1s);

    // TODO(cje): It might be worth checking to see if we have any state changes in any threads before we crack open the
    // modification section, as even doing that is slightly expensive.

    bulk_data_ptr_->modification_begin();

    // Perform L->D
    for (const stk::mesh::Entity &hp1_h_neighbor_genx : hp1_h_neighbor_genxs) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*constraint_perform_state_change_field_ptr_, hp1_h_neighbor_genx)[0]);
      const bool perform_state_change = state_change_action != BINDING_STATE_CHANGE::NONE;
      if (perform_state_change) {
        // Get our connections (as the genx)
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(*constraint_linked_entities_field_ptr_, hp1_h_neighbor_genx));
        const stk::mesh::Entity &crosslinker_hp1 = bulk_data_ptr_->get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &target_sphere = bulk_data_ptr_->get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(crosslinker_hp1), std::invalid_argument,
                           "Encountered invalid crosslinker entity in state_change_crosslinkers.");
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(target_sphere), std::invalid_argument,
                           "Encountered invalid sphere entity in state_change_crosslinkers.");

        const stk::mesh::Entity &target_sphere_node = bulk_data_ptr_->begin_nodes(target_sphere)[0];
        // Call the binding function
        if (state_change_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
          // Unbind the right side of the crosslinker from the left node and bind it to the target node
          const bool bind_worked = ::mundy::alens::crosslinkers::bind_crosslinker_to_node_unbind_existing(
              *bulk_data_ptr_, crosslinker_hp1, target_sphere_node, 1);
          MUNDY_THROW_ASSERT(bind_worked, std::logic_error, "Failed to bind crosslinker to node.");

          std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Binding crosslinker "
                    << bulk_data_ptr_->identifier(crosslinker_hp1) << " to node "
                    << bulk_data_ptr_->identifier(target_sphere_node) << std::endl;

          // Now change the part from left to doubly bound.
          const bool is_crosslinker_locally_owned =
              bulk_data_ptr_->parallel_owner_rank(crosslinker_hp1) == bulk_data_ptr_->parallel_rank();
          if (is_crosslinker_locally_owned) {
            auto add_parts = stk::mesh::PartVector{doubly_hp1_h_part_ptr_};
            auto remove_parts = stk::mesh::PartVector{left_hp1_part_ptr_};
            bulk_data_ptr_->change_entity_parts(crosslinker_hp1, add_parts, remove_parts);
          }
        }
      }
    }
    // Do the periphery binding as well
    if (enable_periphery_binding_) {
      for (const stk::mesh::Entity &hp1_bs_neighbor_genx : hp1_bs_neighbor_genxs) {
        // Decode the binding type enum for this entity
        auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
            stk::mesh::field_data(*constraint_perform_state_change_field_ptr_, hp1_bs_neighbor_genx)[0]);
        const bool perform_state_change = state_change_action != BINDING_STATE_CHANGE::NONE;
        if (perform_state_change) {
          // Get our connections (as the genx)
          const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
              stk::mesh::field_data(*constraint_linked_entities_field_ptr_, hp1_bs_neighbor_genx));
          const stk::mesh::Entity &crosslinker_hp1 = bulk_data_ptr_->get_entity(key_t_ptr[0]);
          const stk::mesh::Entity &target_sphere = bulk_data_ptr_->get_entity(key_t_ptr[1]);

          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(crosslinker_hp1), std::invalid_argument,
                             "Encountered invalid crosslinker entity in state_change_crosslinkers.");
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(target_sphere), std::invalid_argument,
                             "Encountered invalid sphere entity in state_change_crosslinkers.");

          const stk::mesh::Entity &target_sphere_node = bulk_data_ptr_->begin_nodes(target_sphere)[0];
          // Call the binding function
          if (state_change_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
            // Unbind the right side of the crosslinker from the left node and bind it to the target node
            const bool bind_worked = ::mundy::alens::crosslinkers::bind_crosslinker_to_node_unbind_existing(
                *bulk_data_ptr_, crosslinker_hp1, target_sphere_node, 1);
            MUNDY_THROW_ASSERT(bind_worked, std::logic_error, "Failed to bind crosslinker to node.");

            std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Periphery: Binding crosslinker "
                      << bulk_data_ptr_->identifier(crosslinker_hp1) << " to node "
                      << bulk_data_ptr_->identifier(target_sphere_node) << std::endl;

            // Now change the part from left to doubly bound.
            const bool is_crosslinker_locally_owned =
                bulk_data_ptr_->parallel_owner_rank(crosslinker_hp1) == bulk_data_ptr_->parallel_rank();
            if (is_crosslinker_locally_owned) {
              auto add_parts = stk::mesh::PartVector{doubly_hp1_h_part_ptr_};
              auto remove_parts = stk::mesh::PartVector{left_hp1_part_ptr_};
              bulk_data_ptr_->change_entity_parts(crosslinker_hp1, add_parts, remove_parts);
            }
          }
        }
      }
    }

    // Perform D->L
    // This should handle both the HP1 and periphery crosslinkers
    for (const stk::mesh::Entity &crosslinker_hp1 : doubly_bound_hp1s) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*element_perform_state_change_field_ptr_, crosslinker_hp1)[0]);
      if (state_change_action == BINDING_STATE_CHANGE::DOUBLY_TO_LEFT) {
        // Unbind the right side of the crosslinker from the current node and bind it to the left crosslinker node
        const stk::mesh::Entity &left_node = bulk_data_ptr_->begin_nodes(crosslinker_hp1)[0];
        const bool unbind_worked = ::mundy::alens::crosslinkers::bind_crosslinker_to_node_unbind_existing(
            *bulk_data_ptr_, crosslinker_hp1, left_node, 1);
        MUNDY_THROW_ASSERT(unbind_worked, std::logic_error, "Failed to unbind crosslinker from node.");

        std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Unbinding crosslinker "
                  << bulk_data_ptr_->identifier(crosslinker_hp1) << " from node "
                  << bulk_data_ptr_->identifier(bulk_data_ptr_->begin_nodes(crosslinker_hp1)[1]) << std::endl;

        // Now change the part from doubly to left bound.
        const bool is_crosslinker_locally_owned =
            bulk_data_ptr_->parallel_owner_rank(crosslinker_hp1) == bulk_data_ptr_->parallel_rank();
        if (is_crosslinker_locally_owned) {
          auto add_parts = stk::mesh::PartVector{left_hp1_part_ptr_};
          auto remove_parts = stk::mesh::PartVector{doubly_hp1_h_part_ptr_};
          bulk_data_ptr_->change_entity_parts(crosslinker_hp1, add_parts, remove_parts);
        }
      }
    }

    bulk_data_ptr_->modification_end();

    // The above may have invalidated the ghosting for our genx ghosting, so we need to reghost the linked entities to
    // any process that owns any of the other linked entities.
    ghost_linked_entities();

    Kokkos::Profiling::popRegion();
  }

  void update_crosslinker_state() {
    Kokkos::Profiling::pushRegion("HP1::update_crosslinker_state");

    // We want to loop over all LEFT_BOUND_CROSSLINKERS, RIGHT_BOUND_CROSSLINKERS, and DOUBLY_BOUND_CROSSLINKERS to
    // generate state changes. This is done to build up a list of actions that we will take later during a mesh
    // modification step.
    {
      compute_z_partition();
      kmc_crosslinker_sphere_linker_sampling();
    }

    // Loop over the different crosslinkers, look at their actions, and enforce the state change.
    {
      // Call the global state change function
      state_change_crosslinkers();
    }

    Kokkos::Profiling::popRegion();
  }

  void active_euchromatin_sampling() {
    Kokkos::Profiling::pushRegion("HP1::active_euchromatin_sampling");

    // Selectors and aliases
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &euchromatin_state = *euchromatin_state_field_ptr_;
    stk::mesh::Field<unsigned> &euchromatin_perform_state_change = *euchromatin_perform_state_change_field_ptr_;
    stk::mesh::Field<double> &euchromatin_state_change_next_time = *euchromatin_state_change_next_time_field_ptr_;
    stk::mesh::Field<double> &euchromatin_state_change_elapsed_time = *euchromatin_state_change_elapsed_time_field_ptr_;

    const double &timestep_size = timestep_size_;
    double kon_inv = 1.0 / active_euchromatin_force_kon_;
    double koff_inv = 1.0 / active_euchromatin_force_koff_;

    // Loop over the euchromatin spring elements and decide if they switch to the active state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, ee_springs_part,
        [&element_rng_field, &euchromatin_state, &euchromatin_perform_state_change, &euchromatin_state_change_next_time,
         &euchromatin_state_change_elapsed_time, &kon_inv, &koff_inv](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &euchromatin_spring) {
          // We are not going to increment the elapsed time ourselves, but rely on someone outside of this loop to do
          // that at the end of a timestpe, in order to keep it consistent with the total elapsed time in the system.
          unsigned *current_state = stk::mesh::field_data(euchromatin_state, euchromatin_spring);
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, euchromatin_spring);
          double *next_time = stk::mesh::field_data(euchromatin_state_change_next_time, euchromatin_spring);
          double *elapsed_time = stk::mesh::field_data(euchromatin_state_change_elapsed_time, euchromatin_spring);

          if (elapsed_time[0] >= next_time[0]) {
            // Need a random number no matter what
            const stk::mesh::EntityId euchromatin_spring_gid = bulk_data.identifier(euchromatin_spring);
            openrand::Philox rng(euchromatin_spring_gid, element_rng_counter[0]);
            const double randu01 = rng.rand<double>();
            element_rng_counter[0]++;

            // Determine switch based on current state
            if (current_state[0] == 0u) {
              // Currently inactive, set to active and reset the timers
              current_state[0] = 1u;
              next_time[0] = -std::log(randu01) * koff_inv;
              elapsed_time[0] = 0.0;
            } else {
              // Currently active, set to active and reset the timers
              current_state[0] = 0u;
              next_time[0] = -std::log(randu01) * kon_inv;
              elapsed_time[0] = 0.0;
            }

            // #pragma omp critical
            //             {
            //               const unsigned previous_state = current_state[0] == 0u ? 1u : 0u;
            //               std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD)
            //                         << " Detected euchromatin switching event object " <<
            //                         bulk_data.identifier(euchromatin_spring)
            //                         << ", previous state: " << previous_state << ", current_state: " <<
            //                         current_state[0]
            //                         << std::endl;
            //               std::cout << "  next_time: " << next_time[0] << ", elapsed_time: " << elapsed_time[0] <<
            //               std::endl;
            //             }
          }
        });
    Kokkos::Profiling::popRegion();
  }

  void update_euchromatin_state_time() {
    Kokkos::Profiling::pushRegion("HP1::active_euchromatin_sampling");

    // Selectors and aliases
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Field<double> &euchromatin_state_change_elapsed_time = *euchromatin_state_change_elapsed_time_field_ptr_;
    const double &timestep_size = timestep_size_;

    // Loop over the euchromatin spring elements and decide if they switch to the active state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, ee_springs_part,
        [&euchromatin_state_change_elapsed_time, &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                                 const stk::mesh::Entity &euchromatin_spring) {
          // Updated the elapsed time
          stk::mesh::field_data(euchromatin_state_change_elapsed_time, euchromatin_spring)[0] += timestep_size;
        });

    Kokkos::Profiling::popRegion();
  }

  void update_active_euchromatin_state() {
    Kokkos::Profiling::pushRegion("HP1::update_active_euchromatin_state");

    // Determine if we need to update the euchromatin active state in the same way as the crosslinkers,
    active_euchromatin_sampling();
    // active_euchromatin_state_change();

    Kokkos::Profiling::popRegion();
  }

  void check_maximum_overlap_with_hydro_periphery() {
    if (periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) {
      const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
      stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
      stk::mesh::Field<double> &element_hydro_radius_field = *element_radius_field_ptr_;
      double shifted_periphery_hydro_radius = periphery_hydro_radius_ + maximum_allowed_periphery_overlap_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
          [&node_coord_field, &element_hydro_radius_field, &shifted_periphery_hydro_radius](
              const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            const double sphere_radius = stk::mesh::field_data(element_hydro_radius_field, sphere_element)[0];
            const bool overlap_exceeds_threshold =
                mundy::math::norm(node_coords) + sphere_radius > shifted_periphery_hydro_radius;
            if (overlap_exceeds_threshold) {
#pragma omp critical
              {
                std::cout << "Sphere node " << bulk_data.identifier(sphere_node)
                          << " overlaps with the periphery more than the allowable threshold." << std::endl;
                std::cout << "  node_coords: " << node_coords << std::endl;
                std::cout << "  norm(node_coords): " << mundy::math::norm(node_coords) << std::endl;
              }
              MUNDY_THROW_REQUIRE(false, std::runtime_error, "Sphere node outside hydrodynamic periphery.");
            }
          });
    } else if (periphery_hydro_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
      const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
      stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
      stk::mesh::Field<double> &element_hydro_radius_field = *element_radius_field_ptr_;
      double shifted_periphery_axis_radius1 = periphery_hydro_axis_radius1_ + maximum_allowed_periphery_overlap_;
      double shifted_periphery_axis_radius2 = periphery_hydro_axis_radius2_ + maximum_allowed_periphery_overlap_;
      double shifted_periphery_axis_radius3 = periphery_hydro_axis_radius3_ + maximum_allowed_periphery_overlap_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
          [&node_coord_field, &element_hydro_radius_field, &shifted_periphery_axis_radius1,
           &shifted_periphery_axis_radius2, &shifted_periphery_axis_radius3](const stk::mesh::BulkData &bulk_data,
                                                                             const stk::mesh::Entity &sphere_element) {
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            const double sphere_radius = stk::mesh::field_data(element_hydro_radius_field, sphere_element)[0];

            // The following is an in-exact but cheap check.
            // If shrinks the periphery by the maximum allowed overlap and the sphere radius and then checks if the
            // sphere is inside the shrunk periphery. Level sets don't follow the same rules as Euclidean geometry, so
            // this is a rough check.
            const double x = node_coords[0];
            const double y = node_coords[1];
            const double z = node_coords[2];
            const double x2 = x * x;
            const double y2 = y * y;
            const double z2 = z * z;
            const double a2 =
                (shifted_periphery_axis_radius1 - sphere_radius) * (shifted_periphery_axis_radius1 - sphere_radius);
            const double b2 =
                (shifted_periphery_axis_radius2 - sphere_radius) * (shifted_periphery_axis_radius2 - sphere_radius);
            const double c2 =
                (shifted_periphery_axis_radius3 - sphere_radius) * (shifted_periphery_axis_radius3 - sphere_radius);
            const double value = x2 / a2 + y2 / b2 + z2 / c2;
            if (value > 1.0) {
#pragma omp critical
              {
                std::cout << "Sphere node " << bulk_data.identifier(sphere_node)
                          << " overlaps with the periphery more than the allowable threshold." << std::endl;
                std::cout << "  node_coords: " << node_coords << std::endl;
                std::cout << "  value: " << value << std::endl;
              }
              MUNDY_THROW_REQUIRE(false, std::runtime_error, "Sphere node outside hydrodynamic periphery.");
            }
          });
    } else {
      MUNDY_THROW_REQUIRE(false, std::logic_error, "Invalid periphery type.");
    }
  }

  void compute_rpy_hydro() {
    // Before performing the hydro call, check if the spheres are within the periphery (optional)
    if (check_maximum_periphery_overlap_) {
      check_maximum_overlap_with_hydro_periphery();
    }

    Kokkos::Profiling::pushRegion("HP1::compute_rpy_hydro");
    const double viscosity = viscosity_;

    // Fetch the bucket of spheres to act on.
    stk::mesh::EntityVector sphere_elements;
    stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::get_selected_entities(chromatin_spheres_selector, bulk_data_ptr_->buckets(stk::topology::ELEMENT_RANK),
                                     sphere_elements);
    const size_t num_spheres = sphere_elements.size();

    // Copy the sphere positions, radii, forces, and velocities to Kokkos views
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions("sphere_positions", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii("sphere_radii", num_spheres);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces("sphere_forces", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities("sphere_velocities",
                                                                                    num_spheres * 3);

    auto sphere_positions_host = Kokkos::create_mirror_view(sphere_positions);
    auto sphere_radii_host = Kokkos::create_mirror_view(sphere_radii);
    auto sphere_forces_host = Kokkos::create_mirror_view(sphere_forces);
    auto sphere_velocities_host = Kokkos::create_mirror_view(sphere_velocities);

#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      const double *sphere_position = stk::mesh::field_data(*node_coord_field_ptr_, sphere_node);
      const double *sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere_element);
      const double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      const double *sphere_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_positions_host(i * 3 + j) = sphere_position[j];
        sphere_forces_host(i * 3 + j) = sphere_force[j];
        sphere_velocities_host(i * 3 + j) = sphere_velocity[j];
      }
      sphere_radii_host(i) = *sphere_radius;
    }

    Kokkos::deep_copy(sphere_positions, sphere_positions_host);
    Kokkos::deep_copy(sphere_radii, sphere_radii_host);
    Kokkos::deep_copy(sphere_forces, sphere_forces_host);
    Kokkos::deep_copy(sphere_velocities, sphere_velocities_host);

    // Apply the RPY kernel from spheres to spheres
    if (hydro_type_ == HYDRO_TYPE::RPYC) {
      mundy::alens::periphery::apply_rpyc_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, sphere_positions,
                                                 sphere_radii, sphere_radii, sphere_forces, sphere_velocities);
    } else if (hydro_type_ == HYDRO_TYPE::RPY) {
      mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, sphere_positions,
                                                sphere_radii, sphere_radii, sphere_forces, sphere_velocities);
    } else if (hydro_type_ == HYDRO_TYPE::STOKES) {
      mundy::alens::periphery::apply_stokes_kernel(DeviceExecutionSpace(), viscosity, sphere_positions,
                                                   sphere_positions, sphere_forces, sphere_velocities);
    }

    // If enabled, apply the correction for the no-slip boundary condition
    if (enable_periphery_hydrodynamics_) {
      const size_t num_surface_nodes = periphery_ptr_->get_num_nodes();
      auto surface_positions = periphery_ptr_->get_surface_positions();
      auto surface_weights = periphery_ptr_->get_quadrature_weights();
      auto surface_normals = periphery_ptr_->get_surface_normals();
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_radii("surface_radii", num_surface_nodes);
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_velocities("surface_velocities",
                                                                                       3 * num_surface_nodes);
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_forces("surface_forces",
                                                                                   3 * num_surface_nodes);
      Kokkos::deep_copy(surface_radii, 0.0);

      // Apply the RPY kernel from spheres to periphery
      if (hydro_type_ == HYDRO_TYPE::RPYC) {
        mundy::alens::periphery::apply_rpyc_kernel(DeviceExecutionSpace(), viscosity, sphere_positions,
                                                   surface_positions, sphere_radii, surface_radii, sphere_forces,
                                                   surface_velocities);
      } else if (hydro_type_ == HYDRO_TYPE::RPY) {
        mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions,
                                                  surface_positions, sphere_radii, surface_radii, sphere_forces,
                                                  surface_velocities);
      } else if (hydro_type_ == HYDRO_TYPE::STOKES) {
        mundy::alens::periphery::apply_stokes_kernel(DeviceExecutionSpace(), viscosity, sphere_positions,
                                                     surface_positions, sphere_forces, surface_velocities);
      }

      // Apply no-slip boundary conditions
      // This is done in two steps: first, we compute the forces on the periphery necessary to enforce no-slip
      // Then we evaluate the flow these forces induce on the spheres.
      periphery_ptr_->compute_surface_forces(surface_velocities, surface_forces);
      mundy::alens::periphery::apply_stokes_double_layer_kernel(
          DeviceExecutionSpace(), viscosity, num_surface_nodes, num_spheres, surface_positions, sphere_positions,
          surface_normals, surface_weights, surface_forces, sphere_velocities);

      // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
      // NOTE do not uncomment this, it is merely for a historical record to prevent people from adding it!
      // mundy::alens::periphery::apply_local_drag(DeviceExecutionSpace(), viscosity, sphere_velocities, sphere_forces,
      //                                           sphere_radii);
    }

    // Copy the sphere forces and velocities back to STK fields
    Kokkos::deep_copy(sphere_forces_host, sphere_forces);
    Kokkos::deep_copy(sphere_velocities_host, sphere_velocities);
#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      double *sphere_hydro_velocity = stk::mesh::field_data(*node_velocity_hydro_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_force[j] = sphere_forces_host(i * 3 + j);
        sphere_hydro_velocity[j] = sphere_velocities_host(i * 3 + j);
      }
    }
    Kokkos::Profiling::popRegion();
  }

  void compute_ellipsoidal_periphery_collision_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_ellipsoidal_periphery_collision_forces");
    const double spring_constant = periphery_collision_spring_constant_;
    const double a = periphery_collision_axis_radius1_;
    const double b = periphery_collision_axis_radius2_;
    const double c = periphery_collision_axis_radius3_;
    const double inv_a2 = 1.0 / (a * a);
    const double inv_b2 = 1.0 / (b * b);
    const double inv_c2 = 1.0 / (c * c);
    const mundy::math::Vector3<double> center(0.0, 0.0, 0.0);
    const auto orientation = mundy::math::Quaternion<double>::identity();
    auto level_set = [&inv_a2, &inv_b2, &inv_c2, &center,
                      &orientation](const mundy::math::Vector3<double> &point) -> double {
      // const auto body_frame_point = conjugate(orientation) * (point - center);
      const auto body_frame_point = point - center;
      return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
              body_frame_point[2] * body_frame_point[2] * inv_c2) -
             1;
    };

    // Fetch local references to the fields
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_force_field, &element_aabb_field, &element_radius_field, &level_set, &center,
         &orientation, &a, &b, &c,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          // For our coarse search, we check if the coners of the sphere's aabb lie inside the ellipsoidal periphery
          // This can be done via the (body frame) inside outside unftion f(x, y, z) = 1 - (x^2/a^2 + y^2/b^2 + z^2/c^2)
          // This is possible due to the convexity of the ellipsoid
          const double *sphere_aabb = stk::mesh::field_data(element_aabb_field, sphere_element);
          const double x0 = sphere_aabb[0];
          const double y0 = sphere_aabb[1];
          const double z0 = sphere_aabb[2];
          const double x1 = sphere_aabb[3];
          const double y1 = sphere_aabb[4];
          const double z1 = sphere_aabb[5];

          // Compute all 8 corners of the AABB
          const auto bottom_left_front = mundy::math::Vector3<double>(x0, y0, z0);
          const auto bottom_right_front = mundy::math::Vector3<double>(x1, y0, z0);
          const auto top_left_front = mundy::math::Vector3<double>(x0, y1, z0);
          const auto top_right_front = mundy::math::Vector3<double>(x1, y1, z0);
          const auto bottom_left_back = mundy::math::Vector3<double>(x0, y0, z1);
          const auto bottom_right_back = mundy::math::Vector3<double>(x1, y0, z1);
          const auto top_left_back = mundy::math::Vector3<double>(x0, y1, z1);
          const auto top_right_back = mundy::math::Vector3<double>(x1, y1, z1);
          const double all_points_inside_periphery =
              level_set(bottom_left_front) < 0.0 && level_set(bottom_right_front) < 0.0 &&
              level_set(top_left_front) < 0.0 && level_set(top_right_front) < 0.0 &&
              level_set(bottom_left_back) < 0.0 && level_set(bottom_right_back) < 0.0 &&
              level_set(top_left_back) < 0.0 && level_set(top_right_back) < 0.0;

          if (!all_points_inside_periphery) {
            // We might have a collision, perform the more expensive check
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

            // Note, the ellipsoid for the ssd calc has outward normal, whereas the periphery has inward normal.
            // Hence, the sign flip.
            mundy::math::Vector3<double> contact_point;
            mundy::math::Vector3<double> ellipsoid_nhat;
            const double shared_normal_ssd =
                -mundy::math::distance::shared_normal_ssd_between_ellipsoid_and_point(
                    center, orientation, a, b, c, node_coords, contact_point, ellipsoid_nhat) -
                sphere_radius;

            if (shared_normal_ssd < 0.0) {
              // We have a collision, compute the force
              auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);
              auto periphery_nhat = -ellipsoid_nhat;
              node_force[0] -= spring_constant * periphery_nhat[0] * shared_normal_ssd;
              node_force[1] -= spring_constant * periphery_nhat[1] * shared_normal_ssd;
              node_force[2] -= spring_constant * periphery_nhat[2] * shared_normal_ssd;
            }
          }
        });
    Kokkos::Profiling::popRegion();
  }

  void compute_ellipsoidal_periphery_collision_forces_fast_approximate() {
    Kokkos::Profiling::pushRegion("HP1::compute_ellipsoidal_periphery_collision_forces_fast_approximate");
    const double spring_constant = periphery_collision_spring_constant_;
    // Adjust for our standoff distance
    const double a = periphery_collision_axis_radius1_;
    const double b = periphery_collision_axis_radius2_;
    const double c = periphery_collision_axis_radius3_;
    const mundy::math::Vector3<double> center(0.0, 0.0, 0.0);
    const auto orientation = mundy::math::Quaternion<double>::identity();
    auto level_set = [&a, &b, &c, &center, &orientation](const double &radius,
                                                         const mundy::math::Vector3<double> &point) -> double {
      // const auto body_frame_point = conjugate(orientation) * (point - center);
      const auto body_frame_point = point - center;
      const double inv_a2 = 1.0 / ((a - radius) * (a - radius));
      const double inv_b2 = 1.0 / ((b - radius) * (b - radius));
      const double inv_c2 = 1.0 / ((c - radius) * (c - radius));
      return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
              body_frame_point[2] * body_frame_point[2] * inv_c2) -
             1;
    };
    // Fast compute of the outward 'normal' at the point
    auto outward_normal = [&a, &b, &c, &center, &orientation](
                              const double &radius,
                              const mundy::math::Vector3<double> &point) -> mundy::math::Vector3<double> {
      const auto body_frame_point = point - center;
      const double inv_a2 = 1.0 / ((a - radius) * (a - radius));
      const double inv_b2 = 1.0 / ((b - radius) * (b - radius));
      const double inv_c2 = 1.0 / ((c - radius) * (c - radius));
      return mundy::math::Vector3<double>(2.0 * body_frame_point[0] * inv_a2, 2.0 * body_frame_point[1] * inv_b2,
                                          2.0 * body_frame_point[2] * inv_c2);
    };

    // Fetch local references to the fields
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_force_field, &element_radius_field, &level_set, &outward_normal, &center,
         &orientation, &a, &b, &c,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          // Do a fast loop over all of the spheres we are checking, e.g., brute-force the calc.
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

          // Simply check if we are outside the sphere via the level-set function
          if (level_set(sphere_radius, node_coords) > 0.0) {
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);

            // Compute the outward normal
            auto out_normal = outward_normal(sphere_radius, node_coords);
            node_force[0] -= spring_constant * out_normal[0];
            node_force[1] -= spring_constant * out_normal[1];
            node_force[2] -= spring_constant * out_normal[2];
          }
        });
    Kokkos::Profiling::popRegion();
  }

  void compute_spherical_periphery_collision_forces() {
    const double spring_constant = periphery_collision_spring_constant_;
    const double periphery_collision_radius = periphery_collision_radius_;

    // Fetch local references to the fields
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_force_field, &element_aabb_field, &element_radius_field, &periphery_collision_radius,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);

          const double node_coords_norm = mundy::math::two_norm(node_coords);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
          const double shared_normal_ssd = periphery_collision_radius - node_coords_norm - sphere_radius;
          const bool sphere_collides_with_periphery = shared_normal_ssd < 0.0;
          if (sphere_collides_with_periphery) {
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);
            auto inward_normal = -node_coords / node_coords_norm;
            node_force[0] -= spring_constant * inward_normal[0] * shared_normal_ssd;
            node_force[1] -= spring_constant * inward_normal[1] * shared_normal_ssd;
            node_force[2] -= spring_constant * inward_normal[2] * shared_normal_ssd;
          }
        });
  }

  void compute_spherical_periphery_collision_forces_fast_approximate() {
    const double periphery_collision_radius = periphery_collision_radius_;
    // Fetch local references to the fields
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_force_field, &element_aabb_field, &element_radius_field, &periphery_collision_radius](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
          const double node_coords_norm = mundy::math::two_norm(node_coords);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
          const double shared_normal_ssd = periphery_collision_radius - node_coords_norm - sphere_radius;
          const bool sphere_collides_with_periphery = shared_normal_ssd < 0.0;
          if (sphere_collides_with_periphery) {
            // Just shift the nodes along the normal to no longer overlap
            auto inward_normal = -node_coords / node_coords_norm;
            node_coords -= inward_normal * shared_normal_ssd;
          }
        });
  }

  void compute_periphery_collision_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_periphery_collision_forces");
    if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
      if (periphery_collision_use_fast_approx_) {
        compute_spherical_periphery_collision_forces_fast_approximate();
      } else {
        compute_spherical_periphery_collision_forces();
      }
    } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
      if (periphery_collision_use_fast_approx_) {
        compute_ellipsoidal_periphery_collision_forces_fast_approximate();
      } else {
        compute_ellipsoidal_periphery_collision_forces();
      }
    } else {
      MUNDY_THROW_REQUIRE(false, std::logic_error, "Invalid periphery type.");
    }
    Kokkos::Profiling::popRegion();
  }

  void compute_euchromatin_active_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_euchromatin_active_forces");

    // We are going to do the forces as such.
    // nhat is the unit director along the segment.
    // sigma is the force density we are applying
    // F = f nhat
    // sigma = f * n --> f = sigma / n
    // F = sigma / n * nhat
    // NOTE: CJE we are going to throw out a factor of 1/n to match what Achal did.

    // Selectors and aliases
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Field<unsigned> &euchromatin_state = *euchromatin_state_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const double &active_force_sigma = active_euchromatin_force_sigma_;

    // Loop over the euchromatin spring elements and decide if they switch to the active state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, ee_springs_part,
        [&euchromatin_state, &node_coord_field, &node_force_field, &active_force_sigma](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &euchromatin_spring) {
          // We are not going to increment the elapsed time ourselves, but rely on someone outside of this loop to do
          // that at the end of a timestep, in order to keep it consistent with the total elapsed time in the system.
          unsigned *current_state = stk::mesh::field_data(euchromatin_state, euchromatin_spring);

          if (current_state[0] == 1u) {
            // Fetch the connected nodes
            const stk::mesh::Entity *nodes = bulk_data.begin_nodes(euchromatin_spring);
            const stk::mesh::Entity &node1 = nodes[0];
            const stk::mesh::Entity &node2 = nodes[1];
            const double *node1_coord = stk::mesh::field_data(node_coord_field, node1);
            const double *node2_coord = stk::mesh::field_data(node_coord_field, node2);

            // Calculate the force on each node from the above equation, which winds up
            // F = sigma / n / n * nvec ----> sigma / n^2 * nvec
            // NOTE: CJE we are going to throw out a factor of 1/n to match what Achal did.
            const double nvec[3] = {node2_coord[0] - node1_coord[0], node2_coord[1] - node1_coord[1],
                                    node2_coord[2] - node1_coord[2]};
            const double nsqr = nvec[0] * nvec[0] + nvec[1] * nvec[1] + nvec[2] * nvec[2];
            const double right_node_force[3] = {active_force_sigma / std::sqrt(nsqr) * nvec[0],
                                                active_force_sigma / std::sqrt(nsqr) * nvec[1],
                                                active_force_sigma / std::sqrt(nsqr) * nvec[2]};

            // Add the force dipole to the nodes.
            double *node1_force = stk::mesh::field_data(node_force_field, node1);
            double *node2_force = stk::mesh::field_data(node_force_field, node2);

#pragma omp atomic
            node1_force[0] -= right_node_force[0];
#pragma omp atomic
            node1_force[1] -= right_node_force[1];
#pragma omp atomic
            node1_force[2] -= right_node_force[2];
#pragma omp atomic
            node2_force[0] += right_node_force[0];
#pragma omp atomic
            node2_force[1] += right_node_force[1];
#pragma omp atomic
            node2_force[2] += right_node_force[2];
          }
        });
    // Sum the forces on shared nodes.
    stk::mesh::parallel_sum(*bulk_data_ptr_, {node_force_field_ptr_});

    Kokkos::Profiling::popRegion();
  }

  void compute_hertzian_contact_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_hertzian_contact_forces");

    // Potential evaluation (Hertzian contact)
    auto backbone_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);

    compute_ssd_and_cn_ptr_->execute(backbone_backbone_neighbor_genx_selector);
    evaluate_linker_potentials_ptr_->execute(backbone_backbone_neighbor_genx_selector);
    linker_potential_force_reduction_ptr_->execute(backbone_selector);

    // Apply a correction force to the endpoints of the chain segments
    //
    // When three segments connect, even though the collision force between segments that share a node will be zero,
    // the two segments on either end will collide if the segement in the middle shrinks too much. This is a behavior
    // which we want to be applied consistantly to all segments. However, the segments at the beginning and end of the
    // chain will not have a segment on one side, so we need to apply a correction force to account for this.
    //
    // Assume hertzian contact force.
    if (backbone_collision_type_ == COLLISION_TYPE::HERTZIAN) {
      const double backbone_excluded_volume_radius = backbone_excluded_volume_radius_;
      const double backbone_youngs_modulus = backbone_youngs_modulus_;
      const double backbone_poissons_ratio = backbone_poissons_ratio_;
      const double effective_radius = (backbone_excluded_volume_radius * backbone_excluded_volume_radius) /
                                      (backbone_excluded_volume_radius + backbone_excluded_volume_radius);
      const double effective_youngs_modulus =
          (backbone_youngs_modulus * backbone_youngs_modulus) /
          (backbone_youngs_modulus - backbone_youngs_modulus * backbone_poissons_ratio * backbone_poissons_ratio +
           backbone_youngs_modulus - backbone_youngs_modulus * backbone_poissons_ratio * backbone_poissons_ratio);
      stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
      stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
      stk::mesh::Field<double> &element_requires_endpoint_correction_field =
          *element_requires_endpoint_correction_field_ptr_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEM_RANK, backbone_selector,
          [&backbone_excluded_volume_radius, &effective_radius, &effective_youngs_modulus, &node_coord_field,
           &node_force_field, &element_requires_endpoint_correction_field](
              [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &seg) {
            if (static_cast<bool>(stk::mesh::field_data(element_requires_endpoint_correction_field, seg)[0])) {
              // Get the segment length and tangent
              const stk::mesh::Entity *nodes = bulk_data.begin_nodes(seg);
              const stk::mesh::Entity &node1 = nodes[0];
              const stk::mesh::Entity &node2 = nodes[1];
              const double *node1_coords = stk::mesh::field_data(node_coord_field, node1);
              const double *node2_coords = stk::mesh::field_data(node_coord_field, node2);
              const double segment_length =
                  std::sqrt((node2_coords[0] - node1_coords[0]) * (node2_coords[0] - node1_coords[0]) +
                            (node2_coords[1] - node1_coords[1]) * (node2_coords[1] - node1_coords[1]) +
                            (node2_coords[2] - node1_coords[2]) * (node2_coords[2] - node1_coords[2]));
              const double segment_tangent[3] = {(node2_coords[0] - node1_coords[0]) / segment_length,
                                                 (node2_coords[1] - node1_coords[1]) / segment_length,
                                                 (node2_coords[2] - node1_coords[2]) / segment_length};

              // Get the amount of overlap
              const double signed_sep = segment_length - 2.0 * backbone_excluded_volume_radius;
              const bool particles_overlap = signed_sep < 0.0;
              const double normal_force_magnitude = particles_overlap
                                                        ? (4.0 / 3.0) * effective_youngs_modulus *
                                                              std::sqrt(effective_radius) * std::pow(-signed_sep, 1.5)
                                                        : 0.0;

              // Apply the force to the nodes
              double *node1_force = stk::mesh::field_data(node_force_field, node1);
              double *node2_force = stk::mesh::field_data(node_force_field, node2);

#pragma omp atomic
              node1_force[0] -= normal_force_magnitude * segment_tangent[0];
#pragma omp atomic
              node1_force[1] -= normal_force_magnitude * segment_tangent[1];
#pragma omp atomic
              node1_force[2] -= normal_force_magnitude * segment_tangent[2];

#pragma omp atomic
              node2_force[0] += normal_force_magnitude * segment_tangent[0];
#pragma omp atomic
              node2_force[1] += normal_force_magnitude * segment_tangent[1];
#pragma omp atomic
              node2_force[2] += normal_force_magnitude * segment_tangent[2];
            }
          });
    } else if (backbone_collision_type_ == COLLISION_TYPE::WCA) {
      stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
      stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
      stk::mesh::Field<double> &element_requires_endpoint_correction_field =
          *element_requires_endpoint_correction_field_ptr_;
      const double wca_epsilon = backbone_wca_epsilon_;
      const double wca_sigma = backbone_wca_sigma_;
      const double wca_wca_cutoff = backbone_wca_cutoff_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEM_RANK, backbone_selector,
          [&node_coord_field, &wca_epsilon, &wca_sigma, &wca_wca_cutoff, &node_force_field,
           &element_requires_endpoint_correction_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                        const stk::mesh::Entity &seg) {
            if (static_cast<bool>(stk::mesh::field_data(element_requires_endpoint_correction_field, seg)[0])) {
              // Get the segment length and tangent
              const stk::mesh::Entity *nodes = bulk_data.begin_nodes(seg);
              const stk::mesh::Entity &node1 = nodes[0];
              const stk::mesh::Entity &node2 = nodes[1];
              const double *node1_coords = stk::mesh::field_data(node_coord_field, node1);
              const double *node2_coords = stk::mesh::field_data(node_coord_field, node2);
              const double segment_length =
                  std::sqrt((node2_coords[0] - node1_coords[0]) * (node2_coords[0] - node1_coords[0]) +
                            (node2_coords[1] - node1_coords[1]) * (node2_coords[1] - node1_coords[1]) +
                            (node2_coords[2] - node1_coords[2]) * (node2_coords[2] - node1_coords[2]));
              const double segment_tangent[3] = {(node2_coords[0] - node1_coords[0]) / segment_length,
                                                 (node2_coords[1] - node1_coords[1]) / segment_length,
                                                 (node2_coords[2] - node1_coords[2]) / segment_length};

              double lennard_jones_force = 0.0;
              if (segment_length < wca_wca_cutoff) {
                const double rho2 = wca_sigma * wca_sigma / (segment_length * segment_length);
                const double rho6 = rho2 * rho2 * rho2;
                const double rho12 = rho6 * rho6;
                lennard_jones_force = 24.0 * wca_epsilon * (2.0 * rho12 - rho6) / (segment_length);
              }

              // Apply the force to the nodes
              double *node1_force = stk::mesh::field_data(node_force_field, node1);
              double *node2_force = stk::mesh::field_data(node_force_field, node2);

#pragma omp atomic
              node1_force[0] -= lennard_jones_force * segment_tangent[0];
#pragma omp atomic
              node1_force[1] -= lennard_jones_force * segment_tangent[1];
#pragma omp atomic
              node1_force[2] -= lennard_jones_force * segment_tangent[2];

#pragma omp atomic
              node2_force[0] += lennard_jones_force * segment_tangent[0];
#pragma omp atomic
              node2_force[1] += lennard_jones_force * segment_tangent[1];
#pragma omp atomic
              node2_force[2] += lennard_jones_force * segment_tangent[2];
            }
          });
    }

    Kokkos::Profiling::popRegion();
  }

  void compute_backbone_harmonic_bond_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_backbone_harmonic_bond_forces");

    auto backbone_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    compute_constraint_forcing_ptr_->execute(backbone_selector);

    Kokkos::Profiling::popRegion();
  }

  void compute_crosslinker_harmonic_bond_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_crosslinker_harmonic_bond_forces");

    // Select only active springs in the system. Aka, not left bound.
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto left_hp1_selector = stk::mesh::Selector(*left_hp1_part_ptr_);
    auto actively_bound_springs = hp1_selector - left_hp1_selector;
    compute_constraint_forcing_ptr_->execute(actively_bound_springs);

    Kokkos::Profiling::popRegion();
  }

  void compute_brownian_velocity() {
    // Compute the velocity due to brownian motion
    Kokkos::Profiling::pushRegion("HP1::compute_brownian_velocity");

    // Selectors and aliases
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<unsigned> &node_rng_field = *node_rng_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    double &timestep_size = timestep_size_;
    double &kt = brownian_kt_;
    double sphere_drag_coeff = 6.0 * M_PI * viscosity_ * backbone_sphere_hydrodynamic_radius_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_velocity_field, &node_rng_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff, &kt](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_field, sphere_node);

          // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * kt * sphere_drag_coeff / timestep_size) * inv_drag_coeff;
          node_velocity[0] += coeff * rng.randn<double>();
          node_velocity[1] += coeff * rng.randn<double>();
          node_velocity[2] += coeff * rng.randn<double>();
          node_rng_counter[0]++;
        });

    Kokkos::Profiling::popRegion();
  }

  void compute_dry_velocity() {
    // Compute both the dry velocity due to external forces
    Kokkos::Profiling::pushRegion("HP1::compute_dry_velocity");

    // Selectors and aliases
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    double &timestep_size = timestep_size_;
    double sphere_drag_coeff = 6.0 * M_PI * viscosity_ * backbone_sphere_hydrodynamic_radius_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_velocity_field, &node_force_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          double *node_force = stk::mesh::field_data(node_force_field, sphere_node);

          // Uext = Fext * inv_drag_coeff
          node_velocity[0] += node_force[0] * inv_drag_coeff;
          node_velocity[1] += node_force[1] * inv_drag_coeff;
          node_velocity[2] += node_force[2] * inv_drag_coeff;
        });

    Kokkos::Profiling::popRegion();
  }

  void check_maximum_speed_pre_position_update() {
    // Selectors and aliases
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    double max_allowable_speed = max_allowable_speed_;
    bool maximum_speed_exceeded = false;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_velocity_field, &max_allowable_speed, &maximum_speed_exceeded](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, sphere_node);
          const auto speed = mundy::math::norm(node_velocity);
          if (speed > max_allowable_speed) {
            maximum_speed_exceeded = true;
          }
        });

    MUNDY_THROW_REQUIRE(!maximum_speed_exceeded, std::runtime_error,
                        fmt::format("Maximum speed exceeded on timestep {}", timestep_index_));
  }

  void update_positions() {
    // Check to see if the maximum speed is exceeded before updating the positions
    if (check_maximum_speed_pre_position_update_) {
      check_maximum_speed_pre_position_update();
    }

    Kokkos::Profiling::pushRegion("HP1::update_positions");

    // Selectors and aliases
    size_t &timestep_index = timestep_index_;
    double &timestep_size = timestep_size_;
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;

    // Update the positions for all spheres based on velocity
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_velocity_field, &timestep_size, &timestep_index](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_coord = stk::mesh::field_data(node_coord_field, sphere_node);
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);

          // x(t+dt) = x(t) + dt * v(t)
          node_coord[0] += timestep_size * node_velocity[0];
          node_coord[1] += timestep_size * node_velocity[1];
          node_coord[2] += timestep_size * node_velocity[2];
        });

    Kokkos::Profiling::popRegion();
  }

  void run(int argc, char **argv) {
    // Preprocess
    parse_user_inputs(argc, argv);
    dump_user_inputs();

    // Setup
    Kokkos::Profiling::pushRegion("HP1::Setup");
    build_our_mesh_and_method_instances();

    fetch_fields_and_parts();
    instantiate_metamethods();
    set_mutable_parameters();
    declare_and_initialize_hp1();
    if (enable_periphery_hydrodynamics_) {
      initialize_hydrodynamic_periphery();
    }
    if (enable_periphery_binding_ && !restart_performed_) {
      declare_and_initialize_periphery_bind_sites();
    }
    if (enable_active_euchromatin_forces_ && !restart_performed_) {
      initialize_euchromatin();
    }

    std::cout << "Finished setup." << std::endl;
    detect_neighbors_initial();
    std::cout << "Finished initial neighbor detection." << std::endl;
    Kokkos::Profiling::popRegion();

    // Post setup
    Kokkos::Profiling::pushRegion("HP1::Loadbalance");
    if (loadbalance_post_initialization_) {
      loadbalance();
    }
    Kokkos::Profiling::popRegion();

    // Reset simulation control variables
    timestep_index_ = 0;
    timestep_current_time_ = 0.0;
    if (enable_continuation_if_available_ && restart_performed_) {
      timestep_index_ = restart_timestep_index_;
      timestep_current_time_ = restart_timestep_index_ * timestep_size_;
      // Force an update of the positions to pretend we are at the END of the restarted timestep
      update_positions();
      // Update the time for the euchromatin active forces
      if (enable_active_euchromatin_forces_) update_euchromatin_state_time();
      // Increment the index to prevent the off-by-1 error
      timestep_index_ += 1;
      timestep_current_time_ += timestep_size_;
    }

    // Check to see if we need to do anything for compressing the system.
    if (enable_periphery_collision_ && shrink_periphery_over_time_) {
      if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
        periphery_collision_radius_ *= periphery_collision_scale_factor_before_shrinking_;
      } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
        periphery_collision_axis_radius1_ *= periphery_collision_scale_factor_before_shrinking_;
        periphery_collision_axis_radius2_ *= periphery_collision_scale_factor_before_shrinking_;
        periphery_collision_axis_radius3_ *= periphery_collision_scale_factor_before_shrinking_;
      }
    }

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    Kokkos::Timer overall_timer;
    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion("MainLoop");
    // We have pre-loaded the starting index and time...
    for (; timestep_index_ < num_time_steps_; timestep_index_++, timestep_current_time_ += timestep_size_) {
      // Prepare the current configuration.
      Kokkos::Profiling::pushRegion("HP1::PrepareStep");
      zero_out_transient_node_fields();
      zero_out_transient_element_fields();
      zero_out_transient_constraint_fields();
      rotate_field_states();
      Kokkos::Profiling::popRegion();

      // If we are doing a compression run, shrink the periphery
      if (enable_periphery_collision_ && shrink_periphery_over_time_) {
        const double shrink_factor = std::pow(1.0 / periphery_collision_scale_factor_before_shrinking_,
                                              1.0 / periphery_collision_shrinkage_num_steps_);
        if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
          periphery_collision_radius_ *= shrink_factor;
        } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
          periphery_collision_axis_radius1_ *= shrink_factor;
          periphery_collision_axis_radius2_ *= shrink_factor;
          periphery_collision_axis_radius3_ *= shrink_factor;
        }
      }

      // Detect sphere-sphere and crosslinker-sphere neighbors
      update_neighbor_list_ = false;
      detect_neighbors();

      // Determine KMC events
      if (enable_crosslinkers_) update_crosslinker_state();
      if (enable_active_euchromatin_forces_) update_active_euchromatin_state();

      // Evaluate forces f(x(t)).
      if (enable_backbone_collision_) compute_hertzian_contact_forces();
      if (enable_backbone_springs_) compute_backbone_harmonic_bond_forces();
      if (enable_crosslinkers_) compute_crosslinker_harmonic_bond_forces();
      if (enable_periphery_collision_) compute_periphery_collision_forces();
      if (enable_active_euchromatin_forces_) compute_euchromatin_active_forces();

      // Compute velocities.
      if (enable_chromatin_brownian_motion_) compute_brownian_velocity();
      if (enable_backbone_n_body_hydrodynamics_) {
        // RPY hydro is only the long range hydrodynamics. We still need to compute the self-interaction.
        compute_dry_velocity();

        stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
        stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
        stk::mesh::Field<double> &node_velocity_hydro_field = *node_velocity_hydro_field_ptr_;
        stk::mesh::Field<double> &node_velocity_hydro_old_field = *node_velocity_hydro_old_field_ptr_;
        if (timestep_index_ % hydro_update_frequency_ == 0) {
          // Copy the current hydro velocity to the previous hydro velocity
          if (hydro_update_frequency_ != 1) {
            mundy::mesh::for_each_entity_run(
                *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
                [&node_velocity_hydro_field, &node_velocity_hydro_old_field](
                    [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
                  double *node_velocity_hydro = stk::mesh::field_data(node_velocity_hydro_field, sphere_node);
                  double *node_velocity_hydro_old = stk::mesh::field_data(node_velocity_hydro_old_field, sphere_node);
                  node_velocity_hydro_old[0] = node_velocity_hydro[0];
                  node_velocity_hydro_old[1] = node_velocity_hydro[1];
                  node_velocity_hydro_old[2] = node_velocity_hydro[2];
                });
          }

          compute_rpy_hydro();

          // Compute the finite difference regurized l2 norm of the change in the hydro velocity
          if (hydro_update_frequency_ != 1) {
            double hydro_velocity_change_norm = 0.0;
            mundy::mesh::for_each_entity_run(
                *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
                [&node_velocity_hydro_field, &node_velocity_hydro_old_field, &hydro_velocity_change_norm](
                    [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
                  double *node_velocity_hydro = stk::mesh::field_data(node_velocity_hydro_field, sphere_node);
                  double *node_velocity_hydro_old = stk::mesh::field_data(node_velocity_hydro_old_field, sphere_node);
                  const double dx = node_velocity_hydro[0] - node_velocity_hydro_old[0];
                  const double dy = node_velocity_hydro[1] - node_velocity_hydro_old[1];
                  const double dz = node_velocity_hydro[2] - node_velocity_hydro_old[2];
#pragma omp atomic
                  hydro_velocity_change_norm += dx * dx + dy * dy + dz * dz;
                });

            size_t len_v = 3 * num_chromosomes_ * num_chromatin_repeats_ *
                           (num_euchromatin_per_repeat_ + num_heterochromatin_per_repeat_);
            hydro_velocity_change_norm = std::sqrt(hydro_velocity_change_norm / len_v);
            std::cout << "||v_new - v_old||_2 / sqrt(len(v_old)): " << hydro_velocity_change_norm << std::endl;
          }
        }

        // Sum the hydro velocity into the total velocity
        mundy::mesh::for_each_entity_run(
            *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
            [&node_velocity_field, &node_velocity_hydro_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                               const stk::mesh::Entity &sphere_node) {
              double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
              double *node_velocity_hydro = stk::mesh::field_data(node_velocity_hydro_field, sphere_node);
              node_velocity[0] += node_velocity_hydro[0];
              node_velocity[1] += node_velocity_hydro[1];
              node_velocity[2] += node_velocity_hydro[2];
            });
      } else {
        compute_dry_velocity();
      }

      // Logging, if desired, write to console
      Kokkos::Profiling::pushRegion("HP1::Logging");
      if (timestep_index_ % log_frequency_ == 0) {
        if (bulk_data_ptr_->parallel_rank() == 0) {
          double tps = static_cast<double>(log_frequency_) / static_cast<double>(timer.seconds());
          std::cout << "Step: " << std::setw(15) << timestep_index_ << ", tps: " << std::setprecision(15) << tps;
          if (enable_periphery_collision_ && shrink_periphery_over_time_) {
            if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
              std::cout << ", periphery_collision_radius: " << periphery_collision_radius_;
            } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
              std::cout << ", periphery_collision_axis_radius1: " << periphery_collision_axis_radius1_
                        << ", periphery_collision_axis_radius2: " << periphery_collision_axis_radius2_
                        << ", periphery_collision_axis_radius3: " << periphery_collision_axis_radius3_;
            }
          }
          std::cout << std::endl;
          timer.reset();
        }
      }
      Kokkos::Profiling::popRegion();

      // IO. If desired, write out the data for time t (STK or mundy)
      if (timestep_index_ % io_frequency_ == 0) {
        bulk_data_ptr_->sort_entities(z_morton_sorter_);

        Kokkos::Profiling::pushRegion("HP1::IO");
        io_broker_ptr_->write_io_broker_timestep(static_cast<int>(timestep_index_), timestep_current_time_);
        Kokkos::Profiling::popRegion();
      }

      // Update positions. x(t + dt) = x(t) + dt * v(t).
      update_positions();

      // Update the time for the euchromatin active forces
      if (enable_active_euchromatin_forces_) update_euchromatin_state_time();
    }
    Kokkos::Profiling::popRegion();

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep =
          static_cast<double>(overall_timer.seconds()) / static_cast<double>(num_time_steps_);
      double tps = 1.0 / avg_time_per_timestep;
      std::cout << "******************Final statistics (Rank 0)**************\n";
      if (print_neighborlist_statistics_) {
        std::cout << "****************\n";
        std::cout << "Neighbor list statistics\n";
        for (auto &neighborlist_entry : neighborlist_update_steps_times_) {
          auto [timestep, elasped_step, elapsed_time] = neighborlist_entry;
          auto tps_nl = static_cast<double>(elasped_step) / elapsed_time;
          std::cout << "  Rebuild timestep: " << timestep << ", elapsed_steps: " << elasped_step
                    << ", elapsed_time: " << elapsed_time << ", tps: " << tps_nl << std::endl;
        }
      }
      std::cout << "****************\n";
      std::cout << "Simulation statistics\n";
      std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
      std::cout << "Timesteps per second: " << std::setprecision(15) << tps << std::endl;
    }
  }

 private:
  //! \name Useful aliases
  //@{

  static constexpr auto node_rank_ = stk::topology::NODE_RANK;
  static constexpr auto edge_rank_ = stk::topology::EDGE_RANK;
  static constexpr auto element_rank_ = stk::topology::ELEMENT_RANK;
  static constexpr auto constraint_rank_ = stk::topology::CONSTRAINT_RANK;
  //@}

  //! \name Internal state
  //@{

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_;
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr_;
  std::shared_ptr<mundy::io::IOBroker> io_broker_ptr_ = nullptr;
  size_t timestep_index_;
  size_t restart_timestep_index_;
  double timestep_current_time_;
  std::shared_ptr<mundy::alens::periphery::Periphery> periphery_ptr_;
  bool restart_performed_ = false;
  //@}

  //! \name Neighborlist rebuild information
  //@{

  size_t last_neighborlist_update_step_ = 0;
  Kokkos::Timer neighborlist_update_timer_;
  std::vector<std::tuple<size_t, size_t, double>>
      neighborlist_update_steps_times_;  // [timestep, elapsed_timesteps, elapsed_time]
  bool update_neighbor_list_;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<unsigned> *node_rng_field_ptr_;
  stk::mesh::Field<double> *node_coord_field_ptr_;
  stk::mesh::Field<double> *node_velocity_field_ptr_;
  stk::mesh::Field<double> *node_force_field_ptr_;
  stk::mesh::Field<double> *node_velocity_hydro_field_ptr_;
  stk::mesh::Field<double> *node_velocity_hydro_old_field_ptr_;

  stk::mesh::Field<unsigned> *element_rng_field_ptr_;
  stk::mesh::Field<double> *element_radius_field_ptr_;
  stk::mesh::Field<double> *element_spring_constant_field_ptr_;
  stk::mesh::Field<double> *element_spring_r0_field_ptr_;
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr_;
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr_;
  stk::mesh::Field<double> *element_aabb_field_ptr_;
  stk::mesh::Field<double> *element_corner_displacement_field_ptr_;
  stk::mesh::Field<double> *element_binding_rates_field_ptr_;
  stk::mesh::Field<double> *element_unbinding_rates_field_ptr_;
  stk::mesh::Field<unsigned> *element_perform_state_change_field_ptr_;
  stk::mesh::Field<unsigned> *element_chainid_field_ptr_;
  stk::mesh::Field<double> *element_requires_endpoint_correction_field_ptr_;

  stk::mesh::Field<unsigned> *euchromatin_state_field_ptr_;
  stk::mesh::Field<unsigned> *euchromatin_perform_state_change_field_ptr_;
  stk::mesh::Field<double> *euchromatin_state_change_next_time_field_ptr_;
  stk::mesh::Field<double> *euchromatin_state_change_elapsed_time_field_ptr_;

  stk::mesh::Field<double> *constraint_potential_force_field_ptr_;
  stk::mesh::Field<double> *constraint_state_change_rate_field_ptr_;
  stk::mesh::Field<unsigned> *constraint_perform_state_change_field_ptr_;
  stk::mesh::Field<int> *constraint_linked_entity_owners_field_ptr_;
  mundy::linkers::LinkedEntitiesFieldType *constraint_linked_entities_field_ptr_;
  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *spheres_part_ptr_ = nullptr;
  stk::mesh::Part *e_part_ptr_ = nullptr;
  stk::mesh::Part *h_part_ptr_ = nullptr;
  stk::mesh::Part *bs_part_ptr_ = nullptr;

  stk::mesh::Part *hp1_part_ptr_ = nullptr;
  stk::mesh::Part *left_hp1_part_ptr_ = nullptr;
  stk::mesh::Part *doubly_hp1_h_part_ptr_ = nullptr;
  stk::mesh::Part *doubly_hp1_bs_part_ptr_ = nullptr;

  stk::mesh::Part *backbone_segments_part_ptr_ = nullptr;
  stk::mesh::Part *ee_springs_part_ptr_ = nullptr;
  stk::mesh::Part *ee_springs_active_part_ptr_ = nullptr;
  stk::mesh::Part *ee_springs_inactive_part_ptr_ = nullptr;
  stk::mesh::Part *eh_springs_part_ptr_ = nullptr;
  stk::mesh::Part *hh_springs_part_ptr_ = nullptr;

  stk::mesh::Part *backbone_backbone_neighbor_genx_part_ptr_ = nullptr;
  stk::mesh::Part *hp1_h_neighbor_genx_part_ptr_ = nullptr;
  stk::mesh::Part *hp1_bs_neighbor_genx_part_ptr_ = nullptr;
  //@}

  //! \name MetaMethod instances
  //@{

  std::shared_ptr<mundy::meta::MetaMethodExecutionInterface<void>> declare_and_init_constraints_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_aabb_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_constraint_forcing_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> compute_ssd_and_cn_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> evaluate_linker_potentials_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> linker_potential_force_reduction_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> destroy_neighbor_linkers_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> destroy_bound_neighbor_linkers_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> generate_scs_scs_genx_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> generate_hp1_h_genx_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>> generate_hp1_bs_genx_ptr_;
  //@}

  //! \name Fixed params for MetaMethods
  //@{

  Teuchos::ParameterList compute_constraint_forcing_fixed_params_;
  Teuchos::ParameterList compute_ssd_and_cn_fixed_params_;
  Teuchos::ParameterList compute_aabb_fixed_params_;
  Teuchos::ParameterList generate_scs_scs_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList generate_hp1_h_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList generate_hp1_bs_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList evaluate_linker_potentials_fixed_params_;
  Teuchos::ParameterList linker_potential_force_reduction_fixed_params_;
  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList destroy_bound_neighbor_linkers_fixed_params_;
  //@}

  //! \name Partitioning settings
  //@{

  class RcbSettings : public stk::balance::BalanceSettings {
   public:
    RcbSettings() {
    }
    virtual ~RcbSettings() {
    }

    virtual bool isIncrementalRebalance() const {
      return false;
    }
    virtual std::string getDecompMethod() const {
      return std::string("rcb");
    }
    virtual std::string getCoordinateFieldName() const {
      return std::string("NODE_COORDS");
    }
    virtual bool shouldPrintMetrics() const {
      return false;
    }
  };  // RcbSettings

  RcbSettings balance_settings_;

  class EntityLessZMortonCoords {
   public:
    EntityLessZMortonCoords(const stk::mesh::BulkData &bulk)
        : mesh(bulk), coords_base(bulk.mesh_meta_data().coordinate_field()) {
    }

    bool operator()(stk::mesh::Entity a, stk::mesh::Entity b) {
      if (mesh.entity_rank(a) != stk::topology::NODE_RANK) {
        return stk::mesh::EntityLess(mesh)(a, b);
      }
      double *a_coords = static_cast<double *>(stk::mesh::field_data(*coords_base, a));
      double *b_coords = static_cast<double *>(stk::mesh::field_data(*coords_base, b));
      math::Vector3<double> a_vec(a_coords[0], a_coords[1], a_coords[2]);
      math::Vector3<double> b_vec(b_coords[0], b_coords[1], b_coords[2]);
      return math::zmorton_less(a_vec, b_vec);
    }

   private:
    const stk::mesh::BulkData &mesh;
    const stk::mesh::FieldBase *coords_base;
  };

  class ZMortonSorter : public stk::mesh::EntitySorterBase {
   public:
    ZMortonSorter() {
    }
    virtual ~ZMortonSorter() {
    }
    virtual void sort(stk::mesh::BulkData &bulk, stk::mesh::EntityVector &entity_vector) const {
      std::sort(entity_vector.begin(), entity_vector.end(), EntityLessZMortonCoords(bulk));
    }
  };

  ZMortonSorter z_morton_sorter_;
  //@}

  //! \name User parameters
  //@{

  // Setup params
  std::string input_parameter_filename_ = "hp1.yaml";

  // Simulation params
  size_t num_time_steps_;
  double timestep_size_;
  double viscosity_;
  size_t num_chromosomes_;
  size_t num_chromatin_repeats_;
  size_t num_euchromatin_per_repeat_;
  size_t num_heterochromatin_per_repeat_;
  double backbone_sphere_hydrodynamic_radius_;
  double initial_chromosome_separation_;
  INITIALIZATION_TYPE initialization_type_;
  std::string initialize_from_exo_filename_;
  std::string initialize_from_dat_filename_;
  std::string hilbert_centers_filename_;
  double unit_cell_size_[3];
  bool loadbalance_post_initialization_;
  bool check_maximum_speed_pre_position_update_;
  double max_allowable_speed_;
  HYDRO_TYPE hydro_type_;

  // IO params
  size_t io_frequency_;
  size_t log_frequency_;
  std::string output_filename_;
  bool enable_continuation_if_available_;

  // Control flags
  bool enable_chromatin_brownian_motion_;
  bool enable_backbone_springs_;
  bool enable_backbone_collision_;
  bool enable_backbone_n_body_hydrodynamics_;
  bool enable_crosslinkers_;
  bool enable_periphery_collision_;
  bool enable_periphery_hydrodynamics_;
  bool enable_periphery_binding_;
  bool enable_active_euchromatin_forces_;

  // Brownian params
  double brownian_kt_;

  // Backbone springs params
  BOND_TYPE backbone_spring_type_;
  double backbone_spring_constant_;
  double backbone_spring_r0_;

  // Backbone collisions params
  COLLISION_TYPE backbone_collision_type_;
  double backbone_excluded_volume_radius_;
  double backbone_youngs_modulus_;
  double backbone_poissons_ratio_;
  double backbone_wca_epsilon_;
  double backbone_wca_sigma_;
  double backbone_wca_cutoff_;

  // Crosslinker params
  BOND_TYPE crosslinker_spring_type_;
  double crosslinker_kt_;
  double crosslinker_spring_constant_;
  double crosslinker_r0_;
  double crosslinker_left_binding_rate_;
  double crosslinker_right_binding_rate_;
  double crosslinker_left_unbinding_rate_;
  double crosslinker_right_unbinding_rate_;
  double crosslinker_cutoff_radius_;

  // Periphery hydro params
  size_t hydro_update_frequency_;
  bool check_maximum_periphery_overlap_;
  double maximum_allowed_periphery_overlap_;
  PERIPHERY_SHAPE periphery_hydro_shape_;
  double periphery_hydro_radius_;        // For spheres
  double periphery_hydro_axis_radius1_;  // For ellipsoids
  double periphery_hydro_axis_radius2_;  // For ellipsoids
  double periphery_hydro_axis_radius3_;  // For ellipsoids
  PERIPHERY_QUADRATURE periphery_hydro_quadrature_;
  size_t periphery_hydro_spectral_order_;
  size_t periphery_hydro_num_quadrature_points_;
  std::string periphery_hydro_quadrature_points_filename_;
  std::string periphery_hydro_quadrature_weights_filename_;
  std::string periphery_hydro_quadrature_normals_filename_;

  // Periphery collision params
  PERIPHERY_SHAPE periphery_collision_shape_;
  double periphery_collision_radius_;        // For spheres
  double periphery_collision_axis_radius1_;  // For ellipsoids
  double periphery_collision_axis_radius2_;  // For ellipsoids
  double periphery_collision_axis_radius3_;  // For ellipsoids
  double periphery_collision_scale_factor_for_equilibriation_;
  double periphery_collision_spring_constant_;
  bool periphery_collision_use_fast_approx_;
  bool shrink_periphery_over_time_;
  size_t periphery_collision_shrinkage_num_steps_;
  double periphery_collision_scale_factor_before_shrinking_;

  // Periphery binding params
  double periphery_binding_rate_;
  double periphery_unbinding_rate_;
  double periphery_spring_constant_;
  double periphery_spring_r0_;
  PERIPHERY_BIND_SITES_TYPE periphery_bind_sites_type_;
  size_t periphery_num_bind_sites_;
  std::string periphery_bind_site_locations_filename_;

  // Active euchromatin forces params
  double active_euchromatin_force_sigma_;
  double active_euchromatin_force_kon_;
  double active_euchromatin_force_koff_;

  // Neighbor list params
  double skin_distance_;
  bool force_neighborlist_update_;
  size_t force_neighborlist_update_nsteps_;
  bool print_neighborlist_statistics_;
  //@}

  //! \name Default user parameters
  //@{

  // Simulation params
  static constexpr size_t default_num_time_steps_ = 100;
  static constexpr double default_timestep_size_ = 0.001;
  static constexpr double default_viscosity_ = 1.0;
  static constexpr size_t default_num_chromosomes_ = 1;
  static constexpr size_t default_num_chromatin_repeats_ = 2;
  static constexpr size_t default_num_euchromatin_per_repeat_ = 1;
  static constexpr size_t default_num_heterochromatin_per_repeat_ = 1;
  static constexpr double default_initial_chromosome_separation_ = 1.0;
  static constexpr std::string_view default_initialization_type_string_ = "GRID";
  static constexpr std::string_view default_initialize_from_exo_filename_ = "HP1";
  static constexpr std::string_view default_initialize_from_dat_filename_ = "HP1_pos.dat";
  static constexpr std::string_view default_hilbert_centers_filename_ = "HP1_hilbert_centers.dat";
  static constexpr bool default_loadbalance_post_initialization_ = false;
  static constexpr double default_unit_cell_size_[3] = {10.0, 10.0, 10.0};
  static constexpr bool default_check_maximum_speed_pre_position_update_ = false;
  static constexpr double default_max_allowable_speed_ = std::numeric_limits<double>::max();

  // IO params
  static constexpr size_t default_io_frequency_ = 10;
  static constexpr size_t default_log_frequency_ = 10;
  static constexpr std::string_view default_output_filename_ = "HP1";
  static constexpr bool default_enable_continuation_if_available_ = true;

  // Control params
  static constexpr bool default_enable_chromatin_brownian_motion_ = true;
  static constexpr bool default_enable_backbone_springs_ = true;
  static constexpr bool default_enable_backbone_collision_ = true;
  static constexpr bool default_enable_backbone_n_body_hydrodynamics_ = true;
  static constexpr bool default_enable_crosslinkers_ = true;
  static constexpr bool default_enable_periphery_collision_ = true;
  static constexpr bool default_enable_periphery_hydrodynamics_ = true;
  static constexpr bool default_enable_periphery_binding_ = true;
  static constexpr bool default_enable_active_euchromatin_forces_ = true;

  // Brownian params
  static constexpr double default_brownian_kt_ = 1.0;

  // Backbone springs params
  static constexpr std::string_view default_backbone_spring_type_string_ = "HARMONIC";
  static constexpr double default_backbone_spring_constant_ = 100.0;
  static constexpr double default_backbone_spring_r0_ = 1.0;

  // Backbone collisions params
  static constexpr std::string_view default_backbone_collision_type_string_ = "HERTZIAN";
  static constexpr double default_backbone_excluded_volume_radius_ = 0.5;
  static constexpr double default_backbone_youngs_modulus_ = 1000.0;
  static constexpr double default_backbone_poissons_ratio_ = 0.3;
  static constexpr double default_backbone_wca_epsilon_ = 1.0;
  static constexpr double default_backbone_wca_sigma_ = 1.0;
  static constexpr double default_backbone_wca_cutoff_ = 1.12246204831;  // 2^(1/6)

  // Backbone hydrodynamic params
  static constexpr std::string_view default_backbone_hydro_type_ = "RPYC";
  static constexpr double default_backbone_sphere_hydrodynamic_radius_ = 0.05;

  // Crosslinker params
  static constexpr std::string_view default_crosslinker_spring_type_string_ = "HARMONIC";
  static constexpr double default_crosslinker_kt_ = 1.0;
  static constexpr double default_crosslinker_spring_constant_ = 10.0;
  static constexpr double default_crosslinker_r0_ = 2.5;
  static constexpr double default_crosslinker_left_binding_rate_ = 1.0;
  static constexpr double default_crosslinker_right_binding_rate_ = 1.0;
  static constexpr double default_crosslinker_left_unbinding_rate_ = 1.0;
  static constexpr double default_crosslinker_right_unbinding_rate_ = 1.0;

  // Periphery hydro params
  static constexpr size_t default_hydro_update_frequency_ = 1;
  static constexpr bool default_check_maximum_periphery_overlap_ = false;
  static constexpr double default_maximum_allowed_periphery_overlap_ = 1e-6;
  static constexpr std::string_view default_periphery_hydro_shape_string_ = "SPHERE";
  static constexpr double default_periphery_hydro_radius_ = 5.0;
  static constexpr double default_periphery_hydro_axis_radius1_ = 5.0;
  static constexpr double default_periphery_hydro_axis_radius2_ = 5.0;
  static constexpr double default_periphery_hydro_axis_radius3_ = 5.0;
  static constexpr std::string_view default_periphery_hydro_quadrature_string_ = "GAUSS_LEGENDRE";
  static constexpr size_t default_periphery_hydro_spectral_order_ = 32;
  static constexpr size_t default_periphery_hydro_num_quadrature_points_ = 1000;
  static constexpr std::string_view default_periphery_hydro_quadrature_points_filename_ =
      "hp1_periphery_hydro_quadrature_points.dat";
  static constexpr std::string_view default_periphery_hydro_quadrature_weights_filename_ =
      "hp1_periphery_hydro_quadrature_weights.dat";
  static constexpr std::string_view default_periphery_hydro_quadrature_normals_filename_ =
      "hp1_periphery_hydro_quadrature_normals.dat";

  // Periphery collision params
  static constexpr std::string_view default_periphery_collision_shape_string_ = "SPHERE";
  static constexpr double default_periphery_collision_radius_ = 5.0;
  static constexpr double default_periphery_collision_axis_radius1_ = 5.0;
  static constexpr double default_periphery_collision_axis_radius2_ = 5.0;
  static constexpr double default_periphery_collision_axis_radius3_ = 5.0;
  static constexpr double default_periphery_collision_scale_factor_for_equilibriation_ = 2.0;
  static constexpr double default_periphery_collision_spring_constant_ = 1000.0;
  static constexpr bool default_periphery_collision_use_fast_approx_ = false;
  static constexpr bool default_shrink_periphery_over_time_ = false;
  static constexpr size_t default_periphery_collision_shrinkage_num_steps_ = 1000;
  static constexpr double default_periphery_collision_scale_factor_before_shrinking_ = 1.0;

  // Periphery binding params
  static constexpr double default_periphery_binding_rate_ = 1.0;
  static constexpr double default_periphery_unbinding_rate_ = 1.0;
  static constexpr double default_periphery_spring_constant_ = 1000.0;
  static constexpr double default_periphery_spring_r0_ = 1.0;
  static constexpr std::string_view default_periphery_bind_sites_type_string_ = "RANDOM";
  static constexpr size_t default_periphery_num_bind_sites_ = 1000;
  static constexpr std::string_view default_periphery_bind_site_locations_filename_ = "periphery_bind_sites.dat";

  // Active euchromatin forces params
  static constexpr double default_active_euchromatin_force_sigma_ = 1.0;
  static constexpr double default_active_euchromatin_force_kon_ = 1.0;
  static constexpr double default_active_euchromatin_force_koff_ = 1.0;

  // Neighbor list params
  static constexpr double default_skin_distance_ = 1.0;
  static constexpr bool default_force_neighborlist_update_ = false;
  static constexpr size_t default_force_neighborlist_update_nsteps_ = 10;
  static constexpr bool default_print_neighborlist_statistics_ = false;
  //@}
};  // class HP1

}  // namespace hp1

}  // namespace alens

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  // Run the simulation using the given parameters
  mundy::alens::hp1::HP1().run(argc, argv);

  // Before exiting, sleep for some amount of time to force Kokkos to print better at the end.
  std::this_thread::sleep_for(std::chrono::milliseconds(stk::parallel_machine_rank(MPI_COMM_WORLD)));

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
