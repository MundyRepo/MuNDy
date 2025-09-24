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

/* Notes:

The goal of this example is to simulate the swimming motion of a multiple, non-interacting long sperm.
*/

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Kokkos_Core.hpp>                       // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>      // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>             // for Teuchos::ParameterList
#include <Teuchos_YamlParameterListHelpers.hpp>  // for Teuchos::getParametersFromYamlFile
#include <stk_balance/balance.hpp>               // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>            // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Comm.hpp>                // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>        // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>              // for stk::mesh::Entity
#include <stk_mesh/base/FEMHelpers.hpp>          // for stk::mesh::declare_element, stk::mesh::declare_element_edge
#include <stk_mesh/base/Field.hpp>               // for stk::mesh::Field, stk::mesh::field_data
#include <stk_mesh/base/FieldParallel.hpp>       // for stk::mesh::parallel_sum
#include <stk_mesh/base/Part.hpp>                // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>            // for stk::mesh::Selector
#include <stk_topology/topology.hpp>             // for stk::topology
#include <stk_util/parallel/Parallel.hpp>        // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_core/MakeStringArray.hpp>  // for mundy::core::make_string_array
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_math/Matrix3.hpp>          // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>       // for mundy::math::Quaternion, mundy::math::quat_from_parallel_transport
#include <mundy_math/Vector3.hpp>          // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/utils/FillFieldWithValue.hpp>  // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/FieldReqs.hpp>                 // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>                  // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>                  // for mundy::meta::PartReqs
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

/// \brief The main function for the sperm simulation broken down into digestible chunks.
///
/// This class is in charge of setting up and executing the following tasks. In the future, this will all be handled by
/// the Configurator.
///   // Preprocess
///   - Parse user parameter
///
///   // Setup
///   - Declare the fixed and mutable params for the desired MetaMethods
///   - Construct the mesh and the method instances
///   - Declare and initialize the sperm's nodes, edges, and elements (centerline twist springs)
///    (Using BulkData's declare_node, declare_element, declare_element_edge functions)
///
///   // Timeloop
///   - Run the timeloop for t in range(0, T):
///       // IO.
///       - If desired, write out the data for time t
///         (Using stk::io::StkMeshIoBroker)
///
///       // Prepare the current configuration.
///       - Rotate the field states
///         (Using BulkData's update_field_data_states function)
///
///       - Zero the node forces and velocities for time t + dt
///         (Using mundy's fill_field_with_value function)
///
///       // Motion from t -> t + dt:
///        - Apply velocity/acceleration constraints like no motion for particle 1
///         (By directly looping over all nodes and setting the velocity/acceleration to zero)
///
///        - Evaluate x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2
///         (By looping over all nodes and updating the coordinates)
///
///       // Evaluate forces f(x(t + dt))
///       {
///         // Centerline twist rod model
///         - Compute the edge information (length, tangent, and binormal)
///          (By looping over all edges and computing the edge length, tangent, binormal)
///
///         - Compute the node curvature and rotation gradient
///          (By looping over all centerline twist spring elements and computing the curvature & rotation gradient at
///          their center node using the edge orientations)
///
///         - Compute the internal force and twist torque
///          (By looping over all centerline twist spring elements, using the curvature to compute the induced
///          benching/twisting torque and stretch force, and using them to compute the force and twist-torque on the
///          nodes. .)
///       }
///
///       // Compute velocity and acceleration
///        - Evaluate a(t + dt) = M^{-1} f(x(t + dt))
///         (By looping over all nodes and computing the node acceleration and twist acceleration from the force and
///         twist torque)
///
///        - Evaluate v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2
///         (By looping over all nodes and updating the node velocity and twist rate using the corresponding
///         accelerations)
///
/// The sperm themselves are modeled as a chain of rods with a centerline twist spring connecting pairs of adjacent
/// edges:
/*
/// n1       n3        n5        n7
///  \      /  \      /  \      /
///   s1   s2   s3   s4   s5   s6
///    \  /      \  /      \  /
///     n2        n4        n6
*/
/// The centerline twist springs are hard to draw with ASCII art, but they are centered at every interior node and
/// connected to the node's neighbors:
///   c1 has a center node at n2 and connects to n1 and n3.
///   c2 has a center node at n3 and connects to n2 and n4.
///   and so on.
///
/// STK EntityId-wise. Nodes are numbered sequentially from 1 to num_nodes. Centerline twist springs are numbered
/// sequentially from 1 to num_nodes-2.
///
/// To turn this into a movie use:
/// ffmpeg -framerate 20 -pattern_type glob -i 'mov_stiff*.png' -c:v libvpx-vp9 -b:v 0 -crf 24 -threads 16 -row-mt 1
/// -pass 1 -f null /dev/null && ffmpeg -framerate 60 -pattern_type glob -i 'mov_stiff*.png' -c:v libvpx-vp9 -b:v 0 -crf
/// 24 -threads 16 -row-mt 1 -pass 2 stiff_movie.webm
class SpermSimulation {
 public:
  SpermSimulation() = default;

  void print_rank0(auto think_to_print, int indent_level = 0) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::string indent(indent_level * 2, ' ');
      std::cout << indent << think_to_print << std::endl;
    }
  }

  void debug_print([[maybe_unused]] auto thing_to_print, [[maybe_unused]] int indent_level = 0) {
#ifdef DEBUG
    print_rank0(thing_to_print, indent_level);
#endif
  }

  void parse_user_inputs(int argc, char **argv) {
    debug_print("Parsing user inputs.");

    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, false);

    // If we should accept the parameters directly from the command line or from a file
    bool use_input_file = false;
    cmdp.setOption("use_input_file", "no_use_input_file", &use_input_file, "Use an input file.");
    bool use_input_file_found = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    MUNDY_THROW_REQUIRE(use_input_file_found, std::invalid_argument, "Failed to parse the command line arguments.");

    // Switch to requiring that all options must be recognized.
    cmdp.recogniseAllOptions(true);

    if (!use_input_file) {
      // Parse the command line options.

      //   Sperm initialization:
      cmdp.setOption("num_sperm", &num_sperm_, "Number of sperm.");
      cmdp.setOption("num_nodes_per_sperm", &num_nodes_per_sperm_, "Number of nodes per sperm.");
      cmdp.setOption("sperm_radius", &sperm_radius_, "The radius of each sperm.");
      cmdp.setOption("sperm_initial_segment_length", &sperm_initial_segment_length_, "Initial sperm segment length.");
      cmdp.setOption("sperm_rest_segment_length", &sperm_rest_segment_length_, "Rest sperm segment length.");
      cmdp.setOption("sperm_rest_curvature_twist", &sperm_rest_curvature_twist_,
                     "Rest curvature (twist) of the sperm.");
      cmdp.setOption("sperm_rest_curvature_bend1", &sperm_rest_curvature_bend1_,
                     "Rest curvature (bend along the first coordinate direction) of the sperm.");
      cmdp.setOption("sperm_rest_curvature_bend2", &sperm_rest_curvature_bend2_,
                     "Rest curvature (bend along the second coordinate direction) of the sperm.");

      cmdp.setOption("sperm_density", &sperm_density_, "Density of the sperm.");
      cmdp.setOption("sperm_youngs_modulus", &sperm_youngs_modulus_, "Young's modulus of the sperm.");
      cmdp.setOption("sperm_poissons_ratio", &sperm_poissons_ratio_, "Poisson's ratio of the sperm.");
      cmdp.setOption("sperm_density", &sperm_density_, "Density of the sperm.");

      //   The simulation:
      cmdp.setOption("num_time_steps", &num_time_steps_, "Number of time steps.");
      cmdp.setOption("timestep_size", &timestep_size_, "Time step size.");
      cmdp.setOption("io_frequency", &io_frequency_, "Number of timesteps between writing output.");

      bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
      MUNDY_THROW_REQUIRE(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");
    } else {
      cmdp.setOption("input_file", &input_file_name_, "The name of the input file.");
      bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
      MUNDY_THROW_REQUIRE(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");

      // Read in the parameters from the parameter list.
      Teuchos::ParameterList param_list_ = *Teuchos::getParametersFromYamlFile(input_file_name_);

      num_sperm_ = param_list_.get<int>("num_sperm");
      num_nodes_per_sperm_ = param_list_.get<int>("num_nodes_per_sperm");
      sperm_radius_ = param_list_.get<double>("sperm_radius");
      sperm_initial_segment_length_ = param_list_.get<double>("sperm_initial_segment_length");
      sperm_rest_segment_length_ = param_list_.get<double>("sperm_rest_segment_length");
      sperm_rest_curvature_twist_ = param_list_.get<double>("sperm_rest_curvature_twist");
      sperm_rest_curvature_bend1_ = param_list_.get<double>("sperm_rest_curvature_bend1");
      sperm_rest_curvature_bend2_ = param_list_.get<double>("sperm_rest_curvature_bend2");

      sperm_density_ = param_list_.get<double>("sperm_density");
      sperm_youngs_modulus_ = param_list_.get<double>("sperm_youngs_modulus");
      sperm_poissons_ratio_ = param_list_.get<double>("sperm_poissons_ratio");
      sperm_poissons_ratio_ = param_list_.get<double>("sperm_poissons_ratio");

      num_time_steps_ = param_list_.get<int>("num_time_steps");
      timestep_size_ = param_list_.get<double>("timestep_size");
    }

    check_input_parameters();
  }

  void check_input_parameters() {
    debug_print("Checking input parameters.");
    MUNDY_THROW_REQUIRE(num_sperm_ > 0, std::invalid_argument, "num_sperm_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(num_nodes_per_sperm_ > 0, std::invalid_argument,
                        "num_nodes_per_sperm_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(sperm_radius_ > 0, std::invalid_argument, "sperm_radius_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(sperm_initial_segment_length_ > -1e-12, std::invalid_argument,
                        "sperm_initial_segment_length_ must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(sperm_rest_segment_length_ > -1e-12, std::invalid_argument,
                        "sperm_rest_segment_length_ must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(sperm_youngs_modulus_ > 0, std::invalid_argument,
                        "sperm_youngs_modulus_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(sperm_poissons_ratio_ > 0, std::invalid_argument,
                        "sperm_poissons_ratio_ must be greater than 0.");

    MUNDY_THROW_REQUIRE(num_time_steps_ > 0, std::invalid_argument, "num_time_steps_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(io_frequency_ > 0, std::invalid_argument, "io_frequency_ must be greater than 0.");
  }

  void dump_user_inputs() {
    debug_print("Dumping user inputs.");
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;
      std::cout << "  num_sperm_: " << num_sperm_ << std::endl;
      std::cout << "  num_nodes_per_sperm_: " << num_nodes_per_sperm_ << std::endl;
      std::cout << "  sperm_radius_: " << sperm_radius_ << std::endl;
      std::cout << "  sperm_initial_segment_length_: " << sperm_initial_segment_length_ << std::endl;
      std::cout << "  sperm_rest_segment_length_: " << sperm_rest_segment_length_ << std::endl;
      std::cout << "  sperm_rest_curvature_twist_: " << sperm_rest_curvature_twist_ << std::endl;
      std::cout << "  sperm_rest_curvature_bend1_: " << sperm_rest_curvature_bend1_ << std::endl;
      std::cout << "  sperm_rest_curvature_bend2_: " << sperm_rest_curvature_bend2_ << std::endl;
      std::cout << "  sperm_youngs_modulus_: " << sperm_youngs_modulus_ << std::endl;
      std::cout << "  sperm_poissons_ratio_: " << sperm_poissons_ratio_ << std::endl;
      std::cout << "  sperm_density_: " << sperm_density_ << std::endl;
      std::cout << "  num_time_steps_: " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size_: " << timestep_size_ << std::endl;
      std::cout << "  io_frequency_: " << io_frequency_ << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  void build_our_mesh_and_method_instances() {
    debug_print("Building our mesh and method instances.");

    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshReqs>(MPI_COMM_WORLD);
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements for this example. These are requirements that exceed those of the enabled methods and
    // allow us to extend the functionality offered natively by Mundy.
    //
    // We require that the centerline twist springs part exists, has a SHELL_TRI_3 topology, and has the desired
    // node/edge/element fields.
    auto clt_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    clt_part_reqs->set_part_name("CENTERLINE_TWIST_SPRINGS")
        .set_part_topology(stk::topology::SHELL_TRI_3)

        // Add the node fields
        .add_field_reqs<double>("NODE_COORDS", node_rank_, 3, 2)
        .add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 2)
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_ACCELERATION", node_rank_, 3, 2)

        .add_field_reqs<double>("NODE_TWIST", node_rank_, 1, 2)
        .add_field_reqs<double>("NODE_TWIST_VELOCITY", node_rank_, 1, 2)
        .add_field_reqs<double>("NODE_TWIST_TORQUE", node_rank_, 1, 1)
        .add_field_reqs<double>("NODE_TWIST_ACCELERATION", node_rank_, 1, 2)

        .add_field_reqs<double>("NODE_CURVATURE", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_REST_CURVATURE", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_ROTATION_GRADIENT", node_rank_, 4, 1)
        .add_field_reqs<double>("NODE_RADIUS", node_rank_, 1, 1)
        .add_field_reqs<double>("NODE_ARCHLENGTH", node_rank_, 1, 1)

        // Add the edge fields
        .add_field_reqs<double>("EDGE_ORIENTATION", edge_rank_, 4, 2)
        .add_field_reqs<double>("EDGE_TANGENT", edge_rank_, 3, 2)
        .add_field_reqs<double>("EDGE_BINORMAL", edge_rank_, 3, 1)
        .add_field_reqs<double>("EDGE_LENGTH", edge_rank_, 1, 1)

        // Add the element fields
        .add_field_reqs<double>("ELEMENT_RADIUS", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_YOUNGS_MODULUS", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_POISSONS_RATIO", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_REST_LENGTH", element_rank_, 1, 1);
    mesh_reqs_ptr_->add_and_sync_part_reqs(clt_part_reqs);

#ifdef DEBUG
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      mesh_reqs_ptr_->print();
    }
#endif

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDS");
    meta_data_ptr_->use_simple_fields();
    meta_data_ptr_->commit();
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
    debug_print("Fetching fields and parts.");

    // Fetch the fields
    node_coord_field_ptr_ = fetch_field<double>("NODE_COORDS", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_acceleration_field_ptr_ = fetch_field<double>("NODE_ACCELERATION", node_rank_);
    node_twist_field_ptr_ = fetch_field<double>("NODE_TWIST", node_rank_);
    node_twist_velocity_field_ptr_ = fetch_field<double>("NODE_TWIST_VELOCITY", node_rank_);
    node_twist_torque_field_ptr_ = fetch_field<double>("NODE_TWIST_TORQUE", node_rank_);
    node_twist_acceleration_field_ptr_ = fetch_field<double>("NODE_TWIST_ACCELERATION", node_rank_);
    node_curvature_field_ptr_ = fetch_field<double>("NODE_CURVATURE", node_rank_);
    node_rest_curvature_field_ptr_ = fetch_field<double>("NODE_REST_CURVATURE", node_rank_);
    node_rotation_gradient_field_ptr_ = fetch_field<double>("NODE_ROTATION_GRADIENT", node_rank_);
    node_radius_field_ptr_ = fetch_field<double>("NODE_RADIUS", node_rank_);
    node_archlength_field_ptr_ = fetch_field<double>("NODE_ARCHLENGTH", node_rank_);

    edge_orientation_field_ptr_ = fetch_field<double>("EDGE_ORIENTATION", edge_rank_);
    edge_tangent_field_ptr_ = fetch_field<double>("EDGE_TANGENT", edge_rank_);
    edge_binormal_field_ptr_ = fetch_field<double>("EDGE_BINORMAL", edge_rank_);
    edge_length_field_ptr_ = fetch_field<double>("EDGE_LENGTH", edge_rank_);

    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
    element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    element_rest_length_field_ptr_ = fetch_field<double>("ELEMENT_REST_LENGTH", element_rank_);

    // Fetch the parts
    centerline_twist_springs_part_ptr_ = fetch_part("CENTERLINE_TWIST_SPRINGS");
    MUNDY_THROW_REQUIRE(centerline_twist_springs_part_ptr_->topology() == stk::topology::SHELL_TRI_3, std::logic_error,
                        "CENTERLINE_TWIST_SPRINGS part must have SHELL_TRI_3 topology.");
  }

  void setup_io() {
    debug_print("Setting up IO.");

    // Declare each part as an IO part
    stk::io::put_io_part_attribute(*centerline_twist_springs_part_ptr_);

    // Setup the IO broker
    stk_io_broker_.use_simple_fields();
    stk_io_broker_.set_bulk_data(*bulk_data_ptr_);

    output_file_index_ = stk_io_broker_.create_output_mesh("Sperm.exo", stk::io::WRITE_RESULTS);
    stk_io_broker_.add_field(output_file_index_, *node_coord_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_velocity_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_force_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_acceleration_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_velocity_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_torque_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_acceleration_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_curvature_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_rest_curvature_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_rotation_gradient_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_radius_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_archlength_field_ptr_);

    stk_io_broker_.add_field(output_file_index_, *edge_orientation_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *edge_tangent_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *edge_binormal_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *edge_length_field_ptr_);

    stk_io_broker_.add_field(output_file_index_, *element_radius_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *element_youngs_modulus_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *element_poissons_ratio_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *element_rest_length_field_ptr_);
  }

  void declare_and_initialize_sperm() {
    debug_print("Declaring and initializing the sperm.");

    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;

    // Declare N spring chains with a slight shift to each chain
    for (size_t j = 0; j < num_sperm_; j++) {
      // To make our lives easier, we alogn the sperm with the z-axis, as this makes our edge orientation a unit
      // quaternion.
      mundy::math::Vector3d tail_coord(0.0, 2 * j * sperm_radius_, 0.0);
      mundy::math::Vector3d sperm_axis(0.0, 0.0, 1.0);

      // Because we are creating multiple sperm, we need to determine the node and element index ranges for each sperm.
      size_t start_node_id = num_nodes_per_sperm_ * j + 1u;
      size_t start_edge_id = (num_nodes_per_sperm_ - 1) * j + 1u;
      size_t start_centerline_twist_spring_id = (num_nodes_per_sperm_ - 2) * j + 1u;

      auto get_node_id = [start_node_id](const size_t &seq_node_index) { return start_node_id + seq_node_index; };

      auto get_node = [get_node_id, &bulk_data](const size_t &seq_node_index) {
        return bulk_data.get_entity(stk::topology::NODE_RANK, get_node_id(seq_node_index));
      };

      auto get_edge_id = [start_edge_id](const size_t &seq_node_index) { return start_edge_id + seq_node_index; };

      auto get_edge = [get_edge_id, &bulk_data](const size_t &seq_node_index) {
        return bulk_data.get_entity(stk::topology::EDGE_RANK, get_edge_id(seq_node_index));
      };

      auto get_centerline_twist_spring_id = [start_centerline_twist_spring_id](const size_t &seq_spring_index) {
        return start_centerline_twist_spring_id + seq_spring_index;
      };

      auto get_centerline_twist_spring = [get_centerline_twist_spring_id, &bulk_data](const size_t &seq_spring_index) {
        return bulk_data.get_entity(stk::topology::ELEMENT_RANK, get_centerline_twist_spring_id(seq_spring_index));
      };

      // Create the springs and their connected nodes, distributing the work across the ranks.
      const size_t rank = bulk_data_ptr_->parallel_rank();
      const size_t nodes_per_rank = num_nodes_per_sperm_ / bulk_data_ptr_->parallel_size();
      const size_t remainder = num_nodes_per_sperm_ % bulk_data_ptr_->parallel_size();
      const size_t start_seq_node_index = rank * nodes_per_rank + std::min(rank, remainder);
      const size_t end_seq_node_index = start_seq_node_index + nodes_per_rank + (rank < remainder ? 1 : 0);

      bulk_data_ptr_->modification_begin();

      // Temporary/scatch variables
      stk::mesh::PartVector empty;
      stk::mesh::Permutation perm = stk::mesh::Permutation::INVALID_PERMUTATION;
      stk::mesh::OrdinalVector scratch1, scratch2, scratch3;
      auto spring_part = stk::mesh::PartVector{centerline_twist_springs_part_ptr_};
      stk::topology elem_topo = stk::topology::SHELL_TRI_3;
      stk::topology edge_topo = stk::topology::LINE_2;
      auto spring_and_edge_part =
          stk::mesh::PartVector{centerline_twist_springs_part_ptr_, &meta_data_ptr_->get_topology_root_part(edge_topo)};

      // Centerline twist springs connect nodes i, i+1, and i+2. We need to start at node i=0 and end at node N - 2.
      const size_t start_element_chain_index = start_seq_node_index;
      const size_t end_start_element_chain_index =
          (rank == bulk_data_ptr_->parallel_size() - 1) ? end_seq_node_index - 2 : end_seq_node_index - 1;
      for (size_t i = start_element_chain_index; i < end_start_element_chain_index; ++i) {
        // Note, the connectivity for a SHELL_TRI_3 is as follows:
        /*                    2
        //                    o
        //                   / \
        //                  /   \
        //                 /     \
        //   Edge #2      /       \     Edge #1
        //               /         \
        //              /           \
        //             /             \
        //            o---------------o
        //           0                 1
        //
        //                  Edge #0
        */
        // We use SHELL_TRI_3 for the centerline twist springs, so that we have access to two edges (edge #0 and #1) and
        // three nodes. As such, our diagram is
        /*                    2
        //                    o
        //                     \
        //                      \
        //                       \
        //                        \     Edge #1
        //                         \
        //                          \
        //                           \
        //            o---------------o
        //           0                 1
        //
        //                  Edge #0
        */

        // Fetch the nodes
        stk::mesh::EntityId left_node_id = get_node_id(i);
        stk::mesh::EntityId center_node_id = get_node_id(i + 1);
        stk::mesh::EntityId right_node_id = get_node_id(i + 2);
        stk::mesh::Entity left_node = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, left_node_id);
        stk::mesh::Entity center_node = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, center_node_id);
        stk::mesh::Entity right_node = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, right_node_id);
        if (!bulk_data_ptr_->is_valid(left_node)) {
          left_node = bulk_data_ptr_->declare_node(left_node_id, empty);
        }
        if (!bulk_data_ptr_->is_valid(center_node)) {
          center_node = bulk_data_ptr_->declare_node(center_node_id, empty);
        }
        if (!bulk_data_ptr_->is_valid(right_node)) {
          right_node = bulk_data_ptr_->declare_node(right_node_id, empty);
        }

        // Declare and connect the edges
        stk::mesh::EntityId left_edge_id = get_edge_id(i);
        stk::mesh::EntityId right_edge_id = get_edge_id(i + 1);
        stk::mesh::Entity left_edge = bulk_data_ptr_->get_entity(stk::topology::EDGE_RANK, left_edge_id);
        stk::mesh::Entity right_edge = bulk_data_ptr_->get_entity(stk::topology::EDGE_RANK, right_edge_id);
        if (!bulk_data_ptr_->is_valid(left_edge)) {
          // Declare the edge and connect it to the nodes
          left_edge = bulk_data_ptr_->declare_edge(left_edge_id, spring_and_edge_part);
          bulk_data_ptr_->declare_relation(left_edge, left_node, 0, perm, scratch1, scratch2, scratch3);
          bulk_data_ptr_->declare_relation(left_edge, center_node, 1, perm, scratch1, scratch2, scratch3);
        }
        if (!bulk_data_ptr_->is_valid(right_edge)) {
          // Declare the edge and connect it to the nodes
          right_edge = bulk_data_ptr_->declare_edge(right_edge_id, spring_and_edge_part);
          bulk_data_ptr_->declare_relation(right_edge, center_node, 0, perm, scratch1, scratch2, scratch3);
          bulk_data_ptr_->declare_relation(right_edge, right_node, 1, perm, scratch1, scratch2, scratch3);
        }

        // Fetch the centerline twist spring
        stk::mesh::EntityId spring_id = get_centerline_twist_spring_id(i);
        stk::mesh::Entity spring = bulk_data_ptr_->declare_element(spring_id, spring_part);

        // Connect the spring to the edges
        stk::mesh::Entity elem_nodes[3] = {left_node, center_node, right_node};
        stk::mesh::Entity left_edge_nodes[2] = {left_node, center_node};
        stk::mesh::Entity right_edge_nodes[2] = {center_node, right_node};

        stk::mesh::Permutation left_perm =
            bulk_data_ptr_->find_permutation(elem_topo, elem_nodes, edge_topo, left_edge_nodes, 0);
        stk::mesh::Permutation right_perm =
            bulk_data_ptr_->find_permutation(elem_topo, elem_nodes, edge_topo, right_edge_nodes, 1);
        bulk_data_ptr_->declare_relation(spring, left_edge, 0, left_perm, scratch1, scratch2, scratch3);
        bulk_data_ptr_->declare_relation(spring, right_edge, 1, right_perm, scratch1, scratch2, scratch3);

        // Connect the spring to the nodes
        bulk_data_ptr_->declare_relation(spring, left_node, 0, perm, scratch1, scratch2, scratch3);
        bulk_data_ptr_->declare_relation(spring, center_node, 1, perm, scratch1, scratch2, scratch3);
        bulk_data_ptr_->declare_relation(spring, right_node, 2, perm, scratch1, scratch2, scratch3);
        MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(spring).topology() != stk::topology::INVALID_TOPOLOGY,
                           std::logic_error,
                           fmt::format("The centerline twist spring with id {} has an invalid topology.", spring_id));

        // Populate the spring's data
        stk::mesh::field_data(*element_radius_field_ptr_, spring)[0] = sperm_radius_;
        stk::mesh::field_data(*element_youngs_modulus_field_ptr_, spring)[0] = sperm_youngs_modulus_;
        stk::mesh::field_data(*element_poissons_ratio_field_ptr_, spring)[0] = sperm_poissons_ratio_;
        stk::mesh::field_data(*element_rest_length_field_ptr_, spring)[0] = sperm_rest_segment_length_;
      }

      // Share the nodes with the neighboring ranks.
      //
      // Note, node sharing is symmetric. If we don't own the node that we intend to share, we need to declare it before
      // marking it as shared. If we are rank 0, we share our final node with rank 1 and receive their first node. If we
      // are rank N, we share our first node with rank N - 1 and receive their final node. Otherwise, we share our first
      // and last nodes with the corresponding neighboring ranks and receive their corresponding nodes.
      if (bulk_data_ptr_->parallel_size() > 1) {
        debug_print("Sharing nodes with neighboring ranks.");
        if (rank == 0) {
          // Share the last node with rank 1.
          stk::mesh::Entity node = get_node(end_seq_node_index - 1);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::logic_error,
                             fmt::format("The node with id {} is not valid.", get_node_id(end_seq_node_index - 1)));
          bulk_data_ptr_->add_node_sharing(node, rank + 1);

          // Receive the first node from rank 1
          stk::mesh::EntityId received_node_id = get_node_id(end_seq_node_index);
          stk::mesh::Entity received_node = bulk_data_ptr_->declare_node(received_node_id);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(received_node), std::logic_error,
                             fmt::format("The node with id {} is not valid.", received_node_id));
          bulk_data_ptr_->add_node_sharing(received_node, rank + 1);
        } else if (rank == bulk_data_ptr_->parallel_size() - 1) {
          // Share the first node with rank N - 1.
          stk::mesh::Entity node = get_node(start_seq_node_index);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::logic_error,
                             fmt::format("The node with id {} is not valid.", get_node_id(start_seq_node_index)));
          bulk_data_ptr_->add_node_sharing(node, rank - 1);

          // Receive the last node from rank N - 1.
          stk::mesh::EntityId received_node_id = get_node_id(start_seq_node_index - 1);
          stk::mesh::Entity received_node = bulk_data_ptr_->declare_node(received_node_id);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(received_node), std::logic_error,
                             fmt::format("The node with id {} is not valid.", received_node_id));
          bulk_data_ptr_->add_node_sharing(received_node, rank - 1);
        } else {
          // Share the first and last nodes with the corresponding neighboring ranks.
          stk::mesh::Entity first_node = get_node(start_seq_node_index);
          stk::mesh::Entity last_node = get_node(end_seq_node_index - 1);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(first_node), std::logic_error,
                             fmt::format("The node with id {} is not valid.", get_node_id(start_seq_node_index)));
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(last_node), std::logic_error,
                             fmt::format("The node with id {} is not valid.", get_node_id(end_seq_node_index - 1)));
          bulk_data_ptr_->add_node_sharing(first_node, rank - 1);
          bulk_data_ptr_->add_node_sharing(last_node, rank + 1);

          // Receive the corresponding nodes from the neighboring ranks.
          stk::mesh::EntityId received_first_node_id = get_node_id(start_seq_node_index - 1);
          stk::mesh::EntityId received_last_node_id = get_node_id(end_seq_node_index);
          stk::mesh::Entity received_first_node = bulk_data_ptr_->declare_node(received_first_node_id);
          stk::mesh::Entity received_last_node = bulk_data_ptr_->declare_node(received_last_node_id);
          bulk_data_ptr_->add_node_sharing(received_first_node, rank - 1);
          bulk_data_ptr_->add_node_sharing(received_last_node, rank + 1);
        }
      }

      bulk_data_ptr_->modification_end();

      // Set the node data for all nodes (even the shared ones)
      for (size_t i = start_seq_node_index - 1 * (rank > 0);
           i < end_seq_node_index + 1 * (rank < bulk_data_ptr_->parallel_size() - 1); ++i) {
        stk::mesh::Entity node = get_node(i);
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::logic_error,
                           fmt::format("The node with id {} is not valid.", get_node_id(i)));
        MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(node).member(*centerline_twist_springs_part_ptr_), std::logic_error,
                           "The node must be a member of the centerline twist part.");

        mundy::mesh::vector3_field_data(*node_coord_field_ptr_, node) =
            tail_coord + sperm_axis * static_cast<double>(i) * sperm_initial_segment_length_;
        mundy::mesh::vector3_field_data(*node_velocity_field_ptr_, node).set(0.0, 0.0, 0.0);
        mundy::mesh::vector3_field_data(*node_force_field_ptr_, node).set(0.0, 0.0, 0.0);
        mundy::mesh::vector3_field_data(*node_acceleration_field_ptr_, node).set(0.0, 0.0, 0.0);
        stk::mesh::field_data(*node_twist_field_ptr_, node)[0] = 0.0;
        stk::mesh::field_data(*node_twist_velocity_field_ptr_, node)[0] = 0.0;
        stk::mesh::field_data(*node_twist_torque_field_ptr_, node)[0] = 0.0;
        stk::mesh::field_data(*node_twist_acceleration_field_ptr_, node)[0] = 0.0;
        mundy::mesh::vector3_field_data(*node_curvature_field_ptr_, node).set(0.0, 0.0, 0.0);
        mundy::mesh::vector3_field_data(*node_rest_curvature_field_ptr_, node)
            .set(sperm_rest_curvature_bend1_, sperm_rest_curvature_bend2_, sperm_rest_curvature_twist_);
        stk::mesh::field_data(*node_radius_field_ptr_, node)[0] = sperm_radius_;
        stk::mesh::field_data(*node_archlength_field_ptr_, node)[0] = i * sperm_initial_segment_length_;
      }

#ifdef DEBUG
      for (size_t i = start_element_chain_index; i < end_start_element_chain_index; ++i) {
        stk::mesh::Entity spring = get_centerline_twist_spring(i);
        MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(spring).member(*centerline_twist_springs_part_ptr_), std::logic_error,
                           "The centerline twist spring must be a member of the centerline twist part.");
        MUNDY_THROW_ASSERT(
            centerline_twist_springs_part_ptr_->topology() == stk::topology::SHELL_TRI_3, std::logic_error,
            std::string("The centerline twist part must have SHELL_TRI_3 topology. Instead, it has topology ") +
                centerline_twist_springs_part_ptr_->topology());
        MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(spring).entity_rank() == stk::topology::ELEMENT_RANK,
                           std::logic_error,
                           std::string("The centerline twist spring must have element rank. Instead, it has rank ") +
                               bulk_data_ptr_->bucket(spring).entity_rank());
        MUNDY_THROW_ASSERT(
            bulk_data_ptr_->bucket(spring).topology() == stk::topology::SHELL_TRI_3, std::logic_error,
            std::string("The centerline twist spring must have SHELL_TRI_3 topology. Instead, it has topology ") +
                bulk_data_ptr_->bucket(spring).topology());
      }

      {
        std::vector<size_t> entity_counts;
        stk::mesh::comm_mesh_counts(*bulk_data_ptr_, entity_counts);
        debug_print(std::string("Num nodes: ") + std::to_string(entity_counts[stk::topology::NODE_RANK]));
        debug_print(std::string("Num edges: ") + std::to_string(entity_counts[stk::topology::EDGE_RANK]));
        debug_print(std::string("Num elements: ") + std::to_string(entity_counts[stk::topology::ELEMENT_RANK]));
      }
#endif

      // Populate the edge data
      stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
      stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
      stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;
      const double sperm_initial_segment_length = sperm_initial_segment_length_;
      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::EDGE_RANK, meta_data_ptr_->locally_owned_part(),
          [&edge_orientation_field, &edge_tangent_field, &edge_length_field, &sperm_axis,
           &sperm_initial_segment_length]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                          const stk::mesh::Entity &edge) {
            // The orientation of the edge is the identity since we are currently in the reference configuration.
            mundy::mesh::quaternion_field_data(edge_orientation_field, edge) = mundy::math::Quaterniond::identity();
            mundy::mesh::vector3_field_data(edge_tangent_field, edge) = sperm_axis;
            stk::mesh::field_data(edge_length_field, edge)[0] = sperm_initial_segment_length;
          });
    }
  }

  void loadbalance() {
    debug_print("Load balancing the mesh.");
    stk::balance::balanceStkMesh(balance_settings_, *bulk_data_ptr_);
  }

  void rotate_field_states() {
    debug_print("Rotating the field states.");
    bulk_data_ptr_->update_field_data_states();
  }

  void zero_out_transient_node_fields() {
    debug_print("Zeroing out the transient node fields.");
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_acceleration_field_ptr_,
                                                      std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_velocity_field_ptr_, std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_torque_field_ptr_, std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_acceleration_field_ptr_, std::array<double, 1>{0.0});
  }

  void update_generalized_position() {
    debug_print("Updating the generalized position.");
    // x(t + dt) = x(t) + dt v(t) + 0.5 dt^2 (1 - 2 beta) a(t)
    // beta = 0.25, gamma = 0.5 -> unconditionally stable, constant-average acceleration method

    // No data to communicate since we only act on locally owned nodes.

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_springs_part = *centerline_twist_springs_part_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_coord_field_old = node_coord_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_velocity_field_old = node_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_acceleration_field_old =
        node_acceleration_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &node_twist_field_old = node_twist_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_velocity_field_old =
        node_twist_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_acceleration_field_old =
        node_twist_acceleration_field_ptr_->field_of_state(stk::mesh::StateN);
    const double time_step_size = timestep_size_;

    // Update the position and twist of the nodes
    auto locally_owned_selector =
        stk::mesh::Selector(centerline_twist_springs_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, node_rank_, locally_owned_selector,
        [&node_coord_field, &node_coord_field_old, &node_velocity_field_old, &node_acceleration_field_old,
         &node_twist_field, &node_twist_field_old, &node_twist_velocity_field_old, &node_twist_acceleration_field_old,
         time_step_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const auto node_coords_old = mundy::mesh::vector3_field_data(node_coord_field_old, node);
          const auto node_velocity_old = mundy::mesh::vector3_field_data(node_velocity_field_old, node);
          const auto node_acceleration_old = mundy::mesh::vector3_field_data(node_acceleration_field_old, node);
          const double node_twist_old = stk::mesh::field_data(node_twist_field_old, node)[0];
          const double node_twist_velocity_old = stk::mesh::field_data(node_twist_velocity_field_old, node)[0];
          const double node_twist_acceleration_old = stk::mesh::field_data(node_twist_acceleration_field_old, node)[0];

          // Get the output fields
          auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, node);
          double *node_twist = stk::mesh::field_data(node_twist_field, node);

          // Update the current configuration
          const double beta = 0.25;
          const double coeff = 0.5 * time_step_size * time_step_size * (1.0 - 2.0 * beta);
          node_coords = node_coords_old + time_step_size * node_velocity_old + coeff * node_acceleration_old;
          node_twist[0] =
              node_twist_old + time_step_size * node_twist_velocity_old + coeff * node_twist_acceleration_old;
        });
  }

  void propogate_rest_curvature() {
    debug_print("Propogating the rest curvature.");

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_springs_part = *centerline_twist_springs_part_ptr_;
    stk::mesh::Field<double> &node_rest_curvature_field = *node_rest_curvature_field_ptr_;
    stk::mesh::Field<double> &node_archlength_field = *node_archlength_field_ptr_;

    // Propogate the rest curvature of the nodes according to
    // kappa_rest = amplitude * sin(spacial_frequency * archlength + temporal_frequency * time).
    const double sperm_length = num_nodes_per_sperm_ * sperm_initial_segment_length_;
    const double amplitude = 0.01;
    const double spacial_wavelength = sperm_length / 5.0;
    const double spacial_frequency = 2.0 * M_PI / spacial_wavelength;
    const double temporal_frequency = 0.01;
    const double time = (timestep_index_ == 0)
                            ? timestep_index_ * timestep_size_
                            : (timestep_index_ - 5000000) * timestep_size_;  // HARDCODING THE TIME OFFSET

    auto locally_owned_selector =
        stk::mesh::Selector(centerline_twist_springs_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, node_rank_, locally_owned_selector,
        [&node_rest_curvature_field, &node_archlength_field, &amplitude, &spacial_frequency, &temporal_frequency,
         &time]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const double node_archlength = stk::mesh::field_data(node_archlength_field, node)[0];

          // Get the output fields
          double *node_rest_curvature = stk::mesh::field_data(node_rest_curvature_field, node);

          // Propogate the rest curvature
          node_rest_curvature[0] =
              amplitude * std::sin(spacial_frequency * node_archlength + temporal_frequency * time);
        });
  }

  void compute_edge_information() {
    debug_print("Computing the edge information.");

    // Communicate ghosted fields.
    stk::mesh::communicate_field_data(*bulk_data_ptr_, {node_coord_field_ptr_});

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_springs_part = *centerline_twist_springs_part_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field_old = edge_orientation_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field_old = edge_tangent_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;

    // For each edge in the centerline twist part, compute the edge tangent, binormal, length, and orientation.
    // length^i = ||x_{i+1} - x_i||
    // edge_tangent^i = (x_{i+1} - x_i) / length
    // edge_binormal^i = (2 edge_tangent_old^i x edge_tangent^i) / (1 + edge_tangent_old^i dot edge_tangent^i)
    // edge_orientation^j(x_j, twist^j, x_{j+1}) = p^j(x_{j}, x_{j+1}) r_{T^j} D^j
    //
    // r_{T^j} = [ cos(twist^j / 2), sin(twist^j / 2) T^j ]
    //
    // p^j(x_{j}, x_{j+1}) = p_{ T^i }^{ t^j(x_{j}, x_{j+1}) } is the parallel transport quaternion from the reference
    // tangent T^i to the current tangent t^j(x_{j}, x_{j+1}).
    auto locally_owned_selector =
        stk::mesh::Selector(centerline_twist_springs_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::EDGE_RANK, locally_owned_selector,
        [&node_coord_field, &node_twist_field, &edge_orientation_field, &edge_orientation_field_old,
         &edge_tangent_field, &edge_tangent_field_old, &edge_binormal_field,
         &edge_length_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &edge) {
          if (bulk_data.is_valid(edge)) {
            // Get the nodes of the edge
            stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(edge);
            MUNDY_THROW_ASSERT(bulk_data.num_nodes(edge) >= 2, std::logic_error,
                               "The edge must have at least two nodes.");
            const stk::mesh::Entity &node_i = edge_nodes[0];
            const stk::mesh::Entity &node_ip1 = edge_nodes[1];

            // Get the required input fields
            const auto node_i_coords = mundy::mesh::vector3_field_data(node_coord_field, node_i);
            const auto node_ip1_coords = mundy::mesh::vector3_field_data(node_coord_field, node_ip1);
            const double node_i_twist = stk::mesh::field_data(node_twist_field, node_i)[0];
            const auto edge_tangent_old = mundy::mesh::vector3_field_data(edge_tangent_field_old, edge);
            const auto edge_orientation_old = mundy::mesh::quaternion_field_data(edge_orientation_field_old, edge);

            // Get the output fields
            auto edge_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge);
            auto edge_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge);
            double *edge_length = stk::mesh::field_data(edge_length_field, edge);
            auto edge_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, edge);

            // Compute the un-normalized edge tangent
            edge_tangent = node_ip1_coords - node_i_coords;
            edge_length[0] = mundy::math::norm(edge_tangent);
            edge_tangent /= edge_length[0];

            // Compute the edge binormal
            edge_binormal = (2.0 * mundy::math::cross(edge_tangent_old, edge_tangent)) /
                            (1.0 + mundy::math::dot(edge_tangent_old, edge_tangent));

            // Compute the edge orientations
            const double cos_half_t = std::cos(0.5 * node_i_twist);
            const double sin_half_t = std::sin(0.5 * node_i_twist);
            const auto rot_via_twist =
                mundy::math::Quaterniond(cos_half_t, sin_half_t * edge_tangent_old[0], sin_half_t * edge_tangent_old[1],
                                         sin_half_t * edge_tangent_old[2]);
            const auto rot_via_parallel_transport =
                mundy::math::quat_from_parallel_transport(edge_tangent_old, edge_tangent);
            edge_orientation = rot_via_parallel_transport * rot_via_twist * edge_orientation_old;

            // Two things to check:
            //  1. Is the quaternion produced by the parallel transport normalized?
            //  2. Does the application of this quaternion to the old edge tangent produce the new edge tangent?

            // std::cout << "rot_via_parallel_transport: " << rot_via_parallel_transport
            //           << " has norm: " << mundy::math::norm(rot_via_parallel_transport) << std::endl;
            // std::cout << "rot_via_twist: " << rot_via_twist << " has norm: " << mundy::math::norm(rot_via_twist)
            //           << std::endl;
            // std::cout << "Edge tangent : " << edge_tangent << std::endl;
            // std::cout << " Edge tangent via transp: " << rot_via_parallel_transport * edge_tangent_old <<
            // std::endl; std::cout << " Edge tangent via orient: " << edge_orientation *
            // mundy::math::Vector3d(0.0, 0.0, 1.0)
            //           << std::endl;
          }
        });
  }

  void compute_node_curvature_and_rotation_gradient() {
    debug_print("Computing the node curvature and rotation gradient.");

    // Communicate ghosted fields.
    // TODO(palmerb4): Technically, we could avoid this communication if we compute the edge information for locally
    // owned and ghosted edges. Computation is cheaper than communication.
    stk::mesh::communicate_field_data(*bulk_data_ptr_, {edge_orientation_field_ptr_});

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_springs_part = *centerline_twist_springs_part_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;

    // Bug fix:
    // Originally this function acted on the locally owned elements of the centerline twist part, using them to fetch
    // the nodes/edges in the correct order and performing the computation. However, this assumes that the center node
    // of this element is locally owned as well. If this assumption fails, we'll end up writing the result to a shared
    // but not locally owned node. The corresponding locally owned node on a different process won't have its
    // curvature updated. That node is, thankfully, connected to a ghosted version of the element on this process, so
    // we can fix this issue by looping over all elements, including ghosted ones.
    //
    // We'll have to double check that this indeed works. I know that it will properly ensure that all locally owned
    // nodes are updated, but we also write to some non-locally owned nodes. I want to make sure that the values in
    // the non-locally owned nodes are updated using the locally-owned values. I think this is the case, but I want to
    // double check.

    // For each element in the centerline twist part, compute the node curvature at the center node.
    // The curvature can be computed from the edge orientations using
    //   kappa^i = q_i - conj(q_i) = 2 * vec(q_i)
    // where
    //   q_i = conj(d^{i-1}) d^i is the Lagrangian rotation gradient.
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::ELEMENT_RANK, centerline_twist_springs_part,
        [&edge_orientation_field, &node_curvature_field, &node_rotation_gradient_field](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Curvature needs to "know" about the order of edges, so it's best to loop over
          // the slt elements and not the nodes. Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(element);
          MUNDY_THROW_ASSERT(bulk_data.num_nodes(element) >= 3, std::logic_error,
                             "The element must have at least 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(element) >= 2, std::logic_error,
                             "The element must have at least 2 edges.");
          const stk::mesh::Entity &center_node = element_nodes[1];
          const stk::mesh::Entity &left_edge = element_edges[0];
          const stk::mesh::Entity &right_edge = element_edges[1];

          // Get the required input fields
          const auto edge_im1_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, left_edge);
          const auto edge_i_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, right_edge);

          // Get the output fields
          auto node_curvature = mundy::mesh::vector3_field_data(node_curvature_field, center_node);
          auto node_rotation_gradient = mundy::mesh::quaternion_field_data(node_rotation_gradient_field, center_node);

          // Compute the node curvature
          node_rotation_gradient = mundy::math::conjugate(edge_im1_orientation) * edge_i_orientation;
          node_curvature = 2.0 * node_rotation_gradient.vector();
        });
  }

  void compute_internal_force_and_twist_torque() {
    debug_print("Computing the internal force and twist torque.");

    // Communicate ghosted fields.
    // TODO(palmerb4): Technically, we could avoid this entire communication if we compute the edge information for
    // locally owned and ghosted edges. Computation is cheaper than communication.
    stk::mesh::communicate_field_data(
        *bulk_data_ptr_, {node_radius_field_ptr_, node_curvature_field_ptr_, node_rest_curvature_field_ptr_,
                          node_twist_field_ptr_, node_rotation_gradient_field_ptr_, edge_tangent_field_ptr_,
                          edge_binormal_field_ptr_, edge_length_field_ptr_, edge_orientation_field_ptr_});

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_springs_part = *centerline_twist_springs_part_ptr_;
    stk::mesh::Field<double> &node_radius_field = *node_radius_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_twist_torque_field = *node_twist_torque_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rest_curvature_field = *node_rest_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    const double sperm_rest_segment_length = sperm_rest_segment_length_;
    const double sperm_youngs_modulus = sperm_youngs_modulus_;
    const double sperm_poissons_ratio = sperm_poissons_ratio_;

    // Compute internal force and torque induced by differences in rest and current curvature
    auto locally_owned_selector =
        stk::mesh::Selector(centerline_twist_springs_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, element_rank_, locally_owned_selector,
        [&node_radius_field, &node_force_field, &node_twist_torque_field, &node_curvature_field,
         &node_rest_curvature_field, &node_rotation_gradient_field, &edge_tangent_field, &edge_binormal_field,
         &edge_length_field, &edge_orientation_field, &sperm_rest_segment_length, &sperm_youngs_modulus,
         &sperm_poissons_ratio](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Ok. This is a bit involved.
          // First, we need to use the node curvature to compute the induced lagrangian torque according to the
          // Kirchhoff rod model. Then, we need to use a convoluted map to take this torque to force and torque on the
          // nodes.
          //
          // The torque induced by the curvature is
          //  T = B (kappa - kappa_rest)
          // where B is the diagonal matrix of bending moduli and kappa_rest is the rest curvature. Here, the first
          // two components of curvature are the bending curvatures and the third component is the twist curvature.
          // The bending moduli are
          //  B[0,0] = E * I / l_rest, B[1,1] = E * I / l_rest, B[2,2] = 2 * G * I / l_rest
          // where l_rest is the rest length of the element, G is the shear modulus, E is the Young's modulus, and I
          // is the moment of inertia of the cross section.

          // Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(element);
          MUNDY_THROW_ASSERT(bulk_data.num_nodes(element) >= 2, std::logic_error,
                             "The element must have at least 2 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(element) >= 2, std::logic_error,
                             "The element must have at least 2 edges.");

          const stk::mesh::Entity &node_im1 = element_nodes[0];
          const stk::mesh::Entity &node_i = element_nodes[1];
          const stk::mesh::Entity &node_ip1 = element_nodes[2];
          const stk::mesh::Entity &edge_im1 = element_edges[0];
          const stk::mesh::Entity &edge_i = element_edges[1];

          // Get the required input fields
          const auto node_i_curvature = mundy::mesh::vector3_field_data(node_curvature_field, node_i);
          const auto node_i_rest_curvature = mundy::mesh::vector3_field_data(node_rest_curvature_field, node_i);
          const auto node_i_rotation_gradient =
              mundy::mesh::quaternion_field_data(node_rotation_gradient_field, node_i);
          const auto node_radius = stk::mesh::field_data(node_radius_field, node_i)[0];
          const auto edge_im1_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge_im1);
          const auto edge_i_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge_i);
          const auto edge_im1_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge_im1);
          const auto edge_i_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge_i);
          const double edge_im1_length = stk::mesh::field_data(edge_length_field, edge_im1)[0];
          const double edge_i_length = stk::mesh::field_data(edge_length_field, edge_i)[0];
          const auto edge_im1_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, edge_im1);

          // Get the output fields
          double *node_im1_force = stk::mesh::field_data(node_force_field, node_im1);
          double *node_i_force = stk::mesh::field_data(node_force_field, node_i);
          double *node_ip1_force = stk::mesh::field_data(node_force_field, node_ip1);
          double *node_im1_twist_torque = stk::mesh::field_data(node_twist_torque_field, node_im1);
          double *node_i_twist_torque = stk::mesh::field_data(node_twist_torque_field, node_i);

          // Compute the Lagrangian torque induced by the curvature
          auto delta_curvature = node_i_curvature - node_i_rest_curvature;
          const double moment_of_inertia = 0.25 * M_PI * node_radius * node_radius * node_radius * node_radius;
          const double shear_modulus = 0.5 * sperm_youngs_modulus / (1.0 + sperm_poissons_ratio);
          const double inv_rest_segment_length = 1.0 / sperm_rest_segment_length;
          auto bending_torque = mundy::math::Vector3d(
              -inv_rest_segment_length * sperm_youngs_modulus * moment_of_inertia * delta_curvature[0],
              -inv_rest_segment_length * sperm_youngs_modulus * moment_of_inertia * delta_curvature[1],
              -inv_rest_segment_length * 2 * shear_modulus * moment_of_inertia *
                  delta_curvature[2]);  // WARNING USING NEGATIVE SIGN

          // We'll reuse the bending torque for the rotated bending torque
          bending_torque =
              edge_im1_orientation * (node_i_rotation_gradient.w() * bending_torque +
                                      mundy::math::cross(node_i_rotation_gradient.vector(), bending_torque));

          // Compute the force and torque on the nodes
          const auto tmp_force_ip1 = 1.0 / edge_i_length *
                                     (mundy::math::cross(bending_torque, edge_i_tangent) +
                                      0.5 * mundy::math::dot(edge_i_tangent, bending_torque) *
                                          (mundy::math::dot(edge_i_tangent, edge_i_binormal) * edge_i_tangent) -
                                      edge_i_binormal);
          const auto tmp_force_im1 =
              1.0 / edge_im1_length *
              (mundy::math::cross(bending_torque, edge_im1_tangent) +
               0.5 * mundy::math::dot(edge_im1_tangent, bending_torque) *
                   (mundy::math::dot(edge_im1_tangent, edge_im1_binormal) * edge_im1_tangent - edge_im1_binormal));

#pragma omp atomic
          node_i_twist_torque[0] += mundy::math::dot(edge_i_tangent, bending_torque);
#pragma omp atomic
          node_im1_twist_torque[0] -= mundy::math::dot(edge_im1_tangent, bending_torque);
#pragma omp atomic
          node_ip1_force[0] += tmp_force_ip1[0];
#pragma omp atomic
          node_ip1_force[1] += tmp_force_ip1[1];
#pragma omp atomic
          node_ip1_force[2] += tmp_force_ip1[2];
#pragma omp atomic
          node_i_force[0] -= tmp_force_ip1[0] + tmp_force_im1[0];
#pragma omp atomic
          node_i_force[1] -= tmp_force_ip1[1] + tmp_force_im1[1];
#pragma omp atomic
          node_i_force[2] -= tmp_force_ip1[2] + tmp_force_im1[2];
#pragma omp atomic
          node_im1_force[0] += tmp_force_im1[0];
#pragma omp atomic
          node_im1_force[1] += tmp_force_im1[1];
#pragma omp atomic
          node_im1_force[2] += tmp_force_im1[2];
        });

    // Compute internal force induced by differences in rest and current length
    mundy::mesh::for_each_entity_run(
        bulk_data, edge_rank_, locally_owned_selector,
        [&node_radius_field, &node_force_field, &edge_tangent_field, &edge_length_field, &sperm_rest_segment_length,
         &sperm_youngs_modulus,
         &sperm_poissons_ratio](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &edge) {
          // F_left = k (l - l_rest) tangent
          // F_right = -k (l - l_rest) tangent
          //
          // k can be computed using the material properties of the rod according to k = E A / l_rest where E is the
          // Young's modulus, A is the cross-sectional area, and l_rest is the rest length of the rod.

          // Get the lower rank entities
          stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(edge);
          MUNDY_THROW_ASSERT(bulk_data.num_nodes(edge) >= 2, std::logic_error, "The edge must have at least 2 nodes.");
          const stk::mesh::Entity &node_im1 = edge_nodes[0];
          const stk::mesh::Entity &node_i = edge_nodes[1];

          // Get the required input fields
          const auto edge_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge);
          const double edge_length = stk::mesh::field_data(edge_length_field, edge)[0];
          const auto node_radius = stk::mesh::field_data(node_radius_field, node_i)[0];

          // Get the output fields
          double *node_im1_force = stk::mesh::field_data(node_force_field, node_im1);
          double *node_i_force = stk::mesh::field_data(node_force_field, node_i);

          // Compute the internal force
          const double spring_constant =
              sperm_youngs_modulus * M_PI * node_radius * node_radius / sperm_rest_segment_length;
          const auto right_node_force = -spring_constant * (edge_length - sperm_rest_segment_length) * edge_tangent;
#pragma omp atomic
          node_im1_force[0] -= right_node_force[0];
#pragma omp atomic
          node_im1_force[1] -= right_node_force[1];
#pragma omp atomic
          node_im1_force[2] -= right_node_force[2];
#pragma omp atomic
          node_i_force[0] += right_node_force[0];
#pragma omp atomic
          node_i_force[1] += right_node_force[1];
#pragma omp atomic
          node_i_force[2] += right_node_force[2];
        });

    // Sum the node force and torque over shared nodes.
    stk::mesh::parallel_sum(*bulk_data_ptr_, {node_force_field_ptr_, node_twist_torque_field_ptr_});
  }

  void compute_centerline_twist_force_and_torque() {
    debug_print("Computing the centerline twist force and torque.");
    // Compute the edge information in the current timestep (length, tangent, and binormal).
    compute_edge_information();

    // Compute the node curvature and rotation gradient.
    compute_node_curvature_and_rotation_gradient();

    // Compute the internal force and twist torque.
    compute_internal_force_and_twist_torque();
  }

  void update_generalized_position_velocity_and_acceleration() {
    debug_print("Updating the generalized position, velocity, and acceleration.");
    //  Prediction:
    //   v(t + dt) = v(t) + dt (1 - gamma) a(t)
    //   a(t + dt) = M^{-1} [ f(x(t + dt) - C v(t + dt) ]
    //
    //  Correction:
    //   x(t + dt) = x(t + dt) + dt^2 beta a(t + dt)
    //   v(t + dt) = v(t + dt) + dt gamma a(t + dt)
    // beta = 0.25, gamma = 0.5 -> unconditionally stable, constant-average acceleration method

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_springs_part = *centerline_twist_springs_part_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field_old = node_velocity_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_acceleration_field = *node_acceleration_field_ptr_;
    stk::mesh::Field<double> &node_acceleration_field_old = node_acceleration_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_radius_field = *node_radius_field_ptr_;

    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &node_twist_velocity_field = *node_twist_velocity_field_ptr_;
    stk::mesh::Field<double> &node_twist_velocity_field_old =
        node_twist_velocity_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_acceleration_field = *node_twist_acceleration_field_ptr_;
    stk::mesh::Field<double> &node_twist_acceleration_field_old =
        node_twist_acceleration_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_torque_field = *node_twist_torque_field_ptr_;
    const double time_step_size = timestep_size_;
    const double sperm_density = sperm_density_;

    // Solve the prediction and correction steps simultaniously
    stk::mesh::Selector locally_owned_selector = centerline_twist_springs_part & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, node_rank_, locally_owned_selector,
        [&node_coord_field, &node_velocity_field, &node_velocity_field_old, &node_acceleration_field,
         &node_acceleration_field_old, &node_force_field, &node_radius_field, &node_twist_field,
         &node_twist_velocity_field, &node_twist_velocity_field_old, &node_twist_acceleration_field,
         &node_twist_acceleration_field_old, &node_twist_torque_field, &time_step_size,
         &sperm_density]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const double node_radius = stk::mesh::field_data(node_radius_field, node)[0];
          const auto node_force = mundy::mesh::vector3_field_data(node_force_field, node);
          const double node_twist_torque = stk::mesh::field_data(node_twist_torque_field, node)[0];
          const auto node_velocity_old = mundy::mesh::vector3_field_data(node_velocity_field_old, node);
          const auto node_acceleration_old = mundy::mesh::vector3_field_data(node_acceleration_field_old, node);
          const double node_twist_velocity_old = stk::mesh::field_data(node_twist_velocity_field_old, node)[0];
          const double node_twist_acceleration_old = stk::mesh::field_data(node_twist_acceleration_field_old, node)[0];

          // Get the output fields
          auto node_coord = mundy::mesh::vector3_field_data(node_coord_field, node);
          auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node);
          auto node_acceleration = mundy::mesh::vector3_field_data(node_acceleration_field, node);
          double *node_twist = stk::mesh::field_data(node_twist_field, node);
          double *node_twist_velocity = stk::mesh::field_data(node_twist_velocity_field, node);
          double *node_twist_acceleration = stk::mesh::field_data(node_twist_acceleration_field, node);

          // Compute the mass and moment of inertia assuming sphere mass lumping at the nodes
          const double node_radius2 = node_radius * node_radius;
          const double node_radius3 = node_radius2 * node_radius;
          const double node_mass = 4.0 / 3.0 * M_PI * node_radius3 * sperm_density;
          const double node_mass_moment_of_inertia = 2.0 / 5.0 * node_mass * node_radius2;

          // Prediction step
          //   v(t + dt) = v(t) + dt (1 - gamma) a(t)
          //   a(t + dt) = M^{-1} [ f(x(t + dt) - C v(t + dt) ]
          //
          // Here, we break C into a constant translational damping and a constant twist damping.
          const double gamma = 0.5;
          const double beta = 0.25;
          const double translational_damping = 1.0;
          const double twist_damping = 1.0;
          const double coeff1 = (1.0 - gamma) * time_step_size;
          const auto pred_node_velocity = node_velocity_old + coeff1 * node_acceleration_old;
          const auto pred_node_twist_velocity = node_twist_velocity_old + coeff1 * node_twist_acceleration_old;

          node_acceleration = 1.0 / node_mass * (node_force - translational_damping * node_velocity_old);
          node_twist_acceleration[0] =
              1.0 / node_mass_moment_of_inertia * (node_twist_torque - twist_damping * node_twist_velocity_old);

          // Correction step
          //   x(t + dt) = x(t + dt) + dt^2 beta a(t + dt)
          //   v(t + dt) = v(t + dt) + dt gamma a(t + dt)
          const double coeff2 = time_step_size * time_step_size * beta;
          node_coord += coeff2 * node_acceleration;
          node_twist[0] += coeff2 * node_twist_acceleration[0];

          const double coeff3 = time_step_size * gamma;
          node_velocity = pred_node_velocity + coeff3 * node_acceleration;
          node_twist_velocity[0] = pred_node_twist_velocity + coeff3 * node_twist_acceleration[0];
        });
  }

  void clamp_edge1() {
    debug_print("Clamping edge 0.");

    // Clamping the first edge equates to setting the velocity and accelerations (both positional and twist) of the
    // first two nodes to zero.
    stk::mesh::Entity node1 = bulk_data_ptr_->get_entity(node_rank_, 1);
    stk::mesh::Entity node2 = bulk_data_ptr_->get_entity(node_rank_, 2);

    if (bulk_data_ptr_->is_valid(node1)) {
      mundy::mesh::vector3_field_data(*node_velocity_field_ptr_, node1).set(0.0, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_acceleration_field_ptr_, node1).set(0.0, 0.0, 0.0);
      stk::mesh::field_data(*node_twist_velocity_field_ptr_, node1)[0] = 0.0;
      stk::mesh::field_data(*node_twist_acceleration_field_ptr_, node1)[0] = 0.0;
    }

    if (bulk_data_ptr_->is_valid(node2)) {
      mundy::mesh::vector3_field_data(*node_velocity_field_ptr_, node2).set(0.0, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_acceleration_field_ptr_, node2).set(0.0, 0.0, 0.0);
      stk::mesh::field_data(*node_twist_velocity_field_ptr_, node2)[0] = 0.0;
      stk::mesh::field_data(*node_twist_acceleration_field_ptr_, node2)[0] = 0.0;
    }
  }

  void disable_twist() {
    debug_print("Disabling twist.");

    // Set the twist, twist, velocity, and twist acceleration to zero.
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_field_ptr_, std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_velocity_field_ptr_, std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_acceleration_field_ptr_, std::array<double, 1>{0.0});
  }

  void run(int argc, char **argv) {
    debug_print("Running the simulation.");

    // Preprocess
    parse_user_inputs(argc, argv);
    dump_user_inputs();

    // Setup
    timestep_index_ = 0;
    build_our_mesh_and_method_instances();
    fetch_fields_and_parts();
    declare_and_initialize_sperm();
    propogate_rest_curvature();
    setup_io();

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    Kokkos::Timer timer;
    for (; timestep_index_ < num_time_steps_; timestep_index_++) {
      debug_print(std::string("Time step ") + std::to_string(timestep_index_) + " of " +
                  std::to_string(num_time_steps_));

      // IO. If desired, write out the data for time t.
      if (timestep_index_ % io_frequency_ == 0) {
        stk_io_broker_.begin_output_step(output_file_index_, static_cast<double>(timestep_index_));
        stk_io_broker_.write_defined_output_fields(output_file_index_);
        stk_io_broker_.end_output_step(output_file_index_);
        stk_io_broker_.flush_output();
      }

      // Prepare the current configuration.
      {
        // Rotate the field states.
        rotate_field_states();

        // Predict the motion from t -> t + dt.
        //   x(t + dt) = x(t) + dt v(t) + 0.5 dt^2 (1 - 2 beta) a(t)
        update_generalized_position();

        // Update the rest curvature.
        // kappa_rest(t + dt) = amplitude * sin(spacial_frequency * archlength + temporal_frequency * (t + dt)).
        if (timestep_index_ > 5000000) {
          propogate_rest_curvature();
        }

        // Zero the node velocities, accelerations, and forces/torques for time t + dt.
        zero_out_transient_node_fields();
      }

      // Evaluate forces f(x(t + dt)).
      {
        // Centerline twist rod model
        compute_centerline_twist_force_and_torque();
      }

      // Compute velocity and acceleration.
      {
        // Update the position, velocity, and acceleration for t + dt.
        //
        //  Prediction:
        //   v(t + dt) = v(t) + dt (1 - gamma) a(t)
        //   a(t + dt) = M^{-1} [ f(x(t + dt) - C v(t + dt) ]
        //
        //  Correction:
        //   x(t + dt) = x(t + dt) + dt^2 beta a(t + dt)
        //   v(t + dt) = v(t + dt) + dt gamma a(t + dt)
        //
        // Note, the above is entirely local to a given node, so there's no need to break it into multiple loops.
        update_generalized_position_velocity_and_acceleration();

        // Apply constraints
        // clamp_edge1();
        disable_twist();
      }
    }

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps_);
      std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
    }
  }

 private:
  //! \name Useful aliases
  //@{

  static constexpr auto node_rank_ = stk::topology::NODE_RANK;
  static constexpr auto edge_rank_ = stk::topology::EDGE_RANK;
  static constexpr auto element_rank_ = stk::topology::ELEMENT_RANK;
  //@}

  //! \name Internal state
  //@{

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_;
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr_;
  stk::io::StkMeshIoBroker stk_io_broker_;
  size_t output_file_index_;
  size_t timestep_index_ = 0;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<double> *node_coord_field_ptr_;
  stk::mesh::Field<double> *node_velocity_field_ptr_;
  stk::mesh::Field<double> *node_force_field_ptr_;
  stk::mesh::Field<double> *node_acceleration_field_ptr_;
  stk::mesh::Field<double> *node_twist_field_ptr_;
  stk::mesh::Field<double> *node_twist_velocity_field_ptr_;
  stk::mesh::Field<double> *node_twist_torque_field_ptr_;
  stk::mesh::Field<double> *node_twist_acceleration_field_ptr_;
  stk::mesh::Field<double> *node_curvature_field_ptr_;
  stk::mesh::Field<double> *node_rest_curvature_field_ptr_;
  stk::mesh::Field<double> *node_rotation_gradient_field_ptr_;
  stk::mesh::Field<double> *node_radius_field_ptr_;
  stk::mesh::Field<double> *node_archlength_field_ptr_;

  stk::mesh::Field<double> *edge_orientation_field_ptr_;
  stk::mesh::Field<double> *edge_tangent_field_ptr_;
  stk::mesh::Field<double> *edge_binormal_field_ptr_;
  stk::mesh::Field<double> *edge_length_field_ptr_;

  stk::mesh::Field<double> *element_radius_field_ptr_;
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr_;
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr_;
  stk::mesh::Field<double> *element_rest_length_field_ptr_;
  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *centerline_twist_springs_part_ptr_;
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
  //@}

  //! \name User parameters
  //@{
  std::string input_file_name_ = "input.yaml";

  size_t num_sperm_ = 1;
  size_t num_nodes_per_sperm_ = 10;
  double sperm_radius_ = 0.5;
  double sperm_initial_segment_length_ = 2 * sperm_radius_;
  double sperm_rest_segment_length_ = 2 * sperm_radius_;
  double sperm_rest_curvature_twist_ = 0.0;
  double sperm_rest_curvature_bend1_ = 0.0;
  double sperm_rest_curvature_bend2_ = 0.0;

  double sperm_youngs_modulus_ = 1000.0;
  double sperm_poissons_ratio_ = 0.3;
  double sperm_density_ = 1.0;

  size_t num_time_steps_ = 1000;
  double timestep_size_ = 0.01;
  size_t io_frequency_ = 10;
  //@}
};  // SpermSimulation

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  SpermSimulation().run(argc, argv);

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
