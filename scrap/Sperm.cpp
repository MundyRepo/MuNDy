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

The goal of this example is to simulate the swimming motion of long sperm confined to a 2D plane. For the sake of
collisions, the sperm are a chain of spherocylinder segments. For the sake of particle motion, we use "mass lumping" to
represent the chain as a collection of spheres located at the endpoints of each segment. To model inextensibility, we
add distance-based Hookean springs between adjacent spheres. To model flexibility, we employ a finite difference
discretization of a Kirchhoff rod model with a centerline-twist parametrization of the rod. To model the sperm's
swimming motion, we vary the rest curvature with time and archlength along the chain.

We will initialize the sperm as straight lines within the x-y plane. They can either all be initialized in the same
direction or in alternating direction. You can also, optionally, enable, two boundary sperm at the top and bottom of the
domain. These sperm will be able to move, swim, and collide with the other sperm, BUT we'll only consider one way
collision forces. As in, the boundary perm can exert a collision force on the bulk sperm, but the bulk sperm can't exert
a collision force on the boundary sperm.
  x---x---x---x---x---x---x---x
    <-o---o---o---o---o---o
    <-o---o---o---o---o---o
    <-o---o---o---o---o---o
    <-o---o---o---o---o---o
  x---x---x---x---x---x---x---x
or
  x---x---x---x---x---x---x---x
    <-o---o---o---o---o---o
      o---o---o---o---o---o->
    <-o---o---o---o---o---o
      o---o---o---o---o---o->
  x---x---x---x---x---x---x---x
The coordinate mapping used to set the initial position of the sperm in the bulk may differ from the coordinate mapping
used to set the initial position of the sperm on the boundary. The two choices of coordinate mapping are straight and
sinusoidal.

To start, we won't employ Mundy's requirements encapsulation. That's supposed to come last in the development process.
This can't be emphasized more: the goal is to get something working first and then wrap it in a MetaMethod so it can be
integrated into Mundy and runnable via our Configurator/Driver system.
*/

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>   // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>            // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>        // for stk::mesh::Selector
#include <stk_topology/topology.hpp>         // for stk::topology
#include <stk_util/parallel/Parallel.hpp>    // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_core/MakeStringArray.hpp>                   // for mundy::core::make_string_array
#include <mundy_core/throw_assert.hpp>                      // for MUNDY_THROW_ASSERT
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>                  // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>                // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>                 // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                         // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>  // for mundy::...::SpherocylinderSegmentSpherocylinderSegmentLinkers
#include <mundy_math/Matrix3.hpp>     // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>    // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_mesh/utils/FillFieldWithValue.hpp>            // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs
#include <mundy_shapes/ComputeAABB.hpp>     // for mundy::shapes::ComputeAABB

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
///   - Declare and initialize the chain of rods and their connecting springs, and the STL elements
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
///         // Neighbor detection rod-rod
///         - Check if the rod-rod neighbor list needs updated or not
///             - Compute the AABBs for the rods
///              (Using mundy's ComputeAABB function)
///
///             - Delete rod-rod neighbor linkers that are too far apart
///              (Using the DestroyDistantNeighbors technique of mundy's DestroyNeighborLinkers function)
///
///             - Generate neighbor linkers between nearby rods
///              (Using the GenerateNeighborLinkers function of mundy's GenerateNeighborLinkers function)
///
///         // Hertzian contact
///         - Compute the signed separation distance and contact normal between neighboring rods
///          (Using mundy's ComputeSignedSeparationDistanceAndContactNormal function)
///
///         - Evaluate the Hertzian contact potential between neighboring rods
///          (Using mundy's EvaluateLinkerPotentials function)
///
///         - Sum the linker potential force to get the induced node force on each rod
///          (Using mundy's LinkerPotentialForceReduction function)
///
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

  bool parse_user_inputs(int argc, char **argv) {
    debug_print("Parsing user inputs.");

    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, true);

    // Optional command line arguments for controlling
    //   Sperm initialization:
    cmdp.setOption("num_sperm", &num_sperm_, "Number of sperm.");
    cmdp.setOption("num_nodes_per_sperm", &num_nodes_per_sperm_, "Number of nodes per sperm.");
    cmdp.setOption("sperm_radius", &sperm_radius_, "The radius of each sperm.");
    cmdp.setOption("sperm_initial_segment_length", &sperm_initial_segment_length_, "Initial sperm segment length.");
    cmdp.setOption("sperm_rest_segment_length", &sperm_rest_segment_length_, "Rest sperm segment length.");
    cmdp.setOption("sperm_rest_curvature_twist", &sperm_rest_curvature_twist_, "Rest curvature (twist) of the sperm.");
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
    cmdp.setOption("skin_distance", &skin_distance_, "Skin distance for neighbor detection. \n As long as no particle moves more than 
    one skin distance, the current neighbor list is valid. This is used to avoid the expensive neighbor detection step.");
    cmdp.setOption("io_frequency", &io_frequency_, "Number of timesteps between writing output.");


    bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    return was_parse_successful;
  }

  void check_input_parameters() {
    debug_print("Checking input parameters.");
    MUNDY_THROW_REQUIRE(num_sperm_ > 0, std::invalid_argument, "num_sperm_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(num_nodes_per_sperm_ > 0, std::invalid_argument, "num_nodes_per_sperm_ must be greater than 0.");
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
    MUNDY_THROW_REQUIRE(skin_distance_ > -1e-12, std::invalid_argument,
                       "skin_distance_ must be greater than or equal to 0.");
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
      std::cout << "  skin_distance_: " << skin_distance_ << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  void build_our_mesh_and_method_instances() {
    debug_print("Building our mesh and method instances.");

    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements for this example. These are requirements that exceed those of the enabled methods and
    // allow us to extend the functionality offered natively by Mundy.
    //
    // We require that the centerline twist springs part exists, has a BEAM_3 topology, and has the desired
    // node/edge/element fields.
    auto clt_part_reqs = std::make_shared<mundy::meta::PartReqs>()
                             ->set_part_name("CENTERLINE_TWIST_SPRINGS")
                             .set_part_topology(stk::topology::BEAM_3)

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

                             // Add the edge fields
                             .add_field_reqs<double>("EDGE_ORIENTATION", edge_rank_, 4, 2)
                             .add_field_reqs<double>("EDGE_TANGENT", edge_rank_, 3, 2)
                             .add_field_reqs<double>("EDGE_BINORMAL", edge_rank_, 3, 1)
                             .add_field_reqs<double>("EDGE_LENGTH", edge_rank_, 1, 1);

    // Create the mesh requirements
    mesh_reqs_ptr_->add_and_sync_part_reqs(clt_part_reqs);

    // ComputeSignedSeparationDistanceAndContactNormal fixed parameters
    auto compute_ssd_and_cn_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER"));
    mesh_reqs_ptr_->sync(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params));

    // ComputeAABB fixed parameters
    auto compute_aabb_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));
    mesh_reqs_ptr_->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params));

    // GenerateNeighborLinkers fixed parameters
    auto generate_neighbor_linkers_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKERS"))
            .sublist("STK_SEARCH")
            .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"))
            .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"));
    mesh_reqs_ptr_->sync(
        mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(generate_neighbor_linkers_fixed_params));

    // EvaluateLinkerPotentials fixed parameters
    auto evaluate_linker_potentials_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names",
        mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));
    mesh_reqs_ptr_->sync(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params));

    // LinkerPotentialForceReduction fixed parameters
    auto linker_potential_force_reduction_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));
    mesh_reqs_ptr_->sync(mundy::linkers::LinkerPotentialForceReduction::get_mesh_requirements(
        linker_potential_force_reduction_fixed_params));

    // DestroyNeighborLinkers fixed parameters
    auto destroy_neighbor_linkers_fixed_params =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params));

    // DeclareAndInitConstraints fixed parameters
    auto declare_and_init_constraints_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "CHAIN_OF_SPRINGS")
            .sublist("CHAIN_OF_SPRINGS")
            .set("hookean_springs_part_names",
                 mundy::core::make_string_array(mundy::constraints::HookeanSprings::get_name()))
            .set("angular_springs_part_names",
                 mundy::core::make_string_array(mundy::constraints::AngularSprings::get_name()))
            .set("sphere_part_names", mundy::core::make_string_array(mundy::shapes::Spheres::get_name()))
            .set("spherocylinder_segment_part_names",
                 mundy::core::make_string_array(mundy::shapes::SpherocylinderSegments::get_name()))
            .set<bool>("generate_hookean_springs", true)
            .set<bool>("generate_angular_springs", false)
            .set<bool>("generate_spheres_at_nodes", false)
            .set<bool>("generate_spherocylinder_segments_along_edges", true);
    mesh_reqs_ptr_->sync(mundy::constraints::DeclareAndInitConstraints::get_mesh_requirements(
        declare_and_init_constraints_fixed_params));

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDS");
    meta_data_ptr_->commit();

    // Create the class instances and populate their mutable parameters
    node_euler_ptr_ = NodeEuler::create_new_instance(bulk_data_ptr_.get(), node_euler_fixed_params);
    compute_ssd_and_cn_ptr_ = ComputeSignedSeparationDistanceAndContactNormal::create_new_instance(
        bulk_data_ptr_.get(), compute_ssd_and_cn_fixed_params);
    compute_aabb_ptr_ = ComputeAABB::create_new_instance(bulk_data_ptr_.get(), compute_aabb_fixed_params);
    generate_neighbor_linkers_ptr_ =
        GenerateNeighborLinkers::create_new_instance(bulk_data_ptr_.get(), generate_neighbor_linkers_fixed_params);
    evaluate_linker_potentials_ptr_ =
        EvaluateLinkerPotentials::create_new_instance(bulk_data_ptr_.get(), evaluate_linker_potentials_fixed_params);
    linker_potential_force_reduction_ptr_ = LinkerPotentialForceReduction::create_new_instance(
        bulk_data_ptr_.get(), linker_potential_force_reduction_fixed_params);
    destroy_neighbor_linkers_ptr_ =
        DestroyNeighborLinkers::create_new_instance(bulk_data_ptr_.get(), destroy_neighbor_linkers_fixed_params);
    declare_and_init_constraints_ptr_ =
        DeclareAndInitConstraints::create_new_instance(bulk_data_ptr_.get(), declare_and_init_constraints_fixed_params);

    // Set up the mutable parameters for the classes
    // If a class doesn't have mutable parameters, we can skip setting them.

    // ComputeCenterlineTwistSpringConstraintForce mutable parameters
    auto compute_centerline_twist_constraint_force_mutable_params =
        Teuchos::ParameterList()
            .set("sperm_youngs_modulus_", sperm_youngs_modulus_)
            .set("sperm_poissons_ratio_", sperm_poissons_ratio_)
            .set("shear_modulus", sperm_youngs_modulus_ / (2.0 * (1.0 + sperm_poissons_ratio_)))
            .set("density", 1.0);
    compute_centerline_twist_constraint_force_ptr_->set_mutable_params(
        compute_centerline_twist_constraint_force_mutable_params);

    // ComputeAABB mutable parameters
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", 0.0);
    compute_aabb_ptr_->set_mutable_params(compute_aabb_mutable_params);
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
    node_coordinates_field_ptr_ = fetch_field<double>("NODE_COORDS", stk::topology::NODE_RANK);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", stk::topology::NODE_RANK);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", stk::topology::NODE_RANK);
    node_acceleration_field_ptr_ = fetch_field<double>("NODE_ACCELERATION", stk::topology::NODE_RANK);
    node_twist_field_ptr_ = fetch_field<double>("NODE_TWIST", stk::topology::NODE_RANK);
    node_twist_velocity_field_ptr_ = fetch_field<double>("NODE_TWIST_VELOCITY", stk::topology::NODE_RANK);
    node_twist_torque_field_ptr_ = fetch_field<double>("NODE_TWIST_TORQUE", stk::topology::NODE_RANK);
    node_twist_acceleration_field_ptr_ = fetch_field<double>("NODE_TWIST_ACCELERATION", stk::topology::NODE_RANK);
    node_curvature_field_ptr_ = fetch_field<double>("NODE_CURVATURE", stk::topology::NODE_RANK);
    node_rest_curvature_field_ptr_ = fetch_field<double>("NODE_REST_CURVATURE", stk::topology::NODE_RANK);
    node_rotation_gradient_field_ptr_ = fetch_field<double>("NODE_ROTATION_GRADIENT", stk::topology::NODE_RANK);
    node_radius_field_ptr_ = fetch_field<double>("NODE_RADIUS", stk::topology::NODE_RANK);

    edge_orientation_field_ptr_ = fetch_field<double>("EDGE_ORIENTATION", stk::topology::EDGE_RANK);
    edge_tangent_field_ptr_ = fetch_field<double>("EDGE_TANGENT", stk::topology::EDGE_RANK);
    edge_binormal_field_ptr_ = fetch_field<double>("EDGE_BINORMAL", stk::topology::EDGE_RANK);
    edge_length_field_ptr_ = fetch_field<double>("EDGE_LENGTH", stk::topology::EDGE_RANK);

    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", stk::topology::ELEMENT_RANK);
    element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", stk::topology::ELEMENT_RANK);
    element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", stk::topology::ELEMENT_RANK);
    element_rest_length_field_ptr_ = fetch_field<double>("ELEMENT_REST_LENGTH", stk::topology::ELEMENT_RANK);
    element_spring_constant_field_ptr_ = fetch_field<double>("ELEMENT_SPRING_CONSTANT", stk::topology::ELEMENT_RANK);

    // Fetch the parts
    centerline_twist_springs_part_ptr_ = fetch_part("CENTERLINE_TWIST_SPRINGS");
    spherocylinder_segments_part_ptr_ = fetch_part("SPHEROCYLINDER_SEGMENTS");
    hookean_springs_part_ptr_ = fetch_part("HOOKEAN_SPRINGS");
  }

  void setup_io() {
    debug_print("Setting up IO.");

    // Declare each part as an IO part
    stk::io::put_io_part_attribute(*centerline_twist_springs_part_ptr_);
    stk::io::put_io_part_attribute(*spherocylinder_segments_part_ptr_);
    stk::io::put_io_part_attribute(*hookean_springs_part_ptr_);

    // Setup the IO broker
    stk_io_broker_.use_simple_fields();
    stk_io_broker_.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()));

    output_file_index_ = stk_io_broker_.create_output_mesh("Sperm.exo", stk::io::WRITE_RESULTS);
    stk_io_broker_.add_field(output_file_index_, *node_coordinates_field_ptr_);
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

    // Declare N spring chains with a slight shift to each chain
    for (int i = 0; i < num_sperm_; i++) {
      // DeclareAndInitConstraints mutable parameters
      Teuchos::ParameterList declare_and_init_constraints_mutable_params;
      using CoordinateMappingType =
          mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
      using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::StraightLine;

      double center_x = 0.0;
      double center_y = 2 * i * sperm_radius_;
      double center_z = 0.0;
      double length = num_nodes_per_sperm_ * sperm_initial_segment_length_;
      double axis_x = 1.0;
      double axis_y = 0.0;
      double axis_z = 0.0;
      auto straight_line_mapping_ptr = std::make_shared<OurCoordinateMappingType>(
          num_nodes_per_sperm_, center_x, center_y, center_z, length, axis_x, axis_y, axis_z);
      declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
          .set<size_t>("num_nodes", num_nodes_per_sperm_)
          .set<size_t>("node_id_start", i * num_nodes_per_sperm_ + 1)
          .set<size_t>("element_id_start", i * (4 * num_nodes_per_sperm_ - 4) + 1)
          .set("hookean_spring_constant", spring_constant)
          .set("hookean_spring_rest_length", sperm_rest_segment_length_)
          .set("angular_spring_constant", angular_spring_constant)
          .set("angular_spring_rest_angle", angular_spring_rest_angle)
          .set("sperm_radius_", sperm_radius_)
          .set("spherocylinder_segment_radius", sperm_radius_)
          .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", straight_line_mapping_ptr);
      declare_and_init_constraints_ptr_->set_mutable_params(declare_and_init_constraints_mutable_params);
      declare_and_init_constraints_ptr_->execute();
    }

    mundy::mesh::utils::fill_field_with_value<double>(*element_youngs_modulus_field_ptr_,
                                                      std::array<double, 1>{sperm_youngs_modulus_});
    mundy::mesh::utils::fill_field_with_value<double>(*element_poissons_ratio_field_ptr_,
                                                      std::array<double, 1>{sperm_poissons_ratio_});
  }

  void loadbalance() {
    debug_print("Load balancing the mesh.");
    stk::balance::balanceStkMesh(balance_settings_, *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()));
  }

  void rotate_field_states() {
    debug_print("Rotating the field states.");
    bulk_data_ptr_->update_field_data_states();
  }

  void update_position_and_twist() {
    debug_print("Updating the position and twist.");
    // x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2.

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_coordinates_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &node_coordinates_field_old = node_coordinates_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_velocity_field_old = node_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_acceleration_field_old =
        node_acceleration_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &node_twist_field_old = node_twist_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_velocity_field_old =
        node_twist_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_acceleration_field_old =
        node_twist_acceleration_field_ptr_->field_of_state(stk::mesh::StateN);
    const time_step_size = timestep_size_;

    // Update the position and twist of the nodes
    auto locally_owned_selector = stk::mesh::Selector(centerline_twist_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, node_rank_, locally_owned_selector,
        [&node_coordinates_field, &node_coordinates_field_old, &node_velocity_field_old, &node_acceleration_field_old,
         &node_twist_field, &node_twist_field_old, &node_twist_velocity_field_old, &node_twist_acceleration_field_old,
         time_step_size](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const auto node_coords_old = mundy::mesh::vector3_field_data(node_coordinates_field_old, node);
          const auto node_velocity_old = mundy::mesh::vector3_field_data(node_velocity_field_old, node);
          const auto node_acceleration_old = mundy::mesh::vector3_field_data(node_acceleration_field_old, node);
          const double node_twist_old = stk::mesh::field_data(node_twist_field_old, node)[0];
          const double node_twist_rate_old = stk::mesh::field_data(node_twist_rate_field_old, node)[0];
          const double node_twist_acceleration_old = stk::mesh::field_data(node_twist_acceleration_field_old, node)[0];

          // Get the output fields
          auto node_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node);
          double *node_twist = stk::mesh::field_data(node_twist_field, node);

          // Update the current configuration
          node_coords = node_coords_old + node_velocity_old * time_step_size +
                        0.5 * node_acceleration_old * time_step_size * time_step_size;
          node_twist[0] = node_twist_old + node_twist_rate_old * time_step_size +
                          0.5 * node_twist_acceleration_old * time_step_size * time_step_size;
        });
  }

  void compute_aabb() {
    debug_print("Computing the AABB.");
    compute_aabb_ptr_->execute();
  }

  void destroy_distant_neighbors() {
    debug_print("Destroying distant neighbors.");
    destroy_neighbor_linkers_ptr_->execute();
  }

  void generate_neighbor_linkers() {
    debug_print("Generating neighbor linkers.");
    generate_neighbor_linkers_ptr_->execute();
  }

  void compute_ssd_and_cn() {
    debug_print("Computing the signed separation distance and contact normal.");
    compute_ssd_and_cn_ptr_->execute();
  }

  void evaluate_linker_potentials() {
    debug_print("Evaluating the linker potentials.");
    evaluate_linker_potentials_ptr_->execute();
  }

  void linker_potential_force_reduction() {
    debug_print("Reducing the linker potential force.");
    linker_potential_force_reduction_ptr_->execute();
  }

  void compute_edge_information() {
    debug_print("Computing the edge information.");

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_coordinates_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field_old = edge_tangent_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;

    // For each edge in the centerline twist part, compute the edge tangent, binormal, and length.
    // length^i = ||x_{i+1} - x_i||
    // edge_tangent^i = (x_{i+1} - x_i) / length
    // edge_binormal^i = (2 edge_tangent_old^i x edge_tangent^i) / (1 + edge_tangent_old^i dot edge_tangent^i)
    auto locally_owned_selector = stk::mesh::Selector(centerline_twist_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::EDGE_RANK, locally_owned_selector,
        [&node_coordinates_field, &edge_orientation_field, &edge_tangent_field, &edge_tangent_field_old,
         &edge_binormal_field,
         &edge_length_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &edge) {
          // Get the nodes of the edge
          stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(edge);
          stk::mesh::Entity const &node_i = edge_nodes[0];
          stk::mesh::Entity const &node_ip1 = edge_nodes[1];

          // Get the required input fields
          const auto node_i_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node_i);
          const auto node_ip1_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node_ip1);
          const auto edge_tangent_old = mundy::mesh::vector3_field_data(edge_tangent_field_old, edge);

          // Get the output fields
          auto edge_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge);
          auto edge_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge);
          double* edge_length = stk::mesh::field_data(edge_length_field, edge);

          // Compute the un-normalized edge tangent
          edge_tangent = node_ip1_coords - node_i_coords;
          edge_length[0] = mundy::math::norm(edge_tangent);
          edge_tangent /= edge_length[0];

          // Compute the edge binormal
          edge_binormal = (2.0 * mundy::math::cross(edge_tangent_old, edge_tangent)) /
                          (1.0 + mundy::math::dot(edge_tangent_old, edge_tangent));
        });
  }

  void compute_node_curvature_and_rotation_gradient() {
    debug_print("Computing the node curvature and rotation gradient.");

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;

    // For each element in the centerline twist part, compute the node curvature at the center node.
    // The curvature can be computed from the edge orientations using
    //   kappa^i = q_i - conj(q_i) = 2 * vec(q_i)
    // where
    //   q_i = conj(d^{i-1}) d^i is the Lagrangian rotation gradient.
    auto locally_owned_selector = stk::mesh::Selector(centerline_twist_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::ELEMENT_RANK, locally_owned_selector,
        [&edge_orientation_field, &node_curvature_field, &node_rotation_gradient_field](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Curvature needs to "know" about the order of edges, so it's best to loop over
          // the slt elements and not the nodes. Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(element);

          MUNDY_THROW_ASSERT(bulk_data.num_nodes(element) == 3, std::logic_error,
                             "Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(element) == 2, std::logic_error,
                             "Elements must have exactly 2 edges.");

          // Get the required input fields
          const auto edge_im1_orientation =
              mundy::mesh::quaternion_field_data(edge_orientation_field, element_edges[0]);
          const auto edge_i_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, element_edges[1]);

          // Get the output fields
          auto node_curvature = mundy::mesh::vector3_field_data(node_curvature_field, element_nodes[1]);
          auto node_rotation_gradient =
              mundy::mesh::quaternion_field_data(node_rotation_gradient_field, element_nodes[1]);

          // Compute the node curvature
          node_rotation_gradient = mundy::math::conj(edge_im1_orientation) * edge_i_orientation;
          node_curvature = 2.0 * node_rotation_gradient.vector();
        });
  }

  void compute_internal_force_and_twist_torque() {
    debug_print("Computing the internal force and twist torque.");

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_twist_torque_field = *node_twist_torque_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rest_curvature_field = *node_rest_curvature_field_ptr_;
    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;

    // Compute internal force and torque induced by differences in rest and current curvature
    auto locally_owned_selector = stk::mesh::Selector(centerline_twist_part) & meta_data_ptr_->locally_owned_part();
    mundy::mesh::for_each_entity_run(
        bulk_data, element_rank_, locally_owned_selector,
        [&node_force_field, &node_twist_torque_field, &node_curvature_field, &node_rest_curvature_field,
         &node_twist_field, &node_rotation_gradient_field, &edge_tangent_field, &edge_binormal_field,
         &edge_length_field,
         &edge_orientation_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Ok. This is a bit involved.
          // First, we need to use the node curvature to compute the induced lagrangian torque according to the
          // Kirchhoff rod model. Then, we need to use a convoluted map to take this torque to force and torque on the
          // nodes.
          //
          // The torque induced by the curvature is
          //  T = B (kappa - kappa_rest)
          // where B is the diagonal matrix of bending moduli and kappa_rest is the rest curvature.
          //
          // To start, we set B = I

          // Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(element);
          MUNDY_THROW_ASSERT(bulk_data.num_nodes(element) == 3, std::logic_error,
                             "SLT: Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(element) == 2, std::logic_error,
                             "SLT: Elements must have exactly 2 edges.");
          stk::mesh::Entity &node_im1 = element_nodes[0];
          stk::mesh::Entity &node_i = element_nodes[1];
          stk::mesh::Entity &node_ip1 = element_nodes[2];
          stk::mesh::Entity &edge_im1 = element_edges[0];
          stk::mesh::Entity &edge_i = element_edges[1];

          // Get the required input fields
          const auto node_i_curvature = mundy::mesh::vector3_field_data(node_curvature_field, node_i);
          const auto node_i_rest_curvature = mundy::mesh::vector3_field_data(node_rest_curvature_field, node_i);
          const double node_i_twist = stk::mesh::field_data(node_twist_field, node_i)[0];
          const double node_im1_twist = stk::mesh::field_data(node_twist_field, node_im1)[0];
          const auto node_i_rotation_gradient =
              mundy::mesh::quaternion_field_data(node_rotation_gradient_field, node_i);
          const auto edge_im1_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge_im1);
          const auto edge_i_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge_i);
          const auto edge_im1_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge_im1);
          const auto edge_i_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge_i);
          const double edge_im1_length = stk::mesh::field_data(edge_length_field, edge_im1)[0]
          const double edge_i_length = stk::mesh::field_data(edge_length_field, edge_i)[0]
          const auto edge_im1_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, edge_im1);

          // Get the output fields
          auto node_im1_force = mundy::mesh::vector3_field_data(node_force_field, node_im1);
          auto node_i_force = mundy::mesh::vector3_field_data(node_force_field, node_i);
          auto node_ip1_force = mundy::mesh::vector3_field_data(node_force_field, node_ip1);
          double *node_im1_twist_torque = stk::mesh::field_data(node_twist_torque_field, node_im1);
          double *node_i_twist_torque = stk::mesh::field_data(node_twist_torque_field, node_i);

          // Compute the torque induced by the curvature
          // To start, the torque is in the lagrangian frame.
          auto bending_modulus = mundy::math::Matrix3::identity();  // TODO(palmerb4): Replace with Kirchhoff rod model
          auto bending_torque = bending_modulus * (node_i_curvature - node_i_rest_curvature);

          // We'll use the bending torque as a placeholder for the rotated bending torque
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
          node_i_twist_torque[0] += edge_i_tangent.dot(bending_torque);
#pragma omp atomic
          node_im1_twist_torque[0] += -edge_im1_tangent.dot(bending_torque);
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
    double rest_length = sperm_rest_segment_length_;
    mundy::mesh::for_each_entity_run(
        bulk_data, edge_rank_, locally_owned_selector,
        [&node_force_field, &edge_tangent_field, &edge_length_field, &rest_length](const stk::mesh::BulkData &bulk_data,
                                                                                   const stk::mesh::Entity &edge) {
          // F_left = k (l - l_rest) tangent
          // F_right = -k (l - l_rest) tangent
          //
          // k can be computed using the material properties of the rod
          // For now, we set k = 1.

          // Get the lower rank entities
          stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(edge);
          MUNDY_THROW_ASSERT(bulk_data.num_nodes(edge) == 2, std::logic_error, "Edges must have exactly 2 nodes.");
          stk::mesh::Entity &node_im1 = element_nodes[0];
          stk::mesh::Entity &node_i = element_nodes[1];

          // Get the required input fields
          const auto edge_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge);
          const double edge_length = stk::mesh::field_data(edge_length_field, edge)[0];

          // Get the output fields
          double *node_im1_force = stk::mesh::field_data(node_force_field, node_im1);
          double *node_i_force = stk::mesh::field_data(node_force_field, node_i);

          // Compute the internal force
          const auto tmp_force = (edge_length - rest_length) * edge_tangent;
#pragma omp atomic
          node_i_force[0] += tmp_force[0];
#pragma omp atomic
          node_i_force[1] += tmp_force[1];
#pragma omp atomic
          node_i_force[2] += tmp_force[2];
#pragma omp atomic
          node_im1_force[0] -= tmp_force[0];
#pragma omp atomic
          node_im1_force[1] -= tmp_force[1];
#pragma omp atomic
          node_im1_force[2] -= tmp_force[2];
        });
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

  void compute_hertzian_contact_force() {
    debug_print("Computing the Hertzian contact force.");

    // Check if we need to rebuild the neighbor list.
    // For now, we rebuild the neighbor list every time step.
    rebuild_neighbor_list_ = true;

    // If necessary, rebuild the neighbor list.
    if (rebuild_neighbor_list_) {
      // Compute the AABBs for the rods.
      compute_aabb_ptr_->execute();

      // Delete rod-rod neighbor linkers that are too far apart.
      destroy_neighbor_linkers_ptr_->execute();

      // Generate neighbor linkers between nearby rods.
      generate_neighbor_linkers_ptr_->execute();
    }

    // Compute the signed separation distance and contact normal between neighboring rods.
    compute_ssd_and_cn_ptr_->execute();

    // Evaluate the Hertzian contact potential between neighboring rods.
    evaluate_linker_potentials_ptr_->execute();

    // Sum the linker potential force to get the induced node force on each rod.
    linker_potential_force_reduction_ptr_->execute();
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

  void compute_velocity_and_acceleration() {
    debug_print("Computing the velocity and acceleration.");
  }

  void run(int argc, char **argv) {
    debug_print("Running the simulation.");

    // Preprocess
    if (!parse_user_inputs(argc, argv)) {
      std::cerr << "Failed to parse user inputs." << std::endl;
      return;
    }
    check_input_parameters();
    dump_user_inputs();

    // Setup
    build_our_mesh_and_method_instances();
    fetch_fields_and_parts();
    setup_io();
    declare_and_initialize_the_sperm();

    // Time loop
    if (bulk_data_ptr_->parallel_rank() == 0) {
      std::cout << "Running the simulation for " << num_time_steps_ << " time steps." << std::endl;
    }

    Kokkos::Timer timer;
    for (size_t i = 0; i < num_time_steps_; i++) {
      debug_print("Time step " << i << ".");

      // Rotate the field states.
      bulk_data.update_field_data_states();

      // IO. If desired, write out the data for time t.
      if (i % io_frequency_ == 0) {
        stk_io_broker_.begin_output_step(output_file_index_, static_cast<double>(i));
        stk_io_broker_.write_defined_output_fields(output_file_index_);
        stk_io_broker_.end_output_step(output_file_index_);
        stk_io_broker_.flush_output();
      }

      // Prepare the current configuration.
      {
        // Rotate the field states.
        rotate_field_states();

        // Motion from t -> t + dt.
        // x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2.
        update_position_and_twist();

        // Zero the node velocities, accelerations, and forces/torques for time t + dt.
        zero_out_transient_node_fields();
      }

      // Evaluate forces f(x(t + dt)).
      {
        // Hertzian contact force
        compute_hertzian_contact_force();

        // Centerline twist rod model
        compute_centerline_twist_force_and_torque();
      }

      // Compute velocity and acceleration.
      {
        // Evaluate a(t + dt) = M^{-1} f(x(t + dt)).
        // Evaluate v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2.
        compute_velocity_and_acceleration();
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

  constexpr auto node_rank_ = stk::topology::NODE_RANK;
  constexpr auto edge_rank_ = stk::topology::EDGE_RANK;
  constexpr auto element_rank = stk::topology::ELEMENT_RANK;
  //@}

  //! \name Internal state
  //@{

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_;
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr_;
  stk::io::StkMeshIoBroker stk_io_broker_;
  size_t output_file_index_;
  bool rebuild_neighbor_list_ = true;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<double> *node_coordinates_field_ptr_;
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
  stk::mesh::Part *spherocylinder_segments_part_ptr_;
  //@}

  //! \name Class instances
  //@{

  std::shared_ptr<NodeEuler> node_euler_ptr_;
  std::shared_ptr<ComputeCenterlineTwistSpringConstraintForce> compute_centerline_twist_constraint_force_ptr_;
  std::shared_ptr<ComputeSignedSeparationDistanceAndContactNormal> compute_ssd_and_cn_ptr_;
  std::shared_ptr<ComputeAABB> compute_aabb_ptr_;
  std::shared_ptr<GenerateNeighborLinkers> generate_neighbor_linkers_ptr_;
  std::shared_ptr<EvaluateLinkerPotentials> evaluate_linker_potentials_ptr_;
  std::shared_ptr<LinkerPotentialForceReduction> linker_potential_force_reduction_ptr_;
  std::shared_ptr<DestroyNeighborLinkers> destroy_neighbor_linkers_ptr_;
  std::shared_ptr<DeclareAndInitConstraints> declare_and_init_constraints_ptr_;
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

  size_t num_sperm_ = 1;
  size_t num_nodes_per_sperm_ = 10;
  double sperm_radius_ = 0.6;
  double sperm_initial_segment_length_ = 1.0;
  double sperm_rest_segment_length_ = 2 * sperm_radius_;
  double sperm_rest_curvature_twist_ = 0.0;
  double sperm_rest_curvature_bend1_ = 0.0;
  double sperm_rest_curvature_bend2_ = 0.0;

  double sperm_youngs_modulus_ = 1000.0;
  double sperm_poissons_ratio_ = 0.3;
  double sperm_density_ = 1.0;

  size_t num_time_steps_ = 100;
  double timestep_size_ = 0.01;
  size_t io_frequency_ = 10;
  double skin_distance_ = sperm_radius_;
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
