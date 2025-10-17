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

The goal of this example is to simulate the swimming motion of a multiple, colliding long sperm.
*/

// External libs
#include <openrand/philox.h>

// Boost
// #include <boost/math/tools/roots.hpp>

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// Teuchos
#include <Teuchos_CommandLineProcessor.hpp>      // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>             // for Teuchos::ParameterList
#include <Teuchos_YamlParameterListHelpers.hpp>  // for Teuchos::getParametersFromYamlFile

// STK Balance
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings

// STK IO
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::StkMeshIoBroker

// STK Mesh
#include <stk_mesh/base/Comm.hpp>           // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>   // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/FEMHelpers.hpp>     // for stk::mesh::declare_element, stk::mesh::declare_element_edge
#include <stk_mesh/base/Field.hpp>          // for stk::mesh::Field, stk::mesh::field_data
#include <stk_mesh/base/FieldParallel.hpp>  // for stk::mesh::parallel_sum
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpFieldParallel.hpp>  // for stk::mesh::parallel_sum
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// STK Topology
#include <stk_topology/topology.hpp>  // for stk::topology

// STK Util
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// STK Search
#include <stk_search/BoxIdent.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/Point.hpp>
#include <stk_search/SearchMethod.hpp>
#include <stk_search/Sphere.hpp>

// Mundy libs
#include <mundy_core/MakeStringArray.hpp>  // for mundy::core::make_string_array
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_geom/periodicity.hpp>
#include <mundy_geom/primitives.hpp>
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>         // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>       // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>        // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>  // for mundy::...::SpherocylinderSegmentSpherocylinderSegmentLinkers
#include <mundy_math/Matrix3.hpp>     // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion, mundy::math::quat_from_parallel_transport
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_math/distance/SegmentSegment.hpp>  // for mundy::math::distance::distance_sq_from_point_to_line_segment
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>    // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>  // for mundy::mesh::field_fill
#include <mundy_mesh/utils/FillFieldWithValue.hpp>  // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/FieldReqs.hpp>                 // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>                  // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>                  // for mundy::meta::PartReqs
#include <mundy_shapes/ComputeAABB.hpp>             // for mundy::shapes::ComputeAABB

namespace mundy {

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
///         // Hertzian contact
///         {
///             // Neighbor detection rod-rod
///             - Check if the rod-rod neighbor list needs updated or not
///                 - Compute the AABBs for the rods
///                  (Using mundy's ComputeAABB function)
///
///                 - Delete rod-rod neighbor linkers that are too far apart
///                  (Using the DestroyDistantNeighbors technique of mundy's DestroyNeighborLinkers function)
///
///                 - Generate neighbor linkers between nearby rods
///                  (Using the GenerateNeighborLinkers function of mundy's GenerateNeighborLinkers function)
///
///             // Hertzian contact force evaluation
///             - Compute the signed separation distance and contact normal between neighboring rods
///              (Using mundy's ComputeSignedSeparationDistanceAndContactNormal function)
///
///             - Evaluate the Hertzian contact potential between neighboring rods
///              (Using mundy's EvaluateLinkerPotentials function)
///
///             - Sum the linker potential force to get the induced node force on each rod
///              (Using mundy's LinkerPotentialForceReduction function)
///         }
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
using DoubleField = stk::mesh::Field<double>;
using IntField = stk::mesh::Field<int>;
using NgpDoubleField = stk::mesh::NgpField<double>;
using NgpIntField = stk::mesh::NgpField<int>;

KOKKOS_INLINE_FUNCTION
constexpr bool fma_equal(stk::mesh::FastMeshIndex lhs, stk::mesh::FastMeshIndex rhs) {
  return (lhs.bucket_id == rhs.bucket_id) && (lhs.bucket_ord == rhs.bucket_ord);
}

KOKKOS_INLINE_FUNCTION
constexpr bool fma_less(stk::mesh::FastMeshIndex lhs, stk::mesh::FastMeshIndex rhs) {
  return lhs.bucket_id == rhs.bucket_id ? lhs.bucket_ord < rhs.bucket_ord : lhs.bucket_id < rhs.bucket_id;
}

KOKKOS_INLINE_FUNCTION
constexpr bool fma_greater(stk::mesh::FastMeshIndex lhs, stk::mesh::FastMeshIndex rhs) {
  return lhs.bucket_id == rhs.bucket_id ? lhs.bucket_ord > rhs.bucket_ord : lhs.bucket_id < rhs.bucket_id;
}

inline void print_rank0(auto think_to_print, int indent_level = 0) {
  if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
    std::string indent(indent_level * 2, ' ');
    std::cout << indent << think_to_print << std::endl;
  }
}

inline void debug_print([[maybe_unused]] auto thing_to_print, [[maybe_unused]] int indent_level = 0) {
#ifdef DEBUG
  print_rank0(thing_to_print, indent_level);
#endif
}

struct RunConfig {
  void parse_user_inputs(int argc, char **argv) {
    debug_print("Parsing user inputs.");

    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, false);

    // If we should accept the parameters directly from the command line or from a file
    bool use_input_file = false;
    cmdp.setOption("use_input_file", "no_use_input_file", &use_input_file, "Use an input file.");
    bool use_input_file_found = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    MUNDY_THROW_ASSERT(use_input_file_found, std::invalid_argument, "Failed to parse the command line arguments.");

    // Switch to requiring that all options must be recognized.
    cmdp.recogniseAllOptions(true);

    if (!use_input_file) {
      // Parse the command line options.

      //   Sperm initialization:
      cmdp.setOption("num_sperm", &num_sperm, "Number of sperm.");
      cmdp.setOption("num_nodes_per_sperm", &num_nodes_per_sperm, "Number of nodes per sperm.");
      cmdp.setOption("sperm_radius", &sperm_radius, "The radius of each sperm.");
      cmdp.setOption("sperm_initial_segment_length", &sperm_initial_segment_length, "Initial sperm segment length.");
      cmdp.setOption("sperm_rest_segment_length", &sperm_rest_segment_length, "Rest sperm segment length.");
      cmdp.setOption("sperm_rest_curvature_twist", &sperm_rest_curvature_twist, "Rest curvature (twist) of the sperm.");
      cmdp.setOption("sperm_rest_curvature_bend1", &sperm_rest_curvature_bend1,
                     "Rest curvature (bend along the first coordinate direction) of the sperm.");
      cmdp.setOption("sperm_rest_curvature_bend2", &sperm_rest_curvature_bend2,
                     "Rest curvature (bend along the second coordinate direction) of the sperm.");

      cmdp.setOption("sperm_density", &sperm_density, "Density of the sperm.");
      cmdp.setOption("sperm_youngs_modulus", &sperm_youngs_modulus, "Young's modulus of the sperm.");
      cmdp.setOption("sperm_poissons_ratio", &sperm_poissons_ratio, "Poisson's ratio of the sperm.");

      //   The simulation:
      cmdp.setOption("num_time_steps", &num_time_steps, "Number of time steps.");
      cmdp.setOption("timestep_size", &timestep_size, "Time step size.");
      cmdp.setOption("io_frequency", &io_frequency, "Number of timesteps between writing output.");

      bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
      MUNDY_THROW_ASSERT(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");
    } else {
      cmdp.setOption("input_file", &input_file_name, "The name of the input file.");
      bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
      MUNDY_THROW_ASSERT(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");

      // Read in the parameters from the parameter list.
      Teuchos::ParameterList param_list = *Teuchos::getParametersFromYamlFile(input_file_name);

      num_sperm = param_list.get<int>("num_sperm");
      num_nodes_per_sperm = param_list.get<int>("num_nodes_per_sperm");
      sperm_radius = param_list.get<double>("sperm_radius");
      sperm_initial_segment_length = param_list.get<double>("sperm_initial_segment_length");
      sperm_rest_segment_length = param_list.get<double>("sperm_rest_segment_length");
      sperm_rest_curvature_twist = param_list.get<double>("sperm_rest_curvature_twist");
      sperm_rest_curvature_bend1 = param_list.get<double>("sperm_rest_curvature_bend1");
      sperm_rest_curvature_bend2 = param_list.get<double>("sperm_rest_curvature_bend2");

      sperm_density = param_list.get<double>("sperm_density");
      sperm_youngs_modulus = param_list.get<double>("sperm_youngs_modulus");
      sperm_poissons_ratio = param_list.get<double>("sperm_poissons_ratio");

      num_time_steps = param_list.get<int>("num_time_steps");
      timestep_size = param_list.get<double>("timestep_size");
    }

    check_input_parameters();
  }

  void check_input_parameters() {
    debug_print("Checking input parameters.");
    MUNDY_THROW_ASSERT(num_sperm > 0, std::invalid_argument, "num_sperm must be greater than 0.");
    MUNDY_THROW_ASSERT(num_nodes_per_sperm > 0, std::invalid_argument, "num_nodes_per_sperm must be greater than 0.");
    MUNDY_THROW_ASSERT(sperm_radius > 0, std::invalid_argument, "sperm_radius must be greater than 0.");
    MUNDY_THROW_ASSERT(sperm_initial_segment_length > -1e-12, std::invalid_argument,
                       "sperm_initial_segment_length must be greater than or equal to 0.");
    MUNDY_THROW_ASSERT(sperm_rest_segment_length > -1e-12, std::invalid_argument,
                       "sperm_rest_segment_length must be greater than or equal to 0.");
    MUNDY_THROW_ASSERT(sperm_youngs_modulus > 0, std::invalid_argument, "sperm_youngs_modulus must be greater than 0.");
    MUNDY_THROW_ASSERT(sperm_poissons_ratio > 0, std::invalid_argument, "sperm_poissons_ratio must be greater than 0.");

    MUNDY_THROW_ASSERT(num_time_steps > 0, std::invalid_argument, "num_time_steps must be greater than 0.");
    MUNDY_THROW_ASSERT(timestep_size > 0, std::invalid_argument, "timestep_size must be greater than 0.");
    MUNDY_THROW_ASSERT(io_frequency > 0, std::invalid_argument, "io_frequency must be greater than 0.");
  }

  void print() {
    debug_print("Dumping user inputs.");
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;
      std::cout << "  num_sperm: " << num_sperm << std::endl;
      std::cout << "  num_nodes_per_sperm: " << num_nodes_per_sperm << std::endl;
      std::cout << "  sperm_radius: " << sperm_radius << std::endl;
      std::cout << "  sperm_initial_segment_length: " << sperm_initial_segment_length << std::endl;
      std::cout << "  sperm_rest_segment_length: " << sperm_rest_segment_length << std::endl;
      std::cout << "  spatial_wavelength: " << spatial_wavelength << std::endl;
      std::cout << "  temporal_wavelength: " << temporal_wavelength << std::endl;
      std::cout << "  sperm_youngs_modulus: " << sperm_youngs_modulus << std::endl;
      std::cout << "  sperm_poissons_ratio: " << sperm_poissons_ratio << std::endl;
      std::cout << "  sperm_density: " << sperm_density << std::endl;
      std::cout << "  num_time_steps: " << num_time_steps << std::endl;
      std::cout << "  timestep_size: " << timestep_size << std::endl;
      std::cout << "  io_frequency: " << io_frequency << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  //! \name User parameters
  //@{
  std::string input_file_name = "input.yaml";

  size_t num_sperm = 20;
  size_t num_nodes_per_sperm = 301;
  double sperm_radius = 0.5;
  double sperm_initial_segment_length = 2.0 * sperm_radius;
  double sperm_rest_segment_length = 2.0 * sperm_radius;
  double sperm_rest_curvature_twist = 0.0;
  double sperm_rest_curvature_bend1 = 0.0;
  double sperm_rest_curvature_bend2 = 0.0;

  double sperm_youngs_modulus = 1000000.00;
  double sperm_relaxed_youngs_modulus = sperm_youngs_modulus;
  double sperm_normal_youngs_modulus = sperm_youngs_modulus;
  double sperm_poissons_ratio = 0.3;
  double sperm_density = 1.0;

  double amplitude = 0.1;
  double spatial_wavelength = (num_nodes_per_sperm - 1) * sperm_initial_segment_length / 5.0;
  double temporal_wavelength = 2 * M_PI;  // Units: seconds per oscillations
  // double temporal_wavelength = std::numeric_limits<double>::infinity();  // Units: seconds per oscillations
  double viscosity = 1;

  double timestep_size = 1e-6;
  size_t num_time_steps = 200000000;
  size_t io_frequency = 100000;
  double search_buffer = 2 * sperm_radius;
  double domain_width =
      2 * num_sperm * sperm_radius / 0.8;  // One diameter separation between sperm == 50% area fraction
  double domain_height = (num_nodes_per_sperm - 1) * sperm_initial_segment_length + 11.0;
  //@}
};

std::vector<bool> interleaved_vector(int N, int i) {
  // Ensure N is divisible by 2 and i is within range
  if (N % 2 != 0) {
    throw std::invalid_argument("N must be divisible by 2");
  }
  std::vector<bool> result(N, 0);  // Initialize result vector with 0s

  for (int n = 0; n < N; ++n) {
    if (n < N / 2 - i) {
      result[n] = 0;  // First region: all 0s
    } else if (n >= N / 2 + i) {
      result[n] = 1;  // Last region: all 1s
    } else {
      // Alternating region: starts with 1
      result[n] = ((n - (N / 2 - i)) % 2 == 0) ? 1 : 0;
    }
  }

  return result;
}

void declare_and_initialize_sperm(stk::mesh::BulkData &bulk_data, stk::mesh::Part &centerline_twist_springs_part,
                                  stk::mesh::Part &boundary_sperm_part, stk::mesh::Part &spherocylinder_segments_part,
                                  const size_t &num_sperm, const size_t &num_nodes_per_sperm,
                                  const double &sperm_radius, const double &segment_length,
                                  const double &rest_segment_length, const DoubleField &node_coords_field,
                                  const DoubleField &node_velocity_field, const DoubleField &node_force_field,
                                  const DoubleField &node_twist_field, const DoubleField &node_twist_velocity_field,
                                  const DoubleField &node_twist_torque_field, const DoubleField &node_archlength_field,
                                  const DoubleField &node_curvature_field, const DoubleField &node_rest_curvature_field,
                                  const DoubleField &node_radius_field, const IntField &node_sperm_id_field,
                                  const DoubleField &edge_orientation_field, const DoubleField &edge_tangent_field,
                                  const DoubleField &edge_length_field, const DoubleField &elem_radius_field,
                                  const DoubleField &elem_rest_length_field) {
  debug_print("Declaring and initializing the sperm.");

  stk::mesh::MetaData &meta_data = bulk_data.mesh_meta_data();

  // Declare N sperm side-by side.
  // Each sperm points up or down the z-axis. Half will point up and half down. We will control which ones point up vs
  // down by varying the amount of interleaving between the sperm.
  //
  // i=0: ^^^^^vvvvv
  // i=1: ^^^^v^vvvv
  // i=2: ^^^v^v^vvv
  // i=3: ^^v^v^v^vv
  // i=4: ^v^v^v^v^v
  int degree_of_interleaving = 4;
  std::cout << "degree_of_interleaving: " << degree_of_interleaving << std::endl;
  std::vector<bool> sperm_directions = interleaved_vector(num_sperm, degree_of_interleaving);

  for (size_t j = 0; j < num_sperm; j++) {
    // To make our lives easier, we align the sperm with the z-axis, as this makes our edge orientation a unit
    // quaternion.
    // const bool is_boundary_sperm = (j == 0) || (j == num_sperm_ - 1);
    // const double segment_length =
    //     is_boundary_sperm ? 3 * sperm_initial_segment_length_ : sperm_initial_segment_length_;
    const bool is_boundary_sperm = false;

    // TODO(palmerb4): Notice that we are shifting the sperm to be separated by a diameter.
    bool flip_sperm = sperm_directions[j];
    // const bool flip_sperm = false;
    // math::Vector3d tail_coord(0.0, 2.0 * j * (2.0 * sperm_radius),
    //                                         (flip_sperm ? segment_length * (num_nodes_per_sperm - 1) : 0.0) -
    //                                             (is_boundary_sperm ? segment_length * (num_nodes_per_sperm - 1) :
    //                                             0.0));
    double random_shift = static_cast<double>(rand()) / RAND_MAX * ((segment_length) * (num_nodes_per_sperm - 1) + 10);

    math::Vector3d tail_coord(
        0.0, j * (2.0 * sperm_radius) / 0.8,
        (flip_sperm ? (segment_length * (num_nodes_per_sperm - 1) + random_shift) : random_shift));

    math::Vector3d sperm_axis(0.0, 0.0, flip_sperm ? -1.0 : 1.0);

    // Because we are creating multiple sperm, we need to determine the node and element index ranges for each sperm.
    size_t start_node_id = num_nodes_per_sperm * j + 1u;
    size_t start_edge_id = (num_nodes_per_sperm - 1) * j + 1u;
    size_t start_centerline_twist_spring_id = (num_nodes_per_sperm - 2) * j + 1u;
    size_t start_spherocylinder_segment_spring_id =
        (num_nodes_per_sperm - 1) * j + (num_nodes_per_sperm - 2) * num_sperm + 1u;

    auto get_node_id = [start_node_id](const size_t &seq_node_index) { return start_node_id + seq_node_index; };

    auto get_node = [get_node_id, &bulk_data](const size_t &seq_node_index) {
      return bulk_data.get_entity(stk::topology::NODE_RANK, get_node_id(seq_node_index));
    };

    auto get_edge_id = [start_edge_id](const size_t &seq_node_index) { return start_edge_id + seq_node_index; };

    auto get_edge = [get_edge_id, &bulk_data](const size_t &seq_node_index) {
      return bulk_data.get_entity(stk::topology::EDGE_RANK, get_edge_id(seq_node_index));
    };

    auto get_centerline_twist_spring_id = [start_centerline_twist_spring_id](const size_t &seq_node_index) {
      return start_centerline_twist_spring_id + seq_node_index;
    };

    auto get_centerline_twist_spring = [get_centerline_twist_spring_id, &bulk_data](const size_t &seq_node_index) {
      return bulk_data.get_entity(stk::topology::ELEM_RANK, get_centerline_twist_spring_id(seq_node_index));
    };

    auto get_spherocylinder_segment_id = [&start_spherocylinder_segment_spring_id](const size_t &seq_node_index) {
      return start_spherocylinder_segment_spring_id + seq_node_index;
    };

    auto get_spherocylinder_segment = [get_spherocylinder_segment_id, &bulk_data](const size_t &seq_node_index) {
      return bulk_data.get_entity(stk::topology::ELEM_RANK, get_spherocylinder_segment_id(seq_node_index));
    };

    // Create the springs and their connected nodes, distributing the work across the ranks.
    const size_t rank = bulk_data.parallel_rank();
    const size_t nodes_per_rank = num_nodes_per_sperm / bulk_data.parallel_size();
    const size_t remainder = num_nodes_per_sperm % bulk_data.parallel_size();
    const size_t start_seq_node_index = rank * nodes_per_rank + std::min(rank, remainder);
    const size_t end_seq_node_index = start_seq_node_index + nodes_per_rank + (rank < remainder ? 1 : 0);

    bulk_data.modification_begin();

    // Temporary/scatch variables
    stk::mesh::Permutation invalid_perm = stk::mesh::Permutation::INVALID_PERMUTATION;
    stk::mesh::OrdinalVector scratch1, scratch2, scratch3;
    stk::topology spring_topo = stk::topology::SHELL_TRI_3;
    stk::topology spherocylinder_topo = stk::topology::BEAM_2;
    stk::topology edge_topo = stk::topology::LINE_2;
    auto spring_part = is_boundary_sperm ? stk::mesh::PartVector{&centerline_twist_springs_part, &boundary_sperm_part}
                                         : stk::mesh::PartVector{&centerline_twist_springs_part};
    auto spherocylinder_part = is_boundary_sperm
                                   ? stk::mesh::PartVector{&spherocylinder_segments_part, &boundary_sperm_part}
                                   : stk::mesh::PartVector{&spherocylinder_segments_part};
    auto spring_and_edge_part =
        is_boundary_sperm
            ? stk::mesh::PartVector{&centerline_twist_springs_part, &meta_data.get_topology_root_part(edge_topo),
                                    &boundary_sperm_part}
            : stk::mesh::PartVector{&centerline_twist_springs_part, &meta_data.get_topology_root_part(edge_topo)};

    // Centerline twist springs connect nodes i, i+1, and i+2. We need to start at node i=0 and end at node N - 2.
    const size_t start_elem_chain_index = (rank == 0) ? start_seq_node_index : start_seq_node_index - 1;
    const size_t end_start_elem_chain_index =
        (rank == bulk_data.parallel_size() - 1) ? end_seq_node_index - 2 : end_seq_node_index - 1;
    for (size_t i = start_elem_chain_index; i < end_start_elem_chain_index; ++i) {
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

      stk::mesh::Entity left_node = bulk_data.get_entity(stk::topology::NODE_RANK, left_node_id);
      stk::mesh::Entity center_node = bulk_data.get_entity(stk::topology::NODE_RANK, center_node_id);
      stk::mesh::Entity right_node = bulk_data.get_entity(stk::topology::NODE_RANK, right_node_id);
      if (!bulk_data.is_valid(left_node)) {
        left_node = bulk_data.declare_node(left_node_id);
      }
      if (!bulk_data.is_valid(center_node)) {
        center_node = bulk_data.declare_node(center_node_id);
      }
      if (!bulk_data.is_valid(right_node)) {
        right_node = bulk_data.declare_node(right_node_id);
      }

      // Fetch the edges
      stk::mesh::EntityId left_edge_id = get_edge_id(i);
      stk::mesh::EntityId right_edge_id = get_edge_id(i + 1);
      stk::mesh::Entity left_edge = bulk_data.get_entity(stk::topology::EDGE_RANK, left_edge_id);
      stk::mesh::Entity right_edge = bulk_data.get_entity(stk::topology::EDGE_RANK, right_edge_id);
      if (!bulk_data.is_valid(left_edge)) {
        // Declare the edge and connect it to the nodes
        left_edge = bulk_data.declare_edge(left_edge_id, spring_and_edge_part);
        bulk_data.declare_relation(left_edge, left_node, 0, invalid_perm, scratch1, scratch2, scratch3);
        bulk_data.declare_relation(left_edge, center_node, 1, invalid_perm, scratch1, scratch2, scratch3);
      }
      if (!bulk_data.is_valid(right_edge)) {
        // Declare the edge and connect it to the nodes
        right_edge = bulk_data.declare_edge(right_edge_id, spring_and_edge_part);
        bulk_data.declare_relation(right_edge, center_node, 0, invalid_perm, scratch1, scratch2, scratch3);
        bulk_data.declare_relation(right_edge, right_node, 1, invalid_perm, scratch1, scratch2, scratch3);
      }

      // Fetch the centerline twist spring
      stk::mesh::EntityId spring_id = get_centerline_twist_spring_id(i);
      stk::mesh::Entity spring = bulk_data.declare_element(spring_id, spring_part);

      // Connect the spring to the edges
      stk::mesh::Entity spring_nodes[3] = {left_node, center_node, right_node};
      stk::mesh::Entity left_edge_nodes[2] = {left_node, center_node};
      stk::mesh::Entity right_edge_nodes[2] = {center_node, right_node};
      stk::mesh::Permutation left_spring_perm =
          bulk_data.find_permutation(spring_topo, spring_nodes, edge_topo, left_edge_nodes, 0);
      stk::mesh::Permutation right_spring_perm =
          bulk_data.find_permutation(spring_topo, spring_nodes, edge_topo, right_edge_nodes, 1);
      bulk_data.declare_relation(spring, left_edge, 0, left_spring_perm, scratch1, scratch2, scratch3);
      bulk_data.declare_relation(spring, right_edge, 1, right_spring_perm, scratch1, scratch2, scratch3);

      // Connect the spring to the nodes
      bulk_data.declare_relation(spring, left_node, 0, invalid_perm, scratch1, scratch2, scratch3);
      bulk_data.declare_relation(spring, center_node, 1, invalid_perm, scratch1, scratch2, scratch3);
      bulk_data.declare_relation(spring, right_node, 2, invalid_perm, scratch1, scratch2, scratch3);
      MUNDY_THROW_ASSERT(bulk_data.bucket(spring).topology() != stk::topology::INVALID_TOPOLOGY, std::logic_error,
                         "A centerline twist spring has an invalid topology.");

      // Fetch the sphero-cylinder segments
      stk::mesh::EntityId left_spherocylinder_segment_id = get_spherocylinder_segment_id(i);
      stk::mesh::EntityId right_spherocylinder_segment_id = get_spherocylinder_segment_id(i + 1);
      stk::mesh::Entity left_spherocylinder_segment =
          bulk_data.get_entity(stk::topology::ELEM_RANK, left_spherocylinder_segment_id);
      stk::mesh::Entity right_spherocylinder_segment =
          bulk_data.get_entity(stk::topology::ELEM_RANK, right_spherocylinder_segment_id);
      if (!bulk_data.is_valid(left_spherocylinder_segment)) {
        // Declare the spherocylinder segment and connect it to the nodes
        left_spherocylinder_segment = bulk_data.declare_element(left_spherocylinder_segment_id, spherocylinder_part);
        bulk_data.declare_relation(left_spherocylinder_segment, left_node, 0, invalid_perm, scratch1, scratch2,
                                   scratch3);
        bulk_data.declare_relation(left_spherocylinder_segment, center_node, 1, invalid_perm, scratch1, scratch2,
                                   scratch3);
      }
      if (!bulk_data.is_valid(right_spherocylinder_segment)) {
        // Declare the spherocylinder segment and connect it to the nodes
        right_spherocylinder_segment = bulk_data.declare_element(right_spherocylinder_segment_id, spherocylinder_part);
        bulk_data.declare_relation(right_spherocylinder_segment, center_node, 0, invalid_perm, scratch1, scratch2,
                                   scratch3);
        bulk_data.declare_relation(right_spherocylinder_segment, right_node, 1, invalid_perm, scratch1, scratch2,
                                   scratch3);
      }

      // Connect the segments to the edges
      stk::mesh::Entity left_spherocylinder_segment_nodes[2] = {left_node, center_node};
      stk::mesh::Entity right_spherocylinder_segment_nodes[2] = {center_node, right_node};
      stk::mesh::Permutation left_spherocylinder_perm = bulk_data.find_permutation(
          spherocylinder_topo, left_spherocylinder_segment_nodes, edge_topo, left_edge_nodes, 0);
      stk::mesh::Permutation right_spherocylinder_perm = bulk_data.find_permutation(
          spherocylinder_topo, right_spherocylinder_segment_nodes, edge_topo, right_edge_nodes, 1);
      bulk_data.declare_relation(left_spherocylinder_segment, left_edge, 0, left_spherocylinder_perm, scratch1,
                                 scratch2, scratch3);
      bulk_data.declare_relation(right_spherocylinder_segment, right_edge, 0, right_spherocylinder_perm, scratch1,
                                 scratch2, scratch3);

      // Connect the segments to the nodes
      bulk_data.declare_relation(left_spherocylinder_segment, left_node, 0, invalid_perm, scratch1, scratch2, scratch3);
      bulk_data.declare_relation(left_spherocylinder_segment, center_node, 1, invalid_perm, scratch1, scratch2,
                                 scratch3);
      bulk_data.declare_relation(right_spherocylinder_segment, center_node, 0, invalid_perm, scratch1, scratch2,
                                 scratch3);
      bulk_data.declare_relation(right_spherocylinder_segment, right_node, 1, invalid_perm, scratch1, scratch2,
                                 scratch3);

      // Populate the spring's data
      stk::mesh::field_data(elem_radius_field, spring)[0] = sperm_radius;
      stk::mesh::field_data(elem_rest_length_field, spring)[0] = rest_segment_length;

      // Populate the spherocylinder segment's data
      stk::mesh::field_data(elem_radius_field, left_spherocylinder_segment)[0] = sperm_radius;
      stk::mesh::field_data(elem_radius_field, right_spherocylinder_segment)[0] = sperm_radius;
    }

    // Share the nodes with the neighboring ranks. At this point, these nodes should all exist.
    //
    // Note, node sharing is symmetric. If we don't own the node that we intend to share, we need to declare it before
    // marking it as shared. If we are rank 0, we share our final node with rank 1 and receive their first node. If we
    // are rank N, we share our first node with rank N - 1 and receive their final node. Otherwise, we share our first
    // and last nodes with the corresponding neighboring ranks and receive their corresponding nodes.
    if (bulk_data.parallel_size() > 1) {
      debug_print("Sharing nodes with neighboring ranks.");
      if (rank == 0) {
        // Share the last node with rank 1.
        stk::mesh::Entity node = get_node(end_seq_node_index - 1);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(node), std::logic_error,
                           "A node is invalid. Ghosting may not be correct.");
        bulk_data.add_node_sharing(node, rank + 1);

        // Receive the first node from rank 1
        stk::mesh::Entity received_node = get_node(end_seq_node_index);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(received_node), std::logic_error,
                           "A node is invalid. Ghosting may not be correct.");
        bulk_data.add_node_sharing(received_node, rank + 1);
      } else if (rank == bulk_data.parallel_size() - 1) {
        // Share the first node with rank N - 1.
        stk::mesh::Entity node = get_node(start_seq_node_index);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(node), std::logic_error,
                           "A node is invalid. Ghosting may not be correct.");
        bulk_data.add_node_sharing(node, rank - 1);

        // Receive the last node from rank N - 1.
        stk::mesh::Entity received_node = get_node(start_seq_node_index - 1);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(received_node), std::logic_error,
                           "A node is invalid. Ghosting may not be correct.");
        bulk_data.add_node_sharing(received_node, rank - 1);
      } else {
        // Share the first and last nodes with the corresponding neighboring ranks.
        stk::mesh::Entity first_node = get_node(start_seq_node_index);
        stk::mesh::Entity last_node = get_node(end_seq_node_index - 1);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(first_node), std::logic_error,
                           "A node is invalid. Ghosting may not be correct.");
        MUNDY_THROW_ASSERT(bulk_data.is_valid(last_node), std::logic_error,
                           "A node is invalid. Ghosting may not be correct.");
        bulk_data.add_node_sharing(first_node, rank - 1);
        bulk_data.add_node_sharing(last_node, rank + 1);

        // Receive the corresponding nodes from the neighboring ranks.
        stk::mesh::Entity received_first_node = get_node(start_seq_node_index - 1);
        stk::mesh::Entity received_last_node = get_node(end_seq_node_index);
        bulk_data.add_node_sharing(received_first_node, rank - 1);
        bulk_data.add_node_sharing(received_last_node, rank + 1);
      }
    }

    std::cerr << "Edge sharing is currently not implemented" << std::endl;

    bulk_data.modification_end();

    // Set the node data for all nodes (even the shared ones)
    for (size_t i = start_seq_node_index - 1 * (rank > 0);
         i < end_seq_node_index + 1 * (rank < bulk_data.parallel_size() - 1); ++i) {
      stk::mesh::Entity node = get_node(i);
      MUNDY_THROW_ASSERT(bulk_data.is_valid(node), std::logic_error, "A node is invalid. Ghosting may not be correct.");
      MUNDY_THROW_ASSERT(bulk_data.bucket(node).member(centerline_twist_springs_part), std::logic_error,
                         "The node must be a member of the centerline twist part.");

      mesh::vector3_field_data(node_coords_field, node) =
          tail_coord + sperm_axis * static_cast<double>(i) * segment_length;
      mesh::vector3_field_data(node_velocity_field, node).set(0.0, 0.0, 0.0);
      mesh::vector3_field_data(node_force_field, node).set(0.0, 0.0, 0.0);
      stk::mesh::field_data(node_twist_field, node)[0] = 0.0;
      stk::mesh::field_data(node_twist_velocity_field, node)[0] = 0.0;
      stk::mesh::field_data(node_twist_torque_field, node)[0] = 0.0;
      mesh::vector3_field_data(node_curvature_field, node).set(0.0, 0.0, 0.0);
      mesh::vector3_field_data(node_rest_curvature_field, node).set(0.0, 0.0, 0.0);
      stk::mesh::field_data(node_radius_field, node)[0] = sperm_radius;
      stk::mesh::field_data(node_archlength_field, node)[0] = i * segment_length;
      stk::mesh::field_data(node_sperm_id_field, node)[0] = j;
    }

    // Populate the edge data
    mesh::for_each_entity_run(
        bulk_data, stk::topology::EDGE_RANK, meta_data.locally_owned_part(),
        [&node_coords_field, &edge_orientation_field, &edge_tangent_field, &edge_length_field, &flip_sperm](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &edge) {
          // We are currently in the reference configuration, so the orientation must map from Cartesian to reference
          // lab frame.
          const stk::mesh::Entity *edge_nodes = bulk_data.begin_nodes(edge);
          const auto edge_node0_coords = mesh::vector3_field_data(node_coords_field, edge_nodes[0]);
          const auto edge_node1_coords = mesh::vector3_field_data(node_coords_field, edge_nodes[1]);
          math::Vector3d edge_tangent = edge_node1_coords - edge_node0_coords;
          const double edge_length = math::norm(edge_tangent);
          edge_tangent /= edge_length;
          // Using the triad to generate the orientation
          auto d1 = math::Vector3d(flip_sperm ? -1.0 : 1.0, 0.0, 0.0);
          math::Vector3d d3 = edge_tangent;
          math::Vector3d d2 = math::cross(d3, d1);
          d2 /= math::norm(d2);
          MUNDY_THROW_ASSERT(math::dot(d3, math::cross(d1, d2)) > 0.0, std::logic_error,
                             "The triad is not right-handed.");
          math::Matrix3d D;
          D.set_column(0, d1);
          D.set_column(1, d2);
          D.set_column(2, d3);
          mesh::quaternion_field_data(edge_orientation_field, edge) = math::rotation_matrix_to_quaternion(D);
          mesh::vector3_field_data(edge_tangent_field, edge) = edge_tangent;
          stk::mesh::field_data(edge_length_field, edge)[0] = edge_length;
        });
  }

  // Mark the fields modified on the host
  node_coords_field.modify_on_host();
  node_velocity_field.modify_on_host();
  node_force_field.modify_on_host();
  node_twist_field.modify_on_host();
  node_twist_velocity_field.modify_on_host();
  node_twist_torque_field.modify_on_host();
  node_archlength_field.modify_on_host();
  node_curvature_field.modify_on_host();
  node_rest_curvature_field.modify_on_host();
  node_radius_field.modify_on_host();
  node_sperm_id_field.modify_on_host();
  edge_orientation_field.modify_on_host();
  edge_tangent_field.modify_on_host();
  edge_length_field.modify_on_host();
  elem_radius_field.modify_on_host();
  elem_rest_length_field.modify_on_host();
}

void validate_node_radius(stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &selector,
                          NgpDoubleField &node_radius_field) {
  debug_print("Validating the node radius.");

  // Assert that the radius is positive
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        const double radius = node_radius_field(node_index, 0);
        MUNDY_THROW_ASSERT(radius > 0.0, std::invalid_argument, "The radius must be positive.");
      });
}

void propagate_rest_curvature(stk::mesh::NgpMesh &ngp_mesh, const double &current_time, const double &amplitude,
                              const double &spatial_wavelength, const double &temporal_wavelength,
                              const stk::mesh::Part &centerline_twist_springs_part,
                              NgpDoubleField &node_archlength_field, NgpIntField &node_sperm_id_field,
                              NgpDoubleField &node_rest_curvature_field) {
  debug_print("Propogating the rest curvature.");
  node_archlength_field.sync_to_device();
  node_sperm_id_field.sync_to_device();

  const double spatial_frequency = 2.0 * M_PI / spatial_wavelength;
  const double temporal_frequency = 2.0 * M_PI / temporal_wavelength;

  // Propagate the rest curvature of the nodes according to
  // kappa_rest = amplitude * sin(spatial_frequency * archlength + temporal_frequency * time).
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, centerline_twist_springs_part,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        const double node_archlength = node_archlength_field(node_index, 0);
        const int node_sperm_id = node_sperm_id_field(node_index, 0);

        // Propagate the rest curvature
        // To avoid synchronized states, we add a random number to the phase of the sine wave for each sperm.
        // The same RNG is used for all time.
        openrand::Philox rng(node_sperm_id, 0);
        const double phase = 2.0 * M_PI * rng.rand<double>();
        node_rest_curvature_field(node_index, 0) =
            amplitude * Kokkos::sin(spatial_frequency * node_archlength + temporal_frequency * current_time + phase);
        node_rest_curvature_field(node_index, 1) = 0.0;
        node_rest_curvature_field(node_index, 2) = 0.0;
      });

  node_rest_curvature_field.modify_on_device();
}

void compute_edge_information(stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Part &centerline_twist_springs_part,
                              NgpDoubleField &node_coords_field, NgpDoubleField &node_twist_field,
                              NgpDoubleField &edge_orientation_field, NgpDoubleField &old_edge_orientation_field,
                              NgpDoubleField &edge_tangent_field, NgpDoubleField &old_edge_tangent_field,
                              NgpDoubleField &edge_binormal_field, NgpDoubleField &edge_length_field) {
  debug_print("Computing the edge information.");
  node_coords_field.sync_to_device();
  node_twist_field.sync_to_device();
  edge_orientation_field.sync_to_device();
  old_edge_orientation_field.sync_to_device();
  edge_tangent_field.sync_to_device();
  old_edge_tangent_field.sync_to_device();
  edge_binormal_field.sync_to_device();
  edge_length_field.sync_to_device();

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
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::EDGE_RANK, centerline_twist_springs_part,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &edge_index) {
        // Get the nodes of the edge
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::EDGE_RANK, edge_index);
        const stk::mesh::Entity node_i = nodes[0];
        const stk::mesh::Entity node_ip1 = nodes[1];
        const stk::mesh::FastMeshIndex node_i_index = ngp_mesh.fast_mesh_index(node_i);
        const stk::mesh::FastMeshIndex node_ip1_index = ngp_mesh.fast_mesh_index(node_ip1);

        // Get the required input fields
        const auto node_i_coords = mesh::vector3_field_data(node_coords_field, node_i_index);
        const auto node_ip1_coords = mesh::vector3_field_data(node_coords_field, node_ip1_index);
        const double node_i_twist = node_twist_field(node_i_index, 0);
        const auto edge_tangent_old = mesh::vector3_field_data(old_edge_tangent_field, edge_index);
        const auto edge_orientation_old = mesh::quaternion_field_data(old_edge_orientation_field, edge_index);

        // Get the output fields
        auto edge_tangent = mesh::vector3_field_data(edge_tangent_field, edge_index);
        auto edge_binormal = mesh::vector3_field_data(edge_binormal_field, edge_index);
        auto edge_orientation = mesh::quaternion_field_data(edge_orientation_field, edge_index);

        // Compute the un-normalized edge tangent
        edge_tangent = node_ip1_coords - node_i_coords;
        edge_length_field(edge_index, 0) = math::norm(edge_tangent);
        edge_tangent /= edge_length_field(edge_index, 0);

        // Compute the edge binormal
        edge_binormal =
            (2.0 * math::cross(edge_tangent_old, edge_tangent)) / (1.0 + math::dot(edge_tangent_old, edge_tangent));

        // Compute the edge orientations
        const double cos_half_t = Kokkos::cos(0.5 * node_i_twist);
        const double sin_half_t = Kokkos::sin(0.5 * node_i_twist);
        const auto rot_via_twist =
            math::Quaterniond(cos_half_t, sin_half_t * edge_tangent_old[0], sin_half_t * edge_tangent_old[1],
                              sin_half_t * edge_tangent_old[2]);
        const auto rot_via_parallel_transport = math::quat_from_parallel_transport(edge_tangent_old, edge_tangent);
        edge_orientation = rot_via_parallel_transport * rot_via_twist * edge_orientation_old;

        // Two things to check:
        //  1. Is the quaternion produced by the parallel transport normalized?
        //  2. Does the application of this quaternion to the old edge tangent produce the new edge tangent?
        //
        // std::cout << "rot_via_parallel_transport: " << rot_via_parallel_transport
        //           << " has norm: " << math::norm(rot_via_parallel_transport) << std::endl;
        // std::cout << "rot_via_twist: " << rot_via_twist << " has norm: " << math::norm(rot_via_twist)
        //           << std::endl;
        // std::cout << "Edge tangent : " << edge_tangent << " Edge tangent old: " << edge_tangent_old << std::endl;
        // std::cout << " Edge tangent via transp: " << rot_via_parallel_transport * edge_tangent_old << std::endl;
        // std::cout << " Edge tangent via orient: " << edge_orientation * math::Vector3d(0.0, 0.0, 1.0)
        //           << std::endl;
      });

  edge_orientation_field.modify_on_device();
  edge_tangent_field.modify_on_device();
  edge_binormal_field.modify_on_device();
  edge_length_field.modify_on_device();
}

void compute_node_curvature_and_rotation_gradient(stk::mesh::NgpMesh &ngp_mesh,
                                                  const stk::mesh::Part &centerline_twist_springs_part,
                                                  NgpDoubleField &edge_orientation_field,
                                                  NgpDoubleField &node_curvature_field,
                                                  NgpDoubleField &node_rotation_gradient_field) {
  debug_print("Computing the node curvature and rotation gradient.");

  edge_orientation_field.sync_to_device();
  node_curvature_field.sync_to_device();
  node_rotation_gradient_field.sync_to_device();

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
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, centerline_twist_springs_part,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &spring_index) {
        // Curvature needs to "know" about the order of edges, so it's best to loop over
        // the slt elements and not the nodes. Get the lower rank entities
        const stk::mesh::NgpMesh::ConnectedEntities edges = ngp_mesh.get_edges(stk::topology::ELEM_RANK, spring_index);
        const stk::mesh::NgpMesh::ConnectedEntities nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spring_index);
        assert(edges.size() == 2);
        assert(nodes.size() == 3);

        const stk::mesh::Entity center_node = nodes[1];
        const stk::mesh::Entity left_edge = edges[0];
        const stk::mesh::Entity right_edge = edges[1];
        const stk::mesh::FastMeshIndex center_node_index = ngp_mesh.fast_mesh_index(center_node);
        const stk::mesh::FastMeshIndex left_edge_index = ngp_mesh.fast_mesh_index(left_edge);
        const stk::mesh::FastMeshIndex right_edge_index = ngp_mesh.fast_mesh_index(right_edge);

        // Get the required input fields
        const auto edge_im1_orientation = mesh::quaternion_field_data(edge_orientation_field, left_edge_index);
        const auto edge_i_orientation = mesh::quaternion_field_data(edge_orientation_field, right_edge_index);

        // Get the output fields
        auto node_curvature = mesh::vector3_field_data(node_curvature_field, center_node_index);
        auto node_rotation_gradient = mesh::quaternion_field_data(node_rotation_gradient_field, center_node_index);

        // Compute the node curvature
        node_rotation_gradient = math::conjugate(edge_im1_orientation) * edge_i_orientation;
        node_curvature = 2.0 * node_rotation_gradient.vector();
      });

  node_curvature_field.modify_on_device();
  node_rotation_gradient_field.modify_on_device();
}

void compute_internal_force_and_twist_torque(
    stk::mesh::NgpMesh &ngp_mesh, const double sperm_rest_segment_length, const double sperm_youngs_modulus,
    const double sperm_poissons_ratio, const stk::mesh::Part &centerline_twist_springs_part,
    NgpDoubleField &node_radius_field, NgpDoubleField &node_curvature_field, NgpDoubleField &node_rest_curvature_field,
    NgpDoubleField &node_rotation_gradient_field, NgpDoubleField &edge_tangent_field,
    NgpDoubleField &edge_binormal_field, NgpDoubleField &edge_length_field, NgpDoubleField &edge_orientation_field,
    NgpDoubleField &node_force_field, NgpDoubleField &node_twist_torque_field) {
  debug_print("Computing the internal force and twist torque.");

  node_radius_field.sync_to_device();
  node_curvature_field.sync_to_device();
  node_rest_curvature_field.sync_to_device();
  node_rotation_gradient_field.sync_to_device();
  edge_tangent_field.sync_to_device();
  edge_binormal_field.sync_to_device();
  edge_length_field.sync_to_device();
  edge_orientation_field.sync_to_device();
  node_force_field.sync_to_device();
  node_twist_torque_field.sync_to_device();

  // Compute internal force and torque induced by differences in rest and current curvature
  // Note, we only loop over locally owned edges to avoid double counting the influence of ghosted edges.
  auto locally_owned_selector = stk::mesh::Selector(centerline_twist_springs_part) &
                                ngp_mesh.get_bulk_on_host().mesh_meta_data().locally_owned_part();
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, locally_owned_selector,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &spring_index) {
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
        const stk::mesh::NgpMesh::ConnectedEntities nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spring_index);
        const stk::mesh::NgpMesh::ConnectedEntities edges = ngp_mesh.get_edges(stk::topology::ELEM_RANK, spring_index);
        const stk::mesh::Entity &node_im1 = nodes[0];
        const stk::mesh::Entity &node_i = nodes[1];
        const stk::mesh::Entity &node_ip1 = nodes[2];
        const stk::mesh::Entity &edge_im1 = edges[0];
        const stk::mesh::Entity &edge_i = edges[1];
        const stk::mesh::FastMeshIndex node_im1_index = ngp_mesh.fast_mesh_index(node_im1);
        const stk::mesh::FastMeshIndex node_i_index = ngp_mesh.fast_mesh_index(node_i);
        const stk::mesh::FastMeshIndex node_ip1_index = ngp_mesh.fast_mesh_index(node_ip1);
        const stk::mesh::FastMeshIndex edge_im1_index = ngp_mesh.fast_mesh_index(edge_im1);
        const stk::mesh::FastMeshIndex edge_i_index = ngp_mesh.fast_mesh_index(edge_i);

        // Get the required input fields
        const auto node_i_curvature = mesh::vector3_field_data(node_curvature_field, node_i_index);
        const auto node_i_rest_curvature = mesh::vector3_field_data(node_rest_curvature_field, node_i_index);
        const auto node_i_rotation_gradient = mesh::quaternion_field_data(node_rotation_gradient_field, node_i_index);
        const double node_radius = node_radius_field(node_i_index, 0);
        const auto edge_im1_tangent = mesh::vector3_field_data(edge_tangent_field, edge_im1_index);
        const auto edge_i_tangent = mesh::vector3_field_data(edge_tangent_field, edge_i_index);
        const auto edge_im1_binormal = mesh::vector3_field_data(edge_binormal_field, edge_im1_index);
        const auto edge_i_binormal = mesh::vector3_field_data(edge_binormal_field, edge_i_index);
        const double edge_im1_length = edge_length_field(edge_im1_index, 0);
        const double edge_i_length = edge_length_field(edge_i_index, 0);
        const auto edge_im1_orientation = mesh::quaternion_field_data(edge_orientation_field, edge_im1_index);

        // Get the output fields
        auto node_im1_force = mesh::vector3_field_data(node_force_field, node_im1_index);
        auto node_i_force = mesh::vector3_field_data(node_force_field, node_i_index);
        auto node_ip1_force = mesh::vector3_field_data(node_force_field, node_ip1_index);

        // Compute the Lagrangian torque induced by the curvature
        auto delta_curvature = node_i_curvature - node_i_rest_curvature;
        const double moment_of_inertia = 0.25 * M_PI * node_radius * node_radius * node_radius * node_radius;
        const double shear_modulus = 0.5 * sperm_youngs_modulus / (1.0 + sperm_poissons_ratio);
        const double inv_rest_segment_length = 1.0 / sperm_rest_segment_length;
        auto bending_torque =
            math::Vector3d(-inv_rest_segment_length * sperm_youngs_modulus * moment_of_inertia * delta_curvature[0],
                           -inv_rest_segment_length * sperm_youngs_modulus * moment_of_inertia * delta_curvature[1],
                           -inv_rest_segment_length * 2 * shear_modulus * moment_of_inertia * delta_curvature[2]);

        // We'll reuse the bending torque for the rotated bending torque
        bending_torque = edge_im1_orientation * (node_i_rotation_gradient.w() * bending_torque +
                                                 math::cross(node_i_rotation_gradient.vector(), bending_torque));

        // Compute the force and torque on the nodes
        const auto tmp_force_ip1 = 1.0 / edge_i_length *
                                   (math::cross(bending_torque, edge_i_tangent) +
                                    0.5 * math::dot(edge_i_tangent, bending_torque) *
                                        (math::dot(edge_i_tangent, edge_i_binormal) * edge_i_tangent) -
                                    edge_i_binormal);
        const auto tmp_force_im1 =
            1.0 / edge_im1_length *
            (math::cross(bending_torque, edge_im1_tangent) +
             0.5 * math::dot(edge_im1_tangent, bending_torque) *
                 (math::dot(edge_im1_tangent, edge_im1_binormal) * edge_im1_tangent - edge_im1_binormal));

        Kokkos::atomic_add(&node_twist_torque_field(node_i_index, 0), math::dot(edge_i_tangent, bending_torque));
        Kokkos::atomic_add(&node_twist_torque_field(node_im1_index, 0), -math::dot(edge_im1_tangent, bending_torque));
        Kokkos::atomic_add(&node_ip1_force[0], tmp_force_ip1[0]);
        Kokkos::atomic_add(&node_ip1_force[1], tmp_force_ip1[1]);
        Kokkos::atomic_add(&node_ip1_force[2], tmp_force_ip1[2]);
        Kokkos::atomic_add(&node_i_force[0], -tmp_force_ip1[0] - tmp_force_im1[0]);
        Kokkos::atomic_add(&node_i_force[1], -tmp_force_ip1[1] - tmp_force_im1[1]);
        Kokkos::atomic_add(&node_i_force[2], -tmp_force_ip1[2] - tmp_force_im1[2]);
        Kokkos::atomic_add(&node_im1_force[0], tmp_force_im1[0]);
        Kokkos::atomic_add(&node_im1_force[1], tmp_force_im1[1]);
        Kokkos::atomic_add(&node_im1_force[2], tmp_force_im1[2]);
      });

  // Compute internal force induced by differences in rest and current length
  // Note, we only loop over locally owned edges to avoid double counting the influence of ghosted edges.
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::EDGE_RANK, locally_owned_selector,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &edge_index) {
        // F_left = k (l - l_rest) tangent
        // F_right = -k (l - l_rest) tangent
        //
        // k can be computed using the material properties of the rod according to k = E A / l_rest where E is the
        // Young's modulus, A is the cross-sectional area, and l_rest is the rest length of the rod.

        // Get the lower rank entities
        const stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::EDGE_RANK, edge_index);
        const stk::mesh::Entity &node_im1 = nodes[0];
        const stk::mesh::Entity &node_i = nodes[1];
        const stk::mesh::FastMeshIndex node_im1_index = ngp_mesh.fast_mesh_index(node_im1);
        const stk::mesh::FastMeshIndex node_i_index = ngp_mesh.fast_mesh_index(node_i);

        // Get the required input fields
        const auto edge_tangent = mesh::vector3_field_data(edge_tangent_field, edge_index);
        const double edge_length = edge_length_field(edge_index, 0);
        const double node_radius = node_radius_field(node_i_index, 0);

        // Get the output fields
        auto node_im1_force = mesh::vector3_field_data(node_force_field, node_im1_index);
        auto node_i_force = mesh::vector3_field_data(node_force_field, node_i_index);

        // Compute the internal force
        const double spring_constant =
            sperm_youngs_modulus * M_PI * node_radius * node_radius / sperm_rest_segment_length;
        const auto right_node_force = -spring_constant * (edge_length - sperm_rest_segment_length) * edge_tangent;
        Kokkos::atomic_add(&node_im1_force[0], -right_node_force[0]);
        Kokkos::atomic_add(&node_im1_force[1], -right_node_force[1]);
        Kokkos::atomic_add(&node_im1_force[2], -right_node_force[2]);

        Kokkos::atomic_add(&node_i_force[0], right_node_force[0]);
        Kokkos::atomic_add(&node_i_force[1], right_node_force[1]);
        Kokkos::atomic_add(&node_i_force[2], right_node_force[2]);
      });

  // Sum the node force and torque over shared nodes (only if multiple ranks are present)
  if (ngp_mesh.get_bulk_on_host().parallel_size() > 1) {
    stk::mesh::parallel_sum(ngp_mesh.get_bulk_on_host(),
                            std::vector<NgpDoubleField *>{&node_force_field, &node_twist_torque_field});
  }

  node_force_field.modify_on_device();
  node_twist_torque_field.modify_on_device();
}

//! \name Search
//@{

struct FastMeshIndexAndPeriodicShift {
  stk::mesh::FastMeshIndex mesh_index;
  math::Vector3d shift;
};

KOKKOS_INLINE_FUNCTION
constexpr bool operator<(const FastMeshIndexAndPeriodicShift &lhs, const FastMeshIndexAndPeriodicShift &rhs) {
  return fma_less(lhs.mesh_index, rhs.mesh_index);
}

KOKKOS_INLINE_FUNCTION
constexpr bool operator==(const FastMeshIndexAndPeriodicShift &lhs, const FastMeshIndexAndPeriodicShift &rhs) {
  return fma_equal(lhs.mesh_index, rhs.mesh_index);
}

using ExecSpace = stk::ngp::ExecSpace;
using IdentProc = stk::search::IdentProc<FastMeshIndexAndPeriodicShift, int>;
using BoxIdentProc = stk::search::BoxIdentProc<stk::search::Box<double>, IdentProc>;
using Intersection = stk::search::IdentProcIntersection<IdentProc, IdentProc>;
using SearchBoxesViewType = Kokkos::View<BoxIdentProc *, ExecSpace>;
using ResultViewType = Kokkos::View<Intersection *, ExecSpace>;
using FastMeshIndicesViewType = Kokkos::View<stk::mesh::FastMeshIndex *, ExecSpace>;

using LocalIdentProc = stk::search::IdentProc<stk::mesh::FastMeshIndex, int>;
using LocalIntersection = stk::search::IdentProcIntersection<LocalIdentProc, LocalIdentProc>;
using LocalResultViewType = Kokkos::View<LocalIntersection *, ExecSpace>;

// Create local entities on host and copy to device
// Create local entities on host and copy to device
FastMeshIndicesViewType get_local_entity_indices(const stk::mesh::BulkData &bulk_data, stk::mesh::EntityRank rank,
                                                 const stk::mesh::Selector &selector) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities);

  FastMeshIndicesViewType mesh_indices("mesh_indices", local_entities.size());
  FastMeshIndicesViewType::HostMirror host_mesh_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, mesh_indices);

  Kokkos::parallel_for(stk::ngp::HostRangePolicy(0, local_entities.size()), [&bulk_data, &local_entities,
                                                                             &host_mesh_indices](const int i) {
    const stk::mesh::MeshIndex &mesh_index = bulk_data.mesh_index(local_entities[i]);
    host_mesh_indices(i) = stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
  });

  Kokkos::deep_copy(mesh_indices, host_mesh_indices);
  return mesh_indices;
}

void compute_aabbs(stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &segments,
                   stk::mesh::NgpField<double> &node_coords_field, stk::mesh::NgpField<double> &elem_radius_field,
                   stk::mesh::NgpField<double> &elem_aabb_field) {
  node_coords_field.sync_to_device();
  elem_radius_field.sync_to_device();
  elem_aabb_field.sync_to_device();

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, segments, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &segment_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, segment_index);
        stk::mesh::FastMeshIndex node0_index = ngp_mesh.fast_mesh_index(nodes[0]);
        stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(nodes[1]);

        const auto node0_coords = mesh::vector3_field_data(node_coords_field, node0_index);
        const auto node1_coords = mesh::vector3_field_data(node_coords_field, node1_index);
        const double radius = elem_radius_field(segment_index, 0);

        double min_x = Kokkos::min(node0_coords[0], node1_coords[0]) - radius;
        double min_y = Kokkos::min(node0_coords[1], node1_coords[1]) - radius;
        double min_z = Kokkos::min(node0_coords[2], node1_coords[2]) - radius;
        double max_x = Kokkos::max(node0_coords[0], node1_coords[0]) + radius;
        double max_y = Kokkos::max(node0_coords[1], node1_coords[1]) + radius;
        double max_z = Kokkos::max(node0_coords[2], node1_coords[2]) + radius;

        elem_aabb_field(segment_index, 0) = min_x;
        elem_aabb_field(segment_index, 1) = min_y;
        elem_aabb_field(segment_index, 2) = min_z;
        elem_aabb_field(segment_index, 3) = max_x;
        elem_aabb_field(segment_index, 4) = max_y;
        elem_aabb_field(segment_index, 5) = max_z;
      });

  elem_aabb_field.modify_on_device();
}

template <typename Metric>
Kokkos::pair<SearchBoxesViewType, SearchBoxesViewType> create_search_aabbs(
    const stk::mesh::BulkData &bulk_data, const stk::mesh::NgpMesh &ngp_mesh, const double search_buffer,
    const Metric &metric, const stk::mesh::Selector &segments, stk::mesh::NgpField<double> &elem_aabb_field) {
  elem_aabb_field.sync_to_device();

  auto locally_owned_segments = segments & bulk_data.mesh_meta_data().locally_owned_part();
  const unsigned num_local_segments =
      stk::mesh::count_entities(bulk_data, stk::topology::ELEM_RANK, locally_owned_segments);
  SearchBoxesViewType target_search_aabbs("target_search_aabbs", num_local_segments);      // no periodicity
  SearchBoxesViewType source_search_aabbs("source_search_aabbs", 9 * num_local_segments);  // 2d periodicity in y and z

  // Slow host operation that is needed to get an index. There is plans to add this to the stk::mesh::NgpMesh.
  FastMeshIndicesViewType segment_indices =
      get_local_entity_indices(bulk_data, stk::topology::ELEM_RANK, locally_owned_segments);
  const int my_rank = bulk_data.parallel_rank();

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, num_local_segments), KOKKOS_LAMBDA(const unsigned &i) {
        stk::mesh::FastMeshIndex segment_index = segment_indices(i);

        // Methodology is as follows:
        //  - Wrap the bottom left corner of the LAB frame AABB of the segment into the domain. This is our target
        //  segment
        //  - Compute the wrap displacement vector from original to wrapped reference position
        //  - Stamp out the 9 periodic images of source segments, computing their shift vector and displacement from
        //  original
        //  - Compute the source AABBs
        //  - Stash both the source and target AABBs
        auto aabb = mesh::aabb_field_data(elem_aabb_field, segment_index);
        auto wrapped_aabb = geom::wrap_rigid(aabb, metric);
        FastMeshIndexAndPeriodicShift target_fma_and_shift{segment_index,
                                                           wrapped_aabb.min_corner() - aabb.min_corner()};
        target_search_aabbs(i) =
            BoxIdentProc{stk::search::Box<double>{wrapped_aabb[0] - search_buffer, wrapped_aabb[1] - search_buffer,
                                                  wrapped_aabb[2] - search_buffer, wrapped_aabb[3] + search_buffer,
                                                  wrapped_aabb[4] + search_buffer, wrapped_aabb[5] + search_buffer},
                         IdentProc(target_fma_and_shift, my_rank)};

        for (int s0 = 0; s0 < 3; s0++) {  // s0, s1 in [0, 1, 2]
          for (int s1 = 0; s1 < 3; s1++) {
            math::Vector3<int> lattice_shift{0, s0 - 1, s1 - 1};
            auto shifted_aabb = geom::shift_image(wrapped_aabb, lattice_shift, metric);
            FastMeshIndexAndPeriodicShift source_fma_and_shift{segment_index,
                                                               shifted_aabb.min_corner() - aabb.min_corner()};
            source_search_aabbs(9 * i + 3 * s0 + s1) =
                BoxIdentProc{stk::search::Box<double>{shifted_aabb[0] - search_buffer, shifted_aabb[1] - search_buffer,
                                                      shifted_aabb[2] - search_buffer, shifted_aabb[3] + search_buffer,
                                                      shifted_aabb[4] + search_buffer, shifted_aabb[5] + search_buffer},
                             IdentProc(source_fma_and_shift, my_rank)};
          }
        }
      });

  return Kokkos::make_pair(target_search_aabbs, source_search_aabbs);
}

void compute_hertzian_contact_force_and_torque(const stk::mesh::BulkData &bulk_data, stk::mesh::NgpMesh &ngp_mesh,
                                               const double sperm_youngs_modulus, const double sperm_poissons_ratio,
                                               const stk::mesh::Part &spherocylinder_segments_part,
                                               const ResultViewType &search_results, NgpDoubleField &node_coords_field,
                                               NgpDoubleField &elem_radius_field, NgpDoubleField &node_force_field) {
  debug_print("Computing the Hertzian contact force and torque.");

  // Plan:
  //   Loop over each spherocylinder segment in a for_each_entity_run. (These are our target segments.)
  //   Use a regular for loop over all other spherocylinder segments. (These are our source segments.)
  //   Use an initial cancellation step to check if the bounding spheres of the segments overlap.
  //   If they do, find the minimum signed separation distance between the segments.
  //   If the signed signed separation distance is less than the sum of the radii, compute the contact force and torque.
  //   Sum the result into the target segment. By construction, this sum need not be atomic.
  node_coords_field.sync_to_device();
  elem_radius_field.sync_to_device();
  node_force_field.sync_to_device();

  const double effective_youngs_modulus =
      (sperm_youngs_modulus * sperm_youngs_modulus) /
      (sperm_youngs_modulus - sperm_youngs_modulus * sperm_poissons_ratio * sperm_poissons_ratio +
       sperm_youngs_modulus - sperm_youngs_modulus * sperm_poissons_ratio * sperm_poissons_ratio);
  constexpr double four_thirds = 4.0 / 3.0;
  Kokkos::parallel_for(
      "apply_hertzian_contact_between_segments", stk::ngp::DeviceRangePolicy(0, search_results.size()),
      KOKKOS_LAMBDA(const unsigned &i) {
        const auto search_result = search_results(i);
        auto source_fma_and_shift = search_result.domainIdentProc.id();
        auto target_fma_and_shift = search_result.rangeIdentProc.id();
        const stk::mesh::FastMeshIndex source_segment_index = source_fma_and_shift.mesh_index;
        const stk::mesh::FastMeshIndex target_segment_index = target_fma_and_shift.mesh_index;

        if (fma_less(target_segment_index, source_segment_index) ||
            fma_equal(source_segment_index, target_segment_index)) {
          // Skip self interaction, assuming that the domain is large enough that we cannot collide with our own
          // periodic image. Also, skip double counting.
          return;
        }

        // Fetch the source segment nodes
        stk::mesh::NgpMesh::ConnectedNodes source_nodes =
            ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_segment_index);
        stk::mesh::FastMeshIndex source_node0_index = ngp_mesh.fast_mesh_index(source_nodes[0]);
        stk::mesh::FastMeshIndex source_node1_index = ngp_mesh.fast_mesh_index(source_nodes[1]);

        // Fetch the target segment nodes
        stk::mesh::NgpMesh::ConnectedNodes target_nodes =
            ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_segment_index);
        stk::mesh::FastMeshIndex target_node0_index = ngp_mesh.fast_mesh_index(target_nodes[0]);
        stk::mesh::FastMeshIndex target_node1_index = ngp_mesh.fast_mesh_index(target_nodes[1]);

        // Skip neighboring segments (those that share a node)
        if (fma_equal(source_node0_index, target_node0_index) || fma_equal(source_node0_index, target_node1_index) ||
            fma_equal(source_node1_index, target_node0_index) || fma_equal(source_node1_index, target_node1_index)) {
          return;
        }

        /////////////////
        // Source data //
        const auto source_node0_coords =
            mesh::vector3_field_data(node_coords_field, source_node0_index) + source_fma_and_shift.shift;
        const auto source_node1_coords =
            mesh::vector3_field_data(node_coords_field, source_node1_index) + source_fma_and_shift.shift;
        const double source_radius = elem_radius_field(source_segment_index, 0);
        auto source_node0_force = mesh::vector3_field_data(node_force_field, source_node0_index);
        auto source_node1_force = mesh::vector3_field_data(node_force_field, source_node1_index);

        // Target data
        const auto target_node0_coords =
            mesh::vector3_field_data(node_coords_field, target_node0_index) + target_fma_and_shift.shift;
        const auto target_node1_coords =
            mesh::vector3_field_data(node_coords_field, target_node1_index) + target_fma_and_shift.shift;
        const double target_radius = elem_radius_field(target_segment_index, 0);
        auto target_node0_force = mesh::vector3_field_data(node_force_field, target_node0_index);
        auto target_node1_force = mesh::vector3_field_data(node_force_field, target_node1_index);

        // Compute the minimum signed separation distance between the segments
        math::Vector3d closest_point_source;
        math::Vector3d closest_point_target;
        double archlength_source;
        double archlength_target;
        const double distance = Kokkos::sqrt(math::distance::distance_sq_between_line_segments(
            source_node0_coords, source_node1_coords, target_node0_coords, target_node1_coords, closest_point_source,
            closest_point_target, archlength_source, archlength_target));

        const auto source_to_target_vector = closest_point_target - closest_point_source;
        double signed_separation_distance = distance - source_radius - target_radius;
        if (signed_separation_distance > 0) {
          signed_separation_distance = 0.0;
        }

        // Compute the contact force and torque
        const double inv_distance = 1.0 / distance;
        const auto source_normal = inv_distance * source_to_target_vector;

        // Compute the Hertzian contact force magnitude
        // Note, signed separation distance is negative when particles overlap,
        // so delta = -signed_separation_distance.
        const double effective_radius = (source_radius * target_radius) / (source_radius + target_radius);
        const double normal_force_magnitude = four_thirds * effective_youngs_modulus * Kokkos::sqrt(effective_radius) *
                                              Kokkos::pow(-signed_separation_distance, 1.5);
        const auto source_contact_force = -normal_force_magnitude * source_normal;

        {
          // Sum the force into the source segment nodes.
          const auto left_to_cp = closest_point_source - source_node0_coords;
          const auto left_to_right = source_node1_coords - source_node0_coords;
          const double length = math::norm(left_to_right);
          const double inv_length = 1.0 / length;
          const auto tangent = left_to_right * inv_length;
          const auto term1 = math::dot(tangent, source_contact_force) * left_to_cp * inv_length;
          const auto term2 = math::dot(left_to_cp, tangent) *
                             (source_contact_force + math::dot(tangent, source_contact_force) * tangent) * inv_length;
          const auto sum = term2 - term1;

          // Use an atomic add to sum the forces into the source
          Kokkos::atomic_add(&source_node0_force[0], source_contact_force[0] - sum[0]);
          Kokkos::atomic_add(&source_node0_force[1], source_contact_force[1] - sum[1]);
          Kokkos::atomic_add(&source_node0_force[2], source_contact_force[2] - sum[2]);
          Kokkos::atomic_add(&source_node1_force[0], sum[0]);
          Kokkos::atomic_add(&source_node1_force[1], sum[1]);
          Kokkos::atomic_add(&source_node1_force[2], sum[2]);
        }
        {
          // Sum the force into the target segment nodes.
          const auto left_to_cp = closest_point_target - target_node0_coords;
          const auto left_to_right = target_node1_coords - target_node0_coords;
          const double length = math::norm(left_to_right);
          const double inv_length = 1.0 / length;
          const auto tangent = left_to_right * inv_length;
          const auto term1 = math::dot(tangent, -source_contact_force) * left_to_cp * inv_length;
          const auto term2 = math::dot(left_to_cp, tangent) *
                             (-source_contact_force + math::dot(tangent, -source_contact_force) * tangent) * inv_length;
          const auto sum = term2 - term1;
          // Use an atomic add to sum the forces into the target
          Kokkos::atomic_add(&target_node0_force[0], -source_contact_force[0] - sum[0]);
          Kokkos::atomic_add(&target_node0_force[1], -source_contact_force[1] - sum[1]);
          Kokkos::atomic_add(&target_node0_force[2], -source_contact_force[2] - sum[2]);
          Kokkos::atomic_add(&target_node1_force[0], sum[0]);
          Kokkos::atomic_add(&target_node1_force[1], sum[1]);
          Kokkos::atomic_add(&target_node1_force[2], sum[2]);
        }
      });

  node_force_field.modify_on_device();
}

void compute_generalized_velocity(stk::mesh::NgpMesh &ngp_mesh, const double viscosity,
                                  const stk::mesh::Part &spherocylinder_segments_part,
                                  NgpDoubleField &node_radius_field, NgpDoubleField &node_force_field,
                                  NgpDoubleField &node_twist_torque_field, NgpDoubleField &node_velocity_field,
                                  NgpDoubleField &node_twist_velocity_field) {
  debug_print("Computing the generalized velocity using the mobility problem.");

  node_radius_field.sync_to_device();
  node_force_field.sync_to_device();
  node_twist_torque_field.sync_to_device();
  node_velocity_field.sync_to_device();
  node_twist_velocity_field.sync_to_device();

  // For us, we consider dry local drag with mass lumping at the nodes. This diagonalized the mobility problem and
  // makes each node independent, coupled only through the internal and constrainmt forces. The mobility problem is
  //
  // \dot{x}(t) = f(t) / (6 pi viscosity r)
  // \dot{twist}(t) = torque(t) / (8 pi viscosity r^3)

  // Solve the mobility problem for the nodes
  const double one_over_6_pi_viscosity = 1.0 / (6.0 * M_PI * viscosity);
  const double one_over_8_pi_viscosity = 1.0 / (8.0 * M_PI * viscosity);
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, spherocylinder_segments_part,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        // Get the required input fields
        const auto node_force = mesh::vector3_field_data(node_force_field, node_index);
        const double node_radius = node_radius_field(node_index, 0);
        const double node_twist_torque = node_twist_torque_field(node_index, 0);

        assert(node_radius > 1e-12);

        // Get the output fields
        auto node_velocity = mesh::vector3_field_data(node_velocity_field, node_index);
        auto node_twist_velocity = node_twist_velocity_field(node_index, 0);

        // Compute the generalized velocity
        const double inv_node_radius = 1.0 / node_radius;
        const double inv_node_radius3 = inv_node_radius * inv_node_radius * inv_node_radius;
        node_velocity = (one_over_6_pi_viscosity * inv_node_radius) * node_force;
        node_twist_velocity = (one_over_8_pi_viscosity * inv_node_radius3) * node_twist_torque;
      });

  node_velocity_field.modify_on_device();
  node_twist_velocity_field.modify_on_device();
}

void update_generalized_position(stk::mesh::NgpMesh &ngp_mesh, const double timestep_size,
                                 const stk::mesh::Part &centerline_twist_springs_part,
                                 NgpDoubleField &old_node_coords_field, NgpDoubleField &old_node_twist_field,
                                 NgpDoubleField &old_node_velocity_field, NgpDoubleField &old_node_twist_velocity_field,
                                 NgpDoubleField &node_coords_field, NgpDoubleField &node_twist_field) {
  debug_print("Updating the generalized position using Euler's method.");

  old_node_coords_field.sync_to_device();
  old_node_twist_field.sync_to_device();
  old_node_velocity_field.sync_to_device();
  old_node_twist_velocity_field.sync_to_device();
  node_coords_field.sync_to_device();
  node_twist_field.sync_to_device();

  // Update the generalized position using Euler's method
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, centerline_twist_springs_part,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        // Update the generalized position
        const auto old_node_coord = mesh::vector3_field_data(old_node_coords_field, node_index);
        const auto old_node_velocity = mesh::vector3_field_data(old_node_velocity_field, node_index);
        const auto old_node_twist = old_node_twist_field(node_index, 0);
        const auto old_node_twist_velocity = old_node_twist_velocity_field(node_index, 0);

        auto node_coord = mesh::vector3_field_data(node_coords_field, node_index);
        auto node_twist = node_twist_field(node_index, 0);

        node_coord = old_node_coord + timestep_size * old_node_velocity;
        node_twist = old_node_twist + timestep_size * old_node_twist_velocity;
      });

  node_coords_field.modify_on_device();
  node_twist_field.modify_on_device();
}

void disable_twist(stk::mesh::NgpMesh &ngp_mesh, NgpDoubleField &node_twist_field,
                   NgpDoubleField &node_twist_velocity_field) {
  debug_print("Disabling twist.");

  // Set the twist and twist velocity, to zero.
  node_twist_field.sync_to_device();
  node_twist_velocity_field.sync_to_device();

  node_twist_field.set_all(ngp_mesh, 0.0);
  node_twist_velocity_field.set_all(ngp_mesh, 0.0);

  node_twist_field.modify_on_device();
  node_twist_velocity_field.modify_on_device();
}

void apply_monolayer(stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Part &centerline_twist_springs_part,
                     NgpDoubleField &node_coords_field, NgpDoubleField &node_velocity_field) {
  debug_print("Applying the monolayer (y-z plane).");

  node_coords_field.sync_to_device();
  node_velocity_field.sync_to_device();

  // Set the x-coordinate of the nodes to zero.
  // Set the x-velocity of the nodes to zero.
  mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, centerline_twist_springs_part,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        // Apply the monolayer
        node_coords_field(node_index, 0) = 0.0;
        node_velocity_field(node_index, 0) = 0.0;
      });

  node_coords_field.modify_on_device();
  node_velocity_field.modify_on_device();
}

// void equilibriate() {
//   debug_print("Equilibriating the system.");

//   // Notes:
//   // Equilibiration is necessary because we are  starting from straight rods and attempting to equilibriate to a
//   // sinusoidal shape.
//   //
//   // This isn't so much the problem as the fact that our Young's modulus is ~6e16! This is astronomically high and is
//   // causing instabilities. We need to equilibriate to a system that is stable under these conditions.
//   //
//   // I have tried the following:
//   //  1. Starting at a Young's Modulous of 1e6 and running for a million timesteps before increasing to 6e16 (kaboom)
//   //  2. Starting at a Young's Modulous of 1e6 and increasing by a factor of 10 every 100,000 timesteps (kaboom)
//   //  3. Starting at a Young's Modulous of 1e6 and running for a million timesteps before increasing to exponentially
//   //  as in 2.
//   //
//   // I now want to try a more intentional approach based on the kinetic energy of the system.
//   // Starying at 1e6, I will let the system evolve until its kinetic energy is less than some threshold value. I will
//   // then increase the Young's modulus by a factor of 10 and repeat.
//   size_t count = 0;
//   for (size_t count = 0; count < 1000000; count++) {
//     if (count % 1000 == 0) {
//       std::cout << "Equilibriating the system. Iteration " << count << std::endl;
//     }
//     // Prepare the current configuration.
//     {
//       // Apply constraints before we move the nodes.
//       disable_twist();
//       apply_monolayer();

//       // Rotate the field states.
//       rotate_field_states();

//       // Move the nodes from t -> t + dt.
//       //   x(t + dt) = x(t) + dt v(t)
//       update_generalized_position();

//       // Reset the fields in the current timestep.
//       zero_out_transient_node_fields();
//     }

//     // Evaluate forces f(x(t + dt)).
//     {
//       // Hertzian contact force
//       compute_hertzian_contact_force_and_torque();

//       // Centerline twist rod forces
//       compute_edge_information();
//       compute_node_curvature_and_rotation_gradient();
//       compute_internal_force_and_twist_torque();
//     }

//     // Compute velocity v(x(t+dt))
//     {
//       // Compute the current velocity from the current forces.
//       compute_generalized_velocity();
//     }
//   }
// }

template <typename FieldValueType, int FieldDimension>
void deep_copy(stk::mesh::NgpMesh &ngp_mesh, stk::mesh::NgpField<FieldValueType> &target_field,
               stk::mesh::NgpField<FieldValueType> &source_field, const stk::mesh::Selector &selector) {
  target_field.sync_to_device();
  source_field.sync_to_device();

  stk::mesh::for_each_entity_run(
      ngp_mesh, target_field.get_rank(), selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &index) {
        for (int i = 0; i < FieldDimension; ++i) {
          target_field(index, i) = source_field(index, i);
        }
      });

  target_field.modify_on_device();
}

void run(int argc, char **argv) {
  debug_print("Running the simulation.");

  // Preprocess
  RunConfig run_config;
  run_config.parse_user_inputs(argc, argv);
  run_config.print();

  // Setup the STK mesh
  stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3);
  mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEM", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition to
                                       // all fields are simple.
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create(meta_data_ptr);
  stk::mesh::MetaData &meta_data = *meta_data_ptr;
  stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

  // Declare all the fields
  DoubleField &node_coords_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  DoubleField &old_node_coords_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "OLD_NODE_COORDS");
  DoubleField &node_displacement_since_last_rebuild_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "OUR_DISP");
  DoubleField &node_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_VELOCITY");
  DoubleField &old_node_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "OLD_NODE_VELOCITY");
  DoubleField &node_force_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_FORCE");
  DoubleField &node_twist_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_TWIST");
  DoubleField &old_node_twist_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "OLD_NODE_TWIST");
  DoubleField &node_twist_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_TWIST_VELOCITY");
  DoubleField &old_node_twist_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "OLD_NODE_TWIST_VELOCITY");
  DoubleField &node_twist_torque_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_TWIST_TORQUE");
  DoubleField &node_curvature_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_CURVATURE");
  DoubleField &node_rest_curvature_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_REST_CURVATURE");
  DoubleField &node_rotation_gradient_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_ROTATION_GRADIENT");
  DoubleField &node_radius_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_RADIUS");
  DoubleField &node_archlength_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_ARCHLENGTH");
  IntField &node_sperm_id_field =  //
      meta_data.declare_field<int>(stk::topology::NODE_RANK, "NODE_SPERM_ID");

  DoubleField &edge_orientation_field =  //
      meta_data.declare_field<double>(stk::topology::EDGE_RANK, "EDGE_ORIENTATION");
  DoubleField &old_edge_orientation_field =  //
      meta_data.declare_field<double>(stk::topology::EDGE_RANK, "OLD_EDGE_ORIENTATION");
  DoubleField &edge_tangent_field =  //
      meta_data.declare_field<double>(stk::topology::EDGE_RANK, "EDGE_TANGENT");
  DoubleField &old_edge_tangent_field =  //
      meta_data.declare_field<double>(stk::topology::EDGE_RANK, "OLD_EDGE_TANGENT");
  DoubleField &edge_binormal_field =  //
      meta_data.declare_field<double>(stk::topology::EDGE_RANK, "EDGE_BINORMAL");
  DoubleField &edge_length_field =  //
      meta_data.declare_field<double>(stk::topology::EDGE_RANK, "EDGE_LENGTH");

  DoubleField &elem_radius_field =  //
      meta_data.declare_field<double>(stk::topology::ELEM_RANK, "ELEM_RADIUS");
  DoubleField &elem_rest_length_field =  //
      meta_data.declare_field<double>(stk::topology::ELEM_RANK, "ELEM_REST_LENGTH");

  DoubleField &elem_aabb_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "ELEM_AABB");
  DoubleField &elem_old_aabb_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "ELEM_OLD_AABB");
  DoubleField &elem_aabb_disp_since_last_rebuild_field =
      meta_data.declare_field<double>(stk::topology::ELEM_RANK, "ELEM_AABB_DISPLACEMENT");

  // Assign the field output types
  stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_force_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_twist_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_twist_velocity_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_twist_torque_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_curvature_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_rest_curvature_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_rotation_gradient_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_archlength_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_sperm_id_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(edge_orientation_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(edge_tangent_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(edge_binormal_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(edge_length_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_length_field, Ioss::Field::TRANSIENT);

  stk::io::set_field_output_type(node_coords_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_twist_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_twist_velocity_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_twist_torque_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_curvature_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_rest_curvature_field, stk::io::FieldOutputType::VECTOR_3D);
  // stk::io::set_field_output_type(node_rotation_gradient_field, ...);  // No quaternion type with Ioss/Exodus/VTK
  stk::io::set_field_output_type(node_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_archlength_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_sperm_id_field, stk::io::FieldOutputType::SCALAR);
  // stk::io::set_field_output_type(edge_orientation_field, ...);   // No quaternion type with Ioss/Exodus/VTK
  stk::io::set_field_output_type(edge_tangent_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(edge_binormal_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(edge_length_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_length_field, stk::io::FieldOutputType::SCALAR);

  // Declare the parts
  stk::mesh::Part &boundary_sperm_part = meta_data.declare_part("BOUNDARY_SPERM", stk::topology::ELEM_RANK);
  stk::mesh::Part &centerline_twist_springs_part =
      meta_data.declare_part_with_topology("CENTERLINE_TWIST_SPRINGS", stk::topology::SHELL_TRI_3);
  stk::mesh::Part &spherocylinder_segments_part =
      meta_data.declare_part_with_topology("SPHEROCYLINDER_SEGMENTS", stk::topology::BEAM_2);
  // stk::io::put_io_part_attribute(boundary_sperm_part);  // There are special ways to write out element-rank parts
  stk::io::put_io_part_attribute(centerline_twist_springs_part);
  stk::io::put_io_part_attribute(spherocylinder_segments_part);

  // Assign fields to parts
  double init_zero6[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  stk::mesh::Part &universal_part = meta_data.universal_part();
  stk::mesh::put_field_on_mesh(node_coords_field, universal_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(old_node_coords_field, universal_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, universal_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(old_node_velocity_field, universal_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, universal_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_twist_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(old_node_twist_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_twist_velocity_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(old_node_twist_velocity_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_twist_torque_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_curvature_field, universal_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rest_curvature_field, universal_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rotation_gradient_field, universal_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_radius_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_archlength_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_sperm_id_field, universal_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(edge_orientation_field, centerline_twist_springs_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(old_edge_orientation_field, centerline_twist_springs_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(edge_tangent_field, centerline_twist_springs_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(old_edge_tangent_field, centerline_twist_springs_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(edge_binormal_field, centerline_twist_springs_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(edge_length_field, centerline_twist_springs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, spherocylinder_segments_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, centerline_twist_springs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rest_length_field, centerline_twist_springs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_aabb_field, spherocylinder_segments_part, 6, init_zero6);
  stk::mesh::put_field_on_mesh(elem_old_aabb_field, spherocylinder_segments_part, 6, init_zero6);
  stk::mesh::put_field_on_mesh(elem_aabb_disp_since_last_rebuild_field, spherocylinder_segments_part, 6, init_zero6);

  // Concretize the mesh
  meta_data.commit();

  // Setup the IO broker
  stk::io::StkMeshIoBroker stk_io_broker(MPI_COMM_WORLD);
  stk_io_broker.use_simple_fields();
  stk_io_broker.set_bulk_data(bulk_data);
  stk_io_broker.property_add(Ioss::Property("MAXIMUM_NAME_LENGTH", 180));
  size_t output_file_index = stk_io_broker.create_output_mesh("Sperm.exo", stk::io::WRITE_RESULTS);
  stk_io_broker.add_field(output_file_index, node_coords_field);
  stk_io_broker.add_field(output_file_index, node_velocity_field);
  stk_io_broker.add_field(output_file_index, node_force_field);
  stk_io_broker.add_field(output_file_index, node_twist_field);
  stk_io_broker.add_field(output_file_index, node_twist_velocity_field);
  stk_io_broker.add_field(output_file_index, node_twist_torque_field);
  stk_io_broker.add_field(output_file_index, node_curvature_field);
  stk_io_broker.add_field(output_file_index, node_rest_curvature_field);
  stk_io_broker.add_field(output_file_index, node_rotation_gradient_field);
  stk_io_broker.add_field(output_file_index, node_radius_field);
  stk_io_broker.add_field(output_file_index, node_archlength_field);
  stk_io_broker.add_field(output_file_index, node_sperm_id_field);
  stk_io_broker.add_field(output_file_index, edge_orientation_field);
  stk_io_broker.add_field(output_file_index, edge_tangent_field);
  stk_io_broker.add_field(output_file_index, edge_binormal_field);
  stk_io_broker.add_field(output_file_index, edge_length_field);
  stk_io_broker.add_field(output_file_index, elem_radius_field);
  stk_io_broker.add_field(output_file_index, elem_rest_length_field);
  stk_io_broker.add_field(output_file_index, elem_aabb_field);

  declare_and_initialize_sperm(bulk_data, centerline_twist_springs_part, boundary_sperm_part,
                               spherocylinder_segments_part,  //
                               run_config.num_sperm, run_config.num_nodes_per_sperm, run_config.sperm_radius,
                               run_config.sperm_initial_segment_length,
                               run_config.sperm_rest_segment_length,  //
                               node_coords_field, node_velocity_field, node_force_field, node_twist_field,
                               node_twist_velocity_field, node_twist_torque_field, node_archlength_field,
                               node_curvature_field, node_rest_curvature_field, node_radius_field,
                               node_sperm_id_field,                                            //
                               edge_orientation_field, edge_tangent_field, edge_length_field,  //
                               elem_radius_field, elem_rest_length_field);

  // At this point, the sperm have been declared. We can fetch the NGP mesh and fields.
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  NgpDoubleField ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  NgpDoubleField ngp_old_node_coords_field = stk::mesh::get_updated_ngp_field<double>(old_node_coords_field);
  NgpDoubleField ngp_node_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_velocity_field);
  NgpDoubleField ngp_old_node_velocity_field = stk::mesh::get_updated_ngp_field<double>(old_node_velocity_field);
  NgpDoubleField ngp_node_force_field = stk::mesh::get_updated_ngp_field<double>(node_force_field);
  NgpDoubleField ngp_node_twist_field = stk::mesh::get_updated_ngp_field<double>(node_twist_field);
  NgpDoubleField ngp_old_node_twist_field = stk::mesh::get_updated_ngp_field<double>(old_node_twist_field);
  NgpDoubleField ngp_node_twist_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_twist_velocity_field);
  NgpDoubleField ngp_old_node_twist_velocity_field =
      stk::mesh::get_updated_ngp_field<double>(old_node_twist_velocity_field);
  NgpDoubleField ngp_node_twist_torque_field = stk::mesh::get_updated_ngp_field<double>(node_twist_torque_field);
  NgpDoubleField ngp_node_curvature_field = stk::mesh::get_updated_ngp_field<double>(node_curvature_field);
  NgpDoubleField ngp_node_rest_curvature_field = stk::mesh::get_updated_ngp_field<double>(node_rest_curvature_field);
  NgpDoubleField ngp_node_rotation_gradient_field =
      stk::mesh::get_updated_ngp_field<double>(node_rotation_gradient_field);
  NgpDoubleField ngp_node_radius_field = stk::mesh::get_updated_ngp_field<double>(node_radius_field);
  NgpDoubleField ngp_node_archlength_field = stk::mesh::get_updated_ngp_field<double>(node_archlength_field);
  NgpIntField ngp_node_sperm_id_field = stk::mesh::get_updated_ngp_field<int>(node_sperm_id_field);
  NgpDoubleField ngp_edge_orientation_field = stk::mesh::get_updated_ngp_field<double>(edge_orientation_field);
  NgpDoubleField ngp_old_edge_orientation_field = stk::mesh::get_updated_ngp_field<double>(old_edge_orientation_field);
  NgpDoubleField ngp_edge_tangent_field = stk::mesh::get_updated_ngp_field<double>(edge_tangent_field);
  NgpDoubleField ngp_old_edge_tangent_field = stk::mesh::get_updated_ngp_field<double>(old_edge_tangent_field);
  NgpDoubleField ngp_edge_binormal_field = stk::mesh::get_updated_ngp_field<double>(edge_binormal_field);
  NgpDoubleField ngp_edge_length_field = stk::mesh::get_updated_ngp_field<double>(edge_length_field);
  NgpDoubleField ngp_elem_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_radius_field);
  NgpDoubleField ngp_elem_rest_length_field = stk::mesh::get_updated_ngp_field<double>(elem_rest_length_field);
  NgpDoubleField ngp_elem_aabb_field = stk::mesh::get_updated_ngp_field<double>(elem_aabb_field);
  NgpDoubleField ngp_elem_old_aabb_field = stk::mesh::get_updated_ngp_field<double>(elem_old_aabb_field);
  NgpDoubleField ngp_elem_aabb_disp_since_last_rebuild_field =
      stk::mesh::get_updated_ngp_field<double>(elem_aabb_disp_since_last_rebuild_field);

  double current_time = 0.0;
  propagate_rest_curvature(ngp_mesh, current_time, run_config.amplitude, run_config.spatial_wavelength,
                           run_config.temporal_wavelength, centerline_twist_springs_part, ngp_node_archlength_field,
                           ngp_node_sperm_id_field, ngp_node_rest_curvature_field);

  // Equilibriate the system
  // equilibriate();

  // Time loop
  print_rank0(std::string("Running the simulation for ") + std::to_string(run_config.num_time_steps) + " time steps.");
  bool rebuild_neighbors = true;
  ResultViewType search_results;
  SearchBoxesViewType search_aabbs;

  Kokkos::Timer timer;
  for (size_t timestep_index = 0; timestep_index < run_config.num_time_steps; timestep_index++) {
    current_time = static_cast<double>(timestep_index) * run_config.timestep_size;

    if (timestep_index % 1000 == 0) {
      std::cout << "Time step " << timestep_index << " of " << run_config.num_time_steps << std::endl;
    }

    // Check if we need to recreate the neighbors
    mesh::field_copy<double>(elem_aabb_field, elem_old_aabb_field, stk::ngp::ExecSpace{});
    compute_aabbs(ngp_mesh, spherocylinder_segments_part, ngp_node_coords_field, ngp_elem_radius_field,
                  ngp_elem_aabb_field);
    mesh::field_axpbygz(1.0, elem_aabb_field, -1.0, elem_old_aabb_field, 1.0, elem_aabb_disp_since_last_rebuild_field,
                        stk::ngp::ExecSpace{});
    const double max_abs_disp =
        mesh::field_amax<double>(elem_aabb_disp_since_last_rebuild_field, stk::ngp::ExecSpace{});
    if (max_abs_disp > run_config.search_buffer) {
      rebuild_neighbors = true;
    }

    if (rebuild_neighbors) {
      std::cout << "Rebuilding neighbors." << std::endl;

      geom::PeriodicMetricYZ<double> periodic_metric(run_config.domain_width, run_config.domain_height);
      auto [target_search_aabbs, source_search_aabbs] =
          create_search_aabbs(bulk_data, ngp_mesh, run_config.search_buffer, periodic_metric,
                              spherocylinder_segments_part, ngp_elem_aabb_field);

      stk::search::SearchMethod search_method = stk::search::MORTON_LBVH;
      const bool results_parallel_symmetry = true;   // create source -> target and target -> source pairs
      const bool auto_swap_domain_and_range = true;  // swap source and target if target is owned and source is not
      const bool sort_search_results = false;        // sort the search results by source id
      stk::search::coarse_search(source_search_aabbs, target_search_aabbs, search_method, bulk_data.parallel(),
                                 search_results, stk::ngp::ExecSpace{}, results_parallel_symmetry);

      std::cout << "Search results size: " << search_results.size() << std::endl;
      rebuild_neighbors = false;
      mesh::field_fill(0.0, elem_aabb_disp_since_last_rebuild_field, stk::ngp::ExecSpace{});
    }

    // Prepare the current configuration.
    {
      // Apply constraints before we move the nodes.
      disable_twist(ngp_mesh, ngp_node_twist_field, ngp_node_twist_velocity_field);
      apply_monolayer(ngp_mesh, centerline_twist_springs_part, ngp_node_coords_field, ngp_node_velocity_field);

      // Rotate the field states. Use a deep copy to update the old fields.
      deep_copy<double, 3>(ngp_mesh, ngp_old_node_coords_field, ngp_node_coords_field, universal_part);
      deep_copy<double, 3>(ngp_mesh, ngp_old_node_velocity_field, ngp_node_velocity_field, universal_part);
      deep_copy<double, 1>(ngp_mesh, ngp_old_node_twist_field, ngp_node_twist_field, universal_part);
      deep_copy<double, 1>(ngp_mesh, ngp_old_node_twist_velocity_field, ngp_node_twist_velocity_field, universal_part);
      deep_copy<double, 4>(ngp_mesh, ngp_old_edge_orientation_field, ngp_edge_orientation_field, universal_part);
      deep_copy<double, 3>(ngp_mesh, ngp_old_edge_tangent_field, ngp_edge_tangent_field, universal_part);

      // Move the nodes from t -> t + dt.
      //   x(t + dt) = x(t) + dt v(t)
      update_generalized_position(ngp_mesh, run_config.timestep_size, centerline_twist_springs_part,  //
                                  ngp_old_node_coords_field, ngp_old_node_twist_field, ngp_old_node_velocity_field,
                                  ngp_old_node_twist_velocity_field,  //
                                  ngp_node_coords_field, ngp_node_twist_field);

      // Reset the fields in the current timestep.
      ngp_node_velocity_field.sync_to_device();
      ngp_node_force_field.sync_to_device();
      ngp_node_twist_velocity_field.sync_to_device();
      ngp_node_twist_torque_field.sync_to_device();

      ngp_node_velocity_field.set_all(ngp_mesh, 0.0);
      ngp_node_force_field.set_all(ngp_mesh, 0.0);
      ngp_node_twist_velocity_field.set_all(ngp_mesh, 0.0);
      ngp_node_twist_torque_field.set_all(ngp_mesh, 0.0);

      ngp_node_velocity_field.modify_on_device();
      ngp_node_force_field.modify_on_device();
      ngp_node_twist_velocity_field.modify_on_device();
      ngp_node_twist_torque_field.modify_on_device();
    }

    // Evaluate forces f(x(t + dt)).
    {
      // Hertzian contact force
      compute_hertzian_contact_force_and_torque(bulk_data, ngp_mesh, run_config.sperm_youngs_modulus,
                                                run_config.sperm_poissons_ratio, spherocylinder_segments_part,
                                                search_results, ngp_node_coords_field, ngp_elem_radius_field,
                                                ngp_node_force_field);

      // Centerline twist rod forces

      if (timestep_index > 1000000) {
        auto corrected_time = current_time - 1000000 * run_config.timestep_size;
        propagate_rest_curvature(ngp_mesh, corrected_time, run_config.amplitude, run_config.spatial_wavelength,
                                 run_config.temporal_wavelength,  //
                                 centerline_twist_springs_part, ngp_node_archlength_field, ngp_node_sperm_id_field,
                                 ngp_node_rest_curvature_field);
      }

      compute_edge_information(ngp_mesh, centerline_twist_springs_part,  //
                               ngp_node_coords_field, ngp_node_twist_field, ngp_edge_orientation_field,
                               ngp_old_edge_orientation_field, ngp_edge_tangent_field, ngp_old_edge_tangent_field,
                               ngp_edge_binormal_field, ngp_edge_length_field);

      compute_node_curvature_and_rotation_gradient(ngp_mesh, centerline_twist_springs_part,  //
                                                   ngp_edge_orientation_field, ngp_node_curvature_field,
                                                   ngp_node_rotation_gradient_field);

      compute_internal_force_and_twist_torque(
          ngp_mesh, run_config.sperm_rest_segment_length, run_config.sperm_youngs_modulus,
          run_config.sperm_poissons_ratio,  //
          centerline_twist_springs_part, ngp_node_radius_field, ngp_node_curvature_field, ngp_node_rest_curvature_field,
          ngp_node_rotation_gradient_field, ngp_edge_tangent_field, ngp_edge_binormal_field, ngp_edge_length_field,
          ngp_edge_orientation_field, ngp_node_force_field, ngp_node_twist_torque_field);
    }

    // Compute velocity v(x(t+dt))
    {
      // Compute the current velocity from the current forces.
      compute_generalized_velocity(ngp_mesh, run_config.viscosity, spherocylinder_segments_part,  //
                                   ngp_node_radius_field, ngp_node_force_field, ngp_node_twist_torque_field,
                                   ngp_node_velocity_field, ngp_node_twist_velocity_field);
    }

    // IO. If desired, write out the data for time t.
    if (timestep_index % run_config.io_frequency == 0) {
      stk::mesh::ngp_field_fence(meta_data);

      if (bulk_data.parallel_rank() == 0) {
        double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(timestep_index);
        std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
      }

      // Sync every io field to the host
      ngp_node_coords_field.sync_to_host();
      ngp_node_velocity_field.sync_to_host();
      ngp_node_force_field.sync_to_host();
      ngp_node_twist_field.sync_to_host();
      ngp_node_twist_velocity_field.sync_to_host();
      ngp_node_twist_torque_field.sync_to_host();
      ngp_node_curvature_field.sync_to_host();
      ngp_node_rest_curvature_field.sync_to_host();
      ngp_node_rotation_gradient_field.sync_to_host();
      ngp_node_radius_field.sync_to_host();
      ngp_node_archlength_field.sync_to_host();
      ngp_node_sperm_id_field.sync_to_host();
      ngp_edge_orientation_field.sync_to_host();
      ngp_edge_tangent_field.sync_to_host();
      ngp_edge_binormal_field.sync_to_host();
      ngp_edge_length_field.sync_to_host();
      ngp_elem_radius_field.sync_to_host();
      ngp_elem_rest_length_field.sync_to_host();

      stk_io_broker.begin_output_step(output_file_index, static_cast<double>(timestep_index));
      stk_io_broker.write_defined_output_fields(output_file_index);
      stk_io_broker.end_output_step(output_file_index);
      stk_io_broker.flush_output();
    }
  }

  // Do a synchronize to force everybody to stop here, then write the time
  stk::parallel_machine_barrier(bulk_data.parallel());
  if (bulk_data.parallel_rank() == 0) {
    double avg_time_per_timestep =
        static_cast<double>(timer.seconds()) / static_cast<double>(run_config.num_time_steps);
    std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
  }
}

}  // namespace mundy

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  // Run the simulation using the given parameters
  mundy::run(argc, argv);

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
