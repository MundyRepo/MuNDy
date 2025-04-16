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

Our goal is to simulate a chain of N Brownian diffusing spheres in a 3D domain. Sphere-sphere collision will be modeled
via Hertzian contact. Neighboring spheres in the chain will be connected by Hookean springs. Each sphere will be
attached to a crosslinker that can bind to (and unbind from) other nearby spheres. The crosslinker is also modeled as a
hookean spring, although it will only induce force when doubly bound.


Crosslinkers have the following states: unbound, left bound, right bound, and doubly bound.
Associated with these states are the following rates: left/right binding/unbinding rates.

Given the current state of the crosslinker, we have to use these rates to decide which state the crosslinker will
transition to next.

              0------------------------------------Normalized probability-----------------------------------------1
Unbound:      |--prob of becoming left bound--|--prob of becoming right bound--|--prob of remaining unbound-------|
Left bound:   |--prob of unbinding left node--|--prob of binding right node----|--prob of remaining left bound----|
Right bound:  |--prob of binding left node----|--prob of unbinding right node--|--prob of remaining right bound---|
Doubly bound: |--prob of unbinding left node--|--prob of unbinding right node--|--prob of remaining doubly bound--|

Parts: spheres, spherocylinders, hookean_springs, crosslinkers, doubly_bound_crosslinkers left_bound_crosslinkers,
right_bound_crosslinkers, unbound_crosslinkers doubly_bound_crosslinkers, left_bound_crosslinkers,
right_bound_crosslinkers, and unbound_crosslinkers will be subparts of crosslinkers, which is, in turn a subpart of
hookean_springs. For now, we add crosslinkers as a subpart of spherocylinders. This is a temporary workaround until we
implement a planned refactor of the agent hierarchy and merely serves to give our crosslinkers access to shape-based
methods. The element radius should be the crosslinker search radius. In the long term, we'll still use the same
shape-based methods but the crosslinker radius will be stored in SEARCH_RADIUS and the crosslinkers won't need to be a
subpart of spherocylinders.


Nearby spheres are connected by sphere-sphere neighbor linkers.
Adjacent spheres in the chain are connected by hookean springs.
Attached to each sphere is
               __________________________________________
              /    ____________________________________  \
             /    /                                    \  \
CONSTRAINT  |    |  neighbor_linker1          ___neighbor_linker2___
            |    | /  |    |    |   \        /           |           \
ELEMENT     | sphere1 | spring1 | sphere2   / left_bound_crosslinker1 \
            |    |    |   / \   |    |     / /                       \ \
FACE        |    |    |  /   \  |    |    / /                         \ \
            |    |    | /     \ |    |   / /                           \ \
EDGE         \   |   / /       \ \   |  / /                             \ \
              \  |  / /         \ \  | / /                               \ \
NODE           node1               node2                                 node3

      |
      x
      |
o-----o------o

NeighborLinkers are designed such that the lower rank connections (nodes, edges, faces) of neighbors are shared between
processes. This means that a process that has access to one of the spheres will have (at the very least) ghosted access
to the neighboring spheres and shared access to their nodes, making it perfectly valid for a crosslinker to fetch the
constraint rank entities attached to its node... Using the node to fetch the neighbor linker is nice but it does lose
information about connectivity order. This information can be determined


If a crosslinker is doubly bound, how do we decide which of the two sides to unbind? They both unbind at independent
rates.

If a crosslinker completely unbinds from both nodes, it will be deleted, as all entities must connect to nodes.

If some crosslinkers are to float around in space without being connected to the nodes of another element, then we
should replace the lost nodes of a crosslinker when they unbind. This, in turn, means that we cannot rely on the lack of
a connected node to determine the bound state of a crosslinker. Instead, we can break the singly bound crosslinkers part
into left bound and right bound. This makes logical sense, as these two types of crosslinkers have distinct behavior.

Can a crosslinker attached to a sphere along the backbone attach to one of the adjacent spheres?

Crosslinkers needs to store a rate for which their left and right nodes bind and unbind. This can differ from
crosslinker-to-crosslinker. The crosslinkers part itself con't be used to unbind the left/right nodes, since the
existence of connected nodes doesn't indicate that the crosslinker is bound to another object at that point. We won't,
however, restrict the left/right unbind rate to the respective singly bound parts, as external factors may change these
rates independent of the current bind status.

Unbinding the left node of a crosslinker can be done by looping over all left bound and doubly bound crosslinkers. The
same works for right nodes. Binding is harder. I don't think the correct solution is to rely on the
sphere-sphere neighbor list to determine if a singly bound crosslinker attached to a sphere binds to one of the
neighbors of said sphere. Separation of concerns seems to demand that we introduce crosslinker sphere neighbor linkers.
Actually, without crosslinker sphere linkers how would we handle the binding of unbound crosslinkers?


What is the easiest way to exclude certain types of neighbors? If a crosslinker is connected to a sphere, delete the
linker from the crosslinker to that sphere. If a crosslinker connected to a sphere, delete the linker from the
crosslinker to the adjacent spheres.





We need to modify our existing spring constraint forcing functions to account for the possibility that some springs will
lack some lower rank entities. Basically, if num connected nodes is less than the max amount, continue.

Order of operations:

// Preprocess
- Parse user parameter

// Setup
- Declare the fixed and mutable params for the desired MetaMethods
- Construct the mesh and the method instances
- Declare and initialize the chain of spheres, their connecting springs, and the crosslinkers along their backbone

// Timeloop
- Run the timeloop for t in range(0, T):
    // IO.
    - If desired, write out the data

    // Setup the current configuration. Think of this step as computing values for t + dt
    - Zero the node forces and velocities

    // Neighbor detection sphere-sphere
    - Check if the sphere-sphere neighbor list needs updated or not
        - Compute the AABB for the spheres and the neighbor linkers
        - Delete sphere-sphere and crosslinker-sphere neighbor linkers that are too far apart
        - Generate neighbor linkers between nearby spheres
        - Compute the signed separation distance and contact normal between neighboring spheres

    // Neighbor detection sphere-crosslinker

    // Pairwise potentials
    - Evaluate the Hertzian contact potential between neighboring spheres
    - Sum the linker potential force to get the induced node force on each sphere

    // Motion
    - Compute the velocity induced by the node forces using local drag
    - Compute the brownian velocity for the nodes
    - Apply velocity/position constraints like no motion for particle 1
    - Update the node positions using a first order Euler method
*/

// External libs
#include <fmt/format.h>  // for fmt::format
#include <openrand/philox.h>

// Trilinos libs
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Comm.hpp>            // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>   // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>            // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>        // for stk::mesh::Selector
#include <stk_topology/topology.hpp>         // for stk::topology
#include <stk_util/parallel/Parallel.hpp>    // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_alens/actions_crosslinkers.hpp>             // for mundy::alens::crosslinkers...
#include <mundy_constraints/AngularSprings.hpp>             // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#include <mundy_core/MakeStringArray.hpp>                   // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_io/IOBroker.hpp>         // for mundy::io::IOBroker
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>         // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>       // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>        // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                // for mundy::linkers::NeighborLinkers
#include <mundy_math/Vector3.hpp>                           // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>  // adds fmt::format for stk types
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>        // for mundy::mesh::utils::destroy_flagged_entities
#include <mundy_mesh/utils/FillFieldWithValue.hpp>            // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

namespace mundy {

namespace alens {

namespace crosslinkers {

class StickySettings {
 public:
  enum BINDING_STATE_CHANGE : unsigned { NONE = 0u, LEFT_TO_DOUBLY, RIGHT_TO_DOUBLY, DOUBLY_TO_LEFT, DOUBLY_TO_RIGHT };
  enum BOND_TYPE : unsigned { HARMONIC = 0u, FENE };

  StickySettings() = default;

  void print_rank0(auto thing_to_print, int indent_level = 0) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::string indent(indent_level * 2, ' ');
      std::cout << indent << thing_to_print << std::endl;
    }
  }

  void parse_user_inputs(int argc, char **argv) {
    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, true);
    std::string crosslinker_spring_type;

    //   Backbone initialization:
    cmdp.setOption("num_spheres", &num_spheres_, "Number of spheres in backbone.");
    cmdp.setOption("sphere_radius", &sphere_radius_, "Backbone sphere radius.");
    cmdp.setOption("initial_sphere_separation", &initial_sphere_separation_, "Initial backbone sphere separation.");
    cmdp.setOption("sphere_youngs_modulus", &sphere_youngs_modulus_, "Backbone sphere Youngs modulus.");
    cmdp.setOption("sphere_poissons_ratio", &sphere_poissons_ratio_, "Backbone sphere poissons ratio.");
    cmdp.setOption("sphere_drag_coeff", &sphere_drag_coeff_, "Backbone sphere drag coefficient.");

    //   Backbone spring:
    cmdp.setOption("backbone_spring_constant", &backbone_spring_constant_, "Backbone spring constant.");
    cmdp.setOption("backbone_spring_rest_length", &backbone_spring_rest_length_, "Backbone rest length.");

    //   Crosslinker (spring and other):
    cmdp.setOption("crosslinker_spring_type", &crosslinker_spring_type, "Crosslinker spring type.");
    cmdp.setOption("crosslinker_spring_constant", &crosslinker_spring_constant_, "Crosslinker spring constant.");
    cmdp.setOption("crosslinker_rest_length", &crosslinker_rest_length_, "Crosslinker rest length.");
    cmdp.setOption("crosslinker_left_binding_rate", &crosslinker_left_binding_rate_, "Crosslinker left binding rate.");
    cmdp.setOption("crosslinker_right_binding_rate", &crosslinker_right_binding_rate_,
                   "Crosslinker right binding rate.");
    cmdp.setOption("crosslinker_left_unbinding_rate", &crosslinker_left_unbinding_rate_,
                   "Crosslinker left unbinding rate.");
    cmdp.setOption("crosslinker_right_unbinding_rate", &crosslinker_right_unbinding_rate_,
                   "Crosslinker right unbinding rate.");

    // Neighbor list
    cmdp.setOption("skin_distance", &skin_distance_, "Neighbor list skin distance.");
    cmdp.setOption("force_neighborlist_update", "no_force_update_neighbor_list", &force_neighborlist_update_,
                   "Force update of the neighbor list.");
    cmdp.setOption("force_neighborlist_update_nsteps", &force_neighborlist_update_nsteps_,
                   "Number of timesteps between updating the neighbor list (if forced).");
    cmdp.setOption("print_neighborlist_statistics", "no_print_neighborlist_statistics", &print_neighborlist_statistics_,
                   "Print neighbor list statistics.");

    //   The simulation:
    cmdp.setOption("num_time_steps", &num_time_steps_, "Number of time steps.");
    cmdp.setOption("timestep_size", &timestep_size_, "Time step size.");
    cmdp.setOption("kt_brownian", &kt_brownian_, "Temperature kT for Brownian Motion.");
    cmdp.setOption("kt_kmc", &kt_kmc_, "Temperature kT for KMC.");
    cmdp.setOption("io_frequency", &io_frequency_, "Number of timesteps between writing output.");
    cmdp.setOption("initial_loadbalance", "no_initial_loadbalance", &initial_loadbalance_, "Initial loadbalance.");

    bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    MUNDY_THROW_REQUIRE(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");
    MUNDY_THROW_REQUIRE(num_spheres_ > 0, std::invalid_argument, "num_spheres_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(sphere_radius_ > 0, std::invalid_argument, "sphere_radius_ must be greater than 0.");

    MUNDY_THROW_REQUIRE(num_time_steps_ > 0, std::invalid_argument, "num_time_steps_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(io_frequency_ > 0, std::invalid_argument, "io_frequency_ must be greater than 0.");

    // Modify any variables into their final form
    skin_distance2_over4_ = skin_distance_ * skin_distance_ / 4.0;
    // Compute the cutoff radius for the crosslinker
    crosslinker_rcut_ = crosslinker_rest_length_ + 5.0 * std::sqrt(1.0 / kt_kmc_ / crosslinker_spring_constant_);
    // Set the type of springs we are using (hookean or fene)
    if (crosslinker_spring_type == "harmonic") {
      crosslinker_spring_type_ = BOND_TYPE::HARMONIC;
    } else if (crosslinker_spring_type == "fene") {
      crosslinker_spring_type_ = BOND_TYPE::FENE;
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Invalid crosslinker spring type.");
    }
  }

  void dump_user_inputs() {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;

      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  num_time_steps: " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size: " << timestep_size_ << std::endl;
      std::cout << "  io_frequency: " << io_frequency_ << std::endl;
      std::cout << "  kT (Brownian): " << kt_brownian_ << std::endl;
      std::cout << "  kT (KMC): " << kt_kmc_ << std::endl;

      std::cout << "NEIGHBOR LIST:" << std::endl;
      std::cout << "  force_update_neighborlist: " << force_neighborlist_update_ << std::endl;
      std::cout << "  force_update_neighborlist_nsteps: " << force_neighborlist_update_nsteps_ << std::endl;
      std::cout << "  skin_distance: " << 2.0 * std::sqrt(skin_distance2_over4_) << std::endl;
      std::cout << "    (skin_distance2_over4): " << skin_distance2_over4_ << std::endl;

      std::cout << "BACKBONE SPHERES:" << std::endl;
      std::cout << "  num_spheres: " << num_spheres_ << std::endl;
      std::cout << "  sphere_radius: " << sphere_radius_ << std::endl;
      std::cout << "  initial_sphere_separation: " << initial_sphere_separation_ << std::endl;
      std::cout << "  youngs_modulus: " << sphere_youngs_modulus_ << std::endl;
      std::cout << "  poissons_ratio: " << sphere_poissons_ratio_ << std::endl;
      std::cout << "  drag_coeff: " << sphere_drag_coeff_ << std::endl;

      std::cout << "BACKBONE SPRINGS:" << std::endl;
      std::cout << "  backbone_spring_constant: " << backbone_spring_constant_ << std::endl;
      std::cout << "  backbone_spring_rest_length: " << backbone_spring_rest_length_ << std::endl;

      std::string crosslinker_spring_type = crosslinker_spring_type_ == BOND_TYPE::HARMONIC ? "harmonic" : "fene";
      std::cout << "CROSSLINKERS:" << std::endl;
      std::cout << "  crosslinker_spring_type: " << crosslinker_spring_type << std::endl;
      std::cout << "  crosslinker_spring_constant: " << crosslinker_spring_constant_ << std::endl;
      std::cout << "  crosslinker_rest_length: " << crosslinker_rest_length_ << std::endl;
      std::cout << "  crosslinker_rcut: " << crosslinker_rcut_ << std::endl;
      std::cout << "  crosslinker_left_binding_rate: " << crosslinker_left_binding_rate_ << std::endl;
      std::cout << "  crosslinker_right_binding_rate: " << crosslinker_right_binding_rate_ << std::endl;
      std::cout << "  crosslinker_left_unbinding_rate: " << crosslinker_left_unbinding_rate_ << std::endl;
      std::cout << "  crosslinker_right_unbinding_rate: " << crosslinker_right_binding_rate_ << std::endl;
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

  void assert_invariant(const std::string &message = std::string()) {
    Kokkos::Profiling::pushRegion("StickySettings::assert_invariant");
    stk::mesh::Part &left_bound_crosslinkers_part = *left_bound_crosslinkers_part_ptr_;
    stk::mesh::Part &doubly_bound_crosslinkers_part = *doubly_bound_crosslinkers_part_ptr_;

    auto left_crosslinkers_selector =
        stk::mesh::Selector(left_bound_crosslinkers_part) & meta_data_ptr_->locally_owned_part();
    auto doubly_crosslinkers_selector =
        stk::mesh::Selector(doubly_bound_crosslinkers_part) & meta_data_ptr_->locally_owned_part();
    // std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Num left bound crosslinkers: "
    //           << stk::mesh::count_selected_entities(left_crosslinkers_selector,
    //           bulk_data_ptr_->buckets(element_rank_))
    //           << std::endl;
    // std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Num doubly bound crosslinkers: "
    //           << stk::mesh::count_selected_entities(doubly_crosslinkers_selector,
    //                                                 bulk_data_ptr_->buckets(element_rank_))
    //           << std::endl;

    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, left_crosslinkers_selector,
        [&message, &left_bound_crosslinkers_part, &doubly_bound_crosslinkers_part](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          auto print_bucket = [](const stk::mesh::Bucket &bucket) {
            std::ostringstream out;
            out << "    bucket " << bucket.bucket_id() << " parts: { ";
            const stk::mesh::PartVector &supersets = bucket.supersets();
            for (const stk::mesh::Part *part : supersets) {
              out << "(" << part->mesh_meta_data_ordinal() << "," << part->name() << ") ";
            }
            out << "}" << std::endl;
            return out.str();
          };

          MUNDY_THROW_ASSERT(
              bulk_data.bucket(crosslinker).member(left_bound_crosslinkers_part.mesh_meta_data_ordinal()),
              std::logic_error,
              std::string("The crosslinker is not a left bound crosslinker.\n") + message +
                  print_bucket(bulk_data.bucket(crosslinker)));
          MUNDY_THROW_ASSERT(
              !bulk_data.bucket(crosslinker).member(doubly_bound_crosslinkers_part.mesh_meta_data_ordinal()),
              std::logic_error,
              std::string("The crosslinker is somehow also a doubly bound crosslinker.\n") + message +
                  print_bucket(bulk_data.bucket(crosslinker)));
          const stk::mesh::Entity left_sphere_node = bulk_data.begin_nodes(crosslinker)[0];
          const stk::mesh::Entity right_sphere_node = bulk_data.begin_nodes(crosslinker)[1];

          // For left-bound crosslinkers, the right node should be the same as the left.
          MUNDY_THROW_ASSERT(bulk_data.is_valid(left_sphere_node), std::logic_error,
                             std::string("Left node is not valid.\n") + message);
          MUNDY_THROW_ASSERT(bulk_data.bucket(left_sphere_node).member(left_bound_crosslinkers_part), std::logic_error,
                             std::string("Left node is not a left bound crosslinker.\n") + message +
                                 print_bucket(bulk_data.bucket(crosslinker)));
          MUNDY_THROW_ASSERT(left_sphere_node == right_sphere_node, std::logic_error,
                             std::string("Left and right nodes are not the same.\n") + message);
        });

    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_crosslinkers_selector,
        [&message, &left_bound_crosslinkers_part, &doubly_bound_crosslinkers_part](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          MUNDY_THROW_ASSERT(bulk_data.bucket(crosslinker).member(doubly_bound_crosslinkers_part), std::logic_error,
                             std::string("The crosslinker is not a doubly bound crosslinker.\n") + message);
          MUNDY_THROW_ASSERT(!bulk_data.bucket(crosslinker).member(left_bound_crosslinkers_part), std::logic_error,
                             std::string("The crosslinker is somehow also a left bound crosslinker.\n") + message);

          const stk::mesh::Entity left_sphere_node = bulk_data.begin_nodes(crosslinker)[0];
          const stk::mesh::Entity right_sphere_node = bulk_data.begin_nodes(crosslinker)[1];
          const bool left_sphere_correct = bulk_data.bucket(left_sphere_node).member(doubly_bound_crosslinkers_part);
          const bool right_sphere_correct = bulk_data.bucket(right_sphere_node).member(doubly_bound_crosslinkers_part);
          MUNDY_THROW_ASSERT(bulk_data.is_valid(left_sphere_node), std::logic_error,
                             std::string("Left node is not valid.\n") + message);
          MUNDY_THROW_ASSERT(bulk_data.is_valid(right_sphere_node), std::logic_error,
                             std::string("Right node is not valid.\n") + message);
          MUNDY_THROW_ASSERT(left_sphere_correct, std::logic_error,
                             std::string("Left node is not a left bound crosslinker.\n") + message);
          MUNDY_THROW_ASSERT(right_sphere_correct, std::logic_error,
                             std::string("Right node is not a right bound crosslinker.\n") + message);
        });
    Kokkos::Profiling::popRegion();
  }

  void build_our_mesh_and_method_instances() {
    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshReqs>(MPI_COMM_WORLD);
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements to the sphere part for this example. These are requirements that exceed those of the
    // enabled methods and allow us to extend the functionality offered natively by Mundy.
    // Add these to the spheres part.
    auto custom_sphere_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_sphere_part_reqs->add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<unsigned>("NODE_RNG_COUNTER", node_rank_, 1, 1)
        .add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", node_rank_, 3, 1);
    mundy::shapes::Spheres::add_and_sync_part_reqs(custom_sphere_part_reqs);
    mesh_reqs_ptr_->sync(mundy::shapes::Spheres::get_mesh_requirements());

    // Build the crosslinkers part requirements based on the needs of our custom methods.
    // Add this part to the mesh directly
    auto custom_crosslinkers_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_crosslinkers_part_reqs->set_part_name("CROSSLINKERS")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ELEMENT_UNBINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<double>("ELEMENT_BINDING_RATES", element_rank_, 2, 1)
        .add_field_reqs<unsigned>("ELEMENT_RNG_COUNTER", element_rank_, 1, 1)
        .add_field_reqs<unsigned>("ELEMENT_PERFORM_STATE_CHANGE", element_rank_, 1, 1)
        .add_subpart_reqs("LEFT_BOUND_CROSSLINKERS", stk::topology::BEAM_2)
        .add_subpart_reqs("RIGHT_BOUND_CROSSLINKERS", stk::topology::BEAM_2)
        .add_subpart_reqs("DOUBLY_BOUND_CROSSLINKERS", stk::topology::BEAM_2);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_crosslinkers_part_reqs);

    // Create the generalized interaction entities that connect crosslinkers and spheres
    //   This entity "knows" how to compute the binding probability between a crosslinker and a sphere and how to
    //   perform binding between a crosslinker and a sphere. It is a constraint rank entitiy because itc must connect
    //   element rank entities.
    auto custom_crosslinker_sphere_linkers_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_crosslinker_sphere_linkers_part_reqs->set_part_name("CROSSLINKER_SPHERE_LINKERS")
        .set_part_rank(constraint_rank_)
        .add_field_reqs<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_, 1, 1)
        .add_field_reqs<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_, 1, 1);
    mundy::linkers::NeighborLinkers::add_and_sync_subpart_reqs(custom_crosslinker_sphere_linkers_part_reqs);
    mesh_reqs_ptr_->sync(mundy::linkers::NeighborLinkers::get_mesh_requirements());

    // Add the custom neighbor list implementation to the axis-aligned bounding box. In theory, we only care if the
    // corners of the AABB move more than skin distance/2. Set the AABB field to have an additional state that it keeps
    // track of at the end so that we can do dr calculations trivially. This create the two element aabb fields, but
    // also adds a second entire set of fields to the mesh.
    //
    // We also need an accumulator for summing up the total distance traveled by the AABB corners.
    mesh_reqs_ptr_->add_field_reqs<double>("ELEMENT_AABB", element_rank_, 6, 2);
    auto custom_sphere_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_sphere_aabb_accumulator_reqs->set_part_name("SPHERES")
        .set_part_topology(stk::topology::PARTICLE)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    auto custom_crosslinker_aabb_accumulator_reqs = std::make_shared<mundy::meta::PartReqs>();
    custom_crosslinker_aabb_accumulator_reqs->set_part_name("CROSSLINKERS")
        .set_part_topology(stk::topology::BEAM_2)
        .add_field_reqs<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_, 6, 1);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_sphere_aabb_accumulator_reqs);
    mesh_reqs_ptr_->add_and_sync_part_reqs(custom_crosslinker_aabb_accumulator_reqs);

    // Setup our fixed parameters for any of methods that we intend to use
    // When we eventually switch to the configurator, these individual fixed params will become sublists within a single
    // master parameter list. Note, sublist will return a reference to the sublist with the given name.
    compute_constraint_forcing_fixed_params_ =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRINGS"));
    compute_constraint_forcing_fixed_params_.sublist("HOOKEAN_SPRINGS")
        .set("valid_entity_part_names", mundy::core::make_string_array("HOOKEAN_SPRINGS", "CROSSLINKERS"));

    compute_ssd_and_cn_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names",
        mundy::core::make_string_array("SPHERE_SPHERE_LINKER", "SPHERE_SPHEROCYLINDER_SEGMENT_LINKER"));

    compute_aabb_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE", "SPHEROCYLINDER_SEGMENT"));
    compute_aabb_fixed_params_.sublist("SPHEROCYLINDER_SEGMENT")
        .set("valid_entity_part_names", mundy::core::make_string_array("CROSSLINKERS"));

    generate_sphere_sphere_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"));
    generate_sphere_sphere_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHERES"))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));

    generate_crosslinker_sphere_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("CROSSLINKER_SPHERE_LINKERS"));
    generate_crosslinker_sphere_neighbor_linkers_fixed_params_.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array(std::string("CROSSLINKERS")))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"));

    evaluate_linker_potentials_fixed_params_ = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHERE_SPHERE_HERTZIAN_CONTACT"));

    linker_potential_force_reduction_fixed_params_ =
        Teuchos::ParameterList()
            .set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
            .set("name_of_linker_part_to_reduce_over", "SPHERE_SPHERE_LINKERS");

    destroy_neighbor_linkers_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");
    destroy_neighbor_linkers_fixed_params_.sublist("DESTROY_DISTANT_NEIGHBORS")
        .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
        .set("valid_connected_source_and_target_part_names",
             mundy::core::make_string_array(std::string("SPHERES"), std::string("CROSSLINKERS")));

    declare_and_init_constraints_fixed_params_ =
        Teuchos::ParameterList().set("enabled_technique_name", "CHAIN_OF_SPRINGS");
    declare_and_init_constraints_fixed_params_.sublist("CHAIN_OF_SPRINGS")
        .set("hookean_springs_part_names", mundy::core::make_string_array("BACKBONE_SPRINGS"))
        .set("sphere_part_names", mundy::core::make_string_array("SPHERES"))
        .set<bool>("generate_hookean_springs", true)
        .set<bool>("generate_spheres_at_nodes", true);

    // Synchronize (merge and rectify differences) the requirements for each method based on the fixed parameters.
    // For now, we will directly use the types that each method corresponds to. The configurator will
    // fetch the static members of these methods using the configurable method factory.
    mesh_reqs_ptr_->sync(
        mundy::constraints::ComputeConstraintForcing::get_mesh_requirements(compute_constraint_forcing_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_sphere_sphere_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(
        generate_crosslinker_sphere_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::linkers::LinkerPotentialForceReduction::get_mesh_requirements(
        linker_potential_force_reduction_fixed_params_));
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_neighbor_linkers_fixed_params_));
    mesh_reqs_ptr_->sync(mundy::constraints::DeclareAndInitConstraints::get_mesh_requirements(
        declare_and_init_constraints_fixed_params_));

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->use_simple_fields();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDS");
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
    // Fetch the fields
    node_coord_field_ptr_ = fetch_field<double>("NODE_COORDS", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_rng_field_ptr_ = fetch_field<unsigned>("NODE_RNG_COUNTER", node_rank_);

    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    element_hookean_spring_constant_field_ptr_ = fetch_field<double>("ELEMENT_HOOKEAN_SPRING_CONSTANT", element_rank_);
    element_hookean_spring_rest_length_field_ptr_ =
        fetch_field<double>("ELEMENT_HOOKEAN_SPRING_REST_LENGTH", element_rank_);
    element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
    element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    element_rng_field_ptr_ = fetch_field<unsigned>("ELEMENT_RNG_COUNTER", element_rank_);
    element_binding_rates_field_ptr_ = fetch_field<double>("ELEMENT_BINDING_RATES", element_rank_);
    element_unbinding_rates_field_ptr_ = fetch_field<double>("ELEMENT_UNBINDING_RATES", element_rank_);
    element_perform_state_change_field_ptr_ = fetch_field<unsigned>("ELEMENT_PERFORM_STATE_CHANGE", element_rank_);
    element_corner_displacement_field_ptr_ = fetch_field<double>("ACCUMULATED_AABB_CORNER_DISPLACEMENT", element_rank_);
    element_aabb_field_ptr_ = fetch_field<double>("ELEMENT_AABB", element_rank_);

    linker_destroy_flag_field_ptr_ = fetch_field<int>("LINKER_DESTROY_FLAG", constraint_rank_);
    constraint_state_change_rate_field_ptr_ =
        fetch_field<double>("CONSTRAINT_STATE_CHANGE_PROBABILITY", constraint_rank_);
    constraint_perform_state_change_field_ptr_ =
        fetch_field<unsigned>("CONSTRAINT_PERFORM_STATE_CHANGE", constraint_rank_);
    constraint_linked_entities_field_ptr_ =
        fetch_field<mundy::linkers::LinkedEntitiesFieldType::value_type>("LINKED_NEIGHBOR_ENTITIES", constraint_rank_);
    constraint_linked_entity_owners_field_ptr_ = fetch_field<int>("LINKED_NEIGHBOR_ENTITY_OWNERS", constraint_rank_);

    // Fetch the parts
    spheres_part_ptr_ = fetch_part("SPHERES");
    crosslinkers_part_ptr_ = fetch_part("CROSSLINKERS");
    agents_part_ptr_ = fetch_part("AGENTS");
    springs_part_ptr_ = fetch_part("HOOKEAN_SPRINGS");
    sphere_sphere_linkers_part_ptr_ = fetch_part("SPHERE_SPHERE_LINKERS");
    crosslinker_sphere_linkers_part_ptr_ = fetch_part("CROSSLINKER_SPHERE_LINKERS");

    left_bound_crosslinkers_part_ptr_ = fetch_part("LEFT_BOUND_CROSSLINKERS");
    right_bound_crosslinkers_part_ptr_ = fetch_part("RIGHT_BOUND_CROSSLINKERS");
    doubly_bound_crosslinkers_part_ptr_ = fetch_part("DOUBLY_BOUND_CROSSLINKERS");
  }

  void instantiate_metamethods() {
    // Create the non-custom MetaMethods
    // MetaMethodExecutionInterface
    declare_and_init_constraints_ptr_ = mundy::constraints::DeclareAndInitConstraints::create_new_instance(
        bulk_data_ptr_.get(), declare_and_init_constraints_fixed_params_);

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

    // MetaMethodPairwiseSubsetExecutionInterface
    generate_sphere_sphere_neighbor_linkers_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_sphere_sphere_neighbor_linkers_fixed_params_);
    generate_crosslinker_sphere_neighbor_linkers_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_crosslinker_sphere_neighbor_linkers_fixed_params_);
  }

  void set_mutable_parameters() {
    double skin_distance = 2.0 * std::sqrt(skin_distance2_over4_);
    // ComputeAABB mutable parameters
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", skin_distance);
    compute_aabb_ptr_->set_mutable_params(compute_aabb_mutable_params);

    // DeclareAndInitConstraints mutable parameters
    Teuchos::ParameterList declare_and_init_constraints_mutable_params;
    using CoordinateMappingType =
        mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
    using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::StraightLine;

    double center_x = 0.0;
    double center_y = 0.0;
    double center_z = 0.0;
    double orientation_x = 1.0;
    double orientation_y = 0.0;
    double orientation_z = 0.0;
    auto coord_mapping_ptr =
        std::make_shared<OurCoordinateMappingType>(num_spheres_, center_x, center_y, center_z,
                                                   (static_cast<double>(num_spheres_) - 1) * initial_sphere_separation_,
                                                   orientation_x, orientation_y, orientation_z);
    declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
        .set<size_t>("num_nodes", num_spheres_)
        .set<size_t>("node_id_start", 1u)
        .set<size_t>("element_id_start", 1u)
        .set("hookean_spring_constant", backbone_spring_constant_)
        .set("hookean_spring_rest_length", backbone_spring_rest_length_)
        .set("sphere_radius", sphere_radius_)
        .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", coord_mapping_ptr);
    declare_and_init_constraints_ptr_->set_mutable_params(declare_and_init_constraints_mutable_params);
  }

  void setup_io_mundy() {
    // Create a mundy io broker via it's fixed parameters
    // Dump everything for now
    auto fixed_params_iobroker =
        Teuchos::ParameterList()
            .set("enabled_io_parts",
                 mundy::core::make_string_array("SPHERES", "BACKBONE_SPRINGS", "LEFT_BOUND_CROSSLINKERS",
                                                "RIGHT_BOUND_CROSSLINKERS", "DOUBLY_BOUND_CROSSLINKERS"))
            .set("enabled_io_fields_node_rank",
                 mundy::core::make_string_array("NODE_VELOCITY", "NODE_FORCE", "NODE_RNG_COUNTER"))
            .set("enabled_io_fields_element_rank",
                 mundy::core::make_string_array(
                     "ELEMENT_RADIUS", "ELEMENT_HOOKEAN_SPRING_CONSTANT", "ELEMENT_HOOKEAN_SPRING_REST_LENGTH",
                     "ELEMENT_YOUNGS_MODULUS", "ELEMENT_POISSONS_RATIO", "ELEMENT_RNG_COUNTER", "ELEMENT_BINDING_RATES",
                     "ELEMENT_UNBINDING_RATES", "ELEMENT_PERFORM_STATE_CHANGE"))
            .set("coordinate_field_name", "NODE_COORDS")
            .set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES")
            .set("exodus_database_output_filename", "Sticky.exo")
            .set("parallel_io_mode", "hdf5")
            .set("database_purpose", "results");
    // Create the IO broker
    io_broker_ptr_ = mundy::io::IOBroker::create_new_instance(bulk_data_ptr_.get(), fixed_params_iobroker);
  }

  void ghost_linked_entities() {
    bulk_data_ptr_->modification_begin();
    const stk::mesh::Selector linker_parts_selector = stk::mesh::selectUnion(
        stk::mesh::ConstPartVector{sphere_sphere_linkers_part_ptr_, crosslinker_sphere_linkers_part_ptr_});
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

  void declare_and_initialize_sticky() {
    //////////////////////////////////////
    // Initialize the spheres and nodes //
    //////////////////////////////////////
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;

    // Initialize the backbone and its springs
    declare_and_init_constraints_ptr_->execute();

    // Initialize the parameters on each particle needed for the Hertzian contacts. We need to initialize the poisson
    // ratio and youngs modulus on every sphere as well. Scope this so that things dont leak out.
    {
      // Set up some aliases and the selector
      stk::mesh::Part &spheres_part = *spheres_part_ptr_;
      const stk::mesh::Field<double> &element_youngs_modulus_field = *element_youngs_modulus_field_ptr_;
      const stk::mesh::Field<double> &element_poissons_ratio_field = *element_poissons_ratio_field_ptr_;
      double &youngs_modulus = sphere_youngs_modulus_;
      double &poissons_ratio = sphere_poissons_ratio_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_part,
          [&element_youngs_modulus_field, &element_poissons_ratio_field, &youngs_modulus, &poissons_ratio](
              [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere) {
            // const stk::mesh::Entity &node = bulk_data.begin_nodes(sphere)[0];

            // stk::mesh::field_data(node_rng_counter_field, node)[0] = 0;
            stk::mesh::field_data(element_youngs_modulus_field, sphere)[0] = youngs_modulus;
            stk::mesh::field_data(element_poissons_ratio_field, sphere)[0] = poissons_ratio;
          });  // for_each_entity_run
    }

    // Declare the crosslinkers along the backbone
    // Every sphere gets a left bound crosslinker
    //  o : spheres
    //  | : crosslinkers
    // ---: backbone springs
    //
    //  |   |   |   |   |   |   |   |   |   |
    //  o---o---o---o---o---o---o---o---o---o
    //
    // One of the design features that we're working on is better composition/extension for declaring and initializing
    // entities. The following will be replaced with a more elegant solution in the future that enforcing a certain
    // structure on the chain of springs. Specifically, we want to enforce the node ordering and partitioning to be the
    // following:
    //
    // o---o---o---o---o---o---o---o---o---o
    // 1   2   3   4   5   6   7   8   9   10 <- node ids
    // <--- rank 0 ---> <--- rank 1 ---> <--- rank 2 --->
    //
    // What follows is the same decomposition strategy used by the DeclareAndInitConstraints' ChainOfSprings technique.
    // To enforce consistency, we should make ChainOfSpheres take in a decomposition strategy (or something along those
    // lines) such that we can use the same strategy here when declaring the crosslinkers.

    // Because we are creating multiple sperm, we need to determine the node and element index ranges for each sperm.
    size_t start_node_id = 1u;

    // Shift by number of spheres, then numbers of spheres-1 to get through the already declared hookean springs for the
    // backbone.
    size_t start_crosslinker_id = num_spheres_ + num_spheres_ - 1u + 1u;

    auto get_node_id = [start_node_id](const size_t &seq_node_index) { return start_node_id + seq_node_index; };

    [[maybe_unused]] auto get_node = [get_node_id, &bulk_data](const size_t &seq_node_index) {
      return bulk_data.get_entity(stk::topology::NODE_RANK, get_node_id(seq_node_index));
    };

    auto get_crosslinker_id = [start_crosslinker_id](const size_t &seq_crosslinker_index) {
      return start_crosslinker_id + seq_crosslinker_index;
    };

    [[maybe_unused]] auto get_crosslinker = [get_crosslinker_id, &bulk_data](const size_t &seq_crosslinker_index) {
      return bulk_data.get_entity(stk::topology::ELEMENT_RANK, get_crosslinker_id(seq_crosslinker_index));
    };

    // Create the springs and their connected nodes, distributing the work across the ranks.
    const size_t rank = bulk_data_ptr_->parallel_rank();
    const size_t nodes_per_rank = num_spheres_ / bulk_data_ptr_->parallel_size();
    const size_t remainder = num_spheres_ % bulk_data_ptr_->parallel_size();
    const size_t start_seq_node_index = rank * nodes_per_rank + std::min(rank, remainder);
    const size_t end_seq_node_index = start_seq_node_index + nodes_per_rank + (rank < remainder ? 1 : 0);

    bulk_data_ptr_->modification_begin();
    // Temporary/scatch variables
    stk::mesh::PartVector empty;
    stk::mesh::Permutation invalid_perm = stk::mesh::Permutation::INVALID_PERMUTATION;
    stk::mesh::OrdinalVector scratch1, scratch2, scratch3;
    auto left_bound_crosslinker_part_vector = stk::mesh::PartVector{left_bound_crosslinkers_part_ptr_};

    // Create the elements for the crosslinkers
    const size_t start_element_chain_index = start_seq_node_index;
    const size_t end_start_element_chain_index = end_seq_node_index;
    for (size_t i = start_element_chain_index; i < end_start_element_chain_index; ++i) {
      // Bind left and right nodes to the same node to start simulation (everybody is left bound)
      stk::mesh::EntityId left_node_id = get_node_id(i);
      stk::mesh::Entity left_node = bulk_data_ptr_->get_entity(node_rank_, left_node_id);
      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(left_node), std::invalid_argument,
                         fmt::format("Node {} is not valid.", i));

      // Fetch the centerline twist spring and connect it to the nodes/edges
      // Connect back onto the same node for now, as it is a left bound crosslinker
      stk::mesh::EntityId crosslinker_id = get_crosslinker_id(i);
      stk::mesh::Entity crosslinker =
          bulk_data_ptr_->declare_element(crosslinker_id, left_bound_crosslinker_part_vector);
      bulk_data_ptr_->declare_relation(crosslinker, left_node, 0, invalid_perm, scratch1, scratch2, scratch3);
      bulk_data_ptr_->declare_relation(crosslinker, left_node, 1, invalid_perm, scratch1, scratch2, scratch3);
      MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(crosslinker).topology() != stk::topology::INVALID_TOPOLOGY,
                         std::logic_error,
                         fmt::format("The crosslinker with id {} has an invalid topology.", crosslinker_id));

      // Set the crosslinker fields
      stk::mesh::field_data(*element_rng_field_ptr_, crosslinker)[0] = 0;
      stk::mesh::field_data(*element_hookean_spring_constant_field_ptr_, crosslinker)[0] = crosslinker_spring_constant_;
      stk::mesh::field_data(*element_hookean_spring_rest_length_field_ptr_, crosslinker)[0] = crosslinker_rest_length_;
      stk::mesh::field_data(*element_radius_field_ptr_, crosslinker)[0] =
          crosslinker_rcut_;  // This is the search radius. Note that the skin distance is included separately.
    }
    bulk_data_ptr_->modification_end();
  }

  void debug_print_meta_data() {
    std::cout << "############################################" << std::endl;
    const std::vector<std::string> &rank_names = meta_data_ptr_->entity_rank_names();
    for (size_t i = 0, e = rank_names.size(); i < e; ++i) {
      stk::mesh::EntityRank rank = static_cast<stk::mesh::EntityRank>(i);
      std::cout << "  All " << rank_names[i] << " entities:" << std::endl;

      const stk::mesh::BucketVector &buckets = bulk_data_ptr_->buckets(rank);
      for (stk::mesh::Bucket *bucket : buckets) {
        std::cout << "    bucket " << bucket->bucket_id() << " parts: { ";
        const stk::mesh::PartVector &supersets = bucket->supersets();
        for (const stk::mesh::Part *part : supersets) {
          std::cout << part->name() << " ";
        }
        std::cout << "}" << std::endl;
      }
    }

    auto print_super_and_subsets([](const stk::mesh::Part *part) {
      std::cout << "    part " << part->name() << " supersets: { ";
      const stk::mesh::PartVector &supersets = part->supersets();
      for (const stk::mesh::Part *part : supersets) {
        std::cout << part->name() << " ";
      }
      std::cout << "}" << std::endl;
      std::cout << "    part " << part->name() << " subsets: { ";
      const stk::mesh::PartVector &subsets = part->subsets();
      for (const stk::mesh::Part *part : subsets) {
        std::cout << part->name() << " ";
      }
      std::cout << "}" << std::endl;
    });

    // print_super_and_subsets(spheres_part_ptr_);
    // print_super_and_subsets(agents_part_ptr_);
    // print_super_and_subsets(springs_part_ptr_);
    // print_super_and_subsets(sphere_sphere_linkers_part_ptr_);
    // print_super_and_subsets(crosslinker_sphere_linkers_part_ptr_);
    print_super_and_subsets(crosslinkers_part_ptr_);
    print_super_and_subsets(left_bound_crosslinkers_part_ptr_);
    print_super_and_subsets(right_bound_crosslinkers_part_ptr_);
    print_super_and_subsets(doubly_bound_crosslinkers_part_ptr_);
    std::cout << "############################################" << std::endl;
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
  }

  void zero_out_transient_constraint_fields() {
    mundy::mesh::utils::fill_field_with_value<unsigned>(*constraint_perform_state_change_field_ptr_,
                                                        std::array<unsigned, 1>{0u});
    mundy::mesh::utils::fill_field_with_value<double>(*constraint_state_change_rate_field_ptr_,
                                                      std::array<double, 1>{0.0});
  }

  void zero_out_accumulator_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*element_corner_displacement_field_ptr_,
                                                      std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  }

  void destroy_crosslinker_sphere_linker_self_interactions() {
    // This is very similar to what is done in DestroyDistantNeighbors, except we are doing it ourselves.

    // Step 0:
    // Populate the destroy field on our ghosted elements.
    stk::mesh::communicate_field_data(*bulk_data_ptr_, {linker_destroy_flag_field_ptr_});

    // Step 1:
    // Loop over each locally owned linker in the selector and mark them for destruction if they self-interact.
    const stk::mesh::Field<int> &linker_destroy_flag_field = *linker_destroy_flag_field_ptr_;
    stk::mesh::Part &left_bound_crosslinkers_part = *left_bound_crosslinkers_part_ptr_;
    stk::mesh::Part &right_bound_crosslinkers_part = *right_bound_crosslinkers_part_ptr_;
    // stk::mesh::Part &doubly_bound_crosslinkers_part = *doubly_bound_crosslinkers_part_ptr_;

    const stk::mesh::Selector locally_owned_input_selector =
        stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_) &
        bulk_data_ptr_->mesh_meta_data().locally_owned_part();

    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, locally_owned_input_selector,
        [&left_bound_crosslinkers_part, &right_bound_crosslinkers_part, &linker_destroy_flag_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &linker) {
          // Get the sphere anc crosslinker attached to the linker.
          const stk::mesh::Entity *crosslinker_and_sphere_elements =
              bulk_data.begin(linker, stk::topology::ELEMENT_RANK);
          const stk::mesh::Entity &crosslinker = crosslinker_and_sphere_elements[0];
          const stk::mesh::Entity &sphere = crosslinker_and_sphere_elements[1];

          // Determine if the node attached to the sphere is the same as the bound node of the crosslinker.
          const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];
          bool is_self_interaction = false;
          if (bulk_data.bucket(crosslinker).member(left_bound_crosslinkers_part)) {
            is_self_interaction = bulk_data.begin_nodes(crosslinker)[0] == sphere_node;
          } else if (bulk_data.bucket(crosslinker).member(right_bound_crosslinkers_part)) {
            is_self_interaction = bulk_data.begin_nodes(crosslinker)[1] == sphere_node;
          }

          // Mark the linker for destruction if there is self-interaction.
          stk::mesh::field_data(linker_destroy_flag_field, linker)[0] = is_self_interaction;
        });

    // Step 2: Destroy the marked linkers
    bulk_data_ptr_->modification_begin();
    const int value_that_indicates_destruction = 1;
    mundy::mesh::utils::destroy_flagged_entities(*bulk_data_ptr_, constraint_rank_, locally_owned_input_selector,
                                                 linker_destroy_flag_field, value_that_indicates_destruction);
    bulk_data_ptr_->modification_end();
    ghost_linked_entities();
  }

  void detect_neighbors_initial() {
    Kokkos::Profiling::pushRegion("StickySettings::detect_neighbors_initial");

    // Set the timing information for the neighbor list rebuilds
    last_neighborlist_update_step_ = 0;
    neighborlist_update_timer_.reset();

    // ComputeAABB for everyone (assume same buffer distance)
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto crosslinkers_selector = stk::mesh::Selector(*crosslinkers_part_ptr_);
    auto sphere_sphere_linkers_selector = stk::mesh::Selector(*sphere_sphere_linkers_part_ptr_);
    auto crosslinker_sphere_linkers_selector = stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_);

    compute_aabb_ptr_->execute(spheres_selector | crosslinkers_selector);
    destroy_neighbor_linkers_ptr_->execute(sphere_sphere_linkers_selector | crosslinker_sphere_linkers_selector);
    ghost_linked_entities();
    generate_sphere_sphere_neighbor_linkers_ptr_->execute(spheres_selector, spheres_selector);
    ghost_linked_entities();
    generate_crosslinker_sphere_neighbor_linkers_ptr_->execute(crosslinkers_selector, spheres_selector);
    ghost_linked_entities();

    Kokkos::Profiling::popRegion();
  }

  void detect_neighbors() {
    Kokkos::Profiling::pushRegion("StickySettings::detect_neighbors");

    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto crosslinkers_selector = stk::mesh::Selector(*crosslinkers_part_ptr_);
    auto sphere_sphere_linkers_selector = stk::mesh::Selector(*sphere_sphere_linkers_part_ptr_);
    auto crosslinker_sphere_linkers_selector = stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_);

    // ComputeAABB for everybody at each time step. The accumulator then always uses this updated information to
    // calculate if we need to update the entire neighbor list.
    compute_aabb_ptr_->execute(spheres_selector | crosslinkers_selector);
    update_accumulators();

    // Check if we need to update the neighbor list. Eventually this will be replaced with a mesh attribute to
    // synchronize across multiple tasks. For now, make sure that the default is to not update neighbor lists.
    check_update_neighbor_list();

    // Now do a check to see if we need to update the neighbor list.
    if (((force_neighborlist_update_) && (timestep_index_ % force_neighborlist_update_nsteps_ == 0)) ||
        update_neighbor_list_) {
      Kokkos::Profiling::pushRegion("StickySettings::detect_neighbors::update_neighbor_list");
      // Read off the timing information before doing anything else and reset it
      auto elapsed_steps = timestep_index_ - last_neighborlist_update_step_;
      auto elapsed_time = neighborlist_update_timer_.seconds();
      neighborlist_update_steps_times_.push_back(std::make_tuple(timestep_index_, elapsed_steps, elapsed_time));
      last_neighborlist_update_step_ = timestep_index_;
      neighborlist_update_timer_.reset();

      // Reset the accumulators
      zero_out_accumulator_fields();

      // Update the neighbor list
      destroy_neighbor_linkers_ptr_->execute(sphere_sphere_linkers_selector | crosslinker_sphere_linkers_selector);
      ghost_linked_entities();
      generate_sphere_sphere_neighbor_linkers_ptr_->execute(spheres_selector, spheres_selector);
      ghost_linked_entities();
      generate_crosslinker_sphere_neighbor_linkers_ptr_->execute(crosslinkers_selector, spheres_selector);
      ghost_linked_entities();
      Kokkos::Profiling::popRegion();
    }
    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for left-bound crosslinkers
  void compute_z_partition_left_bound() {
    Kokkos::Profiling::pushRegion("StickySettings::compute_z_partition_left_bound");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &constraint_state_change_probability = *constraint_state_change_rate_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_constant = *element_hookean_spring_constant_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_rest_length = *element_hookean_spring_rest_length_field_ptr_;
    const mundy::linkers::LinkedEntitiesFieldType &constraint_linked_entities_field =
        *constraint_linked_entities_field_ptr_;
    stk::mesh::Part &left_bound_crosslinkers_part = *left_bound_crosslinkers_part_ptr_;
    stk::mesh::Part &crosslinker_sphere_linkers_part = *crosslinker_sphere_linkers_part_ptr_;
    const double inv_kt = 1.0 / kt_kmc_;
    const double &crosslinker_right_binding_rate = crosslinker_right_binding_rate_;
    const BOND_TYPE &crosslinker_spring_type = crosslinker_spring_type_;

    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, crosslinker_sphere_linkers_part,
        [&node_coord_field, &constraint_state_change_probability, &crosslinker_spring_type,
         &crosslinker_spring_constant, &crosslinker_spring_rest_length, &left_bound_crosslinkers_part, &inv_kt,
         &crosslinker_right_binding_rate, &constraint_linked_entities_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &linker) {
          // Get the sphere and crosslinker attached to the linker.
          const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
              stk::mesh::field_data(constraint_linked_entities_field, linker));
          const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
          const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);

          MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker), std::invalid_argument,
                             "Encountered invalid crosslinker entity in compute_z_partition_left_bound_harmonic.");
          MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere), std::invalid_argument,
                             "Encountered invalid sphere entity in compute_z_partition_left_bound_harmonic.");

          // We need to figure out if this is a self-interaction or not. Since we are a left-bound crosslinker.
          const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];
          bool is_self_interaction = false;
          if (bulk_data.bucket(crosslinker).member(left_bound_crosslinkers_part)) {
            is_self_interaction = bulk_data.begin_nodes(crosslinker)[0] == sphere_node;
          }

          // Only act on the left-bound crosslinkers
          if (bulk_data.bucket(crosslinker).member(left_bound_crosslinkers_part) && !is_self_interaction) {
            const auto dr = mundy::mesh::vector3_field_data(node_coord_field, sphere_node) -
                            mundy::mesh::vector3_field_data(node_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
            const double dr_mag = mundy::math::norm(dr);

            // Compute the Z-partition score
            // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
            // A = crosslinker_binding_rates
            // k = crosslinker_spring_constant
            // r0 = crosslinker_spring_rest_length
            const double A = crosslinker_right_binding_rate;
            const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
            const double r0 = stk::mesh::field_data(crosslinker_spring_rest_length, crosslinker)[0];
#pragma TODO Move the selection of hookean or fene bonds somewhere else, for now, internal if / else
            double Z = 0.0;
            if (crosslinker_spring_type == BOND_TYPE::HARMONIC) {
              Z = A * std::exp(-0.5 * inv_kt * k * (dr_mag - r0) * (dr_mag - r0));
            } else if (crosslinker_spring_type == BOND_TYPE::FENE) {
              // Z = A * (1 - (r/r0)^2)^(0.5*beta*k*r0^2)
              // Notice the minus sign change for this one! It matters, as FENE bonds are fundamentally different.
              Z = A * std::pow((1.0 - (dr_mag / r0) * (dr_mag / r0)), 0.5 * k * inv_kt * r0 * r0);
            }
            stk::mesh::field_data(constraint_state_change_probability, linker)[0] = Z;
          }
        });
    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for doubly_bound crosslinkers
  void compute_z_partition_doubly_bound() {
    Kokkos::Profiling::pushRegion("StickySettings::compute_z_partition_doubly_bound");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    stk::mesh::Part &doubly_bound_crosslinkers_part = *doubly_bound_crosslinkers_part_ptr_;
    const double &crosslinker_right_unbinding_rate = crosslinker_right_unbinding_rate_;

    // Loop over the neighbor list of the crosslinkers, then select down to the ones that are left-bound only.
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_bound_crosslinkers_part,
        [&node_coord_field, &crosslinker_unbinding_rates, &doubly_bound_crosslinkers_part,
         &crosslinker_right_unbinding_rate]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                            const stk::mesh::Entity &crosslinker) {
          // This is a left-bound crosslinker, so just calculate the right unbinding rate and store on the crosslinker
          // itself in the correct position.
          stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1] = crosslinker_right_unbinding_rate;
        });
    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function for everybody
  void compute_z_partition() {
    Kokkos::Profiling::pushRegion("StickySettings::compute_z_partition");

    // Compute the left-bound to doubly-bound score
    compute_z_partition_left_bound();

    // Compute the doubly-bound to left-bound score
    compute_z_partition_doubly_bound();

    Kokkos::Profiling::popRegion();
  }

  void kmc_crosslinker_left_to_doubly() {
    Kokkos::Profiling::pushRegion("StickySettings::kmc_crosslinker_left_to_doubly");
    // Selectors and aliases
    stk::mesh::Part &crosslinker_sphere_linkers_part = *crosslinker_sphere_linkers_part_ptr_;
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    stk::mesh::Field<unsigned> &constraint_perform_state_change_field = *constraint_perform_state_change_field_ptr_;
    stk::mesh::Field<double> &constraint_state_change_rate_field = *constraint_state_change_rate_field_ptr_;
    const double &timestep_size = timestep_size_;
    stk::mesh::Part &left_bound_crosslinkers_part = *left_bound_crosslinkers_part_ptr_;

    // Loop over left-bound crosslinkers and decide if they bind or not
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, left_bound_crosslinkers_part,
        [&crosslinker_sphere_linkers_part, &element_rng_field, &constraint_perform_state_change_field,
         &element_perform_state_change_field, &constraint_state_change_rate_field,
         &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          // Get all of my associated crosslinker_sphere_linkers
          const stk::mesh::Entity &any_arbitrary_crosslinker_node = bulk_data.begin_nodes(crosslinker)[0];
          const stk::mesh::Entity *neighbor_linkers =
              bulk_data.begin(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);
          const unsigned num_neighbor_linkers =
              bulk_data.num_connectivity(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);

          // Loop over the attached crosslinker_sphere_linkers and bind if the rqng falls in their range.
          double z_tot = 0.0;
          for (unsigned j = 0; j < num_neighbor_linkers; j++) {
            const auto &constraint_rank_entity = neighbor_linkers[j];
            const bool is_crosslinker_sphere_linker =
                bulk_data.bucket(constraint_rank_entity).member(crosslinker_sphere_linkers_part);
            if (is_crosslinker_sphere_linker) {
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
            for (unsigned j = 0; j < num_neighbor_linkers; j++) {
              auto &constraint_rank_entity = neighbor_linkers[j];
              bool is_crosslinker_sphere_linker =
                  bulk_data.bucket(constraint_rank_entity).member(crosslinker_sphere_linkers_part);
              if (is_crosslinker_sphere_linker) {
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
    Kokkos::Profiling::pushRegion("StickySettings::kmc_crosslinker_doubly_to_left");

    // Selectors and aliases
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    const double &timestep_size = timestep_size_;
    stk::mesh::Part &doubly_bound_crosslinkers_part = *doubly_bound_crosslinkers_part_ptr_;

    // This is just a loop over the doubly bound crosslinkers, since we know that the right head in is [1].
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_bound_crosslinkers_part,
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
    Kokkos::Profiling::pushRegion("StickySettings::kmc_crosslinker_sphere_linker_sampling");

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
    Kokkos::Profiling::pushRegion("StickySettings::state_change_crosslinkers");

    // Loop over both the CROSSLINKER_SPHERE_LINKERS and the CROSSLINKERS to perform the state changes.
    stk::mesh::Part &left_bound_crosslinkers_part = *left_bound_crosslinkers_part_ptr_;
    stk::mesh::Part &doubly_bound_crosslinkers_part = *doubly_bound_crosslinkers_part_ptr_;

    // Get the vector of entities to modify
    stk::mesh::EntityVector crosslinker_sphere_linkers;
    stk::mesh::EntityVector doubly_bound_crosslinkers;
    stk::mesh::get_selected_entities(stk::mesh::Selector(*crosslinker_sphere_linkers_part_ptr_),
                                     bulk_data_ptr_->buckets(constraint_rank_), crosslinker_sphere_linkers);
    stk::mesh::get_selected_entities(stk::mesh::Selector(*doubly_bound_crosslinkers_part_ptr_),
                                     bulk_data_ptr_->buckets(element_rank_), doubly_bound_crosslinkers);

    // TODO(cje): It might be worth checking to see if we have any state changes in any threads before we crack open the
    // modification section, as even doing that is slightly expensive.

    bulk_data_ptr_->modification_begin();

    // Perform L->D
    for (const stk::mesh::Entity &crosslinker_sphere_linker : crosslinker_sphere_linkers) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*constraint_perform_state_change_field_ptr_, crosslinker_sphere_linker)[0]);
      const bool perform_state_change = state_change_action != BINDING_STATE_CHANGE::NONE;
      if (perform_state_change) {
        // Get our connections (as the genx)
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(*constraint_linked_entities_field_ptr_, crosslinker_sphere_linker));
        const stk::mesh::Entity &crosslinker = bulk_data_ptr_->get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &target_sphere = bulk_data_ptr_->get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(crosslinker), std::invalid_argument,
                           "Encountered invalid crosslinker entity in compute_z_partition_left_bound_harmonic.");
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(target_sphere), std::invalid_argument,
                           "Encountered invalid sphere entity in compute_z_partition_left_bound_harmonic.");

        const stk::mesh::Entity &target_sphere_node = bulk_data_ptr_->begin_nodes(target_sphere)[0];

        // Call the binding function
        if (state_change_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
          // Unbind the right side of the crosslinker from the left node and bind it to the target node
          const bool bind_worked =
              bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, crosslinker, target_sphere_node, 1);
          MUNDY_THROW_ASSERT(bind_worked, std::logic_error, "Failed to bind crosslinker to node.");

          // Now change the part from left to doubly bound.
          const bool is_crosslinker_locally_owned =
              bulk_data_ptr_->parallel_owner_rank(crosslinker) == bulk_data_ptr_->parallel_rank();
          if (is_crosslinker_locally_owned) {
            auto add_parts = stk::mesh::PartVector{doubly_bound_crosslinkers_part_ptr_};
            auto remove_parts = stk::mesh::PartVector{left_bound_crosslinkers_part_ptr_};
            bulk_data_ptr_->change_entity_parts(crosslinker, add_parts, remove_parts);
          }
        }
      }
    }

    // Perform D->L
    for (const stk::mesh::Entity &crosslinker : doubly_bound_crosslinkers) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*element_perform_state_change_field_ptr_, crosslinker)[0]);
      if (state_change_action == BINDING_STATE_CHANGE::DOUBLY_TO_LEFT) {
        // Unbind the right side of the crosslinker from the current node and bind it to the left crosslinker node
        const stk::mesh::Entity &left_node = bulk_data_ptr_->begin_nodes(crosslinker)[0];
        const bool unbind_worked = bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, crosslinker, left_node, 1);
        MUNDY_THROW_ASSERT(unbind_worked, std::logic_error, "Failed to unbind crosslinker from node.");

        // Now change the part from doubly to left bound.
        const bool is_crosslinker_locally_owned =
            bulk_data_ptr_->parallel_owner_rank(crosslinker) == bulk_data_ptr_->parallel_rank();
        if (is_crosslinker_locally_owned) {
          auto add_parts = stk::mesh::PartVector{left_bound_crosslinkers_part_ptr_};
          auto remove_parts = stk::mesh::PartVector{doubly_bound_crosslinkers_part_ptr_};
          bulk_data_ptr_->change_entity_parts(crosslinker, add_parts, remove_parts);
        }
      }
    }

    bulk_data_ptr_->modification_end();
    ghost_linked_entities();

    Kokkos::Profiling::popRegion();
  }

  void update_crosslinker_state() {
    Kokkos::Profiling::pushRegion("StickySettings::update_crosslinker_state");

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

  void compute_hertzian_contact_forces() {
    Kokkos::Profiling::pushRegion("StickySettings::compute_hertzian_contact_forces");
    // Potential evaluation (Hertzian contact)
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto sphere_sphere_linkers_selector = stk::mesh::Selector(*sphere_sphere_linkers_part_ptr_);

    compute_ssd_and_cn_ptr_->execute(sphere_sphere_linkers_selector);
    evaluate_linker_potentials_ptr_->execute(sphere_sphere_linkers_selector);
    linker_potential_force_reduction_ptr_->execute(spheres_selector);
    Kokkos::Profiling::popRegion();
  }

  void compute_harmonic_bond_forces() {
    Kokkos::Profiling::pushRegion("StickySettings::compute_harmonic_bond_forces");

    // Need a compound selector for all springs, including those that are not unbound of singly bound crosslinkers.
    auto actively_bound_springs = stk::mesh::Selector(*springs_part_ptr_) -
                                  (*left_bound_crosslinkers_part_ptr_ | *right_bound_crosslinkers_part_ptr_);

    // Potentials
    compute_constraint_forcing_ptr_->execute(actively_bound_springs);
    Kokkos::Profiling::popRegion();
  }

  void compute_brownian_velocity() {
    // Compute the velocity due to brownian motion
    Kokkos::Profiling::pushRegion("StickySettings::compute_brownian_velocity");

    // Selectors and aliases
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Field<unsigned> &node_rng_field = *node_rng_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    double &timestep_size = timestep_size_;
    double &sphere_drag_coeff = sphere_drag_coeff_;
    double &kt = kt_brownian_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, spheres_part,
        [&node_velocity_field, &node_force_field, &node_rng_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff,
         &kt](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_field, sphere_node);

          // U_brown = sqrt(2 * kt * gamma / dt) * randn
          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * kt * sphere_drag_coeff / timestep_size) * inv_drag_coeff;
          node_velocity[0] += coeff * rng.randn<double>();
          node_velocity[1] += coeff * rng.randn<double>();
          node_velocity[2] += coeff * rng.randn<double>();
          node_rng_counter[0]++;
        });

    Kokkos::Profiling::popRegion();
  }

  void compute_external_velocity() {
    // Compute both the velocity due to external forces
    Kokkos::Profiling::pushRegion("StickySettings::compute_external_velocity");

    // Selectors and aliases
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    double &timestep_size = timestep_size_;
    double &sphere_drag_coeff = sphere_drag_coeff_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, spheres_part,
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

  void update_positions() {
    Kokkos::Profiling::pushRegion("StickySettings::update_positions");

    // Selectors and aliases
    double &timestep_size = timestep_size_;
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;

    // Update the positions for all spheres based on velocity
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, spheres_part,
        [&node_coord_field, &node_velocity_field, &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                                  const stk::mesh::Entity &sphere_node) {
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

  void update_accumulators() {
#pragma TODO This will be at the mercy of periodic boundary condition calculations.
    Kokkos::Profiling::pushRegion("StickySettings::update_accumulators");

    // Selectors and aliases
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Part &crosslinkers_part = *crosslinkers_part_ptr_;
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_aabb_field_old = element_aabb_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;

    stk::mesh::Selector spheres_and_crosslinkers_selector =
        (stk::mesh::Selector(spheres_part) | stk::mesh::Selector(crosslinkers_part));

    // Update the accumulators based on the difference to the previous state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_and_crosslinkers_selector,
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
    Kokkos::Profiling::pushRegion("StickySettings::check_update_neighbor_list");

    // Local variable for if we should update the neighbor list (do as an integer for now because MPI doesn't like
    // bools)
    int local_update_neighbor_list_int = 0;

    // Selectors and aliases
    stk::mesh::Part &spheres_part = *spheres_part_ptr_;
    stk::mesh::Part &crosslinkers_part = *crosslinkers_part_ptr_;
    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;
    const double &skin_distance2_over4 = skin_distance2_over4_;

    stk::mesh::Selector spheres_and_crosslinkers_selector =
        (stk::mesh::Selector(spheres_part) | stk::mesh::Selector(crosslinkers_part));

    // Check if each corner has moved skin_distance/2. Or, if dr_mag2 >= skin_distance^2/4
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_and_crosslinkers_selector,
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
            local_update_neighbor_list_int = 1;
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

  void run(int argc, char **argv) {
    // Preprocess
    parse_user_inputs(argc, argv);
    dump_user_inputs();

    // Setup
    Kokkos::Profiling::pushRegion("StickySettings::Setup");
    build_our_mesh_and_method_instances();

    fetch_fields_and_parts();
    instantiate_metamethods();
    set_mutable_parameters();
    setup_io_mundy();
    declare_and_initialize_sticky();
    Kokkos::Profiling::popRegion();

    // Loadbalance?
    Kokkos::Profiling::pushRegion("StickySettings::Loadbalance");
    if (initial_loadbalance_) {
      loadbalance();
    }
    Kokkos::Profiling::popRegion();

    // Run an initial detection of neighbors not using an adaptive neighborlist to set the initial state, and calculate
    // the initial AABB for everybody.
    detect_neighbors_initial();

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion("MainLoop");
    for (timestep_index_ = 0; timestep_index_ < num_time_steps_; timestep_index_++) {
      // Prepare the current configuration.
      Kokkos::Profiling::pushRegion("StickySettings::ZeroTransient");
      zero_out_transient_node_fields();
      zero_out_transient_element_fields();
      zero_out_transient_constraint_fields();
      Kokkos::Profiling::popRegion();

      rotate_field_states();

      // Detect neighbors of spheres-spheres and crosslinkers-spheres
      update_neighbor_list_ = false;
      detect_neighbors();

      // Update the state changes in the system s(t)
      update_crosslinker_state();

      // Evaluate forces f(x(t)).
      compute_hertzian_contact_forces();
      compute_harmonic_bond_forces();

      // Compute velocity: v(t) = M fext(t) + Ubrown(t)
      compute_brownian_velocity();
      compute_external_velocity();

      // IO. If desired, write out the data for time t (STK or mundy)
      Kokkos::Profiling::pushRegion("StickySettings::IO");
      if (timestep_index_ % io_frequency_ == 0) {
        // Also write out a 'log'
        if (bulk_data_ptr_->parallel_rank() == 0) {
          double tps = static_cast<double>(timestep_index_) / static_cast<double>(timer.seconds());
          std::cout << "Step: " << std::setw(15) << timestep_index_ << ", tps: " << std::setprecision(15) << tps
                    << std::endl;
        }

        io_broker_ptr_->write_io_broker_timestep(static_cast<int>(timestep_index_),
                                                 static_cast<double>(timestep_index_));
      }
      Kokkos::Profiling::popRegion();

      // Update positions:  x(t + dt) = x(t) + dt * v(t).
      update_positions();
    }  // End of time loop
    Kokkos::Profiling::popRegion();

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps_);
      double tps = static_cast<double>(timestep_index_) / static_cast<double>(timer.seconds());
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
  size_t output_file_index_;
  size_t timestep_index_;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<unsigned> *node_rng_field_ptr_;

  stk::mesh::Field<double> *node_coord_field_ptr_;
  stk::mesh::Field<double> *node_velocity_field_ptr_;
  stk::mesh::Field<double> *node_force_field_ptr_;

  stk::mesh::Field<unsigned> *element_rng_field_ptr_;
  stk::mesh::Field<double> *element_radius_field_ptr_;
  stk::mesh::Field<double> *element_hookean_spring_constant_field_ptr_;
  stk::mesh::Field<double> *element_hookean_spring_rest_length_field_ptr_;
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr_;
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr_;
  stk::mesh::Field<double> *element_binding_rates_field_ptr_;
  stk::mesh::Field<double> *element_unbinding_rates_field_ptr_;
  stk::mesh::Field<unsigned> *element_perform_state_change_field_ptr_;
  stk::mesh::Field<double> *element_corner_displacement_field_ptr_;
  stk::mesh::Field<double> *element_aabb_field_ptr_;

  stk::mesh::Field<unsigned> *constraint_perform_state_change_field_ptr_;
  stk::mesh::Field<int> *linker_destroy_flag_field_ptr_;
  stk::mesh::Field<double> *constraint_state_change_rate_field_ptr_;
  stk::mesh::Field<int> *constraint_linked_entity_owners_field_ptr_;
  mundy::linkers::LinkedEntitiesFieldType *constraint_linked_entities_field_ptr_;
  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *spheres_part_ptr_ = nullptr;
  stk::mesh::Part *crosslinkers_part_ptr_ = nullptr;
  stk::mesh::Part *sphere_sphere_linkers_part_ptr_ = nullptr;
  stk::mesh::Part *crosslinker_sphere_linkers_part_ptr_ = nullptr;
  stk::mesh::Part *springs_part_ptr_ = nullptr;
  stk::mesh::Part *agents_part_ptr_ = nullptr;
  stk::mesh::Part *left_bound_crosslinkers_part_ptr_ = nullptr;
  stk::mesh::Part *right_bound_crosslinkers_part_ptr_ = nullptr;
  stk::mesh::Part *doubly_bound_crosslinkers_part_ptr_ = nullptr;
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
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>
      generate_sphere_sphere_neighbor_linkers_ptr_;
  std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>
      generate_crosslinker_sphere_neighbor_linkers_ptr_;
  //@}

  //! \name Fixed params for MetaMethods
  //@{

  Teuchos::ParameterList compute_constraint_forcing_fixed_params_;
  Teuchos::ParameterList compute_aabb_fixed_params_;
  Teuchos::ParameterList compute_ssd_and_cn_fixed_params_;
  Teuchos::ParameterList generate_sphere_sphere_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList generate_crosslinker_sphere_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList evaluate_linker_potentials_fixed_params_;
  Teuchos::ParameterList linker_potential_force_reduction_fixed_params_;
  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params_;
  Teuchos::ParameterList declare_and_init_constraints_fixed_params_;
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

  // Sphere params
  size_t num_spheres_ = 10;
  double initial_sphere_separation_ = 0.5;
  double sphere_radius_ = 0.5;
  double sphere_youngs_modulus_ = 1000.0;
  double sphere_poissons_ratio_ = 0.3;
  double sphere_drag_coeff_ = 1.0;

  // Backbone spring params
  double backbone_spring_constant_ = 100.0;
  double backbone_spring_rest_length_ = 1.0;

  // Crosslinker params
  BOND_TYPE crosslinker_spring_type_ = BOND_TYPE::HARMONIC;
  double crosslinker_spring_constant_ = 100.0;
  double crosslinker_rest_length_ = 1.0;
  double crosslinker_rcut_ = 1.0;
  double crosslinker_left_binding_rate_ = 1.0;
  double crosslinker_right_binding_rate_ = 1.0;
  double crosslinker_left_unbinding_rate_ = 1.0;
  double crosslinker_right_unbinding_rate_ = 1.0;

  // Flags for simulation control
  // Neighbor list
  double skin_distance_ = 1.0;
  double skin_distance2_over4_ = 1.0;
  bool update_neighbor_list_ = false;
  bool force_neighborlist_update_ = false;
  size_t force_neighborlist_update_nsteps_ = 10;
  // Neighborlist rebuild information
  size_t last_neighborlist_update_step_ = 0;
  Kokkos::Timer neighborlist_update_timer_;
  // [timestep, elapsed_timesteps, elapsed_time]
  std::vector<std::tuple<size_t, size_t, double>> neighborlist_update_steps_times_;
  bool print_neighborlist_statistics_ = false;

  // Simulation params
  bool initial_loadbalance_ = false;
  size_t num_time_steps_ = 100;
  size_t io_frequency_ = 10;
  double timestep_size_ = 0.001;
  double kt_brownian_ = 1.0;
  double kt_kmc_ = 1.0;
  //@}
};  // class StickySettings

}  // namespace crosslinkers

}  // namespace alens

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  mundy::alens::crosslinkers::StickySettings().run(argc, argv);

  // Before exiting, sleep for some amount of time to force Kokkos to print better at the end.
  std::this_thread::sleep_for(std::chrono::milliseconds(stk::parallel_machine_rank(MPI_COMM_WORLD)));

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
