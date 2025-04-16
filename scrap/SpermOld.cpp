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
domain. These sperm will be considered fixed and will not move.
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
#include <mundy_constraints/AngularSprings.hpp>             // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#include <mundy_core/MakeStringArray.hpp>                   // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
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

//////////////////////////////
// Finite element rod model //
//////////////////////////////

/* Section nodes:

Centerline-twist rods store the following information
  Element
    Radius | double | len 1 | states 2
  Edge
    Orientation | double | len 4 | states 2
    Tangent     | double | len 3 | states 2
    Binormal    | double | len 3 | states 1
    Length      | double | len 3 | states 1
  Node
    Coord              | double | len 3 | states 2
    Velocity           | double | len 3 | states 1
    Acceleration       | double | len 3 | states 2
    Twist              | double | len 1 | states 2
    Twist rate         | double | len 1 | states 2
    Twist acceleration | double | len 1 | states 2
    Curvature          | double | len 3 | states 1
    Rest curvature     | double | len 3 | states 1
    Rotation gradient  | double | len 4 | states 1
    Force              | double | len 3 | states 1
    Twist torque       | double | len 1 | states 1

We require the following constant material properties for the rods:
  - Young's modulus
  - Poisson's ratio
  - Shear modulus
  - Density

Other miscellaneous parameters:
  - Time step size

We need to be able to perform the following operations on these rods:
  - Compute edge information (length, tangent, and binormal).
  - Compute node curvature and rotation gradient
  - Compute node rest curvature
  - Compute internal force and torque
  - Compute stretch force
  - Compute collision force
  - Compute node motion using a multi-step scheme
  - Write the rods to disk
*/

class SLT : public mundy::meta::MetaMethodExecutionInterface<void> {
 public:
  SLT(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument, "SLT: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(SLT::get_valid_fixed_params());

    // Store the SLT part acton on by this method.,
    slt_part_ptr_ = meta_data_ptr_->get_part(default_part_name_);

    // Fetch the fields.
    node_coordinates_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_coordinates_field_name_);
    node_velocity_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_velocity_field_name_);
    node_acceleration_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_acceleration_field_name_);
    node_twist_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_twist_field_name_);
    node_twist_rate_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_twist_rate_field_name_);
    node_twist_acceleration_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_twist_acceleration_field_name_);
    node_curvature_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_curvature_field_name_);
    node_rest_curvature_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_rest_curvature_field_name_);
    node_rotation_gradient_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_rotation_gradient_field_name_);
    node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_force_field_name_);
    node_twist_torque_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, default_node_twist_torque_field_name_);

    edge_orientation_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_orientation_field_name_);
    edge_tangent_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_tangent_field_name_);
    edge_binormal_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_binormal_field_name_);
    edge_length_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::EDGE_RANK, default_edge_length_field_name_);

    element_radius_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, default_element_radius_field_name_);

    auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
      MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                         std::string("SLT: Field ") + field_name + " cannot be a nullptr. Check that the field exists.");
    };  // field_exists

    field_exists(node_coordinates_field_ptr_, default_node_coordinates_field_name_);
    field_exists(node_velocity_field_ptr_, default_node_velocity_field_name_);
    field_exists(node_acceleration_field_ptr_, default_node_acceleration_field_name_);
    field_exists(node_twist_field_ptr_, default_node_twist_field_name_);
    field_exists(node_twist_rate_field_ptr_, default_node_twist_rate_field_name_);
    field_exists(node_twist_acceleration_field_ptr_, default_node_twist_acceleration_field_name_);
    field_exists(node_curvature_field_ptr_, default_node_curvature_field_name_);
    field_exists(node_rest_curvature_field_ptr_, default_node_rest_curvature_field_name_);
    field_exists(node_rotation_gradient_field_ptr_, default_node_rotation_gradient_field_name_);
    field_exists(node_force_field_ptr_, default_node_force_field_name_);
    field_exists(node_twist_torque_field_ptr_, default_node_twist_torque_field_name_);

    field_exists(edge_orientation_field_ptr_, default_edge_orientation_field_name_);
    field_exists(edge_tangent_field_ptr_, default_edge_tangent_field_name_);
    field_exists(edge_binormal_field_ptr_, default_edge_binormal_field_name_);
    field_exists(edge_length_field_ptr_, default_edge_length_field_name_);

    field_exists(element_radius_field_ptr_, default_element_radius_field_name_);
  }
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshReqs
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(SLT::get_valid_fixed_params());

    // We require that the SLT part exists, has a BEAM_3 topology, and has the desired node/edge/element fields.
    auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
    part_reqs->set_part_name(default_part_name_);
    part_reqs->set_part_topology(stk::topology::BEAM_3);

    // Add the node fields
    part_reqs->add_field_reqs<double>(default_node_coordinates_field_name_, stk::topology::NODE_RANK, 3, 2);
    part_reqs->add_field_reqs<double>(default_node_velocity_field_name_, stk::topology::NODE_RANK, 3, 2);
    part_reqs->add_field_reqs<double>(default_node_acceleration_field_name_, stk::topology::NODE_RANK, 3, 2);
    part_reqs->add_field_reqs<double>(default_node_twist_field_name_, stk::topology::NODE_RANK, 1, 2);
    part_reqs->add_field_reqs<double>(default_node_twist_rate_field_name_, stk::topology::NODE_RANK, 1, 2);
    part_reqs->add_field_reqs<double>(default_node_twist_acceleration_field_name_, stk::topology::NODE_RANK, 1, 2);
    part_reqs->add_field_reqs<double>(default_node_curvature_field_name_, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_node_rest_curvature_field_name_, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_node_rotation_gradient_field_name_, stk::topology::NODE_RANK, 4, 1);
    part_reqs->add_field_reqs<double>(default_node_force_field_name_, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_node_twist_torque_field_name_, stk::topology::NODE_RANK, 1, 1);

    // Add the edge fields
    part_reqs->add_field_reqs<double>(default_edge_orientation_field_name_, stk::topology::EDGE_RANK, 4, 2);
    part_reqs->add_field_reqs<double>(default_edge_tangent_field_name_, stk::topology::EDGE_RANK, 3, 2);
    part_reqs->add_field_reqs<double>(default_edge_binormal_field_name_, stk::topology::EDGE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(default_edge_length_field_name_, stk::topology::EDGE_RANK, 1, 1);

    // Add the element fields
    part_reqs->add_field_reqs<double>(default_element_radius_field_name_, stk::topology::ELEMENT_RANK, 1, 1);

    // Create the mesh requirements
    auto mesh_reqs = std::make_shared<mundy::meta::MeshReqs>();
    mesh_reqs->add_and_sync_part_reqs(part_reqs);
    return mesh_reqs;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;

    // Material properties
    default_parameter_list.set("youngs_modulus", default_youngs_modulus_, "Young's modulus.");
    default_parameter_list.set("poissons_ratio", default_poissons_ratio_, "Poisson's ratio.");
    default_parameter_list.set("shear_modulus", default_shear_modulus_, "Shear modulus.");
    default_parameter_list.set("density", default_density_, "Density.");

    // Miscellaneous parameters
    default_parameter_list.set("time_step_size", default_time_step_size_, "The timestep size.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<SLT>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(SLT::get_valid_mutable_params());

    // Material properties
    youngs_modulus_ = valid_mutable_params.get<double>("youngs_modulus");
    poissons_ratio_ = valid_mutable_params.get<double>("poissons_ratio");
    shear_modulus_ = valid_mutable_params.get<double>("shear_modulus");
    density_ = valid_mutable_params.get<double>("density");

    // Miscellaneous parameters
    time_step_size_ = valid_mutable_params.get<double>("time_step_size");

    // Check invariants: All material properties must be positive. Time step size must be positive.
    MUNDY_THROW_REQUIRE(youngs_modulus_ > 0.0, std::invalid_argument, "SLT: Young's modulus must be greater than zero.");
    MUNDY_THROW_REQUIRE(poissons_ratio_ > 0.0, std::invalid_argument, "SLT: Poisson's ratio must be greater than zero.");
    MUNDY_THROW_REQUIRE(shear_modulus_ > 0.0, std::invalid_argument, "SLT: Shear modulus must be greater than zero.");
    MUNDY_THROW_REQUIRE(density_ > 0.0, std::invalid_argument, "SLT: Density must be greater than zero.");
    MUNDY_THROW_REQUIRE(time_step_size_ > 0.0, std::invalid_argument, "SLT: time_step_size must be greater than zero.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return {slt_part_ptr_};
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute() {
    // - Compute edge information (length, tangent, and binormal).
    // - Compute node curvature
    // - Compute internal force and torque
    // - Compute stretch force
    // - Compute collision force
    // - Compute node motion using a multi-step scheme

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &slt_part = *slt_part_ptr_;
    stk::mesh::Field<double> &node_coordinates_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_acceleration_field = *node_acceleration_field_ptr_;
    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &node_twist_rate_field = *node_twist_rate_field_ptr_;
    stk::mesh::Field<double> &node_twist_acceleration_field = *node_twist_acceleration_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rest_curvature_field = *node_rest_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_twist_torque_field = *node_twist_torque_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    const double youngs_modulus = youngs_modulus_;
    const double poissons_ratio = poissons_ratio_;
    const double shear_modulus = shear_modulus_;
    const double density = density_;
    const double time_step_size = time_step_size_;
    const double node_mass = node_mass_;
    const double node_moment_of_inertia = node_moment_of_inertia_;

    // For some of our fields, we need to know about the reference configuration of the SLT part.
    // This information is stored in StateN
    // A note about states: StateNone, StateNew, and StateNP1 all refer to the same default accessed field.
    // StateOld and StateN refer to the old state, which, in our case, is the reference configuration.
    //
    stk::mesh::Field<double> &node_coordinates_field_ref = node_coordinates_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_velocity_field_ref = node_velocity_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_acceleration_field_ref = node_acceleration_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_field_ref = node_twist_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_rate_field_ref = node_twist_rate_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_acceleration_field_ref =
        node_twist_acceleration_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_orientation_field_ref = edge_orientation_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_tangent_field_ref = edge_tangent_field.field_of_state(stk::mesh::StateN);

    // Most of our calculations will be done on the locally owned part of the SLT part.
    stk::mesh::Selector locally_owned_selector = slt_part & meta_data_ptr_->locally_owned_part();

    // Compute node motion using generalized velocity Verlet.
    //
    // x is the generalized position containing node coordinate and twist
    // v, a, f are the generalized velocity, acceleration, and force.
    // M is the generalized mass matrix.
    //
    //   x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2
    //   Evaluate internal force f(x(t + dt))
    //   a(t + dt) = M^{-1} f(x(t + dt))
    //   v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2

    // First x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::NODE_RANK, locally_owned_selector,
        [&node_coordinates_field, &node_velocity_field, &node_acceleration_field, &node_twist_field,
         &node_coordinates_field_ref, &node_velocity_field_ref, &node_acceleration_field_ref, &node_twist_field_ref,
         &node_twist_rate_field_ref, &node_twist_acceleration_field_ref,
         time_step_size](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          // We update the current configuration using the old.
          const auto node_coords_ref = mundy::mesh::vector3_field_data(node_coordinates_field_ref, node);
          const auto node_velocity_ref = mundy::mesh::vector3_field_data(node_velocity_field_ref, node);
          const auto node_acceleration_ref = mundy::mesh::vector3_field_data(node_acceleration_field_ref, node);
          const double node_twist_ref = stk::mesh::field_data(node_twist_field_ref, node)[0];
          const double node_twist_rate_ref = stk::mesh::field_data(node_twist_rate_field_ref, node)[0];
          const double node_twist_acceleration_ref = stk::mesh::field_data(node_twist_acceleration_field_ref, node)[0];

          // Get the output fields
          auto node_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node);
          double *node_twist = stk::mesh::field_data(node_twist_field, node);

          // Update the current configuration
          node_coords = node_coords_ref + node_velocity_ref * time_step_size +
                        node_acceleration_ref * time_step_size * time_step_size / 2.0;
          node_twist[0] = node_twist_ref + node_twist_rate_ref * time_step_size +
                          node_twist_acceleration_ref * time_step_size * time_step_size / 2.0;
        });

    // Now, evaluate the internal force f(x(t + dt))
    // This requires updating all the fields that vary with the current configuration.

    // For each edge in the SLT part, compute the edge tangent, binormal, and length.
    // length^i = ||x_{i+1} - x_i||
    // edge_tangent^i = (x_{i+1} - x_i) / length
    // edge_binormal^i = (2 edge_tangent_ref^i x edge_tangent^i) / (1 + edge_tangent_ref^i dot edge_tangent^i)
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::EDGE_RANK, locally_owned_selector,
        [&node_coordinates_field, &edge_orientation_field, &edge_tangent_field, &edge_tangent_field_ref,
         &edge_binormal_field,
         &edge_length_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &slt_edge) {
          // Get the nodes of the edge
          stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(slt_edge);
          stk::mesh::Entity const &node_i = edge_nodes[0];
          stk::mesh::Entity const &node_ip1 = edge_nodes[1];

          // Get the required input fields
          const auto node_i_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node_i);
          const auto node_ip1_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node_ip1);
          const auto edge_tangent_ref = mundy::mesh::vector3_field_data(edge_tangent_field_ref, slt_edge);

          // Get the output fields
          auto edge_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, slt_edge);
          auto edge_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, slt_edge);
          auto edge_length = mundy::math::get_scalar_view<double>(stk::mesh::field_data(edge_length_field, slt_edge));

          // Compute the un-normalized edge tangent
          edge_tangent = node_ip1_coords - node_i_coords;
          edge_length[0] = mundy::math::norm(edge_tangent);
          edge_tangent /= edge_length[0];

          // Compute the edge binormal
          edge_binormal = (2.0 * mundy::math::cross(edge_tangent_ref, edge_tangent)) /
                          (1.0 + mundy::math::dot(edge_tangent_ref, edge_tangent));
        });

    // For each element in the SLT part, compute the node curvature at the center node.
    // The curvature can be computed from the edge orientations using
    //   kappa^i = q_i - conj(q_i) = 2 * vec(q_i)
    // where
    //   q_i = conj(d^{i-1}) d^i is the Lagrangian rotation gradient.
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::ELEMENT_RANK, locally_owned_selector,
        [&edge_orientation_field, &node_curvature_field, &node_rotation_gradient_field](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &slt_element) {
          // Curvature needs to "know" about the order of edges, so it's best to loop over the slt elements and not the
          // nodes. Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(slt_element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(slt_element);

          MUNDY_THROW_ASSERT(bulk_data.num_nodes(slt_element) == 3, std::logic_error,
                             "SLT: Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(slt_element) == 2, std::logic_error,
                             "SLT: Elements must have exactly 2 edges.");

          // Get the required input fields
          const auto edge_im1_orientation =
              mundy::mesh::quaternion_field_data(edge_orientation_field, element_edges[0]);
          const auto edge_i_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, element_edges[1]);

          // Get the output fields
          auto node_curvature = mundy::mesh::vector3_field_data(node_curvature_field, element_nodes[1]);
          auto node_rotation_gradient = mundy::math::get_quaternion_view<double>(
              stk::mesh::field_data(node_rotation_gradient_field, element_nodes[1]));

          // Compute the node curvature
          node_rotation_gradient = mundy::math::conj(edge_im1_orientation) * edge_i_orientation;
          node_curvature = 2.0 * node_rotation_gradient.vector();
        });

    // Compute internal force and torque
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::ELEMENT_RANK, locally_owned_selector,
        [&node_force_field, &&node_curvature_field, &node_rest_curvature_field, &node_twist_field,
         &node_rotation_gradient_field, &edge_tangent_field, &edge_binormal_field, &edge_length_field,
         &edge_orientation_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &slt_element) {
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
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(slt_element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(slt_element);

          MUNDY_THROW_ASSERT(bulk_data.num_nodes(slt_element) == 3, std::logic_error,
                             "SLT: Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(slt_element) == 2, std::logic_error,
                             "SLT: Elements must have exactly 2 edges.");

          // Get the required input fields
          const auto node_i_curvature = mundy::mesh::vector3_field_data(node_curvature_field, element_nodes[1]);
          const auto node_i_rest_curvature =
              mundy::mesh::vector3_field_data(node_rest_curvature_field, element_nodes[1]);
          const auto node_i_twist =
              mundy::math::get_scalar_view<double>(stk::mesh::field_data(node_twist_field, element_nodes[1]));
          const auto node_im1_twist =
              mundy::math::get_scalar_view<double>(stk::mesh::field_data(node_twist_field, element_nodes[0]));
          const auto node_i_rotation_gradient = mundy::math::get_quaternion_view<double>(
              stk::mesh::field_data(node_rotation_gradient_field, element_nodes[1]));
          const auto edge_im1_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, element_edges[0]);
          const auto edge_i_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, element_edges[1]);
          const auto edge_im1_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, element_edges[0]);
          const auto edge_i_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, element_edges[1]);
          const auto edge_im1_length =
              mundy::math::get_scalar_view<double>(stk::mesh::field_data(edge_length_field, element_edges[0]));
          const auto edge_i_length =
              mundy::math::get_scalar_view<double>(stk::mesh::field_data(edge_length_field, element_edges[1]));
          const auto edge_im1_orientation =
              mundy::mesh::quaternion_field_data(edge_orientation_field, element_edges[0]);

          // Get the output fields
          auto node_im1_force = mundy::mesh::vector3_field_data(node_force_field, element_nodes[0]);
          auto node_i_force = mundy::mesh::vector3_field_data(node_force_field, element_nodes[1]);
          auto node_ip1_force = mundy::mesh::vector3_field_data(node_force_field, element_nodes[2]);
          double *node_im1_twist_torque = stk::mesh::field_data(node_twist_torque_field, element_nodes[0]);
          double *node_i_twist_torque = stk::mesh::field_data(node_twist_torque_field, element_nodes[1]);

          // Compute the torque induced by the curvature
          // To start, the torque is in the lagrangian frame.
          auto bending_modulus = mundy::math::Matrix3::identity();  // TODO(palmerb4): Replace with Kirchhoff rod model
          auto bending_torque = bending_modulus * (node_i_curvature - node_i_rest_curvature);

          // We'll use the bending torque as a placeholder for the rotated bending torque
          bending_torque =
              edge_im1_orientation * (node_i_rotation_gradient.w() * bending_torque +
                                      mundy::math::cross(node_i_rotation_gradient.vector(), bending_torque));

          // Compute the force and torque on the nodes
          const auto tmp_force_ip1 = 1.0 / edge_i_length[0] *
                                     (mundy::math::cross(bending_torque, edge_i_tangent) +
                                      0.5 * mundy::math::dot(edge_i_tangent, bending_torque) *
                                          (mundy::math::dot(edge_i_tangent, edge_i_binormal) * edge_i_tangent) -
                                      edge_i_binormal);
          const auto tmp_force_im1 =
              1.0 / edge_im1_length[0] *
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

    // Compute stretch forces

    // Compute collision forces

    // At this point, we finally have f(x(t + dt)). Now we need to compute a(t + dt) = M^{-1} f(x(t + dt))
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::NODE_RANK, locally_owned_selector,
        [&node_acceleration_field, &node_force_field, $node_twist_acceleration_field, &node_twist_torque_field,
         node_mass, node_moment_of_inertia](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const auto node_force = mundy::mesh::vector3_field_data(node_force_field, node);
          const double node_twist_torque = stk::mesh::field_data(node_twist_torque_field, node)[0];

          // Get the output fields
          auto node_acceleration = mundy::mesh::vector3_field_data(node_acceleration_field, node);
          double *node_twist_acceleration = stk::mesh::field_data(node_twist_acceleration_field, node);

          // Compute the acceleration
          node_acceleration = node_force / node_mass;
          node_twist_acceleration = node_twist_torque / node_moment_of_inertia;
        });

    // Finally, we can compute v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2
    mundy::mesh::for_each_entity_run(
        bulk_data, stk::topology::NODE_RANK, locally_owned_selector,
        [&node_velocity_field, &node_velocity_field_ref, &node_acceleration_field, &node_acceleration_field_ref,
         &node_twist_rate_field, &node_twist_rate_field_ref, &node_twist_acceleration_field,
         &node_twist_acceleration_field_ref,
         time_step_size](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const auto node_velocity_ref = mundy::mesh::vector3_field_data(node_velocity_field_ref, node);
          const auto node_acceleration = mundy::mesh::vector3_field_data(node_acceleration_field, node);
          const auto node_acceleration_ref = mundy::mesh::vector3_field_data(node_acceleration_field_ref, node);
          const double node_twist_rate_ref = stk::mesh::field_data(node_twist_rate_field_ref, node)[0];
          const double node_twist_acceleration = stk::mesh::field_data(node_twist_acceleration_field, node)[0];
          const double node_twist_acceleration_ref = stk::mesh::field_data(node_twist_acceleration_field_ref, node)[0];

          // Get the output fields
          auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node);
          double *node_twist_rate = stk::mesh::field_data(node_twist_rate_field, node);

          // Compute the velocity
          node_velocity = node_velocity_ref + 0.5 * (node_acceleration + node_acceleration_ref) * time_step_size;
          node_twist_rate =
              node_twist_rate_ref + 0.5 * (node_twist_acceleration + node_twist_acceleration_ref) * time_step_size;
        });
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_time_step_size_ = 0.0;
  static inline double default_youngs_modulus_ = 1000.0;
  static inline double default_poissons_ratio_ = 0.3;
  static inline double default_shear_modulus_ = 1000.0;
  static inline double default_density_ = 1.0;
  static constexpr std::string_view default_part_name_ = "SLT";
  static constexpr std::string_view default_node_coordinates_field_name_ = "NODE_COORDS";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_twist_field_name_ = "NODE_TWIST";
  static constexpr std::string_view default_node_curvature_field_name_ = "NODE_CURVATURE";
  static constexpr std::string_view default_node_rest_curvature_field_name_ = "NODE_REST_CURVATURE";
  static constexpr std::string_view default_edge_orientation_field_name_ = "EDGE_ORIENTATION";
  static constexpr std::string_view default_edge_tangent_field_name_ = "EDGE_TANGENT";
  static constexpr std::string_view default_edge_binormal_field_name_ = "EDGE_BINORMAL";
  static constexpr std::string_view default_edge_length_field_name_ = "EDGE_LENGTH";
  static constexpr std::string_view default_element_radius_field_name_ = "ELEMENT_RADIUS";
  //@}

  //! \name Internal members
  //@{

  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  // Material properties
  double youngs_modulus_;
  double poissons_ratio_;
  double shear_modulus_;
  double density_;

  // Miscellaneous parameters
  double time_step_size_;

  // Time invariant quantities
  double node_mass_;
  double node_moment_of_inertia_;

  // Node rank fields
  stk::mesh::Field<double> *node_coordinates_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_twist_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_curvature_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_rest_curvature_field_ptr_ = nullptr;
  stk::mesh::Field<double> *node_rotation_gradient_field_ptr_ = nullptr;

  // Edge rank fields
  stk::mesh::Field<double> *edge_orientation_field_ptr_ = nullptr;
  stk::mesh::Field<double> *edge_tangent_field_ptr_ = nullptr;
  stk::mesh::Field<double> *edge_binormal_field_ptr_ = nullptr;
  stk::mesh::Field<double> *edge_length_field_ptr_ = nullptr;

  // Element rank fields
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // SLT

///////////////////////////
// Partitioning settings //
///////////////////////////
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

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Default values for the inputs
  size_t num_spheres = 10;
  double sphere_radius = 0.6;
  double initial_segment_length = 1.0;
  double rest_length = 2 * sphere_radius;
  bool loadbalance_initial_config = true;

  size_t num_time_steps = 100;
  double timestep_size = 0.01;
  double diffusion_coeff = 1.0;
  double viscosity = 1.0;
  double youngs_modulus = 1000.0;
  double poissons_ratio = 0.3;
  double spring_constant = 1.0;
  double angular_spring_constant = 1.0;
  double angular_spring_rest_angle = M_PI;

  // Parse the command line options.
  Teuchos::CommandLineProcessor cmdp(false, true);

  // Optional command line arguments for controlling
  //   sphere initialization:
  cmdp.setOption("num_spheres", &num_spheres, "Number of spheres.");
  cmdp.setOption("sphere_radius", &sphere_radius, "The radius of the spheres.");
  cmdp.setOption("initial_segment_length", &initial_segment_length, "Initial segment length.");
  cmdp.setOption("rest_length", &rest_length, "Rest length of the spring.");
  cmdp.setOption("loadbalance", "no_loadbalance", &loadbalance_initial_config,
                 "Load balance the initial configuration.");

  //   spring initialization
  cmdp.setOption("spring_constant", &spring_constant, "Spring constant.");
  cmdp.setOption("angular_spring_constant", &angular_spring_constant, "Angular spring constant.");
  cmdp.setOption("angular_spring_rest_angle", &angular_spring_rest_angle, "Angular spring rest angle.");

  //   the simulation:
  cmdp.setOption("num_time_steps", &num_time_steps, "Number of time steps.");
  cmdp.setOption("timestep_size", &timestep_size, "Time step size.");
  cmdp.setOption("diffusion_coeff", &diffusion_coeff, "Diffusion coefficient.");
  cmdp.setOption("viscosity", &viscosity, "Viscosity.");
  cmdp.setOption("youngs_modulus", &youngs_modulus, "Young's modulus.");
  cmdp.setOption("poissons_ratio", &poissons_ratio, "Poisson's ratio.");

  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    Kokkos::finalize();
    stk::parallel_machine_finalize();
    return EXIT_FAILURE;
  }

  MUNDY_THROW_REQUIRE(timestep_size > 0, std::invalid_argument, "Time step size must be greater than zero.");

  // Dump the parameters to screen on rank 0
  if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
    std::cout << "##################################################" << std::endl;
    std::cout << "INPUT PARAMETERS:" << std::endl;
    std::cout << "  num_spheres: " << num_spheres << std::endl;
    std::cout << "  sphere_radius: " << sphere_radius << std::endl;
    std::cout << "  initial_segment_length: " << initial_segment_length << std::endl;
    std::cout << "  rest_length: " << rest_length << std::endl;
    std::cout << "  num_time_steps: " << num_time_steps << std::endl;
    std::cout << "  timestep_size: " << timestep_size << std::endl;
    std::cout << "  diffusion_coeff: " << diffusion_coeff << std::endl;
    std::cout << "  viscosity: " << viscosity << std::endl;
    std::cout << "  youngs_modulus: " << youngs_modulus << std::endl;
    std::cout << "  poissons_ratio: " << poissons_ratio << std::endl;
    std::cout << "  spring_constant: " << spring_constant << std::endl;
    std::cout << "  angular_spring_constant: " << angular_spring_constant << std::endl;
    std::cout << "##################################################" << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  // Setup the fixed parameters and generate the corresponding class instances and mesh //
  ////////////////////////////////////////////////////////////////////////////////////////

  // ComputeConstraintForcing fixed parameters
  Teuchos::ParameterList compute_constraint_forcing_fixed_params;
  compute_constraint_forcing_fixed_params.set(
      "enabled_kernel_names", mundy::core::make_string_array(mundy::constraints::HookeanSprings::get_name()));

  // ComputeSignedSeparationDistanceAndContactNormal fixed parameters
  Teuchos::ParameterList compute_ssd_and_cn_fixed_params;
  compute_ssd_and_cn_fixed_params.set(
      "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER"));

  // ComputeAABB fixed parameters
  Teuchos::ParameterList compute_aabb_fixed_params;
  compute_aabb_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));

  // GenerateNeighborLinkers fixed parameters
  Teuchos::ParameterList generate_neighbor_linkers_fixed_params;
  generate_neighbor_linkers_fixed_params.set("enabled_technique_name", "STK_SEARCH")
      .set("specialized_neighbor_linkers_part_names",
           mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKERS"));
  generate_neighbor_linkers_fixed_params.sublist("STK_SEARCH")
      .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"))
      .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENTS"));

  // EvaluateLinkerPotentials fixed parameters
  Teuchos::ParameterList evaluate_linker_potentials_fixed_params;
  evaluate_linker_potentials_fixed_params.set(
      "enabled_kernel_names",
      mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));

  // LinkerPotentialForceReduction fixed parameters
  Teuchos::ParameterList linker_potential_force_reduction_fixed_params;
  linker_potential_force_reduction_fixed_params.set("enabled_kernel_names",
                                                              mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));

  // DestroyNeighborLinkers fixed parameters
  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params = Teuchos::ParameterList();
  destroy_neighbor_linkers_fixed_params.set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");

  // DeclareAndInitConstraints fixed parameters
  Teuchos::ParameterList declare_and_init_constraints_fixed_params;
  declare_and_init_constraints_fixed_params.set("enabled_technique_name", "CHAIN_OF_SPRINGS")
      .sublist("CHAIN_OF_SPRINGS")
      .set("hookean_springs_part_names", mundy::core::make_string_array(mundy::constraints::HookeanSprings::get_name()))
      .set("angular_springs_part_names", mundy::core::make_string_array(mundy::constraints::AngularSprings::get_name()))
      .set("sphere_part_names", mundy::core::make_string_array(mundy::shapes::Spheres::get_name()))
      .set("spherocylinder_segment_part_names",
           mundy::core::make_string_array(mundy::shapes::SpherocylinderSegments::get_name()))
      .set<bool>("generate_hookean_springs", true)
      .set<bool>("generate_angular_springs", false)
      .set<bool>("generate_spheres_at_nodes", false)
      .set<bool>("generate_spherocylinder_segments_along_edges", true);

  // Create the class instances and mesh based on the given fixed requirements.
  auto [compute_constraint_forcing_ptr, compute_ssd_and_cn_ptr, compute_aabb_ptr, generate_neighbor_linkers_ptr,
        evaluate_linker_potentials_ptr, linker_potential_force_reduction_ptr, destroy_neighbor_linkers_ptr,
        declare_and_init_constraints_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
          mundy::constraints::ComputeConstraintForcing, mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal,
          mundy::shapes::ComputeAABB, mundy::linkers::GenerateNeighborLinkers, mundy::linkers::EvaluateLinkerPotentials,
          mundy::linkers::LinkerPotentialForceReduction, mundy::linkers::DestroyNeighborLinkers,
          mundy::constraints::DeclareAndInitConstraints>(
          {node_euler_fixed_params, compute_mobility_fixed_params, compute_constraint_forcing_fixed_params,
           compute_ssd_and_cn_fixed_params, compute_aabb_fixed_params, generate_neighbor_linkers_fixed_params,
           evaluate_linker_potentials_fixed_params, linker_potential_force_reduction_fixed_params,
           destroy_neighbor_linkers_fixed_params, declare_and_init_constraints_fixed_params});

  auto check_class_instance = [](auto &class_instance_ptr, const std::string &class_name) {
    MUNDY_THROW_REQUIRE(class_instance_ptr != nullptr, std::invalid_argument,
                       std::string("Failed to create class instance with name << ") + class_name + " >>.");
  };  // check_class_instance

  check_class_instance(node_euler_ptr, "NodeEuler");
  check_class_instance(compute_mobility_ptr, "ComputeMobility");
  check_class_instance(compute_constraint_forcing_ptr, "ComputeConstraintForces");
  check_class_instance(compute_ssd_and_cn_ptr, "ComputeSignedSeparationDistanceAndContactNormal");
  check_class_instance(compute_aabb_ptr, "ComputeAABB");
  check_class_instance(generate_neighbor_linkers_ptr, "GenerateNeighborLinkers");
  check_class_instance(evaluate_linker_potentials_ptr, "EvaluateLinkerPotentials");
  check_class_instance(linker_potential_force_reduction_ptr, "LinkerPotentialForceReduction");
  check_class_instance(destroy_neighbor_linkers_ptr, "DestroyNeighborLinkers");
  check_class_instance(declare_and_init_constraints_ptr, "DeclareAndInitConstraints");

  MUNDY_THROW_REQUIRE(bulk_data_ptr != nullptr, std::invalid_argument, "Bulk dta pointer cannot be a nullptr.");
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  MUNDY_THROW_REQUIRE(meta_data_ptr != nullptr, std::invalid_argument, "Meta data pointer cannot be a nullptr.");
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");

  ///////////////////////////////////////////////////
  // Set up the mutable parameters for the classes //
  ///////////////////////////////////////////////////

  // NodeEuler mutable parameters
  Teuchos::ParameterList node_euler_mutable_params;
  node_euler_mutable_params.set("timestep_size", timestep_size);
  node_euler_ptr->set_mutable_params(node_euler_mutable_params);

  // ComputeMobility mutable parameters
  Teuchos::ParameterList compute_mobility_mutable_params;
  compute_mobility_mutable_params.sublist("LOCAL_DRAG").set("viscosity", viscosity);
  compute_mobility_ptr->set_mutable_params(compute_mobility_mutable_params);

  // ComputeConstraintForces mutable parameters
  // Doesn't have any mutable parameters to set

  // ComputeSignedSeparationDistanceAndContactNormal mutable parameters
  // Doesn't have any mutable parameters to set

  // ComputeAABB mutable parameters
  Teuchos::ParameterList compute_aabb_mutable_params;
  compute_aabb_mutable_params.set("buffer_distance", 0.0);
  compute_aabb_ptr->set_mutable_params(compute_aabb_mutable_params);

  // GenerateNeighborLinkers mutable parameters
  // Doesn't have any mutable parameters to set

  // EvaluateLinkerPotentials mutable parameters
  // Doesn't have any mutable parameters to set

  // LinkerPotentialForceReduction mutable parameters
  // Doesn't have any mutable parameters to set

  // DestroyNeighborLinkers mutable parameters
  // Doesn't have any mutable parameters to set

  ////////////////////////////////
  // Fetch the fields and parts //
  ////////////////////////////////
  // Node rank fields
  auto node_coordinates_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  auto node_velocity_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_VELOCITY");
  auto node_force_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_FORCE");

  // Element rank fields
  auto element_radius_field_ptr = meta_data_ptr->get_field<double>(
      stk::topology::ELEMENT_RANK, mundy::shapes::Spheres::get_element_radius_field_name());
  auto element_youngs_modulus_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_YOUNGS_MODULUS");
  auto element_poissons_ratio_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_POISSONS_RATIO");
  auto element_rest_length_field_ptr = meta_data_ptr->get_field<double>(
      stk::topology::ELEMENT_RANK, mundy::constraints::HookeanSprings::get_element_rest_length_field_name());
  auto element_spring_constant_field_ptr = meta_data_ptr->get_field<double>(
      stk::topology::ELEMENT_RANK, mundy::constraints::HookeanSprings::get_element_spring_constant_field_name());
  auto element_angular_spring_rest_angle_field_ptr = meta_data_ptr->get_field<double>(
      stk::topology::ELEMENT_RANK, mundy::constraints::AngularSprings::get_element_rest_angle_field_name());
  auto element_angular_spring_constant_field_ptr = meta_data_ptr->get_field<double>(
      stk::topology::ELEMENT_RANK, mundy::constraints::AngularSprings::get_element_spring_constant_field_name());

  // Linker (constraint rank) fields
  auto linker_contact_normal_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_CONTACT_NORMAL");
  auto linker_signed_separation_distance_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_SIGNED_SEPARATION_DISTANCE");
  auto linker_potential_force_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_POTENTIAL_FORCE");
  auto linker_destroy_flag_field_ptr =
      meta_data_ptr->get_field<int>(stk::topology::CONSTRAINT_RANK, "LINKER_DESTROY_FLAG");

  auto check_if_exists = [](const stk::mesh::FieldBase *const field_ptr, const std::string &name) {
    MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                       name + "cannot be a nullptr. Check that the field exists.");
  };

  check_if_exists(node_coordinates_field_ptr, "NODE_COORDS");
  check_if_exists(node_velocity_field_ptr, "NODE_VELOCITY");
  check_if_exists(node_force_field_ptr, "NODE_FORCE");
  check_if_exists(element_radius_field_ptr, "ELEMENT_RADIUS");
  check_if_exists(element_youngs_modulus_field_ptr, "ELEMENT_YOUNGS_MODULUS");
  check_if_exists(element_poissons_ratio_field_ptr, "ELEMENT_POISSONS_RATIO");
  check_if_exists(element_rest_length_field_ptr, "ELEMENT_REST_LENGTH");
  check_if_exists(element_spring_constant_field_ptr, "ELEMENT_SPRING_CONSTANT");
  check_if_exists(linker_contact_normal_field_ptr, "LINKER_CONTACT_NORMAL");
  check_if_exists(linker_signed_separation_distance_field_ptr, "LINKER_SIGNED_SEPARATION_DISTANCE");
  check_if_exists(linker_potential_force_field_ptr, "LINKER_POTENTIAL_FORCE");
  check_if_exists(linker_destroy_flag_field_ptr, "LINKER_DESTROY_FLAG");
  check_if_exists(element_angular_spring_rest_angle_field_ptr, "ELEMENT_ANGULAR_SPRING_REST_ANGLE");
  check_if_exists(element_angular_spring_constant_field_ptr, "ELEMENT_ANGULAR_SPRING_CONSTANT");

  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part(mundy::shapes::Spheres::get_name());
  MUNDY_THROW_REQUIRE(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");
  stk::mesh::Part &spheres_part = *spheres_part_ptr;
  stk::io::put_io_part_attribute(spheres_part);

  stk::mesh::Part *spherocylinder_segments_part_ptr =
      meta_data_ptr->get_part(mundy::shapes::SpherocylinderSegments::get_name());
  MUNDY_THROW_REQUIRE(spherocylinder_segments_part_ptr != nullptr, std::invalid_argument,
                     "SPHEROCYLINDER_SEGMENTS part not found.");
  stk::mesh::Part &spherocylinder_segments_part = *spherocylinder_segments_part_ptr;
  stk::io::put_io_part_attribute(spherocylinder_segments_part);

  stk::mesh::Part *spherocylinder_segment_spherocylinder_segment_linkers_part_ptr = meta_data_ptr->get_part(
      mundy::linkers::neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::get_name());
  MUNDY_THROW_REQUIRE(spherocylinder_segment_spherocylinder_segment_linkers_part_ptr != nullptr, std::invalid_argument,
                     "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKERS part not found.");
  stk::mesh::Part &spherocylinder_segment_spherocylinder_segment_linkers_part =
      *spherocylinder_segment_spherocylinder_segment_linkers_part_ptr;
  stk::io::put_io_part_attribute(spherocylinder_segment_spherocylinder_segment_linkers_part);

  stk::mesh::Part *springs_part_ptr = meta_data_ptr->get_part(mundy::constraints::HookeanSprings::get_name());
  MUNDY_THROW_REQUIRE(springs_part_ptr != nullptr, std::invalid_argument, "HOOKEAN_SPRINGS part not found.");
  stk::mesh::Part &springs_part = *springs_part_ptr;
  stk::io::put_io_part_attribute(springs_part);

  stk::mesh::Part *angular_springs_part_ptr = meta_data_ptr->get_part(mundy::constraints::AngularSprings::get_name());
  MUNDY_THROW_REQUIRE(angular_springs_part_ptr != nullptr, std::invalid_argument, "ANGULAR_SPRINGS part not found.");
  stk::mesh::Part &angular_springs_part = *angular_springs_part_ptr;
  stk::io::put_io_part_attribute(angular_springs_part);

  ///////////////////
  // Setup our IO  //
  ///////////////////

  stk::io::StkMeshIoBroker stk_io_broker;
  stk_io_broker.use_simple_fields();
  stk_io_broker.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));

  size_t output_file_index = stk_io_broker.create_output_mesh("Springs.exo", stk::io::WRITE_RESULTS);
  stk_io_broker.add_field(output_file_index, *node_coordinates_field_ptr);
  stk_io_broker.add_field(output_file_index, *node_velocity_field_ptr);
  stk_io_broker.add_field(output_file_index, *node_force_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_radius_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_youngs_modulus_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_poissons_ratio_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_rest_length_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_spring_constant_field_ptr);
  stk_io_broker.add_field(output_file_index, *linker_contact_normal_field_ptr);
  stk_io_broker.add_field(output_file_index, *linker_signed_separation_distance_field_ptr);
  stk_io_broker.add_field(output_file_index, *linker_potential_force_field_ptr);
  stk_io_broker.add_field(output_file_index, *linker_destroy_flag_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_angular_spring_rest_angle_field_ptr);
  stk_io_broker.add_field(output_file_index, *element_angular_spring_constant_field_ptr);

  //////////////////////////////////////
  // Initialize the spheres and nodes //
  //////////////////////////////////////

  // Declare N spring chains with a slight shift to each chain
  const int num_chains = 10;
  for (int i = 0; i < num_chains; i++) {
    // DeclareAndInitConstraints mutable parameters
    Teuchos::ParameterList declare_and_init_constraints_mutable_params;
    using CoordinateMappingType =
        mundy::constraints::declare_and_initialize_constraints::techniques::ArchlengthCoordinateMapping;
    using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::StraightLine;

    double center_x = 0.0;
    double center_y = 2 * i * sphere_radius;
    double center_z = 0.0;
    double length = num_spheres * initial_segment_length;
    double axis_x = 1.0;
    double axis_y = 0.0;
    double axis_z = 0.0;
    auto straight_line_mapping_ptr = std::make_shared<OurCoordinateMappingType>(
        num_spheres, center_x, center_y, center_z, length, axis_x, axis_y, axis_z);
    declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
        .set<size_t>("num_nodes", num_spheres)
        .set<size_t>("node_id_start", i * num_spheres + 1)
        .set<size_t>("element_id_start", i * (4 * num_spheres - 4) + 1)
        .set("hookean_spring_constant", spring_constant)
        .set("hookean_spring_rest_length", rest_length)
        .set("angular_spring_constant", angular_spring_constant)
        .set("angular_spring_rest_angle", angular_spring_rest_angle)
        .set("sphere_radius", sphere_radius)
        .set("spherocylinder_segment_radius", sphere_radius)
        .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", straight_line_mapping_ptr);
    declare_and_init_constraints_ptr->set_mutable_params(declare_and_init_constraints_mutable_params);
    declare_and_init_constraints_ptr->execute();
  }

  mundy::mesh::utils::fill_field_with_value<double>(*element_youngs_modulus_field_ptr,
                                                    std::array<double, 1>{youngs_modulus});
  mundy::mesh::utils::fill_field_with_value<double>(*element_poissons_ratio_field_ptr,
                                                    std::array<double, 1>{poissons_ratio});

  ////////////////////////
  // Balancing the mesh //
  ////////////////////////
  if (loadbalance_initial_config) {
    RcbSettings balanceSettings;
    stk::balance::balanceStkMesh(balanceSettings, *static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()));
  }

  ////////////////////////
  // Run the simulation //
  ////////////////////////

  // // Write the initial mesh to file
  // stk_io_broker.begin_output_step(output_file_index, 0.0);
  // stk_io_broker.write_defined_output_fields(output_file_index);
  // stk_io_broker.end_output_step(output_file_index);
  // stk_io_broker.flush_output();

  if (bulk_data_ptr->parallel_rank() == 0) {
    std::cout << "Running the simulation for " << num_time_steps << " time steps." << std::endl;
  }

  Kokkos::Timer timer;
  for (size_t i = 0; i < num_time_steps; i++) {
    // Rotate the field states.
    bulk_data.update_field_data_states();

    // Output
    if (i % 10000 == 0) {
      stk_io_broker.begin_output_step(output_file_index, static_cast<double>(i));
      stk_io_broker.write_defined_output_fields(output_file_index);
      stk_io_broker.end_output_step(output_file_index);
      stk_io_broker.flush_output();
    }

    // Setup
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr, std::array{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr, std::array{0.0, 0.0, 0.0});

    // Potentials
    compute_constraint_forcing_ptr->execute(stk::mesh::Selector(springs_part) |
                                            stk::mesh::Selector(angular_springs_part));

    // Collisions
    if (i % 100 == 0) {
      compute_aabb_ptr->execute(spherocylinder_segments_part);
      destroy_neighbor_linkers_ptr->execute(spherocylinder_segment_spherocylinder_segment_linkers_part);
      generate_neighbor_linkers_ptr->execute(spherocylinder_segments_part, spherocylinder_segments_part);
    }
    compute_ssd_and_cn_ptr->execute(spherocylinder_segment_spherocylinder_segment_linkers_part);
    evaluate_linker_potentials_ptr->execute(spherocylinder_segment_spherocylinder_segment_linkers_part);
    linker_potential_force_reduction_ptr->execute(spherocylinder_segments_part);

    // Motion
    compute_mobility_ptr->execute(spheres_part);
    node_euler_ptr->execute(spheres_part);
  }

  // Do a synchronize to force everybody to stop here, then write the time
  stk::parallel_machine_barrier(bulk_data_ptr->parallel());

  if (bulk_data_ptr->parallel_rank() == 0) {
    double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps);
    std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
  }

  // Write the final mesh to file
  // stk_io_broker.begin_output_step(output_file_index, static_cast<double>(num_time_steps));
  // stk_io_broker.write_defined_output_fields(output_file_index);
  // stk_io_broker.end_output_step(output_file_index);
  // stk_io_broker.flush_output();

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
