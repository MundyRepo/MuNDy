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

Our goal is to simulate N Brownian diffusing spheres in a 3D domain with Hertzian contact. We will use the following
free parameters:
  - number_of_particles
  - length_of_domain
  - total_time
  - timestep_size
  - diffusion_coeff

Each timestep will consist of
  1. Compute the Brownian contribution to the velocity of each particle
  2. Update the position of each particle using a first order Euler timestep

We'll need two MetaMethods: one for computing the brownian motion and one for taking the timestep.
*/

// #define ENABLE_STKFMM

// External libs
#include <fmt/format.h>  // for fmt::format
#include <openrand/philox.h>

#include <MundyAlens_config.hpp>  // for HAVE_MUNDYALENS_*

#ifdef HAVE_MUNDYALENS_STKFMM
#include <STKFMM/STKFMM.hpp>
#endif

// Trilinos libs
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
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
#include <mundy_linkers/DestroyNeighborLinkers.hpp>           // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>         // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>          // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>    // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                  // for mundy::linkers::NeighborLinkers
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>                       // adds fmt::format for stk types
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
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

class NodeEuler
    : public mundy::meta::MetaKernelDispatcher<NodeEuler, mundy::meta::make_registration_string("NODE_EULER")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  NodeEuler() = delete;

  /// \brief Constructor
  NodeEuler(mundy::mesh::BulkData *const bulk_data_ptr,
            const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaKernelDispatcher<NodeEuler, mundy::meta::make_registration_string("NODE_EULER")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set(
        "node_velocity_field_name", std::string(default_node_velocity_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("timestep_size", default_timestep_size_, "The timestep size.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_timestep_size_ = 0.0;
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}
};  // NodeEuler

/// \class NodeEulerSphere
/// \brief Concrete implementation of \c MetaKernel for computing the node euler timestep of spheres.
class NodeEulerSphere : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit NodeEulerSphere(mundy::mesh::BulkData *const bulk_data_ptr,
                           const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                        "NodeEulerSphere: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(NodeEulerSphere::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                          std::string("NodeEulerSphere: Part '") + part_name +
                              "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_coord_field_name = mundy::shapes::Spheres::get_node_coord_field_name();
    const std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");

    node_coordinates_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
    node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name);
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
    valid_fixed_params.validateParametersAndSetDefaults(NodeEulerSphere::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs<double>(node_velocity_field_name, stk::topology::NODE_RANK, 3, 1);

      if (part_name == mundy::shapes::Spheres::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("valid_entity_part_names", mundy::core::make_string_array(default_part_name_),
                               "Name of the parts associated with this kernel.");
    default_parameter_list.set(
        "node_velocity_field_name", std::string(default_node_velocity_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("timestep_size", default_timestep_size_, "The timestep size.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<NodeEulerSphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(NodeEulerSphere::get_valid_mutable_params());
    timestep_size_ = valid_mutable_params.get<double>("timestep_size");

    MUNDY_THROW_REQUIRE(timestep_size_ > 0.0, std::invalid_argument,
                        "NodeEulerSphere: timestep_size must be greater than zero.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Selector &sphere_selector) {
    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_coord_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    double timestep_size = timestep_size_;

    stk::mesh::Selector intersection_with_valid_entity_parts =
        stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() & sphere_selector;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, intersection_with_valid_entity_parts,
        [&node_coord_field, &node_velocity_field, &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                                  const stk::mesh::Entity &sphere_node) {
          double *node_coords = stk::mesh::field_data(node_coord_field, sphere_node);
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          node_coords[0] += timestep_size * node_velocity[0];
          node_coords[1] += timestep_size * node_velocity[1];
          node_coords[2] += timestep_size * node_velocity[2];
        });
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_timestep_size_ = 0.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The numerical timestep size.
  double timestep_size_;

  /// \brief Node field containing the node's spatial coordinate.
  stk::mesh::Field<double> *node_coordinates_field_ptr_ = nullptr;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;
  //@}
};  // NodeEulerSphere

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_node_euler_kernels_ = []() {
  // Register our default kernels
  NodeEuler::OurKernelFactory::register_new_class<NodeEulerSphere>("SPHERE");
  return true;
}();

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

class ComputeBrownianVelocity
    : public mundy::meta::MetaKernelDispatcher<ComputeBrownianVelocity,
                                               mundy::meta::make_registration_string("COMPUTE_BROWNIAN_VELOCITY")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeBrownianVelocity() = delete;

  /// \brief Constructor
  ComputeBrownianVelocity(mundy::mesh::BulkData *const bulk_data_ptr,
                          const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaKernelDispatcher<ComputeBrownianVelocity,
                                          mundy::meta::make_registration_string("COMPUTE_BROWNIAN_VELOCITY")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_brownian_velocity_field_name",
                               std::string(default_node_brownian_velocity_field_name_),
                               "Name of the node broenian velocity field.");
    default_parameter_list.set("node_rng_counter_field_name", std::string(default_node_rng_counter_field_name_),
                               "Name of the node rng counter for generating new parallel streams.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("timestep_size", default_timestep_size_, "The timestep size.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_timestep_size_ = 0.0;
  static constexpr std::string_view default_node_brownian_velocity_field_name_ = "NODE_BROWNIAN_VELOCITY";
  static constexpr std::string_view default_node_rng_counter_field_name_ = "NODE_RNG_COUNTER";
  //@}
};  // ComputeBrownianVelocity

/// \class ComputeBrownianVelocitySphere
/// \brief Concrete implementation of \c MetaKernel for computing the node euler timestep of spheres.
class ComputeBrownianVelocitySphere : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeBrownianVelocitySphere(mundy::mesh::BulkData *const bulk_data_ptr,
                                         const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                        "ComputeBrownianVelocitySphere: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(ComputeBrownianVelocitySphere::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                          std::string("ComputeBrownianVelocitySphere: Part '") + part_name +
                              "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_brownian_velocity_field_name =
        valid_fixed_params.get<std::string>("node_brownian_velocity_field_name");
    const std::string node_rng_counter_field_name = valid_fixed_params.get<std::string>("node_rng_counter_field_name");

    node_brownian_velocity_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_brownian_velocity_field_name);
    node_rng_counter_field_ptr_ =
        meta_data_ptr_->get_field<unsigned>(stk::topology::NODE_RANK, node_rng_counter_field_name);
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
    valid_fixed_params.validateParametersAndSetDefaults(ComputeBrownianVelocitySphere::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    std::string node_brownian_velocity_field_name =
        valid_fixed_params.get<std::string>("node_brownian_velocity_field_name");
    std::string node_rng_counter_field_name = valid_fixed_params.get<std::string>("node_rng_counter_field_name");
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs<double>(node_brownian_velocity_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<unsigned>(node_rng_counter_field_name, stk::topology::NODE_RANK, 1, 1);

      if (part_name == mundy::shapes::Spheres::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("valid_entity_part_names", mundy::core::make_string_array(default_part_name_),
                               "Name of the parts associated with this kernel.");
    default_parameter_list.set(
        "node_brownian_velocity_field_name", std::string(default_node_brownian_velocity_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    default_parameter_list.set(
        "node_rng_counter_field_name", std::string(default_node_rng_counter_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("timestep_size", default_timestep_size_, "The timestep size.");
    default_parameter_list.set("diffusion_coeff", default_diffusion_coeff_, "The diffusion coefficient.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<ComputeBrownianVelocitySphere>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(ComputeBrownianVelocitySphere::get_valid_mutable_params());
    timestep_size_ = valid_mutable_params.get<double>("timestep_size");
    diffusion_coeff_ = valid_mutable_params.get<double>("diffusion_coeff");

    MUNDY_THROW_REQUIRE(timestep_size_ > 0.0, std::invalid_argument,
                        "ComputeBrownianVelocitySphere: timestep_size must be greater than zero.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Selector &sphere_selector) {
    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_brownian_velocity_field = *node_brownian_velocity_field_ptr_;
    stk::mesh::Field<unsigned> &node_rng_counter_field = *node_rng_counter_field_ptr_;
    double timestep_size = timestep_size_;
    double diffusion_coeff = diffusion_coeff_;

    stk::mesh::Selector intersection_with_valid_entity_parts =
        stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() & sphere_selector;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, intersection_with_valid_entity_parts,
        [&node_brownian_velocity_field, &node_rng_counter_field, &timestep_size, &diffusion_coeff](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          double *node_brownian_velocity = stk::mesh::field_data(node_brownian_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_counter_field, sphere_node);

          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * diffusion_coeff / timestep_size);
          node_brownian_velocity[0] += coeff * rng.randn<double>();
          node_brownian_velocity[1] += coeff * rng.randn<double>();
          node_brownian_velocity[2] += coeff * rng.randn<double>();
          node_rng_counter[0]++;
        });
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_timestep_size_ = 0.0;
  static inline double default_diffusion_coeff_ = 0.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_brownian_velocity_field_name_ = "NODE_BROWNIAN_VELOCITY";
  static constexpr std::string_view default_node_rng_counter_field_name_ = "NODE_RNG_COUNTER";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The numerical timestep size.
  double timestep_size_;

  /// \brief The diffusion coefficient.
  double diffusion_coeff_;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_brownian_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's random number generator counter.
  stk::mesh::Field<unsigned> *node_rng_counter_field_ptr_ = nullptr;
  //@}
};  // ComputeBrownianVelocitySphere

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_brownian_velocity_kernels_ = []() {
  // Register our default kernels
  ComputeBrownianVelocity::OurKernelFactory::register_new_class<ComputeBrownianVelocitySphere>("SPHERE");
  return true;
}();

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

class ComputeMobility
    : public mundy::meta::MetaMethodSubsetExecutionDispatcher<ComputeMobility, void,
                                                              mundy::meta::make_registration_string("COMPUTE_MOBILITY"),
                                                              mundy::meta::make_registration_string("LOCAL_DRAG")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeMobility() = delete;

  /// \brief Constructor
  ComputeMobility(mundy::mesh::BulkData *const bulk_data_ptr,
                  const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaMethodSubsetExecutionDispatcher<ComputeMobility, void,
                                                         mundy::meta::make_registration_string("COMPUTE_MOBILITY"),
                                                         mundy::meta::make_registration_string("LOCAL_DRAG")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaTechniqueDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                               "Name of the node force field.");
    default_parameter_list.set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                               "Name of the node velocity field.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}
};  // ComputeMobility

class LocalDrag
    : public mundy::meta::MetaKernelDispatcher<LocalDrag, mundy::meta::make_registration_string("LOCAL_DRAG")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  LocalDrag() = delete;

  /// \brief Constructor
  LocalDrag(mundy::mesh::BulkData *const bulk_data_ptr,
            const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaKernelDispatcher<LocalDrag, mundy::meta::make_registration_string("LOCAL_DRAG")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our kernels have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our kernels have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                               "Name of the node force field.");
    default_parameter_list.set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                               "Name of the node velocity field.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("viscosity", default_viscosity_, "The fluid viscosity.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_viscosity_ = 1.0;
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}
};  // LocalDrag

class LocalDragNonorientableSphere : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit LocalDragNonorientableSphere(mundy::mesh::BulkData *const bulk_data_ptr,
                                        const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                        "LocalDragNonorientableSphere: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(LocalDragNonorientableSphere::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                          std::string("LocalDragNonorientableSphere: Part '") + part_name +
                              "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    const std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();

    node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
    node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name);
    element_radius_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
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
    valid_fixed_params.validateParametersAndSetDefaults(LocalDragNonorientableSphere::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs<double>(node_force_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(node_velocity_field_name, stk::topology::NODE_RANK, 3, 1);

      if (part_name == mundy::shapes::Spheres::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("valid_entity_part_names", mundy::core::make_string_array(default_part_name_),
                               "Name of the parts associated with this kernel.");
    default_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                               "Name of the node force field.");
    default_parameter_list.set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                               "Name of the node velocity field.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("viscosity", default_viscosity_, "The fluid viscosity.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<LocalDragNonorientableSphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(LocalDragNonorientableSphere::get_valid_mutable_params());
    viscosity_ = valid_mutable_params.get<double>("viscosity");

    MUNDY_THROW_REQUIRE(viscosity_ > 0.0, std::invalid_argument,
                        "LocalDragNonorientableSphere: viscosity must be greater than zero.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Selector &sphere_selector) {
    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    double viscosity = viscosity_;

    stk::mesh::Selector intersection_with_valid_entity_parts =
        stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() & sphere_selector;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, intersection_with_valid_entity_parts,
        [&node_force_field, &node_velocity_field, &element_radius_field, &viscosity](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          const stk::mesh::Entity &node = bulk_data.begin_nodes(sphere_element)[0];

          const double *element_radius = stk::mesh::field_data(element_radius_field, sphere_element);
          const double *node_force = stk::mesh::field_data(node_force_field, node);
          double *node_velocity = stk::mesh::field_data(node_velocity_field, node);
          const double inv_drag_coeff = 1.0 / (6.0 * M_PI * viscosity * element_radius[0]);
          node_velocity[0] += inv_drag_coeff * node_force[0];
          node_velocity[1] += inv_drag_coeff * node_force[1];
          node_velocity[2] += inv_drag_coeff * node_force[2];
        });
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_viscosity_ = 1.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The fluid viscosity.
  double viscosity_;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's force.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Element field containing the sphere's radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // LocalDragNonorientableSphere

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_local_drag_kernels_ = []() {
  // Register our default kernels
  LocalDrag::OurKernelFactory::register_new_class<LocalDragNonorientableSphere>("NONORIENTABLE_SPHERE");
  return true;
}();

#ifdef HAVE_MUNDYALENS_STKFMM
class RPYSphere : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit RPYSphere(mundy::mesh::BulkData *const bulk_data_ptr,
                     const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                        "RPYSphere: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(RPYSphere::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                          std::string("RPYSphere: Part '") + part_name +
                              "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_coord_field_name = mundy::shapes::Spheres::get_node_coord_field_name();
    const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    const std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();

    node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
    node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
    node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name);
    element_radius_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
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
    valid_fixed_params.validateParametersAndSetDefaults(RPYSphere::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs<double>(node_force_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(node_velocity_field_name, stk::topology::NODE_RANK, 3, 1);

      if (part_name == mundy::shapes::Spheres::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static auto default_parameter_list =
        Teuchos::ParameterList()
            .set("valid_entity_part_names", mundy::core::make_string_array(default_part_name_),
                 "Name of the parts associated with this kernel.")
            .set("node_force_field_name", std::string(default_node_force_field_name_), "Name of the node force field.")
            .set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                 "Name of the node velocity field.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static auto default_parameter_list =
        Teuchos::ParameterList()
            .set("viscosity", default_viscosity_, "The fluid viscosity.")
            .set("fmm_multipole_order", default_fmm_multipole_order_, "The multipole order for the FMM.")
            .set("max_num_leaf_pts", default_max_num_leaf_pts_, "The maximum number of leaf points for the FMM.")
            .set("periodic_in_x", default_periodic_in_x_, "Whether the domain is periodic in the x direction.")
            .set("periodic_in_y", default_periodic_in_y_, "Whether the domain is periodic in the y direction.")
            .set("periodic_in_z", default_periodic_in_z_, "Whether the domain is periodic in the z direction.")
            .set<Teuchos::Array<double>>(
                "domain_origin",
                Teuchos::tuple<double>(default_domain_origin_[0], default_domain_origin_[1], default_domain_origin_[2]),
                "The origin of the domain.")
            .set("domain_length", default_domain_length_, "The length of the domain.");

    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<RPYSphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(RPYSphere::get_valid_mutable_params());
    viscosity_ = valid_mutable_params.get<double>("viscosity");
    fmm_multipole_order_ = valid_mutable_params.get<int>("fmm_multipole_order");
    max_num_leaf_pts_ = valid_mutable_params.get<int>("max_num_leaf_pts");
    periodic_in_x_ = valid_mutable_params.get<bool>("periodic_in_x");
    periodic_in_y_ = valid_mutable_params.get<bool>("periodic_in_y");
    periodic_in_z_ = valid_mutable_params.get<bool>("periodic_in_z");
    auto domain_origin_array = valid_mutable_params.get<Teuchos::Array<double>>("domain_origin");
    domain_origin = {domain_origin_array[0], domain_origin_array[1], domain_origin_array[2]};
    domain_length = valid_mutable_params.get<double>("domain_length");

    MUNDY_THROW_REQUIRE(viscosity_ > 0.0, std::invalid_argument, "RPYSphere: viscosity must be greater than zero.");
    MUNDY_THROW_REQUIRE(fmm_multipole_order_ > 0, std::invalid_argument,
                        "RPYSphere: fmm_multipole_order must be greater than zero.");
    MUNDY_THROW_REQUIRE(max_num_leaf_pts_ > 0, std::invalid_argument,
                        "RPYSphere: max_num_leaf_pts must be greater than zero.");
    MUNDY_THROW_REQUIRE(domain_length > 0.0, std::invalid_argument,
                        "RPYSphere: domain_length must be greater than zero.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Selector &sphere_selector) {
    std::cout << "RPYSphere::execute" << std::endl;

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    double viscosity = viscosity_;
    double Pi = 3.14159265358979323846;

    // Get the total number of local spheres
    stk::mesh::Selector intersection_with_valid_entity_parts =
        stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() & sphere_selector;
    stk::mesh::EntityVector local_spheres;
    stk::mesh::get_selected_entities(intersection_with_valid_entity_parts,
                                     bulk_data.buckets(stk::topology::ELEMENT_RANK), local_spheres);
    const int num_local_spheres = local_spheres.size();

    // Setup our periodic boundary conditions
    stkfmm::PAXIS fmm_pbc;
    if (!periodic_in_x_ && !periodic_in_y_ && !periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::NONE;
    } else if (periodic_in_x_ && !periodic_in_y_ && !periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::PX;
    } else if (periodic_in_x_ && periodic_in_y_ && !periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::PXY;
    } else if (periodic_in_x_ && periodic_in_y_ && periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::PXYZ;
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          fmt::format("Unsupported pbc configuration. The current configuration is periodic_in_x = {}, "
                                      "periodic_in_y = {}, periodic_in_z = {}",
                                      periodic_in_x_, periodic_in_y_, periodic_in_z_));
    }
    std::cout << "RPYSphere::execute pbc setup" << std::endl;

    // Initialize the FMM evaluator
    std::cout << "fmm_multipole_order: " << fmm_multipole_order_ << " | max_num_leaf_pts: " << max_num_leaf_pts_
              << std::endl;

    auto fmm_evaluator =
        stkfmm::Stk3DFMM(fmm_multipole_order_, max_num_leaf_pts_, fmm_pbc, stkfmm::asInteger(stkfmm::KERNEL::RPY));

    std::cout << "Domain origin: " << default_domain_origin_[0] << " " << default_domain_origin_[1] << " "
              << default_domain_origin_[2] << std::endl;
    std::cout << "Domain length: " << default_domain_length_ << std::endl;
    fmm_evaluator.setBox(domain_origin.data(), domain_length);
    std::cout << "fmm_evaluator initialized" << std::endl;

    // Setup the source and target points
    std::vector<double> src_single_layer_coord(3 * num_local_spheres);
    std::vector<double> trg_coord(3 * num_local_spheres);
    std::vector<double> src_single_layer_value(4 * num_local_spheres);
    std::vector<double> trg_value(6 * num_local_spheres, 0.0);

#pragma omp parallel for
    for (size_t i = 0; i < num_local_spheres; i++) {
      const auto &sphere = local_spheres[i];
      const auto &node = bulk_data.begin_nodes(sphere)[0];

      const double radius = stk::mesh::field_data(element_radius_field, sphere)[0];
      const double *node_coord = stk::mesh::field_data(node_coord_field, node);
      const double *node_force = stk::mesh::field_data(node_force_field, node);

      const bool coordinate_out_of_domain_in_x =
          periodic_in_x_ ? false
                         : ((node_coord[0] < domain_origin[0]) || (node_coord[0] >= domain_origin[0] + domain_length));
      const bool coordinate_out_of_domain_in_y =
          periodic_in_y_ ? false
                         : ((node_coord[1] < domain_origin[1]) || (node_coord[1] >= domain_origin[1] + domain_length));
      const bool coordinate_out_of_domain_in_z =
          periodic_in_z_ ? false
                         : ((node_coord[2] < domain_origin[2]) || (node_coord[2] >= domain_origin[2] + domain_length));
      const bool coordinate_out_of_domain_in_non_periodic_direction =
          coordinate_out_of_domain_in_x || coordinate_out_of_domain_in_y || coordinate_out_of_domain_in_z;
      MUNDY_THROW_REQUIRE(!coordinate_out_of_domain_in_non_periodic_direction, std::logic_error,
                          fmt::format("RPYSphere: Node coordinate is out of domain. The current coordinate is {} {} {} "
                                      "and the domain is {} {} {} with length {}",
                                      node_coord[0], node_coord[1], node_coord[2], domain_origin[0], domain_origin[1],
                                      domain_origin[2], domain_length));

      src_single_layer_coord[3 * i] = node_coord[0];
      src_single_layer_coord[3 * i + 1] = node_coord[1];
      src_single_layer_coord[3 * i + 2] = node_coord[2];
      trg_coord[3 * i] = node_coord[0];
      trg_coord[3 * i + 1] = node_coord[1];
      trg_coord[3 * i + 2] = node_coord[2];

      src_single_layer_value[4 * i] = node_force[0];
      src_single_layer_value[4 * i + 1] = node_force[1];
      src_single_layer_value[4 * i + 2] = node_force[2];
      src_single_layer_value[4 * i + 3] = radius;
    }
    fmm_evaluator.setPoints(num_local_spheres, src_single_layer_coord.data(), num_local_spheres, trg_coord.data());
    fmm_evaluator.setupTree(stkfmm::KERNEL::RPY);
    std::cout << "fmm_evaluator tree setup" << std::endl;

    // Evaluate the FMM
    fmm_evaluator.clearFMM(stkfmm::KERNEL::RPY);
    fmm_evaluator.evaluateFMM(stkfmm::KERNEL::RPY, num_local_spheres, src_single_layer_value.data(), num_local_spheres,
                              trg_value.data());
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    std::cout << "fmm_evaluator evaluated" << std::endl;

    // Update the node velocities
#pragma omp parallel for
    for (size_t i = 0; i < num_local_spheres; i++) {
      const auto &sphere = local_spheres[i];
      const auto &node = bulk_data.begin_nodes(sphere)[0];

      const double radius = stk::mesh::field_data(element_radius_field, sphere)[0];
      const double *node_force = stk::mesh::field_data(node_force_field, node);
      double *node_velocity = stk::mesh::field_data(node_velocity_field, node);

      // Convert the target values to the external velocity
      const double rpyfac = radius * radius / 6.0;
      node_velocity[0] += (trg_value[6 * i + 0] + rpyfac * trg_value[6 * i + 3]) / viscosity;
      node_velocity[1] += (trg_value[6 * i + 1] + rpyfac * trg_value[6 * i + 4]) / viscosity;
      node_velocity[2] += (trg_value[6 * i + 2] + rpyfac * trg_value[6 * i + 5]) / viscosity;

      // Add on the self-mobility
      const double inv_drag_trans = 1.0 / (6 * Pi * radius * viscosity);
      node_velocity[0] += inv_drag_trans * node_force[0];
      node_velocity[1] += inv_drag_trans * node_force[1];
      node_velocity[2] += inv_drag_trans * node_force[2];
    }

    std::cout << "RPYSphere::execute done" << std::endl;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_viscosity_ = 1.0;
  static inline int default_fmm_multipole_order_ = 8;
  static inline int default_max_num_leaf_pts_ = 2000;
  static inline bool default_periodic_in_x_ = false;
  static inline bool default_periodic_in_y_ = false;
  static inline bool default_periodic_in_z_ = false;
  static inline std::array<double, 3> default_domain_origin_ = {0.0, 0.0, 0.0};
  static inline double default_domain_length_ = 1.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The fluid viscosity.
  double viscosity_;

  /// \brief The order of the multipole expansion used by PVFMM
  int fmm_multipole_order_;

  /// \brief The maximum number of leaf points per octree node used by PVFMM
  int max_num_leaf_pts_;

  /// \brief Are we periodic in x, y, and z?
  bool periodic_in_x_ = false;
  bool periodic_in_y_ = false;
  bool periodic_in_z_ = false;

  /// \brief Bottom left corner of the domain
  std::array<double, 3> domain_origin;

  /// \brief The length of the domain
  double domain_length;

  /// \brief Node coord containing the node's position.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's force.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Element field containing the sphere's radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // RPYSphere

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_mobility_kernels_ = []() {
  // Register our default kernels
  ComputeMobility::OurTechniqueFactory::register_new_class<RPYSphere>("RPY_SPHERE");
  ComputeMobility::OurTechniqueFactory::register_new_class<LocalDrag>("LOCAL_DRAG");
  return true;
}();
#endif

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
  bool loadbalance_initial_config = false;

  size_t num_time_steps = 100;
  double timestep_size = 0.01;
  double diffusion_coeff = 1.0;
  double viscosity = 1.0;
  double youngs_modulus = 1000.0;
  double poissons_ratio = 0.3;
  double spring_constant = 1.0;
  double angular_spring_constant = 1.0;
  double angular_spring_rest_angle = M_PI;
  bool generate_hookean_springs = true;
  bool generate_angular_springs = false;

  bool consider_collisions = true;

  // Parse the command line options.
  Teuchos::CommandLineProcessor cmdp(false, true);

  // Optional command line arguments for controlling sphere initialization:
  cmdp.setOption("num_spheres", &num_spheres, "Number of spheres.");
  cmdp.setOption("sphere_radius", &sphere_radius, "The radius of the spheres.");
  cmdp.setOption("initial_segment_length", &initial_segment_length, "Initial segment length.");
  cmdp.setOption("rest_length", &rest_length, "Rest length of the spring.");
  cmdp.setOption("loadbalance", "no_loadbalance", &loadbalance_initial_config,
                 "Load balance the initial configuration.");
  cmdp.setOption("consider_collisions", "no_consider_collisions", &consider_collisions,
                 "Consider collisions between spheres.");

  // Optional command line arguments for controlling the simulation:
  cmdp.setOption("num_time_steps", &num_time_steps, "Number of time steps.");
  cmdp.setOption("timestep_size", &timestep_size, "Time step size.");
  cmdp.setOption("diffusion_coeff", &diffusion_coeff, "Diffusion coefficient.");
  cmdp.setOption("viscosity", &viscosity, "Viscosity.");
  cmdp.setOption("youngs_modulus", &youngs_modulus, "Young's modulus.");
  cmdp.setOption("poissons_ratio", &poissons_ratio, "Poisson's ratio.");
  cmdp.setOption("spring_constant", &spring_constant, "Spring constant.");
  cmdp.setOption("angular_spring_constant", &angular_spring_constant, "Angular spring constant.");
  cmdp.setOption("angular_spring_rest_angle", &angular_spring_rest_angle, "Angular spring rest angle.");
  cmdp.setOption("generate_hookean_springs", "no_generate_hookean_springs", &generate_hookean_springs,
                 "If we should generate Hookean springs or not.");
  cmdp.setOption("generate_angular_springs", "no_generate_angular_springs", &generate_angular_springs,
                 "If we should generate angular springs or not.");

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
    std::cout << "  generate_hookean_springs: " << generate_hookean_springs << std::endl;
    std::cout << "  generate_angular_springs: " << generate_angular_springs << std::endl;
    std::cout << "##################################################" << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  // Setup the fixed parameters and generate the corresponding class instances and mesh //
  ////////////////////////////////////////////////////////////////////////////////////////
  // IMPORTANT NOTE: Often, users will simply use the default fixed parameters with a small number of manual
  // overrides. However, in this example, we will explicitly set all fixed and mutable params for each class instance.

  // ComputeBrownianVelocity fixed parameters
  Teuchos::ParameterList compute_brownian_velocity_fixed_params;
  compute_brownian_velocity_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
      .set("node_rng_counter_field_name", "NODE_RNG_COUNTER")
      .set("node_brownian_velocity_field_name", "NODE_VELOCITY");
  compute_brownian_velocity_fixed_params.sublist("SPHERE").set("valid_entity_part_names",
                                                               mundy::core::make_string_array("SPHERES"));

  // NodeEuler fixed parameters
  Teuchos::ParameterList node_euler_fixed_params;
  node_euler_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
      .set("node_velocity_field_name", "NODE_VELOCITY");
  node_euler_fixed_params.sublist("SPHERE").set("valid_entity_part_names", mundy::core::make_string_array("SPHERES"));

  // ComputeMobility fixed parameters
  // Teuchos::ParameterList compute_mobility_fixed_params;
  // compute_mobility_fixed_params.set("enabled_technique_name", "LOCAL_DRAG")
  //     .set("node_force_field_name", "NODE_FORCE")
  //     .set("node_velocity_field_name", "NODE_VELOCITY");
  // compute_mobility_fixed_params.sublist("LOCAL_DRAG")
  //     .set("enabled_kernel_names", mundy::core::make_string_array("NONORIENTABLE_SPHERE"));
  // compute_mobility_fixed_params.sublist("LOCAL_DRAG")
  //     .sublist("NONORIENTABLE_SPHERE")
  //     .set("valid_entity_part_names", mundy::core::make_string_array("SPHERES"));
  Teuchos::ParameterList compute_mobility_fixed_params;
  compute_mobility_fixed_params.set("enabled_technique_name", "RPY_SPHERE")
      .set("node_force_field_name", "NODE_FORCE")
      .set("node_velocity_field_name", "NODE_VELOCITY");
  compute_mobility_fixed_params.sublist("RPY_SPHERE")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHERES"));

  // ComputeConstraintForcing fixed parameters
  Teuchos::ParameterList compute_constraint_forcing_fixed_params;
  compute_constraint_forcing_fixed_params.set(
      "enabled_kernel_names", mundy::core::make_string_array(mundy::constraints::HookeanSprings::get_name(),
                                                             mundy::constraints::AngularSprings::get_name()));
  compute_constraint_forcing_fixed_params.sublist("HOOKEAN_SPRINGS").set("node_force_field_name", "NODE_FORCE");
  compute_constraint_forcing_fixed_params.sublist("ANGULAR_SPRINGS").set("node_force_field_name", "NODE_FORCE");

  // ComputeSignedSeparationDistanceAndContactNormal fixed parameters
  Teuchos::ParameterList compute_ssd_and_cn_fixed_params;
  compute_ssd_and_cn_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKER"))
      .set("linker_contact_normal_field_name", "LINKER_CONTACT_NORMAL")
      .set("linker_signed_separation_distance_field_name", "LINKER_SIGNED_SEPARATION_DISTANCE");
  compute_ssd_and_cn_fixed_params.sublist("SPHERE_SPHERE_LINKER")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"))
      .set("valid_sphere_part_names", mundy::core::make_string_array("SPHERES"));

  // ComputeAABB fixed parameters
  Teuchos::ParameterList compute_aabb_fixed_params;
  compute_aabb_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
      .set("element_aabb_field_name", "ELEMENT_AABB");
  compute_aabb_fixed_params.sublist("SPHERE").set("valid_entity_part_names", mundy::core::make_string_array("SPHERES"));

  // GenerateNeighborLinkers fixed parameters
  Teuchos::ParameterList generate_neighbor_linkers_fixed_params;
  generate_neighbor_linkers_fixed_params.set("enabled_technique_name", "STK_SEARCH")
      .set("specialized_neighbor_linkers_part_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"));
  generate_neighbor_linkers_fixed_params.sublist("STK_SEARCH")
      .set("valid_source_entity_part_names", mundy::core::make_string_array("SPHERES"))
      .set("valid_target_entity_part_names", mundy::core::make_string_array("SPHERES"))
      .set("element_aabb_field_name", std::string("ELEMENT_AABB"));

  // EvaluateLinkerPotentials fixed parameters
  Teuchos::ParameterList evaluate_linker_potentials_fixed_params;
  evaluate_linker_potentials_fixed_params.set("enabled_kernel_names",
                                              mundy::core::make_string_array("SPHERE_SPHERE_HERTZIAN_CONTACT"));
  evaluate_linker_potentials_fixed_params.sublist("SPHERE_SPHERE_HERTZIAN_CONTACT")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"))
      .set("valid_sphere_part_names", mundy::core::make_string_array("SPHERES"))
      .set("linker_potential_force_field_name", "LINKER_POTENTIAL_FORCE")
      .set("linker_signed_separation_distance_field_name", "LINKER_SIGNED_SEPARATION_DISTANCE")
      .set("element_youngs_modulus_field_name", "ELEMENT_YOUNGS_MODULUS")
      .set("element_poissons_ratio_field_name", "ELEMENT_POISSONS_RATIO");

  // LinkerPotentialForceReduction fixed parameters
  Teuchos::ParameterList linker_potential_force_reduction_fixed_params;
  linker_potential_force_reduction_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
      .set("name_of_linker_part_to_reduce_over", "SPHERE_SPHERE_LINKERS")
      .set("linker_potential_force_field_name", "LINKER_POTENTIAL_FORCE");
  linker_potential_force_reduction_fixed_params.sublist("SPHERE")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHERES"))
      .set("node_force_field_name", "NODE_FORCE");

  // DestroyNeighborLinkers fixed parameters
  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params = Teuchos::ParameterList();
  destroy_neighbor_linkers_fixed_params.set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS")
      .sublist("DESTROY_DISTANT_NEIGHBORS")
      .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
      .set("valid_connected_source_and_target_part_names", mundy::core::make_string_array("SPHERES"))
      .set("linker_destroy_flag_field_name", "LINKER_DESTROY_FLAG")
      .set("element_aabb_field_name", "ELEMENT_AABB");

  // DeclareAndInitConstraints fixed parameters
  Teuchos::ParameterList declare_and_init_constraints_fixed_params;
  declare_and_init_constraints_fixed_params.set("enabled_technique_name", "CHAIN_OF_SPRINGS")
      .sublist("CHAIN_OF_SPRINGS")
      .set("hookean_springs_part_names", mundy::core::make_string_array(mundy::constraints::HookeanSprings::get_name()))
      .set("angular_springs_part_names", mundy::core::make_string_array(mundy::constraints::AngularSprings::get_name()))
      .set("sphere_part_names", mundy::core::make_string_array(mundy::shapes::Spheres::get_name()))
      .set<bool>("generate_hookean_springs", generate_hookean_springs)
      .set<bool>("generate_angular_springs", generate_angular_springs)
      .set<bool>("generate_spheres_at_nodes", true);

  // Create the class instances and mesh based on the given fixed requirements.
  auto [compute_brownian_velocity_ptr, node_euler_ptr, compute_mobility_ptr, compute_constraint_forcing_ptr,
        compute_ssd_and_cn_ptr, compute_aabb_ptr, generate_neighbor_linkers_ptr, evaluate_linker_potentials_ptr,
        linker_potential_force_reduction_ptr, destroy_neighbor_linkers_ptr, declare_and_init_constraints_ptr,
        bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
          ComputeBrownianVelocity, NodeEuler, ComputeMobility, mundy::constraints::ComputeConstraintForcing,
          mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal, mundy::shapes::ComputeAABB,
          mundy::linkers::GenerateNeighborLinkers, mundy::linkers::EvaluateLinkerPotentials,
          mundy::linkers::LinkerPotentialForceReduction, mundy::linkers::DestroyNeighborLinkers,
          mundy::constraints::DeclareAndInitConstraints>(
          {compute_brownian_velocity_fixed_params, node_euler_fixed_params, compute_mobility_fixed_params,
           compute_constraint_forcing_fixed_params, compute_ssd_and_cn_fixed_params, compute_aabb_fixed_params,
           generate_neighbor_linkers_fixed_params, evaluate_linker_potentials_fixed_params,
           linker_potential_force_reduction_fixed_params, destroy_neighbor_linkers_fixed_params,
           declare_and_init_constraints_fixed_params});

  auto check_class_instance = [](auto &class_instance_ptr, const std::string &class_name) {
    MUNDY_THROW_REQUIRE(class_instance_ptr != nullptr, std::invalid_argument,
                        std::string("Failed to create class instance with name << ") + class_name + " >>.");
  };  // check_class_instance

  check_class_instance(compute_brownian_velocity_ptr, "ComputeBrownianVelocity");
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

  // ComputeBrownianVelocity mutable parameters
  Teuchos::ParameterList compute_brownian_velocity_mutable_params;
  compute_brownian_velocity_mutable_params.set("timestep_size", timestep_size)
      .sublist("SPHERE")
      .set("diffusion_coeff", diffusion_coeff);
  compute_brownian_velocity_ptr->set_mutable_params(compute_brownian_velocity_mutable_params);

  // NodeEuler mutable parameters
  Teuchos::ParameterList node_euler_mutable_params;
  node_euler_mutable_params.set("timestep_size", timestep_size);
  node_euler_ptr->set_mutable_params(node_euler_mutable_params);

  // ComputeMobility mutable parameters
  Teuchos::ParameterList compute_mobility_mutable_params;
  // compute_mobility_mutable_params.sublist("LOCAL_DRAG").set("viscosity", viscosity);
  compute_mobility_mutable_params.sublist("RPY_SPHERE")
      .set("viscosity", viscosity)
      .set("fmm_multipole_order", 8)
      .set("max_num_leaf_pts", 2000)
      .set("periodic_in_x", false)
      .set("periodic_in_y", false)
      .set("periodic_in_z", false)
      .set<Teuchos::Array<double>>("domain_origin", Teuchos::tuple<double>(-100.0, -100.0, -100.0),
                                   "The origin of the domain.")
      .set("domain_length", 200.0, "The length of the domain.");
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
  auto node_rng_counter_field_ptr = meta_data_ptr->get_field<unsigned>(stk::topology::NODE_RANK, "NODE_RNG_COUNTER");

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
  check_if_exists(node_rng_counter_field_ptr, "NODE_RNG_COUNTER");
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

  stk::mesh::Part *sphere_sphere_linkers_part_ptr =
      meta_data_ptr->get_part(mundy::linkers::neighbor_linkers::SphereSphereLinkers::get_name());
  MUNDY_THROW_REQUIRE(sphere_sphere_linkers_part_ptr != nullptr, std::invalid_argument,
                      "SPHERE_SPHERE_LINKERS part not found.");
  stk::mesh::Part &sphere_sphere_linkers_part = *sphere_sphere_linkers_part_ptr;
  stk::io::put_io_part_attribute(sphere_sphere_linkers_part);

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
  stk_io_broker.add_field(output_file_index, *node_rng_counter_field_ptr);
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
    using OurCoordinateMappingType = mundy::constraints::declare_and_initialize_constraints::techniques::Helix;
    double helical_radius = 10 * sphere_radius;
    double pitch = 2 * num_chains * sphere_radius;
    double distance_between_spheres = rest_length;
    double center_x = 2 * i * sphere_radius;
    double center_y = 0.0;
    double center_z = 0.0;
    double helical_axis_x = 1.0;
    double helical_axis_y = 0.0;
    double helical_axis_z = 0.0;
    auto levis_function_mapping_ptr = std::make_shared<OurCoordinateMappingType>(
        num_spheres, helical_radius, pitch, distance_between_spheres, center_x, center_y, center_z, helical_axis_x,
        helical_axis_y, helical_axis_z);
    declare_and_init_constraints_mutable_params.sublist("CHAIN_OF_SPRINGS")
        .set<size_t>("num_nodes", num_spheres)
        .set<size_t>("node_id_start", i * num_spheres + +1)
        .set<size_t>("element_id_start", i * (num_spheres + (num_spheres - 1) * generate_hookean_springs +
                                              (num_spheres - 2) * generate_angular_springs) +
                                             1)
        .set("hookean_spring_constant", spring_constant)
        .set("hookean_spring_rest_length", rest_length)
        .set("angular_spring_constant", angular_spring_constant)
        .set("angular_spring_rest_angle", angular_spring_rest_angle)
        .set("sphere_radius", sphere_radius)
        .set<std::shared_ptr<CoordinateMappingType>>("coordinate_mapping", levis_function_mapping_ptr);
    declare_and_init_constraints_ptr->set_mutable_params(declare_and_init_constraints_mutable_params);
    declare_and_init_constraints_ptr->execute();
  }

  mundy::mesh::utils::fill_field_with_value<unsigned>(*node_rng_counter_field_ptr, std::array<unsigned, 1>{0u});
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
    std::cout << "Time step: " << i << std::endl;
    // As a first pass, we will:
    //  - Zero out the node forces and velocities
    //  - Compute the AABB for the spheres
    //  - Delete SphereSphereLinkers that are too far apart
    //  - Generate SphereSphereLinkers neighbor linkers between nearby spheres
    //  - Compute the signed separation distance and contact normal for the SphereSphereLinkers
    //  - Evaluate the Hertzian contact potential for the SphereSphereLinkers
    //  - Reduce the linker potential force to the Sphere nodes
    //  - Compute the velocity induced by the node forces using local drag
    //  - Compute the brownian velocity for the nodes
    //  - Update the node positions using a first order Euler method

    // Output
    if (i % 1 == 0) {
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
    if (consider_collisions) {
      if (i % 100 == 0) {
        compute_aabb_ptr->execute(spheres_part);
        destroy_neighbor_linkers_ptr->execute(sphere_sphere_linkers_part);
        generate_neighbor_linkers_ptr->execute(spheres_part, spheres_part);
      }
      compute_ssd_and_cn_ptr->execute(sphere_sphere_linkers_part);
      evaluate_linker_potentials_ptr->execute(sphere_sphere_linkers_part);
      linker_potential_force_reduction_ptr->execute(spheres_part);
    }

    // Motion
    compute_mobility_ptr->execute(spheres_part);
    compute_brownian_velocity_ptr->execute(spheres_part);
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
