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

// Define a helper macros for turning on and off debug
// #define BROWNIAN_DEBUG_WRITE_MESH 1
// #define BROWNIAN_DEBUG_WRITE_MESH_INFO 1

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
#include <mundy_core/MakeStringArray.hpp>  // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>    // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>           // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>         // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>          // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>    // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                  // for mundy::linkers::NeighborLinkers
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
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

/*// A macro for a block of stuff
#define TIME_BLOCK(thing_to_time, rank, message)                          \
  {                                                                       \
    {                                                                     \
      Kokkos::Timer timer;                                                \
      thing_to_time;                                                      \
      double time = timer.seconds();                                      \
      if (rank == 0) {                                                    \
        std::cout << message << ": " << time << std::endl; \
      }                                                                   \
    }                                                                     \
  }
*/

// A macro for a block of stuff
#define TIME_BLOCK(thing_to_time, rank, message) \
  {{thing_to_time;                               \
  }                                              \
  }

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
    default_parameter_list.set("alpha", default_alpha_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
    default_parameter_list.set("beta", default_beta_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_timestep_size_ = 0.0;
  static constexpr double default_alpha_ = 1.0;
  static constexpr double default_beta_ = 0.0;
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
    default_parameter_list.set("alpha", default_alpha_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
    default_parameter_list.set("beta", default_beta_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
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
    alpha_ = valid_mutable_params.get<double>("alpha");
    beta_ = valid_mutable_params.get<double>("beta");

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
    double alpha = alpha_;
    double beta = beta_;

    stk::mesh::Selector intersection_with_valid_entity_parts =
        stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() & sphere_selector;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, intersection_with_valid_entity_parts,
        [&node_brownian_velocity_field, &node_rng_counter_field, &timestep_size, &diffusion_coeff, &alpha, &beta](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          double *node_brownian_velocity = stk::mesh::field_data(node_brownian_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_counter_field, sphere_node);

          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          node_brownian_velocity[0] = alpha * std::sqrt(2.0 * diffusion_coeff / timestep_size) * rng.randn<double>() +
                                      beta * node_brownian_velocity[0];
          node_brownian_velocity[1] = alpha * std::sqrt(2.0 * diffusion_coeff / timestep_size) * rng.randn<double>() +
                                      beta * node_brownian_velocity[1];
          node_brownian_velocity[2] = alpha * std::sqrt(2.0 * diffusion_coeff / timestep_size) * rng.randn<double>() +
                                      beta * node_brownian_velocity[2];
          node_rng_counter[0]++;
        });
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_timestep_size_ = 0.0;
  static inline double default_diffusion_coeff_ = 0.0;
  static constexpr double default_alpha_ = 1.0;
  static constexpr double default_beta_ = 0.0;
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

  /// \brief Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.
  double alpha_;

  /// \brief Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.
  double beta_;

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

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_mobility_kernels_ = []() {
  // Register our default kernels
  ComputeMobility::OurTechniqueFactory::register_new_class<LocalDrag>("LOCAL_DRAG");
  return true;
}();

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
static inline volatile const bool register_local_drag_kernels_ = []() {
  // Register our default kernels
  LocalDrag::OurKernelFactory::register_new_class<LocalDragNonorientableSphere>("NONORIENTABLE_SPHERE");
  return true;
}();

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

class ComputeConstraintForces
    : public mundy::meta::MetaKernelDispatcher<ComputeConstraintForces,
                                               mundy::meta::make_registration_string("COMPUTE_CONSTRAINT_FORCES")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintForces() = delete;

  /// \brief Constructor
  ComputeConstraintForces(mundy::mesh::BulkData *const bulk_data_ptr,
                          const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaKernelDispatcher<ComputeConstraintForces,
                                          mundy::meta::make_registration_string("COMPUTE_CONSTRAINT_FORCES")>(
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
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}
};  // ComputeConstraintForces

/// \class ComputeConstraintForcesHookeanSpring
/// \brief Concrete implementation of \c MetaKernel for computing the node force induced by a Hookean spring.
///
/// This class assumes that springs have BEAM_2 topology.
///
/// The force is computed as F1 = k * (L - L0) * u12, F2 = -F1, where k is the spring constant, L is the current spring
/// length, L0 is the rest length, and u12 is the unit vector pointing from the first node to the second.
class ComputeConstraintForcesHookeanSpring : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeConstraintForcesHookeanSpring(mundy::mesh::BulkData *const bulk_data_ptr,
                                                const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                        "ComputeConstraintForcesHookeanSpring: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(ComputeConstraintForcesHookeanSpring::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                          std::string("ComputeConstraintForcesHookeanSpring: Part '") + part_name +
                              "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    const std::string node_coord_field_name = mundy::shapes::Spheres::get_node_coord_field_name();
    const std::string element_rest_length_field_name =
        valid_fixed_params.get<std::string>("element_rest_length_field_name");
    const std::string element_spring_constant_field_name =
        valid_fixed_params.get<std::string>("element_spring_constant_field_name");

    node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
    node_coordinates_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
    element_rest_length_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_rest_length_field_name);
    element_spring_constant_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_spring_constant_field_name);
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
    valid_fixed_params.validateParametersAndSetDefaults(ComputeConstraintForcesHookeanSpring::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string element_rest_length_field_name = valid_fixed_params.get<std::string>("element_rest_length_field_name");
    std::string element_spring_constant_field_name =
        valid_fixed_params.get<std::string>("element_spring_constant_field_name");

    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->set_part_topology(stk::topology::BEAM_2);
      part_reqs->add_field_reqs<double>(node_force_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(element_rest_length_field_name, stk::topology::ELEMENT_RANK, 1, 1);
      part_reqs->add_field_reqs<double>(element_spring_constant_field_name, stk::topology::ELEMENT_RANK, 1, 1);

      mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs);
    }
    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("valid_entity_part_names", mundy::core::make_string_array(default_part_name_),
                               "Name of the parts associated with this kernel.");
    default_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                               "Name of the node force field to be used for storing the computed spring force.");
    default_parameter_list.set("element_rest_length_field_name", std::string(default_element_rest_length_field_name_),
                               "Name of the element field containing the rest length of the spring.");
    default_parameter_list.set("element_spring_constant_field_name",
                               std::string(default_element_spring_constant_field_name_),
                               "Name of the element field containing the spring constant.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<ComputeConstraintForcesHookeanSpring>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(
        ComputeConstraintForcesHookeanSpring::get_valid_mutable_params());
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
  void execute(const stk::mesh::Selector &spring_selector) {
    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &element_rest_length_field = *element_rest_length_field_ptr_;
    stk::mesh::Field<double> &element_spring_constant_field = *element_spring_constant_field_ptr_;

    stk::mesh::Selector intersection_with_valid_entity_parts =
        stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() & spring_selector;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, intersection_with_valid_entity_parts,
        [&node_force_field, &node_coord_field, &element_rest_length_field, &element_spring_constant_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spring_element) {
          // Fetch the connected nodes.
          const stk::mesh::Entity *nodes = bulk_data.begin_nodes(spring_element);
          const stk::mesh::Entity &node1 = nodes[0];
          const stk::mesh::Entity &node2 = nodes[1];

          // Fetch the required node and element field data.
          const double *node1_coord = stk::mesh::field_data(node_coord_field, node1);
          const double *node2_coord = stk::mesh::field_data(node_coord_field, node2);
          const double *element_rest_length = stk::mesh::field_data(element_rest_length_field, spring_element);
          const double *element_spring_constant = stk::mesh::field_data(element_spring_constant_field, spring_element);

          // Compute the separation distance and the unit vector from node1 to node2.
          double separation[3] = {node2_coord[0] - node1_coord[0], node2_coord[1] - node1_coord[1],
                                  node2_coord[2] - node1_coord[2]};
          const double separation_length =
              std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] + separation[2] * separation[2]);

          // Compute the spring force.
          const double spring_force_magnitude =
              element_spring_constant[0] * (separation_length - element_rest_length[0]);
          const double spring_force[3] = {spring_force_magnitude * separation[0] / separation_length,
                                          spring_force_magnitude * separation[1] / separation_length,
                                          spring_force_magnitude * separation[2] / separation_length};

          // Add the spring force to the nodes.
          double *node1_force = stk::mesh::field_data(node_force_field, node1);
          double *node2_force = stk::mesh::field_data(node_force_field, node2);

#pragma omp atomic
          node1_force[0] += spring_force[0];
#pragma omp atomic
          node1_force[1] += spring_force[1];
#pragma omp atomic
          node1_force[2] += spring_force[2];
#pragma omp atomic
          node2_force[0] -= spring_force[0];
#pragma omp atomic
          node2_force[1] -= spring_force[1];
#pragma omp atomic
          node2_force[2] -= spring_force[2];
        });
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_part_name_ = "SPRINGS";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_element_rest_length_field_name_ = "ELEMENT_REST_LENGTH";
  static constexpr std::string_view default_element_spring_constant_field_name_ = "ELEMENT_SPRING_CONSTANT";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief Node field containing the node's force.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Node field containing the node's coordinates.
  stk::mesh::Field<double> *node_coordinates_field_ptr_ = nullptr;

  /// \brief Element field containing the spring's rest length.
  stk::mesh::Field<double> *element_rest_length_field_ptr_ = nullptr;

  /// \brief Element field containing the spring's spring constant.
  stk::mesh::Field<double> *element_spring_constant_field_ptr_ = nullptr;
  //@}
};  // ComputeConstraintForcesHookeanSpring

// Workaround due to CUDA not liking our meta factory registration
static inline volatile const bool register_compute_constraint_force_kernels_ = []() {
  // Register our default kernels
  ComputeConstraintForces::OurKernelFactory::register_new_class<ComputeConstraintForcesHookeanSpring>("HOOKEAN_SPRING");
  return true;
}();

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
  double sphere_radius_lower_bound = 0.6;
  double sphere_radius_upper_bound = 0.6;
  double initial_segment_length = 1.0;
  double rest_length = 1.0;
  bool loadbalance_initial_config = false;

  size_t num_time_steps = 100;
  double timestep_size = 0.01;
  double diffusion_coeff = 1.0;
  double viscosity = 1.0;
  double youngs_modulus = 1000.0;
  double poissons_ratio = 0.3;
  double spring_constant = 1.0;

  // Parse the command line options.
  Teuchos::CommandLineProcessor cmdp(false, true);

  // Optional command line arguments for controlling sphere initialization:
  cmdp.setOption("num_spheres", &num_spheres, "Number of spheres.");
  cmdp.setOption("sphere_radius_lower_bound", &sphere_radius_lower_bound,
                 "Lower bound of the sphere radius. Sphere radii will be chosen between the lower and upper bound");
  cmdp.setOption("sphere_radius_upper_bound", &sphere_radius_upper_bound,
                 "Upper bound of the sphere radius. Sphere radii will be chosen between the lower and upper bound");
  cmdp.setOption("initial_segment_length", &initial_segment_length, "Initial segment length.");
  cmdp.setOption("rest_length", &rest_length, "Rest length of the spring.");
  cmdp.setOption("loadbalance", "no_loadbalance", &loadbalance_initial_config,
                 "Load balance the initial configuration.");

  // Optional command line arguments for controlling the simulation:
  cmdp.setOption("num_time_steps", &num_time_steps, "Number of time steps.");
  cmdp.setOption("timestep_size", &timestep_size, "Time step size.");
  cmdp.setOption("diffusion_coeff", &diffusion_coeff, "Diffusion coefficient.");
  cmdp.setOption("viscosity", &viscosity, "Viscosity.");
  cmdp.setOption("youngs_modulus", &youngs_modulus, "Young's modulus.");
  cmdp.setOption("poissons_ratio", &poissons_ratio, "Poisson's ratio.");
  cmdp.setOption("spring_constant", &spring_constant, "Spring constant.");

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
    std::cout << "  sphere_radius_lower_bound: " << sphere_radius_lower_bound << std::endl;
    std::cout << "  sphere_radius_upper_bound: " << sphere_radius_upper_bound << std::endl;
    std::cout << "  initial_segment_length: " << initial_segment_length << std::endl;
    std::cout << "  rest_length: " << rest_length << std::endl;
    std::cout << "  num_time_steps: " << num_time_steps << std::endl;
    std::cout << "  timestep_size: " << timestep_size << std::endl;
    std::cout << "  diffusion_coeff: " << diffusion_coeff << std::endl;
    std::cout << "  viscosity: " << viscosity << std::endl;
    std::cout << "  youngs_modulus: " << youngs_modulus << std::endl;
    std::cout << "  poissons_ratio: " << poissons_ratio << std::endl;
    std::cout << "  spring_constant: " << spring_constant << std::endl;
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
  Teuchos::ParameterList compute_mobility_fixed_params;
  compute_mobility_fixed_params.set("enabled_technique_name", "LOCAL_DRAG")
      .set("node_force_field_name", "NODE_FORCE")
      .set("node_velocity_field_name", "NODE_VELOCITY");
  compute_mobility_fixed_params.sublist("LOCAL_DRAG")
      .set("enabled_kernel_names", mundy::core::make_string_array("NONORIENTABLE_SPHERE"));
  compute_mobility_fixed_params.sublist("LOCAL_DRAG")
      .sublist("NONORIENTABLE_SPHERE")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHERES"));

  // ComputeConstraintForces fixed parameters
  Teuchos::ParameterList compute_constraint_forces_fixed_params;
  compute_constraint_forces_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("HOOKEAN_SPRING"))
      .sublist("HOOKEAN_SPRING")
      .set("node_force_field_name", "NODE_FORCE")
      .set("element_rest_length_field_name", "ELEMENT_REST_LENGTH")
      .set("element_spring_constant_field_name", "ELEMENT_SPRING_CONSTANT");

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
      .set("linker_potential_force_magnitude_field_name", "LINKER_POTENTIAL_FORCE")
      .set("linker_signed_separation_distance_field_name", "LINKER_SIGNED_SEPARATION_DISTANCE")
      .set("element_youngs_modulus_field_name", "ELEMENT_YOUNGS_MODULUS")
      .set("element_poissons_ratio_field_name", "ELEMENT_POISSONS_RATIO");

  // LinkerPotentialForceReduction fixed parameters
  Teuchos::ParameterList linker_potential_force_reduction_fixed_params;
  linker_potential_force_reduction_fixed_params.set("enabled_kernel_names", mundy::core::make_string_array("SPHERE"))
      .set("name_of_linker_part_to_reduce_over", "SPHERE_SPHERE_LINKERS")
      .set("linker_potential_force_magnitude_field_name", "LINKER_POTENTIAL_FORCE")
      .set("linker_contact_normal_field_name", "LINKER_CONTACT_NORMAL");
  linker_potential_force_reduction_fixed_params.sublist("SPHERE")
      .set("valid_entity_part_names", mundy::core::make_string_array("SPHERES"))
      .set("node_force_field_name", "NODE_FORCE");

  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params = Teuchos::ParameterList();
  destroy_neighbor_linkers_fixed_params.set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS")
      .sublist("DESTROY_DISTANT_NEIGHBORS")
      .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
      .set("valid_connected_source_and_target_part_names", mundy::core::make_string_array("SPHERES"))
      .set("linker_destroy_flag_field_name", "LINKER_DESTROY_FLAG")
      .set("element_aabb_field_name", "ELEMENT_AABB");

  // Create the class instances and mesh based on the given fixed requirements.
  auto [compute_brownian_velocity_ptr, node_euler_ptr, compute_mobility_ptr, compute_constraint_forces_ptr,
        compute_ssd_and_cn_ptr, compute_aabb_ptr, generate_neighbor_linkers_ptr, evaluate_linker_potentials_ptr,
        linker_potential_force_reduction_ptr, destroy_neighbor_linkers_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
          ComputeBrownianVelocity, NodeEuler, ComputeMobility, ComputeConstraintForces,
          mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal, mundy::shapes::ComputeAABB,
          mundy::linkers::GenerateNeighborLinkers, mundy::linkers::EvaluateLinkerPotentials,
          mundy::linkers::LinkerPotentialForceReduction, mundy::linkers::DestroyNeighborLinkers>(
          {compute_brownian_velocity_fixed_params, node_euler_fixed_params, compute_mobility_fixed_params,
           compute_constraint_forces_fixed_params, compute_ssd_and_cn_fixed_params, compute_aabb_fixed_params,
           generate_neighbor_linkers_fixed_params, evaluate_linker_potentials_fixed_params,
           linker_potential_force_reduction_fixed_params, destroy_neighbor_linkers_fixed_params});

  auto check_class_instance = [](auto &class_instance_ptr, const std::string &class_name) {
    MUNDY_THROW_REQUIRE(class_instance_ptr != nullptr, std::invalid_argument,
                        std::string("Failed to create class instance with name << ") + class_name + " >>.");
  };  // check_class_instance

  check_class_instance(compute_brownian_velocity_ptr, "ComputeBrownianVelocity");
  check_class_instance(node_euler_ptr, "NodeEuler");
  check_class_instance(compute_mobility_ptr, "ComputeMobility");
  check_class_instance(compute_constraint_forces_ptr, "ComputeConstraintForces");
  check_class_instance(compute_ssd_and_cn_ptr, "ComputeSignedSeparationDistanceAndContactNormal");
  check_class_instance(compute_aabb_ptr, "ComputeAABB");
  check_class_instance(generate_neighbor_linkers_ptr, "GenerateNeighborLinkers");
  check_class_instance(evaluate_linker_potentials_ptr, "EvaluateLinkerPotentials");
  check_class_instance(linker_potential_force_reduction_ptr, "LinkerPotentialForceReduction");
  check_class_instance(destroy_neighbor_linkers_ptr, "DestroyNeighborLinkers");

  MUNDY_THROW_REQUIRE(bulk_data_ptr != nullptr, std::invalid_argument, "Bulk dta pointer cannot be a nullptr.");
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  MUNDY_THROW_REQUIRE(meta_data_ptr != nullptr, std::invalid_argument, "Meta data pointer cannot be a nullptr.");
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");

  ///////////////////////////////////////////////////
  // Set up the mutable parameters for the classes //
  ///////////////////////////////////////////////////

  // ComputeBrownianVelocity mutable parameters
  Teuchos::ParameterList compute_brownian_velocity_mutable_params;
  compute_brownian_velocity_mutable_params.set("timestep_size", timestep_size).set("alpha", 1.0).set("beta", 1.0);
  compute_brownian_velocity_mutable_params.sublist("SPHERE").set("diffusion_coeff", diffusion_coeff);
  compute_brownian_velocity_ptr->set_mutable_params(compute_brownian_velocity_mutable_params);

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

  ////////////////////////////////
  // Fetch the fields and parts //
  ////////////////////////////////
  // Node rank fields
  auto node_coordinates_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  auto node_velocity_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_VELOCITY");
  auto node_force_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_FORCE");
  auto node_rng_counter_field_ptr = meta_data_ptr->get_field<unsigned>(stk::topology::NODE_RANK, "NODE_RNG_COUNTER");

  // Element rank fields
  auto element_radius_field_ptr = meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  auto element_youngs_modulus_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_YOUNGS_MODULUS");
  auto element_poissons_ratio_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_POISSONS_RATIO");
  auto element_rest_length_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_REST_LENGTH");
  auto element_spring_constant_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_SPRING_CONSTANT");

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

  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  MUNDY_THROW_REQUIRE(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");
  stk::mesh::Part &spheres_part = *spheres_part_ptr;

  stk::mesh::Part *sphere_sphere_linkers_part_ptr = meta_data_ptr->get_part("SPHERE_SPHERE_LINKERS");
  MUNDY_THROW_REQUIRE(sphere_sphere_linkers_part_ptr != nullptr, std::invalid_argument,
                      "SPHERE_SPHERE_LINKERS part not found.");
  stk::mesh::Part &sphere_sphere_linkers_part = *sphere_sphere_linkers_part_ptr;

  stk::mesh::Part *springs_part_ptr = meta_data_ptr->get_part("SPRINGS");
  MUNDY_THROW_REQUIRE(springs_part_ptr != nullptr, std::invalid_argument, "SPRINGS part not found.");
  stk::mesh::Part &springs_part = *springs_part_ptr;

  ///////////////////
  // Setup our IO  //
  ///////////////////
  stk::io::put_io_part_attribute(spheres_part);
  stk::io::put_io_part_attribute(sphere_sphere_linkers_part);
  stk::io::put_io_part_attribute(springs_part);

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

  //////////////////////////////////////
  // Initialize the spheres and nodes //
  //////////////////////////////////////

  // Declare and connect the spheres, springs, and nodes. Initialize their positions and radius.
  const size_t rank = bulk_data_ptr->parallel_rank();
  const size_t spheres_per_rank = num_spheres / bulk_data_ptr->parallel_size();
  const size_t remainder = num_spheres % bulk_data_ptr->parallel_size();
  const size_t sphere_id_start = rank * spheres_per_rank + std::min(rank, remainder) + 1;
  const size_t sphere_id_end = sphere_id_start + spheres_per_rank + (rank < remainder ? 1 : 0);
  bulk_data_ptr->modification_begin();
  openrand::Philox rng(1, 0);

  // Spheres first.
  for (size_t i = sphere_id_start; i < sphere_id_end; ++i) {
    // Create the sphere.
    stk::mesh::EntityId our_sphere_id = i;
    stk::mesh::Entity sphere = bulk_data_ptr->declare_element(our_sphere_id);
    bulk_data_ptr->change_entity_parts(sphere, stk::mesh::ConstPartVector({&spheres_part}));

    // Create the node and connect it to the sphere.
    stk::mesh::EntityId our_node_id = i;
    stk::mesh::Entity node = bulk_data_ptr->declare_node(our_node_id);
    bulk_data_ptr->declare_relation(sphere, node, 0);

    // Set the node's coordinates using the given coordinate map.
    double *const node_coords = stk::mesh::field_data(*node_coordinates_field_ptr, node);
    node_coords[0] = static_cast<double>(i) * initial_segment_length;
    node_coords[1] = 0.0;
    node_coords[2] = 0.0;

    // Zero out the node's force and velocity.
    double *node_force = stk::mesh::field_data(*node_force_field_ptr, node);
    node_force[0] = 0.0;
    node_force[1] = 0.0;
    node_force[2] = 0.0;

    double *node_velocity = stk::mesh::field_data(*node_velocity_field_ptr, node);
    node_velocity[0] = 0.0;
    node_velocity[1] = 0.0;
    node_velocity[2] = 0.0;

    // Set the misc fields.
    stk::mesh::field_data(*node_rng_counter_field_ptr, node)[0] = 0;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sphere)[0] = youngs_modulus;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sphere)[0] = poissons_ratio;

    // Set the sphere's radius.
    stk::mesh::field_data(*element_radius_field_ptr, sphere)[0] =
        rng.rand<double>() * (sphere_radius_upper_bound - sphere_radius_lower_bound) + sphere_radius_lower_bound;
  }

  // Note, node sharing is symmetric. If we don't own the node that we intend to share, we need to declare it before
  // marking it as shared. If we are rank 0, we share our final node with rank 1 and receive their first node. If we are
  // rank N, we share our first node with rank N - 1 and receive their final node. Otherwise, we share our first and
  // last nodes with the corresponding neighboring ranks and receive their corresponding nodes.
  if (bulk_data_ptr->parallel_size() > 1) {
    if (rank == 0) {
      // Share the last node with rank 1.
      stk::mesh::EntityId node_id = sphere_id_end - 1;
      stk::mesh::Entity node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, node_id);
      bulk_data_ptr->add_node_sharing(node, rank + 1);

      // Receive the first node from rank 1
      stk::mesh::EntityId received_node_id = sphere_id_end;
      stk::mesh::Entity received_node = bulk_data_ptr->declare_node(received_node_id);
      bulk_data_ptr->add_node_sharing(received_node, rank + 1);
    } else if (rank == bulk_data_ptr->parallel_size() - 1) {
      // Share the first node with rank N - 1.
      stk::mesh::EntityId node_id = sphere_id_start;
      stk::mesh::Entity node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, node_id);
      bulk_data_ptr->add_node_sharing(node, rank - 1);

      // Receive the last node from rank N - 1.
      stk::mesh::EntityId received_node_id = sphere_id_start - 1;
      stk::mesh::Entity received_node = bulk_data_ptr->declare_node(received_node_id);
      bulk_data_ptr->add_node_sharing(received_node, rank - 1);
    } else {
      // Share the first and last nodes with the corresponding neighboring ranks.
      stk::mesh::EntityId first_node_id = sphere_id_start;
      stk::mesh::EntityId last_node_id = sphere_id_end - 1;
      stk::mesh::Entity first_node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, first_node_id);
      stk::mesh::Entity last_node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, last_node_id);
      bulk_data_ptr->add_node_sharing(first_node, rank - 1);
      bulk_data_ptr->add_node_sharing(last_node, rank + 1);

      // Receive the corresponding nodes from the neighboring ranks.
      stk::mesh::EntityId received_first_node_id = sphere_id_start - 1;
      stk::mesh::EntityId received_last_node_id = sphere_id_end;
      stk::mesh::Entity received_first_node = bulk_data_ptr->declare_node(received_first_node_id);
      stk::mesh::Entity received_last_node = bulk_data_ptr->declare_node(received_last_node_id);
      bulk_data_ptr->add_node_sharing(received_first_node, rank - 1);
      bulk_data_ptr->add_node_sharing(received_last_node, rank + 1);
    }
  }

  // Now the springs.
  const size_t spring_id_start = sphere_id_start + num_spheres;
  const size_t spring_id_end =
      (rank == bulk_data_ptr->parallel_size() - 1) ? sphere_id_end - 1 + num_spheres : sphere_id_end + num_spheres;
  for (size_t i = spring_id_start; i < spring_id_end; ++i) {
    // Create the spring.
    stk::mesh::EntityId our_spring_id = i;
    stk::mesh::Entity spring = bulk_data_ptr->declare_element(our_spring_id);
    bulk_data_ptr->change_entity_parts(spring, stk::mesh::ConstPartVector({&springs_part}));

    // Connect the spring to the nodes of the neighboring spheres. We've already declared the nodes and rectified
    // sharing, so we can just fetch them.
    stk::mesh::EntityId node1_id = i - num_spheres;
    stk::mesh::EntityId node2_id = i + 1 - num_spheres;
    stk::mesh::Entity node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, node1_id);
    stk::mesh::Entity node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, node2_id);

    bulk_data_ptr->declare_relation(spring, node1, 0);
    bulk_data_ptr->declare_relation(spring, node2, 1);

    // Set the spring's properties.
    stk::mesh::field_data(*element_rest_length_field_ptr, spring)[0] = rest_length;
    stk::mesh::field_data(*element_spring_constant_field_ptr, spring)[0] = spring_constant;
  }
  bulk_data_ptr->modification_end();

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

#ifdef BROWNIAN_DEBUG_WRITE_MESH_INFO
  // Dump the initial mesh to screen
  auto dump_mesh_info = [&bulk_data_ptr](const std::string &message) {
    std::cout << "############################################" << std::endl;
    std::cout << message << std::endl;
    stk::mesh::impl::dump_all_mesh_info(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()), std::cout);
    std::cout << "############################################" << std::endl;
  };
  dump_mesh_info("Dumping initial mesh info.");
#else
  auto dump_mesh_info = []([[maybe_unused]] const std::string &message) {};
#endif

#ifdef BROWNIAN_DEBUG_WRITE_MESH
  // Write the initial mesh to file
  auto write_mesh = [&stk_io_broker, &output_file_index](double time) {
    stk_io_broker.begin_output_step(output_file_index, time);
    stk_io_broker.write_defined_output_fields(output_file_index);
    stk_io_broker.end_output_step(output_file_index);
    stk_io_broker.flush_output();
  };
#else
  auto write_mesh = []([[maybe_unused]] double time) {};
#endif

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
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr, std::array{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr, std::array{0.0, 0.0, 0.0});

    TIME_BLOCK(compute_constraint_forces_ptr->execute(springs_part), rank,
               "compute_constraint_forces_ptr->execute(springs_part)")
    dump_mesh_info("After compute_constraint_forces_ptr->execute(springs_part)");
    write_mesh(static_cast<double>(i) + 0.01);

    if (i % 10 == 0) {
      compute_aabb_ptr->execute(spheres_part);
      dump_mesh_info("After compute_aabb_ptr->execute(spheres_part)");
      write_mesh(static_cast<double>(i) + 0.02);

      destroy_neighbor_linkers_ptr->execute(sphere_sphere_linkers_part);
      dump_mesh_info("After destroy_neighbor_linkers_ptr->execute(sphere_sphere_linkers_part)");
      write_mesh(static_cast<double>(i) + 0.03);

      generate_neighbor_linkers_ptr->execute(spheres_part, spheres_part);
      dump_mesh_info("After generate_neighbor_linkers_ptr->execute(spheres_part, spheres_part)");
      write_mesh(static_cast<double>(i) + 0.04);
    }

    compute_ssd_and_cn_ptr->execute(sphere_sphere_linkers_part);
    dump_mesh_info("After compute_ssd_and_cn_ptr->execute(sphere_sphere_linkers_part)");
    write_mesh(static_cast<double>(i) + 0.05);

    evaluate_linker_potentials_ptr->execute(sphere_sphere_linkers_part);
    dump_mesh_info("After evaluate_linker_potentials_ptr->execute(sphere_sphere_linkers_part)");
    write_mesh(static_cast<double>(i) + 0.06);

    linker_potential_force_reduction_ptr->execute(spheres_part);
    dump_mesh_info("After linker_potential_force_reduction_ptr->execute(spheres_part)");
    write_mesh(static_cast<double>(i) + 0.07);

    compute_mobility_ptr->execute(spheres_part);
    dump_mesh_info("After compute_mobility_ptr->execute(spheres_part)");
    write_mesh(static_cast<double>(i) + 0.08);

    compute_brownian_velocity_ptr->execute(spheres_part);
    dump_mesh_info("After compute_brownian_velocity_ptr->execute(spheres_part)");
    write_mesh(static_cast<double>(i) + 0.09);

    node_euler_ptr->execute(spheres_part);
    dump_mesh_info("After node_euler_ptr->execute(spheres_part)");
    write_mesh(static_cast<double>(i) + 0.10);

    if (i % 1000 == 0) {
      stk_io_broker.begin_output_step(output_file_index, static_cast<double>(i));
      stk_io_broker.write_defined_output_fields(output_file_index);
      stk_io_broker.end_output_step(output_file_index);
      stk_io_broker.flush_output();
    }
  }

  // compute_aabb_ptr->execute(spheres_part);
  // generate_neighbor_linkers_ptr->execute(spheres_part, spheres_part);
  // for (size_t i = 0; i < num_time_steps; i++) {
  //   // As a first pass, we will:
  //   //  - Zero out the node forces and velocities
  //   //  - Compute the AABB for the spheres
  //   //  - Delete SphereSphereLinkers that are too far apart
  //   //  - Generate SphereSphereLinkers neighbor linkers between nearby spheres
  //   //  - Compute the signed separation distance and contact normal for the SphereSphereLinkers
  //   //  - Evaluate the Hertzian contact potential for the SphereSphereLinkers
  //   //  - Reduce the linker potential force to the Sphere nodes
  //   //  - Compute the velocity induced by the node forces using local drag
  //   //  - Compute the brownian velocity for the nodes
  //   //  - Update the node positions using a first order Euler method
  //   mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr, std::array{0.0, 0.0, 0.0});
  //   mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr, std::array{0.0, 0.0, 0.0});

  //   TIME_BLOCK(compute_constraint_forces_ptr->execute(springs_part), rank,
  //   "compute_constraint_forces_ptr->execute(springs_part)") dump_mesh_info("After
  //   compute_constraint_forces_ptr->execute(springs_part)"); write_mesh(static_cast<double>(i) + 0.1);

  //   TIME_BLOCK(compute_ssd_and_cn_ptr->execute(sphere_sphere_linkers_part), rank, "compute_ssd_and_cn_ptr->execute")
  //   dump_mesh_info("After compute_ssd_and_cn_ptr->execute(sphere_sphere_linkers_part)");
  //   write_mesh(static_cast<double>(i) + 0.2);

  //   TIME_BLOCK(evaluate_linker_potentials_ptr->execute(sphere_sphere_linkers_part), rank,
  //              "evaluate_linker_potentials_ptr->execute(sphere_sphere_linkers_part)")
  //   dump_mesh_info("After evaluate_linker_potentials_ptr->execute(sphere_sphere_linkers_part)");
  //   write_mesh(static_cast<double>(i) + 0.3);

  //   TIME_BLOCK(linker_potential_force_reduction_ptr->execute(spheres_part), rank,
  //              "linker_potential_force_reduction_ptr->execute(spheres_part)")
  //   dump_mesh_info("After linker_potential_force_reduction_ptr->execute(spheres_part)");
  //   write_mesh(static_cast<double>(i) + 0.4);

  //   TIME_BLOCK(compute_mobility_ptr->execute(spheres_part), rank, "compute_mobility_ptr->execute(spheres_part)")
  //   dump_mesh_info("After compute_mobility_ptr->execute(spheres_part)");
  //   write_mesh(static_cast<double>(i) + 0.5);

  //   TIME_BLOCK(compute_brownian_velocity_ptr->execute(spheres_part), rank,
  //              "compute_brownian_velocity_ptr->execute(spheres_part)")
  //   dump_mesh_info("After compute_brownian_velocity_ptr->execute(spheres_part)");
  //   write_mesh(static_cast<double>(i) + 0.6);

  //   TIME_BLOCK(node_euler_ptr->execute(spheres_part), rank, "node_euler_ptr->execute(spheres_part)")
  //   dump_mesh_info("After node_euler_ptr->execute(spheres_part)");
  //   write_mesh(static_cast<double>(i) + 0.7);

  //   if (i % 10 == 0) {
  //     stk_io_broker.begin_output_step(output_file_index, static_cast<double>(i));
  //     stk_io_broker.write_defined_output_fields(output_file_index);
  //     stk_io_broker.end_output_step(output_file_index);
  //     stk_io_broker.flush_output();
  //   }
  // }

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

// export OMP_NUM_THREADS=1 && export OMP_PROC_BIND=spread && OMP_PLACES=threads && mpirun -n 2
// ./MundyLinker_PerformanceTestSphereBrownianMotionWithContactNew.exe --num_spheres_x=40 --num_spheres_y=40
// --num_spheres_z=1 --num_time_steps=10 mpirun -n 2
// ./MundyLinker_PerformanceTestSphereBrownianMotionWithContactNew.exe
// --num_spheres_x=40 --num_spheres_y=40 --num_spheres_z=1 --num_time_steps=4

// mpirun -n 4 ./MundyLinker_PerformanceTestSphereBrownianMotionWithContactNew.exe --num_spheres_x=4 --num_spheres_y=4
// --num_spheres_z=1 --num_time_steps=4
// export OMP_NUM_THREADS=1 && export OMP_PROC_BIND=spread && OMP_PLACES=threads && mpirun -n 1
// ./MundyLinker_PerformanceTestSphereBrownianMotionWithContactNew.exe --num_spheres_x=2 --num_spheres_y=2
// --num_spheres_z=1 --num_time_steps=1 --length_of_domain=1