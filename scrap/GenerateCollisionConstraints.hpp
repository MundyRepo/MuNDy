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

#ifndef MUNDY_CONSTRAINTS_GENERATECOLLISIONCONSTRAINTS_HPP_
#define MUNDY_CONSTRAINTS_GENERATECOLLISIONCONSTRAINTS_HPP_

/// \file GenerateCollisionConstraints.hpp
/// \brief Declaration of the GenerateCollisionConstraints class

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <string>   // for std::string
#include <utility>  // for std::pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_search/IdentProc.hpp>    // for stk::search::IdentProc
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MeshReqs.hpp>  // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKWayKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>        // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_constraints/generate_collision_constraints/kernels/CollisionSphereSphere.hpp>  // for mundy::constraint::...::CollisionSphereSphere

namespace mundy {

namespace constraints {

// TODO(palmerb4): Who should be the one to decide on this type?
using SearchIdentProc = stk::search::IdentProc<stk::mesh::EntityKey>;
using IdentProcPairVector = std::vector<std::pair<SearchIdentProc, SearchIdentProc>>;

/// \class GenerateCollisionConstraints
/// \brief Method for generating collision constraints between nearby bodies.
///
/// Possible types of generation routines include, minimum separation distance, surface tesselation, multiblob, and
/// recursive generation to name a few. For now, we only implement minimum separation distance.
/// TODO(palmerb4): Break this class into techniques and make sure it generalizes.
class GenerateCollisionConstraints : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  using OurThreeWayKernelFactory = mundy::meta::MetaKWayKernelFactory<3, void, GenerateCollisionConstraints>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  GenerateCollisionConstraints() = delete;

  /// \brief Constructor
  GenerateCollisionConstraints(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshReqs
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);
    Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels");
    const int num_specified_kernels = kernels_sublist.get<int>("count");

    auto mesh_requirements_ptr = std::make_shared<mundy::meta::MeshReqs>();
    for (int i = 0; i < num_specified_kernels; i++) {
      Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
      const std::string kernel_name = kernel_params.get<std::string>("name");
      mesh_requirements_ptr->sync(OurThreeWayKernelFactory::get_mesh_requirements(kernel_name, kernel_params));
    }

    return mesh_requirements_ptr;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    Teuchos::ParameterList params = *fixed_params_ptr;

    if (params.isSublist("kernels")) {
      // Only validate and fill parameters for the given kernels.
      Teuchos::ParameterList &kernels_sublist = params.sublist("kernels", true);
      const int num_specified_kernels = kernels_sublist.get<int>("count");
      for (int i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
        const std::string kernel_name = kernel_params.get<std::string>("name");
        OurThreeWayKernelFactory::validate_fixed_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = params.sublist("kernels", false);
      const int num_specified_kernels = OurThreeWayKernelFactory::num_registered_classes();
      kernels_sublist.set("count", num_specified_kernels);
      int i = 0;
      for (auto &key : OurThreeWayKernelFactory::get_keys()) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), false);
        kernel_params.set("name", std::string(key));
        OurThreeWayKernelFactory::validate_fixed_parameters_and_set_defaults(key, &kernel_params);
        i++;
      }
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isSublist("kernels")) {
      // Only validate and fill parameters for the given kernels.
      Teuchos::ParameterList &kernels_sublist = mutable_params_ptr->sublist("kernels", true);
      const int num_specified_kernels = kernels_sublist.get<int>("count");
      for (int i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
        const std::string kernel_name = kernel_params.get<std::string>("name");
        OurThreeWayKernelFactory::validate_mutable_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = mutable_params_ptr->sublist("kernels", false);
      const unsigned num_specified_kernels = OurThreeWayKernelFactory::num_registered_classes();
      kernels_sublist.set("count", num_specified_kernels);
      int i = 0;
      for (auto &key : OurThreeWayKernelFactory::get_keys()) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), false);
        kernel_params.set("name", std::string(key));
        OurThreeWayKernelFactory::validate_mutable_parameters_and_set_defaults(key, &kernel_params);
        i++;
      }
    }
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<GenerateCollisionConstraints>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute([[maybe_unused]] const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Internal helper functions
  //@{

  /// \brief Given an old and a new neighbor list, get the gained neighbors.
  ///
  /// Gained means all neighbors that are in the new list but not in the old list (minus those that are invalid).
  ///
  /// \param old_neighbor_pairs The previous neighbor list to compair against.
  /// \param new_neighbor_pairs The new neighbor list.
  /// \return The gained neighbor pairs.
  IdentProcPairVector find_our_gained_neighbor_pairs(const IdentProcPairVector &old_neighbor_pairs,
                                                     const IdentProcPairVector &new_neighbor_pairs);

  /// \brief Fetch all lower-ranked entities connected to the given entity.
  ///
  /// \param bulk_data_ptr The bulk data within which the entitiy resides.
  /// \param entity The entitity to get the connections of.
  /// \param entity_rank The rank of the given entity.
  /// \return A vector containing all lower-ranked entities connected to the given entity.
  std::vector<stk::mesh::Entity> get_connected_lower_rank_entities(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                   const stk::mesh::Entity &entity,
                                                                   const stk::topology::rank_t &entity_rank);

  /// \brief Fetch all higher-ranked entities connected to the given entity.
  ///
  /// \param bulk_data_ptr The bulk data within which the entitiy resides.
  /// \param entity The entitity to get the connections of.
  /// \param entity_rank The rank of the given entity.
  /// \return A vector containing all higher-ranked entities connected to the given entity.
  std::vector<stk::mesh::Entity> get_connected_higher_rank_entities(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                    const stk::mesh::Entity &entity,
                                                                    const stk::topology::rank_t &entity_rank);

  /// \brief Given a neighbor list, ghost any neighbor we don't already have access to.
  ///
  /// Optionally, ghost the downward connectivity and/or upward connectivity of the neighbor.
  ///
  /// \param bulk_data_ptr The BulkData object to add the ghosting to (must be in a modifiable state).
  /// \param pairs_to_ghost The set of neighbors to consider for potential ghosting (allowed to have self-neighbors and
  /// non-locally owned domain/range)
  /// \param ghost_downward_connectivity Flag specifying if we should ghost the neighbor's
  /// downward connectivity.
  /// \param ghost_upward_connectivity Flag specifying if we should ghost the neighbor's upward
  /// connectivity.
  /// \return A reference to the generated ghosting, owned by the BulkData.
  stk::mesh::Ghosting &ghost_neighbors(mundy::mesh::BulkData *const bulk_data_ptr,
                                       const IdentProcPairVector &pairs_to_ghost,
                                       const std::string &name_of_ghosting = "geometric_ghosts",
                                       bool ghost_downward_connectivity = false,
                                       bool ghost_upward_connectivity = false);

  /// \brief Given a neighbor list, generate empty collision constraints between neighbors.
  ///
  /// Note, by empty we mean that the constraints will have the proper connectivity structure, but will not have the
  /// correct internal fields.
  ///
  /// \param bulk_data_ptr The BulkData object to add the collision constraints to (must be in a modifiable state).
  /// \param collision_part_ptr The part to assign to the collision constraints.
  /// \param pairs_to_connect The set of neighbors to connect with constraints.
  void generate_empty_collision_constraints_between_pairs(mundy::mesh::BulkData *const bulk_data_ptr,
                                                          stk::mesh::Part *const collision_part_ptr,
                                                          const IdentProcPairVector &pairs_to_connect);
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "GENERATE_COLLISION_CONSTRAINTS";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of active multibody types.
  size_t num_multibody_types_ = 0;

  /// \brief Vector of pointers to the active multibody parts this class acts upon.
  std::vector<stk::mesh::Part *> multibody_part_ptr_vector_;

  /// \brief Pointer to the collision constraint part.
  stk::mesh::Part *collision_part_ptr_;

  /// \brief Vector of kernels, one for each active multibody part.
  std::vector<std::shared_ptr<mundy::meta::MetaKWayKernel<3, void>>> multibody_kernel_ptrs_;

  /// \brief The set of neighbor pairs
  std::shared_ptr<IdentProcPairVector> old_neighbor_pairs_ptr_;

  std::shared_ptr<IdentProcPairVector> current_neighbor_pairs_ptr_;

  //@}
};  // GenerateCollisionConstraints

}  // namespace constraints

}  // namespace mundy

//! \name Registration
//@{

/// \brief Register our default kernels
MUNDY_REGISTER_METACLASS("SPHERE_SPHERE", mundy::constraint::generate_collision_constraints::kernels::CollisionSphereSphere,
                         mundy::constraint::GenerateCollisionConstraints::OurThreeWayKernelFactory)
//@}

#endif  // MUNDY_CONSTRAINTS_GENERATECOLLISIONCONSTRAINTS_HPP_
