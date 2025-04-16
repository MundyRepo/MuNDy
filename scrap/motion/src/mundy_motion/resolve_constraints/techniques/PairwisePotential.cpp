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

/// \file PairwisePotential.cpp
/// \brief Definition of the PairwisePotential class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_meta/MeshReqs.hpp>      // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>    // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_motion/resolve_constraints/techniques/PairwisePotential.hpp>  // for mundy::methods::...::PairwisePotential

namespace mundy {

namespace motion {

namespace resolve_constraints {

namespace techniques {

// \name Constructors and destructor
//{

PairwisePotential::PairwisePotential(mundy::mesh::BulkData *const bulk_data_ptr,
                                     const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "PairwisePotential: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Fetch the parameters for this part's sub-methods.
  Teuchos::ParameterList &compute_pairwise_potential_params =
      valid_fixed_params.sublist("subkernels").sublist("compute_pairwise_potential");

  // Initialize and store the sub-methods.
  const std::string compute_pairwise_potential_name = compute_pairwise_potential_params.get<std::string>("name");
  compute_pairwise_potential_kernel_ptr_ = OurKernelFactory::create_new_instance(
      compute_pairwise_potential_name, bulk_data_ptr_, compute_pairwise_potential_params);
}
//}

// \name MetaMethodSubsetExecutionInterface interface implementation
//{

void PairwisePotential::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  static_validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fetch the parameters for this part's sub-methods.
  Teuchos::ParameterList &compute_pairwise_invariants_params =
      valid_fixed_params.sublist("subkernels").sublist("compute_pairwise_invariants");
  Teuchos::ParameterList &compute_pairwise_potential_params =
      valid_fixed_params.sublist("subkernels").sublist("compute_pairwise_potential");

  // Set the mutable params for each of our sub-methods.
  compute_pairwise_potential_method_ptr_->set_mutable_params(compute_pairwise_potential_params);
}
//}

// \name Actions
//{

void PairwisePotential::execute(const stk::mesh::Selector &input_selector) {
  // Evaluate the users pairwise potential

  /*
  Things we are missing. Accepting multiple potentials.
  Pass in the neighbor list as a mutable param: neighborlist_ptr_

  For each particle pair, compute their invariants. Pass these invariant to the mutable parameters of the various
  potentials. Loop over each pair, call the potentials.


  Users enable multiple kernels just like ComputeAABB. In this case, we have two sets of kernels:

  Users enable multiple potential drivers
  Potential MetaMethodSubsetExecutionInterface/Driver (Sphere-sphere)
    (mutable) Neighbor list
    (computed and passed into the potentials as mutable parameters) Some invariant

    Use selectors to refine the pairs that we are

    (registered TwoWayKernels) Some potentials Functions that take in an entity.


    Truly, the purpose of the driver is to hide the type of the invariant parameters.

    There are multiple different combinations of possibilities.
    We want to reuse the neighbor list as much as possible for a single cutoff radius.
    Each neighbor list may have multiple invariant associated with it, but for each driver these are fixed.
    The issue is that pairwise potentials may be isolated to certain subsets.








    1. invariant kernels
    2. pairwise potentials
  */

  for (size_t i = 0; i < neighbor) }
//}

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace motion

}  // namespace mundy
