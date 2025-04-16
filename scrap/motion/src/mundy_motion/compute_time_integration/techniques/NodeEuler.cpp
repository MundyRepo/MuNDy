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

/// \file NodeEuler.cpp
/// \brief Definition of ComputeTimeIntegration's NodeEuler technique.

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
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_meta/MeshReqs.hpp>                            // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaTwoWayKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_motion/compute_time_integration/techniques/NodeEuler.hpp>  // for mundy::motion::...::NodeEuler

namespace mundy {

namespace motion {

namespace compute_time_integration {

namespace techniques {

// \name Constructors and destructor
//{
// TODO(palmerb4): This class is outdated and needs to be updated to the input_selector paradigm.
NodeEuler::NodeEuler(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument, "NodeEuler: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Parse the parameters
  Teuchos::ParameterList &parts_params = valid_fixed_params.sublist("input_parts");
  num_parts_ = parts_params.get<int>("count");
  part_ptr_vector_.resize(num_parts_);
  for (size_t i = 0; i < num_parts_; i++) {
    // Fetch the i'th part and its parameters
    Teuchos::ParameterList &part_params = parts_params.sublist("input_part_" + std::to_string(i));
    const std::string part_name = part_params.get<std::string>("name");
    part_ptr_vector_[i] = meta_data_ptr_->get_part(part_name);
  }

  // Fill the internal members using the given parameter list.
  node_coord_field_name_ = valid_fixed_params.get<std::string>("node_coord_field_name");
  node_velocity_field_name_ = valid_fixed_params.get<std::string>("node_velocity_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name_);
}
//}

// \name MetaKernel interface implementation
//{

void NodeEuler::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Fill the internal members using the given parameter list.
  time_step_size_ = valid_mutable_params.get<double>("time_step_size");
}
//}

// \name Actions
//{

void NodeEuler::execute(const stk::mesh::Selector &input_selector) {
  // TODO(palmerb4): NodeEuler should only act on the mulitbody Body type. Take the intersection.
  for (size_t i = 0; i < num_parts_; i++) {
    stk::mesh::Selector locally_owned_part = input_selector & stk::mesh::Selector(meta_data_ptr_->locally_owned_part());
    mundy::mesh::for_each_entity_run(*bulk_data_ptr_, stk::topology::NODE_RANK, locally_owned_part,
                                   [&]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, stk::mesh::Entity node) {
                                     // TODO(palmerb4): Add a flag for specifying that node position has changed
                                     // This is the best way to indicate that things like the normal vector need
                                     // updated. Does STK have an observer that lets us check if fields need updated?

                                     // Euler step position.
                                     double *node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node);
                                     double *node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, node);
                                     node_coords[0] += time_step_size_ * node_velocity[0];
                                     node_coords[1] += time_step_size_ * node_velocity[1];
                                     node_coords[2] += time_step_size_ * node_velocity[2];
                                   });
  }
}
//}

}  // namespace techniques

}  // namespace compute_time_integration

}  // namespace motion

}  // namespace mundy
