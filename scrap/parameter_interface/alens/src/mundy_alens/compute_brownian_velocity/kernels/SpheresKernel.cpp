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

/// \file SpheresKernel.cpp
/// \brief Definition of the ComputeBrownianVelocity's SpheresKernel kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_alens/compute_brownian_velocity/kernels/SpheresKernel.hpp>  // for mundy::alens::compute_brownian_velocity::kernels::SpheresKernel
#include <mundy_core/throw_assert.hpp>                                      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                                          // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>                                     // for mundy::mesh::for_each_entity_run
#include <mundy_shapes/Spheres.hpp>                                         // for mundy::shapes::Spheres

namespace mundy {

namespace alens {

namespace compute_brownian_velocity {

namespace kernels {

// \name Constructors and destructor
//{

SpheresKernel::SpheresKernel(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "SpheresKernel: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SpheresKernel::get_valid_fixed_params());

  // Store the valid entity parts for the kernel.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  for (const std::string &part_name : valid_entity_part_names) {
    valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
    MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                        std::string("SpheresKernel: Part '") + part_name +
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
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SpheresKernel::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SpheresKernel::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SpheresKernel::get_valid_mutable_params());
  time_step_size_ = valid_mutable_params.get<double>("time_step_size");
  diffusion_coeff_ = valid_mutable_params.get<double>("diffusion_coeff");
}
//}

// \name Actions
//{

void SpheresKernel::execute(const stk::mesh::Selector &sphere_selector) {
  // Get references to internal members so we aren't passing around *this
  stk::mesh::Field<double> &node_brownian_velocity_field = *node_brownian_velocity_field_ptr_;
  stk::mesh::Field<unsigned> &node_rng_counter_field = *node_rng_counter_field_ptr_;
  double time_step_size = time_step_size_;
  double diffusion_coeff = diffusion_coeff_;

  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & sphere_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::NODE_RANK, intersection_with_valid_entity_parts,
      [&node_brownian_velocity_field, &node_rng_counter_field, &time_step_size, &diffusion_coeff](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
        double *node_brownian_velocity = stk::mesh::field_data(node_brownian_velocity_field, sphere_node);
        const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
        unsigned *node_rng_counter = stk::mesh::field_data(node_rng_counter_field, sphere_node);

        openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
        node_brownian_velocity[0] += std::sqrt(2.0 * diffusion_coeff / time_step_size) * rng.randn<double>();
        node_brownian_velocity[1] += std::sqrt(2.0 * diffusion_coeff / time_step_size) * rng.randn<double>();
        node_brownian_velocity[2] += std::sqrt(2.0 * diffusion_coeff / time_step_size) * rng.randn<double>();
        node_rng_counter[0]++;
      });
}
//}

}  // namespace kernels

}  // namespace compute_brownian_velocity

}  // namespace alens

}  // namespace mundy
