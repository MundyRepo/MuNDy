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

#include "PerfTestSharedComponentsSupport.hpp"

namespace mundy::mesh::perf_test_shared_components {

RigidBodyPerfState::RigidBodyPerfState(const size_t num_entities)
    : meta_data_ptr_(),
      bulk_data_ptr_(),
      selector_(),
      force_field_ptr_(nullptr),
      torque_field_ptr_(nullptr),
      orientation_field_ptr_(nullptr),
      mobility_field_ptr_(nullptr),
      velocity_field_ptr_(nullptr),
      stress_field_ptr_(nullptr),
      orientation_out_field_ptr_(nullptr),
      energy_field_ptr_(nullptr),
      dt_field_ptr_(nullptr),
      drag_scalar_field_ptr_(nullptr),
      ambient_field_ptr_(nullptr),
      drag_field_ptr_(nullptr),
      target_orientation_field_ptr_(nullptr),
      entities_(),
      shared_dt_(0.015),
      shared_drag_scalar_(0.65),
      shared_ambient_(0.25, -0.15, 0.35),
      shared_drag_(make_spd_matrix3(0.35)),
      shared_target_orientation_(normalize(quaternion_t(1.0, 0.04, -0.08, 0.03))),
      ngp_dt_view_("shared_dt", 1),
      ngp_drag_scalar_view_("shared_drag_scalar", 1),
      ngp_ambient_view_("shared_ambient", 3),
      ngp_drag_view_("shared_drag", 9),
      ngp_target_orientation_view_("shared_target_orientation", 4) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  meta_data_ptr_ = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr_;
  meta_data.use_simple_fields();

  bulk_data_ptr_ = builder.create(meta_data_ptr_);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr_;

  force_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "FORCE");
  torque_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "TORQUE");
  orientation_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "ORIENTATION");
  mobility_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "MOBILITY");
  velocity_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "VELOCITY");
  stress_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "STRESS");
  orientation_out_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "ORIENTATION_OUT");
  energy_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "ENERGY");
  dt_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "DT");
  drag_scalar_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "DRAG_SCALAR");
  ambient_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "AMBIENT");
  drag_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "DRAG");
  target_orientation_field_ptr_ =
      &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "TARGET_ORIENTATION");

  stk::mesh::put_field_on_mesh(*force_field_ptr_, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(*torque_field_ptr_, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(*orientation_field_ptr_, meta_data.universal_part(), 4, nullptr);
  stk::mesh::put_field_on_mesh(*mobility_field_ptr_, meta_data.universal_part(), 9, nullptr);
  stk::mesh::put_field_on_mesh(*velocity_field_ptr_, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(*stress_field_ptr_, meta_data.universal_part(), 9, nullptr);
  stk::mesh::put_field_on_mesh(*orientation_out_field_ptr_, meta_data.universal_part(), 4, nullptr);
  stk::mesh::put_field_on_mesh(*energy_field_ptr_, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(*dt_field_ptr_, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(*drag_scalar_field_ptr_, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(*ambient_field_ptr_, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(*drag_field_ptr_, meta_data.universal_part(), 9, nullptr);
  stk::mesh::put_field_on_mesh(*target_orientation_field_ptr_, meta_data.universal_part(), 4, nullptr);
  meta_data.commit();

  selector_ = meta_data.universal_part();

  bulk_data.modification_begin();
  entities_.reserve(num_entities);
  for (size_t entity_index = 0; entity_index < num_entities; ++entity_index) {
    const stk::mesh::Entity node = bulk_data.declare_node(entity_index + 1);
    entities_.push_back(node);

    const scalar_t phase = 1.0e-5 * static_cast<scalar_t>(entity_index);
    const vector3_t force(1.25 + 0.25 * phase, -0.85 + 0.15 * phase, 0.65 - 0.10 * phase);
    const vector3_t torque(0.35 + 0.12 * phase, -0.28 + 0.05 * phase, 0.18 - 0.02 * phase);
    const quaternion_t orientation = normalize(quaternion_t(1.0, 0.18 * phase, -0.11 * phase, 0.09 * phase));
    const matrix3_t mobility = make_spd_matrix3(0.20 + 0.05 * phase);

    vector3_field_data(*force_field_ptr_, node) = force;
    vector3_field_data(*torque_field_ptr_, node) = torque;
    quaternion_field_data(*orientation_field_ptr_, node) = orientation;
    matrix3_field_data(*mobility_field_ptr_, node) = mobility;
    vector3_field_data(*velocity_field_ptr_, node) = vector3_t(0.0, 0.0, 0.0);
    matrix3_field_data(*stress_field_ptr_, node) = matrix3_t(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    quaternion_field_data(*orientation_out_field_ptr_, node) = quaternion_t(1.0, 0.0, 0.0, 0.0);
    scalar_field_data(*energy_field_ptr_, node)[0] = 0.0;
    scalar_field_data(*dt_field_ptr_, node)[0] = shared_dt_;
    scalar_field_data(*drag_scalar_field_ptr_, node)[0] = 0.75 + 0.10 * phase;
    vector3_field_data(*ambient_field_ptr_, node) = shared_ambient_;
    matrix3_field_data(*drag_field_ptr_, node) = shared_drag_;
    quaternion_field_data(*target_orientation_field_ptr_, node) = shared_target_orientation_;
  }
  bulk_data.modification_end();

  initialize_ngp_shared_views();
}

void RigidBodyPerfState::sync_outputs_to_host() const {
  velocity_field().sync_to_host();
  stress_field().sync_to_host();
  orientation_out_field().sync_to_host();
  energy_field().sync_to_host();
}

scalar_t RigidBodyPerfState::checksum() const {
  const size_t stride = std::max<size_t>(size_t(1), entities_.size() / 32);
  scalar_t sum = 0.0;

  for (size_t entity_index = 0; entity_index < entities_.size(); entity_index += stride) {
    const stk::mesh::Entity entity = entities_[entity_index];
    const auto velocity = vector3_field_data(*velocity_field_ptr_, entity);
    const auto stress = matrix3_field_data(*stress_field_ptr_, entity);
    const auto orientation = quaternion_field_data(*orientation_out_field_ptr_, entity);
    const auto energy = scalar_field_data(*energy_field_ptr_, entity);

    sum += velocity[0] + 2.0 * velocity[1] + 3.0 * velocity[2];
    sum += 0.5 * (stress[0] + stress[4] + stress[8]);
    sum += orientation.w() + orientation.x() - orientation.y() + 0.25 * orientation.z();
    sum += energy[0];
  }

  return sum;
}

void RigidBodyPerfState::initialize_ngp_shared_views() {
  auto dt_host = Kokkos::create_mirror_view(ngp_dt_view_);
  dt_host(0) = shared_dt_;
  Kokkos::deep_copy(ngp_dt_view_, dt_host);

  auto drag_scalar_host = Kokkos::create_mirror_view(ngp_drag_scalar_view_);
  drag_scalar_host(0) = shared_drag_scalar_;
  Kokkos::deep_copy(ngp_drag_scalar_view_, drag_scalar_host);

  auto ambient_host = Kokkos::create_mirror_view(ngp_ambient_view_);
  for (size_t i = 0; i < 3; ++i) {
    ambient_host(i) = shared_ambient_[i];
  }
  Kokkos::deep_copy(ngp_ambient_view_, ambient_host);

  auto drag_host = Kokkos::create_mirror_view(ngp_drag_view_);
  for (size_t i = 0; i < 9; ++i) {
    drag_host(i) = shared_drag_[i];
  }
  Kokkos::deep_copy(ngp_drag_view_, drag_host);

  auto target_orientation_host = Kokkos::create_mirror_view(ngp_target_orientation_view_);
  target_orientation_host(0) = shared_target_orientation_.x();
  target_orientation_host(1) = shared_target_orientation_.y();
  target_orientation_host(2) = shared_target_orientation_.z();
  target_orientation_host(3) = shared_target_orientation_.w();
  Kokkos::deep_copy(ngp_target_orientation_view_, target_orientation_host);
}

}  // namespace mundy::mesh::perf_test_shared_components
