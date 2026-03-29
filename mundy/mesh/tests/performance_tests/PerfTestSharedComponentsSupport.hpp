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

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_mesh/FieldComponent.hpp>
#include <mundy_mesh/FieldViews.hpp>
#include <mundy_mesh/ForEachEntity.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/SharedComponent.hpp>
#include <mundy_utils/throw_assert.hpp>
#include <stdexcept>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <string>
#include <vector>

namespace mundy::mesh::perf_test_shared_components {

using scalar_t = double;
using vector3_t = Vector3<scalar_t>;
using matrix3_t = Matrix3<scalar_t>;
using quaternion_t = Quaternion<scalar_t>;
using double_field_t = stk::mesh::Field<scalar_t>;

inline constexpr size_t kNumEntities = 100000;

KOKKOS_INLINE_FUNCTION
matrix3_t make_spd_matrix3(const scalar_t base) {
  return matrix3_t(1.25 + base, 0.08 + 0.05 * base, -0.04 + 0.01 * base, 0.08 + 0.05 * base, 1.75 + 0.5 * base,
                   0.06 - 0.02 * base, -0.04 + 0.01 * base, 0.06 - 0.02 * base, 2.25 + 0.75 * base);
}

template <typename DragType, typename ForceType, typename VelocityType>
KOKKOS_INLINE_FUNCTION void drag_velocity_step(const DragType& drag, const ForceType& force, VelocityType velocity) {
  auto drag_value = drag;
  velocity = force / static_cast<scalar_t>(drag_value);
}

template <typename AmbientType, typename MobilityType, typename ForceType, typename OrientationType,
          typename VelocityType>
KOKKOS_INLINE_FUNCTION void rigid_body_mobility_step(const AmbientType& ambient, const MobilityType& mobility,
                                                     const ForceType& force, const OrientationType& orientation,
                                                     VelocityType velocity) {
  const auto body_force = conjugate(orientation) * force;
  velocity = ambient + orientation * (mobility * body_force);
}

template <typename AmbientType, typename DragType, typename TargetOrientationType, typename ForceType,
          typename TorqueType, typename OrientationType, typename MobilityType, typename VelocityType,
          typename StressType, typename OrientationOutType, typename EnergyType>
KOKKOS_INLINE_FUNCTION void rigid_body_step(const scalar_t dt, const AmbientType& ambient, const DragType& drag,
                                            const TargetOrientationType& target_orientation, const ForceType& force,
                                            const TorqueType& torque, const OrientationType& orientation,
                                            const MobilityType& mobility, VelocityType velocity, StressType stress,
                                            OrientationOutType orientation_out, EnergyType energy) {
  const auto total_force = force + ambient;
  const auto body_force = conjugate(orientation) * total_force;
  const auto body_spin = torque + cross(body_force, ambient);
  const quaternion_t spin_quaternion(0.0, body_spin[0], body_spin[1], body_spin[2]);
  const auto trial_orientation =
      normalize(target_orientation * orientation + (0.5 * dt) * (spin_quaternion * orientation));
  const auto effective_mobility = mobility + drag;
  const auto lab_velocity = orientation * (effective_mobility * body_force);
  const scalar_t inv_force_scale = 1.0 / (1.0 + norm(body_force));
  const auto corrected_velocity = lab_velocity + (dt * inv_force_scale) * cross(body_spin, total_force);
  const auto updated_stress = trial_orientation * effective_mobility + 0.25 * (mobility - drag);

  velocity = corrected_velocity;
  stress = updated_stress;
  orientation_out = trial_orientation;
  energy = dot(corrected_velocity, corrected_velocity) + 0.1 * dot(body_spin, body_spin) +
           dot(trial_orientation, target_orientation);
}

template <typename FieldType>
struct HostScalarFieldAccessor {
  FieldType* field = nullptr;

  inline decltype(auto) operator()(const stk::mesh::Entity entity) const {
    return scalar_field_data(*field, entity);
  }
};

template <typename FieldType>
struct HostVector3FieldAccessor {
  FieldType* field = nullptr;

  inline decltype(auto) operator()(const stk::mesh::Entity entity) const {
    return vector3_field_data(*field, entity);
  }
};

template <typename FieldType>
struct HostMatrix3FieldAccessor {
  FieldType* field = nullptr;

  inline decltype(auto) operator()(const stk::mesh::Entity entity) const {
    return matrix3_field_data(*field, entity);
  }
};

template <typename FieldType>
struct HostQuaternionFieldAccessor {
  FieldType* field = nullptr;

  inline decltype(auto) operator()(const stk::mesh::Entity entity) const {
    return quaternion_field_data(*field, entity);
  }
};

template <typename SharedType>
struct HostRawSharedAccessor {
  const SharedType* value = nullptr;

  inline decltype(auto) operator()(const stk::mesh::Entity /*entity*/) const {
    return *value;
  }
};

template <typename SharedType>
struct HostRefSharedAccessor {
  const SharedType& value;

  inline decltype(auto) operator()(const stk::mesh::Entity /*entity*/) const {
    return value;
  }
};

template <typename NgpFieldType>
struct NgpScalarFieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return scalar_field_data(ngp_field, entity_index);
  }
};

template <typename NgpFieldType>
struct NgpVector3FieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return vector3_field_data(ngp_field, entity_index);
  }
};

template <typename NgpFieldType>
struct NgpMatrix3FieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return matrix3_field_data(ngp_field, entity_index);
  }
};

template <typename NgpFieldType>
struct NgpQuaternionFieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return quaternion_field_data(ngp_field, entity_index);
  }
};

template <typename ViewType>
struct NgpRawScalarAccessor {
  ViewType view;

  KOKKOS_INLINE_FUNCTION
  scalar_t operator()(const stk::mesh::FastMeshIndex& /*entity_index*/) const {
    return view(0);
  }
};

template <typename ViewType>
struct NgpRawVector3Accessor {
  ViewType view;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& /*entity_index*/) const {
    return get_vector3_view<scalar_t>(view.data());
  }
};

template <typename ViewType>
struct NgpRawMatrix3Accessor {
  ViewType view;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& /*entity_index*/) const {
    return get_matrix3_view<scalar_t>(view.data());
  }
};

template <typename ViewType>
struct NgpRawQuaternionAccessor {
  ViewType view;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& /*entity_index*/) const {
    return get_quaternion_view<scalar_t>(view.data());
  }
};

template <typename SharedType>
struct NgpOwningSharedAccessor {
  SharedType value;

  KOKKOS_INLINE_FUNCTION
  const SharedType& operator()(const stk::mesh::FastMeshIndex& /*entity_index*/) const {
    return value;
  }
};

template <typename DtAccessor, typename AmbientAccessor, typename DragAccessor, typename TargetAccessor,
          typename ForceAccessor, typename TorqueAccessor, typename OrientationAccessor, typename MobilityAccessor,
          typename VelocityAccessor, typename StressAccessor, typename OrientationOutAccessor, typename EnergyAccessor>
void run_host_rigid_body_kernel(const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
                                DtAccessor dt_accessor, AmbientAccessor ambient_accessor, DragAccessor drag_accessor,
                                TargetAccessor target_accessor, ForceAccessor force_accessor,
                                TorqueAccessor torque_accessor, OrientationAccessor orientation_accessor,
                                MobilityAccessor mobility_accessor, VelocityAccessor velocity_accessor,
                                StressAccessor stress_accessor, OrientationOutAccessor orientation_out_accessor,
                                EnergyAccessor energy_accessor) {
  for_each_entity_run(bulk_data, stk::topology::NODE_RANK, selector,
                      [=](const stk::mesh::BulkData& /*bulk_data*/, const stk::mesh::Entity entity) {
                        rigid_body_step(dt_accessor(entity), ambient_accessor(entity), drag_accessor(entity),
                                        target_accessor(entity), force_accessor(entity), torque_accessor(entity),
                                        orientation_accessor(entity), mobility_accessor(entity),
                                        velocity_accessor(entity), stress_accessor(entity),
                                        orientation_out_accessor(entity), energy_accessor(entity));
                      });
}

template <typename DragAccessor, typename ForceAccessor, typename VelocityAccessor>
void run_host_drag_velocity_kernel(const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
                                   DragAccessor drag_accessor, ForceAccessor force_accessor,
                                   VelocityAccessor velocity_accessor) {
  for_each_entity_run(bulk_data, stk::topology::NODE_RANK, selector,
                      [=](const stk::mesh::BulkData& /*bulk_data*/, const stk::mesh::Entity entity) {
                        drag_velocity_step(drag_accessor(entity), force_accessor(entity), velocity_accessor(entity));
                      });
}

template <typename AmbientAccessor, typename MobilityAccessor, typename ForceAccessor, typename OrientationAccessor,
          typename VelocityAccessor>
void run_host_body_mobility_kernel(const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
                                   AmbientAccessor ambient_accessor, MobilityAccessor mobility_accessor,
                                   ForceAccessor force_accessor, OrientationAccessor orientation_accessor,
                                   VelocityAccessor velocity_accessor) {
  for_each_entity_run(bulk_data, stk::topology::NODE_RANK, selector,
                      [=](const stk::mesh::BulkData& /*bulk_data*/, const stk::mesh::Entity entity) {
                        rigid_body_mobility_step(ambient_accessor(entity), mobility_accessor(entity),
                                                 force_accessor(entity), orientation_accessor(entity),
                                                 velocity_accessor(entity));
                      });
}

template <typename DtAccessor, typename AmbientAccessor, typename DragAccessor, typename TargetAccessor,
          typename ForceAccessor, typename TorqueAccessor, typename OrientationAccessor, typename MobilityAccessor,
          typename VelocityAccessor, typename StressAccessor, typename OrientationOutAccessor, typename EnergyAccessor>
void run_ngp_rigid_body_kernel(stk::mesh::NgpMesh& ngp_mesh, const stk::mesh::Selector& selector,
                               const DtAccessor& dt_accessor, const AmbientAccessor& ambient_accessor,
                               const DragAccessor& drag_accessor, const TargetAccessor& target_accessor,
                               const ForceAccessor& force_accessor, const TorqueAccessor& torque_accessor,
                               const OrientationAccessor& orientation_accessor,
                               const MobilityAccessor& mobility_accessor, const VelocityAccessor& velocity_accessor,
                               const StressAccessor& stress_accessor,
                               const OrientationOutAccessor& orientation_out_accessor,
                               const EnergyAccessor& energy_accessor) {
  for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index) {
        rigid_body_step(dt_accessor(entity_index), ambient_accessor(entity_index), drag_accessor(entity_index),
                        target_accessor(entity_index), force_accessor(entity_index), torque_accessor(entity_index),
                        orientation_accessor(entity_index), mobility_accessor(entity_index),
                        velocity_accessor(entity_index), stress_accessor(entity_index),
                        orientation_out_accessor(entity_index), energy_accessor(entity_index));
      });
}

template <typename DragAccessor, typename ForceAccessor, typename VelocityAccessor>
void run_ngp_drag_velocity_kernel(stk::mesh::NgpMesh& ngp_mesh, const stk::mesh::Selector& selector,
                                  DragAccessor drag_accessor, ForceAccessor force_accessor,
                                  VelocityAccessor velocity_accessor) {
  for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index) {
        drag_velocity_step(drag_accessor(entity_index), force_accessor(entity_index), velocity_accessor(entity_index));
      });
}

template <typename AmbientAccessor, typename MobilityAccessor, typename ForceAccessor, typename OrientationAccessor,
          typename VelocityAccessor>
void run_ngp_body_mobility_kernel(stk::mesh::NgpMesh& ngp_mesh, const stk::mesh::Selector& selector,
                                  AmbientAccessor ambient_accessor, MobilityAccessor mobility_accessor,
                                  ForceAccessor force_accessor, OrientationAccessor orientation_accessor,
                                  VelocityAccessor velocity_accessor) {
  for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index) {
        rigid_body_mobility_step(ambient_accessor(entity_index), mobility_accessor(entity_index),
                                 force_accessor(entity_index), orientation_accessor(entity_index),
                                 velocity_accessor(entity_index));
      });
}

class RigidBodyPerfState {
 public:
  using ngp_scalar_view_t = Kokkos::View<scalar_t*, stk::ngp::MemSpace>;

  explicit RigidBodyPerfState(size_t num_entities = kNumEntities);

  stk::mesh::BulkData& bulk_data() const {
    return *bulk_data_ptr_;
  }
  const stk::mesh::Selector& selector() const {
    return selector_;
  }
  double_field_t& force_field() const {
    return *force_field_ptr_;
  }
  double_field_t& torque_field() const {
    return *torque_field_ptr_;
  }
  double_field_t& orientation_field() const {
    return *orientation_field_ptr_;
  }
  double_field_t& mobility_field() const {
    return *mobility_field_ptr_;
  }
  double_field_t& velocity_field() const {
    return *velocity_field_ptr_;
  }
  double_field_t& stress_field() const {
    return *stress_field_ptr_;
  }
  double_field_t& orientation_out_field() const {
    return *orientation_out_field_ptr_;
  }
  double_field_t& energy_field() const {
    return *energy_field_ptr_;
  }
  double_field_t& dt_field() const {
    return *dt_field_ptr_;
  }
  double_field_t& drag_scalar_field() const {
    return *drag_scalar_field_ptr_;
  }
  double_field_t& ambient_field() const {
    return *ambient_field_ptr_;
  }
  double_field_t& drag_field() const {
    return *drag_field_ptr_;
  }
  double_field_t& target_orientation_field() const {
    return *target_orientation_field_ptr_;
  }
  const scalar_t& shared_dt() const {
    return shared_dt_;
  }
  const scalar_t& shared_drag_scalar() const {
    return shared_drag_scalar_;
  }
  const vector3_t& shared_ambient() const {
    return shared_ambient_;
  }
  const matrix3_t& shared_drag() const {
    return shared_drag_;
  }
  const quaternion_t& shared_target_orientation() const {
    return shared_target_orientation_;
  }
  const ngp_scalar_view_t& ngp_dt_view() const {
    return ngp_dt_view_;
  }
  const ngp_scalar_view_t& ngp_drag_scalar_view() const {
    return ngp_drag_scalar_view_;
  }
  const ngp_scalar_view_t& ngp_ambient_view() const {
    return ngp_ambient_view_;
  }
  const ngp_scalar_view_t& ngp_drag_view() const {
    return ngp_drag_view_;
  }
  const ngp_scalar_view_t& ngp_target_orientation_view() const {
    return ngp_target_orientation_view_;
  }

  void sync_outputs_to_host() const;
  scalar_t checksum() const;

 private:
  void initialize_ngp_shared_views();

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr_;
  stk::mesh::Selector selector_;
  double_field_t* force_field_ptr_;
  double_field_t* torque_field_ptr_;
  double_field_t* orientation_field_ptr_;
  double_field_t* mobility_field_ptr_;
  double_field_t* velocity_field_ptr_;
  double_field_t* stress_field_ptr_;
  double_field_t* orientation_out_field_ptr_;
  double_field_t* energy_field_ptr_;
  double_field_t* dt_field_ptr_;
  double_field_t* drag_scalar_field_ptr_;
  double_field_t* ambient_field_ptr_;
  double_field_t* drag_field_ptr_;
  double_field_t* target_orientation_field_ptr_;
  std::vector<stk::mesh::Entity> entities_;
  scalar_t shared_dt_;
  scalar_t shared_drag_scalar_;
  vector3_t shared_ambient_;
  matrix3_t shared_drag_;
  quaternion_t shared_target_orientation_;
  ngp_scalar_view_t ngp_dt_view_;
  ngp_scalar_view_t ngp_drag_scalar_view_;
  ngp_scalar_view_t ngp_ambient_view_;
  ngp_scalar_view_t ngp_drag_view_;
  ngp_scalar_view_t ngp_target_orientation_view_;
};

template <typename RunnerA, typename RunnerB>
void validate_equal(const std::string& label, RunnerA&& runner_a, RunnerB&& runner_b, RigidBodyPerfState& state,
                    const bool sync_outputs_from_device) {
  runner_a();
  if (sync_outputs_from_device) {
    state.sync_outputs_to_host();
  }
  const scalar_t checksum_a = state.checksum();

  runner_b();
  if (sync_outputs_from_device) {
    state.sync_outputs_to_host();
  }
  const scalar_t checksum_b = state.checksum();

  const scalar_t scale = 1.0 + std::max(std::abs(checksum_a), std::abs(checksum_b));
  MUNDY_THROW_REQUIRE(std::abs(checksum_a - checksum_b) <= 1.0e-11 * scale, std::runtime_error,
                      "PerfTestSharedComponents validation failed for " + label + ". checksum_a = " +
                          std::to_string(checksum_a) + ", checksum_b = " + std::to_string(checksum_b));
}

void run_drag_benchmarks();
void run_body_mobility_benchmarks();
void run_complex_benchmarks();

}  // namespace mundy::mesh::perf_test_shared_components
