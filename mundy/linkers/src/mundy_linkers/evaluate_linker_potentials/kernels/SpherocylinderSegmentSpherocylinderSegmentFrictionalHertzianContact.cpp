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

/// \file SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.cpp
/// \brief Definition of the EvaluateLinkerPotentials'
/// SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/evaluate_linker_potentials/kernels/SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact
#include <mundy_math/Vector3.hpp>                  // for mundy::math::Vector3
#include <mundy_math/distance/SegmentSegment.hpp>  // for mundy::math::distance::distance_sq_from_point_to_line_segment
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace linkers {

namespace evaluate_linker_potentials {

namespace kernels {

// template <class DeviceType>
// template <int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, int SHEARUPDATE>
// KOKKOS_INLINE_FUNCTION void PairGranHookeHistoryKokkos<DeviceType>::operator()(
//     TagPairGranHookeHistoryCompute<NEIGHFLAG, NEWTON_PAIR, EVFLAG, SHEARUPDATE>, const int ii, EV_FLOAT &ev) const {
//   // The f and torque arrays are atomic for Half/Thread neighbor style
//   Kokkos::View<F_FLOAT *[3], typename DAT::t_f_array::array_layout, typename KKDevice<DeviceType>::value,
//                Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>>
//       a_f = f;
//   Kokkos::View<F_FLOAT *[3], typename DAT::t_f_array::array_layout, typename KKDevice<DeviceType>::value,
//                Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value>>
//       a_torque = torque;

//   const int right_idx = d_ilist[ii];
//   const X_FLOAT x_right = x(right_idx, 0);
//   const X_FLOAT y_right = x(right_idx, 1);
//   const X_FLOAT z_right = x(right_idx, 2);
//   const LMP_FLOAT mass_right = rmass[right_idx];
//   const LMP_FLOAT radius_right = radius[right_idx];
//   const int jnum = d_numneigh[right_idx];
//   const int mask_right = mask[right_idx];

//   const V_FLOAT vx_right = v(right_idx, 0);
//   const V_FLOAT vy_right = v(right_idx, 1);
//   const V_FLOAT vz_right = v(right_idx, 2);

//   const V_FLOAT omegax_right = omega(right_idx, 0);
//   const V_FLOAT omegay_right = omega(right_idx, 1);
//   const V_FLOAT omegaz_right = omega(right_idx, 2);

//   F_FLOAT fx_right = 0.0;
//   F_FLOAT fy_right = 0.0;
//   F_FLOAT fz_right = 0.0;

//   F_FLOAT torquex_right = 0.0;
//   F_FLOAT torquey_right = 0.0;
//   F_FLOAT torquez_right = 0.0;

//   for (int jj = 0; jj < jnum; jj++) {
//     int left_idx = d_neighbors(right_idx, jj);
//     F_FLOAT factor_lj = special_lj[sbmask(left_idx)];
//     left_idx &= NEIGHMASK;

//     if (factor_lj == 0) continue;

//     const X_FLOAT delx = x_right - x(left_idx, 0);
//     const X_FLOAT dely = y_right - x(left_idx, 1);
//     const X_FLOAT delz = z_right - x(left_idx, 2);
//     const X_FLOAT rsq = delx * delx + dely * dely + delz * delz;
//     const LMP_FLOAT mass_left = rmass[left_idx];
//     const LMP_FLOAT radius_left = radius[left_idx];
//     const LMP_FLOAT radius_sum = radius_right + radius_left;

//     // check for touching neighbors
//     if (rsq >= radius_sum * radius_sum) {
//       d_firstshear(right_idx, 3 * jj) = 0;
//       d_firstshear(right_idx, 3 * jj + 1) = 0;
//       d_firstshear(right_idx, 3 * jj + 2) = 0;
//       continue;
//     }

//     d_firsttouch(right_idx, jj) = 1;

//     const LMP_FLOAT r = sqrt(rsq);
//     const LMP_FLOAT rinv = 1.0 / r;
//     const LMP_FLOAT rsqinv = 1 / rsq;

//     // relative translational velocity
//     V_FLOAT vr1 = vx_right - v(left_idx, 0);
//     V_FLOAT vr2 = vy_right - v(left_idx, 1);
//     V_FLOAT vr3 = vz_right - v(left_idx, 2);

//     // normal component
//     V_FLOAT vnnr = vr1 * delx + vr2 * dely + vr3 * delz;
//     V_FLOAT vn1 = delx * vnnr * rsqinv;
//     V_FLOAT vn2 = dely * vnnr * rsqinv;
//     V_FLOAT vn3 = delz * vnnr * rsqinv;

//     // tangential component
//     V_FLOAT vt1 = vr1 - vn1;
//     V_FLOAT vt2 = vr2 - vn2;
//     V_FLOAT vt3 = vr3 - vn3;

//     // relative rotational velocity
//     V_FLOAT wr1 = (radius_right * omegax_right + radius_left * omega(left_idx, 0)) * rinv;
//     V_FLOAT wr2 = (radius_right * omegay_right + radius_left * omega(left_idx, 1)) * rinv;
//     V_FLOAT wr3 = (radius_right * omegaz_right + radius_left * omega(left_idx, 2)) * rinv;

//     LMP_FLOAT meff = mass_right * mass_left / (mass_right + mass_left);
//     if (mask_right & freeze_group_bit) meff = mass_left;
//     if (mask[left_idx] & freeze_group_bit) meff = mass_right;

//     F_FLOAT damp = meff * gamman * vnnr * rsqinv;
//     F_FLOAT ccel = kn * (radius_sum - r) * rinv - damp;
//     if (limit_damping && (ccel < 0.0)) ccel = 0.0;

//     // relative velocities
//     V_FLOAT vtr1 = vt1 - (delz * wr2 - dely * wr3);
//     V_FLOAT vtr2 = vt2 - (delx * wr3 - delz * wr1);
//     V_FLOAT vtr3 = vt3 - (dely * wr1 - delx * wr2);

//     // shear history effects
//     X_FLOAT shear1 = d_firstshear(right_idx, 3 * jj);
//     X_FLOAT shear2 = d_firstshear(right_idx, 3 * jj + 1);
//     X_FLOAT shear3 = d_firstshear(right_idx, 3 * jj + 2);

//     if (SHEARUPDATE) {
//       shear1 += vtr1 * dt;
//       shear2 += vtr2 * dt;
//       shear3 += vtr3 * dt;
//     }
//     X_FLOAT shrmag = sqrt(shear1 * shear1 + shear2 * shear2 + shear3 * shear3);

//     if (SHEARUPDATE) {
//       // rotate shear displacements
//       X_FLOAT rsht = shear1 * delx + shear2 * dely + shear3 * delz;
//       rsht *= rsqinv;

//       shear1 -= rsht * delx;
//       shear2 -= rsht * dely;
//       shear3 -= rsht * delz;
//     }

//     // tangential forces = shear + tangential velocity damping
//     F_FLOAT fs1 = -(kt * shear1 + meff * gammat * vtr1);
//     F_FLOAT fs2 = -(kt * shear2 + meff * gammat * vtr2);
//     F_FLOAT fs3 = -(kt * shear3 + meff * gammat * vtr3);

//     // rescale frictional displacements and forces if needed
//     F_FLOAT fs = sqrt(fs1 * fs1 + fs2 * fs2 + fs3 * fs3);
//     F_FLOAT fn = xmu * fabs(ccel * r);

//     if (fs > fn) {
//       if (shrmag != 0.0) {
//         shear1 = (fn / fs) * (shear1 + meff * gammat * vtr1 / kt) - meff * gammat * vtr1 / kt;
//         shear2 = (fn / fs) * (shear2 + meff * gammat * vtr2 / kt) - meff * gammat * vtr2 / kt;
//         shear3 = (fn / fs) * (shear3 + meff * gammat * vtr3 / kt) - meff * gammat * vtr3 / kt;
//         fs1 *= fn / fs;
//         fs2 *= fn / fs;
//         fs3 *= fn / fs;
//       } else
//         fs1 = fs2 = fs3 = 0.0;
//     }

//     if (SHEARUPDATE) {
//       d_firstshear(right_idx, 3 * jj) = shear1;
//       d_firstshear(right_idx, 3 * jj + 1) = shear2;
//       d_firstshear(right_idx, 3 * jj + 2) = shear3;
//     }

//     // forces & torques
//     F_FLOAT fx = delx * ccel + fs1;
//     F_FLOAT fy = dely * ccel + fs2;
//     F_FLOAT fz = delz * ccel + fs3;
//     fx *= factor_lj;
//     fy *= factor_lj;
//     fz *= factor_lj;
//     fx_right += fx;
//     fy_right += fy;
//     fz_right += fz;

//     F_FLOAT tor1 = rinv * (dely * fs3 - delz * fs2);
//     F_FLOAT tor2 = rinv * (delz * fs1 - delx * fs3);
//     F_FLOAT tor3 = rinv * (delx * fs2 - dely * fs1);
//     tor1 *= factor_lj;
//     tor2 *= factor_lj;
//     tor3 *= factor_lj;
//     torquex_right -= radius_right * tor1;
//     torquey_right -= radius_right * tor2;
//     torquez_right -= radius_right * tor3;

//     if (NEWTON_PAIR || left_idx < nlocal) {
//       a_f(left_idx, 0) -= fx;
//       a_f(left_idx, 1) -= fy;
//       a_f(left_idx, 2) -= fz;
//       a_torque(left_idx, 0) -= radius_left * tor1;
//       a_torque(left_idx, 1) -= radius_left * tor2;
//       a_torque(left_idx, 2) -= radius_left * tor3;
//     }

//     if (EVFLAG == 2)
//       ev_tally_xyz_atom<NEIGHFLAG, NEWTON_PAIR>(ev, right_idx, left_idx, fx_i, fy_i, fz_i, delx, dely, delz);
//     if (EVFLAG == 1) ev_tally_xyz<NEWTON_PAIR>(ev, right_idx, left_idx, fx_i, fy_i, fz_i, delx, dely, delz);
//   }

//   a_f(right_idx, 0) += fx_i;
//   a_f(right_idx, 1) += fy_i;
//   a_f(right_idx, 2) += fz_i;
//   a_torque(right_idx, 0) += torquex_i;
//   a_torque(right_idx, 1) += torquey_i;
//   a_torque(right_idx, 2) += torquez_i;
// }

// \name Constructors and destructor
//{

SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::
    SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                        const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(
      bulk_data_ptr_ != nullptr, std::invalid_argument,
      "SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(
      SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coords_field_name = mundy::shapes::SpherocylinderSegments::get_node_coord_field_name();
  const std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
  const std::string element_radius_field_name = mundy::shapes::SpherocylinderSegments::get_element_radius_field_name();
  const std::string linker_potential_force_field_name =
      valid_fixed_params.get<std::string>("linker_potential_force_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
  const std::string linker_tangential_displacement_field_name =
      valid_fixed_params.get<std::string>("linker_tangential_displacement_field_name");
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_contact_points_field_name =
      valid_fixed_params.get<std::string>("linker_contact_points_field_name");
  const std::string linked_entities_field_name = NeighborLinkers::get_linked_entities_field_name();

  node_coords_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coords_field_name);
  node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  linker_potential_force_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_potential_force_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);
  linker_tangential_displacement_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_tangential_displacement_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_contact_points_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_points_field_name);
  linked_entities_field_ptr_ = meta_data_ptr_->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, linked_entities_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                        std::string("SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: Field ") +
                            field_name + " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(node_coords_field_ptr_, node_coords_field_name);
  field_exists(node_velocity_field_ptr_, node_velocity_field_name);
  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);
  field_exists(linker_tangential_displacement_field_ptr_, linker_tangential_displacement_field_name);
  field_exists(linker_potential_force_field_ptr_, linker_potential_force_field_name);
  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linker_contact_points_field_ptr_, linker_contact_points_field_name);
  field_exists(linked_entities_field_ptr_, linked_entities_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_sy_seg_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_segment_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(part != nullptr, std::invalid_argument,
                          std::string("SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: Part ") +
                              part_name + " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };  // parts_from_names

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_sy_seg_parts_ = parts_from_names(*meta_data_ptr_, valid_sy_seg_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *>
SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::set_mutable_params(
    const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(
      SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::get_valid_mutable_params());
}
//}

// \name Actions
//{

namespace {

template <typename Accessor, typename OwnershipType>
mundy::math::Vector3d get_contact_point_velocity(
    const mundy::math::AVector3<double, Accessor, OwnershipType> &contact_point, const stk::mesh::Entity *nodes,
    const stk::mesh::Field<double> &node_velocity_field, const stk::mesh::Field<double> &node_coords_field) {
  const auto pos0 = mundy::mesh::vector3_field_data(node_coords_field, nodes[0]);
  const auto pos1 = mundy::mesh::vector3_field_data(node_coords_field, nodes[1]);
  const auto vel0 = mundy::mesh::vector3_field_data(node_velocity_field, nodes[0]);
  const auto vel1 = mundy::mesh::vector3_field_data(node_velocity_field, nodes[1]);

  // Derived by hand using relative motion about the leftmost point.
  // Unlike what can be found on Wikipedia or most undergrad mechanics textbooks, our rods may extend.
  // This version drops the dependence on twist.
  const auto rel_vel = vel1 - vel0;
  const auto left_to_cp = contact_point - pos0;
  const auto left_to_right = pos1 - pos0;
  const double length = mundy::math::norm(left_to_right);
  const double inv_length = 1.0 / length;
  const auto tangent = left_to_right * inv_length;

  const auto term1 = mundy::math::dot(left_to_cp, rel_vel) * tangent * inv_length;
  const auto term2 =
      mundy::math::dot(left_to_cp, tangent) * (rel_vel - mundy::math::dot(tangent, rel_vel) * tangent) * inv_length;
  return vel0 + term1 + term2;
}

}  // namespace

void SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::execute(
    const stk::mesh::Selector &sy_seg_sy_seg_linker_selector) {
  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &node_coords_field = *node_coords_field_ptr_;
  const stk::mesh::Field<double> &node_velocity_field_old = node_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
  const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  const stk::mesh::Field<double> &linker_signed_separation_distance_field =
      *linker_signed_separation_distance_field_ptr_;
  const stk::mesh::Field<double> &linker_tangential_displacement_field = *linker_tangential_displacement_field_ptr_;
  const stk::mesh::Field<double> &linker_contact_normal_field = *linker_contact_normal_field_ptr_;
  const stk::mesh::Field<double> &linker_contact_points_field = *linker_contact_points_field_ptr_;
  const LinkedEntitiesFieldType &linked_entities_field = *linked_entities_field_ptr_;
  stk::mesh::Field<double> &linker_potential_force_field = *linker_potential_force_field_ptr_;

  // Communicate ghosted fields.
  stk::mesh::communicate_field_data(
      *bulk_data_ptr_,
      {node_coords_field_ptr_, &node_velocity_field_old, element_radius_field_ptr_,
       linker_signed_separation_distance_field_ptr_, linker_tangential_displacement_field_ptr_,
       linker_contact_normal_field_ptr_, linker_contact_points_field_ptr_, linker_potential_force_field_ptr_});

  // TODO(palmerb4): For now, we hardcode some of the parameters. We'll need to take them in as mutable params.
  const double density = 1.0;
  const double youngs_modulus = 500000.00;
  const double poissons_ratio = 0.3;
  const double shear_modulus = 0.5 * youngs_modulus / (1.0 + poissons_ratio);
  const double normal_spring_coeff = 4.0 / 3.0 * shear_modulus / (1.0 - poissons_ratio);
  const double tang_spring_coeff = 4.0 * shear_modulus / (2.0 - poissons_ratio);

  const double friction_coeff = 0.5;  // Typically between 0 and 1
  const double normal_damping_coeff = 0.0;
  const double tang_damping_coeff = 0.0;  // A good choice is 0.5 * normal_damping_coeff

  const double time_step_size = 1e-5;

  // At the end of this loop, all locally owned and ghosted linkers will be up-to-date.
  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & sy_seg_sy_seg_linker_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, intersection_with_valid_entity_parts,
      [&node_coords_field, &node_velocity_field_old, &element_radius_field, &linker_potential_force_field,
       &linker_signed_separation_distance_field, &linker_tangential_displacement_field, &linker_contact_normal_field,
       &linker_contact_points_field, &time_step_size, &density, &normal_spring_coeff, &tang_spring_coeff,
       &normal_damping_coeff, &tang_damping_coeff, &friction_coeff, &linked_entities_field](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sy_seg_sy_seg_linker) {
        // This is an expensive kernel, so we only run it if the particles actually overlap.
        const double linker_signed_separation_distance =
            stk::mesh::field_data(linker_signed_separation_distance_field, sy_seg_sy_seg_linker)[0];
        auto tang_disp = mundy::mesh::vector3_field_data(linker_tangential_displacement_field, sy_seg_sy_seg_linker);
        if (linker_signed_separation_distance > 0) {
          // No contact, reset the tangential displacement
          tang_disp.set(0.0, 0.0, 0.0);
        } else {
          // Contact, compute the contact forces

          // Fetch the attached entities. Use references to avoid copying or pointer dereferencing.
          const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
              stk::mesh::field_data(linked_entities_field, sy_seg_sy_seg_linker));
          const stk::mesh::Entity &left_sy_seg_element = bulk_data.get_entity(key_t_ptr[0]);
          const stk::mesh::Entity &right_sy_seg_element = bulk_data.get_entity(key_t_ptr[1]);

          MUNDY_THROW_ASSERT(bulk_data.is_valid(left_sy_seg_element), std::invalid_argument,
                             "SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: left_sy_seg_element "
                             "entity is not valid.");
          MUNDY_THROW_ASSERT(bulk_data.is_valid(right_sy_seg_element), std::invalid_argument,
                             "SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact: "
                             "right_sy_seg_element entity is not valid.");

          const stk::mesh::Entity *left_sy_seg_nodes = bulk_data.begin_nodes(left_sy_seg_element);
          const stk::mesh::Entity *right_sy_seg_nodes = bulk_data.begin_nodes(right_sy_seg_element);

          // Determine the velocity of the contact points
          const auto left_contact_normal =
              mundy::mesh::vector3_field_data(linker_contact_normal_field, sy_seg_sy_seg_linker);
          const auto left_cp = mundy::math::get_vector3_view<double>(
              stk::mesh::field_data(linker_contact_points_field, sy_seg_sy_seg_linker));
          const auto right_cp = mundy::math::get_vector3_view<double>(
              stk::mesh::field_data(linker_contact_points_field, sy_seg_sy_seg_linker) + 3);
          const auto left_cp_vel =
              get_contact_point_velocity(left_cp, left_sy_seg_nodes, node_velocity_field_old, node_coords_field);
          const auto right_cp_vel =
              get_contact_point_velocity(right_cp, right_sy_seg_nodes, node_velocity_field_old, node_coords_field);

          // Compute the relative normal and tangential velocities
          const auto rel_cp_vel = right_cp_vel - left_cp_vel;
          const auto rel_vel_normal = mundy::math::dot(rel_cp_vel, left_contact_normal) * left_contact_normal;
          const auto rel_vel_tang = rel_cp_vel - rel_vel_normal;

          // Compute the tangential displacement (history variable)
          // First add on the current tangential displacement, then project onto the tangent plane.
          tang_disp += rel_vel_tang * time_step_size;
          tang_disp -= mundy::math::dot(tang_disp, left_contact_normal) * left_contact_normal;
          const double tang_disp_mag = mundy::math::norm(tang_disp);

          // Compute the contact force
          // Note, for LAMMPS' delta is the negative of our signed separation distance.
          // As well, they compute the force on the RIGHT particle. We compute the force on the LEFT, introducing a
          // negative sign.
          const double left_radius = stk::mesh::field_data(element_radius_field, left_sy_seg_element)[0];
          const double right_radius = stk::mesh::field_data(element_radius_field, right_sy_seg_element)[0];
          const double left_mass = 4.0 / 3.0 * M_PI * left_radius * left_radius * left_radius * density;
          const double right_mass = 4.0 / 3.0 * M_PI * right_radius * right_radius * right_radius * density;

          const double effective_radius = (left_radius * right_radius) / (left_radius + right_radius);
          const double effective_mass = (left_mass * right_mass) / (left_mass + right_mass);

          const double hertz_poly = std::sqrt(-effective_radius * linker_signed_separation_distance);
          auto normal_force =
              hertz_poly * (normal_spring_coeff * linker_signed_separation_distance * left_contact_normal +
                            effective_mass * normal_damping_coeff * rel_vel_normal);
          auto tang_force =
              hertz_poly * (tang_spring_coeff * tang_disp + effective_mass * tang_damping_coeff * rel_vel_tang);

          // Rescale frictional displacements and forces if needed to satisfy the Coulomb friction law
          // Ft = min(friction_coeff*Fn, Ft)
          const double normal_force_mag = mundy::math::norm(normal_force);
          const double tang_force_mag = mundy::math::norm(tang_force);
          const double scaled_normal_force_mag = friction_coeff * normal_force_mag;
          if (tang_force_mag > scaled_normal_force_mag) {
            if (tang_disp_mag != 0.0) {  // TODO(palmerb4): Exact comparison to 0.0 is bad. Use a tol.
              tang_disp = (scaled_normal_force_mag / tang_force_mag) *
                              (tang_disp + effective_mass * tang_damping_coeff * rel_vel_tang / tang_spring_coeff) -
                          effective_mass * tang_damping_coeff * rel_vel_tang / tang_spring_coeff;
              tang_force *= scaled_normal_force_mag / tang_force_mag;
            } else {
              tang_force.set(0.0, 0.0, 0.0);
            }
          }

          // Save the contact force (Forces are equal and opposite, so we only save the left force)
          mundy::mesh::vector3_field_data(linker_potential_force_field, sy_seg_sy_seg_linker) +=
              normal_force + tang_force;
        }
      });
}

//}

}  // namespace kernels

}  // namespace evaluate_linker_potentials

}  // namespace linkers

}  // namespace mundy
