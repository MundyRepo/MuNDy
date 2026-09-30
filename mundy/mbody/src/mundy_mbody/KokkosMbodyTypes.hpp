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

#ifndef MUNDY_MBODY_KOKKOSMBODYTYPES_HPP_
#define MUNDY_MBODY_KOKKOSMBODYTYPES_HPP_

/// \file
/// \brief Plain data model for the rod/spring/contact MCQPP solver (mundy::mbody::solve).
///
/// Each *Views type stores its fields as flat Kokkos::Views behind accessors. RodViews packs
/// force+torque and velocity+omega into one 6-wide-per-rod buffer each (the generalized-coordinate
/// layout the operators consume); pose fields stay separate. Per-element accessors (center(i),
/// lambda(p), ...) return a mundy view over the flat storage, not a copy, so
/// `rods.center(i) += dt * rods.velocity(i)` writes straight through. Whole-array accessors (*_view())
/// return the raw Kokkos::View.

// C++ core
#include <type_traits>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/Scalar.hpp>
#include <mundy_math/Vector3.hpp>

namespace mundy {

namespace mbody {

//! \name Generalized-coordinate extractors
//@{
// Generalized force: flat double[6*N], [force(3), torque(3)] per object. Generalized velocity:
// same layout, [velocity(3), omega(3)].

template <typename GenForceView>
KOKKOS_INLINE_FUNCTION auto rod_force(const GenForceView& gen_force, int i) {
  return get_vector3<double>(&gen_force(6 * i));
}
template <typename GenForceView>
KOKKOS_INLINE_FUNCTION auto rod_torque(const GenForceView& gen_force, int i) {
  return get_vector3<double>(&gen_force(6 * i + 3));
}

template <typename GenVelocityView>
KOKKOS_INLINE_FUNCTION auto rod_velocity(const GenVelocityView& gen_velocity, int i) {
  return get_vector3<double>(&gen_velocity(6 * i));
}
template <typename GenVelocityView>
KOKKOS_INLINE_FUNCTION auto rod_omega(const GenVelocityView& gen_velocity, int i) {
  return get_vector3<double>(&gen_velocity(6 * i + 3));
}
//@}

/// \brief A contiguous set of rods (spherocylinders).
///
/// force/torque hold external load on entry and accumulated (external + constraint) force/torque on
/// exit. velocity/omega hold any imposed velocity on entry and the final (external + mobility +
/// constraint) velocity/omega on exit.
template <typename ExecSpace>
class RodViews {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using vector_view_t = Kokkos::View<double*, memory_space>;

  RodViews() = default;

  explicit RodViews(size_t num_rods)
      : center_("center", 3 * num_rods),
        orientation_("orientation", 4 * num_rods),
        radius_("radius", num_rods),
        length_("length", num_rods),
        force_torque_("force_torque", 6 * num_rods),
        velocity_omega_("velocity_omega", 6 * num_rods) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto center(int i)      const { return get_vector3<double>(&center_(3 * i)); }
  KOKKOS_INLINE_FUNCTION auto orientation(int i) const { return get_quaternion<double>(&orientation_(4 * i)); }
  KOKKOS_INLINE_FUNCTION auto radius(int i)      const { return get_scalar<double>(&radius_(i)); }
  KOKKOS_INLINE_FUNCTION auto length(int i)      const { return get_scalar<double>(&length_(i)); }
  KOKKOS_INLINE_FUNCTION auto force(int i)       const { return rod_force(force_torque_, i); }
  KOKKOS_INLINE_FUNCTION auto torque(int i)      const { return rod_torque(force_torque_, i); }
  KOKKOS_INLINE_FUNCTION auto velocity(int i)    const { return rod_velocity(velocity_omega_, i); }
  KOKKOS_INLINE_FUNCTION auto omega(int i)       const { return rod_omega(velocity_omega_, i); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  vector_view_t center_view()         const { return center_; }
  vector_view_t orientation_view()    const { return orientation_; }
  vector_view_t radius_view()         const { return radius_; }
  vector_view_t length_view()         const { return length_; }
  vector_view_t force_torque_view()   const { return force_torque_; }
  vector_view_t velocity_omega_view() const { return velocity_omega_; }
  // clang-format on
  //@}

  size_t size() const {
    return radius_.extent(0);
  }

 private:
  vector_view_t center_;
  vector_view_t orientation_;
  vector_view_t radius_;
  vector_view_t length_;
  vector_view_t force_torque_;
  vector_view_t velocity_omega_;
};

/// \brief Rod-center-to-rod-center Hookean springs. lambda is an output (spring force magnitude).
template <typename ExecSpace>
class LinearSpringViews {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  LinearSpringViews() = default;

  explicit LinearSpringViews(size_t num_springs)
      : rod_i_("rod_i", num_springs),
        rod_j_("rod_j", num_springs),
        rest_length_("rest_length", num_springs),
        spring_constant_("spring_constant", num_springs),
        lambda_("lambda", num_springs) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod_i(int k) const {
    return get_scalar<int>(&rod_i_(k));
  }
  KOKKOS_INLINE_FUNCTION auto rod_j(int k) const {
    return get_scalar<int>(&rod_j_(k));
  }
  KOKKOS_INLINE_FUNCTION auto rest_length(int k) const {
    return get_scalar<double>(&rest_length_(k));
  }
  KOKKOS_INLINE_FUNCTION auto spring_constant(int k) const {
    return get_scalar<double>(&spring_constant_(k));
  }
  KOKKOS_INLINE_FUNCTION auto lambda(int k) const {
    return get_scalar<double>(&lambda_(k));
  }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_i_view()              const { return rod_i_; }
  int_view_t rod_j_view()              const { return rod_j_; }
  scalar_view_t rest_length_view()     const { return rest_length_; }
  scalar_view_t spring_constant_view() const { return spring_constant_; }
  scalar_view_t lambda_view()          const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_i_.extent(0);
  }

  /// \brief The number of scalar constraints this family contributes to the multiplier vector.
  size_t num_constraints() const {
    return size();
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t rest_length_;
  scalar_view_t spring_constant_;
  scalar_view_t lambda_;
};

/// \brief Rod-tangent-to-rod-tangent angle bend springs. lambda is an output (spring torque magnitude).
template <typename ExecSpace>
class AngularSpringViews {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  AngularSpringViews() = default;

  explicit AngularSpringViews(size_t num_springs)
      : rod_i_("rod_i", num_springs),
        rod_j_("rod_j", num_springs),
        rest_angle_("rest_angle", num_springs),
        spring_constant_("spring_constant", num_springs),
        lambda_("lambda", num_springs) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod_i(int k)           const { return get_scalar<int>(&rod_i_(k)); }
  KOKKOS_INLINE_FUNCTION auto rod_j(int k)           const { return get_scalar<int>(&rod_j_(k)); }
  KOKKOS_INLINE_FUNCTION auto rest_angle(int k)      const { return get_scalar<double>(&rest_angle_(k)); }
  KOKKOS_INLINE_FUNCTION auto spring_constant(int k) const { return get_scalar<double>(&spring_constant_(k)); }
  KOKKOS_INLINE_FUNCTION auto lambda(int k)          const { return get_scalar<double>(&lambda_(k)); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_i_view()              const { return rod_i_; }
  int_view_t rod_j_view()              const { return rod_j_; }
  scalar_view_t rest_angle_view()      const { return rest_angle_; }
  scalar_view_t spring_constant_view() const { return spring_constant_; }
  scalar_view_t lambda_view()          const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_i_.extent(0);
  }

  /// \brief The number of scalar constraints this family contributes to the multiplier vector.
  size_t num_constraints() const {
    return size();
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t rest_angle_;
  scalar_view_t spring_constant_;
  scalar_view_t lambda_;
};

/// \brief Three-point angle bend springs. lambda is an output (spring torque, k*(angle-rest_angle)).
///
/// rod_i/rod_j are the outer points, rod_k the vertex the angle is measured at (node1/node2/node3 of a
/// BEAM_3 element; node3 is the vertex). Constrains the angle at rod_k between the position vectors to
/// rod_i and rod_j -- a position-only, three-body constraint with no orientation dependence, unlike
/// AngularSpringViews. Harmonic in the angle itself, not its cosine (see
/// compute_triple_point_angular_spring_geometry). rod_i/rod_j/rod_k need not be distinct.
template <typename ExecSpace>
class TriplePointAngularSpringViews {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  TriplePointAngularSpringViews() = default;

  explicit TriplePointAngularSpringViews(size_t num_springs)
      : rod_i_("rod_i", num_springs),
        rod_j_("rod_j", num_springs),
        rod_k_("rod_k", num_springs),
        rest_angle_("rest_angle", num_springs),
        spring_constant_("spring_constant", num_springs),
        lambda_("lambda", num_springs) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod_i(int k)           const { return get_scalar<int>(&rod_i_(k)); }
  KOKKOS_INLINE_FUNCTION auto rod_j(int k)           const { return get_scalar<int>(&rod_j_(k)); }
  KOKKOS_INLINE_FUNCTION auto rod_k(int k)           const { return get_scalar<int>(&rod_k_(k)); }
  KOKKOS_INLINE_FUNCTION auto rest_angle(int k)      const { return get_scalar<double>(&rest_angle_(k)); }
  KOKKOS_INLINE_FUNCTION auto spring_constant(int k) const { return get_scalar<double>(&spring_constant_(k)); }
  KOKKOS_INLINE_FUNCTION auto lambda(int k)          const { return get_scalar<double>(&lambda_(k)); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_i_view()              const { return rod_i_; }
  int_view_t rod_j_view()              const { return rod_j_; }
  int_view_t rod_k_view()              const { return rod_k_; }
  scalar_view_t rest_angle_view()      const { return rest_angle_; }
  scalar_view_t spring_constant_view() const { return spring_constant_; }
  scalar_view_t lambda_view()          const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_i_.extent(0);
  }

  /// \brief The number of scalar constraints this family contributes to the multiplier vector.
  size_t num_constraints() const {
    return size();
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  int_view_t rod_k_;
  scalar_view_t rest_angle_;
  scalar_view_t spring_constant_;
  scalar_view_t lambda_;
};

/// \brief Rod material points held at fixed world locations. lambda is an output (the support reaction).
///
/// Anchors the material point at body_offset in the rod's own frame rather than the rod centre: rods
/// have length, so clamping a filament means clamping its end. A zero offset recovers the centre.
///
/// Each anchor is three scalar constraints, one per world axis, so its reaction comes back as a
/// vector. compliance is the inverse stiffness of each of those constraints: zero holds the point
/// rigidly, and a small positive value softens it.
template <typename ExecSpace>
class FixedPositionViews {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  FixedPositionViews() = default;

  explicit FixedPositionViews(size_t num_anchors)
      : rod_("rod", num_anchors),
        target_point_("target_point", 3 * num_anchors),
        body_offset_("body_offset", 3 * num_anchors),
        compliance_("compliance", 3 * num_anchors),
        lambda_("lambda", 3 * num_anchors) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod(int a)          const { return get_scalar<int>(&rod_(a)); }
  KOKKOS_INLINE_FUNCTION auto target_point(int a) const { return get_vector3<double>(&target_point_(3 * a)); }
  KOKKOS_INLINE_FUNCTION auto body_offset(int a)  const { return get_vector3<double>(&body_offset_(3 * a)); }
  KOKKOS_INLINE_FUNCTION auto compliance(int a)   const { return get_vector3<double>(&compliance_(3 * a)); }
  KOKKOS_INLINE_FUNCTION auto lambda(int a)       const { return get_vector3<double>(&lambda_(3 * a)); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_view()             const { return rod_; }
  scalar_view_t target_point_view() const { return target_point_; }
  scalar_view_t body_offset_view()  const { return body_offset_; }
  scalar_view_t compliance_view()   const { return compliance_; }
  scalar_view_t lambda_view()       const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_.extent(0);
  }

  /// \brief The number of scalar constraints this family contributes to the multiplier vector.
  size_t num_constraints() const {
    return 3 * size();
  }

 private:
  int_view_t rod_;
  scalar_view_t target_point_;
  scalar_view_t body_offset_;
  scalar_view_t compliance_;
  scalar_view_t lambda_;
};

/// \brief Rod material points and orientations held fixed. lambda is an output (the support reaction).
///
/// A clamped rod end: the anchored material point FixedPositionViews holds, plus the rod's
/// orientation held at target_orientation. Six scalar constraints per anchor, laid out as one
/// generalized [force(3), torque(3)] block, so compliance and lambda split the same way.
///
/// The orientation rows carry the exact map from angular velocity to the orientation error's rate,
/// so they are as exact as the position rows. They still converge onto a target rather than reaching
/// it within one step of arbitrary size: a step rotates by omega*dt, which closes the error only to
/// the accuracy of that single linearization.
template <typename ExecSpace>
class FixedPoseViews {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  FixedPoseViews() = default;

  explicit FixedPoseViews(size_t num_anchors)
      : rod_("rod", num_anchors),
        target_point_("target_point", 3 * num_anchors),
        target_orientation_("target_orientation", 4 * num_anchors),
        body_offset_("body_offset", 3 * num_anchors),
        compliance_("compliance", 6 * num_anchors),
        lambda_("lambda", 6 * num_anchors) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod(int a)                    const { return get_scalar<int>(&rod_(a)); }
  KOKKOS_INLINE_FUNCTION auto target_point(int a)           const { return get_vector3<double>(&target_point_(3 * a)); }
  KOKKOS_INLINE_FUNCTION auto target_orientation(int a)     const { return get_quaternion<double>(&target_orientation_(4 * a)); }
  KOKKOS_INLINE_FUNCTION auto body_offset(int a)            const { return get_vector3<double>(&body_offset_(3 * a)); }
  KOKKOS_INLINE_FUNCTION auto position_compliance(int a)    const { return rod_force(compliance_, a); }
  KOKKOS_INLINE_FUNCTION auto orientation_compliance(int a) const { return rod_torque(compliance_, a); }
  KOKKOS_INLINE_FUNCTION auto position_lambda(int a)        const { return rod_force(lambda_, a); }
  KOKKOS_INLINE_FUNCTION auto orientation_lambda(int a)     const { return rod_torque(lambda_, a); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_view()                  const { return rod_; }
  scalar_view_t target_point_view()      const { return target_point_; }
  scalar_view_t target_orientation_view() const { return target_orientation_; }
  scalar_view_t body_offset_view()       const { return body_offset_; }
  scalar_view_t compliance_view()        const { return compliance_; }
  scalar_view_t lambda_view()            const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_.extent(0);
  }

  /// \brief The number of scalar constraints this family contributes to the multiplier vector.
  size_t num_constraints() const {
    return 6 * size();
  }

 private:
  int_view_t rod_;
  scalar_view_t target_point_;
  scalar_view_t target_orientation_;
  scalar_view_t body_offset_;
  scalar_view_t compliance_;
  scalar_view_t lambda_;
};

/// \brief Rod-rod unilateral (spherocylinder-spherocylinder) contacts. lambda is an output (contact force magnitude).
template <typename ExecSpace>
class ContactViews {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  ContactViews() = default;

  explicit ContactViews(size_t num_contacts)
      : rod_i_("rod_i", num_contacts), rod_j_("rod_j", num_contacts), lambda_("lambda", num_contacts) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod_i(int p)  const { return get_scalar<int>(&rod_i_(p)); }
  KOKKOS_INLINE_FUNCTION auto rod_j(int p)  const { return get_scalar<int>(&rod_j_(p)); }
  KOKKOS_INLINE_FUNCTION auto lambda(int p) const { return get_scalar<double>(&lambda_(p)); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_i_view()     const { return rod_i_; }
  int_view_t rod_j_view()     const { return rod_j_; }
  scalar_view_t lambda_view() const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_i_.extent(0);
  }

  /// \brief The number of scalar constraints this family contributes to the multiplier vector.
  size_t num_constraints() const {
    return size();
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t lambda_;
};

/// \brief The half-open range [begin, end) that one constraint family occupies in a flat array.
struct IndexRange {
  size_t begin = 0;
  size_t end = 0;

  KOKKOS_INLINE_FUNCTION size_t size() const {
    return end - begin;
  }
};

/// \brief Where each bilateral family's constraints land in the flat multiplier vector.
///
/// The bilateral families hold differing numbers of constraints and are stored separately, but the
/// solve works on one contiguous multiplier vector. Packing them is the standard compressed-row
/// treatment of ragged data: an exclusive prefix scan over the per-family counts gives each family
/// its start offset, and a family's own local index added to that offset is its flat position.
///
/// The constraint values, the compliance diagonal, the Jacobian operators and the multiplier
/// write-back all index that vector. They agree by construction only if they share one mapping, so
/// the packing order is decided in exactly one place and the result is passed, never re-derived.
///
/// A trivially copyable aggregate holding only sizes, so it captures by value into a kernel.
struct ConstraintIndexMap {
  IndexRange linear_springs;
  IndexRange angular_springs;
  IndexRange triple_springs;
  IndexRange fixed_positions;
  IndexRange fixed_poses;
  size_t total = 0;
};

/// \brief Every constraint acting on a set of rods over one solve.
///
/// Each family is independently optional: a default-constructed member is empty and contributes no
/// constraints. Contacts are unilateral (their force magnitudes are constrained non-negative); every
/// other family is bilateral, and the bilateral families share one flat multiplier vector.
template <typename ExecSpace>
struct ConstraintSet {
  LinearSpringViews<ExecSpace> linear_springs{0};
  AngularSpringViews<ExecSpace> angular_springs{0};
  TriplePointAngularSpringViews<ExecSpace> triple_springs{0};
  FixedPositionViews<ExecSpace> fixed_positions{0};
  FixedPoseViews<ExecSpace> fixed_poses{0};
  ContactViews<ExecSpace> contacts{0};
};

/// \brief Pack a constraint set's bilateral families into the flat multiplier vector.
///
/// Build this once per set and pass it along; it is derived from the family sizes, so a set whose
/// families are reassigned needs a fresh one.
template <typename ExecSpace>
ConstraintIndexMap make_constraint_index_map(const ConstraintSet<ExecSpace>& constraints) {
  ConstraintIndexMap index_map;
  size_t offset = 0;

  index_map.linear_springs = IndexRange{offset, offset + constraints.linear_springs.num_constraints()};
  offset = index_map.linear_springs.end;
  index_map.angular_springs = IndexRange{offset, offset + constraints.angular_springs.num_constraints()};
  offset = index_map.angular_springs.end;
  index_map.triple_springs = IndexRange{offset, offset + constraints.triple_springs.num_constraints()};
  offset = index_map.triple_springs.end;
  index_map.fixed_positions = IndexRange{offset, offset + constraints.fixed_positions.num_constraints()};
  offset = index_map.fixed_positions.end;
  index_map.fixed_poses = IndexRange{offset, offset + constraints.fixed_poses.num_constraints()};
  offset = index_map.fixed_poses.end;

  index_map.total = offset;
  return index_map;
}

/// \brief Tunables for the outer (PGD) and inner (CG) solves.
struct SolveConfig {
  double dt = 1.0;
  double viscosity = 1.0;
  unsigned max_outer_iters = 1000;
  double outer_tol = 1e-6;
  unsigned max_cg_iters = 200;
  double cg_tol = 1e-8;
};

//! \name Copying between memory spaces
//@{
// The Kokkos view idioms, lifted to the containers above. A mirror's Space is an execution space, and a mirror
// aliases its source when the two share a memory space, as a Kokkos view mirror does.

/// \brief Copy every field of src into dst; the two must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const RodViews<DstSpace>& dst, const RodViews<SrcSpace>& src) {
  Kokkos::deep_copy(dst.center_view(), src.center_view());
  Kokkos::deep_copy(dst.orientation_view(), src.orientation_view());
  Kokkos::deep_copy(dst.radius_view(), src.radius_view());
  Kokkos::deep_copy(dst.length_view(), src.length_view());
  Kokkos::deep_copy(dst.force_torque_view(), src.force_torque_view());
  Kokkos::deep_copy(dst.velocity_omega_view(), src.velocity_omega_view());
}

/// \brief Copy every field of src into dst; the two must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const LinearSpringViews<DstSpace>& dst, const LinearSpringViews<SrcSpace>& src) {
  Kokkos::deep_copy(dst.rod_i_view(), src.rod_i_view());
  Kokkos::deep_copy(dst.rod_j_view(), src.rod_j_view());
  Kokkos::deep_copy(dst.rest_length_view(), src.rest_length_view());
  Kokkos::deep_copy(dst.spring_constant_view(), src.spring_constant_view());
  Kokkos::deep_copy(dst.lambda_view(), src.lambda_view());
}

/// \brief Copy every field of src into dst; the two must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const AngularSpringViews<DstSpace>& dst, const AngularSpringViews<SrcSpace>& src) {
  Kokkos::deep_copy(dst.rod_i_view(), src.rod_i_view());
  Kokkos::deep_copy(dst.rod_j_view(), src.rod_j_view());
  Kokkos::deep_copy(dst.rest_angle_view(), src.rest_angle_view());
  Kokkos::deep_copy(dst.spring_constant_view(), src.spring_constant_view());
  Kokkos::deep_copy(dst.lambda_view(), src.lambda_view());
}

/// \brief Copy every field of src into dst; the two must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const TriplePointAngularSpringViews<DstSpace>& dst,
               const TriplePointAngularSpringViews<SrcSpace>& src) {
  Kokkos::deep_copy(dst.rod_i_view(), src.rod_i_view());
  Kokkos::deep_copy(dst.rod_j_view(), src.rod_j_view());
  Kokkos::deep_copy(dst.rod_k_view(), src.rod_k_view());
  Kokkos::deep_copy(dst.rest_angle_view(), src.rest_angle_view());
  Kokkos::deep_copy(dst.spring_constant_view(), src.spring_constant_view());
  Kokkos::deep_copy(dst.lambda_view(), src.lambda_view());
}

/// \brief Copy every field of src into dst; the two must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const FixedPositionViews<DstSpace>& dst, const FixedPositionViews<SrcSpace>& src) {
  Kokkos::deep_copy(dst.rod_view(), src.rod_view());
  Kokkos::deep_copy(dst.target_point_view(), src.target_point_view());
  Kokkos::deep_copy(dst.body_offset_view(), src.body_offset_view());
  Kokkos::deep_copy(dst.compliance_view(), src.compliance_view());
  Kokkos::deep_copy(dst.lambda_view(), src.lambda_view());
}

/// \brief Copy every field of src into dst; the two must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const FixedPoseViews<DstSpace>& dst, const FixedPoseViews<SrcSpace>& src) {
  Kokkos::deep_copy(dst.rod_view(), src.rod_view());
  Kokkos::deep_copy(dst.target_point_view(), src.target_point_view());
  Kokkos::deep_copy(dst.target_orientation_view(), src.target_orientation_view());
  Kokkos::deep_copy(dst.body_offset_view(), src.body_offset_view());
  Kokkos::deep_copy(dst.compliance_view(), src.compliance_view());
  Kokkos::deep_copy(dst.lambda_view(), src.lambda_view());
}

/// \brief Copy every field of src into dst; the two must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const ContactViews<DstSpace>& dst, const ContactViews<SrcSpace>& src) {
  Kokkos::deep_copy(dst.rod_i_view(), src.rod_i_view());
  Kokkos::deep_copy(dst.rod_j_view(), src.rod_j_view());
  Kokkos::deep_copy(dst.lambda_view(), src.lambda_view());
}

/// \brief Copy every family of src into dst; each pair of families must have the same size.
template <typename DstSpace, typename SrcSpace>
void deep_copy(const ConstraintSet<DstSpace>& dst, const ConstraintSet<SrcSpace>& src) {
  deep_copy(dst.linear_springs, src.linear_springs);
  deep_copy(dst.angular_springs, src.angular_springs);
  deep_copy(dst.triple_springs, src.triple_springs);
  deep_copy(dst.fixed_positions, src.fixed_positions);
  deep_copy(dst.fixed_poses, src.fixed_poses);
  deep_copy(dst.contacts, src.contacts);
}

/// \brief Whether T is one of the *Views specializations.
template <typename T>
struct is_views_container : std::false_type {};
template <typename E>
struct is_views_container<RodViews<E>> : std::true_type {};
template <typename E>
struct is_views_container<LinearSpringViews<E>> : std::true_type {};
template <typename E>
struct is_views_container<AngularSpringViews<E>> : std::true_type {};
template <typename E>
struct is_views_container<TriplePointAngularSpringViews<E>> : std::true_type {};
template <typename E>
struct is_views_container<FixedPositionViews<E>> : std::true_type {};
template <typename E>
struct is_views_container<FixedPoseViews<E>> : std::true_type {};
template <typename E>
struct is_views_container<ContactViews<E>> : std::true_type {};

template <typename T>
inline constexpr bool is_views_container_v = is_views_container<T>::value;

/// \brief Whether T is a ConstraintSet specialization.
template <typename T>
struct is_constraint_set : std::false_type {};
template <typename E>
struct is_constraint_set<ConstraintSet<E>> : std::true_type {};

template <typename T>
inline constexpr bool is_constraint_set_v = is_constraint_set<T>::value;

/// \brief Matches exactly the containers that can be copied between memory spaces.
template <typename T>
concept MirrorableType = is_views_container_v<T> || is_constraint_set_v<T>;

/// \brief A container shaped like src in Space's memory: src itself when the memory spaces match, else a new
/// (zero-initialized) allocation.
template <typename Space, template <typename> class Views, typename SrcSpace>
  requires is_views_container_v<Views<SrcSpace>>
auto create_mirror_view(const Space& /*space*/, const Views<SrcSpace>& src) {
  static_assert(Kokkos::is_execution_space<Space>::value, "create_mirror_view: Space must be an execution space.");
  if constexpr (std::is_same_v<typename Space::memory_space, typename SrcSpace::memory_space>) {
    return src;
  } else {
    return Views<Space>(src.size());
  }
}

/// \brief A constraint set shaped like src in Space's memory: src itself when the memory spaces match, else a
/// family-by-family mirror.
template <typename Space, typename SrcSpace>
auto create_mirror_view(const Space& space, const ConstraintSet<SrcSpace>& src) {
  static_assert(Kokkos::is_execution_space<Space>::value, "create_mirror_view: Space must be an execution space.");
  if constexpr (std::is_same_v<typename Space::memory_space, typename SrcSpace::memory_space>) {
    return src;
  } else {
    ConstraintSet<Space> mirror;
    mirror.linear_springs = create_mirror_view(space, src.linear_springs);
    mirror.angular_springs = create_mirror_view(space, src.angular_springs);
    mirror.triple_springs = create_mirror_view(space, src.triple_springs);
    mirror.fixed_positions = create_mirror_view(space, src.fixed_positions);
    mirror.fixed_poses = create_mirror_view(space, src.fixed_poses);
    mirror.contacts = create_mirror_view(space, src.contacts);
    return mirror;
  }
}

/// \brief create_mirror_view(space, src), holding a copy of src's contents.
template <typename Space, MirrorableType T>
auto create_mirror_view_and_copy(const Space& space, const T& src) {
  auto mirror = create_mirror_view(space, src);
  if constexpr (!std::is_same_v<decltype(mirror), T>) {
    deep_copy(mirror, src);
  }
  return mirror;
}
//@}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_KOKKOSMBODYTYPES_HPP_
