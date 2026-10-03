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
/// \brief Plain data model for rods and the constraint families acting on them.
///
/// Each *Views type stores its fields as flat Kokkos::Views behind accessors. RodViews packs
/// force+torque and velocity+omega into one 6-wide-per-rod buffer each (the generalized-coordinate
/// layout); pose fields stay separate. Per-element accessors (center(i),
/// lambda(p), ...) return a mundy view over the flat storage, not a copy, so
/// `rods.center(i) += dt * rods.velocity(i)` writes straight through. Whole-array accessors (*_view())
/// return the raw Kokkos::View.

// C++ core
#include <concepts>     // for std::constructible_from, std::convertible_to, std::same_as
#include <cstddef>      // for size_t
#include <stdexcept>    // for std::invalid_argument
#include <type_traits>  // for std::is_same_v, std::remove_cvref_t
#include <utility>      // for std::index_sequence, std::make_index_sequence

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/Scalar.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE
#include <mundy_utils/tuple.hpp>         // for mundy::{tuple, make_tuple, get, tuple_size_v}
#include <mundy_utils/type_traits.hpp>   // for mundy::count_type_v

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

/// \brief What a views container holds: bodies, or constraints between them.
enum class ContainerType { BODY, CONSTRAINT };

/// \brief Whether a constraint's multipliers are constrained non-negative (UNILATERAL) or free (BILATERAL).
enum class ConstraintType { UNILATERAL, BILATERAL };

/// \brief The unit a row is measured in.
enum class RowUnit { LENGTH, ANGLE };

/// \brief The storage layout of a views container, specialized beside each one: fields(c), every view c stores.
template <typename T>
struct family_traits {};

/// \brief A container of flat views: constructible from its entry count, with every stored view exposed by fields.
template <typename T>
concept ViewsContainer = std::constructible_from<T, size_t> && requires(const T& c) {
  { T::container_type } -> std::convertible_to<ContainerType>;
  family_traits<T>::fields(c);
  { c.size() } -> std::convertible_to<size_t>;
};

/// \brief A contiguous set of rods (spherocylinders).
///
/// force/torque is each rod's generalized force about its center and velocity/omega its generalized velocity.
template <typename ExecSpace>
class RodViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using vector_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::BODY;

  /// \brief The number of generalized coordinates per rod: [force(3), torque(3)] and [velocity(3), omega(3)].
  static constexpr size_t rows_per_entry = 6;

  /// \brief The unit of each generalized coordinate of a rod: translation, then rotation.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t row_in_entry) {
    return row_in_entry < 3 ? RowUnit::LENGTH : RowUnit::ANGLE;
  }

  RodViews() = default;

  explicit RodViews(size_t num_rods)
      : center_("center", 3 * num_rods),
        orientation_("orientation", 4 * num_rods),
        radius_("radius", num_rods),
        length_("length", num_rods),
        force_torque_("force_torque", rows_per_entry * num_rods),
        velocity_omega_("velocity_omega", rows_per_entry * num_rods) {
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

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

 private:
  vector_view_t center_;
  vector_view_t orientation_;
  vector_view_t radius_;
  vector_view_t length_;
  vector_view_t force_torque_;
  vector_view_t velocity_omega_;
};

template <typename ExecSpace>
struct family_traits<RodViews<ExecSpace>> {
  static auto fields(const RodViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.center_view(), c.orientation_view(), c.radius_view(), c.length_view(),
                               c.force_torque_view(), c.velocity_omega_view());
  }
};

/// \brief Rod-center-to-rod-center Hookean springs. lambda is the spring force magnitude.
template <typename ExecSpace>
class LinearSpringViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 1;

  static constexpr ConstraintType constraint_type = ConstraintType::BILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 2;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t /*row_in_entry*/) {
    return RowUnit::LENGTH;
  }

  LinearSpringViews() = default;

  explicit LinearSpringViews(size_t num_springs)
      : rod_i_("rod_i", num_springs),
        rod_j_("rod_j", num_springs),
        rest_length_("rest_length", num_springs),
        spring_constant_("spring_constant", num_springs),
        lambda_("lambda", rows_per_entry * num_springs) {
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

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, the inverse of its spring constant.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t row) const {
    return 1.0 / spring_constant_(row);
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t rest_length_;
  scalar_view_t spring_constant_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<LinearSpringViews<ExecSpace>> {
  static auto fields(const LinearSpringViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_i_view(), c.rod_j_view(), c.rest_length_view(), c.spring_constant_view(),
                               c.lambda_view());
  }
};

/// \brief Rod-tangent-to-rod-tangent angle bend springs. lambda is the spring torque magnitude.
template <typename ExecSpace>
class AngularSpringViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 1;

  static constexpr ConstraintType constraint_type = ConstraintType::BILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 2;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t /*row_in_entry*/) {
    return RowUnit::ANGLE;
  }

  AngularSpringViews() = default;

  explicit AngularSpringViews(size_t num_springs)
      : rod_i_("rod_i", num_springs),
        rod_j_("rod_j", num_springs),
        rest_angle_("rest_angle", num_springs),
        spring_constant_("spring_constant", num_springs),
        lambda_("lambda", rows_per_entry * num_springs) {
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

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, the inverse of its spring constant.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t row) const {
    return 1.0 / spring_constant_(row);
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t rest_angle_;
  scalar_view_t spring_constant_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<AngularSpringViews<ExecSpace>> {
  static auto fields(const AngularSpringViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_i_view(), c.rod_j_view(), c.rest_angle_view(), c.spring_constant_view(),
                               c.lambda_view());
  }
};

/// \brief Three-point angle bend springs. lambda is the spring torque, k*(angle-rest_angle).
///
/// rod_i/rod_j are the outer points, rod_k the vertex the angle is measured at (node1/node2/node3 of a
/// BEAM_3 element; node3 is the vertex). Constrains the angle at rod_k between the position vectors to
/// rod_i and rod_j -- a position-only, three-body constraint with no orientation dependence, unlike
/// AngularSpringViews. Harmonic in the angle itself, not its cosine. rod_i/rod_j/rod_k need not be distinct.
template <typename ExecSpace>
class TriplePointAngularSpringViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 1;

  static constexpr ConstraintType constraint_type = ConstraintType::BILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 3;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t /*row_in_entry*/) {
    return RowUnit::ANGLE;
  }

  TriplePointAngularSpringViews() = default;

  explicit TriplePointAngularSpringViews(size_t num_springs)
      : rod_i_("rod_i", num_springs),
        rod_j_("rod_j", num_springs),
        rod_k_("rod_k", num_springs),
        rest_angle_("rest_angle", num_springs),
        spring_constant_("spring_constant", num_springs),
        lambda_("lambda", rows_per_entry * num_springs) {
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

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, the inverse of its spring constant.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t row) const {
    return 1.0 / spring_constant_(row);
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  int_view_t rod_k_;
  scalar_view_t rest_angle_;
  scalar_view_t spring_constant_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<TriplePointAngularSpringViews<ExecSpace>> {
  static auto fields(const TriplePointAngularSpringViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_i_view(), c.rod_j_view(), c.rod_k_view(), c.rest_angle_view(),
                               c.spring_constant_view(), c.lambda_view());
  }
};

/// \brief Rod material points held at fixed world locations. lambda is the support reaction.
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
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 3;

  static constexpr ConstraintType constraint_type = ConstraintType::BILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 1;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t /*row_in_entry*/) {
    return RowUnit::LENGTH;
  }

  FixedPositionViews() = default;

  explicit FixedPositionViews(size_t num_anchors)
      : rod_("rod", num_anchors),
        target_point_("target_point", 3 * num_anchors),
        body_offset_("body_offset", 3 * num_anchors),
        compliance_("compliance", rows_per_entry * num_anchors),
        lambda_("lambda", rows_per_entry * num_anchors) {
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

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, its stored compliance.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t row) const {
    return compliance_(row);
  }

 private:
  int_view_t rod_;
  scalar_view_t target_point_;
  scalar_view_t body_offset_;
  scalar_view_t compliance_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<FixedPositionViews<ExecSpace>> {
  static auto fields(const FixedPositionViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_view(), c.target_point_view(), c.body_offset_view(), c.compliance_view(),
                               c.lambda_view());
  }
};

/// \brief Rod material points and orientations held fixed. lambda is the support reaction.
///
/// A clamped rod end: the anchored material point FixedPositionViews holds, plus the rod's
/// orientation held at target_orientation. Six scalar constraints per anchor, laid out as one
/// generalized [force(3), torque(3)] block, so compliance and lambda split the same way.
///
/// The orientation rows carry the exact map from angular velocity to the orientation error's rate,
/// so they are as exact as the position rows.
template <typename ExecSpace>
class FixedPoseViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 6;

  static constexpr ConstraintType constraint_type = ConstraintType::BILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 1;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t row_in_entry) {
    return row_in_entry < 3 ? RowUnit::LENGTH : RowUnit::ANGLE;
  }

  FixedPoseViews() = default;

  explicit FixedPoseViews(size_t num_anchors)
      : rod_("rod", num_anchors),
        target_point_("target_point", 3 * num_anchors),
        target_orientation_("target_orientation", 4 * num_anchors),
        body_offset_("body_offset", 3 * num_anchors),
        compliance_("compliance", rows_per_entry * num_anchors),
        lambda_("lambda", rows_per_entry * num_anchors) {
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

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, its stored compliance.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t row) const {
    return compliance_(row);
  }

 private:
  int_view_t rod_;
  scalar_view_t target_point_;
  scalar_view_t target_orientation_;
  scalar_view_t body_offset_;
  scalar_view_t compliance_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<FixedPoseViews<ExecSpace>> {
  static auto fields(const FixedPoseViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_view(), c.target_point_view(), c.target_orientation_view(), c.body_offset_view(),
                               c.compliance_view(), c.lambda_view());
  }
};

/// \brief Material points of two rods held coincident. lambda is the force on rod_i.
///
/// The two-body peer of FixedPositionViews: the point at body_offset_i in rod_i's frame is held on the
/// point at body_offset_j in rod_j's frame. Each pin is three scalar constraints, one per world axis of
/// p_i - p_j, so its reaction comes back as a vector, and rod_j receives its negation. A pin is rigid: it
/// has no compliance. rod_i and rod_j must differ.
template <typename ExecSpace>
class PinViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 3;

  static constexpr ConstraintType constraint_type = ConstraintType::BILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 2;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t /*row_in_entry*/) {
    return RowUnit::LENGTH;
  }

  PinViews() = default;

  explicit PinViews(size_t num_pins)
      : rod_i_("rod_i", num_pins),
        rod_j_("rod_j", num_pins),
        body_offset_i_("body_offset_i", 3 * num_pins),
        body_offset_j_("body_offset_j", 3 * num_pins),
        lambda_("lambda", rows_per_entry * num_pins) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod_i(int k)         const { return get_scalar<int>(&rod_i_(k)); }
  KOKKOS_INLINE_FUNCTION auto rod_j(int k)         const { return get_scalar<int>(&rod_j_(k)); }
  KOKKOS_INLINE_FUNCTION auto body_offset_i(int k) const { return get_vector3<double>(&body_offset_i_(3 * k)); }
  KOKKOS_INLINE_FUNCTION auto body_offset_j(int k) const { return get_vector3<double>(&body_offset_j_(3 * k)); }
  KOKKOS_INLINE_FUNCTION auto lambda(int k)        const { return get_vector3<double>(&lambda_(3 * k)); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_i_view()            const { return rod_i_; }
  int_view_t rod_j_view()            const { return rod_j_; }
  scalar_view_t body_offset_i_view() const { return body_offset_i_; }
  scalar_view_t body_offset_j_view() const { return body_offset_j_; }
  scalar_view_t lambda_view()        const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_i_.extent(0);
  }

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, zero: a pin is rigid.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t /*row*/) const {
    return 0.0;
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t body_offset_i_;
  scalar_view_t body_offset_j_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<PinViews<ExecSpace>> {
  static auto fields(const PinViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_i_view(), c.rod_j_view(), c.body_offset_i_view(), c.body_offset_j_view(),
                               c.lambda_view());
  }
};

/// \brief Distances between material points of two rods held at a rest length. lambda is the axial force.
///
/// The rigid counterpart of a linear spring, measured between the points at body_offset_i and
/// body_offset_j in the two rods' own frames rather than between rod centres. Each entry is one scalar
/// constraint, |p_j - p_i| - rest_length, whose multiplier is negative in tension. rest_length must be
/// positive, since the distance is not differentiable where it vanishes, and rod_i and rod_j must differ.
template <typename ExecSpace>
class FixedLengthViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 1;

  static constexpr ConstraintType constraint_type = ConstraintType::BILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 2;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t /*row_in_entry*/) {
    return RowUnit::LENGTH;
  }

  FixedLengthViews() = default;

  explicit FixedLengthViews(size_t num_lengths)
      : rod_i_("rod_i", num_lengths),
        rod_j_("rod_j", num_lengths),
        body_offset_i_("body_offset_i", 3 * num_lengths),
        body_offset_j_("body_offset_j", 3 * num_lengths),
        rest_length_("rest_length", num_lengths),
        lambda_("lambda", rows_per_entry * num_lengths) {
  }

  //! \name Per-element accessors: each returns a view into the flat storage below, not a copy.
  //@{
  // clang-format off
  KOKKOS_INLINE_FUNCTION auto rod_i(int k)         const { return get_scalar<int>(&rod_i_(k)); }
  KOKKOS_INLINE_FUNCTION auto rod_j(int k)         const { return get_scalar<int>(&rod_j_(k)); }
  KOKKOS_INLINE_FUNCTION auto body_offset_i(int k) const { return get_vector3<double>(&body_offset_i_(3 * k)); }
  KOKKOS_INLINE_FUNCTION auto body_offset_j(int k) const { return get_vector3<double>(&body_offset_j_(3 * k)); }
  KOKKOS_INLINE_FUNCTION auto rest_length(int k)   const { return get_scalar<double>(&rest_length_(k)); }
  KOKKOS_INLINE_FUNCTION auto lambda(int k)        const { return get_scalar<double>(&lambda_(k)); }
  // clang-format on
  //@}

  //! \name Whole-array accessors
  //@{
  // clang-format off
  int_view_t rod_i_view()            const { return rod_i_; }
  int_view_t rod_j_view()            const { return rod_j_; }
  scalar_view_t body_offset_i_view() const { return body_offset_i_; }
  scalar_view_t body_offset_j_view() const { return body_offset_j_; }
  scalar_view_t rest_length_view()   const { return rest_length_; }
  scalar_view_t lambda_view()        const { return lambda_; }
  // clang-format on
  //@}

  size_t size() const {
    return rod_i_.extent(0);
  }

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, zero: a fixed length is rigid.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t /*row*/) const {
    return 0.0;
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t body_offset_i_;
  scalar_view_t body_offset_j_;
  scalar_view_t rest_length_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<FixedLengthViews<ExecSpace>> {
  static auto fields(const FixedLengthViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_i_view(), c.rod_j_view(), c.body_offset_i_view(), c.body_offset_j_view(),
                               c.rest_length_view(), c.lambda_view());
  }
};

/// \brief Rod-rod unilateral (spherocylinder-spherocylinder) contacts. lambda is the contact force magnitude.
template <typename ExecSpace>
class ContactViews {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using scalar_view_t = Kokkos::View<double*, memory_space>;

  static constexpr ContainerType container_type = ContainerType::CONSTRAINT;

  /// \brief The number of scalar constraints, and so of multipliers, per entry.
  static constexpr size_t rows_per_entry = 1;

  static constexpr ConstraintType constraint_type = ConstraintType::UNILATERAL;

  /// \brief The number of rods each entry couples.
  static constexpr size_t bodies_per_entry = 2;

  /// \brief The unit of each row within an entry.
  KOKKOS_INLINE_FUNCTION static constexpr RowUnit row_unit(size_t /*row_in_entry*/) {
    return RowUnit::LENGTH;
  }

  ContactViews() = default;

  explicit ContactViews(size_t num_contacts)
      : rod_i_("rod_i", num_contacts), rod_j_("rod_j", num_contacts), lambda_("lambda", rows_per_entry * num_contacts) {
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

  /// \brief The number of rows all entries occupy, rows_per_entry * size().
  size_t num_rows() const {
    return rows_per_entry * size();
  }

  /// \brief The compliance of row row, zero: a contact is rigid.
  KOKKOS_INLINE_FUNCTION double row_compliance(size_t /*row*/) const {
    return 0.0;
  }

 private:
  int_view_t rod_i_;
  int_view_t rod_j_;
  scalar_view_t lambda_;
};

template <typename ExecSpace>
struct family_traits<ContactViews<ExecSpace>> {
  static auto fields(const ContactViews<ExecSpace>& c) {
    return ::mundy::make_tuple(c.rod_i_view(), c.rod_j_view(), c.lambda_view());
  }
};

/// \brief A family of constraints: a views container whose entries each own rows_per_entry multipliers.
template <typename F>
concept ConstraintFamily =
    ViewsContainer<F> && (F::container_type == ContainerType::CONSTRAINT) && requires(const F& f) {
      typename F::execution_space;
      { F::rows_per_entry } -> std::convertible_to<size_t>;
      { F::constraint_type } -> std::convertible_to<ConstraintType>;
      { F::bodies_per_entry } -> std::convertible_to<size_t>;
      { F::row_unit(size_t{0}) } -> std::same_as<RowUnit>;
      { f.row_compliance(size_t{0}) } -> std::convertible_to<double>;
      { f.num_rows() } -> std::convertible_to<size_t>;
      f.lambda_view();
    };

/// \brief Whether no type appears twice among Types.
template <typename... Types>
inline constexpr bool are_distinct_v = ((::mundy::count_type_v<Types, Types...> == 1) && ...);

/// \brief Whether every given container has the same execution space; true for none.
template <typename... Containers>
inline constexpr bool share_execution_space_v = true;
template <typename First, typename... Rest>
inline constexpr bool share_execution_space_v<First, Rest...> =
    (std::is_same_v<typename First::execution_space, typename Rest::execution_space> && ...);

/// \brief Every constraint acting on a set of rods, one family per type.
///
/// Families are distinct types sharing one execution space and are reached by type through get<F>(set).
template <typename... Families>
class ConstraintSet {
  static_assert((ConstraintFamily<Families> && ...), "ConstraintSet: every element must be a constraint family.");
  static_assert(are_distinct_v<Families...>, "ConstraintSet: each family type may appear only once.");
  static_assert(share_execution_space_v<Families...>, "ConstraintSet: every family must share one execution space.");

 public:
  ConstraintSet() = default;

  explicit ConstraintSet(const Families&... families)
    requires(sizeof...(Families) > 0)
      : families_(families...) {
  }

  template <typename F, typename... Fs>
  friend const F& get(const ConstraintSet<Fs...>& set);

 private:
  ::mundy::tuple<Families...> families_;
};

/// \brief The family of type F in set.
template <typename F, typename... Fs>
const F& get(const ConstraintSet<Fs...>& set) {
  return ::mundy::get<F>(set.families_);
}

/// \brief A constraint set holding the given families: distinct types sharing one execution space.
template <typename... Families>
  requires(ConstraintFamily<Families> && ...) && are_distinct_v<Families...> && share_execution_space_v<Families...>
ConstraintSet<Families...> make_constraint_set(const Families&... families) {
  return ConstraintSet<Families...>(families...);
}

//! \name Copying between memory spaces
//@{
// The Kokkos view idioms, lifted to the containers above. A mirror's Space is an execution space. create_mirror always
// allocates; create_mirror_view aliases its source when the two share a memory space, as a Kokkos view mirror does.

/// \brief Copy every field of src into dst; the two must have the same size.
template <template <typename> class Views, typename DstSpace, typename SrcSpace>
  requires ViewsContainer<Views<DstSpace>> && ViewsContainer<Views<SrcSpace>>
void deep_copy(const Views<DstSpace>& dst, const Views<SrcSpace>& src) {
  MUNDY_THROW_REQUIRE(dst.size() == src.size(), std::invalid_argument, "mbody::deep_copy: size mismatch.");
  const auto dst_fields = family_traits<Views<DstSpace>>::fields(dst);
  const auto src_fields = family_traits<Views<SrcSpace>>::fields(src);
  [&]<size_t... I>(std::index_sequence<I...>) {
    (Kokkos::deep_copy(::mundy::get<I>(dst_fields), ::mundy::get<I>(src_fields)), ...);
  }(std::make_index_sequence<::mundy::tuple_size_v<std::remove_cvref_t<decltype(dst_fields)>>>{});
}

/// \brief Copy every family of src into the family in the same position of dst; each pair must have the same size.
template <typename... DstFamilies, typename... SrcFamilies>
  requires(sizeof...(DstFamilies) == sizeof...(SrcFamilies))
void deep_copy(const ConstraintSet<DstFamilies...>& dst, const ConstraintSet<SrcFamilies...>& src) {
  (deep_copy(get<DstFamilies>(dst), get<SrcFamilies>(src)), ...);
}

/// \brief Whether T is a ConstraintSet specialization.
template <typename T>
struct is_constraint_set : std::false_type {};
template <typename... Families>
struct is_constraint_set<ConstraintSet<Families...>> : std::true_type {};

template <typename T>
inline constexpr bool is_constraint_set_v = is_constraint_set<T>::value;

/// \brief Matches exactly the containers that can be copied between memory spaces.
template <typename T>
concept MirrorableType = ViewsContainer<T> || is_constraint_set_v<T>;

/// \brief A new zero-initialized container shaped like src in Space's memory.
template <typename Space, template <typename> class Views, typename SrcSpace>
  requires ViewsContainer<Views<SrcSpace>>
Views<Space> create_mirror(const Space& /*space*/, const Views<SrcSpace>& src) {
  static_assert(Kokkos::is_execution_space<Space>::value, "create_mirror: Space must be an execution space.");
  return Views<Space>(src.size());
}

/// \brief A new zero-initialized constraint set shaped like src in Space's memory.
template <typename Space, typename... Families>
auto create_mirror(const Space& space, const ConstraintSet<Families...>& src) {
  return make_constraint_set(create_mirror(space, get<Families>(src))...);
}

/// \brief src itself when Space shares its memory space, else create_mirror(space, src).
template <typename Space, ViewsContainer T>
auto create_mirror_view(const Space& space, const T& src) {
  static_assert(Kokkos::is_execution_space<Space>::value, "create_mirror_view: Space must be an execution space.");
  if constexpr (std::is_same_v<typename Space::memory_space, typename T::memory_space>) {
    return src;
  } else {
    return create_mirror(space, src);
  }
}

/// \brief A constraint set shaped like src in Space's memory, mirroring it family by family.
template <typename Space, typename... Families>
auto create_mirror_view(const Space& space, const ConstraintSet<Families...>& src) {
  return make_constraint_set(create_mirror_view(space, get<Families>(src))...);
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
