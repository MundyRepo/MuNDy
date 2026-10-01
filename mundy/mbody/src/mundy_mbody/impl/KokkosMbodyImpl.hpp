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

#ifndef MUNDY_MBODY_IMPL_KOKKOSMBODYIMPL_HPP_
#define MUNDY_MBODY_IMPL_KOKKOSMBODYIMPL_HPP_

/// \file
/// \brief Operators and geometry kernels for the rod/spring/contact MCQPP solver (mundy::mbody::solve).
///
/// The D/D^T and B/B^T Jacobian operators, the local drag mobility operator, and the geometry kernels
/// that fill them.

// C++ core
#include <concepts>      // for std::same_as
#include <type_traits>  // for std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/distance.hpp>
#include <mundy_geom/primitives.hpp>
#include <mundy_math/Quaternion.hpp>  // for mundy::{quaternion_to_rotation_vector, rotation_vector_jacobian}
#include <mundy_math/Scalar.hpp>
#include <mundy_math/Vector3.hpp>  // for mundy::{cross, perp}
#include <mundy_math/solver_backends.hpp>  // for mundy::{KokkosBackend, LinearOperator, HasScaledApplyMember}
#include <mundy_utils/throw_assert.hpp>
#include <mundy_mbody/KokkosMbodyTypes.hpp>

namespace mundy {

namespace mbody {

namespace impl {

/// \brief Per-body generalized Jacobian mapping a unit Lagrange multiplier to one rod's force/torque.
///
/// For a unit scalar multiplier the row contributes (force, torque) to owner. The arity-one peer of
/// PairGeometry and TripleGeometry, carrying the same 6-wide generalized block per owner that
/// PairGeometry does, since a single-body constraint can restrain orientation as well as position.
/// Nothing about which constraints fill it lives here: a row's Jacobian may be a unit basis vector
/// (a rod's position or orientation held fixed) or configuration-dependent (a material point offset
/// from the rod centre, whose torque row is r x e). force/torque share one 6-wide-per-row buffer;
/// per-element accessors return views into it, *_view() exposes it whole.
template <typename ExecSpace>
class SingleGeometry {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using vector_view_t = Kokkos::View<double*, memory_space>;

  SingleGeometry() = default;

  // Allocates fresh (zero-initialized) Jacobian storage for owner.extent(0) rows.
  explicit SingleGeometry(const int_view_t& owner) : owner_(owner), jacobian_("jacobian", 6 * owner.extent(0)) {
  }

  // Wraps already-populated storage (e.g. the result of concat_single_geometry).
  SingleGeometry(const int_view_t& owner, const vector_view_t& jacobian) : owner_(owner), jacobian_(jacobian) {
  }

  KOKKOS_INLINE_FUNCTION auto owner(int p) const {
    return get_scalar<int>(&owner_(p));
  }
  KOKKOS_INLINE_FUNCTION auto force(int p) const {
    return rod_force(jacobian_, p);
  }
  KOKKOS_INLINE_FUNCTION auto torque(int p) const {
    return rod_torque(jacobian_, p);
  }

  size_t size() const {
    return owner_.extent(0);
  }

  int_view_t owner_view() const {
    return owner_;
  }
  vector_view_t jacobian_view() const {
    return jacobian_;
  }

 private:
  int_view_t owner_;
  vector_view_t jacobian_;
};

/// \brief Whether T is one of the SingleGeometry specializations.
template <typename T>
struct is_single_geometry : std::false_type {};

template <typename ExecSpace>
struct is_single_geometry<SingleGeometry<ExecSpace>> : std::true_type {};

template <typename T>
inline constexpr bool is_single_geometry_v = is_single_geometry<T>::value;

/// \brief Matches exactly the SingleGeometry specializations.
template <typename T>
concept SingleGeometryType = is_single_geometry_v<T>;

static_assert(SingleGeometryType<SingleGeometry<Kokkos::DefaultExecutionSpace>>,
              "SingleGeometry must satisfy SingleGeometryType");
static_assert(!SingleGeometryType<Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "A view must not satisfy SingleGeometryType");
static_assert(!SingleGeometryType<double>, "A scalar must not satisfy SingleGeometryType");

/// \brief Maps a scalar Lagrange multiplier per row to generalized rod-space force/torque (B).
template <typename ExecSpace>
class SingleForceOp {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using x_view_t = Kokkos::View<double*, memory_space>;
  using y_view_t = Kokkos::View<double*, memory_space>;

  SingleForceOp(const SingleGeometry<ExecSpace>& singles, size_t num_rods) : singles_(singles), num_rods_(num_rods) {
  }

  size_t domain_size() const {
    return singles_.size();
  }
  size_t range_size() const {
    return 6 * num_rods_;
  }

  auto make_domain_vector() const {
    return x_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "SingleForceOp_domain"), domain_size());
  }
  auto make_range_vector() const {
    return y_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "SingleForceOp_range"), range_size());
  }

  void apply(const x_view_t& lambda, y_view_t& out) const {
    MUNDY_THROW_ASSERT(lambda.extent(0) == domain_size(), std::invalid_argument,
                       "SingleForceOp: lambda size mismatch.");
    MUNDY_THROW_ASSERT(out.extent(0) == range_size(), std::invalid_argument, "SingleForceOp: out size mismatch.");

    Kokkos::deep_copy(out, 0.0);
    auto singles = singles_;
    Kokkos::parallel_for(
        "SingleForceOp::apply", Kokkos::RangePolicy<ExecSpace>(0, singles.size()), KOKKOS_LAMBDA(const int p) {
          const double lam = lambda(p);
          const int i = singles.owner(p);

          // Rows sharing an owner is the normal case here, not an edge case: one row per constrained
          // degree of freedom of the same rod. Their contributions accumulate, so the add is atomic.
          auto force = rod_force(out, i);
          auto torque = rod_torque(out, i);
          atomic_add(&force, lam * singles.force(p));
          atomic_add(&torque, lam * singles.torque(p));
        });
  }

 private:
  SingleGeometry<ExecSpace> singles_;
  size_t num_rods_;
};

/// \brief Maps rod velocity/omega to each row's scalar constraint rate (B^T).
///
/// The exact transpose of SingleForceOp.
template <typename ExecSpace>
class SingleForceOpT {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using x_view_t = Kokkos::View<double*, memory_space>;
  using y_view_t = Kokkos::View<double*, memory_space>;

  SingleForceOpT(const SingleGeometry<ExecSpace>& singles, size_t num_rods) : singles_(singles), num_rods_(num_rods) {
  }

  size_t domain_size() const {
    return 6 * num_rods_;
  }
  size_t range_size() const {
    return singles_.size();
  }

  auto make_domain_vector() const {
    return y_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "SingleForceOpT_domain"), domain_size());
  }
  auto make_range_vector() const {
    return x_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "SingleForceOpT_range"), range_size());
  }

  void apply(const y_view_t& vel_omega, x_view_t& rate) const {
    MUNDY_THROW_ASSERT(vel_omega.extent(0) == domain_size(), std::invalid_argument,
                       "SingleForceOpT: vel_omega size mismatch.");
    MUNDY_THROW_ASSERT(rate.extent(0) == range_size(), std::invalid_argument, "SingleForceOpT: rate size mismatch.");

    auto singles = singles_;
    Kokkos::parallel_for(
        "SingleForceOpT::apply", Kokkos::RangePolicy<ExecSpace>(0, singles.size()), KOKKOS_LAMBDA(const int p) {
          const int i = singles.owner(p);
          const Vector3d vel = rod_velocity(vel_omega, i);
          const Vector3d omega = rod_omega(vel_omega, i);

          rate(p) = dot(singles.force(p), vel) + dot(singles.torque(p), omega);
        });
  }

 private:
  SingleGeometry<ExecSpace> singles_;
  size_t num_rods_;
};

static_assert(::mundy::LinearOperator<::mundy::KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                      SingleForceOp<Kokkos::DefaultExecutionSpace>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "SingleForceOp must satisfy ::mundy::LinearOperator");
static_assert(::mundy::LinearOperator<::mundy::KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                      SingleForceOpT<Kokkos::DefaultExecutionSpace>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "SingleForceOpT must satisfy ::mundy::LinearOperator");

/// \brief Per-pair generalized Jacobian mapping a unit Lagrange multiplier to rod force/torque.
///
/// For a unit scalar multiplier the pair contributes (force_i, torque_i) to owner_i and (force_j,
/// torque_j) to owner_j. One representation covers:
///   - contacts:        force_{i,j} = -+n,  torque_{i,j} = -+r_{i,j} x n
///   - linear springs:  force_{i,j} = -+d,  torque_{i,j} = 0
///   - angular springs: force_{i,j} = 0,    torque_{i,j} = -+a
/// force_i/torque_i (and force_j/torque_j) share one 6-wide-per-pair buffer; per-element accessors
/// return views into it, *_view() expose it whole.
template <typename ExecSpace>
class PairGeometry {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using vector_view_t = Kokkos::View<double*, memory_space>;

  PairGeometry() = default;

  // Allocates fresh (zero-initialized) Jacobian storage for owner_i.extent(0) pairs.
  PairGeometry(const int_view_t& owner_i, const int_view_t& owner_j)
      : owner_i_(owner_i),
        owner_j_(owner_j),
        jacobian_i_("jacobian_i", 6 * owner_i.extent(0)),
        jacobian_j_("jacobian_j", 6 * owner_i.extent(0)) {
  }

  // Wraps already-populated storage (e.g. the result of concat_pair_geometry).
  PairGeometry(const int_view_t& owner_i, const int_view_t& owner_j, const vector_view_t& jacobian_i,
               const vector_view_t& jacobian_j)
      : owner_i_(owner_i), owner_j_(owner_j), jacobian_i_(jacobian_i), jacobian_j_(jacobian_j) {
  }

  KOKKOS_INLINE_FUNCTION auto owner_i(int p) const {
    return get_scalar<int>(&owner_i_(p));
  }
  KOKKOS_INLINE_FUNCTION auto owner_j(int p) const {
    return get_scalar<int>(&owner_j_(p));
  }
  KOKKOS_INLINE_FUNCTION auto force_i(int p) const {
    return rod_force(jacobian_i_, p);
  }
  KOKKOS_INLINE_FUNCTION auto torque_i(int p) const {
    return rod_torque(jacobian_i_, p);
  }
  KOKKOS_INLINE_FUNCTION auto force_j(int p) const {
    return rod_force(jacobian_j_, p);
  }
  KOKKOS_INLINE_FUNCTION auto torque_j(int p) const {
    return rod_torque(jacobian_j_, p);
  }

  size_t size() const {
    return owner_i_.extent(0);
  }

  int_view_t owner_i_view() const {
    return owner_i_;
  }
  int_view_t owner_j_view() const {
    return owner_j_;
  }
  vector_view_t jacobian_i_view() const {
    return jacobian_i_;
  }
  vector_view_t jacobian_j_view() const {
    return jacobian_j_;
  }

 private:
  int_view_t owner_i_;
  int_view_t owner_j_;
  vector_view_t jacobian_i_;
  vector_view_t jacobian_j_;
};

/// \brief Whether T is one of the PairGeometry specializations.
template <typename T>
struct is_pair_geometry : std::false_type {};

template <typename ExecSpace>
struct is_pair_geometry<PairGeometry<ExecSpace>> : std::true_type {};

template <typename T>
inline constexpr bool is_pair_geometry_v = is_pair_geometry<T>::value;

/// \brief Matches exactly the PairGeometry specializations.
template <typename T>
concept PairGeometryType = is_pair_geometry_v<T>;

static_assert(PairGeometryType<PairGeometry<Kokkos::DefaultExecutionSpace>>,
              "PairGeometry must satisfy PairGeometryType");
static_assert(!PairGeometryType<Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "A view must not satisfy PairGeometryType");
static_assert(!PairGeometryType<double>, "A scalar must not satisfy PairGeometryType");
static_assert(!PairGeometryType<SingleGeometry<Kokkos::DefaultExecutionSpace>>,
              "The arity-one and arity-two geometries must not be confusable");

/// \brief Maps a scalar Lagrange multiplier per pair to generalized rod-space force/torque (D or B).
template <typename ExecSpace>
class PairForceOp {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using x_view_t = Kokkos::View<double*, memory_space>;
  using y_view_t = Kokkos::View<double*, memory_space>;

  PairForceOp(const PairGeometry<ExecSpace>& pairs, size_t num_rods) : pairs_(pairs), num_rods_(num_rods) {
  }

  size_t domain_size() const {
    return pairs_.size();
  }
  size_t range_size() const {
    return 6 * num_rods_;
  }

  auto make_domain_vector() const {
    return x_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "PairForceOp_domain"), domain_size());
  }
  auto make_range_vector() const {
    return y_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "PairForceOp_range"), range_size());
  }

  void apply(const x_view_t& lambda, y_view_t& out) const {
    MUNDY_THROW_ASSERT(lambda.extent(0) == domain_size(), std::invalid_argument, "PairForceOp: lambda size mismatch.");
    MUNDY_THROW_ASSERT(out.extent(0) == range_size(), std::invalid_argument, "PairForceOp: out size mismatch.");

    Kokkos::deep_copy(out, 0.0);
    auto pairs = pairs_;
    Kokkos::parallel_for(
        "PairForceOp::apply", Kokkos::RangePolicy<ExecSpace>(0, pairs.size()), KOKKOS_LAMBDA(const int p) {
          const double lam = lambda(p);
          const int i = pairs.owner_i(p);
          const int j = pairs.owner_j(p);

          auto force_i = rod_force(out, i);
          auto torque_i = rod_torque(out, i);
          auto force_j = rod_force(out, j);
          auto torque_j = rod_torque(out, j);
          atomic_add(&force_i, lam * pairs.force_i(p));
          atomic_add(&torque_i, lam * pairs.torque_i(p));
          atomic_add(&force_j, lam * pairs.force_j(p));
          atomic_add(&torque_j, lam * pairs.torque_j(p));
        });
  }

 private:
  PairGeometry<ExecSpace> pairs_;
  size_t num_rods_;
};

/// \brief Maps rod velocity/omega to each pair's scalar constraint rate (D^T or B^T).
///
/// The exact transpose of PairForceOp.
template <typename ExecSpace>
class PairForceOpT {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using x_view_t = Kokkos::View<double*, memory_space>;
  using y_view_t = Kokkos::View<double*, memory_space>;

  PairForceOpT(const PairGeometry<ExecSpace>& pairs, size_t num_rods) : pairs_(pairs), num_rods_(num_rods) {
  }

  size_t domain_size() const {
    return 6 * num_rods_;
  }
  size_t range_size() const {
    return pairs_.size();
  }

  auto make_domain_vector() const {
    return y_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "PairForceOpT_domain"), domain_size());
  }
  auto make_range_vector() const {
    return x_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "PairForceOpT_range"), range_size());
  }

  void apply(const y_view_t& vel_omega, x_view_t& rate) const {
    MUNDY_THROW_ASSERT(vel_omega.extent(0) == domain_size(), std::invalid_argument,
                       "PairForceOpT: vel_omega size mismatch.");
    MUNDY_THROW_ASSERT(rate.extent(0) == range_size(), std::invalid_argument, "PairForceOpT: rate size mismatch.");

    auto pairs = pairs_;
    Kokkos::parallel_for(
        "PairForceOpT::apply", Kokkos::RangePolicy<ExecSpace>(0, pairs.size()), KOKKOS_LAMBDA(const int p) {
          const int i = pairs.owner_i(p);
          const int j = pairs.owner_j(p);
          const Vector3d vel_i = rod_velocity(vel_omega, i);
          const Vector3d omega_i = rod_omega(vel_omega, i);
          const Vector3d vel_j = rod_velocity(vel_omega, j);
          const Vector3d omega_j = rod_omega(vel_omega, j);

          rate(p) = dot(pairs.force_i(p), vel_i) + dot(pairs.torque_i(p), omega_i) + dot(pairs.force_j(p), vel_j) +
                    dot(pairs.torque_j(p), omega_j);
        });
  }

 private:
  PairGeometry<ExecSpace> pairs_;
  size_t num_rods_;
};

static_assert(::mundy::LinearOperator<::mundy::KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                      PairForceOp<Kokkos::DefaultExecutionSpace>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "PairForceOp must satisfy ::mundy::LinearOperator");
static_assert(::mundy::LinearOperator<::mundy::KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                      PairForceOpT<Kokkos::DefaultExecutionSpace>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "PairForceOpT must satisfy ::mundy::LinearOperator");

/// \brief Per-triple generalized Jacobian for a three-point (vertex-angle) bend spring.
///
/// For a unit scalar multiplier the triple contributes force_1/force_2/force_3 to owner_1/owner_2/
/// owner_3 -- no torque, since this constraint is a function of the three rods' positions alone (see
/// compute_triple_point_angular_spring_geometry). owner_3 is the vertex the angle is measured at.
template <typename ExecSpace>
class TripleGeometry {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using int_view_t = Kokkos::View<int*, memory_space>;
  using vector_view_t = Kokkos::View<double*, memory_space>;

  TripleGeometry() = default;

  // Allocates fresh (zero-initialized) Jacobian storage for owner_1.extent(0) triples.
  TripleGeometry(const int_view_t& owner_1, const int_view_t& owner_2, const int_view_t& owner_3)
      : owner_1_(owner_1),
        owner_2_(owner_2),
        owner_3_(owner_3),
        jacobian_1_("jacobian_1", 3 * owner_1.extent(0)),
        jacobian_2_("jacobian_2", 3 * owner_1.extent(0)),
        jacobian_3_("jacobian_3", 3 * owner_1.extent(0)) {
  }

  KOKKOS_INLINE_FUNCTION auto owner_1(int p) const {
    return get_scalar<int>(&owner_1_(p));
  }
  KOKKOS_INLINE_FUNCTION auto owner_2(int p) const {
    return get_scalar<int>(&owner_2_(p));
  }
  KOKKOS_INLINE_FUNCTION auto owner_3(int p) const {
    return get_scalar<int>(&owner_3_(p));
  }
  KOKKOS_INLINE_FUNCTION auto force_1(int p) const {
    return get_vector3<double>(&jacobian_1_(3 * p));
  }
  KOKKOS_INLINE_FUNCTION auto force_2(int p) const {
    return get_vector3<double>(&jacobian_2_(3 * p));
  }
  KOKKOS_INLINE_FUNCTION auto force_3(int p) const {
    return get_vector3<double>(&jacobian_3_(3 * p));
  }

  size_t size() const {
    return owner_1_.extent(0);
  }

 private:
  int_view_t owner_1_;
  int_view_t owner_2_;
  int_view_t owner_3_;
  vector_view_t jacobian_1_;
  vector_view_t jacobian_2_;
  vector_view_t jacobian_3_;
};

/// \brief Maps a scalar Lagrange multiplier per triple to rod-space force (torque zero, see TripleGeometry).
template <typename ExecSpace>
class TripleForceOp {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using x_view_t = Kokkos::View<double*, memory_space>;
  using y_view_t = Kokkos::View<double*, memory_space>;

  TripleForceOp(const TripleGeometry<ExecSpace>& triples, size_t num_rods) : triples_(triples), num_rods_(num_rods) {
  }

  size_t domain_size() const {
    return triples_.size();
  }
  size_t range_size() const {
    return 6 * num_rods_;
  }

  auto make_domain_vector() const {
    return x_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "TripleForceOp_domain"), domain_size());
  }
  auto make_range_vector() const {
    return y_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "TripleForceOp_range"), range_size());
  }

  void apply(const x_view_t& lambda, y_view_t& out) const {
    MUNDY_THROW_ASSERT(lambda.extent(0) == domain_size(), std::invalid_argument,
                       "TripleForceOp: lambda size mismatch.");
    MUNDY_THROW_ASSERT(out.extent(0) == range_size(), std::invalid_argument, "TripleForceOp: out size mismatch.");

    Kokkos::deep_copy(out, 0.0);
    auto triples = triples_;
    Kokkos::parallel_for(
        "TripleForceOp::apply", Kokkos::RangePolicy<ExecSpace>(0, triples.size()), KOKKOS_LAMBDA(const int p) {
          const double lam = lambda(p);
          const int i = triples.owner_1(p);
          const int j = triples.owner_2(p);
          const int k = triples.owner_3(p);

          auto force_i = rod_force(out, i);
          auto force_j = rod_force(out, j);
          auto force_k = rod_force(out, k);
          atomic_add(&force_i, lam * triples.force_1(p));
          atomic_add(&force_j, lam * triples.force_2(p));
          atomic_add(&force_k, lam * triples.force_3(p));
        });
  }

 private:
  TripleGeometry<ExecSpace> triples_;
  size_t num_rods_;
};

/// \brief Maps rod velocity to each triple's scalar constraint rate (transpose of TripleForceOp).
///
/// Omega never enters: the constraint (a vertex angle between two position vectors) has no dependence
/// on any rod's orientation.
template <typename ExecSpace>
class TripleForceOpT {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using x_view_t = Kokkos::View<double*, memory_space>;
  using y_view_t = Kokkos::View<double*, memory_space>;

  TripleForceOpT(const TripleGeometry<ExecSpace>& triples, size_t num_rods) : triples_(triples), num_rods_(num_rods) {
  }

  size_t domain_size() const {
    return 6 * num_rods_;
  }
  size_t range_size() const {
    return triples_.size();
  }

  auto make_domain_vector() const {
    return y_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "TripleForceOpT_domain"), domain_size());
  }
  auto make_range_vector() const {
    return x_view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "TripleForceOpT_range"), range_size());
  }

  void apply(const y_view_t& vel_omega, x_view_t& rate) const {
    MUNDY_THROW_ASSERT(vel_omega.extent(0) == domain_size(), std::invalid_argument,
                       "TripleForceOpT: vel_omega size mismatch.");
    MUNDY_THROW_ASSERT(rate.extent(0) == range_size(), std::invalid_argument, "TripleForceOpT: rate size mismatch.");

    auto triples = triples_;
    Kokkos::parallel_for(
        "TripleForceOpT::apply", Kokkos::RangePolicy<ExecSpace>(0, triples.size()), KOKKOS_LAMBDA(const int p) {
          const int i = triples.owner_1(p);
          const int j = triples.owner_2(p);
          const int k = triples.owner_3(p);
          const Vector3d vel_i = rod_velocity(vel_omega, i);
          const Vector3d vel_j = rod_velocity(vel_omega, j);
          const Vector3d vel_k = rod_velocity(vel_omega, k);

          rate(p) = dot(triples.force_1(p), vel_i) + dot(triples.force_2(p), vel_j) + dot(triples.force_3(p), vel_k);
        });
  }

 private:
  TripleGeometry<ExecSpace> triples_;
  size_t num_rods_;
};

static_assert(::mundy::LinearOperator<::mundy::KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                      TripleForceOp<Kokkos::DefaultExecutionSpace>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "TripleForceOp must satisfy ::mundy::LinearOperator");
static_assert(::mundy::LinearOperator<::mundy::KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                      TripleForceOpT<Kokkos::DefaultExecutionSpace>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "TripleForceOpT must satisfy ::mundy::LinearOperator");

/// \brief Per-rod local drag mobility (force/torque -> velocity/omega), no hydrodynamic coupling.
template <typename ExecSpace>
class LocalDragMobilityOp {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view_t = Kokkos::View<double*, memory_space>;

  LocalDragMobilityOp(double viscosity, const RodViews<ExecSpace>& rods) : viscosity_(viscosity), rods_(rods) {
  }

  size_t domain_size() const {
    return 6 * rods_.size();
  }
  size_t range_size() const {
    return 6 * rods_.size();
  }

  auto make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "M_domain"), domain_size());
  }
  auto make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "M_range"), range_size());
  }

  void apply(const view_t& force_torque, view_t& vel_omega) const {
    apply(1.0, force_torque, 0.0, vel_omega);
  }

  /// \brief vel_omega := alpha * M(force_torque) + beta * vel_omega.
  ///
  /// alpha == 0 skips the per-rod drag-coefficient computation entirely.
  void apply(double alpha, const view_t& force_torque, double beta, view_t& vel_omega) const {
    MUNDY_THROW_ASSERT(force_torque.extent(0) == domain_size(), std::invalid_argument,
                       "LocalDragMobilityOp: size mismatch.");
    const bool alpha_is_zero = Kokkos::abs(alpha) < get_zero_tolerance<double>();
    const bool beta_is_zero = Kokkos::abs(beta) < get_zero_tolerance<double>();

    if (alpha_is_zero) {
      if (beta_is_zero) {
        Kokkos::deep_copy(vel_omega, 0.0);
      } else {
        auto vel_omega_l = vel_omega;
        Kokkos::parallel_for(
            "LocalDragMobilityOp::apply(beta-only)", Kokkos::RangePolicy<ExecSpace>(0, vel_omega.extent(0)),
            KOKKOS_LAMBDA(const int i) { vel_omega_l(i) *= beta; });
      }
      return;
    }

    const double viscosity = viscosity_;
    auto rods = rods_;

    constexpr double pi = Kokkos::numbers::pi_v<double>;
    const double inv_four_pi_visc = 1.0 / (4.0 * pi * viscosity);
    const double inv_two_pi_visc = 1.0 / (2.0 * pi * viscosity);
    const double inv_pi_visc = 1.0 / (pi * viscosity);

    Kokkos::parallel_for(
        "LocalDragMobilityOp::apply", Kokkos::RangePolicy<ExecSpace>(0, rods.size()), KOKKOS_LAMBDA(const int i) {
          const double length = rods.length(i);
          const double radius = rods.radius(i);
          const Vector3d tangent = rods.orientation(i) * Vector3d{0.0, 0.0, 1.0};

          const double lprime = length + 2.0 * radius;
          const double p = lprime / (2.0 * radius);
          const double log_p = Kokkos::log(p);
          const double inv_p = 1.0 / p;
          const double inv_p2 = inv_p * inv_p;
          const double inv_lprime = 1.0 / lprime;
          const double inv_lprime3 = inv_lprime * inv_lprime * inv_lprime;

          const double inv_drag_perp = (log_p + 0.839 + 0.185 * inv_p + 0.233 * inv_p2) * inv_lprime * inv_four_pi_visc;
          const double inv_drag_para = (log_p - 0.207 + 0.98 * inv_p - 0.133 * inv_p2) * inv_lprime * inv_two_pi_visc;
          const double inv_drag_rot = 3.0 * (log_p - 0.662 + 0.917 * inv_p - 0.05 * inv_p2) * inv_lprime3 * inv_pi_visc;

          const Vector3d force = rod_force(force_torque, i);
          const Vector3d torque = rod_torque(force_torque, i);

          const Vector3d force_para = dot(force, tangent) * tangent;
          const Vector3d force_perp = force - force_para;

          const Vector3d velocity = inv_drag_perp * force_perp + inv_drag_para * force_para;
          const Vector3d omega = inv_drag_rot * torque;

          if (beta_is_zero) {
            rod_velocity(vel_omega, i) = alpha * velocity;
            rod_omega(vel_omega, i) = alpha * omega;
          } else {
            rod_velocity(vel_omega, i) = alpha * velocity + beta * rod_velocity(vel_omega, i);
            rod_omega(vel_omega, i) = alpha * omega + beta * rod_omega(vel_omega, i);
          }
        });
  }

 private:
  double viscosity_;
  RodViews<ExecSpace> rods_;
};

static_assert(::mundy::LinearOperator<::mundy::KokkosBackend<Kokkos::DefaultExecutionSpace>,
                                      LocalDragMobilityOp<Kokkos::DefaultExecutionSpace>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                      Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "LocalDragMobilityOp must satisfy ::mundy::LinearOperator");
static_assert(::mundy::HasScaledApplyMember<LocalDragMobilityOp<Kokkos::DefaultExecutionSpace>, double,
                                            Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                            Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "LocalDragMobilityOp must satisfy ::mundy::HasScaledApplyMember");

//! \name Geometry kernels: fill a PairGeometry (and the corresponding q/b contribution) from rod state
//@{

/// \brief A rank-1 view of doubles a geometry kernel writes its constraint values into.
///
/// Taken by const reference: constness of a Kokkos handle does not reach the data, so a kernel still
/// writes through it. Callers pass either a whole view or a subview of a larger one, which is how a
/// family's rows are filled in place rather than allocated and copied.
template <typename T>
concept ConstraintValueView =
    Kokkos::is_view<T>::value && (T::rank == 1) && std::same_as<typename T::value_type, double>;

/// \brief Contact normal/lever-arm Jacobian and initial signed separation, via spherocylinder centerlines.
template <typename ExecSpace, typename Sep0View>
  requires ConstraintValueView<Sep0View>
PairGeometry<ExecSpace> compute_contact_geometry(const RodViews<ExecSpace>& rods,
                                                 const ContactViews<ExecSpace>& contacts, const Sep0View& sep0) {
  const size_t n = contacts.size();
  MUNDY_THROW_ASSERT(sep0.extent(0) == n, std::invalid_argument,
                     "compute_contact_geometry: sep0 must have one entry per contact.");
  PairGeometry<ExecSpace> geo(contacts.rod_i_view(), contacts.rod_j_view());

  auto rods_l = rods;
  auto geo_l = geo;
  auto sep0_l = sep0;
  Kokkos::parallel_for(
      "compute_contact_geometry", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int p) {
        const int i = geo_l.owner_i(p);
        const int j = geo_l.owner_j(p);

        const double radius_i = rods_l.radius(i);
        const double radius_j = rods_l.radius(j);
        const double length_i = rods_l.length(i);
        const double length_j = rods_l.length(j);
        const Vector3d tangent_i = rods_l.orientation(i) * Vector3d{0.0, 0.0, 1.0};
        const Vector3d tangent_j = rods_l.orientation(j) * Vector3d{0.0, 0.0, 1.0};
        const Vector3d pos_i = rods_l.center(i);
        const Vector3d pos_j = rods_l.center(j);

        const LineSegment<double> centerline_i{pos_i - 0.5 * length_i * tangent_i, pos_i + 0.5 * length_i * tangent_i};
        const LineSegment<double> centerline_j{pos_j - 0.5 * length_j * tangent_j, pos_j + 0.5 * length_j * tangent_j};

        Point<double> contact_point_i, contact_point_j;
        double arc_i, arc_j;
        Vector3d rij;
        const double dist = distance(centerline_i, centerline_j, contact_point_i, contact_point_j, arc_i, arc_j, rij);
        MUNDY_THROW_ASSERT(dist > 1e-12, std::runtime_error, "Rod-rod contact distance is nearly degenerate.");

        const Vector3d n_ij = rij / dist;
        const Vector3d rel_i = contact_point_i - pos_i;
        const Vector3d rel_j = contact_point_j - pos_j;

        geo_l.force_i(p) = -n_ij;
        geo_l.torque_i(p) = -cross(rel_i, n_ij);
        geo_l.force_j(p) = n_ij;
        geo_l.torque_j(p) = cross(rel_j, n_ij);
        sep0_l(p) = dist - radius_i - radius_j;
      });

  return geo;
}

/// \brief Linear spring direction Jacobian and initial stretch, via rod centers.
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
PairGeometry<ExecSpace> compute_linear_spring_geometry(const RodViews<ExecSpace>& rods, const LinearSpringViews<ExecSpace>& springs,
                                                       const B0View& b0) {
  const size_t n = springs.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == n, std::invalid_argument,
                     "compute_linear_spring_geometry: b0 must have one entry per spring.");
  PairGeometry<ExecSpace> geo(springs.rod_i_view(), springs.rod_j_view());

  auto rods_l = rods;
  auto geo_l = geo;
  auto springs_l = springs;
  auto b0_l = b0;
  Kokkos::parallel_for(
      "compute_linear_spring_geometry", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int p) {
        const int i = geo_l.owner_i(p);
        const int j = geo_l.owner_j(p);
        const Vector3d sep = rods_l.center(j) - rods_l.center(i);
        const double dist = norm(sep);
        MUNDY_THROW_ASSERT(dist > 1e-12, std::runtime_error, "Linear spring endpoints are nearly coincident.");
        const Vector3d dir = sep / dist;

        geo_l.force_i(p) = -dir;
        geo_l.torque_i(p) = Vector3d{0.0, 0.0, 0.0};
        geo_l.force_j(p) = dir;
        geo_l.torque_j(p) = Vector3d{0.0, 0.0, 0.0};
        b0_l(p) = dist - springs_l.rest_length(p);
      });

  return geo;
}

/// \brief Angular spring axis Jacobian and initial bend angle, via rod tangents (orientation * e_z).
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
PairGeometry<ExecSpace> compute_angular_spring_geometry(const RodViews<ExecSpace>& rods, const AngularSpringViews<ExecSpace>& springs,
                                                       const B0View& b0) {
  const size_t n = springs.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == n, std::invalid_argument,
                     "compute_angular_spring_geometry: b0 must have one entry per spring.");
  PairGeometry<ExecSpace> geo(springs.rod_i_view(), springs.rod_j_view());

  auto rods_l = rods;
  auto geo_l = geo;
  auto springs_l = springs;
  auto b0_l = b0;
  Kokkos::parallel_for(
      "compute_angular_spring_geometry", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int p) {
        const int i = geo_l.owner_i(p);
        const int j = geo_l.owner_j(p);
        const Vector3d tangent_i = rods_l.orientation(i) * Vector3d{0.0, 0.0, 1.0};
        const Vector3d tangent_j = rods_l.orientation(j) * Vector3d{0.0, 0.0, 1.0};

        const Vector3d cross_ij = cross(tangent_i, tangent_j);
        const double cross_norm = norm(cross_ij);
        // At tangent alignment the bend axis is a genuine 0/0 (cross_norm ~ sin(angle) -> 0) but only
        // underdetermined, not undefined: every unit vector perpendicular to the shared tangent is an
        // equally valid axis, so pick one deterministically instead of dividing by ~0. (minor_angle
        // itself stays well-behaved there; acos clamps -- see Vector.hpp.)
        const Vector3d axis = (cross_norm > 1e-12) ? cross_ij / cross_norm : perp(tangent_i);
        const double angle = minor_angle(tangent_i, tangent_j);

        geo_l.force_i(p) = Vector3d{0.0, 0.0, 0.0};
        geo_l.torque_i(p) = -axis;
        geo_l.force_j(p) = Vector3d{0.0, 0.0, 0.0};
        geo_l.torque_j(p) = axis;
        b0_l(p) = angle - springs_l.rest_angle(p);
      });

  return geo;
}

/// \brief Three-point bend spring Jacobian and initial angle, via rod centers.
///
/// The angle is measured at rod_k (the vertex, node3 in BEAM_3 numbering) between the position
/// vectors to rod_i and rod_j. Harmonic in the angle itself (0.5*k*(angle-rest_angle)^2), not in
/// cos(angle): a cosine-harmonic spring's stiffness d^2U/d(angle)^2 = k*sin^2(rest_angle) vanishes at
/// rest_angle=pi, so it cannot resist small bends of a filament that is straight at rest. b0 = angle -
/// rest_angle is the constraint value; the per-rod Jacobians are d(angle)/dx = cross(v1, n_hat)/|v1|^2
/// (mirror image for rod_j; force_3 = -(force_1 + force_2)), with v1 := center(rod_i) - center(rod_k)
/// and n_hat the unit normal to the v1/v2 plane. n_hat is degenerate at angle 0 or pi and is resolved
/// by mundy::perp, exactly as in compute_angular_spring_geometry. Position-only: no
/// orientation, no torque, so rod_i/rod_j/rod_k need not be distinct (only rod_i/rod_k and rod_j/rod_k
/// must not coincide in position, checked below).
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
TripleGeometry<ExecSpace> compute_triple_point_angular_spring_geometry(
    const RodViews<ExecSpace>& rods, const TriplePointAngularSpringViews<ExecSpace>& springs, const B0View& b0) {
  const size_t n = springs.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == n, std::invalid_argument,
                     "compute_triple_point_angular_spring_geometry: b0 must have one entry per spring.");
  TripleGeometry<ExecSpace> geo(springs.rod_i_view(), springs.rod_j_view(), springs.rod_k_view());

  auto rods_l = rods;
  auto geo_l = geo;
  auto springs_l = springs;
  auto b0_l = b0;
  Kokkos::parallel_for(
      "compute_triple_point_angular_spring_geometry", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const int p) {
        const int i = geo_l.owner_1(p);
        const int j = geo_l.owner_2(p);
        const int k = geo_l.owner_3(p);
        const Vector3d v1 = rods_l.center(i) - rods_l.center(k);  // vertex k -> outer point i
        const Vector3d v2 = rods_l.center(j) - rods_l.center(k);  // vertex k -> outer point j

        const double d1_sq = dot(v1, v1);
        const double d2_sq = dot(v2, v2);
        MUNDY_THROW_ASSERT(d1_sq > 1e-24 && d2_sq > 1e-24, std::runtime_error,
                           "Triple-point angular spring: an outer point coincides with the vertex.");

        const Vector3d cross_v1v2 = cross(v1, v2);
        const double cross_norm = norm(cross_v1v2);
        // Relative (scale-invariant) threshold: cross_norm = d1*d2*sin(angle), so cross_norm/(d1*d2)
        // is sin(angle) directly. An absolute cutoff would miss near-degenerate cross products at large
        // length scales and reject valid ones at small scales.
        const double d1_d2 = Kokkos::sqrt(d1_sq * d2_sq);
        const Vector3d n_hat =
            (cross_norm > 1e-9 * d1_d2) ? cross_v1v2 / cross_norm : perp(v1);
        const double angle = minor_angle(v1, v2);

        const Vector3d force_1 = cross(v1, n_hat) / d1_sq;
        const Vector3d force_2 = -cross(v2, n_hat) / d2_sq;
        const Vector3d force_3 = -(force_1 + force_2);

        geo_l.force_1(p) = force_1;
        geo_l.force_2(p) = force_2;
        geo_l.force_3(p) = force_3;
        b0_l(p) = angle - springs_l.rest_angle(p);
      });

  return geo;
}
//@}

/// \brief Fixed-position anchor Jacobian and initial offset from target, via rod poses.
///
/// The anchored material point is p = center + R(q) r_local, moving at v + omega x r_world. Row c of
/// an anchor constrains p's world component c, so its Jacobian is force = e_c and
/// torque = r_world x e_c -- the same lever-arm form a contact takes -- and its constraint value is
/// (p - target)[c]. A zero body offset leaves the torque rows zero and anchors the rod centre.
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
SingleGeometry<ExecSpace> compute_fixed_position_geometry(const RodViews<ExecSpace>& rods,
                                                          const FixedPositionViews<ExecSpace>& anchors,
                                                          const B0View& b0) {
  using memory_space = typename ExecSpace::memory_space;
  const size_t num_anchors = anchors.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == anchors.num_constraints(), std::invalid_argument,
                     "compute_fixed_position_geometry: b0 must have three entries per anchor.");

  Kokkos::View<int*, memory_space> owner("fixed_position_owner", anchors.num_constraints());
  SingleGeometry<ExecSpace> geo(owner);

  auto rods_l = rods;
  auto geo_l = geo;
  auto anchors_l = anchors;
  auto owner_l = owner;
  auto b0_l = b0;
  Kokkos::parallel_for(
      "compute_fixed_position_geometry", Kokkos::RangePolicy<ExecSpace>(0, num_anchors), KOKKOS_LAMBDA(const int a) {
        const int i = anchors_l.rod(a);
        const Vector3d r_world = rods_l.orientation(i) * anchors_l.body_offset(a);
        const Vector3d offset = rods_l.center(i) + r_world - anchors_l.target_point(a);

        for (int c = 0; c < 3; ++c) {
          const int row = 3 * a + c;
          Vector3d axis{0.0, 0.0, 0.0};
          axis[c] = 1.0;

          owner_l(row) = i;
          geo_l.force(row) = axis;
          geo_l.torque(row) = cross(r_world, axis);
          b0_l(row) = offset[c];
        }
      });

  return geo;
}

/// \brief The matrix carrying a world-frame angular velocity to the rate of a rotation vector.
///
/// A rotation whose vector is theta, turning under an angular velocity omega expressed in the fixed
/// frame, has d(theta)/dt = J(theta) omega, with
///
///   J(theta) = I - [theta]_x / 2 + a [theta]_x^2,   a = (1 - (t/2) cot(t/2)) / t^2,   t = |theta|
///
/// This is SO(3)'s inverse left Jacobian.
///
/// \param[in] rotation_vector The rotation vector (axis scaled by angle).
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr Matrix3<std::remove_const_t<T>> rotation_vector_jacobian(
    const AVector3<T, Accessor>& rotation_vector) {
  using Scalar = std::remove_const_t<T>;
  const Scalar t = norm(rotation_vector);
  const Scalar t_squared = t * t;
  const Scalar half_t = Scalar(0.5) * t;

  // a's closed form is a difference of two O(1) quantities whose gap is O(t^2), so it sheds digits as
  // t -> 0; its series does not. The two agree to roughly 1e-12 at the crossover.
  constexpr Scalar kSeriesCutoff = Scalar(1e-2);
  const Scalar a = (t < kSeriesCutoff) ? (Scalar(1) / Scalar(12) + t_squared / Scalar(720))
                                       : (Scalar(1) - half_t * cos(half_t) / sin(half_t)) / t_squared;

  const Scalar x = rotation_vector[0];
  const Scalar y = rotation_vector[1];
  const Scalar z = rotation_vector[2];

  // [theta]_x^2 is theta theta^T - t^2 I, so the whole matrix is written out directly.
  Matrix3<Scalar> jacobian;
  jacobian(0, 0) = Scalar(1) + a * (x * x - t_squared);
  jacobian(0, 1) = Scalar(0.5) * z + a * x * y;
  jacobian(0, 2) = -Scalar(0.5) * y + a * x * z;
  jacobian(1, 0) = -Scalar(0.5) * z + a * y * x;
  jacobian(1, 1) = Scalar(1) + a * (y * y - t_squared);
  jacobian(1, 2) = Scalar(0.5) * x + a * y * z;
  jacobian(2, 0) = Scalar(0.5) * y + a * z * x;
  jacobian(2, 1) = -Scalar(0.5) * x + a * z * y;
  jacobian(2, 2) = Scalar(1) + a * (z * z - t_squared);
  return jacobian;
}

/// \brief Fixed-pose anchor Jacobian and initial pose error, via rod poses.
///
/// Six rows per anchor, laid out as one generalized [force(3), torque(3)] block. The first three are
/// the anchored material point's world components, exactly as a fixed position holds them. The last
/// three hold the rod's orientation. Their constraint value is the rotation vector of the error
/// rotation carrying the target orientation onto the current one, and their Jacobian is the matrix
/// taking angular velocity to that vector's rate, which is SO(3)'s inverse left Jacobian rather than
/// the identity usually substituted for it. Both halves are therefore exact at any pose error, not
/// only near zero.
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
SingleGeometry<ExecSpace> compute_fixed_pose_geometry(const RodViews<ExecSpace>& rods,
                                                      const FixedPoseViews<ExecSpace>& anchors, const B0View& b0) {
  using memory_space = typename ExecSpace::memory_space;
  const size_t num_anchors = anchors.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == anchors.num_constraints(), std::invalid_argument,
                     "compute_fixed_pose_geometry: b0 must have six entries per anchor.");

  Kokkos::View<int*, memory_space> owner("fixed_pose_owner", anchors.num_constraints());
  SingleGeometry<ExecSpace> geo(owner);

  auto rods_l = rods;
  auto geo_l = geo;
  auto anchors_l = anchors;
  auto owner_l = owner;
  auto b0_l = b0;
  Kokkos::parallel_for(
      "compute_fixed_pose_geometry", Kokkos::RangePolicy<ExecSpace>(0, num_anchors), KOKKOS_LAMBDA(const int a) {
        const int i = anchors_l.rod(a);
        const Vector3d r_world = rods_l.orientation(i) * anchors_l.body_offset(a);
        const Vector3d offset = rods_l.center(i) + r_world - anchors_l.target_point(a);
        const Vector3d rotation_error =
            quaternion_to_rotation_vector(rods_l.orientation(i) * inverse(anchors_l.target_orientation(a)));
        const Matrix3<double> rotation_jacobian = rotation_vector_jacobian(rotation_error);

        for (int c = 0; c < 3; ++c) {
          Vector3d axis{0.0, 0.0, 0.0};
          axis[c] = 1.0;

          const int position_row = 6 * a + c;
          owner_l(position_row) = i;
          geo_l.force(position_row) = axis;
          geo_l.torque(position_row) = cross(r_world, axis);
          b0_l(position_row) = offset[c];

          const int orientation_row = 6 * a + 3 + c;
          owner_l(orientation_row) = i;
          geo_l.force(orientation_row) = Vector3d{0.0, 0.0, 0.0};
          geo_l.torque(orientation_row) =
              Vector3d{rotation_jacobian(c, 0), rotation_jacobian(c, 1), rotation_jacobian(c, 2)};
          b0_l(orientation_row) = rotation_error[c];
        }
      });

  return geo;
}

/// \brief How many rods carry more than one fixed-position or fixed-pose anchor.
///
/// Each anchor removes three or six of one rod's degrees of freedom by constraining the same
/// translational block, so two of them on one rod give B linearly dependent columns and leave
/// dt B^T M B singular wherever the hold is rigid. Two anchors on two *different* rods are fine and
/// are the ordinary case: a beam supported at both ends.
///
/// The Schur complement's conjugate gradient takes its operator's definiteness on trust, so such a
/// set would otherwise surface only as a failure to converge, a long way from its cause.
template <typename ExecSpace>
size_t count_doubly_anchored_rods(const ConstraintSet<ExecSpace>& constraints, size_t num_rods) {
  Kokkos::View<int*, typename ExecSpace::memory_space> anchor_count("anchor_count", num_rods);

  auto count = anchor_count;
  auto fixed_positions = constraints.fixed_positions;
  Kokkos::parallel_for(
      "count_fixed_position_anchors", Kokkos::RangePolicy<ExecSpace>(0, fixed_positions.size()),
      KOKKOS_LAMBDA(const int a) {
        const int rod = fixed_positions.rod(a);
        Kokkos::atomic_add(&count(rod), 1);
      });

  auto fixed_poses = constraints.fixed_poses;
  Kokkos::parallel_for(
      "count_fixed_pose_anchors", Kokkos::RangePolicy<ExecSpace>(0, fixed_poses.size()),
      KOKKOS_LAMBDA(const int a) {
        const int rod = fixed_poses.rod(a);
        Kokkos::atomic_add(&count(rod), 1);
      });

  size_t doubly_anchored = 0;
  Kokkos::parallel_reduce(
      "count_doubly_anchored_rods", Kokkos::RangePolicy<ExecSpace>(0, num_rods),
      KOKKOS_LAMBDA(const int i, size_t& partial) { partial += (count(i) > 1) ? 1u : 0u; }, doubly_anchored);
  return doubly_anchored;
}

//! \name Small vector/geometry utilities used to assemble the combined bilateral (y-)block
//@{

/// \brief A Kokkos view that concat_vectors can allocate and copy into.
///
/// Rank one, and allocatable from a label and an extent -- which additionally rules out views over
/// const data and unmanaged views, neither of which can be the destination of a copy. The rank test
/// has to be explicit: a rank-2 view accepts a single extent too, leaving the rest zero.
template <typename T>
concept ConcatenableVector = Kokkos::is_view<T>::value && (T::rank == 1) && requires(const T& v, size_t n) {
  { T("label", n) } -> std::same_as<T>;
  { v.extent(0) } -> std::convertible_to<size_t>;
};

static_assert(ConcatenableVector<Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "A rank-1 owning view must satisfy ConcatenableVector");
static_assert(!ConcatenableVector<Kokkos::View<double**, Kokkos::DefaultExecutionSpace::memory_space>>,
              "A rank-2 view must not satisfy ConcatenableVector");
static_assert(!ConcatenableVector<double>, "A scalar must not satisfy ConcatenableVector");

/// \brief One vector holding every given vector end to end, in the order passed.
///
/// Joining all of them in a single pass rather than pairwise: one allocation and one copy per input,
/// instead of a fresh allocation per join that re-copies everything joined so far.
template <typename FirstView, typename... OtherViews>
  requires ConcatenableVector<FirstView> && (std::same_as<OtherViews, FirstView> && ...)
FirstView concat_vectors(const FirstView& first, const OtherViews&... others) {
  FirstView out("concat_vectors", (first.extent(0) + ... + others.extent(0)));

  size_t offset = 0;
  auto append = [&out, &offset](const FirstView& v) {
    Kokkos::deep_copy(Kokkos::subview(out, Kokkos::pair<size_t, size_t>(offset, offset + v.extent(0))), v);
    offset += v.extent(0);
  };
  append(first);
  (append(others), ...);

  return out;
}

/// \brief One geometry holding every given single geometry end to end, in the order passed.
///
/// A single geometry is two parallel arrays, so concatenating geometries concatenates each of them;
/// the multiplier ordering of the result is the order the geometries are passed in.
template <typename FirstGeometry, typename... OtherGeometries>
  requires SingleGeometryType<FirstGeometry> && (std::same_as<OtherGeometries, FirstGeometry> && ...)
FirstGeometry concat_single_geometry(const FirstGeometry& first, const OtherGeometries&... others) {
  return FirstGeometry(concat_vectors(first.owner_view(), others.owner_view()...),
                       concat_vectors(first.jacobian_view(), others.jacobian_view()...));
}

/// \brief One geometry holding every given pair geometry end to end, in the order passed.
///
/// A pair geometry is four parallel arrays, so concatenating geometries is just concatenating each
/// of those arrays; the multiplier ordering of the result is the order the geometries are passed in.
template <typename FirstGeometry, typename... OtherGeometries>
  requires PairGeometryType<FirstGeometry> && (std::same_as<OtherGeometries, FirstGeometry> && ...)
FirstGeometry concat_pair_geometry(const FirstGeometry& first, const OtherGeometries&... others) {
  return FirstGeometry(concat_vectors(first.owner_i_view(), others.owner_i_view()...),
                       concat_vectors(first.owner_j_view(), others.owner_j_view()...),
                       concat_vectors(first.jacobian_i_view(), others.jacobian_i_view()...),
                       concat_vectors(first.jacobian_j_view(), others.jacobian_j_view()...));
}

/// \brief The entries of a flat vector lying in a given index range.
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto subrange(const ViewType& v, const IndexRange& range) {
  return Kokkos::subview(v, Kokkos::pair<size_t, size_t>(range.begin, range.end));
}

template <typename ExecSpace>
Kokkos::View<double*, typename ExecSpace::memory_space> reciprocal(
    const Kokkos::View<double*, typename ExecSpace::memory_space>& v) {
  using memory_space = typename ExecSpace::memory_space;
  Kokkos::View<double*, memory_space> out("reciprocal", v.extent(0));
  Kokkos::parallel_for(
      "reciprocal", Kokkos::RangePolicy<ExecSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(const int i) { out(i) = 1.0 / v(i); });
  return out;
}
//@}

}  // namespace impl

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_IMPL_KOKKOSMBODYIMPL_HPP_
