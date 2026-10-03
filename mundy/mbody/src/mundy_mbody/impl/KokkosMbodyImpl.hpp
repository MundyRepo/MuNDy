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
/// \brief Operators and geometry kernels for the multibody step solvers (mbody::solve_mixed_lcp, solve_mixed_slcp).
///
/// The D/D^T and B/B^T Jacobian operators, the local drag mobility operator, and the geometry kernels
/// that fill them.

// C++ core
#include <algorithm>    // for std::max, std::count
#include <array>        // for std::array
#include <concepts>     // for std::same_as
#include <type_traits>  // for std::false_type, std::true_type, std::remove_cvref_t
#include <utility>      // for std::declval, std::index_sequence, std::move

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/distance.hpp>
#include <mundy_geom/primitives.hpp>
#include <mundy_math/Quaternion.hpp>  // for mundy::{quaternion_to_rotation_vector, rotation_vector_jacobian}
#include <mundy_math/Scalar.hpp>
#include <mundy_math/Vector3.hpp>          // for mundy::{cross, perp}
#include <mundy_math/convex_spaces.hpp>    // for mundy::LowerBoundSpace
#include <mundy_math/cqpp.hpp>             // for mundy::{make_mixed_cqpp, solve_mixed_cqpp}
#include <mundy_math/lcp.hpp>              // for mundy::{make_lcp, solve_lcp}
#include <mundy_math/linear_ops.hpp>       // for mundy::{make_concat_domain_op, make_scaled_op, make_sum_op, ...}
#include <mundy_math/linear_system.hpp>    // for mundy::{CGConfig, make_cg_inv_op}
#include <mundy_math/pgd.hpp>              // for mundy::{PGDConfig, PGDResult, make_pgd_solution_strategy}
#include <mundy_math/solver_backends.hpp>  // for mundy::{KokkosBackend, LinearOperator, HasScaledApplyMember}
#include <mundy_mbody/KokkosMbodyTypes.hpp>
#include <mundy_utils/throw_assert.hpp>
#include <mundy_utils/tuple.hpp>        // for mundy::{tuple, make_tuple, tuple_cat, get, tuple_size_v}
#include <mundy_utils/type_traits.hpp>  // for mundy::index_finder_v

namespace mundy {

namespace mbody {

namespace impl {

/// \brief Per-body generalized Jacobian mapping a unit multiplier to one rod's center-of-mass force and torque.
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

/// \brief Maps one multiplier per row to center-of-mass force and torque (B).
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
    if (singles_.size() == 0) {
      return;
    }
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

/// \brief Maps center-of-mass translational and rotational velocity to each row's constraint rate (B^T).
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
    if (singles_.size() == 0) {
      return;
    }

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

/// \brief Per-pair generalized Jacobian mapping a unit Lagrange multiplier to center-of-mass force and torque.
///
/// For a unit scalar multiplier the pair contributes (force_i, torque_i) to owner_i and (force_j,
/// torque_j) to owner_j. One representation covers:
///   - contacts:        force_{i,j} = -+n,  torque_{i,j} = -+r_{i,j} x n
///   - linear springs:  force_{i,j} = -+d,  torque_{i,j} = 0
///   - angular springs: force_{i,j} = 0,    torque_{i,j} = -+a
///   - pins:            force_{i,j} = +-e_c, torque_{i,j} = +-r_{i,j} x e_c
///   - fixed lengths:   force_{i,j} = -+n,  torque_{i,j} = -+r_{i,j} x n
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

/// \brief Maps one multiplier per pair to center-of-mass force and torque (D or B).
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
    if (pairs_.size() == 0) {
      return;
    }
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

/// \brief Maps center-of-mass translational and rotational velocity to each pair's rate (D^T or B^T).
///
/// The exact transpose of PairForceOp. A contact's rate is that of its separation; a bilateral row's, that of its
/// constraint value.
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
    if (pairs_.size() == 0) {
      return;
    }

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
/// owner_3 -- no torque, since this constraint is a function of the three rods' positions alone. owner_3 is the
/// vertex the angle is measured at.
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

  // Wraps already-populated storage.
  TripleGeometry(const int_view_t& owner_1, const int_view_t& owner_2, const int_view_t& owner_3,
                 const vector_view_t& jacobian_1, const vector_view_t& jacobian_2, const vector_view_t& jacobian_3)
      : owner_1_(owner_1),
        owner_2_(owner_2),
        owner_3_(owner_3),
        jacobian_1_(jacobian_1),
        jacobian_2_(jacobian_2),
        jacobian_3_(jacobian_3) {
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

  int_view_t owner_1_view() const {
    return owner_1_;
  }
  int_view_t owner_2_view() const {
    return owner_2_;
  }
  int_view_t owner_3_view() const {
    return owner_3_;
  }
  vector_view_t jacobian_1_view() const {
    return jacobian_1_;
  }
  vector_view_t jacobian_2_view() const {
    return jacobian_2_;
  }
  vector_view_t jacobian_3_view() const {
    return jacobian_3_;
  }

 private:
  int_view_t owner_1_;
  int_view_t owner_2_;
  int_view_t owner_3_;
  vector_view_t jacobian_1_;
  vector_view_t jacobian_2_;
  vector_view_t jacobian_3_;
};

/// \brief Whether T is one of the TripleGeometry specializations.
template <typename T>
struct is_triple_geometry : std::false_type {};

template <typename ExecSpace>
struct is_triple_geometry<TripleGeometry<ExecSpace>> : std::true_type {};

template <typename T>
inline constexpr bool is_triple_geometry_v = is_triple_geometry<T>::value;

/// \brief Matches exactly the TripleGeometry specializations.
template <typename T>
concept TripleGeometryType = is_triple_geometry_v<T>;

/// \brief Maps one multiplier per triple to center-of-mass force (torque zero, see TripleGeometry).
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
    if (triples_.size() == 0) {
      return;
    }
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

/// \brief Maps center-of-mass translational velocity to each triple's constraint rate (transpose of TripleForceOp).
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
    if (triples_.size() == 0) {
      return;
    }

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
PairGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods, const ContactViews<ExecSpace>& contacts,
                                         const Sep0View& sep0) {
  const size_t n = contacts.size();
  MUNDY_THROW_ASSERT(sep0.extent(0) == n, std::invalid_argument,
                     "compute_geometry: sep0 must have one entry per contact.");
  PairGeometry<ExecSpace> geo(contacts.rod_i_view(), contacts.rod_j_view());
  if (n == 0) {
    return geo;
  }

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
PairGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods, const LinearSpringViews<ExecSpace>& springs,
                                         const B0View& b0) {
  const size_t n = springs.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == n, std::invalid_argument, "compute_geometry: b0 must have one entry per spring.");
  PairGeometry<ExecSpace> geo(springs.rod_i_view(), springs.rod_j_view());
  if (n == 0) {
    return geo;
  }

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
PairGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods, const AngularSpringViews<ExecSpace>& springs,
                                         const B0View& b0) {
  const size_t n = springs.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == n, std::invalid_argument, "compute_geometry: b0 must have one entry per spring.");
  PairGeometry<ExecSpace> geo(springs.rod_i_view(), springs.rod_j_view());
  if (n == 0) {
    return geo;
  }

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

/// \brief Pin Jacobian and initial offset, via rod poses.
///
/// Row c of a pin constrains the world component c of p_i - p_j, where p = center + R(q) body_offset moves at
/// v + omega x r_world. Its Jacobian is force_i = e_c, torque_i = r_i x e_c, force_j = -e_c,
/// torque_j = -r_j x e_c -- a fixed position's lever-arm form on each body -- and its constraint value is
/// (p_i - p_j)[c].
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
PairGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods, const PinViews<ExecSpace>& pins,
                                         const B0View& b0) {
  using memory_space = typename ExecSpace::memory_space;
  const size_t num_pins = pins.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == pins.num_rows(), std::invalid_argument,
                     "compute_geometry: b0 must have three entries per pin.");

  Kokkos::View<int*, memory_space> owner_i("pin_owner_i", pins.num_rows());
  Kokkos::View<int*, memory_space> owner_j("pin_owner_j", pins.num_rows());
  PairGeometry<ExecSpace> geo(owner_i, owner_j);
  if (num_pins == 0) {
    return geo;
  }

  auto rods_l = rods;
  auto geo_l = geo;
  auto pins_l = pins;
  auto owner_i_l = owner_i;
  auto owner_j_l = owner_j;
  auto b0_l = b0;
  int num_self_pins = 0;
  Kokkos::parallel_reduce(
      "compute_pin_geometry", Kokkos::RangePolicy<ExecSpace>(0, num_pins),
      KOKKOS_LAMBDA(const int a, int& self_pins) {
        const int i = pins_l.rod_i(a);
        const int j = pins_l.rod_j(a);
        self_pins += (i == j) ? 1 : 0;

        const Vector3d r_i = rods_l.orientation(i) * pins_l.body_offset_i(a);
        const Vector3d r_j = rods_l.orientation(j) * pins_l.body_offset_j(a);
        const Vector3d offset = (rods_l.center(i) + r_i) - (rods_l.center(j) + r_j);

        for (int c = 0; c < 3; ++c) {
          const int row = 3 * a + c;
          Vector3d axis{0.0, 0.0, 0.0};
          axis[c] = 1.0;

          owner_i_l(row) = i;
          owner_j_l(row) = j;
          geo_l.force_i(row) = axis;
          geo_l.torque_i(row) = cross(r_i, axis);
          geo_l.force_j(row) = -axis;
          geo_l.torque_j(row) = -cross(r_j, axis);
          b0_l(row) = offset[c];
        }
      },
      num_self_pins);
  MUNDY_THROW_REQUIRE(num_self_pins == 0, std::invalid_argument, "compute_geometry: a pin joins a rod to itself.");

  return geo;
}

/// \brief Fixed-length Jacobian and initial stretch, via rod poses.
///
/// With n the unit vector from p_i to p_j, the distance |p_j - p_i| changes at
/// n . (v_j + omega_j x r_j - v_i - omega_i x r_i), so force_i = -n, torque_i = -r_i x n, force_j = n,
/// torque_j = r_j x n -- a linear spring's direction on a contact's lever arms -- and the constraint value is
/// |p_j - p_i| - rest_length.
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
PairGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods, const FixedLengthViews<ExecSpace>& lengths,
                                         const B0View& b0) {
  const size_t n = lengths.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == n, std::invalid_argument,
                     "compute_geometry: b0 must have one entry per fixed length.");
  PairGeometry<ExecSpace> geo(lengths.rod_i_view(), lengths.rod_j_view());
  if (n == 0) {
    return geo;
  }

  constexpr int self_join = 1;
  constexpr int nonpositive_rest_length = 2;
  constexpr int coincident_points = 4;
  auto rods_l = rods;
  auto geo_l = geo;
  auto lengths_l = lengths;
  auto b0_l = b0;
  int defects = 0;
  Kokkos::parallel_reduce(
      "compute_fixed_length_geometry", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int k, int& defect) {
        const int i = geo_l.owner_i(k);
        const int j = geo_l.owner_j(k);
        const Vector3d r_i = rods_l.orientation(i) * lengths_l.body_offset_i(k);
        const Vector3d r_j = rods_l.orientation(j) * lengths_l.body_offset_j(k);
        const Vector3d sep = (rods_l.center(j) + r_j) - (rods_l.center(i) + r_i);
        const double dist = norm(sep);
        const double rest_length = lengths_l.rest_length(k);
        defect |= (i == j) ? self_join : 0;
        defect |= (rest_length <= 0.0) ? nonpositive_rest_length : 0;
        defect |= (dist <= 1e-12) ? coincident_points : 0;
        const Vector3d dir = sep / dist;

        geo_l.force_i(k) = -dir;
        geo_l.torque_i(k) = -cross(r_i, dir);
        geo_l.force_j(k) = dir;
        geo_l.torque_j(k) = cross(r_j, dir);
        b0_l(k) = dist - rest_length;
      },
      Kokkos::BOr<int>(defects));
  MUNDY_THROW_REQUIRE((defects & self_join) == 0, std::invalid_argument,
                      "compute_geometry: a fixed length joins a rod to itself.");
  MUNDY_THROW_REQUIRE((defects & nonpositive_rest_length) == 0, std::invalid_argument,
                      "compute_geometry: a rest length is not positive.");
  MUNDY_THROW_REQUIRE((defects & coincident_points) == 0, std::runtime_error,
                      "compute_geometry: the endpoints of a fixed length are nearly coincident.");

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
/// by mundy::perp, exactly as for an angular spring. Position-only: no
/// orientation, no torque, so rod_i/rod_j/rod_k need not be distinct (only rod_i/rod_k and rod_j/rod_k
/// must not coincide in position, checked below).
template <typename ExecSpace, typename B0View>
  requires ConstraintValueView<B0View>
TripleGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods,
                                           const TriplePointAngularSpringViews<ExecSpace>& springs, const B0View& b0) {
  const size_t n = springs.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == n, std::invalid_argument, "compute_geometry: b0 must have one entry per spring.");
  TripleGeometry<ExecSpace> geo(springs.rod_i_view(), springs.rod_j_view(), springs.rod_k_view());
  if (n == 0) {
    return geo;
  }

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
        const Vector3d n_hat = (cross_norm > 1e-9 * d1_d2) ? cross_v1v2 / cross_norm : perp(v1);
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
SingleGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods,
                                           const FixedPositionViews<ExecSpace>& anchors, const B0View& b0) {
  using memory_space = typename ExecSpace::memory_space;
  const size_t num_anchors = anchors.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == anchors.num_rows(), std::invalid_argument,
                     "compute_geometry: b0 must have three entries per anchor.");

  Kokkos::View<int*, memory_space> owner("fixed_position_owner", anchors.num_rows());
  SingleGeometry<ExecSpace> geo(owner);
  if (num_anchors == 0) {
    return geo;
  }

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
SingleGeometry<ExecSpace> compute_geometry(const RodViews<ExecSpace>& rods, const FixedPoseViews<ExecSpace>& anchors,
                                           const B0View& b0) {
  using memory_space = typename ExecSpace::memory_space;
  const size_t num_anchors = anchors.size();
  MUNDY_THROW_ASSERT(b0.extent(0) == anchors.num_rows(), std::invalid_argument,
                     "compute_geometry: b0 must have six entries per anchor.");

  Kokkos::View<int*, memory_space> owner("fixed_pose_owner", anchors.num_rows());
  SingleGeometry<ExecSpace> geo(owner);
  if (num_anchors == 0) {
    return geo;
  }

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

//! \name Flat-vector and geometry utilities
//@{

/// \brief The half-open range [begin, end) that one constraint family occupies in a flat array.
struct IndexRange {
  size_t begin = 0;
  size_t end = 0;

  KOKKOS_INLINE_FUNCTION size_t size() const {
    return end - begin;
  }
};

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
    if (v.extent(0) == 0) {
      return;
    }
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

/// \brief One geometry holding every given triple geometry end to end, in the order passed.
template <typename FirstGeometry, typename... OtherGeometries>
  requires TripleGeometryType<FirstGeometry> && (std::same_as<OtherGeometries, FirstGeometry> && ...)
FirstGeometry concat_triple_geometry(const FirstGeometry& first, const OtherGeometries&... others) {
  return FirstGeometry(concat_vectors(first.owner_1_view(), others.owner_1_view()...),
                       concat_vectors(first.owner_2_view(), others.owner_2_view()...),
                       concat_vectors(first.owner_3_view(), others.owner_3_view()...),
                       concat_vectors(first.jacobian_1_view(), others.jacobian_1_view()...),
                       concat_vectors(first.jacobian_2_view(), others.jacobian_2_view()...),
                       concat_vectors(first.jacobian_3_view(), others.jacobian_3_view()...));
}

/// \brief One geometry holding every given geometry of one arity end to end; a lone geometry is returned as is.
template <typename FirstGeometry, typename... OtherGeometries>
FirstGeometry concat_geometry(const FirstGeometry& first, const OtherGeometries&... others) {
  if constexpr (sizeof...(OtherGeometries) == 0) {
    return first;
  } else if constexpr (SingleGeometryType<FirstGeometry>) {
    return concat_single_geometry(first, others...);
  } else if constexpr (PairGeometryType<FirstGeometry>) {
    return concat_pair_geometry(first, others...);
  } else {
    return concat_triple_geometry(first, others...);
  }
}

/// \brief The entries of a flat vector lying in a given index range.
template <typename ViewType>
KOKKOS_INLINE_FUNCTION auto subrange(const ViewType& v, const IndexRange& range) {
  return Kokkos::subview(v, Kokkos::pair<size_t, size_t>(range.begin, range.end));
}
//@}

//! \name Constraint packing
//@{

/// \brief Where each family's rows land in the flat unilateral and bilateral multiplier vectors.
///
/// The families hold differing numbers of rows and are stored separately, but the solve works on one contiguous
/// vector per block. Packing them is the standard compressed-row treatment of ragged data: an exclusive prefix scan
/// over the per-family row counts gives each family its start offset, and a family's own local row added to that
/// offset is its flat position. Within a block, families pack by arity, pairs, then triples, then single-body rows,
/// each in the set's order, so each arity's Jacobians concatenate into one operator.
///
/// The constraint values, the compliance diagonal, the Jacobian operators and the multiplier write-back all index
/// these vectors. They agree by construction only if they share one mapping, so the packing order is decided in
/// exactly one place and the result is passed, never re-derived.
///
/// A trivially copyable aggregate holding only sizes, so it captures by value into a kernel.
template <typename... Families>
struct ConstraintIndexMap {
  Kokkos::Array<IndexRange, sizeof...(Families)> ranges{};
  size_t num_unilateral = 0;
  size_t num_bilateral = 0;

  /// \brief The range of family F's rows in its block's vector.
  template <typename F>
  KOKKOS_INLINE_FUNCTION IndexRange range() const {
    return ranges[::mundy::index_finder_v<F, Families...>];
  }
};

/// \brief The arity order in which each block packs its families.
inline constexpr size_t arity_pack_order[] = {2, 3, 1};

/// \brief Give family the next range of its block's vector, if it has the given arity.
template <typename F, typename... Families>
void place_family(ConstraintIndexMap<Families...>& index_map, const F& family, size_t arity, size_t& unilateral_end,
                  size_t& bilateral_end) {
  if (F::bodies_per_entry != arity) {
    return;
  }
  size_t& end = F::constraint_type == ConstraintType::UNILATERAL ? unilateral_end : bilateral_end;
  index_map.ranges[::mundy::index_finder_v<F, Families...>] = IndexRange{end, end + family.num_rows()};
  end += family.num_rows();
}

/// \brief Pack a constraint set's families into its unilateral and bilateral vectors.
///
/// Build this once per set and pass it along; it is derived from the family sizes, so a set whose
/// families are resized needs a fresh one.
template <typename... Families>
ConstraintIndexMap<Families...> make_constraint_index_map(const ConstraintSet<Families...>& constraints) {
  ConstraintIndexMap<Families...> index_map;
  size_t unilateral_end = 0;
  size_t bilateral_end = 0;
  for ([[maybe_unused]] const size_t arity : arity_pack_order) {
    (place_family(index_map, get<Families>(constraints), arity, unilateral_end, bilateral_end), ...);
  }
  index_map.num_unilateral = unilateral_end;
  index_map.num_bilateral = bilateral_end;
  return index_map;
}
//@}

//! \name The unilateral and bilateral blocks at one configuration
//@{

/// \brief A block's Jacobian at one configuration: one concatenated geometry per arity present, in packing order.
template <typename... Geometries>
struct BlockGeometry {
  ::mundy::tuple<Geometries...> groups;
};

/// \brief The positions within Families of a block's families of one arity, in the set's order.
template <ConstraintType Block, size_t Arity, typename... Families>
struct BlockGroup {
  static constexpr std::array<bool, sizeof...(Families)> is_member{
      {(Families::constraint_type == Block && Families::bodies_per_entry == Arity)...}};
  static constexpr size_t size = static_cast<size_t>(std::count(is_member.begin(), is_member.end(), true));
  static constexpr std::array<size_t, size> positions = [] {
    std::array<size_t, size> out{};
    size_t n = 0;
    for (size_t i = 0; i < is_member.size(); ++i) {
      if (is_member[i]) {
        out[n++] = i;
      }
    }
    return out;
  }();
};

/// \brief Stands in for a family's geometry in the other block.
struct OutsideBlock {};

/// \brief Family F's Jacobian and constraint values in its block's vector, or nothing if F is in the other block.
template <ConstraintType Block, typename F, typename ExecSpace, typename ValueView, typename... Families>
auto compute_block_family_geometry(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                                   const ConstraintIndexMap<Families...>& index_map, const ValueView& values) {
  if constexpr (F::constraint_type == Block) {
    return compute_geometry(rods, get<F>(constraints), subrange(values, index_map.template range<F>()));
  } else {
    return OutsideBlock{};
  }
}

/// \brief A group's family geometries concatenated, or no geometry if the group is empty.
template <typename Group, typename FamilyGeometries>
auto concat_block_group(const FamilyGeometries& family_geometries) {
  if constexpr (Group::size == 0) {
    return ::mundy::tuple<>{};
  } else {
    return [&]<size_t... J>(std::index_sequence<J...>) {
      return ::mundy::make_tuple(concat_geometry(::mundy::get<Group::positions[J]>(family_geometries)...));
    }(std::make_index_sequence<Group::size>{});
  }
}

/// \brief A block's Jacobian and constraint values at rods' configuration; an empty block is one empty pair group.
template <ConstraintType Block, typename ExecSpace, typename ValueView, typename... Families>
  requires ConstraintValueView<ValueView>
auto compute_block_geometry(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                            const ConstraintIndexMap<Families...>& index_map, const ValueView& values) {
  MUNDY_THROW_ASSERT(
      values.extent(0) == (Block == ConstraintType::UNILATERAL ? index_map.num_unilateral : index_map.num_bilateral),
      std::invalid_argument, "compute_block_geometry: values must have one entry per block row.");
  const auto family_geometries =
      ::mundy::make_tuple(compute_block_family_geometry<Block, Families>(rods, constraints, index_map, values)...);
  const auto groups = ::mundy::tuple_cat(
      ::mundy::tuple_cat(concat_block_group<BlockGroup<Block, arity_pack_order[0], Families...>>(family_geometries),
                         concat_block_group<BlockGroup<Block, arity_pack_order[1], Families...>>(family_geometries)),
      concat_block_group<BlockGroup<Block, arity_pack_order[2], Families...>>(family_geometries));
  if constexpr (::mundy::tuple_size_v<std::remove_cvref_t<decltype(groups)>> == 0) {
    return BlockGeometry<PairGeometry<ExecSpace>>{::mundy::make_tuple(PairGeometry<ExecSpace>{})};
  } else {
    return [&]<typename... Geometries>(const ::mundy::tuple<Geometries...>& g) {
      return BlockGeometry<Geometries...>{g};
    }(groups);
  }
}

/// \brief The unilateral families' Jacobian at rods' configuration; phi receives their constraint values.
template <typename ExecSpace, typename PhiView, typename... Families>
auto compute_unilateral_geometry(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                                 const ConstraintIndexMap<Families...>& index_map, const PhiView& phi) {
  return compute_block_geometry<ConstraintType::UNILATERAL>(rods, constraints, index_map, phi);
}

/// \brief The bilateral families' Jacobian at rods' configuration; psi receives their constraint values.
template <typename ExecSpace, typename PsiView, typename... Families>
auto compute_bilateral_geometry(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                                const ConstraintIndexMap<Families...>& index_map, const PsiView& psi) {
  return compute_block_geometry<ConstraintType::BILATERAL>(rods, constraints, index_map, psi);
}

// The map from a geometry's multipliers to center-of-mass force and torque, per arity.
template <typename ExecSpace>
SingleForceOp<ExecSpace> make_force_op(const SingleGeometry<ExecSpace>& geo, size_t num_rods) {
  return SingleForceOp<ExecSpace>(geo, num_rods);
}
template <typename ExecSpace>
PairForceOp<ExecSpace> make_force_op(const PairGeometry<ExecSpace>& geo, size_t num_rods) {
  return PairForceOp<ExecSpace>(geo, num_rods);
}
template <typename ExecSpace>
TripleForceOp<ExecSpace> make_force_op(const TripleGeometry<ExecSpace>& geo, size_t num_rods) {
  return TripleForceOp<ExecSpace>(geo, num_rods);
}

// The map from center-of-mass translational and rotational velocity to a geometry's constraint rates, per arity.
template <typename ExecSpace>
SingleForceOpT<ExecSpace> make_rate_op(const SingleGeometry<ExecSpace>& geo, size_t num_rods) {
  return SingleForceOpT<ExecSpace>(geo, num_rods);
}
template <typename ExecSpace>
PairForceOpT<ExecSpace> make_rate_op(const PairGeometry<ExecSpace>& geo, size_t num_rods) {
  return PairForceOpT<ExecSpace>(geo, num_rods);
}
template <typename ExecSpace>
TripleForceOpT<ExecSpace> make_rate_op(const TripleGeometry<ExecSpace>& geo, size_t num_rods) {
  return TripleForceOpT<ExecSpace>(geo, num_rods);
}

/// \brief The operators side by side in the domain, [A B C] = [[A B] C]; a lone operator is returned as is.
template <typename Backend, typename Op>
Op concat_domain_ops(Op op) {
  return op;
}
template <typename Backend, typename Op1, typename Op2, typename... Ops>
auto concat_domain_ops(Op1 op1, Op2 op2, Ops... ops) {
  return concat_domain_ops<Backend>(make_concat_domain_op<Backend>(std::move(op1), std::move(op2)), std::move(ops)...);
}

/// \brief The operators stacked in the range, [A; B; C] = [[A; B]; C]; a lone operator is returned as is.
template <typename Backend, typename Op>
Op concat_range_ops(Op op) {
  return op;
}
template <typename Backend, typename Op1, typename Op2, typename... Ops>
auto concat_range_ops(Op1 op1, Op2 op2, Ops... ops) {
  return concat_range_ops<Backend>(make_concat_range_op<Backend>(std::move(op1), std::move(op2)), std::move(ops)...);
}

/// \brief A block's map from multipliers to center-of-mass force and torque (D or B).
template <typename ExecSpace, typename... Geometries>
auto make_block_force_op(const BlockGeometry<Geometries...>& block, size_t num_rods) {
  return [&]<size_t... I>(std::index_sequence<I...>) {
    return concat_domain_ops<KokkosBackend<ExecSpace>>(make_force_op(::mundy::get<I>(block.groups), num_rods)...);
  }(std::index_sequence_for<Geometries...>{});
}

/// \brief A block's map from center-of-mass translational and rotational velocity to constraint rates (D^T or B^T).
template <typename ExecSpace, typename... Geometries>
auto make_block_rate_op(const BlockGeometry<Geometries...>& block, size_t num_rods) {
  return [&]<size_t... I>(std::index_sequence<I...>) {
    return concat_range_ops<KokkosBackend<ExecSpace>>(make_rate_op(::mundy::get<I>(block.groups), num_rods)...);
  }(std::index_sequence_for<Geometries...>{});
}

/// \brief kinv(row) := the compliance of family's row.
template <typename ExecSpace, typename F, typename KinvView>
void fill_compliance(const F& family, const KinvView& kinv) {
  if (kinv.extent(0) == 0) {
    return;
  }
  Kokkos::parallel_for(
      "fill_compliance", Kokkos::RangePolicy<ExecSpace>(0, kinv.extent(0)),
      KOKKOS_LAMBDA(const int row) { kinv(row) = family.row_compliance(row); });
}

/// \brief Family F's compliances into its rows of kinv, if F is bilateral.
template <typename ExecSpace, typename F, typename KinvView, typename... Families>
void fill_family_compliance(const F& family, const KinvView& kinv, const ConstraintIndexMap<Families...>& index_map) {
  if constexpr (F::constraint_type == ConstraintType::BILATERAL) {
    fill_compliance<ExecSpace>(family, subrange(kinv, index_map.template range<F>()));
  }
}

/// \brief The bilateral compliance diagonal K^-1 in the index map's order.
template <typename ExecSpace, typename... Families>
Kokkos::View<double*, typename ExecSpace::memory_space> compliance_diagonal(
    const ConstraintSet<Families...>& constraints, const ConstraintIndexMap<Families...>& index_map) {
  Kokkos::View<double*, typename ExecSpace::memory_space> kinv("kinv", index_map.num_bilateral);
  (fill_family_compliance<ExecSpace>(get<Families>(constraints), kinv, index_map), ...);
  return kinv;
}

/// \brief units(row) := the unit of family F's row, for F's rows.
template <typename ExecSpace, typename F, typename UnitView>
void fill_row_units(const UnitView& units) {
  if (units.extent(0) == 0) {
    return;
  }
  Kokkos::parallel_for(
      "fill_row_units", Kokkos::RangePolicy<ExecSpace>(0, units.extent(0)),
      KOKKOS_LAMBDA(const int row) { units(row) = F::row_unit(row % F::rows_per_entry); });
}

/// \brief Family F's row units into its rows of units, if F is bilateral.
template <typename ExecSpace, typename F, typename UnitView, typename... Families>
void fill_family_row_units(const UnitView& units, const ConstraintIndexMap<Families...>& index_map) {
  if constexpr (F::constraint_type == ConstraintType::BILATERAL) {
    fill_row_units<ExecSpace, F>(subrange(units, index_map.template range<F>()));
  }
}

/// \brief The unit of every bilateral row, in the index map's order.
template <typename ExecSpace, typename... Families>
Kokkos::View<RowUnit*, typename ExecSpace::memory_space> make_row_units(
    const ConstraintIndexMap<Families...>& index_map) {
  Kokkos::View<RowUnit*, typename ExecSpace::memory_space> units("row_units", index_map.num_bilateral);
  (fill_family_row_units<ExecSpace, Families>(units, index_map), ...);
  return units;
}

/// \brief The largest magnitudes of a quantity, split into its length and angle parts.
struct LengthAngleMax {
  double length = 0.0;
  double angle = 0.0;
};

/// \brief The largest |psi + K^-1 y| over the bilateral rows measured in length and over those measured in angle.
template <typename ExecSpace, typename PsiView, typename KinvView, typename YView, typename UnitView>
LengthAngleMax max_bilateral_residual(const PsiView& psi, const KinvView& kinv, const YView& y,
                                      const UnitView& row_units) {
  const size_t n = row_units.extent(0);
  if (n == 0) {
    return LengthAngleMax{};
  }
  double length_max = 0.0;
  double angle_max = 0.0;
  Kokkos::parallel_reduce(
      "max_bilateral_residual_length", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int row, double& m) {
        if (row_units(row) == RowUnit::LENGTH) {
          m = Kokkos::max(m, Kokkos::abs(psi(row) + kinv(row) * y(row)));
        }
      },
      Kokkos::Max<double>(length_max));
  Kokkos::parallel_reduce(
      "max_bilateral_residual_angle", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int row, double& m) {
        if (row_units(row) == RowUnit::ANGLE) {
          m = Kokkos::max(m, Kokkos::abs(psi(row) + kinv(row) * y(row)));
        }
      },
      Kokkos::Max<double>(angle_max));
  return LengthAngleMax{Kokkos::max(length_max, 0.0), Kokkos::max(angle_max, 0.0)};
}

/// \brief The largest per-rod translation and rotation in a generalized displacement, split by row unit.
template <typename ExecSpace, typename DisplacementView>
LengthAngleMax max_displacement(const DisplacementView& displacement, size_t num_rods) {
  using rods_t = RodViews<ExecSpace>;
  if (num_rods == 0) {
    return LengthAngleMax{};
  }
  double translation_max = 0.0;
  double rotation_max = 0.0;
  Kokkos::parallel_reduce(
      "max_displacement_translation", Kokkos::RangePolicy<ExecSpace>(0, num_rods),
      KOKKOS_LAMBDA(const int i, double& m) {
        for (size_t c = 0; c < rods_t::rows_per_entry; ++c) {
          if (rods_t::row_unit(c) == RowUnit::LENGTH) {
            m = Kokkos::max(m, Kokkos::abs(displacement(rods_t::rows_per_entry * i + c)));
          }
        }
      },
      Kokkos::Max<double>(translation_max));
  Kokkos::parallel_reduce(
      "max_displacement_rotation", Kokkos::RangePolicy<ExecSpace>(0, num_rods),
      KOKKOS_LAMBDA(const int i, double& m) {
        for (size_t c = 0; c < rods_t::rows_per_entry; ++c) {
          if (rods_t::row_unit(c) == RowUnit::ANGLE) {
            m = Kokkos::max(m, Kokkos::abs(displacement(rods_t::rows_per_entry * i + c)));
          }
        }
      },
      Kokkos::Max<double>(rotation_max));
  return LengthAngleMax{Kokkos::max(translation_max, 0.0), Kokkos::max(rotation_max, 0.0)};
}
//@}

//! \name One step
//@{

/// \brief The parts of a step that no linearization changes.
///
/// The unilateral block is linearized at the start of the step, C^k: its Jacobian and q = Phi(C^k) + dt D^T U_free.
/// The mobility is that of C^k, and U_free = V_ext + M F_ext is the velocity without constraint forces.
template <typename ExecSpace, typename... Families>
struct StepData {
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;
  using unilateral_geometry_t = decltype(compute_unilateral_geometry(
      std::declval<const RodViews<ExecSpace>&>(), std::declval<const ConstraintSet<Families...>&>(),
      std::declval<const ConstraintIndexMap<Families...>&>(), std::declval<const view_t&>()));

  ConstraintIndexMap<Families...> index_map;
  size_t num_rods;
  double dt;
  unilateral_geometry_t unilateral_geo;
  view_t q;
  view_t kinv;
  LocalDragMobilityOp<ExecSpace> mobility;
  view_t u_free;
};

/// \brief The step data of rods and constraints at their current configuration.
template <typename ExecSpace, typename... Families>
StepData<ExecSpace, Families...> make_step_data(const RodViews<ExecSpace>& rods,
                                                const ConstraintSet<Families...>& constraints, double dt,
                                                double viscosity) {
  using view_t = typename StepData<ExecSpace, Families...>::view_t;
  using backend_t = KokkosBackend<ExecSpace>;

  const size_t num_rods = rods.size();
  const ConstraintIndexMap<Families...> index_map = make_constraint_index_map(constraints);
  const size_t num_unilateral = index_map.num_unilateral;

  view_t q("q", num_unilateral);
  const auto unilateral_geo = compute_unilateral_geometry(rods, constraints, index_map, q);
  const LocalDragMobilityOp<ExecSpace> mobility(viscosity, rods);

  view_t u_free("u_free", rods.num_rows());
  Kokkos::deep_copy(u_free, rods.velocity_omega_view());
  view_t m_force_torque_ext("m_force_torque_ext", rods.num_rows());
  mobility.apply(rods.force_torque_view(), m_force_torque_ext);
  backend_t::axpby(1.0, m_force_torque_ext, 1.0, u_free);

  if (num_unilateral > 0) {
    view_t q_rate("q_rate", num_unilateral);
    make_block_rate_op<ExecSpace>(unilateral_geo, num_rods).apply(u_free, q_rate);
    backend_t::axpby(dt, q_rate, 1.0, q);
  }

  return StepData<ExecSpace, Families...>{
      index_map, num_rods, dt, unilateral_geo, q, compliance_diagonal<ExecSpace>(constraints, index_map),
      mobility,  u_free};
}

/// \brief One linearization's solution and the wrench and velocity it produces.
///
/// x and y are the unilateral and bilateral multipliers, bilateral_wrench is B y, wrench is W = D x + B y,
/// m_wrench is M W, and velocity is U_free + M W.
template <typename ExecSpace>
struct LinearizedStep {
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;

  view_t x;
  view_t y;
  view_t bilateral_wrench;
  view_t wrench;
  view_t m_wrench;
  view_t velocity;
  PGDResult<double> result;
};

/// \brief The displacement duration * velocity of every rod's center of mass and orientation.
template <typename ExecSpace>
struct Displacement {
  Kokkos::View<double*, typename ExecSpace::memory_space> velocity;
  double duration;
};

/// \brief Solve the mixed LCP with the step's contact block and the bilateral block linearized at geo.
///
/// to_free_end is the displacement Delta from geo's configuration to the step's constraint-free end configuration, so
/// the bilateral linear term b = psi + B^T Delta is the linear model about geo's configuration of psi at that end. The
/// contact solve starts from x_start.
template <typename ExecSpace, typename BilateralGeometry, typename... Families>
LinearizedStep<ExecSpace> solve_linearization(const StepData<ExecSpace, Families...>& step,
                                              const BilateralGeometry& geo,
                                              const Kokkos::View<double*, typename ExecSpace::memory_space>& psi,
                                              const Displacement<ExecSpace>& to_free_end,
                                              const Kokkos::View<double*, typename ExecSpace::memory_space>& x_start,
                                              const PGDConfig<double>& pgd_cfg, const CGConfig<double>& cg_cfg) {
  using view_t = typename LinearizedStep<ExecSpace>::view_t;
  using backend_t = KokkosBackend<ExecSpace>;

  const size_t num_rods = step.num_rods;
  const size_t num_rows = RodViews<ExecSpace>::rows_per_entry * num_rods;
  const size_t num_unilateral = step.index_map.num_unilateral;
  const size_t num_bilateral = step.index_map.num_bilateral;
  const bool has_unilateral = num_unilateral > 0;
  const bool has_bilateral = num_bilateral > 0;

  const auto D = make_block_force_op<ExecSpace>(step.unilateral_geo, num_rods);
  const auto DT = make_block_rate_op<ExecSpace>(step.unilateral_geo, num_rods);
  const auto B = make_block_force_op<ExecSpace>(geo, num_rods);
  const auto BT = make_block_rate_op<ExecSpace>(geo, num_rods);
  // The mixed CQPP's "M" is dt * mobility: it maps a constraint force to the displacement it causes over the step,
  // not to a velocity. The velocity this step returns uses raw M.
  const auto M_dt = make_scaled_op<backend_t>(step.dt, step.mobility);

  view_t b("b", num_bilateral);
  Kokkos::deep_copy(b, psi);
  if (has_bilateral) {
    view_t b_rate("b_rate", num_bilateral);
    BT.apply(to_free_end.velocity, b_rate);
    backend_t::axpby(to_free_end.duration, b_rate, 1.0, b);
  }

  // Schur complement S := (B^T M B + K^{-1})^{-1}, realized via matrix-free CG.
  const auto btmb_plus_kinv =
      make_sum_op<backend_t>(make_quadratic_form<backend_t>(BT, M_dt, B), make_diagonal_op<backend_t>(step.kinv));
  const auto S = make_cg_inv_op<backend_t>(btmb_plus_kinv, cg_cfg);

  // x* (unilateral multipliers): the reduced CQPP, which without bilateral rows is the LCP. Without unilateral rows
  // x* is empty, and its projected residual is zero before any iteration.
  LinearizedStep<ExecSpace> out;
  out.x = view_t("x", num_unilateral);
  Kokkos::deep_copy(out.x, x_start);
  out.result = PGDResult<double>{0, 0.0, 0.0 <= pgd_cfg.tol};
  if (has_unilateral) {
    view_t grad("grad", num_unilateral);
    view_t x_tmp("x_tmp", num_unilateral);
    view_t grad_tmp("grad_tmp", num_unilateral);

    auto pgd = make_pgd_solution_strategy(pgd_cfg);
    auto pgd_state = make_pgd_state(out.x, grad, x_tmp, grad_tmp);
    if (has_bilateral) {
      const auto mcqpp =
          make_mixed_cqpp<backend_t>(DT, M_dt, D, step.q, B, S, BT, b, LowerBoundSpace<double>{.lower_bound = 0.0});
      out.result = solve_mixed_cqpp(mcqpp, pgd, pgd_state);
    } else {
      out.result = solve_lcp(make_lcp<backend_t>(DT, M_dt, D, step.q), pgd, pgd_state);
    }
  }
  MUNDY_THROW_REQUIRE(out.result.converged, std::runtime_error, "mbody: outer PGD solve failed to converge.");

  // y* = -S (b + B^T M D x*), and the constraint force/torque D x* + B y*.
  view_t Dx("Dx", num_rows);
  if (has_unilateral) {
    D.apply(out.x, Dx);
  }
  out.y = view_t("y", num_bilateral);
  out.bilateral_wrench = view_t("bilateral_wrench", num_rows);
  if (has_bilateral) {
    view_t y_rhs("y_rhs", num_bilateral);
    if (has_unilateral) {
      view_t MDx("MDx", num_rows);
      M_dt.apply(Dx, MDx);
      BT.apply(MDx, y_rhs);
      backend_t::axpby(1.0, b, 1.0, y_rhs);  // y_rhs = b + B^T M D x*
    } else {
      Kokkos::deep_copy(y_rhs, b);
    }
    S.apply(y_rhs, out.y);
    backend_t::axpby(-1.0, out.y, 0.0, out.y);  // y := -y
    B.apply(out.y, out.bilateral_wrench);
  }
  out.wrench = view_t("wrench", num_rows);
  Kokkos::deep_copy(out.wrench, out.bilateral_wrench);
  if (has_unilateral) {
    backend_t::axpby(1.0, Dx, 1.0, out.wrench);
  }

  out.m_wrench = view_t("m_wrench", num_rows);
  step.mobility.apply(out.wrench, out.m_wrench);
  out.velocity = view_t("velocity", num_rows);
  Kokkos::deep_copy(out.velocity, step.u_free);
  backend_t::axpby(1.0, out.m_wrench, 1.0, out.velocity);
  return out;
}

/// \brief Apply a linearization to rods and constraints.
///
/// force/torque gains W, velocity/omega becomes U_free + M W, and every family receives its multipliers.
template <typename ExecSpace, typename... Families>
void write_step(const RodViews<ExecSpace>& rods, const ConstraintSet<Families...>& constraints,
                const ConstraintIndexMap<Families...>& index_map, const LinearizedStep<ExecSpace>& step) {
  auto force_torque = rods.force_torque_view();
  KokkosBackend<ExecSpace>::axpby(1.0, step.wrench, 1.0, force_torque);
  Kokkos::deep_copy(rods.velocity_omega_view(), step.velocity);

  (Kokkos::deep_copy(get<Families>(constraints).lambda_view(),
                     subrange(Families::constraint_type == ConstraintType::UNILATERAL ? step.x : step.y,
                              index_map.template range<Families>())),
   ...);
}

/// \brief An iterate's largest acceptance residual over its tolerance, at the configuration it moves the rods to.
///
/// The residuals are |psi + K^-1 y| there, each in its row's unit, and M (B' y - B y), the displacement by which the
/// bilateral force directions there would change the step, where B and B' map multipliers to center-of-mass force and
/// torque at the iterate's linearization point and at that configuration. B y is the iterate's bilateral wrench.
template <typename ExecSpace, typename BilateralGeometry, typename... Families>
double slcp_merit(const StepData<ExecSpace, Families...>& step, const LinearizedStep<ExecSpace>& iterate,
                  const BilateralGeometry& trial_geo,
                  const Kokkos::View<double*, typename ExecSpace::memory_space>& trial_psi,
                  const Kokkos::View<RowUnit*, typename ExecSpace::memory_space>& row_units, double length_tol,
                  double angle_tol) {
  using view_t = typename StepData<ExecSpace, Families...>::view_t;
  using backend_t = KokkosBackend<ExecSpace>;
  const size_t num_rods = step.num_rods;
  const size_t num_rows = RodViews<ExecSpace>::rows_per_entry * num_rods;

  const LengthAngleMax rows = max_bilateral_residual<ExecSpace>(trial_psi, step.kinv, iterate.y, row_units);

  view_t wrench_change("wrench_change", num_rows);
  make_block_force_op<ExecSpace>(trial_geo, num_rods).apply(iterate.y, wrench_change);
  backend_t::axpby(-1.0, iterate.bilateral_wrench, 1.0, wrench_change);
  view_t displacement_change("displacement_change", num_rows);
  step.mobility.apply(step.dt, wrench_change, 0.0, displacement_change);
  const LengthAngleMax moved = max_displacement<ExecSpace>(displacement_change, num_rods);

  return std::max(
      {rows.length / length_tol, rows.angle / angle_tol, moved.length / length_tol, moved.angle / angle_tol});
}

//@}

}  // namespace impl

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_IMPL_KOKKOSMBODYIMPL_HPP_
