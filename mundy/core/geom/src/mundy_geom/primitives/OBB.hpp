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

#ifndef MUNDY_GEOM_PRIMITIVES_OBB_HPP_
#define MUNDY_GEOM_PRIMITIVES_OBB_HPP_

/// \file primitives/OBB.hpp
/// \brief Oriented Bounding Box (OBB) primitive.
///
/// ## Storage
///
/// An OBB is stored as three independent quantities:
///   - **center** — a `Point<Scalar>` giving the box centroid in world space.
///   - **orientation** — a unit `Quaternion<Scalar>` representing the rotation that maps local
///     box axes to world space.  The local x/y/z axes are the first/second/third columns of the
///     rotation matrix derived from this quaternion.
///   - **half_extents** — a `Vector3<Scalar>` giving the half-lengths of the box along each
///     local axis (all non-negative).
///
/// ## Why quaternion orientation?
///
/// Performance tests show that computing the relative rotation matrix R = mat(conj(q_A)*q_B)
/// needed for an OBB–OBB SAT test costs essentially the same as R = Q_A^T Q_B when orientations
/// are stored as rotation matrices (~6 ns on a Cascade Lake core), while the quaternion
/// representation reduces the per-OBB memory footprint from 15 scalars to 10 scalars. On
/// memory-bandwidth-bound GPU workloads — the primary use case for MundySearch narrow-phase
/// excluders — the smaller footprint is preferred.

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Mundy
#include <mundy_geom/primitives/Point.hpp>   // for mundy::Point, ValidPointType
#include <mundy_math/Quaternion.hpp>         // for mundy::Quaternion, ValidQuaternionType, conjugate, quaternion_to_rotation_matrix
#include <mundy_math/Vector3.hpp>            // for mundy::Vector3, ValidVector3Type
#include <mundy_math/cmath.hpp>              // for mundy::abs
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>      // for MUNDY_THROW_ASSERT

namespace mundy {

/// \addtogroup MundyGeomPrimitives
/// @{

/// \class OBB
/// \brief Oriented Bounding Box: a center, a unit-quaternion orientation, and per-axis half-extents.
///
/// \tparam Scalar          Floating-point scalar type.
/// \tparam PointType       Point type for the center; defaults to `Point<Scalar>`.
/// \tparam QuaternionType  Quaternion type for the orientation; defaults to `Quaternion<Scalar>`.
/// \tparam HalfExtentsType Vector3 type for the half-extents; defaults to `Vector3<Scalar>`.
///                        A view variant (e.g. an owning Vector3 aliasing field storage) may be
///                        substituted here to enable field-backed OBB components.
template <typename Scalar, //
          ValidPointType      PointType      = Point<Scalar>, //
          ValidQuaternionType QuaternionType = Quaternion<Scalar>, //
          ValidVector3Type    HalfExtentsType = Vector3<Scalar>> //
class OBB {
  static_assert(std::is_same_v<typename PointType::value_type,      Scalar> &&
                std::is_same_v<typename QuaternionType::value_type,  Scalar> &&
                std::is_same_v<typename HalfExtentsType::value_type, Scalar>,
                "The value_type of PointType, QuaternionType, and HalfExtentsType must match Scalar.");

 public:
  //! \name Type aliases
  //@{

  using value_type     = Scalar;
  using point_t        = PointType;
  using orientation_t  = QuaternionType;
  using half_extents_t = HalfExtentsType;

  static constexpr bool is_finite = true;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor.  Center and half-extents are zero-initialized; orientation is
  /// the identity quaternion; half-extents are set to the invalid sentinel -1.
  KOKKOS_FUNCTION
  constexpr OBB()
      MUNDY_REQUIRES(HasNArgConstructor<point_t, value_type, 3> && HasNArgConstructor<orientation_t, value_type, 4>)
      : center_(value_type(), value_type(), value_type()),
        orientation_(static_cast<value_type>(1),
                     static_cast<value_type>(0),
                     static_cast<value_type>(0),
                     static_cast<value_type>(0)),
        half_extents_(static_cast<value_type>(-1),
                      static_cast<value_type>(-1),
                      static_cast<value_type>(-1)) {}

  /// \brief Construct from a center, identity orientation, and uniform half-extent.
  /// \param[in] center      Centroid of the box in world space.
  /// \param[in] half_extent Half-length of the box along each local axis.
  KOKKOS_FUNCTION
  constexpr OBB(const point_t& center, const value_type& half_extent)
      MUNDY_REQUIRES(HasNArgConstructor<orientation_t, value_type, 4>)
      : center_(center),
        orientation_(static_cast<value_type>(1),
                     static_cast<value_type>(0),
                     static_cast<value_type>(0),
                     static_cast<value_type>(0)),
        half_extents_(half_extent, half_extent, half_extent) {}

  /// \brief Construct from a center, orientation, and per-axis half-extents.
  /// \param[in] center       Centroid of the box in world space.
  /// \param[in] orientation  Unit quaternion mapping local axes to world space.
  /// \param[in] half_extents Half-lengths along each local axis (must be non-negative).
  KOKKOS_FUNCTION
  constexpr OBB(const point_t&       center,
                const orientation_t& orientation,
                const half_extents_t& half_extents)
      : center_(center), orientation_(orientation), half_extents_(half_extents) {}

  /// \brief Construct from a center, orientation, and per-axis half-extents given as scalars.
  /// \param[in] center       Centroid of the box in world space.
  /// \param[in] orientation  Unit quaternion mapping local axes to world space.
  /// \param[in] hx           Half-length along local x-axis.
  /// \param[in] hy           Half-length along local y-axis.
  /// \param[in] hz           Half-length along local z-axis.
  KOKKOS_FUNCTION
  constexpr OBB(const point_t& center, const orientation_t& orientation,
                const value_type& hx, const value_type& hy, const value_type& hz)
      : center_(center), orientation_(orientation), half_extents_(hx, hy, hz) {}

  /// \brief Destructor.
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~OBB() = default;

  /// \brief Copy constructor.
  KOKKOS_FUNCTION
  constexpr OBB(const OBB& other)
      : center_(other.center_), orientation_(other.orientation_), half_extents_(other.half_extents_) {}

  /// \brief Move constructor.
  KOKKOS_FUNCTION
  constexpr OBB(OBB&& other)
      : center_(std::move(other.center_)),
        orientation_(std::move(other.orientation_)),
        half_extents_(std::move(other.half_extents_)) {}
  //@}

  //! \name Assignment operators
  //@{

  KOKKOS_FUNCTION
  constexpr OBB& operator=(const OBB& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign OBB to itself.");
    center_       = other.center_;
    orientation_  = other.orientation_;
    half_extents_ = other.half_extents_;
    return *this;
  }

  KOKKOS_FUNCTION
  constexpr OBB& operator=(OBB&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign OBB to itself.");
    center_       = std::move(other.center_);
    orientation_  = std::move(other.orientation_);
    half_extents_ = std::move(other.half_extents_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  // clang-format off
  KOKKOS_FUNCTION constexpr const point_t&       center()       const { return center_; }
  KOKKOS_FUNCTION constexpr       point_t&       center()             { return center_; }
  KOKKOS_FUNCTION constexpr const orientation_t& orientation()  const { return orientation_; }
  KOKKOS_FUNCTION constexpr       orientation_t& orientation()        { return orientation_; }
  KOKKOS_FUNCTION constexpr const half_extents_t& half_extents() const { return half_extents_; }
  KOKKOS_FUNCTION constexpr       half_extents_t& half_extents()       { return half_extents_; }
  KOKKOS_FUNCTION constexpr const value_type& half_extent(int i) const { return half_extents_[i]; }
  KOKKOS_FUNCTION constexpr       value_type& half_extent(int i)       { return half_extents_[i]; }
  // clang-format on
  //@}

  //! \name Setters
  //@{

  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr void set_center(const OtherPointType& c) { center_ = c; }

  template <ValidQuaternionType OtherQuatType>
  KOKKOS_FUNCTION constexpr void set_orientation(const OtherQuatType& q) { orientation_ = q; }

  template <ValidVector3Type OtherVecType>
  KOKKOS_FUNCTION constexpr void set_half_extents(const OtherVecType& he) { half_extents_ = he; }

  KOKKOS_FUNCTION
  constexpr void set_half_extents(const value_type& hx, const value_type& hy, const value_type& hz) {
    half_extents_[0] = hx;
    half_extents_[1] = hy;
    half_extents_[2] = hz;
  }
  //@}

 private:
  template <typename, ValidPointType, ValidQuaternionType, ValidVector3Type>
  friend class OBB;

  point_t        center_;
  orientation_t  orientation_;
  half_extents_t half_extents_;
};

// =============================================================================
// Deduction guides
// =============================================================================

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <ValidPointType P, ValidQuaternionType Q, ValidVector3Type V>
OBB(P, Q, V) -> OBB<typename P::value_type, P, Q, V>;
#endif

// =============================================================================
// Type trait and concept
// =============================================================================

template <typename T>
struct is_obb_impl : std::false_type {};
template <typename Scalar, typename P, typename Q, typename H>
struct is_obb_impl<OBB<Scalar, P, Q, H>> : std::true_type {};

template <typename T>
struct is_obb : is_obb_impl<std::remove_cv_t<T>> {};
template <typename T>
inline constexpr bool is_obb_v = is_obb<T>::value;

template <typename OBBType>
concept ValidOBBType = is_obb_v<OBBType>;

// =============================================================================
// Non-member functions
// =============================================================================

//! \name Approximate equality
//@{

template <ValidOBBType T1, ValidOBBType T2>
KOKKOS_FUNCTION constexpr bool is_close(
    const T1& a, const T2& b,
    typename T1::value_type tol = get_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(a.center(),       b.center(),       tol) &&
         is_close(a.orientation(),  b.orientation(),  tol) &&
         is_close(a.half_extents(), b.half_extents(), tol);
}

template <ValidOBBType T1, ValidOBBType T2>
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const T1& a, const T2& b,
    typename T1::value_type tol = get_relaxed_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(a, b, tol);
}
//@}

//! \name Intersection test
//@{

/// \brief Test whether two OBBs overlap using the Separating Axis Theorem (SAT).
///
/// Tests 15 candidate separating axes: 3 face normals of A, 3 face normals of B,
/// and 9 edge cross-products A_i × B_j.  Returns `true` if the boxes overlap on
/// every axis (i.e. no separating axis was found).
///
/// The relative rotation is computed via `mat(conj(q_A)*q_B)` rather than
/// materialising both rotation matrices — consistent with the quaternion storage
/// choice documented in the class header.
///
/// \param[in] a  First OBB.
/// \param[in] b  Second OBB.
template <ValidOBBType OBBType1, ValidOBBType OBBType2>
KOKKOS_FUNCTION constexpr bool intersects(const OBBType1& a, const OBBType2& b) {
  using S = typename OBBType1::value_type;
  // Epsilon added to absolute rotation entries so that when two edges are nearly
  // parallel (off-diagonal R entries ≈ 0 from floating-point noise rather than
  // true geometry), the test doesn't spuriously report separation.  The noise in
  // R(i,j) from the quaternion product is O(machine_eps), so get_zero_tolerance()
  // is the right scale — NOT sqrt(get_zero_tolerance()) used in squared-quantity
  // parallel checks (e.g. line-segment D = |u×v|²), and NOT the graphics-derived
  // magic number 1e-6, which creates an unacceptable 1e-6-length-unit safety margin.
  const S eps = get_zero_tolerance<S>();

  // R(i,j) = dot(a_axis_i, b_axis_j), computed as mat(conj(q_A)*q_B).
  const Matrix3<S> R = quaternion_to_rotation_matrix(conjugate(a.orientation()) * b.orientation());

  // Translation in A's local frame.
  const Vector3<S> T_world{b.center()[0] - a.center()[0],
                             b.center()[1] - a.center()[1],
                             b.center()[2] - a.center()[2]};
  const auto T  = conjugate(a.orientation()) * T_world;
  const S    t0 = T[0], t1 = T[1], t2 = T[2];

  // Absolute rotation entries (with epsilon for numerical robustness).
  const S r00 = abs(R(0,0))+eps, r01 = abs(R(0,1))+eps, r02 = abs(R(0,2))+eps;
  const S r10 = abs(R(1,0))+eps, r11 = abs(R(1,1))+eps, r12 = abs(R(1,2))+eps;
  const S r20 = abs(R(2,0))+eps, r21 = abs(R(2,1))+eps, r22 = abs(R(2,2))+eps;

  const S ha0 = a.half_extent(0), ha1 = a.half_extent(1), ha2 = a.half_extent(2);
  const S hb0 = b.half_extent(0), hb1 = b.half_extent(1), hb2 = b.half_extent(2);

  // Face normals of A (axes A0, A1, A2).
  if (abs(t0) > ha0 + hb0*r00 + hb1*r01 + hb2*r02) return false;
  if (abs(t1) > ha1 + hb0*r10 + hb1*r11 + hb2*r12) return false;
  if (abs(t2) > ha2 + hb0*r20 + hb1*r21 + hb2*r22) return false;

  // Face normals of B (axes B0, B1, B2).
  if (abs(t0*R(0,0)+t1*R(1,0)+t2*R(2,0)) > ha0*r00+ha1*r10+ha2*r20+hb0) return false;
  if (abs(t0*R(0,1)+t1*R(1,1)+t2*R(2,1)) > ha0*r01+ha1*r11+ha2*r21+hb1) return false;
  if (abs(t0*R(0,2)+t1*R(1,2)+t2*R(2,2)) > ha0*r02+ha1*r12+ha2*r22+hb2) return false;

  // Edge cross-products A_i × B_j (9 tests).
  if (abs(t2*R(1,0)-t1*R(2,0)) > ha1*r20+ha2*r10+hb1*r02+hb2*r01) return false;  // A0×B0
  if (abs(t2*R(1,1)-t1*R(2,1)) > ha1*r21+ha2*r11+hb0*r02+hb2*r00) return false;  // A0×B1
  if (abs(t2*R(1,2)-t1*R(2,2)) > ha1*r22+ha2*r12+hb0*r01+hb1*r00) return false;  // A0×B2
  if (abs(t0*R(2,0)-t2*R(0,0)) > ha0*r20+ha2*r00+hb1*r12+hb2*r11) return false;  // A1×B0
  if (abs(t0*R(2,1)-t2*R(0,1)) > ha0*r21+ha2*r01+hb0*r12+hb2*r10) return false;  // A1×B1
  if (abs(t0*R(2,2)-t2*R(0,2)) > ha0*r22+ha2*r02+hb0*r11+hb1*r10) return false;  // A1×B2
  if (abs(t1*R(0,0)-t0*R(1,0)) > ha0*r10+ha1*r00+hb1*r22+hb2*r21) return false;  // A2×B0
  if (abs(t1*R(0,1)-t0*R(1,1)) > ha0*r11+ha1*r01+hb0*r22+hb2*r20) return false;  // A2×B1
  if (abs(t1*R(0,2)-t0*R(1,2)) > ha0*r12+ha1*r02+hb0*r21+hb1*r20) return false;  // A2×B2

  return true;
}
//@}

//! \name Stream output
//@{

template <ValidOBBType T>
std::ostream& operator<<(std::ostream& os, const T& obb) {
  os << "{center=" << obb.center()
     << " orientation=" << obb.orientation()
     << " half_extents=" << obb.half_extents() << "}";
  return os;
}
//@}

//! \name Point visitation
//
// An OBB has a single geometric reference point: its center.
//@{

template <ValidOBBType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point(const T& obb, Functor&& f) {
  f(obb.center());
}

template <ValidOBBType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point_mutable(T& obb, Functor&& f) {
  f(obb.center());
}
//@}

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_OBB_HPP_
