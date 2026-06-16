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

// External libs
#include <gtest/gtest.h>
#include <openrand/philox.h>

// C++ core
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/periodicity.hpp>
#include <mundy_geom/primitives.hpp>
#include <mundy_geom/randomize.hpp>
#include <mundy_geom/transform.hpp>
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Tolerance.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_utils/rng.hpp>
#include <mundy_utils/throw_assert.hpp>
#include <mundy_math/cmath.hpp>

namespace mundy {

namespace {

// ============================================================
//! \name Shared test helpers
//@{
// ============================================================

KOKKOS_INLINE_FUNCTION bool is_point_in_box(const Point<double>& p, const AABB<double>& box) {
  return p[0] >= box.x_min() && p[0] <= box.x_max() && p[1] >= box.y_min() && p[1] <= box.y_max() &&
         p[2] >= box.z_min() && p[2] <= box.z_max();
}

/// \brief Per-type traits for the generic wrapping tests.
///
/// Each specialization provides: type, num_points, generate(), reference_point(), for_each_point().
enum class TestObjectType : std::uint8_t {
  POINT = 0,
  LINE,
  LINE_SEGMENT,
  SPHERE,
  ELLIPSOID,
  CIRCLE_3D,
  NUM_TYPES,
  INVALID = std::numeric_limits<std::uint8_t>::max()
};

inline std::ostream& operator<<(std::ostream& os, TestObjectType t) {
  switch (t) {
    case TestObjectType::POINT:
      return os << "POINT";
    case TestObjectType::LINE:
      return os << "LINE";
    case TestObjectType::LINE_SEGMENT:
      return os << "LINE_SEGMENT";
    case TestObjectType::SPHERE:
      return os << "SPHERE";
    case TestObjectType::ELLIPSOID:
      return os << "ELLIPSOID";
    case TestObjectType::CIRCLE_3D:
      return os << "CIRCLE_3D";
    default:
      return os << "INVALID";
  }
}

template <TestObjectType Type>
struct TestObjectTraits;

template <>
struct TestObjectTraits<TestObjectType::POINT> {
  using type = Point<double>;
  static constexpr unsigned num_points = 1;
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_point<double>(box, rng);
  }
  static KOKKOS_FUNCTION Point<double> reference_point(const type& p) {
    return p;
  }
  template <typename F>
  KOKKOS_FUNCTION static void for_each_point(const type& p, const F& f) {
    f(p);
  }
};

template <>
struct TestObjectTraits<TestObjectType::LINE> {
  using type = Line<double>;
  static constexpr unsigned num_points = 1;
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_line<double>(box, rng);
  }
  static KOKKOS_FUNCTION Point<double> reference_point(const type& l) {
    return l.center();
  }
  template <typename F>
  KOKKOS_FUNCTION static void for_each_point(const type& l, const F& f) {
    f(l.center());
  }
};

template <>
struct TestObjectTraits<TestObjectType::LINE_SEGMENT> {
  using type = LineSegment<double>;
  static constexpr unsigned num_points = 2;
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    return generate_random_line_segment<double>(box, rng);
  }
  static KOKKOS_FUNCTION Point<double> reference_point(const type& ls) {
    return ls.start();
  }
  template <typename F>
  KOKKOS_FUNCTION static void for_each_point(const type& ls, const F& f) {
    f(ls.start());
    f(ls.end());
  }
};

template <>
struct TestObjectTraits<TestObjectType::SPHERE> {
  using type = Sphere<double>;
  static constexpr unsigned num_points = 1;
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    double w =
        min(min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
    return generate_random_sphere<double>(box, 0.0, 0.25 * w, rng);
  }
  static KOKKOS_FUNCTION Point<double> reference_point(const type& s) {
    return s.center();
  }
  template <typename F>
  KOKKOS_FUNCTION static void for_each_point(const type& s, const F& f) {
    f(s.center());
  }
};

template <>
struct TestObjectTraits<TestObjectType::ELLIPSOID> {
  using type = Ellipsoid<double>;
  static constexpr unsigned num_points = 1;
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    double w =
        min(min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
    double r = 0.25 * w;
    return generate_random_ellipsoid<double>(box, Vector3<double>{0.0, 0.0, 0.0}, Vector3<double>{r, r, r}, rng);
  }
  static KOKKOS_FUNCTION Point<double> reference_point(const type& e) {
    return e.center();
  }
  template <typename F>
  KOKKOS_FUNCTION static void for_each_point(const type& e, const F& f) {
    f(e.center());
  }
};

template <>
struct TestObjectTraits<TestObjectType::CIRCLE_3D> {
  using type = Circle3D<double>;
  static constexpr unsigned num_points = 1;
  static type generate(const AABB<double>& box, openrand::Philox& rng) {
    double w =
        min(min(box.x_max() - box.x_min(), box.y_max() - box.y_min()), box.z_max() - box.z_min());
    return generate_random_circle3D<double>(box, 0.0, 0.25 * w, rng);
  }
  static KOKKOS_FUNCTION Point<double> reference_point(const type& c) {
    return c.center();
  }
  template <typename F>
  KOKKOS_FUNCTION static void for_each_point(const type& c, const F& f) {
    f(c.center());
  }
};

template <typename Functor>
struct apply_functor {
  using return_type = typename Functor::return_type;

  KOKKOS_DEFAULTED_FUNCTION apply_functor() = default;
  KOKKOS_FUNCTION apply_functor(const Functor& f) : functor_(f) {
  }

  template <typename... Args>
  KOKKOS_FUNCTION return_type operator()(TestObjectType rt, Args&&... args) const {
    switch (rt) {
      case TestObjectType::POINT:
        return functor_(TestObjectTraits<TestObjectType::POINT>{}, std::forward<Args>(args)...);
      case TestObjectType::LINE:
        return functor_(TestObjectTraits<TestObjectType::LINE>{}, std::forward<Args>(args)...);
      case TestObjectType::LINE_SEGMENT:
        return functor_(TestObjectTraits<TestObjectType::LINE_SEGMENT>{}, std::forward<Args>(args)...);
      case TestObjectType::SPHERE:
        return functor_(TestObjectTraits<TestObjectType::SPHERE>{}, std::forward<Args>(args)...);
      case TestObjectType::ELLIPSOID:
        return functor_(TestObjectTraits<TestObjectType::ELLIPSOID>{}, std::forward<Args>(args)...);
      case TestObjectType::CIRCLE_3D:
        return functor_(TestObjectTraits<TestObjectType::CIRCLE_3D>{}, std::forward<Args>(args)...);
      default:
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unsupported TestObjectType");
        return return_type{};
    }
  }

  Functor functor_;
};

//@}

// ============================================================
//! \name Layer 0 — safe_unit_mod1 edge cases
//@{
// ============================================================

TEST(SafeUnitMod1, ValuesInUnit) {
  EXPECT_NEAR(impl::safe_unit_mod1(0.0), 0.0, 1e-15);
  EXPECT_NEAR(impl::safe_unit_mod1(0.3), 0.3, 1e-15);
  EXPECT_NEAR(impl::safe_unit_mod1(0.7), 0.7, 1e-15);
}

TEST(SafeUnitMod1, ExactlyOneMapsToZero) {
  EXPECT_NEAR(impl::safe_unit_mod1(1.0), 0.0, 1e-14);
}

TEST(SafeUnitMod1, FractionalParts) {
  EXPECT_NEAR(impl::safe_unit_mod1(1.5), 0.5, 1e-14);
  EXPECT_NEAR(impl::safe_unit_mod1(-0.5), 0.5, 1e-14);
  EXPECT_NEAR(impl::safe_unit_mod1(2.25), 0.25, 1e-14);
  EXPECT_NEAR(impl::safe_unit_mod1(-1.75), 0.25, 1e-14);
}

// The guard in safe_unit_mod1 exists precisely for this case: floor(-1e-300) = -1,
// so t = -1e-300 + 1 rounds to exactly 1.0 in IEEE 754 without the clamp.
TEST(SafeUnitMod1, TinyNegativeIsInUnit) {
  double result = impl::safe_unit_mod1(-1e-300);
  EXPECT_GE(result, 0.0);
  EXPECT_LT(result, 1.0);
}

TEST(SafeUnitMod1, LargePositiveIsInUnit) {
  double result = impl::safe_unit_mod1(1e15);
  EXPECT_GE(result, 0.0);
  EXPECT_LT(result, 1.0);
}

TEST(SafeUnitMod1, LargeNegativeIsInUnit) {
  double result = impl::safe_unit_mod1(-1e15);
  EXPECT_GE(result, 0.0);
  EXPECT_LT(result, 1.0);
}

//@}

// ============================================================
//! \name Layer 1 — Mathematical contracts on concrete metrics
//@{
// ============================================================

// --- FreeSpaceMetric ---

TEST(FreeSpaceMetric, FractionalRoundtrip) {
  FreeSpaceMetric<double> m;
  Point<double> p{3.14, -2.71, 1.0};
  EXPECT_TRUE(is_close(m.from_fractional(m.to_fractional(p)), p));
}

TEST(FreeSpaceMetric, WrapIsIdentity) {
  FreeSpaceMetric<double> m;
  Point<double> p{1e10, -1e10, 42.0};
  EXPECT_TRUE(is_close(m.wrap(p), p));
}

TEST(FreeSpaceMetric, SepIsDirectDifference) {
  FreeSpaceMetric<double> m;
  Point<double> a{1.0, 2.0, 3.0};
  Point<double> b{4.0, 5.0, 6.0};
  auto s = m.sep(a, b);
  EXPECT_NEAR(s[0], 3.0, 1e-15);
  EXPECT_NEAR(s[1], 3.0, 1e-15);
  EXPECT_NEAR(s[2], 3.0, 1e-15);
}

TEST(FreeSpaceMetric, SepAntisymmetry) {
  FreeSpaceMetric<double> m;
  Point<double> a{1.0, 2.0, 3.0};
  Point<double> b{4.0, -1.0, 6.0};
  auto ab = m.sep(a, b);
  auto ba = m.sep(b, a);
  EXPECT_NEAR(ab[0], -ba[0], 1e-15);
  EXPECT_NEAR(ab[1], -ba[1], 1e-15);
  EXPECT_NEAR(ab[2], -ba[2], 1e-15);
}

TEST(FreeSpaceMetric, DirectLatticeVectorsIsIdentity) {
  FreeSpaceMetric<double> m;
  auto dlv = m.direct_lattice_vectors();
  auto id = Matrix3<double>::identity();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) EXPECT_NEAR(dlv(i, j), id(i, j), 1e-15);
}

TEST(FreeSpaceMetric, ShiftImageIsNoop) {
  FreeSpaceMetric<double> m;
  Point<double> p{5.0, 5.0, 5.0};
  EXPECT_TRUE(is_close(m.shift_image(p, Vector3<int>{3, -2, 1}), p));
}

TEST(FreeSpaceMetric, IsPeriodicAlwaysFalse) {
  FreeSpaceMetric<double> m;
  EXPECT_FALSE(m.is_periodic(0));
  EXPECT_FALSE(m.is_periodic(1));
  EXPECT_FALSE(m.is_periodic(2));
  EXPECT_FALSE((FreeSpaceMetric<double>::is_periodic<0>()));
  EXPECT_EQ(m.num_periodic_dimensions(), 0u);
}

// --- OrthorhombicMetric ---

TEST(OrthorhombicMetric, FractionalRoundtrip) {
  OrthorhombicMetric<AXIS_XYZ, double> m{Vector3<double>{10.0, 20.0, 30.0}};
  Point<double> p{3.14, 15.0, 27.0};
  auto rt = m.from_fractional(m.to_fractional(p));
  EXPECT_NEAR(rt[0], p[0], 1e-12);
  EXPECT_NEAR(rt[1], p[1], 1e-12);
  EXPECT_NEAR(rt[2], p[2], 1e-12);
}

TEST(OrthorhombicMetric, WrapIdempotency) {
  OrthorhombicMetric<AXIS_XYZ, double> m{Vector3<double>{10.0, 10.0, 10.0}};
  Point<double> p{-3.0, 14.0, 7.5};
  auto wp = m.wrap(p);
  auto wwp = m.wrap(wp);
  EXPECT_NEAR(wwp[0], wp[0], 1e-12);
  EXPECT_NEAR(wwp[1], wp[1], 1e-12);
  EXPECT_NEAR(wwp[2], wp[2], 1e-12);
}

TEST(OrthorhombicMetric, WrapPlacesInCell) {
  const Vector3<double> L{10.0, 20.0, 30.0};
  OrthorhombicMetric<AXIS_XYZ, double> m{L};
  Point<double> p{-101.0, 45.0, 77.0};
  auto wp = m.wrap(p);
  EXPECT_GE(wp[0], 0.0);
  EXPECT_LT(wp[0], L[0]);
  EXPECT_GE(wp[1], 0.0);
  EXPECT_LT(wp[1], L[1]);
  EXPECT_GE(wp[2], 0.0);
  EXPECT_LT(wp[2], L[2]);
}

TEST(OrthorhombicMetric, SepAntisymmetry) {
  OrthorhombicMetric<AXIS_XYZ, double> m{Vector3<double>{10.0, 10.0, 10.0}};
  Point<double> a{1.0, 2.0, 3.0};
  Point<double> b{7.0, 8.0, 9.0};
  auto ab = m.sep(a, b);
  auto ba = m.sep(b, a);
  EXPECT_NEAR(ab[0], -ba[0], 1e-12);
  EXPECT_NEAR(ab[1], -ba[1], 1e-12);
  EXPECT_NEAR(ab[2], -ba[2], 1e-12);
}

TEST(OrthorhombicMetric, SepChoosesShortPath) {
  // a at 0.1*Lx, b at 0.9*Lx: short path is 0.2*Lx backward, not 0.8*Lx forward
  const double Lx = 10.0;
  OrthorhombicMetric<AXIS_X, double> m{Vector3<double>{Lx, 1.0, 1.0}};
  Point<double> a{0.1 * Lx, 0.0, 0.0};
  Point<double> b{0.9 * Lx, 0.0, 0.0};
  EXPECT_NEAR(m.sep(a, b)[0], -0.2 * Lx, 1e-12);
}

TEST(OrthorhombicMetric, NonPeriodicAxesUntouchedByWrap) {
  // Only X is periodic; Y and Z should be left exactly where they are.
  OrthorhombicMetric<AXIS_X, double> m{Vector3<double>{10.0, 1.0, 1.0}};
  Point<double> p{-3.0, 500.0, -700.0};
  auto wp = m.wrap(p);
  EXPECT_GE(wp[0], 0.0);
  EXPECT_LT(wp[0], 10.0);
  // Y and Z untouched (scale for non-periodic axes is 1 internally, so wrap is identity)
  EXPECT_NEAR(wp[1], p[1], 1e-12);
  EXPECT_NEAR(wp[2], p[2], 1e-12);
}

TEST(OrthorhombicMetric, NonPeriodicAxisSepIsDirectDifference) {
  // For a non-periodic axis sep should be the plain difference, not minimum-image.
  OrthorhombicMetric<AXIS_X, double> m{Vector3<double>{10.0, 1.0, 1.0}};
  Point<double> a{0.0, 0.0, 0.0};
  Point<double> b{0.0, 500.0, 0.0};
  EXPECT_NEAR(m.sep(a, b)[1], 500.0, 1e-12);
}

TEST(OrthorhombicMetric, ShiftImageByOneCell) {
  const double Lx = 10.0, Ly = 20.0, Lz = 30.0;
  OrthorhombicMetric<AXIS_XYZ, double> m{Vector3<double>{Lx, Ly, Lz}};
  Point<double> p{5.0, 10.0, 15.0};
  auto sx = m.shift_image(p, Vector3<int>{1, 0, 0});
  EXPECT_NEAR(sx[0], p[0] + Lx, 1e-12);
  EXPECT_NEAR(sx[1], p[1], 1e-12);
  EXPECT_NEAR(sx[2], p[2], 1e-12);
  auto sxyz = m.shift_image(p, Vector3<int>{1, -1, 2});
  EXPECT_NEAR(sxyz[0], p[0] + Lx, 1e-12);
  EXPECT_NEAR(sxyz[1], p[1] - Ly, 1e-12);
  EXPECT_NEAR(sxyz[2], p[2] + 2.0 * Lz, 1e-12);
}

TEST(OrthorhombicMetric, DirectLatticeVectorsDiagonal) {
  const double Lx = 10.0, Ly = 20.0, Lz = 30.0;
  OrthorhombicMetric<AXIS_XYZ, double> m{Vector3<double>{Lx, Ly, Lz}};
  auto dlv = m.direct_lattice_vectors();
  EXPECT_NEAR(dlv(0, 0), Lx, 1e-12);
  EXPECT_NEAR(dlv(1, 1), Ly, 1e-12);
  EXPECT_NEAR(dlv(2, 2), Lz, 1e-12);
  EXPECT_NEAR(dlv(0, 1), 0.0, 1e-15);
  EXPECT_NEAR(dlv(0, 2), 0.0, 1e-15);
  EXPECT_NEAR(dlv(1, 0), 0.0, 1e-15);
}

TEST(OrthorhombicMetric, IsPeriodicMatchesBitmask) {
  OrthorhombicMetric<AXIS_XY, double> m{Vector3<double>{10.0, 10.0, 1.0}};
  EXPECT_TRUE(m.is_periodic(0));
  EXPECT_TRUE(m.is_periodic(1));
  EXPECT_FALSE(m.is_periodic(2));
  EXPECT_EQ(m.num_periodic_dimensions(), 2u);
}

// --- TriclinicMetric ---

TEST(TriclinicMetric, DefaultConstructedIsIdentity) {
  TriclinicMetric<AXIS_XYZ, double> m;
  Point<double> p{3.14, -2.71, 1.0};
  auto rt = m.from_fractional(m.to_fractional(p));
  EXPECT_NEAR(rt[0], p[0], 1e-12);
  EXPECT_NEAR(rt[1], p[1], 1e-12);
  EXPECT_NEAR(rt[2], p[2], 1e-12);
}

TEST(TriclinicMetric, FractionalRoundtrip) {
  Matrix3<double> h{10.0, 2.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 10.0};
  TriclinicMetric<AXIS_XYZ, double> m{h};
  Point<double> p{3.5, 7.2, 5.8};
  auto rt = m.from_fractional(m.to_fractional(p));
  EXPECT_NEAR(rt[0], p[0], 1e-10);
  EXPECT_NEAR(rt[1], p[1], 1e-10);
  EXPECT_NEAR(rt[2], p[2], 1e-10);
}

TEST(TriclinicMetric, WrapPlacesFracCoordInUnit) {
  // Orthogonal cell as a matrix; fractional coords of wrapped point must be in [0,1).
  Matrix3<double> h{10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  TriclinicMetric<AXIS_XYZ, double> m{h};
  Point<double> p{-3.0, 14.0, 7.5};
  auto frac = m.to_fractional(m.wrap(p));
  EXPECT_GE(frac[0], 0.0);
  EXPECT_LT(frac[0], 1.0);
  EXPECT_GE(frac[1], 0.0);
  EXPECT_LT(frac[1], 1.0);
  EXPECT_GE(frac[2], 0.0);
  EXPECT_LT(frac[2], 1.0);
}

TEST(TriclinicMetric, SepAntisymmetry) {
  Matrix3<double> h{10.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  TriclinicMetric<AXIS_XYZ, double> m{h};
  Point<double> a{1.0, 2.0, 3.0};
  Point<double> b{7.0, 8.0, 9.0};
  auto ab = m.sep(a, b);
  auto ba = m.sep(b, a);
  EXPECT_NEAR(ab[0], -ba[0], 1e-10);
  EXPECT_NEAR(ab[1], -ba[1], 1e-10);
  EXPECT_NEAR(ab[2], -ba[2], 1e-10);
}

TEST(TriclinicMetric, SepChoosesShortPathAlongFirstLatticeVector) {
  // Sheared cell. Points at fractional (0.1, 0, 0) and (0.9, 0, 0).
  // Min image displacement is -0.2 * first column of h.
  Matrix3<double> h{10.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  TriclinicMetric<AXIS_XYZ, double> m{h};
  // Construct points from fractional coordinates
  Point<double> a = m.from_fractional(Point<double>{0.1, 0.0, 0.0});
  Point<double> b = m.from_fractional(Point<double>{0.9, 0.0, 0.0});
  // sep(a,b) = from_frac(frac_min_image(to_frac(b-a))) = from_frac({-0.2,0,0}) = -0.2*col0(h)
  auto s = m.sep(a, b);
  EXPECT_NEAR(s[0], -0.2 * 10.0, 1e-10);
  EXPECT_NEAR(s[1], 0.0, 1e-10);
  EXPECT_NEAR(s[2], 0.0, 1e-10);
}

TEST(TriclinicMetric, NonPeriodicAxesUntouchedByWrap) {
  // Only first lattice direction is periodic. Y/Z fractional coords should be unchanged.
  Matrix3<double> h{10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  TriclinicMetric<AXIS_X, double> m{h};
  Point<double> p{-3.0, 500.0, -700.0};
  auto frac_orig = m.to_fractional(p);
  auto frac_wrap = m.to_fractional(m.wrap(p));
  EXPECT_GE(frac_wrap[0], 0.0);
  EXPECT_LT(frac_wrap[0], 1.0);
  EXPECT_NEAR(frac_wrap[1], frac_orig[1], 1e-12);
  EXPECT_NEAR(frac_wrap[2], frac_orig[2], 1e-12);
}

TEST(TriclinicMetric, ShiftImageByOneCell) {
  Matrix3<double> h{10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0};
  TriclinicMetric<AXIS_XYZ, double> m{h};
  Point<double> p{5.0, 10.0, 15.0};
  auto sx = m.shift_image(p, Vector3<int>{1, 0, 0});
  EXPECT_NEAR(sx[0], p[0] + 10.0, 1e-12);
  EXPECT_NEAR(sx[1], p[1], 1e-12);
  EXPECT_NEAR(sx[2], p[2], 1e-12);
}

//@}

// ============================================================
//! \name Layer 2 — Runtime metric API
//@{
// ============================================================

TEST(RuntimeMetric, DefaultIsInvalid) {
  metric m;
  EXPECT_FALSE(m.is_valid());
}

TEST(RuntimeMetric, ConstructFromEachEnumIsValid) {
  const std::vector<metric::metric_t> all_values = {
      metric::FREE_SPACE,      metric::ORTHORHOMBIC_X,  metric::ORTHORHOMBIC_Y,  metric::ORTHORHOMBIC_Z,
      metric::ORTHORHOMBIC_XY, metric::ORTHORHOMBIC_XZ, metric::ORTHORHOMBIC_YZ, metric::ORTHORHOMBIC_XYZ,
      metric::TRICLINIC_X,     metric::TRICLINIC_Y,     metric::TRICLINIC_Z,     metric::TRICLINIC_XY,
      metric::TRICLINIC_XZ,    metric::TRICLINIC_YZ,    metric::TRICLINIC_XYZ,
  };
  for (auto v : all_values) {
    metric m(v);
    EXPECT_TRUE(m.is_valid()) << "value=" << static_cast<unsigned>(v);
    EXPECT_EQ(m.value(), v);
  }
}

TEST(RuntimeMetric, StringRoundtrip) {
  const std::vector<metric::metric_t> all_values = {
      metric::FREE_SPACE,      metric::ORTHORHOMBIC_X,  metric::ORTHORHOMBIC_Y,  metric::ORTHORHOMBIC_Z,
      metric::ORTHORHOMBIC_XY, metric::ORTHORHOMBIC_XZ, metric::ORTHORHOMBIC_YZ, metric::ORTHORHOMBIC_XYZ,
      metric::TRICLINIC_X,     metric::TRICLINIC_Y,     metric::TRICLINIC_Z,     metric::TRICLINIC_XY,
      metric::TRICLINIC_XZ,    metric::TRICLINIC_YZ,    metric::TRICLINIC_XYZ,
  };
  for (auto v : all_values) {
    metric m(v);
    auto roundtrip = metric::from_string(m.name());
    EXPECT_EQ(roundtrip.value(), v) << "name=" << m.name();
  }
}

TEST(RuntimeMetric, StreamOutputMatchesName) {
  metric m(metric::ORTHORHOMBIC_XYZ);
  std::ostringstream oss;
  oss << m;
  EXPECT_EQ(oss.str(), std::string(m.name()));
}

TEST(RuntimeMetric, FromStringAliases) {
  EXPECT_EQ(metric::from_string("ORTHORHOMBIC").value(), metric::ORTHORHOMBIC_XYZ);
  EXPECT_EQ(metric::from_string("TRICLINIC").value(), metric::TRICLINIC_XYZ);
}

TEST(RuntimeMetric, FromStringThrowsOnUnrecognized) {
#ifdef NDEBUG
  GTEST_SKIP() << "Throw asserts disabled in release builds";
#else
  EXPECT_THROW(metric::from_string("NOT_A_METRIC"), std::invalid_argument);
  EXPECT_THROW(metric::from_string(""), std::invalid_argument);
  EXPECT_THROW(metric::from_string("orthorhombic"), std::invalid_argument);  // case-sensitive
#endif
}

TEST(RuntimeMetric, IsPeriodicFreeSpace) {
  metric m(metric::FREE_SPACE);
  EXPECT_FALSE(m.is_periodic(0));
  EXPECT_FALSE(m.is_periodic(1));
  EXPECT_FALSE(m.is_periodic(2));
}

TEST(RuntimeMetric, IsPeriodicOrthorhombicX) {
  metric m(metric::ORTHORHOMBIC_X);
  EXPECT_TRUE(m.is_periodic(0));
  EXPECT_FALSE(m.is_periodic(1));
  EXPECT_FALSE(m.is_periodic(2));
}

TEST(RuntimeMetric, IsPeriodicTriclinicYZ) {
  metric m(metric::TRICLINIC_YZ);
  EXPECT_FALSE(m.is_periodic(0));
  EXPECT_TRUE(m.is_periodic(1));
  EXPECT_TRUE(m.is_periodic(2));
}

TEST(RuntimeMetric, VisitFreeSpaceDispatchesCorrectType) {
  metric m(metric::FREE_SPACE);
  bool correct_type = false;
  m.visit([&](auto&& c) {
    using T = std::decay_t<decltype(c)>;
    correct_type = std::is_same_v<T, FreeSpaceMetric<double>>;
  });
  EXPECT_TRUE(correct_type);
}

TEST(RuntimeMetric, VisitOrthorhombicDispatchesCorrectType) {
  metric m(metric::ORTHORHOMBIC_XY);
  // visit() constructs OrthorhombicMetric from stored cell widths; the constructor
  // asserts widths > 0 for enabled periodic axes, so we must set valid widths first.
  m.set_cell_widths({1.0, 1.0, 1.0});
  bool correct_type = false;
  m.visit([&](auto&& c) {
    using T = std::decay_t<decltype(c)>;
    correct_type = std::is_same_v<T, OrthorhombicMetric<AXIS_XY, double>>;
  });
  EXPECT_TRUE(correct_type);
}

TEST(RuntimeMetric, VisitOrthorhombicSepMatchesDirect) {
  const Vector3<double> cell_widths{10.0, 20.0, 30.0};
  const Point<double> a{1.0, 2.0, 3.0};
  const Point<double> b{8.0, 18.0, 27.0};

  OrthorhombicMetric<AXIS_XYZ, double> direct{cell_widths};
  auto expected = direct.sep(a, b);

  metric m(metric::ORTHORHOMBIC_XYZ);
  m.set_cell_widths(cell_widths);
  m.visit([&](auto&& concrete) {
    auto actual = concrete.sep(a, b);
    EXPECT_NEAR(actual[0], expected[0], 1e-12);
    EXPECT_NEAR(actual[1], expected[1], 1e-12);
    EXPECT_NEAR(actual[2], expected[2], 1e-12);
  });
}

TEST(RuntimeMetric, VisitTriclinicSepMatchesDirect) {
  const Matrix3<double> h{10.0, 2.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 10.0};
  const Point<double> a{1.0, 2.0, 3.0};
  const Point<double> b{8.0, 7.0, 9.0};

  TriclinicMetric<AXIS_XYZ, double> direct{h};
  auto expected = direct.sep(a, b);

  metric m(metric::TRICLINIC_XYZ);
  m.set_cell_matrix(h);
  m.visit([&](auto&& concrete) {
    auto actual = concrete.sep(a, b);
    EXPECT_NEAR(actual[0], expected[0], 1e-10);
    EXPECT_NEAR(actual[1], expected[1], 1e-10);
    EXPECT_NEAR(actual[2], expected[2], 1e-10);
  });
}

TEST(RuntimeMetric, SetCellWidthsOnFreeSpaceThrows) {
#ifdef NDEBUG
  GTEST_SKIP() << "Throw asserts disabled in release builds";
#else
  metric m(metric::FREE_SPACE);
  EXPECT_THROW(m.set_cell_widths({10.0, 10.0, 10.0}), std::invalid_argument);
#endif
}

TEST(RuntimeMetric, SetCellMatrixOnOrthorhombicThrows) {
#ifdef NDEBUG
  GTEST_SKIP() << "Throw asserts disabled in release builds";
#else
  metric m(metric::ORTHORHOMBIC_XYZ);
  EXPECT_THROW(m.set_cell_matrix(Matrix3<double>::identity()), std::invalid_argument);
#endif
}

TEST(RuntimeMetric, AliasesHaveExpectedValues) {
  EXPECT_EQ(metric::ORTHORHOMBIC, metric::ORTHORHOMBIC_XYZ);
  EXPECT_EQ(metric::TRICLINIC, metric::TRICLINIC_XYZ);
  EXPECT_EQ(metric::ORTHORHOMBIC_START, metric::ORTHORHOMBIC_X);
  EXPECT_EQ(metric::ORTHORHOMBIC_END, metric::ORTHORHOMBIC_XYZ);
  EXPECT_EQ(metric::TRICLINIC_START, metric::TRICLINIC_X);
  EXPECT_EQ(metric::TRICLINIC_END, metric::TRICLINIC_XYZ);
}

//@}

// ============================================================
//! \name Layer 3 — Wrapping API (random property tests)
//@{
// ============================================================

// Stateless test functors used by the random-testing harness.

struct test_wrap_rigid_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& primary_box,
                                                const AABB<double>& other_box, RNG& rng, const Metric& metric) const {
    constexpr unsigned N = ShapeTraits::num_points;
    Kokkos::Array<Vector3<double>, N> orig_disp, wrap_disp;

    auto s = ShapeTraits::generate(other_box, rng);
    Point<double> ref = ShapeTraits::reference_point(s);

    unsigned i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& p) { orig_disp[i++] = p - ref; });

    s = wrap_rigid(s, metric);
    Point<double> wrapped_ref = ShapeTraits::reference_point(s);
    bool ref_in_box = is_point_in_box(wrapped_ref, primary_box);

    i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& p) { wrap_disp[i++] = p - wrapped_ref; });

    bool disps_preserved = true;
    for (unsigned j = 0; j < N; ++j) {
      if (norm(orig_disp[j] - wrap_disp[j]) > get_relaxed_zero_tolerance<double>()) {
        disps_preserved = false;
      }
    }
    return ref_in_box && disps_preserved;
  }
};

struct test_wrap_points_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& primary_box,
                                                const AABB<double>& disjoint_box, RNG& rng,
                                                const Metric& metric) const {
    if constexpr (is_finite_v<typename ShapeTraits::type>) {
      auto s = ShapeTraits::generate(disjoint_box, rng);
      s = wrap_points(s, metric);
      bool all_in = true;
      ShapeTraits::for_each_point(s, [&](const auto& p) {
        if (!is_point_in_box(p, primary_box)) all_in = false;
      });
      return all_in;
    } else {
      return true;
    }
  }
};

struct test_unwrap_to_ref_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& primary_box,
                                                const AABB<double>& disjoint_box, RNG& rng,
                                                const Metric& metric) const {
    if constexpr (is_finite_v<typename ShapeTraits::type>) {
      auto s = ShapeTraits::generate(primary_box, rng);
      Point<double> ref = generate_random_point<double>(disjoint_box, rng);

      s = unwrap_points_to_ref(s, metric, ref);

      // Each point is independently moved to its nearest image of ref. The only invariant
      // that holds for all object types is that the reference point ends up within half a
      // cell of ref on each periodic axis.
      Vector3<double> box_len{primary_box.x_max() - primary_box.x_min(), primary_box.y_max() - primary_box.y_min(),
                              primary_box.z_max() - primary_box.z_min()};
      bool within_half_cell = true;
      for (int d = 0; d < 3; ++d) {
        if (metric.is_periodic(d) && abs(reference_point(s)[d] - ref[d]) > 0.5 * box_len[d]) {
          within_half_cell = false;
          break;
        }
      }
      return within_half_cell;
    } else {
      return true;
    }
  }
};

struct test_shift_image_impl {
  using return_type = bool;

  template <typename ShapeTraits, typename RNG, typename Metric>
  KOKKOS_INLINE_FUNCTION return_type operator()(ShapeTraits, const AABB<double>& box, const Vector3<int>& lv, RNG& rng,
                                                const Metric& metric) const {
    constexpr unsigned N = ShapeTraits::num_points;
    Kokkos::Array<Vector3<double>, N> orig_disp, shift_disp;

    auto s = ShapeTraits::generate(box, rng);
    Point<double> ref = ShapeTraits::reference_point(s);

    unsigned i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& p) { orig_disp[i++] = p - ref; });

    s = shift_image(s, lv, metric);
    Point<double> new_ref = ShapeTraits::reference_point(s);

    auto expected_disp = metric.shift_image(ref, lv) - ref;
    bool disp_correct = norm((new_ref - ref) - expected_disp) < get_relaxed_zero_tolerance<double>();

    i = 0;
    ShapeTraits::for_each_point(s, [&](const auto& p) { shift_disp[i++] = p - new_ref; });

    bool disps_preserved = true;
    for (unsigned j = 0; j < N; ++j) {
      if (norm(orig_disp[j] - shift_disp[j]) > get_relaxed_zero_tolerance<double>()) {
        disps_preserved = false;
      }
    }
    return disp_correct && disps_preserved;
  }
};

// Convenience wrappers

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_wrap_rigid(TestObjectType type, const AABB<double>& primary, const AABB<double>& other,
                                            size_t seed, size_t ctr, const Metric& m) {
  MUNDY_THROW_ASSERT(ctr <= std::numeric_limits<uint32_t>::max(), std::overflow_error, "Counter exceeds uint32_t max.");
  openrand::Philox rng = make_philox(seed, ctr);
  return apply_functor<test_wrap_rigid_impl>{}(type, primary, other, rng, m);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_wrap_points(TestObjectType type, const AABB<double>& primary,
                                             const AABB<double>& disjoint, size_t seed, size_t ctr, const Metric& m) {
  MUNDY_THROW_ASSERT(ctr <= std::numeric_limits<uint32_t>::max(), std::overflow_error, "Counter exceeds uint32_t max.");
  openrand::Philox rng = make_philox(seed, ctr);
  return apply_functor<test_wrap_points_impl>{}(type, primary, disjoint, rng, m);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_unwrap_to_ref(TestObjectType type, const AABB<double>& primary,
                                               const AABB<double>& disjoint, size_t seed, size_t ctr, const Metric& m) {
  MUNDY_THROW_ASSERT(ctr <= std::numeric_limits<uint32_t>::max(), std::overflow_error, "Counter exceeds uint32_t max.");
  openrand::Philox rng = make_philox(seed, ctr);
  return apply_functor<test_unwrap_to_ref_impl>{}(type, primary, disjoint, rng, m);
}

template <typename Metric>
KOKKOS_INLINE_FUNCTION bool test_shift_image(TestObjectType type, const AABB<double>& box, const Vector3<int>& lv,
                                             size_t seed, size_t ctr, const Metric& m) {
  MUNDY_THROW_ASSERT(ctr <= std::numeric_limits<uint32_t>::max(), std::overflow_error, "Counter exceeds uint32_t max.");
  openrand::Philox rng = make_philox(seed, ctr);
  return apply_functor<test_shift_image_impl>{}(type, box, lv, rng, m);
}

// ---- Tests ----

// Compare OrthorhombicMetric<AXIS_XYZ> against brute-force 27-image search.
TEST(WrappingAPI, MinImageVsBruteForce_Orthorhombic) {
  const size_t seed = 1234;
  const size_t n = 100000;
  const Vector3<double> L{100.0, 100.0, 100.0};
  const AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  FreeSpaceMetric<double> free_m;
  OrthorhombicMetric<AXIS_XYZ, double> ortho_m{L};

  for (size_t t = 0; t < n; ++t) {
    openrand::Philox rng = make_philox(seed, t);
    Point<double> a = generate_random_point<double>(box, rng);
    Point<double> b = generate_random_point<double>(box, rng);

    // Brute-force minimum over 27 images
    double min_dist = std::numeric_limits<double>::max();
    for (int i = -1; i <= 1; ++i)
      for (int j = -1; j <= 1; ++j)
        for (int k = -1; k <= 1; ++k) {
          Vector3<double> disp{i * L[0], j * L[1], k * L[2]};
          min_dist = min(min_dist, norm(free_m.sep(a, b + disp)));
        }

    ASSERT_NEAR(norm(ortho_m.sep(a, b)), min_dist, get_relaxed_zero_tolerance<double>())
        << "Orthorhombic sep disagrees with brute-force minimum image.";
  }
}

// Same check for TriclinicMetric with a diagonal h (should match Orthorhombic).
TEST(WrappingAPI, MinImageVsBruteForce_TriclinicDiagonal) {
  const size_t seed = 5678;
  const size_t n = 10000;
  const double Lx = 10.0, Ly = 15.0, Lz = 20.0;
  Matrix3<double> h{Lx, 0.0, 0.0, 0.0, Ly, 0.0, 0.0, 0.0, Lz};
  const AABB<double> box{0.0, 0.0, 0.0, Lx, Ly, Lz};
  TriclinicMetric<AXIS_XYZ, double> tri_m{h};
  OrthorhombicMetric<AXIS_XYZ, double> ortho_m{Vector3<double>{Lx, Ly, Lz}};

  for (size_t t = 0; t < n; ++t) {
    openrand::Philox rng = make_philox(seed, t);
    Point<double> a = generate_random_point<double>(box, rng);
    Point<double> b = generate_random_point<double>(box, rng);

    double ortho_dist = norm(ortho_m.sep(a, b));
    double tri_dist = norm(tri_m.sep(a, b));
    ASSERT_NEAR(tri_dist, ortho_dist, get_relaxed_zero_tolerance<double>())
        << "Diagonal TriclinicMetric sep disagrees with OrthorhombicMetric.";
  }
}

static const std::vector<TestObjectType> kWrappableTypes = {
    TestObjectType::POINT, TestObjectType::LINE_SEGMENT, TestObjectType::SPHERE,
    TestObjectType::ELLIPSOID, TestObjectType::CIRCLE_3D,
};
static const std::vector<TestObjectType> kWrapRigidTypes = {
    TestObjectType::POINT,  TestObjectType::LINE,      TestObjectType::LINE_SEGMENT,
    TestObjectType::SPHERE, TestObjectType::ELLIPSOID, TestObjectType::CIRCLE_3D,
};

TEST(WrappingAPI, WrapRigid_Orthorhombic) {
  const size_t seed = 1234;
  const size_t num_trials = 1000;
  const Vector3<double> L{100.0, 100.0, 100.0};
  const AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  const AABB<double> overlapping = translate(box, Vector3<double>{50.0, 50.0, 50.0});
  OrthorhombicMetric<AXIS_XYZ, double> m{L};

  size_t ctr = 0;
  for (auto type : kWrapRigidTypes) {
    for (size_t t = 0; t < num_trials; ++t, ++ctr) {
      EXPECT_TRUE(test_wrap_rigid(type, box, overlapping, seed, ctr, m)) << "wrap_rigid failed for type " << type;
    }
  }
}

TEST(WrappingAPI, WrapRigid_FreeSpace_IsNoop) {
  const size_t seed = 1234;
  const size_t num_trials = 200;
  const AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  FreeSpaceMetric<double> m;

  size_t ctr = 0;
  for (auto type : kWrapRigidTypes) {
    for (size_t t = 0; t < num_trials; ++t, ++ctr) {
      EXPECT_TRUE(test_wrap_rigid(type, box, box, seed, ctr, m)) << "FreeSpace wrap_rigid failed for type " << type;
    }
  }
}

TEST(WrappingAPI, WrapPoints_Orthorhombic) {
  const size_t seed = 1234;
  const size_t num_trials = 1000;
  const Vector3<double> L{100.0, 100.0, 100.0};
  const AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  const AABB<double> disjoint{900.0, 900.0, 900.0, 1000.0, 1000.0, 1000.0};
  OrthorhombicMetric<AXIS_XYZ, double> m{L};

  size_t ctr = 0;
  for (auto type : kWrappableTypes) {
    for (size_t t = 0; t < num_trials; ++t, ++ctr) {
      EXPECT_TRUE(test_wrap_points(type, box, disjoint, seed, ctr, m)) << "wrap_points failed for type " << type;
    }
  }
}

TEST(WrappingAPI, WrapPoints_FreeSpace_IsNoop) {
  const size_t seed = 1234;
  const size_t num_trials = 200;
  const AABB<double> disjoint{900.0, 900.0, 900.0, 1000.0, 1000.0, 1000.0};
  FreeSpaceMetric<double> m;

  size_t ctr = 0;
  for (auto type : kWrappableTypes) {
    for (size_t t = 0; t < num_trials; ++t, ++ctr) {
      EXPECT_TRUE(test_wrap_points(type, disjoint, disjoint, seed, ctr, m))
          << "FreeSpace wrap_points failed for type " << type;
    }
  }
}

TEST(WrappingAPI, UnwrapPointsToRef_Orthorhombic) {
  const size_t seed = 1234;
  const size_t num_trials = 100;
  const Vector3<double> L{100.0, 100.0, 100.0};
  const AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  const AABB<double> disjoint{900.0, 900.0, 900.0, 1000.0, 1000.0, 1000.0};
  OrthorhombicMetric<AXIS_XYZ, double> m{L};

  size_t ctr = 0;
  for (auto type : kWrappableTypes) {
    for (size_t t = 0; t < num_trials; ++t, ++ctr) {
      EXPECT_TRUE(test_unwrap_to_ref(type, box, disjoint, seed, ctr, m))
          << "unwrap_points_to_ref failed for type " << type;
    }
  }
}

TEST(WrappingAPI, UnwrapPointsToRef_FreeSpace_IsNoop) {
  const size_t seed = 1234;
  const size_t num_trials = 100;
  const AABB<double> disjoint{900.0, 900.0, 900.0, 1000.0, 1000.0, 1000.0};
  FreeSpaceMetric<double> m;

  size_t ctr = 0;
  for (auto type : kWrappableTypes) {
    for (size_t t = 0; t < num_trials; ++t, ++ctr) {
      EXPECT_TRUE(test_unwrap_to_ref(type, disjoint, disjoint, seed, ctr, m))
          << "FreeSpace unwrap_points_to_ref failed for type " << type;
    }
  }
}

TEST(WrappingAPI, ShiftImage_Orthorhombic) {
  const size_t seed = 1234;
  const Vector3<double> L{100.0, 100.0, 100.0};
  const AABB<double> box{0.0, 0.0, 0.0, 100.0, 100.0, 100.0};
  OrthorhombicMetric<AXIS_XYZ, double> m{L};
  FreeSpaceMetric<double> free_m;

  size_t ctr = 0;
  for (auto type : kWrapRigidTypes) {
    openrand::Philox rng = make_philox(seed, ctr);
    Vector3<int> lv{rng.uniform<int>(-10, 10), rng.uniform<int>(-10, 10), rng.uniform<int>(-10, 10)};
    EXPECT_TRUE(test_shift_image(type, box, lv, seed, ctr, m)) << "shift_image (orthorhombic) failed for type " << type;
    EXPECT_TRUE(test_shift_image(type, box, lv, seed, ctr, free_m))
        << "shift_image (free space) failed for type " << type;
    ++ctr;
  }
}

// Verify that wrap_points on a spanning line segment wraps each endpoint independently.
static LineSegment<double> make_spanning_segment(const AABB<double>& box1, const AABB<double>& box2,
                                                 openrand::Philox& rng, double max_len) {
  while (true) {
    Point<double> p1 = generate_random_point<double>(box1, rng);
    Point<double> p2 = generate_random_point<double>(box2, rng);
    if (norm(p1 - p2) <= max_len) return LineSegment<double>(p1, p2);
  }
}

TEST(WrappingAPI, WrapPoints_SpanningSegment) {
  const size_t seed = 1234;
  const size_t n = 1000;
  const double W = 100.0;
  const AABB<double> box{0.0, 0.0, 0.0, W, W, W};
  const AABB<double> adj = translate(box, Vector3<double>{W, W, W});
  OrthorhombicMetric<AXIS_XYZ, double> m{Vector3<double>{W, W, W}};

  for (size_t t = 0; t < n; ++t) {
    openrand::Philox rng = make_philox(seed, t);
    auto seg = make_spanning_segment(box, adj, rng, 0.5 * W);

    auto expected_start = wrap_points(seg.start(), m);
    auto expected_end = wrap_points(seg.end(), m);
    ASSERT_GT(norm(expected_start - expected_end), norm(seg.start() - seg.end()))
        << "Test setup: wrapping should lengthen the spanning segment.";

    auto wrapped = wrap_points(seg, m);
    EXPECT_NEAR(norm(wrapped.start() - expected_start), 0.0, get_relaxed_zero_tolerance<double>());
    EXPECT_NEAR(norm(wrapped.end() - expected_end), 0.0, get_relaxed_zero_tolerance<double>());
  }
}

TEST(WrappingAPI, UnwrapPoints_SpanningSegmentInverse) {
  const size_t seed = 1234;
  const size_t n = 1000;
  const double W = 100.0;
  const AABB<double> box{0.0, 0.0, 0.0, W, W, W};
  const AABB<double> adj = translate(box, Vector3<double>{W, W, W});
  OrthorhombicMetric<AXIS_XYZ, double> m{Vector3<double>{W, W, W}};

  for (size_t t = 0; t < n; ++t) {
    openrand::Philox rng = make_philox(seed, t);
    auto seg = make_spanning_segment(box, adj, rng, 0.5 * W);

    double orig_len = norm(seg.start() - seg.end());
    auto wrapped = wrap_points(seg, m);
    double wrapped_len = norm(wrapped.start() - wrapped.end());
    ASSERT_GT(wrapped_len, orig_len) << "Test setup: wrapping should lengthen segment.";

    auto restored = unwrap_points_to_ref(wrapped, m, seg.start());
    EXPECT_NEAR(norm(restored.start() - seg.start()), 0.0, get_relaxed_zero_tolerance<double>());
    EXPECT_NEAR(norm(restored.end() - seg.end()), 0.0, get_relaxed_zero_tolerance<double>());
  }
}

// Partial-periodicity: only X is periodic; Y/Z of wrapped points must be unchanged.
TEST(WrappingAPI, PartialPeriodicity_WrapPoints_X) {
  const double Lx = 10.0;
  const AABB<double> primary{0.0, -1e9, -1e9, Lx, 1e9, 1e9};
  const AABB<double> source{Lx, 0.0, 0.0, 2.0 * Lx, 10.0, 10.0};
  OrthorhombicMetric<AXIS_X, double> m{Vector3<double>{Lx, 1.0, 1.0}};

  const size_t seed = 9999;
  const size_t n = 500;
  for (size_t t = 0; t < n; ++t) {
    openrand::Philox rng = make_philox(seed, t);
    Point<double> p = generate_random_point<double>(source, rng);
    auto wp = wrap_points(p, m);
    // X wraps into [0, Lx)
    EXPECT_GE(wp[0], 0.0);
    EXPECT_LT(wp[0], Lx);
    // Y and Z are unchanged
    EXPECT_NEAR(wp[1], p[1], 1e-12);
    EXPECT_NEAR(wp[2], p[2], 1e-12);
  }
}

// Verify wrap_rigid and unwrap_points_to_ref work through the runtime metric::visit().
TEST(WrappingAPI, RuntimeMetricVisitWrapRigid) {
  const Vector3<double> L{10.0, 10.0, 10.0};
  const AABB<double> box{0.0, 0.0, 0.0, 10.0, 10.0, 10.0};
  const AABB<double> overlapping = translate(box, Vector3<double>{5.0, 5.0, 5.0});

  metric m(metric::ORTHORHOMBIC_XYZ);
  m.set_cell_widths(L);

  const size_t seed = 42;
  const size_t n = 100;
  for (size_t t = 0; t < n; ++t) {
    openrand::Philox rng = make_philox(seed, t);
    Point<double> p = generate_random_point<double>(overlapping, rng);
    m.visit([&](auto&& concrete) {
      auto wp = wrap_rigid(p, concrete);
      EXPECT_TRUE(is_point_in_box(wp, box)) << "Runtime metric visit: point not in primary box.";
    });
  }
}

//@}

}  // namespace

}  // namespace mundy
