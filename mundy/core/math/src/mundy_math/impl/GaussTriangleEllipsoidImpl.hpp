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

#ifndef MUNDY_MATH_IMPL_GAUSSTRIANGLEELLIPSOIDIMPL_HPP_
#define MUNDY_MATH_IMPL_GAUSSTRIANGLEELLIPSOIDIMPL_HPP_

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// Mundy
#include <mundy_math/DoubleDouble.hpp>           // for mundy::DoubleDouble, mundy::sqrt
#include <mundy_math/impl/DoubleDoubleImpl.hpp>  // for mundy::impl::{SinCos, sin_cos_of_turn_fraction}

namespace mundy {

namespace impl {

//! \name Construction of Gauss triangle ellipsoid rules, at compile time or run time
//@{

/// \brief A point (x, y, z) in double-double.
using DoubleDouble3 = Kokkos::Array<DoubleDouble, 3>;

/// \brief A triangle's three vertices, or its three edge nodes.
using DoubleDoubleTriangle = Kokkos::Array<DoubleDouble3, 3>;

/// \brief Vertex k of BEMLIB's octahedron (trgl6_octa.f).
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble3 octahedron_vertex(unsigned k) {
  // clang-format off
  constexpr Kokkos::Array<double, 18> vertices = { 0.0,  0.0,  1.0,    1.0,  0.0,  0.0,    0.0,  1.0,  0.0,
                                                  -1.0,  0.0,  0.0,    0.0, -1.0,  0.0,    0.0,  0.0, -1.0};
  // clang-format on
  return {vertices[3 * k], vertices[3 * k + 1], vertices[3 * k + 2]};
}

/// \brief Vertex k of BEMLIB's icosahedron (trgl6_icos.f), on the unit sphere.
///
/// Vertices 0 and 11 are the poles. Vertices 1-5 lie on the ring z = 1/sqrt(5) at the longitudes (5 - 4j)/20 of a
/// turn, j = k - 1, and vertices 6-10 on the ring z = -1/sqrt(5) at (7 - 4j)/20, j = k - 6. Both rings have radius
/// 2/sqrt(5).
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble3 icosahedron_vertex(unsigned k) {
  if (k == 0 || k == 11) {
    return {0.0, 0.0, k == 0 ? 1.0 : -1.0};
  }
  const bool upper = k <= 5;
  const unsigned j = upper ? k - 1 : k - 6;
  const unsigned twentieths = (20 + (upper ? 5 : 7) - 4 * j) % 20;
  const SinCos<DoubleDouble> longitude = sin_cos_of_turn_fraction<DoubleDouble>(twentieths, 20);
  const DoubleDouble z = DoubleDouble(1.0) / sqrt(DoubleDouble(5.0));
  return {2.0 * z * longitude.cos, 2.0 * z * longitude.sin, upper ? z : -z};
}

/// \brief Face f of BEMLIB's octahedron (num_faces = 8) or icosahedron (num_faces = 20).
///
/// The faces and their vertex orders are those of trgl6_octa.f and trgl6_icos.f: counterclockwise seen from outside.
KOKKOS_INLINE_FUNCTION constexpr DoubleDoubleTriangle sphere_polyhedron_face(unsigned num_faces, unsigned f) {
  // clang-format off
  constexpr Kokkos::Array<unsigned, 24> octahedron_faces = {0, 1, 2,   3, 0, 2,   3, 4, 0,   0, 4, 1,
                                                            1, 5, 2,   5, 3, 2,   5, 4, 3,   1, 4, 5};
  constexpr Kokkos::Array<unsigned, 60> icosahedron_faces = {0, 2,  1,  0, 3,  2,  0, 4,  3,  0,  5,  4,   0, 1,  5,
                                                             1, 2,  7,  2, 3,  8,  3, 4,  9,  4,  5, 10,   5, 1,  6,
                                                             1, 7,  6,  2, 8,  7,  3, 9,  8,  4, 10,  9,   5, 6, 10,
                                                             6, 7, 11,  7, 8, 11,  8, 9, 11,  9, 10, 11,  10, 6, 11};
  // clang-format on
  DoubleDoubleTriangle face;
  for (unsigned k = 0; k < 3; ++k) {
    face[k] = num_faces == 8 ? octahedron_vertex(octahedron_faces[3 * f + k])
                             : icosahedron_vertex(icosahedron_faces[3 * f + k]);
  }
  return face;
}

/// \brief The midpoints of the edges v0 v1, v1 v2, and v2 v0 of a triangle, projected onto the unit sphere.
KOKKOS_INLINE_FUNCTION constexpr DoubleDoubleTriangle projected_edge_midpoints(const DoubleDoubleTriangle& v) {
  DoubleDoubleTriangle m;
  for (unsigned k = 0; k < 3; ++k) {
    const DoubleDouble3& p = v[k];
    const DoubleDouble3& q = v[(k + 1) % 3];
    const DoubleDouble3 sum = {p[0] + q[0], p[1] + q[1], p[2] + q[2]};
    const DoubleDouble inverse_norm = 1.0 / sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    m[k] = {sum[0] * inverse_norm, sum[1] * inverse_norm, sum[2] * inverse_norm};
  }
  return m;
}

/// \brief The vertices on the unit sphere of element e of the polyhedron with num_faces faces refined l times.
///
/// Refinement splits a triangle with vertices v and projected edge midpoints m into the corner children 0, 1, 2 and the
/// center child 3, in BEMLIB's order. Corner child c keeps v[c] and takes m[c] and m[c + 2] as its next two vertices;
/// the center child is m. Element e lies on face e / 4^l, and the base-4 digits of e mod 4^l, most significant first,
/// pick the child at each level.
KOKKOS_INLINE_FUNCTION constexpr DoubleDoubleTriangle sphere_element_vertices(unsigned num_faces, unsigned l,
                                                                              unsigned e) {
  DoubleDoubleTriangle v = sphere_polyhedron_face(num_faces, e >> (2 * l));
  for (unsigned level = l; level-- > 0;) {
    const unsigned child = (e >> (2 * level)) & 3u;
    const DoubleDoubleTriangle m = projected_edge_midpoints(v);
    if (child == 3) {
      v = m;
    } else {
      v[(child + 1) % 3] = m[child];
      v[(child + 2) % 3] = m[(child + 2) % 3];
    }
  }
  return v;
}

/// \brief |p - q|.
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble distance(const DoubleDouble3& p, const DoubleDouble3& q) {
  const DoubleDouble3 d = {p[0] - q[0], p[1] - q[1], p[2] - q[2]};
  return sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

/// \brief A six-node triangle: vertices v, edge nodes m (m[k] between v[k] and v[k + 1]), and the edge nodes'
/// placement.
///
/// ratio[i][j] = |m_ij - v_j| / |m_ij - v_i| for the edge node m_ij between v_i and v_j, 1 at a midpoint. As in
/// BEMLIB's abc.f, m_ij sits at the fraction 1 / (1 + ratio[i][j]) of its edge from v_i in the reference triangle.
struct SixNodeTriangle {
  DoubleDoubleTriangle v;
  DoubleDoubleTriangle m;
  Kokkos::Array<DoubleDouble3, 3> ratio;

  KOKKOS_INLINE_FUNCTION constexpr const DoubleDouble3& edge_node(unsigned i, unsigned j) const {
    return m[j == (i + 1) % 3 ? i : j];
  }

  /// \brief 1 / (t_ij t_ji) for the fractions t_ij of edge ij from v_i to m_ij: 4 at a midpoint.
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble edge_weight(unsigned i, unsigned j) const {
    return 2.0 + (ratio[i][j] + ratio[j][i]);
  }

  /// \brief sum_k coefficient[k] node[k] over the nodes (v_a, v_b, v_c, m_ab, m_bc, m_ca), adding the nodes that the
  /// swap b <-> c exchanges in pairs.
  KOKKOS_INLINE_FUNCTION constexpr DoubleDouble3 combine(unsigned a, unsigned b, unsigned c,
                                                         const Kokkos::Array<DoubleDouble, 6>& coefficient) const {
    const DoubleDouble3& m_ab = edge_node(a, b);
    const DoubleDouble3& m_bc = edge_node(b, c);
    const DoubleDouble3& m_ca = edge_node(c, a);
    DoubleDouble3 sum;
    for (unsigned d = 0; d < 3; ++d) {
      sum[d] = coefficient[0] * v[a][d] + (coefficient[1] * v[b][d] + coefficient[2] * v[c][d]) +
               (coefficient[3] * m_ab[d] + coefficient[5] * m_ca[d]) + coefficient[4] * m_bc[d];
    }
    return sum;
  }
};

/// \brief The six-node triangle of element e of the polyhedron with num_faces faces refined l times, stretched by
/// (x, y, z) -> (x, b_over_a y, c_over_a z).
KOKKOS_INLINE_FUNCTION constexpr SixNodeTriangle ellipsoid_element(unsigned num_faces, unsigned l, double b_over_a,
                                                                   double c_over_a, unsigned e) {
  const DoubleDoubleTriangle v = sphere_element_vertices(num_faces, l, e);
  const DoubleDoubleTriangle m = projected_edge_midpoints(v);
  const Kokkos::Array<double, 3> axes = {1.0, b_over_a, c_over_a};
  SixNodeTriangle t;
  for (unsigned k = 0; k < 3; ++k) {
    for (unsigned d = 0; d < 3; ++d) {
      t.v[k][d] = axes[d] * v[k][d];
      t.m[k][d] = axes[d] * m[k][d];
    }
  }
  for (unsigned i = 0; i < 3; ++i) {
    const unsigned j = (i + 1) % 3;
    const DoubleDouble to_v_i = distance(t.m[i], t.v[i]);
    const DoubleDouble to_v_j = distance(t.m[i], t.v[j]);
    t.ratio[i][j] = to_v_j / to_v_i;
    t.ratio[j][i] = to_v_i / to_v_j;
  }
  return t;
}

/// \brief The patch's derivative toward v_b at the Gauss point nearest v_a, (2/3, 1/6, 1/6) on (v_a, v_b, v_c).
///
/// It combines (v_a, v_b, v_c, m_ab, m_bc, m_ca) with (-8 - 3 r_ab + r_ac, 2 - 3 r_ba - r_bc, r_ca - r_cb, 3 w_ab,
/// w_bc, -w_ca)/6, in the notation of write_gauss_triangle_ellipsoid_element.
KOKKOS_INLINE_FUNCTION constexpr DoubleDouble3 gauss_point_tangent(const SixNodeTriangle& t, unsigned a, unsigned b,
                                                                   unsigned c) {
  const auto& r = t.ratio;
  return t.combine(
      a, b, c,
      {(-8.0 - 3.0 * r[a][b] + r[a][c]) / 6.0, (2.0 - 3.0 * r[b][a] - r[b][c]) / 6.0, (r[c][a] - r[c][b]) / 6.0,
       t.edge_weight(a, b) / 2.0, t.edge_weight(b, c) / 6.0, -t.edge_weight(c, a) / 6.0});
}

/// \brief Write the three points of element e of the polyhedron with num_faces faces refined l times and stretched onto
/// the ellipsoid with semi-axes 1, b_over_a, c_over_a, rounded to Scalar, into points, normals, and weights.
///
/// The element is BEMLIB's quadratic patch through its six nodes: the interpolant on the reference triangle with the
/// vertices at its corners and each edge node m_ij at the fraction 1 / (1 + r_ij) of its edge from v_i. Its shape
/// functions are, in barycentric coordinates L,
///   vertex i:   L_i (L_i - r_ij L_j - r_ik L_k),
///   edge node:  w_ij L_i L_j, with w_ij = 2 + r_ij + r_ji.
/// Point 3e + a is the Gauss point nearest v_a, at L = (2/3, 1/6, 1/6) on (v_a, v_b, v_c) with (b, c) = (a + 1, a + 2).
/// There the point combines (v_a, v_b, v_c, m_ab, m_bc, m_ca) with the coefficients
///   x:    (16 - 4 (r_ab + r_ac), 1 - 4 r_ba - r_bc, 1 - 4 r_ca - r_cb, 4 w_ab, w_bc, 4 w_ca)/36,
/// and gauss_point_tangent gives its derivatives x_b toward v_b and x_c toward v_c. With s = x_b + x_c and
/// t = x_b - x_c, x_b cross x_c = (t cross s)/2, so the normal is t cross s over its norm, and the weight is
/// |t cross s|/12: the Gauss weight 1/3 times the reference triangle's area 1/2 times the Jacobian |t cross s|/2. The
/// swap b <-> c maps x_b to x_c, s to s, and t to -t, so a point and normal on a plane of symmetry have exact zeros at
/// any precision.
template <class Scalar, class Points, class Normals, class Weights>
KOKKOS_INLINE_FUNCTION constexpr void write_gauss_triangle_ellipsoid_element(unsigned num_faces, unsigned l,
                                                                             double b_over_a, double c_over_a,
                                                                             unsigned e, Points& points,
                                                                             Normals& normals, Weights& weights) {
  const SixNodeTriangle element = ellipsoid_element(num_faces, l, b_over_a, c_over_a, e);
  const auto& r = element.ratio;
  for (unsigned a = 0; a < 3; ++a) {
    const unsigned b = (a + 1) % 3;
    const unsigned c = (a + 2) % 3;
    const DoubleDouble3 x =
        element.combine(a, b, c,
                        {(16.0 - 4.0 * (r[a][b] + r[a][c])) / 36.0, (1.0 - 4.0 * r[b][a] - r[b][c]) / 36.0,
                         (1.0 - 4.0 * r[c][a] - r[c][b]) / 36.0, 4.0 * element.edge_weight(a, b) / 36.0,
                         element.edge_weight(b, c) / 36.0, 4.0 * element.edge_weight(c, a) / 36.0});
    const DoubleDouble3 x_b = gauss_point_tangent(element, a, b, c);
    const DoubleDouble3 x_c = gauss_point_tangent(element, a, c, b);
    DoubleDouble3 s;
    DoubleDouble3 t;
    for (unsigned d = 0; d < 3; ++d) {
      s[d] = x_b[d] + x_c[d];
      t[d] = x_b[d] - x_c[d];
    }
    const DoubleDouble3 t_cross_s = {t[1] * s[2] - t[2] * s[1], t[2] * s[0] - t[0] * s[2], t[0] * s[1] - t[1] * s[0]};
    const DoubleDouble t_cross_s_norm =
        sqrt(t_cross_s[0] * t_cross_s[0] + t_cross_s[1] * t_cross_s[1] + t_cross_s[2] * t_cross_s[2]);
    const DoubleDouble inverse_norm = 1.0 / t_cross_s_norm;
    const unsigned i = 3 * e + a;
    for (unsigned d = 0; d < 3; ++d) {
      points[3 * i + d] = static_cast<Scalar>(x[d].hi());
      normals[3 * i + d] = static_cast<Scalar>((t_cross_s[d] * inverse_norm).hi());
    }
    weights[i] = static_cast<Scalar>((t_cross_s_norm / 12.0).hi());
  }
}

/// \brief The points, normals, and weights of the Gauss triangle rule on the polyhedron with NumFaces faces refined L
/// times.
template <class Scalar, unsigned NumFaces, unsigned L>
struct GaussTriangleEllipsoidRule {
  static constexpr unsigned num_points = 3 * (NumFaces << (2 * L));
  Kokkos::Array<Scalar, 3 * num_points> points;   //!< (x, y, z) triples
  Kokkos::Array<Scalar, 3 * num_points> normals;  //!< (x, y, z) triples
  Kokkos::Array<Scalar, num_points> weights;
};

/// \brief Build the Gauss triangle rule on the polyhedron with NumFaces faces refined L times and stretched onto the
/// ellipsoid with semi-axes 1, BOverA, COverA.
template <class Scalar, unsigned NumFaces, unsigned L, double BOverA, double COverA>
KOKKOS_INLINE_FUNCTION constexpr GaussTriangleEllipsoidRule<Scalar, NumFaces, L> make_gauss_triangle_ellipsoid_rule() {
  GaussTriangleEllipsoidRule<Scalar, NumFaces, L> rule{};
  for (unsigned e = 0; e < (NumFaces << (2 * L)); ++e) {
    write_gauss_triangle_ellipsoid_element<Scalar>(NumFaces, L, BOverA, COverA, e, rule.points, rule.normals,
                                                   rule.weights);
  }
  return rule;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_GAUSSTRIANGLEELLIPSOIDIMPL_HPP_
