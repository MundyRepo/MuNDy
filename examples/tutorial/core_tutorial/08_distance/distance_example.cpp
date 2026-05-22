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

#include <Kokkos_Core.hpp>
#include <iostream>
#include <mundy_geom/compute_aabb.hpp>             // for mundy::compute_aabb
#include <mundy_geom/compute_bounding_radius.hpp>  // for mundy::compute_bounding_radius
#include <mundy_geom/distance.hpp>                 // for mundy::distance
#include <mundy_geom/primitives.hpp>               // for mundy::Point, mundy::Sphere, mundy::LineSegment, mundy::AABB
#include <mundy_math/Quaternion.hpp>               // for mundy::Quaterniond
#include <mundy_math/Vector3.hpp>                  // for mundy::Vector3d

//---------------------------------------------------------------------------------------------------------------------//
// Signed distance convention.
//---------------------------------------------------------------------------------------------------------------------//
void signedDistanceExample() {
  std::cout << "\n--- Signed Distance Convention ---\n" << std::endl;

  /*
    mundy::distance(shape1, shape2) is the core query in Mundy.  For most
    surface-bearing shape pairs it returns a *signed* separation distance:

      d > 0  -- the shapes are separated; d is the gap between surfaces.
      d = 0  -- the shapes are touching (surface contact).
      d < 0  -- the shapes overlap; |d| is the penetration depth.

    This convention is physically meaningful for soft-body simulations where
    forces activate when d < 0 (overlap) and vanish when d > 0 (separation).
    It is more informative than unsigned distance because you can tell at a
    glance whether two particles interpenetrate.

    For point-to-point and point-to-line distances the "signed" concept does
    not apply; those functions return the unsigned Euclidean distance.
  */

  // Two touching spheres: centers 1.5 apart, radii 0.5 and 1.0.
  mundy::Sphere<double> a{mundy::Point<double>{0.0, 0.0, 0.0}, 0.5};
  mundy::Sphere<double> b{mundy::Point<double>{1.5, 0.0, 0.0}, 1.0};

  double gap = mundy::distance(a, b);
  std::cout << "sphere-sphere distance (touching) = " << gap << "  (should be 0)" << std::endl;

  // Separate the centers: a has radius 0.5 at origin, c has radius 0.5 at x=3.
  // Surface-to-surface gap = 3.0 - 0.5 - 0.5 = 2.0.
  mundy::Sphere<double> c{mundy::Point<double>{3.0, 0.0, 0.0}, 0.5};
  std::cout << "sphere-sphere distance (separate) = " << mundy::distance(a, c)
            << "  (should be 2.0: center-to-center=3.0, radii=0.5+0.5)" << std::endl;

  // Overlapping spheres: center distance 0.5, radii 0.5 each -> penetration 0.5.
  mundy::Sphere<double> d{mundy::Point<double>{0.5, 0.0, 0.0}, 0.5};
  double overlap = mundy::distance(a, d);
  std::cout << "sphere-sphere distance (overlap)  = " << overlap << "  (should be -0.5)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Point and segment distances.
//---------------------------------------------------------------------------------------------------------------------//
void pointAndSegmentExample() {
  std::cout << "\n--- Point and Segment Distances ---\n" << std::endl;

  /*
    Point-to-point distance is unsigned Euclidean:
      distance(p, q) = norm(q - p)

    Point-to-line-segment distance:
      distance(p, segment) = shortest Euclidean distance from p to the
                             segment (clamped to the endpoints).

    The witness overload provides additional geometric information:
      distance(p, segment, closest, u, sep)
        closest -- the closest point on the segment to p
        u       -- the parametric coordinate in [0, 1] (0 = start, 1 = end)
        sep     -- the vector from p to closest

    Witness overloads are useful when building contact forces: you need
    the contact point and the outward normal, not just the scalar distance.
  */

  mundy::Point<double> p{0.5, 1.0, 0.0};
  mundy::LineSegment<double> seg{mundy::Point<double>{0.0, 0.0, 0.0}, mundy::Point<double>{1.0, 0.0, 0.0}};

  double d = mundy::distance(p, seg);
  std::cout << "distance(p, seg) = " << d << "  (should be 1.0)" << std::endl;

  // Witness overload.
  mundy::Point<double> closest;
  double u;
  mundy::Vector3d sep;
  mundy::distance(p, seg, closest, u, sep);

  std::cout << "closest point on seg = [" << closest[0] << ", " << closest[1] << ", " << closest[2] << "]"
            << "  (should be [0.5, 0, 0])" << std::endl;
  std::cout << "parametric coord u   = " << u << "  (should be 0.5)" << std::endl;
  std::cout << "sep (p -> closest)   = [" << sep[0] << ", " << sep[1] << ", " << sep[2] << "]"
            << "  (should be [0, -1, 0])" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Sphere-sphere witness overload.
//---------------------------------------------------------------------------------------------------------------------//
void sphereSphereWitnessExample() {
  std::cout << "\n--- Sphere-Sphere with Witness ---\n" << std::endl;

  /*
    The witness overload for sphere-sphere distance:
      distance(s0, s1, sep)
        sep -- the vector from the center of s0 toward the center of s1
               (unit vector scaled to the center-to-center distance).
               The outward contact normal for s0 is sep / norm(sep).

    This is the information you need to build a Hertz contact force or a
    simple harmonic repulsion: the gap d and the contact normal sep/|sep|.
  */

  mundy::Sphere<double> s0{mundy::Point<double>{0.0, 0.0, 0.0}, 0.5};
  mundy::Sphere<double> s1{mundy::Point<double>{0.8, 0.0, 0.0}, 0.5};

  mundy::Vector3d sep;
  double d = mundy::distance(s0, s1, sep);

  std::cout << "gap d              = " << d << "  (should be -0.2: 0.8-0.5-0.5)" << std::endl;
  std::cout << "sep (s0 -> s1)     = [" << sep[0] << ", " << sep[1] << ", " << sep[2] << "]" << std::endl;

  // Contact normal pointing away from s0 is just sep normalized.
  auto normal = sep / mundy::norm(sep);
  std::cout << "contact normal     = [" << normal[0] << ", " << normal[1] << ", " << normal[2] << "]"
            << "  (should be [1, 0, 0])" << std::endl;

  // Pattern used in repulsion forces:
  if (d < 0.0) {
    double penetration = -d;
    std::cout << "overlap detected: penetration = " << penetration << "  -> apply repulsion force along normal"
              << std::endl;
  }
}

//---------------------------------------------------------------------------------------------------------------------//
// Broad-phase filtering with bounding geometry.
//---------------------------------------------------------------------------------------------------------------------//
void broadPhaseExample() {
  std::cout << "\n--- Broad-Phase Filtering ---\n" << std::endl;

  /*
    Exact distance queries are expensive.  For ellipsoids and spherocylinders
    the computation involves iterative solvers.  In a typical simulation with
    thousands or millions of particles, you cannot afford to run the exact
    query on every pair.

    The standard approach is a two-phase strategy:
      1. Broad phase: cheap conservative test that quickly eliminates pairs
         that are definitely not close enough to interact.  False positives
         are allowed; false negatives are not.
      2. Narrow phase: exact distance query only on the surviving pairs.

    Mundy provides:
      compute_aabb(shape)  -- axis-aligned bounding box
      compute_bounding_radius(shape) -- conservative enclosing sphere radius
      intersects(aabb0, aabb1) -- true if two AABBs overlap or touch

    The idiom for a broad-phase filter:
      auto box_a = compute_aabb(shape_a);
      auto box_b = compute_aabb(shape_b);
      if (!intersects(box_a, box_b)) continue;  // skip -- definitely no contact
      auto d = distance(shape_a, shape_b);       // only reached if boxes overlap
  */

  mundy::Sphere<double> s_a{mundy::Point<double>{0.0, 0.0, 0.0}, 1.0};
  mundy::Sphere<double> s_b{mundy::Point<double>{1.5, 0.0, 0.0}, 1.0};  // overlapping
  mundy::Sphere<double> s_c{mundy::Point<double>{5.0, 0.0, 0.0}, 1.0};  // far away

  auto box_a = mundy::compute_aabb(s_a);
  auto box_b = mundy::compute_aabb(s_b);
  auto box_c = mundy::compute_aabb(s_c);

  std::cout << "AABB of s_a: [" << box_a.min_corner()[0] << " to " << box_a.max_corner()[0] << "] x "
            << "[" << box_a.min_corner()[1] << " to " << box_a.max_corner()[1] << "]" << std::endl;

  bool ab_overlap = mundy::intersects(box_a, box_b);
  bool ac_overlap = mundy::intersects(box_a, box_c);

  std::cout << "AABB(s_a) intersects AABB(s_b)? " << (ab_overlap ? "yes" : "no")
            << "  (should be yes -- close together)" << std::endl;
  std::cout << "AABB(s_a) intersects AABB(s_c)? " << (ac_overlap ? "yes" : "no") << "  (should be no  -- far apart)"
            << std::endl;

  // For the pair that passed the broad phase, run the exact query.
  if (ab_overlap) {
    double d = mundy::distance(s_a, s_b);
    std::cout << "exact distance(s_a, s_b) = " << d << "  (negative means overlap)" << std::endl;
  }
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  signedDistanceExample();
  pointAndSegmentExample();
  sphereSphereWitnessExample();
  broadPhaseExample();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
