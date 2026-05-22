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
#include <cmath>  // for M_PI
#include <iostream>
#include <mundy_geom/primitives.hpp>  // for mundy::Point, mundy::Sphere, mundy::LineSegment, ...
#include <mundy_geom/transform.hpp>   // for mundy::translate
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaterniond
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3d, mundy::get_vector3_view

//---------------------------------------------------------------------------------------------------------------------//
// Point and Sphere example.
//---------------------------------------------------------------------------------------------------------------------//
void pointAndSphereExample() {
  std::cout << "\n--- Point and Sphere ---\n" << std::endl;

  /*
    Mundy's geometric primitives are light value types.  They store the
    minimal data that defines a shape; they do not allocate heap memory,
    own mesh entities, or hide the underlying math.

    mundy::Point<Scalar>
      Stores {x, y, z}.  Behaves identically to Vector3<Scalar> -- all
      vector arithmetic applies.  The separate name makes code self-
      documenting: "this is a position, not a direction."

    mundy::Sphere<Scalar>
      Stores {center: Point<Scalar>, radius: Scalar}.
      Named accessors: center(), radius(), set_center(...), set_radius(r).

    By convention the Scalar template parameter defaults to double, so
    Point<double> may be written as Point<double> but there is no additional
    shorthand alias beyond that.
  */

  mundy::Point<double> origin{0.0, 0.0, 0.0};
  mundy::Point<double> p{3.0, 4.0, 0.0};

  // Point arithmetic is full Vector3 arithmetic.
  auto displacement = p - origin;
  std::cout << "displacement = [" << displacement[0] << ", " << displacement[1] << ", " << displacement[2] << "]"
            << std::endl;

  // Construction and named accessors.
  mundy::Sphere<double> ball{p, 0.5};
  std::cout << "ball center = [" << ball.center()[0] << ", " << ball.center()[1] << ", " << ball.center()[2] << "]"
            << std::endl;
  std::cout << "ball radius = " << ball.radius() << std::endl;

  // Mutation.
  ball.set_radius(1.0);
  ball.center()[0] = 5.0;
  std::cout << "after mutation: radius=" << ball.radius() << "  center.x=" << ball.center()[0] << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// LineSegment example.
//---------------------------------------------------------------------------------------------------------------------//
void lineSegmentExample() {
  std::cout << "\n--- LineSegment ---\n" << std::endl;

  /*
    mundy::LineSegment<Scalar>
      Stores {start: Point<Scalar>, end: Point<Scalar>}.
      Named accessors: start(), end().

    A segment is the most common primitive for modeling stiff fibers,
    cross-links, and bond pairs.  The tangent and length are not stored
    explicitly; compute them from start() and end() as needed.
  */

  mundy::Point<double> a{0.0, 0.0, 0.0};
  mundy::Point<double> b{3.0, 0.0, 0.0};

  mundy::LineSegment<double> seg{a, b};

  auto tangent = seg.end() - seg.start();
  double length = mundy::norm(tangent);
  auto unit_tangent = tangent / length;

  std::cout << "segment length   = " << length << std::endl;
  std::cout << "unit tangent     = [" << unit_tangent[0] << ", " << unit_tangent[1] << ", " << unit_tangent[2] << "]"
            << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Oriented shapes example.
//---------------------------------------------------------------------------------------------------------------------//
void orientedShapesExample() {
  std::cout << "\n--- Oriented Shapes (Spherocylinder) ---\n" << std::endl;

  /*
    Oriented shapes store a Quaternion<Scalar> that maps the body's local
    frame to the lab frame.  The body z-axis is the conventional long axis
    (for rods and capsules) or normal (for circular primitives).

    mundy::Spherocylinder<Scalar>
      Stores {center: Point<Scalar>, orientation: Quaternion<Scalar>,
              radius: Scalar, length: Scalar}.

      The rod's long axis in lab space is orientation * e_z where
      e_z = {0, 0, 1}.

      Named accessors: center(), orientation(), radius(), length(), and
      their corresponding setters.

    The identity quaternion means the rod is aligned with the lab z-axis.
  */

  mundy::Point<double> center{0.0, 0.0, 0.0};
  mundy::Quaterniond q = mundy::Quaterniond::identity();
  double radius = 0.1;
  double length = 2.0;

  mundy::Spherocylinder<double> rod{center, q, radius, length};

  std::cout << "rod radius = " << rod.radius() << std::endl;
  std::cout << "rod length = " << rod.length() << std::endl;

  // The body z-axis direction in lab space.
  mundy::Vector3d e_z{0.0, 0.0, 1.0};
  auto rod_axis = rod.orientation() * e_z;
  std::cout << "rod axis   = [" << rod_axis[0] << ", " << rod_axis[1] << ", " << rod_axis[2] << "]"
            << "  (identity -> aligned with lab z)" << std::endl;

  // Tilt the rod 90 degrees about x.
  auto tilt = mundy::axis_angle_to_quaternion(mundy::Vector3d{1.0, 0.0, 0.0}, M_PI / 2.0);
  rod.set_orientation(tilt);
  auto tilted_axis = rod.orientation() * e_z;
  std::cout << "tilted axis = [" << tilted_axis[0] << ", " << tilted_axis[1] << ", " << tilted_axis[2] << "]"
            << "  (90 deg about x -> axis points in y direction)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Ellipsoid example.
//---------------------------------------------------------------------------------------------------------------------//
void ellipsoidExample() {
  std::cout << "\n--- Ellipsoid ---\n" << std::endl;

  /*
    mundy::Ellipsoid<Scalar>
      Stores {center: Point<Scalar>, orientation: Quaternion<Scalar>,
              radii: Point<Scalar>}.

      radii stores the three semi-axis lengths (a, b, c) in the body frame:
        radii[0] -- semi-axis along the body x-axis
        radii[1] -- semi-axis along the body y-axis
        radii[2] -- semi-axis along the body z-axis (body z is the natural
                    long axis for prolate shapes, same convention as
                    Spherocylinder)

      A sphere is the special case radii[0] == radii[1] == radii[2].
      A prolate spheroid (rod-like, typical bacterium model) has
        radii[2] > radii[0] == radii[1].
      An oblate spheroid (disc-like, red blood cell model) has
        radii[2] < radii[0] == radii[1].

      Named accessors: center(), orientation(), radii(), and their setters.
  */

  mundy::Point<double> center{0.0, 0.0, 0.0};

  // Prolate spheroid: long axis (body z) has semi-length 3, short axes 1.
  mundy::Ellipsoid<double> rod_like{center, mundy::Quaterniond::identity(), mundy::Point<double>{1.0, 1.0, 3.0}};

  std::cout << "semi-axes (a, b, c) = [" << rod_like.radii()[0] << ", " << rod_like.radii()[1] << ", "
            << rod_like.radii()[2] << "]" << std::endl;

  // The long-axis direction in lab space: body z rotated by the orientation.
  mundy::Vector3d e_z{0.0, 0.0, 1.0};
  auto long_axis = rod_like.orientation() * e_z;
  std::cout << "long-axis direction = [" << long_axis[0] << ", " << long_axis[1] << ", " << long_axis[2] << "]"
            << "  (identity -> lab z)" << std::endl;

  // Tilt the ellipsoid 90 degrees about x so the long axis points in the y direction.
  auto tilt = mundy::axis_angle_to_quaternion(mundy::Vector3d{1.0, 0.0, 0.0}, M_PI / 2.0);
  rod_like.set_orientation(tilt);
  auto tilted_axis = rod_like.orientation() * e_z;
  std::cout << "tilted long-axis    = [" << tilted_axis[0] << ", " << tilted_axis[1] << ", " << tilted_axis[2] << "]"
            << "  (90 deg about x -> points in y)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// AABB example.
//---------------------------------------------------------------------------------------------------------------------//
void aabbExample() {
  std::cout << "\n--- AABB ---\n" << std::endl;

  /*
    mundy::AABB<Scalar>
      Stores {min_corner: Point<Scalar>, max_corner: Point<Scalar>}.

      The named accessors min_corner() and max_corner() return the bounding
      corners.  The operator[] treats the AABB as a flat array of six
      scalars: {x_min, y_min, z_min, x_max, y_max, z_max}.

    AABBs are used almost exclusively as fast-reject bounding boxes in
    broad-phase collision detection (Tutorial 08 covers the pattern).
    Construct them explicitly when setting up spatial data structures, or
    obtain them automatically from mundy::compute_aabb(shape) for any
    supported primitive.
  */

  mundy::AABB<double> box{-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};

  std::cout << "min_corner = [" << box.min_corner()[0] << ", " << box.min_corner()[1] << ", " << box.min_corner()[2]
            << "]" << std::endl;
  std::cout << "max_corner = [" << box.max_corner()[0] << ", " << box.max_corner()[1] << ", " << box.max_corner()[2]
            << "]" << std::endl;

  // Flat index access: 0=x_min, 1=y_min, 2=z_min, 3=x_max, 4=y_max, 5=z_max.
  std::cout << "box[3] = " << box[3] << "  (x_max, should be 1)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// View-backed primitives example.
//---------------------------------------------------------------------------------------------------------------------//
void viewBackedPrimitiveExample() {
  std::cout << "\n--- View-Backed Primitives ---\n" << std::endl;

  /*
    Primitives are templated on their point and orientation types, so they
    can hold view-backed coordinates directly.  This matters when your
    particle system stores all positions in a single flat array and you want
    to work with individual shapes without copying.

    The pattern:
      1. Create views into the relevant slices of your flat array.
      2. Construct the primitive with those views as arguments.
      3. Use the primitive in any geometry query.

    Writes to the primitive's accessors modify the underlying array.
  */

  // A flat coordinate buffer for two sphere endpoints (a segment).
  double coords[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};

  auto start_view = mundy::get_vector3_view<double>(coords + 0);
  auto end_view = mundy::get_vector3_view<double>(coords + 3);

  mundy::LineSegment<double, decltype(start_view), decltype(end_view)> seg_view{start_view, end_view};

  auto tangent = seg_view.end() - seg_view.start();
  std::cout << "segment tangent = [" << tangent[0] << ", " << tangent[1] << ", " << tangent[2] << "]" << std::endl;

  // Move the endpoint -- this writes into coords[3..5].
  seg_view.end()[0] = 3.0;
  std::cout << "After moving end: coords[3]=" << coords[3] << "  (should be 3)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Transforms example.
//---------------------------------------------------------------------------------------------------------------------//
void transformsExample() {
  std::cout << "\n--- Transforms ---\n" << std::endl;

  /*
    Mundy provides translate and rotate as free functions that follow a
    simple convention:

      translate(shape, displacement)  -- returns a translated copy
      translate_inplace(shape, disp)  -- mutates shape in place
      rotate(shape, q)                -- returns a rotated copy
      rotate_inplace(shape, q)        -- mutates shape in place

    Rotation is defined for all primitives except AABB.  Rotating an AABB
    is intentionally not supported: an oriented box and its enclosing
    axis-aligned box are different things and the right choice depends on
    the context.
  */

  mundy::Sphere<double> s{{0.0, 0.0, 0.0}, 0.5};

  // Translate by [1, 2, 3].
  auto shifted = mundy::translate(s, mundy::Vector3d{1.0, 2.0, 3.0});
  std::cout << "translated sphere center = [" << shifted.center()[0] << ", " << shifted.center()[1] << ", "
            << shifted.center()[2] << "]" << std::endl;

  // Rotate a Spherocylinder (changes its orientation quaternion).
  mundy::Spherocylinder<double> rod{{0.0, 0.0, 0.0}, mundy::Quaterniond::identity(), 0.1, 2.0};
  auto q_y = mundy::axis_angle_to_quaternion(mundy::Vector3d{0.0, 1.0, 0.0}, M_PI / 2.0);  // 90 deg about y
  auto rotated_rod = mundy::rotate(rod, q_y);

  mundy::Vector3d e_z{0.0, 0.0, 1.0};
  auto new_axis = rotated_rod.orientation() * e_z;
  std::cout << "rod axis after 90-deg y rotation = [" << new_axis[0] << ", " << new_axis[1] << ", " << new_axis[2]
            << "]"
            << "  (body z -> lab x direction)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  pointAndSphereExample();
  lineSegmentExample();
  orientedShapesExample();
  ellipsoidExample();
  aabbExample();
  viewBackedPrimitiveExample();
  transformsExample();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
