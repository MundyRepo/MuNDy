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
#include <mundy_geom/periodicity.hpp>  // for mundy::PeriodicScaledMetric, mundy::EuclideanMetric,
                                       //     mundy::wrap_rigid, mundy::wrap_points,
                                       //     mundy::unwrap_points_to_ref, mundy::shift_image
#include <cmath>                       // for std::abs
#include <iostream>
#include <mundy_geom/primitives.hpp>  // for mundy::Point, mundy::Sphere, mundy::LineSegment
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3d, mundy::Vector3

//---------------------------------------------------------------------------------------------------------------------//
// Metrics and minimum-image separation.
//---------------------------------------------------------------------------------------------------------------------//
void metric_and_sep_example() {
  std::cout << "\n--- Metric and Minimum-Image Separation ---\n" << std::endl;

  /*
    In a periodic simulation the "distance" between two points is ambiguous:
    a particle near one edge of the box and a particle near the opposite
    edge might be very close to each other through the periodic boundary even
    though their raw coordinates are nearly as far apart as the box is wide.

    A metric defines what "nearby" means.  It encapsulates the answer to the
    question: "given two points, what is the shortest vector from one to the
    other, accounting for periodicity?"

    mundy::PeriodicScaledMetric<Scalar>
      An orthorhombic (axis-aligned) periodic box with independent side
      lengths Lx, Ly, Lz.  Constructed from a Vector3 of side lengths.

    mundy::EuclideanMetric<Scalar>
      No periodicity; sep(x, y) = y - x.  A useful baseline when you want
      the same code path in periodic and non-periodic modes.

    The key operation is:
      metric.sep(x, y)  -- minimum-image separation from x to y.
                           The returned vector has components in (-L/2, L/2].
  */

  const double Lx = 10.0, Ly = 10.0, Lz = 10.0;
  mundy::PeriodicScaledMetric<double> metric{mundy::Vector3d{Lx, Ly, Lz}};

  // Two points near opposite edges of the box.
  mundy::Point<double> x{9.8, 0.0, 0.0};
  mundy::Point<double> y{0.2, 0.0, 0.0};

  // The raw (non-periodic) vector from x to y.
  auto raw_sep = y - x;
  std::cout << "raw separation = [" << raw_sep[0] << ", " << raw_sep[1] << ", " << raw_sep[2] << "]"
            << "  (magnitude " << mundy::norm(raw_sep) << ")" << std::endl;

  // The minimum-image separation correctly identifies these points as close.
  auto min_image_sep = metric.sep(x, y);
  std::cout << "min-image sep  = [" << min_image_sep[0] << ", " << min_image_sep[1] << ", " << min_image_sep[2] << "]"
            << "  (magnitude " << mundy::norm(min_image_sep) << ", should be ~0.4)" << std::endl;

  // EuclideanMetric -- same interface, no wrapping.
  mundy::EuclideanMetric<double> euclidean;
  auto eucl_sep = euclidean.sep(x, y);
  std::cout << "euclidean sep  = [" << eucl_sep[0] << ", " << eucl_sep[1] << ", " << eucl_sep[2] << "]"
            << "  (same as raw)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Wrapping points into the primary cell.
//---------------------------------------------------------------------------------------------------------------------//
void wrap_example() {
  std::cout << "\n--- Wrapping Points ---\n" << std::endl;

  /*
    metric.wrap(x)  brings a point into the primary cell [0, L).

    A particle that drifts outside the box through Brownian or deterministic
    motion is wrapped back in at each time step so that coordinates stay
    bounded and the neighbor list remains valid.

    Wrapping is always applied to the *reference point* of a shape.  For
    simple shapes like spheres the reference point is the center.  For
    multi-point shapes like line segments the reference point is the start.

    See the reference_point table in the Geom primer for the full list.
  */

  const double L = 10.0;
  mundy::PeriodicScaledMetric<double> metric{mundy::Vector3d{L, L, L}};

  mundy::Point<double> outside{-0.3, 11.5, 5.0};
  auto inside = metric.wrap(outside);

  std::cout << "wrap(-0.3, 11.5, 5.0) = [" << inside[0] << ", " << inside[1] << ", " << inside[2] << "]"
            << "  (should be [9.7, 1.5, 5.0])" << std::endl;

  // Wrapping a sphere moves its center; its radius is unchanged.
  mundy::Sphere<double> escaped_sphere{mundy::Point<double>{-0.3, 11.5, 5.0}, 0.5};
  auto wrapped_sphere = mundy::wrap_rigid(escaped_sphere, metric);
  std::cout << "wrapped sphere center = [" << wrapped_sphere.center()[0] << ", " << wrapped_sphere.center()[1] << ", "
            << wrapped_sphere.center()[2] << "]  radius unchanged = " << wrapped_sphere.radius() << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Rigid wrapping of multi-point shapes.
//---------------------------------------------------------------------------------------------------------------------//
void wrap_rigid_example() {
  std::cout << "\n--- Rigid Wrapping of Multi-Point Shapes ---\n" << std::endl;

  /*
    For a shape with more than one stored point (e.g. a LineSegment), there
    are two wrapping strategies:

      wrap_rigid(shape, metric)
        Translates the whole shape as one rigid unit.  The reference point
        (start of a segment) is wrapped first; all other points move by the
        same translation vector.  The shape's geometry is preserved: the
        vector from start to end is unchanged.

        Use this when the shape represents a physical body that must stay
        coherent.  A fiber segment crossing a periodic boundary can be
        understood as one piece, not as two halves on opposite sides.

      wrap_points(shape, metric)
        Wraps each stored point independently.  After this call each point
        lies in [0, L), but the shape may look "broken": for a segment near
        a periodic boundary the start and end can end up on opposite sides.

        This is useful only if you plan to immediately follow it with
        unwrap_points_to_ref to re-cohere the shape near a chosen reference,
        or if you are doing something exotic that requires each point to be
        independently in the primary cell.
  */

  const double L = 10.0;
  mundy::PeriodicScaledMetric<double> metric{mundy::Vector3d{L, L, L}};

  // A segment whose start and end are both outside the box.
  // wrap_rigid uses start as the reference point: start wraps to [0, L),
  // and end shifts by the same integer-lattice translation.
  mundy::LineSegment<double> seg{mundy::Point<double>{10.5, 0.0, 0.0}, mundy::Point<double>{11.5, 0.0, 0.0}};

  auto rigid = mundy::wrap_rigid(seg, metric);
  std::cout << "wrap_rigid: start=[" << rigid.start()[0] << "] end=[" << rigid.end()[0] << "]"
            << "  (start wraps to 0.5, end follows: 1.5)" << std::endl;

  // Compare with wrap_points: each point wraps independently.
  // The segment geometry (end - start) is preserved by wrap_rigid but not by wrap_points.
  auto pointwise = mundy::wrap_points(seg, metric);
  std::cout << "wrap_points: start=[" << pointwise.start()[0] << "] end=[" << pointwise.end()[0] << "]"
            << "  (both independently in [0,10): 0.5 and 1.5)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Unwrapping points to restore coherence near a reference.
//---------------------------------------------------------------------------------------------------------------------//
void unwrap_example() {
  std::cout << "\n--- Unwrap Points to Reference ---\n" << std::endl;

  /*
    unwrap_points_to_ref(shape, metric, ref)  is the complement of
    wrap_points.  Given a shape whose stored points may be independently
    wrapped (so the segment looks "broken"), it shifts each point by an
    integer lattice vector to place it as close as possible to ref.

    A common pattern when loading particle data from file or after a
    point-by-point wrap:
      1. Call wrap_points to ensure all coordinates are in [0, L).
      2. Call unwrap_points_to_ref(shape, metric, start()) to re-cohere
         the shape so that all points are near the start point.

    After step 2 the segment is geometrically coherent again.
  */

  const double L = 10.0;
  mundy::PeriodicScaledMetric<double> metric{mundy::Vector3d{L, L, L}};

  // A segment that was independently wrapped: start near 9.5, end near 0.5 --
  // they should be connected through the periodic boundary.
  mundy::LineSegment<double> broken{mundy::Point<double>{9.5, 0.0, 0.0}, mundy::Point<double>{0.5, 0.0, 0.0}};

  auto coherent = mundy::unwrap_points_to_ref(broken, metric, broken.start());
  std::cout << "broken  segment: start=" << broken.start()[0] << "  end=" << broken.end()[0] << std::endl;
  std::cout << "coherent segment: start=" << coherent.start()[0] << "  end=" << coherent.end()[0]
            << "  (end moved to 10.5, adjacent to start=9.5)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Explicit image shift.
//---------------------------------------------------------------------------------------------------------------------//
void shift_image_example() {
  std::cout << "\n--- Shift Image ---\n" << std::endl;

  /*
    shift_image(shape, lattice_vector, metric) moves the shape to a specific
    periodic image.  The lattice_vector is an integer triple (nx, ny, nz)
    indicating how many box lengths to shift in each direction.

    This is used in algorithms that enumerate neighbors across periodic
    boundaries, where you need the coordinates of a particle's image in an
    adjacent box without modifying the primary copy.
  */

  const double L = 10.0;
  mundy::PeriodicScaledMetric<double> metric{mundy::Vector3d{L, L, L}};

  mundy::Sphere<double> s{mundy::Point<double>{1.0, 2.0, 3.0}, 0.5};

  // Shift one box length in the +x direction.
  auto image_px = mundy::shift_image(s, mundy::Vector3<int>{1, 0, 0}, metric);
  std::cout << "primary image:  center.x = " << s.center()[0] << std::endl;
  std::cout << "+x image:       center.x = " << image_px.center()[0] << "  (should be 11.0)" << std::endl;

  // The primary copy is unmodified.
  std::cout << "primary still:  center.x = " << s.center()[0] << "  (unchanged)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Fractional coordinates.
//---------------------------------------------------------------------------------------------------------------------//
void fractional_coord_example() {
  std::cout << "\n--- Fractional Coordinates ---\n" << std::endl;

  /*
    metric.to_fractional(x)   converts a point from real (lab) coordinates
                               to fractional coordinates in [0, 1).
    metric.from_fractional(f) converts back from fractional to real.

    Fractional coordinates are the natural language of spatial hashing,
    many neighbor-list algorithms, and lattice-based data structures.  A
    particle at fractional (0.5, 0.5, 0.5) is always at the box center
    regardless of the box dimensions, which makes code that spans different
    box sizes easier to read.

    For an orthorhombic box with side lengths L = (Lx, Ly, Lz),
    to_fractional simply divides each component: f_i = x_i / L_i.
    from_fractional does the reverse.
  */

  const double Lx = 20.0, Ly = 10.0, Lz = 5.0;
  mundy::PeriodicScaledMetric<double> metric{mundy::Vector3d{Lx, Ly, Lz}};

  mundy::Point<double> p{10.0, 7.5, 2.5};

  auto frac = metric.to_fractional(p);
  std::cout << "real coords:        [" << p[0] << ", " << p[1] << ", " << p[2] << "]" << std::endl;
  std::cout << "fractional coords:  [" << frac[0] << ", " << frac[1] << ", " << frac[2] << "]"
            << "  (should be [0.5, 0.75, 0.5])" << std::endl;

  auto back = metric.from_fractional(frac);
  std::cout << "round-trip:         [" << back[0] << ", " << back[1] << ", " << back[2] << "]"
            << "  (should match original)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  metric_and_sep_example();
  wrap_example();
  wrap_rigid_example();
  unwrap_example();
  shift_image_example();
  fractional_coord_example();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
