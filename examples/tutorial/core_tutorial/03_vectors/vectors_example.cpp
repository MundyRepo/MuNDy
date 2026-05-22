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

#include <Kokkos_Core.hpp>  // for Kokkos::ScopeGuard
#include <cmath>            // for std::sqrt
#include <iostream>
#include <mundy_math/Vector.hpp>   // for mundy::Vector
#include <mundy_math/Vector3.hpp>  // for mundy::Vector3, mundy::Vector3d, mundy::Vector3f

//---------------------------------------------------------------------------------------------------------------------//
// Construction example.
//---------------------------------------------------------------------------------------------------------------------//
void construction_example() {
  std::cout << "\n--- Construction ---\n" << std::endl;

  /*
    mundy::Vector<Scalar, N> is a fixed-size mathematical vector.  It is
    templated on the element type and the number of components.  Because the
    size is known at compile time, every operation is stack-allocated and
    inlinable.

    For the most common case in 3D physics -- a double-precision 3-vector --
    Mundy provides the shorthand alias Vector3d.  Similar aliases exist for
    other sizes and scalar types:

      Vector3<Scalar>   -- N=3, arbitrary Scalar
      Vector3d          -- N=3, double
      Vector3f          -- N=3, float
      Vector2d          -- N=2, double
      Vector6d          -- N=6, double
      ...and so on up to N=6.

    Two constructors are worth knowing:
      Vector3d{x, y, z}  -- initialize each component explicitly.
      Vector3d{s}        -- broadcast: all components equal s.
  */

  mundy::Vector3d v1{1.0, 2.0, 3.0};
  mundy::Vector3d v2{0.5, 1.5, 2.5};

  // Broadcast constructor: both components become 3.3.
  mundy::Vector2d v_broadcast{3.3};
  std::cout << "broadcast v2d{3.3} = [" << v_broadcast[0] << ", " << v_broadcast[1] << "]" << std::endl;

  // Factory helpers for common initializations.
  auto zeros = mundy::Vector3d::zeros();  // [0, 0, 0]
  auto ones = mundy::Vector3d::ones();    // [1, 1, 1]
  std::cout << "zeros = [" << zeros[0] << ", " << zeros[1] << ", " << zeros[2] << "]" << std::endl;
  std::cout << "ones  = [" << ones[0] << ", " << ones[1] << ", " << ones[2] << "]" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Accessors example.
//---------------------------------------------------------------------------------------------------------------------//
void accessors_example() {
  std::cout << "\n--- Accessors ---\n" << std::endl;

  /*
    Mundy uses a deliberate two-accessor convention:

      v[i]  -- the programming accessor.  Treats the vector as a flat array
               of scalars in storage order.  Useful when you are iterating
               over components or interfacing with raw data.

      v(i)  -- the mathematical accessor.  For vectors these are identical;
               the distinction matters for matrices (Tutorial 04), where ()
               takes two indices (row, column) and [] takes a single
               flattened index.

    Using () everywhere in mathematical code and [] when you are doing
    low-level data access keeps the intent clear.
  */

  mundy::Vector3d v{1.1, 2.2, 3.3};

  std::cout << "v[0]=" << v[0] << "  v(0)=" << v(0) << "  (should be equal)" << std::endl;
  std::cout << "v[1]=" << v[1] << "  v(1)=" << v(1) << std::endl;
  std::cout << "v[2]=" << v[2] << "  v(2)=" << v(2) << std::endl;

  // Both accessors return references, so you can assign through them.
  v[2] = 99.0;
  std::cout << "After v[2]=99: v(2)=" << v(2) << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Arithmetic example.
//---------------------------------------------------------------------------------------------------------------------//
void arithmetic_example() {
  std::cout << "\n--- Arithmetic ---\n" << std::endl;

  /*
    Vector arithmetic is element-wise.  All operators work with a
    same-sized vector on the right-hand side, or a scalar that is broadcast
    to all elements.

    All binary operators (+, -, *, /) return a new vector by value.
    The compound assignment operators (+=, -=, *=, /=) mutate in place.
  */

  mundy::Vector3d a{1.0, 2.0, 3.0};
  mundy::Vector3d b{4.0, 5.0, 6.0};

  auto sum = a + b;       // [5, 7, 9]
  auto diff = b - a;      // [3, 3, 3]
  auto scaled = 2.0 * a;  // [2, 4, 6]
  auto halved = b / 2.0;  // [2, 2.5, 3]

  std::cout << "a+b   = [" << sum[0] << ", " << sum[1] << ", " << sum[2] << "]" << std::endl;
  std::cout << "b-a   = [" << diff[0] << ", " << diff[1] << ", " << diff[2] << "]" << std::endl;
  std::cout << "2*a   = [" << scaled[0] << ", " << scaled[1] << ", " << scaled[2] << "]" << std::endl;
  std::cout << "b/2   = [" << halved[0] << ", " << halved[1] << ", " << halved[2] << "]" << std::endl;

  // In-place mutation.
  mundy::Vector3d c{1.0, 1.0, 1.0};
  c += a;    // c = [2, 3, 4]
  c *= 3.0;  // c = [6, 9, 12]
  std::cout << "c (after +=a, *=3) = [" << c[0] << ", " << c[1] << ", " << c[2] << "]" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Reductions example.
//---------------------------------------------------------------------------------------------------------------------//
void reductions_example() {
  std::cout << "\n--- Reductions ---\n" << std::endl;

  /*
    Mundy provides scalar reductions over the components of a vector.
    These are free functions (not member functions) to keep the same call
    syntax across vectors, matrices, and other mathematical objects.
  */

  mundy::Vector3d v{3.0, 1.0, 4.0};

  std::cout << "v = [" << v[0] << ", " << v[1] << ", " << v[2] << "]" << std::endl;
  std::cout << "sum(v)      = " << mundy::sum(v) << std::endl;      // 8
  std::cout << "product(v)  = " << mundy::product(v) << std::endl;  // 3*1*4 = 12
  std::cout << "min(v)      = " << mundy::min(v) << std::endl;      // 1
  std::cout << "max(v)      = " << mundy::max(v) << std::endl;      // 4
  std::cout << "mean(v)     = " << mundy::mean(v) << std::endl;     // 8/3
  std::cout << "variance(v) = " << mundy::variance(v) << std::endl;
  std::cout << "stddev(v)   = " << mundy::stddev(v) << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Norms and products example.
//---------------------------------------------------------------------------------------------------------------------//
void norms_and_products_example() {
  std::cout << "\n--- Norms and Products ---\n" << std::endl;

  /*
    The norms follow standard names.  norm() is an alias for two_norm()
    (the Euclidean length), which is the most common norm in physics code.

    dot() and cross() are the usual inner and outer products.  cross() is
    only defined for 3-component vectors; attempting it on a 2-vector is a
    compile error.
  */

  mundy::Vector3d a{1.0, 0.0, 0.0};
  mundy::Vector3d b{0.0, 1.0, 0.0};

  std::cout << "a = [1, 0, 0],  b = [0, 1, 0]" << std::endl;
  std::cout << "dot(a, b)          = " << mundy::dot(a, b) << std::endl;            // 0 (orthogonal)
  std::cout << "norm(a)            = " << mundy::norm(a) << std::endl;              // 1
  std::cout << "two_norm(a)        = " << mundy::two_norm(a) << std::endl;          // 1
  std::cout << "two_norm_squared(a)= " << mundy::two_norm_squared(a) << std::endl;  // 1
  std::cout << "one_norm(a)        = " << mundy::one_norm(a) << std::endl;          // 1
  std::cout << "infinity_norm(a)   = " << mundy::infinity_norm(a) << std::endl;     // 1

  auto c = mundy::cross(a, b);  // [0, 0, 1]  (right-hand rule: x cross y = z)
  std::cout << "cross(a, b) = [" << c[0] << ", " << c[1] << ", " << c[2] << "]" << std::endl;

  // Angle between vectors.
  mundy::Vector3d u{1.0, 0.0, 0.0};
  mundy::Vector3d w{1.0, 1.0, 0.0};
  std::cout << "minor_angle(u, w)  = " << mundy::minor_angle(u, w) << " rad"
            << "  (should be pi/4 = " << M_PI / 4.0 << ")" << std::endl;

  // A typical physics use case: normalizing a separation vector.
  mundy::Vector3d x1{0.0, 0.0, 0.0};
  mundy::Vector3d x2{3.0, 4.0, 0.0};
  auto sep = x2 - x1;
  double dist = mundy::norm(sep);
  auto unit_sep = sep / dist;  // unit vector from x1 toward x2
  std::cout << "dist(x1, x2) = " << dist << "  unit_sep = [" << unit_sep[0] << ", " << unit_sep[1] << ", "
            << unit_sep[2] << "]" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  construction_example();
  accessors_example();
  arithmetic_example();
  reductions_example();
  norms_and_products_example();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
