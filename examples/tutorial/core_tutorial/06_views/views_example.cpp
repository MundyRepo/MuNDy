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
#include <mundy_math/Matrix3.hpp>     // for mundy::get_matrix3_view
#include <mundy_math/Quaternion.hpp>  // for mundy::get_quaternion_view
#include <mundy_math/Vector3.hpp>     // for mundy::get_vector3_view
#include <vector>

//---------------------------------------------------------------------------------------------------------------------//
// Basic vector view example.
//---------------------------------------------------------------------------------------------------------------------//
void vectorViewExample() {
  std::cout << "\n--- Vector Views ---\n" << std::endl;

  /*
    Every Vector, Matrix, and Quaternion type in Mundy can operate in one of
    two modes:

      Owning  -- the object holds its own data (the default).  This is what
                 you get from Vector3d{...} or Matrix3d{...}.

      View    -- the object is a non-owning window into data that lives
                 somewhere else.  The math operations are identical; the only
                 difference is that reads and writes go through the external
                 storage instead of internal storage.

    Views matter for two reasons:

      1.  Zero-copy interoperability.  Your particle system stores positions
          as a flat array of doubles.  A view lets you treat any three
          consecutive doubles as a Vector3d without copying them.  Operations
          on the view modify the original array in place.

      2.  Geometric primitives can hold view-backed coordinates.  A
          LineSegment whose start and end points are views into a coordinate
          array participates in all distance queries just like an owning
          LineSegment.  Tutorial 07 shows this in detail.

    get_vector3_view<T>(ptr)  returns a view into three doubles starting at
    ptr.  The template parameter T is the element type.

    The returned type is AVector3<T, T*> -- a Vector3 parameterized on a
    raw-pointer accessor.  You typically use auto and let the compiler figure
    out the type.
  */

  // A flat array of particle position data.
  // Three particles stored as x0 y0 z0 | x1 y1 z1 | x2 y2 z2.
  double positions[9] = {
      1.0, 2.0, 3.0,  // particle 0
      4.0, 5.0, 6.0,  // particle 1
      7.0, 8.0, 9.0   // particle 2
  };

  // Create a view for particle 1 (offset by 3 doubles).
  auto p1 = mundy::get_vector3_view<double>(positions + 3);

  std::cout << "p1 = [" << p1[0] << ", " << p1[1] << ", " << p1[2] << "]" << std::endl;

  // Arithmetic works exactly as with owning vectors.
  auto shifted = p1 + mundy::Vector3d{10.0, 10.0, 10.0};
  std::cout << "p1 + 10 = [" << shifted[0] << ", " << shifted[1] << ", " << shifted[2] << "]" << std::endl;

  // Writing through the view modifies the underlying array.
  p1[1] = 99.0;
  std::cout << "After p1[1]=99: positions[4] = " << positions[4] << "  (should be 99)" << std::endl;

  // The separation between two view-backed particles -- no copies made.
  auto p0 = mundy::get_vector3_view<double>(positions + 0);
  auto sep = p1 - p0;  // returns an owning Vector3d
  std::cout << "sep(p0, p1) = [" << sep[0] << ", " << sep[1] << ", " << sep[2] << "]" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Custom accessor example.
//---------------------------------------------------------------------------------------------------------------------//
void customAccessorExample() {
  std::cout << "\n--- Custom Accessor (Strided View) ---\n" << std::endl;

  /*
    Any type with an operator[](size_t) can act as an accessor for a view.
    This is useful when coordinates are interleaved with other data at a
    known stride, or when they start at a non-zero offset within a larger
    buffer.

    Example: a struct-of-arrays where x, y, z are stored separately at
    stride 2 (even indices hold one field, odd indices hold another).
  */

  // Interleaved data: position_x[0], velocity_x[0], position_x[1], velocity_x[1], ...
  double x_channel[6] = {1.0, 10.0, 2.0, 20.0, 3.0, 30.0};

  struct StridedAccessor {
    double* base;
    size_t stride;
    double& operator[](size_t i) {
      return base[i * stride];
    }
    double operator[](size_t i) const {
      return base[i * stride];
    }
  };

  // View the position values (every other element starting at index 0).
  StridedAccessor pos_accessor{x_channel, 2};
  auto pos_view = mundy::get_vector3_view<double>(pos_accessor);

  std::cout << "pos_view = [" << pos_view[0] << ", " << pos_view[1] << ", " << pos_view[2] << "]"
            << "  (should be [1, 2, 3])" << std::endl;

  // Write through the view -- only the position slots change.
  pos_view[0] = 100.0;
  std::cout << "After pos_view[0]=100: x_channel[0]=" << x_channel[0] << "  x_channel[1]=" << x_channel[1]
            << "  (velocity untouched)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Matrix and quaternion views example.
//---------------------------------------------------------------------------------------------------------------------//
void matrixAndQuaternionViewExample() {
  std::cout << "\n--- Matrix and Quaternion Views ---\n" << std::endl;

  /*
    The same view mechanism works for Matrix3 and Quaternion.

    get_matrix3_view<T>(ptr)   -- view nine doubles as a 3x3 matrix
                                  (row-major, same as Matrix3d's storage)
    get_quaternion_view<T>(ptr) -- view four doubles as a quaternion
                                  (Eigen-coefficient order: x, y, z, w)

    A typical use case is a particle system that stores orientation as four
    consecutive doubles per particle.  A quaternion view lets you rotate
    vectors, compose rotations, or call slerp without copying the data.
  */

  // Per-particle orientation storage: x0 y0 z0 w0 | x1 y1 z1 w1 ...
  // Stored in raw (x,y,z,w) Eigen order, not (w,x,y,z) semantic order.
  double orientations[8] = {
      0.0, 0.0, 0.0, 1.0,  // particle 0: identity (w=1)
      0.0, 0.0, 1.0, 0.0   // particle 1: 180-deg rotation about z (w=0, z=1)
  };

  auto q0 = mundy::get_quaternion_view<double>(orientations + 0);
  auto q1 = mundy::get_quaternion_view<double>(orientations + 4);

  std::cout << "q0: w=" << q0.w() << " z=" << q0.z() << "  (identity)" << std::endl;
  std::cout << "q1: w=" << q1.w() << " z=" << q1.z() << "  (180 deg about z)" << std::endl;

  // The slerp result is an owning quaternion; the inputs are view-backed.
  auto q_half = mundy::slerp(q0, q1, 0.5);
  std::cout << "slerp(q0, q1, 0.5): w=" << q_half.w() << "  (should be ~0.707)" << std::endl;

  // Similarly for a matrix.
  double mat_data[9] = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};

  auto m = mundy::get_matrix3_view<double>(mat_data);
  std::cout << "trace of view-backed matrix = " << mundy::trace(m) << "  (should be 6)" << std::endl;

  // Modifying the view modifies the underlying storage.
  m(1, 1) = 5.0;
  std::cout << "After m(1,1)=5: mat_data[4]=" << mat_data[4] << "  (should be 5)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  vectorViewExample();
  customAccessorExample();
  matrixAndQuaternionViewExample();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
