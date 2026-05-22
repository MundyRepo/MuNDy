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
#include <mundy_math/Matrix.hpp>   // for mundy::Matrix
#include <mundy_math/Matrix3.hpp>  // for mundy::Matrix3, mundy::Matrix3d
#include <mundy_math/Vector3.hpp>  // for mundy::Vector3d

//---------------------------------------------------------------------------------------------------------------------//
// Construction example.
//---------------------------------------------------------------------------------------------------------------------//
void construction_example() {
  std::cout << "\n--- Construction ---\n" << std::endl;

  /*
    mundy::Matrix<Scalar, R, C> is a fixed-size dense matrix with R rows and
    C columns, stored in row-major order.  Like vectors, the scalar type and
    dimensions are compile-time constants, so every matrix lives on the stack
    and every operation can be inlined.

    Alias naming mirrors Eigen's convention for sizes 1-6:

      Matrix3<Scalar>  -- 3x3, arbitrary Scalar
      Matrix3d         -- 3x3, double
      Matrix3f         -- 3x3, float
      Matrix23d        -- 2 rows, 3 columns, double
      Matrix32d        -- 3 rows, 2 columns, double
      ...etc.

    Two initialization forms are supported:

      Nested braces   -- each inner list is one row.
      Flat brace list -- values fill the matrix in row-major order.

    Both compile to the same storage.
  */

  // Nested row initialization.
  mundy::Matrix3d m1{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Equivalent flat row-major initialization.
  mundy::Matrix3d m2{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  std::cout << "m1(0,1) = " << m1(0, 1) << "  (row 0, col 1 = 2)" << std::endl;
  std::cout << "m2(2,0) = " << m2(2, 0) << "  (row 2, col 0 = 7)" << std::endl;

  // Factory helpers.
  auto I = mundy::Matrix3d::identity();
  auto Z = mundy::Matrix3d::zeros();
  auto O = mundy::Matrix3d::ones();
  std::cout << "identity (0,0)=" << I(0, 0) << " (0,1)=" << I(0, 1) << std::endl;
  (void)Z;
  (void)O;
}

//---------------------------------------------------------------------------------------------------------------------//
// Accessors example.
//---------------------------------------------------------------------------------------------------------------------//
void accessors_example() {
  std::cout << "\n--- Accessors ---\n" << std::endl;

  /*
    mundy::Matrix has two access conventions, mirroring mundy::Vector:

      m(r, c)  -- the mathematical accessor.  Takes row and column indices.
                  This is the preferred accessor for math code.

      m[i]     -- the programming accessor.  Treats the matrix as a flat
                  array in row-major order: i = r * num_cols + c.
                  Use this when iterating over all elements or copying raw
                  data.

    Both return references, so assignments through either accessor work.
  */

  mundy::Matrix3d m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // m(0,1) and m[1] are the same element: row 0, col 1.
  std::cout << "m(0,1) = " << m(0, 1) << "  m[1] = " << m[1] << "  (same element)" << std::endl;

  // m(1,2) is row 1, col 2: flattened index = 1*3+2 = 5.
  std::cout << "m(1,2) = " << m(1, 2) << "  m[5] = " << m[5] << "  (same element)" << std::endl;

  // Assign through the mathematical accessor.
  m(2, 2) = 99.0;
  std::cout << "After m(2,2)=99: m[8]=" << m[8] << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Arithmetic example.
//---------------------------------------------------------------------------------------------------------------------//
void arithmetic_example() {
  std::cout << "\n--- Arithmetic ---\n" << std::endl;

  /*
    Element-wise operators (+, -, *, /) follow the same rules as vectors:
    same-shape matrices operate component-by-component; a scalar is broadcast
    to all elements.  The compound assignment forms (+=, -=, *=, /=) mutate
    in place.

    Matrix-vector and matrix-matrix products use the * operator and follow
    the usual linear algebra rules:

      m * v   -- matrix-vector product: result[i] = sum_j m(i,j) * v[j]
      v * m   -- row-vector times matrix: result[j] = sum_i v[i] * m(i,j)
                 Left-multiplication by a row vector; no explicit transpose needed.
      m1 * m2 -- matrix-matrix product: result(i,k) = sum_j m1(i,j) * m2(j,k)
  */

  mundy::Matrix3d A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  mundy::Matrix3d B{{9.0, 8.0, 7.0}, {6.0, 5.0, 4.0}, {3.0, 2.0, 1.0}};

  // Element-wise addition and subtraction.
  auto sum = A + B;   // each element sums to 10
  auto diff = A - B;  // each element: A[i]-B[i]
  std::cout << "(A+B)(0,0) = " << sum(0, 0) << "  (should be 10)" << std::endl;
  std::cout << "(A-B)(0,0) = " << diff(0, 0) << "  (should be -8)" << std::endl;

  // Scalar multiplication and division (broadcast).
  auto scaled = 2.0 * A;
  auto halved = A / 2.0;
  std::cout << "(2*A)(1,1) = " << scaled(1, 1) << "  (should be 10)" << std::endl;
  std::cout << "(A/2)(1,1) = " << halved(1, 1) << "  (should be 2.5)" << std::endl;

  // In-place mutation.
  mundy::Matrix3d C = mundy::Matrix3d::identity();
  C *= 3.0;  // scale identity by 3
  C += A;    // add A element-wise
  std::cout << "(3*I + A)(0,0) = " << C(0, 0) << "  (should be 4: 3+1)" << std::endl;
  std::cout << "(3*I + A)(1,1) = " << C(1, 1) << "  (should be 8: 3+5)" << std::endl;

  // Matrix-vector product.
  mundy::Matrix3d D{{2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 4.0}};  // diagonal
  mundy::Vector3d v{1.0, 1.0, 1.0};
  auto Dv = D * v;
  std::cout << "D * [1,1,1] = [" << Dv[0] << ", " << Dv[1] << ", " << Dv[2] << "]"
            << "  (diagonal scales each component)" << std::endl;

  // Row-vector times matrix (v^T * D): same result for a symmetric diagonal D.
  auto vtD = v * D;
  std::cout << "[1,1,1] * D = [" << vtD[0] << ", " << vtD[1] << ", " << vtD[2] << "]" << std::endl;

  // Matrix-matrix product.
  mundy::Matrix3d I = mundy::Matrix3d::identity();
  auto DI = D * I;  // D * I = D
  std::cout << "(D*I)(2,2) = " << DI(2, 2) << "  (should be 4)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Special operations example.
//---------------------------------------------------------------------------------------------------------------------//
void special_operations_example() {
  std::cout << "\n--- Special Operations ---\n" << std::endl;

  /*
    Mundy provides the standard operations needed for rigid body and
    continuum mechanics code: trace, determinant, transpose, and inverse.

    outer_product(v1, v2) builds the rank-1 matrix v1 * v2^T.
  */

  mundy::Matrix3d m{{4.0, 3.0, 0.0}, {6.0, 3.0, 0.0}, {0.0, 0.0, 2.0}};

  std::cout << "trace(m)       = " << mundy::trace(m) << std::endl;        // 4+3+2=9
  std::cout << "determinant(m) = " << mundy::determinant(m) << std::endl;  // (4*3-3*6)*2 = -12

  auto m_T = mundy::transpose(m);
  std::cout << "transpose(m)(0,1) = " << m_T(0, 1) << "  (was m(1,0)=6)" << std::endl;

  // Frobenius inner product: sum of element-wise products.
  auto I = mundy::Matrix3d::identity();
  std::cout << "frobenius_inner_product(m, I) = " << mundy::frobenius_inner_product(m, I) << "  (= trace(m) = 9)"
            << std::endl;

  // outer_product: useful for building dyadic stress tensors, rotation updates, etc.
  mundy::Vector3d a{1.0, 0.0, 0.0};
  mundy::Vector3d b{0.0, 1.0, 0.0};
  auto ab = mundy::outer_product(a, b);  // the matrix e_x * e_y^T
  std::cout << "outer_product(e_x, e_y)(0,1) = " << ab(0, 1) << "  (should be 1)" << std::endl;
  std::cout << "outer_product(e_x, e_y)(0,0) = " << ab(0, 0) << "  (should be 0)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Inverse example.
//---------------------------------------------------------------------------------------------------------------------//
void inverse_example() {
  std::cout << "\n--- Inverse ---\n" << std::endl;

  /*
    inverse(m) returns the matrix inverse for any fixed-size square NxN
    matrix.  The implementation uses recursive Laplace expansion to compute
    the determinant and the cofactor/adjugate method to compute the inverse
    -- both fully inlined, stack-allocated, and free of LU decomposition or
    heap allocation.

    This means inverse works out of the box for all sizes that Mundy supports
    (Matrix2, Matrix3, Matrix4, Matrix6, and arbitrary Matrix<T,N,N>).  For
    very large N the O(N!) recursive expansion becomes expensive; at that
    point a dedicated linear algebra library (Eigen, Belos, etc.) is the right
    tool.  For the 2x2 through 6x6 matrices that appear in rigid-body and
    continuum mechanics code, the analytical approach is fast and exact.
  */

  // 2x2 inverse: A = [[1, 2], [3, 4]], det = -2.
  // inv(A) = (1/-2) * [[4, -2], [-3, 1]] = [[-2, 1], [1.5, -0.5]].
  mundy::Matrix2d A2{{1.0, 2.0}, {3.0, 4.0}};
  auto A2_inv = mundy::inverse(A2);
  auto check2 = A2 * A2_inv;
  std::cout << "2x2: A * inv(A) diagonal = [" << check2(0, 0) << ", " << check2(1, 1) << "]"
            << "  (should be [1, 1])" << std::endl;
  std::cout << "2x2: inv(A)(0,0) = " << A2_inv(0, 0) << "  (should be -2)" << std::endl;

  // 3x3 inverse with full m * inv(m) = I verification.
  mundy::Matrix3d m{{2.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}};
  auto m_inv = mundy::inverse(m);
  auto product = m * m_inv;

  std::cout << "3x3: m * inv(m):" << std::endl;
  for (int r = 0; r < 3; ++r) {
    std::cout << "  [";
    for (int c = 0; c < 3; ++c) {
      std::cout << product(r, c);
      if (c < 2) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  // 4x4 diagonal matrix: inv(diag(a,b,c,d)) = diag(1/a, 1/b, 1/c, 1/d).
  mundy::Matrix4d D4 = mundy::Matrix4d::zeros();
  D4(0, 0) = 1.0;
  D4(1, 1) = 2.0;
  D4(2, 2) = 4.0;
  D4(3, 3) = 8.0;
  auto D4_inv = mundy::inverse(D4);
  std::cout << "4x4: inv(diag(1,2,4,8)) diagonal = [" << D4_inv(0, 0) << ", " << D4_inv(1, 1) << ", " << D4_inv(2, 2)
            << ", " << D4_inv(3, 3) << "]"
            << "  (should be [1, 0.5, 0.25, 0.125])" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  construction_example();
  accessors_example();
  arithmetic_example();
  special_operations_example();
  inverse_example();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
