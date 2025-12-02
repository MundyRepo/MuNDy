# MundyMath
This subpackage contains shared functionality for performing small highly-optimized mathematical operations. The core functionality includes
 - **Array<Scalar, N>**
 - **Vector<Scalar, N>**
 - **Matrix<Scalar, N, M>**
 - **Quaternion\<Scalar\>**

## **`Array`**
The `Array` class is a container with constexpr access and construction equivalent to Kokkos::Array.
```cpp
Array<int, 6> a{5, 4, 3, 2, 1, 0};
a[2];  // Returns 4
```

## **`Vector`**
The `Vector` class is endowed with the mathematical properties of vectors. It supports addition, multiplication, dot products, norms, etc. It is templated by the vector's scalar type and size so `Vector<double, 3>` would correspond to a double-typed vector in R3. Similar to Eigen, we offer shorthand naming for scalar types float/double and sizes 1-6. For example:
- `Vector3<Scalar>` -> `Vector<Scalar, 3>`
- `Vector3d`        -> `Vector<double, 3>`
- `Vector2f`        -> `Vector<float, 2>`

### Construction
```cpp
Vector2d v1{1.1, 2.2};  // Set all values explicitly
Vector2d v2{3.3};       // Same as Vector2d{3.3, 3.3}; 
```

### Accessors
Canonically, we treat the `()` accessor as the mathematical accessor and `[]` as the programming accessor into the flattened underlying data. So for a 3x3 matrix this would be `mat33(i, j) == mat33[3 * i + j]`. Of course, for vectors, the two are equivalent. 
```cpp
std::cout << v1[0] << "," << v1(0) << std::end;  // Prints 1.1,1.1
```

### Math!

Mathematical formula in comment.

#### Element-wise addition/subtraction
```cpp
auto v3 = v1 + v2;   // v3_i = v1_i + v2_i
auto v4 = v1 + 2.2;  // v4_i = v1_i + 2.2
v1 += v2;   // v1_i += v2_i
v2 += 2.2;  // v2_i += 2.2
```

#### Element-wise scalar multiplication/division
```cpp
auto v3 = 2.2 * v1;   // v3_i = 2.2 * v1_i
auto v4 = v1 / 2.2;   // v4_i = v1_i / 2.2
v1 *= 2.2;   // v1_i *= 2.2
v2 /= 2.2;   // v2_i /= 2.2
```

#### Special vectors
```cpp
Vector3d::zeros() == Vector3d{0., 0., 0.};
Vector3d::ones() == Vector3d{1., 1., 1.};
```

#### Basic arithmetic reduction operations
The following operations can be performed on a vector to compute various reductions:

| **Operation**       | **Description**                                   | **Example** (for `Vector2d v1{1.1, 2.2}`) |
|----------------------|---------------------------------------------------|-------------------------------------------|
| `size(v)`           | Number of components in the vector.              | `size(v1)` → `2`                          |
| `sum(v)`            | Sum of all components.                           | `sum(v1)` → `3.3`                         |
| `product(v)`        | Product of all components.                       | `product(v1)` → `2.42`                    |
| `min(v)`            | Minimum component value.                         | `min(v1)` → `1.1`                         |
| `max(v)`            | Maximum component value.                         | `max(v1)` → `2.2`                         |
| `mean(v)`           | Mean of all components.                          | `mean(v1)` → `1.65`                       |
| `variance(v)`       | Variance of the components.                      | `variance(v1)` → `0.3025`                 |
| `stddev(v)`         | Standard deviation of the components.            | `stddev(v1)` → `0.55`                     |

#### Special vector operations
The following operations can be performed on vectors to compute various properties:

| **Operation**          | **Description**                                                                 | **Example** (for `Vector2d v1{1., 2.}, v2{3., 4.}`) |
|-------------------------|---------------------------------------------------------------------------------|----------------------------------------------------------|
| `dot(v1, v2)`          | Dot product of two vectors.                                                     | `dot(v1, v2)` → `1. * 3. + 2. * 4. = 11.`          |
| `inf_norm(v)`     | Infinity norm (maximum absolute value of components).                           | `inf_norm(v1)` → `2.`                              |
| `one_norm(v)`          | 1-norm (sum of absolute values of components).                                  | `one_norm(v1)` → `1. + 2. = 3.`                       |
| `two_norm(v)`          | 2-norm (Euclidean norm, square root of the sum of squares).                     | `two_norm(v1)` → `sqrt(1.^2 + 2.^2) = sqrt(5.)`       |
| `two_norm_squared(v)`  | Squared 2-norm (sum of squares of components).                                   | `two_norm_squared(v1)` → `1.^2 + 2.^2 = 5.`           |
| `norm(v)`              | Default norm (same as `two_norm`).                                              | `norm(v1)` → `sqrt(5.)`                                 |
| `norm_squared(v)`      | Default squared norm (same as `two_norm_squared`).                              | `norm_squared(v1)` → `5.`                               |
| `minor_angle(v1, v2)`  | Minor angle between two vectors (angle in radians, between 0 and π).            | `minor_angle(v1, v2)` → `acos(dot(v1, v2) / (||v1|| * ||v2||))` |
| `major_angle(v1, v2)`  | Major angle between two vectors (angle in radians, between 0 and π).           | `major_angle(v1, v2)` → `π - minor_angle(v1, v2)`        |

#### Operations for Vectors of Certain Sizes

Some operations are only defined for vectors of specific sizes. For example, the cross product is only valid for 3D vectors (`Vector3`).

| **Operation**          | **Description**                                   | **Example** (for `Vector3d v1{1., 2., 3.}, v2{4., 5., 6.}`) |
|-------------------------|---------------------------------------------------|--------------------------------------------------------------------|
| `cross(v1, v2)`         | Cross product of two 3D vectors.                 | `cross(v1, v2)` → `Vector3d{-3., 6., -3.}`                     |

## **`Matrix`**
The `Matrix` class is endowed with the mathematical properties of dense matrices. It supports addition, multiplication, matrix-vector, matrix-scalar arithmetic, norms, etc. It is templated by the matrices's scalar type and sizes so `Matrix<double, 3, 2>` would correspond to a double-typed matrix with 3 rows and 2 columns. Similar to Eigen, we offer shorthand naming for scalar types float/double and sizes 1-6. For example:
- `Matrix23<Scalar>` -> `Matrix<Scalar, 2, 3>`
- `Matrix23d`        -> `Matrix<double, 2, 3>`
- `Matrix3d`        -> `Matrix<double, 3, 3>`


### Construction
```cpp
Matrix3d m1{{1., 2., 3.}, 
            {4., 5., 6.}, 
            {7., 8., 9.}};  // Initialize with rows
Matrix3d m2{1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.};    // Initialize row-major
```

### Accessors
Matrices support both row-column accessors (`()`) and flattened row-major data accessors (`[]`). For example:
```cpp
std::cout << m1(0, 1) << "," << m1[1] << std::endl;  // Prints 2.0,2.0
```

### Math!

#### Element-wise addition/subtraction
```cpp
auto m3 = m1 + m2;   // m3_ij = m1_ij + m2_ij
auto m4 = m1 - m2;   // m4_ij = m1_ij - m2_ij
m1 += m2;            // m1_ij += m2_ij
m1 -= m2;            // m1_ij -= m2_ij
```

#### Scalar multiplication/division
```cpp
auto m3 = 2.0 * m1;   // m3_ij = 2. * m1_ij
auto m4 = m1 / 2.0;   // m4_ij = m1_ij / 2.
m1 *= 2.0;            // m1_ij *= 2.
m1 /= 2.0;            // m1_ij /= 2.
```

#### Matrix-vector/matrix-matrix multiplication
```cpp
auto v2 = m1 * v1;  // v2_i = m1_ij * v1_j
auto m3 = m1 * m2;  // m3_ij = m1_ik * m2_kj
```
Of course, the matrices and vectors must be of the correct size. We do, however, take the pythonic approach and use lazy tranposes for vectors, so left vector-matrix multiplication is just
```cpp
auto v4 = v1 * m1;  // v4_j = v1_i * m1_ij
```

### Special Matrices
```cpp
Matrix3d::identity();  // Identity matrix (1's along diagonal)
Matrix3d::zeros();     // Zero matrix
Matrix3d::ones();      // Matrix filled with ones
```
### Basic arithmetic reduction operations
The following operations can be performed on a matrix to compute various reductions:

| **Operation**       | **Description**                                   |
|----------------------|---------------------------------------------------|
| `sum(m)`            | Sum of all elements in the matrix.                |
| `product(m)`        | Product of all elements in the matrix.            |
| `min(m)`            | Minimum element value in the matrix.              |
| `max(m)`            | Maximum element value in the matrix.              |
| `mean(m)`           | Mean of all elements in the matrix.               |
| `variance(m)`       | Variance of the elements in the matrix.           |
| `stddev(m)`         | Standard deviation of the elements in the matrix. |

#### Special matrix operations
The following operations can be performed on matrices to compute various properties or transformations:

| **Operation**              | **Description**                                                                 |
|-----------------------------|---------------------------------------------------------------------------------|
| `determinant(m)`           | Compute the determinant of a square matrix.                                          |
| `trace(m)`                 | Sum of the diagonal elements of the matrix.                                     |
| `transpose(m)`             | Transpose of the matrix.                                                        |
| `cofactors(m)`             | Matrix of cofactors.                                                            |
| `adjugate(m)`              | Adjugate (transpose of the cofactor matrix).                                    |
| `inverse(m)`               | Inverse of the matrix (if invertible).                                          |
| `frobenius_inner_product(m1, m2)` | Frobenius inner product of two matrices.                                   |
| `outer_product(v1, v2)`    | Outer product of two vectors, resulting in a matrix.                            |
| `frobenius_norm(m)`        | Frobenius norm (square root of the sum of squares of all elements).              |
| `inf_norm(m)`         | Maximum absolute row sum.                                                       |
| `one_norm(m)`              | Maximum absolute column sum.                                                    |
| `two_norm(m)`              | 2-norm (largest singular value of the matrix).                                  |




# Things that are missing
- Everyone needs cast.
- 
