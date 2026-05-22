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
#include <mundy_math/Matrix3.hpp>     // for mundy::Matrix3d
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaternion, mundy::Quaterniond
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3d

//---------------------------------------------------------------------------------------------------------------------//
// Construction example.
//---------------------------------------------------------------------------------------------------------------------//
void construction_example() {
  std::cout << "\n--- Construction ---\n" << std::endl;

  /*
    A quaternion q = w + xi + yj + zk encodes an orientation.  When q has
    unit norm (w^2 + x^2 + y^2 + z^2 = 1) it represents a pure rotation.

    Mundy provides mundy::Quaternion<Scalar> and the concrete alias
    mundy::Quaterniond (double-precision).

    IMPORTANT -- component order:
      The constructor takes components in SEMANTIC (w, x, y, z) order.
      Internally Mundy stores them in Eigen-coefficient order (x, y, z, w).
      You never see the internal order unless you poke at raw memory, but
      you do need to remember that the first constructor argument is the
      scalar part w, not x.

    In practice you rarely construct a quaternion by hand.  Use the
    conversion functions described in conversion_example below; they are
    harder to get wrong and make the intent of the rotation obvious at the
    call site.

    Identity quaternion (no rotation): w=1, x=y=z=0.
    Quaterniond::identity() is the canonical way to express "no rotation".
  */

  // Identity: no rotation.
  auto id = mundy::Quaterniond::identity();
  std::cout << "identity: w=" << id.w() << " x=" << id.x() << " y=" << id.y() << " z=" << id.z() << std::endl;

  // The recommended way to build a rotation: axis-angle.
  // A 90-degree rotation about the z-axis.
  mundy::Vector3d z_axis{0.0, 0.0, 1.0};
  auto q_z90 = mundy::axis_angle_to_quaternion(z_axis, M_PI / 2.0);
  std::cout << "90 deg about z: w=" << q_z90.w() << " z=" << q_z90.z() << std::endl;
  std::cout << "norm(q_z90) = " << mundy::norm(q_z90) << "  (should be 1)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Conversion functions.
//---------------------------------------------------------------------------------------------------------------------//
void conversion_example() {
  std::cout << "\n--- Conversion Functions ---\n" << std::endl;

  /*
    Building a quaternion from raw (w, x, y, z) components is error-prone:
    you have to remember to halve the angle, multiply by the axis, and get
    the component order right.  Mundy provides three conversion paths that
    let you stay in the representation that is natural for your data.

    axis_angle_to_quaternion(axis, angle)
      The most direct route.  Give a unit axis vector and an angle in
      radians; get back a unit quaternion.  The formula is
        w = cos(angle/2),  (x, y, z) = sin(angle/2) * axis
      but you do not have to remember it.

    euler_to_quat(roll, pitch, yaw)
      Converts ZYX intrinsic Euler angles (roll about x, then pitch about y,
      then yaw about z) to a quaternion.  Useful when angles come from
      external data sources or rigid-body dynamics integrators.

    rotation_matrix_to_quaternion(R)
      Converts a 3x3 orthonormal rotation matrix to the equivalent
      quaternion.  Useful when the rotation is computed or loaded as a
      matrix (e.g., from SVD or Gram-Schmidt).

    quaternion_to_rotation_matrix(q)
      The inverse: recover the rotation matrix from a quaternion.  The
      round-trip rotation_matrix_to_quaternion(quaternion_to_rotation_matrix(q))
      should recover q (up to the sign ambiguity: q and -q represent the
      same rotation).
  */

  // axis_angle_to_quaternion: 90 degrees about x.
  mundy::Vector3d x_axis{1.0, 0.0, 0.0};
  auto q_x90 = mundy::axis_angle_to_quaternion(x_axis, M_PI / 2.0);
  std::cout << "axis_angle 90° about x: w=" << q_x90.w() << " x=" << q_x90.x() << std::endl;

  // euler_to_quat: a pure yaw of 90 degrees (roll=0, pitch=0, yaw=π/2)
  // should give the same quaternion as a 90-degree rotation about z.
  auto q_yaw90 = mundy::euler_to_quat(0.0, 0.0, M_PI / 2.0);
  mundy::Vector3d z_axis{0.0, 0.0, 1.0};
  auto q_z90 = mundy::axis_angle_to_quaternion(z_axis, M_PI / 2.0);
  std::cout << "euler yaw=90° w=" << q_yaw90.w() << " z=" << q_yaw90.z() << std::endl;
  std::cout << "axis_angle z 90° w=" << q_z90.w() << " z=" << q_z90.z() << "  (should match)" << std::endl;

  // rotation_matrix_to_quaternion: 90-degree rotation about z.
  // The rotation matrix is [[0,-1,0],[1,0,0],[0,0,1]].
  mundy::Matrix3d R_z90{{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
  auto q_from_mat = mundy::rotation_matrix_to_quaternion(R_z90);
  std::cout << "from_matrix z 90°:    w=" << q_from_mat.w() << " z=" << q_from_mat.z() << "  (should match above)"
            << std::endl;

  // quaternion_to_rotation_matrix: round-trip back to matrix.
  auto R_recovered = mundy::quaternion_to_rotation_matrix(q_z90);
  std::cout << "round-trip R(0,1) = " << R_recovered(0, 1) << "  (should be -1)" << std::endl;
  std::cout << "round-trip R(1,0) = " << R_recovered(1, 0) << "  (should be  1)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Inverse and conjugate example.
//---------------------------------------------------------------------------------------------------------------------//
void inverse_example() {
  std::cout << "\n--- Inverse and Conjugate ---\n" << std::endl;

  /*
    For a unit quaternion the conjugate and the inverse are the same thing:

      conjugate(q) = w - xi - yj - zk
      inverse(q)   = conjugate(q) / norm(q)^2

    Because unit quaternions have norm 1, inverse(q) == conjugate(q).  In
    practice you can use either; conjugate is slightly cheaper to compute.

    Composing a quaternion with its inverse gives the identity:
      q * inverse(q) == identity
  */

  auto q = mundy::axis_angle_to_quaternion(mundy::Vector3d{0.0, 0.0, 1.0}, M_PI / 2.0);

  auto q_conj = mundy::conjugate(q);
  auto q_inv = mundy::inverse(q);

  std::cout << "q * conjugate(q): w=" << (q * q_conj).w() << " x=" << (q * q_conj).x()
            << "  (should be identity: w=1, x=y=z=0)" << std::endl;

  std::cout << "conjugate == inverse for unit quat: " << (std::abs(q_conj.w() - q_inv.w()) < 1e-12 ? "yes" : "no")
            << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Composition example.
//---------------------------------------------------------------------------------------------------------------------//
void composition_example() {
  std::cout << "\n--- Composition ---\n" << std::endl;

  /*
    Quaternion multiplication composes rotations.  The order matters:
      (q1 * q2) applies q2 first, then q1.

    This is the same convention as matrix multiplication: the rightmost
    rotation is applied first.

    Example: rotate 90 degrees about x, then 90 degrees about z.
    The result should differ from the reverse order.
  */

  auto q_x = mundy::axis_angle_to_quaternion(mundy::Vector3d{1.0, 0.0, 0.0}, M_PI / 2.0);  // 90 deg about x
  auto q_z = mundy::axis_angle_to_quaternion(mundy::Vector3d{0.0, 0.0, 1.0}, M_PI / 2.0);  // 90 deg about z

  auto q_xz = q_x * q_z;  // apply q_z first, then q_x
  auto q_zx = q_z * q_x;  // apply q_x first, then q_z

  // For non-commuting rotations, q_xz != q_zx.
  std::cout << "q_x*q_z: w=" << q_xz.w() << " x=" << q_xz.x() << " y=" << q_xz.y() << " z=" << q_xz.z() << std::endl;
  std::cout << "q_z*q_x: w=" << q_zx.w() << " x=" << q_zx.x() << " y=" << q_zx.y() << " z=" << q_zx.z() << std::endl;
  std::cout << "(these should differ -- rotations do not commute)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Rotating a vector example.
//---------------------------------------------------------------------------------------------------------------------//
void rotate_vector_example() {
  std::cout << "\n--- Rotating a Vector ---\n" << std::endl;

  /*
    To rotate a vector v by a unit quaternion q, use the * operator:
      q * v

    Internally this performs the sandwich product q * v_quat * q_inv where
    v_quat = (0, v.x, v.y, v.z) is v embedded as a pure quaternion.  The
    result is the rotated vector; the * operator returns a Vector3, not a
    Quaternion.

    For Mundy's oriented shapes (Spherocylinder, Ellipsoid, Circle3D, Ring)
    the stored orientation quaternion maps the body's local frame to the lab
    frame.  By convention, the body's z-axis is the natural long axis or
    normal.  To find where the body's z-axis points in lab space, rotate the
    unit z-vector by the orientation quaternion.
  */

  // 90-degree rotation about the z-axis should map e_x -> e_y.
  auto q_z90 = mundy::axis_angle_to_quaternion(mundy::Vector3d{0.0, 0.0, 1.0}, M_PI / 2.0);

  mundy::Vector3d e_x{1.0, 0.0, 0.0};
  mundy::Vector3d e_z{0.0, 0.0, 1.0};

  auto rotated = q_z90 * e_x;
  std::cout << "q_z90 * e_x = [" << rotated[0] << ", " << rotated[1] << ", " << rotated[2] << "]"
            << "  (should be [0, 1, 0] = e_y)" << std::endl;

  // e_z is the rotation axis -- it is unchanged.
  auto still_z = q_z90 * e_z;
  std::cout << "q_z90 * e_z = [" << still_z[0] << ", " << still_z[1] << ", " << still_z[2] << "]"
            << "  (should be [0, 0, 1])" << std::endl;

  // Practical use: find where the body z-axis of an oriented rod points.
  mundy::Quaterniond orientation = q_z90;  // this rod's body z-axis has been rotated 90 deg about lab z
  auto rod_axis = orientation * e_z;
  std::cout << "Rod axis in lab frame = [" << rod_axis[0] << ", " << rod_axis[1] << ", " << rod_axis[2] << "]"
            << "  (body z unchanged by rotation about lab z)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Rotating a tensor example.
//---------------------------------------------------------------------------------------------------------------------//
void rotate_tensor_example() {
  std::cout << "\n--- Rotating a Tensor ---\n" << std::endl;

  /*
    Mundy provides two operators that act on a Matrix3 with a quaternion:

      q * M  -- rotates each column of M by q.  Equivalent to R * M,
                where R is the rotation matrix for q.  This alone is NOT a
                full tensor rotation.

      M * q  -- rotates each row of M by the inverse rotation.
                Equivalent to M * R^T.

    Together they compose into the similarity transform that correctly
    rotates any rank-2 tensor (inertia, stress, diffusion, ...):

      M_lab = q * M_body * conj(q)   ==   R * M_body * R^T

    The asymmetry is intentional: q on the left rotates the column frame;
    conj(q) on the right rotates the row frame in the opposite sense.  The
    combination is the standard change-of-basis for tensors.

    Physical picture: M_body stores the tensor's components in the body frame.
    M_lab gives those same components in the lab frame.  After a 90-degree
    rotation about z, body x maps onto lab y, so the (0,0) entry of the
    inertia tensor moves to the (1,1) entry.
  */

  // Inertia tensor in the body frame: I_x=1 (long axis) < I_y=I_z=3.
  mundy::Matrix3d M_body{{1.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}};

  auto q_z90 = mundy::axis_angle_to_quaternion(mundy::Vector3d{0.0, 0.0, 1.0}, M_PI / 2.0);

  // Full similarity transform using the quaternion operators directly.
  auto M_lab = q_z90 * M_body * mundy::conjugate(q_z90);

  std::cout << "M_body diagonal: [" << M_body(0, 0) << ", " << M_body(1, 1) << ", " << M_body(2, 2) << "]"
            << std::endl;
  std::cout << "M_lab  diagonal: [" << M_lab(0, 0) << ", " << M_lab(1, 1) << ", " << M_lab(2, 2) << "]"
            << "  (x and y swapped: small inertia is now on the y-axis)" << std::endl;
  std::cout << "M_lab(0,1)     = " << M_lab(0, 1) << "  (should be 0: diagonal tensor stays diagonal)" << std::endl;

  // For contrast: q * M alone only left-rotates the columns and is not a
  // valid tensor transform -- the result is not even symmetric.
  auto M_half = q_z90 * M_body;
  std::cout << "q * M alone -- not a tensor rotation: "
            << "(0,0)=" << M_half(0, 0) << " (1,1)=" << M_half(1, 1) << " (0,1)=" << M_half(0, 1) << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Slerp example.
//---------------------------------------------------------------------------------------------------------------------//
void slerp_example() {
  std::cout << "\n--- Slerp (Spherical Linear Interpolation) ---\n" << std::endl;

  /*
    slerp(q1, q2, t) interpolates between two orientations at parameter t
    in [0, 1].  At t=0 you get q1; at t=1 you get q2.

    Slerp travels along the great circle arc on the unit quaternion sphere,
    giving a constant angular velocity interpolation.  This is the correct
    way to blend between two orientations; linear interpolation of the
    quaternion components is not correct because it does not preserve unit
    norm or angular velocity.

    Typical use case: generating smooth orientation trajectories for
    visualization, or blending between sampled orientations in a simulation.
  */

  mundy::Quaterniond q_start = mundy::Quaterniond::identity();

  // 180-degree rotation about z.
  auto q_end = mundy::axis_angle_to_quaternion(mundy::Vector3d{0.0, 0.0, 1.0}, M_PI);

  auto q_half = mundy::slerp(q_start, q_end, 0.5);

  // At t=0.5 we should have a 90-degree rotation about z.
  auto q_expected = mundy::axis_angle_to_quaternion(mundy::Vector3d{0.0, 0.0, 1.0}, M_PI / 2.0);

  std::cout << "slerp(identity, 180_z, 0.5): w=" << q_half.w() << " z=" << q_half.z() << std::endl;
  std::cout << "expected (90_z):             w=" << q_expected.w() << " z=" << q_expected.z() << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  construction_example();
  conversion_example();
  inverse_example();
  composition_example();
  rotate_vector_example();
  rotate_tensor_example();
  slerp_example();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
