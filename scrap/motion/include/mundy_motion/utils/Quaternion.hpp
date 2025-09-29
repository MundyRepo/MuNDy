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

#ifndef MUNDY_METHODS_UTILS_QUATERNION_HPP_
#define MUNDY_METHODS_UTILS_QUATERNION_HPP_

/// \file Quaternion.hpp
/// \brief Declaration of the Quaternion helper class

// C++ core libs
#include <limits>  // for std::numeric_limits
#include <random>  // for std::rand

// Trilinos libs
#include <stk_math/StkMath.hpp>    // for stk::math::cos, stk::math::sqrt, etc.
#include <stk_math/StkVector.hpp>  // for stk::math::Vec

namespace mundy {

namespace motion {

namespace utils {

/// \class Quaternion
/// \brief A helper class for describing rotations using quaternions
struct Quaternion {
  // w,x,y,z
  double w;
  double x;
  double y;
  double z;

  //! \name Constructors
  //@{

  /// \brief Construct from existing quaternion stored as a vector.
  /// \param q An existing quaternion satisfying q = q[0] + q[1]i + q[2]j + q[3]k.
  explicit Quaternion(const stk::math::Vec<double, 4> &q) {
    w = q[0];
    x = q[1];
    y = q[2];
    z = q[3];
  }

  /// \brief Construct from existing quaternion stored component-wise.
  /// The quaternion components correspond to q = qw + qx i + qy j + qz k.
  /// \param qw The scalar component.
  /// \param qx The first complex component.
  /// \param qy The second complex component.
  /// \param qz The third complex component.
  Quaternion(const double qw, const double qx, const double qy, const double qz) {
    w = qw;
    x = qx;
    y = qy;
    z = qz;
  }

  /// \brief Construct from rotation around a given axis, given the rotation angle.
  /// \param v The axis to rotate about.
  /// \param angle The rotation angle (no range restriction).
  Quaternion(const stk::math::Vec<double, 3> &v, const double angle) {
    from_rot(v, angle);
  }

  /// \brief Construct a unit random quaternion.
  /// Here, the unit random quaternion following a uniform distribution law on SO(3).
  /// \param u1 A uniform random number sampled from U[0,1].
  /// \param u2 A uniform random number sampled from U[0,1].
  /// \param u3 A uniform random number sampled from U[0,1].
  Quaternion(const double u1, const double u2, const double u3) {
    from_unit_random(u1, u2, u3);
  }

  /// \brief Default constructor stores a random unit quaternion.
  Quaternion() {
    from_unit_random();
  }
  //@}

  /// \brief Update the stored quaternion based on from rotation around a given axis, given the rotation angle.
  /// \param v The axis to rotate about.
  /// \param angle The rotation angle (no range restriction).
  void from_rot(const stk::math::Vec<double, 3> &v, const double angle) {
    const double sina_2 = stk::math::sin(angle / 2);
    const double cosa_2 = stk::math::cos(angle / 2);
    w = cosa_2;
    x = sina_2 * v[0];
    y = sina_2 * v[1];
    z = sina_2 * v[2];
  }

  /// \brief Update the stored quaternion using a unit random quaternion.
  /// Here, the random unit quaternion following a uniform distribution law on SO(3)
  /// \param u1 A uniform random number sampled from U[0,1].
  /// \param u2 A uniform random number sampled from U[0,1].
  /// \param u3 A uniform random number sampled from U[0,1].
  void from_unit_random(const double u1, const double u2, const double u3) {
    constexpr double pi = 3.14159265358979323846;
    const double a = stk::math::sqrt(1 - u1);
    const double b = stk::math::sqrt(u1);
    const double su2 = stk::math::sin(2 * pi * u2);
    const double cu2 = stk::math::cos(2 * pi * u2);
    const double su3 = stk::math::sin(2 * pi * u3);
    const double cu3 = stk::math::cos(2 * pi * u3);
    w = a * su2;
    x = a * cu2;
    y = b * su3;
    z = b * cu3;
  }

  // set a unit random quaternion representing uniform distribution on sphere surface
  void from_unit_random() {
    // non threadsafe random unit quaternion
    const double u1 = static_cast<double>(std::rand()) / RAND_MAX;
    const double u2 = static_cast<double>(std::rand()) / RAND_MAX;
    const double u3 = static_cast<double>(std::rand()) / RAND_MAX;
    from_unit_random(u1, u2, u3);
  }

  // normalize the quaternion q / ||q||
  void normalize() {
    const double norm = sqrt(w * w + x * x + y * y + z * z);
    w = w / norm;
    x = x / norm;
    y = y / norm;
    z = z / norm;
  }

  // rotate a point v in 3D space around the origin using this quaternion
  // see EN Wikipedia on Quaternions and spatial rotation
  stk::math::Vec<double, 3> rotate(const stk::math::Vec<double, 3> &v) const {
    const double t2 = x * y;
    const double t3 = x * z;
    const double t4 = x * w;
    const double t5 = -y * y;
    const double t6 = y * z;
    const double t7 = y * w;
    const double t8 = -z * z;
    const double t9 = z * w;
    const double t10 = -w * w;
    return stk::math::Vec<double, 3>({2.0 * ((t8 + t10) * v[0] + (t6 - t4) * v[1] + (t3 + t7) * v[2]) + v[0],
                                      2.0 * ((t4 + t6) * v[0] + (t5 + t10) * v[1] + (t9 - t2) * v[2]) + v[1],
                                      2.0 * ((t7 - t3) * v[0] + (t2 + t9) * v[1] + (t5 + t8) * v[2]) + v[2]});
  }

  // rotate a point v in 3D space around a given point p using this quaternion
  stk::math::Vec<double, 3> rotate_around_point(const stk::math::Vec<double, 3> &v,
                                                const stk::math::Vec<double, 3> &p) {
    return rotate(v - p) + p;
  }

  /**
   * \brief rotate the quaternion itself based on rotational velocity omega
   *
   * Delong, JCP, 2015, Appendix A eq1, not linearized
   * \param q
   * \param omega rotational velocity
   * \param dt time interval
   */
  void rotate_self(const stk::math::Vec<double, 3> &rot_vel, const double dt) {
    const double rot_vel_norm =
        stk::math::sqrt(rot_vel[0] * rot_vel[0] + rot_vel[1] * rot_vel[1] + rot_vel[2] * rot_vel[2]);
    if (rot_vel_norm < std::numeric_limits<double>::epsilon()) {
      return;
    }
    const double rot_vel_norm_inv = 1.0 / rot_vel_norm;
    const double sw = stk::math::sin(rot_vel_norm * dt / 2);
    const double cw = stk::math::cos(rot_vel_norm * dt / 2);
    const double rot_vel_cross_xyz_0 = rot_vel[1] * z - rot_vel[2] * y;
    const double rot_vel_cross_xyz_1 = rot_vel[2] * x - rot_vel[0] * z;
    const double rot_vel_cross_xyz_2 = rot_vel[0] * y - rot_vel[1] * x;
    const double rot_vel_dot_xyz = rot_vel[0] * x + rot_vel[1] * y + rot_vel[2] * z;

    x = w * sw * rot_vel[0] * rot_vel_norm_inv + cw * x + sw * rot_vel_norm_inv * rot_vel_cross_xyz_0;
    y = w * sw * rot_vel[1] * rot_vel_norm_inv + cw * y + sw * rot_vel_norm_inv * rot_vel_cross_xyz_1;
    z = w * sw * rot_vel[2] * rot_vel_norm_inv + cw * z + sw * rot_vel_norm_inv * rot_vel_cross_xyz_2;
    w = w * cw - rot_vel_dot_xyz * sw * rot_vel_norm_inv;
    normalize();
  }

  /**
   * \brief rotate the quaternion itself based on rotational velocity omega
   *
   * Delong, JCP, 2015, Appendix A eq1, not linearized
   * \param q
   * \param omega rotational velocity
   * \param dt time interval
   */
  void rotate_self(const double rot_vel_x, const double rot_vel_y, const double rot_vel_z, const double dt) {
    const double rot_vel_norm = sqrt(rot_vel_x * rot_vel_x + rot_vel_y * rot_vel_y + rot_vel_z * rot_vel_z);
    if (rot_vel_norm < std::numeric_limits<double>::epsilon()) {
      return;
    }
    const double rot_vel_norm_inv = 1.0 / rot_vel_norm;
    const double sw = stk::math::sin(rot_vel_norm * dt / 2);
    const double cw = stk::math::cos(rot_vel_norm * dt / 2);
    const double rot_vel_cross_xyz_0 = rot_vel_y * z - rot_vel_z * y;
    const double rot_vel_cross_xyz_1 = rot_vel_z * x - rot_vel_x * z;
    const double rot_vel_cross_xyz_2 = rot_vel_x * y - rot_vel_y * x;
    const double rot_vel_dot_xyz = rot_vel_x * x + rot_vel_y * y + rot_vel_z * z;

    x = w * sw * rot_vel_x * rot_vel_norm_inv + cw * x + sw * rot_vel_norm_inv * rot_vel_cross_xyz_0;
    y = w * sw * rot_vel_y * rot_vel_norm_inv + cw * y + sw * rot_vel_norm_inv * rot_vel_cross_xyz_1;
    z = w * sw * rot_vel_z * rot_vel_norm_inv + cw * z + sw * rot_vel_norm_inv * rot_vel_cross_xyz_2;
    w = w * cw - rot_vel_dot_xyz * sw * rot_vel_norm_inv;
    normalize();
  }
};  // Quaternion

}  // namespace utils

}  // namespace motion

}  // namespace mundy

#endif  // MUNDY_METHODS_UTILS_QUATERNION_HPP_
