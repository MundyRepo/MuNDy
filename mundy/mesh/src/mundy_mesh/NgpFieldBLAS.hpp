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

#ifndef MUNDY_MESH_NGPFIELDBLAS_HPP_
#define MUNDY_MESH_NGPFIELDBLAS_HPP_

/// \file FieldBLAS.hpp
/// \brief A set of BLAS-like operations for stk::mesh::FieldBase objects

// Trilinos
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_mesh/impl/NgpFieldBLASImpl.hpp>  // for stk::mesh::impl::field_dot, nrm2, asum, amax, amin, eamin, eamax

namespace mundy {

namespace mesh {

/*
List of available field operations:
  - field_fill
  - field_randomize
  - field_copy
  - field_swap
  - field_scale
  - field_product
  - field_axpy
  - field_axpby
  - field_axpbyz
  - field_axpbygz
  - field_dot
  - field_nrm2
  - field_sum
  - field_asum
  - field_max
  - field_amax
  - field_min
  - field_amin
*/

/// \brief Fill a component of a field with a scalar value
template <typename Scalar, typename ExecSpace>
void field_fill(const Scalar alpha,                   //
                stk::mesh::FieldBase &field,          //
                int component,                        //
                const stk::mesh::Selector &selector,  //
                const ExecSpace &exec_space) {
  impl::ngp_field_fill_component(alpha, field, component, &selector, exec_space);
}

/// \brief Fill a component of a field with a scalar value
template <typename Scalar, typename ExecSpace>
void field_fill(const Scalar alpha,           //
                stk::mesh::FieldBase &field,  //
                int component,                //
                const ExecSpace &exec_space) {
  impl::ngp_field_fill_component(alpha, field, component, nullptr, exec_space);
}

/// \brief Fill all components of a field with a scalar value
template <typename Scalar, typename ExecSpace>
void field_fill(const Scalar alpha,                   //
                stk::mesh::FieldBase &field,          //
                const stk::mesh::Selector &selector,  //
                const ExecSpace &exec_space) {
  impl::ngp_field_fill(alpha, field, &selector, exec_space);
}

/// \brief Fill all components of a field with a scalar value
template <typename Scalar, typename ExecSpace>
void field_fill(const Scalar alpha,           //
                stk::mesh::FieldBase &field,  //
                const ExecSpace &exec_space) {
  impl::ngp_field_fill(alpha, field, nullptr, exec_space);
}

/// \brief Randomize a component of a field (uniform between 0 and 1)
/// \note This function uses the a counter-based random number generator (Philox) from OpenRAND
///   to draw random number in a performance portable way. Will increment the counter for all
///   entities in the selector that own the field.
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     int component,                        //
                     const stk::mesh::Selector &selector,  //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize_component(seed, counter_field, field, component, &selector, exec_space);
}

/// \brief Randomize a component of a field (uniform between 0 and 1)
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     int component,                        //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize_component(seed, counter_field, field, component, nullptr, exec_space);
}

/// \brief Randomize all components of a field (uniform between 0 and 1)
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     const stk::mesh::Selector &selector,  //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize(seed, counter_field, field, &selector, exec_space);
}

/// \brief Randomize all components of a field (uniform between 0 and 1)
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize(seed, counter_field, field, nullptr, exec_space);
}

/// \brief Randomize a component of a field (between given min and max)
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     Scalar min, Scalar max,               //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     int component,                        //
                     const stk::mesh::Selector &selector,  //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize_component(seed, min, max, counter_field, field, component, &selector, exec_space);
}

/// \brief Randomize a component of a field (between given min and max)
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     Scalar min, Scalar max,               //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     int component,                        //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize_component(seed, min, max, counter_field, field, component, nullptr, exec_space);
}

/// \brief Randomize all components of a field (between given min and max)
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     Scalar min, Scalar max,               //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     const stk::mesh::Selector &selector,  //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize(seed, min, max, counter_field, field, &selector, exec_space);
}

/// \brief Randomize all components of a field (between given min and max)
template <typename Scalar, typename ExecSpace>
void field_randomize(size_t seed,                          //
                     Scalar min, Scalar max,               //
                     stk::mesh::FieldBase &counter_field,  //
                     stk::mesh::FieldBase &field,          //
                     const ExecSpace &exec_space) {
  impl::ngp_field_randomize(seed, min, max, counter_field, field, nullptr, exec_space);
}

/// \brief Deep copy y = x
template <typename Scalar, typename ExecSpace>
void field_copy(stk::mesh::FieldBase &x,              //
                stk::mesh::FieldBase &y,              //
                const stk::mesh::Selector &selector,  //
                const ExecSpace &exec_space) {
  impl::ngp_field_copy<Scalar>(x, y, &selector, exec_space);
}

/// \brief Deep copy y = x
template <typename Scalar, typename ExecSpace>
void field_copy(stk::mesh::FieldBase &x,  //
                stk::mesh::FieldBase &y,  //
                const ExecSpace &exec_space) {
  impl::ngp_field_copy<Scalar>(x, y, nullptr, exec_space);
}

/// \brief Swap the contents of two fields
template <typename Scalar, typename ExecSpace>
void field_swap(stk::mesh::FieldBase &x,              //
                stk::mesh::FieldBase &y,              //
                const stk::mesh::Selector &selector,  //
                const ExecSpace &exec_space) {
  impl::ngp_field_swap<Scalar>(x, y, &selector, exec_space);
}

/// \brief Swap the contents of two fields
template <typename Scalar, typename ExecSpace>
void field_swap(stk::mesh::FieldBase &x,  //
                stk::mesh::FieldBase &y,  //
                const ExecSpace &exec_space) {
  impl::ngp_field_swap<Scalar>(x, y, nullptr, exec_space);
}

/// \brief Scale a field by a scalar x = alpha x
template <typename Scalar, typename ExecSpace>
void field_scale(const Scalar alpha,                   //
                 stk::mesh::FieldBase &x,              //
                 const stk::mesh::Selector &selector,  //
                 const ExecSpace &exec_space) {
  impl::ngp_field_scale(alpha, x, &selector, exec_space);
}

/// \brief Scale a field by a scalar x = alpha x
template <typename Scalar, typename ExecSpace>
void field_scale(const Scalar alpha,       //
                 stk::mesh::FieldBase &x,  //
                 const ExecSpace &exec_space) {
  impl::ngp_field_scale(alpha, x, nullptr, exec_space);
}

/// \brief Compute the element-wise product of two fields z = x * y
template <typename Scalar, typename ExecSpace>
void field_product(stk::mesh::FieldBase &x,              //
                   stk::mesh::FieldBase &y,              //
                   stk::mesh::FieldBase &z,              //
                   const stk::mesh::Selector &selector,  //
                   const ExecSpace &exec_space) {
  impl::ngp_field_product<Scalar>(x, y, z, &selector, exec_space);
}

/// \brief Compute the element-wise product of two fields z = x * y
template <typename Scalar, typename ExecSpace>
void field_product(stk::mesh::FieldBase &x,  //
                   stk::mesh::FieldBase &y,  //
                   stk::mesh::FieldBase &z,  //
                   const ExecSpace &exec_space) {
  impl::ngp_field_product<Scalar>(x, y, z, nullptr, exec_space);
}

/// \brief Compute the element-wise sum of two fields y += alpha x
template <typename Scalar, typename ExecSpace>
void field_axpy(const Scalar alpha,                   //
                stk::mesh::FieldBase &x,              //
                stk::mesh::FieldBase &y,              //
                const stk::mesh::Selector &selector,  //
                const ExecSpace &exec_space) {
  Scalar beta = 1;
  impl::ngp_field_axpbyz(alpha, x, beta, y, y, &selector, exec_space);
}

/// \brief Compute the element-wise sum of two fields y += alpha x
template <typename Scalar, typename ExecSpace>
void field_axpy(const Scalar alpha,       //
                stk::mesh::FieldBase &x,  //
                stk::mesh::FieldBase &y,  //
                const ExecSpace &exec_space) {
  Scalar beta = 1;
  impl::ngp_field_axpbyz(alpha, x, beta, y, y, nullptr, exec_space);
}

/// \brief Compute the element-wise sum of two fields y = alpha x + beta y
template <typename Scalar, typename ExecSpace>
void field_axpby(const Scalar alpha,                   //
                 stk::mesh::FieldBase &x,              //
                 const Scalar beta,                    //
                 stk::mesh::FieldBase &y,              //
                 const stk::mesh::Selector &selector,  //
                 const ExecSpace &exec_space) {
  impl::ngp_field_axpbyz(alpha, x, beta, y, y, &selector, exec_space);
}

/// \brief Compute the element-wise sum of two fields y = alpha x + beta y
template <typename Scalar, typename ExecSpace>
void field_axpby(const Scalar alpha,       //
                 stk::mesh::FieldBase &x,  //
                 const Scalar beta,        //
                 stk::mesh::FieldBase &y,  //
                 const ExecSpace &exec_space) {
  impl::ngp_field_axpbyz(alpha, x, beta, y, y, nullptr, exec_space);
}

/// \brief Compute the element-wise sum of three fields z = alpha x + beta y
template <typename Scalar, typename ExecSpace>
void field_axpbyz(const Scalar alpha,                   //
                  stk::mesh::FieldBase &x,              //
                  const Scalar beta,                    //
                  stk::mesh::FieldBase &y,              //
                  stk::mesh::FieldBase &z,              //
                  const stk::mesh::Selector &selector,  //
                  const ExecSpace &exec_space) {
  impl::ngp_field_axpbyz(alpha, x, beta, y, z, &selector, exec_space);
}

/// \brief Compute the element-wise sum of three fields z = alpha x + beta y
template <typename Scalar, typename ExecSpace>
void field_axpbyz(const Scalar alpha,       //
                  stk::mesh::FieldBase &x,  //
                  const Scalar beta,        //
                  stk::mesh::FieldBase &y,  //
                  stk::mesh::FieldBase &z,  //
                  const ExecSpace &exec_space) {
  impl::ngp_field_axpbyz(alpha, x, beta, y, z, nullptr, exec_space);
}

/// \brief Compute the element-wise sum of three fields z = alpha x + beta y + gamma z
template <typename Scalar, typename ExecSpace>
void field_axpbygz(const Scalar alpha,                   //
                   stk::mesh::FieldBase &x,              //
                   const Scalar beta,                    //
                   stk::mesh::FieldBase &y,              //
                   const Scalar gamma,                   //
                   stk::mesh::FieldBase &z,              //
                   const stk::mesh::Selector &selector,  //
                   const ExecSpace &exec_space) {
  impl::ngp_field_axpbygz(alpha, x, beta, y, gamma, z, &selector, exec_space);
}

/// \brief Compute the element-wise sum of three fields z = alpha x + beta y + gamma z
template <typename Scalar, typename ExecSpace>
void field_axpbygz(const Scalar alpha,       //
                   stk::mesh::FieldBase &x,  //
                   const Scalar beta,        //
                   stk::mesh::FieldBase &y,  //
                   const Scalar gamma,       //
                   stk::mesh::FieldBase &z,  //
                   const ExecSpace &exec_space) {
  impl::ngp_field_axpbygz(alpha, x, beta, y, gamma, z, nullptr, exec_space);
}

/// \brief Compute the dot product of two fields
template <typename Scalar, typename ExecSpace>
inline Scalar field_dot(stk::mesh::FieldBase &x,              //
                        stk::mesh::FieldBase &y,              //
                        const stk::mesh::Selector &selector,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_dot<Scalar>(x, y, &selector, exec_space);
}

/// \brief Compute the dot product of two fields
template <typename Scalar, typename ExecSpace>
inline Scalar field_dot(stk::mesh::FieldBase &x,  //
                        stk::mesh::FieldBase &y,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_dot<Scalar>(x, y, nullptr, exec_space);
}

/// \brief Compute the 2-norm of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_nrm2(stk::mesh::FieldBase &x,              //
                         const stk::mesh::Selector &selector,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_nrm2<Scalar>(x, &selector, exec_space);
}

/// \brief Compute the 2-norm of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_nrm2(stk::mesh::FieldBase &x,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_nrm2<Scalar>(x, nullptr, exec_space);
}

/// \brief Compute the sum of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_sum(stk::mesh::FieldBase &x,              //
                        const stk::mesh::Selector &selector,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_sum<Scalar>(x, &selector, exec_space);
}

/// \brief Compute the sum of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_sum(stk::mesh::FieldBase &x,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_sum<Scalar>(x, nullptr, exec_space);
}

/// \brief Compute the 1-norm of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_asum(stk::mesh::FieldBase &x,              //
                         const stk::mesh::Selector &selector,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_asum<Scalar>(x, &selector, exec_space);
}

/// \brief Compute the 1-norm of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_asum(stk::mesh::FieldBase &x,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_asum<Scalar>(x, nullptr, exec_space);
}

/// \brief Compute the maximum value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_max(stk::mesh::FieldBase &x,              //
                        const stk::mesh::Selector &selector,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_max<Scalar>(x, &selector, exec_space);
}

/// \brief Compute the maximum value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_max(stk::mesh::FieldBase &x,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_max<Scalar>(x, nullptr, exec_space);
}

/// \brief Compute the maximum absolute value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_amax(stk::mesh::FieldBase &x,              //
                         const stk::mesh::Selector &selector,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_amax<Scalar>(x, &selector, exec_space);
}

/// \brief Compute the maximum absolute value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_amax(stk::mesh::FieldBase &x,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_amax<Scalar>(x, nullptr, exec_space);
}

/// \brief Compute the minimum value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_min(stk::mesh::FieldBase &x,              //
                        const stk::mesh::Selector &selector,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_min<Scalar>(x, &selector, exec_space);
}

/// \brief Compute the minimum value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_min(stk::mesh::FieldBase &x,  //
                        const ExecSpace &exec_space) {
  return impl::ngp_field_min<Scalar>(x, nullptr, exec_space);
}

/// \brief Compute the minimum absolute value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_amin(stk::mesh::FieldBase &x,              //
                         const stk::mesh::Selector &selector,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_amin<Scalar>(x, &selector, exec_space);
}

/// \brief Compute the minimum absolute value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar field_amin(stk::mesh::FieldBase &x,  //
                         const ExecSpace &exec_space) {
  return impl::ngp_field_amin<Scalar>(x, nullptr, exec_space);
}

/// We currently do not support arg-min/max operations. This is because we are not sure how to handle entity conflicts
/// when two entities have the same value. If it has use, we'll consider implementing it.
// /// \brief Compute the arg minimum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_emin(Mesh& stk::mesh::FieldBase&x,  //
//                                                                      const stk::mesh::Selector& selector) {
//   return impl::ngp_field_emin(x, &selector);
// }

// /// \brief Compute the arg minimum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_emin(Mesh& stk::mesh::FieldBase&x) {
//   return impl::ngp_field_emin(x, nullptr);
// }

// /// \brief Compute the arg absolute minimum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_eamin(Mesh& stk::mesh::FieldBase&x,  //
//                                                                       const stk::mesh::Selector& selector) {
//   return impl::ngp_field_eamin(x, &selector);
// }

// /// \brief Compute the arg absolute minimum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_eamin(Mesh& stk::mesh::FieldBase&x) {
//   return impl::ngp_field_eamin(x, nullptr);
// }

// /// \brief Compute the arg maximum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_emax(Mesh& stk::mesh::FieldBase&x,  //
//                                                                      const stk::mesh::Selector& selector) {
//   return impl::ngp_field_emax(x, &selector);
// }

// /// \brief Compute the arg maximum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_emax(Mesh& stk::mesh::FieldBase&x) {
//   return impl::ngp_field_emax(x, nullptr);
// }

// /// \brief Compute the arg absolute maximum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_eamax(Mesh& stk::mesh::FieldBase&x,  //
//                                                                       const stk::mesh::Selector& selector) {
//   return impl::ngp_field_eamax(x, &selector);
// }

// /// \brief Compute the arg absolute maximum entity of a field
// template <typename Scalar, typename ExecSpace>
// inline Kokkos::pair<stk::mesh::Entity, typename Scalar> field_eamax(Mesh& stk::mesh::FieldBase&x) {
//   return impl::ngp_field_eamax(x, nullptr);
// }

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPFIELDBLAS_HPP_
