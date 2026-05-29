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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRREDUCTIONS_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRREDUCTIONS_HPP_

/// \file NgpAccessorExprReductions.hpp
/// \brief Reduction helpers for NGP accessor expressions.

#include <mundy_mesh/impl/NgpAccessorExprEntityBase.hpp>
#include <mundy_mesh/impl/NgpAccessorExprMathBase.hpp>
#include <mundy_utils/requires.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <typename Expr, typename ReductionOp>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
void reduce_local_impl(Expr&& expr, ReductionOp& reduction) {
  auto driver = expr.driver();
  driver->reduce_local(expr, reduction);
}

template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto reduce_local_sum_impl(Expr&& expr) {
  Scalar local_sum = 0;
  Kokkos::Sum<Scalar> sum_reduction(local_sum);
  reduce_local_impl(std::forward<Expr>(expr), sum_reduction);
  return local_sum;
}

template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto reduce_local_max_impl(Expr&& expr) {
  Scalar local_max;
  Kokkos::Max<Scalar> max_reduction(local_max);
  reduce_local_impl(std::forward<Expr>(expr), max_reduction);
  return local_max;
}

template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto reduce_local_min_impl(Expr&& expr) {
  Scalar local_min;
  Kokkos::Min<Scalar> min_reduction(local_min);
  reduce_local_impl(std::forward<Expr>(expr), min_reduction);
  return local_min;
}

template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto all_reduce_sum_impl(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_sum = reduce_local_sum_impl<Scalar>(std::forward<Expr>(expr));
  Scalar global_sum = 0;
  stk::all_reduce_sum(driver->bulk_data().parallel(), &local_sum, &global_sum, 1);
  return global_sum;
}

template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto all_reduce_max_impl(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_max = reduce_local_max_impl<Scalar>(std::forward<Expr>(expr));
  Scalar global_max = 0;
  stk::all_reduce_max(driver->bulk_data().parallel(), &local_max, &global_max, 1);
  return global_max;
}

template <typename Scalar, typename Expr>
MUNDY_REQUIRES(is_crtp_base_of_v<MathExprBase, Expr> || is_crtp_base_of_v<EntityExprBase, Expr>)
auto all_reduce_min_impl(Expr&& expr) {
  auto* driver = expr.driver();
  Scalar local_min = reduce_local_min_impl<Scalar>(std::forward<Expr>(expr));
  Scalar global_min = 0;
  stk::all_reduce_min(driver->bulk_data().parallel(), &local_min, &global_min, 1);
  return global_min;
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRREDUCTIONS_HPP_
