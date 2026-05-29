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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRUTILS_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRUTILS_HPP_

/// \file NgpAccessorExprUtils.hpp
/// \brief Utility meta-functions and expression-chain helpers for the NGP expression system.

#include <mundy_mesh/impl/NgpAccessorExprTypes.hpp>
#include <mundy_utils/tuple.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <template <class> class B, class E>
struct is_crtp_base_of_impl : std::is_base_of<B<E>, E> {};

/// \brief is_crtp_base_of<B, E>
///
/// Resembles std::is_base_of, but addresses the problem of whether _some_ instantiation
/// of a CRTP templated class B is a base of class E. A CRTP templated class is correctly
/// templated with the most derived type in the CRTP hierarchy. Using this assumption,
/// this implementation deals with either CRTP final classes (checks for inheritance
/// with E as the CRTP parameter of B) or CRTP base classes (which are singly templated
/// by the most derived class, and that's pulled out to use as a template parameter for B).
template <template <class> class B, class E>
using is_crtp_base_of = impl::is_crtp_base_of_impl<B, std::decay_t<E>>;

template <template <class> class B, class E>
static constexpr bool is_crtp_base_of_v = is_crtp_base_of<B, E>::value;

template <typename EvalCountsType, EvalCountsType eval_counts, size_t I = 0, class ExprTuple, size_t NumEntities,
          class CacheType, class Ctx>
KOKKOS_FUNCTION auto cached_expr_chain(const ExprTuple& exprs,
                                       const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis,
                                       CacheType&& cache, const Ctx& ctx) {
  constexpr size_t num_expr = ExprTuple::size();
  if constexpr (I == num_expr) {
    return Kokkos::make_pair(tuple<>{}, std::forward<CacheType>(cache));
  } else {
    auto& expr = get<I>(exprs);
    auto result_i = expr.template cached_eval<EvalCountsType, eval_counts>(fmis, std::forward<CacheType>(cache), ctx);
    auto value_handle_i = std::move(result_i.first);
    auto next_cache = std::move(result_i.second);
    auto [vals_tail, final_cache] =
        cached_expr_chain<EvalCountsType, eval_counts, I + 1>(exprs, fmis, std::move(next_cache), ctx);
    auto vals_all = tuple_cat(tuple{std::move(value_handle_i)}, std::move(vals_tail));
    return Kokkos::make_pair(vals_all, std::move(final_cache));
  }
}

template <size_t I = 0, class ExprTuple, size_t NumEntities, class Ctx>
KOKKOS_FUNCTION auto expr_chain(const ExprTuple& exprs,
                                const Kokkos::Array<stk::mesh::FastMeshIndex, NumEntities>& fmis, const Ctx& ctx) {
  constexpr size_t num_expr = ExprTuple::size();
  if constexpr (I == num_expr) {
    return tuple<>{};
  } else {
    auto val_i = get<I>(exprs).eval(fmis, ctx);
    auto vals_tail = expr_chain<I + 1>(exprs, fmis, ctx);
    return tuple_cat(tuple{std::move(val_i)}, std::move(vals_tail));
  }
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRUTILS_HPP_
