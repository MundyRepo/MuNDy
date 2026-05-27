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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRBUILTINS_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRBUILTINS_HPP_

/// \file NgpAccessorExprBuiltins.hpp
/// \brief MUNDY_ACCESSOR_EXPR_FORWARD_FUNC / FORWARD_SINK_FUNC macros and all standard
///        mathematical function wrappers (norm, dot, cross, rotate_quaternion, etc.)
///        in the mundy::mesh namespace.
///
/// These macros and the generated wrappers are implementation details. Users do not need to
/// include this file directly; it is pulled in transitively by NgpAccessorExpr.hpp.

#include <mundy_mesh/impl/NgpAccessorExprApplyValue.hpp>
#include <mundy_mesh/impl/NgpAccessorExprSink.hpp>

// Mundy math headers needed for the forwarded function calls
#include <mundy_math/Matrix.hpp>
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/ScalarWrapper.hpp>
#include <mundy_math/Vector.hpp>

namespace mundy {

namespace mesh {

//! \name Expression forwarding macros
//@{

/// \brief Define a named value-expression wrapper around an arbitrary function call.
///
/// Usage: MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Norm, norm, ::mundy::norm)
/// Defines: NormFunc (callable struct), NormExpr<Exprs...> (type alias), norm(args...) (free function).
#define MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(ExprClassName, FuncName, FuncCall)                                    \
  namespace impl {                                                                                            \
  struct ExprClassName##Func {                                                                                 \
    template <typename... Values>                                                                              \
    KOKKOS_INLINE_FUNCTION auto operator()(const Values&... values) const -> decltype(FuncCall(values...)) {   \
      return FuncCall(values...);                                                                              \
    }                                                                                                          \
  };                                                                                                           \
  template <typename... Exprs>                                                                                 \
  using ExprClassName##Expr = ApplyValueExpr<ExprClassName##Func, Exprs...>;                                   \
  }                                                                                                            \
  template <typename... Args>                                                                                  \
  MUNDY_REQUIRES((impl::is_math_expr_arg_v<Args> || ...))                                                      \
  auto FuncName(const Args&... args) {                                                                         \
    return impl::apply_expr_impl(impl::ExprClassName##Func{}, args...);                                        \
  }

/// \brief SinkAccessMode constant aliases for use with MUNDY_ACCESSOR_EXPR_FORWARD_SINK_FUNC.
#define MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY ::mundy::mesh::impl::SinkAccessMode::ReadOnly
#define MUNDY_ACCESSOR_EXPR_SINK_READ_WRITE ::mundy::mesh::impl::SinkAccessMode::ReadWrite
#define MUNDY_ACCESSOR_EXPR_SINK_OVERWRITE_ALL ::mundy::mesh::impl::SinkAccessMode::OverwriteAll

/// \brief Define a named sink-expression wrapper around an arbitrary mutating function call.
///
/// Usage: MUNDY_ACCESSOR_EXPR_FORWARD_SINK_FUNC(RotateQuaternion, rotate_quaternion,
///                                              ::mundy::rotate_quaternion,
///                                              MUNDY_ACCESSOR_EXPR_SINK_READ_WRITE,
///                                              MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY,
///                                              MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY)
/// Defines: RotateQuaternionSinkFunc, RotateQuaternionSinkPolicy, RotateQuaternionSinkExpr<...>,
///          rotate_quaternion(args...) (free function that runs immediately).
#define MUNDY_ACCESSOR_EXPR_FORWARD_SINK_FUNC(ExprClassName, FuncName, FuncCall, ...)                         \
  namespace impl {                                                                                            \
  struct ExprClassName##SinkFunc {                                                                            \
    template <typename... Values>                                                                             \
    KOKKOS_INLINE_FUNCTION auto operator()(Values&&... values) const                                          \
        -> decltype(FuncCall(std::forward<Values>(values)...)) {                                              \
      return FuncCall(std::forward<Values>(values)...);                                                       \
    }                                                                                                         \
  };                                                                                                          \
  using ExprClassName##SinkPolicy = SinkArgPolicy<__VA_ARGS__>;                                               \
  template <typename... SinkArgs>                                                                             \
  using ExprClassName##SinkExpr = ApplySinkExpr<ExprClassName##SinkFunc, SinkArgs...>;                        \
  }                                                                                                           \
  template <typename... Args>                                                                                 \
  MUNDY_REQUIRES((impl::is_math_expr_arg_v<Args> || ...))                                                     \
  void FuncName(const Args&... args) {                                                                        \
    auto expr = impl::make_named_sink_expr<impl::ExprClassName##SinkPolicy>(impl::ExprClassName##SinkFunc{}, args...); \
    expr.driver()->run(expr);                                                                                 \
  }
//@}

//! \name Standard mathematical function wrappers
//@{

// Vector/Matrix/Quaternion functions
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Copy, copy, copy)                       // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Sum, sum, sum)                          // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Product, product, ::mundy::product)     // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Min, min, ::mundy::min)                 // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Max, max, ::mundy::max)                 // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Mean, mean, ::mundy::mean)              // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Variance, variance, ::mundy::variance)  // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(StdDev, stddev, ::mundy::stddev)        // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Norm, norm, ::mundy::norm)              // v, q, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(OneNorm, one_norm, ::mundy::one_norm)   // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(InfNorm, inf_norm, ::mundy::inf_norm)   // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(TwoNorm, two_norm, ::mundy::two_norm)   // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(TwoNormSquared, two_norm_squared, ::mundy::two_norm_squared)  // v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(NormSquared, norm_squared, ::mundy::norm_squared)             // v, q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(InfinityNorm, infinity_norm, ::mundy::inf_norm)               // v, m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Inverse, inverse, ::mundy::inverse)                           // m, q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Conjugate, conjugate, ::mundy::conjugate)                     // q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Normalize, normalize, ::mundy::normalize)                     // q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Trace, trace, ::mundy::trace)                                 // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Transpose, transpose, ::mundy::transpose)                     // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Determinant, determinant, ::mundy::determinant)               // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Adjugate, adjugate, ::mundy::adjugate)                        // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Cofactors, cofactors, ::mundy::cofactors)                     // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(FrobeniusNorm, frobenius_norm, ::mundy::frobenius_norm)       // m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Dot, dot, ::mundy::dot)                                       // v-v, q-q
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(MinorAngle, minor_angle, ::mundy::minor_angle)                // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(MajorAngle, major_angle, ::mundy::major_angle)                // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(FrobeniusInnerProduct, frobenius_inner_product,
                                 ::mundy::frobenius_inner_product)                           // m-m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(OuterProduct, outer_product, ::mundy::outer_product)        // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Cross, cross, ::mundy::cross)                               // v-v
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(ElementwiseMul, elementwise_mul, ::mundy::elementwise_mul)  // v-v, m-m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(ElementwiseDiv, elementwise_div, ::mundy::elementwise_div)  // v-v, m-m
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Slerp, slerp, ::mundy::slerp)                               // q-q
MUNDY_ACCESSOR_EXPR_FORWARD_SINK_FUNC(RotateQuaternion, rotate_quaternion, ::mundy::rotate_quaternion,
                                      MUNDY_ACCESSOR_EXPR_SINK_READ_WRITE, MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY,
                                      MUNDY_ACCESSOR_EXPR_SINK_READ_ONLY)  // q, v, s

// Scalar functions
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Abs, abs, Kokkos::abs)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Sqrt, sqrt, Kokkos::sqrt)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Exp, exp, Kokkos::exp)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Log, log, Kokkos::log)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Sin, sin, Kokkos::sin)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Cos, cos, Kokkos::cos)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Tan, tan, Kokkos::tan)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Asin, asin, Kokkos::asin)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Acos, acos, Kokkos::acos)
MUNDY_ACCESSOR_EXPR_FORWARD_FUNC(Atan, atan, Kokkos::atan)
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRBUILTINS_HPP_
