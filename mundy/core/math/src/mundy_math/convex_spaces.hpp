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

#ifndef MUNDY_MATH_CONVEX_SPACES_HPP_
#define MUNDY_MATH_CONVEX_SPACES_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <cstddef>
#include <type_traits>

// Mundy
#include <mundy_math/NumTraits.hpp>                 // for mundy::NumTraits
#include <mundy_math/cmath.hpp>                     // for mundy::max, mundy::min
#include <mundy_math/impl/convex_spaces_impl.hpp>  // for mundy::impl::on_bound
#include <mundy_math/solver_backends.hpp>           // for mundy::BackendVectorExpr and the expression operands

namespace mundy {

//! \name Convex spaces
//@{
// Separable convex spaces: the same interval holds every entry of a vector. A space is its bounds; the functions below
// give its geometry, one entry at a time and over whole vectors.

/// \brief Every entry is free.
template <typename Scalar>
struct UnconstrainedSpace {
  using value_type = Scalar;
};

/// \brief x_i >= lower_bound for every entry.
template <typename Scalar>
struct LowerBoundSpace {
  using value_type = Scalar;
  value_type lower_bound;
};

/// \brief x_i <= upper_bound for every entry.
template <typename Scalar>
struct UpperBoundSpace {
  using value_type = Scalar;
  value_type upper_bound;
};

/// \brief lower_bound <= x_i <= upper_bound for every entry.
template <typename Scalar>
struct BoundedSpace {
  using value_type = Scalar;
  value_type lower_bound;
  value_type upper_bound;
};
//@}

//! \name Geometry of one entry
//@{
// For a space, an entry x of a feasible point, and entries g of a gradient, v of any vector, and d of a direction:
//   project(space, y)                the feasible value nearest y;
//   projected_gradient(space, x, g)  the part of g that a feasible descent step can follow, -P_T(x)(-g);
//   free_part(space, x, v)           v where x is off every bound, 0 where it lies on one;
//   step_to_boundary(space, x, d)    the largest t >= 0 with x + t d feasible, +infinity where d approaches no bound.
// x lies on a bound when it is within epsilon of it.

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar project(const UnconstrainedSpace<Scalar>&, std::type_identity_t<Scalar> y) {
  return y;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar project(const LowerBoundSpace<Scalar>& space, std::type_identity_t<Scalar> y) {
  return max(y, space.lower_bound);
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar project(const UpperBoundSpace<Scalar>& space, std::type_identity_t<Scalar> y) {
  return min(y, space.upper_bound);
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar project(const BoundedSpace<Scalar>& space, std::type_identity_t<Scalar> y) {
  return min(max(y, space.lower_bound), space.upper_bound);
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar projected_gradient(const UnconstrainedSpace<Scalar>&,
                                                           std::type_identity_t<Scalar> /*x*/,
                                                           std::type_identity_t<Scalar> g) {
  return g;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar projected_gradient(const LowerBoundSpace<Scalar>& space,
                                                           std::type_identity_t<Scalar> x,
                                                           std::type_identity_t<Scalar> g) {
  return impl::on_bound(x, space.lower_bound) ? min(g, Scalar(0)) : g;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar projected_gradient(const UpperBoundSpace<Scalar>& space,
                                                           std::type_identity_t<Scalar> x,
                                                           std::type_identity_t<Scalar> g) {
  return impl::on_bound(x, space.upper_bound) ? max(g, Scalar(0)) : g;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar projected_gradient(const BoundedSpace<Scalar>& space,
                                                           std::type_identity_t<Scalar> x,
                                                           std::type_identity_t<Scalar> g) {
  const bool on_lower = impl::on_bound(x, space.lower_bound);
  const bool on_upper = impl::on_bound(x, space.upper_bound);
  if (on_lower && on_upper) {
    return Scalar(0);  // lower_bound = upper_bound: no direction is feasible
  } else if (on_lower) {
    return min(g, Scalar(0));
  } else if (on_upper) {
    return max(g, Scalar(0));
  }
  return g;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar free_part(const UnconstrainedSpace<Scalar>&, std::type_identity_t<Scalar> /*x*/,
                                                  std::type_identity_t<Scalar> v) {
  return v;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar free_part(const LowerBoundSpace<Scalar>& space, std::type_identity_t<Scalar> x,
                                                  std::type_identity_t<Scalar> v) {
  return impl::on_bound(x, space.lower_bound) ? Scalar(0) : v;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar free_part(const UpperBoundSpace<Scalar>& space, std::type_identity_t<Scalar> x,
                                                  std::type_identity_t<Scalar> v) {
  return impl::on_bound(x, space.upper_bound) ? Scalar(0) : v;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar free_part(const BoundedSpace<Scalar>& space, std::type_identity_t<Scalar> x,
                                                  std::type_identity_t<Scalar> v) {
  return impl::on_bound(x, space.lower_bound) || impl::on_bound(x, space.upper_bound) ? Scalar(0) : v;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar step_to_boundary(const UnconstrainedSpace<Scalar>&,
                                                         std::type_identity_t<Scalar> /*x*/,
                                                         std::type_identity_t<Scalar> /*d*/) {
  return NumTraits<Scalar>::infinity();
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar step_to_boundary(const LowerBoundSpace<Scalar>& space,
                                                         std::type_identity_t<Scalar> x,
                                                         std::type_identity_t<Scalar> d) {
  return d < Scalar(0) ? (space.lower_bound - x) / d : NumTraits<Scalar>::infinity();
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar step_to_boundary(const UpperBoundSpace<Scalar>& space,
                                                         std::type_identity_t<Scalar> x,
                                                         std::type_identity_t<Scalar> d) {
  return d > Scalar(0) ? (space.upper_bound - x) / d : NumTraits<Scalar>::infinity();
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar step_to_boundary(const BoundedSpace<Scalar>& space,
                                                         std::type_identity_t<Scalar> x,
                                                         std::type_identity_t<Scalar> d) {
  if (d < Scalar(0)) {
    return (space.lower_bound - x) / d;
  } else if (d > Scalar(0)) {
    return (space.upper_bound - x) / d;
  }
  return NumTraits<Scalar>::infinity();
}
//@}

//! \name Geometry of vectors
//@{
// The operations above over vectors or expressions, as expressions whose entry i is the operation at entry i:
//   projection(backend, space, y), projected_gradient(backend, space, x, g), free_part(backend, space, x, v), and
//   step_to_boundary(backend, space, x, d).
// The step to the boundary of a whole vector is the least entry of step_to_boundary, reduce_min of it.

/// \brief P(y)_i = project(space, y_i).
template <class Space, class Y>
class ProjectionExpr : public impl::UnaryOperand<Y> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = typename Space::value_type;

  KOKKOS_INLINE_FUNCTION ProjectionExpr(const Space& space, const Y& y) : impl::UnaryOperand<Y>(y), space_(space) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return project(space_, this->operand(i));
  }

 private:
  Space space_;
};

/// \brief g^P_i = projected_gradient(space, x_i, g_i).
template <class Space, class X, class G>
class ProjectedGradientExpr : public impl::BinaryOperands<X, G> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = typename Space::value_type;

  KOKKOS_INLINE_FUNCTION ProjectedGradientExpr(const Space& space, const X& x, const G& g)
      : impl::BinaryOperands<X, G>(x, g), space_(space) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return projected_gradient(space_, this->left(i), this->right(i));
  }

 private:
  Space space_;
};

/// \brief free_part(space, x_i, v_i).
template <class Space, class X, class V>
class FreePartExpr : public impl::BinaryOperands<X, V> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = typename Space::value_type;

  KOKKOS_INLINE_FUNCTION FreePartExpr(const Space& space, const X& x, const V& v)
      : impl::BinaryOperands<X, V>(x, v), space_(space) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return free_part(space_, this->left(i), this->right(i));
  }

 private:
  Space space_;
};

/// \brief step_to_boundary(space, x_i, d_i).
template <class Space, class X, class D>
class StepToBoundaryExpr : public impl::BinaryOperands<X, D> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = typename Space::value_type;

  KOKKOS_INLINE_FUNCTION StepToBoundaryExpr(const Space& space, const X& x, const D& d)
      : impl::BinaryOperands<X, D>(x, d), space_(space) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return step_to_boundary(space_, this->left(i), this->right(i));
  }

 private:
  Space space_;
};

/// \brief P(y), the projection of the vector or expression y onto space's set.
template <class Backend, class Space, class Y>
MUNDY_REQUIRES(impl::HasEntryGeometry<Space> && (BackendVectorExpr<Y> || VectorBackend<Backend, Y>))
KOKKOS_INLINE_FUNCTION auto projection(const Backend&, const Space& space, const Y& y) {
  return ProjectionExpr(space, Backend::entries(y));
}

/// \brief g^P, the projected gradient of g at the feasible point x.
template <class Backend, class Space, class X, class G>
MUNDY_REQUIRES(impl::HasEntryGeometry<Space> && (BackendVectorExpr<X> || VectorBackend<Backend, X>))
KOKKOS_INLINE_FUNCTION auto projected_gradient(const Backend&, const Space& space, const X& x, const G& g) {
  return ProjectedGradientExpr(space, Backend::entries(x), Backend::entries(g));
}

/// \brief v where the feasible point x is off every bound, 0 where it lies on one.
template <class Backend, class Space, class X, class V>
MUNDY_REQUIRES(impl::HasEntryGeometry<Space> && (BackendVectorExpr<X> || VectorBackend<Backend, X>))
KOKKOS_INLINE_FUNCTION auto free_part(const Backend&, const Space& space, const X& x, const V& v) {
  return FreePartExpr(space, Backend::entries(x), Backend::entries(v));
}

/// \brief The largest step t >= 0 each entry of the feasible point x can take along d and stay feasible.
template <class Backend, class Space, class X, class D>
MUNDY_REQUIRES(impl::HasEntryGeometry<Space> && (BackendVectorExpr<X> || VectorBackend<Backend, X>))
KOKKOS_INLINE_FUNCTION auto step_to_boundary(const Backend&, const Space& space, const X& x, const D& d) {
  return StepToBoundaryExpr(space, Backend::entries(x), Backend::entries(d));
}

/// \brief Concept for a convex space whose geometry a backend computes over vectors of type Vector.
template <class Space, class Backend, class Vector>
concept ConvexSpace = requires(const Space& space, const Backend& backend, const Vector& x) {
  { projection(backend, space, x) } -> BackendVectorExpr;
  { projected_gradient(backend, space, x, x) } -> BackendVectorExpr;
  { free_part(backend, space, x, x) } -> BackendVectorExpr;
  { step_to_boundary(backend, space, x, x) } -> BackendVectorExpr;
};
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_CONVEX_SPACES_HPP_
