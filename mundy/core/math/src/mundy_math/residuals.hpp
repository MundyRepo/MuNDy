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

#ifndef MUNDY_MATH_RESIDUALS_HPP_
#define MUNDY_MATH_RESIDUALS_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>

// Mundy
#include <mundy_math/Tolerance.hpp>        // for mundy::get_zero_tolerance<T>
#include <mundy_math/cmath.hpp>            // for mundy::sqrt, mundy::abs, mundy::max
#include <mundy_math/convex_spaces.hpp>    // for mundy::{projection, projected_gradient}
#include <mundy_math/solver_backends.hpp>  // for mundy::impl::vector_value_type and the Backend vector ops

namespace mundy {

//! \name Residual policies
//@{
// A residual policy maps a solver's state to a scalar convergence measure. The convergence check (residual <= tol) is
// fixed; which residual is measured is the swappable choice. Policies are grouped by the contract they satisfy, each
// named by a concept; a solver states which contracts it accepts.
//@}

//! \name Vector residuals: policy(backend, r, b), a residual vector r measured against a reference vector b
//@{

template <class Policy, class Backend, class RVector, class BVector>
concept VectorResidualPolicy =
    requires(const Policy& policy, const Backend& backend, const RVector& r, const BVector& b) {
      { policy(backend, r, b) } -> std::convertible_to<impl::vector_value_type<RVector>>;
    };

/// \brief res = ||r||_2 (absolute).
struct L2Residual {
  template <class Backend, class RVector, class BVector, class ReductionScalar = impl::vector_value_type<RVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend&, const RVector& r, const BVector&) const {
    return sqrt(dot(Backend{}, r, r));
  }
};

/// \brief res = ||r||_2 / ||b||_2 (relative). Scale-independent, unlike L2Residual.
struct RelativeL2Residual {
  template <class Backend, class RVector, class BVector, class ReductionScalar = impl::vector_value_type<RVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend&, const RVector& r, const BVector& b) const {
    const ReductionScalar r_norm = sqrt(dot(Backend{}, r, r));
    const ReductionScalar b_norm = sqrt(dot(Backend{}, b, b));
    return r_norm / max(b_norm, get_zero_tolerance<ReductionScalar>());
  }
};

/// \brief res = ||r||_inf, the largest absolute entry of r, or 0 if r has none.
struct LinfResidual {
  template <class Backend, class RVector, class BVector, class ReductionScalar = impl::vector_value_type<RVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend& backend, const RVector& r, const BVector&) const {
    return max(static_cast<ReductionScalar>(reduce_max(backend, abs(Backend::entries(r)))), ReductionScalar(0));
  }
};

//@}

//! \name Projected residuals: policy(backend, x, grad, space), an iterate and its gradient against a convex space
//@{

template <class Policy, class Backend, class XVector, class GradVector, class ConvexSpace>
concept ProjectedResidualPolicy = requires(const Policy& policy, const Backend& backend, const XVector& x,
                                           const GradVector& grad, const ConvexSpace& space) {
  { policy(backend, x, grad, space) } -> std::convertible_to<impl::vector_value_type<GradVector>>;
};

/// \brief res = ||g^P||_inf, the largest absolute entry of the projected gradient of grad at x in space's set (Eq 2.2
/// of Dai & Fletcher 2005), or 0 if x has none.
struct LinfNormProjectedGradientResidual {
  template <typename Backend, typename XVector, typename GradVector, typename Space,
            typename ReductionScalar = impl::vector_value_type<GradVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend& backend, const XVector& x, const GradVector& grad,
                                             const Space& space) const {
    const auto largest = reduce_max(backend, abs(projected_gradient(backend, space, x, grad)));
    return max(static_cast<ReductionScalar>(largest), ReductionScalar(0));
  }
};

/// \brief res = ||x - P(x - h grad)||_inf / h for the step h = 1e-6 (line 17 / Eq 25 of Mazhar 2015), or 0 if x has
/// none.
///
/// The difference rounds to ulp(x), so res resolves no finer than about eps ||x||_inf / h, a floor that scales with x.
struct LinfNormProjectedDiffResidual {
  template <typename Backend, typename XVector, typename GradVector, typename Space,
            typename ReductionScalar = impl::vector_value_type<GradVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend& backend, const XVector& x, const GradVector& grad,
                                             const Space& space) const {
    constexpr ReductionScalar h = static_cast<ReductionScalar>(1e-6);
    const auto X = Backend::entries(x);
    const auto largest = reduce_max(backend, abs(X - projection(backend, space, X - h * Backend::entries(grad))));
    return max(static_cast<ReductionScalar>(largest), ReductionScalar(0)) / h;
  }
};

//@}

//! \name Change residuals: policy(current, previous), the change of a scalar estimate between iterates
//@{

template <class Policy, class Scalar>
concept ChangeResidualPolicy = requires(const Policy& policy, Scalar current, Scalar previous) {
  { policy(current, previous) } -> std::convertible_to<Scalar>;
};

/// \brief res = |current - previous| (absolute).
struct ChangeResidual {
  template <class Scalar>
  KOKKOS_FUNCTION Scalar operator()(Scalar current, Scalar previous) const {
    return abs(current - previous);
  }
};

/// \brief res = |current - previous| / |current| (relative), or 0 if current = previous. Scale-independent, unlike
/// ChangeResidual.
struct RelativeChangeResidual {
  template <class Scalar>
  KOKKOS_FUNCTION Scalar operator()(Scalar current, Scalar previous) const {
    return current == previous ? Scalar(0) : abs(current - previous) / abs(current);
  }
};
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_RESIDUALS_HPP_
