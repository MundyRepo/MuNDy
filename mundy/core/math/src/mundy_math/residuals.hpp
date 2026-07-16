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
#include <stdexcept>

// Mundy
#include <mundy_math/Tolerance.hpp>        // for mundy::get_zero_tolerance<T>
#include <mundy_math/cmath.hpp>            // for mundy::sqrt, mundy::abs, mundy::max
#include <mundy_math/convex_spaces.hpp>    // for mundy::LowerBoundSpace, mundy::ValidConvexSpace
#include <mundy_math/solver_backends.hpp>  // for mundy::impl::vector_value_type and the Backend vector ops
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

//! \name Residual policies
//@{
// A residual policy maps a solver's current state to a scalar convergence measure computed through a Backend. The
// convergence check (residual <= tol) is fixed; which residual is measured is the swappable choice. Two families
// live here: plain-vector residuals over a residual vector r (for linear-system solves) and projected-gradient
// residuals over (x, grad, space) (for constrained convex solves).

/// \brief res = ||r||_2 (absolute). The classic linear-solve convergence measure.
struct L2Residual {
  template <class Backend, class RVector, class BVector, class ReductionScalar = impl::vector_value_type<RVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend&, const RVector& r, const BVector&) const {
    return sqrt(Backend::template dot<ReductionScalar>(r, r));
  }
};

/// \brief res = ||r||_2 / ||b||_2 (relative to the right-hand side). Scale-independent, unlike L2Residual.
struct RelativeL2Residual {
  template <class Backend, class RVector, class BVector, class ReductionScalar = impl::vector_value_type<RVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend&, const RVector& r, const BVector& b) const {
    const ReductionScalar r_norm = sqrt(Backend::template dot<ReductionScalar>(r, r));
    const ReductionScalar b_norm = sqrt(Backend::template dot<ReductionScalar>(b, b));
    return r_norm / max(b_norm, get_zero_tolerance<ReductionScalar>());
  }
};

/// \brief res = ||r||_inf (max absolute residual entry), via Backend::reduce_max.
struct LinfResidual {
  template <class Backend, class RVector, class BVector, class ReductionScalar = impl::vector_value_type<RVector>>
  KOKKOS_FUNCTION ReductionScalar operator()(const Backend& backend, const RVector& r, const BVector&) const {
    ReductionScalar max_val;
    Backend::template reduce_max<ReductionScalar>(
        r, Backend::size(r),
        KOKKOS_LAMBDA(const int i, ReductionScalar& m) {
          const ReductionScalar v = abs(Backend::vector_data(r, i));
          if (v > m) m = v;
        },
        max_val);
    return max_val;
  }
};

/// \brief res = ||proj_grad(x)||_inf, the max absolute projected gradient entry (Eq 2.2 of Dai & Fletcher 2005).
/// Only defined for a non-negativity lower bound.
struct LinfNormProjectedGradientResidual {
  template <typename Backend, typename XVector, typename GradVector,
            typename ReductionScalar = impl::vector_value_type<GradVector>>
  KOKKOS_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                             const XVector& x,                         //
                                             const GradVector& grad,                   //
                                             const LowerBoundSpace<ReductionScalar>& convex_space) const {
    MUNDY_THROW_REQUIRE(convex_space.lower_bound == static_cast<ReductionScalar>(0), std::invalid_argument,
                        "LinfNormProjectedGradientResidual is only implemented for non-negativity constraints.");

    using value_type = ReductionScalar;

    size_t n = Backend::size(x);
    value_type largest_abs_gradient;
    Backend::template reduce_max<value_type>(
        x, n,
        KOKKOS_LAMBDA(const int i, value_type& max_val) {
          value_type x_i = Backend::vector_data(x, i);
          value_type grad_i = Backend::vector_data(grad, i);

          value_type abs_projected_grad;
          if (x_i < get_zero_tolerance<value_type>()) {
            abs_projected_grad = max(value_type(0), grad_i);
          } else {
            abs_projected_grad = abs(grad_i);
          }

          if (abs_projected_grad > max_val) {
            max_val = abs_projected_grad;
          }
        },
        largest_abs_gradient);

    return largest_abs_gradient;
  }
};

/// \brief res = ||x - proj(x - h grad)||_inf / h for a small step h (line 17 / Eq 25 of Mazhar 2015). Works for
/// any ValidConvexSpace.
struct LinfNormProjectedDiffResidual {
  template <typename Backend, typename XVector, typename GradVector, ValidConvexSpace ConvexSpace,
            typename ReductionScalar = impl::vector_value_type<GradVector>>
  KOKKOS_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                             const XVector& x,                         //
                                             const GradVector& grad,                   //
                                             const ConvexSpace& convex_space) const {
    using value_type = ReductionScalar;

    size_t num_unknowns = Backend::size(x);
    constexpr value_type small_step_size = static_cast<value_type>(1e-6);
    value_type largest_abs_diff;
    Backend::template reduce_max<value_type>(
        x, num_unknowns,
        KOKKOS_LAMBDA(const int i, value_type& max_val) {
          value_type x_i = Backend::vector_data(x, i);
          value_type grad_i = Backend::vector_data(grad, i);
          value_type x_i_proj = convex_space.project(x_i - small_step_size * grad_i);
          value_type abs_diff = abs(x_i - x_i_proj);
          if (abs_diff > max_val) {
            max_val = abs_diff;
          }
        },
        largest_abs_diff);

    return largest_abs_diff / small_step_size;
  }
};
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_RESIDUALS_HPP_
