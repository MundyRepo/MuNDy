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

#ifndef MUNDY_MATH_IMPL_EIGENVALUES_IMPL_HPP_
#define MUNDY_MATH_IMPL_EIGENVALUES_IMPL_HPP_

// External
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/NumTraits.hpp>  // for mundy::NumTraits
#include <mundy_math/cmath.hpp>      // for mundy::{abs, sqrt}

namespace mundy {

namespace impl {

/// \brief The last LDL^T pivot d(sigma) = det(sign T_k - sigma I) / det(sign T_{k-1} - sigma I) and d'(sigma), for
/// T_k the leading k x k block of the tridiagonal (alpha, beta) and sigma above the spectrum of sign T_{k-1}.
template <class Scalar, class AlphaArray, class BetaArray>
KOKKOS_FUNCTION void tridiagonal_last_pivot(const AlphaArray& alpha, const BetaArray& beta, unsigned k, Scalar sign,
                                            Scalar sigma, Scalar& pivot, Scalar& derivative) {
  constexpr Scalar one = static_cast<Scalar>(1);
  Scalar d = sign * alpha[0] - sigma;
  Scalar d_prime = -one;
  for (unsigned i = 1; i < k; ++i) {
    // Above every leading block's spectrum each pivot is negative, so a zero pivot is that limit.
    if (d == static_cast<Scalar>(0)) {
      d = -NumTraits<Scalar>::epsilon() * (abs(sigma) + one);
    }
    const Scalar beta_squared = beta[i - 1] * beta[i - 1];
    d_prime = -one + beta_squared * d_prime / (d * d);
    d = sign * alpha[i] - sigma - beta_squared / d;
  }
  pivot = d;
  derivative = d_prime;
}

/// \brief The largest eigenvalue theta of sign T_k and its Ritz residual beta_k |s_k|, for s its unit eigenvector and
/// previous the largest eigenvalue of sign T_{k-1}.
///
/// theta lies in [previous, max(previous, sign alpha_k) + beta_{k-1}] (interlacing, Weyl), where d falls from +infinity
/// to zero at theta; bisection-safeguarded Newton finds it from the right. s_k^2 = -1 / d'(theta).
template <class Scalar, class AlphaArray, class BetaArray>
KOKKOS_FUNCTION void largest_ritz_value(const AlphaArray& alpha, const BetaArray& beta, unsigned k, Scalar sign,
                                        Scalar previous, Scalar& theta, Scalar& ritz_residual) {
  constexpr Scalar zero = static_cast<Scalar>(0);
  constexpr Scalar two = static_cast<Scalar>(2);
  constexpr Scalar eps = NumTraits<Scalar>::epsilon();
  if (k == 1) {
    theta = sign * alpha[0];
    ritz_residual = beta[0];
    return;
  }

  Scalar lower = previous;
  Scalar upper = (sign * alpha[k - 1] > previous ? sign * alpha[k - 1] : previous) + beta[k - 2];
  Scalar sigma = upper;
  Scalar pivot = zero;
  Scalar derivative = -static_cast<Scalar>(1);
  for (int step = 0; step < 200; ++step) {
    tridiagonal_last_pivot(alpha, beta, k, sign, sigma, pivot, derivative);
    if (pivot == zero) {
      break;
    }
    (pivot > zero ? lower : upper) = sigma;
    Scalar next = sigma - pivot / derivative;
    if (!(next > lower && next < upper)) {
      next = (lower + upper) / two;
    }
    const bool settled =
        abs(next - sigma) <= two * eps * abs(sigma) || upper - lower <= two * eps * (abs(lower) + abs(upper));
    sigma = next;
    if (settled) {
      tridiagonal_last_pivot(alpha, beta, k, sign, sigma, pivot, derivative);
      break;
    }
  }
  theta = sigma;
  ritz_residual = beta[k - 1] / sqrt(-derivative);
}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_EIGENVALUES_IMPL_HPP_
