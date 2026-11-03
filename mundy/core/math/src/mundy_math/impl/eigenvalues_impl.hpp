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

// C++ core
#include <cstddef>  // for size_t
#include <cstdint>  // for uint64_t
#include <limits>   // for std::numeric_limits
#include <type_traits>

// Mundy
#include <mundy_math/NumTraits.hpp>                  // for mundy::NumTraits
#include <mundy_math/cmath.hpp>                      // for mundy::{abs, sqrt}
#include <mundy_math/impl/solver_backends_impl.hpp>  // for mundy::impl::vector_value_type
#include <mundy_utils/requires.hpp>                  // for MUNDY_REQUIRES
#include <mundy_utils/rng.hpp>                       // for mundy::make_philox

namespace mundy {

namespace impl {

/// \brief The cold start of an eigen solve over vectors like x: entry i is a counter-based Philox draw keyed by i,
/// mapped into (-1, 1).
///
/// Its entries follow no sign pattern or index structure, so it is not orthogonal to the structured directions an
/// operator may single out, such as the all-ones vector of a circulant or a shifted Laplacian.
template <class Backend, class Vector>
class ColdStartExpr {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = vector_value_type<std::remove_cvref_t<Vector>>;

  KOKKOS_INLINE_FUNCTION explicit ColdStartExpr(const Vector& x) : size_(Backend::size(x)) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    // the draw's top d - 1 bits h give (h + 1/2) 2^(2 - d) - 1 in [-1 + 2^(1 - d), 1 - 2^(1 - d)]
    constexpr int d = std::numeric_limits<value_type>::digits;
    constexpr size_t stream = 0;
    const uint64_t z = make_philox(i, stream).template draw<uint64_t>();
    const uint64_t h = z >> (64 - (d - 1));
    constexpr value_type spacing = static_cast<value_type>(1) / static_cast<value_type>(uint64_t{1} << (d - 2));
    return (static_cast<value_type>(h) + static_cast<value_type>(0.5)) * spacing - static_cast<value_type>(1);
  }

  KOKKOS_INLINE_FUNCTION size_t size() const {
    return size_;
  }

  KOKKOS_INLINE_FUNCTION static constexpr size_t static_size() MUNDY_REQUIRES(Backend::has_static_sizes) {
    return Backend::template static_size<Vector>();
  }

 private:
  size_t size_;
};

/// \brief The cold start for vectors like x.
template <class Backend, class Vector>
KOKKOS_INLINE_FUNCTION auto cold_start(const Vector& x) {
  return ColdStartExpr<Backend, std::remove_cvref_t<Vector>>(x);
}

/// \brief The largest power of two at most x, for finite x > 0, and 1 otherwise.
///
/// Dividing by it is exact unless the quotient is subnormal.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar power_of_two_at_most(Scalar x) {
  constexpr Scalar one = 1;
  constexpr Scalar two = 2;
  if (!(x > Scalar(0)) || !(x < NumTraits<Scalar>::infinity())) {
    return one;
  }
  Scalar m = one;
  while (m > x) {
    m /= two;
  }
  while (two * m <= x) {
    m *= two;
  }
  return m;
}

/// \brief How many scalars a scalar array holds: a view's extent or a Kokkos::Array's size.
template <class Array>
KOKKOS_INLINE_FUNCTION constexpr size_t scalar_array_size(const Array& a) {
  if constexpr (requires { a.extent(0); }) {
    return a.extent(0);
  } else {
    return a.size();
  }
}

/// \brief Gershgorin's bounds on the spectrum of T_k, the leading k x k block of a symmetric tridiagonal T.
template <class Scalar>
struct GershgorinBounds {
  Scalar lower = NumTraits<Scalar>::infinity();
  Scalar upper = -NumTraits<Scalar>::infinity();
  Scalar norm_bound = 0;  // max_i |alpha_i| + beta_{i-1} + beta_i >= ||T_k||_2, over T_k's rows, beta_0 = beta_k = 0

  /// \brief The bounds of T_k from these of T_{k-1}, for T's diagonal alpha and off-diagonal beta.
  ///
  /// Row k joins with radius beta_{k-1}, and row k - 1's radius grows from beta_{k-2} to beta_{k-2} + beta_{k-1}.
  KOKKOS_INLINE_FUNCTION GershgorinBounds grown(const Scalar* alpha, const Scalar* beta, unsigned k) const {
    GershgorinBounds bounds = *this;
    const Scalar coupling = k > 1 ? beta[k - 2] : Scalar(0);
    bounds.add_disk(alpha[k - 1], coupling);
    if (k > 1) {
      bounds.add_disk(alpha[k - 2], (k > 2 ? beta[k - 3] : Scalar(0)) + coupling);
    }
    return bounds;
  }

 private:
  /// \brief Cover the disk of a row with diagonal entry center and the given radius.
  KOKKOS_INLINE_FUNCTION void add_disk(Scalar center, Scalar radius) {
    lower = center - radius < lower ? center - radius : lower;
    upper = center + radius > upper ? center + radius : upper;
    norm_bound = abs(center) + radius > norm_bound ? abs(center) + radius : norm_bound;
  }
};

/// \brief A pivot of T - sigma I's LDL^T, held away from zero by pivmin with its sign kept (-pivmin for an exact zero).
///
/// Each guard perturbs a diagonal entry of T by at most pivmin.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar guarded_pivot(Scalar d, Scalar pivmin) {
  if (abs(d) < pivmin) {
    return d > Scalar(0) ? pivmin : -pivmin;
  }
  return d;
}

/// \brief The pivot alpha - sigma - beta^2 / d after the pivot d of the row coupled to this one by beta, guarded.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar next_pivot(Scalar alpha, Scalar sigma, Scalar beta, Scalar d, Scalar pivmin) {
  return guarded_pivot<Scalar>(alpha - sigma - beta * beta / d, pivmin);
}

/// \brief The number of eigenvalues below sigma of the leading k x k block of the tridiagonal (alpha, beta).
///
/// It is the number of negative pivots of the block's LDL^T (Sturm count): exact for a matrix within pivmin of the
/// block in each diagonal entry (Kahan, Stanford report CS41, 1966).
template <class Scalar>
KOKKOS_FUNCTION unsigned eigenvalues_below(const Scalar* alpha, const Scalar* beta, unsigned k, Scalar sigma,
                                           Scalar pivmin) {
  unsigned count = 0;
  Scalar d = guarded_pivot<Scalar>(alpha[0] - sigma, pivmin);
  count += d < Scalar(0);
  for (unsigned i = 1; i < k; ++i) {
    d = next_pivot<Scalar>(alpha[i], sigma, beta[i - 1], d, pivmin);
    count += d < Scalar(0);
  }
  return count;
}

/// \brief The largest (or smallest) eigenvalue of the leading k x k block of the tridiagonal (alpha, beta).
///
/// Bisection on Sturm counts to an absolute width of 2 eps norm. Each count is exact for the block perturbed by
/// pivmin = eps norm on its diagonal and by rounding in its entries, so the result is within 4 eps norm of the block's
/// extreme eigenvalue whatever its multiplicity: half the width, pivmin, and under 2 eps norm of rounding.
///
/// [lower, upper] must hold it to within the width; lower may be a lower bound on the largest eigenvalue (any earlier
/// block's, by interlacing) and upper an upper bound on the smallest.
template <class Scalar>
KOKKOS_FUNCTION Scalar extreme_eigenvalue(const Scalar* alpha, const Scalar* beta, unsigned k, bool largest,
                                          Scalar lower, Scalar upper, Scalar norm) {
  constexpr Scalar two = 2;
  const Scalar width = two * NumTraits<Scalar>::epsilon() * norm;
  const Scalar pivmin = NumTraits<Scalar>::epsilon() * norm;
  // the ends are accurate to within the width; the sweep cap stops a zero width
  upper += width;
  lower -= width;
  for (unsigned sweep = 0; sweep < 128 && upper - lower > width; ++sweep) {
    const Scalar mid = (lower + upper) / two;
    const unsigned below = eigenvalues_below<Scalar>(alpha, beta, k, mid, pivmin);
    if (largest ? below == k : below >= 1) {
      upper = mid;
    } else {
      lower = mid;
    }
  }
  return (lower + upper) / two;
}

/// \brief The Ritz residual beta_k |s_k| / ||s|| at an eigenvalue theta of T_k, the leading k x k block of T.
///
/// s is T_k's eigenvector at theta, from the twisted factorization of T_k - theta I: the pivots D+ of L+ D+ L+^T
/// (forward) and D- of U- D- U-^T (backward) meet at the twist r = argmin |gamma_r|, gamma_r = D+_r + D-_r -
/// (alpha_r - theta), where s_r = 1, which keeps s stable even where T_k has eigenvalues within rounding of theta
/// (Parlett & Dhillon, Linear Algebra Appl. 267, 1997). s / ||s|| is written to s_out unless it is null. forward and
/// backward are scratch of k entries.
template <class Scalar>
KOKKOS_FUNCTION Scalar ritz_residual(const Scalar* alpha, const Scalar* beta, unsigned k, Scalar theta, Scalar norm,
                                     Scalar* forward, Scalar* backward, Scalar* s_out) {
  const Scalar pivmin = NumTraits<Scalar>::epsilon() * norm;
  forward[0] = guarded_pivot<Scalar>(alpha[0] - theta, pivmin);
  for (unsigned i = 1; i < k; ++i) {
    forward[i] = next_pivot<Scalar>(alpha[i], theta, beta[i - 1], forward[i - 1], pivmin);
  }
  backward[k - 1] = guarded_pivot<Scalar>(alpha[k - 1] - theta, pivmin);
  for (unsigned i = k - 1; i-- > 0;) {
    backward[i] = next_pivot<Scalar>(alpha[i], theta, beta[i], backward[i + 1], pivmin);
  }
  unsigned twist = 0;
  Scalar least = NumTraits<Scalar>::infinity();
  for (unsigned i = 0; i < k; ++i) {
    const Scalar gamma = abs(forward[i] + backward[i] - (alpha[i] - theta));
    if (gamma < least) {
      least = gamma;
      twist = i;
    }
  }

  // s_r = 1; s_i = -(beta_i / D+_i) s_{i+1} above r and -(beta_{i-1} / D-_i) s_{i-1} below
  Scalar s = 1;
  Scalar norm_squared = 1;
  if (s_out) {
    s_out[twist] = s;
  }
  for (unsigned i = twist; i-- > 0;) {
    s = -beta[i] / forward[i] * s;
    norm_squared += s * s;
    if (s_out) {
      s_out[i] = s;
    }
  }
  s = 1;
  for (unsigned i = twist + 1; i < k; ++i) {
    s = -beta[i - 1] / backward[i] * s;
    norm_squared += s * s;
    if (s_out) {
      s_out[i] = s;
    }
  }
  const Scalar s_norm = sqrt(norm_squared);
  const Scalar s_last = twist == k - 1 ? Scalar(1) : s;  // the loop below the twist ends at s_k
  if (s_out) {
    for (unsigned i = 0; i < k; ++i) {
      s_out[i] /= s_norm;
    }
  }
  const Scalar last_share = abs(s_last) / s_norm;  // at most 1; rounding may not say so
  return beta[k - 1] * (last_share < Scalar(1) ? last_share : Scalar(1));
}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_EIGENVALUES_IMPL_HPP_
