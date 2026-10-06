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

#ifndef MUNDY_MATH_PRECONDITIONERS_HPP_
#define MUNDY_MATH_PRECONDITIONERS_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/solver_backends.hpp>  // for mundy::LinearOperator and the Backend vector ops
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>  // for mundy::storage
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

//! \name Preconditioners
//@{
// A preconditioner P approximates the inverse of a solver's operator A, so that P A is better conditioned than A. A
// solver only applies P; forming it and keeping it current is the caller's concern.

/// \brief No preconditioner: the solver runs on its operator alone.
struct NoPreconditioner {};

/// \brief A preconditioner for vectors of type Vector under Backend: NoPreconditioner, or an operator whose apply(r, z)
/// computes z = P r.
template <class Precond, class Backend, class Vector>
concept Preconditioner = std::same_as<std::remove_cvref_t<Precond>, NoPreconditioner> ||
                         LinearOperator<Backend, std::remove_cvref_t<Precond>, Vector, Vector>;

/// \brief The Jacobi preconditioner z := r ./ d, for d the diagonal of the operator.
///
/// d is read at every apply.
template <class Backend, class DiagVector>
class JacobiPreconditioner {
 public:
  using backend_t = Backend;

  KOKKOS_INLINE_FUNCTION
  explicit JacobiPreconditioner(backend_t, DiagVector&& diag) : diag_storage_(std::forward<DiagVector>(diag)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& diag() const { return diag_storage_.get(); }
  // clang-format on

  KOKKOS_INLINE_FUNCTION size_t domain_size() const {
    return Backend::size(diag());
  }

  KOKKOS_INLINE_FUNCTION size_t range_size() const {
    return Backend::size(diag());
  }

  KOKKOS_INLINE_FUNCTION static constexpr size_t static_domain_size() MUNDY_REQUIRES(Backend::has_static_sizes) {
    return Backend::template static_size<DiagVector>();
  }

  KOKKOS_INLINE_FUNCTION static constexpr size_t static_range_size() MUNDY_REQUIRES(Backend::has_static_sizes) {
    return Backend::template static_size<DiagVector>();
  }

  KOKKOS_INLINE_FUNCTION auto make_domain_vector() const {
    return Backend::make_vector_like(diag());
  }

  KOKKOS_INLINE_FUNCTION auto make_range_vector() const {
    return Backend::make_vector_like(diag());
  }

  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    MUNDY_THROW_ASSERT(Backend::size(x) == Backend::size(diag()), std::invalid_argument,
                       "JacobiPreconditioner: size mismatch.");
    Backend::elementwise_div(x, diag(), y);
  }

 private:
  ::mundy::storage<DiagVector> diag_storage_;
};
//@}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

template <class Backend, class DiagVector>
JacobiPreconditioner(Backend, DiagVector&&) -> JacobiPreconditioner<Backend, DiagVector>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <typename Backend, class DiagVector>
KOKKOS_INLINE_FUNCTION auto make_jacobi_preconditioner(DiagVector&& diag) {
  return JacobiPreconditioner(Backend{}, std::forward<DiagVector>(diag));
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_PRECONDITIONERS_HPP_
