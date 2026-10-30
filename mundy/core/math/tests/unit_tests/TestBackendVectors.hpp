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

#ifndef MUNDY_MATH_TESTS_UNIT_TESTS_TESTBACKENDVECTORS_HPP_
#define MUNDY_MATH_TESTS_UNIT_TESTS_TESTBACKENDVECTORS_HPP_

/// \file TestBackendVectors.hpp
/// \brief Vectors of either backend built from host entries, and their entries read back on the host, for the unit
/// tests that check one contract on MundyMathBackend and KokkosBackend alike.

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <array>
#include <cstddef>
#include <vector>

// Mundy
#include <mundy_math/Vector.hpp>  // for mundy::Vector

namespace mundy {

namespace test {

/// \brief The given entries as a Backend vector: a MundyMath vector, or a Kokkos::View on the default space.
template <class Backend, size_t N>
auto make_backend_vector(const std::array<double, N>& entries) {
  if constexpr (Backend::has_static_sizes) {
    Vector<double, N> v;
    for (size_t i = 0; i < N; ++i) {
      v[i] = entries[i];
    }
    return v;
  } else {
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> v("v", N);
    const auto v_host = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < N; ++i) {
      v_host(i) = entries[i];
    }
    Kokkos::deep_copy(v, v_host);
    return v;
  }
}

/// \brief Every entry of a MundyMath vector or a Kokkos::View, on the host.
template <class Vector>
std::vector<double> host_entries(const Vector& v) {
  if constexpr (Kokkos::is_view_v<Vector>) {
    const auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    return std::vector<double>(v_host.data(), v_host.data() + v_host.extent(0));
  } else {
    std::vector<double> entries(Vector::size);
    for (size_t i = 0; i < Vector::size; ++i) {
      entries[i] = v[i];
    }
    return entries;
  }
}

}  // namespace test

}  // namespace mundy

#endif  // MUNDY_MATH_TESTS_UNIT_TESTS_TESTBACKENDVECTORS_HPP_
