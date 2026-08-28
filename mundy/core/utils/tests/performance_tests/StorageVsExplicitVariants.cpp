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

//! \file StorageVsExplicitVariants.cpp
/// \brief Performance test: explicit owned/pointer/reference workspaces vs mundy::storage variants.

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/storage.hpp>

using value_type = double;
using vec_t = std::vector<value_type>;

KOKKOS_INLINE_FUNCTION
constexpr void workspace_step(value_type& x, value_type& y, value_type& z, const value_type alpha,
                              const value_type beta, const size_t i, const size_t round) {
  const value_type wave = static_cast<value_type>((i % 11) + 1) + static_cast<value_type>((round % 7) + 1);
  const value_type t0 = x + beta * y;
  const value_type t1 = y - alpha * z + 0.125 * wave;
  z = t0 * t1 + 0.01 * x;
  x = z - 0.25 * y + 0.5 * alpha;
  y = y + alpha * x - beta * z + 0.001 * wave;
}

struct WorkspaceExplicitOwned {
  value_type x;
  value_type y;
  value_type z;

  KOKKOS_INLINE_FUNCTION
  constexpr void step(const value_type alpha, const value_type beta, const size_t i, const size_t round) {
    workspace_step(x, y, z, alpha, beta, i, round);
  }
};

struct WorkspaceExplicitPointer {
  value_type* x;
  value_type* y;
  value_type* z;

  KOKKOS_INLINE_FUNCTION
  constexpr void step(const value_type alpha, const value_type beta, const size_t i, const size_t round) {
    workspace_step(*x, *y, *z, alpha, beta, i, round);
  }
};

struct WorkspaceExplicitReference {
  value_type& x;
  value_type& y;
  value_type& z;

  KOKKOS_INLINE_FUNCTION
  constexpr void step(const value_type alpha, const value_type beta, const size_t i, const size_t round) {
    workspace_step(x, y, z, alpha, beta, i, round);
  }
};

template <class T>
KOKKOS_INLINE_FUNCTION constexpr T& storage_element(T& value) {
  return value;
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr T& storage_element(T* value) {
  return *value;
}

template <class XStorage, class YStorage, class ZStorage>
struct WorkspaceStorage {
  XStorage x_storage;
  YStorage y_storage;
  ZStorage z_storage;

  KOKKOS_INLINE_FUNCTION
  constexpr void step(const value_type alpha, const value_type beta, const size_t i, const size_t round) {
    auto& x = storage_element(x_storage.get());
    auto& y = storage_element(y_storage.get());
    auto& z = storage_element(z_storage.get());
    workspace_step(x, y, z, alpha, beta, i, round);
  }
};

void fill_deterministic(vec_t& x, vec_t& y, vec_t& z) {
  std::uint64_t seed = 0xC0FFEEULL;
  auto next_unit = [&seed]() {
    seed = seed * 1664525ULL + 1013904223ULL;
    return static_cast<value_type>(seed & 0xFFFFULL) / static_cast<value_type>(0x10000ULL);
  };

  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = 0.1 + next_unit();
    y[i] = 0.2 + next_unit();
    z[i] = 0.3 + next_unit();
  }
}

value_type compute_checksum(const vec_t& x, const vec_t& y, const vec_t& z) {
  value_type checksum = 0.0;
  for (size_t i = 0; i < x.size(); i += 7) {
    checksum += x[i] * 0.5 + y[i] * 0.25 + z[i] * 0.125;
  }
  return checksum;
}

value_type run_explicit_owned(vec_t& x, vec_t& y, vec_t& z, const value_type alpha, const value_type beta,
                              const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceExplicitOwned workspace{x[i], y[i], z[i]};
      workspace.step(alpha, beta, i, round);
      x[i] = workspace.x;
      y[i] = workspace.y;
      z[i] = workspace.z;
    }
  }
  return compute_checksum(x, y, z);
}

value_type run_storage_owned(vec_t& x, vec_t& y, vec_t& z, const value_type alpha, const value_type beta,
                             const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceStorage workspace{mundy::store(value_type{x[i]}), mundy::store(value_type{y[i]}),
                                 mundy::store(value_type{z[i]})};
      workspace.step(alpha, beta, i, round);
      x[i] = workspace.x_storage.get();
      y[i] = workspace.y_storage.get();
      z[i] = workspace.z_storage.get();
    }
  }
  return compute_checksum(x, y, z);
}

value_type run_explicit_pointer(vec_t& x, vec_t& y, vec_t& z, const value_type alpha, const value_type beta,
                                const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceExplicitPointer workspace{&x[i], &y[i], &z[i]};
      workspace.step(alpha, beta, i, round);
    }
  }
  return compute_checksum(x, y, z);
}

value_type run_storage_pointer(vec_t& x, vec_t& y, vec_t& z, const value_type alpha, const value_type beta,
                               const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceStorage workspace{mundy::store(&x[i]), mundy::store(&y[i]), mundy::store(&z[i])};
      workspace.step(alpha, beta, i, round);
    }
  }
  return compute_checksum(x, y, z);
}

value_type run_explicit_reference(vec_t& x, vec_t& y, vec_t& z, const value_type alpha, const value_type beta,
                                  const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceExplicitReference workspace{x[i], y[i], z[i]};
      workspace.step(alpha, beta, i, round);
    }
  }
  return compute_checksum(x, y, z);
}

value_type run_storage_reference(vec_t& x, vec_t& y, vec_t& z, const value_type alpha, const value_type beta,
                                 const size_t rounds) {
  for (size_t round = 0; round < rounds; ++round) {
    for (size_t i = 0; i < x.size(); ++i) {
      WorkspaceStorage workspace{mundy::store(x[i]), mundy::store(y[i]), mundy::store(z[i])};
      workspace.step(alpha, beta, i, round);
    }
  }
  return compute_checksum(x, y, z);
}

template <typename Func>
void run_case(ankerl::nanobench::Bench& bench, const std::string& name, const vec_t& x0, const vec_t& y0,
              const vec_t& z0, Func&& func) {
  bench.run(name, [&] {
    vec_t x = x0;
    vec_t y = y0;
    vec_t z = z0;
    const value_type checksum = func(x, y, z);
    ankerl::nanobench::doNotOptimizeAway(checksum);
    ankerl::nanobench::doNotOptimizeAway(x);
    ankerl::nanobench::doNotOptimizeAway(y);
    ankerl::nanobench::doNotOptimizeAway(z);
  });
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    constexpr size_t num_entries = 200000;
    constexpr size_t rounds = 8;
    constexpr value_type alpha = 1.75;
    constexpr value_type beta = 0.65;

    vec_t x0(num_entries);
    vec_t y0(num_entries);
    vec_t z0(num_entries);
    fill_deterministic(x0, y0, z0);

    ankerl::nanobench::Bench bench;
    bench.relative(true)
        .title("storage workspace vs explicit workspace variants")
        .unit("iteration")
        .performanceCounters(true)
        .minEpochIterations(60);

    run_case(bench, "explicit/owned", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_explicit_owned(x, y, z, alpha, beta, rounds); });

    run_case(bench, "storage/owned", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_storage_owned(x, y, z, alpha, beta, rounds); });

    run_case(bench, "explicit/pointer", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_explicit_pointer(x, y, z, alpha, beta, rounds); });

    run_case(bench, "storage/pointer", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_storage_pointer(x, y, z, alpha, beta, rounds); });

    run_case(bench, "explicit/reference", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_explicit_reference(x, y, z, alpha, beta, rounds); });

    run_case(bench, "storage/reference", x0, y0, z0,
             [&](vec_t& x, vec_t& y, vec_t& z) { return run_storage_reference(x, y, z, alpha, beta, rounds); });

    std::cout << "Storage workspace benchmark completed." << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
