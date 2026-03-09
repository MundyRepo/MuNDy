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

//! \file NgpStorageVsExplicitVariants.cpp
/// \brief Performance test: explicit owned/pointer/reference workspaces vs mundy::storage variants using
/// Kokkos views/kernels.

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <cstddef>
#include <iostream>
#include <string>

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/storage.hpp>

using scalar_t = double;
using View1D = Kokkos::View<scalar_t*, Kokkos::DefaultExecutionSpace>;

KOKKOS_INLINE_FUNCTION
void workspace_step(scalar_t& x, scalar_t& y, scalar_t& z, const scalar_t alpha, const scalar_t beta, const size_t i,
                    const size_t round) {
  const scalar_t wave = static_cast<scalar_t>((i % 11) + 1) + static_cast<scalar_t>((round % 7) + 1);
  const scalar_t t0 = x + beta * y;
  const scalar_t t1 = y - alpha * z + 0.125 * wave;
  z = t0 * t1 + 0.01 * x;
  x = z - 0.25 * y + 0.5 * alpha;
  y = y + alpha * x - beta * z + 0.001 * wave;
}

struct WorkspaceExplicitOwned {
  scalar_t x;
  scalar_t y;
  scalar_t z;

  KOKKOS_INLINE_FUNCTION
  void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
    workspace_step(x, y, z, alpha, beta, i, round);
  }
};

struct WorkspaceExplicitPointer {
  scalar_t* x;
  scalar_t* y;
  scalar_t* z;

  KOKKOS_INLINE_FUNCTION
  void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
    workspace_step(*x, *y, *z, alpha, beta, i, round);
  }
};

struct WorkspaceExplicitReference {
  scalar_t& x;
  scalar_t& y;
  scalar_t& z;

  KOKKOS_INLINE_FUNCTION
  void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
    workspace_step(x, y, z, alpha, beta, i, round);
  }
};

template <class T>
KOKKOS_INLINE_FUNCTION T& storage_element(T& value) {
  return value;
}

template <class T>
KOKKOS_INLINE_FUNCTION T& storage_element(T* value) {
  return *value;
}

template <class XStorage, class YStorage, class ZStorage>
struct WorkspaceStorage {
  XStorage x_storage;
  YStorage y_storage;
  ZStorage z_storage;

  KOKKOS_INLINE_FUNCTION
  void step(const scalar_t alpha, const scalar_t beta, const size_t i, const size_t round) {
    auto& x = storage_element(x_storage.get());
    auto& y = storage_element(y_storage.get());
    auto& z = storage_element(z_storage.get());
    workspace_step(x, y, z, alpha, beta, i, round);
  }
};

void fill_deterministic(View1D x, View1D y, View1D z) {
  const size_t n = x.extent(0);
  Kokkos::parallel_for(
      "fill_deterministic", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
        x(i) = 0.1 + static_cast<scalar_t>((17 * i + 13) % 1024) / 1024.0;
        y(i) = 0.2 + static_cast<scalar_t>((31 * i + 7) % 1024) / 1024.0;
        z(i) = 0.3 + static_cast<scalar_t>((43 * i + 19) % 1024) / 1024.0;
      });
  Kokkos::fence();
}

scalar_t compute_checksum(const View1D x, const View1D y, const View1D z) {
  scalar_t checksum = 0.0;
  const size_t n = x.extent(0);
  Kokkos::parallel_reduce(
      "checksum", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const size_t i, scalar_t& local_sum) {
        if ((i % 7) == 0) {
          local_sum += x(i) * 0.5 + y(i) * 0.25 + z(i) * 0.125;
        }
      },
      checksum);
  Kokkos::fence();
  return checksum;
}

scalar_t run_explicit_owned(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                            const size_t rounds) {
  const size_t n = x.extent(0);
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_explicit_owned", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceExplicitOwned workspace{x(i), y(i), z(i)};
          workspace.step(alpha, beta, i, round);
          x(i) = workspace.x;
          y(i) = workspace.y;
          z(i) = workspace.z;
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

scalar_t run_storage_owned(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                           const size_t rounds) {
  const size_t n = x.extent(0);
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_storage_owned", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceStorage workspace{mundy::store(scalar_t{x(i)}), mundy::store(scalar_t{y(i)}),
                                     mundy::store(scalar_t{z(i)})};
          workspace.step(alpha, beta, i, round);
          x(i) = workspace.x_storage.get();
          y(i) = workspace.y_storage.get();
          z(i) = workspace.z_storage.get();
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

scalar_t run_explicit_pointer(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                              const size_t rounds) {
  const size_t n = x.extent(0);
  auto x_data = x.data();
  auto y_data = y.data();
  auto z_data = z.data();
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_explicit_pointer", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceExplicitPointer workspace{x_data + i, y_data + i, z_data + i};
          workspace.step(alpha, beta, i, round);
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

scalar_t run_storage_pointer(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                             const size_t rounds) {
  const size_t n = x.extent(0);
  auto x_data = x.data();
  auto y_data = y.data();
  auto z_data = z.data();
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_storage_pointer", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceStorage workspace{mundy::store(x_data + i), mundy::store(y_data + i), mundy::store(z_data + i)};
          workspace.step(alpha, beta, i, round);
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

scalar_t run_explicit_reference(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                                const size_t rounds) {
  const size_t n = x.extent(0);
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_explicit_reference", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceExplicitReference workspace{x(i), y(i), z(i)};
          workspace.step(alpha, beta, i, round);
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

scalar_t run_storage_reference(View1D x, View1D y, View1D z, const scalar_t alpha, const scalar_t beta,
                               const size_t rounds) {
  const size_t n = x.extent(0);
  for (size_t round = 0; round < rounds; ++round) {
    Kokkos::parallel_for(
        "run_storage_reference", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const size_t i) {
          WorkspaceStorage workspace{mundy::store(x(i)), mundy::store(y(i)), mundy::store(z(i))};
          workspace.step(alpha, beta, i, round);
        });
  }
  Kokkos::fence();
  return compute_checksum(x, y, z);
}

template <typename Func>
void run_case(ankerl::nanobench::Bench& bench, const std::string& name, const View1D x0, const View1D y0,
              const View1D z0, const View1D x, const View1D y, const View1D z, Func&& func) {
  bench.run(name, [&] {
    Kokkos::deep_copy(x, x0);
    Kokkos::deep_copy(y, y0);
    Kokkos::deep_copy(z, z0);

    const scalar_t checksum = func(x, y, z);
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
    constexpr scalar_t alpha = 1.75;
    constexpr scalar_t beta = 0.65;

    View1D x0("x0", num_entries);
    View1D y0("y0", num_entries);
    View1D z0("z0", num_entries);
    View1D x("x", num_entries);
    View1D y("y", num_entries);
    View1D z("z", num_entries);

    fill_deterministic(x0, y0, z0);

    ankerl::nanobench::Bench bench;
    bench.relative(true)
        .title("NGP storage workspace vs explicit workspace variants")
        .unit("iteration")
        .performanceCounters(true)
        .minEpochIterations(60);

    run_case(bench, "explicit/owned", x0, y0, z0, x, y, z, [&](View1D x_view, View1D y_view, View1D z_view) {
      return run_explicit_owned(x_view, y_view, z_view, alpha, beta, rounds);
    });

    run_case(bench, "storage/owned", x0, y0, z0, x, y, z, [&](View1D x_view, View1D y_view, View1D z_view) {
      return run_storage_owned(x_view, y_view, z_view, alpha, beta, rounds);
    });

    run_case(bench, "explicit/pointer", x0, y0, z0, x, y, z, [&](View1D x_view, View1D y_view, View1D z_view) {
      return run_explicit_pointer(x_view, y_view, z_view, alpha, beta, rounds);
    });

    run_case(bench, "storage/pointer", x0, y0, z0, x, y, z, [&](View1D x_view, View1D y_view, View1D z_view) {
      return run_storage_pointer(x_view, y_view, z_view, alpha, beta, rounds);
    });

    run_case(bench, "explicit/reference", x0, y0, z0, x, y, z, [&](View1D x_view, View1D y_view, View1D z_view) {
      return run_explicit_reference(x_view, y_view, z_view, alpha, beta, rounds);
    });

    run_case(bench, "storage/reference", x0, y0, z0, x, y, z, [&](View1D x_view, View1D y_view, View1D z_view) {
      return run_storage_reference(x_view, y_view, z_view, alpha, beta, rounds);
    });

    std::cout << "NGP storage benchmark completed." << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
