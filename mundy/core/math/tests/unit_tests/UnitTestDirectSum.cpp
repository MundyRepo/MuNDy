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

/// \file UnitTestDirectSum.cpp
/// \brief The dense direct sum (direct_sum.hpp) against serial references, for every panel size, shape, and value.
///
/// Interactions and accumulators are functors, so the same per-pair code computes the serial host reference, and no
/// KOKKOS_LAMBDA sits in a test body (CUDA forbids it there).

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>        // for std::abs, std::floor, std::isnan
#include <cstddef>      // for size_t
#include <limits>       // for std::numeric_limits
#include <type_traits>  // for std::remove_cvref_t
#include <vector>       // for std::vector

// Mundy
#include <mundy_math/DoubleDouble.hpp>  // for mundy::DoubleDouble
#include <mundy_math/Matrix.hpp>        // for mundy::Matrix
#include <mundy_math/Vector.hpp>        // for mundy::Vector, mundy::dot
#include <mundy_math/direct_sum.hpp>    // for mundy::direct_sum

namespace mundy {

namespace {

using DeviceSpace = Kokkos::DefaultExecutionSpace;
using DeviceView = Kokkos::View<double*, DeviceSpace::memory_space>;
using HostView = Kokkos::View<double*, Kokkos::HostSpace>;

//! \name Points, interactions, and accumulators
//@{

/// \brief n distinct points in the unit cube as a flat (x, y, z) device view; phase selects the point set.
DeviceView make_points(const size_t n, const double phase) {
  const auto fraction = [](const double x) { return x - std::floor(x); };
  DeviceView points("points", 3 * n);
  auto host = Kokkos::create_mirror_view(points);
  for (size_t i = 0; i < n; ++i) {
    host(3 * i + 0) = fraction(0.6180339887 * static_cast<double>(i) + phase);
    host(3 * i + 1) = fraction(0.7548776662 * static_cast<double>(i) + 2.0 * phase);
    host(3 * i + 2) = fraction(0.5698402910 * static_cast<double>(i) + 3.0 * phase);
  }
  Kokkos::deep_copy(points, host);
  return points;
}

/// \brief The separation x_t - y_s and 1 / |x_t - y_s|, which is 0 for coincident points (branch-free).
template <class View>
KOKKOS_INLINE_FUNCTION Vector<double, 3> separation(const View& targets, const View& sources, const size_t t,
                                                    const size_t s, double& rinv) {
  const Vector<double, 3> r{targets(3 * t) - sources(3 * s), targets(3 * t + 1) - sources(3 * s + 1),
                            targets(3 * t + 2) - sources(3 * s + 2)};
  const double r2 = dot(r, r);
  const bool coincident = r2 < 1e-24;
  rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : r2);
  return r;
}

/// \brief The scalar potential 1 / |x_t - y_s|.
template <class View>
struct InverseDistance {
  View targets;
  View sources;
  KOKKOS_INLINE_FUNCTION double operator()(const size_t t, const size_t s) const {
    double rinv;
    separation(targets, sources, t, s, rinv);
    return rinv;
  }
};

/// \brief The same potential as a double-double, a custom passive scalar.
template <class View>
struct InverseDistanceDD {
  View targets;
  View sources;
  KOKKOS_INLINE_FUNCTION DoubleDouble operator()(const size_t t, const size_t s) const {
    double rinv;
    separation(targets, sources, t, s, rinv);
    return DoubleDouble(rinv) / 3.0;  // a low part, so the sum exercises both halves
  }
};

/// \brief The Stokeslet velocity (f + (f . r) r / r^2) / r at target t from the force at source s.
template <class View>
struct Stokeslet {
  View targets;
  View sources;
  View forces;
  KOKKOS_INLINE_FUNCTION Vector<double, 3> operator()(const size_t t, const size_t s) const {
    double rinv;
    const Vector<double, 3> r = separation(targets, sources, t, s, rinv);
    const Vector<double, 3> f{forces(3 * s), forces(3 * s + 1), forces(3 * s + 2)};
    return rinv * f + (dot(f, r) * rinv * rinv * rinv) * r;
  }
};

/// \brief The Stokeslet tensor (I + r r^T / r^2) / r between target t and source s.
template <class View>
struct StokesletTensor {
  View targets;
  View sources;
  KOKKOS_INLINE_FUNCTION Matrix<double, 3, 3> operator()(const size_t t, const size_t s) const {
    double rinv;
    const Vector<double, 3> r = separation(targets, sources, t, s, rinv);
    Matrix<double, 3, 3> tensor;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        tensor(i, j) = (i == j ? rinv : 0.0) + r[i] * r[j] * rinv * rinv * rinv;
      }
    }
    return tensor;
  }
};

/// \brief Stores target t's sum in its slot of a flat view and counts the calls, for scalar, vector, and matrix sums.
template <class View, class CountView>
struct StoreSum {
  View out;
  CountView calls;
  KOKKOS_INLINE_FUNCTION void operator()(const size_t t, const double sum) const {
    out(t) = sum;
    Kokkos::atomic_inc(&calls());
  }
  KOKKOS_INLINE_FUNCTION void operator()(const size_t t, const DoubleDouble& sum) const {
    out(2 * t) = sum.hi();
    out(2 * t + 1) = sum.lo();
    Kokkos::atomic_inc(&calls());
  }
  KOKKOS_INLINE_FUNCTION void operator()(const size_t t, const Vector<double, 3>& sum) const {
    for (size_t c = 0; c < 3; ++c) {
      out(3 * t + c) = sum[c];
    }
    Kokkos::atomic_inc(&calls());
  }
  KOKKOS_INLINE_FUNCTION void operator()(const size_t t, const Matrix<double, 3, 3>& sum) const {
    for (size_t c = 0; c < 9; ++c) {
      out(9 * t + c) = sum(c / 3, c % 3);
    }
    Kokkos::atomic_inc(&calls());
  }
};
//@}

//! \name Running and checking direct sums
//@{

/// \brief The flat components of the value returned by an interaction, for the serial reference.
inline void append_components(const double value, std::vector<double>& out) {
  out.push_back(value);
}
inline void append_components(const DoubleDouble& value, std::vector<double>& out) {
  out.push_back(value.hi());
  out.push_back(value.lo());
}
inline void append_components(const Vector<double, 3>& value, std::vector<double>& out) {
  for (size_t c = 0; c < 3; ++c) {
    out.push_back(value[c]);
  }
}
inline void append_components(const Matrix<double, 3, 3>& value, std::vector<double>& out) {
  for (size_t c = 0; c < 9; ++c) {
    out.push_back(value(c / 3, c % 3));
  }
}

/// \brief Each target's sum over every source, in source order, computed serially on the host.
template <class Interaction>
std::vector<double> serial_reference(const size_t num_targets, const size_t num_sources,
                                     const Interaction& interaction) {
  using value_t = std::remove_cvref_t<decltype(interaction(0, 0))>;
  std::vector<double> out;
  for (size_t t = 0; t < num_targets; ++t) {
    value_t sum = Kokkos::reduction_identity<value_t>::sum();
    for (size_t s = 0; s < num_sources; ++s) {
      sum += interaction(t, s);
    }
    append_components(sum, out);
  }
  return out;
}

/// \brief direct_sum<PanelSize> on space into a NaN-filled output; checks accumulate ran once per target.
template <size_t PanelSize, class ExecSpace, class Interaction>
std::vector<double> run_direct_sum(const ExecSpace& space, const size_t num_targets, const size_t num_sources,
                                   const size_t num_components, const Interaction& interaction) {
  using memory_space = typename ExecSpace::memory_space;
  Kokkos::View<double*, memory_space> out("out", num_components * num_targets);
  Kokkos::View<size_t, memory_space> calls("calls");
  Kokkos::deep_copy(out, std::numeric_limits<double>::quiet_NaN());
  direct_sum<PanelSize>(space, num_targets, num_sources, interaction,
                        StoreSum<decltype(out), decltype(calls)>{out, calls});
  space.fence();

  size_t host_calls = 0;
  Kokkos::deep_copy(host_calls, calls);
  EXPECT_EQ(host_calls, num_targets) << "accumulate must run exactly once per target (panel size " << PanelSize << ")";
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  return std::vector<double>(host.data(), host.data() + host.extent(0));
}

/// \brief Expect every component within rel_tol of the reference, relative to the largest reference component.
void expect_near_reference(const std::vector<double>& actual, const std::vector<double>& reference, double rel_tol,
                           const char* what) {
  ASSERT_EQ(actual.size(), reference.size()) << what;
  double scale = 0.0;
  for (const double r : reference) {
    scale = std::max(scale, std::abs(r));
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_FALSE(std::isnan(actual[i])) << what << ": component " << i << " was never written";
    EXPECT_NEAR(actual[i], reference[i], rel_tol * scale) << what << ": component " << i;
  }
}

/// \brief Run interaction through direct_sum with panel sizes 1, 3, 4, 8, 16 and the default, against the reference.
template <class DeviceInteraction, class HostInteraction>
void expect_every_panel_size_matches(const size_t num_targets, const size_t num_sources, const size_t num_components,
                                     const DeviceInteraction& device_interaction,
                                     const HostInteraction& host_interaction, const double rel_tol) {
  const std::vector<double> reference = serial_reference(num_targets, num_sources, host_interaction);
  const DeviceSpace space;
  expect_near_reference(run_direct_sum<1>(space, num_targets, num_sources, num_components, device_interaction),
                        reference, rel_tol, "panel size 1");
  expect_near_reference(run_direct_sum<3>(space, num_targets, num_sources, num_components, device_interaction),
                        reference, rel_tol, "panel size 3");
  expect_near_reference(run_direct_sum<4>(space, num_targets, num_sources, num_components, device_interaction),
                        reference, rel_tol, "panel size 4");
  expect_near_reference(run_direct_sum<8>(space, num_targets, num_sources, num_components, device_interaction),
                        reference, rel_tol, "panel size 8");
  expect_near_reference(run_direct_sum<16>(space, num_targets, num_sources, num_components, device_interaction),
                        reference, rel_tol, "panel size 16");
  expect_near_reference(
      run_direct_sum<impl::default_direct_sum_panel_size<DeviceSpace>>(space, num_targets, num_sources,
                                                                       num_components, device_interaction),
      reference, rel_tol, "default panel size");
}

/// \brief The host mirror of a device view.
HostView to_host(const DeviceView& view) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
}
//@}

TEST(DirectSum, ScalarSumsMatchTheSerialReference) {
  // 37 targets are not a multiple of 3, 4, 8, or 16, so every panel size also takes the partial-panel path.
  const DeviceView targets = make_points(37, 0.1);
  const DeviceView sources = make_points(53, 0.2);
  expect_every_panel_size_matches(37, 53, 1, InverseDistance<DeviceView>{targets, sources},
                                  InverseDistance<HostView>{to_host(targets), to_host(sources)}, 1e-14);
}

TEST(DirectSum, VectorSumsMatchTheSerialReference) {
  const DeviceView targets = make_points(29, 0.3);
  const DeviceView sources = make_points(41, 0.4);
  const DeviceView forces = make_points(41, 0.5);
  expect_every_panel_size_matches(29, 41, 3, Stokeslet<DeviceView>{targets, sources, forces},
                                  Stokeslet<HostView>{to_host(targets), to_host(sources), to_host(forces)}, 1e-14);
}

TEST(DirectSum, MatrixSumsMatchTheSerialReference) {
  const DeviceView targets = make_points(10, 0.6);
  const DeviceView sources = make_points(13, 0.7);
  expect_every_panel_size_matches(10, 13, 9, StokesletTensor<DeviceView>{targets, sources},
                                  StokesletTensor<HostView>{to_host(targets), to_host(sources)}, 1e-14);
}

TEST(DirectSum, CustomPassiveScalarSumsMatchTheSerialReference) {
  // Double-double sums keep their low parts: compare both halves.
  const DeviceView targets = make_points(11, 0.15);
  const DeviceView sources = make_points(17, 0.25);
  expect_every_panel_size_matches(11, 17, 2, InverseDistanceDD<DeviceView>{targets, sources},
                                  InverseDistanceDD<HostView>{to_host(targets), to_host(sources)}, 1e-30);
}

TEST(DirectSum, SharedPointsSkipTheirOwnPair) {
  // Targets and sources are the same points: the coincident pair contributes 0 through the branch-free guard.
  const DeviceView points = make_points(24, 0.35);
  expect_every_panel_size_matches(24, 24, 1, InverseDistance<DeviceView>{points, points},
                                  InverseDistance<HostView>{to_host(points), to_host(points)}, 1e-14);
}

TEST(DirectSum, PanelsLargerThanTheTargetCount) {
  // 3 targets fill only part of a panel of 4, 8, or 16; 8 targets fill a panel of 8 exactly.
  for (const size_t num_targets : {3, 8}) {
    const DeviceView targets = make_points(num_targets, 0.45);
    const DeviceView sources = make_points(19, 0.55);
    expect_every_panel_size_matches(num_targets, 19, 1, InverseDistance<DeviceView>{targets, sources},
                                    InverseDistance<HostView>{to_host(targets), to_host(sources)}, 1e-14);
  }
}

TEST(DirectSum, NoSourcesGivesZeroSums) {
  const DeviceView targets = make_points(6, 0.65);
  const DeviceView sources = make_points(0, 0.75);
  const std::vector<double> sums = run_direct_sum<4>(DeviceSpace(), 6, 0, 3,
                                                     Stokeslet<DeviceView>{targets, sources, sources});
  for (const double sum : sums) {
    EXPECT_EQ(sum, 0.0);
  }
}

TEST(DirectSum, NoTargetsRunsNothing) {
  const DeviceView points = make_points(5, 0.85);
  EXPECT_TRUE(run_direct_sum<4>(DeviceSpace(), 0, 5, 1, InverseDistance<DeviceView>{points, points}).empty());
}

#if defined(KOKKOS_ENABLE_SERIAL)
TEST(DirectSum, RunsOnTheGivenExecutionSpace) {
  // Serial need not be the host views' default execution space; direct_sum must run there anyway.
  const HostView targets = to_host(make_points(9, 0.95));
  const HostView sources = to_host(make_points(14, 0.05));
  const InverseDistance<HostView> interaction{targets, sources};
  expect_near_reference(run_direct_sum<4>(Kokkos::Serial(), 9, 14, 1, interaction),
                        serial_reference(9, 14, interaction), 1e-14, "Kokkos::Serial");
}
#endif

}  // namespace

}  // namespace mundy
