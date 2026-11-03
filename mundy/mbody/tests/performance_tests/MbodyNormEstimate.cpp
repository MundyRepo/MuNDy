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

/// \file
/// \brief The work of mbody steps by solver phase: the unilateral solve's estimate of its operator's largest
/// eigenvalue against the rest of the solve.
///
/// Rods, or fibers of rods joined tip to tail by fixed lengths, are placed at random without overlap, drawn toward the
/// box center by a force proportional to their distance from it, and stepped under local drag with contacts between
/// every pair within cutoff of touching at the start. Contacts are ReLCP pairs, or plain contacts for mlcp and slcp.
///
/// Usage: MbodyNormEstimate --mode mlcp|slcp|relcp|relcp_relin [--fibers N] [--rods-per-fiber F] [--steps S]
///          [--dt DT] [--load K] [--cutoff C] [--density D] [--rebuild-every R] [--seed SEED] [--csv FILE]
///
/// --rebuild-every R replaces the pair list every R steps by the same number of pairs closest to touching, each
/// rebuild a new integrator over the new pairs in the last one's storage (relcp modes only).
///
/// Work is counted through Kokkos Tools kernel callbacks, by kernel label: a mobility apply is a LocalDragMobilityOp
/// kernel, and it belongs to the estimate from a Lanczos or power kernel until the next MPRGP or PGD kernel. A Lanczos
/// iteration is a LanczosStrategy::alpha kernel and an MPRGP step an MPRGP::proportioning_test kernel. The counts
/// follow those labels. Each step's row goes to the CSV; the totals go to stdout.

// C++ core
#include <algorithm>  // for std::clamp, std::sort, std::min
#include <chrono>     // for std::chrono::steady_clock
#include <cmath>      // for std::sqrt, std::cbrt, std::acos, std::cos, std::sin
#include <cstdio>     // for std::printf, std::fprintf, std::FILE
#include <cstring>    // for std::strncmp, std::strcmp
#include <map>        // for std::map
#include <random>     // for std::mt19937_64
#include <stdexcept>  // for std::invalid_argument
#include <string>     // for std::string, std::stod, std::stoi
#include <utility>    // for std::pair
#include <vector>     // for std::vector

// External
#include <Kokkos_Core.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy
#include <mundy_mbody/KokkosMbody.hpp>  // for mundy::mbody::{RodViews, make_mixed_*_integrator, solve_step, ...}

namespace mundy {

namespace mbody {

namespace {

using Space = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::DefaultHostExecutionSpace;

//! \name Work counters
//@{

enum Phase { OTHER = 0, ESTIMATE = 1, SOLVE = 2 };

struct Counts {
  long mobility[3] = {0, 0, 0};
  long lanczos_iters = 0;
  long power_iters = 0;
  long mprgp_steps = 0;
  long estimates = 0;
};

Counts counts;
int phase = OTHER;

bool starts_with(const char* name, const char* prefix) {
  return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

void count_kernel(const char* name, const uint32_t, uint64_t*) {
  if (starts_with(name, "LanczosStrategy::") || starts_with(name, "PowerStrategy::")) {
    counts.estimates += phase != ESTIMATE ? 1 : 0;
    phase = ESTIMATE;
    counts.lanczos_iters += std::strcmp(name, "LanczosStrategy::alpha") == 0 ? 1 : 0;
    counts.power_iters += std::strcmp(name, "PowerStrategy::step") == 0 ? 1 : 0;
  } else if (starts_with(name, "MPRGP::") || starts_with(name, "PGD::")) {
    phase = SOLVE;
    counts.mprgp_steps += std::strcmp(name, "MPRGP::proportioning_test") == 0 ? 1 : 0;
  } else if (starts_with(name, "LocalDragMobilityOp::apply")) {
    ++counts.mobility[phase];
  } else if (starts_with(name, "compute_") || starts_with(name, "count_overlaps") || starts_with(name, "append_") ||
             starts_with(name, "advance_poses") || starts_with(name, "study_")) {
    phase = OTHER;
  }
}

void set_counting(bool on) {
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(on ? count_kernel : nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(on ? count_kernel : nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(on ? count_kernel : nullptr);
}
//@}

//! \name Scene
//@{

struct Options {
  std::string mode = "relcp";
  int num_fibers = 1000;
  int rods_per_fiber = 1;
  int num_steps = 100;
  double dt = 0.01;
  double load = 1.0;
  double cutoff = 0.25;
  double density = 8.0;
  int rebuild_every = 0;
  unsigned seed = 42;
  std::string csv;
};

constexpr double kLength = 1.0;
constexpr double kRadius = 0.05;

struct Segment {
  double center[3];
  double tangent[3];
};

/// \brief The distance between two segments' centerlines of length kLength (Ericson, Real-Time Collision Detection,
/// 5.1.9).
double centerline_distance(const Segment& a, const Segment& b) {
  double d1[3], d2[3], r[3], p1[3], p2[3];
  for (int k = 0; k < 3; ++k) {
    d1[k] = a.tangent[k] * kLength;
    d2[k] = b.tangent[k] * kLength;
    p1[k] = a.center[k] - 0.5 * d1[k];
    p2[k] = b.center[k] - 0.5 * d2[k];
    r[k] = p1[k] - p2[k];
  }
  const auto dot = [](const double* x, const double* y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
  const double aa = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r), c = dot(d1, r), bb = dot(d1, d2);
  const double den = aa * e - bb * bb;
  double s = den > 1e-14 ? std::clamp((bb * f - c * e) / den, 0.0, 1.0) : 0.0;
  double t = (bb * s + f) / e;
  if (t < 0.0) {
    t = 0.0;
    s = std::clamp(-c / aa, 0.0, 1.0);
  } else if (t > 1.0) {
    t = 1.0;
    s = std::clamp((bb - c) / aa, 0.0, 1.0);
  }
  double dd = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double x = p1[k] + s * d1[k] - p2[k] - t * d2[k];
    dd += x * x;
  }
  return std::sqrt(dd);
}

bool centers_near(const Segment& a, const Segment& b, double reach) {
  double d2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    d2 += (a.center[k] - b.center[k]) * (a.center[k] - b.center[k]);
  }
  return d2 < (kLength + reach) * (kLength + reach);
}

/// \brief Fibers placed one at a time at a random center and direction, each rod bent from the last by a small random
/// turn, rejected if any rod comes within 0.02 of touching a placed one.
std::vector<Segment> place_fibers(const Options& opts, double box, double gap) {
  std::mt19937_64 rng(opts.seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> normal;
  const auto unit = [&](double* v) {
    double nn = 0.0;
    for (int k = 0; k < 3; ++k) {
      nn += v[k] * v[k];
    }
    for (int k = 0; k < 3; ++k) {
      v[k] /= std::sqrt(nn);
    }
  };
  std::vector<Segment> segments;
  const size_t n = static_cast<size_t>(opts.num_fibers * opts.rods_per_fiber);
  while (segments.size() < n) {
    std::vector<Segment> fiber;
    double t[3] = {normal(rng), normal(rng), normal(rng)};
    unit(t);
    double c[3] = {box * uniform(rng), box * uniform(rng), box * uniform(rng)};
    for (int s = 0; s < opts.rods_per_fiber; ++s) {
      Segment g;
      for (int k = 0; k < 3; ++k) {
        g.center[k] = c[k];
        g.tangent[k] = t[k];
      }
      fiber.push_back(g);
      double t2[3] = {t[0] + 0.2 * normal(rng), t[1] + 0.2 * normal(rng), t[2] + 0.2 * normal(rng)};
      unit(t2);
      for (int k = 0; k < 3; ++k) {
        c[k] += (0.5 * kLength + gap) * t[k] + 0.5 * kLength * t2[k];
        t[k] = t2[k];
      }
    }
    bool clear = true;
    for (const Segment& s : fiber) {
      for (const Segment& o : segments) {
        if (centers_near(o, s, 2 * kRadius) && centerline_distance(o, s) < 2 * kRadius + 0.02) {
          clear = false;
          break;
        }
      }
      if (!clear) {
        break;
      }
    }
    if (clear) {
      segments.insert(segments.end(), fiber.begin(), fiber.end());
    }
  }
  return segments;
}

/// \brief Every pair of rods within cutoff of touching, nearest first, except neighbors along a fiber.
std::vector<std::pair<int, int>> pairs_within(const std::vector<Segment>& segments, int rods_per_fiber, double cutoff) {
  std::vector<std::pair<double, std::pair<int, int>>> found;
  const int n = static_cast<int>(segments.size());
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (i / rods_per_fiber == j / rods_per_fiber && j - i <= 1) {
        continue;
      }
      if (centers_near(segments[i], segments[j], 2 * kRadius + cutoff)) {
        const double gap = centerline_distance(segments[i], segments[j]) - 2 * kRadius;
        if (gap < cutoff) {
          found.push_back({gap, {i, j}});
        }
      }
    }
  }
  std::sort(found.begin(), found.end());
  std::vector<std::pair<int, int>> pairs;
  for (const auto& [gap, pair] : found) {
    pairs.push_back(pair);
  }
  return pairs;
}

/// \brief The segments of rods' current configuration.
std::vector<Segment> segments_of(const RodViews<Space>& rods) {
  const auto host = create_mirror_view_and_copy(Host{}, rods);
  std::vector<Segment> segments(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    const Vector3d t = Quaterniond(host.orientation(i)) * Vector3d{0.0, 0.0, 1.0};
    for (int k = 0; k < 3; ++k) {
      segments[i].center[k] = host.center(i)[k];
      segments[i].tangent[k] = t[k];
    }
  }
  return segments;
}

template <typename Contacts>
Contacts make_contacts(const std::vector<std::pair<int, int>>& pairs) {
  Contacts contacts(pairs.size());
  for (size_t p = 0; p < pairs.size(); ++p) {
    contacts.rod_i(p) = pairs[p].first;
    contacts.rod_j(p) = pairs[p].second;
  }
  return contacts;
}
//@}

//! \name Stepping
//@{

struct Totals {
  long sequence_iters = 0;
  long unconverged_steps = 0;
  double seconds = 0.0;
};

/// \brief Whether a step's sequence, and the unilateral solve of its returned linearization, converged.
bool step_converged(const MixedLCPResult& result) {
  return result.converged;
}
template <typename SequenceResult>
bool step_converged(const SequenceResult& result) {
  return result.converged && result.accepted_lcp_result.converged;
}

/// \brief One step of integrator under the central load, its counts appended to csv.
template <typename Integrator, typename Config>
void step(const Integrator& integrator, const Config& cfg, const Options& opts, double mid, int index, Totals& totals,
          std::FILE* csv) {
  const RodViews<Space>& rods = integrator.rods();
  const auto force_torque = rods.force_torque_view();
  const auto velocity_omega = rods.velocity_omega_view();
  const auto center = rods.center_view();
  const double k = opts.load;
  Kokkos::parallel_for(
      "study_load", Kokkos::RangePolicy<Space>(0, rods.size()), KOKKOS_LAMBDA(const int i) {
        for (int c = 0; c < 3; ++c) {
          force_torque(6 * i + c) = -k * (center(3 * i + c) - mid);
          force_torque(6 * i + 3 + c) = 0.0;
        }
        for (int c = 0; c < 6; ++c) {
          velocity_omega(6 * i + c) = 0.0;
        }
      });
  Kokkos::fence();
  const Counts before = counts;
  const auto start = std::chrono::steady_clock::now();
  const auto result = solve_step(integrator, cfg);
  Kokkos::fence();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  advance_rods(rods, opts.dt);

  const bool converged = step_converged(result);
  totals.sequence_iters += result.num_iters;
  totals.unconverged_steps += converged ? 0 : 1;
  totals.seconds += seconds;
  if (csv) {
    std::fprintf(csv, "%d,%u,%d,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.6e\n", index, result.num_iters, converged ? 1 : 0,
                 counts.mobility[ESTIMATE] - before.mobility[ESTIMATE], counts.mobility[SOLVE] - before.mobility[SOLVE],
                 counts.mobility[OTHER] - before.mobility[OTHER], counts.lanczos_iters - before.lanczos_iters,
                 counts.power_iters - before.power_iters, counts.mprgp_steps - before.mprgp_steps,
                 counts.estimates - before.estimates, seconds);
  }
}

/// \brief opts.num_steps steps of the integrator make(constraints) makes, each rebuild through rebuild(constraints,
/// last integrator).
template <typename Contacts, typename Make, typename Rebuild, typename Config>
void run(const Options& opts, const RodViews<Space>& rods, const FixedLengthViews<Host>& links_h,
         std::vector<std::pair<int, int>> pairs, const Make& make, const Rebuild& rebuild, const Config& cfg,
         double mid, Totals& totals, std::FILE* csv) {
  auto constraints = create_mirror_view_and_copy(Space{}, make_constraint_set(links_h, make_contacts<Contacts>(pairs)));
  auto integrator = make(constraints);
  for (int s = 0; s < opts.num_steps; ++s) {
    if (opts.rebuild_every > 0 && s > 0 && s % opts.rebuild_every == 0) {
      // The same number of pairs, nearest first, each keeping its lambda if it was a pair before.
      set_counting(false);
      auto rebuilt = pairs_within(segments_of(rods), opts.rods_per_fiber, 4.0 * opts.cutoff);
      MUNDY_THROW_REQUIRE(rebuilt.size() >= pairs.size(), std::runtime_error, "MbodyNormEstimate: too few pairs.");
      rebuilt.resize(pairs.size());
      const auto old_lambda = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, get<rebind_space_t<Contacts, Space>>(constraints).lambda_view());
      std::map<std::pair<int, int>, double> lambda_of;
      for (size_t p = 0; p < pairs.size(); ++p) {
        lambda_of[pairs[p]] = old_lambda(p);
      }
      Contacts contacts_h = make_contacts<Contacts>(rebuilt);
      for (size_t p = 0; p < rebuilt.size(); ++p) {
        const auto it = lambda_of.find(rebuilt[p]);
        contacts_h.lambda(p) = it == lambda_of.end() ? 0.0 : it->second;
      }
      pairs = rebuilt;
      constraints = create_mirror_view_and_copy(Space{}, make_constraint_set(links_h, contacts_h));
      integrator = rebuild(constraints, integrator);
      set_counting(true);
    }
    step(integrator, cfg, opts, mid, s, totals, csv);
  }
}
//@}

Options parse(int argc, char** argv) {
  Options opts;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string value = argv[i + 1];
    if (key == "--mode") {
      opts.mode = value;
    } else if (key == "--fibers") {
      opts.num_fibers = std::stoi(value);
    } else if (key == "--rods-per-fiber") {
      opts.rods_per_fiber = std::stoi(value);
    } else if (key == "--steps") {
      opts.num_steps = std::stoi(value);
    } else if (key == "--dt") {
      opts.dt = std::stod(value);
    } else if (key == "--load") {
      opts.load = std::stod(value);
    } else if (key == "--cutoff") {
      opts.cutoff = std::stod(value);
    } else if (key == "--density") {
      opts.density = std::stod(value);
    } else if (key == "--rebuild-every") {
      opts.rebuild_every = std::stoi(value);
    } else if (key == "--seed") {
      opts.seed = static_cast<unsigned>(std::stoul(value));
    } else if (key == "--csv") {
      opts.csv = value;
    } else {
      throw std::invalid_argument("MbodyNormEstimate: unknown option " + key);
    }
  }
  return opts;
}

void study(const Options& opts) {
  const int n = opts.num_fibers * opts.rods_per_fiber;
  const double gap = 2 * kRadius + 0.02;
  const double box = std::cbrt(n / opts.density);
  const std::vector<Segment> segments = place_fibers(opts, box, gap);
  const std::vector<std::pair<int, int>> pairs = pairs_within(segments, opts.rods_per_fiber, opts.cutoff);

  RodViews<Host> rods_h(n);
  for (int i = 0; i < n; ++i) {
    // the rotation taking z to the tangent, about z x t
    const Segment& s = segments[i];
    double ax = -s.tangent[1], ay = s.tangent[0];
    const double an = std::sqrt(ax * ax + ay * ay);
    const double angle = std::acos(std::clamp(s.tangent[2], -1.0, 1.0));
    ax = an > 1e-12 ? ax / an : 1.0;
    ay = an > 1e-12 ? ay / an : 0.0;
    auto q = rods_h.orientation(i);
    q[0] = std::cos(0.5 * angle);
    q[1] = std::sin(0.5 * angle) * ax;
    q[2] = std::sin(0.5 * angle) * ay;
    q[3] = 0.0;
    auto c = rods_h.center(i);
    for (int k = 0; k < 3; ++k) {
      c[k] = s.center[k];
    }
    rods_h.radius(i) = kRadius;
    rods_h.length(i) = kLength;
  }
  const int num_links = opts.num_fibers * (opts.rods_per_fiber - 1);
  FixedLengthViews<Host> links_h(num_links);
  for (int f = 0, l = 0; f < opts.num_fibers; ++f) {
    for (int s = 0; s + 1 < opts.rods_per_fiber; ++s, ++l) {
      links_h.rod_i(l) = f * opts.rods_per_fiber + s;
      links_h.rod_j(l) = f * opts.rods_per_fiber + s + 1;
      links_h.body_offset_i(l) = Vector3d{0.0, 0.0, 0.5 * kLength};
      links_h.body_offset_j(l) = Vector3d{0.0, 0.0, -0.5 * kLength};
      links_h.rest_length(l) = gap;
    }
  }
  const auto rods = create_mirror_view_and_copy(Space{}, rods_h);

  MixedLCPConfig lcp;
  lcp.outer_tol = 1e-6;
  lcp.max_outer_iters = 50000;
  lcp.cg_tol = 1e-8;
  lcp.max_cg_iters = 2000;
  const LocalDragMobility model{.viscosity = 1.0};
  std::printf("rods %d links %d pairs %zu box %.3f mode %s\n", n, num_links, pairs.size(), box, opts.mode.c_str());

  std::FILE* csv = opts.csv.empty() ? nullptr : std::fopen(opts.csv.c_str(), "w");
  if (csv) {
    std::fprintf(csv,
                 "step,sequence_iters,converged,estimate_mobility,solve_mobility,other_mobility,lanczos_iters,"
                 "power_iters,mprgp_steps,estimates,seconds\n");
  }
  Totals totals;
  const double mid = 0.5 * box;
  set_counting(true);
  // mlcp and slcp never rebuild
  const auto unused_rebuild = [](const auto& make) {
    return [make](const auto& constraints, const auto&) { return make(constraints); };
  };
  if (opts.mode == "mlcp" || opts.mode == "slcp") {
    MUNDY_THROW_REQUIRE(opts.rebuild_every == 0, std::invalid_argument, "MbodyNormEstimate: rebuilds are relcp only.");
    if (opts.mode == "mlcp") {
      const auto make = [&](const auto& constraints) {
        return make_mixed_lcp_integrator(rods, constraints, model, opts.dt);
      };
      run<ContactViews<Host>>(opts, rods, links_h, pairs, make, unused_rebuild(make), lcp, mid, totals, csv);
    } else {
      const auto make = [&](const auto& constraints) {
        return make_mixed_slcp_integrator(rods, constraints, model, opts.dt);
      };
      run<ContactViews<Host>>(opts, rods, links_h, pairs, make, unused_rebuild(make),
                              MixedSLCPConfig{lcp, 20, 1e-6, 1e-6}, mid, totals, csv);
    }
  } else if (opts.mode == "relcp" || opts.mode == "relcp_relin") {
    const auto make = [&](const auto& constraints) {
      return make_mixed_relcp_integrator(rods, constraints, model, opts.dt);
    };
    const auto rebuild = [&](const auto& constraints, const auto& last) {
      return make_mixed_relcp_integrator(rods, constraints, model, opts.dt, NoPreconditioner{}, last.workspace());
    };
    run<ReLCPContactViews<Host>>(opts, rods, links_h, pairs, make, rebuild,
                                 MixedReLCPConfig{lcp, 20, 1e-6, 1e-6, opts.mode == "relcp_relin"}, mid, totals, csv);
  } else {
    throw std::invalid_argument("MbodyNormEstimate: unknown mode " + opts.mode);
  }
  set_counting(false);
  if (csv) {
    std::fclose(csv);
  }

  const long total = counts.mobility[OTHER] + counts.mobility[ESTIMATE] + counts.mobility[SOLVE];
  std::printf(
      "TOTAL sequence_iters %ld unconverged_steps %ld seconds %.3f mobility_estimate %ld mobility_solve %ld "
      "mobility_other %ld mobility_total %ld estimate_share %.4f lanczos_iters %ld power_iters %ld mprgp_steps %ld "
      "estimates %ld\n",
      totals.sequence_iters, totals.unconverged_steps, totals.seconds, counts.mobility[ESTIMATE],
      counts.mobility[SOLVE], counts.mobility[OTHER], total,
      total > 0 ? static_cast<double>(counts.mobility[ESTIMATE]) / total : 0.0, counts.lanczos_iters,
      counts.power_iters, counts.mprgp_steps, counts.estimates);
}

}  // namespace

}  // namespace mbody

}  // namespace mundy

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  mundy::mbody::study(mundy::mbody::parse(argc, argv));
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
