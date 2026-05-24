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

/// \file PerfTestNeighborList.cpp
/// \brief Nanobench performance benchmarks for ArborX1dNeighborList and ArborX2dNeighborList.
///
/// Usage: PerfTestNeighborList [--simple]
///   (default) Full nanobench output with per-variant tables and scaling fits.
///   --simple   Compact summary: one table per benchmark phase showing median time per
///              operation for each (variant, N) pair plus best-fit Big-O class.
///
/// \section Geometry
/// N spheres are placed at Philox-generated positions in a unit-density cubic domain
/// [0, cbrt(N)]³ (L = cbrt(N) keeps number density ρ = 1 particle / unit³ exactly).
/// Each sphere carries an AABB of half-width kDetectRadius = cbrt(kTargetNeighbors+1)/4
/// ≈ 0.6300.  Two AABBs overlap iff their centres are within ∞-norm 2r = 1.2600, giving
/// E[neighbors] = (4r)³ − 1 = kTargetNeighbors = 15 at unit density, independent of N.
///
/// \section List variants under test
///   Full list: ExcludeSelfInteraction only.  Every overlapping pair (t,s) and its
///              reverse (s,t) both appear; each target owns its own CSR row.
///   Half list: ExcludeSelfInteraction + ExcludeSymmetricDuplicates.  Only the (t,s)
///              orientation where src_entity > trg_entity is retained, halving pair count.
///
/// \section Benchmark phases
///   1. Construction  — build the list from scratch each call.
///   2. Target loop   — for_each_target_with_neighbors, empty body.
///   3. Pair loop     — for_each_neighbor_pair, empty body.
///   4. Global Kokkos reduction — for_each_target_with_neighbors_reduce and
///      for_each_neighbor_pair_reduce computing a scalar sum over all pairs.
///   5. Per-pair atomic into target — single atomic_add per pair.
///   6. Bilateral atomic (half list)  — two atomic_adds per pair (target + source).
///
/// \section N sweep
/// kNValues = {250, 1000, 4000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000}.
/// Batch counts are set to the operation count per call so nanobench reports
/// time-per-operation.

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// External
#include <nanobench.h>

// Kokkos / ArborX
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/EntitySorterBase.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy search
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_ARBORX

#ifdef HAVE_MUNDYSEARCH_ARBORX

#include <mundy_search/ArborX1dNeighborList.hpp>
#include <mundy_search/ArborX2dNeighborList.hpp>
#include <mundy_search/Excluder.hpp>
#include <mundy_search/ForEach.hpp>
#include <mundy_search/NeighborListBuilder.hpp>
#include <mundy_search/Neighbors.hpp>
#include <mundy_search/impl/ArborXSearchBoxes.hpp>

// Mundy utils
#include <mundy_utils/rng.hpp>

// =============================================================================
// Constants
// =============================================================================

// Desired average neighbor count per sphere at unit number density.
// The AABB half-width r is derived from E[neighbors] = (4r)³ − 1 = kTargetNeighbors,
// giving r = cbrt(kTargetNeighbors + 1) / 4 = cbrt(16) / 4 ≈ 0.6300.
static constexpr int   kTargetNeighbors = 15;
static constexpr float kDetectRadius = 0.6300f;  // cbrt(kTargetNeighbors + 1) / 4
static constexpr int kNValues[] = {1000, 8000, 64000, 256000, 512000};
static constexpr int kNumNValues = sizeof(kNValues) / sizeof(kNValues[0]);

// =============================================================================
// Type aliases
// =============================================================================

using HostSpace = Kokkos::HostSpace;
using HostExec = Kokkos::DefaultHostExecutionSpace;
using SearchBoxes = mundy::search::impl::ArborXSearchBoxesT<HostSpace>;
using List1d = mundy::search::ArborX1dNeighborList<HostSpace>;
using List2d = mundy::search::ArborX2dNeighborList<HostSpace>;
using PosView = Kokkos::View<float**, HostSpace>;
using ForceView = Kokkos::View<float**, HostSpace>;

// =============================================================================
// Minimal mesh + search-box fixture
// =============================================================================

struct PerfFixture {
  int N;
  float L;  // domain side length = cbrt(N) for unit number density
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  std::vector<stk::mesh::Entity> nodes;
  PosView pos;        // [N x 3] sphere centres
  SearchBoxes boxes;  // AABB half-width = kDetectRadius, selector = universal
};

static PerfFixture build_fixture(int N) {
  PerfFixture f;
  f.N = N;
  f.L = std::cbrt(static_cast<float>(N));

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  f.meta = builder.create_meta_data();
  f.meta->use_simple_fields();
  f.meta->commit();  // must be committed before BulkData is created
  f.bulk = builder.create(f.meta);
  f.bulk->modification_begin();
  for (int id = 1; id <= N; ++id) f.bulk->declare_node(id);
  f.bulk->modification_end();

  f.nodes.resize(N);
  for (int id = 1; id <= N; ++id)
    f.nodes[id - 1] = f.bulk->get_entity(stk::topology::NODE_RANK, id);

  f.pos = PosView("pos", N, 3);
  for (int i = 0; i < N; ++i) {
    openrand::Philox rng = mundy::make_philox(0xBEEF, static_cast<uint32_t>(i));
    f.pos(i, 0) = rng.uniform<float>(0.f, f.L);
    f.pos(i, 1) = rng.uniform<float>(0.f, f.L);
    f.pos(i, 2) = rng.uniform<float>(0.f, f.L);
  }

  Kokkos::View<ArborX::Box*, HostSpace> arb_boxes("arb_boxes", N);
  Kokkos::View<stk::mesh::Entity*, HostSpace> entities("entities", N);
  for (int i = 0; i < N; ++i) {
    const float cx = f.pos(i, 0), cy = f.pos(i, 1), cz = f.pos(i, 2);
    arb_boxes(i) = ArborX::Box{ArborX::Point{cx - kDetectRadius, cy - kDetectRadius, cz - kDetectRadius},
                                ArborX::Point{cx + kDetectRadius, cy + kDetectRadius, cz + kDetectRadius}};
    entities(i) = f.nodes[i];
  }
  const stk::mesh::Selector universal = f.meta->universal_part();
  f.boxes = SearchBoxes{universal, arb_boxes, entities};
  return f;
}

// =============================================================================
// List builder helpers
// =============================================================================

static List1d build_full_1d(const PerfFixture& f, bool sort_neighbors = false) {
  return mundy::search::make_neighbor_list_builder<List1d>()
      .exec_space(HostExec{})
      .target_input(f.boxes)
      .source_input(f.boxes)
      .exclude(mundy::search::ExcludeSelfInteraction{})
      .sort_neighbors(sort_neighbors)
      .build(*f.bulk);
}

static List2d build_full_2d(const PerfFixture& f, bool sort_neighbors = false) {
  return mundy::search::make_neighbor_list_builder<List2d>()
      .exec_space(HostExec{})
      .target_input(f.boxes)
      .source_input(f.boxes)
      .exclude(mundy::search::ExcludeSelfInteraction{})
      .sort_neighbors(sort_neighbors)
      .build(*f.bulk);
}

static List1d build_half_1d(const PerfFixture& f, bool sort_neighbors = false) {
  return mundy::search::make_neighbor_list_builder<List1d>()
      .exec_space(HostExec{})
      .target_input(f.boxes)
      .source_input(f.boxes)
      .exclude(mundy::search::ExcludeSelfInteraction{})
      .exclude(mundy::search::ExcludeSymmetricDuplicates{})
      .sort_neighbors(sort_neighbors)
      .build(*f.bulk);
}

static List2d build_half_2d(const PerfFixture& f, bool sort_neighbors = false) {
  return mundy::search::make_neighbor_list_builder<List2d>()
      .exec_space(HostExec{})
      .target_input(f.boxes)
      .source_input(f.boxes)
      .exclude(mundy::search::ExcludeSelfInteraction{})
      .exclude(mundy::search::ExcludeSymmetricDuplicates{})
      .sort_neighbors(sort_neighbors)
      .build(*f.bulk);
}

// =============================================================================
// Target-ordering diagnostic
//
// Measures how spatially coherent the target sequence is by comparing the
// average distance between consecutively-indexed targets to the same metric
// under three reference orderings:
//   current  — entity creation order (STK assigns IDs 1..N; positions are random)
//   x-sorted — targets sorted by ascending x coordinate
//   morton   — targets sorted by 3D Morton (Z-curve) code
//   random   — sampled random pairs (the expected distance for a random permutation)
//
// ratio = avg_consecutive / avg_random
//   ≈ 1  → ordering is essentially random w.r.t. spatial position
//   << 1 → ordering has spatial coherence (nearby indices ↔ nearby positions)
// =============================================================================

// Interleave the low 21 bits of three integers into a 63-bit Morton code.
static uint64_t morton_expand(uint64_t v) {
  v &= 0x1FFFFFu;
  v = (v | (v << 32u)) & 0x001F00000000FFFFu;
  v = (v | (v << 16u)) & 0x001F0000FF0000FFu;
  v = (v | (v <<  8u)) & 0x100F00F00F00F00Fu;
  v = (v | (v <<  4u)) & 0x10C30C30C30C30C3u;
  v = (v | (v <<  2u)) & 0x1249249249249249u;
  return v;
}

static uint64_t morton3d(float x, float y, float z, float L) {
  constexpr float kScale = static_cast<float>((1u << 21u) - 1u);
  const uint64_t xi = static_cast<uint64_t>(std::max(0.f, std::min(x / L, 1.f)) * kScale);
  const uint64_t yi = static_cast<uint64_t>(std::max(0.f, std::min(y / L, 1.f)) * kScale);
  const uint64_t zi = static_cast<uint64_t>(std::max(0.f, std::min(z / L, 1.f)) * kScale);
  return morton_expand(xi) | (morton_expand(yi) << 1u) | (morton_expand(zi) << 2u);
}

// Sorts entities within STK's bucket structure by ascending 3D Morton code.
// Positions are looked up by entity identifier so the original PosView (indexed by id-1)
// is used only as a read-only source during the comparison.
class ZMortonSorter : public stk::mesh::EntitySorterBase {
 public:
  ZMortonSorter(const PosView& pos, float L) : pos_(pos), L_(L) {}
  void sort(stk::mesh::BulkData& bulk, stk::mesh::EntityVector& entity_vector) const override {
    std::sort(entity_vector.begin(), entity_vector.end(),
              [this, &bulk](stk::mesh::Entity a, stk::mesh::Entity b) {
                const int ia = static_cast<int>(bulk.identifier(a)) - 1;
                const int ib = static_cast<int>(bulk.identifier(b)) - 1;
                return morton3d(pos_(ia, 0), pos_(ia, 1), pos_(ia, 2), L_) <
                       morton3d(pos_(ib, 0), pos_(ib, 1), pos_(ib, 2), L_);
              });
  }
 private:
  PosView pos_;
  float L_;
};

// Sort entities in STK's bucket ordering by Morton code, then rebuild the fixture's
// auxiliary arrays (nodes, pos, boxes) so that index i refers to the i-th Morton-sorted entity.
static void sort_fixture_targets_by_morton(PerfFixture& f) {
  f.bulk->sort_entities(ZMortonSorter{f.pos, f.L});

  // Rebuild f.nodes in the new STK bucket-iteration order.
  {
    int idx = 0;
    const stk::mesh::Selector universal = f.meta->universal_part();
    for (const auto* bucket : f.bulk->get_buckets(stk::topology::NODE_RANK, universal))
      for (stk::mesh::Entity e : *bucket)
        f.nodes[idx++] = e;
  }

  // Rebuild f.pos so that f.pos(i,...) holds the position of f.nodes[i].
  // The original f.pos was indexed by (entity_id - 1); use the identifier to remap.
  PosView new_pos("pos", f.N, 3);
  for (int i = 0; i < f.N; ++i) {
    const int eid = static_cast<int>(f.bulk->identifier(f.nodes[i])) - 1;
    new_pos(i, 0) = f.pos(eid, 0);
    new_pos(i, 1) = f.pos(eid, 1);
    new_pos(i, 2) = f.pos(eid, 2);
  }
  f.pos = new_pos;

  // Rebuild search boxes from the new ordering.
  Kokkos::View<ArborX::Box*, HostSpace> arb_boxes("arb_boxes", f.N);
  Kokkos::View<stk::mesh::Entity*, HostSpace> entities("entities", f.N);
  for (int i = 0; i < f.N; ++i) {
    const float cx = f.pos(i, 0), cy = f.pos(i, 1), cz = f.pos(i, 2);
    arb_boxes(i) = ArborX::Box{ArborX::Point{cx - kDetectRadius, cy - kDetectRadius, cz - kDetectRadius},
                                ArborX::Point{cx + kDetectRadius, cy + kDetectRadius, cz + kDetectRadius}};
    entities(i) = f.nodes[i];
  }
  f.boxes = SearchBoxes{f.meta->universal_part(), arb_boxes, entities};
}

static void diagnose_target_ordering() {
  std::cout << "\n=== Target Ordering Diagnostic ===\n";
  std::cout << "  Measures avg distance between consecutively-indexed targets.\n";
  std::cout << "  ratio = avg_consecutive / avg_random_pair  (1=random, <<1=spatially coherent)\n\n";

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    const auto f = build_fixture(N);

    // ---- helper: average distance along a permutation ----
    auto avg_consec_dist = [&](const std::vector<int>& order) {
      double sum = 0.0;
      for (int i = 0; i < N - 1; ++i) {
        const int a = order[i], b = order[i + 1];
        const double dx = f.pos(b, 0) - f.pos(a, 0);
        const double dy = f.pos(b, 1) - f.pos(a, 1);
        const double dz = f.pos(b, 2) - f.pos(a, 2);
        sum += std::sqrt(dx * dx + dy * dy + dz * dz);
      }
      return sum / (N - 1);
    };

    // ---- 1. current order (entity creation order = random positions) ----
    std::vector<int> current(N);
    std::iota(current.begin(), current.end(), 0);
    const double avg_current = avg_consec_dist(current);

    // ---- 2. x-sorted order ----
    std::vector<int> x_sorted = current;
    std::sort(x_sorted.begin(), x_sorted.end(),
              [&](int a, int b) { return f.pos(a, 0) < f.pos(b, 0); });
    const double avg_x = avg_consec_dist(x_sorted);

    // ---- 3. Morton-sorted order ----
    std::vector<int> morton_sorted = current;
    std::sort(morton_sorted.begin(), morton_sorted.end(), [&](int a, int b) {
      return morton3d(f.pos(a, 0), f.pos(a, 1), f.pos(a, 2), f.L) <
             morton3d(f.pos(b, 0), f.pos(b, 1), f.pos(b, 2), f.L);
    });
    const double avg_morton = avg_consec_dist(morton_sorted);

    // ---- 4. average random-pair distance (sample 2000 pairs) ----
    const int n_samp = std::min(N * (N - 1), 2000);
    double sum_rand = 0.0;
    for (int k = 0; k < n_samp; ++k) {
      const int i = k % N;
      const int j = (k * 6271 + 1337) % N;
      const double dx = f.pos(j, 0) - f.pos(i, 0);
      const double dy = f.pos(j, 1) - f.pos(i, 1);
      const double dz = f.pos(j, 2) - f.pos(i, 2);
      sum_rand += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    const double avg_rand = sum_rand / n_samp;

    const auto pct = [&](double d) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * d / avg_rand);
      return std::string(buf);
    };

    std::cout << "  N=" << std::setw(5) << N
              << "  L=" << std::fixed << std::setprecision(2) << f.L
              << "  r_det=" << kDetectRadius << "\n";
    std::cout << "    avg dist  current order : " << std::setprecision(4) << avg_current
              << "  (ratio " << pct(avg_current) << " of random)\n";
    std::cout << "    avg dist  x-sorted      : " << avg_x
              << "  (ratio " << pct(avg_x) << " of random)\n";
    std::cout << "    avg dist  morton-sorted : " << avg_morton
              << "  (ratio " << pct(avg_morton) << " of random)\n";
    std::cout << "    avg dist  random pairs  : " << avg_rand << "\n\n";
  }
}

// =============================================================================
// --simple mode helpers
// =============================================================================

static constexpr double kNoData = -1.0;

struct VariantSummary {
  std::string label;
  std::array<double, kNumNValues> ns_per_op{};
  std::string big_o;
};

// Extract the best-fit Big-O name from nanobench's complexityBigO() result.
// The vector is sorted by ascending normalised RMS, so front() is the best fit.
static std::string parse_big_o(const std::vector<ankerl::nanobench::BigO>& bigos) {
  if (bigos.empty()) return "?";
  return bigos.front().name();
}

// Collect median ns-per-op from each run (runs added in N order).
// result.median(elapsed) returns seconds per operation (time / batch).
static std::array<double, kNumNValues> collect_ns_per_op(const ankerl::nanobench::Bench& b) {
  std::array<double, kNumNValues> ns{};
  ns.fill(kNoData);
  const auto& results = b.results();
  for (int i = 0; i < kNumNValues && i < static_cast<int>(results.size()); ++i)
    ns[i] = results[i].median(ankerl::nanobench::Result::Measure::elapsed) * 1e9;
  return ns;
}

// Auto-scale: ns → "  12.3 ns", us → "  1.23 us", ms → "  1.23 ms".
// Always returns a fixed-width 9-char string for column alignment.
static std::string fmt_time(double ns) {
  if (ns < 0.0) return "   ---   ";
  char buf[24];
  if (ns < 1e3)
    std::snprintf(buf, sizeof(buf), "%6.1f ns", ns);
  else if (ns < 1e6)
    std::snprintf(buf, sizeof(buf), "%6.2f us", ns * 1e-3);
  else
    std::snprintf(buf, sizeof(buf), "%6.2f ms", ns * 1e-6);
  return buf;
}

/// \brief Options controlling which sorting passes are applied before and after list construction.
struct BenchOptions {
  bool simple         = false;  ///< Compact summary mode (suppresses nanobench full output).
  bool sort_targets   = false;  ///< Pre-sort STK entities by Morton code before building.
  bool sort_neighbors = false;  ///< Post-sort each target's neighbor row by source ordinal.
};

static void print_simple_section(const std::string& title, const std::string& unit,
                                  const std::vector<VariantSummary>& rows) {
  constexpr int kLabelW = 18;
  constexpr int kColW   = 12;
  std::cout << "\n[" << title << "]  (median " << unit << ")\n";
  // Header
  std::cout << "  " << std::left << std::setw(kLabelW) << "";
  for (int ni = 0; ni < kNumNValues; ++ni)
    std::cout << std::right << std::setw(kColW) << ("N=" + std::to_string(kNValues[ni]));
  std::cout << "  scaling\n";
  // Separator
  std::cout << "  " << std::string(kLabelW + kNumNValues * kColW + 12, '-') << "\n";
  // Data rows
  for (const auto& v : rows) {
    std::cout << "  " << std::left << std::setw(kLabelW) << (v.label + ":");
    for (int ni = 0; ni < kNumNValues; ++ni)
      std::cout << std::right << std::setw(kColW) << fmt_time(v.ns_per_op[ni]);
    std::cout << "  " << v.big_o << "\n";
  }
}

// =============================================================================
// Benchmark 1: Construction
//
// Each lambda call builds the list from scratch: includes BVH construction,
// spatial query, and CSR/2D fill passes.  Batch = 1 (one list per call).
// =============================================================================

static void bench_construction(const BenchOptions& opts) {
  // One Bench per variant so complexityBigO() fits each independently.
  // In normal mode output is routed to ostringstreams to avoid interleaving.
  // In simple mode output is suppressed (nullptr); results are read back programmatically.
  std::ostringstream out1f, out2f, out1h, out2h;
  ankerl::nanobench::Bench b1f, b2f, b1h, b2h;

  auto cfg = [&](ankerl::nanobench::Bench& b, std::ostringstream& os,
                 const char* title) -> ankerl::nanobench::Bench& {
    return b.output(opts.simple ? nullptr : static_cast<std::ostream*>(&os))
            .title(title)
            .unit("build")
            .relative(true)
            .performanceCounters(true)
            .warmup(2)
            .epochs(10)
            .minEpochIterations(2)
            .minEpochTime(std::chrono::milliseconds(5));
  };
  cfg(b1f, out1f, "Construction: 1d | Full");
  cfg(b2f, out2f, "Construction: 2d | Full");
  cfg(b1h, out1h, "Construction: 1d | Half");
  cfg(b2h, out2h, "Construction: 2d | Half");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const std::string tag = " N=" + std::to_string(N);

    b1f.complexityN(N).batch(1).run("1d | Full" + tag, [&] {
      ankerl::nanobench::doNotOptimizeAway(build_full_1d(f, opts.sort_neighbors));
    });
    b2f.complexityN(N).batch(1).run("2d | Full" + tag, [&] {
      ankerl::nanobench::doNotOptimizeAway(build_full_2d(f, opts.sort_neighbors));
    });
    b1h.complexityN(N).batch(1).run("1d | Half" + tag, [&] {
      ankerl::nanobench::doNotOptimizeAway(build_half_1d(f, opts.sort_neighbors));
    });
    b2h.complexityN(N).batch(1).run("2d | Half" + tag, [&] {
      ankerl::nanobench::doNotOptimizeAway(build_half_2d(f, opts.sort_neighbors));
    });
  }

  if (opts.simple) {
    print_simple_section("Construction", "ns/build", {
      {"1d Full", collect_ns_per_op(b1f), parse_big_o(b1f.complexityBigO())},
      {"2d Full", collect_ns_per_op(b2f), parse_big_o(b2f.complexityBigO())},
      {"1d Half", collect_ns_per_op(b1h), parse_big_o(b1h.complexityBigO())},
      {"2d Half", collect_ns_per_op(b2h), parse_big_o(b2h.complexityBigO())},
    });
  } else {
    std::cout << out1f.str() << out2f.str() << out1h.str() << out2h.str();
    std::cout << "Construction complexity fits:\n"
              << "  1d | Full : " << b1f.complexityBigO()
              << "  2d | Full : " << b2f.complexityBigO()
              << "  1d | Half : " << b1h.complexityBigO()
              << "  2d | Half : " << b2h.complexityBigO();
  }
}

// =============================================================================
// Benchmark 2: Iteration dispatch overhead (minimal body, no shared state)
//
// Measures the scheduling cost of the for_each loops with one independent write
// per item.  Each thread owns a distinct memory location so there is no atomic
// contention, which was the flaw in the original single-counter version.
//
// Target loop: one write dummy(t) = t per target  (batch = N).
// Pair loop:   one non-atomic += per pair into dummy(t)  (batch = list.size()).
//   The pair loop is contention-free because DeployFunctorOnNeighborPairs gives
//   each outer thread one target's full row; all writes for that row share the
//   same target slot and run on the same thread.
// =============================================================================

static void bench_iteration_overhead(const BenchOptions& opts) {
  std::ostringstream out[4];
  ankerl::nanobench::Bench b[4];

  auto cfg = [&](ankerl::nanobench::Bench& b, std::ostringstream& os,
                 const char* title) -> ankerl::nanobench::Bench& {
    return b.output(opts.simple ? nullptr : static_cast<std::ostream*>(&os))
            .title(title)
            .unit("item")
            .relative(true)
            .performanceCounters(true)
            .warmup(2)
            .epochs(10)
            .minEpochIterations(2)
            .minEpochTime(std::chrono::milliseconds(5));
  };
  cfg(b[0], out[0], "Iteration overhead: 1d | Full | Target loop");
  cfg(b[1], out[1], "Iteration overhead: 2d | Full | Target loop");
  cfg(b[2], out[2], "Iteration overhead: 1d | Full | Pair loop");
  cfg(b[3], out[3], "Iteration overhead: 2d | Full | Pair loop");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto full1 = build_full_1d(f, opts.sort_neighbors);
    const auto full2 = build_full_2d(f, opts.sort_neighbors);
    const std::string tag = " N=" + std::to_string(N);
    const size_t npairs1 = full1.size();
    const size_t npairs2 = full2.size();

    // One slot per target — no shared writes, no atomics required.
    Kokkos::View<size_t*, HostSpace> dummy("dummy", N);

    b[0].complexityN(N).batch(static_cast<size_t>(N)).run("1d | Full | Target" + tag, [&] {
      mundy::search::for_each_target_with_neighbors(
          HostExec{}, full1,
          KOKKOS_LAMBDA(const mundy::search::Neighbors<List1d>& nbrs) {
            dummy(nbrs.target_index()) = nbrs.target_index();
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(dummy(0));
    });

    b[1].complexityN(N).batch(static_cast<size_t>(N)).run("2d | Full | Target" + tag, [&] {
      mundy::search::for_each_target_with_neighbors(
          HostExec{}, full2,
          KOKKOS_LAMBDA(const mundy::search::Neighbors<List2d>& nbrs) {
            dummy(nbrs.target_index()) = nbrs.target_index();
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(dummy(0));
    });

    b[2].complexityN(N).batch(npairs1).run("1d | Full | Pair" + tag, [&] {
      Kokkos::deep_copy(dummy, size_t(0));
      mundy::search::for_each_neighbor_pair(
          HostExec{}, full1,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List1d>& pair) {
            dummy(pair.target_index()) += 1;  // safe: one thread owns each target's row
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(dummy(0));
    });

    b[3].complexityN(N).batch(npairs2).run("2d | Full | Pair" + tag, [&] {
      Kokkos::deep_copy(dummy, size_t(0));
      mundy::search::for_each_neighbor_pair(
          HostExec{}, full2,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List2d>& pair) {
            dummy(pair.target_index()) += 1;
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(dummy(0));
    });
  }

  if (opts.simple) {
    print_simple_section("Iteration overhead (target)", "ns/target", {
      {"1d Full target", collect_ns_per_op(b[0]), parse_big_o(b[0].complexityBigO())},
      {"2d Full target", collect_ns_per_op(b[1]), parse_big_o(b[1].complexityBigO())},
    });
    print_simple_section("Iteration overhead (pair)", "ns/pair", {
      {"1d Full pair",   collect_ns_per_op(b[2]), parse_big_o(b[2].complexityBigO())},
      {"2d Full pair",   collect_ns_per_op(b[3]), parse_big_o(b[3].complexityBigO())},
    });
  } else {
    for (int i = 0; i < 4; ++i) std::cout << out[i].str();
    std::cout << "Iteration overhead complexity fits:\n"
              << "  1d | Full | Target : " << b[0].complexityBigO()
              << "  2d | Full | Target : " << b[1].complexityBigO()
              << "  1d | Full | Pair   : " << b[2].complexityBigO()
              << "  2d | Full | Pair   : " << b[3].complexityBigO();
  }
}

// =============================================================================
// Benchmark 3: Global Kokkos reduction (full list)
//
// Exercises for_each_target_with_neighbors_reduce and for_each_neighbor_pair_reduce.
// Kernel: energy += sum_{(t,s)} ||pos_s - pos_t||²  (total squared displacement)
// The result is a single scalar — no per-entity write, no atomics.
// Batch = total pair count so ns/pair is comparable across variants.
//
// Two sub-benchmarks:
//   target-reduce: outer parallel_reduce over targets, serial inner walk per row.
//   pair-reduce:   outer parallel_reduce over targets, one contribution per pair.
// =============================================================================

static void bench_global_reduce(const BenchOptions& opts) {
  std::ostringstream out_tw1, out_tw2, out_pr1, out_pr2;
  ankerl::nanobench::Bench b_tw1, b_tw2, b_pr1, b_pr2;

  auto cfg = [&](ankerl::nanobench::Bench& b, std::ostringstream& os,
                 const char* title) -> ankerl::nanobench::Bench& {
    return b.output(opts.simple ? nullptr : static_cast<std::ostream*>(&os))
            .title(title)
            .unit("pair")
            .relative(true)
            .performanceCounters(true)
            .warmup(2)
            .epochs(10)
            .minEpochIterations(2)
            .minEpochTime(std::chrono::milliseconds(5));
  };
  cfg(b_tw1, out_tw1, "Global reduce (target): 1d | Full");
  cfg(b_tw2, out_tw2, "Global reduce (target): 2d | Full");
  cfg(b_pr1, out_pr1, "Global reduce (pair): 1d | Full");
  cfg(b_pr2, out_pr2, "Global reduce (pair): 2d | Full");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto full1 = build_full_1d(f, opts.sort_neighbors);
    const auto full2 = build_full_2d(f, opts.sort_neighbors);
    const size_t npairs1 = full1.size();
    const size_t npairs2 = full2.size();
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;

    // ---- for_each_target_with_neighbors_reduce ----
    b_tw1.complexityN(N).batch(npairs1).run("1d | Full | tw-reduce" + tag, [&] {
      float energy = 0.f;
      Kokkos::Sum<float> reducer(energy);
      mundy::search::for_each_target_with_neighbors_reduce(
          HostExec{}, full1,
          KOKKOS_LAMBDA(const mundy::search::Neighbors<List1d>& nbrs, float& update) {
            const size_t t = nbrs.target_index();
            for (size_t k = 0; k < nbrs.size(); ++k) {
              const size_t s = nbrs.source_index(k);
              const float dx = pos(s, 0) - pos(t, 0);
              const float dy = pos(s, 1) - pos(t, 1);
              const float dz = pos(s, 2) - pos(t, 2);
              update += dx * dx + dy * dy + dz * dz;
            }
          },
          reducer);
      ankerl::nanobench::doNotOptimizeAway(energy);
    });

    b_tw2.complexityN(N).batch(npairs2).run("2d | Full | tw-reduce" + tag, [&] {
      float energy = 0.f;
      Kokkos::Sum<float> reducer(energy);
      mundy::search::for_each_target_with_neighbors_reduce(
          HostExec{}, full2,
          KOKKOS_LAMBDA(const mundy::search::Neighbors<List2d>& nbrs, float& update) {
            const size_t t = nbrs.target_index();
            for (size_t k = 0; k < nbrs.size(); ++k) {
              const size_t s = nbrs.source_index(k);
              const float dx = pos(s, 0) - pos(t, 0);
              const float dy = pos(s, 1) - pos(t, 1);
              const float dz = pos(s, 2) - pos(t, 2);
              update += dx * dx + dy * dy + dz * dz;
            }
          },
          reducer);
      ankerl::nanobench::doNotOptimizeAway(energy);
    });

    // ---- for_each_neighbor_pair_reduce ----
    b_pr1.complexityN(N).batch(npairs1).run("1d | Full | pair-reduce" + tag, [&] {
      float energy = 0.f;
      Kokkos::Sum<float> reducer(energy);
      mundy::search::for_each_neighbor_pair_reduce(
          HostExec{}, full1,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List1d>& pair, float& update) {
            const size_t t = pair.target_index(), s = pair.source_index();
            const float dx = pos(s, 0) - pos(t, 0);
            const float dy = pos(s, 1) - pos(t, 1);
            const float dz = pos(s, 2) - pos(t, 2);
            update += dx * dx + dy * dy + dz * dz;
          },
          reducer);
      ankerl::nanobench::doNotOptimizeAway(energy);
    });

    b_pr2.complexityN(N).batch(npairs2).run("2d | Full | pair-reduce" + tag, [&] {
      float energy = 0.f;
      Kokkos::Sum<float> reducer(energy);
      mundy::search::for_each_neighbor_pair_reduce(
          HostExec{}, full2,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List2d>& pair, float& update) {
            const size_t t = pair.target_index(), s = pair.source_index();
            const float dx = pos(s, 0) - pos(t, 0);
            const float dy = pos(s, 1) - pos(t, 1);
            const float dz = pos(s, 2) - pos(t, 2);
            update += dx * dx + dy * dy + dz * dz;
          },
          reducer);
      ankerl::nanobench::doNotOptimizeAway(energy);
    });
  }

  if (opts.simple) {
    print_simple_section("Global reduce (target)", "ns/pair", {
      {"1d Full", collect_ns_per_op(b_tw1), parse_big_o(b_tw1.complexityBigO())},
      {"2d Full", collect_ns_per_op(b_tw2), parse_big_o(b_tw2.complexityBigO())},
    });
    print_simple_section("Global reduce (pair)", "ns/pair", {
      {"1d Full", collect_ns_per_op(b_pr1), parse_big_o(b_pr1.complexityBigO())},
      {"2d Full", collect_ns_per_op(b_pr2), parse_big_o(b_pr2.complexityBigO())},
    });
  } else {
    std::cout << out_tw1.str() << out_tw2.str() << out_pr1.str() << out_pr2.str();
    std::cout << "Global reduce complexity fits:\n"
              << "  1d | Full | tw  : " << b_tw1.complexityBigO()
              << "  2d | Full | tw  : " << b_tw2.complexityBigO()
              << "  1d | Full | pair: " << b_pr1.complexityBigO()
              << "  2d | Full | pair: " << b_pr2.complexityBigO();
  }
}

// =============================================================================
// Benchmark 4: Per-pair atomic into target (full list)
//
// Flat pair-loop accumulation: each pair (t,s) makes one atomic write to force[t].
// Kernel: atomic_add(force(t,xyz), pos(s,xyz) - pos(t,xyz))
// =============================================================================

static void bench_pair_atomic_target(const BenchOptions& opts) {
  std::ostringstream out1, out2;
  ankerl::nanobench::Bench b1, b2;

  auto cfg = [&](ankerl::nanobench::Bench& b, std::ostringstream& os,
                 const char* title) -> ankerl::nanobench::Bench& {
    return b.output(opts.simple ? nullptr : static_cast<std::ostream*>(&os))
            .title(title)
            .unit("pair")
            .relative(true)
            .performanceCounters(true)
            .warmup(2)
            .epochs(10)
            .minEpochIterations(2)
            .minEpochTime(std::chrono::milliseconds(5));
  };
  cfg(b1, out1, "Atomic into target: 1d | Full");
  cfg(b2, out2, "Atomic into target: 2d | Full");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto full1 = build_full_1d(f, opts.sort_neighbors);
    const auto full2 = build_full_2d(f, opts.sort_neighbors);
    const size_t npairs = full1.size();
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;

    ForceView force("force", N, 3);

    b1.complexityN(N).batch(npairs).run("1d | Full | Atomic target" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      mundy::search::for_each_neighbor_pair(
          HostExec{}, full1,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List1d>& pair) {
            const size_t t = pair.target_index();
            const size_t s = pair.source_index();
            Kokkos::atomic_add(&force(t, 0), pos(s, 0) - pos(t, 0));
            Kokkos::atomic_add(&force(t, 1), pos(s, 1) - pos(t, 1));
            Kokkos::atomic_add(&force(t, 2), pos(s, 2) - pos(t, 2));
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(force(0, 0));
    });

    b2.complexityN(N).batch(npairs).run("2d | Full | Atomic target" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      mundy::search::for_each_neighbor_pair(
          HostExec{}, full2,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List2d>& pair) {
            const size_t t = pair.target_index();
            const size_t s = pair.source_index();
            Kokkos::atomic_add(&force(t, 0), pos(s, 0) - pos(t, 0));
            Kokkos::atomic_add(&force(t, 1), pos(s, 1) - pos(t, 1));
            Kokkos::atomic_add(&force(t, 2), pos(s, 2) - pos(t, 2));
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(force(0, 0));
    });
  }

  if (opts.simple) {
    print_simple_section("Atomic into target (full list)", "ns/pair", {
      {"1d Full", collect_ns_per_op(b1), parse_big_o(b1.complexityBigO())},
      {"2d Full", collect_ns_per_op(b2), parse_big_o(b2.complexityBigO())},
    });
  } else {
    std::cout << out1.str() << out2.str();
    std::cout << "Atomic into target complexity fits:\n"
              << "  1d | Full : " << b1.complexityBigO()
              << "  2d | Full : " << b2.complexityBigO();
  }
}

// =============================================================================
// Benchmark 5: Bilateral atomic into target AND source (half list)
//
// Canonical half-list usage: each retained pair (t,s) makes two atomic writes.
// Kernel: atomic_add(force(t,xyz),  dx)
//         atomic_add(force(s,xyz), -dx)
// Batch = half-list pair count.
// =============================================================================

static void bench_bilateral_atomic(const BenchOptions& opts) {
  std::ostringstream out1, out2;
  ankerl::nanobench::Bench b1, b2;

  auto cfg = [&](ankerl::nanobench::Bench& b, std::ostringstream& os,
                 const char* title) -> ankerl::nanobench::Bench& {
    return b.output(opts.simple ? nullptr : static_cast<std::ostream*>(&os))
            .title(title)
            .unit("pair")
            .relative(true)
            .performanceCounters(true)
            .warmup(2)
            .epochs(10)
            .minEpochIterations(2)
            .minEpochTime(std::chrono::milliseconds(5));
  };
  cfg(b1, out1, "Bilateral atomic: 1d | Half");
  cfg(b2, out2, "Bilateral atomic: 2d | Half");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto half1 = build_half_1d(f, opts.sort_neighbors);
    const auto half2 = build_half_2d(f, opts.sort_neighbors);
    const size_t npairs1 = half1.size();
    const size_t npairs2 = half2.size();
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;

    ForceView force("force", N, 3);

    b1.complexityN(N).batch(npairs1).run("1d | Half | Bilateral" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      mundy::search::for_each_neighbor_pair(
          HostExec{}, half1,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List1d>& pair) {
            const size_t t = pair.target_index();
            const size_t s = pair.source_index();
            const float dx = pos(s, 0) - pos(t, 0);
            const float dy = pos(s, 1) - pos(t, 1);
            const float dz = pos(s, 2) - pos(t, 2);
            Kokkos::atomic_add(&force(t, 0),  dx);
            Kokkos::atomic_add(&force(t, 1),  dy);
            Kokkos::atomic_add(&force(t, 2),  dz);
            Kokkos::atomic_add(&force(s, 0), -dx);
            Kokkos::atomic_add(&force(s, 1), -dy);
            Kokkos::atomic_add(&force(s, 2), -dz);
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(force(0, 0));
    });

    b2.complexityN(N).batch(npairs2).run("2d | Half | Bilateral" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      mundy::search::for_each_neighbor_pair(
          HostExec{}, half2,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<List2d>& pair) {
            const size_t t = pair.target_index();
            const size_t s = pair.source_index();
            const float dx = pos(s, 0) - pos(t, 0);
            const float dy = pos(s, 1) - pos(t, 1);
            const float dz = pos(s, 2) - pos(t, 2);
            Kokkos::atomic_add(&force(t, 0),  dx);
            Kokkos::atomic_add(&force(t, 1),  dy);
            Kokkos::atomic_add(&force(t, 2),  dz);
            Kokkos::atomic_add(&force(s, 0), -dx);
            Kokkos::atomic_add(&force(s, 1), -dy);
            Kokkos::atomic_add(&force(s, 2), -dz);
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(force(0, 0));
    });
  }

  if (opts.simple) {
    print_simple_section("Bilateral atomic (half list)", "ns/pair", {
      {"1d Half", collect_ns_per_op(b1), parse_big_o(b1.complexityBigO())},
      {"2d Half", collect_ns_per_op(b2), parse_big_o(b2.complexityBigO())},
    });
  } else {
    std::cout << out1.str() << out2.str();
    std::cout << "Bilateral atomic complexity fits:\n"
              << "  1d | Half : " << b1.complexityBigO()
              << "  2d | Half : " << b2.complexityBigO();
  }
}

// =============================================================================
// Benchmark 6: N² brute force (baseline)
//
// All-pairs loop — provides a lower bound on per-pair reduction cost and a
// reference point for construction cost scaling.  Capped at N ≤ 1000.
// =============================================================================

static void bench_n2_baseline(const BenchOptions& opts) {
  std::ostringstream out;
  ankerl::nanobench::Bench b;
  b.output(opts.simple ? nullptr : static_cast<std::ostream*>(&out))
   .title("N² brute force baseline (N <= 1000)")
   .unit("pair")
   .relative(true)
   .performanceCounters(true)
   .warmup(2)
   .epochs(10)
   .minEpochIterations(2)
   .minEpochTime(std::chrono::milliseconds(5));

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    if (N > 1000) break;
    auto f = build_fixture(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const size_t npairs = static_cast<size_t>(N) * static_cast<size_t>(N - 1);
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;

    ForceView force("force", N, 3);

    b.complexityN(N).batch(npairs).run("N2 reduction" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      Kokkos::parallel_for(
          "n2_loop", Kokkos::RangePolicy<HostExec>(0, N),
          KOKKOS_LAMBDA(int t) {
            float fx = 0.f, fy = 0.f, fz = 0.f;
            for (int s = 0; s < N; ++s) {
              if (s == t) continue;
              const float dx = pos(s, 0) - pos(t, 0);
              const float dy = pos(s, 1) - pos(t, 1);
              const float dz = pos(s, 2) - pos(t, 2);
              const float r2 = dx * dx + dy * dy + dz * dz;
              if (r2 < kDetectRadius * kDetectRadius * 4.f) {
                fx += dx;
                fy += dy;
                fz += dz;
              }
            }
            force(t, 0) = fx;
            force(t, 1) = fy;
            force(t, 2) = fz;
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(force(0, 0));
    });
  }

  if (opts.simple) {
    print_simple_section("N2 brute force baseline (N<=1000)", "ns/pair", {
      {"N2 reduce", collect_ns_per_op(b), parse_big_o(b.complexityBigO())},
    });
  } else {
    std::cout << out.str();
    std::cout << "N2 baseline complexity fit:\n  " << b.complexityBigO();
  }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
  BenchOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--simple")          opts.simple         = true;
    if (arg == "--sort-targets")    opts.sort_targets   = true;
    if (arg == "--sort-neighbors")  opts.sort_neighbors = true;
  }

  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    std::cout << "=== Neighbor List Performance Benchmarks ===\n"
              << "  Geometry : unit-density cubic domain, detect_radius=" << kDetectRadius
              << " (" << kTargetNeighbors << " avg neighbors)\n"
              << "  N values :";
    for (int ni = 0; ni < kNumNValues; ++ni) std::cout << " " << kNValues[ni];
    std::cout << "\n";
    if (opts.simple)         std::cout << "  Mode     : --simple (compact summary)\n";
    if (opts.sort_targets)   std::cout << "  Sorting  : targets pre-sorted by Z-Morton via STK sort_entities\n";
    if (opts.sort_neighbors) std::cout << "  Sorting  : neighbor rows sorted by source ordinal after construction\n";
    std::cout << "\n";

    diagnose_target_ordering();

    bench_construction(opts);
    bench_iteration_overhead(opts);
    bench_global_reduce(opts);
    bench_pair_atomic_target(opts);
    bench_bilateral_atomic(opts);
    bench_n2_baseline(opts);
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}

#else  // !HAVE_MUNDYSEARCH_ARBORX

#include <stk_util/parallel/Parallel.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  std::cout << "PerfTestNeighborList: HAVE_MUNDYSEARCH_ARBORX not defined; skipping.\n";
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}

#endif  // HAVE_MUNDYSEARCH_ARBORX
