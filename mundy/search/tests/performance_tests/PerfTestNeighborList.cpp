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
/// \brief Nanobench performance benchmarks for ArborX1dNeighborList, ArborX2dNeighborList,
///        and STKSearchNeighborList.
///
/// Usage: PerfTestNeighborList [--simple] [--search arborx|stk]
///   (default) Full nanobench output with per-variant tables and scaling fits.
///   --simple             Compact summary: one table per benchmark phase.
///   --search arborx|stk  Which backend to benchmark (default: arborx).
///
/// \section Geometry
/// N spheres are placed at Philox-generated positions in a unit-density cubic domain
/// [0, cbrt(N)]³ (L = cbrt(N) keeps number density ρ = 1 particle / unit³ exactly).
/// Each sphere carries an AABB of half-width kDetectRadius so that E[neighbors] =
/// kTargetNeighbors at unit density, independent of N.

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

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/EntitySorterBase.hpp>
#include <stk_mesh/base/Field.hpp>      // for stk::mesh::Field
#include <stk_mesh/base/FieldBase.hpp>  // for stk::mesh::field_data
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>  // for declare_field, put_field_on_mesh
#include <stk_mesh/base/Selector.hpp>
#include <stk_search/BoundingBox.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy math / mesh
#include <mundy_math/Vector3.hpp>         // for mundy::Vector3
#include <mundy_math/zmort.hpp>           // for mundy::zmorton_less
#include <mundy_mesh/FieldComponent.hpp>  // for mundy::mesh::AABBFieldComponent

// Mundy search — always available
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_ARBORX
#include <mundy_search/Excluder.hpp>
#include <mundy_search/ForEach.hpp>
#include <mundy_search/NeighborListBuilder.hpp>
#include <mundy_search/Neighbors.hpp>
#include <mundy_search/STKSearchNeighborList.hpp>
#include <mundy_search/SearchInput.hpp>  // for mundy::search::SearchInput (component input)
#include <mundy_search/impl/STKSearchBoxes.hpp>

// Mundy utils
#include <mundy_utils/rng.hpp>

#ifdef HAVE_MUNDYSEARCH_ARBORX
#include <ArborX.hpp>
#include <mundy_search/ArborX1dNeighborList.hpp>
#include <mundy_search/ArborX2dNeighborList.hpp>
#include <mundy_search/impl/ArborXSearchBoxes.hpp>
#endif

// =============================================================================
// Constants
// =============================================================================

static constexpr int kTargetNeighbors = 14;
static constexpr float kDetectRadius = 0.6165530185826175f;  // cbrt(kTargetNeighbors + 1) / 4
static constexpr int kNValues[] = {1000, 8000, 64000, 256000, 512000};
static constexpr int kNumNValues = sizeof(kNValues) / sizeof(kNValues[0]);

// =============================================================================
// Type aliases
// =============================================================================

using HostSpace = Kokkos::HostSpace;
using HostExec = Kokkos::DefaultHostExecutionSpace;
using PosView = Kokkos::View<float**, HostSpace>;
using ForceView = Kokkos::View<float**, HostSpace>;
using STKList = mundy::search::STKSearchNeighborList<HostSpace>;

#ifdef HAVE_MUNDYSEARCH_ARBORX
using List1d = mundy::search::ArborX1dNeighborList<HostSpace>;
using List2d = mundy::search::ArborX2dNeighborList<HostSpace>;
#endif

// Component-backed search input shared by all backends (STK + ArborX consume the same AABB component).
using PerfComponent = mundy::mesh::AABBFieldComponent<double>;
using PerfInput = mundy::search::SearchInput<PerfComponent>;

// Declare a 6-scalar `aabb` node field (min xyz 0-2, max xyz 3-5). Must be called before commit.
inline stk::mesh::Field<double>& perf_declare_aabb_field(stk::mesh::MetaData& meta) {
  auto& field = meta.declare_field<double>(stk::topology::NODE_RANK, "aabb_perf_field");
  stk::mesh::put_field_on_mesh(field, meta.universal_part(), 6, nullptr);
  return field;
}

// Write one node's AABB (center ± half) into the field.
inline void perf_store_aabb(stk::mesh::Field<double>& field, stk::mesh::Entity node, float cx, float cy, float cz,
                            float h) {
  double* d = stk::mesh::field_data(field, node);
  d[0] = cx - h;
  d[1] = cy - h;
  d[2] = cz - h;
  d[3] = cx + h;
  d[4] = cy + h;
  d[5] = cz + h;
}

// =============================================================================
// Backend tags — the fixture/benchmarks are still parameterized by a backend tag, but all backends now consume the
// same component `SearchInput`, so the tag no longer carries box construction.
// =============================================================================

struct STKBoxTrait {};

#ifdef HAVE_MUNDYSEARCH_ARBORX
struct ArborXBoxTrait {};
#endif

// =============================================================================
// Fixture
// =============================================================================

template <typename Trait>
struct PerfFixtureT {
  int N;
  float L;
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  std::vector<stk::mesh::Entity> nodes;
  PosView pos;
  stk::mesh::Field<double>* aabb_field = nullptr;
  PerfInput boxes;
};

template <typename Trait>
static PerfFixtureT<Trait> build_fixture(int N) {
  PerfFixtureT<Trait> f;
  f.N = N;
  f.L = std::cbrt(static_cast<float>(N));

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  f.meta = builder.create_meta_data();
  f.meta->use_simple_fields();
  f.aabb_field = &perf_declare_aabb_field(*f.meta);  // declared before commit
  f.meta->commit();
  f.bulk = builder.create(f.meta);
  f.bulk->modification_begin();
  for (int id = 1; id <= N; ++id) f.bulk->declare_node(id);
  f.bulk->modification_end();

  f.nodes.resize(N);
  for (int id = 1; id <= N; ++id) f.nodes[id - 1] = f.bulk->get_entity(stk::topology::NODE_RANK, id);

  f.pos = PosView("pos", N, 3);
  for (int i = 0; i < N; ++i) {
    openrand::Philox rng = mundy::make_philox(0xBEEF, static_cast<uint32_t>(i));
    f.pos(i, 0) = rng.uniform<float>(0.f, f.L);
    f.pos(i, 1) = rng.uniform<float>(0.f, f.L);
    f.pos(i, 2) = rng.uniform<float>(0.f, f.L);
    perf_store_aabb(*f.aabb_field, f.nodes[i], f.pos(i, 0), f.pos(i, 1), f.pos(i, 2), kDetectRadius);
  }

  PerfComponent component(*f.aabb_field);
  component.modify_on_host();
  f.boxes = PerfInput{f.meta->universal_part(), component};
  return f;
}

// =============================================================================
// List builders
// =============================================================================

template <typename ListType, typename Trait>
static ListType build_full(const PerfFixtureT<Trait>& f, bool sort_neighbors = false) {
  return mundy::search::make_neighbor_list_builder<ListType>()
      .exec_space(HostExec{})
      .target_input(f.boxes)
      .source_input(f.boxes)
      .broad_phase(mundy::search::ExcludeSelfInteraction{})
      .sort_neighbors(sort_neighbors)
      .build(*f.bulk);
}

template <typename ListType, typename Trait>
static ListType build_half(const PerfFixtureT<Trait>& f, bool sort_neighbors = false) {
  return mundy::search::make_neighbor_list_builder<ListType>()
      .exec_space(HostExec{})
      .target_input(f.boxes)
      .source_input(f.boxes)
      .broad_phase(mundy::search::ExcludeSelfInteraction{})
      .broad_phase(mundy::search::ExcludeSymmetricDuplicates{})
      .sort_neighbors(sort_neighbors)
      .build(*f.bulk);
}

// =============================================================================
// Target-ordering diagnostic (uses STKBoxTrait — backend-independent)
// =============================================================================

// ZMortonSorter uses mundy::zmorton_less (exact floating-point Z-order comparison,
// unit-tested in UnitTestZMorton.cpp) — no quantization, no domain normalization needed.
class ZMortonSorter : public stk::mesh::EntitySorterBase {
 public:
  explicit ZMortonSorter(const PosView& pos) : pos_(pos) {
  }
  void sort(stk::mesh::BulkData& bulk, stk::mesh::EntityVector& ev) const override {
    std::sort(ev.begin(), ev.end(), [this, &bulk](stk::mesh::Entity a, stk::mesh::Entity b) {
      const int ia = static_cast<int>(bulk.identifier(a)) - 1;
      const int ib = static_cast<int>(bulk.identifier(b)) - 1;
      return mundy::zmorton_less(mundy::Vector3<float>{pos_(ia, 0), pos_(ia, 1), pos_(ia, 2)},
                                 mundy::Vector3<float>{pos_(ib, 0), pos_(ib, 1), pos_(ib, 2)});
    });
  }

 private:
  PosView pos_;
};

template <typename Trait>
static void sort_fixture_targets_by_morton(PerfFixtureT<Trait>& f) {
  f.bulk->sort_entities(ZMortonSorter{f.pos});
  {
    int idx = 0;
    for (const auto* b : f.bulk->get_buckets(stk::topology::NODE_RANK, f.meta->universal_part()))
      for (stk::mesh::Entity e : *b) f.nodes[idx++] = e;
  }
  PosView new_pos("pos", f.N, 3);
  for (int i = 0; i < f.N; ++i) {
    const int eid = static_cast<int>(f.bulk->identifier(f.nodes[i])) - 1;
    new_pos(i, 0) = f.pos(eid, 0);
    new_pos(i, 1) = f.pos(eid, 1);
    new_pos(i, 2) = f.pos(eid, 2);
  }
  f.pos = new_pos;

  for (int i = 0; i < f.N; ++i)
    perf_store_aabb(*f.aabb_field, f.nodes[i], f.pos(i, 0), f.pos(i, 1), f.pos(i, 2), kDetectRadius);
  PerfComponent component(*f.aabb_field);
  component.modify_on_host();
  f.boxes = PerfInput{f.meta->universal_part(), component};
}

static void diagnose_target_ordering(bool sort_targets) {
  std::cout << "\n=== Target Ordering Diagnostic ===\n"
            << "  ratio = avg_consecutive / avg_random_pair  (1=random, <<1=spatially coherent)\n\n";
  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture<STKBoxTrait>(N);
    if (sort_targets) sort_fixture_targets_by_morton(f);

    auto avg_consec = [&](const std::vector<int>& order) {
      double sum = 0.0;
      for (int i = 0; i < N - 1; ++i) {
        const int a = order[i], b = order[i + 1];
        const double dx = f.pos(b, 0) - f.pos(a, 0), dy = f.pos(b, 1) - f.pos(a, 1), dz = f.pos(b, 2) - f.pos(a, 2);
        sum += std::sqrt(dx * dx + dy * dy + dz * dz);
      }
      return sum / (N - 1);
    };

    std::vector<int> cur(N);
    std::iota(cur.begin(), cur.end(), 0);
    std::vector<int> xsrt = cur;
    std::sort(xsrt.begin(), xsrt.end(), [&](int a, int b) { return f.pos(a, 0) < f.pos(b, 0); });
    std::vector<int> msrt = cur;
    std::sort(msrt.begin(), msrt.end(), [&](int a, int b) {
      return mundy::zmorton_less(mundy::Vector3<float>{f.pos(a, 0), f.pos(a, 1), f.pos(a, 2)},
                                 mundy::Vector3<float>{f.pos(b, 0), f.pos(b, 1), f.pos(b, 2)});
    });

    const int n_samp = static_cast<int>(std::min(static_cast<long long>(N) * static_cast<long long>(N - 1), 2000LL));
    double sum_rand = 0.0;
    for (int k = 0; k < n_samp; ++k) {
      const int i = k % N, j = (k * 6271 + 1337) % N;
      const double dx = f.pos(j, 0) - f.pos(i, 0), dy = f.pos(j, 1) - f.pos(i, 1), dz = f.pos(j, 2) - f.pos(i, 2);
      sum_rand += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    const double avg_rand = sum_rand / n_samp;
    const auto pct = [&](double d) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%.1f%%", 100.0 * d / avg_rand);
      return std::string(buf);
    };

    const double ac = avg_consec(cur), ax = avg_consec(xsrt), am = avg_consec(msrt);
    std::cout << "  N=" << std::setw(5) << N << "  L=" << std::fixed << std::setprecision(2) << f.L
              << "  r_det=" << kDetectRadius << "\n"
              << "    current order : " << std::setprecision(4) << ac << "  (" << pct(ac) << ")\n"
              << "    x-sorted      : " << ax << "  (" << pct(ax) << ")\n"
              << "    morton-sorted : " << am << "  (" << pct(am) << ")\n"
              << "    random pairs  : " << avg_rand << "\n\n";
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

static std::string parse_big_o(const std::vector<ankerl::nanobench::BigO>& bigos) {
  return bigos.empty() ? "?" : bigos.front().name();
}

static std::array<double, kNumNValues> collect_ns_per_op(const ankerl::nanobench::Bench& b) {
  std::array<double, kNumNValues> ns{};
  ns.fill(kNoData);
  const auto& results = b.results();
  for (int i = 0; i < kNumNValues && i < static_cast<int>(results.size()); ++i)
    ns[i] = results[i].median(ankerl::nanobench::Result::Measure::elapsed) * 1e9;
  return ns;
}

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

struct BenchOptions {
  enum class Backend { ArborX, STK };
  bool simple = false;
  bool sort_targets = false;
  bool sort_neighbors = false;
  bool time_phases = false;  ///< Print per-phase timing for each N (STK only)
  Backend backend = Backend::ArborX;
};

static void print_simple_section(const std::string& title, const std::string& unit,
                                 const std::vector<VariantSummary>& rows) {
  constexpr int kLabelW = 22, kColW = 12;
  std::cout << "\n[" << title << "]  (median " << unit << ")\n"
            << "  " << std::left << std::setw(kLabelW) << "";
  for (int ni = 0; ni < kNumNValues; ++ni)
    std::cout << std::right << std::setw(kColW) << ("N=" + std::to_string(kNValues[ni]));
  std::cout << "  scaling\n"
            << "  " << std::string(kLabelW + kNumNValues * kColW + 12, '-') << "\n";
  for (const auto& v : rows) {
    std::cout << "  " << std::left << std::setw(kLabelW) << (v.label + ":");
    for (int ni = 0; ni < kNumNValues; ++ni) std::cout << std::right << std::setw(kColW) << fmt_time(v.ns_per_op[ni]);
    std::cout << "  " << v.big_o << "\n";
  }
}

static ankerl::nanobench::Bench& configure_bench(ankerl::nanobench::Bench& b, std::ostringstream& os, bool simple,
                                                 const char* title, const char* unit) {
  return b.output(simple ? nullptr : static_cast<std::ostream*>(&os))
      .title(title)
      .unit(unit)
      .relative(true)
      .performanceCounters(true)
      .warmup(2)
      .epochs(10)
      .minEpochIterations(2)
      .minEpochTime(std::chrono::milliseconds(5));
}

// =============================================================================
// Generic benchmark kernels
//
// Each *_append function runs benchmarks for one (ListType, Trait) combination,
// identified by a short prefix string (e.g. "1d", "2d", "stk").
// In non-simple mode it writes nanobench tables to stdout.
// In simple mode it appends VariantSummary entries to the caller's row vectors
// for later printing; bench output is suppressed.
// =============================================================================

// --- 1: Construction (full + half) -------------------------------------------

template <typename ListType, typename Trait>
static void bench_construction_append(const BenchOptions& opts, const std::string& pfx,
                                      std::vector<VariantSummary>& rows) {
  std::ostringstream outf, outh;
  ankerl::nanobench::Bench bf, bh;
  configure_bench(bf, outf, opts.simple, ("Construction: " + pfx + " | Full").c_str(), "build");
  configure_bench(bh, outh, opts.simple, ("Construction: " + pfx + " | Half").c_str(), "build");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture<Trait>(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const std::string tag = " N=" + std::to_string(N);
    bf.complexityN(N).batch(1).run(pfx + " | Full" + tag, [&] {
      ankerl::nanobench::doNotOptimizeAway(build_full<ListType>(f, opts.sort_neighbors));
    });
    bh.complexityN(N).batch(1).run(pfx + " | Half" + tag, [&] {
      ankerl::nanobench::doNotOptimizeAway(build_half<ListType>(f, opts.sort_neighbors));
    });
  }
  if (!opts.simple)
    std::cout << outf.str() << outh.str() << "  " << pfx << " | Full : " << bf.complexityBigO() << "  " << pfx
              << " | Half : " << bh.complexityBigO();
  rows.push_back({pfx + " Full", collect_ns_per_op(bf), parse_big_o(bf.complexityBigO())});
  rows.push_back({pfx + " Half", collect_ns_per_op(bh), parse_big_o(bh.complexityBigO())});
}

// --- 2: Iteration overhead (target loop + pair loop, full list) ---------------

template <typename ListType, typename Trait>
static void bench_iteration_overhead_append(const BenchOptions& opts, const std::string& pfx,
                                            std::vector<VariantSummary>& target_rows,
                                            std::vector<VariantSummary>& pair_rows) {
  std::ostringstream out_t, out_p;
  ankerl::nanobench::Bench b_t, b_p;
  configure_bench(b_t, out_t, opts.simple, ("Iteration overhead: " + pfx + " | Full | Target").c_str(), "item");
  configure_bench(b_p, out_p, opts.simple, ("Iteration overhead: " + pfx + " | Full | Pair").c_str(), "item");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture<Trait>(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto full = build_full<ListType>(f, opts.sort_neighbors);
    const size_t npairs = full.size();
    const std::string tag = " N=" + std::to_string(N);
    Kokkos::View<size_t*, HostSpace> dummy("dummy", N);

    b_t.complexityN(N).batch(static_cast<size_t>(N)).run(pfx + " | Full | Target" + tag, [&] {
      mundy::search::for_each_target_with_neighbors(
          HostExec{}, full, KOKKOS_LAMBDA(const mundy::search::Neighbors<ListType>& nbrs) {
            dummy(nbrs.target_index()) = nbrs.target_index();
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(dummy(0));
    });

    b_p.complexityN(N).batch(npairs).run(pfx + " | Full | Pair" + tag, [&] {
      Kokkos::deep_copy(dummy, size_t(0));
      mundy::search::for_each_neighbor_pair(
          HostExec{}, full,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<ListType>& pair) { dummy(pair.target_index()) += 1; });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(dummy(0));
    });
  }
  if (!opts.simple)
    std::cout << out_t.str() << out_p.str() << "  " << pfx << " | Full | Target : " << b_t.complexityBigO() << "  "
              << pfx << " | Full | Pair   : " << b_p.complexityBigO();
  target_rows.push_back({pfx + " Full target", collect_ns_per_op(b_t), parse_big_o(b_t.complexityBigO())});
  pair_rows.push_back({pfx + " Full pair", collect_ns_per_op(b_p), parse_big_o(b_p.complexityBigO())});
}

// --- 3: Global Kokkos reduction (tw-reduce + pair-reduce, full list) ----------

template <typename ListType, typename Trait>
static void bench_global_reduce_append(const BenchOptions& opts, const std::string& pfx,
                                       std::vector<VariantSummary>& tw_rows, std::vector<VariantSummary>& pr_rows) {
  std::ostringstream out_tw, out_pr;
  ankerl::nanobench::Bench b_tw, b_pr;
  configure_bench(b_tw, out_tw, opts.simple, ("Global reduce (target): " + pfx + " | Full").c_str(), "pair");
  configure_bench(b_pr, out_pr, opts.simple, ("Global reduce (pair): " + pfx + " | Full").c_str(), "pair");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture<Trait>(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto full = build_full<ListType>(f, opts.sort_neighbors);
    const size_t npairs = full.size();
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;

    b_tw.complexityN(N).batch(npairs).run(pfx + " | Full | tw-reduce" + tag, [&] {
      float energy = 0.f;
      Kokkos::Sum<float> red(energy);
      mundy::search::for_each_target_with_neighbors_reduce(
          HostExec{}, full,
          KOKKOS_LAMBDA(const mundy::search::Neighbors<ListType>& nbrs, float& upd) {
            const size_t t = nbrs.target_index();
            for (size_t k = 0; k < nbrs.size(); ++k) {
              const size_t s = nbrs.source_index(k);
              const float dx = pos(s, 0) - pos(t, 0), dy = pos(s, 1) - pos(t, 1), dz = pos(s, 2) - pos(t, 2);
              upd += dx * dx + dy * dy + dz * dz;
            }
          },
          red);
      ankerl::nanobench::doNotOptimizeAway(energy);
    });

    b_pr.complexityN(N).batch(npairs).run(pfx + " | Full | pair-reduce" + tag, [&] {
      float energy = 0.f;
      Kokkos::Sum<float> red(energy);
      mundy::search::for_each_neighbor_pair_reduce(
          HostExec{}, full,
          KOKKOS_LAMBDA(const mundy::search::NeighborPair<ListType>& pair, float& upd) {
            const size_t t = pair.target_index(), s = pair.source_index();
            const float dx = pos(s, 0) - pos(t, 0), dy = pos(s, 1) - pos(t, 1), dz = pos(s, 2) - pos(t, 2);
            upd += dx * dx + dy * dy + dz * dz;
          },
          red);
      ankerl::nanobench::doNotOptimizeAway(energy);
    });
  }
  if (!opts.simple)
    std::cout << out_tw.str() << out_pr.str() << "  " << pfx << " | Full | tw  : " << b_tw.complexityBigO() << "  "
              << pfx << " | Full | pair: " << b_pr.complexityBigO();
  tw_rows.push_back({pfx + " Full", collect_ns_per_op(b_tw), parse_big_o(b_tw.complexityBigO())});
  pr_rows.push_back({pfx + " Full", collect_ns_per_op(b_pr), parse_big_o(b_pr.complexityBigO())});
}

// --- 4: Per-pair atomic into target (full list) -------------------------------

template <typename ListType, typename Trait>
static void bench_pair_atomic_target_append(const BenchOptions& opts, const std::string& pfx,
                                            std::vector<VariantSummary>& rows) {
  std::ostringstream out;
  ankerl::nanobench::Bench b;
  configure_bench(b, out, opts.simple, ("Atomic into target: " + pfx + " | Full").c_str(), "pair");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture<Trait>(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto full = build_full<ListType>(f, opts.sort_neighbors);
    const size_t npairs = full.size();
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;
    ForceView force("force", N, 3);

    b.complexityN(N).batch(npairs).run(pfx + " | Full | Atomic target" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      mundy::search::for_each_neighbor_pair(
          HostExec{}, full, KOKKOS_LAMBDA(const mundy::search::NeighborPair<ListType>& pair) {
            const size_t t = pair.target_index(), s = pair.source_index();
            Kokkos::atomic_add(&force(t, 0), pos(s, 0) - pos(t, 0));
            Kokkos::atomic_add(&force(t, 1), pos(s, 1) - pos(t, 1));
            Kokkos::atomic_add(&force(t, 2), pos(s, 2) - pos(t, 2));
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(force(0, 0));
    });
  }
  if (!opts.simple) std::cout << out.str() << "  " << pfx << " | Full : " << b.complexityBigO();
  rows.push_back({pfx + " Full", collect_ns_per_op(b), parse_big_o(b.complexityBigO())});
}

// --- 5: Bilateral atomic into target AND source (half list) ------------------
//
// Source ordinals are in [0, N) on 1 process for both backends.

template <typename ListType, typename Trait>
static void bench_bilateral_atomic_append(const BenchOptions& opts, const std::string& pfx,
                                          std::vector<VariantSummary>& rows) {
  std::ostringstream out;
  ankerl::nanobench::Bench b;
  configure_bench(b, out, opts.simple, ("Bilateral atomic: " + pfx + " | Half").c_str(), "pair");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture<Trait>(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const auto half = build_half<ListType>(f, opts.sort_neighbors);
    const size_t npairs = half.size();
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;
    ForceView force("force", N, 3);

    b.complexityN(N).batch(npairs).run(pfx + " | Half | Bilateral" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      mundy::search::for_each_neighbor_pair(
          HostExec{}, half, KOKKOS_LAMBDA(const mundy::search::NeighborPair<ListType>& pair) {
            const size_t t = pair.target_index(), s = pair.source_index();
            const float dx = pos(s, 0) - pos(t, 0), dy = pos(s, 1) - pos(t, 1), dz = pos(s, 2) - pos(t, 2);
            Kokkos::atomic_add(&force(t, 0), dx);
            Kokkos::atomic_add(&force(t, 1), dy);
            Kokkos::atomic_add(&force(t, 2), dz);
            Kokkos::atomic_add(&force(s, 0), -dx);
            Kokkos::atomic_add(&force(s, 1), -dy);
            Kokkos::atomic_add(&force(s, 2), -dz);
          });
      Kokkos::fence();
      ankerl::nanobench::doNotOptimizeAway(force(0, 0));
    });
  }
  if (!opts.simple) std::cout << out.str() << "  " << pfx << " | Half : " << b.complexityBigO();
  rows.push_back({pfx + " Half", collect_ns_per_op(b), parse_big_o(b.complexityBigO())});
}

// =============================================================================
// Dispatch functions — select list type(s), call _append, print simple sections
// =============================================================================
// STK build phase timing  (--time-phases, STK backend only)
// =============================================================================
//
// Enables STKSearchNeighborList's built-in phase profiling, runs a single
// full-list build per N, and prints a table of per-phase wall times (ms).
// Helps isolate which phase dominates construction cost.

static void bench_stk_phase_timing(const BenchOptions& opts) {
  using mundy::search::enable_stk_build_profiling;
  using mundy::search::stk_build_last_timings;

  // Column widths
  constexpr int kNW = 8;
  constexpr int kPhW = 8;  // per-phase column
  constexpr int kTotW = 10;

  // Header
  std::cout << "\n=== STK Build Phase Timing (full list, single build per N) ===\n";
  std::cout << std::right << std::setw(kNW) << "N" << std::setw(kPhW) << "A(ms)" << std::setw(kPhW) << "B(ms)"
            << std::setw(kPhW) << "C(ms)" << std::setw(kPhW) << "D(ms)" << std::setw(kPhW) << "E(ms)" << std::setw(kPhW)
            << "F(ms)" << std::setw(kPhW) << "G0(ms)" << std::setw(kPhW) << "G(ms)" << std::setw(kPhW) << "H(ms)"
            << std::setw(kPhW) << "I(ms)" << std::setw(kPhW) << "J(ms)" << std::setw(kPhW) << "K(ms)"
            << std::setw(kTotW) << "Total(ms)"
            << "\n";
  const int ruler_w = kNW + 12 * kPhW + kTotW;
  std::cout << "  " << std::string(ruler_w, '-') << "\n";

  const auto fmt_ms = [](double ms) {
    char buf[16];
    if (ms < 0.001)
      std::snprintf(buf, sizeof(buf), "   <0.001");
    else if (ms < 10.0)
      std::snprintf(buf, sizeof(buf), "%8.3f", ms);
    else
      std::snprintf(buf, sizeof(buf), "%8.2f", ms);
    return std::string(buf);
  };

  enable_stk_build_profiling = true;

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    auto f = build_fixture<STKBoxTrait>(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);

    // Single build — timings written into stk_build_last_timings.
    ankerl::nanobench::doNotOptimizeAway(build_full<STKList>(f, opts.sort_neighbors));

    const auto& t = stk_build_last_timings;
    std::cout << std::right << std::setw(kNW) << N << fmt_ms(t.phase_a_ms) << fmt_ms(t.phase_b_ms)
              << fmt_ms(t.phase_c_ms) << fmt_ms(t.phase_d_ms) << fmt_ms(t.phase_e_ms) << fmt_ms(t.phase_f_ms)
              << fmt_ms(t.phase_g0_ms) << fmt_ms(t.phase_g_ms) << fmt_ms(t.phase_h_ms) << fmt_ms(t.phase_i_ms)
              << fmt_ms(t.phase_j_ms) << fmt_ms(t.phase_k_ms) << std::setw(kTotW) << fmt_ms(t.total_ms()) << "\n";
  }

  enable_stk_build_profiling = false;

  std::cout << "\n  Phase key:\n"
            << "    A = build BoxIdentProc search views\n"
            << "    B = build target EntityKey->ordinal map\n"
            << "    C = stk::search::coarse_search (MORTON_LBVH)\n"
            << "    D = mirror results to host + ghosting (multi-rank only)\n"
            << "    E = build extended source entity view\n"
            << "    F = refresh NgpMesh + build source EntityKey->ordinal map\n"
            << "    G0 = precompute valid target/source ordinal pairs\n"
            << "    G = count pass (atomic per-target increments)\n"
            << "    H = prefix scan + write-position init\n"
            << "    I = fill pass (atomic slot allocation)\n"
            << "    J = per-row insertion sort (disabled here)\n"
            << "    K = construct list object\n\n";
}

// =============================================================================

// Helper: run _append for all active list type(s).
// ArborX variants are wrapped in #ifdef so callers don't need to repeat it.
#ifdef HAVE_MUNDYSEARCH_ARBORX
#define FOR_EACH_ARBORX_VARIANT(func, opts, ...)           \
  func<List1d, ArborXBoxTrait>((opts), "1d", __VA_ARGS__); \
  func<List2d, ArborXBoxTrait>((opts), "2d", __VA_ARGS__)
#else
#define FOR_EACH_ARBORX_VARIANT(func, opts, ...) ((void)0) /* ArborX not built */
#endif

static void bench_construction(const BenchOptions& opts) {
  std::vector<VariantSummary> rows;
  if (opts.backend == BenchOptions::Backend::STK) {
    bench_construction_append<STKList, STKBoxTrait>(opts, "stk", rows);
  } else {
    FOR_EACH_ARBORX_VARIANT(bench_construction_append, opts, rows);
  }
  if (opts.simple && !rows.empty()) print_simple_section("Construction", "ns/build", rows);
}

static void bench_iteration_overhead(const BenchOptions& opts) {
  std::vector<VariantSummary> target_rows, pair_rows;
  if (opts.backend == BenchOptions::Backend::STK) {
    bench_iteration_overhead_append<STKList, STKBoxTrait>(opts, "stk", target_rows, pair_rows);
  } else {
    FOR_EACH_ARBORX_VARIANT(bench_iteration_overhead_append, opts, target_rows, pair_rows);
  }
  if (opts.simple) {
    if (!target_rows.empty()) print_simple_section("Iteration overhead (target)", "ns/target", target_rows);
    if (!pair_rows.empty()) print_simple_section("Iteration overhead (pair)", "ns/pair", pair_rows);
  }
}

static void bench_global_reduce(const BenchOptions& opts) {
  std::vector<VariantSummary> tw_rows, pr_rows;
  if (opts.backend == BenchOptions::Backend::STK) {
    bench_global_reduce_append<STKList, STKBoxTrait>(opts, "stk", tw_rows, pr_rows);
  } else {
    FOR_EACH_ARBORX_VARIANT(bench_global_reduce_append, opts, tw_rows, pr_rows);
  }
  if (opts.simple) {
    if (!tw_rows.empty()) print_simple_section("Global reduce (target)", "ns/pair", tw_rows);
    if (!pr_rows.empty()) print_simple_section("Global reduce (pair)", "ns/pair", pr_rows);
  }
}

static void bench_pair_atomic_target(const BenchOptions& opts) {
  std::vector<VariantSummary> rows;
  if (opts.backend == BenchOptions::Backend::STK) {
    bench_pair_atomic_target_append<STKList, STKBoxTrait>(opts, "stk", rows);
  } else {
    FOR_EACH_ARBORX_VARIANT(bench_pair_atomic_target_append, opts, rows);
  }
  if (opts.simple && !rows.empty()) print_simple_section("Atomic into target (full list)", "ns/pair", rows);
}

static void bench_bilateral_atomic(const BenchOptions& opts) {
  std::vector<VariantSummary> rows;
  if (opts.backend == BenchOptions::Backend::STK) {
    bench_bilateral_atomic_append<STKList, STKBoxTrait>(opts, "stk", rows);
  } else {
    FOR_EACH_ARBORX_VARIANT(bench_bilateral_atomic_append, opts, rows);
  }
  if (opts.simple && !rows.empty()) print_simple_section("Bilateral atomic (half list)", "ns/pair", rows);
}

#undef FOR_EACH_ARBORX_VARIANT

// =============================================================================
// Benchmark 6: N² brute force (baseline, no neighbor list)
// =============================================================================

static void bench_n2_baseline(const BenchOptions& opts) {
  std::ostringstream out;
  ankerl::nanobench::Bench b;
  configure_bench(b, out, opts.simple, "N² brute force baseline (N <= 1000)", "pair");

  for (int ni = 0; ni < kNumNValues; ++ni) {
    const int N = kNValues[ni];
    if (N > 1000) break;
    auto f = build_fixture<STKBoxTrait>(N);
    if (opts.sort_targets) sort_fixture_targets_by_morton(f);
    const size_t npairs = static_cast<size_t>(N) * static_cast<size_t>(N - 1);
    const std::string tag = " N=" + std::to_string(N);
    const PosView pos = f.pos;
    ForceView force("force", N, 3);

    b.complexityN(N).batch(npairs).run("N2 reduction" + tag, [&] {
      Kokkos::deep_copy(force, 0.f);
      Kokkos::parallel_for(
          "n2_loop", Kokkos::RangePolicy<HostExec>(0, N), KOKKOS_LAMBDA(int t) {
            float fx = 0.f, fy = 0.f, fz = 0.f;
            for (int s = 0; s < N; ++s) {
              if (s == t) continue;
              const float dx = pos(s, 0) - pos(t, 0), dy = pos(s, 1) - pos(t, 1), dz = pos(s, 2) - pos(t, 2);
              if (dx * dx + dy * dy + dz * dz < kDetectRadius * kDetectRadius * 4.f) {
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
    print_simple_section("N2 brute force baseline (N<=1000)", "ns/pair",
                         {
                             {"N2 reduce", collect_ns_per_op(b), parse_big_o(b.complexityBigO())},
                         });
  } else {
    std::cout << out.str() << "N2 baseline complexity fit:\n  " << b.complexityBigO();
  }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
  BenchOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--simple") {
      opts.simple = true;
      continue;
    }
    if (arg == "--sort-targets") {
      opts.sort_targets = true;
      continue;
    }
    if (arg == "--sort-neighbors") {
      opts.sort_neighbors = true;
      continue;
    }
    if (arg == "--time-phases") {
      opts.time_phases = true;
      continue;
    }
    if (arg == "--search" && i + 1 < argc) {
      const std::string backend(argv[++i]);
      if (backend == "stk")
        opts.backend = BenchOptions::Backend::STK;
      else if (backend == "arborx")
        opts.backend = BenchOptions::Backend::ArborX;
      else {
        std::cerr << "Unknown backend: " << backend << "\n";
        return 1;
      }
      continue;
    }
  }
  if (opts.backend == BenchOptions::Backend::ArborX) {
#ifndef HAVE_MUNDYSEARCH_ARBORX
    std::cerr << "ERROR: --search arborx requires HAVE_MUNDYSEARCH_ARBORX.\n"
              << "  Rebuild with ArborX support or use --search stk.\n";
    return 1;
#endif
  }

  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    const std::string bname = (opts.backend == BenchOptions::Backend::STK) ? "stk" : "arborx";
    std::cout << "=== Neighbor List Performance Benchmarks ===\n"
              << "  Backend  : " << bname << "\n"
              << "  Geometry : unit-density cubic domain, detect_radius=" << kDetectRadius << " (" << kTargetNeighbors
              << " avg neighbors)\n"
              << "  N values :";
    for (int ni = 0; ni < kNumNValues; ++ni) std::cout << " " << kNValues[ni];
    std::cout << "\n";
    if (opts.simple) std::cout << "  Mode     : --simple\n";
    if (opts.sort_targets) std::cout << "  Sorting  : targets pre-sorted by Z-Morton\n";
    if (opts.sort_neighbors) std::cout << "  Sorting  : neighbor rows sorted by source ordinal\n";
    if (opts.time_phases) std::cout << "  Mode     : --time-phases (STK build phase breakdown)\n";
    std::cout << "\n";

    diagnose_target_ordering(opts.sort_targets);

    if (opts.time_phases && opts.backend == BenchOptions::Backend::STK) {
      bench_stk_phase_timing(opts);
    } else {
      bench_construction(opts);
      bench_iteration_overhead(opts);
      bench_global_reduce(opts);
      bench_pair_atomic_target(opts);
      bench_bilateral_atomic(opts);
      bench_n2_baseline(opts);
    }
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
