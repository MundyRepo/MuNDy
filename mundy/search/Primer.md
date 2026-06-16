# MundySearch {#MundySearch}

\brief Neighbor-list construction and iteration.

See the \ref mundy/search "MundySearch directory reference".

MundySearch is the part of Mundy that builds and iterates neighbor lists over STK mesh entities.

## The neighbor list model

A neighbor list answers one query: for each entity in a **target** selector, which entities in a **source** selector lie within range? The two selectors may be identical (a population searched against itself), disjoint, or overlapping. The built list assigns every target and every source a dense, list-local **ordinal** — `0 … num − 1`, unrelated to STK entity IDs — and its accessors are indexed by these ordinals.

Search proceeds in two phases. The **broad phase** tests cheap bounding volumes (axis-aligned boxes) and deliberately over-approximates: it proposes *candidate* pairs that might be neighbors. The optional **narrow phase** applies more expensive geometric tests — for example, oriented-box intersection — to each candidate and discards those that only overlapped at the bounding-volume level. **Excluders** are the predicates that reject candidates; each attaches to one phase or the other, so cheap rejections (self-pairs, symmetric duplicates) run in the broad phase while expensive ones run in the narrow phase.

Geometry is never handed in as a raw array. Each population is described by a **component** — a field-backed geometry accessor — paired with its selector, together a `SearchInput`. Reading geometry from a field rather than a fixed table is what keeps a build correct across MPI ranks: when the search ghosts a source entity onto a remote rank, its geometry field is communicated and re-read. The single-rank ArborX backends and the multi-rank `STKSearchNeighborList` differ precisely in this machinery; see [Concrete list types](#concrete-list-types).

## NeighborListBuilder

`make_neighbor_list_builder<ListType>()` is the entry point for all neighbor-list construction. It returns a
*type-state* builder: its type changes with each setter, so an incomplete configuration fails to compile rather
than at runtime. The `<ListType>` argument selects the storage backend and search algorithm
([Concrete list types](#concrete-list-types)).

The example below sets up rod geometry, builds a rod–rod neighbor list with a broad-phase self-pair filter and a
narrow-phase oriented-box test, then visits the stored pairs:

```cpp
using namespace mundy::search;
using List = ArborX1dNeighborList<>;

// Geometry lives in STK fields; components read it. modify_on_host() marks the host-side
// writes so the build syncs them to device before searching.
mundy::mesh::AABBFieldComponent<double> aabb_comp(aabb_field);
mundy::mesh::OBBFieldComponent<double>  obb_comp(obb_field);
aabb_comp.modify_on_host();
obb_comp.modify_on_host();

// A SearchInput pairs a selector with a geometry component. Using one input for both the
// target and source searches the rod population against itself.
SearchInput rods{rod_selector, aabb_comp};

auto list = make_neighbor_list_builder<List>()
    .exec_space(Kokkos::DefaultExecutionSpace{})
    .target_input(rods)
    .source_input(rods)
    .broad_phase(ExcludeSelfInteraction{})                // drop self-pairs cheaply
    .narrow_phase(ExcludeNonIntersectingOBBs{obb_comp})   // exact OBB test on the survivors
    .build(bulk_data, {.buffer_size = 16});

// Visit every stored (target, source) pair.
for_each_neighbor_pair(list,
    KOKKOS_LAMBDA(const NeighborPair<List>& pair) {
      // compute the interaction between pair.target_entity() and pair.source_entity()
    });
```

The builder setters are:

| **Setter** | **Meaning** |
|------------|-------------|
| `.exec_space(exec)` | Kokkos execution space used for the search query. |
| `.target_input(input)` | Search geometry for the target population. |
| `.source_input(input)` | Search geometry for the source population. |
| `.broad_phase(excluder)` / `.narrow_phase(excluder)` | Append an excluder to the broad or narrow phase. |
| `.sort_neighbors(bool)` | Sort each target's neighbor row by ascending source ordinal after construction. Default: `false`. |
| `.manage(rebuilder)` | Wrap the builder in a `ManagedNeighborList` driven by a rebuilder policy. |

### Search inputs

`.target_input()` and `.source_input()` each take a `SearchInput<Component>` — a `stk::mesh::Selector` paired with
a geometry-yielding component (e.g. `AABBFieldComponent<double>`, `OBBFieldComponent<double>`). The selector
fixes which entities are searched and their dense ordering; the component supplies each entity's geometry. Call
`component.modify_on_host()` after writing geometry into the backing field on the host, and the build syncs it to
device before the search. The selectors are carried through to the finished list, reachable via its
`target_selector()` and `source_selector()` accessors.

### Build arguments

ArborX-backed list types accept a `buffer_size` hint — how many neighbors per target to pre-allocate. Larger
values trade memory for fewer reallocation passes; `0` (the default) lets ArborX choose.

```cpp
auto list = builder.build(bulk_data, {.buffer_size = 16});
```

`STKSearchNeighborList` takes no build arguments; call `build(bulk_data)` with no second argument.

### Excluders

Excluders are the build-time predicates introduced [above](#the-neighbor-list-model): each rejects candidate
pairs in one phase. Broad-phase excluders run during the bounding-volume query; narrow-phase excluders see only
the candidates that survived it and may apply more expensive geometry tests. Repeated `.broad_phase()` or
`.narrow_phase()` calls chain excluders in series. Excluders must be lightweight and copyable.

| **Excluder** | **Behavior** |
|--------------|--------------|
| `NoExcluder` | Rejects nothing. Default starting point. |
| `ExcludeSelfInteraction` | Rejects pairs where target and source are the same entity. For periodic candidates, a same-entity pair with nonzero image shift is a genuine interaction and is retained. |
| `ExcludeConnectedEntities` | Rejects pairs where target and source share a connected entity at a specified rank. Constructed with a `stk::mesh::EntityRank` (e.g. `NODE_RANK`). |
| `ExcludeSymmetricDuplicates` | Rejects one orientation of each symmetric pair. Handles identical, disjoint, and partially-overlapping selectors. Pass through the builder rather than constructing directly. |
| `ExcludeNonIntersectingOBBs<Scalar, MemSpace>` | Narrow-phase OBB–OBB separating-axis test. Rejects pairs whose oriented boxes do not intersect. Constructed from one `OBBFieldComponent<Scalar>` (symmetric) or two (asymmetric target/source). |

Any type satisfying the `ExcluderType` concept works. A custom excluder caches mesh-dependent state in `setup()`
and returns `true` from `operator()` to reject a candidate:

```cpp
struct ExcludeByPartMembership {
  void setup(const stk::mesh::BulkData& bulk, const stk::mesh::Selector&,
             const stk::mesh::Selector&) {
    ngp_mesh_ = stk::mesh::get_updated_ngp_mesh(bulk);
    // cache any other mesh-dependent state needed by operator()
  }

  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION bool operator()(const Candidate& c) const {
    return /* some device-callable predicate */;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
};
```

The candidate passed to `operator()` exposes `target_entity()`, `source_entity()`, `target_index()`,
`source_index()`, and `is_degenerate()`. Periodic candidates additionally expose `target_image_shift()` and
`source_image_shift()`.

### Managed lists and rebuilders

In a time-stepping simulation the geometry moves a little each step but the neighbor list rarely needs to be
rebuilt every step. `.manage(rebuilder)` wraps the builder in a `ManagedNeighborList` that caches the last-built
list and consults a **rebuilder** to decide when the cache is stale. The builder holds the fixed configuration
(execution space, excluders, rebuilder); the per-step inputs — which may change as entities move — are passed to
`update()`, which returns the cached list or rebuilds it:

```cpp
auto nl_manager = make_neighbor_list_builder<List>()
    .exec_space(Kokkos::DefaultExecutionSpace{})
    .broad_phase(ExcludeSelfInteraction{})
    .narrow_phase(ExcludeNonIntersectingOBBs{obb_comp})
    .manage(RebuildOnOBBDisplacement<double>{obb_comp, skin_distance});

for (int step = 0; step < num_steps; ++step) {
  // ... advance positions, refresh aabb_field / obb_field on host ...
  aabb_comp.modify_on_host();
  obb_comp.modify_on_host();

  const auto& nl = nl_manager.update(bulk_data, rods, rods);  // rebuilds only when the rebuilder fires

  for_each_neighbor_pair(nl,
      KOKKOS_LAMBDA(const NeighborPair<List>& pair) {
        // compute the interaction between pair.target_entity() and pair.source_entity()
      });
}
```

`.manage()` may appear at any point in the chain, not only last. `ManagedNeighborList` exposes:

| **Method** | **Meaning** |
|------------|-------------|
| `update(bulk, targets, sources)` | Consult the rebuilder; rebuild if needed; return `UpdateResult{list, rebuilt}`, which converts implicitly to `const list_type&`. |
| `has_valid_list()` | `true` after the first `update()` or any build. |
| `current()` | Returns the cached list; throws if `has_valid_list()` is `false`. |
| `invalidate()` | Clears the cache, forcing a rebuild on the next `update()`. |

The built-in rebuilders are:

| **Rebuilder** | **Rebuild condition** |
|---------------|-----------------------|
| `AlwaysRebuild` | Rebuilds every step. |
| `NeverRebuild` | Never rebuilds after the first build. |
| `RebuildOnEntityChange` | Target or source entity count changes. |
| `RebuildOnAABBDisplacement` | Any AABB corner moves farther than a threshold since the last snapshot. Reads boxes from the search input. |
| `RebuildOnOBBDisplacement` | Any OBB escapes its inflated snapshot region. Reads OBBs from an `OBBFieldComponent` supplied at construction. |

Rebuilders combine with `operator|` into a short-circuit OR — the list is rebuilt if any constituent fires, and
every rebuilder re-snapshots after a build so all stay current:

```cpp
.manage(RebuildOnEntityChange{} | RebuildOnAABBDisplacement<double>{skin_distance})
```

## Iterating over a neighbor list

Once built, a neighbor list is immutable. All access goes through the `for_each_*` free functions, at one of two
granularities:

| **Function** | **Granularity** | **Functor argument** |
|--------------|-----------------|----------------------|
| `for_each_neighbor_pair(list, functor)` | One invocation per stored pair | `NeighborPair<ListType>` |
| `for_each_target_with_neighbors(list, functor)` | One invocation per target | `Neighbors<ListType>` |
| `for_each_neighbor_pair_reduce(list, functor, reducer)` | Reduction over pairs | `NeighborPair<ListType>` + `value_type&` |
| `for_each_target_with_neighbors_reduce(list, functor, reducer)` | Reduction over targets | `Neighbors<ListType>` + `value_type&` |

Each accepts an optional execution space as its first argument; when omitted, `ListType::execution_space{}` is used.

### Pair granularity: `NeighborPair<List>`

The payload passed to pair-granular functors exposes `target_index()` and `source_index()` (dense ordinals) and
`target_entity()` and `source_entity()` (STK entities):

```cpp
for_each_neighbor_pair(list,
    KOKKOS_LAMBDA(const NeighborPair<List>& pair) {
      stk::mesh::Entity target = pair.target_entity();
      stk::mesh::Entity source = pair.source_entity();
      // compute pairwise interaction
    });
```

### Target granularity: `Neighbors<List>`

The per-target range view exposes `size()`, `operator[](k)` / `operator()(k)` (source entity at neighbor ordinal
`k`), `source_index(k)`, `target_entity()`, `target_index()`, and `list()` — an escape hatch to the concrete list
for type-specific data such as periodic image shifts:

```cpp
for_each_target_with_neighbors(list,
    KOKKOS_LAMBDA(const Neighbors<List>& nbrs) {
      stk::mesh::Entity target = nbrs.target_entity();
      for (size_t k = 0; k < nbrs.size(); ++k) {
        stk::mesh::Entity source = nbrs[k];
        // accumulate into target-local storage without contention
      }
    });
```

Target granularity is preferred when each invocation accumulates into a single target's data, since it avoids the
write contention that pair-granular dispatch can cause when multiple threads update the same target.

### Reductions

The `_reduce` variants follow the Kokkos reducer pattern — pass a Kokkos built-in reducer alongside the functor:

```cpp
size_t total_pairs = 0;
Kokkos::Sum<size_t> sum_reducer(total_pairs);

for_each_neighbor_pair_reduce(list,
    KOKKOS_LAMBDA(const NeighborPair<List>& /*pair*/, size_t& count) {
      ++count;
    },
    sum_reducer);
```

## Periodic neighbor lists

Periodic boundary conditions pair entities with images of other entities that wrap around the domain. The ArborX
list types have periodic variants, fed by a `PeriodicSearchInput`:

| **Aperiodic type** | **Periodic variant** |
|---------------------|----------------------|
| `ArborX1dNeighborList<MemSpace>` | `PeriodicArborX1dNeighborList<MemSpace, ShiftScalar>` |
| `ArborX2dNeighborList<MemSpace>` | `PeriodicArborX2dNeighborList<MemSpace, ShiftScalar>` |

A `PeriodicSearchInput<Component, Metric>` adds a periodicity metric to the usual selector and component; the build
generates the periodic images on device, so the caller never constructs image boxes by hand:

```cpp
using Metric = mundy::OrthorhombicMetric<mundy::AXIS_XYZ, float>;

mundy::mesh::AABBFieldComponent<double> component(aabb_field);
component.modify_on_host();
PeriodicSearchInput input{rod_selector, component, Metric{mundy::Vector3<float>{Lx, Ly, Lz}}};

auto list = make_neighbor_list_builder<PeriodicArborX1dNeighborList<>>()
    .exec_space(exec)
    .target_input(input)
    .source_input(input)
    .broad_phase(ExcludeSelfInteraction{})
    .build(bulk_data, {.buffer_size = 16});
```

Each stored pair carries the image shift that brought the source into range. These shifts are periodic-specific,
so they live on the concrete periodic list rather than on the generic pair payload; reach them through the
`Neighbors::list()` escape hatch. A kernel reconstructs an image position by adding the relative shift —
`source_image_shift − target_image_shift` — to the source's stored position:

```cpp
using PeriodicList = PeriodicArborX1dNeighborList<>;

for_each_target_with_neighbors(list,
    KOKKOS_LAMBDA(const Neighbors<PeriodicList>& neighbors) {
      const PeriodicList& periodic = neighbors.list();
      const auto target_index = neighbors.target_index();
      const auto target_shift = periodic.target_image_shift(target_index);

      for (std::size_t k = 0; k < neighbors.size(); ++k) {
        auto shift = periodic.source_image_shift(target_index, k) - target_shift;
        // reconstructed source position = ngp_position(neighbors[k]) + shift
      }
    });
```

`ExcludeSelfInteraction` is periodic-aware: a same-entity pair with a nonzero shift is **not** excluded, because
it is a genuine interaction between an entity and its own image across the boundary.

## Concrete list types

MundySearch provides three concrete list types, differing in search backend, storage layout, and rank model.

**`ArborX1dNeighborList`** — ArborX BVH; compressed CSR storage.
- *Benefits:* memory tracks the actual neighbor count with no padding; a thin wrapper over an ArborX BVH, so
  build overhead is low. Absorbs sparse and highly variable neighbor counts gracefully.
- *Costs:* builds a bounding-volume hierarchy on each rebuild; the default iteration walks each target's row
  serially rather than parallelizing over pairs.
- *Limitations:* single rank. It sees only the entities local to its rank, so in a distributed run any pair that
  straddles a rank boundary is missed. **STKSearchNeighborList** corrects this at the cost of more complex machinery, which can exceed 3× the build time even on a single rank.

**`ArborX2dNeighborList`** — ArborX BVH; dense 2D per-target storage.
- *Benefits:* the dense `num_targets × max_row_width` layout gives every target a constant-width row, which lets
  the iteration dispatch use an `MDRangePolicy` to parallelize over pairs directly — higher GPU occupancy than a
  serial per-row walk when neighborhoods are large and uniform.
- *Costs:* the row width is sized to the largest neighborhood, so every shorter row is padded; for sparse or
  highly non-uniform neighborhoods that padding dominates memory.
- *Limitations:* single rank, exactly as the 1d list. For many applications the 1d list's lower build time and memory usage outweigh the 2d list's faster iteration.

**`STKSearchNeighborList`** — STK MORTON_LBVH coarse search; MPI-distributed.
- *Benefits:* automatically ghosts source entities onto the ranks that need them and communicates the geometry
  fields so narrow-phase excluders have valid data for the ghosted set. The only backend that produces a correct
  list when targets and sources are distributed across ranks.
- *Costs:* the multi-rank machinery — ghosting groups, field communication, synchronization accounting — is
  substantial, and is borne even on a single rank.
- *Limitations:* uses STK's coarse-search backend rather than ArborX.

Periodic boundary conditions have dedicated variants (`PeriodicArborX1dNeighborList`,
`PeriodicArborX2dNeighborList`) that store a per-object image shift alongside each pair; see
[Periodic neighbor lists](#periodic-neighbor-lists).

To halve the stored pair count when iterating a symmetric interaction — same target and source populations, both
pair orientations processed identically — add `.broad_phase(ExcludeSymmetricDuplicates{})` to any list type.
