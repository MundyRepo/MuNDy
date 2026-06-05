# MundySearch {#MundySearch}

See the \ref mundy/search "MundySearch directory reference".

MundySearch is the part of Mundy that builds and iterates neighbor lists over STK mesh entities.

If MundyMesh provides the mesh data model and MundyMath provides small math objects, MundySearch answers the
question "for each target entity, which source entities lie within a detection radius?" in a way that is
independent of the search backend and compatible with Kokkos parallel dispatch.

The package centers on a single abstraction: the **neighbor list**. A neighbor list is the result of a spatial
query. It stores which source entities lie within the detection region of each target entity, and it exposes a
uniform access surface so that kernels can iterate over those pairs without knowing which search backend built
the list.

## Architecture at a glance

| **Layer** | **Main types / functions** | **Use when you need** |
|-----------|-----------------------------|------------------------|
| Search inputs | `impl::ArborXSearchBoxesT`, `impl::PeriodicArborXSearchBoxesT`, `impl::STKSearchBoxesT` | A geometry-indexed pairing of STK entities with search boxes to feed the builder. |
| Builder | `NeighborListBuilder`, `make_neighbor_list_builder<ListType>()` | A fluent, type-safe way to build a concrete neighbor list. |
| Build traits | `NeighborListBuildTraits<ListType>`, `NeighborListInputType` | Coupling a concrete list type to its build logic and type-specific parameters. |
| Excluders | `NoExcluder`, `ExcludeSelfInteraction`, `ExcludeSymmetricDuplicates`, `ExcluderChain` | Filtering candidate pairs at build time before they enter storage. |
| Common access surface | `NeighborListType` concept, `Neighbors<List>`, `NeighborPair<List>` | Querying the stored list in a backend-independent way. |
| Iteration | `for_each_neighbor_pair`, `for_each_target_with_neighbors`, and their `_reduce` variants | Parallel dispatch over stored pairs or targets. |
| Iteration traits | `NeighborListIterationTraits<ListType>` | Specializing the parallel dispatch strategy for a list type. |
| Concrete list types | `ArborX1dNeighborList`, `ArborX2dNeighborList`, `STKSearchNeighborList`, and periodic variants | Backend-specific storage and build logic. |

## The neighbor list model

### Targets and sources

A neighbor list answers one query:

> For each entity in the **target** selector, which entities in the **source** selector fall within the detection
> region?

The two selectors may describe the same set of entities, two completely disjoint sets, or partially overlapping
sets. The list does not care — it stores whatever the search backend reports as candidate pairs that pass the
excluder chain.

The list assigns a **dense target ordinal** to each target entity (0 through `num_targets() - 1`) and a **dense
source ordinal** to each source entity (0 through `num_sources() - 1`). These ordinals are local to the list and
have no relation to STK entity identifiers. They are the index space used by all accessor methods.

### Storage layouts

Two concrete storage layouts are provided. Both satisfy the `NeighborListType` concept and support identical
iteration via `for_each_neighbor_pair` and `for_each_target_with_neighbors`.

| **Layout** | **Concrete type** | **Best for** |
|------------|-------------------|--------------|
| Compressed 1D | `ArborX1dNeighborList<MemSpace>` | Sparse or variable-length neighbor lists; minimum memory overhead. |
| Dense 2D | `ArborX2dNeighborList<MemSpace>` | Uniform-density lists where GPU pair-parallel dispatch is valuable. |

The 1D layout is the better default. The 2D layout trades memory for a constant per-target row width, which can
improve GPU thread utilization when specialized parallel dispatch is used.

### The `NeighborListType` concept

Any type that satisfies `NeighborListType` may be used with `for_each_neighbor_pair` and
`for_each_target_with_neighbors`. The required surface is:

| **Required** | **Meaning** |
|--------------|-------------|
| `size_type` | Index type for target/source ordinals and total pair counts. |
| `source_index_type` | Index type for source ordinals specifically. |
| `execution_space` | Kokkos execution space associated with the stored data. |
| `memory_space` | Kokkos memory space where the list data lives. |
| `num_targets()` | Number of enumerable target entities. |
| `num_sources()` | Number of enumerable source entities. |
| `size()` | Total number of stored neighbor pairs. |
| `target_selector()` / `source_selector()` | Selectors used to build the list. |
| `num_neighbors(target_index)` | Number of neighbors for one target. |
| `get_neighbor(target_index, neighbor_ordinal)` | Source entity at a given target/neighbor position. |
| `target_entity(target_index)` | STK entity for a target ordinal. |
| `source_entity(source_index)` | STK entity for a source ordinal. |
| `source_index(target_index, neighbor_ordinal)` | Dense source ordinal at a given position. |

## Building a neighbor list

### Search inputs

The builder requires geometry for both the target and source populations. For ArborX-backed lists, this geometry
is provided as `impl::ArborXSearchBoxesT<MemSpace>` — a paired collection of `ArborX::Box` objects and
`stk::mesh::Entity` values sharing a single selector.

```cpp
using MemSpace = stk::ngp::MemSpace;
using SearchBoxes = mundy::search::impl::ArborXSearchBoxesT<MemSpace>;

// N entities selected by spheres_selector with positions and radii already computed
size_t n = /* number of selected entities */;
Kokkos::View<ArborX::Box*, MemSpace> boxes("boxes", n);
Kokkos::View<stk::mesh::Entity*, MemSpace> entities("entities", n);

// Populate boxes and entities on the device:
// for each entity at position center and detection radius r:
//   boxes(i) = ArborX::Box{center - r, center + r};
//   entities(i) = entity;

SearchBoxes target_boxes(spheres_selector, boxes, entities);
SearchBoxes source_boxes(spheres_selector, boxes, entities);  // same set → self-interaction search
```

The selector is carried through to the finished list so that its `target_selector()` and `source_selector()`
accessors reflect the selectors used to build it.

### `NeighborListBuilder`

`NeighborListBuilder` is a type-state fluent builder. Each setter returns a new builder type that carries the
supplied field in a template parameter. Calling `build()` on a builder that is missing any required field is a
compile error.

```cpp
using namespace mundy::search;

auto list = make_neighbor_list_builder<ArborX1dNeighborList<>>()
    .exec_space(Kokkos::DefaultExecutionSpace{})
    .target_input(target_boxes)
    .source_input(source_boxes)
    .broad_phase(ExcludeSelfInteraction{})
    .build(bulk_data, {.buffer_size = 16});
```

The builder setters are:

| **Setter** | **Meaning** |
|------------|-------------|
| `.exec_space(exec)` | Kokkos execution space used for the search query. |
| `.target_input(input)` | Search geometry for the target population. |
| `.source_input(input)` | Search geometry for the source population. |
| `.broad_phase(excluder)` / `.narrow_phase(excluder)` | Append an excluder to the build-time filter chain. |
| `.sort_neighbors(bool)` | Whether to sort each target's neighbor row by ascending source ordinal after construction. Default: `false`. |

### Build arguments

Every concrete list type accepts optional backend-specific build parameters. For ArborX-backed types, the only
parameter is `buffer_size`, which hints at how many neighbors per target to pre-allocate. Larger values reduce
reallocation passes at the cost of more memory; `0` (the default) lets ArborX choose.

```cpp
// Pass inline using designated initialization — type is deduced:
auto list = builder.build(bulk_data, {.buffer_size = 16});

// Or name the type explicitly when pre-declaring:
using MyBuilder = NeighborListBuilder<ArborX1dNeighborList<>>;
MyBuilder::build_args_type args{};
args.buffer_size = 16;
auto list = builder.build(bulk_data, args);
```

`STKSearchNeighborList` requires no extra parameters; call `build(bulk_data)` with no second argument.

### Neighbor row sorting

When source-field data (positions, radii, or other per-source properties) will be accessed for multiple targets
that share neighbors, sorting each target's neighbor row by ascending source ordinal improves spatial
locality in the source data accesses:

```cpp
auto list = make_neighbor_list_builder<ArborX1dNeighborList<>>()
    .exec_space(exec)
    .target_input(target_boxes)
    .source_input(source_boxes)
    .sort_neighbors(true)
    .build(bulk_data, {.buffer_size = 16});
```

Sorting is efficient for the short rows typical in neighbor lists.

## Excluders

Excluders are build-time predicates that reject candidate target/source pairs before those pairs enter the stored
list. They are evaluated at build time, not at iteration time.

The `ExcluderType` concept requires:

| **Required** | **Meaning** |
|--------------|-------------|
| `setup(bulk_data, target_selector, source_selector)` | Prepare any mesh-dependent state before the backend callback runs. |
| `operator()(candidate) → bool` | Return `true` to reject the candidate pair (exclude it from storage). |

### Built-in excluders

| **Excluder** | **Behavior** |
|--------------|--------------|
| `NoExcluder` | Rejects nothing. Default starting point in all builders. |
| `ExcludeSelfInteraction` | Rejects self-interactions. For non-periodic candidates, a self-interaction is any pair where target and source are the same entity. For periodic candidates, it additionally requires the relative image shift to be zero — a same-entity pair across a periodic boundary (nonzero shift) is a genuine interaction and is retained. |
| `ExcludeConnectedEntities` | Rejects pairs where the target and source share at least one connected entity at a specified rank. Constructed with a `stk::mesh::EntityRank` (e.g., `NODE_RANK` to exclude element pairs sharing a common node). Requires `setup()` before use. |
| `ExcludeSymmetricDuplicates` | Rejects one orientation of each symmetric pair when targets and sources overlap. Handles identical, disjoint, and partially-overlapping selectors with one type. Requires `setup()` before use; pass through the builder rather than constructing directly. |

### Chaining excluders

`.broad_phase(excluder)` / `.narrow_phase(excluder)` on the builder (or on an existing excluder) returns a new `ExcluderChain` that applies all
previously accumulated excluders in series. Excluders must be lightweight and copyable.

```cpp
auto list = make_neighbor_list_builder<ArborX1dNeighborList<>>()
    .exec_space(exec)
    .target_input(target_boxes)
    .source_input(source_boxes)
    .broad_phase(ExcludeSelfInteraction{})
    .broad_phase(ExcludeSymmetricDuplicates{})  // appended; both will run
    .build(bulk_data, {.buffer_size = 16});
```

### Search candidates

Excluder `operator()` receives a candidate object. Both candidate types expose:

| **Method** | **Meaning** |
|------------|-------------|
| `target_entity()` / `source_entity()` | STK entities for the target and source. |
| `target_index()` / `source_index()` | Dense ordinals for the target and source. |
| `is_degenerate()` | `true` when the candidate is a self-interaction. |
| `operator==` / `operator!=` | Equality by entity (non-periodic) or entity + shift (periodic). |
| `operator<` / `operator>` | Consistent strict ordering — target entity first, then source entity, then shift components — suitable for sorted containers. |

`PeriodicNeighborSearchCandidate` additionally exposes `relative_image_shift()`.

### Custom excluders

Any type that satisfies the `ExcluderType` concept can be used. A minimal custom excluder looks like:

```cpp
struct ExcludeByPartMembership {
  void setup(const stk::mesh::BulkData& bulk, const stk::mesh::Selector&,
             const stk::mesh::Selector&) {
    ngp_mesh_ = stk::mesh::get_updated_ngp_mesh(bulk);
    // cache any other mesh-dependent state needed by operator()
  }

  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION bool operator()(const Candidate& c) const {
    // return true to exclude this pair
    return /* some device-callable predicate */;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
};
```

## Iterating over a neighbor list

Once built, a neighbor list is immutable. All access goes through the `for_each_*` free functions in
`ForEach.hpp`. Two iteration granularities are provided:

| **Function** | **Granularity** | **Functor argument** |
|--------------|-----------------|----------------------|
| `for_each_neighbor_pair(list, functor)` | One invocation per stored pair | `NeighborPair<ListType>` |
| `for_each_target_with_neighbors(list, functor)` | One invocation per target | `Neighbors<ListType>` |
| `for_each_neighbor_pair_reduce(list, functor, reducer)` | Reduction over pairs | `NeighborPair<ListType>` + `value_type&` |
| `for_each_target_with_neighbors_reduce(list, functor, reducer)` | Reduction over targets | `Neighbors<ListType>` + `value_type&` |

All four have an overload that accepts an explicit execution space as the first argument:

```cpp
for_each_neighbor_pair(exec_space, list, functor);
for_each_target_with_neighbors(exec_space, list, functor);
```

When no execution space is provided, `ListType::execution_space{}` is used.

### `NeighborPair<List>`

`NeighborPair<List>` is the payload passed to pair-granular functors. It exposes:

| **Accessor** | **Returns** |
|--------------|-------------|
| `target_index()` | Dense target ordinal. |
| `source_index()` | Dense source ordinal for this pair. |
| `target_entity()` | STK target entity. |
| `source_entity()` | STK source entity. |
| `relative_image_shift()` | Periodic relative image shift (periodic list types only). |

### `Neighbors<List>`

`Neighbors<List>` is the per-target range view passed to target-granular functors. It exposes:

| **Accessor** | **Returns** |
|--------------|-------------|
| `size()` | Number of stored neighbors for this target. |
| `operator[](neighbor_ordinal)` / `operator()(neighbor_ordinal)` | Source entity at the given ordinal. |
| `source_index(neighbor_ordinal)` | Dense source ordinal at the given ordinal. |
| `target_entity()` | STK target entity. |
| `target_index()` | Dense target ordinal. |
| `list()` | Escape hatch: the concrete list being viewed. Use when type-specific behavior is needed (e.g., reading periodic shifts directly). |

### Pair-granular iteration

```cpp
using List = mundy::search::ArborX1dNeighborList<>;

mundy::search::for_each_neighbor_pair(list,
    KOKKOS_LAMBDA(const mundy::search::NeighborPair<List>& pair) {
      stk::mesh::Entity target = pair.target_entity();
      stk::mesh::Entity source = pair.source_entity();
      // compute pairwise interaction between target and source
    });
```

### Target-granular iteration

```cpp
mundy::search::for_each_target_with_neighbors(list,
    KOKKOS_LAMBDA(const mundy::search::Neighbors<List>& nbrs) {
      stk::mesh::Entity target = nbrs.target_entity();
      for (size_t k = 0; k < nbrs.size(); ++k) {
        stk::mesh::Entity source = nbrs[k];
        // accumulate into target-local storage without contention
      }
    });
```

Target-granular iteration is preferred when each invocation accumulates into a single target's data, since it
avoids the inter-thread contention that pair-granular dispatch can cause when multiple threads race to write into
the same target.

### Reductions

Reductions follow the Kokkos reducer pattern used by `stk::mesh::for_each_entity_reduce`. Pass a Kokkos built-in
reducer alongside the functor:

```cpp
size_t total_pairs = 0;
Kokkos::Sum<size_t> sum_reducer(total_pairs);

mundy::search::for_each_neighbor_pair_reduce(list,
    KOKKOS_LAMBDA(const mundy::search::NeighborPair<List>& /*p*/, size_t& count) {
      ++count;
    },
    sum_reducer);
// total_pairs now holds the total number of stored neighbor pairs
```

## Periodic neighbor lists

Periodic boundary conditions require pairing entities with images of other entities that wrap around the domain.
MundySearch supports this through two dedicated list types and a corresponding periodic search input type.

### Periodic concrete types

| **Type** | **Periodic variant** |
|----------|----------------------|
| `ArborX1dNeighborList<MemSpace>` | `PeriodicArborX1dNeighborList<MemSpace, ShiftScalar>` |
| `ArborX2dNeighborList<MemSpace>` | `PeriodicArborX2dNeighborList<MemSpace, ShiftScalar>` |

Both periodic types store a `relative_image_shift` — the source image's translation relative to the target image
— alongside each stored pair. The shift is of type `Vector3<ShiftScalar>` (default `float`).

### Periodic search inputs

Periodic search boxes are `impl::PeriodicArborXSearchBoxesT<MemSpace, ShiftScalar>`. Unlike the non-periodic
variant, this input tracks owner ordinals and per-image shift vectors so that ArborX candidate matches can be
traced back to owner entities with their associated relative shifts.

**Target and source boxes must be built differently** to avoid duplicate list entries.

- **Target boxes** — one image per owner, wrapped into the primary cell. The stored shift is
  `wrapped_position − original_position`.
- **Source boxes** — up to 27 images per owner (one per lattice neighbor at shifts n·L for n ∈ {−1,0,+1}³), each
  stamped at `wrapped_position + n·L`. The stored shift for image n is
  `(wrapped_position + n·L) − original_position`.

The `relative_image_shift` stored in the list for a pair is always
`source_image_shift − target_image_shift`.

```cpp
using PeriodicSearchBoxes = mundy::search::impl::PeriodicArborXSearchBoxesT<MemSpace>;

// Target: one wrapped image per owner.
PeriodicSearchBoxes target_boxes(selector,
    target_image_boxes,    // one AABB per owner, centered at wrap(original)
    owner_entities,        // indexed by dense owner ordinal
    target_owner_indices,  // one entry per owner
    target_image_shifts);  // wrap(original_k) - original_k

// Source: up to 27 images per owner.
PeriodicSearchBoxes source_boxes(selector,
    source_image_boxes,    // up to 27 AABBs per owner
    owner_entities,        // same owner entity list
    source_owner_indices,  // owner ordinal for each image
    source_image_shifts);  // (wrapped + n*L) - original, one per image

auto list = make_neighbor_list_builder<PeriodicArborX1dNeighborList<>>()
    .exec_space(exec)
    .target_input(target_boxes)
    .source_input(source_boxes)
    .broad_phase(ExcludeSelfInteraction{})
    .build(bulk_data, {.buffer_size = 16});
```

Source images can be pruned before building by computing the union AABB of all target boxes and discarding any
source image that doesn't intersect it — such images cannot produce a neighbor pair.

### Using relative image shifts in kernels

Each stored pair carries the shift that maps the source owner's geometry to the image that actually fell within
the detection radius. Kernels reconstruct the image position by adding this shift to the source entity's stored
position:

```cpp
using PeriodicList = mundy::search::PeriodicArborX1dNeighborList<>;

mundy::search::for_each_neighbor_pair(list,
    KOKKOS_LAMBDA(const mundy::search::NeighborPair<PeriodicList>& pair) {
      stk::mesh::Entity target = pair.target_entity();
      stk::mesh::Entity source = pair.source_entity();

      // pair.relative_image_shift() gives source image shift - target image shift
      auto shift = pair.relative_image_shift();
      // reconstructed source position = ngp_position(source) + shift
    });
```

`ExcludeSelfInteraction` handles periodic candidates correctly: a same-entity pair with a nonzero shift is **not**
excluded because it represents a genuine interaction between the entity and its own periodic image across the
boundary.

## Extension points

### Adding a new concrete list type: `NeighborListBuildTraits`

To add a new backend, specialize `NeighborListBuildTraits<ListType>` and provide:

```cpp
template <>
struct mundy::search::NeighborListBuildTraits<MyNeighborList> {
  // Input types expected by the builder
  using target_input_type = impl::ArborXSearchBoxesT<stk::ngp::MemSpace>;
  using source_input_type = impl::ArborXSearchBoxesT<stk::ngp::MemSpace>;

  // Backend-specific build parameters (may be an empty struct)
  struct args_type {
    int my_param = 0;
  };

  // Build function; called by NeighborListBuilder::build()
  template <typename Builder>
  static MyNeighborList build(const Builder& builder,
                              const stk::mesh::BulkData& bulk_data,
                              const args_type& args) {
    // use builder.exec_space(), builder.target_input(), builder.source_input(),
    // builder.setup_excluder(bulk_data), builder.sort_neighbors()
    // to construct and return the list
  }
};
```

The `NeighborListInputType` concept requires only that the input type exposes
`const stk::mesh::Selector& selector() const`.

### Specializing the parallel dispatch strategy: `NeighborListIterationTraits`

The default `NeighborListIterationTraits<ListType>` primary template parallelizes over targets with a
`RangePolicy` and walks each target's neighbor row serially in the inner loop. To override this for a concrete
type, specialize the four dispatch methods:

```cpp
template <>
struct mundy::search::NeighborListIterationTraits<MyNeighborList> {
  using size_type = typename MyNeighborList::size_type;

  template <typename ExecutionSpace, typename Functor>
  static void dispatch_pair(const ExecutionSpace& exec, const MyNeighborList& list, const Functor& f) {
    // e.g., use MDRangePolicy for GPU pair-level parallelism
  }

  template <typename ExecutionSpace, typename Functor, typename ReducerType>
  static void dispatch_pair_reduce(const ExecutionSpace& exec, const MyNeighborList& list,
                                   const Functor& f, ReducerType& r) { /* ... */ }

  template <typename ExecutionSpace, typename Functor>
  static void dispatch_target(const ExecutionSpace& exec, const MyNeighborList& list, const Functor& f) {
    // target dispatch is inherently target-granular; usually no need to override
  }

  template <typename ExecutionSpace, typename Functor, typename ReducerType>
  static void dispatch_target_reduce(const ExecutionSpace& exec, const MyNeighborList& list,
                                     const Functor& f, ReducerType& r) { /* ... */ }
};
```

Specializations live alongside their list-type definition headers. The `dispatch_target` and
`dispatch_target_reduce` methods are not typically specialized because `Neighbors`-callback iteration is
inherently target-granular; overriding them is possible but unusual.

## Choosing a concrete list type

| **Situation** | **Recommendation** |
|---------------|--------------------|
| Default case | `ArborX1dNeighborList<>` — compressed CSR storage, good scalability, no padding overhead. |
| Dense uniform neighborhoods, GPU pair dispatch desired | `ArborX2dNeighborList<>` — constant row width enables `MDRangePolicy` specialization. |
| Already using STK coarse search infrastructure | `STKSearchNeighborList<>` — wraps STK output in the common access surface. |
| Periodic boundary conditions | `PeriodicArborX1dNeighborList<>` or `PeriodicArborX2dNeighborList<>` — stores relative image shifts with each pair. |
| Building a half list (no symmetric duplicates) | Any list type with `.broad_phase(ExcludeSymmetricDuplicates{})`. The half list stores roughly half the pairs, which cuts both memory and kernel cost when both pair orientations would be processed anyway. |
