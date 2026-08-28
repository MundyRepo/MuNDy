# Link restart correctness — API surface design

## Context

`LinkData` keeps a runtime cache field `MUNDY_LINKED_ENTITIES` (arrays of `stk::mesh::Entity`
local offsets) that is derived from two persisted, restart-stable fields,
`MUNDY_LINKED_ENTITY_IDS` and `MUNDY_LINKED_ENTITY_RANKS`. Only the ids/ranks are written to a
restart file (they are tagged `Ioss::Field::TRANSIENT` in the `LinkMetaData` ctor); the entity
handles are runtime caches that must be rebuilt after a read by resolving each `(id, rank)` pair
back to a live `stk::mesh::Entity` via `BulkData::get_entity`.

The rebuild already exists — `LinkCOOData::rebuild_runtime_fields_from_ids_and_ranks()` — and is
reached through `LinkData::refresh_coo_runtime_caches()`. The question this document answers is:
**what public API lets a user run that rebuild at the right moment, regardless of when they declared
their link data, in a way that feels native to the existing Class.hpp / `add_link_restart_fields`
IO idiom?**

### The gap, precisely

- **Declare after the read → already correct.** The `LinkData` ctor ends with
  `refresh_coo_runtime_caches()` (`LinkData.hpp:420-421`). If the link data is declared after the
  mesh is read, the entities exist and the reconcile lands. This is the only ordering currently
  exercised in tests.
- **Declare before the read → broken.** If the `LinkData` object outlives the read (declared, then
  the mesh is read into the same `BulkData`), nothing re-runs the reconcile. `get_linked_entity`
  returns an invalid entity. This ordering is disabled in the tests behind a TODO.

### Why this cannot be fixed inside the observer

`impl::LinkDataObserver` forwards only `notify_crs_may_be_invalid` on mesh-modification signals; it
does not touch the COO. The reconcile must run **after** STK commits both persisted id/rank
transient fields, and STK commits those *after* the mesh-creation modification cycle closes — so it
cannot be driven from a `finished_modification_end_notification` callback (the fields are still
unpopulated, and their storage may be unallocated, at that point → reading it segfaults). An explicit
post-read call is therefore architecturally required. This is not a bug to patch in the observer; it
is the seam where a public API belongs.

### Grounding facts (verified against source)

- `impl::notify_coo_may_be_invalid(const LinkData&)` (`LinkData.hpp:553`) is the public callable
  entry point → `LinkData::notify_coo_may_be_invalid()` → `refresh_coo_runtime_caches()` →
  `LinkCOOData::rebuild_runtime_fields_from_ids_and_ranks()` (`LinkCOOData.hpp:263`).
- The rebuild needs **only** a `BulkData` — no Ioss/region handle. It resolves ids/ranks locally via
  `get_entity`, writes `MUNDY_LINKED_ENTITIES`, resets the CSR cache, and marks `coo_modified_on_host`
  + `crs_structure_dirty` when anything changed. It is **idempotent**: it returns `false` and mutates
  nothing when no value differs, so repeat calls are true no-ops (`LinkCOOData.hpp:308-328`).
- Every `LinkData` on a mesh is enumerable read-only:
  `const LinkDataMap* m = meta.get_attribute<LinkDataMap>();` then iterate `m->contents[rank]`
  (`std::map<std::string, std::unique_ptr<LinkData>>` per rank) for
  `rank ∈ [BEGIN_RANK, NUM_RANKS)` (`LinkData.hpp:559-562`). A null pointer means no links.
  `LinkData` references are address-stable (heap `unique_ptr`, non-movable).
- The write counterpart to mirror: `add_link_restart_fields(StkMeshIoBroker&, size_t output_index,
  LinkMetaData&)` (`LinkMetaData.hpp:455`) — broker-facing, registers the ids/ranks fields via
  `add_class_field`.
- `stk::io` FillMesh surface to mirror (`stk_io/FillMesh.hpp:51-60`) — eight entry points:
  `fill_mesh` (2 overloads), `fill_mesh_with_auto_decomp` (2), `fill_mesh_preexisting`,
  `fill_mesh_save_step_info`, and `fill_mesh_with_fields` (2).
- Include direction constraint: `LinkData.hpp` includes `LinkMetaData.hpp`, not the reverse, and
  `LinkMetaData.hpp` never names `LinkData`/`LinkDataMap`. So the read-side reconcile (which needs
  `LinkDataMap`) **cannot** live beside `add_link_restart_fields` in `LinkMetaData.hpp` without an
  include cycle. `LinkData.hpp` currently pulls in no `stk_io` header.

### The Class.hpp idiom being mirrored

- Public helpers are `inline void` free functions in `mundy::mesh`; plumbing lives in
  `mundy::mesh::impl`; header-only.
- Mundy already mirrors STK names in its own namespace to mean "the Mundy-aware version":
  `mundy::mesh::BulkData`, `mundy::mesh::MetaData`, the Class-aware `mundy::mesh::put_field_on_mesh`.
- Overload precedent: `get_classes(MetaData&)` / `get_classes(BulkData&)` pairs; convenience
  overloads default trailing args (e.g. `db_name = field.name()`).

---

## Design A — mesh-facing reconcile free functions (the primitive)

The smallest honest operation that fixes the gap. Pure mesh-facing, no broker, no new includes.

```cpp
namespace mundy { namespace mesh {

/// \brief Reconcile a single LinkData's COO runtime caches (persisted linked-entity id/rank ->
///        live Entity) with the current mesh. Idempotent.
inline void reconcile_links_after_read(const LinkData& link_data) {
  impl::notify_coo_may_be_invalid(link_data);
}

/// \brief Reconcile every LinkData declared on the mesh. A no-op when the mesh has no links.
inline void reconcile_links_after_read(stk::mesh::BulkData& bulk_data) {
  const stk::mesh::MetaData& meta = bulk_data.mesh_meta_data();
  const LinkDataMap* link_data_map = meta.get_attribute<LinkDataMap>();
  if (link_data_map == nullptr) {
    return;
  }
  for (stk::mesh::EntityRank rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    for (const auto& named_link_data : link_data_map->contents[rank]) {
      impl::notify_coo_may_be_invalid(*named_link_data.second);
    }
  }
}

}}  // namespace mundy::mesh
```

Call site — declare-before-read now works:

```cpp
LinkData& link_data = declare_link_data(bulk, link_meta_data);        // declared BEFORE the read
stk::io::fill_mesh_with_fields(spec, io_broker, bulk, stk::io::READ_RESTART);
reconcile_links_after_read(bulk);                                     // <- the fix
get_updated_ngp_link_data(link_data).update_crs_from_coo();           // NGP as usual
```

**Tradeoffs.** Zero new dependencies (only `LinkDataMap` + `impl::notify_coo_may_be_invalid`, both
already in `LinkData.hpp`); reads as a natural sibling of `declare_link_data`/`get_link_data`; the
overload pair mirrors the existing `get_classes(meta)`/`get_classes(bulk)` precedent. The `const
LinkData&` single-arg form matches the const-through-mutable style of the existing notify functions.
It takes a `stk::mesh::BulkData&` (not the Mundy subtype) so it reconciles links on any bulk. Its one
weakness: it is a step the user must remember, and **forgetting it is a silent wrong answer**
(`get_linked_entity` quietly returns an invalid entity) — which is what Design B removes.

---

## Design B — the Mundy-aware `FillMesh` family (mirror every `stk::io` fill entry point)

Built on Design A. A Mundy mesh has links the way an STK mesh has connectivity, so **every** Mundy
read reconciles link runtime caches by definition. Mundy mirrors the full `stk::io` FillMesh surface
under its own namespace — the same move Mundy already makes for `BulkData`, `MetaData`, and the
Class-aware `put_field_on_mesh`. The uniform rule:

> `mundy::mesh::<fill>(...)` ≡ `stk::io::<fill>(...)`, then `reconcile_links_after_read(bulk)`.

Documented to the same standard as `MeshBuilder` (one-line `\brief`, `\param`, expound only where it
helps). Each definition is a thin inline wrapper — the identically-named `stk::io` call (qualified)
followed by `reconcile_links_after_read(bulk_data)` — so the bodies are omitted here per the uniform
rule above.

```cpp
// mundy_mesh/FillMesh.hpp — mirrors stk_io/FillMesh.hpp under the Mundy namespace.
#include <stk_io/DatabasePurpose.hpp>  // for stk::io::DatabasePurpose
#include <stk_io/FillMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::StkMeshIoBroker
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/LinkData.hpp>     // for mundy::mesh::reconcile_links_after_read

namespace mundy {

namespace mesh {

//! \name Reading a mesh from file
//@{

/// \brief Read a mesh into \p bulk_data and reconcile its links.
/// \param mesh_spec [in] The mesh to read: a file path, or a generated-mesh spec (e.g. "generated:4x4x4").
/// \param bulk_data [in] The mesh to populate.
inline void fill_mesh(const std::string& mesh_spec, mundy::mesh::BulkData& bulk_data);

/// \brief Read a mesh through a caller-provided broker into \p bulk_data and reconcile its links.
///
/// Use this overload to configure broker properties before the read.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
/// \param io_broker [in] The IO broker to read through.
inline void fill_mesh(const std::string& mesh_spec, mundy::mesh::BulkData& bulk_data,
                      stk::io::StkMeshIoBroker& io_broker);

/// \brief Read and automatically decompose a mesh into \p bulk_data and reconcile its links.
///
/// The mesh is distributed across the MPI ranks by recursive coordinate bisection.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
inline void fill_mesh_with_auto_decomp(const std::string& mesh_spec, mundy::mesh::BulkData& bulk_data);

/// \brief Read and automatically decompose a mesh through a caller-provided broker, reconciling its links.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
/// \param io_broker [in] The IO broker to read through.
inline void fill_mesh_with_auto_decomp(const std::string& mesh_spec, mundy::mesh::BulkData& bulk_data,
                                        stk::io::StkMeshIoBroker& io_broker);

/// \brief Read a mesh through a caller-owned broker into \p bulk_data and reconcile its links.
///
/// The broker is used as configured; set its databases, properties, and selectors before calling.
/// \param io_broker [in] The IO broker to read through.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
/// \param purpose [in] Whether the database is read as a mesh or as a restart.
inline void fill_mesh_preexisting(stk::io::StkMeshIoBroker& io_broker, const std::string& mesh_spec,
                                  mundy::mesh::BulkData& bulk_data,
                                  stk::io::DatabasePurpose purpose = stk::io::READ_MESH);

/// \brief Read a mesh with its field data into \p in_bulk, reconcile its links, and report the database time steps.
/// \param in_file [in] The mesh file to read.
/// \param in_bulk [in] The mesh to populate.
/// \param num_steps [out] The number of time steps on the database.
/// \param max_time [out] The largest time value on the database.
inline void fill_mesh_save_step_info(const std::string& in_file, mundy::mesh::BulkData& in_bulk,
                                     int& num_steps, double& max_time);

/// \brief Read a mesh with its field data into \p bulk_data and reconcile its links.
///
/// Field data at the database's final time step is restored in addition to the mesh.
/// \param in_file [in] The mesh file to read.
/// \param bulk_data [in] The mesh to populate.
/// \param purpose [in] Whether the database is read as a mesh or as a restart.
inline void fill_mesh_with_fields(const std::string& in_file, mundy::mesh::BulkData& bulk_data,
                                  stk::io::DatabasePurpose purpose = stk::io::READ_MESH);

/// \brief Read a mesh with its field data through a caller-provided broker into \p bulk_data and reconcile its links.
/// \param in_file [in] The mesh file to read.
/// \param io_broker [in] The IO broker to read through.
/// \param bulk_data [in] The mesh to populate.
/// \param purpose [in] Whether the database is read as a mesh or as a restart.
inline void fill_mesh_with_fields(const std::string& in_file, stk::io::StkMeshIoBroker& io_broker,
                                  mundy::mesh::BulkData& bulk_data,
                                  stk::io::DatabasePurpose purpose = stk::io::READ_MESH);
//@}

}  // namespace mesh

}  // namespace mundy
```

Call site — a user drops the `stk::io::` qualifier (or writes `mundy::mesh::`) and links are restored:

```cpp
mundy::mesh::BulkData& bulk = ...;                              // MeshBuilder produces this
LinkData& link_data = declare_link_data(bulk, link_meta_data); // before OR after: irrelevant
fill_mesh_with_fields(spec, io_broker, bulk, stk::io::READ_RESTART);  // Mundy overload, reconciles
get_updated_ngp_link_data(link_data).update_crs_from_coo();
```

**Links are fundamental mesh constructs.** Like STK connectivity, links are part of the mesh — not an
optional results payload — so reading a mesh must leave the link runtime fields updated. Every Mundy
fill therefore reconciles links after the underlying read; no Mundy read path leaves links stale.
Mirroring the whole family (not just `fill_mesh_with_fields`) means a Mundy user never reaches into
`stk::io` for a read and never has to think about link synchrony — reading a Mundy mesh updates its
links by definition.

**Defaults mirror STK.** `purpose` defaults to `stk::io::READ_MESH` exactly as the STK signatures do;
a restart read passes `stk::io::READ_RESTART` explicitly, identically to `stk::io` usage. Faithful
mirroring keeps the signatures truly drop-in.

**Name & overload resolution.** Reusing STK's fill names is deliberate and matches Mundy's convention
of mirroring STK names in its own namespace. Each overload is distinguished by taking
`mundy::mesh::BulkData&` — precisely how `put_field_on_mesh` is distinguished by taking `Class&`.
Mechanics:
- Argument-dependent lookup already pulls the `stk::io` overload into the candidate set at any call
  (the `stk::io::DatabasePurpose` / `stk::io::StkMeshIoBroker` parameters associate namespace
  `stk::io`).
- For a `mundy::mesh::BulkData` argument the Mundy overload is an **exact** match on the bulk
  parameter while STK's requires a derived→base reference binding (a worse rank). So an unqualified
  `fill_mesh_with_fields(spec, bulk, purpose)` resolves unambiguously to the Mundy overload and
  reconciles — no qualification, no ambiguity. This is the drop-in ergonomics.
- Each body **must** qualify its inner `stk::io::<fill>(...)` call; an unqualified inner call would
  re-select the Mundy overload (exact match) and recurse infinitely.
- One narrow silent case: Mundy links declared on a *plain* `stk::mesh::BulkData` (not
  `mundy::mesh::BulkData`) called unqualified resolve to STK's overload (no reconcile). Mundy's own
  path (`MeshBuilder`) always yields `mundy::mesh::BulkData`, so this does not arise idiomatically;
  such users fall back to the Design A primitive (which takes `stk::mesh::BulkData&`).

  (Sub-decision — the bulk parameter type. `mundy::mesh::BulkData&` gives the drop-in resolution above
  and mirrors `put_field_on_mesh(Class&)`. The alternative, `stk::mesh::BulkData&`, covers links on
  any bulk and has no silent case, but makes an unqualified call an ambiguous compile error so users
  must write `mundy::mesh::fill_...`. Recommended: `mundy::mesh::BulkData&`.)

**Convention note.** Mundy's strict broker-vs-mesh split governs *primitives* (`add_class_field` vs
`declare_class`). These are *orchestrators* — the Mundy analogues of the `stk::io` fill functions,
several of which take `(broker, bulk)` together in their own signatures. They mirror STK signatures
rather than violating the split, and each reconcile substep is cleanly mesh-facing (Design A).

**Tradeoffs.** Best discoverability and the only shape where forgetting is impossible; users already
fluent in `stk::io`'s fill family transfer that knowledge for free. Cost: the new
`mundy_mesh/FillMesh.hpp` takes a `<stk_io/FillMesh.hpp>` dependency (isolated out of core
`LinkData.hpp`).

---

## Design C — a `link_io(bulk)` interface object (evaluated, not recommended)

```cpp
namespace mundy { namespace mesh {
struct BulkDataLinkIoInterface {
  stk::mesh::BulkData& bulk_data;
  void add_restart_fields(stk::io::StkMeshIoBroker& io_broker, size_t output_index);
  void read(const std::string& spec, stk::io::StkMeshIoBroker& io_broker,
            stk::io::DatabasePurpose purpose = stk::io::READ_RESTART);
  void read(const std::string& spec, stk::io::DatabasePurpose purpose = stk::io::READ_RESTART);
  void reconcile_after_read();
};
inline BulkDataLinkIoInterface link_io(stk::mesh::BulkData& bulk_data) { return {bulk_data}; }
}}
```

**Why it is not worth it.** The analogy to `class_interface(bulk)` is superficial.
`BulkDataClassInterface` bundles operations that are *all* bulk-facing, mesh-mutating, and share a
single collaborator (`declare_entity`, `change_entity_classes`); it deliberately contains no IO. A
`link_io` object, to justify itself, would have to bundle `add_restart_fields` (write-side, needs a
broker + output index it cannot hold), `read` (needs a broker), and `reconcile` (bulk-only) — three
different collaborator shapes stapled to a bulk reference. That *breaks* the `class_interface`
parallel rather than honoring it, and re-mixes broker and mesh roles inside one type, which is the
exact thing the strict convention exists to prevent. If a bundling object is ever wanted, the right
axis is a *broker* wrapper, not a bulk wrapper — a larger, separate design not needed here.

---

## UX analysis

| Axis | A: `reconcile_links_after_read` | B: Mundy `fill_*` family | C: `link_io(bulk)` |
|---|---|---|---|
| **Declaration-order independence** | Achieved if the user calls it after the read | Achieved and enforced by construction | Same as B for `read` |
| **Discoverability** | Sits beside `declare_link_data`; pairs with `add_link_restart_fields`. Nothing at the call site forces it | Highest: the whole read surface lives under `mundy::mesh`, so a Mundy user never reaches into `stk::io`; names advertise behavior | Only if the user first finds `link_io` |
| **Failure mode if forgotten** | **Silent wrong answer**: `get_linked_entity` returns invalid; downstream NGP checks *may* throw but are not guaranteed to | Cannot be forgotten by adopters — read and reconcile are one call | Same as B |
| **Symmetry with `add_link_restart_fields`** | Strong: write/broker ↔ read/bulk; verbs differ because the operations genuinely differ (register fields vs rebuild caches) | Complementary: the higher-altitude read side of the round-trip | Bundles both halves; asymmetric collaborators |
| **Cognitive load** | One extra line vs today | Zero — same call shape as `stk::io`, minus the qualifier | Two lines + learning the handle |
| **Broker-vs-mesh convention** | Perfect (pure mesh-facing) | Orchestrators mirroring STK's own signatures; split preserved at the primitive layer | Mixes broker + mesh roles in one type — weakest |

**Net:** A is the correct primitive and the honest mesh-facing operation; B is the ergonomic surface
that closes the silent-failure hole across the whole read API. They are complementary, not competing.

---

## Edge cases / correctness

1. **Multiple LinkData on one mesh.** The `BulkData&` overload reconciles every rank/name in
   `LinkDataMap::contents`; the `const LinkData&` overload targets one. Both correct.
2. **Zero links (null `LinkDataMap`).** The bulk overload early-returns; every Mundy `fill_*`
   degrades to a plain read with no throw. Important so the family can be the default read in generic
   pipelines that may or may not contain links.
3. **Parallel / distributed.** The rebuild resolves each `(id, rank)` locally via `get_entity` and
   **requires** the result be locally valid (throws otherwise). This is safe under the link weak-aura
   invariant: after a *completed* read every locally-owned-or-shared link has at least ghosted access
   to its linked entities. Two consequences: (a) reconcile must run after the read fully finishes,
   which A's documented usage and B's internal sequencing guarantee; (b) reconcile performs **no
   communication** — purely local resolution — so calling it on every rank is collectively safe and
   needs no barrier. Same locality assumption the existing after-read ctor path already relies on.
4. **Reconcile before entities exist.** Called on empty link buckets → no-op. Called mid-partial-state
   → may throw on an unresolvable id/rank. This is the real footgun for A used bare; B eliminates it
   by owning the ordering. A's doc comment states "call after a read."
5. **NGP CRS state — reconcile does not own `update_crs_from_coo`.** A successful rebuild marks
   `coo_modified_on_host` + `crs_structure_dirty` but never rebuilds the CSR or syncs the device. This
   is deliberately consistent with the documented contract that `declare_relation`/`destroy_relation`
   only mark dirty and leave the CSR/device sync to an explicit `update_crs_from_coo`. Reconcile
   behaves exactly like a batch of relation edits, so a subsequent
   `get_updated_ngp_link_data(link_data).update_crs_from_coo()` remaining the user's responsibility is
   the consistent choice. Folding `update_crs_from_coo` into reconcile would force an unwanted device
   sync, cross the host-only layer (it lives on `NgpLinkData`), and break reconcile's allocation-free
   character.
6. **Idempotency / double-call.** The rebuild flips modified flags only when a value actually differs,
   so a second reconcile is a genuine no-op that does not re-dirty state. Robust to every ordering:
   before-read + fill (reconcile once), after-read + fill (ctor reconciled, the fill's reconcile is a
   no-op), and a stray manual reconcile after a fill. Over-calling cannot corrupt state.

---

## Recommendation

Ship **A** (in `LinkData.hpp`) and **B** (the full fill family, in a new `mundy_mesh/FillMesh.hpp`);
skip **C**.

- **A → `LinkData.hpp`.** Needs only `LinkDataMap` + `impl::notify_coo_may_be_invalid`, both already
  present; adds zero new includes to this widely-included core header. Belongs beside
  `declare_link_data`/`get_link_data`. Takes `stk::mesh::BulkData&` so it works for any bulk.
- **B → new `mundy_mesh/FillMesh.hpp`.** Mirrors `stk_io/FillMesh.hpp` name-for-name under
  `mundy::mesh`, isolating the `<stk_io/FillMesh.hpp>` dependency out of core `LinkData.hpp`. Includes
  `<mundy_mesh/LinkData.hpp>`, `<mundy_mesh/BulkData.hpp>`, `<stk_io/FillMesh.hpp>`.

A is the honest, dependency-free operation that fills the seam the observer cannot cover, and it is
what every B function delegates to. B is the surface most users touch — it makes the
silent-wrong-answer failure structurally impossible across the entire read API while faithfully
mirroring `stk::io`'s fill family. The pair honors the broker-vs-mesh convention, completes the
round-trip symmetry with `add_link_restart_fields`, and is provably safe under every declaration
order and repeated call.

### Naming

- **Fill family (B):** `mundy::mesh::fill_mesh`, `fill_mesh_with_auto_decomp`, `fill_mesh_preexisting`,
  `fill_mesh_save_step_info`, `fill_mesh_with_fields` — decided: mirror `stk::io` name-for-name.
  Remaining sub-decision: the bulk parameter type — `mundy::mesh::BulkData&` (recommended; drop-in
  resolution) vs `stk::mesh::BulkData&` (covers any bulk but forces qualified calls). See Design B.
- **Reconcile primitive (A):** `reconcile_links_after_read` (recommended — read-oriented, because it
  applies after *any* Mundy fill, not only restart) · `reconcile_links_after_restart` (pairs literally
  with `add_link_restart_fields`) · `refresh_link_runtime_data` — open.

---

## Test changes (`tests/unit_tests/UnitTestNgpLinkData.cpp`)

- **Re-enable the parameterization.** Restore `LinkDeclOrder::BeforeRestart` to the
  `INSTANTIATE_TEST_SUITE_P` `Values(...)` list so all three round-trips run both orderings.
- **Route the read through the family.** In `read_restart_and_declare`, replace the raw
  `stk::io::fill_mesh_with_fields` with `mundy::mesh::fill_mesh_with_fields(...)`. Because reconcile is
  idempotent, this is correct for both orderings — the after-read branch's ctor reconcile becomes a
  no-op, and the before-read branch is fixed. The existing assertions already prove correctness:
  `expect_link_relation` checks `get_linked_entity` resolves to the live entity, and
  `update_crs_from_coo()` / `check_crs_coo_consistency()` / `is_crs_up_to_date()` must succeed.
- **Add focused tests:** (i) `reconcile_links_after_read(bulk)` does not throw and is a no-op on a
  mesh with zero links (null `LinkDataMap`); (ii) calling it twice is idempotent and leaves NGP
  consistency intact; (iii) `mundy::mesh::fill_mesh_with_fields` on a link-free mesh behaves as a plain
  read; (iv) a test that calls `reconcile_links_after_read(bulk)` explicitly (exercising A in isolation).

## Critical files

- `mundy/mesh/src/mundy_mesh/LinkData.hpp` — add Design A here.
- `mundy/mesh/src/mundy_mesh/FillMesh.hpp` — **new**; the Design B fill family here.
- `mundy/mesh/src/mundy_mesh/LinkMetaData.hpp` — reference for `add_link_restart_fields` (the write
  counterpart); no change required.
- `mundy/mesh/src/mundy_mesh/LinkCOOData.hpp` — reference for the rebuild; no change required.
- `mundy/mesh/tests/unit_tests/UnitTestNgpLinkData.cpp` — test changes above.
- `mundy/mesh/CMakeLists.txt` — register `FillMesh.hpp` among the headers if the module lists headers
  explicitly.
