# MundyMesh {#MundyMesh}

See the \ref mundy/mesh "MundyMesh directory reference".

MundyMesh is the part of Mundy that manages mesh data, mesh-side access patterns, and mesh-side algorithms.

If MundyMath provides small math objects and MundyGeom provides small geometry objects, MundyMesh provides the mesh
storage and workflows that let those objects live on entities and be used in host/device algorithms.

Under the hood, MundyMesh extends Trilinos/STK. It keeps STK's core model—`MetaData`, `BulkData`, parts, fields,
selectors, and NGP access—but adds convenience layers and new capabilities needed by Mundy's multibody workflows.

The package is organized in layers:
1. Build or extend an STK mesh with Mundy `MetaData`, `BulkData`, and `MeshBuilder`.
2. Use helpers to declare parts, fields, and semantic classes with less boilerplate.
3. Use components and aggregates to decouple algorithms from storage details.
4. Use Mundy-specific features such as links, staged device-side modification requests, and accessor expressions.
5. Run host/device algorithms through views, BLAS helpers, link data, or accessor expressions.

If you are new to STK, the main idea is simple: MundyMesh does not replace STK's mesh model, it builds a friendlier
workflow on top of it. This primer starts from the public concepts you use in application code and only then drills
into the STK-shaped vocabulary underneath.

## STK domain model and Mundy vocabulary
MundyMesh follows STK's domain model rather than introducing a separate one. STK provides a runtime-extensible,
heterogeneous, dynamic, ranked mesh system built around entities, parts, fields, selectors, and a separation between
mesh schema and live mesh state. MundyMesh extends that model rather than replacing it.

That viewpoint is useful throughout the package:

| **STK concept** | **What it means** | **Mundy extension** |
|-----------------|-------------------|---------------------|
| `Entity` | A runtime mesh object such as a node, edge, face, element, constraint, or custom-ranked object. | `Link` uses ordinary mesh entities to represent dynamic non-topological relationships. |
| `Rank` | The role of an entity, such as node-rank versus element-rank. | Mundy builds higher-level workflows that still stay rank-aware. |
| `Part` | STK's native grouping mechanism for entities. Parts drive selectors, topology declarations, and IO membership. | `Class` extends parts into a semantic hierarchy like “all rods” or “boundary nodes”. |
| `Field` | Per-entity data attached to a rank and usually restricted by part membership. | `Component` provides a typed accessor into entity data. `FieldComponent` wraps a field; `SharedComponent` holds one value returned for every entity. |
| `Selector` | A query over part membership. | Mundy adds `string_to_selector(...)` and class-aware workflows on top. |
| `MetaData` | The schema of the mesh: what parts, fields, and ranks exist. | Mundy extends it with attributes and helper-driven declaration workflows. |
| `BulkData` | The live mesh state: which entities currently exist and how they are related. | Mundy extends it with links, staged modification helpers, and richer NGP workflows. |

Two Mundy terms appear frequently because they sit directly beside STK concepts rather than replacing them:

| **Mundy term** | **Relationship to STK** |
|----------------|--------------------------|
| `Class` | A semantic layer built on top of coordinated STK parts so that hierarchy, subset logic, and IO behave like a real class system. |
| `Component` / `Aggregate` | A usability layer built on top of STK fields so that field data can be accessed as typed, tagged objects and passed around as one aggregate. |

Three pairings summarize the relationship:

 - `Part` → native STK grouping, `Class` → semantic Mundy grouping.
 - `Field` → native STK storage, `Component` → typed Mundy accessor.
 - STK relations/connectivity → native mesh relations, `Link` → dynamic non-topological relation entity.

Some of these layers exist to reduce STK boilerplate. Others add capabilities that STK does not provide directly,
including dynamic GPU-compatible links and ticketed device-side modification requests.

## Architecture at a glance

| **Layer** | **Main types / functions** | **Use when you need** |
|-----------|-----------------------------|------------------------|
| Mesh construction | `MeshBuilder`, `MetaData`, `BulkData` | A Mundy-aware STK mesh object. |
| Declaration helpers | `FieldDeclarationHelper`, `ComponentDeclarationHelper`, `PartDeclarationHelper`, `ClassDeclarationHelper`, `DeclareEntitiesHelper` | Less boilerplate when setting up a mesh. |
| Semantic structure | `Class`, `declare_class(...)` | A flattened class hierarchy on top of parts that also behaves correctly with STK IO. |
| Field access | `scalar_field_data`, `vector3_field_data`, `quaternion_field_data`, ... | Typed math/geom views into raw field storage. |
| Components / aggregates | `Component`, `FieldComponent`, `SharedComponent`, `Aggregate`, `NgpAggregate` | Storage-independent data access and logical grouping of many accessors. |
| Iteration helpers | `for_each_entity_run`, `for_each_link_run` | Uniform host/device loops over entities or links. |
| String parsing | `string_to_selector`, `string_to_rank`, `string_to_topology` | Config-driven mesh queries and declarations. |
| Dynamic links | `LinkMetaData`, `LinkData`, `NgpLinkData`, `get_updated_ngp_link_data(...)` | A new dynamic GPU-compatible connectivity model distinct from STK relations. |
| Mesh modification staging | `NgpModRequests` | Ticketed device-side requests for later host-side mesh modification. |
| Usability layer | `NgpFieldBLAS`, `NgpAccessorExpr` | Higher-level field math and expression-style device programming. |

## Mesh construction
At the lowest level, MundyMesh extends STK rather than replacing it.

### `MeshBuilder`
`MeshBuilder` mirrors STK's mesh builder, but produces Mundy `MetaData` and `BulkData` objects.

```cpp
stk::ParallelMachine comm = MPI_COMM_WORLD;

mundy::mesh::MeshBuilder builder(comm);
builder.set_spatial_dimension(3)
       .set_auto_aura_option(stk::mesh::BulkData::AUTO_AURA)
       .set_upward_connectivity_flag(true);

std::unique_ptr<mundy::mesh::BulkData> bulk = builder.create_bulk_data();
mundy::mesh::MetaData& meta = bulk->mesh_meta_data();
```

The main builder setters are:

| **Setter** | **Meaning** |
|------------|-------------|
| `set_spatial_dimension(dim)` | Set mesh spatial dimension before creation. |
| `set_entity_rank_names(names)` | Override STK rank names. |
| `set_communicator(comm)` | Choose the MPI communicator. |
| `set_auto_aura_option(option)` | Control aura behavior. |
| `set_bucket_capacity(...)` / `set_initial_bucket_capacity(...)` / `set_maximum_bucket_capacity(...)` | Tune bucket sizes. |
| `set_upward_connectivity_flag(bool)` | Enable or disable upward connectivity. |

Creation methods are:

| **Action** | **Result** |
|------------|------------|
| `create_meta_data()` | Create a standalone Mundy `MetaData`. |
| `create_bulk_data()` | Create a `BulkData` and its metadata. |
| `create_bulk_data(meta_ptr)` | Create `BulkData` around existing Mundy metadata. |

### `MetaData`
`MetaData` currently extends STK mainly through attribute storage.

```cpp
meta.declare_attribute("time_step", 1.0e-3);
meta.declare_attribute(spheres_part, "material", std::string("steel"));
meta.declare_attribute(radius_field, "units", std::string("m"));
```

The attribute interface is symmetric across mesh, part, and field scope:

| **Operation** | **Meaning** |
|---------------|-------------|
| `declare_attribute(name, value)` | Attach metadata to the mesh. |
| `declare_attribute(part, name, value)` | Attach metadata to a part. |
| `declare_attribute(field, name, value)` | Attach metadata to a field. |
| `get_attribute(...)` | Retrieve a stored `std::any*` if present. |
| `remove_attribute(...)` | Erase a stored attribute by name. |

Use mesh attributes for configuration and provenance, not as a replacement for entity fields.

## Reduced-boilerplate declaration helpers
STK setup code is often repetitive. Mundy's declaration helpers exist to reduce that boilerplate while keeping the
underlying STK behavior visible.

### `FieldDeclarationHelper`
Use `FieldDeclarationHelper` when you want to declare a plain STK field through a fluent interface.

```cpp
using namespace mundy::mesh;

FieldDeclarationHelper field_decl(meta);

stk::mesh::Field<double>& coords =
    field_decl.type<double>()
              .rank(stk::topology::NODE_RANK)
              .name("coordinates")
              .role(Ioss::Field::MESH)
              .output_type(stk::io::FieldOutputType::VECTOR_3D)
              .declare();

stk::mesh::Field<double>& velocity =
    field_decl.type<double>()
              .rank(stk::topology::NODE_RANK)
              .name("velocity")
              .role(Ioss::Field::TRANSIENT)
              .output_type(stk::io::FieldOutputType::VECTOR_3D)
              .declare();
```

Because each setter returns a new builder value rather than mutating in place, you can capture a partial
builder and branch from it to declare several related fields with shared attributes:

```cpp
auto transient_vec3 = field_decl.type<double>()
                                 .role(Ioss::Field::TRANSIENT)
                                 .output_type(stk::io::FieldOutputType::VECTOR_3D);

stk::mesh::Field<double>& node_vel   = transient_vec3.rank(stk::topology::NODE_RANK).name("velocity").declare();
stk::mesh::Field<double>& node_force = transient_vec3.rank(stk::topology::NODE_RANK).name("force").declare();
stk::mesh::Field<double>& elem_stress = transient_vec3.rank(stk::topology::ELEM_RANK).name("stress").declare();
```

The required inputs before `declare()` are `type<T>()`, `rank(...)`, and `name(...)`. `role(...)` and
`output_type(...)` are optional.

| **Setter** | **Meaning** |
|------------|-------------|
| `type<T>()` | Choose the scalar storage type. Returns a typed `FieldDeclarationHelperT<T>`. |
| `rank(rank)` | Choose the entity rank for the field. |
| `name("...")` | Choose the field name. |
| `role(role)` | Set the STK IO role such as `Ioss::Field::MESH` or `Ioss::Field::TRANSIENT`. |
| `output_type(type)` | Control component labeling in IO output. |

`FieldDeclarationHelper` declares raw STK fields only. Use `ComponentDeclarationHelper` when the intent
is to declare a field-backed or shared component.

### `ComponentDeclarationHelper`
`ComponentDeclarationHelper` is the preferred entry point when you want to declare a field-backed or
shared-backed component. It offers the same fluent setters as `FieldDeclarationHelper` but routes the
declaration through a typed builder chain that produces a `FieldComponent` or `SharedComponent` rather
than a raw `stk::mesh::Field`.

The backend and access shape are selected together with `.field<A>()` or `.shared<A>(source)`, then
committed with `.declare()`. All other setters
(`rank`, `name`, `role`, `output_type`, `tag`) may appear anywhere in the chain before `.declare()`.

#### Builder state machine

The chain accumulates type information as it goes. Each transition is an implementation-detail type;
use `auto` throughout:

| **Builder** | **State** | **Available transitions** |
|-------------|-----------|---------------------------|
| `ComponentDeclarationHelper` | Nothing fixed | `.rank()` `.name()` `.role()` `.output_type()` `.type<T>()` `.tag<Tag>()` `.field<A>()` `.shared<A>(source)` |
| `TaggedFieldDeclarationHelperT` | Tag and/or field scalar fixed | same backend transitions as `ComponentDeclarationHelper` |
| `TaggedFieldBackedDeclarationHelperT` | Access + field backend | `.rank()` `.name()` `.role()` `.output_type()` `.type<T>()` `.tag<Tag>()` `.declare()` |
| `TaggedSharedComponentDeclarationHelperT` | Access + shared backend | `.rank()` `.name()` `.tag<Tag>()` `.declare()` |

`.field<A>()` and `.shared<A>(source)` are the backend commitment steps. The `A` is the access shape.

#### Access shapes

The `AccessLike` argument to `.field<>()` or `.shared<>()` selects the component shape. Explicit `access::` tag types
and their corresponding Mundy math/geometry types are both accepted:

| **`AccessLike`** | **Also accepts** | **View type per entity** | **Field scalars** |
|------------------|------------------|--------------------------|-------------------|
| `access::scalar<T>` | arithmetic `T` | `ScalarWrapper<T>` | 1 |
| `access::vector<T, N>` | `Vector<T, N>` | `Vector<T, N>` | N |
| `access::matrix3<T>` | `Matrix3<T>` | `Matrix3<T>` | 9 |
| `access::quaternion<T>` | `Quaternion<T>` | `Quaternion<T>` | 4 |
| `access::aabb<T>` | `AABB<T>` | `AABB<T>` | 6 |
| `access::raw<T>` | — | `T&` | unspecified |

`access::raw<T>` is the escape hatch: it maps to a plain `FieldComponent<T>` and makes no assumptions
about how the field scalars compose. Use it when no existing shape matches your storage layout.

The scalar type carried by the `AccessLike` determines the STK field scalar type. Providing `.type<T>()`
separately is allowed but must agree with the access shape — a compile-time assertion catches mismatches.

#### Field-backed component declaration

Call `.field<A>()` to commit to a field-backed component, then `.declare()`:

```cpp
using namespace mundy::mesh;

ComponentDeclarationHelper decl(meta);

// Untagged scalar field component
auto mass =
    decl.rank(stk::topology::ELEM_RANK)
        .name("mass")
        .role(Ioss::Field::ATTRIBUTE)
        .field<access::scalar<double>>()
        .declare();

// Tagged vector3 field component
auto center =
    decl.rank(stk::topology::NODE_RANK)
        .name("center")
        .role(Ioss::Field::TRANSIENT)
        .field<access::vector<double, 3>>()
        .tag<CENTER>()
        .declare();
```

`.tag<Tag>()` may be called before or after `.field<>()` — the two orderings are equivalent:

```cpp
decl.field<access::quaternion<double>>().tag<ORIENT>().declare();  // equivalent
decl.tag<ORIENT>().field<access::quaternion<double>>().declare();  // to this
```

When a tag is present, `.declare()` returns `TaggedComponent<Tag, ComponentType>`. Without a tag it
returns the plain `ComponentType` directly.

#### Shared-backed component declaration

Call `.shared<A>(source)` to commit to a shared-backed component. `source` can be a raw value
(stored by copy) or a length-1 Kokkos view in `HostSpace` (aliased directly):

```cpp
using namespace mundy::mesh;

ComponentDeclarationHelper decl(meta);

// Shared scalar: one collision radius for all elements in a part
auto collision_radius =
    decl.rank(stk::topology::ELEM_RANK)
        .name("collision_radius")
        .tag<COLLISION_RADIUS>()
        .shared<access::scalar<double>>(0.5)
        .declare();
```

Shared declarations require `.rank()` before `.declare()`. The rank is recorded in the component's
metadata so the component knows which entity rank it applies to when attached to an aggregate or
part-restricted context. An assertion fires at `.declare()` time if rank is missing.

The order of `.shared<>()` relative to other setters is free:

```cpp
// backend-first
decl.shared<double>(1.5).rank(ELEM_RANK).name("radius").declare();

// name/rank-first
decl.rank(ELEM_RANK).name("radius").shared<double>(1.5).declare();
```

#### Snapshot reuse

Because each setter returns the builder by value, a partial builder can be captured and reused to declare
multiple components that share common attributes:

```cpp
using namespace mundy::mesh;

ComponentDeclarationHelper decl(meta);

// Capture common attributes at the node transient vector3 level.
// field<access::vector<double, 3>>() and field<Vector3<double>>() are equivalent.
auto node_transient_vec3 =
    decl.rank(stk::topology::NODE_RANK)
        .role(Ioss::Field::TRANSIENT)
        .field<Vector3<double>>();

// Branch with distinct names and tags
auto center   = node_transient_vec3.name("center").tag<CENTER>().declare();
auto velocity = node_transient_vec3.name("velocity").tag<LIN_VEL>().declare();
auto force    = node_transient_vec3.name("force").tag<FORCE>().declare();
```

### `PartDeclarationHelper`
Use `PartDeclarationHelper` to declare named, ranked, or topological parts and optionally attach restrictions.

```cpp
using namespace mundy::mesh;

PartDeclarationHelper part_decl(meta);

stk::mesh::Part& spheres =
    part_decl.name("spheres")
             .topology(stk::topology::PARTICLE)
             .role(IOPartRole::IO)
             .declare();

stk::mesh::Part& rigid_bodies =
    part_decl.name("rigid_bodies")
             .rank(stk::topology::ELEM_RANK)
             .role(IOPartRole::ASSEMBLY)
             .subpart(spheres)
             .declare();
```

The allowed declaration modes are explicit:

| **Mode** | **Inputs** | **Result** |
|----------|------------|------------|
| Named part | `name` only | `meta.declare_part(name)` |
| Ranked part | `name + rank` | `meta.declare_part(name, rank)` |
| Topological part | `name + topology` | `meta.declare_part_with_topology(name, topology)` |

Useful extras include:

| **Helper** | **Meaning** |
|------------|-------------|
| `role(IOPartRole::IO)` | Make the part an IO part. |
| `role(IOPartRole::ASSEMBLY)` | Make the part an assembly. |
| `subpart(part)` | Declare subset relations. |
| `put_field(field, ...)` | Attach field restrictions with initial values. |
| `put_component(component, ...)` | Attach restrictions from field-backed components. |

### `ClassDeclarationHelper` and `declare_class(...)`
Classes are a core MundyMesh abstraction, so they have both direct and helper-based APIs.

```cpp
using namespace mundy::mesh;

Class& spheres = declare_class(meta, "spheres", stk::topology::PARTICLE);
Class& all_nodes = declare_class(meta, "all_nodes", stk::topology::NODE_RANK);

ClassDeclarationHelper class_decl(meta);
Class& boundary_nodes = class_decl.name("boundary_nodes").rank(stk::topology::NODE_RANK).declare();
Class& loading_nodes = class_decl.name("loading_nodes").rank(stk::topology::NODE_RANK).declare();
Class& special_nodes = class_decl.name("special_nodes")
                                 .rank(stk::topology::NODE_RANK)
                                 .subclass(boundary_nodes)
                                 .subclass(loading_nodes)
                                 .declare();
```

As with parts, classes are declared in one of three ways:

| **Mode** | **Inputs** | **Interpretation** |
|----------|------------|--------------------|
| Named class | `name` only | Reuse or complete a compatible class declaration by name. |
| Ranked class | `name + rank` | Create a rank-specific set class. |
| Topological class | `name + topology` | Create a primary class with a topology. |

### `DeclareEntitiesHelper`
`DeclareEntitiesHelper` is a host-side builder for declaring nodes, elements, classes, field data, and link
relationships without handling ownership and sharing details in each call.

```cpp
mundy::mesh::DeclareEntitiesHelper builder;

builder.create_node().owning_proc(0).id(1);
builder.create_node().owning_proc(0).id(2);

builder.create_element()
       .owning_proc(0)
       .id(10)
       .topology(stk::topology::BEAM_2)
       .nodes({1, 2});

builder.declare_entities(*bulk);
```

This helper is useful when you want a deterministic serial description of a mesh and then need the corresponding
sharing and connectivity declared on all ranks.

## Classes: semantic structure on top of STK IO
STK IO works with a hierarchy of disjoint parts. Mundy's `Class` layer builds on top of that representation so code can
be written more like a flattened class hierarchy while still satisfying STK's IO rules.

### Why `Class` exists
Many multibody models need names such as “all rods”, “boundary nodes”, “contact links”, or “bead centers” rather
than raw part membership. `Class` wraps the part hierarchy needed to make those concepts consistent with STK's IO and
subset rules.

A `Class` is the semantic object. Its underlying STK parts are the implementation detail that keeps selectors, subset
relations, inherited data behavior, and IO consistent.

This matters because parts already contain much of the raw machinery needed for class-like behavior, but making that
behave like a usable class hierarchy in the presence of STK IO takes additional logic. `Class` automates that logic so
algorithms can be written against a class and still apply naturally to entities in its subclasses.

### Primary classes vs sets

| **Kind** | **How declared** | **Membership rule** |
|----------|------------------|----------------------|
| Primary class | Declared with a topology | An entity can belong to only one primary class of the same rank. |
| Set | Declared with a rank but no topology | An entity may belong to zero or more sets of that rank. |

The main consequences are:

 - Primary classes encode mutually-exclusive concrete kinds like particles, beams, shells, or linker types.
 - Sets encode semantic overlays like boundary nodes, loaded elements, or output subsets.
 - Subclasses inherit data and selector membership from their parent classes.

### Core `Class` queries

| **Query** | **Meaning** |
|-----------|-------------|
| `name()` | Semantic class name. |
| `primary_entity_rank()` | Rank of the class. |
| `topology()` | STK topology if it is a primary class. |
| `class_type()` / `is_primary()` / `is_set()` | Distinguish primary classes from sets. |
| `has_io_support()` | Whether IO hierarchy rules are active. |
| `data_part()` / `leaf_part()` / `assembly_part()` / `leaf_assembly_part()` | Underlying STK parts used to implement the class. |

Use classes when you want semantic, class-like code and raw parts when you need direct STK interoperation.

## String parsing helpers
MundyMesh provides a set of parsing and configuration utilities for turning strings into STK objects.

### `string_to_selector`
`string_to_selector(bulk_data, selector_string)` parses selector expressions built from part names.

Supported operators are the usual selector math:

| **Operator** | **Meaning** |
|--------------|-------------|
| `|` | union |
| `&` | intersection |
| `!` | complement |
| `(` `)` | grouping |

```cpp
auto sel = mundy::mesh::string_to_selector(*bulk, "(rods | spheres) & !ghosted");
```

This is mainly useful for user input, parameter lists, and tests.

### `string_to_rank` and `string_to_topology`

```cpp
#include <mundy_mesh/StringToRank.hpp>
#include <mundy_mesh/StringToTopology.hpp>

auto rank = mundy::mesh::string_to_rank("NODE_RANK");
auto topo = mundy::mesh::string_to_topology("HEX_8");
```

Use these when mesh configuration is text-driven rather than hard-coded.

## Typed views into field data
Raw `stk::mesh::field_data(...)` returns raw pointers. `FieldViews.hpp` maps that storage into Mundy math and geometry
types.

### Host entity access

```cpp
auto x = mundy::mesh::vector3_field_data(coords, node);
auto v = mundy::mesh::vector3_field_data(velocity, node);

x += dt * v;
```

### Device `FastMeshIndex` access

```cpp
auto ngp_coords = stk::mesh::get_updated_ngp_field<double>(coords);

KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& i) {
  auto x = mundy::mesh::vector3_field_data(ngp_coords, i);
  x[0] += 1.0;
};
```

### Common field-view helpers

| **Helper** | **Interprets entity field storage as** |
|------------|----------------------------------------|
| `scalar_field_data(...)` | `ScalarWrapper` / scalar view |
| `vector_field_data<N>(...)` | `Vector<N>`-like view |
| `vector3_field_data(...)` | `Vector3` |
| `matrix_field_data<N, M>(...)` | `Matrix<N, M>` |
| `matrix3_field_data(...)` | `Matrix3` |
| `quaternion_field_data(...)` | `Quaternion` |
| `aabb_field_data(...)` | `AABB` |

Use these helpers when a field is logically a math object, not just a flat array.

## Components and aggregates

### Components
Components provide algorithmic separation of concerns: most algorithms that act on entities care only about accessing a
given data object (a center, a radius, a velocity, an orientation) for a given entity, not about *how* that data is
stored. They rarely care whether a matrix is stored row-major vs. column-major or whether a radius is shared by all
entities vs. stored as a field value on each entity.

This is achieved through a simple set of accessor classes called `Component`s, each of which offers an access operator
`operator()(entity)` that returns a view into the data for that entity of the given semantic type. For example,
```cpp
// center_field is an existing stk::mesh::Field<double>; 0.5 is the single shared collision radius
auto center_accessor = Vector3FieldComponent<double>(center_field);
auto radius_accessor = SharedScalarComponent<double>(0.5);

auto center = center_accessor(node1);          // Vector3 view into the center field for node1
auto collision_radius = radius_accessor(elem1); // ScalarWrapper view returning 0.5 for any entity
```

In both cases the calling code asks only for the desired semantic quantity. The accessor itself determines where that
data comes from and how it is presented. These are lightweight views, not values or references. Their explicit type
should be treated as an implementation detail of the accessor. What matters is that their behavior matches the
semantic quantity they represent.
```cpp
center_accessor(node1) += dt * velocity_accessor(node1);  // component-wise addition on the center vector
```

Components themselves are views into data rather than owners of that data. They are cheap to construct and cheap to copy. 
Like STK `NgpField`, they offer a unified interface for synchronization: a component exposes `sync_to_host()`,
`sync_to_device()`, `modify_on_host()`, and `modify_on_device()` so algorithms can manage host/device coherence
without caring where the underlying data came from.

| **Supported view type** | **Field-backed component** | **Shared component** | **What `operator()(entity)` acts like** |
|-------------------------|------------------------------|----------------------|----------------------------------------|
| Scalar | `ScalarFieldComponent<T>` | `SharedScalarComponent<T>` | `ScalarWrapper<T>` |
| Vector of length N | `VectorFieldComponent<T, N>` | `SharedVectorComponent<T, N>` | `Vector<T, N>` |
| Matrix3 | `Matrix3FieldComponent<T>` | `SharedMatrix3Component<T>` | `Matrix3<T>` |
| Quaternion | `QuaternionFieldComponent<T>` | `SharedQuaternionComponent<T>` | `Quaternion<T>` |
| AABB | `AABBFieldComponent<T>` | `SharedAABBComponent<T>` | `AABB<T>` |

Fixed-length aliases `Vector1FieldComponent<T>` through `Vector6FieldComponent<T>` are provided for the most common vector sizes.

**Pre-defined semantic tags.** `Component.hpp` ships a set of common tag types so you do not have to define your own for the standard physical quantities. The pre-defined tags are: `CENTER`, `ORIENT`, `LIN_VEL`, `ANG_VEL`, `FORCE`, and `COLLISION_RADIUS`. These are plain forward-declared structs; they carry no data and impose no behavior. You can define additional tag types anywhere in your own code using the same pattern.

At present, these patterns are supported for field-backed and shared components. We intend to
offer the same interface through `PartMappedComponent`s in the future so that per-part access policies can be exposed
through the same semantic accessor model.

Just like STK's `NgpField`s, each component is backed by an NGP/Kokkos-compatible counterpart that can be obtained
through `get_updated_ngp_component(...)`. For these components, the access operator is instead `operator()(FastMeshIndex)` and 
returns a Kokkos-compatible, performance-portable view into the data for the given entity.

As with `NgpField`, these ngp components have a lifetime that is valid until the next mesh modification cycle, so the
best pattern is to pass around raw components, fetch their updated ngp components immediately before entering a
Kokkos kernel, and use the ngp component inside the kernel. If no modification has occurred, `get_updated_ngp_component(...)`
will simply return a reference to the existing ngp component, making it inexpensive to call repeatedly.
```cpp
auto ngp_center_accessor = mundy::mesh::get_updated_ngp_component(center_accessor);
auto ngp_velocity_accessor = mundy::mesh::get_updated_ngp_component(velocity_accessor);
mundy::mesh::for_each_entity_run(ngp_mesh, stk::topology::NODE_RANK, selector,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& i) {
            auto center = ngp_center_accessor(i);
            auto velocity = ngp_velocity_accessor(i);
            center += dt * velocity;
        });
```

### `Aggregate`
An `Aggregate` is a logical collection of tagged components, along with the `BulkData` and `Selector` describing the
subset of the mesh on which they are intended to act.

In practice, an aggregate answers two questions:
1. Which entities are we iterating over?
2. Which semantic components are available on those entities?

This makes aggregates a natural way to package the data requirements of a system. If a collision algorithm needs a
center and a collision radius, we do not define a hard-coded `Sphere` class. We instead assemble an aggregate that
contains a `CENTER` component and a `COLLISION_RADIUS` component and pass that aggregate into the algorithm.

```cpp
using namespace mundy::mesh;

// center_field is a stk::mesh::Field<double> declared elsewhere; 0.5 is the shared collision radius
auto center_accessor = Vector3FieldComponent<double>(center_field);
auto radius_accessor = SharedScalarComponent<double>(0.5);

auto sphere_data = Aggregate<>(bulk_data, selector)
  .add_component<CENTER>(center_accessor)
  .add_component<COLLISION_RADIUS>(radius_accessor);

sphere_data.sync_to_device<CENTER, COLLISION_RADIUS>();

stk::mesh::Entity center_node = bulk_data.begin(sphere, stk::topology::NODE_RANK)[0];
auto c = sphere_data.get<CENTER>(center_node);
auto r = sphere_data.get<COLLISION_RADIUS>(sphere);
```

The core operations are:

| **Operation** | **Meaning** |
|---------------|-------------|
| `.add_component<Tag>(component)` | Return a new aggregate with one more tagged component. |
| `.add_component(tagged_component)` | Append a component that already carries its tag. |
| `get_component<Tag>()` | Fetch the accessor object itself. |
| `get<Tag>(entity)` | Read/write the tagged value for a host entity. |
| `sync_to_device<Tags...>()` | Sync selected components to device. |
| `sync_to_host<Tags...>()` | Sync selected components to host. |
| `modify_on_device<Tags...>()` | Mark selected components modified on device. |
| `modify_on_host<Tags...>()` | Mark selected components modified on host. |

The main benefit is that a system can depend on a small semantic interface instead of a long list of concrete storage
objects. Instead of threading several accessors through every function, you pass one aggregate and ask it for the tags
you need.

Like `NgpField`, `Aggregate` is the synchronization-owning object. For Kokkos-compatible, performance-portable code,
fetch its NGP counterpart with `get_updated_ngp_aggregate(...)` immediately before entering a kernel.
`NgpAggregate` provides the same tagged access pattern using `FastMeshIndex`, and if no modification has occurred,
refreshing it is inexpensive.

```cpp
auto ngp_sphere_data = mundy::mesh::get_updated_ngp_aggregate(sphere_data);
```

## Dynamic links
Links are one of the major features added by MundyMesh.

### Links vs STK connectivity
Links are non-topological connections managed by an STK entity itself. They are not just “extra edges”. They are a
new mesh-side relationship model with a weaker but very useful parallel contract.

The most important conceptual point is that a link behaves like a **fixed-size ghosting contract** between the link
entity and the entities stored in its slots. In practical terms, Mundy maintains things so that a locally owned or
shared link and the entities it links have at least ghosted visibility to one another. That is what makes links useful
for dynamic non-topological relationships: you can change who occupies a slot without rebuilding ordinary STK
connectivity or treating the relationship as mesh topology.

The most useful mental model is still:

 - a **link** is an ordinary mesh entity of some chosen rank,
 - each link has a fixed number of slots,
 - each slot may point to one linked entity,
 - those slot values may be changed later without recreating the link entity.

That is why links are useful for dynamic relationships that are not mesh topology: contact pairs, parent-child ties,
attachments, temporary neighborhoods, or grouped objects that need link-local data.

Another good way to say it is that links are **dynamic non-topological connectivity**. They express “these entities are
currently related in an algorithmically meaningful way” without claiming that they are topologically connected in the
STK sense.

Unlike STK connectivity, links:

 - do not change ownership or sharing rules,
 - may connect entities of arbitrary ranks,
 - are themselves entities, so they may carry fields, parts, classes, and IO behavior,
 - support both **link → linked entity** access and **linked entity → incident links** access.

Those differences matter because many relationships in multibody or particle-style problems are not well modeled as
permanent mesh connectivity. A contact pair, a point attached to a surface patch, or a temporary neighborhood graph may
need data to live **on the relation itself**, may need to change frequently, and may connect objects of unrelated rank
or topology. Links are designed for exactly that use case.

Internally, Mundy keeps two views of the same relationship:

| **Representation** | **What it is good at** |
|--------------------|------------------------|
| COO-like | Editing or directly querying a link's slot contents. |
| CSR-like | Reverse traversal, consistency checks, and algorithms that need to start from the linked entity side. |

In practice, you edit the COO side first, then explicitly rebuild the CSR side when you need CSR-backed behavior.
That split is not incidental; it is part of why links are both dynamic and scalable. The COO side is the easy-to-edit
representation, while the CSR side is the heavier reverse-traversal structure derived from it.

### `LinkMetaData`
`LinkMetaData` declares the fields and class structure needed for a family of links. In most user code, this is the
schema step done before `meta.commit()`.

```cpp
using namespace mundy::mesh;

LinkMetaData& contact_links =
    declare_link_meta_data(meta, "contact_links", stk::topology::CONSTRAINT_RANK);

stk::mesh::Part& sphere_contacts = contact_links.declare_link_part("sphere_contacts", 2);
```

Its main jobs are:

| **Responsibility** | **Meaning** |
|--------------------|-------------|
| declare link-compatible parts/classes | Ensure link fields and subsets are attached consistently. |
| own link bookkeeping fields | Store linked ids, linked ranks, CSR caches, destruction flags, etc. |
| provide `universal_link_class()` | Give a selector/class root for all links in that family. |

### `LinkData`
`LinkData` is the runtime object that manages the actual link relations on a `BulkData`.

```cpp
LinkData& link_data = declare_link_data(*bulk, contact_links);
```

The most important distinction is:

 - use **ordinary STK mesh modification** to create or destroy the **link entities themselves**,
 - use `LinkData` to create or destroy the **relations stored on those links**.

### Minimal host workflow
The explicit host-side workflow usually looks like this:
1. Declare `LinkMetaData` before commit.
2. Construct `LinkData` for a live `BulkData`.
3. Enter a mesh modification cycle only to create the link entity itself.
4. Exit the modification cycle.
5. Call `declare_relation(...)` or `destroy_relation(...)` on the COO side.
6. Mark the COO side modified.
7. Rebuild CSR only when some later algorithm needs it.

```cpp
using namespace mundy::mesh;

LinkMetaData& contact_links =
    declare_link_meta_data(meta, "contact_links", stk::topology::CONSTRAINT_RANK);
stk::mesh::Part& sphere_contacts = contact_links.declare_link_part("sphere_contacts", 2);

LinkData& link_data = declare_link_data(*bulk, contact_links);

// The targets already exist on the mesh.
stk::mesh::Entity node_a = /* existing entity */;
stk::mesh::Entity node_b = /* existing entity */;

// Create the link entity using normal STK mesh modification.
bulk->modification_begin();
stk::mesh::PartVector link_parts{&sphere_contacts};
stk::mesh::Entity link = bulk->declare_entity(contact_links.link_rank(), 100, link_parts);
bulk->modification_end();

// Populate the link's slots outside the modification cycle.
link_data.coo_data().declare_relation(link, node_a, 0u);
link_data.coo_data().declare_relation(link, node_b, 1u);
link_data.coo_modify_on_host();

// Immediate link -> linked-entity queries use the COO side.
auto first = link_data.coo_data().get_linked_entity(link, 0u);
auto second = link_data.coo_data().get_linked_entity(link, 1u);
```

### Common operations

| **Operation family** | **Meaning** |
|----------------------|-------------|
| `coo_data().declare_relation(link, linked, ordinal)` | Fill or overwrite one slot on a link. |
| `coo_data().destroy_relation(link, ordinal)` | Clear one slot on a link. |
| `coo_data().get_linked_entity(link, ordinal)` | Read one slot immediately from the COO side. |
| `coo_modify_on_host()` / `coo_modify_on_device()` | Tell the synchronizer which side performed the edit. |
| `coo_sync_to_host()` / `coo_sync_to_device()` | Push COO edits across host/device memory. |
| `update_crs_from_coo()` | Rebuild reverse connectivity after COO edits. |
| `crs_sync_to_host()` / `crs_sync_to_device()` | Move the CSR mirror when needed. |

### When you need `update_crs_from_coo()`
This is the step the old text tended to hide.

You **do not** need CSR just to ask “what does this link point to right now?” because COO already answers that.

You **do** need CSR rebuilt when you are about to do work that depends on the reverse. A safe rule is:

1. edit relations on the COO side,
2. call `update_crs_from_coo()` before reverse traversal, consistency checks, or CSR-backed device workflows.

Like other mesh-side access objects, `LinkData` has an NGP counterpart for Kokkos-compatible,
performance-portable work. Fetch it with `get_updated_ngp_link_data(...)` immediately before a kernel when you need
link access, CSR/COO checks, or performance-portable traversal. `NgpLinkData` mirrors the same interface in NGP
memory space, and if the CSR side is stale you can refresh it there before use.

```cpp
auto& ngp_link_data = mundy::mesh::get_updated_ngp_link_data(link_data);

ngp_link_data.coo_sync_to_device();

if (!ngp_link_data.is_crs_up_to_date()) {
  ngp_link_data.update_crs_from_coo();
}
```

### Link-centric traversal
Once relations exist, iterating over links is explicit and simple.

```cpp
for_each_link_run(link_data, sphere_contacts,
                  [&](const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& linker) {
                    auto a = link_data.coo_data().get_linked_entity(linker, 0u);
                    auto b = link_data.coo_data().get_linked_entity(linker, 1u);

                    if (bulk_data.is_valid(a) && bulk_data.is_valid(b)) {
                      // do host-side work with the link and its two targets
                    }
                  });
```

On device, the pattern is the same except that the loop body receives a `FastMeshIndex` and uses the NGP COO data:

```cpp
auto& ngp_link_data = mundy::mesh::get_updated_ngp_link_data(link_data);
ngp_link_data.coo_sync_to_device();

for_each_link_run(ngp_link_data, sphere_contacts,
                  KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& linker) {
                    auto a = ngp_link_data.coo_data().get_linked_entity(linker, 0u);
                    auto b = ngp_link_data.coo_data().get_linked_entity(linker, 1u);
                    // do device-side work
                  });
```

## Staged mesh modification from device code
`NgpModRequests` is Mundy's ticket-based framework for device-originated mesh changes.

### The three-stage model
This is not just a small helper around STK mesh modification. It adds a new workflow in which device code can request
mesh changes through tickets, those requests are delayed, and they are later fulfilled together in a single host
modification cycle.

The public model has three stages:

1. **Claim tickets**: count how many requests will be made.
2. **Make requests**: use those tickets to describe the requested entities or connections.
3. **Fetch results**: after processing, recover the created entities associated with each ticket.

### Typical workflow

```cpp
mundy::mesh::NgpModRequests reqs;
auto req_entities = reqs.request_entities_new_ids(spheres);
auto req_conns = reqs.request_connections();

reqs.activate_host();
req_entities.element_tickets().claim(num_spheres);
req_entities.node_tickets().claim(num_spheres);
req_conns.tickets().claim(num_spheres);
reqs.finalize_counts();

reqs.activate_device();
// request_element(ticket), request_node(ticket), request(...)

reqs.process_requests(*bulk);

// Later: req_entities.get_entity(ticket, rank)
```

### Main request families

| **Factory** | **Use** |
|-------------|---------|
| `request_entities_new_ids(parts)` | Request new entities with generated ids. |
| `request_entities_known_ids(parts)` | Request entities whose ids are already known. |
| `request_connections()` | Request new connectivity between entities or future entities. |

### Control methods

| **Method** | **Meaning** |
|------------|-------------|
| `activate_host()` | Make the host side the active ticket/request space. |
| `activate_device()` | Make the device side the active ticket/request space. |
| `finalize_counts()` | Freeze counts and allocate request storage. |
| `process_requests(bulk_data)` | Enter the host-side processing phase and realize requests. |

`NgpModRequests` lets kernels describe requested mesh changes on device and then has the host realize them safely in a
later modification phase.

## Field BLAS
`NgpFieldBLAS.hpp` provides BLAS-like operations over `stk::mesh::FieldBase` objects with unified host/device style.

### Common operations

| **Operation** | **Meaning** |
|---------------|-------------|
| `field_fill(alpha, field, ...)` | Fill one field or one component with a scalar. |
| `field_randomize(seed, counter_field, field, ...)` | Randomize field entries using Philox. |
| `field_copy(x, y, ...)` | Deep-copy one field into another. |
| `field_swap(x, y, ...)` | Swap field contents. |
| `field_scale(alpha, x, ...)` | Scale a field by a scalar. |
| `field_product(x, y, z, ...)` | Elementwise product. |
| `field_axpy(alpha, x, y, ...)` | `y += alpha x`. |
| `field_axpby(alpha, x, beta, y, ...)` | `y = alpha x + beta y`. |
| `field_axpbyz(alpha, x, beta, y, z, ...)` | `z = alpha x + beta y`. |
| `field_axpbygz(alpha, x, beta, y, gamma, z, ...)` | `z = alpha x + beta y + gamma z`. |
| `field_dot(x, y, ...)` | Global dot product. |
| `field_nrm2(x, ...)` | Global 2-norm. |
| `field_sum(x, ...)` / `field_max(x, ...)` / `field_min(x, ...)` | Standard global reductions. |
| `field_asum(x, ...)` / `field_amax(x, ...)` / `field_amin(x, ...)` | Absolute-value reductions. |

These are useful when your algorithm wants whole-field linear algebra rather than per-entity accessors.

## Accessor expressions

Accessor expressions let you write mesh-field arithmetic the same way you would write it against plain
math objects, without manually managing field synchronization, kernel boundaries, or per-entity loops.
The expression is *lazy*—nothing runs until an accessor expression appears as an *lvalue* (on the
left-hand side of an assignment) or until the expression tree is passed to a reduction. At evaluation
time, Mundy inspects the full expression tree, syncs every field it touches to the device (marking output
fields as modified on device), and launches a single Kokkos kernel.

A quick comparison illustrates the intent. The manual approach:

```cpp
rod_agg.sync_to_device<CENTER, VELOCITY>();
stk::mesh::for_each_entity_run(
    ngp_mesh, rod_selector, ELEM_RANK, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& rod_index) {
      auto nodes    = ngp_mesh.get_connected_entities(rod_index, NODE_RANK);
      auto center   = rod_agg.get<CENTER>(nodes[0]);
      auto velocity = rod_agg.get<VELOCITY>(nodes[0]);
      center += dt * velocity;
    });
rod_agg.modify_on_device<CENTER>();
```

The expression approach:

```cpp
auto rods  = make_entity_expr(bulk_data, rod_selector, ELEM_RANK);
auto nodes = rods.get_connectivity(NODE_RANK);
center(nodes[0]) += dt * velocity(nodes[0]);
```

Field sync and marking are automatic. The loop body is gone.

### Evaluation

The system has two kinds of objects: *entity expressions* define what to iterate over, and *accessor
expressions* are lazy handles to field data for those entities. You build up a tree of accessor
expressions using arithmetic and named functions, then trigger evaluation in one of four ways:

- `accessor(entity_expr) = rhs_expr;` — direct assignment
- `accessor(entity_expr) += rhs_expr;` — compound assignment (also `-=`, `*=`, `/=`)
- `fused_assign(lhs1, rhs1, lhs2, rhs2, ...)` — fused multi-assignment (see below)
- `reduce_local_*(expr)` or `all_reduce_*(expr)` — reduction

Accessor expressions are *lvalues*: when one appears on the left-hand side of an assignment, the
right-hand side expression tree is evaluated and its result written back to the underlying field. Each
trigger produces exactly one kernel launch. Before the kernel runs, all input fields are synced to
device and all output fields are marked as modified on device.

```cpp
x(es) = 3.14;                          // fill
x(es) = y(es);                         // copy
x(es) *= alpha;                        // scale in-place
x(es) = alpha * x(es) + beta * y(es); // axpby
z(es) = x(es) * y(es);                // element-wise product
```

### Entity expressions

An entity expression defines *what to iterate over*. Create one with `make_entity_expr`:

```cpp
auto es = make_entity_expr(bulk_data, selector, stk::topology::NODE_RANK);
```

You can also express connectivity—iterating over elements while reading data from connected nodes:

```cpp
auto elements = make_entity_expr(bulk_data, rod_selector, stk::topology::ELEMENT_RANK);
auto nodes = elements.get_connectivity(stk::topology::NODE_RANK);
// nodes[0] is the first connected node of each element, nodes[1] the second, and so on.
```

For pairwise algorithms (e.g., neighbor lists), there is a corresponding pairwise variant:

```cpp
auto pairs = make_pairwise_entity_expr(bulk_data, left_rank, right_rank, pair_view, fmi_extractor);
```

### Accessor expression objects

A component accessor applied to an entity expression produces an *accessor expression*: a lazy, typed
handle to the field data for those entities. Each component type corresponds to the math type stored in
the field:

```cpp
auto x   = make_tagged_component<XTag>(ScalarFieldComponent(*x_field));
auto vel = make_tagged_component<VTag>(Vector3FieldComponent(*vel_field));
auto q   = make_tagged_component<QTag>(QuaternionFieldComponent(*q_field));

auto es     = make_entity_expr(bulk_data, selector, NODE_RANK);
auto x_expr = x(es);    // lazy handle to x for every entity in es
```

Alternatively, components can be accessed through an `Aggregate`:

```cpp
aggregate.get<XTag>(es) = aggregate.get<YTag>(es);
```

Accessor expressions compose freely with arithmetic and named functions before anything runs.

### Arithmetic operators

Within an expression, the standard arithmetic operators produce new expression nodes rather than computing
immediately. Scalars on either side are promoted automatically, so mixing scalars and expressions works
without any manual wrapping.

| **Operator** | **Effect** |
|---|---|
| `a + b` | element-wise addition |
| `a - b` | element-wise subtraction |
| `a * b` | scalar multiplication, or matrix-vector / matrix-matrix product |
| `a / b` | element-wise scalar division |
| `a += b`, `a -= b`, `a *= b`, `a /= b` | in-place variants (trigger evaluation immediately) |

### Named functions

The builtins below produce expression nodes and can be freely combined with arithmetic before anything runs.

#### Vectors

| **Function** | **Description** |
|---|---|
| `copy(v)` | explicit value copy (important for swaps; see Fused assignment) |
| `sum(v)`, `product(v)`, `min(v)`, `max(v)` | component reductions |
| `mean(v)`, `variance(v)`, `stddev(v)` | component statistics |
| `norm(v)` | Euclidean (2-)norm |
| `one_norm(v)`, `two_norm(v)`, `inf_norm(v)`, `infinity_norm(v)` | 1-, 2-, and ∞-norms |
| `norm_squared(v)`, `two_norm_squared(v)` | squared norms |
| `dot(v1, v2)` | dot product |
| `cross(v1, v2)` | cross product (3D vectors) |
| `outer_product(v1, v2)` | outer product → matrix |
| `elementwise_mul(v1, v2)`, `elementwise_div(v1, v2)` | component-wise multiply / divide |
| `minor_angle(v1, v2)`, `major_angle(v1, v2)` | angle between vectors |

#### Matrices

| **Function** | **Description** |
|---|---|
| `copy(m)` | explicit value copy |
| `sum(m)`, `product(m)`, `min(m)`, `max(m)`, `mean(m)`, `variance(m)`, `stddev(m)` | element reductions |
| `trace(m)`, `determinant(m)`, `transpose(m)` | standard matrix operations |
| `inverse(m)`, `adjugate(m)`, `cofactors(m)` | inverse and cofactor operations |
| `frobenius_norm(m)`, `one_norm(m)`, `two_norm(m)`, `inf_norm(m)`, `infinity_norm(m)` | matrix norms |
| `frobenius_inner_product(m1, m2)` | Frobenius inner product |
| `elementwise_mul(m1, m2)`, `elementwise_div(m1, m2)` | component-wise multiply / divide |

#### Quaternions

| **Function** | **Description** |
|---|---|
| `copy(q)` | explicit value copy |
| `sum(q)`, `product(q)`, `min(q)`, `max(q)`, `mean(q)`, `variance(q)`, `stddev(q)` | component reductions |
| `norm(q)`, `norm_squared(q)` | norms |
| `inverse(q)`, `conjugate(q)`, `normalize(q)` | quaternion-specific operations |
| `dot(q1, q2)` | dot product |
| `slerp(q1, q2)` | spherical linear interpolation |

#### Scalars

The following pointwise functions lift standard math operations to scalar expressions:
`abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`.

#### Mutating functions

Some operations mutate their argument in-place and run a kernel immediately rather than building an
expression node. These are exposed as named wrappers:

```cpp
rotate_quaternion(q(es), omega(es), dt);  // modifies q; reads omega and dt; runs immediately
```

### Automatic sub-expression reuse

Within any single kernel—whether a simple assignment, a `fused_assign`, or a reduction—the compiler
identifies sub-expression nodes that are structurally identical and whose type guarantees that the same
entity always yields the same result. Any such node that appears more than once in the tree is evaluated
only once, with zero runtime overhead and no branching. Field reads and entity lookups benefit from this.
For example, `norm(vel(es))` appearing twice in a single expression is computed once per entity:

```cpp
x(es) = norm(vel(es)) / (1.0 + norm(vel(es)));  // norm(vel) computed once
```

Constants captured at expression-build time are treated as potentially distinct regardless of their
apparent equality, so they do not qualify for compile-time reuse. The system handles this conservatively
and automatically—you do not need to annotate anything.

### Fused assignment

`fused_assign` packs any number of assignments into a single kernel. All right-hand sides are evaluated
before any left-hand side is written, which makes simultaneous updates safe—in the same spirit as Python's
tuple assignment `x, y = y, x`. For example, a field swap:

```cpp
fused_assign(x(es), /*=*/ copy(y(es)),   // x ← old y
             y(es), /*=*/ copy(x(es)));  // y ← old x
```

The `copy()` calls are necessary. `x(es)` and `y(es)` return *views* into field storage. Without `copy()`,
the stashed right-hand sides would be views that still alias the field buffers being written, so reads and
writes would interfere. `copy()` forces an upfront value snapshot before any write occurs.

### Reductions

To reduce an expression over entities to a scalar, use the reduction helpers. Process-local (single MPI
rank) variants:

```cpp
double s  = reduce_local_sum<double>(x(es) * y(es));
double hi = reduce_local_max<double>(norm(vel(es)));
double lo = reduce_local_min<double>(norm(vel(es)));
```

MPI-global (across all ranks) variants:

```cpp
double dot       = all_reduce_sum<double>(x(es) * y(es));
double max_speed = all_reduce_max<double>(norm(velocity(es)));
```

Compared to the manual approach, no scratch field is needed to hold an intermediate result:

```cpp
// Manual: compute norm into a scratch field, then reduce the scratch field separately.
// Expression: compute and reduce in one step, no scratch storage.
double max_speed = all_reduce_max<double>(norm(velocity(es)));
```

### Random numbers

`rng` creates a per-entity counter-based random number generator (Philox by default). The seed and
counter can each be a field expression or a plain constant, with at least one being an expression:

```cpp
auto our_rng = rng(seed(es), counter(es));  // both from fields
auto our_rng = rng(42u,      counter(es));  // fixed seed, per-entity counter
auto our_rng = rng(seed(es), 0u);           // per-entity seed, fixed counter
fused_assign(a(es), /*=*/ our_rng.rand<double>(),   // first draw
             b(es), /*=*/ our_rng.rand<double>());  // second draw — a != b
```

Sequential draws from the same `rng` expression advance the generator state so successive calls return
different values, exactly as you would expect. The `rng` node is the runtime analog of compile-time reuse:
the generator is constructed once per entity within the fused tree and then `.rand<T>()` calls advance its
counter, rather than each draw re-seeding from scratch.

### Custom operations

The operations below are for users who need to call functions that the expression system does not wrap
directly. Most users will not need them—the named builtins cover the common cases.

#### `apply_expr` — custom read-only functions

Wrap any device-callable function object that takes values and returns a value, including Kokkos lambdas:

```cpp
auto func = KOKKOS_LAMBDA(const auto& x, const auto& y, const auto& bias) {
    return x + 2.0 * y + bias;
};
z(es) = 2.0 * apply_expr(func, x(es), y(es), some_scalar);
```

Scalars are promoted automatically. The result is a regular expression node that composes freely with
arithmetic and other builtins.

#### `sink_expr` — custom mutating functions

For functions that mutate one of their arguments, use `sink_expr` with explicit access-mode wrappers and
call `.driver()->run()` to evaluate:

```cpp
auto func = KOKKOS_LAMBDA(auto& y, const auto& x, const auto& scale) {
    y += scale * x;
};
auto expr = sink_expr(func, read_write(y(es)), read_only(x(es)), alpha);
expr.driver()->run(expr);
```

The three access modes control how each argument's field sync state is managed:

| **Wrapper** | **Sync behavior** |
|---|---|
| `read_only(expr)` | field is synced to device; not marked modified |
| `read_write(expr)` | field is synced to device; marked modified on device after the kernel |
| `overwrite_all(expr)` | host sync state is cleared; field is marked modified on device |

Unwrapped arguments default to `read_only`. The callable must return `void`.
