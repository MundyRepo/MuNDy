# MundyMesh

See the {ref}`MundyMesh API directory <dir_mundy_mesh>`.

MundyMesh is the part of Mundy that manages mesh data, mesh-side access patterns, and mesh-side algorithms.

If MundyMath provides small math objects and MundyGeom provides small geometry objects, MundyMesh provides the mesh
storage and workflows that let those objects live on entities and be used in host/device algorithms.

Under the hood, MundyMesh extends Trilinos/STK. It keeps STK's core model—{ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>`, {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>`, parts, fields,
selectors, and NGP access—but adds convenience layers and new capabilities needed by Mundy's multibody workflows.

The package is organized in layers:
1. Build or extend an STK mesh with Mundy {ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>`, {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>`, and {ref}`MeshBuilder <exhale_class_classmundy_1_1mesh_1_1MeshBuilder>`.
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
| `Part` | STK's native grouping mechanism for entities. Parts drive selectors, topology declarations, and IO membership. | {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` extends parts into a semantic hierarchy like “all rods” or “boundary nodes”. |
| `Field` | Per-entity data attached to a rank and usually restricted by part membership. | {ref}`Component <exhale_class_classmundy_1_1mesh_1_1FieldComponent>` wraps a field as a typed, tagged accessor. |
| `Selector` | A query over part membership. | Mundy adds {ref}`string_to_selector(...) <namespace_mundy__mesh>` and class-aware workflows on top. |
| {ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>` | The schema of the mesh: what parts, fields, and ranks exist. | Mundy extends it with attributes and helper-driven declaration workflows. |
| {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>` | The live mesh state: which entities currently exist and how they are related. | Mundy extends it with links, staged modification helpers, and richer NGP workflows. |

Two Mundy terms appear frequently because they sit directly beside STK concepts rather than replacing them:

| **Mundy term** | **Relationship to STK** |
|----------------|--------------------------|
| {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` | A semantic layer built on top of coordinated STK parts so that hierarchy, subset logic, and IO behave like a real class system. |
| {ref}`Component <exhale_class_classmundy_1_1mesh_1_1FieldComponent>` / {ref}`Aggregate <exhale_class_classmundy_1_1mesh_1_1Aggregate>` | A usability layer built on top of STK fields so that field data can be accessed as typed, tagged objects and passed around as one aggregate. |

Three pairings summarize the relationship:

 - `Part` -> native STK grouping, {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` -> semantic Mundy grouping.
 - `Field` -> native STK storage, {ref}`Component <exhale_class_classmundy_1_1mesh_1_1FieldComponent>` -> typed Mundy accessor.
 - STK relations/connectivity -> native mesh relations, `Link` -> dynamic non-topological relation entity.

Some of these layers exist to reduce STK boilerplate. Others add capabilities that STK does not provide directly,
including dynamic GPU-compatible links and ticketed device-side modification requests.

## Architecture at a glance

| **Layer** | **Main types / functions** | **Use when you need** |
|-----------|-----------------------------|------------------------|
| Mesh construction | {ref}`MeshBuilder <exhale_class_classmundy_1_1mesh_1_1MeshBuilder>`, {ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>`, {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>` | A Mundy-aware STK mesh object. |
| Declaration helpers | {ref}`FieldDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1FieldDeclarationHelper>`, {ref}`PartDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1PartDeclarationHelper>`, {ref}`ClassDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1ClassDeclarationHelper>`, {ref}`DeclareEntitiesHelper <exhale_class_classmundy_1_1mesh_1_1DeclareEntitiesHelper>` | Less boilerplate when setting up a mesh. |
| Semantic structure | {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>`, `declare_class(...)` | A flattened class hierarchy on top of parts that also behaves correctly with STK IO. |
| Field access | `scalar_field_data`, `vector3_field_data`, `quaternion_field_data`, ... | Typed math/geom views into raw field storage. |
| Components / aggregates | {ref}`Component <exhale_class_classmundy_1_1mesh_1_1FieldComponent>`, {ref}`FieldComponent <exhale_class_classmundy_1_1mesh_1_1FieldComponent>`, {ref}`SharedComponent <exhale_class_classmundy_1_1mesh_1_1SharedComponent>`, {ref}`Aggregate <exhale_class_classmundy_1_1mesh_1_1Aggregate>`, {ref}`NgpAggregate <exhale_class_classmundy_1_1mesh_1_1NgpAggregate>` | Storage-independent data access and logical grouping of many accessors. |
| Iteration helpers | {ref}`for_each_entity_run <namespace_mundy__mesh>`, {ref}`for_each_link_run <namespace_mundy__mesh>` | Uniform host/device loops over entities or links. |
| String parsing | {ref}`string_to_selector <namespace_mundy__mesh>`, {ref}`string_to_rank <namespace_mundy__mesh>`, {ref}`string_to_topology <namespace_mundy__mesh>` | Config-driven mesh queries and declarations. |
| Dynamic links | {ref}`LinkMetaData <exhale_class_classmundy_1_1mesh_1_1LinkMetaData>`, {ref}`LinkData <exhale_class_classmundy_1_1mesh_1_1LinkData>`, {ref}`NgpLinkData <exhale_class_classmundy_1_1mesh_1_1NgpLinkDataT>`, {ref}`get_updated_ngp_link_data(...) <namespace_mundy__mesh>` | A new dynamic GPU-compatible connectivity model distinct from STK relations. |
| Mesh modification staging | {ref}`NgpModRequests <exhale_class_classmundy_1_1mesh_1_1NgpModRequestsT>` | Ticketed device-side requests for later host-side mesh modification. |
| Usability layer | {ref}`NgpFieldBLAS <file_mundy_mesh_src_mundy_mesh_NgpFieldBLAS.hpp>`, {ref}`NgpAccessorExpr <file_mundy_mesh_src_mundy_mesh_NgpAccessorExpr.hpp>` | Higher-level field math and expression-style device programming. |

## Mesh construction
At the lowest level, MundyMesh extends STK rather than replacing it.

### {ref}`MeshBuilder <exhale_class_classmundy_1_1mesh_1_1MeshBuilder>`
{ref}`MeshBuilder <exhale_class_classmundy_1_1mesh_1_1MeshBuilder>` mirrors STK's mesh builder, but produces Mundy {ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>` and {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>` objects.

```cpp
stk::ParallelMachine comm = MPI_COMM_WORLD;

mundy::mesh::MeshBuilder builder(comm);
builder.set_spatial_dimension(3)
       .set_auto_aura_option(stk::mesh::BulkData::AUTO_AURA)
       .set_upward_connectivity_flag(true);

std::unique_ptr<mundy::mesh::BulkData> bulk = builder.create_bulk_data();
mundy::mesh::MetaData& meta = static_cast<mundy::mesh::MetaData&>(bulk->mesh_meta_data());
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
| `create_meta_data()` | Create a standalone Mundy {ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>`. |
| `create_bulk_data()` | Create a {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>` and its metadata. |
| `create_bulk_data(meta_ptr)` | Create {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>` around existing Mundy metadata. |

### {ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>`
{ref}`MetaData <exhale_class_classmundy_1_1mesh_1_1MetaData>` currently extends STK mainly through attribute storage.

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

### {ref}`FieldDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1FieldDeclarationHelper>`
Use {ref}`FieldDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1FieldDeclarationHelper>` when you want to declare a field through a fluent interface.

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

The required inputs are:

| **Required before `declare()`** | **Optional** |
|---------------------------------|--------------|
| `type<T>()` | `role(...)` |
| `rank(...)` | `output_type(...)` |
| `name(...)` | |

The most useful setters are:

| **Setter** | **Meaning** |
|------------|-------------|
| `type<T>()` | Choose the scalar storage type. |
| `rank(rank)` | Choose the entity rank for the field. |
| `name("...")` | Choose the field name. |
| `role(role)` | Set STK IO role such as `MESH` or `TRANSIENT`. |
| `output_type(type)` | Control component labeling in IO. |
| `tag<Tag>()` | Continue into tagged field/component declaration. |
| `access<AccessLike>()` | Continue into field-backed component declaration. |

### {ref}`PartDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1PartDeclarationHelper>`
Use {ref}`PartDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1PartDeclarationHelper>` to declare named, ranked, or topological parts and optionally attach restrictions.

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

### {ref}`ClassDeclarationHelper <exhale_class_classmundy_1_1mesh_1_1ClassDeclarationHelper>` and `declare_class(...)`
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

### {ref}`DeclareEntitiesHelper <exhale_class_classmundy_1_1mesh_1_1DeclareEntitiesHelper>`
{ref}`DeclareEntitiesHelper <exhale_class_classmundy_1_1mesh_1_1DeclareEntitiesHelper>` is a host-side builder for declaring nodes, elements, classes, field data, and link
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
STK IO works with a hierarchy of disjoint parts. Mundy's {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` layer builds on top of that representation so code can
be written more like a flattened class hierarchy while still satisfying STK's IO rules.

### Why {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` exists
Many multibody models need names such as “all rods”, “boundary nodes”, “contact links”, or “bead centers” rather
than raw part membership. {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` wraps the part hierarchy needed to make those concepts consistent with STK's IO and
subset rules.

A {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` is the semantic object. Its underlying STK parts are the implementation detail that keeps selectors, subset
relations, inherited data behavior, and IO consistent.

This matters because parts already contain much of the raw machinery needed for class-like behavior, but making that
behave like a usable class hierarchy in the presence of STK IO takes additional logic. {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` automates that logic so
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

### Core {ref}`Class <exhale_class_classmundy_1_1mesh_1_1Class>` queries

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

### {ref}`string_to_selector <namespace_mundy__mesh>`
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

### {ref}`string_to_rank <namespace_mundy__mesh>` and {ref}`string_to_topology <namespace_mundy__mesh>`

```cpp
auto rank = mundy::mesh::string_to_rank("NODE_RANK");
auto topo = mundy::mesh::string_to_topology("HEX_8");
```

Use these when mesh configuration is text-driven rather than hard-coded.

## Typed views into field data
Raw `stk::mesh::field_data(...)` returns raw pointers. {ref}`FieldViews.hpp <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>` maps that storage into Mundy math and geometry
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
| {ref}`scalar_field_data(...) <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>` | `ScalarWrapper` / scalar view |
| {ref}`vector_field_data <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>`\<N\>(...) | `Vector<N>`-like view |
| {ref}`vector3_field_data(...) <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>` | {ref}`Vector3 <exhale_class_classmundy_1_1AVector>` |
| {ref}`matrix_field_data <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>`\<N, M\>(...) | `Matrix<N, M>` |
| {ref}`matrix3_field_data(...) <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>` | {ref}`Matrix3 <exhale_class_classmundy_1_1AMatrix>` |
| {ref}`quaternion_field_data(...) <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>` | {ref}`Quaternion <exhale_class_classmundy_1_1AQuaternion>` |
| {ref}`aabb_field_data(...) <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>` | {ref}`AABB <exhale_class_classmundy_1_1AABB>` |

Use these helpers when a field is logically a math object, not just a flat array.

## Components and aggregates

### Components
Components provide algorithmic separation of concerns: most algorithms that act on entities care only about accessing a
given data object (a center, a radius, a velocity, an orientation) for a given entity, not about *how* that data is
stored. They rarely care whether a matrix is stored row-major vs. column-major or whether a radius is shared by all
entities vs. stored as a field value on each entity.

This is achieved through a simple set of accessor classes called {ref}`Component <exhale_class_classmundy_1_1mesh_1_1FieldComponent>`s, each of which offers an access operator
`operator()(entity)` that returns a view into the data for that entity of the given semantic type. For example,
```cpp
auto center_accessor = Vector3FieldComponent<double>(center_field);
auto radius_accessor = SharedScalarComponent<double>(radius);

auto center = center_accessor(node1);  // returns a Vector3 view into the center field data for node1
auto collision_radius = radius_accessor(elem1);  // returns a scalar view into the shared radius value for elem1
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
| Scalar | {ref}`ScalarFieldComponent <exhale_class_classmundy_1_1mesh_1_1ScalarFieldComponent>`\<T\> | {ref}`SharedScalarComponent <exhale_class_classmundy_1_1mesh_1_1SharedScalarComponent>`\<T\> | ScalarWrapper<T> |
| Vector of length `N` | {ref}`VectorNFieldComponent <exhale_class_classmundy_1_1mesh_1_1VectorFieldComponent>`\<T, N\> | {ref}`SharedVectorComponent <exhale_class_classmundy_1_1mesh_1_1SharedVectorComponent>`\<T, N\> | Vector<T, N> |
| Matrix3 | {ref}`Matrix3FieldComponent <exhale_class_classmundy_1_1mesh_1_1Matrix3FieldComponent>`\<T\> | {ref}`SharedMatrix3Component <exhale_class_classmundy_1_1mesh_1_1SharedMatrix3Component>`\<T\> | Matrix3<T> |
| Quaternion | {ref}`QuaternionFieldComponent <exhale_class_classmundy_1_1mesh_1_1QuaternionFieldComponent>`\<T\> | {ref}`SharedQuaternionComponent <exhale_class_classmundy_1_1mesh_1_1SharedQuaternionComponent>`\<T\> | Quaternion<T> |
| AABB | {ref}`AABBFieldComponent <exhale_class_classmundy_1_1mesh_1_1AABBFieldComponent>`\<T\> | {ref}`SharedAABBComponent <exhale_class_classmundy_1_1mesh_1_1SharedAABBComponent>`\<T\> | AABB<T> |

At present, these patterns are supported for field-backed and shared components. We intend to
offer the same interface through `PartMappedComponent`s in the future so that per-part access policies can be exposed
through the same semantic accessor model.

Just like STK's `NgpField`s, each component is backed by an NGP/Kokkos-compatible counterpart that can be obtained
through {ref}`get_updated_ngp_component(...) <namespace_mundy__mesh>`. For these components, the access operator is instead `operator()(FastMeshIndex)` and 
returns a Kokkos-compatible, performance-portable view into the data for the given entity.

As with `NgpField`, these ngp components have a lifetime that is valid until the next mesh modification cycle, so the
best pattern is to pass around raw components, fetch their updated ngp components immediately before entering a
Kokkos kernel, and use the ngp component inside the kernel. If no modification has occurred, {ref}`get_updated_ngp_component(...) <namespace_mundy__mesh>`
will simply return a reference to the existing ngp component, making it inexpensive to call repeatedly.
```cpp
auto ngp_center_accessor = mundy::mesh::get_updated_ngp_component(center_accessor);
auto ngp_velocity_accessor = mundy::mesh::get_updated_ngp_component(velocity_accessor);
stk::mesh::for_each_entity_run(ngp_mesh, stk::topology::NODE_RANK, selector,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& i) {
            auto center = ngp_center_accessor(i);
            auto velocity = ngp_velocity_accessor(i);
            center += dt * velocity;
        });
```

### {ref}`Aggregate <exhale_class_classmundy_1_1mesh_1_1Aggregate>`
An {ref}`Aggregate <exhale_class_classmundy_1_1mesh_1_1Aggregate>` is a logical collection of tagged components, along with the {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>` and `Selector` describing the
subset of the mesh on which they are intended to act.

In practice, an aggregate answers two questions:
1. Which entities are we iterating over?
2. Which semantic components are available on those entities?

This makes aggregates a natural way to package the data requirements of a system. If a collision algorithm needs a
center and a collision radius, we do not define a hard-coded {ref}`Sphere <exhale_class_classmundy_1_1Sphere>` class. We instead assemble an aggregate that
contains a `CENTER` component and a `COLLISION_RADIUS` component and pass that aggregate into the algorithm.

```cpp
using namespace mundy::mesh;

auto center_accessor = Vector3FieldComponent(center_field);
auto radius_accessor = SharedScalarComponent(radius);

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

Like `NgpField`, {ref}`Aggregate <exhale_class_classmundy_1_1mesh_1_1Aggregate>` is the synchronization-owning object. For Kokkos-compatible, performance-portable code,
fetch its NGP counterpart with {ref}`get_updated_ngp_aggregate(...) <namespace_mundy__mesh>` immediately before entering a kernel.
{ref}`NgpAggregate <exhale_class_classmundy_1_1mesh_1_1NgpAggregate>` provides the same tagged access pattern using `FastMeshIndex`, and if no modification has occurred,
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
 - support both **link -> linked entity** access and **linked entity -> incident links** access.

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

### {ref}`LinkMetaData <exhale_class_classmundy_1_1mesh_1_1LinkMetaData>`
{ref}`LinkMetaData <exhale_class_classmundy_1_1mesh_1_1LinkMetaData>` declares the fields and class structure needed for a family of links. In most user code, this is the
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

### {ref}`LinkData <exhale_class_classmundy_1_1mesh_1_1LinkData>`
{ref}`LinkData <exhale_class_classmundy_1_1mesh_1_1LinkData>` is the runtime object that manages the actual link relations on a {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>`.

```cpp
LinkData& link_data = declare_link_data(*bulk, contact_links);
```

The most important distinction is:

 - use **ordinary STK mesh modification** to create or destroy the **link entities themselves**,
 - use {ref}`LinkData <exhale_class_classmundy_1_1mesh_1_1LinkData>` to create or destroy the **relations stored on those links**.

### Minimal host workflow
The explicit host-side workflow usually looks like this:
1. Declare {ref}`LinkMetaData <exhale_class_classmundy_1_1mesh_1_1LinkMetaData>` before commit.
2. Construct {ref}`LinkData <exhale_class_classmundy_1_1mesh_1_1LinkData>` for a live {ref}`BulkData <exhale_class_classmundy_1_1mesh_1_1BulkData>`.
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

Like other mesh-side access objects, {ref}`LinkData <exhale_class_classmundy_1_1mesh_1_1LinkData>` has an NGP counterpart for Kokkos-compatible,
performance-portable work. Fetch it with {ref}`get_updated_ngp_link_data(...) <namespace_mundy__mesh>` immediately before a kernel when you need
link access, CSR/COO checks, or performance-portable traversal. {ref}`NgpLinkData <exhale_class_classmundy_1_1mesh_1_1NgpLinkDataT>` mirrors the same interface in NGP
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
{ref}`NgpModRequests <exhale_class_classmundy_1_1mesh_1_1NgpModRequestsT>` is Mundy's ticket-based framework for device-originated mesh changes.

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

{ref}`NgpModRequests <exhale_class_classmundy_1_1mesh_1_1NgpModRequestsT>` lets kernels describe requested mesh changes on device and then has the host realize them safely in a
later modification phase.

## Field BLAS
{ref}`NgpFieldBLAS.hpp <file_mundy_mesh_src_mundy_mesh_NgpFieldBLAS.hpp>` provides BLAS-like operations over `stk::mesh::FieldBase` objects with unified host/device style.

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
{ref}`NgpAccessorExpr <file_mundy_mesh_src_mundy_mesh_NgpAccessorExpr.hpp>` is Mundy's highest-level device expression layer.

### Goal
The point of accessor expressions is to reduce device-loop code down to a higher-level, math-like syntax.

Instead of writing a manual loop that says

 - iterate these entities,
 - fetch these accessors,
 - compute this intermediate quantity,
 - write these results,
 - remember which fields must sync,
 - remember which fields must be marked modified,

you write something much closer to PyTorch, TensorFlow, or MATLAB-style algebra over entity-backed data.

That is why this layer is best viewed as a small domain-specific language for device-side mesh loops. It describes what
should be computed over a set of entities and then compiles that description into optimized loop bodies.

This is more than syntax sugar. The expression tree carries enough structure for Mundy to:

 - delay evaluation until assignment or reduction,
 - automatically synchronize fields read by the expression,
 - automatically mark written fields as modified,
 - fuse multiple assignments into one traversal,
 - reuse repeated subexpressions,
 - reuse repeated subexpressions that appear on different branches of the expression tree,
 - generate compile-time-optimized loop bodies rather than treating the expression as an interpreted runtime object.

So the real mental model is not “fancy accessor syntax”. It is “write the body of a device loop in compact algebraic
form, then let the expression system build and optimize that loop for you.”

At user level, that means delayed evaluation, automatic sync/modify handling, fused multi-assignment, optional reuse,
and reductions directly over expressions.

### The four pieces of an expression loop
Most expression code has four conceptual pieces:

1. define the **iteration domain**,
2. lift accessors into **entity-indexed expressions**,
3. combine them into **algebraic expressions**,
4. evaluate them with an **assignment or reduction**.

Seen that way, an expression block is just a compact spelling of a mesh loop.

### Step 1: define the iteration domain
The first object is usually created with {ref}`make_entity_expr(...) <namespace_mundy__mesh>`.

```cpp
using namespace mundy::mesh;

auto rods = make_entity_expr(*bulk, rod_selector, stk::topology::ELEM_RANK);
```

This says, “the logical loop domain is all selected rod entities of this rank.”

You can think of `rods` as the loop variable family, not as data itself.

### Step 2: lift accessors into expression form
Accessor expressions operate on tagged components or other accessors. The most explicit pattern is to fetch the
component first, then apply it to the entity expression.

```cpp
auto center = rod_data.get_component<CENTER>();
auto velocity = rod_data.get_component<VELOCITY>();

auto center_e = center(rods);
auto velocity_e = velocity(rods);
```

At this point, `center_e` and `velocity_e` are not values yet. They are delayed expressions that mean “center of the
current rod in the loop” and “velocity of the current rod in the loop”.

### Step 3: build algebraic expressions
Once lifted, accessors can participate in ordinary arithmetic and in many MundyMath-style free functions supported by
their underlying value type.

```cpp
auto predicted_center = center_e + dt * velocity_e;
```

This is the point of the DSL: the loop body starts to look like math instead of explicit gather/compute/scatter code.

For connected-entity workflows, derive a new entity expression first and then apply the accessor to that expression.

```cpp
auto rod_nodes = rods.get_connectivity(stk::topology::NODE_RANK);

auto node_center_e = center(rod_nodes[0]);
auto node_velocity_e = velocity(rod_nodes[0]);
```

That pattern lets you express multi-entity loop logic while keeping the code algebraic.

### Step 4: trigger evaluation
Nothing runs until an evaluation call happens. In the common case, evaluation is triggered simply by assigning to an
expression l-value. As soon as you assign to it with `=`, `+=`, `-=`, `*=`, or `/=`, evaluation occurs over the loop
domain.

```cpp
center(rods) += dt * velocity(rods);
```

Before the kernel runs, the expression system synchronizes fields that are read and marks written fields as modified on
device. That is the main reason to use this layer instead of manually writing every `sync_to_device()` and
`modify_on_device()` call.

So a helpful rule is:

 - building expressions is like building a computation graph,
 - assignment to an expression l-value and reduction calls are the points where that graph is realized as an optimized
   loop.

### A full update example
Written out explicitly, a typical update looks like this:

```cpp
using namespace mundy::mesh;

auto rods = make_entity_expr(*bulk, rod_selector, stk::topology::ELEM_RANK);

auto center = rod_data.get_component<CENTER>();
auto velocity = rod_data.get_component<VELOCITY>();

center(rods) += dt * velocity(rods);
```

### {ref}`fused_assign(...) <namespace_mundy__mesh>` for simultaneous multi-assignment
Use {ref}`fused_assign(...) <namespace_mundy__mesh>` when you want multiple different left-hand sides to be updated in the same loop evaluation.
That is its real role: not ordinary single-target assignment, but simultaneous multi-target assignment.

```cpp
auto x = data.get_component<X>();
auto y = data.get_component<Y>();
auto z = data.get_component<Z>();

auto es = make_entity_expr(*bulk, selector, stk::topology::NODE_RANK);

fused_assign(y(es), /*=*/2.0 * x(es) + y(es),
             z(es), /*=*/x(es) * y(es));
```

This performs both updates in one traversal and one kernel launch.

This is also where reuse matters most. If several outputs depend on the same intermediate expression, the expression
system can avoid recomputing that intermediate by reusing it within the fused evaluation.

### Reductions over expressions
Reductions are also explicit evaluation points.

```cpp
auto es = make_entity_expr(*bulk, selector, stk::topology::NODE_RANK);

auto x = make_tagged_component<XTag>(ScalarFieldComponent(*field_x));
auto y = make_tagged_component<YTag>(ScalarFieldComponent(*field_y));

double dot_xy = all_reduce_sum<double>(x(es) * y(es));
double local_max_x = reduce_local_max<double>(x(es));
```

Use the reduction family that matches the scope you want:

| **Function** | **Meaning** |
|--------------|-------------|
| {ref}`reduce_local_sum <namespace_mundy__mesh>`\<Scalar\>(expr) | Sum on this MPI rank only. |
| {ref}`reduce_local_max <namespace_mundy__mesh>`\<Scalar\>(expr) / {ref}`reduce_local_min <namespace_mundy__mesh>`\<Scalar\>(expr) | Rank-local extrema. |
| {ref}`all_reduce_sum <namespace_mundy__mesh>`\<Scalar\>(expr) | Global MPI sum. |
| {ref}`all_reduce_max <namespace_mundy__mesh>`\<Scalar\>(expr) / {ref}`all_reduce_min <namespace_mundy__mesh>`\<Scalar\>(expr) | Global MPI extrema. |

### Important evaluation rules

| **What you write** | **What happens** |
|--------------------|------------------|
| `lhs = rhs` or `lhs += rhs` | If `lhs` is an expression l-value such as `center(rods)`, evaluate over the entity set and write back to `lhs`. |
| `fused_assign(lhs1, rhs1, lhs2, rhs2, ...)` | Evaluate several left/right pairs in the same fused loop. Use this when you want simultaneous multi-target assignment. |
| {ref}`all_reduce_* <namespace_mundy__mesh>`\<Scalar\>(expr) / {ref}`reduce_local_* <namespace_mundy__mesh>`\<Scalar\>(expr) | Evaluate the expression and reduce the resulting values. |
| `auto expr = ...;` | Build an expression tree only; no kernel runs yet. |

### Two common gotchas

1. **Swapping view-backed accessors requires {ref}`copy(...) <namespace_mundy>`.**

```cpp
auto es = make_entity_expr(*bulk, selector, stk::topology::NODE_RANK);

fused_assign(x(es), /*=*/copy(y(es)),
             y(es), /*=*/copy(x(es)));
```

Without {ref}`copy(...) <namespace_mundy>`, the right-hand side may just stash another view rather than the value you meant to preserve.

2. **Use `reuse(expr)` only for repeated pure subexpressions inside one fused evaluation.**

If the same expensive expression is used multiple times in one fused kernel, {ref}`reuse(...) <namespace_mundy__mesh>` makes that intent explicit.
This includes cases where the repeated expression appears in different branches of the overall expression tree rather
than in one obvious linear formula. If you are only using the subexpression once, it is usually unnecessary.

Use this abstraction when you want expression-style math without manually managing every `sync_to_device()` and
`modify_on_device()` call.
