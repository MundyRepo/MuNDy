At the moment, this code is testbed for some of Mundy's new features. For example, our streamlined entity/field/part
declaration helpers. It's also a testbed for aggregates, which are ever-evolving, as we figure out the best way for
users to "aggregate" their data into logical groupings. Given that this is a pure C++ application, we will directly
touch the compile-time tagged, STK aggregates and operate on them directly.

Now, we have a design decision to make. How should the aggregates "look":
  1. core::aggregate:
    - They are a tagged collection of types identical to a compile-time extensible struct.
      stk::mesh::for_each_entity_run(
          ngp_mesh, stk::topology::ELEM_RANK, sphere_selector, KOKKOS_LAMBDA(stk::mesh::FastMeshIndex sphere_index) {
            stk::mesh::FastMeshIndex center_node_index = ngp_mesh.fast_mesh_index(ngp_mesh.nodes(sphere_index)[0]);
            auto center = agg.get<CENTER>(center_node_index);
            auto radius = agg.get<RADIUS>(sphere_index);
            center += radius[0];
          });
    - They don't offer for_each directly and don't necessarily store a selector, rank, or topology.
    - Benefits:
      1. They are extremely flexible and natural to act upon since they directly rely upon stk for each entity runs
      and Kokkos parallel for's as opposed to having users use for_each sometimes and entity_view other times.
      2. They more naturally reuse connectivity since users pass in the desired entity index when accessing node data
instead of padding in the node ordinal (requiring multiple calls to ngp_mesh.nodes(sphere_index)[node_ord]). This makes
the interface more readable, since users write agg.get<RADIUS>(node2_idx)[0] instead of entity_view.get<RADIUS>(2)[0],
which has led to confusion about which is the ordinal and which is the operator[] of the returned view!
      3. core::aggregate can be used elsewhere in Mundy and isn't tied to STK or ECS.
      4. Tags are pretty flexible
    - Flaws:
      1. This design requires more direct interaction with STK
      2. One step withdrawn from mundy::geom::primitives. These are just data containers that must be unpacked into geom
      objects.
  2. mesh::STKComponentAggregate:
    - These are a tagged collection of STKAccessors and contain a core::aggregate internally.
    - They offer for_each directly and store a selector, rank, and/or topology.
    - They can be designed a couple of different ways:
      1. agg.for_each(  // Aggregates accept entity ordinals into get
          KOKKOS_LAMBDA(const auto& entity_view) {
            auto left_end = entity_view.get<COORDS>(0);   // node ordinal 0
            auto right_end = entity_view.get<COORDS>(1);  // node ordinal 1
            auto radius = entity_view.get<RADIUS>();
          });
      2. agg.for_each(  // Aggregates use named getters allowing their EntityView to "look like" geom objects
          KOKKOS_LAMBDA(const auto& entity_view) {
            auto left_end = entity_view.get<LEFT_ENDPOINT>();
            auto right_end = entity_view.get<RIGHT_ENDPOINT>();
            auto radius = entity_view.get<RADIUS>();
          });
    - The first design is flawed in that it leads to a significant amount of confusion about ordinals vs operator[]
calls. This has led to multiple bugs in user code and will not be pursued further.
    - Benefits of the second design:
      1. We can make STKComponentAggregate::EntityView and geom::primitives share a common interface, allowing users to
      do things like
        sphere_agg.for_each(
            KOKKOS_LAMBDA(const auto& sphere_view) {
              geom::wrap_rigid_inplace(sphere_view, metric);
            });
      2. There is no need for map from aggregates to geom object.
      3. Primitives become core::aggregate specializations, meaning that they can store subsets of data. If you only act
on the sphere's radius, then you need not store its center. This is nice because it allows for more memory efficient
data access since we're not forces to access a view of data we don't need.
    - Flaws:
      1. More moving parts. Users must understand STKComponentAggregate, its EntityView, and geom::primitives.
      2. More tightly coupled to STK and ECS.
      3. BIGGEST FLAW: Tags map to access. What if I have 8 nodes. I'm not writing FORCE_0, FORCE_1, ..., FORCE_7 tags.
      I'm writing FORCE and passing in the node ordinal.

We're back to taking in the ordinal.


In either case, it's not explicitly clear in the API which tags are "used" by each method, meaning that flow is lost.
This is important since some methods write to FORCE and others require a FORCE_EXT and a CONTACT_FORCE. Explicitness is
important in APIs, but has the cost of introducing bloat.

We could be a bit more explicit by introducing recipes and augments. So like Dynamic Hookean Springs.

We still end up in the issue that we'll have a method that will need to act on all spheres. For this method to work,
it will need a single aggregate that has the CENTER and RADIUS tag. It's up to the accessors in that aggregate to take
the given entity and turn it into the correct data, be it shared, field-valued, or part-valued. As such, having a single
sphere aggregate isn't limiting. I would love to use a setter here, but that would require a class with templated type
not resolved until set, which is impossible. That's part of why Raggs are nice since they remove this restriction.

If we only allow field-valued accessors with aggregate slicing, then we can have setters without templates. This would allow
users to use setters

Mundy uses tags in the same way that most ECPs use component structs to identify types. We do it this way
to abstract away the underlying data storage mechanism, be it shared values, fields, or part-based fields.
All that matters is that a given tag map to the desired view type. This is why we have to be given aggs with
compile-time type since their explicit types are required to generate the correct accessors. If we only had
a single accessor type, then 

Honestly, having aggs be compatible with primitives has a steep cost.
The same is true for having any type of accessor. 

If we only had field accessors, then every system could act on a purely runtime aggregate that it 
packs into an agg. This gives us a runtime entity component system. ECP does NOT offer the concept of 
shared values and we partly see why. Shared values require a stronger coupling between data and type
that isn't present with fields.

But sometimes I want it to be shared or I want it to be part mapped. This is why I lean towards Raggs
that users can specify valid input types for each tag, the total combinations of which should not exceed
a reasonable limit.

Look, we need a design that can be simple now and can grow in the future. If we stick with runtime aggs
that use strings to lookup accessors, then we can grow into more complex systems later. We can start 
by only supporting stk fields and not shared values. Then, later, use visitation to unpack into concrete
accessors for more complex systems.

Think like a runtime entity component system with components (variant accessors) that are accessed via 
string tag

 by runtime than accessed via their
stored type. Do we even want 







The interface for KMC needs generalized. 

In each state, we must decide which, if either, of the heads changes states. 

For unbinding, we need only the information about the spring and its heads.
For binding, we need the spring information and all of its potential bind sights.

I like the idea of a to_spring(entity_index) -> mech::Spring2View. This way, instead of 
taking in the necessary data to make a spring.

This is exactly like the current Aggregate concept with its get_view(Entity) design except 
with the creation of a true mech object. The problem we had with this is that sometimes, you really
do want a temporary object and not a view. Right now, Aggregates always return views. You can't have
a function that acts on an aggregate that's an owning spring2. We need this for our energy functionals
since we want to try out different configurations. 

Basically, instead of making a bunch of named objects like Springs, DynamicSprings, etc, we want a general
aggregate type. But, with python, we'll need to make that concrete anyway for users to use them within 
expressions! Ok, you still use aggregates but you allow for named aggregates within python. Ohhhh, one
of the niceties of aggregates compared to concrete types is that you only fetch local views when get is 
called. Otherwise, no view is creates. So just because you are acting on a spring doesn't mean you need 
to fetch spring constant, if all you touch is the node coordinates.

Every primitive should be a specialized aggregate. Does this work? Well, we need to be able to create 
local owning aggregates within kernels. That aren't necessarily of the full named type. I don't want 
to act on a spring, I want to act on the nodes of a spring and only some of their data. My copy should
reflect that.

One of the flaws in this design, exposed by the existing aggregates was that accessing the fields of 
lower ranked entities is inefficient if their data is accessed multiple times since we repeatedly call 
get_connected_entities. This cost is traded for a uniform interface that can act on aggregate's whose
get<Tag>() can either access an accessor using said entity or access an internally stored shared value.

We're starting to overload the term aggregate here. We have the idea of "a collection of tagged types"
much like a struct where the names of the types are the tags. We also have the idea of "a collection
of tagged accessors, which perform get internally when performed". This is what we call an EntityView,
which follows the same interface as an aggregate plus some additional helper functions for fetching the
involved entity.

According to this design
- core::Aggregate: A tagged bag of types. get<Tag>(agg) returns a reference to an internally stored 
  instance of the object associated with the tag.
- core::WrappedAggregate: A relayed tagged bag of types. get<Tag>(wrapped_agg) returns 
  wrapper_(get<Tag>(agg_)).
- mesh::ComponentAggregate: A tagged bag of components (entity accessors).
- mesh::ComponentAggregate.get_view(entity) -> entity_view.
- mesh::ComponentAggregate::EntityView: A wrapped aggregate with a wrapper that has
  decltype(auto) operator()(const auto& accessor) { return accessor(entity_); }
- EntityView.copy<TAG1, TAG2>() -> agg

So each primitive should inherit from its respective aggregate and would no longer offer getters that aren't
get<TAG>(prim). The IsValidPrimitive would simply check that the given tags exists and that they return the
correct types. In this way, geom::Primitive would be the "owning default" and any aggregate with the same tags
and same abstract return type would be a valid version of this primitive. This naturally makes EntityViews
compatible with all of mundy::geom. If we make get_view compatable with entity expressions, then this would
allow all of mundy::geom to be used with our expression system.

In this regard, we don't need a to_spring. The aggregate's entity_view *is a* spring. To me, this does mean
that we want to continue using ComponentAggregates, for_each, and their EntityView as opposed to raw accessors.
We will not attempt to support runtime aggregates. We will, however, attempt to create a python interface with
auto-compiled tag-dispatched methods. I don't think we should use operator() for the kernels tho, passing
*this is confusing to Chris. 
