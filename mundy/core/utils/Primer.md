# MundyUtils
This subpackage contains small Kokkos-friendly utilities that smooth over standard-library tools that are missing,
awkward, or unavailable in device code. The core functionality includes
 - **Device-friendly helpers**
 - **Device-friendly containers**
 - **Kokkos-like views**

## **Device-friendly helpers**
These helpers are mostly about writing portable host/device code without giving up readable errors, compile-time strings,
or small metaprogramming conveniences.

### `throw_assert` and `StringSink`
`MUNDY_THROW_REQUIRE` and `MUNDY_THROW_ASSERT` are the main tools for writing portable assertions. They are designed to be used in both host and device code and provide rich error information while still being safe to use in device code.

Their signature is as follows:
```cpp
MUNDY_THROW_REQUIRE(bool assertion, ExceptionType, message);  // Always checks the assertion and host throws on failure/device aborts.
MUNDY_THROW_ASSERT(bool assertion, ExceptionType, message);   // The assertion is only evaluated/checked in debug builds.
```

Here, assertion may be either a bool or an inline expression that produces a bool. The `message` may be a string literal, `StringLiteral`, `std::string`, or `StringSink` expression. However, only compile-time-printable messages are guaranteed to appear in the abort output on the device. If the message is an `std::string` or a `StringSink` expression involving runtime values, the device will still compile and run, but the message won't be included in the error output.

StringSinks are particularly useful for constructing both compile-time and runtime messages via streaming (`<<`) syntax:
```cpp
MUNDY_THROW_REQUIRE(i < n, std::out_of_range,
                    sink() << "i = " << i << " but n = " << n);
```

### `StringLiteral`
`StringLiteral` wraps a string literal so it can be used as a compile-time value. This is most useful when a string is part
of a type or policy.

```cpp
template <StringLiteral Name>
struct FieldTag {
  inline static constexpr auto name = Name;
};

using Position = FieldTag<make_string_literal("position")>;
static_assert(Position::name == make_string_literal("position"));
```

String literals can also be concatenated and streamed at compile time.
```cpp
constexpr auto full_name1 = make_string_literal("particle.") + Position::name;
constexpr auto full_name2 = sink() << "particle." << Position::name;
static_assert(full_name1 == make_string_literal("particle.position"));
static_assert(full_name2 == make_string_literal("particle.position"));
```

### `type_traits`
Mundy offers a few tiny traits for working with type packs. They are intentionally simple and mostly support the containers below.

| **Trait** | **Description** |
|-----------|-----------------|
| `contains_type_v<T, Ts...>` | True if `T` appears in `Ts...`. |
| `count_type_v<T, Ts...>` | Number of times `T` appears in `Ts...`. |
| `index_finder_v<T, Ts...>` | Index of a unique `T` in `Ts...`. |
| `type_at_index_t<I, Ts...>` | The `I`'th type in `Ts...`. |

### `rng`
`make_philox` constructs an `openrand::Philox` generator from a `size_t` seed and counter value.

The resulting generator can be used to produce random values in host or device code:
```cpp
openrand::Philox rng = make_philox(seed, counter);
auto u = rng.rand<double>();
```

This helper exists to provide a single, consistent way to construct Philox generators across the codebase while accounting for the differences between Philox's interface and Mundy's needs. Notably, Philox's counter is `uint32_t`-based and Mundy's counter is `size_t`-based. We static_cast between the two and throw in debug mode if the given counter exceeds the maximum representable value for a `uint32_t`.

### Minor helpers

| **Helper** | **Use** |
|------------|---------|
| `MUNDY_ATTRIBUTE_UNUSED` | Mark intentionally unused declarations. |
| `MUNDY_SUPPRESS_*` macros | Locally silence compiler diagnostics around unavoidable warnings. |
| `do_not_optimize_away(x)` | Keep benchmark values live so the compiler cannot erase the work. |

## **Device-friendly containers**
These containers mirror familiar standard-library ideas while remaining friendly to Kokkos kernels and constexpr use.

### `tuple`
`mundy::tuple<Ts...>` stores a fixed set of heterogeneous values. It offers the tuple operations we use in device code:
`get<I>`, unique-type `get<T>`, `tuple_size_v`, `tuple_element_t`, `make_tuple`, and binary
`tuple_cat`.

It is not `std::tuple` with a different namespace. `std::tuple` also supports `tie`, `forward_as_tuple`, `apply`,
`make_from_tuple`, variadic `tuple_cat` over tuple-like inputs, pair/tuple-like conversions, allocator-aware
construction, comparisons, swap, formatting, and rvalue/ref-qualified `get`. `mundy::tuple` leaves those out.

The practical differences are:

| **`std::tuple`** | **`mundy::tuple`** |
|------------------|--------------------|
| Uses the standard `std::tuple_size` / `std::tuple_element` protocol. | Uses Mundy's own `tuple_size_v` / `tuple_element_t`; standard tuple-like algorithms do not automatically see it. |
| Has forwarding and reference helpers such as `tie` and `forward_as_tuple`. | Stores values directly; `make_tuple` copies its inputs and does not build forwarding-reference tuples. |
| `get` preserves value category with `&`, `const&`, `&&`, and `const&&` overloads. | `get` returns mutable or const lvalue references. |
| `tuple_cat` is variadic and works with tuple-like inputs. | `tuple_cat` joins two `mundy::tuple`s and copies the elements into the result. |

```cpp
auto t = make_tuple(1, 2.5, "test");
get<0>(t);        // Returns 1
get<double>(t);   // Returns 2.5

auto both = tuple_cat(make_tuple(1, 2), make_tuple(3.0));
```

### `variant`
`mundy::variant<Ts...>` is a Kokkos-compatible `std::variant`-like class. It has one active alternative at a time and provides type/index access, assignment to a new active type, `holds_alternative`, `variant_size_v`, `variant_alternative_t`, and `visit`.

Compared with `std::variant`, this version is more restrictive: alternatives must be default constructible and copy assignable, wrong-type access is debug-checked, and visitors must return the same type for every alternative.

```cpp
variant<int, double> v(2.5);
holds_alternative<double>(v);  // true
get<double>(v);                // Returns 2.5

v = 4;
auto shifted = visit([](const auto& value) { return static_cast<double>(value) + 1.0; }, v);
```

### `reference_wrapper`
`reference_wrapper` is a Kokkos-compatible `std::reference_wrapper`-like class. It stores a reference as a
pointer, supports `get()`, implicit conversion back to `T&`, and copies like a reference rather than copying the
referenced object.

It follows the spirit of `std::reference_wrapper`, but adds Kokkos annotations and forwards `operator()` and
`operator[]` when the wrapped object supports them. This makes it useful for accessors and functors captured into device
objects.

```cpp
int value = 3;
auto r = ref(value);
r.get() = 7;  // value -> 7
```

If the wrapped object is callable or subscriptable, the wrapper forwards those operations.
```cpp
auto wrapped_accessor = ref(accessor);
wrapped_accessor(i);  // Same as accessor(i)
```

### `aggregate`
`aggregate` is a tagged bag of types: Boost::hana-like, but Kokkos-friendly. Tags may be any type (complete or incomplete).

```cpp
struct DT;
struct MAX_ITERS;
struct CENTER;

auto cfg = aggregate()
    .append<DT>(0.01)
    .append<MAX_ITERS>(1000);

cfg.get<DT>();         // Returns 0.01
has<MAX_ITERS>(cfg);   // true
```

If a stored value is callable, `get<Tag>(args...)` calls it directly.
```cpp
auto particles = aggregate()
    .append<CENTER>(center_accessor)
    .append<DT>(0.01);

auto center = particles.get<CENTER>(i);
```

Use `project` to copy out a smaller aggregate while preserving the requested tag order.
```cpp
auto small = project<CENTER, DT>(particles);
```

`variant_aggregate` stores a tagged bag of variants. `runtime_aggregate` stores variants behind runtime string tags.

### `storage`
`storage` normalizes owning and non-owning data semantics. The `store` helper follows the rule we usually want:
lvalues are stored by reference, rvalues are stored by value, and pointers remain pointers.

```cpp
int value = 4;

auto a = store(value);  // Non-owning; a.get() returns int&
auto b = store(7);      // Owning;     b.get() returns int&
auto c = store(&value); // Pointer;    c.get() returns int*

a.get() = 9;  // value -> 9
```

## **Kokkos-like views**
These utilities build on Kokkos views and DualViews while using Mundy's `sync_to_*` and `modify_on_*` naming.

### `NgpView`
`NgpView` is Mundy's default `Kokkos::DualView` wrapper. The most important pattern is: sync before reading in a memory
space, mark modified after writing in that memory space. `NgpView` is exactly `Kokkos::DualView` except with the same API as STK.

```cpp
NgpView<int*> values("values", n);

auto h_values = values.view_host();
for (size_t i = 0; i < n; ++i) {
  h_values(i) = i;
}
values.modify_on_host();

values.sync_to_device();
auto d_values = values.view_device();
Kokkos::parallel_for("scale", n, KOKKOS_LAMBDA(const int i) {
  d_values(i) *= 2;
});
values.modify_on_device();

values.sync_to_host();
```

### `NgpPool`
`NgpPool` is a host/device pool of default-constructible objects. Single-value use is intentionally simple.

```cpp
NgpPool<int> pool(10);

pool.add_host(42);
auto value = pool.acquire_host();  // Returns 42

pool.modify_on_host();  // Mark host-side changes before syncing the pool elsewhere.
```

For bulk movement, use the batch APIs.
```cpp
pool.batch_add_host(std::vector<int>{1, 2, 3, 4});
NgpView<int*> values = pool.batch_acquire(4);
pool.batch_add(values);
```
