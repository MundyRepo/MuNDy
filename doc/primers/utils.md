# MundyUtils

See the {ref}`MundyUtils API directory <dir_mundy_core_utils>`.

This subpackage contains small Kokkos-friendly utilities that fill the gaps left by the standard library and by
host-only APIs. The core functionality includes
 - **Portable assertions and compile-time strings**
 - **Type-pack utilities**
 - **Device-friendly containers and ownership wrappers**
 - **Tagged aggregates**
 - **Kokkos-like host/device data structures**

The overall style is the same as the rest of Mundy: small focused types, explicit free functions, and APIs that work in
host/device code without hiding ownership or synchronization.

## Portable helpers
These are the lowest-level tools in the package. They are mostly about writing portable code without giving up readable
errors, compile-time strings, or simple metaprogramming.

### `throw_assert`
{ref}`MUNDY_THROW_REQUIRE <file_mundy_core_utils_src_mundy_utils_throw_assert.hpp>` and {ref}`MUNDY_THROW_ASSERT <file_mundy_core_utils_src_mundy_utils_throw_assert.hpp>` are the main assertion macros.

```cpp
MUNDY_THROW_REQUIRE(assertion, ExceptionType, message);
MUNDY_THROW_ASSERT(assertion, ExceptionType, message);
```

Their behavior is:

| **Macro** | **Behavior** |
|-----------|--------------|
| `MUNDY_THROW_REQUIRE(...)` | Always checks the assertion. On host it throws the requested exception; on device it aborts. |
| `MUNDY_THROW_ASSERT(...)` | Same behavior in debug builds; compiled out in release builds except for minimal type use. |

The message can be a string literal, {ref}`StringLiteral <exhale_struct_structmundy_1_1StringLiteral>`, `std::string`, or a sink expression such as `sink() << ...`.
Only compile-time-printable messages are guaranteed to appear in device abort output.

```cpp
MUNDY_THROW_REQUIRE(i < n, std::out_of_range,
                    sink() << "i = " << i << " but n = " << n);
```

Use {ref}`MUNDY_THROW_REQUIRE <file_mundy_core_utils_src_mundy_utils_throw_assert.hpp>` for contract-like checks that must always run. Use {ref}`MUNDY_THROW_ASSERT <file_mundy_core_utils_src_mundy_utils_throw_assert.hpp>` for debug-only
internal invariants.

### {ref}`StringLiteral <exhale_struct_structmundy_1_1StringLiteral>`
{ref}`StringLiteral <exhale_struct_structmundy_1_1StringLiteral>`\<N\> wraps a string literal as a compile-time value. This is useful when a string participates in a type,
a policy, or a compile-time comparison.

```cpp
template <StringLiteral Name>
struct FieldTag {
  inline static constexpr auto name = Name;
};

using Position = FieldTag<make_string_literal("position")>;
static_assert(Position::name == make_string_literal("position"));
```

{ref}`StringLiteral <exhale_struct_structmundy_1_1StringLiteral>`s can be concatenated at compile time.

```cpp
constexpr auto full = make_string_literal("particle.") + Position::name;
static_assert(full == make_string_literal("particle.position"));
```

### {ref}`StringSink <exhale_struct_structmundy_1_1StringSink>`
`sink()` starts a `<<`-style message builder. Literal-only pipelines remain compile-time objects; once you stream a
runtime value, the pipeline becomes a chunked runtime sink.

```cpp
constexpr auto a = sink() << "left " << "right";
auto b = sink() << "i = " << i << ", n = " << n;
```

The two important sink families are:

| **Type** | **When you get it** | **Main use** |
|----------|----------------------|--------------|
| {ref}`StringLiteralSink <exhale_struct_structmundy_1_1StringLiteralSink>` | Every streamed chunk is compile-time text. | `constexpr` strings and device-printable assertion messages. |
| {ref}`StringSink <exhale_struct_structmundy_1_1StringSink>` | At least one streamed chunk is a runtime value. | Lazy runtime message construction with `to_string()`. |

This is why the same `sink() << ...` syntax works for both compile-time and runtime diagnostics.

### `type_traits`
Mundy offers a few intentionally small type-pack traits.

| **Trait** | **Meaning** |
|-----------|-------------|
| `contains_type_v<T, Ts...>` | `true` if `T` appears in `Ts...`. |
| `count_type_v<T, Ts...>` | Number of times `T` appears in `Ts...`. |
| `index_finder_v<T, Ts...>` | Index of a unique `T` in `Ts...`. |
| `type_at_index_t<I, Ts...>` | The `I`th type in `Ts...`. |

These traits are mostly there to support {ref}`tuple <exhale_struct_structmundy_1_1tuple>`, {ref}`variant <exhale_struct_structmundy_1_1variant>`, and the aggregate types.

### `rng`
`make_philox(seed, counter)` constructs an `openrand::Philox` generator from `size_t` inputs.

```cpp
auto rng = make_philox(seed, counter);
auto u = rng.rand<double>();
```

This helper exists because Philox expects a `uint64_t` seed and `uint32_t` counter, while Mundy often works with
`size_t`. The helper performs the conversion and debug-checks that the values fit.

### Minor helpers

| **Helper** | **Use** |
|------------|---------|
| `MUNDY_ATTRIBUTE_UNUSED` | Mark intentionally unused declarations. |
| `MUNDY_SUPPRESS_*` macros | Locally silence compiler warnings around unavoidable patterns. |
| `do_not_optimize_away(x)` | Keep values live in benchmarks or micro-tests. |
| `make_string_array(...)` | Build a `Teuchos::Array<std::string>` from a parameter pack. |

## Containers and ownership wrappers
These utilities mirror familiar standard-library ideas while staying friendly to Kokkos kernels and constexpr use.

### {ref}`tuple <exhale_struct_structmundy_1_1tuple>`
{ref}`mundy::tuple <exhale_struct_structmundy_1_1tuple>`\<Ts...\> is a small heterogeneous value container. The core interface is:

```cpp
auto t = make_tuple(1, 2.5, "test");

get<0>(t);      // 1
get<double>(t); // 2.5

auto both = tuple_cat(make_tuple(1, 2), make_tuple(3.0));
```

The main supported operations are:

| **Operation** | **Meaning** |
|---------------|-------------|
| `get<I>(t)` | Access by index. |
| `get<T>(t)` | Access by unique type. |
| `tuple_size_v<T>` | Number of elements. |
| `tuple_element_t<I, T>` | Type of the `I`th element. |
| `make_tuple(...)` | Build a tuple by value. |
| `tuple_cat(a, b)` | Concatenate two {ref}`mundy::tuple <exhale_struct_structmundy_1_1tuple>`s. |

Compared with `std::tuple`, this is smaller out of time constraints. If you find there is a missing operation you need or would make your life easier, please open an issue or submit a PR. The main differences are:

| **`std::tuple`** | **{ref}`mundy::tuple <exhale_struct_structmundy_1_1tuple>`** |
|------------------|--------------------|
| Supports the full standard tuple protocol. | Supports only the tuple operations Mundy uses. |
| Has `tie`, `forward_as_tuple`, `apply`, variadic `tuple_cat`, and richer conversions. | Omits those helpers. |
| `get` preserves full value category. | `get` returns mutable or const lvalue references. |
| `tuple_cat` works on tuple-like inputs. | `tuple_cat` joins two {ref}`mundy::tuple <exhale_struct_structmundy_1_1tuple>`s and copies values into the result. |

### {ref}`variant <exhale_struct_structmundy_1_1variant>`
{ref}`mundy::variant <exhale_struct_structmundy_1_1variant>`\<Ts...\> is a Kokkos-compatible `std::variant`-like class with one active alternative at a time.

```cpp
variant<int, double> v(2.5);

holds_alternative<double>(v);  // true
get<double>(v);                // 2.5

v = 4;
auto shifted = visit([](const auto& value) { return static_cast<double>(value) + 1.0; }, v);
```

The main API is:

| **Operation** | **Meaning** |
|---------------|-------------|
| `v.index()` | Active alternative index. |
| `holds_alternative<T>(v)` | Check the active type. |
| `get<T>(v)` / `get<I>(v)` | Access the active alternative. |
| `visit(visitor, v)` | Dispatch on the active alternative. |
| `variant_size_v<Variant>` | Number of alternatives. |
| `variant_alternative_t<I, Variant>` | `I`th alternative type. |

Relative to `std::variant`, Mundy's {ref}`variant <exhale_struct_structmundy_1_1variant>` is more restrictive: all alternatives must be default constructible and
copy assignable, wrong-type access is debug-checked, and `visit` requires a homogeneous return type across all
alternatives.

### {ref}`reference_wrapper <exhale_class_classmundy_1_1reference__wrapper>`
{ref}`reference_wrapper <exhale_class_classmundy_1_1reference__wrapper>`\<T\> is a Kokkos-compatible `std::reference_wrapper`-like class.

```cpp
int value = 3;
auto r = ref(value);
r.get() = 7;  // value -> 7
```

Core operations are:

| **Operation** | **Meaning** |
|---------------|-------------|
| `ref(x)` | Wrap a mutable lvalue reference. |
| `cref(x)` | Wrap a const lvalue reference. |
| `r.get()` | Recover the wrapped reference. |
| implicit conversion to `T&` | Use it where a reference is expected. |
| `r(args...)` / `r[i]` | Forward call or subscript when the wrapped object supports it. |

This makes it especially useful for accessors and functors that need reference-like semantics inside device-friendly
objects.

### {ref}`storage <exhale_class_classmundy_1_1storage>`
{ref}`storage <exhale_class_classmundy_1_1storage>`\<T\> normalizes owning and non-owning storage.

The `store(...)` helper follows the rule Mundy usually wants:

| **Input** | **Stored as** |
|-----------|---------------|
| lvalue | {ref}`reference_wrapper <exhale_class_classmundy_1_1reference__wrapper>`\<T\> |
| rvalue | owning value |
| pointer | pointer |
| existing {ref}`storage <exhale_class_classmundy_1_1storage>`\<U\> | unwrapped stored policy |

```cpp
int value = 4;

auto a = store(value);  // non-owning
auto b = store(7);      // owning
auto c = store(&value); // pointer

a.get() = 9;  // value -> 9
```

This is a small but useful way to write APIs that accept either borrowed or owned data without manual overload sets.

## Tagged aggregates
The aggregate family gives Mundy a lightweight, Kokkos-compatible alternative to larger metaprogramming libraries.

### {ref}`aggregate <exhale_class_classmundy_1_1aggregate>`
{ref}`aggregate <exhale_class_classmundy_1_1aggregate>` is a tagged bag of values where tags are types. It is similar in spirit to boost::hana::Map. Tags may be any type (complete or incomplete). In most cases, we use empty incomplete structs as tags, but there is no requirement to do so.

```cpp
struct DT;
struct MAX_ITERS;
struct CENTER;

auto cfg = aggregate()
    .append<DT>(0.01)
    .append<MAX_ITERS>(1000);

cfg.get<DT>();        // 0.01
has<MAX_ITERS>(cfg);  // true
```

The core operations are:

| **Operation** | **Meaning** |
|---------------|-------------|
| `aggregate()` | Start an empty aggregate. |
| `.append<Tag>(value)` | Return a new aggregate with one more tagged value. |
| `get<Tag>(agg)` / `agg.get<Tag>()` | Access by tag. |
| `get<I>(agg)` / `agg.get<I>()` | Access by index. |
| `has<Tag>(agg)` | Check whether a tag is present. |
| `project<Tags...>(agg)` | Copy out a smaller aggregate in the requested tag order. |

If a stored value is callable, `get<Tag>(args...)` is syntactic sugar for calling that stored callable directly.

```cpp
auto particles = aggregate()
    .append<CENTER>(center_accessor)
    .append<DT>(0.01);

auto center = particles.get<CENTER>(i);
```

### {ref}`variant_aggregate <exhale_class_classmundy_1_1variant__aggregate>`
`variant_aggregate<VariantType, Tags...>` stores one `VariantType` per compile-time tag.

```cpp
using Value = variant<int, double>;

struct A;
struct B;

auto vagg = make_variant_aggregate<Value>()
    .append<A>(Value(1))
    .append<B>(Value(2.0));
```

Use this when the tags are known at compile time but each tagged slot needs variant-valued runtime content.

### {ref}`runtime_aggregate <exhale_class_classmundy_1_1runtime__aggregate>`
`runtime_aggregate<VariantType>` stores variant values behind runtime string tags.

```cpp
using Value = variant<int, double>;

auto ragg = make_runtime_aggregate<Value>()
    .append("dt", Value(0.01))
    .append("iters", Value(1000));
```

This is the runtime-tagged counterpart to {ref}`variant_aggregate <exhale_class_classmundy_1_1variant__aggregate>`.

## Kokkos-like host/device data structures
These utilities build on Kokkos views and dual views while using Mundy's `sync_to_*` and `modify_on_*` naming.

### {ref}`NgpView <exhale_class_classmundy_1_1NgpViewT>`
{ref}`NgpView <exhale_class_classmundy_1_1NgpViewT>` is Mundy's default wrapper around `Kokkos::DualView`. The synchronization rule is the one you already expect:
sync before reading in a memory space, mark modified after writing in that memory space.

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

The important helpers are:

| **Operation** | **Meaning** |
|---------------|-------------|
| `view_host()` / `view_device()` | Access the host or device view. |
| `modify_on_host()` / `modify_on_device()` | Mark which side was written. |
| `sync_to_host()` / `sync_to_device()` | Synchronize stale data. |
| `need_sync_to_host()` / `need_sync_to_device()` | Check whether a sync is needed. |

### {ref}`NgpPool <exhale_class_classmundy_1_1NgpPoolT>`
{ref}`NgpPool <exhale_class_classmundy_1_1NgpPoolT>` is a host/device pool of default-constructible objects built on top of {ref}`NgpView <exhale_class_classmundy_1_1NgpViewT>`.

```cpp
NgpPool<int> pool(10);

pool.add_host(42);
auto value = pool.acquire_host();  // value -> 42

pool.modify_on_host();
```

Single-item and batch APIs are both available.

| **Operation** | **Meaning** |
|---------------|-------------|
| `add(x)` / `add_host(x)` | Push one item on device or host. |
| `acquire()` / `acquire_host()` | Pop one item on device or host. |
| `batch_add(view)` / `batch_add_host(vector)` | Push many items. |
| `batch_acquire(n)` / `batch_acquire_host(n)` | Pop many items. |
| `reserve(n)` | Increase capacity. |
| `size()` / `capacity()` | Device-side size and capacity accessors. |
| `size_host()` / `capacity_host()` | Host-side size and capacity accessors. |

```cpp
pool.batch_add_host(std::vector<int>{1, 2, 3, 4});
NgpView<int*> values = pool.batch_acquire(4);
pool.batch_add(values);
```

Just like {ref}`NgpView <exhale_class_classmundy_1_1NgpViewT>`, you still use `modify_on_*`, `sync_to_*`, and `need_sync_to_*` to manage host/device state.
