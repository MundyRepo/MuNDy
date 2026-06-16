# MundyGeom {#MundyGeom}

\brief Geometric primitives and utilities.

See the \ref mundy/core/geom "MundyGeom directory reference".

This subpackage contains small Kokkos-friendly geometric primitives and the free functions that act on them. The core
functionality includes
 - **Primitives**
 - **Distance queries**
 - **Bounding geometry**
 - **Periodicity and image management**
 - **Rigid transforms**
 - **Random generation**

MundyGeom builds directly on MundyMath. A `Point<Scalar>` behaves like a `Vector3<Scalar>`, so most geometry code is
ordinary vector arithmetic wrapped in shape-specific names. Just like MundyMath, the API is intentionally free-function
heavy: you construct light primitives, then call `distance`, `compute_aabb`, `wrap_rigid`, `translate`, and similar
algorithms on them.

The default primitive aliases own their coordinate data, but the primitive templates can also store view-backed points
and orientations. The same free functions work with both owning and view-backed primitives.

## Primitives
Primitives are light value types. They store the data that defines a shape; they do not allocate, own mesh entities, or
hide the underlying math. Prefer explicit construction, since default constructors for sized shapes often do not encode
meaningful radii or lengths.

### Common primitive aliases

| **Type** | **Stored data** | **Use when you mean** |
|----------|------------------|------------------------|
| `Point<Scalar>` | `x, y, z` | A position in 3D space. |
| `Line<Scalar>` | `center, direction` | An infinite oriented line. |
| `LineSegment<Scalar>` | `start, end` | A finite segment between two points. |
| `VSegment<Scalar>` | `start, middle, end` | Two connected finite segments. |
| `AABB<Scalar>` | `min_corner, max_corner` | An axis-aligned bounding box. |
| `Sphere<Scalar>` | `center, radius` | A ball with no orientation. |
| `Ellipsoid<Scalar>` | `center, orientation, radii` | An oriented ellipsoid. |
| `Spherocylinder<Scalar>` | `center, orientation, radius, length` | A capsule-like body aligned with its body z-axis. |
| `SpherocylinderSegment<Scalar>` | `start, end, radius` | A capsule-like body stored by endpoints. |
| `Circle3D<Scalar>` | `center, orientation, radius` | An oriented circle in 3D. |
| `Ring<Scalar>` | `center, orientation, major_radius, minor_radius` | An oriented thick ring. |

Oriented shapes use a `Quaternion<Scalar>`. The center is always in the lab frame, and the orientation maps the body's
local frame into the lab frame. For rods and circular primitives, the body z-axis is the natural long axis or normal.

### Construction
```cpp
Point<double> x{1.0, 2.0, 3.0};

LineSegment<double> segment{Point<double>{0.0, 0.0, 0.0},
                            Point<double>{1.0, 0.0, 0.0}};

Sphere<double> sphere{x, 0.5};

Ellipsoid<double> ellipsoid{x,
                            Quaterniond::identity(),
                            Point<double>{1.0, 2.0, 3.0}};

Spherocylinder<double> rod{x,
                           Quaterniond::identity(),
                           0.25,
                           2.0};
```

### Accessors and mutation
Accessors are named after the geometry rather than hidden behind a generic trait layer.

```cpp
sphere.center()[0] = 2.0;
sphere.set_radius(0.25);

auto start = segment.start();
auto tangent = segment.end() - segment.start();

auto center = ellipsoid.center();
auto radii = ellipsoid.radii();
```

For `AABB`, `[]` treats the box as `{x_min, y_min, z_min, x_max, y_max, z_max}`.

```cpp
AABB<double> box{-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};

box.min_corner();  // Point<double>{-1.0, -1.0, -1.0}
box.max_corner();  // Point<double>{ 1.0,  1.0,  1.0}
box[3];            // x_max
```

### Views
Geom views come from MundyMath views. Since `Point` is an accessor-backed `Vector3`, and primitives are templated on
their point and orientation types, a primitive can hold non-owning components without copying the underlying data.

```cpp
double coords[6] = {0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0};

auto start = get_vector3_view<double>(coords + 0);
auto end = get_vector3_view<double>(coords + 3);

LineSegment<double, decltype(start), decltype(end)> segment_view{start, end};
segment_view.end()[1] = 2.0;  // coords -> {0, 0, 0, 1, 2, 0}
```

The same pattern applies to oriented shapes by viewing both the point data and the quaternion data.

```cpp
double center_data[3] = {0.0, 0.0, 0.0};
double quat_data[4] = {0.0, 0.0, 0.0, 1.0};  // raw storage: x, y, z, w
double radii_data[3] = {1.0, 2.0, 3.0};

auto center = get_vector3_view<double>(center_data + 0);
auto orient = get_quaternion_view<double>(quat_data + 0);
auto radii = get_vector3_view<double>(radii_data + 0);

Ellipsoid<double, decltype(center), decltype(orient)> ellipsoid_view{center, orient, radii};
```

Use views when geometry is a typed window into existing storage. View semantics apply to accessor-backed math
components; extra scalar fields such as radius and length remain scalar values. Algorithms that synthesize new geometry,
such as `translate`, generally return owning results.

### Reference points
Several periodic helpers are defined in terms of a primitive's reference point. This is the point used for rigid image
shifts and rigid wrapping.

| **Primitive** | **`reference_point(shape)`** |
|---------------|-------------------------------|
| `Point` | the point itself |
| `Line` | `center()` |
| `LineSegment` | `start()` |
| `VSegment` | `start()` |
| `AABB` | `min_corner()` |
| `Sphere` | `center()` |
| `Ellipsoid` | `center()` |
| `Spherocylinder` | `center()` |
| `SpherocylinderSegment` | `start()` |
| `Circle3D` | `center()` |
| `Ring` | `center()` |

That reference point is what `shift_image(shape, lattice_vector, metric)` moves between periodic images.

## distance
The main interface is a single overloaded family of free functions:

```cpp
auto d = distance(shape1, shape2);
```

For most surface-bearing objects, the default meaning is a signed separation distance: positive means separated, zero
means touching, and negative means overlapping.

```cpp
Sphere<double> a{Point<double>{0.0, 0.0, 0.0}, 0.5};
Sphere<double> b{Point<double>{2.0, 0.0, 0.0}, 0.25};

auto gap = distance(a, b);  // 2.0 - 0.5 - 0.25
```

When you need geometric witnesses, use an overload with output arguments.

```cpp
Point<double> p{0.5, 1.0, 0.0};
LineSegment<double> s{Point<double>{0.0, 0.0, 0.0},
                      Point<double>{1.0, 0.0, 0.0}};

Point<double> closest;
double u;
Vector3d sep;

auto d = distance(p, s, closest, u, sep);
```

Here `closest` is the closest point on `s`, `u` is the segment parameter, and `sep` points from the first argument
toward the second.

### Distance tags
Most calls use the default overload directly, but some families support an explicit distance tag.

| **Tag** | **Meaning** | **Common use** |
|---------|-------------|----------------|
| `SharedNormalSigned{}` | Signed separation measured along a shared normal. | Surface-based queries such as sphere-sphere or ellipsoid-ellipsoid. |
| `Euclidean{}` | Standard Euclidean distance. | Queries such as `Circle3D-Circle3D`. |

```cpp
auto signed_gap = distance(SharedNormalSigned{}, a, b);
```

### Output conventions
The overload family is broad, but the outputs follow a small set of recurring patterns.

| **Output** | **Meaning** |
|------------|-------------|
| `sep` | Separation vector from the first object toward the second. |
| `closest_point` | Closest point on the second queried object for one-sided queries. |
| `closest_point1`, `closest_point2` | Closest points on both objects for pair queries. |
| `arch_length` | Query-specific line/segment parameter. For lines it is the signed coordinate along `direction()`. For line segments it is the affine parameter on the stored endpoints. |
| `shared_normal` | Shared witness normal for implicit surface methods. |

### Supported free-function families
The currently supported distance families are explicit rather than fully generic.

| **Family** | **Default call** | **Common witness overloads** |
|------------|------------------|------------------------------|
| Point-point | `distance(p, q)` | `distance(p, q, sep)` |
| Point-line | `distance(p, line)` | `distance(p, line, closest_point, arch_length, sep)` |
| Point-line segment | `distance(p, segment)` | `distance(p, segment, closest_point, arch_length, sep)` |
| Line-line | `distance(line0, line1)` | `distance(line0, line1, closest_point0, closest_point1, arch_length0, arch_length1, sep)` |
| Point-sphere | `distance(p, sphere)` | `distance(p, sphere, sep)` |
| Line-sphere | `distance(line, sphere)` | `distance(line, sphere, closest_point, arch_length, sep)` |
| Line segment-sphere | `distance(segment, sphere)` | `distance(segment, sphere, closest_point, arch_length, sep)` |
| Sphere-sphere | `distance(sphere0, sphere1)` | `distance(sphere0, sphere1, sep)` |
| Point-ellipsoid | `distance(p, ellipsoid)` | `distance(p, ellipsoid, closest_point, shared_normal)` |
| Ellipsoid-ellipsoid | `distance(e0, e1)` | `distance(e0, e1, closest_point0, closest_point1, shared_normal0, shared_normal1)` |
| Circle3D-circle3D | `distance(circle0, circle1)` | `distance(circle0, circle1, closest_point0, closest_point1, shared_normal0, shared_normal1)` |

Two especially useful witness-heavy calls are

```cpp
Point<double> foot;
Vector3d normal;
auto phi = distance(p, ellipsoid, foot, normal);
```

and

```cpp
Point<double> c0;
Point<double> c1;
Vector3d n0;
Vector3d n1;

auto d = distance(circle0, circle1, c0, c1, n0, n1);
```

Not every primitive pair has a distance overload. When in doubt, treat the table above as the supported surface area.

## Bounding geometry
MundyGeom provides cheap conservative bounding queries for broad-phase filtering and neighbor pruning.

```cpp
auto box = compute_aabb(sphere);
auto radius = compute_bounding_radius(sphere);
```

`compute_aabb` returns an axis-aligned box in the current coordinate frame. `compute_bounding_radius` returns a scalar
radius large enough to contain the primitive about its geometric center.

### Bounding free functions

| **Operation** | **Supported shapes** | **Meaning** |
|---------------|----------------------|-------------|
| `compute_aabb(shape)` | `Point`, `LineSegment`, `Sphere`, `Ellipsoid`, `Spherocylinder`, `SpherocylinderSegment` | Conservative axis-aligned box in the current frame. |
| `compute_aabb(shape, metric)` | same families | Periodic-aware AABB, unwrapping multi-point objects near their reference point before boxing. |
| `compute_bounding_radius(shape)` | `Point`, `LineSegment`, `Sphere`, `Ellipsoid`, `Spherocylinder`, `SpherocylinderSegment` | Conservative scalar radius about the geometric center. |
| `intersects(aabb0, aabb1)` | `AABB` pairs | `true` when two AABBs overlap or touch. |

```cpp
auto maybe_close = intersects(compute_aabb(a), compute_aabb(b));
```

This is intentionally a conservative layer: it is cheap enough for broad-phase work, then you follow with an exact
query such as `distance` only when needed.

## Periodicity
A metric defines what “nearby” means. In free space, the separation from `x` to `y` is just `y - x`. In periodic space,
the metric applies the minimum-image convention.

### Available metric types

| **Metric** | **Interpretation** |
|------------|--------------------|
| `EuclideanMetric<Scalar>` | No periodic directions. |
| `PeriodicMetric<Scalar>` | Fully periodic general cell given by a cell matrix. |
| `PeriodicMetricX<Scalar>` | Periodic only in x. |
| `PeriodicMetricY<Scalar>` | Periodic only in y. |
| `PeriodicMetricXY<Scalar>` | Periodic in x and y. |
| `PeriodicMetricYZ<Scalar>` | Periodic in y and z. |
| `PeriodicScaledMetric<Scalar>` | Axis-aligned periodic box given by side lengths. |

```cpp
PeriodicScaledMetric<double> metric{Vector3d{10.0, 10.0, 10.0}};

Point<double> x{9.8, 0.0, 0.0};
Point<double> y{0.2, 0.0, 0.0};

auto sep = metric.sep(x, y);  // approximately Point<double>{0.4, 0.0, 0.0}
```

### Point-level metric operations

| **Operation** | **Meaning** |
|---------------|-------------|
| `metric.sep(x, y)` | Minimum-image separation from `x` to `y`. |
| `metric.wrap(x)` | Wrap a point into the primary cell. |
| `metric.shift_image(x, lattice_vector)` | Shift a point by an integer lattice image. |
| `metric.to_fractional(x)` / `metric.from_fractional(x_frac)` | Convert between real and fractional coordinates. |

### Geometry-level periodic helpers

| **Operation** | **Use when** |
|---------------|--------------|
| `shift_image(shape, lattice_vector, metric)` | The whole object should move to a specified image. |
| `wrap_rigid(shape, metric)` / `wrap_rigid_inplace(shape, metric)` | The shape should move as one rigid object using its reference point. |
| `wrap_points(shape, metric)` / `wrap_points_inplace(shape, metric)` | Each stored point may be wrapped independently. |
| `unwrap_points_to_ref(shape, metric, ref)` / `unwrap_points_to_ref_inplace(shape, metric, ref)` | A multi-point shape should be made coherent near `ref`. |

```cpp
auto wrapped_segment = wrap_rigid(segment, metric);

auto pointwise = wrap_points(segment, metric);
auto coherent = unwrap_points_to_ref(pointwise, metric, segment.start());

auto shifted = shift_image(segment, Vector3<int>{1, 0, 0}, metric);
```

For multi-point physical bodies, `wrap_rigid` is usually the right choice. `wrap_points` is useful when you want each
stored point independently in the primary cell, but it can make long objects look broken across image boundaries until
you call `unwrap_points_to_ref`.

## Transforms
Transforms follow the same pattern as MundyMath: return-by-value by default, `_inplace` when mutation is desired.

### Translation

| **Operation** | **Meaning** |
|---------------|-------------|
| `translate(shape, disp)` | Return a translated copy. |
| `translate_inplace(shape, disp)` | Translate the object in place. |

```cpp
auto shifted = translate(sphere, Vector3d{1.0, 0.0, 0.0});
translate_inplace(sphere, Vector3d{-1.0, 0.0, 0.0});
```

### Rotation

| **Operation** | **Meaning** |
|---------------|-------------|
| `rotate(shape, q)` | Rotate about the origin by quaternion `q`. |
| `rotate_inplace(shape, q)` | Rotate in place about the origin. |

```cpp
auto q = Quaterniond::identity();

auto rotated_point = rotate(x, q);
auto rotated_rod = rotate(rod, q);
auto rotated_ellipsoid = rotate(ellipsoid, q);
```

Rotation is defined for all common primitives except `AABB`. Rotating an `AABB` is intentionally not supported because
the result is ambiguous: an oriented box and its enclosing axis-aligned box are different objects.

## Random generation
Random helpers build test geometry and synthetic examples from a Kokkos-friendly random generator. The low-level
generators give you points, directions, and orientations; the higher-level generators assemble full primitives.

### Base random samples

| **Generator** | **Meaning** |
|---------------|-------------|
| `generate_random_point<Scalar>(box, rng)` | Uniform point in an `AABB`. |
| `generate_random_unit_vector<Scalar>(rng)` | Uniform unit vector on the sphere. |
| `generate_random_unit_quaternion<Scalar>(rng)` | Random unit quaternion. |

```cpp
AABB<double> domain{-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};

auto x = generate_random_point<double>(domain, rng);
auto u = generate_random_unit_vector<double>(rng);
auto q = generate_random_unit_quaternion<double>(rng);
```

### Primitive generators

| **Generator family** | **Available forms** |
|----------------------|---------------------|
| `generate_random_line` | random center in `box` and random direction |
| `generate_random_line_segment` | endpoint-sampled form, or center/length-bounded form |
| `generate_random_vsegment` | point-sampled form, or center/length-bounded form |
| `generate_random_circle3D` | center in `box`, bounded radius, random orientation |
| `generate_random_aabb` | point-sampled form, or center/size-bounded form |
| `generate_random_sphere` | center in `box`, bounded radius |
| `generate_random_spherocylinder` | endpoint-sampled form, or center/radius/length-bounded form |
| `generate_random_spherocylinder_segment` | endpoint-sampled form, or radius/length-bounded form |
| `generate_random_ring` | center in `box`, bounded major/minor radii, random orientation |
| `generate_random_ellipsoid` | center in `box`, bounded semi-axis radii, random orientation |

```cpp
auto segment = generate_random_line_segment(domain, 0.2, 1.0, rng);
auto sphere = generate_random_sphere(domain, 0.1, 0.5, rng);
auto rod = generate_random_spherocylinder_segment(domain, 0.05, 0.1, 0.2, 1.0, rng);
auto ellipsoid = generate_random_ellipsoid(domain,
                                           Vector3d{0.1, 0.2, 0.3},
                                           Vector3d{0.5, 0.6, 0.7},
                                           rng);
```

In practice, the random API mirrors the rest of MundyGeom: explicit primitive constructors, explicit free functions,
and no hidden ownership model.
