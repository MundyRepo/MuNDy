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

/// \file UnitTestAccessorExprCoverage.cpp
/// \brief Comprehensive coverage tests for the mundy::mesh accessor expression framework.
///
/// The tests are organized into numbered sections that correspond to the coverage matrix
/// described in each section header.  Every operation family has exactly one TEST_F whose body
/// calls for_each_context<InputFamilies...>, which automatically sweeps the full Cartesian product of:
///
///   ExprInputShape  : {RawAccessor, IntermediateMathExpr}     — how the primary operand is formed
///   EntityExprShape : {OrdinaryEntity, ConnectedEntities}     — how entities are selected
///
/// Full host-side value verification is performed for both entity shapes.
/// For OrdinaryEntity shapes, verification uses expect_field_near over all selected nodes.
/// For ConnectedEntities shapes, verification iterates selected elements and checks the first
/// connected node's field value, using the same coordinate-derived expected value functions.
///
/// Test sections
/// =============
///   Section 0  — Compile-time concept checks (has_static_eval, supports_runtime_reuse).
///   Section 1  — Test functor definitions (apply_expr and sink_expr callables).
///   Section 2  — Fixture infrastructure (mesh, fields, sweep helpers, assertions).
///   Section 3  — Scalar operation coverage.
///   Section 4  — Scalar builtin coverage.
///   Section 5  — Vector operation coverage.
///   Section 6  — Vector builtin coverage.
///   Section 7  — Matrix operation coverage.
///   Section 8  — Matrix builtin coverage.
///   Section 9  — Quaternion operation coverage.
///   Section 10 — Quaternion builtin coverage.
///   Section 11 — Evaluation trigger coverage.
///   Section 12 — Custom expression coverage.
///   Section 13 — RNG expression coverage.

// External
#include <gtest/gtest.h>

// C++ core
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// STK
#include <Trilinos_version.h>

#include <stk_io/FillMesh.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldDataManager.hpp>
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>         // mundy::mesh::BulkData (: stk::mesh::BulkData)
#include <mundy_mesh/FieldComponent.hpp>   // ScalarFieldComponent, Vector3FieldComponent, etc.
#include <mundy_mesh/MeshBuilder.hpp>      // mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // mundy::mesh::MetaData
#include <mundy_mesh/NgpAccessorExpr.hpp>  // full accessor expression API
#include <mundy_utils/rng.hpp>             // ::mundy::make_philox
#include <mundy_utils/throw_assert.hpp>    // MUNDY_THROW_REQUIRE

namespace mundy {
namespace mesh {
namespace {

// =============================================================================
// Section 0 — Compile-time concept checks
//
// Each static_assert targets one property of one expression type family.
// The assertions are grouped and annotated so they serve as living documentation
// of the framework's type-system contracts.
//
// has_static_eval = true
//   The expression's value at a given entity does not change between multiple
//   evaluations during the same kernel invocation.  The fused-assign machinery
//   may cache the result instead of re-evaluating.
//
// supports_runtime_reuse = true
//   The expression maintains internal state (e.g., an RNG counter) and MUST be
//   reused — not reconstructed — across sequential uses within one fused_assign.
//   Reconstructing such an expression twice produces the same result rather than
//   advancing state.
// =============================================================================

// ---- Ordinary entity expression ----

// An ordinary entity expression directly iterates entities selected by a Selector.
// Its field accessor sub-expressions are stable within a single kernel invocation,
// so caching is valid.
using OrdinaryEntityDriver = impl::NgpForEachEntityExprDriver<>;
using OrdinaryEntityExpr = impl::EntityExpr<1, 0, OrdinaryEntityDriver>;
static_assert(OrdinaryEntityExpr::has_static_eval,
              "OrdinaryEntityExpr must have static eval: field values are stable within one kernel.");

// ---- Connected entities expression ----

// A ConnectedEntitiesExpr wraps an ordinary entity expression and yields entities
// reachable through mesh connectivity.  Because the set of connected entities
// depends on which target entity is being processed, the value is dynamic —
// neither caching nor runtime reuse is safe.
using ConnectedEntitiesFromOrdinary = impl::ConnectedEntitiesExpr<OrdinaryEntityExpr>;
static_assert(!ConnectedEntitiesFromOrdinary::has_static_eval,
              "ConnectedEntitiesExpr must NOT have static eval: connected entity set is dynamic per target.");
static_assert(!ConnectedEntitiesFromOrdinary::supports_runtime_reuse,
              "ConnectedEntitiesExpr must NOT support runtime reuse.");

// ---- Constant expression ----

// A compile-time constant does not benefit from per-entity caching (there is nothing
// to cache — the value is always the same).  The framework therefore marks it
// non-static to prevent mixing it up with field-accessor caching semantics.
using ConstantDouble = impl::ConstantMathExpr<double>;
static_assert(!ConstantDouble::has_static_eval, "ConstantMathExpr must NOT have static eval (no caching benefit).");

// ---- Binary expression propagation ----

// AddExpr propagates has_static_eval: the sum is static iff both operands are static.
using StaticPlusStatic = impl::AddExpr<OrdinaryEntityExpr, OrdinaryEntityExpr>;
using NonStaticPlusStatic = impl::AddExpr<ConstantDouble, OrdinaryEntityExpr>;
static_assert(StaticPlusStatic::has_static_eval, "AddExpr<static, static> must have static eval.");
static_assert(!NonStaticPlusStatic::has_static_eval, "AddExpr<non-static, static> must NOT have static eval.");

// ---- RNG expression family ----

// A CounterBasedRNGExpr wraps seed and counter sub-expressions.  Sequential draws on
// the *same* rng_expr object advance the internal counter, so reuse is both safe and
// required.  Reconstructing rng(seed, counter) twice returns the same value.
using StaticSeedRNG = impl::CounterBasedRNGExpr<impl::ConstantMathExpr<size_t>, impl::ConstantMathExpr<size_t>,
                                                openrand::Philox, make_philox>;
static_assert(StaticSeedRNG::supports_runtime_reuse,
              "CounterBasedRNGExpr must support runtime reuse: sequential draws must advance state.");

// A RandomDistributionExpr wraps one rng_expr.rand<T>() call.  Drawing randomness is
// a one-shot event; neither static eval nor runtime reuse is meaningful here.
using RNGDraw = impl::RandomDistributionExpr<StaticSeedRNG, double>;
static_assert(!RNGDraw::has_static_eval, "RandomDistributionExpr must NOT have static eval: each draw is independent.");
static_assert(!RNGDraw::supports_runtime_reuse,
              "RandomDistributionExpr must NOT support runtime reuse: draw is consumed immediately.");

// A UniformDistributionExpr wraps rng_expr.uniform<T>(lo, hi).  Same reasoning as draw.
using RNGUniform = impl::UniformDistributionExpr<StaticSeedRNG, double, impl::ConstantMathExpr<double>,
                                                 impl::ConstantMathExpr<double>>;
static_assert(!RNGUniform::has_static_eval, "UniformDistributionExpr must NOT have static eval.");
static_assert(!RNGUniform::supports_runtime_reuse, "UniformDistributionExpr must NOT support runtime reuse.");

// An AddExpr whose operands are RNG draws does NOT propagate runtime reuse even if
// the underlying RNG supports it.  The draw has already been consumed.
using NonReusableAddOfDraws = impl::AddExpr<RNGDraw, impl::ConstantMathExpr<double>>;
static_assert(!NonReusableAddOfDraws::supports_runtime_reuse,
              "AddExpr wrapping RNG draws must NOT support runtime reuse.");

// =============================================================================
// Section 1 — Test functor definitions
//
// These callables are passed to apply_expr and sink_expr throughout the
// coverage file.  All are KOKKOS_INLINE_FUNCTION so they may appear inside
// device kernels.
//
// Naming convention:
//   <Descriptor>ApplyFunc  — read-only: passed to apply_expr, returns a value.
//   <AccessMode>SinkFunc   — side-effecting: passed to sink_expr, mutates arguments.
// =============================================================================

/// Ternary scalar apply: (x, y, bias) -> x + 2*y + bias.
/// Tests variadic apply_expr with a mix of field and constant arguments.
struct ScalarVariadicApplyFunc {
  template <typename X, typename Y, typename Bias>
  KOKKOS_INLINE_FUNCTION auto operator()(const X& x, const Y& y, const Bias& bias) const {
    return x + 2.0 * y + bias;
  }
};

/// Binary vector apply: (v, w) -> 2*v + w (component-wise).
/// Tests apply_expr returning a vector result from two vector inputs.
struct VectorBinaryApplyFunc {
  template <typename V, typename W>
  KOKKOS_INLINE_FUNCTION auto operator()(const V& v, const W& w) const {
    return 2.0 * v + w;
  }
};

/// Vector-scalar mixed apply: (v, s) -> s * v.
/// Tests apply_expr accepting a vector and a scalar and returning a vector.
struct VectorScalarMixedApplyFunc {
  template <typename V, typename S>
  KOKKOS_INLINE_FUNCTION auto operator()(const V& v, const S& s) const {
    return s * v;
  }
};

/// Read-write sink: y += scale * x.
/// Tests sink_expr with a read_write output argument.
struct ReadWriteScaleAddSinkFunc {
  template <typename Y, typename X, typename Scale>
  KOKKOS_INLINE_FUNCTION void operator()(Y& y, const X& x, const Scale& scale) const {
    y += scale * x;
  }
};

/// Overwrite-all sink: y = x + bias.
/// Tests sink_expr with an overwrite_all output argument.
struct OverwriteAffineSinkFunc {
  template <typename Y, typename X, typename Bias>
  KOKKOS_INLINE_FUNCTION void operator()(Y& y, const X& x, const Bias& bias) const {
    y = x + bias;
  }
};

/// Read-only sink: no-op passthrough.
/// Tests sink_expr with a read_only argument (verifies that read_only compiles).
struct ReadOnlySinkFunc {
  template <typename X>
  KOKKOS_INLINE_FUNCTION void operator()(const X& /*x*/) const {
  }
};

/// Mixed-access sink: y += x (read_write), z = x (overwrite_all).
/// Tests sink_expr with three distinct access modes in a single call.
struct MixedAccessSinkFunc {
  template <typename Y, typename X, typename Z>
  KOKKOS_INLINE_FUNCTION void operator()(Y& y, const X& x, Z& z) const {
    y += x;
    z  = x;
  }
};

// =============================================================================
// Section 2 — Fixture infrastructure
//
// The AccessorExprCoverageFixture manages a fixed five-hex mesh with one field
// for every data type tested by the coverage matrix.  Tests call for_each_context,
// which automatically sweeps the Cartesian product of (ExprInputShape × EntityExprShape)
// and presents each combination to the test body as a (CoverageContext, es) pair.
//
// Five-hex mesh layout
// --------------------
//   Element 1: nodes { 1.. 8} in block_1
//   Element 2: nodes { 5..12} in block_1   (shares nodes 5-8 with element 1)
//   Element 3: nodes { 9,13..19} in block_2 (node 9 is shared with elements 2,4,5)
//   Element 4: nodes { 9,20..26} in block_2
//   Element 5: nodes { 9,27..33} in block_3
//
// Test selector: parts_.selected = block_1 - block_2 = elements {1, 2} and their nodes.
// Unselected:    parts_.unselected = everything else (verified to be untouched).
//
// Field data convention
// ---------------------
// Every field is initialized on ALL nodes via coordinate-derived formulas in ExpectedValues.
// The coordinate field is set to coords[i] = {0.01*id, 0.02*id+0.25, 0.03*id+0.5}.
// Input field values are always positive and well-behaved for the operations that need it
// (log, sqrt, division, matrix inverse).
// Output fields (out, vout, mout, qout) start from zero / identity and are re-initialized
// between sub-cases by reinitialize_output_fields().
//
// Coverage context
// ----------------
// ExprInputShape controls how the primary input expression is formed:
//   RawAccessor         : tmp = r()(es)         — a direct field accessor
//   IntermediateMathExpr: tmp = r()(es) / 1.0   — a DivExpr wrapping the accessor
//
// Note: the intermediate case uses division by 1.0 to produce a different static type
// (DivExpr) without changing the numeric value.  The expected function uses the divisor
// to reproduce the same arithmetic on the host.
//
// EntityExprShape controls how 'es' is constructed:
//   OrdinaryEntity   : es = make_entity_expr(bulk, selector, NODE_RANK)
//   ConnectedEntities: es = elements.get_connectivity(NODE_RANK)[0]
//                       (first node of each element in selector)
//
// Value verification is performed for both entity shapes: OrdinaryEntity verifies
// all selected nodes; ConnectedEntities verifies the first connected node of each
// selected element.
// =============================================================================

using DoubleField = stk::mesh::Field<double>;

// ---- Component tags ----

struct RTag;
struct QTag;
struct OutTag;
struct AuxTag;
struct SeedTag;
struct CounterTag;
struct FirstDrawTag;
struct SecondDrawTag;
struct VelTag;
struct ForceTag;
struct VOutTag;
struct MatTag;
struct MatBTag;
struct MOutTag;
struct QuatTag;
struct QuatBTag;
struct QOutTag;
struct OmegaTag;

// ---- Aggregates ----

struct Parts {
  stk::mesh::Part* block1 = nullptr;
  stk::mesh::Part* block2 = nullptr;
  stk::mesh::Part* block3 = nullptr;
  stk::mesh::Selector all_blocks;
  stk::mesh::Selector selected;    // block_1 - block_2: the test-active region
  stk::mesh::Selector unselected;  // complement: must not be written by any test
};

struct Fields {
  // clang-format off
  DoubleField* r           = nullptr;  // scalar input A    (1 component, always > 2)
  DoubleField* q           = nullptr;  // scalar input B    (1 component, always > 3)
  DoubleField* out         = nullptr;  // scalar output     (1 component, reinitialized each sub-case)
  DoubleField* aux         = nullptr;  // scalar auxiliary  (1 component)
  DoubleField* seed        = nullptr;  // RNG seed          (1 component, large positive)
  DoubleField* counter     = nullptr;  // RNG counter       (1 component, large positive)
  DoubleField* first_draw  = nullptr;  // RNG scratch       (1 component)
  DoubleField* second_draw = nullptr;  // RNG scratch       (1 component)
  DoubleField* vel         = nullptr;  // vector3 input A   (3 components)
  DoubleField* force       = nullptr;  // vector3 input B   (3 components)
  DoubleField* vout        = nullptr;  // vector3 output    (3 components)
  DoubleField* mat         = nullptr;  // matrix3 input A   (9 components, invertible)
  DoubleField* mat_b       = nullptr;  // matrix3 input B   (9 components)
  DoubleField* mout        = nullptr;  // matrix3 output    (9 components)
  DoubleField* quat        = nullptr;  // quaternion input A(4 components)
  DoubleField* quat_b      = nullptr;  // quaternion input B(4 components)
  DoubleField* qout        = nullptr;  // quaternion output (4 components)
  DoubleField* omega       = nullptr;  // angular velocity  (3 components)
  // clang-format on
};

// ---- Deterministic expected values ----

/// Static functions mapping coordinate triple (c[0], c[1], c[2]) to expected field values.
///
/// Design constraints:
///   r, q       > 0 on all nodes: safe for log, sqrt, and division.
///   mat        diagonally dominant: safe for inverse and determinant tests.
///   quat, quat_b non-zero norm: safe for normalize and slerp tests.
///   out, vout, mout, qout start from zero/identity: distinguishable after write.
namespace ExpectedValues {

using S = std::array<double, 1>;
using V = std::array<double, 3>;
using M = std::array<double, 9>;
using Q = std::array<double, 4>;

inline S r(const double* c) {
  return {2.0 + c[0] + 0.25 * c[1]};
}
inline S q(const double* c) {
  return {3.0 + 0.5 * c[1] + c[2]};
}
inline S out(const double* /*c*/) {
  return {0.0};
}
inline S aux(const double* c) {
  return {-1.0 + c[0] - c[2]};
}

inline S seed(const double* c) {
  return {1000.0 + 17.0 * c[0] + 31.0 * c[1] + 43.0 * c[2]};
}
inline S counter(const double* c) {
  return {2000.0 + 7.0 * c[0] + 11.0 * c[1] + 13.0 * c[2]};
}
inline S first_draw(const double* /*c*/) {
  return {0.0};
}
inline S second_draw(const double* /*c*/) {
  return {0.0};
}

inline V vel(const double* c) {
  return {1.0 + c[0], 2.0 + c[1], 3.0 + c[2]};
}
inline V force(const double* c) {
  return {4.0 + c[1], 5.0 + c[2], 6.0 + c[0]};
}
inline V vout(const double* /*c*/) {
  return {0.0, 0.0, 0.0};
}
inline V omega(const double* /*c*/) {
  return {0.0, 0.0, 1.0};
}

// Row-major 3x3 storage.  Diagonal entries are c-coordinate-shifted; off-diagonal
// entries are small constants.  Diagonal dominance is preserved for all node IDs
// (entity IDs start at 1, so c values are strictly positive).
inline M mat(const double* c) {
  return {2.0 + c[0], 0.1,        0.2,  //
          0.3,        3.0 + c[1], 0.4,  //
          0.5,        0.6,        4.0 + c[2]};
}
inline M mat_b(const double* c) {
  return {1.0 + c[2], 0.7,        0.8,  //
          0.9,        2.0 + c[0], 1.0,  //
          1.1,        1.2,        3.0 + c[1]};
}
inline M mout(const double* /*c*/) {
  return {0.0, 0.0, 0.0,  //
          0.0, 0.0, 0.0,  //
          0.0, 0.0, 0.0};
}

// Quaternion layout: {x, y, z, w}.  Non-unit but non-zero; both inputs have
// different component patterns so operations like multiply are non-trivial.
inline Q quat(const double* c) {
  return {1.0 + 0.1 * c[0], 0.2 + 0.1 * c[1], -0.3 + 0.1 * c[2], 0.4};
}
inline Q quat_b(const double* c) {
  return {0.9 - 0.1 * c[1], -0.4, 0.1 + 0.1 * c[0], 0.2 + 0.1 * c[2]};
}
inline Q qout(const double* /*c*/) {
  return {0.0, 0.0, 0.0, 1.0};
}

}  // namespace ExpectedValues

// =============================================================================
// Host-side matrix and quaternion helpers
//
// All 3×3 matrices use row-major flat storage (9 elements).
// Quaternion layout: {x, y, z, w} — x/y/z imaginary, w real scalar.
// =============================================================================

inline std::array<double, 9> mat3_scale(const std::array<double, 9>& A, double s) {
  std::array<double, 9> C{};
  for (size_t i = 0; i < 9; ++i) C[i] = A[i] * s;
  return C;
}
inline std::array<double, 9> mat3_add(const std::array<double, 9>& A, const std::array<double, 9>& B) {
  std::array<double, 9> C{};
  for (size_t i = 0; i < 9; ++i) C[i] = A[i] + B[i];
  return C;
}
inline std::array<double, 9> mat3_sub(const std::array<double, 9>& A, const std::array<double, 9>& B) {
  std::array<double, 9> C{};
  for (size_t i = 0; i < 9; ++i) C[i] = A[i] - B[i];
  return C;
}
inline std::array<double, 9> mat3_mul(const std::array<double, 9>& A, const std::array<double, 9>& B) {
  std::array<double, 9> C{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        C[static_cast<size_t>(i * 3 + j)] += A[static_cast<size_t>(i * 3 + k)] * B[static_cast<size_t>(k * 3 + j)];
  return C;
}
inline std::array<double, 3> mat3_vec3_mul(const std::array<double, 9>& A, const std::array<double, 3>& v) {
  std::array<double, 3> w{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      w[static_cast<size_t>(i)] += A[static_cast<size_t>(i * 3 + j)] * v[static_cast<size_t>(j)];
  return w;
}
inline std::array<double, 9> mat3_transpose(const std::array<double, 9>& A) {
  return {A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]};
}
inline std::array<double, 9> mat3_elementwise_mul(const std::array<double, 9>& A, const std::array<double, 9>& B) {
  std::array<double, 9> C{};
  for (size_t i = 0; i < 9; ++i) C[i] = A[i] * B[i];
  return C;
}
inline std::array<double, 9> mat3_elementwise_div(const std::array<double, 9>& A, const std::array<double, 9>& B) {
  std::array<double, 9> C{};
  for (size_t i = 0; i < 9; ++i) C[i] = A[i] / B[i];
  return C;
}
inline double mat3_trace(const std::array<double, 9>& A) {
  return A[0] + A[4] + A[8];
}
inline double mat3_det(const std::array<double, 9>& A) {
  return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
}
inline double mat3_frobenius_norm(const std::array<double, 9>& A) {
  double s = 0.0;
  for (auto v : A) s += v * v;
  return std::sqrt(s);
}
inline double mat3_frobenius_inner_product(const std::array<double, 9>& A, const std::array<double, 9>& B) {
  double ip = 0.0;
  for (size_t i = 0; i < 9; ++i) ip += A[i] * B[i];
  return ip;
}

inline std::array<double, 4> quat_scale(const std::array<double, 4>& q, double s) {
  return {q[0] * s, q[1] * s, q[2] * s, q[3] * s};
}
inline std::array<double, 4> quat_add(const std::array<double, 4>& q1, const std::array<double, 4>& q2) {
  return {q1[0] + q2[0], q1[1] + q2[1], q1[2] + q2[2], q1[3] + q2[3]};
}
inline std::array<double, 4> quat_sub(const std::array<double, 4>& q1, const std::array<double, 4>& q2) {
  return {q1[0] - q2[0], q1[1] - q2[1], q1[2] - q2[2], q1[3] - q2[3]};
}
inline double quat_norm(const std::array<double, 4>& q) {
  return std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}
inline std::array<double, 4> quat_conjugate(const std::array<double, 4>& q) {
  return {-q[0], -q[1], -q[2], q[3]};
}
inline std::array<double, 4> quat_normalize(const std::array<double, 4>& q) {
  const double n = quat_norm(q);
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}
// Hamilton product with {x,y,z,w} layout (w is the real part).
inline std::array<double, 4> quat_mul(const std::array<double, 4>& q1, const std::array<double, 4>& q2) {
  const double x1 = q1[0], y1 = q1[1], z1 = q1[2], w1 = q1[3];
  const double x2 = q2[0], y2 = q2[1], z2 = q2[2], w2 = q2[3];
  return {w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2};
}

// =============================================================================
// Additional host-side helpers for Sections 5–13
// (Vector, Matrix, Quaternion extended operations)
// =============================================================================

// ---- Vector helpers ----

// vec3 * mat3: result[j] = sum_i v[i] * A[i,j]  (same as mat3^T * vec3, i.e. vec_quat convention)
inline std::array<double, 3> vec3_mat3_mul(const std::array<double, 3>& v, const std::array<double, 9>& A) {
  std::array<double, 3> w{};
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 3; ++i)
      w[static_cast<size_t>(j)] += v[static_cast<size_t>(i)] * A[static_cast<size_t>(i * 3 + j)];
  return w;
}

// outer_product(u, v): result[i,j] = u[i] * v[j] (row-major)
inline std::array<double, 9> vec3_outer_product(const std::array<double, 3>& u, const std::array<double, 3>& v) {
  std::array<double, 9> R{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      R[static_cast<size_t>(i * 3 + j)] = u[static_cast<size_t>(i)] * v[static_cast<size_t>(j)];
  return R;
}

// one_norm(v) = sum(|v_i|)
inline double vec3_one_norm(const std::array<double, 3>& v) {
  return std::abs(v[0]) + std::abs(v[1]) + std::abs(v[2]);
}

// inf_norm(v) = max(|v_i|)
inline double vec3_inf_norm(const std::array<double, 3>& v) {
  return std::max({std::abs(v[0]), std::abs(v[1]), std::abs(v[2])});
}

// two_norm_squared(v) = v0^2 + v1^2 + v2^2
inline double vec3_two_norm_squared(const std::array<double, 3>& v) {
  return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

// variance(v): population variance (divide by N=3)
inline double vec3_variance(const std::array<double, 3>& v) {
  const double m = (v[0] + v[1] + v[2]) / 3.0;
  return ((v[0] - m) * (v[0] - m) + (v[1] - m) * (v[1] - m) + (v[2] - m) * (v[2] - m)) / 3.0;
}

// stddev(v) = sqrt(variance(v))
inline double vec3_stddev(const std::array<double, 3>& v) {
  return std::sqrt(vec3_variance(v));
}

// minor_angle(a, b) = acos(dot(a,b) / (|a| * |b|))  — matches mundy's implementation exactly
inline double vec3_minor_angle(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  const double dot_ab = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const double norm_a = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const double norm_b = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  double norm_product = norm_a * norm_b;
  if (norm_product == 0.0) norm_product = 1.0;
  double cosine = dot_ab / norm_product;
  if (cosine > 1.0) cosine = 1.0;
  if (cosine < -1.0) cosine = -1.0;
  return std::acos(cosine);
}

// major_angle(a, b) = pi - minor_angle(a, b)
inline double vec3_major_angle(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return std::acos(-1.0) - vec3_minor_angle(a, b);
}

// ---- Matrix helpers ----

// mat3_sum: sum of all 9 elements
inline double mat3_sum(const std::array<double, 9>& A) {
  double s = 0.0;
  for (auto v : A) s += v;
  return s;
}

// mat3_product: product of all 9 elements
inline double mat3_product(const std::array<double, 9>& A) {
  double p = 1.0;
  for (auto v : A) p *= v;
  return p;
}

// mat3_min: minimum element
inline double mat3_min(const std::array<double, 9>& A) {
  double m = A[0];
  for (auto v : A) m = std::min(m, v);
  return m;
}

// mat3_max: maximum element
inline double mat3_max(const std::array<double, 9>& A) {
  double m = A[0];
  for (auto v : A) m = std::max(m, v);
  return m;
}

// mat3_mean: mean of all 9 elements
inline double mat3_mean(const std::array<double, 9>& A) {
  return mat3_sum(A) / 9.0;
}

// mat3_variance: population variance over 9 elements
inline double mat3_variance(const std::array<double, 9>& A) {
  const double m = mat3_mean(A);
  double var = 0.0;
  for (auto v : A) var += (v - m) * (v - m);
  return var / 9.0;
}

// mat3_stddev: sqrt(variance)
inline double mat3_stddev(const std::array<double, 9>& A) {
  return std::sqrt(mat3_variance(A));
}

// mat3_one_norm: maximum absolute column sum
inline double mat3_one_norm(const std::array<double, 9>& A) {
  double max_col = 0.0;
  for (int j = 0; j < 3; ++j) {
    double col_sum = 0.0;
    for (int i = 0; i < 3; ++i)
      col_sum += std::abs(A[static_cast<size_t>(i * 3 + j)]);
    max_col = std::max(max_col, col_sum);
  }
  return max_col;
}

// mat3_inf_norm: maximum absolute row sum
inline double mat3_inf_norm(const std::array<double, 9>& A) {
  double max_row = 0.0;
  for (int i = 0; i < 3; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < 3; ++j)
      row_sum += std::abs(A[static_cast<size_t>(i * 3 + j)]);
    max_row = std::max(max_row, row_sum);
  }
  return max_row;
}

// mat3_cofactors: cofactor matrix C where C[i,j] = (-1)^(i+j) * minor(i,j)
// mundy stores in row-major order and uses linear index parity for the sign
inline std::array<double, 9> mat3_cofactors(const std::array<double, 9>& A) {
  // Cofactor C[i][j] = (-1)^(linear_index) * M_ij
  // For a 3x3 in row-major, linear index = i*3+j, and parity of (i+j) == parity of (i*3+j).
  std::array<double, 9> C{};
  // Row 0
  C[0] = +(A[4] * A[8] - A[5] * A[7]);
  C[1] = -(A[3] * A[8] - A[5] * A[6]);
  C[2] = +(A[3] * A[7] - A[4] * A[6]);
  // Row 1
  C[3] = -(A[1] * A[8] - A[2] * A[7]);
  C[4] = +(A[0] * A[8] - A[2] * A[6]);
  C[5] = -(A[0] * A[7] - A[1] * A[6]);
  // Row 2
  C[6] = +(A[1] * A[5] - A[2] * A[4]);
  C[7] = -(A[0] * A[5] - A[2] * A[3]);
  C[8] = +(A[0] * A[4] - A[1] * A[3]);
  return C;
}

// mat3_adjugate: transpose of cofactor matrix
inline std::array<double, 9> mat3_adjugate(const std::array<double, 9>& A) {
  return mat3_transpose(mat3_cofactors(A));
}

// mat3_inv: inverse = adjugate / determinant
inline std::array<double, 9> mat3_inv(const std::array<double, 9>& A) {
  const double det = mat3_det(A);
  return mat3_scale(mat3_adjugate(A), 1.0 / det);
}

// mat3_copy: identity copy
inline std::array<double, 9> mat3_copy(const std::array<double, 9>& A) {
  return A;
}

// ---- Quaternion helpers ----

// quat_inverse: conjugate / norm_squared  (layout {x,y,z,w})
inline std::array<double, 4> quat_inverse(const std::array<double, 4>& q) {
  const double ns = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  return {-q[0] / ns, -q[1] / ns, -q[2] / ns, q[3] / ns};
}

// quat_rotate_vec: q * [0,v] * q^{-1}  — rotate v by q
// Using Rodrigues: v' = 2*(u.v)*u + (w^2 - u.u)*v + 2*w*(u×v)
// where u = {q[0],q[1],q[2]}, w = q[3], q need not be unit
// Implementation: via sandwich with inverse
inline std::array<double, 3> quat_rotate_vec(const std::array<double, 4>& q, const std::array<double, 3>& v) {
  // vec_quat representation: [0, v[0], v[1], v[2]] with w=0
  const std::array<double, 4> vq = {v[0], v[1], v[2], 0.0};
  const auto q_inv = quat_inverse(q);
  const auto tmp = quat_mul(q, vq);
  const auto res = quat_mul(tmp, q_inv);
  return {res[0], res[1], res[2]};
}

// vec_rotate_quat: q^{-1} * [0,v] * q  (mundy's vec*quat convention)
inline std::array<double, 3> vec_rotate_quat(const std::array<double, 3>& v, const std::array<double, 4>& q) {
  const std::array<double, 4> vq = {v[0], v[1], v[2], 0.0};
  const auto q_inv = quat_inverse(q);
  const auto tmp = quat_mul(q_inv, vq);
  const auto res = quat_mul(tmp, q);
  return {res[0], res[1], res[2]};
}

// quat_rotate_mat: apply quat_rotate_vec to each column of A
inline std::array<double, 9> quat_rotate_mat(const std::array<double, 4>& q, const std::array<double, 9>& A) {
  std::array<double, 9> R{};
  for (int j = 0; j < 3; ++j) {
    const std::array<double, 3> col = {A[static_cast<size_t>(0 * 3 + j)], A[static_cast<size_t>(1 * 3 + j)],
                                       A[static_cast<size_t>(2 * 3 + j)]};
    const auto rotated = quat_rotate_vec(q, col);
    for (int i = 0; i < 3; ++i)
      R[static_cast<size_t>(i * 3 + j)] = rotated[static_cast<size_t>(i)];
  }
  return R;
}

// mat_rotate_quat: apply vec_rotate_quat to each row of A
inline std::array<double, 9> mat_rotate_quat(const std::array<double, 9>& A, const std::array<double, 4>& q) {
  std::array<double, 9> R{};
  for (int i = 0; i < 3; ++i) {
    const std::array<double, 3> row = {A[static_cast<size_t>(i * 3 + 0)], A[static_cast<size_t>(i * 3 + 1)],
                                       A[static_cast<size_t>(i * 3 + 2)]};
    const auto rotated = vec_rotate_quat(row, q);
    for (int j = 0; j < 3; ++j)
      R[static_cast<size_t>(i * 3 + j)] = rotated[static_cast<size_t>(j)];
  }
  return R;
}

// quat_copy
inline std::array<double, 4> quat_copy(const std::array<double, 4>& q) {
  return q;
}

// quat_slerp: matches mundy's slerp implementation exactly
// Layout {x,y,z,w}. dot uses all 4 components.
inline std::array<double, 4> quat_slerp(const std::array<double, 4>& q1, const std::array<double, 4>& q2_in,
                                         double t) {
  constexpr double epsilon = 1.0e-8;  // matches get_relaxed_zero_tolerance<double>()
  double dot_q12 = q1[0] * q2_in[0] + q1[1] * q2_in[1] + q1[2] * q2_in[2] + q1[3] * q2_in[3];
  std::array<double, 4> q2 = q2_in;
  if (dot_q12 < 0.0) {
    dot_q12 = -dot_q12;
    q2 = {-q2_in[0], -q2_in[1], -q2_in[2], -q2_in[3]};
  }
  if (dot_q12 > 1.0) dot_q12 = 1.0;
  if (1.0 - dot_q12 < epsilon) {
    // Linear fallback
    return {q1[0] + t * (q2[0] - q1[0]), q1[1] + t * (q2[1] - q1[1]), q1[2] + t * (q2[2] - q1[2]),
            q1[3] + t * (q2[3] - q1[3])};
  }
  const double theta = std::acos(dot_q12);
  const double sin_theta = std::sin(theta);
  const double inv_sin_theta = 1.0 / sin_theta;
  const double s1 = std::sin((1.0 - t) * theta) * inv_sin_theta;
  const double s2 = std::sin(t * theta) * inv_sin_theta;
  return {s1 * q1[0] + s2 * q2[0], s1 * q1[1] + s2 * q2[1], s1 * q1[2] + s2 * q2[2],
          s1 * q1[3] + s2 * q2[3]};
}


// ---- Coverage context and enumerations ----

enum class EntityExprShape { OrdinaryEntity, ConnectedEntities };
enum class ExprInputShape { RawAccessor, IntermediateMathExpr };

// Input-family tags: declare which axes of ExprInputShape should be swept.
struct Scalar {};
struct Vector {};
struct Matrix {};
struct Quaternion {};

struct CoverageContext {
  std::string case_name;
  EntityExprShape entity_shape = EntityExprShape::OrdinaryEntity;
  ExprInputShape scalar_shape = ExprInputShape::RawAccessor;
  ExprInputShape vector_shape = ExprInputShape::RawAccessor;
  ExprInputShape matrix_shape = ExprInputShape::RawAccessor;
  ExprInputShape quaternion_shape = ExprInputShape::RawAccessor;
};

static constexpr std::array<ExprInputShape, 2> kAllInputShapes = {ExprInputShape::RawAccessor,
                                                                  ExprInputShape::IntermediateMathExpr};

const char* to_cstr(EntityExprShape s) {
  return (s == EntityExprShape::OrdinaryEntity) ? "ordinary entity" : "connected entities";
}
const char* to_cstr(ExprInputShape s) {
  return (s == ExprInputShape::RawAccessor) ? "raw" : "intermediate";
}

// Set the shape axis for a specific input family.
template <typename InputFamily>
void set_context_shape(CoverageContext& ctx, ExprInputShape shape) {
  if constexpr (std::is_same_v<InputFamily, Scalar>) {
    ctx.scalar_shape = shape;
  } else if constexpr (std::is_same_v<InputFamily, Vector>) {
    ctx.vector_shape = shape;
  } else if constexpr (std::is_same_v<InputFamily, Matrix>) {
    ctx.matrix_shape = shape;
  } else if constexpr (std::is_same_v<InputFamily, Quaternion>) {
    ctx.quaternion_shape = shape;
  } else {
    static_assert(std::is_same_v<InputFamily, Scalar>, "Unknown input family.");
  }
}

// Return a human-readable fragment for the given family's current shape.
template <typename InputFamily>
std::string shape_fragment(const CoverageContext& ctx) {
  if constexpr (std::is_same_v<InputFamily, Scalar>) {
    return std::string("scalar=") + to_cstr(ctx.scalar_shape);
  } else if constexpr (std::is_same_v<InputFamily, Vector>) {
    return std::string("vector=") + to_cstr(ctx.vector_shape);
  } else if constexpr (std::is_same_v<InputFamily, Matrix>) {
    return std::string("matrix=") + to_cstr(ctx.matrix_shape);
  } else if constexpr (std::is_same_v<InputFamily, Quaternion>) {
    return std::string("quaternion=") + to_cstr(ctx.quaternion_shape);
  } else {
    static_assert(std::is_same_v<InputFamily, Scalar>, "Unknown input family.");
    return "";
  }
}

// =============================================================================
// Fixture class
// =============================================================================

class AccessorExprCoverageFixture : public ::testing::Test {
 protected:
  static constexpr double kTolerance = 1.0e-12;
  static constexpr double kInitialValue[] = {-1.0, 2.0, -0.3, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0};

  // --------------------------------------------------------------------------
  // GTest lifecycle
  // --------------------------------------------------------------------------

  void SetUp() override {
    if (stk::parallel_machine_size(MPI_COMM_WORLD) > 2) {
      GTEST_SKIP() << "AccessorExprCoverage tests require 1 or 2 MPI ranks.";
    }
    setup_mesh();
    declare_parts();
    declare_fields();
    build_five_hex_mesh();
    initialize_fields();
  }

  void TearDown() override {
    bulk_data_.reset();
    meta_data_.reset();
  }

  // --------------------------------------------------------------------------
  // Mesh / meta accessors
  // --------------------------------------------------------------------------

  BulkData& bulk() {
    EXPECT_NE(bulk_data_, nullptr);
    return *bulk_data_;
  }

  const BulkData& bulk() const {
    EXPECT_NE(bulk_data_, nullptr);
    return *bulk_data_;
  }

  MetaData& meta() {
    EXPECT_NE(meta_data_, nullptr);
    return *meta_data_;
  }

  const Parts& parts() const {
    return parts_;
  }
  const Fields& fields() const {
    return fields_;
  }

  // --------------------------------------------------------------------------
  // Component accessor factories
  //
  // Each factory wraps the underlying STK field in the appropriate FieldComponent
  // and attaches a type tag so the accessor expression framework can distinguish
  // two fields of the same type.
  // --------------------------------------------------------------------------

  // clang-format off
  auto r()           const { return make_tagged_component<RTag>         (ScalarFieldComponent<double>    (*fields_.r));          }
  auto q()           const { return make_tagged_component<QTag>         (ScalarFieldComponent<double>    (*fields_.q));          }
  auto out()         const { return make_tagged_component<OutTag>       (ScalarFieldComponent<double>    (*fields_.out));        }
  auto aux()         const { return make_tagged_component<AuxTag>       (ScalarFieldComponent<double>    (*fields_.aux));        }
  auto seed()        const { return make_tagged_component<SeedTag>      (ScalarFieldComponent<double>    (*fields_.seed));       }
  auto counter()     const { return make_tagged_component<CounterTag>   (ScalarFieldComponent<double>    (*fields_.counter));    }
  auto first_draw()  const { return make_tagged_component<FirstDrawTag> (ScalarFieldComponent<double>    (*fields_.first_draw)); }
  auto second_draw() const { return make_tagged_component<SecondDrawTag>(ScalarFieldComponent<double>    (*fields_.second_draw));}
  auto vel()         const { return make_tagged_component<VelTag>       (Vector3FieldComponent<double>   (*fields_.vel));        }
  auto force()       const { return make_tagged_component<ForceTag>     (Vector3FieldComponent<double>   (*fields_.force));      }
  auto vout()        const { return make_tagged_component<VOutTag>      (Vector3FieldComponent<double>   (*fields_.vout));       }
  auto mat()         const { return make_tagged_component<MatTag>       (Matrix3FieldComponent<double>   (*fields_.mat));        }
  auto mat_b()       const { return make_tagged_component<MatBTag>      (Matrix3FieldComponent<double>   (*fields_.mat_b));      }
  auto mout()        const { return make_tagged_component<MOutTag>      (Matrix3FieldComponent<double>   (*fields_.mout));       }
  auto quat()        const { return make_tagged_component<QuatTag>      (QuaternionFieldComponent<double>(*fields_.quat));       }
  auto quat_b()      const { return make_tagged_component<QuatBTag>     (QuaternionFieldComponent<double>(*fields_.quat_b));     }
  auto qout()        const { return make_tagged_component<QOutTag>      (QuaternionFieldComponent<double>(*fields_.qout));       }
  auto omega()       const { return make_tagged_component<OmegaTag>     (Vector3FieldComponent<double>   (*fields_.omega));      }
  // clang-format on

  // --------------------------------------------------------------------------
  // Entity expression factories
  // --------------------------------------------------------------------------

  auto make_node_entities() {
    return make_entity_expr(bulk(), parts_.selected, stk::topology::NODE_RANK);
  }

  auto make_element_entities() {
    return make_entity_expr(bulk(), parts_.selected, stk::topology::ELEMENT_RANK);
  }

  // --------------------------------------------------------------------------
  // Input shape helpers
  //
  // Each with_*_input invokes fn with an expression of the correct compile-time
  // type: the raw field accessor (RawAccessor branch) or a DivExpr wrapping it
  // (IntermediateMathExpr branch).  The two branches have distinct C++ types,
  // so a runtime conditional inside a single return statement cannot be used;
  // the callback pattern is required.
  //
  // Usage:
  //   with_scalar_input(ctx, es, [&](auto tmp) { ... });
  // --------------------------------------------------------------------------

  template <typename Es, typename Fn>
  void with_scalar_input(const CoverageContext& ctx, const Es& es, Fn&& fn) const {
    if (ctx.scalar_shape == ExprInputShape::RawAccessor) {
      std::forward<Fn>(fn)(r()(es));
    } else {
      std::forward<Fn>(fn)(r()(es) / 2.0);
    }
  }

  template <typename Es, typename Fn>
  void with_vector_input(const CoverageContext& ctx, const Es& es, Fn&& fn) const {
    if (ctx.vector_shape == ExprInputShape::RawAccessor) {
      std::forward<Fn>(fn)(vel()(es));
    } else {
      std::forward<Fn>(fn)(vel()(es) / 2.0);
    }
  }

  template <typename Es, typename Fn>
  void with_matrix_input(const CoverageContext& ctx, const Es& es, Fn&& fn) const {
    if (ctx.matrix_shape == ExprInputShape::RawAccessor) {
      std::forward<Fn>(fn)(mat()(es));
    } else {
      std::forward<Fn>(fn)(mat()(es) / 2.0);
    }
  }

  template <typename Es, typename Fn>
  void with_quaternion_input(const CoverageContext& ctx, const Es& es, Fn&& fn) const {
    if (ctx.quaternion_shape == ExprInputShape::RawAccessor) {
      std::forward<Fn>(fn)(quat()(es));
    } else {
      std::forward<Fn>(fn)(quat()(es) / 2.0);
    }
  }

  // --------------------------------------------------------------------------
  // Coverage sweep
  //
  // for_each_context<InputFamilies...> is the primary entry point for tests.
  // It sweeps all combinations of:
  //   - ExprInputShape for each listed InputFamily (Cartesian product)
  //   - EntityExprShape: OrdinaryEntity then ConnectedEntities
  //
  // Before each body invocation, output fields are re-initialized so that
  // earlier sub-cases cannot influence later ones.
  //
  // Body signature: void(const CoverageContext& ctx, auto es)
  // --------------------------------------------------------------------------

  template <typename... InputFamilies, typename Body>
  void for_each_context(std::string_view op_name, Body&& body) {
    CoverageContext ctx;
    auto run_entity_shapes = [&](CoverageContext& cfg) {
      run_for_each_entity_shape<InputFamilies...>(op_name, cfg, body);
    };
    sweep_input_shapes<InputFamilies...>(ctx, run_entity_shapes);
  }

  // --------------------------------------------------------------------------
  // Field assertion helpers
  // --------------------------------------------------------------------------

  template <typename ExpectedFunc>
  void expect_scalar_field_near(const std::string& case_name, const stk::mesh::FieldBase& field,
                                const stk::mesh::Selector& sel, ExpectedFunc fn) {
    expect_field_near<1>(case_name, field, sel, fn);
  }

  template <typename ExpectedFunc>
  void expect_vector3_field_near(const std::string& case_name, const stk::mesh::FieldBase& field,
                                 const stk::mesh::Selector& sel, ExpectedFunc fn) {
    expect_field_near<3>(case_name, field, sel, fn);
  }

  template <typename ExpectedFunc>
  void expect_matrix3_field_near(const std::string& case_name, const stk::mesh::FieldBase& field,
                                 const stk::mesh::Selector& sel, ExpectedFunc fn) {
    expect_field_near<9>(case_name, field, sel, fn);
  }

  template <typename ExpectedFunc>
  void expect_quaternion_field_near(const std::string& case_name, const stk::mesh::FieldBase& field,
                                    const stk::mesh::Selector& sel, ExpectedFunc fn) {
    expect_field_near<4>(case_name, field, sel, fn);
  }

  // --------------------------------------------------------------------------
  // Value-verification dispatch
  //
  // For OrdinaryEntity shapes, delegates to expect_field_near over the selector.
  // For ConnectedEntities shapes, syncs fields and checks the first connected
  // node of each owned selected element against the expected function.
  // --------------------------------------------------------------------------

  template <typename ExpectedFunc>
  void verify_scalar_output(const CoverageContext& ctx, const stk::mesh::FieldBase& field, ExpectedFunc fn) {
    if (ctx.entity_shape == EntityExprShape::OrdinaryEntity) {
      expect_field_near<1>(ctx.case_name, field, parts_.selected, fn);
    } else {
      verify_connected_entity_output<1>(ctx.case_name, field, fn);
    }
  }

  template <typename ExpectedFunc>
  void verify_vector3_output(const CoverageContext& ctx, const stk::mesh::FieldBase& field, ExpectedFunc fn) {
    if (ctx.entity_shape == EntityExprShape::OrdinaryEntity) {
      expect_field_near<3>(ctx.case_name, field, parts_.selected, fn);
    } else {
      verify_connected_entity_output<3>(ctx.case_name, field, fn);
    }
  }

  template <typename ExpectedFunc>
  void verify_matrix3_output(const CoverageContext& ctx, const stk::mesh::FieldBase& field, ExpectedFunc fn) {
    if (ctx.entity_shape == EntityExprShape::OrdinaryEntity) {
      expect_field_near<9>(ctx.case_name, field, parts_.selected, fn);
    } else {
      verify_connected_entity_output<9>(ctx.case_name, field, fn);
    }
  }

  template <typename ExpectedFunc>
  void verify_quaternion_output(const CoverageContext& ctx, const stk::mesh::FieldBase& field, ExpectedFunc fn) {
    if (ctx.entity_shape == EntityExprShape::OrdinaryEntity) {
      expect_field_near<4>(ctx.case_name, field, parts_.selected, fn);
    } else {
      verify_connected_entity_output<4>(ctx.case_name, field, fn);
    }
  }

  // --------------------------------------------------------------------------
  // Re-initialization helper (called before each body invocation)
  // --------------------------------------------------------------------------

  void reinitialize_output_fields() {
    // clang-format off
    set_field_values<1>(*fields_.out,         ExpectedValues::out);
    set_field_values<1>(*fields_.first_draw,  ExpectedValues::first_draw);
    set_field_values<1>(*fields_.second_draw, ExpectedValues::second_draw);
    set_field_values<3>(*fields_.vout,        ExpectedValues::vout);
    set_field_values<9>(*fields_.mout,        ExpectedValues::mout);
    set_field_values<4>(*fields_.qout,        ExpectedValues::qout);
    // clang-format on
  }

 private:
  // --------------------------------------------------------------------------
  // ConnectedEntity output verifier
  //
  // For each owned selected ELEMENT, obtains the first connected node and
  // compares actual field values against fn(coords) with EXPECT_NEAR.
  // The field must be allocated for every first node's bucket; if it is not,
  // that is a test-setup failure and is reported as such.
  // --------------------------------------------------------------------------

  template <size_t N, typename ExpectedFunc>
  void verify_connected_entity_output(const std::string& case_name, const stk::mesh::FieldBase& field,
                                      ExpectedFunc fn) {
    field.sync_to_host();
    coordinate_field_->sync_to_host();
    stk::mesh::for_each_entity_run(
        static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::ELEMENT_RANK, parts_.selected,
        [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity element) {
          const stk::mesh::Entity first_node = mesh.begin_nodes(element)[0];
          ASSERT_TRUE(stk::mesh::field_is_allocated_for_bucket(field, mesh.bucket(first_node)))
              << "case=" << case_name << ": field '" << field.name()
              << "' not allocated for bucket of first_node " << mesh.entity_key(first_node);
          const double* coords =
              static_cast<const double*>(stk::mesh::field_data(*coordinate_field_, first_node));
          const std::array<double, N> expected = fn(coords);
          const double* actual = static_cast<const double*>(stk::mesh::field_data(field, first_node));
          for (size_t i = 0; i < N; ++i) {
            EXPECT_NEAR(actual[i], expected[i], kTolerance)
                << "case=" << case_name << ", field=" << field.name()
                << ", first_node=" << mesh.entity_key(first_node) << ", component=" << i;
          }
        });
  }

  // --------------------------------------------------------------------------
  // Private mesh / field setup
  // --------------------------------------------------------------------------

  void setup_mesh() {
    MeshBuilder builder(MPI_COMM_WORLD);
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    builder.set_auto_aura_option(stk::mesh::BulkData::AUTO_AURA);
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
    builder.set_field_data_manager(std::make_unique<stk::mesh::DefaultFieldDataManager>(5));
#else
    builder.set_field_data_manager(new stk::mesh::DefaultFieldDataManager(5));
#endif
    meta_data_ = builder.create_meta_data();
    meta_data_->use_simple_fields();
    meta_data_->set_coordinate_field_name("coordinates");
    bulk_data_ = builder.create_bulk_data(meta_data_);
  }

  void declare_parts() {
    parts_.block1 = &meta().declare_part_with_topology("block_1", stk::topology::HEX_8);
    parts_.block2 = &meta().declare_part_with_topology("block_2", stk::topology::HEX_8);
    parts_.block3 = &meta().declare_part_with_topology("block_3", stk::topology::HEX_8);
    parts_.all_blocks = *parts_.block1 | *parts_.block2 | *parts_.block3;
    parts_.selected = *parts_.block1 - *parts_.block2;
    parts_.unselected = !parts_.selected;
  }

  void declare_fields() {
    coordinate_field_ = &meta().declare_field<double>(stk::topology::NODE_RANK, "coordinates");
    // clang-format off
    fields_.r           = declare_node_field("r",           1);
    fields_.q           = declare_node_field("q",           1);
    fields_.out         = declare_node_field("out",         1);
    fields_.aux         = declare_node_field("aux",         1);
    fields_.seed        = declare_node_field("seed",        1);
    fields_.counter     = declare_node_field("counter",     1);
    fields_.first_draw  = declare_node_field("first_draw",  1);
    fields_.second_draw = declare_node_field("second_draw", 1);
    fields_.vel         = declare_node_field("vel",         3);
    fields_.force       = declare_node_field("force",       3);
    fields_.vout        = declare_node_field("vout",        3);
    fields_.mat         = declare_node_field("mat",         9);
    fields_.mat_b       = declare_node_field("mat_b",       9);
    fields_.mout        = declare_node_field("mout",        9);
    fields_.quat        = declare_node_field("quat",        4);
    fields_.quat_b      = declare_node_field("quat_b",      4);
    fields_.qout        = declare_node_field("qout",        4);
    fields_.omega       = declare_node_field("omega",       3);
    // clang-format on
  }

  DoubleField* declare_node_field(const std::string& name, unsigned num_components) {
    DoubleField& field = meta().declare_field<double>(stk::topology::NODE_RANK, name);
    stk::mesh::put_field_on_mesh(field, *parts_.block1, num_components, kInitialValue);
    stk::mesh::put_field_on_mesh(field, *parts_.block2, num_components, kInitialValue);
    stk::mesh::put_field_on_mesh(field, *parts_.block3, num_components, kInitialValue);
    return &field;
  }

  void build_five_hex_mesh() {
    const int nranks = bulk().parallel_size();
    ASSERT_TRUE(nranks == 1 || nranks == 2) << "AccessorExprCoverageFixture requires 1 or 2 MPI ranks.";

    std::string mesh_desc;
    if (nranks == 1) {
      mesh_desc =
          "textmesh:"
          "0,1,HEX_8,1,2,3,4,5,6,7,8,block_1\n"
          "0,2,HEX_8,5,6,7,8,9,10,11,12,block_1\n"
          "0,3,HEX_8,9,13,14,15,16,17,18,19,block_2\n"
          "0,4,HEX_8,9,20,21,22,23,24,25,26,block_2\n"
          "0,5,HEX_8,9,27,28,29,30,31,32,33,block_3";
    } else {
      mesh_desc =
          "textmesh:"
          "0,1,HEX_8,1,2,3,4,5,6,7,8,block_1\n"
          "1,2,HEX_8,5,6,7,8,9,10,11,12,block_1\n"
          "0,3,HEX_8,9,13,14,15,16,17,18,19,block_2\n"
          "1,4,HEX_8,9,20,21,22,23,24,25,26,block_2\n"
          "0,5,HEX_8,9,27,28,29,30,31,32,33,block_3";
    }
    stk::io::fill_mesh_with_auto_decomp(mesh_desc, bulk());
    validate_five_hex_mesh();
  }

  void validate_five_hex_mesh() {
    // Node connectivity expected for each of the five hexes.
    const std::array<std::vector<int>, 5> expected_nodes = {
        std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}, std::vector<int>{5, 6, 7, 8, 9, 10, 11, 12},
        std::vector<int>{9, 13, 14, 15, 16, 17, 18, 19}, std::vector<int>{9, 20, 21, 22, 23, 24, 25, 26},
        std::vector<int>{9, 27, 28, 29, 30, 31, 32, 33}};
    const std::array<stk::mesh::Part*, 5> expected_parts = {parts_.block1, parts_.block1, parts_.block2, parts_.block2,
                                                            parts_.block3};

    for (size_t i = 0; i < 5; ++i) {
      const stk::mesh::Entity hex = bulk().get_entity(stk::topology::ELEMENT_RANK, i + 1);
      if (!bulk().is_valid(hex)) continue;  // owned by other rank in 2-rank run
      EXPECT_TRUE(bulk().bucket(hex).member(*expected_parts[i]));
      const stk::mesh::Entity* hex_nodes = bulk().begin_nodes(hex);
      for (size_t n = 0; n < expected_nodes[i].size(); ++n) {
        EXPECT_EQ(bulk().identifier(hex_nodes[n]), (unsigned)expected_nodes[i][n]);
      }
    }
  }

  void initialize_fields() {
    initialize_coordinates();
    // clang-format off
    set_field_values<1>(*fields_.r,           ExpectedValues::r);
    set_field_values<1>(*fields_.q,           ExpectedValues::q);
    set_field_values<1>(*fields_.out,         ExpectedValues::out);
    set_field_values<1>(*fields_.aux,         ExpectedValues::aux);
    set_field_values<1>(*fields_.seed,        ExpectedValues::seed);
    set_field_values<1>(*fields_.counter,     ExpectedValues::counter);
    set_field_values<1>(*fields_.first_draw,  ExpectedValues::first_draw);
    set_field_values<1>(*fields_.second_draw, ExpectedValues::second_draw);
    set_field_values<3>(*fields_.vel,         ExpectedValues::vel);
    set_field_values<3>(*fields_.force,       ExpectedValues::force);
    set_field_values<3>(*fields_.vout,        ExpectedValues::vout);
    set_field_values<9>(*fields_.mat,         ExpectedValues::mat);
    set_field_values<9>(*fields_.mat_b,       ExpectedValues::mat_b);
    set_field_values<9>(*fields_.mout,        ExpectedValues::mout);
    set_field_values<4>(*fields_.quat,        ExpectedValues::quat);
    set_field_values<4>(*fields_.quat_b,      ExpectedValues::quat_b);
    set_field_values<4>(*fields_.qout,        ExpectedValues::qout);
    set_field_values<3>(*fields_.omega,       ExpectedValues::omega);
    // clang-format on
  }

  void initialize_coordinates() {
    coordinate_field_->clear_host_sync_state();
    stk::mesh::for_each_entity_run(
        static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, meta().universal_part(),
        [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
          double* coords = static_cast<double*>(stk::mesh::field_data(*coordinate_field_, entity));
          const double id = static_cast<double>(mesh.identifier(entity));
          coords[0] = 0.01 * id;
          coords[1] = 0.02 * id + 0.25;
          coords[2] = 0.03 * id + 0.5;
        });
    coordinate_field_->modify_on_host();
  }

  template <size_t N, typename ExpectedFunc>
  void set_field_values(DoubleField& field, ExpectedFunc fn) {
    field.clear_sync_state();
    coordinate_field_->sync_to_host();
    stk::mesh::for_each_entity_run(
        static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts_.all_blocks,
        [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
          const double* coords = static_cast<const double*>(stk::mesh::field_data(*coordinate_field_, entity));
          const std::array<double, N> vals = fn(coords);
          double* data = static_cast<double*>(stk::mesh::field_data(field, entity));
          for (size_t i = 0; i < N; ++i) data[i] = vals[i];
        });
    field.modify_on_host();
  }

  template <size_t N, typename ExpectedFunc>
  void expect_field_near(const std::string& case_name, const stk::mesh::FieldBase& field,
                         const stk::mesh::Selector& sel, ExpectedFunc fn) {
    field.sync_to_host();
    coordinate_field_->sync_to_host();
    stk::mesh::for_each_entity_run(
        static_cast<const stk::mesh::BulkData&>(bulk()), field.entity_rank(), sel,
        [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
          const double* coords = static_cast<const double*>(stk::mesh::field_data(*coordinate_field_, entity));
          const std::array<double, N> expected = fn(coords);
          const double* actual = static_cast<const double*>(stk::mesh::field_data(field, entity));
          for (size_t i = 0; i < N; ++i) {
            EXPECT_NEAR(actual[i], expected[i], kTolerance)
                << "case=" << case_name << ", field=" << field.name() << ", entity=" << mesh.entity_key(entity)
                << ", component=" << i;
          }
        });
  }

  // --------------------------------------------------------------------------
  // Private sweep implementation
  // --------------------------------------------------------------------------

  // Build the case name from operation name, entity shape, and all requested input shapes.
  template <typename... InputFamilies>
  std::string make_case_name(std::string_view op_name, const CoverageContext& ctx) const {
    std::string name(op_name);
    name += " / ";
    name += to_cstr(ctx.entity_shape);
    ((name += " / ", name += shape_fragment<InputFamilies>(ctx)), ...);
    return name;
  }

  // For each requested InputFamily, sweep RawAccessor then IntermediateMathExpr.
  // Base case: no more families — just call body.
  template <typename Body>
  void sweep_input_shapes(CoverageContext& ctx, Body& body) {
    body(ctx);
  }

  template <typename FirstFamily, typename... RestFamilies, typename Body>
  void sweep_input_shapes(CoverageContext& ctx, Body& body) {
    for (ExprInputShape shape : kAllInputShapes) {
      set_context_shape<FirstFamily>(ctx, shape);
      sweep_input_shapes<RestFamilies...>(ctx, body);
    }
  }

  // Run body for OrdinaryEntity then ConnectedEntities, reinitializing output
  // fields before each call.
  template <typename... InputFamilies, typename Body>
  void run_for_each_entity_shape(std::string_view op_name, const CoverageContext& base_ctx, Body& body) {
    // Ordinary entity: full value verification.
    {
      CoverageContext ctx = base_ctx;
      ctx.entity_shape = EntityExprShape::OrdinaryEntity;
      ctx.case_name = make_case_name<InputFamilies...>(op_name, ctx);
      reinitialize_output_fields();
      body(ctx, make_node_entities());
    }
    // Connected entities: value verification on first connected node.
    {
      CoverageContext ctx = base_ctx;
      ctx.entity_shape = EntityExprShape::ConnectedEntities;
      ctx.case_name = make_case_name<InputFamilies...>(op_name, ctx);
      reinitialize_output_fields();
      auto elements = make_element_entities();
      auto connected_nodes = elements.get_connectivity(stk::topology::NODE_RANK);
      body(ctx, connected_nodes[0]);
    }
  }

  // --------------------------------------------------------------------------
  // Data members
  // --------------------------------------------------------------------------

  std::shared_ptr<MetaData> meta_data_;
  std::unique_ptr<BulkData> bulk_data_;
  DoubleField* coordinate_field_ = nullptr;
  Parts parts_;
  Fields fields_;
};

// =============================================================================
// Section 3 — Scalar operation coverage
//
// Coverage matrix:
//
//   Operation family        Expression
//   ----------------------  -------------------------
//   left scalar multiply    2.0 * tmp
//   right scalar multiply   tmp * 2.0
//   left scalar divide      2.0 / tmp
//   right scalar divide     tmp / 2.0
//   left scalar add         2.0 + tmp
//   right scalar add        tmp + 2.0
//   left scalar subtract    2.0 - tmp
//   right scalar subtract   tmp - 2.0
//   field add               tmp + q()(es)
//   field subtract          tmp - q()(es)
//   field multiply          tmp * q()(es)
//   field divide            tmp / q()(es)
//
// Each TEST_F sweeps:
//   scalar_shape  in {RawAccessor, IntermediateMathExpr}
//   entity_shape  in {OrdinaryEntity, ConnectedEntities}
//
// with_scalar_input dispatches to either r()(es) (raw accessor) or r()(es)/2.0
// (intermediate DivExpr) depending on ctx.scalar_shape, so the expected value is:
//   t = r_val / divisor   (divisor 1.0 raw, 2.0 intermediate)
//
// The macro SCALAR_OP_TEST is defined locally and undef'd at end of section.
// It may NOT be used for expressions whose ExprBody or ExpVal contain unprotected
// commas — use parentheses or named temporaries in those cases.
//
// Variables injected into ExprBody and ExpVal by the macro:
//   tmp     — r()(es) raw accessor (RawAccessor) or r()(es)/2.0 DivExpr (Intermediate)
//   es      — entity expression (type varies by entity-shape axis)
//   q()(es) — scalar q field accessor applied to es
//   t       — host-side double: r_val / divisor
//   r_val   — host-side double: ExpectedValues::r(c)[0]
//   q_val   — host-side double: ExpectedValues::q(c)[0]  (may be unused for non-field ops)
// =============================================================================

#define SCALAR_OP_TEST(CaseName, OpNameStr, ExprBody, ExpVal)                                               \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                           \
    for_each_context<Scalar>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                          \
      with_scalar_input(ctx, es, [&](auto tmp) {                                                            \
        out()(es) = ExprBody;                                                                               \
        verify_scalar_output(ctx, *fields().out, [&](const double* c) -> std::array<double, 1> {           \
          const double r_val = ExpectedValues::r(c)[0];                                                     \
          const double divisor = (ctx.scalar_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;             \
          const double t = r_val / divisor;                                                                 \
          [[maybe_unused]] const double q_val = ExpectedValues::q(c)[0];                                    \
          return {ExpVal};                                                                                  \
        });                                                                                                 \
      });                                                                                                   \
    });                                                                                                     \
  }

// ---- Left- and right-hand scalar arithmetic ----
// clang-format off
SCALAR_OP_TEST(ScalarOp_LeftScalarMultiply,  "scalar / left scalar multiply",  2.0 * tmp,  2.0 * t)
SCALAR_OP_TEST(ScalarOp_RightScalarMultiply, "scalar / right scalar multiply", tmp * 2.0,  t * 2.0)
SCALAR_OP_TEST(ScalarOp_LeftScalarDivide,    "scalar / left scalar divide",    2.0 / tmp,  2.0 / t)
SCALAR_OP_TEST(ScalarOp_RightScalarDivide,   "scalar / right scalar divide",   tmp / 2.0,  t / 2.0)
SCALAR_OP_TEST(ScalarOp_LeftScalarAdd,       "scalar / left scalar add",       2.0 + tmp,  2.0 + t)
SCALAR_OP_TEST(ScalarOp_RightScalarAdd,      "scalar / right scalar add",      tmp + 2.0,  t + 2.0)
SCALAR_OP_TEST(ScalarOp_LeftScalarSubtract,  "scalar / left scalar subtract",  2.0 - tmp,  2.0 - t)
SCALAR_OP_TEST(ScalarOp_RightScalarSubtract, "scalar / right scalar subtract", tmp - 2.0,  t - 2.0)
// clang-format on

// ---- Field-field scalar arithmetic ----
// q()(es) is the second scalar field; q_val > 3 on all nodes, so q-division is safe.
// clang-format off
SCALAR_OP_TEST(ScalarOp_FieldAdd,      "scalar / field add",      tmp + q()(es), t + q_val)
SCALAR_OP_TEST(ScalarOp_FieldSubtract, "scalar / field subtract", tmp - q()(es), t - q_val)
SCALAR_OP_TEST(ScalarOp_FieldMultiply, "scalar / field multiply", tmp * q()(es), t * q_val)
SCALAR_OP_TEST(ScalarOp_FieldDivide,   "scalar / field divide",   tmp / q()(es), t / q_val)
// clang-format on

#undef SCALAR_OP_TEST

// =============================================================================
// Section 4 — Scalar builtin coverage
//
// tmp = r()(es) raw accessor (RawAccessor) or r()(es)/2.0 DivExpr (Intermediate)
// r_val ∈ (2.07, 2.56) on all selected nodes, so t > 1 for both divisors.
// Domain notes:
//   sqrt, log, exp, abs, sin, cos, tan, atan: safe for t > 1
//   tan: t ∈ (1.04, 2.56), well away from π/2 ≈ 1.5708
//   asin, acos: require t ∈ [-1,1]; use d_safe = 3.5 (raw) or 7.0 (intermediate)
//               giving t ∈ (0.30, 0.73) ⊂ [-1, 1]
// =============================================================================

#define SCALAR_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, ExpVal)                                          \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                           \
    for_each_context<Scalar>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                          \
      with_scalar_input(ctx, es, [&](auto tmp) {                                                            \
        out()(es) = ExprBody;                                                                               \
        verify_scalar_output(ctx, *fields().out, [&](const double* c) -> std::array<double, 1> {           \
          const double r_val = ExpectedValues::r(c)[0];                                                     \
          const double divisor = (ctx.scalar_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;             \
          const double t = r_val / divisor;                                                                 \
          return {ExpVal};                                                                                  \
        });                                                                                                 \
      });                                                                                                   \
    });                                                                                                     \
  }

// asin/acos require inputs in [-1,1]; our r field has values > 2, so a raw accessor
// cannot be passed to these functions.  Both shape variants use a DivExpr with a
// domain-safe divisor (3.5 raw / 7.0 intermediate) — the raw-vs-intermediate axis
// here exercises different scaling rather than different expression types.
#define SCALAR_BUILTIN_DOMAIN_SAFE_TEST(CaseName, OpNameStr, ExprBody, ExpVal)               \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                            \
    for_each_context<Scalar>(OpNameStr, [&](const CoverageContext& ctx, auto es) {           \
      const double d_safe = (ctx.scalar_shape == ExprInputShape::RawAccessor) ? 3.5 : 7.0;  \
      auto tmp = r()(es) / d_safe;                                                           \
      out()(es) = ExprBody;                                                                  \
      verify_scalar_output(ctx, *fields().out, [&](const double* c) -> std::array<double, 1> { \
        const double r_val = ExpectedValues::r(c)[0];                                         \
        const double t = r_val / d_safe;                                                      \
        return {ExpVal};                                                                      \
      });                                                                                     \
    });                                                                                       \
  }

// clang-format on
SCALAR_BUILTIN_TEST(ScalarBuiltin_Abs, "scalar / abs", abs(tmp), std::abs(t))
SCALAR_BUILTIN_TEST(ScalarBuiltin_Sqrt, "scalar / sqrt", sqrt(tmp), std::sqrt(t))
SCALAR_BUILTIN_TEST(ScalarBuiltin_Exp, "scalar / exp", exp(tmp), std::exp(t))
SCALAR_BUILTIN_TEST(ScalarBuiltin_Log, "scalar / log", log(tmp), std::log(t))
SCALAR_BUILTIN_TEST(ScalarBuiltin_Sin, "scalar / sin", sin(tmp), std::sin(t))
SCALAR_BUILTIN_TEST(ScalarBuiltin_Cos, "scalar / cos", cos(tmp), std::cos(t))
SCALAR_BUILTIN_TEST(ScalarBuiltin_Tan, "scalar / tan", tan(tmp), std::tan(t))
SCALAR_BUILTIN_TEST(ScalarBuiltin_Atan, "scalar / atan", atan(tmp), std::atan(t))
SCALAR_BUILTIN_DOMAIN_SAFE_TEST(ScalarBuiltin_Asin, "scalar / asin", asin(tmp), std::asin(t))
SCALAR_BUILTIN_DOMAIN_SAFE_TEST(ScalarBuiltin_Acos, "scalar / acos", acos(tmp), std::acos(t))
// clang-format off

#undef SCALAR_BUILTIN_TEST
#undef SCALAR_BUILTIN_DOMAIN_SAFE_TEST

// =============================================================================
// Section 5 — Vector operation coverage
//
// Coverage matrix:
//
//   Operation family            Expression
//   --------------------------  --------------------------------
//   scalar left multiply        2.0 * vtmp
//   scalar right multiply       vtmp * 2.0
//   scalar right divide         vtmp / 2.0
//   scalar expression multiply  vtmp * stmp      (cross-axis, see below)
//   expression scalar multiply  stmp * vtmp      (cross-axis, see below)
//   expression scalar divide    vtmp / stmp      (cross-axis, see below)
//   vector add                  vtmp + force(es)
//   vector subtract             vtmp - force(es)
//   component multiply          elementwise_mul(vtmp, force(es))  — see Section 6
//   component divide            elementwise_div(vtmp, force(es))  — see Section 6
//
// Note: elementwise_mul and elementwise_div appear under Vector Operation Coverage
// in the spec but are categorized as builtins in Section 6 of this file, consistent
// with the matrix treatment of those same operations.
//
// Each VECTOR_OP_TEST sweeps:
//   vector_shape in {RawAccessor, IntermediateMathExpr}
//   entity_shape in {OrdinaryEntity, ConnectedEntities}
//
// The three cross-axis tests (vtmp*stmp, stmp*vtmp, vtmp/stmp) additionally sweep
// scalar_shape independently; they use VECTOR_SCALAR_OP_TEST which calls
// for_each_context<Vector, Scalar>.
//
// Input values:
//   vtmp = vel()(es) / d_v    vel_val[i] = {1+c0, 2+c1, 3+c2}; all components > 1
//   force()(es)               force_val[i] = {4+c1, 5+c2, 6+c0}; all components > 4
//   stmp = r()(es) / d_s      r_val = 2+c0+c1+c2 ∈ (2.07, 2.56); safe as divisor
//
// Variables injected by VECTOR_OP_TEST into ExprBody and expected lambda:
//   tmp_v     — vel()(es) raw accessor (RawAccessor) or vel()(es)/2.0 DivExpr (Intermediate)
//   es        — entity expression
//   force()(es) — vector force field accessor
//   tv0/tv1/tv2 — vel_val[i] / d_v  (host-side)
//   fv0/fv1/fv2 — force_val[i]      (host-side, raw)
// =============================================================================

#define VECTOR_OP_TEST(CaseName, OpNameStr, ExprBody, Comp0, Comp1, Comp2)                         \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                   \
    for_each_context<Vector>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                  \
      with_vector_input(ctx, es, [&](auto tmp_v) {                                                  \
        vout()(es) = ExprBody;                                                                      \
        verify_vector3_output(ctx, *fields().vout, [&](const double* c) -> std::array<double, 3> { \
          const auto vel_val   = ExpectedValues::vel(c);                                             \
          const auto force_val = ExpectedValues::force(c);                                           \
          const double d  = (ctx.vector_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;         \
          [[maybe_unused]] const double tv0 = vel_val[0] / d;                                       \
          [[maybe_unused]] const double tv1 = vel_val[1] / d;                                       \
          [[maybe_unused]] const double tv2 = vel_val[2] / d;                                       \
          [[maybe_unused]] const double fv0 = force_val[0];                                         \
          [[maybe_unused]] const double fv1 = force_val[1];                                         \
          [[maybe_unused]] const double fv2 = force_val[2];                                         \
          return {Comp0, Comp1, Comp2};                                                             \
        });                                                                                         \
      });                                                                                           \
    });                                                                                             \
  }

// clang-format off
VECTOR_OP_TEST(VectorOp_LeftScalarMultiply,  "vector / left scalar multiply",  (2.0 * tmp_v),         2.0*tv0, 2.0*tv1, 2.0*tv2)
VECTOR_OP_TEST(VectorOp_RightScalarMultiply, "vector / right scalar multiply", (tmp_v * 2.0),         tv0*2.0, tv1*2.0, tv2*2.0)
VECTOR_OP_TEST(VectorOp_RightScalarDivide,   "vector / right scalar divide",   (tmp_v / 2.0),         tv0/2.0, tv1/2.0, tv2/2.0)
VECTOR_OP_TEST(VectorOp_FieldAdd,            "vector / field add",             (tmp_v + force()(es)), tv0+fv0, tv1+fv1, tv2+fv2)
VECTOR_OP_TEST(VectorOp_FieldSubtract,       "vector / field subtract",        (tmp_v - force()(es)), tv0-fv0, tv1-fv1, tv2-fv2)
// clang-format on

#undef VECTOR_OP_TEST

// =============================================================================
// Section 5 (continued) — Vector × Scalar cross-axis operation coverage
//
// These three operations take one vector expression and one scalar expression
// as operands.  Both input axes are swept independently via for_each_context<Vector, Scalar>.
//
//   Operation              Expression         Axes swept
//   ---------------------  -----------------  -------------------------
//   vector * scalar_expr   tmp_v * tmp_s      Vector ∈ {raw, interm}
//   scalar_expr * vector   tmp_s * tmp_v        Scalar ∈ {raw, interm}
//   vector / scalar_expr   tmp_v / tmp_s        entity ∈ {ord, conn}
//
// Variables injected by VECTOR_SCALAR_OP_TEST:
//   tmp_v — vel()(es) raw or vel()(es)/2.0 DivExpr depending on vector_shape
//   tmp_s — r()(es)   raw or r()(es)/2.0   DivExpr depending on scalar_shape
//   tv0/tv1/tv2 — vel_val[i] / vd
//   ts          — r_val / sd
// =============================================================================

#define VECTOR_SCALAR_OP_TEST(CaseName, OpNameStr, ExprBody, Comp0, Comp1, Comp2)                                \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                                \
    for_each_context<Vector, Scalar>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                       \
      with_vector_input(ctx, es, [&](auto tmp_v) {                                                               \
        with_scalar_input(ctx, es, [&](auto tmp_s) {                                                             \
          vout()(es) = ExprBody;                                                                                 \
          verify_vector3_output(ctx, *fields().vout, [&](const double* c) -> std::array<double, 3> {             \
            const auto vel_val = ExpectedValues::vel(c);                                                         \
            const double vd = (ctx.vector_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;                    \
            const double sd = (ctx.scalar_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;                    \
            const double tv0 = vel_val[0] / vd;                                                                 \
            const double tv1 = vel_val[1] / vd;                                                                 \
            const double tv2 = vel_val[2] / vd;                                                                 \
            const double ts = ExpectedValues::r(c)[0] / sd;                                                     \
            return {Comp0, Comp1, Comp2};                                                                        \
          });                                                                                                    \
        });                                                                                                      \
      });                                                                                                        \
    });                                                                                                          \
  }

// clang-format off
VECTOR_SCALAR_OP_TEST(VectorOp_VectorScalarExprMultiply, "vector / vector * scalar_expr", (tmp_v * tmp_s), tv0*ts, tv1*ts, tv2*ts)
VECTOR_SCALAR_OP_TEST(VectorOp_ScalarExprVectorMultiply, "vector / scalar_expr * vector", (tmp_s * tmp_v), ts*tv0, ts*tv1, ts*tv2)
VECTOR_SCALAR_OP_TEST(VectorOp_VectorScalarExprDivide,   "vector / vector / scalar_expr", (tmp_v / tmp_s), tv0/ts, tv1/ts, tv2/ts)
// clang-format on

#undef VECTOR_SCALAR_OP_TEST

// =============================================================================
// Section 6 — Vector builtin coverage
//
// Coverage matrix:
//
//   Function             Output   Expression
//   -------------------  -------  ------------------------------
//   sum                  scalar   sum(vtmp)
//   product              scalar   product(vtmp)
//   min                  scalar   min(vtmp)
//   max                  scalar   max(vtmp)
//   mean                 scalar   mean(vtmp)
//   norm                 scalar   norm(vtmp)
//   norm_squared         scalar   norm_squared(vtmp)
//   dot                  scalar   dot(vtmp, force(es))
//   copy                 vector   copy(vtmp)
//   cross                vector   cross(vtmp, force(es))
//   elementwise_mul      vector   elementwise_mul(vtmp, force(es))
//   elementwise_div      vector   elementwise_div(vtmp, force(es))
//   variance             scalar   variance(vtmp)
//   stddev               scalar   stddev(vtmp)
//   one_norm             scalar   one_norm(vtmp)
//   two_norm             scalar   two_norm(vtmp)
//   inf_norm             scalar   inf_norm(vtmp)
//   infinity_norm        scalar   infinity_norm(vtmp)
//   two_norm_squared     scalar   two_norm_squared(vtmp)
//   minor_angle          scalar   minor_angle(vtmp, force(es))
//   major_angle          scalar   major_angle(vtmp, force(es))
//   outer_product        matrix   outer_product(vtmp, force(es))  → mout
//
// Each test sweeps:
//   vector_shape in {RawAccessor, IntermediateMathExpr}
//   entity_shape in {OrdinaryEntity, ConnectedEntities}
//
// Input values:
//   vtmp = vel()(es) / d    vel_val[i] = {1+c0, 2+c1, 3+c2}; tv0 < tv1 < tv2 > 0
//   force()(es)             force_val[i] = {4+c1, 5+c2, 6+c0}; all > 4, safe for division
//
// Norms (all tv_i > 0 so abs = identity):
//   one_norm(v) = tv0 + tv1 + tv2
//   two_norm(v) = sqrt(tv0²+tv1²+tv2²) = norm(v)
//   inf_norm(v) = infinity_norm(v) = max(tv0,tv1,tv2) = tv2
//
// variance = population variance (divide by N=3):
//   mean = (tv0+tv1+tv2)/3;  variance = sum((tv_i-mean)²)/3
//
// minor_angle(u,v) = acos(|dot(u,v)| / (|u|*|v|)) ∈ [0, π/2]
// major_angle(u,v) = π − minor_angle(u,v)
//
// Scalar-output builtins write to out()(es); vector-output to vout()(es).
// outer_product writes to mout()(es): result[i,j] = u[i]*v[j].
// cross(u,v) = {u1*v2 - u2*v1, u2*v0 - u0*v2, u0*v1 - u1*v0}
// =============================================================================

#define VECTOR_SCALAR_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, ExpVal)                                          \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                                  \
    for_each_context<Vector>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                                 \
      with_vector_input(ctx, es, [&](auto tmp_v) {                                                                 \
        out()(es) = ExprBody;                                                                                      \
        verify_scalar_output(ctx, *fields().out, [&](const double* c) -> std::array<double, 1> {                  \
          const auto vel_val = ExpectedValues::vel(c);                                                             \
          const auto force_val = ExpectedValues::force(c);                                                         \
          const double d = (ctx.vector_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;                         \
          [[maybe_unused]] const double tv0 = vel_val[0] / d;                                                      \
          [[maybe_unused]] const double tv1 = vel_val[1] / d;                                                      \
          [[maybe_unused]] const double tv2 = vel_val[2] / d;                                                      \
          [[maybe_unused]] const double fv0 = force_val[0];                                                        \
          [[maybe_unused]] const double fv1 = force_val[1];                                                        \
          [[maybe_unused]] const double fv2 = force_val[2];                                                        \
          return {ExpVal};                                                                                          \
        });                                                                                                         \
      });                                                                                                           \
    });                                                                                                             \
  }

#define VECTOR_VECTOR_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, Comp0, Comp1, Comp2)                             \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                                   \
    for_each_context<Vector>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                                  \
      with_vector_input(ctx, es, [&](auto tmp_v) {                                                                  \
        vout()(es) = ExprBody;                                                                                      \
        verify_vector3_output(ctx, *fields().vout, [&](const double* c) -> std::array<double, 3> {                 \
          const auto vel_val = ExpectedValues::vel(c);                                                              \
          const auto force_val = ExpectedValues::force(c);                                                          \
          const double d = (ctx.vector_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;                          \
          [[maybe_unused]] const double tv0 = vel_val[0] / d;                                                       \
          [[maybe_unused]] const double tv1 = vel_val[1] / d;                                                       \
          [[maybe_unused]] const double tv2 = vel_val[2] / d;                                                       \
          [[maybe_unused]] const double fv0 = force_val[0];                                                         \
          [[maybe_unused]] const double fv1 = force_val[1];                                                         \
          [[maybe_unused]] const double fv2 = force_val[2];                                                         \
          return {Comp0, Comp1, Comp2};                                                                             \
        });                                                                                                         \
      });                                                                                                           \
    });                                                                                                             \
  }

#define VECTOR_MATRIX_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, ExpMat)                             \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Vector>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                     \
      with_vector_input(ctx, es, [&](auto tmp_v) {                                                     \
        mout()(es) = ExprBody;                                                                         \
        verify_matrix3_output(ctx, *fields().mout, [&](const double* c) -> std::array<double, 9> {    \
          const auto vel_val   = ExpectedValues::vel(c);                                               \
          const auto force_val = ExpectedValues::force(c);                                             \
          const double d = (ctx.vector_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;             \
          [[maybe_unused]] const double tv0 = vel_val[0] / d;                                         \
          [[maybe_unused]] const double tv1 = vel_val[1] / d;                                         \
          [[maybe_unused]] const double tv2 = vel_val[2] / d;                                         \
          [[maybe_unused]] const double fv0 = force_val[0];                                           \
          [[maybe_unused]] const double fv1 = force_val[1];                                           \
          [[maybe_unused]] const double fv2 = force_val[2];                                           \
          return ExpMat;                                                                               \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

// ---- Scalar-output builtins ----
// clang-format off
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Sum,            "vector / sum",             sum(tmp_v),                          tv0 + tv1 + tv2)
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Product,        "vector / product",         product(tmp_v),                      tv0 * tv1 * tv2)
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Min,            "vector / min",             min(tmp_v),                          tv0)
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Max,            "vector / max",             max(tmp_v),                          tv2)
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Mean,           "vector / mean",            mean(tmp_v),                         (tv0 + tv1 + tv2) / 3.0)
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Norm,           "vector / norm",            norm(tmp_v),                         std::sqrt(tv0*tv0 + tv1*tv1 + tv2*tv2))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_NormSquared,    "vector / norm_squared",    norm_squared(tmp_v),                 tv0*tv0 + tv1*tv1 + tv2*tv2)
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Dot,            "vector / dot",             dot(tmp_v, force()(es)),             tv0*fv0 + tv1*fv1 + tv2*fv2)
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Variance,       "vector / variance",        variance(tmp_v),                     vec3_variance(std::array<double, 3>{tv0, tv1, tv2}))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_Stddev,         "vector / stddev",          stddev(tmp_v),                       vec3_stddev(std::array<double, 3>{tv0, tv1, tv2}))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_OneNorm,        "vector / one_norm",        one_norm(tmp_v),                     vec3_one_norm(std::array<double, 3>{tv0, tv1, tv2}))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_TwoNorm,        "vector / two_norm",        two_norm(tmp_v),                     std::sqrt(tv0*tv0 + tv1*tv1 + tv2*tv2))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_InfNorm,        "vector / inf_norm",        inf_norm(tmp_v),                     vec3_inf_norm(std::array<double, 3>{tv0, tv1, tv2}))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_InfinityNorm,   "vector / infinity_norm",   infinity_norm(tmp_v),                vec3_inf_norm(std::array<double, 3>{tv0, tv1, tv2}))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_TwoNormSquared, "vector / two_norm_squared",two_norm_squared(tmp_v),             vec3_two_norm_squared(std::array<double, 3>{tv0, tv1, tv2}))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_MinorAngle,     "vector / minor_angle",     minor_angle(tmp_v, force()(es)),     vec3_minor_angle(std::array<double, 3>{tv0, tv1, tv2}, force_val))
VECTOR_SCALAR_BUILTIN_TEST(VectorBuiltin_MajorAngle,     "vector / major_angle",     major_angle(tmp_v, force()(es)),     vec3_major_angle(std::array<double, 3>{tv0, tv1, tv2}, force_val))

// ---- Vector-output builtins ----
VECTOR_VECTOR_BUILTIN_TEST(VectorBuiltin_Copy,           "vector / copy",            copy(tmp_v),                         tv0, tv1, tv2)
VECTOR_VECTOR_BUILTIN_TEST(VectorBuiltin_Cross,          "vector / cross",           cross(tmp_v, force()(es)),           tv1*fv2-tv2*fv1, tv2*fv0-tv0*fv2, tv0*fv1-tv1*fv0)
VECTOR_VECTOR_BUILTIN_TEST(VectorBuiltin_ElementwiseMul, "vector / elementwise_mul", elementwise_mul(tmp_v, force()(es)), tv0*fv0, tv1*fv1, tv2*fv2)
VECTOR_VECTOR_BUILTIN_TEST(VectorBuiltin_ElementwiseDiv, "vector / elementwise_div", elementwise_div(tmp_v, force()(es)), tv0/fv0, tv1/fv1, tv2/fv2)

// ---- Matrix-output builtins ----
// ExpMat must be a single expression of type std::array<double,9>.
// Commas inside vec3_outer_product(...) are protected by its parentheses.
VECTOR_MATRIX_BUILTIN_TEST(VectorBuiltin_OuterProduct,   "vector / outer_product",   outer_product(tmp_v, force()(es)),   vec3_outer_product(std::array<double, 3>{tv0, tv1, tv2}, force_val))
// clang-format on

#undef VECTOR_SCALAR_BUILTIN_TEST
#undef VECTOR_VECTOR_BUILTIN_TEST
#undef VECTOR_MATRIX_BUILTIN_TEST

// =============================================================================
// Section 7 — Matrix operation coverage
//
// Coverage matrix:
//
//   Operation family        Expression               Output
//   ----------------------  -----------------------  ------
//   scalar left multiply    2.0 * mtmp               mout
//   scalar right multiply   mtmp * 2.0               mout
//   scalar right divide     mtmp / 2.0               mout
//   matrix add              mtmp + mat_b(es)          mout
//   matrix subtract         mtmp - mat_b(es)          mout
//   matrix multiply         mtmp * mat_b(es)          mout
//   matrix-vector multiply  mtmp * vtmp              vout
//   vector-matrix multiply  vtmp * mtmp              vout
//   component multiply      elementwise_mul(mtmp, mat_b(es))  — see Section 8
//   component divide        elementwise_div(mtmp, mat_b(es))  — see Section 8
//
// Note: elementwise_mul and elementwise_div appear under Matrix Operation Coverage
// in the spec but are categorized as builtins in Section 8 of this file.
//
// Each test sweeps:
//   matrix_shape in {RawAccessor, IntermediateMathExpr}
//   entity_shape in {OrdinaryEntity, ConnectedEntities}
//
// Input values:
//   mtmp = mat()(es) / d     mat is diagonally dominant (det > 0) and has all-positive
//                            elements; safe for division and inversion
//   vtmp = vel()(es)         vel_val[i] = {1+c0, 2+c1, 3+c2}; used raw (not through vtmp)
//   mat_b()(es)              second matrix field; also all-positive, diagonally dominant
//
// Matrix-vector: mtmp * vtmp → mat·vel → vout   (row · column)
// Vector-matrix: vtmp * mtmp → mat^T · vel → vout  (mundy convention: vec*mat = mat^T * vec)
// =============================================================================

#define MATRIX_OP_TEST(CaseName, OpNameStr, ExprBody, ExpMat)                                          \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Matrix>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                     \
      with_matrix_input(ctx, es, [&](auto tmp_m) {                                                     \
        mout()(es) = ExprBody;                                                                         \
        verify_matrix3_output(ctx, *fields().mout, [&](const double* c) -> std::array<double, 9> {    \
          const double d = (ctx.matrix_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;             \
          const auto tm = mat3_scale(ExpectedValues::mat(c), 1.0 / d);                                \
          [[maybe_unused]] const auto mb = ExpectedValues::mat_b(c);                                  \
          return ExpMat;                                                                               \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

#define MATRIX_VECTOR_OP_TEST(CaseName, OpNameStr, ExprBody, ExpVec)                                   \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Matrix>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                     \
      with_matrix_input(ctx, es, [&](auto tmp_m) {                                                     \
        vout()(es) = ExprBody;                                                                         \
        verify_vector3_output(ctx, *fields().vout, [&](const double* c) -> std::array<double, 3> {    \
          const double d = (ctx.matrix_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;             \
          const auto tm = mat3_scale(ExpectedValues::mat(c), 1.0 / d);                                \
          const auto vel_val = ExpectedValues::vel(c);                                                 \
          return ExpVec;                                                                               \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

// clang-format off
MATRIX_OP_TEST(MatrixOp_LeftScalarMultiply,          "matrix / left scalar multiply",   2.0 * tmp_m,         mat3_scale(tm, 2.0))
MATRIX_OP_TEST(MatrixOp_RightScalarMultiply,         "matrix / right scalar multiply",  tmp_m * 2.0,         mat3_scale(tm, 2.0))
MATRIX_OP_TEST(MatrixOp_RightScalarDivide,           "matrix / right scalar divide",    tmp_m / 2.0,         mat3_scale(tm, 0.5))
MATRIX_OP_TEST(MatrixOp_FieldAdd,                    "matrix / field add",              tmp_m + mat_b()(es), mat3_add(tm, mb))
MATRIX_OP_TEST(MatrixOp_FieldSubtract,               "matrix / field subtract",         tmp_m - mat_b()(es), mat3_sub(tm, mb))
MATRIX_OP_TEST(MatrixOp_MatrixMultiply,              "matrix / matrix-matrix multiply", tmp_m * mat_b()(es), mat3_mul(tm, mb))
MATRIX_VECTOR_OP_TEST(MatrixOp_MatrixVectorMultiply, "matrix / matrix-vector multiply", tmp_m * vel()(es),   mat3_vec3_mul(tm, vel_val))
MATRIX_VECTOR_OP_TEST(MatrixOp_VectorMatrixMultiply, "matrix / vector-matrix multiply", vel()(es) * tmp_m,   vec3_mat3_mul(vel_val, tm))
// clang-format on

#undef MATRIX_OP_TEST
#undef MATRIX_VECTOR_OP_TEST

// =============================================================================
// Section 8 — Matrix builtin coverage
//
// Coverage matrix:
//
//   Function                  Output   Expression
//   ------------------------  -------  -----------------------------------
//   trace                     scalar   trace(mtmp)
//   determinant               scalar   determinant(mtmp)
//   frobenius_norm            scalar   frobenius_norm(mtmp)
//   frobenius_inner_product   scalar   frobenius_inner_product(mtmp, mat_b(es))
//   sum                       scalar   sum(mtmp)
//   product                   scalar   product(mtmp)
//   min                       scalar   min(mtmp)
//   max                       scalar   max(mtmp)
//   mean                      scalar   mean(mtmp)
//   variance                  scalar   variance(mtmp)
//   stddev                    scalar   stddev(mtmp)
//   one_norm                  scalar   one_norm(mtmp)
//   two_norm                  scalar   two_norm(mtmp)   = frobenius_norm
//   inf_norm                  scalar   inf_norm(mtmp)
//   infinity_norm             scalar   infinity_norm(mtmp)
//   transpose                 matrix   transpose(mtmp)
//   elementwise_mul           matrix   elementwise_mul(mtmp, mat_b(es))
//   elementwise_div           matrix   elementwise_div(mtmp, mat_b(es))
//   copy                      matrix   copy(mtmp)
//   inverse                   matrix   inverse(mtmp)
//   adjugate                  matrix   adjugate(mtmp)
//   cofactors                 matrix   cofactors(mtmp)
//
// Each test sweeps:
//   matrix_shape in {RawAccessor, IntermediateMathExpr}
//   entity_shape in {OrdinaryEntity, ConnectedEntities}
//
// Matrix norms (over all 9 elements in row-major order):
//   one_norm      = max column sum   = max_j sum_i |A[i,j]|
//   two_norm      = frobenius_norm   = sqrt(sum of all A[i,j]^2)
//   inf_norm      = max row sum      = max_i sum_j |A[i,j]|
//   infinity_norm = same as inf_norm
//
// variance/stddev = population statistics (divide by N=9)
//
// cofactors[i,j] = (-1)^(i+j) * M_ij  (M_ij = minor = det of 2×2 submatrix)
// adjugate = transpose of cofactors
// inverse  = adjugate / det
// =============================================================================

#define MATRIX_SCALAR_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, ExpScalar)                           \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Matrix>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                     \
      with_matrix_input(ctx, es, [&](auto tmp_m) {                                                     \
        out()(es) = ExprBody;                                                                          \
        verify_scalar_output(ctx, *fields().out, [&](const double* c) -> std::array<double, 1> {      \
          const double d = (ctx.matrix_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;             \
          const auto tm = mat3_scale(ExpectedValues::mat(c), 1.0 / d);                                \
          [[maybe_unused]] const auto mb = ExpectedValues::mat_b(c);                                  \
          return {ExpScalar};                                                                          \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

#define MATRIX_MATRIX_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, ExpMat)                              \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Matrix>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                     \
      with_matrix_input(ctx, es, [&](auto tmp_m) {                                                     \
        mout()(es) = ExprBody;                                                                         \
        verify_matrix3_output(ctx, *fields().mout, [&](const double* c) -> std::array<double, 9> {    \
          const double d = (ctx.matrix_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;             \
          const auto tm = mat3_scale(ExpectedValues::mat(c), 1.0 / d);                                \
          [[maybe_unused]] const auto mb = ExpectedValues::mat_b(c);                                  \
          return ExpMat;                                                                               \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

// ---- Scalar-output builtins ----
// clang-format off
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Trace,                 "matrix / trace",                   trace(tmp_m),                                mat3_trace(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Determinant,           "matrix / determinant",             determinant(tmp_m),                          mat3_det(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_FrobeniusNorm,         "matrix / frobenius_norm",          frobenius_norm(tmp_m),                       mat3_frobenius_norm(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_FrobeniusInnerProduct, "matrix / frobenius_inner_product", frobenius_inner_product(tmp_m, mat_b()(es)), mat3_frobenius_inner_product(tm, mb))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Sum,                   "matrix / sum",                     sum(tmp_m),                                  mat3_sum(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Product,               "matrix / product",                 product(tmp_m),                              mat3_product(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Min,                   "matrix / min",                     min(tmp_m),                                  mat3_min(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Max,                   "matrix / max",                     max(tmp_m),                                  mat3_max(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Mean,                  "matrix / mean",                    mean(tmp_m),                                 mat3_mean(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Variance,              "matrix / variance",                variance(tmp_m),                             mat3_variance(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_Stddev,                "matrix / stddev",                  stddev(tmp_m),                               mat3_stddev(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_OneNorm,               "matrix / one_norm",                one_norm(tmp_m),                             mat3_one_norm(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_TwoNorm,               "matrix / two_norm",                two_norm(tmp_m),                             mat3_frobenius_norm(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_InfNorm,               "matrix / inf_norm",                inf_norm(tmp_m),                             mat3_inf_norm(tm))
MATRIX_SCALAR_BUILTIN_TEST(MatrixBuiltin_InfinityNorm,          "matrix / infinity_norm",           infinity_norm(tmp_m),                        mat3_inf_norm(tm))

// ---- Matrix-output builtins ----
MATRIX_MATRIX_BUILTIN_TEST(MatrixBuiltin_Transpose,             "matrix / transpose",               transpose(tmp_m),                            mat3_transpose(tm))
MATRIX_MATRIX_BUILTIN_TEST(MatrixBuiltin_ElementwiseMul,        "matrix / elementwise_mul",         elementwise_mul(tmp_m, mat_b()(es)),         mat3_elementwise_mul(tm, mb))
MATRIX_MATRIX_BUILTIN_TEST(MatrixBuiltin_ElementwiseDiv,        "matrix / elementwise_div",         elementwise_div(tmp_m, mat_b()(es)),         mat3_elementwise_div(tm, mb))
MATRIX_MATRIX_BUILTIN_TEST(MatrixBuiltin_Copy,                  "matrix / copy",                    copy(tmp_m),                                 tm)
MATRIX_MATRIX_BUILTIN_TEST(MatrixBuiltin_Inverse,               "matrix / inverse",                 inverse(tmp_m),                              mat3_inv(tm))
MATRIX_MATRIX_BUILTIN_TEST(MatrixBuiltin_Adjugate,              "matrix / adjugate",                adjugate(tmp_m),                             mat3_adjugate(tm))
MATRIX_MATRIX_BUILTIN_TEST(MatrixBuiltin_Cofactors,             "matrix / cofactors",               cofactors(tmp_m),                            mat3_cofactors(tm))
// clang-format on

#undef MATRIX_SCALAR_BUILTIN_TEST
#undef MATRIX_MATRIX_BUILTIN_TEST

// =============================================================================
// Section 9 — Quaternion operation coverage
//
// Coverage matrix:
//
//   Operation family          Expression             Output
//   ------------------------  ---------------------  ------
//   scalar left multiply      2.0 * qtmp             qout
//   scalar right multiply     qtmp * 2.0             qout
//   scalar right divide       qtmp / 2.0             qout
//   quaternion add            qtmp + quat_b(es)      qout
//   quaternion subtract       qtmp - quat_b(es)      qout
//   Hamilton product          qtmp * quat_b(es)      qout
//   quaternion-vector rotate  qtmp * vel(es)         vout
//   vector-quaternion rotate  vel(es) * qtmp         vout
//   quaternion-matrix rotate  qtmp * mat(es)         mout
//   matrix-quaternion rotate  mat(es) * qtmp         mout
//
// Each test sweeps:
//   quaternion_shape in {RawAccessor, IntermediateMathExpr}
//   entity_shape     in {OrdinaryEntity, ConnectedEntities}
//
// The rotation tests use vel()(es) and mat()(es) directly (not through vtmp/mtmp)
// so only the quaternion input shape axis is swept.
//
// Input values:
//   qtmp  = quat()(es) / d     quat = {1+0.1c0, 0.2+0.1c1, -0.3+0.1c2, 0.4}
//   quat_b()(es)               quat_b = {0.9-0.1c1, -0.4, 0.1+0.1c0, 0.2+0.1c2}
//   vel()(es)                  vel_val = {1+c0, 2+c1, 3+c2}
//   mat()(es)                  as described in Section 7
//
// Quaternion layout: {x, y, z, w} → {q[0], q[1], q[2], q[3]}, w is the real scalar.
// Constructor order: AQuaternion(w, x, y, z) — first argument is the real part.
//
// Rotation conventions (sandwich product with general, possibly non-unit q):
//   qtmp * vec  =  q * [0,v] * q^{-1}  (rotate v by q)         → vec result
//   vec * qtmp  =  q^{-1} * [0,v] * q  (rotate v by q^{-1})    → vec result
//   qtmp * mat  =  rotate each column of mat by q               → mat result
//   mat * qtmp  =  rotate each row of mat by q^{-1}             → mat result
//
// Hamilton product {x,y,z,w} convention (used by quat_mul helper):
//   (q1*q2).x = w1*x2 + x1*w2 + y1*z2 - z1*y2
//   (q1*q2).y = w1*y2 - x1*z2 + y1*w2 + z1*x2
//   (q1*q2).z = w1*z2 + x1*y2 - y1*x2 + z1*w2
//   (q1*q2).w = w1*w2 - x1*x2 - y1*y2 - z1*z2
// =============================================================================

#define QUATERNION_OP_TEST(CaseName, OpNameStr, ExprBody, ExpQuat)                                      \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Quaternion>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                 \
      with_quaternion_input(ctx, es, [&](auto tmp_q) {                                                 \
        qout()(es) = ExprBody;                                                                         \
        verify_quaternion_output(ctx, *fields().qout, [&](const double* c) -> std::array<double, 4> { \
          const double d = (ctx.quaternion_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;         \
          const auto tq = quat_scale(ExpectedValues::quat(c), 1.0 / d);                               \
          [[maybe_unused]] const auto qb = ExpectedValues::quat_b(c);                                 \
          return ExpQuat;                                                                              \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

#define QUATERNION_VECTOR_OP_TEST(CaseName, OpNameStr, ExprBody, ExpVec)                               \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Quaternion>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                 \
      with_quaternion_input(ctx, es, [&](auto tmp_q) {                                                 \
        vout()(es) = ExprBody;                                                                         \
        verify_vector3_output(ctx, *fields().vout, [&](const double* c) -> std::array<double, 3> {    \
          const double d = (ctx.quaternion_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;         \
          const auto tq = quat_scale(ExpectedValues::quat(c), 1.0 / d);                               \
          const auto vel_val = ExpectedValues::vel(c);                                                 \
          return ExpVec;                                                                               \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

#define QUATERNION_MATRIX_OP_TEST(CaseName, OpNameStr, ExprBody, ExpMat)                               \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Quaternion>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                 \
      with_quaternion_input(ctx, es, [&](auto tmp_q) {                                                 \
        mout()(es) = ExprBody;                                                                         \
        verify_matrix3_output(ctx, *fields().mout, [&](const double* c) -> std::array<double, 9> {    \
          const double d = (ctx.quaternion_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;         \
          const auto tq = quat_scale(ExpectedValues::quat(c), 1.0 / d);                               \
          const auto mat_val = ExpectedValues::mat(c);                                                 \
          return ExpMat;                                                                               \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

// clang-format off
QUATERNION_OP_TEST(QuaternionOp_LeftScalarMultiply,  "quaternion / left scalar multiply",  2.0 * tmp_q,          quat_scale(tq, 2.0))
QUATERNION_OP_TEST(QuaternionOp_RightScalarMultiply, "quaternion / right scalar multiply", tmp_q * 2.0,          quat_scale(tq, 2.0))
QUATERNION_OP_TEST(QuaternionOp_RightScalarDivide,   "quaternion / right scalar divide",   tmp_q / 2.0,          quat_scale(tq, 0.5))
QUATERNION_OP_TEST(QuaternionOp_FieldAdd,            "quaternion / field add",             tmp_q + quat_b()(es), quat_add(tq, qb))
QUATERNION_OP_TEST(QuaternionOp_FieldSubtract,       "quaternion / field subtract",        tmp_q - quat_b()(es), quat_sub(tq, qb))
QUATERNION_OP_TEST(QuaternionOp_HamiltonProduct,     "quaternion / Hamilton product",      tmp_q * quat_b()(es), quat_mul(tq, qb))
QUATERNION_VECTOR_OP_TEST(QuaternionOp_QuaternionVectorRotate, "quaternion / quaternion-vector rotate", tmp_q * vel()(es),  quat_rotate_vec(tq, vel_val))
QUATERNION_VECTOR_OP_TEST(QuaternionOp_VectorQuaternionRotate, "quaternion / vector-quaternion rotate", vel()(es) * tmp_q,  vec_rotate_quat(vel_val, tq))
QUATERNION_MATRIX_OP_TEST(QuaternionOp_QuaternionMatrixRotate, "quaternion / quaternion-matrix rotate", tmp_q * mat()(es),  quat_rotate_mat(tq, mat_val))
QUATERNION_MATRIX_OP_TEST(QuaternionOp_MatrixQuaternionRotate, "quaternion / matrix-quaternion rotate", mat()(es) * tmp_q,  mat_rotate_quat(mat_val, tq))
// clang-format on

#undef QUATERNION_OP_TEST
#undef QUATERNION_VECTOR_OP_TEST
#undef QUATERNION_MATRIX_OP_TEST

// =============================================================================
// Section 10 — Quaternion builtin coverage
//
// Coverage matrix:
//
//   Function        Output   Expression
//   --------------  -------  --------------------------
//   norm            scalar   norm(qtmp)
//   norm_squared    scalar   norm_squared(qtmp)
//   dot             scalar   dot(qtmp, quat_b(es))
//   copy            quat     copy(qtmp)
//   conjugate       quat     conjugate(qtmp)
//   normalize       quat     normalize(qtmp)
//   inverse         quat     inverse(qtmp)
//   slerp           quat     slerp(qtmp, quat_b(es), 0.25)
//
// Each test sweeps:
//   quaternion_shape in {RawAccessor, IntermediateMathExpr}
//   entity_shape     in {OrdinaryEntity, ConnectedEntities}
//
// Input values (layout {x,y,z,w}):
//   qtmp     = quat()(es) / d     quat = {1+0.1c0, 0.2+0.1c1, -0.3+0.1c2, 0.4}
//   quat_b   = quat_b()(es)       quat_b = {0.9-0.1c1, -0.4, 0.1+0.1c0, 0.2+0.1c2}
//
// conjugate({x,y,z,w}) = {-x,-y,-z,w}
// inverse(q) = conjugate(q) / norm_squared(q)
// normalize(q) = q / norm(q)
// slerp uses get_relaxed_zero_tolerance<double>() = 1e-8 as the near-parallel
//   threshold; our test quaternions have dot ≈ 0.91 so slerp is active.
// =============================================================================

#define QUATERNION_SCALAR_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, ExpScalar)                        \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Quaternion>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                 \
      with_quaternion_input(ctx, es, [&](auto tmp_q) {                                                 \
        out()(es) = ExprBody;                                                                          \
        verify_scalar_output(ctx, *fields().out, [&](const double* c) -> std::array<double, 1> {      \
          const double d = (ctx.quaternion_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;         \
          const auto tq = quat_scale(ExpectedValues::quat(c), 1.0 / d);                               \
          [[maybe_unused]] const auto qb = ExpectedValues::quat_b(c);                                 \
          return {ExpScalar};                                                                          \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

#define QUATERNION_QUATERNION_BUILTIN_TEST(CaseName, OpNameStr, ExprBody, ExpQuat)                     \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                                      \
    for_each_context<Quaternion>(OpNameStr, [&](const CoverageContext& ctx, auto es) {                 \
      with_quaternion_input(ctx, es, [&](auto tmp_q) {                                                 \
        qout()(es) = ExprBody;                                                                         \
        verify_quaternion_output(ctx, *fields().qout, [&](const double* c) -> std::array<double, 4> { \
          const double d = (ctx.quaternion_shape == ExprInputShape::RawAccessor) ? 1.0 : 2.0;         \
          const auto tq = quat_scale(ExpectedValues::quat(c), 1.0 / d);                               \
          [[maybe_unused]] const auto qb = ExpectedValues::quat_b(c);                                 \
          return ExpQuat;                                                                              \
        });                                                                                            \
      });                                                                                              \
    });                                                                                                \
  }

// ---- Scalar-output builtins ----
// clang-format off
QUATERNION_SCALAR_BUILTIN_TEST(QuaternionBuiltin_Norm,        "quaternion / norm",         norm(tmp_q),              quat_norm(tq))
QUATERNION_SCALAR_BUILTIN_TEST(QuaternionBuiltin_NormSquared, "quaternion / norm_squared", norm_squared(tmp_q),      tq[0]*tq[0] + tq[1]*tq[1] + tq[2]*tq[2] + tq[3]*tq[3])
QUATERNION_SCALAR_BUILTIN_TEST(QuaternionBuiltin_Dot,         "quaternion / dot",          dot(tmp_q, quat_b()(es)), tq[0]*qb[0] + tq[1]*qb[1] + tq[2]*qb[2] + tq[3]*qb[3])

// ---- Quaternion-output builtins ----
QUATERNION_QUATERNION_BUILTIN_TEST(QuaternionBuiltin_Conjugate, "quaternion / conjugate",   conjugate(tmp_q),                 quat_conjugate(tq))
QUATERNION_QUATERNION_BUILTIN_TEST(QuaternionBuiltin_Normalize, "quaternion / normalize",   normalize(tmp_q),                 quat_normalize(tq))
QUATERNION_QUATERNION_BUILTIN_TEST(QuaternionBuiltin_Copy,      "quaternion / copy",        copy(tmp_q),                      tq)
QUATERNION_QUATERNION_BUILTIN_TEST(QuaternionBuiltin_Inverse,   "quaternion / inverse",     inverse(tmp_q),                   quat_inverse(tq))
QUATERNION_QUATERNION_BUILTIN_TEST(QuaternionBuiltin_Slerp,     "quaternion / slerp",       slerp(tmp_q, quat_b()(es), 0.25), quat_slerp(tq, qb, 0.25))
// clang-format on

#undef QUATERNION_SCALAR_BUILTIN_TEST
#undef QUATERNION_QUATERNION_BUILTIN_TEST

// =============================================================================
// Section 11 — Evaluation trigger coverage
//
// Coverage matrix:
//
//   Trigger family             Expression                               Status
//   -------------------------  ---------------------------------------  ------
//   direct assignment          out(es) = r(es)                          ✓
//   constant assignment        out(es) = 2.0                            ✓
//   compound add               out(es) += q(es)                         ✓
//   compound subtract          out(es) -= q(es)                         ✓
//   compound multiply          out(es) *= 2.0 (constant)                ✓
//   compound divide            out(es) /= 2.0 (constant)                ✓
//   fused assignment           fused_assign(out, r+q, draw, 2*r)        ✓
//   fused swap                 fused_assign(out, q, aux, r)             ✓
//   local sum reduction        reduce_local_sum<double>(r(es))          ✓
//   local min reduction        reduce_local_min<double>(r(es))          ✓
//   local max reduction        reduce_local_max<double>(r(es))          ✓
//   global sum reduction       all_reduce_sum<double>(r(es))            ✓
//   global min reduction       all_reduce_min<double>(r(es))            ✓
//   global max reduction       all_reduce_max<double>(r(es))            ✓
//
// The abs-sum and abs-max reduction variants are listed in the spec but are not
// yet exposed in the accessor expression public API (NgpAccessorExpr.hpp); tests
// will be added once they are implemented.
//
// No input-shape sweep in this section — the trigger mechanism is the focus.
// All tests use OrdinaryEntity only (the entity-shape sweep belongs in the
// operation-coverage sections, not the trigger section).
// =============================================================================

TEST_F(AccessorExprCoverageFixture, EvalTrigger_Assignment) {
  auto es = make_node_entities();
  out()(es) = r()(es);
  expect_scalar_field_near("EvalTrigger_Assignment", *fields().out, parts().selected,
                           [](const double* c) -> std::array<double, 1> { return ExpectedValues::r(c); });
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_ConstantAssignment) {
  // Assign a compile-time constant (not a field expression) to verify the
  // constant-expression overload of operator= is exercised.
  auto es = make_node_entities();
  out()(es) = 2.0;
  expect_scalar_field_near("EvalTrigger_ConstantAssignment", *fields().out, parts().selected,
                           [](const double* /*c*/) -> std::array<double, 1> { return {2.0}; });
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_CompoundAdd) {
  auto es = make_node_entities();
  out()(es) = r()(es);   // out = r
  out()(es) += q()(es);  // out = r + q
  expect_scalar_field_near(
      "EvalTrigger_CompoundAdd", *fields().out, parts().selected,
      [](const double* c) -> std::array<double, 1> { return {ExpectedValues::r(c)[0] + ExpectedValues::q(c)[0]}; });
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_CompoundSubtract) {
  auto es = make_node_entities();
  out()(es) = r()(es);   // out = r
  out()(es) -= q()(es);  // out = r - q
  expect_scalar_field_near(
      "EvalTrigger_CompoundSubtract", *fields().out, parts().selected,
      [](const double* c) -> std::array<double, 1> { return {ExpectedValues::r(c)[0] - ExpectedValues::q(c)[0]}; });
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_CompoundMultiply) {
  auto es = make_node_entities();
  out()(es) = r()(es);  // out = r
  out()(es) *= 2.0;     // out = 2*r (constant overload)
  expect_scalar_field_near("EvalTrigger_CompoundMultiply", *fields().out, parts().selected,
                           [](const double* c) -> std::array<double, 1> { return {2.0 * ExpectedValues::r(c)[0]}; });
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_CompoundDivide) {
  auto es = make_node_entities();
  out()(es) = r()(es);  // out = r
  out()(es) /= 2.0;     // out = r/2
  expect_scalar_field_near("EvalTrigger_CompoundDivide", *fields().out, parts().selected,
                           [](const double* c) -> std::array<double, 1> { return {ExpectedValues::r(c)[0] / 2.0}; });
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_FusedAssign) {
  auto es = make_node_entities();
  fused_assign(out()(es), r()(es) + q()(es), first_draw()(es), r()(es) * 2.0);
  expect_scalar_field_near(
      "EvalTrigger_FusedAssign_out", *fields().out, parts().selected,
      [](const double* c) -> std::array<double, 1> { return {ExpectedValues::r(c)[0] + ExpectedValues::q(c)[0]}; });
  expect_scalar_field_near("EvalTrigger_FusedAssign_first_draw", *fields().first_draw, parts().selected,
                           [](const double* c) -> std::array<double, 1> { return {2.0 * ExpectedValues::r(c)[0]}; });
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_ReduceLocalSum) {
  // Expected sum: accumulate r(c) over locally-owned selected nodes on the host.
  // r(c) = 2.0 + c[0] + 0.25*c[1]; nodes in block_1 – block_2 are IDs {1..8, 10..12}.
  fields().r->sync_to_host();
  double expected = 0.0;
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
        if (!mesh.bucket(entity).owned()) return;
        expected += stk::mesh::field_data(*fields().r, entity)[0];
      });
  auto es = make_node_entities();
  const double result = reduce_local_sum<double>(r()(es));
  EXPECT_NEAR(result, expected, kTolerance * std::abs(expected) + kTolerance);
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_ReduceLocalMax) {
  // Expected max: maximum r(c) over locally-owned selected nodes on the host.
  fields().r->sync_to_host();
  double expected = 0.0;  // r > 2 always; sentinel 0 is safely below any r value
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
        if (!mesh.bucket(entity).owned()) return;
        expected = std::max(expected, stk::mesh::field_data(*fields().r, entity)[0]);
      });
  auto es = make_node_entities();
  const double result = reduce_local_max<double>(r()(es));
  EXPECT_NEAR(result, expected, kTolerance);
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_ReduceLocalMin) {
  // Expected min: minimum r(c) over locally-owned selected nodes on the host.
  fields().r->sync_to_host();
  double expected = 1.0e10;  // r < 3 always; sentinel 1e10 is safely above any r value
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
        if (!mesh.bucket(entity).owned()) return;
        expected = std::min(expected, stk::mesh::field_data(*fields().r, entity)[0]);
      });
  auto es = make_node_entities();
  const double result = reduce_local_min<double>(r()(es));
  EXPECT_NEAR(result, expected, kTolerance);
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_AllReduceSum) {
  // Compare all_reduce_sum against the local sum: the global value must be >= the
  // local contribution and must account for all ranks (1-rank: equal; 2-rank: larger).
  // Also verify the local reduction result exactly, which tests both code paths.
  fields().r->sync_to_host();
  double local_expected = 0.0;
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
        if (!mesh.bucket(entity).owned()) return;
        local_expected += stk::mesh::field_data(*fields().r, entity)[0];
      });
  auto es = make_node_entities();
  const double local_result  = reduce_local_sum<double>(r()(es));
  const double global_result = all_reduce_sum<double>(r()(es));
  EXPECT_NEAR(local_result, local_expected, kTolerance * std::abs(local_expected) + kTolerance);
  EXPECT_GE(global_result, local_expected);  // global >= local on every rank
  EXPECT_GT(global_result, 0.0);
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_AllReduceMax) {
  // all_reduce_max must be >= the local maximum on every rank.
  fields().r->sync_to_host();
  double local_expected = 0.0;
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
        if (!mesh.bucket(entity).owned()) return;
        local_expected = std::max(local_expected, stk::mesh::field_data(*fields().r, entity)[0]);
      });
  auto es = make_node_entities();
  const double local_result  = reduce_local_max<double>(r()(es));
  const double global_result = all_reduce_max<double>(r()(es));
  EXPECT_NEAR(local_result, local_expected, kTolerance);
  EXPECT_GE(global_result, local_expected);  // global max >= local max
  EXPECT_GT(global_result, 2.0);             // r > 2 on all selected nodes
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_AllReduceMin) {
  // all_reduce_min must be <= the local minimum on every rank.
  fields().r->sync_to_host();
  double local_expected = 1.0e10;
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& mesh, const stk::mesh::Entity entity) {
        if (!mesh.bucket(entity).owned()) return;
        local_expected = std::min(local_expected, stk::mesh::field_data(*fields().r, entity)[0]);
      });
  auto es = make_node_entities();
  const double local_result  = reduce_local_min<double>(r()(es));
  const double global_result = all_reduce_min<double>(r()(es));
  EXPECT_NEAR(local_result, local_expected, kTolerance);
  EXPECT_LE(global_result, local_expected);  // global min <= local min
  EXPECT_GT(global_result, 2.0);             // r > 2 on all selected nodes
}

TEST_F(AccessorExprCoverageFixture, EvalTrigger_FusedSwap) {
  // Swap out and aux simultaneously via fused_assign:
  //   out_new = q,  aux_new = r
  // Initialize out = r, aux = q, then fused-assign swaps them.
  auto es = make_node_entities();
  out()(es) = r()(es);   // out = r
  aux()(es) = q()(es);   // aux = q (note: aux field already holds its own init value,
                          //         but we write q here as "initial out-value for swap")
  // fused_assign: out ← q()(es),  aux ← r()(es)
  fused_assign(out()(es), q()(es), aux()(es), r()(es));
  expect_scalar_field_near(
      "EvalTrigger_FusedSwap_out", *fields().out, parts().selected,
      [](const double* c) -> std::array<double, 1> { return {ExpectedValues::q(c)[0]}; });
  expect_scalar_field_near(
      "EvalTrigger_FusedSwap_aux", *fields().aux, parts().selected,
      [](const double* c) -> std::array<double, 1> { return {ExpectedValues::r(c)[0]}; });
}

// =============================================================================
// Section 12 — Custom expression coverage
//
// Coverage matrix:
//
//   Feature family                  Expression                                Status
//   ------------------------------  ----------------------------------------  ------
//   apply_expr scalar fields        out = apply_expr(func, r(es), q(es), 1.0)  ✓
//   apply_expr scalar intermediate  out = apply_expr(func, tmp, q(es), 1.0)    ✓
//   apply_expr scalar constants     same as above (1.0 is the constant)         ✓
//   apply_expr composed result      out = 2.0 * apply_expr(func, r(es), q(es), 1.0)  ✓
//   apply_expr vector fields        vout = apply_expr(func, vel(es), force(es)) ✓
//   apply_expr vector intermediate  vout = apply_expr(func, vtmp, force(es))    ✓
//   apply_expr mixed scalar/vector  vout = apply_expr(func, vel(es), r(es))     ✓
//   sink_expr read only             sink_expr(func, read_only(r(es)))              ✓
//   sink_expr read write            sink_expr(func, read_write(out(es)), r(es))    ✓
//   sink_expr overwrite all         sink_expr(func, overwrite_all(out(es)), r(es)) ✓
//   sink_expr default read only     sink_expr(func, read_write(out(es)), r(es))    ✓
//   sink_expr mixed access modes    sink_expr(func, rw(out), ro(r), oa(aux))       ✓
//   atomic add                      atomic_add(out(es), r(es))                  ✓
//   atomic subtract                 atomic_sub(out(es), r(es))                  ✓
//   atomic multiply                 atomic_mul(out(es), r(es))                  ✓
//   atomic divide                   atomic_div(out(es), r(es))                  ✓
//   named sink wrapper              rotate_quaternion(qout(es), quat(es), omega(es)) ✓
//
// Functors are the ones defined in Section 1:
//   ScalarVariadicApplyFunc  — (x, y, bias) → x + 2*y + bias
//   VectorBinaryApplyFunc    — (v, w) → 2*v + w
//   VectorScalarMixedApplyFunc — (v, s) → s * v
//   ReadWriteScaleAddSinkFunc  — y += scale * x
//   OverwriteAffineSinkFunc    — y = x + bias
//   ReadOnlySinkFunc           — no-op
//   MixedAccessSinkFunc        — y += x (read_write), z = x (overwrite_all)
//
// Both sink_expr(...) and named sink wrappers like rotate_quaternion(...)
// execute immediately and return void.  Sinks are terminal: they evaluate
// the expression tree on the spot, matching the semantics of assigning to
// an accessor expression as an lvalue.
//
// Each test uses for_each_context<> (no input-shape axis) to sweep:
//   entity_shape in {OrdinaryEntity, ConnectedEntities}
// =============================================================================

TEST_F(AccessorExprCoverageFixture, CustomExpr_ScalarVariadicApply) {
  // apply_expr(ScalarVariadicApplyFunc{}, r, q, 1.0) = r + 2*q + 1
  // Primary inputs are raw field accessors.
  for_each_context<>("custom / ScalarVariadicApply", [&](const CoverageContext& ctx, auto es) {
    out()(es) = apply_expr(ScalarVariadicApplyFunc{}, r()(es), q()(es), 1.0);
    verify_scalar_output(ctx, *fields().out, [](const double* c) -> std::array<double, 1> {
      return {ExpectedValues::r(c)[0] + 2.0 * ExpectedValues::q(c)[0] + 1.0};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_ScalarIntermediateApply) {
  // apply_expr with an intermediate math expression as the first argument.
  // tmp = r(es) / 2.0  (a DivExpr, not a raw accessor) — verifies that apply_expr
  // accepts non-accessor inputs and that the value passes through correctly.
  for_each_context<>("custom / ScalarIntermediateApply", [&](const CoverageContext& ctx, auto es) {
    auto tmp = r()(es) / 2.0;
    out()(es) = apply_expr(ScalarVariadicApplyFunc{}, tmp, q()(es), 1.0);
    verify_scalar_output(ctx, *fields().out, [](const double* c) -> std::array<double, 1> {
      return {ExpectedValues::r(c)[0] / 2.0 + 2.0 * ExpectedValues::q(c)[0] + 1.0};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_ComposedApplyResult) {
  // 2.0 * apply_expr(...) — verifies that the return value of apply_expr can be
  // used as an operand in further arithmetic (composed expression).
  for_each_context<>("custom / ComposedApplyResult", [&](const CoverageContext& ctx, auto es) {
    out()(es) = 2.0 * apply_expr(ScalarVariadicApplyFunc{}, r()(es), q()(es), 1.0);
    verify_scalar_output(ctx, *fields().out, [](const double* c) -> std::array<double, 1> {
      return {2.0 * (ExpectedValues::r(c)[0] + 2.0 * ExpectedValues::q(c)[0] + 1.0)};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_VectorBinaryApply) {
  // apply_expr(VectorBinaryApplyFunc{}, vel, force) = 2*vel + force (component-wise)
  // Primary inputs are raw field accessors.
  for_each_context<>("custom / VectorBinaryApply", [&](const CoverageContext& ctx, auto es) {
    vout()(es) = apply_expr(VectorBinaryApplyFunc{}, vel()(es), force()(es));
    verify_vector3_output(ctx, *fields().vout, [](const double* c) -> std::array<double, 3> {
      const auto vv = ExpectedValues::vel(c);
      const auto fv = ExpectedValues::force(c);
      return {2.0 * vv[0] + fv[0], 2.0 * vv[1] + fv[1], 2.0 * vv[2] + fv[2]};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_VectorIntermediateApply) {
  // apply_expr with an intermediate vector expression as the first argument.
  // vtmp = vel(es) / 2.0  (a DivExpr) — verifies vector intermediate inputs to apply_expr.
  for_each_context<>("custom / VectorIntermediateApply", [&](const CoverageContext& ctx, auto es) {
    auto vtmp = vel()(es) / 2.0;
    vout()(es) = apply_expr(VectorBinaryApplyFunc{}, vtmp, force()(es));
    verify_vector3_output(ctx, *fields().vout, [](const double* c) -> std::array<double, 3> {
      const auto vv = ExpectedValues::vel(c);
      const auto fv = ExpectedValues::force(c);
      return {2.0 * (vv[0] / 2.0) + fv[0], 2.0 * (vv[1] / 2.0) + fv[1], 2.0 * (vv[2] / 2.0) + fv[2]};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_VectorScalarMixedApply) {
  // apply_expr(VectorScalarMixedApplyFunc{}, vel, r) = r * vel (scalar * vector)
  for_each_context<>("custom / VectorScalarMixedApply", [&](const CoverageContext& ctx, auto es) {
    vout()(es) = apply_expr(VectorScalarMixedApplyFunc{}, vel()(es), r()(es));
    verify_vector3_output(ctx, *fields().vout, [](const double* c) -> std::array<double, 3> {
      const auto vv = ExpectedValues::vel(c);
      const double s = ExpectedValues::r(c)[0];
      return {s * vv[0], s * vv[1], s * vv[2]};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_SinkExprReadWrite) {
  // ReadWriteScaleAddSinkFunc: y += scale * x  →  out += 2*r
  // out is initialized to q before the sink; out_new = q + 2*r.
  for_each_context<>("custom / SinkExprReadWrite", [&](const CoverageContext& ctx, auto es) {
    out()(es) = q()(es);
    sink_expr(ReadWriteScaleAddSinkFunc{}, read_write(out()(es)), read_only(r()(es)), 2.0);
    verify_scalar_output(ctx, *fields().out, [](const double* c) -> std::array<double, 1> {
      return {ExpectedValues::q(c)[0] + 2.0 * ExpectedValues::r(c)[0]};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_SinkExprOverwriteAll) {
  // OverwriteAffineSinkFunc: y = x + bias  →  out = r + 1
  for_each_context<>("custom / SinkExprOverwriteAll", [&](const CoverageContext& ctx, auto es) {
    sink_expr(OverwriteAffineSinkFunc{}, overwrite_all(out()(es)), read_only(r()(es)), 1.0);
    verify_scalar_output(ctx, *fields().out,
                         [](const double* c) -> std::array<double, 1> { return {ExpectedValues::r(c)[0] + 1.0}; });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_SinkExprReadOnly) {
  // ReadOnlySinkFunc: no-op.  r field must be unchanged after the kernel.
  for_each_context<>("custom / SinkExprReadOnly", [&](const CoverageContext& ctx, auto es) {
    sink_expr(ReadOnlySinkFunc{}, read_only(r()(es)));
    verify_scalar_output(ctx, *fields().r,
                         [](const double* c) -> std::array<double, 1> { return ExpectedValues::r(c); });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_SinkExprDefaultReadOnly) {
  // Passing r()(es) without any explicit access-mode wrapper exercises the
  // "default read only" path: the framework must treat unwrapped expressions
  // as read_only.  ReadWriteScaleAddSinkFunc: y += scale * x  →  out = q + 2*r.
  // The expected result is the same as CustomExpr_SinkExprReadWrite, which passes
  // read_only(r()(es)) explicitly — confirming that the default matches read_only.
  for_each_context<>("custom / SinkExprDefaultReadOnly", [&](const CoverageContext& ctx, auto es) {
    out()(es) = q()(es);
    sink_expr(ReadWriteScaleAddSinkFunc{}, read_write(out()(es)), r()(es), 2.0);
    verify_scalar_output(ctx, *fields().out, [](const double* c) -> std::array<double, 1> {
      return {ExpectedValues::q(c)[0] + 2.0 * ExpectedValues::r(c)[0]};
    });
  });
}

TEST_F(AccessorExprCoverageFixture, CustomExpr_SinkExprMixedAccess) {
  // MixedAccessSinkFunc: y += x (read_write out), z = x (overwrite_all aux).
  // Exercises read_write, read_only, and overwrite_all in a single sink_expr call.
  // out starts at q; after the kernel: out = q + r, aux = r.
  for_each_context<>("custom / SinkExprMixedAccess", [&](const CoverageContext& ctx, auto es) {
    out()(es) = q()(es);
    sink_expr(MixedAccessSinkFunc{}, read_write(out()(es)), read_only(r()(es)), overwrite_all(aux()(es)));
    verify_scalar_output(ctx, *fields().out, [](const double* c) -> std::array<double, 1> {
      return {ExpectedValues::q(c)[0] + ExpectedValues::r(c)[0]};
    });
    verify_scalar_output(ctx, *fields().aux,
                         [](const double* c) -> std::array<double, 1> { return ExpectedValues::r(c); });
  });
}

// All atomic tests: init out (or no-op), apply atomic_fn(out, r), verify out.
// r > 2 on all nodes so AtomicDiv is safe.
// Each test sweeps entity_shape in {OrdinaryEntity, ConnectedEntities}.
#define ATOMIC_OP_TEST(CaseName, AtomicFn, InitStmt, ExpVal)                                    \
  TEST_F(AccessorExprCoverageFixture, CaseName) {                                               \
    for_each_context<>("custom / " #CaseName, [&](const CoverageContext& ctx, auto es) {        \
      InitStmt;                                                                                 \
      auto expr = AtomicFn(out()(es), r()(es));                                                 \
      expr.driver()->run(expr);                                                                 \
      verify_scalar_output(ctx, *fields().out,                                                  \
                           [](const double* c) -> std::array<double, 1> { return ExpVal; });   \
    });                                                                                         \
  }

// clang-format off
//                                                          InitStmt               ExpVal
ATOMIC_OP_TEST(CustomExpr_AtomicAdd, atomic_add, (void)0,               ExpectedValues::r(c))
ATOMIC_OP_TEST(CustomExpr_AtomicSub, atomic_sub, out()(es) = q()(es),   {ExpectedValues::q(c)[0] - ExpectedValues::r(c)[0]})
ATOMIC_OP_TEST(CustomExpr_AtomicMul, atomic_mul, out()(es) = q()(es),   {ExpectedValues::q(c)[0] * ExpectedValues::r(c)[0]})
ATOMIC_OP_TEST(CustomExpr_AtomicDiv, atomic_div, out()(es) = q()(es),   {ExpectedValues::q(c)[0] / ExpectedValues::r(c)[0]})
// clang-format on

#undef ATOMIC_OP_TEST

TEST_F(AccessorExprCoverageFixture, CustomExpr_RotateQuaternionSink) {
  // rotate_quaternion(qout, omega, dt): integrates angular velocity omega into qout.
  // omega = {0, 0, 1} (unit z-axis rotation), dt = 0.0 → qout unchanged (sin(0)=0).
  // Use dt = 0.0 so that the expected output equals the initial qout value (identity).
  // That is: w = |omega| = 1, dt = 0, sw = sin(0) = 0, cw = cos(0) = 1 → no change.
  for_each_context<>("custom / RotateQuaternionSink", [&](const CoverageContext& ctx, auto es) {
    // qout starts at identity {0,0,0,1}.
    rotate_quaternion(qout()(es), omega()(es), 0.0);
    // With dt=0, rotate_quaternion is a no-op, so qout stays at identity.
    verify_quaternion_output(ctx, *fields().qout, [](const double* c) -> std::array<double, 4> {
      return ExpectedValues::qout(c);  // {0,0,0,1}
    });
  });
}

// =============================================================================
// Section 13 — RNG expression coverage
//
// Coverage matrix:
//
//   RNG family                         Expression                              Status
//   ---------------------------------  --------------------------------------  ------
//   field seed, field counter          rng(seed(es), counter(es)).rand()        ✓
//   field seed, constant counter       rng(seed(es), counter*0.0).rand()        ✓
//   constant seed, field counter       rng(seed*0+42, counter(es)).rand()       ✓
//   intermediate seed expression       rng(seed(es)+0.0, counter(es)).rand()    ✓
//   intermediate counter expression    rng(seed(es), counter(es)+1.0).rand()    ✓
//   intermediate seed and counter      rng(seed(es)+1.0, counter(es)+1.0).rand() ✓
//   uniform distribution constants     rng(...).uniform<double>(0.0, 1.0)       ✓
//   uniform distribution expr lower    rng(...).uniform(r(es)/4.0, 1.0)         ✓
//   uniform distribution expr upper    rng(...).uniform(0.0, q(es)/5.0)         ✓
//   uniform distribution expr bounds   rng(...).uniform(r(es), q(es))           ✓
//   sequential rand draws              fused_assign(a, rng.rand(), b, rng.rand()) ✓
//   sequential uniform draws           fused_assign(a, rng.uniform(0,1), b, ...)  ✓
//   composed random result             out = 2.0*rng.rand() + 1.0                ✓
//   runtime reuse valid                (reserved for future test)
//   runtime reuse invalid              (reserved for future test)
//
// Verification strategy:
//   openrand::Philox is a deterministic counter-based RNG.  The device kernel
//   evaluates make_philox(seed_field, counter_field) and calls rand<T>() or
//   uniform<T>() on the resulting object.  The Philox object maintains its own
//   internal counter that advances with each draw; the field counter is never
//   modified.  We can therefore replicate every draw exactly on the host:
//
//     auto expected_rng = ::mundy::make_philox(
//         static_cast<size_t>(seed_field[e]),
//         static_cast<size_t>(counter_field[e]));
//     double expected = expected_rng.template rand<double>();  // or .uniform<double>(lo, hi)
//
//   For fused multi-draw tests the same rng object is reused across draws
//   (CounterBasedRNGExpr has supports_runtime_reuse = true), so successive
//   .rand() calls on the cached object produce the sequence.
//
// Note: the "constant counter" and "constant seed" tests use arithmetic
// expressions that evaluate to constants (counter*0, seed*0+42) rather than
// bare integer literals because the framework expects expression nodes.
// =============================================================================


TEST_F(AccessorExprCoverageFixture, RNG_SingleDraw) {
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  first_draw()(es) = rng_.template rand<double>();
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template rand<double>())
            << "RNG_SingleDraw: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_FusedAssignTwoDraws) {
  // CounterBasedRNGExpr has supports_runtime_reuse=true, so both draw expressions
  // in the fused kernel share one cached Philox object; successive rand() calls
  // advance its internal counter, producing two distinct values.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  fused_assign(first_draw()(es), rng_.template rand<double>(),
               second_draw()(es), rng_.template rand<double>());
  fields().first_draw->sync_to_host();
  fields().second_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,        entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,     entity)[0];
        const double actual_fd   = stk::mesh::field_data(*fields().first_draw,  entity)[0];
        const double actual_sd   = stk::mesh::field_data(*fields().second_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        const double expected_fd = expected_rng.template rand<double>();
        const double expected_sd = expected_rng.template rand<double>();
        EXPECT_DOUBLE_EQ(actual_fd, expected_fd) << "RNG_FusedAssignTwoDraws: first draw mismatch";
        EXPECT_DOUBLE_EQ(actual_sd, expected_sd) << "RNG_FusedAssignTwoDraws: second draw mismatch";
        EXPECT_NE(actual_fd, actual_sd)           << "RNG_FusedAssignTwoDraws: both draws are equal";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_UniformDraw) {
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  first_draw()(es) = rng_.template uniform<double>(0.0, 1.0);
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template uniform<double>(0.0, 1.0))
            << "RNG_UniformDraw: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_ExpressionSeed) {
  // seed(es)+0.0 wraps the accessor in a BinaryValueExpr.  The numeric result is
  // identical to seed(es) (adding 0.0 is exact in IEEE 754), so the draw must
  // match make_philox(seed_val, counter_val).rand<double>() exactly.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es) + 0.0, counter()(es));
  first_draw()(es) = rng_.template rand<double>();
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val + 0.0),
                                                      static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template rand<double>())
            << "RNG_ExpressionSeed: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_UniformExprBounds) {
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  first_draw()(es) = rng_.template uniform<double>(r()(es), q()(es));
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double r_val       = stk::mesh::field_data(*fields().r,          entity)[0];
        const double q_val       = stk::mesh::field_data(*fields().q,          entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template uniform<double>(r_val, q_val))
            << "RNG_UniformExprBounds: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_ConstantCounter) {
  // counter(es)*0.0 evaluates to 0.0 for every entity: exercises the path where the
  // counter expression is a constant (BinaryValueExpr), not a raw accessor.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es) * 0.0);
  first_draw()(es) = rng_.template rand<double>();
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double actual   = stk::mesh::field_data(*fields().first_draw, entity)[0];
        // Effective counter = counter_field * 0.0 = 0.0 for all entities.
        auto expected_rng = ::mundy::make_philox(static_cast<size_t>(seed_val), static_cast<size_t>(0.0));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template rand<double>()) << "RNG_ConstantCounter: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_ConstantSeed) {
  // seed(es)*0.0+42.0 evaluates to 42.0 for every entity: exercises the path where the
  // seed expression is a constant (BinaryValueExpr), not a raw accessor.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es) * 0.0 + 42.0, counter()(es));
  first_draw()(es) = rng_.template rand<double>();
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        // Effective seed = seed_field*0.0 + 42.0 = 42.0 for all entities.
        auto expected_rng = ::mundy::make_philox(static_cast<size_t>(42.0),
                                               static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template rand<double>()) << "RNG_ConstantSeed: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_IntermediateCounter) {
  // counter(es)+1.0 wraps the accessor in a BinaryValueExpr.  The effective counter
  // passed to make_philox is counter_field + 1.0.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es) + 1.0);
  first_draw()(es) = rng_.template rand<double>();
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val + 1.0));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template rand<double>()) << "RNG_IntermediateCounter: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_IntermediateSeedAndCounter) {
  // Both seed and counter are BinaryValueExpr nodes (each shifted by +1.0).
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es) + 1.0, counter()(es) + 1.0);
  first_draw()(es) = rng_.template rand<double>();
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val    + 1.0),
                                                      static_cast<size_t>(counter_val + 1.0));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template rand<double>())
            << "RNG_IntermediateSeedAndCounter: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_UniformExprLower) {
  // uniform<double>(r/4.0, 1.0): lower bound is a per-entity expression.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  first_draw()(es) = rng_.template uniform<double>(r()(es) / 4.0, 1.0);
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double r_val       = stk::mesh::field_data(*fields().r,          entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template uniform<double>(r_val / 4.0, 1.0))
            << "RNG_UniformExprLower: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_UniformExprUpper) {
  // uniform<double>(0.0, q/5.0): upper bound is a per-entity expression.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  first_draw()(es) = rng_.template uniform<double>(0.0, q()(es) / 5.0);
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double q_val       = stk::mesh::field_data(*fields().q,          entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, expected_rng.template uniform<double>(0.0, q_val / 5.0))
            << "RNG_UniformExprUpper: value mismatch";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_FusedUniformDraws) {
  // Two sequential uniform draws sharing one cached Philox; the second draw gets a
  // different state because the internal counter has advanced after the first draw.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  fused_assign(first_draw()(es),  rng_.template uniform<double>(0.0, 1.0),
               second_draw()(es), rng_.template uniform<double>(0.0, 1.0));
  fields().first_draw->sync_to_host();
  fields().second_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,        entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,     entity)[0];
        const double actual_fd   = stk::mesh::field_data(*fields().first_draw,  entity)[0];
        const double actual_sd   = stk::mesh::field_data(*fields().second_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        const double expected_fd = expected_rng.template uniform<double>(0.0, 1.0);
        const double expected_sd = expected_rng.template uniform<double>(0.0, 1.0);
        EXPECT_DOUBLE_EQ(actual_fd, expected_fd) << "RNG_FusedUniformDraws: first draw mismatch";
        EXPECT_DOUBLE_EQ(actual_sd, expected_sd) << "RNG_FusedUniformDraws: second draw mismatch";
        EXPECT_NE(actual_fd, actual_sd)           << "RNG_FusedUniformDraws: both draws are equal";
      });
}

TEST_F(AccessorExprCoverageFixture, RNG_ComposedResult) {
  // 2.0 * rng.rand<double>() + 1.0: verifies arithmetic composition with a random draw.
  // The composed expression is deterministic: expected = 2.0*make_philox(s,c).rand() + 1.0.
  auto es   = make_node_entities();
  auto rng_ = rng(seed()(es), counter()(es));
  first_draw()(es) = 2.0 * rng_.template rand<double>() + 1.0;
  fields().first_draw->sync_to_host();
  stk::mesh::for_each_entity_run(
      static_cast<const stk::mesh::BulkData&>(bulk()), stk::topology::NODE_RANK, parts().selected,
      [&](const stk::mesh::BulkData& /*mesh*/, const stk::mesh::Entity entity) {
        const double seed_val    = stk::mesh::field_data(*fields().seed,       entity)[0];
        const double counter_val = stk::mesh::field_data(*fields().counter,    entity)[0];
        const double actual      = stk::mesh::field_data(*fields().first_draw, entity)[0];
        auto expected_rng        = ::mundy::make_philox(static_cast<size_t>(seed_val),
                                                      static_cast<size_t>(counter_val));
        EXPECT_DOUBLE_EQ(actual, 2.0 * expected_rng.template rand<double>() + 1.0)
            << "RNG_ComposedResult: value mismatch";
      });
}

}  // namespace
}  // namespace mesh
}  // namespace mundy
