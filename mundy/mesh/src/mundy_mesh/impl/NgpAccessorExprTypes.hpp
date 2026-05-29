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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRTYPES_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRTYPES_HPP_

/// \file NgpAccessorExprTypes.hpp
/// \brief Shared types for the NGP expression system.

// Kokkos
#include <Kokkos_Core.hpp>

// STL
#include <any>
#include <map>
#include <string>
#include <unordered_map>

// STK mesh
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>

// Mundy
#include <mundy_utils/StringLiteral.hpp>
#include <mundy_utils/aggregate.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <size_t NumEntities, size_t Ord, typename DriverType>
class EntityExpr;

//! \name Evaluation contexts
//@{

class NgpEvalContext {
 public:
  KOKKOS_INLINE_FUNCTION
  NgpEvalContext(stk::mesh::NgpMesh ngp_mesh) : ngp_mesh_(ngp_mesh) {
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::NgpMesh ngp_mesh() const {
    return ngp_mesh_;
  }

 private:
  stk::mesh::NgpMesh ngp_mesh_;
};
//@}

// Cached eval cannot safely return a raw reference alongside a by-value cache object.
template <typename ValueType>
class OwnedCachedValue {
 public:
  using value_type = std::remove_cvref_t<ValueType>;

  KOKKOS_DEFAULTED_FUNCTION
  OwnedCachedValue() = default;

  KOKKOS_FUNCTION
  explicit OwnedCachedValue(const value_type& value) : value_(value) {
  }

  KOKKOS_FUNCTION
  explicit OwnedCachedValue(value_type&& value) : value_(std::move(value)) {
  }

  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr value_type& get([[maybe_unused]] CacheType& cache) {
    return value_;
  }

  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr const value_type& get([[maybe_unused]] const CacheType& cache) const {
    return value_;
  }

 private:
  value_type value_;
};

template <typename ValueType>
OwnedCachedValue(ValueType&&) -> OwnedCachedValue<std::remove_cvref_t<ValueType>>;

template <typename Tag>
class CachedTagGetter {
 public:
  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(CacheType& cache) const {
    return ::mundy::get<Tag>(cache);
  }

  template <typename CacheType>
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) get(const CacheType& cache) const {
    return ::mundy::get<Tag>(cache);
  }
};

class RuntimeReuseValidator {
 public:
  RuntimeReuseValidator() = default;

  template <typename EvalCountsType, EvalCountsType eval_counts, typename Expr>
  void validate(const Expr& expr) {
    static_assert(has<typename Expr::our_tag>(eval_counts), "eval_counts must contain the expression tag");
    if constexpr (Expr::supports_runtime_reuse && !Expr::has_static_eval &&
                  get<typename Expr::our_tag>(eval_counts) > 1) {
      validate_or_track(expr);
    }
  }

 private:
  template <typename Expr>
  void validate_or_track(const Expr& expr) {
    using tag_t = typename Expr::our_tag;
    auto [it, inserted] = first_expr_ptr_by_tag_.emplace(tag_key<tag_t>(), static_cast<const void*>(&expr));
    if (!inserted) {
      const auto* first_expr = static_cast<const Expr*>(it->second);
      MUNDY_THROW_REQUIRE(
          expr.runtime_reuse_equivalent(*first_expr), std::logic_error,
          "Unsupported runtime reuse found in expression tree...\n"
          "\n"
          "Hi, you've surpassed mundy's current supported behavior related to runtime reuse in an expression tree.\n"
          "This means that you managed to create two identical random number generator expressions with different "
          "runtime seed/counter combos.\n"
          "\n"
          "For example:\n"
          ">>>  auto rng_a = rng(seed(es) + 1.0, counter(es));\n"
          ">>>  auto rng_b = rng(seed(es) + 2.0, counter(es));\n"
          ">>>  fused_assign(first_draw(es), /*=*/rng_a.template rand<double>(),\n"
          ">>>               second_draw(es), /*=*/rng_b.template rand<double>());\n"
          "\n"
          "This is worked around by changing the inputs to the random number generators to have different forms:\n"
          "Example workaround 1 (change order of addition):\n"
          ">>>  auto rng_a = rng(1.0 + seed(es), counter(es));\n"
          ">>>  auto rng_b = rng(seed(es) + 2.0, counter(es));\n"
          "\n"
          "Example workaround 2 (add zero):\n"
          ">>>  auto rng_a = rng(seed(es) + 1.0, counter(es) + 0.0);\n"
          ">>>  auto rng_b = rng(seed(es) + 2.0, counter(es));\n"
          "\n"
          "If this remains a problem, let us know and we'll test an arbitrary runtime-reuse implementation.\n"
          "It's fully possible, we simply haven't needed it yet.");
    }
  }

  template <typename Tag>
  static const void* tag_key() {
    static const int key = 0;
    return &key;
  }

  std::unordered_map<const void*, const void*> first_expr_ptr_by_tag_;
};

template <StringLiteral map_name>
class AnyRankSelectorMap {
 public:
  AnyRankSelectorMap() = default;

  static std::string name() {
    return map_name.to_string();
  }

  bool contains(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) const {
    const auto& selector_map = ranked_selector_maps_[rank];
    return selector_map.find(selector) != selector_map.end();
  }

  template <typename T>
  void insert(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector, T value) {
    auto& selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(!contains(rank, selector), std::logic_error,
                       "Attempting to insert a rank and selector pair into AnyRankSelectorMap that is already present");
    selector_map.emplace(selector, std::move(value));
  }

  template <typename T>
  T& at(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) {
    auto& selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(contains(rank, selector), std::logic_error,
                       "Attempting to access a rank and selector pair into AnyRankSelectorMap that isn't present");
    return std::any_cast<T&>(selector_map.at(selector));
  }

  template <typename T>
  const T& at(stk::mesh::EntityRank rank, const stk::mesh::Selector& selector) const {
    auto& selector_map = ranked_selector_maps_[rank];
    MUNDY_THROW_ASSERT(contains(rank, selector), std::logic_error,
                       "Attempting to access a rank and selector pair into AnyRankSelectorMap that isn't present");
    return std::any_cast<const T&>(selector_map.at(selector));
  }

 private:
  using selector_map_t = std::map<stk::mesh::Selector, std::any>;
  selector_map_t ranked_selector_maps_[stk::topology::NUM_RANKS];
};

template <size_t NumEntities, typename DriverType>
class IntermediaryEntityArray {
 public:
  static constexpr size_t num_entities = NumEntities;

  KOKKOS_INLINE_FUNCTION
  IntermediaryEntityArray(const Kokkos::Array<stk::mesh::EntityRank, NumEntities>& ranks, const DriverType* driver)
      : ranks_(ranks), driver_(driver) {
  }

  template <size_t Ord>
  KOKKOS_INLINE_FUNCTION EntityExpr<NumEntities, Ord, DriverType> get() const;

  const DriverType* driver() const {
    return driver_;
  }

 private:
  Kokkos::Array<stk::mesh::EntityRank, NumEntities> ranks_;
  const DriverType* driver_;
};

template <typename DriverType>
class EntityPair {
 public:
  static constexpr size_t num_entities = 2;

  KOKKOS_INLINE_FUNCTION
  EntityPair(const stk::mesh::EntityRank& first_rank, const stk::mesh::EntityRank& second_rank,
             const DriverType* driver)
      : first_rank_(first_rank), second_rank_(second_rank), driver_(driver) {
  }

  template <size_t Ord>
  KOKKOS_INLINE_FUNCTION EntityExpr<2, Ord, DriverType> get() const;

  KOKKOS_INLINE_FUNCTION
  EntityExpr<2, 0, DriverType> first() const;

  KOKKOS_INLINE_FUNCTION
  EntityExpr<2, 1, DriverType> second() const;

  const DriverType* driver() const {
    return driver_;
  }

 private:
  stk::mesh::EntityRank first_rank_;
  stk::mesh::EntityRank second_rank_;
  const DriverType* driver_;
};

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRTYPES_HPP_
