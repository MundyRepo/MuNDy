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

#ifndef MUNDY_MESH_IMPL_NGPACCESSOREXPRDRIVERS_HPP_
#define MUNDY_MESH_IMPL_NGPACCESSOREXPRDRIVERS_HPP_

/// \file NgpAccessorExprDrivers.hpp
/// \brief NGP expression kernel drivers: NgpForEachEntityExprDriver and NgpForEachEntityPairExprDriver.

#include <mundy_mesh/impl/NgpAccessorExprCachable.hpp>
#include <mundy_mesh/impl/NgpAccessorExprEntityBase.hpp>
#include <mundy_mesh/impl/NgpAccessorExprEntityExpr.hpp>
#include <mundy_mesh/impl/NgpAccessorExprTypes.hpp>
#include <mundy_utils/StringLiteral.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

namespace mesh {

namespace impl {

template <typename ExecSpace = stk::ngp::ExecSpace>
class NgpForEachEntityExprDriver {
 public:
  NgpForEachEntityExprDriver(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector,
                             stk::mesh::EntityRank rank, const ExecSpace& exec_space = ExecSpace())
      : bulk_data_ptr_(&bulk_data), selector_(selector), rank_(rank), exec_space_(exec_space) {
  }

  // Default copy/move constructor and assignment operator are fine
  NgpForEachEntityExprDriver(const NgpForEachEntityExprDriver&) = default;
  NgpForEachEntityExprDriver(NgpForEachEntityExprDriver&&) = default;
  NgpForEachEntityExprDriver& operator=(const NgpForEachEntityExprDriver&) = default;
  NgpForEachEntityExprDriver& operator=(NgpForEachEntityExprDriver&&) = default;
  virtual ~NgpForEachEntityExprDriver() = default;

  const stk::mesh::BulkData& bulk_data() const {
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::logic_error,
                        "NgpForEachEntityExprDriver has a null BulkData pointer");
    return *bulk_data_ptr_;
  }

  stk::mesh::Selector selector() const {
    return selector_;
  }

  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityRank rank() const {
    return rank_;
  }

  ExecSpace exec_space() const {
    return exec_space_;
  }

  template <typename Expr>
  void run(CachableExprBase<Expr>& expr_base) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    ::mundy::mesh::for_each_entity_run(
        ngp_mesh, rank_, selector_, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index) {
          // Non-cached eval for debugging
          // expr.eval(Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, evaluation_context);

          // Perform the eval
          auto empty_cache = aggregate();
          expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, empty_cache, evaluation_context);
        });
  }

  template <typename Expr, typename ReductionOp>
  void reduce_local(CachableExprBase<Expr>& expr_base, ReductionOp& reduction) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation.
    // Intersect with locally_owned_part so that shared entities (present on this rank
    // but owned by another) are not double-counted across MPI ranks.
    const stk::mesh::Selector owned_selector = selector_ & bulk_data().mesh_meta_data().locally_owned_part();
    using value_type = typename ReductionOp::value_type;
    stk::mesh::for_each_entity_reduce(
        ngp_mesh, rank_, owned_selector, reduction,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index, value_type& value) {
          // Perform the eval
          auto empty_cache = aggregate();
          auto result = expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 1>{entity_index}, empty_cache, evaluation_context);

          // Combine into the reduction
          // To avoid CUDA being CUDA, we must "touch" the reduction
          [[maybe_unused]] auto meaningless_return_to_make_cuda_happy = reduction.reference();
          auto&& val_ref = result.first.get(result.second);
          using val_t = std::remove_cvref_t<decltype(val_ref)>;

          if constexpr (std::is_same_v<val_t, value_type>) {
            // Directly compatible types; just combine
            reduction.join(value, val_ref);
          } else if constexpr (is_scalar_wrapper_v<val_t>) {
            // val is a scalar wrapper; extract the underlying value and combine
            reduction.join(value, val_ref[0]);
          } else {
            // Unknown return type, attempt to use it directly
            reduction.join(value, val_ref);
          }
        });
  }

 private:
  const stk::mesh::BulkData* bulk_data_ptr_;
  stk::mesh::Selector selector_;
  stk::mesh::EntityRank rank_;
  ExecSpace exec_space_;
};

template <typename T>
struct is_ngp_for_each_entity_expr_driver : std::false_type {};

template <typename ExecSpace>
struct is_ngp_for_each_entity_expr_driver<NgpForEachEntityExprDriver<ExecSpace>> : std::true_type {};

template <typename T>
static constexpr bool is_ngp_for_each_entity_expr_driver_v = is_ngp_for_each_entity_expr_driver<std::decay_t<T>>::value;

template <typename PairView, typename FMIExtractor, typename ExecSpace = stk::ngp::ExecSpace>
class NgpForEachEntityPairExprDriver {
 public:
  NgpForEachEntityPairExprDriver(const stk::mesh::BulkData& bulk_data, const PairView& pair_view,
                                 const ExecSpace& exec_space = ExecSpace())
      : bulk_data_ptr_(&bulk_data), pair_view_(pair_view), exec_space_(exec_space) {
  }

  // Default copy/move constructor and assignment operator are fine
  NgpForEachEntityPairExprDriver(const NgpForEachEntityPairExprDriver&) = default;
  NgpForEachEntityPairExprDriver(NgpForEachEntityPairExprDriver&&) = default;
  NgpForEachEntityPairExprDriver& operator=(const NgpForEachEntityPairExprDriver&) = default;
  NgpForEachEntityPairExprDriver& operator=(NgpForEachEntityPairExprDriver&&) = default;
  virtual ~NgpForEachEntityPairExprDriver() = default;

  const stk::mesh::BulkData& bulk_data() const {
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::logic_error,
                        "NgpForEachEntityPairExprDriver has a null BulkData pointer");
    return *bulk_data_ptr_;
  }

  ExecSpace exec_space() const {
    return exec_space_;
  }

  template <typename Expr>
  void run(CachableExprBase<Expr>& expr_base) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    auto pair_view = pair_view_;  // Make a local copy for lambda capture
    Kokkos::parallel_for(
        "NgpForEachEntityPairExprDriver::run", Kokkos::RangePolicy<ExecSpace>(exec_space(), 0, pair_view.extent(0)),
        KOKKOS_LAMBDA(const int i) {
          auto entity_pair = pair_view(i);
          stk::mesh::FastMeshIndex left_fmi = FMIExtractor::get_left_index(entity_pair);
          stk::mesh::FastMeshIndex right_fmi = FMIExtractor::get_right_index(entity_pair);

          // Non-cached eval
          // expr.eval(Kokkos::Array<stk::mesh::FastMeshIndex, 2>{left_fmi, right_fmi}, evaluation_context);

          // Perform the eval
          auto empty_cache = aggregate();
          expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 2>{left_fmi, right_fmi}, empty_cache, evaluation_context);
        });
  }

  template <typename Expr, typename ReductionOp>
  void reduce_local(CachableExprBase<Expr>& expr_base, ReductionOp& reduction) const {
    // Copy to derived expression type for lambda capture
    auto expr = expr_base.self();

    // Sum the counts of each expression in the tree.
    constexpr auto empty_eval_counts = aggregate();
    constexpr auto eval_counts = Expr::template increment_eval_counts<decltype(empty_eval_counts), empty_eval_counts>();

    // Fail fast on host: validate runtime-reuse equivalence before any field synchronization or kernel launch.
    impl::RuntimeReuseValidator runtime_reuse_validator;
    expr.template validate_runtime_reuse<decltype(eval_counts), eval_counts>(runtime_reuse_validator);

    // Get the up-to-date NGP mesh
    stk::mesh::NgpMesh& ngp_mesh = get_updated_ngp_mesh(bulk_data());

    // Sync all fields to the appropriate space and mark modified where necessary
    NgpEvalContext evaluation_context(ngp_mesh);
    expr.propagate_synchronize(evaluation_context);

    // Perform the evaluation
    auto pair_view = pair_view_;  // Make a local copy for lambda capture
    using value_type = typename ReductionOp::value_type;
    Kokkos::parallel_reduce(
        "NgpForEachEntityPairExprDriver::reduce_local",
        Kokkos::RangePolicy<ExecSpace>(exec_space(), 0, pair_view.extent(0)),
        KOKKOS_LAMBDA(const int i, value_type& value) {
          auto entity_pair = pair_view(i);
          stk::mesh::FastMeshIndex left_fmi = FMIExtractor::get_left_index(entity_pair);
          stk::mesh::FastMeshIndex right_fmi = FMIExtractor::get_right_index(entity_pair);

          // Perform the eval
          auto empty_cache = aggregate();
          auto result = expr.template cached_eval<decltype(eval_counts), eval_counts>(
              Kokkos::Array<stk::mesh::FastMeshIndex, 2>{left_fmi, right_fmi}, empty_cache, evaluation_context);

          // Combine into the reduction
          // To avoid CUDA being CUDA, we must "touch" the reduction
          [[maybe_unused]] auto meaningless_return_to_make_cuda_happy = reduction.reference();
          auto&& val_ref = result.first.get(result.second);
          using val_t = std::remove_cvref_t<decltype(val_ref)>;

          if constexpr (std::is_same_v<val_t, value_type>) {
            // Directly compatible types; just combine
            reduction.join(value, val_ref);
          } else if constexpr (is_scalar_wrapper_v<val_t>) {
            // val is a scalar wrapper; extract the underlying value and combine
            reduction.join(value, val_ref[0]);
          } else {
            // Unknown return type, attempt to use it directly
            reduction.join(value, val_ref);
          }
        });
  }

 private:
  const stk::mesh::BulkData* bulk_data_ptr_;
  PairView pair_view_;
  ExecSpace exec_space_;
};

template <typename T>
struct is_ngp_for_each_entity_pair_expr_driver : std::false_type {};

template <typename PairView, typename FMIExtractor, typename ExecSpace>
struct is_ngp_for_each_entity_pair_expr_driver<NgpForEachEntityPairExprDriver<PairView, FMIExtractor, ExecSpace>>
    : std::true_type {};

template <typename T>
static constexpr bool is_ngp_for_each_entity_pair_expr_driver_v =
    is_ngp_for_each_entity_pair_expr_driver<std::decay_t<T>>::value;

template <typename ExecSpace = stk::ngp::ExecSpace>
auto make_entity_expr_impl(stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& selector,
                           const stk::mesh::EntityRank& rank, const ExecSpace& exec_space = ExecSpace()) {
  using driver_t = NgpForEachEntityExprDriver<ExecSpace>;
  using driver_map_t = AnyRankSelectorMap<make_string_literal("NgpExprDrivers")>;
  stk::mesh::MetaData& meta_data = bulk_data.mesh_meta_data();
  driver_map_t* driver_map = const_cast<driver_map_t*>(meta_data.get_attribute<driver_map_t>());
  if (driver_map == nullptr) {
    const driver_map_t* new_driver_map = new driver_map_t();
    driver_map = const_cast<driver_map_t*>(meta_data.declare_attribute_with_delete(new_driver_map));
  }

  const driver_t* driver_ptr;
  if (driver_map->contains(rank, selector)) {
    driver_t& existing_driver = driver_map->at<driver_t>(rank, selector);
    driver_ptr = &existing_driver;
  } else {
    driver_t new_driver(bulk_data, selector, rank, exec_space);
    driver_map->insert<driver_t>(rank, selector, std::move(new_driver));
    const driver_t& inserted_driver = driver_map->at<driver_t>(rank, selector);
    driver_ptr = &inserted_driver;
  }

  return EntityExpr<1, 0, driver_t>(rank, driver_ptr);
}

template <typename PairView, typename FMIExtractor, typename ExecSpace = stk::ngp::ExecSpace>
auto make_pairwise_entity_expr_impl(stk::mesh::BulkData& bulk_data,                                    //
                                    const stk::mesh::EntityRank& left_rank,                            //
                                    const stk::mesh::EntityRank& right_rank,                           //
                                    const PairView& pair_view, const FMIExtractor& /*fmi_extractor*/,  //
                                    const ExecSpace& exec_space = ExecSpace()) {
  using driver_t = NgpForEachEntityPairExprDriver<PairView, FMIExtractor, ExecSpace>;
  using driver_map_t = AnyRankSelectorMap<make_string_literal("NgpPairExprDrivers")>;
  stk::mesh::MetaData& meta_data = bulk_data.mesh_meta_data();
  driver_map_t* driver_map = const_cast<driver_map_t*>(meta_data.get_attribute<driver_map_t>());
  if (driver_map == nullptr) {
    const driver_map_t* new_driver_map = new driver_map_t();
    driver_map = const_cast<driver_map_t*>(meta_data.declare_attribute_with_delete(new_driver_map));
  }

  const driver_t* driver_ptr;
  stk::mesh::EntityRank dummy_rank = stk::topology::NODE_RANK;
  stk::mesh::Selector dummy_selector = stk::mesh::Selector();
  if (driver_map->contains(dummy_rank, dummy_selector)) {
    driver_t& existing_driver = driver_map->at<driver_t>(dummy_rank, dummy_selector);
    driver_ptr = &existing_driver;
  } else {
    driver_t new_driver(bulk_data, pair_view, exec_space);
    driver_map->insert<driver_t>(dummy_rank, dummy_selector, std::move(new_driver));
    const driver_t& inserted_driver = driver_map->at<driver_t>(dummy_rank, dummy_selector);
    driver_ptr = &inserted_driver;
  }

  return EntityPair(left_rank, right_rank, driver_ptr);
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPACCESSOREXPRDRIVERS_HPP_
