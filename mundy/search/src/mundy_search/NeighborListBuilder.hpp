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

#ifndef MUNDY_SEARCH_NEIGHBORLISTBUILDER_HPP_
#define MUNDY_SEARCH_NEIGHBORLISTBUILDER_HPP_

/// \file NeighborListBuilder.hpp
/// \brief Type-state fluent neighbor-list builder and make_neighbor_list_builder factory.

// C++ core
#include <concepts>     // for std::same_as
#include <type_traits>  // for std::is_void_v

// Trilinos
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_search/Excluder.hpp>                // for ExcluderType, NoExcluder
#include <mundy_search/NeighborListBuildTraits.hpp>  // for NeighborListInputType, NeighborListBuildTraits

namespace mundy {

namespace search {

namespace impl {

/// \struct UnsetNeighborListBuilderField
/// \brief Marker used by NeighborListBuilder before a fluent setter supplies a field.
struct UnsetNeighborListBuilderField {};

}  // namespace impl

/// \class NeighborListBuilder
/// \brief Type-state fluent neighbor-list builder.
///
/// A builder starts empty. Calls to `exec_space(...)`, `target_input(...)`, and `source_input(...)` return new builder
/// types carrying the supplied field. Calls to `.exclude(...)` may be made at any point and append excluders without
/// changing the selected inputs. `build(bulk_data)` delegates to `NeighborListBuildTraits<ListType>::build`.
///
/// \par Build arguments
/// Every concrete list type publishes a `NeighborListBuildTraits<ListType>::args_type` struct for its backend-specific
/// parameters (e.g., `buffer_size` for ArborX types). At a call site, pass the args inline using designated aggregate
/// initialization — the type is deduced from the function parameter and need not be spelled out:
/// \code{.cpp}
///   builder.build(bulk_data, {.buffer_size = 16});
/// \endcode
/// When the args type must be named explicitly (pre-declaring, storing, forwarding), use the `build_args_type`
/// alias exposed directly on the builder:
/// \code{.cpp}
///   using MyBuilder = NeighborListBuilder<ArborX1dNeighborList<>>;
///   MyBuilder::build_args_type args{};
///   args.buffer_size = 16;
/// \endcode
/// List types that require no extra parameters (e.g., `STKSearchNeighborList`) have an empty `args_type`; omit
/// the second argument entirely and call `build(bulk_data)`.
///
/// \par Example — ArborX 1D list with an excluder
/// \code{.cpp}
///   // target_boxes and source_boxes are impl::ArborXSearchBoxesT<MemSpace> values already populated.
///   auto list = make_neighbor_list_builder<ArborX1dNeighborList<>>()
///       .exec_space(Kokkos::DefaultExecutionSpace{})
///       .target_input(target_boxes)
///       .source_input(source_boxes)
///       .exclude(ExcludeSelfInteraction{})
///       .build(bulk_data, {.buffer_size = 16});
/// \endcode
///
/// \par Example — STK search list (no build args)
/// \code{.cpp}
///   auto list = make_neighbor_list_builder<STKSearchNeighborList<>>()
///       .exec_space(exec_space)
///       .target_input(stk_target_boxes)
///       .source_input(stk_source_boxes)
///       .build(bulk_data);
/// \endcode
///
/// \tparam ListType Concrete neighbor-list type returned by `build()`.
/// \tparam ExecutionSpace Kokkos execution space used by the eventual build, or an unset marker.
/// \tparam TargetInput Selected target input type, or an unset marker.
/// \tparam SourceInput Selected source input type, or an unset marker.
/// \tparam Excluder Excluder type stored by the builder.
template <typename ListType, typename ExecutionSpace = impl::UnsetNeighborListBuilderField,
          typename TargetInput = impl::UnsetNeighborListBuilderField,
          typename SourceInput = impl::UnsetNeighborListBuilderField, ExcluderType Excluder = NoExcluder>
class NeighborListBuilder {
  static_assert(std::same_as<TargetInput, impl::UnsetNeighborListBuilderField> || NeighborListInputType<TargetInput>,
                "NeighborListBuilder target input must be set with a NeighborListInputType.");
  static_assert(std::same_as<SourceInput, impl::UnsetNeighborListBuilderField> || NeighborListInputType<SourceInput>,
                "NeighborListBuilder source input must be set with a NeighborListInputType.");

 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = ListType;
  using execution_space = ExecutionSpace;
  using target_input_type = TargetInput;
  using source_input_type = SourceInput;
  using excluder_type = Excluder;
  /// \brief Shorthand for the build-specific parameter struct of this list type.
  ///
  /// Equivalent to `NeighborListBuildTraits<neighbor_list_type>::args_type`. Use this alias when the args
  /// type must be named explicitly; at a call site, prefer passing `{.field = value}` inline instead.
  using build_args_type = typename NeighborListBuildTraits<neighbor_list_type>::args_type;
  //@}

  //! \name State
  //@{

  /// \brief Whether the execution space has been supplied.
  static constexpr bool has_exec_space = !std::same_as<execution_space, impl::UnsetNeighborListBuilderField>;
  /// \brief Whether the target input has been supplied.
  static constexpr bool has_target_input = NeighborListInputType<target_input_type>;
  /// \brief Whether the source input has been supplied.
  static constexpr bool has_source_input = NeighborListInputType<source_input_type>;
  /// \brief Whether both selected inputs have been supplied.
  static constexpr bool has_selected_inputs = has_target_input && has_source_input;
  /// \brief Whether all fields required by build() have been supplied.
  static constexpr bool is_complete = has_exec_space && has_selected_inputs;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct an empty builder.
  NeighborListBuilder()
    requires(!has_exec_space && !has_target_input && !has_source_input)
  = default;
  //@}

  //! \name Builder modifiers
  //@{

  /// \brief Return a new builder with the execution space supplied.
  /// \tparam NewExecutionSpace Execution space type.
  /// \param exec_space [in] Execution space used by the eventual build.
  template <typename NewExecutionSpace>
  auto exec_space(const NewExecutionSpace& exec_space) const {
    return NeighborListBuilder<neighbor_list_type, NewExecutionSpace, target_input_type, source_input_type,
                               excluder_type>(exec_space, target_input_, source_input_, excluder_);
  }

  /// \brief Return a new builder with the target input supplied.
  /// \tparam NewTargetInput Selected target input type.
  /// \param target_input [in] Selected target input.
  template <NeighborListInputType NewTargetInput>
  auto target_input(const NewTargetInput& target_input) const {
    return NeighborListBuilder<neighbor_list_type, execution_space, NewTargetInput, source_input_type, excluder_type>(
        exec_space_, target_input, source_input_, excluder_);
  }

  /// \brief Return a new builder with the source input supplied.
  /// \tparam NewSourceInput Selected source input type.
  /// \param source_input [in] Selected source input.
  template <NeighborListInputType NewSourceInput>
  auto source_input(const NewSourceInput& source_input) const {
    return NeighborListBuilder<neighbor_list_type, execution_space, target_input_type, NewSourceInput, excluder_type>(
        exec_space_, target_input_, source_input, excluder_);
  }

  /// \brief Return a new builder type with an appended excluder.
  /// \tparam NextExcluder Excluder type to append.
  /// \param next_excluder [in] Excluder to append.
  template <typename NextExcluder>
    requires ExcluderType<NextExcluder>
  auto exclude(const NextExcluder& next_excluder) const {
    const auto new_excluder = excluder_.exclude(next_excluder);
    using new_excluder_type = decltype(new_excluder);
    return NeighborListBuilder<neighbor_list_type, execution_space, target_input_type, source_input_type,
                               new_excluder_type>(exec_space_, target_input_, source_input_, new_excluder);
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the execution space used by the eventual build.
  const execution_space& exec_space() const noexcept
    requires(has_exec_space)
  {
    return exec_space_;
  }

  /// \brief Get the selected target input.
  const target_input_type& target_input() const noexcept
    requires(has_target_input)
  {
    return target_input_;
  }

  /// \brief Get the selected source input.
  const source_input_type& source_input() const noexcept
    requires(has_source_input)
  {
    return source_input_;
  }

  /// \brief Get the selector defining the target chunk.
  const stk::mesh::Selector& target_selector() const noexcept
    requires(has_target_input)
  {
    return target_input_.selector();
  }

  /// \brief Get the selector defining the source chunk.
  const stk::mesh::Selector& source_selector() const noexcept
    requires(has_source_input)
  {
    return source_input_.selector();
  }

  /// \brief Get the excluder stored by the builder.
  const excluder_type& excluder() const noexcept {
    return excluder_;
  }

  /// \brief Return a prepared copy of the excluder.
  /// \param bulk_data [in] STK bulk data used for mesh-dependent excluder setup.
  excluder_type setup_excluder(const stk::mesh::BulkData& bulk_data) const {
    if constexpr (!has_selected_inputs) {
      static_assert(has_selected_inputs,
                    "NeighborListBuilder::setup_excluder(bulk_data) requires target_input(...) and source_input(...) "
                    "before setup.");
    } else {
      excluder_type prepared_excluder = excluder_;
      prepared_excluder.setup(bulk_data, target_selector(), source_selector());
      return prepared_excluder;
    }

    return excluder_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Build the concrete neighbor list.
  ///
  /// Delegates to `NeighborListBuildTraits<neighbor_list_type>::build(*this, bulk_data, args)`.
  /// Specialize `NeighborListBuildTraits<ListType>` to define build behavior and type-specific parameters
  /// for a new concrete neighbor-list type.
  ///
  /// \param bulk_data [in] STK bulk data used for excluder setup.
  /// \param args [in] Build-specific parameters; defaults to default-constructed `args_type`.
  neighbor_list_type build(const stk::mesh::BulkData& bulk_data, const build_args_type& args = {}) const {
    if constexpr (!is_complete) {
      static_assert(is_complete,
                    "NeighborListBuilder::build(bulk_data) requires exec_space(...), target_input(...), and "
                    "source_input(...) before build.");
    } else {
      return NeighborListBuildTraits<neighbor_list_type>::build(*this, bulk_data, args);
    }
    return neighbor_list_type{};
  }
  //@}

 private:
  template <typename, typename, typename, typename, ExcluderType>
  friend class NeighborListBuilder;

  //! \name Internal constructors
  //@{

  /// \brief Construct a builder from all type-state fields.
  NeighborListBuilder(const execution_space& exec_space, const target_input_type& target_input,
                      const source_input_type& source_input, const excluder_type& excluder)
      : exec_space_(exec_space), target_input_(target_input), source_input_(source_input), excluder_(excluder) {
  }
  //@}

  //! \name Internal members
  //@{

  //! Execution space used by the eventual build, or unset marker.
  execution_space exec_space_;
  //! Selected target input, or unset marker.
  target_input_type target_input_;
  //! Selected source input, or unset marker.
  source_input_type source_input_;
  //! Excluder stored by the builder.
  excluder_type excluder_;
  //@}
};

/// \brief Create an empty fluent builder for a concrete neighbor-list type.
/// \tparam ListType Concrete neighbor-list type returned by `build()`.
template <typename ListType>
NeighborListBuilder<ListType> make_neighbor_list_builder() {
  return NeighborListBuilder<ListType>{};
}

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORLISTBUILDER_HPP_
