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

#ifndef MUNDY_SEARCH_EXCLUDER_HPP_
#define MUNDY_SEARCH_EXCLUDER_HPP_

/// \file Excluder.hpp
/// \brief ExcluderType concept, excluder implementations, and excluder chaining.

// C++ core
#include <concepts>  // for std::same_as, std::convertible_to

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_search/SearchCandidate.hpp>  // for NeighborSearchCandidate, PeriodicNeighborSearchCandidate
#include <mundy_utils/throw_assert.hpp>      // for MUNDY_THROW_ASSERT

namespace mundy {

namespace search {

/// \concept ExcluderType
/// \brief Specifies a build-time excluder object that can prepare itself for selected target/source chunks.
///
/// Excluders are stored in the builder and prepared during neighbor-list construction. `setup(...)` gives an excluder a
/// chance to cache selector- and mesh-dependent state before backend callbacks apply it to candidate pairs.
/// `operator()` must be callable on at least `NeighborSearchCandidate<size_t>` (checked here as a representative
/// non-periodic candidate type; excluders that also handle periodic candidates do so via their own template overloads).
template <typename T>
concept ExcluderType =
    requires(T& excluder, const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& target_selector,
             const stk::mesh::Selector& source_selector) {
      { excluder.setup(bulk_data, target_selector, source_selector) } -> std::same_as<void>;
    } && requires(const T& excluder, const NeighborSearchCandidate<size_t>& candidate) {
      { excluder(candidate) } -> std::convertible_to<bool>;
    };

// Forward declaration needed by NoExcluder::exclude().
template <typename PriorExcluder, typename Excluder>
class ExcluderChain;

/// \class NoExcluder
/// \brief Empty excluder used as the starting point for neighbor-list builders.
///
/// Excluders are build-time predicates that reject candidate target/source pairs before those pairs are materialized
/// in a neighbor list. `NoExcluder` rejects nothing and provides the first `.exclude(...)` step for type-level
/// chaining.
class NoExcluder {
 public:
  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NoExcluder() = default;
  //@}

  //! \name Setup
  //@{

  /// \brief Prepare the empty excluder.
  /// \param bulk_data [in] Unused bulk data.
  /// \param target_selector [in] Unused target selector.
  /// \param source_selector [in] Unused source selector.
  void setup(const stk::mesh::BulkData& /*bulk_data*/, const stk::mesh::Selector& /*target_selector*/,
             const stk::mesh::Selector& /*source_selector*/) {
  }
  //@}

  //! \name Filtering
  //@{

  /// \brief Return whether a candidate pair should be excluded.
  /// \tparam Candidate Candidate pair type.
  /// \param candidate [in] Candidate pair produced by a search backend.
  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION bool operator()(const Candidate& /*candidate*/) const noexcept {
    return false;
  }

  /// \brief Return a new excluder chain with one appended excluder.
  /// \tparam NewExcluder Excluder type to append.
  /// \param excluder [in] Excluder to append to the chain.
  template <ExcluderType NewExcluder>
  ExcluderChain<NoExcluder, NewExcluder> exclude(const NewExcluder& excluder) const {
    return ExcluderChain<NoExcluder, NewExcluder>(*this, excluder);
  }
  //@}
};

/// \class ExcluderChain
/// \brief Type-level chain of Kokkos-callable excluders.
///
/// Each `.exclude(...)` call returns a new `ExcluderChain` containing the previous filtering behavior plus the newly
/// appended excluder.
///
/// \tparam PriorExcluder Previous excluder type.
/// \tparam Excluder Newly appended excluder type.
template <typename PriorExcluder, typename Excluder>
class ExcluderChain {
  static_assert(ExcluderType<PriorExcluder>, "ExcluderChain requires prior excluder to satisfy ExcluderType.");
  static_assert(ExcluderType<Excluder>, "ExcluderChain requires appended excluder to satisfy ExcluderType.");

 public:
  //! \name Aliases
  //@{

  using prior_excluder_type = PriorExcluder;
  using appended_excluder_type = Excluder;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ExcluderChain() = default;

  /// \brief Construct a chain from a previous excluder and a newly appended excluder.
  /// \param prior_excluder [in] Previous excluder.
  /// \param appended_excluder [in] Newly appended excluder.
  KOKKOS_INLINE_FUNCTION
  ExcluderChain(const prior_excluder_type& prior_excluder, const appended_excluder_type& appended_excluder)
      : prior_excluder_(prior_excluder), appended_excluder_(appended_excluder) {
  }
  //@}

  //! \name Setup
  //@{

  /// \brief Prepare every excluder in the chain for the selected source and target chunks.
  /// \param bulk_data [in] STK bulk data used for mesh-dependent preparation.
  /// \param target_selector [in] Selector defining the target chunk.
  /// \param source_selector [in] Selector defining the source chunk.
  void setup(const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& target_selector,
             const stk::mesh::Selector& source_selector) {
    prior_excluder_.setup(bulk_data, target_selector, source_selector);
    appended_excluder_.setup(bulk_data, target_selector, source_selector);
  }
  //@}

  //! \name Filtering
  //@{

  /// \brief Return whether any excluder in the chain rejects the candidate pair.
  /// \tparam Candidate Candidate pair type.
  /// \param candidate [in] Candidate pair produced by a search backend.
  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION bool operator()(const Candidate& candidate) const {
    return prior_excluder_(candidate) || appended_excluder_(candidate);
  }

  /// \brief Return a new excluder chain with one additional appended excluder.
  /// \tparam NextExcluder Excluder type to append.
  /// \param next_excluder [in] Excluder to append to the chain.
  template <ExcluderType NextExcluder>
  ExcluderChain<ExcluderChain, NextExcluder> exclude(const NextExcluder& next_excluder) const {
    return ExcluderChain<ExcluderChain, NextExcluder>(*this, next_excluder);
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the previous excluder.
  KOKKOS_INLINE_FUNCTION
  prior_excluder_type prior_excluder() const {
    return prior_excluder_;
  }

  /// \brief Get the newly appended excluder.
  KOKKOS_INLINE_FUNCTION
  appended_excluder_type appended_excluder() const {
    return appended_excluder_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Previous excluder.
  prior_excluder_type prior_excluder_;
  //! Newly appended excluder.
  appended_excluder_type appended_excluder_;
  //@}
};

/// \struct ExcludeSelfInteraction
/// \brief Exclude self interactions.
///
/// For non-periodic candidates, self means same owner entity. For periodic candidates, self means same owner entity and
/// zero relative image shift, preserving legitimate interactions with nonzero periodic images of the same owner when a
/// builder chooses to generate them.
struct ExcludeSelfInteraction {
  //! \name Setup
  //@{

  /// \brief Prepare self-interaction excluder for the selected source and target chunks.
  /// \param bulk_data [in] Unused bulk data.
  /// \param target_selector [in] Unused target selector.
  /// \param source_selector [in] Unused source selector.
  void setup(const stk::mesh::BulkData& /*bulk_data*/, const stk::mesh::Selector& /*target_selector*/,
             const stk::mesh::Selector& /*source_selector*/) {
  }
  //@}

  //! \name Filtering
  //@{

  /// \brief Return whether a candidate should be excluded as a self interaction.
  /// \tparam Candidate Candidate pair type.
  /// \param candidate [in] Candidate pair produced by a search backend.
  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION bool operator()(const Candidate& candidate) const {
    return candidate.target_entity() == candidate.source_entity() && relative_shift_is_zero(candidate);
  }
  //@}

 private:
  /// \brief Non-periodic candidates have no image shift, so self is determined by owner identity alone.
  /// \tparam Candidate Non-periodic candidate type.
  /// \param candidate [in] Candidate pair produced by a search backend.
  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION static bool relative_shift_is_zero(const Candidate& /*candidate*/) {
    return true;
  }

  /// \brief Periodic candidates are self interactions only when the relative image shift is zero.
  /// \tparam ImageShiftType Vector type used for relative image shifts.
  /// \tparam SizeType Integral type used for local owner ordinals.
  /// \param candidate [in] Periodic candidate pair produced by a search backend.
  template <typename ImageShiftType, typename SizeType>
  KOKKOS_INLINE_FUNCTION static bool relative_shift_is_zero(
      const PeriodicNeighborSearchCandidate<ImageShiftType, SizeType>& candidate) {
    const ImageShiftType shift = candidate.relative_image_shift();
    using scalar_type = typename ImageShiftType::scalar_t;
    return shift[0] == static_cast<scalar_type>(0) && shift[1] == static_cast<scalar_type>(0) &&
           shift[2] == static_cast<scalar_type>(0);
  }
};

// clang-format off
/* ExcludeConnectedEntities — reserved for future design pass — current implementation is a placeholder.
template <typename Relation>
class ExcludeConnectedEntities {
 public:
  KOKKOS_DEFAULTED_FUNCTION ExcludeConnectedEntities() = default;
  KOKKOS_INLINE_FUNCTION explicit ExcludeConnectedEntities(const Relation& relation) : relation_(relation) {}
  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION bool operator()(const Candidate& candidate) const { return relation_.connected(candidate); }
 private:
  Relation relation_;
};
*/
// clang-format on

/// \class ExcludeSymmetricDuplicates
/// \brief Builder-prepared excluder that suppresses one orientation of symmetric target/source pairs.
///
/// Handles all three selector-relationship cases (identical, disjoint, overlapping) with one type.
class ExcludeSymmetricDuplicates {
 public:
  //! \name Constructors
  //@{

  KOKKOS_DEFAULTED_FUNCTION
  ExcludeSymmetricDuplicates() = default;
  //@}

  //! \name Setup
  //@{

  /// \brief Prepare selector-overlap state for the selected source and target chunks.
  /// \param bulk_data [in] STK bulk data used to walk buckets and compute the intersection mask.
  /// \param target_selector [in] Target-side selector.
  /// \param source_selector [in] Source-side selector.
  void setup(const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& target_selector,
             const stk::mesh::Selector& source_selector);
  //@}

  //! \name Filtering
  //@{

  /// \brief Return whether a candidate pair should be excluded as the suppressed orientation.
  /// \tparam Candidate Candidate pair type.
  /// \param candidate [in] Candidate pair produced by a search backend.
  template <typename Candidate>
  KOKKOS_INLINE_FUNCTION bool operator()(const Candidate& candidate) const {
    stk::mesh::Entity trg_entity = candidate.target_entity();
    stk::mesh::Entity src_entity = candidate.source_entity();
    unsigned trg_bucket_id = ngp_mesh_.fast_mesh_index(trg_entity).bucket_id;
    unsigned src_bucket_id = ngp_mesh_.fast_mesh_index(src_entity).bucket_id;
    MUNDY_THROW_ASSERT(trg_bucket_id < num_buckets_, std::runtime_error,
                       "ExcludeSymmetricDuplicates: target bucket ID out of range.");
    MUNDY_THROW_ASSERT(src_bucket_id < num_buckets_, std::runtime_error,
                       "ExcludeSymmetricDuplicates: source bucket ID out of range.");
    if (bucket_in_intersection_(trg_bucket_id) && bucket_in_intersection_(src_bucket_id)) {
      return src_entity < trg_entity;
    }
    return false;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! NGP mesh used for fast bucket-id lookup on device.
  stk::mesh::NgpMesh ngp_mesh_;
  //! Number of valid slots in `bucket_in_intersection_`.
  unsigned num_buckets_ = 0;
  //! Per-bucket flag: true when the bucket belongs to the target/source selector intersection.
  Kokkos::View<bool*, stk::ngp::MemSpace> bucket_in_intersection_;
  //@}
};

inline void ExcludeSymmetricDuplicates::setup(const stk::mesh::BulkData& bulk_data,
                                              const stk::mesh::Selector& target_selector,
                                              const stk::mesh::Selector& source_selector) {
  ngp_mesh_ = stk::mesh::get_updated_ngp_mesh(bulk_data);
  unsigned max_bucket_id = 0;
  bool any_buckets = false;
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    for (const stk::mesh::Bucket* bucket : bulk_data.buckets(rank)) {
      max_bucket_id = std::max(max_bucket_id, bucket->bucket_id());
      any_buckets = true;
    }
  }
  num_buckets_ = any_buckets ? max_bucket_id + 1 : 0;

  bucket_in_intersection_ = Kokkos::View<bool*, stk::ngp::MemSpace>("mundy_bucket_in_intersection", num_buckets_);
  auto host_mask = Kokkos::create_mirror_view(bucket_in_intersection_);
  Kokkos::deep_copy(host_mask, false);

  const stk::mesh::Selector intersection = target_selector & source_selector;
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    for (const stk::mesh::Bucket* bucket : bulk_data.get_buckets(rank, intersection)) {
      host_mask(bucket->bucket_id()) = true;
    }
  }
  Kokkos::deep_copy(bucket_in_intersection_, host_mask);
}

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_EXCLUDER_HPP_
