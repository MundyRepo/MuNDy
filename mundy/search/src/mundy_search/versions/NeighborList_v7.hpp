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

#ifndef MUNDY_MESH_NEIGHBORLIST_HPP_
#define MUNDY_MESH_NEIGHBORLIST_HPP_

/// \file NeighborList.hpp
/// \brief First-pass Mundy neighbor-list interface sketch.

// C++ core
#include <concepts>     // for std::convertible_to, std::same_as
#include <cstddef>      // for size_t
#include <stdexcept>    // for std::invalid_argument, std::out_of_range
#include <type_traits>  // for std::is_void_v

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_search/BoundingBox.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_math/Vector3.hpp>        // for mundy::Vector3
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace mesh {

/// \concept ExcluderType
/// \brief Specifies a build-time excluder object that can prepare itself for selected target/source chunks.
///
/// Excluders are stored in the builder and prepared during neighbor-list construction. `setup(...)` gives an excluder a
/// chance to cache selector- and mesh-dependent state before backend callbacks apply it to candidate pairs.
template <typename T>
concept ExcluderType =
    requires(T& excluder, const stk::mesh::BulkData& bulk_data, const stk::mesh::Selector& target_selector,
             const stk::mesh::Selector& source_selector) {
      { excluder.setup(bulk_data, target_selector, source_selector) } -> std::same_as<void>;
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
  KOKKOS_INLINE_FUNCTION ExcluderChain<NoExcluder, NewExcluder> exclude(const NewExcluder& excluder) const {
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
  KOKKOS_INLINE_FUNCTION ExcluderChain<ExcluderChain, NextExcluder> exclude(const NextExcluder& next_excluder) const {
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

/// \class NeighborSearchCandidate
/// \brief Non-periodic target/source candidate passed to excluders.
///
/// Candidate objects are produced during search construction and are not final neighbor-list storage. They expose owner
/// accessors as aliases of the normal target/source accessors so excluders can use one owner-based vocabulary for both
/// periodic and non-periodic search.
/// \tparam SizeType Integral type used for local target/source ordinals.
template <typename SizeType = size_t>
class NeighborSearchCandidate {
 public:
  //! \name Aliases
  //@{

  using size_type = SizeType;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NeighborSearchCandidate() = default;

  /// \brief Construct a non-periodic search candidate.
  /// \param target_index [in] Dense target ordinal.
  /// \param source_index [in] Dense source ordinal.
  /// \param target_entity [in] STK target entity.
  /// \param source_entity [in] STK source entity.
  KOKKOS_INLINE_FUNCTION
  NeighborSearchCandidate(size_type target_index, size_type source_index, stk::mesh::Entity target_entity,
                          stk::mesh::Entity source_entity)
      : target_index_(target_index),
        source_index_(source_index),
        target_entity_(target_entity),
        source_entity_(source_entity) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type source_index() const noexcept {
    return source_index_;
  }

  /// \brief Get the target entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const noexcept {
    return target_entity_;
  }

  /// \brief Get the source entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity() const noexcept {
    return source_entity_;
  }

  //@}

 private:
  //! \name Internal members
  //@{

  //! Dense target ordinal.
  size_type target_index_;
  //! Dense source ordinal.
  size_type source_index_;
  //! STK target entity.
  stk::mesh::Entity target_entity_;
  //! STK source entity.
  stk::mesh::Entity source_entity_;
  //@}
};

/// \class PeriodicNeighborSearchCandidate
/// \brief Periodic owner-pair candidate passed to excluders.
///
/// The candidate stores owner ordinals/entities and the source image shift relative to the target image shift. Images
/// are not entities; excluders should reason in terms of owner identity plus relative shift.
/// \tparam ImageShiftType Vector type used for relative image shifts.
/// \tparam SizeType Integral type used for local owner ordinals.
template <typename ImageShiftType, typename SizeType = size_t>
class PeriodicNeighborSearchCandidate {
 public:
  //! \name Aliases
  //@{

  using image_shift_type = ImageShiftType;
  using size_type = SizeType;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicNeighborSearchCandidate() = default;

  /// \brief Construct a periodic search candidate.
  /// \param target_owner_index [in] Dense target owner ordinal.
  /// \param source_owner_index [in] Dense source owner ordinal.
  /// \param target_entity [in] STK target owner entity.
  /// \param source_entity [in] STK source owner entity.
  /// \param relative_image_shift [in] Source image shift minus target image shift.
  KOKKOS_INLINE_FUNCTION
  PeriodicNeighborSearchCandidate(size_type target_owner_index, size_type source_owner_index,
                                  stk::mesh::Entity target_entity, stk::mesh::Entity source_entity,
                                  const image_shift_type& relative_image_shift)
      : target_owner_index_(target_owner_index),
        source_owner_index_(source_owner_index),
        target_entity_(target_entity),
        source_entity_(source_entity),
        relative_image_shift_(relative_image_shift) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_owner_index_;
  }

  /// \brief Get the dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type source_index() const noexcept {
    return source_owner_index_;
  }

  /// \brief Get the target owner entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const noexcept {
    return target_entity_;
  }

  /// \brief Get the source owner entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity() const noexcept {
    return source_entity_;
  }

  /// \brief Get the source image shift relative to the target image shift.
  KOKKOS_INLINE_FUNCTION
  image_shift_type relative_image_shift() const noexcept {
    return relative_image_shift_;
  }

  //@}

 private:
  //! \name Internal members
  //@{

  //! Dense target owner ordinal.
  size_type target_owner_index_;
  //! Dense source owner ordinal.
  size_type source_owner_index_;
  //! STK target owner entity.
  stk::mesh::Entity target_entity_;
  //! STK source owner entity.
  stk::mesh::Entity source_entity_;
  //! Source image shift minus target image shift.
  image_shift_type relative_image_shift_;
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

namespace impl {

/// \class ArborXSearchCandidateFactory
/// \brief Create non-periodic search candidates from ArborX predicate/source matches.
///
/// The factory adapts ArborX callback inputs to Mundy's excluder-candidate interface. It is a build-time helper and
/// does not own final neighbor-list storage.
/// \tparam TargetBoxes Target search-box wrapper type.
/// \tparam SourceBoxes Source search-box wrapper type.
template <typename TargetBoxes, typename SourceBoxes>
class ArborXSearchCandidateFactory {
 public:
  //! \name Aliases
  //@{

  using target_boxes_type = TargetBoxes;
  using source_boxes_type = SourceBoxes;
  using size_type = typename target_boxes_type::size_type;
  using candidate_type = NeighborSearchCandidate<size_type>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ArborXSearchCandidateFactory() = default;

  /// \brief Construct from target and source search boxes.
  /// \param targets [in] Target search boxes.
  /// \param sources [in] Source search boxes.
  KOKKOS_INLINE_FUNCTION
  ArborXSearchCandidateFactory(const target_boxes_type& targets, const source_boxes_type& sources)
      : targets_(targets), sources_(sources) {
  }
  //@}

  //! \name Candidate creation
  //@{

  /// \brief Create a candidate from an ArborX predicate and source primitive ordinal.
  /// \tparam Predicate ArborX predicate type with attached target ordinal.
  /// \param predicate [in] ArborX predicate.
  /// \param source_index [in] Dense source ordinal reported by ArborX.
  template <typename Predicate>
  KOKKOS_INLINE_FUNCTION candidate_type operator()(const Predicate& predicate, size_type source_index) const {
    const size_type target_index = ArborX::getData(predicate);
    return candidate_type(target_index, source_index, targets_.entity(target_index), sources_.entity(source_index));
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Target search boxes.
  target_boxes_type targets_;
  //! Source search boxes.
  source_boxes_type sources_;
  //@}
};

/// \class PeriodicArborXSearchCandidateFactory
/// \brief Create periodic owner-pair candidates from ArborX image-box matches.
///
/// ArborX reports image ordinals. This factory maps them to target/source owner ordinals and computes the relative
/// image shift that filters and final periodic neighbor-list storage need.
/// \tparam TargetBoxes Target periodic search-box wrapper type.
/// \tparam SourceBoxes Source periodic search-box wrapper type.
template <typename TargetBoxes, typename SourceBoxes>
class PeriodicArborXSearchCandidateFactory {
 public:
  //! \name Aliases
  //@{

  using target_boxes_type = TargetBoxes;
  using source_boxes_type = SourceBoxes;
  using size_type = typename target_boxes_type::size_type;
  using image_shift_type = typename target_boxes_type::image_shift_type;
  using candidate_type = PeriodicNeighborSearchCandidate<image_shift_type, size_type>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicArborXSearchCandidateFactory() = default;

  /// \brief Construct from target and source periodic search boxes.
  /// \param targets [in] Target periodic search boxes.
  /// \param sources [in] Source periodic search boxes.
  KOKKOS_INLINE_FUNCTION
  PeriodicArborXSearchCandidateFactory(const target_boxes_type& targets, const source_boxes_type& sources)
      : targets_(targets), sources_(sources) {
  }
  //@}

  //! \name Candidate creation
  //@{

  /// \brief Create a candidate from an ArborX predicate and source image ordinal.
  /// \tparam Predicate ArborX predicate type with attached target image ordinal.
  /// \param predicate [in] ArborX predicate.
  /// \param source_image_index [in] Dense source image ordinal reported by ArborX.
  template <typename Predicate>
  KOKKOS_INLINE_FUNCTION candidate_type operator()(const Predicate& predicate, size_type source_image_index) const {
    const size_type target_image_index = ArborX::getData(predicate);
    const size_type target_owner_index = targets_.owner_index(target_image_index);
    const size_type source_owner_index = sources_.owner_index(source_image_index);
    const image_shift_type relative_image_shift =
        sources_.image_shift(source_image_index) - targets_.image_shift(target_image_index);
    return candidate_type(target_owner_index, source_owner_index, targets_.owner_entity(target_owner_index),
                          sources_.owner_entity(source_owner_index), relative_image_shift);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Target periodic search boxes.
  target_boxes_type targets_;
  //! Source periodic search boxes.
  source_boxes_type sources_;
  //@}
};

/// \class ArborXExcluderCallback
/// \brief ArborX query callback that applies an excluder.
///
/// The callback constructs a Mundy candidate for each ArborX hit and emits the source primitive only when the excluder
/// keeps the candidate.
/// \tparam CandidateFactory Factory that converts ArborX hits to Mundy candidates.
/// \tparam Excluder Excluder applied to each candidate.
template <typename CandidateFactory, ExcluderType Excluder>
class ArborXExcluderCallback {
 public:
  //! \name Aliases
  //@{

  using candidate_factory_type = CandidateFactory;
  using excluder_type = Excluder;
  using size_type = typename candidate_factory_type::size_type;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ArborXExcluderCallback() = default;

  /// \brief Construct from a candidate factory and an excluder.
  /// \param candidate_factory [in] Candidate factory used to adapt ArborX hits.
  /// \param excluder [in] Excluder applied to each candidate.
  KOKKOS_INLINE_FUNCTION
  ArborXExcluderCallback(const candidate_factory_type& candidate_factory, const excluder_type& excluder)
      : candidate_factory_(candidate_factory), excluder_(excluder) {
  }
  //@}

  //! \name ArborX callback interface
  //@{

#if ARBORX_VERSION >= 10799
  /// \brief Filter an ArborX hit for newer ArborX callback signatures.
  /// \tparam Predicate ArborX predicate type.
  /// \tparam Geometry ArborX primitive geometry type.
  /// \tparam OutputFunctor ArborX output functor type.
  /// \param predicate [in] ArborX predicate with attached target ordinal.
  /// \param value_pair [in] ArborX primitive geometry/source-index pair.
  /// \param out [in] ArborX output functor.
  template <typename Predicate, typename Geometry, typename OutputFunctor>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& predicate,
                                         const ArborX::PairValueIndex<Geometry, int>& value_pair,
                                         const OutputFunctor& out) const {
    const size_type source_index = static_cast<size_type>(value_pair.index);
    const auto candidate = candidate_factory_(predicate, source_index);
    if (!excluder_(candidate)) {
      out(value_pair.index);
    }
  }
#else
  /// \brief Filter an ArborX hit for older ArborX callback signatures.
  /// \tparam Predicate ArborX predicate type.
  /// \tparam OutputFunctor ArborX output functor type.
  /// \param predicate [in] ArborX predicate with attached target ordinal.
  /// \param source_index [in] ArborX source primitive ordinal.
  /// \param out [in] ArborX output functor.
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_INLINE_FUNCTION void operator()(const Predicate& predicate, int source_index, const OutputFunctor& out) const {
    const auto candidate = candidate_factory_(predicate, static_cast<size_type>(source_index));
    if (!excluder_(candidate)) {
      out(source_index);
    }
  }
#endif
  //@}

 private:
  //! \name Internal members
  //@{

  //! Candidate factory used to adapt ArborX hits.
  candidate_factory_type candidate_factory_;
  //! Excluder applied to each candidate.
  excluder_type excluder_;
  //@}
};

/// \class ArborXSearchBoxesT
/// \brief Build-time ArborX boxes paired with STK entity identities.
///
/// This object is an input to ArborX neighbor-list construction. It is not the storage model of the final neighbor
/// list. The final list stores target/source entities and neighbor indices, while search boxes remain a construction
/// detail.
/// \tparam MemorySpace Kokkos memory space in which the boxes and entity view live.
template <typename MemorySpace>
class ArborXSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using box_view_t = Kokkos::View<ArborX::Box*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  ArborXSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  ArborXSearchBoxesT(const ArborXSearchBoxesT&) = default;
  ArborXSearchBoxesT(ArborXSearchBoxesT&&) = default;
  ArborXSearchBoxesT& operator=(const ArborXSearchBoxesT&) = default;
  ArborXSearchBoxesT& operator=(ArborXSearchBoxesT&&) = default;

  /// \brief Construct ArborX search boxes from a selector, matching box and entity views.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] ArborX boxes used as primitives or predicates.
  /// \param entities [in] STK entity associated with each search box.
  ArborXSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes, const entity_view_t& entities)
      : selector_(selector), boxes_(boxes), entities_(entities) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == entities_.extent(0), std::invalid_argument,
                       "ArborXSearchBoxesT: boxes and entities must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector used to populate this box set.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  /// \brief Get the number of search boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get a box by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  ArborX::Box box(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "ArborXSearchBoxesT::box index out of range.");
    return boxes_(index);
  }

  /// \brief Get an entity by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity entity(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "ArborXSearchBoxesT::entity index out of range.");
    return entities_(index);
  }

  /// \brief Get the raw box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t entities() const noexcept {
    return entities_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! ArborX boxes used during construction.
  box_view_t boxes_;
  //! STK entities associated one-to-one with `boxes_`.
  entity_view_t entities_;
  //@}
};

/// \class PeriodicArborXSearchBoxesT
/// \brief Build-time ArborX boxes for periodic images of STK owner entities.
///
/// A periodic image is not a mesh entity. Each image box stores an owner ordinal into `owner_entities_` and a shift
/// vector describing the translation applied to that owner geometry when the box was generated. ArborX sees image boxes
/// during construction; the final periodic neighbor list collapses matches back to owner ordinals and stores only the
/// relative image shift needed by pair kernels.
/// \tparam MemorySpace Kokkos memory space in which the image boxes and metadata live.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar = float>
class PeriodicArborXSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using box_view_t = Kokkos::View<ArborX::Box*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using owner_index_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  PeriodicArborXSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  PeriodicArborXSearchBoxesT(const PeriodicArborXSearchBoxesT&) = default;
  PeriodicArborXSearchBoxesT(PeriodicArborXSearchBoxesT&&) = default;
  PeriodicArborXSearchBoxesT& operator=(const PeriodicArborXSearchBoxesT&) = default;
  PeriodicArborXSearchBoxesT& operator=(PeriodicArborXSearchBoxesT&&) = default;

  /// \brief Construct periodic image search boxes from a selector, owner entities and per-image metadata.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] Search boxes for each periodic image.
  /// \param owner_entities [in] STK owner entities indexed by dense owner ordinal.
  /// \param owner_indices [in] Dense owner ordinal for each image box.
  /// \param image_shifts [in] Translation applied to the owner geometry for each image box.
  PeriodicArborXSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes,
                             const entity_view_t& owner_entities, const owner_index_view_t& owner_indices,
                             const image_shift_view_t& image_shifts)
      : selector_(selector),
        boxes_(boxes),
        owner_entities_(owner_entities),
        owner_indices_(owner_indices),
        image_shifts_(image_shifts) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == owner_indices_.extent(0), std::invalid_argument,
                       "PeriodicArborXSearchBoxesT: boxes and owner_indices must have the same extent.");
    MUNDY_THROW_ASSERT(boxes_.extent(0) == image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicArborXSearchBoxesT: boxes and image_shifts must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector used to populate this box set.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  /// \brief Get the number of periodic image boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get the number of owner entities.
  KOKKOS_INLINE_FUNCTION
  size_type num_owners() const noexcept {
    return owner_entities_.extent(0);
  }

  /// \brief Get a periodic image box by local image ordinal.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  ArborX::Box box(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicArborXSearchBoxesT::box image index out of range.");
    return boxes_(image_index);
  }

  /// \brief Get the owner ordinal associated with a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type owner_index(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicArborXSearchBoxesT::owner_index image index out of range.");
    return owner_indices_(image_index);
  }

  /// \brief Get an owner entity by dense owner ordinal.
  /// \param owner_index [in] Dense owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity owner_entity(size_type owner_index) const {
    MUNDY_THROW_ASSERT(owner_index < num_owners(), std::out_of_range,
                       "PeriodicArborXSearchBoxesT::owner_entity owner index out of range.");
    return owner_entities_(owner_index);
  }

  /// \brief Get the owner entity associated with a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity image_owner_entity(size_type image_index) const {
    return owner_entity(owner_index(image_index));
  }

  /// \brief Get the shift applied to an owner to generate a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  image_shift_type image_shift(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicArborXSearchBoxesT::image_shift image index out of range.");
    return image_shifts_(image_index);
  }

  /// \brief Get the raw periodic image box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t owner_entities() const noexcept {
    return owner_entities_;
  }

  /// \brief Get the raw image-to-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  owner_index_view_t owner_indices() const noexcept {
    return owner_indices_;
  }

  /// \brief Get the raw image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t image_shifts() const noexcept {
    return image_shifts_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! ArborX boxes for periodic images of owner entities.
  box_view_t boxes_;
  //! Owner entities indexed by dense owner ordinal.
  entity_view_t owner_entities_;
  //! Dense owner ordinal for each image box.
  owner_index_view_t owner_indices_;
  //! Translation applied to each owner to generate its image box.
  image_shift_view_t image_shifts_;
  //@}
};

/// \class STKSearchBoxesT
/// \brief Build-time STK search boxes paired with STK entity identities.
///
/// This is the STK coarse-search counterpart to `ArborXSearchBoxesT`. It is a construction input, not persistent
/// neighbor-list storage.
/// \tparam MemorySpace Kokkos memory space in which the boxes and entity view live.
template <typename MemorySpace>
class STKSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using size_type = size_t;
  using box_type = stk::search::Box<double>;
  using box_view_t = Kokkos::View<box_type*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  STKSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  STKSearchBoxesT(const STKSearchBoxesT&) = default;
  STKSearchBoxesT(STKSearchBoxesT&&) = default;
  STKSearchBoxesT& operator=(const STKSearchBoxesT&) = default;
  STKSearchBoxesT& operator=(STKSearchBoxesT&&) = default;

  /// \brief Construct STK search boxes from a selector, matching box and entity views.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] STK boxes used for coarse search.
  /// \param entities [in] STK entity associated with each search box.
  STKSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes, const entity_view_t& entities)
      : selector_(selector), boxes_(boxes), entities_(entities) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == entities_.extent(0), std::invalid_argument,
                       "STKSearchBoxesT: boxes and entities must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector used to populate this box set.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  /// \brief Get the number of search boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get a box by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  box_type box(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "STKSearchBoxesT::box index out of range.");
    return boxes_(index);
  }

  /// \brief Get an entity by local search ordinal.
  /// \param index [in] Local search ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity entity(size_type index) const {
    MUNDY_THROW_ASSERT(index < size(), std::out_of_range, "STKSearchBoxesT::entity index out of range.");
    return entities_(index);
  }

  /// \brief Get the raw box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t entities() const noexcept {
    return entities_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! STK boxes used during construction.
  box_view_t boxes_;
  //! STK entities associated one-to-one with `boxes_`.
  entity_view_t entities_;
  //@}
};

/// \class PeriodicSTKSearchBoxesT
/// \brief Build-time STK search boxes for periodic images of STK owner entities.
///
/// This is the STK coarse-search counterpart to `PeriodicArborXSearchBoxesT`. It expands owner entities into image
/// boxes for construction, while preserving the owner mapping needed to collapse search results back to owner entities.
/// \tparam MemorySpace Kokkos memory space in which the image boxes and metadata live.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar = float>
class PeriodicSTKSearchBoxesT {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using box_type = stk::search::Box<double>;
  using box_view_t = Kokkos::View<box_type*, memory_space>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using owner_index_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  PeriodicSTKSearchBoxesT() = default;

  /// \brief Default copy and move constructors/operators.
  PeriodicSTKSearchBoxesT(const PeriodicSTKSearchBoxesT&) = default;
  PeriodicSTKSearchBoxesT(PeriodicSTKSearchBoxesT&&) = default;
  PeriodicSTKSearchBoxesT& operator=(const PeriodicSTKSearchBoxesT&) = default;
  PeriodicSTKSearchBoxesT& operator=(PeriodicSTKSearchBoxesT&&) = default;

  /// \brief Construct periodic image search boxes from a selector, owner entities and per-image metadata.
  /// \param selector [in] Selector used to populate this box set.
  /// \param boxes [in] STK search boxes for each periodic image.
  /// \param owner_entities [in] STK owner entities indexed by dense owner ordinal.
  /// \param owner_indices [in] Dense owner ordinal for each image box.
  /// \param image_shifts [in] Translation applied to the owner geometry for each image box.
  PeriodicSTKSearchBoxesT(const stk::mesh::Selector& selector, const box_view_t& boxes,
                          const entity_view_t& owner_entities, const owner_index_view_t& owner_indices,
                          const image_shift_view_t& image_shifts)
      : selector_(selector),
        boxes_(boxes),
        owner_entities_(owner_entities),
        owner_indices_(owner_indices),
        image_shifts_(image_shifts) {
    MUNDY_THROW_ASSERT(boxes_.extent(0) == owner_indices_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchBoxesT: boxes and owner_indices must have the same extent.");
    MUNDY_THROW_ASSERT(boxes_.extent(0) == image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchBoxesT: boxes and image_shifts must have the same extent.");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the selector used to populate this box set.
  const stk::mesh::Selector& selector() const noexcept {
    return selector_;
  }

  /// \brief Get the number of periodic image boxes.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return boxes_.extent(0);
  }

  /// \brief Get the number of owner entities.
  KOKKOS_INLINE_FUNCTION
  size_type num_owners() const noexcept {
    return owner_entities_.extent(0);
  }

  /// \brief Get a periodic image box by local image ordinal.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  box_type box(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::box image index out of range.");
    return boxes_(image_index);
  }

  /// \brief Get the owner ordinal associated with a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type owner_index(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::owner_index image index out of range.");
    return owner_indices_(image_index);
  }

  /// \brief Get an owner entity by dense owner ordinal.
  /// \param owner_index [in] Dense owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity owner_entity(size_type owner_index) const {
    MUNDY_THROW_ASSERT(owner_index < num_owners(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::owner_entity owner index out of range.");
    return owner_entities_(owner_index);
  }

  /// \brief Get the owner entity associated with a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity image_owner_entity(size_type image_index) const {
    return owner_entity(owner_index(image_index));
  }

  /// \brief Get the shift applied to an owner to generate a periodic image box.
  /// \param image_index [in] Local periodic-image ordinal.
  KOKKOS_INLINE_FUNCTION
  image_shift_type image_shift(size_type image_index) const {
    MUNDY_THROW_ASSERT(image_index < size(), std::out_of_range,
                       "PeriodicSTKSearchBoxesT::image_shift image index out of range.");
    return image_shifts_(image_index);
  }

  /// \brief Get the raw periodic image box view.
  KOKKOS_INLINE_FUNCTION
  box_view_t boxes() const noexcept {
    return boxes_;
  }

  /// \brief Get the raw owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t owner_entities() const noexcept {
    return owner_entities_;
  }

  /// \brief Get the raw image-to-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  owner_index_view_t owner_indices() const noexcept {
    return owner_indices_;
  }

  /// \brief Get the raw image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t image_shifts() const noexcept {
    return image_shifts_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector used to populate this box set.
  stk::mesh::Selector selector_;
  //! STK search boxes for periodic images of owner entities.
  box_view_t boxes_;
  //! Owner entities indexed by dense owner ordinal.
  entity_view_t owner_entities_;
  //! Dense owner ordinal for each image box.
  owner_index_view_t owner_indices_;
  //! Translation applied to each owner to generate its image box.
  image_shift_view_t image_shifts_;
  //@}
};

/// \class ExcludeSymmetricDuplicates
/// \brief Builder-prepared excluder that suppresses one orientation of symmetric target/source pairs.
///
/// Handles all three selector-relationship cases (identical, disjoint, overlapping) with one type.
/// Constructed by NeighborListBuilder::exclude_symmetric_duplicates(); do not construct directly.
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

}  // namespace impl

/// \concept NeighborListInputType
/// \brief Specifies a selected source or target chunk used to build a neighbor list.
///
/// Search boxes, periodic image boxes, and future source/target input types may have different geometry and indexing
/// APIs, but they must expose the selector that defines the semantic entity chunk being searched.
template <typename T>
concept NeighborListInputType = requires(const T& input) {
  { input.selector() } -> std::same_as<const stk::mesh::Selector&>;
};

/// \class Neighbors
/// \brief Lightweight neighbor-range view for one target.
///
/// `Neighbors` stores the concrete list and a dense target ordinal. This deliberately keeps the first-pass interface
/// simple. Periodic concrete list types can forward relative image shifts without requiring the common range to carry
/// image state. A future non-contiguous list should introduce its own handle-aware facade when the real use case
/// appears.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class Neighbors {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using size_type = typename neighbor_list_type::size_type;
  using source_index_type = typename neighbor_list_type::source_index_type;
  //@}

  //! \brief Forward iterator over neighbor entities for use in range-for on host and device.
  class EntityIterator {
   public:
    KOKKOS_DEFAULTED_FUNCTION EntityIterator() = default;

    KOKKOS_INLINE_FUNCTION
    EntityIterator(const neighbor_list_type& list, size_type target_index, size_type neighbor_ordinal)
        : list_(list), target_index_(target_index), neighbor_ordinal_(neighbor_ordinal) {
    }

    KOKKOS_INLINE_FUNCTION
    stk::mesh::Entity operator*() const {
      return list_.get_neighbor(target_index_, neighbor_ordinal_);
    }

    KOKKOS_INLINE_FUNCTION
    EntityIterator& operator++() {
      ++neighbor_ordinal_;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    EntityIterator operator++(int) {
      auto tmp = *this;
      ++neighbor_ordinal_;
      return tmp;
    }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const EntityIterator& other) const {
      return neighbor_ordinal_ == other.neighbor_ordinal_;
    }

    KOKKOS_INLINE_FUNCTION
    bool operator!=(const EntityIterator& other) const {
      return neighbor_ordinal_ != other.neighbor_ordinal_;
    }

   private:
    neighbor_list_type list_;
    size_type target_index_ = 0;
    size_type neighbor_ordinal_ = 0;
  };

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  Neighbors() = default;

  /// \brief Construct a neighbor range for a target.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  Neighbors(const neighbor_list_type& list, size_type target_index) : list_(list), target_index_(target_index) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of neighbors for the target.
  KOKKOS_INLINE_FUNCTION
  size_type size() const {
    return list_.num_neighbors(target_index_);
  }

  /// \brief Get the neighbor entity for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity operator[](size_type neighbor_ordinal) const {
    return list_.get_neighbor(target_index_, neighbor_ordinal);
  }

  /// \brief Get the neighbor entity for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity operator()(size_type neighbor_ordinal) const {
    return (*this)[neighbor_ordinal];
  }

  /// \brief Get the source ordinal for a neighbor ordinal.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type neighbor_ordinal) const {
    return list_.source_index(target_index_, neighbor_ordinal);
  }

  /// \brief Get the target STK entity for this neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const {
    return list_.target_entity(target_index_);
  }

  /// \brief Get the relative periodic image shift for a neighbor ordinal.
  ///
  /// This is a compile-time extension for periodic concrete list types. Non-periodic list types intentionally do not
  /// provide image-shift storage; calling this accessor for them is a design error caught by normal template
  /// instantiation.
  /// \param neighbor_ordinal [in] Ordinal in `[0, size())`.
  KOKKOS_INLINE_FUNCTION
  auto relative_image_shift(size_type neighbor_ordinal) const {
    return list_.relative_image_shift(target_index_, neighbor_ordinal);
  }

  /// \brief Get the dense target ordinal associated with this range.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Return an iterator to the first neighbor entity.
  KOKKOS_INLINE_FUNCTION
  EntityIterator begin() const {
    return EntityIterator(list_, target_index_, 0);
  }

  /// \brief Return a past-the-end iterator over neighbor entities.
  KOKKOS_INLINE_FUNCTION
  EntityIterator end() const {
    return EntityIterator(list_, target_index_, size());
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Dense target ordinal whose neighbor range is being viewed.
  size_type target_index_;
  //@}
};

/// \class NeighborPair
/// \brief Payload passed to pair-iteration functors.
///
/// The payload carries a dense target ordinal and a neighbor ordinal. It exposes source/target entities and source
/// ordinals, but does not expose storage internals such as compact pair ids or dense row slots. Periodic concrete list
/// types may additionally provide a relative image shift through the forwarding `relative_image_shift()` accessor.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
template <typename NeighborListType>
class NeighborPair {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = NeighborListType;
  using size_type = typename neighbor_list_type::size_type;
  using source_index_type = typename neighbor_list_type::source_index_type;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  NeighborPair() = default;

  /// \brief Construct a pair payload.
  /// \param list [in] Concrete neighbor list to view.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal of the source neighbor for the target.
  KOKKOS_INLINE_FUNCTION
  NeighborPair(const neighbor_list_type& list, size_type target_index, size_type neighbor_ordinal)
      : list_(list), target_index_(target_index), neighbor_ordinal_(neighbor_ordinal) {
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the dense target ordinal for this pair.
  KOKKOS_INLINE_FUNCTION
  size_type target_index() const noexcept {
    return target_index_;
  }

  /// \brief Get the dense source ordinal for this pair.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index() const {
    return list_.source_index(target_index_, neighbor_ordinal_);
  }

  /// \brief Get the target STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity() const {
    return list_.target_entity(target_index_);
  }

  /// \brief Get the source STK entity.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity() const {
    return list_.source_entity(source_index());
  }

  /// \brief Get the source image shift relative to the target image shift.
  ///
  /// This accessor forwards to periodic concrete list types. For non-periodic lists, there is deliberately no neutral
  /// fake shift value because that would hide whether a kernel is using periodic geometry.
  KOKKOS_INLINE_FUNCTION
  auto relative_image_shift() const {
    return list_.relative_image_shift(target_index_, neighbor_ordinal_);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Concrete list instance being viewed.
  neighbor_list_type list_;
  //! Dense target ordinal for the pair.
  size_type target_index_;
  //! Ordinal of the source inside the target's neighbor range.
  size_type neighbor_ordinal_;
  //@}
};

namespace impl {

/// \class DeployFunctorOnNeighborPairs
/// \brief Kokkos functor that expands target-parallel work into neighbor-pair callbacks.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `NeighborPair<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnNeighborPairs {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every neighbor pair.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnNeighborPairs(const NeighborListType& list, const Functor& functor) : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for every neighbor of one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
    const size_type num_neighbors = list_.num_neighbors(target_index);
    for (size_type neighbor_ordinal = 0; neighbor_ordinal < num_neighbors; ++neighbor_ordinal) {
      functor_(NeighborPair<NeighborListType>(list_, target_index, neighbor_ordinal));
    }
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list copied into the Kokkos functor.
  NeighborListType list_;
  //! User callback invoked once for each stored neighbor pair.
  Functor functor_;
  //@}
};

/// \class DeployFunctorOnTargetNeighbors
/// \brief Kokkos functor that invokes a target-neighbors callback for each target ordinal.
/// \tparam NeighborListType Concrete neighbor-list implementation type.
/// \tparam Functor User functor callable with `Neighbors<NeighborListType>`.
template <typename NeighborListType, typename Functor>
class DeployFunctorOnTargetNeighbors {
 public:
  //! Size type used by the concrete neighbor list.
  using size_type = typename NeighborListType::size_type;

  /// \brief Construct the deployment functor.
  /// \param list [in] Concrete neighbor list.
  /// \param functor [in] User callback to run for every target.
  KOKKOS_INLINE_FUNCTION
  DeployFunctorOnTargetNeighbors(const NeighborListType& list, const Functor& functor)
      : list_(list), functor_(functor) {
  }

  /// \brief Run the user callback for one target ordinal.
  /// \param target_index [in] Dense target ordinal in `[0, list.num_targets())`.
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type target_index) const {
    functor_(Neighbors<NeighborListType>(list_, target_index));
  }

 private:
  //! \name Internal members
  //@{

  //! Concrete list copied into the Kokkos functor.
  NeighborListType list_;
  //! User callback invoked once for each target-neighbors payload.
  Functor functor_;
  //@}
};

}  // namespace impl

/// \concept NeighborListType
/// \brief Specifies the protocol that all Mundy neighbor-list implementations must satisfy.
///
/// Any type T that satisfies this concept can be used with for_each_neighbor_pair and
/// for_each_target_with_neighbors. Concrete list types are checked against this protocol at
/// every call site, providing better error messages than a static-facade approach.
template <typename T>
concept NeighborListType = requires {
  typename T::size_type;
  typename T::source_index_type;
  typename T::execution_space;
  typename T::memory_space;
} && requires(const T& list, typename T::size_type i, typename T::size_type j) {
  { list.num_targets() } -> std::convertible_to<typename T::size_type>;
  { list.num_sources() } -> std::convertible_to<typename T::size_type>;
  { list.size() } -> std::convertible_to<typename T::size_type>;
  { list.target_selector() } -> std::same_as<const stk::mesh::Selector&>;
  { list.source_selector() } -> std::same_as<const stk::mesh::Selector&>;
  { list.num_neighbors(i) } -> std::convertible_to<typename T::size_type>;
  { list.get_neighbor(i, j) } -> std::same_as<stk::mesh::Entity>;
  { list.target_entity(i) } -> std::same_as<stk::mesh::Entity>;
  { list.source_entity(i) } -> std::same_as<stk::mesh::Entity>;
  { list.source_index(i, j) } -> std::convertible_to<typename T::source_index_type>;
};

/// \brief Run a callback for every stored neighbor pair using the list's default execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam Functor Callback callable with NeighborPair<ListType>.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per stored neighbor pair.
template <NeighborListType ListType, typename Functor>
void for_each_neighbor_pair(const ListType& list, const Functor& functor) {
  typename ListType::execution_space exec_space{};
  for_each_neighbor_pair(exec_space, list, functor);
}

/// \brief Run a callback for every stored neighbor pair using a provided execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam ExecutionSpace Kokkos execution space.
/// \tparam Functor Callback callable with NeighborPair<ListType>.
/// \param exec_space [in] Execution space for the outer target-parallel loop.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per stored neighbor pair.
template <NeighborListType ListType, typename ExecutionSpace, typename Functor>
void for_each_neighbor_pair(const ExecutionSpace& exec_space, const ListType& list, const Functor& functor) {
  using size_type = typename ListType::size_type;
  using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
  impl::DeployFunctorOnNeighborPairs<ListType, Functor> deploy_functor(list, functor);
  Kokkos::parallel_for("mundy::mesh::for_each_neighbor_pair", range_policy_t(exec_space, 0, list.num_targets()),
                       deploy_functor);
}

/// \brief Run a callback for every target and its neighbors using the list's default execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam Functor Callback callable with Neighbors<ListType>.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per target.
template <NeighborListType ListType, typename Functor>
void for_each_target_with_neighbors(const ListType& list, const Functor& functor) {
  typename ListType::execution_space exec_space{};
  for_each_target_with_neighbors(exec_space, list, functor);
}

/// \brief Run a callback for every target and its neighbors using a provided execution space.
/// \tparam ListType Concrete neighbor list type satisfying NeighborListType.
/// \tparam ExecutionSpace Kokkos execution space.
/// \tparam Functor Callback callable with Neighbors<ListType>.
/// \param exec_space [in] Execution space for the target-parallel loop.
/// \param list [in] Concrete neighbor list.
/// \param functor [in] Callback invoked once per target.
template <NeighborListType ListType, typename ExecutionSpace, typename Functor>
void for_each_target_with_neighbors(const ExecutionSpace& exec_space, const ListType& list, const Functor& functor) {
  using size_type = typename ListType::size_type;
  using range_policy_t = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>;
  impl::DeployFunctorOnTargetNeighbors<ListType, Functor> deploy_functor(list, functor);
  Kokkos::parallel_for("mundy::mesh::for_each_target_with_neighbors", range_policy_t(exec_space, 0, list.num_targets()),
                       deploy_functor);
}

/// \class NeighborListBuilder
/// \brief Fluent neighbor-list builder.
///
/// The builder owns selected target/source inputs and construction policy. Each `.exclude(...)` call returns a new
/// builder type whose excluder state contains the previous filtering behavior plus the new predicate. This is the
/// intended extension point for half-list behavior, connected-entity excluders, and user-defined rejection rules.
/// \tparam ListType Concrete neighbor-list type returned by `build()`.
/// \tparam ExecutionSpace Kokkos execution space used by the eventual build.
/// \tparam TargetInput Selected target input type.
/// \tparam SourceInput Selected source input type.
/// \tparam Excluder Excluder type stored by the builder.
template <typename ListType, typename ExecutionSpace,                            //
          NeighborListInputType TargetInput, NeighborListInputType SourceInput,  //
          ExcluderType Excluder = NoExcluder>
class NeighborListBuilder {
 public:
  //! \name Aliases
  //@{

  using neighbor_list_type = ListType;
  using execution_space = ExecutionSpace;
  using target_input_type = TargetInput;
  using source_input_type = SourceInput;
  using excluder_type = Excluder;
  //@}

  //! \name Constructors
  //@{

  /// \brief Construct a builder with the default empty excluder.
  /// \param exec_space [in] Execution space used by the eventual build.
  /// \param target_input [in] Selected target input.
  /// \param source_input [in] Selected source input.
  NeighborListBuilder(const execution_space& exec_space, const target_input_type& target_input,
                      const source_input_type& source_input)
      : exec_space_(exec_space), targets_(target_input), sources_(source_input), excluder_() {
  }

  /// \brief Construct a builder from all construction state.
  /// \param exec_space [in] Execution space used by the eventual build.
  /// \param target_input [in] Selected target input.
  /// \param source_input [in] Selected source input.
  /// \param excluder [in] Excluder stored by the builder.
  NeighborListBuilder(const execution_space& exec_space, const target_input_type& target_input,
                      const source_input_type& source_input, const excluder_type& excluder)
      : exec_space_(exec_space), targets_(target_input), sources_(source_input), excluder_(excluder) {
  }
  //@}

  //! \name Builder modifiers
  //@{

  /// \brief Return a new builder type with an appended excluder.
  /// \tparam NextExcluder Excluder type to append.
  /// \param next_excluder [in] Excluder to append.
  template <typename NextExcluder>
    requires ExcluderType<NextExcluder>
  auto exclude(const NextExcluder& next_excluder) const {
    const auto new_excluder = excluder_.exclude(next_excluder);
    using new_excluder_type = decltype(new_excluder);
    return NeighborListBuilder<neighbor_list_type, execution_space, target_input_type, source_input_type,
                               new_excluder_type>(exec_space_, targets_, sources_, new_excluder);
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the execution space used by the eventual build.
  const execution_space& exec_space() const noexcept {
    return exec_space_;
  }

  /// \brief Get the selected target input.
  const target_input_type& target_input() const noexcept {
    return targets_;
  }

  /// \brief Get the selected source input.
  const source_input_type& source_input() const noexcept {
    return sources_;
  }

  /// \brief Get the selector defining the target chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return targets_.selector();
  }

  /// \brief Get the selector defining the source chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return sources_.selector();
  }

  /// \brief Get the selected target input.
  target_input_type targets() const noexcept {
    return targets_;
  }

  /// \brief Get the selected source input.
  source_input_type sources() const noexcept {
    return sources_;
  }

  /// \brief Get the excluder stored by the builder.
  excluder_type excluder() const noexcept {
    return excluder_;
  }

  /// \brief Return a prepared copy of the excluder.
  /// \param bulk_data [in] STK bulk data used for mesh-dependent excluder setup.
  excluder_type setup_excluder(const stk::mesh::BulkData& bulk_data) const {
    excluder_type prepared_excluder = excluder_;
    prepared_excluder.setup(bulk_data, target_selector(), source_selector());
    return prepared_excluder;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Build the concrete neighbor list.
  ///
  /// Not yet implemented for the generic case. Specialize NeighborListBuilder for your concrete
  /// neighbor list type, or use the appropriate typed factory function.
  neighbor_list_type build(const stk::mesh::BulkData& bulk_data) const {
    const excluder_type prepared_excluder = setup_excluder(bulk_data);
    (void)prepared_excluder;
    static_assert(std::is_void_v<neighbor_list_type>,
                  "NeighborListBuilder::build() is not yet implemented for this list type. "
                  "Specialize NeighborListBuilder for your concrete neighbor list type.");
    return neighbor_list_type{};
  }

  /// \brief Build the concrete neighbor list.
  ///
  /// Excluders require `setup(...)` with mesh and selector state, so concrete build paths should call
  /// `build(bulk_data)` or `setup_excluder(bulk_data)`.
  neighbor_list_type build() const {
    static_assert(std::is_void_v<neighbor_list_type>,
                  "NeighborListBuilder::build() requires STK BulkData so excluders can be set up. "
                  "Call build(bulk_data), or specialize NeighborListBuilder for your concrete neighbor list type.");
    return neighbor_list_type{};
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Execution space used by the eventual build.
  execution_space exec_space_;
  //! Selected target input.
  target_input_type targets_;
  //! Selected source input.
  source_input_type sources_;
  //! Excluder stored by the builder.
  excluder_type excluder_;
  //@}
};

/// \class ArborX1dNeighborList
/// \brief ArborX neighbor list with Cabana-style compressed 1D storage.
///
/// This implementation stores target entities, source entities, a flattened source-index array, and per-target offsets.
/// Search boxes are not retained after construction.
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename MemorySpace = stk::ngp::MemSpace>
class ArborX1dNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using size_type = size_t;
  using source_index_type = size_type;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ArborX1dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList(const ArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList(ArborX1dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList& operator=(const ArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX1dNeighborList& operator=(ArborX1dNeighborList&&) = default;

  /// \brief Construct a list from already-built compressed storage.
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param source_indices [in] Dense source ordinal for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  ArborX1dNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                       const source_index_view_t& source_indices, const offset_view_t& offsets)
      : target_entities_(target_entities),
        source_entities_(source_entities),
        source_indices_(source_indices),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "ArborX1dNeighborList: offsets extent must be num_targets + 1.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable targets.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable sources.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return target_selector_;
  }

  /// \brief Get the selector defining the source chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
  }

  /// \brief Get the total number of stored neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_indices_.extent(0);
  }

  /// \brief Get the number of neighbors for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "ArborX1dNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the source ordinal for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    return source_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the neighbor entity for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target entity for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "ArborX1dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source entity for a source ordinal.
  /// \param source_index [in] Dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "ArborX1dNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw source-index view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_indices() const noexcept {
    return source_indices_;
  }

  /// \brief Get the raw target-offset view.
  KOKKOS_INLINE_FUNCTION
  offset_view_t offsets() const noexcept {
    return offsets_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target chunk.
  stk::mesh::Selector target_selector_;
  //! Selector defining the source chunk.
  stk::mesh::Selector source_selector_;
  //! Target entities indexed by dense target ordinal.
  entity_view_t target_entities_;
  //! Source entities indexed by dense source ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source ordinals for each stored target/source pair.
  source_index_view_t source_indices_;
  //! Per-target offsets into `source_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;

  /// \brief Get the compact storage index for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX1dNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

/// \class PeriodicArborX1dNeighborList
/// \brief ArborX compressed 1D neighbor list whose stored pairs carry relative periodic image shifts.
///
/// Targets and sources are indexed by owner ordinals, not image ordinals. Multiple stored pairs may therefore reference
/// the same source owner with different relative shifts. Kernels should reconstruct shifted source geometry from the
/// source owner fields and `relative_image_shift(target_index, neighbor_ordinal)`.
/// \tparam MemorySpace Kokkos memory space for owned views.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
class PeriodicArborX1dNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using source_index_type = size_type;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicArborX1dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList(const PeriodicArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList(PeriodicArborX1dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList& operator=(const PeriodicArborX1dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX1dNeighborList& operator=(PeriodicArborX1dNeighborList&&) = default;

  /// \brief Construct a periodic list from already-built compressed storage.
  /// \param target_entities [in] Target owner entities indexed by dense target owner ordinal.
  /// \param source_entities [in] Source owner entities indexed by dense source owner ordinal.
  /// \param source_owner_indices [in] Dense source owner ordinal for every stored pair.
  /// \param relative_image_shifts [in] Source image shift minus target image shift for every stored pair.
  /// \param offsets [in] Target owner offsets into `source_owner_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  PeriodicArborX1dNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                               const source_index_view_t& source_owner_indices,
                               const image_shift_view_t& relative_image_shifts, const offset_view_t& offsets)
      : target_entities_(target_entities),
        source_entities_(source_entities),
        source_owner_indices_(source_owner_indices),
        relative_image_shifts_(relative_image_shifts),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "PeriodicArborX1dNeighborList: offsets extent must be num_targets + 1.");
    MUNDY_THROW_ASSERT(source_owner_indices_.extent(0) == relative_image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicArborX1dNeighborList: source_owner_indices and relative_image_shifts must have the "
                       "same extent.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable target owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable source owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target owner chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return target_selector_;
  }

  /// \brief Get the selector defining the source owner chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
  }

  /// \brief Get the total number of stored periodic neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_owner_indices_.extent(0);
  }

  /// \brief Get the number of neighbors for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicArborX1dNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the source owner ordinal for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    return source_owner_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the source image shift relative to the target image shift for a stored pair.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  image_shift_type relative_image_shift(size_type target_index, size_type neighbor_ordinal) const {
    return relative_image_shifts_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the neighbor owner entity for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target owner entity for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicArborX1dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source owner entity for a source owner ordinal.
  /// \param source_index [in] Dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "PeriodicArborX1dNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw flattened source-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_owner_indices() const noexcept {
    return source_owner_indices_;
  }

  /// \brief Get the raw flattened relative-image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t relative_image_shifts() const noexcept {
    return relative_image_shifts_;
  }

  /// \brief Get the raw target-offset view.
  KOKKOS_INLINE_FUNCTION
  offset_view_t offsets() const noexcept {
    return offsets_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target owner chunk.
  stk::mesh::Selector target_selector_;
  //! Selector defining the source owner chunk.
  stk::mesh::Selector source_selector_;
  //! Target owner entities indexed by dense target owner ordinal.
  entity_view_t target_entities_;
  //! Source owner entities indexed by dense source owner ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source owner ordinals for each stored periodic pair.
  source_index_view_t source_owner_indices_;
  //! Flattened source-image shift minus target-image shift for each stored periodic pair.
  image_shift_view_t relative_image_shifts_;
  //! Per-target-owner offsets into `source_owner_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;

  /// \brief Get the compact storage index for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "PeriodicArborX1dNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

/// \class ArborX2dNeighborList
/// \brief ArborX neighbor list with Cabana-style dense 2D per-target storage.
///
/// This implementation stores target entities, source entities, per-target neighbor counts, and dense rows of source
/// ordinals. It does not expose compact pair ids through the generic payload.
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename MemorySpace = stk::ngp::MemSpace>
class ArborX2dNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using size_type = size_t;
  using source_index_type = size_type;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type**, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  ArborX2dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList(const ArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList(ArborX2dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList& operator=(const ArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION ArborX2dNeighborList& operator=(ArborX2dNeighborList&&) = default;

  /// \brief Construct a list from already-built dense storage.
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param neighbor_counts [in] Number of valid entries in each target row.
  /// \param source_indices [in] Dense target-by-neighbor source ordinal view.
  ArborX2dNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                       const count_view_t& neighbor_counts, const source_index_view_t& source_indices)
      : target_entities_(target_entities),
        source_entities_(source_entities),
        neighbor_counts_(neighbor_counts),
        source_indices_(source_indices),
        total_pairs_(0) {
    MUNDY_THROW_ASSERT(neighbor_counts_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "ArborX2dNeighborList: neighbor_counts extent must equal num_targets.");
    MUNDY_THROW_ASSERT(source_indices_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "ArborX2dNeighborList: source_indices row extent must equal num_targets.");
    size_type total = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<execution_space>(0, neighbor_counts_.extent(0)),
        KOKKOS_LAMBDA(size_type i, size_type & partial_sum) { partial_sum += neighbor_counts_(i); }, total);
    total_pairs_ = total;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable targets.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable sources.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return target_selector_;
  }

  /// \brief Get the selector defining the source chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
  }

  /// \brief Get the total number of stored neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return total_pairs_;
  }

  /// \brief Get the allocated row width for each target.
  KOKKOS_INLINE_FUNCTION
  size_type max_neighbors_per_target() const noexcept {
    return source_indices_.extent(1);
  }

  /// \brief Get the number of neighbors for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "ArborX2dNeighborList::num_neighbors target index out of range.");
    return neighbor_counts_(target_index);
  }

  /// \brief Get the source ordinal for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "ArborX2dNeighborList::source_index neighbor ordinal out of range.");
    return source_indices_(target_index, neighbor_ordinal);
  }

  /// \brief Get the neighbor entity for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target entity for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "ArborX2dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source entity for a source ordinal.
  /// \param source_index [in] Dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "ArborX2dNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw per-target neighbor count view.
  KOKKOS_INLINE_FUNCTION
  count_view_t neighbor_counts() const noexcept {
    return neighbor_counts_;
  }

  /// \brief Get the raw dense source-index view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_indices() const noexcept {
    return source_indices_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target chunk.
  stk::mesh::Selector target_selector_;
  //! Selector defining the source chunk.
  stk::mesh::Selector source_selector_;
  //! Target entities indexed by dense target ordinal.
  entity_view_t target_entities_;
  //! Source entities indexed by dense source ordinal.
  entity_view_t source_entities_;
  //! Number of valid entries in each dense target row.
  count_view_t neighbor_counts_;
  //! Dense per-target source ordinals; extent is `num_targets() x max_neighbors_per_target`.
  source_index_view_t source_indices_;
  //! Total number of stored neighbor pairs, computed once at construction.
  size_type total_pairs_;
  //@}
};

/// \class PeriodicArborX2dNeighborList
/// \brief ArborX dense 2D neighbor list whose stored entries carry relative periodic image shifts.
///
/// This layout stores a fixed-width row of source owner ordinals and relative shifts for each target owner. It is
/// useful when downstream kernels prefer dense per-target neighbor rows over compressed storage.
/// \tparam MemorySpace Kokkos memory space for owned views.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
class PeriodicArborX2dNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using source_index_type = size_type;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using count_view_t = Kokkos::View<size_type*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type**, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type**, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicArborX2dNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList(const PeriodicArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList(PeriodicArborX2dNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList& operator=(const PeriodicArborX2dNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicArborX2dNeighborList& operator=(PeriodicArborX2dNeighborList&&) = default;

  /// \brief Construct a periodic list from already-built dense storage.
  /// \param target_entities [in] Target owner entities indexed by dense target owner ordinal.
  /// \param source_entities [in] Source owner entities indexed by dense source owner ordinal.
  /// \param neighbor_counts [in] Number of valid entries in each target owner row.
  /// \param source_owner_indices [in] Dense source owner ordinals in target-by-neighbor rows.
  /// \param relative_image_shifts [in] Relative image shifts in target-by-neighbor rows.
  KOKKOS_INLINE_FUNCTION
  PeriodicArborX2dNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                               const count_view_t& neighbor_counts, const source_index_view_t& source_owner_indices,
                               const image_shift_view_t& relative_image_shifts)
      : target_entities_(target_entities),
        source_entities_(source_entities),
        neighbor_counts_(neighbor_counts),
        source_owner_indices_(source_owner_indices),
        relative_image_shifts_(relative_image_shifts) {
    MUNDY_THROW_ASSERT(neighbor_counts_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "PeriodicArborX2dNeighborList: neighbor_counts extent must equal num_targets.");
    MUNDY_THROW_ASSERT(source_owner_indices_.extent(0) == target_entities_.extent(0), std::invalid_argument,
                       "PeriodicArborX2dNeighborList: source_owner_indices row extent must equal num_targets.");
    MUNDY_THROW_ASSERT(relative_image_shifts_.extent(0) == source_owner_indices_.extent(0) &&
                           relative_image_shifts_.extent(1) == source_owner_indices_.extent(1),
                       std::invalid_argument,
                       "PeriodicArborX2dNeighborList: relative_image_shifts extent must equal source_owner_indices "
                       "extent.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable target owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable source owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target owner chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return target_selector_;
  }

  /// \brief Get the selector defining the source owner chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
  }

  /// \brief Get the total number of stored periodic neighbor pairs.
  ///
  /// This is intentionally a linear scan for this first-pass dense layout. If callers need this frequently, store the
  /// total during construction instead of introducing a compact pair-id abstraction.
  KOKKOS_INLINE_FUNCTION
  size_type size() const {
    size_type total_neighbors = 0;
    for (size_type target_index = 0; target_index < num_targets(); ++target_index) {
      total_neighbors += num_neighbors(target_index);
    }
    return total_neighbors;
  }

  /// \brief Get the allocated row width for each target owner.
  KOKKOS_INLINE_FUNCTION
  size_type max_neighbors_per_target() const noexcept {
    return source_owner_indices_.extent(1);
  }

  /// \brief Get the number of neighbors for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicArborX2dNeighborList::num_neighbors target index out of range.");
    return neighbor_counts_(target_index);
  }

  /// \brief Get the source owner ordinal for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "PeriodicArborX2dNeighborList::source_index neighbor ordinal out of range.");
    return source_owner_indices_(target_index, neighbor_ordinal);
  }

  /// \brief Get the source image shift relative to the target image shift for a stored pair.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  image_shift_type relative_image_shift(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "PeriodicArborX2dNeighborList::relative_image_shift neighbor ordinal out of range.");
    return relative_image_shifts_(target_index, neighbor_ordinal);
  }

  /// \brief Get the neighbor owner entity for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target owner entity for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicArborX2dNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source owner entity for a source owner ordinal.
  /// \param source_index [in] Dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "PeriodicArborX2dNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw per-target-owner neighbor count view.
  KOKKOS_INLINE_FUNCTION
  count_view_t neighbor_counts() const noexcept {
    return neighbor_counts_;
  }

  /// \brief Get the raw dense source-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_owner_indices() const noexcept {
    return source_owner_indices_;
  }

  /// \brief Get the raw dense relative-image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t relative_image_shifts() const noexcept {
    return relative_image_shifts_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target owner chunk.
  stk::mesh::Selector target_selector_;
  //! Selector defining the source owner chunk.
  stk::mesh::Selector source_selector_;
  //! Target owner entities indexed by dense target owner ordinal.
  entity_view_t target_entities_;
  //! Source owner entities indexed by dense source owner ordinal.
  entity_view_t source_entities_;
  //! Number of valid entries in each dense target-owner row.
  count_view_t neighbor_counts_;
  //! Dense per-target source owner ordinals.
  source_index_view_t source_owner_indices_;
  //! Dense per-target source-image shift minus target-image shift values.
  image_shift_view_t relative_image_shifts_;
  //@}
};

/// \class STKSearchNeighborList
/// \brief STK coarse-search neighbor list mapped into Mundy's common access surface.
///
/// This implementation is intended to consume STK coarse-search candidate pairs and materialize the same compressed
/// target-to-source storage shape as `ArborX1dNeighborList`.
/// \tparam MemorySpace Kokkos memory space for owned views.
template <typename MemorySpace = stk::ngp::MemSpace>
class STKSearchNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using size_type = size_t;
  using source_index_type = size_type;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  STKSearchNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList(const STKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList(STKSearchNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList& operator=(const STKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION STKSearchNeighborList& operator=(STKSearchNeighborList&&) = default;

  /// \brief Construct a list from already-built compressed storage.
  /// \param target_entities [in] Target entities indexed by dense target ordinal.
  /// \param source_entities [in] Source entities indexed by dense source ordinal.
  /// \param source_indices [in] Dense source ordinal for every stored pair.
  /// \param offsets [in] Target offsets into `source_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  STKSearchNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                        const source_index_view_t& source_indices, const offset_view_t& offsets)
      : target_entities_(target_entities),
        source_entities_(source_entities),
        source_indices_(source_indices),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "STKSearchNeighborList: offsets extent must be num_targets + 1.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable targets.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable sources.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return target_selector_;
  }

  /// \brief Get the selector defining the source chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
  }

  /// \brief Get the total number of stored neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_indices_.extent(0);
  }

  /// \brief Get the number of neighbors for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "STKSearchNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the source ordinal for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    return source_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the neighbor entity for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target entity for a target ordinal.
  /// \param target_index [in] Dense target ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "STKSearchNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source entity for a source ordinal.
  /// \param source_index [in] Dense source ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "STKSearchNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw source-index view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_indices() const noexcept {
    return source_indices_;
  }

  /// \brief Get the raw target-offset view.
  KOKKOS_INLINE_FUNCTION
  offset_view_t offsets() const noexcept {
    return offsets_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target chunk.
  stk::mesh::Selector target_selector_;
  //! Selector defining the source chunk.
  stk::mesh::Selector source_selector_;
  //! Target entities indexed by dense target ordinal.
  entity_view_t target_entities_;
  //! Source entities indexed by dense source ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source ordinals for each stored target/source pair.
  source_index_view_t source_indices_;
  //! Per-target offsets into `source_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;

  /// \brief Get the compact storage index for a target and neighbor ordinal.
  /// \param target_index [in] Dense target ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "STKSearchNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

/// \class PeriodicSTKSearchNeighborList
/// \brief STK coarse-search neighbor list with compressed owner-pair storage and relative periodic image shifts.
///
/// This implementation is intended to consume periodic STK coarse-search image pairs, collapse them to owner ordinals,
/// and retain one relative image shift for each stored owner pair.
/// \tparam MemorySpace Kokkos memory space for owned views.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
class PeriodicSTKSearchNeighborList {
 public:
  //! \name Aliases
  //@{

  using memory_space = MemorySpace;
  using execution_space = stk::ngp::ExecSpace;
  using image_shift_scalar = ImageShiftScalar;
  using size_type = size_t;
  using source_index_type = size_type;
  using image_shift_type = mundy::Vector3<image_shift_scalar>;
  using entity_view_t = Kokkos::View<stk::mesh::Entity*, memory_space>;
  using source_index_view_t = Kokkos::View<source_index_type*, memory_space>;
  using offset_view_t = Kokkos::View<size_type*, memory_space>;
  using image_shift_view_t = Kokkos::View<image_shift_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicSTKSearchNeighborList() = default;

  /// \brief Default copy and move constructors/operators.
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList(const PeriodicSTKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList(PeriodicSTKSearchNeighborList&&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList& operator=(const PeriodicSTKSearchNeighborList&) = default;
  KOKKOS_DEFAULTED_FUNCTION PeriodicSTKSearchNeighborList& operator=(PeriodicSTKSearchNeighborList&&) = default;

  /// \brief Construct a periodic list from already-built compressed storage.
  /// \param target_entities [in] Target owner entities indexed by dense target owner ordinal.
  /// \param source_entities [in] Source owner entities indexed by dense source owner ordinal.
  /// \param source_owner_indices [in] Dense source owner ordinal for every stored pair.
  /// \param relative_image_shifts [in] Source image shift minus target image shift for every stored pair.
  /// \param offsets [in] Target owner offsets into `source_owner_indices`; extent must be `num_targets + 1`.
  KOKKOS_INLINE_FUNCTION
  PeriodicSTKSearchNeighborList(const entity_view_t& target_entities, const entity_view_t& source_entities,
                                const source_index_view_t& source_owner_indices,
                                const image_shift_view_t& relative_image_shifts, const offset_view_t& offsets)
      : target_entities_(target_entities),
        source_entities_(source_entities),
        source_owner_indices_(source_owner_indices),
        relative_image_shifts_(relative_image_shifts),
        offsets_(offsets) {
    MUNDY_THROW_ASSERT(offsets_.extent(0) == target_entities_.extent(0) + 1, std::invalid_argument,
                       "PeriodicSTKSearchNeighborList: offsets extent must be num_targets + 1.");
    MUNDY_THROW_ASSERT(source_owner_indices_.extent(0) == relative_image_shifts_.extent(0), std::invalid_argument,
                       "PeriodicSTKSearchNeighborList: source_owner_indices and relative_image_shifts must have the "
                       "same extent.");
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Get the number of enumerable target owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_targets() const noexcept {
    return target_entities_.extent(0);
  }

  /// \brief Get the number of enumerable source owners.
  KOKKOS_INLINE_FUNCTION
  size_type num_sources() const noexcept {
    return source_entities_.extent(0);
  }

  /// \brief Get the selector defining the target owner chunk.
  const stk::mesh::Selector& target_selector() const noexcept {
    return target_selector_;
  }

  /// \brief Get the selector defining the source owner chunk.
  const stk::mesh::Selector& source_selector() const noexcept {
    return source_selector_;
  }

  /// \brief Get the total number of stored periodic neighbor pairs.
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept {
    return source_owner_indices_.extent(0);
  }

  /// \brief Get the number of neighbors for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  size_type num_neighbors(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::num_neighbors target index out of range.");
    return offsets_(target_index + 1) - offsets_(target_index);
  }

  /// \brief Get the source owner ordinal for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  source_index_type source_index(size_type target_index, size_type neighbor_ordinal) const {
    return source_owner_indices_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the source image shift relative to the target image shift for a stored pair.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  image_shift_type relative_image_shift(size_type target_index, size_type neighbor_ordinal) const {
    return relative_image_shifts_(pair_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the neighbor owner entity for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity get_neighbor(size_type target_index, size_type neighbor_ordinal) const {
    return source_entity(source_index(target_index, neighbor_ordinal));
  }

  /// \brief Get the target owner entity for a target owner ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity target_entity(size_type target_index) const {
    MUNDY_THROW_ASSERT(target_index < num_targets(), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::target_entity target index out of range.");
    return target_entities_(target_index);
  }

  /// \brief Get the source owner entity for a source owner ordinal.
  /// \param source_index [in] Dense source owner ordinal.
  KOKKOS_INLINE_FUNCTION
  stk::mesh::Entity source_entity(source_index_type source_index) const {
    MUNDY_THROW_ASSERT(source_index < num_sources(), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::source_entity source index out of range.");
    return source_entities_(source_index);
  }

  /// \brief Get the raw target owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t target_entities() const noexcept {
    return target_entities_;
  }

  /// \brief Get the raw source owner entity view.
  KOKKOS_INLINE_FUNCTION
  entity_view_t source_entities() const noexcept {
    return source_entities_;
  }

  /// \brief Get the raw flattened source-owner ordinal view.
  KOKKOS_INLINE_FUNCTION
  source_index_view_t source_owner_indices() const noexcept {
    return source_owner_indices_;
  }

  /// \brief Get the raw flattened relative-image-shift view.
  KOKKOS_INLINE_FUNCTION
  image_shift_view_t relative_image_shifts() const noexcept {
    return relative_image_shifts_;
  }

  /// \brief Get the raw target-offset view.
  KOKKOS_INLINE_FUNCTION
  offset_view_t offsets() const noexcept {
    return offsets_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Selector defining the target owner chunk.
  stk::mesh::Selector target_selector_;
  //! Selector defining the source owner chunk.
  stk::mesh::Selector source_selector_;
  //! Target owner entities indexed by dense target owner ordinal.
  entity_view_t target_entities_;
  //! Source owner entities indexed by dense source owner ordinal.
  entity_view_t source_entities_;
  //! Flattened dense source owner ordinals for each stored periodic pair.
  source_index_view_t source_owner_indices_;
  //! Flattened source-image shift minus target-image shift for each stored periodic pair.
  image_shift_view_t relative_image_shifts_;
  //! Per-target-owner offsets into `source_owner_indices_`; extent is `num_targets() + 1`.
  offset_view_t offsets_;

  /// \brief Get the compact storage index for a target owner and neighbor ordinal.
  /// \param target_index [in] Dense target owner ordinal.
  /// \param neighbor_ordinal [in] Ordinal in the target's neighbor range.
  KOKKOS_INLINE_FUNCTION
  size_type pair_index(size_type target_index, size_type neighbor_ordinal) const {
    MUNDY_THROW_ASSERT(neighbor_ordinal < num_neighbors(target_index), std::out_of_range,
                       "PeriodicSTKSearchNeighborList::pair_index neighbor ordinal out of range.");
    return offsets_(target_index) + neighbor_ordinal;
  }
  //@}
};

//! \name Factory declarations
//@{

/// \brief Build a compressed 1D ArborX neighbor list from target and source search boxes.
///
/// Declaration only for this design pass. The eventual definition should run ArborX, apply the selected neighbor
/// semantics, and return list storage containing entities plus source indices. It must not silently return an empty
/// list. Selectors are retrieved from `targets.selector()` and `sources.selector()`.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target search boxes and entity identities.
/// \param sources [in] Source search boxes and entity identities.
/// \param buffer_size [in] Optional ArborX traversal buffer-size hint.
template <typename ExecutionSpace, typename MemorySpace = stk::ngp::MemSpace>
ArborX1dNeighborList<MemorySpace> make_arborx_1d_neighbor_list(const ExecutionSpace& exec_space,
                                                               const impl::ArborXSearchBoxesT<MemorySpace>& targets,
                                                               const impl::ArborXSearchBoxesT<MemorySpace>& sources,
                                                               int buffer_size = 0);

/// \brief Build a dense 2D ArborX neighbor list from target and source search boxes.
///
/// Declaration only for this design pass. The eventual definition should run ArborX's two-pass count/fill flow and
/// return list storage containing entities, per-target counts, and dense source rows. It must not silently return an
/// empty list. Selectors are retrieved from `targets.selector()` and `sources.selector()`.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target search boxes and entity identities.
/// \param sources [in] Source search boxes and entity identities.
/// \param buffer_size [in] Optional maximum-neighbor preallocation guess.
template <typename ExecutionSpace, typename MemorySpace = stk::ngp::MemSpace>
ArborX2dNeighborList<MemorySpace> make_arborx_2d_neighbor_list(const ExecutionSpace& exec_space,
                                                               const impl::ArborXSearchBoxesT<MemorySpace>& targets,
                                                               const impl::ArborXSearchBoxesT<MemorySpace>& sources,
                                                               int buffer_size = 0);

/// \brief Build an STK coarse-search neighbor list from target and source search boxes.
///
/// Declaration only for this design pass. The eventual definition should run `stk::search::coarse_search`, apply the
/// selected neighbor semantics, group by target, and return compressed list storage. It must not silently return an
/// empty list. Selectors are retrieved from `targets.selector()` and `sources.selector()`.
/// \tparam ExecutionSpace Execution-space tag associated with search preparation.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \param exec_space [in] Execution space associated with search preparation.
/// \param targets [in] Target search boxes and entity identities.
/// \param sources [in] Source search boxes and entity identities.
template <typename ExecutionSpace, typename MemorySpace = stk::ngp::MemSpace>
STKSearchNeighborList<MemorySpace> make_stk_search_neighbor_list(const ExecutionSpace& exec_space,
                                                                 const impl::STKSearchBoxesT<MemorySpace>& targets,
                                                                 const impl::STKSearchBoxesT<MemorySpace>& sources);

/// \brief Build a compressed 1D periodic ArborX neighbor list from target and source image boxes.
///
/// Declaration only for this design pass. The eventual definition should run ArborX over periodic image boxes, collapse
/// every match back to target/source owner ordinals, and store `source_image_shift - target_image_shift` for each
/// retained owner pair. Selectors are retrieved from `targets.selector()` and `sources.selector()`.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target periodic image boxes, owner entities, owner ordinals, and image shifts.
/// \param sources [in] Source periodic image boxes, owner entities, owner ordinals, and image shifts.
/// \param buffer_size [in] Optional ArborX traversal buffer-size hint.
template <typename ExecutionSpace, typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
PeriodicArborX1dNeighborList<MemorySpace, ImageShiftScalar> make_periodic_arborx_1d_neighbor_list(
    const ExecutionSpace& exec_space, const impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>& targets,
    const impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>& sources, int buffer_size = 0);

/// \brief Build a dense 2D periodic ArborX neighbor list from target and source image boxes.
///
/// Declaration only for this design pass. The eventual definition should run ArborX's periodic count/fill flow,
/// collapse image matches to owner ordinals, and store a relative image shift in the same dense slot as each source
/// owner ordinal. Selectors are retrieved from `targets.selector()` and `sources.selector()`.
/// \tparam ExecutionSpace Kokkos execution space used for build work.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
/// \param exec_space [in] Execution space used for ArborX build/query work.
/// \param targets [in] Target periodic image boxes, owner entities, owner ordinals, and image shifts.
/// \param sources [in] Source periodic image boxes, owner entities, owner ordinals, and image shifts.
/// \param buffer_size [in] Optional maximum-neighbor preallocation guess.
template <typename ExecutionSpace, typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
PeriodicArborX2dNeighborList<MemorySpace, ImageShiftScalar> make_periodic_arborx_2d_neighbor_list(
    const ExecutionSpace& exec_space, const impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>& targets,
    const impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>& sources, int buffer_size = 0);

/// \brief Build a compressed periodic STK coarse-search neighbor list from target and source image boxes.
///
/// Declaration only for this design pass. The eventual definition should run STK coarse search on image boxes, collapse
/// results to owner ordinals, group them by target owner, and store one relative image shift for each retained pair.
/// Selectors are retrieved from `targets.selector()` and `sources.selector()`.
/// \tparam ExecutionSpace Execution-space tag associated with search preparation.
/// \tparam MemorySpace Kokkos memory space for the returned list.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
/// \param exec_space [in] Execution space associated with search preparation.
/// \param targets [in] Target periodic image boxes, owner entities, owner ordinals, and image shifts.
/// \param sources [in] Source periodic image boxes, owner entities, owner ordinals, and image shifts.
template <typename ExecutionSpace, typename MemorySpace = stk::ngp::MemSpace, typename ImageShiftScalar = float>
PeriodicSTKSearchNeighborList<MemorySpace, ImageShiftScalar> make_periodic_stk_search_neighbor_list(
    const ExecutionSpace& exec_space, const impl::PeriodicSTKSearchBoxesT<MemorySpace, ImageShiftScalar>& targets,
    const impl::PeriodicSTKSearchBoxesT<MemorySpace, ImageShiftScalar>& sources);

//@}

}  // namespace mesh

}  // namespace mundy

namespace ArborX {

/// \struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>, PrimitivesTag>
/// \brief ArborX primitive access traits for Mundy ArborX search boxes.
///
/// This specialization tells ArborX how many source primitives exist and how to fetch the `ArborX::Box` for each source
/// ordinal. The specialization must live in namespace `ArborX`.
/// \tparam MemorySpace Kokkos memory space for the Mundy search boxes.
template <typename MemorySpace>
struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>
#if ARBORX_VERSION < 10799
                    ,
                    PrimitivesTag
#endif
                    > {
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>::size_type;

  /// \brief Get the number of primitives.
  /// \param boxes [in] Mundy ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes) {
    return boxes.size();
  }

  /// \brief Get the primitive box for a source ordinal.
  /// \param boxes [in] Mundy ArborX search boxes.
  /// \param index [in] Source ordinal.
  static KOKKOS_FUNCTION ArborX::Box get(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes,
                                         size_type index) {
    return boxes.box(index);
  }
};

/// \struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>, PredicatesTag>
/// \brief ArborX predicate access traits for Mundy ArborX search boxes.
///
/// This specialization tells ArborX how many target predicates exist and how to convert each target box into an
/// intersection predicate. The attached data is the dense target ordinal used during construction.
/// \tparam MemorySpace Kokkos memory space for the Mundy search boxes.
template <typename MemorySpace>
struct AccessTraits<mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>
#if ARBORX_VERSION < 10799
                    ,
                    PredicatesTag
#endif
                    > {
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>::size_type;

  /// \brief Get the number of predicates.
  /// \param boxes [in] Mundy ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes) {
    return boxes.size();
  }

  /// \brief Get the intersection predicate for a target ordinal.
  /// \param boxes [in] Mundy ArborX search boxes.
  /// \param index [in] Target ordinal to attach as predicate data.
  static KOKKOS_FUNCTION auto get(const mundy::mesh::impl::ArborXSearchBoxesT<MemorySpace>& boxes, size_type index) {
    return ArborX::attach(ArborX::intersects(boxes.box(index)), index);
  }
};

/// \struct AccessTraits<mundy::mesh::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>, PrimitivesTag>
/// \brief ArborX primitive access traits for Mundy periodic ArborX image boxes.
///
/// This specialization exposes periodic image boxes to ArborX. The owner-entity mapping and image shifts remain in the
/// Mundy wrapper and are consumed by the neighbor-list builder after ArborX reports image-image matches.
/// \tparam MemorySpace Kokkos memory space for the Mundy periodic search boxes.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar>
struct AccessTraits<mundy::mesh::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>
#if ARBORX_VERSION < 10799
                    ,
                    PrimitivesTag
#endif
                    > {
  //! Periodic search-box wrapper type.
  using boxes_type = mundy::mesh::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename boxes_type::size_type;

  /// \brief Get the number of primitive image boxes.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const boxes_type& boxes) {
    return boxes.size();
  }

  /// \brief Get the primitive image box for an image ordinal.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  /// \param index [in] Image ordinal.
  static KOKKOS_FUNCTION ArborX::Box get(const boxes_type& boxes, size_type index) {
    return boxes.box(index);
  }
};

/// \struct AccessTraits<mundy::mesh::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>, PredicatesTag>
/// \brief ArborX predicate access traits for Mundy periodic ArborX image boxes.
///
/// The attached data is the dense target image ordinal. Builders must translate that image ordinal to a target owner
/// ordinal and image shift before materializing the final periodic neighbor-list storage.
/// \tparam MemorySpace Kokkos memory space for the Mundy periodic search boxes.
/// \tparam ImageShiftScalar Scalar type used by image-shift vectors.
template <typename MemorySpace, typename ImageShiftScalar>
struct AccessTraits<mundy::mesh::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>
#if ARBORX_VERSION < 10799
                    ,
                    PredicatesTag
#endif
                    > {
  //! Periodic search-box wrapper type.
  using boxes_type = mundy::mesh::impl::PeriodicArborXSearchBoxesT<MemorySpace, ImageShiftScalar>;
  //! Kokkos memory space for the search boxes.
  using memory_space = MemorySpace;
  //! Size type used by the search-box wrapper.
  using size_type = typename boxes_type::size_type;

  /// \brief Get the number of predicate image boxes.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  static KOKKOS_FUNCTION size_type size(const boxes_type& boxes) {
    return boxes.size();
  }

  /// \brief Get the intersection predicate for a target image ordinal.
  /// \param boxes [in] Mundy periodic ArborX search boxes.
  /// \param index [in] Target image ordinal to attach as predicate data.
  static KOKKOS_FUNCTION auto get(const boxes_type& boxes, size_type index) {
    return ArborX::attach(ArborX::intersects(boxes.box(index)), index);
  }
};

}  // namespace ArborX

#endif  // MUNDY_MESH_NEIGHBORLIST_HPP_
