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

#ifndef MUNDY_SEARCH_IMPL_ARBORXCALLBACK_HPP_
#define MUNDY_SEARCH_IMPL_ARBORXCALLBACK_HPP_

/// \file impl/ArborXCallback.hpp
/// \brief ArborX candidate factories and query callback with excluder filtering.

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

// Trilinos
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_search/Excluder.hpp>                // for ExcluderType
#include <mundy_search/impl/ArborXSearchBoxes.hpp>  // for ArborXSearchBoxesT, PeriodicArborXSearchBoxesT

namespace mundy {

namespace search {

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
  using candidate_type = mundy::search::NeighborSearchCandidate<size_type>;
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
    return candidate_type(target_index, source_index, targets_.identity(target_index), sources_.identity(source_index));
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
  using memory_space = typename target_boxes_type::memory_space;
  using size_type = typename target_boxes_type::size_type;
  using image_shift_type = typename target_boxes_type::identity_type::shift_type;
  using candidate_type = mundy::search::PeriodicNeighborSearchCandidate<image_shift_type, size_type>;
  using owner_index_view_t = Kokkos::View<size_type*, memory_space>;
  //@}

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  KOKKOS_DEFAULTED_FUNCTION
  PeriodicArborXSearchCandidateFactory() = default;

  /// \brief Construct from target and source periodic search boxes plus the source image→owner-ordinal recovery view.
  ///
  /// Targets carry one image per owner, so a target image ordinal is its owner ordinal. Sources carry many images per
  /// owner, so `source_owner_indices(source_image)` recovers the dense source owner ordinal the final list indexes by.
  /// \param targets [in] Target periodic search boxes.
  /// \param sources [in] Source periodic search boxes.
  /// \param source_owner_indices [in] Dense source owner ordinal for each source image.
  KOKKOS_INLINE_FUNCTION
  PeriodicArborXSearchCandidateFactory(const target_boxes_type& targets, const source_boxes_type& sources,
                                       const owner_index_view_t& source_owner_indices)
      : targets_(targets), sources_(sources), source_owner_indices_(source_owner_indices) {
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
    const auto target_id = targets_.identity(target_image_index);
    const auto source_id = sources_.identity(source_image_index);
    const size_type target_owner_index = target_image_index;  // targets: one image per owner
    const size_type source_owner_index = source_owner_indices_(source_image_index);
    return candidate_type(target_owner_index, source_owner_index, target_id.owner, source_id.owner, target_id.shift,
                          source_id.shift);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  //! Target periodic search boxes.
  target_boxes_type targets_;
  //! Source periodic search boxes.
  source_boxes_type sources_;
  //! Dense source owner ordinal for each source image (image→owner recovery for the final list).
  owner_index_view_t source_owner_indices_;
  //@}
};

/// \class ArborXExcluderCallback
/// \brief ArborX query callback that applies an excluder.
///
/// The callback constructs a Mundy candidate for each ArborX hit and emits the source primitive only when the excluder
/// keeps the candidate.
/// \tparam CandidateFactory Factory that converts ArborX hits to Mundy candidates.
/// \tparam Excluder Excluder applied to each candidate.
template <typename CandidateFactory, mundy::search::ExcluderType Excluder>
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

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX

#endif  // MUNDY_SEARCH_IMPL_ARBORXCALLBACK_HPP_
