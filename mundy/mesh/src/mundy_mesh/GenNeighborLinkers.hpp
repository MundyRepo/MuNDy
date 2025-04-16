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

#ifndef MUNDY_MESH_GENNEIGHBORLINKERS_HPP_
#define MUNDY_MESH_GENNEIGHBORLINKERS_HPP_

/// \file GenerateNeighborLinkers.hpp

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <Kokkos_UnorderedMap.hpp>                // for Kokkos::UnorderedMap
#include <stk_mesh/base/Entity.hpp>               // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>                // for stk::mesh::Field
#include <stk_mesh/base/FindRestriction.hpp>      // for stk::mesh::find_restriction
#include <stk_mesh/base/GetEntities.hpp>          // for stk::mesh::get_selected_entities
#include <stk_mesh/base/GetNgpField.hpp>          // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>           // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>             // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>              // for stk::mesh::NgpMesh
#include <stk_mesh/base/Part.hpp>                 // stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>             // stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>                // for stk::mesh::EntityRank
#include <stk_mesh/baseImpl/PartVectorUtils.hpp>  // for stk::mesh::impl::fill_add_parts_and_supersets
#include <stk_search/BoundingBox.hpp>             // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>            // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>            // for stk::search::KDTREE
#include <stk_util/parallel/Parallel.hpp>         // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_core/NgpView.hpp>                // for mundy::core::NgpView
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives.hpp>             // for mundy::geom::Sphere/mundy::geom::AABB
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>          // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/LinkData.hpp>               // for mundy::mesh::LinkData
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>           // for mundy::mesh::field_copy
#include <mundy_mesh/impl/LinkedBucketConn.hpp>  // for mundy::mesh::impl::LinkedBucketConn
namespace mundy {

namespace mesh {

/// \brief Get the local fast mesh indices for a set of entities as an NgpView.
template <typename OurExecSpace>
core::NgpViewT<stk::mesh::FastMeshIndex *, OurExecSpace> get_local_entity_indices(const stk::mesh::BulkData &bulk_data,
                                                                                  stk::mesh::EntityRank rank,
                                                                                  const stk::mesh::Selector &selector,
                                                                                  const OurExecSpace & /*exec_space*/) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities);

  core::NgpViewT<stk::mesh::FastMeshIndex *, OurExecSpace> ngp_local_entity_indices("local_entity_indices",
                                                                                    local_entities.size());

  Kokkos::parallel_for(stk::ngp::HostRangePolicy(0, local_entities.size()),
                       [&bulk_data, &local_entities, &ngp_local_entity_indices](const int i) {
                         const stk::mesh::MeshIndex &mesh_index = bulk_data.mesh_index(local_entities[i]);
                         ngp_local_entity_indices.view_host()(i) =
                             stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
                       });

  ngp_local_entity_indices.modify_on_host();
  return ngp_local_entity_indices;
}

Kokkos::UnorderedMap<Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>, void, stk::ngp::ExecSpace>
get_linked_neighbors_set(LinkData &link_data, const stk::mesh::Selector &link_selector) {
  auto ngp_link_data = get_updated_ngp_data(link_data);
  auto &ngp_mesh = ngp_link_data.ngp_mesh();

  stk::mesh::Selector selected_links = link_selector & link_data.link_meta_data().universal_link_part();
  size_t num_linkers = stk::mesh::count_selected_entities(
      selected_links, link_data.bulk_data().buckets(link_data.link_meta_data().link_rank()));

  using entity_set_t =
      Kokkos::UnorderedMap<Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>, void, stk::ngp::ExecSpace>;
  entity_set_t linked_neighbors_set(num_linkers);

  link_data.sync_to_device();
  for_each_link_run(
      ngp_link_data, selected_links, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &link_index) {
        stk::mesh::Entity source = ngp_link_data.get_linked_entity(link_index, 0);
        stk::mesh::Entity target = ngp_link_data.get_linked_entity(link_index, 1);
        linked_neighbors_set.insert(Kokkos::make_pair(source, target));
      });

  return linked_neighbors_set;
}

template <typename OurExecSpace>
struct GenNeighborLinksTypeInfo {
  using exec_space_t = OurExecSpace;

  // Host
  using ident_proc_t = stk::search::IdentProc<stk::mesh::EntityId, int>;
  using intersection_t = stk::search::IdentProcIntersection<ident_proc_t, ident_proc_t>;
  using result_view_t = Kokkos::View<intersection_t *, exec_space_t>;

  // Device
  using local_ident_proc_t = stk::search::IdentProc<stk::mesh::FastMeshIndex, int>;
  using local_intersection_t = stk::search::IdentProcIntersection<local_ident_proc_t, local_ident_proc_t>;
  using local_result_view_t = Kokkos::View<local_intersection_t *, exec_space_t>;

  // Box-base search
  using box_ident_proc_t = stk::search::BoxIdentProc<stk::search::Box<double>, ident_proc_t>;
  using search_boxes_view_t = Kokkos::View<box_ident_proc_t *, exec_space_t>;

  // Sphere-base search
  using sphere_ident_proc_t = stk::search::BoxIdentProc<stk::search::Sphere<double>, ident_proc_t>;
  using search_spheres_view_t = Kokkos::View<sphere_ident_proc_t *, exec_space_t>;
};

namespace search_filters {

template <typename ViewType, typename... Predicates>
void filter_view(const ViewType &in_view, ViewType &out_view, Predicates... predicates) {
  size_t count = in_view.extent(0);

  // Prefix sum (exclusive scan) to determine positions for valid results
  Kokkos::View<size_t *> scan_results("scan_results", count + 1);

  Kokkos::parallel_scan(
      "PrefixSum", count, KOKKOS_LAMBDA(size_t i, size_t &update, const bool final) {
        bool keep = (predicates(in_view(i)) && ...);
        if (final) scan_results(i) = update;
        if (keep) update++;
      });

  // Total valid count is the final value of the scan
  size_t total_valid;
  Kokkos::deep_copy(total_valid, Kokkos::subview(scan_results, count));

  // Scatter valid results to the output view
  bool in_place = in_view.data() == out_view.data();
  if (in_place && !std::is_same_v<typename ViewType::memory_traits, Kokkos::MemoryTraits<Kokkos::Unmanaged>>) {
    // In/out views are the same and managed, so we can reorder in place and then truncate.
    //   Unmanaged views can't be resized.
    Kokkos::parallel_for(
        "ScatterValid", count, KOKKOS_LAMBDA(size_t i) {
          bool keep = (predicates(in_view(i)) && ...);
          if (keep) {
            size_t new_index = scan_results(i);
            out_view(new_index) = in_view(i);
          }
        });
    Kokkos::resize(out_view, total_valid);
  } else {
    out_view = ViewType("out_view", total_valid);
    Kokkos::parallel_for(
        "ScatterValid", count, KOKKOS_LAMBDA(size_t i) {
          bool keep = (predicates(in_view(i)) && ...);
          if (keep) {
            size_t new_index = scan_results(i);
            out_view(new_index) = in_view(i);
          }
        });
  }
}

struct ExcludeSelfInteractions {
  using type_info_t = GenNeighborLinksTypeInfo<stk::ngp::ExecSpace>;
  using intersection_t = typename type_info_t::intersection_t;
  ExcludeSelfInteractions(stk::mesh::EntityRank source_rank, stk::mesh::EntityRank target_rank)
      : source_rank_(source_rank), target_rank_(target_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const intersection_t &intersection) const {
    return source_rank_ != target_rank_ || intersection.domainIdentProc.id() != intersection.rangeIdentProc.id();
  }

 private:
  stk::mesh::EntityRank source_rank_;
  stk::mesh::EntityRank target_rank_;
};

struct ExcludeConnectedEntities {
  using type_info_t = GenNeighborLinksTypeInfo<stk::ngp::ExecSpace>;
  using intersection_t = typename type_info_t::intersection_t;
  ExcludeConnectedEntities(const stk::mesh::NgpMesh &ngp_mesh, stk::mesh::EntityRank source_rank,
                           stk::mesh::EntityRank target_rank)
      : ngp_mesh_(ngp_mesh), source_rank_(source_rank), target_rank_(target_rank) {
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const intersection_t &intersection) const {
    if (source_rank_ == target_rank_) {
      return true;  // Cannot have connected entities on the same rank
    }

    // At this point either source_rank_ > target_rank_ or target_rank_ > source_rank_
    // Because STK stored upward and downward connections, we can always check if the source is connected to the target.
    const auto source_connected_entities_of_target_rank =
        ngp_mesh_.get_connected_entities(source_rank_, intersection.domainIdentProc.id(), target_rank_);
    bool is_connected = false;
    for (const auto &connected_entity : source_connected_entities_of_target_rank) {
      if (ngp_mesh_.identifier(connected_entity) == intersection.rangeIdentProc.id()) {
        is_connected = true;
        break;
      }
    }

    return !is_connected;
  }

 private:
  const stk::mesh::NgpMesh &ngp_mesh_;
  stk::mesh::EntityRank source_rank_;
  stk::mesh::EntityRank target_rank_;
};

template <typename OurExecSpace>
class SearchFilterBase {
 public:
  using exec_space_t = OurExecSpace;
  using type_info_t = GenNeighborLinksTypeInfo<exec_space_t>;
  using result_view_t = typename type_info_t::result_view_t;
  virtual ~SearchFilterBase() = default;
  virtual void apply(result_view_t &search_results, result_view_t &filtered_results) = 0;
};

template <typename OurExecSpace, typename... Predicates>
class SearchFilter : public SearchFilterBase<OurExecSpace> {
 public:
  using exec_space_t = OurExecSpace;
  using type_info_t = GenNeighborLinksTypeInfo<exec_space_t>;
  using result_view_t = typename type_info_t::result_view_t;

  SearchFilter(OurExecSpace /*space*/, Predicates... predicates) : predicates_(predicates...) {
  }

  void apply(result_view_t &search_results, result_view_t &filtered_results) override {
    filter_view(search_results, filtered_results, std::get<Predicates>(predicates_)...);
  }

 private:
  std::tuple<Predicates...> predicates_;
};

template <typename OurExecSpace, typename... Predicates>
std::shared_ptr<SearchFilterBase<OurExecSpace>> make_search_filter(OurExecSpace space, Predicates... predicates) {
  return std::make_shared<SearchFilter<OurExecSpace, Predicates...>>(space, predicates...);
}

// Deduction guide
template <typename OurExecSpace, typename... Predicates>
SearchFilter(OurExecSpace, Predicates...) -> SearchFilter<OurExecSpace, Predicates...>;

}  // namespace search_filters

/// \brief A helper class for generating and regenerating neighbor linkers
///
/// Example usage:
/// \code {.cpp}
///   auto gen_links = create_gen_neighbor_links(link_data, search_type::Spheres{}, stk::ngp::ExecSpace{});
///   gen_neighbor_links
///       .set_source_target_rank(stk::topology::ELEM_RANK, stk::topology::ELEM_RANK)                    //
///       .acts_on(source_selector1, target_selector1, source_generator1, target_generator1, link_part1) //
///       .acts_on(source_selector2, target_selector2, source_generator2, target_generator2, link_part2) //
///       .set_enforce_source_target_symmetry(true)  //
///       .set_allow_duplicate_links(true)           //
///       .set_sort_search_results(true)             //
///       .set_search_method(search_method)          //
///       .set_search_buffer(search_buffer)          //
///       .set_parallel_machine(bulk_data.parallel()) //
///       .concretize();
///   gen_neighbor_links.generate();
/// \endcode
template <typename OurExecSpace>
class GenNeighborLinks {
 public:
  //! \name Type aliases
  //@{

  using exec_space_t = OurExecSpace;
  using type_info_t = GenNeighborLinksTypeInfo<exec_space_t>;
  //@}

  //! \name Constructors and destructors
  //@{

  //! \brief Canonical constructor
  GenNeighborLinks(LinkData &link_data, [[maybe_unused]] const OurExecSpace &exec_space)
      : link_data_(link_data),
        parallel_machine_(link_data_.bulk_data().parallel()),
        source_rank_(stk::topology::INVALID_RANK),
        target_rank_(stk::topology::INVALID_RANK),
        concretized_(false),
        enforce_source_target_symmetry_(false),
        allow_duplicate_links_(false),
        sort_search_results_(false),
        has_been_generated_(false),
        search_method_(stk::search::MORTON_LBVH),
        search_filter_ptr_(nullptr),
        search_buffer_(0.0),
        source_search_accumulators_(),
        target_search_accumulators_(),
        source_search_checkers_(),
        target_search_checkers_(),
        specializations_() {
  }

  //! \brief Destructor
  virtual ~GenNeighborLinks() = default;
  //@}

  //! \name Getters
  //@{

  /// \brief Get if concretize has been called
  bool is_concretized() const {
    return concretized_;
  }

  /// \brief Fetch the link data
  const LinkData &get_link_data() const {
    return link_data_;
  }
  LinkData &get_link_data() {
    return link_data_;
  }

  /// \brief Fetch the parallel machine
  stk::ParallelMachine get_parallel_machine() const {
    return parallel_machine_;
  }
  stk::ParallelMachine get_parallel_machine() {
    return parallel_machine_;
  }

  /// \brief Fetch the source rank
  stk::mesh::EntityRank get_source_rank() const {
    return source_rank_;
  }

  /// \brief Fetch the target rank
  stk::mesh::EntityRank get_target_rank() const {
    return target_rank_;
  }

  /// \brief Get if source-target symmetry is enforced
  bool get_enforce_source_target_symmetry() const {
    return enforce_source_target_symmetry_;
  }

  /// \brief Get if duplicate links are allowed
  bool get_allow_duplicate_links() const {
    return allow_duplicate_links_;
  }

  /// \brief Get if we should sort the search results
  bool get_sort_search_results() const {
    return sort_search_results_;
  }

  /// \brief Get the search method
  stk::search::SearchMethod get_search_method() const {
    return search_method_;
  }

  /// \brief Get the search filter
  std::shared_ptr<search_filters::SearchFilterBase<exec_space_t>> get_search_filter() const {
    return search_filter_ptr_;
  }

  /// \brief Get the search buffer
  double get_search_buffer() const {
    return search_buffer_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the parallel machine
  GenNeighborLinks &set_parallel_machine(stk::ParallelMachine parallel_machine) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error, "Cannot set parallel machine after concretization.");
    parallel_machine_ = parallel_machine;
    return *this;
  }

  /// \brief Set the source and target ranks
  GenNeighborLinks &set_source_target_rank(stk::mesh::EntityRank source_rank, stk::mesh::EntityRank target_rank) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error,
                        "Cannot set source and target ranks after concretization.");
    MUNDY_THROW_REQUIRE(source_rank == stk::topology::ELEM_RANK && target_rank == stk::topology::ELEM_RANK,
                        std::runtime_error, "To start, we will only support element to element neighbor searches.");

    source_rank_ = source_rank;
    target_rank_ = target_rank;
    return *this;
  }

  /// \brief Set enforce source-target symmetry
  GenNeighborLinks &set_enforce_source_target_symmetry(bool new_value) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error,
                        "Cannot set enforce source-target symmetry after concretization.");
    enforce_source_target_symmetry_ = new_value;
    return *this;
  }

  /// \brief Set allow duplicate links
  GenNeighborLinks &set_allow_duplicate_links(bool new_value) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error,
                        "Cannot set allow duplicate links after concretization.");
    allow_duplicate_links_ = new_value;
    return *this;
  }

  /// \brief Set sort search results
  GenNeighborLinks &set_sort_search_results(bool new_value) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error, "Cannot set sort search results after concretization.");
    sort_search_results_ = new_value;
    return *this;
  }

  /// \brief Set search method
  GenNeighborLinks &set_search_method(stk::search::SearchMethod search_method) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error, "Cannot set search method after concretization.");
    search_method_ = search_method;
    return *this;
  }

  /// \brief Set search filter directly via a shared pointer
  GenNeighborLinks &set_search_filter(
      std::shared_ptr<search_filters::SearchFilterBase<exec_space_t>> search_filter_ptr) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error, "Cannot set search filter after concretization.");
    search_filter_ptr_ = search_filter_ptr;
    return *this;
  }

  /// \brief Set search buffer
  GenNeighborLinks &set_search_buffer(double search_buffer) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error, "Cannot set search buffer after concretization.");
    search_buffer_ = search_buffer;
    return *this;
  }

  /// \brief Inform us of a set of sources/targets to act on, their search generators, and their link specialization
  template <typename SourceGenerator, typename TargetGenerator>
  GenNeighborLinks &acts_on(const stk::mesh::Selector &source, const stk::mesh::Selector &target,
                            const SourceGenerator &source_generator, const TargetGenerator &target_generator,
                            const stk::mesh::PartVector &link_parts) {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error, "Cannot set source/targets after concretization.");

    // Each pair of objects we act on gets two function objects:
    //   search_accumulator: accumulates the search spheres/boxes for each source/target and
    //     operator()(const ngp_local_entity_indices_t &local_entity_indices, double search_buffer) ->
    //     search_spheres_view_t
    //   search_checker: checks if a rebuild is necessary based on the given search buffer.
    //     operator()(const ngp_local_entity_indices_t &local_entity_indices, const search_spheres_view_t&
    //     old_search_spheres, double search_buffer) -> bool

    source_search_accumulators_.emplace_back(make_search_accumulator(source_generator, source_rank_));
    target_search_accumulators_.emplace_back(make_search_accumulator(target_generator, target_rank_));
    source_search_checkers_.emplace_back(make_search_checker(source_generator));
    target_search_checkers_.emplace_back(make_search_checker(target_generator));
    specializations_.emplace_back(source, target, link_parts);
    return *this;
  }
  //@}

  //! \name Search
  //@{

  /// \brief Concretize the settings. Once set, no more settings can be changed.
  /// Must set source and target rank and add at least one set of sources/targets to act on.
  void concretize() {
    MUNDY_THROW_REQUIRE(!is_concretized(), std::runtime_error, "Cannot concretize more than once.");
    MUNDY_THROW_REQUIRE(source_rank_ != stk::topology::INVALID_RANK, std::runtime_error,
                        "Source rank must be set before calling concretize.");
    MUNDY_THROW_REQUIRE(target_rank_ != stk::topology::INVALID_RANK, std::runtime_error,
                        "Target rank must be set before calling concretize.");
    MUNDY_THROW_REQUIRE(!specializations_.empty(), std::runtime_error, "Must act on at least one source/target pair.");
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");
    concretized_ = true;

    // In the future, concretize will invert the specialization to go from link partition to source/target buckets.
    // When this occurs, we will store one set of local entity indices for each source/target pair per link partition.
  }

  /// \brief Generate neighbor linkers between a given set of source and target spheres
  /// \return True if the search was performed. False if no regeneration was necessary.
  bool generate() {
    MUNDY_THROW_REQUIRE(is_concretized(), std::runtime_error, "Cannot generate links before concretization.");

    if (has_been_generated_) {
      // Check if we need to regenerate
      if (indices_changed()) {
        collect_local_entity_indices();
        collect_search_spheres();
        if (!allow_duplicate_links_) {
          collect_existing_linked_neighbors();  // If the indices have changed this list is invalid
        }
      } else if (objects_moved_too_much()) {
        collect_search_spheres();
      } else {
        return false;  // No search performed
      }
    } else {
      // This is the first call to generate
      collect_local_entity_indices();
      collect_search_spheres();
      if (!allow_duplicate_links_) {
        collect_existing_linked_neighbors();
      }
    }

    perform_search();
    copy_search_results_to_host();
    ghost_neighbors();
    request_link_creation_and_destruction();
    process_requests();

    has_been_generated_ = true;
    return true;  // Performed the search
  }

 protected:
  //! \name Internal types
  //@{

  // Three internal vectors that store the search accumulators, checkers, and specializations
  using ngp_local_entity_indices_t = core::NgpViewT<stk::mesh::FastMeshIndex *, exec_space_t>;
  using search_accumulator_t =
      std::function<typename type_info_t::search_spheres_view_t(ngp_local_entity_indices_t &, double)>;
  using search_checker_t =
      std::function<bool(ngp_local_entity_indices_t &, const typename type_info_t::search_spheres_view_t &, double)>;
  using specialization_t = std::tuple<stk::mesh::Selector, stk::mesh::Selector, stk::mesh::PartVector>;
  using specialization_vec_t = std::vector<specialization_t>;
  //@}

  //! \name Internal functions
  //@{

  /// \brief Make a search accumulator
  template <typename Generator>
  search_accumulator_t make_search_accumulator(const Generator &gen, const stk::mesh::EntityRank rank) {
    LinkData &local_link_data = link_data_;
    return
        [&gen, &local_link_data, rank](const ngp_local_entity_indices_t &local_entity_indices, double search_buffer) {
          auto ngp_link_data = get_updated_ngp_data(local_link_data);
          auto &ngp_mesh = ngp_link_data.ngp_mesh();
          const int parallel_rank = local_link_data.bulk_data().parallel_rank();

          size_t num_local_spheres = local_entity_indices.extent(0);
          typename type_info_t::search_spheres_view_t search_spheres("gen_neighbors_search_spheres", num_local_spheres);

          Kokkos::parallel_for(
              stk::ngp::RangePolicy<exec_space_t>(0, num_local_spheres), KOKKOS_LAMBDA(const unsigned &i) {
                stk::mesh::FastMeshIndex index = local_entity_indices.view_device()(i);
                stk::mesh::Entity entity = ngp_mesh.get_entity(rank, index);
                geom::Sphere<double> sphere = gen(index);
                stk::search::Point<double> center(sphere.center()[0], sphere.center()[1], sphere.center()[2]);
                search_spheres(i) = typename type_info_t::sphere_ident_proc_t{
                    stk::search::Sphere<double>(center, sphere.radius() + search_buffer),
                    typename type_info_t::ident_proc_t(ngp_mesh.identifier(entity), parallel_rank)};
              });

          return search_spheres;
        };
  }

  /// \brief Make a search checker
  template <typename Generator>
  search_checker_t make_search_checker(const Generator &gen) {
    LinkData &local_link_data = link_data_;
    return [&local_link_data, &gen](ngp_local_entity_indices_t &local_entity_indices,
                                    const typename type_info_t::search_spheres_view_t &old_search_spheres,
                                    double search_buffer) -> bool {
      auto ngp_link_data = get_updated_ngp_data(local_link_data);
      auto &ngp_mesh = ngp_link_data.ngp_mesh();

      local_entity_indices.template sync_to<exec_space_t>();
      size_t num_local_spheres = local_entity_indices.extent(0);
      bool moved_too_much = false;
      Kokkos::parallel_reduce(
          "CheckSearchBufferViolation", stk::ngp::HostRangePolicy(0, num_local_spheres),
          KOKKOS_LAMBDA(int i, bool &local_result) {
            stk::mesh::FastMeshIndex index = local_entity_indices.view_device()(i);
            auto new_center = gen(index).center();
            auto old_center = old_search_spheres(i).box.center();
            double dx = new_center[0] - old_center[0];
            double dy = new_center[1] - old_center[1];
            double dz = new_center[2] - old_center[2];
            double disp = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
            local_result = local_result || (disp > 0.5 * search_buffer);
          },
          Kokkos::LOr<bool>(moved_too_much));

      return moved_too_much;
    };
  }

  void collect_local_entity_indices() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);
    source_local_entity_indices_ =
        get_local_entity_indices(link_data_.bulk_data(), source_rank_, source_selector, stk::ngp::ExecSpace{});
    target_local_entity_indices_ =
        get_local_entity_indices(link_data_.bulk_data(), target_rank_, target_selector, stk::ngp::ExecSpace{});
  }

  void collect_search_spheres() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);
    const auto &source_search_accumulator = source_search_accumulators_[0];
    const auto &target_search_accumulator = target_search_accumulators_[0];
    source_search_spheres_ = source_search_accumulator(source_local_entity_indices_, search_buffer_);
    target_search_spheres_ = target_search_accumulator(target_local_entity_indices_, search_buffer_);
  }

  void collect_existing_linked_neighbors() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");
    stk::mesh::Selector link_selector = stk::mesh::selectIntersection(std::get<2>(specializations_[0]));
    existing_linked_neighbors_ = get_linked_neighbors_set(link_data_, link_selector);
  }

  void perform_search() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");

    // TODO(palmerb4): Some of these settings don't come into play until after Trilinos 16.0.0
    const bool results_parallel_symmetry = true;    // ensures that the source/target owners know about all pairs
    const bool auto_swap_domain_and_range = false;  // swap source and target if target is owned and source is not
    const bool sort_search_results = sort_search_results_;  // sort the search results by source id
    stk::search::coarse_search(source_search_spheres_, target_search_spheres_, search_method_, parallel_machine_,
                               search_results_, exec_space_t{}, results_parallel_symmetry);

    if (search_filter_ptr_) {
      search_filter_ptr_->apply(search_results_, search_results_);
    }
  }

  void copy_search_results_to_host() {
    host_search_results_ = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search_results_);
  }

  void ghost_neighbors() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");
    BulkData &bulk_data = link_data_.bulk_data();
    if (bulk_data.parallel_size() == 1) {
      return;
    }

    // TODO(palmerb4): Reduce the serial cost of this operation
    //
    // Plan:
    //  On the device, we can create a Kokkos::View of if we need to ghost or not.
    //  we can then partial sum this to get both the total number of ghosted entities and where in the
    //  entities_to_ghost_view we should insert the entities we want ghosted. This gives a view of relatively small size
    //  that we can copy to the host, loop over in serial, and perform get_entity on to get the entities we want
    //  ghosted.
    const int my_parallel_rank = bulk_data.parallel_rank();
    bulk_data.modification_begin();
    stk::mesh::Ghosting &neighbor_ghosting = bulk_data.create_ghosting("MUNDY_NEIGHBOR_LINKERS");
    std::vector<stk::mesh::EntityProc> elements_to_ghost;

    const size_t num_search_results = host_search_results_.size();
    for (size_t i = 0; i < num_search_results; ++i) {
      auto result = host_search_results_(i);
      const bool i_own_source = result.domainIdentProc.proc() == my_parallel_rank;
      const bool i_own_target = result.rangeIdentProc.proc() == my_parallel_rank;
      if (!i_own_source && i_own_target) {
        // Send the target to the source
        stk::mesh::Entity target = bulk_data.get_entity(target_rank_, result.rangeIdentProc.id());
        elements_to_ghost.emplace_back(target, result.domainIdentProc.proc());
      } else if (i_own_source && !i_own_target) {
        // Send the source to the target
        stk::mesh::Entity source = bulk_data.get_entity(source_rank_, result.domainIdentProc.id());
        elements_to_ghost.emplace_back(source, result.rangeIdentProc.proc());
      } else if (!i_own_source && !i_own_target) {
        MUNDY_THROW_REQUIRE(false, std::runtime_error,
                            "Invalid search result. Somehow we received a pair of elements that we don't own.");
      }
    }

    bulk_data.change_ghosting(neighbor_ghosting, elements_to_ghost);
    bulk_data.modification_end();
  }

  void request_link_creation_and_destruction() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");

    // TODO(palmerb4): Reduce the serial cost of this operation
    //
    // Plan:
    //  Before this function or ghost_neighbors is called, perform a single serial loop over the host search results
    //  and get there source and target entities. This way, neither this function nor ghost_neighbors need to call
    //  get_entity and can safely employ host parallel_for loops.

    stk::mesh::PartVector locally_owned_link_parts = std::get<2>(specializations_[0]);
    locally_owned_link_parts.push_back(&link_data_.bulk_data().mesh_meta_data().locally_owned_part());
    LinkPartition &link_partition = link_data_.get_partition(link_data_.get_partition_key(locally_owned_link_parts));
    size_t num_search_results = host_search_results_.size();
    link_partition.increase_request_link_capacity(num_search_results);
    for (size_t i = 0; i < num_search_results; ++i) {
      const auto search_result = host_search_results_(i);
      stk::mesh::Entity source_entity =
          link_data_.bulk_data().get_entity(source_rank_, search_result.domainIdentProc.id());
      stk::mesh::Entity target_entity =
          link_data_.bulk_data().get_entity(target_rank_, search_result.rangeIdentProc.id());
      link_partition.request_link(source_entity, target_entity);
    }
  }

  void process_requests() {
    bool fully_consistent = source_rank_ >= stk::topology::ELEM_RANK && target_rank_ >= stk::topology::ELEM_RANK;
    link_data_.process_requests(fully_consistent);
  }

  bool indices_changed() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");
    BulkData &bulk_data = link_data_.bulk_data();
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);

    auto new_source_local_entity_indices =
        get_local_entity_indices(bulk_data, source_rank_, source_selector, exec_space_t{});
    auto new_target_local_entity_indices =
        get_local_entity_indices(bulk_data, target_rank_, target_selector, exec_space_t{});

    auto source_local_entity_indices = source_local_entity_indices_;
    auto target_local_entity_indices = target_local_entity_indices_;

    new_source_local_entity_indices.template sync_to<exec_space_t>();
    new_target_local_entity_indices.template sync_to<exec_space_t>();
    source_local_entity_indices.template sync_to<exec_space_t>();
    target_local_entity_indices.template sync_to<exec_space_t>();

    // Terminate early if counts differ
    if (new_source_local_entity_indices.extent(0) != source_local_entity_indices.extent(0) ||
        target_local_entity_indices.extent(0) != target_local_entity_indices.extent(0)) {
      source_local_entity_indices_ = new_source_local_entity_indices;
      target_local_entity_indices_ = new_target_local_entity_indices;
      return true;
    }

    // Check if source entities differ using Kokkos parallel_reduce over a logical OR
    bool local_indices_changed = false;
    using range_policy = stk::ngp::RangePolicy<exec_space_t>;
    Kokkos::parallel_reduce(
        "CheckIndexChangesSource", range_policy(0, source_local_entity_indices.extent(0)),
        KOKKOS_LAMBDA(int i, bool &local_result) {
          const stk::mesh::FastMeshIndex old_fast_mesh_index = source_local_entity_indices.view_device()(i);
          const stk::mesh::FastMeshIndex new_fast_mesh_index = new_source_local_entity_indices.view_device()(i);
          local_result = local_result || (old_fast_mesh_index != new_fast_mesh_index);
        },
        Kokkos::LOr<bool>(local_indices_changed));

    Kokkos::parallel_reduce(
        "CheckIndexChangesTarget", range_policy(0, target_local_entity_indices.extent(0)),
        KOKKOS_LAMBDA(int i, bool &local_result) {
          const stk::mesh::FastMeshIndex old_fast_mesh_index = target_local_entity_indices.view_device()(i);
          const stk::mesh::FastMeshIndex new_fast_mesh_index = new_target_local_entity_indices.view_device()(i);
          local_result = local_result || (old_fast_mesh_index != new_fast_mesh_index);
        },
        Kokkos::LOr<bool>(local_indices_changed));
    Kokkos::fence();

    int global_indices_changed_int = false;
    int local_indices_changed_int = local_indices_changed;
    stk::all_reduce_max(bulk_data.parallel(), &local_indices_changed_int, &global_indices_changed_int, 1);

    return static_cast<bool>(global_indices_changed_int);
  }

  bool objects_moved_too_much() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, std::runtime_error,
                        "Only one specialization is allowed for now.");
    BulkData &bulk_data = link_data_.bulk_data();
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);
    bool source_moved_too_much =
        source_search_checkers_[0](source_local_entity_indices_, source_search_spheres_, search_buffer_);
    bool target_moved_too_much =
        target_search_checkers_[0](target_local_entity_indices_, target_search_spheres_, search_buffer_);

    // Return the MPI consensus on if any objects moved too much
    int local_objects_moved_too_much_int = source_moved_too_much || target_moved_too_much;
    int global_objects_moved_too_much_int = false;
    stk::all_reduce_max(bulk_data.parallel(), &local_objects_moved_too_much_int, &global_objects_moved_too_much_int, 1);

    return static_cast<bool>(global_objects_moved_too_much_int);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  LinkData &link_data_;
  stk::ParallelMachine parallel_machine_;
  stk::mesh::EntityRank source_rank_;
  stk::mesh::EntityRank target_rank_;

  // Flags
  bool concretized_;
  bool enforce_source_target_symmetry_;
  bool allow_duplicate_links_;
  bool sort_search_results_;
  bool has_been_generated_;

  // Search settings
  stk::search::SearchMethod search_method_;
  std::shared_ptr<search_filters::SearchFilterBase<exec_space_t>> search_filter_ptr_;
  double search_buffer_;

  // Cached data
  std::vector<search_accumulator_t> source_search_accumulators_;
  std::vector<search_accumulator_t> target_search_accumulators_;
  std::vector<search_checker_t> source_search_checkers_;
  std::vector<search_checker_t> target_search_checkers_;
  specialization_vec_t specializations_;

  // Internal views
  ngp_local_entity_indices_t source_local_entity_indices_;
  ngp_local_entity_indices_t target_local_entity_indices_;
  typename type_info_t::search_spheres_view_t source_search_spheres_;
  typename type_info_t::search_spheres_view_t target_search_spheres_;
  typename type_info_t::result_view_t search_results_;
  typename type_info_t::result_view_t::HostMirror host_search_results_;
  Kokkos::UnorderedMap<Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>, void, exec_space_t>
      existing_linked_neighbors_;
  //@}
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_GENNEIGHBORLINKERS_HPP_
