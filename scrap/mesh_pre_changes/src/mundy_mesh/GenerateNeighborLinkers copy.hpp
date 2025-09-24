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

#ifndef MUNDY_MESH_GENERATENEIGHBORLINKERS_HPP_
#define MUNDY_MESH_GENERATENEIGHBORLINKERS_HPP_

/// \file GenerateNeighborLinkers.hpp

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

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

// Mundy libs
#include <mundy_core/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/ForEachEntity.hpp>          // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>           // for mundy::mesh::field_copy
#include <mundy_mesh/impl/LinkedBucketConn.hpp>  // for mundy::mesh::impl::LinkedBucketConn

namespace mundy {

namespace mesh {

/// \brief Get the local fast mesh indices for a set of entities as an NgpView.
template <typename OurExecSpace>
core::NgpView<stk::mesh::FastMeshIndex *, OurExecSpace> get_local_entity_indices(const stk::mesh::BulkData &bulk_data,
                                                                                 stk::mesh::EntityRank rank,
                                                                                 const stk::mesh::Selector &selector,
                                                                                 const OurExecSpace &exec_space) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities);

  core::NgpView<stk::mesh::FastMeshIndex *, OurExecSpace> ngp_local_entity_indices("local_entity_indices",
                                                                                   local_entities.size());

  Kokkos::parallel_for(stk::ngp::HostRangePolicy(0, local_entities.size()),
                       [&bulk_data, &local_entities, &host_mesh_indices](const int i) {
                         const stk::mesh::MeshIndex &mesh_index = bulk_data.mesh_index(local_entities[i]);
                         ngp_local_entity_indices.host_view()(i) =
                             stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
                       });

  ngp_local_entity_indices.modify_on_host();
  return ngp_local_entity_indices;
}

Kokkos::UnorderedMap<Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>, void, stk::ngp::ExecSpace>
get_linked_neighbors_set(const LinkData &link_data, const stk::mesh::Selector &link_selector) {
  auto &ngp_link_data = get_updated_ngp_data(link_data);
  auto &ngp_mesh = link_data.ngp_mesh();

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
struct SearchTypes {
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

class SearchFilterBase {
 public:
  using bool_view_t = Kokkos::View<bool *, stk::ngp::ExecSpace>;
  using search_types_t = SearchTypes<stk::ngp::ExecSpace>;
  virtual ~SearchFilterBase() = default;
  virtual void apply(const search_types_t::result_view_t &search_results, const bool_view_t &mask) const = 0;
};

/// \brief A helper class for generating and regenerating neighbor linkers
///
/// Example usage:
/// \code {.cpp}
///   auto gen_links = create_gen_neighbor_links(link_data, search_type::Spheres, stk::ngp::ExecSpace{});
///   gen_neighbor_links.set_source_target_rank(stk::topology::ELEM_RANK, stk::topology::ELEM_RANK) //
///                     .acts_on(source_agg1, target_agg1, source_generator1, target_generator1)    //
///                     .acts_on(source_agg2, target_agg2, source_generator2, target_generator2)    //
///                     .set_enforce_source_target_symmetry(true)  //
///                     .set_allow_duplicate_links(true)           //
///                     .set_sort_search_results(true)             //
///                     .set_search_method(search_method)          //
///                     .set_search_buffer(search_buffer)          //
///                     .set_search_filter(search_filters::ExcludeConnectedEntities{}) //
///                     .set_parallel_machine(bulk_data.parallel()) //
///                     .concretize();
///   gen_neighbor_links.generate(source_spheres, target_spheres, search_method);
/// \endcode
template <typename SearchType>
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
  GenNeighborLinks(const LinkData &link_data, [[maybe_unused]] const OurExecSpace &exec_space)
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
        search_buffer_(0.0),
        search_filter_(nullptr),
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

  /// \brief Get the search buffer
  double get_search_buffer() const {
    return search_buffer_;
  }

  /// \brief Get the search filter
  const std::shared_ptr<SearchFilterBaseT<exec_space_t>> &get_search_filter() const {
    return search_filter_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the parallel machine
  GenNeighborLinks &set_parallel_machine(stk::ParallelMachine parallel_machine) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set parallel machine after concretization.");
    parallel_machine_ = parallel_machine;
    return *this;
  }

  /// \brief Set the source and target ranks
  GenNeighborLinks &set_source_target_rank(stk::mesh::EntityRank source_rank, stk::mesh::EntityRank target_rank) {
    MUNDY_THROW_REQUIRE(is_not_concreteized(), "Cannot set source and target ranks after concretization.");
    MUNDY_THROW_REQUIRE(source_rank == stk::topology::ELEM_RANK && target_rank == stk::topology::ELEM_RANK,
                        "To start, we will only support element to element neighbor searches.");

    source_rank_ = source_rank;
    target_rank_ = target_rank;
    return *this;
  }

  /// \brief Set enforce source-target symmetry
  GenNeighborLinks &set_enforce_source_target_symmetry(bool new_value) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set enforce source-target symmetry after concretization.");
    enforce_source_target_symmetry_ = new_value;
    return *this;
  }

  /// \brief Set allow duplicate links
  GenNeighborLinks &set_allow_duplicate_links(bool new_value) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set allow duplicate links after concretization.");
    allow_duplicate_links_ = new_value;
    return *this;
  }

  /// \brief Set sort search results
  GenNeighborLinks &set_sort_search_results(bool new_value) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set sort search results after concretization.");
    sort_search_results_ = new_value;
    return *this;
  }

  /// \brief Set search method
  GenNeighborLinks &set_search_method(stk::search::SearchMethod search_method) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set search method after concretization.");
    search_method_ = search_method;
    return *this;
  }

  /// \brief Set search buffer
  GenNeighborLinks &set_search_buffer(double search_buffer) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set search buffer after concretization.");
    search_buffer_ = search_buffer;
    return *this;
  }

  /// \brief Set search filter
  GenNeighborLinks &set_search_filter(const std::shared_ptr<SearchFilter> &search_filter) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set search filter after concretization.");
    MUNDY_THROW_REQUIRE(search_filter == nullptr, "Search filtering not yet implemented.");
    search_filter_ = search_filter;
    return *this;
  }

  /// \brief Inform us of a set of sources/targets to act on, their search generators, and their link specialization
  template <typename SourceAggregate, typename TargetAggregate, typename SourceGenerator, typename TargetGenerator>
  GenNeighborLinks &acts_on(const SourceAggregate &source_agg, const TargetAggregate &target_agg,
                            const SourceGenerator &source_generator, const TargetGenerator &target_generator,
                            const stk::mesh::PartVector &link_parts) {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot set source/targets after concretization.");

    // Each pair of objects we act on gets two function objects:
    //   search_accumulator: accumulates the search spheres/boxes for each source/target and
    //     operator()(const ngp_local_entity_indices_t &local_entity_indices, double search_buffer) ->
    //     search_spheres_view_t
    //   search_checker: checks if a rebuild is necessary based on the given search buffer.
    //     operator()(const ngp_local_entity_indices_t &local_entity_indices, const search_spheres_view_t&
    //     old_search_spheres, double search_buffer) -> bool

    source_search_accumulators_.emplace_back(make_search_accumulator(source_agg, source_generator));
    target_search_accumulators_.emplace_back(make_search_accumulator(target_agg, target_generator));
    source_search_checkers_.emplace_back(make_search_checker(source_agg, source_generator));
    target_search_checkers_.emplace_back(make_search_checker(target_agg, target_generator));
    specializations_.emplace_back(source_agg.selector(), target_agg.selector(), link_parts);
    return *this;
  }
  //@}

  //! \name Search
  //@{

  /// \brief Concretize the settings. Once set, no more settings can be changed.
  /// Must set source and target rank and add at least one set of sources/targets to act on.
  void concretize() {
    MUNDY_THROW_REQUIRE(!is_concretized(), "Cannot concretize more than once.");
    MUNDY_THROW_REQUIRE(source_rank_ != stk::topology::INVALID_RANK,
                        "Source rank must be set before calling concretize.");
    MUNDY_THROW_REQUIRE(target_rank_ != stk::topology::INVALID_RANK,
                        "Target rank must be set before calling concretize.");
    MUNDY_THROW_REQUIRE(!specializations_.empty(), "Must act on at least one source/target pair.");
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");
    concretized_ = true;

    // In the future, concretize will invert the specialization to go from link partition to source/target buckets.
    // When this occurs, we will store one set of local entity indices for each source/target pair per link partition.
  }

  /// \brief Generate neighbor linkers between a given set of source and target spheres
  /// \return True if the search was performed. False if no regeneration was necessary.
  bool generate() {
    MUNDY_THROW_REQUIRE(is_concretized(), "Cannot generate links before concretization.");

    if (has_been_generated_) {
      // Check if we need to regenerate
      // TODO(palmerb4): Collect information from other processes.
      if (indices_changed()) {
        collect_local_entity_indices();
        collect_search_spheres();
        if (!allow_duplicate_links_) {
          collect_existing_linked_neighbors();  // If the indices have changed this list is invalid
        }
      } else if (objects_moved_too_much()) {
        collect_search_spheres();
      } else {
        needs_regenerated_ = false;
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
  using ngp_local_entity_indices_t = core::NgpView<stk::mesh::FastMeshIndex *, exec_space_t>;
  using search_accumulator_t = std::function<search_spheres_view_t(const ngp_local_entity_indices_t &, double)>;
  using search_checker_t =
      std::function<bool(const ngp_local_entity_indices_t &, const type_info_t::search_spheres_view_t &, double)>;
  using specialization_t = std::tuple<stk::mesh::Selector, stk::mesh::Selector, stk::mesh::PartVector>;
  using specialization_vec_t = std::vector<specialization_t>;
  //@}

  //! \name Internal functions
  //@{

  /// \brief Make a search accumulator
  template <typename Aggregate, typename Generator>
  search_accumulator_t make_search_accumulator(const Aggregate &agg, const Generator &gen) {
    return [&agg, &gen](const ngp_local_entity_indices_t &local_entity_indices, double search_buffer) {
      auto ngp_agg = get_updated_ngp_data(agg);
      auto &ngp_mesh = ngp_agg.ngp_mesh();
      const int my_rank = agg.bulk_data().parallel_rank();

      size_t num_local_spheres = local_entity_indices.extent(0);
      type_info_t::search_spheres_view_t search_spheres("gen_neighbors_search_spheres", num_local_spheres);

      Kokkos::parallel_for(
          stk::ngp::RangePolicy<exec_space_t>(0, num_local_spheres), KOKKOS_LAMBDA(const unsigned &i) {
            stk::mesh::Entity sphere = ngp_mesh.get_entity(spheres.rank(), sphere_indices(i));
            stk::mesh::FastMeshIndex sphere_index = ngp_mesh.fast_mesh_index(sphere);

            geom::Sphere<double> sphere = gen(sphere_index);
            stk::search::Point<double> center(sphere.center[0], sphere.center[1], sphere.center[2]);
            search_spheres(i) =
                type_info_t::sphere_ident_proc_t{stk::search::Sphere<double>(center, sphere.radius + search_buffer),
                                                 IdentProc(ngp_mesh.identifier(sphere), my_rank)};
          });

      return search_spheres;
    };
  }

  /// \brief Make a search checker
  template <typename Aggregate, typename Generator>
  search_checker_t make_search_checker(const Aggregate &agg, const Generator &gen) {
    return [&agg, &gen](const ngp_local_entity_indices_t &local_entity_indices,
                        const type_info_t::search_spheres_view_t &old_search_spheres, double search_buffer) -> bool {
      auto ngp_agg = get_updated_ngp_data(agg);
      auto &ngp_mesh = ngp_agg.ngp_mesh();
      const int my_rank = agg.bulk_data().parallel_rank();

      local_entity_indices.sync_to<exec_space_t>();
      size_t num_local_spheres = local_entity_indices.extent(0);
      bool moved_too_much = false;
      Kokkos::parallel_reduce(
          "CheckSearchBufferViolation", stk::ngp::HostRangePolicy(0, num_local_spheres),
          KOKKOS_LAMBDA(int i, bool &local_result) {
            stk::mesh::Entity sphere = ngp_mesh.get_entity(spheres.rank(), sphere_indices(i));
            stk::mesh::FastMeshIndex sphere_index = ngp_mesh.fast_mesh_index(sphere);

            auto new_center = gen(sphere_index).center();
            auto old_center = old_search_spheres(i).box().center();
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
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);
    source_local_entity_indices_ =
        get_local_entity_indices(link_data_.bulk_data(), source_rank_, source_selector, stk::ngp::ExecSpace{});
    target_local_entity_indices_ =
        get_local_entity_indices(link_data_.bulk_data(), target_rank_, target_selector, stk::ngp::ExecSpace{});
  }

  void collect_search_spheres() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);
    const auto &source_search_accumulator = source_search_accumulators_[0];
    const auto &target_search_accumulator = target_search_accumulators_[0];
    source_search_spheres_ = source_search_accumulator(source_local_entity_indices_, search_buffer_);
    target_search_spheres_ = target_search_accumulator(target_local_entity_indices_, search_buffer_);
  }

  void collect_existing_linked_neighbors() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");
    stk::mesh::Selector link_selector = stk::mesh::selectIntersection(std::get<2>(specializations_[0]));
    existing_linked_neighbors_ = get_linked_neighbors_set(link_data_, link_selector);
  }

  void perform_search() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");

    // TODO(palmerb4): Some of these settings don't come into play until after Trilinos 16.0.0
    const bool results_parallel_symmetry = true;    // ensures that the source/target owners know about all pairs
    const bool auto_swap_domain_and_range = false;  // swap source and target if target is owned and source is not
    const bool sort_search_results = sort_search_results_;  // sort the search results by source id
    stk::search::coarse_search(source_search_spheres_, target_search_spheres_, search_method_, parallel_machine_,
                               search_results_, exec_space_t{}, results_parallel_symmetry);
  }

  void copy_search_results_to_host() {
    host_search_results_ = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search_results_);
  }

  void ghost_neighbors() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");
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
        MUNDY_THROW_REQUIRE(false, "Invalid search result. Somehow we received a pair of elements that we don't own.");
      }
    }

    bulk_data.change_ghosting(neighbor_ghosting, elements_to_ghost);
    bulk_data.modification_end();
  }

  void request_link_creation_and_destruction() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");

    // TODO(palmerb4): Reduce the serial cost of this operation
    //
    // Plan:
    //  Before this function or ghost_neighbors is called, perform a single serial loop over the host search results
    //  and get there source and target entities. This way, neither this function nor ghost_neighbors need to call
    //  get_entity and can safely employ host parallel_for loops.

    stk::mesh::PartVector locally_owned_link_parts = std::get<2>(specializations_[0]);
    locally_owned_link_parts.push_back(&link_data_.bulk_data().mesh_meta_data().locally_owned_part());
    LinkPartition &link_partition =
        link_data_.get_link_partition(link_data_.get_partition_key(locally_owned_link_parts));

    size_t num_search_results = host_search_results_.size();
    for (size_t i = 0; i < num_search_results; ++i) {
      const auto search_result = host_search_results_(i);
      stk::mesh::Entity source_entity =
          link_data_.bulk_data().get_entity(source_rank_, search_result.domainIdentProc.id());
      stk::mesh::Entity target_entity =
          link_data_.bulk_data().get_entity(target_rank_, search_result.rangeIdentProc.id());
      if (SourceTargetFilter::allow(source_entity, target_entity)) {
        link_partition.request_link(source_entity, target_entity);
      }
    }
  }

  void process_requests() {
    bool fully_consistent = source_rank_ >= stk::topology::ELEM_RANK && target_rank_ >= stk::topology::ELEM_RANK;
    link_data_.process_requests(fully_consistent);
  }

  bool indices_changed() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");
    BulkData &bulk_data = link_data_.bulk_data();
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);

    std::vector<stk::mesh::Entity> local_source_entities;
    std::vector<stk::mesh::Entity> local_target_entities;
    stk::mesh::get_entities(bulk_data, source_rank_, source_selector, local_source_entities);
    stk::mesh::get_entities(bulk_data, target_rank_, target_selector, local_target_entities);

    auto source_local_entity_indices = source_local_entity_indices_;
    auto target_local_entity_indices = target_local_entity_indices_;

    // Check if counts differ
    if (local_source_entities.size() != source_local_entity_indices.extent(0) ||
        local_target_entities.size() != target_local_entity_indices.extent(0)) {
      return true;
    }

    // Check if source entities differ using Kokkos parallel_reduce over a logical OR
    bool local_indices_changed = false;
    Kokkos::parallel_reduce(
        "CheckIndexChangesSource", stk::ngp::HostRangePolicy(0, local_source_entities.size()),
        KOKKOS_LAMBDA(int i, bool &local_result) {
          const stk::mesh::MeshIndex &old_mesh_index = bulk_data.mesh_index(local_source_entities[i]);
          const stk::mesh::FastMeshIndex old_fast_mesh_index{old_mesh_index.bucket->bucket_id(),
                                                             old_mesh_index.bucket_ordinal};
          local_result = local_result || (old_fast_mesh_index != source_local_entity_indices(i));
        },
        Kokkos::LOr<bool>(local_indices_changed));

    Kokkos::parallel_reduce(
        "CheckIndexChangesTarget", stk::ngp::HostRangePolicy(0, local_source_entities.size()),
        KOKKOS_LAMBDA(int i, bool &local_result) {
          const stk::mesh::MeshIndex &old_mesh_index = bulk_data.mesh_index(local_source_entities[i]);
          const stk::mesh::FastMeshIndex old_fast_mesh_index{old_mesh_index.bucket->bucket_id(),
                                                             old_mesh_index.bucket_ordinal};
          local_result = local_result || (old_fast_mesh_index != target_local_entity_indices(i));
        },
        Kokkos::LOr<bool>(local_indices_changed));

    Kokkos::fence();

    bool global_indices_changed = false;
    stk::all_reduce(bulk_data.parallel(), stk::ReduceMax<1>(&local_indices_changed, &global_indices_changed));
    return global_indices_changed;
  }

  bool objects_moved_too_much() {
    MUNDY_THROW_REQUIRE(specializations_.size() == 1, "Only one specialization is allowed for now.");
    BulkData &bulk_data = link_data_.bulk_data();
    const stk::mesh::Selector &source_selector = std::get<0>(specializations_[0]);
    const stk::mesh::Selector &target_selector = std::get<1>(specializations_[0]);
    bool source_moved_too_much =
        source_search_checkers_[0](source_local_entity_indices_, source_search_spheres_, search_buffer_);
    bool target_moved_too_much =
        target_search_checkers_[0](target_local_entity_indices_, target_search_spheres_, search_buffer_);

    // Return the MPI consensus on if any objects moved too much
    bool local_objects_moved_too_much = source_moved_too_much || target_moved_too_much;
    bool global_objects_moved_too_much = false;
    stk::all_reduce(bulk_data.parallel(),
                    stk::ReduceMax<1>(&local_objects_moved_too_much, &global_objects_moved_too_much));

    return global_objects_moved_too_much;
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
  bool has_been_regenerated_;

  // Search settings
  stk::search::SearchMethod search_method_;
  double search_buffer_;
  std::shared_ptr<SearchFilterBaseT<exec_space_t>> search_filter_;

  // Cached data
  std::vector<search_accumulator_t> source_search_accumulators_;
  std::vector<search_accumulator_t> target_search_accumulators_;
  std::vector<search_checker_t> source_search_checkers_;
  std::vector<search_checker_t> target_search_checkers_;
  specialization_vec_t specializations_;

  // Internal views
  ngp_local_entity_indices_t source_local_entity_indices_;
  ngp_local_entity_indices_t target_local_entity_indices_;
  type_info_t::search_spheres_view_t source_search_spheres_;
  type_info_t::search_spheres_view_t target_search_spheres_;
  type_info_t::result_view_t search_results_;
  type_info_t::result_view_t::HostMirror host_search_results_;
  //@}
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_GENERATENEIGHBORLINKERS_HPP_
