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

/// \file STKSearch.cpp
/// \brief Definition of GenerateNeighborLinkers's STKSearch technique.

// C++ core libs
#include <iomanip>
#include <iostream>
#include <memory>         // for std::shared_ptr, std::unique_ptr
#include <string>         // for std::string
#include <unordered_map>  // for std::unordered_map
#include <vector>         // for std::vector

// Trilinos libs
#include <Kokkos_Core.hpp>            // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Kokkos_Pair.hpp>            // for Kokkos::pair
#include <Kokkos_UnorderedMap.hpp>    // for Kokkos::UnorderedMap
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/HashEntityAndEntityKey.hpp>  // for std::hash<stk::mesh::Entity>, std::hash<stk::mesh::EntityKey>
#include <stk_mesh/base/MeshUtils.hpp>               // for stk::mesh::fixup_ghosted_to_shared_nodes
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpField.hpp>  // for stk::mesh::NgpField
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_search/BoundingBox.hpp>           // for stk::search::Box
// #include <stk_search/BoxIdent.hpp>              // for stk::search::IdentProcIntersection
#include <stk_search/CoarseSearch.hpp>          // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>          // for stk::search::KDTREE
#include <stk_util/parallel/CommNeighbors.hpp>  // for stk::CommNeighbors

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/Linkers.hpp>    // for mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing
#include <mundy_linkers/generate_neighbor_linkers/techniques/STKSearch.hpp>  // for mundy::linkers::...::STKSearch
#include <mundy_mesh/BulkData.hpp>                                           // for mundy::mesh::BulkData
#include <mundy_mesh/fmt_stk_types.hpp>

namespace mundy {

namespace linkers {

namespace generate_neighbor_linkers {

namespace techniques {

namespace {

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash1 ^ hash2;
  }
};

}  // namespace

// \name Constructors and destructor
//{

STKSearch::STKSearch(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "STKSearch: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(STKSearch::get_valid_fixed_params());

  // Get the field pointers.
  const std::string element_aabb_field_name = valid_fixed_params.get<std::string>("element_aabb_field_name");
  const std::string linked_entities_field_name = NeighborLinkers::get_linked_entities_field_name();
  const std::string linked_entity_owners_field_name = NeighborLinkers::get_linked_entity_owners_field_name();
  element_aabb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_aabb_field_name);
  linked_entities_field_ptr_ = meta_data_ptr_->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, linked_entities_field_name);
  linked_entity_owners_field_ptr_ =
      meta_data_ptr_->get_field<int>(stk::topology::CONSTRAINT_RANK, linked_entity_owners_field_name);
  MUNDY_THROW_REQUIRE(element_aabb_field_ptr_ != nullptr, std::invalid_argument,
                      "STKSearch: element_aabb_field_ptr_ cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(linked_entities_field_ptr_ != nullptr, std::invalid_argument,
                      "STKSearch: linked_entities_field_ptr_ cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(linked_entity_owners_field_ptr_ != nullptr, std::invalid_argument,
                      "STKSearch: linked_entity_owners_field_ptr_ cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  neighbor_linkers_part_ptr_ = meta_data_ptr_->get_part("NEIGHBOR_LINKERS");
  MUNDY_THROW_REQUIRE(neighbor_linkers_part_ptr_ != nullptr, std::invalid_argument,
                      "STKSearch: Expected a part with name 'NEIGHBOR_LINKERS' but part does not exist.");

  const Teuchos::Array<std::string> specialized_neighbor_linkers_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("specialized_neighbor_linkers_part_names");
  const Teuchos::Array<std::string> valid_source_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_source_entity_part_names");
  const Teuchos::Array<std::string> valid_target_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_target_entity_part_names");
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(
          part != nullptr, std::invalid_argument,
          std::string("STKSearch: Expected a part with name '") + part_name + "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };

  specialized_neighbor_linkers_part_ptrs_ = parts_from_names(*meta_data_ptr_, specialized_neighbor_linkers_part_names);
  valid_source_entity_part_ptrs_ = parts_from_names(*meta_data_ptr_, valid_source_entity_part_names);
  valid_target_entity_part_ptrs_ = parts_from_names(*meta_data_ptr_, valid_target_entity_part_names);
}
//}

// \name Setters
//{

void STKSearch::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(STKSearch::get_valid_mutable_params());
  enforce_symmetry_ = valid_mutable_params.get<bool>("enforce_symmetry");
}
//}

// \name Getters
//{

std::vector<stk::mesh::Part *> STKSearch::get_valid_source_entity_parts() const {
  return valid_source_entity_part_ptrs_;
}

std::vector<stk::mesh::Part *> STKSearch::get_valid_target_entity_parts() const {
  return valid_target_entity_part_ptrs_;
}
//}

// \name Actions
//{

#define MUNDY_USE_OLD_STK_SEARCH

#ifdef MUNDY_USE_OLD_STK_SEARCH

namespace {

using SearchIdentProc = stk::search::IdentProc<stk::mesh::EntityKey>;
using BoxIdVector = std::vector<std::pair<stk::search::Box<double>, SearchIdentProc>>;
using SearchIdPairVector = std::vector<std::pair<SearchIdentProc, SearchIdentProc>>;

inline std::pair<SearchIdentProc, SearchIdentProc> make_sorted_pair(const SearchIdentProc &a,
                                                                    const SearchIdentProc &b) {
  return (a.id() < b.id()) ? std::make_pair(a, b) : std::make_pair(b, a);
}

inline void get_existing_pairs(std::set<std::pair<SearchIdentProc, SearchIdentProc>> &existing_pairs,
                               const std::vector<stk::mesh::Entity> &existing_linkers,
                               const stk::mesh::Field<LinkedEntitiesFieldType::value_type> &linked_entities_field,
                               const stk::mesh::Field<int> &linked_entity_owners_field,
                               const bool enforce_symmetry = true) {
  size_t num_linkers = existing_linkers.size();

// Only use local variables if OpenMP is enabled
#ifdef _OPENMP
#pragma omp parallel
  {
    std::set<std::pair<SearchIdentProc, SearchIdentProc>> thread_local_existing_pairs;
#pragma omp for schedule(static)
    for (size_t i = 0; i < num_linkers; ++i) {
      const stk::mesh::Entity &linker = existing_linkers[i];

      const stk::mesh::EntityKey::entity_key_t *key_t_ptr =
          reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(stk::mesh::field_data(linked_entities_field, linker));
      const stk::mesh::EntityKey source_entity_key(key_t_ptr[0]);
      const stk::mesh::EntityKey target_entity_key(key_t_ptr[1]);

      const int *owner_ptr = stk::mesh::field_data(linked_entity_owners_field, linker);
      const int source_proc = owner_ptr[0];
      const int target_proc = owner_ptr[1];

      const SearchIdentProc source_entity_proc(source_entity_key, source_proc);
      const SearchIdentProc target_entity_proc(target_entity_key, target_proc);

      auto sorted_pair = enforce_symmetry ? make_sorted_pair(source_entity_proc, target_entity_proc)
                                          : std::make_pair(source_entity_proc, target_entity_proc);

      thread_local_existing_pairs.insert(sorted_pair);
    }

// Merge thread-local results into the global set
#pragma omp critical
    existing_pairs.insert(thread_local_existing_pairs.begin(), thread_local_existing_pairs.end());
  }
#else
  for (size_t i = 0; i < num_linkers; ++i) {
    const stk::mesh::Entity &linker = existing_linkers[i];

    const stk::mesh::EntityKey::entity_key_t *key_t_ptr =
        reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(stk::mesh::field_data(linked_entities_field, linker));
    const stk::mesh::EntityKey source_entity_key(key_t_ptr[0]);
    const stk::mesh::EntityKey target_entity_key(key_t_ptr[1]);

    const int *owner_ptr = stk::mesh::field_data(linked_entity_owners_field, linker);
    const int source_proc = owner_ptr[0];
    const int target_proc = owner_ptr[1];

    const SearchIdentProc source_entity_proc(source_entity_key, source_proc);
    const SearchIdentProc target_entity_proc(target_entity_key, target_proc);

    auto sorted_pair = enforce_symmetry ? make_sorted_pair(source_entity_proc, target_entity_proc)
                                        : std::make_pair(source_entity_proc, target_entity_proc);

    existing_pairs.insert(sorted_pair);
  }
#endif  // _OPENMP
}

inline void filter_and_add_pairs(const std::set<std::pair<SearchIdentProc, SearchIdentProc>> &existing_pairs,
                                 std::set<std::pair<SearchIdentProc, SearchIdentProc>> &new_pairs,
                                 const std::vector<std::pair<SearchIdentProc, SearchIdentProc>> &search_id_pairs,
                                 const bool enforce_symmetry = true) {
  size_t num_search_pairs = search_id_pairs.size();

// Only use local variables if OpenMP is enabled
#ifdef _OPENMP

#pragma omp parallel
  {
    std::set<std::pair<SearchIdentProc, SearchIdentProc>> thread_local_new_pairs;

#pragma omp for schedule(static)
    for (size_t i = 0; i < num_search_pairs; ++i) {
      const auto &search_pair = search_id_pairs[i];
      if (search_pair.first.id() == search_pair.second.id()) continue;  // Skip self-interactions

      auto sorted_pair = enforce_symmetry ? make_sorted_pair(search_pair.first, search_pair.second) : search_pair;

      // Check existence in existing_pairs or thread_local_new_pairs and add to new_pairs if not found
      if ((existing_pairs.find(sorted_pair) == existing_pairs.end()) &&
          (thread_local_new_pairs.find(sorted_pair) == thread_local_new_pairs.end())) {
        thread_local_new_pairs.insert(sorted_pair);
      }
    }

// Merge thread-local results into the global set
#pragma omp critical
    new_pairs.insert(thread_local_new_pairs.begin(), thread_local_new_pairs.end());
  }
#else
  for (size_t i = 0; i < num_search_pairs; ++i) {
    const auto &search_pair = search_id_pairs[i];
    if (search_pair.first.id() == search_pair.second.id()) continue;  // Skip self-interactions

    auto sorted_pair = enforce_symmetry ? make_sorted_pair(search_pair.first, search_pair.second) : search_pair;

    // Check existence in existing_pairs or thread_local_new_pairs and add to new_pairs if not found
    if (existing_pairs.find(sorted_pair) == existing_pairs.end()) {
      new_pairs.insert(sorted_pair);
    }
  }
#endif  // _OPENMP
}

inline void fill_box_id_vector(mundy::mesh::BulkData &bulk_data, mundy::mesh::MetaData &meta_data,
                               const stk::mesh::Field<double> &element_aabb_field, const stk::mesh::Selector &selector,
                               BoxIdVector &boxes) {
  const int rank = bulk_data.parallel_rank();
  const size_t num_local_elements_in_selector = stk::mesh::count_selected_entities(
      meta_data.locally_owned_part() & selector, bulk_data.buckets(stk::topology::ELEMENT_RANK));
  boxes.reserve(num_local_elements_in_selector);

  const stk::mesh::BucketVector &element_buckets = bulk_data.get_buckets(
      stk::topology::ELEMENT_RANK, stk::mesh::Selector(meta_data.locally_owned_part()) & selector);
  for (size_t bucket_idx = 0; bucket_idx < element_buckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &element_bucket = *element_buckets[bucket_idx];
    for (size_t elem_idx = 0; elem_idx < element_bucket.size(); ++elem_idx) {
      // Skip invalid entities.
      if (bulk_data.is_valid(element_bucket[elem_idx])) {
        stk::mesh::Entity const &element = element_bucket[elem_idx];

        const double *aabb = stk::mesh::field_data(element_aabb_field, element);
        stk::search::Box<double> box(aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]);

        SearchIdentProc search_id(bulk_data.entity_key(element), rank);

        boxes.emplace_back(box, search_id);
      }
    }
  }
}

}  // namespace

void STKSearch::execute(const stk::mesh::Selector &domain_input_selector,
                        const stk::mesh::Selector &range_input_selector) {
  // std::cout << "##############################################################" << std::endl;
  // std::cout << "STKSearch: Executing..." << " Enforce symmetry: " << enforce_symmetry_ << std::endl;
  // std::cout << std::scientific << std::setprecision(3);
  Kokkos::Timer overall_timer;

  // Step 1: Copy the AABBs, owning procs, and entity keys into a separate contiguous vector for the sources and
  // targets.<< value << std::endl
  Kokkos::Timer fill_box_timer;
  const int parallel_size = bulk_data_ptr_->parallel_size();
  const int parallel_rank = bulk_data_ptr_->parallel_rank();

  BoxIdVector source_boxes;
  BoxIdVector target_boxes;

  fill_box_id_vector(*bulk_data_ptr_, *meta_data_ptr_, *element_aabb_field_ptr_, domain_input_selector, source_boxes);
  fill_box_id_vector(*bulk_data_ptr_, *meta_data_ptr_, *element_aabb_field_ptr_, range_input_selector, target_boxes);
  double fill_box_time = fill_box_timer.seconds();
  // std::cout << "  Num source/target boxes: " << source_boxes.size() << " / " << target_boxes.size() << std::endl;
  // std::cout << "  Fill box time: " << fill_box_time << std::endl;

  // Step 2: Perform the search.
  Kokkos::Timer search_timer;
  SearchIdPairVector search_id_pairs;
  stk::search::coarse_search(source_boxes, target_boxes, stk::search::KDTREE, bulk_data_ptr_->parallel(),
                             search_id_pairs);
  const size_t num_neighbors = search_id_pairs.size();
  // std::cout << "  Search time: " << search_timer.seconds() << std::endl;

  // Step 3: Determine which process should create the linker between the source and target entities, if any.
  // No linker will be created if one already exists.
  Kokkos::Timer communicate_linker_field_data_timer;
  stk::mesh::Part &neighbor_linkers_part = *neighbor_linkers_part_ptr_;
  LinkedEntitiesFieldType &linked_entities_field = *linked_entities_field_ptr_;
  stk::mesh::Field<int> &linked_entity_owners_field = *linked_entity_owners_field_ptr_;
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {linked_entities_field_ptr_, linked_entity_owners_field_ptr_});
  // std::cout << "  Communicate linker field data time: " << communicate_linker_field_data_timer.seconds() << std::endl;

  /**
   * We have a collection of pairs and a collection of linkers which connect pairs. We need to optimize the process of
   * determining if the inverse of a pair exists and if a linker already exists between a pair.
   *
   * We will try to use an std::set to aid us in this process. We will loop over each linker and store their source and
   * target entities in a sorted pair (such that we don't need to check both orders) and stash that in an std::set. We
   * will then loop over the search pairs and append the sorted pair into the set if it doesn't already exist. This will
   * automatically skip duplicates.
   *
   * If enforce_symmetry is true, we will only store the sorted pair in the set. If it is false, we will store the pair
   * as is.
   */
  Kokkos::Timer get_existing_pairs_timer;

  // Identify if the inverse of a (source, target) pair exists in the search_id_pairs vector.
  stk::mesh::EntityVector existing_linkers;
  stk::mesh::get_selected_entities(neighbor_linkers_part, bulk_data_ptr_->buckets(stk::topology::CONSTRAINT_RANK),
                                   existing_linkers);

  std::set<std::pair<SearchIdentProc, SearchIdentProc>> existing_pairs;
  get_existing_pairs(existing_pairs, existing_linkers, linked_entities_field, linked_entity_owners_field,
                     enforce_symmetry_);
  // std::cout << "  Get existing pairs time: " << get_existing_pairs_timer.seconds() << std::endl;

  Kokkos::Timer get_new_pairs_timer;
  std::set<std::pair<SearchIdentProc, SearchIdentProc>> new_pairs;
  filter_and_add_pairs(existing_pairs, new_pairs, search_id_pairs, enforce_symmetry_);
  // std::cout << "  Get new pairs time: " << get_new_pairs_timer.seconds() << std::endl;

  // Step 5: Ghost the search results (only if using multiple mpi processes)
  Kokkos::Timer ghost_timer;
  if (parallel_size > 1) {
    bulk_data_ptr_->modification_begin();
    std::vector<stk::mesh::EntityProc> send_ghosts;
    for (const auto &new_pair : new_pairs) {
      const bool ghosting_is_required = (new_pair.first.proc() != new_pair.second.proc());
      if (ghosting_is_required) {
        const bool we_need_to_create_linker = (new_pair.first.proc() == parallel_rank);
        if (we_need_to_create_linker) {
          const int target_proc = new_pair.second.proc();
          const stk::mesh::EntityKey source_entity_key = new_pair.first.id();
          const stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(source_entity).owned(), std::invalid_argument,
                             "STKSearch: source entity must be owned.");
          send_ghosts.push_back(std::make_pair(source_entity, target_proc));
        } else {
          const int source_proc = new_pair.first.proc();
          const stk::mesh::EntityKey target_entity_key = new_pair.second.id();
          const stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(target_entity).owned(), std::invalid_argument,
                             "STKSearch: target entity must be owned.");
          send_ghosts.push_back(std::make_pair(target_entity, source_proc));
        }
      }
    }

    stk::mesh::Ghosting &ghosting = bulk_data_ptr_->create_ghosting("MUNDYS_STK_SEARCH_GHOSTING");
    bulk_data_ptr_->change_ghosting(ghosting, send_ghosts);
    bulk_data_ptr_->modification_end();
  }
  // std::cout << "  Ghost time: " << ghost_timer.seconds() << std::endl;

  // Step 6: Generate all the new linkers we will own.
  // The total number of linkers to create is the sum of the we_need_to_create_linker vector.
  // Because the total number of linkers does not match the number of search_id_pairs, we use an offset vector to allow
  // each pair to know its index in the requested_entities vector. This is just the cumulative sum of the
  // we_need_to_create_linker boolean vector minus 1.
  Kokkos::Timer generate_new_entities_timer;
  size_t num_linkers_to_create = 0;
  for (const auto &new_pair : new_pairs) {
    const bool we_need_to_create_linker = (new_pair.first.proc() == parallel_rank);
    num_linkers_to_create += we_need_to_create_linker;
  }

  // std::cout << "  parallel_rank: " << parallel_rank << " | Num linkers to create: " << num_linkers_to_create
  //           << std::endl;

  bulk_data_ptr_->modification_begin();
  std::vector<size_t> requests(bulk_data_ptr_->mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::CONSTRAINT_RANK] = num_linkers_to_create;
  std::vector<stk::mesh::Entity> requested_entities;
  // Unpacked version of generate_new_entities to avoid a costly change of part post creation.
  {
    size_t num_ranks = requests.size();
    std::vector<std::vector<stk::mesh::EntityId>> requested_ids(num_ranks);
    for (size_t i = 0; i < num_ranks; ++i) {
      stk::topology::rank_t rank = static_cast<stk::topology::rank_t>(i);
      bulk_data_ptr_->generate_new_ids(rank, requests[i], requested_ids[i]);
    }

    // generating 'owned' entities
    stk::mesh::PartVector add_parts;
    add_parts.push_back(&bulk_data_ptr_->mesh_meta_data().locally_owned_part());
    for (stk::mesh::Part *part : specialized_neighbor_linkers_part_ptrs_) {
      add_parts.push_back(part);
    }

    requested_entities.clear();
    for (size_t i = 0; i < num_ranks; ++i) {
      stk::topology::rank_t rank = static_cast<stk::topology::rank_t>(i);
      std::vector<stk::mesh::Entity> new_entities;
      bulk_data_ptr_->declare_entities(rank, requested_ids[i], add_parts, new_entities);
      requested_entities.insert(requested_entities.end(), new_entities.begin(), new_entities.end());
    }
  }
  // std::cout << "  Generate new entities time: " << generate_new_entities_timer.seconds() << std::endl;

  // Connect the linkers that were created on this process.
  Kokkos::Timer connect_linker_timer;
  size_t offset = 0;
  for (const auto &new_pair : new_pairs) {
    const bool we_need_to_create_linker = (new_pair.first.proc() == parallel_rank);
    if (we_need_to_create_linker) {
      const stk::mesh::EntityKey source_entity_key = new_pair.first.id();
      const stk::mesh::EntityKey target_entity_key = new_pair.second.id();
      const stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
      const stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);
      stk::mesh::Entity linker = requested_entities[offset];

      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(source_entity), std::invalid_argument,
                         "STKSearch: source entity is invalid");
      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(target_entity), std::invalid_argument,
                         "STKSearch: target entity is invalid.");
      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(linker), std::invalid_argument,
                         "STKSearch: linker entity is invalid.");

      // Connect the linker to the source and target entities' nodes and stash their entity keys in the linker's field.
      mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr_, linker, source_entity, target_entity);
      stk::mesh::field_data(linked_entities_field, linker)[0] = source_entity_key;
      stk::mesh::field_data(linked_entities_field, linker)[1] = target_entity_key;
      offset++;
    }
  }
  // std::cout << "  Connect linker time: " << connect_linker_timer.seconds() << std::endl;

  // We're using one-sided linker creation, so we need to fixup the ghosted to shared nodes.
  Kokkos::Timer fixup_ghosted_timer;
  stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr_);
  // std::cout << "  Fixup ghosted time: " << fixup_ghosted_timer.seconds() << std::endl;

  bulk_data_ptr_->modification_end();

  // Ghost the linked entities to any process that owns any of the other linked entities.
  if (parallel_size > 1) {
    Kokkos::Timer ghost_linked_entities_timer;
    bulk_data_ptr_->modification_begin();
    stk::mesh::Ghosting &ghosting = bulk_data_ptr_->create_ghosting("MUNDYS_STK_SEARCH_GHOSTING");
    const stk::mesh::Selector specialized_parts_selector =
        stk::mesh::selectUnion(specialized_neighbor_linkers_part_ptrs_);
    mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr_, *linked_entities_field_ptr_,
                                                 *linked_entity_owners_field_ptr_, specialized_parts_selector);
    bulk_data_ptr_->modification_end();
    // std::cout << "  Ghost linked entities time: " << ghost_linked_entities_timer.seconds() << std::endl;

    // Step 7: Communicate every field defined on the source and target selectors. (only if using multiple mpi
    // processes)
    Kokkos::Timer communicate_fields_timer;
    // for each rank: Selector -> buckets -> bucket to field -> sort and unique all fields
    // TODO(palmerb4: Use the following once we upgrade STK)
    // std::vector<stk::mesh::Part *> domain_parts;
    // std::vector<stk::mesh::Part *> range_parts;
    // domain_input_selector.get_parts(domain_parts);
    // range_input_selector.get_parts(domain_parts);
    // std::vector<stk::mesh::FieldBase *> fields_to_communicate;
    // for (const stk::mesh::FieldBase *field : meta_data_ptr_->get_fields()) {
    //   for (const stk::mesh::FieldRestriction &restriction : field->restrictions()) {
    //     for (const stk::mesh::Part *part : domain_parts) {
    //       if (restriction.selects(*part)) {
    //         fields_to_communicate.push_back(field);
    //         break;
    //       }
    //     }
    //     for (const stk::mesh::Part *part : range_parts) {
    //       if (restriction.selects(*part)) {
    //         fields_to_communicate.push_back(field);
    //         break;
    //       }
    //     }
    //   }
    // }

    std::vector<const stk::mesh::FieldBase *> fields_to_communicate;
    for (const stk::mesh::FieldBase *field : meta_data_ptr_->get_fields()) {
      for (const stk::mesh::FieldRestriction &restriction : field->restrictions()) {
        if (stk::mesh::is_subset(domain_input_selector, restriction.selector())) {
          fields_to_communicate.push_back(field);
          break;
        }
        if (stk::mesh::is_subset(range_input_selector, restriction.selector())) {
          fields_to_communicate.push_back(field);
          break;
        }
      }
    }
    fields_to_communicate.erase(std::unique(fields_to_communicate.begin(), fields_to_communicate.end()),
                                fields_to_communicate.end());
    stk::mesh::communicate_field_data(ghosting, fields_to_communicate);
    // std::cout << "  Communicate fields time: " << communicate_fields_timer.seconds() << std::endl;
  }


  // std::cout << "  Overall time: " << overall_timer.seconds() << std::endl;
  // std::cout << "##############################################################" << std::endl;
}
//}

#else

namespace {

using SearchIdentProc = stk::search::IdentProc<stk::mesh::EntityKey>;
using BoxIdVector = std::vector<std::pair<stk::search::Box<double>, SearchIdentProc>>;
using IdentProcIntersection = stk::search::IdentProcIntersection<SearchIdentProc, SearchIdentProc>;
using SearchIdPairVector =
    std::vector<std::pair<SearchIdentProc, SearchIdentProc>>;  // IdentProcIntersection not supported yet. Need
                                                               // Trilinos 16.0
using SearchResultVector = std::vector<IdentProcIntersection>;
using EntityKeyPair = Kokkos::pair<stk::mesh::EntityKey, stk::mesh::EntityKey>;
inline IdentProcIntersection make_sorted_intersection(const SearchIdentProc &a, const SearchIdentProc &b) {
  return (a.id() < b.id())
             ? IdentProcIntersection(SearchIdentProc(a.id(), a.proc()), SearchIdentProc(b.id(), b.proc()))
             : IdentProcIntersection(SearchIdentProc(b.id(), b.proc()), SearchIdentProc(a.id(), a.proc()));
}

inline Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection> get_existing_pairs(
    const stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &existing_linkers_selector,
    stk::mesh::NgpField<LinkedEntitiesFieldType::value_type> &ngp_linked_entities_field,
    stk::mesh::NgpField<int> &ngp_linked_entity_owners_field, const bool enforce_symmetry = true) {
  ngp_linked_entities_field.sync_to_device();
  ngp_linked_entity_owners_field.sync_to_device();
  size_t num_linkers = stk::mesh::count_selected_entities(
      existing_linkers_selector, ngp_mesh.get_bulk_on_host().buckets(stk::topology::CONSTRAINT_RANK));
  Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection> existing_pairs(num_linkers);

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::CONSTRAINT_RANK, existing_linkers_selector,
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &entity_index) {
        const stk::mesh::Entity linker = ngp_mesh.get_entity(stk::topology::CONSTRAINT_RANK, entity_index);

        stk::mesh::EntityKey source_entity_key(
            static_cast<stk::mesh::EntityKey::entity_key_t>(ngp_linked_entities_field(entity_index, 0)));
        stk::mesh::EntityKey target_entity_key(
            static_cast<stk::mesh::EntityKey::entity_key_t>(ngp_linked_entities_field(entity_index, 1)));

        int source_proc = ngp_linked_entity_owners_field(entity_index, 0);
        int target_proc = ngp_linked_entity_owners_field(entity_index, 1);

        SearchIdentProc source_entity_proc(source_entity_key, source_proc);
        SearchIdentProc target_entity_proc(target_entity_key, target_proc);

        // WARNING: For some unfathomable reason, we need to create the IdentProcIntersection entirely from scratch for
        // Kokkos's hasher to correctly identify duplicate keys.
        auto maybe_sorted_pair = enforce_symmetry
                                     ? make_sorted_intersection(source_entity_proc, target_entity_proc)
                                     : IdentProcIntersection(SearchIdentProc(source_entity_key, source_proc),
                                                             SearchIdentProc(target_entity_key, target_proc));
        existing_pairs.insert(
            Kokkos::make_pair(maybe_sorted_pair.domainIdentProc.id(), maybe_sorted_pair.rangeIdentProc.id()),
            maybe_sorted_pair);
      });

  return existing_pairs;
}

inline Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection> get_new_pairs(
    const Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection> &existing_pairs,
    const SearchResultVector &search_id_pairs, const bool enforce_symmetry = true) {
  // Insert elements into new_pairs if they are not in existing_pairs
  size_t num_search_pairs = search_id_pairs.size();
  size_t estimate_of_num_new_pairs = 0;

  // Use a reduction to estimate the number of new pairs
  Kokkos::parallel_reduce(
      "EstimateNumNewPairs", num_search_pairs,
      KOKKOS_LAMBDA(const size_t i, size_t &local_estimate) {
        const auto &search_pair = search_id_pairs[i];

        const stk::mesh::EntityKey source_entity_key(search_pair.domainIdentProc.id());
        const stk::mesh::EntityKey target_entity_key(search_pair.rangeIdentProc.id());

        if (source_entity_key == target_entity_key) return;  // Skip self-interactions

        int source_proc = search_pair.domainIdentProc.proc();
        int target_proc = search_pair.rangeIdentProc.proc();

        SearchIdentProc source_entity_proc(source_entity_key, source_proc);
        SearchIdentProc target_entity_proc(target_entity_key, target_proc);

        // WARNING: For some unfathomable reason, we need to create the IdentProcIntersection entirely from scratch for
        // Kokkos's hasher to correctly identify duplicate keys.
        auto maybe_sorted_pair = enforce_symmetry
                                     ? make_sorted_intersection(source_entity_proc, target_entity_proc)
                                     : IdentProcIntersection(SearchIdentProc(source_entity_key, source_proc),
                                                             SearchIdentProc(target_entity_key, target_proc));

        // If the pair isn't in the existing pairs increment the estimate of the number of new pairs
        // This will overcount duplicates but that is fine for an estimate
        local_estimate += static_cast<size_t>(!existing_pairs.exists(
            Kokkos::make_pair(maybe_sorted_pair.domainIdentProc.id(), maybe_sorted_pair.rangeIdentProc.id())));
      },
      estimate_of_num_new_pairs);

  Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection> new_pairs(estimate_of_num_new_pairs);
  Kokkos::parallel_for(
      "InsertNewPairs", num_search_pairs, KOKKOS_LAMBDA(const size_t i) {
        const auto &search_pair = search_id_pairs[i];

        stk::mesh::EntityKey source_entity_key(search_pair.domainIdentProc.id());
        stk::mesh::EntityKey target_entity_key(search_pair.rangeIdentProc.id());

        if (source_entity_key == target_entity_key) return;  // Skip self-interactions

        int source_proc = search_pair.domainIdentProc.proc();
        int target_proc = search_pair.rangeIdentProc.proc();

        SearchIdentProc source_entity_proc(source_entity_key, source_proc);
        SearchIdentProc target_entity_proc(target_entity_key, target_proc);

        // WARNING: For some unfathomable reason, we need to create the IdentProcIntersection entirely from scratch for
        // Kokkos's hasher to correctly identify duplicate keys.
        auto maybe_sorted_pair = enforce_symmetry
                                     ? make_sorted_intersection(source_entity_proc, target_entity_proc)
                                     : IdentProcIntersection(SearchIdentProc(source_entity_key, source_proc),
                                                             SearchIdentProc(target_entity_key, target_proc));

        // If the pair isn't in the existing pairs, insert it and let Kokkos handle collision.
        bool pair_exists = existing_pairs.exists(
            Kokkos::make_pair(maybe_sorted_pair.domainIdentProc.id(), maybe_sorted_pair.rangeIdentProc.id()));
        if (!pair_exists) {
          auto result = new_pairs.insert(
              Kokkos::make_pair(maybe_sorted_pair.domainIdentProc.id(), maybe_sorted_pair.rangeIdentProc.id()),
              maybe_sorted_pair);
        }
      });
  return new_pairs;
}

inline void fill_box_id_vector(mundy::mesh::BulkData &bulk_data, mundy::mesh::MetaData &meta_data,
                               const stk::mesh::Field<double> &element_aabb_field, const stk::mesh::Selector &selector,
                               BoxIdVector &boxes) {
  const int rank = bulk_data.parallel_rank();
  const size_t num_local_elements_in_selector = stk::mesh::count_selected_entities(
      meta_data.locally_owned_part() & selector, bulk_data.buckets(stk::topology::ELEMENT_RANK));
  boxes.reserve(num_local_elements_in_selector);

  const stk::mesh::BucketVector &element_buckets = bulk_data.get_buckets(
      stk::topology::ELEMENT_RANK, stk::mesh::Selector(meta_data.locally_owned_part()) & selector);
  for (size_t bucket_idx = 0; bucket_idx < element_buckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &element_bucket = *element_buckets[bucket_idx];
    for (size_t elem_idx = 0; elem_idx < element_bucket.size(); ++elem_idx) {
      // Skip invalid entities.
      if (bulk_data.is_valid(element_bucket[elem_idx])) {
        stk::mesh::Entity const &element = element_bucket[elem_idx];

        const double *aabb = stk::mesh::field_data(element_aabb_field, element);
        stk::search::Box<double> box(aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]);

        SearchIdentProc search_id(bulk_data.entity_key(element), rank);

        boxes.emplace_back(box, search_id);
      }
    }
  }
}

size_t count_num_linkers_to_create(
    const Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection>::HostMirror &new_pairs_host,
    const int parallel_rank) {
  size_t num_linkers_to_create = 0;
  Kokkos::parallel_reduce(
      "CountLinkersToCreate", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, new_pairs_host.capacity()),
      KOKKOS_LAMBDA(const uint32_t i, size_t &local_num_linkers) {
        if (new_pairs_host.valid_at(i)) {
          const auto &new_pair = new_pairs_host.value_at(i);
          const bool we_need_to_create_linker = (new_pair.domainIdentProc.proc() == parallel_rank);
          local_num_linkers += static_cast<size_t>(we_need_to_create_linker);
        }
      },
      num_linkers_to_create);
  return num_linkers_to_create;
}

}  // namespace

void STKSearch::execute(const stk::mesh::Selector &domain_input_selector,
                        const stk::mesh::Selector &range_input_selector) {
  std::cout << "##############################################################" << std::endl;
  std::cout << "STKSearch: Executing..." << " Enforce symmetry: " << enforce_symmetry_ << std::endl;
  std::cout << std::scientific << std::setprecision(3);
  Kokkos::Timer overall_timer;
  // Step 1: Copy the AABBs, owning procs, and entity keys into a separate contiguous vector for the sources and
  // targets.
  const int parallel_size = bulk_data_ptr_->parallel_size();
  const int parallel_rank = bulk_data_ptr_->parallel_rank();

  BoxIdVector source_boxes;
  BoxIdVector target_boxes;

  Kokkos::Timer fill_box_timer;
  fill_box_id_vector(*bulk_data_ptr_, *meta_data_ptr_, *element_aabb_field_ptr_, domain_input_selector, source_boxes);
  fill_box_id_vector(*bulk_data_ptr_, *meta_data_ptr_, *element_aabb_field_ptr_, range_input_selector, target_boxes);
  double fill_box_time = fill_box_timer.seconds();
  std::cout << "  Num source/target boxes: " << source_boxes.size() << " / " << target_boxes.size() << std::endl;
  std::cout << "  Fill box time: " << fill_box_time << std::endl;

  // Step 2: Perform the search.
  Kokkos::Timer search_timer;
  SearchIdPairVector search_id_pairs_tmp;
  stk::search::coarse_search(source_boxes, target_boxes, stk::search::KDTREE, bulk_data_ptr_->parallel(),
                             search_id_pairs_tmp);
  // Convert the search_id_pairs_tmp to a SearchResultVector. Use move and consume the search_id_pairs_tmp.
  SearchResultVector search_id_pairs(search_id_pairs_tmp.size());
  for (size_t i = 0; i < search_id_pairs.size(); ++i) {
    search_id_pairs[i] =
        IdentProcIntersection(std::move(search_id_pairs_tmp[i].first), std::move(search_id_pairs_tmp[i].second));
  }

  const size_t num_neighbors = search_id_pairs.size();
  double search_time = search_timer.seconds();
  std::cout << "  Search time: " << search_time << std::endl;

  // Step 2.5: Update the existing linker entity ownership, as it might have changed
  stk::mesh::Part &neighbor_linkers_part = *neighbor_linkers_part_ptr_;
  LinkedEntitiesFieldType &linked_entities_field = *linked_entities_field_ptr_;
  stk::mesh::Field<int> &linked_entity_owners_field = *linked_entity_owners_field_ptr_;
  stk::mesh::Selector linker_selector = neighbor_linkers_part;
  const stk::mesh::BulkData &bulk_data = *bulk_data_ptr_;
  stk::mesh::for_each_entity_run(
      bulk_data, stk::topology::CONSTRAINT_RANK,
      linker_selector & bulk_data_ptr_->mesh_meta_data().locally_owned_part(),
      [&linked_entities_field, &linked_entity_owners_field, &parallel_rank](const stk::mesh::BulkData &bulk_data,
                                                                            const stk::mesh::Entity &linker) {
        const unsigned bytes_per_entity =
            stk::mesh::field_bytes_per_entity(linked_entities_field, bulk_data.bucket(linker));
        const size_t max_num_linked_entities = bytes_per_entity / linked_entities_field.data_traits().size_of;
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(linked_entities_field, linker));
        int *owner_ptr = stk::mesh::field_data(linked_entity_owners_field, linker);
        for (unsigned i = 0; i < max_num_linked_entities; ++i) {
          const bool entity_exist = key_t_ptr[i] != stk::mesh::EntityKey::entity_key_t::INVALID;
          if (entity_exist) {
            const stk::mesh::Entity linked_entity = bulk_data.get_entity(key_t_ptr[i]);
            MUNDY_THROW_ASSERT(bulk_data.is_valid(linked_entity), std::runtime_error,
                               "The linked entity is somehow invalid.");
            owner_ptr[i] = bulk_data.parallel_owner_rank(linked_entity);
          }
        }
      });

  // Step 3: Determine which process should create the linker between the source and target entities, if any.
  // No linker will be created if one already exists.
  Kokkos::Timer communicate_linker_field_data_timer;
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {linked_entities_field_ptr_, linked_entity_owners_field_ptr_});
  double communicate_linker_field_data_time = communicate_linker_field_data_timer.seconds();
  std::cout << "  Communicate linker field data time: " << communicate_linker_field_data_time << std::endl;

  /**
   * We have a collection of pairs and a collection of linkers which connect pairs. We need to optimize the process of
   * determining if the inverse of a pair exists and if a linker already exists between a pair.
   *
   * We will try to use an std::set to aid us in this process. We will loop over each linker and store their source and
   * target entities in a sorted pair (such that we don't need to check both orders) and stash that in an std::set. We
   * will then loop over the search pairs and append the sorted pair into the set if it doesn't already exist. This will
   * automatically skip duplicates.
   *
   * If enforce_symmetry is true, we will only store the sorted pair in the set. If it is false, we will store the pair
   * as is.
   */

  // Identify if the inverse of a (source, target) pair exists in the search_id_pairs vector.
  Kokkos::Timer get_existing_pairs_timer;
  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(*bulk_data_ptr_);
  auto &ngp_linked_entities_field =
      stk::mesh::get_updated_ngp_field<LinkedEntitiesFieldType::value_type>(linked_entities_field);
  auto &ngp_linked_entity_owners_field = stk::mesh::get_updated_ngp_field<int>(linked_entity_owners_field);
  Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection> existing_pairs = get_existing_pairs(
      ngp_mesh, neighbor_linkers_part, ngp_linked_entities_field, ngp_linked_entity_owners_field, enforce_symmetry_);

  double get_existing_pairs_time = get_existing_pairs_timer.seconds();
  std::cout << "  Get existing pairs time: " << get_existing_pairs_time << " for " << existing_pairs.size()
            << " existing pairs." << std::endl;

  // Identify the new pairs that need to be created.
  Kokkos::Timer determine_new_pairs_timer;
  Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection> new_pairs =
      get_new_pairs(existing_pairs, search_id_pairs, enforce_symmetry_);
  Kokkos::fence();

// Kokkos version over 4.4 required for create_mirror
#if (KOKKOS_VERSION >= 40400)
  Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection>::HostMirror new_pairs_host =
      Kokkos::create_mirror(new_pairs);
#else
  Kokkos::UnorderedMap<EntityKeyPair, IdentProcIntersection>::HostMirror new_pairs_host;
  Kokkos::deep_copy(new_pairs_host, new_pairs);
#endif
  const size_t new_pairs_capacity = new_pairs_host.capacity();
  double determine_new_pairs_time = determine_new_pairs_timer.seconds();
  std::cout << "  Determine new pairs time: " << determine_new_pairs_time << " for " << new_pairs.size()
            << " new pairs." << std::endl;

  // Step 5: Ghost the search results (only if using multiple mpi procs)
  Kokkos::Timer ghost_timer;
  if (parallel_size > 1) {
    bulk_data_ptr_->modification_begin();
    std::vector<std::pair<stk::mesh::Entity, int>> send_ghosts;
    for (size_t i = 0; i < new_pairs_capacity; i++) {
      if (new_pairs_host.valid_at(i)) {
        const auto &new_pair = new_pairs_host.value_at(i);
        const bool ghosting_is_required = (new_pair.domainIdentProc.proc() != new_pair.rangeIdentProc.proc());

        if (ghosting_is_required) {
          const bool we_need_to_create_linker = (new_pair.domainIdentProc.proc() == parallel_rank);
          if (we_need_to_create_linker) {
            const int target_proc = new_pair.rangeIdentProc.proc();
            const stk::mesh::EntityKey source_entity_key = new_pair.domainIdentProc.id();
            const stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
            MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(source_entity).owned(), std::invalid_argument,
                               "STKSearch: source entity must be owned.");
            send_ghosts.push_back(std::make_pair(source_entity, target_proc));
          } else {
            const int source_proc = new_pair.domainIdentProc.proc();
            const stk::mesh::EntityKey target_entity_key = new_pair.rangeIdentProc.id();
            const stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);
            MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(target_entity).owned(), std::invalid_argument,
                               "STKSearch: target entity must be owned.");
            send_ghosts.push_back(std::make_pair(target_entity, source_proc));
          }
        }
      }
    }

    stk::mesh::Ghosting &ghosting = bulk_data_ptr_->create_ghosting("MUNDYS_STK_SEARCH_GHOSTING");
    bulk_data_ptr_->change_ghosting(ghosting, send_ghosts);
    bulk_data_ptr_->modification_end();
  }

  double ghost_time = ghost_timer.seconds();
  std::cout << "  Ghost time: " << ghost_time << std::endl;

  // Step 6: Generate all the new linkers we will own.
  // The total number of linkers to create is the sum of the we_need_to_create_linker vector.
  // Because the total number of linkers does not match the number of search_id_pairs, we use an offset vector to allow
  // each pair to know its index in the requested_entities vector. This is just the cumulative sum of the
  // we_need_to_create_linker boolean vector minus 1.
  size_t num_linkers_to_create = count_num_linkers_to_create(new_pairs_host, parallel_rank);
  std::cout << "  parallel_rank: " << parallel_rank << " | Num linkers to create: " << num_linkers_to_create
            << std::endl;

  bulk_data_ptr_->modification_begin();
  std::vector<size_t> requests(bulk_data_ptr_->mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::CONSTRAINT_RANK] = num_linkers_to_create;
  std::vector<stk::mesh::Entity> requested_entities;

  Kokkos::Timer generate_new_entities_timer;
  bulk_data_ptr_->generate_new_entities(requests, requested_entities);
  double generate_new_entities_time = generate_new_entities_timer.seconds();
  std::cout << "  Generate new entities time: " << generate_new_entities_time << std::endl;

  Kokkos::Timer change_entity_parts_timer;
  bulk_data_ptr_->change_entity_parts(requested_entities, specialized_neighbor_linkers_part_ptrs_);
  double change_entity_parts_time = change_entity_parts_timer.seconds();
  std::cout << "  Change entity parts time: " << change_entity_parts_time << std::endl;

  // Connect the linkers that were created on this process.
  Kokkos::Timer connect_linker_timer;
  size_t offset = 0;
  stk::mesh::Permutation perm = stk::mesh::Permutation::INVALID_PERMUTATION;
  stk::mesh::OrdinalVector scratch1, scratch2, scratch3;
  for (size_t i = 0; i < new_pairs_capacity; i++) {
    if (new_pairs_host.valid_at(i)) {
      const auto &new_pair = new_pairs_host.value_at(i);
      const bool we_need_to_create_linker = (new_pair.domainIdentProc.proc() == parallel_rank);
      if (we_need_to_create_linker) {
        const stk::mesh::EntityKey source_entity_key = new_pair.domainIdentProc.id();
        const stk::mesh::EntityKey target_entity_key = new_pair.rangeIdentProc.id();
        const stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
        const stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);
        stk::mesh::Entity linker = requested_entities[offset];

        MUNDY_THROW_ASSERT(
            bulk_data_ptr_->is_valid(source_entity), std::invalid_argument,
            fmt::format("STKSearch: source entity {} is invalid. On rank {}", source_entity_key, parallel_rank));
        MUNDY_THROW_ASSERT(
            bulk_data_ptr_->is_valid(target_entity), std::invalid_argument,
            fmt::format("STKSearch: target entity {} is invalid. On rank {}", target_entity_key, parallel_rank));
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(linker), std::invalid_argument,
                           fmt::format("STKSearch: linker entity {} is invalid. On rank {}",
                                       bulk_data_ptr_->entity_key(linker), parallel_rank));

        // Connect the linker to the source and target entities' nodes and stash their entity keys in the linker's
        // field.
        mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr_, linker, source_entity, target_entity);
        stk::mesh::field_data(linked_entities_field, linker)[0] = source_entity_key;
        stk::mesh::field_data(linked_entities_field, linker)[1] = target_entity_key;
        offset++;
      }
    }
  }
  double connect_linker_time = connect_linker_timer.seconds();
  std::cout << "  Connect linker time: " << connect_linker_time << std::endl;

  // We're using one-sided linker creation, so we need to fixup the ghosted to shared nodes.
  Kokkos::Timer fixup_ghost_timer;
  stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr_);
  double fixup_ghost_time = fixup_ghost_timer.seconds();
  std::cout << "  Fixup ghost time: " << fixup_ghost_time << std::endl;

  bulk_data_ptr_->modification_end();

  // Ghost the linked entities to any process that owns any of the other linked entities.
  if (parallel_size > 1) {
    Kokkos::Timer ghost_linked_entities_timer;
    bulk_data_ptr_->modification_begin();
    const stk::mesh::Selector specialized_parts_selector =
        stk::mesh::selectUnion(specialized_neighbor_linkers_part_ptrs_);
    mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr_, *linked_entities_field_ptr_,
                                                 *linked_entity_owners_field_ptr_, specialized_parts_selector);

    stk::mesh::Ghosting &ghosting = bulk_data_ptr_->create_ghosting("MUNDYS_STK_SEARCH_GHOSTING");
    bulk_data_ptr_->modification_end();
    double ghost_linked_entities_time = ghost_linked_entities_timer.seconds();

    // Step 7: Communicate every field defined on the source and target selectors.
    // for each rank: Selector -> buckets -> bucket to field -> sort and unique all fields
    Kokkos::Timer communicate_fields_timer;
    // TODO(palmerb4: Use the following once we upgrade STK)
    // std::vector<stk::mesh::Part *> domain_parts;
    // std::vector<stk::mesh::Part *> range_parts;
    // domain_input_selector.get_parts(domain_parts);
    // range_input_selector.get_parts(domain_parts);
    // std::vector<stk::mesh::FieldBase *> fields_to_communicate;
    // for (const stk::mesh::FieldBase *field : meta_data_ptr_->get_fields()) {
    //   for (const stk::mesh::FieldRestriction &restriction : field->restrictions()) {
    //     for (const stk::mesh::Part *part : domain_parts) {
    //       if (restriction.selects(*part)) {
    //         fields_to_communicate.push_back(field);
    //         break;
    //       }
    //     }
    //     for (const stk::mesh::Part *part : range_parts) {
    //       if (restriction.selects(*part)) {
    //         fields_to_communicate.push_back(field);
    //         break;
    //       }
    //     }
    //   }
    // }

    std::vector<const stk::mesh::FieldBase *> fields_to_communicate;
    for (const stk::mesh::FieldBase *field : meta_data_ptr_->get_fields()) {
      for (const stk::mesh::FieldRestriction &restriction : field->restrictions()) {
        if (stk::mesh::is_subset(domain_input_selector, restriction.selector())) {
          fields_to_communicate.push_back(field);
          break;
        }
        if (stk::mesh::is_subset(range_input_selector, restriction.selector())) {
          fields_to_communicate.push_back(field);
          break;
        }
      }
    }
    fields_to_communicate.erase(std::unique(fields_to_communicate.begin(), fields_to_communicate.end()),
                                fields_to_communicate.end());

    stk::mesh::communicate_field_data(ghosting, fields_to_communicate);
    double communicate_fields_time = communicate_fields_timer.seconds();
    std::cout << "  Communicate fields time: " << communicate_fields_time << std::endl;
  }

  double overall_time = overall_timer.seconds();
  std::cout << "  Overall time: " << overall_time << std::endl;
  std::cout << "##############################################################" << std::endl;
}
//}

#endif  // MUNDY_USE_OLD_STK_SEARCH

}  // namespace techniques

}  // namespace generate_neighbor_linkers

}  // namespace linkers

}  // namespace mundy
