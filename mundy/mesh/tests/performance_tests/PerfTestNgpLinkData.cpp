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

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <queue>        // for std::priority_queue
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// External
#include "nanobench.h"

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy libs
#include <mundy_math/Tolerance.hpp>           // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>             // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>            // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>         // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>            // for mundy::mesh::MetaData
#include <mundy_mesh/NewNgpLinkData.hpp>      // for mundy::mesh::NewNgpLinkData
#include <mundy_mesh/NewNgpLinkMetaData.hpp>  // for mundy::mesh::NewNgpLinkMetaData

namespace mundy {

namespace mesh {

namespace {

/*
Test setup:
  - Controllable number of entities in each rank
  - Controllable link rank
  - Controllable link dimensionality
  - Controllable number of link partitions
  - Controllable number of links
  - Controllable link distribution across partitions:
    - equal: each partition gets the same number of links
    - log-normal: links are distributed according to a log-normal distribution (sigma = 2, p = 0)
  - Controllable link generation locality/spacity:
    - id_locality: 0.0 = Gaussian, 1.0 = Uniform, in between = linear interpolation
    - id_sigma: Standard deviation of the Gaussian distribution for entity id selection

Desired questions:
  CRS and COO creation/updating performance:
  - Does the number of entities in each rank affect performance vs having the same number of entities in a single rank?
  - Does link rank affect performance?
  - How many clock cycles to declare a new link relation? Is this influenced by any external factors?
  - How does the cost of checking the status of the CRS and subsequently updating it scale with the
    - link rank/dimensionality?
    - number of links?
    - distribution of links across partitions?
    - number of partitions?
    - number of stk buckets?
    - organization of links vs their linked entities (ordered vs random)?
  - How do these parameters affect the cost of looping over each link, fetching its linked entities, and performing an
    operation on them?
    - Force reduction: Compute an equal and opposite force on each linked entity given their node coordinates and
      atomically update their node force field. The function need not make physical sense.
  - How do these parameters affect the cost of looping over each linked entity, fetching its links, and performing an
    operation on their downward connections?
    - Same force reduction as above but without an atomic since we never loop over the same entity twice.

Notes:

- For our force reduction, because we need an equal and opposite force that works for any number of ranks, we can do a
spring system where the first linked entity connect to each of the other linked entities via a spring with a constant
spring constant and rest length.

- For creating new link partitions, we don't really care *why* the partition was created just *that* it was created. We
can just declare N link parts within the same link data and add the requested number of links to each.
get_or_create_crs_partitions(universal_part()) should then return N partitions, one for each link part.

- For CRS connectivity tests, link/entity declaration is tricky. We will use the following formula to allow for
controllable locality:
  1. Generate a collection of num_entities entities,
  2. For partition i, generate num_links_per_partition[i] links.
  3. For each link, perform weighted reservoir sampling (Efraimidis–Spirakis) to select link_dimensionality entities to
     link.
    3.5. The weights for each entity are selected as the linear interpolation of a Gaussian distribution (about the
         center id achieved by rescaling the linker id into the entity id range with a shift to make it symmetric) and a
         uniform distribution. The control parameters will be the linear interpolation factor (locality) and the
         standard deviation of the Gaussian (sigma).
  4. Declare the link relations between the link and the selected entities.

We need the weighted sampling to cost less then O(N_e N_l log(k)) where N_e is the number of entities,
N_l is the number of links, and k is the dimensionality. We can do so via adding one layer of randomness at the bucket
level. For each link bucket, perform weighted reservoir sampling to choose which entity bucket to draw from O(N_b
log(k)), where N_eb is the number of entity buckets. Then, we can do one independent weighted draw from each selected
bucket to get the entity to link to O(N_bc k) where N_bc is the bucket capacity. The total cost of the new scheme (under
the assumption that N_bc~500, log(k) ~ 1) is: O(N_l N_e / 500^2 + N_l 500 k) vs O(N_l N_e).
*/

/// \brief The strategy for choosing the ranks of linked entities.
enum class LinkedEntityRanksType {
  SAME,         ///< All linked entities are of NODE_RANK
  RANDOM,       ///< All linked entities are of random ranks (NODE_RANK, EDGE_RANK, FACE_RANK, or ELEM_RANK)
  ONE_TO_MANY,  ///< First entity is of a ELEM_RANK rank, the rest are of NODE_RANK
};

/// \brief The weighting use for distributing links to each partition
enum class LinkDistribution {
  EQUAL,       ///< Uniform distribution of links across partitions
  LOG_NORMAL,  ///< Log-normal distribution of links across partitions (sigma = 2, p = 0)
};

std::ostream& operator<<(std::ostream& os, const LinkDistribution& dist) {
  switch (dist) {
    case LinkDistribution::EQUAL:
      os << "EQUAL";
      break;
    case LinkDistribution::LOG_NORMAL:
      os << "LOG_NORMAL";
      break;
    default:
      os << "UNKNOWN";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const LinkedEntityRanksType& type) {
  switch (type) {
    case LinkedEntityRanksType::SAME:
      os << "SAME";
      break;
    case LinkedEntityRanksType::RANDOM:
      os << "RANDOM";
      break;
    case LinkedEntityRanksType::ONE_TO_MANY:
      os << "ONE_TO_MANY";
      break;
    default:
      os << "UNKNOWN";
  }
  return os;
}

/// \brief Convert nonnegative weights to integer counts summing to total.
/// Uses largest-remainder (Hamilton) apportionment.
/// \param weights length-num_partitions, nonnegative (not all zero)
/// \param total total sum of integer counts to allocate
/// \return vector<size_t> of counts, length-num_partitions, summing to total
std::vector<size_t> apportion_largest_remainder(const std::vector<double>& weights, size_t total) {
  const size_t num_partitions = weights.size();
  std::vector<size_t> counts(num_partitions, 0);
  if (num_partitions == 0 || total == 0) return counts;

  // Sum weights; handle all-zero corner case by putting everything in slot 0
  const double wsum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (!(wsum > 0.0)) {
    counts[0] = total;
    return counts;
  }

  // Proportional allocation (floors) + collect fractional remainders
  std::vector<double> quotas(num_partitions);
  for (size_t i = 0; i < num_partitions; ++i) quotas[i] = (weights[i] / wsum) * static_cast<double>(total);

  size_t allocated = 0;
  for (size_t i = 0; i < num_partitions; ++i) {
    const auto base = static_cast<size_t>(std::floor(quotas[i]));
    counts[i] = base;
    allocated += base;
  }

  // Distribute leftover ones to the largest fractional parts (stable tie-breaker = lower index)
  size_t rem = total - allocated;
  if (rem > 0) {
    std::vector<std::pair<double, size_t>> frac_idx;
    frac_idx.reserve(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
      const double frac = quotas[i] - std::floor(quotas[i]);
      frac_idx.emplace_back(frac, i);
    }
    std::stable_sort(frac_idx.begin(), frac_idx.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t k = 0; k < rem; ++k) counts[frac_idx[k].second] += 1;
  }
  return counts;
}

/// \brief Generate partition counts
///
/// \param num_partitions number of partitions
/// \param total_links total count to apportion
/// \return vector<size_t> of length num_partitions whose sum is total_links
std::vector<size_t> make_partition_counts(size_t num_partitions, size_t total_links, LinkDistribution dist) {
  std::vector<size_t> counts;
  if (num_partitions == 0) return counts;
  if (total_links == 0) return std::vector<size_t>(num_partitions, 0);

  if (dist == LinkDistribution::EQUAL) {
    // Uniform distribution: each partition gets the same number of links, up to a remainder
    counts.resize(num_partitions, total_links / num_partitions);
    size_t remainder = total_links % num_partitions;
    for (size_t i = 0; i < remainder; ++i) {
      counts[i] += 1;  // Distribute the remainder evenly
    }
    return counts;
  } else if (dist == LinkDistribution::LOG_NORMAL) {
    double mu = 0.0;     ///< Mean of the log-normal distribution
    double sigma = 2.0;  ///< Standard deviation of the log-normal distribution
    std::mt19937_64 rng(1234);
    std::lognormal_distribution<double> dist(mu, sigma);

    std::vector<double> weights(num_partitions);
    for (size_t i = 0; i < num_partitions; ++i) {
      // Draw strictly positive weight; lognormal already guarantees > 0.
      weights[i] = dist(rng);
    }

    return apportion_largest_remainder(weights, total_links);
  } else {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                        std::string("Unsupported link distribution type: ") + std::to_string(static_cast<int>(dist)));
  }

  return counts;  // Impossible to reach.
}

/// \brief The control parameters for a single performance test.
struct TestParameters {
  size_t num_entities_per_rank;
  size_t num_links;
  LinkDistribution link_distribution;
  LinkedEntityRanksType linked_entity_ranks_type;
  stk::mesh::EntityRank link_rank;
  size_t link_dimensionality;
  size_t num_link_partitions;
  double id_locality;      ///< 0.0 = Gaussian, 1.0 = Uniform, in between = linear interpolation
  double id_sigma_bucket;  ///< Standard deviation of the Gaussian distribution for bucket id selection
  double id_sigma_entity;  ///< Standard deviation of the Gaussian distribution for entity ord selection

  std::string to_string() const {
    std::ostringstream s;
    s << "num_entities_per_rank=" << num_entities_per_rank << ", num_links=" << num_links
      << ", link_distribution=" << link_distribution << ", linked_entity_ranks_type=" << linked_entity_ranks_type
      << ", link_dimensionality=" << link_dimensionality << ", num_link_partitions=" << num_link_partitions
      << ", id_locality=" << id_locality << ", id_sigma_bucket=" << id_sigma_bucket
      << ", id_sigma_entity=" << id_sigma_entity;
    return s.str();
  }
};

/// \brief Shared context for a single test.
struct TestContext {
  std::shared_ptr<MetaData> meta_data;
  std::shared_ptr<BulkData> bulk_data;
  std::shared_ptr<NewLinkMetaData> link_meta_data;
  NewNgpLinkData link_data;
  stk::mesh::PartVector link_parts;  ///< One per link partition
};

void setup_mesh_and_metadata(TestContext& context, const TestParameters& params) {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  context.meta_data = builder.create_meta_data();
  context.meta_data->use_simple_fields();

  context.bulk_data = builder.create_bulk_data(context.meta_data);

  context.link_meta_data = declare_link_meta_data_ptr(*context.meta_data, "ALL_LINKS", params.link_rank);
  context.link_data = declare_ngp_link_data(*context.bulk_data, *context.link_meta_data);
}

void declare_link_parts(TestContext& context, const TestParameters& params) {
  context.link_parts.clear();
  for (size_t i = 0; i < params.num_link_partitions; ++i) {
    std::string part_name = "LINK_PART_" + std::to_string(i);
    stk::mesh::Part* link_part = &context.link_meta_data->declare_link_part(part_name, params.link_dimensionality);
    context.link_parts.push_back(link_part);
  }

  context.meta_data->commit();
}

void declare_entities(TestContext& context, const TestParameters& params) {
  size_t num_ranks = context.bulk_data->mesh_meta_data().entity_rank_count();
  std::vector<std::vector<stk::mesh::EntityId>> requested_ids(num_ranks);
  for (size_t i = 0; i < num_ranks; ++i) {
    stk::topology::rank_t rank = static_cast<stk::topology::rank_t>(i);
    context.bulk_data->generate_new_ids(rank, params.num_entities_per_rank, requested_ids[i]);
  }

  // generating 'owned' entities
  stk::mesh::PartVector add_parts;
  add_parts.push_back(&context.bulk_data->mesh_meta_data().locally_owned_part());

  for (size_t i = 0; i < num_ranks; ++i) {
    stk::topology::rank_t rank = static_cast<stk::topology::rank_t>(i);
    std::vector<stk::mesh::Entity> new_entities;
    context.bulk_data->declare_entities(rank, requested_ids[i], add_parts, new_entities);
  }
}

void declare_links(TestContext& context, const TestParameters& params) {
  std::vector<size_t> partition_counts =
      make_partition_counts(params.num_link_partitions, params.num_links, params.link_distribution);

  // Validate partition counts
  MUNDY_THROW_ASSERT(partition_counts.size() == params.num_link_partitions, std::logic_error,
                     "Partition counts size does not match number of link partitions.");
  size_t count = std::accumulate(partition_counts.begin(), partition_counts.end(), 0);
  MUNDY_THROW_ASSERT(count == params.num_links, std::logic_error,
                     "Total number of links does not match requested number of links.");

  // Declare links in each partition
  for (size_t i = 0; i < params.num_link_partitions; ++i) {
    std::vector<stk::mesh::EntityId> requested_ids;
    context.bulk_data->generate_new_ids(params.link_rank, partition_counts[i], requested_ids);

    stk::mesh::PartVector add_parts{context.link_parts[i], &context.link_meta_data->universal_link_part()};
    stk::mesh::EntityVector new_entities;
    context.bulk_data->declare_entities(params.link_rank, requested_ids, add_parts, new_entities);
  }
}

/// \brief Weighted reservoir sampler (Efraimidis–Spirakis A-ExpJ).
/// \details
/// Selects \c k distinct indices from \c {0..n-1} with probability proportional
/// to on-demand weights w(j) > 0, using one streaming pass, O(n log k) time,
/// and O(k) memory. You provide weights via a callable at \c consider().
/// Keys are K_j = -log(U_j) / w(j), and we retain the k smallest keys.
///
/// Usage:
/// \code
/// WeightedSampler sampler(/*k=*/dim);
/// std::mt19937_64 gen(42);
/// auto w = [&](size_t j){ /* compute positive weight */ };
/// for (size_t j = 0; j < n_points; ++j) sampler.consider(j, gen, w);
/// auto members = sampler.extract_sorted(); // size()==dim
/// \endcode
template <typename IndexType>
class WeightedSampler {
 public:
  using index_type = IndexType;
  using our_size_t = size_t;

  /// \brief Construct with a target sample size k.
  explicit WeightedSampler(our_size_t capacity) : capacity_(capacity) {
  }

  /// \brief Return target sample size k.
  our_size_t sample_size() const {
    return capacity_;
  }

  /// \brief Set target sample size k and clear any existing contents.
  void set_sample_size(our_size_t capacity) {
    capacity_ = capacity;
    clear();
  }

  /// \brief Remove all currently considered items.
  void clear() {
    Heap empty;
    heap_.swap(empty);  // clear trick
  }

  /// \brief Number of items currently held in the reservoir (<= k).
  our_size_t size() const {
    return static_cast<our_size_t>(heap_.size());
  }

  /// \brief True if no items are currently held.
  bool empty() const {
    return heap_.empty();
  }

  /// \brief Stream a candidate item into the sampler.
  ///
  /// \param item          Candidate object.
  /// \param rng        Random number generator.
  template <class RNG>
  void consider(index_type item, RNG& rng, double weight) {
    if (capacity_ == 0) return;

    if (!(weight > 0.0)) return;  // skip zero/nonpositive or NaN

    // Draw U ~ (0,1); clamp away from 0 to avoid log(0).
    double u = rng.template rand<double>();
    const double tiny = math::get_zero_tolerance<double>() * 10;
    if (u <= tiny) u = tiny;

    const double key = -std::log(u) / weight;  // smaller is better

    if (heap_.size() < capacity_) {
      heap_.emplace(key, item);
    } else if (key < heap_.top().first) {
      heap_.pop();
      heap_.emplace(key, item);
    }
  }

  /// \brief Extract sampled indices and clear the sampler.
  /// \return Vector of selected indices, size <= k.
  std::vector<index_type> extract_indices() {
    std::vector<index_type> idx;
    idx.reserve(heap_.size());
    while (!heap_.empty()) {
      idx.push_back(heap_.top().second);
      heap_.pop();
    }
    return idx;
  }

 private:
  using KeyIndex = std::pair<double, index_type>;  // (key, index)
  using Heap = std::priority_queue<KeyIndex>;      // max-heap by key

  our_size_t capacity_{0};
  Heap heap_;
};

void connect_entities_and_links(TestContext& context, const TestParameters& params) {
  // At this point, all of the entities and links have been declared and the mesh is no longer in a modification cycle.

  // 1.0. Choose the ranks for each downward linked entity. The modes are
  //   same: All entities of NODE_RANK
  //   random: All entities of random ranks
  //   one_to_many: First entity is of a ELEM_RANK rank, the rest are of NODE_RANK
  stk::mesh::BulkData& bulk_data = *context.bulk_data;
  std::vector<stk::mesh::EntityRank> linked_entity_ranks(params.link_dimensionality);
  MUNDY_THROW_REQUIRE(params.link_dimensionality > 0, std::invalid_argument, "Link dimensionality must be non-zero.");

  if (params.linked_entity_ranks_type == LinkedEntityRanksType::SAME) {
    std::fill(linked_entity_ranks.begin(), linked_entity_ranks.end(), stk::topology::NODE_RANK);
  } else if (params.linked_entity_ranks_type == LinkedEntityRanksType::RANDOM) {
    std::mt19937_64 rng(1234);
    std::uniform_int_distribution<int> rank_dist(0, bulk_data.mesh_meta_data().entity_rank_count() - 1);
    for (size_t d = 0; d < params.link_dimensionality; ++d) {
      linked_entity_ranks[d] = static_cast<stk::mesh::EntityRank>(rank_dist(rng));
    }
  } else if (params.linked_entity_ranks_type == LinkedEntityRanksType::ONE_TO_MANY) {
    MUNDY_THROW_REQUIRE(params.link_dimensionality >= 2, std::invalid_argument,
                        "Link dimensionality must be at least 2 for one-to-many linked entity ranks.");
    linked_entity_ranks[0] = stk::topology::ELEM_RANK;
    std::fill(linked_entity_ranks.begin() + 1, linked_entity_ranks.end(), stk::topology::NODE_RANK);
  } else {
    MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                        std::string("Unsupported linked entity ranks type: ") +
                            std::to_string(static_cast<int>(params.linked_entity_ranks_type)));
  }

  // 1. Get all non-link entities in the mesh.
  stk::mesh::BucketVector rank_buckets[stk::topology::NUM_RANKS];
  stk::mesh::Selector link_selector = stk::mesh::selectUnion(context.link_parts);
  for (size_t i = 0; i < stk::topology::NUM_RANKS; ++i) {
    stk::topology::rank_t rank = static_cast<stk::topology::rank_t>(i);
    rank_buckets[i] = bulk_data.get_buckets(rank, !link_selector);
  }

  // 2.
  const size_t link_dimensionality = params.link_dimensionality;
  LinkedEntityRanksType linked_entity_ranks_type = params.linked_entity_ranks_type;
  const double id_sigma_bucket = params.id_sigma_bucket;
  const double id_sigma_entity = params.id_sigma_entity;
  const double id_locality = params.id_locality;
  auto link_data = context.link_data;

  const stk::mesh::BucketVector& link_buckets = bulk_data.get_buckets(params.link_rank, link_selector);
  const size_t num_link_buckets = link_buckets.size();
  for (size_t link_bucket_id = 0; link_bucket_id < num_link_buckets; ++link_bucket_id) {
    stk::mesh::Bucket& link_bucket = *link_buckets[link_bucket_id];

    // 2.1. For each link bucket, preselect L entity buckets to maybe draw from.
    stk::mesh::BucketVector buckets_to_maybe_draw_from[stk::topology::NUM_RANKS];
    openrand::Philox bucket_rng(link_bucket_id, 0);  // no need for counter to be non-zero.

    for (size_t i = 0; i < stk::topology::NUM_RANKS; ++i) {
      const size_t num_entity_buckets = rank_buckets[i].size();
      const size_t num_buckets_to_consider =
          std::max(static_cast<size_t>(1u),
                   std::min(num_entity_buckets, static_cast<size_t>(std::ceil(6.0 * params.id_sigma_bucket))));
      const double center_in_bucket_space =
          (static_cast<double>(link_bucket_id) + 0.5) *
          (static_cast<double>(num_entity_buckets) / static_cast<double>(num_link_buckets));

      WeightedSampler<size_t> bucket_sampler(num_buckets_to_consider);
      for (size_t entity_bucket_id = 0; entity_bucket_id < num_entity_buckets; ++entity_bucket_id) {
        const double g = std::exp(
            -0.5 * std::pow((static_cast<double>(entity_bucket_id) - center_in_bucket_space) / id_sigma_bucket, 2.0));
        const double weight = (1.0 - id_locality) + id_locality * g;
        bucket_sampler.consider(entity_bucket_id, bucket_rng, weight);
      }

      // 2.2. Extract the selected entity buckets and store them.
      std::vector<size_t> chosen_bucket_ids = bucket_sampler.extract_indices();
      for (const auto& bucket_id : chosen_bucket_ids) {
        MUNDY_THROW_ASSERT(bucket_id < num_entity_buckets, std::logic_error, "Chosen bucket id is out of range.");
        stk::mesh::Bucket* entity_bucket = rank_buckets[i][bucket_id];
        MUNDY_THROW_ASSERT(entity_bucket != nullptr, std::logic_error, "Chosen bucket is a null pointer.");
        buckets_to_maybe_draw_from[i].push_back(rank_buckets[i][bucket_id]);
      }
    }

    // 2.3. For each link in said bucket...
    size_t num_links_in_bucket = link_bucket.size();
    for (size_t link_ord = 0; link_ord < num_links_in_bucket; ++link_ord) {
      stk::mesh::Entity link_entity = link_bucket[link_ord];
      openrand::Philox link_rng(link_ord, 0);  // no need for counter to be non-zero.

      if (linked_entity_ranks_type == LinkedEntityRanksType::SAME) {
        WeightedSampler<stk::mesh::Entity> node_entity_sampler(link_dimensionality);
        for (stk::mesh::Bucket* entity_bucket : buckets_to_maybe_draw_from[stk::topology::NODE_RANK]) {
          const size_t num_entities_in_bucket = entity_bucket->size();
          const double center_in_entity_space =
              (static_cast<double>(link_ord) + 0.5) *
              (static_cast<double>(num_entities_in_bucket) / static_cast<double>(num_links_in_bucket));

          for (size_t entity_ord = 0; entity_ord < num_entities_in_bucket; ++entity_ord) {
            const double g = std::exp(
                -0.5 * std::pow((static_cast<double>(entity_ord) - center_in_entity_space) / id_sigma_entity, 2.0));
            const double weight = (1.0 - id_locality) + id_locality * g;
            MUNDY_THROW_ASSERT(bulk_data.is_valid((*entity_bucket)[entity_ord]), std::logic_error,
                               std::string("Entity in bucket is not valid. Bucket idx: ") +
                                   std::to_string(entity_bucket->bucket_id()) +
                                   ", entity ord: " + std::to_string(entity_ord));
            node_entity_sampler.consider((*entity_bucket)[entity_ord], link_rng, weight);
          }
        }

        // Declare the link relations between the link and the chosen entities.
        std::vector<stk::mesh::Entity> selected_nodes = node_entity_sampler.extract_indices();
        MUNDY_THROW_ASSERT(selected_nodes.size() <= link_dimensionality, std::logic_error,
                           "Selected nodes size must not exceed link dimensionality. It may be less when there aren't "
                           "enough nodes in the mesh.");
        for (unsigned d = 0; d < selected_nodes.size(); ++d) {
          stk::mesh::Entity selected_node = selected_nodes[d];
          MUNDY_THROW_ASSERT(bulk_data.is_valid(selected_node), std::logic_error, "Selected entity is not valid.");
          link_data.declare_relation_host(link_entity, selected_node, d);
        }

      } else if (linked_entity_ranks_type == LinkedEntityRanksType::RANDOM) {
        std::array<unsigned, stk::topology::NUM_RANKS> num_entities_per_rank{0};
        std::array<unsigned, stk::topology::NUM_RANKS + 1> num_entities_per_rank_shift{0};
        for (unsigned d = 0; d < link_dimensionality; ++d) {
          unsigned selected_rank = link_rng.uniform<unsigned>(0, stk::topology::NUM_RANKS /* non-inclusive*/);
          MUNDY_THROW_ASSERT(selected_rank < stk::topology::NUM_RANKS, std::logic_error,
                             "Selected rank is out of range.");
          num_entities_per_rank[selected_rank] += 1;
        }
        for (size_t i = 0; i < stk::topology::NUM_RANKS; ++i) {
          num_entities_per_rank_shift[i + 1] = num_entities_per_rank_shift[i] + num_entities_per_rank[i];
        }
        MUNDY_THROW_ASSERT(num_entities_per_rank_shift[stk::topology::NUM_RANKS] == link_dimensionality,
                           std::logic_error, "Sum of entities per rank must equal link dimensionality.");

        for (size_t i = 0; i < stk::topology::NUM_RANKS; ++i) {
          WeightedSampler<stk::mesh::Entity> entity_sampler(num_entities_per_rank[i]);
          for (stk::mesh::Bucket* entity_bucket : buckets_to_maybe_draw_from[i]) {
            const size_t num_entities_in_bucket = entity_bucket->size();
            const double center_in_entity_space =
                (static_cast<double>(link_ord) + 0.5) *
                (static_cast<double>(num_entities_in_bucket) / static_cast<double>(num_links_in_bucket));

            for (size_t entity_ord = 0; entity_ord < num_entities_in_bucket; ++entity_ord) {
              const double g = std::exp(
                  -0.5 * std::pow((static_cast<double>(entity_ord) - center_in_entity_space) / id_sigma_entity, 2.0));
              const double weight = (1.0 - id_locality) + id_locality * g;
              entity_sampler.consider((*entity_bucket)[entity_ord], link_rng, weight);
            }
          }

          // Declare the link relations between the link and the chosen entities.
          std::vector<stk::mesh::Entity> selected_entities = entity_sampler.extract_indices();
          MUNDY_THROW_ASSERT(selected_entities.size() <= num_entities_per_rank[i], std::logic_error,
                             "Selected entities size must not exceed number of entities per rank. It may be less when "
                             "there aren't enough entities in the mesh.");
          for (unsigned j = 0; j < selected_entities.size();
               ++j) {  // We might end up with holes in the downward connections.
            stk::mesh::Entity selected_entity = selected_entities[j];
            MUNDY_THROW_ASSERT(bulk_data.is_valid(selected_entity), std::logic_error, "Selected entity is not valid.");
            link_data.declare_relation_host(link_entity, selected_entity, j + num_entities_per_rank_shift[i]);
          }
        }

      } else if (linked_entity_ranks_type == LinkedEntityRanksType::ONE_TO_MANY) {
        WeightedSampler<stk::mesh::Entity> node_entity_sampler(link_dimensionality - 1);
        WeightedSampler<stk::mesh::Entity> elem_entity_sampler(1);
        for (stk::mesh::Bucket* entity_bucket : buckets_to_maybe_draw_from[stk::topology::NODE_RANK]) {
          const size_t num_entities_in_bucket = entity_bucket->size();
          const double center_in_entity_space =
              (static_cast<double>(link_ord) + 0.5) *
              (static_cast<double>(num_entities_in_bucket) / static_cast<double>(num_links_in_bucket));

          for (size_t entity_ord = 0; entity_ord < num_entities_in_bucket; ++entity_ord) {
            const double g = std::exp(
                -0.5 * std::pow((static_cast<double>(entity_ord) - center_in_entity_space) / id_sigma_entity, 2.0));
            const double weight = (1.0 - id_locality) + id_locality * g;
            node_entity_sampler.consider((*entity_bucket)[entity_ord], link_rng, weight);
          }
        }
        for (stk::mesh::Bucket* entity_bucket : buckets_to_maybe_draw_from[stk::topology::ELEM_RANK]) {
          const size_t num_entities_in_bucket = entity_bucket->size();
          const double center_in_entity_space =
              (static_cast<double>(link_ord) + 0.5) *
              (static_cast<double>(num_entities_in_bucket) / static_cast<double>(num_links_in_bucket));

          for (size_t entity_ord = 0; entity_ord < num_entities_in_bucket; ++entity_ord) {
            const double g = std::exp(
                -0.5 * std::pow((static_cast<double>(entity_ord) - center_in_entity_space) / id_sigma_entity, 2.0));
            const double weight = (1.0 - id_locality) + id_locality * g;
            elem_entity_sampler.consider((*entity_bucket)[entity_ord], link_rng, weight);
          }
        }

        // Declare the link relations between the link and the chosen element
        std::vector<stk::mesh::Entity> selected_elems = elem_entity_sampler.extract_indices();
        stk::mesh::Entity selected_elem = selected_elems[0];
        MUNDY_THROW_ASSERT(bulk_data.is_valid(selected_elem), std::logic_error, "Selected entity is not valid.");
        link_data.declare_relation_host(link_entity, selected_elem, 0);

        // Declare the link relations between the link and the chosen nodes
        std::vector<stk::mesh::Entity> selected_nodes = node_entity_sampler.extract_indices();
        MUNDY_THROW_ASSERT(selected_nodes.size() <= link_dimensionality - 1, std::logic_error,
                           "Number of selected nodes must not exceed link dimensionality - 1. It may be less when "
                           "there aren't enough nodes in the mesh.");
        for (unsigned j = 0; j < selected_nodes.size(); ++j) {
          stk::mesh::Entity selected_node = selected_nodes[j];
          MUNDY_THROW_ASSERT(bulk_data.is_valid(selected_node), std::logic_error, "Selected entity is not valid.");
          link_data.declare_relation_host(link_entity, selected_node, j + 1);  // Start at 1 since 0 is the element.
        }

      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument,
                           std::string("Unsupported linked entity ranks type: ") +
                               std::to_string(static_cast<int>(linked_entity_ranks_type)));
      }
    }
  }
}

/// \brief The driver for a single performance test given a set of parameters.
void run_test(ankerl::nanobench::Bench& bench, const TestParameters& params) {
  TestContext context;
  setup_mesh_and_metadata(context, params);
  declare_link_parts(context, params);

  context.bulk_data->modification_begin();
  declare_entities(context, params);
  declare_links(context, params);
  context.bulk_data->modification_end();

  connect_entities_and_links(context, params);
  context.link_data.sync_to_device();

  // Benchmark the get_or_create_crs_partitions() call
  // bench.run(params.to_string(),
  //           [&]() { context.link_data.get_or_create_crs_partitions(context.link_meta_data->universal_link_part());
  //           });
}

/// \brief The driver that runs the performance tests over a range of parameters.
void run_tests() {
  // 7 * 7 * 2 * 3 * 2 * 13 * 13 * 6 * 7 * 7 = 29,215,368
  // std::vector<size_t> num_entities_per_rank_iter{1, 10, 100, 1'000, 10'000, 100'000, 1'000'000};           // 7 tests
  // std::vector<size_t> num_links_iter{1, 10, 100, 1'000, 10'000, 100'000, 1'000'000};                       // 7 tests
  // std::vector<LinkDistribution> link_distribution_iter{LinkDistribution::EQUAL,                            //
  //                                                      LinkDistribution::LOG_NORMAL};                      // 2 tests
  // std::vector<LinkedEntityRanksType> linked_entity_ranks_iter{LinkedEntityRanksType::SAME,                 //
  //                                                             LinkedEntityRanksType::RANDOM,               //
  //                                                             LinkedEntityRanksType::ONE_TO_MANY};         // 3 tests
  // std::vector<stk::mesh::EntityRank> link_ranks_iter{stk::topology::NODE_RANK, stk::topology::ELEM_RANK};  // 2 tests
  // std::vector<unsigned> link_dimensionality_iter{2, 3, 4, 6, 8, 10, 20, 30, 40, 60, 80, 100};              // 12
  // tests std::vector<unsigned> num_link_partitions_iter{1, 2, 3, 4, 6, 8, 10, 20, 30, 40, 60, 80, 100};           //
  // 13 tests std::vector<double> id_locality_iter{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}; // 6 tests std::vector<double>
  // id_sigma_bucket_iter{0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0};  // 7 tests (in bucket_id space) std::vector<double>
  // id_sigma_entity_iter{0., 2., 4., 8., 16., 32., 64.};        // 7 tests (in bucket_ord space)

  std::vector<size_t> num_entities_per_rank_iter{1, 1'000'000};                                            // 7 tests
  std::vector<size_t> num_links_iter{1, 1'000'000};                                                        // 7 tests
  std::vector<LinkDistribution> link_distribution_iter{LinkDistribution::EQUAL,                            //
                                                       LinkDistribution::LOG_NORMAL};                      // 2 tests
  std::vector<LinkedEntityRanksType> linked_entity_ranks_iter{LinkedEntityRanksType::SAME,                 //
                                                              LinkedEntityRanksType::RANDOM,               //
                                                              LinkedEntityRanksType::ONE_TO_MANY};         // 3 tests
  std::vector<stk::mesh::EntityRank> link_ranks_iter{stk::topology::NODE_RANK, stk::topology::ELEM_RANK};  // 2 tests
  std::vector<unsigned> link_dimensionality_iter{2, 100};                                                  // 12 tests
  std::vector<unsigned> num_link_partitions_iter{1, 100};                                                  // 13 tests
  std::vector<double> id_locality_iter{0.0};                                                               // 6 tests
  std::vector<double> id_sigma_bucket_iter{0.0};  // 7 tests (in bucket_id space)
  std::vector<double> id_sigma_entity_iter{0.};   // 7 tests (in bucket_ord space)

  // Setup the I/O and bench
  std::ofstream csv("link_data_perf_results.csv");
  csv << "num_entities_per_rank, num_links, link_distribution, linked_entity_ranks_type, "
         "link_dimensionality, num_link_partitions, id_locality, id_sigma_bucket, id_sigma_entity, ns_per_op\n";

  ankerl::nanobench::Bench bench;
  bench
      .output(nullptr)  // no console output
      .title("Link Data Performance Tests")
      .unit("op")
      .warmup(10)
      .minEpochTime(std::chrono::milliseconds(50))  // stable-ish per row
      .epochs(1)                                    // rely on time-based epochs to cap cost
      .performanceCounters(false)                   // avoid overhead at million-scale
      .relative(false);

  for (const size_t num_entities_per_rank : num_entities_per_rank_iter) {
    for (const size_t num_links : num_links_iter) {
      for (const LinkDistribution link_distribution : link_distribution_iter) {
        for (const LinkedEntityRanksType linked_entity_ranks_type : linked_entity_ranks_iter) {
          for (const stk::mesh::EntityRank link_rank : link_ranks_iter) {
            for (const unsigned link_dimensionality : link_dimensionality_iter) {
              for (const unsigned num_link_partitions : num_link_partitions_iter) {
                for (const double id_locality : id_locality_iter) {
                  for (const double id_sigma_bucket : id_sigma_bucket_iter) {
                    for (const double id_sigma_entity : id_sigma_entity_iter) {
                      TestParameters params{.num_entities_per_rank = num_entities_per_rank,
                                            .num_links = num_links,
                                            .link_distribution = link_distribution,
                                            .linked_entity_ranks_type = linked_entity_ranks_type,
                                            .link_rank = link_rank,
                                            .link_dimensionality = link_dimensionality,
                                            .num_link_partitions = num_link_partitions,
                                            .id_locality = id_locality,
                                            .id_sigma_bucket = id_sigma_bucket,
                                            .id_sigma_entity = id_sigma_entity};
                      std::cout << "Running test with parameters: " << params.to_string() << std::endl;
                      run_test(bench, params);

                      // After bench.run(), the most recent results are available via results().back()
                      if (!bench.results().empty()) {
                        const auto& rlast = bench.results().back();
                        double ns_per_op = rlast.average(ankerl::nanobench::Result::Measure::elapsed);

                        // Output the results
                        std::ofstream csv("link_data_perf_results.csv", std::ios_base::app);
                        csv << params.num_entities_per_rank << ", " << params.num_links << ", "
                            << static_cast<int>(params.link_distribution) << ", " << static_cast<int>(params.link_rank)
                            << ", " << params.link_dimensionality << ", " << params.num_link_partitions << ", "
                            << params.id_locality << ", " << params.id_sigma_bucket << ", " << params.id_sigma_entity
                            << ", " << ns_per_op << "\n";
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

}  // namespace mesh

}  // namespace mundy

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  mundy::mesh::run_tests();

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}