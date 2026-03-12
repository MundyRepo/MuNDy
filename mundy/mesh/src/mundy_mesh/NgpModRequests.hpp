// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
//
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is empty software: you can redistribute it and/or modify it under the terms of the GNU General Public License
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

#ifndef MUNDY_MESH_NGPMODREQUESTS_HPP_
#define MUNDY_MESH_NGPMODREQUESTS_HPP_

// C++ core
#include <array>
#include <concepts>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_mesh/impl/PartitionKey.hpp>  // for mundy::mesh::impl::PartitionKey, mundy::mesh::impl::get_partition_key
#include <mundy_utils/NgpView.hpp>           // for mundy::NgpViewT
#include <mundy_utils/StringLiteral.hpp>     // for mundy::StringLiteral, mundy::make_string_literal
#include <mundy_utils/throw_assert.hpp>
#include <mundy_utils/variant.hpp>  // for mundy::variant

namespace mundy {

namespace mesh {

/// \brief A helper interface for requesting mesh modifications from the device.
/// This is intended to be used in a three-stage command pattern:
/// ## Stage 1. Claim tickets
///   Just like a ticketing system at a deli counter, you first claim tickets from a ticketing machine. You haven't
///   actually made any purchases yet but you have stated your intent to make a certain number of orders (requests).
///
/// ## Stage 2. Make requests
///   After all of your claims are in and you've finalized the counts, you can then use your claimed tickets
///   to make requests.
///
/// ## Stage 3. Fetch
///   After all requests are processed, use the same ticket to fetch the result of your request (if any).
///
/// During each stage, there are multiple types of requests you can make (requesting entities, requesting connections,
/// destroying entities, etc). Each type of request has its own interface and its own ticket issuer. For example, if you
/// want to request 10 new entities and 10 new connections, you would claim 10 tickets from the entity ticket issuer and
/// 10 tickets from the connection ticket issuer. You would then use the entity tickets to make your entity requests and
/// the connection tickets to make your connection requests (potentially requesting connections to requested entities),
/// allowing you to mix and match requests as needed.
///
/// \note For the time being, you must use each ticket you claim (aka, you cannot over-claim).
///
/// \note You can claim tickets and make requests from either the host or device, but not both simultaneously.
///
/// Example usage 1: Create N spheres (N elems + N nodes) and connect each sphere to its node
/// \code{.cpp}
///  size_t num_spheres = ...;
///  stk::mesh::BulkData& bulk_data = ...;
///  stk::mesh::Part& sphere_part = ...;
///
///  NgpModRequests reqs;
///  auto req_spheres = reqs.request_entities_new_ids(sphere_part);
///  auto req_conns = reqs.request_connections();
///
///  // Stage 1: Claim tickets
///  reqs.activate_host();  // Ensures that claim can be called from host
///  req_spheres.element_tickets().claim(num_spheres);
///  req_spheres.node_tickets().claim(num_spheres);
///  req_conns.tickets().claim(num_spheres);
///  reqs.finalize_counts();
///
///  // Stage 2: Request
///  reqs.activate_device();  // Ensures that request can be made from device
///  Kokkos::parallel_for(
///      "RequestSpheres", Kokkos::RangePolicy<size_t>(0, num_spheres), KOKKOS_LAMBDA(size_t ticket) {
///        FutureEntity future_sphere = req_spheres.request_element(ticket);
///        FutureEntity future_node = req_spheres.request_node(ticket);
///        req_conns.request(ticket, future_sphere, future_node, 0);
///      });
///  reqs.process_requests(bulk_data);
///
///  // Stage 3: Fetch
///  Kokkos::parallel_for(
///      "UseSpheres", Kokkos::RangePolicy<size_t>(0, num_spheres), KOKKOS_LAMBDA(size_t ticket) {
///        stk::mesh::Entity sphere = req_spheres.get_entity(ticket, stk::topology::ELEMENT_RANK);
///        stk::mesh::Entity node = req_spheres.get_entity(ticket, stk::topology::NODE_RANK);
///
///        // Do something with sphere and its node...
///      });
/// \endcode
///
/// Example usage 2: Divide all bacteria that exceeds twice the starting length into two bacteria
/// Bacteria are spherocylinder segments (PARTICLE topology elements with a single node at their center)
/// They have a "length", "radius", and "orientation" field
/// Their center nodes have a "coords" field
///
/// Because division is delayed, we also need bacteria to store the ticket for fetching their new child later
///
/// We will assume that these are already declared and that you have wrapped them into a single bacteria aggregate
/// \code{.cpp}
///   auto ngp_bacteria_data = get_updated_ngp_aggregate(bacteria_data);
///   auto ngp_mesh = /* refreshed NgpMesh view for your workflow */;
///
///   size_t num_bacteria = ...;  // total number of bacteria
///   double starting_length = ...;
///   stk::mesh::BulkData& bulk_data = ...;
///   stk::mesh::Part& bacteria_part = ...;
///
///   NgpModRequests reqs;
///   auto req_bacteria = reqs.request_entities_new_ids(bacteria_part);
///   auto req_conns = reqs.request_connections();
///
///   // Stage 1: Claim tickets
///   reqs.activate_device();
///   stk::mesh::for_each_entity_run(
///       ngp_mesh, stk::topology::ELEM_RANK, ngp_bacteria_data.get<SELECTOR>(),
///       KOKKOS_LAMBDA(stk::mesh::FastMeshIndex bacteria_index) {
///         double centerline_length = ngp_bacteria_data.get<LENGTH>(bacteria_index);
///         double radius = ngp_bacteria_data.get<RADIUS>(bacteria_index);
///         double total_length = centerline_length + 2.0 * radius;
///         if (total_length > 2.0 * starting_length) {
///           size_t ticket = req_bacteria.element_tickets().claim();
///           req_bacteria.node_tickets().claim();
///           req_conns.tickets().claim();
///
///           // Store the ticket for later use
///           ngp_bacteria_data.get<CHILD_TICKET>(bacteria_index) = ticket;
///         }
///       });
///   reqs.finalize_counts();
///
///   // Stage 2: Request
///   reqs.activate_device();
///   stk::mesh::for_each_entity_run(
///       ngp_mesh, stk::topology::ELEM_RANK, ngp_bacteria_data.get<SELECTOR>(),
///       KOKKOS_LAMBDA(stk::mesh::FastMeshIndex bacteria_index) {
///         double centerline_length = ngp_bacteria_data.get<LENGTH>(bacteria_index);
///         double radius = ngp_bacteria_data.get<RADIUS>(bacteria_index);
///         double total_length = centerline_length + 2.0 * radius;
///         if (total_length > 2.0 * starting_length) {
///           // One new bacteria since the parent becomes one of the children
///           size_t ticket = ngp_bacteria_data.get<CHILD_TICKET>(bacteria_index);
///           FutureEntity future_bacteria = req_bacteria.request_element(ticket);
///           FutureEntity future_node = req_bacteria.request_node(ticket);
///           req_conns.request(ticket, future_bacteria, future_node, 0);
///         }
///       });
///   reqs.process_requests(bulk_data);
///
///   // Stage 3: Fetch
///   stk::mesh::for_each_entity_run(
///       ngp_mesh, stk::topology::ELEM_RANK, ngp_bacteria_data.get<SELECTOR>(),
///       KOKKOS_LAMBDA(stk::mesh::FastMeshIndex bacteria_index) {
///         double centerline_length = ngp_bacteria_data.get<LENGTH>(bacteria_index);
///         double radius = ngp_bacteria_data.get<RADIUS>(bacteria_index);
///         double total_length = centerline_length + 2.0 * radius;
///         if (total_length > 2.0 * starting_length) {
///           size_t ticket = ngp_bacteria_data.get<CHILD_TICKET>(bacteria_index);
///
///           stk::mesh::Entity new_bacteria = req_bacteria.get_entity(ticket, stk::topology::ELEMENT_RANK);
///           stk::mesh::Entity center_node = req_bacteria.get_entity(ticket, stk::topology::NODE_RANK);
///
///           // The parent becomes the leftmost child and the new bacteria is the rightmost child
///           //
///           // New length = length / 2 - radius
///           // New centers = old center +/- (tangent * (new length / 2 + radius))
///           // ...
///         }
///       });
/// \endcode
///
/// The above design is amicable to a future python interface:
/// \code{.py}
///   # Example usage 1 (but in python): Create N spheres (N elems + N nodes)
///   num_spheres = ...
///   our_proc_id = bulk_data.parallel_rank()
///
///   reqs = NgpModRequests()
///   req_spheres = reqs.request_entities(sphere_part)
///   req_nodes = reqs.request_entities()
///   req_conns = reqs.request_connections()
///
///   # Stage 1: Claim tickets
///   with reqs.host():
///     req_spheres.tickets().claim(num_spheres)
///     req_nodes.tickets().claim(num_spheres)
///     req_conns.tickets().claim(num_spheres)
///
///   reqs.finalize_counts()
///
///   # Stage 2: Request
///   with reqs.device():
///     plan = reqs.plan()  # collects device ops that must be executed
///
///     t = req_spheres.tickets().range(num_spheres)  # 0..num_spheres-1, typed object
///     future_spheres = req_spheres.request(t, our_proc_id)
///     future_nodes = req_nodes.request(t, our_proc_id)
///     req_conns.request(t, future_spheres, future_nodes)
///     ticket_expr1.exec()
///
///   reqs.process_requests();
///
///   # Stage 3: Fetch
///   with reqs.device():
///     ticket_expr2 = make_ticket_expr(0, num_spheres)
///     spheres = req_spheres.get_entity(ticket_expr2)
///     nodes = req_nodes.get_entity(ticket_expr2)
///     center(nodes) = ...
///     radius(spheres) = ...
///     ticket_expr2.exec()
/// \endcode
template <typename NgpMemSpace>
class NgpModRequestsT;

/// \brief A range of tickets issued by a TicketIssuer.
class TicketRange {
 public:
  using ticket_id = size_t;

  TicketRange() = default;
  TicketRange(size_t begin, size_t count) : begin_(begin), count_(count) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION size_t begin() const noexcept { return begin_; }
  KOKKOS_INLINE_FUNCTION size_t end() const noexcept { return begin_ + count_; }
  KOKKOS_INLINE_FUNCTION size_t count() const noexcept { return count_; }
  // clang-format on

 private:
  size_t begin_{0};
  size_t count_{0};
};  // TicketRange

/// \brief Issues tickets for modifications in a given memory space.
///
/// \note This class uses dual-view-like semantics to manage ticket state. The NgpMemSpace is the device memory space
/// but the host can also claim tickets when activated. Importantly, only one memory space can claim tickets at a time
/// and changing the active space acts as a phase boundary that synchronizes ticket state between host and device.
template <typename NgpMemSpace>
class TicketIssuer {
 private:
  using bool_view_t = Kokkos::View<bool, NgpMemSpace>;
  using size_view_t = Kokkos::View<size_t, NgpMemSpace>;

 public:
  using host_space = Kokkos::HostSpace;
  using memory_space = NgpMemSpace;
  using execution_space = typename NgpMemSpace::execution_space;
  using ticket_id = size_t;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor w/ delayed initialization. Call initialize(device_active) to set up.
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer() = default;

  KOKKOS_DEFAULTED_FUNCTION TicketIssuer(const TicketIssuer&) = default;
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer& operator=(const TicketIssuer&) = default;
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer(TicketIssuer&&) = default;
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer& operator=(TicketIssuer&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~TicketIssuer() = default;

  /// \brief Construct a TicketIssuer with the specified initial active memory space.
  TicketIssuer(bool activate_device)
      : active_space_dev_view_("TicketIssuer::active_space_dev_view"),
        ticket_counter_dev_view_("TicketIssuer::ticket_counter_dev_view"),
        count_finalized_dev_view_("TicketIssuer::count_finalized_dev_view"),
        active_space_host_view_(Kokkos::create_mirror_view(active_space_dev_view_)),
        ticket_counter_host_view_(Kokkos::create_mirror_view(ticket_counter_dev_view_)),
        count_finalized_host_view_(Kokkos::create_mirror_view(count_finalized_dev_view_)) {
    // Initialize to device active
    active_space_host_view_() = activate_device;
    ticket_counter_host_view_() = 0;
    count_finalized_host_view_() = false;
    Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    Kokkos::deep_copy(ticket_counter_dev_view_, ticket_counter_host_view_);
    Kokkos::deep_copy(count_finalized_dev_view_, count_finalized_host_view_);
  }

  /// \brief Initialize a default-constructed TicketIssuer.
  /// Cannot be called on an already initialized TicketIssuer, will throw (in debug mode).
  void initialize(bool activate_device = true) {
    // Never allow re-initialization.
    MUNDY_THROW_ASSERT(!(active_space_dev_view_.is_allocated() || ticket_counter_dev_view_.is_allocated() ||
                         count_finalized_dev_view_.is_allocated()),
                       std::runtime_error, "TicketIssuer::initialize() called on already initialized TicketIssuer.");

    active_space_dev_view_ = bool_view_t("TicketIssuer::active_space_dev_view");
    ticket_counter_dev_view_ = size_view_t("TicketIssuer::ticket_counter_dev_view");
    count_finalized_dev_view_ = bool_view_t("TicketIssuer::count_finalized_dev_view");
    ticket_counter_host_view_ = Kokkos::create_mirror_view(ticket_counter_dev_view_);
    active_space_host_view_ = Kokkos::create_mirror_view(active_space_dev_view_);
    count_finalized_host_view_ = Kokkos::create_mirror_view(count_finalized_dev_view_);

    active_space_host_view_() = activate_device;
    ticket_counter_host_view_() = 0;
    count_finalized_host_view_() = false;
    Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    Kokkos::deep_copy(ticket_counter_dev_view_, ticket_counter_host_view_);
    Kokkos::deep_copy(count_finalized_dev_view_, count_finalized_host_view_);
  }
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes ticket state if needed.
  /// While active, calling claim or count from device will result in a throw.
  void activate_host() {
    auto device_is_active = active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      active_space_host_view_() = false;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
      Kokkos::deep_copy(ticket_counter_host_view_, ticket_counter_dev_view_);
    }
  }

  /// \brief Sets the active memory space to device and synchronizes ticket state if needed.
  /// While active, calling claim or count from host will result in a throw.
  void activate_device() {
    auto host_is_active = !active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      active_space_host_view_() = true;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
      Kokkos::deep_copy(ticket_counter_dev_view_, ticket_counter_host_view_);
    }
  }

  /// \brief Synchronize ticket state between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = active_space_host_view_();
    if (device_is_active) {
      Kokkos::deep_copy(ticket_counter_host_view_, ticket_counter_dev_view_);
      Kokkos::deep_copy(count_finalized_host_view_, count_finalized_dev_view_);
    } else {
      Kokkos::deep_copy(ticket_counter_dev_view_, ticket_counter_host_view_);
      Kokkos::deep_copy(count_finalized_dev_view_, count_finalized_host_view_);
    }
  }

  void reset() {
    ticket_counter_host_view_() = 0;
    count_finalized_host_view_() = false;
    Kokkos::deep_copy(ticket_counter_dev_view_, ticket_counter_host_view_);
    Kokkos::deep_copy(count_finalized_dev_view_, count_finalized_host_view_);
  }

  /// \brief Finalize the count of issued tickets
  size_t finalize_count() {
    count_finalized_host_view_() = true;
    Kokkos::deep_copy(count_finalized_dev_view_, count_finalized_host_view_);

    sync();
    return ticket_counter_host_view_();
  }
  //@}

  //! \name Data plane (HOST or DEVICE)
  //@{

  /// \brief Claim N contiguous tickets atomically. Returns a TicketRange representing the claimed tickets.
  /// If you claim more tickets than are available, an exception is thrown (in debug mode).
  KOKKOS_INLINE_FUNCTION TicketRange claim(size_t n) const {
    constexpr auto name = make_string_literal("TicketIssuer::claim");
    assert_active_space<name>();
    assert_count_not_finalized<name>();

    KOKKOS_IF_ON_HOST(size_t start_ticket = Kokkos::atomic_fetch_add(&ticket_counter_host_view_(), n);
                      return TicketRange(start_ticket, n);)
    KOKKOS_IF_ON_DEVICE(size_t start_ticket = Kokkos::atomic_fetch_add(&ticket_counter_dev_view_(), n);
                        return TicketRange(start_ticket, n);)
  }

  /// \brief Claim a single ticket atomically. Returns the claimed ticket ID.
  /// If you claim more tickets than are available, an exception is thrown (in debug mode).
  KOKKOS_INLINE_FUNCTION size_t claim() const {
    constexpr auto name = make_string_literal("TicketIssuer::claim");
    assert_active_space<name>();
    assert_count_not_finalized<name>();

    KOKKOS_IF_ON_HOST(return Kokkos::atomic_fetch_add(&ticket_counter_host_view_(), 1);)
    KOKKOS_IF_ON_DEVICE(return Kokkos::atomic_fetch_add(&ticket_counter_dev_view_(), 1);)
  }

  /// \brief Get the current count of issued tickets atomically.
  KOKKOS_INLINE_FUNCTION size_t count() const {
    constexpr auto name = make_string_literal("TicketIssuer::count");
    assert_active_space<name>();

    KOKKOS_IF_ON_HOST(return ticket_counter_host_view_();)
    KOKKOS_IF_ON_DEVICE(return ticket_counter_dev_view_();)
  }
  //@}

 private:
  //! \name Helper functions
  //@{

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    constexpr bool has_separate_host_and_device_storage =
        !Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible;
    if constexpr (has_separate_host_and_device_storage) {
      KOKKOS_IF_ON_HOST(MUNDY_THROW_ASSERT(
                            /*host_is_active =*/!active_space_host_view_(), std::runtime_error,
                            name + " called from host when device is active.");)
      KOKKOS_IF_ON_DEVICE(MUNDY_THROW_ASSERT(
                              /*device_is_active =*/active_space_dev_view_(), std::runtime_error,
                              name + " called from device when host is active.");)
    }
  }

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_count_not_finalized() const {
    KOKKOS_IF_ON_HOST(MUNDY_THROW_ASSERT(!count_finalized_host_view_(), std::runtime_error,
                                         name + " called after finalize_count() on host.");)
    KOKKOS_IF_ON_DEVICE(MUNDY_THROW_ASSERT(!count_finalized_dev_view_(), std::runtime_error,
                                           name + " called after finalize_count() on device.");)
  }
  //@}

  bool_view_t active_space_dev_view_;
  size_view_t ticket_counter_dev_view_;
  bool_view_t count_finalized_dev_view_;
  bool_view_t::HostMirror active_space_host_view_;
  size_view_t::HostMirror ticket_counter_host_view_;
  bool_view_t::HostMirror count_finalized_host_view_;
};  // TicketIssuer

struct FutureEntity {
  size_t ticket;
  unsigned request_helper_index;
  stk::mesh::EntityRank entity_rank;
  bool has_known_id;
};

template <typename NgpMemSpace, bool HasKnownEntityId>
class NgpRequestEntitiesImplT {
 private:
  using entity_view_t = NgpViewT<stk::mesh::Entity*, NgpMemSpace>;
  using control_space = Kokkos::SharedSpace;

 public:
  using memory_space = NgpMemSpace;
  using ticket_issuer_t = TicketIssuer<NgpMemSpace>;

  //! \name Constructors / Destructors
  //@{

  /// \brief No default constructor. Must provide a unique index for this request helper to distinguish it from others.
  NgpRequestEntitiesImplT() = delete;

  /// \brief Constructor.
  NgpRequestEntitiesImplT(unsigned helper_index) : state_("NgpRequestEntitiesImplT::state") {
    state_().index_ = helper_index;
    state_().active_space_dev_view_ = bool_view_t("NgpRequestEntitiesImplT::active_space_dev_view");
    state_().active_space_host_view_ = Kokkos::create_mirror_view(state_().active_space_dev_view_);
    for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
      state_().ticket_issuer_[rank] = ticket_issuer_t(/*activate_device*/ true);
      if constexpr (HasKnownEntityId) {
        state_().requests_[rank] = request_view_t("NgpRequestEntitiesImplT::requests", 0);
      }
      state_().created_entities_[rank] = entity_view_t("NgpRequestEntitiesImplT::created_entities", 0);
    }
    // Initialize to device active
    state_().active_space_host_view_() = true;
    Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
  }

  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesImplT(const NgpRequestEntitiesImplT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesImplT& operator=(const NgpRequestEntitiesImplT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesImplT(NgpRequestEntitiesImplT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesImplT& operator=(NgpRequestEntitiesImplT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpRequestEntitiesImplT() = default;
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from device
  void activate_host() {
    auto device_is_active = state_().active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      state_().active_space_host_view_() = false;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
        state_().ticket_issuer_[rank].activate_host();
      }
    }
  }

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from host
  void activate_device() {
    auto host_is_active = !state_().active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      state_().active_space_host_view_() = true;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
        state_().ticket_issuer_[rank].activate_device();
      }
    }
  }

  /// \brief Synchronize between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = state_().active_space_host_view_();
    if (device_is_active) {
      for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
        state_().ticket_issuer_[rank].sync();
        if constexpr (HasKnownEntityId) {
          Kokkos::deep_copy(state_().requests_[rank].view_host(), state_().requests_[rank].view_device());
        }
        Kokkos::deep_copy(state_().created_entities_[rank].view_host(), state_().created_entities_[rank].view_device());
      }
    } else {
      for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
        state_().ticket_issuer_[rank].sync();
        if constexpr (HasKnownEntityId) {
          Kokkos::deep_copy(state_().requests_[rank].view_device(), state_().requests_[rank].view_host());
        }
        Kokkos::deep_copy(state_().created_entities_[rank].view_device(), state_().created_entities_[rank].view_host());
      }
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Get our unique index among multiple request helpers.
  unsigned id() const noexcept {
    return state_().index_;
  }

  /// \brief Get the ticket issuer for entity requests of a specific rank.
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& tickets(stk::mesh::EntityRank entity_rank) const {
    constexpr auto name = make_string_literal("NgpRequestEntitiesImplT::tickets");
    assert_valid_rank<name>(entity_rank);
    return state_().ticket_issuer_[entity_rank];
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& node_tickets() const { return tickets(stk::topology::NODE_RANK); }
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& edge_tickets() const { return tickets(stk::topology::EDGE_RANK); }
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& face_tickets() const { return tickets(stk::topology::FACE_RANK); }
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& element_tickets() const { return tickets(stk::topology::ELEMENT_RANK); }
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& constraint_tickets() const { return tickets(stk::topology::CONSTRAINT_RANK); }
  // clang-format on

  /// \brief Record an entity creation request for the new-id variant.
  /// The current process will own the created entity. It will be assigned a new ID by the mesh.
  KOKKOS_INLINE_FUNCTION FutureEntity request(size_t ticket, stk::mesh::EntityRank entity_rank) const
    requires(!HasKnownEntityId)
  {
    constexpr auto name = make_string_literal("NgpRequestEntitiesImplT::request");
    assert_active_space<name>();
    assert_valid_rank<name>(entity_rank);
    assert_ticket_out_of_range<name>(ticket, entity_rank);

    // Nothing actually needs to be recorded for the new-id variant since the ticket alone is enough to identify the
    // request and fetch the created entity later.

    // TODO(palmerb4): The above statement is wrong and the code need updated accordingly. We need to allow users the
    // ability to claim more tickets than they end up using. This means that the ticket issuer alone isn't sufficient to
    // identify the request since there may be "gaps" in the claimed tickets that aren't used for requests. To start, we
    // will force users to only claim exactly as many tickets as they need, and then come back with an update later to
    // allow for more flexible claiming, allowing us to performance test the overhead of allowing for unused tickets.

    return FutureEntity{
        .ticket = ticket, .request_helper_index = state_().index_, .entity_rank = entity_rank, .has_known_id = false};
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION FutureEntity request_node(size_t ticket) const requires(!HasKnownEntityId) { return request(ticket, stk::topology::NODE_RANK); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_edge(size_t ticket) const requires(!HasKnownEntityId) { return request(ticket, stk::topology::EDGE_RANK); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_face(size_t ticket) const requires(!HasKnownEntityId) { return request(ticket, stk::topology::FACE_RANK); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_element(size_t ticket) const requires(!HasKnownEntityId) { return request(ticket, stk::topology::ELEMENT_RANK); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_constraint(size_t ticket) const requires(!HasKnownEntityId) { return request(ticket, stk::topology::CONSTRAINT_RANK); }
  // clang-format on

  /// \brief Record an entity creation request with a specified entity ID for the known-id variant.
  /// The current process will own the created entity. It will be assigned the given ID.
  KOKKOS_INLINE_FUNCTION FutureEntity request(size_t ticket,                      //
                                              stk::mesh::EntityRank entity_rank,  //
                                              stk::mesh::EntityId entity_id) const
    requires(HasKnownEntityId)
  {
    constexpr auto name = make_string_literal("NgpRequestEntitiesImplT::request");
    assert_active_space<name>();
    assert_valid_rank<name>(entity_rank);
    assert_ticket_out_of_range<name>(ticket, entity_rank);

    KOKKOS_IF_ON_HOST(auto& req = state_().requests_[entity_rank].view_host()(ticket);  //
                      req.entity_id = entity_id;)
    KOKKOS_IF_ON_DEVICE(auto& req = state_().requests_[entity_rank].view_device()(ticket);  //
                        req.entity_id = entity_id;)

    return FutureEntity{
        .ticket = ticket, .request_helper_index = state_().index_, .entity_rank = entity_rank, .has_known_id = true};
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION FutureEntity request_node(size_t ticket, stk::mesh::EntityId entity_id) const requires(HasKnownEntityId) { return request(ticket, stk::topology::NODE_RANK, entity_id); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_edge(size_t ticket, stk::mesh::EntityId entity_id) const requires(HasKnownEntityId) { return request(ticket, stk::topology::EDGE_RANK, entity_id); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_face(size_t ticket, stk::mesh::EntityId entity_id) const requires(HasKnownEntityId) { return request(ticket, stk::topology::FACE_RANK, entity_id); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_element(size_t ticket, stk::mesh::EntityId entity_id) const requires(HasKnownEntityId) { return request(ticket, stk::topology::ELEMENT_RANK, entity_id); }
  KOKKOS_INLINE_FUNCTION FutureEntity request_constraint(size_t ticket, stk::mesh::EntityId entity_id) const requires(HasKnownEntityId) { return request(ticket, stk::topology::CONSTRAINT_RANK, entity_id); }
  // clang-format on

  /// \brief Fetch the requested entity ID using the given ticket. Only valid for the known-id variant.
  KOKKOS_INLINE_FUNCTION stk::mesh::EntityId get_entity_id(size_t ticket, stk::mesh::EntityRank entity_rank) const
    requires(HasKnownEntityId)
  {
    constexpr auto name = make_string_literal("NgpRequestEntitiesImplT::get_entity_id");
    assert_active_space<name>();
    assert_valid_rank<name>(entity_rank);
    assert_ticket_out_of_range<name>(ticket, entity_rank);

    KOKKOS_IF_ON_HOST(return state_().requests_[entity_rank].view_host()(ticket).entity_id;)
    KOKKOS_IF_ON_DEVICE(return state_().requests_[entity_rank].view_device()(ticket).entity_id;)
  }

  /// \brief Fetch the requested entity using the given ticket.
  /// After process_requests, this can be called on either host or device. Is NOT valid before process_requests.
  KOKKOS_INLINE_FUNCTION stk::mesh::Entity get_entity(size_t ticket, stk::mesh::EntityRank entity_rank) const {
    constexpr auto name = make_string_literal("NgpRequestEntitiesImplT::get_entity");
    assert_valid_rank<name>(entity_rank);
    assert_ticket_out_of_range<name>(ticket, entity_rank);

    KOKKOS_IF_ON_HOST(return state_().created_entities_[entity_rank].view_host()(ticket);)
    KOKKOS_IF_ON_DEVICE(return state_().created_entities_[entity_rank].view_device()(ticket);)
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    /// Instead of resizing, we'll just zero out the existing requests to avoid reallocations.
    for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
      state_().ticket_issuer_[rank].reset();
      if constexpr (HasKnownEntityId) {
        Kokkos::deep_copy(state_().requests_[rank].view_host(), known_id_request_t{});
        Kokkos::deep_copy(state_().requests_[rank].view_device(), known_id_request_t{});
      }
      Kokkos::deep_copy(state_().created_entities_[rank].view_host(), stk::mesh::Entity{});
      Kokkos::deep_copy(state_().created_entities_[rank].view_device(), stk::mesh::Entity{});
    }
  }

  /// \brief Finalize counts for this request class. Users may no longer claim tickets after this call.
  size_t finalize_count() {
    size_t total_count = 0;
    for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
      size_t count = state_().ticket_issuer_[rank].finalize_count();
      total_count += count;
      if constexpr (HasKnownEntityId) {
        if (state_().requests_[rank].extent(0) < count) {
          Kokkos::resize(state_().requests_[rank], count);
        }
      }
      if (state_().created_entities_[rank].extent(0) < count) {
        Kokkos::resize(state_().created_entities_[rank], count);
      }
    }
    return total_count;
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  template <typename>
  friend class NgpModRequestsT;
  //@}

  //! \name Helper functions
  //@{

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    constexpr bool has_separate_host_and_device_storage =
        !Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible;
    if constexpr (has_separate_host_and_device_storage) {
      KOKKOS_IF_ON_HOST(MUNDY_THROW_ASSERT(
                            /*host_is_active =*/!state_().active_space_host_view_(), std::runtime_error,
                            name + " called from host when device is active.");)
      KOKKOS_IF_ON_DEVICE(MUNDY_THROW_ASSERT(
                              /*device_is_active =*/state_().active_space_dev_view_(), std::runtime_error,
                              name + " called from device when host is active.");)
    }
  }

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_ticket_out_of_range(size_t ticket, stk::mesh::EntityRank entity_rank) const {
    MUNDY_THROW_ASSERT(ticket < state_().ticket_issuer_[entity_rank].count(), std::out_of_range,
                       name + " called with invalid ticket.");
  }

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_valid_rank(stk::mesh::EntityRank rank) const {
    MUNDY_THROW_ASSERT(rank >= 0 && rank < stk::topology::NUM_RANKS, std::out_of_range,
                       name + " called with invalid entity rank.");
  }

  entity_view_t get_created_entity_view(stk::mesh::EntityRank entity_rank) const {
    constexpr auto name = make_string_literal("NgpRequestEntitiesImplT::get_created_entity_view");
    assert_valid_rank<name>(entity_rank);
    return state_().created_entities_[entity_rank];
  }
  //@}

  //! \name Member variables
  //@{

  struct no_requests_t {};

  struct known_id_request_t {
    stk::mesh::EntityId entity_id = stk::mesh::InvalidEntityId;
  };

  using bool_view_t = Kokkos::View<bool, memory_space>;
  using request_view_t = NgpViewT<known_id_request_t*, NgpMemSpace>;
  using requests_storage_t = std::conditional_t<HasKnownEntityId, request_view_t, no_requests_t>;
  struct State {
    unsigned index_{0};
    bool_view_t active_space_dev_view_;
    bool_view_t::HostMirror active_space_host_view_;
    ticket_issuer_t ticket_issuer_[stk::topology::NUM_RANKS];
    requests_storage_t requests_[stk::topology::NUM_RANKS];
    entity_view_t created_entities_[stk::topology::NUM_RANKS];
  };

  using state_view_t = Kokkos::View<State, control_space>;
  mutable state_view_t state_;
  //@}
};  // NgpRequestEntitiesImplT

template <typename NgpMemSpace>
using NgpRequestEntitiesKnownIdsT = NgpRequestEntitiesImplT<NgpMemSpace, true>;

template <typename NgpMemSpace>
using NgpRequestEntitiesNewIdsT = NgpRequestEntitiesImplT<NgpMemSpace, false>;

struct FutureConnection {
  size_t ticket;
  unsigned request_helper_index;
};

template <typename NgpMemSpace>
class NgpRequestConnectionsT {
 public:
  using memory_space = NgpMemSpace;
  using ticket_issuer_t = TicketIssuer<NgpMemSpace>;
  using control_space = Kokkos::SharedSpace;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpRequestConnectionsT() : state_("NgpRequestConnectionsT::state") {
    state_().active_space_dev_view_ = bool_view_t("NgpRequestConnectionsT::active_space_dev_view");
    state_().active_space_host_view_ = Kokkos::create_mirror_view(state_().active_space_dev_view_);
    state_().ticket_issuer_ = ticket_issuer_t(/*activate_device*/ true);
    state_().requests_ = request_view_t("NgpRequestConnectionsT::requests", 0);
    // Initialize to device active
    state_().active_space_host_view_() = true;
    Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
  }

  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionsT(const NgpRequestConnectionsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionsT& operator=(const NgpRequestConnectionsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionsT(NgpRequestConnectionsT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionsT& operator=(NgpRequestConnectionsT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpRequestConnectionsT() = default;
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from device
  void activate_host() {
    auto device_is_active = state_().active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      state_().active_space_host_view_() = false;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      state_().ticket_issuer_.activate_host();
    }
  }

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from host
  void activate_device() {
    auto host_is_active = !state_().active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      state_().active_space_host_view_() = true;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      state_().ticket_issuer_.activate_device();
    }
  }

  /// \brief Synchronize between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = state_().active_space_host_view_();
    if (device_is_active) {
      state_().ticket_issuer_.sync();
      Kokkos::deep_copy(state_().requests_.view_host(), state_().requests_.view_device());
    } else {
      state_().ticket_issuer_.sync();
      Kokkos::deep_copy(state_().requests_.view_device(), state_().requests_.view_host());
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Get our unique index among multiple request helpers.
  unsigned id() const noexcept {
    return state_().index_;
  }

  /// \brief Get the ticket issuer for entity requests.
  ///
  /// The lifetime of the tickets for this class should not exceed the class itself
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& tickets() const noexcept {
    return state_().ticket_issuer_;
  }

  /// \brief Record a connection between two entities with the given ordinal.
  ///
  /// The mapping ( from_entity , ordinal ) -> to_entity  must be unique.
  ///
  /// Both entities may be either real entities or future entities.
  ///
  /// \note [IMPORTANT] Relation-declarations must be symmetric across all sharers of the involved entities
  ///   within a modification cycle. For now, all future entities are owned by the current process.
  KOKKOS_INLINE_FUNCTION FutureConnection request(size_t ticket, variant<stk::mesh::Entity, FutureEntity> from_entity,
                                                  variant<stk::mesh::Entity, FutureEntity> to_entity,
                                                  const stk::mesh::RelationIdentifier ordinal) const {
    constexpr auto name = make_string_literal("NgpRequestConnectionsT::request");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(auto& req = state_().requests_.view_host()(ticket); req.from_entity = from_entity;
                      req.to_entity = to_entity; req.ordinal = ordinal;)
    KOKKOS_IF_ON_DEVICE(auto& req = state_().requests_.view_device()(ticket); req.from_entity = from_entity;
                        req.to_entity = to_entity; req.ordinal = ordinal;)

    return FutureConnection{.ticket = ticket, .request_helper_index = state_().index_};
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    /// Instead of resizing, we'll just zero out the existing requests to avoid reallocations.
    state_().ticket_issuer_.reset();
    Kokkos::deep_copy(state_().requests_.view_host(), ConnectionRequest{});
    Kokkos::deep_copy(state_().requests_.view_device(), ConnectionRequest{});
  }

  /// \brief Finalize counts for this request class. Users may no longer claim tickets after this call.
  size_t finalize_count() {
    size_t count = state_().ticket_issuer_.finalize_count();
    if (state_().requests_.extent(0) < count) {
      Kokkos::resize(state_().requests_, count);
    }
    return count;
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  template <typename>
  friend class NgpModRequestsT;
  //@}

  //! \name Helper functions
  //@{

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    constexpr bool has_separate_host_and_device_storage =
        !Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible;
    if constexpr (has_separate_host_and_device_storage) {
      KOKKOS_IF_ON_HOST(MUNDY_THROW_ASSERT(
                            /*host_is_active =*/!state_().active_space_host_view_(), std::runtime_error,
                            name + " called from host when device is active.");)
      KOKKOS_IF_ON_DEVICE(MUNDY_THROW_ASSERT(
                              /*device_is_active =*/state_().active_space_dev_view_(), std::runtime_error,
                              name + " called from device when host is active.");)
    }
  }

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_ticket_out_of_range(size_t ticket) const {
    MUNDY_THROW_ASSERT(ticket < state_().ticket_issuer_.count(), std::out_of_range,
                       name + " called with invalid ticket.");
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief Internal struct representing a single connection request.
  /// Requests must specify a pair of entities to connect. These may either be a real entity or a future entity (by
  /// ticket and request helper).
  struct ConnectionRequest {
    variant<stk::mesh::Entity, FutureEntity> from_entity;
    variant<stk::mesh::Entity, FutureEntity> to_entity;
    stk::mesh::RelationIdentifier ordinal;
  };

  KOKKOS_INLINE_FUNCTION ConnectionRequest get_request(size_t ticket) const {
    constexpr auto name = make_string_literal("NgpRequestConnectionsT::get_request");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(return state_().requests_.view_host()(ticket);)
    KOKKOS_IF_ON_DEVICE(return state_().requests_.view_device()(ticket);)
  }

  using request_view_t = NgpViewT<ConnectionRequest*, NgpMemSpace>;
  using bool_view_t = Kokkos::View<bool, memory_space>;
  struct State {
    unsigned index_{0};
    bool_view_t active_space_dev_view_;
    bool_view_t::HostMirror active_space_host_view_;
    ticket_issuer_t ticket_issuer_;
    request_view_t requests_;
  };

  using state_view_t = Kokkos::View<State, control_space>;
  mutable state_view_t state_;
  //@}
};  // NgpRequestConnectionsT

struct FutureDestroyEntity {
  size_t ticket;
  unsigned request_helper_index;
};

template <typename NgpMemSpace>
class NgpDestroyEntitiesT {
 public:
  using memory_space = NgpMemSpace;
  using ticket_issuer_t = TicketIssuer<NgpMemSpace>;
  using control_space = Kokkos::SharedSpace;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpDestroyEntitiesT() : state_("NgpDestroyEntitiesT::state") {
    state_().active_space_dev_view_ = bool_view_t("NgpDestroyEntitiesT::active_space_dev_view");
    state_().active_space_host_view_ = Kokkos::create_mirror_view(state_().active_space_dev_view_);
    state_().ticket_issuer_ = ticket_issuer_t(/*activate_device*/ true);
    state_().requests_ = request_view_t("NgpDestroyEntitiesT::requests", 0);
    state_().created_entities_ = entity_view_t("NgpDestroyEntitiesT::created_entities", 0);
    // Initialize to device active
    state_().active_space_host_view_() = true;
    Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
  }

  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntitiesT(const NgpDestroyEntitiesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntitiesT& operator=(const NgpDestroyEntitiesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntitiesT(NgpDestroyEntitiesT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntitiesT& operator=(NgpDestroyEntitiesT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpDestroyEntitiesT() = default;
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from device
  void activate_host() {
    auto device_is_active = state_().active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      state_().active_space_host_view_() = false;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      state_().ticket_issuer_.activate_host();
    }
  }

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from host
  void activate_device() {
    auto host_is_active = !state_().active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      state_().active_space_host_view_() = true;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      state_().ticket_issuer_.activate_device();
    }
  }

  /// \brief Synchronize between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = state_().active_space_host_view_();
    if (device_is_active) {
      state_().ticket_issuer_.sync();
      Kokkos::deep_copy(state_().requests_.view_host(), state_().requests_.view_device());
    } else {
      state_().ticket_issuer_.sync();
      Kokkos::deep_copy(state_().requests_.view_device(), state_().requests_.view_host());
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Get our unique index among multiple request helpers.
  unsigned id() const noexcept {
    return state_().index_;
  }

  /// \brief Get the ticket issuer for entity requests.
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& tickets() const noexcept {
    return state_().ticket_issuer_;
  }

  /// \brief Record an entity destruction request
  KOKKOS_INLINE_FUNCTION FutureDestroyEntity destroy(size_t ticket, stk::mesh::Entity entity) const {
    constexpr auto name = make_string_literal("NgpDestroyEntitiesT::destroy");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(state_().requests_.view_host()(ticket) = entity;)
    KOKKOS_IF_ON_DEVICE(state_().requests_.view_device()(ticket) = entity;)

    return FutureDestroyEntity{.ticket = ticket, .request_helper_index = state_().index_};
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    /// Instead of resizing, we'll just zero out the existing requests to avoid reallocations.
    state_().ticket_issuer_.reset();
    Kokkos::deep_copy(state_().requests_.view_host(), stk::mesh::Entity{});
    Kokkos::deep_copy(state_().requests_.view_device(), stk::mesh::Entity{});
  }

  /// \brief Finalize counts for this request class. Users may no longer claim tickets after this call.
  size_t finalize_count() {
    size_t count = state_().ticket_issuer_.finalize_count();
    if (state_().requests_.extent(0) < count) {
      Kokkos::resize(state_().requests_, count);
    }
    return count;
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  template <typename>
  friend class NgpModRequestsT;
  //@}

  //! \name Helper functions
  //@{

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    constexpr bool has_separate_host_and_device_storage =
        !Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible;
    if constexpr (has_separate_host_and_device_storage) {
      KOKKOS_IF_ON_HOST(MUNDY_THROW_ASSERT(
                            /*host_is_active =*/!state_().active_space_host_view_(), std::runtime_error,
                            name + " called from host when device is active.");)
      KOKKOS_IF_ON_DEVICE(MUNDY_THROW_ASSERT(
                              /*device_is_active =*/state_().active_space_dev_view_(), std::runtime_error,
                              name + " called from device when host is active.");)
    }
  }

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_ticket_out_of_range(size_t ticket) const {
    MUNDY_THROW_ASSERT(ticket < state_().ticket_issuer_.count(), std::out_of_range,
                       name + " called with invalid ticket.");
  }

  KOKKOS_INLINE_FUNCTION stk::mesh::Entity get_entity_to_destroy(size_t ticket) const {
    constexpr auto name = make_string_literal("NgpDestroyEntitiesT::get_entity_to_destroy");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(return state_().requests_.view_host()(ticket);)
    KOKKOS_IF_ON_DEVICE(return state_().requests_.view_device()(ticket);)
  }
  //@}

  //! \name Member variables
  //@{

  using request_view_t = NgpViewT<stk::mesh::Entity*, NgpMemSpace>;
  using entity_view_t = NgpViewT<stk::mesh::Entity*, NgpMemSpace>;
  using bool_view_t = Kokkos::View<bool, memory_space>;
  struct State {
    unsigned index_{0};
    bool_view_t active_space_dev_view_;
    bool_view_t::HostMirror active_space_host_view_;
    ticket_issuer_t ticket_issuer_;
    request_view_t requests_;
    entity_view_t created_entities_;
  };

  using state_view_t = Kokkos::View<State, control_space>;
  mutable state_view_t state_;
  //@}
};  // NgpDestroyEntitiesT

struct FutureDestroyConnection {
  size_t ticket;
  unsigned request_helper_index;
};

template <typename NgpMemSpace>
class NgpDestroyConnectionsT {
 public:
  using memory_space = NgpMemSpace;
  using ticket_issuer_t = TicketIssuer<NgpMemSpace>;
  using control_space = Kokkos::SharedSpace;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpDestroyConnectionsT() : state_("NgpDestroyConnectionsT::state") {
    state_().active_space_dev_view_ = bool_view_t("NgpDestroyConnectionsT::active_space_dev_view");
    state_().active_space_host_view_ = Kokkos::create_mirror_view(state_().active_space_dev_view_);
    state_().ticket_issuer_ = ticket_issuer_t(/*activate_device*/ true);
    state_().requests_ = request_view_t("NgpDestroyConnectionsT::requests", 0);
    // Initialize to device active
    state_().active_space_host_view_() = true;
    Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
  }

  KOKKOS_DEFAULTED_FUNCTION NgpDestroyConnectionsT(const NgpDestroyConnectionsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyConnectionsT& operator=(const NgpDestroyConnectionsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyConnectionsT(NgpDestroyConnectionsT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyConnectionsT& operator=(NgpDestroyConnectionsT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpDestroyConnectionsT() = default;
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from device
  void activate_host() {
    auto device_is_active = state_().active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      state_().active_space_host_view_() = false;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      state_().ticket_issuer_.activate_host();
    }
  }

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from host
  void activate_device() {
    auto host_is_active = !state_().active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      state_().active_space_host_view_() = true;
      Kokkos::deep_copy(state_().active_space_dev_view_, state_().active_space_host_view_);
      state_().ticket_issuer_.activate_device();
    }
  }

  /// \brief Synchronize between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = state_().active_space_host_view_();
    if (device_is_active) {
      state_().ticket_issuer_.sync();
      Kokkos::deep_copy(state_().requests_.view_host(), state_().requests_.view_device());
    } else {
      state_().ticket_issuer_.sync();
      Kokkos::deep_copy(state_().requests_.view_device(), state_().requests_.view_host());
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Get our unique index among multiple request helpers.
  unsigned id() const noexcept {
    return state_().index_;
  }

  /// \brief Get the ticket issuer for entity requests.
  KOKKOS_INLINE_FUNCTION ticket_issuer_t& tickets() const noexcept {
    return state_().ticket_issuer_;
  }

  /// \brief Record a connection destruction between two entities (from_entity -> to_entity).
  /// Both entities must be real entities.
  KOKKOS_INLINE_FUNCTION FutureDestroyConnection request(size_t ticket, stk::mesh::Entity from_entity,
                                                         stk::mesh::Entity to_entity,
                                                         const stk::mesh::RelationIdentifier ordinal) const {
    constexpr auto name = make_string_literal("NgpDestroyConnectionsT::request");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(auto& req = state_().requests_.view_host()(ticket); req.from_entity = from_entity;
                      req.to_entity = to_entity; req.ordinal = ordinal;)
    KOKKOS_IF_ON_DEVICE(auto& req = state_().requests_.view_device()(ticket); req.from_entity = from_entity;
                        req.to_entity = to_entity; req.ordinal = ordinal;)

    return FutureDestroyConnection{.ticket = ticket, .request_helper_index = state_().index_};
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    /// Instead of resizing, we'll just zero out the existing requests to avoid reallocations.
    state_().ticket_issuer_.reset();
    Kokkos::deep_copy(state_().requests_.view_host(), DestroyConnectionRequest{});
    Kokkos::deep_copy(state_().requests_.view_device(), DestroyConnectionRequest{});
  }

  /// \brief Finalize counts for this request class. Users may no longer claim tickets after this call.
  size_t finalize_count() {
    size_t count = state_().ticket_issuer_.finalize_count();
    if (state_().requests_.extent(0) < count) {
      Kokkos::resize(state_().requests_, count);
    }
    return count;
  }
  //@}

 private:
  //! \name Friends <3
  //@{

  template <typename>
  friend class NgpModRequestsT;
  //@}

  //! \name Helper functions
  //@{

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    constexpr bool has_separate_host_and_device_storage =
        !Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible;
    if constexpr (has_separate_host_and_device_storage) {
      KOKKOS_IF_ON_HOST(MUNDY_THROW_ASSERT(
                            /*host_is_active =*/!state_().active_space_host_view_(), std::runtime_error,
                            name + " called from host when device is active.");)
      KOKKOS_IF_ON_DEVICE(MUNDY_THROW_ASSERT(
                              /*device_is_active =*/state_().active_space_dev_view_(), std::runtime_error,
                              name + " called from device when host is active.");)
    }
  }

  template <StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_ticket_out_of_range(size_t ticket) const {
    MUNDY_THROW_ASSERT(ticket < state_().ticket_issuer_.count(), std::out_of_range,
                       name + " called with invalid ticket.");
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief Internal struct representing a single connection destruction request.
  /// Requests must specify a pair of entities to destroy the connection between. Both entities must be real entities.
  struct DestroyConnectionRequest {
    stk::mesh::Entity from_entity;
    stk::mesh::Entity to_entity;
    stk::mesh::RelationIdentifier ordinal;
  };

  KOKKOS_INLINE_FUNCTION DestroyConnectionRequest get_request(size_t ticket) const {
    constexpr auto name = make_string_literal("NgpDestroyConnectionsT::get_request");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(return state_().requests_.view_host()(ticket);)
    KOKKOS_IF_ON_DEVICE(return state_().requests_.view_device()(ticket);)
  }

  using request_view_t = NgpViewT<DestroyConnectionRequest*, NgpMemSpace>;
  using bool_view_t = Kokkos::View<bool, memory_space>;
  struct State {
    unsigned index_{0};
    bool_view_t active_space_dev_view_;
    bool_view_t::HostMirror active_space_host_view_;
    ticket_issuer_t ticket_issuer_;
    request_view_t requests_;
  };

  using state_view_t = Kokkos::View<State, control_space>;
  mutable state_view_t state_;
  //@}
};  // NgpDestroyConnectionsT

/// \brief Manages all modification requests in a given NGP memory space.
template <typename NgpMemSpace>
class NgpModRequestsT {
 public:
  using memory_space = NgpMemSpace;
  using ticket_id = size_t;
  using control_space = Kokkos::SharedSpace;
  using host_state_space = Kokkos::HostSpace;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpModRequestsT() : shared_state_("NgpModRequestsT::shared_state"), host_state_("NgpModRequestsT::host_state") {
  }

  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT(const NgpModRequestsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT& operator=(const NgpModRequestsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT(NgpModRequestsT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT& operator=(NgpModRequestsT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpModRequestsT() = default;
  //@}

  //! \name Accessors for request classes
  //@{

  /// \brief Get the entity request class for the given partition key (for requests of entities with new Ids).
  NgpRequestEntitiesNewIdsT<NgpMemSpace> request_entities_new_ids(const stk::mesh::PartVector& parts) const {
    return get_or_create_entity_requests_new_ids(parts);
  }
  //
  NgpRequestEntitiesNewIdsT<NgpMemSpace> request_entities_new_ids(const stk::mesh::Part& part) const {
    stk::mesh::PartVector part_vec(1);
    part_vec[0] = const_cast<stk::mesh::Part*>(&part);
    return get_or_create_entity_requests_new_ids(part_vec);
  }

  /// \brief Get the entity request class for the given partition key (for requests of entities with known Ids).
  /// Aka you assign the Id at request time instead of it being automatically generated by STK.
  NgpRequestEntitiesKnownIdsT<NgpMemSpace> request_entities_known_ids(const stk::mesh::PartVector& parts) const {
    return get_or_create_entity_requests_known_ids(parts);
  }
  //
  NgpRequestEntitiesKnownIdsT<NgpMemSpace> request_entities_known_ids(const stk::mesh::Part& part) const {
    stk::mesh::PartVector part_vec(1);
    part_vec[0] = const_cast<stk::mesh::Part*>(&part);
    return get_or_create_entity_requests_known_ids(part_vec);
  }

  /// \brief Get the destroy entity request class.
  NgpDestroyEntitiesT<NgpMemSpace> destroy_entities() const {
    return shared_state_().destroy_entity_requests_;
  }

  /// \brief Get the connection request class.
  NgpRequestConnectionsT<NgpMemSpace> request_connections() const {
    return shared_state_().connection_requests_;
  }

  /// \brief Get the destroy connection request class.
  NgpDestroyConnectionsT<NgpMemSpace> destroy_connections() const {
    return shared_state_().destroy_connection_requests_;
  }
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host for all request helpers and ticket issuers.
  void activate_host() {
    for (auto& entry : get_entity_requests_new_ids_entries()) {
      entry.requests.activate_host();
    }
    for (auto& entry : get_entity_requests_known_ids_entries()) {
      entry.requests.activate_host();
    }
    shared_state_().destroy_entity_requests_.activate_host();
    shared_state_().connection_requests_.activate_host();
    shared_state_().destroy_connection_requests_.activate_host();
  }

  /// \brief Sets the active memory space to device for all request helpers and ticket issuers.
  void activate_device() {
    for (auto& entry : get_entity_requests_new_ids_entries()) {
      entry.requests.activate_device();
    }
    for (auto& entry : get_entity_requests_known_ids_entries()) {
      entry.requests.activate_device();
    }
    shared_state_().destroy_entity_requests_.activate_device();
    shared_state_().connection_requests_.activate_device();
    shared_state_().destroy_connection_requests_.activate_device();
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    for (auto& entry : get_entity_requests_new_ids_entries()) {
      entry.requests.reset();
    }
    for (auto& entry : get_entity_requests_known_ids_entries()) {
      entry.requests.reset();
    }
    shared_state_().destroy_entity_requests_.reset();
    shared_state_().connection_requests_.reset();
    shared_state_().destroy_connection_requests_.reset();
  }

  /// \brief Finalize counts for all request classes. Users may no longer claim tickets after this call.
  void finalize_counts() {
    for (auto& entry : get_entity_requests_new_ids_entries()) {
      entry.requests.finalize_count();
    }
    for (auto& entry : get_entity_requests_known_ids_entries()) {
      entry.requests.finalize_count();
    }
    shared_state_().destroy_entity_requests_.finalize_count();
    shared_state_().connection_requests_.finalize_count();
    shared_state_().destroy_connection_requests_.finalize_count();
  }

  /// \brief Process all requests in all request classes.
  /// The order goes:
  ///  1) Entity requests
  ///  2) Connection requests
  ///  3) Destroy connection requests
  ///  4) Destroy entity requests
  ///
  /// Importantly, this allows you to delete all of the connections stemming from an entity before deleting the entity
  /// itself, all within a single modification cycle. I can't forsee a use case for deleting the entities before the
  /// connections.
  ///
  /// If this is a desired feature, let me know.
  void process_requests(stk::mesh::BulkData& bulk_data) {
    // Determine if a modification cycle is even necessary, aka no requests were made.
    size_t total_requests = 0;
    for (auto& entry : get_entity_requests_new_ids_entries()) {
      for (stk::mesh::EntityRank rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        total_requests += entry.requests.tickets(rank).count();
      }
    }
    for (auto& entry : get_entity_requests_known_ids_entries()) {
      for (stk::mesh::EntityRank rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        total_requests += entry.requests.tickets(rank).count();
      }
    }
    total_requests += shared_state_().destroy_entity_requests_.tickets().count();
    total_requests += shared_state_().connection_requests_.tickets().count();
    total_requests += shared_state_().destroy_connection_requests_.tickets().count();
    if (total_requests == 0) {
      return;
    }

    // Crack open a modification cycle if we aren't already in one.
    bool we_started_mod_cycle = false;
    if (!bulk_data.in_modifiable_state()) {
      bulk_data.modification_begin();
      we_started_mod_cycle = true;
    }

    //////////////////////////////////
    // Entity requests with new ids //
    //////////////////////////////////
    for (auto& entry : get_entity_requests_new_ids_entries()) {
      stk::mesh::PartVector parts = impl::get_parts_for_partition_key(entry.partition_key, bulk_data.mesh_meta_data());
      size_t num_actual_ranks = bulk_data.mesh_meta_data().entity_rank_count();
      std::vector<size_t> num_requests_per_rank(num_actual_ranks, 0);
      for (size_t rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
        if (rank < num_actual_ranks) {
          num_requests_per_rank[rank] = entry.requests.tickets(static_cast<stk::mesh::EntityRank>(rank)).count();
        } else {
          MUNDY_THROW_ASSERT(entry.requests.tickets(static_cast<stk::mesh::EntityRank>(rank)).count() == 0,
                             std::logic_error, "Received requests for invalid entity rank " + std::to_string(rank));
        }
      }

      std::vector<stk::mesh::EntityVector> requested_entities_per_rank(num_actual_ranks);
      our_generate_new_entities(bulk_data, num_requests_per_rank, parts, requested_entities_per_rank);

      for (size_t rank = stk::topology::BEGIN_RANK; rank < num_actual_ranks; ++rank) {
        auto created_entities_view = entry.requests.get_created_entity_view(static_cast<stk::mesh::EntityRank>(rank));
        size_t count = num_requests_per_rank[rank];
        if (count == 0) {
          continue;
        }

        Kokkos::parallel_for(
            "NgpModRequestsT::process_entity_requests::copy_to_view",
            Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, count),
            KOKKOS_LAMBDA(size_t i) { created_entities_view.view_host()(i) = requested_entities_per_rank[rank][i]; });
      }
    }

    ////////////////////////////////////
    // Entity requests with known ids //
    ////////////////////////////////////
    {
      for (auto& entry : get_entity_requests_known_ids_entries()) {
        stk::mesh::PartVector parts =
            impl::get_parts_for_partition_key(entry.partition_key, bulk_data.mesh_meta_data());
        for (size_t rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
          size_t count = entry.requests.tickets(static_cast<stk::mesh::EntityRank>(rank)).count();
          if (count == 0) {
            continue;
          } else {
            MUNDY_THROW_ASSERT(rank < static_cast<size_t>(bulk_data.mesh_meta_data().entity_rank_count()),
                               std::logic_error, "Received requests for invalid entity rank " + std::to_string(rank));
          }

          // This simply has to be done in serial since declare_entity is not thread safe.
          auto created_entities_view = entry.requests.get_created_entity_view(static_cast<stk::mesh::EntityRank>(rank));
          for (size_t ticket = 0; ticket < count; ++ticket) {
            stk::mesh::EntityId entity_id =
                entry.requests.get_entity_id(ticket, static_cast<stk::mesh::EntityRank>(rank));
            stk::mesh::Entity entity =
                bulk_data.declare_entity(static_cast<stk::mesh::EntityRank>(rank), entity_id, parts);
            created_entities_view.view_host()(ticket) = entity;
          }
        }
      }
    }

    /////////////////////////
    // Connection requests //
    /////////////////////////
    {
      stk::mesh::Permutation perm = stk::mesh::Permutation::INVALID_PERMUTATION;
      stk::mesh::OrdinalVector scratch1, scratch2, scratch3;

      auto resolve_entity = [&](const variant<stk::mesh::Entity, FutureEntity>& maybe_future) {
        if (holds_alternative<stk::mesh::Entity>(maybe_future)) {
          return get<stk::mesh::Entity>(maybe_future);
        }

        const FutureEntity future = get<FutureEntity>(maybe_future);
        if (future.has_known_id) {
          return get_entity_requests_known_ids_by_index(future.request_helper_index)
              .get_entity(future.ticket, future.entity_rank);
        }
        return get_entity_requests_new_ids_by_index(future.request_helper_index)
            .get_entity(future.ticket, future.entity_rank);
      };

      // Must be done in serial since declare_relation is not thread safe.
      size_t num_connection_requests = shared_state_().connection_requests_.tickets().count();
      for (size_t ticket = 0; ticket < num_connection_requests; ++ticket) {
        auto req = shared_state_().connection_requests_.get_request(ticket);
        stk::mesh::Entity from_entity = resolve_entity(req.from_entity);
        stk::mesh::Entity to_entity = resolve_entity(req.to_entity);
        bulk_data.declare_relation(from_entity, to_entity, req.ordinal, perm, scratch1, scratch2, scratch3);
      }
    }

    /////////////////////////////////
    // Destroy connection requests //
    /////////////////////////////////
    {
      // Must be done in serial since destroy_relation is not thread safe.
      size_t num_destroy_connection_requests = shared_state_().destroy_connection_requests_.tickets().count();
      for (size_t ticket = 0; ticket < num_destroy_connection_requests; ++ticket) {
        auto req = shared_state_().destroy_connection_requests_.get_request(ticket);
        bulk_data.destroy_relation(req.from_entity, req.to_entity, req.ordinal);
      }
    }

    /////////////////////////////
    // Destroy entity requests //
    /////////////////////////////
    {
      // Must be done in serial since destroy_entity is not thread safe.
      size_t num_destroy_entity_requests = shared_state_().destroy_entity_requests_.tickets().count();
      for (size_t ticket = 0; ticket < num_destroy_entity_requests; ++ticket) {
        stk::mesh::Entity entity = shared_state_().destroy_entity_requests_.get_entity_to_destroy(ticket);
        bulk_data.destroy_entity(entity);
      }
    }

    //////////////
    // Finalize //
    //////////////
    {
      // At this point, requests within each helper are up-to-date on the host. Synchronize the data to the device if
      // the device is active
      for (auto& entry : get_entity_requests_new_ids_entries()) {
        const bool device_is_active = entry.requests.state_().active_space_host_view_();
        if (!device_is_active) {
          continue;
        }

        for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
          Kokkos::deep_copy(entry.requests.state_().created_entities_[rank].view_device(),
                            entry.requests.state_().created_entities_[rank].view_host());
        }
      }

      for (auto& entry : get_entity_requests_known_ids_entries()) {
        const bool device_is_active = entry.requests.state_().active_space_host_view_();
        if (!device_is_active) {
          continue;
        }

        for (size_t rank = 0; rank < stk::topology::NUM_RANKS; ++rank) {
          Kokkos::deep_copy(entry.requests.state_().requests_[rank].view_device(),
                            entry.requests.state_().requests_[rank].view_host());
          Kokkos::deep_copy(entry.requests.state_().created_entities_[rank].view_device(),
                            entry.requests.state_().created_entities_[rank].view_host());
        }
      }

      if (we_started_mod_cycle) {
        bulk_data.modification_end();
      }
    }
  }
  //@}

 private:
  struct entity_requests_new_ids_entry_t {
    impl::PartitionKey partition_key;
    NgpRequestEntitiesNewIdsT<NgpMemSpace> requests;
  };

  struct entity_requests_known_ids_entry_t {
    impl::PartitionKey partition_key;
    NgpRequestEntitiesKnownIdsT<NgpMemSpace> requests;
  };

  using entity_requests_new_ids_entries_t = std::deque<entity_requests_new_ids_entry_t>;
  using entity_requests_known_ids_entries_t = std::deque<entity_requests_known_ids_entry_t>;
  using entity_requests_new_ids_index_map_t = std::map<impl::PartitionKey, unsigned>;
  using entity_requests_known_ids_index_map_t = std::map<impl::PartitionKey, unsigned>;

  struct SharedState {
    NgpDestroyEntitiesT<NgpMemSpace> destroy_entity_requests_;
    NgpRequestConnectionsT<NgpMemSpace> connection_requests_;
    NgpDestroyConnectionsT<NgpMemSpace> destroy_connection_requests_;
  };

  struct HostState {
    entity_requests_new_ids_entries_t entity_requests_new_ids_entries_;
    entity_requests_known_ids_entries_t entity_requests_known_ids_entries_;
    entity_requests_new_ids_index_map_t entity_requests_new_ids_index_map_;
    entity_requests_known_ids_index_map_t entity_requests_known_ids_index_map_;
  };

  entity_requests_new_ids_entries_t& get_entity_requests_new_ids_entries() const {
    return host_state_().entity_requests_new_ids_entries_;
  }

  entity_requests_known_ids_entries_t& get_entity_requests_known_ids_entries() const {
    return host_state_().entity_requests_known_ids_entries_;
  }

  entity_requests_new_ids_index_map_t& get_entity_requests_new_ids_index_map() const {
    return host_state_().entity_requests_new_ids_index_map_;
  }

  entity_requests_known_ids_index_map_t& get_entity_requests_known_ids_index_map() const {
    return host_state_().entity_requests_known_ids_index_map_;
  }

  NgpRequestEntitiesNewIdsT<NgpMemSpace> get_entity_requests_new_ids_by_index(unsigned index) const {
    auto& entries = get_entity_requests_new_ids_entries();
    MUNDY_THROW_ASSERT(index < entries.size(), std::out_of_range, "new-id entity request helper index out of range.");
    return entries[index].requests;
  }

  NgpRequestEntitiesKnownIdsT<NgpMemSpace> get_entity_requests_known_ids_by_index(unsigned index) const {
    auto& entries = get_entity_requests_known_ids_entries();
    MUNDY_THROW_ASSERT(index < entries.size(), std::out_of_range, "known-id entity request helper index out of range.");
    return entries[index].requests;
  }

  NgpRequestEntitiesNewIdsT<NgpMemSpace> get_or_create_entity_requests_new_ids(
      const stk::mesh::PartVector& parts) const {
    impl::PartitionKey key = impl::get_partition_key(parts);

    // Map the key to a linearized index of existing request helpers.
    auto& index_map = get_entity_requests_new_ids_index_map();
    auto map_it = index_map.find(key);

    // Return the existing request helper if the key already exists
    if (map_it != index_map.end()) {
      return get_entity_requests_new_ids_by_index(map_it->second);
    }

    // Key doesn't exist, create a new request helper, add it to the entries vector and map, then return it
    auto& entries = get_entity_requests_new_ids_entries();
    const unsigned helper_index = static_cast<unsigned>(entries.size());
    entries.push_back(entity_requests_new_ids_entry_t{key, NgpRequestEntitiesNewIdsT<NgpMemSpace>(helper_index)});
    index_map.emplace(entries.back().partition_key, helper_index);
    return entries.back().requests;
  }

  NgpRequestEntitiesKnownIdsT<NgpMemSpace> get_or_create_entity_requests_known_ids(
      const stk::mesh::PartVector& parts) const {
    impl::PartitionKey key = impl::get_partition_key(parts);

    // Map the key to a linearized index of existing request helpers.
    auto& index_map = get_entity_requests_known_ids_index_map();
    auto map_it = index_map.find(key);

    // Return the existing request helper if the key already exists
    if (map_it != index_map.end()) {
      return get_entity_requests_known_ids_by_index(map_it->second);
    }

    // Key doesn't exist, create a new request helper, add it to the entries vector and map, then return it
    auto& entries = get_entity_requests_known_ids_entries();
    const unsigned helper_index = static_cast<unsigned>(entries.size());
    entries.push_back(entity_requests_known_ids_entry_t{key, NgpRequestEntitiesKnownIdsT<NgpMemSpace>(helper_index)});
    index_map.emplace(entries.back().partition_key, helper_index);
    return entries.back().requests;
  }

  /// Unpacked version of generate_new_entities to avoid a costly change of part post creation.
  void our_generate_new_entities(stk::mesh::BulkData& bulk_data, const std::vector<size_t>& num_requests_per_rank,
                                 stk::mesh::PartVector& add_parts,
                                 std::vector<stk::mesh::EntityVector>& requested_entities_per_rank) const {
    size_t num_ranks = num_requests_per_rank.size();
    std::vector<std::vector<stk::mesh::EntityId>> requested_ids(num_ranks);
    for (size_t i = 0; i < num_ranks; ++i) {
      stk::topology::rank_t rank = static_cast<stk::topology::rank_t>(i);
      bulk_data.generate_new_ids(rank, num_requests_per_rank[i], requested_ids[i]);
    }

    // generating 'owned' entities in the given parts
    stk::mesh::PartVector add_parts_plus_owned = add_parts;
    add_parts_plus_owned.push_back(&bulk_data.mesh_meta_data().locally_owned_part());
    requested_entities_per_rank.clear();
    requested_entities_per_rank.resize(num_ranks);
    for (size_t i = 0; i < num_ranks; ++i) {
      stk::topology::rank_t rank = static_cast<stk::topology::rank_t>(i);
      bulk_data.declare_entities(rank, requested_ids[i], add_parts_plus_owned, requested_entities_per_rank[i]);
    }
  }

  // Entity requests are mapped by the partition key that identifies their collection of parts.
  // We'll always append the locally owned part to the given part list to ensure ownership.
  using shared_state_view_t = Kokkos::View<SharedState, control_space>;
  using host_state_view_t = Kokkos::View<HostState, host_state_space>;

  mutable shared_state_view_t shared_state_;
  mutable host_state_view_t host_state_;
};

// Following STK's default naming convention for ngp classes in the default memory space
using NgpModRequests = NgpModRequestsT<stk::ngp::MemSpace>;
using NgpRequestEntitiesKnownIds = NgpRequestEntitiesKnownIdsT<stk::ngp::MemSpace>;
using NgpRequestEntitiesNewIds = NgpRequestEntitiesNewIdsT<stk::ngp::MemSpace>;
using NgpRequestConnections = NgpRequestConnectionsT<stk::ngp::MemSpace>;
using NgpDestroyEntities = NgpDestroyEntitiesT<stk::ngp::MemSpace>;
using NgpDestroyConnections = NgpDestroyConnectionsT<stk::ngp::MemSpace>;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPMODREQUESTS_HPP_
