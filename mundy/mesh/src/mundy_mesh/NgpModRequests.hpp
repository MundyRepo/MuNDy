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
#include <stdexcept>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral, mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>
#include <mundy_mesh/impl/PartitionKey.hpp>  // for mundy::mesh::impl::PartitionKey, mundy::mesh::impl::get_partition_key

namespace mundy {

namespace mesh {

/*
## Design

TicketRange<SizeT>:
  using ticket_id = SizeT;
  KOKKOS_FUNCTION begin() -> SizeT
  KOKKOS_FUNCTION end() -> SizeT
  KOKKOS_FUNCTION count() -> SizeT
 private:
  begin_
  count_

TicketIssuer<SizeT>:
  using ticket_id = SizeT;

  // Control plane (HOST only)
  // Exactly one memory space is allowed to claim tickets/fetch counts at a time.
  // Activation is a phase boundary. It fences and synchronizes ticket state
  activate_host()   -> void   // switches active writer to host; syncs if needed
  activate_device() -> void   // switches active writer to device; syncs if needed

  // Data plane (HOST or DEVICE)
  KOKKOS_FUNCTION claim(n: SizeT) -> TicketRange<SizeT>  // (atomic)
  KOKKOS_FUNCTION claim() -> SizeT                       // (atomic)
  KOKKOS_FUNCTION count() -> uint32_t                    // (atomic)

 private:
  active_space_host_view_
  active_space_dev_view_
  ticket_counter_host_view_
  ticket_counter_dev_view_


NgpModRequests<MemSpace>:
  /// clears all helpers + internal allocations; ready for a fresh cycle
  reset() -> void

  /// Sync counts to the host and allocate memory. Users may no longer claim tickets.
  finalize_counts() -> void

  /// The user has finished making their requests and wants us to process the requests
  /// If you are not in a mod cycle, this will open one
  process_requests() -> void


  /// opts.ghost_nonowned = false by default
  request_entities(parts: span<PartOrd>, opts = {}) -> NgpRequestEntities<MemSpace>&
  destroy_entities(opts = {}) -> NgpDestroyEntities<MemSpace>&
  request_connections(opts = {}) -> NgpRequestConnections<MemSpace>&
  destroy_connections(opts = {}) -> NgpDestroyConnections<MemSpace>&

NgpRequestEntities<MemSpace>:
  KOKKOS_FUNCTION tickets() -> TicketIssuer<size_t>
  KOKKOS_FUNCTION request(ticket, OwningProc) -> NgpRequestEntities<MemSpace>&
  KOKKOS_FUNCTION request(ticket, OwningProc, EntityId) -> NgpRequestEntities<MemSpace>&
  reset() -> void


NgpDestroyEntities<MemSpace>:
  KOKKOS_FUNCTION tickets() -> TicketIssuer<size_t>
  KOKKOS_FUNCTION request(ticket, Entity) -> NgpDestroyEntities<MemSpace>&

NgpRequestConnections<MemSpace>:
  KOKKOS_FUNCTION tickets() -> TicketIssuer<size_t>
  KOKKOS_FUNCTION request(ticket, FromEntity, ToEntity) -> NgpRequestConnections<MemSpace>&
  KOKKOS_FUNCTION request(ticket, FromEntity, ToEntityFuture) -> NgpRequestConnections<MemSpace>&
  KOKKOS_FUNCTION request(ticket, FromEntityFuture, ToEntity) -> NgpRequestConnections<MemSpace>&
  KOKKOS_FUNCTION request(ticket, FromEntityFuture, ToEntityFuture) -> NgpRequestConnections<MemSpace>&

NgpDestroyConnections<MemSpace>:
  KOKKOS_FUNCTION tickets() -> TicketIssuer<size_t>
  KOKKOS_FUNCTION request(ticket, FromEntity, ToEntity) -> NgpRequestConnections<MemSpace>&
*/

/// \brief A range of tickets issued by a TicketIssuer.
template <typename SizeT>
class TicketRange {
 public:
  using ticket_id = SizeT;
  using our_size_t = SizeT;

  TicketRange() = default;
  TicketRange(our_size_t begin, our_size_t count) : begin_(begin), count_(count) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION our_size_t begin() const noexcept { return begin_; }
  KOKKOS_INLINE_FUNCTION our_size_t end() const noexcept { return begin_ + count_; }
  KOKKOS_INLINE_FUNCTION our_size_t count() const noexcept { return count_; }
  // clang-format on

 private:
  our_size_t begin_{0};
  our_size_t count_{0};
};  // TicketRange

/// \brief Issues tickets for modifications in a given memory space.
///
/// \note This class uses dual-view-like semantics to manage ticket state. The NgpMemSpace is the device memory space
/// but the host can also claim tickets when activated. Importantly, only one memory space can claim tickets at a time
/// and changing the active space acts as a phase boundary that synchronizes ticket state between host and device.
template <typename NgpMemSpace, typename SizeT = size_t>
class TicketIssuer {
 public:
  using host_space = Kokkos::HostSpace;
  using memory_space = NgpMemSpace;
  using execution_space = typename NgpMemSpace::execution_space;
  using ticket_id = SizeT;
  using our_size_t = SizeT;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor w/ delayed initialization. Call initialize(device_active) to set up.
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer() = delete;

  KOKKOS_DEFAULTED_FUNCTION TicketIssuer(const TicketIssuer&) = default;
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer& operator=(const TicketIssuer&) = default;
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer(TicketIssuer&&) = default;
  KOKKOS_DEFAULTED_FUNCTION TicketIssuer& operator=(TicketIssuer&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~TicketIssuer() = default;

  /// \brief Construct a TicketIssuer with the specified initial active memory space.
  TicketIssuer(bool activate_device = true) active_space_dev_view_("TicketIssuer::active_space_dev_view"),
      ticket_counter_dev_view_("TicketIssuer::ticket_counter_dev_view"),
      count_finalized_dev_view_("TicketIssuer::count_finalized_dev_view"),
      ticket_counter_host_view_(Kokkos::create_mirror_view(ticket_counter_dev_view_)),
      active_space_host_view_(Kokkos::create_mirror_view(active_space_dev_view_)),
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
    MUNDY_THROW_ASSERT(active_space_dev_view_.is_allocated() == ticket_counter_dev_view_.is_allocated(),
                       std::runtime_error, "TicketIssuer::initialize() called on already initialized TicketIssuer.");

    active_space_dev_view_ = size_view_t("TicketIssuer::active_space_dev_view");
    ticket_counter_dev_view_ = size_view_t("TicketIssuer::ticket_counter_dev_view");
    ticket_counter_host_view_ = Kokkos::create_mirror_view(ticket_counter_dev_view_);
    active_space_host_view_ = Kokkos::create_mirror_view(active_space_dev_view_);

    active_space_host_view_() = activate_device;
    ticket_counter_host_view_() = 0;
    Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    Kokkos::deep_copy(ticket_counter_dev_view_, ticket_counter_host_view_);
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
  our_size_t finalize_count() {
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
  KOKKOS_INLINE_FUNCTION TicketRange<our_size_t> claim(our_size_t n) const {
    assert_active_space();
    assert_count_not_finalized();

    KOKKOS_IF_ON_HOST(our_size_t start_ticket = Kokkos::atomic_fetch_add(&ticket_counter_host_view_(), n);
                      return TicketRange<our_size_t>(start_ticket, n);)
    KOKKOS_IF_ON_DEVICE(our_size_t start_ticket = Kokkos::atomic_fetch_add(&ticket_counter_dev_view_(), n);
                        return TicketRange<our_size_t>(start_ticket, n);)
  }

  /// \brief Claim a single ticket atomically. Returns the claimed ticket ID.
  /// If you claim more tickets than are available, an exception is thrown (in debug mode).
  KOKKOS_INLINE_FUNCTION our_size_t claim() const {
    assert_active_space();
    assert_count_not_finalized();

    KOKKOS_IF_ON_HOST(return Kokkos::atomic_fetch_add(&ticket_counter_host_view_(), 1);)
    KOKKOS_IF_ON_DEVICE(return Kokkos::atomic_fetch_add(&ticket_counter_dev_view_(), 1);)
  }

  /// \brief Get the current count of issued tickets atomically.
  KOKKOS_INLINE_FUNCTION our_size_t count() const {
    assert_active_space();

    KOKKOS_IF_ON_HOST(return ticket_counter_host_view_();)
    KOKKOS_IF_ON_DEVICE(return ticket_counter_dev_view_();)
  }
  //@}

 private:
  //! \name Helper functions
  //@{

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    KOKKOS_IF_ON_HOST(bool host_is_active = active_space_host_view_(); MUNDY_THROW_ASSERT(
                          host_is_active, std::runtime_error, name + " called from host when device is active.");)
    KOKKOS_IF_ON_DEVICE(bool device_is_active = active_space_dev_view_(); MUNDY_THROW_ASSERT(
                            device_is_active, std::runtime_error, name + " called from device when host is active.");)
  }

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_count_not_finalized() const {
    KOKKOS_IF_ON_HOST(MUNDY_THROW_ASSERT(!count_finalized_host_view_(), std::runtime_error,
                                         name + " called after finalize_count() on host.");)
    KOKKOS_IF_ON_DEVICE(MUNDY_THROW_ASSERT(!count_finalized_dev_view_(), std::runtime_error,
                                           name + " called after finalize_count() on device.");)
  }
  //@}

  using bool_view_t = Kokkos::View<bool, memory_space>;
  using size_view_t = Kokkos::View<our_size_t, memory_space>;
  bool_view_t active_space_dev_view_;
  size_view_t ticket_counter_dev_view_;
  bool_view_t count_finalized_dev_view_;
  bool_view_t::HostMirror active_space_host_view_;
  size_view_t::HostMirror ticket_counter_host_view_;
  bool_view_t::HostMirror count_finalized_host_view_;
};  // TicketIssuer

struct FutureEntity {
  our_size_t ticket;
  unsigned request_helper_index;
};

template <typename NgpMemSpace, typename SizeT = size_t>
class NgpRequestEntitiesT {
 public:
  using memory_space = NgpMemSpace;
  using our_size_t = SizeT;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpRequestEntitiesT(unsigned helper_index)
      : index_(helper_index),
        active_space_dev_view_("NgpRequestEntitiesT::active_space_dev_view"),
        active_space_host_view_(Kokkos::create_mirror_view(active_space_dev_view_)),
        ticket_issuer_(),
        requests_("NgpRequestEntitiesT::requests", 0),
        created_entities_("NgpRequestEntitiesT::created_entities", 0) {
    // Initialize to device active
    active_space_host_view_() = true;
    Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
  }

  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesT(const NgpRequestEntitiesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesT& operator=(const NgpRequestEntitiesT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesT(NgpRequestEntitiesT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestEntitiesT& operator=(NgpRequestEntitiesT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpRequestEntitiesT() = default;
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from device
  void activate_host() {
    auto device_is_active = active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      active_space_host_view_() = false;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    }
  }

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from host
  void activate_device() {
    auto host_is_active = !active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      active_space_host_view_() = true;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    }
  }

  /// \brief Synchronize between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = active_space_host_view_();
    if (device_is_active) {
      ticket_issuer_.sync();
      Kokkos::deep_copy(requests_.view_host(), requests_.view_device());
    } else {
      ticket_issuer_.sync();
      Kokkos::deep_copy(requests_.view_device(), requests_.view_host());
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Get the ticket issuer for entity requests.
  KOKKOS_INLINE_FUNCTION TicketIssuer<NgpMemSpace, our_size_t>& tickets() {
    return ticket_issuer_;
  }

  /// \brief Record an entity creation request
  KOKKOS_INLINE_FUNCTION NgpRequestEntitiesT<NgpMemSpace>& request(our_size_t ticket, int owning_proc) {
    constexpr auto name = core::make_string_literal("NgpRequestEntitiesT::request");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(requests_.view_host()(ticket).owning_proc = owning_proc;)
    KOKKOS_IF_ON_DEVICE(requests_.view_device()(ticket).owning_proc = owning_proc;)

    return *this;
  }

  /// \brief Record an entity creation request with a specified entity ID.
  KOKKOS_INLINE_FUNCTION NgpRequestEntitiesT<NgpMemSpace>& request(our_size_t ticket,  //
                                                                   int owning_proc, stk::mesh::EntityId entity_id) {
    constexpr auto name = core::make_string_literal("NgpRequestEntitiesT::request");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(auto& req = requests_.view_host()(ticket); req.owning_proc = owning_proc;
                      req.entity_id = entity_id; req.has_entity_id = true;)
    KOKKOS_IF_ON_DEVICE(auto& req = requests_.view_device()(ticket); req.owning_proc = owning_proc;
                        req.entity_id = entity_id; req.has_entity_id = true;)

    return *this;
  }

  /// \brief Turn a ticket into a future entity.
  KOKKOS_INLINE_FUNCTION FutureEntity make_future_entity(our_size_t ticket) const {
    constexpr auto name = core::make_string_literal("NgpRequestEntitiesT::make_future_entity");
    assert_ticket_out_of_range<name>(ticket);

    FutureEntity future_entity{.ticket = ticket, .request_helper_index = index_};
    return future_entity;
  }

  /// \brief Fetch the requested entity using the given ticket.
  /// After process_requests, this can be called on either host or device.
  KOKKOS_INLINE_FUNCTION stk::mesh::Entity get_requested_entity(our_size_t ticket) const {
    constexpr auto name = core::make_string_literal("NgpRequestEntitiesT::get_requested_entity");
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(return created_entities_.view_host()(ticket);)
    KOKKOS_IF_ON_DEVICE(return created_entities_.view_device()(ticket);)
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    /// Instead of resizing, we'll just zero out the existing requests to avoid reallocations.
    ticket_issuer_.reset();
    Kokkos::deep_copy(requests_.view_host(), EntityRequest{});
    Kokkos::deep_copy(requests_.view_device(), EntityRequest{});
  }

  /// \brief Finalize counts for this request class. Users may no longer claim tickets after this call.
  our_size_t finalize_count() {
    our_size_t count = ticket_issuer_.finalize_count();
    if (requests_.extent(0) < count) {
      Kokkos::resize(requests_, count);
    }
  }
  //@}

 private:
  //! \name Helper functions
  //@{

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    KOKKOS_IF_ON_HOST(bool host_is_active = active_space_host_view_(); MUNDY_THROW_ASSERT(
                          host_is_active, std::runtime_error, name + " called from host when device is active.");)
    KOKKOS_IF_ON_DEVICE(bool device_is_active = active_space_dev_view_(); MUNDY_THROW_ASSERT(
                            device_is_active, std::runtime_error, name + " called from device when host is active.");)
  }

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_ticket_out_of_range(our_size_t ticket) const {
    MUNDY_THROW_ASSERT(ticket < ticket_issuer_.count(), std::out_of_range, name + " called with invalid ticket.");
  }
  //@}

  //! \name Member variables
  //@{

  /// @brief Internal struct representing a single entity request.
  /// Requests must always specify the owning processor, but entity ID is optional.
  struct EntityRequest {
    int owning_proc;
    stk::mesh::EntityId entity_id = stk::mesh::InvalidEntityId;
    bool has_entity_id = false;
  };

  using bool_view_t = Kokkos::View<bool, memory_space>;
  bool_view_t active_space_dev_view_;
  bool_view_t::HostMirror active_space_host_view_;

  TicketIssuer<NgpMemSpace, our_size_t> ticket_issuer_;

  using request_view_t = core::NgpViewT<EntityRequest*, NgpMemSpace>;
  request_view_t requests_;

  using entity_view_t = core::NgpViewT<stk::mesh::Entity*, NgpMemSpace>;
  entity_view_t created_entities_;
  //@}
};  // NgpRequestEntitiesT

template <typename NgpMemSpace, typename SizeT = size_t>
class NgpRequestConnectionT {
 public:
  using memory_space = NgpMemSpace;
  using our_size_t = SizeT;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpRequestConnectionT()
      : active_space_dev_view_("NgpRequestConnectionT::active_space_dev_view"),
        active_space_host_view_(Kokkos::create_mirror_view(active_space_dev_view_)),
        ticket_issuer_(),
        requests_("NgpRequestConnectionT::requests", 0),
        created_entities_("NgpRequestConnectionT::created_entities", 0) {
    // Initialize to device active
    active_space_host_view_() = true;
    Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
  }

  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionT(const NgpRequestConnectionT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionT& operator=(const NgpRequestConnectionT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionT(NgpRequestConnectionT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpRequestConnectionT& operator=(NgpRequestConnectionT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpRequestConnectionT() = default;
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from device
  void activate_host() {
    auto device_is_active = active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      active_space_host_view_() = false;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    }
  }

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from host
  void activate_device() {
    auto host_is_active = !active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      active_space_host_view_() = true;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    }
  }

  /// \brief Synchronize between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = active_space_host_view_();
    if (device_is_active) {
      ticket_issuer_.sync();
      Kokkos::deep_copy(requests_.view_host(), requests_.view_device());
    } else {
      ticket_issuer_.sync();
      Kokkos::deep_copy(requests_.view_device(), requests_.view_host());
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Get the ticket issuer for entity requests.
  KOKKOS_INLINE_FUNCTION TicketIssuer<NgpMemSpace, our_size_t>& tickets() {
    return ticket_issuer_;
  }

  /// \brief Record a connection between two entities (from_entity -> to_entity).
  /// Both entities may be either real entities or future entities.
  KOKKOS_INLINE_FUNCTION NgpRequestConnectionT<NgpMemSpace>& request(
      our_size_t ticket, core::variant<stk::mesh::Entity, FutureEntity> from_entity,
      core::variant<stk::mesh::Entity, FutureEntity> to_entity) {
    constexpr auto name = core::make_string_literal("NgpRequestConnectionT::request");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(auto& req = requests_.view_host()(ticket); req.from_entity = from_entity;
                      req.to_entity = to_entity;)
    KOKKOS_IF_ON_DEVICE(auto& req = requests_.view_device()(ticket); req.from_entity = from_entity;
                        req.to_entity = to_entity;)

    return *this;
  }

  /// \brief Fetch the requested entity using the given ticket.
  /// After process_requests, this can be called on either host or device.
  KOKKOS_INLINE_FUNCTION stk::mesh::Entity get_requested_entity(our_size_t ticket) const {
    constexpr auto name = core::make_string_literal("NgpRequestConnectionT::get_requested_entity");
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(return created_entities_.view_host()(ticket);)
    KOKKOS_IF_ON_DEVICE(return created_entities_.view_device()(ticket);)
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    /// Instead of resizing, we'll just zero out the existing requests to avoid reallocations.
    ticket_issuer_.reset();
    Kokkos::deep_copy(requests_.view_host(), EntityRequest{});
    Kokkos::deep_copy(requests_.view_device(), EntityRequest{});
  }

  /// \brief Finalize counts for this request class. Users may no longer claim tickets after this call.
  our_size_t finalize_count() {
    our_size_t count = ticket_issuer_.finalize_count();
    if (requests_.extent(0) < count) {
      Kokkos::resize(requests_, count);
    }
  }
  //@}

 private:
  //! \name Helper functions
  //@{

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    KOKKOS_IF_ON_HOST(bool host_is_active = active_space_host_view_(); MUNDY_THROW_ASSERT(
                          host_is_active, std::runtime_error, name + " called from host when device is active.");)
    KOKKOS_IF_ON_DEVICE(bool device_is_active = active_space_dev_view_(); MUNDY_THROW_ASSERT(
                            device_is_active, std::runtime_error, name + " called from device when host is active.");)
  }

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_ticket_out_of_range(our_size_t ticket) const {
    MUNDY_THROW_ASSERT(ticket < ticket_issuer_.count(), std::out_of_range, name + " called with invalid ticket.");
  }
  //@}

  //! \name Member variables
  //@{

  struct FutureEntity {
    our_size_t ticket;
    int entity_request_helper_id;
  };

  /// @brief Internal struct representing a single connection request.
  /// Requests must specify a pair of entities to connect. These may either be a real entity or a future entity (by
  /// ticket and request helper).
  struct ConnectionRequest {
    core::variant<stk::mesh::Entity, FutureEntity> from_entity;
    core::variant<stk::mesh::Entity, FutureEntity> to_entity;
  };

  using bool_view_t = Kokkos::View<bool, memory_space>;
  bool_view_t active_space_dev_view_;
  bool_view_t::HostMirror active_space_host_view_;

  TicketIssuer<NgpMemSpace, our_size_t> ticket_issuer_;

  using request_view_t = core::NgpViewT<ConnectionRequest*, NgpMemSpace>;
  request_view_t requests_;

  using entity_view_t = core::NgpViewT<stk::mesh::Entity*, NgpMemSpace>;
  entity_view_t created_entities_;
  //@}
};  // NgpRequestConnectionT

template <typename NgpMemSpace, typename SizeT = size_t>
class NgpDestroyEntityT {
 public:
  using memory_space = NgpMemSpace;
  using our_size_t = SizeT;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpDestroyEntityT()
      : active_space_dev_view_("NgpDestroyEntityT::active_space_dev_view"),
        active_space_host_view_(Kokkos::create_mirror_view(active_space_dev_view_)),
        ticket_issuer_(),
        requests_("NgpDestroyEntityT::requests", 0),
        created_entities_("NgpDestroyEntityT::created_entities", 0) {
    // Initialize to device active
    active_space_host_view_() = true;
    Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
  }

  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntityT(const NgpDestroyEntityT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntityT& operator=(const NgpDestroyEntityT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntityT(NgpDestroyEntityT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpDestroyEntityT& operator=(NgpDestroyEntityT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpDestroyEntityT() = default;
  //@}

  //! \name Control plane (HOST only)
  //@{

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from device
  void activate_host() {
    auto device_is_active = active_space_host_view_();
    if (device_is_active) {  // No-op if host is already active
      Kokkos::fence();
      active_space_host_view_() = false;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    }
  }

  /// \brief Sets the active memory space to host and synchronizes if needed.
  /// While active, both request and the ticket issuer will throw if used from host
  void activate_device() {
    auto host_is_active = !active_space_host_view_();
    if (host_is_active) {  // No-op if device is already active
      Kokkos::fence();
      active_space_host_view_() = true;
      Kokkos::deep_copy(active_space_dev_view_, active_space_host_view_);
    }
  }

  /// \brief Synchronize between active and inactive memory spaces.
  void sync() {
    Kokkos::fence();
    bool device_is_active = active_space_host_view_();
    if (device_is_active) {
      ticket_issuer_.sync();
      Kokkos::deep_copy(requests_.view_host(), requests_.view_device());
    } else {
      ticket_issuer_.sync();
      Kokkos::deep_copy(requests_.view_device(), requests_.view_host());
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Get the ticket issuer for entity requests.
  KOKKOS_INLINE_FUNCTION TicketIssuer<NgpMemSpace, our_size_t>& tickets() const noexcept {
    return ticket_issuer_;
  }

  /// \brief Record an entity destruction request
  KOKKOS_INLINE_FUNCTION NgpDestroyEntityT<NgpMemSpace>& destroy(our_size_t ticket, stk::mesh::Entity entity) {
    constexpr auto name = core::make_spring_literal("NgpDestroyEntityT::destroy");
    assert_active_space<name>();
    assert_ticket_out_of_range<name>(ticket);

    KOKKOS_IF_ON_HOST(requests_.view_host()(ticket) = entity;)
    KOKKOS_IF_ON_DEVICE(requests_.view_device()(ticket) = entity;)

    return *this;
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    /// Instead of resizing, we'll just zero out the existing requests to avoid reallocations.
    ticket_issuer_.reset();
    Kokkos::deep_copy(requests_.view_host(), stk::mesh::Entity{});
    Kokkos::deep_copy(requests_.view_device(), stk::mesh::Entity{});
  }

  /// \brief Finalize counts for this request class. Users may no longer claim tickets after this call.
  our_size_t finalize_count() {
    our_size_t count = ticket_issuer_.finalize_count();
    if (requests_.extent(0) < count) {
      Kokkos::resize(requests_, count);
    }
  }
  //@}

 private:
  //! \name Helper functions
  //@{

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_active_space() const {
    KOKKOS_IF_ON_HOST(bool host_is_active = active_space_host_view_(); MUNDY_THROW_ASSERT(
                          host_is_active, std::runtime_error, name + " called from host when device is active.");)
    KOKKOS_IF_ON_DEVICE(bool device_is_active = active_space_dev_view_(); MUNDY_THROW_ASSERT(
                            device_is_active, std::runtime_error, name + " called from device when host is active.");)
  }

  template <core::StringLiteral name>
  KOKKOS_INLINE_FUNCTION void assert_ticket_out_of_range(our_size_t ticket) const {
    MUNDY_THROW_ASSERT(ticket < ticket_issuer_.count(), std::out_of_range, name + " called with invalid ticket.");
  }
  //@}

  //! \name Member variables
  //@{

  using bool_view_t = Kokkos::View<bool, memory_space>;
  bool_view_t active_space_dev_view_;
  bool_view_t::HostMirror active_space_host_view_;

  TicketIssuer<NgpMemSpace, our_size_t> ticket_issuer_;

  using request_view_t = core::NgpViewT<stk::mesh::Entity*, NgpMemSpace>;
  request_view_t requests_;

  using entity_view_t = core::NgpViewT<stk::mesh::Entity*, NgpMemSpace>;
  entity_view_t created_entities_;
  //@}
};  // NgpDestroyEntityT

/// \brief Manages all modification requests in a given NGP memory space.
template <typename NgpMemSpace, typename SizeT = size_t>
class NgpModRequestsT {
 public:
  using memory_space = NgpMemSpace;
  using our_size_t = SizeT;

  //! \name Constructors / Destructors
  //@{

  /// \brief Default constructor.
  NgpModRequestsT() = default;

  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT(const NgpModRequestsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT& operator=(const NgpModRequestsT&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT(NgpModRequestsT&&) = default;
  KOKKOS_DEFAULTED_FUNCTION NgpModRequestsT& operator=(NgpModRequestsT&&) = default;

  KOKKOS_DEFAULTED_FUNCTION ~NgpModRequestsT() = default;
  //@}

  //! \name Accessors for request classes
  //@{

  /// \brief Get the entity request class for the given partition key.
  NgpRequestEntitiesT<NgpMemSpace>& request_entities(const stk::mesh::PartVector& parts) {
    return entity_requests_map_[impl::get_partition_key(parts)];
  }

  /// \brief Get the destroy entity request class.
  NgpDestroyEntitiesT<NgpMemSpace>& destroy_entities() {
    return destroy_entity_requests_;
  }

  /// \brief Get the connection request class.
  NgpRequestConnectionsT<NgpMemSpace>& request_connections() {
    return connection_requests_;
  }

  /// \brief Get the destroy connection request class.
  NgpDestroyConnectionsT<NgpMemSpace>& destroy_connections() {
    return destroy_connection_requests_;
  }
  //@}

  //! \name Mod cycle management
  //@{

  /// \brief Clears all internal request data to prepare for a fresh modification cycle.
  void reset() {
    for (auto& pair : entity_requests_map_) {
      pair.second.reset();
    }
    destroy_entity_requests_.reset();
    connection_requests_.reset();
    destroy_connection_requests_.reset();
  }

  /// \brief Finalize counts for all request classes. Users may no longer claim tickets after this call.
  void finalize_counts() {
    for (auto& pair : entity_requests_map_) {
      pair.second.finalize_counts();
    }
    destroy_entity_requests_.finalize_counts();
    connection_requests_.finalize_counts();
    destroy_connection_requests_.finalize_counts();
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
    bool we_started_mod_cycle = false;
    if (!bulk_data.in_modifiable_state()) {
      bulk_data.modification_begin();
      we_started_mod_cycle = true;
    }

    // Processing requests cannot be separated from one class to the next since we need cross cutting information such
    // as entity requests to determine sharing and allow for things like the requesting of connections to future
    // entities.

    if (we_started_mod_cycle) {
      bulk_data.modification_end();
    }
  }
  //@}

 private:
  // Entity requests are mapped by the partition key that identifies their collection of parts.
  // We'll always append the locally owned part to the given part list to ensure ownership.
  std::map<impl::PartitionKey, NgpRequestEntitiesT<NgpMemSpace>> entity_requests_map_;

  // All other request classes only require a single instance.
  NgpDestroyEntitiesT<NgpMemSpace> destroy_entity_requests_;
  NgpRequestConnectionsT<NgpMemSpace> connection_requests_;
  NgpDestroyConnectionsT<NgpMemSpace> destroy_connection_requests_;
};

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPMODREQUESTS_HPP_
