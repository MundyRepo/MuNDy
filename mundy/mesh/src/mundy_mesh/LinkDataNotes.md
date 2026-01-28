TODO(palmerb4): Cyclic dependency refactor:
LinkData is just a thin interface over LinkDataCSRManager and LinkDataCOOManager.
LinkDataCSRManager need not depend pn the LinkData itself. It can instead depend on LinkDataCOOManager.

The only issue that I'm having is that the LinkDataCOOManager has functions that must be called inside of
kernels. For example, declare_relation.

The current design of NgpLinkDataCSRManager is nice in that, we have a single class that works for any 
memory space with LinkDataCSRManager in charge of wrangling multiple memory spaces.

This works because the user touches the CSR connectivity via the partitions, which contain efficient inlined
functions. The same is not true for the COO connectivity, since the user simply interacts with functions like
declare_relation, which modify stk::mesh::Fields. We could employ the new strategy of fields

LinkData.coo(mem_space) -> NgpLinkDataCOOT<MemSpace>

The core problem is that NgpLinkDataT needs to call get_updated_ngp_mesh, which isn't valid during mod cycles.
That's why LinkData currently implements a custom non-ngp interface.

Ok, what about:

LinkData.crs_data<READ_WRITE>(mem_space) -> NgpLinkCSRDataT<MemSpace>                          
LinkData.synchronize_crs(mem_space) -> void

LinkData.coo_data<READ_WRITE>(mem_space) -> NgpLinkCOODataT<MemSpace>                     
LinkData.synchronize_coo(mem_space) -> void

Accessing coo or crs data in a memory space that isn't host is forbidden during mesh modifications.

We'll use type specializations to differentiate between host/device behavior.


///////////////////////
// Some new thoughts //

STK appears to have each FieldBase store a host and device FieldData pointer.
If a device pointer already exists, then they call update on the device pointer.
The memory space of the device pointer is the first non-host memory space given to
update_or_create_device_field_data. This is also true for the deviceFieldDataManager.
Async updates simply take the existing manager in the first mem-space and have it perform 
an update using the given exec space and fence on this exec space. Non-async updates use 
the stk::ngp::ExecSpace for the update, under the assertion that the given exec space
can access the memory space of the device pointer.

A field data is only considered host if its memory space is exactly stk::ngp::HostMemSpace.
Must there exist a m_hostFieldData? Yes. STK always assumes that there exists a valid 
host field data given to the field data at construction. Field's constructor is the one
that creates the new host field data.

The important thing to remember is that, simply because there exists a host data,
doesn't mean that it is up-to-date or contains any amount of data. In STK's case, the
second you create a new bucket, allocate_bucket_field_data is called to allocate the host 
field data. We want to instead allow for the host data to be nearly non-existent unless
the user explicitly requests it via synchronize(stk::ngp::HostMemSpace{}). It also feels like
we want to only declare a single host space and a single device space, rather than one per
memory space and then cast from our device space to the provided memory space.

I misunderstood STK's design. You may have one and only one non-host memory space. Once that
space is set, you may perform updates in different execution spaces that can access said 
memory space, but you may not request field data in different memory spaces. Basically,
you either use stk::ngp::MemSpace everywhere or you use your own space everywhere.

We can similarly assume that there is one and only one non-host memory space.

LinkData knows that it (in an abstract sense) is really a dual view and that LinkData is just the
host data and NgpLinkDataT is just the device data. It will store a typeindex of the
NgpMemSpace. get_updated_ngp_link_data will handle the logic of creating the NgpLinkDataT.

///////////////
// A new day //

What we really want is the ability to have a single 

If we remove the ngp_mesh from the following
    CSRBucketConn, CSRPartitionView, LinkCSRData
then they may live in any memory space, as they are really just pure data. They can become
stale as a result of mesh modifications and can be updated with deep_copy(trg_space, src_space).

The same is not true for LinkCOOData, as it contains ngp_fields. That said, even if those fields
are stale, that doesn't mean we can't touch the regular host fields during mod cycles. This 
brings us to a more dual-view-like approach, where we have a host-only mod_safe_declare_relation 
function.

The problem with this design is that it makes the LinkCSRData not offering a dual-view design 
feel weird/inconsistent. It feels less weird if we label mod_safe_declare_relation as host-only
for now... I take that back, it still feels weird. We have a declare_relation, which can only
modify data in the given MemSpace and a mod_safe_declare_relation, which can only modify data
on the host. mod_safe_* makes it seem like its the same operation but safe to call during mod
cycles. We would be forced to call these host_declare_relation, as we do currently. This again
makes LinkCSRData feel weird for not offering host_* functions.

LocalLinkData calls update_crs_from_coo, which then calls impl::COOtoCSRSynchronizer. This class
relies on the NgpMesh, which lives in the NgpMeshSpace. Consequently, either we attempt to fetch
the NgpMesh in MemSpace, which will fail when MemSpace doesn't equal every other NgpMemSpace, or
we fetch NgpMesh in the default NgpMeshSpace, and accept an exec space that must have SpaceAccessible.

I'm not going to add Const vs non-const to start. We'll refactor for that after. The user will have 
the ability to modify the CSR data even though its really just a copy and they shouldn't

Side note, to avoid needing to use friendship to hide content, have LinkMetaData be friends with
impl:: non-member functions for getting its internal fields.


Quick, we are missing a runtime to compile time jump table over memory spaces.
I want to perform
deep_copy(MyTypeBase, MyTypeBase) -> deep_copy(MyType<MemSpace1>, MyType<MemSpace2>)




/////////////////////////////////////////////////////////////////////////////
// Classes and functions. Each may depend only on things written above it. //

LinkMetaData:
  public:
  // Getters
  name() -> std::string
  link_rank() -> EntityRank
  linked_entity_ids_field() -> const auto
  linked_entity_ranks_field() -> const auto
  universal_link_part() -> Part
  mesh_meta_data() -> MetaData

  // Actions
  declare_link_part(part_name, link_dimensionality_for_this_part) -> Part&
  declare_link_assembly_part(part_name) -> Part&
  add_link_support_to_part(part, link_dimensionality_for_this_part) -> void
  add_link_support_to_assembly_part(part) -> void
  
  private:
  // Internal getters
  linked_entity_ids_field() -> auto
  linked_entity_ranks_field() -> auto
  linked_entities_field() -> auto
  linked_entities_crs_field() -> auto
  linked_entity_bucket_ids_field() -> auto
  linked_entity_bucket_ords_field() -> auto
  link_crs_needs_updated_field() -> auto
  link_marked_for_destruction_field() -> auto

  // Helpers 
  put_link_fields_on_part(part, link_dimensionality_for_this_part)

impl::NgpLinkMetaDataT<NgpMemSpace>:
  public:
  // Getters
  link_rank() -> EntityRank
  universal_link_part_ordinal() -> Part
  mesh_meta_data() -> MetaData
  link_meta_meta_data() -> LinkMetaData

  ngp_linked_entity_ids_field() -> const auto
  ngp_linked_entity_ranks_field() -> const auto
  ngp_linked_entity_ids_field() -> auto
  ngp_linked_entity_ranks_field() -> auto
  ngp_linked_entities_field() -> auto
  ngp_linked_entities_crs_field() -> auto
  ngp_linked_entity_bucket_ids_field() -> auto
  ngp_linked_entity_bucket_ords_field() -> auto
  ngp_link_crs_needs_updated_field() -> auto
  ngp_link_marked_for_destruction_field() -> auto

impl::get_updated_ngp_link_meta_data<NgpMemSpace>(link_meta_data) 
  -> impl::NgpLinkMetaDataT<NgpMemSpace>

LinkCSRBucketConnT<MemSpace>:  // Raw data in any space
  public:
  // Getters
  bucket_id() -> unsigned
  size() -> size_t
  capacity() -> size_t
  get_connected_links(offset_into_bucket) -> ConnectedEntities
  num_connected_links(offset_into_bucket) -> unsigned

  // Actions
  initialize_bucket_attributes(Bucket) -> void
  dump() -> void

// For those using default STK spaces
using LinkCSRBucketConn = LinkCSRBucketConnT<stk::ngp::HostMemSpace>;
using NgpLinkCSRBucketConn = LinkCSRBucketConnT<stk::ngp::MemSpace>;

LinkCSRPartitionT<MemSpace>: // Raw data in any space
  public:
  // Getters
  id() -> unsigned
  key() -> PartitionKey<MemSpace>
  link_rank() -> EntityRank
  link_dimensionality() -> unsigned
  selector() -> Selector
  
  // Actions
  contains(PartOrdinal) -> bool
  initialize_attributes(id, key, rank, dim, bulk_data) -> void
  connects_to(rank, bucket_id) -> bool
  num_buckets(rank) -> unsigned
  get_crs_bucket_conn(rank, bucket_id) -> LinkCSRBucketConnT<MemSpace>&
  get_connected_links(rank, entity_index) -> ConnectedEntities
  num_connected_links(rank, entity_index) -> unsigned

using LinkCSRPartition = LinkCSRPartitionT<stk::ngp::HostMemSpace>;
using NgpLinkCSRPartition = LinkCSRPartitionT<stk::ngp::MemSpace>;

(Const)LinkCSRDataT<MemSpace>:  // Raw data in any space
  public:
  // Getters
  is_valid() -> bool
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData

  // Actions
  get_all_partitions() -> const (Const)CSRPartitionViewT<MemSpace>&
  get_or_create_partitions(selector) -> const (Const)CSRPartitionViewT<MemSpace>&
  select_all_partitions() -> Selector
  private:
  sort_partitions_by_id() -> void

using (Const)LinkCSRData = (Const)LinkCSRDataT<stk::ngp::HostMemSpace>;
using (Const)NgpLinkCSRData = (Const)LinkCSRDataT<stk::ngp::MemSpace>;

ConstLinkCOOData:  // Host only | Valid during mesh modifications
  public:
  // Getters
  is_valid() -> bool
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData

  // Actions (each one will accept Entity/MeshIndex/FastMeshIndex overloads)
  get_linked_entity(linker, link_ordinal) -> Entity
  get_linked_entity_index(linker, link_ordinal) -> FastMeshIndex
  get_linked_entity_id(linker, link_ordinal) -> EntityId
  get_linked_entity_rank(linker, link_ordinal) -> EntityRank

ConstNgpLinkCOODataT<NgpMemSpace>:  // Device only | Invalid during mesh modifications
  public:
  // Getters
  is_valid() -> bool
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData

  // Actions (each one will accept Entity/MeshIndex/FastMeshIndex overloads)
  get_linked_entity(linker, link_ordinal) -> Entity
  get_linked_entity_index(linker, link_ordinal) -> FastMeshIndex
  get_linked_entity_id(linker, link_ordinal) -> EntityId
  get_linked_entity_rank(linker, link_ordinal) -> EntityRank

LinkCOOData : ConstLinkCOOData:  // Host only | Valid during mesh modifications
  public:
  // Getters
  is_valid() -> bool
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData

  // Actions (each one will accept Entity/MeshIndex/FastMeshIndex overloads)
  declare_relation(linker, linked_entity, link_ordinal) -> void
  destroy_relation(linker, link_ordinal) -> void

NgpLinkCOODataT<NgpMemSpace> : ConstLinkCOOData:  // Device only | Invalid during mesh modifications
  public:
  // Getters
  is_valid() -> bool
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData

  // Actions (each one will accept Entity/MeshIndex/FastMeshIndex overloads)
  declare_relation(linker, linked_entity, link_ordinal) -> void
  destroy_relation(linker, link_ordinal) -> void

using (Const)NgpLinkCOOData = (Const)NgpLinkCOODataT<stk::ngp::MemSpace>;

impl::NgpCOOtoCSRSynchronizerT<NgpMemSpace>:  // Not valid during mesh modifications | NgpMemSpace only
  public:
  NgpCOOtoCSRSynchronizerT(ConstLinkCSRDataT<NgpMemSpace>, ConstLinkCOODataT<NgpMemSpace>, Selector)
  is_crs_up_to_date() -> bool
  update_crs_from_coo() -> void
  check_crs_coo_consistency() -> void
  flag_dirty_linked_buckets_of_modified_links() -> void
  reset_dirty_linked_buckets() -> void
  gather_part_1_count() -> void
  gather_part_2_partial_sum() -> void
  scatter_part_1_setup() -> void
  scatter_part_2_fill() -> void
  finalize_crs_update() -> void

LinkDeclarationRequestsT<MemSpace>:  // Raw data in any space
  public:
  link_meta_data() -> LinkMetaData
  link_parts() -> PartVector
  link_rank() -> EntityRank
  link_dimensionality() -> unsigned
  capacity() -> size_t
  size() -> size_t
  clear() -> void
  reserve(new_capacity) -> void
  request_link(linked_entities) -> void

using LinkDeclarationRequests = LinkDeclarationRequestsT<stk::ngp::HostMemSpace>;
using NgpLinkDeclarationRequests = LinkDeclarationRequestsT<stk::ngp::MemSpace>;

LinkDestructionRequests:  // Host only
  public:
  request_destruction(Linker) -> void
  request_destruction(LinkerMeshIndex) -> void
  request_destruction(LinkerFastMeshIndex) -> void

NgpLinkDestructionRequestsT<NgpMemSpace>:  // Device only
  public:
  request_destruction(Linker) -> void
  request_destruction(LinkerMeshIndex) -> void
  request_destruction(LinkerFastMeshIndex) -> void

using NgpLinkDestructionRequests = NgpLinkDestructionRequestsT<stk::ngp::MemSpace>;

LinkDataBase: 
  public:
  // Sync
  sync_crs_to_host()   -> virtual void
  sync_crs_to_device() -> virtual void
  modify_crs_on_host()   -> virtual void
  modify_crs_on_device() -> virtual void
  clear_crs_host_sync_state()   -> virtual void
  clear_crs_device_sync_state() -> virtual void

  sync_coo_to_host()   -> virtual void
  sync_coo_to_device() -> virtual void
  modify_coo_on_host()   -> virtual void
  modify_coo_on_device() -> virtual void
  clear_coo_host_sync_state()   -> virtual void
  clear_coo_device_sync_state() -> virtual void

LinkData : public LinkDataBase: 
  public:
  // Getters
  is_valid() -> bool
  mesh_meta_data() -> MetaData
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData
  link_rank() -> EntityRank

  // COO and CSR on the host
  crs_data<DataAccess>() -> LinkCSRData<HostMemSpace>& or ConstLinkCSRData<HostMemSpace>&
  coo_data<DataAccess>() -> LinkCOOData<HostMemSpace>& or ConstLinkCOOData<HostMemSpace>&

  check_crs_coo_consistency(exec_space, selector) -> void
  check_crs_coo_consistency(exec_space)           -> void
  check_crs_coo_consistency(selector)             -> void
  check_crs_coo_consistency()                     -> void

  // Sync
  sync_crs_to_host()   -> override void
  sync_crs_to_device() -> override void
  modify_crs_on_host()   -> override void
  modify_crs_on_device() -> override void
  clear_crs_host_sync_state()   -> override void
  clear_crs_device_sync_state() -> override void

  sync_coo_to_host()   -> override void
  sync_coo_to_device() -> override void
  modify_coo_on_host()   -> override void
  modify_coo_on_device() -> override void
  clear_coo_host_sync_state()   -> override void
  clear_coo_device_sync_state() -> override void

  // Requests
  declaration_requests(
      link_parts, requested_dimensionality, requested_capacity) -> LinkDeclarationRequestsT<HostMemSpace>
  destruction_requests() -> LinkDestructionRequests
  process_requests(assume_fully_consistent) -> void | Processes all requests in the current space

NgpLinkDataT<NgpMemSpace> : public LinkDataBase:
  public:
  // Getters
  is_valid() -> bool
  mesh_meta_data() -> MetaData
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData
  link_rank() -> EntityRank

  // COO and CSR within MemSpace
  crs_data<DataAccess>() -> LinkCSRData<NgpMemSpace>& or ConstLinkCSRData<NgpMemSpace>&
  coo_data<DataAccess>() -> LinkCOOData<NgpMemSpace>& or ConstLinkCOOData<NgpMemSpace>&

  // CSR and COO synchronization | device only | not valid during mesh modifications
  update_crs_from_coo(exec_space) -> void
  update_crs_from_coo()           -> void

  check_crs_coo_consistency(exec_space, selector) -> void
  check_crs_coo_consistency(exec_space)           -> void
  check_crs_coo_consistency(selector)             -> void
  check_crs_coo_consistency()                     -> void

  is_crs_up_to_date(exec_space) -> bool
  is_crs_up_to_date()           -> bool

  // Sync
  sync_crs_to_host()   -> override void
  sync_crs_to_device() -> override void
  modify_crs_on_host()   -> override void
  modify_crs_on_device() -> override void
  clear_crs_host_sync_state()   -> override void
  clear_crs_device_sync_state() -> override void

  sync_coo_to_host()   -> override void
  sync_coo_to_device() -> override void
  modify_coo_on_host()   -> override void
  modify_coo_on_device() -> override void
  clear_coo_host_sync_state()   -> override void
  clear_coo_device_sync_state() -> override void

  // Requests
  declaration_requests(
      link_parts, requested_dimensionality, requested_capacity) -> LinkDeclarationRequestsT<NgpMemSpace>
  destruction_requests(mem_space) -> NgpLinkDestructionRequestsT<NgpMemSpace>
  process_requests(assume_fully_consistent) -> void | Processes all requests in the current space

get_updated_ngp_link_data<NgpMemSpace>(link_data) -> NgpLinkDataT<NgpMemSpace>















































LocalLinkData<MemSpace>:  // Memory-space "local". Doesn't care about other memory spaces
  public:
  // Getters
  is_valid() -> bool
  mesh_meta_data() -> MetaData
  link_meta_data() -> LinkMetaData
  bulk_data() -> BulkData
  link_rank() -> EntityRank

  // COO and CSR within MemSpace
  crs_data() -> LinkCSRDataT<MemSpace>& or ConstLinkCSRDataT<MemSpace>&
  coo_data() -> LinkCOODataT<MemSpace>& or ConstLinkCOODataT<MemSpace>&

  // CSR and COO synchronization | not valid during mesh modifications
  update_crs_from_coo(exec_space) -> void
  update_crs_from_coo()           -> void

  check_crs_coo_consistency(exec_space, selector) -> void
  check_crs_coo_consistency(exec_space)           -> void
  check_crs_coo_consistency(selector)             -> void
  check_crs_coo_consistency()                     -> void

  is_crs_up_to_date(exec_space) -> bool
  is_crs_up_to_date()           -> bool

  // Requests
  declaration_requests(
      link_parts, requested_dimensionality, requested_capacity) -> LinkDeclarationRequests<MemSpace>
  destruction_requests(mem_space) -> LinkDestructionRequests<MemSpace>
  process_requests(assume_fully_consistent) -> void

LinkData : public LocalLinkData<HostMemSpace>:
  public:
  sync_crs_to_host()   -> void
  sync_crs_to_device() -> void
  modify_crs_on_host()   -> void
  modify_crs_on_device() -> void
  clear_crs_host_sync_state()   -> void
  clear_crs_device_sync_state() -> void

  sync_coo_to_host()   -> void
  sync_coo_to_device() -> void
  modify_coo_on_host()   -> void
  modify_coo_on_device() -> void
  clear_coo_host_sync_state()   -> void
  clear_coo_device_sync_state() -> void

NgpLinkDataT : public LocalLinkData<NgpMemSpace>:
  public:
  sync_crs_to_host()   -> void
  sync_crs_to_device() -> void
  modify_crs_on_host()   -> void
  modify_crs_on_device() -> void
  clear_crs_host_sync_state()   -> void
  clear_crs_device_sync_state() -> void

  sync_coo_to_host()   -> void
  sync_coo_to_device() -> void
  modify_coo_on_host()   -> void
  modify_coo_on_device() -> void
  clear_coo_host_sync_state()   -> void
  clear_coo_device_sync_state() -> void

get_updated_ngp_link_data<NgpMemSpace>(link_data) -> NgpLinkDataT<NgpMemSpace>
