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

#ifndef MUNDY_META_MESHREQS_HPP_
#define MUNDY_META_MESHREQS_HPP_

/// \file MeshReqs.hpp
/// \brief Declaration of the MeshReqs class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <sstream>      // for std::stringstream
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_Array.hpp>               // for Teuchos::Array
#include <stk_mesh/base/Bucket.hpp>        // for stk::mesh::get_default_bucket_capacity
#include <stk_mesh/base/Part.hpp>          // for stk::mesh::Part
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>   // for mundy::mesh::BulkData
#include <mundy_meta/FieldReqs.hpp>  // for mundy::meta::FieldReqs, mundy::meta::FieldReqsBase
#include <mundy_meta/PartReqs.hpp>   // for mundy::meta::PartReqs

namespace mundy {

namespace meta {

/// \class MeshReqs
/// \brief A set requirements imposed upon a MetaMesh, its Parts, and its Fields.
class MeshReqs {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  MeshReqs() = default;

  /// \brief Construct a fully specified set of mesh requirements.
  ///
  /// \param comm [in] The MPI communicator.
  explicit MeshReqs(const stk::ParallelMachine& comm);
  //@}

  //! \name Setters
  //@{

  /// \brief Set the spatial dimension of the mash.
  /// \param spatial_dimension [in] The dimension of the space within which the parts and entities reside.
  MeshReqs& set_spatial_dimension(const unsigned spatial_dimension);

  /// \brief Set the node coordinates name.
  /// \param node_coordinates_name [in] The name of the node coordinates.
  MeshReqs& set_node_coordinates_name(const std::string& node_coordinates_name);

  /// \brief Set the names assigned to each rank.
  /// \param entity_rank_names [in] The names assigned to each rank.
  MeshReqs& set_entity_rank_names(const std::vector<std::string>& entity_rank_names);

  /// \brief Set the MPI communicator to be used by STK.
  /// \param comm [in] The MPI communicator.
  MeshReqs& set_communicator(const stk::ParallelMachine& comm);

  /// \brief Set the chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  /// \param aura_option [in] The chosen Aura option.
  MeshReqs& set_aura_option(const mundy::mesh::BulkData::AutomaticAuraOption& aura_option);

  /// \brief Set the field data manager.
  /// \param field_data_manager_ptr [in] Pointer to an existing field data manager.
  MeshReqs& set_field_data_manager(stk::mesh::FieldDataManager* const field_data_manager_ptr);

  /// \brief Set the upper bound on the number of mesh entities that may be associated with a single bucket.
  ///
  /// Although subject to change, the maximum bucket capacity is currently 1024 and the default capacity is 512.
  ///
  /// \param bucket_capacity [in] The bucket capacity.
  MeshReqs& set_bucket_capacity(const unsigned bucket_capacity);

  /// \brief Set the flag specifying if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  MeshReqs& set_upward_connectivity_flag(const bool enable_upward_connectivity);

  /// \brief Delete the spatial dimension constraint (if it exists).
  MeshReqs& delete_spatial_dimension();

  /// \brief Delete the node coordinates name constraint (if it exists).
  MeshReqs& delete_node_coordinates_name();

  /// \brief Delete the entity rank names constraint (if it exists).
  MeshReqs& delete_entity_rank_names();

  /// \brief Delete the communicator constraint (if it exists).
  MeshReqs& delete_communicator();

  /// \brief Delete the aura option constraint (if it exists).
  MeshReqs& delete_aura_option();

  /// \brief Delete the field data manager constraint (if it exists).
  MeshReqs& delete_field_data_manager();

  /// \brief Delete the bucket capacity constraint (if it exists).
  MeshReqs& delete_bucket_capacity();

  /// \brief Delete the upward connectivity flag constraint (if it exists).
  MeshReqs& delete_upward_connectivity_flag();

  /// \brief Add the provided field to the mesh, given that it is valid and does not conflict with existing fields.
  ///
  /// When a field is added to the entire mesh, we also add it to all parts, which, in turn, adds it to all their
  /// subparts. If the field already exists, we sync the two fields with each other.
  ///
  /// \param field_reqs_ptr [in] Pointer to the field parameters to add to the mesh.
  MeshReqs& add_and_sync_field_reqs(std::shared_ptr<FieldReqsBase> field_reqs_ptr);

  /// \brief Add the provided field to the mesh, given that it is valid and does not conflict with existing fields.
  ///
  /// When a field is added to the entire mesh, we also add it to all parts, which, in turn, adds it to all their
  /// subparts. If the field already exists, we sync the two fields with each other.
  ///
  /// \param field_name [in] Name of the field to add to the mesh.
  /// \param field_rank [in] Rank of the field to add to the mesh.
  /// \param field_dimension [in] Dimension of the field to add to the mesh.
  /// \param field_min_number_of_states [in] Minimum number of states for the field to add to the mesh.
  ///
  /// \tparam FieldType [in] The type of the field to add to the mesh.
  template <typename FieldType>
  MeshReqs& add_field_reqs(const std::string& field_name, const stk::topology::rank_t& field_rank,
                           const unsigned& field_dimension, const unsigned& field_min_number_of_states) {
    return add_and_sync_field_reqs(
        std::make_shared<FieldReqs<FieldType>>(field_name, field_rank, field_dimension, field_min_number_of_states));
  }

  /// \brief Add the provided part to the mesh, given that it is valid.
  ///
  /// TODO(palmerb4): Are there any restrictions on what can and cannot be a part? If so, encode them here.
  ///
  /// Whenever a part is added, we add all of our fields to it. If the part already exists, we sync the two parts with
  /// each other.
  ///
  /// \param part_reqs_ptr [in] Pointer to the part requirements to add to the mesh.
  MeshReqs& add_and_sync_part_reqs(std::shared_ptr<PartReqs> part_reqs_ptr);

  /// \brief Add the provided part to the mesh, given that it is valid.
  ///
  /// \param part_name [in] The name of the part to add to the mesh.
  MeshReqs& add_part_reqs(const std::string& part_name) {
    return add_and_sync_part_reqs(std::make_shared<PartReqs>(part_name));
  }
  /// \brief Add the provided part to the mesh, given that it is valid.
  ///
  /// Whenever a part is added, we add all of our fields to it.
  ///
  /// \param part_name [in] The name of the part to add to the mesh.
  /// \param part_topology [in] Topology of entities within the sub-part.
  MeshReqs& add_part_reqs(const std::string& part_name, const stk::topology::topology_t part_topology) {
    return add_and_sync_part_reqs(std::make_shared<PartReqs>(part_name, part_topology));
  }

  /// \brief Add the provided part to the mesh, given that it is valid.
  ///
  /// Whenever a part is added, we add all of our fields to it.
  ///
  /// \param part_name [in] The name of the part to add to the mesh.
  /// \param part_rank [in] The rank of the part to add to the mesh.
  MeshReqs& add_part_reqs(const std::string& part_name, const stk::topology::rank_t part_rank) {
    return add_and_sync_part_reqs(std::make_shared<PartReqs>(part_name, part_rank));
  }

  /// \brief Require that an attribute with the given name be present on the mesh.
  ///
  /// \param attribute_name [in] The name of the attribute that must be present on the mesh.
  MeshReqs& add_mesh_attribute(const std::string& attribute_name);
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the spatial dimension is constrained or not.
  bool constrains_spatial_dimension() const;

  /// \brief Get if the node coordinates name is constrained or not.
  bool constrains_node_coordinates_name() const;

  /// \brief Get if the entity rank names are constrained or not.
  bool constrains_entity_rank_names() const;

  /// \brief Get if the communicator is constrained or not.
  bool constrains_communicator() const;

  /// \brief Get if the aura option is constrained or not.
  bool constrains_aura_option() const;

  /// \brief Get if the field data manager is constrained or not.
  bool constrains_field_data_manager() const;

  /// \brief Get if the bucket capacity is constrained or not.
  bool constrains_bucket_capacity() const;

  /// \brief Get if the upward connectivity flag is constrained or not.
  bool constrains_upward_connectivity_flag() const;

  /// \brief Get if the mesh is fully specified.
  bool is_fully_specified() const;

  /// \brief Return the dimension of the space within which the parts and entities reside.
  unsigned get_spatial_dimension() const;

  /// \brief Return the node coordinates name.
  std::string get_node_coordinates_name() const;

  /// \brief Return the names assigned to each rank.
  std::vector<std::string> get_entity_rank_names() const;

  /// \brief Return the MPI communicator to be used by STK.
  stk::ParallelMachine get_communicator() const;

  /// \brief Return the chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  mundy::mesh::BulkData::AutomaticAuraOption get_aura_option() const;

  /// \brief Return the field data manager.
  stk::mesh::FieldDataManager* get_field_data_manager() const;

  /// \brief Return the upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned get_bucket_capacity() const;

  /// \brief Return if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  bool get_upward_connectivity_flag() const;

  /// \brief Return the mesh field map for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldReqsBase>>>& get_mesh_ranked_field_map();

  /// \brief Return the mesh part map.
  std::map<std::string, std::shared_ptr<PartReqs>>& get_mesh_part_map();

  /// \brief Return the required mesh attribute names.
  std::vector<std::string>& get_mesh_attribute_names();
  //@}

  //! \name Actions
  //@{

  /// \brief Declare the mesh that this class defines including any of its fields, parts, and their
  /// fields/parts/subparts.
  ///
  /// The only setting that must be specified before declaring the mesh is the MPI communicator; all other settings have
  /// default options which will be used if not set.
  ///
  /// Notice that this function isn't const, as it adds the node coordinates field to our mesh reqs and syncs the node
  /// coordinates number of states.
  std::shared_ptr<mundy::mesh::BulkData> declare_mesh();

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   - TODO(palmerb4): What are the mesh invariants set by STK?
  MeshReqs& check_if_valid();

  /// \brief Synchronize (merge and rectify differences) the current requirements with another \c MeshReqs.
  ///
  /// \param part_reqs_ptr [in] An \c MeshReqs object to sync with the current object.
  MeshReqs& sync(std::shared_ptr<MeshReqs> mesh_reqs_ptr);

  /// \brief Dump the contents of \c MeshReqs to the given stream (defaults to std::cout).
  void print(std::ostream& os = std::cout, int indent_level = 0) const;

  /// \brief Dump the parts of \c MeshReqs to the given stream (defaults to std::cout).
  void print_parts(std::ostream& os = std::cout, int indent_level = 0) const;

  /// \brief Return a string representation of the current set of requirements.
  std::string get_reqs_as_a_string() const;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr unsigned default_spatial_dimension_ = 3;
  static constexpr std::string_view default_node_coordinates_name_ = "NODE_COORDS";
  static const inline std::vector<std::string> default_entity_rank_names_ = {"NODE", "EDGE", "FACE", "ELEMENT",
                                                                             "CONSTRAINT"};
  static constexpr mundy::mesh::BulkData::AutomaticAuraOption default_aura_option_ = mundy::mesh::BulkData::AUTO_AURA;
  static constexpr stk::mesh::FieldDataManager* default_field_data_manager_ptr_ = nullptr;
  static const unsigned default_bucket_capacity_;  // Unlike the others, this parameter cannot be filled inline.
  static constexpr bool default_upward_connectivity_flag_ = true;
  //@}

  //! \name Private member functions
  //@{

  /// \brief Set the master field requirements for this class.
  MeshReqs& set_master_mesh_reqs(std::shared_ptr<MeshReqs> master_mesh_req_ptr);

  /// \brief Get the master mesh requirements for this class.
  std::shared_ptr<MeshReqs> get_master_mesh_reqs();

  /// \brief Get if the current reqs have a master mesh reqs.
  bool has_master_mesh_reqs() const;
  //@}

  //! \name Private data
  //@{

  /// \brief Pointer to the master mesh requirements.
  std::shared_ptr<MeshReqs> master_mesh_reqs_ptr_ = nullptr;

  /// \brief The dimension of the space within which the parts and entities reside.
  unsigned spatial_dimension_;

  /// \brief The name of the node coordinates.
  std::string node_coordinates_name_;

  /// \brief The names assigned to each rank.
  std::vector<std::string> entity_rank_names_;

  /// \brief The MPI communicator to be used by STK.
  stk::ParallelMachine communicator_;

  /// \brief The chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  mundy::mesh::BulkData::AutomaticAuraOption aura_option_;

  /// \brief Pointer to an existing field data manager.
  stk::mesh::FieldDataManager* field_data_manager_ptr_;

  /// \brief Upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned bucket_capacity_;

  /// \brief A flag specifying if upward connectivity will be enabled or not.
  bool upward_connectivity_flag_;

  /// \brief If we are driven by a master MeshReqs object.
  bool has_master_mesh_reqs_ = false;

  /// \brief If the spacial dimension is set or not.
  bool spatial_dimension_is_set_ = false;

  /// \brief If the node coordinates name is set or not.
  bool node_coordinates_name_is_set_ = false;

  /// \brief If the names of each rank are set or not.
  bool entity_rank_names_is_set_ = false;

  /// \brief If the MPI communicator is set or not.
  bool communicator_is_set_ = false;

  /// \brief If the aura option is set or not.
  bool aura_option_is_set_ = false;

  /// \brief If the field manager is set or not.
  bool field_data_manager_ptr_is_set_ = false;

  /// \brief If the bucket capacity is set or not.
  bool bucket_capacity_is_set_ = false;

  /// \brief If the upward connectivity flag is set or not.
  bool upward_connectivity_flag_is_set_ = false;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldReqsBase>>> mesh_ranked_field_maps_{stk::topology::NUM_RANKS};

  /// \brief A map from part name to the part params for that part.
  std::map<std::string, std::shared_ptr<PartReqs>> mesh_part_map_;

  /// \brief A vector of required mesh attribute names.
  std::vector<std::string> required_mesh_attribute_names_;
  //@}
};  // MeshReqs

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_MESHREQS_HPP_
