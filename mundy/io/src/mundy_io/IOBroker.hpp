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

#ifndef MUNDY_IO_IOBROKER_HPP_
#define MUNDY_IO_IOBROKER_HPP_

/// \file IOBroker.hpp
/// \brief Declaration of the IOBroker class

// C++ core lib
#include <algorithm>      // for std::transform
#include <filesystem>     // for std::filesystem::path::stem()
#include <memory>         // for std::shared_ptr, std::unique_ptr
#include <string>         // for std::string
#include <unordered_map>  // for std::unordered_map
#include <vector>         // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>                      // for Teuchos::ParameterList
#include <stk_io/DatabasePurpose.hpp>                     // for stk::io::DatabasePurpose
#include <stk_io/StkMeshIoBroker.hpp>                     // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/DumpMeshInfo.hpp>                 // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>                       // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>                // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>                         // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>                     // for stk::mesh::Selector
#include <stk_topology/topology.hpp>                      // for stk::topology
#include <stk_util/environment/LogWithTimeAndMemory.hpp>  // for stk::log_with_time_and_memory

// Mundy libs
#include <mundy_core/StringLiteral.hpp>     // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_mesh/StringToTopology.hpp>  // for mundy::mesh::string_to_rank
#include <mundy_meta/MeshReqs.hpp>          // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodExecutionInterface.hpp>  // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                  // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace io {

class IOBroker {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  IOBroker() = delete;

  /// \brief Constructor
  IOBroker(mundy::mesh::BulkData *const bulk_data_ptr, [[maybe_unused]] const Teuchos::ParameterList &fixed_params)
      : bulk_data_ptr_(bulk_data_ptr),
        meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()),
        stk_io_broker_(bulk_data_ptr->parallel()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                        "IOBroker: bulk_data_ptr cannot be a nullptr.");

    stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), "MuNDy IO");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(IOBroker::get_valid_fixed_params());

    // Set variables from the fixed_parameters
    exodus_database_output_filename_ = valid_fixed_params.get<std::string>("exodus_database_output_filename");
    exodus_database_output_filename_base_ = std::filesystem::path(exodus_database_output_filename_).stem();
    coordinate_field_name_ = valid_fixed_params.get<std::string>("coordinate_field_name");
    transient_coordinate_field_name_ = valid_fixed_params.get<std::string>("transient_coordinate_field_name");
    parallel_io_mode_ = valid_fixed_params.get<std::string>("parallel_io_mode");
    // Check the database purpose for output
    auto database_purpose = valid_fixed_params.get<std::string>("database_purpose");
    if (database_purpose == "results") {
      database_purpose_ = stk::io::WRITE_RESULTS;
    } else if (database_purpose == "restart") {
      database_purpose_ = stk::io::WRITE_RESTART;
    } else if (database_purpose == "append") {
      database_purpose_ = stk::io::APPEND_RESULTS;
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("IOBroker: incorrect database purpose: ") + database_purpose);
    }

    // Check if we are reading a RESTART or not, otherwise make sure that the input filename is set to an empty string
    if (valid_fixed_params.get<std::string>("enable_restart") == "true") {
      enable_restart_ = true;
      exodus_database_input_filename_ = valid_fixed_params.get<std::string>("exodus_database_input_filename");
    }

    // Set the coordinate field (tags as a Ioss::MESH role if not a restart, otherwise, need to defer setting the
    // internal pointer until later)
    set_coordinate_field();

    // Add IO to the parts we have enabled
    Teuchos::Array<std::string> enabled_io_parts =
        valid_fixed_params.get<Teuchos::Array<std::string>>("enabled_io_parts");
    for (const std::string &io_part_name : enabled_io_parts) {
      // Grab the Part and then assign it to IO
      stk::mesh::Part *io_part_ptr = meta_data_ptr_->get_part(io_part_name);
      stk::io::put_io_part_attribute(*io_part_ptr);
    }

    // Set the stk::io::StkMeshIoBroker bulk data to our bulk data
    stk_io_broker_.set_bulk_data(*bulk_data_ptr_);
    if (!parallel_io_mode_.empty()) {
      stk_io_broker_.property_add(Ioss::Property("PARALLEL_IO_MODE", parallel_io_mode_));
    }
    stk_io_broker_.property_add(Ioss::Property("MAXIMUM_NAME_LENGTH", 180));

    // Set the TRANSIENT fields and keep track of them.
    set_transient_fields(valid_fixed_params);

    // If we are restarting the mesh, read in the values here. This requires that the mesh be set up, but not committed.
    // Additionally, it requires that we have stored pointers to the fields that are going to be used for IO so that we
    // can add them as input fields.
    if (enable_restart_) {
      restart_mesh();
      synchronize_node_coordinates_from_transient();
    }
  }
  //@}

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("exodus_database_output_filename", std::string(default_exodus_database_output_filename_),
                               "Filename of the EXODUS database output file.");
    default_parameter_list.set("exodus_database_input_filename", std::string(default_exodus_database_input_filename_),
                               "Filename of the EXODUS database input file.");
    default_parameter_list.set("coordinate_field_name", std::string(default_coordinate_field_name_),
                               "Coordinates field for entire mesh.");
    default_parameter_list.set(
        "transient_coordinate_field_name", std::string(default_transient_coordinate_field_name_),
        "TRANSIENT coordinates field for entire mesh. Will be used for IO and mirrors the set COORDINATES field");
    default_parameter_list.set("parallel_io_mode", std::string(default_parallel_io_mode_), "Parallel IO mode [hdf5].");
    default_parameter_list.set("database_purpose", std::string(default_database_purpose_),
                               "Database Purpose [results,restart,append]");
    default_parameter_list.set("enable_restart", std::string(default_enable_restart_), "Enable RESTART.");

    // Create an empty vector of part names (forces it to exist)
    std::vector<std::string> default_io_part_names{""};
    Teuchos::Array<std::string> default_array_of_io_part_names(default_io_part_names);
    default_parameter_list.set("enabled_io_parts", default_array_of_io_part_names,
                               "The names of all parts to enable IO.");

    // Create an empty vector of field names (forces it to exist)
    std::vector<std::string> default_io_field_names{""};
    Teuchos::Array<std::string> default_array_of_io_field_names(default_io_field_names);
    // Do all 5 given ranks
    default_parameter_list.set("enabled_io_fields_node_rank", default_array_of_io_field_names,
                               "The names of all fields to enable IO for NODE_RANK.");
    default_parameter_list.set("enabled_io_fields_edge_rank", default_array_of_io_field_names,
                               "The names of all fields to enable IO for EDGE_RANK.");
    default_parameter_list.set("enabled_io_fields_face_rank", default_array_of_io_field_names,
                               "The names of all fields to enable IO for FACE_RANK.");
    default_parameter_list.set("enabled_io_fields_element_rank", default_array_of_io_field_names,
                               "The names of all fields to enable IO for ELEMENT_RANK.");

    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<IOBroker> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                       const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<IOBroker>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the coordinate field for IO (and the mesh)
  void set_coordinate_field();

  /// \brief Set the IO fields to TRANSIENT and keep track of Fields
  ///
  /// \param fixed_params [in] Parameters for IO fields.
  ///
  /// Set the field roles (Ioss::TRANSIENT). Do by looking at each rank individually. The two IF statements are there
  /// as sentinels to guard against the defaults that we have to set the parameter lists.
  void set_transient_fields(const Teuchos::ParameterList &valid_fixed_params);

  //@}

  //! \name Getters
  //@{

  /// \brief Get if RESTART is enabled
  bool get_enable_restart() {
    return enable_restart_;
  }

  //@}

  //! \name Printers
  //@{

  /// \brief Print mesh roles to std::cout
  void print_field_roles();

  /// \brief Print entire IOBroker
  void print_io_broker();

  //@}

  //! \name Actions
  //@{

  /// \brief Finalize ioBroker (close)
  void finalize_io_broker();

  /// \brief Setup the IO Broker output (open files if not a single write)
  void setup_io_broker();

  /// \brief Read the RESTART file
  void restart_mesh();

  /// \brief Update node coordinates from the TRANSIENT node coordinates
  void synchronize_node_coordinates_from_transient();

  /// \brief Copy node coordinates to TRANSIENT node coordinates
  void synchronize_node_coordinates_to_transient();

  /// \brief Write to disk
  void write_io_broker(double time);

  /// \brief Write a single timestep to disk
  void write_io_broker_timestep(size_t timestep, double time);

  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_exodus_database_output_filename_ = "results.exo";
  static constexpr std::string_view default_exodus_database_input_filename_ = "restart.exo";
  static constexpr std::string_view default_coordinate_field_name_ = "coordinates";
  static constexpr std::string_view default_transient_coordinate_field_name_ = "transient_coordinates";
  static constexpr std::string_view default_parallel_io_mode_ = "hdf5";
  static constexpr std::string_view default_database_purpose_ = "results";
  static constexpr std::string_view default_enable_restart_ = "false";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The stk::io::StkMeshIoBroker for output operations
  stk::io::StkMeshIoBroker stk_io_broker_;

  /// \brief EXODUS output database filename
  std::string exodus_database_output_filename_ = "";

  /// \brief EXODUS output database filename base (for timestep-delimited files)
  std::string exodus_database_output_filename_base_ = "";

  /// \brief EXODUS input database filename
  std::string exodus_database_input_filename_ = "";

  /// \brief Flag controlling if we do a RESTART or not
  bool enable_restart_ = false;

  /// \brief COORDINATES field name
  std::string coordinate_field_name_ = "";

  /// \brief TRANSIENT_COORDINATES field name
  std::string transient_coordinate_field_name_ = "";

  /// \brief Parallel IO mode
  std::string parallel_io_mode_ = "";

  /// \brief Purpose of database
  stk::io::DatabasePurpose database_purpose_ = stk::io::PURPOSE_UNKNOWN;

  /// \brief Enabled IO FIELDS
  std::vector<stk::mesh::FieldBase *> enabled_io_fields_;

  /// \brief coordinate field
  stk::mesh::FieldBase *coordinate_field_ptr_ = nullptr;

  /// \brief transient coordinate field
  stk::mesh::FieldBase *transient_coordinate_field_ptr_ = nullptr;

  /// \brief Index of output file
  size_t io_index_ = 0;
  //@}
};  // IoBroker

}  // namespace io

}  // namespace mundy

#endif  // MUNDY_IO_IOBROKER_HPP_
