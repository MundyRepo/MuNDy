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

#ifndef MUNDY_MESH_FILLMESH_HPP_
#define MUNDY_MESH_FILLMESH_HPP_

/// \file FillMesh.hpp
/// \brief Mundy-aware mirrors of stk::io's mesh-reading entry points

// C++ core libs
#include <string>  // for std::string

// Trilinos libs
#include <stk_io/DatabasePurpose.hpp>  // for stk::io::DatabasePurpose
#include <stk_io/FillMesh.hpp>         // for stk::io::fill_mesh, fill_mesh_with_fields, ...
#include <stk_io/StkMeshIoBroker.hpp>  // for stk::io::StkMeshIoBroker

// Mundy libs
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_mesh/LinkData.hpp>  // for mundy::mesh::reconcile_links_after_read

namespace mundy {

namespace mesh {

//! \name Reading a mesh from file
//@{

/// \brief Read a mesh into \p bulk_data and reconcile its links.
/// \param mesh_spec [in] The mesh to read: a file path, or a generated-mesh spec (e.g. "generated:4x4x4").
/// \param bulk_data [in] The mesh to populate.
inline void fill_mesh(const std::string& mesh_spec, BulkData& bulk_data) {
  stk::io::fill_mesh(mesh_spec, bulk_data);
  reconcile_links_after_read(bulk_data);
}

/// \brief Read a mesh through a caller-provided broker into \p bulk_data and reconcile its links.
///
/// Use this overload to configure broker properties before the read.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
/// \param io_broker [in] The IO broker to read through.
inline void fill_mesh(const std::string& mesh_spec, BulkData& bulk_data,
                      stk::io::StkMeshIoBroker& io_broker) {
  stk::io::fill_mesh(mesh_spec, bulk_data, io_broker);
  reconcile_links_after_read(bulk_data);
}

/// \brief Read and automatically decompose a mesh into \p bulk_data and reconcile its links.
///
/// The mesh is distributed across the MPI ranks by recursive coordinate bisection.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
inline void fill_mesh_with_auto_decomp(const std::string& mesh_spec, BulkData& bulk_data) {
  stk::io::fill_mesh_with_auto_decomp(mesh_spec, bulk_data);
  reconcile_links_after_read(bulk_data);
}

/// \brief Read and automatically decompose a mesh through a caller-provided broker, reconciling its links.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
/// \param io_broker [in] The IO broker to read through.
inline void fill_mesh_with_auto_decomp(const std::string& mesh_spec, BulkData& bulk_data,
                                        stk::io::StkMeshIoBroker& io_broker) {
  stk::io::fill_mesh_with_auto_decomp(mesh_spec, bulk_data, io_broker);
  reconcile_links_after_read(bulk_data);
}

/// \brief Read a mesh through a caller-owned broker into \p bulk_data and reconcile its links.
///
/// The broker is used as configured; set its databases, properties, and selectors before calling.
/// \param io_broker [in] The IO broker to read through.
/// \param mesh_spec [in] The mesh to read.
/// \param bulk_data [in] The mesh to populate.
/// \param purpose [in] Whether the database is read as a mesh or as a restart.
inline void fill_mesh_preexisting(stk::io::StkMeshIoBroker& io_broker, const std::string& mesh_spec,
                                  BulkData& bulk_data,
                                  stk::io::DatabasePurpose purpose = stk::io::READ_MESH) {
  stk::io::fill_mesh_preexisting(io_broker, mesh_spec, bulk_data, purpose);
  reconcile_links_after_read(bulk_data);
}

/// \brief Read a mesh with its field data into \p in_bulk, reconcile its links, and report the database time steps.
/// \param in_file [in] The mesh file to read.
/// \param in_bulk [in] The mesh to populate.
/// \param num_steps [out] The number of time steps on the database.
/// \param max_time [out] The largest time value on the database.
inline void fill_mesh_save_step_info(const std::string& in_file, BulkData& in_bulk, int& num_steps,
                                     double& max_time) {
  stk::io::fill_mesh_save_step_info(in_file, in_bulk, num_steps, max_time);
  reconcile_links_after_read(in_bulk);
}

/// \brief Read a mesh with its field data into \p bulk_data and reconcile its links.
///
/// Field data at the database's final time step is restored in addition to the mesh.
/// \param in_file [in] The mesh file to read.
/// \param bulk_data [in] The mesh to populate.
/// \param purpose [in] Whether the database is read as a mesh or as a restart.
inline void fill_mesh_with_fields(const std::string& in_file, BulkData& bulk_data,
                                  stk::io::DatabasePurpose purpose = stk::io::READ_MESH) {
  stk::io::fill_mesh_with_fields(in_file, bulk_data, purpose);
  reconcile_links_after_read(bulk_data);
}

/// \brief Read a mesh with its field data through a caller-provided broker into \p bulk_data and reconcile its links.
/// \param in_file [in] The mesh file to read.
/// \param io_broker [in] The IO broker to read through.
/// \param bulk_data [in] The mesh to populate.
/// \param purpose [in] Whether the database is read as a mesh or as a restart.
inline void fill_mesh_with_fields(const std::string& in_file, stk::io::StkMeshIoBroker& io_broker,
                                  BulkData& bulk_data,
                                  stk::io::DatabasePurpose purpose = stk::io::READ_MESH) {
  stk::io::fill_mesh_with_fields(in_file, io_broker, bulk_data, purpose);
  reconcile_links_after_read(bulk_data);
}
//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_FILLMESH_HPP_
