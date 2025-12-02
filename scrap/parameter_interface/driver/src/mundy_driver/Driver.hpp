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

#ifndef MUNDY_DRIVER_DRIVER_HPP_
#define MUNDY_DRIVER_DRIVER_HPP_

/// \file Driver.hpp
/// \brief Declaration of the Driver class

// C++ core libs
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy includes
#include <MundyDriver_config.hpp>                                     // for HAVE_MUNDYDRIVER_*
#include <mundy_meta/MeshReqs.hpp>                                    // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                                 // for mundy::meta::StringBasedMetaFactory
#include <mundy_meta/MetaMethodExecutionInterface.hpp>                // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionInterface
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS

#ifdef HAVE_MUNDYDRIVER_MUNDYAGENTS
#include <mundy_agents/Agents.hpp>             // for mundy::agents::Agents
#include <mundy_agents/HierarchyOfAgents.hpp>  // for mundy::agents::HierarchyOfAgents
#include <mundy_agents/RegisterAgents.hpp>     // MUNDY_REGISTER_AGENTS
#endif                                         // HAVE_MUNDYDRIVER_MUNDYAGENTS

#ifdef HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/Constraints.hpp>                // for mundy::constraints::Constraints
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#endif                                                      // HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYDRIVER_MUNDYIO
#include <mundy_io/IOBroker.hpp>  // for mundy::io::IOBroker
#endif                            // HAVE_MUNDYDRIVER_MUNDYIO

#ifdef HAVE_MUNDYDRIVER_MUNDYLINKERS
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>                // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>              // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>               // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>         // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/Linkers.hpp>                               // for mundy::linkers::Linkers
#include <mundy_linkers/NeighborLinkers.hpp>                       // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSphereLinkers
#endif                                                             // HAVE_MUNDYDRIVER_MUNDYLINKERS

#ifdef HAVE_MUNDYDRIVER_MUNDYSHAPES
#include <mundy_shapes/ComputeAABB.hpp>            // for mundy::shapes::ComputeAABB
#include <mundy_shapes/ComputeBoundingRadius.hpp>  // for mundy::shapes::ComputeBoundingRadius
#include <mundy_shapes/ComputeOBB.hpp>             // for mundy::shapes::ComputeOBB
#include <mundy_shapes/DeclareAndInitShapes.hpp>   // for mundy::shapes::DeclareAndInitShapes
#include <mundy_shapes/Spheres.hpp>                // for mundy::shapes::Spheres
#include <mundy_shapes/Spherocylinders.hpp>        // for mundy::shapes::Spherocylinders
#endif                                             // HAVE_MUNDYDRIVER_MUNDYSHAPES

namespace mundy {

namespace driver {

/// \brief A factory for creating meta methods that are configurable via the Configurator.
/// The core requirements for these classes is that they have a void execute function which the Configurator knows how
/// to call.
template <typename PolymorphicBaseType>
using DriverMetaMethodFactory =
    mundy::meta::StringBasedMetaFactory<PolymorphicBaseType,
                                        mundy::meta::make_registration_string("DRIVER_META_METHODS")>;

/// \class Driver
/// \brief Master simulation class that holds all of the information, and most importantly, the mesh
class Driver {
 public:
  //! \name Constructors and destructors
  //@{

  /// \brief Default constructor
  /// This will not set the parallel environment
  Driver();

  /// \brief Construct with parallel environment
  ///
  /// \param comm [in] The MPI Communicator.
  explicit Driver(const stk::ParallelMachine &comm);

  //@}

  //! \name Print/format/query information
  //@{

  /// \brief Get the registered MetaMethodExecutionInterface
  static std::string get_registered_meta_method_execution_interface();

  /// \brief Get the registered MetaMethodSubsetExecutionInterface
  static std::string get_registered_meta_method_subset_execution_interface();

  /// \brief Get the registered MetaMethodPairwiseSubsetExecutionInterface
  static std::string get_registered_meta_method_pairwise_subset_execution_interface();

  /// \brief Get all registered classes
  static std::string get_registered_classes();

  //@}

  //! @name Setters and Getters
  //@{

  Driver &set_spatial_dimension(const unsigned spatial_dimension);

  Driver &set_communicator(const stk::ParallelMachine &comm);

  Driver &set_node_coordinate_field_name(const std::string &node_coordinate_field_name);

  //@}

  //   /// \brief Add a Meta Class instance to the driver
  //   void add_meta_class_instance(const std::string& method_type, const std::string& method_name,
  //                                const Teuchos::ParameterList& fixed_params,
  //                                const Teuchos::ParameterList& mutable_params);

 private:
  //! \name Default parameters
  //@{

  /// \brief MetaMethod type names
  static constexpr std::array<std::string_view, 3> metamethod_types_ = {
      "meta_method_execution_interface", "meta_method_subset_execution_interface",
      "meta_method_pairwise_subset_execution_interface"};

  //@}

  //! \name Internal members
  //@{

  /// \brief Has a communicator
  bool has_comm_ = false;

  /// \brief Number of dimensions
  unsigned spatial_dimension_ = 0;

  /// \brief Mundy mesh requirements pointer
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr_ = nullptr;

  /// \brief Mundy bulk data pointer
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_ = nullptr;

  /// \brief Mundy meta data pointer
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_ = nullptr;

  /// \brief Coordinate field name
  std::string node_coordinate_field_name_ = "";

  /// \brief The MPI communicator to use (STK)
  stk::ParallelMachine comm_ = MPI_COMM_NULL;

  //}

  //! \name MetaMethod vectors and string lookup maps
  //@{

  std::vector<std::shared_ptr<mundy::meta::MetaMethodExecutionInterface<void>>> meta_methods_;
  std::vector<std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>>> meta_methods_subset_;
  std::vector<std::shared_ptr<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>>
      meta_methods_pairwise_subset_;

  std::unordered_map<std::string, unsigned int> meta_method_string_to_id_;
  std::unordered_map<std::string, unsigned int> meta_method_subset_string_to_id_;
  std::unordered_map<std::string, unsigned int> meta_method_pairwise_subset_string_to_id_;

  //@}
};  // Driver

}  // namespace driver

}  // namespace mundy

// TODO(cje): Think about moving these into their own location? Right now we have the Driver and Configurator which both
// sort of need access to these. For now its okay because the Configurator knows how to build the Driver.
#ifdef HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS
MUNDY_REGISTER_AGENTS(mundy::constraints::Constraints)
MUNDY_REGISTER_AGENTS(mundy::constraints::HookeanSprings)
MUNDY_REGISTER_METACLASS("COMPUTE_CONSTRAINT_FORCING", mundy::constraints::ComputeConstraintForcing,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("DECLARE_AND_INIT_CONSTRAINTS", mundy::constraints::DeclareAndInitConstraints,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodExecutionInterface<void>>)
#endif  // HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYDRIVER_MUNDYLINKERS
MUNDY_REGISTER_AGENTS(mundy::linkers::Linkers)
MUNDY_REGISTER_AGENTS(mundy::linkers::NeighborLinkers)
MUNDY_REGISTER_AGENTS(mundy::linkers::neighbor_linkers::SphereSphereLinkers)
MUNDY_REGISTER_METACLASS("COMPUTE_SIGNED_SEPARATION_DISTANCE_AND_CONTACT_NORMAL",
                         mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("DESTROY_NEIGHBOR_LINKERS", mundy::linkers::DestroyNeighborLinkers,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("EVALUATE_LINKER_POTENTIALS", mundy::linkers::EvaluateLinkerPotentials,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS(
    "GENERATE_NEIGHBOR_LINKERS", mundy::linkers::GenerateNeighborLinkers,
    mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("LINKER_POTENTIAL_FORCE_REDUCTION", mundy::linkers::LinkerPotentialForceReduction,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
#endif  // HAVE_MUNDYDRIVER_MUNDYLINKERS

#ifdef HAVE_MUNDYDRIVER_MUNDYSHAPES
MUNDY_REGISTER_AGENTS(mundy::shapes::Spheres)
MUNDY_REGISTER_AGENTS(mundy::shapes::Spherocylinders)
MUNDY_REGISTER_METACLASS("COMPUTE_AABB", mundy::shapes::ComputeAABB,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("COMPUTE_BOUNDING_RADIUS", mundy::shapes::ComputeBoundingRadius,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("COMPUTE_OBB", mundy::shapes::ComputeOBB,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodSubsetExecutionInterface<void>>)
MUNDY_REGISTER_METACLASS("DECLARE_AND_INIT_SHAPES", mundy::shapes::DeclareAndInitShapes,
                         mundy::driver::DriverMetaMethodFactory<mundy::meta::MetaMethodExecutionInterface<void>>)
#endif  // HAVE_MUNDYDRIVER_MUNDYSHAPES

#endif  // MUNDY_DRIVER_DRIVER_HPP_
