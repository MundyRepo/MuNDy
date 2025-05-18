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

#ifndef MUNDY_CONSTRAINTS_DECLARE_AND_INITIALIZE_CONSTRAINTS_TECHNIQUES_CHAINOFSPRINGS_HPP_
#define MUNDY_CONSTRAINTS_DECLARE_AND_INITIALIZE_CONSTRAINTS_TECHNIQUES_CHAINOFSPRINGS_HPP_

/// \file ChainOfSprings.hpp
/// \brief Declaration of the ChainOfSprings class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_topology/topology.hpp>        // for stk::topology

// Mundy libs
#include <mundy_constraints/AngularSprings.hpp>  // for mundy::constraints::AngularSprings
#include <mundy_constraints/HookeanSprings.hpp>  // for mundy::constraints::HookeanSprings
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ArchlengthCoordinateMapping.hpp>  // for mundy::constraints::...::ArchlengthCoordinateMapping
#include <mundy_core/MakeStringArray.hpp>               // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>                 // for mundy::core::StringLiteral
#include <mundy_core/throw_assert.hpp>                  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                      // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                      // for mundy::mesh::MetaData
#include <mundy_meta/MeshReqs.hpp>                      // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                   // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>                    // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodExecutionInterface.hpp>  // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                  // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_shapes/Spheres.hpp>                 // for mundy::shapes::Spheres
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace constraints {

namespace declare_and_initialize_constraints {

namespace techniques {

/// \brief The ChainOfSprings class is a MetaMethod that declares and initializes a chain of springs.
/// Our goal is to declare a chain of HookeanSprings and AngularSprings and initialize them in 3D space along a line.
///
/// The chain of springs contains num_nodes nodes. Each node is connected to its neighbors like so
/*
/// n1       n3        n5        n7
///  \      /  \      /  \      /
///   s1   s2   s3   s4   s5   s6
///    \  /      \  /      \  /
///     n2        n4        n6
/// Angular springs are hard to draw with ASCII art, but they are centered at every node and connected to the node's
neighbors:
///   a1 connects has a center node at n2 and connects to n1 and n3.
///   a2 connects has a center node at n3 and connects to n2 and n4.
///   and so on.
///
/// STK EntityId-wise. Nodes are numbered sequentially from 1 to num_nodes. Springs are numbered sequentially from 1 to
num_nodes-1.
/// and angular springs are numbered sequentially from num_springs+1 to num_spring+num_nodes-2.
*/
/// Parameters:
///   - 'generate_hookean_springs' (bool): [Optional, Defaults to true] Whether to generate the hookean springs.
///   - 'generate_angular_springs' (bool): [Optional, Defaults to false] Whether to generate the angular springs.
///   - 'generate_spheres_at_nodes' (bool): [Optional, Defaults to false] Whether to generate spheres at the nodes.
///   - 'generate_spherocylinder_segments_along_edges' (bool): [Optional, Defaults to false] Whether to generate
///     spherocylinder segments along the edges.
///   - 'hookean_springs_part_names' (Array of strings): [Optional, Defaults to 'HOOKEAN_SPRINGS'] The names of the
///   parts to
/// which we will add the generated hookean springs. Only used if the user sets generate_hookean_springs to true.
///   - 'angular_springs_part_names' (Array of strings): [Optional, Defaults to 'ANGULAR_SPRINGS'] The names of the
///   parts to
/// which we will add the generated angular springs. Only used if the user sets generate_angular_springs to true.
///   - 'num_nodes' (int): [Optional, Defaults to 2] The number of nodes in the chain.
///   - 'coordinate_map' (std::shared_ptr<ArchlengthCoordinateMapping<3,3>>): [Optional, Defaults to a straight line
///   along x
/// with center 0 and length 1.
class ChainOfSprings : public mundy::meta::MetaMethodExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodExecutionInterface<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ChainOfSprings() = delete;

  /// \brief Constructor
  ChainOfSprings(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshReqs
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(ChainOfSprings::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    const bool generate_hookean_springs = valid_fixed_params.get<bool>("generate_hookean_springs");
    const bool generate_angular_springs = valid_fixed_params.get<bool>("generate_angular_springs");
    const bool generate_spheres_at_nodes = valid_fixed_params.get<bool>("generate_spheres_at_nodes");
    const bool generate_spherocylinder_segments_along_edges =
        valid_fixed_params.get<bool>("generate_spherocylinder_segments_along_edges");
    MUNDY_THROW_REQUIRE(
        generate_hookean_springs || generate_angular_springs || generate_spheres_at_nodes ||
            generate_spherocylinder_segments_along_edges,
        std::invalid_argument,
        "ChainOfSprings: At least one of the objects must be generated. Currently, all are turned off!");

    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();

    if (generate_hookean_springs) {
      Teuchos::Array<std::string> hookean_spring_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>(
          "hookean_springs_part_names", mundy::core::make_string_array(HookeanSprings::get_name()));

      for (int i = 0; i < hookean_spring_part_names.size(); i++) {
        const std::string part_name = hookean_spring_part_names[i];
        if (part_name == HookeanSprings::get_name()) {
          // No specialization is required.
        } else {
          // The specialized part must be a subset of the hookean springs part.
          auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
          part_reqs->set_part_name(part_name);
          part_reqs->set_part_topology(stk::topology::BEAM_2);
          HookeanSprings::add_and_sync_subpart_reqs(part_reqs);
        }
      }

      mesh_reqs_ptr->sync(HookeanSprings::get_mesh_requirements());
    }

    if (generate_angular_springs) {
      Teuchos::Array<std::string> angular_spring_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>(
          "angular_springs_part_names", mundy::core::make_string_array(AngularSprings::get_name()));

      for (int i = 0; i < angular_spring_part_names.size(); i++) {
        const std::string part_name = angular_spring_part_names[i];
        if (part_name == AngularSprings::get_name()) {
          // No specialization is required.
        } else {
          // The specialized part must be a subset of the angular springs part.
          auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
          part_reqs->set_part_name(part_name);
          part_reqs->set_part_topology(stk::topology::BEAM_3);
          AngularSprings::add_and_sync_subpart_reqs(part_reqs);
        }
      }

      mesh_reqs_ptr->sync(AngularSprings::get_mesh_requirements());
    }

    if (generate_spheres_at_nodes) {
      Teuchos::Array<std::string> sphere_part_names =
          mundy::core::make_string_array(mundy::shapes::Spheres::get_name());
      for (int i = 0; i < sphere_part_names.size(); i++) {
        const std::string part_name = sphere_part_names[i];
        if (part_name == mundy::shapes::Spheres::get_name()) {
          // No specialization is required.
        } else {
          // The specialized part must be a subset of the sphere part.
          auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
          part_reqs->set_part_name(part_name);
          mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
        }
      }

      mesh_reqs_ptr->sync(mundy::shapes::Spheres::get_mesh_requirements());
    }

    if (generate_spherocylinder_segments_along_edges) {
      Teuchos::Array<std::string> spherocylinder_part_names =
          mundy::core::make_string_array(mundy::shapes::SpherocylinderSegments::get_name());
      for (int i = 0; i < spherocylinder_part_names.size(); i++) {
        const std::string part_name = spherocylinder_part_names[i];
        if (part_name == mundy::shapes::SpherocylinderSegments::get_name()) {
          // No specialization is required.
        } else {
          // The specialized part must be a subset of the spherocylinder part.
          auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
          part_reqs->set_part_name(part_name);
          mundy::shapes::SpherocylinderSegments::add_and_sync_subpart_reqs(part_reqs);
        }
      }

      mesh_reqs_ptr->sync(mundy::shapes::SpherocylinderSegments::get_mesh_requirements());
    }

    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set<bool>("generate_hookean_springs", true, "Whether to generate the hookean springs.")
            .set<bool>("generate_angular_springs", false, "Whether to generate the angular springs.")
            .set<bool>("generate_spheres_at_nodes", false, "Whether to generate spheres at the nodes.")
            .set<bool>("generate_spherocylinder_segments_along_edges", false,
                       "Whether to generate spherocylinder segments along the edges.")
            .set("hookean_springs_part_names", mundy::core::make_string_array(HookeanSprings::get_name()),
                 "The names of the parts to which we will add the generated hookean springs.")
            .set("angular_springs_part_names", mundy::core::make_string_array(AngularSprings::get_name()),
                 "The names of the parts to which we will add the generated angular springs.")
            .set("sphere_part_names", mundy::core::make_string_array(mundy::shapes::Spheres::get_name()),
                 "The names of the parts to which we will add the generated spheres.")
            .set("spherocylinder_segment_part_names",
                 mundy::core::make_string_array(mundy::shapes::SpherocylinderSegments::get_name()),
                 "The names of the parts to which we will add the generated spherocylinder segments.");

    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    const double center_x = 0.0;
    const double center_y = 0.0;
    const double center_z = 0.0;
    const double length = 1.0;
    const double orientation_x = 1.0;
    const double orientation_y = 0.0;
    const double orientation_z = 0.0;
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("num_nodes", default_num_nodes_, "The number of nodes in the chain.")
            .set<size_t>("element_id_start", 1u, "The starting ID for the elements.")
            .set<size_t>("node_id_start", 1u, "The starting ID for the nodes.")
            .set("hookean_spring_constant", 1.0, "The spring constant for the hookean springs.")
            .set("hookean_spring_rest_length", 1.0, "The rest length for the hookean springs.")
            .set("angular_spring_constant", 1.0, "The spring constant for the angular springs.")
            .set("angular_spring_rest_angle", 0.0, "The rest angle for the angular springs.")
            .set("sphere_radius", 1.0, "The radius of the spheres at the nodes.")
            .set("spherocylinder_segment_radius", 1.0, "The radius of the spherocylinder segments.")
            .set<std::shared_ptr<ArchlengthCoordinateMapping>>(
                "coordinate_mapping",
                std::make_shared<StraightLine>(default_num_nodes_, center_x, center_y, center_z, length, orientation_x,
                                               orientation_y, orientation_z),
                "The user-defined map function for the spring coordinates.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<ChainOfSprings>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the ID of the i'th node in the chain.
  /// \note Simply because the i'th index increases sequentially, the IDs need not.
  stk::mesh::EntityId get_node_id(const size_t &sequential_node_index) const;

  /// \brief Get the ID of the i'th hookean spring in the chain.
  stk::mesh::EntityId get_hookean_spring_id(const size_t &sequential_hookean_spring_index) const;

  /// \brief Get the ID of the i'th angular spring in the chain.
  stk::mesh::EntityId get_angular_spring_id(const size_t &sequential_angular_spring_index) const;

  /// \brief Get the ID of the i'th sphere in the chain.
  stk::mesh::EntityId get_sphere_id(const size_t &sequential_sphere_index) const;

  /// \brief Get the ID of the i'th spherocylinder segment in the chain.
  stk::mesh::EntityId get_spherocylinder_segment_id(const size_t &sequential_spherocylinder_segment_index) const;

  /// \brief Get the i'th node in the chain.
  /// Returns an invalid entity if the node does not exist.
  stk::mesh::Entity get_node(const size_t &sequential_node_index) const;

  /// \brief Get the i'th hookean spring in the chain.
  /// Returns an invalid entity if the element does not exist.
  stk::mesh::Entity get_hookean_spring(const size_t &sequential_hookean_spring_index) const;

  /// \brief Get the i'th angular spring in the chain.
  /// Returns an invalid entity if the element does not exist.
  stk::mesh::Entity get_angular_spring(const size_t &sequential_angular_spring_index) const;

  /// \brief Get the i'th sphere in the chain.
  /// Returns an invalid entity if the element does not exist.
  stk::mesh::Entity get_sphere(const size_t &sequential_sphere_index) const;

  /// \brief Get the i'th spherocylinder segment in the chain.
  /// Returns an invalid entity if the element does not exist.
  stk::mesh::Entity get_spherocylinder_segment(const size_t &sequential_spherocylinder_segment_index) const;

  /// \brief Get the i'th node in the chain.
  /// Throws if the node does not exist.
  stk::mesh::Entity get_valid_node(const size_t &sequential_node_index) const;

  /// \brief Get the i'th hookean spring in the chain.
  /// Throws if the element does not exist.
  stk::mesh::Entity get_valid_hookean_spring(const size_t &sequential_hookean_spring_index) const;

  /// \brief Get the i'th angular spring in the chain.
  /// Throws if the element does not exist.
  stk::mesh::Entity get_valid_angular_spring(const size_t &sequential_angular_spring_index) const;

  /// \brief Get the i'th sphere in the chain.
  /// Throws if the element does not exist.
  stk::mesh::Entity get_valid_sphere(const size_t &sequential_sphere_index) const;

  /// \brief Get the i'th spherocylinder segment in the chain.
  /// Throws if the element does not exist.
  stk::mesh::Entity get_valid_spherocylinder_segment(const size_t &sequential_spherocylinder_segment_index) const;
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute() override;

 private:
  //! \name Default parameters
  //@{

  static constexpr size_t default_num_nodes_ = 2;
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of nodes in the chain.
  size_t num_nodes_;

  /// \brief Number of hookean springs in the chain.
  size_t num_hookean_springs_;

  /// \brief Number of angular springs in the chain.
  size_t num_angular_springs_;

  /// \brief Number of spheres in the chain.
  size_t num_spheres_;

  /// \brief Number of spherocylinder segments in the chain.
  size_t num_spherocylinder_segments_;

  /// \brief If we should generate the hookean springs or not.
  bool generate_hookean_springs_;

  /// \brief If we should generate the angular springs or not.
  bool generate_angular_springs_;

  /// \brief If we should generate spheres at the nodes or not.
  bool generate_spheres_at_nodes_;

  /// \brief If we should generate spherocylinder segments along the edges or not.
  bool generate_spherocylinder_segments_along_edges_;

  // Temporary variables TODO(palmerb4) replace these with parameter initialization maps.
  double hookean_spring_constant_;
  double hookean_spring_rest_length_;
  double angular_spring_constant_;
  double angular_spring_rest_angle_;
  double sphere_radius_;
  double spherocylinder_segment_radius_;

  /// \brief The user-defined map function for the node coordinates.
  std::shared_ptr<ArchlengthCoordinateMapping> coordinate_map_ptr_;

  /// \brief Element id start.
  size_t element_id_start_ = 1;

  /// \brief Node id start.
  size_t node_id_start_ = 1;

  /// \brief The hookean spring parts.
  std::vector<stk::mesh::Part *> hookean_spring_part_ptrs_;

  /// \brief The angular spring parts.
  std::vector<stk::mesh::Part *> angular_spring_part_ptrs_;

  /// \brief The sphere parts.
  std::vector<stk::mesh::Part *> sphere_part_ptrs_;

  /// \brief The spherocylinder segment parts.
  std::vector<stk::mesh::Part *> spherocylinder_segment_part_ptrs_;

  /// \brief The spring node coordinate field pointer.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief The hookean spring constant element field pointer.
  stk::mesh::Field<double> *element_hookean_spring_constant_field_ptr_ = nullptr;

  /// \brief The hookean spring rest length element field pointer.
  stk::mesh::Field<double> *element_hookean_spring_rest_length_field_ptr_ = nullptr;

  /// \brief The angular spring constant element field pointer.
  stk::mesh::Field<double> *element_angular_spring_constant_field_ptr_ = nullptr;

  /// \brief The angular spring rest angle element field pointer.
  stk::mesh::Field<double> *element_angular_spring_rest_angle_field_ptr_ = nullptr;

  /// \brief The sphere radius element field pointer.
  stk::mesh::Field<double> *element_sphere_radius_field_ptr_ = nullptr;

  /// \brief The spherocylinder segment radius element field pointer.
  stk::mesh::Field<double> *element_spherocylinder_segment_radius_field_ptr_ = nullptr;
  //@}
};  // ChainOfSprings

}  // namespace techniques

}  // namespace declare_and_initialize_constraints

}  // namespace constraints

}  // namespace mundy

#endif  // MUNDY_CONSTRAINTS_DECLARE_AND_INITIALIZE_CONSTRAINTS_TECHNIQUES_CHAINOFSPRINGS_HPP_
