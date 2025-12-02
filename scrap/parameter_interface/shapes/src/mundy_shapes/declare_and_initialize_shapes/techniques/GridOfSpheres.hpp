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

#ifndef MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_GRIDOFSPHERES_HPP_
#define MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_GRIDOFSPHERES_HPP_

/// \file GridOfSpheres.hpp
/// \brief Declaration of the GridOfSpheres class

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
#include <mundy_shapes/Spheres.hpp>  // for mundy::shapes::Spheres
#include <mundy_shapes/declare_and_initialize_shapes/techniques/GridCoordinateMapping.hpp>  // for mundy::shapes::...::GridCoordinateMapping

namespace mundy {

namespace shapes {

namespace declare_and_initialize_shapes {

namespace techniques {

/*
Our goal is to declare mundy::shapes::Shapes and initialize them in 3D space.

We have a bunch of functionality built into this initialization routine that can be turned on and off depending on your
needs. To start, by grid of spheres, we mean that if neighboring spheres were connected, they would be
homeomorphic to a structured grid of size $n_x \times n_y \times n_z$. We initialize the spheres with a uniform random
radius between 'sphere_radius_lower_bound' and 'sphere_radius_upper_bound'. Of course, if you want to initialize the
spheres with a fixed radius, you can set 'sphere_radius_lower_bound' and 'sphere_radius_upper_bound' to the same value.
By default, IDs are assigned to the spheres in a simple row-major ordering. If you want to use the Z-order Morton curve
to decide how to assign IDs to the spheres in a way that is more cache-friendly, you can set 'zmorton' to true.
Otherwise, if you want to shuffle the IDs of the spheres in a cache-unfriendly way, you can set 'shuffle' to true. This
final option is useful for testing purposes; please do not use it in production code.

If this method is run using multiple ranks, the grid will be distributed accross the ranks with i as the partitioned
index. That is, each rank will recieve n_x / num_ranks columns in the x direction with rank 0 recieving the first chunk
of columns, rank 1 the next, and so on. Because n_x / num_ranks may not be an integer, we will add one column to the
first n_x % num_ranks ranks. Importantly, sphere ID generation (linear sorting, znorm sorting, and shuffling) are
independent of the number of ranks.

Parameters:
  - 'sphere_part_names' (Array of strings): [Optional, Defaults to 'SPHERES'] The names of the parts to which we will
add the generated spheres.
  - 'num_spheres_x' (size_t): [Optional, Defaults to 1] The number of spheres in the x direction.
  - 'num_spheres_y' (size_t): [Optional, Defaults to 1] The number of spheres in the y direction.
  - 'num_spheres_z' (size_t): [Optional, Defaults to 1] The number of spheres in the z direction.
  - 'coordinate_map' (std::shared_ptr<GridCoordinateMapping<3,3>>): [Optional, Defaults to the identity map from R3 to
R3] The user-defined map function for the performing the map from grid index to sphere coordinates.
  - 'sphere_radius_lower_bound' (double): [Optional, Defaults to 0.0] The lower bound on the sphere radius.
  - 'sphere_radius_upper_bound' (double): [Optional, Defaults to 1.0] The upper bound on the sphere radius.
  - 'zmorton' (bool): [Optional, Defaults to true] If true, we use the Z-order Morton curve to decide how to assign IDs
      to the spheres in a way that is more cache-friendly. If false, we use a simple row-major ordering. Note, Z-Morton
      doesn't require a sort. Morton's function is extraordinarily beautiful and simply requires a bit manipulation to
      determine the unique sphere ID from the grid index (i,j,k). This makes its computation negligible.
  - 'shuffle' (bool): [Optional, Defaults to false] If true, we shuffle the order of the spheres post initialization.
*/
class GridOfSpheres : public mundy::meta::MetaMethodExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodExecutionInterface<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  GridOfSpheres() = delete;

  /// \brief Constructor
  GridOfSpheres(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    valid_fixed_params.validateParametersAndSetDefaults(GridOfSpheres::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    Teuchos::Array<std::string> sphere_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("sphere_part_names");

    for (int i = 0; i < sphere_part_names.size(); i++) {
      const std::string part_name = sphere_part_names[i];
      if (part_name == mundy::shapes::Spheres::get_name()) {
        // No specialization is required.
      } else {
        // The specialized part must be a subset of the spheres part.
        auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
        part_reqs->set_part_name(part_name);
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList().set("sphere_part_names", mundy::core::make_string_array(default_sphere_part_name_),
                                     "The names of the parts to which we will add the generated spheres.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("num_spheres_x", default_num_spheres_x_, "The number of spheres in the x direction.")
            .set("num_spheres_y", default_num_spheres_y_, "The number of spheres in the y direction.")
            .set("num_spheres_z", default_num_spheres_z_, "The number of spheres in the z direction.")
            .set<std::shared_ptr<GridCoordinateMapping>>("coordinate_mapping", std::make_shared<IdentityMap>(),
                                                         "The user-defined map function for the sphere coordinates.")
            .set("sphere_radius_lower_bound", default_sphere_radius_lower_bound_,
                 "The lower bound on the sphere radius.")
            .set("sphere_radius_upper_bound", default_sphere_radius_upper_bound_,
                 "The upper bound on the sphere radius.")
            .set("zmorton", default_zmorton_,
                 "If true, we use the Z-order Morton curve to decide how to assign IDs to the spheres "
                 "in a way that is more cache-friendly. If false, we use a simple row-major ordering.")
            .set("shuffle", default_shuffle_, "If true, we shuffle the order of the spheres post initialization.");

    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<GridOfSpheres>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get the ID of the node at index (i, j, k) in the grid.
  stk::mesh::EntityId node_id(size_t i, size_t j, size_t k) const;

  /// \brief Get the ID of the element at index (i, j, k) in the grid.
  stk::mesh::EntityId element_id(size_t i, size_t j, size_t k) const;

  /// \brief Get the node at index (i, j, k) in the grid.
  /// Returns an invalid entity if the node does not exist.
  stk::mesh::Entity node(size_t i, size_t j, size_t k) const;

  /// \brief Get the element at index (i, j, k) in the grid.
  /// Returns an invalid entity if the element does not exist.
  stk::mesh::Entity element(size_t i, size_t j, size_t k) const;
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

  static constexpr size_t default_num_spheres_x_ = 1;
  static constexpr size_t default_num_spheres_y_ = 1;
  static constexpr size_t default_num_spheres_z_ = 1;
  static constexpr double default_sphere_radius_lower_bound_ = 0.0;
  static constexpr double default_sphere_radius_upper_bound_ = 1.0;
  static constexpr bool default_zmorton_ = true;
  static constexpr bool default_shuffle_ = false;
  static constexpr std::string_view default_sphere_part_name_ = "SPHERES";
  //@}

  //! \name Internal methods
  //@{

  /// \brief Build the map from linearized index to sequential z-morton ordered index.
  void build_zmorton_map();
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of spheres in the x direction.
  size_t num_spheres_x_;

  /// \brief Number of spheres in the y direction.
  size_t num_spheres_y_;

  /// \brief Number of spheres in the z direction.
  size_t num_spheres_z_;

  /// \brief The user-defined map function for the sphere coordinates.
  std::shared_ptr<GridCoordinateMapping> coordinate_map_ptr_;

  /// \brief The map from linearized index to sequential z-morton ordered index.
  std::vector<size_t> zmorton_map_;

  /// \brief The lower bound on the sphere radius.
  double sphere_radius_lower_bound_;

  /// \brief The upper bound on the sphere radius.
  double sphere_radius_upper_bound_;

  /// \brief Z-Morton ID-assignment flag.
  bool zmorton_;

  /// \brief Shuffle flag.
  bool shuffle_;

  /// \brief Sphere element id start.
  size_t sphere_element_id_start_ = 1;

  /// \brief Sphere node id start.
  size_t sphere_node_id_start_ = 1;

  /// \brief The sphere parts.
  std::vector<stk::mesh::Part *> sphere_part_ptrs_;

  /// \brief The sphere node coordinate field pointer.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief The sphere radius field pointer.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // GridOfSpheres

}  // namespace techniques

}  // namespace declare_and_initialize_shapes

}  // namespace shapes

}  // namespace mundy

#endif  // MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_GRIDOFSPHERES_HPP_
