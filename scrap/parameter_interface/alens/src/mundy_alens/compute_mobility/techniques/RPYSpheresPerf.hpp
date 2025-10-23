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

#ifndef MUNDY_ALENS_COMPUTE_MOBILITY_RPYSPHERESPERF_HPP_
#define MUNDY_ALENS_COMPUTE_MOBILITY_RPYSPHERESPERF_HPP_

/// \file RPYSpheresPerfPerf.hpp
/// \brief Declaration of the ComputeMobility's RPYSpheresPerf technique.

/*
Notes on Periphery:

To interface with SkellySim's Periphery, we need to understand how some of its components work.
The LHS and RHS of the linear system associated with the periphery are computed using
LHS = periphery.matvec(x_periphery, v_periphery(x));

periphery.update_RHS(v_ext_periphery);
RHS = periphery.get_RHS();
 - x is the unknown force density on both the fibers and the periphery and is computed by the hydro operator.
 - x_periphery is the subset of x restricted to the periphery.
 - v_periphery(x) is the velocity on the periphery induced by x.
 - v_ext_periphery is the velocity on the periphery induced by external contributions, such as the external force/torque
on the rods.

v_periphery(x) = v_periphery(x_fibers)
v_fiber(x) = v_fiber(x_periphery) + v_fiber(x_fibers)

Notice, self-interaction v_periphery(x_periphery) is not included; this is because it is included by periphery.matvec.
Unlike the rods, the periphery may not have a surface slip velocity.


The local number of degrees of freedom in the LHS/RHS can be fetched using
periphery.get_local_solution_size();

The target coordinates on the periphery can be fetched using
periphery.get_local_node_positions();
*/

#include <MundyAlens_config.hpp>  // for HAVE_MUNDYALENS_*
#ifdef HAVE_MUNDYALENS_STKFMM

// External libs
#include <STKFMM/STKFMM.hpp>

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/MakeStringArray.hpp>  // for mundy::core::make_string_array
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>        // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>      // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>       // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>     // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartReqs.hpp>         // for mundy::meta::PartReqs
#include <mundy_shapes/Spheres.hpp>        // for mundy::shapes::Spheres

namespace mundy {

namespace alens {

namespace compute_mobility {

/// \class RPYSpheresPerf
/// \brief Concrete implementation of \c MetaKernel for computing the hydrodynamic interaction between spheres using the
/// RPY kernel within a periphery.
class RPYSpheresPerf : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit RPYSpheresPerf(mundy::mesh::BulkData *const bulk_data_ptr,
                          const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                        "RPYSpheresPerf: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(RPYSpheresPerf::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_REQUIRE(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                          std::string("RPYSpheresPerf: Part '") + part_name +
                              "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_coord_field_name = mundy::shapes::Spheres::get_node_coord_field_name();
    const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    const std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();

    node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
    node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
    node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name);
    element_radius_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  }
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshReqs
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(RPYSpheresPerf::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs<double>(node_force_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(node_velocity_field_name, stk::topology::NODE_RANK, 3, 1);

      if (part_name == mundy::shapes::Spheres::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static auto default_parameter_list =
        Teuchos::ParameterList()
            .set("valid_entity_part_names", mundy::core::make_string_array(default_part_name_),
                 "Name of the parts associated with this kernel.")
            .set("node_force_field_name", std::string(default_node_force_field_name_), "Name of the node force field.")
            .set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                 "Name of the node velocity field.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static auto default_parameter_list =
        Teuchos::ParameterList()
            .set("viscosity", default_viscosity_, "The fluid viscosity.")
            .set("fmm_multipole_order", default_fmm_multipole_order_, "The multipole order for the FMM.")
            .set("max_num_leaf_pts", default_max_num_leaf_pts_, "The maximum number of leaf points for the FMM.")
            .set("periodic_in_x", default_periodic_in_x_, "Whether the domain is periodic in the x direction.")
            .set("periodic_in_y", default_periodic_in_y_, "Whether the domain is periodic in the y direction.")
            .set("periodic_in_z", default_periodic_in_z_, "Whether the domain is periodic in the z direction.")
            .set<Teuchos::Array<double>>(
                "domain_origin_",
                Teuchos::tuple<double>(default_domain_origin_[0], default_domain_origin_[1], default_domain_origin_[2]),
                "The origin of the domain.")
            .set("domain_length_", default_domain_length_, "The length of the domain.");

    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<RPYSpheresPerf>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(RPYSpheresPerf::get_valid_mutable_params());
    viscosity_ = valid_mutable_params.get<double>("viscosity");
    fmm_multipole_order_ = valid_mutable_params.get<int>("fmm_multipole_order");
    max_num_leaf_pts_ = valid_mutable_params.get<int>("max_num_leaf_pts");
    periodic_in_x_ = valid_mutable_params.get<bool>("periodic_in_x");
    periodic_in_y_ = valid_mutable_params.get<bool>("periodic_in_y");
    periodic_in_z_ = valid_mutable_params.get<bool>("periodic_in_z");
    auto domain_origin_array = valid_mutable_params.get<Teuchos::Array<double>>("domain_origin_");
    domain_origin_ = {domain_origin_array[0], domain_origin_array[1], domain_origin_array[2]};
    domain_length_ = valid_mutable_params.get<double>("domain_length_");
    precomputed_periphery_file_ = valid_mutable_params.get<std::string>("precomputed_periphery_file");
    periphery_evaluator_ = valid_mutable_params.get<std::string>("periphery_evaluator");

    MUNDY_THROW_REQUIRE(viscosity_ > 0.0, std::invalid_argument,
                        "RPYSpheresPerf: viscosity must be greater than zero.");
    MUNDY_THROW_REQUIRE(fmm_multipole_order_ > 0, std::invalid_argument,
                        "RPYSpheresPerf: fmm_multipole_order must be greater than zero.");
    MUNDY_THROW_REQUIRE(max_num_leaf_pts_ > 0, std::invalid_argument,
                        "RPYSpheresPerf: max_num_leaf_pts must be greater than zero.");
    MUNDY_THROW_REQUIRE(domain_length_ > 0.0, std::invalid_argument,
                        "RPYSpheresPerf: domain_length_ must be greater than zero.");
    MUNDY_THROW_REQUIRE(precomputed_periphery_file_.size() > 0, std::invalid_argument,
                        "RPYSpheresPerf: precomputed_periphery_file must be a non-empty string.");
    MUNDY_THROW_REQUIRE(periphery_evaluator_ == "FMM" || periphery_evaluator_ == "DIRECT_CPU", std::invalid_argument,
                        "RPYSpheresPerf: precomputed_periphery_evaluator must be either 'FMM' or 'DIRECT_CPU'.");

    // Precompute the periphery
    periphery_ = std::make_shared<periphery_utils::Periphery>(precomputed_periphery_file_, periphery_evaluator_,
                                                              fmm_multipole_order_, max_num_leaf_pts_);
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Selector &sphere_selector) {
    std::cout << "RPYSpheresPerf::execute" << std::endl;

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    double viscosity = viscosity_;
    double Pi = 3.14159265358979323846;

    // Get the total number of local spheres
    stk::mesh::Selector intersection_with_valid_entity_parts =
        stk::mesh::selectUnion(valid_entity_parts_) & meta_data_ptr_->locally_owned_part() & sphere_selector;
    stk::mesh::EntityVector local_spheres;
    stk::mesh::get_selected_entities(intersection_with_valid_entity_parts,
                                     bulk_data.buckets(stk::topology::ELEMENT_RANK), local_spheres);
    const int num_local_spheres = local_spheres.size();

    // Setup our periodic boundary conditions
    stkfmm::PAXIS fmm_pbc;
    if (!periodic_in_x_ && !periodic_in_y_ && !periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::NONE;
    } else if (periodic_in_x_ && !periodic_in_y_ && !periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::PX;
    } else if (periodic_in_x_ && periodic_in_y_ && !periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::PXY;
    } else if (periodic_in_x_ && periodic_in_y_ && periodic_in_z_) {
      fmm_pbc = stkfmm::PAXIS::PXYZ;
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                          std::string("Unsupported pbc configuration. The current configuration is ") +
                              "periodic_in_x = " + periodic_in_x_ + ", periodic_in_y = " + periodic_in_y_ +
                              ", periodic_in_z = " + periodic_in_z_);
    }
    std::cout << "RPYSpheresPerf::execute pbc setup" << std::endl;

    // Initialize the FMM evaluator
    std::cout << "fmm_multipole_order: " << fmm_multipole_order_ << " | max_num_leaf_pts: " << max_num_leaf_pts_
              << std::endl;

    auto fmm_evaluator =
        stkfmm::Stk3DFMM(fmm_multipole_order_, max_num_leaf_pts_, fmm_pbc, stkfmm::asInteger(stkfmm::KERNEL::RPY));

    std::cout << "Domain origin: " << default_domain_origin_[0] << " " << default_domain_origin_[1] << " "
              << default_domain_origin_[2] << std::endl;
    std::cout << "Domain length: " << default_domain_length_ << std::endl;
    fmm_evaluator.setBox(domain_origin_.data(), domain_length_);
    std::cout << "fmm_evaluator initialized" << std::endl;

    // Setup the source and target points
    std::vector<double> src_single_layer_coord(3 * num_local_spheres);
    std::vector<double> trg_coord(3 * num_local_spheres);
    std::vector<double> src_single_layer_value(4 * num_local_spheres);
    std::vector<double> trg_value(6 * num_local_spheres, 0.0);

#pragma omp parallel for
    for (size_t i = 0; i < num_local_spheres; i++) {
      const auto &sphere = local_spheres[i];
      const auto &node = bulk_data.begin_nodes(sphere)[0];

      const double radius = stk::mesh::field_data(element_radius_field, sphere)[0];
      const double *node_coord = stk::mesh::field_data(node_coord_field, node);
      const double *node_force = stk::mesh::field_data(node_force_field, node);

      const bool coordinate_out_of_domain_in_x =
          periodic_in_x_
              ? false
              : ((node_coord[0] < domain_origin_[0]) || (node_coord[0] >= domain_origin_[0] + domain_length_));
      const bool coordinate_out_of_domain_in_y =
          periodic_in_y_
              ? false
              : ((node_coord[1] < domain_origin_[1]) || (node_coord[1] >= domain_origin_[1] + domain_length_));
      const bool coordinate_out_of_domain_in_z =
          periodic_in_z_
              ? false
              : ((node_coord[2] < domain_origin_[2]) || (node_coord[2] >= domain_origin_[2] + domain_length_));
      const bool coordinate_out_of_domain_in_non_periodic_direction =
          coordinate_out_of_domain_in_x || coordinate_out_of_domain_in_y || coordinate_out_of_domain_in_z;
      MUNDY_THROW_ASSERT(!coordinate_out_of_domain_in_non_periodic_direction, std::logic_error,
                         "RPYSpheresPerf: Node coordinate is out of domain. The current coordinate is "
                             << node_coord[0] << " " << node_coord[1] << " " << node_coord[2] << " and the origin is "
                             << domain_origin_[0] << " " << domain_origin_[1] << " " << domain_origin_[2]
                             << " with length " << domain_length_);

      src_single_layer_coord[3 * i] = node_coord[0];
      src_single_layer_coord[3 * i + 1] = node_coord[1];
      src_single_layer_coord[3 * i + 2] = node_coord[2];
      trg_coord[3 * i] = node_coord[0];
      trg_coord[3 * i + 1] = node_coord[1];
      trg_coord[3 * i + 2] = node_coord[2];

      src_single_layer_value[4 * i] = node_force[0];
      src_single_layer_value[4 * i + 1] = node_force[1];
      src_single_layer_value[4 * i + 2] = node_force[2];
      src_single_layer_value[4 * i + 3] = radius;
    }
    fmm_evaluator.setPoints(num_local_spheres, src_single_layer_coord.data(), num_local_spheres, trg_coord.data());
    fmm_evaluator.setupTree(stkfmm::KERNEL::RPY);
    std::cout << "fmm_evaluator tree setup" << std::endl;

    // Evaluate the FMM
    fmm_evaluator.clearFMM(stkfmm::KERNEL::RPY);
    fmm_evaluator.evaluateFMM(stkfmm::KERNEL::RPY, num_local_spheres, src_single_layer_value.data(), num_local_spheres,
                              trg_value.data());
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    std::cout << "fmm_evaluator evaluated" << std::endl;

    // Update the node velocities
#pragma omp parallel for
    for (size_t i = 0; i < num_local_spheres; i++) {
      const auto &sphere = local_spheres[i];
      const auto &node = bulk_data.begin_nodes(sphere)[0];

      const double radius = stk::mesh::field_data(element_radius_field, sphere)[0];
      const double *node_force = stk::mesh::field_data(node_force_field, node);
      double *node_velocity = stk::mesh::field_data(node_velocity_field, node);

      // Convert the target values to the external velocity
      const double rpyfac = radius * radius / 6.0;
      node_velocity[0] += (trg_value[6 * i + 0] + rpyfac * trg_value[6 * i + 3]) / viscosity;
      node_velocity[1] += (trg_value[6 * i + 1] + rpyfac * trg_value[6 * i + 4]) / viscosity;
      node_velocity[2] += (trg_value[6 * i + 2] + rpyfac * trg_value[6 * i + 5]) / viscosity;

      // Add on the self-mobility
      const double inv_drag_trans = 1.0 / (6 * Pi * radius * viscosity);
      node_velocity[0] += inv_drag_trans * node_force[0];
      node_velocity[1] += inv_drag_trans * node_force[1];
      node_velocity[2] += inv_drag_trans * node_force[2];
    }

    std::cout << "RPYSpheresPerf::execute done" << std::endl;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_viscosity_ = 1.0;
  static inline int default_fmm_multipole_order_ = 8;
  static inline int default_max_num_leaf_pts_ = 2000;
  static inline bool default_periodic_in_x_ = false;
  static inline bool default_periodic_in_y_ = false;
  static inline bool default_periodic_in_z_ = false;
  static inline std::array<double, 3> default_domain_origin_ = {0.0, 0.0, 0.0};
  static inline double default_domain_length_ = 1.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The fluid viscosity.
  double viscosity_;

  /// \brief The order of the multipole expansion used by PVFMM
  int fmm_multipole_order_;

  /// \brief The maximum number of leaf points per octree node used by PVFMM
  int max_num_leaf_pts_;

  /// \brief Are we periodic in x, y, and z?
  bool periodic_in_x_ = false;
  bool periodic_in_y_ = false;
  bool periodic_in_z_ = false;

  /// \brief Bottom left corner of the domain
  std::array<double, 3> domain_origin_;

  /// \brief The length of the domain
  double domain_length_;

  /// \brief The precomputed periphery file name/location
  std::string precomputed_periphery_file_;

  /// \brief The periphery evaluator type
  std::string periphery_evaluator_;

  /// \brief The periphery object
  std::shared_ptr<Periphery> periphery_ptr_ = nullptr;

  /// \brief Node coord containing the node's position.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's force.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Element field containing the sphere's radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // RPYSpheresPerf

}  // namespace compute_mobility

}  // namespace alens

}  // namespace mundy

#endif  //  HAVE_MUNDYALENS_STKFMM

#endif  // MUNDY_ALENS_COMPUTE_MOBILITY_RPYSPHERESPERF_HPP_
