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

// External libs
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <openrand/philox.h>

// C++ core
#include <algorithm>   // for std::transform
#include <filesystem>  // for std::filesystem::path
#include <fstream>     // for std::ofstream
#include <iostream>    // for std::cout, std::endl
#include <memory>      // for std::shared_ptr, std::unique_ptr
#include <numeric>     // for std::accumulate
#include <regex>       // for std::regex
#include <string>      // for std::string
#include <vector>      // for std::vector

// Trilinos libs
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Comm.hpp>            // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
#include <stk_mesh/base/FieldParallel.hpp>   // for stk::parallel_sum
#include <stk_mesh/base/ForEachEntity.hpp>   // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>            // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>        // for stk::mesh::Selector
#include <stk_topology/topology.hpp>         // for stk::topology
#include <stk_util/parallel/Parallel.hpp>    // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_alens/actions_crosslinkers.hpp>                // for mundy::alens::crosslinkers...
#include <mundy_alens/periphery/Periphery.hpp>                 // for gen_sphere_quadrature
#include <mundy_constraints/AngularSprings.hpp>                // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>      // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/DeclareAndInitConstraints.hpp>     // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>                // for mundy::constraints::HookeanSprings
#include <mundy_core/MakeStringArray.hpp>                      // for mundy::core::make_string_array
#include <mundy_core/OurAnyNumberParameterEntryValidator.hpp>  // for mundy::core::OurAnyNumberParameterEntryValidator
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_io/IOBroker.hpp>         // for mundy::io::IOBroker
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>         // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>       // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>        // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                // for mundy::linkers::NeighborLinkers
#include <mundy_math/Hilbert.hpp>                           // for mundy::math::create_hilbert_positions_and_directors
#include <mundy_math/Vector3.hpp>                           // for mundy::math::Vector3
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>       // for mundy::math::distance::ellipsoid_ellipsoid
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>  // adds fmt::format for stk types
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>        // for mundy::mesh::utils::destroy_flagged_entities
#include <mundy_mesh/utils/FillFieldWithValue.hpp>            // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

namespace mundy {

namespace geom {

// struct SphereFields {
//   stk::mesh::Field<double> &node_coords;
//   stk::mesh::Field<double> &elem_radius;
// };

// struct SpherocylinderSegmentFields {
//   stk::mesh::Field<double> &node_coords;
//   stk::mesh::Field<double> &elem_radius;
// };

// struct HookeanSpringFields {
//   stk::mesh::Field<double> &node_coords;
//   stk::mesh::Field<double> &elem_spring_constant;
//   stk::mesh::Field<double> &elem_rest_length;
// };

// struct FeneSpringFields {
//   stk::mesh::Field<double> &node_coords;
//   stk::mesh::Field<double> &elem_spring_constant;
//   stk::mesh::Field<double> &elem_rest_length;
// };

// struct DynamicSpringFields {
//   stk::mesh::Field<double> &elem_binding_rates;
//   stk::mesh::Field<double> &elem_unbinding_rates;
//   stk::mesh::Field<size_t> &elem_rng_counter;
// };

// struct NgpSphereFields {
//   stk::mesh::NgpField<double> node_coords;
//   stk::mesh::NgpField<double> elem_radius;
// };

// struct NgpSpherocylinderSegmentFields {
//   stk::mesh::NgpField<double> node_coords;
//   stk::mesh::NgpField<double> elem_radius;
// };

// struct NgpHookeanSpringFields {
//   stk::mesh::NgpField<double> node_coords;
//   stk::mesh::NgpField<double> elem_spring_constant;
//   stk::mesh::NgpField<double> elem_rest_length;
// };

// struct NgpFeneSpringFields {
//   stk::mesh::NgpField<double> node_coords;
//   stk::mesh::NgpField<double> elem_spring_constant;
//   stk::mesh::NgpField<double> elem_rest_length;
// };

// struct NgpDynamicSpringFields {
//   stk::mesh::NgpField<double> elem_binding_rates;
//   stk::mesh::NgpField<double> elem_unbinding_rates;
//   stk::mesh::NgpField<size_t> elem_rng_counter;
// };

// NgpSphereFields get_updated_ngp_fields(SphereFields &fields) {
//   return {stk::mesh::get_updated_ngp_field<double>(fields.node_coords),
//           stk::mesh::get_updated_ngp_field<double>(fields.elem_radius)};
// }

// NgpSpherocylinderSegmentFields get_updated_ngp_fields(SpherocylinderSegmentFields &fields) {
//   return {stk::mesh::get_updated_ngp_field<double>(fields.node_coords),
//           stk::mesh::get_updated_ngp_field<double>(fields.elem_radius)};
// }

// NgpHookeanSpringFields get_updated_ngp_fields(HookeanSpringFields &fields) {
//   return {stk::mesh::get_updated_ngp_field<double>(fields.node_coords),
//           stk::mesh::get_updated_ngp_field<double>(fields.elem_spring_constant),
//           stk::mesh::get_updated_ngp_field<double>(fields.elem_rest_length)};
// }

// NgpFeneSpringFields get_updated_ngp_fields(FeneSpringFields &fields) {
//   return {stk::mesh::get_updated_ngp_field<double>(fields.node_coords),
//           stk::mesh::get_updated_ngp_field<double>(fields.elem_spring_constant),
//           stk::mesh::get_updated_ngp_field<double>(fields.elem_rest_length)};
// }

// NgpDynamicSpringFields get_updated_ngp_fields(DynamicSpringFields &fields) {
//   return {stk::mesh::get_updated_ngp_field<double>(fields.elem_binding_rates),
//           stk::mesh::get_updated_ngp_field<double>(fields.elem_unbinding_rates),
//           stk::mesh::get_updated_ngp_field<size_t>(fields.elem_rng_counter)};
// }

void compute_aabb_spheres(stk::mesh::NgpMesh &ngp_mesh, const double &skin_distance, stk::mesh::NgpField<double> &node_coords,
                          stk::mesh::NgpField<double> &elem_radius, stk::mesh::NgpField<double> &elem_aabb_field,
                          const stk::mesh::Selector &selector) {
  node_coords.sync_to_device();
  elem_radius.sync_to_device();
  elem_aabb_field.sync_to_device();

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &elem_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, elem_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);

        const auto coords = mundy::mesh::vector3_field_data(node_coords, node_index);
        const double radius = elem_radius(elem, 0);
        double min_x = coords[0] - radius;
        double min_y = coords[1] - radius;
        double min_z = coords[2] - radius;
        double max_x = coords[0] + radius;
        double max_y = coords[1] + radius;
        double max_z = coords[2] + radius;
        elem_aabb_field(elem, 0) = min_x - skin_distance;
        elem_aabb_field(elem, 1) = min_y - skin_distance;
        elem_aabb_field(elem, 2) = min_z - skin_distance;
        elem_aabb_field(elem, 3) = max_x + skin_distance;
        elem_aabb_field(elem, 4) = max_y + skin_distance;
        elem_aabb_field(elem, 5) = max_z + skin_distance;
      });

  elem_aabb_field.modify_on_device();
}

void compute_aabb_segs(stk::mesh::NgpMesh &ngp_mesh, const double &skin_distance, stk::mesh::NgpField<double> &node_coords,
                       stk::mesh::NgpField<double> &elem_radius,
                  stk::mesh::NgpField<double> &elem_aabb_field, const stk::mesh::Selector &selector) {
  node_coords.sync_to_device();
  elem_radius.sync_to_device();
  elem_aabb_field.sync_to_device();

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &elem_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, elem_index);
        stk::mesh::FastMeshIndex node0_index = ngp_mesh.fast_mesh_index(nodes[0]);
        stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(nodes[1]);

        const auto coord0 = mundy::mesh::vector3_field_data(node_coords, node0_index);
        const auto coord1 = mundy::mesh::vector3_field_data(node_coords, node1_index);

        const double radius = elem_radius(elem, 0);
        double min_x = Kokkos::min(coord0[0], coord1[0]) - radius;
        double min_y = Kokkos::min(coord0[1], coord1[1]) - radius;
        double min_z = Kokkos::min(coord0[2], coord1[2]) - radius;
        double max_x = Kokkos::max(coord0[0], coord1[0]) + radius;
        double max_y = Kokkos::max(coord0[1], coord1[1]) + radius;
        double max_z = Kokkos::max(coord0[2], coord1[2]) + radius;
        elem_aabb_field(elem, 0) = min_x - skin_distance;
        elem_aabb_field(elem, 1) = min_y - skin_distance;
        elem_aabb_field(elem, 2) = min_z - skin_distance;
        elem_aabb_field(elem, 3) = max_x + skin_distance;
        elem_aabb_field(elem, 4) = max_y + skin_distance;
        elem_aabb_field(elem, 5) = max_z + skin_distance;
      });

  elem_aabb_field.modify_on_device();
}

void accumulate_aabb_displacements(stk::mesh::NgpMesh &ngp_mesh,
                                   stk::mesh::NgpField<double> &old_elem_aabb_field,
                                   stk::mesh::NgpField<double> &elem_aabb_field,
                                   stk::mesh::NgpField<double> &elem_displacement_field,
                                   const stk::mesh::Selector &selector) {
  old_elem_aabb_field.sync_to_device();
  elem_aabb_field.sync_to_device();
  elem_displacement_field.sync_to_device();

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &elem_index) {
        for (int i = 0; i < 6; ++i) {
          elem_displacement_field(elem, i) += elem_aabb_field(elem, i) - old_elem_aabb_field(elem, i);
        }
      });

  elem_displacement_field.modify_on_device();
}

}  // namespace geom

}  // namespace mundy

namespace mundy {

namespace alens {

namespace hp1 {

// template <typename SpringFieldType, typename DynamicSpringFieldType>
// struct Hp1Fields {
//   using spring_fields_t = SpringFieldType;
//   using dynamic_spring_fields_t = DynamicSpringFieldType;

//   SpringFieldType &spring_fields;
//   DynamicSpringFieldType &dynamic_spring_fields;
// };

// template <typename NgpSpringFieldType, typename NgpDynamicSpringFieldType>
// struct NgpHp1Fields {
//   using ngp_spring_fields_t = NgpSpringFieldType;
//   using ngp_dynamic_spring_fields_t = NgpDynamicSpringFieldType;

//   NgpSpringFieldType spring_fields;
//   NgpDynamicSpringFieldType dynamic_spring_fields;
// };

// template <typename SpringFieldType, typename DynamicSpringFieldType>
// auto get_updated_ngp_fields(Hp1Fields<SpringFieldType, DynamicSpringFieldType> &fields) {
//   return NgpHp1Fields{geom::get_updated_ngp_fields(fields.spring_fields),
//                       geom::get_updated_ngp_fields(fields.dynamic_spring_fields)};
// }

// template <typename ShapeFieldsType, typename MotilityFieldsType, typename CollisionFieldsType,
//           typename SpringFieldsType, typename NeighborSearchFieldsType>
// struct BackboneSegments {
//   using shape_fields_t = ShapeFieldsType;
//   using motility_fields_t = MotilityFieldsType;
//   using collision_fields_t = CollisionFieldsType;
//   using spring_fields_t = SpringFieldsType;
//   using neighbor_search_fields_t = NeighborSearchFieldsType;

//   ShapeFieldsType &shape_fields;
//   MotilityFieldsType &motility_fields;
//   CollisionFieldsType &collision_fields;
//   SpringFieldsType &spring_fields;
//   NeighborSearchFieldsType &neighbor_search_fields;
// };

// template <typename NgpShapeFieldsType, typename NgpMotilityFieldsType, typename NgpCollisionFieldsType,
//           typename NgpSpringFieldsType, typename NgpNeighborSearchFieldsType>
// struct NgpBackboneSegments {
//   using ngp_shape_fields_t = NgpShapeFieldsType;
//   using ngp_motility_fields_t = NgpMotilityFieldsType;
//   using ngp_collision_fields_t = NgpCollisionFieldsType;
//   using ngp_spring_fields_t = NgpSpringFieldsType;
//   using ngp_neighbor_search_fields_t = NgpNeighborSearchFieldsType;

//   NgpShapeFieldsType shape_fields;
//   NgpMotilityFieldsType motility_fields;
//   NgpCollisionFieldsType collision_fields;
//   NgpSpringFieldsType spring_fields;
//   NgpNeighborSearchFieldsType neighbor_search_fields;
// };

// template <typename ShapeFieldsType, typename MotilityFieldsType, typename CollisionFieldsType,
//           typename SpringFieldsType, typename NeighborSearchFieldsType>
// auto get_updated_ngp_fields(BackboneSegments<ShapeFieldsType, MotilityFieldsType, CollisionFieldsType, SpringFieldsType,
//                                              NeighborSearchFieldsType> &fields) {
//   return NgpBackboneSegments{
//       geom::get_updated_ngp_fields(fields.shape_fields), geom::get_updated_ngp_fields(fields.motility_fields),
//       geom::get_updated_ngp_fields(fields.collision_fields), geom::get_updated_ngp_fields(fields.spring_fields),
//       geom::get_updated_ngp_fields(fields.neighbor_search_fields)};
// }

enum class BINDING_STATE_CHANGE : unsigned {
  NONE = 0u,
  LEFT_TO_DOUBLY,
  RIGHT_TO_DOUBLY,
  DOUBLY_TO_LEFT,
  DOUBLY_TO_RIGHT
};
enum class INITIALIZATION_TYPE : unsigned {
  GRID = 0u,
  RANDOM_UNIT_CELL,
  OVERLAP_TEST,
  HILBERT_RANDOM_UNIT_CELL,
  USHAPE_TEST,
  FROM_EXO,
  FROM_DAT
};

enum class BOND_TYPE : unsigned { HARMONIC = 0u, FENE };
enum class PERIPHERY_BIND_SITES_TYPE : unsigned { RANDOM = 0u, FROM_FILE };
enum class PERIPHERY_SHAPE : unsigned { SPHERE = 0u, ELLIPSOID };
enum class PERIPHERY_QUADRATURE : unsigned { GAUSS_LEGENDRE = 0u, FROM_FILE };

// Cast from string to the above enaums via the = operator
BINDING_STATE_CHANGE &operator=(BINDING_STATE_CHANGE &state, const std::string &str) {
  if (str == "NONE") {
    state = BINDING_STATE_CHANGE::NONE;
  } else if (str == "LEFT_TO_DOUBLY") {
    state = BINDING_STATE_CHANGE::LEFT_TO_DOUBLY;
  } else if (str == "RIGHT_TO_DOUBLY") {
    state = BINDING_STATE_CHANGE::RIGHT_TO_DOUBLY;
  } else if (str == "DOUBLY_TO_LEFT") {
    state = BINDING_STATE_CHANGE::DOUBLY_TO_LEFT;
  } else if (str == "DOUBLY_TO_RIGHT") {
    state = BINDING_STATE_CHANGE::DOUBLY_TO_RIGHT;
  } else {
    MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unknown binding state change: " + str);
  }
  return state;
}

INITIALIZATION_TYPE &operator=(INITIALIZATION_TYPE &init_type, const std::string &str) {
  if (str == "GRID") {
    init_type = INITIALIZATION_TYPE::GRID;
  } else if (str == "RANDOM_UNIT_CELL") {
    init_type = INITIALIZATION_TYPE::RANDOM_UNIT_CELL;
  } else if (str == "OVERLAP_TEST") {
    init_type = INITIALIZATION_TYPE::OVERLAP_TEST;
  } else if (str == "HILBERT_RANDOM_UNIT_CELL") {
    init_type = INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL;
  } else if (str == "USHAPE_TEST") {
    init_type = INITIALIZATION_TYPE::USHAPE_TEST;
  } else if (str == "FROM_EXO") {
    init_type = INITIALIZATION_TYPE::FROM_EXO;
  } else if (str == "FROM_DAT") {
    init_type = INITIALIZATION_TYPE::FROM_DAT;
  } else {
    MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unknown initialization type: " + str);
  }
  return init_type;
}

BOND_TYPE &operator=(BOND_TYPE &bond_type, const std::string &str) {
  if (str == "HARMONIC") {
    bond_type = BOND_TYPE::HARMONIC;
  } else if (str == "FENE") {
    bond_type = BOND_TYPE::FENE;
  } else {
    MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unknown bond type: " + str);
  }
  return bond_type;
}

PERIPHERY_BIND_SITES_TYPE &operator=(PERIPHERY_BIND_SITES_TYPE &bind_sites_type, const std::string &str) {
  if (str == "RANDOM") {
    bind_sites_type = PERIPHERY_BIND_SITES_TYPE::RANDOM;
  } else if (str == "FROM_FILE") {
    bind_sites_type = PERIPHERY_BIND_SITES_TYPE::FROM_FILE;
  } else {
    MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unknown periphery bind sites type: " + str);
  }
  return bind_sites_type;
}

PERIPHERY_SHAPE &operator=(PERIPHERY_SHAPE &periphery_shape, const std::string &str) {
  if (str == "SPHERE") {
    periphery_shape = PERIPHERY_SHAPE::SPHERE;
  } else if (str == "ELLIPSOID") {
    periphery_shape = PERIPHERY_SHAPE::ELLIPSOID;
  } else {
    MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unknown periphery shape: " + str);
  }
  return periphery_shape;
}

PERIPHERY_QUADRATURE &operator=(PERIPHERY_QUADRATURE &periphery_quadrature, const std::string &str) {
  if (str == "GAUSS_LEGENDRE") {
    periphery_quadrature = PERIPHERY_QUADRATURE::GAUSS_LEGENDRE;
  } else if (str == "FROM_FILE") {
    periphery_quadrature = PERIPHERY_QUADRATURE::FROM_FILE;
  } else {
    MUNDY_THROW_ASSERT(false, std::invalid_argument, "Unknown periphery quadrature: " + str);
  }
  return periphery_quadrature;
}

std::ostream &operator<<(std::ostream &os, const BINDING_STATE_CHANGE &state) {
  switch (state) {
    case BINDING_STATE_CHANGE::NONE:
      os << "NONE";
      break;
    case BINDING_STATE_CHANGE::LEFT_TO_DOUBLY:
      os << "LEFT_TO_DOUBLY";
      break;
    case BINDING_STATE_CHANGE::RIGHT_TO_DOUBLY:
      os << "RIGHT_TO_DOUBLY";
      break;
    case BINDING_STATE_CHANGE::DOUBLY_TO_LEFT:
      os << "DOUBLY_TO_LEFT";
      break;
    case BINDING_STATE_CHANGE::DOUBLY_TO_RIGHT:
      os << "DOUBLY_TO_RIGHT";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const INITIALIZATION_TYPE &init_type) {
  switch (init_type) {
    case INITIALIZATION_TYPE::GRID:
      os << "GRID";
      break;
    case INITIALIZATION_TYPE::RANDOM_UNIT_CELL:
      os << "RANDOM_UNIT_CELL";
      break;
    case INITIALIZATION_TYPE::OVERLAP_TEST:
      os << "OVERLAP_TEST";
      break;
    case INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL:
      os << "HILBERT_RANDOM_UNIT_CELL";
      break;
    case INITIALIZATION_TYPE::USHAPE_TEST:
      os << "USHAPE_TEST";
      break;
    case INITIALIZATION_TYPE::FROM_EXO:
      os << "FROM_EXO";
      break;
    case INITIALIZATION_TYPE::FROM_DAT:
      os << "FROM_DAT";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const BOND_TYPE &bond_type) {
  switch (bond_type) {
    case BOND_TYPE::HARMONIC:
      os << "HARMONIC";
      break;
    case BOND_TYPE::FENE:
      os << "FENE";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const PERIPHERY_BIND_SITES_TYPE &bind_sites_type) {
  switch (bind_sites_type) {
    case PERIPHERY_BIND_SITES_TYPE::RANDOM:
      os << "RANDOM";
      break;
    case PERIPHERY_BIND_SITES_TYPE::FROM_FILE:
      os << "FROM_FILE";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const PERIPHERY_SHAPE &periphery_shape) {
  switch (periphery_shape) {
    case PERIPHERY_SHAPE::SPHERE:
      os << "SPHERE";
      break;
    case PERIPHERY_SHAPE::ELLIPSOID:
      os << "ELLIPSOID";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const PERIPHERY_QUADRATURE &periphery_quadrature) {
  switch (periphery_quadrature) {
    case PERIPHERY_QUADRATURE::GAUSS_LEGENDRE:
      os << "GAUSS_LEGENDRE";
      break;
    case PERIPHERY_QUADRATURE::FROM_FILE:
      os << "FROM_FILE";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

}  // namespace hp1

}  // namespace alens

}  // namespace mundy

template <>
struct fmt::formatter<mundy::alens::hp1::BINDING_STATE_CHANGE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::INITIALIZATION_TYPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::BOND_TYPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::PERIPHERY_BIND_SITES_TYPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::PERIPHERY_SHAPE> : fmt::ostream_formatter {};

template <>
struct fmt::formatter<mundy::alens::hp1::PERIPHERY_QUADRATURE> : fmt::ostream_formatter {};

namespace mundy {

namespace alens {

namespace hp1 {

using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

void print_rank0(auto thing_to_print, int indent_level = 0) {
  if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
    std::string indent(indent_level * 2, ' ');
    std::cout << indent << thing_to_print << std::endl;
  }
}

template <typename FieldDataType, size_t field_size>
void print_field(const stk::mesh::Field<FieldDataType> &field) {
  stk::mesh::BulkData &bulk_data = field.get_mesh();
  stk::mesh::Selector selector = stk::mesh::Selector(field);

  stk::mesh::EntityVector entities;
  stk::mesh::get_selected_entities(selector, bulk_data_ptr_->buckets(field.entity_rank()), entities);

  for (const stk::mesh::Entity &entity : entities) {
    const FieldDataType *field_data = stk::mesh::field_data(field, entity);
    std::cout << "Entity " << bulk_data.identifier(entity) << " field data: ";
    for (size_t i = 0; i < field_size; ++i) {
      std::cout << field_data[i] << " ";
    }
    std::cout << std::endl;
  }
}

class RcbSettings : public stk::balance::BalanceSettings {
 public:
  RcbSettings() {
  }
  virtual ~RcbSettings() {
  }

  virtual bool isIncrementalRebalance() const {
    return false;
  }
  virtual std::string getDecompMethod() const {
    return std::string("rcb");
  }
  virtual std::string getCoordinateFieldName() const {
    return std::string("NODE_COORDS");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // RcbSettings

void setup_mundy_io() {
  // IO fixed parameters
  auto fixed_params_iobroker =
      Teuchos::ParameterList()
          .set("enabled_io_parts", mundy::core::make_string_array("E", "H", "BS", "EESPRINGS", "EHSPRINGS", "HHSPRINGS",
                                                                  "LEFT_HP1", "DOUBLY_HP1_H", "DOUBLY_HP1_BS"))
          .set("enabled_io_fields_node_rank",
               mundy::core::make_string_array("NODE_VELOCITY", "NODE_FORCE", "NODE_RNG_COUNTER"))
          .set("enabled_io_fields_element_rank",
               mundy::core::make_string_array("ELEMENT_RADIUS", "ELEMENT_RNG_COUNTER", "ELEMENT_REALIZED_BINDING_RATES",
                                              "ELEMENT_REALIZED_UNBINDING_RATES", "ELEMENT_PERFORM_STATE_CHANGE",
                                              "EUCHROMATIN_STATE", "EUCHROMATIN_STATE_CHANGE_NEXT_TIME",
                                              "EUCHROMATIN_STATE_CHANGE_ELAPSED_TIME", "ELEMENT_CHAINID"))
          .set("coordinate_field_name", "NODE_COORDS")
          .set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES")
          .set("exodus_database_output_filename", output_filename_)
          .set("parallel_io_mode", "hdf5")
          .set("database_purpose", "restart");

  // Check if we are continuing a previous sim or if we are initializing from a file.
  // In either case, we must perform a restart.
  if (enable_continuation_if_available_) {
    // The filename pattern is stem + ".e-s." + timestep_number
    // We want to determine if if such a file exist, and if so, the file with the largest timestep number
    auto find_file_with_largest_timestep_number = [](const std::string &stem) {
      // Pattern to match files like stem.e-s.*
      std::string pattern = stem + R"(\.e-s\.(\d+))";
      std::regex regex_pattern(pattern);
      int largest_number = -1;
      std::string largest_file;

      // Iterate through the directory of the stem (or the current directory if the stem doesn't provide a filepath)
      std::filesystem::path filepath(stem);
      if (!std::filesystem::exists(filepath)) {
        filepath = std::filesystem::current_path();
      }
      for (const auto &entry : std::filesystem::directory_iterator(filepath)) {
        std::string filename = entry.path().filename().string();
        std::smatch match;
        if (std::regex_match(filename, match, regex_pattern)) {
          int number = std::stoi(match[1].str());
          if (number > largest_number) {
            largest_number = number;
            largest_file = entry.path().string();
          }
        }
      }

      const double file_found = largest_number != -1;
      return std::make_tuple(file_found, largest_file, largest_number);
    };

    auto [file_found, restart_filename, largest_number] = find_file_with_largest_timestep_number(output_filename_);
    if (file_found) {
      std::cout << "Restarting from file: " << restart_filename << " at step " << largest_number << std::endl;
      fixed_params_iobroker.set("exodus_database_input_filename", restart_filename);
      fixed_params_iobroker.set("enable_restart", "true");
      restart_performed_ = true;
      restart_timestep_index_ = largest_number;
    }
  }

  // Continuing a previous simulation takes priority over initializing from a file.
  // Initialization should have already been performed in the previous simulation.
  if (initialization_type_ == INITIALIZATION_TYPE::FROM_EXO && !restart_performed_) {
    fixed_params_iobroker.set("exodus_database_input_filename", initialize_from_exo_filename_);
    fixed_params_iobroker.set("enable_restart", "false");
  }

  io_broker_ptr_ = mundy::io::IOBroker::create_new_instance(bulk_data_ptr_.get(), fixed_params_iobroker);
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_from_file(
    const std::string &file_name, const unsigned num_chromosomes) {
  // The file should be formatted as follows:
  // chromosome_id x y z
  // 0 x1 y1 z1
  // 0 x2 y2 z2
  // ...
  // 1 x1 y1 z1
  // 1 x2 y2 z2
  //
  // chromosome_id should start at 1 and increase by 1 for each new chromosome.
  //
  // And so on for each chromosome. The total number of chromosomes should match the expected total, lest we throw an
  // exception.
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes)
      std::ifstream infile(file_name);
  MUNDY_THROW_REQUIRE(infile.is_open(), std::invalid_argument, fmt::format("Could not open file {}", file_name));

  // Read each line. While the chromosome_id is the same, keep adding nodes to the chromosome.
  size_t current_chromosome_id = 1;
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    int chromosome_id;
    double x, y, z;
    if (!(iss >> chromosome_id >> x >> y >> z)) {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Could not parse line " << line);
    }
    if (chromosome_id != current_chromosome_id) {
      // We are starting a new chromosome
      MUNDY_THROW_REQUIRE(chromosome_id == current_chromosome_id + 1, std::invalid_argument,
                          "Chromosome IDs should be sequential.");
      MUNDY_THROW_REQUIRE(chromosome_id <= num_chromosomes_, std::invalid_argument,
                          fmt::format("Chromosome ID {} is greater than the number of chromosomes.", chromosome_id));
      current_chromosome_id = chromosome_id;
    }
    // Add the node to the chromosome
    all_chromosome_positions[current_chromosome_id - 1].emplace_back(x, y, z);
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_grid(
    const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  const mundy::math::Vector3d alignment_dir{0.0, 0.0, 1.0};
  for (size_t j = 0; j < num_chromosomes; j++) {
    all_chromosome_positions[j].reserve(num_nodes_per_chromosome);
    openrand::Philox rng(j, 0);
    mundy::math::Vector3d start_pos(2.0 * static_cast<double>(j), 0.0, 0.0);
    for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
      const auto pos = start_pos + static_cast<double>(i) * segment_length * alignment_dir;
      all_chromosome_positions[j].emplace_back(pos);
    }
  }
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_random_unit_cell(
    const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length,
    const double domain_low[3], const double domain_high[3]) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  for (size_t j = 0; j < num_chromosomes; j++) {
    all_chromosome_positions[j].reserve(num_nodes_per_chromosome);

    // Find a random place within the unit cell with a random orientation for the chain.
    openrand::Philox rng(j, 0);
    mundy::math::Vector3d r_start {
      rng.uniform<double>(domain_low[0], domain_high[0]), rng.uniform<double>(domain_low[1], domain_high[1]),
          rng.uniform<double>(domain_low[2], domain_high[2])
    }

    // Find a random unit vector direction
    const double zrand = rng.rand<double>() - 1.0;
    const double wrand = std::sqrt(1.0 - zrand * zrand);
    const double trand = 2.0 * M_PI * rng.rand<double>();
    mundy::math::Vector3d u_hat{wrand * std::cos(trand), wrand * std::sin(trand), zrand};

    for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
      auto pos = pos_start + static_cast<double>(i) * segment_length * u_hat;
      all_chromosome_positions[j].emplace_back(pos);
    }
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_hilbert_random_unit_cell(
    const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length,
    const double domain_low[3], const double domain_high[3]) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  std::vector<mundy::geom::Point<double>> chromosome_centers_array(num_chromosomes);
  std::vector<double> chromosome_radii_array(num_chromosomes);
  for (size_t ichromosome = 0; ichromosome < num_chromosomes_; ichromosome++) {
    // Figure out which nodes we are doing
    const size_t num_heterochromatin_spheres = num_chromatin_repeats_ / 2 * num_heterochromatin_per_repeat_ +
                                               num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat_;
    const size_t num_euchromatin_spheres = num_chromatin_repeats_ / 2 * num_euchromatin_per_repeat_;
    const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
    size_t start_node_index = num_nodes_per_chromosome * ichromosome + 1u;
    size_t end_node_index = num_nodes_per_chromosome * (ichromosome + 1) + 1u;

    // Generate a random unit vector (will be used for creating the locatino of the nodes, the random position in
    // the unit cell will be handled later).
    openrand::Philox rng(ichromosome, 0);
    const double zrand = rng.rand<double>() - 1.0;
    const double wrand = std::sqrt(1.0 - zrand * zrand);
    const double trand = 2.0 * M_PI * rng.rand<double>();
    mundy::math::Vector3d u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

    // Once we have the number of chromosome spheres we can get the hilbert curve set up. This will be at some
    // orientation and then have sides with a length of initial_chromosome_separation.
    auto [hilbert_position_array, hilbert_directors] = mundy::math::create_hilbert_positions_and_directors(
        num_nodes_per_chromosome, u_hat, initial_chromosome_separation_);

    // Create the local positions of the spheres
    std::vector<mundy::math::Vector3d> sphere_position_array;
    for (size_t isphere = 0; isphere < num_nodes_per_chromosome; isphere++) {
      sphere_position_array.push_back(hilbert_position_array[isphere]);
    }

    // Figure out where the center of the chromosome is, and its radius, in its own local space
    mundy::math::Vector3d r_chromosome_center_local(0.0, 0.0, 0.0);
    double r_max = 0.0;
    for (size_t i = 0; i < sphere_position_array.size(); i++) {
      r_chromosome_center_local += sphere_position_array[i];
    }
    r_chromosome_center_local /= static_cast<double>(sphere_position_array.size());
    for (size_t i = 0; i < sphere_position_array.size(); i++) {
      r_max = std::max(r_max, mundy::math::two_norm(r_chromosome_center_local - sphere_position_array[i]));
    }

    // Do max_trials number of insertion attempts to get a random position and orientation within the unit cell that
    // doesn't overlap with exiting chromosomes.
    const size_t max_trials = 1000;
    size_t itrial = 0;
    bool chromosome_inserted = false;
    while (itrial <= max_trials) {
      // Generate a random position within the unit cell.
      mundy::math::Vector3d r_start(rng.uniform<double>(domain_low[0], domain_high[0]),
                                           rng.uniform<double>(domain_low[1], domain_high[1]),
                                           rng.uniform<double>(domain_low[2], domain_high[2]));

      // Check for overlaps with existing chromosomes
      bool found_overlap = false;
      for (size_t jchromosome = 0; jchromosome < chromosome_centers_array.size(); ++jchromosome) {
        double r_chromosome_distance = mundy::math::two_norm(chromosome_centers_array[jchromosome] - r_start);
        if (r_chromosome_distance < (r_max + chromosome_radii_array[jchromosome])) {
          found_overlap = true;
          break;
        }
      }
      if (found_overlap) {
        itrial++;
      } else {
        chromosome_inserted = true;
        chromosome_centers_array[ichromosome] = r_start;
        chromosome_radii_array[ichromosome] = r_max;
        break;
      }
    }
    MUNDY_THROW_REQUIRE(chromosome_inserted, std::runtime_error,
                        fmt::format("Failed to insert chromosome after {} trials.", max_trials));

    // Generate all the positions along the curve due to the placement in the global space
    const size_t num_nodes_per_chromosome = sphere_position_array.size();
    for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
      all_chromosome_positions[ichromosome].emplace_back(chromosome_centers_array.back() + r_chromosome_center_local -
                                                         sphere_position_array[i]);
    }
  }

  return all_chromosome_positions;
}

struct HP1ParamParser {
  void print_help_message() {
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "To run this code, please pass in --params=<input.yaml> as a command line argument." << std::endl;
    std::cout << std::endl;
    std::cout << "Note, all parameters and sublists in input.yaml must be contained in a single top-level list."
              << std::endl;
    std::cout << "Such as:" << std::endl;
    std::cout << std::endl;
    std::cout << "HP1:" << std::endl;
    std::cout << "  num_time_steps: 1000" << std::endl;
    std::cout << "  timestep_size: 1e-6" << std::endl;
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "The valid parameters that can be set in the input file are:" << std::endl;
    Teuchos::ParameterList valid_params = HP1::get_valid_params();

    auto print_options =
        Teuchos::ParameterList::PrintOptions().showTypes(false).showDoc(true).showDefault(true).showFlags(false).indent(
            1);
    valid_params.print(std::cout, print_options);
    std::cout << "#############################################################################################"
              << std::endl;
  }

  Teuchos::ParameterList parse(int argc, char **argv) {
    // Parse the command line options to find the input filename
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("params", &input_parameter_filename_, "The name of the input file.");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_result = cmdp.parse(argc, argv);
    if (parse_result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      print_help_message();

      // Safely exit the program
      // If we print the help message, we don't need to do anything else.
      exit(0);

    } else if (parse_result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      throw std::invalid_argument("Failed to parse the command line arguments.");
    }

    // Read, validate, and parse in the parameters from the parameter list.
    try {
      Teuchos::ParameterList param_list = *Teuchos::getParametersFromYamlFile(input_parameter_filename_);
      return parse(param_list);
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Failed to read the input parameter file." << std::endl;
      std::cerr << "During read, the following error occurred: " << e.what() << std::endl;
      std::cerr << "NOTE: This can happen for any number of reasons. Check that the file exists and contains the "
                   "expected parameters."
                << std::endl;
      throw e;
    }

    return Teuchos::ParameterList();
  }

  Teuchos::ParameterList parse(const Teuchos::ParameterList &param_list) {
    // Validate the parameters and set the defaults.
    Teuchos::ParameterList valid_param_list = param_list;
    valid_param_list.validateParametersAndSetDefaults(get_valid_params());
    check_invariants(valid_param_list);

    // if (crosslinker_spring_type_ == BOND_TYPE::HARMONIC) {
    //   crosslinker_rcut_ = crosslinker_r0_ + 5.0 * std::sqrt(1.0 / (crosslinker_kt_ * crosslinker_spring_constant_));
    // } else if (crosslinker_spring_type_ == BOND_TYPE::FENE) {
    //   // The r0 quantity for FENE bonds is the rmax at which force goes to infinity, so anything beyond this in
    //   invalid! crosslinker_rcut_ = crosslinker_r0_;
    // }

    dump_parameters(valid_param_list);

    return valid_param_list;
  }

  void check_invariants(const Teuchos::ParameterList &param_list) {
    // Check the sim params
    Teuchos::ParameterList &sim_params = valid_param_list.sublist("sim");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("timestep_size") > 0, std::invalid_argument,
                        "timestep_size must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("viscosity") > 0, std::invalid_argument,
                        "viscosity must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("initial_chromosome_separation") >= 0, std::invalid_argument,
                        "initial_chromosome_separation must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<bool>("enable_periphery_hydrodynamics")
                            ? sim_params.get<bool>("enable_backbone_n_body_hydrodynamics")
                            : true,
                        std::invalid_argument,
                        "Periphery hydrodynamics requires backbone hydrodynamics to be enabled.");

    // Check the periphery_hydro params
    if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
      Teuchos::ParameterList &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");
      PERIPHERY_SHAPE periphery_hydro_shape = periphery_hydro_params.get<std::string>("shape");
      PERIPHERY_QUADRATURE periphery_hydro_quadrature = periphery_hydro_params.get<std::string>("quadrature");
      if (periphery_hydro_quadrature == PERIPHERY_QUADRATURE::GAUSS_LEGENDRE) {
        double periphery_hydro_axis_radius1 = periphery_hydro_params.get<double>("axis_radius1");
        double periphery_hydro_axis_radius2 = periphery_hydro_params.get<double>("axis_radius2");
        double periphery_hydro_axis_radius3 = periphery_hydro_params.get<double>("axis_radius3");
        MUNDY_THROW_REQUIRE((periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) ||
                                ((periphery_hydro_shape_ == PERIPHERY_SHAPE::ELLIPSOID) &&
                                 (periphery_hydro_axis_radius1 == periphery_hydro_axis_radius2) &&
                                 (periphery_hydro_axis_radius2 == periphery_hydro_axis_radius3) &&
                                 (periphery_hydro_axis_radius3 == periphery_hydro_axis_radius1)),
                            std::invalid_argument,
                            "Gauss-Legendre quadrature is only valid for spherical peripheries.");
      }
    }
  }

  static Teuchos::ParameterList get_valid_params() {
    // Create a paramater entity validator for our large integers to allow for both int and long long.
    auto prefer_size_t = []() {
      if (std::is_same_v<size_t, unsigned short>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_SHORT;
      } else if (std::is_same_v<size_t, unsigned int>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      } else if (std::is_same_v<size_t, unsigned long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG;
      } else if (std::is_same_v<size_t, unsigned long long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG_LONG;
      } else {
        throw std::runtime_error("Unknown size_t type.");
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      }
    }();
    const bool allow_all_types_by_default = false;
    mundy::core::OurAnyNumberParameterEntryValidator::AcceptedTypes accept_int(allow_all_types_by_default);
    accept_int.allow_all_integer_types(true);
    auto make_new_validator = [](const auto &preferred_type, const auto &accepted_types) {
      return Teuchos::rcp(new mundy::core::OurAnyNumberParameterEntryValidator(preferred_type, accepted_types));
    };

    static Teuchos::ParameterList valid_parameter_list;

    valid_parameter_list.sublist("sim")
        .set("num_time_steps", 100, "Number of time steps.", make_new_validator(prefer_size_t, accept_int))
        .set("timestep_size", 0.001, "Time step size.")
        .set("viscosity", 1.0, "Viscosity.")
        .set("num_chromosomes", 1, "Number of chromosomes.", make_new_validator(prefer_size_t, accept_int))
        .set("num_chromatin_repeats", 2, "Number of chromatin repeats per chain.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_euchromatin_per_repeat", 1, "Number of euchromatin beads per repeat.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_heterochromatin_per_repeat", 1, "Number of heterochromatin beads per repeat.",
             make_new_validator(prefer_size_t, accept_int))
        .set("backbone_sphere_hydrodynamic_radius", 0.05,
             "Backbone sphere hydrodynamic radius. Even if n-body hydrodynamics is disabled, we still have "
             "self-interaction.")
        .set("initial_chromosome_separation", 1.0, "Initial chromosome separation.")
        .set("initialization_type", std::string("GRID"), "Initialization_type.")
        .set("initialize_from_exo_filename", std::string("HP1"),
             "Exo file to initialize from if initialization_type is FROM_EXO.")
        .set("initialize_from_dat_filename", std::string("HP1_pos.dat"),
             "Dat file to initialize from if initialization_type is FROM_DAT.")
        .set<Teuchos::Array<double>>(
            "unit_cell_size", Teuchos::tuple<double>(10.0, 10.0, 10.0),
            "Unit cell size in each dimension. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set("check_maximum_speed_pre_position_update", false, "Check maximum speed before updating positions.")
        .set("max_allowable_speed", std::numeric_limits<double>::max(),
             "Maximum allowable speed (only used if "
             "check_maximum_speed_pre_position_update is true).")
        // IO
        .set("loadbalance_post_initialization", false, "If we should load balance post-initialization or not.")
        .set("io_frequency", 10, "Number of timesteps between writing output.",
             make_new_validator(prefer_size_t, accept_int))
        .set("log_frequency", 10, "Number of timesteps between logging.", make_new_validator(prefer_size_t, accept_int))
        .set("output_filename", std::string("HP1"), "Output filename.")
        .set("enable_continuation_if_available", true,
             "Enable continuing a previous simulation if an output file already exists.")
        // Control flags
        .set("enable_chromatin_brownian_motion", true, "Enable chromatin Brownian motion.")
        .set("enable_backbone_springs", true, "Enable backbone springs.")
        .set("enable_backbone_collision", true, "Enable backbone collision.")
        .set("enable_backbone_n_body_hydrodynamics", true, "Enable backbone N-body hydrodynamics.")
        .set("enable_crosslinkers", true, "Enable crosslinkers.")
        .set("enable_periphery_collision", true, "Enable periphery collision.")
        .set("enable_periphery_hydrodynamics", true, "Enable periphery hydrodynamics.")
        .set("enable_periphery_binding", true, "Enable periphery binding.")
        .set("enable_active_euchromatin_forces", true, "Enable active euchromatin forces.");

    valid_parameter_list.sublist("brownian_motion").set("kt", 1.0, "Temperature kT for Brownian Motion.");

    valid_parameter_list.sublist("backbone_springs")
        .set("spring_type", std::string("HARMONIC"), "Chromatin spring type.")
        .set("spring_constant", 100.0, "Chromatin spring constant.")
        .set("spring_r0", 1.0, "Chromatin rest length (HARMONIC) or rmax (FENE).");

    valid_parameter_list.sublist("backbone_collision")
        .set("backbone_excluded_volume_radius", 0.5, "Backbone excluded volume radius.")
        .set("backbone_youngs_modulus", 1000.0, "Backbone Young's modulus.")
        .set("backbone_poissons_ratio", 0.3, "Backbone Poisson's ratio.");

    valid_parameter_list.sublist("crosslinker")
        .set("spring_type", std::string("HARMONIC"), "Crosslinker spring type.")
        .set("kt", 1.0, "Temperature kT for crosslinkers.")
        .set("spring_constant", 10.0, "Crosslinker spring constant.")
        .set("r0", 2.5, "Crosslinker rest length.")
        .set("left_binding_rate", 1.0, "Crosslinker left binding rate.")
        .set("right_binding_rate", 1.0, "Crosslinker right binding rate.")
        .set("left_unbinding_rate", 1.0, "Crosslinker left unbinding rate.")
        .set("right_unbinding_rate", 1.0, "Crosslinker right unbinding rate.");

    valid_parameter_list.sublist("periphery_hydro")
        .set("check_maximum_periphery_overlap", false, "Check maximum periphery overlap.")
        .set("maximum_allowed_periphery_overlap", 1e-6, "Maximum allowed periphery overlap.")
        .set("shape", std::string("SPHERE"), "Periphery hydrodynamic shape.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("quadrature", std::string("GAUSS_LEGENDRE"), "Periphery quadrature.")
        .set("spectral_order", 32,
             "Periphery spectral order (only used if periphery is spherical is Gauss-Legendre quadrature).",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_quadrature_points", 1000,
             "Periphery number of quadrature points (only used if quadrature type is FROM_FILE). Number of points in "
             "the files must match this quantity.",
             make_new_validator(prefer_size_t, accept_int))
        .set("quadrature_points_filename", std::string("hp1_periphery_hydro_quadrature_points.dat"),
             "Periphery quadrature points filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_weights_filename", std::string("hp1_periphery_hydro_quadrature_weights.dat"),
             "Periphery quadrature weights filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_normals_filename", std::string("hp1_periphery_hydro_quadrature_normals.dat"),
             "Periphery quadrature normals filename (only used if quadrature type is FROM_FILE).");

    valid_parameter_list.sublist("periphery_collision")
        .set("shape", std::string("SPHERE"), "Periphery collision shape.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("collision_spring_constant", 1000.0, "Periphery collision spring constant.")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("use_fast_approx", false, "Use fast periphery collision.")
        .set("shrink_periphery_over_time", false, "Shrink periphery over time.")
        .sublist("shrinkage")
        .set("num_shrinkage_steps", 1000,
             "Number of steps over which to perform the shrinking process (should not exceed num_time_steps).",
             make_new_validator(prefer_size_t, accept_int))
        .set("scale_factor_before_shrinking", 1.0, "Scale factor before shrinking.");

    valid_parameter_list.sublist("periphery_binding")
        .set("binding_rate", 1.0, "Periphery binding rate.")
        .set("unbinding_rate", 1.0, "Periphery unbinding rate.")
        .set("spring_constant", 1000.0, "Periphery spring constant.")
        .set("r0", 1.0, "Periphery spring rest length.")
        .set("bind_sites_type", std::string("RANDOM"), "Periphery bind sites type.")
        .set("num_bind_sites", 1000,
             "Periphery number of binding sites (only used if periphery_binding_sites_type is RANDOM and periphery "
             "has spherical or ellipsoidal shape).",
             make_new_validator(prefer_size_t, accept_int))
        .set("bind_site_locations_filename", std::string("periphery_bind_sites.dat"),
             "Periphery binding sites filename (only used if periphery_binding_sites_type is FROM_FILE).");

    valid_parameter_list.sublist("active_euchromatin_forces")
        .set("force_sigma", 1.0, "Active euchromatin force sigma.")
        .set("kon", 1.0, "Active euchromatin force kon.")
        .set("koff", 1.0, "Active euchromatin force koff.");

    valid_parameter_list.sublist("neighbor_list")
        .set("skin_distance", 1.0, "Neighbor list skin distance.")
        .set("force_neighborlist_update", false, "Force update of the neighbor list.")
        .set("force_neighborlist_update_nsteps", 10, "Number of timesteps between force update of the neighbor list.",
             make_new_validator(prefer_size_t, accept_int))
        .set("print_neighborlist_statistics", false, "Print neighbor list statistics.");

    return valid_parameter_list;
  }

  void dump_parameters(const Teuchos::ParameterList &valid_param_list) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;

      std::cout << std::endl;
      Teuchos::ParameterList &sim_params = valid_param_list.sublist("sim");
      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  num_time_steps:  " << sim_params.get<size_t>("num_time_steps") << std::endl;
      std::cout << "  timestep_size:   " << sim_params.get<double>("timestep_size") << std::endl;
      std::cout << "  viscosity:       " << sim_params.get<double>("viscosity") << std::endl;
      std::cout << "  num_chromosomes: " << sim_params.get<size_t>("num_chromosomes") << std::endl;
      std::cout << "  num_chromatin_repeats:      " << sim_params.get<size_t>("num_chromatin_repeats") << std::endl;
      std::cout << "  num_euchromatin_per_repeat: " << sim_params.get<size_t>("num_euchromatin_per_repeat")
                << std::endl;
      std::cout << "  num_heterochromatin_per_repeat:  " << sim_params.get<size_t>("num_heterochromatin_per_repeat")
                << std::endl;
      std::cout << "  backbone_sphere_hydrodynamic_radius: "
                << sim_params.get<double>("backbone_sphere_hydrodynamic_radius") << std::endl;
      std::cout << "  initial_chromosome_separation:   " << sim_params.get<double>("initial_chromosome_separation")
                << std::endl;
      std::cout << "  initialization_type:             " << sim_params.get<std::string>("initialization_type")
                << std::endl;
      if (sim_params.get<std::string>("initialization_type") == "FROM_EXO") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_exo_filename")
                  << std::endl;
      }

      if (sim_params.get<std::string>("initialization_type") == "FROM_DAT") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_dat_filename")
                  << std::endl;
      }

      if ((sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL") ||
          (sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL")) {
        std::cout << "  unit_cell_size: {" << sim_params.get<Teuchos::Array<double>>("unit_cell_size")[0] << ", "
                  << sim_params.get<Teuchos::Array<double>>("unit_cell_size")[1] << ", "
                  << sim_params.get<Teuchos::Array<double>>("unit_cell_size")[2] << "}" << std::endl;
      }

      std::cout << "  loadbalance_post_initialization: " << sim_params.get<bool>("loadbalance_post_initialization")
                << std::endl;
      std::cout << "  check_maximum_speed_pre_position_update: "
                << sim_params.get<bool>("check_maximum_speed_pre_position_update") << std::endl;
      if (sim_params.get<bool>("check_maximum_speed_pre_position_update")) {
        std::cout << "  max_allowable_speed: " << sim_params.get<double>("max_allowable_speed") << std::endl;
      }
      std::cout << std::endl;

      std::cout << "IO:" << std::endl;
      std::cout << "  io_frequency:    " << sim_params.get<size_t>("io_frequency") << std::endl;
      std::cout << "  log_frequency:   " << sim_params.get<size_t>("log_frequency") << std::endl;
      std::cout << "  output_filename: " << sim_params.get<std::string>("output_filename") << std::endl;
      std::cout << "  enable_continuation_if_available: " << sim_params.get<bool>("enable_continuation_if_available")
                << std::endl;
      std::cout << std::endl;

      std::cout << "CONTROL FLAGS:" << std::endl;
      std::cout << "  enable_chromatin_brownian_motion: " << sim_params.get<bool>("enable_chromatin_brownian_motion")
                << std::endl;
      std::cout << "  enable_backbone_springs:          " << sim_params.get<bool>("enable_backbone_springs")
                << std::endl;
      std::cout << "  enable_backbone_collision:        " << sim_params.get<bool>("enable_backbone_collision")
                << std::endl;
      std::cout << "  enable_backbone_n_body_hydrodynamics:    "
                << sim_params.get<bool>("enable_backbone_n_body_hydrodynamics") << std::endl;
      std::cout << "  enable_crosslinkers:              " << sim_params.get<bool>("enable_crosslinkers") << std::endl;
      std::cout << "  enable_periphery_hydrodynamics:   " << sim_params.get<bool>("enable_periphery_hydrodynamics")
                << std::endl;
      std::cout << "  enable_periphery_collision:       " << sim_params.get<bool>("enable_periphery_collision")
                << std::endl;
      std::cout << "  enable_periphery_binding:         " << sim_params.get<bool>("enable_periphery_binding")
                << std::endl;
      std::cout << "  enable_active_euchromatin_forces: " << sim_params.get<bool>("enable_active_euchromatin_forces")
                << std::endl;

      if (sim_params.get<bool>("enable_chromatin_brownian_motion")) {
        Teuchos::ParameterList &brownian_motion_params = valid_param_list.sublist("brownian_motion");

        std::cout << std::endl;
        std::cout << "BROWNIAN MOTION:" << std::endl;
        std::cout << "  kt: " << brownian_motion_params.get<double>("kt") << std::endl;
      }

      if (sim_params.get<bool>("enable_backbone_springs")) {
        Teuchos::ParameterList &backbone_springs_params = valid_param_list.sublist("backbone_springs");

        std::cout << std::endl;
        std::cout << "BACKBONE SPRINGS:" << std::endl;
        std::cout << "  spring_type:      " << backbone_springs_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  spring_constant:  " << backbone_springs_params.get<double>("spring_constant") << std::endl;
        if (backbone_springs_params.get<std::string>("spring_type") == "HARMONIC") {
          std::cout << "  spring_rest_length: " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        } else if (backbone_springs_params.get<std::string>("spring_type") == "FENE") {
          std::cout << "  spring_rmax:        " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_backbone_collision")) {
        Teuchos::ParameterList &backbone_collision_params = valid_param_list.sublist("backbone_collision");

        std::cout << std::endl;
        std::cout << "BACKBONE COLLISION:" << std::endl;
        std::cout << "  excluded_volume_radius: "
                  << backbone_collision_params.get<double>("backbone_excluded_volume_radius") << std::endl;
        std::cout << "  youngs_modulus: " << backbone_collision_params.get<double>("backbone_youngs_modulus")
                  << std::endl;
        std::cout << "  poissons_ratio: " << backbone_collision_params.get<double>("backbone_poissons_ratio")
                  << std::endl;
      }

      if (sim_params.get<bool>("enable_crosslinkers")) {
        Teuchos::ParameterList &crosslinker_params = valid_param_list.sublist("crosslinker");

        std::cout << std::endl;
        std::cout << "CROSSLINKERS:" << std::endl;
        std::cout << "  spring_type: " << crosslinker_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  kt: " << crosslinker_params.get<double>("kt") << std::endl;
        std::cout << "  spring_constant: " << crosslinker_params.get<double>("spring_constant") << std::endl;
        std::cout << "  r0: " << crosslinker_params.get<double>("r0") << std::endl;
        std::cout << "  left_binding_rate: " << crosslinker_params.get<double>("left_binding_rate") << std::endl;
        std::cout << "  right_binding_rate: " << crosslinker_params.get<double>("right_binding_rate") << std::endl;
        std::cout << "  left_unbinding_rate: " << crosslinker_params.get<double>("left_unbinding_rate") << std::endl;
        std::cout << "  right_unbinding_rate: " << crosslinker_params.get<double>("right_unbinding_rate") << std::endl;
        std::cout << "  rcut: " << crosslinker_params.get<double>("rcut") << std::endl;
      }

      if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
        Teuchos::ParameterList &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");

        std::cout << std::endl;
        std::cout << "PERIPHERY HYDRODYNAMICS:" << std::endl;
        std::cout << "  check_maximum_periphery_overlap: "
                  << periphery_hydro_params.get<bool>("check_maximum_periphery_overlap") << std::endl;
        if (periphery_hydro_params.get<bool>("check_maximum_periphery_overlap")) {
          std::cout << "  maximum_allowed_periphery_overlap: "
                    << periphery_hydro_params.get<double>("maximum_allowed_periphery_overlap") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_hydro_params.get<double>("radius") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_hydro_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_hydro_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_hydro_params.get<double>("axis_radius3") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("quadrature") == "GAUSS_LEGENDRE") {
          std::cout << "  quadrature: GAUSS_LEGENDRE" << std::endl;
          std::cout << "  spectral_order: " << periphery_hydro_params.get<size_t>("spectral_order") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("quadrature") == "FROM_FILE") {
          std::cout << "  quadrature: FROM_FILE" << std::endl;
          std::cout << "  num_quadrature_points: " << periphery_hydro_params.get<size_t>("num_quadrature_points")
                    << std::endl;
          std::cout << "  quadrature_points_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_points_filename") << std::endl;
          std::cout << "  quadrature_weights_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_weights_filename") << std::endl;
          std::cout << "  quadrature_normals_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_normals_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_periphery_collision")) {
        Teuchos::ParameterList &periphery_collision_params = valid_param_list.sublist("periphery_collision");

        std::cout << std::endl;
        std::cout << "PERIPHERY COLLISION:" << std::endl;
        if (periphery_collision_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_collision_params.get<double>("radius") << std::endl;
        } else if (periphery_collision_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_collision_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_collision_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_collision_params.get<double>("axis_radius3") << std::endl;
        }
        std::cout << "  collision_spring_constant: "
                  << periphery_collision_params.get<double>("collision_spring_constant") << std::endl;
        std::cout << "  periphery_collision_use_fast_approx: "
                  << periphery_collision_params.get<bool>("use_fast_approx") << std::endl;
        std::cout << "  shrink_periphery_over_time: "
                  << periphery_collision_params.get<bool>("shrink_periphery_over_time") << std::endl;
        if (periphery_collision_params.get<bool>("shrink_periphery_over_time")) {
          std::cout << "  SHRINKAGE:" << std::endl;
          std::cout << "    num_shrinkage_steps: " << periphery_collision_params.get<size_t>("num_shrinkage_steps")
                    << std::endl;
          std::cout << "    scale_factor_before_shrinking: "
                    << periphery_collision_params.get<double>("scale_factor_before_shrinking") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_periphery_binding")) {
        Teuchos::ParameterList &periphery_binding_params = valid_param_list.sublist("periphery_binding");

        std::cout << std::endl;
        std::cout << "PERIPHERY BINDING:" << std::endl;
        std::cout << "  binding_rate: " << periphery_binding_params.get<double>("binding_rate") << std::endl;
        std::cout << "  unbinding_rate: " << periphery_binding_params.get<double>("unbinding_rate") << std::endl;
        std::cout << "  spring_constant: " << periphery_binding_params.get<double>("spring_constant") << std::endl;
        std::cout << "  r0: " << periphery_binding_params.get<double>("r0") << std::endl;
        if (periphery_binding_params.get<std::string>("bind_sites_type") == "RANDOM") {
          std::cout << "  bind_sites_type: RANDOM" << std::endl;
          std::cout << "  num_bind_sites: " << periphery_binding_params.get<size_t>("num_bind_sites") << std::endl;
        } else if (periphery_binding_params.get<std::string>("bind_sites_type") == "FROM_FILE") {
          std::cout << "  bind_sites_type: FROM_FILE" << std::endl;
          std::cout << "  bind_site_locations_filename: "
                    << periphery_binding_params.get<std::string>("bind_site_locations_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_active_euchromatin_forces")) {
        Teuchos::ParameterList &active_euchromatin_forces_params =
            valid_param_list.sublist("active_euchromatin_forces");

        std::cout << std::endl;
        std::cout << "ACTIVE EUCHROMATIN FORCES:" << std::endl;
        std::cout << "  force_sigma: " << active_euchromatin_forces_params.get<double>("force_sigma") << std::endl;
        std::cout << "  kon: " << active_euchromatin_forces_params.get<double>("kon") << std::endl;
        std::cout << "  koff: " << active_euchromatin_forces_params.get<double>("koff") << std::endl;
      }

      std::cout << std::endl;

      std::cout << "NEIGHBOR LIST:" << std::endl;
      Teuchos::ParameterList &neighbor_list_params = valid_param_list.sublist("neighbor_list");
      std::cout << "  skin_distance: " << neighbor_list_params.get<double>("skin_distance") << std::endl;
      std::cout << "  force_neighborlist_update: " << neighbor_list_params.get<bool>("force_neighborlist_update")
                << std::endl;
      std::cout << "  force_neighborlist_update_nsteps: "
                << neighbor_list_params.get<size_t>("force_neighborlist_update_nsteps") << std::endl;
      std::cout << "  print_neighborlist_statistics: "
                << neighbor_list_params.get<bool>("print_neighborlist_statistics") << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

 private:
  /// \brief Default parameter filename if none is provided.
  std::string input_parameter_filename_ = "hp1.yaml";
};

void old_run(int argc, char **argv) {
  // Preprocess
  parse_user_inputs(argc, argv);
  dump_user_inputs();

  // Setup
  Kokkos::Profiling::pushRegion("HP1::Setup");
  build_our_mesh_and_method_instances();

  fetch_fields_and_parts();
  instantiate_metamethods();
  set_mutable_parameters();
  declare_and_initialize_hp1();
  if (enable_periphery_hydrodynamics_) {
    initialize_hydrodynamic_periphery();
  }
  if (enable_periphery_binding_ && !restart_performed_) {
    declare_and_initialize_periphery_bind_sites();
  }
  if (enable_active_euchromatin_forces_) {
    initialize_euchromatin();
  }

  detect_neighbors_initial();
  Kokkos::Profiling::popRegion();

  // Post setup
  Kokkos::Profiling::pushRegion("HP1::Loadbalance");
  // Post setup but pre-run
  if (loadbalance_post_initialization_) {
    stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);
  }
  Kokkos::Profiling::popRegion();

  // Reset simulation control variables
  timestep_index_ = 0;
  timestep_current_time_ = 0.0;
  if (enable_continuation_if_available_ && restart_performed_) {
    timestep_index_ = restart_timestep_index_;
    timestep_current_time_ = restart_timestep_index_ * timestep_size_;
  }

  // Check to see if we need to do anything for compressing the system.
  if (enable_periphery_collision_ && shrink_periphery_over_time_) {
    if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
      periphery_collision_radius_ *= periphery_collision_scale_factor_before_shrinking_;
    } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
      periphery_collision_axis_radius1_ *= periphery_collision_scale_factor_before_shrinking_;
      periphery_collision_axis_radius2_ *= periphery_collision_scale_factor_before_shrinking_;
      periphery_collision_axis_radius3_ *= periphery_collision_scale_factor_before_shrinking_;
    }
  }

  // Time loop
  print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

  Kokkos::Timer overall_timer;
  Kokkos::Timer timer;
  Kokkos::Profiling::pushRegion("MainLoop");
  // We have pre-loaded the starting index and time...
  for (; timestep_index_ < num_time_steps_; timestep_index_++, timestep_current_time_ += timestep_size_) {
    // Prepare the current configuration.
    Kokkos::Profiling::pushRegion("HP1::PrepareStep");
    mundy::mesh::utils::fill_field_with_value<double>(  //
        *node_velocity_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(  //
        *node_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(  //
        *element_binding_rates_field_ptr_, std::array<double, 2>{0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(  //
        *element_unbinding_rates_field_ptr_, std::array<double, 2>{0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<unsigned>(  //
        *element_perform_state_change_field_ptr_, std::array<unsigned, 1>{0u});
    mundy::mesh::utils::fill_field_with_value<unsigned>(  //
        *euchromatin_perform_state_change_field_ptr_, std::array<unsigned, 1>{0u});
    mundy::mesh::utils::fill_field_with_value<unsigned>(  //
        *constraint_perform_state_change_field_ptr_, std::array<unsigned, 1>{0u});
    mundy::mesh::utils::fill_field_with_value<double>(  //
        *constraint_state_change_rate_field_ptr_, std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(  //
        *constraint_potential_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    rotate_field_states();
    Kokkos::Profiling::popRegion();

    // If we are doing a compression run, shrink the periphery
    if (enable_periphery_collision_ && shrink_periphery_over_time_) {
      const double shrink_factor = std::pow(1.0 / periphery_collision_scale_factor_before_shrinking_,
                                            1.0 / periphery_collision_shrinkage_num_steps_);
      if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
        periphery_collision_radius_ *= shrink_factor;
      } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
        periphery_collision_axis_radius1_ *= shrink_factor;
        periphery_collision_axis_radius2_ *= shrink_factor;
        periphery_collision_axis_radius3_ *= shrink_factor;
      }
    }

    // Detect sphere-sphere and crosslinker-sphere neighbors
    update_neighbor_list_ = false;
    detect_neighbors();

    // Determine KMC events
    if (enable_crosslinkers_) update_crosslinker_state();

    if (enable_active_euchromatin_forces_) update_active_euchromatin_state();

    // Evaluate forces f(x(t)).
    if (enable_backbone_collision_) compute_hertzian_contact_forces();

    if (enable_backbone_springs_) compute_backbone_harmonic_bond_forces();

    if (enable_crosslinkers_) compute_crosslinker_harmonic_bond_forces();

    if (enable_periphery_collision_) compute_periphery_collision_forces();

    if (enable_active_euchromatin_forces_) compute_euchromatin_active_forces();

    // Compute velocities.
    if (enable_chromatin_brownian_motion_) compute_brownian_velocity();

    if (enable_backbone_n_body_hydrodynamics_) {
      compute_rpy_hydro();
    } else {
      compute_dry_velocity();
    }

    // Logging, if desired, write to console
    Kokkos::Profiling::pushRegion("HP1::Logging");
    if (timestep_index_ % log_frequency_ == 0) {
      if (bulk_data_ptr_->parallel_rank() == 0) {
        double tps = static_cast<double>(log_frequency_) / static_cast<double>(timer.seconds());
        std::cout << "Step: " << std::setw(15) << timestep_index_ << ", tps: " << std::setprecision(15) << tps;
        if (enable_periphery_collision_ && shrink_periphery_over_time_) {
          if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
            std::cout << ", periphery_collision_radius: " << periphery_collision_radius_;
          } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
            std::cout << ", periphery_collision_axis_radius1: " << periphery_collision_axis_radius1_
                      << ", periphery_collision_axis_radius2: " << periphery_collision_axis_radius2_
                      << ", periphery_collision_axis_radius3: " << periphery_collision_axis_radius3_;
          }
        }
        std::cout << std::endl;
        timer.reset();
      }
    }
    Kokkos::Profiling::popRegion();

    // IO. If desired, write out the data for time t (STK or mundy)
    Kokkos::Profiling::pushRegion("HP1::IO");
    if (timestep_index_ % io_frequency_ == 0) {
      io_broker_ptr_->write_io_broker_timestep(static_cast<int>(timestep_index_), timestep_current_time_);
    }
    Kokkos::Profiling::popRegion();

    // Update positions. x(t + dt) = x(t) + dt * v(t).
    update_positions();

    // Update the time for the euchromatin active forces
    if (enable_active_euchromatin_forces_) update_euchromatin_state_time();
  }
  Kokkos::Profiling::popRegion();

  // Do a synchronize to force everybody to stop here, then write the time
  stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
  if (bulk_data_ptr_->parallel_rank() == 0) {
    double avg_time_per_timestep = static_cast<double>(overall_timer.seconds()) / static_cast<double>(num_time_steps_);
    double tps = 1.0 / avg_time_per_timestep;
    std::cout << "******************Final statistics (Rank 0)**************\n";
    if (print_neighborlist_statistics_) {
      std::cout << "****************\n";
      std::cout << "Neighbor list statistics\n";
      for (auto &neighborlist_entry : neighborlist_update_steps_times_) {
        auto [timestep, elasped_step, elapsed_time] = neighborlist_entry;
        auto tps_nl = static_cast<double>(elasped_step) / elapsed_time;
        std::cout << "  Rebuild timestep: " << timestep << ", elapsed_steps: " << elasped_step
                  << ", elapsed_time: " << elapsed_time << ", tps: " << tps_nl << std::endl;
      }
    }
    std::cout << "****************\n";
    std::cout << "Simulation statistics\n";
    std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
    std::cout << "Timesteps per second: " << std::setprecision(15) << tps << std::endl;
  }
}

void run(int argc, char **argv) {
  debug_print("Running the simulation.");

  // Preprocess
  Teuchos::ParameterList params = HP1ParamParser().parse_user_inputs(argc, argv);
  Teuchos::ParameterList &sim_params = params.sublist("sim");
  Teuchos::ParameterList &brownian_motion_params = params.sublist("brownian_motion");
  Teuchos::ParameterList &backbone_springs_params = params.sublist("backbone_springs");
  Teuchos::ParameterList &backbone_collision_params = params.sublist("backbone_collision");
  Teuchos::ParameterList &crosslinker_params = params.sublist("crosslinker");
  Teuchos::ParameterList &periphery_hydro_params = params.sublist("periphery_hydro");
  Teuchos::ParameterList &periphery_collision_params = params.sublist("periphery_collision");
  Teuchos::ParameterList &periphery_binding_params = params.sublist("periphery_binding");
  Teuchos::ParameterList &active_euchromatin_forces_params = params.sublist("active_euchromatin_forces");
  Teuchos::ParameterList &neighbor_list_params = params.sublist("neighbor_list");

  // Setup the STK mesh
  stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3);
  mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition to
                                       // all fields are simple.
  meta_data_ptr->set_coordinate_field_name("NODE_COORDS");
  stk::mesh::MetaData &meta_data = *meta_data_ptr;

  // Parts and their subsets
  auto &e_part = mundy::declare_io_part(meta_data, "EUCHROMATIN_SPHERES", stk::topology::PARTICLE);
  auto &h_part = mundy::declare_io_part(meta_data, "HETEROCHROMATIN_SPHERES", stk::topology::PARTICLE);
  auto &bs_part = mundy::declare_io_part(meta_data, "BIND_SITES", stk::topology::NODE);

  auto &hp1_part = mundy::declare_io_part(meta_data, "HP1", stk::topology::BEAM_2);
  auto &left_hp1_part = mundy::declare_io_part(meta_data, "LEFT_HP1", stk::topology::BEAM_2);
  auto &doubly_hp1_h_part = mundy::declare_io_part(meta_data, "DOUBLY_HP1_H", stk::topology::BEAM_2);
  auto &doubly_hp1_bs_part = mundy::declare_io_part(meta_data, "DOUBLY_HP1_BS", stk::topology::BEAM_2);
  meta_data.declare_part_subset(hp1_part, left_hp1_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_h_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_bs_part);

  auto &backbone_segs_part = mundy::declare_io_part(meta_data, "BACKBONE_SEGMENTS", stk::topology::BEAM_2);
  auto &ee_segs_part = mundy::declare_io_part(meta_data, "EE_SEGMENTS", stk::topology::BEAM_2);
  auto &eh_segs_part = mundy::declare_io_part(meta_data, "EH_SEGMENTS", stk::topology::BEAM_2);
  auto &hh_segs_part = mundy::declare_io_part(meta_data, "HH_SEGMENTS", stk::topology::BEAM_2);
  meta_data.declare_part_subset(backbone_segs_part, ee_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, eh_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, hh_segs_part);

  // Fields
  auto &n_coord_field = meta_data.declare_field<double>(NODE_RANK, "COORDS");
  auto &n_velocity_field = meta_data.declare_field<double>(NODE_RANK, "VELOCITY");
  auto &n_force_field = meta_data.declare_field<double>(NODE_RANK, "FORCE");
  auto &n_rng_field = meta_data.declare_field<unsigned>(NODE_RANK, "RNG_COUNTER");

  auto &el_hydrodynamic_radius_field = meta_data.declare_field<double>(ELEMENT_RANK, "HYDRODYNAMIC_RADIUS");
  auto &el_binding_radius_field = meta_data.declare_field<double>(ELEMENT_RANK, "BINDING_RADIUS");
  auto &el_collision_radius_field = meta_data.declare_field<double>(ELEMENT_RANK, "COLLISION_RADIUS");
  auto &el_hookean_spring_constant_field = meta_data.declare_field<double>(ELEMENT_RANK, "SPRING_CONSTANT");
  auto &el_hookean_spring_rest_length_field = meta_data.declare_field<double>(ELEMENT_RANK, "SPRING_REST_LENGTH");
  auto &el_youngs_modulus_field = meta_data.declare_field<double>(ELEMENT_RANK, "YOUNGS_MODULUS");
  auto &el_poissons_ratio_field = meta_data.declare_field<double>(ELEMENT_RANK, "POISSONS_RATIO");
  auto &el_aabb_field = meta_data.declare_field<double>(ELEMENT_RANK, "AABB");
  auto &el_aabb_displacement_field = meta_data.declare_field<double>(ELEMENT_RANK, "AABB_DISPLACEMENT");
  auto &el_binding_rates_field = meta_data.declare_field<double>(ELEMENT_RANK, "BINDING_RATES");
  auto &el_unbinding_rates_field = meta_data.declare_field<double>(ELEMENT_RANK, "UNBINDING_RATES");
  auto &el_perform_state_change_field = meta_data.declare_field<unsigned>(ELEMENT_RANK, "PERFORM_STATE_CHANGE");
  auto &el_rng_field = meta_data.declare_field<unsigned>(ELEMENT_RANK, "RNG_COUNTER");

  // Sew it all together
  // Any field with a constant initial value should have that value set here. If it is
  // to be uninitialized, use nullptr.
  const double zero_vector3d[3] = {0.0, 0.0, 0.0};
  const double zero_scalar[1] = 0.0;
  const unsigned zero_unsigned[1] = 0;
  stk::mesh::put_field_on_entire_mesh(n_coord_field, nullptr);

  // Heterochromatin and euchromatin spheres are used for hydrodynamics. They move and
  // have forces applied to them. If brownian motion is enabled, they will have a
  // stocastic velocity. Heterochromatin spheres are considered for hp1 binding and
  // require an AABB for neighbor detection.
  stk::mesh::put_field_on_mesh(n_velocity_field, e_part | h_part, 3, zero_vector3d);
  stk::mesh::put_field_on_mesh(n_force_field, e_part | h_part, 3, zero_vector3d);
  stk::mesh::put_field_on_mesh(n_rng_field, e_part | h_part, 1, zero_unsigned);
  stk::mesh::put_field_on_mesh(el_hydrodynamic_radius_field, e_part | h_part, 1, chromatin_hydrodynamic_radius);
  stk::mesh::put_field_on_mesh(el_aabb_field, h_part, 6, nullptr);
  stk::mesh::put_field_on_mesh(el_aabb_displacement_field, h_part, 6, nullptr);

  // Backbone segs apply spring forces and act as spherocylinders for the sake of
  // collision. They apply forces to their nodes and have a collision radius. The
  // difference between ee, eh, and hh segs is that ee segs can exert an active
  // dipole.
  stk::mesh::put_field_on_mesh(n_force_field, backbone_segs_part, 3, zero_vector3d);
  stk::mesh::put_field_on_mesh(el_collision_radius_field, backbone_segs_part, 1, backbone_collision_radius);
  stk::mesh::put_field_on_mesh(el_hookean_spring_constant_field, backbone_segs_part, 1, backbone_spring_constant);
  stk::mesh::put_field_on_mesh(el_hookean_spring_rest_length_field, backbone_segs_part, 1, backbone_rest_length);
  stk::mesh::put_field_on_mesh(el_youngs_modulus_field, backbone_segs_part, 1, collision_youngs_modulus);
  stk::mesh::put_field_on_mesh(el_poissons_ratio_field, backbone_segs_part, 1, collision_poissons_ratio);
  stk::mesh::put_field_on_mesh(el_aabb_field, backbone_segs_part, 6, nullptr);
  stk::mesh::put_field_on_mesh(el_aabb_displacement_field, backbone_segs_part, 6, nullptr);

  // HP1 crosslinkers are used for binding/unbinding and apply forces to their nodes.
  const double left_and_right_binding_rates[2] = {left_binding_rate, right_binding_rate};
  const double left_and_right_unbinding_rates[2] = {left_unbinding_rate, right_unbinding_rate};
  stk::mesh::put_field_on_mesh(n_force_field, hp1_part, 3, zero_vector3d);
  stk::mesh::put_field_on_mesh(el_binding_rates_field, hp1_part, 2, left_and_right_binding_rates);
  stk::mesh::put_field_on_mesh(el_unbinding_rates_field, hp1_part, 2, left_and_right_unbinding_rates);
  stk::mesh::put_field_on_mesh(el_perform_state_change_field, hp1_part, 1, zero_unsigned);
  stk::mesh::put_field_on_mesh(el_binding_radius_field, hp1_part, 1, crosslinker_binding_radius);
  stk::mesh::put_field_on_mesh(el_rng_field, hp1_part, 1, zero_unsigned);

  // That's it for the mesh. Commit it's structure and create the bulk data.
  meta_data.commit();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

  // Perform restart (optional)
  bool restart_performed = false;

  if (!restart_performed) {
    /* Declare the chromatin and HP1
    //  E : euchromatin spheres
    //  H : heterochromatin spheres
    //  | : crosslinkers
    // ---: backbone springs/backbone segments
    //
    //  |   |                           |   |
    //  H---H---E---E---E---E---E---E---H---H
    //
    // The actual connectivity looks like this:
    //  n : node, s : segment and or spring, c : crosslinker
    //
    // c1_      c3_       c5_       c7_
    // | /      | /       | /       | /
    // n1       n3        n5        n7
    //  \      /  \      /  \      /
    //   s1   s2   s3   s4   s5   s6
    //    \  /      \  /      \  /
    //     n2        n4        n6
    //     | \       | \       | \
    //     c2⎻       c4⎻       c6⎻
    //
    // If you look at this long enough, the pattern is clear.
    //  - One less segment than nodes.
    //  - Same number of crosslinkers as heterochromatin nodes.
    //  - Segment i connects to nodes i and i+1.
    //  - Crosslinker i connects to nodes i and i.
    //
    // We need to use this information to populate the node and element info vectors.
    // Mundy will handle passing off this information to the bulk data. Just make sure that all
    // MPI ranks contain the same node and element info. This way, we can determine which nodes
    // should become shared.
    //
    // Rules (non-exhaustive):
    //  - Neither nodes nor elements need to have parts or fields.
    //  - The rank and type of the fields must be consistant. You can't pass an element field to a node,
    //    nor can you set the value of a field to a different type or size than it was declared as.
    //  - The owner of a node must be the same as one of the elements that connects to it.
    //  - A node connected to an element not on the same rank as the node will be shared with the owner of the
    element.
    //  - Field/Part names are case-sensitive but don't attempt to declare "field_1" and "Field_1" as if
    //    that will give two different fields since STKIO will not be able to distinguish between them.
    //  - A negative node id in the element connection list can be used to indicate that a node should be left
    unassigned.
    //  - All parts need to be able to contain an element of the given topology.
    */

    // Fill the declare entities helper
    mundy::mesh::DeclareEntitiesHelper dec_helper;
    size_t node_count = 0;
    size_t element_count = 0;

    // Setup in the ellipsoidal nucleus
    std::string filename = "ellipsoid.ply";
    happly::PLYData ply_in(filename);

    std::vector<double> x = ply_in.getElement("vertex").getProperty<double>("x");
    std::vector<double> y = ply_in.getElement("vertex").getProperty<double>("y");
    std::vector<double> z = ply_in.getElement("vertex").getProperty<double>("z");
    std::vector<std::vector<size_t>> face_ind = ply_in.getFaceIndices<size_t>();

    const size_t num_verts = x.size();
    const size_t num_faces = face_ind.size();
    for (size_t i = 0; i < num_verts; ++i) {
      dec_helper.create_node()
          .owning_proc(0)      //
          .id(node_count + 1)  //
          .add_field_data<double>(&node_coords_field, {x[i], y[i], z[i]});
      node_count++;
    }

    for (size_t i = 0; i < num_faces; ++i) {
      MUNDY_THROW_REQUIRE(face_ind[i].size() == 3, std::runtime_error, "Triangle face must have 3 vertices");
      dec_helper.create_element()
          .owning_proc(0)                        //
          .id(element_count + 1)                 //
          .topology(stk::topology::SHELL_TRI_3)  //
          .add_part(&nucleus_part)               //
          .nodes({face_ind[i][0] + 1, face_ind[i][1] + 1,
                  face_ind[i][2] + 1});  // only works because we declared the nucleus first
      element_count++;
    }

    // Setup the nucleus bind sites
    const size_t num_bind_sites = 10000;
    const double nucleus_axis_radius1 = 1.0;
    const double nucleus_axis_radius2 = 2.0;
    const double nucleus_axis_radius3 = 1.5;

    const double a = nucleus_axis_radius1;
    const double b = nucleus_axis_radius2;
    const double c = nucleus_axis_radius3;
    const double inv_mu_max = 1.0 / std::max({b * c, a * c, a * b});
    auto keep = [&a, &b, &c, &inv_mu_max](double x, double y, double z) {
      const double mu_xyz =
          std::sqrt((b * c * x) * (b * c * x) + (a * c * y) * (a * c * y) + (a * b * z) * (a * b * z));
      return inv_mu_max * mu_xyz > (static_cast<double>(rand()) / RAND_MAX);
    };

    for (size_t i = 0; i < num_bind_sites; i++) {
      // Rejection sampling to place the periphery binding sites
      double node_coords[3];
      while (true) {
        // Generate a random point on the unit sphere
        const double u1 = (static_cast<double>(rand()) / RAND_MAX);
        const double u2 = (static_cast<double>(rand()) / RAND_MAX);
        const double theta = 2.0 * M_PI * u1;
        const double phi = std::acos(2.0 * u2 - 1.0);
        node_coords[0] = std::sin(phi) * std::cos(theta);
        node_coords[1] = std::sin(phi) * std::sin(theta);
        node_coords[2] = std::cos(phi);

        // Keep this point with probability proportional to the surface area element
        if (keep(node_coords[0], node_coords[1], node_coords[2])) {
          // Pushforward the point to the ellipsoid
          node_coords[0] *= a;
          node_coords[1] *= b;
          node_coords[2] *= c;
          break;
        }
      }

      // Declare the node
      dec_helper.create_node()
          .owning_proc(0)      //
          .id(node_count + 1)  //
          .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
      node_count++;

      // Declare the element
      dec_helper.create_element()
          .owning_proc(0)                        //
          .id(element_count + 1)                 //
          .topology(stk::topology::PARTICLE)     //
          .add_part(&nucleus_binding_site_part)  //
          .nodes({node_count});
      element_count++;
    }

    // Setup the chromatin fibers
    const mundy::math::Vector3d nucleus_center = {0.0, 0.0, 0.0};
    const mundy::math::Vector3d nucleus_radii = {a, b, c};
    const size_t num_fibers = 100;
    const size_t num_hetero_euchromatin_repeats = 7;
    const size_t num_heterochromatin_per_repeat = 90;
    const size_t num_euchromatin_per_repeat = 315;
    const size_t num_nodes_per_heterochromatin =
        num_hetero_euchromatin_repeats * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat);
    const double fiber_radius = 0.01;
    const double segment_length = 0.02;
    const double segment_spring_constant = 1.0;
    const double euchromatin_hydrodynamic_radius = 0.01;
    const double heterochromatin_hydrodynamic_radius = 0.01;

    for (size_t f = 0; f < num_fibers; f++) {
      // Generate a random walk for the fiber starting from a random location in the nucleus
      mundy::math::Vector3d start = random_point_inside_ellipsoid(nucleus_center, nucleus_radii);
      std::vector<mundy::math::Vector3d> fiber_walk = random_walk_inside_ellipsoid(
          start, nucleus_center, nucleus_radii, num_nodes_per_heterochromatin, segment_length);

      // Declare the nodes, segments, and heterochromatin/euchromatin
      for (size_t r = 0; r < num_hetero_euchromatin_repeats; ++r) {
        // Heterochromatin
        for (size_t h = 0; h < num_heterochromatin_per_repeat; ++h) {
          const size_t node_index = r * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat) + h;

          dec_helper.create_node()
              .owning_proc(0)      //
              .id(node_count + 1)  //
              .add_field_data<double>(&node_coords_field, {fiber_walk[node_index][0], fiber_walk[node_index][1],
                                                           fiber_walk[node_index][2]});
          node_count++;

          // Only create the segment if the node is not the last node in this fiber
          if (node_index < num_nodes_per_heterochromatin - 1) {
            dec_helper.create_element()
                .owning_proc(0)                     //
                .id(element_count + 1)              //
                .topology(stk::topology::BEAM_2)    //
                .add_part(&chromatin_segment_part)  //
                .nodes({node_count, node_count + 1})
                .add_field_data<double>(&segment_radius_field, {fiber_radius})
                .add_field_data<double>(&segment_spring_constant_field, {segment_spring_constant});

            element_count++;
          }

          // Declare the heterochromatin
          dec_helper.create_element()
              .owning_proc(0)                     //
              .id(element_count + 1)              //
              .topology(stk::topology::PARTICLE)  //
              .add_part(&heterochromatin_part)    //
              .nodes({node_count})
              .add_field_data<double>(&hydrodynamic_radius_field, {heterochromatin_hydrodynamic_radius});
          element_count++;
        }

        for (size_t e = 0; e < num_euchromatin_per_repeat; ++e) {
          const size_t node_index =
              r * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat) + num_heterochromatin_per_repeat + e;

          dec_helper.create_node()
              .owning_proc(0)      //
              .id(node_count + 1)  //
              .add_field_data<double>(&node_coords_field, {fiber_walk[node_index][0], fiber_walk[node_index][1],
                                                           fiber_walk[node_index][2]});
          node_count++;

          // Only create the segment if the node is not the last node in this fiber
          if (node_index < num_nodes_per_heterochromatin - 1) {
            dec_helper.create_element()
                .owning_proc(0)                     //
                .id(element_count + 1)              //
                .topology(stk::topology::BEAM_2)    //
                .add_part(&chromatin_segment_part)  //
                .nodes({node_count, node_count + 1})
                .add_field_data<double>(&segment_radius_field, {fiber_radius})
                .add_field_data<double>(&segment_spring_constant_field, {segment_spring_constant});

            element_count++;
          }

          // Declare the euchromatin
          dec_helper.create_element()
              .owning_proc(0)                     //
              .id(element_count + 1)              //
              .topology(stk::topology::PARTICLE)  //
              .add_part(&euchromatin_part)        //
              .nodes({node_count})
              .add_field_data<double>(&hydrodynamic_radius_field, {euchromatin_hydrodynamic_radius});
          element_count++;
        }
      }
    }

    dec_helper.check_consistency(bulk_data);

    // Declare the entities
    bulk_data.modification_begin();
    dec_helper.declare_entities(bulk_data);
    bulk_data.modification_end();
  }

  // Post-setup but pre-run
  if (sim_params.get<bool>("loadbalance_post_initialization")) {
    stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);
  }

  // Get the NGP stuff
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  auto &ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  auto &ngp_node_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_velocity_field);
  auto &ngp_node_force_field = stk::mesh::get_updated_ngp_field<double>(node_force_field);
  auto &ngp_node_rng_field = stk::mesh::get_updated_ngp_field<unsigned>(node_rng_field);
  auto &ngp_elem_hydrodynamic_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_hydrodynamic_radius_field);
  auto &ngp_elem_binding_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_binding_radius_field);
  auto &ngp_elem_collision_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_collision_radius_field);
  auto &ngp_elem_hookean_spring_constant_field =
      stk::mesh::get_updated_ngp_field<double>(elem_hookean_spring_constant_field);
  auto &ngp_elem_hookean_spring_rest_length_field =
      stk::mesh::get_updated_ngp_field<double>(elem_hookean_spring_rest_length_field);
  auto &ngp_elem_youngs_modulus_field = stk::mesh::get_updated_ngp_field<double>(elem_youngs_modulus_field);
  auto &ngp_elem_poissons_ratio_field = stk::mesh::get_updated_ngp_field<double>(elem_poissons_ratio_field);
  auto &ngp_elem_aabb_field = stk::mesh::get_updated_ngp_field<double>(elem_aabb_field);
  auto &ngp_elem_aabb_displacement_field = stk::mesh::get_updated_ngp_field<double>(elem_aabb_displacement_field);
  auto &ngp_elem_binding_rates_field = stk::mesh::get_updated_ngp_field<double>(elem_binding_rates_field);
  auto &ngp_elem_unbinding_rates_field = stk::mesh::get_updated_ngp_field<double>(elem_unbinding_rates_field);
  auto &ngp_elem_perform_state_change_field = stk::mesh::get_updated_ngp_field<unsigned>(elem_perform_state_change_field);
  auto &ngp_elem_rng_field = stk::mesh::get_updated_ngp_field<unsigned>(elem_rng_field);

  // Time loop
  print_rank0(std::string("Running the simulation for ") + std::to_string(sim_params.get<size_t>("num_time_steps")) +
              " timesteps.");

  Kokkos::Timer overall_timer;
  Kokkos::Timer timer;
  for (size_t timestep_idx = 0; timestep_idx < sim_params.get<size_t>("num_time_steps"); timestep_idx++) {
    // Prepare the current configuration.
    ngp_node_velocity_field.sync_to_device();
    ngp_node_force_field.sync_to_device();
    ngp_elem_binding_rates_field.sync_to_device();
    ngp_elem_unbinding_rates_field.sync_to_device();
    ngp_elem_perform_state_change_field.sync_to_device();
    ngp_elem_euchromatin_perform_state_change_field.sync_to_device();
    ngp_elem_constraint_perform_state_change_field.sync_to_device();
    ngp_elem_constraint_state_change_rate_field.sync_to_device();
    ngp_elem_constraint_potential_force_field.sync_to_device();

    ngp_node_velocity_field.set_all(ngp_mesh, 0.0);
    ngp_node_force_field.set_all(ngp_mesh, 0.0);
    ngp_elem_binding_rates_field.set_all(ngp_mesh, 0.0);
    ngp_elem_unbinding_rates_field.set_all(ngp_mesh, 0.0);
    ngp_elem_perform_state_change_field.set_all(ngp_mesh, 0u);
    ngp_elem_euchromatin_perform_state_change_field.set_all(ngp_mesh, 0u);
    ngp_elem_constraint_perform_state_change_field.set_all(ngp_mesh, 0u);
    ngp_elem_constraint_state_change_rate_field.set_all(ngp_mesh, 0.0);
    ngp_elem_constraint_potential_force_field.set_all(ngp_mesh, 0.0);

    ngp_node_velocity_field.modify_on_device();
    ngp_node_force_field.modify_on_device();
    ngp_elem_binding_rates_field.modify_on_device();
    ngp_elem_unbinding_rates_field.modify_on_device();
    ngp_elem_perform_state_change_field.modify_on_device();
    ngp_elem_euchromatin_perform_state_change_field.modify_on_device();
    ngp_elem_constraint_perform_state_change_field.modify_on_device();
    ngp_elem_constraint_state_change_rate_field.modify_on_device();
    ngp_elem_constraint_potential_force_field.modify_on_device();

    rotate_field_states();  // TODO(palmerb4): Add "old" fields where necessary

    //////////////////////
    // Detect neighbors //
    //////////////////////
    update_neighbor_list_ = false;
    mundy::geom::compute_aabb(ngp_mesh, neighbor_list_params.get<double>("skin_distance"),

  }
}

// Forward declare the enums and declare their fmt formatters.
class HP1 {
 public:
  HP1() = default;

  void detect_neighbors() {
    Kokkos::Profiling::pushRegion("HP1::detect_neighbors");

    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto h_selector = stk::mesh::Selector(*h_part_ptr_);
    auto bs_selector = stk::mesh::Selector(*bs_part_ptr_);

    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);
    auto hp1_h_neighbor_genx_selector = stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_);
    auto hp1_bs_neighbor_genx_selector = stk::mesh::Selector(*hp1_bs_neighbor_genx_part_ptr_);

    // ComputeAABB for everybody at each time step. The accumulator uses this updated information to
    // calculate if we need to update the entire neighbor list.
    compute_aabb_ptr_->execute(backbone_segments_selector | hp1_selector | h_selector | bs_selector);
    update_accumulators();

    // Check if we need to update the neighbor list. Eventually this will be replaced with a mesh attribute to
    // synchronize across multiple tasks. For now, make sure that the default is to not update neighbor lists.
    check_update_neighbor_list();

    // Now do a check to see if we need to update the neighbor list.
    if (((force_neighborlist_update_) && (timestep_index_ % force_neighborlist_update_nsteps_ == 0)) ||
        update_neighbor_list_) {
      // Read off the timing information before doing anything else and reset it
      auto elapsed_steps = timestep_index_ - last_neighborlist_update_step_;
      auto elapsed_time = neighborlist_update_timer_.seconds();
      neighborlist_update_steps_times_.push_back(std::make_tuple(timestep_index_, elapsed_steps, elapsed_time));
      last_neighborlist_update_step_ = timestep_index_;
      neighborlist_update_timer_.reset();

      // Reset the accumulators
      mundy::mesh::utils::fill_field_with_value<double>(*element_corner_displacement_field_ptr_,
                                                        std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
      // Update the neighbor list
      if (enable_backbone_collision_ || enable_crosslinkers_ || enable_periphery_binding_) {
        destroy_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector | hp1_h_neighbor_genx_selector |
                                               hp1_bs_neighbor_genx_selector);
        ghost_linked_entities();
      }

      // Generate the GENX neighbor linkers
      if (enable_backbone_collision_) {
        generate_scs_scs_genx_ptr_->execute(backbone_segments_selector, backbone_segments_selector);
        ghost_linked_entities();
      }
      if (enable_crosslinkers_) {
        generate_hp1_h_genx_ptr_->execute(hp1_selector, h_selector);
        ghost_linked_entities();
      }
      if (enable_periphery_binding_) {
        generate_hp1_bs_genx_ptr_->execute(hp1_selector, bs_selector);
        ghost_linked_entities();
      }

      // Destroy linkers along backbone chains
      if (enable_backbone_collision_) {
        destroy_bound_neighbor_linkers_ptr_->execute(backbone_backbone_neighbor_genx_selector);
        ghost_linked_entities();
      }
    }

    Kokkos::Profiling::popRegion();
  }

  void update_accumulators() {
    Kokkos::Profiling::pushRegion("HP1::update_accumulators");

    // Selectors and aliases
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_aabb_field_old = element_aabb_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;

    stk::mesh::Selector combined_selector = spheres_selector | backbone_segments_selector | hp1_selector;

    // Update the accumulators based on the difference to the previous state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, combined_selector,
        [&element_aabb_field, &element_aabb_field_old, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_aabb = stk::mesh::field_data(element_aabb_field, aabb_entity);
          double *element_aabb_old = stk::mesh::field_data(element_aabb_field_old, aabb_entity);
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Add the (new_aabb - old_aabb) to the corner displacement
          element_corner_displacement[0] += element_aabb[0] - element_aabb_old[0];
          element_corner_displacement[1] += element_aabb[1] - element_aabb_old[1];
          element_corner_displacement[2] += element_aabb[2] - element_aabb_old[2];
          element_corner_displacement[3] += element_aabb[3] - element_aabb_old[3];
          element_corner_displacement[4] += element_aabb[4] - element_aabb_old[4];
          element_corner_displacement[5] += element_aabb[5] - element_aabb_old[5];
        });

    Kokkos::Profiling::popRegion();
  }

  void check_update_neighbor_list() {
    Kokkos::Profiling::pushRegion("HP1::check_update_neighbor_list");

    // Local variable for if we should update the neighbor list (do as an integer for now because MPI doesn't like
    // bools)
    int local_update_neighbor_list_int = 0;

    // Selectors and aliases
    auto spheres_selector = stk::mesh::Selector(*spheres_part_ptr_);
    auto backbone_segments_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);

    stk::mesh::Field<double> &element_corner_displacement_field = *element_corner_displacement_field_ptr_;
    const double skin_distance2_over4 = 0.25 * skin_distance_ * skin_distance_;

    stk::mesh::Selector combined_selector = spheres_selector | backbone_segments_selector | hp1_selector;

    // Check if each corner has moved skin_distance/2. Or, if dr_mag2 >= skin_distance^2/4
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, combined_selector,
        [&local_update_neighbor_list_int, &skin_distance2_over4, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Compute dr2 for each corner
          double dr2_corner0 = element_corner_displacement[0] * element_corner_displacement[0] +
                               element_corner_displacement[1] * element_corner_displacement[1] +
                               element_corner_displacement[2] * element_corner_displacement[2];
          double dr2_corner1 = element_corner_displacement[3] * element_corner_displacement[3] +
                               element_corner_displacement[4] * element_corner_displacement[4] +
                               element_corner_displacement[5] * element_corner_displacement[5];

          if (dr2_corner0 >= skin_distance2_over4 || dr2_corner1 >= skin_distance2_over4) {
            local_update_neighbor_list_int = 1;
          }
        });

    // Communicate local_update_neighbor_list to all ranks. Convert to an integer first (MPI doesn't handle booleans
    // well).
    int global_update_neighbor_list_int = 0;
    MPI_Allreduce(&local_update_neighbor_list_int, &global_update_neighbor_list_int, 1, MPI_INT, MPI_LOR,
                  MPI_COMM_WORLD);
    // Convert back to the boolean for the global version and or it with the original value (in case somebody else set
    // the neighbor list update 'signal').
    update_neighbor_list_ = update_neighbor_list_ || (global_update_neighbor_list_int == 1);

    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for left-bound crosslinkers binding to a sphere
  void compute_z_partition_left_bound() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition_left_bound");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &constraint_state_change_probability = *constraint_state_change_rate_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_constant = *element_spring_constant_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_spring_r0 = *element_spring_r0_field_ptr_;
    const mundy::linkers::LinkedEntitiesFieldType &constraint_linked_entities_field =
        *constraint_linked_entities_field_ptr_;
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;
    stk::mesh::Part &hp1_h_neighbor_genx_part = *hp1_h_neighbor_genx_part_ptr_;
    const double crosslinker_right_binding_rate = crosslinker_right_binding_rate_;
    const double inv_kt = 1.0 / crosslinker_kt_;

    const auto &crosslinker_spring_type = crosslinker_spring_type_;

    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, hp1_h_neighbor_genx_part,
        [&node_coord_field, &constraint_linked_entities_field, &constraint_state_change_probability,
         &crosslinker_spring_constant, &crosslinker_spring_r0, &left_hp1_part, &inv_kt, &crosslinker_right_binding_rate,
         &crosslinker_spring_type]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                   const stk::mesh::Entity &neighbor_genx) {
          // Get the sphere and crosslinker attached to the linker.
          const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
              stk::mesh::field_data(constraint_linked_entities_field, neighbor_genx));
          const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
          const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);

          MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker), std::invalid_argument,
                             "Encountered invalid crosslinker entity in compute_z_partition_left_bound_harmonic.");
          MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere), std::invalid_argument,
                             "Encountered invalid sphere entity in compute_z_partition_left_bound_harmonic.");

          // We need to figure out if this is a self-interaction or not. Since we are a left-bound crosslinker.
          const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];
          bool is_self_interaction = false;
          if (bulk_data.bucket(crosslinker).member(left_hp1_part)) {
            is_self_interaction = bulk_data.begin_nodes(crosslinker)[0] == sphere_node;
          }

          // Only act on the left-bound crosslinkers
          if (bulk_data.bucket(crosslinker).member(left_hp1_part) && !is_self_interaction) {
            const auto dr = mundy::mesh::vector3_field_data(node_coord_field, sphere_node) -
                            mundy::mesh::vector3_field_data(node_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
            const double dr_mag = mundy::math::norm(dr);

            // Compute the Z-partition score
            if (crosslinker_spring_type == BOND_TYPE::HARMONIC) {
              // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
              // A = crosslinker_binding_rates
              // k = crosslinker_spring_constant
              // r0 = crosslinker_spring_rest_length
              const double A = crosslinker_right_binding_rate;
              const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
              const double r0 = stk::mesh::field_data(crosslinker_spring_r0, crosslinker)[0];
              double Z = A * std::exp(-0.5 * inv_kt * k * (dr_mag - r0) * (dr_mag - r0));
              stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
            } else if (crosslinker_spring_type == BOND_TYPE::FENE) {
              // Z = A * (1 - (r/r0)^2)^(0.5 * 1/kt * k * r0^2)
              // A = crosslinker_binding_rates
              // k = crosslinker_spring_constant
              // r0 = crosslinker_spring_max_length (FENE)
              // R = crosslinker_fene_max_distance
              const double A = crosslinker_right_binding_rate;
              const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
              const double r0 = stk::mesh::field_data(crosslinker_spring_r0, crosslinker)[0];
              double Z = A * std::pow(1.0 - (dr_mag / r0) * (dr_mag / r0), 0.5 * inv_kt * k * r0 * r0);
              stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
            }
          }
        });

    if (enable_periphery_binding_) {
      const double periphery_binding_rate = periphery_binding_rate_;
      const double periphery_spring_constant = periphery_spring_constant_;
      const double periphery_spring_r0 = periphery_spring_r0_;
      stk::mesh::Part &hp1_bs_neighbor_genx_part = *hp1_bs_neighbor_genx_part_ptr_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, hp1_bs_neighbor_genx_part,
          [&node_coord_field, &constraint_linked_entities_field, &constraint_state_change_probability,
           &periphery_spring_constant, &periphery_spring_r0, &left_hp1_part, &inv_kt, &periphery_binding_rate,
           &crosslinker_spring_type]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                     const stk::mesh::Entity &neighbor_genx) {
            // Get the sphere and crosslinker attached to the linker.
            const stk::mesh::EntityKey::entity_key_t *key_t_ptr =
                reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
                    stk::mesh::field_data(constraint_linked_entities_field, neighbor_genx));
            const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
            const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);
            const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];

            MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker), std::invalid_argument,
                               "Encountered invalid crosslinker entity in compute_z_partition_left_bound_harmonic.");
            MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere), std::invalid_argument,
                               "Encountered invalid sphere entity in compute_z_partition_left_bound_harmonic.");

            // Only act on the left-bound crosslinkers
            if (bulk_data.bucket(crosslinker).member(left_hp1_part)) {
              const auto dr = mundy::mesh::vector3_field_data(node_coord_field, sphere_node) -
                              mundy::mesh::vector3_field_data(node_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
              const double dr_mag = mundy::math::norm(dr);

              // Compute the Z-partition score
              if (crosslinker_spring_type == BOND_TYPE::HARMONIC) {
                // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
                // A = periphery_binding_rate
                // k = periphery_spring_constant
                // r0 = periphery_spring_r0
                const double A = periphery_binding_rate;
                const double k = periphery_spring_constant;
                const double r0 = periphery_spring_r0;
                double Z = A * std::exp(-0.5 * inv_kt * k * (dr_mag - r0) * (dr_mag - r0));
                stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
              } else if (crosslinker_spring_type == BOND_TYPE::FENE) {
                // Z = A * (1 - (r/r0)^2)^(0.5 * 1/kt * k * r0^2)
                // A = periphery_binding_rate
                // k = periphery_spring_constant
                // r0 = periphery_spring_r0
                const double A = periphery_binding_rate;
                const double k = periphery_spring_constant;
                const double r0 = periphery_spring_r0;
                double Z = A * std::pow(1.0 - (dr_mag / r0) * (dr_mag / r0), 0.5 * inv_kt * k * r0 * r0);
                stk::mesh::field_data(constraint_state_change_probability, neighbor_genx)[0] = Z;
              }
            }
          });
    }
    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function score for doubly_bound crosslinkers
  void compute_z_partition_doubly_bound() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition_doubly_bound");

    // Selectors and aliases
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;
    const double &crosslinker_right_unbinding_rate = crosslinker_right_unbinding_rate_;

    // Loop over the neighbor list of the crosslinkers, then select down to the ones that are left-bound only.
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_hp1_h_part,
        [&node_coord_field, &crosslinker_unbinding_rates, &doubly_hp1_h_part, &crosslinker_right_unbinding_rate](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
// This is a left-bound crosslinker, so just calculate the right unbinding rate and store on the crosslinker
// itself in the correct position.
#pragma todo This needs to have a different rate for the periphery versus hp1 type bindings
          stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1] = crosslinker_right_unbinding_rate;
        });

    Kokkos::Profiling::popRegion();
  }

  /// \brief Compute the Z-partition function for everybody
  void compute_z_partition() {
    Kokkos::Profiling::pushRegion("HP1::compute_z_partition");

    // Compute the left-bound to doubly-bound score
    // Works for both binding to an h-sphere and binding to a bs-sphere
    compute_z_partition_left_bound();

    // Compute the doubly-bound to left-bound score
    compute_z_partition_doubly_bound();

    Kokkos::Profiling::popRegion();
  }

  void kmc_crosslinker_left_to_doubly() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_left_to_doubly");

    // Selectors and aliases
    stk::mesh::Part &hp1_h_neighbor_genx_part = *hp1_h_neighbor_genx_part_ptr_;
    stk::mesh::Part &hp1_bs_neighbor_genx_part = *hp1_bs_neighbor_genx_part_ptr_;
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    stk::mesh::Field<unsigned> &constraint_perform_state_change_field = *constraint_perform_state_change_field_ptr_;
    stk::mesh::Field<double> &constraint_state_change_rate_field = *constraint_state_change_rate_field_ptr_;
    const mundy::linkers::LinkedEntitiesFieldType &constraint_linked_entities_field =
        *constraint_linked_entities_field_ptr_;
    const double timestep_size = timestep_size_;
    const double enable_periphery_binding = enable_periphery_binding_;
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;

    // Loop over left-bound crosslinkers and decide if they bind or not
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, left_hp1_part,
        [&hp1_h_neighbor_genx_part, &hp1_bs_neighbor_genx_part, &element_rng_field,
         &constraint_perform_state_change_field, &element_perform_state_change_field,
         &constraint_state_change_rate_field, &constraint_linked_entities_field, &timestep_size,
         &enable_periphery_binding]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                    const stk::mesh::Entity &crosslinker) {
          // Get all of my associated crosslinker_sphere_linkers
          const stk::mesh::Entity &any_arbitrary_crosslinker_node = bulk_data.begin_nodes(crosslinker)[0];
          const stk::mesh::Entity *neighbor_genx_linkers =
              bulk_data.begin(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);
          const unsigned num_neighbor_genx_linkers =
              bulk_data.num_connectivity(any_arbitrary_crosslinker_node, stk::topology::CONSTRAINT_RANK);

          // Loop over the attached crosslinker_sphere_linkers and bind if the rqng falls in their range.
          double z_tot = 0.0;
          for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
            const auto &constraint_rank_entity = neighbor_genx_linkers[j];
            const bool is_hp1_h_neighbor_genx =
                bulk_data.bucket(constraint_rank_entity).member(hp1_h_neighbor_genx_part);
            const bool is_hp1_bs_neighbor_genx =
                bulk_data.bucket(constraint_rank_entity).member(hp1_bs_neighbor_genx_part);
            if (is_hp1_h_neighbor_genx || (enable_periphery_binding && is_hp1_bs_neighbor_genx)) {
              const double z_i =
                  timestep_size * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
              z_tot += z_i;
            }
          }

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, crosslinker);
          const stk::mesh::EntityId crosslinker_gid = bulk_data.identifier(crosslinker);
          openrand::Philox rng(crosslinker_gid, element_rng_counter[0]);
          const double randu01 = rng.rand<double>();
          element_rng_counter[0]++;

          // Notice that the sum of all probabilities is 1.
          // The probability of nothing happening is
          //   std::exp(-z_tot)
          // The probability of an individual event happening is
          //   z_i / z_tot * (1 - std::exp(-z_tot))
          //
          // This is (by construction) true since
          //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
          //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
          //
          // This means that binding only happens if randu01 < (1 - std::exp(-z_tot))
          const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
          const double scale_factor = probability_of_no_state_change * timestep_size / z_tot;
          if (randu01 < (1.0 - std::exp(-z_tot))) {
            // Binding occurs.
            // Loop back over the neighbor linkers to see if one of them binds in the running sum

            double cumsum = 0.0;
            for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
              auto &constraint_rank_entity = neighbor_genx_linkers[j];
              bool is_hp1_h_neighbor_genx = bulk_data.bucket(constraint_rank_entity).member(hp1_h_neighbor_genx_part);
              if (is_hp1_h_neighbor_genx) {
                const double binding_probability =
                    scale_factor * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
                cumsum += binding_probability;
                if (randu01 < cumsum) {
                  // We have a binding event, set this, then bail on the for loop
                  // Store the state change on both the genx and the crosslinker
                  stk::mesh::field_data(constraint_perform_state_change_field, constraint_rank_entity)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  stk::mesh::field_data(element_perform_state_change_field, crosslinker)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  break;
                }
              }
            }
          }
        });

    Kokkos::Profiling::popRegion();
  }

  /// \brief Perform the binding of a crosslinker to a sphere (doubly to left)
  void kmc_crosslinker_doubly_to_left() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_doubly_to_left");

    // Selectors and aliases
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    const stk::mesh::Field<double> &crosslinker_unbinding_rates = *element_unbinding_rates_field_ptr_;
    const double &timestep_size = timestep_size_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;

    // This is just a loop over the doubly bound crosslinkers, since we know that the right head in is [1].
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_hp1_h_part,
        [&element_rng_field, &element_perform_state_change_field, &crosslinker_unbinding_rates, &timestep_size](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker) {
          // We only have a single node, our right node, that is bound that we can unbind.
          // TODO(cje): Right now this is coded to have a loop wrapping it, maybe not needed?
          const double unbinding_probability =
              timestep_size * stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1];
          double Z_tot = unbinding_probability;
          const double unbind_scale_factor = (1.0 - exp(-Z_tot)) * timestep_size;

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, crosslinker);
          const stk::mesh::EntityId crosslinker_gid = bulk_data.identifier(crosslinker);
          openrand::Philox rng(crosslinker_gid, element_rng_counter[0]);
          double randZ = rng.rand<double>() * Z_tot;
          double cumsum = 0.0;
          element_rng_counter[0]++;

          // Now check the cummulative sum and if less than perform the unbinding
          cumsum += unbind_scale_factor * stk::mesh::field_data(crosslinker_unbinding_rates, crosslinker)[1];
          if (randZ < cumsum) {
            // Set the state change on the element
            stk::mesh::field_data(element_perform_state_change_field, crosslinker)[0] =
                static_cast<unsigned>(BINDING_STATE_CHANGE::DOUBLY_TO_LEFT);
          }
        });

    Kokkos::Profiling::popRegion();
  }

  void kmc_crosslinker_sphere_linker_sampling() {
    Kokkos::Profiling::pushRegion("HP1::kmc_crosslinker_sphere_linker_sampling");

    // Perform the left to doubly bound crosslinker binding calc
    kmc_crosslinker_left_to_doubly();

    // Perform the doubly to left bound crosslinker binding calc
    kmc_crosslinker_doubly_to_left();

    // At this point, constraint_state_change_rate_field is only up-to-date for locally-owned entities. We need
    // to communicate this information to all other processors.
    stk::mesh::communicate_field_data(
        *bulk_data_ptr_, {element_perform_state_change_field_ptr_, constraint_perform_state_change_field_ptr_});

    Kokkos::Profiling::popRegion();
  }

  /// \brief Perform the state change of the crosslinkers
  void state_change_crosslinkers() {
    Kokkos::Profiling::pushRegion("HP1::state_change_crosslinkers");

    // Loop over both the CROSSLINKER_SPHERE_LINKERS and the CROSSLINKERS to perform the state changes.
    stk::mesh::Part &left_hp1_part = *left_hp1_part_ptr_;
    stk::mesh::Part &doubly_hp1_h_part = *doubly_hp1_h_part_ptr_;

    // Get the vector of entities to modify
    stk::mesh::EntityVector hp1_h_neighbor_genxs;
    stk::mesh::EntityVector doubly_bound_hp1s;
    stk::mesh::get_selected_entities(stk::mesh::Selector(*hp1_h_neighbor_genx_part_ptr_),
                                     bulk_data_ptr_->buckets(stk::topology::CONSTRAINT_RANK), hp1_h_neighbor_genxs);
    stk::mesh::get_selected_entities(stk::mesh::Selector(*doubly_hp1_h_part_ptr_),
                                     bulk_data_ptr_->buckets(stk::topology::ELEMENT_RANK), doubly_bound_hp1s);

    // TODO(cje): It might be worth checking to see if we have any state changes in any threads before we crack open the
    // modification section, as even doing that is slightly expensive.

    bulk_data_ptr_->modification_begin();

    // Perform L->D
    for (const stk::mesh::Entity &hp1_h_neighbor_genx : hp1_h_neighbor_genxs) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*constraint_perform_state_change_field_ptr_, hp1_h_neighbor_genx)[0]);
      const bool perform_state_change = state_change_action != BINDING_STATE_CHANGE::NONE;
      if (perform_state_change) {
        // Get our connections (as the genx)
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(*constraint_linked_entities_field_ptr_, hp1_h_neighbor_genx));
        const stk::mesh::Entity &crosslinker_hp1 = bulk_data_ptr_->get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &target_sphere = bulk_data_ptr_->get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(crosslinker_hp1), std::invalid_argument,
                           "Encountered invalid crosslinker entity in state_change_crosslinkers.");
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(target_sphere), std::invalid_argument,
                           "Encountered invalid sphere entity in state_change_crosslinkers.");

        const stk::mesh::Entity &target_sphere_node = bulk_data_ptr_->begin_nodes(target_sphere)[0];
        // Call the binding function
        if (state_change_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
          // Unbind the right side of the crosslinker from the left node and bind it to the target node
          const bool bind_worked = ::mundy::alens::crosslinkers::bind_crosslinker_to_node_unbind_existing(
              *bulk_data_ptr_, crosslinker_hp1, target_sphere_node, 1);
          MUNDY_THROW_ASSERT(bind_worked, std::logic_error, "Failed to bind crosslinker to node.");

          std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Binding crosslinker "
                    << bulk_data_ptr_->identifier(crosslinker_hp1) << " to node "
                    << bulk_data_ptr_->identifier(target_sphere_node) << std::endl;

          // Now change the part from left to doubly bound.
          const bool is_crosslinker_locally_owned =
              bulk_data_ptr_->parallel_owner_rank(crosslinker_hp1) == bulk_data_ptr_->parallel_rank();
          if (is_crosslinker_locally_owned) {
            auto add_parts = stk::mesh::PartVector{doubly_hp1_h_part_ptr_};
            auto remove_parts = stk::mesh::PartVector{left_hp1_part_ptr_};
            bulk_data_ptr_->change_entity_parts(crosslinker_hp1, add_parts, remove_parts);
          }
        }
      }
    }

    // Perform D->L
    for (const stk::mesh::Entity &crosslinker_hp1 : doubly_bound_hp1s) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(*element_perform_state_change_field_ptr_, crosslinker_hp1)[0]);
      if (state_change_action == BINDING_STATE_CHANGE::DOUBLY_TO_LEFT) {
        // Unbind the right side of the crosslinker from the current node and bind it to the left crosslinker node
        const stk::mesh::Entity &left_node = bulk_data_ptr_->begin_nodes(crosslinker_hp1)[0];
        const bool unbind_worked = ::mundy::alens::crosslinkers::bind_crosslinker_to_node_unbind_existing(
            *bulk_data_ptr_, crosslinker_hp1, left_node, 1);
        MUNDY_THROW_ASSERT(unbind_worked, std::logic_error, "Failed to unbind crosslinker from node.");

        std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Unbinding crosslinker "
                  << bulk_data_ptr_->identifier(crosslinker_hp1) << " from node "
                  << bulk_data_ptr_->identifier(bulk_data_ptr_->begin_nodes(crosslinker_hp1)[1]) << std::endl;

        // Now change the part from doubly to left bound.
        const bool is_crosslinker_locally_owned =
            bulk_data_ptr_->parallel_owner_rank(crosslinker_hp1) == bulk_data_ptr_->parallel_rank();
        if (is_crosslinker_locally_owned) {
          auto add_parts = stk::mesh::PartVector{left_hp1_part_ptr_};
          auto remove_parts = stk::mesh::PartVector{doubly_hp1_h_part_ptr_};
          bulk_data_ptr_->change_entity_parts(crosslinker_hp1, add_parts, remove_parts);
        }
      }
    }

    bulk_data_ptr_->modification_end();

    // The above may have invalidated the ghosting for our genx ghosting, so we need to reghost the linked entities to
    // any process that owns any of the other linked entities.
    ghost_linked_entities();

    Kokkos::Profiling::popRegion();
  }

  void update_crosslinker_state() {
    Kokkos::Profiling::pushRegion("HP1::update_crosslinker_state");

    // We want to loop over all LEFT_BOUND_CROSSLINKERS, RIGHT_BOUND_CROSSLINKERS, and DOUBLY_BOUND_CROSSLINKERS to
    // generate state changes. This is done to build up a list of actions that we will take later during a mesh
    // modification step.
    {
      compute_z_partition();
      kmc_crosslinker_sphere_linker_sampling();
    }

    // Loop over the different crosslinkers, look at their actions, and enforce the state change.
    {
      // Call the global state change function
      state_change_crosslinkers();
    }

    Kokkos::Profiling::popRegion();
  }

  void active_euchromatin_sampling() {
    Kokkos::Profiling::pushRegion("HP1::active_euchromatin_sampling");

    // Selectors and aliases
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Field<unsigned> &element_rng_field = *element_rng_field_ptr_;
    stk::mesh::Field<unsigned> &euchromatin_state = *euchromatin_state_field_ptr_;
    stk::mesh::Field<unsigned> &euchromatin_perform_state_change = *euchromatin_perform_state_change_field_ptr_;
    stk::mesh::Field<double> &euchromatin_state_change_next_time = *euchromatin_state_change_next_time_field_ptr_;
    stk::mesh::Field<double> &euchromatin_state_change_elapsed_time = *euchromatin_state_change_elapsed_time_field_ptr_;

    const double &timestep_size = timestep_size_;
    double kon_inv = 1.0 / active_euchromatin_force_kon_;
    double koff_inv = 1.0 / active_euchromatin_force_koff_;

    // Loop over the euchromatin spring elements and decide if they switch to the active state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, ee_springs_part,
        [&element_rng_field, &euchromatin_state, &euchromatin_perform_state_change, &euchromatin_state_change_next_time,
         &euchromatin_state_change_elapsed_time, &kon_inv, &koff_inv](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &euchromatin_spring) {
          // We are not going to increment the elapsed time ourselves, but rely on someone outside of this loop to do
          // that at the end of a timestpe, in order to keep it consistent with the total elapsed time in the system.
          unsigned *current_state = stk::mesh::field_data(euchromatin_state, euchromatin_spring);
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, euchromatin_spring);
          double *next_time = stk::mesh::field_data(euchromatin_state_change_next_time, euchromatin_spring);
          double *elapsed_time = stk::mesh::field_data(euchromatin_state_change_elapsed_time, euchromatin_spring);

          if (elapsed_time[0] >= next_time[0]) {
            // Need a random number no matter what
            const stk::mesh::EntityId euchromatin_spring_gid = bulk_data.identifier(euchromatin_spring);
            openrand::Philox rng(euchromatin_spring_gid, element_rng_counter[0]);
            const double randu01 = rng.rand<double>();
            element_rng_counter[0]++;

            // Determine switch based on current state
            if (current_state[0] == 0u) {
              // Currently inactive, set to active and reset the timers
              current_state[0] = 1u;
              next_time[0] = -std::log(randu01) * koff_inv;
              elapsed_time[0] = 0.0;
            } else {
              // Currently active, set to active and reset the timers
              current_state[0] = 0u;
              next_time[0] = -std::log(randu01) * kon_inv;
              elapsed_time[0] = 0.0;
            }

#pragma omp critical
            {
              const unsigned previous_state = current_state[0] == 0u ? 1u : 0u;
              std::cout << "Rank" << stk::parallel_machine_rank(MPI_COMM_WORLD)
                        << " Detected euchromatin switching event object " << bulk_data.identifier(euchromatin_spring)
                        << ", previous state: " << previous_state << ", current_state: " << current_state[0]
                        << std::endl;
              std::cout << "  next_time: " << next_time[0] << ", elapsed_time: " << elapsed_time[0] << std::endl;
            }
          }
        });
    Kokkos::Profiling::popRegion();
  }

  void update_euchromatin_state_time() {
    Kokkos::Profiling::pushRegion("HP1::active_euchromatin_sampling");

    // Selectors and aliases
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Field<double> &euchromatin_state_change_elapsed_time = *euchromatin_state_change_elapsed_time_field_ptr_;
    const double &timestep_size = timestep_size_;

    // Loop over the euchromatin spring elements and decide if they switch to the active state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, ee_springs_part,
        [&euchromatin_state_change_elapsed_time, &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                                 const stk::mesh::Entity &euchromatin_spring) {
          // Updated the elapsed time
          stk::mesh::field_data(euchromatin_state_change_elapsed_time, euchromatin_spring)[0] += timestep_size;
        });

    Kokkos::Profiling::popRegion();
  }

  void update_active_euchromatin_state() {
    Kokkos::Profiling::pushRegion("HP1::update_active_euchromatin_state");

    // Determine if we need to update the euchromatin active state in the same way as the crosslinkers,
    active_euchromatin_sampling();
    // active_euchromatin_state_change();

    Kokkos::Profiling::popRegion();
  }

  void check_maximum_overlap_with_hydro_periphery() {
    if (periphery_hydro_shape_ == PERIPHERY_SHAPE::SPHERE) {
      const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
      stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
      stk::mesh::Field<double> &element_hydro_radius_field = *element_radius_field_ptr_;
      double shifted_periphery_hydro_radius = periphery_hydro_radius_ + maximum_allowed_periphery_overlap_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
          [&node_coord_field, &element_hydro_radius_field, &shifted_periphery_hydro_radius](
              const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            const double sphere_radius = stk::mesh::field_data(element_hydro_radius_field, sphere_element)[0];
            const bool overlap_exceeds_threshold =
                mundy::math::norm(node_coords) + sphere_radius > shifted_periphery_hydro_radius;
            if (overlap_exceeds_threshold) {
#pragma omp critical
              {
                std::cout << "Sphere node " << bulk_data.identifier(sphere_node)
                          << " overlaps with the periphery more than the allowable threshold." << std::endl;
                std::cout << "  node_coords: " << node_coords << std::endl;
                std::cout << "  norm(node_coords): " << mundy::math::norm(node_coords) << std::endl;
              }
              MUNDY_THROW_REQUIRE(false, std::runtime_error, "Sphere node outside hydrodynamic periphery.");
            }
          });
    } else if (periphery_hydro_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
      const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
      stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
      stk::mesh::Field<double> &element_hydro_radius_field = *element_radius_field_ptr_;
      double shifted_periphery_axis_radius1 = periphery_hydro_axis_radius1_ + maximum_allowed_periphery_overlap_;
      double shifted_periphery_axis_radius2 = periphery_hydro_axis_radius2_ + maximum_allowed_periphery_overlap_;
      double shifted_periphery_axis_radius3 = periphery_hydro_axis_radius3_ + maximum_allowed_periphery_overlap_;

      mundy::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
          [&node_coord_field, &element_hydro_radius_field, &shifted_periphery_axis_radius1,
           &shifted_periphery_axis_radius2, &shifted_periphery_axis_radius3](const stk::mesh::BulkData &bulk_data,
                                                                             const stk::mesh::Entity &sphere_element) {
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            const double sphere_radius = stk::mesh::field_data(element_hydro_radius_field, sphere_element)[0];

            // The following is an in-exact but cheap check.
            // If shrinks the periphery by the maximum allowed overlap and the sphere radius and then checks if the
            // sphere is inside the shrunk periphery. Level sets don't follow the same rules as Euclidean geometry, so
            // this is a rough check.
            const double x = node_coords[0];
            const double y = node_coords[1];
            const double z = node_coords[2];
            const double x2 = x * x;
            const double y2 = y * y;
            const double z2 = z * z;
            const double a2 =
                (shifted_periphery_axis_radius1 - sphere_radius) * (shifted_periphery_axis_radius1 - sphere_radius);
            const double b2 =
                (shifted_periphery_axis_radius2 - sphere_radius) * (shifted_periphery_axis_radius2 - sphere_radius);
            const double c2 =
                (shifted_periphery_axis_radius3 - sphere_radius) * (shifted_periphery_axis_radius3 - sphere_radius);
            const double value = x2 / a2 + y2 / b2 + z2 / c2;
            if (value > 1.0) {
#pragma omp critical
              {
                std::cout << "Sphere node " << bulk_data.identifier(sphere_node)
                          << " overlaps with the periphery more than the allowable threshold." << std::endl;
                std::cout << "  node_coords: " << node_coords << std::endl;
                std::cout << "  value: " << value << std::endl;
              }
              MUNDY_THROW_REQUIRE(false, std::runtime_error, "Sphere node outside hydrodynamic periphery.");
            }
          });
    } else {
      MUNDY_THROW_REQUIRE(false, std::logic_error, "Invalid periphery type.");
    }
  }

  void compute_rpy_hydro() {
    // Before performing the hydro call, check if the spheres are within the periphery (optional)
    if (check_maximum_periphery_overlap_) {
      check_maximum_overlap_with_hydro_periphery();
    }

    Kokkos::Profiling::pushRegion("HP1::compute_rpy_hydro");
    const double viscosity = viscosity_;

    // Fetch the bucket of spheres to act on.
    stk::mesh::EntityVector sphere_elements;
    stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::get_selected_entities(chromatin_spheres_selector, bulk_data_ptr_->buckets(stk::topology::ELEMENT_RANK),
                                     sphere_elements);
    const size_t num_spheres = sphere_elements.size();

    // Copy the sphere positions, radii, forces, and velocities to Kokkos views
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions("sphere_positions", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii("sphere_radii", num_spheres);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces("sphere_forces", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities("sphere_velocities",
                                                                                    num_spheres * 3);

#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      const double *sphere_position = stk::mesh::field_data(*node_coord_field_ptr_, sphere_node);
      const double *sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere_element);
      const double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      const double *sphere_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_positions(i * 3 + j) = sphere_position[j];
        sphere_forces(i * 3 + j) = sphere_force[j];
        sphere_velocities(i * 3 + j) = sphere_velocity[j];
      }
      sphere_radii(i) = *sphere_radius;
    }

    // Apply the RPY kernel from spheres to spheres
    mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, sphere_positions,
                                              sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

    // If enabled, apply the correction for the no-slip boundary condition
    if (enable_periphery_hydrodynamics_) {
      const size_t num_surface_nodes = periphery_ptr_->get_num_nodes();
      auto surface_positions = periphery_ptr_->get_surface_positions();
      auto surface_weights = periphery_ptr_->get_quadrature_weights();
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_radii("surface_radii", num_surface_nodes);
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_velocities("surface_velocities",
                                                                                       3 * num_surface_nodes);
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_forces("surface_forces",
                                                                                   3 * num_surface_nodes);
      Kokkos::deep_copy(surface_radii, 0.0);

      // Apply the RPY kernel from spheres to periphery
      mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, surface_positions,
                                                sphere_radii, surface_radii, sphere_forces, surface_velocities);

      // Apply no-slip boundary conditions
      // This is done in two steps: first, we compute the forces on the periphery necessary to enforce no-slip
      // Then we evaluate the flow these forces induce on the spheres.
      periphery_ptr_->compute_surface_forces(surface_velocities, surface_forces);

      // // If we evaluate the flow these forces induce on the periphery, do they satisfy no-slip?
      // Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> M("Mnew", 3 * num_surface_nodes,
      //                                                                  3 * num_surface_nodes);
      // fill_skfie_matrix(DeviceExecutionSpace(), viscosity, num_surface_nodes, num_surface_nodes, surface_positions,
      //                   surface_positions, surface_normals, surface_weights, M);
      // KokkosBlas::gemv(DeviceExecutionSpace(), "N", 1.0, M, surface_forces, 1.0, surface_velocities);
      // EXPECT_NEAR(max_speed(surface_velocities), 0.0, 1.0e-10);

      mundy::alens::periphery::apply_weighted_stokes_kernel(DeviceExecutionSpace(), viscosity, surface_positions,
                                                            sphere_positions, surface_forces, surface_weights,
                                                            sphere_velocities);

      // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
      mundy::alens::periphery::apply_local_drag(DeviceExecutionSpace(), viscosity, sphere_velocities, sphere_forces,
                                                sphere_radii);
    }

    // Copy the sphere forces and velocities back to STK fields
#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      double *sphere_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_force[j] = sphere_forces(i * 3 + j);
        sphere_velocity[j] = sphere_velocities(i * 3 + j);
      }
    }
    Kokkos::Profiling::popRegion();
  }

  void compute_ellipsoidal_periphery_collision_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_ellipsoidal_periphery_collision_forces");
    const double spring_constant = periphery_collision_spring_constant_;
    const double a = periphery_collision_axis_radius1_;
    const double b = periphery_collision_axis_radius2_;
    const double c = periphery_collision_axis_radius3_;
    const double inv_a2 = 1.0 / (a * a);
    const double inv_b2 = 1.0 / (b * b);
    const double inv_c2 = 1.0 / (c * c);
    const mundy::math::Vector3d center(0.0, 0.0, 0.0);
    const auto orientation = mundy::math::Quaterniond::identity();
    auto level_set = [&inv_a2, &inv_b2, &inv_c2, &center,
                      &orientation](const mundy::math::Vector3d &point) -> double {
      // const auto body_frame_point = conjugate(orientation) * (point - center);
      const auto body_frame_point = point - center;
      return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
              body_frame_point[2] * body_frame_point[2] * inv_c2) -
             1;
    };

    // Fetch local references to the fields
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_force_field, &element_aabb_field, &element_radius_field, &level_set, &center,
         &orientation, &a, &b, &c,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          // For our coarse search, we check if the coners of the sphere's aabb lie inside the ellipsoidal periphery
          // This can be done via the (body frame) inside outside unftion f(x, y, z) = 1 - (x^2/a^2 + y^2/b^2 + z^2/c^2)
          // This is possible due to the convexity of the ellipsoid
          const double *sphere_aabb = stk::mesh::field_data(element_aabb_field, sphere_element);
          const double x0 = sphere_aabb[0];
          const double y0 = sphere_aabb[1];
          const double z0 = sphere_aabb[2];
          const double x1 = sphere_aabb[3];
          const double y1 = sphere_aabb[4];
          const double z1 = sphere_aabb[5];

          // Compute all 8 corners of the AABB
          const auto bottom_left_front = mundy::math::Vector3d(x0, y0, z0);
          const auto bottom_right_front = mundy::math::Vector3d(x1, y0, z0);
          const auto top_left_front = mundy::math::Vector3d(x0, y1, z0);
          const auto top_right_front = mundy::math::Vector3d(x1, y1, z0);
          const auto bottom_left_back = mundy::math::Vector3d(x0, y0, z1);
          const auto bottom_right_back = mundy::math::Vector3d(x1, y0, z1);
          const auto top_left_back = mundy::math::Vector3d(x0, y1, z1);
          const auto top_right_back = mundy::math::Vector3d(x1, y1, z1);
          const double all_points_inside_periphery =
              level_set(bottom_left_front) < 0.0 && level_set(bottom_right_front) < 0.0 &&
              level_set(top_left_front) < 0.0 && level_set(top_right_front) < 0.0 &&
              level_set(bottom_left_back) < 0.0 && level_set(bottom_right_back) < 0.0 &&
              level_set(top_left_back) < 0.0 && level_set(top_right_back) < 0.0;

          if (!all_points_inside_periphery) {
            // We might have a collision, perform the more expensive check
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

            // Note, the ellipsoid for the ssd calc has outward normal, whereas the periphery has inward normal.
            // Hence, the sign flip.
            mundy::math::Vector3d contact_point;
            mundy::math::Vector3d ellipsoid_nhat;
            const double shared_normal_ssd =
                -mundy::math::distance::shared_normal_ssd_between_ellipsoid_and_point(
                    center, orientation, a, b, c, node_coords, contact_point, ellipsoid_nhat) -
                sphere_radius;

            if (shared_normal_ssd < 0.0) {
              // We have a collision, compute the force
              auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);
              auto periphery_nhat = -ellipsoid_nhat;
              node_force[0] -= spring_constant * periphery_nhat[0] * shared_normal_ssd;
              node_force[1] -= spring_constant * periphery_nhat[1] * shared_normal_ssd;
              node_force[2] -= spring_constant * periphery_nhat[2] * shared_normal_ssd;
            }
          }
        });
    Kokkos::Profiling::popRegion();
  }

  void compute_ellipsoidal_periphery_collision_forces_fast_approximate() {
    Kokkos::Profiling::pushRegion("HP1::compute_ellipsoidal_periphery_collision_forces_fast_approximate");
    const double spring_constant = periphery_collision_spring_constant_;
    // Adjust for our standoff distance
    const double a = periphery_collision_axis_radius1_;
    const double b = periphery_collision_axis_radius2_;
    const double c = periphery_collision_axis_radius3_;
    const mundy::math::Vector3d center(0.0, 0.0, 0.0);
    const auto orientation = mundy::math::Quaterniond::identity();
    auto level_set = [&a, &b, &c, &center, &orientation](const double &radius,
                                                         const mundy::math::Vector3d &point) -> double {
      // const auto body_frame_point = conjugate(orientation) * (point - center);
      const auto body_frame_point = point - center;
      const double inv_a2 = 1.0 / ((a - radius) * (a - radius));
      const double inv_b2 = 1.0 / ((b - radius) * (b - radius));
      const double inv_c2 = 1.0 / ((c - radius) * (c - radius));
      return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
              body_frame_point[2] * body_frame_point[2] * inv_c2) -
             1;
    };
    // Fast compute of the outward 'normal' at the point
    auto outward_normal = [&a, &b, &c, &center, &orientation](
                              const double &radius,
                              const mundy::math::Vector3d &point) -> mundy::math::Vector3d {
      const auto body_frame_point = point - center;
      const double inv_a2 = 1.0 / ((a - radius) * (a - radius));
      const double inv_b2 = 1.0 / ((b - radius) * (b - radius));
      const double inv_c2 = 1.0 / ((c - radius) * (c - radius));
      return mundy::math::Vector3d(2.0 * body_frame_point[0] * inv_a2, 2.0 * body_frame_point[1] * inv_b2,
                                          2.0 * body_frame_point[2] * inv_c2);
    };

    // Fetch local references to the fields
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_force_field, &element_radius_field, &level_set, &outward_normal, &center,
         &orientation, &a, &b, &c,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          // Do a fast loop over all of the spheres we are checking, e.g., brute-force the calc.
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

          // Simply check if we are outside the sphere via the level-set function
          if (level_set(sphere_radius, node_coords) > 0.0) {
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);

            // Compute the outward normal
            auto out_normal = outward_normal(sphere_radius, node_coords);
            node_force[0] -= spring_constant * out_normal[0];
            node_force[1] -= spring_constant * out_normal[1];
            node_force[2] -= spring_constant * out_normal[2];
          }
        });
    Kokkos::Profiling::popRegion();
  }

  void compute_spherical_periphery_collision_forces() {
    const double spring_constant = periphery_collision_spring_constant_;
    const double periphery_collision_radius = periphery_collision_radius_;

    // Fetch local references to the fields
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_force_field, &element_aabb_field, &element_radius_field, &periphery_collision_radius,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);

          const double node_coords_norm = mundy::math::two_norm(node_coords);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
          const double shared_normal_ssd = periphery_collision_radius - node_coords_norm - sphere_radius;
          const bool sphere_collides_with_periphery = shared_normal_ssd < 0.0;
          if (sphere_collides_with_periphery) {
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);
            auto inward_normal = -node_coords / node_coords_norm;
            node_force[0] -= spring_constant * inward_normal[0] * shared_normal_ssd;
            node_force[1] -= spring_constant * inward_normal[1] * shared_normal_ssd;
            node_force[2] -= spring_constant * inward_normal[2] * shared_normal_ssd;
          }
        });
  }

  void compute_periphery_collision_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_periphery_collision_forces");
    if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
      compute_spherical_periphery_collision_forces();
    } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
      if (periphery_collision_use_fast_approx_) {
        compute_ellipsoidal_periphery_collision_forces_fast_approximate();
      } else {
        compute_ellipsoidal_periphery_collision_forces();
      }
    } else {
      MUNDY_THROW_REQUIRE(false, std::logic_error, "Invalid periphery type.");
    }
    Kokkos::Profiling::popRegion();
  }

  void compute_euchromatin_active_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_euchromatin_active_forces");

    // We are going to do the forces as such.
    // nhat is the unit director along the segment.
    // sigma is the force density we are applying
    // F = f nhat
    // sigma = f * n --> f = sigma / n
    // F = sigma / n * nhat

    // Selectors and aliases
    stk::mesh::Part &ee_springs_part = *ee_springs_part_ptr_;
    stk::mesh::Field<unsigned> &euchromatin_state = *euchromatin_state_field_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    const double &active_force_sigma = active_euchromatin_force_sigma_;

    // Loop over the euchromatin spring elements and decide if they switch to the active state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, ee_springs_part,
        [&euchromatin_state, &node_coord_field, &node_force_field, &active_force_sigma](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &euchromatin_spring) {
          // We are not going to increment the elapsed time ourselves, but rely on someone outside of this loop to do
          // that at the end of a timestpe, in order to keep it consistent with the total elapsed time in the system.
          unsigned *current_state = stk::mesh::field_data(euchromatin_state, euchromatin_spring);

          if (current_state[0] == 1u) {
            // Fetch the connected nodes
            const stk::mesh::Entity *nodes = bulk_data.begin_nodes(euchromatin_spring);
            const stk::mesh::Entity &node1 = nodes[0];
            const stk::mesh::Entity &node2 = nodes[1];
            const double *node1_coord = stk::mesh::field_data(node_coord_field, node1);
            const double *node2_coord = stk::mesh::field_data(node_coord_field, node2);

            // Calculate the force on each node from the above equation, which winds up
            // F = sigma / n / n * nvec ----> sigma / n^2 * nvec
            const double nvec[3] = {node2_coord[0] - node1_coord[0], node2_coord[1] - node1_coord[1],
                                    node2_coord[2] - node1_coord[2]};
            const double nsqr = nvec[0] * nvec[0] + nvec[1] * nvec[1] + nvec[2] * nvec[2];
            const double right_node_force[3] = {active_force_sigma / nsqr * nvec[0],
                                                active_force_sigma / nsqr * nvec[1],
                                                active_force_sigma / nsqr * nvec[2]};

            // #pragma omp critical
            //             {
            //               std::cout << "Rank " << bulk_data.parallel_rank() << " Euchromatin spring "
            //                         << bulk_data.identifier(euchromatin_spring) << " is active." << std::endl;
            //               std::cout << "  node1: " << bulk_data.identifier(node1) << " node2: " <<
            //               bulk_data.identifier(node2)
            //                         << std::endl;
            //               std::cout << "  node1 coordinates: " << node1_coord[0] << " " << node1_coord[1] << " " <<
            //               node1_coord[2]
            //                         << std::endl;
            //               std::cout << "  node2 coordinates: " << node2_coord[0] << " " << node2_coord[1] << " " <<
            //               node2_coord[2]
            //                         << std::endl;
            //               std::cout << "  nvec: " << nvec[0] << " " << nvec[1] << " " << nvec[2] << std::endl;
            //               std::cout << "  nsqr: " << nsqr << std::endl;
            //               std::cout << "  right_node_force: " << right_node_force[0] << " " << right_node_force[1] <<
            //               " "
            //                         << right_node_force[2] << std::endl;
            //             }

            // Add the force dipole to the nodes.
            double *node1_force = stk::mesh::field_data(node_force_field, node1);
            double *node2_force = stk::mesh::field_data(node_force_field, node2);

#pragma omp atomic
            node1_force[0] -= right_node_force[0];
#pragma omp atomic
            node1_force[1] -= right_node_force[1];
#pragma omp atomic
            node1_force[2] -= right_node_force[2];
#pragma omp atomic
            node2_force[0] += right_node_force[0];
#pragma omp atomic
            node2_force[1] += right_node_force[1];
#pragma omp atomic
            node2_force[2] += right_node_force[2];
          }
        });
    // Sum the forces on shared nodes.
    stk::mesh::parallel_sum(*bulk_data_ptr_, {node_force_field_ptr_});

    Kokkos::Profiling::popRegion();
  }

  void compute_hertzian_contact_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_hertzian_contact_forces");

    // Potential evaluation (Hertzian contact)
    auto backbone_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    auto backbone_backbone_neighbor_genx_selector = stk::mesh::Selector(*backbone_backbone_neighbor_genx_part_ptr_);

    compute_ssd_and_cn_ptr_->execute(backbone_backbone_neighbor_genx_selector);
    evaluate_linker_potentials_ptr_->execute(backbone_backbone_neighbor_genx_selector);
    linker_potential_force_reduction_ptr_->execute(backbone_selector);

    Kokkos::Profiling::popRegion();
  }

  void compute_backbone_harmonic_bond_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_backbone_harmonic_bond_forces");

    auto backbone_selector = stk::mesh::Selector(*backbone_segments_part_ptr_);
    compute_constraint_forcing_ptr_->execute(backbone_selector);

    Kokkos::Profiling::popRegion();
  }

  void compute_crosslinker_harmonic_bond_forces() {
    Kokkos::Profiling::pushRegion("HP1::compute_crosslinker_harmonic_bond_forces");

    // Select only active springs in the system. Aka, not left bound.
    auto hp1_selector = stk::mesh::Selector(*hp1_part_ptr_);
    auto left_hp1_selector = stk::mesh::Selector(*left_hp1_part_ptr_);
    auto actively_bound_springs = hp1_selector - left_hp1_selector;
    compute_constraint_forcing_ptr_->execute(actively_bound_springs);

    Kokkos::Profiling::popRegion();
  }

  void compute_brownian_velocity() {
    // Compute the velocity due to brownian motion
    Kokkos::Profiling::pushRegion("HP1::compute_brownian_velocity");

    // Selectors and aliases
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<unsigned> &node_rng_field = *node_rng_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    double &timestep_size = timestep_size_;
    double &kt = brownian_kt_;
    double sphere_drag_coeff = 6.0 * M_PI * viscosity_ * backbone_sphere_hydrodynamic_radius_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_velocity_field, &node_force_field, &node_rng_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff,
         &kt](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_field, sphere_node);

          // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * kt * sphere_drag_coeff / timestep_size) * inv_drag_coeff;
          node_velocity[0] += coeff * rng.randn<double>();
          node_velocity[1] += coeff * rng.randn<double>();
          node_velocity[2] += coeff * rng.randn<double>();
          node_rng_counter[0]++;
        });

    Kokkos::Profiling::popRegion();
  }

  void compute_dry_velocity() {
    // Compute both the dry velocity due to external forces
    Kokkos::Profiling::pushRegion("HP1::compute_dry_velocity");

    // Selectors and aliases
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    double &timestep_size = timestep_size_;
    double sphere_drag_coeff = 6.0 * M_PI * viscosity_ * backbone_sphere_hydrodynamic_radius_;
    double inv_drag_coeff = 1.0 / sphere_drag_coeff;

    // Compute the total velocity of the nonorientable spheres
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_velocity_field, &node_force_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          double *node_force = stk::mesh::field_data(node_force_field, sphere_node);

          // Uext = Fext * inv_drag_coeff
          node_velocity[0] += node_force[0] * inv_drag_coeff;
          node_velocity[1] += node_force[1] * inv_drag_coeff;
          node_velocity[2] += node_force[2] * inv_drag_coeff;
        });

    Kokkos::Profiling::popRegion();
  }

  void check_maximum_speed_pre_position_update() {
    // Selectors and aliases
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    double max_allowable_speed = max_allowable_speed_;
    bool maximum_speed_exceeded = false;
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_velocity_field, &max_allowable_speed, &maximum_speed_exceeded](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, sphere_node);
          const auto speed = mundy::math::norm(node_velocity);
          if (speed > max_allowable_speed) {
            maximum_speed_exceeded = true;
          }
        });

    MUNDY_THROW_REQUIRE(!maximum_speed_exceeded, std::runtime_error,
                        fmt::format("Maximum speed exceeded on timestep {}", timestep_index_));
  }

  void update_positions() {
    // Check to see if the maximum speed is exceeded before updating the positions
    if (check_maximum_speed_pre_position_update_) {
      check_maximum_speed_pre_position_update();
    }

    Kokkos::Profiling::pushRegion("HP1::update_positions");

    // Selectors and aliases
    size_t &timestep_index = timestep_index_;
    double &timestep_size = timestep_size_;
    const stk::mesh::Selector chromatin_spheres_selector = *e_part_ptr_ | *h_part_ptr_;
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;

    // Update the positions for all spheres based on velocity
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, chromatin_spheres_selector,
        [&node_coord_field, &node_velocity_field, &timestep_size, &timestep_index](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_coord = stk::mesh::field_data(node_coord_field, sphere_node);
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);

          // x(t+dt) = x(t) + dt * v(t)
          node_coord[0] += timestep_size * node_velocity[0];
          node_coord[1] += timestep_size * node_velocity[1];
          node_coord[2] += timestep_size * node_velocity[2];
        });

    Kokkos::Profiling::popRegion();
  }

};  // class HP1

}  // namespace hp1

}  // namespace alens

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  mundy::alens::hp1::HP1().run(argc, argv);

  // Before exiting, sleep for some amount of time to force Kokkos to print better at the end.
  std::this_thread::sleep_for(std::chrono::milliseconds(stk::parallel_machine_rank(MPI_COMM_WORLD)));

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
