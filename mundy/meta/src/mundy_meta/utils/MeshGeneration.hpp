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

#ifndef MUNDY_META_UTILS_MESHGENERATION_HPP_
#define MUNDY_META_UTILS_MESHGENERATION_HPP_

/// \file MeshGeneration.hpp
/// \brief A set of helper methods for generating meshes for unit tests.

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <tuple>    // for std::tuple, std::make_tuple
#include <utility>  // for std::pair, std::make_pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/throw_assert.hpp>                  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                      // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                      // for mundy::mesh::MetaData
#include <mundy_meta/HasMeshReqsAndIsRegisterable.hpp>  // for mundy::meta::HasMeshReqsAndIsRegisterable
#include <mundy_meta/MeshReqs.hpp>                      // for mundy::meta::MeshReqs

namespace mundy {

namespace meta {

namespace utils {

template <typename MetaClass>
Teuchos::ParameterList get_validated_and_default_set_fixed_params(const Teuchos::ParameterList &fixed_params) {
  // Ensure that the given type has the correct static interface.
  using Checker = HasMeshReqsAndIsRegisterable<MetaClass>;
  static_assert(Checker::value, "The given type does not have the correct static interface for encoding requirements.");

  // Validate the fixed parameters and set defaults.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(MetaClass::get_valid_fixed_params());
  return valid_fixed_params;
}

template <typename... MetaClasses, std::size_t... Is>
std::array<Teuchos::ParameterList, sizeof...(MetaClasses)> get_vector_of_validated_and_default_set_fixed_params(
    const std::array<Teuchos::ParameterList, sizeof...(MetaClasses)> &array_of_fixed_params,
    std::index_sequence<Is...>) {
  return {get_validated_and_default_set_fixed_params<MetaClasses>(array_of_fixed_params[Is])...};
}

template <typename... MetaClasses, std::size_t... Is>
void merge_mesh_requirements_from_valid_params(
    const std::array<Teuchos::ParameterList, sizeof...(MetaClasses)> &array_of_validated_fixed_params,
    std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr, std::index_sequence<Is...>) {
  // Synchronize (merge and rectify differences) the mesh requirements for MetaClass.
  (mesh_reqs_ptr->sync(MetaClasses::get_mesh_requirements(array_of_validated_fixed_params[Is])), ...);
}

template <typename... MetaClasses, std::size_t... Is>
std::tuple<std::shared_ptr<typename MetaClasses::PolymorphicBaseType>...> create_new_instances_from_valid_params(
    const std::array<Teuchos::ParameterList, sizeof...(MetaClasses)> &array_of_validated_fixed_params,
    std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr, std::index_sequence<Is...>) {
  // Using a fold expression with a lambda to construct each type
  return std::tuple{(MetaClasses::create_new_instance(bulk_data_ptr.get(), array_of_validated_fixed_params[Is]))...};
}

/// \brief Helper function for generating a mesh that satisfies the requirements of a given meta class and returning an
/// instance of the class.
template <typename... MetaClasses>
std::tuple<std::shared_ptr<typename MetaClasses::PolymorphicBaseType>..., std::shared_ptr<mundy::mesh::BulkData>>
generate_class_instance_and_mesh_from_meta_class_requirements(
    const std::array<Teuchos::ParameterList, sizeof...(MetaClasses)> &array_of_fixed_params = {
        Teuchos::ParameterList()}) {
  constexpr size_t num_meta_classes = sizeof...(MetaClasses);

  // Setup the mesh requirements.
  auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  // Create an index sequence for the number of MetaClasses, so that we can loop over the MetaClass types and their
  // corresponding fixed params.
  auto index_sequence = std::make_index_sequence<num_meta_classes>();

  // For each MetaClass, get validate and set their default fixed parameters.
  auto array_of_validated_fixed_params =
      get_vector_of_validated_and_default_set_fixed_params<MetaClasses...>(array_of_fixed_params, index_sequence);

  // Synchronize (merge and rectify differences) the mesh requirements for each MetaClass
  merge_mesh_requirements_from_valid_params<MetaClasses...>(array_of_validated_fixed_params, mesh_reqs_ptr,
                                                            index_sequence);

  // At this point, we solidify the mesh structure.
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  meta_data_ptr->commit();

  // Create a new instance of each MetaClass using their validated fixed parameters and the newly constructed mesh.
  auto class_ptrs = create_new_instances_from_valid_params<MetaClasses...>(array_of_validated_fixed_params,
                                                                           bulk_data_ptr, index_sequence);

  return std::tuple_cat(class_ptrs, std::make_tuple(bulk_data_ptr));
}

}  // namespace utils

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_UTILS_MESHGENERATION_HPP_
