// Start
//  Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
//  Solutions of Sandia, LLC (NTESS). Under the terms of Contract
//  DE-NA0003525 with NTESS, the U.S. Government retains certain rights
//  in this software.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are
//  met:
//
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided
//        with the distribution.
//
//      * Neither the name of NTESS nor the names of its contributors
//        may be used to endorse or promote products derived from this
//        software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// #######################  Start Clang Header Tool Managed Headers ########################
// clang-format off
#include <stddef.h>                                  // for size_t
#include <stk_math/StkVector.hpp>                    // foor Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/MeshBuilder.hpp>                // for MeshBuilder
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <vector>                                    // for vector, etc
#include "mpi.h"                                     // for MPI_COMM_WORLD, etc
#include "omp.h"                                     // for pargma omp parallel, etc
#include <random>
#include <chrono>
#include <fstream>  // Required for std::ofstream
// clang-format on
// #######################   End Clang Header Tool Managed Headers  ########################

namespace {

//////////////////
// type defines //
//////////////////
typedef stk::search::IdentProc<stk::mesh::EntityKey> SearchIdentProc;
typedef std::vector<std::pair<stk::search::Sphere<double>, SearchIdentProc> > SphereIdVector;
typedef std::vector<std::pair<stk::search::Box<double>, SearchIdentProc> > BoxIdVector;
typedef std::vector<std::pair<SearchIdentProc, SearchIdentProc> > SearchIdPairVector;
constexpr double PI = 3.14159265358979323846;
constexpr double epsilon = std::numeric_limits<double>::epsilon() * 100;

//////////////////////
// Helper functions //
//////////////////////

// Quaternion helper class describing rotations
// this allows for a nice description and execution of rotations in 3D space.
// source: https://forums.developer.nvidia.com/t/cuda-for-quaternions-hyper-complex-numbers-operations/44116
struct Quaternion {
  // w,x,y,z
  double w;
  double x;
  double y;
  double z;

  // constructors
  Quaternion(const stk::Vec<double, 4> &q) {
    w = q[0];
    x = q[1];
    y = q[2];
    z = q[3];
  }

  Quaternion(const double qw, const double qx, const double qy, const double qz) {
    w = qw;
    x = qx;
    y = qy;
    z = qz;
  }

  Quaternion(const stk::Vec<double, 3> &v, const double sina_2, const double cosa_2) {
    from_rot(v, sina_2, cosa_2);
  }

  Quaternion(const stk::Vec<double, 3> &v, const double angle) {
    from_rot(v, angle);
  }

  Quaternion(const double u1, const double u2, const double u3) {
    from_unit_random(u1, u2, u3);
  }

  Quaternion() {
    from_unit_random();
  }

  // quaternion from rotation around a given axis (given sine and cosine of HALF the rotation angle)
  void from_rot(const stk::Vec<double, 3> &v, const double sina_2, const double cosa_2) {
    w = cosa_2;
    x = sina_2 * v[0];
    y = sina_2 * v[1];
    z = sina_2 * v[2];
  }

  // rotation around a given axis (angle without range restriction)
  void from_rot(const stk::Vec<double, 3> &v, const double angle) {
    const double sina_2 = sin(angle / 2);
    const double cosa_2 = cos(angle / 2);
    w = cosa_2;
    x = sina_2 * v[0];
    y = sina_2 * v[1];
    z = sina_2 * v[2];
  }

  // set a unit random quaternion representing uniform distribution on sphere surface
  void from_unit_random(const double u1, const double u2, const double u3) {
    // a random unit quaternion following a uniform distribution law on SO(3)
    // from three U[0,1] random numbers
    constexpr double pi = 3.14159265358979323846;
    const double a = sqrt(1 - u1);
    const double b = sqrt(u1);
    const double su2 = sin(2 * pi * u2);
    const double cu2 = cos(2 * pi * u2);
    const double su3 = sin(2 * pi * u3);
    const double cu3 = cos(2 * pi * u3);
    w = a * su2;
    x = a * cu2;
    y = b * su3;
    z = b * cu3;
  }

  // set a unit random quaternion representing uniform distribution on sphere surface
  void from_unit_random() {
    // non threadsafe random unit quaternion
    const double u1 = (double)rand() / RAND_MAX;
    const double u2 = (double)rand() / RAND_MAX;
    const double u3 = (double)rand() / RAND_MAX;
    from_unit_random(u1, u2, u3);
  }

  // normalize the quaternion q / ||q||
  void normalize() {
    const double norm = sqrt(w * w + x * x + y * y + z * z);
    w = w / norm;
    x = x / norm;
    y = y / norm;
    z = z / norm;
  }

  // rotate a point v in 3D space around the origin using this quaternion
  // see EN Wikipedia on Quaternions and spatial rotation
  stk::Vec<double, 3> rotate(const stk::Vec<double, 3> &v) const {
    const double t2 = x * y;
    const double t3 = x * z;
    const double t4 = x * w;
    const double t5 = -y * y;
    const double t6 = y * z;
    const double t7 = y * w;
    const double t8 = -z * z;
    const double t9 = z * w;
    const double t10 = -w * w;
    return stk::Vec<double, 3>({double(2.0) * ((t8 + t10) * v[0] + (t6 - t4) * v[1] + (t3 + t7) * v[2]) + v[0],
                                      double(2.0) * ((t4 + t6) * v[0] + (t5 + t10) * v[1] + (t9 - t2) * v[2]) + v[1],
                                      double(2.0) * ((t7 - t3) * v[0] + (t2 + t9) * v[1] + (t5 + t8) * v[2]) + v[2]});
  }

  // rotate a point v in 3D space around a given point p using this quaternion
  stk::Vec<double, 3> rotate_around_point(const stk::Vec<double, 3> &v,
                                                const stk::Vec<double, 3> &p) {
    return rotate(v - p) + p;
  }

  /**
   * \brief rotate the quaternion itself based on rotational velocity omega
   *
   * Delong, JCP, 2015, Appendix A eq1, not linearized
   * \param q
   * \param omega rotational velocity
   * \param dt time interval
   */
  void rotate_self(const stk::Vec<double, 3> &rot_vel, const double dt) {
    const double rot_vel_norm = sqrt(rot_vel[0] * rot_vel[0] + rot_vel[1] * rot_vel[1] + rot_vel[2] * rot_vel[2]);
    if (rot_vel_norm < std::numeric_limits<double>::epsilon()) {
      return;
    }
    const double rot_vel_norm_inv = 1.0 / rot_vel_norm;
    const double sw = sin(rot_vel_norm * dt / 2);
    const double cw = cos(rot_vel_norm * dt / 2);
    const double rot_vel_cross_xyz_0 = rot_vel[1] * z - rot_vel[2] * y;
    const double rot_vel_cross_xyz_1 = rot_vel[2] * x - rot_vel[0] * z;
    const double rot_vel_cross_xyz_2 = rot_vel[0] * y - rot_vel[1] * x;
    const double rot_vel_dot_xyz = rot_vel[0] * x + rot_vel[1] * y + rot_vel[2] * z;

    x = w * sw * rot_vel[0] * rot_vel_norm_inv + cw * x + sw * rot_vel_norm_inv * rot_vel_cross_xyz_0;
    y = w * sw * rot_vel[1] * rot_vel_norm_inv + cw * y + sw * rot_vel_norm_inv * rot_vel_cross_xyz_1;
    z = w * sw * rot_vel[2] * rot_vel_norm_inv + cw * z + sw * rot_vel_norm_inv * rot_vel_cross_xyz_2;
    w = w * cw - rot_vel_dot_xyz * sw * rot_vel_norm_inv;
    normalize();
  }

  /**
   * \brief rotate the quaternion itself based on rotational velocity omega
   *
   * Delong, JCP, 2015, Appendix A eq1, not linearized
   * \param q
   * \param omega rotational velocity
   * \param dt time interval
   */
  void rotate_self(const double rot_vel_x, const double rot_vel_y, const double rot_vel_z, const double dt) {
    const double rot_vel_norm = sqrt(rot_vel_x * rot_vel_x + rot_vel_y * rot_vel_y + rot_vel_z * rot_vel_z);
    if (rot_vel_norm < std::numeric_limits<double>::epsilon()) {
      return;
    }
    const double rot_vel_norm_inv = 1.0 / rot_vel_norm;
    const double sw = sin(rot_vel_norm * dt / 2);
    const double cw = cos(rot_vel_norm * dt / 2);
    const double rot_vel_cross_xyz_0 = rot_vel_y * z - rot_vel_z * y;
    const double rot_vel_cross_xyz_1 = rot_vel_z * x - rot_vel_x * z;
    const double rot_vel_cross_xyz_2 = rot_vel_x * y - rot_vel_y * x;
    const double rot_vel_dot_xyz = rot_vel_x * x + rot_vel_y * y + rot_vel_z * z;

    x = w * sw * rot_vel_x * rot_vel_norm_inv + cw * x + sw * rot_vel_norm_inv * rot_vel_cross_xyz_0;
    y = w * sw * rot_vel_y * rot_vel_norm_inv + cw * y + sw * rot_vel_norm_inv * rot_vel_cross_xyz_1;
    z = w * sw * rot_vel_z * rot_vel_norm_inv + cw * z + sw * rot_vel_norm_inv * rot_vel_cross_xyz_2;
    w = w * cw - rot_vel_dot_xyz * sw * rot_vel_norm_inv;
    normalize();
  }
};  // Quaternion

/////////////////////////////////////////////
// Generating collision constraint helpers //
/////////////////////////////////////////////

void filterOutSelfOverlap(const stk::mesh::BulkData &bulkData, SearchIdPairVector &searchResults) {
  size_t numFiltered = 0;

  for (const auto &searchResult : searchResults) {
    stk::mesh::Entity element1 = bulkData.get_entity(searchResult.first.id());
    stk::mesh::Entity element2 = bulkData.get_entity(searchResult.second.id());
    int owningProcElement1 = searchResult.first.proc();
    int owningProcElement2 = searchResult.second.proc();

    ThrowRequireWithSierraHelpMsg(owningProcElement1 == bulkData.parallel_rank() ||
                                  owningProcElement2 == bulkData.parallel_rank());

    bool anyIntersections = false;

    if (element1 == element2) {
      anyIntersections = true;
    }

    if (!anyIntersections) {
      searchResults[numFiltered] = searchResult;
      numFiltered++;
    }
  }

  searchResults.resize(numFiltered);
}

void filterOutNonLocalResults(const stk::mesh::BulkData &bulkData, SearchIdPairVector &searchResults) {
  const int rank = bulkData.parallel_rank();
  size_t numFiltered = 0;

  for (const auto &searchResult : searchResults) {
    if (searchResult.first.proc() == rank) {
      searchResults[numFiltered] = searchResult;
      numFiltered++;
    }
  }

  searchResults.resize(numFiltered);
}

/* Ghost the search results as appropriate. The search results include results on all procs including
 * locally_owned OR globally_shared. We need to ghost the locally owned node (domain or range)
 * to the processor pointed to by the other node in the periodic pair
 */

// ghost the nodes, the upward connections will be automatically ghosted in the aura
void create_ghosting(stk::mesh::BulkData &bulkData, const SearchIdPairVector &searchResults, const std::string &name) {
  ThrowRequire(bulkData.in_modifiable_state());
  const int parallel_rank = bulkData.parallel_rank();
  std::vector<stk::mesh::EntityProc> send_nodes;
  for (size_t i = 0; i < searchResults.size(); ++i) {
    stk::mesh::Entity domain_node = bulkData.get_entity(searchResults[i].first.id());
    stk::mesh::Entity range_node = bulkData.get_entity(searchResults[i].second.id());

    bool is_owned_domain = bulkData.is_valid(domain_node) ? bulkData.bucket(domain_node).owned() : false;
    bool is_owned_range = bulkData.is_valid(range_node) ? bulkData.bucket(range_node).owned() : false;
    int domain_proc = searchResults[i].first.proc();
    int range_proc = searchResults[i].second.proc();

    if (is_owned_domain && domain_proc == parallel_rank) {
      if (range_proc == parallel_rank) continue;

      ThrowRequire(bulkData.parallel_owner_rank(domain_node) == domain_proc);

      send_nodes.emplace_back(domain_node, range_proc);
    } else if (is_owned_range && range_proc == parallel_rank) {
      if (domain_proc == parallel_rank) continue;

      ThrowRequire(bulkData.parallel_owner_rank(range_node) == range_proc);

      send_nodes.emplace_back(range_node, domain_proc);
    }
  }

  stk::mesh::Ghosting &ghosting = bulkData.create_ghosting(name);
  bulkData.change_ghosting(ghosting, send_nodes);
}

//////////////////////
// Particle kernels //
//////////////////////

void compute_maximum_abs_projected_sep(stk::mesh::BulkData &bulkData, stk::mesh::Field<double> &linkerLagMultField,
                                       stk::mesh::Field<double> &linkerSignedSepField,
                                       stk::mesh::Field<double> &linkerSignedSepDotField, const double dt,
                                       double &global_maximum_abs_projected_sep) {
  // compute the maximum absolute projected sep for each constraint
  double local_maximum_abs_projected_sep = -1.0;

  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalLinkers =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::BEAM_2);
  const stk::mesh::BucketVector &linkerBuckets = bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalLinkers);

#pragma omp parallel for reduction(max : local_maximum_abs_projected_sep)
  for (size_t bucket_idx = 0; bucket_idx < linkerBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &linkerBucket = *linkerBuckets[bucket_idx];
    for (size_t linker_idx = 0; linker_idx < linkerBucket.size(); ++linker_idx) {
      // fetch the entities
      stk::mesh::Entity const &linker = linkerBucket[linker_idx];

      // perform the projection EQ 2.2 of Dai & Fletcher 2005
      const double lag_mult = stk::mesh::field_data(linkerLagMultField, linker)[0];
      const double sep_old = stk::mesh::field_data(linkerSignedSepField, linker)[0];
      const double sep_dot = stk::mesh::field_data(linkerSignedSepDotField, linker)[0];
      const double sep_new = sep_old + dt * sep_dot;

      double abs_projected_sep;
      if (lag_mult < epsilon) {
        abs_projected_sep = std::abs(std::min(sep_new, 0.0));
      } else {
        abs_projected_sep = std::abs(sep_new);
      }

      // store the maximum
      if (abs_projected_sep > local_maximum_abs_projected_sep) {
        local_maximum_abs_projected_sep = abs_projected_sep;
      }
    }
  }

  // compute the global maximum absolute projected sep
  global_maximum_abs_projected_sep = 0.0;
  stk::all_reduce_max(bulkData.parallel(), &local_maximum_abs_projected_sep, &global_maximum_abs_projected_sep, 1);
}

void compute_diff_dots(stk::mesh::BulkData &bulkData, const stk::mesh::Field<double> &linkerLagMultField,
                       const stk::mesh::Field<double> &linkerLagMultTmpField,
                       const stk::mesh::Field<double> &linkerSignedSepDotField,
                       const stk::mesh::Field<double> &linkerSignedSepDotTmpField, const double dt,
                       double &global_dot_xkdiff_xkdiff, double &global_dot_xkdiff_gkdiff,
                       double &global_dot_gkdiff_gkdiff) {
  // compute dot(xkdiff, xkdiff), dot(xkdiff, gkdiff), dot(gkdiff, gkdiff)
  // where xkdiff = xk - xkm1 and gkdiff = gk - gkm1
  double local_dot_xkdiff_xkdiff = 0.0;
  double local_dot_xkdiff_gkdiff = 0.0;
  double local_dot_gkdiff_gkdiff = 0.0;

  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalLinkers =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::BEAM_2);
  const stk::mesh::BucketVector &linkerBuckets = bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalLinkers);

#pragma omp parallel for reduction(+ : local_dot_xkdiff_xkdiff, local_dot_xkdiff_gkdiff, local_dot_gkdiff_gkdiff)
  for (size_t bucket_idx = 0; bucket_idx < linkerBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &linkerBucket = *linkerBuckets[bucket_idx];
    for (size_t linker_idx = 0; linker_idx < linkerBucket.size(); ++linker_idx) {
      // fetch the entities
      stk::mesh::Entity const &linker = linkerBucket[linker_idx];

      // fetch the fields
      const double lag_mult = stk::mesh::field_data(linkerLagMultField, linker)[0];
      const double lag_mult_tmp = stk::mesh::field_data(linkerLagMultTmpField, linker)[0];
      const double sep_dot = stk::mesh::field_data(linkerSignedSepDotField, linker)[0];
      const double sep_dot_tmp = stk::mesh::field_data(linkerSignedSepDotTmpField, linker)[0];

      // xkdiff = xk - xkm1
      const double xkdiff = lag_mult - lag_mult_tmp;

      // gkdiff = gk - gkm1
      const double gkdiff = dt * (sep_dot - sep_dot_tmp);

      // sum up the dot products
      local_dot_xkdiff_xkdiff += xkdiff * xkdiff;
      local_dot_xkdiff_gkdiff += xkdiff * gkdiff;
      local_dot_gkdiff_gkdiff += gkdiff * gkdiff;
    }
  }

  // compute the global sums
  stk::all_reduce_sum(bulkData.parallel(), &local_dot_xkdiff_xkdiff, &global_dot_xkdiff_xkdiff, 1);
  stk::all_reduce_sum(bulkData.parallel(), &local_dot_xkdiff_gkdiff, &global_dot_xkdiff_gkdiff, 1);
  stk::all_reduce_sum(bulkData.parallel(), &local_dot_gkdiff_gkdiff, &global_dot_gkdiff_gkdiff, 1);
}

void generate_neighbor_pairs(const stk::mesh::BulkData &bulkData, const stk::mesh::Field<double> &elemAabbField,
                             SearchIdPairVector &neighborPairs) {
  // setup the search boxes (for each element)
  const stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  BoxIdVector elementBoxes;

  const int rank = bulkData.parallel_rank();
  const size_t num_local_elements =
      stk::mesh::count_entities(bulkData, stk::topology::ELEMENT_RANK, metaData.locally_owned_part());
  elementBoxes.reserve(num_local_elements);

  const stk::mesh::BucketVector &elementBuckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, metaData.locally_owned_part());
  for (size_t bucket_idx = 0; bucket_idx < elementBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &elemBucket = *elementBuckets[bucket_idx];
    for (size_t elem_idx = 0; elem_idx < elemBucket.size(); ++elem_idx) {
      stk::mesh::Entity const &element = elemBucket[elem_idx];

      double *aabb = stk::mesh::field_data(elemAabbField, element);
      stk::search::Box<double> box(aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]);

      SearchIdentProc search_id(bulkData.entity_key(element), rank);

      elementBoxes.emplace_back(box, search_id);
    }
  }

  // perform the aabb search
  stk::search::coarse_search(elementBoxes, elementBoxes, stk::search::KDTREE, bulkData.parallel(), neighborPairs);

  // filter results
  // remove self-overlap and remove nonlocal source boxes
  filterOutSelfOverlap(bulkData, neighborPairs);
  filterOutNonLocalResults(bulkData, neighborPairs);
}

void generate_collision_constraints(stk::mesh::BulkData &bulkData, const SearchIdPairVector &neighborPairs,
                                    stk::mesh::Part &linkerPart, stk::mesh::Field<double> &nodeCoordField,
                                    stk::mesh::Field<double> &particleRadiusField,
                                    stk::mesh::Field<double> &linkerSignedSepField,
                                    stk::mesh::Field<double> &linkerSignedSepDotField,
                                    stk::mesh::Field<double> &linkerSignedSepDotTmpField,
                                    stk::mesh::Field<double> &linkerLagMultField,
                                    stk::mesh::Field<double> &linkerLagMultTmpField,
                                    stk::mesh::Field<double> &conLocField, stk::mesh::Field<double> &conNormField) {
  /*
  Note:
    A niave procedure can generate two linkers between every pair of particles
    This can be remedied by having the particle with the smaller ID generate the constraints

  Procedure:
    1. ghost neighbors that aren't on the current proocess
    2. generate linkers between neighbor particles
      2.1. the process that owns the particle with the smaller ID generates the linker
      2.2. add node sharing between processors
    3. fill the linkers with the collision information
  */

  // populate the ghost using the search results
  std::cout << "Generating ghosting..." << std::endl;
  bulkData.modification_begin();
  create_ghosting(bulkData, neighborPairs, "geometricGhosts");
  bulkData.modification_end();

  // communicate the necessary ghost particle fields
  std::vector<const stk::mesh::FieldBase *> fields{&nodeCoordField, &particleRadiusField};
  stk::mesh::communicate_field_data(bulkData, fields);

  // generate linkers between the neighbors
  // at this point, the number of neighbors == the number of linkers that need generated
  // the particles already have nodes, so we only need to generate linker entities
  // and declare relations/sharing between those entities and the connected nodes
  bulkData.modification_begin();
  const size_t num_linkers = std::count_if(neighborPairs.begin(), neighborPairs.end(),
                                           [](const std::pair<SearchIdentProc, SearchIdentProc> &neighborPair) {
                                             return neighborPair.first.id() < neighborPair.second.id();
                                           });
  std::vector<size_t> requests(bulkData.mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::ELEMENT_RANK] = num_linkers;

  // ex.
  //  requests = { 0, 4,  8}
  //  requests 0 entites of rank 0, 4 entites of rank 1, and 8 entites of rank 2
  //  requested_entities = {0 entites of rank 0, 4 entites of rank 1, 8 entites of rank 2}
  std::vector<stk::mesh::Entity> requested_entities;
  bulkData.generate_new_entities(requests, requested_entities);

  // associate each particle with a single part
  std::vector<stk::mesh::Part *> add_linkerPart(1);
  add_linkerPart[0] = &linkerPart;

  // set topologies of new entities
  std::cout << "Setting topologies..." << std::endl;
  // #pragma omp parallel for
  for (int i = 0; i < num_linkers; i++) {
    stk::mesh::Entity linker_i = requested_entities[i];
    bulkData.change_entity_parts(linker_i, add_linkerPart);
  }

  // the elements should be associated with a topology before they are connected to their nodes/edges
  // set downward relations of entities
  // loop over the neighbor pairs
  size_t count = 0;
  for (int i = 0; i < neighborPairs.size(); i++) {
    stk::mesh::Entity particleI = bulkData.get_entity(neighborPairs[i].first.id());
    stk::mesh::Entity particleJ = bulkData.get_entity(neighborPairs[i].second.id());
    stk::mesh::Entity nodesI = bulkData.begin_nodes(particleI)[0];
    stk::mesh::Entity nodesJ = bulkData.begin_nodes(particleJ)[0];
    int owningProcI = neighborPairs[i].first.proc();
    int owningProcJ = neighborPairs[i].second.proc();

    // share both nodes with other process
    // add_node_sharing must be called symmetrically
    assert(owningProcI == bulkData.parallel_rank());
    if (bulkData.parallel_rank() != owningProcJ) {
      bulkData.add_node_sharing(nodesI, owningProcJ);
      bulkData.add_node_sharing(nodesJ, owningProcJ);
    }

    // only generate linkers if the source particle's id is less
    // than the id of the target particle. this prevents duplicate constraints
    if (neighborPairs[i].first.id() < neighborPairs[i].second.id()) {
      stk::mesh::Entity linker_i = requested_entities[count];
      bulkData.declare_relation(linker_i, nodesI, 0);
      bulkData.declare_relation(linker_i, nodesJ, 1);

      // fill the constraint information
      const double *const posI = stk::mesh::field_data(nodeCoordField, nodesI);
      const double *const posJ = stk::mesh::field_data(nodeCoordField, nodesJ);
      const double *const radiusI = stk::mesh::field_data(particleRadiusField, particleI);
      const double *const radiusJ = stk::mesh::field_data(particleRadiusField, particleJ);

      const stk::Vec<double, 3> distIJ({posJ[0] - posI[0], posJ[1] - posI[1], posJ[2] - posI[2]});
      const double com_sep = sqrt(distIJ[0] * distIJ[0] + distIJ[1] * distIJ[1] + distIJ[2] * distIJ[2]);
      const stk::Vec<double, 3> normIJ = distIJ / com_sep;

      stk::mesh::field_data(linkerSignedSepField, linker_i)[0] = com_sep - radiusI[0] - radiusJ[0];
      stk::mesh::field_data(linkerSignedSepDotField, linker_i)[0] = 0.0;
      stk::mesh::field_data(linkerSignedSepDotTmpField, linker_i)[0] = 0.0;
      stk::mesh::field_data(linkerLagMultField, linker_i)[0] = 0.0;
      stk::mesh::field_data(linkerLagMultTmpField, linker_i)[0] = 0.0;

      // con loc is relative to the com TODO: change name to be more explicit
      stk::mesh::field_data(conLocField, linker_i)[0] = radiusI[0] * normIJ[0];
      stk::mesh::field_data(conLocField, linker_i)[1] = radiusI[0] * normIJ[1];
      stk::mesh::field_data(conLocField, linker_i)[2] = radiusI[0] * normIJ[2];
      stk::mesh::field_data(conLocField, linker_i)[3] = -radiusJ[0] * normIJ[0];
      stk::mesh::field_data(conLocField, linker_i)[4] = -radiusJ[0] * normIJ[1];
      stk::mesh::field_data(conLocField, linker_i)[5] = -radiusJ[0] * normIJ[2];

      stk::mesh::field_data(conNormField, linker_i)[0] = normIJ[0];
      stk::mesh::field_data(conNormField, linker_i)[1] = normIJ[1];
      stk::mesh::field_data(conNormField, linker_i)[2] = normIJ[2];
      stk::mesh::field_data(conNormField, linker_i)[3] = -normIJ[0];
      stk::mesh::field_data(conNormField, linker_i)[4] = -normIJ[1];
      stk::mesh::field_data(conNormField, linker_i)[5] = -normIJ[2];

      count++;
    }
  }
  bulkData.modification_end();
}

void compute_constraint_center_of_mass_force_torque(
    stk::mesh::BulkData &bulkData,
    const stk::mesh::Field<double> &nodeForceField,  // TODO: should these be non-const?
    const stk::mesh::Field<double> &nodeTorqueField, const stk::mesh::Field<double> &linkerLagMultField,
    const stk::mesh::Field<double> &conNormField, const stk::mesh::Field<double> &conLocField) {
  // communicate the necessary ghost linker fields
  std::vector<const stk::mesh::FieldBase *> fields{&linkerLagMultField, &conNormField, &conLocField};
  stk::mesh::communicate_field_data(bulkData, fields);

  // compute D xk
  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalParticles =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::PARTICLE);
  const stk::mesh::BucketVector &particleBuckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalParticles);

  // compute com force and torque from gamma
#pragma omp parallel for
  for (size_t bucket_idx = 0; bucket_idx < particleBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &particleBucket = *particleBuckets[bucket_idx];
    for (size_t particle_idx = 0; particle_idx < particleBucket.size(); ++particle_idx) {
      // fetch the entities
      stk::mesh::Entity const &particle = particleBucket[particle_idx];
      stk::mesh::Entity particleNode = particleBucket.begin_nodes(particle_idx)[0];

      // get the com translational and rotational velocity
      double *com_force = stk::mesh::field_data(nodeForceField, particleNode);
      double *com_torque = stk::mesh::field_data(nodeTorqueField, particleNode);

      // zero out the forces and torques first
      com_force[0] = 0.0;
      com_force[1] = 0.0;
      com_force[2] = 0.0;

      com_torque[0] = 0.0;
      com_torque[1] = 0.0;
      com_torque[2] = 0.0;

      // fetch the connected constraints
      int num_connected_elems = bulkData.num_elements(particleNode);
      const stk::mesh::Entity *connected_elems = bulkData.begin_elements(particleNode);

      // if the connected entity is a linker
      // sum their force into the com force and torque
      for (unsigned entity_idx = 0; entity_idx < num_connected_elems; ++entity_idx) {
        const stk::topology elemTopology = bulkData.bucket(connected_elems[entity_idx]).topology();
        if (elemTopology == stk::topology::BEAM_2) {
          // determine which side of the linker we are connected to
          stk::mesh::Entity const *linkerNodes = bulkData.begin_nodes(connected_elems[entity_idx]);
          bool is_particle_I = bulkData.identifier(particleNode) == bulkData.identifier(linkerNodes[0]);
          if (!is_particle_I) {
            // if it's not particle I, it better be particle J
            assert(bulkData.identifier(particleNode) == bulkData.identifier(linkerNodes[1]));
          }

          // com force = -norm * lag_mult
          // com torque = -con_pos cross (norm * lag_mult)
          const double linker_lag_mult = stk::mesh::field_data(linkerLagMultField, connected_elems[entity_idx])[0];

          double *con_norm;
          double *con_pos;
          if (is_particle_I) {
            con_norm = stk::mesh::field_data(conNormField, connected_elems[entity_idx]);
            con_pos = stk::mesh::field_data(conLocField, connected_elems[entity_idx]);
          } else {
            con_norm = stk::mesh::field_data(conNormField, connected_elems[entity_idx]) + 3;
            con_pos = stk::mesh::field_data(conLocField, connected_elems[entity_idx]) + 3;
          }

          // sum the force/torque generated by each constraint
          // collision between spheres doesn't generate torque
          com_force[0] += -linker_lag_mult * con_norm[0];
          com_force[1] += -linker_lag_mult * con_norm[1];
          com_force[2] += -linker_lag_mult * con_norm[2];

          com_torque[0] += -linker_lag_mult * (con_pos[1] * con_norm[2] - con_pos[2] * con_norm[1]);
          com_torque[1] += -linker_lag_mult * (con_pos[2] * con_norm[0] - con_pos[0] * con_norm[2]);
          com_torque[2] += -linker_lag_mult * (con_pos[0] * con_norm[1] - con_pos[1] * con_norm[0]);
        }
      }
    }
  }
}

void compute_the_mobility_problem(stk::mesh::BulkData &bulkData,
                                  const stk::mesh::Field<double> &particleOrientationField,
                                  const stk::mesh::Field<double> &particleRadiusField,
                                  const stk::mesh::Field<double> &nodeForceField,
                                  const stk::mesh::Field<double> &nodeTorqueField,
                                  stk::mesh::Field<double> &nodeVelocityField, stk::mesh::Field<double> &nodeOmegaField,
                                  const double viscosity) {
  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalParticles =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::PARTICLE);
  const stk::mesh::BucketVector &particleBuckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalParticles);

  // compute U = M F
  // for now, M is block diagonal
#pragma omp parallel for
  for (size_t bucket_idx = 0; bucket_idx < particleBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &particleBucket = *particleBuckets[bucket_idx];
    for (size_t particle_idx = 0; particle_idx < particleBucket.size(); ++particle_idx) {
      // fetch the entities
      stk::mesh::Entity const &particle = particleBucket[particle_idx];
      stk::mesh::Entity node = particleBucket.begin_nodes(particle_idx)[0];

      // compute the mobility matrix for the sphere
      const double *const particle_orientation = stk::mesh::field_data(particleOrientationField, particle);
      const double particle_radius = stk::mesh::field_data(particleRadiusField, particle)[0];

      Quaternion quat(particle_orientation[0], particle_orientation[1], particle_orientation[2],
                      particle_orientation[3]);

      const stk::Vec<double, 3> q = quat.rotate(stk::Vec<double, 3>({0, 0, 1}));
      const double qq[3][3] = {{q[0] * q[0], q[0] * q[1], q[0] * q[2]},
                               {q[1] * q[0], q[1] * q[1], q[1] * q[2]},
                               {q[2] * q[0], q[2] * q[1], q[2] * q[2]}};
      const double Imqq[3][3] = {{1 - qq[0][0], -qq[0][1], -qq[0][2]},
                                 {-qq[1][0], 1 - qq[1][1], -qq[1][2]},
                                 {-qq[2][0], -qq[2][1], 1 - qq[2][2]}};
      const double drag_para = 6 * PI * particle_radius * viscosity;
      const double drag_perp = drag_para;
      const double drag_rot = 8 * PI * particle_radius * particle_radius * particle_radius * viscosity;
      const double drag_para_inv = 1.0 / drag_para;
      const double drag_perp_inv = 1.0 / drag_perp;
      const double drag_rot_inv = 1.0 / drag_rot;
      const double mob_trans[3][3] = {
          {drag_para_inv * qq[0][0] + drag_perp_inv * Imqq[0][0], drag_para_inv * qq[0][1] + drag_perp_inv * Imqq[0][1],
           drag_para_inv * qq[0][2] + drag_perp_inv * Imqq[0][2]},
          {drag_para_inv * qq[1][0] + drag_perp_inv * Imqq[1][0], drag_para_inv * qq[1][1] + drag_perp_inv * Imqq[1][1],
           drag_para_inv * qq[1][2] + drag_perp_inv * Imqq[1][2]},
          {drag_para_inv * qq[2][0] + drag_perp_inv * Imqq[2][0], drag_para_inv * qq[2][1] + drag_perp_inv * Imqq[2][1],
           drag_para_inv * qq[2][2] + drag_perp_inv * Imqq[2][2]}};
      const double mob_rot[3][3] = {
          {drag_rot_inv * qq[0][0] + drag_rot_inv * Imqq[0][0], drag_rot_inv * qq[0][1] + drag_rot_inv * Imqq[0][1],
           drag_rot_inv * qq[0][2] + drag_rot_inv * Imqq[0][2]},
          {drag_rot_inv * qq[1][0] + drag_rot_inv * Imqq[1][0], drag_rot_inv * qq[1][1] + drag_rot_inv * Imqq[1][1],
           drag_rot_inv * qq[1][2] + drag_rot_inv * Imqq[1][2]},
          {drag_rot_inv * qq[2][0] + drag_rot_inv * Imqq[2][0], drag_rot_inv * qq[2][1] + drag_rot_inv * Imqq[2][1],
           drag_rot_inv * qq[2][2] + drag_rot_inv * Imqq[2][2]}};

      // solve for the induced velocity and omega
      const double *const node_force = stk::mesh::field_data(nodeForceField, node);
      const double *const node_torque = stk::mesh::field_data(nodeTorqueField, node);
      double *node_velocity = stk::mesh::field_data(nodeVelocityField, node);
      double *node_omega = stk::mesh::field_data(nodeOmegaField, node);

      node_velocity[0] =
          mob_trans[0][0] * node_force[0] + mob_trans[0][1] * node_force[1] + mob_trans[0][2] * node_force[2];
      node_velocity[1] =
          mob_trans[1][0] * node_force[0] + mob_trans[1][1] * node_force[1] + mob_trans[1][2] * node_force[2];
      node_velocity[2] =
          mob_trans[2][0] * node_force[0] + mob_trans[2][1] * node_force[1] + mob_trans[2][2] * node_force[2];
      node_omega[0] = mob_rot[0][0] * node_torque[0] + mob_rot[0][1] * node_torque[1] + mob_rot[0][2] * node_torque[2];
      node_omega[1] = mob_rot[1][0] * node_torque[0] + mob_rot[1][1] * node_torque[1] + mob_rot[1][2] * node_torque[2];
      node_omega[2] = mob_rot[2][0] * node_torque[0] + mob_rot[2][1] * node_torque[1] + mob_rot[2][2] * node_torque[2];
    }
  }
}

void compute_rate_of_change_of_sep(stk::mesh::BulkData &bulkData, const stk::mesh::Field<double> &nodeVelocityField,
                                   const stk::mesh::Field<double> &nodeOmegaField,
                                   const stk::mesh::Field<double> &conLocField,
                                   const stk::mesh::Field<double> &conNormField,
                                   stk::mesh::Field<double> &linkerSignedSepDotField) {
  // communicate the necessary ghost node fields
  std::vector<const stk::mesh::FieldBase *> fields{&nodeVelocityField, &nodeOmegaField};
  stk::mesh::communicate_field_data(bulkData, fields);

  // compute D^T U for each constraint
  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalLinkers =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::BEAM_2);
  const stk::mesh::BucketVector &linkerBuckets = bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalLinkers);

  // compute the (linearized) rate of change in sep

#pragma omp parallel for
  for (size_t bucket_idx = 0; bucket_idx < linkerBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &linkerBucket = *linkerBuckets[bucket_idx];
    for (size_t linker_idx = 0; linker_idx < linkerBucket.size(); ++linker_idx) {
      // fetch the entities
      stk::mesh::Entity const &linker = linkerBucket[linker_idx];
      stk::mesh::Entity const *nodes = linkerBucket.begin_nodes(linker_idx);

      // sep_dot = -con_normI dot (com_velocityI + com_omegaI x con_posI)
      //           -con_normJ dot (com_velocityJ + com_omegaJ x con_posJ)
      stk::Vec<double, 3> com_velocityI;
      stk::Vec<double, 3> com_velocityJ;
      com_velocityI[0] = *(stk::mesh::field_data(nodeVelocityField, nodes[0]) + 0);
      com_velocityI[1] = *(stk::mesh::field_data(nodeVelocityField, nodes[0]) + 1);
      com_velocityI[2] = *(stk::mesh::field_data(nodeVelocityField, nodes[0]) + 2);
      com_velocityJ[0] = *(stk::mesh::field_data(nodeVelocityField, nodes[1]) + 0);
      com_velocityJ[1] = *(stk::mesh::field_data(nodeVelocityField, nodes[1]) + 1);
      com_velocityJ[2] = *(stk::mesh::field_data(nodeVelocityField, nodes[1]) + 2);

      stk::Vec<double, 3> com_omegaI;
      stk::Vec<double, 3> com_omegaJ;
      com_omegaI[0] = *(stk::mesh::field_data(nodeOmegaField, nodes[0]) + 0);
      com_omegaI[1] = *(stk::mesh::field_data(nodeOmegaField, nodes[0]) + 1);
      com_omegaI[2] = *(stk::mesh::field_data(nodeOmegaField, nodes[0]) + 2);
      com_omegaJ[0] = *(stk::mesh::field_data(nodeOmegaField, nodes[1]) + 0);
      com_omegaJ[1] = *(stk::mesh::field_data(nodeOmegaField, nodes[1]) + 1);
      com_omegaJ[2] = *(stk::mesh::field_data(nodeOmegaField, nodes[1]) + 2);

      stk::Vec<double, 3> con_posI;
      stk::Vec<double, 3> con_posJ;
      con_posI[0] = *(stk::mesh::field_data(conLocField, linker) + 0);
      con_posI[1] = *(stk::mesh::field_data(conLocField, linker) + 1);
      con_posI[2] = *(stk::mesh::field_data(conLocField, linker) + 2);
      con_posJ[0] = *(stk::mesh::field_data(conLocField, linker) + 3);
      con_posJ[1] = *(stk::mesh::field_data(conLocField, linker) + 4);
      con_posJ[2] = *(stk::mesh::field_data(conLocField, linker) + 5);

      stk::Vec<double, 3> con_normI;
      stk::Vec<double, 3> con_normJ;
      con_normI[0] = *(stk::mesh::field_data(conNormField, linker) + 0);
      con_normI[1] = *(stk::mesh::field_data(conNormField, linker) + 1);
      con_normI[2] = *(stk::mesh::field_data(conNormField, linker) + 2);
      con_normJ[0] = *(stk::mesh::field_data(conNormField, linker) + 3);
      con_normJ[1] = *(stk::mesh::field_data(conNormField, linker) + 4);
      con_normJ[2] = *(stk::mesh::field_data(conNormField, linker) + 5);

      // compute D^T U
      const stk::Vec<double, 3> con_velI = com_velocityI + Cross(com_omegaI, con_posI);
      const stk::Vec<double, 3> con_velJ = com_velocityJ + Cross(com_omegaJ, con_posJ);
      stk::mesh::field_data(linkerSignedSepDotField, linker)[0] = -Dot(con_normI, con_velI) - Dot(con_normJ, con_velJ);
    }
  }
}

void update_con_gammas(stk::mesh::BulkData &bulkData, stk::mesh::Field<double> &linkerLagMultField,
                       stk::mesh::Field<double> &linkerLagMultTmpField, stk::mesh::Field<double> &linkerSignedSepField,
                       stk::mesh::Field<double> &linkerSignedSepDotField, const double dt, const double alpha) {
  // compute xk = xkm1 - alpha * gkm1;
  // and perform the bound projection xk = boundProjection(xk)
  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalLinkers =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::BEAM_2);
  const stk::mesh::BucketVector &linkerBuckets = bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalLinkers);

  // set xkm1 = xk and gkm1 = gk
#pragma omp parallel for
  for (size_t bucket_idx = 0; bucket_idx < linkerBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &linkerBucket = *linkerBuckets[bucket_idx];
    for (size_t linker_idx = 0; linker_idx < linkerBucket.size(); ++linker_idx) {
      // fetch the entities
      stk::mesh::Entity const &linker = linkerBucket[linker_idx];

      // fetch the fields
      const double sep_old = stk::mesh::field_data(linkerSignedSepField, linker)[0];
      const double sep_dot = stk::mesh::field_data(linkerSignedSepDotField, linker)[0];
      const double sep_new = sep_old + dt * sep_dot;

      double *lag_mult = stk::mesh::field_data(linkerLagMultField, linker);
      double *lag_mult_tmp = stk::mesh::field_data(linkerLagMultTmpField, linker);

      // xk = boundProjection(xkm1 - alpha * gkm1)
      lag_mult[0] = std::max(lag_mult_tmp[0] - alpha * sep_new, 0.0);
    }
  }
}

void swap_con_gammas(stk::mesh::BulkData &bulkData, stk::mesh::Field<double> &linkerLagMultField,
                     stk::mesh::Field<double> &linkerLagMultTmpField, stk::mesh::Field<double> &linkerSignedSepDotField,
                     stk::mesh::Field<double> &linkerSignedSepDotTmpField) {
  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalLinkers =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::BEAM_2);
  const stk::mesh::BucketVector &linkerBuckets = bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalLinkers);

  // set xkm1 = xk and gkm1 = gk
#pragma omp parallel for
  for (size_t bucket_idx = 0; bucket_idx < linkerBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &linkerBucket = *linkerBuckets[bucket_idx];
    for (size_t linker_idx = 0; linker_idx < linkerBucket.size(); ++linker_idx) {
      // fetch the entities
      stk::mesh::Entity const &linker = linkerBucket[linker_idx];

      double *lag_mult = stk::mesh::field_data(linkerLagMultField, linker);
      double *lag_mult_tmp = stk::mesh::field_data(linkerLagMultTmpField, linker);
      double *sep_dot = stk::mesh::field_data(linkerSignedSepDotField, linker);
      double *sep_dot_tmp = stk::mesh::field_data(linkerSignedSepDotTmpField, linker);

      lag_mult_tmp[0] = lag_mult[0];
      sep_dot_tmp[0] = sep_dot[0];
    }
  }
}

void step_euler(stk::mesh::BulkData &bulkData, const stk::mesh::Field<double> &nodeVelocityField,
                const stk::mesh::Field<double> &nodeOmegaField, stk::mesh::Field<double> &nodeCoordField,
                stk::mesh::Field<double> &particleOrientationField, const double dt) {
  stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  stk::mesh::Selector selectLocalParticles =
      metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::PARTICLE);
  const stk::mesh::BucketVector &particleBuckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalParticles);

  // take an Euler step
#pragma omp parallel for
  for (size_t bucket_idx = 0; bucket_idx < particleBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &particleBucket = *particleBuckets[bucket_idx];
    for (size_t particle_idx = 0; particle_idx < particleBucket.size(); ++particle_idx) {
      // fetch the entities
      stk::mesh::Entity const &particle = particleBucket[particle_idx];
      stk::mesh::Entity const node = particleBucket.begin_nodes(particle_idx)[0];

      // Euler step position
      double *node_velocity = stk::mesh::field_data(nodeVelocityField, node);
      double *coords = stk::mesh::field_data(nodeCoordField, node);
      coords[0] += dt * node_velocity[0];
      coords[1] += dt * node_velocity[1];
      coords[2] += dt * node_velocity[2];

      // Euler step orientation
      double *node_omega = stk::mesh::field_data(nodeOmegaField, node);
      double *particle_orientation = stk::mesh::field_data(particleOrientationField, particle);
      Quaternion quat(particle_orientation[0], particle_orientation[1], particle_orientation[2],
                      particle_orientation[3]);
      quat.rotate_self(node_omega[0], node_omega[1], node_omega[2], dt);
      particle_orientation[0] = quat.w;
      particle_orientation[1] = quat.x;
      particle_orientation[2] = quat.y;
      particle_orientation[3] = quat.z;
    }
  }
}

void resolve_collisions(stk::mesh::BulkData &bulkData, stk::mesh::Field<double> &nodeCoordField,
                        stk::mesh::Field<double> &nodeVelocityField, stk::mesh::Field<double> &nodeOmegaField,
                        stk::mesh::Field<double> &nodeForceField, stk::mesh::Field<double> &nodeTorqueField,
                        stk::mesh::Field<double> &particleOrientationField,
                        stk::mesh::Field<double> &particleRadiusField, stk::mesh::Field<double> &linkerSignedSepField,
                        stk::mesh::Field<double> &linkerSignedSepDotField,
                        stk::mesh::Field<double> &linkerSignedSepDotTmpField,
                        stk::mesh::Field<double> &linkerLagMultField, stk::mesh::Field<double> &linkerLagMultTmpField,
                        stk::mesh::Field<double> &conLocField, stk::mesh::Field<double> &conNormField,
                        const double viscosity, const double dt, const double con_tol, const int con_ite_max) {
  // Matrix-free BBPGD
  int ite_count = 0;

  // compute gkm1 = D^T M D xkm1

  // compute F = D xkm1
  compute_constraint_center_of_mass_force_torque(bulkData, nodeForceField, nodeTorqueField, linkerLagMultTmpField,
                                                 conNormField, conLocField);

  // compute U = M F
  compute_the_mobility_problem(bulkData, particleOrientationField, particleRadiusField, nodeForceField, nodeTorqueField,
                               nodeVelocityField, nodeOmegaField, viscosity);

  // compute gkm1 = D^T U
  compute_rate_of_change_of_sep(bulkData, nodeVelocityField, nodeOmegaField, conLocField, conNormField,
                                linkerSignedSepDotTmpField);

  ///////////////////////
  // check convergence //
  ///////////////////////
  // res = max(abs(projectPhi(gkm1)));
  double maximum_abs_projected_sep = -1.0;
  compute_maximum_abs_projected_sep(bulkData, linkerLagMultTmpField, linkerSignedSepField, linkerSignedSepDotTmpField,
                                    dt, maximum_abs_projected_sep);

  if (bulkData.parallel_rank() == 0) {
    std::cout << "maximum_abs_projected_sep " << maximum_abs_projected_sep << std::endl;
  }

  ///////////////////////
  // loop if necessary //
  ///////////////////////
  if (maximum_abs_projected_sep < con_tol) {
    // the initial guess was correct, nothing more is necessary
  } else {
    // initial guess insufficient, iterate

    // first step, Dai&Fletcher2005 Section 5.
    double alpha = 1.0 / maximum_abs_projected_sep;
    while (ite_count < con_ite_max) {
      ++ite_count;

      // compute xk = xkm1 - alpha * gkm1;
      // and perform the bound projection xk = boundProjection(xk)
      update_con_gammas(bulkData, linkerLagMultField, linkerLagMultTmpField, linkerSignedSepField,
                        linkerSignedSepDotField, dt, alpha);

      // compute new grad with xk: gk = D^T M D xk
      // compute F = D xk
      compute_constraint_center_of_mass_force_torque(bulkData, nodeForceField, nodeTorqueField, linkerLagMultField,
                                                     conNormField, conLocField);

      // compute U = M F
      compute_the_mobility_problem(bulkData, particleOrientationField, particleRadiusField, nodeForceField,
                                   nodeTorqueField, nodeVelocityField, nodeOmegaField, viscosity);

      // compute gk = D^T U
      compute_rate_of_change_of_sep(bulkData, nodeVelocityField, nodeOmegaField, conLocField, conNormField,
                                    linkerSignedSepDotField);

      ///////////////////////
      // check convergence //
      // res = max(abs(projectPhi(gk)));
      compute_maximum_abs_projected_sep(bulkData, linkerLagMultField, linkerSignedSepField, linkerSignedSepDotField, dt,
                                        maximum_abs_projected_sep);

      if (bulkData.parallel_rank() == 0) {
        std::cout << "maximum_abs_projected_sep " << maximum_abs_projected_sep << std::endl;
      }

      if (maximum_abs_projected_sep < con_tol) {
        // con_gammas worked
        // exit the loop
        break;
      }

      ///////////////////////////////////////////////////////////////////////////
      // compute dot(xkdiff, xkdiff), dot(xkdiff, gkdiff), dot(gkdiff, gkdiff) //
      // where xkdiff = xk - xkm1 and gkdiff = gk - gkm1                       //
      ///////////////////////////////////////////////////////////////////////////
      double global_dot_xkdiff_xkdiff = 0.0;
      double global_dot_xkdiff_gkdiff = 0.0;
      double global_dot_gkdiff_gkdiff = 0.0;
      compute_diff_dots(bulkData, linkerLagMultField, linkerLagMultTmpField, linkerSignedSepDotField,
                        linkerSignedSepDotTmpField, dt, global_dot_xkdiff_xkdiff, global_dot_xkdiff_gkdiff,
                        global_dot_gkdiff_gkdiff);

      ////////////////////////////////////////////
      // compute the Barzilai-Borwein step size //
      ////////////////////////////////////////////
      // alternating bb1 and bb2 methods
      double a;
      double b;
      if (ite_count % 2 == 0) {
        // Barzilai-Borwein step size Choice 1
        a = global_dot_xkdiff_xkdiff;
        b = global_dot_xkdiff_gkdiff;
      } else {
        // Barzilai-Borwein step size Choice 2
        a = global_dot_xkdiff_gkdiff;
        b = global_dot_gkdiff_gkdiff;
      }

      if (std::abs(b) < epsilon) {
        b += epsilon;  // prevent div 0 error
      }

      alpha = a / b;

      /////////////////////////////////
      // set xkm1 = xk and gkm1 = gk //
      /////////////////////////////////
      swap_con_gammas(bulkData, linkerLagMultField, linkerLagMultTmpField, linkerSignedSepDotField,
                      linkerSignedSepDotTmpField);
    }
  }

  if (bulkData.parallel_rank() == 0) {
    std::cout << "Num BBPGD iterations: " << ite_count << std::endl;
  }

  // take an Euler step
  // the collision solver already updates the velocity of each particle
  step_euler(bulkData, nodeVelocityField, nodeOmegaField, nodeCoordField, particleOrientationField, dt);
}

///////////////////////////
// Partitioning settings //
///////////////////////////
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
    return std::string("coordinates");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // RcbSettings

class ParMETISSettings : public stk::balance::BalanceSettings {
 public:
  ParMETISSettings() {
  }
  virtual ~ParMETISSettings() {
  }

  virtual bool isIncrementalRebalance() const {
    return false;
  }
  virtual std::string getDecompMethod() const {
    return std::string("parmetis");
  }
  virtual std::string getCoordinateFieldName() const {
    return std::string("coordinates");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // ParMETISSettings

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <box_size> <global_num_particles>" << std::endl;
    return 1;
  }

  double box_size = std::stod(argv[1]);
  size_t num_particles_global = std::stoi(argv[2]);

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  printf("Process %d out of %d processes\n", world_rank, world_size);

  /*
  Initialization procedure
    1. generate N random particles across all processors
    2. balance the particles
    3. perform AABB neighbor detection
    4. generate collision constraints/linkers between neighbors
    5. rebalance the particles/linkers
    6. perform collision resolution using matrix-free BBPGD
    7. write out the initial configuration

  Timestepping procedure
    1. balance the particles, if necessary
    2. take an unconstrained timestep
    2. perform AABB neighbor detection
    3. generate collision constraints/linkers between new neighbors (reuse the old neighbor list)
    4. perform collision resolution using matrix-free BBPGD
    5. write out the results, if necessary
  */

  // Simulation params
  const double viscosity = 0.001;
  const double dt = 5e-3;
  const double time_snap = 5e-3;
  const double time_stop = 1;

  const double R = 0.133;
  const double cutoff = 2 * R;
  const double con_tol = 1e-5;
  const int con_ite_max = 1000;
  const unsigned int spatial_dimension = 3;

  const double domain_low[3] = {0.0, 0.0, 0.0};
  const double domain_high[3] = {box_size, box_size, box_size};

  // build the meta data
  // particles have a single node
  // linkers have 4 nodes, one shared with each connected particle
  // and two for the physical points of contact between linker and particle
  stk::topology particle_top = stk::topology::PARTICLE;
  stk::topology link_top = stk::topology::BEAM_2;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(spatial_dimension);
  builder.set_entity_rank_names({"node", "edge", "face", "elem"});

  std::shared_ptr<stk::mesh::BulkData> bulkPtr = builder.create();
  bulkPtr->mesh_meta_data().use_simple_fields();

  // declare the parts
  stk::mesh::MetaData &metaData = bulkPtr->mesh_meta_data();
  stk::mesh::Part &linkerPart = metaData.declare_part_with_topology("linePart", link_top);
  stk::mesh::Part &particlePart = metaData.declare_part_with_topology("particlePart", particle_top);
  stk::io::put_io_part_attribute(linkerPart);
  stk::io::put_io_part_attribute(particlePart);

  // declare and assign fields
  // node fields (shared between particles and linkers)
  stk::mesh::Field<double> &nodeCoordField =
      metaData.declare_field<double>(stk::topology::NODE_RANK, "coordinates", spatial_dimension);
  stk::mesh::Field<double> &nodeVelocityField =
      metaData.declare_field<double>(stk::topology::NODE_RANK, "velocity", spatial_dimension);
  stk::mesh::Field<double> &nodeOmegaField =
      metaData.declare_field<double>(stk::topology::NODE_RANK, "omega", spatial_dimension);
  stk::mesh::Field<double> &nodeForceField =
      metaData.declare_field<double>(stk::topology::NODE_RANK, "force", spatial_dimension);
  stk::mesh::Field<double> &nodeTorqueField =
      metaData.declare_field<double>(stk::topology::NODE_RANK, "torque", spatial_dimension);
  stk::mesh::put_field_on_entire_mesh(nodeCoordField, spatial_dimension);
  stk::mesh::put_field_on_entire_mesh(nodeVelocityField, spatial_dimension);
  stk::mesh::put_field_on_entire_mesh(nodeOmegaField, spatial_dimension);
  stk::mesh::put_field_on_entire_mesh(nodeForceField, spatial_dimension);
  stk::mesh::put_field_on_entire_mesh(nodeTorqueField, spatial_dimension);

  // element fields (stored on both particles and linkers)
  stk::mesh::Field<int> &elemRankField = metaData.declare_field<int>(stk::topology::ELEMENT_RANK, "rank");
  stk::mesh::Field<double> &elemAabbField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "aabb", 2 * spatial_dimension);
  stk::mesh::put_field_on_entire_mesh(elemRankField);
  stk::mesh::put_field_on_entire_mesh(elemAabbField, 2 * spatial_dimension);

  // element fields (stored only on particles)
  stk::mesh::Field<double> &particleOrientationField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "orientation", spatial_dimension + 1);
  stk::mesh::Field<double> &particleRadiusField = metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");
  stk::mesh::put_field_on_mesh(particleOrientationField, particlePart, spatial_dimension + 1, nullptr);
  stk::mesh::put_field_on_mesh(particleRadiusField, particlePart, nullptr);

  // element fields (stored only on linkers)
  stk::mesh::Field<double> &linkerSignedSepField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "signed_sep_dist");
  stk::mesh::Field<double> &linkerSignedSepDotField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "signed_sep_dist_dot");
  stk::mesh::Field<double> &linkerSignedSepDotTmpField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "signed_sep_dist_dot_tmp");
  stk::mesh::Field<double> &linkerLagMultField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "lagrange_multiplier");
  stk::mesh::Field<double> &linkerLagMultTmpField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "lagrange_multiplier_tmp");
  stk::mesh::Field<double> &conLocField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "constraint_attachment_locs", 2 * spatial_dimension);
  stk::mesh::Field<double> &conNormField =
      metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "constraint_attachment_norms", 2 * spatial_dimension);
  stk::mesh::put_field_on_mesh(linkerSignedSepField, linkerPart, nullptr);
  stk::mesh::put_field_on_mesh(linkerSignedSepDotField, linkerPart, nullptr);
  stk::mesh::put_field_on_mesh(linkerSignedSepDotTmpField, linkerPart, nullptr);
  stk::mesh::put_field_on_mesh(linkerLagMultField, linkerPart, nullptr);
  stk::mesh::put_field_on_mesh(linkerLagMultTmpField, linkerPart, nullptr);
  stk::mesh::put_field_on_mesh(conLocField, linkerPart, 2 * spatial_dimension, nullptr);
  stk::mesh::put_field_on_mesh(conNormField, linkerPart, 2 * spatial_dimension, nullptr);

  metaData.set_coordinate_field_name("coordinates");
  metaData.commit();

  // construct the mesh in parallel
  stk::mesh::BulkData &bulkData = *bulkPtr;

  // get the averge number of particles per process
  size_t num_particles_local = num_particles_global / bulkData.parallel_size();

  // num_particles_local isn't guarenteed to divide perfectly
  // add the extra workload to the first r ranks
  size_t remaining_particles = num_particles_global - num_particles_local * bulkData.parallel_size();
  if (bulkData.parallel_rank() < remaining_particles) {
    num_particles_local += 1;
  }

  bulkData.modification_begin();

  std::vector<size_t> requests(metaData.entity_rank_count(), 0);
  const size_t num_nodes_requested = num_particles_local * particle_top.num_nodes();
  const size_t num_elems_requested = num_particles_local;
  requests[stk::topology::NODE_RANK] = num_nodes_requested;
  requests[stk::topology::ELEMENT_RANK] = num_elems_requested;

  // ex.
  //  requests = { 0, 4,  8}
  //  requests 0 entites of rank 0, 4 entites of rank 1, and 8 entites of rank 2
  //  requested_entities = {0 entites of rank 0, 4 entites of rank 1, 8 entites of rank 2}
  std::vector<stk::mesh::Entity> requested_entities;
  bulkData.generate_new_entities(requests, requested_entities);

  // associate each particle with a single part
  std::vector<stk::mesh::Part *> add_particlePart(1);
  add_particlePart[0] = &particlePart;

  // set topologies of new entities
  for (int i = 0; i < num_particles_local; i++) {
    stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
    bulkData.change_entity_parts(particle_i, add_particlePart);
  }

  // the elements should be associated with a topology before they are connected to their nodes/edges
  // set downward relations of entities
  for (int i = 0; i < num_particles_local; i++) {
    stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
    bulkData.declare_relation(particle_i, requested_entities[i], 0);
  }

  bulkData.modification_end();

  // initialize
  {
    const unsigned int rank = bulkData.parallel_rank();

    stk::mesh::Selector selectLocalParticles =
        metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::PARTICLE);
    const stk::mesh::BucketVector &particleBuckets =
        bulkData.get_buckets(stk::topology::ELEMENT_RANK, selectLocalParticles);
#pragma omp parallel for
    for (size_t bucket_idx = 0; bucket_idx < particleBuckets.size(); ++bucket_idx) {
      stk::mesh::Bucket &particleBucket = *particleBuckets[bucket_idx];
      for (size_t particle_idx = 0; particle_idx < particleBucket.size(); ++particle_idx) {
        // set entity values
        stk::mesh::Entity const &particle = particleBucket[particle_idx];

        // create a random number generator independent of the number of processors
        std::mt19937 gen(bulkData.identifier(particle));
        std::uniform_real_distribution uniform_dist01(0.0, 1.0);

        // store the rank
        int *particle_rank = stk::mesh::field_data(elemRankField, particle);
        particle_rank[0] = rank;

        // random initial orientation
        double *particle_orientation = stk::mesh::field_data(particleOrientationField, particle);
        const double u1 = uniform_dist01(gen);
        const double u2 = uniform_dist01(gen);
        const double u3 = uniform_dist01(gen);
        Quaternion quat(u1, u2, u3);
        particle_orientation[0] = quat.w;
        particle_orientation[1] = quat.x;
        particle_orientation[2] = quat.y;
        particle_orientation[3] = quat.z;

        // initial radius
        double *particle_radius = stk::mesh::field_data(particleRadiusField, particle);
        particle_radius[0] = R;

        // set node values
        unsigned num_nodes = particleBucket.num_nodes(particle_idx);
        stk::mesh::Entity node = particleBucket.begin_nodes(particle_idx)[0];

        // random initial position
        double *coords = stk::mesh::field_data(nodeCoordField, node);
        coords[0] = uniform_dist01(gen) * (domain_high[0] - domain_low[0]) + domain_low[0];
        coords[1] = uniform_dist01(gen) * (domain_high[1] - domain_low[1]) + domain_low[1];
        coords[2] = uniform_dist01(gen) * (domain_high[2] - domain_low[2]) + domain_low[2];

        // bounding box
        double *aabb = stk::mesh::field_data(elemAabbField, particle);
        aabb[0] = coords[0] - R;
        aabb[1] = coords[1] - R;
        aabb[2] = coords[2] - R;
        aabb[3] = coords[0] + R;
        aabb[4] = coords[1] + R;
        aabb[5] = coords[2] + R;

        // clear contents
        double *node_omega = stk::mesh::field_data(nodeOmegaField, node);
        node_omega[0] = 0.0;
        node_omega[1] = 0.0;
        node_omega[2] = 0.0;

        double *node_velocity = stk::mesh::field_data(nodeVelocityField, node);
        node_velocity[0] = 0.0;
        node_velocity[1] = 0.0;
        node_velocity[2] = 0.0;

        double *node_force = stk::mesh::field_data(nodeForceField, node);
        node_force[0] = 0.0;
        node_force[1] = 0.0;
        node_force[2] = 0.0;

        double *node_torque = stk::mesh::field_data(nodeTorqueField, node);
        node_torque[0] = 0.0;
        node_torque[1] = 0.0;
        node_torque[2] = 0.0;
      }
    }
  }

  // // setup the stk io
  // stk::io::StkMeshIoBroker stkIo;
  // stkIo.use_simple_fields();
  // stkIo.set_bulk_data(bulkData);
  // {
  //   // write out the unbalanced results
  //   std::string filename = "mundy-particles-unbalanced.exo";
  //   size_t outputFileIndex = stkIo.create_output_mesh(filename, stk::io::WRITE_RESULTS);

  //   // node fields
  //   stkIo.add_field(outputFileIndex, nodeCoordField);
  //   stkIo.add_field(outputFileIndex, nodeVelocityField);
  //   stkIo.add_field(outputFileIndex, nodeOmegaField);
  //   stkIo.add_field(outputFileIndex, nodeForceField);
  //   stkIo.add_field(outputFileIndex, nodeTorqueField);

  //   // element fields
  //   stkIo.add_field(outputFileIndex, elemRankField);
  //   stkIo.add_field(outputFileIndex, elemAabbField);

  //   // particle fields
  //   stkIo.add_field(outputFileIndex, particleOrientationField);
  //   stkIo.add_field(outputFileIndex, particleRadiusField);

  //   // linker fields
  //   stkIo.add_field(outputFileIndex, linkerSignedSepField);
  //   stkIo.add_field(outputFileIndex, linkerSignedSepDotField);
  //   stkIo.add_field(outputFileIndex, linkerSignedSepDotTmpField);
  //   stkIo.add_field(outputFileIndex, linkerLagMultField);
  //   stkIo.add_field(outputFileIndex, linkerLagMultTmpField);
  //   stkIo.add_field(outputFileIndex, conLocField);
  //   stkIo.add_field(outputFileIndex, conNormField);

  //   stkIo.begin_output_step(outputFileIndex, 0.0);
  //   stkIo.write_defined_output_fields(outputFileIndex);
  //   stkIo.end_output_step(outputFileIndex);
  // }

  // // perform an aperiodic stk balance
  // ParMETISSettings balanceSettings;
  // stk::balance::balanceStkMesh(balanceSettings, bulkData);

  // store the updated processor ID
  {
    const int rank = bulkData.parallel_rank();
    const stk::mesh::BucketVector &elementBuckets = bulkData.buckets(stk::topology::ELEMENT_RANK);
#pragma omp parallel for
    for (size_t bucket_idx = 0; bucket_idx < elementBuckets.size(); ++bucket_idx) {
      stk::mesh::Bucket &elemBucket = *elementBuckets[bucket_idx];
      for (size_t elem_idx = 0; elem_idx < elemBucket.size(); ++elem_idx) {
        int *element_rank = stk::mesh::field_data(elemRankField, elemBucket[elem_idx]);
        *element_rank = rank;
      }
    }
  }

  // resolve initial collisions

  // Time and memory tracking for generate_neighbor_pairs
  std::cout << "Generating neighbor pairs..." << std::endl;
  auto start_time1 = std::chrono::high_resolution_clock::now();
  SearchIdPairVector neighborPairs;
  generate_neighbor_pairs(bulkData, elemAabbField, neighborPairs);
  auto end_time1 = std::chrono::high_resolution_clock::now();
  long long total_time1 = std::chrono::duration_cast<std::chrono::microseconds>(end_time1 - start_time1).count();

  // Time and memory tracking for generate_collision_constraints
  std::cout << "Generating collision constraints..." << std::endl;
  auto start_time2 = std::chrono::high_resolution_clock::now();
  generate_collision_constraints(bulkData, neighborPairs, linkerPart, nodeCoordField, particleRadiusField,
                                 linkerSignedSepField, linkerSignedSepDotField, linkerSignedSepDotTmpField,
                                 linkerLagMultField, linkerLagMultTmpField, conLocField, conNormField);
  auto end_time2 = std::chrono::high_resolution_clock::now();
  long long total_time2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2).count();

  // perform an aperiodic stk balance
  ParMETISSettings balanceSettings;
  stk::balance::balanceStkMesh(balanceSettings, bulkData);

  // Time and memory tracking for resolve_collisions
  std::cout << "Resolving collisions..." << std::endl;
  auto start_time3 = std::chrono::high_resolution_clock::now();
  resolve_collisions(bulkData, nodeCoordField, nodeVelocityField, nodeOmegaField, nodeForceField, nodeTorqueField,
                     particleOrientationField, particleRadiusField, linkerSignedSepField, linkerSignedSepDotField,
                     linkerSignedSepDotTmpField, linkerLagMultField, linkerLagMultTmpField, conLocField, conNormField,
                     viscosity, dt, con_tol, con_ite_max);
  auto end_time3 = std::chrono::high_resolution_clock::now();
  long long total_time3 = std::chrono::duration_cast<std::chrono::microseconds>(end_time3 - start_time3).count();

  const char *labels[] = {"generate_neighbor_pairs Time (μs)", "generate_collision_constraints Time (μs)",
                          "resolve_collisions Time (μs)"};
  long long times[] = {total_time1, total_time2, total_time3};

  if (bulkData.parallel_rank() == 0) {
    // Get total number of processes
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create the filename string
    std::string filename = "timings_for_box_size_" + std::to_string(box_size) + "_and_num_particles_" +
                           std::to_string(num_particles_global) + "_with_" +
                           std::to_string(num_procs * omp_get_max_threads()) + "_procs.csv";

    // Open the file for writing
    std::ofstream outFile(filename);

    // Write headers to the file
    for (int i = 0; i < 3; i++) {
      outFile << labels[i];
      if (i != 2) outFile << ",";  // Add comma for separation except for the last label
    }
    outFile << std::endl;

    // Write timings to the file
    for (int i = 0; i < 3; i++) {
      outFile << times[i];
      if (i != 2) outFile << ",";  // Add comma for separation except for the last timing
    }
    outFile << std::endl;

    // Close the file
    outFile.close();

    // Print timings to console (optional)
    for (int i = 0; i < 3; i++) {
      std::cout << labels[i] << ": " << times[i] << std::endl;
    }
  }

  stk::parallel_print_time_without_output_and_hwm(MPI_COMM_WORLD, total_time1 + total_time2 + total_time3);

  // // write out the balanced results (with linkers)
  // {
  //   std::string filename = "mundy-particles-balanced.exo";
  //   size_t outputFileIndex = stkIo.create_output_mesh(filename, stk::io::WRITE_RESULTS);
  //   // node fields
  //   stkIo.add_field(outputFileIndex, nodeCoordField);
  //   stkIo.add_field(outputFileIndex, nodeVelocityField);
  //   stkIo.add_field(outputFileIndex, nodeOmegaField);
  //   stkIo.add_field(outputFileIndex, nodeForceField);
  //   stkIo.add_field(outputFileIndex, nodeTorqueField);

  //   // element fields
  //   stkIo.add_field(outputFileIndex, elemRankField);
  //   stkIo.add_field(outputFileIndex, elemAabbField);

  //   // particle fields
  //   stkIo.add_field(outputFileIndex, particleOrientationField);
  //   stkIo.add_field(outputFileIndex, particleRadiusField);

  //   // linker fields
  //   stkIo.add_field(outputFileIndex, linkerSignedSepField);
  //   stkIo.add_field(outputFileIndex, linkerSignedSepDotField);
  //   stkIo.add_field(outputFileIndex, linkerSignedSepDotTmpField);
  //   stkIo.add_field(outputFileIndex, linkerLagMultField);
  //   stkIo.add_field(outputFileIndex, linkerLagMultTmpField);
  //   stkIo.add_field(outputFileIndex, conLocField);
  //   stkIo.add_field(outputFileIndex, conNormField);

  //   stkIo.begin_output_step(outputFileIndex, 0.0);
  //   stkIo.write_defined_output_fields(outputFileIndex);
  //   stkIo.end_output_step(outputFileIndex);
  // }

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}