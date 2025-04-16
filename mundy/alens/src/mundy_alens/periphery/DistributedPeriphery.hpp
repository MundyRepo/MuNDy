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

#ifndef MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_DISTRIBUTEDPERIPHERY_HPP_
#define MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_DISTRIBUTEDPERIPHERY_HPP_

/* This class evaluates the fluid flow on some points within the domain induced by the no-slip confition on the
periphery. We do so via storing the precomputed inverse of the self-interaction matrix. Ths class is in charge
of writing out that inverse if it doesn't already exist, reading it into memory if it does, evaluating the surface
forces induced by external flow on the surface (by evaluating f = M^{-1}u), and then evaluating the fluid flow at points
within the domain induced by these surface forces.

The periphery is described by a collection of nodes with inward-pointing surface normals and predefined quadrature
weights. We are not in charge of computing these quantities.

We will write this entire class with Tpetra/Belos for matrix inversion and STKFMM for fast, parallel kernel evaluation.
*/

// C++ core
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

// Mundy
#include <mundy_alens/periphery/Gauss_Legendre_Nodes_and_Weights.hpp>  // for Gauss_Legendre_Nodes_and_Weights
#include <mundy_core/throw_assert.hpp>                                 // for MUNDY_THROW_ASSERT
#define DOUBLE_ZERO 1.0e-12

namespace mundy {

namespace alens {

namespace periphery {

/*
Goals:

Our goal here is to solve the Stokes Fredholm integral of the second kind for the hydrodynamic flow induced by
satisfying no-slip on a periphery.

The periphery is given to us as a collection of nodes with some positions, quadrature weights, and surface normals.

We must take in the uncorrected surface velocity at these nodes and solve for the surface forces at these nodes that
will negate this surface velocity.

In any case, all quantities are distributes so solving for the surface force will require a distributed matrix inversion
with a matrix-free operator. We'll use Tpetra/Belos for this purpose. All kernel evaluations will be done with STKFMM.

Mathematically, we are solving f = M^-1 u = (1/2 I + T + N)^-1 u

M f in a matrix-free sense is

u = T(f) + SS(f) + N(f)

T(f) = -3 viscosity / (4*pi) * r_i * r_j * r_k * normal_k f_j / r**5 can be computed by evaluating the STKFMM Traction
kernel applied to n_k f_i as the so called double layer. Note, STKFMM does not scale by viscosity and expects the double
layer to be passed in in flattened row major form.

SS(f) can be computed by evaluating the STKFMM traction kernel to e1, e2, and e3, which are the basis vectors at each
node concatenated together e1 = [1, 0, 0, 1, 0, 0, 1, 0, 0, ...] e2 = [0, 1, 0, 0, 1, 0, 0, 1, 0, ...] e3 = [0, 0, 1, 0,
0, 1, 0, 0, 1, ...] They have the same size as f. This gives us w1 = T(e1), w2 = T(e2), w3 = T(e3).

SS(f) can then be evaluating for node i as SS(f)_i = [w1_i w2_i w3_i] f_i, which is a 3x3 times 3x1 matrix vector
multiplication with w1_i, w2_i, and w3_i as the columns of the matrix.

N(f) = integral_{S} (normal(t) * normal(s) dot f(s) ds) = normal(t) integral_{S} normal(s) dot f(s) ds

C = integral_{S} normal(s) dot f(s) ds is a scalar-valued constant independent of the target normal.
If we weight f by the quadrature weights, then C = normal dot f_weighted, which can easily be computed directly with
Tpetra.

u_{i}= G_{ij}f_j
+ \frac{1}{8 \pi \mu}\left(-\frac{r_{i}}{r^{3}} trD\right)
+ \frac{1}{8 \pi \mu}\left[-\frac{3 r_{i} r_{j} r_{k}}{r^{5}}\right] D_{j k}
*/

using SL = double;
using LO = Tpetra::Vector<>::local_ordinal_type;
using GO = Tpetra::Vector<>::global_ordinal_type;
using NODE = Tpetra::Vector<>::node_type;
using TMAP = Tpetra::Map<LO, GO, NODE>;
using TOP = Tpetra::Operator<SL, LO, GO, NODE>;
using TCOMM = Teuchos::Comm<int>;
using TMV = Tpetra::MultiVector<SL, LO, GO, NODE>;
using TV = Tpetra::Vector<SL, LO, GO, NODE>;
using TCMAT = Tpetra::CrsMatrix<SL, LO, GO, NODE>;

class NoSlipPeripheryTpetraOp : public TOP {
 public:
  NoSlipPeripheryTpetraOp(const double &viscocity, const std::array<double, 3> &periphery_bounding_box_low,
                          const std::array<double, 3> &periphery_bounding_box_high, const TMAP &periphery_scalar_map,
                          const Tuechos::RCP<TMAP> &periphery_vector_map_rcp,
                          const Tuechos::RCP<TV> &surface_coords_rcp, const Tuechos::RCP<TV> &surface_normals_rcp,
                          const Tuechos::RCP<TV> &surface_weights_rcp)
      : viscocity_(viscocity),
        periphery_scalar_map_rcp_(periphery_scalar_map_rcp),
        periphery_vector_map_rcp_(periphery_vector_map_rcp),
        surface_coords_rcp_(surface_coords_rcp),
        surface_normals_rcp_(surface_normals_rcp),
        surface_weights_rcp_(surface_weights_rcp) {
    MUNDY_THROW_REQUIRE(!periphery_scalar_map_rcp_.is_null() == false, std::invalid_argument,
                        "The periphery map must not be a null.");
    MUNDY_THROW_REQUIRE(!periphery_vector_map_rcp_.is_null() == false, std::invalid_argument,
                        "The periphery map must not be a null.");
    MUNDY_THROW_REQUIRE(!surface_coords_rcp_.is_null() == false, std::invalid_argument,
                        "The surface coordinates must not be a null.");
    MUNDY_THROW_REQUIRE(!surface_normals_rcp_.is_null() == false, std::invalid_argument,
                        "The surface normals must not be a null.");
    MUNDY_THROW_REQUIRE(!surface_weights_rcp_.is_null() == false, std::invalid_argument,
                        "The surface weights must not be a null.");
    MUNDY_THROW_ASSERT(surface_coords_rcp_->getMap()->isSameAs(*periphery_vector_map_rcp_), std::invalid_argument,
                       "The surface coordinates must have the same map as the periphery vector map.");
    MUNDY_THROW_ASSERT(surface_normals_rcp_->getMap()->isSameAs(*periphery_vector_map_rcp_), std::invalid_argument,
                       "The surface normals must have the same map as the periphery vector map.");
    MUNDY_THROW_ASSERT(surface_weights_rcp_->getMap()->isSameAs(*periphery_scalar_map_rcp_), std::invalid_argument,
                       "The surface weights must have the same map as the periphery scalar map.");

    const size_t num_surface_points = surface_weights_rcp_->getLocalLength();
    MUNDY_THROW_ASSERT(surface_coords_rcp_->getLocalLength() == 3 * num_surface_points, std::invalid_argument,
                       "The surface coordinates must have size 3 * num_surface_points.");
    MUNDY_THROW_ASSERT(surface_normals_rcp_->getLocalLength() == 3 * num_surface_points, std::invalid_argument,
                       "The surface normals must have size 3 * num_surface_points.");

    // Initialize the STKFMM object
    const int fmm_multipole_order = 8;
    const int max_num_leaf_pts = 2000;
    const stkfmm::PAXIS pbc = stkfmm::PAXIS::NONE;
    fmm_evaluator_ptr_ =
        std::make_shared<stkfmm::Stk3DFMM>(fmm_multipole_order, max_num_leaf_pts, pbc, asInteger(KERNEL::PVel));

    // FMM must be done in a cubic domain, so we need to expand the periphery bounding box to a cube.
    const double cube_center[3] = {0.5 * (periphery_bounding_box_low[0] + periphery_bounding_box_high[0]),
                                   0.5 * (periphery_bounding_box_low[1] + periphery_bounding_box_high[1]),
                                   0.5 * (periphery_bounding_box_low[2] + periphery_bounding_box_high[2])};
    const double cube_width = std::max({periphery_bounding_box_high[0] - periphery_bounding_box_low[0],
                                        periphery_bounding_box_high[1] - periphery_bounding_box_low[1],
                                        periphery_bounding_box_high[2] - periphery_bounding_box_low[2]});
    const cube_lower_left[3] = {cube_center[0] - 0.5 * cube_width, cube_center[1] - 0.5 * cube_width,
                                cube_center[2] - 0.5 * cube_width};
    fmm_evaluator_ptr_->setBox(cube_lower_left, cube_width);

    // Compute the initial fmm tree
    // The PVel maps single and double layer sources to the pressure and velocity at the target according to
    //  u_{i} = G_{ij}f_j
    //        + \frac{1}{8 \pi \mu}\left(-\frac{r_{i}}{r^{3}} trD\right)
    //        + \frac{1}{8 \pi \mu}\left[-\frac{3 r_{i} r_{j} r_{k}}{r^{5}}\right] D_{j k}
    // We are interested in evaluating T(f), which is slightly
    src_coords_.clear();
    src_single_layer_values_.clear();
    src_double_layer_values_.clear();
    trg_coords_.clear();
    trg_values_.clear();
    src_coords_.resize(3 * num_surface_points);
    src_single_layer_values_.resize(4 * num_surface_points);  // 4 values per point ()
    src_double_layer_values_.resize(9 * num_surface_points);  // 3x3 matrix per point n_i f_j
    trg_coords_.resize(3 * num_surface_points);
  }

  void apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
             scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
             scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const override {
    // We are solving for f = M^-1 u = (1/2 I + T + N)^-1 u
    MUNDY_THROW_ASSERT(X.getNumVectors() == Y.getNumVectors(), std::invalid_argument,
                       "X and Y must have the same number of vectors.");
    MUNDY_THROW_ASSERT(X.getMap()->isSameAs(*periphery_vector_map_), std::invalid_argument,
                       "X must have the same map as the periphery vector map.");
    MUNDY_THROW_ASSERT(Y.getMap()->isSameAs(*periphery_vector_map_), std::invalid_argument,
                       "Y must have the same map as the periphery vector map.");

    // Apply the perphery to each of the vectors in the MV
    const int num_vecs = X.getNumVectors();
    for (int i = 0; i < num_vecs; ++i) {
    }
  }

  Teuchos::RCP<const TMAP> getDomainMap() const {
    return periphery_vector_map_rcp_;
  }

  Teuchos::RCP<const TMAP> getRangeMap() const {
    return periphery_vector_map_rcp_;
  }

  bool hasTransposeApply() const {
    return false;
  }

 private:
  //! \name Private member variables
  //@{

  // Physical constants
  const double viscocity_;  ///< The viscocity of the fluid.

  // Tpetra objects
  const Teuchos::RCP<TMAP> &periphery_map_rcp_;  ///< The Tpetra map for the periphery.
  const Teuchos::RCP<TV> &surface_coords_rcp_;   ///< The Tpetra vector for the surface coordinates.
  const Teuchos::RCP<TV> &surface_normals_rcp_;  ///< The Tpetra vector for the surface normals.
  const Teuchos::RCP<TV> &surface_weights_rcp_;  ///< The Tpetra vector for the surface weights.

  // FMM objects
  std::shared_ptr<stkfmm::Stk3DFMM> fmm_evaluator_ptr_;  ///< The STKFMM object.
  mutable std::vector<double> src_double_layer_coords_;  ///< Scratch space for the double layer source coordinates.
  mutable std::vector<double> src_double_layer_values_;  ///< Scratch space for the double layer source values.
  mutable std::vector<double> trg_coords_;               ///< Scratch space for the target coordinates.
  mutable std::vector<double> trg_values_;               ///< Scratch space for the target values.
  //@}
};

class NoSlipPeripherySTKOp {
 public:
  NoSlipPeripherySTKOp(const double &viscocity, stk::mesh::BulkData &bulk_data, stk::mesh::Selector &periphery_selector,
                       stk::mesh::Field<double> &surface_coords, stk::mesh::Field<double> &surface_normals,
                       stk::mesh::Field<double> &surface_weights)
      : viscocity_(viscocity),
        bulk_data_(bulk_data),
        periphery_selector_(periphery_selector),
        surface_coords_(surface_coords),
        surface_normals_(surface_normals),
        surface_weights_(surface_weights) {
    // Generate a Tpetra map for our locally owned chunk of the operator's domain and range space.
    locally_owned_entities.clear();
    stk::mesh::get_selected_entities(selector & bulk_data_.mesh_meta_data().locally_owned_part(),
                                     bulk_data_.buckets(stk::topology::NODE_RANK), locally_owned_entities);

    const size_t num_surface_nodes = locally_owned_entities.size();
    auto teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(bulk_data_.parallel()));
    periphery_vector_map_rcp_ =
        Teuchos::rcp(new TMAP(Teuchos::OrdinalTraits<GO>::invalid(), 3 * num_surface_nodes, 0, teuchos_comm));
    periphery_scalar_map_rcp_ =
        Teuchos::rcp(new TMAP(Teuchos::OrdinalTraits<GO>::invalid(), num_surface_nodes, 0, teuchos_comm));

    // Copy our surface coordinates, normals, and weights into Tpetra vectors.
    surface_coords_rcp_ = Teuchos::rcp(new TV(periphery_map_rcp_));
    surface_normals_rcp_ = Teuchos::rcp(new TV(periphery_map_rcp_));
    surface_weights_rcp_ = Teuchos::rcp(new TV(periphery_scalar_map_rcp_));

    auto surface_coords_view = surface_coords_rcp_->getLocalView<Kokkos::HostSpace>(Tpetra::Access::WriteOnly);
    auto surface_normals_view = surface_normals_rcp_->getLocalView<Kokkos::HostSpace>(Tpetra::Access::WriteOnly);
    auto surface_weights_view = surface_weights_rcp_->getLocalView<Kokkos::HostSpace>(Tpetra::Access::WriteOnly);
    for (size_t i = 0; i < num_surface_nodes; i++) {
      const stk::mesh::Entity &node = locally_owned_entities[i];
      const double *coords = stk::mesh::field_data(surface_coords_, node);
      const double *normals = stk::mesh::field_data(surface_normals_, node);
      const double *weights = stk::mesh::field_data(surface_weights_, node);

      for (size_t j = 0; j < 3; j++) {
        surface_coords_view(3 * i + j, 0) = coords[j];
        surface_normals_view(3 * i + j, 0) = normals[j];
      }
      surface_weights_view(i, 0) = weights[0];
    }

    // Build our Tpetra operator.
    tpetra_op_rcp_ = Teuchos::rcp(new NoSlipPeripheryTpetraOp(viscocity_, periphery_map_rcp_, surface_coords_rcp_,
                                                              surface_normals_rcp_, surface_weights_rcp_));
  }

 private:
  //! \name Private member variables
  //@{

  // Physical constants
  const double viscocity_;  ///< The viscocity of the fluid.

  // STK objects
  stk::mesh::BulkData &bulk_data_;                        ///< The bulk data object.
  stk::mesh::Selector &periphery_selector_;               ///< The selector for the periphery.
  stk::mesh::Field<double> &surface_coords_;              ///< The field for the surface coordinates.
  stk::mesh::Field<double> &surface_normals_;             ///< The field for the surface normals.
  stk::mesh::Field<double> &surface_weights_;             ///< The field for the surface weights.
  std::vector<stk::mesh::Entity> locally_owned_entities;  ///< The entities in the periphery.

  // Tpetra objects
  Teuchos::RCP<TMAP> periphery_map_rcp_;         ///< The Tpetra map for the periphery.
  Teuchos::RCP<TMAP> periphery_vector_map_rcp_;  ///< The Tpetra map for the periphery vectors.
  Teuchos::RCP<TMAP> periphery_scalar_map_rcp_;  ///< The Tpetra map for the periphery scalars.
  Teuchos::RCP<TV> surface_coords_rcp_;          ///< The Tpetra vector for the surface coordinates.
  Teuchos::RCP<TV> surface_normals_rcp_;         ///< The Tpetra vector for the surface normals.
  Teuchos::RCP<TV> surface_weights_rcp_;         ///< The Tpetra vector for the surface weights.
  Teuchos::RCP<TOP> tpetra_op_rcp_;              ///< The Tpetra operator for the periphery.
  //@}
};

}  // namespace periphery

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_INCLUDE_MUNDY_ALENS_PERIPHERY_DISTRIBUTEDPERIPHERY_HPP_
