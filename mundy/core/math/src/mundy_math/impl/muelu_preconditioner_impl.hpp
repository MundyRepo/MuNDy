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

#ifndef MUNDY_MATH_IMPL_MUELU_PRECONDITIONER_IMPL_HPP_
#define MUNDY_MATH_IMPL_MUELU_PRECONDITIONER_IMPL_HPP_

/// \file muelu_preconditioner_impl.hpp
/// \brief Tpetra/MueLu machinery behind muelu_preconditioner.hpp: a sparse matrix wrapped for Tpetra, its MueLu
/// hierarchy, and the vectors one cycle reads and writes. This header depends on no public muelu_preconditioner.hpp
/// types.

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{MUELU,TPETRA,KOKKOSKERNELS}

#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <cstddef>
#include <type_traits>

// Tpetra / Teuchos / Xpetra / MueLu:
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Xpetra_TpetraMultiVector.hpp>  // for Xpetra::toXpetra

// Mundy:
#include <mundy_math/impl/solver_backends_impl.hpp>  // for mundy::impl::SparseMatView
#include <mundy_math/impl/tpetra_impl.hpp>  // for mundy::impl::{make_serial_map, load_view_into_mv, extract_mv_into_view}
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace impl {

/// \brief A sparse matrix A as Tpetra sees it, the MueLu hierarchy built for it, and the vectors a cycle reads and
/// writes. Copies share all of it.
///
/// The Tpetra matrix views A when A's type is Tpetra's local matrix type and holds a copy of A otherwise.
template <class Backend, class Matrix>
  requires requires { typename Backend::exec_space; }
class MueLuSession {
 public:
  using LO = typename Tpetra::Map<>::local_ordinal_type;
  using GO = typename Tpetra::Map<>::global_ordinal_type;
  using NO = typename Tpetra::Map<>::node_type;
  using scalar_t = typename Matrix::non_const_value_type;
  using map_type = Tpetra::Map<LO, GO, NO>;
  using crs_type = Tpetra::CrsMatrix<scalar_t, LO, GO, NO>;
  using local_matrix_type = typename crs_type::local_matrix_device_type;
  using mv_type = Tpetra::MultiVector<scalar_t, LO, GO, NO>;
  using op_type = MueLu::TpetraOperator<scalar_t, LO, GO, NO>;

  static_assert(SparseMatView<Matrix>, "MueLu preconditioner: A must be a sparse matrix.");
  static_assert(std::is_same_v<typename Backend::exec_space, typename NO::execution_space>,
                "MueLu preconditioner: Backend::exec_space must match the Tpetra default Node execution space (build "
                "Trilinos and Mundy against the same Kokkos execution space).");

  MueLuSession(const Matrix& A, Teuchos::ParameterList params) : map_(make_serial_map<LO, GO, NO>(A.numRows())) {
    build(A, params);
  }

  template <class NearNullSpace>
  MueLuSession(const Matrix& A, const NearNullSpace& near_null_space, Teuchos::ParameterList params)
      : map_(make_serial_map<LO, GO, NO>(A.numRows())) {
    params.sublist("user data").set("Nullspace", make_near_null_space(near_null_space));
    build(A, params);
  }

  /// \brief The number of rows and columns of A.
  size_t size() const {
    return map_->getLocalNumElements();
  }

  /// \brief Rebuild the hierarchy for A's current values; A has the sparsity pattern the session was made with.
  void update(const Matrix& A) {
    MUNDY_THROW_ASSERT(static_cast<size_t>(A.numRows()) == size() && static_cast<size_t>(A.nnz()) == nnz_,
                       std::invalid_argument, "MueLu preconditioner: A's sparsity pattern changed since construction.");
    // A fresh Tpetra matrix, so no estimate cached on the previous one (such as its largest eigenvalue) is reused.
    A_ = Teuchos::rcp(new crs_type(map_, map_, local_matrix_of(A)));
    MueLu::ReuseTpetraPreconditioner(A_, *op_);
  }

  /// \brief Rebuild the hierarchy for A's current values and a new near-null space.
  template <class NearNullSpace>
  void update(const Matrix& A, const NearNullSpace& near_null_space) {
    op_->GetHierarchy()->GetLevel(0)->Set("Nullspace", Xpetra::toXpetra(make_near_null_space(near_null_space)));
    update(A);
  }

  /// \brief z := one cycle applied to r.
  template <class RVector, class ZVector>
  void apply(const RVector& r, ZVector& z) const {
    MUNDY_THROW_ASSERT(r.extent(0) == size() && z.extent(0) == size(), std::invalid_argument,
                       "MueLu preconditioner: r and z must have one entry per row of A.");
    load_view_into_mv(r, *r_);
    op_->apply(*r_, *z_);
    extract_mv_into_view(*z_, z);
  }

 private:
  void build(const Matrix& A, Teuchos::ParameterList& params) {
    MUNDY_THROW_REQUIRE(A.numRows() == A.numCols(), std::invalid_argument, "MueLu preconditioner: A must be square.");
    nnz_ = static_cast<size_t>(A.nnz());
    A_ = Teuchos::rcp(new crs_type(map_, map_, local_matrix_of(A)));
    op_ = MueLu::CreateTpetraPreconditioner<scalar_t, LO, GO, NO>(
        Teuchos::rcp_implicit_cast<Tpetra::Operator<scalar_t, LO, GO, NO>>(A_), params);
    r_ = Teuchos::rcp(new mv_type(map_, 1));
    z_ = Teuchos::rcp(new mv_type(map_, 1));
  }

  /// \brief A as Tpetra's local matrix: A itself when the types agree, otherwise a copy in Tpetra's types.
  static local_matrix_type local_matrix_of(const Matrix& A) {
    if constexpr (std::is_same_v<Matrix, local_matrix_type>) {
      return A;
    } else {
      using row_map_t = typename local_matrix_type::row_map_type::non_const_type;
      using entries_t = typename local_matrix_type::index_type::non_const_type;
      using values_t = typename local_matrix_type::values_type::non_const_type;
      const auto row_map_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A.graph.row_map);
      const auto entries_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A.graph.entries);
      const auto values_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A.values);
      row_map_t row_map(Kokkos::view_alloc(Kokkos::WithoutInitializing, "row_map"), row_map_in.extent(0));
      entries_t entries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "entries"), entries_in.extent(0));
      values_t values(Kokkos::view_alloc(Kokkos::WithoutInitializing, "values"), values_in.extent(0));
      const auto row_map_host = Kokkos::create_mirror_view(row_map);
      const auto entries_host = Kokkos::create_mirror_view(entries);
      const auto values_host = Kokkos::create_mirror_view(values);
      for (size_t i = 0; i < row_map_in.extent(0); ++i) {
        row_map_host(i) = row_map_in(i);
      }
      for (size_t k = 0; k < entries_in.extent(0); ++k) {
        entries_host(k) = entries_in(k);
        values_host(k) = values_in(k);
      }
      Kokkos::deep_copy(row_map, row_map_host);
      Kokkos::deep_copy(entries, entries_host);
      Kokkos::deep_copy(values, values_host);
      return local_matrix_type("A", A.numRows(), A.numCols(), A.nnz(), values, row_map, entries);
    }
  }

  /// \brief The near-null space, an n x k rank-2 view, as a k-column MultiVector.
  template <class NearNullSpace>
  Teuchos::RCP<mv_type> make_near_null_space(const NearNullSpace& near_null_space) const {
    static_assert(Kokkos::is_view_v<NearNullSpace> && NearNullSpace::rank() == 2,
                  "MueLu preconditioner: the near-null space must be a rank-2 view, one column per vector.");
    MUNDY_THROW_REQUIRE(near_null_space.extent(0) == size(), std::invalid_argument,
                        "MueLu preconditioner: the near-null space must have one row per row of A.");
    const auto in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, near_null_space);
    const auto vectors = Teuchos::rcp(new mv_type(map_, near_null_space.extent(1)));
    {
      auto out = vectors->getLocalViewHost(Tpetra::Access::OverwriteAll);
      for (size_t i = 0; i < in.extent(0); ++i) {
        for (size_t j = 0; j < in.extent(1); ++j) {
          out(i, j) = in(i, j);
        }
      }
    }
    return vectors;
  }

  Teuchos::RCP<const map_type> map_;
  size_t nnz_ = 0;
  Teuchos::RCP<crs_type> A_;
  Teuchos::RCP<op_type> op_;
  Teuchos::RCP<mv_type> r_;
  Teuchos::RCP<mv_type> z_;
};

}  // namespace impl

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_MUELU && HAVE_MUNDYMATH_TPETRA && HAVE_MUNDYMATH_KOKKOSKERNELS

#endif  // MUNDY_MATH_IMPL_MUELU_PRECONDITIONER_IMPL_HPP_
