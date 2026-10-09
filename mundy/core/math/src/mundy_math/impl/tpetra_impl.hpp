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

#ifndef MUNDY_MATH_IMPL_TPETRA_IMPL_HPP_
#define MUNDY_MATH_IMPL_TPETRA_IMPL_HPP_

/// \file tpetra_impl.hpp
/// \brief Single-process Tpetra Maps, and copies between rank-1 views and single-column MultiVectors.

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_TPETRA

#ifdef HAVE_MUNDYMATH_TPETRA

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <cstddef>

// Tpetra / Teuchos:
#include <TpetraCore_config.h>  // for HAVE_TPETRA_MPI
#ifdef HAVE_TPETRA_MPI
#include <Teuchos_DefaultMpiComm.hpp>  // for Teuchos::MpiComm
#else
#include <Teuchos_DefaultSerialComm.hpp>  // for Teuchos::SerialComm
#endif
#include <Teuchos_RCP.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

namespace mundy {

namespace impl {

/// \brief A contiguous Map of \p n rows owned entirely by the calling process.
///
/// Its comm holds only the calling process: MPI_COMM_SELF in MPI builds of Trilinos, whose packages assume an MPI comm
/// there (MPI must be initialized), and a serial comm otherwise.
template <class LO, class GO, class NO>
Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> make_serial_map(size_t n) {
#ifdef HAVE_TPETRA_MPI
  const Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_SELF));
#else
  const Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::SerialComm<int>());
#endif
  const auto n_global = static_cast<Tpetra::global_size_t>(n);
  const GO index_base = 0;
  return Teuchos::rcp(new Tpetra::Map<LO, GO, NO>(n_global, index_base, comm));
}

/// \brief Copy a rank-1 Kokkos View into column 0 of a single-column MultiVector (device-to-device).
template <class MvType, class SrcView>
void load_view_into_mv(const SrcView& src, MvType& mv) {
  auto dev = mv.getLocalViewDevice(Tpetra::Access::OverwriteAll);
  auto col0 = Kokkos::subview(dev, Kokkos::ALL(), 0);
  Kokkos::deep_copy(col0, src);
}

/// \brief Copy column 0 of a single-column MultiVector into a rank-1 Kokkos View (device-to-device).
template <class MvType, class DstView>
void extract_mv_into_view(MvType& mv, DstView& dst) {
  auto dev = mv.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto col0 = Kokkos::subview(dev, Kokkos::ALL(), 0);
  Kokkos::deep_copy(dst, col0);
}

}  // namespace impl

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_TPETRA

#endif  // MUNDY_MATH_IMPL_TPETRA_IMPL_HPP_
