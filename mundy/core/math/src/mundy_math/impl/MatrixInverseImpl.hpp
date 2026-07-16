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

#ifndef MUNDY_MATH_IMPL_MATRIX_INVERSE_IMPL_HPP_
#define MUNDY_MATH_IMPL_MATRIX_INVERSE_IMPL_HPP_

/// \file MatrixInverseImpl.hpp
/// \brief Implementation of determinant() and inverse() for fixed-size square matrices
///
/// determinant() and inverse() dispatch on N: a cofactor expansion for small N and LU decomposition for larger N.
///
/// # Bitmasked cofactors
/// The cofactor expansion is implemented with a bitmask-based dynamic programming approach that computes all
/// 2^N subdeterminants in a single pass, rather than recomputing a fresh table for each minor.
///
/// Naive cofactor expansion is recursive: det(A) = sum over one row/column of ± a[i][j] * det(minor(i,j)), and each
/// (N-1)×(N-1) minor recurses the same way. The problem is that the same minor is reached by many different recursive
/// paths (deleting row 2 then row 5 reaches the same submatrix as deleting row 5 then row 2), and naive recursion
/// recomputes it every time it's reached.
///
/// The fix is to give every submatrix that can appear during expansion a canonical, memoizable name, so each gets
/// computed exactly once. The trick used here: instead of naming a submatrix by "which rows/columns a particular
/// recursive path happened to delete," name it by which set of columns it uses, paired with a fixed block of rows (the
/// first k rows, or the last k rows). A set of columns over N ≤ 6 columns fits exactly in an N-bit integer — bit c set
/// means "column c is in this submatrix." That integer is the memoizable key. Hence "bitmask."
///
/// # Laplace expansion with bitmasks
/// The table built above is enough to get the determinant, using the whole matrix as one N-row block. But
/// cofactors -- needed for the adjugate, and so for inverse() -- require the determinant of the minor with an
/// *arbitrary* row r deleted, not just the last row. A single table over one fixed row range can't give you that.
///
/// The fix is the generalized Laplace expansion (a.k.a. Cauchy-Binet): split the N-1 surviving rows into "rows
/// above r" (0..r-1) and "rows below r" (r+1..N-1), then for every way of handing r of the N-1 surviving columns to
/// the top block (the rest going to the bottom block), that split contributes one term:
///
///   cofactor(r, c) = (-1)^(r+c) * sum over F, |F| = r, of (sign) * det(top block, cols F)
///                        * det(bottom block, cols complement of F)
///
/// where F ranges over subsets of the N-1 columns other than c. Both of those block determinants are exactly the
/// same shape of table as before -- determinant of a row block over a column subset -- just now needed for a *top*
/// block (rows above r) and a *bottom* block (rows below r), for every possible r. So the code builds two tables,
/// top_det and bottom_det, once each: top_det[Subset] is the determinant of rows 0..popcount(Subset)-1 using columns
/// Subset, exactly as before; bottom_det[Subset] is its mirror image, using the *last* popcount(Subset) rows and
/// expanding along that block's first row instead of its last. The plain determinant is just the degenerate case
/// top_det[all N bits set], where every row lands in the top block. Building both tables costs the same
/// O(N * 2^N) as before; every cofactor is then just a 2^(N-1)-term sum of table lookups -- no new recursion, and
/// no separate table rebuilt per (row, col) minor the way a naive implementation would need.
///
/// # LU decomposition with partial pivoting
/// Bitmask cofactors cost O(N * 2^N): far better than naive O(N!) recursion, but still exponential in N with a steep
/// compile-time cost, so past kMatrixInverseLaplaceToLUCutoff (found empirically) it's cheaper to switch to an O(N^3)
/// method -- LU decomposition with partial pivoting.
///
/// The idea: factor A as PA = LU, where P is a row permutation, L is unit lower triangular (an implicit 1 on the
/// diagonal), and U is upper triangular. Partial pivoting picks, at each elimination step k, the largest-magnitude
/// entry at or below the diagonal in column k and swaps its row into place before eliminating -- this keeps the
/// division by the pivot numerically stable, and every swap performed builds up P and flips the sign of det(A).
/// Eliminating column k introduces zeros below the diagonal that are never read again, so the elimination
/// multiplier that produced each zero is stored right back into that now-unused slot: one N*N buffer ends up
/// holding L below the diagonal and U on/above it, with no separate storage for either.
///
/// determinant() from here is det(A) = sign(P) * det(U), and det(U) is just the product of U's diagonal (L's
/// diagonal is an implicit 1, so it contributes nothing). inverse() solves A X = I for all N columns of the
/// identity at once: PA = LU means LU X = P (the permuted identity), which splits into two triangular solves --
/// forward substitution for L Y = P, then back substitution for U X = Y -- with all N right-hand-side columns
/// solved together rather than one at a time, since they share the same L and U.

// C++ core
#include <cstddef>

// Mundy
#include <mundy_math/NumTraits.hpp>
#include <mundy_math/cmath.hpp>
#include <mundy_math/impl/MatrixImpl.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace impl {

/// \brief Largest N for which determinant()/inverse() use the bitmask cofactor expansion instead of LU.
inline constexpr size_t kMatrixInverseLaplaceToLUCutoff = 6;

//! \name Bit utilities
//@{

/// \brief Number of set bits of x below bit position Bound.
/// \param[in] x Value to count bits in.
template <size_t Bound, size_t... Bits>
KOKKOS_FORCEINLINE_FUNCTION constexpr int popcount_impl(size_t x, std::index_sequence<Bits...>) {
  return (static_cast<int>((x >> Bits) & size_t{1}) + ...);
}

/// \brief Number of set bits of x below bit position Bound.
/// \param[in] x Value to count bits in.
template <size_t Bound>
KOKKOS_FORCEINLINE_FUNCTION constexpr int popcount(size_t x) {
  return popcount_impl<Bound>(x, std::make_index_sequence<Bound>{});
}

/// \brief Sum of the bit positions set in x below bit position Bound (e.g. 0b101 -> 0 + 2).
/// \param[in] x Value to sum bit positions of.
template <size_t Bound, size_t... Bits>
KOKKOS_FORCEINLINE_FUNCTION constexpr int position_sum_impl(size_t x, std::index_sequence<Bits...>) {
  return ((((x >> Bits) & size_t{1}) ? static_cast<int>(Bits) : 0) + ...);
}

/// \brief Sum of the bit positions set in x below bit position Bound (e.g. 0b101 -> 0 + 2).
/// \param[in] x Value to sum bit positions of.
template <size_t Bound>
KOKKOS_FORCEINLINE_FUNCTION constexpr int position_sum(size_t x) {
  return position_sum_impl<Bound>(x, std::make_index_sequence<Bound>{});
}

/// \brief The compressed_idx-th surviving index of a dimension with `excluded` removed, expressed as
/// an index into the original (uncompressed) dimension.
/// \param[in] compressed_idx Index into the reduced dimension (0..N-2).
/// \param[in] excluded The index removed from the original dimension.
KOKKOS_FORCEINLINE_FUNCTION constexpr size_t skip_index(size_t compressed_idx, size_t excluded) {
  return compressed_idx < excluded ? compressed_idx : compressed_idx + 1;
}

//@}

//! \name Bitmask cofactor expansion
/// top_det[Subset]/bottom_det[Subset] hold the determinant of the top/bottom popcount(Subset) rows
/// of the matrix, using columns Subset in relative order. Every cofactor is read off by combining
/// the two tables instead of recomputing a fresh table per (row, col) minor.
//@{

/// \brief One term of the recursion
///
/// top_det[Subset] = sum over c in Subset of (-1)^(row+col) * mat(row, c) * top_det[Subset with c removed],
///    where k = |Subset| and (row, col) are c's position within the block
/// (row is always k-1, its last row; col is c's rank among Subset's members).
///
/// \tparam Subset Column set (as a bitmask) being summed over.
/// \param[in] mat Source matrix.
/// \param[in] top_det Table of already-computed smaller-subset determinants.
template <size_t N, size_t Subset, typename T, ValidAccessor<T> Accessor, size_t... Cols>
KOKKOS_FORCEINLINE_FUNCTION constexpr T top_row_sum(const AMatrix<T, N, N, Accessor>& mat, const T* top_det,
                                                    std::index_sequence<Cols...>) {
  static_assert(sizeof...(Cols) == N, "Number of columns must match N.");
  constexpr int k = popcount<N>(Subset);   // |Subset|: rows 0..k-1 make up this block
  return ((((Subset >> Cols) & size_t{1})  // only sum over c = Cols actually in Subset
               ? (((k - 1 + popcount<N>(Subset & ((size_t{1} << Cols) - 1))) % 2 == 0) ? T(1) : T(-1)) *
                     mat(static_cast<size_t>(k - 1), Cols) * top_det[Subset & ~(size_t{1} << Cols)]
               : T(0)) +
          ...);
}

/// \brief Fills top_det[1..2^N-1] in increasing order (top_det[Subset] only reads smaller entries).
/// \param[in] mat Source matrix.
/// \param[in,out] top_det Table to fill; top_det[0] = 1 must already be set.
template <size_t N, typename T, ValidAccessor<T> Accessor, size_t... SubsetsMinusOne>
KOKKOS_FORCEINLINE_FUNCTION constexpr void fill_top_det(const AMatrix<T, N, N, Accessor>& mat, T* top_det,
                                                        std::index_sequence<SubsetsMinusOne...>) {
  static_assert(sizeof...(SubsetsMinusOne) == (size_t{1} << N) - 1, "Number of subsets must match 2^N - 1.");
  ((top_det[SubsetsMinusOne + 1] = top_row_sum<N, SubsetsMinusOne + 1>(mat, top_det, std::make_index_sequence<N>{})),
   ...);
}

/// \brief One term of the recursion
///
/// bottom_det[Subset] = sum over c in Subset of (-1)^(row+col) * mat(row, c) * bottom_det[Subset with c removed],
///    where k = |Subset| and (row, col) are c's position within the block
/// (row is always N-k, the block's first row; col is c's rank among Subset's members).
///
/// \tparam Subset Column set (as a bitmask) being summed over.
/// \param[in] mat Source matrix.
/// \param[in] bottom_det Table of already-computed smaller-subset determinants.
template <size_t N, size_t Subset, typename T, ValidAccessor<T> Accessor, size_t... Cols>
KOKKOS_FORCEINLINE_FUNCTION constexpr T bottom_row_sum(const AMatrix<T, N, N, Accessor>& mat, const T* bottom_det,
                                                       std::index_sequence<Cols...>) {
  static_assert(sizeof...(Cols) == N, "Number of columns must match N.");
  constexpr int k = popcount<N>(Subset);
  constexpr size_t row = N - static_cast<size_t>(k);  // first row of the last-k-rows block
  return ((((Subset >> Cols) & size_t{1})
               ? ((popcount<N>(Subset & ((size_t{1} << Cols) - 1)) % 2 == 0) ? T(1) : T(-1)) * mat(row, Cols) *
                     bottom_det[Subset & ~(size_t{1} << Cols)]
               : T(0)) +
          ...);
}

/// \brief Fills bottom_det[1..2^N-1] in increasing order (bottom_det[Subset] only reads smaller entries).
/// \param[in] mat Source matrix.
/// \param[in,out] bottom_det Table to fill; bottom_det[0] = 1 must already be set.
template <size_t N, typename T, ValidAccessor<T> Accessor, size_t... SubsetsMinusOne>
KOKKOS_FORCEINLINE_FUNCTION constexpr void fill_bottom_det(const AMatrix<T, N, N, Accessor>& mat, T* bottom_det,
                                                           std::index_sequence<SubsetsMinusOne...>) {
  static_assert(sizeof...(SubsetsMinusOne) == (size_t{1} << N) - 1, "Number of subsets must match 2^N - 1.");
  ((bottom_det[SubsetsMinusOne + 1] =
        bottom_row_sum<N, SubsetsMinusOne + 1>(mat, bottom_det, std::make_index_sequence<N>{})),
   ...);
}

/// \brief Reindexes a bitmask over the N-1 columns surviving ExcludedCol's removal (renumbered 0..N-2)
/// into the matching bitmask over the original N columns.
/// \tparam ExcludedCol Column removed before renumbering.
/// \tparam Local Bitmask over the renumbered column set.
template <size_t N, size_t ExcludedCol, size_t Local, size_t... J>
KOKKOS_FORCEINLINE_FUNCTION constexpr size_t scatter_impl(std::index_sequence<J...>) {
  static_assert(sizeof...(J) == N - 1, "Number of local column indices must match N - 1.");
  return ((((Local >> J) & size_t{1}) ? (size_t{1} << skip_index(J, ExcludedCol)) : size_t{0}) | ...);
}

/// \brief scatter_impl<N, ExcludedCol, Local>, memoized as a constant.
template <size_t N, size_t ExcludedCol, size_t Local>
inline constexpr size_t scattered_v = scatter_impl<N, ExcludedCol, Local>(std::make_index_sequence<N - 1>{});

/// \brief Sign of the (ExcludedCol, Local) cofactor term.
///
/// cofactor(r, c) is the determinant of the (N-1)x(N-1) minor (row r, col c removed); the generalized Laplace/Cauchy-
/// Binet expansion splits it into a top r-row block (columns F) and a bottom (N-1-r)-row block (columns G, the rest):
///   cofactor(r,c) = (-1)^(r+c) * sum over F, |F|=r, of
///                   (-1)^(r(r-1)/2 + sum of F's local column positions) * top_det[F] * bottom_det[G].
/// This is that sign for one term, with Local = F expressed in the reduced universe's local indices
/// and r = popcount(Local).
///
/// \tparam ExcludedCol The excluded column c.
/// \tparam Local The column set F, in the reduced universe's local indices.
template <size_t N, size_t ExcludedCol, size_t Local>
inline constexpr bool cofactor_term_positive_v = [] {
  constexpr int r = popcount<N - 1>(Local);
  constexpr int exponent = r + static_cast<int>(ExcludedCol) + (r * (r - 1)) / 2 + position_sum<N - 1>(Local);
  return exponent % 2 == 0;
}();

/// \brief Computes cofactor(r, ExcludedCol) for every r in one pass
/// each Local is one column split F from the Laplace sum, and since |F| = r, its term lands directly in col_acc[r].
///
/// \param[in] col_acc A view into result's column, forwarded in directly from view_column(); writes
/// still land in result since the view only wraps a reference to its storage.
/// \param[in] top_det Global top-block determinant table.
/// \param[in] bottom_det Global bottom-block determinant table.
template <size_t N, size_t ExcludedCol, typename T, ValidVectorType VectorType, size_t... Local>
KOKKOS_FORCEINLINE_FUNCTION constexpr void accumulate_column(VectorType&& col_acc, const T* top_det,
                                                             const T* bottom_det, std::index_sequence<Local...>) {
  static_assert(sizeof...(Local) == (size_t{1} << (N - 1)), "Number of local column subsets must match 2^(N-1).");
  constexpr size_t full_mask = (size_t{1} << N) - 1;
  // Note, ~scattered_v<N, ExcludedCol, Local> is the complement of the scattered column set, which still has
  // ExcludedCol's bit set, so we clear that via & ~ (1 << ExcludedCol) to get the actual bottom block's column set.
  ((col_acc[popcount<N - 1>(Local)] +=
    (cofactor_term_positive_v<N, ExcludedCol, Local> ? T(1) : T(-1))  //
    * top_det[scattered_v<N, ExcludedCol, Local>]                     //
    * bottom_det[full_mask & ~(size_t{1} << ExcludedCol) & ~scattered_v<N, ExcludedCol, Local>]),
   ...);
}

/// \brief Fills every column of the cofactor matrix from the global top/bottom determinant tables.
/// \param[in,out] result Matrix being assembled; must already be zero-filled.
/// \param[in] top_det Global top-block determinant table.
/// \param[in] bottom_det Global bottom-block determinant table.
template <size_t N, typename T, size_t... ExcludedCol>
KOKKOS_FORCEINLINE_FUNCTION constexpr void combine_all_columns(AMatrix<T, N, N>& result, const T* top_det,
                                                               const T* bottom_det,
                                                               std::index_sequence<ExcludedCol...>) {
  static_assert(sizeof...(ExcludedCol) == N, "Number of excluded columns must match N.");
  (accumulate_column<N, ExcludedCol>(result.template view_column<ExcludedCol>(), top_det, bottom_det,
                                     std::make_index_sequence<(size_t{1} << (N - 1))>{}),
   ...);
}

/// \brief Determinant via the bitmask cofactor expansion.
/// \param[in] mat Source matrix.
template <size_t N, typename T, ValidAccessor<T> Accessor>
KOKKOS_FORCEINLINE_FUNCTION constexpr T bitmask_determinant(const AMatrix<T, N, N, Accessor>& mat) {
  if constexpr (N == 0) {
    return T(1);
  } else {
    constexpr size_t num_subsets = size_t{1} << N;
    T top_det[num_subsets];
    top_det[0] = T(1);  // det of the empty top block (0 rows, 0 columns) is 1
    fill_top_det<N>(mat, top_det, std::make_index_sequence<num_subsets - 1>{});
    return top_det[num_subsets - 1];  // all N bits set = every column = the whole matrix's determinant
  }
}

/// \brief Cofactor matrix via the bitmask cofactor expansion.
/// \param[in] mat Source matrix.
template <size_t N, typename T, ValidAccessor<T> Accessor>
KOKKOS_FORCEINLINE_FUNCTION constexpr AMatrix<T, N, N> bitmask_cofactors(const AMatrix<T, N, N, Accessor>& mat) {
  static_assert(N >= 1, "cofactors is only defined for N >= 1.");
  if constexpr (N == 1) {
    return AMatrix<T, 1, 1>(T(1));
  } else {
    constexpr size_t num_subsets = size_t{1} << N;
    T top_det[num_subsets];
    top_det[0] = T(1);
    fill_top_det<N>(mat, top_det, std::make_index_sequence<num_subsets - 1>{});

    T bottom_det[num_subsets];
    bottom_det[0] = T(1);
    fill_bottom_det<N>(mat, bottom_det, std::make_index_sequence<num_subsets - 1>{});

    AMatrix<T, N, N> result(T(0));  // accumulate_column writes via +=, so result must start zeroed
    combine_all_columns<N>(result, top_det, bottom_det, std::make_index_sequence<N>{});
    return result;
  }
}

//@}

//! \name LU decomposition with partial pivoting
//@{

/// \brief Swaps the values of a and b.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr void swap_val(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

/// \brief Sets perm to the identity permutation.
/// \param[out] perm Permutation array, size N.
template <size_t N, size_t... Idx>
KOKKOS_INLINE_FUNCTION constexpr void init_perm(size_t* perm, std::index_sequence<Idx...>) {
  static_assert(sizeof...(Idx) == N, "Number of indices must match N.");
  ((perm[Idx] = Idx), ...);
}

/// \brief Finds the largest-magnitude entry at or below the diagonal in column K.
/// \param[in] mat Matrix being factorized in place, row-major layout.
/// \param[in,out] pivot_row Best candidate row found so far.
/// \param[in,out] pivot_mag |mat[pivot_row][K]|, the best candidate's magnitude so far.
template <size_t N, size_t K, typename T, size_t... Offset>
KOKKOS_INLINE_FUNCTION constexpr void find_pivot(const AMatrix<T, N, N>& mat, size_t& pivot_row, T& pivot_mag,
                                                 std::index_sequence<Offset...>) {
  static_assert(sizeof...(Offset) == N - K - 1, "Number of candidate rows must match N - K - 1.");
  (
      [&] {
        constexpr size_t candidate_row = K + 1 + Offset;
        const T mag = mundy::abs(mat[candidate_row * N + K]);
        if (mag > pivot_mag) {
          pivot_mag = mag;
          pivot_row = candidate_row;
        }
      }(),
      ...);
}

/// \brief Swaps rows K and pivot_row across all N columns. pivot_row is a runtime fact about the
/// matrix (found by find_pivot), so this is the one place in this file where mat is addressed by a
/// runtime offset rather than a compile-time one.
/// \param[in,out] mat Matrix being factorized in place, row-major layout.
/// \param[in] pivot_row Row to swap with row K.
template <size_t N, size_t K, typename T, size_t... Col>
KOKKOS_INLINE_FUNCTION constexpr void swap_rows(AMatrix<T, N, N>& mat, size_t pivot_row, std::index_sequence<Col...>) {
  static_assert(sizeof...(Col) == N, "Number of columns must match N.");
  (swap_val(mat[K * N + Col], mat[pivot_row * N + Col]), ...);
}

/// \brief Eliminates mat[Row][K], storing its multiplier in the now-unused lower-triangular slot
/// (the compact in-place LU representation) and updating the rest of the row.
/// \param[in,out] mat Matrix being factorized in place, row-major layout.
/// \param[in] pivot_recip 1/mat[K][K].
template <size_t N, size_t K, size_t Row, typename T, size_t... ColOffset>
KOKKOS_INLINE_FUNCTION constexpr void eliminate_row(AMatrix<T, N, N>& mat, T pivot_recip,
                                                    std::index_sequence<ColOffset...>) {
  static_assert(sizeof...(ColOffset) == N - K - 1, "Number of columns to eliminate must match N - K - 1.");
  const T multiplier = mat[Row * N + K] * pivot_recip;  // L[Row][K] = A[Row][K] / A[K][K]
  mat[Row * N + K] = multiplier;
  // A[Row][j] -= L[Row][K] * A[K][j] for every j > K -- the standard elimination update.
  ((mat[Row * N + (K + 1 + ColOffset)] -= multiplier * mat[K * N + (K + 1 + ColOffset)]), ...);
}

/// \brief Eliminates column K below the diagonal for every row K+1..N-1.
/// \param[in,out] mat Matrix being factorized in place, row-major layout.
template <size_t N, size_t K, typename T, size_t... RowOffset>
KOKKOS_INLINE_FUNCTION constexpr void eliminate_step(AMatrix<T, N, N>& mat, std::index_sequence<RowOffset...>) {
  static_assert(sizeof...(RowOffset) == N - K - 1, "Number of rows to eliminate must match N - K - 1.");
  const T pivot_recip = T(1) / mat[K * N + K];  // 1/A[K][K], shared by every row eliminated below
  (eliminate_row<N, K, K + 1 + RowOffset>(mat, pivot_recip, std::make_index_sequence<N - K - 1>{}), ...);
}

/// \brief One elimination step (pivot, swap, eliminate) for column K.
/// \param[in,out] mat Matrix being factorized in place, row-major layout.
/// \param[in,out] perm Row permutation, size N.
/// \return -1 if a row swap occurred, +1 otherwise.
template <size_t N, size_t K, typename T>
KOKKOS_INLINE_FUNCTION constexpr int lu_step(AMatrix<T, N, N>& mat, size_t* perm) {
  size_t pivot_row = K;
  T pivot_mag = mundy::abs(mat[K * N + K]);
  find_pivot<N, K>(mat, pivot_row, pivot_mag, std::make_index_sequence<N - K - 1>{});

  int sign_flip = 1;
  if (pivot_row != K) {
    swap_rows<N, K>(mat, pivot_row, std::make_index_sequence<N>{});
    swap_val(perm[K], perm[pivot_row]);
    sign_flip = -1;
  }

  eliminate_step<N, K>(mat, std::make_index_sequence<N - K - 1>{});
  return sign_flip;
}

/// \brief Factorizes mat into LU form in place and a row permutation. Steps K = 0..N-2 run in order
/// via a comma fold's guaranteed left-to-right sequencing, so step K+1 always sees step K's result.
/// \param[in,out] mat Matrix to factorize in place; L below diagonal, U on/above.
/// \param[out] perm Row permutation, size N.
/// \return The sign contributed by the permutation (+1 or -1).
template <size_t N, typename OutputType, size_t... K>
KOKKOS_INLINE_FUNCTION constexpr int factorize(AMatrix<OutputType, N, N>& mat, size_t* perm,
                                               std::index_sequence<K...>) {
  static_assert(sizeof...(K) == N - 1, "Number of elimination steps must match N - 1.");
  init_perm<N>(perm, std::make_index_sequence<N>{});
  int sign = 1;
  ((sign *= lu_step<N, K>(mat, perm)), ...);  // sign of the permutation = product of each swap's +-1
  return sign;
}

/// \brief det(U): L's diagonal is an implicit 1, so this is the only real factor left to multiply out.
/// \param[in] mat Factorized matrix, row-major layout.
template <size_t N, typename T, size_t... I>
KOKKOS_INLINE_FUNCTION constexpr T diagonal_product(const AMatrix<T, N, N>& mat, std::index_sequence<I...>) {
  static_assert(sizeof...(I) == N, "Number of indices must match N.");
  return (mat[I * N + I] * ...);
}

/// \brief Sets Y to the permuted identity: to solve A x_c = e_c for every column c at once, note PA =
/// LU means LU x_c = P e_c, and (P e_c)[row] = e_c[perm[row]] = 1 iff perm[row]==c. So Y starts as P
/// itself, one column per c.
/// \param[out] Y Right-hand-side matrix.
/// \param[in] perm Row permutation, size N.
template <size_t N, typename T, size_t... Idx>
KOKKOS_INLINE_FUNCTION constexpr void init_permuted_rhs_matrix(AMatrix<T, N, N>& Y, const size_t* perm,
                                                               std::index_sequence<Idx...>) {
  static_assert(sizeof...(Idx) == N * N, "Number of indices must match N*N.");
  ((Y[Idx] = (perm[Idx / N] == Idx % N) ? T(1) : T(0)), ...);
}

/// \brief sum_{j<Row} L[Row][j] * Y[j][Col], the part of L*Y = rhs already known once rows 0..Row-1
/// are solved.
/// \param[in] mat Factorized matrix, row-major layout.
/// \param[in] Y Right-hand-side/solution matrix.
template <size_t N, size_t Row, size_t Col, typename T, size_t... J>
KOKKOS_INLINE_FUNCTION constexpr T forward_sum_one(const AMatrix<T, N, N>& mat, const AMatrix<T, N, N>& Y,
                                                   std::index_sequence<J...>) {
  static_assert(sizeof...(J) == Row, "Number of summed terms must match Row.");
  if constexpr (sizeof...(J) == 0) {
    return T(0);
  } else {
    return ((mat[Row * N + J] * Y[J * N + Col]) + ...);
  }
}

/// \brief Updates all N columns of row Row at once (they're mutually independent for a fixed row).
/// Y[Row][Col] holds rhs[Row][Col] going in; subtracting the known sum leaves L[Row][Row]*x = ...,
/// and L's diagonal is an implicit 1, so this *is* Y[Row][Col] after solving -- no division needed.
/// \param[in] mat Factorized matrix, row-major layout.
/// \param[in,out] Y Right-hand-side/solution matrix.
template <size_t N, size_t Row, typename T, size_t... Col>
KOKKOS_INLINE_FUNCTION constexpr void forward_row_all_cols(const AMatrix<T, N, N>& mat, AMatrix<T, N, N>& Y,
                                                           std::index_sequence<Col...>) {
  static_assert(sizeof...(Col) == N, "Number of columns must match N.");
  ((Y[Row * N + Col] -= forward_sum_one<N, Row, Col>(mat, Y, std::make_index_sequence<Row>{})), ...);
}

/// \brief Solves L*Y = (permuted identity) in place, L's unit diagonal implicit. Rows must resolve
/// in increasing order (row i needs every Y[j][*], j<i, already finalized), guaranteed by the comma
/// fold's left-to-right sequencing over a plain increasing index_sequence.
/// \param[in] mat Factorized matrix, row-major layout.
/// \param[in,out] Y Right-hand-side/solution matrix.
template <size_t N, typename T, size_t... Row>
KOKKOS_INLINE_FUNCTION constexpr void forward_substitute_batch(const AMatrix<T, N, N>& mat, AMatrix<T, N, N>& Y,
                                                               std::index_sequence<Row...>) {
  static_assert(sizeof...(Row) == N, "Number of rows must match N.");
  (forward_row_all_cols<N, Row>(mat, Y, std::make_index_sequence<N>{}), ...);
}

/// \brief sum_{j>Row} U[Row][j] * Y[j][Col], the part of U*X = Y already known once rows Row+1..N-1
/// are solved.
/// \param[in] mat Factorized matrix, row-major layout.
/// \param[in] Y Right-hand-side/solution matrix.
template <size_t N, size_t Row, size_t Col, typename T, size_t... ColOffset>
KOKKOS_INLINE_FUNCTION constexpr T back_sum_one(const AMatrix<T, N, N>& mat, const AMatrix<T, N, N>& Y,
                                                std::index_sequence<ColOffset...>) {
  static_assert(sizeof...(ColOffset) == N - 1 - Row, "Number of summed terms must match N - 1 - Row.");
  if constexpr (sizeof...(ColOffset) == 0) {
    return T(0);
  } else {
    return ((mat[Row * N + (Row + 1 + ColOffset)] * Y[(Row + 1 + ColOffset) * N + Col]) + ...);
  }
}

/// \brief Updates all N columns of row Row at once: X[Row][Col] = (Y[Row][Col] - sum_{j>Row}
/// U[Row][j]*X[j][Col]) / U[Row][Row], written in place over Y.
/// \param[in] mat Factorized matrix, row-major layout.
/// \param[in] inv_diag_row 1/U[Row][Row], precomputed once for the whole matrix.
/// \param[in,out] Y Right-hand-side/solution matrix.
template <size_t N, size_t Row, typename T, size_t... Col>
KOKKOS_INLINE_FUNCTION constexpr void back_row_all_cols(const AMatrix<T, N, N>& mat, T inv_diag_row,
                                                        AMatrix<T, N, N>& Y, std::index_sequence<Col...>) {
  static_assert(sizeof...(Col) == N, "Number of columns must match N.");
  ((Y[Row * N + Col] =
        (Y[Row * N + Col] - back_sum_one<N, Row, Col>(mat, Y, std::make_index_sequence<N - 1 - Row>{})) * inv_diag_row),
   ...);
}

/// \brief Solves U*X = Y in place. Rows must resolve in decreasing order; folding over the usual
/// increasing 0..N-1 and mapping I -> row = N-1-I visits row N-1 first, then N-2, ..., without
/// needing a descending index_sequence.
/// \param[in] mat Factorized matrix, row-major layout.
/// \param[in] inv_diag 1/U[i][i] for every i.
/// \param[in,out] Y Right-hand-side/solution matrix.
template <size_t N, typename T, size_t... I>
KOKKOS_INLINE_FUNCTION constexpr void back_substitute_batch(const AMatrix<T, N, N>& mat, const T* inv_diag,
                                                            AMatrix<T, N, N>& Y, std::index_sequence<I...>) {
  static_assert(sizeof...(I) == N, "Number of rows must match N.");
  (
      [&] {
        constexpr size_t row = N - 1 - I;
        back_row_all_cols<N, row>(mat, inv_diag[row], Y, std::make_index_sequence<N>{});
      }(),
      ...);
}

/// \brief Computes 1/U[i][i] for every i, shared by every column's back-substitution.
/// \param[in] mat Factorized matrix, row-major layout.
/// \param[out] inv_diag Reciprocal diagonal, size N.
template <size_t N, typename T, size_t... I>
KOKKOS_INLINE_FUNCTION constexpr void compute_inv_diag(const AMatrix<T, N, N>& mat, T* inv_diag,
                                                       std::index_sequence<I...>) {
  static_assert(sizeof...(I) == N, "Number of indices must match N.");
  ((inv_diag[I] = T(1) / mat[I * N + I]), ...);
}

/// \brief Determinant via fixed-size LU decomposition with partial pivoting.
/// \param[in] mat Source matrix.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OutputType>
KOKKOS_INLINE_FUNCTION constexpr OutputType lu_determinant(const AMatrix<T, N, N, Accessor>& mat) {
  if constexpr (N == 0) {
    return OutputType(1);
  } else {
    AMatrix<OutputType, N, N> mat_work = mat.template cast<OutputType>();
    size_t perm[N];
    const int sign = factorize<N>(mat_work, perm, std::make_index_sequence<N - 1>{});
    return static_cast<OutputType>(sign) *
           diagonal_product<N>(mat_work, std::make_index_sequence<N>{});  // det(A) = sign * det(U)
  }
}

/// \brief Inverse via fixed-size LU decomposition with partial pivoting: factorize once, then solve
/// A*X = I for all N columns of the identity together.
/// \param[in] mat Source matrix.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OutputType>
KOKKOS_INLINE_FUNCTION constexpr AMatrix<OutputType, N, N> lu_inverse(const AMatrix<T, N, N, Accessor>& mat) {
  AMatrix<OutputType, N, N> mat_work = mat.template cast<OutputType>();
  size_t perm[N];
  const int sign = factorize<N>(mat_work, perm, std::make_index_sequence<N - 1>{});
  const OutputType det = static_cast<OutputType>(sign) *
                         diagonal_product<N>(mat_work, std::make_index_sequence<N>{});  // det(A) = sign * det(U)
  MUNDY_THROW_ASSERT(det != OutputType(0), std::runtime_error, "inverse: matrix is singular.");

  OutputType inv_diag[N];
  compute_inv_diag<N>(mat_work, inv_diag, std::make_index_sequence<N>{});

  AMatrix<OutputType, N, N> Y;
  init_permuted_rhs_matrix<N>(Y, perm, std::make_index_sequence<N * N>{});
  forward_substitute_batch<N>(mat_work, Y, std::make_index_sequence<N>{});
  back_substitute_batch<N>(mat_work, inv_diag, Y, std::make_index_sequence<N>{});
  return Y;
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_MATRIX_INVERSE_IMPL_HPP_
