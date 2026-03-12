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

#ifndef MUNDY_UTILS_DO_NOT_OPTIMIZE_AWAY_HPP_
#define MUNDY_UTILS_DO_NOT_OPTIMIZE_AWAY_HPP_

/// \file do_not_optimize_away.hpp
/// \brief Compiler logic to prevent a value from being optimized away
///
/// Identical to nanobench's doNotOptimizeAway, which (in turn) is based on the same function in Google Benchmark.

namespace mundy {

namespace impl {

#if defined(_MSC_VER)
// Windows version of doNotOptimizeAway
// see https://github.com/google/benchmark/blob/v1.7.1/include/benchmark/benchmark.h#L514
// see https://github.com/facebook/folly/blob/v2023.01.30.00/folly/lang/Hint-inl.h#L54-L58
// see https://learn.microsoft.com/en-us/cpp/preprocessor/optimize
#    if defined(_MSC_VER)
#        pragma optimize("", off)
void do_not_optimize_away_sink(void const*) {}
#        pragma optimize("", on)
#    endif

template <typename T>
void do_not_optimize_away(T const& val) {
  do_not_optimize_away_sink(&val);
}

#else

// These assembly magic is directly from what Google Benchmark is doing. I have previously used what facebook's folly
// was doing, but this seemed to have compilation problems in some cases. Google Benchmark seemed to be the most well
// tested anyways. see https://github.com/google/benchmark/blob/v1.7.1/include/benchmark/benchmark.h#L443-L446
template <typename T>
void do_not_optimize_away(T const& val) {
  // NOLINTNEXTLINE(hicpp-no-assembler)
  asm volatile("" : : "r,m"(val) : "memory");
}

template <typename T>
void do_not_optimize_away(T& val) {
#if defined(__clang__)
  // NOLINTNEXTLINE(hicpp-no-assembler)
  asm volatile("" : "+r,m"(val) : : "memory");
#else
  // NOLINTNEXTLINE(hicpp-no-assembler)
  asm volatile("" : "+m,r"(val) : : "memory");
#endif
}
#endif

}  // namespace impl

/**
 * @brief Makes sure none of the given arguments are optimized away by the compiler.
 *
 * @tparam Arg Type of the argument that shouldn't be optimized away.
 * @param arg The input that we mark as being used, even though we don't do anything with it.
 */
template <typename Arg>
void do_not_optimize_away(Arg&& arg) {
  impl::do_not_optimize_away(std::forward<Arg>(arg));
}

}  // namespace mundy

#endif  // MUNDY_UTILS_DO_NOT_OPTIMIZE_AWAY_HPP_
