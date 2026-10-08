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

#ifndef MUNDY_MATH_IMPL_MATRIX_MARKET_IMPL_HPP_
#define MUNDY_MATH_IMPL_MATRIX_MARKET_IMPL_HPP_

// C++ core
#include <algorithm>     // for std::min, std::transform
#include <cctype>        // for std::tolower
#include <charconv>      // for std::to_chars, std::from_chars
#include <cstddef>       // for size_t
#include <fstream>       // for std::ifstream, std::ofstream
#include <stdexcept>     // for std::runtime_error
#include <string>        // for std::string, std::getline
#include <string_view>   // for std::string_view
#include <system_error>  // for std::errc
#include <tuple>         // for std::tuple
#include <type_traits>   // for std::is_same_v
#include <vector>        // for std::vector

// Mundy
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE, mundy::sink

namespace mundy {

namespace impl {

//! \name Matrix Market array and coordinate files, shared by every vector and matrix type
//@{

/// \brief The banner of a dense, real, general Matrix Market file.
inline constexpr std::string_view matrix_market_banner = "%%MatrixMarket matrix array real general";

/// \brief The banner of a sparse, real, general Matrix Market file.
inline constexpr std::string_view matrix_market_coordinate_banner = "%%MatrixMarket matrix coordinate real general";

/// \brief Write a rows x cols array to filename as a Matrix Market array file; element(i, j) is entry (i, j).
///
/// Entries go in column-major order, one per line, each as the shortest decimal that reads back to the same Scalar.
template <class Scalar, class Element>
void write_matrix_market(const std::string& filename, size_t rows, size_t cols, const Element& element) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "write_matrix_market: entries must be float or double.");
  std::ofstream file(filename);
  MUNDY_THROW_REQUIRE(file.is_open(), std::runtime_error,
                      mundy::sink() << "write_matrix_market: failed to open " << filename);
  file << matrix_market_banner << '\n' << rows << ' ' << cols << '\n';
  char buffer[64];
  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
      const std::to_chars_result result = std::to_chars(buffer, buffer + sizeof(buffer), Scalar(element(i, j)));
      file.write(buffer, result.ptr - buffer).put('\n');
    }
  }
  file.close();
  MUNDY_THROW_REQUIRE(!file.fail(), std::runtime_error,
                      mundy::sink() << "write_matrix_market: failed while writing " << filename);
}

/// \brief Write a rows x cols matrix of nnz stored entries to filename as a Matrix Market coordinate file; entry(k)
/// is the k-th entry's (row, col, value), zero-based.
///
/// Entries go one per line in the order given, as "row col value" with one-based indices and each value the shortest
/// decimal that reads back to the same Scalar.
template <class Scalar, class Entry>
void write_matrix_market_coordinate(const std::string& filename, size_t rows, size_t cols, size_t nnz,
                                    const Entry& entry) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "write_matrix_market: entries must be float or double.");
  std::ofstream file(filename);
  MUNDY_THROW_REQUIRE(file.is_open(), std::runtime_error,
                      mundy::sink() << "write_matrix_market: failed to open " << filename);
  file << matrix_market_coordinate_banner << '\n' << rows << ' ' << cols << ' ' << nnz << '\n';
  char buffer[64];
  for (size_t k = 0; k < nnz; ++k) {
    const std::tuple<size_t, size_t, Scalar> row_col_value = entry(k);
    file << std::get<0>(row_col_value) + 1 << ' ' << std::get<1>(row_col_value) + 1 << ' ';
    const std::to_chars_result result = std::to_chars(buffer, buffer + sizeof(buffer), std::get<2>(row_col_value));
    file.write(buffer, result.ptr - buffer).put('\n');
  }
  file.close();
  MUNDY_THROW_REQUIRE(!file.fail(), std::runtime_error,
                      mundy::sink() << "write_matrix_market: failed while writing " << filename);
}

/// \brief The whitespace-separated words of text.
inline std::vector<std::string_view> split_words(std::string_view text) {
  std::vector<std::string_view> words;
  size_t end = 0;
  while (true) {
    const size_t begin = text.find_first_not_of(" \t\r\n", end);
    if (begin == std::string_view::npos) {
      return words;
    }
    end = std::min(text.find_first_of(" \t\r\n", begin), text.size());
    words.push_back(text.substr(begin, end - begin));
  }
}

/// \brief Reads a Matrix Market array or coordinate file: the banner and extents on construction, then the entries.
///
/// Accepts what other writers produce: a banner in any case, '%' comment lines, blank lines, the integer field, and
/// entries separated by any whitespace or written with a leading '+'.
class MatrixMarketReader {
 public:
  /// \brief Open filename and read its banner and extents.
  explicit MatrixMarketReader(const std::string& filename) : filename_(filename), file_(filename) {
    MUNDY_THROW_REQUIRE(file_.is_open(), std::runtime_error,
                        mundy::sink() << "read_matrix_market: failed to open " << filename_);
    read_banner();
    read_extents();
  }

  /// \brief The number of rows in the file.
  size_t rows() const {
    return rows_;
  }

  /// \brief The number of columns in the file.
  size_t cols() const {
    return cols_;
  }

  /// \brief The number of stored entries of a coordinate file.
  size_t nnz() const {
    return nnz_;
  }

  /// \brief Throw unless the file holds a rows x cols array.
  void require_extents(size_t rows, size_t cols) const {
    MUNDY_THROW_REQUIRE(rows_ == rows && cols_ == cols, std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << filename_ << " holds a " << rows_ << " x " << cols_
                                      << " array, but a " << rows << " x " << cols << " array is expected.");
  }

  /// \brief Read every entry in column-major order, passing each to store(i, j, value), then require the file to end.
  template <class Scalar, class Store>
  void read(const Store& store) {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                  "read_matrix_market: entries must be float or double.");
    MUNDY_THROW_REQUIRE(!coordinate_, std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << filename_ << " holds a sparse 'coordinate' matrix, "
                                      << "but a dense 'array' is expected.");
    for (size_t j = 0; j < cols_; ++j) {
      for (size_t i = 0; i < rows_; ++i) {
        store(i, j, next_entry<Scalar>(j * rows_ + i, rows_ * cols_));
      }
    }
    MUNDY_THROW_REQUIRE(next_token().empty(), std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": " << filename_
                                      << " holds more than its " << rows_ * cols_ << " entries.");
  }

  /// \brief Read every stored entry of a coordinate file in file order, passing each to store(i, j, value) with
  /// zero-based indices, then require the file to end.
  template <class Scalar, class Store>
  void read_coordinate(const Store& store) {
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                  "read_matrix_market: entries must be float or double.");
    MUNDY_THROW_REQUIRE(coordinate_, std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << filename_ << " holds a dense 'array', but a sparse "
                                      << "'coordinate' matrix is expected.");
    for (size_t k = 0; k < nnz_; ++k) {
      const size_t i = next_index(k, rows_, "row");
      const size_t j = next_index(k, cols_, "column");
      store(i, j, next_entry<Scalar>(k, nnz_));
    }
    MUNDY_THROW_REQUIRE(next_token().empty(), std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": " << filename_
                                      << " holds more than its " << nnz_ << " entries.");
  }

 private:
  /// \brief "filename:line", for messages.
  std::string where() const {
    return filename_ + ":" + std::to_string(line_number_);
  }

  /// \brief Advance to the next line that is neither blank nor a '%' comment; false at the end of the file.
  bool next_content_line() {
    while (std::getline(file_, line_)) {
      ++line_number_;
      position_ = 0;
      const size_t first = line_.find_first_not_of(" \t\r");
      if (first != std::string::npos && line_[first] != '%') {
        return true;
      }
    }
    line_.clear();
    position_ = 0;
    return false;
  }

  /// \brief The next entry token, crossing lines as needed; empty at the end of the file.
  std::string_view next_token() {
    while (true) {
      const size_t begin = line_.find_first_not_of(" \t\r", position_);
      if (begin != std::string::npos) {
        position_ = std::min(line_.find_first_of(" \t\r", begin), line_.size());
        return std::string_view(line_).substr(begin, position_ - begin);
      }
      if (!next_content_line()) {
        return {};
      }
    }
  }

  /// \brief Read and validate the banner, the file's first line.
  void read_banner() {
    std::string banner;
    std::getline(file_, banner);
    line_number_ = 1;
    std::transform(banner.begin(), banner.end(), banner.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const std::vector<std::string_view> words = split_words(banner);
    MUNDY_THROW_REQUIRE(words.size() == 5 && words[0] == "%%matrixmarket" && words[1] == "matrix",
                        std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": " << filename_
                                      << " does not start with a Matrix Market banner such as '"
                                      << std::string(matrix_market_banner) << "'.");
    MUNDY_THROW_REQUIRE(words[2] == "array" || words[2] == "coordinate", std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": " << filename_ << " is in the '"
                                      << std::string(words[2]) << "' format, but only the dense 'array' and sparse "
                                      << "'coordinate' formats are supported.");
    coordinate_ = words[2] == "coordinate";
    MUNDY_THROW_REQUIRE(words[3] == "real" || words[3] == "double" || words[3] == "integer", std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": " << filename_ << " has '"
                                      << std::string(words[3]) << "' entries, but only real and integer entries are "
                                      << "supported.");
    MUNDY_THROW_REQUIRE(words[4] == "general", std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": " << filename_ << " is '"
                                      << std::string(words[4]) << "', but only 'general' (every entry stored) is "
                                      << "supported.");
  }

  /// \brief Read the extents line, the first line after the banner and comments: "rows cols" for an array file and
  /// "rows cols nnz" for a coordinate file.
  void read_extents() {
    const bool found = next_content_line();
    const std::vector<std::string_view> words = split_words(line_);
    MUNDY_THROW_REQUIRE(found && words.size() == (coordinate_ ? 3u : 2u), std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": " << filename_
                                      << " must give its extents as '" << (coordinate_ ? "rows cols nnz" : "rows cols")
                                      << "' after the banner.");
    rows_ = parse_extent(words[0]);
    cols_ = parse_extent(words[1]);
    nnz_ = coordinate_ ? parse_extent(words[2]) : 0;
    position_ = line_.size();
  }

  /// \brief Parse one extent of the extents line.
  size_t parse_extent(std::string_view word) const {
    size_t extent = 0;
    const std::from_chars_result result = std::from_chars(word.data(), word.data() + word.size(), extent);
    MUNDY_THROW_REQUIRE(result.ec == std::errc() && result.ptr == word.data() + word.size(), std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": '" << std::string(word) << "' in "
                                      << filename_ << " is not a valid extent.");
    return extent;
  }

  /// \brief Parse the one-based row or column index of the index-th stored entry, which must lie in [1, extent].
  size_t next_index(size_t index, size_t extent, const char* what) {
    const std::string_view token = next_token();
    MUNDY_THROW_REQUIRE(!token.empty(), std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << filename_ << " ends after " << index << " of its "
                                      << nnz_ << " entries.");
    size_t one_based = 0;
    const std::from_chars_result result = std::from_chars(token.data(), token.data() + token.size(), one_based);
    MUNDY_THROW_REQUIRE(
        result.ec == std::errc() && result.ptr == token.data() + token.size() && one_based >= 1 && one_based <= extent,
        std::runtime_error,
        mundy::sink() << "read_matrix_market: " << where() << ": '" << std::string(token) << "' in " << filename_
                      << " is not a " << what << " index in [1, " << extent << "].");
    return one_based - 1;
  }

  /// \brief Parse the next entry, the index-th of count.
  template <class Scalar>
  Scalar next_entry(size_t index, size_t count) {
    std::string_view token = next_token();
    MUNDY_THROW_REQUIRE(!token.empty(), std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << filename_ << " ends after " << index << " of its "
                                      << count << " entries.");
    if (token.front() == '+') {  // unlike strtod, from_chars rejects a leading '+'
      token.remove_prefix(1);
    }
    Scalar value{};
    const std::from_chars_result result = std::from_chars(token.data(), token.data() + token.size(), value);
    MUNDY_THROW_REQUIRE(result.ec == std::errc() && result.ptr == token.data() + token.size(), std::runtime_error,
                        mundy::sink() << "read_matrix_market: " << where() << ": '" << std::string(token) << "' in "
                                      << filename_ << " is not a representable number.");
    return value;
  }

  std::string filename_;
  std::ifstream file_;
  std::string line_;       //!< The current line
  size_t position_ = 0;    //!< Where the next token starts in line_
  size_t line_number_ = 0;
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t nnz_ = 0;
  bool coordinate_ = false;  //!< Whether the file is in the sparse coordinate format
};
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_MATRIX_MARKET_IMPL_HPP_
