// @HEADER
// **********************************************************************************************************************
//
// This file is part of zorder_knn. (We gleefully stole this code, making minor modifications to improve portability.
// The original code is located here: https://github.com/sebastianlipponer/zorder_knn)
// It is licensed under the following license:
//
// Copyright(c) 2010, 2021 Sebastian Lipponer
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// **********************************************************************************************************************
// @HEADER

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// Mundy libs
#include <mundy_math/Vector3.hpp>  // for mundy::math::Vector3
#include <mundy_math/zmort.hpp>    // for mundy::math::zmorton_less(Vector3, Vector3)

namespace mundy {

namespace math {

namespace {

template <typename Point>
struct BBox {
  using Scalar = typename Point::scalar_t;
  BBox(Scalar bound) {
    p_min.fill(-bound);
    p_max.fill(bound);
  }

  BBox(Point const& p_min, Point const& p_max) : p_min(p_min), p_max(p_max) {
  }

  // Copy constructor
  BBox(BBox const& other) : p_min(other.p_min), p_max(other.p_max) {
  }
  // Copy assignment operator
  BBox& operator=(BBox const& other) {
    if (this != &other) {
      p_min = other.p_min;
      p_max = other.p_max;
    }
    return *this;
  }
  // Move constructor
  BBox(BBox&& other) noexcept : p_min(std::move(other.p_min)), p_max(std::move(other.p_max)) {
  }
  // Move assignment operator
  BBox& operator=(BBox&& other) noexcept {
    if (this != &other) {
      p_min = std::move(other.p_min);
      p_max = std::move(other.p_max);
    }
    return *this;
  }

  Point p_min, p_max;
};

template <typename Point>
typename BBox<Point>::Scalar BoundFromPointsBase2(std::vector<Point> const& points) {
  using Scalar = typename BBox<Point>::Scalar;

  Scalar abs_pj_max{0};
  for (auto const& p : points) {
    for (std::size_t j{0}; j < Point::size; ++j) {
      abs_pj_max = std::max(abs_pj_max, std::abs(p[j]));
    }
  }

  constexpr auto two = static_cast<Scalar>(2.0);
  return std::pow(two, std::ceil(std::log2(abs_pj_max)));
}

template <typename Point>
void SortZOrder(std::vector<Point>& points) {
  // double the obtained bound to prevent SortZOrder() from failing from
  // points located on the boundary
  constexpr std::size_t d = Point::size;
  using Scalar = typename BBox<Point>::Scalar;
  constexpr auto two = static_cast<Scalar>(2.0);
  SortZOrder(points, 0, points.size(), d - 1, BBox<Point>(two * BoundFromPointsBase2<Point>(points)));
}

template <typename Point>
void SortZOrder(std::vector<Point>& points, std::size_t begin, std::size_t end, std::size_t k,
                BBox<Point> const& bbox) {
  assert(end >= begin);
  assert(bbox.p_min[0] <= bbox.p_max[0]);
  assert(bbox.p_min[1] <= bbox.p_max[1]);

  using Scalar = typename BBox<Point>::Scalar;

  // stop unless we are given more than a single point
  if (end - begin <= 1) return;

  // split bounding box in half along k-axis
  BBox<Point> bbox_lower{bbox}, bbox_upper{bbox};

  constexpr auto one_half = static_cast<Scalar>(0.5);
  auto split_k = one_half * (bbox.p_min[k] + bbox.p_max[k]);
  bbox_lower.p_max[k] = split_k;
  bbox_upper.p_min[k] = split_k;

  // sort points into halfspaces
  std::size_t b(begin), e(end);

  for (std::size_t i{b}; i < e; ++i) {
    if (points[i][k] >= bbox_lower.p_min[k] && points[i][k] < bbox_lower.p_max[k]) {
      std::swap(points[b], points[i]);
      ++b;
    }
  }

  for (std::size_t i{e}; i-- > b;) {
    if (points[i][k] > bbox_upper.p_min[k] && points[i][k] <= bbox_upper.p_max[k]) {
      std::swap(points[e - 1], points[i]);
      --e;
    }
  }

  assert(b <= e);

  // if b < e holds there are points on the split plane which are
  // not yet part of any half-space
  constexpr auto zero = static_cast<Scalar>(0.0);
  if (split_k > zero) {
    e = b;
  } else {
    b = e;
  }

  // recurse along k1-axis
  constexpr std::size_t d = Point::size;
  auto k1 = (k + d - 1) % d;

  SortZOrder(points, begin, b, k1, bbox_lower);
  SortZOrder(points, e, end, k1, bbox_upper);
}

template <typename Point>
void GenerateRandomPoints(std::vector<Point>& points) {
  std::random_device rd;
  std::mt19937 e2(rd());
  using Scalar = typename Point::scalar_t;

  auto bound = static_cast<Scalar>(std::pow(2.0, 3));
  std::uniform_real_distribution<Scalar> dist(-bound, bound);

  std::generate(points.begin(), points.end(), [&] {
    Point p;
    for (std::size_t i = 0; i < Point::size; ++i) {
      p[i] = dist(e2);
    }
    return p;
  });
}

template <typename Point>
void TestLessRandom(std::vector<Point> const& points) {
  std::vector<Point> points1(points), points2(points);

  constexpr std::size_t d = Point::size;
  std::sort(points1.begin(), points1.end(), zorder_knn::Less<Point, d>());

  SortZOrder(points2);

  for (std::size_t i{0}; i < points1.size(); ++i) {
    for (std::size_t j{0}; j < Point::size; ++j) {
      EXPECT_EQ(points1[i][j], points2[i][j]);
    }
  }
}

template <std::size_t d>
std::vector<Vector<float, d>> CastDoubleToFloat(std::vector<Vector<double, d>> const& points_double) {
  std::vector<Vector<float, d>> points_float(points_double.size());
  for (std::size_t i{0}; i < points_double.size(); ++i) {
    for (std::size_t j{0}; j < d; ++j) {
      points_float[i][j] = static_cast<float>(points_double[i][j]);
    }
  }

  return points_float;
}

template <std::size_t n, std::size_t d>
void TestLessRandom() {
  std::vector<Vector<double, d>> points(n);
  GenerateRandomPoints(points);

  TestLessRandom(points);
  TestLessRandom(CastDoubleToFloat(points));
}

TEST(Less, Random2D_10k) {
  TestLessRandom<10000, 2>();
}
TEST(Less, Random3D_10k) {
  TestLessRandom<10000, 3>();
}
TEST(Less, Random4D_10k) {
  TestLessRandom<10000, 4>();
}
TEST(Less, Random5D_10k) {
  TestLessRandom<10000, 5>();
}
TEST(Less, Random6D_10k) {
  TestLessRandom<10000, 6>();
}
TEST(Less, Random42D_10k) {
  TestLessRandom<10000, 42>();
}

TEST(ZMortonFloatToUint, Zero) {
  EXPECT_EQ(zorder_knn::detail::FloatToUInt(0.0f), 0x0u);
  EXPECT_EQ(zorder_knn::detail::FloatToUInt(-0.0f), 0x80000000u);

  EXPECT_EQ(zorder_knn::detail::FloatToUInt(0.0), 0x0ll);
  EXPECT_EQ(zorder_knn::detail::FloatToUInt(-0.0), 0x8000000000000000ll);
}

TEST(ZMortonFloatToUInt, Infinity) {
  constexpr auto inff = std::numeric_limits<float>::infinity();
  EXPECT_EQ(zorder_knn::detail::FloatToUInt(inff), 0x7f800000u);
  EXPECT_EQ(zorder_knn::detail::FloatToUInt(-inff), 0xff800000u);

  constexpr auto infd = std::numeric_limits<double>::infinity();
  EXPECT_EQ(zorder_knn::detail::FloatToUInt(infd), 0x7ff0000000000000ll);
  EXPECT_EQ(zorder_knn::detail::FloatToUInt(-infd), 0xfff0000000000000ll);
}

TEST(ZMortonFloatToUInt, NaN) {
  auto nanf_uint = zorder_knn::detail::FloatToUInt(std::numeric_limits<float>::signaling_NaN());
  EXPECT_GT(nanf_uint & 0x7fffffu, 0x0u);
  EXPECT_EQ(nanf_uint & 0x7f800000u, 0x7f800000u);

  auto nand_uint = zorder_knn::detail::FloatToUInt(std::numeric_limits<double>::signaling_NaN());
  EXPECT_GT(nand_uint & 0xfffffffffffffll, 0x0u);
  EXPECT_EQ(nand_uint & 0x7ff0000000000000ll, 0x7ff0000000000000llu);
}

TEST(ZMortonFloatExp, PowerOfTwo) {
  for (int exp(-126); exp < 128; ++exp) {
    auto f_uint = zorder_knn::detail::FloatToUInt(static_cast<float>(std::pow(2.0f, exp)));
    EXPECT_EQ(zorder_knn::detail::FloatExp(f_uint), exp);
  }

  for (int exp(-1022); exp < 1024; ++exp) {
    auto d_uint = zorder_knn::detail::FloatToUInt(std::pow(2.0, exp));
    EXPECT_EQ(zorder_knn::detail::FloatExp(d_uint), exp);
  }
}

TEST(ZMortonFloatSig, OneOverPowerOfTwo) {
  for (int i(23); i > 0; --i) {
    auto f_uint = zorder_knn::detail::FloatToUInt(1.0f + static_cast<float>(std::pow(2.0f, -i)));
    EXPECT_EQ(zorder_knn::detail::FloatSig(f_uint), 0x1u << (23 - i));
  }

  for (int i(52); i > 0; --i) {
    auto d_uint = zorder_knn::detail::FloatToUInt(1.0 + std::pow(2.0, -i));
    EXPECT_EQ(zorder_knn::detail::FloatSig(d_uint), 0x1ll << (52 - i));
  }
}

TEST(ZMortonUIntLogBase2, PowerOfTwo) {
  for (unsigned int i{0}; i < 64; ++i) {
    if (i < 32) {
      uint32_t x = 0x1u << i;
      EXPECT_EQ(zorder_knn::detail::UIntLogBase2(x), i);
      EXPECT_EQ(zorder_knn::detail::UIntLogBase2(2 * x - 1), i);
    }

    uint64_t x = 0x1ll << i;
    EXPECT_EQ(zorder_knn::detail::UIntLogBase2(x), i);
    EXPECT_EQ(zorder_knn::detail::UIntLogBase2(2 * x - 1), i);
  }
}

struct XorMsbTest {
  double p, q;
  int xor_msb;
};

void test_xor_msb(std::vector<XorMsbTest> const& tests, bool only_double = false) {
  for (auto const& t : tests) {
    EXPECT_EQ(zorder_knn::detail::FloatXorMsb(t.p, t.q), t.xor_msb);

    if (!only_double) {
      EXPECT_EQ(zorder_knn::detail::FloatXorMsb(static_cast<float>(t.p), static_cast<float>(t.q)), t.xor_msb);
    }
  }
}

TEST(ZMortonFloatXorMsb, Equal) {
  // clang-format off
  test_xor_msb({
      { 1.0,           1.0,           std::numeric_limits<int>::min() },
      { 42.6666641235, 42.6666641235, std::numeric_limits<int>::min() }
  });
  // clang-format on
}

TEST(ZMortonFloatXorMsb, DifferingSign) {
  // clang-format off
  test_xor_msb({
      { 1.0,           -1.0,           std::numeric_limits<int>::min() },
      { 42.6666641235, -42.6666641235, std::numeric_limits<int>::min() }
  });
  // clang-format on
}

TEST(ZMortonFloatXorMsb, DifferingExponent) {
  // clang-format off
  test_xor_msb({
      { 0.5,  1.0,    0 },
      { 0.5,  1.5,    0 },
      { 0.5,  0.125, -1 },
      { 0.25, 0.125, -2 },
      { 1.0,  2.0,    1 },
      { 2.0,  4.0,    2 }
  });
  // clang-format on
}

TEST(ZMortonFloatXorMsb, DifferingSignificand) {
  // clang-format off
  test_xor_msb({
      {  1.0,     1.5,              -1 },
      {  1.0,     1.75,             -1 },
      {  1.0,     1.875,            -1 },
      {  1.0,     1.375,            -2 },
      {  1.0,     1.125,            -3 },
      {  1.75,    1.875,            -3 },
      {  0.5,     0.5625,           -4 },
      {  0.21875, 0.234375,         -6 },
      {  16.0,    18.0,              1 },
      {  24.0,    26.0,              1 },
      {  28.0,    30.0,              1 },
      {  56.0,    60.0,              2 },
      {  112.0,   120.0,             3 },
      {  80.0,    88.0,              3 },
      {  160.0,   176.0,             4 },
      {  384.0,   448.0,             6 },
      {  1.0,     1.0 + 1.1921e-7, -23 },
      {  1.0,     1.0 + 3.5763e-7, -22 },
      {  1.0,     1.0 + 2.3842e-7, -22 },
      {  1.0,     1.0 + 4.7684e-7, -21 }
  });
  // clang-format on

  // clang-format off
  test_xor_msb({
      { 1.0, 1.0 + 2.2204460e-16, -52 },
      { 1.0, 1.0 + 4.4408921e-16, -51 }
  }, true);
  // clang-format on
}

}  // namespace

}  // namespace math

}  // namespace mundy
