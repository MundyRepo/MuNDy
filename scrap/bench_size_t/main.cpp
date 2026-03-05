/// benchmark_index_width.cpp
///
/// Build example:
///   c++ -O3 -DNDEBUG -march=native -std=c++20 benchmark_index_width.cpp -o bench
///
/// If you use nanobench from a package manager, adjust include path as needed.
/// This benchmark is meant to test where index width matters:
///   1. container metadata size
///   2. loop/index arithmetic
///   3. many-small-object cache behavior
///
/// Important:
/// Changing only operator[](unsigned) vs operator[](unsigned long) in a std::array-like
/// fixed-size container usually shows little or no difference. The meaningful difference
/// comes when the index type is stored in the object (size, capacity, offsets, etc.).

#define ANKERL_NANOBENCH_IMPLEMENT

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <nanobench.h>

template <class IndexT, class T>
class SmallBuffer {
 public:
  using index_type = IndexT;
  using value_type = T;

  SmallBuffer() = default;

  explicit SmallBuffer(IndexT n) : size_(n), capacity_(n), data_(new T[static_cast<std::size_t>(n)]) {}

  SmallBuffer(const SmallBuffer&) = delete;
  SmallBuffer& operator=(const SmallBuffer&) = delete;

  SmallBuffer(SmallBuffer&&) noexcept = default;
  SmallBuffer& operator=(SmallBuffer&&) noexcept = default;

  T& operator[](IndexT i) noexcept { return data_[static_cast<std::size_t>(i)]; }
  const T& operator[](IndexT i) const noexcept { return data_[static_cast<std::size_t>(i)]; }

  IndexT size() const noexcept { return size_; }
  T* data() noexcept { return data_.get(); }
  const T* data() const noexcept { return data_.get(); }

 private:
  IndexT size_{0};
  IndexT capacity_{0};
  std::unique_ptr<T[]> data_{};
};

template <class IndexT>
[[gnu::noinline]] double sum_sequential(const double* x, IndexT n) {
  double s = 0.0;
  for (IndexT i = 0; i < n; ++i) {
    s += x[static_cast<std::size_t>(i)];
  }
  return s;
}

template <class IndexT>
[[gnu::noinline]] double sum_stride(const double* x, IndexT n, IndexT stride) {
  double s = 0.0;
  for (IndexT i = 0; i < n; i += stride) {
    s += x[static_cast<std::size_t>(i)];
  }
  return s;
}

template <class BufferT>
[[gnu::noinline]] double sum_many_small_buffers(const std::vector<BufferT>& bufs) {
  double s = 0.0;
  for (const auto& b : bufs) {
    for (typename BufferT::index_type i = 0; i < b.size(); ++i) {
      s += b[i];
    }
  }
  return s;
}

template <class BufferT>
std::vector<BufferT> make_buffers(std::size_t num_buffers, std::size_t elems_per_buffer) {
  using index_type = typename BufferT::index_type;

  std::vector<BufferT> bufs;
  bufs.reserve(num_buffers);

  for (std::size_t k = 0; k < num_buffers; ++k) {
    bufs.emplace_back(static_cast<index_type>(elems_per_buffer));
  }

  for (std::size_t k = 0; k < num_buffers; ++k) {
    auto& b = bufs[k];
    for (index_type i = 0; i < b.size(); ++i) {
      b[i] = 1.0 + static_cast<double>(k % 13) + 0.001 * static_cast<double>(i);
    }
  }

  return bufs;
}

int main() {
  using U32 = std::uint32_t;
  using U64 = std::uint64_t;

  std::cout << "sizeof(SmallBuffer<uint32_t,double>) = "
            << sizeof(SmallBuffer<U32, double>) << "\n";
  std::cout << "sizeof(SmallBuffer<uint64_t,double>) = "
            << sizeof(SmallBuffer<U64, double>) << "\n";
  std::cout << "sizeof(uint32_t) = " << sizeof(U32) << "\n";
  std::cout << "sizeof(uint64_t) = " << sizeof(U64) << "\n\n";

  // Large flat array benchmark
  constexpr std::size_t N = 50'000'000;
  std::vector<double> x(N);
  std::iota(x.begin(), x.end(), 1.0);

  // Many-small-buffers benchmark
  constexpr std::size_t NUM_BUFFERS = 400'000;
  constexpr std::size_t ELEMS_PER_BUFFER = 8;

  using Buf32 = SmallBuffer<U32, double>;
  using Buf64 = SmallBuffer<U64, double>;

  auto bufs32 = make_buffers<Buf32>(NUM_BUFFERS, ELEMS_PER_BUFFER);
  auto bufs64 = make_buffers<Buf64>(NUM_BUFFERS, ELEMS_PER_BUFFER);

  // Sanity checks so the compiler does not get too clever.
  {
    auto s1 = sum_sequential<U32>(x.data(), static_cast<U32>(x.size()));
    auto s2 = sum_sequential<U64>(x.data(), static_cast<U64>(x.size()));
    auto s3 = sum_many_small_buffers(bufs32);
    auto s4 = sum_many_small_buffers(bufs64);
    std::cout << "sanity sums: " << s1 << ", " << s2 << ", " << s3 << ", " << s4 << "\n\n";
  }

  ankerl::nanobench::Bench bench;
  bench.minEpochIterations(20);
  bench.performanceCounters(true);
  bench.warmup(10);

  volatile double sink = 0.0;

  bench.run("sequential sum u32 index", [&] {
    sink += sum_sequential<U32>(x.data(), static_cast<U32>(x.size()));
  });

  bench.run("sequential sum u64 index", [&] {
    sink += sum_sequential<U64>(x.data(), static_cast<U64>(x.size()));
  });

  bench.run("stride-16 sum u32 index", [&] {
    sink += sum_stride<U32>(x.data(), static_cast<U32>(x.size()), static_cast<U32>(16));
  });

  bench.run("stride-16 sum u64 index", [&] {
    sink += sum_stride<U64>(x.data(), static_cast<U64>(x.size()), static_cast<U64>(16));
  });

  bench.run("many small buffers u32 metadata", [&] {
    sink += sum_many_small_buffers(bufs32);
  });

  bench.run("many small buffers u64 metadata", [&] {
    sink += sum_many_small_buffers(bufs64);
  });

  std::cerr << "ignore sink = " << sink << "\n";
  return 0;
}