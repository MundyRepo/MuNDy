// g++ -O3 -std=gnu++23 -fopenmp ./bench_atomic.cpp
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE static
#endif

/// \brief Return true if the current iteration decides to set the flag.
/// \details Deterministic pseudo-condition so runs are reproducible.
///          About `hit_rate` fraction of iterations will set the flag.
inline bool wants_to_set(std::uint64_t i, double hit_rate) {
  // A simple LCG-ish mix so different i spread out.
  std::uint64_t x = i * 11400714819323198485ull + 0x9e3779b97f4a7c15ull;
  // Map to [0,1)
  double u = (x >> 11) * (1.0 / (1ull << 53));
  return u < hit_rate;
}

/// \brief Time a callable and return elapsed seconds.
template <class F>
double time_it(F&& f, int warmups = 1, int iters = 1) {
  // Warm-up
  for (int w = 0; w < warmups; ++w) f();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < iters; ++r) f();
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dt = t1 - t0;
  return dt.count() / iters;
}

struct Result {
  std::string name;
  bool correct;
  double seconds;
};

int main(int argc, char** argv) {
  const std::uint64_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 50'000'000ull;
  const double hit_rate  = (argc > 2) ? std::atof(argv[2]) : 0.10;
  const int repetitions  = (argc > 3) ? std::atoi(argv[3]) : 1;

  std::cout << "N=" << N << ", hit_rate=" << hit_rate
            << ", repetitions=" << repetitions << "\n";

  std::vector<Result> results;

  // ------------------------------------------------------------
  // 1) Unsafe direct write (UB): non-atomic shared bool
  // ------------------------------------------------------------
  {
    auto run = [&] {
      volatile bool dirty_plain = false; // volatile does NOT make this thread-safe; still UB.
#pragma omp parallel for schedule(OMP_SCHEDULE)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        if (wants_to_set(i, hit_rate)) {
          // DATA RACE: multiple threads may write concurrently to the same object.
          // Included only as a baseline to show "fast but illegal".
          dirty_plain = true;
        }
      }
      // Read once to keep side effect "observable" to optimizer
      return dirty_plain ? 1 : 0;
    };
    // Time it
    double secs = time_it([&]{ (void)run(); }, 1, repetitions);
    // "Correctness" here is meaningless because behavior is undefined; we still check plausibility.
    bool plausibly_set = (run() != 0);
    results.push_back({"unsafe_direct (UB!)", plausibly_set, secs});
  }

  // ------------------------------------------------------------
  // 2) Atomic OR using std::atomic<int>::fetch_or
  // ------------------------------------------------------------
  {
    auto run = [&] {
      std::atomic<int> flag{0};
#pragma omp parallel for schedule(OMP_SCHEDULE)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        if (wants_to_set(i, hit_rate)) {
          // Relaxed ordering is sufficient if we only care about the flag value itself.
          flag.fetch_or(1, std::memory_order_relaxed);
        }
      }
      return flag.load(std::memory_order_relaxed);
    };
    double secs = time_it([&]{ (void)run(); }, 1, repetitions);
    bool ok = (run() != 0) == (hit_rate > 0.0);
    results.push_back({"atomic_fetch_or", ok, secs});
  }

  // ------------------------------------------------------------
  // 3) Atomic MAX via CAS loop (portable "max(flag, 1)" semantics)
  //    This emulates fetch_max(1) which is standardized in C++23,
  //    but we implement it explicitly for clarity and portability.
  // ------------------------------------------------------------
  {
    auto run = [&] {
      std::atomic<int> flag{0};
#pragma omp parallel for schedule(OMP_SCHEDULE)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        if (wants_to_set(i, hit_rate)) {
          // flag.store(true, std::memory_order_relaxed); 
          int old = flag.load(std::memory_order_relaxed);
          while (old < 1 && !flag.compare_exchange_weak(
                               old, 1,
                               std::memory_order_relaxed,
                               std::memory_order_relaxed)) {
            // old reloaded on failure
          }
        }
      }
      return flag.load(std::memory_order_relaxed);
    };
    double secs = time_it([&]{ (void)run(); }, 1, repetitions);
    bool ok = (run() != 0) == (hit_rate > 0.0);
    results.push_back({"atomic_max_via_CAS", ok, secs});
  }

  // ------------------------------------------------------------
  // 4) Atomic store to std::atomic<bool>
  //    This is the simplest & often best for a one-way "publish dirty" flag.
  // ------------------------------------------------------------
  {
    auto run = [&] {
      std::atomic<bool> dirty{false};
#pragma omp parallel for schedule(OMP_SCHEDULE)
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        if (wants_to_set(i, hit_rate)) {
          dirty.store(true, std::memory_order_relaxed);
        }
      }
      return dirty.load(std::memory_order_relaxed);
    };
    double secs = time_it([&]{ (void)run(); }, 1, repetitions);
    bool ok = run() == (hit_rate > 0.0);
    results.push_back({"atomic_store_bool", ok, secs});
  }

  // ------------------------------------------------------------
  // Print results
  // ------------------------------------------------------------
  std::cout << "\nStrategy                        | Correct | Seconds\n";
  std::cout << "--------------------------------+---------+---------\n";
  for (const auto& r : results) {
    std::cout << r.name
              << std::string(std::max(1, 30 - (int)r.name.size()), ' ')
              << "|   " << (r.correct ? "yes" : "NO ")
              << "   | " << r.seconds << "\n";
  }
}
