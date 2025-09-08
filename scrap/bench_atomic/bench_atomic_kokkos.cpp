// Kokkos flag-setting strategies benchmark
// Build (OpenMP backend example):
//   g++ -O3 -std=c++20 bench_atomic_kokkos.cpp -I/mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0/include -L/mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0/lib -lkokkoscore -fopenmp
// export LD_LIBRARY_PATH=/mnt/sw/nix/store/ajfmwdjwipp5rrpkq8dj4aff23ar4cix-trilinos-14.2.0/lib:$LD_LIBRARY_PATH
//    g++ -O3 -std=c++20 bench_atomic_kokkos.cpp -I/mnt/home/bpalmer/envs/kokkos_arborx_host/include -L/mnt/home/bpalmer/envs/kokkos_arborx_host/lib64 -lkokkoscore -fopenmp -ldl
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

struct Result {
  std::string name;
  bool correct;
  double seconds;
};

KOKKOS_INLINE_FUNCTION
bool wants_to_set(std::uint64_t i, double hit_rate) {
  // Simple deterministic mix to mimic a PRNG without std::
  // Map to [0,1). Keep it device-friendly.
  std::uint64_t x = i * 11400714819323198485ull + 0x9e3779b97f4a7c15ull;
  double u = (double)((x >> 11) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
  return u < hit_rate;
}

// Helper: read a device flag (View<int, default memory space>) back on host
template <class ViewType>
int read_flag(const ViewType& dflag) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dflag);
  return h() != 0;
}

// 1) Unsafe direct write (UB)
template <class ExecSpace>
Result test_unsafe_direct(std::uint64_t N, double hit_rate, int repetitions) {
  using MemSpace = typename ExecSpace::memory_space;
  Kokkos::View<int, MemSpace> flag("flag");
  // NOTE: This kernel intentionally does a non-atomic write to the same location (UB).
  auto run = [&] {
    Kokkos::deep_copy(flag, 0);
    Kokkos::parallel_for("unsafe_direct", Kokkos::RangePolicy<ExecSpace>(0, N),
                         KOKKOS_LAMBDA(const std::uint64_t i) {
                           if (wants_to_set(i, hit_rate)) {
                             // Data race (undefined behaviour):
                             flag() = 1;
                           }
                         });
    ExecSpace().fence();
  };
  // Warm-up + time
  for (int w = 0; w < 1; ++w) run();
  Kokkos::Timer timer;
  for (int r = 0; r < repetitions; ++r) run();
  double secs = timer.seconds() / repetitions;
  bool plausible = read_flag(flag);
  return {"unsafe_direct (UB!)", plausible, secs};
}

// 2) Atomic OR via CAS loop: flag = flag | 1
template <class ExecSpace>
Result test_atomic_or(std::uint64_t N, double hit_rate, int repetitions) {
  using MemSpace = typename ExecSpace::memory_space;
  Kokkos::View<int, MemSpace> flag("flag");
  auto run = [&] {
    Kokkos::deep_copy(flag, 0);
    int* ptr = flag.data();
    Kokkos::parallel_for("atomic_or", Kokkos::RangePolicy<ExecSpace>(0, N),
                         KOKKOS_LAMBDA(const std::uint64_t i) {
                           if (wants_to_set(i, hit_rate)) {
                             // CAS loop to OR with 1 (works even if fetch_or is unavailable).
                             int old = Kokkos::atomic_load(ptr);
                             while ((old & 1) == 0 &&
                                    !Kokkos::atomic_compare_exchange(ptr, old, old | 1)) {
                               // old is updated with current *ptr on failure
                             }
                           }
                         });
    ExecSpace().fence();
  };
  for (int w = 0; w < 1; ++w) run();
  Kokkos::Timer timer;
  for (int r = 0; r < repetitions; ++r) run();
  double secs = timer.seconds() / repetitions;
  bool ok = read_flag(flag) == (hit_rate > 0.0);
  return {"atomic_or (CAS)", ok, secs};
}

// 3) Atomic MAX via CAS loop: flag = max(flag, 1)
template <class ExecSpace>
Result test_atomic_max(std::uint64_t N, double hit_rate, int repetitions) {
  using MemSpace = typename ExecSpace::memory_space;
  Kokkos::View<int, MemSpace> flag("flag");
  auto run = [&] {
    Kokkos::deep_copy(flag, 0);
    int* ptr = flag.data();
    Kokkos::parallel_for("atomic_max", Kokkos::RangePolicy<ExecSpace>(0, N),
                         KOKKOS_LAMBDA(const std::uint64_t i) {
    if (wants_to_set(i, hit_rate)) {
      int old = Kokkos::atomic_load(ptr);                     // load
      while (old < 1 && !Kokkos::Impl::atomic_compare_exchange_strong(     // CAS
               ptr, old, 1, desul::MemoryOrderRelaxed(), desul::MemoryOrderRelaxed())) {
        // on failure, old is updated with *ptr's current value
      }
    }
                         });
    ExecSpace().fence();
  };
  for (int w = 0; w < 1; ++w) run();
  Kokkos::Timer timer;
  for (int r = 0; r < repetitions; ++r) run();
  double secs = timer.seconds() / repetitions;
  bool ok = read_flag(flag) == (hit_rate > 0.0);
  return {"atomic_max (CAS)", ok, secs};
}

// 4) Atomic "store" via atomic_exchange
template <class ExecSpace>
Result test_atomic_store(std::uint64_t N, double hit_rate, int repetitions) {
  using MemSpace = typename ExecSpace::memory_space;
  Kokkos::View<int, MemSpace> flag("flag");
  auto run = [&] {
    Kokkos::deep_copy(flag, false);
    int* ptr = flag.data();
    Kokkos::parallel_for("atomic_store", Kokkos::RangePolicy<ExecSpace>(0, N),
                         KOKKOS_LAMBDA(const std::uint64_t i) {
                           if (wants_to_set(i, hit_rate)) {
                            // Kokkos::atomic_store(flag.data(), 1);
                            Kokkos::atomic_store(ptr, true);
                           }
                         });
    ExecSpace().fence();
  };
  for (int w = 0; w < 1; ++w) run();
  Kokkos::Timer timer;
  for (int r = 0; r < repetitions; ++r) run();
  double secs = timer.seconds() / repetitions;
  bool ok = read_flag(flag) == (hit_rate > 0.0);
  return {"atomic_store (exchange)", ok, secs};
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    using ExecSpace = Kokkos::DefaultExecutionSpace;

    const std::uint64_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 50'000'000ull;
    const double hit_rate  = (argc > 2) ? std::atof(argv[2]) : 0.10;
    const int repetitions  = (argc > 3) ? std::atoi(argv[3]) : 1;

    printf("ExecSpace: %s\n", ExecSpace::name());
    printf("N=%llu, hit_rate=%.6f, repetitions=%d\n",
           (unsigned long long)N, hit_rate, repetitions);

    auto r1 = test_unsafe_direct<ExecSpace>(N, hit_rate, repetitions);
    auto r2 = test_atomic_or<ExecSpace>(N, hit_rate, repetitions);
    auto r3 = test_atomic_max<ExecSpace>(N, hit_rate, repetitions);
    auto r4 = test_atomic_store<ExecSpace>(N, hit_rate, repetitions);

    printf("\nStrategy                      | Correct | Seconds\n");
    printf("--------------------------------+---------+---------\n");
    auto print = [](const Result& r) {
      printf("%-30s |   %s   | %.6f\n", r.name.c_str(), r.correct ? "yes" : "NO ", r.seconds);
    };
    print(r1);
    print(r2);
    print(r3);
    print(r4);
  }
  Kokkos::finalize();
  return 0;
}
