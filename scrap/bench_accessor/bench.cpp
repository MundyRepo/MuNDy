#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>
#include <map>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>

// ----- Accessor (vector-backed or shared scalar) -----
enum class variant_t : unsigned {
  SHARED = 0u,
  VECTOR,
  MAPPED_SCALAR,
  MAPPED_VECTOR,
  INVALID
};

class Accessor {
 public:
  Accessor() = default;

  explicit Accessor(double shared_value)
      : our_type_(variant_t::SHARED), shared_value_(shared_value), vector_(), parts_(), part_mapped_scalars_(), part_mapped_vectors_() {}

  explicit Accessor(const std::vector<double>& vec)
      : our_type_(variant_t::VECTOR), shared_value_(), vector_(vec), parts_(), part_mapped_scalars_(), part_mapped_vectors_() {}

  Accessor(std::vector<int> parts, std::map<int, double> part_mapped_scalars)
      : our_type_(variant_t::MAPPED_SCALAR), shared_value_(), vector_(), parts_(std::move(parts)), part_mapped_scalars_(std::move(part_mapped_scalars)), part_mapped_vectors_() {}

  Accessor(std::vector<int> parts, std::map<int,  std::vector<double>> part_mapped_vectors)
      : our_type_(variant_t::MAPPED_VECTOR), shared_value_(), vector_(), parts_(std::move(parts)), part_mapped_scalars_(), part_mapped_vectors_(std::move(part_mapped_vectors)) {}

  inline const double& operator()(std::size_t i) const {
    if (our_type_ == variant_t::SHARED) {
      return shared_value_;
    } else if (our_type_ == variant_t::VECTOR) {
      return vector_[i];
    } else if (our_type_ == variant_t::MAPPED_SCALAR) {
      return part_mapped_scalars_.at(parts_[i]);
    } else {
      return part_mapped_vectors_.at(parts_[i])[i];
    }
  }

 private:
  variant_t our_type_;
  double shared_value_;
  std::vector<double> vector_;

  std::vector<int> parts_;
  std::map<int, double> part_mapped_scalars_;
  std::map<int,  std::vector<double>> part_mapped_vectors_;
};

class ScalarAccessor {
 public:
  ScalarAccessor() = default;

  explicit ScalarAccessor(double shared_value)
      :shared_value_(shared_value) {}

  inline const double& operator()(std::size_t i) const {
    return shared_value_;
  }

 private:
  double shared_value_;
};

class VectorAccessor {
 public:
  VectorAccessor() = default;

  explicit VectorAccessor(const std::vector<double>& vec)
      : vector_(vec) {}

  inline const double& operator()(std::size_t i) const {
    return vector_[i];
  }

 private:
  std::vector<double> vector_;
};


// ----- Data setup -----
static void randomize(std::vector<double>& v, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto& x : v) x = dist(rng);
}

struct Coeffs6 {
  std::vector<double> a, b, c, d, e, f, x;
  std::size_t N{};
};

static Coeffs6& get_coeffs(std::size_t N) {
  static Coeffs6 C;
  static std::size_t cached = 0;
  if (cached != N) {
    C.a.assign(N, 0.0);
    C.b.assign(N, 0.0);
    C.c.assign(N, 0.0);
    C.d.assign(N, 0.0);
    C.e.assign(N, 0.0);
    C.f.assign(N, 0.0);
    C.x.assign(N, 0.0);
    C.N = N;
    randomize(C.a, 101);
    randomize(C.b, 202);
    randomize(C.c, 303);
    randomize(C.d, 404);
    randomize(C.e, 505);
    randomize(C.f, 606);
    randomize(C.x, 707);
    cached = N;
  }
  return C;
}

// ----- Kernels (unchanged) -----
// Accessor-based Horner (6 coefficients)
template <class AAcc, class BAcc, class CAcc, class DAcc, class EAcc, class FAcc, class XAcc>
static inline double poly6_sum_accessor(const AAcc& a, const BAcc& b, const CAcc& c,
                                        const DAcc& d, const EAcc& e, const FAcc& f,
                                        const XAcc& x, std::size_t N) {
  double s = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double xi = x(i);
    s += (((((a(i) * xi + b(i)) * xi + c(i)) * xi + d(i)) * xi + e(i)) * xi + f(i));
  }
  return s;
}

// Direct vectors
static inline double poly6_sum_direct_vecs(const std::vector<double>& a,
                                           const std::vector<double>& b,
                                           const std::vector<double>& c,
                                           const std::vector<double>& d,
                                           const std::vector<double>& e,
                                           const std::vector<double>& f,
                                           const std::vector<double>& x, std::size_t N) {
  double s = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double xi = x[i];
    s += (((((a[i] * xi + b[i]) * xi + c[i]) * xi + d[i]) * xi + e[i]) * xi + f[i]);
  }
  return s;
}

// Direct scalars
static inline double poly6_sum_direct_scalars(double a, double b, double c,
                                              double d, double e, double f,
                                              double x, std::size_t N) {
  double s = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    const double xi = x;
    s += (((((a * xi + b) * xi + c) * xi + d) * xi + e) * xi + f);
  }
  return s;
}

// ----- Simple timing harness -----
using our_clock_t = std::chrono::steady_clock;

template <class Fn>
static double time_avg_ns(Fn&& fn, int iters) {
  using namespace std::chrono;
  long double total_ns = 0.0L;
  for (int k = 0; k < iters; ++k) {
    auto t0 = our_clock_t::now();
    double s = fn();
    auto t1 = our_clock_t::now();
    // Print AFTER stopping timer to avoid timing I/O but still defeat DCE
    std::cout << "sum=" << std::setprecision(17) << s << '\n';
    total_ns += duration_cast<nanoseconds>(t1 - t0).count();
  }
  return static_cast<double>(total_ns / iters);
}

int main(int argc, char** argv) {
  // Parameters
  std::size_t length = 1'000'000; // elements
  int iters = 10;                 // timing iterations
  if (argc > 1) {
    length = static_cast<std::size_t>(std::stoull(argv[1]));
  }
  if (argc > 2) {
    iters = std::max(1, std::stoi(argv[2]));
  }

  auto& C = get_coeffs(length);

  // 1) 6 Accessors backed by vectors (each Accessor copies its vector by design)
  Accessor av(C.a), bv(C.b), cv(C.c), dv(C.d), ev(C.e), fv(C.f), xv(C.x);
  double avg_ns_acc_vec = time_avg_ns([&] {
    return poly6_sum_accessor(av, bv, cv, dv, ev, fv, xv, C.N);
  }, iters);


  // 1) 6 Accessors backed by vectors (each Accessor copies its vector by design)
  VectorAccessor avx(C.a), bvx(C.b), cvx(C.c), dvx(C.d), evx(C.e), fvx(C.f), xvx(C.x);
  double avg_ns_acc_vec_explicit = time_avg_ns([&] {
    return poly6_sum_accessor(avx, bvx, cvx, dvx, evx, fvx, xvx, C.N);
  }, iters);

  // 2) 6 Accessors backed by shared scalars
  Accessor as(0.11), bs(0.22), cs(0.33), ds(0.44), es(0.55), fs(0.66), xs(0.77);
  double avg_ns_acc_sca = time_avg_ns([&] {
    return poly6_sum_accessor(as, bs, cs, ds, es, fs, xs, C.N);
  }, iters);

  // 2) 6 Accessors backed by shared scalars
  ScalarAccessor asx(0.11), bsx(0.22), csx(0.33), dsx(0.44), esx(0.55), fsx(0.66), xsx(0.77);
  double avg_ns_acc_sca_explicit = time_avg_ns([&] {
    return poly6_sum_accessor(asx, bsx, csx, dsx, esx, fsx, xsx, C.N);
  }, iters);


  // 3) Direct vectors (no accessors)
  double avg_ns_dir_vec = time_avg_ns([&] {
    return poly6_sum_direct_vecs(C.a, C.b, C.c, C.d, C.e, C.f, C.x, C.N);
  }, iters);

  // 4) Direct scalars (no accessors)
  const double a0 = 0.11, b0 = 0.22, c0 = 0.33, d0 = 0.44, e0 = 0.55, f0 = 0.66, x0 = 0.77;
  double avg_ns_dir_sca = time_avg_ns([&] {
    return poly6_sum_direct_scalars(a0, b0, c0, d0, e0, f0, x0, C.N);
  }, iters);

  // Report (averages)
  auto to_ms = [](double ns) { return ns / 1e6; };
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\nAverages over " << iters << " iteration(s) for N=" << length << ":\n";
  std::cout << "  Accessor×6 (vector-backed): " << to_ms(avg_ns_acc_vec) << " ms/iter\n";
  std::cout << "  Accessor×6 (scalar-backed): " << to_ms(avg_ns_acc_sca) << " ms/iter\n";
  std::cout << "  Accessor×6 (vector-backed explicit): " << to_ms(avg_ns_acc_vec_explicit) << " ms/iter\n";
  std::cout << "  Accessor×6 (scalar-backed explicit): " << to_ms(avg_ns_acc_sca_explicit) << " ms/iter\n";
  std::cout << "  Direct vectors            : " << to_ms(avg_ns_dir_vec) << " ms/iter\n";
  std::cout << "  Direct scalars            : " << to_ms(avg_ns_dir_sca) << " ms/iter\n";

  return 0;
}
