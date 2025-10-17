#include <vector>
#include <random>
#include <cstddef>
#include <cstdint>
#include <map>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cassert>

// ----- Accessor (vector-backed or shared scalar) -----
enum class variant_t : unsigned {
  SHARED = 0u,
  VECTOR,
  CONDITIONAL,
  MAPPED_SCALAR,
  MAPPED_VECTOR,
  INVALID
};

// **********************************************************************************************************************
/// \brief Check if a type is in a variadic list of types
template <class T, class... Ts>
struct contains_type {
  static constexpr bool value = (std::is_same_v<T, Ts> || ...);
};
//
template <class T, class... Ts>
static constexpr bool contains_type_v = contains_type<T, Ts...>::value;

// **********************************************************************************************************************
/// \brief Count how many times a type appears in a variadic list of types
template <class T, class... Types>
struct count_type {
  static constexpr size_t value = (0 + ... + (std::is_same_v<T, Types> ? 1 : 0));
};
template <class T, class... Types>
static constexpr size_t count_type_v = count_type<T, Types...>::value;

// **********************************************************************************************************************
/// \brief Find the index in the variadic list of types that matches the given type
template <class T, class... Ts>
struct index_finder;
//
template <class T>
struct index_finder<T> {
  static constexpr size_t value = 0;
};
//
template <class T, class First, class... Rest>
struct index_finder<T, First, Rest...> {
  static_assert(sizeof...(Rest) + 1 > 0, "Type not found in list");
  static constexpr size_t value = std::is_same_v<T, First> ? 0 : 1 + index_finder<T, Rest...>::value;
};
//
template <class T, class... Ts>
  requires(count_type_v<T, Ts...> == 1)
static constexpr size_t index_finder_v = index_finder<T, Ts...>::value;

// **********************************************************************************************************************
/// \brief Get the I'th type in a variadic list of types
template <std::size_t I, typename... Ts>
struct type_at_index;
//
template <std::size_t I, typename Head, typename... Tail>
struct type_at_index<I, Head, Tail...> {
    static_assert(I < 1 + sizeof...(Tail), "Index out of bounds in type_at_index");
    using type = typename type_at_index<I - 1, Tail...>::type;
};
//
// Specialization for the base case (I = 0)
template <typename Head, typename... Tail>
struct type_at_index<0, Head, Tail...> {
    using type = Head;
};
//
template <size_t I, class... Ts>
  requires(I < sizeof...(Ts))
using type_at_index_t = typename type_at_index<I, Ts...>::type;



namespace impl {

// The tuple implementation only comes in play when using capabilities
template <class T, size_t Idx>
struct tuple_member {
  T value;

  using value_type = T;

  // If T is default constructible, provide a default constructor
constexpr tuple_member()
    requires std::default_initializable<T>
  = default;

  // Provide a constructor that takes a single argument.
constexpr tuple_member(T const& val) : value(val) {
  }

  // Provide get() or equivalent
constexpr T& get() {
    return value;
  }

constexpr T const& get() const {
    return value;
  }
};

/// \brief Helper class which will be used via a fold expression to select the member with the correct Idx in a pack of tuple_members
template <size_t SearchIdx, size_t Idx, class T>
struct tuple_idx_matcher {
  using type = tuple_member<T, Idx>;

  template <class Other>
  constexpr auto operator|([[maybe_unused]] Other v) const {
    if constexpr (Idx == SearchIdx) {
      return *this;
    } else {
      return v;
    }
  }
};

/// \brief Helper class which will be used via a fold expression to select the member with the correct type in a pack of tuple_members
template <class SearchType, size_t Idx, class T>
struct tuple_type_matcher {
  using type = tuple_member<T, Idx>;

  template <class Other>
  constexpr auto operator|([[maybe_unused]] Other v) const {
    if constexpr (std::is_same_v<T, SearchType>) {
      return *this;
    } else {
      return v;
    }
  }
};


template <class IdxSeq, class... Elements>
struct tuple_impl;

template <size_t... Idx, class... Elements>
struct tuple_impl<std::index_sequence<Idx...>, Elements...> : public tuple_member<Elements, Idx>... {
  // If all elements are default constructible, provide a default constructor
constexpr tuple_impl()
    requires((std::default_initializable<Elements> && ...))
  = default;

constexpr tuple_impl(Elements... vals)
    requires(sizeof...(Elements) > 0)
      : tuple_member<Elements, Idx>{vals}... {
  }

  /// \brief Default copy/move/assign constructors
constexpr tuple_impl(const tuple_impl&) = default;

constexpr tuple_impl(tuple_impl&&) = default;

constexpr tuple_impl& operator=(const tuple_impl&) = default;

constexpr tuple_impl& operator=(tuple_impl&&) = default;

  /// \brief Get the element of the tuple at index N
  template <size_t N>
  constexpr auto& get() {
    static_assert(N < sizeof...(Elements), "Index out of bounds in tuple::get<N>()");
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() | ...));
    return base_t::type::get();
  }
  template <size_t N>
  constexpr const auto& get() const {
    static_assert(N < sizeof...(Elements), "Index out of bounds in tuple::get<N>()");
    using base_t = decltype((tuple_idx_matcher<N, Idx, Elements>() | ...));
    return base_t::type::get();
  }

  /// \brief Get the element of the tuple with the given type T (errors if T is not unique)
  template <typename T>
  constexpr const auto& get() const {
    static_assert(count_type_v<T, Elements...> == 1, "Type must appear exactly once in tuple to use get<T>()");
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() | ...));
    return base_t::type::get();
  }
  template <typename T>
  constexpr auto& get() {
    static_assert(count_type_v<T, Elements...> == 1, "Type must appear exactly once in tuple to use get<T>()");
    using base_t = decltype((tuple_type_matcher<T, Idx, Elements>() | ...));
    return base_t::type::get();
  }

  // Helper alias: select the matching base; sentinel ensures fold is never empty.
  template <size_t N>
     requires(sizeof...(Elements) > 0)
  using base_of =
      typename decltype((tuple_idx_matcher<N, Idx, Elements>() | ... | tuple_idx_matcher<N, N, void>{}))::type;
};

}  // namespace impl

// A simple tuple-like class for representing slices internally and is
// compatible with device code This doesn't support type access since we don't
// need it This is not meant as an external API
template <class... Elements>
struct tuple : public impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...> {
  // If all elements are default constructible, provide a default constructor
constexpr tuple()
    requires((std::default_initializable<Elements> && ...))
  = default;

constexpr tuple(Elements... vals)
    requires(sizeof...(Elements) > 0)
      : impl::tuple_impl<decltype(std::make_index_sequence<sizeof...(Elements)>()), Elements...>(vals...) {
  }

  /// \brief Default copy/move/assign constructors
constexpr tuple(const tuple&) = default;

constexpr tuple(tuple&&) = default;

constexpr tuple& operator=(const tuple&) = default;

constexpr tuple& operator=(tuple&&) = default;

  /// \brief Get the size of the tuple
static constexpr size_t size() {
    return sizeof...(Elements);
  }

  /// \brief Get the type of the N'th element
  template <size_t N>
   requires(sizeof...(Elements) > 0)
  using element_t = type_at_index_t<N, Elements...>;
};

template <size_t Idx, class... Args>
constexpr auto& get(tuple<Args...>& vals) {
  return vals.template get<Idx>();
}

template <size_t Idx, class... Args>
constexpr const auto& get(const tuple<Args...>& vals) {
  return vals.template get<Idx>();
}

template <class T, class... Args>
constexpr auto& get(tuple<Args...>& vals) {
  return vals.template get<T>();
}

template <class T, class... Args>
constexpr const auto& get(const tuple<Args...>& vals) {
  return vals.template get<T>();
}

// -------- tuple_size
template<class T> struct tuple_size; // primary

template<class... Es>
struct tuple_size<tuple<Es...>> {
  static constexpr std::size_t value = sizeof...(Es);
};

template<class T>
static constexpr std::size_t tuple_size_v = tuple_size<T>::value;

// -------- tuple_element
template<std::size_t I, class T>
struct tuple_element; // primary

template<std::size_t I, class... Es>
struct tuple_element<I, tuple<Es...>> {
  static_assert(I < sizeof...(Es), "tuple_element index out of bounds");
  using type = type_at_index_t<I, Es...>;  // your existing meta util; OK with incomplete types
};

template<std::size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

template <class... Elements>
tuple(Elements...) -> tuple<Elements...>;

// Implementation to concatenate two tuples using index sequences
template <class FirstTuple, class SecondTuple, std::size_t... FirstIndices, std::size_t... SecondIndices>
constexpr auto tuple_cat_impl(const FirstTuple& first, const SecondTuple& second,
                                              std::index_sequence<FirstIndices...>,
                                              std::index_sequence<SecondIndices...>) {
  // Extract elements from both tuples and construct the new tuple
  // This copy the elements of the tuples into the new tuple, so we remove const and ref qualifiers
  return tuple<std::decay_t<decltype(get<FirstIndices>(first))>...,
               std::decay_t<decltype(get<SecondIndices>(second))>...>{get<FirstIndices>(first)...,
                                                                      get<SecondIndices>(second)...};
}

// Public-facing `tuple_cat` function
template <class... FirstElements, class... SecondElements>
constexpr auto tuple_cat(const tuple<FirstElements...>& first, const tuple<SecondElements...>& second) {
  constexpr auto first_size = sizeof...(FirstElements);
  constexpr auto second_size = sizeof...(SecondElements);

  // Generate index sequences for both tuples
  using FirstIndices = std::make_index_sequence<first_size>;
  using SecondIndices = std::make_index_sequence<second_size>;

  // Delegate to the implementation
  return tuple_cat_impl(first, second, FirstIndices{}, SecondIndices{});
}

template <typename... input_t>
using tuple_cat_t = decltype(tuple_cat(std::declval<input_t>()...));

/// Make a tuple from a list of values.
template <class... Elements>
constexpr auto make_tuple(Elements... vals) {
  return tuple<Elements...>{vals...};
}

template <class... Alts>
struct variant {
 private:
  static_assert((std::is_copy_assignable_v<Alts> && ...), "All types must be copy assignable.");
  static_assert((std::is_default_constructible_v<Alts> && ...), "All types must be default constructible.");
  tuple<Alts...> storage_;
  size_t active_index_;

  //! \name Helpers
  //@{

  template <size_t... Ids>
  void reset_active_type_impl(std::index_sequence<Ids...>) {
    ((active_index_ == Ids
          ? (storage_.template get<Ids>() = std::decay_t<decltype(storage_.template get<Ids>())>{}, true)
          : false),
     ...);
  }

  // Function to reset the current active type to its default value
  void reset_active_type() {
    reset_active_type_impl(std::make_index_sequence<sizeof...(Alts)>{});
  }
  //@}

 public:
  /// \brief Default constructor initializes the first type as active
  constexpr variant() : storage_{}, active_index_{0} {
  }

  /// \brief Constructor for initializing with a specific type
  template <class T>
    requires(contains_type_v<T, Alts...>)
  constexpr variant(const T& value) : storage_{}, active_index_{index_of<T>()} {
    storage_.template get<T>() = value;
  }

  /// \brief Get the active type index
  constexpr size_t index() const {
    return active_index_;
  }

  /// \brief Get the number of alternatives
  static constexpr size_t size() {
    return sizeof...(Alts);
  }

  template <class T>
  static constexpr size_t index_of() {
    return index_finder_v<T, Alts...>;
  }

  /// \brief Check if a specific type is active
  template <class T>
  constexpr bool holds_alternative() const {
    return active_index_ == index_of<T>();
  }

  /// \brief The J'th alternative type
  template <size_t J>
  using alternative_t = type_at_index_t<J, Alts...>;

  /// \brief Get the value of the active type
  template <class T>
  constexpr T& get() {
    static_assert(contains_type_v<T, Alts...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }
  template <class T>
  constexpr const T& get() const {
    static_assert(contains_type_v<T, Alts...>, "Type is not in variant.");
    assert(holds_alternative<T>() && "Incorrect type access");
    constexpr size_t index_of_t = index_of<T>();
    return storage_.template get<index_of_t>();
  }

  /// \brief Get the value of the active type based on the active index
  template <size_t ActiveIdx>
  constexpr auto get() -> alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    assert(holds_alternative<Alt>() && "Incorrect type access using active index");
    return storage_.template get<ActiveIdx>();
  }
  template <size_t ActiveIdx>
  constexpr const auto get() const -> const alternative_t<ActiveIdx>& {
    using Alt = alternative_t<ActiveIdx>;
    assert(holds_alternative<Alt>() && "Incorrect type access using active index");
    return storage_.template get<ActiveIdx>();
  }


  /// \brief Set a new active type, default-constructing the previous type
  template <class T>
    requires(contains_type_v<T, Alts...>)
  constexpr void operator=(T const& value) {
    reset_active_type();
    active_index_ = index_of<T>();
    storage_.template get<T>() = value;
  }
};

//! \name Non-member functions
//@{

/// \brief Get the index of the given type
template <class T, class... Alts>
constexpr size_t index_of() {
  return variant<Alts...>::template index_of<T>();
}

/// \brief Check if a specific type is active
template <class T, class... Alts>
constexpr bool holds_alternative(const variant<Alts...>& var) {
  return var.template holds_alternative<T>();
}

/// \brief Get the J'th alternative type TODO(palmerb4): Make independent of concrete variant instance
template <size_t J, class VariantType>
using variant_alternative_t = typename VariantType::template alternative_t<J>;

/// \brief Get the value of the active type
template <class T, class... Alts>
constexpr T& get(variant<Alts...>& var) {
  return var.template get<T>();
}
template <class T, class... Alts>
constexpr const T& get(const variant<Alts...>& var) {
  return var.template get<T>();
}

  /// \brief Get the value of the active type based on the active index
template <size_t ActiveIdx, class... Alts>
constexpr auto& get(variant<Alts...>& var) {
  return var.template get<ActiveIdx>();
}
template <size_t ActiveIdx, class... Alts>
constexpr const auto& get(const variant<Alts...>& var) {
  return var.template get<ActiveIdx>();
}

// -------- variant_size
template<class T> struct variant_size; // primary

template<class... Alts>
struct variant_size<variant<Alts...>> {
  static constexpr std::size_t value = sizeof...(Alts);
};

template<class T>
static constexpr std::size_t variant_size_v = variant_size<T>::value;

//@}
class Accessor {
 public:
  Accessor() = default;

  explicit inline Accessor(double shared_value)
      : our_type_(variant_t::SHARED), variant_(shared_value) {}

  explicit inline Accessor(const std::vector<double>& vec)
      : our_type_(variant_t::VECTOR), variant_(vec) {}

  inline const double& operator()(std::size_t i) const {
    if (variant_.holds_alternative<double>()) {
      return get<double>(variant_);
    } else {
      return get<std::vector<double>>(variant_)[i];
    } 
    // if (our_type_ == variant_t::SHARED) {
    //   return shared_value_;
    // } else if (our_type_ == variant_t::VECTOR) {
    //   return vector_[i];
    // } else if (our_type_ == variant_t::MAPPED_SCALAR) {
    //   return part_mapped_scalars_.at(parts_[i]);
    // } else {
    //   return part_mapped_vectors_.at(parts_[i])[i];
    // }
  }

 private:
  const variant_t our_type_;
  using actual_variant_t = variant<double, std::vector<double>, std::map<int, double>>;
  actual_variant_t variant_;

};



// class Accessor {
//  public:
//   Accessor() = default;

//   explicit inline Accessor(double shared_value)
//       : our_type_(variant_t::SHARED), shared_value_(shared_value), vector_(), parts_(), part_mapped_scalars_(), part_mapped_vectors_(), flip_point_() {}

//   explicit inline Accessor(const std::vector<double>& vec)
//       : our_type_(variant_t::VECTOR), shared_value_(), vector_(vec), parts_(), part_mapped_scalars_(), part_mapped_vectors_(), flip_point_() {}

//   inline Accessor(const std::vector<double>& vec1, const std::vector<double>& vec2, int flip_point)
//       : our_type_(variant_t::CONDITIONAL), shared_value_(), vector_(), parts_(), part_mapped_scalars_(), part_mapped_vectors_(), vec1_(vec1), vec2_(vec2), flip_point_(flip_point) {}

//   inline Accessor(std::vector<int> parts, std::map<int, double> part_mapped_scalars)
//       : our_type_(variant_t::MAPPED_SCALAR), shared_value_(), vector_(), parts_(std::move(parts)), part_mapped_scalars_(std::move(part_mapped_scalars)), part_mapped_vectors_(), flip_point_() {}

//   inline Accessor(std::vector<int> parts, std::map<int,  std::vector<double>> part_mapped_vectors)
//       : our_type_(variant_t::MAPPED_VECTOR), shared_value_(), vector_(), parts_(std::move(parts)), part_mapped_scalars_(), part_mapped_vectors_(std::move(part_mapped_vectors)), flip_point_() {}

//   inline const double& operator()(std::size_t i) const {
//     if (our_type_ == variant_t::SHARED) {
//       return shared_value_;
//     } else {
//       return vector_[i];
//     } 
//     // if (our_type_ == variant_t::SHARED) {
//     //   return shared_value_;
//     // } else if (our_type_ == variant_t::VECTOR) {
//     //   return vector_[i];
//     // } else if (our_type_ == variant_t::MAPPED_SCALAR) {
//     //   return part_mapped_scalars_.at(parts_[i]);
//     // } else {
//     //   return part_mapped_vectors_.at(parts_[i])[i];
//     // }
//   }

//  private:
//   const variant_t our_type_;
//   const double shared_value_;
//   const std::vector<double> vector_;

//   const std::vector<int> parts_;
//   const std::map<int, double> part_mapped_scalars_;
//   const std::map<int,  std::vector<double>> part_mapped_vectors_;

//   const std::vector<double> vec1_;
//   const std::vector<double> vec2_;
//   const int flip_point_;
// };

class ScalarAccessor {
 public:
  inline ScalarAccessor() = default;

  explicit inline ScalarAccessor(double shared_value)
      :shared_value_(shared_value) {}

  inline const double& operator()(std::size_t i) const {
    return shared_value_;
  }

 private:
  const double shared_value_;
};

class VectorAccessor {
 public:
  inline VectorAccessor() = default;

  explicit inline VectorAccessor(const std::vector<double>& vec)
      : vector_(vec) {}

  inline const double& operator()(std::size_t i) const {
    return vector_[i];
  }

 private:
  const std::vector<double> vector_;
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
  
  for (int warm = 0; warm < 100; ++warm) {
    double s = fn();
    std::cout << "warm sum=" << std::setprecision(17) << s << '\n';
  }
  
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
  int iters = 100;                 // timing iterations
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
