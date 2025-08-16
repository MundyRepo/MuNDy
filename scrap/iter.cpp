
/// \brief Generate a lazy integer sequence [start, stop) with a given step size.
///
/// \param start First value in the sequence.
/// \param stop Sequence stops before reaching this value.
/// \param step Increment per step (must be non-zero).
///
/// \return A range view producing integers from start, incremented by step, until stop (exclusive).
auto arange(int start, int stop, int step = 1) {
  std::size_t count = 0;
  if ((step > 0 && start < stop) || (step < 0 && start > stop)) {
    count = static_cast<std::size_t>(std::ceil((stop - start) / static_cast<double>(step)));
  }
  return std::views::iota(std::size_t{0}, count) |
         std::views::transform([=](std::size_t i) { return start + static_cast<int>(i) * step; });
}

/// \brief Generate a lazy floating-point sequence [start, stop) with a given step size.
///
/// \param start First value in the sequence.
/// \param stop Sequence stops before reaching this value.
/// \param step Increment per step (must be non-zero).
///
/// \return A range view producing values from start, incremented by step, until stop (exclusive).
template <std::floating_point T>
auto arange(T start, T stop, T step) {
  std::size_t count = 0;
  if ((step > 0 && start < stop) || (step < 0 && start > stop)) {
    count = static_cast<std::size_t>(std::ceil((stop - start) / step));
  }
  return std::views::iota(std::size_t{0}, count) |
         std::views::transform([=](std::size_t i) { return start + static_cast<T>(i) * step; });
}

/// \brief View representing a geometric progression for floating-point types.
/// Produces start, start*ratio, ..., stopping before exceeding stop.
template <std::floating_point T>
class geom_range_fp : public std::ranges::view_base {
  T start_, stop_, ratio_;
  bool increasing_;

  static bool approx_lesser(T a, T b) {
    const T eps = std::numeric_limits<T>::epsilon() * 8;
    return a < b + eps;
  }
  static bool approx_greater(T a, T b) {
    const T eps = std::numeric_limits<T>::epsilon() * 8;
    return a > b - eps;
  }

 public:
  /// \brief Construct a floating-point geometric range.
  ///
  /// \param start First term.
  /// \param stop Limit (sequence stops before surpassing it).
  /// \param ratio Common ratio (must be >0 and !=1).
  geom_range_fp(T start, T stop, T ratio)
      : start_(start), stop_(stop), ratio_(ratio), increasing_(ratio > T(1)) {}

  struct iterator {
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using iterator_concept = std::input_iterator_tag;

    T cur, stop, ratio;
    bool increasing;

    value_type operator*() const { return cur; }

    iterator& operator++() {
      cur *= ratio;
      return *this;
    }
    void operator++(int) { ++(*this); }
  };

  struct sentinel {};

  iterator begin() const {
    if (increasing_) {
      if (!approx_lesser(start_, stop_)) return {stop_, stop_, ratio_, true};
    } else {
      if (!approx_greater(start_, stop_)) return {stop_, stop_, ratio_, false};
    }
    return {start_, stop_, ratio_, increasing_};
  }

  sentinel end() const { return {}; }

  friend bool operator==(const iterator& it, const sentinel&) {
    const auto eps = std::numeric_limits<decltype(it.cur)>::epsilon() * 8;
    if (it.increasing)
      return !approx_lesser(it.cur, it.stop);
    else
      return !approx_greater(it.cur, it.stop);
  }
};

/// \brief View representing a geometric progression for integer types.
/// Produces start, start*ratio, ..., stopping before exceeding stop.
template <std::integral I>
class geom_range_int : public std::ranges::view_base {
  I start_, stop_, ratio_;

 public:

  /// \brief Construct an integer geometric range.
  ///
  /// \param start First term (must be >0).
  /// \param stop Limit (must be >0, sequence stops before surpassing it).
  /// \param ratio Common ratio (must be >1).
  geom_range_int(I start, I stop, I ratio)
      : start_(start), stop_(stop), ratio_(ratio) {}

  struct iterator {
    using value_type = I;
    using difference_type = std::ptrdiff_t;
    using iterator_concept = std::input_iterator_tag;

    I cur, stop, ratio;

    static bool mul_would_overflow(I a, I b) {
      return a > std::numeric_limits<I>::max() / b;
    }

    value_type operator*() const { return cur; }

    iterator& operator++() {
      if (!mul_would_overflow(cur, ratio))
        cur = static_cast<I>(cur * ratio);
      else
        cur = static_cast<I>(std::numeric_limits<I>::max());
      return *this;
    }
    void operator++(int) { ++(*this); }
  };

  struct sentinel {};

  iterator begin() const {
    if (start_ > stop_) return {stop_, stop_, ratio_};
    return {start_, stop_, ratio_};
  }
  sentinel end() const { return {}; }

  friend bool operator==(const iterator& it, const sentinel&) {
    return it.cur >= it.stop;
  }
};

/// \brief Create a floating-point geometric progression view.
///
/// \param start First term.
/// \param stop Limit (exclusive).
/// \param ratio Common ratio.
///
/// \return A view producing the sequence.
template <std::floating_point T>
auto geom_seq(T start, T stop, T ratio) {
  return geom_range_fp<T>(start, stop, ratio);
}

/// \brief Create an integer geometric progression view.
///
/// \param start First term.
/// \param stop Limit (exclusive).
/// \param ratio Common ratio (>1).
///
/// \return A view producing the sequence.
template <std::integral I>
auto geom_seq(I start, I stop, I ratio) {
  return geom_range_int<I>(start, stop, ratio);
}