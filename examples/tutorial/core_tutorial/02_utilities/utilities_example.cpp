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

#include <Kokkos_Core.hpp>  // for Kokkos::ScopeGuard
#include <iostream>
#include <mundy_utils/StringLiteral.hpp>  // for mundy::StringLiteral, mundy::make_string_literal
#include <mundy_utils/StringSink.hpp>     // for mundy::sink
#include <mundy_utils/rng.hpp>            // for mundy::make_philox
#include <mundy_utils/throw_assert.hpp>   // for MUNDY_THROW_REQUIRE, MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>          // for mundy::tuple, mundy::make_tuple, mundy::get, mundy::tuple_cat
#include <mundy_utils/variant.hpp>        // for mundy::variant, mundy::holds_alternative, mundy::visit
#include <stdexcept>                      // for std::out_of_range, std::logic_error

//---------------------------------------------------------------------------------------------------------------------//
// Helper used by the StringLiteral example.
//---------------------------------------------------------------------------------------------------------------------//

/*
  A common pattern is to use a StringLiteral as a non-type template parameter
  so that distinct field names produce distinct types at compile time.  This
  lets Mundy identify named particle fields without runtime string lookups.

  The template must be defined before any call site.
*/
template <mundy::StringLiteral Name>
void printFieldType() {
  std::cout << "  field type with name: " << Name.data() << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Assertions example.
//---------------------------------------------------------------------------------------------------------------------//
void assertionsExample() {
  std::cout << "\n--- Assertions ---\n" << std::endl;

  /*
    Mundy provides two assertion macros.  Both share the same signature:

      MUNDY_THROW_REQUIRE(assertion, ExceptionType, message);
      MUNDY_THROW_ASSERT(assertion, ExceptionType, message);

    The difference is when they fire:

      MUNDY_THROW_REQUIRE  -- always active, even in release builds.
                              Use it for preconditions that must always hold:
                              bounds checks, non-null pointers, valid
                              configuration values.

      MUNDY_THROW_ASSERT   -- active only in debug builds (NDEBUG not set).
                              Use it for internal invariants that you trust
                              in production but want to verify during
                              development.

    On the host a failing assertion throws the specified exception type.
    On the device (inside a Kokkos kernel) it calls Kokkos::abort -- there
    is no way to throw across a GPU kernel boundary.

    The message argument may be:
      - a plain string literal
      - a mundy::StringLiteral (compile-time)
      - a mundy::sink() pipeline (shown below)
  */

  int n = 10;
  int i = 3;

  // Passing assertion -- nothing happens.
  MUNDY_THROW_REQUIRE(i < n, std::out_of_range, "i is out of range");
  std::cout << "MUNDY_THROW_REQUIRE passed  (i=" << i << " < n=" << n << ")" << std::endl;

  // MUNDY_THROW_ASSERT compiles away in release builds, so it costs nothing
  // in production.  It is safe to leave in hot paths.
  MUNDY_THROW_ASSERT(n > 0, std::logic_error, "n must be positive");
  std::cout << "MUNDY_THROW_ASSERT passed   (active in debug builds only)" << std::endl;

  // A failing assertion throws.  Catch it here to demonstrate the message.
  try {
    MUNDY_THROW_REQUIRE(i > n, std::out_of_range, "deliberately failing assertion");
  } catch (const std::out_of_range& e) {
    std::cout << "Caught expected exception: " << e.what() << std::endl;
  }
}

//---------------------------------------------------------------------------------------------------------------------//
// StringSink example.
//---------------------------------------------------------------------------------------------------------------------//
void stringSinkExample() {
  std::cout << "\n--- StringSink ---\n" << std::endl;

  /*
    The sink() helper builds assertion messages lazily using a familiar <<
    syntax.  Mundy automatically picks one of two sink types based on what
    you stream into it:

      StringLiteralSink  -- every streamed value is a compile-time string
                            literal.  The resulting object is constexpr and
                            device-printable.  Use this in hot device-side
                            assertions where runtime allocation is forbidden.

      StringSink         -- at least one streamed value is a runtime quantity
                            (int, double, pointer, ...).  The message is
                            assembled lazily and returned as std::string via
                            to_string().

    You never choose between them explicitly; the type is inferred from what
    you stream.
  */

  // All operands are string literals -> StringLiteralSink, constexpr.
  constexpr auto compile_time_msg = mundy::sink() << "overflow " << "detected";
  static_assert(compile_time_msg == mundy::sink() << "overflow " << "detected");
  std::cout << "Compile-time message: " << compile_time_msg.to_string() << std::endl;

  // A runtime integer makes this a StringSink.
  int i = 42;
  int n = 10;
  auto runtime_msg = mundy::sink() << "i = " << i << " but n = " << n;
  std::cout << "Runtime message: " << runtime_msg.to_string() << std::endl;

  // Using a sink directly as the message argument of an assertion.
  try {
    MUNDY_THROW_REQUIRE(i < n, std::out_of_range, mundy::sink() << "index " << i << " exceeds bound " << n);
  } catch (const std::out_of_range& e) {
    std::cout << "Caught: " << e.what() << std::endl;
  }
}

//---------------------------------------------------------------------------------------------------------------------//
// StringLiteral example.
//---------------------------------------------------------------------------------------------------------------------//
void stringLiteralExample() {
  std::cout << "\n--- StringLiteral ---\n" << std::endl;

  /*
    mundy::StringLiteral<N> wraps a string literal as a first-class
    compile-time value.  It can be used as a template parameter, stored in a
    type, and compared at compile time.

    make_string_literal("text") constructs one with automatic length
    deduction.  StringLiterals concatenate with + at compile time.
  */

  constexpr auto field_name = mundy::make_string_literal("position");
  static_assert(field_name == mundy::make_string_literal("position"));

  constexpr auto prefix = mundy::make_string_literal("particle.");
  constexpr auto full = prefix + field_name;
  static_assert(full == mundy::make_string_literal("particle.position"));

  std::cout << "field_name : " << field_name.data() << std::endl;
  std::cout << "full name  : " << full.data() << std::endl;

  // Use a StringLiteral as a non-type template parameter.
  // Different names produce different instantiations -- zero runtime cost.
  printFieldType<mundy::make_string_literal("position")>();
  printFieldType<mundy::make_string_literal("velocity")>();
}

//---------------------------------------------------------------------------------------------------------------------//
// RNG example.
//---------------------------------------------------------------------------------------------------------------------//
void rngExample() {
  std::cout << "\n--- RNG (Philox) ---\n" << std::endl;

  /*
    Mundy uses OpenRand's Philox counter-based random number generator.
    Counter-based RNGs are ideal for parallel simulation: each particle gets
    a unique counter and therefore a statistically independent random stream,
    without shared state or atomic operations.

    make_philox(seed, counter) wraps openrand::Philox with size_t inputs.
      seed    -- identifies the simulation run.  Change it to produce a
                 different realization of the same setup.
      counter -- identifies the stream within the run.  Typically a particle
                 index, a time-step number, or a combination of both.

    rng.rand<T>() returns the next uniform sample in [0, 1).
  */

  size_t seed = 12345;
  size_t counter = 0;

  auto rng = mundy::make_philox(seed, counter);

  std::cout << "Five uniform samples (seed=" << seed << ", counter=" << counter << "):" << std::endl;
  for (int k = 0; k < 5; ++k) {
    std::cout << "  " << rng.rand<double>() << std::endl;
  }

  // Streams with different counters are statistically independent.
  auto rng_7 = mundy::make_philox(seed, /*counter=*/7);
  std::cout << "First sample for counter=7: " << rng_7.rand<double>() << std::endl;

  // The same seed and counter reproduce the same stream -- useful for
  // reproducible tests and debugging.
  auto rng_replay = mundy::make_philox(seed, counter);
  std::cout << "Replayed first sample:      " << rng_replay.rand<double>() << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Tuple and variant example.
//---------------------------------------------------------------------------------------------------------------------//
void tupleAndVariantExample() {
  std::cout << "\n--- Tuple and Variant ---\n" << std::endl;

  /*
    mundy::tuple<Ts...> is a heterogeneous value container, similar to
    std::tuple but trimmed to the operations Mundy uses.  It is friendly to
    constexpr and device-side code.

    Build one with make_tuple; access elements by index or by type.
    Two tuples can be concatenated with tuple_cat.
  */

  auto t = mundy::make_tuple(42, 3.14, true);

  std::cout << "get<0>(t) = " << mundy::get<0>(t) << "  (int 42)" << std::endl;
  std::cout << "get<1>(t) = " << mundy::get<1>(t) << "  (double 3.14)" << std::endl;
  std::cout << "get<2>(t) = " << mundy::get<2>(t) << "  (bool true)" << std::endl;

  // Access by unique type when the type appears exactly once.
  std::cout << "get<double>(t) = " << mundy::get<double>(t) << "  (same as get<1>)" << std::endl;

  // Concatenate two tuples.
  auto a = mundy::make_tuple(1, 2);
  auto b = mundy::make_tuple(3.0, 4.0);
  auto ab = mundy::tuple_cat(a, b);
  std::cout << "tuple_cat size = " << mundy::tuple_size_v<decltype(ab)> << "  (should be 4)" << std::endl;

  /*
    mundy::variant<Ts...> holds exactly one alternative at a time, similar
    to std::variant.  It is useful when a variable can be one of several
    unrelated types -- for example, a particle field that might hold either
    an int counter or a double value.

    holds_alternative<T>(v) checks the active type.
    get<T>(v) retrieves the active value (debug-checked for the wrong type).
    visit(visitor, v) dispatches a generic lambda on the active alternative.
  */

  mundy::variant<int, double> v(2.5);

  std::cout << "holds_alternative<double>(v) = " << mundy::holds_alternative<double>(v) << "  (true=1)" << std::endl;
  std::cout << "get<double>(v) = " << mundy::get<double>(v) << std::endl;

  // Assign a different alternative.
  v = 7;
  std::cout << "after v=7: holds_alternative<int>(v) = " << mundy::holds_alternative<int>(v) << std::endl;

  // visit dispatches on the active type.
  auto result = mundy::visit([](const auto& x) { return static_cast<double>(x) * 2.0; }, v);
  std::cout << "visit(x -> x*2, v=7) = " << result << "  (should be 14)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  assertionsExample();
  stringSinkExample();
  stringLiteralExample();
  rngExample();
  tupleAndVariantExample();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
