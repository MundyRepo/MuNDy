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

#ifndef MUNDY_CORE_OURANYNUMBERPARAMETERENTRYVALIDATOR_HPP_
#define MUNDY_CORE_OURANYNUMBERPARAMETERENTRYVALIDATOR_HPP_

// Mundy
#include <MundyCore_config.hpp>  // for HAVE_MUNDYCORE_*

#ifdef HAVE_MUNDYCORE_TEUCHOS

// C++ core
#include <stdexcept>  // for std::runtime_error
#include <string>     // for std::string

// Teuchos
#include <Teuchos_ParameterList.hpp>             // for Teuchos::Teuchos::ParameterList
#include <Teuchos_YamlParameterListHelpers.hpp>  // for Teuchos::getParametersFromYamlFile

/// \brief Our custom validator for accepts numbers from a number of different formats and converts them to numbers in
/// another format.
///
/// Current types are:
///  - Intreger: short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long
///  - Floating point: float, double, long double
///  - String: std::string
///
/// Objects of this type are meant to be used as both abstract objects passed
/// to <tt>Teuchos::Teuchos::ParameterList</tt> objects to be used to validate parameter
/// types and values, and to be used by the code that reads parameter values.
/// Having a single definition for the types of valids input and outputs for a
/// parameter value makes it easier to write error-free validated code.
namespace mundy {

namespace core {

class TEUCHOSPARAMETERLIST_LIB_DLL_EXPORT OurAnyNumberParameterEntryValidator
    : public Teuchos::ParameterEntryValidator {
 public:
  /** \name Public types */
  //@{

  /// \brief Determines what type is the preferred type.
  enum EPreferredType {
    PREFER_SHORT,
    PREFER_UNSIGNED_SHORT,
    PREFER_INT,
    PREFER_UNSIGNED_INT,
    PREFER_LONG,
    PREFER_UNSIGNED_LONG,
    PREFER_LONG_LONG,
    PREFER_UNSIGNED_LONG_LONG,
    PREFER_FLOAT,
    PREFER_DOUBLE,
    PREFER_LONG_DOUBLE,
    PREFER_STRING
  };

  /// \brief Determines the types that are accepted.
  class AcceptedTypes {
   public:
    /// \brief Allow all types or not on construction.
    explicit AcceptedTypes(bool allow_all_types_by_default = true)
        : allow_short_(allow_all_types_by_default),
          allow_unsigned_short_(allow_all_types_by_default),
          allow_int_(allow_all_types_by_default),
          allow_unsigned_int_(allow_all_types_by_default),
          allow_long_(allow_all_types_by_default),
          allow_unsigned_long_(allow_all_types_by_default),
          allow_long_long_(allow_all_types_by_default),
          allow_unsigned_long_long_(allow_all_types_by_default),
          allow_float_(allow_all_types_by_default),
          allow_double_(allow_all_types_by_default),
          allow_long_double_(allow_all_types_by_default),
          allow_string_(allow_all_types_by_default) {
    }

    //! \name Setters
    //@{

    /// \brief Set allow all integer types or not
    AcceptedTypes &allow_all_integer_types(bool should_allow_all_integer_types) {
      allow_short_ = should_allow_all_integer_types;
      allow_unsigned_short_ = should_allow_all_integer_types;
      allow_int_ = should_allow_all_integer_types;
      allow_unsigned_int_ = should_allow_all_integer_types;
      allow_long_ = should_allow_all_integer_types;
      allow_unsigned_long_ = should_allow_all_integer_types;
      allow_long_long_ = should_allow_all_integer_types;
      allow_unsigned_long_long_ = should_allow_all_integer_types;
      return *this;
    }

    /// \brief Set allow all floating point types or not
    AcceptedTypes &allow_all_floating_point_types(bool should_allow_all_floating_point_types) {
      allow_float_ = should_allow_all_floating_point_types;
      allow_double_ = should_allow_all_floating_point_types;
      allow_long_double_ = should_allow_all_floating_point_types;
      return *this;
    }

    /// \brief Set allow a <tt>short</tt> value or not
    AcceptedTypes &allow_short(bool should_allow_short) {
      allow_short_ = should_allow_short;
      return *this;
    }

    /// \brief Set allow an <tt>unsigned short</tt> value or not
    AcceptedTypes &allow_unsigned_short(bool should_allow_unsigned_short) {
      allow_unsigned_short_ = should_allow_unsigned_short;
      return *this;
    }

    /// \brief Set allow an <tt>int</tt> value or not
    AcceptedTypes &allow_int(bool should_allow_int) {
      allow_int_ = should_allow_int;
      return *this;
    }

    /// \brief Set allow an <tt>unsigned int</tt> value or not
    AcceptedTypes &allow_unsigned_int(bool should_allow_unsigned_int) {
      allow_unsigned_int_ = should_allow_unsigned_int;
      return *this;
    }

    /// \brief Set allow an <tt>long</tt> value or not
    AcceptedTypes &allow_long(bool should_allow_long) {
      allow_long_ = should_allow_long;
      return *this;
    }

    /// \brief Set allow an <tt>unsigned long</tt> value or not
    AcceptedTypes &allow_unsigned_long(bool should_allow_unsigned_long) {
      allow_unsigned_long_ = should_allow_unsigned_long;
      return *this;
    }

    /// \brief Set allow an <tt>long long</tt> value or not
    AcceptedTypes &allow_long_long(bool should_allow_long_long) {
      allow_long_long_ = should_allow_long_long;
      return *this;
    }

    /// \brief Set allow an <tt>unsigned long long</tt> value or not
    AcceptedTypes &allow_unsigned_long_long(bool should_allow_unsigned_long_long) {
      allow_unsigned_long_long_ = should_allow_unsigned_long_long;
      return *this;
    }

    /// \brief Set allow a <tt>float</tt> value or not
    AcceptedTypes &allow_float(bool should_allow_float) {
      allow_float_ = should_allow_float;
      return *this;
    }

    /// \brief Set allow a <tt>double</tt> value or not
    AcceptedTypes &allow_double(bool should_allow_double) {
      allow_double_ = should_allow_double;
      return *this;
    }

    /// \brief Set allow a <tt>long double</tt> value or not
    AcceptedTypes &allow_long_double(bool should_allow_long_double) {
      allow_long_double_ = should_allow_long_double;
      return *this;
    }

    /// \brief Set allow a <tt>std::string</tt> value or not
    AcceptedTypes &allow_string(bool should_allow_string) {
      allow_string_ = should_allow_string;
      return *this;
    }
    //@}

    //! \name Getters
    //@{

    /// \brief Allow a <tt>short</tt> value?
    bool is_short_allowed() const {
      return allow_short_;
    }

    /// \brief Allow an <tt>unsigned short</tt> value?
    bool is_unsigned_short_allowed() const {
      return allow_unsigned_short_;
    }

    /// \brief Allow an <tt>int</tt> value?
    bool is_int_allowed() const {
      return allow_int_;
    }

    /// \brief Allow an <tt>unsigned int</tt> value?
    bool is_unsigned_int_allowed() const {
      return allow_unsigned_int_;
    }

    /// \brief Allow a <tt>long</tt> value?
    bool is_long_allowed() const {
      return allow_long_;
    }

    /// \brief Allow an <tt>unsigned long</tt> value?
    bool is_unsigned_long_allowed() const {
      return allow_unsigned_long_;
    }

    /// \brief Allow a <tt>long long</tt> value?
    bool is_long_long_allowed() const {
      return allow_long_long_;
    }

    /// \brief Allow an <tt>unsigned long long</tt> value?
    bool is_unsigned_long_long_allowed() const {
      return allow_unsigned_long_long_;
    }

    /// \brief Allow a <tt>float</tt> value?
    bool is_float_allowed() const {
      return allow_float_;
    }

    /// \brief Allow a <tt>double</tt> value?
    bool is_double_allowed() const {
      return allow_double_;
    }

    /// \brief Allow a <tt>long double</tt> value?
    bool is_long_double_allowed() const {
      return allow_long_double_;
    }

    /// \brief Allow a <tt>std::string</tt> value?
    bool is_string_allowed() const {
      return allow_string_;
    }
    //@}

   private:
    bool allow_short_;
    bool allow_unsigned_short_;
    bool allow_int_;
    bool allow_unsigned_int_;
    bool allow_long_;
    bool allow_unsigned_long_;
    bool allow_long_long_;
    bool allow_unsigned_long_long_;
    bool allow_float_;
    bool allow_double_;
    bool allow_long_double_;
    bool allow_string_;
  };
  //@}

  /// \name Constructors
  //@{

  /// \brief Construct with a preferrded type of double and accept all types.
  OurAnyNumberParameterEntryValidator();

  /// \brief Construct with allowed input and output types and the preferred type.
  /// \param preferred_type [in] Determines the preferred type.  This enum value is used to set the default value in the
  /// override <tt>validateAndModify()</tt>.
  /// \param accepted_types [in] Determines the types that are allowed in the parameter list.
  OurAnyNumberParameterEntryValidator(EPreferredType const preferred_type, AcceptedTypes const &accepted_types);
  //@}

  /// \name Local non-virtual validated lookup functions
  //@{

  /// \brief Get a short value from a parameter entry.
  /// Will call std::stoi.
  short get_short(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                  const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get an unsigned short value from a parameter entry.
  /// Will call std::stoi.
  unsigned short get_unsigned_short(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                                    const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get an int value from a parameter entry.
  /// Will call std::stoi.
  int get_int(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
              const std::string &sublist_name = "", const bool active_query = true) const;

  unsigned int get_unsigned_int(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                                const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get a long value from a parameter entry.
  long get_long(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get an unsigned long value from a parameter entry.
  unsigned long get_unsigned_long(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                                  const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get a long long value from a parameter entry.
  long long get_long_long(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                          const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get an unsigned long long value from a parameter entry.
  unsigned long long get_unsigned_long_long(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                                            const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get a float value from a parameter entry.
  float get_float(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                  const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get a double value from a parameter entry.
  double get_double(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                    const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get a long double value from a parameter entry.
  long double get_long_double(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                              const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Get a std::string value from a parameter entry.
  std::string get_string(const Teuchos::ParameterEntry &entry, const std::string &param_name = "",
                         const std::string &sublist_name = "", const bool active_query = true) const;

  /// \brief Lookup parameter from a parameter list and return as a short value.
  short get_short(Teuchos::ParameterList &param_list, const std::string &param_name, const short &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as an unsigned short value.
  unsigned short get_unsigned_short(Teuchos::ParameterList &param_list, const std::string &param_name,
                                    const unsigned short &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as an int value.
  int get_int(Teuchos::ParameterList &param_list, const std::string &param_name, const int &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as an unsigned int value.
  unsigned int get_unsigned_int(Teuchos::ParameterList &param_list, const std::string &param_name,
                                const unsigned int &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as a long value.
  long get_long(Teuchos::ParameterList &param_list, const std::string &param_name, const long &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as an unsigned long value.
  unsigned long get_unsigned_long(Teuchos::ParameterList &param_list, const std::string &param_name,
                                  const unsigned long &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as a long long value.
  long long get_long_long(Teuchos::ParameterList &param_list, const std::string &param_name,
                          const long long &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as an unsigned long long value.
  unsigned long long get_unsigned_long_long(Teuchos::ParameterList &param_list, const std::string &param_name,
                                            const unsigned long long &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as a float value.
  float get_float(Teuchos::ParameterList &param_list, const std::string &param_name, const float &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as a double value.
  double get_double(Teuchos::ParameterList &param_list, const std::string &param_name,
                    const double &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as a long double value.
  long double get_long_double(Teuchos::ParameterList &param_list, const std::string &param_name,
                              const long double &default_value) const;

  /// \brief Lookup parameter from a parameter list and return as an std::string value.
  std::string get_string(Teuchos::ParameterList &param_list, const std::string &param_name,
                         const std::string &default_value) const;

  /// \brief Lookup whether or not shorts are allowed.
  bool is_short_allowed() const;

  /// \brief Lookup whether or not unsigned shorts are allowed.
  bool is_unsigned_short_allowed() const;

  /// \brief Lookup whether or not ints are allowed.
  bool is_int_allowed() const;

  /// \brief Lookup whether or not unsigned ints are allowed.
  bool is_unsigned_int_allowed() const;

  /// \brief Lookup whether or not longs are allowed.
  bool is_long_allowed() const;

  /// \brief Lookup whether or not unsigned longs are allowed.
  bool is_unsigned_long_allowed() const;

  /// \brief Lookup whether or not long longs are allowed.
  bool is_long_long_allowed() const;

  /// \brief Lookup whether or not unsigned long longs are allowed.
  bool is_unsigned_long_long_allowed() const;

  /// \brief Lookup whether or not floats are allowed.
  bool is_float_allowed() const;

  /// \brief Lookup whether or not doubles are allowed.
  bool is_double_allowed() const;

  /// \brief Lookup whether or not long doubles are allowed.
  bool is_long_double_allowed() const;

  /// \brief Lookup whether or not strings are allowed.
  bool is_string_allowed() const;

  /// \brief Lookup the preferred type
  EPreferredType get_preferred_type() const;

  /// \brief Gets the string representation of a given preferred type enum.
  static const std::string &get_preffered_type_string(EPreferredType enum_value) {
    switch (enum_value) {
      case PREFER_SHORT:
        return get_short_enum_string();
      case PREFER_UNSIGNED_SHORT:
        return get_unsigned_short_enum_string();
      case PREFER_INT:
        return get_int_enum_string();
      case PREFER_UNSIGNED_INT:
        return get_unsigned_int_enum_string();
      case PREFER_LONG:
        return get_long_enum_string();
      case PREFER_UNSIGNED_LONG:
        return get_unsigned_long_enum_string();
      case PREFER_LONG_LONG:
        return get_long_long_enum_string();
      case PREFER_UNSIGNED_LONG_LONG:
        return get_unsigned_long_long_enum_string();
      case PREFER_FLOAT:
        return get_float_enum_string();
      case PREFER_DOUBLE:
        return get_double_enum_string();
      case PREFER_LONG_DOUBLE:
        return get_long_double_enum_string();
      case PREFER_STRING:
        return get_string_enum_string();
      default:
        const std::string type_string(Teuchos::toString(enum_value));
        throw std::runtime_error("Cannot convert enum_value: " + type_string + " to a string");
    }
  }

  /// \brief Gets the preferred type enum associated with a give string.
  static EPreferredType get_preffered_type_string_enum(const std::string &enum_string) {
    if (enum_string == get_short_enum_string()) {
      return PREFER_SHORT;
    } else if (enum_string == get_unsigned_short_enum_string()) {
      return PREFER_UNSIGNED_SHORT;
    } else if (enum_string == get_int_enum_string()) {
      return PREFER_INT;
    } else if (enum_string == get_unsigned_int_enum_string()) {
      return PREFER_UNSIGNED_INT;
    } else if (enum_string == get_long_enum_string()) {
      return PREFER_LONG;
    } else if (enum_string == get_unsigned_long_enum_string()) {
      return PREFER_UNSIGNED_LONG;
    } else if (enum_string == get_long_long_enum_string()) {
      return PREFER_LONG_LONG;
    } else if (enum_string == get_unsigned_long_long_enum_string()) {
      return PREFER_UNSIGNED_LONG_LONG;
    } else if (enum_string == get_float_enum_string()) {
      return PREFER_FLOAT;
    } else if (enum_string == get_double_enum_string()) {
      return PREFER_DOUBLE;
    } else if (enum_string == get_long_double_enum_string()) {
      return PREFER_LONG_DOUBLE;
    } else if (enum_string == get_string_enum_string()) {
      return PREFER_STRING;
    } else {
      throw std::runtime_error("Cannot convert enum_string: " + enum_string + " to an enum");
    }
  }
  //@}

  /// \name Overridden from ParameterEntryValidator
  //@{

  /// \brief
  const std::string getXMLTypeName() const;

  /// \brief
  void printDoc(std::string const &docString, std::ostream &out) const;

  /// \brief
  ValidStringsList validStringValues() const;

  /// \brief
  void validate(Teuchos::ParameterEntry const &entry, std::string const &param_name,
                std::string const &sublist_name) const;

  /// \brief
  void validateAndModify(std::string const &param_name, std::string const &sublist_name,
                         Teuchos::ParameterEntry *entry) const;
  //@}

 private:
  // ////////////////////////////
  // Private data members

  EPreferredType preferred_type_;
  std::string accepted_types_string_;

// use pragmas to disable some false-positive warnings for windows sharedlibs export
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
  const AcceptedTypes accepted_types_;
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  // ////////////////////////////
  // Private member functions

  /// \brief Gets the string representing the "short" preferred type enum
  static const std::string &get_short_enum_string() {
    static const std::string short_enum_string_ = Teuchos::TypeNameTraits<short>::name();
    return short_enum_string_;
  }

  /// \brief Gets the string representing the "unsigned short" preferred type enum
  static const std::string &get_unsigned_short_enum_string() {
    static const std::string unsigned_short_enum_string_ = Teuchos::TypeNameTraits<unsigned short>::name();
    return unsigned_short_enum_string_;
  }

  /// \brief Gets the string representing the "int" preferred type enum
  static const std::string &get_int_enum_string() {
    static const std::string int_enum_string_ = Teuchos::TypeNameTraits<int>::name();
    return int_enum_string_;
  }

  /// \brief Gets the string representing the "unsigned int" preferred type enum
  static const std::string &get_unsigned_int_enum_string() {
    static const std::string unsigned_int_enum_string_ = Teuchos::TypeNameTraits<unsigned int>::name();
    return unsigned_int_enum_string_;
  }

  /// \brief Gets the string representing the "long" preferred type enum
  static const std::string &get_long_enum_string() {
    static const std::string long_enum_string_ = Teuchos::TypeNameTraits<long>::name();
    return long_enum_string_;
  }

  /// \brief Gets the string representing the "unsigned long" preferred type enum
  static const std::string &get_unsigned_long_enum_string() {
    static const std::string unsigned_long_enum_string_ = Teuchos::TypeNameTraits<unsigned long>::name();
    return unsigned_long_enum_string_;
  }

  /// \brief Gets the string representing the "long long" preferred type enum
  static const std::string &get_long_long_enum_string() {
    static const std::string long_long_enum_string_ = Teuchos::TypeNameTraits<long long>::name();
    return long_long_enum_string_;
  }

  /// \brief Gets the string representing the "unsigned long long" preferred type enum
  static const std::string &get_unsigned_long_long_enum_string() {
    static const std::string unsigned_long_long_enum_string_ = Teuchos::TypeNameTraits<unsigned long long>::name();
    return unsigned_long_long_enum_string_;
  }

  /// \brief Gets the string representing the "float" preferred type enum
  static const std::string &get_float_enum_string() {
    static const std::string float_enum_string_ = Teuchos::TypeNameTraits<float>::name();
    return float_enum_string_;
  }

  /// \brief Gets the string representing the "double" preferred type enum
  static const std::string &get_double_enum_string() {
    static const std::string double_enum_string_ = Teuchos::TypeNameTraits<double>::name();
    return double_enum_string_;
  }

  /// \brief Gets the string representing the "long double" preferred type enum
  static const std::string &get_long_double_enum_string() {
    static const std::string long_double_enum_string_ = Teuchos::TypeNameTraits<long double>::name();
    return long_double_enum_string_;
  }

  /// \brief Gets the string representing the "string" preferred type enum
  static const std::string &get_string_enum_string() {
    static const std::string string_enum_string_ = Teuchos::TypeNameTraits<std::string>::name();
    return string_enum_string_;
  }

  void finish_initialization();

  void throw_type_error(Teuchos::ParameterEntry const &entry, std::string const &param_name,
                        std::string const &sublist_name) const;
};  // class OurAnyNumberParameterEntryValidator

}  // namespace core

}  // namespace mundy

#endif

#endif  // MUNDY_CORE_OURANYNUMBERPARAMETERENTRYVALIDATOR_HPP_
