// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
// 
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// Teuchos::as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER


// Mundy
#include <MundyCore_config.hpp>                                     // for HAVE_MUNDYCORE_*

#ifdef HAVE_MUNDYCORE_TEUCHOS

// C++ core
#include <limits>     // for std::numeric_limits
#include <sstream>    // for std::ostringstream
#include <stdexcept>  // for std::out_of_range
#include <string>     // for std::string
#include <typeinfo>   // for typeid

// Teuchos
#include <Teuchos_ParameterList.hpp>             // for Teuchos::ParameterList
#include <Teuchos_StrUtils.hpp>                  // for Teuchos::Teuchos::StrUtils::printLines
#include <Teuchos_YamlParameterListHelpers.hpp>  // for Teuchos::getParametersFromYamlFile
#include <Teuchos_any.hpp>                       // for Teuchos::any

// Our variant of Teuchos::AnyNumberParameterEntryValidator
#include <mundy_core/OurAnyNumberParameterEntryValidator.hpp>  // for mundy::core::OurAnyNumberParameterEntryValidator

/// \brief Helper function for concatinating strings
#define MUNDY_CONCAT(a, b) a##b
#define MUNDY_CONCAT2(a, b) MUNDY_CONCAT(a, b)
#define MUNDY_CONCAT3(a, b, c) MUNDY_CONCAT2(a, MUNDY_CONCAT2(b, c))

/// \brief A helper macro for defining the get_XXX() functions without code duplication.
#define MUNDY_DEFINE_GET_BLAH(TYPE, TYPE_NAME, CONVERT_NUMBER_USING_FUNC, CONVERT_STRING_USING_FUNC)                \
  TYPE OurAnyNumberParameterEntryValidator::MUNDY_CONCAT2(get_, TYPE_NAME)(                                         \
      const Teuchos::ParameterEntry &entry, const std::string &param_name, const std::string &sublist_name,         \
      const bool active_query) const {                                                                              \
    const Teuchos::any &any_value = entry.getAny(active_query);                                                     \
    TYPE output;                                                                                                    \
    if (accepted_types_.is_short_allowed() && any_value.type() == typeid(short)) {                                  \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<short>(any_value));                                      \
    } else if (accepted_types_.is_unsigned_short_allowed() && any_value.type() == typeid(unsigned short)) {         \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<unsigned short>(any_value));                             \
    } else if (accepted_types_.is_int_allowed() && any_value.type() == typeid(int)) {                               \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<int>(any_value));                                        \
    } else if (accepted_types_.is_unsigned_int_allowed() && any_value.type() == typeid(unsigned int)) {             \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<unsigned int>(any_value));                               \
    } else if (accepted_types_.is_long_allowed() && any_value.type() == typeid(long)) {                             \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<long>(any_value));                                       \
    } else if (accepted_types_.is_unsigned_long_allowed() && any_value.type() == typeid(unsigned long)) {           \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<unsigned long>(any_value));                              \
    } else if (accepted_types_.is_long_long_allowed() && any_value.type() == typeid(long long)) {                   \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<long long>(any_value));                                  \
    } else if (accepted_types_.is_unsigned_long_long_allowed() && any_value.type() == typeid(unsigned long long)) { \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<unsigned long long>(any_value));                         \
    } else if (accepted_types_.is_float_allowed() && any_value.type() == typeid(float)) {                           \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<float>(any_value));                                      \
    } else if (accepted_types_.is_double_allowed() && any_value.type() == typeid(double)) {                         \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<double>(any_value));                                     \
    } else if (accepted_types_.is_long_double_allowed() && any_value.type() == typeid(long double)) {               \
      output = CONVERT_NUMBER_USING_FUNC(Teuchos::any_cast<long double>(any_value));                                \
    } else if (accepted_types_.is_string_allowed() && any_value.type() == typeid(std::string)) {                    \
      output = CONVERT_STRING_USING_FUNC<TYPE>(Teuchos::any_cast<std::string>(any_value));                          \
    } else {                                                                                                        \
      throw_type_error(entry, param_name, sublist_name);                                                            \
    }                                                                                                               \
    return output;                                                                                                  \
  }                                                                                                                 \
  TYPE OurAnyNumberParameterEntryValidator::MUNDY_CONCAT2(get_, TYPE_NAME)(                                         \
      Teuchos::ParameterList & param_list, const std::string &param_name, const TYPE &default_value) const {        \
    const Teuchos::ParameterEntry *entry = param_list.getEntryPtr(param_name);                                      \
    if (entry) return MUNDY_CONCAT2(get_, TYPE_NAME)(*entry, param_name, param_list.name(), true);                  \
    return param_list.get(param_name, default_value);                                                               \
  }

#define MUNDY_DEFINE_IS_BLAH_ALLOWED(TYPE_NAME)                                               \
  bool OurAnyNumberParameterEntryValidator::MUNDY_CONCAT3(is_, TYPE_NAME, _allowed)() const { \
    return accepted_types_.is_##TYPE_NAME##_allowed();                                        \
  }

// Inline helper functions
template <typename ConvertToType>
inline ConvertToType convert_string_using_stoi(const std::string &str) {
  const int i = std::stoi(str);

  // Check if the integer is negative but the target type is unsigned
  if constexpr (std::is_unsigned_v<ConvertToType>) {
    if (i < 0) {
      throw std::out_of_range("Error: value out of range for type " + std::string(typeid(ConvertToType).name()));
    }
    if (static_cast<unsigned int>(i) > std::numeric_limits<ConvertToType>::max()) {
      throw std::out_of_range("Error: value out of range for type " + std::string(typeid(ConvertToType).name()));
    }
  } else {
    if ((i < std::numeric_limits<ConvertToType>::min()) || (i > std::numeric_limits<ConvertToType>::max())) {
      throw std::out_of_range("Error: value out of range for type " + std::string(typeid(ConvertToType).name()));
    }
  }

  return static_cast<ConvertToType>(i);
}

template <typename ConvertToType>
inline ConvertToType convert_string_using_stoll(const std::string &str) {
  const long long ll = std::stoll(str);

  // Check if the long integer is negative but the target type is unsigned
  if constexpr (std::is_unsigned_v<ConvertToType>) {
    if (ll < 0) {
      throw std::out_of_range("Error: value out of range for type " + std::string(typeid(ConvertToType).name()));
    }
    if (static_cast<unsigned long long>(ll) > std::numeric_limits<ConvertToType>::max()) {
      throw std::out_of_range("Error: value out of range for type " + std::string(typeid(ConvertToType).name()));
    }
  } else {
    if ((ll < std::numeric_limits<ConvertToType>::min()) || (ll > std::numeric_limits<ConvertToType>::max())) {
      throw std::out_of_range("Error: value out of range for type " + std::string(typeid(ConvertToType).name()));
    }
  }

  return static_cast<ConvertToType>(ll);
}

template <typename ConvertToType>
inline ConvertToType convert_string_using_stod(const std::string &str) {
  const long double d = std::stod(str);

  if ((d < static_cast<long double>(std::numeric_limits<ConvertToType>::min())) ||
      (d > static_cast<long double>(std::numeric_limits<ConvertToType>::max()))) {
    throw std::out_of_range("Error: value out of range for type " + std::string(typeid(ConvertToType).name()));
  }
  return static_cast<ConvertToType>(d);
}

template <typename ConvertToType>
inline ConvertToType convert_string_using_nothing(const std::string &str) {
  return str;
}

namespace mundy {

namespace core {

// Constructors

OurAnyNumberParameterEntryValidator::OurAnyNumberParameterEntryValidator()
    : preferred_type_(PREFER_DOUBLE), accepted_types_(AcceptedTypes()) {
  finish_initialization();
}

OurAnyNumberParameterEntryValidator::OurAnyNumberParameterEntryValidator(EPreferredType const preferred_type,
                                                                         AcceptedTypes const &accepted_types)
    : preferred_type_(preferred_type), accepted_types_(accepted_types) {
  finish_initialization();
}

//  Local non-virtual validated lookup functions

MUNDY_DEFINE_GET_BLAH(short, short, Teuchos::as<short>, convert_string_using_stoi)
MUNDY_DEFINE_GET_BLAH(unsigned short, unsigned_short, Teuchos::as<unsigned short>, convert_string_using_stoi)
MUNDY_DEFINE_GET_BLAH(int, int, Teuchos::as<int>, convert_string_using_stoi)
MUNDY_DEFINE_GET_BLAH(unsigned int, unsigned_int, Teuchos::as<unsigned int>, convert_string_using_stoi)
MUNDY_DEFINE_GET_BLAH(long, long, Teuchos::as<long>, convert_string_using_stoll)
MUNDY_DEFINE_GET_BLAH(unsigned long, unsigned_long, Teuchos::as<unsigned long>, convert_string_using_stoll)
MUNDY_DEFINE_GET_BLAH(long long, long_long, Teuchos::as<long long>, convert_string_using_stoll)
MUNDY_DEFINE_GET_BLAH(unsigned long long, unsigned_long_long, Teuchos::as<unsigned long long>,
                      convert_string_using_stoll)
MUNDY_DEFINE_GET_BLAH(float, float, Teuchos::as<float>, convert_string_using_stod)
MUNDY_DEFINE_GET_BLAH(double, double, Teuchos::as<double>, convert_string_using_stod)
MUNDY_DEFINE_GET_BLAH(long double, long_double, Teuchos::as<long double>, convert_string_using_stod)
MUNDY_DEFINE_GET_BLAH(std::string, string, std::to_string, convert_string_using_nothing)

MUNDY_DEFINE_IS_BLAH_ALLOWED(short)
MUNDY_DEFINE_IS_BLAH_ALLOWED(unsigned_short)
MUNDY_DEFINE_IS_BLAH_ALLOWED(int)
MUNDY_DEFINE_IS_BLAH_ALLOWED(unsigned_int)
MUNDY_DEFINE_IS_BLAH_ALLOWED(long)
MUNDY_DEFINE_IS_BLAH_ALLOWED(unsigned_long)
MUNDY_DEFINE_IS_BLAH_ALLOWED(long_long)
MUNDY_DEFINE_IS_BLAH_ALLOWED(unsigned_long_long)
MUNDY_DEFINE_IS_BLAH_ALLOWED(float)
MUNDY_DEFINE_IS_BLAH_ALLOWED(double)
MUNDY_DEFINE_IS_BLAH_ALLOWED(long_double)
MUNDY_DEFINE_IS_BLAH_ALLOWED(string)

OurAnyNumberParameterEntryValidator::EPreferredType OurAnyNumberParameterEntryValidator::get_preferred_type() const {
  return preferred_type_;
}

// Overridden from Teuchos::ParameterEntryValidator

const std::string OurAnyNumberParameterEntryValidator::getXMLTypeName() const {
  return "OurAnyNumberValidator";
}

void OurAnyNumberParameterEntryValidator::printDoc(std::string const &docString, std::ostream &out) const {
  Teuchos::StrUtils::printLines(out, "# ", docString);
  out << "#  Accepted types: " << accepted_types_string_ << ".\n";
}

Teuchos::ParameterEntryValidator::ValidStringsList OurAnyNumberParameterEntryValidator::validStringValues() const {
  return Teuchos::null;
}

void OurAnyNumberParameterEntryValidator::validate(Teuchos::ParameterEntry const &entry, std::string const &param_name,
                                                   std::string const &sublist_name) const {
  // Validate that the parameter exists and can be converted to a double.
  // NOTE: Even if the target type will be an 'int', we don't know that here
  // so it will be better to assert that a 'double' can be created.  The type
  // 'double' has a very large exponent range and, subject to digit
  // truncation, a 'double' can represent every 'int' value.
  get_double(entry, param_name, sublist_name, false);
}

void OurAnyNumberParameterEntryValidator::validateAndModify(std::string const &param_name,
                                                            std::string const &sublist_name,
                                                            Teuchos::ParameterEntry *entry) const {
  TEUCHOS_TEST_FOR_EXCEPT(0 == entry);
  constexpr bool is_default = false;
  constexpr bool active_query = false;
  switch (preferred_type_) {
    case PREFER_SHORT:
      entry->setValue(get_short(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_UNSIGNED_SHORT:
      entry->setValue(get_unsigned_short(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_INT:
      entry->setValue(get_int(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_UNSIGNED_INT:
      entry->setValue(get_unsigned_int(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_LONG:
      entry->setValue(get_long(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_UNSIGNED_LONG:
      entry->setValue(get_unsigned_long(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_LONG_LONG:
      entry->setValue(get_long_long(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_UNSIGNED_LONG_LONG:
      entry->setValue(get_unsigned_long_long(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_FLOAT:
      entry->setValue(get_float(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_DOUBLE:
      entry->setValue(get_double(*entry, param_name, sublist_name, active_query), is_default);
      break;
    case PREFER_LONG_DOUBLE:
      entry->setValue(get_long_double(*entry, param_name, sublist_name, active_query), is_default);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPT("Error, Invalid EPreferredType value!");
  }
}

// private

void OurAnyNumberParameterEntryValidator::finish_initialization() {
  std::ostringstream oss;
  bool addedType = false;
  if (accepted_types_.is_short_allowed()) {
    oss << "\"short\"";
    addedType = true;
  }
  if (accepted_types_.is_unsigned_short_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"unsigned short\"";
    addedType = true;
  }
  if (accepted_types_.is_int_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"int\"";
    addedType = true;
  }
  if (accepted_types_.is_unsigned_int_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"unsigned int\"";
    addedType = true;
  }
  if (accepted_types_.is_long_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"long\"";
    addedType = true;
  }
  if (accepted_types_.is_unsigned_long_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"unsigned long\"";
    addedType = true;
  }
  if (accepted_types_.is_long_long_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"long long\"";
    addedType = true;
  }
  if (accepted_types_.is_unsigned_long_long_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"unsigned long long\"";
    addedType = true;
  }
  if (accepted_types_.is_float_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"float\"";
    addedType = true;
  }
  if (accepted_types_.is_double_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"double\"";
    addedType = true;
  }
  if (accepted_types_.is_long_double_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"long double\"";
    addedType = true;
  }
  if (accepted_types_.is_string_allowed()) {
    if (addedType) oss << ", ";
    oss << "\"string\"";
    addedType = true;
  }
  accepted_types_string_ = oss.str();
}

void OurAnyNumberParameterEntryValidator::throw_type_error(Teuchos::ParameterEntry const &entry,
                                                           std::string const &param_name,
                                                           std::string const &sublist_name) const {
  const std::string &entry_name = entry.getAny(false).typeName();
  TEUCHOS_TEST_FOR_EXCEPTION_PURE_MSG(true, Teuchos::Exceptions::InvalidParameterType,
                                      "Error, the parameter {param_name=\""
                                          << param_name
                                          << "\""
                                             ",type=\""
                                          << entry_name << "\"}" << "\nin the sublist \"" << sublist_name << "\""
                                          << "\nhas the wrong type."
                                          << "\n\nThe accepted types are: " << accepted_types_string_;);
}

}  // namespace core

}  // namespace mundy

#endif
