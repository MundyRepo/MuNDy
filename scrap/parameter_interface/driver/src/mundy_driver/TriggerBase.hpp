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

#ifndef MUNDY_DRIVER_TRIGGERBASE_HPP_
#define MUNDY_DRIVER_TRIGGERBASE_HPP_

/// \file TriggerBase.hpp
/// \brief Declaration of the TriggerBase class

// C++ core libs
#include <memory>
#include <string>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy includes
#include <MundyDriver_config.hpp>  // for HAVE_MUNDYDRIVER_*

namespace mundy {

namespace driver {

/// \brief Enum for firing status of the trigger
enum class TRIGGERSTATUS { FIRED, SKIP, CONTINUE, BREAK };

/// \class TriggerBase
/// \brief Abstract base class for triggers during execution of mundy to decide if a MetaMethod is evaluated or not
///
/// TriggerBase is the generic base class that can control whether or not a MetaMethod is evaluated. This is done by
/// emitting an enum and deciding what to do with it in a TriggerAction.
class TriggerBase {
 public:
  //! \name Constructors and destructors
  //@{

  /// \brief Default constructor
  TriggerBase() = default;

  /// \brief Default destructor
  virtual ~TriggerBase() = default;

  //@}

  //! \name Actions
  //@{

  /// \brief Check the status of a trigger
  ///
  /// This check if the trigger fires, or has already been run this step. This means that once we evaluate a unique
  /// trigger, we can reuse the last status for this timestep, and not evaluate it again.
  /// \param [in] current_step
  /// \return
  virtual TRIGGERSTATUS trigger_check(size_t current_step) {
    if (current_step == last_evaluated_step_) {
      return last_status_;
    }

    last_evaluated_step_ = current_step;
    last_status_ = evaluate_trigger(current_step);
    return last_status_;
  }

  //@}

 protected:
  /// \brief Explicit constructors with parameters
  /// \param mutable_params
  explicit TriggerBase([[maybe_unused]] const Teuchos::ParameterList& mutable_params) {
  }

  /// \brief Evaluate this trigger (must be implemented by inheriting class)
  /// \param [in] current_step
  /// \return TRIGGERSTATUS on fired state
  virtual TRIGGERSTATUS evaluate_trigger(size_t current_step) = 0;

 private:
  /// \brief Last evaluated step
  size_t last_evaluated_step_ = 0;

  /// \brief Last evaluated status
  TRIGGERSTATUS last_status_ = TRIGGERSTATUS::FIRED;
};  // TriggerBase

}  // namespace driver

}  // namespace mundy

#endif  // MUNDY_DRIVER_TRIGGERBASE_HPP_
