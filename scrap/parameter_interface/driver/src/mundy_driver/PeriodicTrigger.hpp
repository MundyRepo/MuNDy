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

#ifndef MUNDY_DRIVER_PERIODICTRIGGER_HPP_
#define MUNDY_DRIVER_PERIODICTRIGGER_HPP_

/// \file PeriodicTrigger.hpp
/// \brief Declaration of the PeriodicTrigger concrete class

// C++ core libs

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy includes
#include <MundyDriver_config.hpp>        // for HAVE_MUNDYDRIVER_*
#include <mundy_driver/TriggerBase.hpp>  // for TriggerBase

namespace mundy {

namespace driver {

/// \class PeriodicTrigger
/// \brief Trigger that will fire every N steps
class PeriodicTrigger : public TriggerBase {
 public:
  //! \name Constructors and destructors
  //@{

  /// \brief Constructor
  explicit PeriodicTrigger(const Teuchos::ParameterList& mutable_params);

  //@}

  //! \name Actions
  //@{

  /// \brief Evaluate this trigger
  /// \param [in] current_step
  /// \return TRIGGERSTATUS on fired state
  TRIGGERSTATUS evaluate_trigger(size_t current_step) override;

  //@}

 private:
  /// \brief Number of steps between evaluation
  size_t n_periodic_;
};  // PeriodicTrigger

}  // namespace driver

}  // namespace mundy

#endif  // MUNDY_DRIVER_PERIODICTRIGGER_HPP_
