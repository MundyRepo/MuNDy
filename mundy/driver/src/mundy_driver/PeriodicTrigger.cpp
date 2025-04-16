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

/// \file PeriodicTrigger.hpp
/// \brief Implementation of the TriggerBase class

// C++ core libs

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy includes
#include <MundyDriver_config.hpp>            // for HAVE_MUNDYDRIVER_*
#include <mundy_driver/PeriodicTrigger.hpp>  // for PeriodicTrigger
#include <mundy_driver/TriggerBase.hpp>      // for TriggerBase

namespace mundy {

namespace driver {

//! \name Constructors and destructor
//@{

PeriodicTrigger::PeriodicTrigger(const Teuchos::ParameterList& mutable_params) : TriggerBase(mutable_params) {
  n_periodic_ = mutable_params.get<int>("n_periodic");
}

//@}

//! \name Actions
//@{

TRIGGERSTATUS PeriodicTrigger::evaluate_trigger(size_t current_step) {
  return (current_step % n_periodic_ == 0) ? TRIGGERSTATUS::FIRED : TRIGGERSTATUS::SKIP;
}

//@}

}  // namespace driver

}  // namespace mundy
