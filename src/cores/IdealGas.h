//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_IDEALGAS_H
#define TOYDNS_SRC_IDEALGAS_H

#include <concepts>
#include <type_traits>

template<typename ScalarType = double>
    requires std::floating_point<ScalarType>
struct IdealGas {
  ScalarType gamma;

  template<typename StateType>
    requires std::same_as<ScalarType, typename StateType::ScalarType>
  KOKKOS_INLINE_FUNCTION
  ScalarType Pressure(const StateType& state) const {
    return (gamma - ScalarType{1}) * (state.E() - state.KE());
  }

  template<typename StateType>
  requires std::same_as<ScalarType, typename StateType::ScalarType>
  KOKKOS_INLINE_FUNCTION
  ScalarType Enthalpy(const StateType& state) const {
    return (state.E() + Pressure(state)) / state.rho();
  }

  template<typename StateType>
  requires std::same_as<ScalarType, typename StateType::ScalarType>
  KOKKOS_INLINE_FUNCTION
  ScalarType SpeedOfSound(const StateType& state) const {
    return std::sqrt(gamma * Pressure(state) / state.rho());
  }

  template<typename StateType>
  requires std::same_as<ScalarType, typename StateType::ScalarType>
  KOKKOS_INLINE_FUNCTION
  ScalarType b(const StateType&) const {
    // let p = p(rho, U), b = (1/rho) (dp/dU)_\rho
    return gamma-1;
  }

  template<typename StateType>
  requires std::same_as<ScalarType, typename StateType::ScalarType>
  KOKKOS_INLINE_FUNCTION
  ScalarType theta(const StateType& state) const {
    return state.KE() / state.rho();
  }

};



static_assert(std::is_standard_layout_v<IdealGas<>>);

#endif //TOYDNS_SRC_IDEALGAS_H
