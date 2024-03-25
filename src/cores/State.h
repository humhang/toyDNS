//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_STATE_H
#define TOYDNS_SRC_STATE_H

#include <cstdlib>
#include <stdexcept>
#include <array>
#include <concepts>
#include <type_traits>

#include "Kokkos_Core.hpp"

#include "Utilities.h"
#include "KokkosTypedefs.h"

// The euler equation state associated with each cell
template<std::size_t DIM_, std::size_t NSCAL_ = 0, std::floating_point ScalarType_ = double>
    requires (DIM_ > 0)
struct State final {
  static constexpr std::size_t DIM = DIM_;
  static constexpr std::size_t NSCAL = NSCAL_;
  using ScalarType = ScalarType_;

  static auto make_field_labels() {
    Kokkos::Array<std::string, 2+DIM+NSCAL> ret;
    ret[0] = "rho";
    for(std::size_t k = 0; k < DIM; ++k) {
      std::ostringstream oss;
      oss << "rhou" << k;
      ret[1+k] = oss.str();
    }
    ret[DIM+1] = "E";
    for(std::size_t k = 0; k < NSCAL; ++k) {
      std::ostringstream oss;
      oss << "rhoY" << k;
      ret[DIM+2+k] = oss.str();
    }
    return ret;
  }

  static const inline Kokkos::Array<std::string, 2+DIM+NSCAL> labels = make_field_labels();

  Kokkos::Array<ScalarType, 1+DIM+1+NSCAL> U {};

  /*
   * accessor
   */
  KOKKOS_FUNCTION constexpr ScalarType& operator()(SizeType k) noexcept { return U[k]; }
  KOKKOS_FUNCTION constexpr const ScalarType& operator()(SizeType k) const noexcept { return U[k]; }
  KOKKOS_FUNCTION constexpr ScalarType& rho() noexcept {return U[0]; }
  KOKKOS_FUNCTION constexpr const ScalarType& rho() const noexcept {return U[0]; }
  KOKKOS_FUNCTION constexpr ScalarType& mu(std::size_t d) {return U[1+d]; }
  KOKKOS_FUNCTION constexpr const ScalarType& mu(std::size_t d) const {return U[1+d]; }
  KOKKOS_FUNCTION constexpr ScalarType& E() noexcept { return U[DIM+1]; }
  KOKKOS_FUNCTION constexpr const ScalarType& E() const noexcept { return U[DIM+1]; }
  KOKKOS_FUNCTION constexpr ScalarType& rhoY(std::size_t d) { return U[DIM+2+d]; }
  KOKKOS_FUNCTION constexpr const ScalarType& rhoY(std::size_t d) const { return U[DIM+2+d]; }
  KOKKOS_FUNCTION constexpr ScalarType KE() const {
    ScalarType ke{ };
    for(std::size_t d = 0; d < DIM; ++d) { ke += mu(d) * mu(d) / rho(); }
    return ScalarType{0.5} * ke;
  }


};


#endif //TOYDNS_SRC_STATE_H
