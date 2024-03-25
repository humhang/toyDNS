//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_MESH_H
#define TOYDNS_SRC_MESH_H

#include <cstdlib>
#include <stdexcept>
#include <array>
#include <tuple>
#include <concepts>
#include <type_traits>

#include "Kokkos_Core.hpp"

#include "Utilities.h"
#include "KokkosTypedefs.h"

template<std::size_t DIM_, std::floating_point ScalarType_ = double>
    requires (DIM_ > 0)
struct Mesh final {
  static constexpr std::size_t DIM = DIM_;
  using ScalarType = ScalarType_;

  Kokkos::Array<ScalarType, DIM> xlo{0};
  Kokkos::Array<ScalarType, DIM> xhi{0};
  Kokkos::Array<SizeType, DIM> N{0};
  Kokkos::Array<ScalarType, DIM> dx{0};

  // constructor
  KOKKOS_FUNCTION
  Mesh(const Kokkos::Array<ScalarType, DIM>& _xlo,
       const Kokkos::Array<ScalarType, DIM>& _xhi,
       const Kokkos::Array<SizeType, DIM>& _n):
       xlo(_xlo),
       xhi(_xhi),
       N(_n),
       dx((xhi - xlo) / N) {};
  KOKKOS_FUNCTION Mesh(const Mesh<DIM_, ScalarType_>&) = default;
  KOKKOS_FUNCTION Mesh(Mesh<DIM_, ScalarType_>&&) noexcept = default;

  // destructor
  KOKKOS_FUNCTION ~Mesh() = default;

  // assignment
  KOKKOS_FUNCTION Mesh<DIM_, ScalarType_>& operator=(const Mesh<DIM_, ScalarType_>&) = default;
  KOKKOS_FUNCTION Mesh<DIM_, ScalarType_>& operator=(Mesh<DIM_, ScalarType_>&&) noexcept = default;

  // query location from indices, Kokkos::Array version
  [[nodiscard]]
  KOKKOS_FUNCTION constexpr
  auto operator()(const Kokkos::Array<SizeType, DIM>& indices) const noexcept {
    Kokkos::Array<ScalarType, DIM> ret;
    for(std::size_t d = 0; d < DIM; ++d) {
      ret[d] = xlo[d] + dx[d]/2 + indices[d] * dx[d];
    }
    return ret;
  };

  template<typename... Args>
  [[nodiscard]]
  KOKKOS_FUNCTION constexpr
  auto operator()(const Args&... indices) const noexcept {
    static_assert(sizeof...(indices) == DIM);
    static_assert(std::conjunction_v<std::is_integral<std::decay_t<Args>>...>);
    return operator()(Kokkos::Array<SizeType, DIM>{indices...});
  }


};

#endif //TOYDNS_SRC_MESH_H
