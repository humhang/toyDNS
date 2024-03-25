//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_INITIALIZER_H
#define TOYDNS_SRC_INITIALIZER_H


#include <string>
#include <filesystem>
#include <ranges>
#include <numeric>
#include <array>
#include <vector>
#include "Kokkos_Core.hpp"

#include "KokkosTypedefs.h"

#include "cores/concepts/MeshConcept.h"
#include "cores/concepts/StateConcept.h"

namespace concepts {
template<typename F, std::size_t DIM, typename StateType, typename ScalarType = double>
concept InitializationFunction = requires(F f, Kokkos::Array<ScalarType, DIM> pos) {
  { f(pos) } -> std::convertible_to<StateType>; // must also be marked as KOKKOS_FUNCTION or KOKKOS_LAMBDA
  requires !std::is_pointer_v<F>;
  requires std::is_copy_constructible_v<F>;
};
}


// implement according to concept Initializer
template<typename InitialzationFunctionType_,
         std::size_t DIM_,
         typename StateArrayType_,
         concepts::Mesh MeshType_,
         std::floating_point ScalarType_ = double>
    requires concepts::InitializationFunction<InitialzationFunctionType_, DIM_, typename StateArrayType_::value_type, ScalarType_> &&
             concepts::State<typename StateArrayType_::value_type>
class Initializer {
public:
  using InitialzationFunctionType = InitialzationFunctionType_;
  static constexpr std::size_t DIM = DIM_;
  using StateArrayType = StateArrayType_;
  using MeshType = MeshType_;
  using ScalarType = ScalarType_;
  using StateType = typename StateArrayType::value_type;

  struct execute_kernel {

    StateArrayType state;
    MeshType mesh;
    InitialzationFunctionType initializer;

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(const Args... indices) const {
      static_assert(sizeof...(indices) == DIM, "MDRangePolicy must be used for DIM > 1!");
      static_assert(std::conjunction_v<std::is_integral<Args>...>);
      state(indices...) = initializer(mesh(indices...));
    }

  };

  explicit Initializer(const InitialzationFunctionType_& fcn): m_fcn(fcn){ };

  void operator()(const MeshType& mesh,
                  const StateArrayType& states) const {

    execute_kernel kernel{.state = states,
                          .mesh = mesh,
                          .initializer = m_fcn};
    const Kokkos::Array<SizeType, DIM> lb{0};
    const Kokkos::Array<SizeType, DIM> ub = mesh.N;
    const std::string label = "Initialization";
    if constexpr (DIM == 1) {
      const auto policy =
          Kokkos::RangePolicy<ExecutionSpace>(lb[0], ub[0]);
      Kokkos::parallel_for(label, policy, kernel);
    } else {
      const auto policy =
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<DIM>>(lb, ub);
      Kokkos::parallel_for(label, policy, kernel);
    }
  }
private:
  const InitialzationFunctionType& m_fcn;
};

#endif //TOYDNS_SRC_INITIALIZER_H
