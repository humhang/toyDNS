//
// Created by hang
//

#ifndef TOYDNS_SRC_BOUNDARY_H
#define TOYDNS_SRC_BOUNDARY_H

#include <cstddef>
#include <type_traits>
#include <array>

#include "Kokkos_Core.hpp"

#include "cores/concepts/StateConcept.h"
#include "cores/concepts/MeshConcept.h"


template<std::size_t DIM_,
         typename StateArrayType_,
         concepts::Mesh MeshType_,
         std::floating_point ScalarType_ = double>
    requires concepts::State<typename StateArrayType_::value_type>
struct Boundary {
  static constexpr std::size_t DIM = DIM_;
  using StateArrayType = StateArrayType_;
  using MeshType = MeshType_;
  using ScalarType = ScalarType_;
  using StateType = typename StateArrayType_::value_type;

  enum class BoundaryType {
    Periodic, Wall
  };

  explicit Boundary(BoundaryType bcs) : m_bcs(bcs) { };

  struct periodic_filling_executor {
    StateArrayType states;
    MeshType mesh;
    Kokkos::Array<SizeType, DIM> NG;

    KOKKOS_INLINE_FUNCTION
    static constexpr SizeType unidirectional_periodic_shift(const SizeType N, const SizeType NG, const SizeType Index) {
      return ((Index + N - NG)) % N + NG;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr auto periodic_shift(const Kokkos::Array<SizeType, DIM>& indices) const {
      Kokkos::Array<SizeType, DIM> ret;
      for(std::size_t d = 0; d < DIM; ++d) {
        ret[d] = unidirectional_periodic_shift(mesh.N[d], NG[d], indices[d]);
      }
      return ret;
    }

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(const Args... indices) const {
      static_assert(sizeof...(indices) == DIM, "MDRangePolicy must be used for DIM > 1!");
      static_assert(std::conjunction_v<std::is_integral<Args>...>);
      const auto shift_indices = periodic_shift({indices...});
      states(indices...) = get_from_index_array(states, shift_indices);
    }
  };

  struct wall_mirror_executor {
    StateArrayType states;
    MeshType mesh;
    Kokkos::Array<SizeType, DIM> NG;

    KOKKOS_INLINE_FUNCTION
    static constexpr SizeType unidirectional_mirror(const SizeType N, const SizeType NG, const SizeType Index) {
      if(Index < NG) return NG;
      else if (Index >= N + NG) return NG+N-1;
      else return Index;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr auto index_mirror(const Kokkos::Array<SizeType, DIM>& indices) const {
      Kokkos::Array<SizeType, DIM> ret;
      for(std::size_t d = 0; d < DIM; ++d) {
        ret[d] = unidirectional_mirror(mesh.N[d], NG[d], indices[d]);
      }
      return ret;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr bool is_bdry(const Kokkos::Array<SizeType, DIM>& indices) const {
      for(std::size_t d = 0; d < DIM; ++d) {
        if (indices[d] < NG[d] || indices[d] >= mesh.N[d]+NG[d]) return true;
      }
      return false;
    }

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(const Args... indices) const {
      static_assert(sizeof...(indices) == DIM, "MDRangePolicy must be used for DIM > 1!");
      static_assert(std::conjunction_v<std::is_integral<Args>...>);
      if(is_bdry({indices...})) {
        const auto mirror_index = index_mirror({indices...});
        const auto& mirror = get_from_index_array(states, mirror_index);
        states(indices...) = mirror;
        for(std::size_t d = 0; d < DIM; ++d) {
          states(indices...).mu(d) = -mirror.mu(d);
        }
      }
    }
  };

  void do_shift(const MeshType& mesh,
                const Kokkos::Array<SizeType, DIM>& NG,
                const StateArrayType& states) const {
    periodic_filling_executor kernel{.states = states,
        .mesh = mesh,
        .NG = NG};
    const Kokkos::Array<SizeType, DIM> lb = {0};
    const Kokkos::Array<SizeType, DIM> ub = mesh.N + 2 * NG;
    const std::string label = "FillBoundary";
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

  void do_mirror(const MeshType& mesh,
                 const Kokkos::Array<SizeType, DIM>& NG,
                 const StateArrayType& states) const {
    wall_mirror_executor kernel{.states = states,
                                .mesh = mesh,
                                .NG = NG};
    const Kokkos::Array<SizeType, DIM> lb = {0};
    const Kokkos::Array<SizeType, DIM> ub = mesh.N + 2 * NG;
    const std::string label = "FillBoundary";
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

  void operator()(const MeshType& mesh,
                  const Kokkos::Array<SizeType, DIM>& NG,
                  const StateArrayType& states) const {
    if(m_bcs == BoundaryType::Periodic) {
      do_shift(mesh, NG, states);
    } else if(m_bcs == BoundaryType::Wall) {
      do_mirror(mesh, NG, states);
    }
  }

private:
  BoundaryType m_bcs;

};
#endif //TOYDNS_SRC_BOUNDARY_H
