//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_TIMEINTEGRATOR_H
#define TOYDNS_SRC_TIMEINTEGRATOR_H

#include <cstdlib>
#include <stdexcept>
#include <array>
#include <concepts>
#include <type_traits>

#include "Kokkos_Core.hpp"

#include "Utilities.h"
#include "KokkosTypedefs.h"

#include "cores/concepts/MeshConcept.h"
#include "cores/concepts/StateConcept.h"

template<std::size_t DIM_,
         typename StateArrayType_,
         concepts::Mesh MeshType_,
         typename AdvectionType_,
         typename BoundaryType_,
         std::floating_point ScalarType_ = double>
             requires (DIM_ > 0)
struct TimeIntegrator {
  static constexpr std::size_t DIM = DIM_;
  using StateArrayType = StateArrayType_;
  using MeshType = MeshType_;
  using AdvectionType = AdvectionType_;
  using BoundaryType = BoundaryType_;
  using ScalarType = ScalarType_;
  using StateType = typename StateArrayType::value_type;

  TimeIntegrator(
      const BoundaryType_& bdry,
      const AdvectionType_& adv,
      const Kokkos::Array<SizeType, DIM>& NG) :
      m_bdry(bdry), m_adv(adv), m_NG(NG) {

  };

  struct update_kernel {
    StateArrayType states;
    StateArrayType scratch;
    StateArrayType rhs;
    ScalarType dt;
    ScalarType alpha;
    ScalarType beta;

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(const Args... indices) const {
      static_assert(sizeof...(indices) == DIM, "MDRangePolicy must be used for DIM > 1!");
      static_assert(std::conjunction_v<std::is_integral<Args>...>);
      scratch(indices...).U = alpha * states(indices...).U
                             +beta * scratch(indices...).U
                             -beta * dt * rhs(indices...).U;
    }
  };


  void operator()(const ScalarType dt,
                  const MeshType& mesh,
                  const StateArrayType& states) const {

    StateArrayType rhs;

    const auto scratch_extents = mesh.N + 2 * m_NG;
    StateArrayType scratch = make_view<StateArrayType>("Us", scratch_extents);
    Kokkos::Array<Kokkos::pair<SizeType, SizeType>, DIM> interior_range;
    for(std::size_t d = 0; d < DIM; ++d) {
      interior_range[d] = Kokkos::make_pair(m_NG[d], mesh.N[d] + m_NG[d]);
    }

    // calculate rhs.
    const auto NS_rhs = [&, this]() {
      m_bdry(mesh, m_NG, scratch);
      rhs = make_view<StateArrayType>("RHS", mesh.N); // realloc
      m_adv(mesh, m_NG, scratch, rhs);
    };
    // update
    const auto updater = [&, this] (ScalarType alpha, ScalarType beta) {
      update_kernel kernel {.states = states,
                            .scratch = subview_(scratch, interior_range),
                            .rhs = rhs, .dt = dt, .alpha = alpha, .beta = beta};
      const Kokkos::Array<SizeType, DIM> lb{0};
      const Kokkos::Array<SizeType, DIM> ub = mesh.N;
      const std::string label = "Update";
      if constexpr (DIM == 1) {
        const auto policy =
            Kokkos::RangePolicy<ExecutionSpace>(lb[0], ub[0]);
        Kokkos::parallel_for(label, policy, kernel);
      } else {
        const auto policy =
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<DIM>>(lb, ub);
        Kokkos::parallel_for(label, policy, kernel);
      }
    };

    constexpr Kokkos::Array<ScalarType, 3> alphas = {0.0, 0.75, 1.0/3.0};
    constexpr Kokkos::Array<ScalarType, 3> betas = {1.0, 0.25, 2.0/3.0};

    Kokkos::deep_copy(subview_(scratch, interior_range), states);
    for(std::size_t stage = 0; stage < 3; ++stage) {
      NS_rhs();
      updater(alphas[stage], betas[stage]);
    }
    Kokkos::deep_copy(states, subview_(scratch, interior_range));
  }

private:
  BoundaryType  m_bdry;
  AdvectionType m_adv;
  Kokkos::Array<SizeType, DIM> m_NG;
};

#endif //TOYDNS_SRC_TIMEINTEGRATOR_H
