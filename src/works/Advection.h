//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_ADVECTION_H
#define TOYDNS_SRC_ADVECTION_H

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
    typename GasType_,
    typename StateArrayType_,
    concepts::Mesh MeshType_,
    std::floating_point ScalarType_ = double>
struct Advection {
  static constexpr std::size_t DIM = DIM_;
  using GasType = GasType_;
  using StateArrayType = StateArrayType_;
  using MeshType = MeshType_;
  using ScalarType = ScalarType_;
  using StateType = typename StateArrayType_::value_type;
  using FluxArrayType = Kokkos::View<add_nlevels_pointer_t<StateType, DIM>, Device>;
  static constexpr std::size_t NSCAL = StateType::NSCAL;

  explicit Advection(
      const GasType_& gas,
      const MeshType_& mesh):
      m_gas(gas){
    const auto extents = mesh.N;
    for(std::size_t d = 0; d < DIM; ++d) {
      auto ext = extents;
      ext[d] = ext[d] + 1;

      const std::string flux_label = std::string("Flux-") + std::to_string(d);
      m_fluxes[d] = make_view<FluxArrayType>(flux_label, ext);
    }
  }

  using EigenVectorsType = DenseMatrix<NSCAL+DIM+2, NSCAL+DIM+2, ScalarType>;
  using EigenvaluesType  = Kokkos::Array<ScalarType, NSCAL+DIM+2>;
  template<std::size_t DIR>
  KOKKOS_FUNCTION
  static void EulerEigenSystem_Impl(const GasType& gas,
                                    const Kokkos::Array<ScalarType, DIM>& u,
                                    const Kokkos::Array<ScalarType, NSCAL>& Y,
                                    const ScalarType& H,
                                    const ScalarType& c,
                                    EigenVectorsType& L,
                                    EigenVectorsType& R,
                                    EigenvaluesType& Lambda) {


    ScalarType q2 = 0.0;
    for(std::size_t d = 0; d < DIM; ++d) {
      q2 += u[d] * u[d];
    }
    //const auto b = gas.b(...);
    //const auto theta = gas.theta(...);
    const auto b = gas.gamma-1;
    const auto theta = 0.5 * q2;

    const auto c2 = c*c;

    /*
     * direction mapper according to DIR,
     * e.g., if DIR = 0, map x to x, y to y, z to z
     *       if DIR = 1, map x to y, y to z, z to x
     */
    std::size_t dir_map[DIM];
    for(std::size_t d = 0; d < DIM; ++d) {
      dir_map[d] = (d + DIR) % DIM;
    }

    /*
     * right eigenvectors
     */
    R = 0.0;
    // 1-wave: acoustic wave: u-c
    R(0, 0) = 1.0;
    R(1+dir_map[0], 0) = u[dir_map[0]] - c;
    for(std::size_t d = 1; d < DIM; ++d) {
      R(1+dir_map[d], 0) = u[dir_map[d]];
    }
    R(DIM+1, 0) = H - c*u[dir_map[0]];
    for(std::size_t n = 0; n < NSCAL; ++n) {
      R(DIM+2+n, 0) = Y[n];
    }

    // 2-wave to DIM-wave: contact wave on transversal velocity
    for(std::size_t k = 1; k < DIM; ++k) {
      R(1+dir_map[k], k) = 1.0;
      R(DIM+1, k) = u[dir_map[k]];
    }

    // DIM+1 wave: contact entropy wave
    R(0, DIM) = 1.0;
    for(std::size_t d = 0; d < DIM; ++d) {
      R(1+dir_map[d], DIM) = u[dir_map[d]];
    }
    R(DIM+1, DIM) = H - c2 / b;

    // DIM+2 wave: acoustic wave: u+c
    R(0, DIM+1) = 1.0;
    R(1+dir_map[0], DIM+1) = u[dir_map[0]] + c;
    for(std::size_t d = 1; d < DIM; ++d) {
      R(1+dir_map[d], DIM+1) = u[dir_map[d]];
    }
    R(DIM+1, DIM+1) = H + c*u[dir_map[0]];
    for(std::size_t n = 0; n < NSCAL; ++n) {
      R(DIM+2+n, DIM+1) = Y[n];
    }

    // DIM+3 to DIM+3+NSCAL-1 wave: contact wave on passive scalar
    for(std::size_t k = DIM+2; k < DIM+2+NSCAL; ++k) {
      R(k, k) = 1.0;
    }

    /*
     * left eigenvectors
     */
    L = 0.0;
    L(0, 0) = theta + u[dir_map[0]] * c / b;
    L(0, 1+dir_map[0]) = -u[dir_map[0]] - c / b;
    for(std::size_t d = 1; d < DIM; ++d) {
      L(0, 1+dir_map[d]) = -u[dir_map[d]];
    }
    L(0, DIM+1) = 1.0;

    for(std::size_t k = 1; k < DIM; ++k) {
      L(k, 0) = -2*c2/b * u[dir_map[k]];
      L(k, 1+dir_map[k]) = 2*c2/b;
    }

    L(DIM, 0) = 2*H - 2*q2;
    for(std::size_t d = 0; d < DIM; ++d) {
      L(DIM, 1+dir_map[d]) = 2 * u[dir_map[d]];
    }
    L(DIM, DIM+1) = -2.0;

    L(DIM+1, 0) = theta - u[dir_map[0]] * c / b;
    L(DIM+1, 1+dir_map[0]) = -u[dir_map[0]] + c / b;
    for(std::size_t d = 1; d < DIM; ++d) {
      L(DIM+1, 1+dir_map[d]) = -u[dir_map[d]];
    }
    L(DIM+1, DIM+1) = 1.0;

    for(std::size_t n = 0; n < NSCAL; ++n) {
      L(DIM+2+n, 0) = -q2 * Y[n];
      for(std::size_t d = 0; d < DIM; ++d) {
        L(DIM+2+n, 1+dir_map[d]) = 2*Y[n]*u[dir_map[d]];
      }
      L(DIM+2+n, DIM+1) = -2.0 * Y[n];
      L(DIM+2+n, DIM+2+n) = 2*c2/b;
    }

    L *= (b / 2 / c2);

    /*
     * wave speed
     */
    Lambda[0] = u[dir_map[0]] - c; //acoustics wave
    for(std::size_t d = 1; d < DIM+1; ++d) {
      Lambda[d] = u[dir_map[0]]; // entropy and transversal wave
    }
    Lambda[DIM+1] = u[dir_map[0]] + c; //acoustics wave
    for(std::size_t k = DIM+2; k < DIM+2+NSCAL; ++k) {
      Lambda[k] = u[dir_map[0]]; // contact wave of passive scalar
    }


  }

  KOKKOS_FUNCTION
  static void RoeAverage(const GasType& gas,
                         const StateType& sL,
                         const StateType& sR,
                         Kokkos::Array<ScalarType, DIM>& u,
                         Kokkos::Array<ScalarType, NSCAL>& Y,
                         ScalarType& H,
                         ScalarType& c) {
    const auto sqrtRhoL = std::sqrt(sL.rho());
    const auto sqrtRhoR = std::sqrt(sR.rho());
    const auto rL = sqrtRhoL / (sqrtRhoL + sqrtRhoR);
    const auto rR = sqrtRhoR / (sqrtRhoL + sqrtRhoR);
    const auto HL = gas.Enthalpy(sL);
    const auto HR = gas.Enthalpy(sR);

    H = HL * rL + HR * rR;
    for(std::size_t d = 0; d < DIM; ++d) {
      u[d] = (sL.mu(d) / sL.rho()) * rL + (sR.mu(d) / sR.rho()) * rR;
    }
    for(std::size_t n = 0; n < NSCAL; ++n) {
      Y[n] = (sL.rhoY(n) / sL.rho()) * rL + (sR.rhoY(n) / sR.rho()) * rR;
    }
    ScalarType q2 = 0;
    for(std::size_t d = 0; d < DIM; ++d) {
      q2 += 0.5 * u[d] * u[d];
    }
    c = std::sqrt((gas.gamma-1)*(H - q2));
  }

  template<std::size_t DIR>
  KOKKOS_FUNCTION
  static void EulerEigenSystem(const GasType& gas,
                               const StateType& states,
                               EigenVectorsType& L,
                               EigenVectorsType& R,
                               EigenvaluesType& Lambda) {

    Kokkos::Array<ScalarType, DIM> u;
    for(std::size_t d = 0; d < DIM; ++d) {
      u[d] = states.mu(d) / states.rho();
    }
    Kokkos::Array<ScalarType, NSCAL> Y;
    for(std::size_t n = 0; n < NSCAL; ++n) {
      Y[n] = states.rhoY(n) / states.rho();
    }

    const auto H = gas.Enthalpy(states);
    const auto c = gas.SpeedOfSound(states);

    EulerEigenSystem_Impl<DIR>(u, Y, H, c, L, R, Lambda);
  }


  template<std::size_t DIR>
  KOKKOS_FUNCTION
  static void EulerRoeEigenSystem(const GasType& gas,
                                  const StateType& statesL,
                                  const StateType& statesR,
                                  EigenVectorsType& L,
                                  EigenVectorsType& R,
                                  EigenvaluesType& Lambda) {
    Kokkos::Array<ScalarType, DIM> u;
    Kokkos::Array<ScalarType, NSCAL> Y;
    ScalarType H;
    ScalarType c;

    RoeAverage(gas, statesL, statesR, u, Y, H, c);
    EulerEigenSystem_Impl<DIR>(gas, u, Y, H, c, L, R, Lambda);
  }


  template<std::size_t DIR>
  KOKKOS_FUNCTION
  static StateType Flux(const GasType& gas,
                        const StateType& states) {

    StateType F;
    const ScalarType ud = states.mu(DIR) / states.rho();
    const ScalarType p = gas.Pressure(states);
    for(std::size_t k = 0; k < DIM+NSCAL+2; ++k) {
      F(k) = ud * states(k);
    }
    F.E() += ud * p;
    F.mu(DIR) += p;
    return F;
  }

  template<std::size_t DIR>
  KOKKOS_FUNCTION
  static StateType RoeApproxSolver(const GasType& gas,
                                   const StateType& statesL,
                                   const StateType& statesR) {
    EigenVectorsType L;
    EigenVectorsType R;
    EigenvaluesType Lambda;
    EulerRoeEigenSystem<DIR>(gas, statesL, statesR, L, R, Lambda);
    const auto FL = Flux<DIR>(gas, statesL);
    const auto FR = Flux<DIR>(gas, statesR);

    ScalarType diff[DIM + NSCAL + 2], wave[DIM + NSCAL + 2];
    for (std::size_t k = 0; k < DIM + NSCAL + 2; ++k) {
      diff[k] = statesR(k) - statesL(k);
    }
    for (std::size_t i = 0; i < DIM + NSCAL + 2; ++i) {
      wave[i] = 0.0;
      for (std::size_t j = 0; j < DIM + NSCAL + 2; ++j) {
        wave[i] += L(i, j) * diff[j];
      }
    }

    for (std::size_t i = 0; i < DIM + NSCAL + 2; ++i) {
      wave[i] *= std::abs(Lambda[i]);
    }

    ScalarType dissp[DIM + NSCAL + 2];
    for (std::size_t i = 0; i < DIM + NSCAL + 2; ++i) {
      dissp[i] = 0.0;
      for (std::size_t j = 0; j < DIM + NSCAL + 2; ++j) {
        dissp[i] += R(i, j) * wave[j];
      }
    }

    StateType F;
    for (std::size_t i = 0; i < DIM + NSCAL + 2; ++i) {
      F(i) = 0.5 * (FL(i) + FR(i)) - 0.5 * dissp[i];
    }
    return F;

  }

  template<bool offset>
  KOKKOS_FUNCTION
  static ScalarType WENO5(const ScalarType* u) {

    static constexpr ScalarType wenojs5_c[3][3]
        = {{2.0/6.0, -7.0/6.0, 11.0/6.0},
           {-1.0/6.0, 5.0/6.0, 2.0/6.0},
           {2.0/6.0, 5.0/6.0, -1.0/6.0}};
    static constexpr ScalarType wenojs5_d[3] = {1.0/10.0, 6.0/10.0, 3.0/10.0};

    ScalarType s[3];
    ScalarType q[3];

    // smoothness indicator
    if constexpr (!offset) {
      // candidates
      for(std::size_t i = 0; i < 3; ++i) {
        q[i] = 0;
        for(std::size_t j = 0; j < 3; ++j) {
          q[i] += wenojs5_c[i][j] * u[j+i];
        }
      }
      // smoothness
      s[0] = (13.0 / 12.0) * std::pow(u[0] - 2.0 * u[1] + u[2], 2) + (1.0/4.0) * std::pow(u[0] - 4.0 * u[1] + 3.0 * u[2], 2);
      s[1] = (13.0 / 12.0) * std::pow(u[1] - 2.0 * u[2] + u[3], 2) + (1.0/4.0) * std::pow(u[1] - u[3], 2);
      s[2] = (13.0 / 12.0) * std::pow(u[2] - 2.0 * u[3] + u[4], 2) + (1.0/4.0) * std::pow(3.0 * u[2] - 4.0 * u[3] + u[4], 2);
    } else {
      // candidates
      for(std::size_t i = 0; i < 3; ++i) {
        q[i] = 0;
        for(std::size_t j = 0; j < 3; ++j) {
          q[i] += wenojs5_c[i][j] * u[4-(j+i)];
        }
      }
      // smoothness
      s[0] = (13.0 / 12.0) * std::pow(u[4] - 2.0 * u[3] + u[2], 2) + (1.0/4.0) * std::pow(u[4] - 4.0 * u[3] + 3.0 * u[2], 2);
      s[1] = (13.0 / 12.0) * std::pow(u[3] - 2.0 * u[2] + u[1], 2) + (1.0/4.0) * std::pow(u[3] - u[1], 2);
      s[2] = (13.0 / 12.0) * std::pow(u[2] - 2.0 * u[1] + u[0], 2) + (1.0/4.0) * std::pow(3.0 * u[2] - 4.0 * u[1] + u[0], 2);
    }

    constexpr ScalarType eps = 1e-6;
    ScalarType w[3];
    for(std::size_t k = 0; k < 3; ++k) {
      w[k] = wenojs5_d[k] / std::pow(eps + s[k], 2);
    }

    ScalarType out = 0.0;
    for(std::size_t i = 0; i < 3; ++i) {
      out += w[i] * q[i];
    }
    out /= (w[0] + w[1] + w[2]);

    return out;
  }



  template<std::size_t DIR>
  struct flux_kernel {
    StateArrayType states;
    FluxArrayType flux;
    Kokkos::Array<SizeType, DIM> NG;
    GasType gas;


    template<typename... Args>
    KOKKOS_FUNCTION
    void operator()(const Args... indices) const {
      static_assert(sizeof...(indices) == DIM, "MDRangePolicy must be used for DIM > 1!");
      static_assert(std::conjunction_v<std::is_integral<Args>...>);
      const Kokkos::Array<SizeType, DIM> index{indices...};

      // stencils slicing
      Kokkos::Array<SizeType, DIR> indices_pre;
      Kokkos::Array<SizeType, DIM-DIR-1> indices_post;
      Kokkos::pair<SizeType, SizeType> range;
      for(std::size_t d = 0; d < DIM; ++d) {
        if(d < DIR) indices_pre[d] = index[d] + NG[d];
        else if(d == DIR) range = Kokkos::make_pair(index[d] + NG[d] - 3, index[d] + NG[d] + 3); // 6 stencils
        else indices_post[d-DIR-1] = index[d] + NG[d];
      }
      flux(indices...) = computeFlux(slice_(states, indices_pre, range, indices_post));
    }


    template<typename Kokkos1DStateViewType>
    KOKKOS_FUNCTION
    StateType computeFlux(const Kokkos1DStateViewType stencils) const {
      // stencils 0,1,2,3,4,5
      static_assert(std::is_same_v<typename Kokkos1DStateViewType::value_type, StateType>);
      EigenVectorsType L, R;
      EigenvaluesType Lambda;
      EulerRoeEigenSystem<DIR>(gas, stencils(2), stencils(3), L, R, Lambda);

      ScalarType ss[DIM+NSCAL+2][6]; // stencils in characteristic coord
      for(std::size_t i = 0; i < 6; ++i) { // all stencils
        for(std::size_t j = 0; j < DIM+NSCAL+2; ++j) { // all waves
          ss[j][i] = 0.0;
          for(std::size_t k = 0; k < DIM+NSCAL+2; ++k) {
            ss[j][i] += L(j, k) * stencils(i)(k);
          }
        }
      }

      ScalarType wL[5], wR[5];
      for(std::size_t j = 0; j < DIM+NSCAL+2; ++j) {
        wL[j] = WENO5<false>(&ss[j][0]);
        wR[j] = WENO5<true>(&ss[j][1]);
      }

      StateType stateL, stateR;
      for(std::size_t j = 0; j < DIM+NSCAL+2; ++j) {
        stateL(j) = 0.0;
        stateR(j) = 0.0;
        for(std::size_t k = 0; k < DIM+NSCAL+2; ++k) {
          stateL(j) += R(j, k) * wL[k];
          stateR(j) += R(j, k) * wR[k];
        }
      }

      return RoeApproxSolver<DIR>(gas, stateL, stateR);

    }
  };

  template<std::size_t DIR>
  struct accum_kernel {
    MeshType mesh;
    FluxArrayType flux;
    StateArrayType rhs;

    template<typename... Args>
    KOKKOS_FUNCTION
    void operator()(const Args... indices) const {
      static_assert(sizeof...(indices) == DIM, "MDRangePolicy must be used for DIM > 1!");
      static_assert(std::conjunction_v<std::is_integral<Args>...>);

      Kokkos::Array<SizeType, DIM> right_index{indices...};
      right_index[DIR]++;

      rhs(indices...).U = rhs(indices...).U +
          (get_from_index_array(flux, right_index).U - flux(indices...).U) / mesh.dx[DIR];
    }

  };

  void operator()(const MeshType& mesh,
                  const Kokkos::Array<SizeType, DIM>& NG,
                  const StateArrayType& states,
                  const StateArrayType& rhs) const {

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      (computeFluxes<Is>(mesh, NG, states), ...);
      (accumulate<Is>(mesh, rhs), ...);
    }(std::make_index_sequence<DIM>());

  }

  template<std::size_t DIR>
  void computeFluxes(const MeshType& mesh,
                     const Kokkos::Array<SizeType, DIM>& NG,
                     const StateArrayType& states) const {
    flux_kernel<DIR> functor {.states = states,
                              .flux = m_fluxes[DIR],
                              .NG = NG,
                              .gas = m_gas};
    Kokkos::Array<SizeType, DIM> lb{0};
    Kokkos::Array<SizeType, DIM> ub = mesh.N; ub[DIR] = ub[DIR] + 1;
    const std::string label = std::string("computeFluxes-Face-") + std::to_string(DIR);
    if constexpr (DIM == 1) {
      const auto policy =
          Kokkos::RangePolicy<ExecutionSpace>(lb[0], ub[0]);
      Kokkos::parallel_for(label, policy, functor);
    } else {
      const auto policy =
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<DIM>>(lb, ub);
      Kokkos::parallel_for(label, policy, functor);
    }
  }

  template<std::size_t DIR>
  void accumulate(const MeshType& mesh,
                  const StateArrayType& rhs) const {
    accum_kernel<DIR> functor {.mesh=mesh,
                               .flux=m_fluxes[DIR],
                               .rhs=rhs };
    const Kokkos::Array<SizeType, DIM> lb{0};
    const Kokkos::Array<SizeType, DIM> ub = mesh.N;
    const std::string label = std::string("accumulate-Face-") + std::to_string(DIR);
    if constexpr (DIM == 1) {
      const auto policy =
          Kokkos::RangePolicy<ExecutionSpace>(lb[0], ub[0]);
      Kokkos::parallel_for(label, policy, functor);
    } else {
      const auto policy =
          Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<DIM>>(lb, ub);
      Kokkos::parallel_for(label, policy, functor);
    }

  }

private:
  GasType m_gas;
  Kokkos::Array<FluxArrayType, DIM> m_fluxes;
};


#endif //TOYDNS_SRC_ADVECTION_H
