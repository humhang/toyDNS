//
// Created by hang
//

#ifndef TOYDNS_SRC_EXAMPLES_SHOCKTUBE_H
#define TOYDNS_SRC_EXAMPLES_SHOCKTUBE_H

#include "KokkosTypedefs.h"

#include "cores/State.h"
#include "cores/Mesh.h"
#include "cores/IdealGas.h"
#include "cores/Solver.h"
#include "works/H5Visualizer.h"
#include "works/Boundary.h"
#include "works/Initializer.h"
#include "works/TimeIntegrator.h"
#include "works/Advection.h"

int shock_tube() {
  /*
   * type alias
   */
  constexpr std::size_t DIM = 1;
  constexpr std::size_t NSCAL = 0;
  using ScalarType = double;

  using GasType = IdealGas<ScalarType>;
  using StateType = State<DIM, NSCAL, ScalarType>;
  using MeshType = Mesh<DIM, ScalarType>;
  using SolverType = Solver<DIM, StateType, MeshType, ScalarType>;
  using StateArrayType = SolverType::StateArrayType;

  /*
   * instantiate solver
   */
  const GasType gas{.gamma=1.4};
  const MeshType mesh({0.0}, {1.0}, {2048});
  const SolverType solver(mesh);

  /*
   * initializer
   */
  const auto initfcn = KOKKOS_LAMBDA(const Kokkos::Array<double, DIM> &pos) {
    constexpr ScalarType rhoL = 1.0, rhoR = 0.125;
    constexpr ScalarType pL = 1.0, pR = 0.1;

    const bool left = pos[0] < 0.5;

    StateType state;
    state.rho() = left ? rhoL : rhoR;
    for(std::size_t d = 0; d < DIM; ++d) state.mu(d) = 0.0;
    state.E() = left ? pL / (gas.gamma-1) : pR / (gas.gamma-1);
    for(std::size_t n = 0; n < NSCAL; ++n) state.rhoY(n) = 0.0;

    return state;
  };
  using InitializerType = Initializer<decltype(initfcn), DIM,  StateArrayType, MeshType, ScalarType>;
  InitializerType initializer(initfcn);

  /*
   * boundary filler
   */
  using BoundaryFillerType = Boundary<DIM, StateArrayType, MeshType, ScalarType>;
  BoundaryFillerType boundary(BoundaryFillerType::BoundaryType::Wall);

  /*
   * advection
   */
  using AdvectionType = Advection<DIM, GasType, StateArrayType, MeshType, ScalarType>;
  AdvectionType advection(gas, mesh);

  /*
   * integrator
   */
  using IntegratorType = TimeIntegrator<DIM, StateArrayType, MeshType, AdvectionType, BoundaryFillerType, ScalarType>;
  IntegratorType integrator(boundary, advection, Kokkos::Array<SizeType, DIM>{3});

  /*
   * visualizer
   */
  H5Visualizer<DIM, StateArrayType, MeshType, ScalarType> writer("./shocktube", "data");

  /*
   * driver
   */
  solver.initialize(initializer);
  constexpr std::size_t max_step = 4000;
  constexpr ScalarType dt = 0.00005;
  for(int s = 0; s < max_step; ++s) {
    solver.advance(dt, integrator);
    if(s % 100 == 0) {
      std::cout << "Step = " << s << std::endl;
      solver.visualize(s, dt*s, writer);
    }
  }
  return 0;

}

#endif //TOYDNS_SRC_EXAMPLES_SHOCKTUBE_H
