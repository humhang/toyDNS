//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_EXAMPLES_EXPLOSION_H
#define TOYDNS_SRC_EXAMPLES_EXPLOSION_H

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


int explosion() {
  /*
   * type alias
   */
  constexpr std::size_t DIM = 2;
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
  const MeshType mesh({0.0, 0.0}, {2.0, 2.0}, {400, 400});
  const SolverType solver(mesh);

  /*
   * initializer
   */
  const auto initfcn = KOKKOS_LAMBDA(const Kokkos::Array<double, DIM> &pos) {
    constexpr ScalarType rho_in = 1.0, rho_out = 0.125;
    constexpr ScalarType p_in = 1.0, p_out = 0.1;
    constexpr ScalarType radius = 0.4;

    ScalarType r = 0;
    for(std::size_t d = 0; d < DIM; ++d) r += std::pow(pos[d] - 1.0, 2);
    r = std::sqrt(r);
    const bool in = r < radius;

    StateType state;
    state.rho() = in ? rho_in : rho_out;
    for(std::size_t d = 0; d < DIM; ++d) state.mu(d) = 0.0;
    state.E() = in ? p_in / (gas.gamma-1) : p_out / (gas.gamma-1);
    for(std::size_t n = 0; n < NSCAL; ++n) state.rhoY(n) = 0.0;

    return state;
  };
  using InitializerType = Initializer<decltype(initfcn), DIM,  StateArrayType, MeshType, ScalarType>;
  InitializerType initializer(initfcn);

  /*
   * boundary filler
   */
  using BoundaryFillerType = Boundary<DIM, StateArrayType, MeshType, ScalarType>;
  BoundaryFillerType boundary(BoundaryFillerType::BoundaryType::Periodic);

  /*
   * advection
   */
  using AdvectionType = Advection<DIM, GasType, StateArrayType, MeshType, ScalarType>;
  AdvectionType advection(gas, mesh);

  /*
   * integrator
   */
  using IntegratorType = TimeIntegrator<DIM, StateArrayType, MeshType, AdvectionType, BoundaryFillerType, ScalarType>;
  IntegratorType integrator(boundary, advection, Kokkos::Array<SizeType, DIM>{3, 3});

  /*
   * visualizer
   */
  H5Visualizer<DIM, StateArrayType, MeshType, ScalarType> writer("./explosion", "data");

  /*
   * driver
   */
  solver.initialize(initializer);
  constexpr std::size_t max_step = 100000;
  constexpr ScalarType dt = 0.0001;
  for(int s = 0; s < max_step; ++s) {
    solver.advance(dt, integrator);
    if(s % 100 == 0) {
      std::cout << "Step = " << s << std::endl;
      solver.visualize(s, dt*s, writer);
    }
  }
  return 0;

}


#endif //TOYDNS_SRC_EXAMPLES_EXPLOSION_H
