//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_SOLVER_H
#define TOYDNS_SRC_SOLVER_H

#include <concepts>
#include <type_traits>
#include <tuple>

#include "Kokkos_Core.hpp"

#include "Utilities.h"
#include "KokkosTypedefs.h"

#include "concepts/MeshConcept.h"
#include "concepts/StateConcept.h"
#include "concepts/VisualizerConcept.h"
#include "concepts/InitializerConcept.h"

// the actual ND problem solver
template<std::size_t DIM_,
    concepts::State StateType_,
    concepts::Mesh MeshType_,
    std::floating_point ScalarType_ = double>
    requires (DIM_ > 0) &&
             (StateType_::DIM == DIM_) &&
             (MeshType_::DIM == DIM_)
class Solver {
public:
  static constexpr std::size_t DIM = DIM_;
  using StateType = StateType_;
  using MeshType = MeshType_;
  using ScalarType = ScalarType_;
  using StateArrayType = Kokkos::View<add_nlevels_pointer_t<StateType, DIM>, MemorySpace>;
  static_assert(std::is_trivially_copy_constructible_v<StateType>);
  static_assert(std::is_trivially_copy_constructible_v<MeshType>);

  explicit Solver(const MeshType& mesh) :
    m_mesh(mesh) {

    const auto extents = mesh.N;
    m_state = make_view<StateArrayType>("U", extents);
  }


  template<typename InitializerType>
    requires concepts::Initializer<InitializerType, DIM, StateArrayType, MeshType, ScalarType>
  decltype(auto) initialize(const InitializerType& initializer) const {
    return initializer(m_mesh, m_state);
  }

  template<typename VisualizerType>
    requires concepts::Visualizer<VisualizerType, DIM, StateArrayType, MeshType, ScalarType>
  decltype(auto) visualize(
      const std::size_t step,
      const ScalarType t ,
      const VisualizerType& viz) const {
    return viz(step, t, m_mesh, m_state);
  }

  template<typename TimeIntegrator>
  decltype(auto) advance(const ScalarType dt, const TimeIntegrator& integrator) const {
    return integrator(dt, m_mesh, m_state);
  }


private:
  const MeshType m_mesh;
  StateArrayType m_state;
};


#endif //TOYDNS_SRC_SOLVER_H
