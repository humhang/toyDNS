//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_CONCEPTS_VISUALIZERCONCEPT_H
#define TOYDNS_SRC_CONCEPTS_VISUALIZERCONCEPT_H

#include <concepts>

namespace concepts {
template<typename V, std::size_t DIM, typename StateArrayType, typename MeshType, typename ScalarType = double>
concept Visualizer = requires(V v, std::size_t step, ScalarType time, MeshType mesh, StateArrayType states) {
  {v(step, time, mesh, states)};
};
}

#endif //TOYDNS_SRC_CONCEPTS_VISUALIZERCONCEPT_H
