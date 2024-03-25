//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_CONCEPTS_INITIALIZERCONCEPT_H
#define TOYDNS_SRC_CONCEPTS_INITIALIZERCONCEPT_H

#include <concepts>
#include <type_traits>
#include <array>

namespace concepts {
/*
 * initializer concept.
 */
template<typename V, std::size_t DIM, typename StateArrayType, typename MeshType, typename ScalarType = double>
concept Initializer = requires(V v, MeshType mesh, StateArrayType states) {
  {v(mesh, states)};
};

}
#endif //TOYDNS_SRC_CONCEPTS_INITIALIZERCONCEPT_H
