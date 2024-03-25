//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_CONCEPTS_MESHCONCEPT_H
#define TOYDNS_SRC_CONCEPTS_MESHCONCEPT_H

#include <concepts>
#include <type_traits>

namespace concepts {
template<typename T>
concept Mesh = requires(T mesh) {
  requires std::is_trivially_copyable_v<T>;
  requires std::is_trivially_copy_constructible_v<T>;
  requires std::is_standard_layout_v<T>;
  requires std::same_as<std::decay_t<decltype(T::DIM)>, std::size_t>;
};
}

#endif //TOYDNS_SRC_CONCEPTS_MESHCONCEPT_H
