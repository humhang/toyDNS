//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_KOKKOSTYPEDEFS_H
#define TOYDNS_SRC_KOKKOSTYPEDEFS_H
#include "Kokkos_Core.hpp"

// Kokkos type alias
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using SizeType = ExecutionSpace::size_type;
using Device = Kokkos::Device<ExecutionSpace, MemorySpace>;
using Layout = ExecutionSpace::array_layout;

#endif //TOYDNS_SRC_KOKKOSTYPEDEFS_H
