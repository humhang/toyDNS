#include <iostream>
#include <chrono>

#include "Kokkos_Core.hpp"

#include "examples/ShockTube.h"
#include "examples/Explosion.h"


int main(int argc, char* argv[]) {

  std::cout << "ExecutionSpace:" << ExecutionSpace::name() << std::endl;
  std::cout << "MemorySpace:" << MemorySpace::name() << std::endl;

  Kokkos::initialize(argc, argv);

  {
    auto t1 = std::chrono::high_resolution_clock::now();
    shock_tube(); // 1D
    auto t2 = std::chrono::high_resolution_clock::now();
    auto wtime = std::chrono::duration_cast<std::chrono::duration<float> >(t2 - t1).count();
    std::cout << "Time lapsed = " << wtime << std::endl;
  }

  {
    //auto t1 = std::chrono::high_resolution_clock::now();
    //explosion(); // 2D
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto wtime = std::chrono::duration_cast<std::chrono::duration<float> >(t2 - t1).count();
    //std::cout << "Time lapsed = " << wtime << std::endl;
  }


  Kokkos::finalize();
  return 0;
}
