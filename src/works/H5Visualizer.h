//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_H5VISUALIZER_H
#define TOYDNS_SRC_H5VISUALIZER_H

#include <string>
#include <filesystem>
#include <ranges>
#include <numeric>
#include <array>
#include <utility>
#include <vector>
#include "hdf5.h"
#include "H5Cpp.h"
#include "Kokkos_Core.hpp"

#include "cores/concepts/MeshConcept.h"
#include "cores/concepts/StateConcept.h"

template <typename T>
struct H5DataType { };

template <>
struct H5DataType<int> {
  static const inline H5::DataType value = H5::PredType::NATIVE_INT;
};

template<>
struct H5DataType<unsigned long> {
  static const inline H5::DataType value = H5::PredType::NATIVE_ULONG;
};

template <>
struct H5DataType<float> {
  static const inline H5::DataType value = H5::PredType::NATIVE_FLOAT;
};

template <>
struct H5DataType<double> {
  static const inline H5::DataType value = H5::PredType::NATIVE_DOUBLE;
};

template<typename T>
static const inline H5::DataType H5DataType_v = H5DataType<T>::value;


template<std::size_t DIM_,
         typename StateArrayType_,
         concepts::Mesh MeshType_,
         std::floating_point ScalarType_ = double>
    requires concepts::State<typename StateArrayType_::value_type>
class H5Visualizer {
public:
  static constexpr std::size_t DIM = DIM_;
  using StateArrayType = StateArrayType_;
  using MeshType = MeshType_;
  using ScalarType = ScalarType_;
  using StateType = typename StateArrayType_::value_type;

  H5Visualizer(std::filesystem::path  _directory,
               std::string  _prefix,
               std::string  _postfix = ".h5"):
               directory(std::move(_directory)),
               prefix(std::move(_prefix)),
               postfix(std::move(_postfix)) {
    std::filesystem::create_directory(directory);
  };

  void operator()(const std::size_t step, const ScalarType t,
                  const MeshType& mesh,
                  const StateArrayType& states) const {

    const std::filesystem::path filename = prefix + "-" + std::to_string(step) + postfix;
    const std::filesystem::path fpath = directory / filename;

    const auto Ns = mesh.N;
    std::size_t npos = 1;
    for(std::size_t d = 0; d < DIM; ++d) {
      npos *= Ns[d];
    }

    constexpr std::size_t num_fields =  DIM + StateType::NSCAL + 2;
    //const auto field_labels = StateType::labels;
    const std::size_t numData = npos * num_fields;

    const auto h_State = Kokkos::create_mirror_view(states);
    Kokkos::deep_copy(h_State, states);

    std::vector<ScalarType> buffer(numData);

    const auto copy_to_buf = [&h_State, &buffer, &Ns, &npos]<typename... Args>(const Args... indices) {
      static_assert(sizeof...(indices) == DIM, "MDRangePolicy must be used for DIM > 1!");
      static_assert(std::conjunction_v<std::is_integral<Args>...>);
      const SizeType ind = sub2ind(Ns, indices...);
      auto state = h_State(indices...);
      for(std::size_t k = 0; k < num_fields; ++k) {
        buffer[k * npos + ind] = state(k);
      }
    };

    const std::string label = "CopyToBuf";
    if constexpr (DIM == 1) {
      const auto policy
        = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, Ns[0]);
      Kokkos::parallel_for(label, policy, copy_to_buf);
    } else {
      const auto policy
        = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<DIM>>(Kokkos::Array<SizeType, DIM>{0},
                                                                                      Ns);
      Kokkos::parallel_for(label, policy, copy_to_buf);
    }

    H5::H5File h5file(fpath, H5F_ACC_TRUNC);
    Kokkos::Array<hsize_t, DIM+1> dims;
    dims[0] = num_fields;
    for(std::size_t d = 0; d < DIM; ++d) {
      dims[1+d] = Ns[DIM-1-d];
    }
    H5::DataSpace data_space(DIM+1, dims.data());

    auto data_set = h5file.createDataSet("Solution",
                                         H5DataType_v<ScalarType>,
                                         data_space);
    data_set.write(buffer.data(), H5DataType_v<ScalarType>);

    H5::Attribute att_t = data_set.createAttribute("Time", H5DataType_v<ScalarType>, H5S_SCALAR);
    att_t.write(H5DataType_v<ScalarType>, &t);

    H5::Attribute att_s = data_set.createAttribute("Step", H5DataType_v<std::size_t>, H5S_SCALAR);
    att_s.write(H5DataType_v<std::size_t>, &step);

    h5file.close();
  }
private:
  const std::filesystem::path directory;
  const std::string prefix;
  const std::string postfix;
};

#endif //TOYDNS_SRC_H5VISUALIZER_H
