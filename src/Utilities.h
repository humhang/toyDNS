//
// Created by Hang Yu
//

#ifndef TOYDNS_SRC_UTILITIES_H
#define TOYDNS_SRC_UTILITIES_H

#include <type_traits>
#include <array>
#include <utility>


/*
 * add nlevels pointer
 */
template<typename T, std::size_t Levels>
struct add_nlevels_pointer {
  using type = typename add_nlevels_pointer<T, Levels-1>::type*;
};

template<typename T>
struct add_nlevels_pointer<T, 0> {
  using type = typename std::remove_reference_t<T>;
};

template<typename T, std::size_t Levels>
using add_nlevels_pointer_t = typename add_nlevels_pointer<T, Levels>::type;


/*
 * make Kokkos::Array operation easier
 */
template<typename T, typename U, std::size_t N>
KOKKOS_FUNCTION
auto operator+(const Kokkos::Array<T, N>& x, const Kokkos::Array<U, N>& y) {
  using V = decltype(x[0] + y[0]);
  Kokkos::Array<V, N> ret;
  for(std::size_t n = 0; n < N; ++n) {
    ret[n] = x[n] + y[n];
  }
  return ret;
}

template<typename T, typename U, std::size_t N>
KOKKOS_FUNCTION
auto operator-(const Kokkos::Array<T, N>& x, const Kokkos::Array<U, N>& y) {
  using V = decltype(x[0] - y[0]);
  Kokkos::Array<V, N> ret;
  for(std::size_t n = 0; n < N; ++n) {
    ret[n] = x[n] - y[n];
  }
  return ret;
}

template<typename T, typename U, std::size_t N>
KOKKOS_FUNCTION
auto operator*(const Kokkos::Array<T, N>& x, const Kokkos::Array<U, N>& y) {
  using V = decltype(x[0] * y[0]);
  Kokkos::Array<V, N> ret;
  for(std::size_t n = 0; n < N; ++n) {
    ret[n] = x[n] * y[n];
  }
  return ret;
}

template<typename T, typename U, std::size_t N>
KOKKOS_FUNCTION
auto operator/(const Kokkos::Array<T, N>& x, const Kokkos::Array<U, N>& y) {
  using V = decltype(x[0] / y[0]);
  Kokkos::Array<V, N> ret;
  for(std::size_t n = 0; n < N; ++n) {
    ret[n] = x[n] / y[n];
  }
  return ret;
}

template<typename T, typename U, std::size_t N>
KOKKOS_FUNCTION
auto operator*(const T& val, const Kokkos::Array<U, N>& x) {
  using V = decltype(val * x[0]);
  Kokkos::Array<V, N> ret;
  for(std::size_t n = 0; n < N; ++n) {
    ret[n] = val * x[n];
  }
  return ret;
}

template<typename T, typename U, std::size_t N>
KOKKOS_FUNCTION
auto operator/(const Kokkos::Array<U, N>& x, const T& val) {
  using V = decltype(x[0]/val);
  Kokkos::Array<V, N> ret;
  for(std::size_t n = 0; n < N; ++n) {
    ret[n] = x[n]/val;
  }
  return ret;
}

/*
 * apply array as variadic function arguments.
 */
template<typename F, typename T, std::size_t N, std::size_t... Is>
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto) apply_array_impl(F&& f, const Kokkos::Array<T, N>& array, std::index_sequence<Is...>) {
  return std::invoke(std::forward<F>(f), array[Is]...);
}

template<typename F, typename T, std::size_t N>
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto) apply_array(F&& f, const Kokkos::Array<T, N>& array) {
  return apply_array_impl(std::forward<F>(f), array, std::make_index_sequence<N>());
}

template<typename KokkosViewType, typename T, std::size_t N>
struct get_from_index_array_impl {
  const KokkosViewType& view;
  const Kokkos::Array<T, N>& indices;

  template<std::size_t... Is>
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(std::index_sequence<Is...>) {
    return view(indices[Is]...);
  }
};

template<typename KokkosViewType, typename T, std::size_t N>
[[nodiscard]]
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto) get_from_index_array(const KokkosViewType& view, const Kokkos::Array<T, N>& indices) {

  get_from_index_array_impl<KokkosViewType, T, N> impl{.view = view, .indices = indices};
  return impl(std::make_index_sequence<N>());

}

template<typename KokkosViewType, std::size_t DIM>
struct subview_impl {
  const KokkosViewType& view;
  const Kokkos::Array<Kokkos::pair<SizeType, SizeType>, DIM>& range;

  template<std::size_t... Is>
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(std::index_sequence<Is...>) {
    return Kokkos::subview(view, range[Is]...);
  }
};

template<typename KokkosViewType, std::size_t DIM>
[[nodiscard]]
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto) subview_(const KokkosViewType& view,
                                  const Kokkos::Array<Kokkos::pair<SizeType, SizeType>, DIM>& range) {

  subview_impl<KokkosViewType, DIM> impl{.view = view, .range = range};
  return impl(std::make_index_sequence<DIM>());

}

template<typename KokkosViewType, std::size_t PRE, std::size_t POST>
struct slice_impl {
  const KokkosViewType& view;
  const Kokkos::Array<SizeType, PRE>& pre_indices;
  const Kokkos::pair<SizeType, SizeType>& range;
  const Kokkos::Array<SizeType, POST>& post_indices;

  template<std::size_t pos>
  KOKKOS_INLINE_FUNCTION
  constexpr auto getarg(){
    if constexpr (pos < PRE) { return pre_indices[pos]; }
    else if constexpr (pos == PRE) { return range; }
    else { return post_indices[pos-PRE-1]; }
  }

  template<std::size_t... Is>
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(std::index_sequence<Is...>) {
    return Kokkos::subview(view, getarg<Is>()...);
  }

};


template<typename KokkosViewType, std::size_t PRE, std::size_t POST>
[[nodiscard]]
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto) slice_(const KokkosViewType& view,
                                const Kokkos::Array<SizeType, PRE>& pre_indices,
                                const Kokkos::pair<SizeType, SizeType>& range,
                                const Kokkos::Array<SizeType, POST>& post_indices) {

  slice_impl<KokkosViewType, PRE, POST> impl{.view = view,
                                             .pre_indices = pre_indices,
                                             .range = range,
                                             .post_indices = post_indices};

  return impl(std::make_index_sequence<PRE+POST+1>());

}

template<typename ViewType, std::size_t N>
struct make_view_impl {
  const std::string& label;
  const Kokkos::Array<SizeType, N>& extents;

  template<std::size_t... Is>
  decltype(auto) operator()(std::index_sequence<Is...>) {
    return ViewType(label, extents[Is]...);
  }

};

template<typename ViewType, std::size_t N>
[[nodiscard]]
decltype(auto) make_view(const std::string& label,
                         const Kokkos::Array<SizeType, N>& extents) {

  make_view_impl<ViewType, N> impl {.label = label, .extents = extents};

  return impl(std::make_index_sequence<N>());
}

/*
 * sub2ind
 */
template<std::size_t DIM, typename SizeType, typename Index, typename... Tails>
[[nodiscard]]
KOKKOS_INLINE_FUNCTION
constexpr SizeType sub2ind_impl(const Kokkos::Array<SizeType, DIM>& size,
                      const SizeType multiplier,
                      const Index sub,
                      const Tails... tails) {

  constexpr std::size_t d = DIM - sizeof...(tails) - 1;

  if constexpr (sizeof...(tails) != 0) {
    return sub * multiplier + sub2ind_impl(size, multiplier * size[d], tails...);
  } else {
    return sub * multiplier;
  }
}

template<std::size_t DIM, typename SizeType, typename... Indices>
[[nodiscard]]
KOKKOS_INLINE_FUNCTION
constexpr SizeType sub2ind(const Kokkos::Array<SizeType, DIM>& size, const Indices... subs) {
  return sub2ind_impl(size, SizeType(1), subs...);
}

/*
 * small Dense matrix
 */
template<std::size_t M, std::size_t N, typename T>
struct DenseMatrix {

  template<std::size_t, std::size_t, typename>
  friend struct DenseMatrix;

  KOKKOS_INLINE_FUNCTION
  T& operator()(std::size_t i, std::size_t j) noexcept {
    return m_array[i][j];
  }

  KOKKOS_INLINE_FUNCTION
  const T& operator()(std::size_t i, std::size_t j) const noexcept {
    return m_array[i][j];
  }

  KOKKOS_INLINE_FUNCTION
  auto& operator=(const T x) {
    for(std::size_t i = 0; i < M; ++i) {
      for(std::size_t j = 0; j < N; ++j) {
        m_array[i][j] = x;
      }
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  auto& operator*=(const T x) {
    for(std::size_t i = 0; i < M; ++i) {
      for(std::size_t j = 0; j < N; ++j) {
        m_array[i][j] *= x;
      }
    }
    return *this;
  }

private:
  Kokkos::Array<Kokkos::Array<T, N> , M> m_array;
};

#endif //TOYDNS_SRC_UTILITIES_H
