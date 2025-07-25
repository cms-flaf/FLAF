#pragma once

#include <any>
#include <iostream>
#include <tuple>
#include <thread>
#include <string>
#include <variant>
#include <typeinfo>
#include <typeindex>
#include <ROOT/RVec.hxx>


using RVecF = ROOT::VecOps::RVec<float>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecUC = ROOT::VecOps::RVec<unsigned char>;
using RVecUL = ROOT::VecOps::RVec<unsigned long>;
using RVecULL = ROOT::VecOps::RVec<unsigned long long>;

namespace analysis {


  // Prima definizione GENERICA
  template<typename T>
  T Delta(const T& shifted, const T& central);

  template<typename T>
  T FromDelta(const T& delta, const T& central);

  template<typename T>
  bool IsSame(const T& shifted, const T& central);

  // Specializzazioni PRIMA di tutto ciò che le usa
  template<>
  inline bool Delta<bool>(const bool& shifted, const bool& central) {
    return shifted == central;
  }

  template<>
  inline bool FromDelta<bool>(const bool& delta, const bool& central) {
    return delta ? central : !central;
  }


} // namespace analysis

namespace detail {

  template<typename T>
  struct DeltaImpl {
    static T Delta(const T& shifted, const T& central) {
      return shifted - central;
    }
    static T FromDelta(const T& delta, const T& central) {
      return delta + central;
    }
  };

  template<typename T>
  struct DeltaImpl<ROOT::VecOps::RVec<T>> {
    static ROOT::VecOps::RVec<T> Delta(const ROOT::VecOps::RVec<T>& shifted, const ROOT::VecOps::RVec<T>& central) {
      ROOT::VecOps::RVec<T> delta = shifted;
      size_t n_max = std::min(shifted.size(), central.size());
      for (size_t n = 0; n < n_max; ++n)
        delta[n] -= central[n];
      return delta;
    }

    static ROOT::VecOps::RVec<T> FromDelta(const ROOT::VecOps::RVec<T>& delta, const ROOT::VecOps::RVec<T>& central) {
      ROOT::VecOps::RVec<T> fromDeltaVec = delta;
      size_t n_max = std::min(delta.size(), central.size());
      for (size_t n = 0; n < n_max; ++n)
        fromDeltaVec[n] += central[n];
      return fromDeltaVec;
    }
  };

  template<>
  struct DeltaImpl<ROOT::VecOps::RVec<bool>> {
    static ROOT::VecOps::RVec<bool> Delta(const ROOT::VecOps::RVec<bool>& shifted, const ROOT::VecOps::RVec<bool>& central) {
      ROOT::VecOps::RVec<bool> delta = shifted;
      size_t n_max = std::min(shifted.size(), central.size());
      for (size_t n = 0; n < n_max; ++n)
        delta[n] = analysis::Delta(shifted[n], central[n]);
      return delta;
    }

    static ROOT::VecOps::RVec<bool> FromDelta(const ROOT::VecOps::RVec<bool>& delta, const ROOT::VecOps::RVec<bool>& central) {
      ROOT::VecOps::RVec<bool> fromDeltaVec = delta;
      size_t n_max = std::min(delta.size(), central.size());
      for (size_t n = 0; n < n_max; ++n)
        fromDeltaVec[n] = analysis::FromDelta(delta[n], central[n]);
      return fromDeltaVec;
    }
  };

  template<typename T>
  struct IsSameImpl {
    static bool IsSame(const T& shifted, const T& central) {
      return shifted == central;
    }
  };

  template<typename T>
  struct IsSameImpl<ROOT::VecOps::RVec<T>> {
    static bool IsSame(const ROOT::VecOps::RVec<T>& shifted, const ROOT::VecOps::RVec<T>& central) {
      const size_t n_shifted = shifted.size();
      if (n_shifted != central.size())
        return false;
      for (size_t n = 0; n < n_shifted; ++n)
        if (!IsSameImpl<T>::IsSame(shifted[n], central[n]))
          return false;
      return true;
    }
  };

} // namespace detail

// Ora puoi definire le implementazioni generiche
namespace analysis {

  template<typename T>
  T Delta(const T& shifted, const T& central) {
    return detail::DeltaImpl<T>::Delta(shifted, central);
  }

  template<typename T>
  T FromDelta(const T& delta, const T& central) {
    return detail::DeltaImpl<T>::FromDelta(delta, central);
  }

  template<typename T>
  bool IsSame(const T& shifted, const T& central) {
    return detail::IsSameImpl<T>::IsSame(shifted, central);
  }

} // namespace analysis