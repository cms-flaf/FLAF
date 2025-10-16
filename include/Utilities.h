#pragma once

#include <ROOT/RVec.hxx>
#include <any>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <variant>
#include <ROOT/RVec.hxx>

using RVecF = ROOT::VecOps::RVec<float>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecUC = ROOT::VecOps::RVec<unsigned char>;
using RVecUL = ROOT::VecOps::RVec<unsigned long>;
using RVecULL = ROOT::VecOps::RVec<unsigned long long>;

namespace detail {

    template <typename T>
    struct DeltaImpl {
        static T Delta(const T& shifted, const T& central) { return shifted - central; }
        static T FromDelta(const T& delta, const T& central) { return delta + central; }
    };

    template <>
    struct DeltaImpl<bool> {
        static bool Delta(const bool& shifted, const bool& central) { return shifted == central; }
        static bool FromDelta(const bool& delta, const bool& central) { return delta ? central : !central; }
    };

    template <typename T>
    struct DeltaImpl<ROOT::VecOps::RVec<T>> {
        static ROOT::VecOps::RVec<T> Delta(const ROOT::VecOps::RVec<T>& shifted, const ROOT::VecOps::RVec<T>& central) {
            ROOT::VecOps::RVec<T> delta = shifted;
            const size_t n_max = std::min(shifted.size(), central.size());
            for (size_t n = 0; n < n_max; ++n)
                delta[n] = DeltaImpl<T>::Delta(shifted[n], central[n]);
            return delta;
        }

        static ROOT::VecOps::RVec<T> FromDelta(const ROOT::VecOps::RVec<T>& delta,
                                               const ROOT::VecOps::RVec<T>& central) {
            ROOT::VecOps::RVec<T> fromDeltaVec = delta;
            const size_t n_max = std::min(delta.size(), central.size());
            for (size_t n = 0; n < n_max; ++n)
                fromDeltaVec[n] = DeltaImpl<T>::FromDelta(delta[n], central[n]);
            return fromDeltaVec;
        }
    };

    template <typename T>
    struct IsSameImpl {
        static bool IsSame(const T& shifted, const T& central) { return shifted == central; }
    };

    template <typename T>
    struct IsSameImpl<ROOT::VecOps::RVec<T>> {
        static bool IsSame(const ROOT::VecOps::RVec<T>& shifted, const ROOT::VecOps::RVec<T>& central) {
            const size_t n = shifted.size();
            if (n != central.size())
                return false;
            for (size_t i = 0; i < n; ++i)
                if (!IsSameImpl<T>::IsSame(shifted[i], central[i]))
                    return false;
            return true;
        }
    };

}  // namespace detail

namespace analysis {

    template <typename T>
    T Delta(const T& shifted, const T& central) {
        return ::detail::DeltaImpl<T>::Delta(shifted, central);
    }

    template <typename T>
    T FromDelta(const T& delta, const T& central) {
        return ::detail::DeltaImpl<T>::FromDelta(delta, central);
    }

    template <typename T>
    bool IsSame(const T& shifted, const T& central) {
        return ::detail::IsSameImpl<T>::IsSame(shifted, central);
    }

}  // namespace analysis
