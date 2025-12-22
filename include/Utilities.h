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
    // based on https://github.com/cms-sw/cmssw/blob/master/DataFormats/Math/interface/libminifloat.h
    template <typename FloatType, typename UIntType, int mantissaSize, int bits>
    struct ReduceMantissaToNbitsWithRoundingImpl {
        static FloatType Reduce(FloatType x) {
            static_assert(bits >= 0 && bits <= mantissaSize - 1, "max mantissa size exceeded");

            static constexpr UIntType lowMask = (static_cast<UIntType>(1) << mantissaSize) - 1; // mask to keep lowest mantissaSize bits = mantissa
            static constexpr UIntType highMask = ~lowMask; // mask to keep highest bits = the rest
            static constexpr int shift = mantissaSize - bits;
            static constexpr UIntType mask = (static_cast<UIntType>(-1) >> shift) << shift;
            static constexpr UIntType test = static_cast<UIntType>(1) << (shift - 1);
            static constexpr UIntType maxn = (static_cast<UIntType>(1) << bits) - 2;

            UIntType i = std::bit_cast<UIntType>(x);
            if (i & test) {  // need to round
                UIntType mantissa = (i & lowMask) >> shift;
                if (mantissa < maxn)
                    mantissa++;
                i = (i & highMask) | (mantissa << shift);
            } else {
                i &= mask;
            }
            return std::bit_cast<FloatType>(i);
        }
    };

    template <typename FloatType, typename UIntType, int mantissaSize>
    struct ReduceMantissaToNbitsWithRoundingImpl<FloatType, UIntType, mantissaSize, mantissaSize> {
        static FloatType Reduce(FloatType x) { return x; }
    };

    template <typename T, int bits>
    struct ReduceMantissaToNbitsWithRounding { };

    template <int bits>
    struct ReduceMantissaToNbitsWithRounding<float, bits> {
        static constexpr int mantissaSize = 23;
        static float Reduce(float x) {
            return ReduceMantissaToNbitsWithRoundingImpl<float, uint32_t, mantissaSize, bits>::Reduce(x);
        }
    };

    template <int bits>
    struct ReduceMantissaToNbitsWithRounding<double, bits> {
        static constexpr int mantissaSize = 52;
        static double Reduce(double x) {
            return ReduceMantissaToNbitsWithRoundingImpl<double, uint64_t, mantissaSize, bits>::Reduce(x);
        }
    };

    template <typename T, int... ints>
    std::array<T (*)(T), sizeof...(ints)> make_function_array(std::integer_sequence<int, ints...>) {
        return { &ReduceMantissaToNbitsWithRounding<T, ints>::Reduce... };
    }

    template <typename T>
    struct DeltaImpl {
        static T Delta(const T& shifted, const T& central) { return shifted - central; }
        static T FromDelta(const T& delta, const T& central) { return delta + central; }
    };

    template <>
    struct DeltaImpl<bool> {
        static bool Delta(bool shifted, bool central) { return shifted == central; }
        static bool FromDelta(bool delta, bool central) { return delta ? central : !central; }
    };

    template <>
    struct DeltaImpl<float> {
        static float Delta(float shifted, float central) {
            static constexpr int n_bits = 8;
            if(std::isnormal(central)) {
                const float delta = shifted - central;
                return ReduceMantissaToNbitsWithRounding<float, n_bits>::Reduce(delta);
            }
            return shifted;
        }

        static float FromDelta(float delta, float central) {
            return std::isnormal(central) ? delta + central : delta;
        }
    };

    template <>
    struct DeltaImpl<double> {
        static double Delta(double shifted, double central) {
            static constexpr int n_bits = 8;
            if(std::isnormal(central)) {
                const float delta = shifted - central;
                return ReduceMantissaToNbitsWithRounding<double, n_bits>::Reduce(delta);
            }
            return shifted;
        }

        static double FromDelta(double delta, double central) {
            return std::isnormal(central) ? delta + central : delta;
        }
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
    T Delta(const T& shifted, const T& central, bool shifted_valid, bool central_valid) {
        return shifted_valid && central_valid ? Delta(shifted, central) : shifted;
    }

    template <typename T>
    T FromDelta(const T& delta, const T& central) {
        return ::detail::DeltaImpl<T>::FromDelta(delta, central);
    }

    template <typename T>
    T FromDelta(const T& delta, const T& central, bool shifted_valid, bool central_valid) {
        return shifted_valid && central_valid ? FromDelta(delta, central) : delta;
    }

    template <typename T>
    bool IsSame(const T& shifted, const T& central) {
        return ::detail::IsSameImpl<T>::IsSame(shifted, central);
    }

    template <typename T>
    T ReduceMantissaToNbitsWithRounding(T x, int bits) {
        static constexpr int mantissaSize = ::detail::ReduceMantissaToNbitsWithRounding<T, 0>::mantissaSize;
        static const std::array<T (*)(T), mantissaSize> reducers = ::detail::make_function_array<T>(
            std::make_integer_sequence<int, mantissaSize>{});

        assert(bits >= 0 && bits <= mantissaSize);
        return reducers[bits](x);
    }

}  // namespace analysis
