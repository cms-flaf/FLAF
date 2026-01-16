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
#include <boost/math/special_functions/erf.hpp>

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

            static constexpr UIntType lowMask =
                (static_cast<UIntType>(1) << mantissaSize) - 1;  // mask to keep lowest mantissaSize bits = mantissa
            static constexpr UIntType highMask = ~lowMask;       // mask to keep highest bits = the rest
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
    struct ReduceMantissaToNbitsWithRounding {};

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
        return {&ReduceMantissaToNbitsWithRounding<T, ints>::Reduce...};
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
            if (std::isnormal(central)) {
                const float delta = shifted - central;
                return ReduceMantissaToNbitsWithRounding<float, n_bits>::Reduce(delta);
            }
            return shifted;
        }

        static float FromDelta(float delta, float central) { return std::isnormal(central) ? delta + central : delta; }
    };

    template <>
    struct DeltaImpl<double> {
        static double Delta(double shifted, double central) {
            static constexpr int n_bits = 8;
            if (std::isnormal(central)) {
                const double delta = shifted - central;
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
        static const std::array<T (*)(T), mantissaSize> reducers =
            ::detail::make_function_array<T>(std::make_integer_sequence<int, mantissaSize>{});

        assert(bits >= 0 && bits <= mantissaSize);
        return reducers[bits](x);
    }



    struct CrystalBall{
        // static constexpr double pi = std::numbers::pi;
        // static constexpr double sqrtPiOver2 = sqrt(pi/2.0);
        // static constexpr double sqrt2 = std::numbers::sqrt2;
        double pi=3.14159;
        double sqrtPiOver2=sqrt(pi/2.0);
        double sqrt2=sqrt(2.0);

        double m;
        double s;
        double a;
        double n;
        double B;
        double C;
        double D;
        double N;
        double NA;
        double Ns;
        double NC;
        double F;
        double G;
        double k;
        double cdfMa;
        double cdfPa;
        CrystalBall():m(0),s(1),a(10),n(10){
            init();
        }
        CrystalBall(double mean, double sigma, double alpha, double n)
            :m(mean),s(sigma),a(alpha),n(n){
            init();
        }
        void init(){
            double fa = fabs(a);
            double ex = exp(-fa*fa/2);
            double A  = pow(n/fa, n) * ex;
            double C1 = n/fa/(n-1) * ex;
            double D1 = 2 * sqrtPiOver2 * erf(fa/sqrt2);
            B = n/fa-fa;
            C = (D1+2*C1)/C1;
            D = (D1+2*C1)/2;
            N = 1.0/s/(D1+2*C1);
            k = 1.0/(n-1);
            NA = N*A;
            Ns = N*s;
            NC = Ns*C1;
            F = 1-fa*fa/n;
            G = s*n/fa;
            cdfMa = cdf(m-a*s);
            cdfPa = cdf(m+a*s);
        }
        double pdf(double x) const{
            double d=(x-m)/s;
            if(d<-a) return NA*pow(B-d, -n);
            if(d>a) return NA*pow(B+d, -n);
            return N*exp(-d*d/2);
        }
        double pdf(double x, double ks, double dm) const{
            double d=(x-m-dm)/(s*ks);
            if(d<-a) return NA/ks*pow(B-d, -n);
            if(d>a) return NA/ks*pow(B+d, -n);
            return N/ks*exp(-d*d/2);

        }
        double cdf(double x) const{
            double d = (x-m)/s;
            if(d<-a) return NC / pow(F-s*d/G, n-1);
            if(d>a) return NC * (C - pow(F+s*d/G, 1-n) );
            return Ns * (D - sqrtPiOver2 * erf(-d/sqrt2));
        }
        double invcdf(double u) const{
            if(u<cdfMa) return m + G*(F - pow(NC/u, k));
            if(u>cdfPa) return m - G*(F - pow(C-u/NC, -k) );
            return m - sqrt2 * s * boost::math::erf_inv((D - u/Ns )/sqrtPiOver2);
        }
    };

    class SeedSequence {
    public:
        explicit SeedSequence(std::initializer_list<uint32_t> seeds)
            : m_seeds(seeds) {}

        template <typename Iter>
        void generate(Iter begin, Iter end) const {
            const size_t n = std::distance(begin, end);
        if (n == 0) return;

        const uint32_t mult = 0x9e3779b9;
        const uint32_t mix_const = 0x85ebca6b;

        std::vector<uint32_t> buffer(n, 0x8b8b8b8b);

        size_t s = m_seeds.size();
        size_t t = (n >= s) ? n - s : 0;

            size_t i = 0;

        for(; i < std::min(n, s); ++i) {
                buffer[i] = buffer[i] ^ (m_seeds[i] + mult * i);
        }
        for(; i < n; ++i) {
                buffer[i] = buffer[i] ^ (mult * i);
            }

        for (size_t k = 0; k < n; ++k) {
                uint32_t z = buffer[(k + n - 1) % n] ^ (buffer[k] >> 27);
            buffer[k] = (z * mix_const) ^ (buffer[k] << 13);
            }

        std::copy(buffer.begin(), buffer.end(), begin);
        }

    private:
        std::vector<uint32_t> m_seeds;
    };

 } // namespace analysis