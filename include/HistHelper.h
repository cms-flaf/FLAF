#pragma once

#include <any>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <variant>

#include "EntryQueue.h"

/*
namespace kin_fit {
struct FitResults {
  double mass, chi2, probability;
  int convergence;
  bool HasValidMass() const { return convergence > 0; }
  FitResults() : convergence(std::numeric_limits<int>::lowest()) {}
  FitResults(double _mass, double _chi2, double _probability, int _convergence)
: mass(_mass), chi2(_chi2), probability(_probability), convergence(_convergence)
{}
};
}*/

using RVecF = ROOT::VecOps::RVec<float>;
using RVecB = ROOT::VecOps::RVec<bool>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecUC = ROOT::VecOps::RVec<unsigned char>;
using RVecUS = ROOT::VecOps::RVec<unsigned short>;
using RVecUL = ROOT::VecOps::RVec<unsigned long>;
using RVecULL = ROOT::VecOps::RVec<unsigned long long>;
using RVecSh = ROOT::VecOps::RVec<short>;

namespace analysis {
    typedef std::variant<int,
                         float,
                         bool,
                         short,
                         unsigned long,
                         unsigned long long,
                         long long,
                         long,
                         unsigned int,
                         unsigned short,
                         RVecI,
                         RVecF,
                         RVecUC,
                         RVecUS,
                         RVecUL,
                         RVecULL,
                         RVecSh,
                         RVecB,
                         double,
                         unsigned char,
                         char>
        MultiType;  // Removed kin_fit::FitResults from the variant

    struct Entry {
        std::vector<MultiType> var_values;

        explicit Entry(size_t size) : var_values(size) {}

        template <typename T>
        void Add(int index, const T& value) {
            var_values.at(index) = value;
        }

        // Konstantin approved that this method can be removed. For bbWW kin_fit is not defined and caused crashes
        // void Add(int index, const kin_fit::FitResults& value)
        // {
        //   kin_fit::FitResults toAdd(value.mass, value.chi2, value.probability, value.convergence) ;
        //   var_values.at(index)= toAdd;
        // }

        template <typename T>
        const T& GetValue(int idx) const {
            return std::get<T>(var_values.at(idx));
        }
    };

    struct StopLoop {};
    static std::map<unsigned long long, std::shared_ptr<Entry>>& GetEntriesMap() {
        static std::map<unsigned long long, std::shared_ptr<Entry>> entries;
        return entries;
    }

    static std::map<unsigned long long, std::shared_ptr<Entry>>& GetCacheEntriesMap(const std::string& cache_name) {
        static std::map<std::string, std::map<unsigned long long, std::shared_ptr<Entry>>> cache_entries;
        return cache_entries[cache_name];
    }

    template <typename... Args>
    struct MapCreator {
        ROOT::RDF::RNode processCentral(ROOT::RDF::RNode df_in,
                                        const std::vector<std::string>& var_names,
                                        bool checkDuplicates = true) {
            auto df_node =
                df_in
                    .Define(
                        "_entry",
                        [=](const Args&... args) {
                            auto entry = std::make_shared<Entry>(var_names.size());
                            int index = 0;
                            (void)std::initializer_list<int>{(entry->Add(index++, args), 0)...};
                            return entry;
                        },
                        var_names)
                    .Define("map_placeholder",
                            [&](const std::shared_ptr<Entry>& entry) {
                                const auto idx = entry->GetValue<unsigned long long>(0);
                                if (GetEntriesMap().find(idx) != GetEntriesMap().end()) {
                                    // std::cout << idx << "\t" << run << "\t" << evt << "\t" << lumi << std::endl;
                                    throw std::runtime_error("Duplicate cache_entry for index " + std::to_string(idx));
                                }

                                GetEntriesMap().emplace(idx, entry);
                                return true;
                            },
                            {"_entry"});
            return df_node;
        }
    };

    template <typename... Args>
    struct CacheCreator {
        ROOT::RDF::RNode processCache(ROOT::RDF::RNode df_in,
                                      const std::vector<std::string>& var_names,
                                      const std::string& map_name,
                                      const std::string& entry_name,
                                      bool checkDuplicates = true) {
            auto df_node =
                df_in
                    .Define(
                        entry_name,
                        [=](const Args&... args) {
                            auto cache_entry = std::make_shared<Entry>(var_names.size());
                            int index = 0;
                            (void)std::initializer_list<int>{(cache_entry->Add(index++, args), 0)...};
                            return cache_entry;
                        },
                        var_names)
                    .Define(
                        map_name,
                        [&](const std::shared_ptr<Entry>& cache_entry) {
                            const auto idx = cache_entry->GetValue<unsigned long long>(0);
                            if (GetCacheEntriesMap(map_name).find(idx) != GetCacheEntriesMap(map_name).end()) {
                                if (checkDuplicates) {
                                    std::cout << idx << std::endl;
                                    throw std::runtime_error("Duplicate cache_entry for index " + std::to_string(idx));
                                }
                                GetCacheEntriesMap(map_name).at(idx) = cache_entry;
                            }
                            GetCacheEntriesMap(map_name).emplace(idx, cache_entry);
                            return true;
                        },
                        {entry_name});
            return df_node;
        }
    };

}  // namespace analysis
