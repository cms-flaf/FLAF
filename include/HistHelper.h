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

#include <vector>
#include <map>
#include <cmath>

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
    
    TH1D* rebinHistogramDict(TH1* hist_initial, int N_bins, 
                                const std::vector<std::pair<float, float>>& y_bin_ranges,
                                const std::vector<std::vector<float>>& output_bin_edges) {
        // Flatten output bin edges into a single sorted array
        std::vector<float> all_output_edges;
        float last_edge = 0.0;
        for (const auto& edges : output_bin_edges) {
            for (float edge : edges) {
                all_output_edges.push_back(edge + last_edge);
            }
            last_edge = all_output_edges.back();
        }
        // Sort and remove duplicates
        std::sort(all_output_edges.begin(), all_output_edges.end());
        all_output_edges.erase(std::unique(all_output_edges.begin(), all_output_edges.end()), all_output_edges.end());

        // Create output histogram with variable binning
        TH1D* hist_output = new TH1D("rebinned", "rebinned", all_output_edges.size() - 1, all_output_edges.data());
        hist_output->Sumw2();

        // Helper function to find bin index from value and edges
        auto findBinIndex = [](float value, const std::vector<float>& edges) -> int {
            if (edges.size() < 2) return -1;
            for (size_t i = 0; i < edges.size() - 1; ++i) {
                if (value >= edges[i] && value < edges[i + 1]) {
                    return i;
                }
            }
            return -1;
        };

        // Iterate through all bins in the original histogram
        for (int i = 0; i < N_bins; ++i) {
            int binX, binY, binZ;
            hist_initial->GetBinXYZ(i, binX, binY, binZ);

            // Get bin centers (actual values)
            float x_value = hist_initial->GetXaxis()->GetBinCenter(binX);
            float y_value = hist_initial->GetYaxis()->GetBinCenter(binY);
            float z_value = hist_initial->GetZaxis()->GetBinCenter(binZ);

            // Get bin content and error
            double bin_content = hist_initial->GetBinContent(i);
            double bin_error = hist_initial->GetBinError(i);
            double bin_error2 = bin_error * bin_error;

            // Find which y_bin range this y_value falls into
            int y_bin_idx = -1;
            for (size_t j = 0; j < y_bin_ranges.size(); ++j) {
                if (y_value >= y_bin_ranges[j].first && y_value < y_bin_ranges[j].second) {
                    y_bin_idx = j;
                    break;
                }
            }
            if (y_bin_idx == -1) continue;  // Skip if y_value doesn't fall in any range
            // Find output bin index within the output_bin_edges for this y_bin
            int local_out_bin = findBinIndex(x_value, output_bin_edges[y_bin_idx]);
            if (local_out_bin == -1) continue;  // Skip if x_value doesn't fall in any output bin
            // Calculate section offset by counting bins in all previous y_bin sections
            int section_offset = 0;
            for (int prev_y = 0; prev_y < y_bin_idx; ++prev_y) {
                section_offset += output_bin_edges[prev_y].size() - 1;  // size - 1 = number of bins
            }
            // Calculate global bin index: offset + local bin position within this section
            int global_bin = section_offset + local_out_bin + 1;  // +1 for ROOT's 1-indexed bins
            // Set bin content and error
            if (global_bin >= 1 && global_bin <= (int)all_output_edges.size() - 1) {
                hist_output->SetBinContent(global_bin, hist_output->GetBinContent(global_bin) + bin_content);
                hist_output->SetBinError(global_bin, std::sqrt(std::pow(hist_output->GetBinError(global_bin), 2) + bin_error2));
            }
        }
        return hist_output;
    }
}  // namespace analysis
