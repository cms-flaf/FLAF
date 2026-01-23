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
    
    TH1D rebinHistogramDict(const TH1& hist_initial, 
                                const std::vector<std::pair<float, float>>& y_bin_ranges,
                                const std::vector<std::vector<float>>& x_bin_edges) {
        // Count nBins, we don't need real edges for the output hist     
        int nBin_Counter = 0;
        for (const auto& edges : x_bin_edges) {
            if (edges.size() <= 1){
                throw std::runtime_error("Invalid x edges definition");
            }
            nBin_Counter = nBin_Counter + edges.size() - 1;
        }

        // Create output histogram with variable binning
        TH1D hist_output = TH1D("rebinned", "rebinned", nBin_Counter, -0.5, nBin_Counter - 0.5);
        hist_output.Sumw2();

        // Helper function to find bin index from value and edges
        auto findBinIndex = [](float value, const std::vector<float>& edges) -> int {
            for (size_t i = 0; i < edges.size() - 1; ++i) {
                if (value >= edges[i] && value < edges[i + 1]) {
                    return i;
                }
            }
            return -1;
        };

        // Iterate through all bins in the original histogram
        std::vector<double> all_bin_content(nBin_Counter, 0.0);
        std::vector<double> all_bin_error2(nBin_Counter, 0.0);
        for (int i = 0; i < hist_initial.GetNcells(); ++i){
            // If bin is overflow or underflow, ignore for linearizing
            if (hist_initial.IsBinUnderflow(i) || hist_initial.IsBinOverflow(i)) continue;

            int binX, binY, binZ;
            hist_initial.GetBinXYZ(i, binX, binY, binZ);

            // // Given our old x/y/z bin, make sure the new bins will be compatible
            // float old_x_low = hist_initial.GetXaxis()->GetBinLowEdge(binX);
            // float old_x_up = hist_initial.GetXaxis()->GetBinUpEdge(binX);
            // float old_y_low = hist_initial.GetYaxis()->GetBinLowEdge(binY);
            // float old_y_up = hist_initial.GetYaxis()->GetBinUpEdge(binY);

            // // Find which y_bin range these y_values fall into
            // for (size_t j = 0; j < y_bin_ranges.size(); ++j) {
            //     if ((old_y_low >= y_bin_ranges[j].first && old_y_low < y_bin_ranges[j].second) && (old_y_up >= y_bin_ranges[j].first && old_y_up < y_bin_ranges[j].second)) {
            //         int new_x_low = findBinIndex(old_x_low, x_bin_edges[j]);
            //         int new_x_up = findBinIndex(old_x_up, x_bin_edges[j]);
            //         if (new_x_low != new_x_up){
            //             std::cout << "Found the bad case" << std::endl;
            //             break;
            //         }
            //     }
            // }

            // Get bin centers (actual values)
            float x_value = hist_initial.GetXaxis()->GetBinCenter(binX);
            float y_value = hist_initial.GetYaxis()->GetBinCenter(binY);
            float z_value = hist_initial.GetZaxis()->GetBinCenter(binZ);

            // Get bin content and error
            const double bin_content = hist_initial.GetBinContent(i);
            const double bin_error = hist_initial.GetBinError(i);
            const double bin_error2 = bin_error * bin_error;

            // Find which y_bin range this y_value falls into
            std::cout << "Finding y bin range" << std::endl;
            int y_bin_idx = -1;
            for (size_t j = 0; j < y_bin_ranges.size(); ++j) {
                if (y_value >= y_bin_ranges[j].first && y_value < y_bin_ranges[j].second) {
                    y_bin_idx = j;
                    break;
                }
            }
            if (y_bin_idx == -1) continue;  // Skip if y_value doesn't fall in any range
            // Find output bin index within the x_bin_edges for this y_bin
            std::cout << "Finding bin index" << std::endl;
            int local_out_bin = findBinIndex(x_value, x_bin_edges[y_bin_idx]);
            if (local_out_bin == -1) continue;  // Skip if x_value doesn't fall in any output bin
            // Calculate section offset by counting bins in all previous y_bin sections
            int section_offset = 0;
            std::cout << "Finding offset" << std::endl;
            for (int prev_y = 0; prev_y < y_bin_idx; ++prev_y) {
                section_offset += x_bin_edges[prev_y].size() - 1;  // size - 1 = number of bins
            }
            // Calculate global bin index: offset + local bin position within this section
            int global_bin = section_offset + local_out_bin;

            // Store content and error2 at global bin index
            all_bin_content[global_bin] = all_bin_content[global_bin] + bin_content;
            all_bin_error2[global_bin] = all_bin_error2[global_bin] + bin_error2;
        }
        std::cout << "Filling" << std::endl;
        std::cout << "content size " << all_bin_content.size() << std::endl;
        std::cout << "And hist out has nBins: " << nBin_Counter << std::endl;
        for (int global_bin = 0; global_bin <= all_bin_content.size(); ++global_bin){
            std::cout << "Filling global bin " << global_bin << std::endl;
            hist_output.SetBinContent(global_bin + 1, all_bin_content[global_bin]);
            hist_output.SetBinError(global_bin + 1, std::sqrt(all_bin_error2[global_bin]));
        }
        std::cout << "Finished filling" << std::endl;
        return hist_output;
    }
}  // namespace analysis
