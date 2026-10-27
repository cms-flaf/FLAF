#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "ROOT/RVec.hxx"
#include "TDirectory.h"
#include "TH1.h"

namespace analysis {

// The histograms HistProducerFromNTuple writes, filled in one event loop per tree: every (selection key,
// weight, variable) histogram of the loop is a block of a flat array. Per event the caller gives which
// keys the event passes, the bin index of every variable component (scalar columns, and vector columns
// that fill one entry per element) and the weights. A fill adds w and w*w to the cell that the producer's
// unit-bin histogram (n bins over -0.5 .. n-0.5) gives the bin index, as TH1::Fill does, in event order
// and within an event in element order: the histograms are identical to filling one RDataFrame histogram
// action per (key, weight, variable) single-threaded. Not thread-safe: one instance per event loop,
// without implicit MT.
class HistFiller {
public:
    // One variable of 1 to 3 dimensions. Per dimension: its number of bins, whether its bin column is a
    // vector, and the index of the column among the loop's scalar or vector bin columns.
    void AddVariable(const std::vector<int>& n_bins, const std::vector<int>& is_vector,
                     const std::vector<int>& index)
    {
        if (n_bins.empty() || n_bins.size() > 3 || is_vector.size() != n_bins.size()
            || index.size() != n_bins.size())
            throw std::invalid_argument("HistFiller: a variable has 1 to 3 dimensions");
        Variable v;
        v.dims = static_cast<int>(n_bins.size());
        v.offset = block_;
        v.n_cells = 1;
        for (int d = 0; d < v.dims; ++d) {
            v.n_bins[d] = n_bins[d];
            v.is_vector[d] = is_vector[d] != 0;
            v.index[d] = index[d];
            v.stride[d] = v.n_cells;
            v.n_cells *= static_cast<size_t>(n_bins[d] + 2);
        }
        block_ += v.n_cells;
        vars_.push_back(v);
    }

    void Allocate(size_t n_keys, size_t n_weights)
    {
        n_keys_ = n_keys;
        n_weights_ = n_weights;
        sum_.assign(n_keys_ * n_weights_ * block_, 0.);
        sum_w2_.assign(n_keys_ * n_weights_ * block_, 0.);
        entries_.assign(n_keys_ * n_weights_ * vars_.size(), 0);
        first_cell_.assign(vars_.size() + 1, 0);
    }

    size_t NumberOfCells() const { return n_keys_ * n_weights_ * block_; }

    int Fill(const ROOT::RVec<char>& pass, const ROOT::RVec<int>& scalar_bins,
             const ROOT::RVec<ROOT::RVec<int>>& vector_bins, const ROOT::RVec<double>& weights)
    {
        if (pass.size() != n_keys_ || weights.size() != n_weights_)
            throw std::runtime_error("HistFiller: wrong number of keys or weights");
        bool any_key = false;
        for (char p : pass)
            any_key = any_key || p;
        if (!any_key)
            return 0;
        // this event's cells, per variable (several for a vector variable)
        cells_.clear();
        for (size_t i = 0; i < vars_.size(); ++i) {
            const Variable& v = vars_[i];
            first_cell_[i] = cells_.size();
            size_t n_entries = 1;
            bool has_vector = false;
            for (int d = 0; d < v.dims; ++d) {
                if (!v.is_vector[d])
                    continue;
                const size_t n = vector_bins[v.index[d]].size();
                if (has_vector && n != n_entries)
                    throw std::runtime_error("HistFiller: vector columns of one variable differ in size");
                n_entries = n;
                has_vector = true;
            }
            for (size_t e = 0; e < n_entries; ++e) {
                size_t cell = v.offset;
                for (int d = 0; d < v.dims; ++d) {
                    const int bin = v.is_vector[d] ? vector_bins[v.index[d]][e] : scalar_bins[v.index[d]];
                    cell += v.stride[d] * UnitCell(bin, v.n_bins[d]);
                }
                cells_.push_back(cell);
            }
        }
        first_cell_[vars_.size()] = cells_.size();
        for (size_t k = 0; k < n_keys_; ++k) {
            if (!pass[k])
                continue;
            for (size_t j = 0; j < n_weights_; ++j) {
                const double w = weights[j];
                const size_t base = (k * n_weights_ + j) * block_;
                int64_t* entries = &entries_[(k * n_weights_ + j) * vars_.size()];
                for (size_t i = 0; i < vars_.size(); ++i) {
                    for (size_t c = first_cell_[i]; c < first_cell_[i + 1]; ++c) {
                        sum_[base + cells_[c]] += w;
                        sum_w2_[base + cells_[c]] += w * w;
                    }
                    entries[i] += static_cast<int64_t>(first_cell_[i + 1] - first_cell_[i]);
                }
            }
        }
        return 1;
    }

    // Writes every histogram: hists[i] is a histogram with variable i's binning, reused for each of its
    // histograms; dirs[i * n_keys + k] the directory of key k in variable i's file; names[i * n_weights + j]
    // the name of variable i's histogram for weight j.
    void Write(const std::vector<TH1*>& hists, const std::vector<TDirectory*>& dirs,
               const std::vector<std::string>& names) const
    {
        if (hists.size() != vars_.size() || dirs.size() != vars_.size() * n_keys_
            || names.size() != vars_.size() * n_weights_)
            throw std::invalid_argument("HistFiller::Write: wrong number of histograms, directories or names");
        for (size_t i = 0; i < vars_.size(); ++i) {
            const Variable& v = vars_[i];
            TH1* h = hists[i];
            if (static_cast<size_t>(h->GetNcells()) != v.n_cells)
                throw std::invalid_argument("HistFiller::Write: histogram binning differs from the variable's");
            for (size_t k = 0; k < n_keys_; ++k) {
                for (size_t j = 0; j < n_weights_; ++j) {
                    const size_t base = (k * n_weights_ + j) * block_ + v.offset;
                    for (size_t c = 0; c < v.n_cells; ++c) {
                        h->SetBinContent(static_cast<int>(c), sum_[base + c]);
                        h->SetBinError(static_cast<int>(c), std::sqrt(sum_w2_[base + c]));
                    }
                    h->SetEntries(static_cast<double>(entries_[(k * n_weights_ + j) * vars_.size() + i]));
                    dirs[i * n_keys_ + k]->WriteTObject(h, names[i * n_weights_ + j].c_str(), "Overwrite");
                }
            }
        }
    }

    // gives the arrays' memory back
    void Release()
    {
        std::vector<double>().swap(sum_);
        std::vector<double>().swap(sum_w2_);
        std::vector<int64_t>().swap(entries_);
    }

private:
    struct Variable {
        int dims = 0;
        int n_bins[3] = {0, 0, 0};
        bool is_vector[3] = {false, false, false};
        int index[3] = {0, 0, 0};
        size_t stride[3] = {0, 0, 0};
        size_t offset = 0;
        size_t n_cells = 0;
    };

    // the cell of a unit-bin axis (n bins over -0.5 .. n-0.5) that TH1::Fill gives an integer bin value
    static size_t UnitCell(int bin, int n_bins)
    {
        return bin < 0 ? 0 : (bin >= n_bins ? static_cast<size_t>(n_bins + 1) : static_cast<size_t>(bin + 1));
    }

    std::vector<Variable> vars_;
    size_t block_ = 0;
    size_t n_keys_ = 0, n_weights_ = 0;
    std::vector<double> sum_, sum_w2_;
    std::vector<int64_t> entries_;
    std::vector<size_t> cells_, first_cell_;
};

// A bin column of integers as RVec<int>, for packing the vector bin columns of a loop.
template <typename T>
ROOT::RVec<int> HistFillerBins(const ROOT::RVec<T>& bins)
{
    static_assert(std::is_integral<T>::value, "HistFiller: bin columns are integers");
    return ROOT::RVec<int>(bins.begin(), bins.end());
}

// Fill() of the filler at `address`, for a JIT-compiled Define of the event loop.
inline int HistFillerFill(std::uintptr_t address, const ROOT::RVec<char>& pass, const ROOT::RVec<int>& scalar_bins,
                          const ROOT::RVec<ROOT::RVec<int>>& vector_bins, const ROOT::RVec<double>& weights)
{
    return reinterpret_cast<HistFiller*>(address)->Fill(pass, scalar_bins, vector_bins, weights);
}

} // namespace analysis
