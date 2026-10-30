#pragma once

// The fill runs once per (event, loop) and dominates the event loop: compile this header at -O2 instead of
// cling's default -O1. Only the code of this transaction is affected, which is why HistFillerFill below is not
// inline (an inline function would be compiled in the transaction of the Define that calls it).
#pragma cling optimize(2)

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
    // weight, variable) histogram of the loop is part of a flat array. Per event the caller gives which keys the
    // event passes, the bin index of every variable component (scalar columns, and vector columns that fill one
    // entry per element) and the weights. A fill adds w and w*w to the cell that the producer's unit-bin
    // histogram (n bins over -0.5 .. n-0.5) gives the bin index, as TH1::Fill does, in event order and within an
    // event in element order. Every slot of the event loop (implicit MT) fills its own copy of the arrays, and
    // Write sums the copies in slot order: with one slot the histograms are identical to filling one RDataFrame
    // histogram action per (key, weight, variable) single-threaded; with several the entries are exact and the
    // sums differ only by the order of the additions. One instance per event loop, allocated for the loop's
    // slots (GetNSlots()); Fill takes rdfslot_.
    class HistFiller {
      public:
        // One variable of 1 to 3 dimensions. Per dimension: its number of bins, whether its bin column is a
        // vector, and the index of the column among the loop's scalar or vector bin columns.
        void AddVariable(const std::vector<int>& n_bins,
                         const std::vector<int>& is_vector,
                         const std::vector<int>& index) {
            if (n_bins.empty() || n_bins.size() > 3 || is_vector.size() != n_bins.size() ||
                index.size() != n_bins.size())
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

        void Allocate(size_t n_keys, size_t n_weights, size_t n_slots) {
            if (n_slots < 1)
                throw std::invalid_argument("HistFiller: an event loop has at least one slot");
            n_keys_ = n_keys;
            n_weights_ = n_weights;
            slots_.assign(n_slots, Slot());
            for (Slot& s : slots_) {
                s.acc.assign(2 * NumberOfCells(), 0.);
                s.entries.assign(n_keys_ * vars_.size(), 0);
                s.first_cell.assign(vars_.size() + 1, 0);
                s.w2.assign(n_weights_, 0.);
            }
        }

        // cells of one slot's copy
        size_t NumberOfCells() const { return n_keys_ * n_weights_ * block_; }

        int Fill(unsigned int slot,
                 const ROOT::RVec<char>& pass,
                 const ROOT::RVec<int>& scalar_bins,
                 const ROOT::RVec<ROOT::RVec<int>>& vector_bins,
                 const ROOT::RVec<double>& weights) {
            if (pass.size() != n_keys_ || weights.size() != n_weights_)
                throw std::runtime_error("HistFiller: wrong number of keys or weights");
            if (slot >= slots_.size())
                throw std::runtime_error("HistFiller: slot out of range: Allocate for the event loop's slots");
            Slot& s = slots_[slot];
            bool any_key = false;
            for (char p : pass)
                any_key = any_key || p;
            if (!any_key)
                return 0;
            // this event's cells, per variable (several for a vector variable)
            s.cells.clear();
            for (size_t i = 0; i < vars_.size(); ++i) {
                const Variable& v = vars_[i];
                s.first_cell[i] = s.cells.size();
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
                    s.cells.push_back(cell);
                }
            }
            s.first_cell[vars_.size()] = s.cells.size();
            for (size_t j = 0; j < n_weights_; ++j)
                s.w2[j] = weights[j] * weights[j];
            // The weights of a (key, cell) are contiguous, sum and squared sum interleaved: one cache line serves
            // several weights. Every accumulator gets the additions it would get in one histogram per (key, weight,
            // variable), in the same order, so the result does not depend on the layout.
            for (size_t k = 0; k < n_keys_; ++k) {
                if (!pass[k])
                    continue;
                int64_t* entries = &s.entries[k * vars_.size()];
                for (size_t i = 0; i < vars_.size(); ++i) {
                    for (size_t c = s.first_cell[i]; c < s.first_cell[i + 1]; ++c) {
                        double* acc = &s.acc[2 * (k * block_ + s.cells[c]) * n_weights_];
                        for (size_t j = 0; j < n_weights_; ++j) {
                            acc[2 * j] += weights[j];
                            acc[2 * j + 1] += s.w2[j];
                        }
                    }
                    entries[i] += static_cast<int64_t>(s.first_cell[i + 1] - s.first_cell[i]);
                }
            }
            return 1;
        }

        // Writes every histogram, the sum of the slots' copies: hists[i] is a histogram with variable i's
        // binning, reused for each of its histograms; dirs[i * n_keys + k] the directory of key k in variable
        // i's file; names[i * n_weights + j] the name of variable i's histogram for weight j.
        void Write(const std::vector<TH1*>& hists,
                   const std::vector<TDirectory*>& dirs,
                   const std::vector<std::string>& names) const {
            if (hists.size() != vars_.size() || dirs.size() != vars_.size() * n_keys_ ||
                names.size() != vars_.size() * n_weights_)
                throw std::invalid_argument("HistFiller::Write: wrong number of histograms, directories or names");
            for (size_t i = 0; i < vars_.size(); ++i) {
                const Variable& v = vars_[i];
                TH1* h = hists[i];
                if (static_cast<size_t>(h->GetNcells()) != v.n_cells)
                    throw std::invalid_argument("HistFiller::Write: histogram binning differs from the variable's");
                for (size_t k = 0; k < n_keys_; ++k) {
                    int64_t entries = 0;
                    for (const Slot& s : slots_)
                        entries += s.entries[k * vars_.size() + i];
                    for (size_t j = 0; j < n_weights_; ++j) {
                        for (size_t c = 0; c < v.n_cells; ++c) {
                            const size_t a = 2 * ((k * block_ + v.offset + c) * n_weights_ + j);
                            double sum = slots_[0].acc[a], sum_w2 = slots_[0].acc[a + 1];
                            for (size_t s = 1; s < slots_.size(); ++s) {
                                sum += slots_[s].acc[a];
                                sum_w2 += slots_[s].acc[a + 1];
                            }
                            h->SetBinContent(static_cast<int>(c), sum);
                            h->SetBinError(static_cast<int>(c), std::sqrt(sum_w2));
                        }
                        h->SetEntries(static_cast<double>(entries));
                        dirs[i * n_keys_ + k]->WriteTObject(h, names[i * n_weights_ + j].c_str(), "Overwrite");
                    }
                }
            }
        }

        // gives the arrays' memory back, every slot's
        void Release() { std::vector<Slot>().swap(slots_); }

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

        // A slot's copy of the arrays and its scratch for the event it fills, on cache lines of its own.
        // acc[2 * ((k * block_ + cell) * n_weights_ + j) + {0, 1}]: sum and squared sum of key k, weight j;
        // entries[k * n_vars + i]: the entries of variable i for key k, the same for every weight.
        struct alignas(64) Slot {
            std::vector<double> acc;
            std::vector<int64_t> entries;
            std::vector<double> w2;
            std::vector<size_t> cells, first_cell;
        };

        // the cell of a unit-bin axis (n bins over -0.5 .. n-0.5) that TH1::Fill gives an integer bin value
        static size_t UnitCell(int bin, int n_bins) {
            return bin < 0 ? 0 : (bin >= n_bins ? static_cast<size_t>(n_bins + 1) : static_cast<size_t>(bin + 1));
        }

        std::vector<Variable> vars_;
        size_t block_ = 0;
        size_t n_keys_ = 0, n_weights_ = 0;
        std::vector<Slot> slots_;
    };

    // Bin columns hold integers that fit an int: a wider or an unsigned int type could be narrowed to another bin.
    template <typename T>
    constexpr bool kHistFillerBinType =
        std::is_integral<T>::value &&
        (sizeof(T) < sizeof(int) || (sizeof(T) == sizeof(int) && std::is_signed<T>::value));

    // A vector bin column as RVec<int>, for packing the vector bin columns of a loop.
    template <typename T>
    ROOT::RVec<int> HistFillerBins(const ROOT::RVec<T>& bins) {
        static_assert(kHistFillerBinType<T>, "HistFiller: a bin column holds integers that fit an int");
        return ROOT::RVec<int>(bins.begin(), bins.end());
    }

    // A key filter's result, which must be a bool as for RDataFrame's Filter: an int or a float would otherwise
    // be narrowed to its low 8 bits.
    inline char HistFillerKey(bool pass) { return pass; }
    template <typename T>
    char HistFillerKey(T) = delete;

    // Fill() of the filler at `address` on loop slot `slot` (rdfslot_), for a JIT-compiled Define of the event
    // loop. Not inline, so that it is compiled with this header (see the pragma above).
    int HistFillerFill(std::uintptr_t address,
                       unsigned int slot,
                       const ROOT::RVec<char>& pass,
                       const ROOT::RVec<int>& scalar_bins,
                       const ROOT::RVec<ROOT::RVec<int>>& vector_bins,
                       const ROOT::RVec<double>& weights) {
        return reinterpret_cast<HistFiller*>(address)->Fill(slot, pass, scalar_bins, vector_bins, weights);
    }

}  // namespace analysis
