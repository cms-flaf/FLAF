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
#include "Utilities.h"

using RVecF = ROOT::VecOps::RVec<float>;
using RVecI = ROOT::VecOps::RVec<int>;
using RVecUC = ROOT::VecOps::RVec<unsigned char>;
using RVecC = ROOT::VecOps::RVec<char>;
using RVecUL = ROOT::VecOps::RVec<unsigned long>;
using RVecULL = ROOT::VecOps::RVec<unsigned long long>;
using RVecS = ROOT::VecOps::RVec<short>;
using RVecUS = ROOT::VecOps::RVec<unsigned short>;
using RVecB = ROOT::VecOps::RVec<bool>;

namespace analysis {
    typedef std::variant<int,
                         float,
                         bool,
                         unsigned long,
                         unsigned long long,
                         long,
                         unsigned int,
                         unsigned char,
                         char,
                         short,
                         RVecI,
                         RVecF,
                         RVecUC,
                         RVecC,
                         RVecS,
                         RVecUS,
                         RVecUL,
                         RVecULL,
                         RVecB>
        MultiType;

    struct Entry {
        std::vector<MultiType> var_values;

        explicit Entry(size_t size) : var_values(size) {}

        template <typename T>
        void Add(unsigned long long index, const T &value) {
            var_values.at(index) = value;
        }

        template <typename T>
        const T &GetValue(unsigned long long idx) const {
            return std::get<T>(var_values.at(idx));
        }
    };

    struct StopLoop {};

    template <typename... Args>
    struct TupleMaker {
        // TupleMaker(const std::string& tree_name, const std::string& in_file, size_t
        // queue_size)
        //   : df_in(tree_name, in_file), queue(queue_size)
        //{
        // }
        TupleMaker(const ROOT::RDataFrame &df_in_, size_t queue_size) : df_in(df_in_), queue(queue_size) {}

        TupleMaker(const TupleMaker &) = delete;
        TupleMaker &operator=(const TupleMaker &) = delete;

        void processIn(const std::vector<std::string> &var_names) {
            auto df_node = df_in.Define(
                "_entry",
                [=, this](const Args &...args) {
                    auto entry = std::make_shared<Entry>(var_names.size());
                    int index = 0;
                    (void)std::initializer_list<int>{(entry->Add(index++, args), 0)...};
                    return entry;
                },
                var_names);
            thread = std::make_unique<std::thread>([=, this]() {
                std::cout << "TupleMaker::processIn: thread started." << std::endl;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cond_var.wait(lock);
                }
                std::cout << "TupleMaker::processIn: starting foreach." << std::endl;
                try {
                    ROOT::RDF::RNode df = df_node;
                    df.Foreach(
                        [&](const std::shared_ptr<Entry> &entry) {
                            if (!queue.Push(entry)) {
                                throw StopLoop();
                            }
                        },
                        {"_entry"});
                } catch (StopLoop) {
                } catch (std::exception &e) {
                    std::cout << "TupleMaker::processIn: exception: " << e.what() << std::endl;
                    throw;
                }
                queue.SetInputAvailable(false);
            });
        }

        ROOT::RDF::RNode processOut(ROOT::RDF::RNode df_out) {
            auto notify = [&]() {
                std::cout << "TupleMaker::processOut: notifying" << std::endl;
                std::unique_lock<std::mutex> lock(mutex);
                cond_var.notify_all();
                return true;
            };
            df_out =
                df_out.Define("_entryCentral",
                              [=, this](ULong64_t FullEventIdShifted) {
                                  std::shared_ptr<Entry> entryCentral;
                                  try {
                                      static bool notified = notify();
                                      static std::shared_ptr<Entry> entry;
                                      static std::set<unsigned long long> processedEntries;
                                      if (processedEntries.count(FullEventIdShifted))
                                          throw std::runtime_error("Entry already processed");
                                      while (!entry || entry->GetValue<unsigned long long>(0) < FullEventIdShifted) {
                                          if (entry) {
                                              processedEntries.insert(entry->GetValue<unsigned long long>(0));
                                              entry.reset();
                                          }
                                          if (!queue.Pop(entry)) {
                                              break;
                                          }
                                      }
                                      if (entry && entry->GetValue<unsigned long long>(0) == FullEventIdShifted) {
                                          entryCentral = entry;
                                      }
                                  } catch (const std::exception &e) {
                                      std::cout << "TupleMaker::processOut: exception: " << e.what() << std::endl;
                                      throw;
                                  }
                                  return entryCentral;
                              },
                              {"FullEventId"});

            return df_out;
        }

        void join() {
            if (thread) {
                queue.SetOutputNeeded(false);
                thread->join();
            }
        }

        ROOT::RDataFrame df_in;
        EntryQueue<std::shared_ptr<Entry>> queue;
        std::unique_ptr<std::thread> thread;
        std::mutex mutex;
        std::condition_variable cond_var;
    };

}  // namespace analysis
