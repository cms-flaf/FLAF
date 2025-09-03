/*! Definiton of a thread-safe fixed size entry queue. */

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace analysis {

    template <typename Entry>
    class EntryQueue {
      public:
        using Queue = std::queue<Entry>;
        using Mutex = std::mutex;
        using Lock = std::unique_lock<Mutex>;
        using CondVar = std::condition_variable;

      public:
        explicit EntryQueue(size_t max_size, size_t max_entries = std::numeric_limits<size_t>::max())
            : max_size_(max_size),
              max_entries_(max_entries),
              n_entries_(0),
              input_available_(true),
              output_needed_(true) {}

        bool Push(const Entry &entry) {
            {
                Lock lock(mutex_);
                cond_var_.wait(
                    lock, [&] { return queue_.size() < max_size_ || n_entries_ >= max_entries_ || !output_needed_; });
                if (n_entries_ >= max_entries_ || !output_needed_)
                    return false;
                queue_.push(entry);
                ++n_entries_;
            }
            cond_var_.notify_all();
            return true;
        }

        bool Pop(Entry &entry) {
            bool entry_is_valid = false;
            ;
            {
                Lock lock(mutex_);
                cond_var_.wait(lock, [&] { return queue_.size() || !input_available_; });
                if (!queue_.empty()) {
                    entry = queue_.front();
                    entry_is_valid = true;
                    queue_.pop();
                }
            }
            cond_var_.notify_all();
            return entry_is_valid;
        }

        void SetInputAvailable(bool value) {
            {
                Lock lock(mutex_);
                input_available_ = value;
            }
            cond_var_.notify_all();
        }

        void SetOutputNeeded(bool value) {
            {
                Lock lock(mutex_);
                output_needed_ = value;
            }
            cond_var_.notify_all();
        }

      private:
        Queue queue_;
        const size_t max_size_, max_entries_;
        size_t n_entries_;
        bool input_available_, output_needed_;
        Mutex mutex_;
        CondVar cond_var_;
    };

}  // namespace analysis
