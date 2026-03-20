#include <mutex>
#include <set>
#include <map>
#include <vector>
#include <optional>
#include <memory>

namespace flaf {
    class RunLumiTracker {
    public:
        using LumiSetType = std::set<unsigned int>;
        using LumiRangeType = std::pair<unsigned int, unsigned int>;
        using LumiRangeVectorType = std::vector<LumiRangeType>;
        using RunMapType = std::map<unsigned int, LumiSetType>;
        using RunLumiRangeMapType = std::map<unsigned int, std::vector<LumiRangeType>>;


        RunLumiTracker() : runMap(std::make_shared<RunMapType>()), mutex(std::make_shared<std::mutex>()) {}
        bool operator()(const unsigned int run, const unsigned int lumi) {
            const std::lock_guard<std::mutex> lock(*mutex);
            auto& lumis = (*runMap)[run];
            return lumis.insert(lumi).second;
        }

        RunLumiRangeMapType getRunLumiRanges() const {
            const std::lock_guard<std::mutex> lock(*mutex);
            RunLumiRangeMapType allRanges;
            for (const auto& [run, lumis] : *runMap) {
                LumiRangeVectorType runRanges;
                std::optional<LumiRangeType> activeRange;
                for(const unsigned int currentLumi : lumis) {
                    if (!activeRange.has_value()) {
                        activeRange = LumiRangeType(currentLumi, currentLumi);
                    } else if (currentLumi == activeRange.value().second + 1) {
                        activeRange.value().second = currentLumi;
                    } else {
                        runRanges.push_back(activeRange.value());
                        activeRange = LumiRangeType(currentLumi, currentLumi);
                    }
                }
                if (activeRange.has_value()) {
                    runRanges.push_back(activeRange.value());
                }
                if (!runRanges.empty()) {
                    allRanges[run] = std::move(runRanges);
                }
            }
            return allRanges;
        }

        void clear() {
            const std::lock_guard<std::mutex> lock(*mutex);
            runMap->clear();
        }

    private:
        std::shared_ptr<RunMapType> runMap;
        std::shared_ptr<std::mutex> mutex;
    };
};
