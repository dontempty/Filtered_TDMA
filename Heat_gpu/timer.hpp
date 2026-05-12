#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <string>
#include <unordered_map>

class MultiTimer {
public:
    void start(const std::string& label) {
        timers_[label] = std::chrono::steady_clock::now();
    }

    double elapsed_sec(const std::string& label) const {
        auto it = timers_.find(label);
        if (it == timers_.end()) return -1.0;
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(now - it->second).count();
    }

    long long elapsed_ns(const std::string& label) const {
        auto it = timers_.find(label);
        if (it == timers_.end()) return -1;
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now - it->second).count();
    }

private:
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> timers_;
};

#endif // TIMER_HPP
