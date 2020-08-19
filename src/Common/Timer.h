#pragma once

#include <chrono>

class Timer {
public:
    double DeltaTime();
    double TotalTime();

    void Reset();
    void Tick();
    void Start();
    void Stop();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> base_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> prev_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_time;

    double delta_time;
    double paused_time;

    bool stopped = true;
};


