#include "Timer.h"

double Timer::DeltaTime() {
    return delta_time;
}

double Timer::TotalTime() {
    if (stopped) {
        std::chrono::duration<double> temp = stop_time - base_time;
        return temp.count() - paused_time;
    } else {
        std::chrono::duration<double> temp = prev_time - base_time;
        return temp.count() - paused_time;
    }
}

void Timer::Reset() {
    stopped = false;
    prev_time = base_time = std::chrono::high_resolution_clock::now();
    delta_time = paused_time = 0.0;
}

void Timer::Tick() {
    auto curr_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> delta = curr_time - prev_time;
    delta_time = delta.count();
    prev_time = curr_time;
}

void Timer::Start() {
    if (stopped) {
        auto curr_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> pause = curr_time - stop_time;
        paused_time += pause.count();
        delta_time = 0.0;
        prev_time = curr_time;
        stopped = false;
    }
}

void Timer::Stop() {
    if (!stopped) {
        stop_time = std::chrono::high_resolution_clock::now();
        delta_time = 0.0;
        stopped = true;
    }
}