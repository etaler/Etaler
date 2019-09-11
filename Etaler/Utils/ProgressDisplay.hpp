#pragma once

#include <chrono>
#include <iostream>

#include <cstdio>

namespace et
{

struct ProgressDisplay
{
        using time_point = std::chrono::high_resolution_clock::time_point;
        time_point start_;
        size_t num_work_;
        double last_time_ = 0;

        ProgressDisplay(size_t num_work)
        {
                using namespace std::chrono;
                start_ = high_resolution_clock::now();
                num_work_ = num_work;
        }

        void update(size_t current_done)
        {
                using namespace std;
                using namespace std::chrono;
                auto now = high_resolution_clock::now();
                auto time_span = duration_cast<duration<double>>(now - start_);
                double spend = time_span.count();

                double pct = 100.0*(double)current_done/num_work_;
                printf("%c[2K\r", 27);
                cout << pct << "% "
                        << current_done << "/" << num_work_ << ", "
                        << "time: " << spend << "s. ETA: "
                        << (size_t)((num_work_-current_done)*(spend/current_done)) << "s"
                        << ". delta = " << spend-last_time_
                                << flush;

                last_time_ = spend;
        }

        void reset()
        {
                *this = ProgressDisplay(num_work_);
        }
};

}