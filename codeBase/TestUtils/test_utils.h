// Date: Dec 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Helper functions for Unit Tests.

#include <chrono>  // For std::chrono:: milliseconds, timepoint, system_clock, etc.
#include <string>

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

using namespace std;
using namespace std::chrono;

namespace test_utils {
  
enum TimerState {
  TIMER_UNKNOWN,    // Unknown timer state.
  TIMER_UNSTARTED,  // Timer has not been started yet.
  TIMER_PAUSED,     // Timer is currently paused (may be restarted later).
  TIMER_RUNNING,    // Timer is currently running.
  TIMER_STOPPED,    // Timer is done tracking time (won't ever be re-started).
};

struct Timer {
  TimerState state_;
  // Only valid if state_ is TIMER_RUNNING; in which case it
  // gives the time that the timer was (most recently) started.
  time_point<steady_clock> start_time_;
  //PHBmilliseconds start_time_;
  // Only valid if state_ is TIMER_PAUSED or TIMER_STOPPED; in which case it
  // gives the time that the timer was (most recently) stopped.
  time_point<steady_clock> stop_time_;
  //PHBmilliseconds stop_time_;
  // Cummulative amount of time timer has been running, i.e.:
  //   \sum (stop_time - start_time)
  milliseconds elapsed_time_;

  Timer() {
    state_ = TimerState::TIMER_UNSTARTED;
    elapsed_time_ = milliseconds::zero();
  }
};

// Returns the current time in format:
//   WEEKDAY MONTH DATE HH:MM:SS YYYY
// e.g.:
//   Wed Nov 25 17:25:43 2015
extern string GetCurrentTime();

// Returns timer.elapsed_time_ in format:
//   DD HH:MM:SS.MILISECONDS
// e.g.:
//   5 Days, 17:25:43.12345
extern string GetElapsedTime(const Timer& timer);

// Checks that timer is in a valid state to be started: TIMER_UNSTARTED or
// TIMER_PAUSED. If so, sets the state_ of the input timer to TIMER_RUNNING,
// and sets start_time_ accordingly.
extern bool StartTimer(Timer* timer);

// Checks that timer is in a valid state to be stopped: TIMER_RUNNING. If so,
// updates elapsed_time_ accordingly, and sets stop_time_ to the current time.
// Sets state_ to TIMER_PAUSED or TIMER_STOPPED based on input 'stop_permanently'.
extern bool StopTimer(const bool stop_permanently, Timer* timer);
// Same as above, passing in default value of stop_permanently = false.
inline bool StopTimer(Timer* timer) {
  return StopTimer(false, timer);
}

// Clears elapsed_time_, and puts timer into state TIMER_UNSTARTED.
extern bool ResetTimer(Timer* timer);

}  // namespace test_utils

#endif
