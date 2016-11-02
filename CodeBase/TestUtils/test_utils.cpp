// Date: Dec 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "test_utils.h"

#include "StringUtils/string_utils.h"

#include <chrono>
#include <iostream>
#include <string>

using namespace string_utils;
using namespace std;
using namespace std::chrono;

namespace test_utils {

string GetCurrentTime() {
  time_t current_time = time(nullptr);
  return asctime(localtime(&current_time));
}

string GetElapsedTime(const Timer& timer) {
  const int64_t miliseconds = timer.elapsed_time_.count();
  if (miliseconds < 0) {
    return "ERROR in GetElapsedTime: Negative time: " + timer.elapsed_time_.count();
  }
  if (miliseconds == 0) return "O.0 seconds";

  const int64_t seconds = miliseconds / 1000;
  const int days = seconds / 86400;
  const int hours = (seconds - (days * 86400)) / 3600;
  const int minutes = (seconds - (days * 86400) - (hours * 3600))/ 60;
  const double seconds_only =
      seconds - (days * 86400) - (hours * 3600) - (minutes * 60);

  if (days < 0 || hours < 0 || hours >= 24 || minutes < 0 || minutes >= 60 ||
      seconds_only < 0 || seconds_only >= 60.0) {
    return "ERROR in GetElapsedTime for time: " + timer.elapsed_time_.count();
  }    

  string to_return = "";
  to_return += days == 0 ? "" : (Itoa(days) + " Days, ");
  to_return += hours == 0 ? "" : (Itoa(hours) + ":");
  const string min_str =
      minutes == 0 ? "00" :
      minutes < 10 ? ("0" + Itoa(minutes)) : Itoa(minutes);
  to_return += (minutes == 0 && hours == 0) ? "" : (min_str + ":");
  if (seconds_only < 10.0 && !to_return.empty()) to_return += "0";
  to_return += Itoa(seconds_only) + ".";
  const int miliseconds_only = miliseconds % 1000;
  const string milisecond_padding =
      miliseconds_only == 0 ? "" :
      miliseconds_only < 10 ? "00" :
      miliseconds_only < 100 ? "0" : "";
  to_return += milisecond_padding + Itoa(miliseconds_only) + " seconds";
  return to_return;
}

bool StartTimer(Timer* timer) {
  if (timer == nullptr ||
      (timer->state_ != TimerState::TIMER_UNSTARTED &&
       timer->state_ != TimerState::TIMER_PAUSED)) {
    return false;
  }

  timer->state_ = TimerState::TIMER_RUNNING;
  timer->start_time_ = steady_clock::now();
  return true;
}

bool StopTimer(const bool stop_permanently, Timer* timer) {
  if (timer == nullptr || timer->state_ != TimerState::TIMER_RUNNING) {
    return false;
  }

  timer->stop_time_ = steady_clock::now();
  if (timer->stop_time_ < timer->start_time_) {
    return false;
  }

  timer->state_ =
      stop_permanently ? TimerState::TIMER_STOPPED : TimerState::TIMER_PAUSED;
  timer->elapsed_time_ +=
      duration_cast<milliseconds>(timer->stop_time_ - timer->start_time_);
  return true;
}

bool ResetTimer(Timer* timer) {
  if (timer == nullptr) return false;
  timer->state_ = TimerState::TIMER_UNSTARTED;
  timer->elapsed_time_ = milliseconds::zero();
  return true;
}

}  // namespace test_utils
