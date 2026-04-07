#if !(defined MPLR_THREAD_STOPWATCH_HPP)

#define MPLR_THREAD_STOPWATCH_HPP

#include <chrono>


// macOS / iOS (Darwin) implementation:
#if defined __APPLE__ and defined __MACH__


#include <mach/mach.h>
#include <mach/mach_time.h>

namespace mplr::detail::impl {

  class thread_stopwatch {
  public:
    thread_stopwatch() noexcept
        : m_current_thread{mach_thread_self()}, m_t0{thread_clock_in_us()} {
    }

    ~thread_stopwatch() {
      mach_port_deallocate(mach_task_self(), m_current_thread);
    }

    auto reset() noexcept -> void {
      m_t0 = thread_clock_in_us();
    }

    auto read() const noexcept -> std::chrono::nanoseconds {
      return std::chrono::nanoseconds{(thread_clock_in_us() - m_t0) * 1000};
    }

  private:
    auto thread_clock_in_us() const noexcept -> long long {
      mach_msg_type_number_t count{THREAD_BASIC_INFO_COUNT};
      thread_basic_info_data_t info{};
      kern_return_t kr{
          thread_info(m_current_thread, THREAD_BASIC_INFO, (thread_info_t)&info, &count)};
      if (kr != KERN_SUCCESS) {
        return 0;
      }
      return (info.user_time.seconds + info.system_time.seconds) * 1'000'000ll +
             (info.user_time.microseconds + info.system_time.microseconds);
    }

  private:
    mach_port_t m_current_thread;
    long long m_t0;
  };

}  // namespace mplr::detail::impl


// Linux/BSD implementation:
#elif (defined linux or defined __linux__ or defined __linux) or           \
    (defined __DragonFly__ or defined __FreeBSD__ or defined __NetBSD__ or \
     defined __OpenBSD__)


#include <time.h>

namespace mplr::detail::impl {

  class thread_stopwatch {
  public:
    thread_stopwatch() noexcept : m_t0{} {
      reset();
    }

    auto reset() noexcept -> void {
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_t0);
    }

    auto read() const noexcept -> std::chrono::nanoseconds {
      struct timespec t{};
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
      return std::chrono::nanoseconds{(t.tv_sec - m_t0.tv_sec) * 1'000'000'000ll +
                                      (t.tv_nsec - m_t0.tv_nsec)};
    }

  private:
    struct timespec m_t0;
  };

}  // namespace mplr::detail::impl


// Windows implementation:
#elif defined _WIN32


#if defined _MSC_VER and not defined __clang__ and not defined __GNUC__ and not defined NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#else
#include <windows.h>
#endif

namespace mplr::detail::impl {

  class thread_stopwatch {
  public:
    thread_stopwatch() noexcept
        : m_current_thread{GetCurrentThread()}, m_t0{thread_clock_in_100ns()} {
    }

    auto reset() noexcept -> void {
      m_t0 = thread_clock_in_100ns();
    }

    auto read() const noexcept -> std::chrono::nanoseconds {
      return std::chrono::nanoseconds{(thread_clock_in_100ns() - m_t0) * 100};
    }

  private:
    auto thread_clock_in_100ns() const noexcept -> ULONGLONG {
      FILETIME _1;
      FILETIME _2;
      FILETIME t_k{};
      FILETIME t_u{};
      GetThreadTimes(m_current_thread, &_1, &_2, &t_k, &t_u);
      ULARGE_INTEGER t_kernel;
      t_kernel.LowPart = t_k.dwLowDateTime;
      t_kernel.HighPart = t_k.dwHighDateTime;
      ULARGE_INTEGER t_user;
      t_user.LowPart = t_u.dwLowDateTime;
      t_user.HighPart = t_u.dwHighDateTime;
      return t_kernel.QuadPart + t_user.QuadPart;
    }

  private:
    HANDLE m_current_thread;
    ULONGLONG m_t0;
  };

}  // namespace mplr::detail::impl


// Fallback implementation:
#else


#include <cmath>
#include <cstdlib>
#include <ctime>

namespace mplr::detail::impl {

  class thread_stopwatch {
  public:
    thread_stopwatch() noexcept : m_t0{std::clock()} {
    }

    auto reset() noexcept -> void {
      m_t0 = std::clock();
    }

    auto read() const noexcept -> std::chrono::nanoseconds {
      const auto clocks{std::clock() - m_t0};
      constexpr auto tick{std::div(static_cast<clock_t>(1'000'000'000), CLOCKS_PER_SEC)};
      if constexpr (tick.rem == 0) {
        return std::chrono::nanoseconds{clocks * tick.quot};
      } else {
        return std::chrono::nanoseconds{std::llround(clocks * (1e9 / CLOCKS_PER_SEC))};
      }
    }

  private:
    std::clock_t m_t0;
  };

}  // namespace mplr::detail::impl


#endif


namespace mplr::detail {

  class thread_stopwatch {
  public:
    using duration = std::chrono::nanoseconds;

  public:
    auto reset() noexcept -> void {
      m_impl.reset();
    }

    auto read() const noexcept -> duration {
      return m_impl.read();
    }

  private:
    impl::thread_stopwatch m_impl;
  };

}  // namespace mplr::detail

#endif
