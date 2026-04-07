#if !(defined MPLR_STOPWATCH_HPP)

#define MPLR_STOPWATCH_HPP

#include <chrono>


// Linux/BSD implementation:
#if (defined linux or defined __linux__ or defined __linux) or             \
    (defined __DragonFly__ or defined __FreeBSD__ or defined __NetBSD__ or \
     defined __OpenBSD__)


#include <time.h>

namespace mplr::detail::impl {

  class stopwatch {
  public:
    stopwatch() noexcept : m_t0{} {
      reset();
    }

    auto reset() noexcept -> void {
      clock_gettime(CLOCK_MONOTONIC, &m_t0);
    }

    auto read() const noexcept -> std::chrono::nanoseconds {
      struct timespec t{};
      clock_gettime(CLOCK_MONOTONIC, &t);
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

#include <cmath>
#include <cstdlib>
#include <variant>

namespace mplr::detail::impl {

  class stopwatch {
  public:
    stopwatch() noexcept : m_tick{}, m_t0{} {
      LARGE_INTEGER frequency{};
      QueryPerformanceFrequency(&frequency);
      const auto tick{std::div(1'000'000'000ll, frequency.QuadPart)};
      if (tick.rem == 0) {
        m_tick = tick.quot;
      } else {
        m_tick = 1e9 / frequency.QuadPart;
      }
      reset();
    }

    auto reset() noexcept -> void {
      QueryPerformanceCounter(&m_t0);
    }

    auto read() const noexcept -> std::chrono::nanoseconds {
      LARGE_INTEGER t{};
      QueryPerformanceCounter(&t);
      const auto clocks{t.QuadPart - m_t0.QuadPart};
      if (m_tick.index() == 0) {
        return std::chrono::nanoseconds{clocks * get<LONGLONG>(m_tick)};
      } else {
        return std::chrono::nanoseconds{std::llround(clocks * get<double>(m_tick))};
      }
    }

  private:
    std::variant<LONGLONG, double> m_tick;
    LARGE_INTEGER m_t0;
  };

}  // namespace mplr::detail::impl


// Fallback implementation:
#else


#include <type_traits>

namespace mplr::detail::impl {

  class stopwatch {
  private:
    using clock =
        std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                           std::chrono::high_resolution_clock, std::chrono::steady_clock>;
    ;

  public:
    stopwatch() noexcept : m_t0{clock::now()} {
    }

    auto reset() noexcept -> void {
      m_t0 = clock::now();
    }

    auto read() const noexcept -> std::chrono::nanoseconds {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - m_t0);
    }

  private:
    clock::time_point m_t0;
  };

}  // namespace mplr::detail::impl


#endif


namespace mplr::detail {

  class stopwatch {
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
    impl::stopwatch m_impl;
  };

}  // namespace mplr::detail


#endif
