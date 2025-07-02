#define BOOST_TEST_MODULE communicator_isend_irecv

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"
#include "test_helper.hpp"

#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>


template<typename T>
bool isend_irecv_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.isend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool isend_irecv_iter_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.isend(std::begin(data), std::end(data), 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      auto r{comm_world.irecv(data_r, 0)};
      while (not r.test()) {
      }
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      while (not r.test()) {
      }
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool ibsend_irecv_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int size;
    if constexpr (has_size<T>::value)
      size = comm_world.bsend_size<typename T::value_type>(data.size());
    else
      size = comm_world.bsend_size<T>();
    mplr::bsend_buffer buff(size);
    auto r{comm_world.ibsend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool ibsend_irecv_iter_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    int size;
    if constexpr (has_size<T>::value)
      size = comm_world.bsend_size<typename T::value_type>(data.size());
    else
      size = comm_world.bsend_size<T>();
    mplr::bsend_buffer buff(size);
    auto r{comm_world.ibsend(std::begin(data), std::end(data), 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      auto r{comm_world.irecv(data_r, 0)};
      while (not r.test()) {
      }
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      while (not r.test()) {
      }
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool issend_irecv_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.issend(data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    auto r{comm_world.irecv(data_r, 0)};
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool issend_irecv_iter_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{comm_world.issend(std::begin(data), std::end(data), 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(*std::begin(data_r))>>) {
      auto r{comm_world.irecv(data_r, 0)};
      while (not r.test()) {
      }
      return data_r == data;
    } else {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
      auto r{comm_world.irecv(std::begin(data_r), std::end(data_r), 0)};
      while (not r.test()) {
      }
      return data_r == data;
    }
  }
  return true;
}


template<typename T>
bool irsend_irecv_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    auto r{comm_world.irsend(data, 1)};
    r.wait();
  } else if (comm_world.rank() == 1) {
    // must ensure that MPI_Recv is called before mplr::communicator::rsend
    if constexpr (has_begin_end<T>() and has_size<T>() and has_resize<T>()) {
      // T is an STL container
      T data_r;
      data_r.resize(data.size());
      mplr::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      r.wait();
      return data_r == data;
    } else if constexpr (has_begin_end<T>() and has_size<T>()) {
      // T is an STL container without resize member, e.g., std::set
      std::vector<typename T::value_type> data_r;
      data_r.resize(data.size());
      mplr::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      while (not r.test()) {
      }
      return std::equal(begin(data_r), end(data_r), begin(data));
    } else {
      // T is some fundamental type
      // mplr::communicator::irecv does not suffice in the cases above as the irecv performs a
      // probe first to receive STL containers
      T data_r;
      mplr::irequest r{comm_world.irecv(data_r, 0)};
      comm_world.barrier();
      while (not r.test()) {
      }
      return data_r == data;
    }
  } else
    comm_world.barrier();
  return true;
}


template<typename T>
bool irsend_irecv_iter_test(const T &data) {
  const auto comm_world = mplr::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    comm_world.barrier();
    auto r{comm_world.irsend(std::begin(data), std::end(data), 1)};
    while (not r.test()) {
    }
  } else if (comm_world.rank() == 1) {
    // must ensure that MPI_Recv is called before mplr::communicator::rsend
    if constexpr (has_begin_end<T>() and has_size<T>() and has_resize<T>()) {
      T data_r;
      data_r.resize(data.size());
      mplr::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      while (not r.test()) {
      }
      return data_r == data;
    } else if constexpr (has_begin_end<T>() and has_size<T>()) {
      // T is an STL container without resize member, e.g., std::set
      std::vector<typename T::value_type> data_r;
      data_r.resize(data.size());
      mplr::irequest r{comm_world.irecv(begin(data_r), end(data_r), 0)};
      comm_world.barrier();
      while (not r.test()) {
      }
      return std::equal(begin(data_r), end(data_r), begin(data));
    } else {
      // T is some fundamental type
      // mplr::communicator::irecv does not suffice in the cases above as the irecv performs a
      // probe first to receive STL containers
      T data_r;
      mplr::irequest r{comm_world.irecv(data_r, 0)};
      comm_world.barrier();
      while (not r.test()) {
      }
      return data_r == data;
    }
  } else
    comm_world.barrier();
  return true;
}


std::optional<mplr::environment> env;

BOOST_AUTO_TEST_CASE(isend_irecv) {
  if (not mplr::initialized())
    env.emplace();

  // integer types
  BOOST_TEST(isend_irecv_test(std::byte(77)));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(isend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(isend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(isend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(isend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(isend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(isend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(isend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(isend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(isend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(isend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(isend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(isend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(isend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(isend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(isend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(isend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(isend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(isend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(isend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(isend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(isend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(isend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(isend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(ibsend_irecv) {
  if (not mplr::initialized())
    env.emplace();

  // integer types
  BOOST_TEST(ibsend_irecv_test(std::byte(77)));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(ibsend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(ibsend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(ibsend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(ibsend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(ibsend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(ibsend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(ibsend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(ibsend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(ibsend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(ibsend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(ibsend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(ibsend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(ibsend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(ibsend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(ibsend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(ibsend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(ibsend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ibsend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ibsend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(ibsend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(ibsend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ibsend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ibsend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(issend_irecv) {
  if (not mplr::initialized())
    env.emplace();

  // integer types
  BOOST_TEST(issend_irecv_test(std::byte(77)));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(issend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(issend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(issend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(issend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(issend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(issend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(issend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(issend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(issend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(issend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(issend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(issend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(issend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(issend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(issend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(issend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(issend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(issend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(issend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(issend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(issend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(issend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(issend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(irsend_irecv) {
  if (not mplr::initialized())
    env.emplace();

  // integer types
  BOOST_TEST(irsend_irecv_test(std::byte(77)));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(irsend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(irsend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(irsend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(irsend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(irsend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(irsend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(irsend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(irsend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(irsend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(irsend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(irsend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(irsend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(irsend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(irsend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(irsend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(irsend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(irsend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(irsend_irecv_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(irsend_irecv_test(std::set<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(irsend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(irsend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(irsend_irecv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(irsend_irecv_iter_test(std::set<int>{1, 2, 3, 4, 5}));
}
