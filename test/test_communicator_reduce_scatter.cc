#define BOOST_TEST_MODULE communicator_reduce_scatter

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"
#include "test_helper.hpp"


template<typename F, typename T>
bool reduce_scatter_test(F f, const T &val) {
  const auto comm_world{mplr::comm_world()};
  T x{val};
  std::vector<T> v_x;
  mplr::contiguous_layouts<T> l;
  for (int i{0}; i < comm_world.size(); ++i) {
    const int block_size{i + 1};
    for (int j{0}; j < block_size; ++j)
      v_x.push_back(x);
    l.push_back(mplr::contiguous_layout<T>(block_size));
    ++x;
  }
  const int block_size{comm_world.rank() + 1};
  std::vector<T> v_y(block_size);
  comm_world.reduce_scatter(f, v_x.data(), v_y.data(), l);
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T expected{x};
  for (int i{1}; i < comm_world.size(); ++i)
    expected = f(expected, x);
  std::vector<T> v_expected(block_size, expected);
  return v_y == v_expected;
}


template<typename F, typename T>
bool ireduce_scatter_test(F f, const T &val) {
  const auto comm_world{mplr::comm_world()};
  T x{val};
  std::vector<T> v_x;
  mplr::contiguous_layouts<T> l;
  for (int i{0}; i < comm_world.size(); ++i) {
    const int block_size{i + 1};
    for (int j{0}; j < block_size; ++j)
      v_x.push_back(x);
    l.push_back(mplr::contiguous_layout<T>(block_size));
    ++x;
  }
  const int block_size{comm_world.rank() + 1};
  std::vector<T> v_y(block_size);
  auto r{comm_world.ireduce_scatter(f, v_x.data(), v_y.data(), l)};
  x = val;
  for (int i{0}; i < comm_world.rank(); ++i)
    ++x;
  T expected{x};
  for (int i{1}; i < comm_world.size(); ++i)
    expected = f(expected, x);
  std::vector<T> v_expected(block_size, expected);
  r.wait();
  return v_y == v_expected;
}



BOOST_AUTO_TEST_CASE(reduce_scatter) {
  if (not mplr::initialized())
    mplr::init();

  BOOST_TEST(reduce_scatter_test(add<double>(), 1.0));
  BOOST_TEST(reduce_scatter_test(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_scatter_test(mplr::plus<double>(), 1.0));
  BOOST_TEST(reduce_scatter_test(mplr::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(reduce_scatter_test([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(reduce_scatter_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));

  BOOST_TEST(ireduce_scatter_test(add<double>(), 1.0));
  BOOST_TEST(ireduce_scatter_test(add<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_scatter_test(mplr::plus<double>(), 1.0));
  BOOST_TEST(ireduce_scatter_test(mplr::plus<tuple>(), tuple{1, 2.0}));
  BOOST_TEST(ireduce_scatter_test([](auto a, auto b) { return a + b; }, 1.0));
  BOOST_TEST(ireduce_scatter_test([](auto a, auto b) { return a + b; }, tuple{1, 2.0}));
}
