#define BOOST_TEST_MODULE communicator_alltoall

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"
#include "test_helper.hpp"


template<typename T>
bool alltoall_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(comm_world.size(), send_val);
  std::vector<T> recv_data(comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  comm_world.alltoall(send_data.data(), recv_data.data());
  return recv_data == expected;
}


template<typename T>
bool alltoall_layout_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(3 * comm_world.size(), send_val);
  std::vector<T> recv_data(2 * comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    expected.push_back(expected_val);
    ++expected_val;
  }
  mplr::indexed_layout<T> sendl({{1, 0}, {1, 2}});
  mplr::vector_layout<T> recvl(2);
  comm_world.alltoall(send_data.data(), sendl, recv_data.data(), recvl);
  return recv_data == expected;
}


template<typename T>
bool alltoall_inplace_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> sendrecv_data(comm_world.size(), send_val);
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  comm_world.alltoall(sendrecv_data.data());
  return sendrecv_data == expected;
}


template<typename T>
bool ialltoall_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(comm_world.size(), send_val);
  std::vector<T> recv_data(comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  auto r{comm_world.ialltoall(send_data.data(), recv_data.data())};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoall_layout_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(3 * comm_world.size(), send_val);
  std::vector<T> recv_data(2 * comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    expected.push_back(expected_val);
    ++expected_val;
  }
  mplr::indexed_layout<T> sendl({{1, 0}, {1, 2}});
  mplr::vector_layout<T> recvl(2);
  auto r{comm_world.ialltoall(send_data.data(), sendl, recv_data.data(), recvl)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoall_inplace_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> sendrecv_data(comm_world.size(), send_val);
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  auto r{comm_world.ialltoall(sendrecv_data.data())};
  r.wait();
  return sendrecv_data == expected;
}

std::optional<mplr::environment> env;

BOOST_AUTO_TEST_CASE(alltoall) {
  if (not mplr::initialized())
    env.emplace();

  BOOST_TEST(alltoall_test(1.0));
  BOOST_TEST(alltoall_test(tuple{1, 2.0}));

  BOOST_TEST(alltoall_layout_test(1.0));
  BOOST_TEST(alltoall_layout_test(tuple{1, 2.0}));

  BOOST_TEST(alltoall_inplace_test(1.0));
  BOOST_TEST(alltoall_inplace_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoall_test(1.0));
  BOOST_TEST(ialltoall_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoall_layout_test(1.0));
  BOOST_TEST(ialltoall_layout_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoall_inplace_test(1.0));
  BOOST_TEST(ialltoall_inplace_test(tuple{1, 2.0}));
}
