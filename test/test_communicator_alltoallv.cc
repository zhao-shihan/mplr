#define BOOST_TEST_MODULE communicator_alltoallv

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"
#include "test_helper.hpp"


template<typename T>
bool alltoallv_with_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};  // number of elements to send to each process
  const int N_recv{(N_processes * N_processes + N_processes) /
                   2};  // total number of elements to receive
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  mplr::layouts<T> sendls;
  mplr::layouts<T> recvls;
  mplr::displacements senddispls;
  mplr::displacements recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mplr::vector_layout<T>(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mplr::vector_layout<T>(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  comm_world.alltoallv(send_data.data(), sendls, senddispls, recv_data.data(), recvls,
                       recvdispls);
  return recv_data == expected;
}


template<typename T>
bool alltoallv_with_displacements_contiguous_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};  // number of elements to send to each process
  const int N_recv{(N_processes * N_processes + N_processes) /
                   2};  // total number of elements to receive
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  mplr::contiguous_layouts<T> sendls;
  mplr::contiguous_layouts<T> recvls;
  mplr::displacements senddispls;
  mplr::displacements recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mplr::contiguous_layout<T>(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mplr::contiguous_layout<T>(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  comm_world.alltoallv(send_data.data(), sendls, senddispls, recv_data.data(), recvls,
                       recvdispls);
  return recv_data == expected;
}


template<typename T>
bool alltoallv_without_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};  // number of elements to send to each process
  const int N_recv{(N_processes * N_processes + N_processes) /
                   2};  // total number of elements to receive
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  mplr::layouts<T> sendls;
  mplr::layouts<T> recvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mplr::indexed_layout<T>({{N_send, j * N_send}}));
    recvls.push_back(mplr::indexed_layout<T>({{j + 1, (j * j + j) / 2}}));
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  comm_world.alltoallv(send_data.data(), sendls, recv_data.data(), recvls);
  return recv_data == expected;
}


template<typename T>
bool ialltoallv_with_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};  // number of elements to send to each process
  const int N_recv{(N_processes * N_processes + N_processes) /
                   2};  // total number of elements to receive
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  mplr::layouts<T> sendls;
  mplr::layouts<T> recvls;
  mplr::displacements senddispls;
  mplr::displacements recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mplr::vector_layout<T>(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mplr::vector_layout<T>(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  auto r{comm_world.ialltoallv(send_data.data(), sendls, senddispls, recv_data.data(), recvls,
                               recvdispls)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoallv_with_displacements_contiguous_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};  // number of elements to send to each process
  const int N_recv{(N_processes * N_processes + N_processes) /
                   2};  // total number of elements to receive
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  mplr::contiguous_layouts<T> sendls;
  mplr::contiguous_layouts<T> recvls;
  mplr::displacements senddispls;
  mplr::displacements recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mplr::contiguous_layout<T>(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mplr::contiguous_layout<T>(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  auto r{comm_world.ialltoallv(send_data.data(), sendls, senddispls, recv_data.data(), recvls,
                               recvdispls)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoallv_without_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};  // number of elements to send to each process
  const int N_recv{(N_processes * N_processes + N_processes) /
                   2};  // total number of elements to receive
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  mplr::layouts<T> sendls;
  mplr::layouts<T> recvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mplr::indexed_layout<T>({{N_send, j * N_send}}));
    recvls.push_back(mplr::indexed_layout<T>({{j + 1, (j * j + j) / 2}}));
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  auto r{comm_world.ialltoallv(send_data.data(), sendls, recv_data.data(), recvls)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool alltoallv_in_place_with_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  mplr::layouts<T> sendrecvls;
  mplr::displacements sendrecvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};  // must be symmetric in j and rank
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    sendrecvls.push_back(mplr::contiguous_layout<T>(N_sendrecv));
    sendrecvdispls.push_back(displ);
    displ += N_sendrecv;
    ++expected_val;
  }
  comm_world.alltoallv(sendrecv_data.data(), sendrecvls, sendrecvdispls);
  return sendrecv_data == expected;
}


template<typename T>
bool alltoallv_in_place_without_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  mplr::layouts<T> sendrecvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};  // must be symmetric in j and rank
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    sendrecvls.push_back(mplr::indexed_layout<T>({{N_sendrecv, displ}}));
    displ += N_sendrecv;
    ++expected_val;
  }
  comm_world.alltoallv(sendrecv_data.data(), sendrecvls);
  return sendrecv_data == expected;
}


template<typename T>
bool ialltoallv_in_place_with_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  mplr::layouts<T> sendrecvls;
  mplr::displacements sendrecvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};  // must be symmetric in j and rank
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    sendrecvls.push_back(mplr::contiguous_layout<T>(N_sendrecv));
    sendrecvdispls.push_back(displ);
    displ += N_sendrecv;
    ++expected_val;
  }
  auto r{comm_world.ialltoallv(sendrecv_data.data(), sendrecvls, sendrecvdispls)};
  r.wait();
  return sendrecv_data == expected;
}


template<typename T>
bool ialltoallv_in_place_without_displacements_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  mplr::layouts<T> sendrecvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};  // must be symmetric in j and rank
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    sendrecvls.push_back(mplr::indexed_layout<T>({{N_sendrecv, displ}}));
    displ += N_sendrecv;
    ++expected_val;
  }
  auto r{comm_world.ialltoallv(sendrecv_data.data(), sendrecvls)};
  r.wait();
  return sendrecv_data == expected;
}


BOOST_AUTO_TEST_CASE(alltoallv) {
  if (not mplr::initialized())
    mplr::init();

  BOOST_TEST(alltoallv_with_displacements_test(1.0));
  BOOST_TEST(alltoallv_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_with_displacements_contiguous_test(1.0));
  BOOST_TEST(alltoallv_with_displacements_contiguous_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_without_displacements_test(1.0));
  BOOST_TEST(alltoallv_without_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_with_displacements_test(1.0));
  BOOST_TEST(ialltoallv_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_with_displacements_contiguous_test(1.0));
  BOOST_TEST(ialltoallv_with_displacements_contiguous_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_without_displacements_test(1.0));
  BOOST_TEST(ialltoallv_without_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_in_place_with_displacements_test(1.0));
  BOOST_TEST(alltoallv_in_place_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_in_place_without_displacements_test(1.0));
  BOOST_TEST(alltoallv_in_place_without_displacements_test(tuple{1, 2.0}));

  // skip tests for older versions of MPICH due to a bug in MPICH's implementation of Alltoallw
#if !defined MPICH_NUMVERSION || MPICH_NUMVERSION > 40100000
  BOOST_TEST(ialltoallv_in_place_with_displacements_test(1.0));
  BOOST_TEST(ialltoallv_in_place_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_in_place_without_displacements_test(1.0));
  BOOST_TEST(ialltoallv_in_place_without_displacements_test(tuple{1, 2.0}));
#endif
}
