#define BOOST_TEST_MODULE cartesian_communicator

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"
#include "test_helper.hpp"


bool cartesian_communicator_test() {
  const auto comm_world{mplr::comm_world()};
  mplr::cartesian_communicator::dimensions dimensions{
      mplr::cartesian_communicator::periodic, mplr::cartesian_communicator::non_periodic};
  mplr::cartesian_communicator comm_c{comm_world,
                                      mplr::dims_create(comm_world.size(), dimensions)};
  if (comm_c.dimensionality() != 2)
    return false;
  const int rank{comm_c.rank()};
  auto coordinate{comm_c.coordinates()};
  if (comm_c.rank(coordinate) != rank)
    return false;
  auto dims{comm_c.get_dimensions()};
  if (dims.size(0) * dims.size(1) != comm_c.size())
    return false;
  if (not(dims.periodicity(0) == mplr::cartesian_communicator::periodic and
          dims.periodicity(1) == mplr::cartesian_communicator::non_periodic))
    return false;
  auto ranks{comm_c.shift(0, 1)};
  ++coordinate[0];
  if (coordinate[0] >= dims.size(0))
    coordinate[0] = 0;
  int destination_1{comm_c.rank(coordinate)};
  coordinate[0] -= 2;
  if (coordinate[0] < 0)
    coordinate[0] += dims.size(0);
  int source_1{comm_c.rank(coordinate)};
  if (not(ranks.source == source_1 and ranks.destination == destination_1))
    return false;
  {
    double x{1};
    std::vector<double> y(4, 0.);
    comm_c.neighbor_allgather(x, y.data());
    if ((y[0] != 0 and y[0] != 1) or (y[1] != 0 and y[1] != 1) or (y[2] != 0 and y[2] != 1) or
        (y[3] != 0 and y[3] != 1))
      return false;
  }
  {
    std::vector<double> x(4, rank + 1.0);
    std::vector<double> y(4, 0.0);
    comm_c.neighbor_alltoall(x.data(), y.data());
    auto ranks_0{comm_c.shift(0, 1)};
    auto ranks_1{comm_c.shift(1, 1)};
    if (ranks_0.source != mplr::proc_null and y[0] != ranks_0.source + 1.)
      return false;
    if (ranks_0.destination != mplr::proc_null and y[1] != ranks_0.destination + 1.)
      return false;
    if (ranks_1.source != mplr::proc_null and y[2] != ranks_1.source + 1.)
      return false;
    if (ranks_1.destination != mplr::proc_null and y[3] != ranks_1.destination + 1.)
      return false;
  }
  {
    std::vector<double> x(4, rank + 1.0);
    std::vector<double> y(4, 0.0);
    mplr::layouts<double> ls;
    ls.push_back(mplr::indexed_layout<double>({{1, 0}}));
    ls.push_back(mplr::indexed_layout<double>({{1, 1}}));
    ls.push_back(mplr::indexed_layout<double>({{1, 2}}));
    ls.push_back(mplr::indexed_layout<double>({{1, 3}}));
    comm_c.neighbor_alltoallv(x.data(), ls, y.data(), ls);
    auto ranks_0{comm_c.shift(0, 1)};
    auto ranks_1{comm_c.shift(1, 1)};
    if (ranks_0.source != mplr::proc_null and y[0] != ranks_0.source + 1.)
      return false;
    if (ranks_0.destination != mplr::proc_null and y[1] != ranks_0.destination + 1.)
      return false;
    if (ranks_1.source != mplr::proc_null and y[2] != ranks_1.source + 1.)
      return false;
    if (ranks_1.destination != mplr::proc_null and y[3] != ranks_1.destination + 1.)
      return false;
  }
  return true;
}


template<typename T>
bool cartesian_communicator_neighbor_alltoall_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  mplr::cartesian_communicator::dimensions dimensions{mplr::cartesian_communicator::periodic};
  mplr::cartesian_communicator comm_c{comm_world,
                                      mplr::dims_create(comm_world.size(), dimensions)};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2, send_val);
  std::vector<T> recv_data(2);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  comm_c.neighbor_alltoall(send_data.data(), recv_data.data());
  return recv_data == expected;
}


template<typename T>
bool cartesian_communicator_neighbor_alltoall_layout_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  mplr::cartesian_communicator::dimensions dimensions{mplr::cartesian_communicator::periodic};
  mplr::cartesian_communicator comm_c{comm_world,
                                      mplr::dims_create(comm_world.size(), dimensions)};
  const int vector_size{3};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2 * vector_size, send_val);
  std::vector<T> recv_data(2 * vector_size);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  mplr::vector_layout<T> sendrecvl(vector_size);
  comm_c.neighbor_alltoall(send_data.data(), sendrecvl, recv_data.data(), sendrecvl);
  return recv_data == expected;
}


template<typename T>
bool cartesian_communicator_ineighbor_alltoall_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  mplr::cartesian_communicator::dimensions dimensions{mplr::cartesian_communicator::periodic};
  mplr::cartesian_communicator comm_c{comm_world,
                                      mplr::dims_create(comm_world.size(), dimensions)};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2, send_val);
  std::vector<T> recv_data(2);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  auto r{comm_c.ineighbor_alltoall(send_data.data(), recv_data.data())};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool cartesian_communicator_ineighbor_alltoall_layout_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  mplr::cartesian_communicator::dimensions dimensions{mplr::cartesian_communicator::periodic};
  mplr::cartesian_communicator comm_c{comm_world,
                                      mplr::dims_create(comm_world.size(), dimensions)};
  const int vector_size{3};
  T send_val{val};
  for (int i{0}; i < comm_c.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(2 * vector_size, send_val);
  std::vector<T> recv_data(2 * vector_size);
  T expected_val{val};
  std::vector<T> expected_values;
  for (int i{0}; i < comm_c.size(); ++i) {
    expected_values.push_back(expected_val);
    ++expected_val;
  }
  std::vector<T> expected;
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() - 1 + comm_c.size()) % comm_c.size()]);
  for (int j{0}; j < vector_size; ++j)
    expected.push_back(expected_values[(comm_c.rank() + 1) % comm_c.size()]);
  mplr::vector_layout<T> sendrecvl(vector_size);
  auto r{comm_c.ineighbor_alltoall(send_data.data(), sendrecvl, recv_data.data(), sendrecvl)};
  r.wait();
  return recv_data == expected;
}



BOOST_AUTO_TEST_CASE(cartesian_communicator) {
  if (not mplr::initialized())
    mplr::init();

  BOOST_TEST(cartesian_communicator_test());
}


BOOST_AUTO_TEST_CASE(cartesian_communicator_vector) {
  if (not mplr::initialized())
    mplr::init();

  mplr::cartesian_communicator::vector vector{1, 2, 3, 4, 5};
  BOOST_TEST(vector.dimensions() == 5);
  vector.add(6);
  BOOST_TEST(vector.dimensions() == 6);
  BOOST_TEST((std::accumulate(vector.begin(), vector.end(), 0) == 21));
  BOOST_TEST((std::accumulate(vector.cbegin(), vector.cend(), 0) == 21));
}


BOOST_AUTO_TEST_CASE(cartesian_communicator_include_tags) {
  if (not mplr::initialized())
    mplr::init();

  const auto included = mplr::cartesian_communicator::included;
  const auto excluded = mplr::cartesian_communicator::excluded;
  mplr::cartesian_communicator::included_tags is_included{10};
  BOOST_TEST(is_included.size() == 10);
  is_included.add(included);
  is_included.add(excluded);
  BOOST_TEST(is_included.size() == 12);
  BOOST_TEST(
      (std::find(is_included.begin(), is_included.end(), included) != is_included.end()));
}


BOOST_AUTO_TEST_CASE(cartesian_communicator_dimensions) {
  if (not mplr::initialized())
    mplr::init();

  mplr::cartesian_communicator::dimensions dimensions{
      mplr::cartesian_communicator::periodic, mplr::cartesian_communicator::non_periodic,
      mplr::cartesian_communicator::non_periodic};

  BOOST_TEST(dimensions.dimensionality() == 3);
  BOOST_TEST(dimensions.periodicity(0) == mplr::cartesian_communicator::periodic);
  BOOST_TEST(dimensions.periodicity(1) == mplr::cartesian_communicator::non_periodic);
  BOOST_TEST(dimensions.periodicity(2) == mplr::cartesian_communicator::non_periodic);
  dimensions[1] = {10, mplr::cartesian_communicator::periodic};
  BOOST_TEST(dimensions.periodicity(1) == mplr::cartesian_communicator::periodic);
  BOOST_TEST(dimensions.size(1) == 10);
  dimensions.add(11, mplr::cartesian_communicator::non_periodic);
  BOOST_TEST(dimensions.periodicity(3) == mplr::cartesian_communicator::non_periodic);
  BOOST_TEST(dimensions.size(3) == 11);
  BOOST_TEST((std::find(dimensions.begin(), dimensions.end(),
                        std::tuple{11, mplr::cartesian_communicator::non_periodic}) !=
              dimensions.end()));
}

BOOST_AUTO_TEST_CASE(cartesian_communicator_neighbor_alltoall) {
  if (not mplr::initialized())
    mplr::init();

  BOOST_TEST(cartesian_communicator_neighbor_alltoall_test(1.0));
  BOOST_TEST(cartesian_communicator_neighbor_alltoall_test(tuple{1, 2.0}));

  BOOST_TEST(cartesian_communicator_neighbor_alltoall_layout_test(1.0));
  BOOST_TEST(cartesian_communicator_neighbor_alltoall_layout_test(tuple{1, 2.0}));

  BOOST_TEST(cartesian_communicator_ineighbor_alltoall_test(1.0));
  BOOST_TEST(cartesian_communicator_ineighbor_alltoall_test(tuple{1, 2.0}));

  BOOST_TEST(cartesian_communicator_ineighbor_alltoall_layout_test(1.0));
  BOOST_TEST(cartesian_communicator_ineighbor_alltoall_layout_test(tuple{1, 2.0}));
}
