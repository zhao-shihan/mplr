#define BOOST_TEST_MODULE graph_communicator

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"


bool graph_communicator_test() {
  const auto comm_world{mplr::comm_world()};
  const int size{comm_world.size()};
  mplr::graph_communicator::edge_set es;
  for (int i{1}; i < size; ++i) {
    es.add({0, i});
    es.add({i, 0});
  }
  mplr::graph_communicator comm_g(comm_world, es);
  if (comm_g.degree(0) != comm_g.size() - 1)
    return false;
  const auto nl_0{comm_g.neighbors(0)};
  if (nl_0.size() != comm_g.size() - 1)
    return false;
  const auto nl_1{comm_g.neighbors(1)};
  if (nl_1.size() != 1)
    return false;
  return true;
}


bool graph_communicator_test_2() {
  const auto comm_world{mplr::comm_world()};
  const int size{comm_world.size()};
  const int rank{comm_world.rank()};
  if (size >= 4) {
    mplr::communicator communicator_4{mplr::communicator::split, comm_world,
                                      rank < 4 ? 0 : rank};
    if (communicator_4.size() < 4)
      return true;
    mplr::graph_communicator::edge_set es{{0, 1}, {0, 3}, {1, 0}, {2, 3}, {3, 0}, {3, 2}};
    mplr::graph_communicator comm_g(communicator_4, es);
    if (comm_g.degree(0) != 2)
      return false;
    if (comm_g.degree(1) != 1)
      return false;
    if (comm_g.degree(2) != 1)
      return false;
    if (comm_g.degree(3) != 2)
      return false;
    if (comm_g.neighbors(0) != mplr::graph_communicator::node_list{1, 3})
      return false;
    if (comm_g.neighbors(1) != mplr::graph_communicator::node_list{0})
      return false;
    if (comm_g.neighbors(2) != mplr::graph_communicator::node_list{3})
      return false;
    if (comm_g.neighbors(3) != mplr::graph_communicator::node_list{0, 2})
      return false;
  }
  return true;
}



BOOST_AUTO_TEST_CASE(graph_communicator) {
  if (not mplr::initialized())
    mplr::init();

  BOOST_TEST(graph_communicator_test());
  BOOST_TEST(graph_communicator_test_2());
}
