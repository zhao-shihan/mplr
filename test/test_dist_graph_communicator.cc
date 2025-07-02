#define BOOST_TEST_MODULE dist_graph_communicator

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"


bool dist_graph_communicator_test() {
  const auto comm_world{mplr::environment::comm_world()};
  const int size{comm_world.size()};
  const int rank{comm_world.rank()};
  mplr::distributed_graph_communicator::neighbours_set sources;
  mplr::distributed_graph_communicator::neighbours_set destination;
  if (rank == 0) {
    for (int i{1}; i < size; ++i) {
      sources.add(i);
      destination.add({i, 0});
    }
  } else {
    sources.add(0);
    destination.add({0, 0});
  }
  mplr::distributed_graph_communicator comm_g(comm_world, sources, destination);
  if (rank == 0) {
    if (comm_g.in_degree() != comm_g.size() - 1)
      return false;
    if (comm_g.out_degree() != comm_g.size() - 1)
      return false;
  } else {
    if (comm_g.in_degree() != 1)
      return false;
    if (comm_g.out_degree() != 1)
      return false;
  }
  return true;
}


std::optional<mplr::environment::environment> env;

BOOST_AUTO_TEST_CASE(dist_graph_communicator) {
  if (not mplr::environment::initialized())
    env.emplace();

  BOOST_TEST(dist_graph_communicator_test());
}
