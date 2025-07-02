#define BOOST_TEST_MODULE communicator_barrier

#include <boost/test/included/unit_test.hpp>
#include <mplr/mplr.hpp>


bool barrier_test() {
  const auto comm_world = mplr::environment::comm_world();
  comm_world.barrier();
  return true;
}


bool ibarrier_test() {
  const auto comm_world = mplr::environment::comm_world();
  auto r{comm_world.ibarrier()};
  r.wait();
  return true;
}

std::optional<mplr::environment::environment> env;

BOOST_AUTO_TEST_CASE(barrier) {
  if (not mplr::environment::initialized())
    env.emplace();

  BOOST_TEST(barrier_test());
  BOOST_TEST(ibarrier_test());
}
