#define BOOST_TEST_MODULE communicator

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"


bool initialization_test() {
  mplr::environment{};
  // Do some MPLR stuff.
  return mplr::comm_world().is_valid() and
         mplr::comm_self().is_valid();
}


BOOST_AUTO_TEST_CASE(initialization) {
  BOOST_TEST(initialization_test());
}
