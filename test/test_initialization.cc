#define BOOST_TEST_MODULE communicator

#include <boost/test/included/unit_test.hpp>
#include <mpl/mpl.hpp>


bool initialization_test() {
  mpl::environment::environment{};
  // Do some MPL stuff.
  return mpl::environment::comm_world().is_valid() and mpl::environment::comm_self().is_valid();
}


BOOST_AUTO_TEST_CASE(initialization) {
  BOOST_TEST(initialization_test());
}
