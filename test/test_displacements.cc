#define BOOST_TEST_MODULE displacements

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"


BOOST_AUTO_TEST_CASE(displacements) {
  if (not mplr::initialized())
    mplr::init();

  mplr::displacements displacements_1(10);
  mplr::displacements displacements_2{1, 2, 3};
  displacements_2.push_back(10);
  BOOST_TEST(displacements_1.size() == 10);
  BOOST_TEST(displacements_2.size() == 4);
}
