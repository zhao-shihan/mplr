#define BOOST_TEST_MODULE info

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"


std::optional<mplr::environment> env;

BOOST_AUTO_TEST_CASE(info) {
  if (not mplr::initialized())
    env.emplace();

  [[maybe_unused]] const auto comm_world{mplr::comm_world()};

  mplr::info info_1;
  BOOST_TEST(info_1.size() == 0);
  info_1.set("Douglas Adams", "The Hitchhiker's Guide to the Galaxy");
  info_1.set("Isaac Asimov", "Nightfall");
  BOOST_TEST(info_1.size() == 2);
  BOOST_TEST(info_1.value("Isaac Asimov").value() == "Nightfall");

  mplr::info info_2{info_1};
  BOOST_TEST(info_1.size() == 2);
  BOOST_TEST(info_2.value("Isaac Asimov").value() == "Nightfall");

  BOOST_TEST(not info_2.value("no such thing"));
}
