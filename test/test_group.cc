#define BOOST_TEST_MODULE group

#include <boost/test/included/unit_test.hpp>
#include <mplr/mplr.hpp>


std::optional<mplr::environment::environment> env;

BOOST_AUTO_TEST_CASE(group) {
  if (not mplr::environment::initialized())
    env.emplace();

  const auto comm_world{mplr::environment::comm_world()};
  const auto comm_self{mplr::environment::comm_self()};

  mplr::group group_world{comm_world};
  mplr::group group_self{comm_self};

  BOOST_TEST((group_world.size() == comm_world.size()));
  BOOST_TEST((group_world.rank() == comm_world.rank()));
  BOOST_TEST((group_self.size() == comm_self.size()));

  mplr::group group_world_copy{group_world};
  BOOST_TEST((group_world == group_world_copy));

  if (comm_world.size() > 1)
    BOOST_TEST((group_world != group_self));
  else
    BOOST_TEST((group_world == group_self));
  if (comm_world.size() > 1)
    BOOST_TEST((group_world.compare(group_self) == mplr::group::unequal));
  else
    BOOST_TEST((group_world.compare(group_self) == mplr::group::identical));

  BOOST_TEST((group_self.translate(0, group_world) == group_world.rank()));

  mplr::group group_union(mplr::group::Union, group_world, group_self);
  mplr::group group_intersection(mplr::group::intersection, group_world, group_self);
  mplr::group group_difference(mplr::group::difference, group_world, group_self);
  mplr::group group_with_0(mplr::group::include, group_world, {0});
  mplr::group group_without_0(mplr::group::exclude, group_world, {0});

  BOOST_TEST((group_union.size() == group_world.size()));
  BOOST_TEST((group_intersection.size() == 1));
  BOOST_TEST((group_difference.size() == group_world.size() - 1));
  BOOST_TEST((group_with_0.size() == 1));
  BOOST_TEST((group_without_0.size() == group_world.size() - 1));
}
