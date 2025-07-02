#define BOOST_TEST_MODULE communicator

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"


// test properties of the predefined communicator comm_word
bool communicator_comm_world_test() {
  const auto comm_world{mplr::comm_world()};
  if (not comm_world.is_valid())
    return false;
  const int size{comm_world.size()};
  const int rank{comm_world.rank()};
  if (size < 1)
    return false;
  if (rank < 0 or rank >= size)
    return false;
  const auto comm_world_2{mplr::comm_world()};
  if (comm_world_2 != comm_world)
    return false;
  const auto comm_self{mplr::comm_self()};
  if (comm_self == comm_world)
    return false;
  if (size == 1 and comm_self.compare(comm_world) != mplr::communicator::congruent)
    return false;
  if (size > 1 and comm_self.compare(comm_world) != mplr::communicator::unequal)
    return false;
  return true;
}


// test properties of a newly created communicator
bool communicator_comm_world_copy_test() {
  const auto comm_world{mplr::comm_world()};
  mplr::communicator comm_new{comm_world};
  if (not comm_new.is_valid())
    return false;
  if (comm_world.size() != comm_new.size())
    return false;
  if (comm_new.compare(comm_world) != mplr::communicator::congruent)
    return false;
  return true;
}


// test properties of a newly created communicator
bool communicator_comm_world_split_test() {
  const auto comm_world{mplr::comm_world()};
  const int size{comm_world.size()};
  const int rank{comm_world.rank()};
  mplr::communicator comm_new{mplr::communicator::split, comm_world, rank % 2 == 0};
  if (not comm_new.is_valid())
    return false;
  const int size_new{comm_new.size()};
  if (size % 2 == 0) {
    if (size_new != size / 2)
      return false;
  } else {
    if (rank % 2 == 0 && size_new != (size + 1) / 2)
      return false;
    if (rank % 2 == 1 && size_new != (size - 1) / 2)
      return false;
  }
  return true;
}


// test properties of a newly created communicator
bool communicator_comm_world_split_shared_memory_test() {
  const auto comm_world{mplr::comm_world()};
  const int rank{comm_world.rank()};
  mplr::communicator comm_new{mplr::communicator::split_shared_memory, comm_world,
                              rank % 2 == 0};
  if (not comm_new.is_valid())
    return false;
  return true;
}


bool communicator_comm_self_test() {
  const auto comm_self{mplr::comm_self()};
  if (not comm_self.is_valid())
    return false;
  const int size{comm_self.size()};
  const int rank{comm_self.rank()};
  if (size != 1)
    return false;
  if (rank != 0)
    return false;
  const auto comm_self_2{mplr::comm_self()};
  if (comm_self_2 != comm_self)
    return false;
  const auto comm_world{mplr::comm_world()};
  if (comm_self == comm_world)
    return false;
  if (comm_world.size() == 1 and comm_self.compare(comm_world) != mplr::communicator::congruent)
    return false;
  if (comm_world.size() > 1 and comm_self.compare(comm_world) != mplr::communicator::unequal)
    return false;
  return true;
}


std::optional<mplr::environment> env;

BOOST_AUTO_TEST_CASE(communicator) {
  if (not mplr::initialized())
    env.emplace();

  BOOST_TEST(communicator_comm_world_test());
  BOOST_TEST(communicator_comm_world_copy_test());
  BOOST_TEST(communicator_comm_world_split_test());
  BOOST_TEST(communicator_comm_world_split_shared_memory_test());
  BOOST_TEST(communicator_comm_self_test());
}
