#include <cstdlib>
#include <iostream>
// include MPLR header file
#include "mplr/mplr.hpp"


int main() {
  mplr::environment::environment env;
  using namespace std::string_literals;
  // get a reference to communicator "world"
  const auto comm_world{mplr::environment::comm_world()};
  // spawn 2 new processes
  mplr::info info;
  info.set("host", "localhost");
  auto inter_comm{comm_world.spawn(0, 2, {"./process_creation_client"s}, info)};
  // broadcast a message to the created processes
  double message;
  if (comm_world.rank() == 0) {
    // root rank
    message = 1.23;
    inter_comm.bcast(mplr::root, message);
  } else
    // non-root ranks
    inter_comm.bcast(mplr::proc_null, message);

  return EXIT_SUCCESS;
}
