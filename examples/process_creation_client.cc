#include <cstdlib>
#include <iostream>
// include MPLR header file
#include "mplr/mplr.hpp"


int main(int argc, char *argv[]) {
  using namespace std::string_literals;

  // get a reference to communicator "world"
  [[maybe_unused]] const auto comm_world{mplr::environment::comm_world()};
  // get the parent inter-communicator
  auto &inter_comm{mplr::inter_communicator::parent()};
  std::cout << "Hello world! I am running on \"" << mplr::environment::processor_name()
            << "\". My rank is " << inter_comm.rank() << " out of " << inter_comm.size()
            << " processes.\n";
  std::cout << "commandline arguments: ";
  for (int i{0}; i < argc; ++i)
    std::cout << argv[i] << ' ';
  std::cout << std::endl;
  double message;
  inter_comm.bcast(0, message);
  std::cout << "got: " << message << '\n';
  return EXIT_SUCCESS;
}
