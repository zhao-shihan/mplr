#include <cstdlib>
#include <iostream>
#include <vector>
#include <mplr/mplr.hpp>


int main() {
  mplr::environment::environment env;
  const auto comm_world{mplr::environment::comm_world()};
  double x{1.23456 + comm_world.rank()};
  mplr::irequest r_send(comm_world.isend(x, 0));  // nonblocking send to rank 0
  // rank 0 receives data from all ranks
  if (comm_world.rank() == 0) {
    std::vector<double> v(comm_world.size());
    mplr::irequest_pool r_pool;
    for (int i{0}; i < comm_world.size(); ++i)
      r_pool.push(comm_world.irecv(v[i], i));
    r_pool.waitall();  // wait to finish all receive operations
    for (int i{0}; i < comm_world.size(); ++i)
      std::cout << i << '\t' << v[i] << '\n';
  }
  r_send.wait();  // wait to finish send operation
  return EXIT_SUCCESS;
}
