#include "mplr/mplr.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>
#include <numeric>
#include <set>
#include <vector>


int main() {
  mplr::environment::environment env;
  const auto comm_world{mplr::environment::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a single vector
  {
    const int n{10};
    std::vector<double> l(n);
    if (comm_world.rank() == 0) {
      std::iota(begin(l), end(l), 1);
      comm_world.send(begin(l), end(l), 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(begin(l), end(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  // send / receive a single list
  {
    const int n{10};
    std::list<double> l(n);
    if (comm_world.rank() == 0) {
      std::iota(begin(l), end(l), 1);
      comm_world.send(begin(l), end(l), 1);
    }
    if (comm_world.rank() == 1) {
      comm_world.recv(begin(l), end(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  // send a set / receive an array
  {
    const int n{10};
    if (comm_world.rank() == 0) {
      std::set<double> s;
      for (int i{1}; i <= n; ++i)
        s.insert(i);
      comm_world.send(s.begin(), s.end(), 1);
    }
    if (comm_world.rank() == 1) {
      std::array<double, n> l;
      comm_world.recv(begin(l), end(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  return EXIT_SUCCESS;
}
