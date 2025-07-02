#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <array>
#include <vector>
// include MPLR header file
#include "mplr/mplr.hpp"


int main() {
  mplr::environment::environment env;
  // get a reference to communicator "world"
  const auto comm_world{mplr::environment::comm_world()};
  // generate some data
  using value_type = std::array<std::uint8_t, 2>;
  std::vector<value_type> vec;
  for (std::uint8_t i{0}; i < 16; ++i)
    vec.push_back(
        {static_cast<std::uint8_t>(i + 1), static_cast<std::uint8_t>(comm_world.rank() + 1)});
  // write data into file
  try {
    // wrap i/o operations in try-catch block, i/o operations may throw
    mplr::file file;
    // opening a file is collective over all processes within the employed communicator
    file.open(comm_world, "test.bin",
              mplr::file::access_mode::create | mplr::file::access_mode::read_write);
    // set file view
    file.set_view<value_type>("native");
    // write data
    mplr::vector_layout<value_type> write_layout(vec.size());
    file.write_at_all(vec.size() * comm_world.rank(), vec.data(), write_layout);
    // close file
    file.close();
  } catch (mplr::error &error) {
    std::cerr << error.what() << '\n';
  }
  return EXIT_SUCCESS;
}
