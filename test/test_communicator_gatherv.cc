#define BOOST_TEST_MODULE communicator_gatherv

#include "boost/test/included/unit_test.hpp"
#include "mplr/mplr.hpp"
#include "test_helper.hpp"

#include <tuple>


template<use_non_root_overload variant, typename T>
bool gatherv_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mplr::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mplr::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mplr::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts);
    else
      comm_world.gatherv(0, v_send.data(), layout);
  } else {
    comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool gatherv_contiguous_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mplr::contiguous_layouts<T> layouts;
  mplr::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mplr::contiguous_layout<T>(i + 1));
    displacements.push_back(offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mplr::contiguous_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements);
    else
      comm_world.gatherv(0, v_send.data(), layout);
  } else {
    comm_world.gatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool igatherv_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mplr::layouts<T> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mplr::indexed_layout<T>({{i + 1, offset}}));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mplr::vector_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts)};
      r.wait();
    } else {
      auto r{comm_world.igatherv(0, v_send.data(), layout)};
      r.wait();
    }
  } else {
    auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool igatherv_contiguous_test(const T &val) {
  const auto comm_world{mplr::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  mplr::contiguous_layouts<T> layouts;
  mplr::displacements displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mplr::contiguous_layout<T>(i + 1));
    displacements.push_back(offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const mplr::contiguous_layout<T> layout(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts,
                                 displacements)};
      r.wait();
    } else {
      auto r{comm_world.igatherv(0, v_send.data(), layout)};
      r.wait();
    }
  } else {
    auto r{
        comm_world.igatherv(0, v_send.data(), layout, v_gather.data(), layouts, displacements)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


BOOST_AUTO_TEST_CASE(gatherv) {
  if (not mplr::initialized())
    mplr::init();

  BOOST_TEST(gatherv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(gatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(gatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));

  // skip tests for older versions of MPICH due to a bug in MPICH's implementation of Alltoallw
#if (defined MPICH and MPICH_NUMVERSION >= 40101000) || defined OPENMPI
  BOOST_TEST(igatherv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(igatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(igatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));
#endif

  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));
}
