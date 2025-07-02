#if !(defined MPLR_HPP)

#define MPLR_HPP

#include <mpi.h>
#include <cstddef>


namespace mplr {

  /// Wildcard value to indicate in a receive operation, e.g., \c communicator::recv, that any
  /// source is acceptable.
  /// \see \c tag_t::any
  constexpr int any_source = MPI_ANY_SOURCE;

  /// Special value that can be used instead of a rank wherever a source or a  destination
  /// argument is required in a call to indicate that the communication shall have no effect.
  constexpr int proc_null = MPI_PROC_NULL;

  /// Special value that is used to indicate an invalid return value or function
  /// parameter in some functions.
  constexpr int undefined = MPI_UNDEFINED;

  /// Special value to indicate the root process in some inter-communicator collective
  /// operations.
  constexpr int root = MPI_ROOT;

  /// Special constant to indicate the start of the address range of message buffers.
  /// \anchor absolute
  constexpr void *absolute = MPI_BOTTOM;

  /// Special constant representing an upper bound on the additional space consumed when
  /// buffering messages.
  /// \see \c communicator::bsend
  /// \anchor bsend_overhead
  constexpr int bsend_overhead = MPI_BSEND_OVERHEAD;

  /// Unsigned integer type used for array indexing and address arithmetic.
  using size_t = std::size_t;

  /// Signed integer type used for array indexing and address arithmetic.
  using ssize_t = std::ptrdiff_t;

}  // namespace mplr

#include <mplr/error.hpp>
#include <mplr/displacements.hpp>
#include <mplr/tag.hpp>
#include <mplr/ranks.hpp>
#include <mplr/flat_memory.hpp>
#include <mplr/datatype.hpp>
#include <mplr/layout.hpp>
#include <mplr/status.hpp>
#include <mplr/message.hpp>
#include <mplr/operator.hpp>
#include <mplr/request.hpp>
#include <mplr/info.hpp>
#include <mplr/comm_group.hpp>
#include <mplr/environment.hpp>
#include <mplr/topology_communicator.hpp>
#include <mplr/cartesian_communicator.hpp>
#include <mplr/graph_communicator.hpp>
#include <mplr/file.hpp>
#include <mplr/distributed_graph_communicator.hpp>
#include <mplr/distributed_grid.hpp>

#endif
