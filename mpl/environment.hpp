#if !(defined MPL_ENVIRONMENT_HPP)

#define MPL_ENVIRONMENT_HPP

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <mpi.h>


namespace mpl {

  /// Represents the various levels of thread support that the underlying MPI
  /// implementation may provide.
  enum class threading_modes : int {
    /// the application is single-threaded
    single = MPI_THREAD_SINGLE,
    /// the application is multi-threaded, however, all MPL calls will be issued from the main
    /// thread only
    funneled = MPI_THREAD_FUNNELED,
    /// the application is multi-threaded and any thread may issue MPL calls, however,
    /// different threads will never issue MPL calls at the same time
    serialized = MPI_THREAD_SERIALIZED,
    /// the application is multi-threaded, any thread may issue MPI calls and different threads
    /// may issue MPL calls at the same time
    multiple = MPI_THREAD_MULTIPLE
  };

  namespace environment {

    /// @brief This routine may be used to determine whether MPI_INIT or MPI_INIT_THREAD has been called.
    /// @return true if MPI_INIT or MPI_INIT_THREAD has been called.
    inline bool initialized() {
      int flag;
      MPI_Initialized(&flag);
      return flag;
    }

    /// @brief This routine returns true if MPI_FINALIZE has completed.
    /// @return true if MPI_FINALIZE has completed.
    inline bool finalized() {
      int flag;
      MPI_Finalized(&flag);
      return flag;
    }

    /// @brief This routine returns true if MPI_INIT or MPI_INIT_THREAD
    /// has been called and MPI_FINALIZE has not completed.
    /// @return true if MPI_INIT or MPI_INIT_THREAD
    /// has been called and MPI_FINALIZE has not completed.
    inline bool available() {
      return initialized() and not finalized();
    }

    /// @brief Get MPI standard version.
    /// @return (version, subversion)
    inline std::pair<int, int> get_version() {
      int version;
      int subversion;
      MPI_Get_version(&version, &subversion);
      return {version, subversion};
    }

    /// @brief This routine returns a string representing the version of the MPI library.
    /// @return library information string
    inline std::string get_library_version() {
      char lib_version[MPI_MAX_LIBRARY_VERSION_STRING + 1];
      int len;
      MPI_Get_processor_name(lib_version, &len);
      lib_version[std::min(len, MPI_MAX_LIBRARY_VERSION_STRING)] = '\0';
      return lib_version;
    }

    //----------------------------------------------------------------

    /// @brief Initialize and finalize MPI in an RAII style.
    class environment {
    public:
      environment(int &argc, char **&argv,
                  mpl::threading_modes threading_mode = mpl::threading_modes::multiple) {
        int _;
        MPI_Init_thread(&argc, &argv, static_cast<int>(threading_mode), &_);
      }

      environment(mpl::threading_modes threading_mode = mpl::threading_modes::multiple) {
        int _;
        MPI_Init_thread(nullptr, nullptr, static_cast<int>(threading_mode), &_);
      }

      ~environment() {
        MPI_Finalize();
      }

      environment(const environment &) = delete;
      auto &operator=(const environment &) = delete;
    };

    //------------------------------------------------------------------

    /// Provides access to a predefined communicator that allows communication with
    /// all processes.
    /// \return communicator to communicate with any other process
    inline mpi_communicator comm_world() {
      return mpi_communicator{MPI_COMM_WORLD};
    }

    /// Provides access to a predefined communicator that includes only the calling
    /// process itself.
    /// \return communicator including only the process itself
    inline mpi_communicator comm_self() {
      return mpi_communicator{MPI_COMM_SELF};
    }

    /// Determines the highest level of thread support that is provided by the underlying
    /// MPI implementation.
    /// \return supported threading level
    inline threading_modes query_thread() {
      int threading_mode;
      MPI_Query_thread(&threading_mode);
      return static_cast<threading_modes>(threading_mode);
    }

    /// Determines if the current thread is the main thread, i.e., the thread that has
    /// initialized the MPI environment of the underlying MPI implementation.
    /// \return true if the current thread is the main thread
    inline bool is_thread_main() {
      int res;
      MPI_Is_thread_main(&res);
      return static_cast<bool>(res);
    }

    /// Determines if time values given by <tt>\ref wtime</tt> are synchronized with each other
    /// for all processes of the communicator given in <tt>\ref comm_world</tt>.
    /// \return true if times are synchronized
    /// \see <tt>\ref wtime</tt>
    inline bool wtime_is_global() {
      void *val;
      int flag;
      MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &val, &flag);
      return *static_cast<int *>(val);
    }

    /// Gives a unique specifier, the processor name, for the actual (physical) node.
    /// \return name of the node
    /// \note The name is determined by the underlying MPI implementation, i.e., it is
    /// implementation defined and may be different for different MPI implementations.
    inline std::string processor_name() {
      char name[MPI_MAX_PROCESSOR_NAME + 1];
      int len;
      MPI_Get_processor_name(name, &len);
      name[std::min(len, MPI_MAX_PROCESSOR_NAME)] = '\0';
      return name;
    }

    /// Get time.
    /// \return number of seconds of elapsed wall-clock time since some time in the past
    inline double wtime() {
      return MPI_Wtime();
    }

    /// Get resolution of time given by \c wtime.
    /// \return resolution of \c wtime in seconds.
    /// \see \c wtime
    inline double wtick() {
      return MPI_Wtick();
    }

    /// Provides to MPL a buffer in the user's memory to be used for buffering outgoing
    /// messages.
    /// \param buff pointer to user-provided buffer
    /// \param size size of the buffer in bytes, must be non-negative
    /// \see \c buffer_detach
    inline void buffer_attach(void *buff, int size) {
      MPI_Buffer_attach(buff, size);
    }

    /// Detach the buffer currently associated with MPL.
    /// \return pair representing the buffer location and size, i.e., the parameters provided to
    /// <tt>\ref buffer_attach</tt>
    /// \see \c buffer_attach
    inline std::pair<void *, int> buffer_detach() {
      void *buff;
      int size;
      MPI_Buffer_detach(&buff, &size);
      return {buff, size};
    }

  }  // namespace environment

  //--------------------------------------------------------------------

  /// Buffer manager for buffered send operations.
  /// \note There must be not more than one instance of the class \c bsend_buffer at any time
  /// per process.
  class bsend_buffer {
    void *buff_;

  public:
    /// deleted default constructor
    bsend_buffer() = delete;

    /// deleted copy constructor
    /// \param other buffer manager to copy from
    bsend_buffer(const bsend_buffer &other) = delete;

    /// deleted move constructor
    /// \param other buffer manager to move from
    bsend_buffer(bsend_buffer &&other) = delete;

    /// allocates buffer with specific size using a default-constructed allocator
    /// \param size buffer size in bytes
    /// \note The size given should be the sum of the sizes of all outstanding buffered send
    /// operations that will be sent during the lifetime of the \c bsend_buffer object, plus
    /// <tt>\ref bsend_overhead</tt> for each buffered send operation.  Use
    /// \c communicator::bsend_size to calculate the required buffer size.
    /// \see \c communicator::bsend and \c communicator::ibsend
    explicit bsend_buffer(int size) : buff_{operator new(size)} {
      environment::buffer_attach(buff_, size);
    }

    /// waits for uncompleted message transfers and frees the buffer
    /// \note A blocking communication operation is performed when an object of type
    /// \c bsend_buffer goes out of scope.
    ~bsend_buffer() {
      environment::buffer_detach();
      operator delete(buff_);
    }

    /// deleted copy assignment operator
    /// \param other buffer manager to copy-assign from
    auto &operator=(const bsend_buffer &other) = delete;

    /// deleted move assignment operator
    /// \param other buffer manager to move-assign from
    auto &operator=(bsend_buffer &&other) = delete;
  };

}  // namespace mpl

#endif
