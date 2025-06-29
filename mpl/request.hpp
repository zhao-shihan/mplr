#if !(defined MPL_REQUEST_HPP)

#define MPL_REQUEST_HPP

#include <mpi.h>
#include <utility>
#include <optional>
#include <vector>
#include <thread>
#include <mpl/utility.hpp>


namespace mpl {

  class duty_ratio {
  public:
    enum struct preset : char {
      active = 'a',    // 0.1
      moderate = 'm',  // 0.01
      relaxed = 'r'    // 0.001
    };

  public:
    constexpr duty_ratio(preset p)
        : duty_ratio{[p]() -> double {
            switch (p) {
              case preset::active:
                return 0.1;
              case preset::moderate:
                return 0.01;
              case preset::relaxed:
                return 0.001;
            }
            return -1;
          }()} {
    }

    constexpr explicit duty_ratio(double duty_ratio) : duty_ratio_{duty_ratio} {
#if defined MPL_DEBUG
      if (duty_ratio_ <= 0 or duty_ratio_ > 1) {
        throw invalid_argument{};
      }
#endif
    }

    constexpr operator double() const {
      return duty_ratio_;
    }

    constexpr double sleep_ratio() const {
      return 1 - duty_ratio_;
    }

    constexpr double duty_to_sleep_ratio() const {
      return duty_ratio_ / sleep_ratio();
    }

    constexpr double sleep_to_duty_ratio() const {
      return sleep_ratio() / duty_ratio_;
    }

  private:
    double duty_ratio_;
  };

  /// Indicates kind of outcome of test for request completion.
  enum class test_result {
    completed,          ///< some request has been completed
    no_completed,       ///< no request has been completed
    no_active_requests  ///< there is no request waiting for completion
  };

  namespace impl {

    template<typename T>
    class base_request;

    template<typename T>
    class request_pool;

    class base_irequest {
      MPI_Request request_{MPI_REQUEST_NULL};

    public:
      explicit base_irequest(MPI_Request request) : request_{request} {
      }

      friend class base_request<base_irequest>;

      friend class request_pool<base_irequest>;
    };

    class base_prequest {
      MPI_Request request_{MPI_REQUEST_NULL};

    public:
      explicit base_prequest(MPI_Request request) : request_{request} {
      }

      friend class base_request<base_prequest>;

      friend class request_pool<base_prequest>;
    };

    //------------------------------------------------------------------

    template<typename T>
    class base_request {
    protected:
      MPI_Request request_;

    public:
      base_request() : request_{MPI_REQUEST_NULL} {
      }

      base_request(const base_request &) = delete;

      explicit base_request(const base_irequest &req) : request_{req.request_} {
      }
      explicit base_request(const base_prequest &req) : request_{req.request_} {
      }

      base_request(base_request &&other) noexcept : request_(other.request_) {
        other.request_ = MPI_REQUEST_NULL;
      }

      ~base_request() {
        if (is_valid())
          MPI_Request_free(&request_);
      }

      auto &operator=(const base_request &) = delete;

      base_request &operator=(base_request &&other) noexcept {
        if (this != &other) {
          if (is_valid())
            MPI_Request_free(&request_);
          request_ = other.request_;
          other.request_ = MPI_REQUEST_NULL;
        }
        return *this;
      }

      /// Checks if a request is not null.
      /// \return true if request is valid
      /// \note A default constructed request is a non-valid request.
      bool is_valid() {
        return request_ != MPI_REQUEST_NULL;
      }

      /// Cancels the request if it is pending.
      void cancel() {
        if (is_valid())
          MPI_Cancel(&request_);
      }

      /// Tests for the completion.
      /// \return the operation's status if completed successfully
      std::optional<status_t> test() {
        int result{true};
        status_t s;
        MPI_Test(&request_, &result, static_cast<MPI_Status *>(&s));
        if (result != 0)
          return s;
        return {};
      }

      /// Wait for a pending communication operation.
      /// \return operation's status after completion
      status_t wait() {
        status_t s;
        MPI_Wait(&request_, static_cast<MPI_Status *>(&s));
        return s;
      }

      /// @brief A lazy-spin wait.
      /// @param duty_ratio duty ratio of wait
      /// @return operation's status after completion
      status_t wait(duty_ratio duty_ratio) {
        const auto sleep_to_duty_ratio{duty_ratio.sleep_to_duty_ratio()};
        int flag;
        status_t status;
        while (true) {
          const auto t0{detail::steady_high_resolution_clock::now()};
          MPI_Test(&request_, &flag, static_cast<MPI_Status *>(&status));
          if (flag) {
            return status;
          }
          const auto t1{detail::steady_high_resolution_clock::now()};
          std::this_thread::sleep_for(sleep_to_duty_ratio * (t1 - t0));
        }
      }

      /// Access information associated with a request without freeing the request.
      /// \return the operation's status if completed successfully
      std::optional<status_t> get_status() {
        int result{true};
        status_t s;
        MPI_Request_get_status(request_, &result, static_cast<MPI_Status *>(&s));
        if (result != 0)
          return s;
        return {};
      }
    };

    //------------------------------------------------------------------

    template<typename T>
    class request_pool {
    protected:
      std::vector<MPI_Request> requests_;

    public:
      /// Type used in all index-based operations.
      using size_type = std::vector<MPI_Request>::size_type;

      request_pool() = default;

      request_pool(const request_pool &) = delete;

      request_pool(request_pool &&other) noexcept : requests_(std::move(other.requests_)) {
      }

      ~request_pool() {
        for (auto &request : requests_)
          if (request != MPI_REQUEST_NULL)
            MPI_Request_free(&request);
      }

      auto &operator=(const request_pool &) = delete;

      request_pool &operator=(request_pool &&other) noexcept {
        if (this != &other) {
          for (auto &request : requests_)
            if (request != MPI_REQUEST_NULL)
              MPI_Request_free(&request);
          requests_ = std::move(other.requests_);
        }
        return *this;
      }

      /// Determine the size of request pool.
      /// \return number of requests currently in request pool
      [[nodiscard]] size_type size() const {
        return requests_.size();
      }

      /// Determine if request pool is empty.
      /// \return true if number of requests currently in request pool is non-zero
      [[nodiscard]] bool empty() const {
        return requests_.empty();
      }

      /// Tests for the completion for a request in the pool.
      /// \param i index of the request for which shall be tested
      /// \return the operation's status if completed successfully
      std::optional<status_t> test(size_type i) {
        int result{true};
        status_t s;
        MPI_Test(&requests_[i], &result, static_cast<MPI_Status *>(&s));
        if (result != 0)
          return s;
        return {};
      }

      /// Wait for a pending request in the pool.
      /// \param i index of the request for which shall be waited
      /// \return operation's status after completion
      status_t wait(size_type i) {
        status_t s;
        MPI_Wait(&requests_[i], static_cast<MPI_Status *>(&s));
        return s;
      }

      /// @brief A lazy-spin wait.
      /// @param i index of the request for which shall be waited
      /// @param duty_ratio duty ratio of wait
      /// @return operation's status after completion
      status_t wait(size_type i, duty_ratio duty_ratio) {
        const auto sleep_to_duty_ratio{duty_ratio.sleep_to_duty_ratio()};
        int flag;
        status_t status;
        while (true) {
          const auto t0{detail::steady_high_resolution_clock::now()};
          MPI_Test(&requests_[i], &flag, static_cast<MPI_Status *>(&status));
          if (flag) {
            return status;
          }
          const auto t1{detail::steady_high_resolution_clock::now()};
          std::this_thread::sleep_for(sleep_to_duty_ratio * (t1 - t0));
        }
      }

      /// Access information associated with a request in the pool without freeing the request.
      /// \param i index of the request for which the status will be returned
      /// \return the operation's status if completed successfully
      std::optional<status_t> get_status(size_type i) {
        int result{true};
        status_t s;
        MPI_Request_get_status(requests_[i], &result, static_cast<MPI_Status *>(&s));
        if (result != 0)
          return s;
        return {};
      }

      /// Cancels a pending request in the pool.
      /// \param i index of the request for which shall be cancelled
      void cancel(size_type i) {
        if (requests_[i] != MPI_REQUEST_NULL)
          MPI_Cancel(&requests_[i]);
      }

      /// Cancels all requests in the pool.
      void cancelall() {
        for (size_type i = 0; i < requests_.size(); ++i)
          cancel(i);
      }

      /// Move a request into the request pool.
      /// \param request request to move into the pool
      void push(T &&request) {
        requests_.push_back(request.request_);
        request.request_ = MPI_REQUEST_NULL;
      }

      /// Wait for completion of any pending communication operation.
      /// \return pair containing the outcome of the wait operation and an index to the
      /// completed request if there was any pending request
      std::pair<test_result, size_type> waitany() {
        int index;
        MPI_Waitany(size(), requests_.data(), &index, MPI_STATUS_IGNORE);
        if (index != MPI_UNDEFINED) {
          return std::make_pair(test_result::completed, static_cast<size_type>(index));
        }
        return std::make_pair(test_result::no_active_requests, size());
      }

      /// @brief A lazy-spin waitany.
      /// @param duty_ratio duty ratio of wait
      /// @return operation's status after completion
      std::pair<test_result, size_type> waitany(duty_ratio duty_ratio) {
        const auto sleep_to_duty_ratio{duty_ratio.sleep_to_duty_ratio()};
        int index;
        int flag;
        while (true) {
          const auto t0{detail::steady_high_resolution_clock::now()};
          MPI_Testany(size(), requests_.data(), &index, &flag, MPI_STATUS_IGNORE);
          if (flag) {
            if (index == MPI_UNDEFINED) {
              return {test_result::no_active_requests, size()};
            }
            return {test_result::completed, static_cast<size_type>(index)};
          }
          const auto t1{detail::steady_high_resolution_clock::now()};
          std::this_thread::sleep_for(sleep_to_duty_ratio * (t1 - t0));
        }
      }

      /// Test for completion of any pending communication operation.
      /// \return pair containing the outcome of the test and an index to the completed
      /// request if there was any pending request
      std::pair<test_result, size_type> testany() {
        int index, flag;
        MPI_Testany(size(), requests_.data(), &index, &flag, MPI_STATUS_IGNORE);
        if (flag != 0 and index != MPI_UNDEFINED) {
          return std::make_pair(test_result::completed, static_cast<size_type>(index));
        }
        if (flag != 0 and index == MPI_UNDEFINED)
          return std::make_pair(test_result::no_active_requests, size());
        return std::make_pair(test_result::no_completed, size());
      }

      /// Waits for completion of all pending requests.
      void waitall() {
        MPI_Waitall(size(), requests_.data(), MPI_STATUSES_IGNORE);
      }

      /// A lazy-spin waitall.
      /// @param duty_ratio duty ratio of wait
      void waitall(duty_ratio duty_ratio) {
        const auto sleep_to_duty_ratio{duty_ratio.sleep_to_duty_ratio()};
        int flag;
        while (true) {
          const auto t0{detail::steady_high_resolution_clock::now()};
          MPI_Testall(size(), requests_.data(), &flag, MPI_STATUSES_IGNORE);
          if (flag) {
            return;
          }
          const auto t1{detail::steady_high_resolution_clock::now()};
          std::this_thread::sleep_for(sleep_to_duty_ratio * (t1 - t0));
        }
      }

      /// Tests for completion of all pending requests.
      /// \return true if all pending requests have completed
      bool testall() {
        int flag;
        MPI_Testall(size(), requests_.data(), &flag, MPI_STATUSES_IGNORE);
        return static_cast<bool>(flag);
      }

      /// Waits until one or more pending requests have finished.
      /// \return pair containing the outcome of the wait operation and a list of indices to
      /// the completed requests if there was any pending request
      std::pair<test_result, std::vector<size_type>> waitsome() {
        std::vector<int> out_indices(size());
        int count;
        MPI_Waitsome(size(), requests_.data(), &count, out_indices.data(), MPI_STATUSES_IGNORE);
        if (count != MPI_UNDEFINED) {
          return std::make_pair(
              test_result::completed,
              std::vector<size_t>(out_indices.begin(), out_indices.begin() + count));
        }
        return std::make_pair(test_result::no_active_requests, std::vector<size_t>{});
      }

      /// A lazy-spin waitsome.
      /// @param duty_ratio duty ratio of wait
      std::pair<test_result, std::vector<size_type>> waitsome(duty_ratio duty_ratio) {
        const auto sleep_to_duty_ratio{duty_ratio.sleep_to_duty_ratio()};
        std::vector<int> out_indices(size());
        int count;
        while (true) {
          const auto t0{detail::steady_high_resolution_clock::now()};
          MPI_Testsome(size(), requests_.data(), &count, out_indices.data(),
                       MPI_STATUSES_IGNORE);
          if (count == MPI_UNDEFINED) {
            return {test_result::no_active_requests, {}};
          }
          if (count != 0) {
            return {test_result::completed,
                    std::vector<size_t>(out_indices.begin(), out_indices.begin() + count)};
          }
          const auto t1{detail::steady_high_resolution_clock::now()};
          std::this_thread::sleep_for(sleep_to_duty_ratio * (t1 - t0));
        }
      }

      /// Tests if one or more pending requests have finished.
      /// \return pair containing the outcome of the test and a list of indices to the completed
      /// requests if there was any pending request
      std::pair<test_result, std::vector<size_type>> testsome() {
        std::vector<int> out_indices(size());
        int count;
        MPI_Testsome(size(), requests_.data(), &count, out_indices.data(), MPI_STATUSES_IGNORE);
        if (count != MPI_UNDEFINED) {
          return std::make_pair(
              count == 0 ? test_result::no_completed : test_result::completed,
              std::vector<size_t>(out_indices.begin(), out_indices.begin() + count));
        }
        return std::make_pair(test_result::no_active_requests, std::vector<size_t>{});
      }
    };

  }  // namespace impl

  //--------------------------------------------------------------------

  /// Represents a non-blocking communication request.
  class irequest : public impl::base_request<impl::base_irequest> {
    using base = impl::base_request<impl::base_irequest>;
    using base::request_;

  public:
    /// Default null request.
    irequest() = default;

#if (!defined MPL_DOXYGEN_SHOULD_SKIP_THIS)
    irequest(const impl::base_irequest &r) : base{r} {
    }
#endif

    /// Deleted copy constructor.
    irequest(const irequest &) = delete;

    /// Move constructor.
    /// \param other the request to move from
    irequest(irequest &&other) noexcept = default;

    /// Deleted copy operator.
    auto &operator=(const irequest &) = delete;

    /// Move operator.
    /// \param other the request to move from
    /// \return reference to the moved-to request
    irequest &operator=(irequest &&other) noexcept = default;

    friend class impl::request_pool<irequest>;
  };

  //--------------------------------------------------------------------

  /// Container for managing a list of non-blocking communication requests.
  class irequest_pool : public impl::request_pool<irequest> {
    using base = impl::request_pool<irequest>;

  public:
    /// Constructs an empty pool of   communication requests.
    irequest_pool() = default;

    /// Deleted copy constructor.
    irequest_pool(const irequest_pool &) = delete;

    /// Move constructor.
    /// \param other the request pool to move from
    irequest_pool(irequest_pool &&other) noexcept = default;

    /// Deleted copy operator.
    auto &operator=(const irequest_pool &) = delete;

    /// Move operator.
    /// \param other the request pool to move from
    /// \return reference to the moved-to request pool
    irequest_pool &operator=(irequest_pool &&other) noexcept = default;
  };

  //--------------------------------------------------------------------

  /// Represents a persistent communication request.
  class prequest : public impl::base_request<impl::base_prequest> {
    using base = impl::base_request<impl::base_prequest>;
    using base::request_;

  public:
    /// Default null request.
    prequest() = default;

#if (!defined MPL_DOXYGEN_SHOULD_SKIP_THIS)
    prequest(const impl::base_prequest &r) : base{r} {
    }
#endif

    /// Deleted copy constructor.
    prequest(const prequest &) = delete;

    /// Move constructor.
    /// \param other the request to move from
    prequest(prequest &&other) noexcept = default;

    /// Deleted copy operator.
    auto &operator=(const prequest &) = delete;

    /// Move operator.
    /// \param other the request to move from
    /// \return reference to the moved-to request
    prequest &operator=(prequest &&other) noexcept = default;

    /// Start communication operation.
    void start() {
      MPI_Start(&request_);
    }

    friend class impl::request_pool<prequest>;
  };

  //--------------------------------------------------------------------

  /// Container for managing a list of persisting communication requests.
  class prequest_pool : public impl::request_pool<prequest> {
    using base = impl::request_pool<prequest>;
    using base::requests_;

  public:
    /// Constructs an empty pool of persistent communication requests.
    prequest_pool() = default;

    /// Deleted copy constructor.
    prequest_pool(const prequest_pool &) = delete;

    /// Move constructor.
    /// \param other the request pool to move from
    prequest_pool(prequest_pool &&other) noexcept = default;

    /// Deleted copy constructor.
    auto &operator=(const prequest_pool &) = delete;

    /// Move operator.
    /// \param other the request pool to move from
    prequest_pool &operator=(prequest_pool &&other) noexcept = default;

    /// Start a persistent requests in the pool.
    /// \param i index of the request for which shall be started
    void start(size_type i) {
      MPI_Start(&requests_[i]);
    }

    /// Start all persistent requests in the pool.
    void startall() {
      MPI_Startall(size(), requests_.data());
    }
  };

}  // namespace mpl

#endif
