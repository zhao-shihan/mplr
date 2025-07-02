#if !(defined MPLR_DATATYPE_HPP)

#define MPLR_DATATYPE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <deque>
#include <forward_list>
#include <list>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <valarray>
#include <complex>
#include <utility>
#include <tuple>
#include <array>
#include <cstddef>
#include <type_traits>
#if __cplusplus >= 202002L
#include <span>
#endif


namespace mplr {

  namespace detail {

    struct unsupported_type {};
    struct basic_or_fixed_size_type {};
    struct stl_container {};
    struct contiguous_const_stl_container : public stl_container {};
    struct contiguous_stl_container : public contiguous_const_stl_container {};

  }  // namespace detail

  //--- forward declarations -------------------------------------------

  namespace detail {

    template<typename T, typename E>
    class datatype_traits_impl;

    template<typename T>
    class datatype_traits;

  }  // namespace detail

  template<typename T>
  class base_struct_builder;

  template<typename T>
  class struct_builder;

  //--------------------------------------------------------------------

  /// layout class for storing meta information about the public members of structures
  /// \tparam S the struct or class type, the public members of which are managed
  template<typename S>
  class struct_layout {
    template<typename T>
    inline constexpr std::size_t size(const T &) const {
      return 1;
    }

    template<typename T, std::size_t N0>
    inline constexpr std::size_t size(const std::array<T, N0> &) const {
      return N0;
    }

    template<typename T, std::size_t N0>
    inline constexpr std::size_t size(const T (&)[N0]) const {
      return N0;
    }

    template<typename T, std::size_t N0, std::size_t N1>
    inline constexpr std::size_t size(const T (&)[N0][N1]) const {
      return N0 * N1;
    }

    template<typename T, std::size_t N0, std::size_t N1, std::size_t N2>
    inline constexpr std::size_t size(const T (&)[N0][N1][N2]) const {
      return N0 * N1 * N2;
    }

    template<typename T, std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3>
    inline constexpr std::size_t size(const T (&)[N0][N1][N2][N3]) const {
      return N0 * N1 * N2 * N3;
    }

    template<typename T>
    inline MPI_Datatype get_datatype(const T &) const {
      return detail::datatype_traits<T>().get_datatype();
    }

    template<typename T, std::size_t N0>
    inline MPI_Datatype get_datatype(const std::array<T, N0> &) const {
      return detail::datatype_traits<T>().get_datatype();
    }

    template<typename T, std::size_t N0>
    inline MPI_Datatype get_datatype(const T (&)[N0]) const {
      return detail::datatype_traits<T>().get_datatype();
    }

    template<typename T, std::size_t N0, std::size_t N1>
    inline MPI_Datatype get_datatype(const T (&)[N0][N1]) const {
      return detail::datatype_traits<T>().get_datatype();
    }

    template<typename T, std::size_t N0, std::size_t N1, std::size_t N2>
    inline MPI_Datatype get_datatype(const T (&)[N0][N1][N2]) const {
      return detail::datatype_traits<T>().get_datatype();
    }

    template<typename T, std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3>
    inline MPI_Datatype get_datatype(const T (&)[N0][N1][N2][N3]) const {
      return detail::datatype_traits<T>().get_datatype();
    }

    MPI_Aint base_{};
    std::vector<int> block_lengths_;
    std::vector<MPI_Aint> displacements_;
    std::vector<MPI_Datatype> data_types_;

  public:
    /// starts to register a struct type
    /// \param x an instance of type \c S (the template parameter of class \c struct_layout)
    /// \return reference to this \c struct_layout object (allows chaining)
    struct_layout &register_struct(const S &x) {
      MPI_Get_address(const_cast<S *>(&x), &base_);
      return *this;
    }

    /// registers a struct member
    /// \param x a member of an instance of type \c S (the template parameter of class
    /// \c struct_layout)
    /// \return reference to this \c struct_layout object (allows chaining)
    template<typename T>
    struct_layout &register_element(T &x) {
      static_assert(not std::is_const_v<T>, "type must not be const");
      static_assert(not std::is_pointer_v<T>, "type must not be pointer");
      block_lengths_.push_back(size(x));
      MPI_Aint address;
      MPI_Get_address(&x, &address);
      displacements_.push_back(address - base_);
      data_types_.push_back(get_datatype(x));
      return *this;
    }

    friend class base_struct_builder<S>;
  };

  //--------------------------------------------------------------------

  /// Base class used to manage information about structures/classes and their public
  /// members.
  /// \tparam T the structure or class type
  /// \see class \c struct_builder
  template<typename T>
  class base_struct_builder {
  private:
    MPI_Datatype type_;

  protected:
    void define_struct(const struct_layout<T> &str) {
      MPI_Datatype temp_type;
      MPI_Type_create_struct(str.block_lengths_.size(), str.block_lengths_.data(),
                             str.displacements_.data(), str.data_types_.data(), &temp_type);
      MPI_Type_commit(&temp_type);
      MPI_Type_create_resized(temp_type, 0, sizeof(T), &type_);
      MPI_Type_commit(&type_);
      MPI_Type_free(&temp_type);
    }

    base_struct_builder() = default;

  public:
    base_struct_builder(const base_struct_builder &) = delete;
    auto& operator=(const base_struct_builder &) = delete;

  protected:
    ~base_struct_builder() {
      MPI_Type_free(&type_);
    }

    using data_type_category = detail::basic_or_fixed_size_type;

    friend class detail::datatype_traits_impl<T, void>;
  };

  //--------------------------------------------------------------------

  /// Template class used to manage information about structures/classes and their public
  /// members
  /// \tparam T the structure or class type
  /// \note This template class needs to be specialized for each structure/class type that shall
  /// be used in message passing operations.  Specializations must derive from class
  /// \c base_struct_builder. MPLR provides some specializations for common types.
  /// \see class \c base_struct_builder
  template<typename T>
  class struct_builder {
  public:
    using data_type_category = detail::unsupported_type;
  };

  //--------------------------------------------------------------------

  /// Specialization of \c struct_builder for STL pairs.
  /// \tparam T1 type of first member of the STL pair
  /// \tparam T2 type of second member of the STL pair
  /// \see class \c struct_builder
  template<typename T1, typename T2>
  class struct_builder<std::pair<T1, T2>> : public base_struct_builder<std::pair<T1, T2>> {
    using base = base_struct_builder<std::pair<T1, T2>>;
    struct_layout<std::pair<T1, T2>> layout_;

  public:
    struct_builder() {
      std::pair<T1, T2> pair;
      layout_.register_struct(pair);
      layout_.register_element(pair.first);
      layout_.register_element(pair.second);
      base::define_struct(layout_);
    }
  };

  //--------------------------------------------------------------------

  namespace detail {

    template<typename F, typename T, std::size_t N>
    class apply_n {
      F &f_;

    public:
      explicit apply_n(F &f) : f_{f} {
      }

      void operator()(T &x) const {
        apply_n<F, T, N - 1> next{f_};
        next(x);
        f_(std::get<N - 1>(x));
      }
    };

    template<typename F, typename T>
    class apply_n<F, T, 1> {
      F &f_;

    public:
      explicit apply_n(F &f) : f_{f} {
      }

      void operator()(T &x) const {
        f_(std::get<0>(x));
      }
    };

    template<typename F, typename... Args>
    void apply(std::tuple<Args...> &t, F &f) {
      apply_n<F, std::tuple<Args...>, std::tuple_size<std::tuple<Args...>>::value> app{f};
      app(t);
    }

    template<typename... Ts>
    class register_element {
      struct_layout<std::tuple<Ts...>> &layout_;

    public:
      explicit register_element(struct_layout<std::tuple<Ts...>> &layout) : layout_{layout} {
      }

      template<typename T>
      void operator()(T &x) const {
        layout_.register_element(x);
      }
    };

  }  // namespace detail

  /// Specialization of \c struct_builder for STL tuples.
  /// \tparam Ts parameter pack representing the types of the members of the STL tuple
  /// \see class \c struct_builder
  template<typename... Ts>
  class struct_builder<std::tuple<Ts...>> : public base_struct_builder<std::tuple<Ts...>> {
    using base = base_struct_builder<std::tuple<Ts...>>;
    struct_layout<std::tuple<Ts...>> layout_;

  public:
    struct_builder() {
      std::tuple<Ts...> tuple;
      layout_.register_struct(tuple);
      detail::register_element<Ts...> reg{layout_};
      detail::apply(tuple, reg);
      base::define_struct(layout_);
    }
  };

  //--------------------------------------------------------------------

  /// Specialization of \c struct_builder for fixed-size one-dimensional C-style
  /// arrays.
  /// \tparam T type of the array elements
  /// \tparam N0 array size
  /// \see class \c struct_builder
  template<typename T, std::size_t N0>
  class struct_builder<T[N0]> : public base_struct_builder<T[N0]> {
    using base = base_struct_builder<T[N0]>;
    struct_layout<T[N0]> layout_;

  public:
    struct_builder() {
      T array[N0];
      layout_.register_struct(array);
      layout_.register_element(array);
      base::define_struct(layout_);
    }
  };

  /// Specialization of \c struct_builder for fixed-size two-dimensional C-style
  /// arrays.
  /// \tparam T type of the array elements
  /// \tparam N0 array size, first dimension
  /// \tparam N1 array size, second dimension
  /// \see class \c struct_builder
  template<typename T, std::size_t N0, std::size_t N1>
  class struct_builder<T[N0][N1]> : public base_struct_builder<T[N0][N1]> {
    using base = base_struct_builder<T[N0][N1]>;
    struct_layout<T[N0][N1]> layout_;

  public:
    struct_builder() {
      T array[N0][N1];
      layout_.register_struct(array);
      layout_.register_element(array);
      base::define_struct(layout_);
    }
  };

  /// Specialization of \c struct_builder for fixed-size three-dimensional C-style
  /// arrays.
  /// \tparam T type of the array elements
  /// \tparam N0 array size, first dimension
  /// \tparam N1 array size, second dimension
  /// \tparam N2 array size, third dimension
  /// \see class \c struct_builder
  template<typename T, std::size_t N0, std::size_t N1, std::size_t N2>
  class struct_builder<T[N0][N1][N2]> : public base_struct_builder<T[N0][N1][N2]> {
    using base = base_struct_builder<T[N0][N1][N2]>;
    struct_layout<T[N0][N1][N2]> layout_;

  public:
    struct_builder() {
      T array[N0][N1][N2];
      layout_.register_struct(array);
      layout_.register_element(array);
      base::define_struct(layout_);
    }
  };

  /// Specialization of \c struct_builder for fixed-size four-dimensional C-style
  /// arrays.
  /// \tparam T type of the array elements
  /// \tparam N0 array size, first dimension
  /// \tparam N1 array size, second dimension
  /// \tparam N2 array size, third dimension
  /// \tparam N3 array size, fourth dimension
  /// \see class \c struct_builder
  template<typename T, std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3>
  class struct_builder<T[N0][N1][N2][N3]> : public base_struct_builder<T[N0][N1][N2][N3]> {
    using base = base_struct_builder<T[N0][N1][N2][N3]>;
    struct_layout<T[N0][N1][N2][N3]> layout_;

  public:
    struct_builder() {
      T array[N0][N1][N2][N3];
      layout_.register_struct(array);
      layout_.register_element(array);
      base::define_struct(layout_);
    }
  };

  //--------------------------------------------------------------------

  /// Specialization of \c struct_builder for fixed-size one-dimensional STL arrays.
  /// \tparam T type of the array elements
  /// \tparam N array size
  /// \see class \c struct_builder
  template<typename T, std::size_t N>
  class struct_builder<std::array<T, N>> : public base_struct_builder<std::array<T, N>> {
    using base = base_struct_builder<std::array<T, N>>;
    struct_layout<std::array<T, N>> layout_;

  public:
    struct_builder() {
      std::array<T, N> array{};
      layout_.register_struct(array);
      layout_.register_element(array);
      base::define_struct(layout_);
    }
  };

  //--------------------------------------------------------------------

  namespace detail {

    template<typename T, typename Enable = void>
    class datatype_traits_impl {
    public:
      static MPI_Datatype get_datatype() {
        static struct_builder<T> builder;
        return builder.type_;
      }
      using data_type_category = typename struct_builder<T>::data_type_category;
    };

    template<typename T>
    class datatype_traits_impl<T, std::enable_if_t<std::is_enum<T>::value>> {
      using underlying = std::underlying_type_t<T>;

    public:
      static MPI_Datatype get_datatype() {
        return datatype_traits<underlying>::get_datatype();
      }
      using data_type_category = typename datatype_traits<underlying>::data_type_category;
    };

#if defined MPLR_HOMOGENEOUS
    template<typename T>
    class datatype_traits_impl<
        T, std::enable_if_t<std::is_trivially_copyable_v<T> and std::is_copy_assignable_v<T> and
                            not std::is_enum_v<T> and not std::is_array_v<T>>> {
    public:
      static MPI_Datatype get_datatype() {
        return datatype_traits_impl<unsigned char[sizeof(T)]>::get_datatype();
      }
      using data_type_category = typename datatype_traits_impl<T>::data_type_category;
    };
#endif


    template<typename T>
    class datatype_traits {
    public:
      static MPI_Datatype get_datatype() {
        return detail::datatype_traits_impl<T>::get_datatype();
      }
      using data_type_category = typename detail::datatype_traits_impl<T>::data_type_category;
    };

#define MPLR_DATATYPE_TRAITS(type, mpi_type)                      \
  template<>                                                     \
  class datatype_traits<type> {                                  \
  public:                                                        \
    static MPI_Datatype get_datatype() {                         \
      return mpi_type;                                           \
    }                                                            \
    using data_type_category = detail::basic_or_fixed_size_type; \
  }

    MPLR_DATATYPE_TRAITS(char, MPI_CHAR);

    MPLR_DATATYPE_TRAITS(signed char, MPI_SIGNED_CHAR);

    MPLR_DATATYPE_TRAITS(unsigned char, MPI_UNSIGNED_CHAR);

    MPLR_DATATYPE_TRAITS(wchar_t, MPI_WCHAR);

    MPLR_DATATYPE_TRAITS(signed short int, MPI_SHORT);

    MPLR_DATATYPE_TRAITS(unsigned short int, MPI_UNSIGNED_SHORT);

    MPLR_DATATYPE_TRAITS(signed int, MPI_INT);

    MPLR_DATATYPE_TRAITS(unsigned int, MPI_UNSIGNED);

    MPLR_DATATYPE_TRAITS(signed long, MPI_LONG);

    MPLR_DATATYPE_TRAITS(unsigned long, MPI_UNSIGNED_LONG);

    MPLR_DATATYPE_TRAITS(signed long long, MPI_LONG_LONG);

    MPLR_DATATYPE_TRAITS(unsigned long long, MPI_UNSIGNED_LONG_LONG);

    MPLR_DATATYPE_TRAITS(bool, MPI_CXX_BOOL);

    MPLR_DATATYPE_TRAITS(float, MPI_FLOAT);

    MPLR_DATATYPE_TRAITS(double, MPI_DOUBLE);

    MPLR_DATATYPE_TRAITS(long double, MPI_LONG_DOUBLE);

    MPLR_DATATYPE_TRAITS(std::byte, MPI_BYTE);

    MPLR_DATATYPE_TRAITS(std::complex<float>, MPI_CXX_FLOAT_COMPLEX);

    MPLR_DATATYPE_TRAITS(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX);

    MPLR_DATATYPE_TRAITS(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX);

#undef MPLR_DATATYPE_TRAITS

#if __cplusplus >= 202002L
    template<>
    class datatype_traits<char8_t> {
    public:
      static MPI_Datatype get_datatype() {
        return datatype_traits<unsigned char>::get_datatype();
      }
      using data_type_category = detail::basic_or_fixed_size_type;
    };
#endif

    template<>
    class datatype_traits<char16_t> {
    public:
      static MPI_Datatype get_datatype() {
        return datatype_traits<std::uint_least16_t>::get_datatype();
      }
      using data_type_category = detail::basic_or_fixed_size_type;
    };

    template<>
    class datatype_traits<char32_t> {
    public:
      static MPI_Datatype get_datatype() {
        return datatype_traits<std::uint_least32_t>::get_datatype();
      }
      using data_type_category = detail::basic_or_fixed_size_type;
    };

    template<typename T, typename A>
    class datatype_traits<std::vector<T, A>> {
    public:
      using data_type_category = detail::contiguous_stl_container;
    };

#if __cplusplus >= 202002L
    template<typename T, std::size_t N>
    class datatype_traits<std::span<T, N>> {
    public:
      using data_type_category = detail::contiguous_stl_container;
    };
#endif

    template<typename A>
    class datatype_traits<std::vector<bool, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename A>
    class datatype_traits<std::deque<T, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename A>
    class datatype_traits<std::forward_list<T, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename A>
    class datatype_traits<std::list<T, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::set<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::map<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::multiset<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::multimap<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::unordered_set<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::unordered_map<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::unordered_multiset<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename C, typename A>
    class datatype_traits<std::unordered_multimap<T, C, A>> {
    public:
      using data_type_category = detail::stl_container;
    };

    template<typename T, typename Trait, typename Char>
    class datatype_traits<std::basic_string<T, Trait, Char>> {
    public:
      using data_type_category = detail::contiguous_const_stl_container;
    };

    template<typename T>
    class datatype_traits<std::valarray<T>> {
    public:
      using data_type_category = detail::contiguous_stl_container;
    };

  }  // namespace detail

}  // namespace mplr

#define MPLR_GET_NTH_ARG(                                                                      \
    _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, \
    _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, \
    _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, _71, _72, _73, \
    _74, _75, _76, _77, _78, _79, _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, \
    _92, _93, _94, _95, _96, _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107,   \
    _108, _109, _110, _111, _112, _113, _114, _115, _116, _117, _118, _119, N, ...)           \
  N

#define MPLR_FE_0(MPLR_CALL, ...)
#define MPLR_FE_1(MPLR_CALL, x) MPLR_CALL(x)
#define MPLR_FE_2(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_1(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_3(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_2(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_4(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_3(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_5(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_4(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_6(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_5(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_7(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_6(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_8(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_7(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_9(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_8(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_10(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_9(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_11(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_10(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_12(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_11(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_13(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_12(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_14(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_13(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_15(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_14(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_16(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_15(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_17(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_16(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_18(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_17(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_19(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_18(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_20(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_19(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_21(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_20(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_22(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_21(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_23(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_22(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_24(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_23(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_25(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_24(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_26(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_25(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_27(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_26(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_28(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_27(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_29(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_28(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_30(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_29(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_31(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_30(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_32(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_31(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_33(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_32(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_34(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_33(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_35(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_34(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_36(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_35(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_37(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_36(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_38(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_37(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_39(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_38(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_40(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_39(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_41(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_40(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_42(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_41(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_43(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_42(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_44(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_43(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_45(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_44(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_46(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_45(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_47(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_46(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_48(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_47(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_49(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_48(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_50(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_49(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_51(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_50(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_52(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_51(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_53(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_52(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_54(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_53(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_55(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_54(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_56(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_55(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_57(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_56(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_58(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_57(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_59(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_58(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_60(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_59(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_61(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_60(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_62(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_61(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_63(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_62(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_64(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_63(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_65(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_64(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_66(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_65(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_67(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_66(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_68(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_67(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_69(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_68(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_70(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_69(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_71(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_70(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_72(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_71(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_73(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_72(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_74(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_73(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_75(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_74(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_76(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_75(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_77(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_76(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_78(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_77(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_79(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_78(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_80(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_79(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_81(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_80(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_82(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_81(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_83(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_82(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_84(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_83(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_85(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_84(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_86(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_85(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_87(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_86(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_88(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_87(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_89(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_88(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_90(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_89(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_91(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_90(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_92(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_91(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_93(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_92(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_94(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_93(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_95(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_94(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_96(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_95(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_97(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_96(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_98(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_97(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_99(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_98(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_100(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_99(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_101(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_100(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_102(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_101(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_103(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_102(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_104(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_103(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_105(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_104(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_106(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_105(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_107(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_106(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_108(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_107(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_109(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_108(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_110(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_109(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_111(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_110(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_112(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_111(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_113(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_112(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_114(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_113(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_115(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_114(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_116(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_115(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_117(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_116(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_118(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_117(MPLR_CALL, __VA_ARGS__)
#define MPLR_FE_119(MPLR_CALL, x, ...) MPLR_CALL(x) MPLR_FE_118(MPLR_CALL, __VA_ARGS__)
#define MPLR_CALL_MACRO_X_FOR_EACH(x, ...)                                                      \
  MPLR_GET_NTH_ARG(                                                                             \
      "ignored", __VA_ARGS__, MPLR_FE_119, MPLR_FE_118, MPLR_FE_117, MPLR_FE_116, MPLR_FE_115,      \
      MPLR_FE_114, MPLR_FE_113, MPLR_FE_112, MPLR_FE_111, MPLR_FE_110, MPLR_FE_109, MPLR_FE_108,      \
      MPLR_FE_107, MPLR_FE_106, MPLR_FE_105, MPLR_FE_104, MPLR_FE_103, MPLR_FE_102, MPLR_FE_101,      \
      MPLR_FE_100, MPLR_FE_99, MPLR_FE_98, MPLR_FE_97, MPLR_FE_96, MPLR_FE_95, MPLR_FE_94, MPLR_FE_93, \
      MPLR_FE_92, MPLR_FE_91, MPLR_FE_90, MPLR_FE_89, MPLR_FE_88, MPLR_FE_87, MPLR_FE_86, MPLR_FE_85,  \
      MPLR_FE_84, MPLR_FE_83, MPLR_FE_82, MPLR_FE_81, MPLR_FE_80, MPLR_FE_79, MPLR_FE_78, MPLR_FE_77,  \
      MPLR_FE_76, MPLR_FE_75, MPLR_FE_74, MPLR_FE_73, MPLR_FE_72, MPLR_FE_71, MPLR_FE_70, MPLR_FE_69,  \
      MPLR_FE_68, MPLR_FE_67, MPLR_FE_66, MPLR_FE_65, MPLR_FE_64, MPLR_FE_63, MPLR_FE_62, MPLR_FE_61,  \
      MPLR_FE_60, MPLR_FE_59, MPLR_FE_58, MPLR_FE_57, MPLR_FE_56, MPLR_FE_55, MPLR_FE_54, MPLR_FE_53,  \
      MPLR_FE_52, MPLR_FE_51, MPLR_FE_50, MPLR_FE_49, MPLR_FE_48, MPLR_FE_47, MPLR_FE_46, MPLR_FE_45,  \
      MPLR_FE_44, MPLR_FE_43, MPLR_FE_42, MPLR_FE_41, MPLR_FE_40, MPLR_FE_39, MPLR_FE_38, MPLR_FE_37,  \
      MPLR_FE_36, MPLR_FE_35, MPLR_FE_34, MPLR_FE_33, MPLR_FE_32, MPLR_FE_31, MPLR_FE_30, MPLR_FE_29,  \
      MPLR_FE_28, MPLR_FE_27, MPLR_FE_26, MPLR_FE_25, MPLR_FE_24, MPLR_FE_23, MPLR_FE_22, MPLR_FE_21,  \
      MPLR_FE_20, MPLR_FE_19, MPLR_FE_18, MPLR_FE_17, MPLR_FE_16, MPLR_FE_15, MPLR_FE_14, MPLR_FE_13,  \
      MPLR_FE_12, MPLR_FE_11, MPLR_FE_10, MPLR_FE_9, MPLR_FE_8, MPLR_FE_7, MPLR_FE_6, MPLR_FE_5,       \
      MPLR_FE_4, MPLR_FE_3, MPLR_FE_2, MPLR_FE_1, MPLR_FE_0)                                        \
  (x, __VA_ARGS__)

#define MPLR_REGISTER(element) layout_.register_element(str.element);

#define MPLR_REFLECTION(STRUCT, ...)                                     \
  namespace mplr {                                                       \
    template<>                                                          \
    class struct_builder<STRUCT> : public base_struct_builder<STRUCT> { \
      struct_layout<STRUCT> layout_;                                    \
                                                                        \
    public:                                                             \
      struct_builder() {                                                \
        STRUCT str;                                                     \
        layout_.register_struct(str);                                   \
        MPLR_CALL_MACRO_X_FOR_EACH(MPLR_REGISTER, __VA_ARGS__)            \
        define_struct(layout_);                                         \
      }                                                                 \
    };                                                                  \
  }

#endif
