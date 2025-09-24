
template <template <typename> class DerivedTypeT>
class MemSpaceJumpTable {
 public:
  template <typename Func2D, typename NgpBase1, typename NgpBase2, typename... Args>
  static void apply_2d(const Func2D &func_2d, NgpBase1 *base1, NgpBase2 *base2, const std::type_index &mem_space1,
                       const std::type_index &mem_space2, Args &&...args) {
    MUNDY_THROW_REQUIRE(base1 != nullptr, std::invalid_argument, "MemSpaceJumpTable::apply_2d: base1 is null.");
    MUNDY_THROW_REQUIRE(base2 != nullptr, std::invalid_argument, "MemSpaceJumpTable::apply_2d: base2 is null.");
    assert_valid_mem_space(mem_space1);
    assert_valid_mem_space(mem_space2);
    std::size_t mem_space1_hash = mem_space1.hash_code();
    if (mem_space1_hash == typeid(Kokkos::HostSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::HostSpace> &>(*base1)), base2, mem_space2,
               std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::SharedSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::SharedSpace> &>(*base1)), base2, mem_space2,
               std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::SharedHostPinnedSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::SharedHostPinnedSpace> &>(*base1)), base2,
               mem_space2, std::forward<Args>(args)...);
    }
#ifdef KOKKOS_ENABLE_CUDA
    else if (mem_space1_hash == typeid(Kokkos::CudaSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::CudaSpace> &>(*base1)), base2, mem_space2,
               std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::CudaHostPinnedSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::CudaHostPinnedSpace> &>(*base1)), base2,
               mem_space2, std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::CudaUVMSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::CudaUVMSpace> &>(*base1)), base2, mem_space2,
               std::forward<Args>(args)...);
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    else if (mem_space1_hash == typeid(Kokkos::HIPSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::HIPSpace> &>(*base1)), base2, mem_space2,
               std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::HIPHostPinnedSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::HIPHostPinnedSpace> &>(*base1)), base2,
               mem_space2, std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::HIPManagedSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::HIPManagedSpace> &>(*base1)), base2,
               mem_space2, std::forward<Args>(args)...);
    }
#endif
#ifdef KOKKOS_ENABLE_SYCL
    else if (mem_space1_hash == typeid(Kokkos::SYCLDeviceUSMSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::SYCLDeviceUSMSpace> &>(*base1)), base2,
               mem_space2, std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::SYCLHostUSMSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::SYCLHostUSMSpace> &>(*base1)), base2,
               mem_space2, std::forward<Args>(args)...);
    } else if (mem_space1_hash == typeid(Kokkos::SYCLSharedUSMSpace).hash_code()) {
      apply_1d(partial_functor(func_2d, dynamic_cast<DerivedTypeT<Kokkos::SYCLSharedUSMSpace> &>(*base1)), base2,
               mem_space2, std::forward<Args>(args)...);
    }
#endif

    // Impossible to reach due to the is_valid_mem_space check above.
  }

  template <typename Func1D, typename NgpBase, typename... Args>
  static void apply_1d(const Func1D &func_1d, NgpBase *base, const std::type_index &mem_space, Args &&...args) {
    MUNDY_THROW_REQUIRE(base != nullptr, std::invalid_argument, "MemSpaceJumpTable::apply_1d: base is null.");
    assert_valid_mem_space(mem_space);
    std::size_t mem_space_hash = mem_space.hash_code();
    if (mem_space_hash == typeid(Kokkos::HostSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::HostSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::SharedSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::SharedSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::SharedHostPinnedSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::SharedHostPinnedSpace> &>(*base), std::forward<Args>(args)...);
    }
#ifdef KOKKOS_ENABLE_CUDA
    else if (mem_space_hash == typeid(Kokkos::CudaSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::CudaSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::CudaHostPinnedSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::CudaHostPinnedSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::CudaUVMSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::CudaUVMSpace> &>(*base), std::forward<Args>(args)...);
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    else if (mem_space_hash == typeid(Kokkos::HIPSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::HIPSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::HIPHostPinnedSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::HIPHostPinnedSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::HIPManagedSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::HIPManagedSpace> &>(*base), std::forward<Args>(args)...);
    }
#endif
#ifdef KOKKOS_ENABLE_SYCL
    else if (mem_space_hash == typeid(Kokkos::SYCLDeviceUSMSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::SYCLDeviceUSMSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::SYCLHostUSMSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::SYCLHostUSMSpace> &>(*base), std::forward<Args>(args)...);
    } else if (mem_space_hash == typeid(Kokkos::SYCLSharedUSMSpace).hash_code()) {
      func_1d(dynamic_cast<DerivedTypeT<Kokkos::SYCLSharedUSMSpace> &>(*base), std::forward<Args>(args)...);
    }
#endif

    // Impossible to reach due to the is_valid_mem_space check above.
  }

  static bool is_valid_mem_space(const std::type_index &mem_space) {
    // List of valid Kokkos memory spaces
    /*

    // Only if Cuda enabled
    CudaSpace,
    CudaHostPinnedSpace,
    CudaUVMSpace,

    // Only if HIP enabled
    HIPSpace,
    HIPHostPinnedSpace,
    HIPManagedSpace,

    // Only if SYCL enabled
    SYCLDeviceUSMSpace,
    SYCLHostUSMSpace,
    SYCLSharedUSMSpace,

    // Always enabled (maybe)
    HostSpace,
    SharedSpace,
    SharedHostPinnedSpace
    */
    return (
#ifdef KOKKOS_ENABLE_CUDA
        mem_space == std::type_index(typeid(Kokkos::CudaSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::CudaHostPinnedSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::CudaUVMSpace)) ||
#endif
#ifdef KOKKOS_ENABLE_HIP
        mem_space == std::type_index(typeid(Kokkos::HIPSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::HIPHostPinnedSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::HIPManagedSpace)) ||
#endif
#ifdef KOKKOS_ENABLE_SYCL
        mem_space == std::type_index(typeid(Kokkos::SYCLDeviceUSMSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::SYCLHostUSMSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::SYCLSharedUSMSpace)) ||
#endif
        mem_space == std::type_index(typeid(Kokkos::HostSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::SharedSpace)) ||
        mem_space == std::type_index(typeid(Kokkos::SharedHostPinnedSpace)));
  }

  static void assert_valid_mem_space(const std::type_index &mem_space) {
    MUNDY_THROW_REQUIRE(
        is_valid_mem_space(mem_space), std::invalid_argument,
        "MemSpaceJumpTable::assert_valid_mem_space: Given a mem_space that is not in our list of valid Kokkos memory "
        "spaces.\n"
        "Our current list of spaces is as follows:\n"
        "   CudaSpace, CudaHostPinnedSpace, CudaUVMSpace, HIPSpace, HIPHostPinnedSpace, HIPManagedSpace, \n"
        "   SYCLDeviceUSMSpace, SYCLHostUSMSpace, SYCLSharedUSMSpace, HostSpace, SharedSpace, SharedHostPinnedSpace\n"
        "If Kokkos has added a new memory space, please inform the Mundy developers.");
  }

  // partial_functor(func, first_arg) -> partial func with signature pfunc(rest_args...)
  template <typename Func, typename FirstArg>
  static auto partial_functor(const Func &func, FirstArg &&first_arg) {
    return
        [&](auto &&...args) { return func(std::forward<FirstArg>(first_arg), std::forward<decltype(args)>(args)...); };
  }
};