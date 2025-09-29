// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
//
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_MESH_IMPL_NGPFIELDBLASIMPL_HPP_
#define MUNDY_MESH_IMPL_NGPFIELDBLASIMPL_HPP_

/// \file FieldBLAS.hpp
/// \brief A set of BLAS-like operations for stk::mesh::FieldBase objects

// C++ core
#include <algorithm>
#include <complex>
#include <iostream>
#include <string>

// External
#include <fmt/format.h>       // for fmt::format
#include <openrand/philox.h>  // for openrand::Philox

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_util/stk_config.h>

#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/DataTraits.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_mesh/NgpUtils.hpp>  // is_(ngp|device|host)_field, is_(ngp|device|host)_mesh, ngp_ngp_field_and_mesh_compatible
#include <mundy_mesh/fmt_stk_types.hpp>  // for fmt::format support for stk types

namespace mundy {

namespace mesh {

namespace impl {

template <class Field>
struct FieldFill {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldFill requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldFill(Field& field, const value_type alpha) : field_(field), alpha_(alpha) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    const int num_components = field_.get_num_components_per_entity(f);
    for (int d = 0; d < num_components; ++d) {
      field_(f, d) = alpha_;
    }
  }

 private:
  Field field_;
  const value_type alpha_;
};

template <class Field>
struct FieldFillComponent {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldFillComponent requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldFillComponent(Field& field, const value_type alpha, const int component)
      : field_(field), alpha_(alpha), component_(component) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    const int num_components = field_.get_num_components_per_entity(f);
    MUNDY_THROW_ASSERT(component_ < num_components, std::out_of_range,
                       fmt::format("Component index {} is out of bounds for field {}", component_, field_.get_name()));
    field_(f, component_) = alpha_;
  }

 private:
  Field field_;
  const value_type alpha_;
  const int component_;
};

template <class Field, class CounterField>
struct FieldRandomize {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldRandomize requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldRandomize(Field& field, const size_t seed, CounterField& counter_field)
      : field_(field), seed_(seed), counter_field_(counter_field) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    auto& counter = counter_field_(f, 0);
    openrand::Philox rng(seed_, counter);

    const int num_components = field_.get_num_components_per_entity(f);
    for (int d = 0; d < num_components; ++d) {
      field_(f, d) = rng.rand<value_type>();
    }

    counter++;
  }

 private:
  Field field_;
  const size_t seed_;
  CounterField counter_field_;
};

template <class Field, class CounterField>
struct FieldRandomizeMinMax {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldRandomizeMinMax requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldRandomizeMinMax(Field& field, const size_t seed, CounterField& counter_field, const value_type min,
                       const value_type max)
      : field_(field), seed_(seed), counter_field_(counter_field), min_(min), max_(max) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    auto& counter = counter_field_(f, 0);
    openrand::Philox rng(seed_, counter);

    const int num_components = field_.get_num_components_per_entity(f);
    for (int d = 0; d < num_components; ++d) {
      field_(f, d) = rng.uniform<value_type>(min_, max_);
    }

    counter++;
  }

 private:
  Field field_;
  const size_t seed_;
  CounterField counter_field_;
  const value_type min_;
  const value_type max_;
};

template <class Field, class CounterField>
struct FieldRandomizeComponent {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldRandomizeComponent requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldRandomizeComponent(Field& field, const size_t seed, CounterField& counter_field, const int component)
      : field_(field), seed_(seed), counter_field_(counter_field), component_(component) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    const int num_components = field_.get_num_components_per_entity(f);
    MUNDY_THROW_ASSERT(component_ < num_components, std::out_of_range,
                       fmt::format("Component index {} is out of bounds for field {}", component_, field_.get_name()));

    auto& counter = counter_field_(f, 0);
    openrand::Philox rng(seed_, counter);

    field_(f, component_) = rng.rand<value_type>();
    counter++;
  }

 private:
  Field field_;
  const size_t seed_;
  CounterField counter_field_;
  const int component_;
};

template <class Field, class CounterField>
struct FieldRandomizeComponentMinMax {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldRandomizeComponent requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldRandomizeComponentMinMax(Field& field, const size_t seed, CounterField& counter_field, const int component,
                                const value_type min, const value_type max)
      : field_(field), seed_(seed), counter_field_(counter_field), component_(component), min_(min), max_(max) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    const int num_components = field_.get_num_components_per_entity(f);
    MUNDY_THROW_ASSERT(component_ < num_components, std::out_of_range,
                       fmt::format("Component index {} is out of bounds for field {}", component_, field_.get_name()));

    auto& counter = counter_field_(f, 0);
    openrand::Philox rng(seed_, counter);

    field_(f, component_) = rng.uniform<value_type>(min_, max_);
    counter++;
  }

 private:
  Field field_;
  const size_t seed_;
  CounterField counter_field_;
  const int component_;
  const value_type min_;
  const value_type max_;
};

template <class Field>
struct FieldCopy {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldCopy requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldCopy(Field& field_x, Field& field_y) : field_x_(field_x), field_y_(field_y) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    const unsigned num_components = field_x_.get_num_components_per_entity(f);
    MUNDY_THROW_ASSERT(num_components == field_y_.get_num_components_per_entity(f), std::runtime_error,
                       "Field components mismatch in FieldCopy");
    for (unsigned d = 0; d < num_components; ++d) {
      field_y_(f, d) = field_x_(f, d);
    }
  }

 private:
  Field field_x_;
  Field field_y_;
};

template <class Field>
struct FieldAXPBYZFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldAXPBYZFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldAXPBYZFunctor(const value_type alpha, Field& field_x, const value_type beta, Field& field_y, Field& field_z)
      : field_x_(field_x), field_y_(field_y), field_z_(field_z), alpha_(alpha), beta_(beta) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    // Use the smallest number of components of the three fields
    unsigned num_components = field_z_.get_num_components_per_entity(f);
    unsigned other = field_x_.get_num_components_per_entity(f);
    num_components = (other < num_components) ? other : num_components;
    other = field_y_.get_num_components_per_entity(f);
    num_components = (other < num_components) ? other : num_components;

    for (unsigned d = 0; d < num_components; ++d) {
      field_z_.get(f, d) = alpha_ * field_x_.get(f, d) + beta_ * field_y_.get(f, d);
    }
  }

 private:
  Field field_x_;
  Field field_y_;
  Field field_z_;

  const value_type alpha_;
  const value_type beta_;
};

template <class Field>
struct FieldAXPBYGZFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldAXPBYGZFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldAXPBYGZFunctor(const value_type alpha, Field& field_x, const value_type beta, Field& field_y,
                      const value_type gamma, Field& field_z)
      : field_x_(field_x), field_y_(field_y), field_z_(field_z), alpha_(alpha), beta_(beta), gamma_(gamma) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f) const {
    // Use the smallest number of components of the three fields
    unsigned num_components = field_z_.get_num_components_per_entity(f);
    unsigned other = field_x_.get_num_components_per_entity(f);
    num_components = (other < num_components) ? other : num_components;
    other = field_y_.get_num_components_per_entity(f);
    num_components = (other < num_components) ? other : num_components;

    for (unsigned d = 0; d < num_components; ++d) {
      field_z_.get(f, d) = alpha_ * field_x_.get(f, d) + beta_ * field_y_.get(f, d) + gamma_ * field_z_.get(f, d);
    }
  }

 private:
  Field field_x_;
  Field field_y_;
  Field field_z_;

  const value_type alpha_;
  const value_type beta_;
  const value_type gamma_;
};

template <class Field>
struct FieldProductFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldProductFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldProductFunctor(Field& field_x, Field& field_y, Field& field_z)
      : field_x_(field_x), field_y_(field_y), field_z_(field_z) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(stk::mesh::FastMeshIndex f) const {
    // Use the smallest number of components of the three fields
    unsigned num_components = field_z_.get_num_components_per_entity(f);
    unsigned other = field_x_.get_num_components_per_entity(f);
    num_components = (other < num_components) ? other : num_components;
    other = field_y_.get_num_components_per_entity(f);
    num_components = (other < num_components) ? other : num_components;
    for (unsigned d = 0; d < num_components; ++d) {
      field_z_.get(f, d) = field_x_.get(f, d) * field_y_.get(f, d);
    }
  }

 private:
  Field field_x_;
  Field field_y_;
  Field field_z_;
};

template <class Field>
struct FieldScaleFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldScaleFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldScaleFunctor(Field& field, const value_type alpha) : field_(field), alpha_(alpha) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(stk::mesh::FastMeshIndex f) const {
    unsigned num_components = field_.get_num_components_per_entity(f);
    for (unsigned d = 0; d < num_components; ++d) {
      field_.get(f, d) = alpha_ * field_.get(f, d);
    }
  }

 private:
  Field field_;
  const value_type alpha_;
};

template <class Field>
struct FieldSwapFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldSwapFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldSwapFunctor(Field& field_x, Field& field_y) : field_x_(field_x), field_y_(field_y) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(stk::mesh::FastMeshIndex f) const {
    unsigned num_components = field_x_.get_num_components_per_entity(f);
    unsigned other = field_y_.get_num_components_per_entity(f);
    num_components = (other < num_components) ? other : num_components;
    for (unsigned d = 0; d < num_components; ++d) {
      value_type tmp = field_x_.get(f, d);
      field_x_.get(f, d) = field_y_.get(f, d);
      field_y_.get(f, d) = tmp;
    }
  }

 private:
  Field field_x_;
  Field field_y_;
};

template <typename Field>
struct FieldDotReductionFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldDotReductionFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldDotReductionFunctor(Field& field_x, Field& field_y, Kokkos::Sum<value_type> sum_reduction)
      : field_x_(field_x), field_y_(field_y), sum_reduction_(sum_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f, value_type& value) const {
    const unsigned num_components = field_x_.get_num_components_per_entity(f);
    MUNDY_THROW_ASSERT(num_components == field_y_.get_num_components_per_entity(f), std::runtime_error,
                       "Field components mismatch in FieldDotReductionFunctor");
    for (unsigned j = 0; j < num_components; ++j) {
      const value_type prod = field_x_.get(f, j) * field_y_.get(f, j);
      sum_reduction_.join(value, prod);
    }
  }

 private:
  const Field field_x_;
  const Field field_y_;
  const Kokkos::Sum<value_type> sum_reduction_;
};

template <typename Field>
struct FieldSumReductionFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldSumReductionFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldSumReductionFunctor(Field& field, Kokkos::Sum<value_type> sum_reduction)
      : field_(field), sum_reduction_(sum_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f, value_type& value) const {
    const unsigned num_components = field_.get_num_components_per_entity(f);
    for (unsigned j = 0; j < num_components; ++j) {
      sum_reduction_.join(value, field_.get(f, j));
    }
  }

 private:
  const Field field_;
  const Kokkos::Sum<value_type> sum_reduction_;
};

template <typename Field>
struct FieldAbsSumReductionFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldAbsSumReductionFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldAbsSumReductionFunctor(Field& field, Kokkos::Sum<value_type> sum_reduction)
      : field_(field), sum_reduction_(sum_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f, value_type& value) const {
    const unsigned num_components = field_.get_num_components_per_entity(f);
    for (unsigned j = 0; j < num_components; ++j) {
      sum_reduction_.join(value, Kokkos::abs(field_.get(f, j)));
    }
  }

 private:
  const Field field_;
  const Kokkos::Sum<value_type> sum_reduction_;
};

template <typename Field>
struct FieldMaxReductionFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldMaxReductionFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldMaxReductionFunctor(Field& field, Kokkos::Max<value_type> max_reduction)
      : field_(field), max_reduction_(max_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f, value_type& value) const {
    const unsigned num_components = field_.get_num_components_per_entity(f);
    for (unsigned j = 0; j < num_components; ++j) {
      max_reduction_.join(value, field_.get(f, j));
    }
  }

 private:
  const Field field_;
  const Kokkos::Max<value_type> max_reduction_;
};

template <typename Field>
struct FieldAbsMaxReductionFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldAbsMaxReductionFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldAbsMaxReductionFunctor(Field& field, Kokkos::Max<value_type> max_reduction)
      : field_(field), max_reduction_(max_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f, value_type& value) const {
    const unsigned num_components = field_.get_num_components_per_entity(f);
    for (unsigned j = 0; j < num_components; ++j) {
      max_reduction_.join(value, Kokkos::abs(field_.get(f, j)));
    }
  }

 private:
  const Field field_;
  const Kokkos::Max<value_type> max_reduction_;
};

template <typename Field>
struct FieldMinReductionFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldMinReductionFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldMinReductionFunctor(Field& field, Kokkos::Min<value_type> min_reduction)
      : field_(field), min_reduction_(min_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f, value_type& value) const {
    const unsigned num_components = field_.get_num_components_per_entity(f);
    for (unsigned j = 0; j < num_components; ++j) {
      min_reduction_.join(value, field_.get(f, j));
    }
  }

 private:
  const Field field_;
  const Kokkos::Min<value_type> min_reduction_;
};

template <typename Field>
struct FieldAbsMinReductionFunctor {
  using value_type = typename Field::value_type;
  static_assert(is_ngp_field<Field>, "FieldAbsMinReductionFunctor requires an stk::mesh::NgpField");

  KOKKOS_FUNCTION
  FieldAbsMinReductionFunctor(Field& field, Kokkos::Min<value_type> min_reduction)
      : field_(field), min_reduction_(min_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex& f, value_type& value) const {
    const unsigned num_components = field_.get_num_components_per_entity(f);
    for (unsigned j = 0; j < num_components; ++j) {
      min_reduction_.join(value, Kokkos::abs(field_.get(f, j)));
    }
  }

 private:
  const Field field_;
  const Kokkos::Min<value_type> min_reduction_;
};

// requires(std::is_same_v<FieldBases, stk::mesh::FieldBase> && ...)

template <typename... FieldBases>
stk::mesh::Selector if_nullptr_select_fields(const stk::mesh::Selector* const selector_ptr,
                                             const FieldBases&... field_bases) {
  if (selector_ptr == nullptr) {
    // Fold expression on the & operator to combine the field selectors
    return (stk::mesh::Selector(field_bases) & ...);
  } else {
    return *selector_ptr;
  }
}

/// \brief Fill a component of a field with a scalar value
template <typename Scalar, typename ExecSpace>
void ngp_field_fill_component(const Scalar alpha,                             //
                              stk::mesh::FieldBase& field,                    //
                              int component,                                  //
                              const stk::mesh::Selector* const selector_ptr,  //
                              const ExecSpace& exec_space) {
  sync_field_to_space(field, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field = stk::mesh::get_updated_ngp_field<Scalar>(field);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field.get_mesh());

  FieldFillComponent<NgpScalarField> functor(ngp_field, alpha, component);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field, exec_space);
}

/// \brief Fill all components of a field with a scalar value
template <typename Scalar, typename ExecSpace>
void ngp_field_fill(const Scalar alpha,                             //
                    stk::mesh::FieldBase& field,                    //
                    const stk::mesh::Selector* const selector_ptr,  //
                    const ExecSpace& exec_space) {
  sync_field_to_space(field, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field = stk::mesh::get_updated_ngp_field<Scalar>(field);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field.get_mesh());

  FieldFill<NgpScalarField> functor(ngp_field, alpha);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field, exec_space);
}

/// \brief Randomize a component of a field (uniform between 0 and 1)
template <typename Scalar, typename ExecSpace>
void ngp_field_randomize_component(const size_t seed,                              //
                                   stk::mesh::FieldBase& counter_field,            //
                                   stk::mesh::FieldBase& field,                    //
                                   int component,                                  //
                                   const stk::mesh::Selector* const selector_ptr,  //
                                   const ExecSpace& exec_space) {
  sync_field_to_space(field, exec_space);
  sync_field_to_space(counter_field, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field, counter_field);

  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  using NgpCounterField = stk::mesh::NgpField<size_t>;
  NgpScalarField ngp_field = stk::mesh::get_updated_ngp_field<Scalar>(field);
  NgpCounterField ngp_counter_field = stk::mesh::get_updated_ngp_field<size_t>(counter_field);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field.get_mesh());

  FieldRandomizeComponent<NgpScalarField, NgpCounterField> functor(ngp_field, seed, ngp_counter_field, component);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field, exec_space);
  mark_field_modified_on_space(counter_field, exec_space);
}

// \brief Randomize a component of a field (between given min and max)
template <typename Scalar, typename ExecSpace>
void ngp_field_randomize_component(const size_t seed,                              //
                                   const Scalar min, const Scalar max,             //
                                   stk::mesh::FieldBase& counter_field,            //
                                   stk::mesh::FieldBase& field,                    //
                                   int component,                                  //
                                   const stk::mesh::Selector* const selector_ptr,  //
                                   const ExecSpace& exec_space) {
  sync_field_to_space(field, exec_space);
  sync_field_to_space(counter_field, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field, counter_field);

  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  using NgpCounterField = stk::mesh::NgpField<size_t>;
  NgpScalarField ngp_field = stk::mesh::get_updated_ngp_field<Scalar>(field);
  NgpCounterField ngp_counter_field = stk::mesh::get_updated_ngp_field<size_t>(counter_field);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field.get_mesh());

  FieldRandomizeComponentMinMax<NgpScalarField, NgpCounterField> functor(ngp_field, seed, ngp_counter_field, component,
                                                                         min, max);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field, exec_space);
  mark_field_modified_on_space(counter_field, exec_space);
}

/// \brief Randomize a field (uniform between 0 and 1)
template <typename Scalar, typename ExecSpace>
void ngp_field_randomize(const size_t seed,                              //
                         stk::mesh::FieldBase& counter_field,            //
                         stk::mesh::FieldBase& field,                    //
                         const stk::mesh::Selector* const selector_ptr,  //
                         const ExecSpace& exec_space) {
  sync_field_to_space(field, exec_space);
  sync_field_to_space(counter_field, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field, counter_field);

  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  using NgpCounterField = stk::mesh::NgpField<size_t>;
  NgpScalarField ngp_field = stk::mesh::get_updated_ngp_field<Scalar>(field);
  NgpCounterField ngp_counter_field = stk::mesh::get_updated_ngp_field<size_t>(counter_field);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field.get_mesh());

  FieldRandomize<NgpScalarField, NgpCounterField> functor(ngp_field, seed, ngp_counter_field);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field, exec_space);
  mark_field_modified_on_space(counter_field, exec_space);
}

// \brief Randomize a field (between given min and max)
template <typename Scalar, typename ExecSpace>
void ngp_field_randomize(const size_t seed,                              //
                         const Scalar min, const Scalar max,             //
                         stk::mesh::FieldBase& counter_field,            //
                         stk::mesh::FieldBase& field,                    //
                         const stk::mesh::Selector* const selector_ptr,  //
                         const ExecSpace& exec_space) {
  sync_field_to_space(field, exec_space);
  sync_field_to_space(counter_field, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field, counter_field);

  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  using NgpCounterField = stk::mesh::NgpField<size_t>;
  NgpScalarField ngp_field = stk::mesh::get_updated_ngp_field<Scalar>(field);
  NgpCounterField ngp_counter_field = stk::mesh::get_updated_ngp_field<size_t>(counter_field);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field.get_mesh());

  FieldRandomizeMinMax<NgpScalarField, NgpCounterField> functor(ngp_field, seed, ngp_counter_field, min, max);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field, exec_space);
  mark_field_modified_on_space(counter_field, exec_space);
}

/// \brief Deep copy y = x
template <typename Scalar, typename ExecSpace>
void ngp_field_copy(stk::mesh::FieldBase& field_x,                  //
                    stk::mesh::FieldBase& field_y,                  //
                    const stk::mesh::Selector* const selector_ptr,  //
                    const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x, field_y);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  NgpScalarField ngp_field_y = stk::mesh::get_updated_ngp_field<Scalar>(field_y);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  FieldCopy<NgpScalarField> functor(ngp_field_x, ngp_field_y);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field_x.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field_y, exec_space);
}

/// \brief Swap the contents of two fields
template <typename Scalar, typename ExecSpace>
void ngp_field_swap(stk::mesh::FieldBase& field_x,                  //
                    stk::mesh::FieldBase& field_y,                  //
                    const stk::mesh::Selector* const selector_ptr,  //
                    const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  sync_field_to_space(field_y, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x, field_y);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  NgpScalarField ngp_field_y = stk::mesh::get_updated_ngp_field<Scalar>(field_y);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  FieldSwapFunctor<NgpScalarField> functor(ngp_field_x, ngp_field_y);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field_x.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field_x, exec_space);
  mark_field_modified_on_space(field_y, exec_space);
}

/// \brief Scale a field by a scalar x = alpha x
template <typename Scalar, typename ExecSpace>
void ngp_field_scale(const Scalar alpha,                             //
                     stk::mesh::FieldBase& field_x,                  //
                     const stk::mesh::Selector* const selector_ptr,  //
                     const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  FieldScaleFunctor<NgpScalarField> functor(ngp_field_x, alpha);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field_x.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field_x, exec_space);
}

/// \brief Compute the element-wise product of two fields z = x * y
template <typename Scalar, typename ExecSpace>
void ngp_field_product(stk::mesh::FieldBase& field_x,                  //
                       stk::mesh::FieldBase& field_y,                  //
                       stk::mesh::FieldBase& field_z,                  //
                       const stk::mesh::Selector* const selector_ptr,  //
                       const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  sync_field_to_space(field_y, exec_space);
  sync_field_to_space(field_z, exec_space);

  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x, field_y, field_z);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  NgpScalarField ngp_field_y = stk::mesh::get_updated_ngp_field<Scalar>(field_y);
  NgpScalarField ngp_field_z = stk::mesh::get_updated_ngp_field<Scalar>(field_z);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  FieldProductFunctor<NgpScalarField> functor(ngp_field_x, ngp_field_y, ngp_field_z);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field_x.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field_z, exec_space);
}

/// \brief Compute the element-wise sum of three fields z = alpha x + beta y
template <typename Scalar, typename ExecSpace>
void ngp_field_axpbyz(const Scalar alpha,                             //
                      stk::mesh::FieldBase& field_x,                  //
                      const Scalar beta,                              //
                      stk::mesh::FieldBase& field_y,                  //
                      stk::mesh::FieldBase& field_z,                  //
                      const stk::mesh::Selector* const selector_ptr,  //
                      const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  sync_field_to_space(field_y, exec_space);
  sync_field_to_space(field_z, exec_space);

  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x, field_y, field_z);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  NgpScalarField ngp_field_y = stk::mesh::get_updated_ngp_field<Scalar>(field_y);
  NgpScalarField ngp_field_z = stk::mesh::get_updated_ngp_field<Scalar>(field_z);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  FieldAXPBYZFunctor<NgpScalarField> functor(alpha, ngp_field_x, beta, ngp_field_y, ngp_field_z);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field_x.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field_z, exec_space);
}

/// \brief Compute the element-wise sum of three fields z = alpha x + beta y + gamma z
template <typename Scalar, typename ExecSpace>
void ngp_field_axpbygz(const Scalar alpha,                             //
                       stk::mesh::FieldBase& field_x,                  //
                       const Scalar beta,                              //
                       stk::mesh::FieldBase& field_y,                  //
                       const Scalar gamma,                             //
                       stk::mesh::FieldBase& field_z,                  //
                       const stk::mesh::Selector* const selector_ptr,  //
                       const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  sync_field_to_space(field_y, exec_space);
  sync_field_to_space(field_z, exec_space);

  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x, field_y, field_z);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  NgpScalarField ngp_field_y = stk::mesh::get_updated_ngp_field<Scalar>(field_y);
  NgpScalarField ngp_field_z = stk::mesh::get_updated_ngp_field<Scalar>(field_z);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  FieldAXPBYGZFunctor<NgpScalarField> functor(alpha, ngp_field_x, beta, ngp_field_y, gamma, ngp_field_z);
  stk::mesh::for_each_entity_run(ngp_mesh, ngp_field_x.get_rank(), field_selector, functor);

  mark_field_modified_on_space(field_z, exec_space);
}

/// \brief Compute the dot product of two fields
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_dot(stk::mesh::FieldBase& field_x,                  //
                            stk::mesh::FieldBase& field_y,                  //
                            const stk::mesh::Selector* const selector_ptr,  //
                            const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  sync_field_to_space(field_y, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x, field_y);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  NgpScalarField ngp_field_y = stk::mesh::get_updated_ngp_field<Scalar>(field_y);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  Scalar local_dot;
  Kokkos::Sum<Scalar> sum_reduction(local_dot);
  FieldDotReductionFunctor<NgpScalarField> functor(ngp_field_x, ngp_field_y, sum_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, ngp_field_x.get_rank(), field_selector, sum_reduction, functor);

  // MPI reduction to get the global dot product
  Scalar global_dot = 0;
  stk::all_reduce_sum(field_x.get_mesh().parallel(), &local_dot, &global_dot, 1);
  return global_dot;
}

/// \brief Compute the 2-norm of a field (i.e., the square root of the dot product of the field with itself)
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_nrm2(stk::mesh::FieldBase& field_x,                  //
                             const stk::mesh::Selector* const selector_ptr,  //
                             const ExecSpace& exec_space) {
  return Kokkos::sqrt(ngp_field_dot<Scalar>(field_x, field_x, selector_ptr, exec_space));
}

/// \brief Compute the sum of a field
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_sum(stk::mesh::FieldBase& field_x,                  //
                            const stk::mesh::Selector* const selector_ptr,  //
                            const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  Scalar local_sum;
  Kokkos::Sum<Scalar> sum_reduction(local_sum);
  FieldSumReductionFunctor<NgpScalarField> functor(ngp_field_x, sum_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, ngp_field_x.get_rank(), field_selector, sum_reduction, functor);

  // MPI reduction to get the global sum
  Scalar global_sum = 0;
  stk::all_reduce_sum(field_x.get_mesh().parallel(), &local_sum, &global_sum, 1);
  return global_sum;
}

/// \brief Compute the 1-norm of a field
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_asum(stk::mesh::FieldBase& field_x,                  //
                             const stk::mesh::Selector* const selector_ptr,  //
                             const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  stk::mesh::NgpField<Scalar> ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  Scalar local_sum;
  Kokkos::Sum<Scalar> sum_reduction(local_sum);
  FieldAbsSumReductionFunctor<NgpScalarField> functor(ngp_field_x, sum_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, ngp_field_x.get_rank(), field_selector, sum_reduction, functor);

  // MPI reduction to get the global sum
  Scalar global_sum = 0;
  stk::all_reduce_sum(field_x.get_mesh().parallel(), &local_sum, &global_sum, 1);
  return global_sum;
}

/// \brief Compute the maximum value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_max(stk::mesh::FieldBase& field_x,                  //
                            const stk::mesh::Selector* const selector_ptr,  //
                            const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  Scalar local_max;
  Kokkos::Max<Scalar> max_reduction(local_max);
  FieldMaxReductionFunctor<NgpScalarField> functor(ngp_field_x, max_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, ngp_field_x.get_rank(), field_selector, max_reduction, functor);

  // MPI reduction to get the global max
  Scalar global_max = 0;
  stk::all_reduce_max(field_x.get_mesh().parallel(), &local_max, &global_max, 1);
  return global_max;
}

/// \brief Compute the maximum absolute value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_amax(stk::mesh::FieldBase& field_x,                  //
                             const stk::mesh::Selector* const selector_ptr,  //
                             const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  Scalar local_max;
  Kokkos::Max<Scalar> max_reduction(local_max);
  FieldAbsMaxReductionFunctor<NgpScalarField> functor(ngp_field_x, max_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, ngp_field_x.get_rank(), field_selector, max_reduction, functor);

  // MPI reduction to get the global max
  Scalar global_max = 0;
  stk::all_reduce_max(field_x.get_mesh().parallel(), &local_max, &global_max, 1);
  return global_max;
}

/// \brief Compute the minimum value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_min(stk::mesh::FieldBase& field_x,                  //
                            const stk::mesh::Selector* const selector_ptr,  //
                            const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  Scalar local_min;
  Kokkos::Min<Scalar> min_reduction(local_min);
  FieldMinReductionFunctor<NgpScalarField> functor(ngp_field_x, min_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, ngp_field_x.get_rank(), field_selector, min_reduction, functor);

  // MPI reduction to get the global min
  Scalar global_min = 0;
  stk::all_reduce_min(field_x.get_mesh().parallel(), &local_min, &global_min, 1);
  return global_min;
}

/// \brief Compute the minimum absolute value of a field
template <typename Scalar, typename ExecSpace>
inline Scalar ngp_field_amin(stk::mesh::FieldBase& field_x,                  //
                             const stk::mesh::Selector* const selector_ptr,  //
                             const ExecSpace& exec_space) {
  sync_field_to_space(field_x, exec_space);
  stk::mesh::Selector field_selector = if_nullptr_select_fields(selector_ptr, field_x);
  using NgpScalarField = stk::mesh::NgpField<Scalar>;
  NgpScalarField ngp_field_x = stk::mesh::get_updated_ngp_field<Scalar>(field_x);
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(field_x.get_mesh());

  Scalar local_min;
  Kokkos::Min<Scalar> min_reduction(local_min);
  FieldAbsMinReductionFunctor<NgpScalarField> functor(ngp_field_x, min_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, ngp_field_x.get_rank(), field_selector, min_reduction, functor);

  // MPI reduction to get the global min
  Scalar global_min = 0;
  stk::all_reduce_min(field_x.get_mesh().parallel(), &local_min, &global_min, 1);
  return global_min;
}

}  // namespace impl

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_IMPL_NGPFIELDBLASIMPL_HPP_
