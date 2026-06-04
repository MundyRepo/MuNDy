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

#ifndef MUNDY_MESH_DECLARECLASS_HPP_
#define MUNDY_MESH_DECLARECLASS_HPP_

/// \file DeclareClass.hpp
/// \defgroup MundyMeshDeclareClass mundy::mesh::DeclareClass
/// \brief A set of helpers for declaring classes with reduced boilerplate code.

// C++ core
#include <iostream>   // for std::ostream
#include <memory>     // for std::shared_ptr
#include <stdexcept>  // for std::runtime_error
#include <type_traits>
#include <utility>
#include <vector>  // for std::vector

// Trilinos
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData

// Mundy
#include <mundy_mesh/Class.hpp>
#include <mundy_mesh/Component.hpp>
#include <mundy_mesh/FieldComponent.hpp>  // for mundy::mesh::impl::component_backing_field
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE
#include <Mundy_config.hpp>  // for MUNDY_DEPRECATED_MSG

namespace mundy {

namespace mesh {

/// \brief Helper class for declaring a class
///
/// This class is used to aid the declaration of a class on the mesh with reduced boilerplate.
/// It uses a fluent interface to set the class properties and then declare the class.
///
/// There are three types of classes that may be declared:
///   1. Named classes (name, but no rank or topology)
///   2. Ranked class-sets (name and rank, but no topology)
///   3. Topological primary classes (name and topology, but no rank)
///
/// You may not specify both a rank and a topology for the same class.
///
/// For example, to create a node-rank set that contains boundary and loading nodes:
/// \code{.cpp}
///   ClassDeclaration class_decl(meta_data);
///   Class& boundary_nodes = class_decl.name("boundary_nodes").rank(NODE_RANK).declare();
///   Class& loading_nodes  = class_decl.name("loading_nodes").rank(NODE_RANK).declare();
///   Class& all_nodes      =
///   class_decl.name("all_nodes").rank(NODE_RANK).subclass(boundary_nodes).subclass(loading_nodes).declare();
/// \endcode
///
/// These setters may be called in any order. Subclasses are optional, but you must call a valid combination of
/// name, rank, and topology before declare().
///
/// You may also reuse the same ClassDeclaration to declare multiple classes with similar properties:
/// \code{.cpp}
///   ClassDeclaration class_decl(meta_data);
///   auto particle_class_decl = class_decl.topology(stk::topology::PARTICLE).declare();
///   Class& spheres = particle_class_decl.name("spheres").declare();
///   Class& points  = particle_class_decl.name("points").declare();
/// \endcode
class ClassDeclaration {
 public:
  //! \name Constructors and Assignment Operators
  //@{

  /// \brief Canonical constructor.
  explicit ClassDeclaration(stk::mesh::MetaData& meta_data)
      : meta_data_(meta_data),
        class_has_name_(false),
        class_has_rank_(false),
        class_has_topology_(false),
        class_has_subclasses_(false),
        class_has_superclasses_(false) {
  }

  /// \brief Copy/Move constructors and assignment operators.
  ClassDeclaration(const ClassDeclaration&) = default;
  ClassDeclaration(ClassDeclaration&&) = default;
  ClassDeclaration& operator=(const ClassDeclaration&) = default;
  ClassDeclaration& operator=(ClassDeclaration&&) = default;
  //@}

  //! \name Fluent interface
  //@{

  /// \brief Set the name of the class (must be called before declare()).
  ClassDeclaration name(const std::string& class_name) {
    class_has_name_ = true;
    class_name_ = class_name;
    return *this;
  }

  /// \brief Set the entity rank of the class.
  ClassDeclaration rank(stk::mesh::EntityRank class_rank) {
    class_has_rank_ = true;
    class_rank_ = class_rank;
    return *this;
  }

  /// \brief Set the topology of the class.
  ClassDeclaration topology(stk::topology::topology_t class_topology) {
    class_has_topology_ = true;
    class_topology_ = class_topology;
    return *this;
  }

  /// \brief Add a subclass to the class (i.e, declare the given class as a subset of this class).
  ClassDeclaration subclass(const Class& subclass) {
    class_has_subclasses_ = true;
    subset_class_ids_.push_back(subclass.class_ordinal());
    return *this;
  }

  /// \brief Add a superclass to the class (i.e, declare this class as a subset of the given class).
  ClassDeclaration superclass(const Class& superclass) {
    class_has_superclasses_ = true;
    superset_class_ids_.push_back(superclass.class_ordinal());
    return *this;
  }

  /// \brief Create a scalar-valued field restriction for this class (optional) with the given initial value.
  template <typename FieldType>
  ClassDeclaration put_field(FieldType& field, const typename FieldType::value_type* init_value) {
    field_restrictions_.push_back(std::make_shared<DeclareScalarFieldRestriction<FieldType>>(field, init_value));
    return *this;
  }

  /// \brief Create a vector-valued field restriction for this class (optional) with the given initial value.
  template <typename FieldType>
  ClassDeclaration put_field(FieldType& field, unsigned n1, const typename FieldType::value_type* init_value) {
    field_restrictions_.push_back(std::make_shared<DeclareVectorFieldRestriction<FieldType>>(field, n1, init_value));
    return *this;
  }

  /// \brief Create a tensor-valued field restriction for this class (optional) with the given initial value.
  template <typename FieldType>
  ClassDeclaration put_field(FieldType& field, unsigned n1, unsigned n2,
                                   const typename FieldType::value_type* init_value) {
    field_restrictions_.push_back(
        std::make_shared<DeclareTensorFieldRestriction<FieldType>>(field, n1, n2, init_value));
    return *this;
  }

  /// \brief Create a field restriction directly from a field-backed component declaration.
  template <typename ComponentType, typename BackingFieldType = std::remove_cvref_t<
                                        decltype(impl::component_backing_field(std::declval<ComponentType&>()))>>
  MUNDY_REQUIRES(requires(ComponentType component) {
    typename std::remove_cvref_t<ComponentType>::canonical_access;
    { impl::component_backing_field(component) };
  })
  ClassDeclaration
      put_component(ComponentType component, const typename BackingFieldType::value_type* init_value) {
    using component_type = std::remove_cvref_t<ComponentType>;
    using access_shape = component_access_shape<typename component_type::canonical_access>;
    auto& field = impl::component_backing_field(component);

    static_assert(access_shape::has_fixed_field_scalars,
                  "put_component(...) requires a component whose access shape defines a fixed number of field scalars. "
                  "Use put_field(...) for raw field components.");

    if constexpr (access_shape::field_scalars == 1) {
      return put_field(field, init_value);
    } else {
      return put_field(field, access_shape::field_scalars, init_value);
    }
  }

  /// \brief Declare a class with the given properties.
  Class& declare() {
    MUNDY_THROW_REQUIRE(class_has_name_, std::logic_error, "Class name must be set before declaring a class.");

    const bool is_named_class = class_has_name_ && !class_has_rank_ && !class_has_topology_;
    const bool is_ranked_class = class_has_name_ && class_has_rank_ && !class_has_topology_;
    const bool is_topological_class = class_has_name_ && !class_has_rank_ && class_has_topology_;

    MUNDY_THROW_REQUIRE(
        is_named_class || is_ranked_class || is_topological_class, std::logic_error,
        sink() << "Class with name ('" << class_name_ << "') is not properly specified. You may either specify:\n"
               << "   1. A name (but no rank or topology)    -> declare_class(meta_data, 'name')\n"
               << "   2. A name and a rank (but no topology) -> declare_class(meta_data, 'name', rank)\n"
               << "   3. A name and a topology (but no rank) -> declare_class(meta_data, 'name', topology)\n"
               << "However, you have specified both a rank and a topology.");

    if (is_named_class) {
      return internal_declare_named_class();
    }
    if (is_ranked_class) {
      return internal_declare_ranked_class();
    }
    return internal_declare_topological_class();
  }

  /// \brief Print the class declaration information to the output stream.
  void print(std::ostream& os = std::cout) const {
    os << "ClassDeclaration:" << std::endl;
    if (class_has_name_) {
      os << "  Name: " << class_name_ << std::endl;
    }
    if (class_has_rank_) {
      os << "  Rank: " << class_rank_ << std::endl;
    }
    if (class_has_topology_) {
      os << "  Topology: " << stk::topology(class_topology_) << std::endl;
    }
    if (class_has_subclasses_) {
      os << "  Subclasses: ";
      for (Class::class_ordinal_t subclass_id : subset_class_ids_) {
        os << subclass_id << " ";
      }
      os << std::endl;
    }
    if (class_has_superclasses_) {
      os << "  Superclasses: ";
      for (Class::class_ordinal_t superclass_id : superset_class_ids_) {
        os << superclass_id << " ";
      }
      os << std::endl;
    }
  }
  //@}

 private:
  struct DeclareFieldRestrictionBase {
    virtual ~DeclareFieldRestrictionBase() = default;
    virtual void apply(Class& class_instance) = 0;
  };

  template <typename FieldType>
  struct DeclareScalarFieldRestriction : public DeclareFieldRestrictionBase {
    DeclareScalarFieldRestriction(FieldType& field, const typename FieldType::value_type* init_value)
        : field_(field), init_value_(init_value) {
    }

    void apply(Class& class_instance) override {
      put_field_on_mesh(field_, class_instance, init_value_);
    }

   private:
    FieldType& field_;
    const typename FieldType::value_type* init_value_;
  };

  template <typename FieldType>
  struct DeclareVectorFieldRestriction : public DeclareFieldRestrictionBase {
    DeclareVectorFieldRestriction(FieldType& field, unsigned n1, const typename FieldType::value_type* init_value)
        : field_(field), n1_(n1), init_value_(init_value) {
    }

    void apply(Class& class_instance) override {
      put_field_on_mesh(field_, class_instance, n1_, init_value_);
    }

   private:
    FieldType& field_;
    unsigned n1_;
    const typename FieldType::value_type* init_value_;
  };

  template <typename FieldType>
  struct DeclareTensorFieldRestriction : public DeclareFieldRestrictionBase {
    DeclareTensorFieldRestriction(FieldType& field, unsigned n1, unsigned n2,
                                  const typename FieldType::value_type* init_value)
        : field_(field), n1_(n1), n2_(n2), init_value_(init_value) {
    }

    void apply(Class& class_instance) override {
      put_field_on_mesh(field_, class_instance, n1_, n2_, init_value_);
    }

   private:
    FieldType& field_;
    unsigned n1_;
    unsigned n2_;
    const typename FieldType::value_type* init_value_;
  };

  void apply_optional_properties(Class& class_instance) {
    if (class_has_superclasses_) {
      for (Class::class_ordinal_t superclass_id : superset_class_ids_) {
        Class& superclass_instance = get_class(meta_data_, superclass_id);
        declare_subset(superclass_instance, class_instance);
      }
    }

    if (class_has_subclasses_) {
      for (Class::class_ordinal_t subclass_id : subset_class_ids_) {
        Class& subclass_instance = get_class(meta_data_, subclass_id);
        declare_subset(class_instance, subclass_instance);
      }
    }

    for (const auto& restriction : field_restrictions_) {
      restriction->apply(class_instance);
    }
  }

  Class& internal_declare_named_class() {
    Class& class_instance = declare_class(meta_data_, class_name_);
    apply_optional_properties(class_instance);
    return class_instance;
  }

  Class& internal_declare_ranked_class() {
    Class& class_instance = declare_class(meta_data_, class_name_, class_rank_);
    apply_optional_properties(class_instance);
    return class_instance;
  }

  Class& internal_declare_topological_class() {
    Class& class_instance = declare_class(meta_data_, class_name_, class_topology_);
    apply_optional_properties(class_instance);
    return class_instance;
  }

  stk::mesh::MetaData& meta_data_;

  bool class_has_name_;
  bool class_has_rank_;
  bool class_has_topology_;
  bool class_has_subclasses_;
  bool class_has_superclasses_;

  std::string class_name_;
  stk::mesh::EntityRank class_rank_ = stk::topology::INVALID_RANK;
  stk::topology::topology_t class_topology_ = stk::topology::INVALID_TOPOLOGY;
  std::vector<Class::class_ordinal_t> subset_class_ids_;
  std::vector<Class::class_ordinal_t> superset_class_ids_;
  std::vector<std::shared_ptr<DeclareFieldRestrictionBase>> field_restrictions_;
};

using ClassDeclarationHelper MUNDY_DEPRECATED_MSG("use ClassDeclaration") = ClassDeclaration;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_DECLARECLASS_HPP_
