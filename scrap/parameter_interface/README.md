# The previous MetaMethods design was close but had some flaws.

# The good
MetaMethods encapsulated both the requirements for the mesh and the requirements for input parameter files.
This meant that disabling a method removed its parameters from the parameter file. It also allowed users to
inject their own custom methods into mundy and have them used by simply specifying a different string name.

This flavor of requirements encapsulation and code injection is something we still want within Mundy's C++
interface. This can be seen within applications, such as aLENS, where their parameter files are quickly
becoming inextensible without clarity oh who uses the parameters and why. This has already led to confusion
and bloat within even simplistic simulations.

So far, we've done in the past year is fledge out the C++ interface and apply it directly to applications
without relying on either the parameter interface or user API. We'll need to fledge out a formal API in the
coming 6 months (target of mid-summer 2026).

# The bad
The issue with MetaMethods was the visualization of information flow/pipelining, especially when the user
wanted to mix in their own custom methods. Part of the problem was that it took so long to add a new 
MetaMethod, leading to the attempt to mix raw C++ with this parameter interface, which ended terribly. We
redesigned meta methods multiple times. The design that has the most potential was macro-based meta methods.
See scrap/hp1_mock_reworks for details. In summary, the macro-based design reduced boilerplate significantly
by limiting what meta methods could or could not take in while also handling the automatic registration of 
each method and its various setters/getters.

MUNDY_METHOD(mundy::alens, GhostLinkedEntities) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_SELECTOR(selector);
  MUNDY_METHOD_HAS_FIELD(double, constraint_linked_entities_field);
  MUNDY_METHOD_HAS_FIELD(int, constraint_linked_entity_owners_field);

 public:
  void run() override {
    bulk_data_ptr_->modification_begin();
    mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr_, *constraint_linked_entities_field_ptr_,
                                                 *constraint_linked_entity_owners_field_ptr_, *selector_ptr_);
    bulk_data_ptr_->modification_end();
  }
};

Ideally, this design would lead to an equivalent method being created in the python interface automatically.
And, if we go down the path of delayed vs direct evaluation of a DAG, then this could also handle things like
setting up the delayed evaluation wrapper.

# Conclusion
For now, we are deprecating every Mundy sub-package that depends on meta methods until we formalize their interface.