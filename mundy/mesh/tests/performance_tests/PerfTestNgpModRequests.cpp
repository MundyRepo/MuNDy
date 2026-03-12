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

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

// External
#include "nanobench.h"

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/NgpModRequests.hpp>
#include <mundy_utils/rng.hpp>

namespace mundy {
namespace mesh {
namespace {

struct MeshContext {
  std::shared_ptr<MetaData> meta_data_ptr;
  std::shared_ptr<BulkData> bulk_data_ptr;
  stk::mesh::Part* sphere_part{nullptr};
  stk::mesh::Part* bacteria_part{nullptr};
};

MeshContext create_context() {
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  MeshContext context;
  context.meta_data_ptr = builder.create_meta_data();
  context.meta_data_ptr->use_simple_fields();
  context.bulk_data_ptr = builder.create_bulk_data(context.meta_data_ptr);
  context.sphere_part = &context.meta_data_ptr->declare_part_with_topology("SPHERE_PART", stk::topology::PARTICLE);
  context.bacteria_part = &context.meta_data_ptr->declare_part_with_topology("BACTERIA_PART", stk::topology::PARTICLE);
  context.meta_data_ptr->commit();
  return context;
}

struct Example1CreateSpheresCase {
  size_t num_requests{0};
  MeshContext context;

  void setup(size_t requests) {
    num_requests = requests;
    context = create_context();
  }

  void run() {
    BulkData& bulk = *context.bulk_data_ptr;

    NgpModRequests reqs;
    stk::mesh::PartVector no_parts;
    auto req_spheres = reqs.request_entities_new_ids(*context.sphere_part);
    auto req_nodes = reqs.request_entities_new_ids(no_parts);
    auto req_conns = reqs.request_connections();

    reqs.activate_device();
    Kokkos::parallel_for(
        "PerfTestNgpModRequests::Example1Claim", Kokkos::RangePolicy<size_t>(0, num_requests),
        KOKKOS_LAMBDA(const size_t) {
          req_spheres.element_tickets().claim();
          req_nodes.node_tickets().claim();
          req_conns.tickets().claim();
        });
    reqs.finalize_counts();

    reqs.activate_device();
    Kokkos::parallel_for(
        "PerfTestNgpModRequests::Example1Request", Kokkos::RangePolicy<size_t>(0, num_requests),
        KOKKOS_LAMBDA(const size_t ticket) {
          FutureEntity future_sphere = req_spheres.request_element(ticket);
          FutureEntity future_node = req_nodes.request_node(ticket);
          req_conns.request(ticket, future_sphere, future_node, 0);
        });

    reqs.process_requests(bulk);

    size_t checksum = 0;
    if (num_requests > 0) {
      stk::mesh::Entity sphere0 = req_spheres.get_entity(0, stk::topology::ELEMENT_RANK);
      stk::mesh::Entity node0 = req_nodes.get_entity(0, stk::topology::NODE_RANK);
      checksum += bulk.identifier(sphere0);
      checksum += bulk.identifier(node0);
    }
    ankerl::nanobench::doNotOptimizeAway(checksum);
  }
};

struct Example2DivideBacteriaCase {
  size_t num_requests{0};
  double divide_probability{0.1};
  size_t rng_seed{20260311};
  MeshContext context;

  void setup(size_t requests, double divide_prob, size_t seed) {
    num_requests = requests;
    divide_probability = divide_prob;
    rng_seed = seed;
    context = create_context();

    BulkData& bulk = *context.bulk_data_ptr;
    bulk.modification_begin();
    for (size_t i = 0; i < num_requests; ++i) {
      stk::mesh::Entity node = bulk.declare_node(i + 1);
      stk::mesh::Entity bacteria =
          bulk.declare_entity(stk::topology::ELEMENT_RANK, i + 1, stk::mesh::PartVector{context.bacteria_part});
      bulk.declare_relation(bacteria, node, 0);
    }
    bulk.modification_end();
  }

  void run() {
    BulkData& bulk = *context.bulk_data_ptr;

    NgpModRequests reqs;
    stk::mesh::PartVector no_parts;
    auto req_bacteria = reqs.request_entities_new_ids(*context.bacteria_part);
    auto req_nodes = reqs.request_entities_new_ids(no_parts);
    auto req_conns = reqs.request_connections();

    const size_t invalid_ticket = std::numeric_limits<size_t>::max();
    Kokkos::View<size_t*> child_ticket("PerfTestNgpModRequests::child_ticket", num_requests);
    Kokkos::deep_copy(child_ticket, invalid_ticket);

    reqs.activate_device();
    Kokkos::parallel_for(
        "PerfTestNgpModRequests::Example2Claim", Kokkos::RangePolicy<size_t>(0, num_requests),
        KOKKOS_LAMBDA(const size_t i) {
          openrand::Philox rng = make_philox(rng_seed, i);
          const bool divide = rng.uniform<double>(0.0, 1.0) < divide_probability;
          if (divide) {
            const size_t ticket = req_bacteria.element_tickets().claim();
            req_nodes.node_tickets().claim();
            req_conns.tickets().claim();
            child_ticket(i) = ticket;
          }
        });
    reqs.finalize_counts();

    reqs.activate_device();
    Kokkos::parallel_for(
        "PerfTestNgpModRequests::Example2Request", Kokkos::RangePolicy<size_t>(0, num_requests),
        KOKKOS_LAMBDA(const size_t i) {
          const size_t ticket = child_ticket(i);
          if (ticket != invalid_ticket) {
            FutureEntity future_bacteria = req_bacteria.request_element(ticket);
            FutureEntity future_node = req_nodes.request_node(ticket);
            req_conns.request(ticket, future_bacteria, future_node, 0);
          }
        });

    reqs.process_requests(bulk);

    auto child_ticket_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), child_ticket);

    size_t checksum = 0;
    for (size_t i = 0; i < num_requests; ++i) {
      const size_t ticket = child_ticket_host(i);
      if (ticket != invalid_ticket) {
        stk::mesh::Entity child_bacteria = req_bacteria.get_entity(ticket, stk::topology::ELEMENT_RANK);
        stk::mesh::Entity child_node = req_nodes.get_entity(ticket, stk::topology::NODE_RANK);
        checksum += bulk.identifier(child_bacteria);
        checksum += bulk.identifier(child_node);
        break;
      }
    }
    ankerl::nanobench::doNotOptimizeAway(checksum);
  }
};

void run_perf_tests() {
  const std::vector<size_t> request_counts = {1, 50, 100, 500, 1000, 5000, 10000, 50000, 100000};
  const double divide_probability = 0.25;
  const size_t rng_seed = 20260311;

  {
    ankerl::nanobench::Bench bench;
    bench.performanceCounters(true)
        .relative(true)
        .unit("mod-cycle")
        .minEpochIterations(1)
        .epochs(10)  // not 1
        .warmup(2);

    bench.title("NgpModRequests Example1: Create N spheres");
    for (size_t n : request_counts) {
      Example1CreateSpheresCase bench_case;
      bench_case.setup(n);
      bench.complexityN(static_cast<double>(n)).run("create-spheres", [&bench_case] { bench_case.run(); });
    }

    std::cout << "\nBig-O fit for create-spheres\n";
    std::cout << bench.complexityBigO() << "\n";
  }

  {
    ankerl::nanobench::Bench bench;
    bench.performanceCounters(true).relative(true).unit("mod-cycle").minEpochIterations(1).epochs(10).warmup(2);

    bench.title("NgpModRequests Example2: Divide bacteria");
    for (size_t n : request_counts) {
      Example2DivideBacteriaCase bench_case;
      bench_case.setup(n, divide_probability, rng_seed);
      bench.complexityN(static_cast<double>(n)).run("divide-bacteria", [&bench_case] { bench_case.run(); });
    }

    std::cout << "\nBig-O fit for divide-bacteria\n";
    std::cout << bench.complexityBigO() << "\n";
  }
}

}  // namespace
}  // namespace mesh
}  // namespace mundy

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  mundy::mesh::run_perf_tests();

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
