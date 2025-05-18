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

// External libs
#include <openrand/philox.h>  // for openrand::Philox

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

// Switch to host execution space for debugging
// using DeviceExecutionSpace = Kokkos::DefaultHostExecutionSpace;
// using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

#define DOUBLE_ZERO 1e-12

template <typename value_type, int value_size>
void axpby(const value_type alpha, const Kokkos::View<value_type **, DeviceMemorySpace> &x, const value_type beta,
           Kokkos::View<value_type **, DeviceMemorySpace> &y) {
  // Perform the operation y = alpha * x + beta * y
  // Use out own instead of KokkosBlas::axpby
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "Axpby", range_policy(0, x.extent(0)), KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < value_size; ++j) {
          y(i, j) = alpha * x(i, j) + beta * y(i, j);
        }
      });
}

template <class Space>
struct VelocityReducer {
 public:
  // Required
  typedef VelocityReducer reducer;
  typedef mundy::math::Vector3d value_type;
  typedef Kokkos::View<value_type *, Space, Kokkos::MemoryUnmanaged> result_view_type;

 private:
  value_type &value;

 public:
  KOKKOS_INLINE_FUNCTION
  VelocityReducer(value_type &value_) : value(value_) {
  }

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type &dest, const value_type &src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type &val) const {
    val.set(0.0, 0.0, 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  value_type &reference() const {
    return value;
  }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const {
    return result_view_type(&value, 1);
  }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const {
    return true;
  }
};

template <typename Func>
struct VelocityKernelThreadReductionFunctor {
  KOKKOS_INLINE_FUNCTION
  VelocityKernelThreadReductionFunctor(const Func &compute_velocity_contribution, const int t)
      : compute_velocity_contribution_(compute_velocity_contribution), t_(t) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int s, mundy::math::Vector3d &v_accum) const {
    // Call the custom operation to compute the contribution
    compute_velocity_contribution_(t_, s, v_accum[0], v_accum[1], v_accum[2]);
  }

  const Func compute_velocity_contribution_;
  const int t_;
};

template <int panel_size, typename ExecutionSpace, typename Func>
struct VelocityKernelTeamFunctor {
  using TeamMemberType = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;

  KOKKOS_INLINE_FUNCTION
  VelocityKernelTeamFunctor(const TeamMemberType &team_member, const Func &compute_velocity_contribution,
                            const int panel_start, const int num_source_points,
                            Kokkos::Array<double, panel_size> &local_vx, Kokkos::Array<double, panel_size> &local_vy,
                            Kokkos::Array<double, panel_size> &local_vz)
      : team_member_(team_member),
        compute_velocity_contribution_(compute_velocity_contribution),
        panel_start_(panel_start),
        num_source_points_(num_source_points),
        local_vx_(local_vx),
        local_vy_(local_vy),
        local_vz_(local_vz) {
  }

  KOKKOS_FUNCTION
  void operator()(const int t) const {
    mundy::math::Vector3d v_sum = {0.0, 0.0, 0.0};

    // Loop over all source points
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member_, num_source_points_),
                            VelocityKernelThreadReductionFunctor<Func>(compute_velocity_contribution_, t),
                            VelocityReducer<ExecutionSpace>(v_sum));

    // Store the results in the local arrays
    local_vx_[t - panel_start_] = v_sum[0];
    local_vy_[t - panel_start_] = v_sum[1];
    local_vz_[t - panel_start_] = v_sum[2];
  }

  const TeamMemberType &team_member_;
  const Func compute_velocity_contribution_;
  const int panel_start_;
  const int num_source_points_;
  Kokkos::Array<double, panel_size> &local_vx_;
  Kokkos::Array<double, panel_size> &local_vy_;
  Kokkos::Array<double, panel_size> &local_vz_;
};

template <int panel_size, class MemorySpace, class Layout>
struct VelocityKernelTeamTeamAccumulator {
  KOKKOS_INLINE_FUNCTION
  VelocityKernelTeamTeamAccumulator(const Kokkos::View<double *, Layout, MemorySpace> &target_velocities,
                                    const Kokkos::Array<double, panel_size> &local_vx,
                                    const Kokkos::Array<double, panel_size> &local_vy,
                                    const Kokkos::Array<double, panel_size> &local_vz, const int panel_start,
                                    const int panel_end)
      : target_velocities_(target_velocities),
        local_vx_(local_vx),
        local_vy_(local_vy),
        local_vz_(local_vz),
        panel_start_(panel_start),
        panel_end_(panel_end) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()() const {
    for (int t = panel_start_; t < panel_end_; ++t) {
      Kokkos::atomic_add(&target_velocities_(3 * t + 0), local_vx_[t - panel_start_]);
      Kokkos::atomic_add(&target_velocities_(3 * t + 1), local_vy_[t - panel_start_]);
      Kokkos::atomic_add(&target_velocities_(3 * t + 2), local_vz_[t - panel_start_]);
    }
  }

  const Kokkos::View<double *, Layout, MemorySpace> target_velocities_;
  const Kokkos::Array<double, panel_size> &local_vx_;
  const Kokkos::Array<double, panel_size> &local_vy_;
  const Kokkos::Array<double, panel_size> &local_vz_;
  const int panel_start_;
  const int panel_end_;
};

template <int panel_size, class ExecutionSpace, class MemorySpace, class Layout, typename Func>
void panelize_velocity_kernel_over_target_points([[maybe_unused]] const ExecutionSpace &space, int num_target_points,
                                                 int num_source_points,
                                                 Kokkos::View<double *, Layout, MemorySpace> target_velocities,
                                                 const Func &compute_velocity_contribution) {
  int num_panels = (num_target_points + panel_size - 1) / panel_size;

  // Define the team policy with the number of panels
  using team_policy = Kokkos::TeamPolicy<ExecutionSpace>;
  Kokkos::parallel_for(
      "Panalize_Target_Points", team_policy(num_panels, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_policy::member_type &team_member) {
        const int panel_start = team_member.league_rank() * panel_size;
        const int panel_end =
            (panel_start + panel_size) > num_target_points ? num_target_points : (panel_start + panel_size);

        // Local accumulation arrays for each target point in the panel
        Kokkos::Array<double, panel_size> local_vx = {0.0};
        Kokkos::Array<double, panel_size> local_vy = {0.0};
        Kokkos::Array<double, panel_size> local_vz = {0.0};

        // Loop over each target point in the panel
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, panel_start, panel_end),
                             VelocityKernelTeamFunctor<panel_size, ExecutionSpace, Func>(
                                 team_member, compute_velocity_contribution, panel_start, num_source_points, local_vx,
                                 local_vy, local_vz));

        // After processing, update the global output using a single thread per team
        Kokkos::single(Kokkos::PerTeam(team_member),
                       VelocityKernelTeamTeamAccumulator<panel_size, MemorySpace, Layout>(
                           target_velocities, local_vx, local_vy, local_vz, panel_start, panel_end));
      });
}

struct RPYKernel {
  KOKKOS_INLINE_FUNCTION
  RPYKernel(const double viscosity, const double sphere_radius,
            const Kokkos::View<double **, DeviceMemorySpace> &target_positions,
            const Kokkos::View<double **, DeviceMemorySpace> &source_positions,
            const Kokkos::View<double **, DeviceMemorySpace> &source_forces)
      : scale_factor_(1.0 / (8.0 * pi * viscosity)),
        a2_over_three_((1.0 / 3.0) * sphere_radius * sphere_radius),
        target_positions_(target_positions),
        source_positions_(source_positions),
        source_forces_(source_forces) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t t, const size_t s, double &vx_accum, double &vy_accum, double &vz_accum) const {
    // Compute the distance vector
    const double dx = target_positions_(t, 0) - source_positions_(s, 0);
    const double dy = target_positions_(t, 1) - source_positions_(s, 1);
    const double dz = target_positions_(t, 2) - source_positions_(s, 2);

    const double fx = source_forces_(s, 0);
    const double fy = source_forces_(s, 1);
    const double fz = source_forces_(s, 2);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const double rinv = r2 < DOUBLE_ZERO ? 0.0 : 1.0 / sqrt(r2);
    const double rinv3 = rinv * rinv * rinv;
    const double rinv5 = rinv * rinv * rinv3;
    const double fdotr = fx * dx + fy * dy + fz * dz;

    const double three_fdotr_rinv5 = 3 * fdotr * rinv5;
    const double cx = fx * rinv3 - three_fdotr_rinv5 * dx;
    const double cy = fy * rinv3 - three_fdotr_rinv5 * dy;
    const double cz = fz * rinv3 - three_fdotr_rinv5 * dz;

    const double fdotr_rinv3 = fdotr * rinv3;

    // Velocity
    const double v0 = scale_factor_ * (fx * rinv + dx * fdotr_rinv3 + a2_over_three_ * cx);
    const double v1 = scale_factor_ * (fy * rinv + dy * fdotr_rinv3 + a2_over_three_ * cy);
    const double v2 = scale_factor_ * (fz * rinv + dz * fdotr_rinv3 + a2_over_three_ * cz);

    // Laplacian
    const double lap0 = 2.0 * scale_factor_ * cx;
    const double lap1 = 2.0 * scale_factor_ * cy;
    const double lap2 = 2.0 * scale_factor_ * cz;

    // Apply the result
    const double lap_coeff = 0.5 * a2_over_three_;
    vx_accum += v0 + lap_coeff * lap0;
    vy_accum += v1 + lap_coeff * lap1;
    vz_accum += v2 + lap_coeff * lap2;
  }

 private:
  const double pi = Kokkos::numbers::pi_v<double>;
  const double scale_factor_;
  const double a2_over_three_;

  const Kokkos::View<double **, DeviceMemorySpace> target_positions_;
  const Kokkos::View<double **, DeviceMemorySpace> source_positions_;
  const Kokkos::View<double **, DeviceMemorySpace> source_forces_;
};

/// \brief Apply the RPY kernel to map source forces to target velocities: u_target += M f_source
///
/// Note, this does not include self-interaction. If that is desired simply add 1/(6 pi mu) * f to u
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
void apply_rpy_kernel(const double viscosity, const Kokkos::View<double **, DeviceMemorySpace> &source_positions,
                      const Kokkos::View<double **, DeviceMemorySpace> &target_positions, const double sphere_radius,
                      const Kokkos::View<double **, DeviceMemorySpace> &source_forces,
                      Kokkos::View<double **, DeviceMemorySpace> &target_velocities) {
  const size_t num_source_points = source_positions.extent(0);
  const size_t num_target_points = target_positions.extent(0);

  // Launch the parallel kernel
  constexpr unsigned panel_size = 128;
  RPYKernel rpy_computation(viscosity, sphere_radius, target_positions, source_positions, source_forces);
  panelize_velocity_kernel_over_target_points<panel_size>(num_target_points, num_source_points, target_velocities,
                                                          rpy_computation);
}

/// \brief Apply the RPY kernel nearest neighbors to map source forces to target velocities: u_target += M f_source
void apply_rpy_kernel_neighbors(const double viscosity,
                                const Kokkos::View<double **, DeviceMemorySpace> &source_positions,
                                const Kokkos::View<double **, DeviceMemorySpace> &target_positions,
                                const Kokkos::View<int *, DeviceMemorySpace> &neighbor_indices,
                                const Kokkos::View<int *, DeviceMemorySpace> &neighbor_offsets,
                                const double sphere_radius,
                                const Kokkos::View<double **, DeviceMemorySpace> &source_forces,
                                Kokkos::View<double **, DeviceMemorySpace> &target_velocities) {
  // Loop over each target point, fetch its neighbors (sources), and apply the RPY kernel
  // The sum does not need an atomic due to looping over targets.
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  RPYKernel rpy_computation(viscosity, sphere_radius, target_positions, source_positions, source_forces);
  Kokkos::parallel_for(
      "ApplyRPYKernel", range_policy(0, target_positions.extent(0)), KOKKOS_LAMBDA(const int t) {
        const int start = neighbor_offsets(t);
        const int end = neighbor_offsets(t + 1);
        double vx_accum = 0.0;
        double vy_accum = 0.0;
        double vz_accum = 0.0;
        for (int i = start; i < end; ++i) {
          const int s = neighbor_indices(i);
          rpy_computation(t, s, vx_accum, vy_accum, vz_accum);
        }
        target_velocities(t, 0) += vx_accum;
        target_velocities(t, 1) += vy_accum;
        target_velocities(t, 2) += vz_accum;
      });
}

struct BoundingSpheres {
  KOKKOS_FUNCTION
  BoundingSpheres(Kokkos::View<double **, DeviceMemorySpace> positions, const double radius)
      : positions_(positions), radius_(radius) {
  }

  // Function to get the number of bounding spheres
  KOKKOS_FUNCTION int size() const {
    return positions_.extent(0);
  }

  // Function to return the ith bounding sphere
  // WARNING WARNING WARNING: ArborX only supports floating point types for the bounding sphere centers
  // As a result we have to cast the sphere positions to float.
  KOKKOS_FUNCTION ArborX::Sphere get_sphere(int i) const {
    auto x = positions_(i, 0);
    auto y = positions_(i, 1);
    auto z = positions_(i, 2);
    return ArborX::Sphere{{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)},
                          static_cast<float>(radius_)};
  }

  KOKKOS_FUNCTION ArborX::Point get_point(int i) const {
    auto x = positions_(i, 0);
    auto y = positions_(i, 1);
    auto z = positions_(i, 2);
    return ArborX::Point{{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)}};
  }

 private:
  Kokkos::View<double **, DeviceMemorySpace> positions_;
  const double radius_;
};  // BoundingSpheres

// For creating the bounding volume hierarchy given a Boxes object, we
// need to define the memory space, how to get the total number of objects,
// and how to access a specific box. Since there are corresponding functions in
// the Boxes class, we just resort to them.
template <>
struct ArborX::AccessTraits<BoundingSpheres, ArborX::PrimitivesTag> {
  using memory_space = DeviceMemorySpace;
  static KOKKOS_FUNCTION int size(BoundingSpheres const &spheres) {
    return spheres.size();
  }
  static KOKKOS_FUNCTION auto get(BoundingSpheres const &spheres, int i) {
    return spheres.get_point(i);
  }
};  // ArborX::AccessTraits<BoundingSpheres, ArborX::PrimitivesTag>

// For performing the queries given a Boxes object, we need to define memory
// space, how to get the total number of queries, and what the query with index
// i should look like. Since we are using self-intersection (which boxes
// intersect with the given one), the functions here very much look like the
// ones in ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PrimitivesTag>.
template <>
struct ArborX::AccessTraits<BoundingSpheres, ArborX::PredicatesTag> {
  using memory_space = DeviceMemorySpace;
  static KOKKOS_FUNCTION int size(BoundingSpheres const &spheres) {
    return spheres.size();
  }
  static KOKKOS_FUNCTION auto get(BoundingSpheres const &spheres, int i) {
    return ArborX::attach(ArborX::intersects(spheres.get_sphere(i)), (int)i);
  }
};  // ArborX::AccessTraits<BoundingSpheres, ArborX::PredicatesTag>

// Something really odd is happening here. If we don't mark the operator as an inline function, the code gives the wrong
// result, but if we do mark it, then the compiler complains about calling this function on the host.
struct ExcludeDuplicateConstraints {
  template <class Predicate, class OutputFunctor>
  KOKKOS_INLINE_FUNCTION void operator()(Predicate const &predicate, int i, OutputFunctor const &out) const {
    const int j = ArborX::getData(predicate);
    if (i < j) {
      out(i);
    }
  }
};

int generate_neighbor_pairs(const double sphere_radius,
                            const Kokkos::View<double **, DeviceMemorySpace> &sphere_positions,
                            Kokkos::View<int **, DeviceMemorySpace> &neighbor_ids,
                            Kokkos::View<int *, DeviceMemorySpace> &neighbor_indices,
                            Kokkos::View<int *, DeviceMemorySpace> &neighbor_offsets) {
  DeviceExecutionSpace execution_space;
  BoundingSpheres bounding_spheres(sphere_positions, sphere_radius);
  ArborX::BVH<DeviceMemorySpace> bvh_tree(execution_space, bounding_spheres);

  std::cout << "Building the bounding volume hierarchy" << std::endl;
  ArborX::query(bvh_tree, DeviceExecutionSpace{}, bounding_spheres, ExcludeDuplicateConstraints{}, neighbor_indices,
                neighbor_offsets);

  // Send the number of neighbors to the host. Only copy the last element of the neighbor_offsets array
  int num_neighbor_pairs = 0;
  Kokkos::deep_copy(Kokkos::View<int, Kokkos::HostSpace>(&num_neighbor_pairs),
                    Kokkos::subview(neighbor_offsets, neighbor_offsets.extent(0) - 1));

  // Store the neighbor ids. This is a pair of indices (i, j) where i is the source sphere and j is the target sphere.
  std::cout << "Storing the neighbor ids" << std::endl;
  Kokkos::resize(neighbor_ids, num_neighbor_pairs, 2);
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "StoreNeighborIds", range_policy(0, sphere_positions.extent(0)), KOKKOS_LAMBDA(const int i) {
        const int start = neighbor_offsets(i);
        const int end = neighbor_offsets(i + 1);
        for (int j = start; j < end; ++j) {
          neighbor_ids(j, 0) = i;
          neighbor_ids(j, 1) = neighbor_indices(j);
        }
      });

  return num_neighbor_pairs;
}

void compute_signed_separation_distance_and_contact_normal(
    const Kokkos::View<int **, DeviceMemorySpace> &neighbor_ids,
    const Kokkos::View<int *, DeviceMemorySpace> &neighbor_indices,
    const Kokkos::View<int *, DeviceMemorySpace> &neighbor_offsets, const double sphere_radius,
    const Kokkos::View<double **, DeviceMemorySpace> &sphere_positions,
    const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
    const Kokkos::View<double **, DeviceMemorySpace> &con_normals_ij) {
  // Each neighbor pair will generate a constraint between the two spheres
  // Loop over each neighbor id pair, fetch each sphere's position, and compute the signed separation distance
  // defined by \|x_i - x_j\| - (r_i + r_j) where x_i and x_j are the sphere positions and r_i and r_j are the sphere
  // radii

  // Loop over the neighbor pairs
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "GenerateCollisionConstraints", range_policy(0, neighbor_ids.extent(0)), KOKKOS_LAMBDA(const int i) {
        const int source_id = neighbor_ids(i, 0);
        const int target_id = neighbor_ids(i, 1);

        // Fetch the sphere positions
        const double x_i = sphere_positions(source_id, 0);
        const double y_i = sphere_positions(source_id, 1);
        const double z_i = sphere_positions(source_id, 2);
        const double x_j = sphere_positions(target_id, 0);
        const double y_j = sphere_positions(target_id, 1);
        const double z_j = sphere_positions(target_id, 2);

        // Compute the signed separation distance
        const double source_to_target_x = x_j - x_i;
        const double source_to_target_y = y_j - y_i;
        const double source_to_target_z = z_j - z_i;
        const double distance_between_centers =
            Kokkos::sqrt(source_to_target_x * source_to_target_x + source_to_target_y * source_to_target_y +
                         source_to_target_z * source_to_target_z);
        signed_sep_dist(i) = distance_between_centers - 2.0 * sphere_radius;

        // Compute the normal vector
        const double inv_distance_between_centers = 1.0 / distance_between_centers;
        con_normals_ij(i, 0) = source_to_target_x * inv_distance_between_centers;
        con_normals_ij(i, 1) = source_to_target_y * inv_distance_between_centers;
        con_normals_ij(i, 2) = source_to_target_z * inv_distance_between_centers;
      });
}

void compute_maximum_abs_projected_sep(const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
                                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot, const double dt,
                                       double &maximum_abs_projected_sep) {
  // Perform parallel reduction over all linker indices
  maximum_abs_projected_sep = -1.0;
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_reduce(
      "ComputeMaxAbsProjectedSep", range_policy(0, lagrange_multipliers.extent(0)),
      KOKKOS_LAMBDA(const int i, double &max_val) {
        // perform the projection EQ 2.2 of Dai & Fletcher 2005
        const double lag_mult = lagrange_multipliers(i);
        const double sep_old = signed_sep_dist(i);
        const double sep_dot = signed_sep_dot(i);
        const double sep_new = sep_old + dt * sep_dot;

        double abs_projected_sep;
        if (lag_mult < 1e-12) {
          abs_projected_sep = Kokkos::abs(Kokkos::min(sep_new, 0.0));
        } else {
          abs_projected_sep = Kokkos::abs(sep_new);
        }

        // update the maximum value
        if (abs_projected_sep > max_val) {
          max_val = abs_projected_sep;
        }
      },
      Kokkos::Max<double>(maximum_abs_projected_sep));
}

void compute_diff_dots(const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                       const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers_tmp,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot_tmp, const double dt,
                       double &dot_xkdiff_xkdiff, double &dot_xkdiff_gkdiff, double &dot_gkdiff_gkdiff) {
  // Local variables to store dot products
  dot_xkdiff_xkdiff = 0.0;
  dot_xkdiff_gkdiff = 0.0;
  dot_gkdiff_gkdiff = 0.0;

  // Perform parallel reduction to compute the dot products
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_reduce(
      "ComputeDiffDots", range_policy(0, lagrange_multipliers.extent(0)),
      KOKKOS_LAMBDA(const int i, double &xkdiff_xkdiff, double &xkdiff_gkdiff, double &gkdiff_gkdiff) {
        const double lag_mult = lagrange_multipliers(i);
        const double lag_mult_tmp = lagrange_multipliers_tmp(i);
        const double sep_dot = signed_sep_dot(i);
        const double sep_dot_tmp = signed_sep_dot_tmp(i);

        // xkdiff = xk - xkm1
        const double xkdiff = lag_mult - lag_mult_tmp;

        // gkdiff = gk - gkm1
        const double gkdiff = dt * (sep_dot - sep_dot_tmp);

        // Compute the dot products
        xkdiff_xkdiff += xkdiff * xkdiff;
        xkdiff_gkdiff += xkdiff * gkdiff;
        gkdiff_gkdiff += gkdiff * gkdiff;
      },
      Kokkos::Sum<double>(dot_xkdiff_xkdiff), Kokkos::Sum<double>(dot_xkdiff_gkdiff),
      Kokkos::Sum<double>(dot_gkdiff_gkdiff));
}

void sum_collision_force(Kokkos::View<int **, DeviceMemorySpace> &neighbor_ids,
                         Kokkos::View<int *, DeviceMemorySpace> &neighbor_indices,
                         Kokkos::View<int *, DeviceMemorySpace> &neighbor_offsets,
                         Kokkos::View<double **, DeviceMemorySpace> &sphere_force,
                         Kokkos::View<double **, DeviceMemorySpace> &con_normal_ij,
                         Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers) {
  // Zero out the force first
  Kokkos::deep_copy(sphere_force, 0.0);

  // Loop over the neighbor pairs
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "SumCollisionForce", range_policy(0, neighbor_ids.extent(0)), KOKKOS_LAMBDA(const int i) {
        const int source_id = neighbor_ids(i, 0);
        const int target_id = neighbor_ids(i, 1);

        // Fetch the lagrange multiplier
        const double lag_mult = lagrange_multipliers(i);

        // Fetch the normal vector (goes from source to target)
        const double normal_x = con_normal_ij(i, 0);
        const double normal_y = con_normal_ij(i, 1);
        const double normal_z = con_normal_ij(i, 2);

        // Compute the force
        Kokkos::atomic_add(&sphere_force(source_id, 0), -lag_mult * normal_x);
        Kokkos::atomic_add(&sphere_force(source_id, 1), -lag_mult * normal_y);
        Kokkos::atomic_add(&sphere_force(source_id, 2), -lag_mult * normal_z);
        Kokkos::atomic_add(&sphere_force(target_id, 0), lag_mult * normal_x);
        Kokkos::atomic_add(&sphere_force(target_id, 1), lag_mult * normal_y);
        Kokkos::atomic_add(&sphere_force(target_id, 2), lag_mult * normal_z);
      });
}

void compute_the_mobility_problem(const double sphere_radius, const double viscosity,
                                  const Kokkos::View<double **, DeviceMemorySpace> &sphere_positions,
                                  const Kokkos::View<double **, DeviceMemorySpace> &sphere_force,
                                  const Kokkos::View<int *, DeviceMemorySpace> &neighbor_indices,
                                  const Kokkos::View<int *, DeviceMemorySpace> &neighbor_offsets,
                                  Kokkos::View<double **, DeviceMemorySpace> &sphere_velocity, bool enable_hydro = true,
                                  const bool only_use_neighbors = false) {
  // Self-interaction term
  const double pi = Kokkos::numbers::pi_v<double>;
  const double inv_drag_coeff = 1.0 / (6.0 * pi * sphere_radius * viscosity);
  axpby<double, 3>(inv_drag_coeff, sphere_force, 0.0, sphere_velocity);

  if (enable_hydro) {
    // Compute the RPY kernel to map source forces to target velocities: u_target += M f_source
    if (only_use_neighbors) {
      apply_rpy_kernel_neighbors(viscosity, sphere_positions, sphere_positions, neighbor_indices, neighbor_offsets,
                                 sphere_radius, sphere_force, sphere_velocity);
    } else {
      apply_rpy_kernel(viscosity, sphere_positions, sphere_positions, sphere_radius, sphere_force, sphere_velocity);
    }
  }
}

void compute_rate_of_change_of_sep(Kokkos::View<int **, DeviceMemorySpace> &neighbor_ids,
                                   Kokkos::View<int *, DeviceMemorySpace> &neighbor_indices,
                                   Kokkos::View<int *, DeviceMemorySpace> &neighbor_offsets,
                                   const Kokkos::View<double **, DeviceMemorySpace> &sphere_velocity,
                                   const Kokkos::View<double **, DeviceMemorySpace> &con_normal_ij,
                                   Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot) {
  // Compute the (linearized) rate of change in sep
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "ComputeRateOfChangeOfSep", range_policy(0, neighbor_ids.extent(0)), KOKKOS_LAMBDA(const int i) {
        const int source_id = neighbor_ids(i, 0);
        const int target_id = neighbor_ids(i, 1);

        // Fetch the normal vector (goes from source to target)
        const double normal_x = con_normal_ij(i, 0);
        const double normal_y = con_normal_ij(i, 1);
        const double normal_z = con_normal_ij(i, 2);

        // Fetch the velocity of the source and target spheres
        const double source_velocity_x = sphere_velocity(source_id, 0);
        const double source_velocity_y = sphere_velocity(source_id, 1);
        const double source_velocity_z = sphere_velocity(source_id, 2);
        const double target_velocity_x = sphere_velocity(target_id, 0);
        const double target_velocity_y = sphere_velocity(target_id, 1);
        const double target_velocity_z = sphere_velocity(target_id, 2);

        // Compute the rate of change in separation
        signed_sep_dot(i) = -normal_x * (source_velocity_x - target_velocity_x) -
                            normal_y * (source_velocity_y - target_velocity_y) -
                            normal_z * (source_velocity_z - target_velocity_z);
      });
}

void update_con_gammas(const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                       const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers_tmp,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot, const double dt,
                       const double alpha) {
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "UpdateConGammas", range_policy(0, lagrange_multipliers.extent(0)), KOKKOS_LAMBDA(const int i) {
        // Fetch fields for the current linker
        const double sep_old = signed_sep_dist(i);
        const double sep_dot = signed_sep_dot(i);
        const double sep_new = sep_old + dt * sep_dot;

        // Update lagrange multipliers with bound projection
        lagrange_multipliers(i) = Kokkos::max(lagrange_multipliers_tmp(i) - alpha * sep_new, 0.0);
      });
}

struct CollisionResult {
  double max_abs_projected_sep;
  int ite_count;
  double max_displacement;
};

enum HydroType { DRY, HYDRO_NEAREST, HYDRO_DISTANT, HYDRO_ALL };

CollisionResult resolve_collisions(const double viscosity, const double dt, const double max_allowable_overlap,
                                   const int max_col_iterations, Kokkos::View<int **, DeviceMemorySpace> &neighbor_ids,
                                   Kokkos::View<int *, DeviceMemorySpace> &neighbor_indices,
                                   Kokkos::View<int *, DeviceMemorySpace> &neighbor_offsets, const double sphere_radius,
                                   Kokkos::View<double **, DeviceMemorySpace> &sphere_positions,
                                   Kokkos::View<double **, DeviceMemorySpace> &sphere_velocity,
                                   Kokkos::View<double **, DeviceMemorySpace> &sphere_force,
                                   Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
                                   Kokkos::View<double **, DeviceMemorySpace> &con_normal_ij,
                                   Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                                   const bool enable_hydro = false) {
  // Matrix-free BBPGD
  int ite_count = 0;
  int num_collisions = neighbor_ids.extent(0);
  Kokkos::View<double *, DeviceMemorySpace> lagrange_multipliers_tmp("lagrange_multipliers_tmp", num_collisions);
  Kokkos::View<double *, DeviceMemorySpace> signed_sep_dot("signed_sep_dot", num_collisions);
  Kokkos::View<double *, DeviceMemorySpace> signed_sep_dot_tmp("signed_sep_dot_tmp", num_collisions);

  // Use the given lagrange_multipliers as the initial guess
  Kokkos::deep_copy(lagrange_multipliers_tmp, lagrange_multipliers);
  Kokkos::deep_copy(signed_sep_dot, 0.0);
  Kokkos::deep_copy(signed_sep_dot_tmp, 0.0);

  // Compute gkm1 = D^T M D xkm1

  // Compute F = D xkm1
  sum_collision_force(neighbor_ids, neighbor_indices, neighbor_offsets, sphere_force, con_normal_ij,
                      lagrange_multipliers_tmp);

  // Compute U = M F
  HydroType hydro_type = HydroType::DRY;
  if (enable_hydro) {
    hydro_type = HydroType::HYDRO_NEAREST;
  }

  if (hydro_type == HydroType::DRY) {
    compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, neighbor_indices,
                                 neighbor_offsets, sphere_velocity, false, true);
  } else if (hydro_type == HydroType::HYDRO_NEAREST) {
    compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, neighbor_indices,
                                 neighbor_offsets, sphere_velocity, true, true);
  } else if (hydro_type == HydroType::HYDRO_DISTANT) {
    compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, neighbor_indices,
                                 neighbor_offsets, sphere_velocity, true, false);
  } else if (hydro_type == HydroType::HYDRO_ALL) {
    compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, neighbor_indices,
                                 neighbor_offsets, sphere_velocity, true, false);
  } else {
    throw std::runtime_error("Unknown hydro type");
  }

  // Compute gkm1 = dt D^T U
  compute_rate_of_change_of_sep(neighbor_ids, neighbor_indices, neighbor_offsets, sphere_velocity, con_normal_ij,
                                signed_sep_dot_tmp);

  ///////////////////////
  // Check convergence //
  ///////////////////////
  // res = max(abs(projectPhi(gkm1)));
  double maximum_abs_projected_sep = -1.0;
  compute_maximum_abs_projected_sep(lagrange_multipliers_tmp, signed_sep_dist, signed_sep_dot_tmp, dt,
                                    maximum_abs_projected_sep);

  Kokkos::View<int *, DeviceMemorySpace> far_neighbor_indices("far_neighbor_indices", 0);
  Kokkos::View<int *, DeviceMemorySpace> far_neighbor_offsets("far_neighbor_offsets", 0);

  ///////////////////////
  // Loop if necessary //
  ///////////////////////
  if (maximum_abs_projected_sep < max_allowable_overlap) {
    // The initial guess was correct, nothing more is necessary
  } else {
    // Initial guess insufficient, iterate

    // First step, Dai&Fletcher2005 Section 5.
    double alpha = 1.0 / maximum_abs_projected_sep;
    while (ite_count < max_col_iterations) {
      std::cout << "Iteration: " << ite_count << " max abs projected sep: " << maximum_abs_projected_sep << std::endl;
      ++ite_count;

      // Compute xk = xkm1 - alpha * gkm1 and perform the bound projection xk = boundProjection(xk)
      update_con_gammas(lagrange_multipliers, lagrange_multipliers_tmp, signed_sep_dist, signed_sep_dot, dt, alpha);

      // Compute new grad with xk: gk = dt D^T M D xk
      //   Compute F = D xk
      sum_collision_force(neighbor_ids, neighbor_indices, neighbor_offsets, sphere_force, con_normal_ij,
                          lagrange_multipliers);

      // Compute U = M F
      if (hydro_type == HydroType::DRY) {
        compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, neighbor_indices,
                                     neighbor_offsets, sphere_velocity, false, false);
      } else if (hydro_type == HydroType::HYDRO_NEAREST) {
        compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, neighbor_indices,
                                     neighbor_offsets, sphere_velocity, true, true);
      } else if (hydro_type == HydroType::HYDRO_DISTANT) {
        compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, far_neighbor_indices,
                                     far_neighbor_offsets, sphere_velocity, true, true);
      } else if (hydro_type == HydroType::HYDRO_ALL) {
        compute_the_mobility_problem(viscosity, sphere_radius, sphere_positions, sphere_force, neighbor_indices,
                                     neighbor_offsets, sphere_velocity, true, false);
      } else {
        throw std::runtime_error("Unknown hydro type");
      }

      //   Compute gk = dt D^T U
      compute_rate_of_change_of_sep(neighbor_ids, neighbor_indices, neighbor_offsets, sphere_velocity, con_normal_ij,
                                    signed_sep_dot);

      // check convergence via res = max(abs(projectPhi(gk)));
      compute_maximum_abs_projected_sep(lagrange_multipliers, signed_sep_dist, signed_sep_dot, dt,
                                        maximum_abs_projected_sep);

      if (maximum_abs_projected_sep < max_allowable_overlap) {
        // con_gammas worked. Check if we should exit the loop or advance to the next level.
        if (enable_hydro && hydro_type == HydroType::DRY) {
          std::cout << "Level 0: Dry | Convergence reached: " << maximum_abs_projected_sep << " < "
                    << max_allowable_overlap << std::endl;
          hydro_type = HydroType::HYDRO_NEAREST;
        } else if (enable_hydro && hydro_type == HydroType::HYDRO_NEAREST) {
          std::cout << "Level 1: Hydro with nearest neighbors | Convergence reached: " << maximum_abs_projected_sep
                    << " < " << max_allowable_overlap << std::endl;
          hydro_type = HydroType::HYDRO_DISTANT;

          // Compute the far neighbors
          DeviceExecutionSpace execution_space;
          BoundingSpheres far_bounding_spheres(sphere_positions, 4 * sphere_radius);
          ArborX::BVH<DeviceMemorySpace> bvh_tree(execution_space, far_bounding_spheres);
          ArborX::query(bvh_tree, execution_space, far_bounding_spheres, ExcludeDuplicateConstraints{},
                        far_neighbor_indices, far_neighbor_offsets);

        } else if (enable_hydro && hydro_type == HydroType::HYDRO_DISTANT) {
          std::cout << "Level 2: Hydro with more neighbors | Convergence reached: " << maximum_abs_projected_sep
                    << " < " << max_allowable_overlap << std::endl;
          hydro_type = HydroType::HYDRO_ALL;
        } else {
          std::cout << "Final Level: Convergence reached: " << maximum_abs_projected_sep << " < "
                    << max_allowable_overlap << std::endl;
          break;
        }
      }

      ///////////////////////////////////////////////////////////////////////////
      // Compute dot(xkdiff, xkdiff), dot(xkdiff, gkdiff), dot(gkdiff, gkdiff) //
      // where xkdiff = xk - xkm1 and gkdiff = gk - gkm1                       //
      ///////////////////////////////////////////////////////////////////////////
      double global_dot_xkdiff_xkdiff = 0.0;
      double global_dot_xkdiff_gkdiff = 0.0;
      double global_dot_gkdiff_gkdiff = 0.0;
      compute_diff_dots(lagrange_multipliers, lagrange_multipliers_tmp, signed_sep_dot, signed_sep_dot_tmp, dt,
                        global_dot_xkdiff_xkdiff, global_dot_xkdiff_gkdiff, global_dot_gkdiff_gkdiff);

      ////////////////////////////////////////////
      // Compute the Barzilai-Borwein step size //
      ////////////////////////////////////////////
      // Alternating bb1 and bb2 methods
      double a;
      double b;
      if (ite_count % 2 == 0) {
        // Barzilai-Borwein step size Choice 1
        a = global_dot_xkdiff_xkdiff;
        b = global_dot_xkdiff_gkdiff;
      } else {
        // Barzilai-Borwein step size Choice 2
        a = global_dot_xkdiff_gkdiff;
        b = global_dot_gkdiff_gkdiff;
      }

      // Prevent div 0 errors.
      if (Kokkos::abs(b) < 1e-12) {
        b += 1e-12;
      }

      alpha = a / b;

      /////////////////////////////////
      // Set xkm1 = xk and gkm1 = gk //
      /////////////////////////////////
      Kokkos::deep_copy(lagrange_multipliers_tmp, lagrange_multipliers);
      Kokkos::deep_copy(signed_sep_dot_tmp, signed_sep_dot);
    }
  }

  // Compute the maximum velocity
  double max_velocity = 0.0;
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_reduce(
      "ComputeMaxVelocity", range_policy(0, sphere_velocity.extent(0)),
      KOKKOS_LAMBDA(const int i, double &max_val) {
        const double vx = sphere_velocity(i, 0);
        const double vy = sphere_velocity(i, 1);
        const double vz = sphere_velocity(i, 2);
        const double velocity = Kokkos::sqrt(vx * vx + vy * vy + vz * vz);
        if (velocity > max_val) {
          max_val = velocity;
        }
      },
      Kokkos::Max<double>(max_velocity));

  CollisionResult result = {maximum_abs_projected_sep, ite_count, max_velocity * dt};
  return result;
}

void check_overlap(const Kokkos::View<double **, DeviceMemorySpace> &sphere_positions, const double sphere_radius,
                   const double max_allowable_overlap) {
  // Do the check on host for easier printing
  auto sphere_positions_host = Kokkos::create_mirror_view(sphere_positions);
  Kokkos::deep_copy(sphere_positions_host, sphere_positions);

  // Loop over all pairs of spheres
  const int num_spheres = sphere_positions.extent(0);
  bool no_overlap = true;
  for (int t = 0; t < num_spheres; ++t) {
    for (int s = 0; s < num_spheres; ++s) {
      if (s != t) {
        const double x_i = sphere_positions_host(t, 0);
        const double y_i = sphere_positions_host(t, 1);
        const double z_i = sphere_positions_host(t, 2);
        const double x_j = sphere_positions_host(s, 0);
        const double y_j = sphere_positions_host(s, 1);
        const double z_j = sphere_positions_host(s, 2);

        // Compute the distance between the centers of the spheres
        const double source_to_target_x = x_i - x_j;
        const double source_to_target_y = y_i - y_j;
        const double source_to_target_z = z_i - z_j;
        const double distance_between_centers =
            Kokkos::sqrt(source_to_target_x * source_to_target_x + source_to_target_y * source_to_target_y +
                         source_to_target_z * source_to_target_z);

        // Compute the overlap
        const double ssd = distance_between_centers - 2.0 * sphere_radius;
        if (ssd < -max_allowable_overlap) {
          // The spheres are overlapping too much
          no_overlap = false;
          // std::cout << "Overlap detected between spheres " << t << " and " << s << std::endl;
          // std::cout << "Distance between centers: " << distance_between_centers << std::endl;
          // std::cout << "Overlap: " << ssd << std::endl;
          // std::cout << "Sphere positions: (" << x_i << ", " << y_i << ", " << z_i << ") and (" << x_j << ", " << y_j
          //           << ", " << z_j << ")" << std::endl;
        }
      }
    }
  }

  if (no_overlap) {
    std::cout << "No overlap detected!" << std::endl;
  } else {
    std::cout << "Overlap detected!" << std::endl;
  }
}

template <typename value_type>
void randomize_view(Kokkos::View<value_type *, DeviceMemorySpace> &view, const value_type min_val,
                    const value_type max_val, const int seed = 1234) {
  using rage_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "RandomizeView", rage_policy(0, view.extent(0)), KOKKOS_LAMBDA(const int i) {
        openrand::Philox rng(seed, i);
        view(i) = rng.rand<value_type>() * (max_val - min_val) + min_val;
      });
}

template <typename value_type, int value_size>
void randomize_view(Kokkos::View<value_type **, DeviceMemorySpace> &view,
                    const Kokkos::Array<value_type, value_size, DeviceMemorySpace> &min_vals,
                    const Kokkos::Array<value_type, value_size, DeviceMemorySpace> &max_vals, const int seed = 1234) {
  using rage_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "RandomizeView", rage_policy(0, view.extent(0)), KOKKOS_LAMBDA(const int i) {
        openrand::Philox rng(seed, i);
        for (int j = 0; j < value_size; ++j) {
          view(i, j) = rng.rand<value_type>() * (max_vals[j] - min_vals[j]) + min_vals[j];
        }
      });
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <box_size> <num_spheres>" << std::endl;
    return 1;
  }

  double box_size = std::stod(argv[1]);
  int num_spheres = std::stoi(argv[2]);
  Kokkos::initialize(argc, argv);
  {
    // Simulation params
    const double viscosity = 0.001;
    const double dt = 5e-3;

    const double sphere_radius = 1.0;
    const double search_buffer = 3 * sphere_radius;
    const double max_allowable_overlap = 1e-5;
    const int max_col_iterations = 10000;
    constexpr unsigned int spatial_dimension = 3;
    const Kokkos::Array<double, spatial_dimension, DeviceMemorySpace> domain_low = {0.0, 0.0, 0.0};
    const Kokkos::Array<double, spatial_dimension, DeviceMemorySpace> domain_high = {box_size, box_size, box_size};

    // Initialized the spheres
    const double volume_fraction = (4.0 / 3.0 * M_PI * sphere_radius * sphere_radius * sphere_radius * num_spheres) /
                                   (box_size * box_size * box_size);

    std::cout << "Initializing " << num_spheres << " spheres at a volume fraction of " << volume_fraction << std::endl;
    Kokkos::View<double **, DeviceMemorySpace> sphere_positions("sphere_positions", num_spheres, spatial_dimension);
    Kokkos::View<double **, DeviceMemorySpace> sphere_velocity("sphere_velocity", num_spheres, spatial_dimension);
    Kokkos::View<double **, DeviceMemorySpace> sphere_force("sphere_force", num_spheres, spatial_dimension);
    randomize_view<double, spatial_dimension>(sphere_positions, domain_low, domain_high);
    Kokkos::deep_copy(sphere_velocity, 0.0);
    Kokkos::deep_copy(sphere_force, 0.0);

    // Perform the neighbor detection
    std::cout << "Generating neighbor pairs" << std::endl;
    Kokkos::View<int **, DeviceMemorySpace> neighbor_ids("neighbor_ids", 0, 2);
    Kokkos::View<int *, DeviceMemorySpace> neighbor_indices("neighbor_indices", 0);
    Kokkos::View<int *, DeviceMemorySpace> neighbor_offsets("neighbor_offsets", 0);
    const int num_neighbor_pairs = generate_neighbor_pairs(sphere_radius + search_buffer, sphere_positions,
                                                           neighbor_ids, neighbor_indices, neighbor_offsets);
    std::cout << "Number of neighbor pairs: " << num_neighbor_pairs << std::endl;

    // Initialize the constraints
    std::cout << "Computing signed separation distance and contact normal" << std::endl;
    Kokkos::View<double *, DeviceMemorySpace> signed_sep_dist("signed_sep_dist", num_neighbor_pairs);
    Kokkos::View<double **, DeviceMemorySpace> con_normal_ij("con_normal_ij", num_neighbor_pairs, 3);
    compute_signed_separation_distance_and_contact_normal(neighbor_ids, neighbor_indices, neighbor_offsets,
                                                          sphere_radius, sphere_positions, signed_sep_dist,
                                                          con_normal_ij);

    // Resolve initial collisions
    std::cout << "Resolving initial collisions" << std::endl;
    const double enable_hydro = false;
    Kokkos::Timer timer;
    Kokkos::View<double *, DeviceMemorySpace> lagrange_multipliers("lagrange_multipliers", num_neighbor_pairs);
    Kokkos::deep_copy(lagrange_multipliers, 0.0);  // initial guess
    CollisionResult result =
        resolve_collisions(viscosity, dt, max_allowable_overlap, max_col_iterations, neighbor_ids, neighbor_indices,
                           neighbor_offsets, sphere_radius, sphere_positions, sphere_velocity, sphere_force,
                           signed_sep_dist, con_normal_ij, lagrange_multipliers, enable_hydro);

    // Take an Euler step
    axpby<double, 3>(dt, sphere_velocity, 1.0, sphere_positions);
    const double elapsed_time = timer.seconds();
    std::cout << "Time to resolve collisions: " << elapsed_time << " seconds" << std::endl;

    std::cout << "Result: " << std::endl;
    std::cout << "  Max abs projected sep: " << result.max_abs_projected_sep << std::endl;
    std::cout << "  Number of iterations: " << result.ite_count << std::endl;
    std::cout << "  Max displacement: " << result.max_displacement << std::endl;

    if (result.max_displacement > 2 * sphere_radius) {
      std::cout << "***WARNING*** The maximum displacement is larger than the search buffer. Collisions may be missed. "
                   "***WARNING***"
                << std::endl;
    }

    // N^2 validation that the max amount of overlap is less than the allowed overlap
    std::cout << "Checking for overlap" << std::endl;
    // check_overlap(sphere_positions, sphere_radius, max_allowable_overlap);
  }

  Kokkos::finalize();

  return 0;
}