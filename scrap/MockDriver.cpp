/* This file represents my vision for what our eventual driver will look like, as applies to Brato's "worm" problem.

// Preprocess
- Parse user parameter

// Setup
- Declare the fixed and mutable params for the desired MetaMethods
- Construct the mesh and the method instances
- Declare and initialize the sperm's nodes, edges, and elements (centerline twist springs)
(Using BulkData's declare_node, declare_element, declare_element_edge functions)

// Timeloop
- Run the timeloop for t in range(0, T):
    // IO.
    - If desired, write out the data for time t
        (Using stk::io::StkMeshIoBroker)

    // Prepare the current configuration.
    - Rotate the field states
        (Using BulkData's update_field_data_states function)

    - Zero the node forces and velocities for time t + dt
        (Using mundy's fill_field_with_value function)

    // Motion from t -> t + dt:
    - Apply velocity/acceleration constraints like no motion for particle 1
        (By directly looping over all nodes and setting the velocity/acceleration to zero)

    - Evaluate x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2
        (By looping over all nodes and updating the coordinates)

    // Evaluate forces f(x(t + dt))
    {
        // Hertzian contact
        {
            // Neighbor detection rod-rod
            - Check if the rod-rod neighbor list needs updated or not
                - Compute the AABBs for the rods
                (Using mundy's ComputeAABB function)

                - Delete rod-rod neighbor linkers that are too far apart
                (Using the DestroyDistantNeighbors technique of mundy's DestroyNeighborLinkers function)

                - Generate neighbor linkers between nearby rods
                (Using the GenerateNeighborLinkers function of mundy's GenerateNeighborLinkers function)

            // Hertzian contact force evaluation
            - Compute the signed separation distance and contact normal between neighboring rods
            (Using mundy's ComputeSignedSeparationDistanceAndContactNormal function)

            - Evaluate the Hertzian contact potential between neighboring rods
            (Using mundy's EvaluateLinkerPotentials function)

            - Sum the linker potential force to get the induced node force on each rod
            (Using mundy's LinkerPotentialForceReduction function)
        }

        // Centerline twist rod model
        - Compute the edge information (length, tangent, and binormal)
        (By looping over all edges and computing the edge length, tangent, binormal)

        - Compute the node curvature and rotation gradient
        (By looping over all centerline twist spring elements and computing the curvature & rotation gradient at
        their center node using the edge orientations)

        - Compute the internal force and twist torque
        (By looping over all centerline twist spring elements, using the curvature to compute the induced
        benching/twisting torque and stretch force, and using them to compute the force and twist-torque on the
        nodes. .)
    }

    // Compute velocity and acceleration
    - Evaluate a(t + dt) = M^{-1} f(x(t + dt))
        (By looping over all nodes and computing the node acceleration and twist acceleration from the force and
        twist torque)

    - Evaluate v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2
        (By looping over all nodes and updating the node velocity and twist rate using the corresponding
        accelerations)






We should keep agents but add helper functions to expose bits of the lambda function to the user.
Just like mundy::distance offers signed separation distnace utility functions, so to
should we offer helper functions for computing the AABB's, bounding spheres, etc. The creation of
MetaMethods just needs streamlined, so that they are less intimidating to work with.
*/

// C++ core
#include <iostream>  // for std::cout, std::endl

// Mundy
#include <mundy_driver/Configurator.hpp>  // for mundy::driver::Configurator

int main(int argc, char** argv)
{
    // Parse the command line options to find the input file name
    Teuchos::CommandLineProcessor cmdp(false, true);
    std::string input_file_name;
    cmdp.setOption("input_file_name", &input_file_name, "The name of the input configuration yaml file.");
    bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    MUNDY_THROW_REQUIRE(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");

    // Create a configurator from the given config file
    auto configurator = mundy::driver::Configurator(input_file_name);
    std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = configurator.configure();

    // After calling configure, the configurator contains the configured methods
    // and all of the specified fields.
    // 

    // Fetch the configured methods
    // (META) This is a critical point, as it assumes that the input file defines the desired methods






    return 0;
}


int main(int argc, char** argv)
{
    // Define the global variables to be populated by the user
    double timestep_size;
    double num_steps;

    // 


    // Create a configurator
    auto configurator = mundy::driver::Configurator()
        .add_global_variable("timestep_size", &timestep_size)
        .add_global_variable("num_steps", &num_steps)





    return 0;
}


