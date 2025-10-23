import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Take in these parameters from the user
    num_segments = int(input("Enter the number of segments: "))
    ell = float(input("Enter the length of each segment: "))
    amplitude = float(input("Enter the amplitude of the sine curve: "))
    angular_frequency = float(input("Enter the angular frequency of the sine curve: "))
    phase_shift = float(input("Enter the phase shift of the sine curve: "))
    tolerance = float(input("Enter the tolerance for the Brent's method: "))
    enable_plot = input("Do you want to plot the sine curve with the line segments? (y/n): ")

    # Function to find the next x value using Brent's method
    # The function we are discretizing with equal segments is 
    #   y = amplitude sin(angular_frequency x + phase_shift)
    def find_next_x_brent(x_prev):
        return brentq(lambda x: 
        np.sqrt((x - x_prev)**2 + 
        (amplitude * np.sin(angular_frequency * x + phase_shift) 
            - amplitude * np.sin(angular_frequency * x_prev + phase_shift))**2) - ell, 
        x_prev, 
        x_prev + ell, 
        xtol=tolerance)

    # Initial conditions
    x_prev = 0.0
    x_values_brent = [x_prev]

    # Solve for each segment using Brent's method
    for i in range(num_segments):
        x_next = find_next_x_brent(x_prev)
        x_values_brent.append(x_next)
        x_prev = x_next

    # Extract the sine curve points for new ell using Brent's method
    x_points_brent = np.array(x_values_brent)
    y_points_brent = np.sin(x_points_brent)

    # Write x and y to to a vs file. We'll follow the convention of 
    # using [x0, y0, x1, y1, ...] for the x and y values
    np.savetxt('sperm_pos_N_{num_segments}_ell_{ell}_tol_{tolerance}.csv'.format(
        num_segments=num_segments, ell=ell, tolerance=tolerance), 
        np.array([x_points_brent, y_points_brent]).T.flatten(), delimiter=',')
    
    if enable_plot == 'y' or enable_plot == 'Y':
        # Plot the sine curve with new ell using Brent's method
        x_full = np.linspace(0, x_points_brent[-1], 10 * num_segments)
        y_full = np.sin(x_full)

        plt.figure(figsize=(12, 6))
        plt.plot(x_full, y_full, label='Sine Curve', color='lightgray')
        plt.plot(x_points_brent, y_points_brent, 'o-', label='Line Segments (ell = {})'.format(ell), color='red')

        # Marking the segment endpoints for new ell
        for i in range(len(x_points_brent) - 1):
            plt.plot([x_points_brent[i], x_points_brent[i+1]], [y_points_brent[i], y_points_brent[i+1]], 'b-')

        plt.xlabel('x')
        plt.ylabel('sin(x)')
        plt.title('Discretization of Sine Curve into Equal-Length Line Segments (ell = {})'.format(ell))
        plt.legend()
        plt.grid(True)
        plt.show()