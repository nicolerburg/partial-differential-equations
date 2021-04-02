# partial-differential-equations
 
This project contains two main files: ivp.py and bvp.py

ivp.py contains the code for the Cahn-Hilliard Equation Solver, which simulates the separation of oil and water over time.
    In this file you may run a live visualization or free energy data collection simply by running the program and selecting you options in the console.

bvp.py contains the code for the potential field relaxation calculations.
    This file allows you to find the steady state solution for either a electric point charge in 3-D or a current carrying wire through the z-axis.
    You can choose to find the steady state with either the Jacobi, Gauss-Seidel, or Successive Over-Relaxation (with specified omega) algorithms.
    These choices will output the potential field contour, the vector field plot, and the magnitude of each vs distance to the center of the grid.
    Additionally, one can run a sweep through all omega values in the SOR algorithm to find the fastest convergence.