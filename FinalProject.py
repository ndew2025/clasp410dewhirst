#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # turn on interactive mode
plt.style.use('seaborn-v0_8-dark')  # Use a style sheet

'''This file contains Nick's final project code to solve and plot results from the 2D diffusion equation
See docstrings for details on each function.'''

def two_dim_diff(tmax = 10, zmax = 8, xmax = 8, dt = 0.1, dz = 1, dx = 1, K = 1, debug = False):
    '''This function will solve the 2D diffusion equation as it applies to a smoke plume
    initialized in the center of an 8x8 grid.

    PARAMETERS
    ==========

    tmax: float
        The length of time to model diffusion over (seconds)
    
    xmax: float
        The length of the domain in the x direction (meters)
    
    zmax: float
        The altitude extent of the domain (meters)

    dt: float
        The step in time, in seconds, to solve the diffusion equation.

    dx: float
        The step in x, in meters, to solve the diffusion equation.

    dz: float
        The step in z, in metersm to solve the diffusion equation.

    K: float
        The turbulent diffusion coefficient in m^2/s.
    
    debug: boolean
        Helps debug the function by print array/dimension shapes and detecting NaNs.
        Default is False.
        
    RETURNS
    =======

    xgrid: 1D array
        the array of values in the x direction to model diffusion over

    tgrid: 1D array
        the array of times to model diffusion over

    C: 2D array
        The solution to the diffusion equation. Contains a distribution of C02 concentrations
        over time.
    
    K: float
        The turbulent diffusion coefficient in m^2/s.
    '''

    # CFL Criterion for 2D diffusion equation
    CFL_dt = dx**2 * dz**2 /(2*K*(dx**2 + dz**2))

    # Stability check for the model
    if dt > CFL_dt:  # if the timestep is greater than that, throw out an error
        raise ValueError(f"dt is too large! Must be <= {CFL_dt} with dx = {dx}, dz = {dz} and K = {K}")

    # Length of time to simulate diffusion
    T = int(round(tmax / dt + 1))

    # Dimensions of C arrays
    Z = int(round(zmax / dz + 1))
    X = int(round(xmax / dx + 1))
    
    # Create grids in z and x with correct dimensions
    zgrid = np.linspace(0, zmax, Z)
    xgrid = np.linspace(0, xmax, X)

    # Initialize C arrays as zeros with X, Y, T
    C_current = np.zeros((Z, X)) # For the initial/current state of the array
    C_next = np.zeros((Z, X)) # For the future state of the domain

    if debug: # to debug, print the shapes of the arrays and x,z dimensions
        print(C_current.shape)
        print(C_next)
        print(zgrid.shape)
        print(xgrid.shape)

    C_current[Z//2, X//2] = 10
        
    for i in range(T-1): # Loop through all time steps except the last
        # Centered difference in x and z, forward difference in time
        Cxx = (C_current[1:-1, 2:] - 2*C_current[1:-1, 1:-1] + C_current[1:-1, :-2])/dz**2
        Czz = (C_current[2:, 1:-1] - 2*C_current[1:-1, 1:-1] + C_current[:-2, 1:-1])/dx**2 
        
        C_next[1:-1, 1:-1] = dt*K*(Cxx + Czz) + C_current[1:-1, 1:-1]

        # Check for NaNs in Cxx, Czz, or C for debugging
        if debug:
            if np.any(np.isnan(Cxx)) or np.any(np.isnan(Czz)) or np.any(np.isnan(C_next[:, :])):
                print(f"Warning: NaN detected at timestep {i}.")
                print(C_next[:, :])
                break

        # No flux boundary condition at the ground, keeps the aerosols inside the domain
        C_next[0, :] = C_current[1, :]

        C_current = C_next.copy() # Update C_current to be used in the next iteration
    
    C = C_current # just a name change for the returned array

    print(f"C Array: {C}") # print the array for validation
        
    return zgrid, xgrid, C, K  # return the range of z, x, the solution, and K

##############################

def aerosol_diff(tmax = 100, zmax = 4, xmax = 4, dt = 1, dz = 2, dx = 2, daytime = 'night', cont_plume = False,
                 debug = False):
    '''This function will solve the 2D diffusion equation as it applies to a 2m x 8m smoke plume
    initialized in the center of any shape grid. See Parameters for options.

    PARAMETERS
    ==========

    tmax: float
        The length of time to model diffusion over (seconds)
    
    xmax: float
        The length of the domain in the x direction (meters)
    
    zmax: float
        The altitude extent of the domain (meters)

    dt: float
        The step in time, in seconds, to solve the diffusion equation.

    dx: float
        The step in x, in meters, to solve the diffusion equation.

    dz: float
        The step in z, in metersm to solve the diffusion equation.

    daytime: string
        Specifies the time of day to model aoersol diffusion over, and therefore, the stability regime.
        Choices are 'night' and 'afternoon'.
    
    cont_plume: boolean
        Specify whether to only initialize a smoke plume at the start of a simulation or keep is present throughout.
        Default is False, so the plume is only an initial condition.
    
    debug: boolean
        Helps debug the function by print array/dimension shapes and detecting NaNs.
        Default is False.
        
    RETURNS
    =======

    xgrid: 1D array
        the array of values in the x direction to model diffusion over

    tgrid: 1D array
        the array of times to model diffusion over

    C: 2D array
        The solution to the diffusion equation. Contains a distribution of C02 concentrations
        over time.
    
    K: float
        The turbulent diffusion coefficient in m^2/s.
    '''

    # Initial declaration of K so that it is accessible outside the if statements
    K = None
    
    if daytime == 'night': # If modeling the night
        K = 0.1 # Very stable near the surface, small coefficient

    if daytime == 'afternoon': # If modeling the afternoon
        K = 10 # Unstable near the surface, large coefficient
    
    # CFL Criterion for 2D diffusion equation
    CFL_dt = dx**2 * dz**2 /(2*K*(dx**2 + dz**2))

    # Stability check for the model
    if dt > CFL_dt:  # if the timestep is greater than CFL, raise an error
        raise ValueError(f"dt is too large! Must be <= {CFL_dt} with dx = {dx}, dz = {dz} and K = {K}")
    
    # Length of time to simulate diffusion
    T = int(round(tmax / dt + 1))

    # Dimensions of C arrays
    Z = int(round(zmax / dz + 1))
    X = int(round(xmax / dx + 1))
    
    # Create grids in z and x with correct dimensions
    zgrid = np.linspace(0, zmax, Z)
    xgrid = np.linspace(0, xmax, X)

    # Initialize C arrays as zeros with Z, X
    C_current = np.zeros((Z, X)) # For the initial/current state of the domain
    C_next = np.zeros((Z, X)) # For the future state of the domain

    if debug: # to debug, print the shapes of the arrays and x,z dimensions
        print(C_current.shape)
        print(C_next)
        print(zgrid.shape)
        print(xgrid.shape)

    # Initialize an aersol plume near the center of x-axis at the height specified in the function call
    # Get indices in z and x
    plume_index_z = int(zmax/(2*dz))
    plume_index_x = int(xmax/(2*dx))

    # Place the plume. Assume it is 2 meters across and 8 meters tall
    C_current[plume_index_z - int(4/dz): plume_index_z + int(4/dz), plume_index_x - int(1/dz): plume_index_x + int(1/dz)] = 10 
        
    for i in range(T-1): # Loop through all time steps except last
        # Centered difference in x and z, forward difference in time
        Cxx = (C_current[1:-1, 2:] - 2*C_current[1:-1, 1:-1] + C_current[1:-1, :-2])/dz**2
        Czz = (C_current[2:, 1:-1] - 2*C_current[1:-1, 1:-1] + C_current[:-2, 1:-1])/dx**2 
        
        C_next[1:-1, 1:-1] = dt*K*(Cxx + Czz) + C_current[1:-1, 1:-1]

        # Check for NaNs in Cxx, Czz, or C for debugging
        if debug:
            if np.any(np.isnan(Cxx)) or np.any(np.isnan(Czz)) or np.any(np.isnan(C_next[:, :])):
                print(f"Warning: NaN detected at timestep {i}.")
                print(C_next[:, :])
                break

        # No flux boundary condition at the ground, keeps the aerosols inside the domain
        C_next[0, :] = C_current[1, :]

        if daytime == 'night': # if modeling the night
            C_next[-1, :] = C_current[-2, :] # Mimic an inversion with no flux condition at top of domain
        
        C_current = C_next.copy() # Update C_current to be used in the next iteration

        if cont_plume: # If the plume should continue through the whole simulation
            # Place a plume near the center of the domain at every iteration
            C_current[plume_index_z - int(4/dz): plume_index_z + int(4/dz), plume_index_x - int(1/dz): plume_index_x + int(1/dz)] = 10 
    
    C = C_current # just a name change for the returned array
        
    return zgrid, xgrid, C, K  # return the range of z, x, the solution, and K

##############################

def plot_diff(tmax = 10, zmax = 8, xmax = 8, dt = 0.1, dz = 1, dx = 1, validation = True, daytime = 'night', cont_plume = False):
    '''The function will plot the U array from heat_diff_solve() in a heatmap to see how the ground
    temperature profile evolves from initial conditions due to diffusion over time.

    PARAMETERS
    ==========

    tmax: float
        The length of time to model diffusion over (seconds)
    
    xmax: float
        The length of the domain in the x direction (meters)
    
    zmax: float
        The altitude extent of the domain (meters)

    dt: float
        The step in time, in seconds, to solve the diffusion equation.

    dx: float
        The step in x, in meters, to solve the diffusion equation.

    dz: float
        The step in z, in metersm to solve the diffusion equation.

    daytime: string
        Specifies the time of day to model aoersol diffusion over, and therefore, the stability regime.
        Choices are 'night' and 'afternoon'.
    
    cont_plume: boolean
        Specify whether to only initialize a smoke plume at the start of a simulation or keep is present throughout.
        Default is False, so the plume is only an initial condition.
    '''
    # If you want to plot validation of the model setup, call two_dim_diff()
    if validation:
        zgrid, xgrid, C, K = two_dim_diff(tmax = tmax, zmax = zmax, xmax = xmax, dt = dt, dz = dx, dx = dx)

    else: # otherwise, for the main plots, call aerosol_diff()
        zgrid, xgrid, C, K = aerosol_diff(tmax = tmax, zmax = zmax, xmax = xmax, dt = dt, dz = dx, dx = dx,
                                          daytime = daytime, cont_plume = cont_plume)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))  # create a figure for the C array
    
    # plot data from C
    fill = ax.pcolor(xgrid, zgrid, C, cmap = 'Grays')

    cbar = plt.colorbar(fill, pad=0.02)  # colorbar
    cbar.set_label("CO2 Concentration (ppt)", fontsize=15)  # colorbar label

    # Different titles based on scenario (validation or real results)
    if validation:
        ax.set_title(f"Validation of 2D Diffusion Solver \n "
                     f"time = {tmax}s, dx = {dx}m, dz = {dz}m, dt = {dt}s, K = {K}" r" $\frac{m^2}{s}$", fontsize = 20)

    elif cont_plume:
        ax.set_title(f"Diffusion of Pollution in the {daytime.capitalize()} with Continuous Plume \n"
                     f"time = {tmax}s, dx = {dx}m, dz = {dz}m, dt = {dt}s, K = {K}" r" $\frac{m^2}{s}$", fontsize = 20)
    else:
        ax.set_title(f"Diffusion of Pollution in the {daytime.capitalize()} with Instantaneous Plume \n"
                     f"time = {tmax}s, dx = {dx}m, dz = {dz}m, dt = {dt}s, K = {K}" r" $\frac{m^2}{s}$", fontsize = 20)

    # axis labels
    ax.set_xlabel("X (meters)", fontsize = 16)
    ax.set_ylabel("Altitude (meters)", fontsize = 16)

    # modify size of tick and colorbar labels
    ax.tick_params(labelsize = 16)
    cbar.ax.tick_params(labelsize=16)

##############################

def all_plots():
    '''Reproduce all plots and results from the project'''

    plot_diff() # Validation results

    # Early morning, only plume at initialization
    plot_diff(tmax = 10800, zmax = 60, xmax = 60, dt = 0.025, dz = 1, dx = 1, validation = False, daytime = 'night')

    # Early morning, continuous plume through simulation
    plot_diff(tmax = 10800, zmax = 60, xmax = 60, dt = 0.025, dz = 1, dx = 1, validation = False, daytime = 'night', cont_plume = True)

    # Afternoon, only plume at initialization
    plot_diff(tmax = 10800, zmax = 60, xmax = 60, dt = 0.025, dz = 1, dx = 1, validation = False, daytime = 'afternoon')

    # Afternoon, continuous plume through simulation
    plot_diff(tmax = 10800, zmax = 60, xmax = 60, dt = 0.025, dz = 1, dx = 1, validation = False, daytime = 'afternoon', cont_plume = True)
    
    # Discussion section plot for the night
    plot_diff(tmax = 1800, zmax = 60, xmax = 60, dt = 0.025, dz = 1, dx = 1, validation = False, daytime = 'night', cont_plume = True)

    # Discussion section plot for the afternoon
    plot_diff(tmax = 1800, zmax = 60, xmax = 60, dt = 0.025, dz = 1, dx = 1, validation = False, daytime = 'afternoon', cont_plume = True)