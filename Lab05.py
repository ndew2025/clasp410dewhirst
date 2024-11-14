#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.ion() # turn on interactive mode
plt.style.use('seaborn-v0_8-dark') # Use a style sheet

'''This file will be used to solve Lab 5'''
radearth = 6357000
mixed_layer = 50
sigma = 5.67*10**8
C = 4.2*10**6
rho = 1020

#thermal_diff = 100*3600*24*365 #m^2/yr

# 18 latitudes, delta lat of 10 degrees, nEdges = 18+1
def grid_generator(nBins = 18):
    '''Generate a grid from 0 to 180 degrees in latitude from south to north.
    Each point in the grid represents a cell center.
    
    PARAMETERS
    ==========
    
    nBins: integer
        The number of latitude bins to include in the arrat
        
    RETURNS
    =======
    
    dlat: float
        The change in latitude between bin centers'''

    dlat = 180/nBins # Latitude spacing
    lats = np.arange(0, 180, dlat) + dlat/2 # Array of latitude bin centers

    return dlat, lats

def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.
    Parameters
    ----------
    lats_in : Numpy array
    Array of latitudes in degrees where temperature is required.
    0 corresponds to the south pole, 180 to the north.
    Returns
    -------
    temp : Numpy array
    Temperature in Celcius.
    '''
    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
    23, 19, 14, 9, 1, -11, -19, -47])
    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.
    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)
    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2
    return temp

def snowball_earth(nBins=18, dt=1, tmax = 10000, thermal_diff = 100., sphere_corr = True, debug = False):
    '''Solve the snowball earth problem
    
    PARAMETERS
    ==========
    
    nBins
    dt (years)
    tmax (years)
    debug (boolean, default false)
        print the A matrix is true
    thermal_diff: float

    RETURNS
    =======
    
    lats: 1D array
        Latitude grid from the above function
    temps: 1D array
        final temperatures at the end of the simulation as a function of latitude'''

    # Create A matrix, include initial/boundary conditios
    # Create L matrix = I - lamda*dt*A (it never changes, only have to do it once)
    # Implicitly solve for the temperature one step forward in time for maxtime/tstep

    thermal_diff_yr = thermal_diff*3600*24*365 #m^2/yr

    dlat, lats = grid_generator(nBins = nBins)

    # grid spacing in meters
    dy = radearth*np.pi*dlat/180

    Temp = temp_warm(lats)

    # Get number of timesteps
    nStep = int(tmax/dt)

    if debug:
        print(f'Function called for nBins = {nBins}, dt = {dt}, tmax = {tmax}')
        print(f'This results in nStep = {nStep} time steps')
        print("Resulting lat grid:")
        print(lats)

    # Create A matrix for temperature diffusion
    I = np.identity(nBins)
    A = I*-2 # Gets diagonal to be -2
    A[np.arange(nBins-1), np.arange(nBins-1) + 1] = 1 # These lines set the off diagonal elements = 1
    A[np.arange(nBins-1) + 1, np.arange(nBins-1) ] = 1

    A[0, 1], A[-1, -2] = 2, 2 # Set the other remaining bins that aren't zero to 2
    A *= dy**-2

    # Create B matrix for spherical correction term
    B = np.zeros((nBins, nBins)) # Gets diagonal to be -2
    B[np.arange(nBins-1), np.arange(nBins-1) + 1] = 1 # These lines set the off diagonal elements = 1
    B[np.arange(nBins-1) + 1, np.arange(nBins-1) ] = -1

    # Set boundary conditions in B
    B[0, :] = 0
    B[-1, :] = 0

    # Area of onion ring latitude band as a function of latitude (max at equator, 0 at poles)
    Axz = np.pi((radearth+50)**2 + radearth**2)*np.sin(np.pi/180*lats)
    dAxz = np.matmul(B, Axz) / (Axz*4*dy**2)

    if debug:
        print(f'A = {A}')

    # Make L matrix
    #dt_sec = dt*3600*24*365 # year to seconds
    L = I - dt*thermal_diff_yr*A
    L_inv = np.linalg.inv(L)

    for i in range(nStep):
        # Add spherical correction term
        if sphere_corr:
            Temp += dt*thermal_diff_yr*dAxz
        Temp = np.matmul(L_inv, Temp)
    
    return lats, Temp

def test_snowball(nBins = 18, sphere_corr = True):
    '''Reproduce example plot in lecture/handout. 
    
    Using out default values and a warm earth initial condition, plot the initial conditions,
    the simple diffusion, simple diffusion with spherical correction, and diffusion + 
    correction + insolation'''

    dlat, lats = grid_generator(nBins)
    initial_temps = temp_warm(lats)
    lats, temp_diff = snowball_earth(sphere_corr = False)
    lats, temp_diff_spherical = snowball_earth(sphere_corr = True)

    fig, ax = plt.subplots(1, 1, figsize = (10,10))
    ax.plot(lats, initial_temps, lw = 3, label = "Initial Temperatures")
    ax.plot(lats, temp_diff, lw = 3, label = 'Basic Diffusion')
    ax.legend(loc = 'best')
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Temperature (\u00B0C)")
    ax.grid(True)
