#!/usr/bin/env python3

# Import necessary Python libraries and modules
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-dark') # Use a style sheet
plt.ion() # Turn on interactive mode

'''This file will solve the n-layer atmosphere energy balance problem. The first function will calculate 
the temperatures at every layer for N number of layers in the model. The second function will produce a plot
of earth's temperature as a function of emissivity, and the third and fourth functions will produce atmopsheric 
temperature profiles for different planets with different radiative characteristics and a nuclear winter scenario.'''

# Stefan_Boltzmann constant for temperature calculations
sigma = 5.67E-8 

def n_layer_atm(N, E, S0 = 1350, a = 0.33, debug  = False, nuclear_winter = False):

    '''This function solves the n-layer atmosphere problem and returns the temperature of each layer
    and the surface of the earth.
    
    PARAMETERS
    ==========

    N: integer
        The number of atmospheric layers to be used in the energy balance model
    
    E: float
        emissivity of the atmosphere
    
    S0: float
        the planet's solar constant in W/m^2
    
    a: float
        albedo, the fraction of incoming radiation reflected from the planet 
    
    debug: boolean
        default is False. Indicate True to verify that the A matrix was filled correctly.
    
    nuclear_winter: boolean
        default is False. Inidicate true to model a nuclear winter scenario. This changes the distribution of
        shortwave energy, specifically by absorbing it at the top layer instead of the surface.
    
    RETURNS
    =======

    temps: array
        the temperatures at each layer in the model. The first value represents the earth's surface,
        and the last represents the top of the atmosphere.'''

    # Setup the shape of the matrices, N+1 because we need to account for the earth's surface
    A = np.zeros([N+1, N+1])
    B = np.zeros(N+1)

    # Only change needed to B. The shortwave flux is absorbed at the ground.
    B[0] = -S0/4*(1 - a)

    if nuclear_winter: # If the user wants to model a nuclear winter
        B[0] = 0 # No shortwave at the ground
        B[-1] = -S0/4*(1 - a) # all shortwave absorbed at the top of the atmosphere

    # Populate the A matrix
    for i in range(N+1): # Loop through the rows of matrix A
        for j in range(N+1): # Loop through the columns of matrix A
            if i == j: # If i and j are the same value
                A[i, j] = -1*(i > 0) - 1 # [i, j] = -1 when i = 0. When greater than 1, [i, j] = -1 - 1 = -2

            else: # when i and j are different values
                m = np.abs(j-i) - 1
                A[i, j] = E*(1-E)**m # the longwave fluxes are scaled by this factor
    
    # At earth's surface, emissivity is always 1, so divide it out
    A[0, 1:] /= E

    if debug: # If the user isn't sure about the function
        print(A) # verify the A matrix was filled properly

    # Get the inverse of the A matrix
    Ainv = np.linalg.inv(A)

    # Multiply the inverse of A by B to get the fluxes
    fluxes = np.matmul(Ainv, B)

    # Use Stefan-Boltzmann relationship to solve for temperature at all the atmospheric layers
    temps = (fluxes/(E*sigma))**0.25

    # Same as above, but at the ground emmisivity is always 1
    temps[0] = (fluxes[0]/sigma)**0.25

    return temps # Return the array of solved temperatures

def temp_v_emissivity(N, real_temp = 288):
    '''Creates a plot of earth's surface temperature as a function of emissivity
    
    PARAMETERS
    ==========

    N: integer
        The number of atmospheric layers to use in the energy balance solver

    real_temp: float
        The actual temperature of the surface of the earth to identify the assoicated value of emmisivity.
        The figure will plot a horizontal line at this temperature.
    '''

    emissivity_range = np.linspace(0, 1, 100) # Define a range of atmospheric emisivities to calculate temperature with

    sfc_temps = [] # Setup an empty list to append calculated temperatures to
    
    for E in emissivity_range: # Loop through the emissivities
        temps = n_layer_atm(N, E) # call the energy balance solver function
        sfc_temps.append(temps[0]) # Append the first temperature in the returned array, which represents the surface
    
    fig, ax = plt.subplots(1,1) # Setup a figure

    ax.plot(emissivity_range, sfc_temps, lw = 4, c = "black") # plot temperature against emissivity

    # Setup a horizontal line at the actual temperature to check emissivity value
    ax.axhline(real_temp, lw = 2, ls = '--', c = "red", label = f'Surface Temperature = {real_temp}K') 

    # Setup title, labels, and legend
    ax.set_title(f"Earth's Temperature as a Function of Emissivity for a {N}-layer Atmosphere")
    ax.set_xlabel("Emissivity")
    ax.set_ylabel("Surface Temperature (K)")
    ax.legend(loc = 'best')
    ax.grid(True) # turn on grid

def temp_v_altitude(E = 0.255, real_temp = 288, S0 = 1350, a = 0.33):
    ''' The function will calculate temperatures for a range of total number of layers, check which provides the 
    closest solution to the real surface temperature, then creates a plot of the temperature structure 
    of the atmosphere based on the number of layers that best satisifies the above criterion.
    
    PARAMETERS
    ==========

    E: float
        The emissivity of earth's atmosphere, default is 0.255
    
    real_temp: float
        The real average surface temperature of the earth, default is 288K

    S0: float
        the planet's solar constant in W/m^2
    
    a: float
        albedo, the fraction of incoming radiation reflected from the planet 
    '''

    layer_range = np.arange(0,101,1) # Reasonable range of total number of layers to calculate surface temps with

    sfc_temps = [] # Setup an empty list to append calculated temperatures to

    for N in layer_range: # Loop through layer_range
        temps =  n_layer_atm(N, E, S0 = S0) # Call the energy balance solver function
        sfc_temps.append(temps[0]) # Append the first temperature in the returned array, which represents the surface
    
    temp_diffs = [] # Setup an empty list to append temperature differences to
    for temp in sfc_temps: # Loop through new surface temp list 
        temp_diffs.append(np.abs(temp - real_temp)) # Find the absolute difference between the calulcated and real temp
        # and append it to temp_diffs
    
    min_diff = min(temp_diffs) # Find the minimum difference
    nearest_temp_index = temp_diffs.index(min_diff) # Find the index of the minimum difference

    optimal_N = layer_range[nearest_temp_index] # and find the number of layers that corresponds to the minimum

    temps_to_plot = n_layer_atm(optimal_N, E, S0 = S0, a = a) # Recalculate a solution based on the number of layers above

    layers_to_plot = np.arange(0, optimal_N + 1, 1) # Range of layers to plot 
    
    fig, ax = plt.subplots(1,1) # Setup a figure

    ax.plot(temps_to_plot, layers_to_plot, lw = 4, c = "black") # plot temperature against altitude

    # Setup title, labels
    ax.set_title(f"Temperature Profile for a {optimal_N}-layer Atmosphere \n E = {E}, S0 = {S0} $\\text{{W/m}}^2$, a = {a}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Atmospheric Layer/Altitude")
    ax.grid(True) # turn on grid

def nuclear_winter(N, E, S0 = 1350, a = 0):
    '''The function utilizes my nuclear_winter keyword argument in n_layer_atm() to plot the temperature profile of earth's
    atmosphere in a nuclear winter scenario.

    PARAMETERS
    ==========

    N: integer
        The number of atmospheric layers to be used in the energy balance model
    
    E: float
        emissivity of the atmosphere
    
    S0: float
        the planet's solar constant in W/m^2
    
    a: float
        albedo, the fraction of incoming radiation reflected from the planet 
    '''

    temps = n_layer_atm(N, E, S0 = S0, a = a, nuclear_winter = True) # call the energy balance solver function for a nuclear winter

    layers = np.arange(0, N+1, 1) # number of layers for the plot, including earth's surface

    fig, ax = plt.subplots(1,1)

    ax.plot(temps, layers, lw = 4, c = "black") # plot temperature against altitude

    # Setup title, labels
    ax.set_title(f"Temperature Profile for a Nuclear Winter {N}-layer Atmosphere \n E = {E}, S0 = {S0} $\\text{{W/m}}^2$, a = {a}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Atmospheric Layer/Altitude")
    ax.grid(True) # turn on grid

    print(f"Earth's surface temperature is {temps[0]}") # print the resulting surface temperature






    
