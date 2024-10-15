
# This file will solve the n-layer atmosphere energy balance problem

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-dark') # Use a style sheet
plt.ion() # Turn on interactive mode

# function inputs: emissivity, number of layers, solar constant, use alpha = 0.33
# function outputs: temperatures at each layer

# Step 1: construct the 3 matrices
        # 
        # use np.linalg.inv() for inverse
        # use np.matmul(Ainv, B), then use Stefan-Boltzmann relationship to get the temperature at each layer

# Define some constants
sigma = 5.67E-8 

def n_layer_atm(N, E, S0 = 1350, a = 0.33, debug  = False, nuclear_winter = False):

    '''Solve the n-layer atmosphere problem and return the temperature at each layer.
    
    Fill in more docstring later'''

    # Setup the shape matrices
    A = np.zeros([N+1, N+1])
    B = np.zeros(N+1)

    # Only change needed to B
    B[0] = -S0/4*(1 - a)

    if nuclear_winter:
        B[0] = 0
        B[-1] = -S0/4

    # Populate the A matrix
    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                A[i, j] = -1*(i > 0) - 1 # Both of these lines do the same, choose one

            else:
                m = np.abs(j-i) - 1
                A[i, j] = E*(1-E)**m
    
    # At earth's surface, emissivity is always 1, so divide it out when we use emmisivities < 1
    # for the layers of the atmosphere
    A[0, 1:] /= E

    if debug:
        print(A) # verify the code above is working

    # Get the inverse of our A matrix
    Ainv = np.linalg.inv(A)

    # Multiply the inverse of A by B to get the fluxes
    fluxes = np.matmul(Ainv, B)

    # Use Stefan-Boltzmann relationship to solve for temperature at all the atmospheric layers
    temps = (fluxes/(E*sigma))**0.25

    # Same as above, but at the ground (first layer) emmisivity is always 1
    temps[0] = (fluxes[0]/sigma)**0.25

    return temps

def temp_v_emissivity(N, real_temp = 288):
    '''Creates a plot of earth's surface temperature as a function of emmisivity
    
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

def temp_v_altitude(E = 0.255, real_temp = 288, S0 = 1350):
    ''' The function will calculate temperatures for a range of total number of layers, check which provides the 
    closest solution to the real surface temperature of 288K, then create a plot of the temperature structure 
    of earth's atmosphere based on the number of layers that best satisifies the above criterion.
    
    PARAMETERS
    ==========

    E: float
        The emissivity of earth's atmosphere, default is 0.255
    
    real_temp: float
        The real average surface temperature of the earth, default is 288K
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

    temps_to_plot = n_layer_atm(optimal_N, E) # Recalculate an energy balance solution based on the number of layers above

    layers_to_plot = np.arange(0, optimal_N + 1, 1) # Range of layers to plot 
    
    fig, ax = plt.subplots(1,1) # Setup a figure

    ax.plot(temps_to_plot, layers_to_plot, lw = 4, c = "black") # plot temperature against emissivity

    # Setup title, labels
    ax.set_title(f"Temperature Profile for a {optimal_N}-layer Atmosphere, E = {E}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Atmospheric Layer/Altitude")
    ax.grid(True) # turn on grid




    
