#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.ion() # turn on interactive mode
plt.style.use('seaborn-v0_8-dark') # Use a style sheet

'''This file will be used to investigate the snowball earth hypothesis in Lab 5'''

# Constants for all equations
radearth = 6357000 # radius of the earth (meters)
mixed_layer = 50 # mixed layer of the ocean to diffuse heat latitudinally (maters)
sigma = 5.67*10**-8 # Stefan-Boltzmann constant (W/(m^2*K^4))
C = 4.2*10**6 # volumetric heat capacity of water (J/(m^3*K))
rho = 1020 # density of ocean water (kg/m^3)

##############################

def grid_generator(nBins = 18):
    '''Generate a grid from 0 to 180 degrees in latitude from south to north.
    Each point in the grid represents a cell center.
    
    PARAMETERS
    ==========
    
    nBins: integer
        The number of latitude bins to split earth into
        
    RETURNS
    =======
    
    dlat: float
        The change in latitude between bin centers'''

    dlat = 180/nBins # Latitude spacing
    lats = np.arange(0, 180, dlat) + dlat/2 # Array of latitude bin centers

    return dlat, lats

##############################

def temp_warm(lats, hot_earth = False, cold_earth = False):
    '''
    Create a temperature profile for modern day "warm" earth.

    PARAMETERS
    ==========

    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 is the north.

    RETURNS
    =======
    
    temp : Numpy array
        Temperature in Celsius.
    '''
    # Set initial temperatures
    if hot_earth: # For a blazing hot earth
        T_warm = np.zeros(lats.size) + 60 # 60C everywhere
    
    elif cold_earth: # For a frigid cold earth
        T_warm = np.zeros(lats.size) - 60 # -60C everywhere

    else: # For a standard earth
        T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
        23, 19, 14, 9, 1, -11, -19, -47]) 

    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats + coeffs[0] * lats**2

    return temp

##############################

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''
    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

##############################

def snowball_earth(nBins = 18, dt=1, tmax = 10000, thermal_diff = 100., albedo = 0.3, E = 1, S0 = 1370, 
                   gamma = 1, hot_earth = False, cold_earth = False, sphere_corr = True, debug = False):
    '''Investigate how the earth changes its temperature over time
    
    PARAMETERS
    ==========
    
    nBins: integer
        The number of latitude bins to split the earth into

    dt: float
        The time resolution of the model simulation. Default is 1 year.
    
    tmax: float:
        The time to run the model simulation over. Default is 10,000 years.
    
    thermal_diff: float
        The thermal diffusivity of ocean water in m^2/s. Default is 100.

    albedo: float
        The albedo of the earth's surface. Default is 0.3.
    
    E: float
        The emissivity of the earth's surface. Default is 1.
    
    S0: float
        The solar constant in W/m^2. Default and measured value is 1370.
    
    gamma: float
        Insolation scaling factor that changes the shortwave radiation received by earth.
        Default is 1 to match expected results from the solar constant.
    
    hot_earth: boolean
        Specifies whether to use hot earth initial conditions for a model simulation (60C everywhere).
        If toggled on will also use dynamic albedo based on the earth's surface. Default is False.

    cold_earth: boolean
        Specifies whether to use cold earth initial conditions for a model simulation (-60C everywhere).
        If toggled on will also use dynamic albedo based on the earth's surface. Default is False. 

    sphere_corr: boolean
        Specifies whether to take the spherical shape into account in the model.
        Default is true for most realistic results 

    debug: boolean
        Helps debug the function. Prints some function arguments to verify intentions, the latitude grid,
        and the A matrix to make sure it is populated correctly.

    RETURNS
    =======
    
    lats: 1D array
        Latitude grid from the above function
    
    temps: 1D array
        Final temperatures at the end of the simulation as a function of latitude'''

    dt_sec = 365*24*3600 # convert dt in years to seconds to match units of thermal diffusivity

    # initial earth with latitude bands
    dlat, lats = grid_generator(nBins = nBins)

    # grid spacing in meters
    dy = radearth*np.pi*dlat/180

    # Generate insolation
    insol = gamma*insolation(S0, lats)

    # Get the initial conditions for the model based on real average temperatures
    Temp = temp_warm(lats, hot_earth = hot_earth, cold_earth = cold_earth)

    # Get number of timesteps
    nStep = int(tmax/dt)

    # use debugging print statements
    if debug:
        print(f'Function called for nBins = {nBins}, dt = {dt}, tmax = {tmax}')
        print(f'This results in nStep = {nStep} time steps')
        print("Resulting lat grid:")
        print(lats)

    # Create A matrix for temperature diffusion
    I = np.identity(nBins) # properly sized identity matrix
    A = I*-2 # Gets diagonal to be -2

    # Set the off diagonal elements = 1
    A[np.arange(nBins-1), np.arange(nBins-1) + 1] = 1 
    A[np.arange(nBins-1) + 1, np.arange(nBins-1) ] = 1

    # Set boundary conditions at the poles to stop diffusion
    A[0, 1], A[-1, -2] = 2, 2 
    A /= dy**2 # multiplier on the matrix

    # Create B matrix for spherical correction term
    B = np.zeros((nBins, nBins)) # Gets diagonal to be -2

    # Set the off diagonal elements = 1
    B[np.arange(nBins-1), np.arange(nBins-1) + 1] = 1 
    B[np.arange(nBins-1) + 1, np.arange(nBins-1) ] = -1

    # Set boundary conditions in B
    B[0, :] = 0
    B[-1, :] = 0

    # Area of onion ring latitude band as a function of latitude (max at equator, 0 at poles)
    Axz = np.pi*((radearth+50)**2 + radearth**2)*np.sin(np.pi/180*lats)
    dAxz = np.matmul(B, Axz) / (Axz*4*dy**2)

    # print A matrix for debugging
    if debug:
        print(f'A = {A}')

    # Make L matrix
    L = I - dt_sec*thermal_diff*A
    L_inv = np.linalg.inv(L) # invert

    for i in range(nStep): # for each dt in tmax (every step in the model)
        # Add spherical correction term if specified
        if sphere_corr:
            Temp += dt_sec*thermal_diff*dAxz*np.matmul(B, Temp)
        
        if hot_earth or cold_earth: # If either hot or cold earth scenario

            # Setup albedo array for varying reflection based on the presence of ice and snow
            dyn_albedo = np.zeros(lats.size) # Initialize albedo array
            loc_ice = Temp <= -10 # Ice where temperature is less than -10
            dyn_albedo[loc_ice] = 0.6 # Using 0.6 as albedo of ice
            dyn_albedo[~loc_ice] = 0.3 # Using 0.3 as albedo of the ground (where ice is *not* present)

            # Use the varying albedo based on surface type
            radiation = (1-dyn_albedo) * insol - E*sigma*(Temp+273.15)**4

        else: # Otherwise, use the earth's averaged albedo
            radiation = (1-albedo) * insol - E*sigma*(Temp+273.15)**4
        
        # add radiative effects
        Temp += dt_sec*radiation / (rho*C*mixed_layer)
        Temp = np.matmul(L_inv, Temp)
    
    # Return final temperatures and latitudes
    return lats, Temp

##############################

def test_snowball(nBins = 18):
    '''Reproduce example plot in lecture/handout to see how each model component behaves 

    PARAMETERS
    ==========

    nBins: integer
        the number of latitude bands to split the earth into
    '''
    # Initialize the earth latitude bins
    dlat, lats = grid_generator(nBins)

    # Get initial conditions
    initial_temps = temp_warm(lats)

    # Simulation with basic diffusion only
    lats, temp_diff = snowball_earth(sphere_corr = False, S0 = 0, E = 0)

    # Simulation with diffusion and spherical correction.
    lats, temp_diff_spherical = snowball_earth(S0 = 0, E = 0)

    # Simulation with diffusion, spherical correction, and radiative forcing
    lats, temp_diff_sphere_rad = snowball_earth()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize = (10,10))

    # Plot the initial conditions and all 3 solutions above
    ax.plot(lats-90, initial_temps, lw = 3, label = "Initial Temperatures")
    ax.plot(lats-90, temp_diff, lw = 3, label = 'Basic Diffusion')
    ax.plot(lats-90, temp_diff_spherical, lw = 3, label = "Basic Diffusion with Spherical Correction")
    ax.plot(lats-90, temp_diff_sphere_rad, lw = 3, label = "Diffusion with Spherical Correction and Radiative Terms")
    
    # Lengend, labels, title, grid, etc
    ax.legend(loc = 'best', frameon = True, fontsize = 12)
    ax.set_xlabel("Latitude (degrees)", fontsize = 13)
    ax.set_ylabel("Temperature (\u00B0C)", fontsize = 13)
    ax.set_title(f"Components of the Snowball Earth Model", fontsize = 18)
    ax.tick_params(labelsize=13)  # modify size of tick and colorbar labels
    ax.grid(True)

##############################

def question_2():
    '''Vary lambda (thermal diffusivity on the ocean) and epsilon (the emissivity of the earth's surface)
    to create an equilibrium solution including diffusion, net radiation, and spherical correction that
    closely matches the warm Earth equilibrium from the temp_warm() function.'''

    dlat, lats = grid_generator() # Generate the latitudes
    warm_earth = temp_warm(lats) # Generate the warm earth equilibrium

    # Setup empty lists to store the absolute cumulative difference between the warm Earth equilibrium
    # and model solutions
    summed_diffs = []
    all_lambdas = []
    all_emiss = []

    # Loop through each combination of lambda and epsilon
    for thermal_diff in np.arange(0, 155, 5):
        for E in np.arange(0, 1.05, 0.05):
            # Call the snowball earth function
            lats, Temp = snowball_earth(thermal_diff = thermal_diff, E = E)

            all_lambdas.append(thermal_diff) # Append the diffusivities
            all_emiss.append(E) # Append the emmisivities

            # absolute value of the differences between each temperatures
            abs_diffs = np.abs(Temp-warm_earth)
            total_diff = np.sum(abs_diffs) # sum them up
            summed_diffs.append(total_diff) # append the summed differences
    
    ordered_diffs_indices = np.argsort(summed_diffs) # sort the indices so that the min diff is first
    min_diff_index = ordered_diffs_indices[0] # Get the index of the min diff
    optimal_E = round(all_emiss[min_diff_index],2) # Get the associated emissivity
    optimal_lambda = all_lambdas[min_diff_index] # Get associated thermal diffusivity

    # Run a new simulation based on lambda and epsilon above
    lats, Temp = snowball_earth(thermal_diff = optimal_lambda, E = optimal_E)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize = (10,10))

    # Plot the warm earth equilibrium and solution from snowball_earth()
    ax.plot(lats, warm_earth, lw = 3, c = 'red', label = "Warm Earth Equilibrium")
    ax.plot(lats, Temp, lw = 3, c = 'blue', label = "Steady-state from Snowball Earth Function")
    
    # legend, labels, title, grid, etc.
    ax.legend(loc = 'best', frameon = True, fontsize = 12)
    ax.set_xlabel("Latitude (deg)", fontsize = 13)
    ax.set_ylabel("Temperature (\u00B0C)", fontsize = 13)
    ax.set_title("Warm Earth Equilibrium and Nearest Match (Minimum Cumulative Difference) \n"
                 f"$\lambda$ = {optimal_lambda}, $\epsilon$ = {optimal_E}",
                 fontsize = 18)
    ax.tick_params(labelsize=13)  # modify size of tick and colorbar labels
    ax.grid(True)

##############################

def question_3():
    '''Plot temperature curves against latitude for the hot earth and cold earth with dynamic albedo
    and the "flash freeze" scenario with an averaged albedo of 0.6'''

    # Run model simulations for hot and cold initial conditions and a flash freeze scenario
    lats, hot_earth = snowball_earth(thermal_diff = 35, E = 0.7, hot_earth = True)
    lats, cold_earth = snowball_earth(thermal_diff = 35, E = 0.7, cold_earth = True)
    lats, flash_freeze = snowball_earth(thermal_diff = 35, E = 0.7, albedo = 0.6)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize = (10,10))

    # Plot each solution from the simulations
    ax.plot(lats-90, hot_earth, lw = 3, c = "red", label = "Hot Earth (60\u00B0C Initial Condition)")
    ax.plot(lats-90, cold_earth, lw = 3, c = "blue", label = "Cold Earth (-60\u00B0C Initial Condition)")
    ax.plot(lats-90, flash_freeze, lw = 3, c = "black", label = "Flash Freeze (albedo = 0.6)")

    # legend, labels, title, grid, etc.
    ax.legend(loc = 'best', frameon = True, fontsize = 13)
    ax.set_xlabel("Latitude (deg)", fontsize = 13)
    ax.set_ylabel("Temperature (\u00B0C)", fontsize = 13)
    ax.set_title("Extremely Hot, Cold, and High Albedo Cases on Earth \n Temperatures after 10,000 years",
                 fontsize = 18)
    ax.tick_params(labelsize=13)  # modify size of tick and colorbar labels
    ax.grid(True)

##############################

def sequential_earth(all_temps, i, gamma, nBins = 18, dt=1, tmax = 10000, thermal_diff = 100., E = 1, S0 = 1370, 
                     sphere_corr = True):
    ''' Works very similarly to the snowball_earth() function except that it should be used for many 
    consecutive simulations. This model uses the solution from the previous run as the initial conditions
    for the next one when implemented in the question_4() function below.

    PARAMETERS
    ==========
    
    all_temps: 2D array
        An array to store the final temperatures of each consecutive simulation using this model
    
    i: integer
        The index of the row from the all_temps array to access the previous simulation as the new
        initial condition
    
    gamma: float
        Insolation scaling factor that changes the shortwave radiation received by earth.
        Default is 1 to match expected results from the solar constant.
    
    nBins: integer
        The number of latitude bins to split the earth into

    dt: float
        The time resolution of the model simulation. Default is 1 year.
    
    tmax: float:
        The time to run the model simulation over. Default is 10,000 years.
    
    thermal_diff: float
        The thermal diffusivity of ocean water in m^2/s. Default is 100.

    albedo: float
        The albedo of the earth's surface. Default is 0.3.
    
    E: float
        The emissivity of the earth's surface. Default is 1.
    
    S0: float
        The solar constant in W/m^2. Default and measured value is 1370.
    
    sphere_corr: boolean
        Specifies whether to take the spherical shape into account in the model.
        Default is true for most realistic results 

    RETURNS
    =======
    
    lats: 1D array
        Latitude grid from the above function
    
    temps: 1D array
        Final temperatures at the end of the simulation as a function of latitude'''
    
    dt_sec = 365*24*3600 # convert dt in years to seconds to match units of thermal diffusivity

    # initial earth with latitude bands
    dlat, lats = grid_generator(nBins = nBins)

    # grid spacing in meters
    dy = radearth*np.pi*dlat/180

    # Generate insolation
    insol = gamma*insolation(S0, lats)

    # Get initial conditions from the previous simulation
    Temp = all_temps[i,:]

    # Get number of timesteps
    nStep = int(tmax/dt)

    # Create A matrix for temperature diffusion
    I = np.identity(nBins) # properly sized identity matrix
    A = I*-2 # Gets diagonal to be -2

    # Set the off diagonal elements = 1
    A[np.arange(nBins-1), np.arange(nBins-1) + 1] = 1 
    A[np.arange(nBins-1) + 1, np.arange(nBins-1) ] = 1

    # Set boundary conditions at the poles to stop diffusion
    A[0, 1], A[-1, -2] = 2, 2 
    A /= dy**2 # multiplier on the matrix

    # Create B matrix for spherical correction term
    B = np.zeros((nBins, nBins)) # Gets diagonal to be -2

    # Set the off diagonal elements = 1
    B[np.arange(nBins-1), np.arange(nBins-1) + 1] = 1 
    B[np.arange(nBins-1) + 1, np.arange(nBins-1) ] = -1

    # Set boundary conditions in B
    B[0, :] = 0
    B[-1, :] = 0

    # Area of onion ring latitude band as a function of latitude (max at equator, 0 at poles)
    Axz = np.pi*((radearth+50)**2 + radearth**2)*np.sin(np.pi/180*lats)
    dAxz = np.matmul(B, Axz) / (Axz*4*dy**2)

    # Make L matrix
    L = I - dt_sec*thermal_diff*A
    L_inv = np.linalg.inv(L) # invert

    for i in range(nStep):
        # Add spherical correction term if specified
        if sphere_corr:
            Temp += dt_sec*thermal_diff*dAxz*np.matmul(B, Temp)

        # Setup albedo array for varying reflection based on the presence of ice and snow
        dyn_albedo = np.zeros(lats.size) # Initialize albedo array
        loc_ice = Temp <= -10 # Ice where temperature is less than -10
        dyn_albedo[loc_ice] = 0.6 # Using 0.6 as albedo of ice
        dyn_albedo[~loc_ice] = 0.3 # Using 0.3 as albedo of the ground (where ice is *not* present)

        # Use the varying albedo based on surface type
        radiation = (1-dyn_albedo) * insol - E*sigma*(Temp+273.15)**4
        
        # Add radiation term
        Temp += dt_sec*radiation / (rho*C*mixed_layer)
        Temp = np.matmul(L_inv, Temp)
    
    # Return the final temperatures and latitudes
    return lats, Temp

##############################

def question_4(nBins = 18, cold_earth = True):
    '''See whether it is easier to get into a snowball earth or get out of it. This function
    uses sequential_earth() from above to initialize a new simulation based on the previous solution'''

    gamma_up = np.arange(0.45, 1.45, 0.05) # range of increasing gamma 
    gamma_down = np.arange(1.35, 0.35, -0.05) # range of decreasing gamma
    gamma_loop = list(gamma_up) + list(gamma_down) # convert to lists and add for a single loop below

    # total steps in the simulation. +1 because of the simulation from the initial conditions before
    # modifying the solar forcing
    total_steps = gamma_up.size + gamma_down.size + 1

    # Array that will contain the final temperatures for all latitudes every 10,000 years
    all_temps = np.zeros((total_steps, nBins)) 

    # Initial conditions for a cold earth
    lats, Temp = snowball_earth(gamma = 0.4, thermal_diff = 35, E = 0.7, cold_earth = cold_earth) 
    all_temps[0,:] = Temp # First row in the array contains this solution

    for i, gamma in enumerate(gamma_loop): # range of gamma and its indices to loop over
        # Call the sequential_earth() function
        lats, Temp = sequential_earth(all_temps, i, gamma, thermal_diff = 35, E = 0.7)
        all_temps[i+1,:] = Temp # insert the solution to the i+1 column because of the initial conditions
    
    # Once all done, calculate the average planetary temperatures after each simulation 
    avg_temps = np.mean(all_temps, axis = 1)

    # split the temperatures into the portion where insolation is increasing and decreasing
    solar_increase = avg_temps[:21]
    solar_decrease = avg_temps[20:]

    # New ranges for gamma for smooth plotting based on the subsetted temperatures above
    plot_gamma_up = np.arange(0.4, 1.45, 0.05)
    plot_gamma_down = np.arange(1.4, 0.35, -0.05)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize = (10,10))

    # Plot the subsetted temperatures
    ax.plot(plot_gamma_up, solar_increase, lw = 3, c = 'red', label = "Temperatures with Increasing Insolation")
    ax.plot(plot_gamma_down, solar_decrease, lw = 3, c = 'blue', label = "Temperatures with Decreasing Insolation")

    # legend, labels, title, grid, etc.
    ax.legend(loc = 'best', frameon = True, fontsize = 12)
    ax.set_xlabel("Insolation Scaling Factor", fontsize = 13)
    ax.set_ylabel("Average Planetary Temperature (\u00B0C)", fontsize = 13)
    ax.set_title("Changes in Earth's Average Temperature with Varying Solar Forcing \n"
                 f'Initial Temperature = {np.round(avg_temps[0], 1)}\u00B0C, '
                 f'Final Temperature = {np.round(avg_temps[-1], 1)}\u00B0C', fontsize = 18)
    ax.tick_params(labelsize=13)  # modify size of tick and colorbar labels
    ax.grid(True)

##############################

def all_plots():
    '''Reproduce all of the plots in the lab report'''

    test_snowball()
    question_2()
    question_3()
    question_4()