#!/usr/bin/env python3

# Import necessary Python libraries and modules
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-dark') # Use a style sheet
plt.ion() # Turn on interactive mode

'''This file contains 6 functions. The first two return the two set of Lotka-Volterra equations. 
The third and fourth are my Euler and Runge-Kutta ODE solvers. The fifth utilizes both ODE solvers and creates
plots of their solutions to be compared against each other. The sixth and final function aids in describing how varying
the Lotka-Volterra parameters effects the evolution of two species in a competition or predator-prey relationship. 
'''

def dNdt_comp(t, N, a, b, c, d):
    '''
    This function calculates the Lotka-Volterra competition equations for two species. 
    Given normalized populations, `N1` and `N2`, as well as the four coefficients 
    representing population growth and decline, calculate the time derivatives dN_1/dt and dN_2/dt 
    and return to the caller. This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this function.

    PARAMETERS
    ==========
    t: float
        The current time (not used here).

    N: two-element list
        The current values of N1 and N2 (e.g. N = [N1, N2])

    a, b, c, d: floats
        The value of the Lotka-Volterra coefficients.

    RETURNS
    =======
    dN1dt, dN2dt: floats
        The time derivatives of `N1` and `N2`.
    '''
    # Extract the current values of N1 and N2 from the list
    N1 = N[0]
    N2 = N[1]

    # Calculate the time derivatives for N1 and N2
    dN1dt = a*N1*(1-N1) - b*N1*N2
    dN2dt = c*N2*(1-N2) - d*N2*N1

    return dN1dt, dN2dt

def dNdt_pred_prey(t, N, a, b, c, d):
    '''
    This function calculates the Lotka-Volterra predator/prey equations for two species. 
    Given normalized populations, `N1` and `N2`, as well as the four coefficients 
    representing population growth and decline, calculate the time derivatives dN_1/dt and dN_2/dt 
    and return to the caller. This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this function.

    PARAMETERS
    ==========
    t: float
        The current time (not used here).

    N: two-element list
        The current values of N1 and N2

    a, b, c, d: floats
        The value of the Lotka-Volterra coefficients.

    RETURNS
    =======
    dN1dt, dN2dt: floats
        The time derivatives of `N1` and `N2`.
    '''

    # Extract the current values of N1, N2 from the list
    N1 = N[0]
    N2 = N[1]

    # Calculate the time derivatives for N1 and N2
    dN1dt = a*N1 - b*N1*N2
    dN2dt = -c*N2 + d*N2*N1

    return dN1dt, dN2dt

def euler_solve(func, N1_init, N2_init, dt, t_final, a, b, c, d):
    '''
    This function utilizes Euler's Method to solve an ordinary differential equation.

    PARAMETERS
    ==========

    func: function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init: float
    <more good docstring here>
    '''

    # Initialize time array and population arays of the same size
    time = np.arange(0.0, t_final, dt)
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)

    # First indices of the empty arrays are the initial conditions
    N1[0] = N1_init
    N2[0] = N2_init

    # Calculate the time series of the two populations using the time derivatives
    # and initial conditions
    for i in range(time.size-1): # Loop through indices to fill the empty arrays
        dN1dt, dN2dt = func(i+1, [N1[i], N2[i]], a, b, c, d) # Calculate the time derivatives
        N1[i+1] = N1[i] + dt*dN1dt # Calculate the updated populations for each timestep
        N2[i+1] = N2[i] + dt*dN2dt
    
    # Return values to caller.
    return time, N1, N2


def solve_rk8(func, N1_init, N2_init, dt, t_final, a, b, c, d):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.

    PARAMETERS
    ==========
    func: function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.

    N1_init, N2_init: float
        Initial conditions for `N1` and `N2`, ranging from (0,1]

    dt: float
        Largest timestep allowed in years

    t_final: float
        Integrate until this value is reached, in years.

    a, b, c, d: float
        Lotka-Volterra coefficient values

    RETURNS
    =======
    time: Numpy array
        Time elapsed in years.
    N1, N2: Numpy arrays
        Normalized population density solutions.
    '''
    # Import the problem solver from scipy
    from scipy.integrate import solve_ivp

    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
    args=[a, b, c, d], method='DOP853', max_step = dt)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    # Return values to caller.
    return time, N1, N2


def plot_populations(func, N1_init, N2_init, dt, t_final, a, b, c, d):
    '''This function will use the population time series calculated by each of the 
    differential equation solvers and create a plot to visualize the differences
    between them.
    
    PARAMETERS
    ==========
    func: function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
        Options are dNdt_comp or dNdt_pred_prey.

     N1_init, N2_init: float
        Initial conditions for `N1` and `N2`, ranging from (0,1]

    dt: float
        Largest timestep allowed in years

    t_final: float
        Integrate until this value is reached, in years.

    a, b, c, d: floats
        The value of the Lotka-Volterra coefficients.
    '''

    # Call the Euler and RK8 Solvers
    time_euler, N1_euler, N2_euler = euler_solve(func, N1_init, N2_init, dt, t_final, a, b, c, d)
    time_rk8, N1_rk8, N2_rk8 = solve_rk8(func, N1_init, N2_init, dt, t_final, a, b, c, d)

    # Setup the figure with a single subplot
    fig, ax = plt.subplots(1, 1, figsize = (10,6))

    # Plot both populations using both solvers 
    ax.plot(time_euler, N1_euler, lw = 3, c = 'midnightblue')
    ax.plot(time_euler, N2_euler, lw = 3, c = 'darkred')
    ax.plot(time_rk8, N1_rk8, ls = '--', lw = 3, c = 'midnightblue')
    ax.plot(time_rk8, N2_rk8, ls = '--', lw = 3,c = 'darkred')

    # Apropriate x and y-axis labels
    ax.set_xlabel('Time (years)', fontsize = 12)
    ax.set_ylabel('Normalized Population Density', fontsize = 12)

    # Conditional labels and titles based on the type of relationship (competition v. predator/prey) being calculated
    if func == dNdt_comp: # If modeling a competition
        ax.legend(['Competitor 1, Euler', 'Competitor 2, Euler', 'Competitor 1, RK8', 'Competitor 2, RK8']) # Use these labels/title
        ax.set_title('Comparison of Euler and RK8 Solvers for Lotka-Volterra Competition Equations', fontsize = 15)
    if func == dNdt_pred_prey: # If modeling a predator/prey relationship
        ax.legend(['Prey, Euler', 'Predator, Euler', 'Prey, RK8', 'Predator, RK8']) # Use these labels/title
        ax.set_title('Comparison of Euler and RK8 Solvers for Lotka-Volterra Predator/Prey Equations', fontsize = 15)
    
    plt.grid() # Add a grid to the plots


def vary_params(func, a_max, b_max, c_max, d_max, a_step, b_step, c_step, d_step, N_step, N1_init = 0.6, N2_init = 0.3,
                dt = 0.1, t_final = 200, a = 1, b = 1, c = 1, d = 1):
    '''This function will utilize the RK8 solver above to see how varying the initial 
    population densities and equation coefficients impact the eventual fate of the populations.
    The function will print a figure containing 6 subplots to display this. If looking at a predator-prey
    relationship, phase portraits are also displayed.
    
    PARAMETERS
    ==========
    Values specified for N1_init, N2_init, a, b, c, d are used when that parameter is not being varied.
    E.g. when looping through a range of coefficient b, the value for b specified is *not* used.
    More information on those parameters below.

    func: function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
        Options are dNdt_comp or dNdt_pred_prey.
    
    a_max, b_max, c_max, d_max: float
        The maximum values of the coeffcients the user will allow when varying them.
    
    a_step, b_step, c_step, d_step: float
        The amount the coefficients will change when looping through them.
        
     N1_init, N2_init : float, defaults = 0.6, 0.3 (respectively)
        Initial conditions for `N1` and `N2`, ranging from (0,1]

    dt: float, default = 0.1
        Largest timestep allowed in years

    t_final : float, default = 200
        Integrate until this value is reached, in years.

    a, b, c, d : floats, default = 1 for all
        The base values of the Lotka-Volterra coefficients. This also serves as the default value when the coefficients
        are not being varied.
    '''

    if func == dNdt_comp: # If modeling a competition
        type = 'Competition' # add that to the figure title
    if func == dNdt_pred_prey: # If modeling a predator-prey relationship 
        type = "Predator-Prey" # add that to the figure title

    # Create a figure to plot all of the variations on. It will be 3x2 (6 subplots). Add apropriate labels/title.
    fig1, axes1 = plt.subplots(3, 2, figsize = (12, 18))
    fig1.suptitle(f"""Results of Varying Initial Conditions and Coefficients in Lotka-Volterra {type} Equations 
                  N1_init = {N1_init}, N2_init = {N2_init}, a = {a}, b = {b}, c = {c}, d = {d}""", 
                  fontsize = 20)
    fig1.supylabel("Normalized Population Density", fontsize = 17, x = 0.05, y = 0.5) # slightly modified label positions
    fig1.supxlabel("Time (years)", fontsize = 17, x = 0.5, y = 0.05)

    if func == dNdt_pred_prey: # If the user is looking at a predator/prey relationship
        fig2, axes2 = plt.subplots(3, 2, figsize = (12, 18))  # Create a figure specifically for phase diagrams
        fig2.suptitle(f"""Phase Portraits for Varying Initial Conditions and Coefficients in Lotka-Volterra 
                      Predator/Prey Equations (N1_init = {N1_init}, N2_init = {N2_init}, a = {a}, b = {b}, c = {c}, d = {d})""", 
                      fontsize = 20)
        fig2.supylabel("Predator Population Density", fontsize = 17, x = 0.05, y = 0.5) # slightly modified label positions
        fig2.supxlabel("Prey Population Density", fontsize = 17, x = 0.5, y = 0.05)

    for N1_initial in np.arange(0.1,1,N_step): # Loop through a range of N1 initial conditions
        time, N1, N2 = solve_rk8(func, N1_initial, N2_init, dt, t_final, a, b, c, d) # Call the solve_rk8 function

        alpha = 1  # full opacity
        label_N1 = f'N1 (Initial N1 = {N1_initial:2.1f})'  # Only apropriate labels
        label_N2 = f'N2 (Initial N1 = {N1_initial:2.1f})'

        if func == dNdt_comp: # If the user wants to model a competition
            if (np.any(N1 < 0.05)) | (np.any(N2 < 0.05)): # If any values of N1, N2 are near 0 (near collapse of a species)
                alpha = 0.25
                label_N1 = None  # Do not label the scenarios with population collapse
                label_N2 = None

        # Now plot N1, N2 as the loop goes on. Use f-strings for labels labels based parameter variation.
        axes1[0,0].plot(time, N1, lw = 2, alpha = alpha, label = label_N1)
        axes1[0,0].plot(time, N2, lw = 2, alpha = alpha, label = label_N2)

        #Set subplot title, legend, and add a grid
        axes1[0,0].set_title("Parameter Variation: Initial N1")
        axes1[0,0].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
        # make facecolor match style sheet
        axes1[0,0].grid(True)

        if func == dNdt_pred_prey: # If predator-prey relationship, plot the phase protrait
            axes2[0,0].plot(N1, N2, lw = 2, alpha = alpha, label = f'Initial N1 = {N1_initial:2.1f}')

            #Set subplot title, legend, and add a grid
            axes2[0,0].set_title("Parameter Variation: Initial N1")
            axes2[0,0].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
            # make facecolor match style sheet
            axes2[0,0].grid(True)
    
    for N2_initial in np.arange(0.1,1,N_step): # Loop through a range of N2 initial conditions
        time, N1, N2 = solve_rk8(func, N1_init, N2_initial, dt, t_final, a, b, c, d)

        alpha = 1  # full opacity
        label_N1 = f'N1 (Initial N2 = {N2_initial:2.1f})'  # Only apropriate labels
        label_N2 = f'N2 (Initial N2 = {N2_initial:2.1f})'

        if func == dNdt_comp: # If the user wants to model a competition
            if (np.any(N1 < 0.05)) | (np.any(N2 < 0.05)): # If any values of N1, N2 are near 0 (near collapse of a species)
                alpha = 0.25
                label_N1 = None  # Do not label the scenarios with population collapse
                label_N2 = None

        # Now plot N1, N2 as the loop goes on. Use f-strings for labels labels based parameter variation.
        axes1[0,1].plot(time, N1, lw = 2, alpha = alpha, label = label_N1)
        axes1[0,1].plot(time, N2, lw = 2, alpha = alpha, label = label_N2)

        #Set subplot title, legend, and add a grid
        axes1[0,1].set_title("Parameter Variation: Initial N2")
        axes1[0,1].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
        # make facecolor match style sheet
        axes1[0,1].grid(True)

        if func == dNdt_pred_prey: # If predator-prey relationship, plot the phase protrait
            axes2[0,1].plot(N1, N2, lw = 2, alpha = alpha, label = f'Initial N2 = {N2_initial:2.1f}')

            #Set subplot title, legend, and add a grid
            axes2[0,1].set_title("Parameter Variation: Initial N2")
            axes2[0,1].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
            # make facecolor match style sheet
            axes2[0,1].grid(True)
    
    for a_val in np.arange(1,a_max,a_step): # Loop through a range of coeffecient a (reproduction rate of first species)
        time, N1, N2 = solve_rk8(func, N1_init, N2_init, dt, t_final, a_val, b, c, d)

        alpha = 1  # full opacity
        label_N1 = f'N1 (a = {a_val:1.0f})'  # Only add the labels if the populations are stable
        label_N2 = f'N2 (a = {a_val:1.0f})'

        if func == dNdt_comp: # If the user wants to model a competition
            if (np.any(N1 < 0.05)) | (np.any(N2 < 0.05)): # If any values of N1, N2 are near 0 (near collapse of a species)
                alpha = 0.25
                label_N1 = None  # Only add the labels if the populations are stable
                label_N2 = None

        # Now plot N1, N2 as the loop goes on. Use f-strings for labels labels based parameter variation.
        axes1[1,0].plot(time, N1, lw = 2, alpha = alpha, label = label_N1)
        axes1[1,0].plot(time, N2, lw = 2, alpha = alpha, label = label_N2)

        #Set subplot title, legend, and add a grid
        axes1[1,0].set_title("Parameter Variation: Coefficient a")
        axes1[1,0].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
        # make facecolor match style sheet
        axes1[1,0].grid(True)

        if func == dNdt_pred_prey: # If predator-prey relationship, plot the phase protrait
            axes2[1,0].plot(N1, N2, lw = 2, alpha = alpha, label = f'a = {a_val:1.0f}')

            #Set subplot title, legend, and add a grid
            axes2[1,0].set_title("Parameter Variation: Coefficient a")
            axes2[1,0].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
            # make facecolor match style sheet
            axes2[1,0].grid(True)
    
    for b_val in np.arange(1,b_max,b_step): # Loop through a range of coeffecient b (impact of species 2 on species 1)
        time, N1, N2 = solve_rk8(func, N1_init, N2_init, dt, t_final, a, b_val, c, d)

        alpha = 1  # full opacity
        label_N1 = f'N1 (b = {b_val:1.0f})'  # Only add the labels if the populations are stable
        label_N2 = f'N2 (b = {b_val:1.0f})'

        if func == dNdt_comp: # If the user wants to model a competition
            if (np.any(N1 < 0.05)) | (np.any(N2 < 0.05)): # If any values of N1, N2 are near 0 (near collapse of a species)
                alpha = 0.25
                label_N1 = None  # Only add the labels if the populations are stable
                label_N2 = None

        # Now plot N1, N2 as the loop goes on. Use f-strings for labels labels based parameter variation.
        axes1[1,1].plot(time, N1, lw = 2, alpha = alpha, label = label_N1)
        axes1[1,1].plot(time, N2, lw = 2, alpha = alpha, label = label_N2)
        #Set subplot title
        #Set subplot title, legend, and add a grid
        axes1[1,1].set_title("Parameter Variation: Coefficient b")
        axes1[1,1].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
        # make facecolor match style sheet
        axes1[1,1].grid(True)

        if func == dNdt_pred_prey: # If predator-prey relationship, plot the phase protrait
            axes2[1,1].plot(N1, N2, lw = 2, alpha = alpha, label = f'b = {b_val:1.0f}')

            #Set subplot title, legend, and add a grid
            axes2[1,1].set_title("Parameter Variation: Coefficient b")
            axes2[1,1].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
            # make facecolor match style sheet
            axes2[1,1].grid(True)
    
    for c_val in np.arange(1,c_max,c_step): # Loop through a range of coeffecient c (reproduction rate of second species)
        time, N1, N2 = solve_rk8(func, N1_init, N2_init, dt, t_final, a, b, c_val, d)

        alpha = 1  # full opacity
        label_N1 = f'N1 (c = {c_val:1.0f})'  # Only add the labels if the populations are stable
        label_N2 = f'N2 (c = {c_val:1.0f})'

        if func == dNdt_comp: # If the user wants to model a competition
            if (np.any(N1 < 0.05)) | (np.any(N2 < 0.05)): # If any values of N1, N2 are near 0 (near collapse of a species)
                alpha = 0.25
                label_N1 = None  # Only add the labels if the populations are stable
                label_N2 = None

        # Now plot N1, N2 as the loop goes on. Use f-strings for labels labels based parameter variation.
        axes1[2,0].plot(time, N1, lw = 2, alpha = alpha, label = label_N1)
        axes1[2,0].plot(time, N2, lw = 2, alpha = alpha, label = label_N2)
        
        #Set subplot title, legend, and add a grid
        axes1[2,0].set_title("Parameter Variation: Coefficient c")
        axes1[2,0].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
        # make facecolor match style sheet
        axes1[2,0].grid(True)

        if func == dNdt_pred_prey: # If predator-prey relationship, plot the phase protrait
            axes2[2,0].plot(N1, N2, lw = 2, alpha = alpha, label = f'c = {c_val:1.0f}')

            #Set subplot title, legend, and add a grid
            axes2[2,0].set_title("Parameter Variation: Coefficient c")
            axes2[2,0].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
            # make facecolor match style sheet
            axes2[2,0].grid(True)
    
    for d_val in np.arange(1,d_max,d_step): # Loop through a range of coeffecient d (impact of species 1 on species 2)
        time, N1, N2 = solve_rk8(func, N1_init, N2_init, dt, t_final, a, b, c, d_val)

        alpha = 1  # full opacity
        label_N1 = f'N1 (d = {d_val:1.0f})'  # Only add the labels if the populations are stable
        label_N2 = f'N2 (d = {d_val:1.0f})'

        if func == dNdt_comp: # If the user wants to model a competition
            if (np.any(N1 < 0.05)) | (np.any(N2 < 0.05)): # If any values of N1, N2 are near 0 (near collapse of a species)
                alpha = 0.25
                label_N1 = None  # Only add the labels if the populations are stable
                label_N2 = None
        # Now plot N1, N2 as the loop goes on. Use f-strings for labels labels based parameter variation.
        axes1[2,1].plot(time, N1, lw = 2, alpha = alpha, label = label_N1)
        axes1[2,1].plot(time, N2, lw = 2, alpha = alpha, label = label_N2)

        #Set subplot title, legend, and add a grid
        axes1[2,1].set_title("Parameter Variation: Coefficient d")
        axes1[2,1].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
        # make facecolor match style sheet
        axes1[2,1].grid(True)

        if func == dNdt_pred_prey: # If predator-prey relationship, plot the phase protrait
            axes2[2,1].plot(N1, N2, lw = 2, alpha = alpha, label = f'a = {a_val:1.0f}')

            #Set subplot title, legend, and add a grid
            axes2[2,1].set_title("Parameter Variation: Coefficient d")
            axes2[2,1].legend(loc = 'upper right', frameon = True, facecolor = '#eaeaf2') # Turn on frame, 
            # make facecolor match style sheet
            axes2[2,1].grid(True)
