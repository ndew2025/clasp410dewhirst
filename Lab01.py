#!/usr/bin/env python3

# Import necessary Python libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use('seaborn-v0_8-dark') # Use a style sheet
plt.ion() # Turn on interactive mode

'''This file contains 5 functions. One will model the basic spread of forest fires and disease.
The other 4 will utilize the first function to visualize how varying probabilities key to the spread
of forest fires and diseases affect their evolution and aftermath.'''

# Integers for a forest fire simulation
# 1 = Bare/burnt
# 2 = Forested
# 3 = Burning

# Integers for a disease simulation
# 0 = Dead
# 1 = Immune
# 2 = Healthy
# 3 = Infected

def idealized_spread(n_vert, n_horz, p_bare, p_start, p_spread, center_start = True, 
                     disease = False, p_fatal = None, print_plots = True):

    '''
    This function will serve to create an idealized model of anything that can spread, 
    namely forest fires and diseases, on a 2D grid of any shape.

    Parameters
    ==========

     n_vert: Integer
            the number of grid cells in the vertical (rows)

    n_horz: Integer
            the number of grid cells in the horizontal (columns)

    p_bare: float
            the probability that a grid cell in the forest will be bare or a person
            will be immune at the start of a simulation

    p_start: float
            the probability that fire/disease will begin in a grid cell at the start of a simulation

    p_spread: float
            the probability that fire/disease will spread to a cell adjacent to one that is burning/infected.

    center_start: boolean
            this kwarg is used to specify when the user wants to replicate the special cases in 
            question 1 of the lab. By default it is true and will only have fire/disease initially at the center
            of the grid. If center_start = False, then the random probability assignment is used.

    disease: boolean
            indicates what kind of spread is being modeled. By default disease is false, so a forest fire is simulated. 
            If true, a disease will be simulated. No key components of the algorithm change except for plot labels, 
            colors, and a fatality probability.

    p_fatal: NoneType when fire = true, float when fire = false
            the probability that the disease kills a person. Assignment of a probability has no effect when fire is 
            true, but must be specified when fire = false.
    
    print_plots: boolean
            indicates whether the user wants to print the plots directly to screen. Default is true. If false,
            the function will save the figures.
    '''

    # Setup the maximum number of iterations the simulation could take (within reason)
    max_iterations = 1000

    # Create an array of twos to setup the grid for the simulation. This represents a grid of a forest or
    # a healthy population. Dtype needs to be consistent (integers for indexing)
    grid = np.zeros((max_iterations, n_vert, n_horz), dtype = int) + 2

    if center_start is True: # If the user wants the center to begin the simulation
        grid[0, n_vert//2, n_horz//2] = 3 # Set the center cell to be burning initially using floor division

    if center_start is False: # If the user doesn't want the center to begin the simulation
        # Create an array of probabilities that matches the shape of the grid
        # If the random probability in a cell is less than p_bare, that cell is bare/immune in the first iteration.
        bare_or_immune = np.random.rand(n_vert, n_horz) < p_bare

        # Update the forest with the randomly generated bare/immune cells
        grid[0, bare_or_immune] = 1

        # If the random probability in a cell is less than p_start, it's on fire/infected in the first iteration.
        burning_or_infected = np.random.rand(n_vert, n_horz) < p_start

        # Update the forest with the randomly generated burning cells
        grid[0, burning_or_infected] = 3

    # Create a figure for the initial conditions based on the operations performed above.
    fig, ax = plt.subplots(1,1)
    
    if disease is False: # If a forest fire is being modeled
        cmap = ListedColormap(['wheat', 'forestgreen', 'firebrick']) # Colors for a forest fire
        # Assign those colors to the values in my plot, only from 1 to 3
        contour = ax.pcolor(grid[0,:,:], cmap=cmap, vmin=1, vmax=3) 
        cbar = plt.colorbar(contour, ax=ax, pad = 0.02) # Create a colorbar that matches the contours
        cbar.set_ticks(ticks=[1, 2, 3], labels=['Bare', 'Forested', 'Burning']) # cbar labels that make sense
        ax.set_title('Forest Fire Iteration 0') # Indicate that this is the first (0th) iteration in time
    
    if disease is True: # If a disease is being modeled
        cmap = ListedColormap(['black', 'mediumblue', 'lightgray', 'darkred']) # Colors for a disease
        # Assign those colors to the values in my plot, from 0 to 3
        contour = ax.pcolor(grid[0,:,:], cmap=cmap, vmin=0, vmax=3) 
        cbar = plt.colorbar(contour, ax=ax, pad = 0.02) # Create a colorbar that matches the contours
        cbar.set_ticks(ticks=[0, 1, 2, 3], labels=['Dead', 'Immune', 'Healthy', 'Infected']) # cbar labels that make sense
        ax.set_title('Pandemic Iteration 0') # Indicate that this is the first (0th) iteration in time

    ax.set_xlabel('X') # x-axis label
    ax.set_ylabel('Y') # y-axis label

    for k in range(max_iterations - 1): # Repeat the process below as many times as needed
        grid[k+1,:,:] = grid[k,:,:] # Update the next iteration to the state of the current

        #Use a set of nested loops to model the spread
        for i in range(n_vert): # loop through each row
            for j in range(n_horz): # loop through each column
                if grid[k, i, j] == 3: # If a specific cell at k,i,j is on fire/infected

                    if i+1 < grid.shape[1]: # If a cell to the south exists
                        if grid[k, i+1, j] == 2: # and is forested/healthy
                            if np.random.rand() < p_spread: # and satisfies the probability to spread
                                grid[k+1, i+1, j] = 3 # It is on fire/infected in the next iteration

                    if i-1 >= 0: # If a cell to the north exists
                        if grid[k, i-1, j] == 2: # and is forested/healthy
                            if np.random.rand() < p_spread: # and satisfies the probability to spread
                                grid[k+1, i-1, j] = 3 # It is on fire/infected in the next iteration

                    if j+1 < grid.shape[2]: # If a cell to the east exists
                        if grid[k, i, j+1] == 2: # and is forested/healthy
                            if np.random.rand() < p_spread: # and satisfies the probability to spread
                                grid[k+1, i, j+1] = 3 # It is on fire/infected in the next iteration

                    if j-1 >= 0: # If a cell to the west exists
                        if grid[k, i, j-1] == 2: # and is forested/healthy
                            if np.random.rand() < p_spread: # and satisfies the probability to spread
                                grid[k+1, i, j-1] = 3 # It is on fire/infected in the next iteration
                    
        #Set currently burning to bare
        was_burning_or_infected = grid[k,:,:] == 3 # Find cells that were burning in this iteration
        grid[k+1, was_burning_or_infected] = 1 # Now those cells that were burning are bare in the next

        if disease is True: # If a disease is being modeled
            potentially_dead = np.random.rand(n_vert, n_horz) # Create a randomly generated array of probabilites
            # for the potential that people could die from this disease                                                                    
            potentially_dead[was_burning_or_infected == False] = 1 # If a person was healthy, they're safe (1 = 100%).
            dead = potentially_dead < p_fatal # A person dies when the probability is less than p_fatal
            grid[k+1, dead] = 0 # The dead people are then assigned the value of 0

        fig, ax = plt.subplots(1,1)

        if disease is False: # If a forest fire is being modeled
            cmap = ListedColormap(['wheat', 'forestgreen', 'firebrick']) # Colors for a forest fire
            # Assign those colors to the values in my plot, only from 1 to 3
            contour = ax.pcolor(grid[k+1,:,:], cmap=cmap, vmin=1, vmax=3) 
            cbar = plt.colorbar(contour, ax=ax, pad = 0.02) # Create a colorbar that matches the contours
            cbar.set_ticks(ticks=[1, 2, 3], labels=['Bare', 'Forested', 'Burning']) # cbar labels that make sense
            ax.set_title(f'Forest Fire Iteration {k+1}') # Indicate that this is the first (0th) iteration in time
    
        if disease is True: # If a disease is being modeled
            cmap = ListedColormap(['black', 'mediumblue', 'lightgray', 'darkred']) # Colors for a disease
            # Assign those colors to the values in my plot, from 0 to 3
            contour = ax.pcolor(grid[k+1,:,:], cmap=cmap, vmin=0, vmax=3) 
            cbar = plt.colorbar(contour, ax=ax, pad = 0.02) # Create a colorbar that matches the contours
            cbar.set_ticks(ticks=[0, 1, 2, 3], labels=['Dead', 'Immune', 'Healthy', 'Infected']) # cbar labels that make sense
            ax.set_title(f'Pandemic Iteration {k+1}') # Indicate that this is the first (0th) iteration in time
 
        # Label x and y axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        if print_plots is True:
            plt.show()

        if print_plots is False: # If the user doesn't want the plots to print to screen
            plt.ioff() # Turn off interactive mode
            #fig.savefig(f'Figure {k+1}') # Save them
            plt.close('all') # If any figures are there, close all of them

        # After updating the next iteration and making a plot for it, 
        # check if there are any burning/infected cells in the grid. If not, exit all loops.
        if not np.any(grid[k+1, :, :] == 3):
            break

    grid = grid[0:k+1, :, :] # Update the grid so that it doesn't have extra steps in time

    return grid # Return the grid so that I can access it and do further analysis


def vary_p_bare_forest(n_vert, n_horz, p_start, p_spread):
    '''This function will utilize the idealized_spread() function to see how the spread of wildfire
    depends on how dense the forest is from the start. (controlled by p_bare). 
    It will return a plot to show how the remaining forest and duration of the fire changes by varying p_bare.
    
    Parameters
    ==========

    n_vert: integer
        the size of the desired forest in the vertical

    n_horz: integer
        the size of the desired forest in the horizontal

    p_bare: array
        an array of probabilities to use to vary p_bare in indealized_spread()
    
    p_spread: float
        the probability (fraction) of cells that are initially on fire in the forest
    '''
    # Range of p_bare to use 
    p_bare = np.arange(0,1.1,0.1)

    # Setup empty lists to store k, the number of iterations per simulation, percentage of the forest 
    # that is bare to start, and how much untouched land remains. This is for Q2 in the lab.
    bare_cells = []
    num_iters = []
    untouched = []

    for prob in p_bare: # Loop through the probabilities provided by the p_bare array 
        # Call the first function for a simulation
        grid = idealized_spread(n_vert, n_horz, prob, p_start, p_spread, center_start = False, print_plots = False) 

        bare_cells_num = np.sum(grid[0, :, :] == 1) # Check the number of bare cells at the start of the simulation
        bare_percentage = bare_cells_num/(n_vert*n_horz)*100 # Calculate the percentage of bare cells
        bare_cells.append(bare_percentage) # Append percentage of bare cells to the empty list

        # Append the number of iterations at the end of each simulation to the num_iters empty list
        num_iters.append(grid.shape[0]) 

        remaining_forest = np.sum(grid[-1, :, :] == 2) # Check for any forested cells at the end of each simulation
        remaining_percentage = remaining_forest/(n_vert*n_horz)*100 # Calculate the percentage of forest at the end
        untouched.append(remaining_percentage + bare_percentage) # Append the total untouched land to the respective list

    fig, ax = plt.subplots(1,1) # Create a figure for the plot
    ax.plot(bare_cells, num_iters, c = 'midnightblue', lw = 4) # Plot bare cell % against iterations
    ax1 = ax.twinx() # Set a twin axis to make both lines appear on the same subplot
    ax1.plot(bare_cells, untouched, c = 'forestgreen', lw = 4) # Plot bare cell % against remaining forest %

    # Set labels and title for the plot
    ax.set_title("Effects of Forest Density on Fire Evolution", fontsize = 16)
    ax.set_xlabel('Initially Bare Cells (%)', fontsize = 12)
    ax.set_ylabel('Number of Iterations', fontsize = 12)
    ax1.set_ylabel('Remaining Untouched Land (%)', fontsize = 12)

    # Set the color of axis ticks and labels
    ax.tick_params(axis = 'y', colors = 'midnightblue')  
    ax1.tick_params(axis = 'y', colors = 'forestgreen')

    # Set the color of axis labels
    ax.yaxis.label.set_color('midnightblue')
    ax1.yaxis.label.set_color('forestgreen')

    fig.savefig("varying_pbare_analysis.png") # Save the figure


def vary_p_spread_forest(n_vert, n_horz, p_bare, p_start):
    '''This function will utilize the idealized_spread() function to see how the spread of wildfire
    depends on how likely it is to spread. (controlled by p_spread). 
    It will return a plots to show how the remaining forest and duration of the fire changes by varying p_spread.
    
    Parameters
    ==========

    n_vert: integer
        the size of the desired forest in the vertical

    n_horz: integer
        the size of the desired forest in the horizontal

    p_bare: array
        the probability (fraction) of cells that are bare at the beginning of a forest fire simulation
    
    p_spread: array
        an array of probabilities to use to vary p_spread in indealized_spread()
    '''
    # Range of p_bare to use 
    p_spread = np.arange(0,1.1,0.1)

    # Setup empty lists to store k, the number of iterations per simulation and how much forest remaining. 
    # This is for Q2 in the lab.
    num_iters = []
    forest_percentage = []

    for prob in p_spread: # Loop through the probabilities provided by the p_bare array 
        # Call the first function for a simulation
        grid = idealized_spread(n_vert, n_horz, p_bare, p_start, prob, center_start = False, print_plots = False) 

        # Append the number of iterations at the end of each simulation to the num_iters empty list
        num_iters.append(grid.shape[0]) 

        remaining_forest = np.sum(grid[-1, :, :] == 2) # Check for any forested cells at the end of each simulation
        remaining_percentage = remaining_forest/(n_vert*n_horz)*100 # Calculate the percentage of forest at the end
        forest_percentage.append(remaining_percentage) # Append the values to the respective list

    fig, ax = plt.subplots(1,1) # Create a figure for the plot
    ax.plot(p_spread*100, num_iters, c = 'midnightblue', lw = 4) # Plot p_spread against iterations
    ax1 = ax.twinx() # Set a twin axis to make both lines appear on the same subplot
    ax1.plot(p_spread*100, forest_percentage, c = 'forestgreen', lw = 4) # Plot p_spread against remaining forest %

    # Set labels and title for the plot
    ax.set_title("Effects of Burning Conditions on Fire Evolution", fontsize = 16)
    ax.set_xlabel('Fire Spread Chance (%)', fontsize = 12)
    ax.set_ylabel('Number of Iterations', fontsize = 12)
    ax1.set_ylabel('Remaining Forest (%)', fontsize = 12)

    # Set the color of axis ticks and labels
    ax.tick_params(axis = 'y', colors = 'midnightblue')  
    ax1.tick_params(axis = 'y', colors = 'forestgreen')

    # Set the color of axis labels
    ax.yaxis.label.set_color('midnightblue')
    ax1.yaxis.label.set_color('forestgreen')

    fig.savefig("varying_pspread_analysis.png") # Save the figure


def vary_p_bare_population(n_vert, n_horz, p_start, p_spread, p_fatal):
    '''This function will utilize the idealized_spread() function to see how the spread of disease
    depends on how immune the population is from the start. (controlled by p_bare). 
    It will return a plot to show how the remaining healthy population and duration of the pan/epidemic
    changes by varying p_bare/vaccination rate.
    
    Parameters
    ==========

    n_vert: integer
        the size of the population in the vertical

    n_horz: integer
        the size of the population in the horizontal

    p_bare: array
        an array of probabilities to use to vary p_bare in indealized_spread()
    
    p_spread: float
        the probability (fraction) of people that are initially immune/vaccinated

    p_fatal: float
        the probability that someone dies from the disease
    '''
    # Range of p_bare to use 
    p_bare = np.arange(0,1.1,0.1)

    # Setup empty lists to store k, the number of iterations per simulation, percentage of the population 
    # that is vaccinated to start, and how much healthy population remains. This is for Q3 in the lab.
    vaccinated = []
    num_iters = []
    healthy = []

    for prob in p_bare: # Loop through the probabilities provided by the p_bare array 
        # Call the first function for a simulation
        grid = idealized_spread(n_vert, n_horz, prob, p_start, p_spread, center_start = False, disease = True,
                                p_fatal = p_fatal, print_plots = False) 

        vaccinated_num = np.sum(grid[0, :, :] == 1) # Check the number of immune at the start of the simulation
        vaccinated_percentage = vaccinated_num/(n_vert*n_horz)*100 # Calculate the percentage of immune
        vaccinated.append(vaccinated_percentage) # Append percentage of immune to the empty list

        # Append the number of iterations at the end of each simulation to the num_iters empty list
        num_iters.append(grid.shape[0]) 

        remaining_healthy = np.sum(grid[-1, :, :] == 2) # Check for any healthy people at the end of each simulation
        healthy_percentage = remaining_healthy/(n_vert*n_horz)*100 # Calculate the percentage of healthy at the end
        healthy.append(healthy_percentage + vaccinated_percentage) # Append total healthy population to the list

    fig, ax = plt.subplots(1,1) # Create a figure for the plot
    ax.plot(vaccinated, num_iters, c = 'midnightblue', lw = 4) # Plot bare cell % against iterations
    ax1 = ax.twinx() # Set a twin axis to make both lines appear on the same subplot
    ax1.plot(vaccinated, healthy, c = 'peru', lw = 4) # Plot bare cell % against remaining forest %

    # Set labels and title for the plot
    ax.set_title("Effects of Vaccination on Disease Evolution", fontsize = 16)
    ax.set_xlabel('Initially Vaccinated Population (%)', fontsize = 12)
    ax.set_ylabel('Number of Iterations', fontsize = 12)
    ax1.set_ylabel('Remaining Healthy Population (%)', fontsize = 12)

    # Set the color of axis ticks and labels
    ax.tick_params(axis = 'y', colors = 'midnightblue')  
    ax1.tick_params(axis = 'y', colors = 'peru')

    # Set the color of axis labels
    ax.yaxis.label.set_color('midnightblue')
    ax1.yaxis.label.set_color('peru')

    fig.savefig("varying_pbare_disease_analysis.png") # Save the figure


def vary_p_fatal(n_vert, n_horz, p_bare, p_start, p_spread):
    '''This function will utilize the idealized_spread() function to see how the spread of disease
    depends on how fatal it is. (controlled by p_fatal). 
    It will return a plot to show how the remaining healthy population and duration of the pan/epidemic
    changes by varying the fatality rate.
    
    Parameters
    ==========

    n_vert: integer
        the size of the population in the vertical

    n_horz: integer
        the size of the population in the horizontal

    p_bare: float
        the probability (fraction) of vaccinated/immune people from the start
    
    p_start: float
        the probability that a disease infects a person from the start
    
    p_spread: float
        the probability that the disease spreads to adjacent people

    '''
    # Range of p_fatal to use 
    p_fatal = np.arange(0,1.1,0.1)

    # Setup empty lists to store k, the number of iterations per simulation, percentage of the population 
    # that dies due to the disease, and how much healthy population remains. This is for Q3 in the lab.
    dead = []
    num_iters = []
    healthy = []

    for prob in p_fatal: # Loop through the probabilities provided by the p_fatal array 
        # Call the first function for a simulation
        grid = idealized_spread(n_vert, n_horz, p_bare, p_start, p_spread, center_start = False, disease = True,
                                p_fatal = prob, print_plots = False) 

        dead_num = np.sum(grid[-1, :, :] == 0) # Check the number of dead at the end of the simulation
        dead_percentage = dead_num/(n_vert*n_horz)*100 # Calculate the percentage of dead
        dead.append(dead_percentage) # Append percentage of dead to the empty list

        vaccinated_num = np.sum(grid[0, :, :] == 1) # Check the number of immune at the start of the simulation
        vaccinated_percentage = vaccinated_num/(n_vert*n_horz)*100 # Calculate the percentage of immune

        # Append the number of iterations at the end of each simulation to the num_iters empty list
        num_iters.append(grid.shape[0]) 

        remaining_healthy = np.sum(grid[-1, :, :] == 2) # Check for any healthy people at the end of each simulation
        healthy_percentage = remaining_healthy/(n_vert*n_horz)*100 # Calculate the percentage of healthy at the end
        healthy.append(healthy_percentage + vaccinated_percentage) # Append the total healthy population to the list

    fig, ax = plt.subplots(1,1) # Create a figure for the plot
    ax.plot(dead, num_iters, c = 'midnightblue', lw = 4) # Plot bare cell % against iterations
    ax1 = ax.twinx() # Set a twin axis to make both lines appear on the same subplot
    ax1.plot(dead, healthy, c = 'peru', lw = 4) # Plot bare cell % against remaining forest %

    # Set labels and title for the plot
    ax.set_title("Effects of Fatality Rate on Disease Evolution", fontsize = 16)
    ax.set_xlabel('Final Dead Population (%)', fontsize = 12)
    ax.set_ylabel('Number of Iterations', fontsize = 12)
    ax1.set_ylabel('Remaining Healthy Population (%)', fontsize = 12)

    # Set the color of axis ticks and labels
    ax.tick_params(axis = 'y', colors = 'midnightblue')  
    ax1.tick_params(axis = 'y', colors = 'peru')

    # Set the color of axis labels
    ax.yaxis.label.set_color('midnightblue')
    ax1.yaxis.label.set_color('peru')

    fig.savefig("varying_pfatal_disease_analysis.png") # Save the figure