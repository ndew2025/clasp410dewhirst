#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.ion()  # turn on interactive mode
plt.style.use('seaborn-v0_8-dark')  # Use a style sheet

'''This file will be used to solve the diffusion equation as it relates to heated wires and permafrost.
There are 4 functions. The first generates a time series of surface temperatures in Kangerlussuaq, Greenland
to be used as a boundary condition for diffusion. The seconds creates a model of how diffusion evolves the
distribution of temperature in a wire or in the ground over time. The third plots the resulting array from this 
model, and the fourth extracts temperature profiles from a permafrost scenario for comparison puposes.
'''


def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland. 

    PARAMETERS
    ==========

    t: 1D array
        an array of times representing the length of time in the U array.
        This will be used to create a time series of seasonal surface temperatures
        in Greenland.

    RETURNS
    =======

    temp_series: 1D array
        the array of surface temperatures to use for boundary conditions in the permafrost model'''

    # Kangerlussuaq monthly average temperatures:
    t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])

    t_amp = (t_kanger - t_kanger.mean()).max()  # amplitude of the temperature oscillation
    temp_series = t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

    return temp_series

##############################


def heat_diff_solve(xmax, tmax, dx, dt, c2=1, permafrost=False, climate_shift=0):
    '''The function will solve the diffusion equation as it applies to the distribution of heat
    on a wire and through permafrost in the ground. This function is called within the two plotting functions.

    PARAMETERS
    ==========

    xmax: float
        The length of the wire or depth of the ground to model diffusion through (in meters)

    tmax: float
        The length of time to model the diffusion over. The units are in seconds for a wire by default
        (when permaforst = False), but if permafrost = True, tmax is in days, as heat diffusion through the 
        ground is more apropriate over a larger time than seconds.

    dx: float
        The step in x, in meters, to move through the array for diffusion and adjust resolution

    dt: float
        The step in time to move forward through the the array for diffusion and adjust resolution.
        Same units as tmax, depending on scenario.

    c2: float
        The thermal diffusivity of the material, in m^2/s for a wire. When permafrost = True, the function
        internally calculates a different thermal diffusivity in units m^2/day for better application.

    permafrost: boolean
        Used to indicate the scenario to apply diffusion to. If False (default) then diffusion through a wire
        is modeled. If True, a permafrost model is created.

    climate_shift: float
        The increase or decrease in temperature to apply to the top boundary conditions in the permafrost model.
        This is used to model how the permafrost layer could respond to climate change.
        Units are in degrees Celsius.

    RETURNS
    =======

    xgrid: 1D array
        the array of values in the x direction to model diffusion over

    tgrid: 1D array
        the array of times to model diffusion over

    U: 2D array
        The solution to the diffusion equation over xgrid and tgrid
    '''
    # conversion factors to change c2 for permafrost
    day_to_sec = 60*60*24  # number of seconds in a day
    mm_to_m = 1/1000  # fraction of a meter in a millimeter

    if permafrost:  # If modeling permafrost
        c2 = 0.25*day_to_sec*mm_to_m**2  # convert c2 in mm^2/s to m^2/day for easier application

    # Stability check for the forward difference method
    if dt > dx**2/(2*c2):  # if the timestep is greater than that
        raise ValueError("dt too large! Cannot compute.")  # spit an error, try a smaller timestep.

    # Calculate dimensions of the U array
    M = int(round(xmax / dx + 1))
    N = int(round(tmax / dt + 1))

    # Ranges for x and t in values of meters and seconds (days for permafrost)
    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    # Initialize U array with M and N
    U = np.zeros((M, N))

    if permafrost:  # If modeling permafrost
        sfc_temps = temp_kanger(tgrid) + climate_shift
        U[-1, :] = 5  # Bottom row is 5C due to geothermal energy
        for i, temp in enumerate(sfc_temps):  # Set the surface temperatures based on the sine function output
            U[0, i] = temp

    else:  # modeling the wire
        # Set initial conditions
        U[:, 0] = 4*xgrid - 4*xgrid**2  # initial temperature distribution along wire

        # Set boundary conditions
        U[0, :] = 0  # ice cubes keeping the ends of wire at freezing
        U[-1, :] = 0

    r = c2*dt/dx**2  # r coefficient for below

    for j in range(N-1):  # diffusion based on forward difference method
        U[1:-1, j+1] = (1-2*r)*U[1:-1, j] + r*(U[2:, j] + U[:-2, j])

    return xgrid, tgrid, U  # return the range of x, t, and the diffusion solution

##############################


def plot_diffusion(figure_name, xmax, tmax, dx, dt, c2=1, permafrost=False, hline=False, print_U=False):
    '''The function will plot the U array from heat_diff_solve() in a heatmap to see how the ground
    temperature profile evolves from initial conditions due to diffusion over time.

    PARAMETERS
    ==========

    xmax: float
        The length of the wire or depth of the ground to model diffusion through (in meters)

    tmax: float
        The length of time to model the diffusion over. The units are in seconds for a wire by default
        (when permaforst = False), but if permafrost = True, tmax is in days, as heat diffusion through the 
        ground is more apropriate over a larger time than seconds.

    dx: float
        The step in x, in meters, to move through the array for diffusion and adjust resolution

    dt: float
        The step in time to move forward through the the array for diffusion and adjust resolution.
        Same units as tmax, depending on scenario.

    c2: float
        The thermal diffusivity of the material, in m^2/s for a wire. When permafrost = True, the function
        internally calculates a different thermal diffusivity in units m^2/day for better application.

    permafrost: boolean
        Used to indicate the scenario to apply diffusion to. If False (default) then diffusion through a wire
        is modeled. If True, a permafrost model is created.

    hline: boolean
        Used to plot a vertical line at 50m depth on the heatmap if permafrost is true. Default is false.

    print_U: boolean
        Used to print the U array to verify are as they should. Default is false and won't print the array.
    '''

    # Call the forward-difference solver to create the U array
    xgrid, tgrid, U = heat_diff_solve(xmax, tmax, dx, dt, permafrost=permafrost)

    if print_U:  # if you want to print the array
        print(U)  # print it

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))  # create a figure for the U array heatmap

    if permafrost:  # if modeling permafrost
        fill = ax.pcolor(tgrid/365, xgrid, U, cmap='seismic', norm=colors.CenteredNorm())  # data from the array
        cbar = plt.colorbar(fill, pad=0.01)  # colorbar
        cbar.set_label("Temperature (\u00B0C)", fontsize=15)  # colorbar label

        # title and labels
        ax.set_title("Ground Temperatures: Kangerlussuaq, Greenland", fontsize=22)
        ax.set_xlabel("Time (years)", fontsize=14)
        ax.set_ylabel("Depth (meters)", fontsize=14)
        ax.invert_yaxis()  # invert y axis for increasing depth down into the ground

        if hline:  # if you want a horizontal line
            ax.axhline(50, c='black', ls='--', label="50m depth")  # plot one
            ax.legend(loc=[0.01, 0.51], frameon=True, fontsize=13)  # legend for it

    else:  # besides permafrost (wire model)
        fill = ax.pcolor(tgrid, xgrid, U, cmap='hot')  # data from the array
        cbar = plt.colorbar(fill, pad=0.01)  # colorbar
        cbar.set_label("Temperature (\u00B0C)", fontsize=15)  # colorbar label

        # title and labels
        ax.set_title("Wire Temperatures over time", fontsize=22)
        ax.set_xlabel("Time (seconds)", fontsize=15)
        ax.set_ylabel("Length of Wire (meters)", fontsize=15)

    ax.tick_params(labelsize=14)  # modify size of tick and colorbar labels
    cbar.ax.tick_params(labelsize=14)
    # the plot is not showing up on my end, so I added the line below to save the plot
    fig.savefig(figure_name)

##############################


def plot_temp_profiles(figure_name, xmax=100, tmax=73000, dx=0.25, dt=1, climate_shift=0):
    ''' This function will plot peak winter and summer ground temperature profiles when modeling
    permafrost heat diffusion. Default is after 200 years of diffusion in an approximate steady-state.
    It uses data from the U array calculated for Greenland in heat_diff_solve()

    PARAMETERS
    ==========

    xmax: float
        The depth of the ground to model diffusion through (in meters)

    tmax: float
        The length of time to model the diffusion over. Units are in days.

    dx: float
        The step in x, in meters, to move through the array for diffusion and adjust resolution

    dt: float
        The step in time to move forward through the the array for diffusion and adjust resolution.
        Same units as tmax.

    climate_shift: float
        The increase or decrease in temperature to apply to the top boundary conditions in the permafrost model.
        This is used to model how the permafrost layer could respond to climate change.
        Units are in degrees Celsius.'''

    # Call the forward-difference solver to model ground heat diffusion
    xgrid, tgrid, U = heat_diff_solve(xmax, tmax, dx, dt, permafrost=True, climate_shift=climate_shift)

    # Set indexing for the final year of results:
    loc = int(-365/dt)  # Final 365 days of the result.

    # Extract slices including min and max temps over the final year (peak winter and summer).
    winter = U[:, loc:].min(axis=1)
    summer = U[:, loc:].max(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))  # create a figure for the profiles
    ax.plot(winter, xgrid, c='mediumblue', lw=3, label='Winter', zorder=3)  # plot winter
    ax.plot(summer, xgrid, c='red', lw=3, label='Summer', zorder=3)  # plot summer
    ax.invert_yaxis()  # invert y axis for increasing depth into the ground

    limits = ax.set_xlim([winter.min() - 2, summer.max() + 2])  # set limits based on temperature range

    if np.any(summer < 0):  # If any of the peak summer profile is below 0, there is permafrost

        # Sort the temperatures in the top 10 meters to find the top of the permafrost
        # The top is where the temp is nearest to 0C in the top 10 meters
        zero_line_top = np.argsort(np.abs(summer[:40]))[0]

        # Sort the temperatures under 10 meters for the base of the permafrost
        # the base is where the temp is nearest 0C below the top
        zero_line_bottom = np.argsort(np.abs(summer[40:]))[0] + 40  # +40 because of the relative indices

        perm_top = xgrid[zero_line_top]  # get depth of the top
        perm_base = xgrid[zero_line_bottom]  # get depth of the base
        perm_depth = perm_base - perm_top  # extent of the permafrost is base - top

        # horizontal lines for the top and base of the permafrost
        ax.axhline(perm_top, c='deepskyblue', lw=1.5, zorder=1)  # plot a line for the top
        ax.axhline(perm_base, c='deepskyblue', lw=1.5, zorder=1)  # plot a line for the base

        # Fill the area between the top and bottom to shade the permafrost layer
        ax.fill_between(limits, perm_top, perm_base, color='deepskyblue', alpha=0.5,  # fill between the lines
                        label=f"Permafrost: top = {round(perm_top, 1)}m, base = {round(perm_base, 1)}m, "
                        f"thickness = {round(perm_depth, 1)}m")

        # horizontal line at the surface to mark the top of the active layer
        ax.axhline(0, c='tan', lw=1.5, zorder=1)

        # Shade in the active layer
        ax.fill_between(limits, 0, perm_top, color='tan', alpha=0.5,
                        label=f'Active Layer: thickness = {round(perm_top, 1)}m')

    # vertical line at 0C because of its important for permafrost
    ax.axvline(0, c='black', ls='--', label='$0\u00B0C$ isotherm', zorder=2)  # vertical line for 0 degC

    # title, labels, legend, x-axis limits, grid
    ax.set_xlabel("Temperature (\u00B0C)", fontsize=15)
    ax.set_ylabel("Depth (meters)", fontsize=15)
    ax.set_title("Summer and Winter Ground Temperatures at Kangerlussuaq, Greenland", fontsize=20)
    ax.legend(loc='lower left', frameon=True, fontsize=14)
    ax.set_xlim([winter.min() - 2, summer.max() + 2])
    ax.grid(True)

    ax.tick_params(labelsize=14)  # adjust size of tick labels

    if climate_shift > 0:  # if there is a temperature perturbation
        # adjust the title of the plot accordingly
        ax.set_title("Summer and Winter Ground Temperatures at Kangerlussuaq, Greenland. "
                     f"{climate_shift}\u00B0C Climate Shift",
                     fontsize=16)
    fig.savefig(figure_name)


def test_part1(heat):
    sol = np.array([[0.,       0.,      0.,      0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.],
                    [0.64,     0.48,     0.4,      0.32,     0.26,     0.21,
                        0.17,     0.1375,   0.11125,  0.09,     0.072813],
                    [0.96,     0.8,      0.64,     0.52,     0.42,     0.34,
                        0.275,    0.2225,   0.18,     0.145625, 0.117813],
                    [0.96,     0.8,      0.64,     0.52,     0.42,     0.34,
                        0.275,    0.2225,   0.18,     0.145625, 0.117813],
                    [0.64,     0.48,     0.4,      0.32,     0.26,     0.21,
                        0.17,     0.1375,   0.11125,  0.09,     0.072813],
                    [0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.,       0.]])
    if np.sum(heat-sol) < 1e-16:
        print('Part 1 PASSED TEST')
    else:
        print('Part 1 FAILED TEST')


def main():
    # plotting out he wire example and print the heat matrix
    _, _, heat = heat_diff_solve(1, 0.2, 0.2, 0.02, 1)
    test_part1(heat)
    plot_diffusion(1, 0.2, 0.2, 0.02, 1, print_U=True)
    # plotting out the steady state heat map and temp profile, please set the c2 correctly, because I don't
    # have access to your report, and there's no information about how you set the parameters
    plot_diffusion('Steady-State_Heat_map.png', 100, 73000, 0.25, c2=2.5e-7, permafrost=True)
    plot_temp_profiles('Steady-State_Temp_Profile.png')
    temp_shift = [0.5, 1, 3]
    # The loop below loops through the temperature shift defined above, and plots out all the figures
    for shift in temp_shift:
        # it looks like for your diffusion function you don't have climate_shift parameters, maybe consider add it
        # to produce the diffusion figure with climate shift
        plot_diffusion(f'global_warming_Heat_map_{shift}.png', 100, 73000, 0.25, c2=2.5e-7, permafrost=True)
        #
        plot_temp_profiles(f'global_warming_Temp_Profile_{shift}.png', climate_shift=shift)


if __name__ == '__main__':
    main()
