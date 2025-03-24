"""
Author: Sewa Jabar Amin
Number: S2107030

Module for simulating the vibrational motion of diatomic molecules (O2, N2)
using the Morse potential.
This script computes the forces, potential energy, and trajectories of
particles using either the symplectic Euler or velocity Verlet integration
method. The simulation outputs the time evolution of particle positions 
and system energy, and plots the results.

Usage:
    python script_name.py <euler or verlet> <output_file> <input_file> 
    <wavenumber_ideal>

Required Libraries:
    sys, scipy, numpy, matplotlib, particle3Dex3 (for the Particle3D class)

Functions:
    - morse_force: Computes the Morse force between two particles.
    - morse_potential: Computes the potential energy between two particles, 
      according to Morse potential since this is typically used for
      covalent bonds.
    - wavenumber: Computes wavenumber in 1/cm from atomic units in position,
      energy and time. 
      Also returns wavenumber inaccuracy if a guess or ideal wavenumber is
      feed in when running code
    - main: Main function to execute the simulation and plot results.

Adapted units for plots to fit atomic physics to minimise precision loss due
to rounding errors.
-eV, electronvolts, for the energy, 
-Å, angstorm, for length
-u, atomic mass unit for mass.
-T, where T = 1.018050571e−14s for time
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
import scipy
from particle3Dex3 import Particle3D



def morse_force_and_potential(p1, p2, a, De, re):
    """
    Returns tuple, first object is the morse force on a particle p1 due to
    particle p2. (Due to newtons 3rd law this will be equal and opposite to 
    the force on p2 due to p1. This will be used later in the code)
    Secondly calculates the morse potential, U, a pair potential to describe
    covalent bonds in cluters.
   
    The paramenters a, De and re depend on the type of atom being simulated.
    
    The morse force is given by
    F(p1, p2) = 2*a*De*[ 1 - exp(-a*(r12-re))]*unitvector_r12*exp(-a(r12-re))
    
    The Morse potential is given by
    U = De[(1-exp(-a(12-re)^2)-1]

    Parameters
    ----------
    p1: Particle3D  
    p2: Particle3D
    a:  float       #alpha controls the curvature of the potential minimum
    De: float       #controls the delpth of the pontential minimum
    re: float       #controls the position of the potential minimum
    
    

    Returns
    -------
    morse force: array
                 net force between p1 and p2 as calculated in the formula above
                 
    potential:  float
                 potential energy, U, according to formula above
    """
    
    # For efficiency computing vectors needed and exponential term 
    vector_r12 = p2.position - p1.position 
    r12 = np.linalg.norm(vector_r12)       
    unit_r12 = vector_r12/r12             
    exp_term = math.exp(-a*(r12-re))       
    
    force = 2 * a * De * (1-exp_term) * unit_r12 * exp_term
    potential = De*(((1-exp_term)**2)-1)
    
   
    return force, potential

def wavenumner(positions, dt, ideal):
    """
    Uses positions of particle with time, and timestep sixe to return wavenumber
    and wavenumber fraction in si units when input of timeis in atomic units

    Parameters
    ----------
    positions : array
        Array of positions over time in atomic units
    dt        : float
        timestep size
    ideal     : float  
        wavenumber_expected

    Returns
    -------
    Wavenumber : float
        unit is per centimeter
    Wavenumber_fractional
        used to compare to other runs to see how accurate they are compared to
        smallest dt    

    """
    
    # Array with time for peaks
    peaks, _  = (scipy.signal.find_peaks(positions))
    peaks = peaks.astype(float)*dt
    # Period of vibration of particles in T, average not needed
    period= peaks[3]-peaks[2]
    # Period in seconds
    period = period * 1.018050571e-14
    # Vibration frequency in s^-1
    frequency = 1/period
    # Define speed of light c
    c = 2.9979e8
    # Wavenumber = frequency / c
    wavenumber = frequency / c / 100
    # Wavenumber fractional deviation or inaccuracy
    # best wavenumber - current number / best wavenumber
    wavenumber_fraction = (ideal - wavenumber)/ideal
    
    return wavenumber, wavenumber_fraction
    

def main():
    """
    Main function to run a molecular dynamics simulation using either Euler or
    Verlet integration.
    
    Workflow:
    1. Reads inputs from the command line:
       - Simulation mode (euler or verlet)
       - Output file name for storing results
       - Input file name containing simulation parameters and particle initial
         conditions
       - Wavenumber guess (optional, defaults to 1580 for oxygen)

    2. Opens the specified output file for writing results.

    3. Reads simulation parameters and initializes particles based on the
       input file.

    4. Performs time integration over the specified number of steps:
       - Updates particle positions and velocities using the selected 
         integration method.
       - Computes forces, potential energy, kinetic energy, and total energy
         at each step.
       - Records the time, distance between particles, and total energy for
         each time step.

    5. Outputs:
       - **Text Output to Console:**
         - Wavenumber and its fractional deviation
         - Energy fluctuation and its fractional deviation
         - Mean energy of the system
       - **Output to File:**
         - Time, distance between particles, and total energy at each time
           step are saved to the specified output file.
       - **Plots Displayed and saved:**
         - Particle trajectory plot: Distance between particles over time
         - Energy plot: Total energy of the system over time

    6. Closes the output file and ends the simulation.

    This function provides a full simulation of a diatomic molecular system, 
    including outputting key results to the console, saving data to a file,
    and displaying visualizations of the particle trajectories and energy
    changes over time.
    
    """
    
    #default for oxygen, setting up known wavenumber
    ideal = 1580
    
    # Ensuring code is ran with right amount of inputs
    if len(sys.argv) < 4 :
        print("You left out the name of the output file when running.")
        print("In spyder, run like this instead:")
        print(f"    %run {sys.argv[0]} <euler or verlet> <desired output file>\
        <input file> <wavenumber guess>")    
        # (Elememt is decided by first row of input file)
   
        sys.exit(1)
    else:
        mode, outfile_name, input_name, ideal = sys.argv[1], sys.argv[2], \
        sys.argv[3], float(sys.argv[4])
      

    # Open the output file for writing ("w")
    # if the file already exists, it is truncated.
    outfile = open(outfile_name, "w")
    
    
    # Array where particles can be stored
    particles = []
    
    
    with open(input_name) as f:
        
        # Setting up simulation parameters from input file
        for line in f:
            l = line.split()
            
            if len(l) == 6:
                De = float(l[0]) #unit[eV]
                re = float(l[1]) #unit[Å]
                a = float(l[2])
                dt = float(l[3]) #unit[T] where T = 1.018050571 x 10^-14
                numstep = int(l[4])
                time = float(l[5])
                
        # Setting up particle initial conditions from input file     
            elif len(l) == 8:
                particles.append(Particle3D.read_line(line))
                
            else:
                print("the input file is wrong")
                
    #extracting and naming our particles           
    p1 = particles[0]
    p2 = particles[1]
   

    # Open the output file for writing ("w")
    # if the file already exists, it is truncated.
    
    outfile = open(outfile_name, "w")

    # Get initial force and potential
    force, potential = morse_force_and_potential(p1, p2, a, De, re)
    # Compute and write out starting time, position, and energy values
    # to the output file.
    KE_total = p1.kinetic_energy() + p2.kinetic_energy()
    energy = KE_total + potential
    position = np.linalg.norm(p1.position - p2.position)
    
    outfile.write(f"{time}    {position}   {energy}\n")

    # Initialise numpy arrays that we will plot later, to record
    # the trajectories of the particles.
    times = np.zeros(numstep)
    positions = np.zeros(numstep)
    energies = np.zeros(numstep)

    # Start the time integration loop
    for i in range(numstep):

        # Update the positions and velocities.
        # This will depend on whether we are doing an Euler or verlet integration
        if mode == "euler":
            # Update particle position
            p1.update_position_1st(dt)
            p2.update_position_1st(dt)
            
            # Calculate force
            force, _ = morse_force_and_potential(p1, p2, a, De, re)

            # Update particle velocity 
            p1.update_velocity(dt, force)
            p2.update_velocity(dt, -force)
            
            
        elif mode == "verlet":
            # Update particle position using previous force
            p1.update_position_2nd(dt, force)
            p2.update_position_2nd(dt, -force)

            # Get the force value for the new positions
            force_new, _ = morse_force_and_potential(p1, p2, a, De, re)

            # Update particle velocity by averaging
            # current and new forces
            p1.update_velocity(dt, 0.5*(force+force_new))
            p2.update_velocity(dt, 0.5*(-1)*(force+force_new))
            
            # Re-define force value for the next iteration
            force = force_new
            
        else:
            raise ValueError(f"Unknown mode {mode} - should be euler or verlet")

        # Increase time
        time += dt
        
        # Computing total kinetic energy of system
        KE_total = p1.kinetic_energy() + p2.kinetic_energy() 
        
        _, potential = morse_force_and_potential(p1, p2, a, De, re)
       
        # Computing totoal energy of the system, inlcuding potential
        total_energy = KE_total + potential

        # Computing distance between particles
        vector_r12 = p2.position - p1.position 
        distance = np.linalg.norm(vector_r12)
        
        # Output particle information
        outfile.write(f"{time} {distance} {total_energy}\n")

        # Store the things we want to plot later in our arrays
        times[i] = time
        positions[i] = distance
        energies[i] = total_energy

    # Now the simulation has finished we can close our output file
    outfile.close()
    
    wavenumber, delta_wavenumber = wavenumner(positions, dt, ideal)
    
    print( "for", f"{input_name}")
    
    print("Wavenumber: {:.2f}".format(wavenumber), "cm^-1")
    print("Wavenumber_fractional: {:.2g}".format(delta_wavenumber))

    # The fluctuation of the total energy in the system
    delta_E = max(energies) - min(energies)
    # Energy fractional deviation
    E_fraction = delta_E / energies[0]
    print("mean energy", np.mean(energies))
    print("Energy fluctuation: {:.4g}".format(delta_E))
    print("Energy fluctuation fractional: {:.4g}".format(E_fraction))
    
    # Plot particle trajectory to screen. There are no units
    # here because it is an abstract simulation, but you should
    # include units in your plot labels!
    pyplot.figure()
    pyplot.title( f" {sys.argv[1]}: Distance between particles for {sys.argv[3]}")
    pyplot.xlabel('Time / T')
    pyplot.ylabel('Position / Å')
    pyplot.plot(times, positions)
    pyplot.show()
    pyplot.savefig(f" {sys.argv[1]} Distance of system over time")

    # Plot particle energy to screen
    pyplot.figure()
    pyplot.title(f" {sys.argv[1]}: Total energy of system for {sys.argv[3]}")
    pyplot.xlabel('Time / T')
    pyplot.ylabel('Energy / eV')
    pyplot.plot(times, energies )
    pyplot.show()
    pyplot.savefig(f" {sys.argv[1]} Total Energy of system over time")
    

#means that the main function wil only be run if we run this file,
# not if we just import it from another python file
if __name__ == "__main__":
    main()
    
    
    
    
    


