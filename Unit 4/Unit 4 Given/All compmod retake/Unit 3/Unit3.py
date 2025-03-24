"""

Solar System Simulation Module
------------------------------

Module simulates the motion of particles (representing celestial bodies) 
within a solar system-like setup.

Smaller functions included:

1. **compute_separations(particles)**: 
   Computes the separations between a list of particles in 3D space, 
   returning a 3D array representing the vector separations between all pairs
   of particles.

2. **compute_forces_potential(particles, separation)**:
   Computes gravitational forces acting on each body due to all other 
   bodies and the total gravitational potential energy of the system.

MAIN FUNCTION: 
    
   **main()**:
   The main function that runs the simulation, updates the positions and 
   velocities of the particles over time, and outputs a trajectory file in the
   XYZ format. It also generates plots of various trajectories and energies to  
   visualize the system's evolution and check for problems with simulation
   parameters.

The module also contains error checking to ensure that the correct input 
parameters are provided when running the script. 

Usage:
------
This script is designed to be run from the command line with the appropriate 
arguments.
For example:
    python Unit3.py <numstep> <dt> <input_file> <output_file.xyz>

Where:
    - `numstep` is the number of timesteps to simulate.
    - `dt` is the timestep size.
    - `input_file` is the name of the input file containing initial 
        particle data.
    - `output_file` is the name of the file where the simulation results 
        will be saved in xyz format

Requirements:
-------------
- numpy
- matplotlib
- sys
-`Particle3D` class, which is assumed to be defined in `particle3D.py`.

"""
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D
import sys
from matplotlib import ticker



def compute_separations(particles):
    """ Computes separations between a list of n particles
    returns a 3D array of shape (n, n, 3). separation[i, j]
    contains the x, y and z components of the vector between
    particle j and particle i.  
        
    Seperation is given by:
    particle[i].position - particles[j].position
    
    Parameters
    ----------
    particles : list
    a list of particle 3D objects
        
    Returns
    -------
    separation : array
    n x n x 3 
    """

    # Defining n as the number of particles in input list.
    n = len(particles)
    # Setting up an array of the right size.
    separation = np.zeros((n, n, 3))
    
    
    for i in range(n):
        
        for j in range(i):
            
            # Computes all values above the diagonal, 
            # seperation of particle i from particle j.
            # Where i = j the value should be 0.
            separation[i, j] = particles[i].position - particles[j].position
            # Using symmetry to compute S[j, i].
            separation[j, i] = -separation[i, j]
            
    return separation



def compute_forces_potential(particles, separation):
 
    """Returns tuple consisting of: array of force vector on each particle 
    computed bycsumming forces from all other particles acting on it and, total
    potentialc(float) of system using particle masses and seperations.
    
    Force, F:
        -G*m1*m2 *( (r1-r2) / |r1-r2|^3 )   
    Potential energy, PE:
        -G*m1*m2 / |r1-r2| 
        r1 is position of particle i
        r2 is position of particle j

    
    Parameters
    ----------
    particles: list
    list of particle3D objects
    
    separation : array
    a n x n x 3 array of seperation 


    Returns
    -------
    Tuple 
        net_forces: array
        (n, 3) cartesian orthogonal components of net force on each particle n
    
        potential energy: float
        total potential of system

    """   
    # Defining constant G and n the number of particles.
    n = len(particles)
    G = 8.887692593e-10
    
    # Initialise array to put data in and potential energy.
    forces = np.zeros((n,n,3))
    potential = 0.0
    
    # Loop to compute effects on particle i due to particle j.
    for i in range(n):
        for j in range(i):
            
            # Magnitude of separation vector
            separation_normal = np.linalg.norm(separation[i,j])
            
            # Force on particle i due to particle j.
            forces[i,j] = -(G*particles[i].mass*particles[j].mass)\
                *separation[i,j]/separation_normal**3
            # Using Newton's Third Law, force on particle j due to particle i.
            forces[j,i]= -forces[i,j] 
            
            # Accumulating potential energy for each pair (i, j).
            potential -= (G*particles[i].mass*particles[j].mass)\
                    /separation_normal
            
    # Sums over j to get net force on all particles in axis i.   
    forces = np.sum(forces, axis = 1)

    
    return forces, potential



def main():
    """
    Main function for simulating particle dynamics arbitrary number of 
    particles from Particle 3D class

    This function reads initial conditions from input file and uses a second-order 
    Verlet integration scheme to simulate the system over 'numstep' number of time 
    steps. The function outputs the particle positions in XYZ format and generates 
    plots for analysis, including trajectory plots and energy conservation checks.

    It expects four command-line arguments:
       -Numstep (int): Number of time steps
       - dt (float): Time step size
       - Input file name (str)
       - Output file name (str)
 
    If these arguments are not provided, it will print instructions on how to run the 
    program correctly.


    Raises:
    -------
    Error: If the input file does not contain the expected number of data points.
     
    """
    
    if len(sys.argv) != 5 :
        # Ensuring right number of input are given to run correctly
        # Print instructions to remind what inputs are needed
        print("You left out the name of the output file when running.")
        print("In spyder, run like this instead:")
        print(f" %run {sys.argv[0]} <numstep> <dt> <input file> <output file>")    

    else:
        numstep = int(sys.argv[1])
        dt = float(sys.argv[2])
        input_name = sys.argv[3]
        outfile_name = sys.argv[4] 
        
        
    particles = []
    with open(input_name) as f:
        # Setting up simulation parameters from input file
        for index, line in enumerate(f):
            l = line.split()
            name = l[0]
            if name.lower() == "sun":
                sunindex = index
            elif name.lower() == "earth":
                earthindex = index
            elif name.lower() == "moon":
                moonindex = index
            elif  name.lower() == "mercury": 
                mercuryindex = index
            
            # Ensuring input file has right amount of data points for our code
            if len(l) == 8:
                particles.append(Particle3D.read_line(line))
                
            else:
                 print(" expected structure for input file is 8 pieces of data")
                 print ("with spaces in between them") 
      
                
    # Initial conditions of the system.
    time = 0.0
    # Initialise arrays that we will store results in.
    n = len(particles)
    times = np.zeros(numstep)
    energy = np.zeros(numstep)
    positions = np.zeros((n, numstep, 3))
    
    # Defining the centre of mass velocity then subtracting for all particles
    centre_of_mass_velocity = Particle3D.com_velocity(particles)
    for i in range(n):
        particles[i].velocity -= centre_of_mass_velocity

    # Computes initial seperations, forces & potential for 1st loop iteration.
    initial_distance = compute_separations(particles)
    force, potential = compute_forces_potential(particles, initial_distance)
    
    # Main time integration loop.
    for i in range(numstep):
        times[i] = time
        time += dt
        kinetic_energy =  0
        
        for j in range (n):
          # Update all particle positions and store in array.
          particles[j].update_position_2nd(dt, force[j]) 
          positions[j, i] = particles[j].position
          
        # Computes new forces and current potential from new separations.
        new_force, potential = compute_forces_potential(particles,\
        compute_separations(particles))
          
        for k in range(n):  
          # Updates particle velocity and kinetic energy.
          particles[k].update_velocity(dt, force[k], new_force[k])
          kinetic_energy += particles[k].kinetic_energy()
    
        # Computes total energy and stores it in array.
        total_energy = potential + kinetic_energy
        energy[i] = total_energy
        
        # Set up force for next loop.
        force = new_force 

    # Creation of output file in XYZ-format
    with open(outfile_name, "w") as f:
        for i in range(numstep):
            f.write(str(n)+"\n")
            f.write(f"Point = {i+1}\n")
            for k in range(n):
                label = particles[k].label
                x, y, z = positions[k, i, :]
                f.write(f"{label} {x} {y} {z}\n")
            
    
    # Plotting relevant information to check everything is as expected
    #and for convergence testing 
    
    
            
    # Plot of the Mercury - Sun x distance
    pyplot.title('Mercury-Sun x separation')
    pyplot.xlabel('time / days')
    pyplot.ylabel('x / AU')
    pyplot.plot(times, positions[mercuryindex, :, 0] - positions[sunindex, :, 0])
    pyplot.show()

    # The trajectory x-vs-y of the Earth's orbit.
    pyplot.title('Earth Trajectory around Sun')
    pyplot.xlabel('x / AU')
    pyplot.ylabel('y / AU')
    pyplot.plot(positions[earthindex, :, 0],  positions[earthindex, :, 1], marker='.', linestyle='None')
    pyplot.show()

    # Total energy of system over all numsteps
    # made Energy axis more readable for me but can use standard way of reprensentation in report
    pyplot.title('Total Energy')
    pyplot.xlabel('x / days')
    pyplot.ylabel('Energy / $M_{earth} AU^2 day^{-2} *10000$')
    pyplot.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.5f}"))
    pyplot.plot(times, energy * 10000)
    pyplot.show()
    print("min energy", np.min(energy))
    print("max energy", np.max(energy))
    
    # Plot of the Sun x distance
    pyplot.title('Sun trajectory')
    pyplot.xlabel('x / days')
    pyplot.ylabel('x / AU')
    pyplot.plot(times, positions[sunindex, :, 0])
    pyplot.show()
    
    # Trajectory x-vs-y of Moon orbit around Earth
    pyplot.title('Moon Trajectory around earth')
    pyplot.xlabel('x / AU')
    pyplot.ylabel('y / AU')
    pyplot.plot( positions[moonindex, :, 0] - positions[earthindex, :, 0],\
    positions[moonindex, :, 1] - positions[earthindex, :, 1], marker='.', linestyle='None')
    pyplot.show()

    # Plot of the Earth - Sun x distance
    pyplot.title('Earth-Sun x distance')
    pyplot.xlabel('time / days')
    pyplot.ylabel('x / AU')
    pyplot.plot(times, positions[earthindex, :, 0] - positions[sunindex, :, 0])
    pyplot.show()
    
    # Setting up an energy conservation check to ensure energy is conserved to
    # high level of accuracy
    energy_variance = (np.max(energy) - np.min(energy)) / energy[0]
    print ('energy error check, should be much less than 1:', energy_variance)

    
    
# This python standard code makes it so that the "main"
# function is only called when you run the file directly,
# not when you just import it from another python file.
if __name__ == "__main__":
    main()

