To run the code and be able to easily use plot_xyz afterwards, type this in the command line:

%run Unit3.py <numstep> <dt> <name_of_input_file> <desired_name_of_output_file.xyz>

For example: 
%run Unit3.py 365 1 solar_system.txt planets_simulation.xyz

	Warning: if name of output file matches name of existing file it will replace that file in folder.



Parameters and units:

dt: time step size, 1 dt is the lenght of 1 day on earth
numstep: number of time steps

In this simulation these units have been adopted for constants such as G, input_file and plots:
• astronomical units (AU) for length,
• Earth days for time
• Earth mass for mass



Needs Particle3D class to work


