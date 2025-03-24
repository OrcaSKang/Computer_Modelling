"""
CompMod Ex2: Particle3D, a class to describe point particles in 3D space

An instance describes a particle in Euclidean 3D space: 
velocity and position are [3] arrays

Author: Sewa Amin
Number: S2107030

"""
import numpy as np


class Particle3D(object):
    """
    Class to describe point-particles in 3D space

    Attributes
    ----------
    label: name of the particle
    mass: mass of the particle
    position: position of the particle
    velocity: velocity of the particle

    Methods
    -------
    __init__
    __str__
    kinetic_energy: computes the kinetic energy
    momentum: computes the linear momentum
    update_position_1st: updates the position to 1st order
    update_position_2nd: updates the position to 2nd order
    update_velocity: updates the velocity

    Static Methods
    --------------
    read_file: initializes a P3D instance from a file handle
    total_kinetic_energy: computes total K.E. of a list of particles
    com_velocity: computes centre-of-mass velocity of a list of particles
    """

    def __init__(self, label, mass, position, velocity):
        """
        Initialises a particle in 3D space.

        Parameters
        ----------
        label: str
            name of the particle
        mass: float
            mass of the particle
        position: [3] float array
            position vector
        velocity: [3] float array
            velocity vector
        """
        self.label = label
        self.mass = float(mass)
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __str__(self):
        """
        Return an XYZ-format string. The format is
        label    x  y  z

        Returns
        -------]
        str
        """
        
        return f"{self.label} {self.position[0]} {self.position[1]} {self.position[2]}"

    
    def kinetic_energy(self):
        """
        Returns the kinetic energy of a Particle3D instance

        Returns
        -------
        ke: float
            1/2 m v**2
        """
     
        magVsq=(self.velocity[0]**2+self.velocity[1]**2+self.velocity[2]**2)
        
        ke = (1/2)*(self.mass)*magVsq
        
        # Since to get the magnitude of V (velocity) we take square root and
        # that is undone in the next calculation where we take V^2 I skipped
        # both to just get magVsq (magnitude of velocity, squared)
        return  ke



    def momentum(self):
        """
        Returns the momentum of a Particle3D instance
        
        Returns
        --------
        p:  array
            m*v
        """
            
        p = self.mass*(self.velocity)
        
        return  p

   
    def update_position_1st(self, dt):
        
        """
        Takes argument dt as time step and velocity v(t) of particle at time t,
        to update position of particle
        
        Updates
        -------
        r(t+dt): array
                 r(t) + dt*v(t)
        
        """
       
        self.position += dt*self.velocity
        
   
    def update_position_2nd(self, dt, f):
        """
        Takes arguments dt as time step, f as force on particle and 
        velocity of particle at t, v(t), to update position of particle
       
        Updates
        --------
        r(t+dt): array
                r(t) + dt*v(t) + dt^2 * f/2m
        """
        
        self.position += dt*self.velocity + (dt**2)*f/(2*self.mass)
              
        
    def update_velocity(self, dt, force):
        """
        Takes arguemtns dt as time step and f(t) as force to update
        velocity of particle from v(t) to v(t+dt)
        
        Updates
        --------
        v(t+dt) = array
                  dt*f(t)/m
        """
        self.velocity += dt*(force)/(self.mass)
   
    
    @staticmethod
    def read_line(line):
        """
        Creates a Particle3D instance given a line of text.

        The input line should be in the format:
        label   <mass>  <x> <y> <z>    <vx> <vy> <vz>

        Parameters
        ----------
        filename: str
            Readable file handle in the above format

        Returns
        -------
        p: Particle3D
        """
        
        s = line.split()
        
        label = s[0]
        
        mass = float(s[1])
        
        x,y,z = float(s[2]), float(s[3]), float(s[4])
        
        vx,vy,vz = float(s[5]), float(s[6]), float(s[7])
        
        p = Particle3D(label, mass, np.array([x, y, z]), np.array([vx, vy, vz]))
        
        
        return  p

    @staticmethod
    def total_kinetic_energy(particles):
        """
        Computes the TKE (total kinetic energy) of  a list of P3D's
        
        Parameters
        ----------
        particles: list
            A list of Particle3D instances

        Returns
        -------
        TKE: float
             Total Kinetic energy
             Σ(0.5*m*vi^2)
        """
        TKE = 0
        
        l= len(particles)
        
        for i in range(0, l):
            
            TKE += 0.5*particles[i].mass*(np.linalg.norm(particles[i].velocity))**2
    
        return  float(TKE)
    
    @staticmethod
    def com_velocity(particles):
        """
        Computes the CoM( (centre of mass) velocity of a list of P3D's

        Parameters
        ----------
        particles: list
            A list of Particle3D instances

        Returns
        -------
        com_vel: array
                 Centre-of-mass velocity
                 Σmi*vi/Σmi 
                 where mi is the mass of the i'th particle
                
        """
        nominator = 0
        denominator = 0
        
        for p in particles:
            
            nominator += p.mass*p.velocity
            
            denominator += p.mass
        
        CoM = nominator/denominator
        
        return  CoM
    
    