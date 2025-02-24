# This is a template file for Computer Modelling Unit 1.
# You will fill it in, and should update the comments accordingly.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Cosmology:
    def __init__(self, Omega_m, Omega_lambda, H0, n_steps = 1000):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = 1 - Omega_m - Omega_lambda
        self.n_steps = n_steps
        
    # The __init__ method is a special method that is called and initialises when an instance is created.

    # The self parameter is used as the first parameter in methods.

    def integrand(self, z):
        H_z = ((self.Omega_m * (1 + z) ** 3 + self.Omega_k * (1 + z) ** 2 + self.Omega_lambda) ** (-0.5))
        return H_z
    
    def calc_Omega_k(self, Omega_m, Omega_lambda):
        self.Omega_k = 1 - Omega_m - Omega_lambda
        return self.Omega_k
    
    def check_flat(self):
        if self.Omega_k == 0:
            return True
        else: 
            return False
    
    def set_Omega_m(self, Omega_m):
        self.Omega_m = Omega_m
        self.Omega_lambda = 1 - self.Omega_m - self.Omega_k
        
    def set_Omega_lambda(self, Omega_lambda):
        self.Omega_lambda = Omega_lambda
        self.Omega_m = 1 - self.Omega_lambda - self.Omega_k

    def Omega_lambda_h_squared(self):
        h = self.H0 / 100
        return self.Omega_lambda * h ** 2
    
    def __str__(self):
        return "<Cosmology with H0 = {}, Omega_m = {}, Omega_lambda = {}, Omega_k = {}>".format(self.H0, self.Omega_m, self.Omega_lambda, self.Omega_k)
    
    #__str__ method is ruunned when the I print a class to determine the output string representation of the object.

    # Define the rectangle rule integration function
    def rectangle_rule_integrate(self, z, n):
        """
        Compute the integral of f from 0 to z using the rectangle rule with n steps.

        Parameters:
        z (float): The upper limit of integration.
        n (int): The number of steps to use.

        Returns:
        float: The integral value.
        """
        a = 0
        delta_x = (z - a) / n
        result = 0
        for i in range(n):
            x_i = a + i * delta_x
            result += self.integrand(x_i) * delta_x
        return result * (c0 / self.H0)

    # Define the trapezoidal rule integration function
    def trapezoidal_rule_integrate(self, z, n):
        """
        Compute the integral of f from 0 to z using the trapezoidal rule with n steps.
        
        Parameters:
        z (float): The upper limit of integration.
        n (int): The number of steps to use.

        Returns:
        float: The integral value.
        """
        a = 0
        delta_x = (z - a) / (n - 1)
        result = 0.5 * (self.integrand(a) + self.integrand(z)) * delta_x
        for i in range(1, n - 1):
            x_i = a + i * delta_x
            result += self.integrand(x_i) * delta_x
        return result * (c0 / self.H0)

    # Define the Simpson's rule integration function
    def simpsons_rule_integrate(self, z, n):
        """
        Compute the integral of f from 0 to z using Simpson's rule with n steps.

        Parameters:
        z (float): The upper limit of integration.
        n (int): The number of steps to use.

        Returns:
        float: The integral value.
        """
        a = 0
        delta_x = (z - a) / (2 * n)
        result = (self.integrand(a) + self.integrand(z)) * delta_x / 3
        for i in range(1, 2 * n):
            x_i = a + i * delta_x
            if i % 2 == 0:
                result += 2 * self.integrand(x_i) * delta_x / 3
            else:
                result += 4 * self.integrand(x_i) * delta_x / 3
        return result * (c0 / self.H0)

    def cumulative_trapezoid_rule(self, max_z, n_steps):
        """
        Compute the cumulative integral of f from 0 to z using the trapezoid rule with n steps.

        Parameters:
        max_z (float): The upper limit of integration.
        n_steps (int): The number of steps to use.

        Returns:
        np.ndarray: The cumulative integral values at each step.
        """
        #Define the x values (integration range)
        x = np.linspace(0, max_z, n_steps)
    
        # Compute the function values f(x)
        y = np.array([self.integrand(x_i) for x_i in x])
        
        # Compute the cumulative integral using trapezoidal rule
        h = (max_z - 0) / (n_steps - 1)  # Step size
        cumulative_integral = np.zeros(n_steps)

        # Perform cumulative integration
        for i in range(1, n_steps):
            cumulative_integral[i] = cumulative_integral[i - 1] + (y[i - 1] + y[i]) * (h / 2) 

        # Scale by constants
        return x, cumulative_integral * (c0 / self.H0)

    def get_distances(self, z_values, n_steps):
        """
        Returns the comoving distances for an array of z values.

        Parameters:
        z_values (array-like): Array of redshift values.
        n_steps (int): Number of integration steps.

        Returns:
        np.ndarray: Array of corresponding distances D(z) in Mpc.
        """
        # Ensure z_values is an array
        z_values = np.array(z_values)
        
        # Create a dense interpolation grid
        max_z = max(z_values)
        z_interp = np.linspace(0, max_z, n_steps)
        distances_interp = self.cumulative_trapezoid_rule(max_z, n_steps)[1]

        # Create interpolator
        interpolator = interp1d(z_interp, distances_interp, kind='cubic', fill_value="extrapolate")
        
        # Interpolate distances for the provided z_values
        return interpolator(z_values)

    def s_function(self, z_values):
        """
        Compute the S function for an array of redshift values.

        Parameters:
        z_values (array-like): Array of redshift values.

        Returns:
        np.ndarray: Array of corresponding S values.
        """

        # Compute the S function for the given z values
        if self.Omega_k > 0:
            return np.sinh(np.sqrt(np.abs(self.Omega_k)) * self.H0 / c0 * self.get_distances(np.array(z_values), self.n_steps))
        elif self.Omega_k == 0:
            return self.H0 / c0 * self.get_distances(np.array(z_values), self.n_steps) 
        else :
            return np.sin(np.sqrt(np.abs(self.Omega_k)) * self.H0 / c0 * self.get_distances(np.array(z_values), self.n_steps))

    def get_distance_moduli(self, z_values, n_values = 1000):
        """
        Computes the distance moduli μ(z) for an array of redshift values.

        Parameters:
        z_values (array-like): Array of redshift values.
        n_values (int): Number of integration/interpolation points.

        Returns:
        np.ndarray: Array of corresponding distance moduli μ(z).
        """
        # Get comoving distances
        distances = self.get_distances(z_values, n_values)
        # Convert comoving distance to luminosity distance
        if self.Omega_k == 0:
            luminosity_distances = (1 + np.array(z_values)) * c0 / self.H0 * self.s_function(z_values)
        else :
            luminosity_distances = (1 + np.array(z_values)) * c0 / self.H0 * 1/np.sqrt(np.abs(self.Omega_k)) * self.s_function(z_values)

        # Compute distance modulus
        distance_moduli = 5 * np.log10(luminosity_distances) + 25
        
        return distance_moduli

# Create a new Cosmology instance with specific parameters
unit_1 = Cosmology(0.4, 0.65, 80)

c0 = 2.99792e5  # Speed of light in km/s

# Print the Cosmology instance
# print(unit_1.get_distance_moduli([1.5]))
