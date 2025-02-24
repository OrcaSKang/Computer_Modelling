# This is a template file for Computer Modelling Unit 1.
# You will fill it in, and should update the comments accordingly.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.c0 = 3.0e5
        
    # The __init__ method is a special method that is called and initialises when an instance is created.

    # The self parameter is used as the first parameter in methods.

    def integrand(self, z):
        H_z = ((self.Omega_m * (1 + z)**3 + self.Omega_k * (1 + z)**2 + self.Omega_lambda)**(-0.5))
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
        return self.Omega_lambda * h**2
    
    def __str__(self):
        return "<Cosmology with H0 = {}, Omega_m = {}, Omega_lambda = {}, Omega_k = {}>".format(self.H0, self.Omega_m, self.Omega_lambda, self.Omega_k)
    
    #__str__ method is ruunned when the I print a class to determine the output string representation of the object.

    # Define the rectangle rule integration function
    def rectangle_rule_integrate(self, z, n):
        a = 0
        delta_x = (z - a) / n
        result = 0
        for i in range(n):
            x_i = a + i * delta_x
            result += self.integrand(x_i) * delta_x
        return result * (self.c0 / H0)

    # Define the trapezoidal rule integration function
    def trapezoidal_rule_integrate(self, z, n):
        a = 0
        delta_x = (z - a) / (n - 1)
        result = 0.5 * (self.integrand(a) + self.integrand(z)) * delta_x
        for i in range(1, n - 1):
            x_i = a + i * delta_x
            result += self.integrand(x_i) * delta_x
        return result * (self.c0 / H0)

    # Define the Simpson's rule integration function
    def simpsons_rule_integrate(self, z, n):
        a = 0
        delta_x = (z - a) / (2 * n)
        result = (self.integrand(a) + self.integrand(z)) * delta_x / 3
        for i in range(1, 2 * n):
            x_i = a + i * delta_x
            if i % 2 == 0:
                result += 2 * self.integrand(x_i) * delta_x / 3
            else:
                result += 4 * self.integrand(x_i) * delta_x / 3
        return result * (self.c0 / H0)

    def cumulative_trapezoid_rule(self, max_z, n):
        """
        Compute the cumulative integral of f from 0 to z using the trapezoid rule with n steps.

        Returns:
        np.ndarray: The cumulative integral values at each step.
        """
        #Define the x values (integration range)
        x = np.linspace(0, max_z, n)
    
        # Compute the function values f(x)
        y = np.array([self.integrand(x_i) for x_i in x])
        
        # Compute the cumulative integral using trapezoidal rule
        h = (max_z - 0) / (n - 1)  # Step size
        cumulative_integral = np.zeros(n)

        # Perform cumulative integration
        for i in range(1, n):
            cumulative_integral[i] = cumulative_integral[i - 1] + (y[i - 1] + y[i]) * (h / 2) 

        # Scale by constants
        return x, cumulative_integral * (self.c0 / H0)

    def get_distances(self, z_values):
        """
        Returns the comoving distances for an array of z values.

        Parameters:
        z_values (array-like): Array of redshift values.

        Returns:
        np.ndarray: Array of corresponding distances D(z) in Mpc.
        """
        # Ensure z_values is an array
        z_values = np.array(z_values)
        
        # Create a dense interpolation grid
        max_z = max(z_values)
        n_steps = 1000
        z_interp = np.linspace(0, max_z, n_steps)
        distances_interp = self.cumulative_trapezoid_rule(max_z, n_steps)[1]

        # Create interpolator
        interpolator = interp1d(z_interp, distances_interp, kind='cubic', fill_value="extrapolate")
        
        # Interpolate distances for the provided z_values
        return interpolator(z_values)

    def s_function(self, z_values):
        if self.Omega_k > 0:
            return np.sinh(np.sqrt(np.abs(self.Omega_k)) * H0 / self.c0 * self.get_distances(np.array(z_values)))
        elif self.Omega_k == 0:
            return H0 / self.c0 * self.get_distances(np.array(z_values)) 
        else :
            return np.sin(np.sqrt(np.abs(self.Omega_k)) * H0 / self.c0 * self.get_distances(np.array(z_values)))

    def get_distance_moduli(self, z_values):
        """
        Computes the distance moduli μ(z) for an array of redshift values.

        Parameters:
        z_values (array-like): Array of redshift values.

        Returns:
        np.ndarray: Array of corresponding distance moduli μ(z).
        """
        # Get comoving distances
        distances = self.get_distances(z_values)
        
        # Convert comoving distance to luminosity distance
        if self.Omega_k == 0:
            luminosity_distances = (1 + np.array(z_values)) * self.c0 / H0 * self.s_function(z_values)
        else :
            luminosity_distances = (1 + np.array(z_values)) * self.c0 / H0 * 1/np.sqrt(np.abs(self.Omega_k)) * self.s_function(z_values)

        # Compute distance modulus
        distance_moduli = 5 * np.log10(luminosity_distances) + 25
        
        return distance_moduli

# Create a new Cosmology instance with specific parameters
unit_1 = Cosmology(72, 0.3, 0.7)
Omega_k = unit_1.calc_Omega_k(0.3, 0.7)

# The speed of light in km/s

# The Hubble constant in km/s/Mpc
H0 = 72
# Calculate the distance to redshift z=1.0 using different integration methods
z = 1.0

def main():

    print("The distance D is", unit_1.rectangle_rule_integrate(z, 10000), "Mpc by using the rectangle rule")
    print("The distance D is", unit_1.trapezoidal_rule_integrate(z, 10000), "Mpc by using the trapezoidal rule")
    print("The distance D is", unit_1.simpsons_rule_integrate(z, 10000), "Mpc by using Simpson's rule")

    # Reference value for the distance using a very high number of steps
    n_steps_ref = 10**6

    # Calculate the reference distance using Simpson's rule 
    # Since the reference distance is calculated using a very high number of steps, we can assume it is accurate
    reference_distance = unit_1.simpsons_rule_integrate(z, n_steps_ref)

    # Number of steps to test
    n_steps_list = [10, 100, 1000, 10000, 100000]

    # Arrays to store the errors
    errors_rectangle = []
    errors_trapezoidal = []
    errors_simpsons = []

    # Calculate the distances and errors for each method
    for n_steps in n_steps_list:
        distance_rectangle = unit_1.rectangle_rule_integrate(z, n_steps)
        distance_trapezoidal = unit_1.trapezoidal_rule_integrate(z, n_steps)
        distance_simpsons = unit_1.simpsons_rule_integrate(z, n_steps)
        
        error_rectangle = abs((distance_rectangle - reference_distance) / reference_distance)
        error_trapezoidal = abs((distance_trapezoidal - reference_distance) / reference_distance)
        error_simpsons = abs((distance_simpsons - reference_distance) / reference_distance)
        
        errors_rectangle.append(error_rectangle)
        errors_trapezoidal.append(error_trapezoidal)
        errors_simpsons.append(error_simpsons)

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.loglog(n_steps_list, errors_rectangle, label='Rectangle Rule', marker='o')
    plt.loglog(n_steps_list, errors_trapezoidal, label='Trapezoidal Rule', marker='s')
    plt.loglog(n_steps_list, errors_simpsons, label="Simpson's Rule", marker='^')
    plt.xlabel('Number of Steps')
    plt.ylabel('Absolute Fractional Error')
    plt.title('Absolute Fractional Error in Distance to Redshift z=1.0')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example usage
    max_z = 1.0
    n_test = 100
    z_values, distances = unit_1.cumulative_trapezoid_rule(max_z, n_test)

    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, distances, label='Cumulative Trapezoid Rule', marker='o')
    plt.xlabel('Redshift')
    plt.ylabel('Distance (Mpc)')
    plt.title('Distance to Redshift using Cumulative Trapezoid Rule')
    plt.legend()
    plt.grid(True)
    plt.show()


    z_values = [0.1, 0.5, 0.7, 1.0, 0.3]
    
    # Get distances
    distances = unit_1.get_distances(z_values)
    print("Distances D(z) in Mpc:", distances)

    # Get distance moduli for unit 1
    z_values_plot = np.linspace(0, 1, 100)
    distance_moduli_1 = unit_1.get_distance_moduli(z_values_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_values_plot, distance_moduli_1, label='Mu with unit 1', marker='o')
    plt.xlabel('Redshift')
    plt.ylabel('Mu')
    plt.title('Distance modulus to Redshift with unit 1')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    Omega_m_values = [0.1, 0.4, 0.8, 1.0]
    for Omega_m in Omega_m_values:
        Omega_k = unit_1.calc_Omega_k(Omega_m, 0.7)
        distance_moduli = unit_1.get_distance_moduli(z_values_plot)
        plt.plot(z_values_plot, distance_moduli, label='Mu with Omega_k = {}'.format(Omega_k), marker='o')
        plt.legend()

    unit_2 = Cosmology(72, 0.1, 0.1)
    unit_2.calc_Omega_k(0.1, 0.1)
    distance_moduli_2 = unit_2.get_distance_moduli(z_values_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(z_values_plot, distance_moduli_2, label='Mu with unit 2', marker='o')
    plt.xlabel('Redshift')
    plt.ylabel('Mu')
    plt.title('Distance modulus to Redshift with unit 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    unit_3 = Cosmology(72, 0.7, 0.7)
    unit_3.calc_Omega_k(0.7, 0.7)
    distance_moduli_3 = unit_3.get_distance_moduli(z_values_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_values_plot, distance_moduli_3, label='Mu with unit 3', marker='o')
    plt.xlabel('Redshift')
    plt.ylabel('Mu')
    plt.title('Distance modulus to Redshift with unit 3')
    plt.legend()
    plt.grid(True)
    plt.show()

main()