# This is a template file for Computer Modelling Unit 1.
# You will fill it in, and should update the comments accordingly.

import numpy as np
import matplotlib.pyplot as plt

class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        
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

# Create an instance of the class with specific parameters
your_instance_1 = Cosmology(H0=70, Omega_m=0.0, Omega_lambda=0.7)
        
calculate_Omega_k = your_instance_1.calc_Omega_k(0.0, 0.7)

# Define a range of z values
z_values = np.linspace(0, 1, 100)

# Compute the corresponding H_z values
H_z_values = [your_instance_1.integrand(z) for z in z_values]

# Plot the results
plt.plot(z_values, H_z_values)
plt.xlabel('z')
plt.ylabel('H(z)')
plt.title('Plot of integrand function H(z) with Omega_m=0.0, Omega_lambda=0.7')
plt.grid(True)

# Set the range for the x and y axes
plt.xlim(0, 1)  # Adjust the x-axis range as needed
plt.ylim(min(H_z_values), max(H_z_values))  # Adjust the y-axis range as needed

# Define a range of Omega_m values
omega_m_values = [0.2, 0.3, 0.4, 0.5]

# Create a new figure for the second plot
plt.figure()

# Loop over each Omega_m value
for omega_m in omega_m_values:
    # Create a new Cosmology instance with the current Omega_m
    c = Cosmology(70, omega_m, 0.7)

    calculate_Omega_k = c.calc_Omega_k(omega_m, 0.7)
    
    # Calculate H(z) values for the current Omega_m
    H_z_values = [c.integrand(z) for z in z_values]
    
    # Plot the H(z) values
    plt.plot(z_values, H_z_values, label=f'Omega_m={omega_m}')

# Add labels, title, and grid
plt.xlabel('z')
plt.ylabel('H(z)')
plt.title('Effect of varying Omega_m on integrand function H(z)')
plt.grid(True)
plt.legend()

# H(z) decreases as Omega_m increases, indicating a slower expansion rate due to increased matter density.

# Define a range of Omega_lambda values
omega_lambda_values = [0.6, 0.7, 0.8, 0.9]

# Create a new figure for the third plot
plt.figure()

# Loop over each Omega_lambda value
for omega_lambda in omega_lambda_values:
    # Create a new Cosmology instance with the current Omega_lambda
    c = Cosmology(70, 0.3, omega_lambda)

    calculate_Omega_k = c.calc_Omega_k(0.3, omega_lambda)
    
    # Calculate H(z) values for the current Omega_lambda
    H_z_values = [c.integrand(z) for z in z_values]
    
    # Plot the H(z) values
    plt.plot(z_values, H_z_values, label=f'Omega_lambda={omega_lambda}')

# Add labels, title, and grid
plt.xlabel('z')
plt.ylabel('H(z)')
plt.title('Effect of varying Omega_lambda on integrand function H(z)')
plt.grid(True)
plt.legend()

# H(z) increases as Omega_lambda increases, indicating a faster expansion rate due to increased dark energy density.
# Unlike the matter-dominated universe, dark energy causes the expansion to accelerate.

# Show the plots
plt.show()

def print_cosmology_objects():
    # First object: a universe dominated by dark energy
    cosmo_1 = Cosmology(H0=70, Omega_m=0.0, Omega_lambda=1.0)
    cosmo_1.calc_Omega_k(0.0, 1.0)
    print(cosmo_1)
    # Explanation:
    # This object represents a universe where dark energy dominates (Omega_lambda = 1.0),
    # and there's no matter (Omega_m = 0). This universe is expected to expand forever at
    # an accelerating rate due to the influence of dark energy.
    
    # Second object: a flat universe with equal dark energy and matter contributions
    cosmo_2 = Cosmology(H0=70, Omega_m=0.5, Omega_lambda=0.5)
    cosmo_2.calc_Omega_k(0.5, 0.5)
    print(cosmo_2)
    # Explanation:
    # This cosmology object represents a flat universe (Omega_m + Omega_lambda = 1.0),
    # where dark energy and matter contribute equally to the overall density.
    # This could represent a universe in which expansion is balanced by gravity's deceleration.
    
    # Third object: a universe dominated by matter (no dark energy)
    cosmo_3 = Cosmology(H0=70, Omega_m=1.0, Omega_lambda=0.0)
    cosmo_3.calc_Omega_k(1.0, 0.0)
    print(cosmo_3)
    # Explanation:
    # In this model, matter dominates the universe, and there is no dark energy.
    # Without dark energy, this type of universe would eventually stop expanding and potentially collapse.

    # Fourth object: an open universe (negative curvature, Omega_k > 0)
    cosmo_4 = Cosmology(H0=70, Omega_m=0.3, Omega_lambda=0.0)
    cosmo_4.calc_Omega_k(0.3, 0.0)
    print(cosmo_4)
    # Explanation:
    # This model corresponds to an open universe, where the sum of Omega_m and Omega_lambda
    # is less than 1, giving Omega_k > 0. Such a universe expands forever, but at a decreasing rate.

print_cosmology_objects()

# This is a special python idiom that
# allows the code to be run from the command line,
# but if you import this module in another script
# the code below will not be executed.


#c= Cosmology(70, 0.3, 0.7)
#calculate_Omega_k = c.calc_Omega_k(0.3, 0.7)
#if c.check_flat() == True:
#    print("The universe is flat")
