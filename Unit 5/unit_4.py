import numpy as np
import matplotlib.pyplot as plt
from cosmology import Cosmology
import scipy.optimize as opt

class Likelihood:
    def __init__(self, data_file):
        """
        Initialise the Likelihood class.

        Parameters
        ----------
        data_file : str
            The path to the file containing the data.
        """
        self.data = np.loadtxt(data_file)
        self.z_values = self.data[:, 0]
        self.mobs_values = self.data[:, 1]
        self.sigma_values = self.data[:, 2]

    def compute_theoretical_model(self, z_values, Omega_m, Omega_Lambda, H0, N_value = 1000, M = -19.3):
        """
        Compute the theoretical model for the magnitude m  of a dimensionless meausure of brightness.

        Parameters
        ----------
        z_values : np.ndarray
            The redshift values at which to compute the theoretical model.
        Omega_m : float
            The fraction of the critical density that is made up of matter.
        Omega_Lambda : float
            The fraction of the critical density that is made up of dark energy.
        H0 : float
            The Hubble parameter in units of km/s/Mpc.
        N_value : int
            The number of integration points to use.
        M : float
            The absolute magnitude of the supernovae.
        
        Returns
        -------
        np.ndarray
            The theoretical model for the magnitude m.
        """
        self.Omega_k = 1 - Omega_m - Omega_Lambda
        cosmology = Cosmology(Omega_m, Omega_Lambda, H0, N_value)
        return cosmology.get_distance_moduli(z_values, N_value) + M

    def __call__(self, theta = [0.3, 0.7, 70], N_value = 1000):
        """
        Compute the log likelihood of the data given the parameters theta.

        Parameters
        ----------
        theta : list
            A list of parameters [Omega_m, Omega_Lambda, H0].
        N_value : int
            The number of integration points to use.
        
        Returns
        -------
        float
            The log likelihood of the data given the parameters theta.
        """
        Omega_m, Omega_Lambda, H0 = theta
        m_theoretical = self.compute_theoretical_model(self.z_values, Omega_m, Omega_Lambda, H0, N_value)

        # Compute the log of the Gaussian likelihood
        residuals = self.mobs_values - m_theoretical
        chi_squared = np.sum((residuals / self.sigma_values) ** 2)
        log_likelihood = -0.5 * chi_squared
        
        return log_likelihood

    def test_likelihood_convergence(self, data_file, theta, N_values):
        """
        Test the convergence of the likelihood calculation with different numbers of integration points.

        Parameters
        ----------
        data_file : str
            The path to the file containing the data.
        theta : list
            A list of parameters [Omega_m, Omega_Lambda, H0].
        N_values : list
            A list of integration points to test.

        Returns
        -------
        Plot
            A plot of the likelihood values with different numbers of integration points.
        """
        likelihood = Likelihood(data_file)
        likelihood_values = []

        for N in N_values:
            likelihood_value = likelihood(theta, N)
            likelihood_values.append(likelihood_value + 476.20740656030637)

        # Plot the convergence
        plt.plot(N_values, likelihood_values, marker='o')
        plt.xlabel('Number of Integration Points')
        plt.ylabel('Likelihood with different integration points')
        plt.title('Convergence of Likelihood Calculation')
        plt.grid(True)
        plt.show()

def find_best_fitting_parameters(likelihood, initial_guess, bounds):
    """
    Find the best fitting parameters for the given likelihood function.

    Parameters
    ----------
    likelihood : function
        The likelihood function to optimize.
    initial_guess : list
        The initial guess for the parameters.
    bounds : list
        The bounds for the parameters.

    Returns
    -------
    Optimized Parameters
    """
    # Define the negative likelihood function
    def negative_likelihood(theta):
        return -likelihood(theta)
        
    # Perform the optimization
    result = opt.minimize(negative_likelihood, initial_guess, bounds = bounds)
    
    # Return the result
    return result

def main():
    """
    
    """
    # Example usage
    data_file = '/Users/sangwonkang/Library/Mobile Documents/com~apple~CloudDocs/Documents/UK/UOE/Year 3/Computer Modelling/Unit 4/pantheon_data.txt'
    likelihood = Likelihood(data_file)
    theta1 = [0.3, 0.7, 70.0]

    # Should be around -476
    print("Likelihood = ",likelihood(theta1, 1000))  

    # Test the convergence of the likelihood calculation
    NN = [10, 20, 40, 80, 160, 250, 350, 500, 640]

    # Plot the convergence
    print(likelihood.test_likelihood_convergence(data_file, theta1, NN))

    #Initial guess for the parameters
    initial_guess = [0.3, 0.7, 75.0]

    #Bounds for the parameters
    bounds = [(0, 1), (0, 1), (50, 100)]

    # Find the best-fitting parameters
    result = find_best_fitting_parameters(likelihood, initial_guess, bounds)

    # Print the result
    print("Optimization result:", result)
    print("Best-fitting parameters:", result.x)

    # Check the likelihood value at the optimized parameters
    optimized_likelihood = likelihood(result.x)
    print("Optimized likelihood:", optimized_likelihood)

    # Plot the data with error bars
    plt.errorbar(likelihood.z_values, likelihood.mobs_values, yerr=likelihood.sigma_values, fmt='o', label='Data')

    # Generate model points using the best-fitting parameters
    best_fit_params = result.x
    model = Cosmology(best_fit_params[0], best_fit_params[1], best_fit_params[2], 350)
    model_values = model.get_distance_moduli(likelihood.z_values) -19.3

    # Plot the best-fit model points
    plt.plot(likelihood.z_values, model_values, label='Best-fit model', color='red')

    # Add labels and a legend and show the plot
    plt.xlabel('Redshift')
    plt.ylabel('Magnitude')
    plt.title('Supernova Data')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Compute the residuals
    residuals = (likelihood.mobs_values - model_values) / likelihood.sigma_values

    # Plot the residuals
    plt.errorbar(likelihood.z_values, residuals, yerr=likelihood.sigma_values, fmt='o', label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Redshift')
    plt.ylabel('Residuals')
    plt.title('Residuals of Supernova Data')
    plt.grid(True)
    plt.legend()
    plt.show()

    # New theta and bound values
    theta2 = [0.3, 0, 70.0]
    bounds2 = [(0, 1), (0, 0), (50, 100)]

    # Find the best-fitting parameters for theta2
    result2 = find_best_fitting_parameters(likelihood, theta2, bounds2)

    # Check the likelihood value at the optimized parameters
    optimized_likelihood_2 = likelihood(result2.x)
    print("Optimized likelihood for theta 2:", optimized_likelihood_2)

    # Print the result for theta2
    print("Optimization result for theta 2:", result2)
    print("Best-fitting parameters for theta 2:", result2.x)

    # Plot the data with error bars for theta2
    plt.errorbar(likelihood.z_values, likelihood.mobs_values, yerr=likelihood.sigma_values, fmt='o', label='Data')

    # Generate model points using the best-fitting parameters for theta2
    best_fit_params2 = result2.x
    model = Cosmology(best_fit_params2[0], best_fit_params2[1], best_fit_params2[2], 350)
    model_values = model.get_distance_moduli(likelihood.z_values) -19.3

    # Plot the best-fit model points for theta2
    plt.plot(likelihood.z_values, model_values, label='Best-fit model', color='red')

    # Add labels and a legend and show the plot
    plt.xlabel('Redshift')
    plt.ylabel('Magnitude')
    plt.title('Supernova Data with different theta')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.show()

if __name__ == "__main__":
    main()