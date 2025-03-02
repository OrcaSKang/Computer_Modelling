import numpy as np
import matplotlib.pyplot as plt
from unit_4 import Likelihood
import time

class MarginalisedLikelihood:
    def __init__(self, likelihood, bounds, N_value):
        """
        Initialise the MarginalisedLikelihood class.

        Parameters
        ----------
        likelihood : function
            The likelihood function to optimize.
        theta : list
            A list of parameters [Omega_m, Omega_Lambda, H0].
        bounds : list
            The bounds for the parameters.
        N_value : float 
            A float of integration points to test.
        """
        self.likelihood = likelihood
        self.bounds = bounds
        self.N_value = N_value

    def make_grid(self, likelihood, bounds):
        """
        Create a grid of likelihood values.

        Parameters
        ----------
        likelihood : instance
            The likelihood instance to optimize.
        bounds : list
            The bounds for the parameters.

        Returns
        -------
        tuple
            A tuple of the grid values(Omega m range, Omega lambda range, H0 range and likelihood grid).
        """

        # Create a grid of parameter values
        theta1_range = np.linspace(bounds[0][0], bounds[0][1], self.N_value)
        theta2_range = np.linspace(bounds[1][0], bounds[1][1], self.N_value)
        theta3_range = np.linspace(bounds[2][0], bounds[2][1], self.N_value)

        # Initialize the likelihood values
        likelihood_grid = np.zeros((self.N_value, self.N_value, self.N_value))

        # Compute the likelihood values
        for i in range (self.N_value):
            for j in range (self.N_value):
                for k in range (self.N_value):
                    theta = [theta1_range[i], theta2_range[j], theta3_range[k]]
                    likelihood_grid[i, j, k] = likelihood(theta, 350)

        return theta1_range, theta2_range, theta3_range, likelihood_grid

    def plot_3D(self, all_grid):
        """
        Plot the 3D marginalised likelihood.

        Parameters
        ----------
        all_grid : tuple
            A tuple of the grid values(Omega m range, Omega lambda range, H0 range and likelihood grid).

        Returns
        -------
        Plot
            A 3D plot of the likelihood with respect to the parameters.
        """

        # Create a grid of likelihood values
        theta1_range, theta2_range, theta3_range, likelihood_grid = all_grid

        # Flatten the grids and likelihood values for plotting
        theta1_range, theta2_range, theta3_range = np.meshgrid(theta1_range, theta2_range, theta3_range)
        theta1_flat = theta1_range.flatten()
        theta2_flat = theta2_range.flatten()
        theta3_flat = theta3_range.flatten()
        likelihood_flat = likelihood_grid.flatten()

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        # Plot the 3D scatter plot
        sc = ax.scatter(theta1_flat, theta2_flat, theta3_flat, c = likelihood_flat, cmap = 'viridis')

        # Add color bar
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label = 'Log likelihood Density')

        # Set labels
        ax.set_xlabel('Omega m')
        ax.set_ylabel('Omega lambda')
        ax.set_zlabel('H0')
        ax.set_title('3D Marginalised Likelihood')

        plt.show()

    def marginalised_2D(self, all_grid):
        """
        Marginalise over the parameters Omega_m and Omega_Lambda.

        Parameters
        ----------
        all_grid : tuple
            A tuple of the grid values(Omega m range, Omega lambda range, H0 range and likelihood grid).

        Returns
        -------
        Plot
            2D plots of the marginalised likelihood over Omega_m, Omega_Lambda and H0.
        """

        # Assign the grid values
        theta1_range, theta2_range, theta3_range, likelihood_grid = all_grid

        # Flatten the grids and likelihood values for plotting
        likelihood_flat = likelihood_grid.flatten()

        # Find the maximum value of the likelihood
        max_value = np.max(likelihood_flat)

        # Marginalise over Omega_m, Omega_Lambda and H0
        self.likelihood_grid_marginalised_m = np.sum(np.exp(likelihood_grid - max_value), axis=0)

        self.likelihood_grid_marginalised_k = np.sum(np.exp(likelihood_grid - max_value), axis=1)
        
        self.likelihood_grid_marginalised_H0 = np.sum(np.exp(likelihood_grid - max_value), axis=2)

        # Plot the marginalised likelihoods
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(self.likelihood_grid_marginalised_m.T, extent = [theta2_range[0], theta2_range[-1], theta3_range[0], theta3_range[-1]], aspect = 'auto', origin = 'lower')
        plt.xlabel('Omega Lambda')
        plt.ylabel('H0')
        plt.title('Marginalised Likelihood over Omega_m')
        plt.colorbar(label = 'Probability Density')

        plt.subplot(1, 3, 2)
        plt.imshow(self.likelihood_grid_marginalised_k.T, extent = [theta1_range[0], theta1_range[-1], theta3_range[0], theta3_range[-1]], aspect = 'auto', origin = 'lower')
        plt.xlabel('Omega m')
        plt.ylabel('H0')
        plt.title('Marginalised Likelihood over Omega Lambda')
        plt.colorbar(label = 'Probability Density')

        plt.subplot(1, 3, 3)
        plt.imshow(self.likelihood_grid_marginalised_H0.T, extent = [theta1_range[0], theta1_range[-1], theta2_range[0], theta2_range[-1]], aspect = 'auto', origin = 'lower')
        plt.xlabel('Omega m')
        plt.ylabel('Omega Lambda')
        plt.title('Marginalised Likelihood over H0')
        plt.colorbar(label = 'Probability Density')
        plt.tight_layout()
        plt.show()

    def marginalised_1D(self, all_grid):
        """
        Marginalise over the two parameters of Omega_m, Omega_Lambda and H0.

        Parameters
        ----------
        all_grid : tuple
            A tuple of the grid values(Omega m range, Omega lambda range, H0 range and likelihood grid).

        Returns
        -------
        Plot
            1D plots of the marginalised likelihood of Omega_m, Omega_Lambda and H0.
        """

        # Assign the grid values
        theta1_range, theta2_range, theta3_range, likelihood_grid = all_grid

        # Marginalise over two parameters
        likelihood_marginalised_m_lambda = np.sum(self.likelihood_grid_marginalised_m, axis = 0)
        likelihood_marginalised_k_H0 = np.sum(self.likelihood_grid_marginalised_k, axis = 1)
        likelihood_marginalised_H0_m = np.sum(self.likelihood_grid_marginalised_H0, axis = 0)

        # 1D Plot the marginalised likelihoods
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(theta1_range, likelihood_marginalised_k_H0, label = 'Omega m')
        plt.xlabel('Omega m')
        plt.ylabel('Probability Density')
        plt.title('Marginalised Likelihoods of Omega m')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(theta2_range, likelihood_marginalised_H0_m, label = 'Omega Lambda')
        plt.xlabel('Omega Lambda')
        plt.ylabel('Probability Density')
        plt.title('Marginalised Likelihoods of Omega Lambda')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(theta3_range, likelihood_marginalised_m_lambda, label = 'H0')
        plt.xlabel('H0')
        plt.ylabel('Probability Density')
        plt.title('Marginalised Likelihoods of H0')
        plt.legend()

        plt.tight_layout()
        plt.show()

class Metropolis:
    def __init__(self, likelihood, initial_guess, bounds, step_size, N_steps):
        """
        Initialise the Metropolis class.

        Parameters
        ----------
        likelihood : function
            The likelihood function to optimize.
        initial_guess : list
            The initial guess for the parameters.
        bounds : list
            The bounds for the parameters.
        step_size : float
            The step size for the Metropolis algorithm.
        N_steps : int
            The number of steps to take.
        """

        # Initialize the parameters
        self.likelihood = likelihood
        self.initial_guess = initial_guess
        self.bounds = bounds
        self.step_size = step_size
        self.N_steps = N_steps

    def __call__(self):
        """
        Run the Metropolis algorithm.

        Returns
        -------
        np.ndarray
            The parameter values at each step.
        """

        # Make arrays to store the parameter values and likelihoods
        theta = np.array(self.initial_guess)
        theta_values = [theta]
        log_likelihood = self.likelihood(theta, 350)
        likelihood_values = [log_likelihood]
        
        # Run the Metropolis algorithm
        for i in range(self.N_steps):
            theta_new = theta + self.step_size * np.random.randn(len(theta))
            log_likelihood_new = self.likelihood(theta_new, 350)

            if (self.bounds[0][0] <= theta_new[0] <= self.bounds[0][1] and 
                self.bounds[1][0] <= theta_new[1] <= self.bounds[1][1] and 
                self.bounds[2][0] <= theta_new[2] <= self.bounds[2][1]):

                if  log_likelihood_new > log_likelihood or np.log(np.random.uniform(0, 1)) < log_likelihood_new - log_likelihood:
                    theta = theta_new
                    log_likelihood = log_likelihood_new

            # Append the values to the arrays
            theta_values.append(theta)
            likelihood_values.append(log_likelihood)

        return np.array(theta_values), np.array(likelihood_values)

    def plot_3D_metro(self, theta_values, likelihood_values):
        """
        Plot the 3D marginalised likelihood.

        Parameters
        ----------
        theta_values : np.ndarray
            The parameter values at each step.
        likelihood_values : np.ndarray
            The likelihood values at each step.

        Returns
        -------
        Plot
            A 3D plot of the likelihood with respect to the parameters.
        """

        # Assign the values from the Metropolis algorithm
        theta_values, likelihood_values = self.__call__()

        # Compute the likelihood values
        likelihood_values = np.exp(likelihood_values - np.max(likelihood_values))

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        # Plot the 3D scatter plot
        sc = ax.scatter(theta_values[:, 0], theta_values[:, 1], theta_values[:, 2], c = likelihood_values, cmap = 'viridis')

        # Add color bar
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label = 'Probability Density')

        # Set labels
        ax.set_xlabel('Omega m')
        ax.set_ylabel('Omega lambda')
        ax.set_zlabel('H0')
        ax.set_title('MCMC 3D plot')

        plt.show()

    def plot_2D_metro(self, theta_values, likelihood_values):
        """
        Plot the 2D marginalised likelihood.

        Parameters
        ----------
        theta_values : np.ndarray
            The parameter values at each step.
        likelihood_values : np.ndarray
            The likelihood values at each step.

        Returns
        -------
        Plot
            2D plots of the marginalised likelihood over Omega_m, Omega_Lambda and H0.
        """

        # Normalise the likelihood values
        likelihood_values = np.exp(likelihood_values - np.max(likelihood_values))

        # Plot the 2D marginalised likelihoods over Omega_m, Omega_Lambda and H0
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(theta_values[:, 1], theta_values[:, 2], c = likelihood_values, cmap = 'viridis')
        plt.xlabel('Omega Lambda')
        plt.ylabel('H0')
        plt.title('MCMC 2D plot over Omega m')
        plt.colorbar(label = 'Probability Density')

        plt.subplot(1, 3, 2)
        plt.scatter(theta_values[:, 0], theta_values[:, 2], c = likelihood_values, cmap = 'viridis')
        plt.xlabel('Omega m')
        plt.ylabel('H0')
        plt.title('MCMC 2D plot over Omega lambda')
        plt.colorbar(label = 'Probability Density')

        plt.subplot(1, 3, 3)
        plt.scatter(theta_values[:, 0], theta_values[:, 1], c = likelihood_values, cmap = 'viridis')
        plt.xlabel('Omega m')
        plt.ylabel('Omega Lambda')
        plt.title('MCMC 2D plot over H0')
        plt.colorbar(label = 'Probability Density')

        plt.tight_layout()
        plt.show()

    def plot_histogram(self, theta_values):
        """
        Plot the histogram of the parameter values.

        Parameters
        ----------
        theta_values : np.ndarray
            The parameter values at each step.

        Returns
        -------
        Plot
            Histograms of the parameter values.
        """
        # Plot the histogram of the parameter values
        plt.figure(figsize=(15, 5))

        # Plot the histograms of the parameter values
        plt.subplot(1, 3, 1)
        plt.hist(theta_values[:, 0], bins = 50, alpha = 0.5, label = f'Parameter {1}')
        plt.xlabel('Omega m')
        plt.ylabel('Frequency')
        plt.title('Histogram of Omega m')

        plt.subplot(1, 3, 2)
        plt.hist(theta_values[:, 1], bins = 50, alpha = 0.5, label = f'Parameter {2}')
        plt.xlabel('Omega Lambda')
        plt.ylabel('Frequency')
        plt.title('Histogram of Omega Lambda')

        plt.subplot(1, 3, 3)
        plt.hist(theta_values[:, 2], bins = 50, alpha = 0.5, label = f'Parameter {3}')
        plt.xlabel('H0')
        plt.ylabel('Frequency')
        plt.title('Histogram of H0')

        plt.tight_layout()
        plt.show()

data_file = '/Users/sangwonkang/Library/Mobile Documents/com~apple~CloudDocs/Documents/UK/UOE/Year 3/Computer Modelling/Unit 5/pantheon_data.txt'
likelihood = Likelihood(data_file)

def main():
    """
    Run the main function.
    """
    
    # Define the bounds and initial guess for the parameters
    bounds = [(0.1, 0.45), (0.25, 0.67), (69.5, 71.5)]

    # Start the timer
    start_time = time.time()

    # Make an instance of the MarginalisedLikelihood class
    Marginalised = MarginalisedLikelihood(likelihood, bounds, 40)

    # Create a grid of likelihood values
    theta1_range, theta2_range, theta3_range, likelihood_grid = Marginalised.make_grid(likelihood, bounds)

    # Assign the grid values
    all_grid = theta1_range, theta2_range, theta3_range, likelihood_grid

    # End the timer
    end_time = time.time()

    # Print the time taken to compute the grid
    print(f'Time taken to compute the grid: {end_time - start_time} seconds')

    # Plot the 3D marginalised likelihood
    Marginalised.plot_3D(all_grid)

    # Plot the 2D marginalised likelihood over Omega_m, Omega_Lambda and H0
    Marginalised.marginalised_2D(all_grid)

    # Plot the 1D marginalised likelihood of Omega_m, Omega_Lambda and H0
    Marginalised.marginalised_1D(all_grid)

    # Define the initial guess and step size for the metropolis algorithm
    initial_guess = [0.3, 0.7, 70]
    step_size = [0.01, 0.02, 0.3]

    # Make an instance of the Metropolis class
    instance = Metropolis(likelihood, initial_guess, bounds, step_size, 10000)
    
    # Run the Metropolis algorithm
    theta_values, likelihood_values = instance()

    # Plot the 3D likelihood values
    instance.plot_3D_metro(theta_values, likelihood_values)

    # Plot the 2D likelihood values over Omega_m, Omega_Lambda and H0
    instance.plot_2D_metro(theta_values, likelihood_values)   

    # Plot the histogram of the parameter values
    instance.plot_histogram(theta_values)

if __name__ == "__main__":
    main()