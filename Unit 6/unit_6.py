import numpy as np
import matplotlib.pyplot as plt
from unit_4 import Likelihood
from unit_5 import Metropolis

class Statistics:
    def __init__(self):
        pass

    def Gelman_Rubin(self, M, N, B, W):
        """
        Placeholder for Gelman-Rubin diagnostic implementation.

        Parameters
        ----------
        M : int
            The number of separate chain.
        N : int
            The length of the chain.
        B : float
            The variance of the means
        W : float
            The mean of the variances

        Returns
        -------
        float
            R - 1, which should be less than 0.01 or 0.03 for the chain to converge.
        """
        R = np.sqrt((N-1)/N + ((M+1)/M) * (B/W))

        return R - 1


    def plot(self, x, y):
        """
        ---------------------------------------------------------------------------------------------------------------------------------------------
        """
        plt.plot(x, y, label = ["Omega m", "Omega lambda", "H0"])
        plt.xlabel("Number of sample points")
        plt.ylabel("The value of R - 1")
        plt.title("Convergence test by using Gelman Rubin Statistics")
        plt.axhline(0.01, linestyle = "--", label = "Reference of convergence")
        plt.legend()
        plt.show()


    def run_chains(self, n):
        """
        Run the Metropolis algorithm for n times and calculate the Gelman Rubin statistics.

        Parameter
        ---------
        n : int
            The number of times to run the Metropolis algorithm

        Return
        ------
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        """
        convergences = []

        for i in range(10):

            theta_values_means = []
            theta_values_vars = []

            for j in range(n):
                instance = Metropolis(likelihood, initial_guess, bounds, step_size, 500 * (i + 1))
                theta_values_instance, likelihood_values_instance = instance()
                
                theta_values_mean_instance = np.mean(theta_values_instance, axis=0)
                theta_values_var_instance = np.var(theta_values_instance, axis=0)

                theta_values_means.append(theta_values_mean_instance)
                theta_values_vars.append(theta_values_var_instance)

                theta_values_mean = np.var(theta_values_means, axis=0)
                theta_values_var = np.mean(theta_values_vars, axis=0)

            result = self.Gelman_Rubin(j + 1, 500 * (i + 1), theta_values_mean, theta_values_var)

            convergences.append(result)

            print(f'The result with {500 * (i + 1)} points is {result}')

        return convergences


data_file = '/Users/sangwonkang/Library/Mobile Documents/com~apple~CloudDocs/Documents/UK/UOE/Year 3/Computer Modelling/Unit 6/pantheon_data.txt'
likelihood = Likelihood(data_file)

bounds = [(0.1, 0.45), (0.25, 0.67), (68.5, 71.5)]
initial_guess = [0.3, 0.7, 70]
step_size = [0.07, 0.1, 0.3]

statistics = Statistics()

def main():
    convergences = statistics.run_chains(10)

    statistics.plot(np.arange(500, 5001, 500), convergences)

if __name__ == "__main__":
    main()