import numpy as np
import time
start_time = time.time()

"""Base Parameters"""
"Parameters Price Process"
d     = 10      # no. of underlyings
S_0   = 100     # Initial Value of Assets at first of february
K     = 100     # Strike Price
r     = 0.02
mu    = [0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.06]
sigma = [.10, .11, .12, .13, .14, .14, .13, .12, .11, .10]
rho   = 0.1     # Correlation between Brownian Motions

"Parameters Payoff Function"
alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# alpha is the index of the weights in the basket option
"Parameters Monte Carlo"
N     = 3*10**7  # Number of Monte Carlo samples
"""Construct Time Parameter"""
T = .5
Tmat  = np.ones((d, N))*T

"""Implementation of Approximation"""
"Construct Covariance Matrix and Decomposition"
# Create Matrix Eta which is the matrix of Covariances
Eta = np.eye(d) + rho * (np.ones(d) - np.eye(d))
# Use the Cholesky Decomposition to derive L, which has the value that Eta = L*L'
L = np.linalg.cholesky(Eta)
"Construct Brownian Motion step"
# Generate Delta_W trough sqrt(T)*L*Z with Z~N(0,1), Delta_W is a matrix of dim = (d,N)
Delta_W = np.matmul(np.sqrt(T) * L, np.random.normal(0, 1, (d, N)))
"Construct Price Procsses"
# Generate the price processes (mu = r), dimension (d,N)
S = S_0*np.exp((np.diag(mu) - 1/2 * np.diag(sigma)**2)@ Tmat + np.diag(sigma) @ Delta_W)

"Construct Payoff"
# Construct Payoff (Sum alpha_i*S_i - K)^+ for  every simulation step of Monte Carlo, dim = (N,1)
Payoff = np.matmul(alpha, S)-K
Payoff = np.sum(Payoff[(Payoff > 0)])/N  # other entries would be 0 anyway, therefore its okay to use Payoff[(Payoff>0)]

#Discount Payoff to get the fair price of the Option
V = np.exp(- r * T) * Payoff

"Output"
print('In a market consisting of ' + str(d) + ' stocks, \nwe evaluated a classical basket option, \nusing ' + str(N) + ' MC samples.')
print('The true price of the Option is ' + str(round(V, 6)) + '\nusing the exact solution for the SDE')
print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
