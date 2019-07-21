import numpy as np
import datetime
import csv
import time
import cProfile
pr = cProfile.Profile()
pr.enable()

start_time = time.time()
"""Base Parameters"""
"Parameters Price Process"
d     = 10     # no. of underlyings
S_0   = 100     # Initial Value of Assets at first of february
K     = 100     # Strike Price
r     = 0.02   # Since we use the risk neutral measure we constructed that mu = r <-?!
mu    = np.array([.05, .06, .07, .05, .06, .07, .05, .06, .07, .06])
sigma = np.array([.10, .11, .12, .13, .14, .14, .13, .12, .11, .10])
rho   = .1     # Correlation between Brownian Motions
"Parameters Payoff Function"
alpha = np.ones(d)* .1

""""New Parameters"""
T = .5

nIndex = [10**1, 5* 10**1, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5,10**6]
emStepsIx = np.sqrt(nIndex)
for index in range(len(nIndex)):
    start_time = time.time()
    N = nIndex[index]
    emSteps = emStepsIx[index]
    dt    = T/emSteps # Calculation with dt, yields minute steps and is stepsize of EM
    """Implementation of Approximation"""
    "Construct Covariance Matrix and Decomposition"
    # Create Matrix Eta which is the matrix of Covariances
    Eta   = np.eye(d) + rho * (np.ones(d) - np.eye(d))
    print(Eta)
    # Use the Cholesky Decomposition to derive L, which has the value that Eta = L*L'
    L     = np.linalg.cholesky(Eta)
    "Parameters Monte Carlo"
    overallSteps = len(nIndex)
    currentCompTime = np.zeros(int(overallSteps))
    V  = np.zeros(int(overallSteps))
    "Define variables to save computational time"
    stdt = np.sqrt(dt)
    dt2 = dt * 2
    "Construct EM Method"
    # d relates to the dimension of the problem, therefore, is the no. of assets in the market
    Y    = S_0*np.ones([d, N])  # initialisiere Y0! und dann überschreibe Y ständig
    det_part = 1 + mu[:, None] * dt
    det_part2 = stdt * sigma[:, None]
    for i in range(0, int(emSteps)):
        Y = (det_part + det_part2 * L @ np.random.normal(0, 1, (d, N))) * Y
        if i % 250 == 0:
            print('Progress: ' + str(i) + '/' + str(emSteps) + ' of all EM steps, while ' + str(
                np.round((time.time() - start_time), 2)) + ' secs passed.')

    "Construct Payoff"
    Payoff = np.matmul(alpha, Y)-K
    Payoff = np.sum(Payoff[(Payoff > 0)])/N # other entries would be 0 anyway, therefore its okay to use Payoff[(Payoff>0)]
    V  = np.exp(-r*T)*Payoff  # Discount Payoff to get the fair price of the Option

    "Computation Update"
    currentCompTime = np.round((time.time() - start_time), 2)

    np.savetxt("samplesize_BM_MC+EM_%s" %(N), np.transpose(['errorL1', abs(V-3.45763),'solution',V,'time', currentCompTime]), '%s', ', ')

    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
    print("BM_MC+EM_einzel_%s.txt" % (str(N)))
    pr.disable()