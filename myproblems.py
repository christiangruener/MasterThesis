import datetime, os, logging
import pandas as panda
import numpy as np
import tensorflow as tf


class myProblems(object):
    def __init__(self, dimensions, terminal_time):
        self._dimensions = dimensions
        self._terminal_time = terminal_time
    @property
    def dimensions(self):
        return self._dimensions
    @property
    def terminal_time(self): 
        return self._terminal_time
    def generator(self, t, x_t, Y_t, Z_t):
        raise NotImplementedError
    def terminal_condition(self, X_t):
        raise  NotImplementedError
    def sample(self, number_of_samples, number_of_time_intervals):
        raise NotImplementedError

class ClassicalBS(myProblems):
    def __init__(
            self,
            dimensions=1,
            terminal_time=.5,
            start_price=100.,
            strike_price=100.,
            interest_rate=.02,
            volatiltiy=.05,
    ):
        super(ClassicalBS, self).__init__(dimensions, terminal_time)
        self._X_init = start_price
        self._K = strike_price
        self._riskfree = interest_rate
        self._sigma = volatiltiy

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size=[number_of_samples,
                  self._dimensions,
                  number_of_time_intervals]
        ) * sqrt_dt
        X = np.empty([
            number_of_samples,
            self._dimensions,
            number_of_time_intervals + 1
        ])
        X[:, :, 0] = self._X_init

        for i in range(number_of_time_intervals):
            X[:, :, i + 1] = (1. + self._riskfree * dt + self._sigma * dW[:, :, i]) * X[:, :, i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return 0.

    def terminal_condition(self, X_T):
        return np.exp(-self._riskfree * self.terminal_time) * tf.maximum(
            tf.reduce_sum(X_T, axis=1, keepdims= True) - self._K,
            0.
        )
class BasketOption10(myProblems):
    def __init__(
        self,
        dimensions = 10,
        terminal_time = .5,
        start_price = 100.,
        strike_price = 100.,
        interest_rate = .02,
        expectancy = np.array([.05, .06, .07, .05, .06, .07, .05, .06, .07, .06]),
        volatiltiy = np.array([.10, .11, .12, .13, .14, .14, .13, .12, .11, .10]),
        rho = .1,
    ):
        super(BasketOption10, self).__init__(dimensions, terminal_time)
        self._X_init = start_price
        self._K = strike_price
        self._riskfree = interest_rate
        self._sigma = volatiltiy
        self._mu_bar = expectancy
        Eta = np.eye(dimensions) + rho * (np.ones(dimensions) - np.eye(dimensions))
        self._L = np.linalg.cholesky(Eta)

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size= [number_of_samples,
                   self._dimensions,
                   number_of_time_intervals]
        ) * sqrt_dt
        X = np.empty([
            number_of_samples,
            self._dimensions,
            number_of_time_intervals + 1
        ])

        X[:,:, 0] = self._X_init
        dW_corr = np.empty([
            self._dimensions,
            number_of_samples
        ])

        for i in range(number_of_time_intervals):
            dW_corr[:, :] = np.matmul(self._L, np.transpose(dW[:, :, i]))
            X[:,:, i + 1] = (1. + self._mu_bar * dt + self._sigma *  np.transpose(dW_corr[:,:])) * X[:,:,i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return 0.

    def terminal_condition(self, X_T):
        return np.exp(-self._riskfree * self.terminal_time) * tf.maximum(
            1/self.dimensions * tf.reduce_sum(X_T, axis=1, keepdims= True) - self._K,
            0.
        )

class RainbowOption10(myProblems):
    def __init__(
        self,
        dimensions = 10,
        terminal_time = .5,
        start_price = 100.,
        strike_price = 100.,
        interest_rate = .02,
        expectancy = np.array([.05, .06, .07, .05, .06, .07, .05, .06, .07, .06]),
        volatiltiy = np.array([.10, .11, .12, .13, .14, .14, .13, .12, .11, .10]),
        rho = .1,
    ):
        super(RainbowOption10, self).__init__(dimensions, terminal_time)
        self._X_init = start_price
        self._K = strike_price
        self._riskfree = interest_rate
        self._sigma = volatiltiy
        self._mu_bar = expectancy
        Eta = np.eye(dimensions) + rho * (np.ones(dimensions) - np.eye(dimensions))
        self._L = np.linalg.cholesky(Eta)

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size= [number_of_samples,
                   self._dimensions,
                   number_of_time_intervals]
        ) * sqrt_dt
        X = np.empty([
            number_of_samples,
            self._dimensions,
            number_of_time_intervals + 1
        ])
        X[:,:, 0] = self._X_init
        dW_corr = np.empty([
            self._dimensions,
            number_of_samples
        ])

        for i in range(number_of_time_intervals):
            dW_corr[:, :] = np.matmul(self._L, np.transpose(dW[:, :, i]))
            X[:,:, i + 1] = (1. + self._mu_bar * dt + self._sigma *  np.transpose(dW_corr[:,:])) * X[:,:,i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return 0.

    def terminal_condition(self, X_T):
        return np.exp(-self._riskfree * self.terminal_time) * tf.maximum(
            tf.reduce_max(X_T, axis=1, keepdims= True) - self._K,
            0.
        )

class BasketOption100(myProblems):
    def __init__(
        self,
        dimensions = 100,
        terminal_time = .5,
        start_price = 100.,
        strike_price = 100.,
        interest_rate = .02,
        expectancy = .05,
        volatiltiy = .05,
        rho = .2
    ):
        super(BasketOption100, self).__init__(dimensions, terminal_time)
        self._X_init = start_price
        self._K = strike_price
        self._riskfree = interest_rate
        self._sigma = volatiltiy
        self._mu_bar = expectancy
        Eta = np.eye(dimensions) + rho * (np.ones(dimensions) - np.eye(dimensions))
        self._L = np.linalg.cholesky(Eta)

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size= [number_of_samples,
                   self._dimensions,
                   number_of_time_intervals]
        ) * sqrt_dt
        X = np.empty([
            number_of_samples,
            self._dimensions,
            number_of_time_intervals + 1
        ])
        X[:,:, 0] = self._X_init
        dW_corr = np.empty([
            self._dimensions,
            number_of_samples
        ])

        for i in range(number_of_time_intervals):
            dW_corr[:, :] = np.matmul(self._L, np.transpose(dW[:, :, i]))
            X[:,:, i + 1] = (1. + self._mu_bar * dt + self._sigma *  np.transpose(dW_corr[:,:])) * X[:,:,i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return 0.

    def terminal_condition(self, X_T):
        return np.exp(-self._riskfree * self.terminal_time) * tf.maximum(
            1/self.dimensions * tf.reduce_sum(X_T, axis=1, keepdims= True) - self._K,
            0.
        )

class RainbowOption100(myProblems):
    def __init__(
        self,
        dimensions = 100,
        terminal_time = .5,
        start_price = 100.,
        strike_price = 100.,
        interest_rate = .02,
        expectancy = .05,
        volatiltiy = .05,
        rho = .2
    ):
        super(RainbowOption100, self).__init__(dimensions, terminal_time)
        self._X_init = start_price
        self._K = strike_price
        self._riskfree = interest_rate
        self._sigma = volatiltiy
        self._mu_bar = expectancy
        Eta = np.eye(dimensions) + rho * (np.ones(dimensions) - np.eye(dimensions))
        self._L = np.linalg.cholesky(Eta)

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size= [number_of_samples,
                   self._dimensions,
                   number_of_time_intervals]
        ) * sqrt_dt
        X = np.empty([
            number_of_samples,
            self._dimensions,
            number_of_time_intervals + 1
        ])
        X[:,:, 0] = self._X_init

        dW_corr = np.empty([
            self._dimensions,
            number_of_samples
        ])

        for i in range(number_of_time_intervals):
            dW_corr[:, :] = np.matmul(self._L, np.transpose(dW[:, :, i]))
            X[:,:, i + 1] = (1. + self._mu_bar * dt + self._sigma *  np.transpose(dW_corr[:,:])) * X[:,:,i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return 0.

    def terminal_condition(self, X_T):
        return np.exp(-self._riskfree * self.terminal_time) * tf.maximum(
            tf.reduce_max(X_T, axis=1, keepdims= True) - self._K,
            0.
        )
    
class Dax_Basket(myProblems):
    def __init__(
            self,
            dimensions=30,
            terminal_time= 31/365,
            interest_rate=.02,
            expectancy=.07, #long time expectancy
            volatiltiy=.15, #VDax value used (vola expectancy for next 30 days)
    ):
        super(Dax_Basket, self).__init__(dimensions, terminal_time)
        self._riskfree = interest_rate
        self._sigma = volatiltiy
        self._mu_bar = expectancy
        path = os.path.dirname(__file__) # current path
        import_cholesky = panda.read_csv(
            path + '/cholesky_korr.csv',
            header=None,
            delimiter=','
        )
        self._L = np.array(import_cholesky.iloc[:, :])
        import_prices_weights = panda.read_csv(
            path + '/dax_price_weights.csv',
            header=0,
            delimiter=';'
        )
        prices_weights = np.array(import_prices_weights.iloc[:, :])
        self._basket_weights = prices_weights[0,:]
        self._X_init = prices_weights[1,:]
        self._K = np.matmul(self._basket_weights,np.transpose(self._X_init))

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size=[number_of_samples,
                  self._dimensions,
                  number_of_time_intervals]
        ) * sqrt_dt
        X = np.empty([
            number_of_samples,
            self._dimensions,
            number_of_time_intervals + 1
        ])
        X[:, :, 0] = self._X_init

        dW_corr = np.empty([
            self._dimensions,
            number_of_samples
        ])
        for i in range(number_of_time_intervals):
            dW_corr[:, :] = np.matmul(self._L, np.transpose(dW[:, :, i]))
            X[:, :, i + 1] = (1. + self._mu_bar * dt + self._sigma * np.transpose(dW_corr[:, :])) \
                             * X[:, :, i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return 0.

    def terminal_condition(self, X_T):
        X_T_weighted = self._basket_weights*X_T
        return np.exp(-self._riskfree * self.terminal_time) * \
               tf.maximum(
                   tf.reduce_sum(
                       X_T_weighted,
                       axis=1,
                       keepdims=True)
                   - self._K, 0.
               )

class Counterparty_CreditRisk1(myProblems):
    def __init__(
            self,
            dimensions= 1,
            strike_price_1 = 90.,
            strike_price_2 = 110.,
            parameter_L = 10.,
            terminal_time= 2,
            start_price = 100.,
            interest_rate=.0,
            expectancy=.0,
            volatiltiy=.2,
            beta = .03
    ):
        super(Counterparty_CreditRisk1, self).__init__(dimensions, terminal_time)
        self._X_init = start_price
        self._riskfree = interest_rate
        self._sigma = volatiltiy
        self._mu_bar = expectancy
        self._K_1 = strike_price_1
        self._K_2 = strike_price_2
        self._para_L = parameter_L
        self._beta = beta

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size=[number_of_samples,
                  self._dimensions,
                  number_of_time_intervals]
            ) * sqrt_dt
        X = np.empty([
                    number_of_samples,
                    self._dimensions,
                    number_of_time_intervals + 1
                ])
        X[:, :, 0] = self._X_init
        factor = np.exp((-(self._sigma**2)/2)*dt)
        for i in range(number_of_time_intervals):
            X[:, :, i + 1] = (factor * np.exp(self._sigma * dW[:, :, i])) * X[:, :, i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return self._beta * (tf.maximum(Y_t,0)-Y_t)

    def terminal_condition(self, X_T):
        tmp = tf.reduce_min(X_T, 1, keepdims=True)
        return tf.maximum(tmp - self._K_1, 0) - tf.maximum(tmp - self._K_2,0) - self._para_L

class Counterparty_CreditRisk100(myProblems):
    def __init__(
            self,
            dimensions= 100,
            strike_price_1 = 30.,
            strike_price_2 = 60.,
            parameter_L = 15.,
            terminal_time= 2,
            start_price = 100.,
            interest_rate=.0,
            expectancy=.0,
            volatiltiy=.2,
            beta = .03
    ):
        super(Counterparty_CreditRisk100, self).__init__(dimensions, terminal_time)
        self._X_init = start_price
        self._riskfree = interest_rate
        self._sigma = volatiltiy
        self._mu_bar = expectancy
        self._K_1 = strike_price_1
        self._K_2 = strike_price_2
        self._para_L = parameter_L
        self._beta = beta

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        sqrt_dt = np.sqrt(dt)
        dW = np.random.normal(
            size=[number_of_samples,
                  self._dimensions,
                  number_of_time_intervals]
            ) * sqrt_dt
        X = np.empty([
                    number_of_samples,
                    self._dimensions,
                    number_of_time_intervals + 1
                ])
        X[:, :, 0] = self._X_init
        factor = np.exp((self._mu_bar-(self._sigma**2)/2)*dt)
        for i in range(number_of_time_intervals):
            X[:, :, i + 1] = (factor * np.exp(self._sigma * dW[:, :, i])) * X[:, :, i]
        return dW, X

    def generator(self, t, x_t, Y_t, Z_t):
        return self._beta * (tf.maximum(Y_t,0)-Y_t)

    def terminal_condition(self, X_T):
        tmp = tf.reduce_min(X_T, 1, keepdims=True)
        return tf.maximum(tmp - self._K_1, 0) - tf.maximum(tmp - self._K_2,0) - self._para_L

class HJB(myProblems):
    def __init__(self, dimensions=100, terminal_time=1.0):
        super(HJB, self).__init__(dimensions, terminal_time)
        self._sigma = np.sqrt(2.)
        self._lambda = 1.

    def sample(self, number_of_samples, number_of_time_intervals):
        dt = self.terminal_time / number_of_time_intervals
        dW = np.random.normal(size=[
            number_of_samples,
            self._dimensions,
            number_of_time_intervals
        ]) * np.sqrt(dt)
        X = np.empty([
            number_of_samples,
            self._dimensions,
            number_of_time_intervals + 1
        ])
        X[:, :, 0] = 0.
        for i in range(number_of_time_intervals):
            X[:, :, i + 1] = X[:, :, i] + self._sigma * dW[:, :, i]
        return dW, X

    def generator(self, t, X_t, Y_t, Z_t):
        tmp = tf.reduce_sum(tf.square(Z_t), axis=1, keepdims=True)
        return -self._lambda * tmp

    def terminal_condition(self, X_T):
        tmp = tf.reduce_sum(tf.square(X_T), axis=1, keepdims=True)
        return tf.log((1. + tmp) / 2.)