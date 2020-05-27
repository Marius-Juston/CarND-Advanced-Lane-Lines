import numpy as np


class Line:
    def __init__(self, N=10):
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.N = N

        self.fits = np.array([[]]).reshape((0, 3))

        self.__i = 0

    def running_mean(self, var):
        cumsum = np.cumsum(var, axis=0)
        return ((cumsum[self.N:] - cumsum[:-self.N]) / float(self.N))[-1]

    def update_fit(self, fit):
        self.fits = np.concatenate((self.fits, [fit]))

        if self.__i >= self.N:
            self.best_fit = self.running_mean(self.fits)
        else:
            self.best_fit = fit

        self.__i += 1
