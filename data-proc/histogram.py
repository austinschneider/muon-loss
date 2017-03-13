import numpy as np

class histogram_nd(object):
    """ Histogram class for creating weighted average plot """
    def __init__(self, n, bins, store_data=False, internals=None):
        if internals is not None:
            for key in internals.keys():
                setattr(self, key, internals[key])
            return

        self.add_calls = 0
        self.entries = 0

        self.n = n
        self.bins = np.array(bins)
        self.size = len(bins) - 1
        self.store_data = store_data
        size = self.size
        self.weights = np.zeros(size)
        self.weights2 = np.zeros(size)
        self.x_weighted = np.zeros(size)
        self.y_weighted = np.zeros((n, size))
        self.x2_weighted = np.zeros(size)
        self.y2_weighted = np.zeros((n, size))

        if self.store_data:
            self.x_data = []
            self.y_data = np.zeros((n, 0)).tolist()
            self.bin_data = []
            self.weights_data = []
        else:
            self.x_data = None
            self.y_data = None
            self.bin_data = None
            self.weights_data = None

    def get_internals(self):
        return self.__dict__

    def add(self, x, y, weights, bin=None):
        self.add_calls += 1
        self.entries += len(x)
        if self.store_data:
            if len(y) != len(self.y_data):
                raise ValueError("y length must match y_data!")
            self.x_data += list(x)
            for i in  xrange(len(y)):
                self.y_data[i] += list(y[i])
            self.weights_data += list(weights)
            if bin:
                self.bin_data += bin
            else:
                self.bin_data += ([None]*len(y))

        x = np.array(x)
        y = np.array(y)
        weights = np.array(weights)

        if bin:
            bin = np.array(bin)
            weights_by_bin = np.bincount(bin, minlength=self.size, weights=weights)
            weights2_by_bin = np.bincount(bin, minlength=self.size, weights=weights**2)
            x_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=x * weights)
            x2_weighted_by_bin = np.bincount(bin, minlength=self.size, weights=x**2 * weights)
            y_weighted_by_bin = np.zeros((self.n, self.size))
            y2_weighted_by_bin = np.zeros((self.n, self.size))
            for i in xrange(len(y)):
                y_weighted_by_bin[i] = np.bincount(bin, minlength=self.size, weights=y[i] * weights)
                y2_weighted_by_bin[i] = np.bincount(bin, minlength=self.size, weights=y[i]**2 * weights)
        else:
            weights_by_bin, edges = np.histogram(x, bins=self.bins, weights=weights)
            weights2_by_bin, edges = np.histogram(x, bins=self.bins, weights=weights**2)
            x_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=x * weights)
            x2_weighted_by_bin, edges = np.histogram(x, bins=self.bins, weights=x**2 * weights)
            y_weighted_by_bin = np.zeros((self.n, self.size))
            y2_weighted_by_bin = np.zeros((self.n, self.size))
            for i in xrange(len(y)):
                y_weighted_by_bin[i], edges = np.histogram(x, bins=self.bins, weights=y[i] * weights)
                y2_weighted_by_bin[i], edges = np.histogram(x, bins=self.bins, weights=y[i]**2 * weights)

        self.weights += weights_by_bin
        self.weights2 += weights2_by_bin
        self.x_weighted += x_weighted_by_bin
        self.y_weighted += y_weighted_by_bin
        self.x2_weighted += x2_weighted_by_bin
        self.y2_weighted += y2_weighted_by_bin

        return {'w': weights_by_bin, 'w2': weights2_by_bin, 'x': x_weighted_by_bin, 'y': y_weighted_by_bin, 'x2': x2_weighted_by_bin, 'y2': y2_weighted_by_bin}

    def accumulate(self, hist):
        if not isinstance(hist, histogram_nd):
            raise ValueError("Trying to accumulate non histogram object")
        if not hist.n == self.n:
            raise ValueError("Histogram must have same dimension")
        same_shape = (self.weights.shape == hist.weights.shape and
                self.weights2.shape == hist.weights2.shape and
                self.x_weighted.shape == hist.x_weighted.shape and
                self.y_weighted.shape == hist.y_weighted.shape and
                self.x2_weighted.shape == hist.x2_weighted.shape and
                self.y2_weighted.shape == hist.y2_weighted.shape)
        if not same_shape:
            raise ValueError("Histogram shapes are not the same")

        self.weights += hist.weights
        self.weights2 += hist.weights2
        self.x_weighted += hist.x_weighted
        self.y_weighted += hist.y_weighted
        self.x2_weighted += hist.x2_weighted
        self.y2_weighted += hist.y2_weighted

    def get_w(self):
        w = np.copy(self.weights)
        w[w == 0] = 1
        return w
    def get_w2(self):
        w2 = np.copy(self.weights2)
        w2[w2 == 0] = 1
        return w2
    def get_x(self):
        return self.x_weighted / self.get_w()
    def get_y(self):
        w = self.get_w()
        y = self.y_weighted / ([w]*self.n)
        return y
    def get_x2(self):
        return self.x2_weighted / self.get_w()
    def get_y2(self):
        w = self.get_w()
        y2 = self.y2_weighted / ([w]*self.n)
        return y2
    def stddev(self, x, x2):
        variance = np.zeros(x.shape)
        good_var = np.logical_not(np.logical_or(np.isclose(x2, x**2, rtol=1e-05, atol=1e-09), x2 < x**2))
        variance[good_var] = x2[good_var] - x[good_var]**2
        return np.sqrt(variance)
    def get_x_stddev(self):
        return self.stddev(self.get_x(), self.get_x2())
    def get_y_stddev(self):
        return self.stddev(self.get_y(), self.get_y2())
    def get_x_stddev_of_mean(self):
        return self.get_x_stddev() * np.sqrt(self.get_w2()) / self.get_w() 
    def get_y_stddev_of_mean(self):
        return self.get_y_stddev() * ([np.sqrt(self.get_w2()) / self.get_w()]*self.n)

class histogram(histogram_nd):
    """ Histogram class for creating weighted average plot """
    def __init__(self, bins, store_data=False):
        super(histogram, self).__init__(1, bins, store_data)

    def add(self, x, y, weights, bin=None):
        return super(histogram, self).add(x, [y], weights, bin)
    def get_y(self):
        return super(histogram, self).get_y()[0]
    def get_y2(self):
        return super(histogram, self).get_y2()[0]
    def get_y_stddev(self):
        return super(histogram, self).get_y_stddev()[0]
    def get_y_stddev_of_mean(self):
        return self.get_y_stddev() * np.sqrt(self.get_w2()) / self.get_w()
