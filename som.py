from math import sqrt

from numpy import (unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, meshgrid, dot, einsum)

from collections import defaultdict


def _build_iteration_indexes(data_len, num_iterations, random_generator=None):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.
    If random_generator is not None, it must be an instalce
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    iterations = arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    return iterations

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    """
    return sqrt(dot(x, x.T))

def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.
    t : int
        current iteration.
    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))


class SelfMap(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        """Initializes a Self Organizing Map.
        Parameters
        ----------
        x : int
            x dimension of the SOM.
        y : int
            y dimension of the SOM.
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)
        """
        random_seed=None
        self._random_generator = random.RandomState(random_seed)
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)
        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        self.topology = 'hexagonal'
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        self._xx[::-2] -= 0.5    
        self._decay_function = asymptotic_decay
        self.neighborhood = self._gaussian
        self._activation_distance = self._euclidean_distance

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self._weights

    def get_euclidean_coordinates(self):
        """Returns the position of the neurons on an euclidean
        plane that reflects the chosen topology in two meshgrids xx and yy.
        Neuron with map coordinates (1, 4) has coordinate (xx[1, 4], yy[1, 4])
        in the euclidean plane.
        """
        return self._xx.T, self._yy.T

    def convert_map_to_euclidean(self, xy):
        """Converts map coordinates into euclidean coordinates
        that reflects the hexagonal topology
        """
        return self._xx.T[xy], self._yy.T[xy]

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._xx-self._xx.T[c], 2)/d)
        ay = exp(-power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T  # the external product gives a matrix

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # w_new = eta * 'gaussian' * (x-w)
        self._weights += einsum('ij, ijk->ijk', g, x-self._weights)

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data."""
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()

    def train(self, data, num_iteration):
        """Trains the SOM picking samples at random from data.
        Parameters
        ----------
        data : np.array or list
            Data matrix.
        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        """
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration, random_generator)
        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.winner(data[iteration]), t, num_iteration)

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance."""
        um = zeros((self._weights.shape[0],
                    self._weights.shape[1],
                    8))  # 2 spots more for hexagonal topology

        #ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
        #jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2

        if self.topology == 'hexagonal':
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                w_2 = self._weights[x, y]
                e = y % 2 == 0   
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < self._weights.shape[0] and
                            y+j >= 0 and y+j < self._weights.shape[1]):
                        w_1 = self._weights[x+i, y+j]
                        um[x, y, k] = fast_norm(w_2-w_1)

        um = um.sum(axis=2)
        return um/um.max()

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap
