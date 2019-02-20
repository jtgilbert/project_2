import numpy as np


class Filter:

    @staticmethod
    def make_average(shape):
        """Creates average filter with given shape"""

        # Make sure filter is only 2D and square
        assert len(shape) == 2
        assert shape[0] == shape[1]

        return np.ones(shape) * 1 / np.prod(shape)

    @staticmethod
    def make_gauss(shape, sigma=1):
        """Creates a Gaußian filter with given shape and standard deviation"""

        # Make sure filter is only 2D and square
        assert len(shape) == 2
        assert shape[0] == shape[1]

        range_end = shape[0] // 2

        xs = np.arange(-range_end, range_end + 1, 1)
        ys = np.arange(-range_end, range_end + 1, 1)

        xx, yy = np.meshgrid(xs, ys, indexing='xy')

        # Compute gaussian
        g_filter = np.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2)))

        # Normalize
        g_filter /= np.sum(g_filter)

        return g_filter

    @staticmethod
    def make_sobel_u():
        """Creates a Sobel filter for approximating the discrete derivative of the image wrt u (aka x)"""

        return np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    @staticmethod
    def make_sobel_v():
        """Creates a Sobel filter for approximating the discrete derivative of the image wrt v (aka y)"""
        return np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
