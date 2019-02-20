import numpy as np
import matplotlib.pyplot as plt


def convolve(g, h):

    """
    Convolves g with h, returns valid portion
    :param g: image
    :param h: filter
    :return: result of convolution, shape=( max( len(g), len(h) ), max( len(g[0]), len(h[0]) ) )
    result of convolution, shape=( max( len(g), len(h) ) - min( len(g), len(h) ) + 1,
    max( len(g[0]), len(h[0]) ) - min( len(g[0]), len(h[0]) ) + 1 )
    """

    out_m = max(len(g), len(h)) - min(len(g), len(h)) + 1
    out_n = max(len(g[0]), len(h[0])) - min(len(g[0]), len(h[0])) + 1

    out_array = np.zeros(shape=(out_m, out_n))

    # Iterate over entire output image
    for u in range(len(out_array)):
        for v in range(len(out_array[u])):
            # One-liner to do convolution
            out_array[u][v] = np.sum(np.multiply(h, g[u: u + len(h), v: v + len(h[0])]))

    return out_array


def _pad_input(pad_x, pad_y, g, padding_scheme='fill', fill_val=0):

    if padding_scheme == 'fill':

        v_buff = np.zeros(shape=(pad_x, len(g[0]))) + fill_val
        g = np.vstack((v_buff, g, v_buff))

        h_buff = np.zeros(shape=(len(g), pad_y)) + fill_val
        g = np.hstack((h_buff, g, h_buff))

    elif padding_scheme == 'nn':

        v_buff = np.tile(np.zeros(shape=(1, len(g[0]))) + g[0], (pad_x, 1))
        g = np.vstack((v_buff, g, v_buff))

        # Have to do some transposition to get column vectors
        h_buff = np.tile((np.zeros(shape=(len(g), 1)).T + g[:, 0]).T, (1, pad_y))
        g = np.hstack((h_buff, g, h_buff))

        # TODO: fill in diagonals?

    elif padding_scheme == 'mirror':
        raise NotImplementedError("Padding scheme {0} is not yet implemented".format(padding_scheme))

    return g


def non_linear_convolve(g, filter_size):

    # Calculate output shape
    out_m = max(len(g), filter_size)
    out_n = max(len(g[0]), filter_size)

    # Calculate pad size
    pad_x = filter_size - 1
    pad_y = filter_size - 1

    g = _pad_input(pad_x, pad_y, g)

    out_array = np.empty(shape=(out_m, out_n), dtype=bool)

    # Iterate over entire output image
    for u in range(len(out_array)):
        for v in range(len(out_array[u])):
            # Calculate max in neighborhood
            out_array[u][v] = np.equal(g[u + pad_x, v + pad_y], np.max(g[u: u + filter_size, v: v + filter_size]))

    return out_array


def img_convolution(img_fname, img_filter):

    img_color = plt.imread(img_fname)
    img_gray = img_color.mean(axis=2)  # Convert to greyscale

    out = convolve(g=img_gray, h=img_filter)

    # Plot
    plt.imshow(img_gray, cmap='gray')
    plt.show()

    plt.imshow(out, cmap='gray')
    plt.show()
