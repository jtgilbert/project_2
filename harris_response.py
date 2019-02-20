from convolution import *
import sys
from Filter import *
import matplotlib.pyplot as plt


def compute_harris_response(img, smoothing_filter):

    # NOTE: (*) = convolution

    # Make respective Sobel filters for computing gradients
    S_u = Filter.make_sobel_u()
    S_v = Filter.make_sobel_v()

    # Compute gradient of image with respect to u = I_u
    di_du = convolve(img, S_u)

    # Compute gradient of image with respect to v = I_v
    di_dv = convolve(img, S_v)

    # Compute squares of gradient
    # I_uu = w (*) np.multiply(I_u, I_u)
    di_du_sq = convolve(di_du ** 2, smoothing_filter)
    di_dv_sq = convolve(di_dv ** 2, smoothing_filter)

    # Compute product of gradients
    di_du_di_dv = convolve(np.multiply(di_du, di_dv), smoothing_filter)

    # Finally, compute the harris response
    # Note: we add the 1e-16 to prevent division by 0.
    # 1.0 + 1e-16 = 1.0, so we do not have to worry about screwing up division
    return np.divide((np.multiply(di_du_sq, di_dv_sq) - di_du_di_dv ** 2), (di_du_sq + di_dv_sq) + 1e-16)


def non_maximal_suppression(h, n=100, max_filter_size=4):
    """Returns two vectors, one for x point location, one for y point location.
    These points correspond to local maxima point locations"""

    local_max_pts = local_max_loc_and_intensity(h, max_filter_size)

    # Filter out local maxima with relatively weak responses
    local_max_pts = local_max_pts[local_max_pts[:, 2] > 1e-5]

    # Sort by intensity and reverse
    local_max_pts = np.flip(local_max_pts[local_max_pts[:, 2].argsort(kind='mergesort')], axis=0)

    # Keep best n
    return local_max_pts[:n, 1], local_max_pts[:n, 0]


def local_max_loc_and_intensity(h, max_filter_size):

    """
    Helper function to compute locations of local maxima
    :param h: harris repsonse
    :param max_filter_size: size of filter window to be used
    :return: matrix that is shape of number of local maxima.
    First column is i point location, second column is j point location
    Last column is value of harris response at that location
    """
    bool_mask = non_linear_convolve(h, max_filter_size)

    # Get list of points where mask is true
    i, j = np.where(bool_mask == True)

    # Grab the corresponding intensities
    intensities = h[i, j]

    # (n,) -> (n, 1)
    i = np.expand_dims(i, axis=1)
    j = np.expand_dims(j, axis=1)
    intensities = np.expand_dims(intensities, axis=1)

    return np.hstack((i, j, intensities))


def point_distance(p1, p2):
    """Computes euclidean distance between p1 and p2"""

    x_diff = float(p1[0]) - float(p2[0])
    y_diff = float(p1[1]) - float(p2[1])

    return np.sqrt((x_diff * x_diff) + (y_diff * y_diff))


def adaptive_non_maximal_suppression(h, n, max_filter_size, c=0.9):

    # pt = point
    # curr = current
    # loc = location
    # idx = index
    # dist = distance
    # num = number
    # pot = potential
    # kpts = key points

    max_loc_and_intensity = local_max_loc_and_intensity(h, max_filter_size)

    # Filter out local maxima with relatively weak responses
    max_loc_and_intensity = max_loc_and_intensity[max_loc_and_intensity[:, 2] > 1e-5]

    local_max_pts_loc = max_loc_and_intensity[:, 0:2]
    local_max_pts_intensity = max_loc_and_intensity[:, 2]

    num_pot_kpts = len(local_max_pts_loc)

    # this list has index in the closest point w/ higher harris response
    closest_pt_loc = np.empty(shape=(num_pot_kpts, 2), dtype=np.int)
    closest_pt_dist = np.empty(shape=(num_pot_kpts, 1))
    closest_pt_dist.fill(np.inf)

    # for each point, find closest point w/ higher harris response
    for i in range(num_pot_kpts):
        for j in range(num_pot_kpts):

            # Don't want to compare point to itself
            if i == j:
                continue

            # Compute distance between the current point and the point we are looking at
            new_dist = point_distance(local_max_pts_loc[i], local_max_pts_loc[j])

            # If new point has higher intensity and is closer, record it
            if local_max_pts_intensity[j] * c > local_max_pts_intensity[i] and new_dist < closest_pt_dist[i]:
                closest_pt_loc[i] = local_max_pts_loc[j]
                closest_pt_dist[i] = new_dist

    # If dist is still infinity, it is the 'best' local max
    cond = closest_pt_dist[:, 0] == np.inf
    closest_pt_loc[cond] = local_max_pts_loc[cond]

    pt_dist_mtx = np.hstack((closest_pt_loc, closest_pt_dist))

    dt = np.dtype([('i', int),('j', int),('dist', float)])
    pt_dist_mtx.view(dtype=dt).sort(order='dist', axis=0)  # Sort in place, by distance
    pt_dist_mtx = np.flipud(pt_dist_mtx)  # Reverse

    reversed_points = set()

    # Take the first n points we see
    for i in range(len(pt_dist_mtx)):

        current_point = (pt_dist_mtx[i, 1], pt_dist_mtx[i, 0])

        if current_point not in reversed_points:
            reversed_points.add(current_point)

            if len(reversed_points) == n:
                break

    return zip(*reversed_points)
