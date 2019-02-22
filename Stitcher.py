import numpy as np
import skimage.transform as skt
import matplotlib.pyplot as plt

class Stitcher:

    def __init__(self, im1, im2):

        image1 = plt.imread(im1)
        image2 = plt.imread(im2)
        self.im1 = image1.mean(axis=2)
        self.im2 = image2.mean(axis=2)

    def convolve(self, g, h):

        im_out = np.zeros_like(g)
        u, v = g.shape[0], g.shape[1]
        h_off = int((h.shape[0] - 1) / 2)

        for x in range(h_off, u - h_off):
            for y in range(h_off, v - h_off):
                im_sub = g[x - h_off:x + h_off + 1, y - h_off:y + h_off + 1]
                val = np.sum(im_sub * h)
                im_out[x, y] = val

        return im_out

    def gaussian(self, h_size, sigma):

        out = np.zeros((h_size, h_size))
        h_off = h_size // 2
        for j in range(h_size):
            for k in range(h_size):
                m, n = j - h_off, k - h_off
                out[j, k] = np.exp(-(m * m + n * n) / (2 * sigma * sigma))

        return out / out.sum()

    def harris(self, im):

        gauss = self.gaussian(7, 2)
        sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        grad_u = self.convolve(im, sobel)
        grad_v = self.convolve(im, sobel.T)
        I_uu = self.convolve(grad_u * grad_u, gauss)
        I_vv = self.convolve(grad_v * grad_v, gauss)
        I_uv = self.convolve(grad_u * grad_v, gauss)
        H = (I_uu * I_vv - I_uv * I_uv) / (I_uu + I_vv + 1e-10)

        return H

    def loc_max(self, H):
        v_coords = []
        u_coords = []
        vals = []

        for v in range(1, H.shape[0] - 2):
            for u in range(1, H.shape[1] - 2):
                nh = [H[v, u], H[v - 1, u - 1], H[v - 1, u], H[v - 1, u + 1], H[v, u - 1], H[v, u + 1], H[v + 1, u + 1],
                      H[v + 1, u], H[v + 1, u + 1]]
                if nh[0] == max(nh):
                    v_coords.append(v)
                    u_coords.append(u)
                    vals.append(H[v, u])
        max_coords = np.zeros((len(v_coords), 3))
        max_coords[:, 0] = u_coords
        max_coords[:, 1] = v_coords
        max_coords[:, 2] = vals

        return max_coords

    def adaptive_suppression(self, max_coords, n=100, c=0.9):

        dist = []

        for i in range(max_coords.shape[0]):
            val = max_coords[i, 2]
            u1, v1 = max_coords[i, 0], max_coords[i, 1]
            d = np.inf

            for j in range(max_coords.shape[0]):
                u2, v2 = max_coords[j, 0], max_coords[j, 1]
                if u1 != u2 and v1 != v2:
                    val2 = max_coords[j, 2]
                    if c * val2 > val:
                        d2 = np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
                        if d2 < d:
                            d = d2

            dist.append(d)

        order = np.argsort(dist)
        order = order[-n:]
        u = np.asarray(max_coords[:, 0])
        v = np.asarray(max_coords[:, 1])

        return u[order], v[order]

    def descriptors(self, im, u, v, l=21):

        ofs = l // 2
        d_out = []
        u_out = []
        v_out = []
        # check for u and v to be same dimensions
        for i in range(len(u)):
            sub = im[v[i] - ofs:v[i] + ofs + 1, u[i] - ofs:u[i] + ofs + 1]
            if sub.shape[0] == l and sub.shape[1] == l:
                u_out.append(u[i])
                v_out.append(v[i])
                d_out.append(sub)

        return np.stack(d_out), np.asarray(u_out, dtype=int), np.asarray(v_out, dtype=int)

    def D_hat(self, d):
        return (d - d.mean()) / np.std(d)

    def error(self, d1, d2):
        return np.sum((d1 - d2) ** 2)

    def matching_thresh(self, d1, d2, u1, v1, u2, v2, r=0.6):
        kp1 = []
        kp2 = []

        for i in range(len(d1)):
            d1_hat = self.D_hat(d1[i])
            er_best = np.inf
            er_second = np.inf

            for j in range(len(d2)):
                d2_hat = self.D_hat(d2[j])
                err = self.error(d1_hat, d2_hat)
                if err < er_best:
                    er_best = err
                    kp2u = u2[j]
                    kp2v = v2[j]

            for k in range(len(d2)):
                if u2[k] != kp2u and v2[k] != kp2v:
                    d2_hat2 = self.D_hat(d2[k])
                    err2 = self.error(d1_hat, d2_hat2)
                    if err2 < er_second:
                        er_second = err2

            if er_best < r * er_second:
                if [u1[i], v1[i]] not in kp1:
                    if [kp2u, kp2v] not in kp2:
                        kp1.append([u1[i], v1[i]])
                        kp2.append([kp2u, kp2v])

        return np.asarray(kp1), np.asarray(kp2)

    def solve_for_H(self, uv, uv2):
        if uv.shape != uv2.shape:
            raise ValueError("X and X_prime must have matching shapes")
        if uv.shape[0] < 4:
            raise ValueError("Not enough points")

        # matches = np.column_stack(uv, uv2)

        A = np.zeros((2 * len(uv), 9))

        for i in range(len(uv)):
            A[2 * i, :] = [0, 0, 0, -uv[i, 0], -uv[i, 1], -1, uv2[i, 1] * uv[i, 0], uv2[i, 1] * uv[i, 1], uv2[i, 1]]
            A[2 * i + 1, :] = [uv[i, 0], uv[i, 1], 1, 0, 0, 0, -uv2[i, 0] * uv[i, 0], -uv2[i, 0] * uv[i, 1], -uv2[i, 0]]

        # print(A)
        U, Sigma, Vt = np.linalg.svd(A)

        H = Vt[-1, :].reshape((3, 3))

        return H

    def RANSAC(self, number_of_iterations, matches, n, r, d):  # matches is form [u1, v1, u2, v2]
        H_best = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        num_inliers = 0
        for i in range(number_of_iterations):
            # 1. Select a random sample of length n from the matches
            np.random.shuffle(matches)
            sub = matches[0:n, :]
            test = matches[n:, :]

            # 2. Compute a homography based on these points using the methods given above
            H = self.solve_for_H(sub[:, 0:2], sub[:, 2:])

            # 3. Apply this homography to the remaining points that were not randomly selected
            test_p = test[:, 0:2]
            test_p = np.column_stack((test_p, np.ones(len(test_p))))
            uv_p = (H @ test_p.T).T
            test_u = uv_p[:, 0] / uv_p[:, 2]
            test_v = uv_p[:, 1] / uv_p[:, 2]

            # 4. Compute the residual between observed and predicted feature locations
            R = np.zeros_like(test_u)
            for i in range(len(test_p)):
                R[i] = np.sqrt((test_u[i] - test[i, 2]) ** 2 + (test_v[i] - test[i, 3]) ** 2)

            # 5. Flag predictions that lie within a predefined distance r from observations as inliers
            inl = np.zeros_like(R)
            for i in range(len(inl)):
                if R[i] < r:
                    inl[i] = 1
                else:
                    inl[i] = 0
            num_inl = np.sum(inl)

            # 6. If number of inliers is greater than the previous best
            #    and greater than a minimum number of inliers d,
            #    7. update H_best
            #    8. update list_of_inliers
            if num_inl > num_inliers:
                if num_inl > d:
                    H_best = H
                    num_inliers = num_inl

        return H_best, num_inliers

