import Stitcher
import numpy as np
import skimage.transform as skt
import matplotlib.pyplot as plt


im1 = 'room1.jpg'
im2 = 'room2.jpg'

st = Stitcher.Stitcher(im1, im2)

# get harris response for each image
h1 = st.harris(st.im1)
h2 = st.harris(st.im2)

# get local maxima from each harris response
locmax1 = st.loc_max(h1)
locmax2 = st.loc_max(h2)

# obtain highest values from local maxima using adaptive maximal suppression method
u1, v1 = st.adaptive_suppression(locmax1)
u2, v2 = st.adaptive_suppression(locmax2)
u1, v1 = np.asarray(u1, dtype=int), np.asarray(v1, dtype=int)
u2, v2 = np.asarray(u2, dtype=int), np.asarray(v2, dtype=int)

# extract image descriptors for each point and threshold to obtain keypoints
d1, d_u1, d_v1 = st.descriptors(st.im1, u1, v1)
d2, d_u2, d_v2 = st.descriptors(st.im2, u2, v2)

# match keypoints in the two images
m1, m2 = st.matching_thresh(d1, d2, d_u1, d_v1, d_u2, d_v2)

# change this later
# out_im = np.concatenate((st.im1, st.im2), axis=1)
# plt.figure(figsize=(15, 12))
# plt.imshow(out_im, cmap='gray')
# for i in range(len(m1)):
#     plt.plot(m1[i, 0], m1[i, 1], 'r.')
#     plt.plot(m2[i, 0] + st.im1.shape[1], m2[i, 1], 'r.')
#     plt.plot([m1[i,0], m2[i,0]+st.im1.shape[1]],[m1[i,1],m2[i,1]])

# plt.show()

matches = np.column_stack((m1, m2))
H_best, inl = st.RANSAC(10, matches, 10, 3, 8)  # may want to play with these params

# Create a projective transform based on the homography matrix $H$
proj_trans = skt.ProjectiveTransform(H_best)

# Warp the image into image 1's coordinate system
image_2_transformed = skt.warp(st.im2, proj_trans, output_shape=(st.im1.shape[0], st.im1.shape[1]+(int(st.im1.shape[1]*0.4))))  # have to manually make it wide enough....

im1t = np.zeros_like(image_2_transformed)
for v in range(st.im1.shape[0]):
    for u in range(st.im1.shape[1]):
        if st.im1[v,u] != 0.:
            im1t[v,u] = st.im1[v,u]

out_array = np.zeros_like(image_2_transformed)

for v in range(out_array.shape[0]):
    for u in range(out_array.shape[1]):
        if im1t[v,u] == 0. and image_2_transformed[v,u] == 0.:
            out_array[v,u] = 0.
        elif im1t[v,u] != 0. and image_2_transformed[v, u] == 0.:
            out_array[v, u] = st.im1[v,u]
        elif im1t[v,u] == 0. and image_2_transformed[v,u] != 0.:
            out_array[v,u] = image_2_transformed[v,u]
        else:
            out_array[v, u] = (image_2_transformed[v, u] + st.im1[v, u])/2

plt.imshow(out_array)
plt.show()
