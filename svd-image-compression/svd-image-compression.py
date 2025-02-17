from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import PIL.ImageOps
import math
from functools import reduce


"""
1. Read the image of a dog +
2. Invert the image +
3. Find SVD of converted image +
4. Find number of non-zero singular values R +
5. Try plotting pictures of multiplied matrices with ranks R/50, R/25, R/2, R +
6. Plot the graph of y = log(sigma_j) vs x = j +
7. Plot sum of all singular values from 1 to m divided by sum of singular values from 1 to K
"""

# photo taken from: https://www.freepik.com/free-photo/black-schnauzer-park_13962500.htm#fromView=keyword&page=1&position=1&uuid=5ee9ccab-26a4-405c-b857-ceefce3ac04b&query=Dog+Forest
# https://www.istockphoto.com/photos/dog-in-snow


# converts to grayscale. Function taken from https://saturncloud.io/blog/how-to-convert-an-image-to-grayscale-using-numpy-arrays-a-comprehensive-guide/
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.2870, 0.1140])


image_path = "svd-image-compression/dog_photo.jpg"
img = imread(image_path)
gray = rgb2gray(img)  # somehow became a 2d array
# plt.imshow(gray, cmap=plt.get_cmap("gray"))
# plt.show()

# gets SVD of matrix gray. full_matrices=False means that U, Vh are of form MxK, and KxN, respectively. K = min(M, N). If full_matrices = True, then U and Vh are of form MxM, and NxN
U, S, Vh = np.linalg.svd(gray, full_matrices=False)

# Number of non-zero singular values. In this example there were no zero singular values probably because the image from the internet was compressed.
R = len(list(filter(lambda x: (x != 0), S)))
# print(R)
image_pathK = "svd-image-compression/normal.jpg"
image_pathReduced1 = "svd-image-compression/reduced1.jpg"
image_pathReduced2 = "svd-image-compression/reduced2.jpg"
image_pathReduced3 = "svd-image-compression/reduced3.jpg"

plt.imshow(gray, cmap=plt.get_cmap("gray"))
plt.savefig(image_pathK, bbox_inches="tight")
plt.clf()

# get image with first K/2 non-zero singular values
"""Debugging"""
# print(np.shape(U[:, : int(R / 2)]))
# print(np.shape(S[: int(R / 2)]))
# print(np.shape(Vh[: int(R / 2)]))
# print(len(U[: int(R / 2)]))
""""""
gray = (
    U[:, : int(R / 2)] @ (np.diag(S[: int(R / 2)])) @ Vh[: int(R / 2)]
)  # S is initially 1-d array, it needs to be a diagonal matrix
plt.imshow(gray, cmap=plt.get_cmap("gray"))
plt.savefig(image_pathReduced1, bbox_inches="tight")
plt.clf()

gray = U[:, : int(R / 25)] @ (np.diag(S[: int(R / 25)])) @ Vh[: int(R / 25)]
plt.imshow(gray, cmap=plt.get_cmap("gray"))
plt.savefig(image_pathReduced2, bbox_inches="tight")
plt.clf()

gray = U[:, : int(R / 50)] @ (np.diag(S[: int(R / 50)])) @ Vh[: int(R / 50)]
plt.imshow(gray, cmap=plt.get_cmap("gray"))
plt.savefig(image_pathReduced3, bbox_inches="tight")
plt.clf()


"""6. Plotting the graph log sigma_j"""
sing_val_energy_graph = "svd-image-compression/energy_map.jpg"
Y = list(map(lambda el: math.log(el), S))
plt.plot(range(0, len(S)), Y)
plt.savefig(
    sing_val_energy_graph
)  # should be a graph which is very high energetic only in the beginning because singular values are sorted descendingly.
plt.clf()

"""7. Plot sum of all singular values from 1 to m divided by sum of singular values from 1 to K"""
sing_val_sum_graph = "svd-image-compression/sing_val_sum_graph.jpg"
total_sum: float = reduce(lambda x, y: x + y, S)
Y = list()
curr_sum = 0
for sing_val in S:
    curr_sum += sing_val
    Y.append(curr_sum / total_sum)

plt.plot(range(0, len(S)), Y)
plt.savefig(
    sing_val_sum_graph
)  # should be a graph which increases rapidly in the beginning and then continues to but very slowly
plt.clf()  # clear current figure
