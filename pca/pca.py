# steps:
# 1. Get the dataset
# 2. Convert dataset to matrix
# 3. normalize dataset so that points would be in a subspace
# 4. Get the SVD.
# 5. Seach for the 'elbow' to find optimal d
# 6. Calculate matrix Z
# 7. Plot values
# 8. Calculate approximated points

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math

# 1. Step
dataset = pd.read_csv("pca/abalone.csv")  # dataset taken from kaggle

# 2. Conversion to matrix
dataset = dataset.drop(
    ["Sex"], axis=1
)  # Otherwise we will get TypeError: unsupported operand type(s) for /: 'str' and 'int'

dataset = dataset.drop(["Viscera weight"], axis=1)
# were not in the first 2: Length, __Diameter, Height, Shucked weight, Shell weight
# were in the first 2: Whole weight, Rings, Viscera weight - changed a little

matrix = dataset.to_numpy()  # now each row is a column vector
# 3. Normalization
mu = np.sum(matrix, axis=0)
mu = mu / matrix.size

for i in range((int)(matrix.size / matrix[0].size)):
    matrix[i] -= mu

matrix = np.transpose(matrix)
print(matrix.shape)

# 4. Getting SVD of data:
U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
sing_val_energy_graph = "pca/energy_map.jpg"

# 5. Searching for the elbow
Y = list(map(lambda el: math.log(el), S))
plt.plot(range(0, len(S)), Y)
plt.savefig(
    sing_val_energy_graph
)  # should be a graph which is very high energetic only in the beginning because singular values are sorted descendingly.
plt.clf()  # from the picture we can take d = 3 or 4.

# 6. Calculating matrix Z = X_d - truncated matrix d
d = 3
X_d = (
    U[:, :d] @ (np.diag(S[:d])) @ Vh[:d]
)  # k-th column of X_d is approximation of data-point y


# 7. Plot values
features = np.diag(S[:d]) @ Vh[:d]
print(features.shape)  # new coordinates
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
max_no = features[0].size

ax.scatter(
    features[0, :max_no],
    features[1, :max_no],
    features[2, :max_no],
    color="b",
)
ax.set_xlabel("Coord in 1st eigenabalone")
ax.set_ylabel("Coord in 2nd eigenabalone")
ax.set_zlabel("Coord in 3rd eigenabalone")
plt.title("Compressed abalones 3d")
plt.savefig("pca/no_WW_3d.jpg")
plt.clf()
plt.scatter(features[0], features[1], marker="o", linestyle="-", color="b")
plt.xlabel("Coord in 1st eigenabalone")
plt.ylabel("Coord in 2nd eigenabalone")
plt.title("Compressed abalones 2d")
plt.savefig("pca/no_WW_2d.jpg")
plt.clf()
# 8. Calculating approximated data points. Add mu to every column of Z
print(X_d.shape[0])
for i in range(X_d.shape[0]):
    row_vector = np.full(
        X_d.shape[1], mu[i]
    )  # adds i-th coordinate of mean vector mu to all i-th coordinates of points at the same time
    X_d[i] += row_vector
