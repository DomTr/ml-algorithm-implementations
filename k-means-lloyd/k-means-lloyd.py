import numpy as np
import matplotlib.pyplot as plt
import random
import math

"""
k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster
(Wikipedia: https://en.wikipedia.org/wiki/K-means_clustering, looked at 2024 02 14)

This program aims to partition points into k clusters using k-means Lloyd algorithm.

The program generates 3 pictures. One is called initial_centers.png. Initial center points are cyan, their coordinates are set by default to be in range -100, 100.

Picture initial_points.png includes initial points, their coordinates are set by default to be in range -1000, 1000. The reason why center coordinates are not set to be from -1000, 1000 is because so that they could be better seen moving in the third picture.

Picture calculated_centers includes all generated points previously and black points which are the newly moved centers. Each black point is surrounded by a cluster of other points of same color. Previously generated center points can be seen in the area -100, 100, they are set to be cyan.
Algorithm works as follows:
1. Get points
2. Choose starting centers arbitrarily
3. Assign each point to a center
4. Update the centers
"""


# 1. Gets random points
def get_points(amnt, a, b):
    """Gets starting centers
    :param int amnt: number of points to be generated
    :param float: a - left bound of the interval
    :param float: b - right bound of the interbal
    :return: two lists for centers. First one is x coordinates, other is y coordinates
    """
    x, y = [], []
    for _ in range(int(amnt)):
        x.append(random.uniform(a, b))
        y.append(random.uniform(a, b))

    return x, y


# 2. Gets starting points, could be adapted to make starting points to have certain properties
def get_starting_centers(amnt, a, b):
    """Gets starting centers
    :param int amnt: number of points to be generated
    :param float: a - left bound of the interval
    :param float: b - right bound of the interbal
    :return: two lists for centers. First one is x coordinates, other is y coordinates
    """
    return get_points(amnt, a, b)


# Helper function to calculate Euclidean distances between two points
def calc_dist(x_a, y_a, x_b, y_b):
    """Calulate ditance between the points a and b.
    :param float x_a: x-coordinate of point a
    :param float y_a: y-coordinate of point a
    :param float x_b: x-coordinate of point b
    :param float y_b: y-coordinate of point b
    :return: 2d Euclidean distance between the points, float
    """
    return math.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)


# 3. Assigns each point to the nearest center.
def assign_to_center(x, y, center_x, center_y):
    """Assigns each point to the nearest center to it
    :param float list x: x-coordinate of regular points
    :param float list y: y-coordinate of regular points
    :param float list center_x: x-coordinates of center points
    :param float list center_y: y-coordinates of center points
    :return: dictionary with keys as id's of centers, center id is a number from 0 to K-1 (inclusive), K - generated amount of points. Values are lists of normal point ids.
    """
    center_dict = {i: list() for i in range(len(center_x))}
    for j in range(len(x)):
        x_a, y_a = x[j], y[j]
        minimum_cent = 0
        minimum_dist = 1e9
        for i in range(len(center_x)):
            curr_dist = calc_dist(x_a, y_a, center_x[i], center_y[i])
            if curr_dist < minimum_dist:
                minimum_dist = curr_dist
                minimum_cent = i

        center_dict[minimum_cent].append(j)

    return center_dict


# 4. Updates centers accoring to the points that have been assigned to them.
def update_centers(center_dict, x, y, center_x, center_y):
    """Updates centers accoring to the points that have been assigned to them. Updates are in-place, i.e. saved in the center_x and center_y lists.
    :param dict x: center dictionary returned by assign_to_center function
    :param float list x: x-coordinate of regular points
    :param float list y: y-coordinate of regular points
    :param float list center_x: x-coordinates of center points
    :param float list center_y: y-coordinates of center points
    """
    for i in range(len(center_x)):
        center_x[i] = 0
        center_y[i] = 0
        for j in center_dict[i]:
            center_x[i] += x[j]
            center_y[i] += y[j]

        if len(center_dict[i]) == 0:
            continue

        center_x[i] /= len(center_dict[i])
        center_y[i] /= len(center_dict[i])


# Helper method to get the difference of squared distance errors.
# This is needed for the algorithm to terminate if the error distances didn't change much after one algorithm iteration.
def get_curr_difference(center_dict, x, y, center_x, center_y):
    """Gets total error of points being far away from centers.
    :param dict x: center dictionary returned by assign_to_center function
    :param float list x: x-coordinate of regular points
    :param float list y: y-coordinate of regular points
    :param float list center_x: x-coordinates of center points
    :param float list center_y: y-coordinates of center points
    """
    sum = 0.0
    for i in center_dict:
        for j in center_dict:
            sum += calc_dist(x[j], y[j], center_x[i], center_y[i])

    return sum


# colors points in one particular color for each center. Centers are set to be black by default.
def color_code(center_dict, x, y, center_x, center_y):
    """Colors points which belong to circles. The clusters can be then seen better. The final center points are black.
    :param dict x: center dictionary returned by assign_to_center function
    :param float list x: x-coordinate of regular points
    :param float list y: y-coordinate of regular points
    :param float list center_x: x-coordinates of center points
    :param float list center_y: y-coordinates of center points
    """
    for c in center_dict:
        c_x, c_y = [], []
        for ind in center_dict[c]:
            c_x.append(x[ind])
            c_y.append(y[ind])

        plt.plot(c_x, c_y, "o")
    plt.scatter(center_x, center_y, color="black")
    plt.savefig("calculated_centers.png")


# Main algorithm
def lloyd_algorithm(iter, N=300, K=10):
    """Perfomrs Lloyd's k-means algorithm.
    :param int iter: maximal number of iterations to be perfomed
    :param int N: number of regular points
    :param int K: number of center points
    """
    x, y = get_points(N, -1000, 1000)
    center_x, center_y = get_starting_centers(K, -100, 100)
    last_difference = 1000000000.0
    curr_diff = 0.0
    center_dict = {}
    plt.scatter(center_x, center_y, color="cyan")
    plt.savefig("initial_centers.png")
    while iter > 0:
        center_dict = assign_to_center(
            x, y, center_x, center_y
        )  # assigns each point to the nearest center. Returns a dictionary where key is centering points' id and value is the id of a normal point
        # centering id's are counted from 0 to k-1
        update_centers(center_dict, x, y, center_x, center_y)
        curr_diff = get_curr_difference(center_dict, x, y, center_x, center_y)
        if (
            math.fabs(last_difference - curr_diff) < 0.0001
        ):  # delta can be bigger or smaller depending on the desired precision
            print(f"Stopped at {iter} iterations left")
            break
        else:
            iter -= 1
            last_difference = curr_diff

    plt.scatter(x, y, color="orange")
    plt.savefig("initial_points.png")
    color_code(center_dict, x, y, center_x, center_y)
    # plt.scatter(center_x, center_y, color="blue")
    # plt.savefig("calculated_centers.png")


lloyd_algorithm(1000)  # typically doesn't need that many iterations
