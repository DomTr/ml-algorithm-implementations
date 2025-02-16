This is a k-means Lloyd algorithm impelementation in python. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster
(Wikipedia: https://en.wikipedia.org/wiki/K-means_clustering, looked at 2024 02 14)

This program aims to partition points into k clusters using k-means Lloyd algorithm.

The program generates 3 pictures. One is called initial_centers.png. Initial center points are cyan, their coordinates are set by default to be in range -100, 100.

Picture initial_points.png includes initial points, their coordinates are set by default to be in range -1000, 1000. The reason why center coordinates are not set to be from -1000, 1000 is because so that they could be better seen moving in the third picture.

Picture calculated_centers includes all generated points previously and black points which are the newly moved centers. Each black point is surrounded by a cluster of other points of same color. Previously generated center points can be seen in the area -100, 100, they are set to be cyan.