import numpy as np


def read_graphs(filename):
    with open(filename, "r") as f:
        n, m = map(int, f.readline().split())
        A = np.zeros(
            (n - 1, n - 1), dtype=int
        )  # n-1 because we take (p-1)x(p-1) matrix. Adjacency matrix A.
        D = np.zeros((n - 1, n - 1), dtype=int)  # Degree matrix D
        for _ in range(m):
            a, b = map(lambda x: int(x) - 1, f.readline().split())
            # here we check whether edge {a, b} doesn't exist. If not, we can add it to the adjacency matrix A and degree matrix D.
            # This is done because the data in the text file may have duplicate edges, i.e. duplicates or edges of form (b, a) and (a, b).
            if A[a][b] == 0:
                D[a][a] += 1
                D[b][b] += 1
                A[a][b] = 1
                A[b][a] = 1

    return A, D


# returns a set of quadratic residues mod p
def quadratic_residues(p):
    residues = set()
    for x in range(p):
        residue = (x * x) % p
        if residue == 0:
            residue = p
        residues.add(residue)
    return residues


A, D = read_graphs("spectral-clustering/graph.txt")
L = D - A

eigenvalues, eigenvectors = np.linalg.eigh(L)
fiedler_vector = eigenvectors[
    :, 1
]  # eigenvector associated with the 2nd smallest eigenvalue of the Laplacian
p = 43

median = np.median(fiedler_vector)
cluster_1 = np.where(fiedler_vector <= median)[0] + 1
cluster_2 = np.where(fiedler_vector > median)[0] + 1
print(
    "Fiedler (2nd smallest eigenvalue of Laplacian) values:\n",
    fiedler_vector,
)
k = 5
print(f"First {k} λ's:", eigenvalues[:k])  # should be PSD
print()
print("Cluster 1 nodes: %s" % cluster_1)
print("Cluster 2 nodes: %s" % cluster_2)
print()

cluster_1_set = set(cluster_1)
cluster_2_set = set(cluster_2)

qresidues = sorted(quadratic_residues(p) - {43})
intersection1 = cluster_1_set.intersection(qresidues)
intersection2 = cluster_2_set.intersection(qresidues)
print()
if len(intersection1) < len(intersection2):
    cluster_1_set, cluster_2_set = (
        cluster_2_set,
        cluster_1_set,
    )  # swap cluster sets. cluster_1_set will contain the most quadratic residues
    intersection1, intersection2 = intersection2, intersection1

A = set(range(1, 43))  # Note: range is exclusive on the end
non_q_residues = A - set(qresidues)

print("Quadratic residues mod %d without 0: %s\n" % (p, qresidues))
print(
    "Quadratic residues and cluster_1 intersection size: %d, %s\n"
    % ((len(intersection1), sorted(intersection1))),
)
print("Non-quadratic residues mod %d: %s\n" % (p, non_q_residues))
intersection2 = cluster_2_set.intersection(non_q_residues)
print(
    "Non-quadratic residues and cluster_2 intersection size: %d, %s\n"
    % ((len(intersection2), sorted(intersection2)))
)
size = (
    100.0 * (len(intersection1) + len(intersection2)) / (p - 1)
)  # p - 1 because elements from 1 to (p-1) are taken, i.e. [1,..,42]
print("Correctness: %d%%" % size)
"""
Fiedler (2nd smallest eigenvalue of Laplacian) values:
 [ 0.03074237  0.0949252   0.05279351 -0.80643523 -0.01312542 -0.09117854
  0.092252    0.03864372 -0.10273157  0.03103447 -0.26205158  0.07955464
 -0.01506053 -0.05950809 -0.26659872  0.05606123  0.02046452  0.01822129
  0.06629396  0.00498447  0.04651068 -0.06895955  0.03046349  0.00240356
  0.2838955  -0.00204233  0.0360725   0.04244471  0.04645404  0.05635425
 -0.01027148  0.02224451  0.14222642  0.01041944  0.14313155  0.00989641
  0.03357128  0.04680147  0.05649462  0.02078719  0.03560347  0.04621655]
First 5 λ's: [4.85866453e-16 6.75049442e-01 7.77347475e-01 8.26465032e-01
 9.14082387e-01]

Cluster 1 nodes: [ 1  4  5  6  9 11 13 14 15 17 18 20 22 23 24 26 31 32 34 36 40]
Cluster 2 nodes: [ 2  3  7  8 10 12 16 19 21 25 27 28 29 30 33 35 37 38 39 41 42]


Quadratic residues mod 43 without 0: [1, 4, 6, 9, 10, 11, 13, 14, 15, 16, 17, 21, 23, 24, 25, 31, 35, 36, 38, 40, 41]

Quadratic residues and cluster_1 intersection size: 14, [1, 4, 6, 9, 11, 13, 14, 15, 17, 23, 24, 31, 36, 40]

Non-quadratic residues mod 43: {2, 3, 5, 7, 8, 12, 18, 19, 20, 22, 26, 27, 28, 29, 30, 32, 33, 34, 37, 39, 42}

Non-quadratic residues and cluster_2 intersection size: 14, [2, 3, 7, 8, 12, 19, 27, 28, 29, 30, 33, 37, 39, 42]

Correctness: 66%
"""
