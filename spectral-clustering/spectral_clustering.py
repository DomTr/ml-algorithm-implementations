import numpy as np


def read_graphs(filename):
    with open(filename, "r") as f:
        n, m = map(int, f.readline().split())
        A = np.zeros(
            (n - 1, n - 1), dtype=int
        )  # n-1 because we take (p-1)x(p-1) matrix
        D = np.zeros((n - 1, n - 1), dtype=int)
        for _ in range(m):
            a, b = map(lambda x: int(x) - 1, f.readline().split())
            if A[a][b] == 0:
                D[a][a] += 1
                D[b][b] += 1
                A[a][b] = 1
                A[b][a] = 1

    return A, D


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
fiedler_vector = eigenvectors[:, 1]
p = 43

median = np.median(fiedler_vector)
cluster_1 = np.where(fiedler_vector <= median)[0] + 1
cluster_2 = np.where(fiedler_vector > median)[0] + 1

index = np.where(cluster_2 == p)[0]
if index.size > 0:
    cluster_2 = np.delete(cluster_2, index)  # Delete first occurrence only
else:
    print("Value not found.")


index = np.where(cluster_1 == p)[0]
if index.size > 0:
    cluster_1 = np.delete(cluster_1, index)  # Delete first occurrence only
else:
    print("Value not found.")

print("Fiedler vector first values:", fiedler_vector)
print("λ's:", eigenvalues[:5])  # should be PSD
print()
print("Cluster 1 nodes:", cluster_1)
print("Cluster 2 nodes: %s" % cluster_2)
print()

cluster_1_set = set(cluster_1)
cluster_2_set = set(cluster_2)

qresidues = sorted(quadratic_residues(p) - {43})
intersection1 = cluster_1_set.intersection(qresidues)
intersection2 = cluster_2_set.intersection(qresidues)
print()
if len(intersection1) < len(intersection2):
    intersection1, intersection2 = intersection2, intersection1

A = set(range(1, 43))  # Note: range is exclusive on the end
non_q_residues = A - set(qresidues)

print("Quadratic residues mod %d without 0: %s" % (p, qresidues))
print()
print(
    "Quadratic residues and cluster_1 intersection size: %d, %s"
    % ((len(intersection1), sorted(intersection1))),
)
intersection2 = cluster_2_set.intersection(non_q_residues)
print(
    "non-quadratic residues and cluster_2 intersection size: %d, %s"
    % ((len(intersection2), sorted(intersection2)))
)

print(intersection2)
size = (
    100.0 * (len(intersection1) + len(intersection2)) / (p - 1)
)  # p - 1 because elements from 1 to (p-1) are taken, i.e. [1,..,42]
print("Correctness: %d%%" % size)
"""
Quadratic residues mod 43:
43 (should be 0),
1, 4, 6, 9, 10, 11, 13, 14, 15, 16, 17, 21, 23, 24, 25, 31, 35, 36, 38, 40, 41
Cluster 1 nodes: [ 2  4  5  6  8  9 10 11 12 14 15 19 20 22 25 26 29 31 33 35 39]

Correct: 4, 6, 9, 10, 11, 14, 15, 25, 31, 35

2, 3, 5, 7, 8, 12, 18, 19, 20, 22, 26, 27, 28, 29, 30, 32, 33, 34, 37, 39, 42
Cluster 2 nodes: [ 1  3  7 13 16 17 18 21 23 24 27 28 30 32 34 36 37 38 40 41 42 43]

Correct: 3, 7, 18, 27, 28, 30, 32, 34, 37, 42
"""
