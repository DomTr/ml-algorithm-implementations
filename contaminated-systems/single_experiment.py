import numpy as np

from scipy.linalg import orth
import cvxpy as cp
from scipy.linalg import solve
from numpy import mean


def construct_PSI_from_PHI_star(PHI_star):
    # PHI: (n, m), maps ℝ^m → ℝ^n
    # col(PHI) ⊂ ℝ^n → want PSI: (n - r) x n, ker(PSI) = col(PHI)

    col_space = orth(PHI_star)  # col_space: (n, r)
    Q, _ = np.linalg.qr(col_space, mode="complete")  # Q: (n, n)
    r = col_space.shape[1]
    PSI = Q[:, r:].T  # shape: (n - r, n), in our case: 450x500
    print(PSI.shape)
    return PSI


# used to find vector x s.t. Ax = e and |x|_1 is minimal
def l1_minimisation_cvxpy(A, e):
    n = A.shape[1]
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm1(x))
    e = np.asarray(e).flatten()
    print(f"A shape: {A.shape}, x shape: {(n,)}, e shape: {e.shape}")

    constraints = [A @ x == e]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == cp.OPTIMAL:
        return x.value
    else:
        raise ValueError("Optimisation failed: " + problem.status)


"""
from a vector v creates another vector which has k biggest in absolute value entries of v in the same coordinates and 0s everywhere else 
(even though absolute values are compared, the original coordinate i.e. positive or negative, gets written)

This function isn't used now, but vector e recovery was more precise if no noise was added to the top entries.
"""


def keep_top_k_entries(v, k):
    v = np.asarray(v)
    w = np.zeros_like(v)
    top_k_indices = np.argpartition(np.abs(v), -k)[-k:]
    w[top_k_indices] = v[top_k_indices]
    return w


"""
Similarly, from a vector v creates another vector which has k biggest in absolute value entries of v in the same coordinates and 0s everywhere else. To the non-zero coordinates a random lo <= noise <= hi gets added 
"""


def pollute_top_k_entries(v, k):
    v = np.asarray(v)
    w = np.zeros_like(v)
    top_k_indices = np.argpartition(np.abs(v), -k)[-k:]
    w[top_k_indices] = v[top_k_indices]
    lo = -100.0
    hi = 100.0
    for index in range(0, v.size):
        w[index] -= np.random.uniform(low=lo, high=hi)

    return w


def fill_random_k_entries_uniform(v, k):
    v = np.asarray(v)
    w = np.zeros_like(v)
    random_indices = np.random.choice(len(v), k, replace=False)
    mu = 100  # mean
    sigma = 200  # standard deviation
    w[random_indices] = np.random.normal(loc=mu, scale=sigma, size=k)
    return w


def random_integer_vector(size, l, r):
    return np.random.randint(low=l, high=r + 1, size=size)


# 1. Generates the signal vector w of size 500
np.random.seed(42)  # Set seed for reproducibility
N = 500
d = 50
w = random_integer_vector(d, 10, 100)

# 2. Generates the matrix PHI_star (500x50)
low = -10
high = 10
PHI_star = np.random.uniform(low=low, high=high, size=(N, d))

# 3. Computes the vector y = PHI_star w
y = PHI_star @ w

# 4. Generates a sparse error vector e of size 500. e is k-sparse
# e = np.zeros(N)
k = 5
e = -pollute_top_k_entries(y, k)

# 5. Creates the code-word by adding the error vector to y (Reminder: y = PHI @ w)
c = y + e

# 6. Constructs the matrix PSI with ker(PSI) = col(PHI_star)
PSI = construct_PSI_from_PHI_star(PHI_star)
print(f"PHI_star shape: {PHI_star.shape}")
print(f"PSI shape: {PSI.shape}")
print(f"c shape: {c.shape}")

# 7. Calculates PSI @ c
PSI_c = PSI @ c

# 8. Recovers sparse error vector via L1 minimisation
e_guess = l1_minimisation_cvxpy(PSI, PSI_c)

# 9. Recovers original signal with Least Squares (overdetermined system). Variables residuals, rank, s will not be used later
w_guess, residuals, rank, s = np.linalg.lstsq(PHI_star, c - e_guess, rcond=None)

# 10. Calculate worst-case coherence of the matrix PHI:
col_norms = np.linalg.norm(
    PHI_star, axis=1
)  # take rows of PHI_star which are columns of PHI
print(PHI_star.shape)
print(col_norms.shape)
col_norms[col_norms == 0] = 1  # Avoid division by 0
PHI_normalized = PHI_star.T / col_norms  # Normalize
G = PHI_normalized.T @ PHI_normalized  # Gram matrix of PHI
off_diagonal = G.copy()
off_diagonal = np.abs(off_diagonal)
# print(off_diagonal) # check whether all entries non negative
np.fill_diagonal(off_diagonal, -np.inf)
mu = np.max(off_diagonal)  # maximal off-diagonal entry

guarantee = (1 / (2 * d)) * (N + (N - d) / mu)
print(f"Worst-case coherence of PHI is %.6f" % mu)
print(f"Matrix PSI can fully recover signal with l1 minimisation if s < {guarantee}")

avg_entry = np.mean(np.abs(w))
rmse = np.sqrt(((w_guess - w) ** 2).mean())
rmse_relative = 100.0 * rmse / avg_entry
rmse_error = np.sqrt(
    ((e_guess - e) ** 2).mean()
)  # rmse of how well error-vector was being recovered
avg_entry_e = np.mean(np.abs(e))
rmse_relative_e = 100.0 * rmse / avg_entry_e

# 10. Printing results in console
print(f"CVXPY L1 Recovery RMSE: {rmse}")
print("\n=== RESULTS ===")
print(f"Original signal vector w: (first 10 entries)\n{w[:10]}")
print(f"Recovered signal vector w*: (first 10 entries)\n{w_guess[:10]}")
avg_entry = np.mean(np.abs(w))
print(f"RMSE: {rmse}")
print(f"RMSE/length: {rmse_relative}%")
print(f"RMSE of vector e recovery: {rmse_error}")
print(f"RMSE of vector e/length: {rmse_relative_e}%")

# 11. Printing results in file. Relative path is "contaminated-systems/predicions.txt" because .txt file is saved in folder "contaminated-systems"
with open("contaminated-systems/predicions.txt", "w") as f:
    f.write("After normalisation, worst-case coherence (mu) of PHI is: %.6f\n" % mu)
    f.write("Guarantee of s-sparse recovery is %.6f\n" % guarantee)
    f.write(f"RMSE: {rmse}\n")
    f.write(f"RMSE/length: {rmse_relative}%\n")
    f.write(f"RMSE of vector e recovery: {rmse_error}\n")
    f.write(f"RMSE of vector e/length: {rmse_relative_e}%\n")
    f.write("=== PHI_star Matrix ===\n")
    f.write(np.array2string(PHI_star, precision=2, separator=", ") + "\n\n")
    f.write("=== Original Signal Vector w ===\n")
    f.write(np.array2string(w, separator=", ") + "\n\n")

    f.write("=== Recovered Signal Vector w_guess ===\n")
    f.write(
        np.array2string(
            w_guess,
            precision=0,
            formatter={"float_kind": lambda x: f"{int(round(x))}"},
            separator=", ",
        )
        + "\n\n"
    )

    f.write("=== Signal with error vector ===\n")
    f.write(np.array2string(y + e, precision=4, separator=", ") + "\n\n")

    f.write("=== Original Error Vector e ===\n")
    f.write("e is %d-sparse\n" % k)
    f.write(
        np.array2string(
            e,
            precision=4,
            formatter={"float_kind": lambda x: f"{int(round(x))}"},
            separator=", ",
        )
        + "\n\n"
    )

    f.write("=== Recovered Error Vector e_guess ===\n")
    f.write(
        np.array2string(
            e_guess,
            precision=4,
            formatter={"float_kind": lambda x: f"{int(round(x))}"},
            separator=", ",
        )
        + "\n\n"
    )
