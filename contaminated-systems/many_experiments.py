import numpy as np
import cvxpy as cp
from scipy.linalg import orth, solve
import matplotlib.pyplot as plt
import numpy as np


def construct_PSI_from_PHI_star(PHI_star):
    # PHI: (n, m), maps ℝ^m → ℝ^n
    # col(PHI) ⊂ ℝ^n → want PSI: (n - r) x n, ker(PSI) = col(PHI)

    col_space = orth(PHI_star)
    Q, _ = np.linalg.qr(col_space, mode="complete")
    r = col_space.shape[1]
    PSI = Q[:, r:].T
    return PSI


# used to find vector x s.t. Ax = e and |x|_1 is minimal
def l1_minimisation_cvxpy(A, e):
    n = A.shape[1]
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm1(x))
    e = np.asarray(e).flatten()
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


def random_integer_vector(size, l, r):
    return np.random.randint(low=l, high=r + 1, size=size)


def fill_random_k_entries_uniform(v, k):
    v = np.asarray(v)
    w = np.zeros_like(v)
    random_indices = np.random.choice(len(v), k, replace=False)
    mu = 100  # mean
    sigma = 200  # standard deviation
    w[random_indices] = np.random.normal(loc=mu, scale=sigma, size=k)
    return w


"""
Measures how well original vector can be recovered when error-vector sparsity is k.
w, PHI, y stay the same throughout the experiment. Only error-vector e changes.
"""


def contaminated_system_experiment(k):
    global w
    global PHI_star
    global y

    # Generate sparse error vector e
    e = np.zeros(N)
    e = -pollute_top_k_entries(y, k)
    # Create the corrupted codeword c. Vector e pollutes top k entries of y
    c = y + e
    # Construct PSI. I it is enough to do it once since PHI doesn't change throughout the experiments.
    global PSI
    # PSI = construct_PSI_from_PHI(PHI_star)
    # Calculate PSI @ c
    PSI_c = PSI @ c
    # Recover sparse error vector
    e_guess = l1_minimisation_cvxpy(PSI, PSI_c)
    # Recover signal w by least squares
    w_guess, residuals, rank, s = np.linalg.lstsq(
        PHI_star, c - e_guess, rcond=None
    )  # variables residuals, rank, s will not be used
    # Calculate RMSE and relative RMSE
    rmse = np.sqrt(((w_guess - w) ** 2).mean())
    avg_entry = np.mean(np.abs(w))  # mean of absolute values
    rmse_relative = 100.0 * rmse / avg_entry
    rmse_error = np.sqrt(
        ((e_guess - e) ** 2).mean()
    )  # rmse of how well error-vector was being recovered
    avg_entry_e = np.mean(np.abs(e))
    rmse_relative_e = 100.0 * rmse / avg_entry_e
    return rmse, rmse_relative, rmse_error, rmse_relative_e


np.random.seed(42)
N = 500
d = 50
w = random_integer_vector(d, 10, 100)
low = -10
high = 10
PHI_star = np.random.uniform(
    low=low, high=high, size=(N, d)
)  # PHI_star in this case is PHI^T because everything is real
y = PHI_star @ w

PSI = construct_PSI_from_PHI_star(PHI_star)
col_norms = np.linalg.norm(
    PHI_star, axis=1
)  # take rows of PHI_star which are columns of PHI
print(PHI_star.shape)
print(col_norms.shape)
col_norms[col_norms == 0] = 1
PHI_normalized = PHI_star.T / col_norms
G = PHI_normalized.T @ PHI_normalized
# print(np.linalg.norm(PHI_normalized[:, :-1], axis=0)) # check normalized norms
off_diagonal = G.copy()
off_diagonal = np.abs(off_diagonal)  # get absolute values
np.fill_diagonal(off_diagonal, -np.inf)
mu = np.max(off_diagonal)
guarantee = (1 / (2 * d)) * (N + (N - d) / mu)
print(np.linalg.matrix_rank(PHI_star))
k_values = range(1, 21, 1)  # You can adjust range and step as you wish
rmses = []
relative_rmses = []
error_rmses = []
relative_rmses_e = []
for k in k_values:
    rmse, rmse_relative, rmse_error, rmse_relative_e = contaminated_system_experiment(k)
    rmses.append(rmse)
    relative_rmses.append(rmse_relative)
    error_rmses.append(rmse_error)
    relative_rmses_e.append(rmse_relative_e)

# =======PLOTTING=======
# Convert lists to numpy arrays for easier plotting
rmses = np.array(rmses)
relative_rmses = np.array(relative_rmses)
error_rmses = np.array(error_rmses)
relative_rmses_e = np.array(relative_rmses_e)

print(f"Worst-case coherence of PHI is {mu}")
print(
    f"Recovery of error vector e threshold for PSI is {1 / (2 * d) * (N + (N - d) / mu)}"
)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmses, marker="o", label="RMSE")
plt.plot(k_values, relative_rmses, marker="x", label="RMSE/avg.Entry (%)")
plt.plot(k_values, relative_rmses_e, marker="v", label="relative error vector RMSE")
plt.xlabel("Sparsity level k")
plt.ylabel("Error")
plt.title("Recovery Error vs. Sparsity Level")
plt.axvline(
    x=guarantee,
    color="red",
    linestyle="--",
    label=f"Recovery Threshold ≈ {guarantee:.2f}",
)

plt.legend()

plt.grid(True)
# plt.show()
plt.savefig("contaminated-systems/contaminated_systems_errors.png", dpi=300)
plt.plot(k_values, error_rmses, marker="^", label="error vector RMSE")
plt.savefig(
    "contaminated-systems/contaminated_systems_errors_with_error_vector.png", dpi=300
)
