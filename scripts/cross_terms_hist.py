import numpy as np
from math import sqrt
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from math import log, sqrt
from itertools import compress
from collections import namedtuple
import logging
%matplotlib osx

# Constants
d = 10					# Dimension of random walk
N = 100					# Number of steps in random walk
num_trials = 10		# Number of trials to generate histogram
norms = np.zeros(N)		# List of norms of S over time
S = np.zeros((N+1,d))	# Random walk
BadDirection = namedtuple('BadDirection', ['q', 'x', 'w'])

logging.basicConfig(level=logging.WARNING)
# Turn off matplotlib's logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

def bitRound(t: float) -> int:
	if abs(t) < 1/2:
		return 0

	return -1 if t <= -1/2 else 1

def find_worst_direction(u: np.array, t=2) -> BadDirection:
	# This solves the convex program
	# 
	# max_{||z||_inf <= 1, |w| <= 1} min_{p in {-1, 0, 1}} || u + (w-p)z  ||_t
	# 
	# which we rewrite as
	#
	# max_(z, w, r) r
	# s.t. ||z||_inf <= 1,
	# |w| <= 1,
	# r <= || u + wz  ||_t
	# r <= || u + (w-1)z  ||_t
	# r <= || u + (w+1)z  ||_t

	BadDirection = namedtuple('BadDirection', ['q', 'x', 'w'])

	# These constraints implicitly force us to use {-1, 0, 1} as alphabet.
	def constraint_0(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + x[-2]*x[0:dim], t)

	def constraint_1(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + (x[-2]-1)*x[0:dim], t)

	def constraint_neg1(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + (x[-2]+1)*x[0:dim], t)

	def constraint_2norm(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# ||z||_2 <= 1.
		return np.linalg.norm(x[0:-2], 2) - 1

	def constraint_1norm(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# ||z||_1 <= 1.
		return np.linalg.norm(x[0:-2], 1) - 1

	dim = u.shape[0]
	x0 = np.zeros(dim + 2)
	x0[0:dim] = np.sign(np.random.randn(dim))
	x0[-2] = 2*(np.random.rand()-0.5)

	# We will stack all of our variables into one vector as [z, w, r]. z is our putative 
	# direction, w is the putative weight, and r is the t-norm of the residual.
	#
	# We place the following bounds on these variables
	#
	#	-1 <= z <= 1
	#	-1 <= w <= 1
	#	-inf < r < inf

	lb = -np.ones(dim+2)
	lb[-1] = -np.inf
	ub = np.ones(dim+2)
	ub[-1] = np.inf
	bounds = Bounds(lb, ub)

	# Below, we enforce that r <= min_{p} ||u + (w-p) z||_t
	# and ||z||_2 <= 1.

	constraints = [
		NonlinearConstraint(constraint_0, -np.inf, -10**(-15)),
		NonlinearConstraint(constraint_1, -np.inf, -10**(-15)),
		NonlinearConstraint(constraint_neg1, -np.inf, -10**(-15)),
		NonlinearConstraint(constraint_2norm, -2, -10**(-15))
		# NonlinearConstraint(constraint_1norm, -2, -10**(-15))

	]

	# Minimize the negative of r to get maximum.
	result = minimize(lambda x: -x[-1], x0, constraints=constraints, bounds=bounds)

	if result.success:
		logging.info("Sucessfully solved for adversarial step.")
		w = result.x[-2]
		x = result.x[0:dim]
		logging.info("w = {}".format(w))
		logging.info("x = {}".format(x))
		logging.info("u = {}".format(u))


		# Find out which bit was chosen.
		constraint_vals = [np.linalg.norm(u + (w+1)*x, t), np.linalg.norm(u + w*x, t), np.linalg.norm(u + (w-1)*x, t)]
		logging.debug("Residuals are {}".format([u + (w+1)*x, u + w*x, u + (w-1)*x]))
		logging.debug("Norms of residuals at optimum are {}".format(constraint_vals))
		q = constraint_vals.index(min(constraint_vals))-1
		logging.debug("q = {}".format(q))

		return BadDirection(q=q, x=result.x[0:dim], w=w)
	else:
		logging.critical("Unable to solve for adversarial step.")
		raise ValueError

def Quantize(w: float, u: np.array, X: np.array) -> int:
	return bitRound(w + np.dot(u, X))

# Generate num_trial draws of N directions in R^d.
# Pre-allocate a similarly sized tensor for the residuals.
X = np.random.randn(num_trials, N, d)
for i in range(num_trials):
	X[i, :, :] = np.array(list(map(lambda x: x/la.norm(x,2), X[i, :, :])))
U = np.zeros((num_trials, N+1, d))

# Generate num_trial draws of N weights in [-1,1]
W = 2*(np.random.rand(num_trials,N) - 1/2)

# Pre-allocate space for num_trials sequences of N bits.
Q = np.zeros((num_trials, N))

for i in range(num_trials):

	Q[i, 0] = bitRound(W[i,0]) 	# No previous residual, so just round first weight.
	U[i, 0, :] = (W[i, 0] - Q[i, 0])* X[i, 0, :]

	for j in range(1,N):

		Q[i, j] = Quantize(W[i, j], U[i, j-1, :], X[i, j, :])
		U[i,j, :] = U[i, j-1, :] + (W[i, j] - Q[i, j])*X[i, j, :]


##################################################################################
#
#	Histogramming Norms of Residuals in fixed dimension, varying N
#
##################################################################################


# dim_range = list(range(2,502, 50))
N_range = list(range(1, 10001, 100))
norms = np.zeros((len(dim_range), len(N_range)))

for idx, trial_N in enumerate(N_range):

	for i in range(num_trials):

		# Generate num_trial draws of N directions in R^d.
		# Pre-allocate a similarly sized tensor for the residuals.
		X = np.random.randn(num_trials, trial_N, dim)
		for i in range(num_trials):
			X[i, :, :] = np.array(list(map(lambda x: x/la.norm(x,2), X[i, :, :])))
		U = np.zeros((num_trials, N+1, dim))

		# Generate num_trial draws of N weights in [-1,1]
		W = 2*(np.random.rand(num_trials,N) - 1/2)

		# Pre-allocate space for num_trials sequences of N bits.
		Q = np.zeros((num_trials, N))

		Q[i, 0] = bitRound(W[i,0]) 	# No previous residual, so just round first weight.
		U[i, 0, :] = (W[i, 0] - Q[i, 0])* X[i, 0, :]

		for j in range(1,N):

			Q[i, j] = Quantize(W[i, j], U[i, j-1, :], X[i, j, :])
			U[i,j, :] = U[i, j-1, :] + (W[i, j] - Q[i, j])*X[i, j, :]

		norms[idx] += la.norm(U[i, N-1, :], 2)

norms = norms/num_trials
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(dim_range, norms, '-o')
ax.set_title("Expected $\|u_N\|_2^2$ As a Function of d. N = {N} and {num_trials} trials.".format(N=N, num_trials=num_trials))
ax.set_xlabel("dimension")
ax.set_ylabel("$\|u_{N}\|_2^2$")

##################################################################################
#
#	Histogramming 2 (wi - qi) u_{i-1}^T Xi
#
##################################################################################


values = np.zeros((num_trials, N))
for i in range(num_trials):

	# Generate num_trial draws of N directions in R^d.
	# Pre-allocate a similarly sized tensor for the residuals.
	X = np.random.randn(num_trials, N, d)
	for idx in range(num_trials):
		X[idx, :, :] = np.array(list(map(lambda x: x/la.norm(x,2), X[idx, :, :])))
	U = np.zeros((num_trials, N+1, d))

	# Generate num_trial draws of N weights in [-1,1]
	W = 2*(np.random.rand(num_trials,N) - 1/2)

	# Pre-allocate space for num_trials sequences of N bits.
	Q = np.zeros((num_trials, N))

	Q[i, 0] = bitRound(W[i,0]) 	# No previous residual, so just round first weight.
	U[i, 0, :] = (W[i, 0] - Q[i, 0])* X[i, 0, :]

	for j in range(1,N):

		Q[i, j] = Quantize(W[i, j], U[i, j-1, :], X[i, j, :])
		U[i,j, :] = U[i, j-1, :] + (W[i, j] - Q[i, j])*X[i, j, :]

		values[i, j] = 2 * (W[i, j] - Q[i,j]) * np.dot(U[i, j-1, :], X[i, j, :])

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.hist(values[0,:])
ax.set_title("Histogram of $2 (w_i - q_i) u^T X_i$. (N, d) = ({N}, {d}).".format(N=N, d=d))
ax.set_xlabel("$2 (w_i - q_i) u_{i-1}^T X_i$")

##################################################################################
#
#	Histogramming the Norms of the Residuals at Final Step for Various d
#
##################################################################################

# Average the norm of the N^th residual over num_trials trials.

dim_range = list(range(2,502, 50))
norms = np.zeros(len(dim_range))

for idx, dim in enumerate(dim_range):

	for i in range(num_trials):

		# Generate num_trial draws of N directions in R^d.
		# Pre-allocate a similarly sized tensor for the residuals.
		X = np.random.randn(num_trials, N, dim)
		for i in range(num_trials):
			X[i, :, :] = np.array(list(map(lambda x: x/la.norm(x,2), X[i, :, :])))
		U = np.zeros((num_trials, N+1, dim))

		# Generate num_trial draws of N weights in [-1,1]
		W = 2*(np.random.rand(num_trials,N) - 1/2)

		# Pre-allocate space for num_trials sequences of N bits.
		Q = np.zeros((num_trials, N))

		Q[i, 0] = bitRound(W[i,0]) 	# No previous residual, so just round first weight.
		U[i, 0, :] = (W[i, 0] - Q[i, 0])* X[i, 0, :]

		for j in range(1,N):

			Q[i, j] = Quantize(W[i, j], U[i, j-1, :], X[i, j, :])
			U[i,j, :] = U[i, j-1, :] + (W[i, j] - Q[i, j])*X[i, j, :]

		norms[idx] += la.norm(U[i, N-1, :], 2)

norms = norms/num_trials
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(dim_range, norms, '-o')
ax.set_title("Expected $\|u_N\|_2^2$ As a Function of d. N = {N} and {num_trials} trials.".format(N=N, num_trials=num_trials))
ax.set_xlabel("dimension")
ax.set_ylabel("$\|u_{N}\|_2^2$")

##################################################################################
#
#	Histogramming the Norms of the Residuals at Final Step for Various d
#					and for Shuffled Adversarial Paths
#
##################################################################################


# This is very expensive. Try to keep d small.

# Average the norm of the N^th residual over num_trials trials.

dim_range = list(range(10,561,50))
norms = np.zeros(len(dim_range))

for idx, dim in enumerate(dim_range):
	X = np.zeros((num_trials, N, dim))
	U = np.zeros((num_trials, N, dim))
	W = np.zeros((num_trials,N))
	Q = np.zeros((num_trials, N))

	for i in range(num_trials):

		# Generate adversarial weights and directions.
		for j in range(N):

			# Finds an adversarial direction.
			direction = find_worst_direction(U[i, j, :], t=2)

			W[i, j] = direction.w
			Q[i, j] = direction.q
			X[i, j, :] = direction.x
			if j > 0:
				U[i,j, :] = U[i, j-1, :] + (W[i, j] - Q[i, j])*X[i, j, :]
			else:
				U[i,j, :] = (W[i, j] - Q[i, j])*X[i, j, :]

		# Now, reset U and Q. Scramble X and W using the same permutation. Requantize.
		permutation = np.random.permutation(N)
		W[i, :] = W[i, permutation]
		X[i, :, :] = X[i, permutation, :]

		Q[i, 0] = bitRound(W[i,0]) 	# No previous residual, so just round first weight.
		U[i, 0, :] = (W[i, 0] - Q[i, 0])* X[i, 0, :]

		for j in range(1,N):

			Q[i, j] = Quantize(W[i, j], U[i, j-1, :], X[i, j, :])
			U[i,j, :] = U[i, j-1, :] + (W[i, j] - Q[i, j])*X[i, j, :]

		norms[idx] += la.norm(U[i, N-1, :], 2)

norms = norms/num_trials
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(dim_range, norms, '-o')
ax.set_title("Expected $\|u_N\|_2^2$ As a Function of d. N = {N} and {num_trials} trials.".format(N=N, num_trials=num_trials))
ax.set_xlabel("dimension")
ax.set_ylabel("$\|u_{N}\|_2^2$")

##################################################################################
#
#	Histogramming the Norms of the Residuals at Final Step for fixed d
#
##################################################################################
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle("Histogram of $\| u_{N} \|_2^2$ for $(N, d)=({N}, {d})$ and {num_trials} Trials".format(N=N, d=d, num_trials=num_trials))
max_norms = np.zeros(num_trials)
for l in range(num_trials):
	max_norms[l] = la.norm(U[l, N-1, :], 2)**2 
	
axes.hist(max_norms)


##################################################################################
#
#	Histogramming the Variance Term & Cross Term
#
##################################################################################
k = N-4
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle("$(N, d, k)=({N}, {d}, {k})$ and {num_trials} Trials".format(N=N, d=d, num_trials=num_trials, k=k-1, kmin1=k-2))
variance_terms = np.zeros(num_trials)
for l in range(num_trials):

	crossterms[l] = np.dot(2*(W[l,k]-Q[l,k])*X[l,k,:], U[l,k,:])
	variance_terms[l] = (W[l, k] - Q[l, k])**2

ax.hist(variance_terms, density=True, alpha=0.5)
ax.hist(crossterms, density=True, alpha=0.5)
ax.legend(['$(w_{k} - q_{k})^2$ ', '$2(w_{k}-q_{k}) u_{k}^TX_{k}$'])



##################################################################################
#
#	Histogramming the Individual Cross-term Summands
#
##################################################################################

# Choose an index k <= N-1 to histogram the values {(q_l - w_l)X_l^T q_k X_k} for l < k.
k = 8
fig, axes = plt.subplots(1, k, figsize=(k*5, 5), sharey=True, sharex=True)
fig.suptitle("Histogram of Individual Cross Terms with $(N, d)=({N}, {d})$ and {num_trials} Trials".format(N=N, d=d, num_trials=num_trials))
for l in range(k):
	if k == 1:
		ax = axes
	else:
		ax = axes[l]

	crossterms = np.zeros(num_trials)
	for j in range(num_trials):
		crossterms[j] = np.dot((Q[j, l] - W[j, l])*X[j, l, :], Q[j, k] * X[j, k, :])

	ax.hist(crossterms, density=True)
	ax.set_title("$(q_{l} - w_{l})X_{l}^T q_{k}X_{k}$".format(l=l, k=k))

