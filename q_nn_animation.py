import numpy as np
# import cvxpy as cp
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import log, sqrt
from typing import List
from itertools import compress
from collections import namedtuple
import logging
from matplotlib.animation import FuncAnimation

# %matplotlib osx

logging.basicConfig(level=logging.WARNING)
# Turn off matplotlib's logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

BadDirection = namedtuple('BadDirection', ['q', 'x', 'w'])

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

	# These constraints implicitly force us to use {-1, 0, 1} as alphabet.
	def constraint_0(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + x[-2]*x[0:d], t)

	def constraint_1(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + (x[-2]-1)*x[0:d], t)

	def constraint_neg1(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + (x[-2]+1)*x[0:d], t)

	def constraint_2norm(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# ||z||_2 <= 1.
		return np.linalg.norm(x[0:-2], 2) - 1

	def constraint_1norm(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# ||z||_1 <= 1.
		return np.linalg.norm(x[0:-2], 1) - 1


	x0 = np.zeros(d + 2)
	x0[0:d] = np.sign(np.random.randn(d))
	x0[-2] = 2*(np.random.rand()-0.5)

	# We will stack all of our variables into one vector as [z, w, r]. z is our putative 
	# direction, w is the putative weight, and r is the t-norm of the residual.
	#
	# We place the following bounds on these variables
	#
	#	-1 <= z <= 1
	#	-1 <= w <= 1
	#	-inf < r < inf

	lb = -np.ones(d+2)
	lb[-1] = -np.inf
	ub = np.ones(d+2)
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
		x = result.x[0:d]
		logging.info("w = {}".format(w))
		logging.info("x = {}".format(x))
		logging.info("u = {}".format(u))


		# Find out which bit was chosen.
		constraint_vals = [np.linalg.norm(u + (w+1)*x, t), np.linalg.norm(u + w*x, t), np.linalg.norm(u + (w-1)*x, t)]
		logging.debug("Residuals are {}".format([u + (w+1)*x, u + w*x, u + (w-1)*x]))
		logging.debug("Norms of residuals at optimum are {}".format(constraint_vals))
		q = constraint_vals.index(min(constraint_vals))-1
		logging.debug("q = {}".format(q))

		return BadDirection(q=q, x=result.x[0:d], w=w)
	else:
		logging.critical("Unable to solve for adversarial step.")
		raise ValueError

def find_nearby_worst_direction(u: np.array, x_prev: np.array, delta: float, t=2) -> BadDirection:
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

	# These constraints implicitly force us to use {-1, 0, 1} as alphabet.
	def constraint_0(x: np.array):
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + x[-2]*x[0:d], t)

	def constraint_1(x: np.array):
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + (x[-2]-1)*x[0:d], t)

	def constraint_neg1(x: np.array):
		# Given a vector of the form [z, w, r], enforces
		# r <= cp.norm(u + cp.multiply(w,z), 1)
		return x[-1] - np.linalg.norm(u + (x[-2]+1)*x[0:d], t)

	def constraint_nearby(x: np.array):
		# Force us to have a step close to the previous step.
		return np.linalg.norm(x[0:-2] - x_prev, 2)

	def constraint_2norm(x: np.array) -> float:
		# Given a vector of the form [z, w, r], enforces
		# ||z||_2 <= 1.
		return np.linalg.norm(x[0:-2], 2) - 1

	x0 = np.zeros(d + 2)
	x0[0:d] = np.sign(np.random.randn(d))
	x0[-2] = 2*(np.random.rand()-0.5)

	# We will stack all of our variables into one vector as [z, w, r]. z is our putative 
	# direction, w is the putative weight, and r is the t-norm of the residual.
	#
	# We place the following bounds on these variables
	#
	#	-1 <= z <= 1
	#	-1 <= w <= 1
	#	-inf < r < inf
	#
	lb = -np.ones(d+2)
	lb[-1] = -np.inf
	ub = np.ones(d+2)
	ub[-1] = np.inf
	bounds = Bounds(lb, ub)

	constraints = [
		NonlinearConstraint(constraint_0, -np.inf, -10**(-15)),
		NonlinearConstraint(constraint_1, -np.inf, -10**(-15)),
		NonlinearConstraint(constraint_neg1, -np.inf, -10**(-15)),
		NonlinearConstraint(constraint_2norm, -1, -10**(-15)),
		NonlinearConstraint(constraint_nearby, 0, delta-10**(-15))
	]

	# Minimize the negative of r to get maximum.
	result = minimize(lambda x: -x[-1], x0, constraints=constraints, bounds=bounds)

	if result.success:
		logging.info("Sucessfully solved for adversarial step.")
		w = result.x[-2]
		x = result.x[0:d]
		logging.info("w = {}".format(w))
		logging.info("x = {}".format(x))
		logging.info("u = {}".format(u))


		# Find out which bit was chosen.
		constraint_vals = [np.linalg.norm(u + (w+1)*x, t), np.linalg.norm(u + w*x, t), np.linalg.norm(u + (w-1)*x, t)]
		logging.debug("Residuals are {}".format([u + (w+1)*x, u + w*x, u + (w-1)*x]))
		logging.debug("Norms of residuals at optimum are {}".format(constraint_vals))
		q = constraint_vals.index(min(constraint_vals))-1
		logging.debug("q = {}".format(q))

		return BadDirection(q=q, x=result.x[0:d], w=w)
	else:
		logging.critical("Unable to solve for adversarial step.")
		raise ValueError

# def SigmaDeltaRule(u: np.array, )

# Set dimensions, number of data, and alphabet.
d = 2
N = 100
# Set frame smoothness to 1/d. Appears use for constraint constraint_nearby()
delta = 1/d

# Initialize residuals to zero.
X = np.zeros([N, d])
U = np.zeros([N+1, d])
q = np.zeros([N])
w = np.zeros([N])

# For setting the axes
walk_bd = 10
resid_bd = 5
step_bd = 2.5

# For each step, find a unit 2-norm direction which maximizes the 2-norm of the residual.
for i in range(N):

	# Finds an adversarial direction.
	# direction = find_worst_direction(U[i,:], t=2)

	# Finds an adversarial direction that is not too far from previous direction.	
	# if i == 0:
	# 	direction = find_worst_direction(U[i,:], t=2)
	# else:
	# 	direction = find_nearby_worst_direction(U[i,:], X[(i-1),:], delta, t=2)



	w[i] = direction.w
	q[i] = direction.q
	X[i, :] = direction.x
	U[i+1, :] = U[i, :] + (w[i] - q[i])*X[i, :]

	# Sanity check: ||x||_inf <= 1, |w| < 1
	assert(np.max(np.abs(X[i, :])) <= 1)
	assert(np.abs(w[i]) <= 1)

	# if i > 0:
	# 	logging.debug("Frame variation is = {}".format(np.linalg.norm(X[i,:] - X[(i-1),:],2)))
	# 	assert(np.linalg.norm(X[i,:] - X[(i-1),:],2) < delta + 10**(-4))

	# For summing up rows.
	L = np.tril(np.ones([N,N]), k=0)
	walk = np.dot(L, np.dot(np.diag(w), X))
	q_walk = np.dot(L, np.dot(np.diag(q), X))
	

fig, axes = plt.subplots(1,3, figsize=[15,5])
# fig, axes = plt.subplots(1,2, figsize=[10,5])
fig.suptitle('Adversarial Neural Network Walk', fontsize=16)
axes[0].set(xlim=(-walk_bd, walk_bd), ylim=(-walk_bd, walk_bd))
axes[1].set(xlim=(-resid_bd, resid_bd), ylim=(-resid_bd, resid_bd))
axes[2].set(xlim=(-step_bd, step_bd), ylim=(-step_bd, step_bd))

plt_walk, = axes[0].plot([], [], '-x')
plt_qwalk, = axes[0].plot([], [], '-o')
axes[0].legend(['Analog Walk', 'Quantized Walk'])

circle = plt.Circle((0,0), 2, color='green', alpha=0.2)
plt_resid, = axes[1].plot([], [], '-o')
axes[1].add_artist(circle)
axes[1].legend(['Residual'])

circle2 = plt.Circle((0,0), 1, color='pink', alpha=0.5)
plt_curr_direction, = axes[2].plot([], [], marker='o', markersize=10, color='green')
plt_prev_direction, = axes[2].plot([], [], marker='o', markersize=10, color='blue')
axes[2].add_artist(circle2)
axes[2].legend(['Current Step', 'Previous Step'])

lines = [plt_walk, plt_qwalk, plt_resid, plt_curr_direction, plt_prev_direction]
# lines = [plt_walk, plt_qwalk, plt_resid]

def update(frame):

	for i in range(len(lines)):
		vector = frame[i]
		line = lines[i]
		# print("Line = {}".format(line))
		data = np.array(line.get_xydata())
		if data.shape[0] == 0:
			data = vector
		else:
			# Augment the plot if it's the walk axis or the residual axis.	
			if i < len(lines) - 2:
				data = np.vstack((data, vector))
			# Update the directions axes by only the given vector.
			else:
				data = vector

			# data = np.vstack((data, vector))

		line.set_data(data.T)

	return lines

ani = FuncAnimation(fig, update, frames=list(zip(walk, q_walk, U, X, np.vstack((np.array([0,0]), X)))), repeat=False)
# ani = FuncAnimation(fig, update, frames=list(zip(walk, q_walk, U)), repeat=False)
plt.show()
# ani.save('./quant_nn_smooth_frame_15fps.gif', writer='imagemagick', fps=10)

# fig, ax = plt.subplots(1,1, figsize=[7,5])
# ax.plot(range(N+1), list(map(lambda x: np.linalg.norm(x, 2), U)))
# ax.set_title("Growth of L2 Norm of Residual")
# ax.set_ylabel("L2 Norm of Residual")
# ax.set_xlabel("Number of Data")
