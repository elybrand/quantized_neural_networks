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

%matplotlib osx

# logging.basicConfig(level=logging.DEBUG)
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
		NonlinearConstraint(constraint_neg1, -np.inf, -10**(-15))
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


# Set dimensions, number of data, and alphabet.
d = 2
N = 500
# Set frame smoothness to 1/d. Appears use for constraint constraint_nearby()
delta = 1/d
# Initialize residuals to zero.
X = np.zeros([N, d])
U = np.zeros([N+1, d])
q = np.zeros([N])
w = np.zeros([N])

# For setting the axes
walk_bd = 15
resid_bd = 5

# For each step, find a unit 2-norm direction which maximizes the 2-norm of the residual.
for i in range(N):
	# direction = find_worst_direction(U[i,:], t=2)
	if i == 0:
		direction = find_worst_direction(U[i,:], t=2)
	else:
		direction = find_nearby_worst_direction(U[i,:], X[(i-1),:], delta, t=2)
	w[i] = direction.w
	q[i] = direction.q
	X[i, :] = direction.x
	U[i+1, :] = U[i, :] + (w[i] - q[i])*X[i, :]

	# Sanity check: ||x||_inf <= 1, |w| < 1
	assert(np.max(np.abs(X[i, :])) <= 1)
	assert(np.abs(w[i]) <= 1)
	if i > 0:
		logging.debug("Frame variation is = {}".format(np.linalg.norm(X[i,:] - X[(i-1),:],2)))
		assert(np.linalg.norm(X[i,:] - X[(i-1),:],2) < delta + 10**(-4))

	fig, axes = plt.subplots(1,2, figsize=[10,5])
	fig.suptitle('Adversarial Neural Network Walk', fontsize=16)
	axes[0].set(xlim=(-walk_bd, walk_bd), ylim=(-walk_bd, walk_bd))
	axes[1].set(xlim=(-resid_bd, resid_bd), ylim=(-resid_bd, resid_bd))

	# For summing up rows.
	L = np.tril(np.ones([N,N]), k=0)
	walk = np.dot(L, np.dot(np.diag(w), X))
	q_walk = np.dot(L, np.dot(np.diag(q), X))
	axes[0].plot(walk[:,0], walk[:,1], '-x')
	axes[0].plot(q_walk[:,0], q_walk[:,1], '-o')
	axes[0].legend(['Analog Walk', 'Quantized Walk'])

	circle = plt.Circle((0,0), 2, color='green', alpha=0.2)
	axes[1].plot(walk[:,0]-q_walk[:,0], walk[:,1]-q_walk[:,1], '-o')
	axes[1].add_artist(circle)
	axes[1].legend(['Residual', 'Scaled L2-Ball'])
	if i == 0:
		input("Step {}. 2-norm of residual is {}. Click to continue.".format(i, np.linalg.norm(np.dot(w-q, X), 2)))
	else:
		input("Step {}. 2-norm of residual is {}. Frame variation is {}. Click to continue.".format(i, np.linalg.norm(np.dot(w-q, X), 2), np.linalg.norm(X[i,:] - X[(i-1),:],2)))
	plt.close()
