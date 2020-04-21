import numpy as np
from typing import List, Tuple, Optional
from math import sqrt, log
import scipy.linalg as la
import scipy.stats as stats
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.special import gamma
from itertools import compress
from collections import namedtuple
import logging
from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation
from itertools import product
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import RandomUniform
from keras.backend import function as Kfunction
import logging
from sys import stdout
from quantized_network import QuantizedNeuralNetwork
%matplotlib osx

logging.basicConfig(stream=stdout)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

BadDirection = namedtuple('BadDirection', ('q', 'x', 'w'))
ExperimentResult = namedtuple('ExperimentResult', ('N', 'd', 'num_trials', 'result'))
Walk = namedtuple('Walk', ('X', 'w', 'q', 'U'))

class Walk(ABC):
	def __init__(self, N: int, d: int):
		
		# Pre-allocate memory for fields.
		self.X = np.zeros((N,d))
		self.w = np.zeros(N)
		self.q = np.zeros(N)
		self.U = np.zeros((N,d))

	def bit_round(self, t: float) -> int:
		if abs(t) < 1/2:
			return 0
		return -1 if t <= -1/2 else 1

	def quantize_weight(self, w: float, u: np.array, X: np.array) -> int:
		return self.bit_round(w + np.dot(u, X)/la.norm(X,2)**2)

	def quantize(self, u_init: Optional[np.array]=None):

		N, d = self.X.shape

		q = np.zeros(N)
		U = np.zeros((N, d))

		# Start with a residual vector of 0.
		if u_init is None:
			u_init = np.zeros(d)
		q[0] = self.quantize_weight(self.w[0], u_init, self.X[0,:])
		U[0,:] = u_init + (self.w[0] - q[0]) * self.X[0,:]

		for i in range(1,N):
			q[i] = self.quantize_weight(self.w[i], U[i-1,:], self.X[i,:])
			U[i,:] = U[i-1, :] + (self.w[i] - q[i]) * self.X[i,:]

		self.q = q
		self.U = U

	def plot(self):

		# Bounds for the axes.
		walk_bd = 10
		resid_bd = 5

		fig, axes = plt.subplots(1,2, figsize=[10,5])
		fig.suptitle('Adversarial Neural Network Walk', fontsize=16)
		axes[0].set(xlim=(-walk_bd, walk_bd), ylim=(-walk_bd, walk_bd))
		axes[1].set(xlim=(-resid_bd, resid_bd), ylim=(-resid_bd, resid_bd))

		plt_walk, = axes[0].plot([], [], '-x')
		plt_qwalk, = axes[0].plot([], [], '-o')
		axes[0].legend(['Analog Walk', 'Quantized Walk'])

		circle = plt.Circle((0,0), 2, color='green', alpha=0.2)
		plt_resid, = axes[1].plot([], [], '-o')
		axes[1].add_artist(circle)
		axes[1].legend(['Residual'])

		# Create the lines object for the update function call to update.
		lines = [plt_walk, plt_qwalk, plt_resid]

		def update(frame):

			for i in range(len(lines)):
				vector = frame[i]
				line = lines[i]
				data = np.array(line.get_xydata())
				if data.shape[0] == 0:
					data = vector
				else:
					data = np.vstack((data, vector))
				line.set_data(data.T)
			return lines

		# For summing up rows.
		L = np.tril(np.ones([N,N]), k=0)
		walk = np.dot(L, np.dot(np.diag(self.w), self.X))
		q_walk = np.dot(L, np.dot(np.diag(self.q), self.X))
	
		ani = FuncAnimation(fig, update, frames=list(zip(walk, q_walk, self.U)), repeat=False)
		# plt.show()
		return ani

class RandomWalk(Walk):
	def __init__(self, N: int, d: int, u_init: Optional[np.array]=None):
		"""
		Constructs a random walk where the N directions are drawn i.i.d. from the uniform
		distribution on S^(d-1). The weights are drawn i.i.d. the uniform distribution on
		[-1, 1].
		"""

		super().__init__(N, d)

		# First, draw N d-dimensional Gaussians, then normalize.
		X = np.random.randn(N, d)
		X = np.array(list(map(lambda x: x/la.norm(x,2), X)))
		w = 2*(np.random.rand(N) - 1/2)
		

		self.X = X
		self.w = w
		self.quantize(u_init=u_init)

class AdversarialWalk(Walk):
	def __init__(self, N: int, d: int):
		"""
		Constructs a walk where the direction at step i is a unit vector orthogonal
		to the residual at step i-1. The weights are chosen to be 0.501 multiplied by 
		a random sign.
		"""
		super().__init__(N,d)
		self.quantize()

	def find_orthogonal_direction(self, u: np.array) -> np.array:
		"""
		Returns a unit vector that is orthogonal to u.
		"""
		return la.null_space(np.array([u]))[:,0]

	def quantize(self):
		
		X = np.zeros((N, d))
		random_signs = 2*(np.random.binomial(1,0.5, N)-0.5)
		w = 0.501 * random_signs
		q = np.zeros(N)
		U = np.zeros((N, d))

		# Start with a residual vector of 0 and a random unit vector.
		u_init = np.zeros(d)
		X[0,:] = np.random.randn(d)
		X[0,:] = X[0,:]/la.norm(X[0,:],2)
		q[0] = super().quantize_weight(w[0], u_init, X[0,:])
		U[0,:] = u_init + (w[0] - q[0]) * X[0,:]

		for i in range(1,N):
			X[i,:] = self.find_orthogonal_direction(U[i-1,:])
			q[i] = super().quantize_weight(w[i], U[i-1,:], X[i,:])
			U[i,:] = U[i-1, :] + (w[i] - q[i]) * X[i,:]

		self.X = X
		self.w = w
		self.U = U
		self.q = q

class CustomWalk(Walk):
	def __init__(self, X: np.array, w: np.array, u_init: Optional[np.array]=None):

		N, d = X.shape
		super().__init__(N, d)

		self.X = X
		self.w = w
		super().quantize(u_init=u_init)

class HiddenLayerWalk(Walk):\

		def __init__(self, wX: np.array, qX: np.array, w: np.array, u_init: Optional[np.array]=None):
		"""
		Constructs two random walks, one with directions wX the other with qX, and step parameters w_t.
		"""
			N, d  = wX.shape
			self.wX = wX
			self.qX = qX
			self.w = w
			self.q = np.zeros(N)
			self.U = np.zeros((N,d))

		def quantize_weight(self, w: np.float, u: np.array, X: np.array, X_tilde: np.array):
			self.bit_round(np.dot(X_tilde, u + w*X)/(la.norm(X_tilde,2)**2))

		def quantize(self, u_init: Optiona[np.array]=None):

			N, d = self.U.shape
			if u_init is None:
				u_init = np.zeros(N)
			self.q[0] = self.quantize_weight(self.w[0], u_init, self.wX[0,:], self.qX[0,:])
			self.U[0,:] = u_init + self.w[0]*self.wX[0,:] - self.q[0]*self.qX[0,:]

			for t in range(1,N):
				self.q[t] = self.quantize_weight(self.w[t], self.U[t-1,:], self.wX[t,:], self.qX[t,:])
				self.U[t,:] = self.U[t-1,:] + self.w[t]*self.wX[t,:] - self.q[t]*self.qX[t,:]
			

# Function wrapper to display what tuple (N,d) a given experiment is on.
def display_progress(func):
	def display_and_call(*args, **kwargs):
		for kw in kwargs:
			print(kw + '=' + str(kwargs[kw]) + ', ', endline='')
		print('\n')

def main():
	N = 100000
	d = 2
	lw = 4
	num_bins = 100
	walk = RandomWalk(N,d)
	cross_terms = np.array([0] + [2*(walk.w[i] - walk.q[i]) * np.dot(walk.X[i], walk.U[i-1]) for i in range(1,N)])
	update_minus_dither = np.array([walk.q[0] - walk.w[0]] + [np.dot(walk.X[i], walk.U[i]) - np.dot(walk.X[i+1], walk.U[i]) for i in range(0,N-1)])
	XU_terms = np.array([0] + [np.dot(walk.X[i], walk.U[i-1]) for i in range(1,N)])
	variance_terms = np.array([(walk.w[i] - walk.q[i])**2 for i in range(N)])

	# axes[1].set_title("Update Term Minus Residual Term", fontsize=22)
	# axes[1].set_ylabel('$u_i^T X_i - u_i^T X_{i+1}$', fontsize=15)
	# axes[1].set_xlabel('$i$', fontsize=15)


	fig, axes = plt.subplots(1,2,figsize=(25,10))
	fig.suptitle("Exploratory Plots for Quantized Random Walk with (N,d) = ({N}, {d})".format(N=N, d=d), fontsize=24)
	axes[0].plot(np.arange(N), cross_terms, alpha=0.5, linewidth=lw)
	axes[0].plot(np.arange(N), variance_terms, alpha=0.5, linewidth=lw)
	axes[0].plot(np.arange(N), variance_terms + cross_terms, linewidth=lw)
	axes[0].set_title("Terms in Expansion of $||u_N||^2$", fontsize=22)
	axes[0].legend(['$2(w_i - q_i)X_i^Tu_{i-1}$', '$(w_i - q_i)^2$', '$(w_i - q_i)^2 + 2(w_i - q_i)X_i^Tu_{i-1}$'], fontsize=15)
	axes[0].set_ylim([-2, 2])

	axes[1].hist(cross_terms, alpha=0.5, density=True, bins=num_bins)
	axes[1].hist(variance_terms, alpha=0.5, density=True, bins=num_bins)
	axes[1].hist(cross_terms + variance_terms, alpha=0.65, density=True, bins=num_bins)
	axes[1].set_title("Terms in Expansion of $||u_N||^2$", fontsize=22)
	axes[1].set_xlim([-2, 2])
	axes[1].set_ylim([0,20])
	axes[1].legend(['$2(w_i - q_i)X_i^Tu_{i-1}$', '$(w_i - q_i)^2$', '$(w_i - q_i)^2 + 2(w_i - q_i)X_i^Tu_{i-1}$'], fontsize=15)



	# pos_summands = variance_terms + cross_terms > 0
	# neg_summands = np.logical_and(variance_terms + cross_terms < 0, variance_terms < 1)

	# axes[1,0].plot(range(N), walk.w)
	# axes[1,0].set_title("Weights", fontsize=22)
	# axes[1,0].set_ylabel("$w_i$", fontsize=18)
	# # axes[1].set_xlabel("i", fontsize=18)

	# axes[2,0].plot(range(N), walk.q)
	# axes[2,0].set_title("Bits", fontsize=22)
	# axes[2,0].set_ylabel("$q_i$", fontsize=18)
	# axes[2,0].set_xlabel("i", fontsize=18)

	# fig, ax = plt.subplots(1,1,figsize=(10,10))
	# fig.suptitle("Residuals of Adversarial Walk and Shuffled Walk", fontsize=24)
	# ax.plot(range(N), [la.norm(mywalk.U[i,:]) for i in range(N)], '-o')
	# ax.plot(range(N), [la.norm(newwalk.U[i,:]) for i in range(N)], '-o')
	# ax.legend(['Adversarial', 'Shuffled'], fontsize=20)
	# ax.set_ylabel("$||u_{i}||^2$", fontsize=20)
	# ax.set_xlabel("i", fontsize=20)

def distribution_of_wqX():
	# Fix a u_i. Draw w_i+1, X_i+1 and plot the distribution of
	# (w_i+1 - q_i+1)X_i+1.
	N = 100
	d = 2
	nframes = 10
	num_trials = 100000
	walk = RandomWalk(N,d)
	ui = np.array([3,0])
	vects = np.zeros((num_trials, d))
	inner_prods = np.zeros(num_trials)
	means = np.zeros(num_trials)
	for i in range(num_trials):
			# w = 2*(np.random.rand()-1/2)
			w = -1 # * 2*(np.random.binomial(1, 0.5)-0.5)
			x = np.random.randn(d)
			x = x/la.norm(x,2)
			q = walk.quantize_weight(w, ui, x)
			vects[i] = (w - q) * x
			inner_prods[i] = np.dot(ui, (w-q)*x)
			means[i] = (w - q)**2 + 2*(w-q)*np.dot(x, ui)

	fig, ax = plt.subplots(1,1, figsize=(10,10))
	ax.hist(means)
	ax.set_title("$(w_i - q_i)^2 + 2(w_i - q_i) X_i^T u_{{i-1}}$ Given $u_{{i-1}} = ({u1}, {u2})$, $w_i = -1$".format(u1=ui[0], u2=ui[1]))

	# # Now, map the uniform distribution over [0, ||u_{i-1}||] x [0, 2pi] using the parametrization
	# #
	# #	x(r, theta) = 2 (0.5 + r)(1 - cos(theta))*cos(theta)
	# #	y(r, theta) = 2 (0.5 + r)(1 - cos(theta))*sin(theta)

	# # Uniform over the whole rectangular domain.
	# preimage_pts = np.array(
	# 		[np.array(
	# 				[
	# 					np.random.uniform(0, la.norm(ui,2)),
	# 					np.random.uniform(-np.pi, np.pi)
	# 				]
	# 			) for i in range(num_trials)]
	# 	)

	# # This gives the cardioid when the foci is on the x-axis.
	# uniform_pts = np.array(
	# 				[ np.array([
	# 					0.75*r*(1-np.cos(theta))*np.cos(theta),
	# 					0.75*r*(1-np.cos(theta))*np.sin(theta)
	# 					]) 
	# 				for (r, theta) in preimage_pts]
	# 				)

	# # Apply the rotation matrix which takes e1 to ui/||ui|| to get the true cardioid, viz.
	# #
	# #	[ ui[0]  -ui[1] ]
	# #	[ ui[1]   ui[0] ]
	# rotation = 1/la.norm(ui) * np.vstack((
	# 		np.array( [ui[0], -ui[1]] ),
	# 		np.array( [ui[1], ui[0] ] )
	# 		))
	# cardioid_pts = np.dot(uniform_pts, rotation.T)

	fig, axes = plt.subplots(1,2,figsize=(20,10))
	circle = plt.Circle((0,0), 1, color='green', alpha=0.2)
	circle2 = plt.Circle((0,0), 2, color='pink', alpha=0.2)
	axes[0].scatter(vects[:,0], vects[:,1], marker='o', color='blue', alpha=0.5)
	# axes.scatter(cardioid_pts[:,0], cardioid_pts[:,1], marker='o', color='orange', alpha=0.5)
	axes[0].scatter(ui[0], ui[1], marker='o', color='red', linewidth=10)
	axes[0].add_artist(circle2)
	axes[0].add_artist(circle)
	axes[0].set_xlim([-2.5,2.5])
	axes[0].set_ylim([-2.5,2.5])
	axes[0].legend(['$(w_i - q_i)X_i$', 'Image of Uniform Distribution Under Cardioid', '$u_{i-1}$'])
	axes[0].set_title("Empirical Conditional Distribution of $(w_i - q_i)X_i$ Given $u_{{i-1}} = ({u1}, {u2})$ with ${num_trials}$ Samples".format(num_trials=num_trials,
		u1=round(ui[0],3), u2=round(ui[1],3)))


	axes[1].hist(inner_prods, density=True, bins=25)
	axes[1].set_xlim([-201, 1])
	axes[1].set_title("Empirical Conditional Distribution of $(w_i - q_i)X_i^T u_{{i-1}}$ Given $u_{{i-1}} = ({u1}, {u2})$ with ${num_trials}$ Samples".format(num_trials=num_trials,
		u1=round(ui[0],3), u2=round(ui[1],3)))





	# circle2 = plt.Circle((0,0), 1, color='green', alpha=0.2)
	# axes[1].scatter(cardioid_pts[:,0], cardioid_pts[:,1], marker='o', color='orange')
	# axes[1].scatter(ui[0], ui[1], marker='o', color='red', linewidth=10)
	# axes[1].add_artist(circle2)
	# axes[1].set_xlim([-1.5,1.5])
	# axes[1].set_ylim([-1.5,1.5])
	# axes[1].set_title("Image of Empirical Uniform Distribution with {num_trials} Samples".format(num_trials=num_trials))


	# # Plot the pullback under the cardioid parametrization. Rotate the cardioid to have foci e1, then pullback.
	# rotated_vects = np.dot(vects, rotation)
	# pullback_pts = np.array(
	# 		[np.array(
	# 				[
	# 					np.arccos(x/np.sqrt(x**2 + y**2)),
	# 					np.sqrt(x**2 + y**2)
	# 				]
	# 			) for (x,y) in rotated_vects]
	# 	)

	# pullback_uniform_pts = np.array(
	# 		[np.array(
	# 				[
	# 					np.arctan(y/x),
	# 					1/( 1/np.sqrt(x**2 + y**2) + x/(x**2 + y**2))
	# 				]
	# 			) for (x,y) in uniform_pts]
	# 	)
	# axes[1,0].scatter(pullback_pts[:,0], pullback_pts[:,1], marker='o', color='blue')
	# axes[1,0].set_xlim([-2*np.pi, 2*np.pi])
	# axes[1,0].set_ylim([0, 2*la.norm(ui,2)])
	# axes[1,0].set_title("Pre-Image of Empirical Uniform Distribution Under Cardioid Map")
	# axes[1,0].set_xlabel(r'$\theta$')
	# axes[1,0].set_ylabel(r'$r$')

	# axes[1,1].scatter(pullback_uniform_pts[:,1], pullback_uniform_pts[:,0], marker='o', color='orange')
	# axes[1,1].set_xlim([-2*np.pi, 2*np.pi])
	# axes[1,1].set_ylim([0, 2*la.norm(ui,2)])
	# axes[1,1].set_title("Uniform Distribution Over Domain of Cardioid")
	# axes[1,1].set_xlabel(r'$\theta$')
	# axes[1,1].set_ylabel(r'$r$')



	Us = np.zeros((nframes*3+2*nframes, d))
	Us[0:nframes,:] = np.array([np.array([i, 0]) for i in np.arange(0,1,1.0/nframes)])
	Us[nframes:(2*nframes), :] = np.array([np.array([i, 0]) for i in np.arange(1,0,-1.0/nframes)])
	Us[(2*nframes):(3*nframes),:] = np.array([np.array([i, 0]) for i in np.arange(0,1,1.0/nframes)])
	Us[(3*nframes):, :] = np.array([np.array([np.cos(theta), np.sin(theta)]) for theta in np.arange(0,2*np.pi, 2*np.pi/(2*nframes))])
	vects = np.zeros((Us.shape[0], num_trials, d))
	for idx, ui in enumerate(Us):
		# Draw samples of (wi-qi)Xi given ui.
		for i in range(num_trials):
			w = 2*(np.random.rand()-1/2)
			x = np.random.randn(d)
			x = x/la.norm(x,2)
			q = walk.quantize_weight(w, ui, x)
			# vects[idx, i] = (w - q) * x
			vects[idx, i] = (-q) * x

	fig, ax = plt.subplots(1,1,figsize=(10,10))
	circle = plt.Circle((0,0), 1, color='green', alpha=0.2)
	xi_plt = ax.scatter([], [], marker='o', color='blue', alpha=0.1)
	ui_plt = ax.scatter([],[], marker='o', color='red', linewidth=10)
	ax.set_xlim([-1.5,1.5])
	ax.set_ylim([-1.5,1.5])
	ax.add_artist(circle)
	# ax.legend(['$(w_i - q_i)X_i$', '$u_{i-1}$'])
	ax.legend(['$-q_iX_i$', '$u_{i-1}$'])

	# Create the lines object for the update function call to update.
	path_collections = [ui_plt, xi_plt]

	def update(frame):
		for i in range(len(path_collections)):
			vector = frame[i]
			pc = path_collections[i]
			pc.set_offsets(vector)
			if i == 0:
				# Update the title with ui
				pc.axes.set_title("Empirical Conditional Distribution of $(w_i - q_i)X_i$ Given $u_{{i-1}} = ({u1}, {u2})$ with ${num_trials}$ Samples".format(num_trials=num_trials,
	 u1=round(vector[0],3), u2=round(vector[1],3)), fontsize=14)
		return pc

	ani = FuncAnimation(fig, update, frames=list(zip(Us, vects)), repeat=True)
	# # ani.save('./imgs/conditional_distrib_(w-q)X.gif', writer='imagemagick', fps=10)

def spacing_w_q():

	# Unconditional probability.
	# N = 10000
	Ns = range(100, 5101, 500)
	d = 20
	# ds = list(range(2,102, 10))
	num_trials = 5000

	# Order the columns by the gap: first column is for |w_i-q_i| < 1/2,
	# second column is for 1/2 < |w_i - q_i| < 1, and third column is for
	# |w_i - q_i| > 1.
	proportions = np.zeros((len(Ns), 3))
	for i, N in enumerate(Ns):
		# For each dimension, run num_trials trials and average the proportion.
		for j in range(num_trials):
			walk = RandomWalk(N, d)
			gtr_1 = np.abs(walk.w - walk.q) >= 1
			less_1 = np.abs(walk.w - walk.q) < 1
			gtr_05 = np.abs(walk.w - walk.q) >= 0.5
			less_05 = np.abs(walk.w - walk.q) < 0.5

			proportions[i,0] += sum(less_05)/N
			proportions[i,1] += sum(np.logical_and(gtr_05, less_1))/N
			proportions[i,2] += sum(gtr_1)/N

	proportions = proportions/num_trials

	fig, ax = plt.subplots(1, 1, figsize=(10, 10))
	fig.suptitle("Average Empirical Proportions of Distance $|w_i - q_i|$ with $d$ = {d}".format(d=d), fontsize=18)
	ax.set_ylabel("Proportion of Steps", fontsize=14)
	ax_legend = ["$|w_i - q_i| < 1/2$", "$|w_i - q_i| \in [1/2, 1)$", "$|w_i - q_i| \geq 1$"]
	ax.set_xlabel("$N$", fontsize=14)
	ax.plot(Ns, proportions[:, 0], '-o', linewidth=3)
	ax.plot(Ns, proportions[:, 1], '-o', linewidth=3)
	ax.plot(Ns, proportions[:, 2], '-o', linewidth=3)
	ax.legend(ax_legend, fontsize=14)

	# Conditional Probability
	d = 20
	walk = RandomWalk(N,d)
	e1 = np.zeros(d)
	e1[0] = 1
	norms = np.arange(0, 100, 0.5)
	proportions = np.zeros((len(norms),3))


	for i, norm in enumerate(norms):
		# Condition on the appropriately normalized ui.
		ui = norm*e1
		for j in range(num_trials):
			w = 2 * (np.random.rand() - 0.5)
			x = np.random.randn(d)
			x = x/la.norm(x,2) 
			q = walk.quantize_weight(w, ui, x)
			if abs(w-q) < 0.5:
				proportions[i, 0] += 1
			elif abs(w-q) < 1:
				proportions[i,1] += 1
			else:
				proportions[i,2] += 1

	proportions = proportions/num_trials

	fig, ax = plt.subplots(1, 1, figsize=(10, 10))
	fig.suptitle("Average Empirical Proportions of Distance $|w_i - q_i|$ with $d$ = {d}".format(d=d), fontsize=18)
	ax.set_ylabel("Proportion of Steps", fontsize=14)
	ax_legend = ["$|w_i - q_i| < 1/2$", "$|w_i - q_i| \in [1/2, 1)$", "$|w_i - q_i| \geq 1$"]
	ax.set_xlabel("$||u_{i-1}||_2$", fontsize=14)
	ax.plot(norms, proportions[:, 0], '-o', linewidth=3)
	ax.plot(norms, proportions[:, 1], '-o', linewidth=3)
	ax.plot(norms, proportions[:, 2], '-o', linewidth=3)
	ax.legend(ax_legend, fontsize=14)

def norm_phase_transitions():

	N = 10000
	ds = np.arange(2, 63, 10)
	ts = np.arange(0,3,0.05)
	ps = np.zeros((len(ds), len(ts)))
	for i, d in enumerate(ds):
		walk = RandomWalk(N,d)
		norms = list(map(lambda u: la.norm(u,2), walk.U))
		ps[i,:] = list(map(lambda t: sum(norms > t)/N, ts))

	fig, ax = plt.subplots(1,1, figsize=(10,10))
	fig.suptitle("$P(||u_i||_2^2 > t)$, N = {N}".format(N=N), fontsize=20)
	for i, d in enumerate(ds):
		ax.plot(ts, ps[i,:], '-o', linewidth=3, alpha=0.7)
	ax.set_xlabel("$t$", fontsize=14)
	ax.legend(["d = {d}".format(d=d) for d in ds])

def tail_bound_graphic():

	# The tail bound is hard to visualize because there are so many knobs.
	# To illucidate what's going on, this graphic shows for various values
	# of ||u_{i-1}||_2^2 and w_i how the plot of the quantity
	# P((w_i - q_i)^2 + 2(w_i - q_i)y > t) behaves where 
	# y = X_i^T u_{i-1}. We will use the proxy y ~ N(0, ||u_{i-1}||^2/d)
	# to actually plot.

	def p(t: float, w: float, sigma: float) -> float:
		if w < 0:
			# I only know these quantities for positive w
			raise ValueError
		if t < -w-w**2:
			return stats.norm.cdf(
					(t-(w-1)**2)/(2*(w-1)),
					loc=0,
					scale=sigma**2
				) - stats.norm.cdf(
					- 1/2 - w,
					loc=0,
					scale=sigma**2
				)
		elif t > w1 - w1**2:
			return stats.norm.cdf(
					(t-(w+1)**2)/(2*(w+1)),
					loc=0,
					scale=sigma**2
				) - stats.norm.cdf(
					- 1/2 - w,
					loc=0,
					scale=sigma**2
				)
		else:
			mu1 = stats.norm.cdf(
					(t-(w-1)**2)/(2*(w-1)),
					loc=0,
					scale=sigma**2
				) - stats.norm.cdf(
					- 1/2 - w,
					loc=0,
					scale=sigma**2
				)
			mu2 = stats.norm.cdf(
					(t-w**2)/(2*w),
					loc=0,
					scale=sigma**2
				) - stats.norm.cdf(
					(t-(w+1)**2)/(2*(w+1)),
					loc=0,
					scale=sigma**2
				)
			return mu1 - mu2

	d = 2
	ws = np.arange(0,1,0.1)
	ts = np.arange(-5,5,0.01)
	# sigmas = np.arange(1/(4*d), 1, 0.05)
	sigmas = [1/d]
	densities =  np.array( [list(zip(ts,stats.norm.pdf(ts, loc=0, scale=sigma))) for sigma in sigmas] )
	t_stars = np.arange(-2, 2, 0.1)

	# For now, just deal with the density plot. Add cdf plot later.
	fig, ax = plt.subplots(1,1,figsize=(10,10))
	density_plt, = ax.plot([],[], color='blue', linewidth=3)

	# Markers for important quantities
	tstar_marker, = ax.plot([], [], 'ro')
	pos_sq_marker = ax.axvline(x=0, ymin=0, ymax=0, color='red')
	neg_sq_marker = ax.axvline(x=0, ymin=0, ymax=0, color='red')
	dist_half_marker = ax.axvline(x=0, ymin=0, ymax=0, color='black')
	dist_neghalf_marker = ax.axvline(x=0, ymin=0, ymax=0, color='black')
	pos_ugly_marker = ax.axvline(x=0, ymin=0, ymax=0, color='orange')
	neg_ugly_marker = ax.axvline(x=0, ymin=0, ymax=0, color='orange')

	ax.legend(["Density", "$t^*$", "$w-w^2$", "$-w-w^2$", "$1/2 - w$", "$-1/2 - w$"]) #"r$\frac{2w^2 + w + 1}{2(w+1)}$",
		# "r$\frac{2w^2 - w + 1}{2(w-1)}$"])

	ax.set_xlim([min(ts), max(ts)])
	ax.set_ylim([0,1.5])

	def update(frame):

		w = frame[0]
		sigma = frame[1]
		tstar = frame[2]
		density_data = frame[3]

		ax.set_title("Tail Bound Plot for $(t^*, w, \sigma) = ({tstar}, {w}, {sigma})$".format(tstar=tstar, w=w, sigma=sigma))

		# Draw the new p curve
		density_plt.set_data(density_data.T)

		# Place new markers at tstar,
		# -w-w^2, w-w^2,
		# 1/2 - w, -1/2 - w,
		# (2w^2 + w + 1)/(2(w+1)), and (2w^2 - w + 1)/(2(w-1))


		# tstar is really more for the cdf plot.
		# tstar_marker.set_xydata(tstar)


		neg_w_w2 = np.array(
				[ [-w-w**2, -w-w**2],
				  [0 , 1.5] ]
			)
		pos_w_w2 = np.array(
				[ [w-w**2, w-w**2],
				  [0, 1.5] ]
			)
		dist_half = np.array(
				[ [1/2-w, 1/2-w],
				  [0, 1.5] ]
			)
		dist_neg_half = np.array(
				[ [-1/2-w, -1/2-w],
				  [0, 1.5] ]
			)
		pos_ugly_term = np.array(
				[ [(2*w**2 + w + 1)/(2*(w+1)), (2*w**2 + w + 1)/(2*(w+1))],
				  [0, 1.5] ]
			)
		neg_ugly_term = np.array(
				[ [(2*w**2 - w + 1)/(2*(w-1)), (2*w**2 - w + 1)/(2*(w-1))],
				  [0, 1.5] ]
			)
		pos_sq_marker.set_data(pos_w_w2)
		neg_sq_marker.set_data(neg_w_w2)
		dist_half_marker.set_data(dist_half)
		dist_neghalf_marker.set_data(dist_neg_half)
		pos_ugly_marker.set_data(pos_ugly_term)
		neg_ugly_marker.set_data(neg_ugly_term)

		ax.collections.clear()

		# Shade the regions whose measures add to the probability of > t green.
		# Shade those regions whose measures subtract from the probability of > t red.
		# This depends on what tstar is.
		if tstar < -w-w**2:
			# fill green between -1/2-w, (tstar - (w-1)**2)/(2*(w-1))
			green_idxs = np.logical_and(density_data[:,0] > -1/2-w, density_data[:,0] < (tstar - (w-1)**2)/(2*(w-1)))
			green_data = density_data[green_idxs, :]


			ax.fill_between(green_data[:,0], green_data[:, 1], color='green')
		elif tstar > w-w**2:
			# fill green between -1/2-w, (tstar - (w+1)**2)/(2*(w+1))
			green_idxs = np.logical_and(density_data[:,0] > -1/2-w, density_data[:,0] < (tstar - (w+1)**2)/(2*(w+1)))
			green_data = density_data[green_idxs, :]
			ax.fill_between(green_data[:,0], green_data[:, 1], color='green')
		else:
			# fill green between -1/2-w, (tstar - (w-1)**2)/(2*(w-1))
			# fill red between (tstar - (w+1)**2)/(2*(w+1)), (tstar - w**2)/(2*w)
			green_idxs = np.logical_and(density_data[:,0] > -1/2-w, density_data[:,0] < (tstar - (w-1)**2)/(2*(w-1)))
			red_idxs = np.logical_and(density_data[:,0] > -1/2-w, density_data[:,0] < (tstar - w**2)/(2*w))
			green_data = density_data[green_idxs, :]
			red_data = density_data[red_idxs, :]

			ax.fill_between(green_data[:,0], green_data[:, 1], color='green')
			ax.fill_between(red_data[:,0], red_data[:, 1], color='red')


	ani = FuncAnimation(fig, update, frames=list(product(ws, sigmas, t_stars, densities)), repeat=True)
	# # ani.save('./imgs/conditional_distrib_(w-q)X.gif', writer='imagemagick', fps=10)

def toy_random_walk():

	N = 1000
	X = np.zeros(N)
	X[0] = 0.25
	prize = 0.25
	for i in range(1,N):
		# Loss at round i is w(w-1) for w in (0,1).
		# Prize is 1/4.
		w = np.random.rand()
		loss = w * (w-1)

		if X[i-1] <= 0:
			p = 1
		else:
			p = stats.norm.cdf(
					0.5,
					loc=0,
					scale=X[i-1]
				) - stats.norm.cdf(
					-0.5,
					loc=0,
					scale=X[i-1]
				)
		is_winner = np.random.binomial(1,p)
		if is_winner:
			X[i] += X[i-1] + prize
		else:
			X[i] -= loss

	plt.plot(range(N), X)

def init_residual_experiment():

	N = 100000
	d = 10000
	init_norm = 100

	u0 = np.random.randn(d)
	u0 = init_norm * u0/la.norm(u0,2)

	walk = RandomWalk(N=N, d=d, u_init=u0)
	fig, axes = plt.subplots(1,2, figsize=(20,10))
	fig.suptitle("Residual Statistic Plots for Random Walk, (N, d) = ({N}, {d})".format(N=N, d=d), fontsize=20)
	axes[0].plot(range(N), [la.norm(walk.U[i], 2) for i in range(N)], '-o')
	axes[0].set_title("Norm of Residuals", fontsize=18)
	axes[0].set_xlabel("Step Index $i$", fontsize=14)
	axes[0].set_ylabel("$||u_i||$", fontsize=14)

	axes[1].plot(range(1,N), [la.norm(walk.U[i], 2) - la.norm(walk.U[i-1], 2) for i in range(1,N)], 'o')
	axes[1].set_title("Change in Norm of Residuals", fontsize=18)
	axes[1].set_xlabel("Step Index $i$", fontsize=14)
	axes[1].set_ylabel("$\Delta ||u_i||$", fontsize=14)

def mean_drift_experiment():

	N = 2
	d = 500
	num_trials = 10000
	alphas = np.arange(0.1, 5, 0.1)
	means = np.zeros(len(alphas))
	samples = np.zeros((len(alphas), num_trials))
	walk = RandomWalk(N=N, d=d)

	# Calculate E[||u_N+1||_2^2 - ||u_N||^2 | u_N = alpha * d * e_1]
	for i, alpha in enumerate(alphas):
		uN = np.zeros(d)
		uN[0] = alpha*sqrt(d)
		for trial in range(num_trials):
			X_new = np.random.randn(d)
			X_new = X_new/la.norm(X_new, 2)
			w = 2*(np.random.rand()-0.5)
			q = walk.quantize_weight(w, uN, X_new)
			samples[i, trial] = la.norm(uN + (w-q)*X_new,2)**2 - la.norm(uN,2)**2
			means[i] += la.norm(uN + (w-q)*X_new,2)**2 - la.norm(uN,2)**2

	means = 1/num_trials * means
	putative_bound = [4-alpha*np.e**(-9/8) for alpha in alphas]

	fig, axes = plt.subplots(1,1, figsize=(10,10))
	fig.suptitle("Empirical Average of $||u_{{t}}||_2^2 - ||u_{{t-1}}||_2^2 | u_{{t-1}}$, d={d} with {num_trials} Trials".format(d=d, num_trials=num_trials), fontsize=20)
	axes.plot(alphas, means, '-o')
	axes.plot(alphas, putative_bound, '-o')
	axes.set_xlabel(r"$\frac{||u_{t-1}||}{\sqrt{d}}$", fontsize=18)
	axes.set_ylabel("$E[||u_{{t}}||_2^2 - ||u_{{t-1}}||_2^2 | u_{{t-1}}]$", fontsize=14)
	axes.legend(['Empirical Results', r'$4-4 \frac{||u_{t-1}||}{\sqrt{d}e^{9/8}}$'])

def expectation_qX_given_u():

	N = 2
	d=2
	num_trials = 1000000
	walk = RandomWalk(N,d)
	vects = np.zeros((num_trials, d))
	norm_range = np.arange(0,10*d)
	nUs = len(norm_range)
	Us = np.zeros((nUs, d))
	Us[:,0] = norm_range
	qXs = np.zeros((nUs, num_trials, d))
	my_EqXs = np.zeros((nUs, d))

	# My calculation for the expected value.
	fixed_w = 0.25
	area_S_d_minus_2 = (d-1)*np.pi**((d-1)/2)/gamma((d-1)/2 + 1)
	area_S_d_minus_1 = d*np.pi**(d/2)/gamma(d/2 + 1)

	for idx, ui in enumerate(Us):
		# Draw samples of (wi-qi)Xi given ui.
		if la.norm(ui,2) < 10**-8:
			my_EqXs[idx,:] = np.zeros(d)
		else:
			neg1_part = (1-((0.5 + fixed_w)/la.norm(ui,2))**2)**((d-1)/2)/(d-1) if (-0.5-fixed_w)/la.norm(ui,2) > -1 else 0
			pos1_part = (1-((0.5 - fixed_w)/la.norm(ui,2))**2)**((d-1)/2)/(d-1) if (0.5 - fixed_w)/la.norm(ui,2) < 1 else 0
			my_EqXs[idx,:] = -((area_S_d_minus_2/area_S_d_minus_1) * ( pos1_part + 
				neg1_part)) * ui/la.norm(ui,2)
		for i in range(num_trials):
			# w = 2*(np.random.rand()-1/2)
			w = fixed_w
			x = np.random.randn(d)
			x = x/la.norm(x,2)
			q = walk.quantize_weight(w, ui, x)
			# vects[idx, i] = (w - q) * x
			qXs[idx, i] = (-q) * x

	qX_means = np.mean(qXs, 1)
	diffs = [la.norm(qX_means[j,:] - my_EqXs[j,:],2)/la.norm(ui,2) if j != 0 else 0 for j in range(nUs)]
	fig, ax = plt.subplots(1,1,figsize=(10,10))
	ax.plot(norm_range, diffs, '-o')
	ax.set_title("Empirical vs Calculated $E q_t X_t$, d={d}, num_trials={num_trials}".format(d=d, num_trials=num_trials), fontsize=20)
	ax.set_xlabel("$||u_t||$", fontsize=18)
	ax.set_ylabel(r"$\frac{||Calculated - Empirical||}{||u_i||}$", fontsize=18)


	# Plot the means in R^2 to visualize pictorially.
	# circle = plt.Circle((0,0), 1, color='green', alpha=0.2)
	# colors = plt.cm.rainbow(np.linspace(0,1,nUs))
	# labels=[]
	# for i, mean in enumerate(qX_means):
	# 	qX_plt = ax.scatter(mean[0], mean[1], marker='x', color=colors[i])
	# 	labels += ["$u_i = ({u1}, {u2})$".format(u1=round(Us[i,0],2), u2=round(Us[i,1],2))]
	# ax.legend(labels)
	# for i, u in enumerate(Us):
	# 	mean_plt = ax.scatter(u[0], u[1], marker='o', color=colors[i])
	# ax.set_xlim([-1.5,1.5])
	# ax.set_ylim([-1.5,1.5])
	# ax.add_artist(circle)
	# ax.set_title("Plot of $E[-q_t X_t | u_{t-1}]$", fontsize=22)

def absolute_increments_CDF():

	def spherical_sector_msr(d: np.float, a: np.float, b: np.float):
		area_S_d_minus_2 = (d-1)*np.pi**((d-1)/2)/gamma((d-1)/2 + 1)
		area_S_d_minus_1 = d*np.pi**(d/2)/gamma(d/2 + 1)
		return quad(lambda x: (1-x**2)**((d-2)/2), a, b)[0]*area_S_d_minus_2/area_S_d_minus_1

	def myCDF(w: np.float, u_norm: np.float, alpha: np.float):
		# Assuming w, u_norm > 0 and w != 1, output cdf of absolute square increments
		# given that the previous residual has norm u_norm.
		if alpha < w - w**2:
			pass

	N=2
	d=10

	# Fix a previous residual. Compute the empirical distribution of the
	# square increments given the previous residual.
	u0=np.zeros(d)
	u0[0]= sqrt(d)

	num_samples = 10**6
	walk = RandomWalk(N,d)
	w = 0.25
	empirical_distribution = np.zeros(num_samples)

	for i in range(num_samples):
		x = random.randn(d)
		x = x/la.norm(x,2)
		q = walk.quantize_weight(w, u0, x)
		u = u0 + (w-q)*x
		increment = la.norm(u,2)**2 - la.norm(u0,2)**2
		empirical_distribution[i] = increment

def upper_bound_mgf_increments():

	N=2
	d=3
	walk = RandomWalk(N,d)
	num_samples = 10000
	lambdas = np.arange(-1,1,0.1)
	u0 = np.zeros(d)
	norm_u0s = np.arange(2,d, d/10.0)

	# Fix w for now.
	fixed_w = 0.25

	area_S_d_minus_2 = (d-1)*np.pi**((d-1)/2)/gamma((d-1)/2 + 1)
	area_S_d_minus_1 = d*np.pi**(d/2)/gamma(d/2 + 1)
	ratio_areas = area_S_d_minus_2/area_S_d_minus_1

	# Every slice along first component is for fixed u0. Rows in the fixed first slice
	# correspond to a fixed lambda. Columns are random samples.
	mgf_point_samples = np.zeros((norm_u0s.shape[0], lambdas.shape[0], num_samples))

	for norm_idx, norm in enumerate(norm_u0s):
		u0[0] = norm
		for lambda_idx, L in enumerate(lambdas):
			for sample_idx in range(num_samples):
				x = np.random.randn(d)
				x = x/la.norm(x,2)
				w = fixed_w
				q = walk.quantize_weight(w, u0, x)
				u = u0 + (w-q)*x
				increment = la.norm(u,2)**2 - norm**2
				mgf_point_samples[norm_idx, lambda_idx, sample_idx] = np.exp(L * increment)

	mgf_curves = np.mean(mgf_point_samples, 2)
	legend = []
	fig, ax = plt.subplots(1,1,figsize=(10,10))
	ax.set_title("Empirical Moment Generating Function of $ \Delta ||u_t||^2 | ||u_{t-1}||^2$", fontsize=24)
	colors = plt.cm.rainbow(np.linspace(0,1,norm_u0s.shape[0]))

	for idx, color in enumerate(colors):

		norm = norm_u0s[idx]
		my_bound = ratio_areas*np.array([

		np.exp(4*abs(L)) * (

			(np.exp(2*L*(fixed_w-1)*(0.5-fixed_w)) - np.exp(2*L*(fixed_w - 1)*norm))/(2*L * (1-fixed_w) * norm) 

			+ (np.exp(2*L * fixed_w * (0.5-fixed_w)) - np.exp(-2*L * fixed_w * (0.5+fixed_w)))/(2*L * fixed_w * norm) 

			+ (np.exp(-2*L * (fixed_w+1)*(0.5+fixed_w)) - np.exp(-2*L*(fixed_w+1)*norm))/(2*L*(fixed_w + 1)*norm)
		)

		for L in lambdas
		])

		# ax.plot(lambdas[:], mgf_curves[idx,:], '-o', color=color)
		ax.plot(lambdas[:], my_bound[:], '-', color=color)
		# legend += ["||u_t|| = {norm}".format(norm=round(norm_u0s[idx],3))]
		legend += ["Upper Bound When||u_t|| = {norm}".format(norm=round(norm_u0s[idx],3))]

	ax.set_xlabel("$\lambda$", fontsize=18)
	ax.set_ylabel("$E \exp(\lambda ||u_{t}||^2) | u_{{t-1}})$", fontsize=18)
	ax.legend(legend, fontsize=18)

def sanity_check_bound():

	ds = np.array([10*i for i in range(1, 50)])
	sigmas = np.arange(1, 301, 100)
	N = 10**4
	w = 2*(np.random.rand(N)-0.5)
	
	matrix_of_bounds = np.zeros((len(ds), len(sigmas)))

	fig, ax = plt.subplots(1,1, figsize=(10,10))
	ax.set_title(r"$||u_t||_2$ with $X_t \sim N(0, \sigma^2 I)$, N = {N}".format(N=N), fontsize=20)
	ax.set_xlabel(r"$d$", fontsize=18)
	ax.set_ylabel(r"$\frac{Median(||u_t||_2)}{d \sigma}$", fontsize=18)
	ax.set_ylim([0,0.5])
	for sig_idx, sigma in enumerate(sigmas):
		for d_idx, d in enumerate(ds):
			X = np.random.randn(N,d)*sigma
			mywalk = CustomWalk(X=X, w=w)
			norms = [la.norm(u,2) for u in mywalk.U]
			matrix_of_bounds[d_idx, sig_idx] = np.median(norms)/(sigma * d)
		ax.plot(ds, matrix_of_bounds[:, sig_idx].T, '-o')
		ax.legend([r"$\sigma =$ {sigma}".format(sigma=sigma) for sigma in sigmas])

def sigma_vs_q_0():

	d = 100
	N = 10**4
	sigmas = [d**i for i in np.arange(-1,1,0.25)]
	num_trials = 25
	w = [0.25 for i in range(N)]
	proportion_q0 = np.zeros(len(sigmas))
	for sig_idx, sigma in enumerate(sigmas):
		for trial in range(num_trials):
			X = np.random.randn(N,d)*sigma
			mywalk = CustomWalk(X=X, w=w)
			proportion_q0[sig_idx] += sum(mywalk.q==0)
		proportion_q0[sig_idx] = proportion_q0[sig_idx]/(N*num_trials)

	fig, ax = plt.subplots(1,1, figsize=(10,10))
	ax.set_title(r"Proportion of 0 bits as a function of $\sigma$")
	ax.plot(sigmas, proportion_q0, '-o')
	ax.set_xlabel(r"$\sigma$", fontsize=18)
	ax.set_ylabel(r"$P(q_t = 0)$", fontsize=18)

def increments_gaussian():

	d = 20
	N = 100000
	sigmas = [1,2, 3]

	fig, ax = plt.subplots(1,1, figsize=(10,10))
	ax.set_title("Increments", fontsize=24)
	ax.set_xlabel(r"$t$", fontsize=18)
	ax.set_ylabel(r"$\Delta ||u_{t}||_2^2$", fontsize=18)
	for sigma in sigmas:
		X = np.random.randn(N,d)*sigma
		w = 2*(np.random.rand(N)-0.5)
		mywalk = CustomWalk(X=X, w=w)
		q = mywalk.q
		U = mywalk.U
		increments = np.zeros(N)
		increments[0] = (w[0]-q[0])**2
		increments[1:] = [(w[i]-q[i])**2  + 2*(w[i] - q[i])*np.dot(X[i,:]/ la.norm(X[i,:],2)**2, U[i-1,:]) for i in range(1,N)]
		ax.hist(increments, alpha=0.3)

	ax.legend([r'$\sigma$ = {sigma}'.format(sigma=sigma) for sigma in sigmas])

def med_rel_err_batch_size():

	N0 = 100
	N1 = 500
	N2 = 1
	Bs = np.arange(50,1000,50)
	batch_trials = 10

	def get_batch_data(batch_size:int):
		# Gaussian data for now.
		return np.random.randn(batch_size, N0)

	med_rel_errs = np.zeros(Bs.shape)
	ninety_fifth_percentiles = np.zeros(Bs.shape)
	for B_idx, B in enumerate(Bs):
		for trial_idx in range(batch_trials):
			model = Sequential()
			layer1 = Dense(N1, activation=None, use_bias=False, input_dim=N0,
				kernel_initializer=RandomUniform(-1,1))
			layer2 = Dense(N2, activation=None, use_bias=False, kernel_initializer=RandomUniform(-1,1))
			model.add(layer1)
			model.add(layer2)

			my_quant_net = QuantizedNeuralNetwork(model, B, get_batch_data, use_indep_quant_steps=False)
			logger.info(f"Quantizing network with batchsize = {B} trial {trial_idx}...")
			my_quant_net.quantize_network()
			logger.info("done!")

			med_rel_errs[B_idx] += np.median(my_quant_net.layerwise_rel_errs[1])
			ninety_fifth_percentiles[B_idx] += np.percentile(my_quant_net.layerwise_rel_errs[1], 95)

		# Divide by number of trials to average.
		med_rel_errs[B_idx] = med_rel_errs[B_idx]/batch_trials
		ninety_fifth_percentiles[B_idx] = ninety_fifth_percentiles[B_idx]/batch_trials


	fig, ax = plt.subplots(1,1, figsize=(10,10))
	caption = "Caption: Percentiles are taken across all neurons in this layer's weight matrix. " \
	f"For each batch size, we averaged the percentile over {batch_trials} trials to account for variation in data. " \
	"Note that these errors are computed on the data that are used to choose the bits."
	fig.text(0.5, 0.01, caption, ha='center', wrap=True, fontsize=12)
	fig.suptitle(rf"Second Layer Average Percentiles of Relative Errors: $(N_1, N_2)$ = ({N1}, {N2})", fontsize=18)
	ax.set_xlabel("Batch Size", fontsize=14)
	ax.set_ylabel("Median Relative Error", fontsize=14)
	ax.plot(Bs, med_rel_errs, '-o')
	# ax.plot(Bs, ninety_fifth_percentiles, '-o')
	# ax.legend(["Median", r"$95^{th}$ Percentile"]))

def med_rel_err_N():

	N0s = 100
	N1s = np.arange(100, 1001, 100)
	N2 = 1
	B = 100
	N1_trials = 10

	def get_batch_data(batch_size:int, N0: int):
		# Gaussian data for now.
		return np.random.randn(batch_size, N0)

	med_rel_errs = np.zeros(N1s.shape)
	ninety_fifth_percentiles = np.zeros(N1s.shape)
	for N1_idx, N1 in enumerate(N1s):

		def get_my_batch_data(batch_size:int):
			return get_batch_data(batch_size, N0)

		for trial_idx in range(N1_trials):
			model = Sequential()
			layer1 = Dense(N1, activation=None, use_bias=False, input_dim=N0,
				kernel_initializer=RandomUniform(-1,1))
			layer2 = Dense(N2, activation=None, use_bias=False, kernel_initializer=RandomUniform(-1,1))
			model.add(layer1)
			model.add(layer2)

			my_quant_net = QuantizedNeuralNetwork(model, B, get_my_batch_data, use_indep_quant_steps=False)
			logger.info(f"Quantizing network with N1 = {N1} trial {trial_idx}...")
			my_quant_net.quantize_network()
			logger.info("done!")

			med_rel_errs[N1_idx] += np.median(my_quant_net.layerwise_rel_errs[1])
			ninety_fifth_percentiles[N1_idx] += np.percentile(my_quant_net.layerwise_rel_errs[1], 95)

		# Divide by number of trials to average.
		med_rel_errs[N1_idx] = med_rel_errs[N1_idx]/N1_trials
		ninety_fifth_percentiles[N1_idx] = ninety_fifth_percentiles[N1_idx]/N1_trials


	fig, ax = plt.subplots(1,1, figsize=(10,10))
	caption = "Caption: Percentiles are taken across all neurons in this layer's weight matrix. " \
	f"For each value of N1, we averaged the percentile over {N1_trials} trials to account for variation in data. " \
	"Note that these errors are computed on the data that are used to choose the bits."
	fig.text(0.5, 0.01, caption, ha='center', wrap=True, fontsize=12)
	fig.suptitle(rf"Second Layer Average Median Relative Errors: $(N_2, B)$ = ({N2}, {B})", fontsize=18)
	ax.set_xlabel(r"$N_1$", fontsize=14)
	ax.set_ylabel("Relative Error", fontsize=14)
	ax.plot(N1s, med_rel_errs, '-o')
	# ax.plot(N0s, ninety_fifth_percentiles, '-o')
	# ax.legend(["Median", r"$95^{th}$ Percentile"])

def out_of_sample_rel_err():

	N0 = 10
	# N0s = np.arange(100, 1001, 100)
	# N1 = 1000
	N1s = np.arange(100, 1001, 100)
	N2 = 1
	# Bs = np.arange(50,1000,50)
	B = 50
	num_trials = 10
	layer_idx = 1
	def get_batch_data(batch_size:int, N0: int):
		# Gaussian data for now.
		return np.random.randn(batch_size, N0)


	med_rel_errs = np.zeros(N1s.shape)
	ninety_fifth_percentiles = np.zeros(N1s.shape)
	for N1_idx, N1 in enumerate(N1s):

		def get_my_batch_data(batch_size:int):
			return get_batch_data(batch_size, N0)

		for trial_idx in range(num_trials):
			model = Sequential()
			layer1 = Dense(N1, activation=None, use_bias=False, input_dim=N0,
				kernel_initializer=RandomUniform(-1,1))
			layer2 = Dense(N2, activation=None, use_bias=False, kernel_initializer=RandomUniform(-1,1))
			model.add(layer1)
			model.add(layer2)

			my_quant_net = QuantizedNeuralNetwork(model, B, get_my_batch_data, use_indep_quant_steps=False)
			logger.info(f"Quantizing network with N1 = {N1} trial {trial_idx}...")
			my_quant_net.quantize_network()
			logger.info("done!")

			# Get new data.
			X = get_my_batch_data(B)
			q_model = my_quant_net.quantized_net
			trained_output = Kfunction([model.layers[0].input],
										[model.layers[layer_idx].output]
									)
			quant_output = Kfunction([q_model.layers[0].input],
									[q_model.layers[layer_idx].output]
									)
			wX = trained_output([X])[0]
			qX = quant_output([X])[0]
			rel_errs = [la.norm(wX[:, t] - qX[:, t])/la.norm(wX[:,t]) for t in range(N2)]

			med_rel_errs[N1_idx] += np.median(rel_errs)
			ninety_fifth_percentiles[N1_idx] += np.percentile(rel_errs, 95)

		# Divide by number of trials to average.
		med_rel_errs[N1_idx] = med_rel_errs[N1_idx]/num_trials
		ninety_fifth_percentiles[N1_idx] = ninety_fifth_percentiles[N1_idx]/num_trials


	fig, ax = plt.subplots(1,1, figsize=(10,10))
	caption = "Caption: Percentiles are taken across all neurons in this layer's weight matrix. " \
	f"For each value of N1, we averaged the percentile over {num_trials} trials to account for variation in data. " \
	"Note that these errors are computed on the data that are independent of those used to choose the bits."
	fig.text(0.5, 0.01, caption, ha='center', wrap=True, fontsize=12)
	fig.suptitle(rf"Second Layer Average Median OOS Relative Errors: $(N_2, B)$ = ({N2}, {B})", fontsize=18)
	ax.set_xlabel(r"$N_1$", fontsize=14)
	ax.set_ylabel("Relative Error", fontsize=14)
	ax.plot(N1s, med_rel_errs, '-o')
	# ax.plot(N1s, ninety_fifth_percentiles, '-o')
	# ax.legend(["Median", r"$95^{th}$ Percentile"])




