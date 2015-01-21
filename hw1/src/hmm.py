import sys
import numpy
from hmmstats import HmmStatistics

class HiddenMarkovModel:
	def __init__(self, num_states, num_obs_types, start_probs, transition_probs, emission_probs):
		self.start_probs = start_probs
		self.transition_probs = transition_probs
		self.emission_probs = emission_probs
		self.N = num_states
		self.K = num_obs_types
		self.stats = HmmStatistics(self.N, self.K)

	# forward[t][i] = p(X_i, y_1:i | theta) 
	def forward(self, observations):
		forward_probs = numpy.zeros((len(observations), self.N))
		for i in range(len(observations)):
			state_probs = state_probs.dot(self.transition_probs) if i > 0 else self.start_probs
			obs_matrix = numpy.diag(self.emission_probs.transpose()[observations[i]])
			state_probs = state_probs.dot(obs_matrix)
			forward_probs[i] = state_probs

		return forward_probs

	# backward[t][i] = p(y_t+1:T | X_t = i, theta) 
	def backward(self, observations):
		state_probs = numpy.zeros(self.N)
		state_probs.fill(1.0)
		backward_probs = numpy.zeros((len(observations), self.N))
		backward_probs[len(observations) - 1] = state_probs
		for i in range(len(observations) - 1, 0, -1):
			obs_matrix = numpy.diag(self.emission_probs.transpose()[observations[i]])
			state_probs = self.transition_probs.dot(obs_matrix).dot(state_probs)
			backward_probs[i - 1] = state_probs

		return backward_probs

	# gamma[t][i] = p(X_t = i | Y, theta)
	def gamma(self, forward_probs, backward_probs):
		assert forward_probs.shape == backward_probs.shape
		g = forward_probs * backward_probs
		z = numpy.sum(g, axis=1)[:,numpy.newaxis]
		return g/z 

	# xi[t][i][j] = p(X_t = i, X_t+1 = j | Y, theta)
	def xi(self, forward_probs, backward_probs, observations):
		fp = forward_probs[:-1, :, numpy.newaxis]
		tp = self.transition_probs[numpy.newaxis,:,:]
		bp = backward_probs[1:,numpy.newaxis,:]
		ep = numpy.array([self.emission_probs.transpose()[observations[t]] for t in range(0, len(observations))])[1:,numpy.newaxis,:]
		z = numpy.array([1.0/(forward_probs[t].dot(backward_probs[t])) for t in range(0, len(observations))])[:-1,numpy.newaxis,numpy.newaxis]

		x = fp * tp * bp * ep * z
		return x

	# Finds argmax_X of p(X | Y, theta)
	def viterbi(self, observations):
		V = numpy.zeros((len(observations), self.N))
		B = [[0 for i in range(self.N)] for j in range(len(observations))]
		V[0] = self.start_probs * self.emission_probs.transpose()[observations[0]]
		
		for t in range(1, len(observations)):
			for i in range(self.N):
				k = max(range(self.N), key=lambda k: self.transition_probs[k][i] * V[t - 1][k])
				V[t][i] = self.emission_probs[i][observations[t]] * self.transition_probs[k][i] * V[t - 1][k]
				B[t][i] = k

		Xf = max(range(self.N), key=lambda k: V[len(observations) - 1][k])
		X = [Xf]
		for t in range(len(observations) - 1, 0, -1):
			X.append(B[t][X[-1]])
		X = X[::-1]
		return X

	def expectation_step(self, observations):
		forward_probs = self.forward(observations)
		backward_probs = self.backward(observations)
		g = self.gamma(forward_probs, backward_probs)
		x = self.xi(forward_probs, backward_probs, observations)

		expected_start_counts = g[0]
		expected_state_counts = numpy.sum(g, axis=0)
		expected_transition_counts = numpy.sum(x, axis=0)

		# Here are two ways of computing expected_emission_counts
		# The first is faster if self.K is small
		# The second is faster if self.K is large
		if False:
			obs_matrix = numpy.zeros((len(observations), self.K))
			for t in range(len(observations)):
				obs_matrix[t][observations[t]] = 1.0
			expected_emission_counts = (g.transpose().dot(obs_matrix))
		else:
			expected_emission_counts = numpy.zeros((self.N, self.K))
			for t, obs in enumerate(observations):
				expected_emission_counts[:, obs] += g[t]

		self.stats.add(expected_start_counts, expected_state_counts, expected_transition_counts, expected_emission_counts)

		obs_prob = sum(forward_probs[-1])
		return obs_prob

	def maximization_step(self):
		self.start_probs = self.stats.compute_start_probs()
		self.transition_probs = self.stats.compute_transition_probs()
		self.emission_probs = self.stats.compute_emission_probs()

		self.stats.reset()

