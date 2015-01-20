import sys
import math
import numpy

def normalize(v):
	assert abs(sum(v)) > 1.0e-100
	return v / sum(v)

class HmmStatistics:
	def __init__(self, N, K):
		self.N = N
		self.K = K
		self.reset()

	def reset(self):
		self.start_counts = numpy.zeros(self.N)
		self.state_counts = numpy.zeros(self.N)
		self.transition_counts = numpy.zeros((self.N, self.N))
		self.emission_counts = numpy.zeros((self.N, self.K))

	def add(self, start_counts, state_counts, transition_counts, emission_counts):
		self.start_counts += start_counts
		self.state_counts += state_counts
		self.transition_counts += transition_counts
		self.emission_counts += emission_counts

	def compute_start_probs(self):
		return self.start_counts / numpy.sum(self.start_counts)

	def compute_transition_probs(self):
		transition_probs = self.transition_counts / self.state_counts[:, numpy.newaxis]
		transition_probs = transition_probs / transition_probs.sum(axis=1)[:, numpy.newaxis]
		return transition_probs

	def compute_emission_probs(self):
		emission_probs = self.emission_counts / self.state_counts[:, numpy.newaxis]
		emission_probs = emission_probs / numpy.sum(emission_probs, axis=1)
		return emission_probs	

class HiddenMarkovModel:
	def __init__(self, state_names, observation_names, start_probs, transition_probs, emission_probs):
		self.state_names = state_names
		self.observation_names = observation_names
		self.start_probs = start_probs
		self.transition_probs = transition_probs
		self.emission_probs = emission_probs
		self.N = len(states)
		self.K = len(observations)
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
		obs_matrix = numpy.zeros((len(observations), self.K))
		for t in range(len(observations)):
			obs_matrix[t][observations[t]] = 1.0

		forward_probs = self.forward(observations)
		backward_probs = self.backward(observations)
		g = self.gamma(forward_probs, backward_probs)
		x = self.xi(forward_probs, backward_probs, observations)

		expected_start_counts = g[0]
		expected_state_counts = numpy.sum(g, axis=0)
		expected_transition_counts = numpy.sum(x, axis=0)
		expected_emission_counts = (g.transpose().dot(obs_matrix))

		self.stats.add(expected_start_counts, expected_state_counts, expected_transition_counts, expected_emission_counts)

		obs_prob = sum(forward_probs[-1])
		return obs_prob

	def maximization_step(self):
		self.start_probs = self.stats.compute_start_probs()
		self.transition_probs = self.stats.compute_transition_probs()
		self.emission_probs = self.stats.compute_emission_probs()

		self.stats.reset()

if False:
	states = ('Rainy', 'Sunny')
	observations = ('walk', 'shop', 'clean')
	start_probs = numpy.array([0.6, 0.4])
	transition_probs = numpy.array([[0.7, 0.3], [0.4, 0.6]])
	emission_probs = numpy.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
elif False:
	states = ('Healthy', 'Fever')
	observations = ('normal', 'cold', 'dizzy')
	start_probs = numpy.array([0.6, 0.4])
	transition_probs = numpy.array([[0.7, 0.3], [0.4, 0.6]])
	emission_probs = numpy.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
elif False:
	states = ('rain', 'no rain')
	observations = ('umbrella', 'no umbrella')
	start_probs = numpy.array([0.5, 0.5])
	transition_probs = numpy.array([[0.7, 0.3], [0.3, 0.7]])
	emission_probs = numpy.array([[0.9, 0.1], [0.2, 0.8]])
elif False:
	states = ('rain', 'no rain')
	observations = ('umbrella', 'no umbrella')
	start_probs = numpy.array([0.5, 0.5])
	transition_probs = numpy.array([[0.6667, 0.3333], [0.99, 0.01]])
	emission_probs = numpy.array([[0.99, 0.01], [0.01, 0.99]])
elif False:
	states = ('state 1', 'state 2')
	observations = ('no eggs', 'eggs')
	start_probs = numpy.array([0.2, 0.8])
	transition_probs = numpy.array([[0.5, 0.5], [0.3, 0.7]])
	emission_probs = numpy.array([[0.3, 0.7], [0.8, 0.2]])
elif False:
	states = ('1', '2')
	observations = ('1')
	start_probs = numpy.array([1.0, 0.0])
	transition_probs = numpy.array([[0.6, 0.4], [0.1234, 0.8766]])
	emission_probs = numpy.array([[1.0], [1.0]])
else:
	states = ('s', 't')
	observations = ('A', 'B')
	start_probs = numpy.array([0.85, 0.15])
	transition_probs = numpy.array([[0.3, 0.7], [0.1, 0.9]])
	emission_probs = numpy.array([[0.4, 0.6], [0.5, 0.5]])

hmm = HiddenMarkovModel(states, observations, start_probs, transition_probs, emission_probs)
data = [[0, 1, 1, 0] for i in range(10)] + [[1, 0, 1] for i in range(20)]

for iteration in range(1000):
	total_log_prob = 0.0
	for observations in data:
		obs_prob = hmm.expectation_step(observations)
		total_log_prob += math.log(obs_prob)
	hmm.maximization_step()
	print total_log_prob

print hmm.start_probs
print hmm.transition_probs
print hmm.emission_probs
for observations in data:
	print hmm.viterbi(observations)
