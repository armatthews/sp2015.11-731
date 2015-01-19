import sys
import numpy

def normalize(v):
	assert abs(sum(v)) > 1.0e-100
	return v / sum(v)

class HiddenMarkovModel:
	def __init__(self, state_names, observation_names, start_probs, transition_probs, emission_probs):
		self.state_names = state_names
		self.observation_names = observation_names
		self.start_probs = start_probs
		self.transition_probs = transition_probs
		self.emission_probs = emission_probs
		self.N = len(states)
		self.K = len(observations)

	# forward[t][i] = p(X_i, y_1:i | theta) 
	def forward(self, observations):
		state_probs = self.start_probs
		forward_probs = [state_probs]
		for i in range(len(observations)):
			obs_matrix = numpy.diag(self.emission_probs.transpose()[observations[i]])
			state_probs = state_probs.dot(self.transition_probs).dot(obs_matrix)
			forward_probs.append(state_probs)
		return numpy.array(forward_probs)

	# backward[t][i] = p(y_t+1:T | X_t = i, theta) 
	def backward(self, observations):
		state_probs = numpy.array([1.0 for state in states])
		backward_probs = [state_probs]
		for i in range(len(observations))[::-1]:
			obs_matrix = numpy.diag(self.emission_probs.transpose()[observations[i]])
			state_probs = self.transition_probs.dot(obs_matrix).dot(state_probs)
			backward_probs.append(state_probs)
		return numpy.array(backward_probs[::-1])

	# gamma[t][i] = p(X_t = i | Y, theta)
	def gamma(self, forward_probs, backward_probs):
		g = []
		for f, b in zip(forward_probs, backward_probs):
			p = normalize(f * b)
			g.append(p)
		return numpy.array(g)

	# xi[t][i][j] = p(X_t = i, X_t+1 = j | Y, theta)
	def xi(self, forward_probs, backward_probs, observations):
		fp = forward_probs[:len(observations),:, numpy.newaxis]
		tp = self.transition_probs[numpy.newaxis,:,:]
		bp = backward_probs[1:len(observations) + 1,numpy.newaxis,:]
		ep = numpy.array([self.emission_probs.transpose()[observations[t]] for t in range(0, len(observations))])[:,numpy.newaxis,:]
		z = numpy.array([1.0/(forward_probs[t].dot(backward_probs[t])) for t in range(0, len(observations))])[:,numpy.newaxis,numpy.newaxis]
		return (fp * tp * bp * ep * z)

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

	def update_parameters(self, observations):
		forward_probs = self.forward(observations)
		backward_probs = self.backward(observations)
		g = self.gamma(forward_probs, backward_probs)
		x = self.xi(forward_probs, backward_probs, observations)

		# Compute new transition probs
		for i in range(self.N):
			# s is the expected number of times we pass through state i 
			s = sum(g[t][i] for t in range(0, len(observations)))
			T = numpy.zeros(self.N)
			for j in range(self.N):
				t = sum(x[t][i][j] for t in range(0, len(observations)))
				T[j] = t / s if abs(s) > 1.0e-100 else self.transition_probs[i][j]
			T = normalize(T)
			self.transition_probs[i] = T

		# Compute new emission probs
		for i in range(self.N):
			T = numpy.zeros(self.K)
			# s is the expected number of times we pass through state i
			s = sum(g[t][i] for t in range(1, len(observations) + 1))
			for k in range(self.K):
				t = sum(g[t + 1][i] for t in range(len(observations)) if observations[t] == k)
				T[k] = t / s if abs(s) > 1.0e-100 else self.emission_probs[i][k]
			self.emission_probs[i] = T

		# Compute new start probs
		T = normalize(numpy.linalg.inv(self.transition_probs).dot(g[1]))
		#self.start_probs = T

		obs_prob = sum(forward_probs[-1])
		return obs_prob

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
elif True:
	states = ('rain', 'no rain')
	observations = ('umbrella', 'no umbrella')
	start_probs = numpy.array([0.5, 0.5])
	transition_probs = numpy.array([[0.7, 0.3], [0.3, 0.7]])
	emission_probs = numpy.array([[0.9, 0.1], [0.2, 0.8]])
elif False:
	states = ('rain', 'no rain')
	observations = ('umbrella', 'no umbrella')
	start_probs = numpy.array([0.5, 0.5])
	transition_probs = numpy.array([[0.6667, 0.3333], [1.0, 0.0]])
	emission_probs = numpy.array([[1.0, 0.0], [0.0, 1.0]])
elif False:
	states = ('state 1', 'state 2')
	observations = ('no eggs', 'eggs')
	start_probs = numpy.array([0.2, 0.8])
	transition_probs = numpy.array([[0.5, 0.5], [0.3, 0.7]])
	emission_probs = numpy.array([[0.3, 0.7], [0.8, 0.2]])
else:
	states = ('1', '2')
	observations = ('1')
	start_probs = numpy.array([1.0, 0.0])
	transition_probs = numpy.array([[0.6, 0.4], [0.1234, 0.8766]])
	emission_probs = numpy.array([[1.0], [1.0]])

hmm = HiddenMarkovModel(states, observations, start_probs, transition_probs, emission_probs)
observations = [0, 0, 1, 0, 0]
print hmm.viterbi(observations)

for iteration in range(1000):
	obs_prob = hmm.update_parameters(observations)
	print obs_prob 

print hmm.start_probs
print hmm.transition_probs
print hmm.emission_probs
print numpy.linalg.inv(hmm.transition_probs)
