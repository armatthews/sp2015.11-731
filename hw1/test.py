import sys
import numpy

if False:
	states = ('Rainy', 'Sunny')
	observations = ('walk', 'shop', 'clean')
	start_probs = numpy.array([0.6, 0.4])
	transition_probs = numpy.array([[0.7, 0.3], [0.4, 0.6]])
	emission_probs = numpy.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
elif True:
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
N = len(states)
K = len(observations)

def normalize(v):
	assert abs(sum(v)) > 1.0e-100
	return v / sum(v)

# forward[t][i] = p(X_i, y_1:i | theta) 
def forward(start_probs, transition_probs, emission_probs, observations):
	state_probs = start_probs
	forward_probs = [state_probs]
	for i in range(len(observations)):
		obs_matrix = numpy.diag(emission_probs.transpose()[observations[i]])
		state_probs = state_probs.dot(transition_probs).dot(obs_matrix)
		forward_probs.append(state_probs)
	return numpy.array(forward_probs)

# backward[t][i] = p(y_t+1:T | X_t = i, theta) 
def backward(transition_probs, emission_probs, observations):
	state_probs = numpy.array([1.0 for state in states])
	backward_probs = [state_probs]
	for i in range(len(observations))[::-1]:
		obs_matrix = numpy.diag(emission_probs.transpose()[observations[i]])
		state_probs = transition_probs.dot(obs_matrix).dot(state_probs)
		backward_probs.append(state_probs)
	return numpy.array(backward_probs[::-1])

# gamma[t][i] = p(X_t = i | Y, theta)
def gamma(forward, backward):
	g = []
	for f, b in zip(forward, backward):
		p = normalize(f * b)
		g.append(p)
	return numpy.array(g)

# xi[t][i][j] = p(X_t = i, X_t+1 = j | Y, theta)
def xi(forward_probs, backward_probs, transition_probs, emission_probs, observations):
	x = numpy.zeros((len(observations), len(states), len(states)))
	y = forward_probs.dot(transition_probs[:,:])
	z = 0.0
	for t in range(len(observations)):
		z = forward_probs[t][:].dot(backward_probs[t][:])
		assert abs(z) > 1.0e-100
		for i in range(len(states)):
			for j in range(len(states)):
				x[t][i][j] = forward_probs[t][i] * transition_probs[i][j] * backward_probs[t + 1][j] * emission_probs[j][observations[t]] / z
	return x

def xi_v(forward_probs, backward_probs, transition_probs, emission_probs, observations):
	fp = forward_probs[:len(observations),:, numpy.newaxis]
	tp = transition_probs[numpy.newaxis,:,:]
	bp = backward_probs[1:len(observations) + 1,numpy.newaxis,:]
	ep = numpy.array([emission_probs.transpose()[observations[t]] for t in range(0, len(observations))])[:,numpy.newaxis,:]
	z = numpy.array([1.0/(forward_probs[t].dot(backward_probs[t])) for t in range(0, len(observations))])[:,numpy.newaxis,numpy.newaxis]
	return (fp * tp * bp * ep * z)

def viterbi(start_probs, transition_probs, emission_probs, observations):
	V = numpy.zeros((len(observations), N))
	B = [[0 for i in range(N)] for j in range(len(observations))]
	V[0] = start_probs * emission_probs.transpose()[observations[0]]
	
	for t in range(1, len(observations)):
		for i in range(N):
			k = max(range(N), key=lambda k: transition_probs[k][i] * V[t - 1][k])
			V[t][i] = emission_probs[i][observations[t]] * transition_probs[k][i] * V[t - 1][k]
			B[t][i] = k

	Xf = max(range(N), key=lambda k: V[len(observations) - 1][k])
	X = [Xf]
	for t in range(len(observations) - 1, 0, -1):
		X.append(B[t][X[-1]])
	X = X[::-1]
	return X

observations = [2, 1, 0]
viterbi(start_probs, transition_probs, emission_probs, observations)
sys.exit()
for iteration in range(1000):
	forward_probs = forward(start_probs, transition_probs, emission_probs, observations)
	backward_probs = backward(transition_probs, emission_probs, observations)
	g = gamma(forward_probs, backward_probs)
	x = xi_v(forward_probs, backward_probs, transition_probs, emission_probs, observations)
	if False:
		print forward_probs
		print
		print backward_probs
		print
		print g
		print
		print x.shape
		print x
		print
		break

	obs_prob = sum(forward_probs[len(observations)])
	print obs_prob 
	# Compute new transition probs
	for i in range(N):
		# s is the expected number of times we pass through state i 
		s = sum(g[t][i] for t in range(0, len(observations)))
		T = numpy.zeros(N)
		for j in range(N):
			t = sum(x[t][i][j] for t in range(0, len(observations)))
			T[j] = t / s if abs(s) > 1.0e-100 else transition_probs[i][j]
		T = normalize(T)
		transition_probs[i] = T

	# Compute new emission probs
	for i in range(N):
		T = numpy.zeros(K)
		# s is the expected number of times we pass through state i
		s = sum(g[t][i] for t in range(1, len(observations) + 1))
		for k in range(K):
			t = sum(g[t + 1][i] for t in range(len(observations)) if observations[t] == k)
			T[k] = t / s if abs(s) > 1.0e-100 else emission_probs[i][k]
		emission_probs[i] = T

	# Compute new start probs
	#T = normalize(numpy.linalg.inv(transition_probs).dot(g[1]))
	#start_probs = T

print start_probs
print transition_probs
print emission_probs
