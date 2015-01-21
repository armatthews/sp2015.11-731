import sys
import math
import numpy
from hmm import HiddenMarkovModel

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
	observations = list(range(100000)) #('A', 'B')
	start_probs = numpy.array([0.85, 0.15])
	transition_probs = numpy.array([[0.3, 0.7], [0.1, 0.9]])
	emission_probs = numpy.array([[0.4, 0.6], [0.5, 0.5]])

hmm = HiddenMarkovModel(len(states), len(observations), start_probs, transition_probs, emission_probs)
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
