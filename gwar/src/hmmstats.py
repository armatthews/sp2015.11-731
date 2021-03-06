import sys
import numpy
from sklearn.preprocessing import normalize

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
		self.start_counts[:start_counts.shape[0]] += start_counts
		self.state_counts[:state_counts.shape[0]] += state_counts
		self.transition_counts[:transition_counts.shape[0], :transition_counts.shape[1]] += transition_counts
		self.emission_counts[:emission_counts.shape[0], :emission_counts.shape[1]] += emission_counts

	def compute_start_probs(self):
		return self.start_counts / numpy.sum(self.start_counts)

	def compute_transition_probs(self):
		transition_probs = self.transition_counts / self.state_counts[:, numpy.newaxis]
		#return normalize(transition_probs, axis=1, norm='l1')
		sums = transition_probs.sum(axis=1)[:, numpy.newaxis]
		#a = transition_probs / sums
		b = numpy.where(sums > 0, transition_probs / sums, 0.0)
		return b

	def compute_emission_probs(self):
		emission_probs = self.emission_counts / self.state_counts[:, numpy.newaxis]
		#return normalize(emission_probs, axis=1, norm='l1')
		sums = emission_probs.sum(axis=1)[:, numpy.newaxis]
		#a = emission_probs / sums
		b = numpy.where(sums > 0, emission_probs / sums, 0.0)
		return b

