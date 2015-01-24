import sys
import math
import numpy
import random
import argparse
import itertools
from collections import defaultdict
from vocab import Vocabulary
from hmm import HiddenMarkovModel

def generate_initial_translation_table(bitext, source_vocab_size, target_vocab_size):
	counts = numpy.zeros((source_vocab_size, target_vocab_size))
	source_marginals = numpy.zeros(len(source_vocabulary))
	target_marginals = numpy.zeros(len(target_vocabulary))

	for source_sentence, target_sentence in bitext:
		for target_word in target_sentence:
			counts[0][target_word] += 1
		source_marginals[0] += len(target_sentence)

		for source_word in source_sentence:
			for target_word in target_sentence:
				assert source_word != 0 and target_word != 0
				counts[source_word][target_word] += 1
		for word in source_sentence:
			source_marginals[word] += len(target_sentence)
		#for word in target_sentence:
		#	target_marginals[word] += len(source_sentence) + 1
	return counts / source_marginals[:, numpy.newaxis]

def transform(jump_size):
	if jump_size == 0:
		return 0
	elif jump_size < 0:
		return -transform(-jump_size)
	return int(math.floor(math.log(jump_size) * math.sqrt(3.5))) + 1

def normalize(v):
	return v / sum(v)

def row_normalize(m):
	sums = m.sum(axis=1)[:, numpy.newaxis]
	return numpy.where(sums > 0, m / sums, 0.0)

def print_ttable():
	for word_id in range(len(target_vocabulary)):
		sys.stdout.write('\t' + target_vocabulary.get_word(word_id))
	sys.stdout.write('\n')
	for word_id in range(len(source_vocabulary)):
		sys.stdout.write(str(source_vocabulary.get_word(word_id)) + '\t')
		for v in translation_table[word_id]:
			sys.stdout.write('%.2f\t' % v)
		sys.stdout.write('\n')

source_vocabulary = Vocabulary()
target_vocabulary = Vocabulary()

bitext = []
for line in sys.stdin:
	line = line.strip()
	source, target = [part.strip() for part in line.split('|||')]
	source = [source_vocabulary.convert(word) for word in source.split()]
	target = [target_vocabulary.convert(word) for word in target.split()]
	bitext.append((source, target))

L = max([len(s) for (s, t) in bitext]) # max source target length
translation_table = generate_initial_translation_table(bitext, len(source_vocabulary), len(target_vocabulary))
#print 'translation_table:'
#print_ttable()
tension = 1.1
N = L
K = len(target_vocabulary)

start_probs = numpy.array([(1.0/tension) ** i for i in range(N)])
start_probs = normalize(start_probs)
#print 'start_probs:'
#print start_probs

transition_probs = numpy.zeros((N, N))
for i in range(N):
	for j in range(N):
		transition_probs[i][j] = (1.0 / tension) ** abs(i - j + 1)
transition_probs = row_normalize(transition_probs)
#print 'transition_probs:'
#print transition_probs

epsilon_row = numpy.zeros(K)
epsilon_row[0] = 1.0

use_null = True
null_prob = 0.01

ttable_expectations = numpy.zeros(translation_table.shape)
jump_expectations = numpy.zeros(transition_probs.shape)
state_count = 2 * N + 1 if use_null else N
hmm = HiddenMarkovModel(state_count, K, start_probs, transition_probs, None)
for iteration in range(8):
	total_log_prob = 0.0
	shuffled_bitext = bitext[:]
	random.shuffle(shuffled_bitext)
	for source, target in shuffled_bitext:
		emission_probs = numpy.zeros((N, K))
		for i in range(N):
			if i < len(source):
				emission_probs[i] = translation_table[source[i]]
			else:
				emission_probs[i] = epsilon_row

		if use_null:
			hmm.start_probs = numpy.hstack(((1.0 - null_prob) * start_probs, null_prob, numpy.zeros(N)))
			hmm.emission_probs = numpy.vstack((emission_probs, translation_table[0, numpy.newaxis] * numpy.ones((N + 1, 1))))
			hmm.transition_probs = numpy.vstack((transition_probs, start_probs, transition_probs))
			hmm.transition_probs = numpy.hstack(((1.0 - null_prob) * hmm.transition_probs, numpy.zeros((2 * N + 1 , N + 1))))
			hmm.transition_probs[N][N] = null_prob
			for i in range(len(source)):
				hmm.transition_probs[i][i + N + 1] = null_prob
				hmm.transition_probs[i + N + 1][i + N + 1] = null_prob
		else:
			hmm.emission_probs = emission_probs
		assert hmm.start_probs.shape == (state_count,)
		assert hmm.transition_probs.shape == (state_count, state_count)
		assert hmm.emission_probs.shape == (state_count, K)
		#hmm.N = 2 * len(source) + 1
	
		obs_prob = hmm.expectation_step(target)
		total_log_prob += math.log(obs_prob)

		for i in range(len(source)):
			ttable_expectations[source[i]] += hmm.stats.emission_counts[i]
		if use_null:
			for i in range(N, 2 * N + 1):
				ttable_expectations[0] += hmm.stats.emission_counts[i]
		hmm.stats.emission_counts = numpy.zeros(hmm.stats.emission_counts.shape)

		if use_null:
			for i in range(len(source)):
				jump_expectations[i] += hmm.stats.transition_counts[i][:N] + hmm.stats.transition_counts[i + N + 1, : N]
			hmm.stats.transition_counts = numpy.zeros(hmm.stats.transition_counts.shape)

	null_prob = sum(hmm.stats.state_counts[N:]) / sum(hmm.stats.state_counts)
	start_counts = hmm.stats.start_counts[:]
	hmm.maximization_step()
	#ttable_expectations[0][0] = 1.0
	translation_table = row_normalize(ttable_expectations)
	ttable_expectations = numpy.zeros(translation_table.shape)
	if use_null:
		start_probs = start_counts[:N] / sum(start_counts[:N])
		for d in range(-(N-1), N):
			total = sum(jump_expectations[i][i + d] for i in range(jump_expectations.shape[0]) if i + d >= 0 and i + d < N)
			for i in range(jump_expectations.shape[0]):
				if i + d >= 0 and i + d < N:
					jump_expectations[i][i + d] = total
		transition_probs = row_normalize(jump_expectations)
		#print transition_probs
		jump_expectations = numpy.zeros(transition_probs.shape)
		print >>sys.stderr, 'iteration %d null_prob: %f' % (iteration, null_prob)
	print >>sys.stderr, 'iteration %d log_prob: %f' % (iteration, total_log_prob)

for source, target in bitext:
	emission_probs = numpy.zeros((N, K))
	for i in range(N):
		if i < len(source):
			emission_probs[i] = translation_table[source[i]]
		else:
			emission_probs[i] = epsilon_row
	#hmm.N = len(source)
	if use_null:
		hmm.start_probs = numpy.hstack((start_probs, null_prob, numpy.zeros(N)))
		hmm.emission_probs = numpy.vstack((emission_probs, translation_table[0, numpy.newaxis] * numpy.ones((N + 1, 1))))
		hmm.transition_probs = numpy.vstack((transition_probs, start_probs, transition_probs))
		hmm.transition_probs = numpy.hstack(((1.0 - null_prob) * hmm.transition_probs, numpy.zeros((2 * N + 1 , N + 1))))
		hmm.transition_probs[N][N] = null_prob
		for i in range(len(source)):
			hmm.transition_probs[i][i + N + 1] = null_prob
			hmm.transition_probs[i + N + 1][i + N + 1] = null_prob
	else:
		hmm.emission_probs = emission_probs
	assert hmm.start_probs.shape == (state_count,)
	assert hmm.transition_probs.shape == (state_count, state_count)
	assert hmm.emission_probs.shape == (state_count, K)	
	a = hmm.viterbi(target)
	print ' '.join(('%d-%d' % (j, i) for (i, j) in enumerate(a) if j < len(source)))

#print 'start_probs:'
#print hmm.start_probs
#print 'transition_probs:'
#print hmm.transition_probs
#print_ttable()
