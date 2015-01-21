import sys
import math
import numpy
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
				counts[source_word][target_word] += 1
		for word in source_sentence:
			source_marginals[word] += len(target_sentence)
		for word in target_sentence:
			target_marginals[word] += len(source_sentence)
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
	return m / m.sum(axis=1)[:, numpy.newaxis]

source_vocabulary = Vocabulary()
target_vocabulary = Vocabulary()

bitext = []
for line in sys.stdin:
	line = line.strip()
	source, target = [part.strip() for part in line.split('|||')]
	source = [source_vocabulary.convert(word) for word in source.split()]
	target = [target_vocabulary.convert(word) for word in target.split()]
	bitext.append((source, target))

L = max([len(t) for (s, t) in bitext]) # max target sentence length
translation_table = generate_initial_translation_table(bitext, len(source_vocabulary), len(target_vocabulary))
"""translation_table[source_vocabulary.convert('boy')][target_vocabulary.convert('otoko')] = 0.7
translation_table[source_vocabulary.convert('boy')][target_vocabulary.convert('no')] = 0.01
translation_table[source_vocabulary.convert('boy')][target_vocabulary.convert('ko')] = 0.2
translation_table[source_vocabulary.convert('little')][target_vocabulary.convert('otoko')] = 0.01
translation_table[source_vocabulary.convert('little')][target_vocabulary.convert('no')] = 0.05
translation_table[source_vocabulary.convert('little')][target_vocabulary.convert('ko')] = 0.65"""
start_table = numpy.array([1.0 / L for i in range(L)])
#transition_table = numpy.array([[1.0 / (transform(L) - transform(-L + 1)) for i in range(transform(-L + 1), transform(L))] for j in range(transform(-L) + 1, transform(L))])
#transition_table_offset = -transform(-L + 1)
transition_table = numpy.array([[1.0 for i in range(L)] for j in range(L)])
tension = 2.0

# each state is a tuple (a_j, f_{a_j}).
# emissions are e_j
# j \in [1, len(E)]
# a_j \in [1, len(F)], possibly add 0 for NULL
for source, target in bitext:
	print ' '.join([source_vocabulary.get_word(s) for s in source]),
	print '|||',
	print ' '.join([target_vocabulary.get_word(t) for t in target])
	N = len(source)
	K = len(target)
	start_probs = numpy.array([(1.0/tension) ** i for i in range(N)])
	start_probs = normalize(start_probs)
	print 'start_probs:'
	print start_probs

	transition_probs = numpy.zeros((N, N))
	for i in range(N):
		for j in range(N):
			transition_probs[i][j] = (1.0 / tension) ** abs(i - j)
	transition_probs = row_normalize(transition_probs)
	print 'transition_probs:'
	print transition_probs

	emission_probs = numpy.zeros((N, K))
	for i in range(N):
		for j in range(K):
			emission_probs[i][j] = translation_table[source[i]][target[j]]
	emission_probs = row_normalize(emission_probs)
	print 'emission_probs:'
	print emission_probs

	hmm = HiddenMarkovModel(N, K, start_probs, transition_probs, emission_probs)
	print hmm.viterbi(range(len(target)))

	for iteration in range(100):
		hmm.expectation_step(range(len(target)))
		hmm.maximization_step()

	print hmm.start_probs
	print hmm.transition_probs
	print hmm.emission_probs
	print hmm.viterbi(range(len(target)))
	#break
