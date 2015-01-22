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
	counts[0][0] = 1.0
	source_marginals[0] = 1.0
	target_marginals[0] = 1.0
	for source_sentence, target_sentence in bitext:
		for source_word in source_sentence:
			for target_word in target_sentence:
				assert source_word != 0 and target_word != 0
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

L = max([len(t) for (s, t) in bitext]) # max target sentence length
translation_table = generate_initial_translation_table(bitext, len(source_vocabulary), len(target_vocabulary))
print 'translation_table:'
print_ttable()
tension = 1.1
N = L
K = len(target_vocabulary)

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

epsilon_row = numpy.zeros(K)
epsilon_row[0] = 1.0


ttable_expectations = numpy.zeros(translation_table.shape)
hmm = HiddenMarkovModel(N, K, start_probs, transition_probs, None)
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

		hmm.emission_probs = emission_probs
	
		obs_prob = hmm.expectation_step(target)
		print ' '.join([source_vocabulary.get_word(s) for s in source]),
		print '|||',
		print ' '.join([target_vocabulary.get_word(t) for t in target]),
		print hmm.viterbi(target)

		total_log_prob += math.log(obs_prob)
		for i in range(len(source)):
			ttable_expectations[source[i]] += hmm.stats.emission_counts[i]
		hmm.stats.emission_counts = numpy.zeros(hmm.stats.emission_counts.shape)

	hmm.maximization_step()
	ttable_expectations[0][0] = 1.0
	translation_table = row_normalize(ttable_expectations)
	ttable_expectations = numpy.zeros(translation_table.shape)
	print 'iteration %d log_prob: %f' % (iteration, total_log_prob)

print 'start_probs:'
print hmm.start_probs
print 'transition_probs:'
print hmm.transition_probs
print_ttable()
