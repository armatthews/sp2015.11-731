import sys
import math
import argparse
from collections import defaultdict

class vocabulary:
	def __init__(self):
		self.words = []
		self.word2id = {}
		self.add('')
		assert self.lookup('') == 0

	def lookup(self, word):
		return self.word2id[word]

	def convert(self, word):
		if word not in self.word2id:
			self.add(word)
		return self.lookup(word)

	def add(self, word):
		assert (word not in self.word2id)
		self.word2id[word] = len(self.words)
		self.words.append(word)

	def get_word(self, word_id):
		assert (word_id < len(self.words))
		return self.words(word_id)

	def __len__(self):
		return len(self.words)

def transform(jump_size):
	if jump_size == 0:
		return 0
	elif jump_size < 0:
		return -transform(-jump_size)
	return int(math.floor(math.log(jump_size) * math.sqrt(3.5))) + 1

source_vocabulary = vocabulary()
target_vocabulary = vocabulary()

bitext = []
for line in sys.stdin:
	line = line.decode('utf-8').strip()
	source, target = [part.strip() for part in line.split('|||')]
	source = [source_vocabulary.convert(word) for word in source.split()]
	target = [target_vocabulary.convert(word) for word in target.split()]
	bitext.append((source, target))

translation_table = [defaultdict(float) for i in range(len(source_vocabulary))]
L = max([len(t) for (s, t) in bitext])
transition_table = [[1.0 / (transform(L) - transform(-L)) for i in range(transform(-L), transform(L))] for j in range(transform(-L), transform(L))]

counts = defaultdict(lambda: defaultdict(int))
source_marginals = defaultdict(int)
target_marginals = defaultdict(int)
for source_sentence, target_sentence in bitext:
	for source_word in source_sentence:
		for target_word in target_sentence:
			counts[source_word][target_word] += 1
	for word in source_sentence:
		source_marginals[word] += len(target_sentence)
	for word in target_sentence:
		target_marginals[word] += len(source_sentence)

for source_word in counts.keys():
	marginal = source_marginals[source_word]
	for target_word, count in counts[source_word].iteritems():
		translation_table[source_word][target_word] = 1.0 * count / marginal
