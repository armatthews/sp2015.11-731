class Vocabulary:
	def __init__(self):
		self.words = []
		self.word2id = {}
		self.add('')
		assert self.lookup('') == 0

	def __str__(self):
		return str(self.words)

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
		return self.words[word_id]

	def __len__(self):
		return len(self.words)

