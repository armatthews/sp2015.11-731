#coding: utf8
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('original')
parser.add_argument('split')
args = parser.parse_args()

original_file = open(args.original)
split_file = open(args.split)

def edit_distance(s, t):
	if s == t:
		return 0
	if len(s) == 0:
		return len(t)
	if len(t) == 0:
		return len(s)

	v0 = [0 for i in range(len(t) + 1)]
	v1 = [0 for i in range(len(t) + 1)]

	for i in range(len(v0)):
		v0[i] = i

	for i in range(len(s)):
		v1[0] = i + 1
		for j in range(len(t)):
			cost = 0 if s[i] == t[j] else 1
			v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)

		for j in range(len(v0)):
			v0[j] = v1[j]

	return v1[len(t)]

def subtract_words(original, split):
	#for i in range(max(len(original), len(split))):
	#	c = original[i] if i < len(original) else 'x'
	#	d = split[i] if i < len(split) else 'x'
	#	print '%s\t%s\t%x\t%x' % (c.encode('utf-8'), d.encode('utf-8'), ord(c), ord(d))
	#print >>sys.stderr, 'subtract_word("%s", "%s")' % (original.encode('utf-8'), split.encode('utf-8'))
	if len(original) == 0:
		return None
	extras = []
	i = 0
	for j in range(len(split)):
		while True:
			if i >= len(original):
				return None
			if original[i] == split[j]:
				i += 1
				break
			extras.append(original[i])
			i +=  1
			if i >= len(original):
				return None

	return len(extras), original[i:]

def remove_junk(s):
	s = s.replace(u'ü', u'u').replace(u'ä', u'a').replace(u'ö', u'o').replace(u'ß', u'ss').replace(u'á', u'a')
        s = s.replace(u'í', u'i').replace(u'ó', u'o').replace(u'ş', u's').replace(u'ğ', u'g').replace(u'\u0307', u'')
	return s

for (i, (original_line, split_line, alignment_line)) in enumerate(zip(original_file, split_file, sys.stdin)):
	sys.stderr.write('%d\r' % i)
	original_line = original_line.decode('utf-8').lower().strip()
	split_line = split_line.decode('utf-8').lower().strip()

	original_line = remove_junk(original_line)
	split_line = remove_junk(split_line)

	original_line = original_line.split()
	split_line = split_line.split()
	alignment = []
	for link in alignment_line.strip().split():
		i, j = link.split('-')
		alignment.append((int(i), int(j)))
	print >>sys.stderr, ' '.join(original_line).encode('utf-8')
	print >>sys.stderr, ' '.join(split_line).encode('utf-8')
	print >>sys.stderr

	table = [[None for j in range(len(split_line) + 1)] for i in range(len(original_line) + 1)]
	table[0][0] = (0, '', False)

	for i in range(len(original_line)):
		for j in range(i, len(split_line)):
			candidates = []
			if table[i][j] != None:
				prev_loss, prev_remainder, _ = table[i][j]
				sub = subtract_words(original_line[i], split_line[j])
				if sub != None:
					loss, remainder = sub
					candidates.append((prev_loss + loss + len(remainder), remainder, False))
			if table[i + 1][j] != None:
				prev_loss, prev_remainder, _ = table[i + 1][j]
				sub = subtract_words(prev_remainder, split_line[j])
				if sub != None:
					loss, remainder = sub
					candidates.append((prev_loss + loss - len(prev_remainder) + len(remainder), remainder, True))
			if len(candidates) == 0:
				continue
			table[i + 1][j + 1] = min(candidates)

	#for i, line in enumerate(table):
	#	print >>sys.stderr, '%02d' % i, line
	#print >>sys.stderr

	word_map = [None for i in range(len(split_line))]
	i, j = len(original_line), len(split_line)
	while i > 0 and j > 0:
		word_map[j - 1] = i - 1
		if table[i][j] == None:
			print >>sys.stderr, 'ERROR: Table cell (%d, %d) contains NULL inappropriately!' % (i, j)
		assert table[i][j] != None
		loss, remainder, concat = table[i][j]
		if concat:
			i, j = i, j - 1
		else:
			i, j = i - 1, j - 1

	for (i, j) in alignment:
		print '%d-%d' % (word_map[i], j),
	print
	sys.stdout.flush()
