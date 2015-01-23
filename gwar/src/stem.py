import sys
import argparse
from nltk.stem.snowball import SnowballStemmer

parser = argparse.ArgumentParser()
parser.add_argument('language', help='Name of language to stem. E.g. \'english\' or \'german\')')
args = parser.parse_args();

stemmer = SnowballStemmer(args.language)

for line in sys.stdin:
    line = line.decode("utf-8").strip()
    words = line.split()
    words = [stemmer.stem(word.strip()) for word in words]
    line = " ".join(words)
    print line.encode("utf-8")
