__author__ = 'Fabian Schilling'
__email__ = 'fabsch@kth.se'

import numpy as np
import re
import random
import sys
import cPickle as pickle

from nltk.corpus import reuters, stopwords

ACQ = 'acq'
CORN = 'corn'
CRUDE = 'crude'
EARN = 'earn'

TRAIN = 'train'
TEST = 'test'

CATEGORIES = [ACQ, CORN, CRUDE, EARN]
SPLITS = [TRAIN, TEST]

SIZES = {
  TRAIN: {
    ACQ: 114,
    CORN: 38,
    CRUDE: 76,
    EARN: 152
  },
  TEST: {
    ACQ: 25,
    CORN: 10,
    CRUDE: 15,
    EARN: 40
  }
}

def clean(raw_doc):
  # Make all lowercase
  doc = raw_doc.lower()
  # Keep only words containing letters
  doc = re.sub('[^a-z]', ' ', doc)
  # Get rid of multiple adjacent spaces
  doc = re.sub(' +', ' ', doc)
  # Split on whitespace
  word_list = doc.split(' ')
  # Get rid of all english stopwords
  word_list = [word for word in word_list if word not in stopwords.words('english')]
  # Join with space
  clean_doc = ' '.join(word_list)
  # Strip of whitespace
  return clean_doc.strip()


def load_dataset(shuffle=True):
  if shuffle:
    print('Shuffling dataset')
  dataset = {}
  for split in SPLITS:
    dataset[split] = []
    for category in CATEGORIES:
      print('Loading "{}" split "{}" category ({} examples)'.format(split, category, SIZES[split][category]))
      # File names are preceded by 'train' or 'test'
      raw_docs = [reuters.raw(fname) for fname in reuters.fileids(category) if split in fname]
      if shuffle:
        random.shuffle(raw_docs)
      clean_docs = []
      counter = 0
      for raw_doc in raw_docs:
        clean_doc = clean(raw_doc)
        # Get rid of documents that only contain the date!
        if len(clean_doc) > 15:
          dataset[split].append((clean_doc, category))
          counter += 1
        if counter >= SIZES[split][category]:
          break
  return dataset


def main():
  if len(sys.argv) < 2:
    print('Usage: python {} <filename>'.format(sys.argv[0]))
    exit()

  filename = sys.argv[-1] + '.pkl'

  dataset = load_dataset()
  pickle.dump(dataset, open(filename, 'wb'))
  print('Dumped dataset as {}'.format(filename))


if __name__ == '__main__':
  main()
