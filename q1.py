import nltk 
import re
from nltk.util import pr 
import numpy as np 
import heapq
import sys
import sklearn.metrics as sklearn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import spacy
import random
from nltk.data import find
from bllipparser import RerankingParser
from nltk import pos_tag, word_tokenize, RegexpParser
from collections import Counter
from math import sqrt

with open('TextMining.txt') as f:
  contentParagraphs = [line.strip().split('\n') for line in f]

# print("before : ", contentParagraphs[0])
contentSentences = []
for j in range(len(contentParagraphs)):
  paragraph = contentParagraphs[j][0]
  sentences = nltk.sent_tokenize(paragraph)
  for i in range(len(sentences)): 
    sentences[i] = sentences[i].lower() 
    sentences[i] = re.sub(r'\W', ' ', sentences[i]) 
    sentences[i] = re.sub(r'\s+', ' ', sentences[i]) 
  contentSentences.extend(sentences)

# print("\nafter tokenizing to sentences: ", contentSentences[0:5])

stopWord = ['the', 'and', 'a', 'of', 'is', 'it', 'i', 'this', 'to', 'in', 'was', 'that', 's', 'for', 't', 'with', 'you', 
'one', 'are', 'so', 'there', 'at', 'an', 'be', 'have', 'by', 'from', 'his', 'he', 'or', 'who', 
'were', 'can', 'has', 'my', 'they','some', 'your', 'also','its', 'me', 'her', 'any', 'had', 'them',
'their', 've', 'those', 'been', 'such', 'being', 'she']

# remove stopwords and tokenizing words from sentences.
contentWords = []
for sentence in contentSentences: 
  words = nltk.word_tokenize(sentence) 
  for word in words: 
    if word in stopWord:
      words.remove(word)
  contentWords.extend(words)

# print("\nafter tokenizing to words: ",contentWords[:20])

ps = PorterStemmer()
steming = []
for w in contentWords[:10]:
  steming.append((w, ps.stem(w)))

# print("\nafter stemming: ",steming[0:20])

posTagged = nltk.pos_tag(contentWords)

# print("\nafter pos tagging: ",posTagged[:20])
 
ner = []
nlp = spacy.load('en_core_web_sm')
for sent in contentSentences:
  doc = nlp(sent)
  for ent in doc.ents:
    ner.append((ent.text, ent.label_))

# print("\nafter NER: ",ner[:20])

model_dir = find('models/bllip_wsj_no_aux').path
parser = RerankingParser.from_unified_model_dir(model_dir)
best = parser.parse(contentSentences[0])
# print(contentSentences[0],'\n')
# print(best.get_parser_best(),'\n')

best = parser.parse(contentSentences[100])
# print(contentSentences[100],'\n')
# print(best.get_parser_best(),'\n')

verbs = []
for (word,tag) in posTagged:
  if tag[:2] == 'VB':
    verbs.append(word)


verbsSynoyms = []
for word in verbs:
  syn = []
  for synset in wordnet.synsets(word):
    for lemma in synset.lemmas():
      syn.append(lemma.name())
  verbsSynoyms.append((word, syn))

# print("\nafter verb Synoyms: ",verbsSynoyms[:10])

# for verb in verbs[:5]:
#   for ss in wordnet.synsets(verb)[:1]:
#     for hyper in ss.hypernyms()[0:1]:
#       print ('\nhypernym of', ss.name()[:-5], 'is', hyper.name()[:-5])
#     for hypo in ss.hyponyms()[0:1]:
#       print ('\nhyponym of', ss.name()[:-5], 'is', hypo.name()[:-5])

i = 0
while i < 10:
  w1 = random.choice(verbs)
  if wordnet.synsets(w1) == []:
    verbs.remove(w1)
    continue
  w2 = random.choice(verbs)
  if wordnet.synsets(w2) == []:
    verbs.remove(w2)
    continue
  syn1 = wordnet.synsets(w1)[0]
  syn2 = wordnet.synsets(w2)[0]
  similarity = syn1.path_similarity(syn2)
  if similarity == None:
    continue
  i += 1
  print ("semantic distance", syn1.name()[:-5], "and", syn2.name()[:-5], round(1-similarity,2))

i = 0
while i < 10:
  w1 = random.choice(contentWords)
  if wordnet.synsets(w1) == []:
    contentWords.remove(w1)
    continue
  w2 = random.choice(contentWords)
  if wordnet.synsets(w2) == []:
    contentWords.remove(w2)
    continue
  syn1 = wordnet.synsets(w1)[0]
  syn2 = wordnet.synsets(w2)[0]
  similarity = syn1.wup_similarity(syn2)
  if similarity == None:
    continue
  i += 1
  print ("similarity", syn1.name()[:-5], "and", syn2.name()[:-5], similarity)

def word2vec(word):
    cw = Counter(word)
    sw = set(cw)
    lw = sqrt(sum(c*c for c in cw.values()))
    return cw, sw, lw

def cosdis(a, b):
    v1 = word2vec(a)
    v2 = word2vec(b)
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]  *v2[0][ch] for ch in common)/v1[2]/v2[2]

# selected = random.sample(contentWords, 10)
# for i in range(0,10,2):
#   print ("similarity", selected[i], "and", selected[i+1], cosdis(selected[i], selected[i+1]))








