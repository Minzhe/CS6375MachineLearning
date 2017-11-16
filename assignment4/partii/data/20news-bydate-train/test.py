import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
import re
# import enchant

raw = open("/Users/minzhe/Documents/Note/6375MachineLearning/assignment4/20news-bydate/20news-bydate-train/alt.atheism/49960").read()
# tokens = nltk.word_tokenize(raw)

tokens = [w.lower() for w in re.findall(r'[a-zA-Z]+', raw) if len(w) > 2 and w.lower() not in stopwords.words('english')]
WNlemma = nltk.WordNetLemmatizer()
lemma_words = [WNlemma.lemmatize(t) for t in tokens]
words_set = set(words.words())
lemma_words = [w for w in lemma_words if w in words_set]

print(lemma_words)
print()