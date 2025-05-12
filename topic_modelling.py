
# Imports
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# Define POS tag mapping from Universal → WordNet
un2wn_mapping = {"VERB": wn.VERB, "NOUN": wn.NOUN, "ADJ": wn.ADJ, "ADV": wn.ADV}
lemmatizer = WordNetLemmatizer()

# Load the token file (e.g., disney.txt)
with open('data_preprocessed/disney_tokenized.txt', 'r', encoding='utf-8') as f:
    words = f.read().split()  # a long list of tokens

# Get POS tags
tagged = nltk.pos_tag(words, tagset="universal")

# Lemmatize with POS awareness
lemmatized_doc = []
for word, tag in tagged:
    if tag in un2wn_mapping:
        lemma = lemmatizer.lemmatize(word, pos=un2wn_mapping[tag])
    else:
        lemma = lemmatizer.lemmatize(word)
    lemmatized_doc.append(lemma.lower())


''' Construct the document-term matrix '''
# Imports
from gensim import corpora
import itertools

# Creation of dictionary
movie_dictionary = corpora.Dictionary(movie)
print('Number of unique tokens:', len(movie_dictionary))

# Filtering out rare and common words based on their frequency in the document 


from collections import Counter

# !!! I have randomly chosen 5 and 500 for now --> needs to be changed 
word_counts = Counter(lemmatized_doc)
filtered_doc = [word for word in lemmatized_doc if 5 <= word_counts[word] <= 500]

# New dictionary and corpus
movie_docs = [filtered_doc]
movie_dictionary = corpora.Dictionary(movie_docs)

# The document is converted to Bag-of-Words
movie_bow_corpus = [movie_dictionary.doc2bow(doc) for doc in movie_docs]
print(movie_bow_corpus[0][:50])


# Applying the LDA model
movie_ldamodel = models.ldamodel.LdaModel(movie_bow_corpus, num_topics=5, id2word = movie_dictionary, passes = 20)
movie_ldamodel.show_topics(formatted=False, num_words=5)
'''
# IS THIS SECTION RELEVANT?
Group into pseudo-documents (chunks of 100 tokens)
doc_size = 100
disney_docs = [lemmatized_doc[i:i+doc_size] for i in range(0, len(lemmatized_doc), doc_size)]

print(f"Created {len(disney_docs)} documents from disney.txt")

from gensim import corpora, models
from pprint import pprint
import os

# Create dictionary and corpus
dictionary = corpora.Dictionary(disney_docs)
corpus = [dictionary.doc2bow(doc) for doc in disney_docs]

# Train the LDA model
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,       # You can adjust this
    passes=10,
    random_state=42
)

# Print the top topics
print("\n🧠 Top 5 Topics (10 words each):")
pprint(lda_model.print_topics(num_words=10))

# Optional: Save model and dictionary
os.makedirs('models', exist_ok=True)
lda_model.save('models/disney_lda.model')
dictionary.save('models/disney_dictionary.dict')

print("\nLDA model and dictionary saved to /models")
'''
