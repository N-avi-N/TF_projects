# steps for tokenize
# 1. define a new tokenizer object
# 2. fit the corpus(cleaned) data to it
# 3. convert text to sequences(numbered)
# 4. apply padding if required

import csv
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an",
             "and", "any", "are", "as", "at", "be", "because", "been", "before", "being",
             "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing",
             "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
             "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
             "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most",
             "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our",
             "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
             "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
             "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
             "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
             "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where",
             "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you",
             "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]


sentences = []
labels = []

# read the BBC csv
# remove all the stop words
text = pd.read_csv('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv')
text.to_csv('C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\bbc_text.csv', index= False)

text = 'C:\\Users\\navin\\Desktop\\PyCharm Projects\\TF projects\\bbc_text.csv'
with open(text, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter= ',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = ' ' + word + ' '
            sentence = sentence.replace(token, ' ')
            sentence = sentence.replace('  ', ' ')
        sentences.append(sentence)

print(len(sentences))
print(sentences[0])

# Define tokenizer
# tokenize all the words in the corpus
tokenizer = Tokenizer(oov_token= '<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(len(word_index))

# convert texts to token sequences
# add padding to make all sentences same length
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding= 'post')
print(padded[0])
print(padded.shape)

# Create label tokenizer
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)

print(label_seq)
print(label_word_index)
