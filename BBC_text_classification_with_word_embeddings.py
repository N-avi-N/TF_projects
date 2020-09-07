import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# define embedding parameters
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8

sentences = []
labels = []
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
print(len(stopwords))

# Read the BBC text
# remove the headers
# split labels and text
# remove the stop words
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

# split data for test and train for training the embeddings
# check whether the data splits are correct
train_size = int(len(sentences) * training_portion)

training_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print('training proportion is: {}'.format(train_size))
print(len(training_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

# Define tokenizer for train data
# train tokenizer using training data
# convert text to sequences and pad the data
# check some of the data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(training_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

print(len(train_sequences[0]))
print(len(train_padded[0]))

# use train tokenizer on test data as well, all unseen words will convert to OOv
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding= padding_type, maxlen= max_length)

print(len(validation_padded))
print(validation_padded.shape)

# define tokenizer for labels, here for labels we use all the corpus to get the tokens
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq.shape)

# define classification model with embeddings
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['acc'])
model.summary()

num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq), verbose= 2)

# Plot training vs validation accuracy and loss for the model
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel("Epochs")
plt.ylabel('accuracy')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])
plt.show()


# function to convert sequences back to texts
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# check the embedding layer
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)



# Additional code
# import io
#
# out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
# for word_num in range(1, vocab_size):
#   word = reverse_word_index[word_num]
#   embeddings = weights[word_num]
#   out_m.write(word + "\n")
#   out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
# out_v.close()
# out_m.close()
#
# try:
#   from google.colab import files
# except ImportError:
#   pass
# else:
#   files.download('vecs.tsv')
#   files.download('meta.tsv')