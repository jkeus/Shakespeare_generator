import random
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


#get text
filepath = tf.keras.utils.get_file("shakespeare.txt", origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(filepath,'rb').read().decode(encoding='utf-8').lower()

#range for training
text = text[300000:800000]

#get unique list of sorted characters found
characters = sorted(set(text))

#dictionary to convert characters to numbers key:char val:num
char_to_index = dict((c,i) for i, c in enumerate(characters))

#dict key:index/num val:char
index_to_char = dict((i,c) for i, c in enumerate(characters))

#plan scan and predict a character
SEQ_LENGTH = 40
STEP_SIZE = 3


sentences = [] #arry sentences -1
next_character = [] # corresponding array with the correct letter at the end of a sentence

#string and feature data collection
for i in range(0,len(text) - SEQ_LENGTH, STEP_SIZE): #begining to end(valid sequence length)
    sentences.append(text[i: i+SEQ_LENGTH])
    next_character.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(char_to_index)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

#fill arrays
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_character[i]]] = 1

#the model itself
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.01))

model.fit(x, y, batch_size=256, epochs=4)

model.save('shakespeare_model.keras')



model = tf.keras.models.load_model('shakespeare_model.keras')
#helper method
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print('------------0.2-------------')
print(generate_text(300, 0.2))
print('------------0.4-------------')
print(generate_text(300, 0.4))
print('------------0.6-------------')
print(generate_text(300, 0.6))
print('------------0.8-------------')
print(generate_text(300, 0.8))
print('------------1-------------')
print(generate_text(300, 1))
