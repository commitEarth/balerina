

def main():
    
print(""""  from keras.datasets import imdb
(train_data, train_labels),(test_data,test_labels) = imdb.load_data(num_words= 10000)

from tensorflow.keras import models, layers
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Build the model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=4, batch_size=512, validation_split=0.2)

# Evaluate on test data
results = model.evaluate(x_test, y_test)
print("Test Accuracy:", results[1])

""")

main()