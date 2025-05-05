
# balerina/File.py
code2 = '''
from keras.datasets import imdb
from tensorflow.keras import models, layers
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512, validation_split=0.2)

results = model.evaluate(x_test, y_test)
print("Test Accuracy:", results[1])
'''

code1='''
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data = fetch_california_housing()
x=data.data
y=data.target

scaler=StandardScaler().fit(x)
x=scaler.transform(x)

trainx ,testx,trainy,testy=train_test_split(x,y,test_size=.2)


from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(64,input_shape=(8,),activation='relu' ))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error')
model.fit(trainx,trainy,epochs=20,batch_size=64)
res=model.evaluate(testx,testy)
res


ypred=model.predict(testx)

import matplotlib.pyplot as plt
plt.scatter(testy,ypred, alpha=.5)
plt.plot([testy.min(), testy.max()], [testy.min(), testy.max()], color='red')

 '''