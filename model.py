from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
import pandas as pd

dataset = pd.read_csv("tv.csv")
print(dataset.head())

x = dataset["main"].values
y = dataset["xx"].values

xx = np.array(x)
yy = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(xx,yy,test_size = 0.15,random_state = 1)

def dataLoader(X,Y, batch_size):

    L = len(X)
 
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            XX = X[batch_start:limit]
            YY = Y[batch_start:limit]

            yield (XX,YY)     

            batch_start += batch_size   
            batch_end += batch_size


model = Sequential()
model.add(Dense(200, input_dim=1, activation='relu'))
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

#model.fit_generator(dataLoader(xx,yy,5), epochs = 10)
history = model.fit(X_train, y_train, epochs=10, batch_size=5,validation_split=0.15,validation_data=None,verbose=1)

print(model.predict(X_test,batch_size=5))
print("---------------------------------------")
print(y_test)


# plot metrics
# pyplot.plot(history.history['mean_squared_error'])
# pyplot.plot(history.history['mean_absolute_error'])
pyplot.plot(history.history['mean_absolute_percentage_error'])
# pyplot.plot(history.history['cosine_proximity'])
pyplot.show()

score = model.evaluate(X_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])